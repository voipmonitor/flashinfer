"""Lossless dense block-quantized storage prepared from safetensors block payloads."""

from dataclasses import dataclass

import torch

from b12x._lib.quant.block_codec import block_codec


@dataclass(frozen=True)
class BlockQuantLinearWeight:
    """Encoded tiles and codec-specific metadata, in bytes.

    XS metadata is [ceil(N/128),K/256,1280]; XXS bases are [K/256,N,2].
    Q8_0 uses [N/B,K/32,B,32] payload tiles and [K/32,N,2] bases.

    The [N,K/4] descriptor byte view stores [N/B,K/256,B,64] tiles,
    where B is 128 for N divisible by 128, and 8 otherwise.
    Each XS metadata tile holds 128 FP16 block bases followed by eight planes
    of 128 packed scale pairs. Only the final XS N metadata tile is padded.
    Tensors are owned by this object and must remain unmodified during use.
    """

    values: torch.Tensor
    metadata: torch.Tensor
    in_features: int
    out_features: int
    codec: str = "iq2_xs"

    @property
    def padded_in_features(self) -> int:
        return self.in_features


IQ2XSLinearWeight = BlockQuantLinearWeight


def pack_iq2_xs_weight(
    blocks: torch.Tensor, *, codec: str = "iq2_xs", allocate=None
) -> BlockQuantLinearWeight:
    """Retile raw blocks without expanding their packed numeric encoding."""
    from ._a16 import _check_tensor

    spec = block_codec(codec)
    if blocks.device.type != "cuda":
        raise ValueError("IQ2_XS weights must be on CUDA")
    _check_tensor("IQ2_XS blocks", blocks, blocks.device, torch.uint8)
    if (
        blocks.ndim != 3
        or blocks.shape[-1] != spec.block_bytes
        or min(blocks.shape) <= 0
    ):
        raise ValueError(
            f"{codec.upper()} blocks must have positive shape [N,K/{spec.block_weights},{spec.block_bytes}]"
        )
    n, kb, _ = blocks.shape
    if n % 8:
        raise ValueError("IQ2_XS dense weights require N divisible by 8")
    with torch.cuda.device(blocks.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("IQ2_XS weight packing must precede CUDA graph capture")
    bases = blocks[..., :2].contiguous()
    if not torch.isfinite(bases.view(torch.float16)).all().item():
        raise ValueError("IQ2_XS block bases must be finite")
    return _retile_dense_blocks(blocks, bases, spec, codec=codec, allocate=allocate)


def _retile_dense_blocks(blocks, bases, spec, *, codec, allocate=None):
    n, kb, _ = blocks.shape
    if not spec.subscale_bytes:
        # Compact block-major bases: no N-tail padding in resident storage.
        if allocate is None:
            metadata = bases.permute(1, 0, 2).contiguous()
        else:
            metadata = allocate((kb, n, 2), dtype=torch.uint8, device=blocks.device)
            metadata.copy_(bases.permute(1, 0, 2))
    else:
        nt = (n + 127) // 128
        metadata = (
            torch.zeros((nt, kb, 1280), dtype=torch.uint8, device=blocks.device)
            if allocate is None
            else allocate(
                (nt, kb, 1280), dtype=torch.uint8, device=blocks.device
            ).zero_()
        )
        # Row padding is confined to metadata, preserving the descriptor matrix.
        base_rows = torch.zeros(
            (nt * 128, kb, 2), dtype=torch.uint8, device=blocks.device
        )
        scale_rows = torch.zeros(
            (nt * 128, kb, 8), dtype=torch.uint8, device=blocks.device
        )
        base_rows[:n].copy_(bases)
        scale_rows[:n].copy_(blocks[..., 66:])
        metadata[..., :256].copy_(
            base_rows.view(nt, 128, kb, 2).permute(0, 2, 1, 3).reshape(nt, kb, 256)
        )
        metadata[..., 256:].copy_(
            scale_rows.view(nt, 128, kb, 8).permute(0, 2, 3, 1).reshape(nt, kb, 1024)
        )
    tile_n = 128 if n % 128 == 0 else 8
    tiles = (
        blocks[..., 2 : 2 + spec.payload_bytes]
        .reshape(n // tile_n, tile_n, kb, spec.payload_bytes)
        .permute(0, 2, 1, 3)
    )
    if allocate is None:
        values = tiles.contiguous().view(n, kb * spec.payload_bytes)
    else:
        values = allocate(
            (n, kb * spec.payload_bytes), dtype=torch.uint8, device=blocks.device
        )
        values.view(n // tile_n, kb, tile_n, spec.payload_bytes).copy_(tiles)
    return BlockQuantLinearWeight(values, metadata, kb * spec.block_weights, n, codec)
