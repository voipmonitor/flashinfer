"""Resolve retained joint kernels and enforce cooperative resident-grid limits."""


def joint_function(launcher):
    """Return the CUfunction retained by a single-kernel CuTe launcher."""
    from cuda.bindings import driver

    dependencies = launcher.__b12x_dependencies__
    if len(dependencies) != 1:
        raise RuntimeError("joint launcher must retain exactly one compiled program")
    executable = dependencies[0]._executable
    names = tuple(executable.kernel_info)
    if len(names) != 1:
        raise RuntimeError("joint launcher must contain exactly one CUDA kernel")
    result, kernel = driver.cuLibraryGetKernel(
        driver.CUlibrary(int(executable.library)), names[0].encode()
    )
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cannot resolve retained joint kernel: {result}")
    result, function = driver.cuKernelGetFunction(kernel)
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cannot resolve joint CUDA function: {result}")
    return int(function)


def require_resident_grid(launcher, ordinal, threads, blocks):
    """Reject a cooperative grid that cannot be resident without resizing it."""
    import torch
    from cuda.bindings import driver

    with torch.cuda.device(ordinal):
        result, resident = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            driver.CUfunction(joint_function(launcher)), threads, 0
        )
        if result != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cannot inspect cooperative grid residency: {result}")
        capacity = (
            resident * torch.cuda.get_device_properties(ordinal).multi_processor_count
        )
    if blocks > capacity:
        raise ValueError(
            f"cooperative joint grid requires {blocks} resident CTAs; capacity is {capacity}"
        )
