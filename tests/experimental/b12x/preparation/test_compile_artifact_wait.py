"""Preparation waits observe artifacts published before compiler completion."""

from threading import Event
from types import SimpleNamespace

import pytest

from b12x.preparation.session import PreparationJob


@pytest.mark.parametrize("method", ["wait", "trial"])
@pytest.mark.parametrize("publish", [False, True])
def test_completed_compiler_rechecks_artifact_publication(method, publish):
    published = Event()
    observations = []

    class Pool:
        pending = False

        def ready(self, programs):
            if publish:
                published.set()
            return True

    def available(programs):
        observations.append(published.is_set())
        return published.is_set()

    job = PreparationJob.__new__(PreparationJob)
    job.session = SimpleNamespace(_pool=Pool(), _stop=Event(), cache_only=False)
    job.autotune = True
    job._active_request = None
    job._programs_available = available
    job._pump_pipeline = lambda _: iter(())
    programs = frozenset({"program"})
    if method == "wait":
        steps = job._wait_programs(programs)
    else:
        obligation = SimpleNamespace(request=object(), programs={None: programs})
        job._payload = lambda *_: None
        job._compile = lambda *_, **__: iter(())
        job._selection = lambda *_: None
        guard = SimpleNamespace(finish=lambda: None)
        job._call = lambda *_, **__: (None, None, guard, None)
        job._temporaries = [guard.finish]
        job._candidates_prepared = 0
        obligation.request = SimpleNamespace(benchmark_call=None)
        steps = job._trial(obligation, 0, None, None)
    if publish:
        assert list(steps) == (["compile"] if method == "wait" else ["compile", "gpu"])
        assert observations[0] is False
        assert observations[-1] is True
    else:
        with pytest.raises(RuntimeError, match="completed without publishing"):
            list(steps)
