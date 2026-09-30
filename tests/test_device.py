"""GPU-first execution: micro-batching on OOM, fallback to CPU, return to GPU, wait mode.

The test machine has no GPU, so the 'GPU' is simulated by the device label cpu:0 (preferred) versus
cpu (fallback), and out-of-memory errors are raised on purpose by the model.
"""
import copy

import pytest
import torch
import torch.nn.functional as F

from earbench.device import DeviceManager, is_oom

GPU, CPU = torch.device("cpu", 0), torch.device("cpu")


class Flaky(torch.nn.Module):
    """Linear model that raises CUDA OOM on the 'GPU' for batches larger than `limit` (None = never)."""

    def __init__(self, dm_ref, limit=None):
        super().__init__()
        self.lin = torch.nn.Linear(6, 3)
        self.dm_ref, self.limit, self.calls = dm_ref, limit, []

    def forward(self, x):
        on_gpu = self.dm_ref[0].current == GPU
        self.calls.append((len(x), on_gpu))
        if on_gpu and self.limit is not None and len(x) > self.limit:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return self.lin(x)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def cfg(**kw):
    base = {"mode": "cpu", "min_micro_batch": 4, "min_free_mb_to_return": 1000, "check_every_s": 60,
            "max_wait_minutes": 1}
    base.update(kw)
    return base


def make(limit, free=lambda: 0.0, **kw):
    ref = [None]
    clock = Clock()
    dm = DeviceManager(GPU, cfg(**kw), fallback=CPU, free_mb_fn=free, sleep_fn=lambda s: None, clock=clock)
    ref[0] = dm
    model = Flaky(ref, limit)
    dm.place(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return dm, model, opt, clock


def loss_sum(logits, y):
    return F.cross_entropy(logits, y, reduction="sum")


def batch(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 6, generator=g), torch.randint(0, 3, (n,), generator=g)


def test_is_oom():
    assert is_oom(torch.cuda.OutOfMemoryError("x"))
    assert is_oom(RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert not is_oom(RuntimeError("shape mismatch"))
    assert not is_oom(ValueError("out of memory"))


def test_micro_batches_give_exactly_the_full_batch_update():
    x, y = batch()
    dm_ref, ref_model, ref_opt, _ = make(limit=None)
    dm, model, opt, _ = make(limit=8)
    model.lin.load_state_dict(copy.deepcopy(ref_model.lin.state_dict()))
    l_ref = dm_ref.train_step(ref_model, ref_opt, x, y, loss_sum, len(y))
    l_mb = dm.train_step(model, opt, x, y, loss_sum, len(y))
    assert dm.chunks == 4 and not dm.on_fallback and dm.stats["oom_events"] == 2
    assert l_mb == pytest.approx(l_ref, rel=1e-5)
    for a, b in zip(ref_model.parameters(), model.parameters()):
        assert torch.allclose(a, b, atol=1e-6)
    model.calls.clear()
    dm.train_step(model, opt, x, y, loss_sum, len(y))  # the chosen split is remembered: no new OOM
    assert [c[0] for c in model.calls] == [8, 8, 8, 8] and dm.stats["oom_events"] == 2


def test_fallback_to_cpu_then_back_to_gpu():
    free = {"mb": 0.0}
    dm, model, opt, clock = make(limit=0, free=lambda: free["mb"])  # every GPU forward fails
    x, y = batch()
    dm.train_step(model, opt, x, y, loss_sum, len(y))
    assert dm.on_fallback and dm.current == CPU and dm.stats["switches_to_fallback"] == 1
    assert model.calls[-1] == (32, False)  # the step was done on the CPU with the full batch
    clock.t += 30
    dm.maybe_return(model, opt)
    assert dm.on_fallback  # too early to check
    clock.t += 60
    dm.maybe_return(model, opt)
    assert dm.on_fallback  # checked, but not enough free memory
    free["mb"] = 5000
    clock.t += 61
    model.limit = None  # GPU has room again
    dm.train_step(model, opt, x, y, loss_sum, len(y))
    assert not dm.on_fallback and dm.stats["switches_back"] == 1 and model.calls[-1][1]
    assert dm.stats["fallback_batches"] == 1


def test_wait_mode_retries_on_gpu():
    state = {"free": 0.0, "sleeps": 0}
    dm, model, opt, _ = make(limit=0, free=lambda: state["free"], mode="wait")

    def sleep(_):
        state["sleeps"] += 1
        if state["sleeps"] == 3:
            state["free"] = 9999
            model.limit = None
    dm.sleep = sleep
    x, y = batch()
    dm.train_step(model, opt, x, y, loss_sum, len(y))
    assert state["sleeps"] == 3 and not dm.on_fallback and model.calls[-1][1]


def test_wait_mode_gives_up():
    clock_calls = iter(range(0, 10_000, 30))
    dm = DeviceManager(GPU, cfg(mode="wait", max_wait_minutes=1), fallback=CPU, free_mb_fn=lambda: 0.0,
                       sleep_fn=lambda s: None, clock=lambda: next(clock_calls))
    ref = [dm]
    model = Flaky(ref, limit=0)
    dm.place(model)
    x, y = batch()
    with pytest.raises(RuntimeError, match="max_wait_minutes"):
        dm.train_step(model, torch.optim.SGD(model.parameters(), lr=0.1), x, y, loss_sum, len(y))


def test_mode_off_raises():
    dm, model, opt, _ = make(limit=0, mode="off")
    x, y = batch()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        dm.train_step(model, opt, x, y, loss_sum, len(y))


def test_inference_splits_then_falls_back():
    dm, model, _, _ = make(limit=5)
    x, _ = batch(20)
    ref = model.lin(x)
    out = dm.forward(model, x)
    assert torch.allclose(out, ref, atol=1e-6) and not dm.on_fallback
    dm2, model2, _, _ = make(limit=0)
    model2.lin.load_state_dict(model.lin.state_dict())
    out2 = dm2.forward(model2, x)
    assert torch.allclose(out2, ref, atol=1e-6) and dm2.on_fallback


def test_optimizer_state_moves_and_step_stays_on_cpu():
    dm, model, _, _ = make(limit=None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = batch()
    dm.train_step(model, opt, x, y, loss_sum, len(y))
    dm.move(model, opt, CPU)
    st = next(iter(opt.state.values()))
    assert st["exp_avg"].device.type == "cpu" and st["step"].device.type == "cpu"
    dm.train_step(model, opt, x, y, loss_sum, len(y))  # AdamW still works after the move


def test_invalid_mode():
    with pytest.raises(ValueError):
        DeviceManager(GPU, cfg(mode="gpu-only"), fallback=CPU)
