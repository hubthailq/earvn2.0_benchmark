"""Same training conditions for every model: fixed micro-batch plan, channels_last, one precision."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from earbench import train as T
from earbench.config import load_config
from earbench.device import DeviceManager, has_batchnorm, plan_micro_batches, resolve_amp
from earbench.models import ModelSpec, load_registry
from earbench.tables import hyperparameter_table

GPU, CPU = torch.device("cpu", 0), torch.device("cpu")


def loss_sum(logits, y):
    return F.cross_entropy(logits, y, reduction="sum")


class OOM(torch.cuda.OutOfMemoryError):
    pass


# ------------------------------------------------------------------ plan_micro_batches
def test_plan_full_batch_fits():
    assert plan_micro_batches(lambda c: 5.0, 64, 4, budget=8.0) == (1, 5.0)


def test_plan_splits_until_it_fits():
    tried = []

    def step(c):
        tried.append(c)
        if c < 4:
            raise OOM("CUDA out of memory")
        return 12.0 / c
    assert plan_micro_batches(step, 64, 4, budget=8.0) == (4, 3.0)
    assert tried == [1, 2, 4]


def test_plan_peak_over_budget_counts_as_not_fitting():
    assert plan_micro_batches(lambda c: 16.0 / c, 64, 4, budget=8.0) == (2, 8.0)


def test_plan_never_fits_and_min_micro():
    tried = []

    def step(c):
        tried.append(c)
        raise OOM("CUDA out of memory")
    assert plan_micro_batches(step, 64, 16, budget=1.0) == (None, None)
    assert tried == [1, 2, 4]                      # micro-batches of 64, 32, 16 images, never below 16


def test_plan_other_errors_are_not_hidden():
    def step(c):
        raise ValueError("bug")
    with pytest.raises(ValueError):
        plan_micro_batches(step, 64, 4, budget=1.0)


def test_plan_tiny_batch():
    assert plan_micro_batches(lambda c: 1.0, 1, 4, budget=2.0) == (1, 1.0)   # batch smaller than min_micro


# ------------------------------------------------------------------ planned split in DeviceManager
def ln_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 3))


def grads_after_step(dm, model, x, y):
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    dm.place(model)
    loss = dm.train_step(model, opt, x, y, loss_sum, float(len(y)))
    return loss, [p.grad.clone() for p in model.parameters()]


def test_planned_micro_batches_give_the_full_batch_gradient():
    x, y = torch.randn(16, 6), torch.randint(0, 3, (16,))
    m1, m4 = ln_model(), copy.deepcopy(ln_model())
    l1, g1 = grads_after_step(DeviceManager(GPU, {"mode": "cpu"}, fallback=CPU), m1, x, y)
    dm4 = DeviceManager(GPU, {"mode": "cpu"}, fallback=CPU, micro_batches=4)
    calls = []
    m4.register_forward_hook(lambda mod, inp, out: calls.append(len(inp[0])))
    l4, g4 = grads_after_step(dm4, m4, x, y)
    assert calls == [4, 4, 4, 4]                   # split from the very first step, not after an OOM
    assert l1 == pytest.approx(l4, abs=1e-6)
    for a, b in zip(g1, g4):
        assert torch.allclose(a, b, atol=1e-6)
    assert dm4.stats["max_micro_batches"] == 4 and dm4.stats["oom_events"] == 0


def test_return_from_cpu_restores_the_planned_split_not_one():
    t = [0.0]
    dm = DeviceManager(GPU, {"mode": "cpu", "min_free_mb_to_return": 10, "check_every_s": 1}, fallback=CPU,
                       micro_batches=2, free_mb_fn=lambda: 100.0, clock=lambda: t[0])
    model = ln_model()
    dm.place(model)
    dm.chunks = 8
    dm._to_fallback(model, None, "test")
    t[0] = 5.0
    dm.maybe_return(model)
    assert dm.current == GPU and dm.chunks == 2


# ------------------------------------------------------------------ channels_last
def conv_net():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 4))


def test_channels_last_same_result():
    x, y = torch.randn(8, 3, 16, 16), torch.randint(0, 4, (8,))
    a, b = conv_net(), conv_net()
    la, ga = grads_after_step(DeviceManager(GPU, {"mode": "cpu"}, fallback=CPU), a, x, y)
    dm = DeviceManager(GPU, {"mode": "cpu"}, fallback=CPU, channels_last=True)
    lb, gb = grads_after_step(dm, b, x, y)
    assert b[0].weight.is_contiguous(memory_format=torch.channels_last)
    assert la == pytest.approx(lb, abs=1e-5)
    for p, q in zip(ga, gb):
        assert torch.allclose(p, q, atol=1e-5)
    a.eval(), b.eval()
    assert torch.allclose(DeviceManager(GPU, {"mode": "cpu"}, fallback=CPU).forward(a, x),
                          dm.forward(b, x), atol=1e-5)


def test_channels_last_ignores_non_image_inputs():
    dm = DeviceManager(GPU, {"mode": "cpu"}, fallback=CPU, channels_last=True)
    m = ln_model()
    dm.place(m)
    assert dm.forward(m, torch.randn(5, 6)).shape == (5, 3)


def test_channels_last_only_for_cnn_families():
    cfg = load_config()
    fam = {m.family: T.use_channels_last(cfg, m) for m in load_registry()}
    assert fam["ResNet"] and fam["ConvNeXt"] and fam["EfficientNet"]
    assert not fam["ViT / DeiT"] and not fam["Swin"] and not fam["Pretrain nền tảng"]


def test_has_batchnorm():
    assert has_batchnorm(conv_net()) and not has_batchnorm(ln_model())


# ------------------------------------------------------------------ precision
def test_resolve_amp_cpu_and_bad_value():
    cfg = load_config()
    assert resolve_amp(cfg.train, torch.device("cpu")) == (False, torch.float32, "fp32")
    with pytest.raises(ValueError):
        resolve_amp(load_config(overrides=["train.amp_dtype=int8"]).train, torch.device("cpu"))


def test_inference_precision_is_configurable():
    dm = DeviceManager(GPU, {"mode": "cpu"}, fallback=CPU, infer_amp_dtype=None)
    m = ln_model().eval()
    dm.place(m)
    out = dm.forward(m, torch.randn(3, 6))
    assert out.dtype == torch.float32


# ------------------------------------------------------------------ memory plan lookup
SPEC = ModelSpec(key="resnet18", name="ResNet-18", family="ResNet", timm="resnet18.tv_in1k")


def plan_cfg(tmp_path, *extra):
    return load_config(overrides=[f"paths.output_root={tmp_path}", *extra])


def write_plan(tmp_path, **kw):
    row = {"model": "resnet18", "gpu_name": "NVIDIA GeForce RTX 3080", "batch_size": 64, "size": 224,
           "amp_dtype": "bf16", "channels_last": True, "micro_batches": 2, "micro_batch_size": 32,
           "bn_micro_batch": True}
    row.update(kw)
    d = tmp_path / "results" / "per_run"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(d / "memory_plan.csv", index=False)


class Log:
    def __init__(self):
        self.msgs = []

    def warning(self, m):
        self.msgs.append(m)


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(T, "gpu_name", lambda d: "NVIDIA GeForce RTX 3080")
    return torch.device("cuda")


def test_plan_lookup_cpu_is_always_one(tmp_path):
    write_plan(tmp_path)
    assert T.planned_micro_batches(plan_cfg(tmp_path), SPEC, torch.device("cpu"), "fp32") == 1


def test_plan_lookup_match(tmp_path, fake_gpu):
    write_plan(tmp_path)
    log = Log()
    assert T.planned_micro_batches(plan_cfg(tmp_path), SPEC, fake_gpu, "bf16", log) == 2
    assert log.msgs == []


@pytest.mark.parametrize("override,amp", [("train.batch_size=32", "bf16"), ("preprocess.size=192", "bf16"),
                                          ("train.channels_last_families=[]", "bf16"), (None, "fp16")])
def test_plan_lookup_settings_changed(tmp_path, fake_gpu, override, amp):
    write_plan(tmp_path)
    log = Log()
    cfg = plan_cfg(tmp_path, *([override] if override else []))
    assert T.planned_micro_batches(cfg, SPEC, fake_gpu, amp, log) == 1
    assert "different settings" in log.msgs[0]


def test_plan_lookup_other_gpu_missing_model_missing_file_nofit(tmp_path, fake_gpu, monkeypatch):
    log = Log()
    assert T.planned_micro_batches(plan_cfg(tmp_path), SPEC, fake_gpu, "bf16", log) == 1   # no file
    assert "05_model_info" in log.msgs[-1]
    write_plan(tmp_path, model="vgg11")
    assert T.planned_micro_batches(plan_cfg(tmp_path), SPEC, fake_gpu, "bf16", log) == 1   # model missing
    write_plan(tmp_path, micro_batches=np.nan)
    assert T.planned_micro_batches(plan_cfg(tmp_path), SPEC, fake_gpu, "bf16", log) == 1   # does not fit
    assert "does not fit" in log.msgs[-1]
    write_plan(tmp_path)
    monkeypatch.setattr(T, "gpu_name", lambda d: "NVIDIA GeForce RTX 4090")
    assert T.planned_micro_batches(plan_cfg(tmp_path), SPEC, fake_gpu, "bf16", log) == 1   # other GPU
    assert T.planned_micro_batches(plan_cfg(tmp_path, "train.memory_plan.use=false"), SPEC, fake_gpu, "bf16") == 1


def test_hp_table_reports_gpu_and_accumulation():
    plan = pd.DataFrame([
        {"model": "dinov2_b", "gpu_name": "NVIDIA GeForce RTX 3080", "micro_batches": 2, "micro_batch_size": 32,
         "bn_micro_batch": False},
        {"model": "effnetv2_s", "gpu_name": "NVIDIA GeForce RTX 3080", "micro_batches": 2, "micro_batch_size": 32,
         "bn_micro_batch": True},
        {"model": "resnet18", "gpu_name": "NVIDIA GeForce RTX 3080", "micro_batches": 1, "micro_batch_size": 64,
         "bn_micro_batch": False}])
    df = hyperparameter_table(load_config(), load_registry(), {}, plan).set_index("Tham số")["Giá trị"]
    assert df["GPU"] == "NVIDIA GeForce RTX 3080"
    acc = df["Tích luỹ gradient (batch hiệu dụng giữ nguyên)"]
    assert "dinov2_b (2 × 32)" in acc and "resnet18" not in acc
    assert df["BatchNorm trên micro-batch (lưu ý khi so sánh)"] == "effnetv2_s"
