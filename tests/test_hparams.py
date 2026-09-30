"""Training recipe read from configs/default.yaml: optimizer, scheduler, LR priority, hparams table."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from earbench.config import load_config
from earbench.models import load_registry
from earbench.tables import hyperparameter_table
from earbench.tasks import chosen_lr
from earbench.train import build_optimizer, build_scheduler, effective_hparams

def model():
    m = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 2))
    m[0].weight.requires_grad_(True)
    return m

def test_default_recipe_values():
    tc = load_config().train
    assert tc.epochs == 100 and tc.patience == 15 and tc.batch_size == 64
    assert tc.optimizer.name == "adamw" and tc.scheduler.name == "cosine"
    assert list(tc.learning_rate.sweep_grid) == [3e-4, 1e-4, 3e-5]

def test_adamw_param_groups():
    cfg = load_config()
    opt = build_optimizer(model(), cfg.train.optimizer, 1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == pytest.approx(0.05) and no_decay["weight_decay"] == 0.0
    assert all(p.ndim == 2 for p in decay["params"])          # only weight matrices are decayed
    assert all(p.ndim == 1 for p in no_decay["params"])       # bias + LayerNorm
    assert opt.param_groups[0]["betas"] == (0.9, 0.999) and opt.param_groups[0]["lr"] == 1e-3

def test_sgdm_and_no_group_split():
    cfg = load_config(overrides=["train.optimizer.name=sgdm", "train.optimizer.nesterov=true",
                                 "train.optimizer.no_decay_bias_norm=false"])
    m = model()
    for p in m[1].parameters():
        p.requires_grad_(False)                                # frozen params are not optimised
    opt = build_optimizer(m, cfg.train.optimizer, 0.01)
    assert isinstance(opt, torch.optim.SGD) and len(opt.param_groups) == 1
    g = opt.param_groups[0]
    assert g["momentum"] == 0.9 and g["nesterov"] and g["weight_decay"] == pytest.approx(0.05)
    assert len(g["params"]) == 4

def test_bad_optimizer_and_scheduler_names():
    cfg = load_config(overrides=["train.optimizer.name=lion", "train.scheduler.name=poly"])
    with pytest.raises(ValueError, match="optimizer"):
        build_optimizer(model(), cfg.train.optimizer, 1e-3)
    opt = build_optimizer(model(), load_config().train.optimizer, 1e-3)
    with pytest.raises(ValueError, match="scheduler"):
        build_scheduler(opt, cfg.train.scheduler, 10, 5)

def lrs(cfg, epochs, steps):
    opt = build_optimizer(model(), cfg.train.optimizer, 1.0)
    sch = build_scheduler(opt, cfg.train.scheduler, epochs, steps)
    out = []
    for _ in range(epochs * steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    return out

def test_cosine_with_warmup():
    lr = lrs(load_config(), epochs=10, steps=4)               # warm-up 2 epochs = 8 steps
    assert lr[0] == pytest.approx(1 / 8) and lr[7] == pytest.approx(1.0)
    assert lr[8] == pytest.approx(1.0)                         # cosine starts at the peak
    assert all(a >= b - 1e-12 for a, b in zip(lr[8:], lr[9:]))  # then decreases monotonically
    assert lr[-1] < 0.01

def test_cosine_floor_and_step():
    lr = lrs(load_config(overrides=["train.scheduler.min_lr_ratio=0.1", "train.scheduler.warmup_epochs=0"]), 5, 3)
    assert lr[0] == pytest.approx(1.0) and min(lr) >= 0.1 - 1e-9
    lr = lrs(load_config(overrides=["train.scheduler.name=step", "train.scheduler.warmup_epochs=0",
                                    "train.scheduler.step_epochs=[2,4]"]), 5, 2)
    assert lr[:4] == pytest.approx([1.0] * 4) and lr[4:8] == pytest.approx([0.1] * 4)
    assert lr[8:] == pytest.approx([0.01] * 2)

def test_constant_and_warmup_longer_than_training():
    lr = lrs(load_config(overrides=["train.scheduler.name=constant", "train.scheduler.warmup_epochs=0"]), 3, 2)
    assert lr == pytest.approx([1.0] * 6)
    lr = lrs(load_config(overrides=["train.scheduler.warmup_epochs=50"]), 2, 3)   # edge case: never crashes
    assert all(math.isfinite(x) and 0 < x <= 1 for x in lr)

def test_chosen_lr_priority(tmp_path):
    paths = SimpleNamespace(lr_json=tmp_path / "chosen_lr.json")
    cfg = load_config(overrides=['train.learning_rate.by_family={"ResNet": 0.002}'])
    assert chosen_lr(paths, "Swin", cfg) == pytest.approx(1e-4)          # nothing known -> default
    paths.lr_json.write_text(json.dumps({"Swin": 3e-5, "ResNet": 3e-4}))
    assert chosen_lr(paths, "Swin", cfg) == pytest.approx(3e-5)          # sweep result
    assert chosen_lr(paths, "ResNet", cfg) == pytest.approx(0.002)       # config wins over sweep

def test_effective_hparams_is_json_serialisable():
    cfg = load_config()
    spec = load_registry()[0]
    hp = effective_hparams(cfg, spec, "identity", 1e-4, cfg.train.epochs, None)
    back = json.loads(json.dumps(hp))
    assert back["optimizer"]["name"] == "adamw" and back["max_epochs"] == 100 and back["preprocess"]["mode"] == "letterbox"

def test_hyperparameter_table_lists_every_family():
    reg = load_registry()
    cfg = load_config(overrides=['train.learning_rate.by_family={"Swin": 5e-5}'])
    df = hyperparameter_table(cfg, reg, {"ResNet": 3e-4})
    fam_rows = {r["Tham số"].split("– ")[1]: r["Giá trị"] for _, r in df.iterrows() if r["Tham số"].startswith("Learning")}
    assert set(fam_rows) == {m.family for m in reg}
    assert "cố định" in fam_rows["Swin"] and "quét" in fam_rows["ResNet"] and "mặc định" in fam_rows["VGG"]
