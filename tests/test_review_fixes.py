"""Regression tests for the problems found in the pre-run review (one test per problem)."""
import importlib.util
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

from earbench import train as T
from earbench.common import load_s1
from earbench.config import PROJECT_ROOT, _parse_value, load_config
from earbench.device import DeviceManager
from earbench.metrics import pair_scores, verification
from earbench.qc import duplicate_groups, identical_file_pairs
from earbench.scan import scan_dataset
from earbench.splits import sample_gallery, sample_identity_episode
from earbench.tables import build_tables, fmt
from earbench.models import load_registry
from earbench.utils import feature_fingerprint, fingerprint

GPU, CPU = torch.device("cpu", 0), torch.device("cpu")


def OOM():
    return torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")


def loss_sum(logits, y):
    return F.cross_entropy(logits, y, reduction="sum")


def small_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 3))


# =============================================================== data side
def test_truncated_jpeg_is_detected_and_excluded(syn_flat, cfg_factory, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(syn_flat, root)
    src = sorted((root / "001").glob("*.jpg"))[0]
    big = np.random.default_rng(0).integers(0, 255, (300, 200, 3), dtype=np.uint8)
    Image.fromarray(big).save(src, quality=95)
    data = src.read_bytes()
    (root / "001" / "cut.jpg").write_bytes(data[: len(data) // 5])
    before = ImageFile.LOAD_TRUNCATED_IMAGES
    images, _, problems = scan_dataset(cfg_factory(root, tmp_path / "out"), workers=2)
    assert ImageFile.LOAD_TRUNCATED_IMAGES == before            # global flag restored for training
    r = images.set_index("image_id").loc["001/cut.jpg"]
    assert r.status == "truncated" and r.truncated and r.width == 200 and r.height == 300
    assert images.set_index("image_id").loc[src.relative_to(root).as_posix(), "status"] == "ok"
    cfg = cfg_factory(root, tmp_path / "out2")
    cfg.dataset["keep_truncated"] = True
    images, _, problems = scan_dataset(cfg, workers=2)
    assert images.set_index("image_id").loc["001/cut.jpg", "status"] == "ok"
    assert problems.problem.str.contains("truncated").any()


def test_identical_files_are_paired_whatever_their_size():
    md5 = ["a", "b", "a", "", "", "a", "c", "b"]
    p = identical_file_pairs(md5)
    assert sorted(map(tuple, p.values.tolist())) == [(0, 2), (1, 7), (2, 5)]   # '' (unreadable) never paired
    groups = duplicate_groups(len(md5), p, np.array(["s"] * 8))
    assert groups[0] == groups[2] == groups[5] and groups[1] == groups[7] and groups[3] != groups[4]
    assert identical_file_pairs([]).empty and identical_file_pairs(["x", "y"]).empty


def test_load_s1_detects_stale_split(tmp_path):
    images = pd.DataFrame({"image_id": ["a", "b", "c"], "subject": ["1", "1", "1"], "group": ["a", "a", "c"]})
    paths = SimpleNamespace(s1_csv=tmp_path / "s1.csv")
    pd.DataFrame({"image_id": ["a", "b", "c"], "split": ["train", "train", "test"],
                  "group": ["a", "a", "c"]}).to_csv(paths.s1_csv, index=False)
    assert list(load_s1(paths, images).split) == ["train", "train", "test"]
    with pytest.raises(ValueError, match="no longer usable"):          # image removed after the split
        load_s1(paths, images[images.image_id != "c"])
    with pytest.raises(ValueError, match="near-duplicate groups"):     # 02 re-run with other thresholds
        load_s1(paths, images.assign(group=["a", "b", "c"]))
    with pytest.raises(ValueError, match="not in"):                    # new image after the split
        load_s1(paths, pd.concat([images, images.iloc[[0]].assign(image_id="d")]))


@pytest.mark.parametrize("text,value", [("0300", 300), ("-7", -7), ("1e-4", 1e-4), ("3.0e-5", 3e-5), (".5", 0.5),
                                        ("true", True), ("null", None), ("abc", "abc"), ("[1e-4, 3e-5]", [1e-4, 3e-5]),
                                        ("[224, 096]", [224, 96]), ('{ResNet: 3e-4}', {"ResNet": 3e-4}),
                                        ("[true, x]", [True, "x"]), ('["007"]', ["007"]), ('"1e-4"', "1e-4"),
                                        ("[yes, no]", [True, False]), ("[1_000, 0x10]", [1000, 16])])
def test_cli_values_parse_like_people_read_them(text, value):
    assert _parse_value(text) == value


def test_gender_count_with_leading_zero_via_cli():
    assert load_config(overrides=["gender.male_count=0300"]).gender.male_count == 300


# =============================================================== training side
class Spy(DeviceManager):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.moves = []

    def move(self, model, optimizer=None, device=None):
        eff = optimizer if optimizer is not None else self.optimizer
        self.moves.append((str(torch.device(device or self.current)), eff is not None))
        super().move(model, optimizer, device)


def test_optimizer_always_moves_with_the_model():
    t = [0.0]
    dm = Spy(GPU, {"mode": "cpu", "min_free_mb_to_return": 10, "check_every_s": 1}, fallback=CPU,
             free_mb_fn=lambda: 100.0, clock=lambda: t[0])
    m = small_model()
    dm.place(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    dm.init_optimizer_state(m, opt)
    dm._to_fallback(m, None, "test")                   # e.g. an OOM during validation
    t[0] = 10.0
    dm.forward(m.eval(), torch.randn(4, 6))            # validation brings the model back ...
    assert dm.current == GPU
    assert all(with_opt for _, with_opt in dm.moves[1:])   # ... and every move carried the optimizer


def test_state_preallocation_is_numerically_identical():
    x, y = torch.randn(32, 6), torch.randint(0, 3, (32,))
    for make in (lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=0.05),
                 lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9, nesterov=True, weight_decay=0.05)):
        a, b = small_model(), small_model()
        oa, ob = make(a.parameters()), make(b.parameters())
        DeviceManager(CPU).init_optimizer_state(b, ob)
        for _ in range(3):
            for m, o in ((a, oa), (b, ob)):
                o.zero_grad()
                loss_sum(m(x), y).backward()
                o.step()
        for p, q in zip(a.parameters(), b.parameters()):
            assert torch.equal(p, q)


def test_oom_inside_optimizer_step_does_not_crash():
    dm = DeviceManager(GPU, {"mode": "cpu"}, fallback=CPU)
    m = small_model()
    dm.place(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    dm.init_optimizer_state(m, opt)
    real, calls = opt.step, [0]

    def flaky_step(*a, **k):
        calls[0] += 1
        if calls[0] == 1:
            raise OOM()
        return real(*a, **k)
    opt.step = flaky_step
    x, y = torch.randn(8, 6), torch.randint(0, 3, (8,))
    dm.train_step(m, opt, x, y, loss_sum, 8.0)        # interrupted step: flagged, moved to the fallback
    assert dm.stats["step_oom"] == 1 and dm.current == CPU
    dm.train_step(m, opt, x, y, loss_sum, 8.0)        # training continues normally
    assert calls[0] == 2 and all(torch.isfinite(p).all() for p in m.parameters())


class LoadsLate(torch.nn.Linear):
    """Raises OOM the first `fails` times it is moved to the 'GPU'."""

    def __init__(self, fails):
        super().__init__(6, 3)
        self.fails = fails

    def to(self, *args, **kwargs):
        dev = args[0] if args else kwargs.get("device")
        if dev is not None and torch.device(dev) == GPU and self.fails > 0:
            self.fails -= 1
            raise OOM()
        return super().to(*args, **kwargs)


class FakeTime:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def wait_dm(tm, **kw):
    fb = {"mode": "wait", "min_free_mb_to_return": 10, "check_every_s": 60, "max_wait_minutes": 60, **kw}
    return DeviceManager(GPU, fb, fallback=CPU, free_mb_fn=lambda: 100.0, sleep_fn=tm.sleep, clock=tm.clock)


def test_wait_mode_retries_with_growing_pauses():
    tm = FakeTime()
    dm = wait_dm(tm)
    m = LoadsLate(fails=3)
    dm.place(m)                     # GPU looks free, but the model fits only on the 4th try
    assert dm.current == GPU and m.fails == 0
    assert tm.sleeps == [60, 120]   # 1st retry immediately, then pauses grow (not 5 retries in 1 ms)
    tm = FakeTime()
    with pytest.raises(RuntimeError, match="max_wait_minutes"):       # gives up after max_wait, not forever
        wait_dm(tm).place(LoadsLate(fails=10 ** 6))
    assert 3600 <= tm.t <= 3600 + 1800


def test_wait_streak_resets_after_successful_inference():
    tm = FakeTime()
    dm = wait_dm(tm)
    m = small_model().eval()
    dm.place(m)
    dm._wait_streak = 4                       # several earlier, separate OOM waits ...
    dm.forward(m, torch.randn(4, 6))          # ... a successful batch clears the streak
    assert dm._wait_streak == 0
    dm._wait_streak = 4
    tm.t = 10 ** 6                            # a new OOM long after: counted from its own start
    dm._wait_streak = 0
    dm._wait_for_gpu("test")
    assert dm._wait_streak == 1 and tm.sleeps == []


def test_diverged_run_is_recorded_not_raised(monkeypatch):
    def boom(*a, **k):
        raise FloatingPointError("non-finite loss")
    monkeypatch.setattr(T, "train_and_evaluate", boom)
    r = T.train_or_diverge(None, logger=None)
    assert r["status"] == "diverged" and "non-finite" in r["error"]
    monkeypatch.setattr(T, "train_and_evaluate", lambda *a, **k: {"val_top1": 1.0})
    assert T.train_or_diverge(None) == {"status": "ok", "val_top1": 1.0}


def test_run_fingerprint_changes_with_recipe_and_data():
    cfg = load_config()
    spec = load_registry()[2]
    tr = pd.DataFrame({"image_id": ["a", "b"], "label": [0, 1]})
    va = pd.DataFrame({"image_id": ["c"], "label": [0]})
    fp = T.run_fingerprint(cfg, spec, "identity", None, None, tr, va, None)
    assert fp == T.run_fingerprint(load_config(), spec, "identity", None, None, tr.copy(), va, None)
    assert fp != T.run_fingerprint(load_config(overrides=["train.epochs=30"]), spec, "identity", None, None, tr, va, None)
    assert fp != T.run_fingerprint(cfg, spec, "identity", None, None, tr.assign(label=[1, 0]), va, None)
    assert fp != T.run_fingerprint(cfg, spec, "identity", "stretch", None, tr, va, None)
    assert fp.startswith("fp") and fp != T.run_fingerprint(cfg, spec, "identity", None, None, tr, va, None, seed=1)
    assert fp == T.run_fingerprint(load_config(overrides=["gpu_fallback.mode=wait"]), spec, "identity", None, None,
                                   tr, va, None)             # how the run survives OOM is not part of the recipe


def test_transformer_tokens_get_no_weight_decay():
    import timm
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=3)
    opt = T.build_optimizer(m, load_config().train.optimizer, 1e-4)
    no_decay = {id(p) for p in opt.param_groups[1]["params"]}
    named = dict(m.named_parameters())
    assert id(named["pos_embed"]) in no_decay and id(named["cls_token"]) in no_decay
    assert id(named["blocks.0.attn.qkv.weight"]) not in no_decay


def test_memory_probe_runs_like_training(monkeypatch):
    """Script 05's GPU probe, with the CUDA memory calls simulated on the CPU."""
    spec = importlib.util.spec_from_file_location("m05", PROJECT_ROOT / "scripts" / "05_model_info.py")
    m05 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m05)
    reserved = iter([9, 3, 3, 3])                                # GB: the full batch is too big, 2 halves fit
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (8 * 2 ** 30, 10 * 2 ** 30))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d=None: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d=None: next(reserved) * 2 ** 30)
    cfg = load_config(overrides=["train.batch_size=4", "train.eval_batch_size=4", "preprocess.size=64",
                                 "train.memory_plan.probe_steps=1", "train.memory_plan.timed_steps=1",
                                 "gpu_fallback.min_micro_batch=2"])
    r = m05.probe(load_registry()[2], cfg, GPU, 10)              # resnet18
    assert r["micro_batches"] == 2 and r["micro_batch_size"] == 2 and r["peak_gb"] == 3.0
    assert r["batchnorm"] and r["bn_micro_batch"] and r["budget_gb"] == pytest.approx(8 * 0.85, abs=0.01)


# =============================================================== evaluation side
def test_tar_at_far_exact_threshold():
    from sklearn.metrics import roc_curve
    rng = np.random.default_rng(0)
    gen, imp = rng.normal(1, 1, 4), rng.normal(0, 1, 100)
    fpr, tpr, _ = roc_curve(np.r_[np.ones(4), np.zeros(100)], np.r_[gen, imp])
    ref = 100 * tpr[fpr <= 0.01 + 1e-12].max()
    assert verification(gen, imp, 0.01)["tar_at_far"] == pytest.approx(ref)


def test_pair_scores_blocked():
    f = np.random.default_rng(1).normal(size=(50, 16)).astype(np.float32)
    a, b = np.arange(50), np.arange(50)[::-1]
    assert np.allclose(pair_scores(f, a, b, block=7), (f[a] * f[b]).sum(1), atol=1e-5)
    assert pair_scores(f, [], []).shape == (0,)


def test_episode_never_tests_on_copies_of_the_support():
    # subject 0: 12 images in only 2 near-duplicate groups; K=1 + 1 val block both groups
    codes = np.array([0] * 12 + [1] * 12)
    groups = np.array([0] * 6 + [1] * 6 + list(range(2, 14)))
    e = sample_identity_episode(codes, groups, 1, 1, np.random.default_rng(0))
    blocked = set(groups[np.r_[e["support"], e["val"]]])
    assert not (set(groups[e["test"]]) & blocked)
    assert e["subjects_without_test"] == 1 and not (codes[e["test"]] == 0).any()
    g = sample_gallery(codes, groups, 1, np.random.default_rng(0))
    assert not (set(groups[g["probe"]]) & set(groups[g["gallery"]]))


def test_fmt_marks_partial_averages():
    assert fmt([1.0, 2.0, 3.0], expected=3) == "2.00 ± 1.00"
    assert fmt([1.0, np.nan], expected=3) == "1.00 (n=1/3)"
    assert fmt([np.nan], expected=3) == "–"


def test_tables_have_frozen_and_subject_mean_columns(tmp_path):
    cfg = load_config()
    reg = load_registry()
    pd.DataFrame([{"model": "resnet18", "variant": v, "shot": s, "run": 0, "rank1": 50.0}
                  for v in ("finetuned", "frozen") for s in (1, 5)]).to_csv(tmp_path / "E7.csv", index=False)
    pd.DataFrame([{"model": "resnet18", "variant": v, "eer": 10.0, "tar_at_far": 60.0}
                  for v in ("finetuned", "frozen")]).to_csv(tmp_path / "E7_verification.csv", index=False)
    pd.DataFrame([{"model": "resnet18", "variant": v, "dataset": "earvn1", "rank1": 1.0, "rank5": 2.0, "aucmc": 3.0}
                  for v in ("finetuned", "frozen")]).to_csv(tmp_path / "E8.csv", index=False)
    pd.DataFrame([{"model": "resnet18", "classifier": "proto", "shot": 1, "episode": 0, "top1": 10.0,
                   "top1_subject_mean": 12.0}]).to_csv(tmp_path / "E3.csv", index=False)
    tabs = build_tables(reg, tmp_path, cfg)
    e7 = tabs["E7"].set_index("Mô hình").loc["ResNet-18"]
    assert e7["EER (đóng băng)"] == "10.00" and e7["1-shot Top-1 (đóng băng)"].startswith("50.00")
    assert set(tabs["E8"]["Đặc trưng"]) == {"tinh chỉnh S3", "đóng băng"}
    assert tabs["E3s"].set_index("Mô hình").loc["ResNet-18", "Proto 1-shot"].startswith("12.00")
    assert "(n=1/" in tabs["E3"].set_index("Mô hình").loc["ResNet-18", "Proto 1-shot"]
    assert "AUCMC" in tabs["E3b"].columns


def test_fingerprints(tmp_path):
    assert fingerprint({"a": 1}, np.arange(3)) == fingerprint({"a": 1}, [0, 1, 2])
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
    from earbench.features import save_features
    p = tmp_path / "f.npz"
    save_features(p, ["x"], np.zeros((1, 2)), {"model": "m"})
    fp = feature_fingerprint(p)
    assert fp["meta"] == {"model": "m"} and isinstance(fp["mtime"], int)
    json.dumps(fp)


def test_should_skip_compares_fingerprints_as_text(tmp_path):
    from earbench.utils import ResultStore
    st = ResultStore(tmp_path / "r.csv", ["model"])
    st.upsert({"model": "m", "fingerprint": "fp0123456789012e345", "lr": 1e-4})
    assert st.should_skip({"model": "m"}, False, fingerprint="fp0123456789012e345", lr=1e-4)
    assert not st.should_skip({"model": "m"}, False, fingerprint="fp0000000000000000", lr=1e-4)
    assert st.remove({"model": "m"}) == 1 and not st.has({"model": "m"})


def test_run_folder_is_cleaned_before_a_rerun(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    for f in ("predictions.csv", "metrics.json", "best.pt"):
        (run / f).write_text("old")
    cfg = load_config(overrides=["device=cpu"])
    empty = pd.DataFrame({"image_id": [], "label": [], "subject": []})
    with pytest.raises(ValueError):            # fails right after the clean-up (empty train set)
        T.train_and_evaluate(load_registry()[2], cfg, "identity", str(tmp_path), empty, empty, None, 2, 1e-3, 0,
                             run, torch.device("cpu"))
    assert not any((run / f).exists() for f in ("predictions.csv", "metrics.json", "best.pt"))


def test_config_file_uses_the_same_number_rules(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text((PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8").replace(
        "sweep_grid: [3.0e-4, 1.0e-4, 3.0e-5]", "sweep_grid: [3e-4, 1e-4]"), encoding="utf-8")
    assert load_config(p).train.learning_rate.sweep_grid == [3e-4, 1e-4]


def test_s3_features_must_come_from_the_current_checkpoint(tmp_path):
    import os
    from earbench.features import s3_features_are_current, save_features
    from earbench.paths import Paths
    paths = Paths(load_config(overrides=[f"paths.output_root={tmp_path}"]))
    ck = paths.run_dir("S3", "resnet18", "seed0") / "best.pt"
    ck.write_bytes(b"x")
    os.utime(ck, (1_000_000, 1_000_000))
    f = tmp_path / "s3ft.npz"
    save_features(f, ["a"], np.zeros((1, 2)), {"weights": "s3_seed0", "checkpoint_mtime": 1_000_000})
    assert s3_features_are_current(f, paths, "resnet18")
    os.utime(ck, (2_000_000, 2_000_000))                      # retrained after extraction
    assert not s3_features_are_current(f, paths, "resnet18")
    ck.unlink()                                                # retraining diverged: no checkpoint
    assert not s3_features_are_current(f, paths, "resnet18")


def test_released_layout_with_images_subfolder(tmp_path, cfg_factory):
    """EarVN2.0/{Description.txt, Images/001.ALI_HD/001 (1).jpg, ...}: the Images level is found by itself."""
    root = tmp_path / "EarVN2.0"
    (root / "Images").mkdir(parents=True)
    (root / "Description.txt").write_text("EarVN2.0")
    names = ["001.ALI_HD", "002.LeDuong_BL", "010.Chu_B", "099.Xx", "100.Yy"]
    rng = np.random.default_rng(0)
    for n in names:
        (root / "Images" / n).mkdir()
        for i in (1, 2, 10):
            Image.fromarray(rng.integers(0, 255, (40, 30, 3), dtype=np.uint8)).save(root / "Images" / n / f"{n[:3]} ({i}).jpg")
    cfg = cfg_factory(root, tmp_path / "out")
    assert cfg.paths.dataset_root.endswith("Images")
    images, subjects, problems = scan_dataset(cfg, workers=2)
    assert subjects == names and len(images) == 15 and set(images.status) == {"ok"} and problems.empty
    assert images.image_id.iloc[0] == "001.ALI_HD/001 (1).jpg" and set(images.orig_split) == {""}
    from earbench.scan import assign_gender
    g = assign_gender(cfg, subjects).set_index("subject").gender           # cfg_factory: male_count=4
    assert list(g) == ["M", "M", "M", "M", "F"]
    # a single subject that has train/val/test folders is NOT mistaken for the Images level
    one = tmp_path / "one"
    for sp in ("train", "val", "test"):
        (one / "001" / sp).mkdir(parents=True)
    from earbench.config import descend_single_folder
    assert descend_single_folder(str(one)) == str(one)
    assert descend_single_folder(str(root / "Images")) == str(root / "Images")
