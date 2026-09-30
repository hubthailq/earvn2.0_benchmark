"""Few-shot classifiers, K-means, scanning, gender assignment, model registry, tables."""
import numpy as np
import pandas as pd
import pytest

from earbench.fewshot import kmeans_labels, linear_probe, prototype_predict
from earbench.metrics import hungarian_accuracy
from earbench.models import families, load_registry, select_models
from earbench.scan import assign_gender, dataset_statistics, scan_dataset, suggest_nonsquare
from earbench.tables import build_tables, fmt, to_latex, write_tables


def clusters(n_class=5, per=20, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_class, dim)) * 5
    y = np.repeat(np.arange(n_class), per)
    return centers[y] + rng.normal(size=(len(y), dim)), y


def test_prototype_and_linear_probe_on_separable_data():
    x, y = clusters()
    sup = np.arange(0, 100, 20)  # 1 shot per class
    pred, scores, classes = prototype_predict(x[sup], y[sup], x)
    assert (pred == y).mean() > 0.95 and scores.shape == (100, 5)
    pred, proba, cls, C = linear_probe(x[sup], y[sup], x[sup + 1], y[sup + 1], x)
    assert (pred == y).mean() > 0.95 and proba.shape == (100, 5)
    with pytest.raises(ValueError):
        linear_probe(x[:2], np.array([0, 0]), None, None, x)
    pred, _, _ = prototype_predict(x[sup], y[sup], x[:0])
    assert len(pred) == 0


def test_kmeans():
    x, y = clusters()
    assert hungarian_accuracy(y, kmeans_labels(x, 5, seed=0)) > 95
    assert hungarian_accuracy(y, kmeans_labels(x, 5, seed=0, minibatch_above=10)) > 80
    with pytest.raises(ValueError):
        kmeans_labels(x[:3], 5, 0)


def test_scan_edge_cases(syn_presplit, cfg_factory, tmp_path):
    cfg = cfg_factory(syn_presplit, tmp_path / "out")
    images, subjects, problems = scan_dataset(cfg, workers=2)
    assert subjects[:3] == ["001", "002", "003"] and len(subjects) == 10  # + too-few + empty folder
    st = images.set_index("image_id").status
    assert st["001/test/broken.jpg"].startswith("corrupt")
    assert st["001/test/empty.jpg"] == "empty_file"
    for ok in ("gray.png", "alpha.png", "palette.png", "cmyk.jpg", "sixteen_bit.png", "exif_rot.jpg",
               "tai phải ảnh.jpg", "UPPER.JPG", "extra_folder/nested.jpg"):
        assert st[f"001/test/{ok}"] == "ok", ok
    assert images.set_index("image_id").loc["001/test/exif_rot.jpg", "width"] == 60  # EXIF applied
    assert images.set_index("image_id").loc["001/test/tiny.png", "tiny"]
    probs = set(problems.problem)
    assert {"not_an_image_extension", "file_outside_subject_folder", "subject_has_no_images"} <= probs
    assert not images.image_id.str.contains("DS_Store").any()
    assert set(images.orig_split) == {"train", "val", "test", ""}  # '' = subject without split folders
    stats = dataset_statistics(images, assign_gender(cfg, subjects), cfg.resolution_bins)
    assert stats["pre_split_layout"] and stats["n_images_unreadable"] == 2


def test_scan_missing_or_empty_root(cfg_factory, tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_dataset(cfg_factory(tmp_path / "nope", tmp_path / "o"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        scan_dataset(cfg_factory(tmp_path / "empty", tmp_path / "o"))


def test_gender_from_rule_and_file(cfg_factory, tmp_path):
    subs = ["1", "2", "10", "11"]
    cfg = cfg_factory(tmp_path, tmp_path / "o", {"gender.male_count": 2})
    g = assign_gender(cfg, subs)
    assert g.gender.tolist() == ["M", "M", "F", "F"]
    with pytest.raises(ValueError):
        assign_gender(cfg_factory(tmp_path, tmp_path / "o", {"gender.male_count": 4}), subs)
    lab = tmp_path / "labels.csv"
    pd.DataFrame({"Subject": subs, "Gender": ["f", "m", "M", "F"]}).to_csv(lab, index=False)
    cfg2 = cfg_factory(tmp_path, tmp_path / "o", {"paths.gender_labels": str(lab)})
    assert assign_gender(cfg2, subs).gender.tolist() == ["F", "M", "M", "F"]
    pd.DataFrame({"subject": subs[:3], "gender": ["M"] * 3}).to_csv(lab, index=False)
    with pytest.raises(ValueError, match="no gender"):
        assign_gender(cfg2, subs)
    pd.DataFrame({"subject": subs, "gender": ["M", "X", "M", "F"]}).to_csv(lab, index=False)
    with pytest.raises(ValueError, match="must be"):
        assign_gender(cfg2, subs)
    pd.DataFrame({"subject": subs + ["1"], "gender": ["M"] * 5}).to_csv(lab, index=False)
    with pytest.raises(ValueError, match="twice"):
        assign_gender(cfg2, subs)


def test_suggest_nonsquare():
    assert suggest_nonsquare(1.9) == (224, 128)
    assert suggest_nonsquare(0.5) == (224, 224)
    assert suggest_nonsquare(float("nan")) == (224, 224)


def test_registry():
    reg = load_registry()
    assert len(reg) == 30 and len({m.key for m in reg}) == 30
    fams = families(reg)
    assert len(fams) == 11
    for f in fams:
        n = sum(m.family == f for m in reg)
        assert 2 <= n <= 4, f
        assert sum(m.lr_rep for m in reg if m.family == f) == 1, f"family {f} needs exactly one lr_rep"
    assert len(select_models("basic", reg)) == 5
    assert [m.key for m in select_models("vit_s16,resnet18", reg)] == ["resnet18", "vit_s16"]
    with pytest.raises(KeyError):
        select_models("resnet999", reg)


def test_tables_empty_and_filled(tmp_path):
    from earbench.config import load_config
    cfg = load_config()
    reg = load_registry()
    tabs = build_tables(reg, tmp_path, cfg)
    assert len(tabs["E1"]) == 30 and (tabs["E1"]["Top-1 theo ảnh"] == "–").all()
    pd.DataFrame({"model": ["resnet18"] * 3, "split": "S1", "run": [0, 1, 2], "test_top1": [80.0, 82.0, 84.0],
                  "test_top1_subject_mean": 81.0, "test_top5": 95.0, "test_macro_f1": 79.0}).to_csv(
        tmp_path / "E1.csv", index=False)
    tabs = build_tables(reg, tmp_path, cfg)
    row = tabs["E1"][tabs["E1"]["Mô hình"] == "ResNet-18"].iloc[0]
    assert row["Top-1 theo ảnh"] == "82.00 ± 2.00"
    out = write_tables(tabs, tmp_path / "tables")
    assert out.exists() and "ResNet-18" in out.read_text(encoding="utf-8")
    assert r"$\pm$" in to_latex(tabs["E1"], "t") and "82.00" in to_latex(tabs["E1"], "t")
    assert fmt([]) == "–" and fmt([1.234]) == "1.23" and fmt(["x", None]) == "–"


def test_gender_requires_explicit_count_or_file(cfg_factory, tmp_path):
    cfg = cfg_factory(tmp_path, tmp_path / "o", {"gender.male_count": "null"})
    with pytest.raises(ValueError, match="Gender labels are missing"):
        assign_gender(cfg, ["1", "2", "3"])


def test_borderline_sample_and_contact_sheet(tmp_path):
    from PIL import Image

    from earbench.qc import borderline_sample, contact_sheet
    rng = np.random.default_rng(0)
    pairs = pd.DataFrame({"i": range(200), "j": range(1, 201), "cos": np.linspace(0.85, 1.0, 200)})
    rev = borderline_sample(pairs, "cos", 0.95, True, 20, rng)
    assert len(rev) == 20 and set(rev.side) == {"counted_as_duplicate", "not_duplicate"}
    assert ((rev.cos >= 0.95) == (rev.side == "counted_as_duplicate")).all()
    assert rev.cos.between(0.90, 1.0).all()  # closest to the threshold, not the extremes
    assert borderline_sample(pairs.iloc[0:0], "cos", 0.95, True, 20, rng).empty
    Image.new("RGB", (20, 40), (10, 200, 10)).save(tmp_path / "a.png")
    rev = pd.DataFrame({"image_a": ["a.png"], "image_b": ["missing.png"], "subject_a": ["s"], "cos": [0.97],
                        "side": ["counted_as_duplicate"]})
    contact_sheet(tmp_path, rev, "cos", tmp_path / "sheet.jpg")  # a missing image must not crash
    assert Image.open(tmp_path / "sheet.jpg").size[1] == 112
