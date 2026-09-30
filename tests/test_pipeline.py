"""End-to-end: run every numbered script on tiny synthetic datasets (CPU, random weights).

Checks the real command-line entry points, resumability (second run skips finished work), the
leakage decision of Q1, the S1 re-split keeping near-duplicates together, and the final tables.
Run with:  pytest -m slow   (takes a few minutes on CPU)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SMALL = ["pretrained=false", "device=cpu", "num_workers=0", "gender.male_count=4", "preprocess.size=64",
         "preprocess.nonsquare_hw=[64,32]", "train.batch_size=32", "train.epochs=2", "train.scheduler.warmup_epochs=1",
         "train.learning_rate.sweep_epochs=1", "train.learning_rate.sweep_grid=[0.001,0.0003]", "train.seeds=[0,1]",
         "fewshot.episodes_prototype=4", "fewshot.episodes_linear=2", "fewshot.single_gallery_runs=2",
         "fewshot.gender_shots=[1,2]", "fewshot.gender_linear_shots=[2]", "retrieval.runs=2",
         "unsupervised.kmeans_restarts=2", "qc.pair_rule=phash", "feature_batch_size=64", "splits.s2.n_folds=3"]
# random-weight DINOv2 features are all alike, so only pHash defines near-duplicates in these tests
MODELS = "resnet18,mobilenetv3s"


def run(script, root, out, *args, extra=()):
    sets = [f"paths.dataset_root={root}", f"paths.output_root={out}",
            f"paths.gender_labels={out}/none.csv", f"paths.earvn1_root={out}/none1", f"paths.awe_root={out}/none2"]
    cmd = [sys.executable, str(ROOT / "scripts" / script)]
    for s in SMALL + sets + list(extra):
        cmd += ["--set", s]
    cmd += list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout + r.stderr


def inject_leak(root: Path, n_subjects: int = 6):
    """Copy a train image into the test folder of several subjects -> leakage above 1%."""
    for s in sorted(p for p in root.iterdir() if (p / "train").is_dir())[:n_subjects]:
        src = sorted((s / "train").glob("*.jpg"))[0]
        shutil.copy(src, s / "test" / f"leak_{src.name}")


@pytest.mark.slow
def test_full_pipeline_presplit_with_leakage(syn_presplit, tmp_path):
    root = tmp_path / "ds"
    shutil.copytree(syn_presplit, root)
    inject_leak(root)
    out = tmp_path / "out"
    run("00_check_env.py", root, out)
    run("01_scan_dataset.py", root, out)
    run("02_qc_duplicates.py", root, out)
    rep = json.loads((out / "qc" / "leakage_report.json").read_text())
    assert rep["decision_s1"] == "resplit" and rep["n_test_with_train_duplicate"] >= 6
    run("03_qc_labels.py", root, out)
    review = pd.read_csv(out / "qc" / "outliers_for_review.csv", dtype=str, keep_default_na=False)
    review.loc[review.image_id.str.contains("tiny"), "remove"] = "y"
    review.to_csv(out / "qc" / "outliers_for_review.csv", index=False)
    run("03_qc_labels.py", root, out, "--apply")
    assert "tiny" in (out / "qc" / "removed.csv").read_text()
    run("04_make_splits.py", root, out)

    s1 = pd.read_csv(out / "splits" / "s1_identity.csv", dtype=str, keep_default_na=False)
    assert "001/test/tiny.png" not in set(s1.image_id)
    for subj, g in s1[s1.split != "excluded"].groupby("subject"):
        assert (g.split == "train").sum() == 45 and (g.split == "val").sum() == 15, subj
    # the leaked copies now sit on the same side as their original
    for grp, g in s1.groupby("group"):
        if len(g) > 1 and g.subject.nunique() == 1:
            assert g.split.nunique() == 1, f"near-duplicate group {grp} split across {set(g.split)}"

    run("05_model_info.py", root, out, "--models", MODELS)
    run("06_lr_sweep.py", root, out, "--models", MODELS)
    run("07_e0_preprocessing.py", root, out, "--models", MODELS)
    run("08_train_supervised.py", root, out, "--exp", "E1", "--models", MODELS)
    run("08_train_supervised.py", root, out, "--exp", "E2", "--models", "resnet18")
    log = run("08_train_supervised.py", root, out, "--exp", "E1", "--models", MODELS)
    assert "E1 resnet18" not in log  # resumable: finished runs are skipped
    run("09_train_s3.py", root, out, "--models", MODELS)
    run("10_extract_features.py", root, out, "--what", "all", "--models", MODELS)
    run("11_fewshot.py", root, out, "--models", MODELS)
    run("12_unsupervised.py", root, out, "--models", MODELS)
    run("13_new_identity.py", root, out, "--models", MODELS)
    run("14_cross_dataset.py", root, out, "--models", MODELS)
    run("15_resolution_analysis.py", root, out)
    # resumability: unchanged inputs are skipped, a changed setting is re-run
    log = run("11_fewshot.py", root, out, "--models", MODELS)
    assert "skip E3 resnet18 (done)" in log and "mean top1" not in log
    log = run("13_new_identity.py", root, out, "--models", "resnet18")
    assert "EER=" not in log
    log = run("12_unsupervised.py", root, out, "--models", "resnet18", extra=["unsupervised.kmeans_max_iter=50"])
    assert "E56 resnet18" in log
    log = run("16_make_tables.py", root, out)

    res = out / "results" / "per_run"
    e1 = pd.read_csv(res / "E1.csv")
    assert set(e1.amp_dtype_used) == {"fp32"} and set(e1.gpu_name) == {"cpu"}          # CPU test machine
    assert set(e1.planned_micro_batches) == {1} and not e1.bn_micro_batch.astype(bool).any()
    assert e1.channels_last.astype(bool).all()                                          # both are CNNs
    assert len(e1) == 4 and e1.test_top1.between(0, 100).all()
    e2 = pd.read_csv(res / "E2.csv")
    assert set(e2.split) == {"S1", "S2"} and (e2.split == "S2").sum() == 3
    e3 = pd.read_csv(res / "E3.csv")
    assert set(e3.classifier) == {"proto", "lp"} and set(e3.shot) == {1, 2, 5}
    assert len(pd.read_csv(res / "E3b.csv")) == 2 * 2
    cmc = pd.read_csv(res / "E3b_cmc.csv")
    assert set(cmc.model) == set(MODELS.split(",")) and cmc.groupby("model").cmc.apply(
        lambda c: c.is_monotonic_increasing and abs(c.iloc[-1] - 100) < 1e-6).all()
    for f in ("E1", "E2", "E3", "E3b", "E4", "E56", "E7", "E7_verification", "lr_sweep", "E0"):
        d = pd.read_csv(res / f"{f}.csv")
        assert d.fingerprint.notna().all(), f
    assert (pd.read_csv(res / "E1.csv").status == "ok").all()
    e4 = pd.read_csv(res / "E4.csv")
    assert set(e4.fold) == set(range(3))
    assert set(pd.read_csv(res / "E56.csv").task) == {"gender", "identity"}
    e7 = pd.read_csv(res / "E7.csv")
    assert set(e7.variant) == {"finetuned", "frozen"} and set(e7.shot) == {1, 5}
    assert len(pd.read_csv(res / "E7_verification.csv")) == 4
    for f in ("E0", "lr_sweep", "resolution", "model_info"):
        assert (res / f"{f}.csv").exists()
    tables = (out / "tables" / "all_tables.md").read_text(encoding="utf-8")
    assert "ResNet-18" in tables and "E7 – Nhận dạng người mới" in tables
    final = out / "results"                                   # final CSVs: one row per model (+ split/shot/...)
    need = {"E1_identity_supervised": ["test_top1_mean", "test_top5_mean", "test_top10_mean",
                                       "test_macro_precision_mean", "test_macro_recall_mean", "test_macro_f1_mean",
                                       "test_weighted_f1_std", "n_runs"],
            "E2_gender_supervised": ["test_acc_mean", "test_precision_female_mean", "test_recall_male_mean",
                                     "test_f1_mean", "test_auc_mean", "test_mcc_mean"],
            "E3_identity_fewshot": ["top1_mean", "top5_mean", "macro_precision_mean", "macro_recall_ci95"],
            "E3b_single_gallery": ["rank1_mean", "rank5_mean", "mrr_mean", "macro_f1_mean"],
            "E4_gender_fewshot": ["bal_acc_mean", "precision_female_mean", "recall_female_mean"],
            "E5_E6_unsupervised": ["acc_mean", "nmi_mean", "purity_mean", "macro_f1_mean"],
            "E7_new_identity_retrieval": ["rank1_mean", "macro_recall_mean"],
            "E7_new_identity_verification": ["eer_mean", "tar_at_far_mean"],
            "E0_preprocessing": ["test_top1_mean"], "LR_sweep": ["val_top1_mean"], "model_info": ["gmacs"]}
    for name, cols in need.items():
        d = pd.read_csv(final / f"{name}.csv")
        assert set(cols) <= set(d.columns), (name, set(cols) - set(d.columns))
        assert {"model", "name", "family"} <= set(d.columns) and len(d), name
    e1f = pd.read_csv(final / "E1_identity_supervised.csv")
    assert set(e1f.model) == set(MODELS.split(",")) and (e1f.n_runs == 2).all()
    assert (final / "README.md").exists() and (final / "per_run" / "E1.csv").exists()
    for f in ("E1", "E2", "E3", "E4", "E56", "E7", "HP"):
        assert (out / "tables" / f"{f}.tex").exists()
    assert "Siêu tham số huấn luyện" in tables and "AdamW" in tables
    assert (res / "chosen_lr.json").exists() and (res / "chosen_lr_for_config.yaml").exists()
    hp = list((out / "runs").rglob("hparams.json"))
    assert hp and all(json.loads(h.read_text(encoding="utf-8"))["optimizer"]["name"] == "adamw" for h in hp)


@pytest.mark.slow
def test_pipeline_flat_layout_with_external_datasets(syn_flat, syn_presplit, tmp_path):
    """Flat folders (no split on disk), duplicates inside a subject, EarVN1.0 overlap for Q3/E8."""
    root = tmp_path / "flat"
    shutil.copytree(syn_flat, root)
    subj = root / "002"
    src = sorted(subj.glob("*.jpg"))[0]
    for k in range(4):
        shutil.copy(src, subj / f"copy{k}.jpg")
    e1 = tmp_path / "earvn1"
    shutil.copytree(syn_presplit / "005", e1 / "same_person")
    shutil.copytree(syn_presplit / "006", e1 / "other")
    out = tmp_path / "out"
    ext = [f"paths.earvn1_root={e1}"]

    def r(script, *a):
        return run(script, root, out, *a, extra=ext)

    r("01_scan_dataset.py")
    r("02_qc_duplicates.py", "--no-dino")
    r("04_make_splits.py")
    s1 = pd.read_csv(out / "splits" / "s1_identity.csv", dtype=str, keep_default_na=False)
    copies = s1[s1.image_id.str.contains("copy") | (s1.image_id == f"002/{src.name}")]
    assert copies.split.nunique() == 1 and copies.group.nunique() == 1
    r("09_train_s3.py", "--models", "mobilenetv3s")
    r("10_extract_features.py", "--what", "all", "--models", "mobilenetv3s")
    # flat synthetic data and presplit data share no people -> no overlap expected here,
    # so plant one: copy a flat subject into EarVN1.0
    shutil.copytree(root / "003", e1 / "planted")
    (out / "metadata" / "external_earvn1.csv").unlink()
    r("14_cross_dataset.py", "--models", "mobilenetv3s", "--no-dino")
    ov = pd.read_csv(out / "qc" / "overlap_earvn1.csv", dtype=str)
    assert "planted" in set(ov.subject_earvn1)
    r("10_extract_features.py", "--what", "external", "--models", "mobilenetv3s", "--overwrite")
    r("14_cross_dataset.py", "--models", "mobilenetv3s", "--no-dino", "--overwrite")
    e8 = pd.read_csv(out / "results" / "per_run" / "E8.csv")
    assert set(e8.variant) == {"finetuned", "frozen"} and (e8.n_excluded_subjects >= 1).all()
    assert np.isfinite(e8.rank1).all()
