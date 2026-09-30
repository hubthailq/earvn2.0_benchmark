"""Final result CSVs (outputs/results/) and the full metric set (top-k, precision, recall, F1, ...)."""
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from earbench.metrics import (clustering_metrics, gender_metrics, identification, identity_metrics,
                              leave_one_out_identification)
from earbench.models import load_registry
from earbench.summary import summarise, write_summaries


def test_identity_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 20, 300)
    scores = rng.normal(size=(300, 25)) + 2.5 * np.eye(25)[y]
    m = identity_metrics(scores, y)
    pred = scores.argmax(1)
    assert m["top1"] == pytest.approx(100 * skm.accuracy_score(y, pred))
    assert m["top5"] == pytest.approx(100 * skm.top_k_accuracy_score(y, scores, k=5, labels=np.arange(25)))
    assert m["top10"] == pytest.approx(100 * skm.top_k_accuracy_score(y, scores, k=10, labels=np.arange(25)))
    labels = np.unique(y)
    for avg in ("macro", "weighted"):
        assert m[f"{avg}_precision"] == pytest.approx(100 * skm.precision_score(y, pred, labels=labels, average=avg,
                                                                                zero_division=0))
        assert m[f"{avg}_recall"] == pytest.approx(100 * skm.recall_score(y, pred, labels=labels, average=avg,
                                                                          zero_division=0))
        assert m[f"{avg}_f1"] == pytest.approx(100 * skm.f1_score(y, pred, labels=labels, average=avg,
                                                                  zero_division=0))
    assert m["macro_recall"] == pytest.approx(100 * skm.balanced_accuracy_score(y, pred))
    assert all(np.isnan(v) for v in identity_metrics(np.zeros((0, 5)), np.array([])).values())


def test_gender_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.3, 200), 0, 1)
    m = gender_metrics(p, y)
    pred = (p >= 0.5).astype(int)
    assert m["precision_female"] == pytest.approx(100 * skm.precision_score(y, pred, pos_label=1))
    assert m["recall_female"] == pytest.approx(100 * skm.recall_score(y, pred, pos_label=1))
    assert m["recall_male"] == pytest.approx(100 * skm.recall_score(y, pred, pos_label=0))   # specificity
    assert m["f1_male"] == pytest.approx(100 * skm.f1_score(y, pred, pos_label=0))
    assert m["f1"] == pytest.approx(m["macro_f1"]) == pytest.approx(100 * skm.f1_score(y, pred, average="macro"))
    assert m["mcc"] == pytest.approx(100 * skm.matthews_corrcoef(y, pred))
    assert m["bal_acc"] == pytest.approx(m["macro_recall"])
    one = gender_metrics(np.array([0.9, 0.8]), np.array([1, 1]))       # a single class: no crash
    assert np.isnan(one["auc"]) and np.isnan(one["mcc"]) and one["acc"] == 100


def test_clustering_metrics():
    y = np.array([0] * 5 + [1] * 5 + [2] * 2)
    c = np.array([7] * 5 + [8] * 4 + [7] + [9, 9])
    m = clustering_metrics(y, c)
    assert m["acc"] == pytest.approx(100 * 11 / 12) and m["purity"] == pytest.approx(100 * 11 / 12)
    assert m["v_measure"] == pytest.approx(100 * skm.v_measure_score(y, c))
    assert 0 < m["macro_f1"] < 100
    m2 = clustering_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]))   # fewer clusters than classes
    assert m2["acc"] == 50 and m2["macro_recall"] == pytest.approx(50)


def test_identification_precision_recall_and_mrr():
    rng = np.random.default_rng(2)
    gal = rng.normal(size=(10, 16))
    ids = np.arange(10)
    probe_ids = np.repeat(ids, 3)
    probes = gal[probe_ids] + rng.normal(0, 0.8, (30, 16))
    r = identification(probes, probe_ids, gal, ids)
    sims = (probes / np.linalg.norm(probes, axis=1, keepdims=True)) @ (gal / np.linalg.norm(gal, axis=1, keepdims=True)).T
    pred = sims.argmax(1)
    assert r["rank1"] == pytest.approx(100 * (pred == probe_ids).mean())
    assert r["macro_f1"] == pytest.approx(100 * skm.f1_score(probe_ids, pred, average="macro", labels=ids,
                                                             zero_division=0))
    ranks = 1 + (sims > sims[np.arange(30), probe_ids][:, None]).sum(1)
    assert r["mrr"] == pytest.approx(100 * (1 / ranks).mean())
    loo = leave_one_out_identification(np.r_[gal, gal + 0.01], np.r_[ids, ids])
    assert loo["rank1"] == 100 and loo["macro_f1"] == 100


def test_summarise_mean_std_ci_and_diverged():
    reg = load_registry()
    raw = pd.DataFrame([
        {"model": "resnet18", "split": "S1", "run": 0, "seed": 0, "lr": 3e-4, "status": "ok", "test_top1": 80.0,
         "test_macro_f1": 70.0, "gpu_oom_events": 0, "channels_last": True},
        {"model": "resnet18", "split": "S1", "run": 1, "seed": 1, "lr": 3e-4, "status": "ok", "test_top1": 90.0,
         "test_macro_f1": 72.0, "gpu_oom_events": 0, "channels_last": True},
        {"model": "resnet18", "split": "S1", "run": 2, "seed": 2, "lr": 3e-4, "status": "diverged"},
        {"model": "vgg11", "split": "S1", "run": 0, "seed": 0, "lr": 1e-4, "status": "ok", "test_top1": 60.0,
         "test_macro_f1": 50.0, "gpu_oom_events": 0, "channels_last": True}])
    s = summarise(raw, ["split"], True, reg)
    assert list(s.model) == ["vgg11", "resnet18"]                     # registry order
    r = s.set_index("model").loc["resnet18"]
    assert r.n_runs == 2 and r.n_diverged == 1 and r.lr == "0.0003" and r["name"] == "ResNet-18"
    assert r.test_top1_mean == 85 and r.test_top1_std == pytest.approx(7.0711, abs=1e-4)
    assert r.test_top1_ci95 == pytest.approx(63.5310, abs=1e-3)
    assert not any(c.startswith(("gpu_", "seed", "run_", "channels_last")) for c in s.columns)
    assert s.set_index("model").loc["vgg11", "test_top1_std"] == 0.0


def test_write_summaries(tmp_path):
    per_run, out = tmp_path / "per_run", tmp_path
    per_run.mkdir()
    pd.DataFrame([{"model": "resnet18", "classifier": "proto", "shot": k, "episode": e, "top1": 10.0 * k + e,
                   "top5": 50.0, "macro_precision": 1.0, "macro_recall": 2.0, "macro_f1": 3.0}
                  for k in (1, 5) for e in range(3)]).to_csv(per_run / "E3.csv", index=False)
    pd.DataFrame([{"model": "resnet18", "params_m": 11.2, "gmacs": 1.8}]).to_csv(per_run / "model_info.csv",
                                                                                 index=False)
    files = write_summaries(per_run, out, load_registry())
    assert {f.name for f in files} == {"E3_identity_fewshot.csv", "model_info.csv"}
    e3 = pd.read_csv(out / "E3_identity_fewshot.csv")
    assert list(e3.shot) == [1, 5] and list(e3.n_runs) == [3, 3] and list(e3.top1_mean) == [11.0, 51.0]
    assert {"top5_mean", "macro_precision_ci95", "macro_recall_std", "macro_f1_mean"} <= set(e3.columns)
    assert "macro_precision" in (out / "README.md").read_text(encoding="utf-8")
    (per_run / "E3.csv").write_text("")                              # results removed -> stale summary removed
    write_summaries(per_run, out, load_registry())
    assert not (out / "E3_identity_fewshot.csv").exists()
