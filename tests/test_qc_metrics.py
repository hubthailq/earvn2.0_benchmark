"""QC helpers and metrics, checked against brute-force implementations."""
import numpy as np
import pandas as pd
import pytest

from earbench import qc
from earbench.metrics import (clustering_metrics, gender_metrics, hungarian_accuracy, identification,
                              identity_metrics, leave_one_out_identification, mean_ci95, mean_std, sample_pairs,
                              verification)


def test_hamming_pairs_matches_bruteforce():
    rng = np.random.default_rng(0)
    h = rng.integers(0, 2 ** 63, size=300, dtype=np.uint64)
    h[5] = h[7]
    h[10] = h[11] ^ np.uint64(0b111)
    got = set(map(tuple, qc.hamming_pairs(h, 3, block=37)[["i", "j"]].to_numpy()))
    want = {(i, j) for i in range(300) for j in range(i + 1, 300) if bin(int(h[i]) ^ int(h[j])).count("1") <= 3}
    assert got == want and (5, 7) in got and (10, 11) in got


def test_hamming_cross_and_popcount_fallback(monkeypatch):
    a = np.array([0, 1, 2 ** 64 - 1], dtype=np.uint64)
    b = np.array([3, 2 ** 64 - 2], dtype=np.uint64)
    got = qc.hamming_pairs(a, 1, b)
    assert set(map(tuple, got[["i", "j"]].to_numpy())) == {(1, 0), (2, 1)}
    x = np.array([0, 1, 3, 2 ** 64 - 1], dtype=np.uint64)
    monkeypatch.delattr(np, "bitwise_count", raising=False)  # numpy < 2.0 path
    assert qc._popcount64(x).tolist() == [0, 1, 2, 64]
    assert set(map(tuple, qc.hamming_pairs(a, 1, b)[["i", "j"]].to_numpy())) == {(1, 0), (2, 1)}


def test_duplicate_groups_only_merge_same_subject():
    pairs = pd.DataFrame({"i": [0, 1, 3], "j": [1, 2, 4]})
    subj = np.array(["a", "a", "a", "a", "b"])
    g = qc.duplicate_groups(5, pairs, subj)
    assert g[0] == g[1] == g[2] and g[3] != g[4]


def test_leakage_report():
    split = pd.Series(["train", "test", "test", "val"])
    groups = np.array([0, 0, 2, 2])
    r = qc.leakage_report(split, groups, np.array(["a"] * 4))
    assert r["n_test_with_train_duplicate"] == 1 and r["test_leakage_rate"] == 0.5


def test_outlier_detection():
    rng = np.random.default_rng(0)
    f = np.tile([1.0, 0, 0], (30, 1)) + rng.normal(0, 0.01, (30, 3))
    f[7] = [0, 1, 0]
    sc = qc.outlier_scores(f, np.array(["s"] * 30))
    assert sc.z.idxmin() == 7 and sc.z[7] < -3
    assert qc.outlier_scores(np.ones((2, 3)), np.array(["a", "a"])).z.isna().all()  # too few images


def test_subject_overlap():
    pairs = pd.DataFrame({"i": [0, 1, 2, 3], "j": [0, 1, 2, 5]})
    a = np.array(["x", "x", "x", "y"])
    b = np.array(["p", "p", "p", "q", "q", "q"])
    ov = qc.subject_overlap(pairs, a, b, 3)
    assert ov.to_dict("records") == [{"subject_earvn2": "x", "subject_earvn1": "p", "n_matched_images": 3}]


def test_identity_metrics_and_subject_mean():
    scores = np.eye(3)[[0, 0, 0, 0, 1, 2]]
    y = np.array([0, 0, 0, 1, 1, 2])  # subject 1: 1/2 correct
    m = identity_metrics(scores, y)
    assert m["top1"] == pytest.approx(100 * 5 / 6)
    assert m["top1_subject_mean"] == pytest.approx(100 * (1 + 0.5 + 1) / 3)
    assert m["top5"] == 100


def test_gender_metrics_with_subject_vote_and_ties():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.9, 0.8, 0.7, 0.2, 0.3])
    subj = np.array(["a", "a", "b", "b", "c", "c"])  # a: tie -> wrong, b: right, c: wrong
    m = gender_metrics(p, y, subj)
    assert m["acc"] == pytest.approx(100 * 3 / 6)
    assert m["acc_subject"] == pytest.approx(100 / 3)
    assert np.isnan(gender_metrics(np.array([0.2, 0.7]), np.array([1, 1]))["auc"])


def test_hungarian_is_permutation_invariant():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert hungarian_accuracy(y, np.array([5, 5, 3, 3, 9, 9])) == 100
    assert hungarian_accuracy(y, np.array([0, 0, 0, 0, 0, 0])) == pytest.approx(100 / 3)
    m = clustering_metrics(y, np.array([1, 1, 2, 2, 0, 0]))
    assert m["nmi"] == pytest.approx(100) and m["ari"] == pytest.approx(100)


def _brute_rank(pf, pid, gf, gid):
    ranks = []
    for f, t in zip(pf, pid):
        best = {}
        for g, i in zip(gf, gid):
            s = float(f @ g / (np.linalg.norm(f) * np.linalg.norm(g)))
            best[i] = max(best.get(i, -9), s)
        ranks.append(1 + sum(v > best[t] for k, v in best.items()))
    return np.array(ranks)


def test_identification_matches_bruteforce():
    rng = np.random.default_rng(0)
    gf, gid = rng.normal(size=(40, 8)), np.repeat(np.arange(10), 4)
    pf, pid = rng.normal(size=(25, 8)), rng.integers(0, 10, 25)
    r = identification(pf, pid, gf, gid, block=7)
    ranks = _brute_rank(pf, pid, gf, gid)
    assert r["rank1"] == pytest.approx(100 * (ranks <= 1).mean())
    assert r["rank5"] == pytest.approx(100 * (ranks <= 5).mean())
    assert len(r["cmc"]) == 10 and r["cmc"][-1] == 1.0
    with pytest.raises(ValueError):
        identification(pf, pid + 100, gf, gid)


def test_leave_one_out_matches_bruteforce():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(30, 6))
    ids = np.array([0] * 5 + [1] * 5 + [2] * 10 + [3] * 9 + [4])  # id 4 has one image: distractor only
    r = leave_one_out_identification(f, ids, block=4)
    ranks = []
    for i in range(30):
        if (ids == ids[i]).sum() < 2:
            continue
        mask = np.arange(30) != i
        ranks.append(_brute_rank(f[i:i + 1], ids[i:i + 1], f[mask], ids[mask])[0])
    assert r["n_probes"] == 29
    assert r["rank1"] == pytest.approx(100 * np.mean(np.array(ranks) <= 1))


def test_verification():
    v = verification(np.array([0.9, 0.8, 0.95]), np.array([0.1, 0.2, 0.3]))
    assert v["eer"] == 0 and v["tar_at_far"] == 100
    v = verification(np.ones(10), np.ones(10))
    assert 0 <= v["eer"] <= 100
    assert np.isnan(verification(np.array([]), np.array([0.1]))["eer"])


def test_sample_pairs():
    ids = np.repeat(np.arange(5), 4)
    (gi, gj), (ii, jj) = sample_pairs(ids, 50, np.random.default_rng(0))
    assert len(gi) == 5 * 6 and (ids[gi] == ids[gj]).all()
    assert len(ii) == 50 and (ids[ii] != ids[jj]).all()


def test_summaries():
    assert mean_std([1, 2, 3]) == (2.0, 1.0)
    assert mean_std([5]) == (5.0, 0.0)
    m, h = mean_ci95([1, 2, 3, 4])
    assert m == 2.5 and h > 0
    assert np.isnan(mean_std([])[0])


def test_identification_with_fewer_identities_than_rank():
    rng = np.random.default_rng(3)
    r = identification(rng.normal(size=(4, 5)), np.array([0, 1, 0, 1]), rng.normal(size=(2, 5)), np.array([0, 1]))
    assert r["rank5"] == 100 and r["rank10"] == 100 and 0 <= r["rank1"] <= 100
    r = leave_one_out_identification(rng.normal(size=(3, 4)), np.array([0, 1, 2]))  # nobody can be a probe
    assert r["n_probes"] == 0 and np.isnan(r["rank1"])
