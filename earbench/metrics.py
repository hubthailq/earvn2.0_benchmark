"""Metrics for classification, clustering, identification (CMC) and verification."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score, f1_score,
                             normalized_mutual_info_score, roc_auc_score)


# ----------------------------------------------------------------------------- classification
def topk_correct(scores: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Boolean per sample: true class among the k highest scores."""
    scores = np.asarray(scores)
    k = min(k, scores.shape[1])
    topk = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return (topk == np.asarray(y)[:, None]).any(axis=1)


def _prf(y, pred, labels, prefix: str) -> dict:
    """Macro and weighted precision / recall / F1 (%). Macro = every class weighs the same (the headline
    number); weighted = classes weighted by their number of test images."""
    from sklearn.metrics import precision_recall_fscore_support
    out = {}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, average=avg, zero_division=0)
        out.update({f"{prefix}{avg}_precision": 100 * float(p), f"{prefix}{avg}_recall": 100 * float(r),
                    f"{prefix}{avg}_f1": 100 * float(f)})
    return out


def identity_metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    """Closed-set identification from class scores (N x C; y = column index of the true class).

    top1 (= accuracy over images), top1_subject_mean (each subject weighs the same), top5, top10,
    macro / weighted precision, recall, F1 over the classes present in the test set (macro recall is the
    balanced accuracy)."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    if len(y) == 0:
        keys = ["top1", "top1_subject_mean", "top5", "top10"] + [f"{a}_{m}" for a in ("macro", "weighted")
                                                                 for m in ("precision", "recall", "f1")]
        return {k: float("nan") for k in keys}
    pred = scores.argmax(1)
    top1 = pred == y
    per_subject = pd.Series(top1).groupby(y).mean()
    out = {
        "top1": 100 * float(top1.mean()),
        "top1_subject_mean": 100 * float(per_subject.mean()),
        "top5": 100 * float(topk_correct(scores, y, 5).mean()),
        "top10": 100 * float(topk_correct(scores, y, 10).mean()),
    }
    out.update(_prf(y, pred, np.unique(y), ""))
    return out


def gender_metrics(prob_pos: np.ndarray, y: np.ndarray, subjects: np.ndarray | None = None,
                   threshold: float = 0.5, names=("male", "female")) -> dict:
    """y in {0,1} (1 = names[1], i.e. female). Accuracy, balanced accuracy, macro-F1 ('f1'), ROC-AUC,
    Matthews correlation, precision / recall / F1 of each class (recall of one class = sensitivity,
    of the other = specificity), macro / weighted averages, and subject-level (majority vote) accuracy."""
    from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
    y = np.asarray(y).astype(int)
    p = np.asarray(prob_pos, dtype=float)
    pred = (p >= threshold).astype(int)
    two = len(np.unique(y)) > 1
    out = {
        "acc": 100 * float((pred == y).mean()) if len(y) else float("nan"),
        "bal_acc": 100 * float(balanced_accuracy_score(y, pred)) if two else float("nan"),
        "f1": 100 * float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "auc": 100 * float(roc_auc_score(y, p)) if two else float("nan"),
        "mcc": 100 * float(matthews_corrcoef(y, pred)) if two else float("nan"),
    }
    pc, rc, fc, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], average=None, zero_division=0)
    for i, n in enumerate(names):
        out.update({f"precision_{n}": 100 * float(pc[i]), f"recall_{n}": 100 * float(rc[i]),
                    f"f1_{n}": 100 * float(fc[i])})
    out.update(_prf(y, pred, [0, 1], ""))
    if subjects is not None:
        df = pd.DataFrame({"s": subjects, "y": y, "pred": pred})
        g = df.groupby("s").agg(y=("y", "first"), vote=("pred", "mean"))
        # ties (exactly half) count as wrong: the vote is undecided
        subj_pred = np.where(g.vote > 0.5, 1, np.where(g.vote < 0.5, 0, -1))
        out["acc_subject"] = 100 * float((subj_pred == g.y.to_numpy()).mean()) if len(g) else float("nan")
    return out


# ----------------------------------------------------------------------------- clustering
def hungarian_accuracy(y_true: np.ndarray, y_cluster: np.ndarray) -> float:
    """Best one-to-one matching between clusters and classes (unmatched clusters count as wrong)."""
    y_true = np.asarray(y_true)
    y_cluster = np.asarray(y_cluster)
    classes, yt = np.unique(y_true, return_inverse=True)
    clusters, yc = np.unique(y_cluster, return_inverse=True)
    w = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    np.add.at(w, (yc, yt), 1)
    r, c = linear_sum_assignment(-w)
    return 100 * float(w[r, c].sum()) / len(y_true)


def clustering_metrics(y_true, y_cluster) -> dict:
    """ACC (best one-to-one cluster->class mapping, Hungarian), NMI, ARI, purity, homogeneity, completeness,
    V-measure, and precision / recall / F1 of the clusters after that same mapping (all %)."""
    from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score
    y_true = np.asarray(y_true)
    y_cluster = np.asarray(y_cluster)
    classes, yt = np.unique(y_true, return_inverse=True)
    clusters, yc = np.unique(y_cluster, return_inverse=True)
    w = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    np.add.at(w, (yc, yt), 1)
    r, c = linear_sum_assignment(-w)
    mapping = np.full(len(clusters), -1)          # clusters left without a class count as wrong
    mapping[r] = c
    pred = mapping[yc]
    out = {"acc": 100 * float(w[r, c].sum()) / len(y_true),
           "nmi": 100 * float(normalized_mutual_info_score(y_true, y_cluster)),
           "ari": 100 * float(adjusted_rand_score(y_true, y_cluster)),
           "purity": 100 * float(w.max(axis=1).sum()) / len(y_true),
           "homogeneity": 100 * float(homogeneity_score(y_true, y_cluster)),
           "completeness": 100 * float(completeness_score(y_true, y_cluster)),
           "v_measure": 100 * float(v_measure_score(y_true, y_cluster))}
    out.update(_prf(yt, pred, np.arange(len(classes)), ""))
    return out


# ----------------------------------------------------------------------------- identification
def _id_layout(gallery_ids: np.ndarray):
    """Column order that groups gallery images by identity, and the start of each identity block."""
    gallery_ids = np.asarray(gallery_ids)
    order = np.argsort(gallery_ids, kind="stable")
    ids, starts = np.unique(gallery_ids[order], return_index=True)
    return ids, order, starts


def _rank_block(sim_block: np.ndarray, order, starts, true_col: np.ndarray):
    """(rank of the true identity, identity index predicted at rank 1) for a block of probes."""
    per_id = np.maximum.reduceat(sim_block[:, order], starts, axis=1)
    true_score = per_id[np.arange(len(true_col)), true_col]
    # rank = 1 + number of identities scoring strictly higher (ties count in favour of the probe)
    rank = 1 + (per_id > true_score[:, None]).sum(axis=1)
    pred = np.where(rank == 1, true_col, per_id.argmax(axis=1))
    return rank, pred


def _summarise_ranks(rank: np.ndarray, n_ids: int, ranks, true_col=None, pred=None) -> dict:
    prf_keys = [f"{a}_{m}" for a in ("macro", "weighted") for m in ("precision", "recall", "f1")]
    if len(rank) == 0:
        return {**{f"rank{r}": float("nan") for r in ranks}, "aucmc": float("nan"), "mrr": float("nan"),
                **{k: float("nan") for k in prf_keys}, "cmc": np.full(n_ids, np.nan)}
    cmc = np.array([(rank <= r).mean() for r in range(1, n_ids + 1)])
    # rank-k with k >= number of identities is trivially 100%; keep the key so tables never break
    out = {f"rank{r}": 100 * float((rank <= r).mean()) for r in ranks}
    out["aucmc"] = 100 * float(cmc.mean())
    out["mrr"] = 100 * float((1.0 / rank).mean())           # mean reciprocal rank
    if true_col is not None:
        # precision / recall / F1 of the rank-1 decision, over the identities that have probes
        out.update(_prf(true_col, pred, np.unique(true_col), ""))
    out["cmc"] = cmc
    return out


def identification(probe_feats: np.ndarray, probe_ids: np.ndarray, gallery_feats: np.ndarray,
                   gallery_ids: np.ndarray, ranks=(1, 5, 10), block: int = 1024) -> dict:
    """Closed-set identification with cosine similarity.
    A probe's score for an identity is its best similarity over that identity's gallery images.
    Returns rank-k rates, the CMC curve and its normalised area (AUCMC)."""
    from .qc import l2n
    pf, gf = l2n(probe_feats), l2n(gallery_feats)
    probe_ids = np.asarray(probe_ids)
    ids, order, starts = _id_layout(gallery_ids)
    missing = ~np.isin(probe_ids, ids)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} probes have no gallery image of their identity")
    true_col = np.searchsorted(ids, probe_ids)
    parts = [_rank_block(pf[s:s + block] @ gf.T, order, starts, true_col[s:s + block])
             for s in range(0, len(pf), block)]
    rank = np.concatenate([r for r, _ in parts]) if parts else np.array([], dtype=int)
    pred = np.concatenate([p for _, p in parts]) if parts else np.array([], dtype=int)
    return _summarise_ranks(rank, len(ids), ranks, true_col, pred)


def leave_one_out_identification(feats: np.ndarray, ids: np.ndarray, ranks=(1, 5), block: int = 1024) -> dict:
    """All-vs-all identification (UERC style): every image is a probe against all OTHER images.
    Identities with a single image cannot be probes; they stay in the gallery as distractors."""
    from .qc import l2n
    f = l2n(feats)
    ids = np.asarray(ids)
    uniq, order, starts = _id_layout(ids)
    counts = np.diff(np.append(starts, len(ids)))
    inv = np.searchsorted(uniq, ids)
    probes = np.nonzero(counts[inv] >= 2)[0]
    ranks_all, preds_all = [], []
    for s in range(0, len(probes), block):
        p = probes[s:s + block]
        sim = f[p] @ f.T
        sim[np.arange(len(p)), p] = -np.inf  # a probe never matches itself
        r, pr = _rank_block(sim, order, starts, inv[p])
        ranks_all.append(r)
        preds_all.append(pr)
    rank = np.concatenate(ranks_all) if ranks_all else np.array([], dtype=int)
    pred = np.concatenate(preds_all) if preds_all else np.array([], dtype=int)
    out = _summarise_ranks(rank, len(uniq), ranks, inv[probes], pred)
    out["n_probes"] = int(len(probes))
    return out


# ----------------------------------------------------------------------------- verification
def verification(genuine: np.ndarray, impostor: np.ndarray, far: float = 0.01) -> dict:
    """EER and TAR at a given FAR from genuine / impostor similarity scores."""
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if len(genuine) == 0 or len(impostor) == 0:
        return {"eer": float("nan"), "tar_at_far": float("nan")}
    thr = np.unique(np.concatenate([genuine, impostor]))
    thr = np.concatenate([thr, [np.inf]])
    g = np.sort(genuine)
    im = np.sort(impostor)
    # accept if score >= t
    frr = np.searchsorted(g, thr, side="left") / len(g)
    far_curve = (len(im) - np.searchsorted(im, thr, side="left")) / len(im)   # exact (no 1 - x rounding)
    i = int(np.argmin(np.abs(frr - far_curve)))
    eer = (frr[i] + far_curve[i]) / 2
    ok = far_curve <= far
    tar = float((1 - frr[ok]).max()) if ok.any() else 0.0
    return {"eer": 100 * float(eer), "tar_at_far": 100 * tar}


def pair_scores(f: np.ndarray, a: np.ndarray, b: np.ndarray, block: int = 100_000) -> np.ndarray:
    """Cosine scores f[a[k]] . f[b[k]] (f L2-normalised), computed in blocks: 1M pairs of 4096-d VGG
    features would otherwise need tens of GB."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), block):
        out[s:s + block] = np.einsum("ij,ij->i", f[a[s:s + block]], f[b[s:s + block]])
    return out


def sample_pairs(ids: np.ndarray, max_impostor: int, rng: np.random.Generator):
    """All genuine pairs (capped at max_impostor) and random impostor pairs (i<j)."""
    ids = np.asarray(ids)
    n = len(ids)
    gi, gj = [], []
    for u in np.unique(ids):
        idx = np.nonzero(ids == u)[0]
        if len(idx) >= 2:
            a, b = np.triu_indices(len(idx), k=1)
            gi.append(idx[a]); gj.append(idx[b])
    gi = np.concatenate(gi) if gi else np.array([], dtype=int)
    gj = np.concatenate(gj) if gj else np.array([], dtype=int)
    if len(gi) > max_impostor:
        sel = rng.choice(len(gi), size=max_impostor, replace=False)
        gi, gj = gi[sel], gj[sel]
    total_imp = n * (n - 1) // 2 - len(gi)
    m = int(min(max_impostor, max(total_imp, 0)))
    ii = rng.integers(0, n, size=m * 3)
    jj = rng.integers(0, n, size=m * 3)
    keep = (ii != jj) & (ids[ii] != ids[jj])
    ii, jj = ii[keep][:m], jj[keep][:m]
    return (gi, gj), (ii, jj)


# ----------------------------------------------------------------------------- summaries
def mean_std(values) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))],
                   dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def mean_ci95(values) -> tuple[float, float]:
    """Mean and half-width of the 95% confidence interval (t distribution)."""
    from scipy import stats
    v = np.asarray([x for x in values if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), 0.0
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return float(v.mean()), float(half)
