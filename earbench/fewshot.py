"""Few-shot classifiers on frozen features, and K-means clustering."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .qc import l2n


def prototype_predict(sup_x: np.ndarray, sup_y: np.ndarray, query_x: np.ndarray, block: int = 4096):
    """Nearest class mean with cosine similarity. Returns (pred, scores[n_query, n_class], classes)."""
    classes = np.unique(sup_y)
    s = l2n(sup_x)
    protos = np.stack([s[sup_y == c].mean(0) for c in classes])
    protos = l2n(protos)
    q = l2n(query_x)
    scores = np.concatenate([q[i:i + block] @ protos.T for i in range(0, len(q), block)]) if len(q) else \
        np.zeros((0, len(classes)), dtype=np.float32)
    return classes[scores.argmax(1)] if len(q) else np.array([], dtype=sup_y.dtype), scores, classes


def linear_probe(sup_x, sup_y, val_x, val_y, query_x, C_grid=(0.01, 0.1, 1.0, 10.0), max_iter: int = 1000):
    """Logistic regression on L2-normalised features. C is chosen on the validation images, then the
    model is refit on the support set only (so the classifier sees exactly K images per class)."""
    sx, qx = l2n(sup_x), l2n(query_x)
    classes = np.unique(sup_y)
    if len(classes) < 2:
        raise ValueError("linear probe needs at least 2 classes")
    best_C, best_acc = C_grid[0], -1.0
    if val_x is not None and len(val_x):
        vx = l2n(val_x)
        for C in C_grid:
            clf = _fit(sx, sup_y, C, max_iter)
            acc = float((clf.predict(vx) == val_y).mean())
            if acc > best_acc + 1e-12:
                best_acc, best_C = acc, C
    clf = _fit(sx, sup_y, best_C, max_iter)
    proba = clf.predict_proba(qx) if len(qx) else np.zeros((0, len(classes)))
    pred = clf.classes_[proba.argmax(1)] if len(qx) else np.array([], dtype=sup_y.dtype)
    return pred, proba, clf.classes_, best_C


def _fit(x, y, C, max_iter):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(x, y)
    return clf


def kmeans_labels(feats: np.ndarray, k: int, seed: int, minibatch_above: int = 20000, max_iter: int = 300,
                  minibatch_size: int = 4096, minibatch_n_init: int = 3) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > len(feats):
        raise ValueError(f"k={k} clusters but only {len(feats)} images")
    x = l2n(feats)
    if len(x) > minibatch_above:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=minibatch_n_init, batch_size=minibatch_size,
                             max_iter=max_iter)
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init=1, max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        return km.fit_predict(x)
