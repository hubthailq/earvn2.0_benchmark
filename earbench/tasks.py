"""Build the train/val/test tables for each supervised setting."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import encode_labels
from .splits import fold_roles
from .utils import read_csv


def gender_label(series: pd.Series, cfg) -> np.ndarray:
    """Class 1 = female, class 0 = male (so 'prob_1' is the probability of female)."""
    return (series.astype(str) == str(cfg.gender.female_label)).astype(np.int64).to_numpy()


def s1_frames(img_s1: pd.DataFrame, task: str, cfg):
    """S1 train/val/test. identity labels = subject index; gender labels = 0/1."""
    df = img_s1[img_s1.split.isin(["train", "val", "test"])].copy()
    if task == "identity":
        df["label"], classes = encode_labels(df.subject)
    else:
        df["label"] = gender_label(df.gender, cfg)
        classes = [cfg.gender.male_label, cfg.gender.female_label]
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    return parts["train"], parts["val"], parts["test"], classes


def s2_frames(img: pd.DataFrame, s2: pd.DataFrame, fold: int, cfg):
    n_folds = int(cfg.splits.s2.n_folds)
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in 0..{n_folds - 1}")
    roles = fold_roles(s2.set_index("subject").fold, fold, n_folds)
    df = img.copy()
    df["role"] = df.subject.map(roles)
    if df.role.isna().any():
        raise ValueError("some images belong to subjects missing from the S2 file; re-run script 04")
    df["label"] = gender_label(df.gender, cfg)
    parts = {r: df[df.role == r].reset_index(drop=True) for r in ("train", "val", "test")}
    return parts["train"], parts["val"], parts["test"], [cfg.gender.male_label, cfg.gender.female_label]


def s3_frames(img_s1: pd.DataFrame, s3: pd.DataFrame):
    """Fine-tuning data for E7/E8: identity classification over the S3 'train' subjects only.
    Uses their S1 val images for early stopping and all their other images for training."""
    train_subjects = set(s3[s3.role == "train"].subject)
    df = img_s1[img_s1.subject.isin(train_subjects) & img_s1.split.isin(["train", "val", "test"])].copy()
    df["label"], classes = encode_labels(df.subject)
    train = df[df.split != "val"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    return train, val, classes


def chosen_lr(paths, family: str, cfg, logger=None) -> float:
    """Learning rate of a family: fixed in the config > LR-sweep result > default (train.learning_rate)."""
    lr_cfg = cfg.train.learning_rate
    fixed = lr_cfg.get("by_family") or {}
    if family in fixed:
        return float(fixed[family])
    if paths.lr_json.exists():
        table = json.loads(paths.lr_json.read_text(encoding="utf-8"))
        if family in table:
            return float(table[family])
    if logger:
        logger.warning(f"no fixed LR and no LR-sweep result for family '{family}': "
                       f"using train.learning_rate.default={lr_cfg.default}")
    return float(lr_cfg.default)


def load_s2(paths) -> pd.DataFrame:
    return read_csv(paths.s2_csv, dtype={"fold": int})


def load_s3(paths) -> pd.DataFrame:
    return read_csv(paths.s3_csv)
