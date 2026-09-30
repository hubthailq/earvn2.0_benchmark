"""Shared plumbing for the numbered scripts: argument parsing, loading the metadata tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import add_common_args, config_from_args, dump_config
from .paths import Paths
from .utils import get_logger, read_csv


def setup(description: str, extra_args=None):
    """Parse args, load config, create logger. Returns (args, cfg, paths, logger)."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    if extra_args:
        extra_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    script = Path(sys.argv[0]).stem
    logger = get_logger("earbench", paths.root / "logs" / f"{script}.log")
    logger.info(f"=== {script} === output root: {paths.root}")
    dump_config(cfg, paths.root / "logs" / f"{script}.config.json")
    return args, cfg, paths, logger


def load_removed(paths: Paths) -> pd.DataFrame:
    if paths.removed_csv.exists():
        df = read_csv(paths.removed_csv)
        if "image_id" not in df.columns:
            raise ValueError(f"{paths.removed_csv} needs an 'image_id' column")
        return df
    return pd.DataFrame(columns=["image_id", "reason"])


def load_images(paths: Paths, with_groups: bool = True, require_groups: bool = False) -> pd.DataFrame:
    """Usable images (decodable, not removed by QC) with subject, gender, orig_split and near-duplicate group.
    Index is 0..n-1 in a stable order."""
    images = read_csv(paths.images_csv, dtype={"subject": str, "orig_split": str})
    images["orig_split"] = images["orig_split"].fillna("")
    subjects = read_csv(paths.subjects_csv, dtype={"subject": str})
    removed = load_removed(paths)
    df = images[images.status == "ok"]
    if len(removed):
        df = df[~df.image_id.isin(set(removed.image_id))]
    df = df.merge(subjects[["subject", "gender", "order"]], on="subject", how="left")
    if df.gender.isna().any():
        raise ValueError("some images have no gender - re-run scripts/01_scan_dataset.py")
    df = df.sort_values(["order", "image_id"], kind="stable").reset_index(drop=True)
    if with_groups:
        if paths.dup_groups_csv.exists():
            g = read_csv(paths.dup_groups_csv)[["image_id", "group"]]
            df = df.merge(g, on="image_id", how="left")
            if df.group.isna().any():
                raise ValueError(f"{int(df.group.isna().sum())} images missing from {paths.dup_groups_csv}; "
                                 f"re-run scripts/02_qc_duplicates.py")
            df["group"] = df["group"].astype(str)
        elif require_groups:
            raise FileNotFoundError(f"{paths.dup_groups_csv} not found - run scripts/02_qc_duplicates.py first")
        else:
            df["group"] = df["image_id"]  # every image is its own group
    return df


def load_subjects(paths: Paths, images: pd.DataFrame | None = None) -> pd.DataFrame:
    subjects = read_csv(paths.subjects_csv, dtype={"subject": str})
    if images is not None:
        subjects = subjects[subjects.subject.isin(set(images.subject))]
    return subjects.reset_index(drop=True)


def load_s1(paths: Paths, images: pd.DataFrame) -> pd.DataFrame:
    s1_all = read_csv(paths.s1_csv)
    s1 = s1_all[["image_id", "split"]]
    df = images.merge(s1, on="image_id", how="left")
    fix = "QC changed after the split was made: re-run scripts/04_make_splits.py (and the experiments after it)"
    if df.split.isna().any():
        raise ValueError(f"{int(df.split.isna().sum())} usable images are not in {paths.s1_csv}. {fix}")
    extra = set(s1.image_id) - set(images.image_id)
    if extra:
        raise ValueError(f"{len(extra)} images of {paths.s1_csv} are no longer usable (e.g. {sorted(extra)[0]}; "
                         f"removed by scripts/03_qc_labels.py --apply?). {fix}")
    if "group" in s1_all.columns and "group" in images.columns:
        g = images[["image_id", "group"]].merge(s1_all[["image_id", "group"]], on="image_id", suffixes=("", "_s1"))
        changed = int((g.group.astype(str) != g.group_s1.astype(str)).sum())
        if changed:
            raise ValueError(f"near-duplicate groups of {changed} images differ from those used for the split "
                             f"(scripts/02_qc_duplicates.py was re-run with other settings). {fix}")
    return df


def group_codes(groups: pd.Series) -> np.ndarray:
    return pd.factorize(groups.astype(str))[0]
