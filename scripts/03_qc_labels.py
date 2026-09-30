#!/usr/bin/env python
"""03 - Q2: label checks.

Step A (default): write outputs/qc/outliers_for_review.csv listing images to look at by hand:
   * images far from their subject centre in DINOv2 space (z-score < qc.outlier_z) -> wrong person / not an ear
   * near-duplicates shared by two DIFFERENT subjects (from script 02)            -> label error
   * tiny images (min side < dataset.min_side_warn)                              -> check they are ears
   Open the CSV, put 'y' in the column `remove` for every image that must go, save.

Step B (--apply): read the reviewed CSV and write outputs/qc/removed.csv (image_id, reason).
   Unreadable files are excluded automatically and never need to be listed.
   Afterwards run 04_make_splits.py (the split must be rebuilt after removals).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from earbench.common import load_images, load_removed, setup  # noqa: E402
from earbench.features import load_features  # noqa: E402
from earbench.qc import outlier_scores  # noqa: E402
from earbench.utils import atomic_write_csv, read_csv  # noqa: E402

YES = {"y", "yes", "1", "x", "true", "remove", "có", "co"}


def extra(p):
    p.add_argument("--apply", action="store_true", help="convert the reviewed CSV into removed.csv")


def review(cfg, paths, log):
    img = load_images(paths, with_groups=False)
    rows = []
    fpath = paths.qc / "dino_features.npz"
    if fpath.exists():
        _, feats, _ = load_features(fpath, list(img.image_id))
        sc = outlier_scores(feats, img.subject.to_numpy())
        flag = sc.z < float(cfg.qc.outlier_z)
        for i in np.nonzero(flag.to_numpy())[0]:
            rows.append({"image_id": img.image_id[i], "subject": img.subject[i], "reason": "outlier_in_subject",
                         "z": round(float(sc.z[i]), 2), "cos_to_centroid": round(float(sc.cos_to_centroid[i]), 3),
                         "other_image": ""})
        log.info(f"{int(flag.sum())} outliers (z < {cfg.qc.outlier_z})")
    else:
        log.warning("no DINOv2 features (script 02 was run with --no-dino): outlier check skipped")
    if paths.dup_pairs_csv.exists():
        pairs = read_csv(paths.dup_pairs_csv, dtype={"subject_a": str, "subject_b": str})
        cross = pairs[pairs.same_subject.astype(str).str.lower() == "false"]
        for _, r in cross.iterrows():
            for a, b, s in ((r.image_a, r.image_b, r.subject_a), (r.image_b, r.image_a, r.subject_b)):
                rows.append({"image_id": a, "subject": s, "reason": "duplicate_of_other_subject",
                             "z": np.nan, "cos_to_centroid": np.nan, "other_image": b})
    tiny = img[img.tiny.astype(str).str.lower() == "true"]
    for _, r in tiny.iterrows():
        rows.append({"image_id": r.image_id, "subject": r.subject, "reason": f"tiny_{int(r.width)}x{int(r.height)}",
                     "z": np.nan, "cos_to_centroid": np.nan, "other_image": ""})
    df = pd.DataFrame(rows, columns=["image_id", "subject", "reason", "z", "cos_to_centroid", "other_image"])
    df = df.groupby("image_id", as_index=False).agg({
        "subject": "first", "reason": lambda s: ";".join(sorted(set(s))), "z": "min",
        "cos_to_centroid": "min", "other_image": lambda s: ";".join(sorted({x for x in s if x}))})
    df["remove"] = ""
    df = df.sort_values(["subject", "z"], na_position="last")
    if paths.outliers_csv.exists():
        old = read_csv(paths.outliers_csv)
        keep = dict(zip(old.image_id, old["remove"].fillna("")))
        df["remove"] = df.image_id.map(keep).fillna("")
        log.info("kept the 'remove' decisions already written in the previous review file")
    atomic_write_csv(df, paths.outliers_csv)
    log.info(f"{len(df)} images to review -> {paths.outliers_csv}")
    log.info("Fill column 'remove' with y for images to drop, then run: python scripts/03_qc_labels.py --apply")

    # gender boundary reminder
    subj = read_csv(paths.subjects_csv)
    src = subj.gender_source.iloc[0] if len(subj) else ""
    if str(src).startswith("folder_order"):
        change = np.nonzero(subj.gender.to_numpy()[1:] != subj.gender.to_numpy()[:-1])[0]
        if len(change):
            b = int(change[0])
            log.warning(f"Gender comes from folder order: last '{subj.gender[b]}' = {subj.subject[b]}, "
                        f"first '{subj.gender[b + 1]}' = {subj.subject[b + 1]}. Check ~10 folders around it.")


def apply(cfg, paths, log):
    if not paths.outliers_csv.exists():
        raise FileNotFoundError(f"{paths.outliers_csv} not found: run this script without --apply first")
    rev = read_csv(paths.outliers_csv)
    rev["remove"] = rev["remove"].fillna("").astype(str).str.strip().str.lower()
    new = rev[rev["remove"].isin(YES)][["image_id", "reason"]]
    old = load_removed(paths)
    removed = pd.concat([old, new], ignore_index=True).drop_duplicates("image_id", keep="last")
    known = set(read_csv(paths.images_csv).image_id)
    unknown = sorted(set(removed.image_id) - known)
    if unknown:
        raise ValueError(f"removed list has {len(unknown)} unknown image ids, e.g. {unknown[:3]}")
    atomic_write_csv(removed, paths.removed_csv)
    log.info(f"{len(new)} images marked in the review file; {len(removed)} images in {paths.removed_csv}")
    img = load_images(paths, with_groups=False)
    per = img.groupby("subject").size()
    low = per[per < int(cfg.splits.s1.min_images)]
    if len(low):
        log.warning(f"after removal {len(low)} subjects have < {cfg.splits.s1.min_images} images "
                    f"and will be excluded from S1: {low.index.tolist()[:10]}")
    log.info("Now run: python scripts/04_make_splits.py")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    (apply if args.apply else review)(cfg, paths, log)


if __name__ == "__main__":
    main()
