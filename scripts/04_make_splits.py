#!/usr/bin/env python
"""04 - Build the splits.

S1 identity (E0, E1, E2-S1, E5 bins): exactly splits.s1.train / val images per subject, the rest is test.
   - keeps the ORIGINAL on-disk split if script 02 decided 'keep_original' (leakage <= threshold),
   - otherwise re-splits each subject keeping near-duplicate groups on one side.
   Subjects with fewer than splits.s1.min_images usable images are marked 'excluded'.
S2 gender (E2, E4): subject-disjoint folds stratified by gender; fold k = test, fold k+1 = val.
S3 new identities (E7, E8): subjects split into 'train' (fine-tune) and 'novel' (enrol + recognise).

Outputs (outputs/splits/): s1_identity.csv, s2_gender_folds.csv, s3_new_identity.csv,
split_warnings.csv, splits_summary.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from earbench.common import group_codes, load_images, load_subjects, setup  # noqa: E402
from earbench.qc import leakage_report  # noqa: E402
from earbench.splits import make_s1, make_s2, make_s3  # noqa: E402
from earbench.tables import to_markdown  # noqa: E402
from earbench.utils import atomic_write_csv  # noqa: E402


def extra(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--force-resplit", action="store_true", help="ignore the original split, always re-split")
    g.add_argument("--keep-original", action="store_true", help="keep the original split even if leakage is high")
    p.add_argument("--no-groups", action="store_true", help="run without script 02 (every image its own group)")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    img = load_images(paths, with_groups=True, require_groups=not args.no_groups)
    codes = group_codes(img.group)

    decision = "resplit"
    if paths.leakage_json.exists():
        decision = json.loads(paths.leakage_json.read_text()).get("decision_s1", "resplit")
    if args.force_resplit:
        decision = "resplit"
    if args.keep_original:
        decision = "keep_original"
    if decision == "keep_original" and not (img.orig_split != "").any():
        log.warning("no original split on disk: re-splitting")
        decision = "resplit"
    log.info(f"S1 decision: {decision}")

    rng = np.random.default_rng(cfg.seed)
    s1, warns = make_s1(img, codes, cfg.splits.s1, rng, keep_original=(decision == "keep_original"))
    s1["group"] = img.group.to_numpy()
    atomic_write_csv(s1, paths.s1_csv)
    lk = leakage_report(s1.split, codes, img.subject.to_numpy())
    counts = s1.split.value_counts().to_dict()
    eligible = sorted(set(s1[s1.split != "excluded"].subject))
    log.info(f"S1: {len(eligible)} subjects, images {counts}; test leakage after split = "
             f"{100 * lk['test_leakage_rate']:.2f}%")
    if lk["test_leakage_rate"] > float(cfg.qc.leakage_threshold):
        log.error(f"Leakage after the split is still {100 * lk['test_leakage_rate']:.1f}% (> "
                  f"{100 * cfg.qc.leakage_threshold:.1f}%). Near-duplicate groups are too large to keep on one side "
                  f"(see split_warnings.csv). Usually the QC thresholds are too loose: check qc/near_duplicate_pairs.csv, "
                  f"tighten qc.phash_hamming_max / qc.dino_cosine_min (or qc.pair_rule=and) and re-run scripts 02-04.")

    subjects = load_subjects(paths, img)
    s2 = make_s2(subjects[["subject", "gender"]], int(cfg.splits.s2.n_folds), np.random.default_rng(cfg.seed + 1))
    atomic_write_csv(s2, paths.s2_csv)
    fold_tab = s2.groupby(["fold", "gender"]).size().unstack(fill_value=0)
    log.info(f"S2 folds (subjects per fold x gender):\n{fold_tab}")

    s3 = make_s3(subjects[subjects.subject.isin(eligible)][["subject", "gender"]],
                 float(cfg.splits.s3.novel_fraction), np.random.default_rng(cfg.seed + 2))
    atomic_write_csv(s3, paths.s3_csv)
    log.info(f"S3: {int((s3.role == 'train').sum())} train subjects, {int((s3.role == 'novel').sum())} novel subjects")

    atomic_write_csv(warns, paths.split_warnings_csv)
    if len(warns):
        log.warning(f"{len(warns)} split warnings -> {paths.split_warnings_csv}")

    md = ["## Splits", "", f"- Quyết định S1: **{decision}**",
          f"- S1: {len(eligible)} subject; ảnh train/val/test/loại = {counts.get('train', 0)} / {counts.get('val', 0)} / "
          f"{counts.get('test', 0)} / {counts.get('excluded', 0)}",
          f"- Rò rỉ ảnh gần trùng test→train sau khi chia: {lk['n_test_with_train_duplicate']} ảnh "
          f"({100 * lk['test_leakage_rate']:.2f}%)",
          f"- S2: {cfg.splits.s2.n_folds} fold theo người", "", to_markdown(fold_tab.reset_index()),
          "", f"- S3: {int((s3.role == 'train').sum())} người train, {int((s3.role == 'novel').sum())} người mới",
          f"- Cảnh báo khi chia: {len(warns)} (xem split_warnings.csv)"]
    (paths.splits / "splits_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
