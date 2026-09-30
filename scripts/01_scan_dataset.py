#!/usr/bin/env python
"""01 - Scan EarVN2.0: list subjects and images, read sizes, find broken files, assign gender,
compute dataset statistics (incl. height/width ratio used by E0).

Outputs (outputs/metadata/):
  images.csv          one row per file: image_id, subject, orig_split, width, height, status, md5 ...
  subjects.csv        subject, order, gender, gender_source, n_images
  scan_problems.csv   broken / skipped files and layout problems
  dataset_stats.json + dataset_stats.md
  preprocess_preview.jpg   a few images: stretched vs letterboxed
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from earbench.common import setup  # noqa: E402
from earbench.preprocess import save_preview  # noqa: E402
from earbench.scan import assign_gender, dataset_statistics, scan_dataset, suggest_nonsquare  # noqa: E402
from earbench.utils import atomic_write_csv, load_rgb  # noqa: E402


def extra(p):
    p.add_argument("--workers", type=int, default=8, help="threads for reading images")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    log.info(f"dataset folder (one sub-folder per subject): {cfg.paths.dataset_root}")
    images, subjects, problems = scan_dataset(cfg, log, workers=args.workers)
    subj = assign_gender(cfg, subjects, log)
    counts = images[images.status == "ok"].groupby("subject").size()
    subj["n_images"] = subj.subject.map(counts).fillna(0).astype(int)
    few = subj[subj.n_images < int(cfg.splits.s1.min_images)]
    if len(few):
        log.warning(f"{len(few)} subjects have < {cfg.splits.s1.min_images} readable images "
                    f"(they will be excluded from S1): {few.subject.tolist()[:10]}")

    atomic_write_csv(images, paths.images_csv)
    atomic_write_csv(subj, paths.subjects_csv)
    atomic_write_csv(problems, paths.scan_problems_csv)

    stats = dataset_statistics(images, subj, cfg.resolution_bins)
    h, w = suggest_nonsquare(stats["aspect_hw"].get("median", np.nan), int(cfg.preprocess.size))
    stats["suggested_nonsquare_hw"] = [h, w]
    paths.stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Thống kê EarVN2.0 (tự sinh)", "", "| Thông tin | Giá trị |", "| --- | --- |",
          f"| Số subject | {stats['n_subjects']} ({stats['n_subjects_by_gender']}) |",
          f"| Tổng số ảnh đọc được | {stats['n_images_ok']} (không đọc được: {stats['n_images_unreadable']}) |",
          f"| Ảnh theo giới tính | {stats['n_images_by_gender']} |",
          f"| Ảnh mỗi subject (min / trung vị / TB / max) | {stats['images_per_subject']['min']} / "
          f"{stats['images_per_subject']['median']} / {stats['images_per_subject']['mean']} / "
          f"{stats['images_per_subject']['max']} |",
          f"| Cạnh ngắn (min / trung vị / max, px) | {stats['min_side'].get('min')} / {stats['min_side'].get('median')} / "
          f"{stats['min_side'].get('max')} |",
          f"| Tỉ lệ cao/rộng (p25 / trung vị / p75) | {stats['aspect_hw'].get('p25')} / "
          f"{stats['aspect_hw'].get('median')} / {stats['aspect_hw'].get('p75')} |",
          f"| Phân bố cạnh ngắn | {stats['short_side_bins']} |",
          f"| Kích thước đề xuất cho E0 'không vuông' | {h} × {w} |"]
    paths.stats_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("\n" + "\n".join(md))

    ok = images[images.status == "ok"]
    rng = np.random.default_rng(cfg.seed)
    sample = ok.iloc[rng.choice(len(ok), size=min(6, len(ok)), replace=False)]
    save_preview([load_rgb(Path(cfg.paths.dataset_root) / i) for i in sample.image_id], cfg.preprocess,
                 paths.metadata / "preprocess_preview.jpg")
    log.info(f"Set preprocess.nonsquare_hw to [{h}, {w}] in configs/default.yaml before running E0.")
    log.info(f"Problems: {len(problems)} (see {paths.scan_problems_csv})")


if __name__ == "__main__":
    main()
