#!/usr/bin/env python
"""15 - Accuracy by image resolution (short side of the ORIGINAL image, bins in resolution_bins).

Uses the saved E1 test predictions (outputs/runs/E1/<model>/S1_seed*/predictions.csv), so nothing is
re-trained. By default analyses the 5 best models of E1.  Output: outputs/results/per_run/resolution.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from earbench.common import setup  # noqa: E402
from earbench.models import select_models  # noqa: E402
from earbench.utils import atomic_write_csv, read_csv  # noqa: E402


def extra(p):
    p.add_argument("--models", default=None, help="comma-separated keys (default: --top best of E1)")
    p.add_argument("--top", type=int, default=5)


def bin_labels(bins):
    return [f"[{int(a)},{int(b)})" if b < 10000 else f"≥{int(a)}" for a, b in zip(bins[:-1], bins[1:])]


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    e1_path = paths.results_csv("E1")
    if not e1_path.exists():
        raise FileNotFoundError("E1 results not found: run scripts/08_train_supervised.py --exp E1 first")
    e1 = pd.read_csv(e1_path)
    if args.models:
        keys = [m.key for m in select_models(args.models)]
    else:
        keys = e1.groupby("model").test_top1.mean().sort_values(ascending=False).head(args.top).index.tolist()
    sizes = read_csv(paths.images_csv)[["image_id", "min_side"]]
    bins = [float(b) for b in cfg.resolution_bins]
    labels = bin_labels(bins)
    rows = []
    for key in keys:
        run_root = paths.root / "runs" / "E1" / key
        files = sorted(run_root.glob("S1_seed*/predictions.csv"))
        if not files:
            log.warning(f"no E1 predictions for {key}")
            continue
        for f in files:
            p = pd.read_csv(f, dtype={"image_id": str}).merge(sizes, on="image_id", how="left")
            p["bin"] = pd.cut(p.min_side, bins=bins, right=False, labels=labels)
            p["correct"] = p.pred == p.label
            for b, g in p.groupby("bin", observed=False):
                rows.append({"model": key, "seed": f.parent.name, "bin": str(b), "n_images": len(g),
                             "top1": 100 * g.correct.mean() if len(g) else float("nan")})
    out = pd.DataFrame(rows)
    atomic_write_csv(out, paths.results_csv("resolution"))
    if len(out):
        log.info("\n" + out.groupby(["model", "bin"], observed=False).top1.mean().round(2).unstack().to_string())


if __name__ == "__main__":
    main()
