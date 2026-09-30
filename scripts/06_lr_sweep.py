#!/usr/bin/env python
"""06 - Learning-rate sweep per model family.

For the representative (smallest) model of each family, fine-tune identity on S1 for
train.learning_rate.sweep_epochs epochs with every LR in train.learning_rate.sweep_grid, and keep the LR with
the best S1-val Top-1. Families with a fixed LR in train.learning_rate.by_family are skipped.
The chosen LR is then used for every model of that family, for both identity and gender.

Outputs: outputs/results/per_run/lr_sweep.csv, outputs/results/per_run/chosen_lr.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from earbench.common import load_images, load_s1, setup  # noqa: E402
from earbench.models import resolve_device, select_models  # noqa: E402
from earbench.tasks import s1_frames  # noqa: E402
from earbench.train import run_fingerprint, train_and_evaluate  # noqa: E402
from earbench.utils import ResultStore  # noqa: E402


def extra(p):
    p.add_argument("--models", default="lr_rep", help="lr_rep (default) | comma-separated keys")
    p.add_argument("--overwrite", action="store_true")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    img = load_s1(paths, load_images(paths))
    train, val, _, classes = s1_frames(img, "identity", cfg)
    device = resolve_device(cfg.device)
    store = ResultStore(paths.results_csv("lr_sweep"), ["model", "lr"])
    for spec in select_models(args.models):
        if spec.family in (cfg.train.learning_rate.get("by_family") or {}):
            log.info(f"skip {spec.key}: LR of family '{spec.family}' is fixed in the config")
            continue
        ep = int(cfg.train.learning_rate.sweep_epochs)
        fp = run_fingerprint(cfg, spec, "identity", None, ep, train, val, seed=0)
        for lr in cfg.train.learning_rate.sweep_grid:
            key = {"model": spec.key, "lr": float(lr)}
            if store.should_skip(key, args.overwrite, log, fingerprint=fp):
                log.info(f"skip {spec.key} lr={lr} (done)")
                continue
            log.info(f"LR sweep {spec.key} lr={lr}")
            run_dir = paths.run_dir("lr_sweep", spec.key, f"lr{lr:g}")
            try:
                r = train_and_evaluate(spec, cfg, "identity", cfg.paths.dataset_root, train, val, None,
                                       len(classes), float(lr), 0, run_dir, device,
                                       epochs=ep, logger=log)
                store.upsert({**key, "family": spec.family, "val_top1": r["val_top1"],
                              "best_epoch": r["best_epoch"], "status": "ok", "fingerprint": fp})
            except FloatingPointError as exc:
                log.warning(f"  diverged: {exc}")
                store.upsert({**key, "family": spec.family, "val_top1": np.nan, "best_epoch": -1,
                              "status": "diverged", "fingerprint": fp})

    res = store.load()
    res = res[pd.to_numeric(res.val_top1, errors="coerce").notna()]
    grid = [float(x) for x in cfg.train.learning_rate.sweep_grid]
    res = res[res.lr.astype(float).apply(lambda v: any(abs(v - g) <= 1e-12 for g in grid))]  # current grid only
    chosen = {}
    for fam, g in res.groupby("family"):
        best = g.loc[pd.to_numeric(g.val_top1).idxmax()]
        chosen[fam] = float(best.lr)
    paths.lr_json.write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"chosen LR per family: {chosen}")
    # for the public release: freeze the chosen values in the config so others skip the sweep
    fixed = {**chosen, **(cfg.train.learning_rate.get("by_family") or {})}
    snippet = "train:\n  learning_rate:\n    by_family:\n" + "".join(
        f'      "{fam}": {lr:g}\n' for fam, lr in fixed.items())
    (paths.per_run / "chosen_lr_for_config.yaml").write_text(snippet, encoding="utf-8")
    log.info("To publish reproducible results, copy these values into configs/default.yaml "
             f"(train.learning_rate.by_family):\n{snippet}")


if __name__ == "__main__":
    main()
