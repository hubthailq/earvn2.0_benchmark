#!/usr/bin/env python
"""08 - E1 / E2: supervised fine-tuning.

  E1 identity : S1 (45/15/rest), one run per seed in train.seeds
  E2 gender   : S2 (subject-disjoint, 5 folds, main result) and S1 (same subjects in train and test,
                secondary result), one run per fold / seed

Examples
  python scripts/08_train_supervised.py --exp E1 --models basic
  python scripts/08_train_supervised.py --exp E2 --models all --split S2
  python scripts/08_train_supervised.py --exp E1 --models resnet50 --seeds 0

Finished runs are skipped (see outputs/results/per_run/E1.csv, E2.csv); use --overwrite to redo them.
Each run writes log.csv, predictions.csv and metrics.json under outputs/runs/<exp>/<model>/<run>/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earbench.common import load_images, load_s1, setup  # noqa: E402
from earbench.models import resolve_device, select_models  # noqa: E402
from earbench.tasks import chosen_lr, load_s2, s1_frames, s2_frames  # noqa: E402
from earbench.train import run_fingerprint, train_or_diverge  # noqa: E402
from earbench.utils import ResultStore  # noqa: E402


def extra(p):
    p.add_argument("--exp", required=True, choices=["E1", "E2"])
    p.add_argument("--models", default="basic", help="all | basic | comma-separated keys")
    p.add_argument("--seeds", default=None, help="comma-separated, default train.seeds")
    p.add_argument("--split", default="both", choices=["S1", "S2", "both"], help="E2 only")
    p.add_argument("--folds", default=None, help="E2-S2 folds, comma-separated (default all)")
    p.add_argument("--overwrite", action="store_true")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [int(s) for s in cfg.train.seeds]
    n_folds = int(cfg.splits.s2.n_folds)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(n_folds))
    img_all = load_images(paths)
    img = load_s1(paths, img_all)
    device = resolve_device(cfg.device)
    root = cfg.paths.dataset_root
    specs = select_models(args.models)
    save_ck = bool(cfg.train.save_checkpoint)

    jobs = []  # (split, run_id, seed, frames)
    if args.exp == "E1":
        fr = s1_frames(img, "identity", cfg)
        jobs += [("S1", s, s, fr) for s in seeds]
        task = "identity"
    else:
        task = "gender"
        if args.split in ("S1", "both"):
            fr = s1_frames(img, "gender", cfg)
            jobs += [("S1", s, s, fr) for s in seeds]
        if args.split in ("S2", "both"):
            s2 = load_s2(paths)
            jobs += [("S2", f, seeds[0], s2_frames(img_all, s2, f, cfg)) for f in folds]

    store = ResultStore(paths.results_csv(args.exp), ["model", "split", "run"])
    total = len(specs) * len(jobs)
    done = 0
    for spec in specs:
        lr = chosen_lr(paths, spec.family, cfg, log)
        for split, run_id, seed, (train, val, test, classes) in jobs:
            done += 1
            key = {"model": spec.key, "split": split, "run": run_id}
            fp = run_fingerprint(cfg, spec, task, None, None, train, val, test, seed=seed)
            if store.should_skip(key, args.overwrite, log, lr=lr, preprocess=cfg.preprocess.mode, fingerprint=fp):
                continue
            tag = f"{split}_{'seed' if split == 'S1' else 'fold'}{run_id}"
            log.info(f"[{done}/{total}] {args.exp} {spec.key} {tag} lr={lr} "
                     f"(train {len(train)}, val {len(val)}, test {len(test)})")
            r = train_or_diverge(spec, cfg, task, root, train, val, test, len(classes), lr, seed,
                                 paths.run_dir(args.exp, spec.key, tag), device,
                                 save_checkpoint=save_ck, class_names=classes, logger=log)
            store.upsert({**key, "seed": seed, "lr": lr, "preprocess": cfg.preprocess.mode, "fingerprint": fp, **r})


if __name__ == "__main__":
    main()
