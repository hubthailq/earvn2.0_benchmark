#!/usr/bin/env python
"""09 - Fine-tune the models used by E7 / E8 (new identities).

Identity classification over the S3 'train' subjects only (the 'novel' subjects are never seen).
The best checkpoint is always saved (outputs/runs/S3/<model>/seed<k>/best.pt) because script 10
extracts features with it.  Output: outputs/results/per_run/S3_training.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earbench.common import load_images, load_s1, setup  # noqa: E402
from earbench.models import resolve_device, select_models  # noqa: E402
from earbench.tasks import chosen_lr, load_s3, s3_frames  # noqa: E402
from earbench.train import run_fingerprint, train_or_diverge  # noqa: E402
from earbench.utils import ResultStore  # noqa: E402


def extra(p):
    p.add_argument("--models", default="all")
    p.add_argument("--seed", type=int, default=None, help="default: first of train.seeds")
    p.add_argument("--overwrite", action="store_true")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    seed = int(args.seed if args.seed is not None else cfg.train.seeds[0])
    img = load_s1(paths, load_images(paths))
    train, val, classes = s3_frames(img, load_s3(paths))
    log.info(f"S3 training set: {len(classes)} subjects, {len(train)} train / {len(val)} val images")
    device = resolve_device(cfg.device)
    store = ResultStore(paths.results_csv("S3_training"), ["model", "seed"])
    for spec in select_models(args.models):
        run_dir = paths.run_dir("S3", spec.key, f"seed{seed}")
        key = {"model": spec.key, "seed": seed}
        lr = chosen_lr(paths, spec.family, cfg, log)
        fp = run_fingerprint(cfg, spec, "identity", None, None, train, val, seed=seed)
        done = store.should_skip(key, args.overwrite, log, lr=lr, preprocess=cfg.preprocess.mode, fingerprint=fp)
        old = store.get(key) or {}
        if done and ((run_dir / "best.pt").exists() or old.get("status") == "diverged"):
            continue
        log.info(f"S3 fine-tune {spec.key} lr={lr}")
        (run_dir / "best.pt").unlink(missing_ok=True)   # never keep a checkpoint of an older run
        r = train_or_diverge(spec, cfg, "identity", cfg.paths.dataset_root, train, val, None, len(classes), lr,
                             seed, run_dir, device, save_checkpoint=True, class_names=classes, logger=log)
        store.upsert({**key, "lr": lr, "preprocess": cfg.preprocess.mode, "fingerprint": fp, **r})


if __name__ == "__main__":
    main()
