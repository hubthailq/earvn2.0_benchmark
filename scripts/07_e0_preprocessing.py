#!/usr/bin/env python
"""07 - E0: preprocessing ablation (stretch vs letterbox vs non-square input).

Identity on S1, the 5 basic models, one seed. 'nonsquare' uses preprocess.nonsquare_hw and is skipped
for models that need square inputs (ViT, Swin, DINOv2, CLIP).  Output: outputs/results/per_run/E0.csv

Decision rule (from the plan): keep letterbox for the whole benchmark unless it is clearly worse than stretch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earbench.common import load_images, load_s1, setup  # noqa: E402
from earbench.models import resolve_device, select_models  # noqa: E402
from earbench.tasks import chosen_lr, s1_frames  # noqa: E402
from earbench.train import run_fingerprint, train_or_diverge  # noqa: E402
from earbench.utils import ResultStore  # noqa: E402

MODES = ("stretch", "letterbox", "nonsquare")


def extra(p):
    p.add_argument("--models", default="basic")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--seed", type=int, default=None, help="default: first of train.seeds")
    p.add_argument("--overwrite", action="store_true")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise ValueError(f"unknown modes {bad}; choose from {MODES}")
    seed = int(args.seed if args.seed is not None else cfg.train.seeds[0])
    img = load_s1(paths, load_images(paths))
    train, val, test, classes = s1_frames(img, "identity", cfg)
    device = resolve_device(cfg.device)
    store = ResultStore(paths.results_csv("E0"), ["model", "mode", "seed"])
    for spec in select_models(args.models):
        for mode in modes:
            if mode == "nonsquare" and spec.square_only:
                log.info(f"skip {spec.key} nonsquare (model needs square input)")
                continue
            key = {"model": spec.key, "mode": mode, "seed": seed}
            lr = chosen_lr(paths, spec.family, cfg, log)
            fp = run_fingerprint(cfg, spec, "identity", mode, None, train, val, test, seed=seed)
            if store.should_skip(key, args.overwrite, log, lr=lr, fingerprint=fp):
                continue
            log.info(f"E0 {spec.key} mode={mode} lr={lr}")
            r = train_or_diverge(spec, cfg, "identity", cfg.paths.dataset_root, train, val, test, len(classes),
                                 lr, seed, paths.run_dir("E0", spec.key, f"{mode}_seed{seed}"), device,
                                 mode=mode, logger=log)
            store.upsert({**key, "lr": lr, "fingerprint": fp, **r})


if __name__ == "__main__":
    main()
