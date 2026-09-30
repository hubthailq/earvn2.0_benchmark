#!/usr/bin/env python
"""10 - Extract frozen features (the input of E3, E3b, E4, E5/E6, E7, E8).

  --what frozen    pretrained backbones, all usable EarVN2.0 images      -> outputs/features/frozen/<model>.npz
  --what s3ft      S3 fine-tuned models (script 09), 'novel' subjects      -> outputs/features/s3ft/<model>.npz
  --what external  EarVN1.0 / AWE if present (frozen and S3 fine-tuned)    -> outputs/features/{frozen,s3ft}_{earvn1,awe}/

Images are letterboxed exactly as in training (preprocess.mode). Existing files are reused if they
contain all needed images and were made with the same settings; --overwrite forces re-extraction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earbench.common import load_images, load_s1, setup  # noqa: E402
from earbench.device import resolve_amp  # noqa: E402
from earbench.features import extract, load_features, save_features  # noqa: E402
from earbench.models import create_model, resolve_device, select_models  # noqa: E402
from earbench.scan import scan_external  # noqa: E402
from earbench.tasks import load_s3  # noqa: E402
from earbench.train import load_finetuned  # noqa: E402
from earbench.utils import atomic_write_csv  # noqa: E402


def extra(p):
    p.add_argument("--what", default="frozen", choices=["frozen", "s3ft", "external", "all"])
    p.add_argument("--models", default="all")
    p.add_argument("--seed", type=int, default=None, help="S3 checkpoint seed (default: first of train.seeds)")
    p.add_argument("--overwrite", action="store_true")


def cached_ok(path, ids, meta):
    try:
        load_features(path, ids, expect_meta=meta)
        return True
    except (FileNotFoundError, KeyError, ValueError):
        return False


def run(spec, variant, root, ids, model_fn, meta, paths, cfg, device, log, overwrite):
    path = paths.features_file(variant, spec.key)
    if not overwrite and cached_ok(path, ids, meta):
        log.info(f"  {variant:14s} {spec.key:14s} cached ({len(ids)} images)")
        return
    model = model_fn()
    if model is None:
        if path.exists():     # features of an older checkpoint must not outlive it
            path.unlink()
            log.warning(f"  {variant:14s} {spec.key:14s} removed stale features (no current checkpoint)")
        return
    feats = extract(model, root, ids, cfg, device)
    save_features(path, ids, feats, meta)
    log.info(f"  {variant:14s} {spec.key:14s} {feats.shape}")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    device = resolve_device(cfg.device)
    seed = int(args.seed if args.seed is not None else cfg.train.seeds[0])
    what = {"frozen", "s3ft", "external"} if args.what == "all" else {args.what}
    base_meta = {"mode": cfg.preprocess.mode, "size": int(cfg.preprocess.size), "pad": cfg.preprocess.pad,
                 "precision": resolve_amp(cfg.train, device)[2]}
    img = load_images(paths)

    novel_ids = None
    if what & {"s3ft"}:
        s3 = load_s3(paths)
        s1 = load_s1(paths, img)
        novel = set(s3[s3.role == "novel"].subject)
        novel_ids = list(s1[s1.subject.isin(novel) & (s1.split != "excluded")].image_id)

    externals = {}
    if "external" in what:
        for name, key in (("earvn1", "earvn1_root"), ("awe", "awe_root")):
            root = Path(cfg.paths[key])
            if root.exists():
                df = scan_external(root, cfg)
                atomic_write_csv(df, paths.metadata / f"external_{name}.csv")
                externals[name] = (str(root), list(df.image_id))
                log.info(f"external {name}: {df.subject.nunique()} subjects, {len(df)} images")
            else:
                log.info(f"external {name}: {root} not found, skipped")

    for spec in select_models(args.models):
        pre = lambda s=spec: create_model(s, num_classes=0, pretrained=bool(cfg.pretrained))
        ck = paths.run_dir("S3", spec.key, f"seed{seed}") / "best.pt"

        def ft(s=spec, c=ck):
            if not c.exists():
                log.warning(f"  no S3 checkpoint for {s.key} ({c}); run scripts/09_train_s3.py")
                return None
            return load_finetuned(s, c, device)

        meta_f = {**base_meta, "model": spec.timm, "weights": "pretrained" if cfg.pretrained else "random_init"}
        meta_s = {**base_meta, "model": spec.timm, "weights": f"s3_seed{seed}",
                  "checkpoint_mtime": int(ck.stat().st_mtime) if ck.exists() else None}  # retrained -> re-extract
        if "frozen" in what:
            run(spec, "frozen", cfg.paths.dataset_root, list(img.image_id), pre, meta_f, paths, cfg, device, log,
                args.overwrite)
        if "s3ft" in what:
            run(spec, "s3ft", cfg.paths.dataset_root, novel_ids, ft, meta_s, paths, cfg, device, log, args.overwrite)
        for name, (root, ids) in externals.items():
            run(spec, f"frozen_{name}", root, ids, pre, meta_f, paths, cfg, device, log, args.overwrite)
            run(spec, f"s3ft_{name}", root, ids, ft, meta_s, paths, cfg, device, log, args.overwrite)


if __name__ == "__main__":
    main()
