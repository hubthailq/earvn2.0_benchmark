#!/usr/bin/env python
"""00 - Check the environment: Python packages, GPU, dataset folder, and (optionally) pre-download
the pretrained weights of all 30 models so later scripts never stall on downloads.

    python scripts/00_check_env.py              # quick check
    python scripts/00_check_env.py --download   # also download/verify every pretrained model
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earbench.common import setup  # noqa: E402


def extra(p):
    p.add_argument("--download", action="store_true", help="download and test all pretrained weights")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    ok = True
    import importlib
    for mod in ["torch", "torchvision", "timm", "numpy", "pandas", "sklearn", "scipy", "PIL", "imagehash",
                "yaml", "tqdm"]:
        try:
            m = importlib.import_module(mod)
            log.info(f"  {mod:12s} {getattr(m, '__version__', '?')}")
        except ImportError:
            log.error(f"  {mod:12s} MISSING -> pip install -r requirements.txt")
            ok = False
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log.info(f"  GPU: {p.name}, {p.total_memory / 2**30:.1f} GB, bf16={torch.cuda.is_bf16_supported()}")
    else:
        log.warning("  No CUDA GPU found: training will run on CPU (very slow). Feature-based experiments still work.")

    root = Path(cfg.paths.dataset_root)
    if root.exists():
        n = sum(1 for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
        log.info(f"  dataset: {root} ({n} sub-folders)")
    else:
        log.error(f"  dataset folder not found: {root} (see data/README.md)")
        ok = False
    free = shutil.disk_usage(paths.root).free / 2**30
    log.info(f"  free disk at {paths.root}: {free:.0f} GB")
    if free < 20:
        log.warning("  less than 20 GB free: features and checkpoints may not fit")

    if args.download:
        from earbench.models import create_model, load_registry
        for spec in load_registry():
            try:
                m = create_model(spec, num_classes=0, pretrained=True)
                x = torch.zeros(1, 3, 224, 224)
                with torch.no_grad():
                    f = m.forward_head(m.forward_features(x), pre_logits=True)
                log.info(f"  OK   {spec.key:14s} {spec.timm:45s} feat_dim={f.shape[1]}")
            except Exception as exc:
                log.error(f"  FAIL {spec.key:14s} {spec.timm}: {exc}")
                ok = False
    log.info("environment OK" if ok else "environment has problems (see errors above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
