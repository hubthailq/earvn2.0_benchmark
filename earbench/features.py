"""Frozen-feature extraction with an on-disk cache (one .npz per model and variant)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import ImageTable, make_loader
from .device import DeviceManager, resolve_amp
from .preprocess import build_transform, model_norm


@torch.no_grad()
def extract(model: torch.nn.Module, root: str | Path, image_ids, cfg, device, mode: str | None = None,
            batch_size: int | None = None, progress: bool = True) -> np.ndarray:
    """Pooled features (the input of the classifier head) for every image, in the given order."""
    use_amp, amp_dtype, _ = resolve_amp(cfg.train, device)   # same precision as fine-tuning, for every model
    dm = DeviceManager(device, cfg.get("gpu_fallback"), infer_amp_dtype=amp_dtype if use_amp else None)
    model = dm.place(model).eval()
    mean, std = model_norm(model)
    tf = build_transform(cfg.preprocess, mean, std, train=False, mode=mode)
    ds = ImageTable(root, image_ids, None, tf)
    dl = make_loader(ds, batch_size or cfg.feature_batch_size, shuffle=False, num_workers=cfg.num_workers)
    out = []
    it = dl
    if progress:
        from tqdm import tqdm
        it = tqdm(dl, desc="features", leave=False)
    embed = lambda m, t: m.forward_head(m.forward_features(t), pre_logits=True)  # noqa: E731
    for x, _, _ in it:
        out.append(dm.forward(model, x, embed).numpy())
    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    feats = np.concatenate(out).astype(np.float32)
    if not np.isfinite(feats).all():
        raise FloatingPointError("non-finite values in extracted features")
    return feats


def save_features(path: str | Path, ids, feats: np.ndarray, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ids=np.asarray(list(ids), dtype=object), feats=feats.astype(np.float16),
                        meta=json.dumps(meta))
    tmp.replace(path)


def load_features(path: str | Path, ids=None, expect_meta: dict | None = None):
    """Load cached features; if ``ids`` is given, return rows in that order (missing ids -> KeyError)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/10_extract_features.py first")
    z = np.load(path, allow_pickle=True)
    stored_ids = [str(i) for i in z["ids"]]
    feats = z["feats"].astype(np.float32)
    meta = json.loads(str(z["meta"]))
    if expect_meta:
        diff = {k: (meta.get(k), v) for k, v in expect_meta.items() if meta.get(k) != v}
        if diff:
            raise ValueError(f"cached features {path} were made with different settings: {diff}. "
                             f"Delete the file or pass --overwrite.")
    if ids is None:
        return stored_ids, feats, meta
    pos = {k: i for i, k in enumerate(stored_ids)}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise KeyError(f"{len(missing)} images missing from {path} (e.g. {missing[:3]}); re-extract")
    return list(ids), feats[[pos[i] for i in ids]], meta


def s3_features_are_current(path: str | Path, paths, model_key: str) -> bool:
    """True if S3 fine-tuned features were extracted from the checkpoint that exists NOW (script 09 may have
    retrained the model, or the retraining diverged and left no checkpoint)."""
    import re
    _, _, meta = load_features(path)
    m = re.fullmatch(r"s3_seed(\d+)", str(meta.get("weights", "")))
    if not m:
        return False
    ck = paths.run_dir("S3", model_key, f"seed{m.group(1)}") / "best.pt"
    return ck.exists() and meta.get("checkpoint_mtime") == int(ck.stat().st_mtime)
