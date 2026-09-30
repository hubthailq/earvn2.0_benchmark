"""Small shared helpers: logging, seeding, natural sort, safe image loading, result storage."""
from __future__ import annotations

import logging
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# Truncated JPEGs are common in web-cropped data; load what is there instead of crashing.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def get_logger(name: str = "earbench", log_file: str | os.PathLike | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = str(log_file)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
                   for h in logger.handlers):
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
            logger.addHandler(fh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def natural_key(text: str):
    """Sort key so that '2' < '10' and 'sub2' < 'sub10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def natural_sorted(items: Iterable[str]) -> list[str]:
    return sorted(items, key=natural_key)


def load_rgb(path: str | os.PathLike) -> Image.Image:
    """Open any image (grayscale, RGBA, palette, CMYK, 16-bit, EXIF-rotated) as 8-bit RGB."""
    with Image.open(path) as im:
        im.load()
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:  # broken EXIF must not kill the run
            pass
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(im, dtype=np.float64)
            lo, hi = float(arr.min()), float(arr.max())
            arr = np.zeros_like(arr) if hi <= lo else (arr - lo) / (hi - lo) * 255.0
            im = Image.fromarray(arr.astype(np.uint8), mode="L")
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
            im = Image.alpha_composite(bg, im)
        return im.convert("RGB")


def atomic_write_csv(df: pd.DataFrame, path: str | os.PathLike, **kwargs) -> None:
    """Write CSV through a temp file + rename, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_csv(path: str | os.PathLike, dtype: dict | None = None, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the earlier script first (see README).")
    types = {"subject": str, "image_id": str}
    types.update(dtype or {})
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype=types, **kwargs)


def fingerprint(*parts) -> str:
    """Short, stable hash of settings / id lists / arrays (used to re-run results whose inputs changed)."""
    import hashlib
    import json

    def norm(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (pd.Series, pd.Index)):
            return x.astype(str).tolist()
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x
    blob = json.dumps([norm(p) for p in parts], sort_keys=True, default=str)
    return "fp" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]   # prefix: never read back as a number


def feature_fingerprint(path: str | os.PathLike) -> dict:
    """Identity of a features file: its settings and when it was (re)extracted."""
    import json
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta"]))
    return {"meta": meta, "mtime": int(path.stat().st_mtime)}


class ResultStore:
    """A CSV of results keyed by some columns. Re-running a finished run is skipped;
    re-running with ``overwrite`` replaces the row. Safe against crashes (atomic writes)."""

    def __init__(self, path: str | os.PathLike, key_cols: Sequence[str]):
        self.path = Path(path)
        self.key_cols = list(key_cols)

    def load(self) -> pd.DataFrame:
        if self.path.exists() and self.path.stat().st_size > 0:
            return pd.read_csv(self.path, keep_default_na=False, na_values=[""], dtype={"fingerprint": str})
        return pd.DataFrame(columns=self.key_cols)

    def _mask(self, df: pd.DataFrame, key: dict):
        m = pd.Series(True, index=df.index)
        for k in self.key_cols:
            if k not in df.columns:
                return pd.Series(False, index=df.index)
            m &= df[k].astype(str) == str(key[k])
        return m

    def has(self, key: dict) -> bool:
        df = self.load()
        return bool(len(df)) and bool(self._mask(df, key).any())

    def get(self, key: dict) -> dict | None:
        """The stored row for this key, or None."""
        df = self.load()
        if not len(df):
            return None
        hit = df[self._mask(df, key)]
        return None if hit.empty else hit.iloc[-1].to_dict()

    def should_skip(self, key: dict, overwrite: bool, logger=None, **must_match) -> bool:
        """True if the run is already done with the same settings (e.g. lr=...). A finished run made with
        different settings is re-run (with a warning), so changing the LR sweep never leaves stale results."""
        if overwrite:
            return False
        row = self.get(key)
        if row is None:
            return False
        for k, v in must_match.items():
            old = row.get(k)
            if isinstance(v, str):
                same = str(old) == v
                if not same:
                    if logger:
                        logger.warning(f"{key}: stored result used {k}={old}, now {k}={v} -> re-running")
                    return False
                continue
            try:
                same = abs(float(old) - float(v)) <= 1e-12 * max(1.0, abs(float(v)))
            except (TypeError, ValueError):
                same = str(old) == str(v)
            if not same:
                if logger:
                    logger.warning(f"{key}: stored result used {k}={old}, now {k}={v} -> re-running")
                return False
        return True

    def remove(self, match: dict) -> int:
        """Delete every row whose columns equal ``match`` (a subset of columns is fine). Returns the count."""
        df = self.load()
        if not len(df) or any(k not in df.columns for k in match):
            return 0
        m = pd.Series(True, index=df.index)
        for k, v in match.items():
            m &= df[k].astype(str) == str(v)
        if m.any():
            atomic_write_csv(df[~m], self.path)
        return int(m.sum())

    def upsert(self, row: dict) -> None:
        missing = [k for k in self.key_cols if k not in row]
        if missing:
            raise KeyError(f"result row misses key columns {missing}")
        df = self.load()
        if len(df):
            df = df[~self._mask(df, row)]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        atomic_write_csv(df, self.path)
