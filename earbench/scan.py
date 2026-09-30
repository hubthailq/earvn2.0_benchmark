"""Step 01: discover subjects and images, read image sizes, detect broken files, assign gender."""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from .utils import load_rgb, natural_key, natural_sorted

JUNK_NAMES = {"thumbs.db", "desktop.ini", ".ds_store"}


def _is_hidden(name: str) -> bool:
    return name.startswith(".") or name.startswith("__MACOSX") or name.lower() in JUNK_NAMES


def _split_alias_map(cfg) -> dict[str, str]:
    m = {}
    for split, aliases in cfg.dataset.split_folder_aliases.items():
        for a in aliases:
            m[a.lower()] = split
    return m


def list_subject_images(subject_dir: Path, root: Path, exts: set[str], alias: dict[str, str]):
    """Yield (image_id, orig_split) for every image of one subject, plus problem records."""
    items, problems = [], []
    for dirpath, dirnames, filenames in os.walk(subject_dir):
        dirnames[:] = sorted([d for d in dirnames if not _is_hidden(d)], key=natural_key)
        rel_dir = Path(dirpath).relative_to(subject_dir)
        first = rel_dir.parts[0].lower() if rel_dir.parts else ""
        split = alias.get(first, "")
        for fn in natural_sorted(filenames):
            if _is_hidden(fn):
                continue
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            if p.suffix.lower() not in exts:
                problems.append({"image_id": rel, "problem": "not_an_image_extension"})
                continue
            items.append((rel, split))
    return items, problems


def _header_size(path) -> tuple[int, int]:
    """(width, height) from the file header, after EXIF rotation, without decoding the pixels."""
    with Image.open(path) as im:
        w, h = im.size
        try:
            if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                w, h = h, w
        except Exception:
            pass
    return w, h


def _probe(args):
    root, image_id, keep_truncated = args
    path = Path(root) / image_id
    rec = {"image_id": image_id, "width": np.nan, "height": np.nan, "file_bytes": 0,
           "md5": "", "status": "ok", "truncated": False}
    try:
        rec["file_bytes"] = path.stat().st_size
        if rec["file_bytes"] == 0:
            rec["status"] = "empty_file"
            return rec
        with open(path, "rb") as f:
            rec["md5"] = hashlib.md5(f.read()).hexdigest()
        try:
            size = load_rgb(path).size   # full decode; scan_dataset makes truncated files raise here
        except OSError as exc:
            if "truncated" not in str(exc).lower():
                raise
            # the file ends early: when training, PIL would silently paint the missing part grey
            rec["status"] = "ok" if keep_truncated else "truncated"
            rec["truncated"] = True
            size = _header_size(path)
        rec["width"], rec["height"] = size
        if min(size) < 1:
            rec["status"] = "zero_size"
    except Exception as exc:  # any decode failure
        rec["status"] = f"corrupt: {type(exc).__name__}"
    return rec


def scan_dataset(cfg, logger=None, workers: int = 8):
    root = Path(cfg.paths.dataset_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\nPut EarVN2.0 there (one folder per subject), "
            f"or run with --set paths.dataset_root=/your/path")
    exts = {e.lower() for e in cfg.dataset.image_extensions}
    alias = _split_alias_map(cfg)

    subjects = natural_sorted([d.name for d in root.iterdir() if d.is_dir() and not _is_hidden(d.name)])
    loose = [f.name for f in root.iterdir() if f.is_file() and not _is_hidden(f.name)]
    if not subjects:
        raise ValueError(f"No subject folders inside {root}. Expected {root}/<subject>/<images>.")

    rows, problems = [], [{"image_id": f, "problem": "file_outside_subject_folder"} for f in loose]
    for s in subjects:
        items, probs = list_subject_images(root / s, root, exts, alias)
        problems += probs
        if not items:
            problems.append({"image_id": s + "/", "problem": "subject_has_no_images"})
        for image_id, split in items:
            rows.append({"image_id": image_id, "subject": s, "orig_split": split})
    if not rows:
        raise ValueError(f"No images found under {root} (extensions: {sorted(exts)}).")
    images = pd.DataFrame(rows)

    keep = bool(cfg.dataset.get("keep_truncated", False))
    old_flag = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False   # strict decoding while scanning (restored below)
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            probes = list(ex.map(_probe, [(str(root), i, keep) for i in images.image_id]))
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old_flag
    images = images.merge(pd.DataFrame(probes), on="image_id", how="left")
    images["min_side"] = images[["width", "height"]].min(axis=1)
    images["aspect_hw"] = images["height"] / images["width"]
    images["tiny"] = images["min_side"] < cfg.dataset.min_side_warn

    # mixed layouts inside one subject (some images in train/val/test sub-folders, some loose)
    for s, g in images.groupby("subject"):
        if (g.orig_split != "").any() and (g.orig_split == "").any():
            problems.append({"image_id": s + "/", "problem": "mixed_layout: loose images next to split folders"})

    bad = images[images.status != "ok"]
    for _, r in bad.iterrows():
        problems.append({"image_id": r.image_id, "problem": r.status})
    for i in images.image_id[(images.status == "ok") & images.truncated.astype(bool)]:
        problems.append({"image_id": i, "problem": "truncated (kept: dataset.keep_truncated=true)"})
    problems_df = pd.DataFrame(problems, columns=["image_id", "problem"])
    if logger:
        logger.info(f"{len(subjects)} subjects, {len(images)} images, {len(bad)} unreadable, "
                    f"{int(images.tiny.sum())} tiny (<{cfg.dataset.min_side_warn}px)")
    return images, subjects, problems_df


def scan_external(root: str | Path, cfg, workers: int = 8) -> pd.DataFrame:
    """Readable images of another dataset laid out as root/<subject>/.../<image> (EarVN1.0, AWE...)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"external dataset not found: {root}")
    exts = {e.lower() for e in cfg.dataset.image_extensions}
    rows = []
    for s in natural_sorted([d.name for d in root.iterdir() if d.is_dir() and not _is_hidden(d.name)]):
        items, _ = list_subject_images(root / s, root, exts, {})
        rows += [{"image_id": i, "subject": s} for i, _ in items]
    if not rows:
        raise ValueError(f"no images under {root}")
    df = pd.DataFrame(rows)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        probes = list(ex.map(_probe, [(str(root), i, True) for i in df.image_id]))
    df = df.merge(pd.DataFrame(probes), on="image_id", how="left")
    return df[df.status == "ok"].reset_index(drop=True)


def assign_gender(cfg, subjects: list[str], logger=None) -> pd.DataFrame:
    """Gender per subject: from paths.gender_labels if present, else the first `male_count` folders."""
    male, female = cfg.gender.male_label, cfg.gender.female_label
    labels_path = Path(cfg.paths.gender_labels)
    df = pd.DataFrame({"subject": subjects, "order": range(len(subjects))})
    if labels_path.exists():
        lab = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        lab.columns = [c.strip().lower() for c in lab.columns]
        if not {"subject", "gender"} <= set(lab.columns):
            raise ValueError(f"{labels_path} must have columns 'subject,gender'")
        lab["subject"] = lab["subject"].str.strip()
        lab["gender"] = lab["gender"].str.strip().str.upper()
        dup = lab[lab.subject.duplicated()].subject.tolist()
        if dup:
            raise ValueError(f"{labels_path}: subjects listed twice: {dup[:10]}")
        bad = lab[~lab.gender.isin([male, female])]
        if len(bad):
            raise ValueError(f"{labels_path}: gender must be '{male}' or '{female}', got "
                             f"{bad.gender.unique().tolist()[:5]}")
        df = df.merge(lab[["subject", "gender"]], on="subject", how="left")
        missing = df[df.gender.isna()].subject.tolist()
        if missing:
            raise ValueError(f"{labels_path} has no gender for {len(missing)} subjects, e.g. {missing[:10]}")
        extra = sorted(set(lab.subject) - set(subjects))
        if extra and logger:
            logger.warning(f"{labels_path} lists {len(extra)} subjects not in the dataset, e.g. {extra[:5]}")
        df["gender_source"] = "labels_file"
    else:
        if cfg.gender.male_count is None:
            raise ValueError(
                f"Gender labels are missing. Either create {labels_path} (columns subject,gender with M/F), "
                f"or set the number of male folders: --set gender.male_count=<N> "
                f"(or edit gender.male_count in configs/default.yaml). The dataset has {len(subjects)} subject folders.")
        n_male = int(cfg.gender.male_count)
        if not 0 < n_male < len(subjects):
            raise ValueError(f"gender.male_count={n_male} must be between 1 and {len(subjects) - 1} "
                             f"(dataset has {len(subjects)} subjects)")
        df["gender"] = [male if i < n_male else female for i in range(len(subjects))]
        df["gender_source"] = f"folder_order(male_count={n_male})"
        if logger:
            lo, hi = max(0, n_male - 3), min(len(subjects), n_male + 3)
            logger.warning("Gender taken from FOLDER ORDER. Check the boundary below, then either keep "
                           "gender.male_count or write data/gender_labels.csv:")
            for i in range(lo, hi):
                logger.warning(f"   #{i + 1:4d}  {subjects[i]:>20s}  -> {df.gender[i]}")
    return df


def dataset_statistics(images: pd.DataFrame, subjects: pd.DataFrame, bins) -> dict:
    ok = images[images.status == "ok"]
    per = ok.groupby("subject").size()
    g = subjects.set_index("subject").gender
    stats = {
        "n_subjects": int(subjects.shape[0]),
        "n_subjects_by_gender": {k: int(v) for k, v in g.value_counts().items()},
        "n_images_ok": int(len(ok)),
        "n_images_unreadable": int((images.status != "ok").sum()),
        "n_images_truncated": int(images.get("truncated", pd.Series(False, index=images.index)).astype(bool).sum()),
        "n_images_by_gender": {k: int(v) for k, v in ok.subject.map(g).value_counts().items()},
        "images_per_subject": {"min": int(per.min()), "median": float(per.median()),
                               "mean": round(float(per.mean()), 1), "max": int(per.max())},
        "width": _describe(ok.width), "height": _describe(ok.height),
        "min_side": _describe(ok.min_side),
        "aspect_hw": _describe(ok.aspect_hw, 3),
        "n_tiny": int(ok.tiny.sum()),
        "short_side_bins": {},
        "pre_split_layout": bool((images.orig_split != "").any()),
    }
    cut = pd.cut(ok.min_side, bins=list(bins), right=False)
    for interval, n in cut.value_counts(sort=False).items():
        stats["short_side_bins"][f"[{int(interval.left)},{int(interval.right)})"] = int(n)
    return stats


def _describe(s: pd.Series, nd: int = 1) -> dict:
    s = s.dropna()
    if s.empty:
        return {}
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"min": round(float(s.min()), nd), "p5": round(float(q[0.05]), nd), "p25": round(float(q[0.25]), nd),
            "median": round(float(q[0.5]), nd), "p75": round(float(q[0.75]), nd),
            "p95": round(float(q[0.95]), nd), "max": round(float(s.max()), nd), "mean": round(float(s.mean()), nd)}


def suggest_nonsquare(aspect_median: float, height: int = 224, multiple: int = 32) -> tuple[int, int]:
    """Input size (H, W) for E0 'nonsquare' mode from the median height/width ratio."""
    if not np.isfinite(aspect_median) or aspect_median <= 0:
        return height, height
    w = max(multiple, int(round(height / aspect_median / multiple)) * multiple)
    return height, min(w, height)
