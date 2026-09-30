#!/usr/bin/env python
"""Generate a small synthetic EarVN-like dataset for dry runs and tests.

Each subject gets an ear-like shape with its own colour/geometry, so models can learn something.
With --edge-cases the dataset also contains the problems real data has: broken/empty files,
non-image files, grayscale / RGBA / palette / CMYK / 16-bit images, EXIF rotation, tiny and extreme
aspect-ratio images, unicode and upper-case file names, exact and near duplicates (also across
subjects), a subject with too few images, an empty subject folder and a stray file at the root.

    python tools/make_synthetic_dataset.py --out data/synthetic --subjects 12 --images 100 --edge-cases
    python scripts/01_scan_dataset.py --set paths.dataset_root=data/synthetic --set gender.male_count=6 ...
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def ear_image(rng, base, w, h):
    """Draw an ear-like picture: skin background, outer helix ellipse, inner concha, subject-specific."""
    col_bg, col_ear, col_in, ox, oy, tilt = base
    scale = 4
    W, H = max(8, w * scale), max(8, h * scale)
    im = Image.new("RGB", (W, H), tuple(int(c) for c in col_bg))
    d = ImageDraw.Draw(im)
    jit = rng.normal(0, 0.03, 4)
    d.ellipse([W * (0.10 + jit[0]), H * (0.05 + jit[1]), W * (0.90 + jit[2]), H * (0.95 + jit[3])],
              fill=tuple(int(c) for c in col_ear))
    d.ellipse([W * (0.30 + ox), H * (0.30 + oy), W * (0.70 + ox), H * (0.75 + oy)],
              fill=tuple(int(c) for c in col_in))
    d.line([W * 0.5, H * 0.1, W * (0.5 + tilt), H * 0.6], fill=(40, 20, 20), width=max(1, W // 20))
    for _ in range(int(rng.integers(3, 7))):  # hair, earrings, background clutter: differs per photo
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.05, 0.2) * W
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(c) for c in rng.uniform(0, 255, 3)))
    im = im.rotate(float(rng.normal(0, 8)), fillcolor=tuple(int(c) for c in col_bg))
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)  # left / right ear
    arr = np.asarray(im).astype(np.float32)
    arr += rng.normal(0, 8, arr.shape) + rng.normal(0, 15)  # noise + illumination
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return im.resize((max(1, w), max(1, h)), Image.BICUBIC)


def subject_base(rng):
    return (rng.uniform(150, 230, 3), rng.uniform(120, 220, 3), rng.uniform(60, 160, 3),
            rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.2))


def make(out: Path, n_subjects: int, n_images: int, layout: str, edge_cases: bool, seed: int = 0,
         train: int = 45, val: int = 15) -> Path:
    rng = np.random.default_rng(seed)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names = [f"{i:03d}" for i in range(1, n_subjects + 1)]
    for s in names:
        base = subject_base(rng)
        sd = out / s
        files = []
        for k in range(n_images):
            w = int(rng.integers(12, 90))
            h = int(round(w * rng.uniform(1.5, 2.3)))
            files.append((f"{s}_{k:04d}.jpg", ear_image(rng, base, w, h)))
        if layout == "presplit":
            parts = {"train": files[:train], "val": files[train:train + val], "test": files[train + val:]}
        else:
            parts = {"": files}
        for split, items in parts.items():
            d = sd / split if split else sd
            d.mkdir(parents=True, exist_ok=True)
            for fn, im in items:
                im.save(d / fn, quality=92)

    if edge_cases:
        s0 = out / names[0]
        target = s0 / "test" if layout == "presplit" else s0
        base = subject_base(rng)
        (target / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
        (target / "empty.jpg").write_bytes(b"")
        (target / "notes.txt").write_text("not an image")
        (target / ".DS_Store").write_bytes(b"junk")
        ear_image(rng, base, 30, 60).convert("L").save(target / "gray.png")
        rgba = ear_image(rng, base, 30, 60).convert("RGBA")
        rgba.putalpha(128)
        rgba.save(target / "alpha.png")
        ear_image(rng, base, 30, 60).convert("P", palette=Image.ADAPTIVE).save(target / "palette.png")
        ear_image(rng, base, 30, 60).convert("CMYK").save(target / "cmyk.jpg")
        arr16 = (np.asarray(ear_image(rng, base, 30, 60).convert("L")).astype(np.uint16) * 257)
        Image.fromarray(arr16).save(target / "sixteen_bit.png")
        im = ear_image(rng, base, 30, 60)
        exif = im.getexif()
        exif[0x0112] = 6  # rotate 90 on load
        im.save(target / "exif_rot.jpg", exif=exif)
        ear_image(rng, base, 3, 5).save(target / "tiny.png")
        ear_image(rng, base, 5, 200).save(target / "extreme_aspect.png")
        ear_image(rng, base, 30, 60).save(target / "tai phải ảnh.jpg")
        ear_image(rng, base, 30, 60).save(target / "UPPER.JPG")
        nested = target / "extra_folder"
        nested.mkdir(exist_ok=True)
        ear_image(rng, base, 30, 60).save(nested / "nested.jpg")
        # exact + near duplicates inside subject 2, and one shared with subject 3
        s1 = out / names[1]
        src = sorted(s1.rglob("*.jpg"))[0]
        shutil.copy(src, src.parent / "dup_exact.jpg")
        Image.open(src).resize((Image.open(src).width + 1, Image.open(src).height + 1)).save(src.parent / "dup_near.jpg", quality=80)
        s2 = out / names[2]
        dst = (s2 / "test") if layout == "presplit" else s2
        shutil.copy(src, dst / "cross_subject_dup.jpg")
        # subject with too few images, empty subject folder, stray root file
        few = out / f"{n_subjects + 1:03d}"
        few.mkdir()
        for k in range(10):
            ear_image(rng, subject_base(rng), 30, 60).save(few / f"few_{k}.jpg")
        (out / f"{n_subjects + 2:03d}").mkdir()
        (out / "README_stray.txt").write_text("stray")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--layout", choices=["flat", "presplit"], default="presplit")
    ap.add_argument("--edge-cases", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = make(Path(a.out), a.subjects, a.images, a.layout, a.edge_cases, a.seed)
    print(f"synthetic dataset written to {p}")


if __name__ == "__main__":
    main()
