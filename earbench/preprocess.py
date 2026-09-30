"""Image preprocessing (see the 'Tiền xử lý dữ liệu' section of the plan).

Pipeline for one image:
  1. load as RGB (EXIF orientation fixed, grayscale/alpha/CMYK converted)       [utils.load_rgb]
  2. [train only] random crop that keeps the aspect ratio (area in augment.scale)
  3. fit to the canvas:  letterbox (keep ratio + pad)  |  stretch  |  nonsquare letterbox
  4. [train only] rotation / translation, colour jitter, horizontal flip
  5. to tensor + normalise with the model's own mean/std
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
from PIL import Image
from torchvision import transforms as T

_INTERP = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "nearest": Image.NEAREST,
           "lanczos": Image.LANCZOS}
VALID_MODES = ("letterbox", "stretch", "nonsquare")


def canvas_size(cfg_pre, mode: str | None = None) -> tuple[int, int]:
    mode = mode or cfg_pre.mode
    if mode not in VALID_MODES:
        raise ValueError(f"preprocess.mode must be one of {VALID_MODES}, got '{mode}'")
    if mode == "nonsquare":
        h, w = (int(x) for x in cfg_pre.nonsquare_hw)
        return h, w
    return int(cfg_pre.size), int(cfg_pre.size)


def fit_to_canvas(img: Image.Image, height: int, width: int, mode: str = "letterbox",
                  pad: str = "edge", pad_rgb: Sequence[int] = (124, 116, 104),
                  interpolation: str = "bicubic") -> Image.Image:
    """Resize ``img`` onto a height x width canvas.

    stretch   : plain resize (distorts the ear if the aspect ratio differs)
    letterbox / nonsquare : keep the aspect ratio, then pad the short side, image centred.
    pad = 'edge' replicates border pixels (as in VGGFace-Ear); 'mean' uses a constant colour.
    """
    if height < 1 or width < 1:
        raise ValueError("canvas must be at least 1x1")
    interp = _INTERP.get(interpolation, Image.BICUBIC)
    w0, h0 = img.size
    if w0 < 1 or h0 < 1:
        raise ValueError("image has zero size")
    if mode == "stretch":
        return img.resize((width, height), interp)
    scale = min(height / h0, width / w0)
    nw = min(width, max(1, int(round(w0 * scale))))
    nh = min(height, max(1, int(round(h0 * scale))))
    resized = img.resize((nw, nh), interp)
    if nw == width and nh == height:
        return resized
    left = (width - nw) // 2
    top = (height - nh) // 2
    if pad == "edge":
        arr = np.asarray(resized)
        arr = np.pad(arr, ((top, height - nh - top), (left, width - nw - left), (0, 0)), mode="edge")
        return Image.fromarray(arr)
    canvas = Image.new("RGB", (width, height), tuple(int(c) for c in pad_rgb))
    canvas.paste(resized, (left, top))
    return canvas


class AspectPreservingCrop:
    """Random crop covering a fraction of the area, with the ORIGINAL aspect ratio (no distortion)."""

    def __init__(self, scale: Sequence[float] = (0.8, 1.0)):
        lo, hi = float(scale[0]), float(scale[1])
        if not 0 < lo <= hi <= 1:
            raise ValueError(f"augment.scale must satisfy 0 < min <= max <= 1, got {scale}")
        self.lo, self.hi = lo, hi

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        a = random.uniform(self.lo, self.hi)
        cw, ch = max(1, int(round(w * math.sqrt(a)))), max(1, int(round(h * math.sqrt(a))))
        if cw >= w and ch >= h:
            return img
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        return img.crop((x, y, x + cw, y + ch))


class FitToCanvas:
    def __init__(self, height, width, mode, pad, pad_rgb, interpolation):
        self.args = (height, width, mode, pad, tuple(pad_rgb), interpolation)

    def __call__(self, img):
        h, w, mode, pad, pad_rgb, interp = self.args
        return fit_to_canvas(img, h, w, mode, pad, pad_rgb, interp)


def build_transform(cfg_pre, mean: Sequence[float], std: Sequence[float], train: bool,
                    mode: str | None = None) -> T.Compose:
    mode = mode or cfg_pre.mode
    h, w = canvas_size(cfg_pre, mode)
    pad_rgb = [int(round(255 * m)) for m in mean]
    ops = []
    aug = cfg_pre.augment
    if train and aug.scale and float(aug.scale[0]) < 1.0:
        ops.append(AspectPreservingCrop(aug.scale))
    ops.append(FitToCanvas(h, w, "stretch" if mode == "stretch" else "letterbox",
                           cfg_pre.pad, pad_rgb, cfg_pre.interpolation))
    if train:
        if aug.rotate_deg or aug.translate:
            ops.append(T.RandomAffine(degrees=float(aug.rotate_deg or 0),
                                      translate=(float(aug.translate or 0),) * 2 if aug.translate else None,
                                      interpolation=T.InterpolationMode.BILINEAR, fill=pad_rgb))
        if aug.color_jitter and any(float(x) > 0 for x in aug.color_jitter):
            b, c, s = (float(x) for x in aug.color_jitter)
            ops.append(T.ColorJitter(brightness=b, contrast=c, saturation=s))
        if aug.hflip:
            ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))]
    return T.Compose(ops)


def model_norm(model) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mean/std the pretrained weights expect (e.g. CLIP and MobileViT differ from ImageNet)."""
    import timm
    dc = timm.data.resolve_model_data_config(model)
    return tuple(float(x) for x in dc["mean"]), tuple(float(x) for x in dc["std"])


def save_preview(images: list[Image.Image], cfg_pre, path, mode: str | None = None) -> None:
    """Save a strip showing original vs. processed images (for the report / sanity check)."""
    h, w = canvas_size(cfg_pre, mode)
    tiles = []
    for im in images:
        a = fit_to_canvas(im, h, w, "stretch", "edge")
        b = fit_to_canvas(im, h, w, "letterbox", cfg_pre.pad)
        tiles += [a, b]
    if not tiles:
        return
    strip = Image.new("RGB", (w * len(tiles), h), (255, 255, 255))
    for i, t in enumerate(tiles):
        strip.paste(t, (i * w, 0))
    strip.save(path)
