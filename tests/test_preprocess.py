"""Preprocessing: letterbox / stretch / non-square, padding, augmentation shapes."""
import numpy as np
import pytest
import torch
from PIL import Image

from earbench.config import load_config
from earbench.preprocess import AspectPreservingCrop, build_transform, canvas_size, fit_to_canvas


def tall(w=30, h=60, color=(200, 50, 50)):
    return Image.new("RGB", (w, h), color)


@pytest.mark.parametrize("size", [(1, 1), (3, 5), (5, 200), (200, 5), (224, 224), (1000, 300), (30, 60)])
@pytest.mark.parametrize("mode", ["letterbox", "stretch"])
def test_fit_to_canvas_output_size(size, mode):
    out = fit_to_canvas(tall(*size), 224, 224, mode)
    assert out.size == (224, 224) and out.mode == "RGB"


def test_letterbox_keeps_aspect_ratio():
    img = tall(30, 60, (255, 0, 0))
    out = np.asarray(fit_to_canvas(img, 224, 224, "letterbox", pad="mean", pad_rgb=(0, 0, 255)))
    red = (out[..., 0] > 200) & (out[..., 2] < 50)
    ys, xs = np.nonzero(red)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    assert h == 224 and abs(w - 112) <= 1          # 1:2 ratio preserved
    assert abs((xs.min() + xs.max()) / 2 - 111.5) <= 1  # centred
    assert (out[:, :50, 2] > 200).all()             # padding colour on the sides


def test_edge_padding_replicates_border():
    img = Image.new("RGB", (10, 20), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 1, 20))  # white left column
    out = np.asarray(fit_to_canvas(img, 40, 40, "letterbox", pad="edge"))
    assert out[20, 0].min() > 200  # left padding copies the white border


def test_stretch_distorts():
    out = np.asarray(fit_to_canvas(tall(30, 60, (255, 0, 0)), 224, 224, "stretch"))
    assert (out[..., 0] > 200).all()


def test_nonsquare_canvas_and_invalid_mode():
    cfg = load_config(overrides=["preprocess.nonsquare_hw=[224, 128]"])
    assert canvas_size(cfg.preprocess, "nonsquare") == (224, 128)
    assert canvas_size(cfg.preprocess, "letterbox") == (224, 224)
    with pytest.raises(ValueError):
        canvas_size(cfg.preprocess, "crop")


@pytest.mark.parametrize("mode", ["letterbox", "stretch", "nonsquare"])
@pytest.mark.parametrize("train", [True, False])
def test_transform_shapes(mode, train):
    cfg = load_config(overrides=["preprocess.nonsquare_hw=[224, 128]"])
    tf = build_transform(cfg.preprocess, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), train=train, mode=mode)
    for img in (tall(3, 5), tall(5, 300), tall(80, 40)):
        x = tf(img)
        assert isinstance(x, torch.Tensor) and x.shape == ((3, 224, 128) if mode == "nonsquare" else (3, 224, 224))
        assert torch.isfinite(x).all()


def test_eval_transform_is_deterministic():
    cfg = load_config()
    tf = build_transform(cfg.preprocess, (0.5,) * 3, (0.5,) * 3, train=False)
    img = Image.fromarray((np.random.default_rng(0).random((50, 25, 3)) * 255).astype(np.uint8))
    assert torch.equal(tf(img), tf(img))


def test_aspect_preserving_crop():
    crop = AspectPreservingCrop((0.5, 0.5))
    out = crop(tall(40, 80))
    assert abs(out.size[1] / out.size[0] - 2.0) < 0.1
    assert crop(tall(1, 1)).size == (1, 1)
    with pytest.raises(ValueError):
        AspectPreservingCrop((0.0, 1.0))
