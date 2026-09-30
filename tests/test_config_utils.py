"""Config loading, utils (natural sort, robust image loading, result store)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from earbench.config import PROJECT_ROOT, load_config
from earbench.utils import ResultStore, atomic_write_csv, load_rgb, natural_sorted, read_csv


def test_config_defaults_and_overrides():
    cfg = load_config(overrides=["train.epochs=3", "preprocess.nonsquare_hw=[224, 96]", "device=cpu"])
    assert cfg.train.epochs == 3
    assert cfg.preprocess.nonsquare_hw == [224, 96]
    assert cfg.splits.s1.train == 45 and cfg.splits.s1.val == 15
    # relative paths are resolved against the project root
    assert cfg.paths.dataset_root == str((PROJECT_ROOT / "data/EarVN2.0").resolve())


@pytest.mark.parametrize("bad", ["train.epoch=3", "nosection.x=1", "noequals", "=3"])
def test_config_rejects_unknown_keys(bad):
    with pytest.raises((KeyError, ValueError)):
        load_config(overrides=[bad])


def test_natural_sort():
    assert natural_sorted(["10", "2", "1", "sub10", "sub2"]) == ["1", "2", "10", "sub2", "sub10"]
    assert natural_sorted(["001", "010", "002"]) == ["001", "002", "010"]


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK", "LA", "I;16", "1"])
def test_load_rgb_any_mode(tmp_path, mode):
    arr = (np.random.default_rng(0).random((20, 10, 3)) * 255).astype(np.uint8)
    im = Image.fromarray(arr).convert("RGB")
    if mode == "I;16":
        im = Image.fromarray((np.asarray(im.convert("L")).astype(np.uint16) * 257))
    else:
        im = im.convert(mode)
    p = tmp_path / ("x.tif" if mode in ("CMYK", "I;16") else "x.png")
    im.save(p)
    out = load_rgb(p)
    assert out.mode == "RGB" and out.size == (10, 20)


def test_load_rgb_exif_orientation(tmp_path):
    im = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = im.getexif()
    exif[0x0112] = 6
    p = tmp_path / "r.jpg"
    im.save(p, exif=exif)
    assert load_rgb(p).size == (20, 40)


def test_load_rgb_constant_16bit(tmp_path):
    p = tmp_path / "c.png"
    Image.fromarray(np.full((5, 5), 1000, dtype=np.uint16)).save(p)
    assert load_rgb(p).size == (5, 5)


def test_result_store_upsert_and_skip(tmp_path):
    st = ResultStore(tmp_path / "r.csv", ["model", "seed"])
    assert not st.has({"model": "a", "seed": 0})
    st.upsert({"model": "a", "seed": 0, "acc": 1.0})
    st.upsert({"model": "a", "seed": 1, "acc": 2.0})
    st.upsert({"model": "a", "seed": 0, "acc": 3.0})  # replaces
    df = st.load()
    assert len(df) == 2 and df[df.seed == 0].acc.item() == 3.0
    assert st.has({"model": "a", "seed": 1})
    with pytest.raises(KeyError):
        st.upsert({"model": "b"})


def test_atomic_write_and_read_keep_leading_zeros(tmp_path):
    atomic_write_csv(pd.DataFrame({"subject": ["001", "010"], "image_id": ["001/a.jpg", "010/b.jpg"]}),
                     tmp_path / "x.csv")
    df = read_csv(tmp_path / "x.csv")
    assert df.subject.tolist() == ["001", "010"]
    assert not list(tmp_path.glob(".tmp_*"))
    with pytest.raises(FileNotFoundError):
        read_csv(tmp_path / "missing.csv")


def test_result_store_reruns_when_settings_change(tmp_path):
    st = ResultStore(tmp_path / "r.csv", ["model"])
    assert not st.should_skip({"model": "a"}, False, lr=1e-4)
    st.upsert({"model": "a", "lr": 1e-4, "preprocess": "letterbox"})
    assert st.should_skip({"model": "a"}, False, lr=1e-4, preprocess="letterbox")
    assert not st.should_skip({"model": "a"}, False, lr=3e-4)            # LR sweep changed the LR
    assert not st.should_skip({"model": "a"}, False, preprocess="stretch")
    assert not st.should_skip({"model": "a"}, True, lr=1e-4)             # --overwrite
    assert st.get({"model": "zzz"}) is None
