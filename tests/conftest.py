"""Shared fixtures: tiny synthetic datasets and configs pointing to temporary folders."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from make_synthetic_dataset import make  # noqa: E402

from earbench.config import load_config  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end tests that run every script (a few minutes on CPU)")


@pytest.fixture(scope="session")
def syn_presplit(tmp_path_factory):
    return make(tmp_path_factory.mktemp("data") / "presplit", n_subjects=8, n_images=95, layout="presplit",
                edge_cases=True, seed=1)


@pytest.fixture(scope="session")
def syn_flat(tmp_path_factory):
    return make(tmp_path_factory.mktemp("data") / "flat", n_subjects=8, n_images=95, layout="flat",
                edge_cases=False, seed=2)


def make_cfg(dataset_root, output_root, extra: dict | None = None):
    ov = [f"paths.dataset_root={dataset_root}", f"paths.output_root={output_root}", "gender.male_count=4",
          "pretrained=false", "device=cpu", "num_workers=0",
          f"paths.gender_labels={Path(output_root) / 'no_labels.csv'}"]
    ov += [f"{k}={v}" for k, v in (extra or {}).items()]
    return load_config(overrides=ov)


@pytest.fixture
def cfg_factory():
    return make_cfg
