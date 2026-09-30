"""Configuration loading: YAML file + ``--set key=value`` command-line overrides."""
from __future__ import annotations

import argparse
import copy
import re
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"


class Config(dict):
    """A dict that also allows attribute access (``cfg.train.epochs``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"config has no key '{key}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict) and not isinstance(obj, Config):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config.wrap(v) for v in obj]
        return obj


class _Loader(yaml.SafeLoader):
    """YAML as people read it: '0300' is 300 (YAML 1.1 says octal 192), '1e-4' is a float (YAML 1.1 keeps a
    string), '1_000' is 1000. Everything else (quotes, bools, null, lists, dicts) is standard YAML.
    Used for configs/default.yaml and for --set values."""


_INT_TAG, _FLOAT_TAG = "tag:yaml.org,2002:int", "tag:yaml.org,2002:float"
_Loader.yaml_implicit_resolvers = {k: [(t, r) for t, r in v if t not in (_INT_TAG, _FLOAT_TAG)]
                                   for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
_Loader.add_implicit_resolver(_INT_TAG, re.compile(r"^[-+]?[0-9][0-9_]*$|^0x[0-9a-fA-F_]+$"),
                              list("-+0123456789"))
_Loader.add_implicit_resolver(
    _FLOAT_TAG,
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)(?:[eE][-+]?[0-9]+)?$"
               r"|^[-+]?\.(?:inf|Inf|INF)$|^\.(?:nan|NaN|NAN)$"),
    list("-+0123456789."))


def _construct_int(loader, node):
    text = loader.construct_scalar(node).replace("_", "")
    return int(text, 16) if text.lower().startswith("0x") else int(text, 10)


_Loader.add_constructor(_INT_TAG, _construct_int)


def yaml_load(text):
    return yaml.load(text, Loader=_Loader)


def _parse_value(text: str) -> Any:
    """Parse a --set value with the same YAML rules as the config file (see _Loader)."""
    try:
        return yaml_load(text.strip())
    except yaml.YAMLError:
        return text


def apply_override(cfg: dict, assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"--set expects key=value, got '{assignment}'")
    key, value = assignment.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ValueError(f"empty key in '{assignment}'")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"unknown config section '{p}' in '{assignment}'")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"unknown config key '{key}' (check spelling in configs/default.yaml)")
    node[parts[-1]] = Config.wrap(_parse_value(value.strip()))


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml_load(f) or {}
    cfg = Config.wrap(copy.deepcopy(raw))
    for ov in overrides or []:
        apply_override(cfg, ov)
    # resolve relative paths against the project root, so scripts work from any cwd
    for k, v in list(cfg.paths.items()):
        if v is not None and not os.path.isabs(str(v)):
            cfg.paths[k] = str((PROJECT_ROOT / v).resolve())
    for k in ("dataset_root", "earvn1_root", "awe_root"):
        if cfg.paths.get(k):
            cfg.paths[k] = descend_single_folder(cfg.paths[k])
    return cfg


def descend_single_folder(root: str) -> str:
    """A released dataset often looks like EarVN2.0/{Description.txt, Images/<subject folders>}. If the
    folder holds exactly ONE sub-folder (plus files) and that sub-folder holds several folders, the subject
    folders are one level down: use it, so the dataset can be copied into data/ exactly as it is."""
    r = Path(root)
    if not r.is_dir():
        return root
    hidden = lambda n: n.startswith(".") or n.startswith("__MACOSX")  # noqa: E731
    dirs = [d for d in r.iterdir() if d.is_dir() and not hidden(d.name)]
    if len(dirs) != 1:
        return root
    inner = [d for d in dirs[0].iterdir() if d.is_dir() and not hidden(d.name)]
    splits = {"train", "training", "val", "valid", "validation", "test", "testing"}
    if len(inner) < 2 or all(d.name.lower() in splits for d in inner):   # one subject with split folders
        return root
    return str(dirs[0])


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config value (repeatable)")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.overrides)


def dump_config(cfg: dict, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
