"""Model registry (configs/models.yaml) and timm helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from .config import MODELS_CONFIG


@dataclass
class ModelSpec:
    key: str
    name: str
    family: str
    timm: str
    square_only: bool = False
    basic: bool = False
    lr_rep: bool = False
    kwargs: dict = field(default_factory=dict)


def load_registry(path: str | Path = MODELS_CONFIG) -> list[ModelSpec]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    specs = [ModelSpec(**{k: v for k, v in m.items()}) for m in raw["models"]]
    keys = [s.key for s in specs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate model keys in {path}")
    return specs


def select_models(selector: str | None, registry: list[ModelSpec] | None = None) -> list[ModelSpec]:
    """'all' | 'basic' | 'lr_rep' | comma-separated keys (order kept as in models.yaml)."""
    registry = registry or load_registry()
    sel = (selector or "all").strip()
    if sel == "all":
        return registry
    if sel == "basic":
        return [m for m in registry if m.basic]
    if sel == "lr_rep":
        return [m for m in registry if m.lr_rep]
    wanted = [s.strip() for s in sel.split(",") if s.strip()]
    by_key = {m.key: m for m in registry}
    unknown = [w for w in wanted if w not in by_key]
    if unknown:
        raise KeyError(f"unknown model key(s) {unknown}; valid keys: {list(by_key)}")
    return [m for m in registry if m.key in set(wanted)]


def families(registry: list[ModelSpec]) -> list[str]:
    seen = []
    for m in registry:
        if m.family not in seen:
            seen.append(m.family)
    return seen


def create_model(spec: ModelSpec, num_classes: int, pretrained: bool = True):
    import timm
    kwargs = dict(spec.kwargs or {})
    try:
        return timm.create_model(spec.timm, pretrained=pretrained, num_classes=num_classes, **kwargs)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                f"Could not create/download pretrained '{spec.timm}'. Check the internet connection "
                f"(weights come from the Hugging Face hub) or set HF_HUB_OFFLINE=1 with a filled cache. "
                f"Original error: {exc}") from exc
        raise


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model: torch.nn.Module, input_hw=(224, 224)) -> float:
    """Multiply-accumulates (GMACs) for one image. Most papers report this number as 'GFLOPs'."""
    from torch.utils.flop_counter import FlopCounterMode
    model = model.eval()
    x = torch.zeros(1, 3, *input_hw, device=next(model.parameters()).device)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2 / 1e9


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device(pref)
