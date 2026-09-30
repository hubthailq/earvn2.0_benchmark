"""PyTorch datasets built from the metadata tables."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import load_rgb


class ImageTable(Dataset):
    """Rows of a DataFrame with an ``image_id`` column (path relative to ``root``)."""

    def __init__(self, root: str | Path, image_ids: Sequence[str], labels: Sequence[int] | None,
                 transform: Callable):
        self.root = Path(root)
        self.ids = list(image_ids)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        if self.labels is not None and len(self.labels) != len(self.ids):
            raise ValueError("labels and image_ids have different lengths")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img = load_rgb(self.root / self.ids[i])
        x = self.transform(img)
        y = -1 if self.labels is None else int(self.labels[i])
        return x, y, i


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False,
                seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    # drop_last only if it leaves at least one batch
    drop_last = drop_last and len(ds) > batch_size
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last, generator=g,
                      persistent_workers=num_workers > 0)


def encode_labels(values: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Map string labels to 0..C-1 in a deterministic (sorted) order."""
    from .utils import natural_sorted
    classes = natural_sorted(sorted(set(map(str, values))))
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[str(v)] for v in values], dtype=np.int64), classes
