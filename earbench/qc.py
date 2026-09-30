"""Quality control: near-duplicate detection (Q1), label outliers (Q2), overlap with EarVN1.0 (Q3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import load_rgb


# ----------------------------------------------------------------------------- hashing
def phash_int(path) -> int:
    import imagehash
    h = imagehash.phash(load_rgb(path), hash_size=8)
    bits = h.hash.flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    # fallback for numpy < 2.0
    x = x.astype(np.uint64)
    out = np.zeros(x.shape, dtype=np.uint8)
    for shift in range(0, 64, 8):
        out += _POP8[((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)]
    return out


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_pairs(hashes_a: np.ndarray, max_dist: int, hashes_b: np.ndarray | None = None,
                  block: int = 256) -> pd.DataFrame:
    """All pairs (i, j) with Hamming distance <= max_dist.
    Within one set (hashes_b None): only i < j.  Between two sets: every (i in a, j in b)."""
    a = np.asarray(hashes_a, dtype=np.uint64)
    same = hashes_b is None
    b = a if same else np.asarray(hashes_b, dtype=np.uint64)
    I, J, D = [], [], []
    for s in range(0, len(a), block):
        blk = a[s:s + block]
        d = _popcount64(blk[:, None] ^ b[None, :])
        ii, jj = np.nonzero(d <= max_dist)
        ii = ii + s
        if same:
            keep = jj > ii
            ii, jj = ii[keep], jj[keep]
        I.append(ii); J.append(jj); D.append(d[ii - s, jj])
    if not I:
        return pd.DataFrame({"i": [], "j": [], "dist": []}, dtype=int)
    return pd.DataFrame({"i": np.concatenate(I), "j": np.concatenate(J), "dist": np.concatenate(D)}).astype(int)


# ----------------------------------------------------------------------------- features
def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def cosine_pairs_within_groups(feats: np.ndarray, group_ids: np.ndarray, min_cos: float) -> pd.DataFrame:
    """Pairs (i<j) in the same group (subject) with cosine >= min_cos."""
    f = l2n(feats)
    I, J, C = [], [], []
    for g in pd.unique(group_ids):
        idx = np.nonzero(group_ids == g)[0]
        if len(idx) < 2:
            continue
        sim = f[idx] @ f[idx].T
        ii, jj = np.nonzero(np.triu(sim >= min_cos, k=1))
        I.append(idx[ii]); J.append(idx[jj]); C.append(sim[ii, jj])
    if not I:
        return pd.DataFrame({"i": pd.Series(dtype=int), "j": pd.Series(dtype=int), "cos": pd.Series(dtype=float)})
    return pd.DataFrame({"i": np.concatenate(I), "j": np.concatenate(J), "cos": np.concatenate(C)})


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def duplicate_groups(n: int, pairs: pd.DataFrame, subjects: np.ndarray) -> np.ndarray:
    """Group id per image. Only SAME-subject pairs are merged (cross-subject pairs are label errors,
    reported separately). Group id = smallest image index in the group."""
    uf = UnionFind(n)
    for i, j in zip(pairs.i.to_numpy(), pairs.j.to_numpy()):
        if subjects[i] == subjects[j]:
            uf.union(int(i), int(j))
    return np.array([uf.find(i) for i in range(n)], dtype=np.int64)


def leakage_report(split: pd.Series, groups: np.ndarray, subjects: np.ndarray) -> dict:
    """Share of test images that have a near-duplicate (same group) in train."""
    df = pd.DataFrame({"split": split.to_numpy(), "group": groups, "subject": subjects})
    train_groups = set(df.loc[df.split == "train", "group"])
    test = df[df.split == "test"]
    leaked = test.group.isin(train_groups)
    val = df[df.split == "val"]
    return {
        "n_test": int(len(test)), "n_test_with_train_duplicate": int(leaked.sum()),
        "test_leakage_rate": float(leaked.mean()) if len(test) else 0.0,
        "n_val_with_train_duplicate": int(val.group.isin(train_groups).sum()),
        "subjects_affected": int(test[leaked].subject.nunique()),
    }


def outlier_scores(feats: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    """Cosine of every image to its subject centroid and the within-subject z-score."""
    f = l2n(feats)
    cos = np.full(len(f), np.nan, dtype=np.float32)
    z = np.full(len(f), np.nan, dtype=np.float32)
    for s in pd.unique(subjects):
        idx = np.nonzero(subjects == s)[0]
        if len(idx) < 3:
            continue
        c = f[idx].mean(0)
        nrm = np.linalg.norm(c)
        if nrm == 0:
            continue
        cs = f[idx] @ (c / nrm)
        cos[idx] = cs
        sd = cs.std()
        z[idx] = 0.0 if sd < 1e-8 else (cs - cs.mean()) / sd
    return pd.DataFrame({"cos_to_centroid": cos, "z": z})


def subject_overlap(pairs: pd.DataFrame, subj_a: np.ndarray, subj_b: np.ndarray, min_matches: int) -> pd.DataFrame:
    """Subjects of dataset A with >= min_matches matched images to one subject of dataset B."""
    if pairs.empty:
        return pd.DataFrame(columns=["subject_earvn2", "subject_earvn1", "n_matched_images"])
    df = pd.DataFrame({"a": subj_a[pairs.i.to_numpy()], "b": subj_b[pairs.j.to_numpy()],
                       "img": pairs.i.to_numpy()})
    agg = df.groupby(["a", "b"]).img.nunique().reset_index(name="n_matched_images")
    agg = agg[agg.n_matched_images >= min_matches]
    return agg.rename(columns={"a": "subject_earvn2", "b": "subject_earvn1"}).sort_values(
        "n_matched_images", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- threshold calibration
def borderline_sample(pairs: pd.DataFrame, score_col: str, threshold: float, higher_is_duplicate: bool,
                      n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Pick ~n pairs closest to the threshold, half on each side, for a visual check of the threshold."""
    if pairs.empty:
        return pairs.assign(side=pd.Series(dtype=str))
    s = pairs[score_col].to_numpy(dtype=float)
    dup = s >= threshold if higher_is_duplicate else s <= threshold
    out = []
    for side, mask in (("counted_as_duplicate", dup), ("not_duplicate", ~dup)):
        part = pairs[mask].copy()
        part["_dist"] = np.abs(part[score_col].to_numpy(dtype=float) - threshold)
        part = part.sort_values("_dist").head(max(n // 2 * 3, 1))  # closest ones, then a random subset
        if len(part) > n // 2:
            part = part.iloc[np.sort(rng.choice(len(part), size=n // 2, replace=False))]
        out.append(part.drop(columns="_dist").assign(side=side))
    return pd.concat(out, ignore_index=True).sort_values(score_col, ascending=not higher_is_duplicate)


def contact_sheet(root, pairs: pd.DataFrame, score_col: str, path, tile: int = 112) -> None:
    """Save an image with one row per pair: image A | image B | score and decision."""
    from PIL import Image, ImageDraw

    from .preprocess import fit_to_canvas
    if pairs.empty:
        return
    text_w = 260
    sheet = Image.new("RGB", (2 * tile + text_w, tile * len(pairs)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, (_, p) in enumerate(pairs.iterrows()):
        for c, key in enumerate(("image_a", "image_b")):
            try:
                im = fit_to_canvas(load_rgb(root / p[key]), tile, tile, "letterbox", pad="mean",
                                   pad_rgb=(255, 255, 255))
            except Exception:
                im = Image.new("RGB", (tile, tile), (200, 0, 0))
            sheet.paste(im, (c * tile, r * tile))
        colour = (0, 120, 0) if p["side"] == "counted_as_duplicate" else (160, 0, 0)
        draw.text((2 * tile + 8, r * tile + 8), f"#{r}  {score_col} = {p[score_col]:.4g}", fill=(0, 0, 0))
        draw.text((2 * tile + 8, r * tile + 28), p["side"].replace("_", " "), fill=colour)
        draw.text((2 * tile + 8, r * tile + 48), str(p["subject_a"]), fill=(90, 90, 90))
        draw.line([(0, (r + 1) * tile - 1), (sheet.width, (r + 1) * tile - 1)], fill=(220, 220, 220))
    sheet.save(path, quality=90)


def identical_file_pairs(md5) -> pd.DataFrame:
    """Pairs (i < j) of byte-identical files (same md5; empty md5 = unreadable, ignored). A chain per md5
    value (a-b, b-c, ...) is enough for the union-find groups and keeps the number of pairs linear."""
    md5 = np.asarray(md5, dtype=object)
    pairs = []
    for key, idx in pd.Series(np.arange(len(md5))).groupby(md5):
        if key and len(idx) > 1:
            idx = np.sort(idx.to_numpy())
            pairs += list(zip(idx[:-1].tolist(), idx[1:].tolist()))
    return pd.DataFrame(pairs, columns=["i", "j"], dtype=int)
