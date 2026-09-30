"""Split construction (S1 identity, S2 gender folds, S3 new-identity) and few-shot sampling.

All functions are deterministic given a numpy Generator, and every constraint violation is either
raised as a clear error or returned as a warning row (never silently ignored).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd


# ============================================================================ S1: identity
def _subset_exact(sizes: Sequence[int], target: int) -> list[int] | None:
    """Indices of items whose sizes sum exactly to target (first solution in the given order)."""
    if target == 0:
        return []
    reach = {0: None}  # sum -> (prev_sum, item)
    for idx, s in enumerate(sizes):
        if s <= 0:
            continue
        for tot in sorted(reach.keys(), reverse=True):
            nt = tot + s
            if nt <= target and nt not in reach:
                reach[nt] = (tot, idx)
        if target in reach:
            break
    if target not in reach:
        return None
    out, cur = [], target
    while cur:
        prev, idx = reach[cur]
        out.append(idx)
        cur = prev
    return out[::-1]


def _best_below(sizes: Sequence[int], target: int) -> list[int]:
    """Largest achievable sum <= target (used when no exact subset exists)."""
    reach = {0: None}
    for idx, s in enumerate(sizes):
        for tot in sorted(reach.keys(), reverse=True):
            nt = tot + s
            if nt <= target and nt not in reach:
                reach[nt] = (tot, idx)
    best = max(reach)
    out, cur = [], best
    while cur:
        prev, idx = reach[cur]
        out.append(idx)
        cur = prev
    return out


def _take(groups: list, sizes: dict, target: int, rng: np.random.Generator):
    """Choose groups whose total size is exactly target; split one group only if unavoidable.
    Returns (chosen_groups, partial) where partial = (group, n_items_taken) or None."""
    order = list(groups)
    rng.shuffle(order)
    sz = [sizes[g] for g in order]
    pick = _subset_exact(sz, target)
    if pick is not None:
        return [order[i] for i in pick], None
    pick = _best_below(sz, target)
    chosen = [order[i] for i in pick]
    missing = target - sum(sizes[g] for g in chosen)
    rest = [g for g in order if g not in set(chosen)]
    # split the smallest group that is large enough (keeps the damage minimal)
    candidates = sorted([g for g in rest if sizes[g] > missing], key=lambda g: sizes[g])
    if not candidates:
        return chosen, None  # caller detects the shortfall
    return chosen, (candidates[0], missing)


def split_subject(image_idx: np.ndarray, groups: np.ndarray, n_train: int, n_val: int,
                  rng: np.random.Generator):
    """Assign one subject's images to train/val/test with exactly n_train / n_val images,
    keeping near-duplicate groups on one side whenever possible.
    Returns (labels array aligned with image_idx, list of warning strings)."""
    labels = np.array(["test"] * len(image_idx), dtype=object)
    warnings = []
    if len(image_idx) < n_train + n_val + 1:
        raise ValueError(f"subject has {len(image_idx)} images, needs >= {n_train + n_val + 1}")
    members = defaultdict(list)
    for pos, g in enumerate(groups):
        members[g].append(pos)
    sizes = {g: len(v) for g, v in members.items()}

    remaining = list(members)
    for split, target in (("train", n_train), ("val", n_val)):
        chosen, partial = _take(remaining, sizes, target, rng)
        for g in chosen:
            labels[members[g]] = split
        taken = sum(sizes[g] for g in chosen)
        remaining = [g for g in remaining if g not in set(chosen)]
        if partial is not None:
            g, k = partial
            pos = list(members[g])
            rng.shuffle(pos)
            labels[pos[:k]] = split
            members[g] = pos[k:]  # rest of the group stays available for the next split
            sizes[g] = len(members[g])
            taken += k
            warnings.append(f"near-duplicate group of size {k + sizes[g]} split across {split} and a later split")
        if taken != target:
            raise ValueError(f"could not reach {target} {split} images (got {taken})")
    return labels, warnings


def make_s1(images: pd.DataFrame, groups: np.ndarray, cfg_s1, rng: np.random.Generator,
            keep_original: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S1 identity split. images must contain columns image_id, subject, orig_split."""
    n_tr, n_va, n_min = int(cfg_s1.train), int(cfg_s1.val), int(cfg_s1.min_images)
    if n_min < n_tr + n_va + 1:
        raise ValueError("splits.s1.min_images must be > train + val")
    out = images[["image_id", "subject"]].copy()
    out["split"] = ""
    out["group"] = groups
    warn_rows = []
    for s, g in images.groupby("subject", sort=False):
        idx = g.index.to_numpy()
        if len(idx) < n_min:
            out.loc[idx, "split"] = "excluded"
            warn_rows.append({"subject": s, "warning": f"only {len(idx)} usable images (< {n_min}); excluded from S1"})
            continue
        orig = g.orig_split.fillna("").to_numpy()
        if keep_original and (orig == "train").sum() == n_tr and (orig == "val").sum() == n_va \
                and (orig == "test").sum() >= 1 and (orig != "").all():
            out.loc[idx, "split"] = orig
            continue
        if keep_original:
            warn_rows.append({"subject": s, "warning":
                              f"original split not usable (train={int((orig == 'train').sum())}, "
                              f"val={int((orig == 'val').sum())}, loose={int((orig == '').sum())}); re-split"})
        labels, w = split_subject(idx, groups[idx], n_tr, n_va, rng)
        out.loc[idx, "split"] = labels
        warn_rows += [{"subject": s, "warning": x} for x in w]
    return out, pd.DataFrame(warn_rows, columns=["subject", "warning"])


# ============================================================================ S2: gender folds
def make_s2(subjects: pd.DataFrame, n_folds: int, rng: np.random.Generator) -> pd.DataFrame:
    """Subject-disjoint folds, stratified by gender (round-robin after shuffling each gender)."""
    if n_folds < 3:
        raise ValueError("S2 needs at least 3 folds (train / val / test)")
    rows = []
    for gender, g in subjects.groupby("gender"):
        subs = g.subject.tolist()
        if len(subs) < n_folds:
            raise ValueError(f"gender '{gender}' has {len(subs)} subjects, fewer than {n_folds} folds")
        rng.shuffle(subs)
        start = int(rng.integers(n_folds))  # avoid fold 0 always receiving the extra subjects
        for i, s in enumerate(subs):
            rows.append({"subject": s, "gender": gender, "fold": (start + i) % n_folds})
    if len({r["gender"] for r in rows}) < 2:
        raise ValueError("S2 needs both genders present")
    return pd.DataFrame(rows).sort_values("subject", key=lambda c: c.map(str)).reset_index(drop=True)


def fold_roles(fold_of_subject: pd.Series, k: int, n_folds: int) -> pd.Series:
    """Role of each subject in fold k: test = fold k, val = fold k+1, train = the rest."""
    val_fold = (k + 1) % n_folds
    return fold_of_subject.map(lambda f: "test" if f == k else ("val" if f == val_fold else "train"))


# ============================================================================ S3: new identities
def make_s3(subjects: pd.DataFrame, novel_fraction: float, rng: np.random.Generator) -> pd.DataFrame:
    if not 0 < novel_fraction < 1:
        raise ValueError("splits.s3.novel_fraction must be in (0, 1)")
    rows = []
    for gender, g in subjects.groupby("gender"):
        subs = g.subject.tolist()
        rng.shuffle(subs)
        n_novel = int(round(len(subs) * novel_fraction))
        n_novel = min(max(n_novel, 1), len(subs) - 1) if len(subs) > 1 else 0
        rows += [{"subject": s, "gender": gender, "role": "novel" if i < n_novel else "train"}
                 for i, s in enumerate(subs)]
    df = pd.DataFrame(rows)
    if (df.role == "novel").sum() < 2 or (df.role == "train").sum() < 2:
        raise ValueError("S3 needs at least 2 train and 2 novel subjects")
    return df.reset_index(drop=True)


# ============================================================================ few-shot sampling
def _pick_distinct_groups(pos: np.ndarray, groups: np.ndarray, n: int, rng: np.random.Generator):
    """Pick n positions, one per near-duplicate group when possible."""
    order = rng.permutation(len(pos))
    chosen, used = [], set()
    for o in order:
        if groups[o] not in used:
            chosen.append(o)
            used.add(groups[o])
            if len(chosen) == n:
                return chosen, True
    for o in order:  # not enough distinct groups: allow repeats
        if o not in chosen:
            chosen.append(o)
            if len(chosen) == n:
                break
    return chosen, False


def sample_identity_episode(subject_codes: np.ndarray, groups: np.ndarray, k_shot: int, n_val: int,
                            rng: np.random.Generator) -> dict:
    """Per subject: k_shot support + n_val validation images from all images of that subject; the rest
    is test, EXCEPT images that are near-duplicates of a chosen support/val image (they are dropped,
    otherwise the test would contain copies of the support set)."""
    support, val, test = [], [], []
    degraded = no_test = 0
    for c in np.unique(subject_codes):
        pos = np.nonzero(subject_codes == c)[0]
        need = k_shot + n_val
        if len(pos) <= need:
            raise ValueError(f"subject code {c} has {len(pos)} images, needs > {need} for this episode")
        chosen, ok = _pick_distinct_groups(pos, groups[pos], need, rng)
        degraded += not ok
        chosen_pos = pos[chosen]
        support += chosen_pos[:k_shot].tolist()
        val += chosen_pos[k_shot:].tolist()
        blocked = set(groups[chosen_pos])
        # never test on a near-duplicate of a support/val image; a subject whose every remaining image is
        # such a copy simply has no test images in this episode (it stays in the gallery as a distractor)
        rest = [p for p in pos if groups[p] not in blocked]
        no_test += not rest
        test += rest
    return {"support": np.array(support, dtype=int), "val": np.array(val, dtype=int),
            "test": np.array(test, dtype=int),
            "subjects_without_distinct_groups": degraded, "subjects_without_test": no_test}


def sample_gallery(subject_codes: np.ndarray, groups: np.ndarray, k: int, rng: np.random.Generator) -> dict:
    """k gallery images per subject, all other images (not near-duplicates of the gallery) are probes."""
    ep = sample_identity_episode(subject_codes, groups, k, 0, rng)
    return {"gallery": ep["support"], "probe": ep["test"],
            "subjects_without_distinct_groups": ep["subjects_without_distinct_groups"],
            "subjects_without_test": ep["subjects_without_test"]}


def sample_gender_episode(image_subjects: np.ndarray, image_gender: np.ndarray, pool_mask: np.ndarray,
                          k_per_class: int, rng: np.random.Generator) -> np.ndarray:
    """k images per gender from the pool, each from a DIFFERENT subject."""
    chosen = []
    for gcls in np.unique(image_gender[pool_mask]):
        idx = np.nonzero(pool_mask & (image_gender == gcls))[0]
        subs = np.unique(image_subjects[idx])
        if len(subs) < k_per_class:
            raise ValueError(f"gender '{gcls}': only {len(subs)} subjects in the pool, need {k_per_class}")
        for s in rng.choice(subs, size=k_per_class, replace=False):
            cand = idx[image_subjects[idx] == s]
            chosen.append(int(rng.choice(cand)))
    return np.array(chosen)
