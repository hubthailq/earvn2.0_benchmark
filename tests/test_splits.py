"""Splits: exact counts, near-duplicate groups kept together, folds, few-shot sampling."""
import numpy as np
import pandas as pd
import pytest

from earbench.config import load_config
from earbench.splits import (_subset_exact, fold_roles, make_s1, make_s2, make_s3, sample_gallery,
                             sample_gender_episode, sample_identity_episode, split_subject)


def test_subset_exact():
    sizes = [5, 3, 7, 1, 2]
    idx = _subset_exact(sizes, 10)
    assert idx is not None and sum(sizes[i] for i in idx) == 10 and len(set(idx)) == len(idx)
    assert _subset_exact([4, 4], 5) is None
    assert _subset_exact([2, 2], 0) == []


def test_split_subject_keeps_groups_together():
    rng = np.random.default_rng(0)
    groups = np.array([g for g in range(40) for _ in range(1 + (g % 3))])  # sizes 1,2,3 repeated
    labels, warns = split_subject(np.arange(len(groups)), groups, 45, 15, rng)
    assert (labels == "train").sum() == 45 and (labels == "val").sum() == 15
    assert (labels == "test").sum() == len(groups) - 60
    for g in np.unique(groups):
        assert len(set(labels[groups == g])) == 1, "a near-duplicate group was split"
    assert warns == []


def test_split_subject_forced_split_is_reported():
    groups = np.array([0] * 50 + [1] * 50)  # impossible to get exactly 45 without splitting a group
    labels, warns = split_subject(np.arange(100), groups, 45, 15, np.random.default_rng(1))
    assert (labels == "train").sum() == 45 and (labels == "val").sum() == 15
    assert warns, "splitting a group must produce a warning"


def test_split_subject_too_few_images():
    with pytest.raises(ValueError):
        split_subject(np.arange(60), np.arange(60), 45, 15, np.random.default_rng(0))


def _images(n_subjects=3, n=95, orig=True):
    rows = []
    for s in range(n_subjects):
        for k in range(n):
            split = ("train" if k < 45 else "val" if k < 60 else "test") if orig else ""
            rows.append({"image_id": f"{s}/{k}.jpg", "subject": f"{s:03d}", "orig_split": split})
    return pd.DataFrame(rows)


def test_make_s1_keep_original_and_resplit():
    cfg = load_config()
    img = _images()
    groups = np.arange(len(img))
    s1, w = make_s1(img, groups, cfg.splits.s1, np.random.default_rng(0), keep_original=True)
    assert (s1.split == img.orig_split).all() and w.empty
    s1b, _ = make_s1(img, groups, cfg.splits.s1, np.random.default_rng(0), keep_original=False)
    for _, g in s1b.groupby("subject"):
        assert (g.split == "train").sum() == 45 and (g.split == "val").sum() == 15


def test_make_s1_bad_original_counts_are_resplit_and_small_subjects_excluded():
    cfg = load_config()
    img = pd.concat([_images(2), _images(1, n=50).assign(subject="small")], ignore_index=True)
    img.loc[0, "orig_split"] = "val"  # subject 000 now has 44 train / 16 val
    s1, w = make_s1(img, np.arange(len(img)), cfg.splits.s1, np.random.default_rng(0), keep_original=True)
    g0 = s1[s1.subject == "000"]
    assert (g0.split == "train").sum() == 45 and (g0.split == "val").sum() == 15
    assert (s1[s1.subject == "small"].split == "excluded").all()
    assert set(w.subject) == {"000", "small"}


def test_make_s2_folds_are_disjoint_and_stratified():
    subs = pd.DataFrame({"subject": [f"{i:03d}" for i in range(50)], "gender": ["M"] * 30 + ["F"] * 20})
    s2 = make_s2(subs, 5, np.random.default_rng(0))
    assert s2.subject.is_unique and len(s2) == 50
    tab = s2.groupby(["fold", "gender"]).size().unstack()
    assert (tab.M == 6).all() and (tab.F == 4).all()
    for k in range(5):
        r = fold_roles(s2.set_index("subject").fold, k, 5)
        assert (r == "test").sum() == 10 and (r == "val").sum() == 10 and (r == "train").sum() == 30


def test_make_s2_errors():
    one = pd.DataFrame({"subject": list("abcdef"), "gender": ["M"] * 6})
    with pytest.raises(ValueError):
        make_s2(one, 5, np.random.default_rng(0))
    few = pd.DataFrame({"subject": list("abcdefg"), "gender": ["M"] * 5 + ["F"] * 2})
    with pytest.raises(ValueError):
        make_s2(few, 5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        make_s2(few, 2, np.random.default_rng(0))


def test_make_s3():
    subs = pd.DataFrame({"subject": [str(i) for i in range(20)], "gender": ["M"] * 10 + ["F"] * 10})
    s3 = make_s3(subs, 0.3, np.random.default_rng(0))
    assert (s3.role == "novel").sum() == 6 and s3.subject.is_unique
    with pytest.raises(ValueError):
        make_s3(subs, 1.5, np.random.default_rng(0))


def test_identity_episode_excludes_near_duplicates_from_test():
    subj = np.repeat(np.arange(4), 20)
    groups = np.arange(80)
    groups[1] = groups[0]  # image 1 is a copy of image 0
    for seed in range(20):
        ep = sample_identity_episode(subj, groups, 3, 2, np.random.default_rng(seed))
        assert len(ep["support"]) == 12 and len(ep["val"]) == 8
        chosen = set(ep["support"]) | set(ep["val"])
        assert not chosen & set(ep["test"])
        if 0 in chosen or 1 in chosen:
            assert 0 not in ep["test"] and 1 not in ep["test"]
    a = sample_identity_episode(subj, groups, 3, 2, np.random.default_rng(5))
    b = sample_identity_episode(subj, groups, 3, 2, np.random.default_rng(5))
    assert all((a[k] == b[k]).all() for k in ("support", "val", "test"))


def test_identity_episode_errors_and_gallery():
    subj = np.repeat(np.arange(2), 5)
    with pytest.raises(ValueError):
        sample_identity_episode(subj, np.arange(10), 3, 2, np.random.default_rng(0))
    g = sample_gallery(subj, np.arange(10), 1, np.random.default_rng(0))
    assert len(g["gallery"]) == 2 and len(g["probe"]) == 8


def test_gender_episode_uses_distinct_subjects():
    subj = np.repeat(np.arange(10), 5)
    gender = np.where(subj < 5, 0, 1)
    pool = subj != 9
    for seed in range(10):
        idx = sample_gender_episode(subj, gender, pool, 4, np.random.default_rng(seed))
        assert len(idx) == 8 and len(set(subj[idx])) == 8 and pool[idx].all()
    with pytest.raises(ValueError):
        sample_gender_episode(subj, gender, pool, 5, np.random.default_rng(0))  # only 4 female subjects in pool
