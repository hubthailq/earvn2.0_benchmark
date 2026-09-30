#!/usr/bin/env python
"""11 - Few-shot experiments on frozen features (run scripts/10_extract_features.py --what frozen first).

E3  identity few-shot (advisor: 5 train / 5 val / rest test): per subject K in fewshot.identity_shots
    support images + fewshot.identity_val val images drawn from ALL images of the subject; the rest is
    test (near-duplicates of the chosen images are dropped from test). Prototype + logistic regression.
E3b single gallery image per subject (advisor: "gom chung, chọn đại 1 hình train, còn lại test"):
    rank-1/5/10 and AUCMC, fewshot.single_gallery_runs random draws.
E4  gender few-shot: per S2 fold, K images per gender from K different training-fold subjects;
    test = all images of the test-fold subjects. Balanced accuracy.

The random episodes depend only on the seed (not on the model), so every model sees identical episodes.
Outputs: outputs/results/per_run/E3.csv, E3b.csv, E4.csv (one row per episode).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from earbench.common import group_codes, load_images, load_s1, setup  # noqa: E402
from earbench.features import load_features  # noqa: E402
from earbench.fewshot import linear_probe, prototype_predict  # noqa: E402
from earbench.metrics import gender_metrics, identification, identity_metrics  # noqa: E402
from earbench.models import select_models  # noqa: E402
from earbench.splits import fold_roles, sample_gallery, sample_gender_episode, sample_identity_episode  # noqa: E402
from earbench.tasks import gender_label, load_s2  # noqa: E402
from earbench.utils import atomic_write_csv, feature_fingerprint, fingerprint  # noqa: E402


def extra(p):
    p.add_argument("--exp", default="E3,E3b,E4")
    p.add_argument("--models", default="all")
    p.add_argument("--overwrite", action="store_true")


def replace_rows(path: Path, model: str, rows: list[dict]):
    """Replace ALL rows of this model (rows=[] just deletes them)."""
    old = pd.read_csv(path, dtype={"fingerprint": str}) if path.exists() and path.stat().st_size else pd.DataFrame()
    if len(old) and "model" in old.columns:
        old = old[old.model.astype(str) != model]
    atomic_write_csv(pd.concat([old, pd.DataFrame(rows)], ignore_index=True), path)


def done(path: Path, model: str, fp: str) -> bool:
    """Finished for this model with exactly the same inputs (features, split, groups, settings)?"""
    if not path.exists() or not path.stat().st_size:
        return False
    df = pd.read_csv(path, dtype={"fingerprint": str})
    rows = df[df.model.astype(str) == model]
    return bool(len(rows)) and "fingerprint" in rows.columns and bool((rows.fingerprint.astype(str) == fp).all())


def ep_rng(seed, *parts):
    return np.random.default_rng([int(seed)] + [int(p) for p in parts])


def class_metrics(scores, classes, y_true):
    """Top-1/5/10, subject-mean Top-1, macro/weighted precision-recall-F1 from class scores."""
    return identity_metrics(scores, np.searchsorted(np.asarray(classes), y_true))


def run_e3(feats, subj_codes, grp, cfg):
    rows = []
    fs = cfg.fewshot
    n_ep = int(fs.episodes_prototype)
    for k in fs.identity_shots:
        for ep in range(n_ep):
            e = sample_identity_episode(subj_codes, grp, int(k), int(fs.identity_val), ep_rng(cfg.seed, 3, k, ep))
            s, v, t = e["support"], e["val"], e["test"]
            info = {"n_test": len(t), "subjects_without_test": e["subjects_without_test"],
                    "subjects_without_distinct_groups": e["subjects_without_distinct_groups"]}
            _, sc, cls = prototype_predict(feats[s], subj_codes[s], feats[t])
            rows.append({"classifier": "proto", "shot": k, "episode": ep,
                         **class_metrics(sc, cls, subj_codes[t]), **info})
            if ep < int(fs.episodes_linear):
                _, proba, cls, C = linear_probe(feats[s], subj_codes[s], feats[v], subj_codes[v], feats[t],
                                                tuple(fs.linear_C_grid), int(fs.linear_max_iter))
                rows.append({"classifier": "lp", "shot": k, "episode": ep,
                             **class_metrics(proba, cls, subj_codes[t]), **info, "C": C})
    return rows


def run_e3b(feats, subj_codes, grp, cfg, cmc_out=None):
    rows, curves = [], []
    for r in range(int(cfg.fewshot.single_gallery_runs)):
        g = sample_gallery(subj_codes, grp, 1, ep_rng(cfg.seed, 31, r))
        res = identification(feats[g["probe"]], subj_codes[g["probe"]], feats[g["gallery"]], subj_codes[g["gallery"]])
        curves.append(res["cmc"])
        rows.append({"run": r, **{k: v for k, v in res.items() if k != "cmc"}, "n_probe": len(g["probe"]),
                     "subjects_without_test": g["subjects_without_test"]})
    if cmc_out is not None and curves:
        cmc_out.extend(100 * np.nanmean(np.stack(curves), axis=0))   # mean CMC over the random draws
    return rows


def run_e4(feats, img, s2, cfg, log=None):
    rows = []
    fs = cfg.fewshot
    n_folds = int(cfg.splits.s2.n_folds)
    y = gender_label(img.gender, cfg)
    subj = img.subject.to_numpy()
    fold_of = s2.set_index("subject").fold
    per_fold_proto = int(np.ceil(int(fs.episodes_prototype) / n_folds))
    per_fold_lp = int(np.ceil(int(fs.episodes_linear) / n_folds))
    for f in range(n_folds):
        role = img.subject.map(fold_roles(fold_of, f, n_folds)).to_numpy()
        pool, val, test = role == "train", np.nonzero(role == "val")[0], np.nonzero(role == "test")[0]
        rv = ep_rng(cfg.seed, 44, f)
        val_s = rv.choice(val, size=min(len(val), int(fs.gender_val_max_images)), replace=False) if len(val) else val
        for k in fs.gender_shots:
            n_min = min(len(np.unique(subj[pool & (y == c)])) for c in (0, 1))
            if n_min < int(k):
                if log:
                    log.warning(f"E4 fold {f}: only {n_min} training subjects for one gender, K={k} skipped")
                continue
            for ep in range(per_fold_proto):
                sup = sample_gender_episode(subj, y, pool, int(k), ep_rng(cfg.seed, 4, f, k, ep))
                pred, sc, cls = prototype_predict(feats[sup], y[sup], feats[test])
                prob = (sc[:, list(cls).index(1)] - sc[:, list(cls).index(0)] + 1) / 2
                m = gender_metrics(prob, y[test], subj[test])
                rows.append({"classifier": "proto", "shot": k, "fold": f, "episode": ep, **m})
                if int(k) in [int(x) for x in fs.gender_linear_shots] and ep < per_fold_lp:
                    pred, proba, cls, C = linear_probe(feats[sup], y[sup], feats[val_s], y[val_s], feats[test],
                                                       tuple(fs.linear_C_grid), int(fs.linear_max_iter))
                    m = gender_metrics(proba[:, list(cls).index(1)], y[test], subj[test])
                    rows.append({"classifier": "lp", "shot": k, "fold": f, "episode": ep, "C": C, **m})
    return rows


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    exps = [e.strip() for e in args.exp.split(",") if e.strip()]
    img_all = load_images(paths)
    s1 = load_s1(paths, img_all)
    ident = s1[s1.split != "excluded"].reset_index(drop=True)
    subj_codes = pd.factorize(ident.subject)[0]
    grp = group_codes(ident.group)
    s2 = load_s2(paths) if "E4" in exps else None
    for spec in select_models(args.models):
        path = paths.features_file("frozen", spec.key)
        if not path.exists():
            log.warning(f"skip {spec.key}: {path} missing (run script 10)")
            continue
        ids_all, feats_all, _ = load_features(path, list(img_all.image_id))
        pos = {k: i for i, k in enumerate(ids_all)}
        f_ident = feats_all[[pos[i] for i in ident.image_id]]
        feat_fp = feature_fingerprint(path)
        ident_fp = [ident.image_id, ident.subject, ident.group]
        e4_fp = [img_all.image_id, img_all.gender, s2[["subject", "fold"]].to_dict("list") if s2 is not None else None,
                 dict(cfg.splits.s2), dict(cfg.gender)]
        cmc = []
        for exp, fn, parts in (("E3", lambda: run_e3(f_ident, subj_codes, grp, cfg), ident_fp),
                               ("E3b", lambda: run_e3b(f_ident, subj_codes, grp, cfg, cmc), ident_fp),
                               ("E4", lambda: run_e4(feats_all, img_all, s2, cfg, log), e4_fp)):
            if exp not in exps:
                continue
            out = paths.results_csv(exp)
            fp = fingerprint(exp, feat_fp, dict(cfg.fewshot), int(cfg.seed), parts)
            cmc_ok = exp != "E3b" or done(paths.results_csv("E3b_cmc"), spec.key, fp)
            if done(out, spec.key, fp) and cmc_ok and not args.overwrite:
                log.info(f"skip {exp} {spec.key} (done)")
                continue
            rows = fn()
            if not rows:
                log.warning(f"{exp} {spec.key}: nothing could be evaluated (see warnings above)")
                replace_rows(out, spec.key, [])       # no stale rows from an earlier setting survive
                continue
            replace_rows(out, spec.key, [{"model": spec.key, **r, "fingerprint": fp} for r in rows])
            if exp == "E3b" and cmc:
                replace_rows(paths.results_csv("E3b_cmc"), spec.key,
                             [{"model": spec.key, "rank": k + 1, "cmc": float(v), "fingerprint": fp}
                              for k, v in enumerate(cmc)])
            df = pd.DataFrame(rows)
            if "subjects_without_test" in df.columns and df.subjects_without_test.max() > 0:
                log.warning(f"{exp} {spec.key}: up to {int(df.subjects_without_test.max())} subjects per episode had "
                            f"only near-duplicates of their support images left, so no test images")
            col = "bal_acc" if exp == "E4" else ("rank1" if exp == "E3b" else "top1")
            summary = df.groupby([c for c in ("classifier", "shot") if c in df.columns])[col].mean().round(2) \
                if exp != "E3b" else df[col].mean().round(2)
            log.info(f"{exp} {spec.key}: mean {col}\n{summary}")

if __name__ == "__main__":
    main()
