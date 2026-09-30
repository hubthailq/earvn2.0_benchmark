#!/usr/bin/env python
"""14 - Q3 + E8: overlap with EarVN1.0, then cross-dataset evaluation.

Q3  EarVN2.0 vs EarVN1.0 near-duplicates (pHash + DINOv2). An EarVN1.0 subject with >= qc.earvn1_min_matches
    matched images to one EarVN2.0 subject is considered the same person -> outputs/qc/overlap_earvn1.csv.
    These EarVN1.0 subjects are removed before E8 (otherwise E8 would test on people seen in training).
E8  all-vs-all identification (UERC style: each image is a probe against all other images) on EarVN1.0 and
    AWE, with the S3 fine-tuned model and the frozen pretrained model (script 10 --what external).
Outputs: outputs/qc/overlap_earvn1.csv, outputs/results/per_run/E8.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from earbench.common import load_images, setup  # noqa: E402
from earbench.features import extract, load_features, s3_features_are_current, save_features  # noqa: E402
from earbench.metrics import leave_one_out_identification  # noqa: E402
from earbench.models import ModelSpec, create_model, resolve_device, select_models  # noqa: E402
from earbench.qc import hamming_pairs, l2n, phash_int, subject_overlap  # noqa: E402
from earbench.scan import scan_external  # noqa: E402
from earbench.utils import ResultStore, atomic_write_csv, feature_fingerprint, fingerprint, read_csv  # noqa: E402


def extra(p):
    p.add_argument("--models", default="all")
    p.add_argument("--q3-only", action="store_true", help="only compute the overlap with EarVN1.0")
    p.add_argument("--no-dino", action="store_true", help="Q3 with pHash only")
    p.add_argument("--overwrite", action="store_true")


def cross_cosine_pairs(fa, fb, min_cos, block=1024):
    fa, fb = l2n(fa), l2n(fb)
    I, J = [], []
    for s in range(0, len(fa), block):
        sim = fa[s:s + block] @ fb.T
        i, j = np.nonzero(sim >= min_cos)
        I.append(i + s); J.append(j)
    return pd.DataFrame({"i": np.concatenate(I) if I else [], "j": np.concatenate(J) if J else []}).astype(int)


def q3_overlap(cfg, paths, log, no_dino):
    root1 = Path(cfg.paths.earvn1_root)
    if not root1.exists():
        log.info(f"EarVN1.0 not found at {root1}: Q3 skipped")
        return set()
    img2 = load_images(paths, with_groups=False)
    ext = paths.metadata / "external_earvn1.csv"
    e1 = read_csv(ext) if ext.exists() else scan_external(root1, cfg)
    atomic_write_csv(e1, ext)
    ph2 = read_csv(paths.qc / "phash.csv")
    h2map = dict(zip(ph2.image_id, ph2.phash))
    if any(i not in h2map for i in img2.image_id):
        raise FileNotFoundError("EarVN2.0 pHashes incomplete: run scripts/02_qc_duplicates.py first")
    h2 = np.array([int(h2map[i], 16) for i in img2.image_id], dtype=np.uint64)
    p1 = paths.qc / "phash_earvn1.csv"
    c1 = read_csv(p1) if p1.exists() else pd.DataFrame(columns=["image_id", "phash"])
    known = dict(zip(c1.image_id, c1.phash))
    for i in tqdm([x for x in e1.image_id if x not in known], desc="pHash EarVN1.0"):
        known[i] = format(phash_int(root1 / i), "016x")
    atomic_write_csv(pd.DataFrame({"image_id": list(known), "phash": list(known.values())}), p1)
    h1 = np.array([int(known[i], 16) for i in e1.image_id], dtype=np.uint64)
    pairs = hamming_pairs(h2, int(cfg.qc.phash_hamming_max), h1)[["i", "j"]]
    log.info(f"Q3 pHash: {len(pairs)} matched image pairs")
    if not no_dino and (paths.qc / "dino_features.npz").exists():
        _, f2, _ = load_features(paths.qc / "dino_features.npz", list(img2.image_id))
        fp = paths.qc / "dino_features_earvn1.npz"
        meta = {"model": cfg.qc.dino_model, "mode": "letterbox", "size": int(cfg.preprocess.size),
                "pad": cfg.preprocess.pad, "interpolation": cfg.preprocess.interpolation,
                "pretrained": bool(cfg.pretrained)}
        try:
            _, f1, _ = load_features(fp, list(e1.image_id), expect_meta=meta)
        except (FileNotFoundError, KeyError, ValueError):
            kw = {"img_size": int(cfg.preprocess.size)} if "dinov2" in cfg.qc.dino_model else {}
            m = create_model(ModelSpec("qc", "qc", "qc", cfg.qc.dino_model, kwargs=kw), 0, bool(cfg.pretrained))
            f1 = extract(m, root1, list(e1.image_id), cfg, resolve_device(cfg.device), mode="letterbox")
            save_features(fp, e1.image_id, f1, meta)
        dp = cross_cosine_pairs(f2, f1, float(cfg.qc.dino_cosine_min))
        log.info(f"Q3 DINOv2: {len(dp)} matched image pairs")
        pairs = pd.concat([pairs, dp]).drop_duplicates()
    ov = subject_overlap(pairs, img2.subject.astype(str).to_numpy(), e1.subject.astype(str).to_numpy(),
                         int(cfg.qc.earvn1_min_matches))
    atomic_write_csv(ov, paths.overlap_earvn1_csv)
    log.info(f"Q3: {ov.subject_earvn1.nunique() if len(ov) else 0} EarVN1.0 subjects overlap with EarVN2.0 "
             f"-> {paths.overlap_earvn1_csv}")
    return set(ov.subject_earvn1.astype(str)) if len(ov) else set()


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    overlap = q3_overlap(cfg, paths, log, args.no_dino)
    if args.q3_only:
        return
    store = ResultStore(paths.results_csv("E8"), ["model", "variant", "dataset"])
    for name in ("earvn1", "awe"):
        meta_csv = paths.metadata / f"external_{name}.csv"
        if not meta_csv.exists():
            log.info(f"E8 {name}: no features (run scripts/10_extract_features.py --what external)")
            continue
        ext = read_csv(meta_csv)
        excl = overlap if name == "earvn1" else set()
        keep = ~ext.subject.astype(str).isin(excl)
        ids = list(ext.image_id[keep])
        subj = ext.subject.astype(str)[keep].to_numpy()
        log.info(f"E8 {name}: {len(set(subj))} subjects, {len(ids)} images (excluded {len(excl)} overlapping subjects)")
        for spec in select_models(args.models):
            for variant, folder in (("finetuned", f"s3ft_{name}"), ("frozen", f"frozen_{name}")):
                key = {"model": spec.key, "variant": variant, "dataset": name}
                path = paths.features_file(folder, spec.key)
                if not path.exists():
                    continue
                if variant == "finetuned" and not s3_features_are_current(path, paths, spec.key):
                    log.warning(f"skip E8 {spec.key} finetuned {name}: features belong to an older S3 checkpoint "
                                f"(re-run scripts/10_extract_features.py --what external)")
                    store.remove(key)
                    continue
                fp = fingerprint(feature_fingerprint(path), ids, sorted(excl))
                if store.should_skip(key, args.overwrite, log, fingerprint=fp):
                    continue
                _, f, _ = load_features(path, ids)
                res = leave_one_out_identification(f, subj, ranks=(1, 5))
                store.upsert({**key, **{k: v for k, v in res.items() if k != "cmc"},
                              "n_excluded_subjects": len(excl), "fingerprint": fp})
                log.info(f"E8 {name} {spec.key} {variant}: rank1={res['rank1']:.2f} rank5={res['rank5']:.2f}")


if __name__ == "__main__":
    main()
