#!/usr/bin/env python
"""02 - Q1: near-duplicate detection.

  1. pHash (64 bit) of every image; Hamming distance <= qc.phash_hamming_max -> near-duplicate
  2. DINOv2 features; same-subject cosine >= qc.dino_cosine_min -> near-duplicate
  3. union-find -> near-duplicate groups (same subject only)
  4. leakage of the ORIGINAL split (if the dataset already has train/val/test folders):
     share of test images with a near-duplicate in train -> keep (<= qc.leakage_threshold) or re-split

Outputs (outputs/qc/): phash.csv, dino_features.npz, near_duplicate_pairs.csv, dup_groups.csv,
leakage_report.json, qc_summary_q1.md, threshold_review_{dino,phash}.jpg/.csv

REQUIRED CHECK (README section "Bắt buộc trước khi chạy"): open qc/threshold_review_dino.jpg. Every row is
a pair of images of the same subject just above (green, counted as duplicate) or just below (red) the
threshold. Green rows must be copies of the same photo (same crop, maybe resized/recompressed); red rows
must be different photos. If green rows are different photos -> raise qc.dino_cosine_min; if red rows are
copies -> lower it. Same for threshold_review_phash.jpg with qc.phash_hamming_max. Re-run 02 after changing.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from earbench.common import load_images, setup  # noqa: E402
from earbench.features import extract, load_features, save_features  # noqa: E402
from earbench.models import ModelSpec, create_model, resolve_device  # noqa: E402
from earbench.qc import (borderline_sample, contact_sheet, cosine_pairs_within_groups,  # noqa: E402
                         duplicate_groups, hamming_pairs, identical_file_pairs, leakage_report, phash_int)
from earbench.utils import atomic_write_csv, read_csv  # noqa: E402


def extra(p):
    p.add_argument("--no-dino", action="store_true", help="use pHash only (no GPU / weights needed)")
    p.add_argument("--overwrite", action="store_true", help="recompute cached hashes and features")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    root = Path(cfg.paths.dataset_root)
    img = load_images(paths, with_groups=False)
    n = len(img)
    log.info(f"{n} usable images")

    md5 = img.md5.fillna("").astype(str).to_numpy()

    # ---- 1. pHash (cached per file content: a cached hash is reused only if the file's md5 is unchanged)
    ph_path = paths.qc / "phash.csv"
    cached = read_csv(ph_path) if (ph_path.exists() and not args.overwrite) else pd.DataFrame()
    if not {"image_id", "md5", "phash"} <= set(cached.columns):
        cached = pd.DataFrame(columns=["image_id", "md5", "phash"])
    known = {(i, m): h for i, m, h in zip(cached.image_id, cached.md5.astype(str), cached.phash)}
    todo = [(i, m) for i, m in zip(img.image_id, md5) if (i, m) not in known]
    for i, m in tqdm(todo, desc="pHash", disable=not todo):
        known[(i, m)] = format(phash_int(root / i), "016x")
    atomic_write_csv(pd.DataFrame({"image_id": img.image_id, "md5": md5,
                                   "phash": [known[(i, m)] for i, m in zip(img.image_id, md5)]}), ph_path)
    hashes = np.array([int(known[(i, m)], 16) for i, m in zip(img.image_id, md5)], dtype=np.uint64)

    # byte-identical files are always duplicates, whatever their size and whatever qc.pair_rule says
    exact = identical_file_pairs(md5)
    if len(exact):
        log.info(f"{len(exact)} byte-identical copies (same md5): always counted as duplicates")
    ph_thr = int(cfg.qc.phash_hamming_max)
    ph_all = hamming_pairs(hashes, ph_thr + 4)  # a few extra bits: borderline pairs for the threshold review
    small = img.min_side.to_numpy() < float(cfg.qc.phash_min_side)
    too_small = small[ph_all.i.to_numpy()] | small[ph_all.j.to_numpy()]
    n_before = int((ph_all.dist <= ph_thr).sum())
    ph_all = ph_all[~too_small].reset_index(drop=True)
    ph = ph_all[ph_all.dist <= ph_thr].reset_index(drop=True)
    log.info(f"pHash: {len(ph)} near-duplicate pairs (Hamming <= {cfg.qc.phash_hamming_max}; "
             f"{n_before - len(ph)} ignored because an image is < {cfg.qc.phash_min_side}px)")

    # ---- 2. DINOv2 (cached)
    subjects = img.subject.to_numpy()
    dn = pd.DataFrame({"i": pd.Series(dtype=int), "j": pd.Series(dtype=int), "cos": pd.Series(dtype=float)})
    if not args.no_dino:
        fpath = paths.qc / "dino_features.npz"
        content = hashlib.sha1("\n".join(f"{i}\t{m}" for i, m in zip(img.image_id, md5)).encode()).hexdigest()
        meta = {"model": cfg.qc.dino_model, "mode": "letterbox", "size": int(cfg.preprocess.size),
                "pad": cfg.preprocess.pad, "interpolation": cfg.preprocess.interpolation,
                "pretrained": bool(cfg.pretrained), "content": content}
        try:
            if args.overwrite:
                raise FileNotFoundError
            _, feats, _ = load_features(fpath, list(img.image_id), expect_meta=meta)
        except (FileNotFoundError, KeyError, ValueError):
            device = resolve_device(cfg.device)
            kw = {"img_size": int(cfg.preprocess.size)} if "dinov2" in cfg.qc.dino_model else {}
            spec = ModelSpec(key="qc_dino", name="DINOv2", family="qc", timm=cfg.qc.dino_model, kwargs=kw)
            model = create_model(spec, num_classes=0, pretrained=bool(cfg.pretrained))
            feats = extract(model, root, list(img.image_id), cfg, device, mode="letterbox")
            save_features(fpath, img.image_id, feats, meta)
        thr = float(cfg.qc.dino_cosine_min)
        dn_all = cosine_pairs_within_groups(feats, subjects, thr - float(cfg.qc.review_margin))
        dn = dn_all[dn_all.cos >= thr].reset_index(drop=True)
        log.info(f"DINOv2: {len(dn)} same-subject pairs with cosine >= {cfg.qc.dino_cosine_min}")

    # ---- 2b. threshold review: pairs just above / below each threshold, as CSV + picture
    rng = np.random.default_rng(cfg.seed)
    n_rev = int(cfg.qc.review_pairs)
    reviews = [("phash", ph_all.rename(columns={"dist": "phash_dist"}), "phash_dist", ph_thr, False)]
    if not args.no_dino:
        reviews.append(("dino", dn_all, "cos", float(cfg.qc.dino_cosine_min), True))
    for name, cand, col, t, higher in reviews:
        rev = borderline_sample(cand, col, t, higher, n_rev, rng)
        if rev.empty:
            log.info(f"threshold review {name}: no pairs near the threshold")
            continue
        rev["image_a"] = img.image_id.to_numpy()[rev.i.astype(int)]
        rev["image_b"] = img.image_id.to_numpy()[rev.j.astype(int)]
        rev["subject_a"] = subjects[rev.i.astype(int)]
        rev["subject_b"] = subjects[rev.j.astype(int)]
        rev = rev.drop(columns=["i", "j"]).reset_index(drop=True)
        rev.insert(0, "row", range(len(rev)))
        rev["is_duplicate_by_eye"] = ""
        atomic_write_csv(rev, paths.qc / f"threshold_review_{name}.csv")
        contact_sheet(root, rev, col, paths.qc / f"threshold_review_{name}.jpg")
        log.info(f"threshold review {name}: {len(rev)} borderline pairs -> qc/threshold_review_{name}.jpg / .csv")

    # ---- 3. merge pairs according to qc.pair_rule and build groups
    rule = str(cfg.qc.pair_rule).lower()
    if rule not in ("or", "and", "phash", "dino"):
        raise ValueError("qc.pair_rule must be one of: or, and, phash, dino")
    if args.no_dino and rule in ("and", "dino"):
        log.warning(f"qc.pair_rule={rule} needs DINOv2 but --no-dino was given: using pHash only")
        rule = "phash"
    how = {"or": "outer", "and": "inner", "phash": "left", "dino": "right"}[rule]
    pairs = pd.merge(ph.rename(columns={"dist": "phash_dist"}), dn, on=["i", "j"], how=how)
    pairs = pairs.merge(exact.assign(identical_file=True), on=["i", "j"], how="outer")
    pairs["identical_file"] = pairs["identical_file"].astype("boolean").fillna(False).astype(bool)
    pairs.loc[pairs.identical_file, "phash_dist"] = 0
    pairs["image_a"] = img.image_id.to_numpy()[pairs.i.astype(int)]
    pairs["image_b"] = img.image_id.to_numpy()[pairs.j.astype(int)]
    pairs["subject_a"] = subjects[pairs.i.astype(int)]
    pairs["subject_b"] = subjects[pairs.j.astype(int)]
    pairs["same_subject"] = pairs.subject_a == pairs.subject_b
    groups = duplicate_groups(n, pairs[["i", "j"]].astype(int), subjects)
    rep = img.image_id.to_numpy()[groups]
    atomic_write_csv(pd.DataFrame({"image_id": img.image_id, "subject": subjects, "group": rep}), paths.dup_groups_csv)
    atomic_write_csv(pairs.drop(columns=["i", "j"]).sort_values(["same_subject", "subject_a"]), paths.dup_pairs_csv)
    sizes = pd.Series(groups).value_counts()
    multi = sizes[sizes > 1]
    cross = pairs[~pairs.same_subject]
    per_subject = pd.Series(subjects).value_counts()
    big = [(img.subject[g], int(n), int(per_subject[img.subject[g]])) for g, n in multi.items()
           if n > float(cfg.qc.max_group_fraction) * per_subject[img.subject[g]]]
    if big:
        log.warning(f"{len(big)} near-duplicate groups hold > {100 * cfg.qc.max_group_fraction:.0f}% of their subject "
                    f"(e.g. subject {big[0][0]}: {big[0][1]}/{big[0][2]} images). Thresholds are probably too loose: "
                    f"inspect near_duplicate_pairs.csv and lower qc.phash_hamming_max / raise qc.dino_cosine_min "
                    f"or use qc.pair_rule=and.")

    # ---- 4. leakage of the original split
    report = {"n_images": n, "n_pairs": int(len(pairs)), "n_groups_size_gt1": int(len(multi)),
              "n_images_in_groups_size_gt1": int(multi.sum()), "largest_group": int(sizes.max()) if n else 0,
              "n_cross_subject_pairs": int(len(cross)), "threshold": float(cfg.qc.leakage_threshold),
              "pair_rule": rule, "n_oversized_groups": len(big)}
    has_orig = (img.orig_split != "").any()
    if has_orig:
        lk = leakage_report(img.orig_split, groups, subjects)
        report.update(lk)
        report["decision_s1"] = "keep_original" if lk["test_leakage_rate"] <= cfg.qc.leakage_threshold else "resplit"
    else:
        report["decision_s1"] = "resplit"  # no original split on disk
        report["note"] = "dataset has no train/val/test folders: S1 will be created by script 04"
    paths.leakage_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["## Q1 – Ảnh gần trùng", "",
             f"- Số cặp gần trùng: {report['n_pairs']} (khác người: {report['n_cross_subject_pairs']})",
             f"- Số nhóm gần trùng (≥ 2 ảnh): {report['n_groups_size_gt1']}, gồm {report['n_images_in_groups_size_gt1']} ảnh; nhóm lớn nhất: {report['largest_group']} ảnh"]
    if has_orig:
        lines.append(f"- Ảnh test S1 có ảnh gần trùng trong train: {report['n_test_with_train_duplicate']} / "
                     f"{report['n_test']} = {100 * report['test_leakage_rate']:.2f}% "
                     f"(ngưỡng {100 * cfg.qc.leakage_threshold:.1f}%) → **{report['decision_s1']}**")
    else:
        lines.append("- Dataset chưa có split sẵn → S1 sẽ được tạo mới theo nhóm (script 04)")
    (paths.qc / "qc_summary_q1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n" + "\n".join(lines))
    if len(cross):
        log.warning(f"{len(cross)} near-duplicate pairs between DIFFERENT subjects: likely label errors, "
                    f"reviewed in script 03")


if __name__ == "__main__":
    main()
