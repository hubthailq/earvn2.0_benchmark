#!/usr/bin/env python
"""13 - E7: recognising NEW people (subjects never seen during fine-tuning).

For the S3 'novel' subjects: K gallery images per person (retrieval.gallery_shots), all other images
are probes (near-duplicates of gallery images are dropped). Cosine matching, rank-1/5 and AUCMC,
retrieval.runs random draws.  Compared for two feature sets:
  finetuned : model fine-tuned on the S3 'train' subjects (script 09 + 10 --what s3ft)
  frozen    : the same pretrained model without fine-tuning (script 10 --what frozen)
Verification (1:1): EER and TAR@FAR on genuine pairs (same person, not near-duplicates) and sampled
impostor pairs. Outputs: outputs/results/per_run/E7.csv, E7_verification.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from earbench.common import group_codes, load_images, load_s1, setup  # noqa: E402
from earbench.features import load_features, s3_features_are_current  # noqa: E402
from earbench.metrics import identification, pair_scores, sample_pairs, verification  # noqa: E402
from earbench.models import select_models  # noqa: E402
from earbench.qc import l2n  # noqa: E402
from earbench.splits import sample_gallery  # noqa: E402
from earbench.tasks import load_s3  # noqa: E402
from earbench.utils import ResultStore, feature_fingerprint, fingerprint  # noqa: E402


def extra(p):
    p.add_argument("--models", default="all")
    p.add_argument("--overwrite", action="store_true")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    s1 = load_s1(paths, load_images(paths))
    s3 = load_s3(paths)
    novel = set(s3[s3.role == "novel"].subject)
    nv = s1[s1.subject.isin(novel) & (s1.split != "excluded")].reset_index(drop=True)
    ids = list(nv.image_id)
    subj = nv.subject.astype(str).to_numpy()
    grp = group_codes(nv.group)
    log.info(f"E7: {len(novel)} novel subjects, {len(ids)} images")

    rng_pairs = np.random.default_rng([int(cfg.seed), 7])
    (gi, gj), (ii, jj) = sample_pairs(subj, int(cfg.retrieval.max_impostor_pairs), rng_pairs)
    keep = grp[gi] != grp[gj]  # near-duplicates are not honest genuine pairs
    gi, gj = gi[keep], gj[keep]

    store = ResultStore(paths.results_csv("E7"), ["model", "variant", "shot", "run"])
    vstore = ResultStore(paths.results_csv("E7_verification"), ["model", "variant"])
    for spec in select_models(args.models):
        for variant in ("finetuned", "frozen"):
            path = paths.features_file("s3ft" if variant == "finetuned" else "frozen", spec.key)
            if not path.exists():
                log.warning(f"skip {spec.key} {variant}: {path} missing (run scripts 09/10)")
                continue
            if variant == "finetuned" and not s3_features_are_current(path, paths, spec.key):
                log.warning(f"skip {spec.key} finetuned: features belong to an older S3 checkpoint "
                            f"(re-run scripts/10_extract_features.py --what s3ft)")
                store.remove({"model": spec.key, "variant": variant})   # results of the old checkpoint
                vstore.remove({"model": spec.key, "variant": variant})
                continue
            _, f, _ = load_features(path, ids)
            f = l2n(f)
            fp = fingerprint(feature_fingerprint(path), dict(cfg.retrieval), int(cfg.seed), ids, nv.group)
            for shot in cfg.retrieval.gallery_shots:
                for r in range(int(cfg.retrieval.runs)):
                    key = {"model": spec.key, "variant": variant, "shot": int(shot), "run": r}
                    if store.should_skip(key, args.overwrite, log, fingerprint=fp):
                        continue
                    g = sample_gallery(subj, grp, int(shot), np.random.default_rng([int(cfg.seed), 71, int(shot), r]))
                    res = identification(f[g["probe"]], subj[g["probe"]], f[g["gallery"]], subj[g["gallery"]])
                    store.upsert({**key, **{k: v for k, v in res.items() if k != "cmc"}, "n_probe": len(g["probe"]),
                                  "subjects_without_test": g["subjects_without_test"], "fingerprint": fp})
            vkey = {"model": spec.key, "variant": variant}
            if not vstore.should_skip(vkey, args.overwrite, log, fingerprint=fp):
                gen = pair_scores(f, gi, gj)
                imp = pair_scores(f, ii, jj)
                v = verification(gen, imp, float(cfg.retrieval.far))
                vstore.upsert({**vkey, **v, "n_genuine": len(gen), "n_impostor": len(imp), "fingerprint": fp})
                log.info(f"E7 {spec.key} {variant}: EER={v['eer']:.2f} TAR@FAR={v['tar_at_far']:.2f}")


if __name__ == "__main__":
    main()
