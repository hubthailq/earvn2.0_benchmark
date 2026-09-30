#!/usr/bin/env python
"""12 - E5 / E6: unsupervised (no labels used for learning).

K-means on L2-normalised frozen features of ALL usable images:
  E6 gender   : k = 2
  E5 identity : k = number of subjects
Clusters are matched to the true labels with the Hungarian algorithm (ACC); NMI and ARI are also reported.
Repeated unsupervised.kmeans_restarts times with different seeds.   Output: outputs/results/per_run/E56.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from earbench.common import load_images, setup  # noqa: E402
from earbench.features import load_features  # noqa: E402
from earbench.fewshot import kmeans_labels  # noqa: E402
from earbench.metrics import clustering_metrics  # noqa: E402
from earbench.models import select_models  # noqa: E402
from earbench.utils import ResultStore, feature_fingerprint, fingerprint  # noqa: E402


def extra(p):
    p.add_argument("--models", default="all")
    p.add_argument("--tasks", default="gender,identity")
    p.add_argument("--overwrite", action="store_true")


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    img = load_images(paths)
    labels = {"gender": img.gender.astype(str).to_numpy(), "identity": img.subject.astype(str).to_numpy()}
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    store = ResultStore(paths.results_csv("E56"), ["model", "task", "restart"])
    for spec in select_models(args.models):
        path = paths.features_file("frozen", spec.key)
        if not path.exists():
            log.warning(f"skip {spec.key}: {path} missing (run script 10)")
            continue
        _, feats, _ = load_features(path, list(img.image_id))
        fp = fingerprint(feature_fingerprint(path), dict(cfg.unsupervised), int(cfg.seed), img.image_id,
                         labels["gender"].tolist(), labels["identity"].tolist())
        for task in tasks:
            y = labels[task]
            k = len(pd.unique(y))
            for r in range(int(cfg.unsupervised.kmeans_restarts)):
                key = {"model": spec.key, "task": task, "restart": r}
                if store.should_skip(key, args.overwrite, log, fingerprint=fp):
                    continue
                uc = cfg.unsupervised
                pred = kmeans_labels(feats, k, seed=int(cfg.seed) + r, minibatch_above=int(uc.minibatch_above),
                                     max_iter=int(uc.kmeans_max_iter), minibatch_size=int(uc.minibatch_size),
                                     minibatch_n_init=int(uc.minibatch_n_init))
                m = clustering_metrics(y, pred)
                store.upsert({**key, "k": k, **m, "fingerprint": fp})
                log.info(f"E56 {spec.key} {task} restart {r}: ACC={m['acc']:.2f} NMI={m['nmi']:.2f} ARI={m['ari']:.2f}")


if __name__ == "__main__":
    main()
