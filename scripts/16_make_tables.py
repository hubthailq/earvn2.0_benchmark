#!/usr/bin/env python
"""16 - Build all result tables of the plan from outputs/results/per_run/*.csv.

Writes outputs/results/<experiment>.csv: one row per model (and split / shot / ...), every metric as
mean / std (/ ci95) over seeds, folds or episodes, plus outputs/results/README.md explaining each column.

Writes outputs/tables/<table>.md, <table>.tex and all_tables.md (everything in one file, ready to paste
into the report). Can be run at any time: missing results show as '–'."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earbench.common import setup  # noqa: E402
from earbench.models import load_registry  # noqa: E402
from earbench.summary import write_summaries  # noqa: E402
from earbench.tables import build_tables, write_tables  # noqa: E402


def main():
    args, cfg, paths, log = setup(__doc__)
    tabs = build_tables(load_registry(), paths.per_run, cfg)
    for extra in ("dataset_stats.md",):
        p = paths.metadata / extra
        if p.exists():
            (paths.tables / extra).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    out = write_tables(tabs, paths.tables)
    log.info(f"tables written to {paths.tables} (combined: {out})")
    files = write_summaries(paths.per_run, paths.results, load_registry())
    log.info(f"final results: {len(files)} CSV files in {paths.results} (column meanings: README.md there)")
    # fairness checks (README section 8.1): every model must be trained under the same conditions
    import pandas as pd
    flagged, bn_runs, diverged, seen = [], [], [], {}
    ident = ("model", "split", "run", "mode", "seed", "task")
    for f in sorted(paths.per_run.glob("*.csv")):
        if f.stat().st_size == 0 or f.stem in ("memory_plan", "model_info"):
            continue
        df = pd.read_csv(f)
        name = lambda r: f"{f.stem}: " + ", ".join(f"{k}={r[k]}" for k in ident if k in r and pd.notna(r[k]))  # noqa: E731
        if "status" in df.columns and f.stem != "lr_sweep":   # a diverging LR in the sweep is a normal outcome
            diverged += [name(r) for _, r in df[df.status.astype(str) == "diverged"].iterrows()]
        if "gpu_max_micro_batches" not in df.columns:
            continue
        planned = pd.to_numeric(df.get("planned_micro_batches", pd.Series(1, index=df.index)), errors="coerce").fillna(1)
        moved = df[[c for c in ("gpu_switches_to_fallback", "gpu_fallback_batches", "gpu_step_oom")
                    if c in df.columns]].fillna(0)
        bad = df[(moved.sum(axis=1) > 0) | (pd.to_numeric(df.gpu_max_micro_batches, errors="coerce").fillna(1) > planned)]
        flagged += [name(r) for _, r in bad.iterrows()]
        if "bn_micro_batch" in df.columns:
            bn_runs += [name(r) for _, r in df[df.bn_micro_batch.astype(str) == "True"].iterrows()]
        for col in ("gpu_name", "amp_dtype_used"):
            if col in df.columns:
                seen.setdefault(col, set()).update(df[col].dropna().astype(str))
    if diverged:
        log.warning(f"{len(diverged)} runs DIVERGED (loss NaN/inf, shown as '–' in the tables). Set a smaller LR for "
                    f"that family in train.learning_rate.by_family and run the same scripts again (the new LR "
                    f"re-runs exactly these runs; do NOT use --overwrite, it would retrain everything):\n  "
                    + "\n  ".join(diverged))
    if flagged:
        log.warning(f"{len(flagged)} runs left the GPU or needed more micro-batches than planned (another program "
                    f"was using the GPU); re-run them with --overwrite when the GPU is free:\n  " + "\n  ".join(flagged))
    if bn_runs:
        log.warning(f"{len(bn_runs)} runs of BatchNorm models used micro-batches (BN statistics on fewer images); "
                    f"mention it in the paper or lower train.batch_size for all models:\n  " + "\n  ".join(bn_runs))
    mixed = [f"{col}: {sorted(v)}" for col, v in seen.items() if len(v) > 1]
    if mixed:
        log.warning("results were produced on different GPUs / precisions (not comparable run-to-run):\n  "
                    + "\n  ".join(mixed))


if __name__ == "__main__":
    main()
