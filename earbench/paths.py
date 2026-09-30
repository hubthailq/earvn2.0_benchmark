"""Where every generated file lives. All scripts use these helpers so paths never drift."""
from __future__ import annotations

from pathlib import Path


class Paths:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.paths.output_root)

    def _d(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # --- metadata / QC / splits -------------------------------------------------
    @property
    def metadata(self) -> Path: return self._d("metadata")
    @property
    def images_csv(self) -> Path: return self.metadata / "images.csv"
    @property
    def subjects_csv(self) -> Path: return self.metadata / "subjects.csv"
    @property
    def scan_problems_csv(self) -> Path: return self.metadata / "scan_problems.csv"
    @property
    def stats_json(self) -> Path: return self.metadata / "dataset_stats.json"
    @property
    def stats_md(self) -> Path: return self.metadata / "dataset_stats.md"

    @property
    def qc(self) -> Path: return self._d("qc")
    @property
    def removed_csv(self) -> Path: return self.qc / "removed.csv"
    @property
    def dup_groups_csv(self) -> Path: return self.qc / "dup_groups.csv"
    @property
    def dup_pairs_csv(self) -> Path: return self.qc / "near_duplicate_pairs.csv"
    @property
    def leakage_json(self) -> Path: return self.qc / "leakage_report.json"
    @property
    def outliers_csv(self) -> Path: return self.qc / "outliers_for_review.csv"
    @property
    def overlap_earvn1_csv(self) -> Path: return self.qc / "overlap_earvn1.csv"
    @property
    def qc_summary_md(self) -> Path: return self.qc / "qc_summary.md"

    @property
    def splits(self) -> Path: return self._d("splits")
    @property
    def s1_csv(self) -> Path: return self.splits / "s1_identity.csv"
    @property
    def s2_csv(self) -> Path: return self.splits / "s2_gender_folds.csv"
    @property
    def s3_csv(self) -> Path: return self.splits / "s3_new_identity.csv"
    @property
    def split_warnings_csv(self) -> Path: return self.splits / "split_warnings.csv"

    # --- models / training ------------------------------------------------------
    @property
    def results(self) -> Path:
        """Final results: one summary CSV per experiment (mean / std / n of every metric per model)."""
        return self._d("results")

    @property
    def per_run(self) -> Path:
        """Raw results: one row per run / fold / episode (what the summaries are computed from)."""
        return self._d("results", "per_run")

    @property
    def model_info_csv(self) -> Path: return self.per_run / "model_info.csv"
    @property
    def memory_plan_csv(self) -> Path: return self.per_run / "memory_plan.csv"
    @property
    def lr_json(self) -> Path: return self.per_run / "chosen_lr.json"

    def run_dir(self, exp: str, model_key: str, run_name: str) -> Path:
        return self._d("runs", exp, model_key, run_name)

    def features_file(self, variant: str, model_key: str) -> Path:
        return self._d("features", variant) / f"{model_key}.npz"

    def results_csv(self, exp: str) -> Path:
        return self.per_run / f"{exp}.csv"

    @property
    def tables(self) -> Path: return self._d("tables")
