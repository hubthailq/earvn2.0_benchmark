#!/usr/bin/env python
"""Run the benchmark stage by stage (cross-platform: Windows / Linux / macOS).

Stages follow the order agreed with the advisor:
  prepare   00-03   check env, scan, near-duplicates, label review file   -> then REVIEW qc/outliers_for_review.csv
  splits    03-04   apply the review, build S1/S2/S3
  basic     05-08   model info, LR sweep, E0, E1+E2 with the 5 basic models -> report to the advisor
  features  10-12   frozen features for all models, E3/E3b/E4, E5/E6
  full      08-15   E1+E2 for all models, S3 fine-tuning, E7, E8, resolution analysis
Every stage ends with 16_make_tables.py.  Finished runs are skipped, so a stage can be re-run safely.

    python run_all.py prepare
    python run_all.py splits
    python run_all.py basic
    python run_all.py features
    python run_all.py full
    python run_all.py basic --set device=cuda --set num_workers=8     # extra options go to every script
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S = ROOT / "scripts"

STAGES = {
    "prepare": [["00_check_env.py", "--download"], ["01_scan_dataset.py"], ["02_qc_duplicates.py"],
                ["03_qc_labels.py"]],
    "splits": [["03_qc_labels.py", "--apply"], ["04_make_splits.py"]],
    "basic": [["05_model_info.py", "--models", "all"], ["06_lr_sweep.py"], ["07_e0_preprocessing.py"],
              ["08_train_supervised.py", "--exp", "E1", "--models", "basic"],
              ["08_train_supervised.py", "--exp", "E2", "--models", "basic"]],
    "features": [["10_extract_features.py", "--what", "frozen", "--models", "all"],
                 ["11_fewshot.py", "--models", "all"], ["12_unsupervised.py", "--models", "all"]],
    "full": [["08_train_supervised.py", "--exp", "E1", "--models", "all"],
             ["08_train_supervised.py", "--exp", "E2", "--models", "all"],
             ["09_train_s3.py", "--models", "all"],
             ["10_extract_features.py", "--what", "s3ft", "--models", "all"],
             ["10_extract_features.py", "--what", "external", "--models", "all"],
             ["13_new_identity.py", "--models", "all"], ["14_cross_dataset.py", "--models", "all"],
             ["15_resolution_analysis.py"]],
}
NEXT = {
    "prepare": "Do the 4 required checks in README ('Bat buoc truoc khi chay'): (1) gender boundary printed above, "
               "(2) outputs/qc/threshold_review_dino.jpg + threshold_review_phash.jpg -> adjust qc thresholds and "
               "re-run scripts/02 if needed, (3) outputs/qc/outliers_for_review.csv (column 'remove' = y), "
               "(4) preprocess.nonsquare_hw from outputs/metadata/dataset_stats.md. Then run:  python run_all.py splits",
    "splits": "Check outputs/splits/splits_summary.md, then run:  python run_all.py basic",
    "basic": "Send outputs/tables/all_tables.md (E0, E1, E2 for the 5 basic models) to the advisor, then run:  "
             "python run_all.py features",
    "features": "Then run:  python run_all.py full",
    "full": "All experiments done. Final CSVs: outputs/results/ (column meanings in outputs/results/README.md); "
            "tables: outputs/tables/all_tables.md",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-download", action="store_true", help="prepare: do not pre-download pretrained weights")
    a = ap.parse_args()
    common = (["--config", a.config] if a.config else []) + [x for o in a.overrides for x in ("--set", o)]
    steps = STAGES[a.stage] + [["16_make_tables.py"]]
    for step in steps:
        if a.no_download and step[0] == "00_check_env.py":
            step = [step[0]]
        cmd = [sys.executable, str(S / step[0])] + step[1:] + common
        print(f"\n>>> {' '.join(cmd[1:])}", flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"\n!!! {step[0]} failed (exit code {rc}). Fix the problem and re-run the stage; "
                  f"finished work is skipped.", flush=True)
            sys.exit(rc)
    print(f"\n=== stage '{a.stage}' finished. {NEXT[a.stage]}")


if __name__ == "__main__":
    main()
