"""Final result files: outputs/results/<experiment>.csv, one row per model (and per split / shot / ...).

Every metric of the raw files (outputs/results/per_run/*.csv, one row per seed / fold / episode) is
summarised as <metric>_mean and <metric>_std (+ <metric>_ci95 for experiments made of random episodes),
with n_runs = how many runs the mean is taken over. Written by scripts/16_make_tables.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import mean_ci95
from .utils import atomic_write_csv

# raw file -> (summary file, grouping columns besides the model, add CI95?, description)
SUMMARIES = [
    ("E0", "E0_preprocessing", ["mode"], False, "Identity (S1) với 3 cách tiền xử lý: stretch / letterbox / nonsquare"),
    ("lr_sweep", "LR_sweep", ["lr"], False, "Quét learning rate: val Top-1 của mô hình đại diện mỗi họ"),
    ("E1", "E1_identity_supervised", ["split"], False, "Identity có giám sát, S1 (45/15/còn lại), qua các seed"),
    ("E2", "E2_gender_supervised", ["split"], False, "Gender có giám sát: S2 (5 fold, người không trùng) và S1"),
    ("E3", "E3_identity_fewshot", ["classifier", "shot"], True, "Identity few-shot trên đặc trưng đóng băng"),
    ("E3b", "E3b_single_gallery", [], True, "Một ảnh mẫu mỗi người, còn lại là ảnh truy vấn"),
    ("E4", "E4_gender_fewshot", ["classifier", "shot"], True, "Gender few-shot (K ảnh mỗi giới từ K người)"),
    ("E56", "E5_E6_unsupervised", ["task"], False, "Phân cụm K-means không giám sát (gender k=2, identity k=số người)"),
    ("E7", "E7_new_identity_retrieval", ["variant", "shot"], True, "Nhận dạng người mới (S3): truy hồi"),
    ("E7_verification", "E7_new_identity_verification", ["variant"], False, "Người mới (S3): xác thực 1:1"),
    ("E8", "E8_cross_dataset", ["dataset", "variant"], False, "Thử chéo dataset (leave-one-out)"),
    ("resolution", "resolution_analysis", ["bin"], False, "Top-1 E1 theo độ phân giải ảnh gốc"),
    ("S3_training", "S3_training", [], False, "Huấn luyện mô hình S3 (dùng cho E7/E8)"),
]

# numeric columns that describe a run rather than measure it
BOOKKEEPING = {"run", "seed", "fold", "episode", "restart", "lr", "C", "k", "planned_micro_batches",
               "subjects_without_test", "subjects_without_distinct_groups", "n_test", "n_probe", "n_probes",
               "n_genuine", "n_impostor", "n_excluded_subjects", "n_images", "params_m"}

GLOSSARY = [
    ("top1 / acc", "Accuracy (Top-1): tỉ lệ ảnh có dự đoán hạng 1 đúng. Với identity, top1 chính là accuracy"),
    ("top5, top10", "Tỉ lệ ảnh có người đúng nằm trong 5 / 10 dự đoán cao nhất"),
    ("top1_subject_mean", "Top-1 tính riêng từng người rồi lấy trung bình (mỗi người nặng như nhau)"),
    ("macro_precision / macro_recall / macro_f1", "Precision, recall, F1 tính cho từng lớp rồi lấy trung bình "
     "không trọng số (chỉ số chính khi lớp không cân bằng). macro_recall = balanced accuracy"),
    ("weighted_precision / weighted_recall / weighted_f1", "Như trên nhưng trọng số theo số ảnh của lớp"),
    ("bal_acc", "Balanced accuracy (gender): trung bình recall của nam và nữ"),
    ("f1", "Gender: macro-F1 của 2 lớp"),
    ("precision_male / recall_male / f1_male, ..._female", "Chỉ số của từng giới; recall_female = độ nhạy "
     "(sensitivity) với lớp nữ, recall_male = độ đặc hiệu (specificity)"),
    ("auc", "Diện tích dưới đường ROC (gender)"),
    ("mcc", "Matthews correlation coefficient × 100 (gender), bền với lớp lệch"),
    ("acc_subject", "Gender theo người: đa số phiếu của các ảnh của một người"),
    ("rank1 / rank5 / rank10", "Truy hồi: người đúng nằm trong k người giống nhất (CMC tại k)"),
    ("aucmc", "Diện tích dưới đường CMC (chuẩn hoá về %)"),
    ("mrr", "Mean reciprocal rank × 100"),
    ("eer", "Equal error rate (xác thực 1:1), càng thấp càng tốt"),
    ("tar_at_far", "True accept rate tại FAR = retrieval.far"),
    ("nmi / ari", "Normalized mutual information / adjusted Rand index (phân cụm)"),
    ("purity, homogeneity, completeness, v_measure", "Các chỉ số phân cụm khác"),
    ("test_* / val_*", "Chỉ số trên tập test / tập val (val chỉ dùng để chọn epoch)"),
    ("best_epoch, train_seconds", "Epoch được giữ lại (theo val) và thời gian huấn luyện"),
    ("*_mean / *_std / *_ci95", "Trung bình / độ lệch chuẩn / nửa khoảng tin cậy 95% qua n_runs lần chạy"),
    ("n_runs / n_diverged", "Số lần chạy được tính / số lần chạy phân kỳ (loss NaN) bị bỏ ra"),
]


def metric_columns(df: pd.DataFrame, keys: list[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in keys or c == "model" or c in BOOKKEEPING or c.startswith("gpu_"):
            continue
        if df[c].dtype == bool or c in ("channels_last", "bn_micro_batch"):
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().any():
            cols.append(c)
    # test metrics first (what the paper reports), then validation, then run facts (epoch, time)
    rank = lambda c: 2 if c in ("best_epoch", "train_seconds") else (1 if c.startswith("val_") else 0)  # noqa: E731
    return sorted(cols, key=lambda c: (rank(c), cols.index(c)))


def summarise(df: pd.DataFrame, keys: list[str], ci: bool, registry) -> pd.DataFrame:
    keys = [k for k in keys if k in df.columns]
    if df.empty or "model" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    diverged = df["status"].astype(str).eq("diverged") if "status" in df.columns else pd.Series(False, index=df.index)
    metrics = metric_columns(df, keys)
    for c in metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    info = {m.key: m for m in registry}
    order = {m.key: i for i, m in enumerate(registry)}
    rows = []
    for gkey, g in df.groupby(["model"] + keys, sort=False, dropna=False):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        spec = info.get(str(gkey[0]))
        ok = g[~diverged.loc[g.index]]
        r = {"model": gkey[0], "name": spec.name if spec else gkey[0], "family": spec.family if spec else "",
             **dict(zip(keys, gkey[1:])), "n_runs": int(len(ok))}
        if diverged.any():
            r["n_diverged"] = int(diverged.loc[g.index].sum())
        if "lr" in g.columns and "lr" not in keys:
            lrs = sorted({float(v) for v in pd.to_numeric(g.lr, errors="coerce").dropna()})
            r["lr"] = " ".join(f"{v:g}" for v in lrs)
        for c in metrics:
            v = ok[c].dropna().to_numpy(dtype=float)
            r[f"{c}_mean"] = round(float(v.mean()), 4) if len(v) else np.nan
            r[f"{c}_std"] = round(float(v.std(ddof=1)), 4) if len(v) > 1 else (0.0 if len(v) else np.nan)
            if ci:
                r[f"{c}_ci95"] = round(mean_ci95(v)[1], 4) if len(v) else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out["_o"] = out.model.map(order).fillna(len(order))
    return out.sort_values(["_o"] + keys, kind="stable").drop(columns="_o").reset_index(drop=True)


def write_summaries(per_run: str | Path, out_dir: str | Path, registry) -> list[Path]:
    """Write outputs/results/*.csv (+ README.md explaining every column). Returns the files written."""
    per_run, out_dir = Path(per_run), Path(out_dir)
    written = []

    def load(name):
        p = per_run / f"{name}.csv"
        if not p.exists() or p.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(p, keep_default_na=False, na_values=[""], dtype={"fingerprint": str})

    listing = []
    for raw, name, keys, ci, desc in SUMMARIES:
        df = summarise(load(raw), keys, ci, registry)
        path = out_dir / f"{name}.csv"
        if df.empty:
            path.unlink(missing_ok=True)
            continue
        atomic_write_csv(df, path)
        written.append(path)
        listing.append((name, desc, len(df)))
    # tables that are already one row per model / rank: copied with readable names
    names = {m.key: (m.name, m.family) for m in registry}
    for raw, name, desc in (("model_info", "model_info", "Số tham số (M), GMACs, độ trễ bs=1 của từng mô hình"),
                            ("memory_plan", "gpu_memory_plan", "Bộ nhớ GPU, số micro-batch, tốc độ train (ảnh/s)"),
                            ("E3b_cmc", "E3b_cmc_curves", "Đường CMC trung bình (rank, cmc %) để vẽ hình")):
        df = load(raw)
        path = out_dir / f"{name}.csv"
        if df.empty:
            path.unlink(missing_ok=True)
            continue
        df = df.drop(columns=[c for c in ("fingerprint",) if c in df.columns])
        if "name" not in df.columns:
            df.insert(1, "name", df.model.map(lambda k: names.get(str(k), (k, ""))[0]))
            df.insert(2, "family", df.model.map(lambda k: names.get(str(k), ("", ""))[1]))
        atomic_write_csv(df, path)
        written.append(path)
        listing.append((name, desc, len(df)))

    lines = ["# Kết quả benchmark EarVN2.0 (tự sinh bởi scripts/16_make_tables.py)", "",
             "Mỗi file CSV = một thí nghiệm, mỗi dòng = một mô hình (và một split / shot / ... nếu có). "
             "Mọi chỉ số tính bằng %. Kết quả thô từng lần chạy (seed / fold / episode) nằm trong `per_run/`.", "",
             "| File | Nội dung | Số dòng |", "|---|---|---|"]
    lines += [f"| `{n}.csv` | {d} | {k} |" for n, d, k in listing]
    lines += ["", "## Ý nghĩa các cột", "", "| Cột | Ý nghĩa |", "|---|---|"]
    lines += [f"| `{c}` | {d} |" for c, d in GLOSSARY]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written
