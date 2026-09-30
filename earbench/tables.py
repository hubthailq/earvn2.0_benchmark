"""Turn results/per_run/*.csv into the tables of the plan (Markdown + LaTeX). Missing runs show as '–'."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import mean_ci95, mean_std

DASH = "–"


def fmt(values, kind: str = "std", nd: int = 2, expected: int | None = None) -> str:
    """'mean ± std' (kind=std), 'mean ± ci95' (kind=ci) or a single value. When fewer values than
    ``expected`` exist (seeds / folds / episodes not run yet, or diverged), '(n=…)' is appended so a
    partial average is never mistaken for the full one."""
    vals = [v for v in pd.to_numeric(pd.Series(values), errors="coerce") if pd.notna(v)]
    if not vals:
        return DASH
    m, s = (mean_ci95 if kind == "ci" else mean_std)(vals)
    out = f"{m:.{nd}f}" if len(vals) == 1 else f"{m:.{nd}f} ± {s:.{nd}f}"
    if expected and len(vals) < int(expected):
        out += f" (n={len(vals)}/{int(expected)})"
    return out


def _load(results_dir: Path, name: str) -> pd.DataFrame:
    p = results_dir / f"{name}.csv"
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p, keep_default_na=False, na_values=[""])


def _sel(df: pd.DataFrame, **conds) -> pd.DataFrame:
    if df.empty:
        return df
    m = pd.Series(True, index=df.index)
    for k, v in conds.items():
        if k not in df.columns:
            return df.iloc[0:0]
        m &= df[k].astype(str) == str(v)
    return df[m]


def _col(df: pd.DataFrame, c: str):
    return df[c] if (not df.empty and c in df.columns) else []


def hyperparameter_table(cfg, registry, chosen: dict, plan: pd.DataFrame | None = None) -> pd.DataFrame:
    """The fine-tuning recipe as used (for the 'Implementation details' section of the paper)."""
    tc, pre, fs = cfg.train, cfg.preprocess, cfg.fewshot
    o, s = tc.optimizer, tc.scheduler
    if str(o.name).lower() == "adamw":
        opt = f"AdamW (β = {tuple(o.betas)}, ε = {o.eps:g}), weight decay {o.weight_decay:g}"
    else:
        opt = f"SGD momentum {o.momentum:g}{' Nesterov' if o.nesterov else ''}, weight decay {o.weight_decay:g}"
    if o.get("no_decay_bias_norm", True):
        opt += " (không áp dụng cho bias/norm)"
    sch = {"cosine": f"cosine, warm-up tuyến tính {s.warmup_epochs:g} epoch, LR cuối = {s.min_lr_ratio:g} × LR",
           "step": f"step ×{s.step_gamma:g} tại epoch {list(s.step_epochs)}, warm-up {s.warmup_epochs:g} epoch",
           "constant": f"hằng số, warm-up {s.warmup_epochs:g} epoch"}.get(str(s.name).lower(), str(s.name))
    rows = [
        ("Khởi tạo", "trọng số pretrained timm (tag trong configs/models.yaml)" if cfg.get("pretrained", True)
         else "ngẫu nhiên (chỉ để chạy thử)"),
        ("Tiền xử lý", f"{pre.mode}, {pre.size}×{pre.size}, đệm '{pre.pad}', nội suy {pre.interpolation}, "
                       f"chuẩn hoá mean/std riêng từng mô hình"),
        ("Augmentation (train)", f"crop giữ tỉ lệ {list(pre.augment.scale)}, xoay ±{pre.augment.rotate_deg}°, "
                                 f"dịch ±{pre.augment.translate:g}, color jitter {list(pre.augment.color_jitter)}, "
                                 f"lật ngang = {bool(pre.augment.hflip)}"),
        ("Optimizer", opt),
        ("Lịch learning rate", sch),
        ("Số epoch tối đa / early stopping", f"{tc.epochs} / patience {tc.patience} (theo Top-1 hoặc balanced acc trên val)"),
        ("Batch size (train / eval)", f"{tc.batch_size} / {tc.get('eval_batch_size', 2 * tc.batch_size)}"),
        ("Loss", f"cross-entropy, label smoothing {tc.label_smoothing:g}"
                 + (", trọng số lớp nghịch tần suất (gender)" if tc.gender_balanced_loss else "")),
        ("Gradient clipping", f"{tc.grad_clip:g}" if tc.grad_clip else "không"),
        ("Mixed precision", f"{'có' if tc.amp else 'không'} ({tc.get('amp_dtype', 'auto')}; train, val/test và "
                            f"trích đặc trưng dùng cùng một độ chính xác cho mọi mô hình)"),
        ("channels_last (chỉ đổi bố cục bộ nhớ)", ", ".join(tc.get("channels_last_families") or []) or "không"),
        ("Seed", f"{list(tc.seeds)} (seed chia dữ liệu và episode: {cfg.seed})"),
        ("Quét learning rate", f"{list(tc.learning_rate.sweep_grid)}, {tc.learning_rate.sweep_epochs} epoch, "
                               f"mô hình nhỏ nhất mỗi họ, chọn theo Top-1 val S1"),
        ("Few-shot", f"{fs.episodes_prototype} episode (prototype), {fs.episodes_linear} (logistic regression, "
                     f"C ∈ {list(fs.linear_C_grid)})"),
    ]
    fixed = tc.learning_rate.get("by_family") or {}
    for fam in dict.fromkeys(m.family for m in registry):
        if fam in fixed:
            lr = f"{float(fixed[fam]):g} (cố định trong config)"
        elif fam in chosen:
            lr = f"{float(chosen[fam]):g} (từ quét LR)"
        else:
            lr = f"{float(tc.learning_rate.default):g} (mặc định, chưa quét)"
        rows.append((f"Learning rate – {fam}", lr))
    if plan is not None and len(plan):
        rows.append(("GPU", ", ".join(sorted(set(plan.gpu_name.astype(str))))))
        mb = pd.to_numeric(plan.micro_batches, errors="coerce")
        acc = plan[mb > 1]
        rows.append(("Tích luỹ gradient (batch hiệu dụng giữ nguyên)",
                     ", ".join(f"{r.model} ({int(r.micro_batches)} × {int(r.micro_batch_size)})"
                               for r in acc.itertuples()) or "không mô hình nào cần"))
        bn = plan[plan.bn_micro_batch.astype(str) == "True"]
        if len(bn):
            rows.append(("BatchNorm trên micro-batch (lưu ý khi so sánh)", ", ".join(bn.model)))
    return pd.DataFrame(rows, columns=["Tham số", "Giá trị"])


def build_tables(registry, results_dir: str | Path, cfg) -> dict[str, pd.DataFrame]:
    R = Path(results_dir)
    info = _load(R, "model_info")
    tabs: dict[str, pd.DataFrame] = {}
    chosen_path = R / "chosen_lr.json"
    import json
    chosen = json.loads(chosen_path.read_text(encoding="utf-8")) if chosen_path.exists() else {}
    plan = _load(R, "memory_plan")
    tabs["HP"] = hyperparameter_table(cfg, registry, chosen, plan if len(plan) else None)

    def base(m):
        return {"Họ": m.family, "Mô hình": m.name}

    import math
    n_seeds = len(cfg.train.seeds)
    n_folds = int(cfg.splits.s2.n_folds)
    fs = cfg.fewshot
    n_e4 = {"proto": math.ceil(int(fs.episodes_prototype) / n_folds) * n_folds,
            "lp": math.ceil(int(fs.episodes_linear) / n_folds) * n_folds}

    # ---- E0: preprocessing ablation
    e0 = _load(R, "E0")
    rows = []
    for m in [m for m in registry if not _sel(e0, model=m.key).empty] or [m for m in registry if m.basic]:
        r = {"Mô hình": m.name}
        for mode, label in (("stretch", "Kéo dãn"), ("letterbox", "Letterbox"), ("nonsquare", "Không vuông")):
            d = _sel(e0, model=m.key, mode=mode)
            r[label] = ("(không áp dụng)" if (mode == "nonsquare" and m.square_only)
                        else fmt(_col(d, "test_top1")))
        rows.append(r)
    tabs["E0"] = pd.DataFrame(rows)

    # ---- LR sweep
    lr = _load(R, "lr_sweep")
    fams = []
    for m in registry:
        if m.family in [f["Họ"] for f in fams]:
            continue
        d = _sel(lr, family=m.family)
        chosen = DASH
        vals = []
        grid = cfg.train.learning_rate.sweep_grid
        for g in grid:
            v = d[np.isclose(pd.to_numeric(d.lr, errors="coerce"), float(g))] if not d.empty else d
            vals.append(fmt(_col(v, "val_top1")))
        fixed = cfg.train.learning_rate.get("by_family") or {}
        if m.family in fixed:
            chosen = f"{float(fixed[m.family]):g} (cố định)"
        elif not d.empty and pd.to_numeric(d.val_top1, errors="coerce").notna().any():
            chosen = f"{float(d.loc[pd.to_numeric(d.val_top1, errors='coerce').idxmax(), 'lr']):g}"
        fams.append({"Họ": m.family, "LR đã chọn": chosen,
                     "Val Top-1 với " + " / ".join(f"{g:g}" for g in grid): " / ".join(vals)})
    tabs["LR"] = pd.DataFrame(fams)

    # ---- E1
    e1 = _load(R, "E1")
    rows = []
    for m in registry:
        d = _sel(e1, model=m.key)
        i = _sel(info, model=m.key)
        rows.append({**base(m),
                     "Params (M)": fmt(_col(i, "params_m"), nd=1), "GMACs": fmt(_col(i, "gmacs"), nd=2),
                     "Top-1 theo ảnh": fmt(_col(d, "test_top1"), expected=n_seeds),
                     "Top-1 TB theo người": fmt(_col(d, "test_top1_subject_mean"), expected=n_seeds),
                     "Top-5": fmt(_col(d, "test_top5"), expected=n_seeds),
                     "Macro-P": fmt(_col(d, "test_macro_precision"), expected=n_seeds),
                     "Macro-R": fmt(_col(d, "test_macro_recall"), expected=n_seeds),
                     "Macro-F1": fmt(_col(d, "test_macro_f1"), expected=n_seeds)})
    tabs["E1"] = pd.DataFrame(rows)

    # ---- E2
    e2 = _load(R, "E2")
    rows = []
    for m in registry:
        s2 = _sel(e2, model=m.key, split="S2")
        s1 = _sel(e2, model=m.key, split="S1")
        gap = DASH
        if len(_col(s1, "test_acc")) and len(_col(s2, "test_acc")):
            gap = f"{pd.to_numeric(s1.test_acc).mean() - pd.to_numeric(s2.test_acc).mean():+.2f}"
        f2 = lambda c: fmt(_col(s2, c), expected=n_folds)  # noqa: E731
        rows.append({**base(m), "Acc (S2)": f2("test_acc"), "Bal. Acc (S2)": f2("test_bal_acc"),
                     "F1 (S2)": f2("test_f1"), "AUC (S2)": f2("test_auc"),
                     "Acc mức người (S2)": f2("test_acc_subject"),
                     "Acc (S1)": fmt(_col(s1, "test_acc"), expected=n_seeds), "Chênh S1 − S2": gap})
    tabs["E2"] = pd.DataFrame(rows)

    # ---- E3 (identity few-shot)
    e3 = _load(R, "E3")
    n_e3 = {"proto": int(fs.episodes_prototype), "lp": int(fs.episodes_linear)}
    for name, col in (("E3", "top1"), ("E3s", "top1_subject_mean")):
        rows = []
        for m in registry:
            r = base(m)
            for clf, lab in (("proto", "Proto"), ("lp", "LP")):
                for k in cfg.fewshot.identity_shots:
                    r[f"{lab} {k}-shot"] = fmt(_col(_sel(e3, model=m.key, classifier=clf, shot=k), col), "ci",
                                               expected=n_e3[clf])
            rows.append(r)
        tabs[name] = pd.DataFrame(rows)

    # ---- E3b
    e3b = _load(R, "E3b")
    n_runs = int(fs.single_gallery_runs)
    tabs["E3b"] = pd.DataFrame([{**base(m), **{f"Rank-{k}": fmt(_col(_sel(e3b, model=m.key), f"rank{k}"), "ci",
                                                               expected=n_runs) for k in (1, 5, 10)},
                                 "AUCMC": fmt(_col(_sel(e3b, model=m.key), "aucmc"), "ci", expected=n_runs)}
                                for m in registry])

    # ---- E4 (gender few-shot)
    e4 = _load(R, "E4")
    rows = []
    for m in registry:
        r = base(m)
        for k in cfg.fewshot.gender_shots:
            r[f"Proto K={k}"] = fmt(_col(_sel(e4, model=m.key, classifier="proto", shot=k), "bal_acc"), "ci",
                                    expected=n_e4["proto"])
        for k in cfg.fewshot.gender_linear_shots:
            r[f"LP K={k}"] = fmt(_col(_sel(e4, model=m.key, classifier="lp", shot=k), "bal_acc"), "ci",
                                 expected=n_e4["lp"])
        rows.append(r)
    tabs["E4"] = pd.DataFrame(rows)

    # ---- E5 + E6
    e56 = _load(R, "E56")
    n_rs = int(cfg.unsupervised.kmeans_restarts)
    f56 = lambda m, t, c: fmt(_col(_sel(e56, model=m.key, task=t), c), expected=n_rs)  # noqa: E731
    tabs["E56"] = pd.DataFrame([{**base(m),
                                 "Gender ACC": f56(m, "gender", "acc"), "Gender NMI": f56(m, "gender", "nmi"),
                                 "Identity ACC": f56(m, "identity", "acc"), "Identity NMI": f56(m, "identity", "nmi"),
                                 "Identity ARI": f56(m, "identity", "ari")}
                                for m in registry])

    # ---- E7
    e7 = _load(R, "E7")
    e7v = _load(R, "E7_verification")
    rows = []
    n_rr = int(cfg.retrieval.runs)
    far = f"{100 * float(cfg.retrieval.far):g}%"
    for m in registry:
        r = base(m)
        for variant, lab in (("finetuned", "tinh chỉnh"), ("frozen", "đóng băng")):
            for k in cfg.retrieval.gallery_shots:
                r[f"{k}-shot Top-1 ({lab})"] = fmt(_col(_sel(e7, model=m.key, variant=variant, shot=k), "rank1"), "ci",
                                                  expected=n_rr)
            r[f"EER ({lab})"] = fmt(_col(_sel(e7v, model=m.key, variant=variant), "eer"))
            r[f"TAR@FAR={far} ({lab})"] = fmt(_col(_sel(e7v, model=m.key, variant=variant), "tar_at_far"))
        rows.append(r)
    tabs["E7"] = pd.DataFrame(rows)

    # ---- E8
    e8 = _load(R, "E8")
    rows = []
    for m in registry:
        for variant, lab in (("finetuned", "tinh chỉnh S3"), ("frozen", "đóng băng")):
            d = _sel(e8, model=m.key, variant=variant)
            if d.empty:
                continue
            r = {"Mô hình": m.name, "Đặc trưng": lab}
            for ds, dl in (("earvn1", "EarVN1.0"), ("awe", "AWE")):
                for c, cl in (("rank1", "Rank-1"), ("rank5", "Rank-5"), ("aucmc", "AUCMC")):
                    r[f"{dl} {cl}"] = fmt(_col(_sel(d, dataset=ds), c))
            rows.append(r)
    tabs["E8"] = pd.DataFrame(rows)

    # ---- resolution analysis
    res = _load(R, "resolution")
    if not res.empty:
        piv = []
        for m in registry:
            d = _sel(res, model=m.key)
            if d.empty:
                continue
            r = {"Mô hình": m.name}
            for b in d["bin"].drop_duplicates():
                r[str(b)] = fmt(_col(_sel(d, bin=b), "top1"))
            piv.append(r)
        cnt = res.drop_duplicates("bin").set_index("bin").n_images
        piv.append({"Mô hình": "Số ảnh test trong nhóm", **{str(b): str(int(n)) for b, n in cnt.items()}})
        tabs["resolution"] = pd.DataFrame(piv)
    return tabs


TITLES = {
    "HP": "Siêu tham số huấn luyện (tự sinh từ configs/default.yaml)",
    "E0": "E0 – So sánh cách tiền xử lý (Top-1 identity, %)",
    "LR": "Quét learning rate theo họ",
    "E1": "E1 – Identity, có giám sát (%)",
    "E2": "E2 – Gender, có giám sát (%)",
    "E3": "E3 – Identity, few-shot (Top-1 %, trung bình ± CI95)",
    "E3s": "E3 – Identity, few-shot (Top-1 trung bình theo người %, trung bình ± CI95)",
    "E3b": "E3b – Một ảnh mẫu mỗi người (%, trung bình ± CI95; đường CMC trong results/E3b_cmc.csv)",
    "E4": "E4 – Gender, few-shot (Balanced accuracy %, trung bình ± CI95)",
    "E56": "E5 + E6 – Không giám sát (%)",
    "E7": "E7 – Nhận dạng người mới (%)",
    "E8": "E8 – Thử chéo dataset (%)",
    "resolution": "Phân tích theo độ phân giải (Top-1 identity %, theo cạnh ngắn ảnh gốc)",
}


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(chưa có kết quả)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def to_latex(df: pd.DataFrame, caption: str) -> str:
    if df.empty:
        return f"% {caption}: no results yet\n"
    esc = lambda s: str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("±", r"$\pm$")
    cols = list(df.columns)
    out = [r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{esc(caption)}}}",
           r"\begin{tabular}{" + "l" * min(2, len(cols)) + "c" * max(0, len(cols) - 2) + "}", r"\toprule",
           " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        out.append(" & ".join(esc(r[c]) for c in cols) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def write_tables(tabs: dict[str, pd.DataFrame], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = ["# Kết quả benchmark EarVN2.0 (tự sinh bởi scripts/16_make_tables.py)\n"]
    for name, df in tabs.items():
        title = TITLES.get(name, name)
        md = to_markdown(df)
        (out_dir / f"{name}.md").write_text(f"## {title}\n\n{md}", encoding="utf-8")
        (out_dir / f"{name}.tex").write_text(to_latex(df, title), encoding="utf-8")
        combined.append(f"## {title}\n\n{md}")
    path = out_dir / "all_tables.md"
    path.write_text("\n".join(combined), encoding="utf-8")
    return path
