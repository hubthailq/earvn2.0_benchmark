"""Supervised fine-tuning (E0, E1, E2, LR sweep, S3 models for E7/E8)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import ImageTable, make_loader
from .device import DeviceManager, has_batchnorm, resolve_amp
from .metrics import gender_metrics, identity_metrics
from .models import ModelSpec, count_params, create_model
from .paths import Paths
from .preprocess import build_transform, model_norm
from .utils import atomic_write_csv, get_logger, seed_everything


def build_optimizer(model, cfg_opt, lr: float):
    """AdamW or SGD-with-momentum, all settings from train.optimizer."""
    name = str(cfg_opt.name).lower()
    wd = float(cfg_opt.weight_decay)
    if bool(cfg_opt.get("no_decay_bias_norm", True)):
        # as timm does: no decay on biases, norm weights and the parameters the model itself lists
        # (ViT cls_token / pos_embed, Swin relative_position_bias_table, ...)
        skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            listed = n in skip
            (no_decay if p.ndim <= 1 or n.endswith(".bias") or listed else decay).append(p)
        groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": wd}]
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(float(b) for b in cfg_opt.betas), eps=float(cfg_opt.eps))
    if name in ("sgd", "sgdm"):
        return torch.optim.SGD(groups, lr=lr, momentum=float(cfg_opt.momentum), nesterov=bool(cfg_opt.nesterov))
    raise ValueError(f"train.optimizer.name must be adamw or sgd, got '{cfg_opt.name}'")


def build_scheduler(optimizer, cfg_sch, epochs: int, steps_per_epoch: int):
    """Per-iteration LR multiplier: linear warm-up, then cosine / step / constant (train.scheduler)."""
    name = str(cfg_sch.name).lower()
    if name not in ("cosine", "step", "constant"):
        raise ValueError(f"train.scheduler.name must be cosine, step or constant, got '{cfg_sch.name}'")
    total = epochs * steps_per_epoch
    warmup = min(int(float(cfg_sch.warmup_epochs) * steps_per_epoch), max(total - 1, 0))
    floor = float(cfg_sch.get("min_lr_ratio", 0.0))
    milestones = [int(e) * steps_per_epoch for e in (cfg_sch.get("step_epochs") or [])]
    gamma = float(cfg_sch.get("step_gamma", 0.1))

    def f(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        if name == "cosine":
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        if name == "step":
            return gamma ** sum(step >= m for m in milestones)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def use_channels_last(cfg, spec) -> bool:
    return spec.family in set(cfg.train.get("channels_last_families") or [])


def gpu_name(device) -> str:
    device = torch.device(device)
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def planned_micro_batches(cfg, spec, device, amp_name: str, logger=None) -> int:
    """Micro-batch split measured by script 05 for this model on this GPU with these settings (else 1)."""
    mp = cfg.train.get("memory_plan") or {}
    if torch.device(device).type != "cuda" or not mp.get("use", True):
        return 1
    path = Paths(cfg).memory_plan_csv
    warn = logger.warning if logger else (lambda m: None)
    if not path.exists():
        warn(f"no memory plan ({path}); run scripts/05_model_info.py on this GPU first so every model starts "
             f"with its fixed micro-batch split. Continuing with full batches + reactive splitting.")
        return 1
    df = pd.read_csv(path, dtype={"model": str})
    row = df[df.model == spec.key]
    want = {"gpu_name": gpu_name(device), "batch_size": int(cfg.train.batch_size), "amp_dtype": amp_name,
            "channels_last": bool(use_channels_last(cfg, spec)), "size": int(cfg.preprocess.size)}
    if row.empty:
        warn(f"{spec.key}: not in the memory plan; run scripts/05_model_info.py --models {spec.key}")
        return 1
    r = row.iloc[0]
    diff = {k: (r.get(k), v) for k, v in want.items() if str(r.get(k)) != str(v)}
    if diff:
        warn(f"{spec.key}: memory plan was measured with different settings {diff}; re-run "
             f"scripts/05_model_info.py --models {spec.key} --overwrite. Using full batches.")
        return 1
    if pd.isna(r.get("micro_batches")):
        warn(f"{spec.key}: does not fit on this GPU even with the smallest micro-batch (see memory_plan.csv)")
        return 1
    return max(1, int(r.micro_batches))


def effective_hparams(cfg, spec, task: str, lr: float, epochs: int, mode: str | None) -> dict:
    """Everything needed to reproduce one fine-tuning run (saved as hparams.json)."""
    tc = cfg.train
    return {"model": spec.key, "timm": spec.timm, "task": task, "learning_rate": lr, "max_epochs": epochs,
            "batch_size": int(tc.batch_size), "eval_batch_size": int(tc.get("eval_batch_size", 128)),
            "patience": int(tc.patience), "optimizer": dict(tc.optimizer), "scheduler": dict(tc.scheduler),
            "label_smoothing": float(tc.label_smoothing), "grad_clip": float(tc.grad_clip),
            "gender_balanced_loss": bool(tc.gender_balanced_loss), "amp": bool(tc.amp),
            "amp_dtype": str(tc.get("amp_dtype", "auto")), "pretrained": bool(cfg.get("pretrained", True)),
            "channels_last": use_channels_last(cfg, spec),
            "preprocess": {"mode": mode or cfg.preprocess.mode, "size": int(cfg.preprocess.size),
                           "nonsquare_hw": list(cfg.preprocess.nonsquare_hw), "pad": cfg.preprocess.pad,
                           "interpolation": cfg.preprocess.interpolation, "augment": dict(cfg.preprocess.augment)}}


def run_fingerprint(cfg, spec, task: str, mode: str | None, epochs: int | None, *frames, seed=None) -> str:
    """Short hash of everything that defines a fine-tuning run except the learning rate (compared on its own):
    the whole training recipe + the exact train/val/test images and labels. A stored result whose fingerprint
    differs (config changed, split re-made, ...) is re-run instead of silently kept."""
    hp = effective_hparams(cfg, spec, task, 0.0, int(epochs or cfg.train.epochs), mode)
    hp.pop("learning_rate")
    hp.update({"seed": seed, "model_kwargs": dict(spec.kwargs or {})})
    data = [[list(map(str, df.image_id)), list(map(int, df.label))] if df is not None else None for df in frames]
    blob = json.dumps({"hp": hp, "data": data}, sort_keys=True, default=str)
    return "fp" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]   # prefix: never read back as a number


def train_or_diverge(*args, logger=None, **kwargs) -> dict:
    """train_and_evaluate, but a run whose loss becomes NaN/inf is recorded (status=diverged) instead of
    stopping a queue of hundreds of runs. Script 16 lists diverged runs."""
    try:
        return {"status": "ok", **train_and_evaluate(*args, logger=logger, **kwargs)}
    except FloatingPointError as exc:
        (logger or get_logger()).error(f"  run diverged and is recorded as such: {exc}. Fix: set a smaller LR for "
                                       f"this family in train.learning_rate.by_family and run the script again "
                                       f"(the new LR makes it re-run these runs; no --overwrite needed)")
        return {"status": "diverged", "error": str(exc)[:300]}


@torch.no_grad()
def predict(model, loader, device, task: str, dm: DeviceManager | None = None) -> np.ndarray:
    """identity: (N, C) softmax scores; gender: (N,) probability of class 1."""
    dm = dm or DeviceManager(device)
    model.eval()
    out = []
    for x, _, _ in loader:
        logits = dm.forward(model, x)
        p = torch.softmax(logits.float(), dim=1)
        out.append((p[:, 1] if task == "gender" else p).numpy())
    if not out:
        return np.zeros((0,)) if task == "gender" else np.zeros((0, 0))
    return np.concatenate(out)


def score(task: str, pred: np.ndarray, y: np.ndarray, subjects=None) -> dict:
    if task == "identity":
        return identity_metrics(pred, y)
    return gender_metrics(pred, y, subjects)


def val_key(task: str) -> str:
    return "top1" if task == "identity" else "bal_acc"


def train_and_evaluate(spec: ModelSpec, cfg, task: str, root: str, train: pd.DataFrame, val: pd.DataFrame,
                       test: pd.DataFrame | None, n_classes: int, lr: float, seed: int, run_dir: Path,
                       device: torch.device, mode: str | None = None, epochs: int | None = None,
                       save_checkpoint: bool = False, class_names=None, logger=None) -> dict:
    """Fine-tune one model. DataFrames need columns image_id, label (int), subject.
    Writes log.csv, predictions.csv (test), metrics.json and optionally best.pt to run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("log.csv", "predictions.csv", "metrics.json", "best.pt", "hparams.json"):
        (run_dir / stale).unlink(missing_ok=True)   # a re-run (even a failed one) never leaves old files behind
    if task not in ("identity", "gender"):
        raise ValueError(f"task must be identity or gender, got {task}")
    for name, df in (("train", train), ("val", val)):
        if df is None or len(df) == 0:
            raise ValueError(f"{name} set is empty")
    if task == "gender" and train.label.nunique() < 2:
        raise ValueError("gender training set contains a single class")
    logger = logger or get_logger()
    seed_everything(seed)
    tc = cfg.train
    epochs = int(epochs or tc.epochs)
    torch.backends.cudnn.deterministic = bool(tc.get("deterministic", False))
    torch.backends.cudnn.benchmark = not bool(tc.get("deterministic", False))
    use_amp, amp_dtype, amp_name = resolve_amp(tc, device)
    cl = use_channels_last(cfg, spec)
    micro = planned_micro_batches(cfg, spec, device, amp_name, logger)
    dm = DeviceManager(device, cfg.get("gpu_fallback"), logger, micro_batches=micro, channels_last=cl,
                       infer_amp_dtype=amp_dtype if use_amp else None)   # same precision for val/test
    model = dm.place(create_model(spec, n_classes, pretrained=bool(cfg.get("pretrained", True))))
    bn_micro = micro > 1 and has_batchnorm(model)
    if bn_micro:
        logger.warning(f"  [{spec.key}] BatchNorm model trained with {micro} micro-batches on this GPU: BN "
                       f"statistics use {-(-int(tc.batch_size) // micro)} images instead of {tc.batch_size} "
                       f"(flagged in the tables)")
    run_env = {"gpu_name": gpu_name(device), "amp_dtype_used": amp_name, "channels_last": cl,
               "planned_micro_batches": micro, "bn_micro_batch": bn_micro,
               "torch_version": torch.__version__}
    with open(run_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump({**effective_hparams(cfg, spec, task, lr, epochs, mode), **run_env}, f, indent=2,
                  ensure_ascii=False)
    mean, std = model_norm(model)
    tf_train = build_transform(cfg.preprocess, mean, std, train=True, mode=mode)
    tf_eval = build_transform(cfg.preprocess, mean, std, train=False, mode=mode)
    nw = int(cfg.num_workers)
    bs = int(tc.batch_size)
    dl_train = make_loader(ImageTable(root, train.image_id, train.label, tf_train), bs, True, nw, drop_last=True, seed=seed)
    ebs = int(tc.get("eval_batch_size", 2 * bs))
    dl_val = make_loader(ImageTable(root, val.image_id, val.label, tf_eval), ebs, False, nw)

    optimizer = build_optimizer(model, tc.optimizer, lr)
    dm.init_optimizer_state(model, optimizer)
    steps_per_epoch = max(1, len(dl_train))
    sched = build_scheduler(optimizer, tc.scheduler, epochs, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def loss_sum(logits, y):  # summed so that micro-batches add up to exactly the full-batch loss
        w = None if class_w is None else class_w.to(logits.device)
        return F.cross_entropy(logits, y, weight=w, label_smoothing=ls, reduction="sum")
    grad_clip = float(tc.get("grad_clip", 1.0) or 0)
    ls = float(tc.label_smoothing)
    class_w = None
    if task == "gender" and bool(tc.get("gender_balanced_loss", True)):
        counts = np.bincount(train.label.to_numpy(), minlength=n_classes).astype(np.float64)
        class_w = torch.tensor(len(train) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)

    key = val_key(task)
    best, best_state, best_epoch, bad_epochs = -float("inf"), None, -1, 0
    log_rows = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in dl_train:
            denom = float(len(y)) if class_w is None else float(class_w.cpu()[y].sum())
            try:
                loss = dm.train_step(model, optimizer, x, y, loss_sum, denom, scaler, use_amp, amp_dtype, grad_clip)
            except FloatingPointError as exc:
                raise FloatingPointError(f"{exc} (epoch {epoch}, lr={lr})") from exc
            sched.step()
            tot += loss * len(y)
            n += len(y)
        val_pred = predict(model, dl_val, device, task, dm)
        vm = score(task, val_pred, val.label.to_numpy(), val.subject.to_numpy())
        v = vm[key]
        row = {"epoch": epoch + 1, "train_loss": tot / max(n, 1), f"val_{key}": v,
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1),
               "device": str(dm.current), "micro_batches": dm.chunks}
        log_rows.append(row)
        atomic_write_csv(pd.DataFrame(log_rows), run_dir / "log.csv")
        logger.info(f"  [{spec.key}] epoch {epoch + 1}/{epochs} loss={row['train_loss']:.4f} val_{key}={v:.2f}")
        if v > best + 1e-9:
            best, best_epoch, bad_epochs = v, epoch + 1, 0
            best_state = copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= int(tc.patience):
                logger.info(f"  [{spec.key}] early stop at epoch {epoch + 1} (best epoch {best_epoch})")
                break

    if best_state is None:  # val metric was never finite (e.g. NaN): keep the last epoch
        logger.warning(f"  [{spec.key}] validation metric never improved/finite; using the last epoch")
        best_state = {k: t.detach().cpu() for k, t in model.state_dict().items()}
        best_epoch = len(log_rows)
    model.load_state_dict(best_state)
    result = {"best_epoch": best_epoch, f"val_{key}": best, "train_seconds": round(time.time() - t0, 1),
              "params_m": round(count_params(model), 3)}
    if save_checkpoint:
        torch.save({"state_dict": best_state, "timm": spec.timm, "n_classes": n_classes,
                    "classes": list(class_names) if class_names is not None else None,
                    "kwargs": spec.kwargs}, run_dir / "best.pt")
    if test is not None and len(test):
        dl_test = make_loader(ImageTable(root, test.image_id, test.label, tf_eval), ebs, False, nw)
        pred = predict(model, dl_test, device, task, dm)
        tm = score(task, pred, test.label.to_numpy(), test.subject.to_numpy())
        result.update({f"test_{k}": v for k, v in tm.items()})
        pdf = pd.DataFrame({"image_id": test.image_id.to_numpy(), "subject": test.subject.to_numpy(),
                            "label": test.label.to_numpy()})
        if task == "identity":
            top5 = np.argsort(-pred, axis=1)[:, :5]
            pdf["pred"] = top5[:, 0]
            pdf["top5"] = [" ".join(map(str, r)) for r in top5]
            pdf["score"] = pred[np.arange(len(pred)), top5[:, 0]]
        else:
            pdf["prob_1"] = pred
            pdf["pred"] = (pred >= 0.5).astype(int)
        atomic_write_csv(pdf, run_dir / "predictions.csv")
    result.update({f"gpu_{k}": v for k, v in dm.stats.items()})  # 0 everywhere = the run never left the GPU
    result.update({k: v for k, v in run_env.items() if k != "torch_version"})
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def load_finetuned(spec: ModelSpec, ckpt_path: str | Path, device) -> torch.nn.Module:
    """Rebuild a fine-tuned model from best.pt (classifier kept; use forward_head(pre_logits=True))."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_model(spec, int(ck["n_classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()
