#!/usr/bin/env python
"""05 - Model size and cost + GPU memory plan (run it on the GPU you will train on, before script 06).

Part 1 (model_info.csv): parameters (backbone, without the classifier head), GMACs for one 224x224 image,
  and bs=1 inference latency in plain fp32 (same conditions for every model, so these numbers compare
  architectures, not speed tricks). GMACs = multiply-accumulates; most papers call this number "GFLOPs".
Part 2 (memory_plan.csv, GPU only): for every model, a few real training steps with the real settings
  (train.batch_size, mixed precision, channels_last, AdamW state) to find how many micro-batches it needs
  on THIS GPU (1 = the full batch fits). Every later run of the model starts with exactly this split, so
  all models get the same effective batch size and the conditions do not depend on what else was using
  the GPU at that moment. Also records peak memory and training throughput (images/s).
  A BatchNorm model that needs micro-batches is reported: its BN statistics use fewer images.
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from earbench.common import setup  # noqa: E402
from earbench.device import DeviceManager, has_batchnorm, plan_micro_batches, resolve_amp  # noqa: E402
from earbench.models import count_gmacs, count_params, create_model, resolve_device, select_models  # noqa: E402
from earbench.train import build_optimizer, gpu_name, use_channels_last  # noqa: E402
from earbench.utils import ResultStore  # noqa: E402


def extra(p):
    p.add_argument("--models", default="all", help="all | basic | lr_rep | comma-separated keys")
    p.add_argument("--part", default="all", choices=["all", "info", "memory"])
    p.add_argument("--overwrite", action="store_true")


@torch.no_grad()
def latency_ms(model, device, size, reps=20):
    x = torch.zeros(1, 3, size, size, device=device)
    for _ in range(3):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def model_info(specs, cfg, paths, device, log, overwrite):
    store = ResultStore(paths.model_info_csv, ["model"])
    size = int(cfg.preprocess.size)
    for spec in specs:
        if store.has({"model": spec.key}) and not overwrite:
            continue
        try:
            m = create_model(spec, num_classes=0, pretrained=False).eval()
            row = {"model": spec.key, "name": spec.name, "family": spec.family, "timm": spec.timm,
                   "params_m": round(count_params(m), 3), "gmacs": round(count_gmacs(m, (size, size)), 3)}
            m = m.to(device)
            row["latency_ms_bs1"] = round(latency_ms(m, device, size), 2)
            row["latency_device"] = device.type
        except Exception as exc:
            log.error(f"  {spec.key}: {type(exc).__name__}: {exc}")
            continue
        store.upsert(row)
        log.info(f"  {spec.key:14s} params={row['params_m']:.2f}M  GMACs={row['gmacs']:.2f}")


def n_classes_for_probe(paths) -> int:
    try:
        return max(2, len(pd.read_csv(paths.subjects_csv)))
    except Exception:
        return 1000   # upper bound; the head is tiny compared with the backbone anyway


def settings(cfg, spec, device) -> dict:
    return {"gpu_name": gpu_name(device), "batch_size": int(cfg.train.batch_size), "size": int(cfg.preprocess.size),
            "amp_dtype": resolve_amp(cfg.train, device)[2], "channels_last": bool(use_channels_last(cfg, spec))}


def probe(spec, cfg, device, n_classes):
    tc, mp = cfg.train, cfg.train.memory_plan
    use_amp, amp_dtype, _ = resolve_amp(tc, device)
    cl = use_channels_last(cfg, spec)
    bs, size = int(tc.batch_size), int(cfg.preprocess.size)
    x = torch.randn(bs, 3, size, size)
    y = torch.randint(0, n_classes, (bs,))
    x_eval = torch.randn(int(tc.get("eval_batch_size", 2 * bs)), 3, size, size)
    clip = float(tc.get("grad_clip", 0) or 0)
    info = {}

    def loss_sum(logits, t):
        return F.cross_entropy(logits, t, reduction="sum")

    def try_step(chunks):
        model = opt = dm = step = None
        gc.collect()                 # memory of a previous failed attempt is released only now
        torch.cuda.empty_cache()
        try:
            model = create_model(spec, n_classes, pretrained=False)
            # exactly the training conditions: cuDNN autotuning (workspace!), optimizer state, eval batch
            torch.backends.cudnn.deterministic = bool(tc.get("deterministic", False))
            torch.backends.cudnn.benchmark = not bool(tc.get("deterministic", False))
            dm = DeviceManager(device, {"mode": "off"}, micro_batches=chunks, channels_last=cl,
                               infer_amp_dtype=amp_dtype if use_amp else None)
            dm.place(model)
            info["batchnorm"] = has_batchnorm(model)
            opt = build_optimizer(model, tc.optimizer, 1e-5)
            dm.init_optimizer_state(model, opt)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
            torch.cuda.reset_peak_memory_stats(device)
            step = lambda: dm.train_step(model, opt, x, y, loss_sum, float(bs), scaler, use_amp,  # noqa: E731
                                         amp_dtype, clip)
            model.train()
            for _ in range(max(1, int(mp.probe_steps))):
                step()
            model.eval()
            dm.forward(model, x_eval)          # validation / test batches, with the optimizer state still held
            model.train()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_reserved(device)
            t = time.perf_counter()
            for _ in range(max(1, int(mp.timed_steps))):
                step()
            torch.cuda.synchronize(device)
            info[chunks] = bs * max(1, int(mp.timed_steps)) / (time.perf_counter() - t)
            return peak
        finally:
            model = opt = dm = step = None  # noqa: F841  (free GPU memory before the next attempt)
            gc.collect()
            torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info(device)[0]
    budget = free * float(mp.budget_fraction)
    chunks, peak = plan_micro_batches(try_step, bs, int(cfg.gpu_fallback.get("min_micro_batch", 4)), budget)
    bn = bool(info.get("batchnorm", False))
    return {"micro_batches": chunks, "micro_batch_size": None if chunks is None else -(-bs // chunks),
            "peak_gb": None if peak is None else round(peak / 2 ** 30, 2), "budget_gb": round(budget / 2 ** 30, 2),
            "train_img_s": None if chunks is None else round(info[chunks], 1), "batchnorm": bn,
            "bn_micro_batch": bool(bn and chunks is not None and chunks > 1)}


def memory_plan(specs, cfg, paths, device, log, overwrite):
    if device.type != "cuda":
        log.info("memory plan skipped: no CUDA GPU (it is only needed for GPU training)")
        return
    store = ResultStore(paths.memory_plan_csv, ["model"])
    n_classes = n_classes_for_probe(paths)
    log.info(f"memory plan on {gpu_name(device)} (batch {cfg.train.batch_size}, "
             f"{resolve_amp(cfg.train, device)[2]}); close other programs that use the GPU while this runs")
    for spec in specs:
        want = settings(cfg, spec, device)
        old = store.get({"model": spec.key}) if store.has({"model": spec.key}) else None
        if old is not None and not overwrite and all(str(old.get(k)) == str(v) for k, v in want.items()):
            continue
        try:
            r = probe(spec, cfg, device, n_classes)
        except Exception as exc:
            log.error(f"  {spec.key}: memory probe failed: {type(exc).__name__}: {exc}")
            continue
        store.upsert({"model": spec.key, **want, **r})
        log.info(f"  {spec.key:14s} micro-batches={r['micro_batches']} peak={r['peak_gb']} GB "
                 f"train={r['train_img_s']} img/s")
    df = store.load()
    if len(df):
        split = df[pd.to_numeric(df.micro_batches, errors="coerce").fillna(0) > 1]
        nofit = df[df.micro_batches.isna()]
        bn = df[df.bn_micro_batch.astype(str) == "True"]
        if len(split):
            log.info("models trained with gradient accumulation (identical gradient to the full batch): "
                     + ", ".join(f"{r.model}×{int(r.micro_batches)}" for r in split.itertuples()))
        if len(bn):
            log.warning("BatchNorm models that need micro-batches (BN statistics on fewer images, flagged in the "
                        "tables): " + ", ".join(bn.model) + ". Consider train.batch_size that fits for all.")
        if len(nofit):
            log.warning("models that do not fit even with the smallest micro-batch: " + ", ".join(nofit.model))


def main():
    args, cfg, paths, log = setup(__doc__, extra)
    device = resolve_device(cfg.device)
    specs = select_models(args.models)
    if args.part in ("all", "info"):
        model_info(specs, cfg, paths, device, log, args.overwrite)
    if args.part in ("all", "memory"):
        memory_plan(specs, cfg, paths, device, log, args.overwrite)


if __name__ == "__main__":
    main()
