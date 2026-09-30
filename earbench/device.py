"""GPU-first execution that survives running out of GPU memory.

Policy (config section ``gpu_fallback``), applied to every training step and every inference batch:

  1. Always try the preferred device (GPU if available).
  2. On a CUDA out-of-memory error: free the cache and split the batch into micro-batches
     (gradient accumulation, mathematically the same loss/gradient as the full batch), halving the
     micro-batch size until ``min_micro_batch``.
  3. Still out of memory -> ``mode``:
       cpu  : move model + optimizer to the CPU and continue there (slow, but the run never crashes)
       wait : sleep and retry on the GPU until memory is free (up to ``max_wait_minutes``)
       off  : raise the error (old behaviour)
  4. While on the CPU, every ``check_every_s`` seconds look at free GPU memory; when at least
     ``min_free_mb_to_return`` MB are free, move back to the GPU. A failed return doubles the waiting time
     (no flapping).

Every switch is logged and counted, and the counters are saved with the run's metrics so a result that
was partly computed on the CPU (no mixed precision there) can be identified later.

Fair comparison between models (README section 8.1): the number of micro-batches a model needs on a given
GPU is measured ONCE beforehand (``plan_micro_batches``, script 05) and every run of that model starts with
exactly that split, so the training conditions do not depend on what else happened to be using the GPU.
The reactive splitting above is then only a safety net, and runs that needed it are flagged by script 16.
"""
from __future__ import annotations

import gc
import time
from typing import Callable

import torch

OOM_MARKERS = ("out of memory", "CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_ALLOC_FAILED",
               "cudaErrorMemoryAllocation", "CUBLAS_STATUS_NOT_INITIALIZED")


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and any(m in str(exc) for m in OOM_MARKERS)


def _gpu_free_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 2 ** 20


def resolve_amp(tc, device: torch.device):
    """(enabled, dtype, name). 'auto' = bf16 when the GPU supports it (RTX 30xx/40xx: yes), else fp16."""
    want = str(tc.get("amp_dtype", "auto")).lower()
    if want not in ("auto", "bf16", "fp16"):
        raise ValueError("train.amp_dtype must be auto, bf16 or fp16")
    use_amp = bool(tc.amp) and torch.device(device).type == "cuda"
    if not use_amp:
        return False, torch.float32, "fp32"
    if want == "bf16" or (want == "auto" and torch.cuda.is_bf16_supported()):
        return True, torch.bfloat16, "bf16"
    return True, torch.float16, "fp16"


class DeviceManager:
    def __init__(self, preferred: torch.device, cfg_fb=None, logger=None,
                 fallback: torch.device | None = None, free_mb_fn: Callable[[], float] | None = None,
                 sleep_fn: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 micro_batches: int = 1, channels_last: bool = False,
                 infer_amp_dtype: torch.dtype | None = torch.float16):
        get = (lambda k, d: cfg_fb.get(k, d)) if cfg_fb is not None else (lambda k, d: d)
        self.preferred = torch.device(preferred)
        self.fallback = torch.device(fallback) if fallback is not None else torch.device("cpu")
        self.mode = str(get("mode", "cpu")).lower()
        if self.mode not in ("cpu", "wait", "off"):
            raise ValueError("gpu_fallback.mode must be cpu, wait or off")
        self.min_micro = max(1, int(get("min_micro_batch", 4)))
        self.min_free = float(get("min_free_mb_to_return", 4096))
        self.check_every = float(get("check_every_s", 60))
        self.max_wait = float(get("max_wait_minutes", 360)) * 60
        self.log = logger
        self.free_mb = free_mb_fn or _gpu_free_mb
        self.sleep = sleep_fn
        self.clock = clock
        self.current = self.preferred
        self.base_chunks = max(1, int(micro_batches))   # planned split (memory plan), never reduced below
        self.chunks = self.base_chunks       # number of micro-batches per batch on the preferred device
        self.channels_last = bool(channels_last)
        self.infer_amp_dtype = infer_amp_dtype   # autocast dtype for inference on the GPU (None = fp32)
        self.optimizer = None                # attached by train_step: it always moves together with the model
        self._next_check = 0.0
        self._backoff = self.check_every
        self._wait_streak = 0
        self.stats = {"oom_events": 0, "switches_to_fallback": 0, "switches_back": 0,
                      "max_micro_batches": self.chunks, "fallback_batches": 0, "step_oom": 0}

    # ------------------------------------------------------------------ helpers
    @property
    def on_fallback(self) -> bool:
        return self.current != self.preferred

    @property
    def enabled(self) -> bool:
        """Fallback logic only matters when the preferred device is a GPU (or in tests)."""
        return self.preferred != self.fallback

    def _info(self, msg):
        if self.log:
            self.log.warning(msg)

    @staticmethod
    def _cleanup():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def amp_enabled(self, wanted: bool) -> bool:
        return bool(wanted) and self.current.type == "cuda"

    def move(self, model: torch.nn.Module, optimizer=None, device: torch.device | None = None) -> None:
        """Move the model AND its optimizer state (the attached one if none is given): they must never be
        on different devices, whichever code path (training, validation, inference) triggers the move."""
        device = torch.device(device or self.current)
        optimizer = optimizer if optimizer is not None else self.optimizer
        model.to(device)
        if optimizer is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    # AdamW keeps 'step' on the CPU unless capturable=True; moving it breaks the optimizer
                    if torch.is_tensor(v) and k != "step":
                        state[k] = v.to(device)
        self.current = device
        self._cleanup()

    def _input(self, x: torch.Tensor, dev: torch.device) -> torch.Tensor:
        if self.channels_last and x.ndim == 4:
            return x.to(dev, non_blocking=True, memory_format=torch.channels_last)
        return x.to(dev, non_blocking=True)

    def place(self, model: torch.nn.Module) -> torch.nn.Module:
        """Put a freshly built model on the preferred device, or the fallback if it does not fit."""
        if self.channels_last:   # only the memory layout changes, not the computation
            model.to(memory_format=torch.channels_last)
        while True:
            try:
                self.move(model, device=self.preferred)
                self._wait_streak = 0
                return model
            except Exception as exc:
                if not (is_oom(exc) and self.enabled and self.mode != "off"):
                    raise
                self.stats["oom_events"] += 1
                self._cleanup()
                if self.mode == "wait":
                    self._wait_for_gpu("model does not fit on the GPU")   # then try the GPU again
                    continue
                self._to_fallback(model, None, "model does not fit on the GPU")
                return model

    def init_optimizer_state(self, model, optimizer) -> None:
        """Allocate the optimizer state now (zeros), exactly as torch would on the first step.

        Otherwise the first optimizer.step() allocates it, and an out-of-memory error half-way leaves some
        parameters with incomplete state -> KeyError on every later step. Numerically identical: AdamW
        starts from zero moments; SGD's first momentum buffer is 0 * momentum + grad = grad."""
        self.optimizer = optimizer
        while True:
            try:
                _init_state(optimizer)
                return
            except Exception as exc:
                optimizer.state.clear()
                if not (is_oom(exc) and self.enabled and self.mode != "off" and self.current == self.preferred):
                    raise
                self.stats["oom_events"] += 1
                self._cleanup()
                self._to_fallback(model, optimizer, "no GPU memory for the optimizer state")

    def _to_fallback(self, model, optimizer, why: str):
        if self.mode == "wait":
            self._wait_for_gpu(why)
            return
        self._info(f"[gpu-fallback] {why} -> continuing on {self.fallback} (slow). "
                   f"Will try the GPU again when {self.min_free:.0f} MB are free.")
        self.move(model, optimizer, self.fallback)
        self.stats["switches_to_fallback"] += 1
        self._backoff = self.check_every
        self._next_check = self.clock() + self._backoff

    def _wait_for_gpu(self, why: str):
        """mode=wait: pause until the GPU has room again. Consecutive failures (the GPU looks free but the work
        still does not fit, e.g. another program keeps 3 GB of a 10 GB card) pause longer and longer
        (check_every, 2x, 4x, ... capped at 30 min); the run gives up only after max_wait_minutes in total."""
        if self._wait_streak == 0:
            self._wait_start = self.clock()
        self._wait_streak += 1

        def give_up():
            raise RuntimeError(f"{why}: no room on the GPU for {self.max_wait / 60:.0f} minutes "
                               f"(gpu_fallback.max_wait_minutes). Close other GPU programs, lower "
                               f"train.batch_size, or use gpu_fallback.mode=cpu.")
        self._info(f"[gpu-fallback] {why} -> waiting for the GPU (attempt {self._wait_streak})")
        if self._wait_streak > 1:
            if self.clock() - self._wait_start > self.max_wait:
                give_up()
            self.sleep(min(self.check_every * 2 ** (self._wait_streak - 2), 1800.0))
        while self.free_mb() < self.min_free:
            if self.clock() - self._wait_start > self.max_wait:
                give_up()
            self.sleep(self.check_every)
        self.chunks = self.base_chunks   # memory is free again: back to the planned split
        self._cleanup()

    def maybe_return(self, model, optimizer=None) -> None:
        """Called before each batch: go back to the GPU when enough memory is free."""
        if not self.on_fallback or self.clock() < self._next_check:
            return
        free = self.free_mb()
        if free < self.min_free:
            self._next_check = self.clock() + self.check_every
            return
        try:
            self.move(model, optimizer, self.preferred)
            self.chunks = self.base_chunks  # memory is free again: back to the planned split
            self.stats["switches_back"] += 1
            self._info(f"[gpu-fallback] {free:.0f} MB free on the GPU -> back to {self.preferred}")
        except Exception as exc:
            if not is_oom(exc):
                raise
            self.move(model, optimizer, self.fallback)
            self._backoff = min(self._backoff * 2, 1800)
            self._next_check = self.clock() + self._backoff
            self._info(f"[gpu-fallback] return to GPU failed (OOM); next try in {self._backoff:.0f} s")

    # ------------------------------------------------------------------ training
    def train_step(self, model, optimizer, x, y, loss_sum_fn, denominator: float, scaler=None,
                   amp: bool = False, amp_dtype=torch.float16, grad_clip: float = 0.0) -> float:
        """One optimisation step on batch (x, y), robust to OOM. ``loss_sum_fn(logits, y)`` must return the
        SUM of per-sample losses so micro-batches add up exactly; ``denominator`` normalises it (batch size,
        or the sum of class weights). Returns the batch loss (mean)."""
        self.optimizer = optimizer
        self.maybe_return(model, optimizer)
        while True:
            dev = self.current
            chunks = self.chunks if dev == self.preferred else 1
            try:
                return self._do_step(model, optimizer, x, y, loss_sum_fn, denominator, scaler, amp, amp_dtype,
                                     grad_clip, dev, chunks)
            except Exception as exc:
                if not (is_oom(exc) and self.enabled and self.mode != "off" and dev == self.preferred):
                    raise
                optimizer.zero_grad(set_to_none=True)
                self.stats["oom_events"] += 1
                self._cleanup()
                micro = -(-len(x) // chunks)
                if micro // 2 >= self.min_micro:
                    self.chunks *= 2
                    self.stats["max_micro_batches"] = max(self.stats["max_micro_batches"], self.chunks)
                    self._info(f"[gpu-fallback] out of GPU memory -> {self.chunks} micro-batches of "
                               f"~{-(-len(x) // self.chunks)} images (same gradient, gradient accumulation)")
                    continue
                self._to_fallback(model, optimizer, f"out of GPU memory even with micro-batches of {micro}")

    def _do_step(self, model, optimizer, x, y, loss_sum_fn, denom, scaler, amp, amp_dtype, grad_clip, dev, chunks):
        use_amp = self.amp_enabled(amp)
        use_scaler = scaler is not None and scaler.is_enabled() and dev.type == "cuda"
        # phase 1: gradients. An OOM here is harmless (nothing updated yet): the caller retries.
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for xc, yc in zip(torch.tensor_split(x, chunks), torch.tensor_split(y, chunks)):
            if len(xc) == 0:
                continue
            xc, yc = self._input(xc, dev), yc.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(xc)
            loss = loss_sum_fn(logits.float(), yc) / denom
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss; try a smaller learning rate")
            (scaler.scale(loss) if use_scaler else loss).backward()
            total += float(loss.detach())
            del logits, loss
        # phase 2: update. Not retried: an in-place update interrupted by an OOM cannot be repeated safely.
        try:
            if use_scaler:
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        except Exception as exc:
            if not (is_oom(exc) and self.enabled and self.mode != "off" and dev == self.preferred):
                raise
            self._step_oom(model, optimizer, scaler)
            return total
        if dev != self.preferred:
            self.stats["fallback_batches"] += 1
        self._wait_streak = 0
        return total

    def _step_oom(self, model, optimizer, scaler):
        """OOM inside optimizer.step(): this one update may be partially applied (counted in step_oom, so the
        run is flagged by script 16). Continue from a clean state on the fallback device."""
        self.stats["oom_events"] += 1
        self.stats["step_oom"] += 1
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and hasattr(scaler, "_per_optimizer_states"):
            scaler._per_optimizer_states.pop(id(optimizer), None)   # forget "unscale_ already called"
        self._cleanup()
        self._to_fallback(model, optimizer, "out of GPU memory inside optimizer.step (this update may be "
                                            "incomplete; the run is flagged)")

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def forward(self, model, x, fn: Callable | None = None):
        """model(x) (or fn(model, x)) for inference, robust to OOM: splits the batch, then falls back."""
        self.maybe_return(model)
        fn = fn or (lambda m, t: m(t))
        return self._forward(model, x, fn, max(1, self.chunks))

    def _forward(self, model, x, fn, chunks):
        while True:
            dev = self.current
            try:
                outs = []
                for xc in torch.tensor_split(x, min(chunks, len(x))) if dev == self.preferred else [x]:
                    amp_on = dev.type == "cuda" and self.infer_amp_dtype is not None
                    with torch.autocast(device_type=dev.type, dtype=self.infer_amp_dtype or torch.float32,
                                        enabled=amp_on):
                        outs.append(fn(model, self._input(xc, dev)).float().cpu())
                if dev != self.preferred:
                    self.stats["fallback_batches"] += 1
                self._wait_streak = 0
                return torch.cat(outs)
            except Exception as exc:
                if not (is_oom(exc) and self.enabled and self.mode != "off" and dev == self.preferred):
                    raise
                self.stats["oom_events"] += 1
                self._cleanup()
                if -(-len(x) // chunks) // 2 >= 1 and chunks < len(x):
                    chunks *= 2
                    continue
                self._to_fallback(model, None, "out of GPU memory during inference")


def _init_state(optimizer) -> None:
    scalar = torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state[p]
            if len(st):
                continue
            if isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
                if group.get("capturable") or group.get("fused"):
                    raise ValueError("init_optimizer_state does not support capturable/fused Adam")
                st["step"] = torch.tensor(0.0, dtype=scalar)
                st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if group.get("amsgrad"):
                    st["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            elif isinstance(optimizer, torch.optim.SGD):
                if group.get("momentum", 0):
                    st["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            else:
                raise TypeError(f"init_optimizer_state: unsupported optimizer {type(optimizer).__name__}")


# ---------------------------------------------------------------------- memory plan (fairness)
def has_batchnorm(model: torch.nn.Module) -> bool:
    """Micro-batches give exactly the full-batch gradient EXCEPT for BatchNorm (statistics per micro-batch)."""
    return any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def plan_micro_batches(try_step: Callable[[int], float], batch_size: int, min_micro: int, budget: float):
    """Smallest number of micro-batches (1, 2, 4, ...) whose measured peak memory fits in ``budget``.

    ``try_step(chunks)`` runs a few real training steps with that split and returns the peak memory
    (same unit as ``budget``), or raises an out-of-memory error. Returns (chunks, peak), or (None, None)
    when even micro-batches of ``min_micro`` images do not fit."""
    chunks = 1
    while chunks == 1 or -(-batch_size // chunks) >= max(1, min_micro):   # the full batch is always tried
        try:
            peak = float(try_step(chunks))
        except Exception as exc:
            if not is_oom(exc):
                raise
            peak = float("inf")
        if peak <= budget:
            return chunks, peak
        if chunks >= batch_size:
            break
        chunks *= 2
    return None, None
