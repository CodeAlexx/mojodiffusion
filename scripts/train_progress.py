#!/usr/bin/env python3
# Rust-style progress view for the Mojo Klein trainer.
# Pipe the trainer's stdout/stderr through this:
#   stdbuf -oL /tmp/train_klein_real 2>&1 | python3 scripts/train_progress.py
# It keeps the raw Mojo stream in output/train_klein_real.log and prints concise
# [Klein-lora] progress lines matching the Rust trainer's operator view.
import math
import os
import re
import sys
import time

LOG = os.environ.get("TRAIN_PROGRESS_RAW_LOG", os.path.join("output", "train_klein_real.log"))
SCREEN_LOG = os.environ.get(
    "TRAIN_PROGRESS_SCREEN_LOG", os.path.join("output", "train_klein_real_screen.log")
)
os.makedirs("output", exist_ok=True)
read_only = os.environ.get("TRAIN_PROGRESS_READ_ONLY", "0") == "1"
log = None if read_only else open(LOG, "a", buffering=1)
screen_log = open(SCREEN_LOG, "a", buffering=1)

prog_re = re.compile(r"PROG\s+step=\s*(\d+)\s+total=\s*(\d+)\s+loss=\s*([-\d.eE+]+)\s+grad=\s*([-\d.eE+]+)\s+lr=\s*([-\d.eE+]+)\s+clip=\s*([-\d.eE+]+)\s+secs=\s*([-\d.eE+]+)")
evt_re  = re.compile(r"EVENT\s+(\w+)\s+(.*)")
stage_re = re.compile(r"PROG_STAGE\s+step=\s*(\d+)\s+total=\s*(\d+)\s+phase=([a-z_]+)(.*)")
sample_re = re.compile(r"SAMPLE\s+(.*)")
sample_step_re = re.compile(r"denoise\s+step=\s*(\d+)\s+total=\s*(\d+)(.*)")
cache_re = re.compile(r"cache (?:compatible )?samples:\s*(\d+)")

DESC = "klein_alina_lora"
start = time.monotonic()
samples = 0
noise_mps = {}
show_stages = os.environ.get("TRAIN_PROGRESS_STAGES", "0") == "1"


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds + 0.5))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def _epoch(step: int, total: int) -> tuple[int, int]:
    if samples <= 0:
        return (1, 1)
    return ((step - 1) // samples + 1, max(1, math.ceil(total / samples)))


def emit(line: str) -> None:
    print(line, flush=True)
    screen_log.write(line + "\n")


for raw in sys.stdin:
    line = raw.rstrip("\n")
    if log is not None:
        log.write(line + "\n")
    cm = cache_re.search(line)
    if cm:
        samples = int(cm.group(1))
        emit(line)
        continue

    m = prog_re.search(line)
    if m:
        step, total, loss, grad, lr, clip, secs = m.groups()
        step, total = int(step), int(total)
        sec_f = float(secs)
        elapsed = time.monotonic() - start
        eta = sec_f * max(0, total - step)
        ep, ep_total = _epoch(step, total)
        noise = noise_mps.pop(step, None)
        noise_part = f" | noise {noise:.1f}M/s" if noise is not None else ""
        emit(
            f"[Klein-lora] step {step:>4}/{total:<4} | "
            f"epoch {ep}/{ep_total} | "
            f"loss {float(loss):.4f} | grad_norm {float(grad):.4f} | "
            f"{sec_f:.1f}s/step{noise_part} | elapsed {_hms(elapsed)} | ETA {_hms(eta)}"
        )
        continue

    e = evt_re.search(line)
    if e:
        kind, rest = e.groups()
        emit(f"[Klein-lora] {kind}: {rest}")
        continue

    st = stage_re.search(line)
    if st:
        step, total, phase, rest = st.groups()
        if phase == "noise":
            nm = re.search(r"elems_per_sec=\s*([-\d.eE+]+)", rest)
            if nm:
                noise_mps[int(step)] = float(nm.group(1)) / 1.0e6
        if show_stages and phase in ("noise", "forward", "backward", "optim", "load_sample"):
            emit(
                f"[Klein-lora] step {int(step):>4}/{int(total):<4} | "
                f"phase {phase}{rest}"
            )
        continue

    s = sample_re.search(line)
    if s:
        body = s.group(1)
        sm = sample_step_re.search(body)
        if sm:
            step_s, total_s, rest = sm.groups()
            step_i, total_i = int(step_s), int(total_s)
            every = int(os.environ.get("TRAIN_PROGRESS_SAMPLE_EVERY", "5"))
            if step_i == 1 or step_i == total_i or (every > 0 and step_i % every == 0):
                emit(f"[Klein-sample] denoise step={step_i} total={total_i}{rest}")
        elif body.startswith("setup") or body.startswith("saved"):
            emit(f"[Klein-sample] {body}")
        continue

    if any(k in line for k in ("===", "config:", "512px latent", "arch:", "cadence", "[load]", "block stream", "LoRA set", "scratch ring", "PASS", "FAIL", "saved", "Error", "error", "OOM")):
        emit(line)
if log is not None:
    log.close()
screen_log.close()
