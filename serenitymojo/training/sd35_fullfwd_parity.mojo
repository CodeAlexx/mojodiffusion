# Full-model forward parity gate: Mojo sd3.5-medium stack forward vs diffusers.
# Loads ff_noisy/ff_txt/ff_pooled + ff_ref dumped by sd35_fullfwd_oracle.py, runs
# the offload stack forward (LoRA B=0 -> no-op) on the real medium ckpt, and
# compares fwd.out vs the diffusers velocity. bf16 floor expected -> PASS cos>=0.99.
#
# Run (oracle FIRST):
#   /home/alex/OneTrainer/venv/bin/python serenitymojo/models/sd35/parity/sd35_fullfwd_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/sd35_fullfwd_parity.mojo -o /tmp/sd35_fullfwd
#   /tmp/sd35_fullfwd

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sd35.weights import load_sd35_stack_base
from serenitymojo.models.sd35.sd35_stack_lora import (
    build_sd35_lora_set, sd35_stack_lora_forward_offload,
)
from serenitymojo.offload.plan import build_sd35_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

comptime H = 24
comptime Dh = 64
comptime D = H * Dh
comptime FMLP = 6144
comptime IN_CH = 64
comptime TXT_CH = 4096
comptime OUT_CH = 64
comptime NUM_JOINT = 24
comptime NUM_DUAL = 13
comptime TIMESTEP_DIM = 256
comptime POOLED_DIM = 2048
comptime EPS = Float32(1e-6)
comptime QK_EPS = Float32(1e-6)
comptime N_IMG = 4096
comptime N_TXT = 154
comptime S = N_TXT + N_IMG
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime CKPT = "/home/alex/.serenity/models/checkpoints/sd3.5_medium.safetensors"
comptime REF = "/home/alex/mojodiffusion/serenitymojo/models/sd35/parity/"


def _read(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0: raise Error(String("open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd); raise Error(String("empty (run oracle): ") + path)
    var buf = alloc[UInt8](n); var done = 0
    while done < n:
        var g = sys_pread(fd, buf + done, n - done, done)
        if g <= 0: break
        done += g
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32](); var o = List[Float32]()
    for i in range(n // 4): o.append(fp[i])
    buf.free(); return o^


def main() raises:
    var ctx = DeviceContext()
    print("==== sd35 FULL-MODEL forward parity (Mojo stack vs diffusers) ====")
    var base_st = SafeTensors.open(CKPT)
    var base = load_sd35_stack_base(base_st, ctx)
    var plan = build_sd35_block_plan(NUM_JOINT)
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(CKPT, plan^, cfg, ctx)
    var lora = build_sd35_lora_set(NUM_JOINT, D, FMLP, RANK, ALPHA)

    var noisy = _read(REF + "ff_noisy.bin")
    var txt = _read(REF + "ff_txt.bin")
    var pooled = _read(REF + "ff_pooled.bin")
    var refv = _read(REF + "ff_ref.bin")
    print("[in] noisy", len(noisy), " txt", len(txt), " pooled", len(pooled), " ref", len(refv))

    var fwd = sd35_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
        noisy.copy(), txt.copy(), pooled.copy(), Float32(0.5),
        base, loader, lora,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
        EPS, QK_EPS, ctx, NUM_DUAL, True,
    )

    var harness = ParityHarness()
    var r = harness.compare_host(fwd.out, refv)
    print("")
    print("  cos(full_fwd) =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n)
    print("")
    # bf16-floor tolerance: structural-correctness threshold 0.99
    if r.cos >= 0.99:
        print("VERDICT: PASS — Mojo sd3.5-medium stack forward matches diffusers (cos>=0.99, bf16 floor)")
    else:
        print("VERDICT: FAIL — forward diverges from diffusers (cos=", r.cos, ")")
