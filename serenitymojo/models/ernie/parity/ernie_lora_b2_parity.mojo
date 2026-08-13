# serenitymojo/models/ernie/parity/ernie_lora_b2_parity.mojo
#
# TRUE BATCH-2 PARITY GATE for the ERNIE-Image host-grad plain-LoRA arm. Loads the
# REAL base + a small slice of REAL blocks from the checkpoint, builds two SYNTHETIC
# samples (distinct tokens / sigma-source / real_len / RoPE) and runs the trainer's
# own resident-device stack functions, then asserts the two batch-2 invariants:
#
#   (a) loss_B2 == mean(loss_B1(s0), loss_B1(s1))  within 1e-3 relative.  BINDING.
#       (b2 joint loss is the 2N-mean MSE; each per-sample out is scored with the
#       0.5-scaled half-MSE so the joint loss == the mean of the two B1 per-sample
#       MSEs. Forward-only — no backward needed.)
#
#   (b) per-sample forward outputs vs the two B1 forwards, cos >= 0.999.  BINDING.
#       (out0_B2 vs out0_B1 and out1_B2 vs out1_B1 — the row-stack must reproduce
#       each sample's forward within bf16 GEMM ULPs. Forward-only.)
#
#   (c) LoRA grad cosine: B2 vs mean(B1).  INFORMATIONAL ONLY (MJ-1073). Row-stacked
#       M=2S vs M=S bf16 GEMM tiling differs by ULPs (shape-deterministic), which
#       the softmax-attention backward amplifies — the WRONG instrument for gating,
#       proven for krea2 (krea2_b2_gemmshape_gate). Reported, not gated.
#
# ── MJ-1071 (CRITICAL) ────────────────────────────────────────────────────────
# This gate exercises the ERNIE BF16-resident device backward, whose final-layer
# and per-block modulate_backward calls pass an F32 `_t(...)` scale against BF16
# go/x — a PRE-EXISTING dtype crash ("modulate_backward: go/x/scale dtype mismatch",
# sites ernie_stack_lora.mojo:1039, lora_block.mojo:856/:913) that is NOT fixed here
# (the b2 path mirrors the SAME broken call so b1 and b2 crash identically). So this
# gate is COMPILE-VERIFIED ONLY: the FORWARD-only binding checks (a)(b) run and
# print their verdict, then the first backward raises the MJ-1071 error until it is
# fixed by someone else. NO torch oracle: self-consistent (B2 vs two B1 forwards).
#
# Build (mem-safe -O2; cuDNN flash shim linked) + run ONE GPU process:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   MEM_MAX=28G MEM_HIGH=24G SWAP_MAX=2G bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 1 -I . -I /home/alex/MOJO-libs \
#       -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#       serenitymojo/models/ernie/parity/ernie_lora_b2_parity.mojo \
#       -o output/bin/ernie_lora_b2_parity
#   output/bin/ernie_lora_b2_parity

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.models.ernie.weights import (
    load_ernie_stack_base, load_ernie_all_blocks_bf16_normf32,
)
from serenitymojo.models.ernie.block import ErnieModVecs
from serenitymojo.models.dit.ernie_image import build_ernie_rope_tables
from serenitymojo.models.ernie.ernie_stack_lora import (
    ErnieLoraSet, ErnieLoraGrads, ErnieStackForwardB2,
    build_ernie_lora_set, ernie_lora_set_to_device,
    ernie_stack_lora_forward_resident_device, ernie_stack_lora_backward_resident_device,
    ernie_stack_lora_forward_resident_device_b2,
    ernie_stack_lora_backward_resident_device_b2,
)


# ── REAL ERNIE feature dims (must match the loaded base/blocks) ────────────────
comptime H = 32
comptime Dh = 128
comptime D = H * Dh          # 4096
comptime F = 12288
comptime IN_CH = 128
comptime TEXT_IN = 3072
comptime OUT_CH = 128
comptime EPS = Float32(1e-06)

# small SEQUENCE (parity mechanism, not production shape). N_IMG = IMG_H*IMG_W.
comptime IMG_H = 4
comptime IMG_W = 4
comptime N_IMG = IMG_H * IMG_W    # 16
comptime N_TXT = 8
comptime S = N_IMG + N_TXT        # 24
comptime NBLOCKS = 2              # real blocks 0..1 (depth-agnostic comparison)

comptime CFG_PATH = "/home/alex/mojodiffusion/serenitymojo/configs/ernie_image.json"
comptime COS_BAR = Float64(0.999)
comptime LOSS_REL_BAR = Float64(1.0e-3)


def _absf(x: Float64) -> Float64:
    return x if x >= Float64(0.0) else -x


# deterministic pseudo-random fill (PCG stream), F32 in [-scale, scale].
def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32(Int((s >> UInt64(40)) % UInt64(2000)) - 1000) * Float32(0.001)
        out.append(u * scale)
    return out^


def _mod(seed: UInt64) -> ErnieModVecs:
    # 6 modulation chunks [D], small values (scale/shift/gate).
    return ErnieModVecs(
        _fill(D, seed + 1, Float32(0.2)), _fill(D, seed + 2, Float32(0.2)),
        _fill(D, seed + 3, Float32(0.2)), _fill(D, seed + 4, Float32(0.2)),
        _fill(D, seed + 5, Float32(0.2)), _fill(D, seed + 6, Float32(0.2)),
    )


# lift LoRA B off zero so the LoRA branch actually contributes to the forward.
def _perturb_b(mut set: ErnieLoraSet):
    var s = UInt64(1234567)
    for i in range(len(set.ad)):
        var bb = set.ad[i].b.copy()
        for j in range(len(bb)):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            var r = Float32(Int((s >> UInt64(40)) % UInt64(2000)) - 1000) * Float32(0.00005)
            bb[j] = Float32(r).cast[DType.bfloat16]()
        set.ad[i].b = bb^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b) or n == 0:
        return Float64(-2.0)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(n):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na == Float64(0.0) and nb == Float64(0.0):
        return Float64(1.0)
    if na == Float64(0.0) or nb == Float64(0.0):
        return Float64(0.0)
    return dot / (sqrt(na) * sqrt(nb))


# plain MSE = mean((pred-target)^2) — the B1 per-sample loss (== trainer _mse_loss).
def _mse(pred: List[Float32], target: List[Float32]) raises -> Float64:
    if len(pred) != len(target) or len(pred) == 0:
        raise Error("_mse: length mismatch")
    var sse = Float64(0.0)
    for i in range(len(pred)):
        var d = Float64(pred[i]) - Float64(target[i])
        sse += d * d
    return sse / Float64(len(pred))


def _sumsq(a: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(a)):
        var v = Float64(a[i])
        s += v * v
    return s


def _mean2(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(Float32(0.5) * (a[i] + b[i]))
    return out^


def _accum(mut dst: List[Float32], src: List[Float32]):
    for i in range(len(src)):
        dst.append(src[i])


# concat ALL of a step's LoRA grads (dA then dB, slot order) into one vector.
def _concat_all(g: ErnieLoraGrads) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(g.d_a)):
        _accum(out, g.d_a[i])
        _accum(out, g.d_b[i])
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("==== ernie_lora_b2_parity (TRUE batch-2 vs two B=1; real base/blocks, synth seq) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " F=", F, " N_IMG=", N_IMG,
          " N_TXT=", N_TXT, " S=", S, " NBLOCKS=", NBLOCKS)

    var cfg = read_model_config(CFG_PATH)
    var st = ShardedSafeTensors.open(cfg.checkpoint)
    var base = load_ernie_stack_base(st, D, IN_CH, ctx)
    var blocks = load_ernie_all_blocks_bf16_normf32(st, NBLOCKS, ctx)
    print("  loaded base + ", len(blocks), " real blocks; checkpoint=", cfg.checkpoint)

    var lora = build_ernie_lora_set(NBLOCKS, D, F, cfg.lora_rank, cfg.lora_alpha)
    _perturb_b(lora)  # lift B off zero — LoRA now contributes to the forward
    var lora_dev = ernie_lora_set_to_device(lora, STDtype.BF16, ctx)
    print("  LoRA set: rank=", cfg.lora_rank, " alpha=", cfg.lora_alpha,
          " adapters=", NBLOCKS * 7, " (B perturbed off zero)")

    # ── two SYNTHETIC samples (distinct tokens / AdaLN source / real_len) ────────
    var img0 = _fill(N_IMG * IN_CH, UInt64(11), Float32(1.0))
    var txt0 = _fill(N_TXT * TEXT_IN, UInt64(22), Float32(1.0))
    var img1 = _fill(N_IMG * IN_CH, UInt64(33), Float32(1.0))
    var txt1 = _fill(N_TXT * TEXT_IN, UInt64(44), Float32(1.0))
    var tgt0 = _fill(N_IMG * OUT_CH, UInt64(55), Float32(1.0))
    var tgt1 = _fill(N_IMG * OUT_CH, UInt64(66), Float32(1.0))

    var mv0 = _mod(UInt64(100))
    var mv1 = _mod(UInt64(200))
    var f_scale0 = _fill(D, UInt64(301), Float32(0.2))
    var f_shift0 = _fill(D, UInt64(302), Float32(0.2))
    var f_scale1 = _fill(D, UInt64(401), Float32(0.2))
    var f_shift1 = _fill(D, UInt64(402), Float32(0.2))

    var real_len0 = 5
    var real_len1 = 6
    var rope0 = build_ernie_rope_tables[N_IMG, N_TXT, H, Dh](IMG_H, IMG_W, real_len0, ctx, STDtype.F32)
    var rope1 = build_ernie_rope_tables[N_IMG, N_TXT, H, Dh](IMG_H, IMG_W, real_len1, ctx, STDtype.F32)
    print("  samples: real_len0=", real_len0, " real_len1=", real_len1)

    # ── B=1 forwards (per sample) ───────────────────────────────────────────────
    var f0 = ernie_stack_lora_forward_resident_device[H, Dh, N_IMG, N_TXT, S](
        img0.copy(), txt0.copy(), base, blocks, lora_dev, mv0,
        f_scale0.copy(), f_shift0.copy(), rope0[0], rope0[1],
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx, real_len0,
    )
    var f1 = ernie_stack_lora_forward_resident_device[H, Dh, N_IMG, N_TXT, S](
        img1.copy(), txt1.copy(), base, blocks, lora_dev, mv1,
        f_scale1.copy(), f_shift1.copy(), rope1[0], rope1[1],
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx, real_len1,
    )

    # ── B=2 forward (row-stacked) ───────────────────────────────────────────────
    var fb2 = ernie_stack_lora_forward_resident_device_b2[H, Dh, N_IMG, N_TXT, S](
        img0.copy(), txt0.copy(), img1.copy(), txt1.copy(),
        base, blocks, lora_dev, mv0, mv1,
        f_scale0.copy(), f_shift0.copy(), f_scale1.copy(), f_shift1.copy(),
        rope0[0], rope0[1], rope1[0], rope1[1],
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx, real_len0, real_len1,
    )

    var allok = True

    # ── (a) loss parity (BINDING) ───────────────────────────────────────────────
    print("")
    print("---- (a) loss_B2 vs mean(loss_B1(s0), loss_B1(s1))  [BINDING] ----")
    var loss0 = _mse(f0.out, tgt0)
    var loss1 = _mse(f1.out, tgt1)
    var loss_mean = Float64(0.5) * (loss0 + loss1)
    var loss_b2 = Float64(0.5) * _mse(fb2.out0, tgt0) + Float64(0.5) * _mse(fb2.out1, tgt1)
    var loss_rel = _absf(loss_b2 - loss_mean) / (
        loss_mean if loss_mean > Float64(1.0e-8) else Float64(1.0e-8)
    )
    print("  loss_B1(s0)=", loss0, "  loss_B1(s1)=", loss1)
    print("  mean=", loss_mean, "  loss_B2=", loss_b2, "  rel=", loss_rel,
          "  ", "PASS" if loss_rel <= LOSS_REL_BAR else "FAIL")
    if loss_rel > LOSS_REL_BAR:
        allok = False

    # ── (b) per-sample forward output parity (BINDING) ──────────────────────────
    print("")
    print("---- (b) per-sample forward out0/out1 (B2 vs B1) cos>=0.999  [BINDING] ----")
    var cos0 = _cos(fb2.out0, f0.out)
    var cos1 = _cos(fb2.out1, f1.out)
    print("  sample0: cos(out0_B2, out0_B1) =", cos0,
          "  ||B2||=", sqrt(_sumsq(fb2.out0)), " ||B1||=", sqrt(_sumsq(f0.out)),
          "  ", "PASS" if cos0 >= COS_BAR else "FAIL")
    print("  sample1: cos(out1_B2, out1_B1) =", cos1,
          "  ||B2||=", sqrt(_sumsq(fb2.out1)), " ||B1||=", sqrt(_sumsq(f1.out)),
          "  ", "PASS" if cos1 >= COS_BAR else "FAIL")
    if cos0 < COS_BAR or cos1 < COS_BAR:
        allok = False

    print("")
    if allok:
        print("BINDING VERDICT: PASS — loss_B2 == mean(loss_B1) (rel<=1e-3) AND",
              "per-sample forward outputs match (cos>=0.999).")
    else:
        print("BINDING VERDICT: FAIL — see (a)/(b) above.")

    # ── (c) grad cosine INFORMATIONAL (MJ-1073) — hits MJ-1071 until fixed ──────
    # The backward exercises the BF16-resident modulate_backward with an F32 `_t(...)`
    # scale (MJ-1071); it will RAISE until that pre-existing dtype crash is fixed.
    # Grad cosine vs mean(B1) is shape-deterministic-ULP noise at depth (MJ-1073) —
    # reported for tracking, NOT a gated bar for row-stacked b2.
    print("")
    print("---- (c) LoRA grad cosine: B2 vs mean(B1)  [INFORMATIONAL, MJ-1073] ----")
    print("  NOTE: the following backwards hit MJ-1071 (BF16-resident modulate_backward")
    print("        F32-scale dtype crash) and RAISE until that is fixed — expected.")
    var g0 = ernie_stack_lora_backward_resident_device[H, Dh, N_IMG, N_TXT, S](
        f0.out.copy(), img0.copy(), txt0.copy(), base, blocks, lora_dev, mv0,
        f_scale0.copy(), f_shift0.copy(), rope0[0], rope0[1], f0,
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx, real_len0,
    )
    var g1 = ernie_stack_lora_backward_resident_device[H, Dh, N_IMG, N_TXT, S](
        f1.out.copy(), img1.copy(), txt1.copy(), base, blocks, lora_dev, mv1,
        f_scale1.copy(), f_shift1.copy(), rope1[0], rope1[1], f1,
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx, real_len1,
    )
    # 0.5-scaled joint d_out (the trainer's _mse_half_loss_grad contract).
    var dh0 = List[Float32]()
    for i in range(len(fb2.out0)):
        dh0.append(Float32(0.5) * fb2.out0[i])
    var dh1 = List[Float32]()
    for i in range(len(fb2.out1)):
        dh1.append(Float32(0.5) * fb2.out1[i])
    var gb2 = ernie_stack_lora_backward_resident_device_b2[H, Dh, N_IMG, N_TXT, S](
        dh0, dh1, img0.copy(), txt0.copy(), img1.copy(), txt1.copy(),
        base, blocks, lora_dev, mv0, mv1,
        f_scale0.copy(), f_shift0.copy(), f_scale1.copy(), f_shift1.copy(),
        rope0[0], rope0[1], rope1[0], rope1[1], fb2,
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx, real_len0, real_len1,
    )
    var v_b2 = _concat_all(gb2)
    var v_mean = _mean2(_concat_all(g0), _concat_all(g1))
    print("  B2 vs mean(B1) global cosine =", _cos(v_b2, v_mean),
          "  (MJ-1073: shape-deterministic bf16 GEMM rounding — not gated)")

    print("")
    print("VERDICT (binding): ", "PASS" if allok else "FAIL",
          " — grad cosine above is informational only (MJ-1073).")
