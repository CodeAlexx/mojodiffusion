# serenitymojo/models/flux/parity/flux_b2_parity.mojo
#
# TRUE BATCH-2 (row-stacked) flux DEVICE-stack parity gate (MJ-1073 design).
#
# Two DISTINCT samples (s0, s1) with different tokens AND different timestep/vec
# (a real b2 pair whose sigmas -> embed -> [2,D] mods differ). The b2 device stack
# processes both in ONE call; the b1 device stack is the per-sample oracle (run
# twice). Both arms are BLOCK-PROJECTION LoRA ONLY (stack-level LoRA disabled via
# an EMPTY FluxStackLoraSet — the b1 conductor then runs frozen embed/mod, exactly
# what the b2 arm does). Bars follow the krea2/chroma/qwen MJ-1073 re-baseline:
#   BINDING:
#     (b) loss-parity:   |loss_b2 - mean(loss_s0, loss_s1)| / mean  <= 1e-3
#     (c) per-sample fwd: cos(b2.out_s, b1.out_s)                    >= 0.999
#     (a) b2dup:          b2(s0,s0) reproduces b1(s0) loss (rel<=1e-3)
#                         AND per-sample fwd cos(b2dup.out_s, s0.out) >= 0.999
#   INFORMATIONAL (NOT gated — M=2N vs M=N GEMM bf16 tiling makes grad-cos-vs-b1
#   the wrong instrument at depth; reported with the qwen _l2z zero-slot guard):
#     grad-cos:          cos(grad_b2.d_a/d_b[i], mean(grad_s0, grad_s1)[i])
#     b2dup grad-cos:    cos(grad_b2dup[i], grad_s0[i])
#
# CONTRACT under test: b2 sums the two samples' per-adapter d_a/d_b IN-GEMM
# (M=2N); the caller pre-scales each sample's d_out by 0.5, so the sum = the mean
# = the grad-accum=2 gradient. loss_b2 = 0.5*(loss_s0 + loss_s1).
#
# Synthetic reduced-depth checkpoint (small H/Dh) — base + block weights mirror
# flux_stack_device_parity's BFL keys. Each arm gets its OWN freshly-opened loader.
# The device blocks default FLASH=True (S padded to 128 internally, real_len=S);
# both arms use it, so the per-sample comparison is flash-vs-flash.
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run flux-b2-parity-build && output/bin/flux_b2_parity

from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.train_step import LoraAdapter

from serenitymojo.models.flux.flux_stack import (
    FluxStackBase, EmbedMlp, ModLin, DoubleModLin,
)
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxStackLoraSet, build_flux_lora_set, empty_flux_stack_lora_set,
    total_adapters,
    flux_stack_lora_forward_device_offload_full,
    flux_stack_lora_backward_device_offload_full,
    flux_stack_lora_forward_device_offload_full_b2,
    flux_stack_lora_backward_device_offload_full_b2,
)
from serenitymojo.offload.plan import build_flux_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader


comptime TArc = ArcPointer[Tensor]
comptime CKPT_PATH = "/tmp/flux_b2_parity.safetensors"
comptime H = 2
comptime Dh = 8
comptime D = H * Dh            # 16
comptime N_IMG = 3
comptime N_TXT = 2
comptime S = N_TXT + N_IMG     # 5 (txt FIRST then img)
comptime FMLP = 32
comptime IN_CH = 8
comptime TXT_CH = 12
comptime OUT_CH = 8
comptime T_DIM = 16
comptime VEC_DIM = 20
comptime NUM_DOUBLE = 2
comptime NUM_SINGLE = 2
comptime EPS = Float32(1.0e-06)
comptime MAX_PERIOD = Float32(10000.0)
comptime RANK = 4
comptime ALPHA = Float32(4.0)
comptime COS_BAR = 0.999
comptime LOSS_REL_BAR = Float64(1.0e-03)


def _rand(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _ones(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(1.0))
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _arc_bf16(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals.copy(), shape^, STDtype.BF16, ctx))


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("cos: zero vector")
    return dot / (sqrt(na) * sqrt(nb))


def _absf(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _mse(pred: List[Float32], target: List[Float32]) raises -> Float64:
    if len(pred) != len(target):
        raise Error("mse: length mismatch")
    var sse = Float64(0.0)
    for i in range(len(pred)):
        var d = Float64(pred[i]) - Float64(target[i])
        sse += d * d
    return sse / Float64(len(pred))


# d_loss = scale * (2/N) * (pred - target)  (scale=1.0 for b1; scale=0.5 for b2).
def _dloss(pred: List[Float32], target: List[Float32], scale: Float32) raises -> List[Float32]:
    var inv_n = Float32(2.0) / Float32(len(pred))
    var out = List[Float32]()
    for i in range(len(pred)):
        out.append(scale * inv_n * (pred[i] - target[i]))
    return out^


# mean of two grad lists: 0.5*(a + b).
def _mean2(a: List[Float32], b: List[Float32]) raises -> List[Float32]:
    if len(a) != len(b):
        raise Error("mean2: length mismatch")
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(Float32(0.5) * (a[i] + b[i]))
    return out^


# sum of squares (zero-vector guard for the informational grad-cos section).
def _l2z(v: List[Float32]) -> Float64:
    var acc: Float64 = 0.0
    for i in range(len(v)):
        acc += Float64(v[i]) * Float64(v[i])
    return acc


# rebuild the block LoRA set with NONZERO random A and B (moments zeroed) so the
# LoRA contributes and grads are meaningful (make_lora_adapter zero-inits B).
def _nonzero_lora_set(src: FluxLoraSet, seed0: UInt64) -> FluxLoraSet:
    var ad = List[LoraAdapter]()
    var seed = seed0
    for i in range(len(src.ad)):
        var rank = src.ad[i].rank
        var in_f = src.ad[i].in_f
        var out_f = src.ad[i].out_f
        var scale = src.ad[i].scale
        var na = rank * in_f
        var nb = out_f * rank
        var a = _rand(na, seed, Float32(0.30)); seed += 1
        var b = _rand(nb, seed, Float32(0.30)); seed += 1
        ad.append(LoraAdapter(
            a^, b^, rank, in_f, out_f, scale,
            _zeros(na), _zeros(na), _zeros(nb), _zeros(nb),
        ))
    return FluxLoraSet(ad^, src.num_double, src.num_single, src.rank)


def _add(
    mut names: List[String], mut tensors: List[TArc],
    name: String, vals: List[Float32], var shape: List[Int], ctx: DeviceContext,
) raises:
    # Real flux1-dev blocks are bf16; write bf16 (bf16-native block chain).
    names.append(name)
    tensors.append(_arc_bf16(vals, shape^, ctx))


def _write_double_block(
    mut names: List[String], mut tensors: List[TArc], dp: String, seed: UInt64, ctx: DeviceContext
) raises:
    var streams: List[String] = ["img", "txt"]
    for si in range(2):
        var sd = seed + UInt64(si * 50)
        var ap = dp + "." + streams[si] + "_attn"
        var mp = dp + "." + streams[si] + "_mlp"
        _add(names, tensors, ap + ".qkv.weight", _rand(3 * D * D, sd + 1, 0.04), [3 * D, D], ctx)
        _add(names, tensors, ap + ".qkv.bias", _rand(3 * D, sd + 2, 0.02), [3 * D], ctx)
        _add(names, tensors, ap + ".proj.weight", _rand(D * D, sd + 3, 0.04), [D, D], ctx)
        _add(names, tensors, ap + ".proj.bias", _rand(D, sd + 4, 0.02), [D], ctx)
        _add(names, tensors, mp + ".0.weight", _rand(FMLP * D, sd + 5, 0.04), [FMLP, D], ctx)
        _add(names, tensors, mp + ".0.bias", _rand(FMLP, sd + 6, 0.02), [FMLP], ctx)
        _add(names, tensors, mp + ".2.weight", _rand(D * FMLP, sd + 7, 0.04), [D, FMLP], ctx)
        _add(names, tensors, mp + ".2.bias", _rand(D, sd + 8, 0.02), [D], ctx)
        _add(names, tensors, ap + ".norm.query_norm.scale", _ones(Dh), [Dh], ctx)
        _add(names, tensors, ap + ".norm.key_norm.scale", _ones(Dh), [Dh], ctx)


def _write_single_block(
    mut names: List[String], mut tensors: List[TArc], sp: String, seed: UInt64, ctx: DeviceContext
) raises:
    _add(names, tensors, sp + ".linear1.weight", _rand((3 * D + FMLP) * D, seed + 1, 0.04), [3 * D + FMLP, D], ctx)
    _add(names, tensors, sp + ".linear1.bias", _rand(3 * D + FMLP, seed + 2, 0.02), [3 * D + FMLP], ctx)
    _add(names, tensors, sp + ".linear2.weight", _rand(D * (D + FMLP), seed + 3, 0.04), [D, D + FMLP], ctx)
    _add(names, tensors, sp + ".linear2.bias", _rand(D, seed + 4, 0.02), [D], ctx)
    _add(names, tensors, sp + ".norm.query_norm.scale", _ones(Dh), [Dh], ctx)
    _add(names, tensors, sp + ".norm.key_norm.scale", _ones(Dh), [Dh], ctx)


def _open_loader(ctx: DeviceContext) raises -> TurboPlannedLoader:
    var plan = build_flux_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg = OffloadConfig.synchronous_single()
    return TurboPlannedLoader.open(String(CKPT_PATH), plan^, cfg, ctx)


def _base(ctx: DeviceContext) raises -> FluxStackBase:
    var time_in = EmbedMlp(_rand(D * T_DIM, 100, 0.02), _rand(D, 101, 0.02),
                           _rand(D * D, 102, 0.02), _rand(D, 103, 0.02), T_DIM, D, ctx)
    var guid_in = EmbedMlp(_rand(D * T_DIM, 110, 0.02), _rand(D, 111, 0.02),
                           _rand(D * D, 112, 0.02), _rand(D, 113, 0.02), T_DIM, D, ctx)
    var vec_in = EmbedMlp(_rand(D * VEC_DIM, 120, 0.02), _rand(D, 121, 0.02),
                          _rand(D * D, 122, 0.02), _rand(D, 123, 0.02), VEC_DIM, D, ctx)
    var dbl_mod = List[DoubleModLin]()
    for bi in range(NUM_DOUBLE):
        var sd = UInt64(200 + bi * 10)
        var im = ModLin(_rand(6 * D * D, sd + 1, 0.02), _rand(6 * D, sd + 2, 0.02), 6 * D, D, ctx)
        var tm = ModLin(_rand(6 * D * D, sd + 3, 0.02), _rand(6 * D, sd + 4, 0.02), 6 * D, D, ctx)
        dbl_mod.append(DoubleModLin(im^, tm^))
    var sgl_mod = List[ModLin]()
    for bi in range(NUM_SINGLE):
        var sd = UInt64(300 + bi * 10)
        sgl_mod.append(ModLin(_rand(3 * D * D, sd + 1, 0.02), _rand(3 * D, sd + 2, 0.02), 3 * D, D, ctx))
    return FluxStackBase(
        _rand(D * IN_CH, 400, 0.02), _rand(D, 401, 0.02),
        _rand(D * TXT_CH, 402, 0.02), _rand(D, 403, 0.02),
        time_in^, True, guid_in^, vec_in^,
        dbl_mod^, sgl_mod^,
        _rand(2 * D * D, 404, 0.02), _rand(2 * D, 405, 0.02),
        _rand(OUT_CH * D, 406, 0.02), _rand(OUT_CH, 407, 0.02),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )


def main() raises:
    var ctx = DeviceContext()
    print("==== flux_b2_parity (row-stacked TRUE batch-2; DEVICE stack) ====")
    print("D=", D, " FMLP=", FMLP, " double=", NUM_DOUBLE, " single=", NUM_SINGLE,
          " N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S, " RANK=", RANK)

    var names = List[String]()
    var tensors = List[TArc]()
    for bi in range(NUM_DOUBLE):
        _write_double_block(
            names, tensors,
            String("double_blocks.") + String(bi), UInt64(1000 + bi * 100), ctx,
        )
    for bi in range(NUM_SINGLE):
        _write_single_block(
            names, tensors,
            String("single_blocks.") + String(bi), UInt64(5000 + bi * 100), ctx,
        )
    save_safetensors(names, tensors, String(CKPT_PATH), ctx)

    # two DISTINCT samples (different tokens + timestep + CLIP vec -> different mods).
    var img0 = _rand(N_IMG * IN_CH, UInt64(800), Float32(0.50))
    var txt0 = _rand(N_TXT * TXT_CH, UInt64(801), Float32(0.50))
    var timestep0 = _rand(1, UInt64(802), Float32(1.0))
    var vector0 = _rand(VEC_DIM, UInt64(803), Float32(0.50))
    var tgt0 = _rand(N_IMG * OUT_CH, UInt64(806), Float32(0.50))

    var img1 = _rand(N_IMG * IN_CH, UInt64(900), Float32(0.50))
    var txt1 = _rand(N_TXT * TXT_CH, UInt64(901), Float32(0.50))
    var timestep1 = _rand(1, UInt64(902), Float32(1.0))
    var vector1 = _rand(VEC_DIM, UInt64(903), Float32(0.50))
    var tgt1 = _rand(N_IMG * OUT_CH, UInt64(906), Float32(0.50))

    var guidance = Optional[List[Float32]](_rand(1, UInt64(805), Float32(1.0)))
    var cos = _rand(S * H * (Dh // 2), UInt64(807), Float32(0.50))
    var sin = _rand(S * H * (Dh // 2), UInt64(808), Float32(0.50))

    var lora0 = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
    var lora = _nonzero_lora_set(lora0, UInt64(9000))
    var sset = empty_flux_stack_lora_set(NUM_DOUBLE, NUM_SINGLE, RANK)
    var n = total_adapters(lora)

    # ── b1 oracle: sample 0 (frozen embed/mod via empty sset; block LoRA only) ──
    var base_a = _base(ctx)
    var l_s0 = _open_loader(ctx)
    var f_s0 = flux_stack_lora_forward_device_offload_full[H, Dh, N_IMG, N_TXT, S](
        img0.copy(), txt0.copy(), timestep0.copy(), guidance, vector0.copy(),
        base_a, l_s0, lora, sset, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var loss0 = _mse(f_s0.out, tgt0)
    var g_s0 = flux_stack_lora_backward_device_offload_full[H, Dh, N_IMG, N_TXT, S](
        _dloss(f_s0.out, tgt0, Float32(1.0)), img0.copy(), txt0.copy(),
        base_a, l_s0, lora, sset, vector0.copy(), cos.copy(), sin.copy(), f_s0,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    # ── b1 oracle: sample 1 ──
    var base_b = _base(ctx)
    var l_s1 = _open_loader(ctx)
    var f_s1 = flux_stack_lora_forward_device_offload_full[H, Dh, N_IMG, N_TXT, S](
        img1.copy(), txt1.copy(), timestep1.copy(), guidance, vector1.copy(),
        base_b, l_s1, lora, sset, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var loss1 = _mse(f_s1.out, tgt1)
    var g_s1 = flux_stack_lora_backward_device_offload_full[H, Dh, N_IMG, N_TXT, S](
        _dloss(f_s1.out, tgt1, Float32(1.0)), img1.copy(), txt1.copy(),
        base_b, l_s1, lora, sset, vector1.copy(), cos.copy(), sin.copy(), f_s1,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    # ── b2: (s0, s1) in one call ──
    var base_c = _base(ctx)
    var l_b2 = _open_loader(ctx)
    var f_b2 = flux_stack_lora_forward_device_offload_full_b2[H, Dh, N_IMG, N_TXT, S](
        img0.copy(), txt0.copy(), timestep0.copy(), guidance, vector0.copy(),
        img1.copy(), txt1.copy(), timestep1.copy(), guidance, vector1.copy(),
        base_c, l_b2, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var loss_b2 = Float64(0.5) * (_mse(f_b2.out0, tgt0) + _mse(f_b2.out1, tgt1))
    var g_b2 = flux_stack_lora_backward_device_offload_full_b2[H, Dh, N_IMG, N_TXT, S](
        _dloss(f_b2.out0, tgt0, Float32(0.5)), _dloss(f_b2.out1, tgt1, Float32(0.5)),
        base_c, l_b2, lora, cos.copy(), sin.copy(), f_b2,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    # ── b2dup: (s0, s0) ──
    var base_d = _base(ctx)
    var l_dup = _open_loader(ctx)
    var f_dup = flux_stack_lora_forward_device_offload_full_b2[H, Dh, N_IMG, N_TXT, S](
        img0.copy(), txt0.copy(), timestep0.copy(), guidance, vector0.copy(),
        img0.copy(), txt0.copy(), timestep0.copy(), guidance, vector0.copy(),
        base_d, l_dup, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var loss_dup = Float64(0.5) * (_mse(f_dup.out0, tgt0) + _mse(f_dup.out1, tgt0))
    var g_dup = flux_stack_lora_backward_device_offload_full_b2[H, Dh, N_IMG, N_TXT, S](
        _dloss(f_dup.out0, tgt0, Float32(0.5)), _dloss(f_dup.out1, tgt0, Float32(0.5)),
        base_d, l_dup, lora, cos.copy(), sin.copy(), f_dup,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    var allok = True

    # ── (c) per-sample forward cos vs b1 ──
    print("---- (c) per-sample forward cos (b2 vs b1) ----")
    var c0 = _cos(f_b2.out0, f_s0.out)
    var c1 = _cos(f_b2.out1, f_s1.out)
    var okc0 = c0 >= COS_BAR
    var okc1 = c1 >= COS_BAR
    if not (okc0 and okc1):
        allok = False
    print("  out0 cos=", c0, " ", ("PASS" if okc0 else "FAIL"))
    print("  out1 cos=", c1, " ", ("PASS" if okc1 else "FAIL"))
    print("  b2 nonfinite_lora_grads=", g_b2.nonfinite_lora_grads)

    # ── (b) loss-parity vs mean(loss_s0, loss_s1) ──
    print("---- (b) loss-parity (b2 vs mean(b1)) ----")
    var loss_mean = Float64(0.5) * (loss0 + loss1)
    var lden = loss_mean if loss_mean > 1.0e-8 else 1.0e-8
    var lrel = _absf(loss_b2 - loss_mean) / lden
    var okl = lrel <= LOSS_REL_BAR
    if not okl:
        allok = False
    print("  loss_s0=", loss0, " loss_s1=", loss1, " mean=", loss_mean)
    print("  loss_b2=", loss_b2, " rel=", lrel, " ", ("PASS" if okl else "FAIL"))

    # ── (a) b2dup reproduces b1(s0) ──
    print("---- (a) b2dup (s0,s0) vs b1(s0) ----")
    var dc0 = _cos(f_dup.out0, f_s0.out)
    var dc1 = _cos(f_dup.out1, f_s0.out)
    var okdc = dc0 >= COS_BAR and dc1 >= COS_BAR
    if not okdc:
        allok = False
    var lden0 = loss0 if loss0 > 1.0e-8 else 1.0e-8
    var ldrel = _absf(loss_dup - loss0) / lden0
    var okdl = ldrel <= LOSS_REL_BAR
    if not okdl:
        allok = False
    print("  dup out0 cos=", dc0, " out1 cos=", dc1, " ", ("PASS" if okdc else "FAIL"))
    print("  dup loss=", loss_dup, " vs loss_s0=", loss0, " rel=", ldrel, " ",
          ("PASS" if okdl else "FAIL"))

    # ── INFORMATIONAL: grad-cos vs mean(b1) (NOT gated — MJ-1073; qwen _l2z guard) ──
    print("---- [informational] grad-cos b2 vs mean(b1) (NOT gated, MJ-1073) ----")
    var worst_a = Float64(1.0)
    var worst_b = Float64(1.0)
    var zero_b2 = 0
    var zero_b1 = 0
    var zero_mismatch = 0
    for i in range(n):
        if len(g_b2.d_a[i]) == 0 or len(g_s0.d_a[i]) == 0:
            continue
        var ma = _mean2(g_s0.d_a[i], g_s1.d_a[i])
        var mb = _mean2(g_s0.d_b[i], g_s1.d_b[i])
        # zero-vector guard + DIAGNOSTIC: matching zero slots between arms =
        # tiny-dims artifact; b2-zero-where-b1-nonzero = an unfilled-slot BUG.
        var z2a = _l2z(g_b2.d_a[i]) == 0.0
        var z2b = _l2z(g_b2.d_b[i]) == 0.0
        var z1a = _l2z(ma) == 0.0
        var z1b = _l2z(mb) == 0.0
        if z2a or z2b:
            zero_b2 += 1
        if z1a or z1b:
            zero_b1 += 1
        if z2a != z1a or z2b != z1b:
            zero_mismatch += 1
            print("  [zero-slot MISMATCH] slot", i, " b2_zero(a,b)=", z2a, z2b,
                  " b1mean_zero(a,b)=", z1a, z1b)
        if z2a or z1a or z2b or z1b:
            continue
        var ca = _cos(g_b2.d_a[i], ma)
        var cb = _cos(g_b2.d_b[i], mb)
        if ca < worst_a:
            worst_a = ca
        if cb < worst_b:
            worst_b = cb
    print("  slots=", n, " worst d_a cos=", worst_a, " worst d_b cos=", worst_b,
          " zero_slots(b2)=", zero_b2, " zero_slots(b1mean)=", zero_b1,
          " ZERO-MISMATCHES=", zero_mismatch, ("  <-- BUG if nonzero" if zero_mismatch > 0 else ""))

    print("---- [informational] b2dup grad-cos vs b1(s0) ----")
    var dworst_a = Float64(1.0)
    var dworst_b = Float64(1.0)
    for i in range(n):
        if len(g_dup.d_a[i]) == 0 or len(g_s0.d_a[i]) == 0:
            continue
        if _l2z(g_dup.d_a[i]) == 0.0 or _l2z(g_s0.d_a[i]) == 0.0 or _l2z(g_dup.d_b[i]) == 0.0 or _l2z(g_s0.d_b[i]) == 0.0:
            continue
        var ca = _cos(g_dup.d_a[i], g_s0.d_a[i])
        var cb = _cos(g_dup.d_b[i], g_s0.d_b[i])
        if ca < dworst_a:
            dworst_a = ca
        if cb < dworst_b:
            dworst_b = cb
    print("  worst d_a cos=", dworst_a, " worst d_b cos=", dworst_b)

    if g_b2.nonfinite_lora_grads != 0 or g_dup.nonfinite_lora_grads != 0:
        allok = False
        print("  FAIL: b2 produced nonfinite LoRA grads")

    print("==== VERDICT:", ("PASS" if allok else "FAIL"), "====")
    if not allok:
        raise Error("flux_b2_parity FAILED")
