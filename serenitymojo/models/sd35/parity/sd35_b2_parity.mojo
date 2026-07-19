# serenitymojo/models/sd35/parity/sd35_b2_parity.mojo
#
# TRUE BATCH-2 (row-stacked) sd35-Large device-stack parity gate (MJ-1073 design).
#
# Two DISTINCT samples (s0, s1) with different inputs AND different sigma -> own
# conditioning c -> own per-block/final adaLN mods (a real b2 pair). The b2 device
# stack processes both in ONE call; the b1 device stack is the per-sample oracle
# (run twice). Bars follow the krea2/chroma/qwen MJ-1073 re-baseline:
#   BINDING:
#     (b) loss-parity:   |loss_b2 - mean(loss_s0, loss_s1)| / mean  <= 1e-3
#     (c) per-sample fwd: cos(b2.out_s, b1.out_s)                    >= 0.999
#     (a) b2dup:          b2(s0,s0) reproduces b1(s0) loss (rel<=1e-3)
#                         AND per-sample fwd cos(b2dup.out_s, s0.out) >= 0.999
#   INFORMATIONAL (NOT gated — M=2N vs M=N GEMM bf16 tiling makes grad-cos-vs-b1
#   the wrong instrument; reported for tracking, WITH the zero-slot diagnostic).
#
# CONTRACT under test: b2 sums the two samples' per-adapter d_a/d_b IN-GEMM
# (M=2N); the caller pre-scales each sample's d_out by 0.5 so the sum = the mean =
# the grad-accum=2 gradient. loss_b2 = 0.5*(loss_s0 + loss_s1).
#
# ZERO-SLOT DIAGNOSTIC: the final block is context_pre_only — its ctx stream has
# ONLY qkv (no proj/mlp), so LoRA slots ctx_proj/ctx_fc1/ctx_fc2 of that block are
# NEVER filled (structural zeros) in BOTH arms. Matching zeros = a tiny-dims/
# structure artifact; a b2-zero-where-b1-nonzero mismatch would be a real BUG.
#
# Synthetic setup: random base + 1 joint block + 1 context_pre_only block ->
# temp safetensors + inline BlockPlan (joint_blocks.<i>) + TurboPlannedLoader.
# Each arm gets its OWN freshly-opened loader (fwd then bwd consume it once).
# Device blocks default FLASH=True (Dh=64 is cuDNN-valid); small other dims.
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run sd35-b2-parity-build && output/bin/sd35_b2_parity

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.train_step import LoraAdapter

from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35StackBase, SD35LoraSet, build_sd35_lora_set, total_adapters,
    sd35_stack_lora_forward_offload_device, sd35_stack_lora_backward_offload_device,
    sd35_stack_lora_forward_offload_device_b2, sd35_stack_lora_backward_offload_device_b2,
    SLOTS_PER_BLOCK,
)
from serenitymojo.offload.plan import BlockPlan, BlockKind, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader


comptime TArc = ArcPointer[Tensor]
comptime CKPT_PATH = "/tmp/sd35_b2_parity.safetensors"
comptime H = 2
comptime Dh = 64
comptime D = H * Dh             # 128
comptime N_CTX = 2
comptime N_IMG = 4              # perfect square (pos-embed crop); S = 6
comptime S = N_CTX + N_IMG      # 6 (ctx FIRST then img)
comptime MLP = 32
comptime IN_CH = 8
comptime CTX_CH = 12
comptime OUT_CH = 8
comptime TIMESTEP_DIM = 16
comptime POOLED_DIM = 16
comptime POS_MAX = 2            # pos_embed [POS_MAX*POS_MAX, D]; crops to N_IMG=4
comptime NUM_JOINT = 2         # block 0 joint, block 1 context_pre_only
comptime EPS = Float32(1.0e-06)
comptime QK_EPS = Float32(1.0e-06)
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


# near-1 (rms/ln weights).
def _near1(n: Int, seed: UInt64) -> List[Float32]:
    var f = _rand(n, seed, Float32(0.1))
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(1.0) + f[i])
    return out^


def _arc_f32(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals.copy(), shape^, STDtype.F32, ctx))


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: length mismatch")
    var dot = Float64(0.0); var na = Float64(0.0); var nb = Float64(0.0)
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


# d_loss = scale * (2/N) * (pred - target)  (scale=1.0 for b1; 0.5 for b2).
def _dloss(pred: List[Float32], target: List[Float32], scale: Float32) raises -> List[Float32]:
    var inv_n = Float32(2.0) / Float32(len(pred))
    var out = List[Float32]()
    for i in range(len(pred)):
        out.append(scale * inv_n * (pred[i] - target[i]))
    return out^


def _mean2(a: List[Float32], b: List[Float32]) raises -> List[Float32]:
    if len(a) != len(b):
        raise Error("mean2: length mismatch")
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(Float32(0.5) * (a[i] + b[i]))
    return out^


def _l2z(v: List[Float32]) -> Float64:
    var acc: Float64 = 0.0
    for i in range(len(v)):
        acc += Float64(v[i]) * Float64(v[i])
    return acc


# slot label: block index + stream + per-stream slot name.
def _slotname(i: Int) -> String:
    var blk = i // SLOTS_PER_BLOCK
    var within = i % SLOTS_PER_BLOCK
    var stream = String("ctx") if within < 4 else String("x")
    var s = within % 4
    var names = List[String]()
    names.append(String("qkv")); names.append(String("proj"))
    names.append(String("fc1")); names.append(String("fc2"))
    return String("blk") + String(blk) + String(".") + stream + String(".") + names[s]


# rebuild the LoRA set with NONZERO random A and B (moments zeroed) so the LoRA
# actually contributes (build_sd35_lora_set zero-inits B).
def _nonzero_lora_set(src: SD35LoraSet, seed0: UInt64) -> SD35LoraSet:
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
    return SD35LoraSet(ad^, src.depth, src.rank)


def _add(
    mut names: List[String], mut tensors: List[TArc],
    name: String, vals: List[Float32], var shape: List[Int], ctx: DeviceContext,
) raises:
    names.append(name)
    tensors.append(TArc(Tensor.from_host(vals.copy(), shape^, STDtype.BF16, ctx)))


# one standard stream's on-disk tensors (context_block or x_block).
def _write_stream(
    mut names: List[String], mut tensors: List[TArc], bp: String, seed: UInt64, ctx: DeviceContext
) raises:
    _add(names, tensors, bp + "attn.qkv.weight", _rand(3 * D * D, seed + 0, 0.04), [3 * D, D], ctx)
    _add(names, tensors, bp + "attn.qkv.bias", _rand(3 * D, seed + 1, 0.02), [3 * D], ctx)
    _add(names, tensors, bp + "attn.proj.weight", _rand(D * D, seed + 2, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.proj.bias", _rand(D, seed + 3, 0.02), [D], ctx)
    _add(names, tensors, bp + "mlp.fc1.weight", _rand(MLP * D, seed + 4, 0.04), [MLP, D], ctx)
    _add(names, tensors, bp + "mlp.fc1.bias", _rand(MLP, seed + 5, 0.02), [MLP], ctx)
    _add(names, tensors, bp + "mlp.fc2.weight", _rand(D * MLP, seed + 6, 0.04), [D, MLP], ctx)
    _add(names, tensors, bp + "mlp.fc2.bias", _rand(D, seed + 7, 0.02), [D], ctx)
    _add(names, tensors, bp + "attn.ln_q.weight", _near1(Dh, seed + 8), [Dh], ctx)
    _add(names, tensors, bp + "attn.ln_k.weight", _near1(Dh, seed + 9), [Dh], ctx)
    _add(names, tensors, bp + "adaLN_modulation.1.weight", _rand(6 * D * D, seed + 10, 0.04), [6 * D, D], ctx)
    _add(names, tensors, bp + "adaLN_modulation.1.bias", _rand(6 * D, seed + 11, 0.02), [6 * D], ctx)


# context_pre_only ctx stream: qkv only + continuous adaLN [2D].
def _write_ctx_preonly(
    mut names: List[String], mut tensors: List[TArc], bp: String, seed: UInt64, ctx: DeviceContext
) raises:
    _add(names, tensors, bp + "attn.qkv.weight", _rand(3 * D * D, seed + 0, 0.04), [3 * D, D], ctx)
    _add(names, tensors, bp + "attn.qkv.bias", _rand(3 * D, seed + 1, 0.02), [3 * D], ctx)
    _add(names, tensors, bp + "attn.ln_q.weight", _near1(Dh, seed + 2), [Dh], ctx)
    _add(names, tensors, bp + "attn.ln_k.weight", _near1(Dh, seed + 3), [Dh], ctx)
    _add(names, tensors, bp + "adaLN_modulation.1.weight", _rand(2 * D * D, seed + 4, 0.04), [2 * D, D], ctx)
    _add(names, tensors, bp + "adaLN_modulation.1.bias", _rand(2 * D, seed + 5, 0.02), [2 * D], ctx)


def _plan() -> BlockPlan:
    var plan = BlockPlan(String("sd35"))
    for i in range(NUM_JOINT):
        plan.append(String("joint_blocks.") + String(i), BlockKind.double_stream())
    return plan^


def _open_loader(ctx: DeviceContext) raises -> TurboPlannedLoader:
    var cfg = OffloadConfig.synchronous_single()
    return TurboPlannedLoader.open(String(CKPT_PATH), _plan(), cfg, ctx)


def _base(ctx: DeviceContext) raises -> SD35StackBase:
    return SD35StackBase(
        _arc_f32(_rand(D * IN_CH, UInt64(100), 0.05), [D, IN_CH], ctx),
        _arc_f32(_rand(D, UInt64(101), 0.02), [D], ctx),
        _arc_f32(_rand(D * CTX_CH, UInt64(102), 0.05), [D, CTX_CH], ctx),
        _arc_f32(_rand(D, UInt64(103), 0.02), [D], ctx),
        _arc_f32(_rand(D * TIMESTEP_DIM, UInt64(104), 0.05), [D, TIMESTEP_DIM], ctx),
        _arc_f32(_rand(D, UInt64(105), 0.02), [D], ctx),
        _arc_f32(_rand(D * D, UInt64(106), 0.05), [D, D], ctx),
        _arc_f32(_rand(D, UInt64(107), 0.02), [D], ctx),
        _arc_f32(_rand(D * POOLED_DIM, UInt64(108), 0.05), [D, POOLED_DIM], ctx),
        _arc_f32(_rand(D, UInt64(109), 0.02), [D], ctx),
        _arc_f32(_rand(D * D, UInt64(110), 0.05), [D, D], ctx),
        _arc_f32(_rand(D, UInt64(111), 0.02), [D], ctx),
        _arc_f32(_rand(2 * D * D, UInt64(112), 0.05), [2 * D, D], ctx),
        _arc_f32(_rand(2 * D, UInt64(113), 0.02), [2 * D], ctx),
        _arc_f32(_rand(OUT_CH * D, UInt64(114), 0.05), [OUT_CH, D], ctx),
        _arc_f32(_rand(OUT_CH, UInt64(115), 0.02), [OUT_CH], ctx),
        _arc_f32(_rand(POS_MAX * POS_MAX * D, UInt64(116), 0.02), [POS_MAX * POS_MAX * D], ctx),
    )


def main() raises:
    var ctx = DeviceContext()
    print("==== sd35_b2_parity (row-stacked TRUE batch-2, MJ-1073) ====")
    print("D=", D, " MLP=", MLP, " joint=", NUM_JOINT, " (last preonly)",
          " N_CTX=", N_CTX, " N_IMG=", N_IMG, " S=", S, " RANK=", RANK)

    # ── synthetic block safetensors: block 0 joint, block 1 context_pre_only ──
    var names = List[String]()
    var tensors = List[TArc]()
    # block 0 (joint)
    _write_stream(names, tensors, String("joint_blocks.0.context_block."), UInt64(1000), ctx)
    _write_stream(names, tensors, String("joint_blocks.0.x_block."), UInt64(1100), ctx)
    # block 1 (context_pre_only)
    _write_ctx_preonly(names, tensors, String("joint_blocks.1.context_block."), UInt64(1200), ctx)
    _write_stream(names, tensors, String("joint_blocks.1.x_block."), UInt64(1300), ctx)
    save_safetensors(names, tensors, String(CKPT_PATH), ctx)

    # ── two DISTINCT samples (different tokens + sigma -> different mods) ──
    var noisy0 = _rand(N_IMG * IN_CH, UInt64(800), 0.50)
    var text0 = _rand(N_CTX * CTX_CH, UInt64(801), 0.50)
    var pooled0 = _rand(POOLED_DIM, UInt64(802), 0.50)
    var sigma0 = Float32(0.30)
    var tgt0 = _rand(N_IMG * OUT_CH, UInt64(806), 0.50)

    var noisy1 = _rand(N_IMG * IN_CH, UInt64(900), 0.50)
    var text1 = _rand(N_CTX * CTX_CH, UInt64(901), 0.50)
    var pooled1 = _rand(POOLED_DIM, UInt64(902), 0.50)
    var sigma1 = Float32(0.62)
    var tgt1 = _rand(N_IMG * OUT_CH, UInt64(906), 0.50)

    var lora0 = build_sd35_lora_set(NUM_JOINT, D, MLP, RANK, ALPHA)
    var lora = _nonzero_lora_set(lora0, UInt64(7000))
    var n = total_adapters(lora)

    # ── b1 oracle: sample 0 ──
    var l_s0 = _open_loader(ctx)
    var f_s0 = sd35_stack_lora_forward_offload_device[H, Dh, N_IMG, N_CTX, S](
        noisy0.copy(), text0.copy(), pooled0.copy(), sigma0,
        _base(ctx), l_s0, lora, D, MLP, IN_CH, CTX_CH, OUT_CH,
        TIMESTEP_DIM, POOLED_DIM, EPS, QK_EPS, ctx, 0, True,
    )
    var loss0 = _mse(f_s0.out, tgt0)
    var g_s0 = sd35_stack_lora_backward_offload_device[H, Dh, N_IMG, N_CTX, S](
        _dloss(f_s0.out, tgt0, Float32(1.0)), noisy0.copy(), text0.copy(),
        _base(ctx), l_s0, lora, f_s0, D, MLP, IN_CH, CTX_CH, OUT_CH,
        TIMESTEP_DIM, POOLED_DIM, EPS, QK_EPS, ctx, 0, True,
    )

    # ── b1 oracle: sample 1 ──
    var l_s1 = _open_loader(ctx)
    var f_s1 = sd35_stack_lora_forward_offload_device[H, Dh, N_IMG, N_CTX, S](
        noisy1.copy(), text1.copy(), pooled1.copy(), sigma1,
        _base(ctx), l_s1, lora, D, MLP, IN_CH, CTX_CH, OUT_CH,
        TIMESTEP_DIM, POOLED_DIM, EPS, QK_EPS, ctx, 0, True,
    )
    var loss1 = _mse(f_s1.out, tgt1)
    var g_s1 = sd35_stack_lora_backward_offload_device[H, Dh, N_IMG, N_CTX, S](
        _dloss(f_s1.out, tgt1, Float32(1.0)), noisy1.copy(), text1.copy(),
        _base(ctx), l_s1, lora, f_s1, D, MLP, IN_CH, CTX_CH, OUT_CH,
        TIMESTEP_DIM, POOLED_DIM, EPS, QK_EPS, ctx, 0, True,
    )

    # ── b2: (s0, s1) in one call ──
    var l_b2 = _open_loader(ctx)
    var f_b2 = sd35_stack_lora_forward_offload_device_b2[H, Dh, N_IMG, N_CTX, S](
        noisy0.copy(), text0.copy(), pooled0.copy(), sigma0,
        noisy1.copy(), text1.copy(), pooled1.copy(), sigma1,
        _base(ctx), l_b2, lora, D, MLP, IN_CH, CTX_CH, OUT_CH,
        TIMESTEP_DIM, POOLED_DIM, EPS, QK_EPS, ctx, 0, True,
    )
    var loss_b2 = Float64(0.5) * (_mse(f_b2.out0, tgt0) + _mse(f_b2.out1, tgt1))
    var g_b2 = sd35_stack_lora_backward_offload_device_b2[H, Dh, N_IMG, N_CTX, S](
        _dloss(f_b2.out0, tgt0, Float32(0.5)), _dloss(f_b2.out1, tgt1, Float32(0.5)),
        noisy0.copy(), text0.copy(), noisy1.copy(), text1.copy(),
        _base(ctx), l_b2, lora, f_b2, D, MLP, IN_CH, CTX_CH, OUT_CH,
        TIMESTEP_DIM, POOLED_DIM, EPS, QK_EPS, ctx, 0, True,
    )

    # ── b2dup: (s0, s0) ──
    var l_dup = _open_loader(ctx)
    var f_dup = sd35_stack_lora_forward_offload_device_b2[H, Dh, N_IMG, N_CTX, S](
        noisy0.copy(), text0.copy(), pooled0.copy(), sigma0,
        noisy0.copy(), text0.copy(), pooled0.copy(), sigma0,
        _base(ctx), l_dup, lora, D, MLP, IN_CH, CTX_CH, OUT_CH,
        TIMESTEP_DIM, POOLED_DIM, EPS, QK_EPS, ctx, 0, True,
    )
    var loss_dup = Float64(0.5) * (_mse(f_dup.out0, tgt0) + _mse(f_dup.out1, tgt0))
    var g_dup = sd35_stack_lora_backward_offload_device_b2[H, Dh, N_IMG, N_CTX, S](
        _dloss(f_dup.out0, tgt0, Float32(0.5)), _dloss(f_dup.out1, tgt0, Float32(0.5)),
        noisy0.copy(), text0.copy(), noisy0.copy(), text0.copy(),
        _base(ctx), l_dup, lora, f_dup, D, MLP, IN_CH, CTX_CH, OUT_CH,
        TIMESTEP_DIM, POOLED_DIM, EPS, QK_EPS, ctx, 0, True,
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
    print("  b2 nonfinite=", g_b2.nonfinite)

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

    # ── INFORMATIONAL: grad-cos vs mean(b1) (NOT gated — MJ-1073) ──
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
            print("  [zero-slot MISMATCH] ", _slotname(i), " b2_zero(a,b)=", z2a, z2b,
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

    if g_b2.nonfinite != 0 or g_dup.nonfinite != 0:
        allok = False
        print("  FAIL: b2 produced nonfinite LoRA grads")

    print("==== VERDICT:", ("PASS" if allok else "FAIL"), "====")
    if not allok:
        raise Error("sd35_b2_parity FAILED")
