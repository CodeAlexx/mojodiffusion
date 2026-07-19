# serenitymojo/models/qwenimage/parity/qwen_b2_parity.mojo
#
# TRUE BATCH-2 (row-stacked) qwenimage device-stack parity gate (MJ-1073 design).
#
# Two DISTINCT samples (s0, s1) with different inputs AND different silu(temb)
# embeddings (as a real b2 pair whose sigmas -> temb -> mods differ). The b2
# device stack processes both in ONE call; the b1 device stack is the per-sample
# oracle (run twice). Bars follow the krea2/chroma MJ-1073 re-baseline:
#   BINDING:
#     (b) loss-parity:   |loss_b2 - mean(loss_s0, loss_s1)| / mean  <= 1e-3
#     (c) per-sample fwd: cos(b2.out_s, b1.out_s)                    >= 0.999
#     (a) b2dup:          b2(s0,s0) reproduces b1(s0) loss (rel<=1e-3)
#                         AND per-sample fwd cos(b2dup.out_s, s0.out) >= 0.999
#   INFORMATIONAL (NOT gated — M=2N vs M=N GEMM bf16 tiling makes grad-cos-vs-b1
#   the wrong instrument at depth; reported for tracking):
#     grad-cos:          cos(grad_b2.d_a/d_b[i], mean(grad_s0, grad_s1)[i])
#     b2dup grad-cos:    cos(grad_b2dup[i], grad_s0[i])
#
# CONTRACT under test: b2 sums the two samples' per-adapter d_a/d_b IN-GEMM
# (M=2N); the caller pre-scales each sample's d_out by 0.5, so the sum = the mean
# = the grad-accum=2 gradient. loss_b2 = 0.5*(loss_s0 + loss_s1).
#
# Synthetic setup: random base/block weights -> temp safetensors -> a small
# inline BlockPlan (transformer_blocks.<i>) + TurboPlannedLoader + QwenOffloadBase.
# Each arm gets its OWN freshly-opened loader (fwd then bwd consume it once).
# qwen's device path is math sdpa (flash NOT wired) — no FLASH knob. Small dims.
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run qwen-b2-parity-build && output/bin/qwen_b2_parity

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.train_step import LoraAdapter

from serenitymojo.models.qwenimage.qwenimage_stack import QwenStackBase
from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    QwenOffloadBase, QwenLoraSet, build_qwen_lora_set,
    qwenimage_stack_lora_forward_offload_device,
    qwenimage_stack_lora_backward_offload_device,
    qwenimage_stack_lora_forward_offload_device_b2,
    qwenimage_stack_lora_backward_offload_device_b2,
)
from serenitymojo.offload.plan import BlockPlan, BlockKind, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader


comptime TArc = ArcPointer[Tensor]
comptime CKPT_PATH = "/tmp/qwen_b2_parity.safetensors"
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
comptime NUM_DOUBLE = 2
comptime DBL_SLOTS = 12
comptime EPS = Float32(1.0e-06)
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


def _rand_bf16(n: Int, seed: UInt64, scale: Float32) -> List[BFloat16]:
    var f = _rand(n, seed, scale)
    var out = List[BFloat16]()
    for i in range(n):
        out.append(BFloat16(f[i]))
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


def _arc(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
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


# A slot grad is "zero" if empty (unfilled) or all elements exactly 0.0. cos()
# on such a vector is undefined (zero norm) -> guard it in the informational
# section instead of crashing.
def _is_zero(v: List[Float32]) -> Bool:
    if len(v) == 0:
        return True
    for i in range(len(v)):
        if v[i] != 0.0:
            return False
    return True


# human-readable slot label: block index + stream + per-stream slot name.
def _slotname(i: Int) -> String:
    var blk = i // DBL_SLOTS
    var within = i % DBL_SLOTS
    var stream = String("img") if within < 6 else String("txt")
    var s = within % 6
    var names = List[String]()
    names.append(String("q")); names.append(String("k")); names.append(String("v"))
    names.append(String("out")); names.append(String("ff_up")); names.append(String("ff_down"))
    return String("blk") + String(blk) + String(".") + stream + String(".") + names[s]


# rebuild the LoRA set with NONZERO random A and B (moments zeroed) so the LoRA
# actually contributes and grads are meaningful (make_lora_adapter zero-inits B).
def _nonzero_lora_set(src: QwenLoraSet, seed0: UInt64) -> QwenLoraSet:
    var dbl = List[LoraAdapter]()
    var seed = seed0
    for i in range(len(src.dbl)):
        var rank = src.dbl[i].rank
        var in_f = src.dbl[i].in_f
        var out_f = src.dbl[i].out_f
        var scale = src.dbl[i].scale
        var na = rank * in_f
        var nb = out_f * rank
        var a = _rand(na, seed, Float32(0.30)); seed += 1
        var b = _rand(nb, seed, Float32(0.30)); seed += 1
        dbl.append(LoraAdapter(
            a^, b^, rank, in_f, out_f, scale,
            _zeros(na), _zeros(na), _zeros(nb), _zeros(nb),
        ))
    return QwenLoraSet(dbl^, src.num_double, src.rank)


def _add(
    mut names: List[String], mut tensors: List[TArc],
    name: String, vals: List[Float32], var shape: List[Int], ctx: DeviceContext,
) raises:
    names.append(name)
    tensors.append(_arc(vals, shape^, ctx))


# One qwen DOUBLE block's on-disk tensors (keys read by
# _stream_weights_from_block_offload + _modvecs_from_block).
def _write_double_block(
    mut names: List[String], mut tensors: List[TArc], bp: String, seed: UInt64, ctx: DeviceContext
) raises:
    # img stream
    _add(names, tensors, bp + "attn.to_q.weight", _rand(D * D, seed + 0, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.to_q.bias", _rand(D, seed + 1, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.to_k.weight", _rand(D * D, seed + 2, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.to_k.bias", _rand(D, seed + 3, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.to_v.weight", _rand(D * D, seed + 4, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.to_v.bias", _rand(D, seed + 5, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.to_out.0.weight", _rand(D * D, seed + 6, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.to_out.0.bias", _rand(D, seed + 7, 0.01), [D], ctx)
    _add(names, tensors, bp + "img_mlp.net.0.proj.weight", _rand(FMLP * D, seed + 8, 0.04), [FMLP, D], ctx)
    _add(names, tensors, bp + "img_mlp.net.0.proj.bias", _rand(FMLP, seed + 9, 0.01), [FMLP], ctx)
    _add(names, tensors, bp + "img_mlp.net.2.weight", _rand(D * FMLP, seed + 10, 0.04), [D, FMLP], ctx)
    _add(names, tensors, bp + "img_mlp.net.2.bias", _rand(D, seed + 11, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.norm_q.weight", _ones(Dh), [Dh], ctx)
    _add(names, tensors, bp + "attn.norm_k.weight", _ones(Dh), [Dh], ctx)
    # txt stream
    _add(names, tensors, bp + "attn.add_q_proj.weight", _rand(D * D, seed + 12, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.add_q_proj.bias", _rand(D, seed + 13, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.add_k_proj.weight", _rand(D * D, seed + 14, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.add_k_proj.bias", _rand(D, seed + 15, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.add_v_proj.weight", _rand(D * D, seed + 16, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.add_v_proj.bias", _rand(D, seed + 17, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.to_add_out.weight", _rand(D * D, seed + 18, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.to_add_out.bias", _rand(D, seed + 19, 0.01), [D], ctx)
    _add(names, tensors, bp + "txt_mlp.net.0.proj.weight", _rand(FMLP * D, seed + 20, 0.04), [FMLP, D], ctx)
    _add(names, tensors, bp + "txt_mlp.net.0.proj.bias", _rand(FMLP, seed + 21, 0.01), [FMLP], ctx)
    _add(names, tensors, bp + "txt_mlp.net.2.weight", _rand(D * FMLP, seed + 22, 0.04), [D, FMLP], ctx)
    _add(names, tensors, bp + "txt_mlp.net.2.bias", _rand(D, seed + 23, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.norm_added_q.weight", _ones(Dh), [Dh], ctx)
    _add(names, tensors, bp + "attn.norm_added_k.weight", _ones(Dh), [Dh], ctx)
    # per-block modulation MLP (frozen); shift1,scale1,gate1,shift2,scale2,gate2.
    _add(names, tensors, bp + "img_mod.1.weight", _rand(6 * D * D, seed + 24, 0.04), [6 * D, D], ctx)
    _add(names, tensors, bp + "img_mod.1.bias", _rand(6 * D, seed + 25, 0.02), [6 * D], ctx)
    _add(names, tensors, bp + "txt_mod.1.weight", _rand(6 * D * D, seed + 26, 0.04), [6 * D, D], ctx)
    _add(names, tensors, bp + "txt_mod.1.bias", _rand(6 * D, seed + 27, 0.02), [6 * D], ctx)


def _plan() -> BlockPlan:
    var plan = BlockPlan(String("qwen_image"))
    for i in range(NUM_DOUBLE):
        plan.append(String("transformer_blocks.") + String(i), BlockKind.double_stream())
    return plan^


def _open_loader(ctx: DeviceContext) raises -> TurboPlannedLoader:
    var cfg = OffloadConfig.synchronous_single()
    return TurboPlannedLoader.open(String(CKPT_PATH), _plan(), cfg, ctx)


# QwenOffloadBase: frozen non-block weights + norm_out + timestep MLP (te_lin*
# unused here — the conductors take a pre-activated silu_temb_h directly).
def _base(ctx: DeviceContext) raises -> QwenOffloadBase:
    var stack = QwenStackBase(
        _rand(D * IN_CH, UInt64(100), Float32(0.03)), _rand(D, UInt64(101), Float32(0.01)),
        _rand(D * TXT_CH, UInt64(102), Float32(0.03)), _rand(D, UInt64(103), Float32(0.01)),
        _rand(OUT_CH * D, UInt64(104), Float32(0.03)), _rand(OUT_CH, UInt64(105), Float32(0.01)),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )
    return QwenOffloadBase(
        stack^,
        _rand_bf16(2 * D * D, UInt64(110), Float32(0.03)), _rand_bf16(2 * D, UInt64(111), Float32(0.02)),
        _rand_bf16(D, UInt64(112), Float32(0.01)), _rand_bf16(D, UInt64(113), Float32(0.01)),
        _rand_bf16(D * D, UInt64(114), Float32(0.01)), _rand_bf16(D, UInt64(115), Float32(0.01)),
    )


def _l2z(v: List[Float32]) -> Float64:
    var acc: Float64 = 0.0
    for i in range(len(v)):
        acc += Float64(v[i]) * Float64(v[i])
    return acc


def main() raises:
    var ctx = DeviceContext()
    print("==== qwen_b2_parity (row-stacked TRUE batch-2) ====")
    print("D=", D, " FMLP=", FMLP, " double=", NUM_DOUBLE,
          " N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S, " RANK=", RANK)

    var names = List[String]()
    var tensors = List[TArc]()
    for bi in range(NUM_DOUBLE):
        _write_double_block(
            names, tensors,
            String("transformer_blocks.") + String(bi) + String("."),
            UInt64(1000 + bi * 100), ctx,
        )
    save_safetensors(names, tensors, String(CKPT_PATH), ctx)

    # two DISTINCT samples (different tokens + silu(temb) -> different mods).
    var img0 = _rand(N_IMG * IN_CH, UInt64(800), Float32(0.50))
    var txt0 = _rand(N_TXT * TXT_CH, UInt64(801), Float32(0.50))
    var temb0 = _rand(D, UInt64(802), Float32(0.30))
    var tgt0 = _rand(N_IMG * OUT_CH, UInt64(806), Float32(0.50))

    var img1 = _rand(N_IMG * IN_CH, UInt64(900), Float32(0.50))
    var txt1 = _rand(N_TXT * TXT_CH, UInt64(901), Float32(0.50))
    var temb1 = _rand(D, UInt64(902), Float32(0.34))
    var tgt1 = _rand(N_IMG * OUT_CH, UInt64(906), Float32(0.50))

    var cos = _rand(S * H * (Dh // 2), UInt64(803), Float32(0.50))
    var sin = _rand(S * H * (Dh // 2), UInt64(804), Float32(0.50))

    var norm_out_w = _rand_bf16(2 * D * D, UInt64(110), Float32(0.03))
    var norm_out_b = _rand_bf16(2 * D, UInt64(111), Float32(0.02))

    var lora0 = build_qwen_lora_set(NUM_DOUBLE, D, FMLP, RANK, ALPHA)
    var lora = _nonzero_lora_set(lora0, UInt64(9000))
    var n = NUM_DOUBLE * DBL_SLOTS

    # ── b1 oracle: sample 0 ──
    var base_a = _base(ctx)
    var l_s0 = _open_loader(ctx)
    var f_s0 = qwenimage_stack_lora_forward_offload_device[H, Dh, N_IMG, N_TXT, S](
        img0.copy(), txt0.copy(), temb0.copy(),
        base_a, l_s0, lora, cos.copy(), sin.copy(),
        norm_out_w.copy(), norm_out_b.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var loss0 = _mse(f_s0.out, tgt0)
    var g_s0 = qwenimage_stack_lora_backward_offload_device[H, Dh, N_IMG, N_TXT, S](
        _dloss(f_s0.out, tgt0, Float32(1.0)), img0.copy(), txt0.copy(), temb0.copy(),
        base_a, l_s0, lora, cos.copy(), sin.copy(),
        norm_out_w.copy(), norm_out_b.copy(), f_s0,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    # ── b1 oracle: sample 1 ──
    var base_b = _base(ctx)
    var l_s1 = _open_loader(ctx)
    var f_s1 = qwenimage_stack_lora_forward_offload_device[H, Dh, N_IMG, N_TXT, S](
        img1.copy(), txt1.copy(), temb1.copy(),
        base_b, l_s1, lora, cos.copy(), sin.copy(),
        norm_out_w.copy(), norm_out_b.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var loss1 = _mse(f_s1.out, tgt1)
    var g_s1 = qwenimage_stack_lora_backward_offload_device[H, Dh, N_IMG, N_TXT, S](
        _dloss(f_s1.out, tgt1, Float32(1.0)), img1.copy(), txt1.copy(), temb1.copy(),
        base_b, l_s1, lora, cos.copy(), sin.copy(),
        norm_out_w.copy(), norm_out_b.copy(), f_s1,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    # ── b2: (s0, s1) in one call ──
    var base_c = _base(ctx)
    var l_b2 = _open_loader(ctx)
    var f_b2 = qwenimage_stack_lora_forward_offload_device_b2[H, Dh, N_IMG, N_TXT, S](
        img0.copy(), txt0.copy(), temb0.copy(),
        img1.copy(), txt1.copy(), temb1.copy(),
        base_c, l_b2, lora, cos.copy(), sin.copy(),
        norm_out_w.copy(), norm_out_b.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var loss_b2 = Float64(0.5) * (_mse(f_b2.out0, tgt0) + _mse(f_b2.out1, tgt1))
    var g_b2 = qwenimage_stack_lora_backward_offload_device_b2[H, Dh, N_IMG, N_TXT, S](
        _dloss(f_b2.out0, tgt0, Float32(0.5)), _dloss(f_b2.out1, tgt1, Float32(0.5)),
        temb0.copy(), temb1.copy(),
        base_c, l_b2, lora, cos.copy(), sin.copy(),
        norm_out_w.copy(), norm_out_b.copy(), f_b2,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    # ── b2dup: (s0, s0) ──
    var base_d = _base(ctx)
    var l_dup = _open_loader(ctx)
    var f_dup = qwenimage_stack_lora_forward_offload_device_b2[H, Dh, N_IMG, N_TXT, S](
        img0.copy(), txt0.copy(), temb0.copy(),
        img0.copy(), txt0.copy(), temb0.copy(),
        base_d, l_dup, lora, cos.copy(), sin.copy(),
        norm_out_w.copy(), norm_out_b.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var loss_dup = Float64(0.5) * (_mse(f_dup.out0, tgt0) + _mse(f_dup.out1, tgt0))
    var g_dup = qwenimage_stack_lora_backward_offload_device_b2[H, Dh, N_IMG, N_TXT, S](
        _dloss(f_dup.out0, tgt0, Float32(0.5)), _dloss(f_dup.out1, tgt0, Float32(0.5)),
        temb0.copy(), temb0.copy(),
        base_d, l_dup, lora, cos.copy(), sin.copy(),
        norm_out_w.copy(), norm_out_b.copy(), f_dup,
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
        raise Error("qwen_b2_parity FAILED")
