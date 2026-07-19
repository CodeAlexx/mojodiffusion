# serenitymojo/models/krea2/parity/krea2_stack_real_smoke.mojo
#
# REAL-DEPTH FINITE SMOKE for the Krea-2-Raw single-stream STACK LoRA composition
# (models/krea2/krea2_stack.mojo): the FULL 28-block stack forward + LoRA stack
# backward on the REAL raw.safetensors weights, with per-block recompute. torch
# can't do a 28-block real-dim parity cheaply, so reduced-depth parity
# (krea2_stack_parity.mojo, cos≥0.999) + THIS real-depth finite smoke compose the
# proof: this confirms the composition RUNS at real depth without OOM, with finite
# loss + nonzero grad_norm + nonzero LoRA dA/dB.
#
# Weights: the real DiT is bf16 matmul + F32 norms/mod on disk. This smoke loads
# the matmul weights bf16-RESIDENT (the realistic training memory profile; the
# block's linear/rms_norm run F32 activations against bf16 weights exactly like the
# inference krea2_forward) and the norm/mod params F32. LoRA A/B are F32 with NONZERO
# B (so dA/dB are non-degenerate). Synthetic single-stream input (the data path is
# Phase 3); we only exercise the stack fwd+bwd numerics.
#
# Sequence L=256 (the no-pad design from the parity gate). VRAM: 28 blocks of bf16
# matmul weights resident (~all-blocks-resident; the orchestrator runs this and
# notes VRAM — if it OOMs at real depth, that motivates the Phase-4 streaming/turbo
# loader, not a Phase-2 blocker).
#
# Run (ORCHESTRATOR runs this — long + heavy; not backgrounded by the builder):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_stack_real_smoke.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2LoraGrad,
)
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackWeights, Krea2StackLora, Krea2StackForward,
    Krea2StackLoraGrads, KREA2_SLOTS_PER_BLOCK,
    krea2_stack_lora_forward, krea2_stack_lora_backward,
)
from serenitymojo.models.dit.krea2_dit import build_krea2_rope

comptime TArc = ArcPointer[Tensor]
comptime RAW = "/home/alex/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494/raw.safetensors"

comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM       # 6144
comptime MLPDIM = 16384
comptime OUT_CH = 64
comptime NBLOCKS = 28                     # REAL depth
comptime L = 256                          # no-pad sequence
comptime TXTLEN = 46
comptime IMGLEN = 210
comptime RANK = 16
comptime EPS = Float32(1e-5)
comptime LSCALE = Float32(1.0)            # alpha/rank = 16/16


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


# bf16-resident matmul weight (real disk dtype; cast bf16 if F16/other).
def _wb(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_view_as_bf16(st.tensor_view(key), ctx))


# F32 norm/mod param (real disk dtype is F32; upcast-safe).
def _wf(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_view_as_f32(st.tensor_view(key), ctx))


# deterministic non-degenerate host fill (libm-free bounded irrational-ish pattern
# so the binary links with the canonical flags — no -lm). A fractional LCG-style
# wrap keeps values in [-c, c], non-periodic at the H·Dh strides (never modular,
# which would alias to false zero-grad). a/b seed distinct streams per tensor.
def _fill(n: Int, a: Float32, b: Float32, c: Float32) -> List[Float32]:
    var o = List[Float32]()
    var phase = Float64(b)
    var inc = Float64(a) + Float64(0.6180339887)   # golden-ratio step (irrational)
    for _i in range(n):
        phase += inc
        var frac = phase - Float64(Int(phase))     # in [0,1)
        o.append(Float32((frac * 2.0 - 1.0)) * c)  # → [-c, c]
    return o^


def _rand_dev(n: Int, var shape: List[Int], a: Float32, b: Float32, c: Float32, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(_fill(n, a, b, c), shape^, STDtype.F32, ctx)


# LoRA adapter (A small non-degenerate, B NONZERO so dA/dB are non-degenerate).
def _adapter(in_f: Int, out_f: Int, seed: Float32, ctx: DeviceContext) raises -> Optional[LoraAdapterDevice]:
    var a = _rand_dev(RANK * in_f, [RANK, in_f], 0.013, seed, 0.02, ctx)
    var b = _rand_dev(out_f * RANK, [out_f, RANK], 0.017, seed + 0.3, 0.02, ctx)
    return Optional[LoraAdapterDevice](LoraAdapterDevice(TArc(a^), TArc(b^), RANK, in_f, out_f, LSCALE))


def _block_weights(st: ShardedSafeTensors, bi: Int, ctx: DeviceContext) raises -> Krea2BlockWeights:
    var p = "blocks." + String(bi) + "."
    return Krea2BlockWeights(
        _wb(st, p + "attn.wq.weight", ctx),
        _wb(st, p + "attn.wk.weight", ctx),
        _wb(st, p + "attn.wv.weight", ctx),
        _wb(st, p + "attn.gate.weight", ctx),
        _wb(st, p + "attn.wo.weight", ctx),
        _wb(st, p + "mlp.gate.weight", ctx),
        _wb(st, p + "mlp.up.weight", ctx),
        _wb(st, p + "mlp.down.weight", ctx),
        _wf(st, p + "attn.qknorm.qnorm.scale", ctx),
        _wf(st, p + "attn.qknorm.knorm.scale", ctx),
        _wf(st, p + "prenorm.scale", ctx),
        _wf(st, p + "postnorm.scale", ctx),
        _wf(st, p + "mod.lin", ctx),
    )


def _block_lora(bi: Int, ctx: DeviceContext) raises -> Krea2BlockLora:
    var s = Float32(bi) * 0.11
    return Krea2BlockLora(
        _adapter(FEATURES, HEADS * HEADDIM, s + 0.01, ctx),
        _adapter(FEATURES, KVHEADS * HEADDIM, s + 0.02, ctx),
        _adapter(FEATURES, KVHEADS * HEADDIM, s + 0.03, ctx),
        _adapter(FEATURES, FEATURES, s + 0.04, ctx),
        _adapter(FEATURES, FEATURES, s + 0.05, ctx),
        _adapter(FEATURES, MLPDIM, s + 0.06, ctx),
        _adapter(FEATURES, MLPDIM, s + 0.07, ctx),
        _adapter(MLPDIM, FEATURES, s + 0.08, ctx),
    )


def _l2(v: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(v)):
        s += Float64(v[i]) * Float64(v[i])
    return s


def _finite(v: List[Float32]) -> Bool:
    for i in range(len(v)):
        var x = v[i]
        if x != x:        # NaN
            return False
        if x > Float32(3.0e38) or x < Float32(-3.0e38):
            return False
    return True


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(RAW)
    print("==== krea2_stack_real_smoke (REAL 28-block stack fwd+bwd) ====")
    print("NBLOCKS=", NBLOCKS, " L=", L, " FEATURES=", FEATURES, " MLPDIM=", MLPDIM, " RANK=", RANK)

    # ── build all 28 blocks (bf16 matmul resident + F32 norms) + LoRA ────────────
    print("loading 28 blocks ...")
    var blocks_w = List[Krea2BlockWeights]()
    var blocks_l = List[Krea2BlockLora]()
    for bi in range(NBLOCKS):
        blocks_w.append(_block_weights(st, bi, ctx))
        blocks_l.append(_block_lora(bi, ctx))
    var w = Krea2StackWeights(
        blocks_w^,
        _wf(st, "last.norm.scale", ctx),
        _wf(st, "last.modulation.lin", ctx),
        _wb(st, "last.linear.weight", ctx),
        _wb(st, "last.linear.bias", ctx),
    )
    var lora = Krea2StackLora(blocks_l^)
    print("weights loaded.")

    # ── synthetic single-stream inputs (data path is Phase 3) ────────────────────
    var combined = TArc(_rand_dev(L * FEATURES, [1, L, FEATURES], 0.0007, 0.05, 0.5, ctx))
    var blk_vec = _rand_dev(6 * FEATURES, [1, 6 * FEATURES], 0.0003, 0.11, 0.2, ctx)
    var tmlp_out = _rand_dev(FEATURES, [1, 1, FEATURES], 0.0005, 0.2, 0.3, ctx)

    # pos: txt rows zeros, img rows a grid (GH=14, GW=15 → 210 img tokens).
    var pos_host = List[Float32]()
    for _i in range(TXTLEN):
        pos_host.append(0.0); pos_host.append(0.0); pos_host.append(0.0)
    var gh = 14
    var gw = 15
    for r in range(gh):
        for c in range(gw):
            pos_host.append(0.0)
            pos_host.append(Float32(r))
            pos_host.append(Float32(c))
    var pos_flat = Tensor.from_host(pos_host^, _shape1(L * 3), STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, Float32(1.0e3), ctx, STDtype.F32)

    # ── FORWARD ──────────────────────────────────────────────────────────────────
    print("forward (28 blocks) ...")
    var fwd = krea2_stack_lora_forward[L, HEADS, KVHEADS, HEADDIM](
        combined, blk_vec, tmlp_out, w, lora, rope[0], rope[1], EPS, TXTLEN, IMGLEN, ctx,
    )
    var vel_h = fwd.velocity[].to_host(ctx)
    var vel_ok = _finite(vel_h)
    var vel_l2 = _l2(vel_h)
    print("  velocity numel=", len(vel_h), " finite=", vel_ok, " L2^2=", vel_l2)

    # ── loss + BACKWARD ──────────────────────────────────────────────────────────
    # d_velocity = velocity (an MSE-to-zero style grad: loss=0.5*||v||^2 → dL/dv=v).
    var d_velocity = Tensor.from_host(vel_h, [1, IMGLEN, OUT_CH], STDtype.F32, ctx)
    var loss = Float64(0.5) * vel_l2
    print("backward (28 blocks, per-block recompute) ...")
    var grads = krea2_stack_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        d_velocity, blk_vec, tmlp_out, w, lora, fwd, rope[0], rope[1], EPS, ctx,
    )

    # ── grad-norm + non-degeneracy checks ────────────────────────────────────────
    var gn2 = Float64(0.0)
    var n_nonzero = 0
    var all_finite = True
    for bi in range(NBLOCKS):
        var base = bi * KREA2_SLOTS_PER_BLOCK
        for s in range(KREA2_SLOTS_PER_BLOCK):
            var gpair = grads.grads[base + s].copy()
            if not gpair.d_a or not gpair.d_b:
                print("  MISSING grad at block", bi, "slot", s, " — FAIL")
                all_finite = False
                continue
            if not _finite(gpair.d_a.value()) or not _finite(gpair.d_b.value()):
                all_finite = False
            var la = _l2(gpair.d_a.value())
            var lb = _l2(gpair.d_b.value())
            gn2 += la + lb
            if la > Float64(0.0):
                n_nonzero += 1
            if lb > Float64(0.0):
                n_nonzero += 1
    var d_comb_h = grads.d_combined[].to_host(ctx)
    var d_comb_ok = _finite(d_comb_h)
    var d_comb_l2 = _l2(d_comb_h)

    var grad_norm = sqrt(gn2)
    print("")
    print("---- SMOKE RESULTS ----")
    print("  loss (0.5||v||^2)      =", loss, "  finite=", loss == loss)
    print("  velocity finite        =", vel_ok)
    print("  LoRA grad_norm         =", grad_norm)
    print("  nonzero dA/dB tensors  =", n_nonzero, " / ", NBLOCKS * KREA2_SLOTS_PER_BLOCK * 2)
    print("  all LoRA grads finite  =", all_finite)
    print("  d_combined finite      =", d_comb_ok, "  L2^2=", d_comb_l2)
    print("")

    var passed = (
        vel_ok and all_finite and d_comb_ok
        and loss == loss
        and grad_norm > Float64(0.0)
        and n_nonzero == NBLOCKS * KREA2_SLOTS_PER_BLOCK * 2
    )
    if passed:
        print("VERDICT: PASS — real 28-block stack fwd+bwd runs; loss finite, grad_norm nonzero, all LoRA grads nonzero+finite")
    else:
        print("VERDICT: FAIL — see the checks above")
        raise Error("krea2_stack_real_smoke FAILED")
