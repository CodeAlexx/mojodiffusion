# serenitymojo/models/krea2/parity/krea2_b2_block_gate.mojo
#
# BLOCK-LEVEL TRUE batch-2 isolation gate. Runs the b1 single-stream block fwd+bwd
# on two samples SEPARATELY, and the b2 (row-stacked) block fwd+bwd once, and checks
# they agree PER SAMPLE (forward out, backward d_x) and that b2's summed LoRA grads
# equal grads0+grads1. Small dims → runs in seconds (no cache, no streaming). This
# isolates the block backward from the stack/final-layer/conditioning.
#
# Build + run:
#   MEM_MAX=20G bash scripts/mem_safe.sh mojo build --optimization-level 2 --num-threads 1 \
#     -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/krea2/parity/krea2_b2_block_gate.mojo -o output/bin/krea2_b2_block_gate
#   output/bin/krea2_b2_block_gate

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt, cos as mcos, sin as msin
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import concat, slice, mul_scalar
from serenitymojo.models.dit.krea2_dit import _tile_rope_table, krea2_rmsnorm
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackWeights, Krea2StackForward, Krea2StackForwardB2,
    krea2_final_layer_backward, krea2_final_layer_backward_b2,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.ops.attention_flash import sdpa_flash_train_fwd_padmask_bf16
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2LoraGrad, Krea2BlockGrads, Krea2BlockSaved,
    krea2_single_stream_block_lora, krea2_single_stream_block_lora_backward,
    krea2_single_stream_block_lora_b2, krea2_single_stream_block_lora_backward_b2,
)

comptime TArc = ArcPointer[Tensor]

comptime HEADS = 8
comptime KVHEADS = 2
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM      # 1024
comptime HALF = HEADDIM // 2             # 64
comptime MLPDIM = 256
comptime EPS = Float32(1e-5)
comptime RANK = 8
comptime LSCALE = Float32(2.0)
comptime L = 256                         # per-sample length (128-aligned)
comptime RL0 = 200                       # sample0 valid prefix (< L → flash)
comptime RL1 = 176                       # sample1 valid prefix


def _rand(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return randn(shape^, seed, STDtype.BF16, ctx)


# norm-preserving matmul weight (~ N(0, 1/fan_in)) so the block doesn't explode over
# the chain (real krea2 weights are trained/normalized); makes the chain realistic.
def _randw(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    var fan_in = shape[len(shape) - 1]
    return mul_scalar(randn(shape^, seed, STDtype.BF16, ctx), Float32(1.0) / sqrt(Float32(fan_in)), ctx)


# small norm-scale (~0.02) so weight=scale+1 ≈ 1 (real krea2 rms scales are ~0).
def _rands(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return mul_scalar(randn(shape^, seed, STDtype.BF16, ctx), Float32(0.02), ctx)


def _s1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^
def _s4b(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def _adapter(in_f: Int, out_f: Int, seed: UInt64, ctx: DeviceContext) raises -> Optional[LoraAdapterDevice]:
    var a = TArc(_randw(_s2(RANK, in_f), seed, ctx))
    var b = TArc(_randw(_s2(out_f, RANK), seed + 1, ctx))   # small NONZERO B → dA + dB both live
    return Optional[LoraAdapterDevice](LoraAdapterDevice(a^, b^, RANK, in_f, out_f, LSCALE))


def _make_weights(ctx: DeviceContext) raises -> Krea2BlockWeights:
    return Krea2BlockWeights(
        TArc(_randw(_s2(HEADS * HEADDIM, FEATURES), 1, ctx)),
        TArc(_randw(_s2(KVHEADS * HEADDIM, FEATURES), 2, ctx)),
        TArc(_randw(_s2(KVHEADS * HEADDIM, FEATURES), 3, ctx)),
        TArc(_randw(_s2(FEATURES, FEATURES), 4, ctx)),
        TArc(_randw(_s2(FEATURES, FEATURES), 5, ctx)),
        TArc(_randw(_s2(MLPDIM, FEATURES), 6, ctx)),
        TArc(_randw(_s2(MLPDIM, FEATURES), 7, ctx)),
        TArc(_randw(_s2(FEATURES, MLPDIM), 8, ctx)),
        TArc(_rands(_s1(HEADDIM), 9, ctx)),
        TArc(_rands(_s1(HEADDIM), 10, ctx)),
        TArc(_rands(_s1(FEATURES), 11, ctx)),
        TArc(_rands(_s1(FEATURES), 12, ctx)),
        TArc(_rands(_s1(6 * FEATURES), 13, ctx)),
    )


def _make_lora(ctx: DeviceContext) raises -> Krea2BlockLora:
    return Krea2BlockLora(
        _adapter(FEATURES, HEADS * HEADDIM, 100, ctx),
        _adapter(FEATURES, KVHEADS * HEADDIM, 102, ctx),
        _adapter(FEATURES, KVHEADS * HEADDIM, 104, ctx),
        _adapter(FEATURES, FEATURES, 106, ctx),
        _adapter(FEATURES, FEATURES, 108, ctx),
        _adapter(FEATURES, MLPDIM, 110, ctx),
        _adapter(FEATURES, MLPDIM, 112, ctx),
        _adapter(MLPDIM, FEATURES, 114, ctx),
    )


# per-token RoPE table [L, HALF]: token-index-dependent rotation (distinct per sample
# via the `phase` offset so the two samples' tables genuinely differ).
def _table(phase: Float32, is_cos: Bool, ctx: DeviceContext) raises -> Tensor:
    var out = List[Float32]()
    for tok in range(L):
        for c in range(HALF):
            var ang = Float32(0.011) * Float32(tok + 1) * Float32(c + 1) + phase
            out.append(mcos(ang) if is_cos else msin(ang))
    return Tensor.from_host(out^, _s2(L, HALF), STDtype.BF16, ctx)


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b) or n == 0:
        return Float64(-2.0)
    var dot = Float64(0.0); var na = Float64(0.0); var nb = Float64(0.0)
    for i in range(n):
        var av = Float64(a[i]); var bv = Float64(b[i])
        dot += av * bv; na += av * av; nb += bv * bv
    if na == Float64(0.0) and nb == Float64(0.0):
        return Float64(1.0)
    if na == Float64(0.0) or nb == Float64(0.0):
        return Float64(0.0)
    return dot / (sqrt(na) * sqrt(nb))


def _sumsq_(a: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(a)):
        s += Float64(a[i]) * Float64(a[i])
    return s


def _sum2(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(a[i] + b[i])
    return out^


def _rows(t: Tensor, r0: Int, r1: Int, cols: Int, ctx: DeviceContext) raises -> List[Float32]:
    # host slice of a [1, seq, cols] tensor rows [r0:r1] flattened.
    var h = t.to_host(ctx)
    var out = List[Float32]()
    for r in range(r0, r1):
        for c in range(cols):
            out.append(h[r * cols + c])
    return out^


def _chk(name: String, a: List[Float32], b: List[Float32], mut allok: Bool):
    var c = _cos(a, b)
    var ok = c >= Float64(0.999)
    print("  ", name, " cos=", c, "  ", "PASS" if ok else "FAIL")
    if not ok:
        allok = False


def _chk_lora(name: String, gb2: Krea2LoraGrad, g0: Krea2LoraGrad, g1: Krea2LoraGrad, mut allok: Bool) raises:
    if gb2.d_a:
      if g0.d_a:
        if g1.d_a:
            _chk(name + " dA", gb2.d_a.value(), _sum2(g0.d_a.value(), g1.d_a.value()), allok)
    if gb2.d_b:
      if g0.d_b:
        if g1.d_b:
            _chk(name + " dB", gb2.d_b.value(), _sum2(g0.d_b.value(), g1.d_b.value()), allok)


comptime NCHAIN = 28


def _chain_b1(
    x: Tensor, vec: Tensor, dout: Tensor, w: Krea2BlockWeights, lora: Krea2BlockLora,
    cos: Tensor, sin: Tensor, cq: Tensor, sq: Tensor, ck: Tensor, sk: Tensor, rl: Int,
    ctx: DeviceContext,
) raises -> Tuple[List[Float32], List[Float32]]:
    var saveds = List[Krea2BlockSaved]()
    var cur = x.clone(ctx)
    for _ in range(NCHAIN):
        var f = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            TArc(cur.clone(ctx)), vec, w, lora, cos, sin, cq, sq, ck, sk, EPS, ctx, Optional[Int](rl))
        saveds.append(f.saved.copy())
        cur = f.out[].clone(ctx)
    var fwd_out = cur.to_host(ctx)
    var d = dout.clone(ctx)
    var bi = NCHAIN - 1
    while bi >= 0:
        var g = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
            d, vec, w, lora, saveds[bi], cq, sq, ck, sk, EPS, ctx, Optional[Int](rl))
        d = g.d_x[].clone(ctx)
        bi -= 1
    return (fwd_out^, d.to_host(ctx))


def _chain_b2(
    x: Tensor, vec: Tensor, dout: Tensor, w: Krea2BlockWeights, lora: Krea2BlockLora,
    cq: Tensor, sq: Tensor, ck: Tensor, sk: Tensor, rl0: Int, rl1: Int,
    ctx: DeviceContext,
) raises -> Tuple[List[Float32], List[Float32]]:
    var saveds = List[Krea2BlockSaved]()
    var cur = x.clone(ctx)
    for _ in range(NCHAIN):
        var f = krea2_single_stream_block_lora_b2[L, HEADS, KVHEADS, HEADDIM](
            TArc(cur.clone(ctx)), vec, w, lora, cq, sq, ck, sk, EPS, ctx, rl0, rl1)
        saveds.append(f.saved.copy())          # keep the ORIGINAL forward saved (no recompute)
        cur = f.out[].clone(ctx)
    var fwd_out = cur.to_host(ctx)
    var d = dout.clone(ctx)
    var bi = NCHAIN - 1
    while bi >= 0:
        var g = krea2_single_stream_block_lora_backward_b2[L, HEADS, KVHEADS, HEADDIM](
            d, vec, w, lora, saveds[bi], cq, sq, ck, sk, EPS, ctx, rl0, rl1)
        d = g.d_x[].clone(ctx)
        bi -= 1
    return (fwd_out^, d.to_host(ctx))


comptime OUT_CH = 64
comptime IMG = 96
comptime TXT0 = 40
comptime TXT1 = 24


def _dummy(ctx: DeviceContext) raises -> TArc:
    return TArc(randn(_s3(1, 1, 1), 999, STDtype.BF16, ctx))


# MINI-STACK: block chain fwd → final-layer fwd struct → final-layer bwd → block chain
# bwd. Reproduces the FULL stack composition (the one integration the chain omits).
def _ministack_b2(
    x: Tensor, vec: Tensor, cq: Tensor, sq: Tensor, ck: Tensor, sk: Tensor, rl0: Int, rl1: Int,
    dv0: Tensor, dv1: Tensor, tmlp2: Tensor, w: Krea2BlockWeights, lora: Krea2BlockLora,
    wfin: Krea2StackWeights, ln: Tensor, ctx: DeviceContext,
) raises -> List[Float32]:
    var inputs = List[TArc]()
    var cur = x.clone(ctx)
    for _ in range(NCHAIN):
        inputs.append(TArc(cur.clone(ctx)))
        var f = krea2_single_stream_block_lora_b2[L, HEADS, KVHEADS, HEADDIM](
            TArc(cur.clone(ctx)), vec, w, lora, cq, sq, ck, sk, EPS, ctx, rl0, rl1)
        cur = f.out[].clone(ctx)
    var last_xn = krea2_rmsnorm(cur, ln, EPS, ctx)
    var fwd = Krea2StackForwardB2(_dummy(ctx), _dummy(ctx), List[TArc](), TArc(cur.clone(ctx)), TArc(last_xn^), TXT0, TXT1, IMG)
    var d = krea2_final_layer_backward_b2[L, HEADS, HEADDIM](dv0, dv1, fwd, tmlp2, wfin, EPS, ctx)
    var bi = NCHAIN - 1
    while bi >= 0:
        var rb = krea2_single_stream_block_lora_b2[L, HEADS, KVHEADS, HEADDIM](
            inputs[bi].copy(), vec, w, lora, cq, sq, ck, sk, EPS, ctx, rl0, rl1)
        var g = krea2_single_stream_block_lora_backward_b2[L, HEADS, KVHEADS, HEADDIM](
            d, vec, w, lora, rb.saved, cq, sq, ck, sk, EPS, ctx, rl0, rl1)
        d = g.d_x[].clone(ctx)
        bi -= 1
    return d.to_host(ctx)


def _ministack_b1(
    x: Tensor, vec: Tensor, cos: Tensor, sin: Tensor, cq: Tensor, sq: Tensor, ck: Tensor, sk: Tensor, rl: Int,
    dv: Tensor, tvec: Tensor, txt: Int, w: Krea2BlockWeights, lora: Krea2BlockLora,
    wfin: Krea2StackWeights, ln: Tensor, ctx: DeviceContext,
) raises -> List[Float32]:
    var inputs = List[TArc]()
    var cur = x.clone(ctx)
    for _ in range(NCHAIN):
        inputs.append(TArc(cur.clone(ctx)))
        var f = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            TArc(cur.clone(ctx)), vec, w, lora, cos, sin, cq, sq, ck, sk, EPS, ctx, Optional[Int](rl))
        cur = f.out[].clone(ctx)
    var last_xn = krea2_rmsnorm(cur, ln, EPS, ctx)
    var fwd = Krea2StackForward(_dummy(ctx), List[TArc](), TArc(cur.clone(ctx)), TArc(last_xn^), txt, IMG)
    var d = krea2_final_layer_backward[L, HEADS, HEADDIM](dv, fwd, tvec, wfin, EPS, ctx)
    var bi = NCHAIN - 1
    while bi >= 0:
        var rb = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            inputs[bi].copy(), vec, w, lora, cos, sin, cq, sq, ck, sk, EPS, ctx, Optional[Int](rl))
        var g = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
            d, vec, w, lora, rb.saved, cq, sq, ck, sk, EPS, ctx, Optional[Int](rl))
        d = g.d_x[].clone(ctx)
        bi -= 1
    return d.to_host(ctx)


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_b2_block_gate (block-level b2 vs two b1; L=", L, " RL0=", RL0, " RL1=", RL1, ") ====")
    var w = _make_weights(ctx)
    var lora = _make_lora(ctx)

    var x0 = _rand(_s3(1, L, FEATURES), 60, ctx)
    var x1 = _rand(_s3(1, L, FEATURES), 61, ctx)
    var vec0 = _rand(_s2(1, 6 * FEATURES), 70, ctx)
    var vec1 = _rand(_s2(1, 6 * FEATURES), 71, ctx)
    var dout0 = _rand(_s3(1, L, FEATURES), 80, ctx)
    var dout1 = _rand(_s3(1, L, FEATURES), 81, ctx)

    var cos0 = _table(Float32(0.0), True, ctx); var sin0 = _table(Float32(0.0), False, ctx)
    var cos1 = _table(Float32(0.3), True, ctx); var sin1 = _table(Float32(0.3), False, ctx)

    # tiled tables per sample.
    var cq0 = _tile_rope_table(cos0, L, HEADS, HALF, ctx); var sq0 = _tile_rope_table(sin0, L, HEADS, HALF, ctx)
    var ck0 = _tile_rope_table(cos0, L, KVHEADS, HALF, ctx); var sk0 = _tile_rope_table(sin0, L, KVHEADS, HALF, ctx)
    var cq1 = _tile_rope_table(cos1, L, HEADS, HALF, ctx); var sq1 = _tile_rope_table(sin1, L, HEADS, HALF, ctx)
    var ck1 = _tile_rope_table(cos1, L, KVHEADS, HALF, ctx); var sk1 = _tile_rope_table(sin1, L, KVHEADS, HALF, ctx)

    # ── b1 sample0 ──
    var f0 = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
        TArc(x0.clone(ctx)), vec0, w, lora, cos0, sin0, cq0, sq0, ck0, sk0, EPS, ctx, Optional[Int](RL0))
    var g0 = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        dout0.clone(ctx), vec0, w, lora, f0.saved, cq0, sq0, ck0, sk0, EPS, ctx, Optional[Int](RL0))
    # ── b1 sample1 ──
    var f1 = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
        TArc(x1.clone(ctx)), vec1, w, lora, cos1, sin1, cq1, sq1, ck1, sk1, EPS, ctx, Optional[Int](RL1))
    var g1 = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        dout1.clone(ctx), vec1, w, lora, f1.saved, cq1, sq1, ck1, sk1, EPS, ctx, Optional[Int](RL1))

    # ── b2 stacked ──
    var x2 = concat(1, ctx, x0, x1)                 # [1,2L,F]
    var vec2 = concat(0, ctx, vec0, vec1)           # [2,6F]
    var dout2 = concat(1, ctx, dout0, dout1)        # [1,2L,F]
    var cq2 = concat(0, ctx, cq0, cq1); var sq2 = concat(0, ctx, sq0, sq1)
    var ck2 = concat(0, ctx, ck0, ck1); var sk2 = concat(0, ctx, sk0, sk1)

    # ── IN-PLACE MUTATION CHECK (team-lead hypothesis): the STACK does
    # block_inputs.append(x.copy()) [ARC copy, shares the tensor] then runs the block
    # on that same arc. If the b2 block mutates its input in-place, the saved input is
    # corrupted → backward wrong, loss fine. Here: hold an arc copy (shares tensor),
    # deep-clone the pre-image, run the block, compare the (shared) arc vs the clone. ─
    print("---- IN-PLACE MUTATION CHECK (does the b2 block corrupt its shared input?) ----")
    var xin = TArc(x2.clone(ctx))
    var pre_img = xin[].clone(ctx)          # deep snapshot BEFORE the block
    var shared_arc = xin.copy()             # ARC copy — same underlying tensor as xin
    var _fmut = krea2_single_stream_block_lora_b2[L, HEADS, KVHEADS, HEADDIM](
        xin, vec2, w, lora, cq2, sq2, ck2, sk2, EPS, ctx, RL0, RL1)
    ctx.synchronize()
    var mutcos = _cos(shared_arc[].to_host(ctx), pre_img.to_host(ctx))
    print("  b2 block input after fwd vs pre-clone: cos =", mutcos, "  (1.0=clean, <1=MUTATED)")
    # b1 comparison
    var xin1 = TArc(x0.clone(ctx))
    var pre_img1 = xin1[].clone(ctx)
    var shared_arc1 = xin1.copy()
    var _fmut1 = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
        xin1, vec0, w, lora, cos0, sin0, cq0, sq0, ck0, sk0, EPS, ctx, Optional[Int](RL0))
    ctx.synchronize()
    var mutcos1 = _cos(shared_arc1[].to_host(ctx), pre_img1.to_host(ctx))
    print("  b1 block input after fwd vs pre-clone: cos =", mutcos1, "  (b1 reference)")
    print("")

    var f2 = krea2_single_stream_block_lora_b2[L, HEADS, KVHEADS, HEADDIM](
        TArc(x2.clone(ctx)), vec2, w, lora, cq2, sq2, ck2, sk2, EPS, ctx, RL0, RL1)
    var g2 = krea2_single_stream_block_lora_backward_b2[L, HEADS, KVHEADS, HEADDIM](
        dout2.clone(ctx), vec2, w, lora, f2.saved, cq2, sq2, ck2, sk2, EPS, ctx, RL0, RL1)

    var allok = True
    print("")
    print("---- forward out (deterministic) ----")
    _chk("out sample0", _rows(f2.out[], 0, L, FEATURES, ctx), _rows(f0.out[], 0, L, FEATURES, ctx), allok)
    _chk("out sample1", _rows(f2.out[], L, 2 * L, FEATURES, ctx), _rows(f1.out[], 0, L, FEATURES, ctx), allok)

    print("")
    print("---- backward d_x per sample ----")
    _chk("d_x sample0", _rows(g2.d_x[], 0, L, FEATURES, ctx), _rows(g0.d_x[], 0, L, FEATURES, ctx), allok)
    _chk("d_x sample1", _rows(g2.d_x[], L, 2 * L, FEATURES, ctx), _rows(g1.d_x[], 0, L, FEATURES, ctx), allok)

    print("")
    print("---- LoRA grads: b2 vs (grad0 + grad1) ----")
    _chk_lora("wq", g2.wq, g0.wq, g1.wq, allok)
    _chk_lora("wk", g2.wk, g0.wk, g1.wk, allok)
    _chk_lora("wv", g2.wv, g0.wv, g1.wv, allok)
    _chk_lora("gate", g2.gate_w, g0.gate_w, g1.gate_w, allok)
    _chk_lora("wo", g2.wo, g0.wo, g1.wo, allok)
    _chk_lora("mlp_gate", g2.mlp_gate_w, g0.mlp_gate_w, g1.mlp_gate_w, allok)
    _chk_lora("mlp_up", g2.mlp_up_w, g0.mlp_up_w, g1.mlp_up_w, allok)
    _chk_lora("mlp_down", g2.mlp_down_w, g0.mlp_down_w, g1.mlp_down_w, allok)

    # ── IDENTICAL samples: b2(x0,x0) must equal g0+g0 (d_x[0:L]==g0.d_x) ─────────
    print("")
    print("---- IDENTICAL samples: b2(x0,x0) vs 2*g0 ----")
    var x2i = concat(1, ctx, x0, x0)
    var vec2i = concat(0, ctx, vec0, vec0)
    var dout2i = concat(1, ctx, dout0, dout0)
    var cq2i = concat(0, ctx, cq0, cq0); var sq2i = concat(0, ctx, sq0, sq0)
    var ck2i = concat(0, ctx, ck0, ck0); var sk2i = concat(0, ctx, sk0, sk0)
    var f2i = krea2_single_stream_block_lora_b2[L, HEADS, KVHEADS, HEADDIM](
        TArc(x2i.clone(ctx)), vec2i, w, lora, cq2i, sq2i, ck2i, sk2i, EPS, ctx, RL0, RL0)
    var g2i = krea2_single_stream_block_lora_backward_b2[L, HEADS, KVHEADS, HEADDIM](
        dout2i.clone(ctx), vec2i, w, lora, f2i.saved, cq2i, sq2i, ck2i, sk2i, EPS, ctx, RL0, RL0)
    _chk("id out[0:L] vs f0.out", _rows(f2i.out[], 0, L, FEATURES, ctx), _rows(f0.out[], 0, L, FEATURES, ctx), allok)
    _chk("id d_x[0:L] vs g0.d_x", _rows(g2i.d_x[], 0, L, FEATURES, ctx), _rows(g0.d_x[], 0, L, FEATURES, ctx), allok)
    _chk_lora("id wq", g2i.wq, g0.wq, g0.wq, allok)
    _chk_lora("id gate", g2i.gate_w, g0.gate_w, g0.gate_w, allok)
    _chk_lora("id mlp_down", g2i.mlp_down_w, g0.mlp_down_w, g0.mlp_down_w, allok)

    # ── N-BLOCK CHAIN w/ recompute (mirror the stack); systematic vs noise ──────
    print("")
    print("---- ", NCHAIN, "-block chain (recompute): b2 d_combined vs b1, and b2 self-noise ----")
    var r_b1s0 = _chain_b1(x0, vec0, dout0, w, lora, cos0, sin0, cq0, sq0, ck0, sk0, RL0, ctx)
    var r_b1s1 = _chain_b1(x1, vec1, dout1, w, lora, cos1, sin1, cq1, sq1, ck1, sk1, RL1, ctx)
    var r_b2 = _chain_b2(x2, vec2, dout2, w, lora, cq2, sq2, ck2, sk2, RL0, RL1, ctx)
    var cols = L * FEATURES
    var fo_b2_s0 = List[Float32](); var fo_b1_s0 = List[Float32]()
    var fo_b2_s1 = List[Float32](); var fo_b1_s1 = List[Float32]()
    var dc_b2_s0 = List[Float32](); var dc_b1_s0 = List[Float32]()
    var dc_b2_s1 = List[Float32](); var dc_b1_s1 = List[Float32]()
    for j in range(cols):
        fo_b2_s0.append(r_b2[0][j]); fo_b1_s0.append(r_b1s0[0][j])
        fo_b2_s1.append(r_b2[0][cols + j]); fo_b1_s1.append(r_b1s1[0][j])
        dc_b2_s0.append(r_b2[1][j]); dc_b1_s0.append(r_b1s0[1][j])
        dc_b2_s1.append(r_b2[1][cols + j]); dc_b1_s1.append(r_b1s1[1][j])
    print("  FORWARD out  s0: cos =", _cos(fo_b2_s0, fo_b1_s0), "   s1: cos =", _cos(fo_b2_s1, fo_b1_s1))
    print("  BACKWARD d_comb s0: cos =", _cos(dc_b2_s0, dc_b1_s0), "   s1: cos =", _cos(dc_b2_s1, dc_b1_s1))
    # split sample1 d_combined into REAL rows [0:RL1] vs PAD rows [RL1:L]
    var real1_b2 = List[Float32](); var real1_b1 = List[Float32]()
    var pad1_b2 = List[Float32](); var pad1_b1 = List[Float32]()
    for r in range(L):
        for c in range(FEATURES):
            if r < RL1:
                real1_b2.append(dc_b2_s1[r * FEATURES + c]); real1_b1.append(dc_b1_s1[r * FEATURES + c])
            else:
                pad1_b2.append(dc_b2_s1[r * FEATURES + c]); pad1_b1.append(dc_b1_s1[r * FEATURES + c])
    print("  sample1 d_comb REAL rows [0:", RL1, "] cos =", _cos(real1_b2, real1_b1),
          "   PAD rows [", RL1, ":", L, "] cos =", _cos(pad1_b2, pad1_b1))
    print("  sample1 REAL ||b2||=", sqrt(_sumsq_(real1_b2)), " PAD ||b2||=", sqrt(_sumsq_(pad1_b2)))
    _chk("chain d_combined s1 REAL rows (bar 0.999)", real1_b2, real1_b1, allok)

    # ── IDENTICAL-samples chain: bit-identical expected (removes explosion confound) ─
    var xii = concat(1, ctx, x0, x0)
    var vecii = concat(0, ctx, vec0, vec0)
    var doutii = concat(1, ctx, dout0, dout0)
    var cqii = concat(0, ctx, cq0, cq0); var sqii = concat(0, ctx, sq0, sq0)
    var ckii = concat(0, ctx, ck0, ck0); var skii = concat(0, ctx, sk0, sk0)
    var r_b2id = _chain_b2(xii, vecii, doutii, w, lora, cqii, sqii, ckii, skii, RL0, RL0, ctx)
    var idb2_s0 = List[Float32](); var idb1_s0 = List[Float32]()
    var idb2_s1 = List[Float32]()
    for j in range(cols):
        idb2_s0.append(r_b2id[1][j]); idb1_s0.append(r_b1s0[1][j])
        idb2_s1.append(r_b2id[1][cols + j])
    print("  IDENTICAL chain: b2(x0,x0)[0:L] vs b1(x0) cos =", _cos(idb2_s0, idb1_s0),
          "   halves match [0:L]vs[L:2L] cos =", _cos(idb2_s0, idb2_s1))
    _chk("IDENTICAL chain d_comb (bar 0.999)", idb2_s0, idb1_s0, allok)

    # ── MINI-STACK: final-layer bwd → block chain bwd (the untested integration) ─
    print("")
    print("---- MINI-STACK (final layer + ", NCHAIN, "-block chain): b2 d_combined vs b1 ----")
    var last_norm = _rands(_s1(FEATURES), 200, ctx)
    var last_mod_lin = _rands(_s2(2, FEATURES), 201, ctx)
    var last_lin_w = _randw(_s2(OUT_CH, FEATURES), 202, ctx)
    var last_lin_b = _rands(_s1(OUT_CH), 203, ctx)
    var wfin = Krea2StackWeights(List[Krea2BlockWeights](), TArc(last_norm.clone(ctx)),
        TArc(last_mod_lin.clone(ctx)), TArc(last_lin_w.clone(ctx)), TArc(last_lin_b.clone(ctx)))
    var tvec0 = _rand(_s3(1, 1, FEATURES), 210, ctx)
    var tvec1 = _rand(_s3(1, 1, FEATURES), 211, ctx)
    var tmlp2 = concat(0, ctx, tvec0, tvec1)
    var dv0 = _rand(_s3(1, IMG, OUT_CH), 220, ctx)
    var dv1 = _rand(_s3(1, IMG, OUT_CH), 221, ctx)
    var ms_b2 = _ministack_b2(x2, vec2, cq2, sq2, ck2, sk2, RL0, RL1, dv0, dv1, tmlp2, w, lora, wfin, last_norm, ctx)
    var ms_b1s0 = _ministack_b1(x0, vec0, cos0, sin0, cq0, sq0, ck0, sk0, RL0, dv0, tvec0, TXT0, w, lora, wfin, last_norm, ctx)
    var ms_b1s1 = _ministack_b1(x1, vec1, cos1, sin1, cq1, sq1, ck1, sk1, RL1, dv1, tvec1, TXT1, w, lora, wfin, last_norm, ctx)
    var msb2_s0 = List[Float32](); var msb1_s0 = List[Float32]()
    var msb2_s1 = List[Float32](); var msb1_s1 = List[Float32]()
    for j in range(cols):
        msb2_s0.append(ms_b2[j]); msb1_s0.append(ms_b1s0[j])
        msb2_s1.append(ms_b2[cols + j]); msb1_s1.append(ms_b1s1[j])
    print("  MINI-STACK d_combined s0: cos =", _cos(msb2_s0, msb1_s0), "   s1: cos =", _cos(msb2_s1, msb1_s1))
    _chk("mini-stack d_combined s0 (bar 0.999)", msb2_s0, msb1_s0, allok)
    _chk("mini-stack d_combined s1 (bar 0.999)", msb2_s1, msb1_s1, allok)

    # ── FLASH sliced-vs-standalone: is cuDNN flash fwd context-dependent? ───────
    # Same VALUES in [0:L], but once standalone [1,L,H,Dh] and once a slice of a
    # [1,2L,H,Dh] tensor. If o differs, cuDNN flash is context-dependent → the source
    # of the tiny per-block b2-vs-b1 forward diff (amplified over 28 blocks).
    print("")
    print("---- FLASH forward: standalone [1,L] vs sliced-from-[1,2L] (same values) ----")
    var q4 = _rand(_s4b(1, L, HEADS, HEADDIM), 300, ctx)
    var k4 = _rand(_s4b(1, L, HEADS, HEADDIM), 301, ctx)
    var v4 = _rand(_s4b(1, L, HEADS, HEADDIM), 302, ctx)
    var garb = _rand(_s4b(1, L, HEADS, HEADDIM), 303, ctx)
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))
    var o_std = sdpa_flash_train_fwd_padmask_bf16[1, L, HEADS, HEADDIM](q4, k4, v4, RL0, scale, ctx)
    var q2s = concat(1, ctx, q4, garb); var k2s = concat(1, ctx, k4, garb); var v2s = concat(1, ctx, v4, garb)
    var q_sl = slice(q2s, 1, 0, L, ctx); var k_sl = slice(k2s, 1, 0, L, ctx); var v_sl = slice(v2s, 1, 0, L, ctx)
    var o_sl = sdpa_flash_train_fwd_padmask_bf16[1, L, HEADS, HEADDIM](q_sl, k_sl, v_sl, RL0, scale, ctx)
    var oh_std = o_std.att.to_host(ctx); var oh_sl = o_sl.att.to_host(ctx)
    var mad = Float64(0.0)
    for j in range(len(oh_std)):
        var dd = Float64(oh_std[j]) - Float64(oh_sl[j])
        if dd < Float64(0.0):
            dd = -dd
        if dd > mad:
            mad = dd
    print("  flash o  standalone vs sliced: cos =", _cos(oh_std, oh_sl), "  max_abs =", mad)

    print("")
    print("VERDICT:", "PASS" if allok else "FAIL")
