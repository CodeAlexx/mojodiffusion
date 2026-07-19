# models/dit/tests/hidream_o1_block_b2_parity.mojo — TRUE batch-2 (row-stacked)
# block gate for the HiDream-O1 plain-LoRA arm.
#
# SELF-CONSISTENT (no torch oracle): synthetic-but-real-shaped F32 weights/LoRA and
# TWO distinct samples (distinct per-sample rope tables + prefix-causal masks, via
# different ar_len). Runs the b1 block forward/backward TWICE (one per sample) and the
# b2 row-stacked block forward/backward ONCE, then asserts the row-stack invariants:
#
#   (BINDING) per-sample forward-out parity: b2.out[0:S] vs b1_s0.out and
#             b2.out[S:2S] vs b1_s1.out  cosine >= 0.999.  Attention is per-sample
#             (never a B=2 kernel); a cross-sample leak collapses this cosine.
#   (BINDING/structural) per-sample d_hidden: b2.d_hidden[0:S] vs b1_s0.d_hidden and
#             b2.d_hidden[S:2S] vs b1_s1.d_hidden  cosine >= 0.999.  Deterministic F32
#             math sdpa (no flash nondeterminism) — a backward cross-sample leak fails
#             this.  (b2 is fed the plain concat(d_out0,d_out1) so each half's d_hidden
#             equals the matching b1 run exactly.)
#   (INFORMATIONAL, MJ-1073) per-slot LoRA dA/dB: cosine of the b2 batch grad vs
#             mean(b1_s0, b1_s1).  NOT gated: the same rows through the same GEMM at
#             M=S vs M=2S differ by shape-deterministic GEMM tiling (bf16 in the real
#             trainer; tiny F32 ULPs here), so grad-cosine-vs-b1 is the WRONG bar for
#             row-stacked b2 (krea2 MJ-1072/1073).  cosine is scale-invariant, so
#             mean-vs-sum is irrelevant to this number (block has no 0.5 loss scale —
#             that lives in the trainer's d_out).
#
# Reduced-but-faithful dims: D=256 H=8 HKV=2 Dh=32 F=512 S=96, rank-4 LoRA on all 7
# slots (B perturbed to small NONZERO to lift grads off the floor), F32 end-to-end.
#
# Build (mem-safe -O2; NO cuDNN shim — the math-path sdpa/sdpa_backward_masked, same
# flags as hidream_o1_block_parity.mojo):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   MEM_MAX=28G MEM_HIGH=24G SWAP_MAX=2G bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#       serenitymojo/models/dit/tests/hidream_o1_block_b2_parity.mojo \
#       -o /tmp/hidream_block_b2_par
# Run: LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/hidream_block_b2_par

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.lora_block import ZImageLoraAdapterDevice
from serenitymojo.ops.tensor_algebra import concat as ta_concat
from serenitymojo.models.dit.hidream_o1_train_block import (
    HiDreamO1BlockWeights,
    HiDreamO1BlockLora,
    hidream_o1_block_lora_forward,
    hidream_o1_block_lora_backward,
    hidream_o1_block_lora_forward_b2,
    hidream_o1_block_lora_backward_b2,
)

comptime TArc = ArcPointer[Tensor]
comptime D = 256
comptime H = 8
comptime HKV = 2
comptime Dh = 32
comptime F = 512
comptime S = 96
comptime RANK = 4
comptime LSCALE = Float32(0.5)
comptime EPS = Float32(1.0e-6)
comptime COS_BAR = Float64(0.999)


# ── host PRNG (LCG) → deterministic F32 in [-amp, amp] ─────────────────────────
struct _Rng(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float32:
        self.s = self.s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Int((self.s >> UInt64(40)) % UInt64(2000)) - 1000
        return Float32(u) * Float32(0.001)   # [-1, 1]


def _rand_list(mut rng: _Rng, n: Int, amp: Float32) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(rng.next() * amp)
    return out^


def _rand_t(mut rng: _Rng, var shape: List[Int], amp: Float32, ctx: DeviceContext) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return Tensor.from_host(_rand_list(rng, n, amp), shape^, STDtype.F32, ctx)


def _rand_ta(mut rng: _Rng, var shape: List[Int], amp: Float32, ctx: DeviceContext) raises -> TArc:
    return TArc(_rand_t(rng, shape^, amp, ctx))


# ── cosine (F64 accumulation) ─────────────────────────────────────────────────
def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b) or n == 0:
        return -2.0
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(n):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def _mean2(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(Float32(0.5) * (a[i] + b[i]))
    return out^


def _gate(name: String, got: List[Float32], exp: List[Float32]) raises -> Bool:
    var c = _cos(got, exp)
    var ok = c >= COS_BAR
    print("GATE hidream_block_b2 " + name + " cos=", c, "  PASS" if ok else "  FAIL")
    return ok


# host slice of a flat [1,2S,D]-style buffer: rows [r0, r0+rows) each `cols` wide.
def _rows(flat: List[Float32], r0: Int, rows: Int, cols: Int) -> List[Float32]:
    var out = List[Float32]()
    for r in range(rows):
        for c in range(cols):
            out.append(flat[(r0 + r) * cols + c])
    return out^


# ── per-head-replicated rope table: [S,half] positions → [S*heads*half] rows
# (row = s*heads + h), matching x flattened [1,S,h,Dh] row order. ───────────────
def _replicate_heads_host(half_tab: List[Float32], heads: Int) raises -> List[Float32]:
    comptime half = Dh // 2
    var out = List[Float32]()
    for s in range(S):
        for _h in range(heads):
            for c in range(half):
                out.append(half_tab[s * half + c])
    return out^


# prefix-causal additive mask host buffer [H*S*S]: col j allowed for row i if
# j < ar_len (prefix, all rows attend) OR j <= i (causal). masked → -1e9.
def _mask_host(ar_len: Int) raises -> List[Float32]:
    var out = List[Float32]()
    for _h in range(H):
        for i in range(S):
            for j in range(S):
                var allowed = (j < ar_len) or (j <= i)
                out.append(Float32(0.0) if allowed else Float32(-1.0e9))
    return out^


# build one sample's rope tables (cos_q/sin_q/cos_k/sin_k, F32 flat) + masks
# (mask4 [1,H,S,S] + mask_f32 [H*S,S]) from a seed + ar_len.
struct _SampleCond(Movable):
    var cos_q: Tensor
    var sin_q: Tensor
    var cos_k: Tensor
    var sin_k: Tensor
    var mask4: Tensor
    var mask_f32: Tensor

    def __init__(
        out self, var cos_q: Tensor, var sin_q: Tensor,
        var cos_k: Tensor, var sin_k: Tensor,
        var mask4: Tensor, var mask_f32: Tensor,
    ):
        self.cos_q = cos_q^
        self.sin_q = sin_q^
        self.cos_k = cos_k^
        self.sin_k = sin_k^
        self.mask4 = mask4^
        self.mask_f32 = mask_f32^


def _build_sample(seed: UInt64, ar_len: Int, ctx: DeviceContext) raises -> _SampleCond:
    comptime half = Dh // 2
    var rng = _Rng(seed)
    # per-position half-tables of cos/sin from random angles (proper rotation).
    var cos_half = List[Float32]()
    var sin_half = List[Float32]()
    for _ in range(S * half):
        var ang = rng.next() * Float32(3.14159265)
        # cos/sin via truncated series would be overkill; use the identity that any
        # (c,s) with c=cos, s=sin is a valid rotation — approximate with a cheap map.
        cos_half.append(_cosf(ang))
        sin_half.append(_sinf(ang))
    var cq: List[Int] = [S * H * half]
    var ck: List[Int] = [S * HKV * half]
    var cos_q = Tensor.from_host(_replicate_heads_host(cos_half, H), cq.copy(), STDtype.F32, ctx)
    var sin_q = Tensor.from_host(_replicate_heads_host(sin_half, H), cq^, STDtype.F32, ctx)
    var cos_k = Tensor.from_host(_replicate_heads_host(cos_half, HKV), ck.copy(), STDtype.F32, ctx)
    var sin_k = Tensor.from_host(_replicate_heads_host(sin_half, HKV), ck^, STDtype.F32, ctx)
    var mask_h = _mask_host(ar_len)
    var m4_sh: List[Int] = [1, H, S, S]
    var mask4 = Tensor.from_host(mask_h.copy(), m4_sh^, STDtype.F32, ctx)
    var mhs_sh: List[Int] = [H * S, S]
    var mask_f32 = Tensor.from_host(mask_h^, mhs_sh^, STDtype.F32, ctx)
    return _SampleCond(cos_q^, sin_q^, cos_k^, sin_k^, mask4^, mask_f32^)


# minimal cos/sin (range-reduced Maclaurin, enough for a rotation table).
def _cosf(x: Float32) -> Float32:
    var t = x
    var term = Float32(1.0)
    var sum = Float32(1.0)
    for k in range(1, 8):
        term = term * (-(t * t)) / Float32((2 * k - 1) * (2 * k))
        sum += term
    return sum


def _sinf(x: Float32) -> Float32:
    var t = x
    var term = t
    var sum = t
    for k in range(1, 8):
        term = term * (-(t * t)) / Float32((2 * k) * (2 * k + 1))
        sum += term
    return sum


def _adapter(
    mut rng: _Rng, in_f: Int, out_f: Int, ctx: DeviceContext
) raises -> Optional[ZImageLoraAdapterDevice]:
    # A random; B perturbed to small NONZERO (real init B=0 → grads near the floor).
    var a_sh: List[Int] = [RANK, in_f]
    var b_sh: List[Int] = [out_f, RANK]
    var a = _rand_ta(rng, a_sh^, Float32(0.05), ctx)
    var b = _rand_ta(rng, b_sh^, Float32(0.02), ctx)
    return Optional[ZImageLoraAdapterDevice](
        ZImageLoraAdapterDevice(a, b, RANK, in_f, out_f, LSCALE)
    )


def main() raises:
    var ctx = DeviceContext()
    print("==== hidream_o1_block_b2_parity (TRUE batch-2 block vs two b1 runs; F32) ====")
    print("D=", D, " H=", H, " HKV=", HKV, " Dh=", Dh, " F=", F, " S=", S, " RANK=", RANK)

    # ── frozen base weights (F32) ─────────────────────────────────────────────
    var wr = _Rng(UInt64(11))
    var in_ln_sh: List[Int] = [D]
    var qw_sh: List[Int] = [H * Dh, D]
    var kw_sh: List[Int] = [HKV * Dh, D]
    var vw_sh: List[Int] = [HKV * Dh, D]
    var qn_sh: List[Int] = [Dh]
    var kn_sh: List[Int] = [Dh]
    var ow_sh: List[Int] = [D, H * Dh]
    var pln_sh: List[Int] = [D]
    var gw_sh: List[Int] = [F, D]
    var uw_sh: List[Int] = [F, D]
    var dw_sh: List[Int] = [D, F]
    var w = HiDreamO1BlockWeights(
        _rand_ta(wr, in_ln_sh^, Float32(1.0), ctx),
        _rand_ta(wr, qw_sh^, Float32(0.08), ctx),
        _rand_ta(wr, kw_sh^, Float32(0.08), ctx),
        _rand_ta(wr, vw_sh^, Float32(0.08), ctx),
        _rand_ta(wr, qn_sh^, Float32(1.0), ctx),
        _rand_ta(wr, kn_sh^, Float32(1.0), ctx),
        _rand_ta(wr, ow_sh^, Float32(0.08), ctx),
        _rand_ta(wr, pln_sh^, Float32(1.0), ctx),
        _rand_ta(wr, gw_sh^, Float32(0.08), ctx),
        _rand_ta(wr, uw_sh^, Float32(0.08), ctx),
        _rand_ta(wr, dw_sh^, Float32(0.08), ctx),
    )

    # ── LoRA (fresh adapters for b1 and b2 — SAME values via same seeds) ───────
    # Build one HiDreamO1BlockLora; ZImageLoraAdapterDevice is Copyable, and the
    # b1/b2 forwards only READ it, so one instance is shared across all calls.
    var lr = _Rng(UInt64(777))
    var lora = HiDreamO1BlockLora(
        _adapter(lr, D, H * Dh, ctx), _adapter(lr, D, HKV * Dh, ctx),
        _adapter(lr, D, HKV * Dh, ctx), _adapter(lr, H * Dh, D, ctx),
        _adapter(lr, D, F, ctx), _adapter(lr, D, F, ctx),
        _adapter(lr, F, D, ctx),
    )

    # ── two DISTINCT samples (different rope seeds + ar_len) ──────────────────
    var c0 = _build_sample(UInt64(101), 32, ctx)
    var c1 = _build_sample(UInt64(202), 48, ctx)

    # ── hidden inputs: hidden0 [1,S,D], hidden1 [1,S,D], combined [1,2S,D] ─────
    var hr = _Rng(UInt64(9001))
    var h0_sh: List[Int] = [1, S, D]
    var h1_sh: List[Int] = [1, S, D]
    var hidden0 = _rand_ta(hr, h0_sh^, Float32(0.5), ctx)
    var hidden1 = _rand_ta(hr, h1_sh^, Float32(0.5), ctx)
    var combined = TArc(ta_concat(1, ctx, hidden0[], hidden1[]))   # [1,2S,D]

    # ── upstream grads: d_out0 [1,S,D], d_out1 [1,S,D], combined [1,2S,D] ──────
    var dr = _Rng(UInt64(5005))
    var do0_sh: List[Int] = [1, S, D]
    var do1_sh: List[Int] = [1, S, D]
    var d_out0 = _rand_t(dr, do0_sh^, Float32(0.3), ctx)
    var d_out1 = _rand_t(dr, do1_sh^, Float32(0.3), ctx)
    var d_out_b2 = ta_concat(1, ctx, d_out0, d_out1)              # [1,2S,D]

    # ── PRE-TILE rope tables to 2S rows: concat(sample0 rows, sample1 rows) ────
    var cos_q2 = ta_concat(0, ctx, c0.cos_q, c1.cos_q)   # [2S*H*half]
    var sin_q2 = ta_concat(0, ctx, c0.sin_q, c1.sin_q)
    var cos_k2 = ta_concat(0, ctx, c0.cos_k, c1.cos_k)   # [2S*HKV*half]
    var sin_k2 = ta_concat(0, ctx, c0.sin_k, c1.sin_k)

    # ── b1 forward twice ──────────────────────────────────────────────────────
    print("[fwd] b1 sample0, b1 sample1, b2 combined ...")
    var f0 = hidream_o1_block_lora_forward[S, H, HKV, Dh](
        hidden0, w, lora, c0.cos_q, c0.sin_q, c0.cos_k, c0.sin_k, c0.mask4, D, F, EPS, ctx)
    var f1 = hidream_o1_block_lora_forward[S, H, HKV, Dh](
        hidden1, w, lora, c1.cos_q, c1.sin_q, c1.cos_k, c1.sin_k, c1.mask4, D, F, EPS, ctx)
    var fb2 = hidream_o1_block_lora_forward_b2[S, H, HKV, Dh](
        combined, w, lora, cos_q2, sin_q2, cos_k2, sin_k2, c0.mask4, c1.mask4, D, F, EPS, ctx)
    ctx.synchronize()

    var ok = True
    # BINDING: per-sample forward-out parity.
    var b2out = fb2.out[].to_host(ctx)
    var out0 = f0.out[].to_host(ctx)
    var out1 = f1.out[].to_host(ctx)
    print("---- (BINDING) forward-out per sample ----")
    ok = _gate("out[0:S] vs b1_s0", _rows(b2out, 0, S, D), out0) and ok
    ok = _gate("out[S:2S] vs b1_s1", _rows(b2out, S, S, D), out1) and ok

    # ── b1 backward twice + b2 backward once ──────────────────────────────────
    print("[bwd] b1 sample0, b1 sample1, b2 combined ...")
    var g0 = hidream_o1_block_lora_backward[S, H, HKV, Dh](
        d_out0, w, lora, f0.saved, c0.cos_q, c0.sin_q, c0.cos_k, c0.sin_k, c0.mask_f32, D, F, EPS, ctx)
    var g1 = hidream_o1_block_lora_backward[S, H, HKV, Dh](
        d_out1, w, lora, f1.saved, c1.cos_q, c1.sin_q, c1.cos_k, c1.sin_k, c1.mask_f32, D, F, EPS, ctx)
    var gb2 = hidream_o1_block_lora_backward_b2[S, H, HKV, Dh](
        d_out_b2, w, lora, fb2.saved, cos_q2, sin_q2, cos_k2, sin_k2,
        c0.mask_f32, c1.mask_f32, D, F, EPS, ctx)
    ctx.synchronize()

    # BINDING/structural: per-sample d_hidden.
    var b2dh = gb2.d_hidden[].to_host(ctx)
    var dh0 = g0.d_hidden[].to_host(ctx)
    var dh1 = g1.d_hidden[].to_host(ctx)
    print("---- (BINDING/structural) d_hidden per sample ----")
    ok = _gate("d_hidden[0:S] vs b1_s0", _rows(b2dh, 0, S, D), dh0) and ok
    ok = _gate("d_hidden[S:2S] vs b1_s1", _rows(b2dh, S, S, D), dh1) and ok

    # INFORMATIONAL (MJ-1073): per-slot LoRA dA/dB vs mean(b1_s0, b1_s1).
    # NOT gated — grad-cosine-vs-b1 is shape-deterministic-tiling sensitive at M=S
    # vs M=2S (bf16 in the real trainer). cosine is scale-invariant, so comparing
    # the b2 batch grad (== g0+g1 here; no 0.5 loss scale at block level) against
    # mean(b1) gives the same number as vs the sum. Reported for tracking only.
    print("---- (INFORMATIONAL, MJ-1073 — NOT gated) LoRA dA/dB vs mean(b1) ----")
    var names: List[String] = [
        String("q"), String("k"), String("v"), String("o"),
        String("gate"), String("up"), String("down"),
    ]
    for i in range(7):
        if gb2.d_a[i] and g0.d_a[i] and g1.d_a[i]:
            var ma = _mean2(g0.d_a[i].value()[].to_host(ctx), g1.d_a[i].value()[].to_host(ctx))
            var mb = _mean2(g0.d_b[i].value()[].to_host(ctx), g1.d_b[i].value()[].to_host(ctx))
            print("  slot " + names[i] + " dA cos=",
                  _cos(gb2.d_a[i].value()[].to_host(ctx), ma),
                  "  dB cos=", _cos(gb2.d_b[i].value()[].to_host(ctx), mb))

    print("")
    if ok:
        print("=== hidream_o1_block_b2_parity: ALL BINDING GATES PASS ===")
    else:
        print("=== hidream_o1_block_b2_parity: FAIL (see BINDING gates above) ===")
        raise Error("hidream block b2 parity failed")
