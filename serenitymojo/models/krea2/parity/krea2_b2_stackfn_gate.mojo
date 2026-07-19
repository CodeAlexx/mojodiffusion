# serenitymojo/models/krea2/parity/krea2_b2_stackfn_gate.mojo
#
# STACK-FUNCTION isolation: calls krea2_stack_lora_{forward,backward}_streamed_b2
# DIRECTLY with SYNTHETIC conditioning (incl SYNTHETIC rope) + REAL STREAMED weights,
# and compares b2 grads vs mean(b1 s0, b1 s1). Splits the variable:
#   PASS → the bug is in the REAL conditioning (_build_conditioning: real MRoPE/blk_vec).
#   FAIL → the bug is in the stack FUNCTIONS or the STREAMED-weight path.
# Uses random combined + synthetic d_velocity (backward is linear in d_velocity, so a
# random dY exercises the whole chain); b2 gets 0.5*dY per sample so b2 == mean(B1).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt, cos as mcos, sin as msin
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import concat, mul_scalar
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StreamFinal, Krea2StackLora, Krea2StackLoraGrads, Krea2ResidentFp8,
    krea2_stack_lora_forward_streamed, krea2_stack_lora_backward_streamed,
    krea2_stack_lora_forward_streamed_b2, krea2_stack_lora_backward_streamed_b2,
)
from serenitymojo.models.krea2.train_krea2 import (
    _build_host_lora, _host_to_device_lora,
    LFULL, HEADS, KVHEADS, HEADDIM, FEATURES, NBLOCKS, IMGLEN, EPS,
)

comptime TArc = ArcPointer[Tensor]
comptime CFG_PATH = "serenitymojo/configs/krea2_bf16base_1500.json"
comptime HALF = HEADDIM // 2
comptime LT0 = 174
comptime LT1 = 146


def _r(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return randn(shape^, seed, STDtype.BF16, ctx)
def _rf(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return randn(shape^, seed, STDtype.F32, ctx)   # rope tables are F32


# PROPER cos/sin rope [LFULL, HALF] — bounded rotations (norm-preserving), NOT random
# scaling. `phase` differs per sample (mimics real MRoPE where the image block sits at
# a different seq offset per sample). This is the faithful rope-swap test.
def _rope(is_cos: Bool, phase: Float32, ctx: DeviceContext) raises -> Tensor:
    var out = List[Float32]()
    for t in range(LFULL):
        for c in range(HALF):
            var ang = Float32(t) * Float32(0.002) * (Float32(c + 1) / Float32(HALF)) + phase
            out.append(mcos(ang) if is_cos else msin(ang))
    return Tensor.from_host(out^, _s2(LFULL, HALF), STDtype.F32, ctx)
def _s1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _perturb_b(mut ad: List[LoraAdapter]):
    var s = UInt64(1234567)
    for i in range(len(ad)):
        var bb = ad[i].b.copy()
        for j in range(len(bb)):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            var rr = Float32(Int((s >> UInt64(40)) % UInt64(2000)) - 1000) * Float32(0.00005)
            bb[j] = Float32(rr).cast[DType.bfloat16]()
        ad[i].b = bb^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b) or n == 0:
        return Float64(-2.0)
    var dot = Float64(0.0); var na = Float64(0.0); var nb = Float64(0.0)
    for i in range(n):
        dot += Float64(a[i]) * Float64(b[i]); na += Float64(a[i]) * Float64(a[i]); nb += Float64(b[i]) * Float64(b[i])
    if na == Float64(0.0) or nb == Float64(0.0):
        return Float64(0.0)
    return dot / (sqrt(na) * sqrt(nb))
def _sq(a: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(a)):
        s += Float64(a[i]) * Float64(a[i])
    return s


def _concat_all(g: Krea2StackLoraGrads) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(g.grads)):
        if g.grads[i].d_a:
            for j in range(len(g.grads[i].d_a.value())):
                out.append(g.grads[i].d_a.value()[j])
        if g.grads[i].d_b:
            for j in range(len(g.grads[i].d_b.value())):
                out.append(g.grads[i].d_b.value()[j])
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_b2_stackfn_gate (stack fns + SYNTHETIC conditioning + REAL streamed weights) ====")
    var cfg = read_model_config(CFG_PATH)
    var st = ShardedSafeTensors.open(cfg.checkpoint)
    var fin = Krea2StreamFinal.load(st, String(""), ctx)
    var host_lora = _build_host_lora(cfg.lora_rank, cfg.lora_alpha)
    _perturb_b(host_lora)
    var lora = _host_to_device_lora(host_lora, ctx)
    var none = Optional[Krea2ResidentFp8](None)
    var out_ch = fin.last_lin_w[].shape()[0]
    print("  LFULL=", LFULL, " HEADS=", HEADS, " FEATURES=", FEATURES, " NBLOCKS=", NBLOCKS, " IMGLEN=", IMGLEN, " out_ch=", out_ch)

    var rl0 = LT0 + IMGLEN
    var rl1 = LT1 + IMGLEN
    # synthetic conditioning (random; structure irrelevant to a b2-vs-meanB1 parity test)
    var comb0 = _r(_s3(1, LFULL, FEATURES), 1, ctx)
    var comb1 = _r(_s3(1, LFULL, FEATURES), 2, ctx)
    var bv0 = _r(_s2(1, 6 * FEATURES), 3, ctx)
    var bv1 = _r(_s2(1, 6 * FEATURES), 4, ctx)
    var tm0 = _r(_s3(1, 1, FEATURES), 5, ctx)
    var tm1 = _r(_s3(1, 1, FEATURES), 6, ctx)
    var cos0 = _rope(True, Float32(0.0), ctx); var sin0 = _rope(False, Float32(0.0), ctx)
    var cos1 = _rope(True, Float32(0.37), ctx); var sin1 = _rope(False, Float32(0.37), ctx)  # different per sample
    var dv0 = _r(_s3(1, IMGLEN, out_ch), 11, ctx)
    var dv1 = _r(_s3(1, IMGLEN, out_ch), 12, ctx)

    # ── B1 per sample (full d_velocity) ─────────────────────────────────────────
    var fwd0 = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        TArc(comb0.clone(ctx)), bv0, tm0, st, String(""), NBLOCKS, lora, fin,
        cos0, sin0, EPS, LT0, IMGLEN, ctx, Optional[Int](rl0), none)
    var g0 = krea2_stack_lora_backward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        dv0, bv0, tm0, st, String(""), NBLOCKS, lora, fin, fwd0,
        cos0, sin0, EPS, ctx, Optional[Int](rl0), none)
    var fwd1 = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        TArc(comb1.clone(ctx)), bv1, tm1, st, String(""), NBLOCKS, lora, fin,
        cos1, sin1, EPS, LT1, IMGLEN, ctx, Optional[Int](rl1), none)
    var g1 = krea2_stack_lora_backward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        dv1, bv1, tm1, st, String(""), NBLOCKS, lora, fin, fwd1,
        cos1, sin1, EPS, ctx, Optional[Int](rl1), none)

    # ── B2 (row-stacked; 0.5*d_velocity per sample → grads == mean(B1)) ──────────
    var comb2 = concat(1, ctx, comb0, comb1)
    var bv2 = concat(0, ctx, bv0, bv1)
    var tm2 = concat(0, ctx, tm0, tm1)
    var fwd2 = krea2_stack_lora_forward_streamed_b2[LFULL, HEADS, KVHEADS, HEADDIM](
        TArc(comb2^), bv2, tm2, st, String(""), NBLOCKS, lora, fin,
        cos0, sin0, cos1, sin1, EPS, LT0, LT1, IMGLEN, ctx, rl0, rl1, none)
    var g2 = krea2_stack_lora_backward_streamed_b2[LFULL, HEADS, KVHEADS, HEADDIM](
        mul_scalar(dv0, Float32(0.5), ctx), mul_scalar(dv1, Float32(0.5), ctx), bv2, tm2,
        st, String(""), NBLOCKS, lora, fin, fwd2,
        cos0, sin0, cos1, sin1, EPS, ctx, rl0, rl1, none)

    # ── FORWARD velocity consistency (hypothesis A): b2 velocity0 vs b1 so0 ──────
    var vh_b1s0 = fwd0.velocity[].to_host(ctx)
    var vh_b1s1 = fwd1.velocity[].to_host(ctx)
    var vh_b2s0 = fwd2.velocity0[].to_host(ctx)
    var vh_b2s1 = fwd2.velocity1[].to_host(ctx)
    print("  FORWARD velocity: b2_s0 vs b1_s0 cos =", _cos(vh_b2s0, vh_b1s0),
          "   b2_s1 vs b1_s1 cos =", _cos(vh_b2s1, vh_b1s1))

    # ── compare g2 vs mean(g0, g1) ──────────────────────────────────────────────
    var v2 = _concat_all(g2)
    var va = _concat_all(g0)
    var vb = _concat_all(g1)
    var vmean = List[Float32]()
    for j in range(len(va)):
        vmean.append(Float32(0.5) * (va[j] + vb[j]))
    var gc = _cos(v2, vmean)
    print("")
    print("  B2 vs mean(B1) global cosine =", gc,
          "  ||g2||=", sqrt(_sq(v2)), " ||mean||=", sqrt(_sq(vmean)))
    print("  VERDICT:", "PASS (stack fns + streamed weights OK → real conditioning/rope is the bug)" if gc >= Float64(0.999) else "FAIL (bug is in the stack fns or streamed-weight path)")
