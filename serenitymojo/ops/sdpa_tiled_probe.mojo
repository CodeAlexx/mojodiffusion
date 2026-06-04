# sdpa_tiled_probe.mojo — gate for the memory-efficient (online-softmax) SDPA.
#
# GATE 1 (CORRECTNESS): tiled SDPA must EQUAL the trusted math-mode `sdpa` /
#   `sdpa_nomask` (online softmax is exact). Tested at S=1536, Dh=128 — well
#   above _KV_BLOCK=512, so the kv loop runs 3 blocks and the per-block
#   rescaling (the bug surface) is actually exercised. Random Q/K/V, both the
#   nomask case AND a random additive mask. Report cos + magnitude ratio.
# GATE 2 (MEMORY): run the tiled path at LARGE S (8192, Dh=128) where the math
#   path's [S,S] scores buffer would be huge; show it COMPLETES with a finite,
#   correctly-shaped output (no OOM).
#
# pixi run mojo run -I . serenitymojo/ops/sdpa_tiled_probe.mojo
from std.math import sqrt, exp
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention import (
    sdpa, sdpa_nomask, sdpa_tiled, sdpa_nomask_tiled
)


def _cos_ratio(a: List[Float32], b: List[Float32]) -> List[Float32]:
    # returns [cos, |a|/|b|]
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var n = len(a) if len(a) < len(b) else len(b)
    for i in range(n):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    var cos = dot / (sqrt(na) * sqrt(nb) + 1e-30)
    var ratio = sqrt(na) / (sqrt(nb) + 1e-30)
    var out = List[Float32]()
    out.append(Float32(cos))
    out.append(Float32(ratio))
    return out^


def _pseudo(i: Int, salt: Int) -> Float32:
    # deterministic pseudo-random in ~[-1,1]
    var x = (i * 2654435761 + salt * 40503 + 12345) % 100003
    return (Float32(x) / 100003.0 - 0.5) * 2.0


def main() raises:
    var ctx = DeviceContext()

    # ───────────────────────── GATE 1: correctness (multi-block) ────────────
    comptime B = 1
    comptime S = 1536        # 3 * _KV_BLOCK(512) → multi-block rescaling tested
    comptime Hd = 3
    comptime Dh = 128
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var qn = B * S * Hd * Dh
    var qv = List[Float32]()
    var kv = List[Float32]()
    var vv = List[Float32]()
    for i in range(qn):
        qv.append(_pseudo(i, 1) * 0.5)
        kv.append(_pseudo(i, 2) * 0.5)
        vv.append(_pseudo(i, 3) * 0.5)
    var qs = List[Int]()
    qs.append(B); qs.append(S); qs.append(Hd); qs.append(Dh)
    var q = Tensor.from_host(qv, qs.copy(), STDtype.F32, ctx)
    var k = Tensor.from_host(kv, qs.copy(), STDtype.F32, ctx)
    var v = Tensor.from_host(vv, qs.copy(), STDtype.F32, ctx)

    # --- nomask parity: sdpa_nomask (math, trusted) vs sdpa_nomask_tiled ---
    var ref_nm = sdpa_nomask[B, S, Hd, Dh](q, k, v, scale, ctx)
    var til_nm = sdpa_nomask_tiled[B, S, Hd, Dh](q, k, v, scale, ctx)
    var ref_nm_h = ref_nm.to_host(ctx)
    var til_nm_h = til_nm.to_host(ctx)
    var cr_nm = _cos_ratio(til_nm_h, ref_nm_h)
    print("NOMASK  S=", S, " Dh=", Dh, " (3 kv-blocks)")
    print("  cos(tiled, math) =", cr_nm[0], "  |tiled|/|math| =", cr_nm[1])

    # --- masked parity: random additive mask [B,H,S,S] ---
    var mn = B * Hd * S * S
    var mv = List[Float32]()
    for i in range(mn):
        mv.append(_pseudo(i, 7) * 2.0)   # nontrivial bias in ~[-2,2]
    var ms = List[Int]()
    ms.append(B); ms.append(Hd); ms.append(S); ms.append(S)
    var mask = Tensor.from_host(mv, ms^, STDtype.F32, ctx)
    var ref_m = sdpa[B, S, Hd, Dh](q, k, v, mask, scale, ctx)
    var til_m = sdpa_tiled[B, S, Hd, Dh](q, k, v, mask, scale, ctx)
    var ref_m_h = ref_m.to_host(ctx)
    var til_m_h = til_m.to_host(ctx)
    var cr_m = _cos_ratio(til_m_h, ref_m_h)
    print("MASKED  S=", S, " Dh=", Dh, " (random [B,H,S,S] mask)")
    print("  cos(tiled, math) =", cr_m[0], "  |tiled|/|math| =", cr_m[1])

    var pass1 = (cr_nm[0] >= 0.999 and cr_m[0] >= 0.999)
    if pass1:
        print("GATE1 CORRECTNESS PASS")
    else:
        print("GATE1 CORRECTNESS FAIL")

    # ───────────────────────── GATE 2: large-S memory ──────────────────────
    # math-mode scores would be B*H*S*S F32. Tiled keeps only O(S*Dh).
    comptime BL = 1
    comptime SL = 8192
    comptime HL = 2
    comptime DhL = 128
    var scaleL = Float32(1.0) / sqrt(Float32(DhL))
    var qnL = BL * SL * HL * DhL
    var qvL = List[Float32]()
    var kvL = List[Float32]()
    var vvL = List[Float32]()
    for i in range(qnL):
        qvL.append(_pseudo(i, 11) * 0.3)
        kvL.append(_pseudo(i, 12) * 0.3)
        vvL.append(_pseudo(i, 13) * 0.3)
    var qsL = List[Int]()
    qsL.append(BL); qsL.append(SL); qsL.append(HL); qsL.append(DhL)
    var qL = Tensor.from_host(qvL, qsL.copy(), STDtype.F32, ctx)
    var kL = Tensor.from_host(kvL, qsL.copy(), STDtype.F32, ctx)
    var vL = Tensor.from_host(vvL, qsL.copy(), STDtype.F32, ctx)
    print("LARGE-S run: B=", BL, " S=", SL, " H=", HL, " Dh=", DhL)
    var outL = sdpa_nomask_tiled[BL, SL, HL, DhL](qL, kL, vL, scaleL, ctx)
    var osh = outL.shape()
    var outL_h = outL.to_host(ctx)
    # finiteness + shape check on a sample
    var finite = True
    var sample = 0
    while sample < len(outL_h):
        var x = outL_h[sample]
        if not (x == x) or x > 1e30 or x < -1e30:
            finite = False
        sample += 4096
    print("  out shape = [", osh[0], ",", osh[1], ",", osh[2], ",", osh[3], "]")
    print("  out[0] =", outL_h[0], "  finite(sampled) =", finite)
    var pass2 = (
        finite and osh[0] == BL and osh[1] == SL
        and osh[2] == HL and osh[3] == DhL
    )
    if pass2:
        print("GATE2 LARGE-S NO-OOM PASS")
    else:
        print("GATE2 LARGE-S NO-OOM FAIL")

    if pass1 and pass2:
        print("SDPA_TILED ALL GATES PASS")
    else:
        print("SDPA_TILED GATES FAIL")
