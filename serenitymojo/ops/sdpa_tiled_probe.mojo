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
    sdpa, sdpa_nomask, sdpa_tiled, sdpa_nomask_tiled,
    sdpa_cross_nomask, sdpa_cross_masked,
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
    var mask = Tensor.from_host(mv, ms.copy(), STDtype.F32, ctx)
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

    # --- rectangular cross-attn parity: direct Sq x Skv vs square pad+mask ---
    comptime Skv = 1024
    var qn_kv = B * Skv * Hd * Dh
    var kv_short = List[Float32]()
    var vv_short = List[Float32]()
    for i in range(qn_kv):
        kv_short.append(_pseudo(i, 17) * 0.5)
        vv_short.append(_pseudo(i, 19) * 0.5)
    var kvs = List[Int]()
    kvs.append(B); kvs.append(Skv); kvs.append(Hd); kvs.append(Dh)
    var k_short = Tensor.from_host(kv_short.copy(), kvs.copy(), STDtype.F32, ctx)
    var v_short = Tensor.from_host(vv_short.copy(), kvs^, STDtype.F32, ctx)

    var k_pad_v = List[Float32]()
    var v_pad_v = List[Float32]()
    for idx in range(qn):
        var d = idx % Dh
        var t = idx // Dh
        var h = t % Hd
        var s = (t // Hd) % S
        if s < Skv:
            var short_idx = ((s * Hd + h) * Dh) + d
            k_pad_v.append(kv_short[short_idx])
            v_pad_v.append(vv_short[short_idx])
        else:
            k_pad_v.append(Float32(0.0))
            v_pad_v.append(Float32(0.0))
    var k_pad = Tensor.from_host(k_pad_v, qs.copy(), STDtype.F32, ctx)
    var v_pad = Tensor.from_host(v_pad_v, qs.copy(), STDtype.F32, ctx)
    var cross_mask_v = List[Float32]()
    for idx in range(mn):
        var j = idx % S
        cross_mask_v.append(Float32(0.0) if j < Skv else Float32(-1.0e30))
    var cross_mask = Tensor.from_host(cross_mask_v, ms.copy(), STDtype.F32, ctx)
    var ref_cross = sdpa[B, S, Hd, Dh](q, k_pad, v_pad, cross_mask, scale, ctx)
    var rect_cross = sdpa_cross_nomask[B, S, Skv, Hd, Dh](
        q, k_short, v_short, scale, ctx,
    )
    var cr_cross = _cos_ratio(rect_cross.to_host(ctx), ref_cross.to_host(ctx))
    print("CROSS   Sq=", S, " Skv=", Skv, " Dh=", Dh)
    print("  cos(rect, padded-mask) =", cr_cross[0],
          "  |rect|/|padded| =", cr_cross[1])
    var pass1b = cr_cross[0] >= 0.999
    if pass1b:
        print("GATE1B RECTANGULAR CROSS PASS")
    else:
        print("GATE1B RECTANGULAR CROSS FAIL")

    # --- masked rectangular cross-attn parity: direct [Sq,Skv] mask vs padded square oracle ---
    var rect_mask_v = List[Float32]()
    for idx in range(B * Hd * S * Skv):
        rect_mask_v.append(_pseudo(idx, 23) * 1.5)
    var rms = List[Int]()
    rms.append(B); rms.append(Hd); rms.append(S); rms.append(Skv)
    var rect_mask = Tensor.from_host(rect_mask_v.copy(), rms^, STDtype.F32, ctx)
    var padded_masked_v = List[Float32]()
    for idx in range(mn):
        var j = idx % S
        var t = idx // S
        var irow = t % S
        var hrow = (t // S) % Hd
        if j < Skv:
            var ridx = ((hrow * S + irow) * Skv) + j
            padded_masked_v.append(rect_mask_v[ridx])
        else:
            padded_masked_v.append(Float32(-1.0e30))
    var padded_masked = Tensor.from_host(padded_masked_v, ms.copy(), STDtype.F32, ctx)
    var ref_cross_masked = sdpa[B, S, Hd, Dh](q, k_pad, v_pad, padded_masked, scale, ctx)
    var rect_cross_masked = sdpa_cross_masked[B, S, Skv, Hd, Dh](
        q, k_short, v_short, rect_mask, scale, ctx,
    )
    var cr_cross_masked = _cos_ratio(
        rect_cross_masked.to_host(ctx), ref_cross_masked.to_host(ctx),
    )
    print("CROSS-M Sq=", S, " Skv=", Skv, " Dh=", Dh)
    print("  cos(rect-masked, padded-mask) =", cr_cross_masked[0],
          "  |rect-masked|/|padded| =", cr_cross_masked[1])
    var pass1c = cr_cross_masked[0] >= 0.999
    if pass1c:
        print("GATE1C RECTANGULAR MASKED CROSS PASS")
    else:
        print("GATE1C RECTANGULAR MASKED CROSS FAIL")

    # --- BF16 mask storage gate: same paths, no F32 mask boundary ---
    comptime BB = 1
    comptime SB = 64
    comptime HB = 2
    comptime DhB = 64
    comptime SkvB = 48
    var scaleB = Float32(1.0) / sqrt(Float32(DhB))
    var qnB = BB * SB * HB * DhB
    var qvB = List[Float32]()
    var kvB = List[Float32]()
    var vvB = List[Float32]()
    for i in range(qnB):
        qvB.append(_pseudo(i, 31) * 0.5)
        kvB.append(_pseudo(i, 32) * 0.5)
        vvB.append(_pseudo(i, 33) * 0.5)
    var qsB = List[Int]()
    qsB.append(BB); qsB.append(SB); qsB.append(HB); qsB.append(DhB)
    var qB = Tensor.from_host(qvB, qsB.copy(), STDtype.BF16, ctx)
    var kB = Tensor.from_host(kvB, qsB.copy(), STDtype.BF16, ctx)
    var vB = Tensor.from_host(vvB, qsB.copy(), STDtype.BF16, ctx)
    var mnB = BB * HB * SB * SB
    var mvB = List[Float32]()
    for i in range(mnB):
        mvB.append(_pseudo(i, 34) * 1.25)
    var msB = List[Int]()
    msB.append(BB); msB.append(HB); msB.append(SB); msB.append(SB)
    var maskB = Tensor.from_host(mvB, msB.copy(), STDtype.BF16, ctx)
    var ref_m_bf16 = sdpa[BB, SB, HB, DhB](qB, kB, vB, maskB, scaleB, ctx)
    var til_m_bf16 = sdpa_tiled[BB, SB, HB, DhB](qB, kB, vB, maskB, scaleB, ctx)
    if til_m_bf16.dtype() != STDtype.BF16:
        raise Error("sdpa_tiled BF16 masked output returned non-BF16 storage")
    var cr_m_bf16 = _cos_ratio(til_m_bf16.to_host(ctx), ref_m_bf16.to_host(ctx))
    print("MASKED-BF16 S=", SB, " Dh=", DhB)
    print("  cos(tiled, math) =", cr_m_bf16[0],
          "  |tiled|/|math| =", cr_m_bf16[1])

    var qnSkvB = BB * SkvB * HB * DhB
    var kvShortB = List[Float32]()
    var vvShortB = List[Float32]()
    for i in range(qnSkvB):
        kvShortB.append(_pseudo(i, 35) * 0.5)
        vvShortB.append(_pseudo(i, 36) * 0.5)
    var kvsB = List[Int]()
    kvsB.append(BB); kvsB.append(SkvB); kvsB.append(HB); kvsB.append(DhB)
    var kShortB = Tensor.from_host(kvShortB.copy(), kvsB.copy(), STDtype.BF16, ctx)
    var vShortB = Tensor.from_host(vvShortB.copy(), kvsB^, STDtype.BF16, ctx)
    var rectMaskBv = List[Float32]()
    for idx in range(BB * HB * SB * SkvB):
        rectMaskBv.append(_pseudo(idx, 37) * 1.25)
    var rmsB = List[Int]()
    rmsB.append(BB); rmsB.append(HB); rmsB.append(SB); rmsB.append(SkvB)
    var rectMaskB = Tensor.from_host(rectMaskBv.copy(), rmsB^, STDtype.BF16, ctx)
    var kPadBv = List[Float32]()
    var vPadBv = List[Float32]()
    for idx in range(qnB):
        var d = idx % DhB
        var t = idx // DhB
        var h = t % HB
        var s = (t // HB) % SB
        if s < SkvB:
            var short_idx = ((s * HB + h) * DhB) + d
            kPadBv.append(kvShortB[short_idx])
            vPadBv.append(vvShortB[short_idx])
        else:
            kPadBv.append(Float32(0.0))
            vPadBv.append(Float32(0.0))
    var kPadB = Tensor.from_host(kPadBv, qsB.copy(), STDtype.BF16, ctx)
    var vPadB = Tensor.from_host(vPadBv, qsB.copy(), STDtype.BF16, ctx)
    var paddedMaskBv = List[Float32]()
    for idx in range(mnB):
        var j = idx % SB
        var t = idx // SB
        var irow = t % SB
        var hrow = (t // SB) % HB
        if j < SkvB:
            var ridx = ((hrow * SB + irow) * SkvB) + j
            paddedMaskBv.append(rectMaskBv[ridx])
        else:
            paddedMaskBv.append(Float32(-1.0e30))
    var paddedMaskB = Tensor.from_host(paddedMaskBv, msB.copy(), STDtype.BF16, ctx)
    var ref_cross_bf16 = sdpa[BB, SB, HB, DhB](
        qB, kPadB, vPadB, paddedMaskB, scaleB, ctx,
    )
    var rect_cross_bf16 = sdpa_cross_masked[BB, SB, SkvB, HB, DhB](
        qB, kShortB, vShortB, rectMaskB, scaleB, ctx,
    )
    if rect_cross_bf16.dtype() != STDtype.BF16:
        raise Error("sdpa_cross_masked BF16 output returned non-BF16 storage")
    var cr_cross_bf16 = _cos_ratio(
        rect_cross_bf16.to_host(ctx), ref_cross_bf16.to_host(ctx),
    )
    print("CROSS-M-BF16 Sq=", SB, " Skv=", SkvB, " Dh=", DhB)
    print("  cos(rect-masked, padded-mask) =", cr_cross_bf16[0],
          "  |rect-masked|/|padded| =", cr_cross_bf16[1])
    var pass1d = cr_m_bf16[0] >= 0.999 and cr_cross_bf16[0] >= 0.999
    if pass1d:
        print("GATE1D BF16 MASK STORAGE PASS")
    else:
        print("GATE1D BF16 MASK STORAGE FAIL")

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

    if pass1 and pass1b and pass1c and pass1d and pass2:
        print("SDPA_TILED ALL GATES PASS")
    else:
        print("SDPA_TILED GATES FAIL")
        raise Error("sdpa_tiled_probe failed")
