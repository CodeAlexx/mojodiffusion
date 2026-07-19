# ops/pixelshuffle_smoke.mojo — GPU smoke + parity for ops/pixelshuffle.mojo
# (LTX2 P-d2s). Self-contained: host index-permutation reference computed
# in-smoke (no external numpy oracle); F32 storage so the gate is BIT-EXACT.
#
# Gates (HARD RULE: numeric, on GPU, not compile-only):
#   1. pixel_unshuffle(pixel_shuffle(x)) == x   bit-exact (round-trip).
#   2. depth_to_space_3d matches a host index-permutation for each of the 4
#      distinct DECODER_BLOCKS strides — (2,2,2),(2,2,2),(2,1,1),(1,2,2)
#      (ltx2_vae.rs:59-67). The reduction param only sets the upstream conv's
#      output-channel count; d2s itself consumes only `stride`, so the 4
#      distinct strides cover all blocks. Tested with drop_first_temporal=False.
#   3. temporal-stride-2 frame-drop: depth_to_space_3d((2,2,2), drop=True)
#      equals the drop=False output with frame 0 removed, bit-exact.
#   4. pixel_shuffle / pixel_unshuffle each match the host permutation.
#
# Run: pixi run mojo build -I . -Xlinker -lm \
#        serenitymojo/ops/pixelshuffle_smoke.mojo -o /tmp/p_d2s_smoke && /tmp/p_d2s_smoke

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.pixelshuffle import (
    depth_to_space_3d, pixel_shuffle, pixel_unshuffle,
)


comptime F32 = STDtype.F32


def _ramp(n: Int) -> List[Float32]:
    """A distinct-valued ramp so any index mistake shows as a value mismatch."""
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(i))
    return out^


def _max_abs(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error(
            String("length mismatch: ") + String(len(a)) + " vs " + String(len(b))
        )
    var m = Float64(0.0)
    for i in range(len(a)):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


# Host reference for depth_to_space_3d: [B, Ctot, F,H,W] -> [B, C, FO, HO, WO]
# matching Rust reshape [B,C,p1,p2,p3,F,H,W] -> permute [0,1,5,2,6,3,7,4]
# -> reshape, optionally dropping output frame 0.
def _host_d2s(
    x: List[Float32], B: Int, Ctot: Int, F: Int, H: Int, W: Int,
    p1: Int, p2: Int, p3: Int, drop: Bool,
) -> List[Float32]:
    var prod = p1 * p2 * p3
    var C = Ctot // prod
    var FO = F * p1
    var d = 1 if drop else 0
    var FOk = FO - d
    var HO = H * p2
    var WO = W * p3
    var out = List[Float32]()
    out.resize(B * C * FOk * HO * WO, Float32(0.0))
    var oi = 0
    for b in range(B):
        for c in range(C):
            for fo_k in range(FOk):
                var fo = fo_k + d
                var f = fo // p1
                var i1 = fo % p1
                for ho in range(HO):
                    var h = ho // p2
                    var i2 = ho % p2
                    for wo in range(WO):
                        var w = wo // p3
                        var i3 = wo % p3
                        var ct = ((c * p1 + i1) * p2 + i2) * p3 + i3
                        var in_off = (((b * Ctot + ct) * F + f) * H + h) * W + w
                        out[oi] = x[in_off]
                        oi += 1
    return out^


# INDEPENDENT oracle for depth_to_space_3d. Does NOT mirror the kernel's
# output->input gather. Instead it literally simulates the Rust op as a SCATTER:
#   reshape x to 8D [b,c,p1,p2,p3,f,h,w] (row-major decode of flat input idx),
#   then for each source element compute its permuted destination
#   [b,c,f,p1,h,p2,w,p3] -> flat output [b,c,(f*p1+i1),(h*p2+i2),(w*p3+i3)],
#   optionally dropping output frame 0. Different traversal + scatter direction
#   means a shared-logic bug in the gather formulation cannot hide here.
def _oracle_d2s_scatter(
    x: List[Float32], B: Int, Ctot: Int, F: Int, H: Int, W: Int,
    p1: Int, p2: Int, p3: Int, drop: Bool,
) -> List[Float32]:
    var prod = p1 * p2 * p3
    var C = Ctot // prod
    var FO = F * p1
    var d = 1 if drop else 0
    var FOk = FO - d
    var HO = H * p2
    var WO = W * p3
    var out = List[Float32]()
    out.resize(B * C * FOk * HO * WO, Float32(-1.0))
    # iterate over the 8D reshaped source [b,c,p1,p2,p3,f,h,w] in row-major.
    var src = 0
    for b in range(B):
        for c in range(C):
            for i1 in range(p1):
                for i2 in range(p2):
                    for i3 in range(p3):
                        for f in range(F):
                            for h in range(H):
                                for w in range(W):
                                    var fo = f * p1 + i1
                                    var ho = h * p2 + i2
                                    var wo = w * p3 + i3
                                    # apply frame drop: skip / shift output frame.
                                    if not (drop and fo == 0):
                                        var fk = fo - d
                                        var dst = (((b * C + c) * FOk + fk) * HO + ho) * WO + wo
                                        out[dst] = x[src]
                                    src += 1
    return out^


def _host_unshuffle(
    x: List[Float32], B: Int, C: Int, H: Int, W: Int, r: Int
) -> List[Float32]:
    var Co = C * r * r
    var Ho = H // r
    var Wo = W // r
    var out = List[Float32]()
    out.resize(B * Co * Ho * Wo, Float32(0.0))
    var oi = 0
    for b in range(B):
        for c_out in range(Co):
            var c = c_out // (r * r)
            var rem = c_out % (r * r)
            var i = rem // r
            var j = rem % r
            for ho in range(Ho):
                var h = ho * r + i
                for wo in range(Wo):
                    var w = wo * r + j
                    var in_off = ((b * C + c) * H + h) * W + w
                    out[oi] = x[in_off]
                    oi += 1
    return out^


def _host_shuffle(
    x: List[Float32], B: Int, Cin: Int, H: Int, W: Int, r: Int
) -> List[Float32]:
    var C = Cin // (r * r)
    var Ho = H * r
    var Wo = W * r
    var out = List[Float32]()
    out.resize(B * C * Ho * Wo, Float32(0.0))
    var oi = 0
    for b in range(B):
        for c in range(C):
            for ho in range(Ho):
                var h = ho // r
                var i = ho % r
                for wo in range(Wo):
                    var w = wo // r
                    var j = wo % r
                    var c_in = (c * r + i) * r + j
                    var in_off = ((b * Cin + c_in) * H + h) * W + w
                    out[oi] = x[in_off]
                    oi += 1
    return out^


def _check(label: String, dev: List[Float32], refv: List[Float32]) raises -> Bool:
    var m = _max_abs(dev, refv)
    var ok = m == 0.0
    var tag = "PASS" if ok else "FAIL"
    print(
        label, "  max_abs=", m, "  n=", len(dev), "  [", tag, "]"
    )
    return ok


def _d2s_case(
    label: String, B: Int, C: Int, F: Int, H: Int, W: Int,
    p1: Int, p2: Int, p3: Int, drop: Bool, ctx: DeviceContext,
) raises -> Bool:
    var Ctot = C * p1 * p2 * p3
    var n = B * Ctot * F * H * W
    var host = _ramp(n)
    var xt = Tensor.from_host(host, [B, Ctot, F, H, W], F32, ctx)
    var yt = depth_to_space_3d(xt, p1, p2, p3, drop, ctx)
    var dev = yt.to_host(ctx)
    var refv = _host_d2s(host, B, Ctot, F, H, W, p1, p2, p3, drop)
    var oracle = _oracle_d2s_scatter(host, B, Ctot, F, H, W, p1, p2, p3, drop)
    # independent scatter oracle must agree with the gather host-ref AND device.
    var ok_ref = _check(label, dev, refv)
    var ok_oracle = _check(label + " [scatter-oracle]", dev, oracle)
    return ok_ref and ok_oracle


def main() raises:
    var ctx = DeviceContext()
    var all_pass = True

    print("=== P-d2s smoke (pixelshuffle.mojo) — F32 bit-exact gate ===")

    # ── Gate 1: round-trip pixel_unshuffle(pixel_shuffle(x)) == x ────────────
    # Start from the SHUFFLED-domain tensor x[B, C*r*r, H, W]; shuffle -> unshuffle
    # must recover it bit-for-bit.
    var rB = 2; var rC = 3; var rH = 4; var rW = 6; var rr = 2
    var rCin = rC * rr * rr
    var rn = rB * rCin * rH * rW
    var rhost = _ramp(rn)
    var rxt = Tensor.from_host(rhost, [rB, rCin, rH, rW], F32, ctx)
    var shuffled = pixel_shuffle(rxt, rr, ctx)          # [B, C, H*r, W*r]
    var roundtrip = pixel_unshuffle(shuffled, rr, ctx)  # [B, C*r*r, H, W]
    var rt_dev = roundtrip.to_host(ctx)
    var ok1 = _check("1 unshuffle(shuffle(x))==x ", rt_dev, rhost)
    all_pass = all_pass and ok1

    # ── Gate 4: pixel_shuffle / pixel_unshuffle vs host permutation ──────────
    var sh_ref = _host_shuffle(rhost, rB, rCin, rH, rW, rr)
    var ok4a = _check("4a pixel_shuffle host       ", shuffled.to_host(ctx), sh_ref)
    all_pass = all_pass and ok4a

    var uB = 2; var uC = 3; var uH = 8; var uW = 6; var ur = 2
    var un = uB * uC * uH * uW
    var uhost = _ramp(un)
    var uxt = Tensor.from_host(uhost, [uB, uC, uH, uW], F32, ctx)
    var unshuf = pixel_unshuffle(uxt, ur, ctx)
    var ush_ref = _host_unshuffle(uhost, uB, uC, uH, uW, ur)
    var ok4b = _check("4b pixel_unshuffle host     ", unshuf.to_host(ctx), ush_ref)
    all_pass = all_pass and ok4b

    # ── Gate 2: depth_to_space_3d for the 4 distinct DECODER_BLOCKS strides ──
    # B=1, C=2 (so Ctot = 2*prod) over a small [F,H,W] ramp. drop=False.
    var ok2a = _d2s_case("2a d2s stride (2,2,2)       ", 1, 2, 4, 8, 8, 2, 2, 2, False, ctx)
    var ok2b = _d2s_case("2b d2s stride (2,2,2) #2    ", 1, 2, 4, 8, 8, 2, 2, 2, False, ctx)
    var ok2c = _d2s_case("2c d2s stride (2,1,1)       ", 1, 2, 4, 8, 8, 2, 1, 1, False, ctx)
    var ok2d = _d2s_case("2d d2s stride (1,2,2)       ", 1, 2, 4, 8, 8, 1, 2, 2, False, ctx)
    all_pass = all_pass and ok2a and ok2b and ok2c and ok2d

    # Plan's headline shape [1,1024,4,8,8] with (2,2,2) r2 -> C=1024/(8/?) ...
    # 1024 = C * prod => with (2,2,2) prod=8 => C=128. Real decoder shape.
    var ok2e = _d2s_case("2e d2s [1,1024,4,8,8](2,2,2)", 1, 128, 4, 8, 8, 2, 2, 2, False, ctx)
    all_pass = all_pass and ok2e

    # ── Gate 3: temporal-stride-2 frame-drop ─────────────────────────────────
    # drop=True must equal drop=False output with frame 0 removed.
    var dB = 1; var dC = 2; var dF = 3; var dH = 4; var dW = 4
    var dp1 = 2; var dp2 = 2; var dp3 = 2
    var dCtot = dC * dp1 * dp2 * dp3
    var dn = dB * dCtot * dF * dH * dW
    var dhost = _ramp(dn)
    var dxt = Tensor.from_host(dhost, [dB, dCtot, dF, dH, dW], F32, ctx)
    var full = depth_to_space_3d(dxt, dp1, dp2, dp3, False, ctx)   # FO frames
    var dropped = depth_to_space_3d(dxt, dp1, dp2, dp3, True, ctx) # FO-1 frames
    var full_dev = full.to_host(ctx)
    var dropped_dev = dropped.to_host(ctx)
    # Build expected: full with output frame 0 removed.
    var fsh = full.shape()  # [B,C,FO,HO,WO]
    var FO = fsh[2]; var HO = fsh[3]; var WO = fsh[4]
    var frame_stride = HO * WO
    var per_c = FO * frame_stride
    var expected = List[Float32]()
    for b in range(dB):
        for c in range(dC):
            var base = (b * dC + c) * per_c
            # skip frame 0 -> start at base + frame_stride
            for k in range(frame_stride, per_c):
                expected.append(full_dev[base + k])
    var ok3 = _check("3 d2s frame-drop (2,2,2)    ", dropped_dev, expected)
    # also confirm the shape dropped exactly one frame
    var dsh = dropped.shape()
    var shape_ok = dsh[2] == FO - 1
    print("3 shape-drop FO:", FO, "->", dsh[2], "  [", "PASS" if shape_ok else "FAIL", "]")
    all_pass = all_pass and ok3 and shape_ok

    print("============================================================")
    if all_pass:
        print("ALL GATES PASS")
    else:
        print("SOME GATES FAILED")
        raise Error("p_d2s smoke: gate failure")
