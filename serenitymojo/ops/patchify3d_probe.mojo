# ops/patchify3d_probe.mojo — compile+run gate for patchify3d / unpatchify3d.
#
# Self-contained (no oracle .bin): builds a deterministic [C,F,H,W] input on host,
# runs `patchify3d` on GPU, and checks the unfold tensor against a host recompute
# of the wan22/cosmos (c-slowest, F-major) layout. Then runs `unpatchify3d` and
# checks the c-fastest inverse against a host recompute of wan22 einsum
# 'fhwpqrc->cfphqwr'. Exit 0 == pass. (The bf16 patch-embed vs torch conv3d
# equivalence + cos>=0.999 gate is in parity/patchify3d_parity.mojo.)
#
#   pixi run mojo run -I . serenitymojo/ops/patchify3d_probe.mojo

from std.math import abs as fabs
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.patchify3d import patchify3d, unpatchify3d


comptime C = 6
comptime F = 4
comptime H = 6
comptime W = 8
comptime PF = 1
comptime PH = 2
comptime PW = 2
comptime FO = F // PF
comptime HO = H // PH
comptime WO = W // PW
comptime N_PATCHES = FO * HO * WO
comptime PATCH_DIM = C * PF * PH * PW


def main() raises:
    var ctx = DeviceContext()
    print("=== patchify3d probe (C=", C, " F=", F, " H=", H, " W=", W,
          " patch=(", PF, ",", PH, ",", PW, ")) ===")

    # Deterministic input [C,F,H,W] (C-major contiguous), f32.
    var xh = List[Float32]()
    for i in range(C * F * H * W):
        xh.append(Float32((i * 37 + 11) % 101) - 50.0)
    var x = Tensor.from_host(xh.copy(), [C, F, H, W], STDtype.F32, ctx)

    # ── patchify3d ── expect c-slowest within-patch (c,pf,ph,pw), F-major tokens.
    var patches = patchify3d(x, PF, PH, PW, ctx)   # [N_PATCHES, PATCH_DIM] f32
    var ph_out = patches.to_host(ctx)
    if len(ph_out) != N_PATCHES * PATCH_DIM:
        raise Error("patchify3d: bad output size")
    var pmax = Float32(0.0)
    for fi in range(FO):
        for hi in range(HO):
            for wi in range(WO):
                var patch = fi * HO * WO + hi * WO + wi
                for ci in range(C):
                    for pfi in range(PF):
                        for phi in range(PH):
                            for pwi in range(PW):
                                var dst = ((ci * PF + pfi) * PH + phi) * PW + pwi
                                var sf = fi * PF + pfi
                                var sh = hi * PH + phi
                                var sw = wi * PW + pwi
                                var src = ((ci * F + sf) * H + sh) * W + sw
                                var d = fabs(ph_out[patch * PATCH_DIM + dst] - xh[src])
                                if d > pmax:
                                    pmax = d
    print("    patchify3d unfold max-abs vs host (c-slowest, F-major):", pmax)
    if pmax > Float32(1e-6):
        raise Error("patchify3d: FAIL unfold layout")

    # ── unpatchify3d ── feed `patches` as if head output; expect c-FASTEST read
    # (wan22 einsum 'fhwpqrc->cfphqwr'). Build expected [C,F,H,W] on host.
    var img = unpatchify3d(patches, C, F, H, W, PF, PH, PW, ctx)  # [C,F,H,W] f32
    var ih_out = img.to_host(ctx)
    if len(ih_out) != C * F * H * W:
        raise Error("unpatchify3d: bad output size")
    var umax = Float32(0.0)
    for ci in range(C):
        for sf in range(F):
            for sh in range(H):
                for sw in range(W):
                    var fi = sf // PF
                    var pfi = sf % PF
                    var hi = sh // PH
                    var phi = sh % PH
                    var wi = sw // PW
                    var pwi = sw % PW
                    var patch = (fi * HO + hi) * WO + wi
                    var src_ch = ((pfi * PH + phi) * PW + pwi) * C + ci
                    var expect = ph_out[patch * PATCH_DIM + src_ch]
                    var off = ((ci * F + sf) * H + sh) * W + sw
                    var d = fabs(ih_out[off] - expect)
                    if d > umax:
                        umax = d
    print("    unpatchify3d max-abs vs wan22 einsum 'fhwpqrc->cfphqwr':", umax)
    if umax > Float32(1e-6):
        raise Error("unpatchify3d: FAIL inverse layout")

    print("patchify3d_probe PASS")
