# patchify3d_parity.mojo — GPU bf16 parity for the 3D video-DiT patch embed.
#
# Verifies serenitymojo/ops/patchify3d.mojo `patchify3d` + ops/linear.linear
# against the torch Conv3d(stride=kernel) bf16 oracle (patchify3d_oracle.py),
# which PROVES conv3d-patch-embed == unfold+linear (f32 cos=1.0, max_abs=0.0).
# Inputs (x, conv weight, conv bias) are read from the SAME .bin the oracle
# dumped so both sides see byte-identical f32, then run bf16 on GPU.
#   tokens = linear(patchify3d(x, pf,ph,pw), W.reshape([OUT, C*pf*ph*pw]), bias)
# Gate: cos >= 0.999 (bf16). Also reports magnitude ratio ||a||/||ref||.
# Additionally round-trips `unpatchify3d` on the oracle's unfold tensor to confir
# the c-fastest inverse layout reproduces the wan22 einsum 'fhwpqrc->cfphqwr'.
#
# Run the oracle first, then the probe:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/patchify3d_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/patchify3d_parity.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.ops.patchify3d import patchify3d, unpatchify3d
from serenitymojo.ops.linear import linear
from serenitymojo.ops.cast import cast_tensor


comptime DIR = "/home/alex/mojodiffusion/serenitymojo/ops/parity/"
# Must match patchify3d_oracle.py.
comptime C = 16
comptime F = 4
comptime H = 8
comptime W = 8
comptime PF = 1
comptime PH = 2
comptime PW = 2
comptime OUT = 64
comptime FO = F // PF
comptime HO = H // PH
comptime WO = W // PW
comptime N_PATCHES = FO * HO * WO
comptime PATCH_DIM = C * PF * PH * PW


def _read_f32_bin(name: String) raises -> List[Float32]:
    var path = String(DIR) + name
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path + " (run the oracle first)")
    var nbytes = file_size(fd)
    if nbytes <= 0:
        _ = sys_close(fd)
        raise Error(String("empty bin: ") + path)
    var buf = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, buf + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nfloats = done // 4
    var out = List[Float32]()
    var fptr = buf.bitcast[Float32]()
    for i in range(nfloats):
        out.append(fptr[i])
    buf.free()
    return out^


def _mag_ratio(got: List[Float32], refv: List[Float32]) -> Float64:
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(got)):
        na += Float64(got[i]) * Float64(got[i])
    for i in range(len(refv)):
        nb += Float64(refv[i]) * Float64(refv[i])
    return sqrt(na) / sqrt(nb) if nb > 0.0 else 0.0


def main() raises:
    var ctx = DeviceContext()
    print("=== patchify3d patch-embed parity (bf16 GPU) ===")
    print("    C=", C, " F=", F, " H=", H, " W=", W,
          " patch=(", PF, ",", PH, ",", PW, ") OUT=", OUT,
          " n_patches=", N_PATCHES, " patch_dim=", PATCH_DIM)

    var x_h = _read_f32_bin("patchify3d_x.bin")
    var w_h = _read_f32_bin("patchify3d_w.bin")
    var b_h = _read_f32_bin("patchify3d_b.bin")
    var refv = _read_f32_bin("patchify3d_ref.bin")
    var unfold_ref = _read_f32_bin("patchify3d_unfold.bin")
    if (len(x_h) != C * F * H * W or len(w_h) != OUT * PATCH_DIM
            or len(b_h) != OUT or len(refv) != N_PATCHES * OUT
            or len(unfold_ref) != N_PATCHES * PATCH_DIM):
        raise Error("patchify3d_parity: input/ref size mismatch with oracle")

    # ── Build bf16 device tensors from identical f32 bytes ──
    var x_f32 = Tensor.from_host(x_h.copy(), [C, F, H, W], STDtype.F32, ctx)
    var x_bf16 = cast_tensor(x_f32, STDtype.BF16, ctx)
    # weight flat [OUT, PATCH_DIM] (== conv kernel [OUT,C,pf,ph,pw] reshaped, no T)
    var w_f32 = Tensor.from_host(w_h.copy(), [OUT, PATCH_DIM], STDtype.F32, ctx)
    var w_bf16 = cast_tensor(w_f32, STDtype.BF16, ctx)
    var b_f32 = Tensor.from_host(b_h.copy(), [OUT], STDtype.F32, ctx)
    var b_bf16 = cast_tensor(b_f32, STDtype.BF16, ctx)

    # ── unfold then patch-embed linear ──
    var patches = patchify3d(x_bf16, PF, PH, PW, ctx)   # [N_PATCHES, PATCH_DIM] bf16

    # Sanity: the Mojo unfold must equal the oracle's unfold tensor exactly (f32).
    var patches_f32_chk = cast_tensor(patches, STDtype.F32, ctx)
    var pchk = patches_f32_chk.to_host(ctx)
    var unfold_max = Float32(0.0)
    for i in range(len(pchk)):
        var d = pchk[i] - unfold_ref[i]
        if d < 0:
            d = -d
        if d > unfold_max:
            unfold_max = d
    print("    unfold vs oracle max-abs (bf16-rounded):", unfold_max)

    var tokens = linear(patches, w_bf16, Optional[Tensor](b_bf16^), ctx)
    var tokens_f32 = cast_tensor(tokens, STDtype.F32, ctx)

    var harness = ParityHarness(0.999)
    var r = harness.compare(tokens_f32, refv, ctx)
    print("    patchify3d+linear vs conv3d(stride=k) bf16:", r)
    var got = tokens_f32.to_host(ctx)
    print("    magRatio (||a||/||ref||):", _mag_ratio(got, refv))

    # ── unpatchify3d round-trip check (layout-only, f32 exactness) ──
    # Feed the oracle unfold tensor (c-slowest [N_PATCHES, PATCH_DIM]) as if it
    # were head output, then unpatchify3d (c-fastest read) and re-patchify; the
    # composition is NOT identity (asymmetric layout, by design), so we instead
    # verify unpatchify3d reproduces a hand-built reference: build the expected
    # [C,F,H,W] from unfold_ref using the wan22 einsum src order, compare exact.
    var unfold_t = Tensor.from_host(unfold_ref.copy(),
                                    [N_PATCHES, PATCH_DIM], STDtype.F32, ctx)
    var img = unpatchify3d(unfold_t, C, F, H, W, PF, PH, PW, ctx)  # [C,F,H,W] f32
    var img_h = img.to_host(ctx)
    # expected: for each (ci, sf, sh, sw): fi=sf//PF.. patch.. src_ch (c-fastest)
    var up_max = Float32(0.0)
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
                    var expect = unfold_ref[patch * PATCH_DIM + src_ch]
                    var off = ((ci * F + sf) * H + sh) * W + sw
                    var d = img_h[off] - expect
                    if d < 0:
                        d = -d
                    if d > up_max:
                        up_max = d
    print("    unpatchify3d vs wan22 einsum 'fhwpqrc->cfphqwr' max-abs:", up_max)

    if not r.passed:
        raise Error("patchify3d_parity gate FAILED (cos<0.999)")
    if up_max > Float32(1e-5):
        raise Error("patchify3d_parity: unpatchify3d layout mismatch")
    print("PASS: patchify3d+linear bf16 cos>=0.999 AND unpatchify3d layout exact")
