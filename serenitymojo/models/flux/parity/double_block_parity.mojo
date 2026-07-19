# serenitymojo/models/flux/parity/double_block_parity.mojo
#
# PARITY GATE for the Flux DOUBLE-STREAM DiT block training unit
# (models/flux/block.mojo). Loads the EXACT inputs + torch-autograd reference
# grads dumped by block_oracle.py (gen_double), runs the packaged
# double_block_forward + double_block_backward, and compares forward outputs,
# d_img, d_txt, every trainable weight+BIAS grad, and the modulation-vector
# grads at cos >= 0.999.
#
# REAL Flux dims: hidden D = 3072, H = 24, Dh = 128. Small N_IMG/N_TXT/FMLP keep
# the torch oracle fast. NON-DEGENERATE sinusoidal/random inputs + a 3-axis Flux
# RoPE table (asserted non-degenerate in the oracle).
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/flux/parity/block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/flux/parity/double_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, ModVecs,
    double_block_forward, double_block_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/"

# dims MUST match block_oracle.py gen_double()
comptime H = 24
comptime Dh = 128
comptime D = H * Dh        # 3072
comptime N_IMG = 4
comptime N_TXT = 3
comptime FMLP = 32
comptime EPS = Float32(1e-06)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read_bin_f32(REF_DIR + name + ".bin")


def _load_stream(prefix: String, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _in("d_in_" + prefix + "_wqkv"), _in("d_in_" + prefix + "_bqkv"),
        _in("d_in_" + prefix + "_wproj"), _in("d_in_" + prefix + "_bproj"),
        _in("d_in_" + prefix + "_wmlp0"), _in("d_in_" + prefix + "_bmlp0"),
        _in("d_in_" + prefix + "_wmlp2"), _in("d_in_" + prefix + "_bmlp2"),
        _in("d_in_" + prefix + "_q_norm"), _in("d_in_" + prefix + "_k_norm"),
        D, FMLP, Dh, ctx,
    )


def _load_mod(prefix: String) raises -> ModVecs:
    return ModVecs(
        _in("d_in_" + prefix + "_shift1"), _in("d_in_" + prefix + "_scale1"),
        _in("d_in_" + prefix + "_gate1"),
        _in("d_in_" + prefix + "_shift2"), _in("d_in_" + prefix + "_scale2"),
        _in("d_in_" + prefix + "_gate2"),
    )


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== flux double_block_parity (Flux double-stream block vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT, " FMLP=", FMLP)

    var img = _in("d_in_img")
    var txt = _in("d_in_txt")
    var iw = _load_stream("iw", ctx)
    var tw = _load_stream("tw", ctx)
    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var cos_h = _in("d_in_cos")
    var sin_h = _in("d_in_sin")
    var cos = Tensor.from_host(cos_h, [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var w = DoubleBlockWeights(iw^, tw^)

    var fwd = double_block_forward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        img.copy(), txt.copy(), w, im, tm, cos, sin,
        D, FMLP, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward outputs vs torch ----")
    _check(harness, "img_out", fwd.img_out, _in("d_ref_img_out"), allok)
    _check(harness, "txt_out", fwd.txt_out, _in("d_ref_txt_out"), allok)

    var d_img = _in("d_in_d_img")
    var d_txt = _in("d_in_d_txt")
    var g = double_block_backward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        d_img, d_txt, w, im, tm, fwd.saved, cos, sin,
        D, FMLP, EPS, ctx,
    )

    print("")
    print("---- input grads vs torch ----")
    _check(harness, "d_img", g.img.d_x, _in("d_ref_d_img"), allok)
    _check(harness, "d_txt", g.txt.d_x, _in("d_ref_d_txt"), allok)

    print("")
    print("---- IMG trainable weight+bias grads vs torch ----")
    _check(harness, "img d_wqkv ", g.img.d_wqkv, _in("d_ref_im_d_wqkv"), allok)
    _check(harness, "img d_bqkv ", g.img.d_bqkv, _in("d_ref_im_d_bqkv"), allok)
    _check(harness, "img d_wproj", g.img.d_wproj, _in("d_ref_im_d_wproj"), allok)
    _check(harness, "img d_bproj", g.img.d_bproj, _in("d_ref_im_d_bproj"), allok)
    _check(harness, "img d_wmlp0", g.img.d_wmlp0, _in("d_ref_im_d_wmlp0"), allok)
    _check(harness, "img d_bmlp0", g.img.d_bmlp0, _in("d_ref_im_d_bmlp0"), allok)
    _check(harness, "img d_wmlp2", g.img.d_wmlp2, _in("d_ref_im_d_wmlp2"), allok)
    _check(harness, "img d_bmlp2", g.img.d_bmlp2, _in("d_ref_im_d_bmlp2"), allok)
    _check(harness, "img d_qnorm", g.img.d_q_norm, _in("d_ref_im_d_q_norm"), allok)
    _check(harness, "img d_knorm", g.img.d_k_norm, _in("d_ref_im_d_k_norm"), allok)

    print("")
    print("---- IMG modulation-vector grads vs torch ----")
    _check(harness, "img d_shift1", g.img.d_shift1, _in("d_ref_im_d_shift1"), allok)
    _check(harness, "img d_scale1", g.img.d_scale1, _in("d_ref_im_d_scale1"), allok)
    _check(harness, "img d_gate1 ", g.img.d_gate1, _in("d_ref_im_d_gate1"), allok)
    _check(harness, "img d_shift2", g.img.d_shift2, _in("d_ref_im_d_shift2"), allok)
    _check(harness, "img d_scale2", g.img.d_scale2, _in("d_ref_im_d_scale2"), allok)
    _check(harness, "img d_gate2 ", g.img.d_gate2, _in("d_ref_im_d_gate2"), allok)

    print("")
    print("---- TXT trainable weight+bias grads vs torch ----")
    _check(harness, "txt d_wqkv ", g.txt.d_wqkv, _in("d_ref_tm_d_wqkv"), allok)
    _check(harness, "txt d_bqkv ", g.txt.d_bqkv, _in("d_ref_tm_d_bqkv"), allok)
    _check(harness, "txt d_wproj", g.txt.d_wproj, _in("d_ref_tm_d_wproj"), allok)
    _check(harness, "txt d_bproj", g.txt.d_bproj, _in("d_ref_tm_d_bproj"), allok)
    _check(harness, "txt d_wmlp0", g.txt.d_wmlp0, _in("d_ref_tm_d_wmlp0"), allok)
    _check(harness, "txt d_bmlp0", g.txt.d_bmlp0, _in("d_ref_tm_d_bmlp0"), allok)
    _check(harness, "txt d_wmlp2", g.txt.d_wmlp2, _in("d_ref_tm_d_wmlp2"), allok)
    _check(harness, "txt d_bmlp2", g.txt.d_bmlp2, _in("d_ref_tm_d_bmlp2"), allok)
    _check(harness, "txt d_qnorm", g.txt.d_q_norm, _in("d_ref_tm_d_q_norm"), allok)
    _check(harness, "txt d_knorm", g.txt.d_k_norm, _in("d_ref_tm_d_k_norm"), allok)

    print("")
    print("---- TXT modulation-vector grads vs torch ----")
    _check(harness, "txt d_shift1", g.txt.d_shift1, _in("d_ref_tm_d_shift1"), allok)
    _check(harness, "txt d_scale1", g.txt.d_scale1, _in("d_ref_tm_d_scale1"), allok)
    _check(harness, "txt d_gate1 ", g.txt.d_gate1, _in("d_ref_tm_d_gate1"), allok)
    _check(harness, "txt d_shift2", g.txt.d_shift2, _in("d_ref_tm_d_shift2"), allok)
    _check(harness, "txt d_scale2", g.txt.d_scale2, _in("d_ref_tm_d_scale2"), allok)
    _check(harness, "txt d_gate2 ", g.txt.d_gate2, _in("d_ref_tm_d_gate2"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Flux double-stream block fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
