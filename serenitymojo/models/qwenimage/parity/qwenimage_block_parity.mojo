# serenitymojo/models/qwenimage/parity/qwenimage_block_parity.mojo
#
# PARITY GATE for the Qwen-Image MMDiT DOUBLE-STREAM block training unit
# (models/qwenimage/qwenimage_block.mojo). Loads the EXACT inputs + torch-autograd
# reference grads dumped by qwenimage_block_oracle.py, runs the packaged
# double_block_forward + double_block_backward, and compares d_img, d_txt, every
# trainable weight/bias grad, the QK-norm grads, and the modulation-vector grads
# at cos >= 0.999.
#
# REAL Qwen-Image head count H = 24. NON-DEGENERATE sinusoidal/random inputs.
#
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/qwenimage/parity/qwenimage_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/qwenimage/parity/qwenimage_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.qwenimage.qwenimage_block import (
    StreamWeights, DoubleBlockWeights, ModVecs,
    double_block_forward, double_block_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/qwenimage/parity/"

# dims MUST match qwenimage_block_oracle.py
comptime H = 24
comptime Dh = 16
comptime D = H * Dh        # 384
comptime N_IMG = 4
comptime N_TXT = 3
comptime F = 40
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
        _in("in_" + prefix + "_wq"), _in("in_" + prefix + "_wk"), _in("in_" + prefix + "_wv"),
        _in("in_" + prefix + "_bq"), _in("in_" + prefix + "_bk"), _in("in_" + prefix + "_bv"),
        _in("in_" + prefix + "_wout"), _in("in_" + prefix + "_bout"),
        _in("in_" + prefix + "_wup"), _in("in_" + prefix + "_bup"),
        _in("in_" + prefix + "_wdn"), _in("in_" + prefix + "_bdn"),
        _in("in_" + prefix + "_q_norm"), _in("in_" + prefix + "_k_norm"),
        D, F, Dh, ctx,
    )


def _load_mod(prefix: String) raises -> ModVecs:
    return ModVecs(
        _in("in_" + prefix + "_shift1"), _in("in_" + prefix + "_scale1"),
        _in("in_" + prefix + "_gate1"),
        _in("in_" + prefix + "_shift2"), _in("in_" + prefix + "_scale2"),
        _in("in_" + prefix + "_gate2"),
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
    print("==== qwenimage_block_parity (Qwen-Image double-stream block vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT, " F=", F)

    var img = _in("in_img")
    var txt = _in("in_txt")
    var iw = _load_stream("iw", ctx)
    var tw = _load_stream("tw", ctx)
    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var cos_h = _in("in_cos")
    var sin_h = _in("in_sin")
    var cos = Tensor.from_host(cos_h, [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var w = DoubleBlockWeights(iw^, tw^)

    var fwd = double_block_forward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        img.copy(), txt.copy(), w, im, tm, cos, sin,
        D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward outputs vs torch ----")
    _check(harness, "img_out", fwd.img_out, _in("ref_img_out"), allok)
    _check(harness, "txt_out", fwd.txt_out, _in("ref_txt_out"), allok)

    var d_img = _in("in_d_img")
    var d_txt = _in("in_d_txt")
    var g = double_block_backward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        d_img, d_txt, w, im, tm, fwd.saved, cos, sin,
        D, F, EPS, ctx,
    )

    print("")
    print("---- input grads vs torch ----")
    _check(harness, "d_img", g.img.d_x, _in("ref_d_img"), allok)
    _check(harness, "d_txt", g.txt.d_x, _in("ref_d_txt"), allok)

    print("")
    print("---- IMG trainable weight+bias grads vs torch ----")
    _check(harness, "img d_wq  ", g.img.d_wq, _in("ref_img_d_wq"), allok)
    _check(harness, "img d_wk  ", g.img.d_wk, _in("ref_img_d_wk"), allok)
    _check(harness, "img d_wv  ", g.img.d_wv, _in("ref_img_d_wv"), allok)
    _check(harness, "img d_bq  ", g.img.d_bq, _in("ref_img_d_bq"), allok)
    _check(harness, "img d_bk  ", g.img.d_bk, _in("ref_img_d_bk"), allok)
    _check(harness, "img d_bv  ", g.img.d_bv, _in("ref_img_d_bv"), allok)
    _check(harness, "img d_wout", g.img.d_wout, _in("ref_img_d_wout"), allok)
    _check(harness, "img d_bout", g.img.d_bout, _in("ref_img_d_bout"), allok)
    _check(harness, "img d_wup ", g.img.d_wup, _in("ref_img_d_wup"), allok)
    _check(harness, "img d_bup ", g.img.d_bup, _in("ref_img_d_bup"), allok)
    _check(harness, "img d_wdn ", g.img.d_wdn, _in("ref_img_d_wdn"), allok)
    _check(harness, "img d_bdn ", g.img.d_bdn, _in("ref_img_d_bdn"), allok)
    _check(harness, "img d_qnorm", g.img.d_q_norm, _in("ref_img_d_q_norm"), allok)
    _check(harness, "img d_knorm", g.img.d_k_norm, _in("ref_img_d_k_norm"), allok)

    print("")
    print("---- IMG modulation-vector grads vs torch ----")
    _check(harness, "img d_shift1", g.img.d_shift1, _in("ref_img_d_shift1"), allok)
    _check(harness, "img d_scale1", g.img.d_scale1, _in("ref_img_d_scale1"), allok)
    _check(harness, "img d_gate1 ", g.img.d_gate1, _in("ref_img_d_gate1"), allok)
    _check(harness, "img d_shift2", g.img.d_shift2, _in("ref_img_d_shift2"), allok)
    _check(harness, "img d_scale2", g.img.d_scale2, _in("ref_img_d_scale2"), allok)
    _check(harness, "img d_gate2 ", g.img.d_gate2, _in("ref_img_d_gate2"), allok)

    print("")
    print("---- TXT trainable weight+bias grads vs torch ----")
    _check(harness, "txt d_wq  ", g.txt.d_wq, _in("ref_txt_d_wq"), allok)
    _check(harness, "txt d_wk  ", g.txt.d_wk, _in("ref_txt_d_wk"), allok)
    _check(harness, "txt d_wv  ", g.txt.d_wv, _in("ref_txt_d_wv"), allok)
    _check(harness, "txt d_bq  ", g.txt.d_bq, _in("ref_txt_d_bq"), allok)
    _check(harness, "txt d_bk  ", g.txt.d_bk, _in("ref_txt_d_bk"), allok)
    _check(harness, "txt d_bv  ", g.txt.d_bv, _in("ref_txt_d_bv"), allok)
    _check(harness, "txt d_wout", g.txt.d_wout, _in("ref_txt_d_wout"), allok)
    _check(harness, "txt d_bout", g.txt.d_bout, _in("ref_txt_d_bout"), allok)
    _check(harness, "txt d_wup ", g.txt.d_wup, _in("ref_txt_d_wup"), allok)
    _check(harness, "txt d_bup ", g.txt.d_bup, _in("ref_txt_d_bup"), allok)
    _check(harness, "txt d_wdn ", g.txt.d_wdn, _in("ref_txt_d_wdn"), allok)
    _check(harness, "txt d_bdn ", g.txt.d_bdn, _in("ref_txt_d_bdn"), allok)
    _check(harness, "txt d_qnorm", g.txt.d_q_norm, _in("ref_txt_d_q_norm"), allok)
    _check(harness, "txt d_knorm", g.txt.d_k_norm, _in("ref_txt_d_k_norm"), allok)

    print("")
    print("---- TXT modulation-vector grads vs torch ----")
    _check(harness, "txt d_shift1", g.txt.d_shift1, _in("ref_txt_d_shift1"), allok)
    _check(harness, "txt d_scale1", g.txt.d_scale1, _in("ref_txt_d_scale1"), allok)
    _check(harness, "txt d_gate1 ", g.txt.d_gate1, _in("ref_txt_d_gate1"), allok)
    _check(harness, "txt d_shift2", g.txt.d_shift2, _in("ref_txt_d_shift2"), allok)
    _check(harness, "txt d_scale2", g.txt.d_scale2, _in("ref_txt_d_scale2"), allok)
    _check(harness, "txt d_gate2 ", g.txt.d_gate2, _in("ref_txt_d_gate2"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Qwen-Image double-stream block fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
