# serenitymojo/models/klein/parity/double_block_parity.mojo
#
# PARITY GATE for the Klein DOUBLE-STREAM DiT block training unit
# (models/klein/double_block.mojo). Loads the EXACT inputs + torch-autograd
# reference grads dumped by double_block_oracle.py, runs the packaged
# double_block_forward + double_block_backward, and compares d_img, d_txt, every
# trainable weight grad, and the modulation-vector grads at cos >= 0.999.
#
# REAL Klein head count H = 32 (the dim that PASSES sdpa backward). Small N/Dh to
# keep the torch oracle fast. NON-DEGENERATE sinusoidal/random inputs (no modular
# fills that alias and fake zero grads).
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/double_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/klein/parity/double_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.double_block import (
    StreamWeights, DoubleBlockWeights, ModVecs,
    double_block_forward, double_block_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"

# dims MUST match double_block_oracle.py
comptime H = 32
comptime Dh = 16
comptime D = H * Dh        # 512
comptime N_IMG = 4
comptime N_TXT = 2
comptime F = 24
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
        _in("in_" + prefix + "_wqkv"), _in("in_" + prefix + "_wproj"),
        _in("in_" + prefix + "_wgu"), _in("in_" + prefix + "_wd"),
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
    print("==== double_block_parity (Klein double-stream block vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT, " F=", F)

    # ── load inputs (byte-identical to the oracle) ──
    var img = _in("in_img")
    var txt = _in("in_txt")
    var iw = _load_stream("iw", ctx)
    var tw = _load_stream("tw", ctx)
    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var cos_h = _in("in_cos")
    var sin_h = _in("in_sin")
    # Resident rope tables: upload ONCE, pass by borrow (matches the trainer).
    var cos = Tensor.from_host(cos_h, [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var w = DoubleBlockWeights(iw^, tw^)

    # ── forward ──
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

    # ── backward ──
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
    print("---- IMG trainable weight grads vs torch ----")
    _check(harness, "img d_wqkv ", g.img.d_wqkv, _in("ref_img_d_wqkv"), allok)
    _check(harness, "img d_wproj", g.img.d_wproj, _in("ref_img_d_wproj"), allok)
    _check(harness, "img d_wgu  ", g.img.d_wgu, _in("ref_img_d_wgu"), allok)
    _check(harness, "img d_wd   ", g.img.d_wd, _in("ref_img_d_wd"), allok)
    _check(harness, "img d_qnorm", g.img.d_q_norm, _in("ref_img_d_qnorm"), allok)
    _check(harness, "img d_knorm", g.img.d_k_norm, _in("ref_img_d_knorm"), allok)

    print("")
    print("---- IMG modulation-vector grads vs torch ----")
    _check(harness, "img d_shift1", g.img.d_shift1, _in("ref_img_d_shift1"), allok)
    _check(harness, "img d_scale1", g.img.d_scale1, _in("ref_img_d_scale1"), allok)
    _check(harness, "img d_gate1 ", g.img.d_gate1, _in("ref_img_d_gate1"), allok)
    _check(harness, "img d_shift2", g.img.d_shift2, _in("ref_img_d_shift2"), allok)
    _check(harness, "img d_scale2", g.img.d_scale2, _in("ref_img_d_scale2"), allok)
    _check(harness, "img d_gate2 ", g.img.d_gate2, _in("ref_img_d_gate2"), allok)

    print("")
    print("---- TXT trainable weight grads vs torch ----")
    _check(harness, "txt d_wqkv ", g.txt.d_wqkv, _in("ref_txt_d_wqkv"), allok)
    _check(harness, "txt d_wproj", g.txt.d_wproj, _in("ref_txt_d_wproj"), allok)
    _check(harness, "txt d_wgu  ", g.txt.d_wgu, _in("ref_txt_d_wgu"), allok)
    _check(harness, "txt d_wd   ", g.txt.d_wd, _in("ref_txt_d_wd"), allok)
    _check(harness, "txt d_qnorm", g.txt.d_q_norm, _in("ref_txt_d_qnorm"), allok)
    _check(harness, "txt d_knorm", g.txt.d_k_norm, _in("ref_txt_d_knorm"), allok)

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
        print("VERDICT: PASS — Klein double-stream block fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
