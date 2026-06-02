# serenitymojo/models/klein/parity/klein_stack_parity.mojo
#
# PARITY GATE for the Klein FULL DiT STACK (small depth) — models/klein/
# klein_stack.mojo. Loads the EXACT inputs + torch-autograd reference grads dumped
# by klein_stack_oracle.py, runs klein_stack_forward + klein_stack_backward
# (2 double + 2 single, small dims), and compares the output + the load-bearing
# input-token grads + a sample of per-block weight grads + the shared modvec grads
# + the base-weight grads, all at cos >= 0.999.
#
# This proves the COMPOSITION (input proj, the double->single concat/slice
# transition, the final layer, the d_x->d_y inter-block handoff across DEPTH).
# The individual blocks are already proven cos>=0.999; this gate proves they
# STACK.
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/klein_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/klein/parity/klein_stack_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.models.klein.double_block import StreamWeights, DoubleBlockWeights, ModVecs
from serenitymojo.models.klein.single_block import SingleBlockWeights, SingleModVecs
from serenitymojo.models.klein.klein_stack import (
    KleinStackBase, klein_stack_forward, klein_stack_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"

# dims MUST match klein_stack_oracle.py
comptime H = 4
comptime Dh = 8
comptime D = H * Dh            # 32
comptime N_IMG = 4
comptime N_TXT = 2
comptime S = N_TXT + N_IMG
comptime F = 24
comptime IN_CH = 10
comptime TXT_CH = 14
comptime OUT_CH = 6
comptime NUM_DOUBLE = 2
comptime NUM_SINGLE = 2
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


def _load_stream(prefix: String) raises -> StreamWeights:
    return StreamWeights(
        _in("in_" + prefix + "_wqkv"), _in("in_" + prefix + "_wproj"),
        _in("in_" + prefix + "_wgu"), _in("in_" + prefix + "_wd"),
        _in("in_" + prefix + "_q_norm"), _in("in_" + prefix + "_k_norm"),
    )


def _load_single(prefix: String) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _in("in_" + prefix + "_w1"), _in("in_" + prefix + "_w2"),
        _in("in_" + prefix + "_q_norm"), _in("in_" + prefix + "_k_norm"),
    )


def _load_mod(prefix: String) raises -> ModVecs:
    return ModVecs(
        _in("in_" + prefix + "_shift1"), _in("in_" + prefix + "_scale1"),
        _in("in_" + prefix + "_gate1"),
        _in("in_" + prefix + "_shift2"), _in("in_" + prefix + "_scale2"),
        _in("in_" + prefix + "_gate2"),
    )


def _load_single_mod() raises -> SingleModVecs:
    return SingleModVecs(
        _in("in_sm_shift"), _in("in_sm_scale"), _in("in_sm_gate"),
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
    print("==== klein_stack_parity (Klein FULL stack vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " F=", F, " num_double=", NUM_DOUBLE, " num_single=", NUM_SINGLE)

    # ── base weights ──
    var base = KleinStackBase(
        _in("in_img_in"), _in("in_txt_in"), _in("in_final_lin"),
        _in("in_final_shift"), _in("in_final_scale"),
    )

    # ── per-block weights ──
    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        var p = String("d") + String(bi)
        dbw.append(DoubleBlockWeights(_load_stream(p + "_iw"), _load_stream(p + "_tw")))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(_load_single(String("s") + String(bi)))

    # ── shared modulation ──
    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var sm = _load_single_mod()

    # ── tokens + rope ──
    var img_tokens = _in("in_img_tokens")
    var txt_tokens = _in("in_txt_tokens")
    var cos = _in("in_cos")
    var sin = _in("in_sin")

    # ── forward ──
    var fwd = klein_stack_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, im, tm, sm, cos.copy(), sin.copy(),
        D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("ref_out"), allok)

    # ── backward ──
    var d_out = _in("in_d_out")
    var g = klein_stack_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, im, tm, sm, cos.copy(), sin.copy(), fwd,
        D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    print("")
    print("---- load-bearing input-token grads vs torch ----")
    _check(harness, "d_img_tokens", g.d_img_tokens, _in("ref_d_img_tokens"), allok)
    _check(harness, "d_txt_tokens", g.d_txt_tokens, _in("ref_d_txt_tokens"), allok)

    print("")
    print("---- base-weight grads vs torch (input proj + final layer) ----")
    _check(harness, "d_img_in    ", g.d_img_in, _in("ref_d_img_in"), allok)
    _check(harness, "d_txt_in    ", g.d_txt_in, _in("ref_d_txt_in"), allok)
    _check(harness, "d_final_lin ", g.d_final_lin, _in("ref_d_final_lin"), allok)
    _check(harness, "d_final_shift", g.d_final_shift, _in("ref_d_final_shift"), allok)
    _check(harness, "d_final_scale", g.d_final_scale, _in("ref_d_final_scale"), allok)

    print("")
    print("---- sample per-block weight grads vs torch ----")
    # deepest double block (bi=0) rode the WHOLE inter-block chain.
    _check(harness, "d0 img d_wqkv ", g.dbl_grads[0].img.d_wqkv, _in("ref_d0_img_wqkv"), allok)
    _check(harness, "d0 img d_wproj", g.dbl_grads[0].img.d_wproj, _in("ref_d0_img_wproj"), allok)
    _check(harness, "d0 txt d_wqkv ", g.dbl_grads[0].txt.d_wqkv, _in("ref_d0_txt_wqkv"), allok)
    # last double block.
    _check(harness, "dL img d_wqkv ", g.dbl_grads[NUM_DOUBLE - 1].img.d_wqkv, _in("ref_dL_img_wqkv"), allok)
    # single blocks.
    _check(harness, "s0 d_w1       ", g.sgl_grads[0].d_w1, _in("ref_s0_w1"), allok)
    _check(harness, "s0 d_w2       ", g.sgl_grads[0].d_w2, _in("ref_s0_w2"), allok)
    _check(harness, "sL d_w1       ", g.sgl_grads[NUM_SINGLE - 1].d_w1, _in("ref_sL_w1"), allok)

    print("")
    print("---- shared modulation-vector grads vs torch (summed across blocks) ----")
    _check(harness, "d_img_mod   ", g.d_img_mod, _in("ref_d_img_mod"), allok)
    _check(harness, "d_txt_mod   ", g.d_txt_mod, _in("ref_d_txt_mod"), allok)
    _check(harness, "d_single_mod", g.d_single_mod, _in("ref_d_single_mod"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Klein FULL stack fwd+bwd composes correctly (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
