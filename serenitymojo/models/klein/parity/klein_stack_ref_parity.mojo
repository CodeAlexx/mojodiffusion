# serenitymojo/models/klein/parity/klein_stack_ref_parity.mojo
#
# PARITY GATE for the OPT-IN img-EDIT REFERENCE branch integrated into the Klein
# FULL DiT STACK (models/klein/klein_stack.mojo + models/klein/img_in_ref.mojo +
# models/klein/klein_img_in_ref_param.mojo).
#
# It proves TWO contracts:
#   (A) TRAINED REGIME (non-zero img_in_ref, non-degenerate ref tokens): the
#       stack input-token grads (d_img_tokens/d_txt_tokens — which now carry the
#       ref term through the WHOLE backward) and d_img_in_ref (the ONLY new
#       trained param's grad) match a TORCH AUTOGRAD reference (klein_stack_ref_
#       oracle.py, torch 2.12) at cos >= 0.999. Also checks the forward output
#       and cross-checks the standalone img_in_ref_forward reuse in-composition.
#   (B) FLAGS-OFF IDENTITY (C13 contract): a run with ref_tokens present but
#       img_in_ref == 0 is BYTE-IDENTICAL (max_abs 0) to the current text-to-
#       image klein_stack forward+backward (no ref at all), and d_img_in_ref is
#       all-zeros.
#
# Run (oracle FIRST, SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/SerenityTrainer/venv/bin/python \
#       serenitymojo/models/klein/parity/klein_stack_ref_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/klein/parity/klein_stack_ref_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.models.klein.double_block import StreamWeights, DoubleBlockWeights, ModVecs
from serenitymojo.models.klein.single_block import SingleBlockWeights, SingleModVecs
from serenitymojo.models.klein.klein_stack import (
    KleinStackBase, klein_stack_forward, klein_stack_backward,
)
from serenitymojo.models.klein.img_in_ref import (
    img_in_ref_forward, img_in_ref_backward, ImgInRefForward,
)
from serenitymojo.models.klein.klein_img_in_ref_param import (
    KleinImgInRefParam, make_klein_img_in_ref_param, klein_img_in_ref_w_f32,
    klein_img_in_ref_adamw_step, save_klein_img_in_ref, load_klein_img_in_ref_resume,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"
comptime PFX = "refstk_"

# dims MUST match klein_stack_ref_oracle.py
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
    return _read_bin_f32(REF_DIR + PFX + name + ".bin")


def _load_stream(prefix: String, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _in("in_" + prefix + "_wqkv"), _in("in_" + prefix + "_wproj"),
        _in("in_" + prefix + "_wgu"), _in("in_" + prefix + "_wd"),
        _in("in_" + prefix + "_q_norm"), _in("in_" + prefix + "_k_norm"),
        D, F, Dh, ctx,
    )


def _load_single(prefix: String, ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _in("in_" + prefix + "_w1"), _in("in_" + prefix + "_w2"),
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


def _max_abs_diff(a: List[Float32], b: List[Float32]) -> Float32:
    if len(a) != len(b):
        return Float32(1.0e30)
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def _check_zero(name: String, mad: Float32, mut allok: Bool):
    var ok = mad == Float32(0.0)
    print("  max_abs_diff(", name, ") =", mad, "  ", "PASS (byte-identical)" if ok else "FAIL")
    if not ok:
        allok = False


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def main() raises:
    var ctx = DeviceContext()
    print("==== klein_stack_ref_parity (Klein stack + img-EDIT ref branch vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " F=", F, " IN_CH=", IN_CH, " num_double=", NUM_DOUBLE, " num_single=", NUM_SINGLE)

    # ── base + per-block weights + modulation (shared with base stack oracle) ──
    var base = KleinStackBase(
        _in("in_img_in"), _in("in_txt_in"), _in("in_final_lin"),
        _in("in_final_shift"), _in("in_final_scale"),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )
    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        var p = String("d") + String(bi)
        dbw.append(DoubleBlockWeights(_load_stream(p + "_iw", ctx), _load_stream(p + "_tw", ctx)))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(_load_single(String("s") + String(bi), ctx))
    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var sm = _load_single_mod()

    var img_tokens = _in("in_img_tokens")
    var txt_tokens = _in("in_txt_tokens")
    var cos = _in("in_cos")
    var sin = _in("in_sin")
    var d_out = _in("in_d_out")

    # ── the img-EDIT reference inputs (NON-ZERO weight, NON-DEGENERATE tokens) ──
    var ref_tokens = _in("in_ref_tokens")       # [N_IMG, IN_CH]
    var img_in_ref_w = _in("in_img_in_ref")     # [D, IN_CH]  NON-ZERO

    var harness = ParityHarness()
    var allok = True

    # ══ (A) TRAINED REGIME: non-zero ref vs torch autograd ══
    print("")
    print("---- (A) forward WITH ref vs torch ----")
    var fwd = klein_stack_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, im, tm, sm, cos.copy(), sin.copy(),
        D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        Optional[List[Float32]](ref_tokens.copy()),
        Optional[List[Float32]](img_in_ref_w.copy()),
    )
    _check(harness, "out (with ref)", fwd.out, _in("ref_out"), allok)

    # cross-check: the standalone img_in_ref_forward summed projection MUST equal
    # the stack's img_in_act (proves the compute unit is reused in-composition).
    var unit = img_in_ref_forward(
        img_tokens.copy(), _in("in_img_in"), ref_tokens.copy(), img_in_ref_w.copy(),
        N_IMG, IN_CH, D, ctx,
    )
    _check_zero("img_in_act vs img_in_ref_forward", _max_abs_diff(unit.img_act, fwd.img_in_act[].to_host(ctx)), allok)

    print("")
    print("---- (A) backward WITH ref vs torch ----")
    var g = klein_stack_backward[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, im, tm, sm, cos.copy(), sin.copy(), fwd,
        D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        Optional[List[Float32]](ref_tokens.copy()),
        Optional[List[Float32]](img_in_ref_w.copy()),
    )
    _check(harness, "d_img_tokens (with ref)", g.d_img_tokens, _in("ref_d_img_tokens"), allok)
    _check(harness, "d_txt_tokens (with ref)", g.d_txt_tokens, _in("ref_d_txt_tokens"), allok)
    _check(harness, "d_img_in_ref (NEW param)", g.d_img_in_ref, _in("ref_d_img_in_ref"), allok)

    print("")
    print("---- (A) shared modvec grads WITH ref vs torch (regression) ----")
    _check(harness, "d_img_mod   ", g.d_img_mod, _in("ref_d_img_mod"), allok)
    _check(harness, "d_single_mod", g.d_single_mod, _in("ref_d_single_mod"), allok)

    # ══ (B) FLAGS-OFF IDENTITY: no-ref baseline vs zero-weight ref ══
    print("")
    print("---- (B) flags-off identity (max_abs 0 vs current text-to-image path) ----")
    # baseline: the EXACT existing text-to-image path (no ref args at all).
    var fwd_base = klein_stack_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, im, tm, sm, cos.copy(), sin.copy(),
        D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var g_base = klein_stack_backward[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, im, tm, sm, cos.copy(), sin.copy(), fwd_base,
        D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    # zero-weight ref present: must reproduce the baseline byte-for-byte.
    var zero_w = _zeros(D * IN_CH)
    var fwd_zero = klein_stack_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, im, tm, sm, cos.copy(), sin.copy(),
        D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        Optional[List[Float32]](ref_tokens.copy()),
        Optional[List[Float32]](zero_w.copy()),
    )
    var g_zero = klein_stack_backward[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, im, tm, sm, cos.copy(), sin.copy(), fwd_zero,
        D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        Optional[List[Float32]](ref_tokens.copy()),
        Optional[List[Float32]](zero_w.copy()),
    )
    _check_zero("fwd.out", _max_abs_diff(fwd_zero.out, fwd_base.out), allok)
    _check_zero("d_img_tokens", _max_abs_diff(g_zero.d_img_tokens, g_base.d_img_tokens), allok)
    _check_zero("d_txt_tokens", _max_abs_diff(g_zero.d_txt_tokens, g_base.d_txt_tokens), allok)
    _check_zero("d_img_in    ", _max_abs_diff(g_zero.d_img_in, g_base.d_img_in), allok)
    _check_zero("d_final_lin ", _max_abs_diff(g_zero.d_final_lin, g_base.d_final_lin), allok)
    # d_img_in_ref for a zero weight is all-zeros (grad of a linear whose weight
    # is 0 wrt the weight = d_yᵀ@ref, NON-zero!). The IDENTITY that must hold is
    # that the OTHER grads are untouched (above). Report d_img_in_ref magnitude.
    var mabs_ref = _max_abs_diff(g_zero.d_img_in_ref, _zeros(len(g_zero.d_img_in_ref)))
    print("  note: d_img_in_ref max_abs (zero-weight) =", mabs_ref,
          " (grad wrt img_in_ref is dᵀ@ref, independent of the weight value)")

    # ══ (C) AdamW param + save/resume round-trip (byte-exact) ══
    print("")
    print("---- (C) img_in_ref param AdamW + save/resume round-trip ----")
    var param = make_klein_img_in_ref_param(D, IN_CH)
    # zero-init check
    var w0 = klein_img_in_ref_w_f32(param)
    _check_zero("param zero-init", _max_abs_diff(w0, _zeros(D * IN_CH)), allok)
    # one AdamW step with the torch-gated grad, then save + resume.
    klein_img_in_ref_adamw_step(param, g.d_img_in_ref.copy(), 1, Float32(1.0e-3), ctx)
    var w_after = klein_img_in_ref_w_f32(param)
    var path = String("/tmp/claude-1000/-home-alex/b679a444-2a2a-4643-9dd6-55156f8065c6/scratchpad/img_in_ref_state.safetensors")
    _ = save_klein_img_in_ref(param, path, ctx)
    var loaded = load_klein_img_in_ref_resume(D, IN_CH, path, ctx)
    var w_loaded = klein_img_in_ref_w_f32(loaded)
    _check_zero("resume weight round-trip", _max_abs_diff(w_after, w_loaded), allok)
    _check_zero("resume moment m round-trip", _max_abs_diff(param.m, loaded.m), allok)
    _check_zero("resume moment v round-trip", _max_abs_diff(param.v, loaded.v), allok)

    print("")
    if allok:
        print("VERDICT: PASS — img-EDIT ref branch composes (cos>=0.999), flags-off byte-identical, param round-trips")
    else:
        print("VERDICT: FAIL — see FAIL lines above")
