# serenitymojo/models/ernie/parity/stack_parity.mojo
#
# COMPOSITION PARITY GATE for the ERNIE-Image FULL STACK training unit
# (models/ernie/ernie_stack.mojo). Loads the EXACT inputs + torch-autograd
# reference grads from stack_oracle.py, runs ernie_stack_forward +
# ernie_stack_backward at small depth (L=3) / small S (8) / reduced F, and
# compares at cos >= 0.999:
#   * forward output (out)
#   * input-token grads (d_img_tokens, d_txt_tokens) — the full-chain proof
#   * final-layer modulation grads (d_f_scale, d_f_shift)
#   * final-linear weight grad (d_final_lin)
#   * SUMMED shared-AdaLN mod grads [6D] (d_shared_mod) — the composition detail
#   * per-block weight grads, DEEPEST (L-1) and SHALLOWEST (0): d_wq, d_wdown
#
# This proves the COMPOSED backward = grad of the COMPOSED forward (the Klein
# composition-bug lesson: per-block-correct does NOT imply composition-correct).
#
# Run (oracle FIRST, SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/ernie/parity/stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/ernie/parity/stack_parity.mojo -o /tmp/ernie_stack_parity
#   /tmp/ernie_stack_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.ernie.weights import ErnieBlockWeights, ErnieStackBase
from serenitymojo.models.ernie.block import ErnieModVecs
from serenitymojo.models.ernie.ernie_stack import (
    ErnieStackForward, ErnieStackGrads,
    ernie_stack_forward, ernie_stack_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/ernie/parity/"

# dims MUST match stack_oracle.py
comptime H = 32
comptime Dh = 128
comptime D = H * Dh        # 4096
comptime N_IMG = 6
comptime N_TXT = 2
comptime S = N_IMG + N_TXT # 8
comptime F = 96
comptime IN_CH = 16
comptime TEXT_IN = 24
comptime OUT_CH = 16
comptime L = 3
comptime EPS = Float32(1e-06)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run stack_oracle.py first): ") + path)
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


def _t1(vals: List[Float32], n: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [n], STDtype.F32, ctx))


def _t2(vals: List[Float32], a: Int, b: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [a, b], STDtype.F32, ctx))


def _load_block(l: Int, ctx: DeviceContext) raises -> ErnieBlockWeights:
    var pre = String("in_blk") + String(l) + String("_")
    return ErnieBlockWeights(
        _t1(_in(pre + String("sa_norm")), D, ctx),
        _t2(_in(pre + String("wq")), D, D, ctx),
        _t2(_in(pre + String("wk")), D, D, ctx),
        _t2(_in(pre + String("wv")), D, D, ctx),
        _t2(_in(pre + String("wo")), D, D, ctx),
        _t1(_in(pre + String("q_norm")), Dh, ctx),
        _t1(_in(pre + String("k_norm")), Dh, ctx),
        _t1(_in(pre + String("mlp_norm")), D, ctx),
        _t2(_in(pre + String("wgate")), F, D, ctx),
        _t2(_in(pre + String("wup")), F, D, ctx),
        _t2(_in(pre + String("wdown")), D, F, ctx),
    )


def _load_base(ctx: DeviceContext) raises -> ErnieStackBase:
    # te_w1/te_w2/adaln/final_norm are NOT exercised by this gate (the timestep
    # MLP + adaLN MLP + final_norm.linear backprop are the deferred E5 link); the
    # gate passes the shared mod-vecs + f_scale/f_shift PRECOMPUTED. We still need
    # the ErnieStackBase struct populated with the projections it DOES use
    # (patch_w/patch_b/text_proj/final_lin_w/final_lin_b). The unused te/adaln/
    # final_norm slots are filled with the same small tensors (never read).
    var dummy_d = _t1(_in("in_f_scale"), D, ctx)          # [D] placeholder
    return ErnieStackBase(
        _t2(_in("in_patch_w"), D, IN_CH, ctx),            # patch_w  [D, in_ch]
        _t1(_in("in_patch_b"), D, ctx),                   # patch_b  [D]
        _t2(_in("in_text_proj"), D, TEXT_IN, ctx),        # text_proj [D, text_in]
        dummy_d, dummy_d, dummy_d, dummy_d,               # te_w1/b1/w2/b2 (unused)
        dummy_d, dummy_d,                                 # adaln_w/b (unused)
        dummy_d, dummy_d,                                 # final_norm_w/b (unused)
        _t2(_in("in_final_lin"), OUT_CH, D, ctx),         # final_lin_w [out_ch, D]
        _t1(_in("in_final_lin_b"), OUT_CH, ctx),          # final_lin_b [out_ch]
    )


def _load_mod() raises -> ErnieModVecs:
    return ErnieModVecs(
        _in("in_m_shift_msa"), _in("in_m_scale_msa"), _in("in_m_gate_msa"),
        _in("in_m_shift_mlp"), _in("in_m_scale_mlp"), _in("in_m_gate_mlp"),
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
    print("==== ernie stack_parity (ERNIE FULL STACK composition vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " S=", S, " F=", F, " L=", L)

    var img_tokens = _in("in_img_tokens")
    var txt_tokens = _in("in_txt_tokens")
    var base = _load_base(ctx)
    var mv = _load_mod()
    var f_scale = _in("in_f_scale")
    var f_shift = _in("in_f_shift")

    var blocks = List[ErnieBlockWeights]()
    for l in range(L):
        blocks.append(_load_block(l, ctx))

    var cos = Tensor.from_host(_in("in_cos"), [S * H, Dh], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("in_sin"), [S * H, Dh], STDtype.F32, ctx)

    # ── forward ──
    var fwd = ernie_stack_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base, blocks, mv,
        f_scale.copy(), f_shift.copy(), cos, sin,
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("ref_out"), allok)

    # ── backward ──
    var d_out = _in("in_d_out")
    var g = ernie_stack_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base, blocks, mv,
        f_scale.copy(), f_shift.copy(), cos, sin, fwd,
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
    )

    print("")
    print("---- input-token grads vs torch (full-chain proof) ----")
    _check(harness, "d_img_tokens", g.d_img_tokens, _in("ref_d_img_tokens"), allok)
    _check(harness, "d_txt_tokens", g.d_txt_tokens, _in("ref_d_txt_tokens"), allok)

    print("")
    print("---- final-layer grads vs torch ----")
    _check(harness, "d_f_scale  ", g.d_f_scale, _in("ref_d_f_scale"), allok)
    _check(harness, "d_f_shift  ", g.d_f_shift, _in("ref_d_f_shift"), allok)
    _check(harness, "d_final_lin", g.d_final_lin, _in("ref_d_final_lin"), allok)

    print("")
    print("---- SUMMED shared-AdaLN mod grads [6D] vs torch (composition) ----")
    _check(harness, "d_shared_mod", g.d_shared_mod, _in("ref_d_shared_mod"), allok)

    print("")
    print("---- per-block weight grads (deepest + shallowest) vs torch ----")
    _check(harness, "d_wq_deep    ", g.blk_grads[L - 1].d_wq, _in("ref_d_wq_deep"), allok)
    _check(harness, "d_wdown_deep ", g.blk_grads[L - 1].d_wdown, _in("ref_d_wdown_deep"), allok)
    _check(harness, "d_wq_shallow ", g.blk_grads[0].d_wq, _in("ref_d_wq_shallow"), allok)
    _check(harness, "d_wdown_shal ", g.blk_grads[0].d_wdown, _in("ref_d_wdown_shallow"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — ERNIE full-stack composition fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
