# serenitymojo/models/wan22/parity/wan22_block_parity.mojo
#
# PARITY GATE for the Wan2.2 WanAttentionBlock training unit
# (models/wan22/wan22_block.mojo). Loads the EXACT inputs + torch-autograd grads
# dumped by wan22_block_oracle.py, runs wan22_block_forward + wan22_block_backward,
# and compares x_out, d_x (img input grad), d_context (txt input grad), every
# trainable weight/bias grad, the QK-norm grads, the affine-LN grads, and the
# per-token modulation-vector grads at cos >= 0.999.
#
# REAL Wan2.2 head count H = 24. NON-DEGENERATE sinusoidal/random inputs.
#
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/wan22/parity/wan22_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/wan22/parity/wan22_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanModVecs,
    wan22_block_forward, wan22_block_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/wan22/parity/"

# dims MUST match wan22_block_oracle.py
comptime H = 24
comptime Dh = 8
comptime DIM = H * Dh        # 192
comptime S = 5
comptime TXT = 4
comptime FFN = 40
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


def _load_weights(ctx: DeviceContext) raises -> WanBlockWeights:
    return WanBlockWeights(
        _in("in_sa_wq"), _in("in_sa_wk"), _in("in_sa_wv"), _in("in_sa_wo"),
        _in("in_sa_bq"), _in("in_sa_bk"), _in("in_sa_bv"), _in("in_sa_bo"),
        _in("in_sa_qn"), _in("in_sa_kn"),
        _in("in_ca_wq"), _in("in_ca_wk"), _in("in_ca_wv"), _in("in_ca_wo"),
        _in("in_ca_bq"), _in("in_ca_bk"), _in("in_ca_bv"), _in("in_ca_bo"),
        _in("in_ca_qn"), _in("in_ca_kn"),
        _in("in_n3_w"), _in("in_n3_b"),
        _in("in_ffn0_w"), _in("in_ffn0_b"), _in("in_ffn2_w"), _in("in_ffn2_b"),
        DIM, FFN, Dh, ctx,
    )


def _load_mod() raises -> WanModVecs:
    return WanModVecs(
        _in("in_shift_sa"), _in("in_scale_sa"), _in("in_gate_sa"),
        _in("in_shift_ffn"), _in("in_scale_ffn"), _in("in_gate_ffn"),
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
    print("==== wan22_block_parity (Wan2.2 WanAttentionBlock vs torch) ====")
    print("H=", H, " Dh=", Dh, " DIM=", DIM, " S=", S, " TXT=", TXT, " FFN=", FFN)

    var x = _in("in_x")
    var context = _in("in_context")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var cos = Tensor.from_host(_in("in_cos"), [S, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("in_sin"), [S, Dh // 2], STDtype.F32, ctx)

    var fwd = wan22_block_forward[H, Dh, S, TXT](
        x.copy(), context.copy(), mv, w, cos, sin, DIM, FFN, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "x_out", fwd.x_out, _in("ref_x_out"), allok)

    var d_out = _in("in_d_out")
    var g = wan22_block_backward[H, Dh, S, TXT](
        d_out, mv, w, fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )

    print("")
    print("---- input grads vs torch ----")
    _check(harness, "d_x (img)    ", g.d_x, _in("ref_d_x"), allok)
    _check(harness, "d_context(txt)", g.d_context, _in("ref_d_context"), allok)

    print("")
    print("---- self-attn weight/bias grads vs torch ----")
    _check(harness, "d_sa_wq", g.d_sa_wq, _in("ref_d_sa_wq"), allok)
    _check(harness, "d_sa_wk", g.d_sa_wk, _in("ref_d_sa_wk"), allok)
    _check(harness, "d_sa_wv", g.d_sa_wv, _in("ref_d_sa_wv"), allok)
    _check(harness, "d_sa_wo", g.d_sa_wo, _in("ref_d_sa_wo"), allok)
    _check(harness, "d_sa_bq", g.d_sa_bq, _in("ref_d_sa_bq"), allok)
    _check(harness, "d_sa_bk", g.d_sa_bk, _in("ref_d_sa_bk"), allok)
    _check(harness, "d_sa_bv", g.d_sa_bv, _in("ref_d_sa_bv"), allok)
    _check(harness, "d_sa_bo", g.d_sa_bo, _in("ref_d_sa_bo"), allok)
    _check(harness, "d_sa_qn", g.d_sa_qn, _in("ref_d_sa_qn"), allok)
    _check(harness, "d_sa_kn", g.d_sa_kn, _in("ref_d_sa_kn"), allok)

    print("")
    print("---- cross-attn weight/bias grads vs torch ----")
    _check(harness, "d_ca_wq", g.d_ca_wq, _in("ref_d_ca_wq"), allok)
    _check(harness, "d_ca_wk", g.d_ca_wk, _in("ref_d_ca_wk"), allok)
    _check(harness, "d_ca_wv", g.d_ca_wv, _in("ref_d_ca_wv"), allok)
    _check(harness, "d_ca_wo", g.d_ca_wo, _in("ref_d_ca_wo"), allok)
    _check(harness, "d_ca_bq", g.d_ca_bq, _in("ref_d_ca_bq"), allok)
    _check(harness, "d_ca_bk", g.d_ca_bk, _in("ref_d_ca_bk"), allok)
    _check(harness, "d_ca_bv", g.d_ca_bv, _in("ref_d_ca_bv"), allok)
    _check(harness, "d_ca_bo", g.d_ca_bo, _in("ref_d_ca_bo"), allok)
    _check(harness, "d_ca_qn", g.d_ca_qn, _in("ref_d_ca_qn"), allok)
    _check(harness, "d_ca_kn", g.d_ca_kn, _in("ref_d_ca_kn"), allok)

    print("")
    print("---- norm3 affine + ffn grads vs torch ----")
    _check(harness, "d_n3_w  ", g.d_n3_w, _in("ref_d_n3_w"), allok)
    _check(harness, "d_n3_b  ", g.d_n3_b, _in("ref_d_n3_b"), allok)
    _check(harness, "d_ffn0_w", g.d_ffn0_w, _in("ref_d_ffn0_w"), allok)
    _check(harness, "d_ffn0_b", g.d_ffn0_b, _in("ref_d_ffn0_b"), allok)
    _check(harness, "d_ffn2_w", g.d_ffn2_w, _in("ref_d_ffn2_w"), allok)
    _check(harness, "d_ffn2_b", g.d_ffn2_b, _in("ref_d_ffn2_b"), allok)

    print("")
    print("---- per-token modulation-vector grads vs torch ----")
    _check(harness, "d_shift_sa ", g.d_shift_sa, _in("ref_d_shift_sa"), allok)
    _check(harness, "d_scale_sa ", g.d_scale_sa, _in("ref_d_scale_sa"), allok)
    _check(harness, "d_gate_sa  ", g.d_gate_sa, _in("ref_d_gate_sa"), allok)
    _check(harness, "d_shift_ffn", g.d_shift_ffn, _in("ref_d_shift_ffn"), allok)
    _check(harness, "d_scale_ffn", g.d_scale_ffn, _in("ref_d_scale_ffn"), allok)
    _check(harness, "d_gate_ffn ", g.d_gate_ffn, _in("ref_d_gate_ffn"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Wan2.2 block fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
