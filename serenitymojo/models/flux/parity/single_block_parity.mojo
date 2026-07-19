# serenitymojo/models/flux/parity/single_block_parity.mojo
#
# PARITY GATE for the Flux SINGLE-STREAM DiT block training unit
# (models/flux/block.mojo). Loads the EXACT inputs + torch-autograd reference
# grads dumped by block_oracle.py (gen_single), runs the packaged
# single_block_forward + single_block_backward, and compares the forward output,
# d_x, every trainable weight+BIAS grad, and the modulation-vector grads at
# cos >= 0.999.
#
# REAL Flux dims: hidden D = 3072, H = 24, Dh = 128. Small S/FMLP keep the torch
# oracle fast. NON-DEGENERATE inputs + a 3-axis Flux RoPE table (asserted
# non-degenerate in the oracle).
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/flux/parity/block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/flux/parity/single_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.flux.block import (
    SingleBlockWeights, SingleModVecs,
    single_block_forward, single_block_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/"

# dims MUST match block_oracle.py gen_single()
comptime H = 24
comptime Dh = 128
comptime D = H * Dh        # 3072
comptime S = 6
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


def _load_weights(ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _in("s_in_w_w1"), _in("s_in_w_b1"),
        _in("s_in_w_w2"), _in("s_in_w_b2"),
        _in("s_in_w_q_norm"), _in("s_in_w_k_norm"),
        D, FMLP, Dh, ctx,
    )


def _load_mod() raises -> SingleModVecs:
    return SingleModVecs(
        _in("s_in_m_shift"), _in("s_in_m_scale"), _in("s_in_m_gate"),
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
    print("==== flux single_block_parity (Flux single-stream block vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " FMLP=", FMLP)

    var x = _in("s_in_x")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var cos_h = _in("s_in_cos")
    var sin_h = _in("s_in_sin")
    var cos = Tensor.from_host(cos_h, [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [S * H, Dh // 2], STDtype.F32, ctx)

    var fwd = single_block_forward[H, Dh, S](
        x.copy(), w, mv, cos, sin,
        D, FMLP, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("s_ref_out"), allok)

    var d_out = _in("s_in_d_out")
    var g = single_block_backward[H, Dh, S](
        d_out, w, mv, fwd.saved, cos, sin,
        D, FMLP, EPS, ctx,
    )

    print("")
    print("---- input grad vs torch ----")
    _check(harness, "d_x", g.d_x, _in("s_ref_d_x"), allok)

    print("")
    print("---- trainable weight+bias grads vs torch ----")
    _check(harness, "d_w1   ", g.d_w1, _in("s_ref_d_w1"), allok)
    _check(harness, "d_b1   ", g.d_b1, _in("s_ref_d_b1"), allok)
    _check(harness, "d_w2   ", g.d_w2, _in("s_ref_d_w2"), allok)
    _check(harness, "d_b2   ", g.d_b2, _in("s_ref_d_b2"), allok)
    _check(harness, "d_qnorm", g.d_q_norm, _in("s_ref_d_q_norm"), allok)
    _check(harness, "d_knorm", g.d_k_norm, _in("s_ref_d_k_norm"), allok)

    print("")
    print("---- modulation-vector grads vs torch ----")
    _check(harness, "d_shift", g.d_shift, _in("s_ref_d_shift"), allok)
    _check(harness, "d_scale", g.d_scale, _in("s_ref_d_scale"), allok)
    _check(harness, "d_gate ", g.d_gate, _in("s_ref_d_gate"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Flux single-stream block fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
