# serenitymojo/models/klein/parity/single_block_parity.mojo
#
# PARITY GATE for the Klein SINGLE-STREAM DiT block training unit
# (models/klein/single_block.mojo). Loads the EXACT inputs + torch-autograd
# reference grads dumped by single_block_oracle.py, runs the packaged
# single_block_forward + single_block_backward, and compares d_x, every
# trainable weight grad, and the modulation-vector grads at cos >= 0.999.
#
# REAL Klein head count H = 32 (the dim that PASSES sdpa backward). Small S/Dh to
# keep the torch oracle fast. NON-DEGENERATE sinusoidal/random inputs (no modular
# fills that alias and fake zero grads).
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/single_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/klein/parity/single_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecs,
    single_block_forward, single_block_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"

# dims MUST match single_block_oracle.py
comptime H = 32
comptime Dh = 16
comptime D = H * Dh        # 512
comptime S = 6
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


def _load_weights(ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _in("in_w_w1"), _in("in_w_w2"),
        _in("in_w_q_norm"), _in("in_w_k_norm"),
        D, F, Dh, ctx,
    )


def _load_mod() raises -> SingleModVecs:
    return SingleModVecs(
        _in("in_m_shift"), _in("in_m_scale"), _in("in_m_gate"),
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
    print("==== single_block_parity (Klein single-stream block vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F)

    # ── load inputs (byte-identical to the oracle) ──
    var x = _in("in_x")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var cos_h = _in("in_cos")
    var sin_h = _in("in_sin")
    # Resident rope tables: upload ONCE, pass by borrow (matches the trainer).
    var cos = Tensor.from_host(cos_h, [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [S * H, Dh // 2], STDtype.F32, ctx)

    # ── forward ──
    var fwd = single_block_forward[H, Dh, S](
        x.copy(), w, mv, cos, sin,
        D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("ref_out"), allok)

    # ── backward ──
    var d_out = _in("in_d_out")
    var g = single_block_backward[H, Dh, S](
        d_out, w, mv, fwd.saved, cos, sin,
        D, F, EPS, ctx,
    )

    print("")
    print("---- input grad vs torch ----")
    _check(harness, "d_x", g.d_x, _in("ref_d_x"), allok)

    print("")
    print("---- trainable weight grads vs torch ----")
    _check(harness, "d_w1   ", g.d_w1, _in("ref_d_w1"), allok)
    _check(harness, "d_w2   ", g.d_w2, _in("ref_d_w2"), allok)
    _check(harness, "d_qnorm", g.d_q_norm, _in("ref_d_qnorm"), allok)
    _check(harness, "d_knorm", g.d_k_norm, _in("ref_d_knorm"), allok)

    print("")
    print("---- modulation-vector grads vs torch ----")
    _check(harness, "d_shift", g.d_shift, _in("ref_d_shift"), allok)
    _check(harness, "d_scale", g.d_scale, _in("ref_d_scale"), allok)
    _check(harness, "d_gate ", g.d_gate, _in("ref_d_gate"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Klein single-stream block fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
