# serenitymojo/models/zimage/parity/refiner_parity.mojo
#
# PARITY GATE for the Z-Image (NextDiT) UNMODULATED context-refiner block
# (models/zimage/block.mojo `zimage_refiner_forward`/`_backward`). Loads the EXACT
# inputs + torch-autograd reference grads dumped by refiner_oracle.py, runs the
# unmodulated forward+backward, and compares the forward output, d_x, and every
# trainable weight grad at cos >= 0.999. Proves the unmodulated path before it is
# composed into the stack.
#
# REAL Z-Image H=30, Dh=128 (D=3840). Interleaved RoPE, HALF-WIDTH [S*H, Dh/2].
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/refiner_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/zimage/parity/refiner_parity.mojo -o /tmp/zimage_refiner_parity
#   /tmp/zimage_refiner_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import (
    zimage_refiner_forward, zimage_refiner_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/zimage/parity/"

# dims MUST match refiner_oracle.py
comptime H = 30
comptime Dh = 128
comptime D = H * Dh        # 3840
comptime S = 5             # N_TXT context tokens
comptime F = 96
comptime HALF = Dh // 2    # 64 (interleaved rope table width)
comptime EPS = Float32(1e-05)


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


def _t1(vals: List[Float32], n: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [n], STDtype.F32, ctx))


def _t2(vals: List[Float32], a: Int, b: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [a, b], STDtype.F32, ctx))


def _load_weights(ctx: DeviceContext) raises -> ZImageBlockWeights:
    return ZImageBlockWeights(
        _t1(_in("rin_w_n1"), D, ctx),
        _t2(_in("rin_w_wq"), D, D, ctx),
        _t2(_in("rin_w_wk"), D, D, ctx),
        _t2(_in("rin_w_wv"), D, D, ctx),
        _t2(_in("rin_w_wo"), D, D, ctx),
        _t1(_in("rin_w_q_norm"), Dh, ctx),
        _t1(_in("rin_w_k_norm"), Dh, ctx),
        _t1(_in("rin_w_n2"), D, ctx),
        _t1(_in("rin_w_fn1"), D, ctx),
        _t2(_in("rin_w_w1"), F, D, ctx),
        _t2(_in("rin_w_w3"), F, D, ctx),
        _t2(_in("rin_w_w2"), D, F, ctx),
        _t1(_in("rin_w_fn2"), D, ctx),
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
    print("==== zimage refiner_parity (UNMODULATED context-refiner vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F)

    var x = _in("rin_x")
    var w = _load_weights(ctx)

    var cos = Tensor.from_host(_in("rin_cos"), [S * H, HALF], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("rin_sin"), [S * H, HALF], STDtype.F32, ctx)

    # ── forward ──
    var fwd = zimage_refiner_forward[H, Dh, S](
        x.copy(), w, cos, sin, D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("rref_out"), allok)

    # ── backward ──
    var d_out = _in("rin_d_out")
    var g = zimage_refiner_backward[H, Dh, S](
        d_out, w, fwd.saved, cos, sin, D, F, EPS, ctx,
    )

    print("")
    print("---- input grad vs torch ----")
    _check(harness, "d_x", g.d_x, _in("rref_d_x"), allok)

    print("")
    print("---- trainable weight grads vs torch ----")
    _check(harness, "d_n1     ", g.d_n1, _in("rref_d_n1"), allok)
    _check(harness, "d_wq     ", g.d_wq, _in("rref_d_wq"), allok)
    _check(harness, "d_wk     ", g.d_wk, _in("rref_d_wk"), allok)
    _check(harness, "d_wv     ", g.d_wv, _in("rref_d_wv"), allok)
    _check(harness, "d_wo     ", g.d_wo, _in("rref_d_wo"), allok)
    _check(harness, "d_q_norm ", g.d_q_norm, _in("rref_d_q_norm"), allok)
    _check(harness, "d_k_norm ", g.d_k_norm, _in("rref_d_k_norm"), allok)
    _check(harness, "d_n2     ", g.d_n2, _in("rref_d_n2"), allok)
    _check(harness, "d_fn1    ", g.d_fn1, _in("rref_d_fn1"), allok)
    _check(harness, "d_w1     ", g.d_w1, _in("rref_d_w1"), allok)
    _check(harness, "d_w3     ", g.d_w3, _in("rref_d_w3"), allok)
    _check(harness, "d_w2     ", g.d_w2, _in("rref_d_w2"), allok)
    _check(harness, "d_fn2    ", g.d_fn2, _in("rref_d_fn2"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Z-Image unmodulated refiner fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
