# serenitymojo/models/zimage/parity/block_parity.mojo
#
# PARITY GATE for the Z-Image (NextDiT) MAIN-LAYER DiT block training unit
# (models/zimage/block.mojo). Loads the EXACT inputs + torch-autograd reference
# grads dumped by block_oracle.py, runs zimage_block_forward +
# zimage_block_backward, and compares the forward output, d_x, every trainable
# weight grad, and the RAW adaLN modulation-vector grads at cos >= 0.999.
#
# REAL Z-Image head count H=30, head_dim Dh=128 (hidden D=3840). Small S/F to keep
# the torch oracle + GPU memory bounded on the shared 3090.
#
# RoPE convention: INTERLEAVED (pair (x[2i],x[2i+1])), HALF-WIDTH [S*H, Dh/2]
# tables — confirmed against diffusers transformer_z_image.py + zimage_dit.mojo.
# The backward uses rope_backward(..., interleaved=True). (NOT Ernie's half-split.)
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/zimage/parity/block_parity.mojo -o /tmp/zimage_block_parity
#   /tmp/zimage_block_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import (
    ZImageModVecs, zimage_block_forward, zimage_block_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/zimage/parity/"

# dims MUST match block_oracle.py
comptime H = 30
comptime Dh = 128
comptime D = H * Dh        # 3840
comptime S = 8
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
        _t1(_in("in_w_n1"), D, ctx),
        _t2(_in("in_w_wq"), D, D, ctx),
        _t2(_in("in_w_wk"), D, D, ctx),
        _t2(_in("in_w_wv"), D, D, ctx),
        _t2(_in("in_w_wo"), D, D, ctx),
        _t1(_in("in_w_q_norm"), Dh, ctx),
        _t1(_in("in_w_k_norm"), Dh, ctx),
        _t1(_in("in_w_n2"), D, ctx),
        _t1(_in("in_w_fn1"), D, ctx),
        _t2(_in("in_w_w1"), F, D, ctx),
        _t2(_in("in_w_w3"), F, D, ctx),
        _t2(_in("in_w_w2"), D, F, ctx),
        _t1(_in("in_w_fn2"), D, ctx),
    )


def _load_mod() raises -> ZImageModVecs:
    return ZImageModVecs(
        _in("in_m_scale_msa"), _in("in_m_gate_msa"),
        _in("in_m_scale_mlp"), _in("in_m_gate_mlp"),
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
    print("==== zimage block_parity (Z-Image main-layer block vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F)

    var x = _in("in_x")
    var w = _load_weights(ctx)
    var mv = _load_mod()

    # HALF-WIDTH interleaved rope tables [S*H, Dh/2] for BOTH forward and backward.
    var cos = Tensor.from_host(_in("in_cos"), [S * H, HALF], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("in_sin"), [S * H, HALF], STDtype.F32, ctx)

    # ── forward ──
    var fwd = zimage_block_forward[H, Dh, S](
        x.copy(), w, mv, cos, sin, D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("ref_out"), allok)

    # ── backward ──
    var d_out = _in("in_d_out")
    var g = zimage_block_backward[H, Dh, S](
        d_out, w, mv, fwd.saved, cos, sin, D, F, EPS, ctx,
    )

    print("")
    print("---- input grad vs torch ----")
    _check(harness, "d_x", g.d_x, _in("ref_d_x"), allok)

    print("")
    print("---- trainable weight grads vs torch ----")
    _check(harness, "d_n1     ", g.d_n1, _in("ref_d_n1"), allok)
    _check(harness, "d_wq     ", g.d_wq, _in("ref_d_wq"), allok)
    _check(harness, "d_wk     ", g.d_wk, _in("ref_d_wk"), allok)
    _check(harness, "d_wv     ", g.d_wv, _in("ref_d_wv"), allok)
    _check(harness, "d_wo     ", g.d_wo, _in("ref_d_wo"), allok)
    _check(harness, "d_q_norm ", g.d_q_norm, _in("ref_d_q_norm"), allok)
    _check(harness, "d_k_norm ", g.d_k_norm, _in("ref_d_k_norm"), allok)
    _check(harness, "d_n2     ", g.d_n2, _in("ref_d_n2"), allok)
    _check(harness, "d_fn1    ", g.d_fn1, _in("ref_d_fn1"), allok)
    _check(harness, "d_w1     ", g.d_w1, _in("ref_d_w1"), allok)
    _check(harness, "d_w3     ", g.d_w3, _in("ref_d_w3"), allok)
    _check(harness, "d_w2     ", g.d_w2, _in("ref_d_w2"), allok)
    _check(harness, "d_fn2    ", g.d_fn2, _in("ref_d_fn2"), allok)

    print("")
    print("---- RAW adaLN modulation-vector grads vs torch ----")
    _check(harness, "d_scale_msa", g.d_scale_msa, _in("ref_d_scale_msa"), allok)
    _check(harness, "d_gate_msa ", g.d_gate_msa, _in("ref_d_gate_msa"), allok)
    _check(harness, "d_scale_mlp", g.d_scale_mlp, _in("ref_d_scale_mlp"), allok)
    _check(harness, "d_gate_mlp ", g.d_gate_mlp, _in("ref_d_gate_mlp"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Z-Image main-layer block fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
