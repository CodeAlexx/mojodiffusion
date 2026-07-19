# serenitymojo/models/zimage/parity/zimage_controlnet_block_parity.mojo
#
# T2.E PARITY GATE for the Z-Image CONTROLNET control block + 2-block control
# stack (models/zimage/controlnet_block.mojo) vs torch autograd.
#
# Reference: zimage_controlnet_block_oracle.py — F64 hand math GROUNDED against
# the REAL diffusers 0.38.0.dev0 ZImageControlTransformerBlock (the oracle runs
# that cross-check itself and aborts on divergence). Covers:
#   * block 0 (before_proj + body + after_proj) and block 1 (body + after_proj)
#   * the all_c chaining (d_c handoff block1 -> block0)
#   * grads: both hints' injection-site grads -> every trainable tensor
#     (13 body weights + 4 RAW adaLN chunks + projections, per block), plus
#     d_c0 (control input) and d_x (unified input pass-through at block 0)
# Bar: cos >= 0.99999 (F32), the repo's standard block-oracle bar.
#
# Run (oracle FIRST, separate command):
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/models/zimage/parity/zimage_controlnet_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/zimage/parity/zimage_controlnet_block_parity.mojo \
#       -o /tmp/zimage_controlnet_block_parity
#   /tmp/zimage_controlnet_block_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.controlnet_block import (
    ZImageControlBlockWeights,
    zimage_control_stack_forward, zimage_control_stack_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/zimage/parity/"

# dims MUST match zimage_controlnet_block_oracle.py
comptime H = 30
comptime Dh = 128
comptime D = H * Dh        # 3840
comptime S = 8
comptime F = 96
comptime HALF = Dh // 2
comptime EPS = Float32(1e-05)
comptime COS_BAR = 0.99999


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


def _load_body(tag: String, ctx: DeviceContext) raises -> ZImageBlockWeights:
    var p = String("cn_in_") + tag + String("_w_")
    return ZImageBlockWeights(
        _t1(_in(p + String("n1")), D, ctx),
        _t2(_in(p + String("wq")), D, D, ctx),
        _t2(_in(p + String("wk")), D, D, ctx),
        _t2(_in(p + String("wv")), D, D, ctx),
        _t2(_in(p + String("wo")), D, D, ctx),
        _t1(_in(p + String("q_norm")), Dh, ctx),
        _t1(_in(p + String("k_norm")), Dh, ctx),
        _t1(_in(p + String("n2")), D, ctx),
        _t1(_in(p + String("fn1")), D, ctx),
        _t2(_in(p + String("w1")), F, D, ctx),
        _t2(_in(p + String("w3")), F, D, ctx),
        _t2(_in(p + String("w2")), D, F, ctx),
        _t1(_in(p + String("fn2")), D, ctx),
    )


def _load_mod(tag: String) raises -> ZImageModVecs:
    var p = String("cn_in_") + tag + String("_m_")
    return ZImageModVecs(
        _in(p + String("scale_msa")), _in(p + String("gate_msa")),
        _in(p + String("scale_mlp")), _in(p + String("gate_mlp")),
    )


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


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
    print("==== zimage controlnet block parity (vs diffusers ZImageControlNet) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F, " bar cos>=", COS_BAR)

    # block 0: body + before_proj + after_proj
    var b0 = ZImageControlBlockWeights(
        _load_body(String("b0"), ctx),
        _t2(_in("cn_in_b0_before_w"), D, D, ctx),
        _t1(_in("cn_in_b0_before_b"), D, ctx),
        _t2(_in("cn_in_b0_after_w"), D, D, ctx),
        _t1(_in("cn_in_b0_after_b"), D, ctx),
        True,
    )
    # block 1: body + after_proj only (before placeholders never read)
    var b1 = ZImageControlBlockWeights(
        _load_body(String("b1"), ctx),
        _t2(_zeros(1), 1, 1, ctx), _t1(_zeros(1), 1, ctx),
        _t2(_in("cn_in_b1_after_w"), D, D, ctx),
        _t1(_in("cn_in_b1_after_b"), D, ctx),
        False,
    )
    var blocks = List[ZImageControlBlockWeights]()
    blocks.append(b0^)
    blocks.append(b1^)
    var mods = List[ZImageModVecs]()
    mods.append(_load_mod(String("b0")))
    mods.append(_load_mod(String("b1")))

    var cos = Tensor.from_host(_in("cn_in_cos"), [S * H, HALF], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("cn_in_sin"), [S * H, HALF], STDtype.F32, ctx)

    var c0 = _in("cn_in_c0")
    var x = _in("cn_in_x")

    # ── forward: 2-block control stack ──
    var fwd = zimage_control_stack_forward[H, Dh, S](
        c0, x, blocks, mods, cos, sin, D, F, EPS, ctx,
    )

    var harness = ParityHarness(COS_BAR)
    var allok = True

    print("")
    print("---- forward (hints + carried c) vs torch ----")
    _check(harness, "hint0  ", fwd.hints[0], _in("cn_ref_hint0"), allok)
    _check(harness, "hint1  ", fwd.hints[1], _in("cn_ref_hint1"), allok)
    _check(harness, "c_final", fwd.c_final, _in("cn_ref_c_final"), allok)

    # ── backward: injection-site grads on both hints ──
    var d_hints = List[List[Float32]]()
    d_hints.append(_in("cn_in_d_hint0"))
    d_hints.append(_in("cn_in_d_hint1"))
    var g = zimage_control_stack_backward[H, Dh, S](
        d_hints, blocks, mods, fwd.saveds, cos, sin, D, F, EPS, ctx,
    )

    print("")
    print("---- stream grads vs torch ----")
    _check(harness, "d_c0", g.d_c0, _in("cn_ref_d_c0"), allok)
    _check(harness, "d_x ", g.d_x, _in("cn_ref_d_x"), allok)

    var tags = List[String]()
    tags.append(String("b0"))
    tags.append(String("b1"))
    for bi in range(2):
        var tag = tags[bi].copy()
        var rp = String("cn_ref_") + tag + String("_d_")
        print("")
        print("---- block", tag, "trainable grads vs torch ----")
        ref bg = g.blocks[bi]
        _check(harness, tag + String(" d_n1     "), bg.body.d_n1, _in(rp + String("n1")), allok)
        _check(harness, tag + String(" d_wq     "), bg.body.d_wq, _in(rp + String("wq")), allok)
        _check(harness, tag + String(" d_wk     "), bg.body.d_wk, _in(rp + String("wk")), allok)
        _check(harness, tag + String(" d_wv     "), bg.body.d_wv, _in(rp + String("wv")), allok)
        _check(harness, tag + String(" d_wo     "), bg.body.d_wo, _in(rp + String("wo")), allok)
        _check(harness, tag + String(" d_q_norm "), bg.body.d_q_norm, _in(rp + String("q_norm")), allok)
        _check(harness, tag + String(" d_k_norm "), bg.body.d_k_norm, _in(rp + String("k_norm")), allok)
        _check(harness, tag + String(" d_n2     "), bg.body.d_n2, _in(rp + String("n2")), allok)
        _check(harness, tag + String(" d_fn1    "), bg.body.d_fn1, _in(rp + String("fn1")), allok)
        _check(harness, tag + String(" d_w1     "), bg.body.d_w1, _in(rp + String("w1")), allok)
        _check(harness, tag + String(" d_w3     "), bg.body.d_w3, _in(rp + String("w3")), allok)
        _check(harness, tag + String(" d_w2     "), bg.body.d_w2, _in(rp + String("w2")), allok)
        _check(harness, tag + String(" d_fn2    "), bg.body.d_fn2, _in(rp + String("fn2")), allok)
        _check(harness, tag + String(" d_scale_msa"), bg.body.d_scale_msa, _in(rp + String("scale_msa")), allok)
        _check(harness, tag + String(" d_gate_msa "), bg.body.d_gate_msa, _in(rp + String("gate_msa")), allok)
        _check(harness, tag + String(" d_scale_mlp"), bg.body.d_scale_mlp, _in(rp + String("scale_mlp")), allok)
        _check(harness, tag + String(" d_gate_mlp "), bg.body.d_gate_mlp, _in(rp + String("gate_mlp")), allok)
        if bi == 0:
            _check(harness, tag + String(" d_before_w"), bg.d_before_w, _in(rp + String("before_w")), allok)
            _check(harness, tag + String(" d_before_b"), bg.d_before_b, _in(rp + String("before_b")), allok)
        _check(harness, tag + String(" d_after_w "), bg.d_after_w, _in(rp + String("after_w")), allok)
        _check(harness, tag + String(" d_after_b "), bg.d_after_b, _in(rp + String("after_b")), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Z-Image ControlNet control block fwd+bwd matches torch (cos>=0.99999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
