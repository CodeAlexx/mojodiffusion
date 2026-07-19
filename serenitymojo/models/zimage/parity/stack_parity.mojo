# serenitymojo/models/zimage/parity/stack_parity.mojo
#
# COMPOSITION PARITY GATE for the Z-Image (NextDiT) FULL STACK training unit
# (models/zimage/zimage_stack.mojo). Loads the EXACT inputs + torch-autograd
# reference grads from stack_oracle.py, runs zimage_stack_forward +
# zimage_stack_backward at reduced depth (1 noise refiner + 1 context refiner +
# 2 main), REAL H=30/Dh=128/D=3840, and compares at cos >= 0.999:
#   * forward output (out)
#   * input-token grads (d_x_seq, d_cap_seq) — the full-chain proof for both streams
#   * final-layer grads (d_f_scale, d_final_lin)
#   * representative weight grads: deepest main (wq, w2), shallow main (wq),
#     noise refiner (wq, w2), context refiner (wq, w2)
#   * per-block RAW mod-vec grads [4D]: noise refiner + deepest main
#
# Proves the COMPOSED backward = grad of the COMPOSED forward (the Klein
# composition-bug lesson: per-block-correct does NOT imply composition-correct).
#
# Run (oracle FIRST, SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/zimage/parity/stack_parity.mojo -o /tmp/zimage_stack_parity
#   /tmp/zimage_stack_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs, ZImageBlockGrads
from serenitymojo.models.zimage.zimage_stack import (
    ZImageStackForward, ZImageStackGrads,
    zimage_stack_forward, zimage_stack_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/zimage/parity/"

# dims MUST match stack_oracle.py
comptime H = 30
comptime Dh = 128
comptime D = H * Dh        # 3840
comptime IMG_H = 2
comptime IMG_W = 3
comptime N_IMG = IMG_H * IMG_W   # 6
comptime N_TXT = 4
comptime S = N_IMG + N_TXT       # 10
comptime F = 96
comptime OUT_CH = 16
comptime HALF = Dh // 2          # 64
comptime EPS = Float32(1e-05)
comptime FINAL_EPS = Float32(1e-06)
comptime NUM_NR = 1
comptime NUM_CR = 1
comptime NUM_MAIN = 2


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


def _load_block(prefix: String, ctx: DeviceContext) raises -> ZImageBlockWeights:
    return ZImageBlockWeights(
        _t1(_in(prefix + "_n1"), D, ctx),
        _t2(_in(prefix + "_wq"), D, D, ctx),
        _t2(_in(prefix + "_wk"), D, D, ctx),
        _t2(_in(prefix + "_wv"), D, D, ctx),
        _t2(_in(prefix + "_wo"), D, D, ctx),
        _t1(_in(prefix + "_q_norm"), Dh, ctx),
        _t1(_in(prefix + "_k_norm"), Dh, ctx),
        _t1(_in(prefix + "_n2"), D, ctx),
        _t1(_in(prefix + "_fn1"), D, ctx),
        _t2(_in(prefix + "_w1"), F, D, ctx),
        _t2(_in(prefix + "_w3"), F, D, ctx),
        _t2(_in(prefix + "_w2"), D, F, ctx),
        _t1(_in(prefix + "_fn2"), D, ctx),
    )


def _load_mod(prefix: String) raises -> ZImageModVecs:
    return ZImageModVecs(
        _in(prefix + "_scale_msa"), _in(prefix + "_gate_msa"),
        _in(prefix + "_scale_mlp"), _in(prefix + "_gate_mlp"),
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
    print("==== zimage stack_parity (Z-Image FULL STACK composition vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " S=", S, " F=", F, " NR=", NUM_NR, " CR=", NUM_CR, " MAIN=", NUM_MAIN)

    var x_seq = _in("sin_x_seq")
    var cap_seq = _in("sin_cap_seq")
    var f_scale = _in("sin_f_scale")
    var final_lin_w = Tensor.from_host(_in("sin_final_lin"), [OUT_CH, D], STDtype.F32, ctx)
    var final_lin_b = Tensor.from_host(_in("sin_final_lin_b"), [OUT_CH], STDtype.F32, ctx)

    var nr_blocks = List[ZImageBlockWeights]()
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_blocks.append(_load_block(String("sin_nr") + String(i), ctx))
        nr_mod.append(_load_mod(String("sin_nr") + String(i)))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(_load_block(String("sin_cr") + String(i), ctx))
    var main_blocks = List[ZImageBlockWeights]()
    var main_mod = List[ZImageModVecs]()
    for i in range(NUM_MAIN):
        main_blocks.append(_load_block(String("sin_main") + String(i), ctx))
        main_mod.append(_load_mod(String("sin_main") + String(i)))

    var x_cos = Tensor.from_host(_in("sin_x_cos"), [N_IMG * H, HALF], STDtype.F32, ctx)
    var x_sin = Tensor.from_host(_in("sin_x_sin"), [N_IMG * H, HALF], STDtype.F32, ctx)
    var cap_cos = Tensor.from_host(_in("sin_cap_cos"), [N_TXT * H, HALF], STDtype.F32, ctx)
    var cap_sin = Tensor.from_host(_in("sin_cap_sin"), [N_TXT * H, HALF], STDtype.F32, ctx)
    var uni_cos = Tensor.from_host(_in("sin_uni_cos"), [S * H, HALF], STDtype.F32, ctx)
    var uni_sin = Tensor.from_host(_in("sin_uni_sin"), [S * H, HALF], STDtype.F32, ctx)

    # ── forward ──
    var fwd = zimage_stack_forward[H, Dh, N_IMG, N_TXT, S](
        x_seq.copy(), cap_seq.copy(),
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod,
        f_scale.copy(), final_lin_w, final_lin_b,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("sref_out"), allok)

    # ── backward ──
    var d_out = _in("sin_d_out")
    var g = zimage_stack_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod,
        f_scale.copy(), final_lin_w,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin, fwd,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )

    print("")
    print("---- input-token grads vs torch (full-chain proof, both streams) ----")
    _check(harness, "d_x_seq  ", g.d_x_seq, _in("sref_d_x_seq"), allok)
    _check(harness, "d_cap_seq", g.d_cap_seq, _in("sref_d_cap_seq"), allok)

    print("")
    print("---- final-layer grads vs torch ----")
    _check(harness, "d_f_scale  ", g.d_f_scale, _in("sref_d_f_scale"), allok)
    _check(harness, "d_final_lin", g.d_final_lin, _in("sref_d_final_lin"), allok)

    print("")
    print("---- main-layer weight grads (deepest + shallowest) vs torch ----")
    _check(harness, "d_main_deep_wq   ", g.main_grads[NUM_MAIN - 1].d_wq, _in("sref_d_main_deep_wq"), allok)
    _check(harness, "d_main_deep_w2   ", g.main_grads[NUM_MAIN - 1].d_w2, _in("sref_d_main_deep_w2"), allok)
    _check(harness, "d_main_shallow_wq", g.main_grads[0].d_wq, _in("sref_d_main_shallow_wq"), allok)

    print("")
    print("---- refiner weight grads (noise + context) vs torch ----")
    _check(harness, "d_nr0_wq", g.nr_grads[0].d_wq, _in("sref_d_nr0_wq"), allok)
    _check(harness, "d_nr0_w2", g.nr_grads[0].d_w2, _in("sref_d_nr0_w2"), allok)
    _check(harness, "d_cr0_wq", g.cr_grads[0].d_wq, _in("sref_d_cr0_wq"), allok)
    _check(harness, "d_cr0_w2", g.cr_grads[0].d_w2, _in("sref_d_cr0_w2"), allok)

    print("")
    print("---- per-block RAW mod-vec grads [4D] vs torch ----")
    var nr0_mod = _pack4(g.nr_grads[0])
    var main_deep_mod = _pack4(g.main_grads[NUM_MAIN - 1])
    _check(harness, "d_nr0_mod      ", nr0_mod, _in("sref_d_nr0_mod"), allok)
    _check(harness, "d_main_deep_mod", main_deep_mod, _in("sref_d_main_deep_mod"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Z-Image full-stack composition fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")


# pack a modulated block's 4 RAW mod-vec grads [4D] (scale_msa|gate_msa|scale_mlp|gate_mlp)
def _pack4(g: ZImageBlockGrads) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(g.d_scale_msa)):
        o.append(g.d_scale_msa[i])
    for i in range(len(g.d_gate_msa)):
        o.append(g.d_gate_msa[i])
    for i in range(len(g.d_scale_mlp)):
        o.append(g.d_scale_mlp[i])
    for i in range(len(g.d_gate_mlp)):
        o.append(g.d_gate_mlp[i])
    return o^
