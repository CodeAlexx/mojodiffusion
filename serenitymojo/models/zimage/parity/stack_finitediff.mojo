# serenitymojo/models/zimage/parity/stack_finitediff.mojo
#
# #1 RISK GATE — x-path FINITE-DIFFERENCE SELF-CONSISTENCY for the Z-Image full
# stack (models/zimage/zimage_stack.mojo). This is the Klein composition-defect
# check (project_klein_runaway_composition_backward): per-block-correct does NOT
# prove composed-backward-correct, and a torch oracle can mask a composition bug
# if the Mojo forward and the torch forward share the same wrong assumption. So
# this gate compares the Mojo stack's ANALYTIC backward against a NUMERICAL
# gradient computed from the Mojo stack's OWN forward — NO torch in the loop.
#
# For a scalar loss  L = sum(out * d_out)  (out = zimage_stack_forward(...).out),
# the analytic grad into the x_seq input is g.d_x_seq (returned by the backward).
# The numerical grad at element i is the central difference
#   d_num[i] = (L(x_seq + h*e_i) - L(x_seq - h*e_i)) / (2h).
# If the composed backward = grad of the composed forward, d_num[i] ≈ g.d_x_seq[i]
# for every probed i. We probe a spread of indices across the [N_IMG,D] input and
# report the WORST |ratio-1| (ratio = analytic/numerical). PASS if worst small.
#
# Reuses the SAME inputs the composition gate uses (stack_oracle.py dumps them),
# so run that oracle first (only for the inputs; the references are not read here).
#
# Run (oracle FIRST, SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/zimage/parity/stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/zimage/parity/stack_finitediff.mojo -o /tmp/zimage_stack_fd
#   /tmp/zimage_stack_fd

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from math import abs as fabs
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.zimage_stack import (
    ZImageStackForward, ZImageStackGrads,
    zimage_stack_forward, zimage_stack_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/zimage/parity/"

comptime H = 30
comptime Dh = 128
comptime D = H * Dh        # 3840
comptime N_IMG = 6
comptime N_TXT = 4
comptime S = N_IMG + N_TXT # 10
comptime F = 96
comptime OUT_CH = 16
comptime HALF = Dh // 2
comptime EPS = Float32(1e-05)
comptime FINAL_EPS = Float32(1e-06)
comptime NUM_NR = 1
comptime NUM_CR = 1
comptime NUM_MAIN = 2
comptime FD_H = Float32(2.0e-3)   # central-difference step (F32 interior)


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


# scalar loss L = sum(out * d_out) from a forward at the given x_seq.
def _loss(
    x_seq: List[Float32], cap_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    f_scale: List[Float32], final_lin_w: Tensor, final_lin_b: Tensor,
    x_cos: Tensor, x_sin: Tensor, cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor, d_out: List[Float32],
    ctx: DeviceContext,
) raises -> Float64:
    var fwd = zimage_stack_forward[H, Dh, N_IMG, N_TXT, S](
        x_seq.copy(), cap_seq.copy(),
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod,
        f_scale.copy(), final_lin_w, final_lin_b,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var acc = Float64(0.0)
    for i in range(len(fwd.out)):
        acc += Float64(fwd.out[i]) * Float64(d_out[i])
    return acc


def main() raises:
    var ctx = DeviceContext()
    print("==== zimage stack_finitediff (x-path self-consistency, NO torch) ====")
    print("H=", H, " D=", D, " S=", S, " NR=", NUM_NR, " CR=", NUM_CR,
          " MAIN=", NUM_MAIN, " fd_h=", FD_H)

    var x_seq = _in("sin_x_seq")
    var cap_seq = _in("sin_cap_seq")
    var f_scale = _in("sin_f_scale")
    var final_lin_w = Tensor.from_host(_in("sin_final_lin"), [OUT_CH, D], STDtype.F32, ctx)
    var final_lin_b = Tensor.from_host(_in("sin_final_lin_b"), [OUT_CH], STDtype.F32, ctx)
    var d_out = _in("sin_d_out")

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

    # ── analytic backward (the thing under test) ──
    var fwd = zimage_stack_forward[H, Dh, N_IMG, N_TXT, S](
        x_seq.copy(), cap_seq.copy(),
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod,
        f_scale.copy(), final_lin_w, final_lin_b,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var g = zimage_stack_backward[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod,
        f_scale.copy(), final_lin_w,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin, fwd,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )

    # ── probe a spread of x_seq elements across rows and channels ──
    # indices into the flattened [N_IMG, D] x_seq (row-major).
    var probes = List[Int]()
    probes.append(0)                       # row 0, ch 0
    probes.append(7)                       # row 0, ch 7
    probes.append(1 * D + 100)             # row 1, ch 100
    probes.append(2 * D + 1234)            # row 2, ch 1234
    probes.append(3 * D + 2048)            # row 3, ch 2048
    probes.append(4 * D + 3000)            # row 4, ch 3000
    probes.append(5 * D + (D - 1))         # row 5 (last), last ch

    var h = Float64(FD_H)
    print("")
    print("  idx        analytic         numerical        |ana-num|")
    # The composition gate is a VECTOR self-consistency test: does the analytic
    # gradient VECTOR over the probe set match the numerical one? A per-element
    # ratio is meaningless where a single component's gradient is near zero (F32
    # storage of x_seq + central-diff truncation at fd_h then dominate that one
    # tiny number). So we report BOTH: the per-element |ana-num| (absolute), and
    # a VECTOR-relative error  ||ana-num|| / ||ana||  over all probes — the latter
    # is the verdict (the Klein composition defect shows ~0.33 here; a correct
    # composition sits at the F32 truncation floor, a few e-3).
    var sum_diff2 = Float64(0.0)
    var sum_ana2 = Float64(0.0)
    var worst_abs = Float64(0.0)
    var worst_i = 0
    for pi in range(len(probes)):
        var idx = probes[pi]
        var xp = x_seq.copy()
        xp[idx] = xp[idx] + Float32(h)
        var lp = _loss(
            xp, cap_seq, nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod,
            f_scale, final_lin_w, final_lin_b,
            x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin, d_out, ctx,
        )
        var xm = x_seq.copy()
        xm[idx] = xm[idx] - Float32(h)
        var lm = _loss(
            xm, cap_seq, nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod,
            f_scale, final_lin_w, final_lin_b,
            x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin, d_out, ctx,
        )
        var num = (lp - lm) / (2.0 * h)
        var ana = Float64(g.d_x_seq[idx])
        var diff = fabs(ana - num)
        print("  ", idx, "   ", ana, "   ", num, "   ", diff)
        sum_diff2 += diff * diff
        sum_ana2 += ana * ana
        if diff > worst_abs:
            worst_abs = diff
            worst_i = idx

    var denom = sum_ana2 ** 0.5
    if denom < 1e-12:
        denom = 1e-12
    var vec_rel = (sum_diff2 ** 0.5) / denom
    print("")
    print("  worst |ana-num| =", worst_abs, " at idx", worst_i)
    print("  VECTOR-relative err  ||ana-num|| / ||ana|| =", vec_rel)
    # central diff at fd_h=2e-3 through a deep composed F32 graph: ~e-3 floor.
    # PASS threshold 2e-2 (well above the floor, well below the ~0.33 Klein
    # composition-defect signature).
    if vec_rel < 0.02:
        print("VERDICT: PASS — composed backward = grad of composed forward (ratio≈1.0)")
    else:
        print("VERDICT: FAIL — composed backward DIVERGES from numerical grad (composition bug)")
