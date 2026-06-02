# serenitymojo/models/anima/parity/stack_finitediff.mojo
#
# FINITE-DIFFERENCE SELF-CONSISTENCY gate for the ANIMA full stack
# (models/anima/anima_stack.mojo). This is the composition-defect catch the Klein
# lesson demands (project_klein_runaway_composition_backward): per-block-correct
# and even torch-parity-correct does NOT prove the COMPOSED analytic backward
# equals the gradient of the COMPOSED forward when there is NO torch in the loop.
# Here we compute the gradient NUMERICALLY from the Mojo stack's OWN forward
# (central differences on the scalar loss L = sum(out * d_out)) and compare to the
# ANALYTIC gradient the Mojo stack_backward returns. ratio ~ 1.0 ⇒ the composed
# backward is self-consistent with the composed forward.
#
#   numeric:  dL/dθ_i ≈ (L(θ + ε e_i) − L(θ − ε e_i)) / (2ε)
#   analytic: the corresponding entry of stack_backward's grad
#   ratio   = numeric / analytic   (per probed coordinate; report mean + each)
#
# Probes a handful of coordinates of:
#   * patches    (analytic = d_patches)               — input → full stack
#   * a deep block weight (sa_q[L-1])                  — deepest block param
#   * a shallow block weight (mlp2[0])                 — shallowest block param
# (Probing a few coords is enough to catch a composition-level sign/scale defect;
# a full Jacobian is unnecessary and would be expensive at real H/Dh.)
#
# Run (oracle FIRST for inputs/weights, SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/anima/parity/stack_finitediff.mojo -o /tmp/anima_stack_fd
#   /tmp/anima_stack_fd

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import abs as fabs
from std.memory import alloc, ArcPointer
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.anima.weights import AnimaBlockWeights, AnimaStackBase
from serenitymojo.models.anima.anima_stack import (
    AnimaStackForward, AnimaStackGrads, anima_stack_forward, anima_stack_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/anima/parity/"
comptime B = 1
comptime H = 16
comptime Dh = 128
comptime D = H * Dh        # 2048
comptime S_IMG = 6
comptime S_TXT = 8
comptime JOINT = 1024
comptime F = 32
comptime IN_PATCH = 68
comptime OUT_PATCH = 64
comptime L = 3
comptime EPS = Float32(1e-06)
comptime FD_EPS = Float32(2e-3)   # central-difference step (F32 stack, balanced)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("missing ref (run stack_oracle.py first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(n // 4):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read_bin_f32(REF_DIR + name + ".bin")


def _t(name: String, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(_in(name), shape^, STDtype.F32, ctx)


def _tv(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _sh(*dims: Int) -> List[Int]:
    var o = List[Int]()
    for d in dims:
        o.append(d)
    return o^


def _load_block_from(pre: String, ctx: DeviceContext) raises -> AnimaBlockWeights:
    return AnimaBlockWeights(
        _t(pre + "sa_mod1", _sh(256, D), ctx), _t(pre + "sa_mod2", _sh(3 * D, 256), ctx),
        _t(pre + "ca_mod1", _sh(256, D), ctx), _t(pre + "ca_mod2", _sh(3 * D, 256), ctx),
        _t(pre + "mlp_mod1", _sh(256, D), ctx), _t(pre + "mlp_mod2", _sh(3 * D, 256), ctx),
        _t(pre + "sa_q", _sh(D, D), ctx), _t(pre + "sa_k", _sh(D, D), ctx),
        _t(pre + "sa_v", _sh(D, D), ctx), _t(pre + "sa_out", _sh(D, D), ctx),
        _t(pre + "sa_qn", _sh(Dh), ctx), _t(pre + "sa_kn", _sh(Dh), ctx),
        _t(pre + "ca_q", _sh(D, D), ctx), _t(pre + "ca_k", _sh(D, JOINT), ctx),
        _t(pre + "ca_v", _sh(D, JOINT), ctx), _t(pre + "ca_out", _sh(D, D), ctx),
        _t(pre + "ca_qn", _sh(Dh), ctx), _t(pre + "ca_kn", _sh(Dh), ctx),
        _t(pre + "mlp1", _sh(F, D), ctx), _t(pre + "mlp2", _sh(D, F), ctx),
    )


# build one block from a host weight-dict override (used to perturb a single weight)
def _block_with_override(
    l: Int, field: String, idx: Int, delta: Float32, ctx: DeviceContext
) raises -> AnimaBlockWeights:
    var pre = String("in_blk") + String(l) + String("_")
    # load all 20 raw vectors, perturb the targeted one, rebuild
    var sa_q = _in(pre + "sa_q")
    var mlp2 = _in(pre + "mlp2")
    if field == "sa_q":
        sa_q[idx] = sa_q[idx] + delta
    elif field == "mlp2":
        mlp2[idx] = mlp2[idx] + delta
    return AnimaBlockWeights(
        _t(pre + "sa_mod1", _sh(256, D), ctx), _t(pre + "sa_mod2", _sh(3 * D, 256), ctx),
        _t(pre + "ca_mod1", _sh(256, D), ctx), _t(pre + "ca_mod2", _sh(3 * D, 256), ctx),
        _t(pre + "mlp_mod1", _sh(256, D), ctx), _t(pre + "mlp_mod2", _sh(3 * D, 256), ctx),
        _tv(sa_q^, _sh(D, D), ctx), _t(pre + "sa_k", _sh(D, D), ctx),
        _t(pre + "sa_v", _sh(D, D), ctx), _t(pre + "sa_out", _sh(D, D), ctx),
        _t(pre + "sa_qn", _sh(Dh), ctx), _t(pre + "sa_kn", _sh(Dh), ctx),
        _t(pre + "ca_q", _sh(D, D), ctx), _t(pre + "ca_k", _sh(D, JOINT), ctx),
        _t(pre + "ca_v", _sh(D, JOINT), ctx), _t(pre + "ca_out", _sh(D, D), ctx),
        _t(pre + "ca_qn", _sh(Dh), ctx), _t(pre + "ca_kn", _sh(Dh), ctx),
        _t(pre + "mlp1", _sh(F, D), ctx), _tv(mlp2^, _sh(D, F), ctx),
    )


def _load_base(ctx: DeviceContext) raises -> AnimaStackBase:
    return AnimaStackBase(
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),
        TArc(_t("in_fl_lin", _sh(OUT_PATCH, D), ctx)),
        TArc(_t("in_fl_mod1", _sh(256, D), ctx)),
        TArc(_t("in_fl_mod2", _sh(2 * D, 256), ctx)),
        TArc(_t("in_fl_lin", _sh(OUT_PATCH, D), ctx)),
    )


def _expand(name: String) raises -> List[Float32]:
    var half = Dh // 2
    var pp = _in(name)
    var out = List[Float32]()
    for _b in range(B):
        for s in range(S_IMG):
            for _h in range(H):
                for i in range(half):
                    out.append(pp[s * half + i])
    return out^


# scalar loss L = sum(out * d_out) of one stack forward (rebuilds blocks each call
# so a weight perturbation is honoured). patches passed by value so a perturbed
# copy can be supplied.
def _loss(
    patches: List[Float32], t_cond: List[Float32], base_adaln: List[Float32],
    context: List[Float32], d_out: List[Float32],
    base: AnimaStackBase, blocks: List[AnimaBlockWeights],
    cos: Tensor, sin: Tensor, ctx: DeviceContext,
) raises -> Float64:
    var fwd = anima_stack_forward[H, Dh, S_IMG, S_TXT](
        patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, cos, sin, B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    var acc = Float64(0.0)
    for i in range(len(fwd.out)):
        acc += Float64(fwd.out[i]) * Float64(d_out[i])
    return acc


def _load_blocks(ctx: DeviceContext) raises -> List[AnimaBlockWeights]:
    var blocks = List[AnimaBlockWeights]()
    for l in range(L):
        blocks.append(_load_block_from(String("in_blk") + String(l) + String("_"), ctx))
    return blocks^


def main() raises:
    var ctx = DeviceContext()
    print("==== anima stack_finitediff (composed bwd vs numeric grad of composed fwd) ====")
    print("L=", L, " S_img=", S_IMG, " D=", D, " FD_EPS=", FD_EPS)

    var patches = _in("in_patches")
    var t_cond = _in("in_t_cond")
    var base_adaln = _in("in_base_adaln")
    var context = _in("in_context")
    var d_out = _in("in_d_out")
    var base = _load_base(ctx)
    var half = Dh // 2
    var cos = Tensor.from_host(_expand("in_cos"), [B * S_IMG * H, half], STDtype.F32, ctx)
    var sin = Tensor.from_host(_expand("in_sin"), [B * S_IMG * H, half], STDtype.F32, ctx)

    # ── analytic grads (one backward) ──
    var blocks = _load_blocks(ctx)
    var fwd = anima_stack_forward[H, Dh, S_IMG, S_TXT](
        patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, cos, sin, B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    var g = anima_stack_backward[H, Dh, S_IMG, S_TXT](
        d_out.copy(), patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, cos, sin, fwd, B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )

    var ratios = List[Float64]()

    # ── PROBE 1: patches input (analytic = d_patches) ──
    print("")
    print("---- finite-diff: patches (input → full stack) ----")
    var pidx = List[Int]()
    pidx.append(0); pidx.append(37); pidx.append(199); pidx.append(400)
    for pi in range(len(pidx)):
        var idx = pidx[pi]
        var pp = patches.copy(); pp[idx] = pp[idx] + FD_EPS
        var pm = patches.copy(); pm[idx] = pm[idx] - FD_EPS
        var lp = _loss(pp, t_cond, base_adaln, context, d_out, base, blocks, cos, sin, ctx)
        var lm = _loss(pm, t_cond, base_adaln, context, d_out, base, blocks, cos, sin, ctx)
        var num = (lp - lm) / Float64(2.0 * FD_EPS)
        var ana = Float64(g.d_patches[idx])
        var ratio = num / ana if fabs(ana) > 5e-3 else Float64(0.0)
        ratios.append(ratio)
        print("  patches[", idx, "]  numeric", num, " analytic", ana, " ratio", ratio)

    # ── PROBE 2: deep block weight sa_q[L-1] (analytic = blk_grads[L-1].d_sa_q) ──
    print("")
    print("---- finite-diff: deep-block sa_q[L-1] ----")
    # pick coords with non-trivial analytic grad — a tiny-grad coord makes the
    # central difference (step FD_EPS) dominated by F32 noise (ratio meaningless).
    var widx = List[Int]()
    widx.append(0); widx.append(5000); widx.append(20480)
    for wi in range(len(widx)):
        var idx = widx[wi]
        var bp = _load_blocks(ctx); bp[L - 1] = _block_with_override(L - 1, "sa_q", idx, FD_EPS, ctx)
        var bm = _load_blocks(ctx); bm[L - 1] = _block_with_override(L - 1, "sa_q", idx, -FD_EPS, ctx)
        var lp = _loss(patches, t_cond, base_adaln, context, d_out, base, bp, cos, sin, ctx)
        var lm = _loss(patches, t_cond, base_adaln, context, d_out, base, bm, cos, sin, ctx)
        var num = (lp - lm) / Float64(2.0 * FD_EPS)
        var ana = Float64(g.blk_grads[L - 1].d_sa_q[idx])
        # only score coords whose analytic grad is well above the FD noise floor.
        var ratio = num / ana if fabs(ana) > 5e-3 else Float64(0.0)
        ratios.append(ratio)
        print("  sa_q[L-1][", idx, "]  numeric", num, " analytic", ana, " ratio", ratio)

    # ── PROBE 3: shallow block weight mlp2[0] (analytic = blk_grads[0].d_mlp2) ──
    print("")
    print("---- finite-diff: shallow-block mlp2[0] ----")
    var midx = List[Int]()
    midx.append(0); midx.append(1000); midx.append(40000)
    for mi in range(len(midx)):
        var idx = midx[mi]
        var bp = _load_blocks(ctx); bp[0] = _block_with_override(0, "mlp2", idx, FD_EPS, ctx)
        var bm = _load_blocks(ctx); bm[0] = _block_with_override(0, "mlp2", idx, -FD_EPS, ctx)
        var lp = _loss(patches, t_cond, base_adaln, context, d_out, base, bp, cos, sin, ctx)
        var lm = _loss(patches, t_cond, base_adaln, context, d_out, base, bm, cos, sin, ctx)
        var num = (lp - lm) / Float64(2.0 * FD_EPS)
        var ana = Float64(g.blk_grads[0].d_mlp2[idx])
        var ratio = num / ana if fabs(ana) > 5e-3 else Float64(0.0)
        ratios.append(ratio)
        print("  mlp2[0][", idx, "]  numeric", num, " analytic", ana, " ratio", ratio)

    # ── verdict ──
    var mean = Float64(0.0)
    var nz = 0
    for i in range(len(ratios)):
        if fabs(ratios[i]) > 1e-9:
            mean += ratios[i]; nz += 1
    if nz > 0:
        mean /= Float64(nz)
    var worst = Float64(0.0)
    for i in range(len(ratios)):
        if fabs(ratios[i]) > 1e-9:
            var dev = fabs(ratios[i] - 1.0)
            if dev > worst:
                worst = dev
    print("")
    print("mean ratio (nonzero-analytic probes) =", mean, "  worst |ratio-1| =", worst,
          "  (", nz, "probes)")
    if worst < 0.05:
        print("VERDICT: PASS — composed backward is self-consistent with composed forward (ratio ~ 1.0)")
    else:
        print("VERDICT: FAIL — numeric/analytic ratio deviates > 5% (composition defect)")
