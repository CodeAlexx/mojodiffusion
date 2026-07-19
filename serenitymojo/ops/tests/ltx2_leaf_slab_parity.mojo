# ops/tests/ltx2_leaf_slab_parity.mojo — L5 leaf-slab BIT gate: each new _slab
# leaf op must be BYTE-IDENTICAL to its non-slab twin (same kernel, only the
# output alloc source differs — slab.alloc vs enqueue_create_buffer). Bar:
# n_mismatch=0. Grows as leaf slabs land (gelu fwd/bwd here; the gate op has its
# own gate ltx2_gate_backward_parity).
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/ops/tests/ltx2_leaf_slab_parity.mojo -o /tmp/ltx2_leaf_slab_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/ltx2_leaf_slab_parity

from std.math import sin, cos, sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.ops.activations import gelu, gelu_slab
from serenitymojo.ops.activation_backward import gelu_backward, gelu_backward_slab
from serenitymojo.ops.tensor_algebra import slice, slice_slab
from serenitymojo.ops.attention import sdpa_cross_nomask, sdpa_cross_nomask_slab
from serenitymojo.models.dit.ltx2_rope import apply_ltx2_rope, apply_ltx2_rope_slab


def _fill(n: Int, phase: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(0.04) * (sin(Float32(0.8) * Float32(i) + phase)
                   + Float32(0.5) * cos(Float32(1.5) * Float32(i))))
    return out^


def _cmp(name: String, a: List[Float32], b: List[Float32], mut allok: Bool):
    var nm = 0
    var nz = False
    for i in range(len(a)):
        if a[i] != b[i]:
            nm += 1
        if a[i] != Float32(0.0):
            nz = True
    if nm != 0 or not nz:
        allok = False
    print("  ", "PASS" if (nm == 0 and nz) else "FAIL", name, " n_mismatch=", nm, " nonzero=", nz)


def _run_cross[Sq: Int, Skv: Int, H: Int, Dh: Int](
    name: String, dt: STDtype, ctx: DeviceContext, mut allok: Bool
) raises:
    # sdpa_cross_nomask math-branch (matmul-backed) slab twin — the ONLY branch
    # reachable for LTX2 (score_mib << 3584). q4 [1,Sq,H,Dh], k4/v4 [1,Skv,H,Dh],
    # non-degenerate sinusoidal fills through QKᵀ/softmax/P@V.
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var q = Tensor.from_host(_fill(Sq * H * Dh, Float32(0.23)), [1, Sq, H, Dh], dt, ctx)
    var k = Tensor.from_host(_fill(Skv * H * Dh, Float32(0.37)), [1, Skv, H, Dh], dt, ctx)
    var v = Tensor.from_host(_fill(Skv * H * Dh, Float32(0.71)), [1, Skv, H, Dh], dt, ctx)
    # Size the slab for the F32 path (esz=bsz=4 — covers bf16): q/out_f32/out_buf
    # over Sq rows, k/v over Skv rows, F32 scores [BH,Sq,Skv], + headroom.
    comptime SB = (
        H * Sq * Dh * 4 + 2 * (H * Skv * Dh * 4) + H * Sq * Skv * 4
        + H * Sq * Dh * 4 + Sq * H * Dh * 4 + 65536)
    var slab = StepSlab(ctx, SB)
    var hc = sdpa_cross_nomask[1, Sq, Skv, H, Dh](q, k, v, scale, ctx).to_host(ctx)
    var sl = sdpa_cross_nomask_slab[1, Sq, Skv, H, Dh](q, k, v, scale, ctx, slab).to_host(ctx)
    _cmp(name, hc, sl, allok)


def _run(dt: STDtype, tag: String, ctx: DeviceContext, mut allok: Bool) raises:
    var N = 4096
    var x = Tensor.from_host(_fill(N, Float32(0.17)), [N], dt, ctx)
    var g = Tensor.from_host(_fill(N, Float32(0.53)), [N], dt, ctx)
    var slab = StepSlab(ctx, N * 4 + 8192)
    _cmp("gelu " + tag, gelu(x, ctx).to_host(ctx), gelu_slab(x, ctx, slab).to_host(ctx), allok)
    var slab2 = StepSlab(ctx, N * 4 + 8192)
    _cmp("gelu_bwd " + tag, gelu_backward(g, x, ctx).to_host(ctx),
         gelu_backward_slab(g, x, ctx, slab2).to_host(ctx), allok)

    # slice — rank-2/dim-1 fast path (_slice_dim1_rank2_kerneled_slab)
    var xs = Tensor.from_host(_fill(8 * 64, Float32(0.31)), [8, 64], dt, ctx)
    var slab3 = StepSlab(ctx, N * 4 + 8192)
    _cmp("slice_fast " + tag,
         slice(xs, 1, 16, 32, ctx).to_host(ctx),
         slice_slab(xs, 1, 16, 32, ctx, slab3).to_host(ctx), allok)

    # slice — general strided gather path (rank-3, dim=1: outer=4, inner=16)
    var xg = Tensor.from_host(_fill(4 * 8 * 16, Float32(0.44)), [4, 8, 16], dt, ctx)
    var slab4 = StepSlab(ctx, N * 4 + 8192)
    _cmp("slice_gen " + tag,
         slice(xg, 1, 2, 4, ctx).to_host(ctx),
         slice_slab(xg, 1, 2, 4, ctx, slab4).to_host(ctx), allok)

    # apply_ltx2_rope — half-split RoPE (rope_halfsplit_slab), F32 cos/sin tables
    var xr = Tensor.from_host(_fill(32 * 64, Float32(0.19)), [32, 64], dt, ctx)
    var cr = Tensor.from_host(_fill(32 * 32, Float32(0.61)), [32, 32], STDtype.F32, ctx)
    var sr = Tensor.from_host(_fill(32 * 32, Float32(0.83)), [32, 32], STDtype.F32, ctx)
    var slab5 = StepSlab(ctx, N * 4 + 8192)
    _cmp("ltx2_rope " + tag,
         apply_ltx2_rope(xr, cr, sr, ctx).to_host(ctx),
         apply_ltx2_rope_slab(xr, cr, sr, ctx, slab5).to_host(ctx), allok)

    # sdpa_cross_nomask math-branch slab twin — real LTX2 video-block cross dims
    # (H=32, Dh=128): attn2 cross Sq=576/Skv=1024 and a smaller Sq=256 case.
    _run_cross[256, 1024, 32, 128]("sdpa_cross[256x1024] " + tag, dt, ctx, allok)
    _run_cross[576, 1024, 32, 128]("sdpa_cross[576x1024] " + tag, dt, ctx, allok)


def main() raises:
    print("=== ltx2 leaf-slab BIT parity (slab vs non-slab) ===")
    var ctx = DeviceContext()
    var allok = True
    _run(STDtype.BF16, "bf16", ctx, allok)
    _run(STDtype.F32, "f32", ctx, allok)
    if allok:
        print("GATE ltx2_leaf_slab_parity: ALL PASS (bit-exact)")
    else:
        print("GATE ltx2_leaf_slab_parity: FAIL")
        raise Error("ltx2_leaf_slab_parity FAILED")
