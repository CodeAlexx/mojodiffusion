# autograd_v2/tests/krea2_mlp_seg_peak.mojo — MEASURE the krea2 MLP-branch slab
# peaks for activation checkpointing: the WHOLE mlp segment (K=2 worst case) and
# the two K=3 sub-segments (split at swiglu). The mlp branch holds the [1,L,16384]
# swiglu tensors — the biggest in the block — so it sets the per-segment peak.
#
# All three measured in one binary (fresh slab + Graph each), bf16 (production),
# DEVICE-grad proj_lora (record_proj_lora_slab, A/B as engine leaves = the
# engine/True carrier). L=2432 (half the trainer's 4864); L=4864 ≈ 2x (linear-in-L).
#
# Decision: K=2 if the whole-mlp peak fits ~6GB at L=4864; else K=3 (the larger
# sub-segment must fit). 12GB fp8 base + slab + 1.3GB cond + ~2GB working ≤ 24GB.
#
# Build:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/krea2_mlp_seg_peak.mojo -o /tmp/krea2_mlp_seg_peak

from std.gpu.host import DeviceContext
from std.collections import Optional, Dict
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.klein.lora_adapter import make_lora_adapter
from serenitymojo.models.klein.lora_block import lora_adapter_to_device
from serenitymojo.models.zimage.lora_block import ZImageLoraAdapterDevice
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute_slab
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.autograd_v2.ops_record import (
    record_rms_norm_dx_slab, record_modulate_slab, record_proj_lora_slab,
    record_swiglu_slab, record_residual_gate_slab,
    record_add_slab, record_mul_slab, record_repeat_kv_slab,
    record_sigmoid_slab, record_sdpa_nomask_slab, record_rope_slab, record_reshape,
)
from serenitymojo.ops.tensor_algebra import zeros_device_slab
from serenitymojo.models.dit.krea2_dit import _tile_rope_table
from serenitymojo.ops.tensor_algebra import add_scalar, zeros_device
from std.math import sqrt

comptime TArcT = ArcPointer[Tensor]
comptime FEATURES = 6144
comptime MLPDIM = 16384
comptime L = 2432
comptime EPS = Float32(1.0e-5)
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime GB = Float64(1024.0 * 1024.0 * 1024.0)


def _bf(var shape: List[Int], seed: UInt64, sc: Float32, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(mul_scalar(randn(shape^, seed, STDtype.F32, ctx), sc, ctx), STDtype.BF16, ctx)


def _mk_lora(in_f: Int, out_f: Int, seed: UInt64, ctx: DeviceContext) raises -> ZImageLoraAdapterDevice:
    var dev = lora_adapter_to_device(make_lora_adapter(RANK, ALPHA, in_f, out_f, seed), ctx)
    var b = _bf(dev.b[].shape(), seed + UInt64(9000), Float32(0.04), ctx)
    return ZImageLoraAdapterDevice(dev.a.copy(), TArc(b^), dev.rank, dev.in_f, dev.out_f, dev.scale)


def _leaf(mut g: Graph, var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> TArc:
    var t = _bf(shape^, seed, Float32(0.5), ctx)
    t.set_id(g.fresh_tensor_id())
    _ = g.leaf(t.id)
    return TArc(t^)


def _seed(t: Tensor, ctx: DeviceContext) raises -> TArcT:
    return TArcT(Tensor(t.buf.copy(), t.shape(), t.dtype()))


def _report(name: String, slab: StepSlab) -> None:
    print(name, ": peak =", Float64(slab.peak_bytes()) / GB, "GB ,", slab.n_allocs, "allocs  → L=4864 ≈",
          Float64(slab.peak_bytes()) * 2.0 / GB, "GB")


# ── K=2: the WHOLE mlp segment (x1 → ... → x2) ──────────────────────────────
def measure_whole_mlp(ctx: DeviceContext) raises:
    var g = Graph()
    var x1 = _leaf(g, [1, L, FEATURES], UInt64(40), ctx)
    var postscale = TArc(_bf([FEATURES], UInt64(31), Float32(0.1), ctx))
    var postshift = TArc(_bf([FEATURES], UInt64(32), Float32(0.1), ctx))
    var postgate = TArc(_bf([FEATURES], UInt64(33), Float32(0.1), ctx))
    var postnorm_w = TArc(_bf([FEATURES], UInt64(34), Float32(0.1), ctx))
    var mg_a = g.fresh_tensor_id(); var mg_b = g.fresh_tensor_id()
    var mu_a = g.fresh_tensor_id(); var mu_b = g.fresh_tensor_id()
    var md_a = g.fresh_tensor_id(); var md_b = g.fresh_tensor_id()
    var lo_mg = _mk_lora(FEATURES, MLPDIM, UInt64(105), ctx)
    var lo_mu = _mk_lora(FEATURES, MLPDIM, UInt64(106), ctx)
    var lo_md = _mk_lora(MLPDIM, FEATURES, UInt64(107), ctx)
    var w_mg = TArc(_bf([MLPDIM, FEATURES], UInt64(6), Float32(0.04), ctx))
    var w_mu = TArc(_bf([MLPDIM, FEATURES], UInt64(7), Float32(0.04), ctx))
    var w_md = TArc(_bf([FEATURES, MLPDIM], UInt64(8), Float32(0.04), ctx))
    var d_out = _bf([1, L, FEATURES], UInt64(50), Float32(0.5), ctx)
    var slab = StepSlab(ctx, 8 * 1024 * 1024 * 1024)

    var xn2 = record_rms_norm_dx_slab(g, x1, postnorm_w, EPS, ctx, slab)
    var xm2 = record_modulate_slab(g, xn2, postscale, postshift, 0, ctx, slab)
    var mg = record_proj_lora_slab(g, xm2, w_mg, lo_mg, mg_a, mg_b, L, FEATURES, MLPDIM, ctx, slab)
    var mu = record_proj_lora_slab(g, xm2, w_mu, lo_mu, mu_a, mu_b, L, FEATURES, MLPDIM, ctx, slab)
    var sw = record_swiglu_slab(g, mg, mu, ctx, slab)
    var m = record_proj_lora_slab(g, sw, w_md, lo_md, md_a, md_b, L, MLPDIM, FEATURES, ctx, slab)
    var x2 = record_residual_gate_slab(g, x1, postgate, m, ctx, slab)
    var grads = execute_slab(g, g.node_of_tensor[x2[].id], _seed(d_out, ctx), ctx, slab)
    ctx.synchronize()
    _ = grads[x1[].id][].to_host(ctx)
    _report("K=2 WHOLE mlp (x1→x2)        ", slab)


# ── K=3 sub-A: mlp-up (x1 → xn2 → xm2 → mg,mu → sw), seeded d_sw ─────────────
def measure_mlp_up(ctx: DeviceContext) raises:
    var g = Graph()
    var x1 = _leaf(g, [1, L, FEATURES], UInt64(40), ctx)
    var postscale = TArc(_bf([FEATURES], UInt64(31), Float32(0.1), ctx))
    var postshift = TArc(_bf([FEATURES], UInt64(32), Float32(0.1), ctx))
    var postnorm_w = TArc(_bf([FEATURES], UInt64(34), Float32(0.1), ctx))
    var mg_a = g.fresh_tensor_id(); var mg_b = g.fresh_tensor_id()
    var mu_a = g.fresh_tensor_id(); var mu_b = g.fresh_tensor_id()
    var lo_mg = _mk_lora(FEATURES, MLPDIM, UInt64(105), ctx)
    var lo_mu = _mk_lora(FEATURES, MLPDIM, UInt64(106), ctx)
    var w_mg = TArc(_bf([MLPDIM, FEATURES], UInt64(6), Float32(0.04), ctx))
    var w_mu = TArc(_bf([MLPDIM, FEATURES], UInt64(7), Float32(0.04), ctx))
    var d_sw = _bf([1, L, MLPDIM], UInt64(51), Float32(0.5), ctx)   # d_out at the sw boundary
    var slab = StepSlab(ctx, 8 * 1024 * 1024 * 1024)

    var xn2 = record_rms_norm_dx_slab(g, x1, postnorm_w, EPS, ctx, slab)
    var xm2 = record_modulate_slab(g, xn2, postscale, postshift, 0, ctx, slab)
    var mg = record_proj_lora_slab(g, xm2, w_mg, lo_mg, mg_a, mg_b, L, FEATURES, MLPDIM, ctx, slab)
    var mu = record_proj_lora_slab(g, xm2, w_mu, lo_mu, mu_a, mu_b, L, FEATURES, MLPDIM, ctx, slab)
    var sw = record_swiglu_slab(g, mg, mu, ctx, slab)
    var grads = execute_slab(g, g.node_of_tensor[sw[].id], _seed(d_sw, ctx), ctx, slab)
    ctx.synchronize()
    _ = grads[x1[].id][].to_host(ctx)
    _report("K=3 mlp-up (x1→sw)           ", slab)


# ── K=3 sub-B: mlp-down (sw → m → x2 = residual_gate(x1, postgate, m)) ───────
def measure_mlp_down(ctx: DeviceContext) raises:
    var g = Graph()
    var x1 = _leaf(g, [1, L, FEATURES], UInt64(40), ctx)
    var sw = _leaf(g, [1, L, MLPDIM], UInt64(41), ctx)
    var postgate = TArc(_bf([FEATURES], UInt64(33), Float32(0.1), ctx))
    var md_a = g.fresh_tensor_id(); var md_b = g.fresh_tensor_id()
    var lo_md = _mk_lora(MLPDIM, FEATURES, UInt64(107), ctx)
    var w_md = TArc(_bf([FEATURES, MLPDIM], UInt64(8), Float32(0.04), ctx))
    var d_out = _bf([1, L, FEATURES], UInt64(50), Float32(0.5), ctx)
    var slab = StepSlab(ctx, 8 * 1024 * 1024 * 1024)

    var m = record_proj_lora_slab(g, sw, w_md, lo_md, md_a, md_b, L, MLPDIM, FEATURES, ctx, slab)
    var x2 = record_residual_gate_slab(g, x1, postgate, m, ctx, slab)
    var grads = execute_slab(g, g.node_of_tensor[x2[].id], _seed(d_out, ctx), ctx, slab)
    ctx.synchronize()
    _ = grads[x1[].id][].to_host(ctx)
    _report("K=3 mlp-down (sw→x2)         ", slab)


# ── K=2 segment A: the ATTN branch (x → x1), full op chain (4-way d_xm fork,
# qkv+gate+wo device-grad proj, qknorm, rope, GQA, sdpa, sigmoid-gate). ───────
def measure_attn(ctx: DeviceContext) raises:
    comptime HEADS = 48
    comptime KVHEADS = 12
    comptime HEADDIM = 128
    comptime N_REP = HEADS // KVHEADS
    var g = Graph()
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))
    var x = _leaf(g, [1, L, FEATURES], UInt64(40), ctx)
    var prescale = TArc(_bf([FEATURES], UInt64(31), Float32(0.1), ctx))
    var preshift = TArc(_bf([FEATURES], UInt64(32), Float32(0.1), ctx))
    var pregate = TArc(_bf([FEATURES], UInt64(33), Float32(0.1), ctx))
    var prenorm_w = TArc(_bf([FEATURES], UInt64(34), Float32(0.1), ctx))
    var qnorm_w = TArc(_bf([HEADDIM], UInt64(35), Float32(0.1), ctx))
    var knorm_w = TArc(_bf([HEADDIM], UInt64(36), Float32(0.1), ctx))
    # rope tables (identity), tiled.
    var cos = add_scalar(zeros_device([L, HEADDIM // 2], STDtype.BF16, ctx), Float32(1.0), ctx)
    var sin = zeros_device([L, HEADDIM // 2], STDtype.BF16, ctx)
    var cos_q = TArc(_tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx))
    var sin_q = TArc(_tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx))
    var cos_k = TArc(_tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx))
    var sin_k = TArc(_tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx))
    # 5 adapters: wq wk wv gate wo (device-grad leaves).
    var wq_a = g.fresh_tensor_id(); var wq_b = g.fresh_tensor_id()
    var wk_a = g.fresh_tensor_id(); var wk_b = g.fresh_tensor_id()
    var wv_a = g.fresh_tensor_id(); var wv_b = g.fresh_tensor_id()
    var wg_a = g.fresh_tensor_id(); var wg_b = g.fresh_tensor_id()
    var wo_a = g.fresh_tensor_id(); var wo_b = g.fresh_tensor_id()
    var lo_q = _mk_lora(FEATURES, HEADS * HEADDIM, UInt64(100), ctx)
    var lo_k = _mk_lora(FEATURES, KVHEADS * HEADDIM, UInt64(101), ctx)
    var lo_v = _mk_lora(FEATURES, KVHEADS * HEADDIM, UInt64(102), ctx)
    var lo_g = _mk_lora(FEATURES, FEATURES, UInt64(103), ctx)
    var lo_o = _mk_lora(FEATURES, FEATURES, UInt64(104), ctx)
    var w_q = TArc(_bf([HEADS * HEADDIM, FEATURES], UInt64(1), Float32(0.04), ctx))
    var w_k = TArc(_bf([KVHEADS * HEADDIM, FEATURES], UInt64(2), Float32(0.04), ctx))
    var w_v = TArc(_bf([KVHEADS * HEADDIM, FEATURES], UInt64(3), Float32(0.04), ctx))
    var w_g = TArc(_bf([FEATURES, FEATURES], UInt64(4), Float32(0.04), ctx))
    var w_o = TArc(_bf([FEATURES, FEATURES], UInt64(5), Float32(0.04), ctx))
    var d_x1 = _bf([1, L, FEATURES], UInt64(52), Float32(0.5), ctx)   # seed at the x1 boundary
    var slab = StepSlab(ctx, 8 * 1024 * 1024 * 1024)

    var xn = record_rms_norm_dx_slab(g, x, prenorm_w, EPS, ctx, slab)
    var xm = record_modulate_slab(g, xn, prescale, preshift, 0, ctx, slab)
    var zf = TArc(zeros_device_slab(xm[].shape(), xm[].dtype(), ctx, slab))
    var xm_a = record_add_slab(g, xm, zf, ctx, slab)
    var xm_b = record_add_slab(g, xm, zf, ctx, slab)
    var q = record_proj_lora_slab(g, xm_a, w_q, lo_q, wq_a, wq_b, L, FEATURES, HEADS * HEADDIM, ctx, slab)
    var k = record_proj_lora_slab(g, xm_a, w_k, lo_k, wk_a, wk_b, L, FEATURES, KVHEADS * HEADDIM, ctx, slab)
    var v_lin = record_proj_lora_slab(g, xm_b, w_v, lo_v, wv_a, wv_b, L, FEATURES, KVHEADS * HEADDIM, ctx, slab)
    var gate_pre = record_proj_lora_slab(g, xm_b, w_g, lo_g, wg_a, wg_b, L, FEATURES, FEATURES, ctx, slab)
    var q_pre = record_reshape(g, q, [1, L, HEADS, HEADDIM], ctx)
    var k_pre = record_reshape(g, k, [1, L, KVHEADS, HEADDIM], ctx)
    var v = record_reshape(g, v_lin, [1, L, KVHEADS, HEADDIM], ctx)
    var q_rms = record_rms_norm_dx_slab(g, q_pre, qnorm_w, EPS, ctx, slab)
    var k_rms = record_rms_norm_dx_slab(g, k_pre, knorm_w, EPS, ctx, slab)
    var q_rope = record_rope_slab(g, q_rms, cos_q, sin_q, ctx, slab)
    var k_rope = record_rope_slab(g, k_rms, cos_k, sin_k, ctx, slab)
    var k_full = record_repeat_kv_slab(g, k_rope, L, KVHEADS, N_REP, HEADDIM, ctx, slab)
    var v_full = record_repeat_kv_slab(g, v, L, KVHEADS, N_REP, HEADDIM, ctx, slab)
    var att = record_sdpa_nomask_slab[1, L, HEADS, HEADDIM](g, q_rope, k_full, v_full, scale, ctx, slab)
    var attn_flat = record_reshape(g, att, [1, L, FEATURES], ctx)
    var sg = record_sigmoid_slab(g, gate_pre, ctx, slab)
    var gated = record_mul_slab(g, attn_flat, sg, ctx, slab)
    var a = record_proj_lora_slab(g, gated, w_o, lo_o, wo_a, wo_b, L, FEATURES, FEATURES, ctx, slab)
    var x1 = record_residual_gate_slab(g, x, pregate, a, ctx, slab)
    var grads = execute_slab(g, g.node_of_tensor[x1[].id], _seed(d_x1, ctx), ctx, slab)
    ctx.synchronize()
    _ = grads[x[].id][].to_host(ctx)
    _report("K=2/K=3 ATTN (x→x1)          ", slab)


def main() raises:
    var ctx = DeviceContext()
    print("krea2 segment slab peaks @ L=2432 bf16 (device-grad proj_lora):")
    measure_whole_mlp(ctx)
    measure_mlp_up(ctx)
    measure_mlp_down(ctx)
    measure_attn(ctx)
