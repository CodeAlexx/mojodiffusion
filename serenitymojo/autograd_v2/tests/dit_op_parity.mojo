# autograd_v2/tests/dit_op_parity.mojo - Phase P2 per-op BIT-EQUAL parity
# gates (AUTOGRAD_V2_MOJO_DESIGN.md P2): for each zimage DiT op kind, on REAL
# trainer shapes (S=1248, D=3840, F=10240, H=30, Dh=128, B=1), run
#   1. the hand-chain: forward op directly + the existing ops/*_backward call
#      directly on a deterministic upstream grad (the exact sequence of
#      zimage_block_lora_backward_device_tensors_batch,
#      models/zimage/lora_block.mojo:1702-1793);
#   2. the engine: record the same op on the IDENTICAL input tensors (shared
#      arcs - byte-identical by construction) -> execute with the same
#      upstream grad seeded at the root;
# then compare every gradient BIT-EQUAL (to_host_bf16 + UInt bit-pattern
# compare via Scalar.to_bits()).
#
# Composite gate (the C15 proof): x feeds THREE OPK_PROJ_LORA nodes (q,k,v);
# the engine fires them in ready-queue order v,k,q (topo/seq DESC), i.e. the
# REVERSE of registration order - if the InputBuffer folded in arrival order
# the bf16 sum would differ from the hand-chain's fixed left fold
# d_xn1s = add(add(dq, dk), dv) (lora_block.mojo:1768). Slot-ordered fan-in
# must reproduce it bit-for-bit.
#
# Build: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/dit_op_parity.mojo -o /tmp/dit_op_parity
# Run:   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib /tmp/dit_op_parity

from std.gpu.host import DeviceContext
from std.collections import Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.unary import tanh_op
from serenitymojo.ops.tensor_algebra import add
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import (
    rope_backward,
    gate_residual_backward_dxdy,
)
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.models.zimage.lora_block import ZImageLoraAdapterDevice
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute
from serenitymojo.autograd_v2.ops_record import (
    proj_lora_backward,
    record_proj_lora,
    record_rms_norm_dx,
    record_modulate,
    record_rope,
    record_sdpa,
    record_swiglu,
    record_residual_gate,
    record_reshape,
    record_add,
)

# REAL zimage trainer shapes (training/train_zimage_real.mojo:131-168).
comptime B = 1
comptime H = 30
comptime Dh = 128
comptime D = H * Dh          # 3840
comptime F = 10240
comptime S = 1248
comptime ROWS = B * S
comptime RANK = 16
comptime EPS = Float32(1e-6)
comptime LORA_SCALE = Float32(0.5)


# ── deterministic host pattern (32-bit LCG -> bf16 via from_host's RNE cast;
# values in [-1, 1), NOT all-equal) ──────────────────────────────────────────
def _lcg(n: Int, seed: Int) -> List[Float32]:
    var out = List[Float32]()
    var s = UInt64(seed) * UInt64(2654435761) + UInt64(12345)
    s = s & UInt64(0xFFFFFFFF)
    for _ in range(n):
        s = (s * UInt64(1664525) + UInt64(1013904223)) & UInt64(0xFFFFFFFF)
        var u = Int((s >> 9) & UInt64(0xFFFF))
        out.append(Float32(u) * Float32(3.0517578125e-05) - Float32(1.0))
    return out^


def _bf16(var shape: List[Int], seed: Int, ctx: DeviceContext) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return Tensor.from_host(_lcg(n, seed), shape^, STDtype.BF16, ctx)


def _zeros_bf16(var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    var vals = List[Float32]()
    for _ in range(n):
        vals.append(Float32(0.0))
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


# ── bit-equal compare: to_host_bf16 both sides, count to_bits() mismatches ──
def _cmp(name: String, got: Tensor, want: Tensor, ctx: DeviceContext) raises -> Bool:
    var hg = got.to_host_bf16(ctx)
    var hw = want.to_host_bf16(ctx)
    if len(hg) != len(hw):
        print(
            "GATE " + name + " FAIL numel " + String(len(hg))
            + " != " + String(len(hw))
        )
        return False
    var bad = 0
    for i in range(len(hg)):
        if hg[i].to_bits() != hw[i].to_bits():
            bad += 1
    var verdict = String("PASS") if bad == 0 else String("FAIL")
    print(
        "GATE " + name + " " + verdict + " n_mismatch=" + String(bad)
        + "/" + String(len(hg))
    )
    return bad == 0


def _n_diff_bits(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Int:
    var ha = a.to_host_bf16(ctx)
    var hb = b.to_host_bf16(ctx)
    var bad = 0
    for i in range(len(ha)):
        if ha[i].to_bits() != hb[i].to_bits():
            bad += 1
    return bad


# ── leaf helper: bf16 tensor stamped with a fresh tracked id + OPK_LEAF ─────
def _leaf_bf16(
    mut g: Graph, var shape: List[Int], seed: Int, ctx: DeviceContext
) raises -> TArc:
    var t = _bf16(shape^, seed, ctx)
    t.set_id(g.fresh_tensor_id())
    _ = g.leaf(t.id)
    return TArc(t^)


def _root_of(g: Graph, y: TArc) raises -> Int:
    return g.node_of_tensor[y[].id]


def _adapter(
    a: TArc, b: TArc, in_f: Int, out_f: Int
) raises -> ZImageLoraAdapterDevice:
    return ZImageLoraAdapterDevice(
        a.copy(), b.copy(), RANK, in_f, out_f, LORA_SCALE
    )


# ─────────────────────────────────────────────────────────────────────────────
# GATE proj_lora: y = linear(x, W) + scale*(x@Aᵀ)@Bᵀ; d_x/d_a/d_b vs the
# hand-chain _proj_bwd_with_lora_device_tensors sequence (lora_block.mojo:
# 568-577, replicated as ops_record.proj_lora_backward).
# ─────────────────────────────────────────────────────────────────────────────
def gate_proj_lora(
    name: String, in_f: Int, out_f: Int, seed0: Int, ctx: DeviceContext
) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [ROWS, in_f], seed0, ctx)
    var a = _leaf_bf16(g, [RANK, in_f], seed0 + 1, ctx)
    var b = _leaf_bf16(g, [out_f, RANK], seed0 + 2, ctx)
    var x_id = x[].id
    var a_id = a[].id
    var b_id = b[].id
    var w = TArc(_bf16([out_f, in_f], seed0 + 3, ctx))  # frozen base weight
    var gy = TArc(_bf16([ROWS, out_f], seed0 + 4, ctx))  # upstream grad
    var lo = _adapter(a, b, in_f, out_f)

    # hand-chain backward (forward output not needed for these grads)
    var hand = proj_lora_backward(gy[], x[], w[], lo, ROWS, in_f, out_f, ctx)

    # engine: record on the IDENTICAL tensors, seed the same upstream grad
    var y = record_proj_lora(g, x, w, lo, a_id, b_id, ROWS, in_f, out_f, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)

    var ok = _cmp(name + "_dx", grads[x_id][], hand.d_x[], ctx)
    ok = _cmp(name + "_da", grads[a_id][], hand.d_a[], ctx) and ok
    ok = _cmp(name + "_db", grads[b_id][], hand.d_b[], ctx) and ok
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# GATE rms_norm_dx (frozen weight): hand = rms_norm_backward_dx
# (lora_block.mojo:1717/1735 call sites).
# ─────────────────────────────────────────────────────────────────────────────
def gate_rms_norm_dx(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [ROWS, D], 100, ctx)
    var x_id = x[].id
    var w = TArc(_bf16([D], 101, ctx))
    var gy = TArc(_bf16([ROWS, D], 102, ctx))

    var hand = rms_norm_backward_dx(gy[], x[], w[], EPS, ctx)

    var y = record_rms_norm_dx(g, x, w, EPS, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    return _cmp("rms_norm_dx", grads[x_id][], hand, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# GATE modulate, frozen scale (block path: compute_param_grads=False,
# lora_block.mojo:1732-1734) and trained scale (final-layer path: d_scale).
# ─────────────────────────────────────────────────────────────────────────────
def gate_modulate_frozen(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [ROWS, D], 110, ctx)
    var x_id = x[].id
    var scale = TArc(_bf16([D], 111, ctx))
    var shift = TArc(_zeros_bf16([D], ctx))
    var gy = TArc(_bf16([ROWS, D], 112, ctx))

    var hand = modulate_backward(gy[], x[], scale[], ctx, compute_param_grads=False)

    var y = record_modulate(g, x, scale, shift, 0, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    return _cmp("modulate_frozen_dx", grads[x_id][], hand.d_x, ctx)


def gate_modulate_param(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [ROWS, D], 120, ctx)
    var x_id = x[].id
    var scale_t = _bf16([D], 121, ctx)
    scale_t.set_id(g.fresh_tensor_id())
    var scale_id = scale_t.id
    _ = g.leaf(scale_id)
    var scale = TArc(scale_t^)
    var shift = TArc(_zeros_bf16([D], ctx))
    var gy = TArc(_bf16([ROWS, D], 122, ctx))

    var hand = modulate_backward(gy[], x[], scale[], ctx, compute_param_grads=True)

    var y = record_modulate(g, x, scale, shift, scale_id, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    var ok = _cmp("modulate_param_dx", grads[x_id][], hand.d_x, ctx)
    ok = _cmp("modulate_param_dscale", grads[scale_id][], hand.d_scale, ctx) and ok
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# GATE rope: x [B,S,H,Dh] -> rows = B*S*H; cos/sin [rows, Dh/2] bf16 tables;
# hand = rope_backward(g, cos, sin, True) (lora_block.mojo:1749-1750).
# ─────────────────────────────────────────────────────────────────────────────
def gate_rope(ctx: DeviceContext) raises -> Bool:
    comptime RR = B * S * H
    var g = Graph()
    var x = _leaf_bf16(g, [B, S, H, Dh], 130, ctx)
    var x_id = x[].id
    var cos = TArc(_bf16([RR, Dh // 2], 131, ctx))
    var sin = TArc(_bf16([RR, Dh // 2], 132, ctx))
    var gy = TArc(_bf16([B, S, H, Dh], 133, ctx))

    var hand = rope_backward(gy[], cos[], sin[], True, ctx)

    var y = record_rope(g, x, cos, sin, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    return _cmp("rope", grads[x_id][], hand, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# GATE sdpa: q,k,v [1,1248,30,128] (the real zimage B1 attention shape);
# hand = sdpa_backward[B,S,H,Dh] (lora_block.mojo:1746-1748).
# ─────────────────────────────────────────────────────────────────────────────
def gate_sdpa(ctx: DeviceContext) raises -> Bool:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var g = Graph()
    var q = _leaf_bf16(g, [B, S, H, Dh], 140, ctx)
    var k = _leaf_bf16(g, [B, S, H, Dh], 141, ctx)
    var v = _leaf_bf16(g, [B, S, H, Dh], 142, ctx)
    var q_id = q[].id
    var k_id = k[].id
    var v_id = v[].id
    var gy = TArc(_bf16([B, S, H, Dh], 143, ctx))

    var hand = sdpa_backward[B, S, H, Dh](q[], k[], v[], gy[], scale, ctx)

    var y = record_sdpa[B, S, H, Dh](g, q, k, v, scale, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    var ok = _cmp("sdpa_dq", grads[q_id][], hand.d_q, ctx)
    ok = _cmp("sdpa_dk", grads[k_id][], hand.d_k, ctx) and ok
    ok = _cmp("sdpa_dv", grads[v_id][], hand.d_v, ctx) and ok
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# GATE swiglu: gate/up [ROWS, F]; hand = swiglu_backward
# (lora_block.mojo:1722).
# ─────────────────────────────────────────────────────────────────────────────
def gate_swiglu(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var gp = _leaf_bf16(g, [ROWS, F], 150, ctx)
    var up = _leaf_bf16(g, [ROWS, F], 151, ctx)
    var gp_id = gp[].id
    var up_id = up[].id
    var gy = TArc(_bf16([ROWS, F], 152, ctx))

    var hand = swiglu_backward(gy[], gp[], up[], ctx)

    var y = record_swiglu(g, gp, up, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    var ok = _cmp("swiglu_dgate", grads[gp_id][], hand.d_gate, ctx)
    ok = _cmp("swiglu_dup", grads[up_id][], hand.d_up, ctx) and ok
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# GATE residual_gate (dxdy): out = x + gate_t*y, gate frozen; hand =
# gate_residual_backward_dxdy (lora_block.mojo:1715/1738).
# ─────────────────────────────────────────────────────────────────────────────
def gate_residual_gate(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [ROWS, D], 160, ctx)
    var yv = _leaf_bf16(g, [ROWS, D], 161, ctx)
    var x_id = x[].id
    var y_id = yv[].id
    var gate_vec = _bf16([D], 162, ctx)
    var gate_t = TArc(tanh_op(gate_vec, ctx))  # the hand-chain's tanh_op step
    var gy = TArc(_bf16([ROWS, D], 163, ctx))

    var hand = gate_residual_backward_dxdy(gy[], gate_t[], ctx)

    var out = record_residual_gate(g, x, gate_t, yv, ctx)
    var grads = execute(g, _root_of(g, out), gy.copy(), ctx)
    var ok = _cmp("residual_gate_dx", grads[x_id][], hand.d_x, ctx)
    ok = _cmp("residual_gate_dy", grads[y_id][], hand.d_y, ctx) and ok
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# GATE reshape: [ROWS, D] -> [B, S, H, Dh]; grad reshaped back, ZERO kernels;
# bits must be the upstream grad verbatim and the shape must be x's.
# ─────────────────────────────────────────────────────────────────────────────
def gate_reshape(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [ROWS, D], 170, ctx)
    var x_id = x[].id
    var gy = TArc(_bf16([B, S, H, Dh], 171, ctx))

    var y = record_reshape(g, x, [B, S, H, Dh], ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    var ok = _cmp("reshape_bits", grads[x_id][], gy[], ctx)
    var sh = grads[x_id][].shape()
    var shape_ok = len(sh) == 2 and sh[0] == ROWS and sh[1] == D
    var verdict = String("PASS") if shape_ok else String("FAIL")
    print(
        "GATE reshape_backshape " + verdict + " got=["
        + String(sh[0]) + "," + String(sh[1] if len(sh) > 1 else -1) + "]"
    )
    return ok and shape_ok


# ─────────────────────────────────────────────────────────────────────────────
# GATE add: d_a = d_b = upstream grad verbatim.
# ─────────────────────────────────────────────────────────────────────────────
def gate_add(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var a = _leaf_bf16(g, [ROWS, D], 180, ctx)
    var b = _leaf_bf16(g, [ROWS, D], 181, ctx)
    var a_id = a[].id
    var b_id = b[].id
    var gy = TArc(_bf16([ROWS, D], 182, ctx))

    var y = record_add(g, a, b, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    var ok = _cmp("add_da", grads[a_id][], gy[], ctx)
    ok = _cmp("add_db", grads[b_id][], gy[], ctx) and ok
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE GATE (the C15 proof): x feeds three OPK_PROJ_LORA nodes whose
# upstream grads all equal G (via an ADD tree root); the leaf's fan-in MUST
# fold as add(add(dq, dk), dv) - the hand-chain's fixed left fold
# (lora_block.mojo:1768) - even though the engine FIRES the projections in
# reverse (v, k, q: ready-queue topo/seq DESC ordering).
# ─────────────────────────────────────────────────────────────────────────────
def gate_qkv_fanin(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [ROWS, D], 200, ctx)
    var x_id = x[].id
    var aq = _leaf_bf16(g, [RANK, D], 201, ctx)
    var bq = _leaf_bf16(g, [D, RANK], 202, ctx)
    var ak = _leaf_bf16(g, [RANK, D], 203, ctx)
    var bk = _leaf_bf16(g, [D, RANK], 204, ctx)
    var av = _leaf_bf16(g, [RANK, D], 205, ctx)
    var bv = _leaf_bf16(g, [D, RANK], 206, ctx)
    var aq_id = aq[].id
    var bq_id = bq[].id
    var wq = TArc(_bf16([D, D], 207, ctx))
    var wk = TArc(_bf16([D, D], 208, ctx))
    var wv = TArc(_bf16([D, D], 209, ctx))
    var gy = TArc(_bf16([ROWS, D], 210, ctx))
    var loq = _adapter(aq, bq, D, D)
    var lok = _adapter(ak, bk, D, D)
    var lov = _adapter(av, bv, D, D)

    # hand-chain: three projection backwards with the SAME upstream G, then
    # the explicit left fold add(add(dq, dk), dv) (lora_block.mojo:1768).
    var hq = proj_lora_backward(gy[], x[], wq[], loq, ROWS, D, D, ctx)
    var hk = proj_lora_backward(gy[], x[], wk[], lok, ROWS, D, D, ctx)
    var hv = proj_lora_backward(gy[], x[], wv[], lov, ROWS, D, D, ctx)
    var dx_hand = add(add(hq.d_x[], hk.d_x[], ctx), hv.d_x[], ctx)
    # arrival-order fold the engine WOULD compute without C15 (fires v,k,q):
    var dx_arrival = add(add(hv.d_x[], hk.d_x[], ctx), hq.d_x[], ctx)
    var sens = _n_diff_bits(dx_hand, dx_arrival, ctx)
    print(
        "INFO qkv_fanin fold-order sensitivity (hand vs arrival-order): n_diff="
        + String(sens) + "/" + String(ROWS * D)
        + (" (order-sensitive, C15 is load-bearing)" if sens > 0
           else " (inputs happened to be order-insensitive)")
    )

    # engine: registration order q, k, v (= forward order = fold order).
    var yq = record_proj_lora(g, x, wq, loq, aq_id, bq_id, ROWS, D, D, ctx)
    var yk = record_proj_lora(g, x, wk, lok, ak[].id, bk[].id, ROWS, D, D, ctx)
    var yv = record_proj_lora(g, x, wv, lov, av[].id, bv[].id, ROWS, D, D, ctx)
    var s1 = record_add(g, yq, yk, ctx)
    var s2 = record_add(g, s1, yv, ctx)
    var grads = execute(g, _root_of(g, s2), gy.copy(), ctx)

    var ok = _cmp("qkv_fanin_dx", grads[x_id][], dx_hand, ctx)
    # spot-check a LoRA leaf through the composite too
    ok = _cmp("qkv_fanin_da_q", grads[aq_id][], hq.d_a[], ctx) and ok
    ok = _cmp("qkv_fanin_db_q", grads[bq_id][], hq.d_b[], ctx) and ok
    return ok


def main() raises:
    var ctx = DeviceContext()
    var ok = True
    ok = gate_proj_lora("proj_lora_qkv", D, D, 10, ctx) and ok
    ok = gate_proj_lora("proj_lora_w1", D, F, 20, ctx) and ok
    ok = gate_rms_norm_dx(ctx) and ok
    ok = gate_modulate_frozen(ctx) and ok
    ok = gate_modulate_param(ctx) and ok
    ok = gate_rope(ctx) and ok
    ok = gate_sdpa(ctx) and ok
    ok = gate_swiglu(ctx) and ok
    ok = gate_residual_gate(ctx) and ok
    ok = gate_reshape(ctx) and ok
    ok = gate_add(ctx) and ok
    ok = gate_qkv_fanin(ctx) and ok
    if not ok:
        raise Error("dit_op_parity: at least one GATE FAILED")
    print("ALL P2 DIT OP PARITY GATES PASS (bit-equal)")
