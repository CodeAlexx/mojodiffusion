# autograd_v2/tests/wan_ffn_op_parity.mojo — Phase 2 slice 1 per-op BIT gates for
# the Wan2.2 fine-grained FFN vocabulary (OPK_WAN_MOD_PRE / OPK_GELU /
# OPK_WAN_GATED_RESIDUAL). Each gate:
#   1. builds real-shaped, NON-DEGENERATE bf16 inputs (32-bit LCG in [-1,1) — the
#      dit_op_parity.mojo pattern; NEVER modular fills, they alias);
#   2. computes the hand-chain oracle backward DIRECTLY (the EXACT wan22_block.mojo
#      helpers the block backward calls);
#   3. records the op through its record_* wrapper, runs engine.execute;
#   4. asserts the engine grads are BIT-EQUAL (to_bits) to the oracle grads.
# Wan self+cross+ffn AdaLN/gate/gelu are MATH-MODE deterministic (no flash) so a
# TRUE bit gate, not a value class (numeric-parity-testing). The proj_lora op is a
# separate slice (device-LoRA oracle refactor).
#
# Build (serial; rm -f serenitymojo.mojopkg first):
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/wan_ffn_op_parity.mojo -o /tmp/wan_ffn_gate
# Run: env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/wan_ffn_gate
#
# Mojo 1.0.0b1, NVIDIA.

from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.norm_backward import layer_norm_backward_dx
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.models.wan22.wan22_block import (
    wan_mod_pre,
    wan_modulate_backward,
    wan_gate_residual_backward,
)
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute
from serenitymojo.models.wan22.wan22_block import _base_dx, _wan_lora_bwd_device
from serenitymojo.models.klein.lora_block import LoraAdapter
from serenitymojo.ops.tensor_algebra import add as _ta_add
from serenitymojo.ops.rope_struct_backward import rope_backward
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.attention_backward import sdpa_backward_rect
from std.math import sqrt
from serenitymojo.autograd_v2.ops_record import (
    record_wan_mod_pre,
    record_gelu,
    record_wan_gated_residual,
    record_wan_proj_lora,
    record_layer_norm_dx,
    record_wan_rope,
    record_sdpa_rect,
)

# Representative wan block shapes (per-op correctness is shape-agnostic; these are
# real-structured and non-trivial). S = seq tokens, DIM = model dim, FFN = mlp dim.
comptime S = 256
comptime DIM = 1536
comptime FFN = 4096
comptime EPS = Float32(1e-6)


# ── deterministic host pattern (32-bit LCG → bf16 via RNE cast; [-1,1), NOT
# all-equal) — dit_op_parity.mojo:77 ─────────────────────────────────────────
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


def _fill_bf16(
    var shape: List[Int], v: Float32, ctx: DeviceContext
) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    var vals = List[Float32]()
    for _ in range(n):
        vals.append(v)
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


def _cmp(name: String, got: Tensor, want: Tensor, ctx: DeviceContext) raises -> Bool:
    var hg = got.to_host_bf16(ctx)
    var hw = want.to_host_bf16(ctx)
    if len(hg) != len(hw):
        print("GATE " + name + " FAIL numel " + String(len(hg)) + " != " + String(len(hw)))
        return False
    var bad = 0
    for i in range(len(hg)):
        if hg[i].to_bits() != hw[i].to_bits():
            bad += 1
    var verdict = String("PASS") if bad == 0 else String("FAIL")
    print("GATE " + name + " " + verdict + " n_mismatch=" + String(bad) + "/" + String(len(hg)))
    return bad == 0


def _leaf_bf16(
    mut g: Graph, var shape: List[Int], seed: Int, ctx: DeviceContext
) raises -> TArc:
    var t = _bf16(shape^, seed, ctx)
    t.set_id(g.fresh_tensor_id())
    _ = g.leaf(t.id)
    return TArc(t^)


def _root_of(g: Graph, y: TArc) raises -> Int:
    return g.node_of_tensor[y[].id]


# ── GATE wan_mod_pre: o = LN_no_affine(x)*(1+scale)+shift. hand = the oracle's
# mb/lnb pair (wan22_block.mojo:1620/1624): wan_modulate_backward(gy, ln, scale)
# → d_ln, then layer_norm_backward_dx(d_ln, x, ones, eps) → d_x. ─────────────
def gate_wan_mod_pre(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [S, DIM], 200, ctx)
    var x_id = x[].id
    var scale = TArc(_bf16([S, DIM], 201, ctx))
    var shift = TArc(_bf16([S, DIM], 202, ctx))
    var ones = TArc(_fill_bf16([DIM], 1.0, ctx))
    var zeros = TArc(_fill_bf16([DIM], 0.0, ctx))
    var gy = TArc(_bf16([S, DIM], 203, ctx))

    # hand-chain: recompute ln (the saved forward layernorm), then the oracle pair.
    var mp = wan_mod_pre(x[], scale[], shift[], ones[], zeros[], EPS, ctx)
    var mb = wan_modulate_backward(gy[], mp.ln[], scale[], ctx)
    var hand = layer_norm_backward_dx(mb.d_ln, x[], ones[], EPS, ctx)

    var y = record_wan_mod_pre(g, x, scale, shift, ones, zeros, EPS, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    return _cmp("wan_mod_pre_dx", grads[x_id][], hand, ctx)


# ── GATE gelu: act = gelu(x); hand = gelu_backward(gy, x) (wan22_block.mojo:1607).
def gate_gelu(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [S, FFN], 210, ctx)
    var x_id = x[].id
    var gy = TArc(_bf16([S, FFN], 211, ctx))

    var hand = gelu_backward(gy[], x[], ctx)

    var y = record_gelu(g, x, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    return _cmp("gelu_dx", grads[x_id][], hand, ctx)


# ── GATE wan_gated_residual: o = x + gate*y. hand = wan_gate_residual_backward(gy,
# y, gate) → d_x=gy, d_y=gy*gate (wan22_block.mojo:1595/1689). Edges [x, y]. ──
def gate_wan_gated_residual(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [S, DIM], 220, ctx)
    var y_leaf = _leaf_bf16(g, [S, DIM], 221, ctx)
    var x_id = x[].id
    var y_id = y_leaf[].id
    var gate = TArc(_bf16([S, DIM], 222, ctx))
    var gy = TArc(_bf16([S, DIM], 223, ctx))

    var hand = wan_gate_residual_backward(gy[], y_leaf[], gate[], ctx)

    var o = record_wan_gated_residual(g, x, y_leaf, gate, ctx)
    var grads = execute(g, _root_of(g, o), gy.copy(), ctx)
    var ok = _cmp("wan_gated_residual_dx", grads[x_id][], hand.d_x, ctx)
    ok = _cmp("wan_gated_residual_dy", grads[y_id][], hand.d_y, ctx) and ok
    return ok


# ── GATE wan_proj_lora: y = linear(x, W_frozen, bias) + LoRA(x). hand = _base_dx(gy,
# W) + _wan_lora_bwd_device(adapter, gy, x) folded (wan22_block.mojo:1599/1603/1605
# class). NONZERO A AND B → d_A non-degenerate (numeric-parity-testing). Compares
# d_x, d_A, d_B. Backward-only: the forward y value is not exercised. ──────────
def _lora(in_f: Int, out_f: Int, rank: Int, scale: Float32, seed: Int) raises -> LoraAdapter:
    var a = _lcg(rank * in_f, seed)          # NONZERO A
    var b = _lcg(out_f * rank, seed + 1)     # NONZERO B (not the B=0 init!)
    var za = List[Float32]()
    for _ in range(rank * in_f):
        za.append(0.0)
    var zb = List[Float32]()
    for _ in range(out_f * rank):
        zb.append(0.0)
    return LoraAdapter(a^, b^, rank, in_f, out_f, scale, za.copy(), za^, zb.copy(), zb^)


def gate_wan_proj_lora(ctx: DeviceContext) raises -> Bool:
    comptime IN = DIM
    comptime OUT = DIM
    comptime RANK = 16
    var g = Graph()
    var x = _leaf_bf16(g, [S, IN], 230, ctx)
    var a_leaf = _leaf_bf16(g, [RANK, IN], 231, ctx)   # dummy sink (grad routes here)
    var b_leaf = _leaf_bf16(g, [OUT, RANK], 232, ctx)
    var x_id = x[].id
    var a_id = a_leaf[].id
    var b_id = b_leaf[].id
    var w = TArc(_bf16([OUT, IN], 233, ctx))           # frozen base weight
    var gy = TArc(_bf16([S, OUT], 234, ctx))           # upstream grad
    var lo = _lora(IN, OUT, RANK, Float32(1.0), 240)

    # hand-chain: base d_x + device LoRA bwd, folded (the oracle's _base_dx + fold).
    var base_dx = _base_dx(gy[], w[], S, IN, OUT, ctx)
    var lg = _wan_lora_bwd_device(Optional[LoraAdapter](lo.copy()), gy[], x[], S, IN, OUT, ctx)
    var hand_dx = _ta_add(base_dx, lg.d_x[], ctx)

    var y = record_wan_proj_lora(
        g, x, w, Optional[TArc](None), Optional[LoraAdapter](lo.copy()),
        a_id, b_id, S, IN, OUT, ctx,
    )
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    var ok = _cmp("wan_proj_lora_dx", grads[x_id][], hand_dx, ctx)
    ok = _cmp("wan_proj_lora_da", grads[a_id][], lg.d_a[], ctx) and ok
    ok = _cmp("wan_proj_lora_db", grads[b_id][], lg.d_b[], ctx) and ok
    return ok


def _f32(var shape: List[Int], seed: Int, ctx: DeviceContext) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return Tensor.from_host(_lcg(n, seed), shape^, STDtype.F32, ctx)


# ── GATE layer_norm_dx (frozen affine): hand = layer_norm_backward_dx (the n3
# cross-attn backward, wan22_block.mojo:1681). ───────────────────────────────
def gate_layer_norm_dx(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var x = _leaf_bf16(g, [S, DIM], 300, ctx)
    var x_id = x[].id
    var w = TArc(_bf16([DIM], 301, ctx))     # affine gamma (frozen)
    var b = TArc(_bf16([DIM], 302, ctx))     # affine beta (frozen)
    var gy = TArc(_bf16([S, DIM], 303, ctx))

    var hand = layer_norm_backward_dx(gy[], x[], w[], EPS, ctx)

    var y = record_layer_norm_dx(g, x, w, b, EPS, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    return _cmp("layer_norm_dx", grads[x_id][], hand, ctx)


# ── GATE wan_rope: per-head interleaved rope, F32-cast backward dance. hand = the
# oracle's cast(go→F32) → rope_backward(F32,True) → cast(bf16) (wan22_block.mojo:
# 1706-1711). x [1,RS,RH,RDh]; F32 tables [RS*RH, RDh/2]. ────────────────────
def gate_wan_rope(ctx: DeviceContext) raises -> Bool:
    comptime RS = 8
    comptime RH = 8
    comptime RDh = 16
    comptime RROWS = RS * RH
    var g = Graph()
    var x = _leaf_bf16(g, [1, RS, RH, RDh], 310, ctx)
    var x_id = x[].id
    var cos = TArc(_f32([RROWS, RDh // 2], 311, ctx))
    var sin = TArc(_f32([RROWS, RDh // 2], 312, ctx))
    var gy = TArc(_bf16([1, RS, RH, RDh], 313, ctx))

    # hand: the exact F32-cast dance the oracle runs.
    var gy_f32 = cast_tensor(gy[], STDtype.F32, ctx)
    var dx_f32 = rope_backward(gy_f32, cos[], sin[], True, ctx)
    var hand = cast_tensor(dx_f32, STDtype.BF16, ctx)

    var y = record_wan_rope(g, x, cos, sin, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    return _cmp("wan_rope_dx", grads[x_id][], hand, ctx)


# ── GATE sdpa_rect (cross-attn): rect attention, math-mode → TRUE bit gate. hand =
# sdpa_backward_rect[1,Sq,Skv,H,Dh] (wan22_block.mojo:1641). Compares d_q/d_k/d_v.
def gate_sdpa_rect(ctx: DeviceContext) raises -> Bool:
    comptime QB = 1
    comptime SQ = 8
    comptime SKV = 6
    comptime QH = 8
    comptime QDh = 16
    var scale = Float32(1.0) / sqrt(Float32(QDh))
    var g = Graph()
    var q = _leaf_bf16(g, [QB, SQ, QH, QDh], 320, ctx)
    var k = _leaf_bf16(g, [QB, SKV, QH, QDh], 321, ctx)
    var v = _leaf_bf16(g, [QB, SKV, QH, QDh], 322, ctx)
    var q_id = q[].id
    var k_id = k[].id
    var v_id = v[].id
    var gy = TArc(_bf16([QB, SQ, QH, QDh], 323, ctx))

    var hand = sdpa_backward_rect[QB, SQ, SKV, QH, QDh](q[], k[], v[], gy[], scale, ctx)

    var y = record_sdpa_rect[QB, SQ, SKV, QH, QDh](g, q, k, v, scale, ctx)
    var grads = execute(g, _root_of(g, y), gy.copy(), ctx)
    var ok = _cmp("sdpa_rect_dq", grads[q_id][], hand.d_q, ctx)
    ok = _cmp("sdpa_rect_dk", grads[k_id][], hand.d_k, ctx) and ok
    ok = _cmp("sdpa_rect_dv", grads[v_id][], hand.d_v, ctx) and ok
    return ok


def main() raises:
    var ctx = DeviceContext()
    var ok = True
    ok = gate_wan_mod_pre(ctx) and ok
    ok = gate_gelu(ctx) and ok
    ok = gate_wan_gated_residual(ctx) and ok
    ok = gate_wan_proj_lora(ctx) and ok
    ok = gate_layer_norm_dx(ctx) and ok
    ok = gate_wan_rope(ctx) and ok
    ok = gate_sdpa_rect(ctx) and ok
    if ok:
        print("VERDICT: PASS — wan FFN fine-grained ops BIT-EQUAL vs hand-chain")
    else:
        print("VERDICT: FAIL — see GATE lines above")
