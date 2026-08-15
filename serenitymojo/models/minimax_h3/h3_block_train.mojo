# serenitymojo/models/minimax_h3/h3_block_train.mojo
#
# MiniMax-H3 transformer block TRAINING twin: forward-saving-activations +
# hand-chained backward. This is the parity-oracle arm of the trainer
# (production dispatch goes through autograd_v2 later, per the binding
# H3_V2_GRAPH policy); it is gated against torch autograd on REAL block-0
# weights: parity/h3_block_oracle.py -> output/checks/h3_block0_oracle
# .safetensors, gate parity/minimax_h3_block_train_parity.mojo.
#
# WEIGHT LAYOUT: RAW torchref/checkpoint order — `attn.qkv_proj.weight`
# [3*inner, hidden] rows [q|k|v], `mlp.fc1.weight` [2*ffn, hidden] rows
# [gate|value] (torchref model.py:279-291: chunk(2) -> (gate, value),
# silu(gate)*value). NOT the `minimax_h3_load_block_device` transformed
# layout: training grads must land in the oracle/LoRA convention with no
# permutation step. Oracle source (torchref-h3 @ 04324c28
# model.py:336-399):
#   mod   = adaln_proj(temb).view(-1, 6*hidden)        (precomputed table
#           here — the modcache contract; adaln stays FROZEN in training)
#   n1m   = rms_norm(x, norm1_w) * (1+scale_msa[idx]) + shift_msa[idx]
#   q,k,v = chunk3(qkv_proj(n1m)); q/k = rms_norm(., q/k_norm) then
#           partial rope (rotary_dim of head_dim, halfsplit rotate)
#   h_mid = x + gate_msa[idx] * out_proj(sdpa(q,k,v))
#   n2m   = rms_norm(h_mid, norm2_w) * (1+scale_mlp[idx]) + shift_mlp[idx]
#   out   = h_mid + gate_mlp[idx] * fc2(silu(gate)*value)  [fc1 packed]
#
# The per-row (indexed) AdaLN means scale/shift/gate are [S, hidden] after
# the row gather; their grads scatter-ADD back to the mod table rows via
# index_select_backward (repeated indices accumulate).
from std.math import sqrt
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.rope import rope_halfsplit_full_head_broadcast
from serenitymojo.ops.rope_struct_backward import rope_halfsplit_full_backward
from serenitymojo.ops.tensor_algebra import (
    mul, add, add_scalar, slice, concat, reshape, gather_rows, full_device,
)
from serenitymojo.ops.shape_backward import index_select_backward

comptime TArc = ArcPointer[Tensor]


def _reshaped(t: Tensor, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return reshape(t, shape^, ctx)


struct H3BlockTrainWeights(Copyable, Movable):
    """RAW-layout block weights (device tensors, borrowed by fwd/bwd)."""
    var qkv_w: TArc     # [3*inner, hidden] rows [q|k|v]
    var out_w: TArc     # [hidden, inner]
    var fc1_w: TArc     # [2*ffn, hidden] rows [gate|value]
    var fc2_w: TArc     # [hidden, ffn]
    var q_norm: TArc    # [head_dim]
    var k_norm: TArc    # [head_dim]
    var norm1_w: TArc   # [hidden]
    var norm2_w: TArc   # [hidden]

    def __init__(
        out self,
        var qkv_w: TArc, var out_w: TArc, var fc1_w: TArc, var fc2_w: TArc,
        var q_norm: TArc, var k_norm: TArc, var norm1_w: TArc, var norm2_w: TArc,
    ):
        self.qkv_w = qkv_w^
        self.out_w = out_w^
        self.fc1_w = fc1_w^
        self.fc2_w = fc2_w^
        self.q_norm = q_norm^
        self.k_norm = k_norm^
        self.norm1_w = norm1_w^
        self.norm2_w = norm2_w^


struct H3BlockTrainSaved(Copyable, Movable):
    """Activations the backward consumes (device-resident, bf16)."""
    var x: TArc          # [S, D] block input
    var mod_rows: TArc   # [S, 6D] gathered modulation rows
    var n1: TArc         # [S, D] rms_norm(x, norm1)
    var n1m: TArc        # [S, D] modulated (attn-branch input)
    var q_pre: TArc      # [1,S,H,Dh]
    var k_pre: TArc      # [1,S,H,Dh]
    var v: TArc          # [1,S,H,Dh]
    var q_rope: TArc     # [1,S,H,Dh] post qk-norm + rope
    var k_rope: TArc     # [1,S,H,Dh]
    var att_flat: TArc   # [S, inner]
    var attn_y: TArc     # [S, D] out_proj output (pre-gate)
    var h_mid: TArc      # [S, D]
    var n2: TArc         # [S, D]
    var n2m: TArc        # [S, D]
    var fc_gate: TArc    # [S, F]
    var fc_value: TArc   # [S, F]
    var swi: TArc        # [S, F] silu(gate)*value
    var ff_y: TArc       # [S, D] fc2 output (pre-gate)

    def __init__(
        out self,
        var x: TArc, var mod_rows: TArc, var n1: TArc, var n1m: TArc,
        var q_pre: TArc, var k_pre: TArc, var v: TArc,
        var q_rope: TArc, var k_rope: TArc,
        var att_flat: TArc, var attn_y: TArc, var h_mid: TArc,
        var n2: TArc, var n2m: TArc,
        var fc_gate: TArc, var fc_value: TArc, var swi: TArc, var ff_y: TArc,
    ):
        self.x = x^
        self.mod_rows = mod_rows^
        self.n1 = n1^
        self.n1m = n1m^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.att_flat = att_flat^
        self.attn_y = attn_y^
        self.h_mid = h_mid^
        self.n2 = n2^
        self.n2m = n2m^
        self.fc_gate = fc_gate^
        self.fc_value = fc_value^
        self.swi = swi^
        self.ff_y = ff_y^


struct H3BlockTrainForward(Copyable, Movable):
    var out: TArc        # [S, D]
    var saved: H3BlockTrainSaved

    def __init__(out self, var out: TArc, var saved: H3BlockTrainSaved):
        self.out = out^
        self.saved = saved^


struct H3BlockTrainGrads(Copyable, Movable):
    """d_x + every weight grad (raw layout) + the mod-table grad."""
    var d_x: TArc        # [S, D]
    var d_qkv_w: TArc    # [3*inner, hidden]
    var d_out_w: TArc    # [hidden, inner]
    var d_fc1_w: TArc    # [2*ffn, hidden]
    var d_fc2_w: TArc    # [hidden, ffn]
    var d_q_norm: TArc   # [head_dim]
    var d_k_norm: TArc   # [head_dim]
    var d_norm1_w: TArc  # [hidden]
    var d_norm2_w: TArc  # [hidden]
    var d_mod: TArc      # [mod_rows, 6D] scatter-added table grad

    def __init__(
        out self,
        var d_x: TArc, var d_qkv_w: TArc, var d_out_w: TArc,
        var d_fc1_w: TArc, var d_fc2_w: TArc,
        var d_q_norm: TArc, var d_k_norm: TArc,
        var d_norm1_w: TArc, var d_norm2_w: TArc, var d_mod: TArc,
    ):
        self.d_x = d_x^
        self.d_qkv_w = d_qkv_w^
        self.d_out_w = d_out_w^
        self.d_fc1_w = d_fc1_w^
        self.d_fc2_w = d_fc2_w^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_norm1_w = d_norm1_w^
        self.d_norm2_w = d_norm2_w^
        self.d_mod = d_mod^


def _partial_rope(
    x: Tensor, cos: Tensor, sin: Tensor, heads: Int, rotary_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Rotate the first `rotary_dim` channels of [1,S,H,Dh]; pass the rest.
    Mirrors the inference `_minimax_h3_apply_partial_rope`."""
    var sh = x.shape()
    var head_dim = sh[len(sh) - 1]
    var x_rot = slice(x, 3, 0, rotary_dim, ctx)
    var x_pass = slice(x, 3, rotary_dim, head_dim - rotary_dim, ctx)
    var rotated = rope_halfsplit_full_head_broadcast(x_rot, cos, sin, heads, ctx)
    return concat(3, ctx, rotated, x_pass)


def _expand_table(
    tbl: Tensor, s: Int, heads: Int, rotary_dim: Int, ctx: DeviceContext
) raises -> Tensor:
    """[S, rotary] -> [S*H, rotary] token-major head repeat (broadcast-add)."""
    var t3 = reshape(tbl, [s, 1, rotary_dim], ctx)
    var zshape: List[Int] = [s, heads, rotary_dim]
    var zeros = full_device(zshape^, Float32(0.0), tbl.dtype(), ctx)
    var bc = add(t3, zeros, ctx)
    return reshape(bc, [s * heads, rotary_dim], ctx)


def _partial_rope_backward(
    d_y: Tensor, cos_x: Tensor, sin_x: Tensor, heads: Int, rotary_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Backward of `_partial_rope`: true adjoint on the rotated prefix
    (arbitrary full-width tables), identity on the passthrough tail.
    cos_x/sin_x are the HEAD-EXPANDED [S*H, rotary] tables."""
    var sh = d_y.shape()
    var head_dim = sh[len(sh) - 1]
    var d_rot = slice(d_y, 3, 0, rotary_dim, ctx)
    var d_pass = slice(d_y, 3, rotary_dim, head_dim - rotary_dim, ctx)
    var rows = sh[1] * sh[2]
    var d_rot_2d = reshape(d_rot, [rows, rotary_dim], ctx)
    var d_back = rope_halfsplit_full_backward(d_rot_2d, cos_x, sin_x, ctx)
    var d_back4 = reshape(d_back, [sh[0], sh[1], sh[2], rotary_dim], ctx)
    return concat(3, ctx, d_back4, d_pass)


def _row_modulate(
    n: Tensor, scale_rows: Tensor, shift_rows: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """o = (1 + scale_rows) * n + shift_rows, everything [S, D]."""
    var sp1 = add_scalar(scale_rows, Float32(1.0), ctx)
    return add(mul(n, sp1, ctx), shift_rows, ctx)


def h3_block_train_forward[
    H: Int, Dh: Int, S: Int
](
    x_in: Tensor,
    w: H3BlockTrainWeights,
    mod: Tensor,               # [mod_rows, 6D] precomputed modulation table
    adaln_indices: List[Int],  # length S
    cos: Tensor, sin: Tensor,  # [S, rotary_dim] f32
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3BlockTrainForward:
    comptime I = H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var no_bias = Optional[Tensor](None)

    var x = x_in.clone(ctx)

    var mod_rows = gather_rows(mod, adaln_indices, ctx)   # [S, 6D]
    var shift_msa = slice(mod_rows, 1, 0 * D, D, ctx)
    var scale_msa = slice(mod_rows, 1, 1 * D, D, ctx)
    var gate_msa = slice(mod_rows, 1, 2 * D, D, ctx)
    var shift_mlp = slice(mod_rows, 1, 3 * D, D, ctx)
    var scale_mlp = slice(mod_rows, 1, 4 * D, D, ctx)
    var gate_mlp = slice(mod_rows, 1, 5 * D, D, ctx)

    # ── attention branch ──
    var n1 = rms_norm(x, w.norm1_w[], eps, ctx)
    var n1m = _row_modulate(n1, scale_msa, shift_msa, ctx)

    var qkv = linear(n1m, w.qkv_w[], no_bias, ctx)        # [S, 3I]
    var q_flat = slice(qkv, 1, 0 * I, I, ctx)
    var k_flat = slice(qkv, 1, 1 * I, I, ctx)
    var v_flat = slice(qkv, 1, 2 * I, I, ctx)
    var q_pre = _reshaped(q_flat, [1, S, H, Dh], ctx)
    var k_pre = _reshaped(k_flat, [1, S, H, Dh], ctx)
    var v = _reshaped(v_flat, [1, S, H, Dh], ctx)

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = _partial_rope(q_rms, cos, sin, H, rotary_dim, ctx)
    var k_rope = _partial_rope(k_rms, cos, sin, H, rotary_dim, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = _reshaped(att, [S, I], ctx)
    var attn_y = linear(att_flat, w.out_w[], no_bias, ctx)  # [S, D]
    var h_mid = add(x, mul(gate_msa, attn_y, ctx), ctx)

    # ── mlp branch ──
    var n2 = rms_norm(h_mid, w.norm2_w[], eps, ctx)
    var n2m = _row_modulate(n2, scale_mlp, shift_mlp, ctx)
    var fc1_out = linear(n2m, w.fc1_w[], no_bias, ctx)    # [S, 2F] [gate|value]
    var fc_gate = slice(fc1_out, 1, 0, F, ctx)
    var fc_value = slice(fc1_out, 1, F, F, ctx)
    var swi = swiglu(fc_gate, fc_value, ctx)              # silu(gate)*value
    var ff_y = linear(swi, w.fc2_w[], no_bias, ctx)       # [S, D]
    var out = add(h_mid, mul(gate_mlp, ff_y, ctx), ctx)

    var saved = H3BlockTrainSaved(
        TArc(x^), TArc(mod_rows^), TArc(n1^), TArc(n1m^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(attn_y^), TArc(h_mid^),
        TArc(n2^), TArc(n2m^),
        TArc(fc_gate^), TArc(fc_value^), TArc(swi^), TArc(ff_y^),
    )
    return H3BlockTrainForward(TArc(out^), saved^)


def h3_block_train_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: Tensor,             # [S, D] upstream grad
    w: H3BlockTrainWeights,
    saved: H3BlockTrainSaved,
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    mod_table_rows: Int,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3BlockTrainGrads:
    comptime I = H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var shift_msa = slice(saved.mod_rows[], 1, 0 * D, D, ctx)
    var scale_msa = slice(saved.mod_rows[], 1, 1 * D, D, ctx)
    var gate_msa = slice(saved.mod_rows[], 1, 2 * D, D, ctx)
    var scale_mlp = slice(saved.mod_rows[], 1, 4 * D, D, ctx)
    var gate_mlp = slice(saved.mod_rows[], 1, 5 * D, D, ctx)
    _ = shift_msa^  # shifts carry no grad dependency (o linear in shift)

    # out = h_mid + gate_mlp ⊙ ff_y
    var d_h_mid = d_out.clone(ctx)
    var d_ff_y = mul(d_out, gate_mlp, ctx)
    var d_gate_mlp = mul(d_out, saved.ff_y[], ctx)

    # ff_y = linear(swi, fc2_w)
    var lb_fc2 = linear_backward(d_ff_y, saved.swi[], w.fc2_w[], S, F, D, ctx)

    # swi = silu(gate) * value
    var sgb = swiglu_backward(lb_fc2.d_x, saved.fc_gate[], saved.fc_value[], ctx)
    var d_fc1_out = concat(1, ctx, sgb.d_gate, sgb.d_up)   # [gate|value] order

    # fc1_out = linear(n2m, fc1_w)
    var lb_fc1 = linear_backward(d_fc1_out, saved.n2m[], w.fc1_w[], S, D, 2 * F, ctx)

    # n2m = (1+scale_mlp) ⊙ n2 + shift_mlp
    var sp1_mlp = add_scalar(scale_mlp, Float32(1.0), ctx)
    var d_n2 = mul(lb_fc1.d_x, sp1_mlp, ctx)
    var d_scale_mlp = mul(lb_fc1.d_x, saved.n2[], ctx)
    var d_shift_mlp = lb_fc1.d_x.clone(ctx)

    # n2 = rms_norm(h_mid, norm2_w)
    var rb_n2 = rms_norm_backward(d_n2, saved.h_mid[], w.norm2_w[], eps, ctx)
    d_h_mid = add(d_h_mid, rb_n2.d_x, ctx)

    # h_mid = x + gate_msa ⊙ attn_y
    var d_x = d_h_mid.clone(ctx)
    var d_attn_y = mul(d_h_mid, gate_msa, ctx)
    var d_gate_msa = mul(d_h_mid, saved.attn_y[], ctx)

    # attn_y = linear(att_flat, out_w)
    var lb_out = linear_backward(d_attn_y, saved.att_flat[], w.out_w[], S, I, D, ctx)

    # sdpa
    var d_att = _reshaped(lb_out.d_x, [1, S, H, Dh], ctx)
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], d_att, scale, ctx,
    )

    # partial rope backward (cos/sin constant; head-expanded tables)
    var cos_x = _expand_table(cos, S, H, rotary_dim, ctx)
    var sin_x = _expand_table(sin, S, H, rotary_dim, ctx)
    var d_q_rms = _partial_rope_backward(sb.d_q, cos_x, sin_x, H, rotary_dim, ctx)
    var d_k_rms = _partial_rope_backward(sb.d_k, cos_x, sin_x, H, rotary_dim, ctx)

    # qk rms_norm backward
    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)

    # d_qkv [S, 3I] in raw [q|k|v] order
    var d_q_flat = _reshaped(rb_q.d_x, [S, I], ctx)
    var d_k_flat = _reshaped(rb_k.d_x, [S, I], ctx)
    var d_v_flat = _reshaped(sb.d_v, [S, I], ctx)
    var d_qkv = concat(1, ctx, d_q_flat, d_k_flat, d_v_flat)

    # qkv = linear(n1m, qkv_w)
    var lb_qkv = linear_backward(d_qkv, saved.n1m[], w.qkv_w[], S, D, 3 * I, ctx)

    # n1m = (1+scale_msa) ⊙ n1 + shift_msa
    var sp1_msa = add_scalar(scale_msa, Float32(1.0), ctx)
    var d_n1 = mul(lb_qkv.d_x, sp1_msa, ctx)
    var d_scale_msa = mul(lb_qkv.d_x, saved.n1[], ctx)
    var d_shift_msa = lb_qkv.d_x.clone(ctx)

    # n1 = rms_norm(x, norm1_w); x feeds residual AND norm -> SUM
    var rb_n1 = rms_norm_backward(d_n1, saved.x[], w.norm1_w[], eps, ctx)
    d_x = add(d_x, rb_n1.d_x, ctx)

    # mod-table grad: concat the six [S,D] row-grads then scatter-add by index
    var d_mod_rows = concat(
        1, ctx, d_shift_msa, d_scale_msa, d_gate_msa,
        d_shift_mlp, d_scale_mlp, d_gate_mlp,
    )
    var mod_shape: List[Int] = [mod_table_rows, 6 * D]
    var d_mod = index_select_backward(d_mod_rows, adaln_indices, 0, mod_shape^, ctx)

    return H3BlockTrainGrads(
        TArc(d_x^), TArc(lb_qkv.d_w.clone(ctx)^), TArc(lb_out.d_w.clone(ctx)^),
        TArc(lb_fc1.d_w.clone(ctx)^), TArc(lb_fc2.d_w.clone(ctx)^),
        TArc(rb_q.d_g.clone(ctx)^), TArc(rb_k.d_g.clone(ctx)^),
        TArc(rb_n1.d_g.clone(ctx)^), TArc(rb_n2.d_g.clone(ctx)^),
        TArc(d_mod^),
    )


# ═══════════════════════════════════════════════════════════════════════════
# LoRA VARIANT — the four torchref targets (qkv_proj / out_proj / fc1 / fc2).
# Adapters are additive overlays y += scale·B(A(x)) on top of the frozen base
# projections; math + device helpers are the gated Klein LoRA path
# (models/klein/lora_block.mojo). With all four adapters absent this reduces
# byte-for-byte to the base block above.
# ═══════════════════════════════════════════════════════════════════════════
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice,
    klein_lora_fwd_device_resident,
    klein_lora_bwd_device_resident_tensors,
    KleinLoraDeviceGradTensors,
)


struct H3BlockLoraDevice(Copyable, Movable):
    var qkv: Optional[LoraAdapterDevice]
    var out: Optional[LoraAdapterDevice]
    var fc1: Optional[LoraAdapterDevice]
    var fc2: Optional[LoraAdapterDevice]

    def __init__(
        out self,
        var qkv: Optional[LoraAdapterDevice], var out_: Optional[LoraAdapterDevice],
        var fc1: Optional[LoraAdapterDevice], var fc2: Optional[LoraAdapterDevice],
    ):
        self.qkv = qkv^
        self.out = out_^
        self.fc1 = fc1^
        self.fc2 = fc2^


struct H3BlockLoraGrads(Copyable, Movable):
    var qkv: Optional[KleinLoraDeviceGradTensors]
    var out: Optional[KleinLoraDeviceGradTensors]
    var fc1: Optional[KleinLoraDeviceGradTensors]
    var fc2: Optional[KleinLoraDeviceGradTensors]

    def __init__(
        out self,
        var qkv: Optional[KleinLoraDeviceGradTensors],
        var out_: Optional[KleinLoraDeviceGradTensors],
        var fc1: Optional[KleinLoraDeviceGradTensors],
        var fc2: Optional[KleinLoraDeviceGradTensors],
    ):
        self.qkv = qkv^
        self.out = out_^
        self.fc1 = fc1^
        self.fc2 = fc2^


def _lora_add(
    base: Tensor, x: Tensor, lo: Optional[LoraAdapterDevice], rows: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    if not lo:
        return base.clone(ctx)
    var contrib = klein_lora_fwd_device_resident(x, lo.value(), rows, ctx)
    return add(base, contrib, ctx)


def h3_block_train_forward_lora[
    H: Int, Dh: Int, S: Int
](
    x_in: Tensor,
    w: H3BlockTrainWeights,
    lora: H3BlockLoraDevice,
    mod: Tensor,
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3BlockTrainForward:
    comptime I = H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var no_bias = Optional[Tensor](None)

    var x = x_in.clone(ctx)
    var mod_rows = gather_rows(mod, adaln_indices, ctx)
    var shift_msa = slice(mod_rows, 1, 0 * D, D, ctx)
    var scale_msa = slice(mod_rows, 1, 1 * D, D, ctx)
    var gate_msa = slice(mod_rows, 1, 2 * D, D, ctx)
    var shift_mlp = slice(mod_rows, 1, 3 * D, D, ctx)
    var scale_mlp = slice(mod_rows, 1, 4 * D, D, ctx)
    var gate_mlp = slice(mod_rows, 1, 5 * D, D, ctx)

    var n1 = rms_norm(x, w.norm1_w[], eps, ctx)
    var n1m = _row_modulate(n1, scale_msa, shift_msa, ctx)

    var qkv_base = linear(n1m, w.qkv_w[], no_bias, ctx)
    var qkv = _lora_add(qkv_base, n1m, lora.qkv, S, ctx)
    var q_flat = slice(qkv, 1, 0 * I, I, ctx)
    var k_flat = slice(qkv, 1, 1 * I, I, ctx)
    var v_flat = slice(qkv, 1, 2 * I, I, ctx)
    var q_pre = _reshaped(q_flat, [1, S, H, Dh], ctx)
    var k_pre = _reshaped(k_flat, [1, S, H, Dh], ctx)
    var v = _reshaped(v_flat, [1, S, H, Dh], ctx)

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = _partial_rope(q_rms, cos, sin, H, rotary_dim, ctx)
    var k_rope = _partial_rope(k_rms, cos, sin, H, rotary_dim, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = _reshaped(att, [S, I], ctx)
    var attn_base = linear(att_flat, w.out_w[], no_bias, ctx)
    var attn_y = _lora_add(attn_base, att_flat, lora.out, S, ctx)
    var h_mid = add(x, mul(gate_msa, attn_y, ctx), ctx)

    var n2 = rms_norm(h_mid, w.norm2_w[], eps, ctx)
    var n2m = _row_modulate(n2, scale_mlp, shift_mlp, ctx)
    var fc1_base = linear(n2m, w.fc1_w[], no_bias, ctx)
    var fc1_out = _lora_add(fc1_base, n2m, lora.fc1, S, ctx)
    var fc_gate = slice(fc1_out, 1, 0, F, ctx)
    var fc_value = slice(fc1_out, 1, F, F, ctx)
    var swi = swiglu(fc_gate, fc_value, ctx)
    var ff_base = linear(swi, w.fc2_w[], no_bias, ctx)
    var ff_y = _lora_add(ff_base, swi, lora.fc2, S, ctx)
    var out = add(h_mid, mul(gate_mlp, ff_y, ctx), ctx)

    var saved = H3BlockTrainSaved(
        TArc(x^), TArc(mod_rows^), TArc(n1^), TArc(n1m^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(attn_y^), TArc(h_mid^),
        TArc(n2^), TArc(n2m^),
        TArc(fc_gate^), TArc(fc_value^), TArc(swi^), TArc(ff_y^),
    )
    return H3BlockTrainForward(TArc(out^), saved^)


struct H3BlockTrainLoraBackward(Copyable, Movable):
    var base: H3BlockTrainGrads
    var lora: H3BlockLoraGrads

    def __init__(out self, var base: H3BlockTrainGrads, var lora: H3BlockLoraGrads):
        self.base = base^
        self.lora = lora^


def h3_block_train_backward_lora[
    H: Int, Dh: Int, S: Int
](
    d_out: Tensor,
    w: H3BlockTrainWeights,
    lora: H3BlockLoraDevice,
    saved: H3BlockTrainSaved,
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    mod_table_rows: Int,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3BlockTrainLoraBackward:
    comptime I = H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var shift_msa = slice(saved.mod_rows[], 1, 0 * D, D, ctx)
    var scale_msa = slice(saved.mod_rows[], 1, 1 * D, D, ctx)
    var gate_msa = slice(saved.mod_rows[], 1, 2 * D, D, ctx)
    var scale_mlp = slice(saved.mod_rows[], 1, 4 * D, D, ctx)
    var gate_mlp = slice(saved.mod_rows[], 1, 5 * D, D, ctx)
    _ = shift_msa^

    var d_h_mid = d_out.clone(ctx)
    var d_ff_y = mul(d_out, gate_mlp, ctx)
    var d_gate_mlp = mul(d_out, saved.ff_y[], ctx)

    # ff_y = fc2(swi) + lora_fc2(swi): both branches see d_ff_y; d_swi sums.
    var lb_fc2 = linear_backward(d_ff_y, saved.swi[], w.fc2_w[], S, F, D, ctx)
    var d_swi = lb_fc2.d_x.clone(ctx)
    var lg_fc2 = Optional[KleinLoraDeviceGradTensors](None)
    if lora.fc2:
        var g = klein_lora_bwd_device_resident_tensors(
            d_ff_y, saved.swi[], lora.fc2.value(), S, ctx
        )
        d_swi = add(d_swi, g.d_x[], ctx)
        lg_fc2 = Optional[KleinLoraDeviceGradTensors](g^)

    var sgb = swiglu_backward(d_swi, saved.fc_gate[], saved.fc_value[], ctx)
    var d_fc1_out = concat(1, ctx, sgb.d_gate, sgb.d_up)

    var lb_fc1 = linear_backward(d_fc1_out, saved.n2m[], w.fc1_w[], S, D, 2 * F, ctx)
    var d_n2m = lb_fc1.d_x.clone(ctx)
    var lg_fc1 = Optional[KleinLoraDeviceGradTensors](None)
    if lora.fc1:
        var g = klein_lora_bwd_device_resident_tensors(
            d_fc1_out, saved.n2m[], lora.fc1.value(), S, ctx
        )
        d_n2m = add(d_n2m, g.d_x[], ctx)
        lg_fc1 = Optional[KleinLoraDeviceGradTensors](g^)

    var sp1_mlp = add_scalar(scale_mlp, Float32(1.0), ctx)
    var d_n2 = mul(d_n2m, sp1_mlp, ctx)
    var d_scale_mlp = mul(d_n2m, saved.n2[], ctx)
    var d_shift_mlp = d_n2m.clone(ctx)

    var rb_n2 = rms_norm_backward(d_n2, saved.h_mid[], w.norm2_w[], eps, ctx)
    d_h_mid = add(d_h_mid, rb_n2.d_x, ctx)

    var d_x = d_h_mid.clone(ctx)
    var d_attn_y = mul(d_h_mid, gate_msa, ctx)
    var d_gate_msa = mul(d_h_mid, saved.attn_y[], ctx)

    var lb_out = linear_backward(d_attn_y, saved.att_flat[], w.out_w[], S, I, D, ctx)
    var d_att_flat = lb_out.d_x.clone(ctx)
    var lg_out = Optional[KleinLoraDeviceGradTensors](None)
    if lora.out:
        var g = klein_lora_bwd_device_resident_tensors(
            d_attn_y, saved.att_flat[], lora.out.value(), S, ctx
        )
        d_att_flat = add(d_att_flat, g.d_x[], ctx)
        lg_out = Optional[KleinLoraDeviceGradTensors](g^)

    var d_att = _reshaped(d_att_flat, [1, S, H, Dh], ctx)
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], d_att, scale, ctx,
    )

    var cos_x = _expand_table(cos, S, H, rotary_dim, ctx)
    var sin_x = _expand_table(sin, S, H, rotary_dim, ctx)
    var d_q_rms = _partial_rope_backward(sb.d_q, cos_x, sin_x, H, rotary_dim, ctx)
    var d_k_rms = _partial_rope_backward(sb.d_k, cos_x, sin_x, H, rotary_dim, ctx)

    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)

    var d_q_flat = _reshaped(rb_q.d_x, [S, I], ctx)
    var d_k_flat = _reshaped(rb_k.d_x, [S, I], ctx)
    var d_v_flat = _reshaped(sb.d_v, [S, I], ctx)
    var d_qkv = concat(1, ctx, d_q_flat, d_k_flat, d_v_flat)

    var lb_qkv = linear_backward(d_qkv, saved.n1m[], w.qkv_w[], S, D, 3 * I, ctx)
    var d_n1m = lb_qkv.d_x.clone(ctx)
    var lg_qkv = Optional[KleinLoraDeviceGradTensors](None)
    if lora.qkv:
        var g = klein_lora_bwd_device_resident_tensors(
            d_qkv, saved.n1m[], lora.qkv.value(), S, ctx
        )
        d_n1m = add(d_n1m, g.d_x[], ctx)
        lg_qkv = Optional[KleinLoraDeviceGradTensors](g^)

    var sp1_msa = add_scalar(scale_msa, Float32(1.0), ctx)
    var d_n1 = mul(d_n1m, sp1_msa, ctx)
    var d_scale_msa = mul(d_n1m, saved.n1[], ctx)
    var d_shift_msa = d_n1m.clone(ctx)

    var rb_n1 = rms_norm_backward(d_n1, saved.x[], w.norm1_w[], eps, ctx)
    d_x = add(d_x, rb_n1.d_x, ctx)

    var d_mod_rows = concat(
        1, ctx, d_shift_msa, d_scale_msa, d_gate_msa,
        d_shift_mlp, d_scale_mlp, d_gate_mlp,
    )
    var mod_shape: List[Int] = [mod_table_rows, 6 * D]
    var d_mod = index_select_backward(d_mod_rows, adaln_indices, 0, mod_shape^, ctx)

    var base = H3BlockTrainGrads(
        TArc(d_x^), TArc(lb_qkv.d_w.clone(ctx)^), TArc(lb_out.d_w.clone(ctx)^),
        TArc(lb_fc1.d_w.clone(ctx)^), TArc(lb_fc2.d_w.clone(ctx)^),
        TArc(rb_q.d_g.clone(ctx)^), TArc(rb_k.d_g.clone(ctx)^),
        TArc(rb_n1.d_g.clone(ctx)^), TArc(rb_n2.d_g.clone(ctx)^),
        TArc(d_mod^),
    )
    var lgrads = H3BlockLoraGrads(lg_qkv^, lg_out^, lg_fc1^, lg_fc2^)
    return H3BlockTrainLoraBackward(base^, lgrads^)
