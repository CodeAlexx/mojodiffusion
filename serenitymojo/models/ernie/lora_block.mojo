# serenitymojo/models/ernie/lora_block.mojo
#
# LoRA-ON-PROJECTION for the ERNIE-Image single-stream block. Mirrors the PROVEN
# Klein LoRA template (models/klein/lora_block.mojo + single_block.mojo's
# single_block_lora_*), specialized to ERNIE's SEVEN un-fused target projections:
#   self_attention.{to_q, to_k, to_v, to_out.0}  and  mlp.{gate_proj, up_proj, linear_fc2}
# (ERNIE has separate q/k/v — unlike Klein's fused qkv — so 7 separate adapters.)
#
# WHY THE TRAINER MATH IS THE AUTHORITY (identical to Klein lora_block.mojo)
#   For a projection y = linear(x, W) (W [out,in]), the LoRA-adapted output is
#       y' = linear(x, W) + scale·((x @ Aᵀ) @ Bᵀ)
#   A=[rank,in], B=[out,rank], scale = alpha/rank. This MATCHES the inference merge
#   in inference-flame ernie_image.rs `lora.apply(...)` (W' = W + scale·B@A).
#
#   BACKWARD (given d_y' at the projection output, x the saved projection input):
#       d_dy = scale·d_y'                    [M,out]
#       d_B  = d_dyᵀ @ t   (t = x @ Aᵀ)      [out,rank]
#       d_t  = d_dy  @ B                      [M,rank]
#       d_A  = d_tᵀ  @ x                      [rank,in]
#       d_x  = d_t   @ A                      [M,in]   (LoRA branch's contribution
#                                                       to the projection INPUT grad)
#   The base path (frozen W) ALSO yields d_x_base = d_y' @ W; the caller SUMS d_x
#   into that. d_A/d_B go to the optimizer; the base W grad is discarded for LoRA.
#
# These two helpers are byte-identical to train_step._lora_fwd / _lora_bwd (the
# authority), plus the d_x term _lora_bwd discards (which the block needs to thread
# the LoRA grad back into the projection input — same addition Klein lora_block makes).
#
# NO NEW ops/ PRIMITIVE: forward = two linear()s; backward = two linear_backward()s.
#
# Mojo 1.0.0b1: def not fn; Tensor move-only crosses API as host List[Float32].

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx, linear_backward_dw

# REUSE the trainer's LoRA structs (the target authority).
from serenitymojo.training.train_step import LoraAdapter, LoraGrads

# Forward + backward ops shared with the base block (Tenet 1: nothing new here).
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import gelu_exact
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_halfsplit_full
from serenitymojo.ops.attention import sdpa_nomask, sdpa
from serenitymojo.ops.tensor_algebra import reshape_owned, reshape_in_place, mul, add, mul_scalar
from serenitymojo.ops.norm_backward import rms_norm_backward, rms_norm_backward_dx
from serenitymojo.ops.activation_backward import gelu_exact_backward
from serenitymojo.ops.attention_backward import sdpa_backward, sdpa_backward_masked, SdpaGrads
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, rope_halfsplit_full_backward,
)
from serenitymojo.models.ernie.weights import ErnieBlockWeights
from serenitymojo.models.ernie.block import (
    ErnieModVecs, ErnieBlockSaved, ErnieBlockForward, ErnieBlockGrads,
)


comptime TArc = ArcPointer[Tensor]


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


# ── ERNIE text-pad attention mask (mirrors transformer_ernie_image.py:393-401) ─
# OneTrainer's ErnieImageTransformer2DModel builds a per-key bool mask
#   [ones(N_img), arange(Tmax) < text_lens]
# (True=attend, False=mask out), broadcast over query rows and heads, so PADDED
# TEXT KEYS contribute nothing to attention. Token order is IMAGE-first then
# TEXT-second (S = N_IMG + N_TXT). Here `real_len` is the per-sample real token
# count; key columns j in [N_IMG, N_IMG+real_len) stay (text), j in
# [N_IMG+real_len, S) are padded text and get -1e4 (the additive form of the
# bool mask). Image columns [0, N_IMG) are never masked. The mask is the SAME
# for every query row and every head (no causality), so the host list is
# heads*S*S floats: 0.0 where attend, -1e4 where padded text.
#
# Two consumers, two tensor shapes from the ONE host list (Tensor.from_host
# casts F32->dtype): the forward `sdpa` wants [1,H,S,S] in q's compute dtype;
# `sdpa_backward_masked` wants [H*S, S] F32. Built once per step in the stack
# and threaded to every block (constant across the 36 blocks).
def _ernie_text_pad_mask_host(
    heads: Int, S: Int, n_img: Int, real_len: Int
) -> List[Float32]:
    var neg = Float32(-1.0e4)
    var first_pad = n_img + real_len   # first padded-text key column
    var data = List[Float32]()
    for _hh in range(heads):
        for _i in range(S):
            for j in range(S):
                if j >= first_pad:
                    data.append(neg)           # padded text key: blocked
                else:
                    data.append(Float32(0.0))  # image + real text: attend
    return data^


def build_ernie_text_pad_mask_fwd(
    heads: Int, S: Int, n_img: Int, real_len: Int,
    dtype: STDtype, ctx: DeviceContext,
) raises -> Tensor:
    """Forward additive mask [1, H, S, S] in `dtype` (q's compute dtype) for
    `sdpa`. Masks padded-text key columns [n_img+real_len, S)."""
    var data = _ernie_text_pad_mask_host(heads, S, n_img, real_len)
    return Tensor.from_host(data^, [1, heads, S, S], dtype, ctx)


def build_ernie_text_pad_mask_bwd(
    heads: Int, S: Int, n_img: Int, real_len: Int, ctx: DeviceContext,
) raises -> Tensor:
    """Backward additive mask [H*S, S] F32 for `sdpa_backward_masked`. Same
    masked columns as the forward mask (per-head, broadcast over B)."""
    var data = _ernie_text_pad_mask_host(heads, S, n_img, real_len)
    return Tensor.from_host(data^, [heads * S, S], STDtype.F32, ctx)


# Adapter forward contribution on x [M,in] -> [M,out] (host list in/out).
# Byte-identical to train_step._lora_fwd.
def ernie_lora_fwd(
    x_h: List[Float32], lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var nb1 = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb1^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        nb2^, ctx,
    ).to_host(ctx)                                   # [M,out]
    var out = List[Float32]()
    for i in range(len(dy)):
        out.append(lo.scale * dy[i])
    return out^


# Optionally-applied adapter forward: if `lo` is present, return base_y + LoRA;
# else return base_y unchanged (base-path no-regression when an adapter is absent).
def ernie_lora_apply(
    base_y: List[Float32], x_h: List[Float32], lo: Optional[LoraAdapter],
    M: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    if not lo:
        return base_y.copy()
    var contrib = ernie_lora_fwd(x_h, lo.value(), M, ctx)
    var out = List[Float32]()
    for i in range(len(base_y)):
        out.append(base_y[i] + contrib[i])
    return out^


# LoRA backward that ALSO returns the LoRA branch's contribution to d_x.
# d_a/d_b match train_step._lora_bwd exactly; d_x is the term that file drops.
struct ErnieLoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]   # LoRA contribution to the projection INPUT grad [M,in]

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def ernie_lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> ErnieLoraGrads:
    # t = x @ Aᵀ  (recompute; cheap)
    var nb_t = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb_t^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var d_dy = List[Float32]()
    for i in range(len(d_contrib_h)):
        d_dy.append(lo.scale * d_contrib_h[i])       # [M,out]
    # dy = t @ Bᵀ  -> d_B (d_w) and d_t (d_x)
    var lbB = linear_backward(
        Tensor.from_host(d_dy^, [M, lo.out_f], STDtype.BF16, ctx),
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.to_host(ctx)                   # [M,rank]
    var d_b = lbB.d_w.to_host(ctx)                   # [out_f,rank]
    # t = x @ Aᵀ  -> d_A (d_w) and d_x_lo (d_x)
    var lbA = linear_backward(
        Tensor.from_host(d_t^, [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_x_lo = lbA.d_x.to_host(ctx)                # [M,in_f]
    var d_a = lbA.d_w.to_host(ctx)                   # [rank,in_f]
    return ErnieLoraGrads(d_a^, d_b^, d_x_lo^)


# ── device-resident LoRA adapter and helpers ─────────────────────────────────
# A/B live as device tensors for the duration of a training step. This removes
# the per-projection A/B upload and activation readback/re-upload bridge.
struct ErnieLoraAdapterDevice(Copyable, Movable):
    var a: TArc
    var b: TArc
    var rank: Int
    var in_f: Int
    var out_f: Int
    var scale: Float32

    def __init__(
        out self, var a: TArc, var b: TArc,
        rank: Int, in_f: Int, out_f: Int, scale: Float32,
    ):
        self.a = a^
        self.b = b^
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f
        self.scale = scale


def ernie_lora_adapter_to_device(
    lo: LoraAdapter, dtype: STDtype, ctx: DeviceContext
) raises -> ErnieLoraAdapterDevice:
    var a = Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)
    var b = Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)
    if dtype != STDtype.BF16:
        a = cast_tensor(a^, dtype, ctx)
        b = cast_tensor(b^, dtype, ctx)
    return ErnieLoraAdapterDevice(
        TArc(a^),
        TArc(b^),
        lo.rank, lo.in_f, lo.out_f, lo.scale,
    )


def ernie_lora_apply_device(
    var base_y: Tensor, x: Tensor, lo: ErnieLoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    var nb1 = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb1^, ctx)             # [M,rank]
    var nb2 = Optional[Tensor](None)
    var dy = linear(t, lo.b[], nb2^, ctx)            # [M,out]
    var contrib = mul_scalar(dy, lo.scale, ctx)
    return add(base_y^, contrib^, ctx)


struct ErnieLoraDeviceGradTensors(Copyable, Movable):
    var d_a: TArc
    var d_b: TArc
    var d_x: TArc

    def __init__(out self, var d_a: TArc, var d_b: TArc, var d_x: TArc):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def ernie_lora_bwd_device_tensors(
    d_contrib: Tensor, x: Tensor, lo: ErnieLoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> ErnieLoraDeviceGradTensors:
    var nb_t = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb_t^, ctx)            # [M,rank]
    var d_dy = mul_scalar(d_contrib, lo.scale, ctx)  # [M,out]

    var d_t = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)
    var d_b_t = linear_backward_dw(d_dy, t, M, lo.rank, lo.out_f, ctx)

    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)
    var d_a_t = linear_backward_dw(d_t, x, M, lo.in_f, lo.rank, ctx)
    return ErnieLoraDeviceGradTensors(TArc(d_a_t^), TArc(d_b_t^), TArc(d_x_lo^))


# ── per-block LoRA carrier: the 7 optional adapters (slot order is canonical) ──
# slot 0 to_q, 1 to_k, 2 to_v, 3 to_out.0, 4 gate_proj, 5 up_proj, 6 linear_fc2.
comptime ERNIE_SLOTS = 7
comptime SLOT_Q = 0
comptime SLOT_K = 1
comptime SLOT_V = 2
comptime SLOT_O = 3
comptime SLOT_GATE = 4
comptime SLOT_UP = 5
comptime SLOT_DOWN = 6


struct ErnieBlockLora(Copyable, Movable):
    var to_q: Optional[LoraAdapter]
    var to_k: Optional[LoraAdapter]
    var to_v: Optional[LoraAdapter]
    var to_out: Optional[LoraAdapter]
    var gate_proj: Optional[LoraAdapter]
    var up_proj: Optional[LoraAdapter]
    var linear_fc2: Optional[LoraAdapter]

    def __init__(
        out self,
        var to_q: Optional[LoraAdapter], var to_k: Optional[LoraAdapter],
        var to_v: Optional[LoraAdapter], var to_out: Optional[LoraAdapter],
        var gate_proj: Optional[LoraAdapter], var up_proj: Optional[LoraAdapter],
        var linear_fc2: Optional[LoraAdapter],
    ):
        self.to_q = to_q^
        self.to_k = to_k^
        self.to_v = to_v^
        self.to_out = to_out^
        self.gate_proj = gate_proj^
        self.up_proj = up_proj^
        self.linear_fc2 = linear_fc2^


struct ErnieBlockLoraDevice(Copyable, Movable):
    var to_q: ErnieLoraAdapterDevice
    var to_k: ErnieLoraAdapterDevice
    var to_v: ErnieLoraAdapterDevice
    var to_out: ErnieLoraAdapterDevice
    var gate_proj: ErnieLoraAdapterDevice
    var up_proj: ErnieLoraAdapterDevice
    var linear_fc2: ErnieLoraAdapterDevice

    def __init__(
        out self,
        var to_q: ErnieLoraAdapterDevice, var to_k: ErnieLoraAdapterDevice,
        var to_v: ErnieLoraAdapterDevice, var to_out: ErnieLoraAdapterDevice,
        var gate_proj: ErnieLoraAdapterDevice, var up_proj: ErnieLoraAdapterDevice,
        var linear_fc2: ErnieLoraAdapterDevice,
    ):
        self.to_q = to_q^
        self.to_k = to_k^
        self.to_v = to_v^
        self.to_out = to_out^
        self.gate_proj = gate_proj^
        self.up_proj = up_proj^
        self.linear_fc2 = linear_fc2^


# ── per-block LoRA grads (parallel to the 7 slots) ───────────────────────────
# d_a/d_b per present adapter; empty lists for absent slots. The block backward
# fills only the present slots (the others stay empty and AdamW skips them).
struct ErnieBlockLoraGrads(Copyable, Movable):
    var d_a: List[List[Float32]]   # ERNIE_SLOTS entries (empty if slot absent)
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


# ── LoRA-aware block forward ──────────────────────────────────────────────────
# Mirrors ernie_block_forward EXACTLY (models/ernie/block.mojo), adding the LoRA
# contribution to each of the 7 trained projection outputs BEFORE the downstream
# op consumes it. When all 7 adapters are absent this reduces bit-for-bit to the
# base forward (each ernie_lora_apply returns base_y unchanged). The `saved`
# activations are the LoRA-MODIFIED ones, so backward recompute regenerates them
# identically (same checkpoint contract as Klein).
def ernie_block_lora_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: ErnieBlockWeights, mv: ErnieModVecs, lora: ErnieBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var x_t = _t(x, [S, D], ctx)

    # --- self-attention sub-block ---
    var sa_norm = rms_norm(x_t, w.sa_norm[], eps, ctx)
    var sa_in = modulate(
        sa_norm, _t(mv.scale_msa.copy(), [D], ctx), _t(mv.shift_msa.copy(), [D], ctx), ctx
    )
    var sa_in_h = sa_in.to_host(ctx)                            # [S,D] (LoRA input)

    var no_bias = Optional[Tensor](None)
    var q_base = linear(sa_in, w.wq[], no_bias^, ctx).to_host(ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear(sa_in, w.wk[], no_bias_k^, ctx).to_host(ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear(sa_in, w.wv[], no_bias_v^, ctx).to_host(ctx)

    var q_h = ernie_lora_apply(q_base, sa_in_h, lora.to_q, S, ctx)
    var k_h = ernie_lora_apply(k_base, sa_in_h, lora.to_k, S, ctx)
    var v_h = ernie_lora_apply(v_base, sa_in_h, lora.to_v, S, ctx)

    var q_pre = reshape_owned(_t(q_h, [S, D], ctx)^, [1, S, H, Dh])
    var k_pre = reshape_owned(_t(k_h, [S, D], ctx)^, [1, S, H, Dh])
    var v = reshape_owned(_t(v_h, [S, D], ctx)^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_halfsplit_full(q_rms, cos, sin, ctx)
    var k_rope = rope_halfsplit_full(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])
    var att_flat_h = att_flat.to_host(ctx)                      # [S,D] (LoRA input for wo)

    var no_bias_o = Optional[Tensor](None)
    var att_out_base = linear(att_flat, w.wo[], no_bias_o^, ctx).to_host(ctx)
    var att_out_h = ernie_lora_apply(att_out_base, att_flat_h, lora.to_out, S, ctx)
    var att_out = _t(att_out_h, [S, D], ctx)

    var h = residual_gate(x_t, _t(mv.gate_msa.copy(), [D], ctx), att_out, ctx)

    # --- MLP sub-block (GELU-gated) ---
    var mlp_norm = rms_norm(h, w.mlp_norm[], eps, ctx)
    var mlp_in = modulate(
        mlp_norm, _t(mv.scale_mlp.copy(), [D], ctx), _t(mv.shift_mlp.copy(), [D], ctx), ctx
    )
    var mlp_in_h = mlp_in.to_host(ctx)                          # [S,D] (LoRA input)

    var no_bias_g = Optional[Tensor](None)
    var gate_base = linear(mlp_in, w.wgate[], no_bias_g^, ctx).to_host(ctx)
    var no_bias_u = Optional[Tensor](None)
    var up_base = linear(mlp_in, w.wup[], no_bias_u^, ctx).to_host(ctx)
    var gate_pre_h = ernie_lora_apply(gate_base, mlp_in_h, lora.gate_proj, S, ctx)
    var up_h = ernie_lora_apply(up_base, mlp_in_h, lora.up_proj, S, ctx)
    var gate_pre = _t(gate_pre_h, [S, F], ctx)
    var up = _t(up_h, [S, F], ctx)

    var gelu_gate = gelu_exact(gate_pre, ctx)
    var activated = mul(gelu_gate, up, ctx)
    var activated_h = activated.to_host(ctx)                    # [S,F] (LoRA input for wdown)

    var no_bias_d = Optional[Tensor](None)
    var mlp_out_base = linear(activated, w.wdown[], no_bias_d^, ctx).to_host(ctx)
    var mlp_out_h = ernie_lora_apply(mlp_out_base, activated_h, lora.linear_fc2, S, ctx)
    var mlp_out = _t(mlp_out_h, [S, D], ctx)

    var result = residual_gate(
        h, _t(mv.gate_mlp.copy(), [D], ctx), mlp_out, ctx
    ).to_host(ctx)

    var saved = ErnieBlockSaved(
        TArc(x_t^), TArc(sa_norm^), TArc(sa_in^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(h^), TArc(mlp_norm^), TArc(mlp_in^),
        TArc(gate_pre^), TArc(gelu_gate^), TArc(up^), TArc(activated^),
    )
    return ErnieBlockForward(result^, saved^)


struct ErnieBlockForwardLoraTensor(Movable):
    var out: TArc
    var saved: ErnieBlockSaved

    def __init__(out self, var out: TArc, var saved: ErnieBlockSaved):
        self.out = out^
        self.saved = saved^


def ernie_block_lora_forward_device_tensor[
    H: Int, Dh: Int, S: Int
](
    x_arc: TArc,
    w: ErnieBlockWeights, mv: ErnieModVecs, lora: ErnieBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    text_pad_mask: Optional[TArc] = None,  # [1,H,S,S] additive, q dtype; None=full attn
) raises -> ErnieBlockForwardLoraTensor:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    # --- self-attention sub-block ---
    var sa_norm = rms_norm(x_arc[], w.sa_norm[], eps, ctx)
    var sa_in = modulate(
        sa_norm, _t(mv.scale_msa.copy(), [D], ctx), _t(mv.shift_msa.copy(), [D], ctx), ctx
    )

    var no_bias = Optional[Tensor](None)
    var q_base = linear(sa_in, w.wq[], no_bias^, ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear(sa_in, w.wk[], no_bias_k^, ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear(sa_in, w.wv[], no_bias_v^, ctx)

    var q_f = ernie_lora_apply_device(q_base^, sa_in, lora.to_q, S, ctx)
    var k_f = ernie_lora_apply_device(k_base^, sa_in, lora.to_k, S, ctx)
    var v_f = ernie_lora_apply_device(v_base^, sa_in, lora.to_v, S, ctx)

    var q_pre = reshape_owned(q_f^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_f^, [1, S, H, Dh])
    var v = reshape_owned(v_f^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_halfsplit_full(q_rms, cos, sin, ctx)
    var k_rope = rope_halfsplit_full(k_rms, cos, sin, ctx)

    # OneTrainer masks padded TEXT keys (transformer_ernie_image.py:393-421); the
    # additive [1,H,S,S] mask makes those columns contribute ~0 to softmax. When
    # absent (None) this is bit-identical to the previous unmasked path.
    var att: Tensor
    if text_pad_mask:
        att = sdpa[1, S, H, Dh](q_rope, k_rope, v, text_pad_mask.value()[], scale, ctx)
    else:
        att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_out_base = linear(att_flat, w.wo[], no_bias_o^, ctx)
    var att_out = ernie_lora_apply_device(att_out_base^, att_flat, lora.to_out, S, ctx)

    var h = residual_gate(x_arc[], _t(mv.gate_msa.copy(), [D], ctx), att_out, ctx)

    # --- MLP sub-block (GELU-gated) ---
    var mlp_norm = rms_norm(h, w.mlp_norm[], eps, ctx)
    var mlp_in = modulate(
        mlp_norm, _t(mv.scale_mlp.copy(), [D], ctx), _t(mv.shift_mlp.copy(), [D], ctx), ctx
    )

    var no_bias_g = Optional[Tensor](None)
    var gate_base = linear(mlp_in, w.wgate[], no_bias_g^, ctx)
    var no_bias_u = Optional[Tensor](None)
    var up_base = linear(mlp_in, w.wup[], no_bias_u^, ctx)
    var gate_pre = ernie_lora_apply_device(gate_base^, mlp_in, lora.gate_proj, S, ctx)
    var up = ernie_lora_apply_device(up_base^, mlp_in, lora.up_proj, S, ctx)

    var gelu_gate = gelu_exact(gate_pre, ctx)
    var activated = mul(gelu_gate, up, ctx)

    var no_bias_d = Optional[Tensor](None)
    var mlp_out_base = linear(activated, w.wdown[], no_bias_d^, ctx)
    var mlp_out = ernie_lora_apply_device(mlp_out_base^, activated, lora.linear_fc2, S, ctx)

    var result = residual_gate(
        h, _t(mv.gate_mlp.copy(), [D], ctx), mlp_out, ctx
    )

    var saved = ErnieBlockSaved(
        x_arc.copy(), TArc(sa_norm^), TArc(sa_in^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(h^), TArc(mlp_norm^), TArc(mlp_in^),
        TArc(gate_pre^), TArc(gelu_gate^), TArc(up^), TArc(activated^),
    )
    return ErnieBlockForwardLoraTensor(TArc(result^), saved^)


# ── LoRA-aware block backward ─────────────────────────────────────────────────
# Mirrors ernie_block_backward EXACTLY. At each of the 7 trained projections the
# d_y flowing INTO that projection (the same grad the base linear_backward sees)
# is fed to ernie_lora_bwd to produce d_A/d_B + the LoRA d_x contribution, which
# is SUMMED into the projection-input grad (the base d_x from the frozen W still
# flows — LoRA training keeps base weights frozen but their grad path is alive).
# Returns the base ErnieBlockGrads (base weight grads discarded by the LoRA
# optimizer, but d_x / mod-vec grads load-bearing) PLUS the 7-slot LoRA grads.
struct ErnieBlockLoraBackward(Movable):
    var base: ErnieBlockGrads
    var lora: ErnieBlockLoraGrads

    def __init__(out self, var base: ErnieBlockGrads, var lora: ErnieBlockLoraGrads):
        self.base = base^
        self.lora = lora^


# proj-backward result: d_x [M,in] (base + LoRA summed) and d_w (base, discarded).
struct _ProjGrads(Movable):
    var d_x: Tensor

    def __init__(out self, var d_x: Tensor):
        self.d_x = d_x^


# helper: run base linear_backward d_x then add the LoRA branch's d_x (if present),
# collecting the LoRA d_a/d_b into the slot lists. Returns the SUMMED d_x [M,in].
def _proj_bwd_with_lora(
    d_y: Tensor, x_in: Tensor, w: Tensor, x_in_h: List[Float32],
    lo: Optional[LoraAdapter], slot: Int,
    M: Int, in_f: Int, out_f: Int,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> _ProjGrads:
    # Base weights are frozen in LoRA training. Only d_x is load-bearing; d_W/d_b
    # would be computed, read back, then discarded by the stack optimizer.
    var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    if lo:
        var d_y_h = d_y.to_host(ctx)
        var lg = ernie_lora_bwd(d_y_h, x_in_h, lo.value(), M, ctx)
        d_a_slots[slot] = lg.d_a.copy()
        d_b_slots[slot] = lg.d_b.copy()
        # SUM the LoRA d_x into the base d_x (frozen W base path + LoRA path).
        var base_dx_h = base_dx.to_host(ctx)
        var summed = _add_lists(base_dx_h, lg.d_x)
        return _ProjGrads(_t(summed, [M, in_f], ctx))
    return _ProjGrads(base_dx^)


def ernie_block_lora_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: ErnieBlockWeights, mv: ErnieModVecs, lora: ErnieBlockLora,
    saved: ErnieBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieBlockLoraBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var d_out_t = _t(d_out, [S, D], ctx)

    var d_a_slots = List[List[Float32]]()
    var d_b_slots = List[List[Float32]]()
    for _ in range(ERNIE_SLOTS):
        d_a_slots.append(List[Float32]())
        d_b_slots.append(List[Float32]())

    # out = residual_gate(h, gate_mlp, mlp_out); recompute mlp_out (WITH LoRA so
    # the gate_residual_backward y matches the forward mlp_out exactly).
    var nb = Optional[Tensor](None)
    var mlp_out_base = linear(saved.activated[], w.wdown[], nb^, ctx).to_host(ctx)
    var activated_h = saved.activated[].to_host(ctx)
    var mlp_out_h = ernie_lora_apply(mlp_out_base, activated_h, lora.linear_fc2, S, ctx)
    var mlp_out_y = _t(mlp_out_h, [S, D], ctx)
    var grg2 = gate_residual_backward(
        d_out_t, saved.h[], _t(mv.gate_mlp.copy(), [D], ctx), mlp_out_y, ctx
    )
    var d_gate_mlp = grg2.d_g.to_host(ctx)

    # mlp_out = linear(activated, wdown) [+ LoRA(linear_fc2)]  W [D, F]
    var lb_down = _proj_bwd_with_lora(
        grg2.d_y, saved.activated[], w.wdown[], activated_h,
        lora.linear_fc2, SLOT_DOWN, S, F, D, d_a_slots, d_b_slots, ctx,
    )

    # activated = gelu_gate * up
    var d_gelu_gate = mul(lb_down.d_x, saved.up[], ctx)
    var d_up = mul(lb_down.d_x, saved.gelu_gate[], ctx)
    var d_gate_pre = gelu_exact_backward(d_gelu_gate, saved.gate_pre[], ctx)

    # gate_pre = linear(mlp_in, wgate)[+LoRA]; up = linear(mlp_in, wup)[+LoRA]  W [F,D]
    var mlp_in_h = saved.mlp_in[].to_host(ctx)
    var lb_gate = _proj_bwd_with_lora(
        d_gate_pre, saved.mlp_in[], w.wgate[], mlp_in_h,
        lora.gate_proj, SLOT_GATE, S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var lb_up = _proj_bwd_with_lora(
        d_up, saved.mlp_in[], w.wup[], mlp_in_h,
        lora.up_proj, SLOT_UP, S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_mlp_in = add(lb_gate.d_x, lb_up.d_x, ctx)

    var mb_mlp = modulate_backward(d_mlp_in, saved.mlp_norm[], _t(mv.scale_mlp.copy(), [D], ctx), ctx)
    var d_scale_mlp = mb_mlp.d_scale.to_host(ctx)
    var d_shift_mlp = mb_mlp.d_shift.to_host(ctx)
    var rb_mlp = rms_norm_backward(mb_mlp.d_x, saved.h[], w.mlp_norm[], eps, ctx)
    var d_mlp_norm = rb_mlp.d_g.to_host(ctx)
    var d_h = add(grg2.d_x, rb_mlp.d_x, ctx)

    # --- self-attention sub-block backward ---
    var nb2 = Optional[Tensor](None)
    var att_out_base = linear(saved.att_flat[], w.wo[], nb2^, ctx).to_host(ctx)
    var att_flat_h = saved.att_flat[].to_host(ctx)
    var att_out_h = ernie_lora_apply(att_out_base, att_flat_h, lora.to_out, S, ctx)
    var att_out_y = _t(att_out_h, [S, D], ctx)
    var grg1 = gate_residual_backward(
        d_h, saved.x[], _t(mv.gate_msa.copy(), [D], ctx), att_out_y, ctx
    )
    var d_gate_msa = grg1.d_g.to_host(ctx)

    # att_out = linear(att_flat, wo)[+LoRA(to_out)]  W [D, D]
    var lb_o = _proj_bwd_with_lora(
        grg1.d_y, saved.att_flat[], w.wo[], att_flat_h,
        lora.to_out, SLOT_O, S, D, D, d_a_slots, d_b_slots, ctx,
    )

    reshape_in_place(lb_o.d_x, [1, S, H, Dh])
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )
    var d_q_rms = rope_halfsplit_full_backward(sb.d_q, cos, sin, ctx)
    var d_k_rms = rope_halfsplit_full_backward(sb.d_k, cos, sin, ctx)

    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # q/k/v = linear(sa_in, w{q,k,v})[+LoRA]  W [D, D]; sa_in feeds all three.
    var sa_in_h = saved.sa_in[].to_host(ctx)
    var lb_q = _proj_bwd_with_lora(
        rb_q.d_x, saved.sa_in[], w.wq[], sa_in_h,
        lora.to_q, SLOT_Q, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var lb_k = _proj_bwd_with_lora(
        rb_k.d_x, saved.sa_in[], w.wk[], sa_in_h,
        lora.to_k, SLOT_K, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var lb_v = _proj_bwd_with_lora(
        sb.d_v, saved.sa_in[], w.wv[], sa_in_h,
        lora.to_v, SLOT_V, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_sa_in = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)

    var mb_sa = modulate_backward(d_sa_in, saved.sa_norm[], _t(mv.scale_msa.copy(), [D], ctx), ctx)
    var d_scale_msa = mb_sa.d_scale.to_host(ctx)
    var d_shift_msa = mb_sa.d_shift.to_host(ctx)
    var rb_sa = rms_norm_backward(mb_sa.d_x, saved.x[], w.sa_norm[], eps, ctx)
    var d_sa_norm = rb_sa.d_g.to_host(ctx)

    var d_x_res = grg1.d_x.to_host(ctx)
    var d_x_norm = rb_sa.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    var base = ErnieBlockGrads(
        d_x^,
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
        d_q_norm^, d_k_norm^,
        d_sa_norm^, d_mlp_norm^,
        List[Float32](), List[Float32](), List[Float32](),
        d_shift_msa^, d_scale_msa^, d_gate_msa^,
        d_shift_mlp^, d_scale_mlp^, d_gate_mlp^,
    )
    return ErnieBlockLoraBackward(base^, ErnieBlockLoraGrads(d_a_slots^, d_b_slots^))


struct _ProjTensorGrads(Movable):
    var d_x: Tensor
    var d_a: TArc
    var d_b: TArc

    def __init__(out self, var d_x: Tensor, var d_a: TArc, var d_b: TArc):
        self.d_x = d_x^
        self.d_a = d_a^
        self.d_b = d_b^


def _proj_bwd_with_lora_device_tensors(
    d_y: Tensor, x_in: Tensor, w: Tensor, lo: ErnieLoraAdapterDevice,
    M: Int, in_f: Int, out_f: Int,
    ctx: DeviceContext,
) raises -> _ProjTensorGrads:
    var lg = ernie_lora_bwd_device_tensors(d_y, x_in, lo, M, ctx)
    var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    var summed = add(base_dx^, lg.d_x[], ctx)
    return _ProjTensorGrads(summed^, lg.d_a.copy(), lg.d_b.copy())


def _pack_mod6(
    d_shift_msa: List[Float32], d_scale_msa: List[Float32], d_gate_msa: List[Float32],
    d_shift_mlp: List[Float32], d_scale_mlp: List[Float32], d_gate_mlp: List[Float32],
) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(d_shift_msa)):
        out.append(d_shift_msa[i])
    for i in range(len(d_scale_msa)):
        out.append(d_scale_msa[i])
    for i in range(len(d_gate_msa)):
        out.append(d_gate_msa[i])
    for i in range(len(d_shift_mlp)):
        out.append(d_shift_mlp[i])
    for i in range(len(d_scale_mlp)):
        out.append(d_scale_mlp[i])
    for i in range(len(d_gate_mlp)):
        out.append(d_gate_mlp[i])
    return out^


struct ErnieBlockLoraTensorBackward(Movable):
    var d_x: TArc
    var d_a: List[TArc]
    var d_b: List[TArc]
    var d_shared_mod: List[Float32]

    def __init__(
        out self, var d_x: TArc, var d_a: List[TArc], var d_b: List[TArc],
        var d_shared_mod: List[Float32],
    ):
        self.d_x = d_x^
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_shared_mod = d_shared_mod^


def ernie_block_lora_backward_device_tensors[
    H: Int, Dh: Int, S: Int
](
    d_out: Tensor,
    w: ErnieBlockWeights, mv: ErnieModVecs, lora: ErnieBlockLoraDevice,
    saved: ErnieBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    text_pad_mask_f32: Optional[TArc] = None,  # [H*S,S] F32; None=unmasked SDPA bwd
) raises -> ErnieBlockLoraTensorBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    # out = residual_gate(h, gate_mlp, mlp_out); recompute mlp_out with LoRA.
    var nb = Optional[Tensor](None)
    var mlp_out_base = linear(saved.activated[], w.wdown[], nb^, ctx)
    var mlp_out_y = ernie_lora_apply_device(
        mlp_out_base^, saved.activated[], lora.linear_fc2, S, ctx
    )
    var grg2 = gate_residual_backward(
        d_out, saved.h[], _t(mv.gate_mlp.copy(), [D], ctx), mlp_out_y, ctx
    )
    var d_gate_mlp = grg2.d_g.to_host(ctx)

    var pg_down = _proj_bwd_with_lora_device_tensors(
        grg2.d_y, saved.activated[], w.wdown[], lora.linear_fc2, S, F, D, ctx,
    )

    var d_gelu_gate = mul(pg_down.d_x, saved.up[], ctx)
    var d_up = mul(pg_down.d_x, saved.gelu_gate[], ctx)
    var d_gate_pre = gelu_exact_backward(d_gelu_gate, saved.gate_pre[], ctx)

    var pg_gate = _proj_bwd_with_lora_device_tensors(
        d_gate_pre, saved.mlp_in[], w.wgate[], lora.gate_proj, S, D, F, ctx,
    )
    var pg_up = _proj_bwd_with_lora_device_tensors(
        d_up, saved.mlp_in[], w.wup[], lora.up_proj, S, D, F, ctx,
    )
    var d_mlp_in = add(pg_gate.d_x, pg_up.d_x, ctx)

    var mb_mlp = modulate_backward(
        d_mlp_in, saved.mlp_norm[], _t(mv.scale_mlp.copy(), [D], ctx), ctx
    )
    var d_scale_mlp = mb_mlp.d_scale.to_host(ctx)
    var d_shift_mlp = mb_mlp.d_shift.to_host(ctx)
    var d_mlp_norm_dx = rms_norm_backward_dx(mb_mlp.d_x, saved.h[], w.mlp_norm[], eps, ctx)
    var d_h = add(grg2.d_x, d_mlp_norm_dx, ctx)

    # --- self-attention sub-block backward ---
    var nb2 = Optional[Tensor](None)
    var att_out_base = linear(saved.att_flat[], w.wo[], nb2^, ctx)
    var att_out_y = ernie_lora_apply_device(
        att_out_base^, saved.att_flat[], lora.to_out, S, ctx
    )
    var grg1 = gate_residual_backward(
        d_h, saved.x[], _t(mv.gate_msa.copy(), [D], ctx), att_out_y, ctx
    )
    var d_gate_msa = grg1.d_g.to_host(ctx)

    var pg_o = _proj_bwd_with_lora_device_tensors(
        grg1.d_y, saved.att_flat[], w.wo[], lora.to_out, S, D, D, ctx,
    )

    reshape_in_place(pg_o.d_x, [1, S, H, Dh])
    # Same text-pad mask the forward applied (additive, after scale, before
    # softmax — sdpa_backward_masked recomputes the forward with it). None=plain.
    var sb: SdpaGrads
    if text_pad_mask_f32:
        sb = sdpa_backward_masked[1, S, H, Dh](
            saved.q_rope[], saved.k_rope[], saved.v[],
            text_pad_mask_f32.value()[], pg_o.d_x, scale, ctx
        )
    else:
        sb = sdpa_backward[1, S, H, Dh](
            saved.q_rope[], saved.k_rope[], saved.v[], pg_o.d_x, scale, ctx
        )
    var d_q_rms = rope_halfsplit_full_backward(sb.d_q, cos, sin, ctx)
    var d_k_rms = rope_halfsplit_full_backward(sb.d_k, cos, sin, ctx)

    var d_q_pre = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)

    reshape_in_place(d_q_pre, [S, D])
    reshape_in_place(d_k_pre, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    var pg_q = _proj_bwd_with_lora_device_tensors(
        d_q_pre, saved.sa_in[], w.wq[], lora.to_q, S, D, D, ctx,
    )
    var pg_k = _proj_bwd_with_lora_device_tensors(
        d_k_pre, saved.sa_in[], w.wk[], lora.to_k, S, D, D, ctx,
    )
    var pg_v = _proj_bwd_with_lora_device_tensors(
        sb.d_v, saved.sa_in[], w.wv[], lora.to_v, S, D, D, ctx,
    )
    var d_sa_in = add(add(pg_q.d_x, pg_k.d_x, ctx), pg_v.d_x, ctx)

    var mb_sa = modulate_backward(
        d_sa_in, saved.sa_norm[], _t(mv.scale_msa.copy(), [D], ctx), ctx
    )
    var d_scale_msa = mb_sa.d_scale.to_host(ctx)
    var d_shift_msa = mb_sa.d_shift.to_host(ctx)
    var d_sa_norm_dx = rms_norm_backward_dx(mb_sa.d_x, saved.x[], w.sa_norm[], eps, ctx)
    var d_x = add(grg1.d_x, d_sa_norm_dx, ctx)

    var d_a = List[TArc]()
    var d_b = List[TArc]()
    d_a.append(pg_q.d_a.copy()); d_b.append(pg_q.d_b.copy())
    d_a.append(pg_k.d_a.copy()); d_b.append(pg_k.d_b.copy())
    d_a.append(pg_v.d_a.copy()); d_b.append(pg_v.d_b.copy())
    d_a.append(pg_o.d_a.copy()); d_b.append(pg_o.d_b.copy())
    d_a.append(pg_gate.d_a.copy()); d_b.append(pg_gate.d_b.copy())
    d_a.append(pg_up.d_a.copy()); d_b.append(pg_up.d_b.copy())
    d_a.append(pg_down.d_a.copy()); d_b.append(pg_down.d_b.copy())

    var shared = _pack_mod6(
        d_shift_msa, d_scale_msa, d_gate_msa,
        d_shift_mlp, d_scale_mlp, d_gate_mlp,
    )
    return ErnieBlockLoraTensorBackward(TArc(d_x^), d_a^, d_b^, shared^)
