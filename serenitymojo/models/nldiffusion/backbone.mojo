# models/nldiffusion/backbone.mojo — NVIDIA NL-Diffusion-Image backbone.
#
# Pure-Mojo + MAX (GPU-only, INFERENCE) port of the Ministral3 decoder layer
# used by NVIDIA NL-Diffusion-Image (`/mnt/disk1/models/NL-Diffusion-Image`,
# reference `modeling_ministral.py`). See
#   serenitymojo/models/nldiffusion/PORT_SPEC.md
# for the exact layer math, weight names, and chunk gates.
#
# Architecturally identical to the Mistral-family GQA decoder scaffold
#   serenitymojo/models/text_encoder/mistral3b_encoder.mojo
# with these deltas (PORT_SPEC §Backbone / §SCAFFOLD):
#   1. attention is BIDIRECTIONAL (no causal mask) -> `sdpa_nomask`.
#   2. weight prefix `encoder.layers.{N}.*`.
#   3. hidden 4096, 34 layers, 32 heads, 8 kv-heads (GQA groups 4), head_dim 128,
#      intermediate 14336, rms_eps 1e-5, NO biases.
#   4. llama4 attention-scale == IDENTITY (1.0) for seq < 16384 -> skipped.
#   5. RoPE = YaRN; cos/sin are computed once at model level and shared to every
#      layer, so this per-layer forward takes them as inputs.
#
# CHUNK A scope: one decoder layer + its parity probe. The full 34-layer stack
# and the image head are later chunks.

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu
from serenitymojo.models.text_encoder.qwen3_encoder import (
    _add,
    _repeat_kv,
    _reshape,
)


@fieldwise_init
struct NLDiffConfig(Copyable, Movable, ImplicitlyCopyable):
    """NL-Diffusion-Image backbone config (from config.json). NO biases."""

    var hidden_size: Int
    var num_layers: Int
    var num_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var intermediate_size: Int
    var rms_norm_eps: Float32

    @staticmethod
    def default() -> NLDiffConfig:
        return NLDiffConfig(
            4096,  # hidden_size
            34,  # num_hidden_layers
            32,  # num_attention_heads
            8,  # num_key_value_heads (GQA groups = 4)
            128,  # head_dim
            14336,  # intermediate_size
            Float32(1.0e-5),  # rms_norm_eps
        )


struct NLDiffLayerWeights(Movable):
    """The 9 bf16 weights of one `encoder.layers.{N}` decoder layer (no biases)."""

    var input_layernorm: ArcPointer[Tensor]
    var post_attention_layernorm: ArcPointer[Tensor]
    var q_proj: ArcPointer[Tensor]
    var k_proj: ArcPointer[Tensor]
    var v_proj: ArcPointer[Tensor]
    var o_proj: ArcPointer[Tensor]
    var gate_proj: ArcPointer[Tensor]
    var up_proj: ArcPointer[Tensor]
    var down_proj: ArcPointer[Tensor]

    def __init__(
        out self,
        var input_layernorm: ArcPointer[Tensor],
        var post_attention_layernorm: ArcPointer[Tensor],
        var q_proj: ArcPointer[Tensor],
        var k_proj: ArcPointer[Tensor],
        var v_proj: ArcPointer[Tensor],
        var o_proj: ArcPointer[Tensor],
        var gate_proj: ArcPointer[Tensor],
        var up_proj: ArcPointer[Tensor],
        var down_proj: ArcPointer[Tensor],
    ):
        self.input_layernorm = input_layernorm^
        self.post_attention_layernorm = post_attention_layernorm^
        self.q_proj = q_proj^
        self.k_proj = k_proj^
        self.v_proj = v_proj^
        self.o_proj = o_proj^
        self.gate_proj = gate_proj^
        self.up_proj = up_proj^
        self.down_proj = down_proj^


def nldiff_decoder_layer[
    S: Int
](
    x: Tensor,
    w: NLDiffLayerWeights,
    cfg: NLDiffConfig,
    cos_q: Tensor,
    sin_q: Tensor,
    cos_k: Tensor,
    sin_k: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """One Ministral3 decoder layer (pre-norm, residual, GQA attention, SwiGLU).

    Reproduces `modeling_ministral.py` L342-388 EXACTLY (bidirectional):
        residual = x
        h = rms_norm(x, input_layernorm, eps)
        h = self_attn(h)                       # q/k/v -> rope -> repeat_kv ->
        x = residual + h                       #   sdpa NO-MASK scale=Dh^-0.5 -> o
        residual = x
        h = rms_norm(x, post_attention_layernorm, eps)
        h = down(silu(gate(h)) * up(h))        # SwiGLU MLP
        x = residual + h

    `x`, weights, cos/sin, output all live on-GPU. `x: [1, S, hidden]`. cos/sin
    are the shared YaRN tables, half-split layout: `cos_q/sin_q: [S*num_heads, Dh/2]`,
    `cos_k/sin_k: [S*num_kv_heads, Dh/2]` (rows = leading dims of q/k after the
    [1,S,H,Dh] reshape). `S` is a comptime param so `sdpa_nomask` shapes are
    constant. head/kv/head_dim are fixed at (32, 8, 128) here to match the
    comptime sdpa dispatch; `cfg` must agree.
    """
    var n_heads = cfg.num_heads  # 32
    var n_kv = cfg.num_kv_heads  # 8
    var dh = cfg.head_dim  # 128
    # BIDIRECTIONAL sdpa scaling = 128 ** -0.5 == 1/sqrt(Dh) (PORT_SPEC L49).
    var scale = Float32(1.0) / sqrt(Float32(dh))
    var seq = x.shape()[1]

    # ── attention block ──────────────────────────────────────────────────────
    var normed = rms_norm(x, w.input_layernorm[], cfg.rms_norm_eps, ctx)

    var q = linear(normed, w.q_proj[], None, ctx)  # [1,S,32*128]
    var k = linear(normed, w.k_proj[], None, ctx)  # [1,S,8*128]
    var v = linear(normed, w.v_proj[], None, ctx)  # [1,S,8*128]

    q = _reshape(q, [1, seq, n_heads, dh], ctx)
    k = _reshape(k, [1, seq, n_kv, dh], ctx)
    v = _reshape(v, [1, seq, n_kv, dh], ctx)

    # RoPE (HF rotate_half == rope_halfsplit); llama4 scale is identity (skip).
    q = rope_halfsplit(q, cos_q, sin_q, ctx)
    k = rope_halfsplit(k, cos_k, sin_k, ctx)

    var k_rep = _repeat_kv(k^, n_heads, n_kv, ctx)  # 8 -> 32
    var v_rep = _repeat_kv(v^, n_heads, n_kv, ctx)

    var attn = sdpa_nomask[1, S, 32, 128](q, k_rep, v_rep, scale, ctx)
    attn = _reshape(attn, [1, seq, n_heads * dh], ctx)
    var attn_out = linear(attn, w.o_proj[], None, ctx)

    var x1 = _add(x, attn_out, ctx)  # residual + attn

    # ── SwiGLU MLP block ─────────────────────────────────────────────────────
    var normed2 = rms_norm(
        x1, w.post_attention_layernorm[], cfg.rms_norm_eps, ctx
    )
    var gate = linear(normed2, w.gate_proj[], None, ctx)
    var up = linear(normed2, w.up_proj[], None, ctx)
    var act = swiglu(gate, up, ctx)  # silu(gate) * up
    var mlp_out = linear(act, w.down_proj[], None, ctx)

    return _add(x1, mlp_out, ctx)  # residual + mlp
