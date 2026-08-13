# serenitymojo/llm/vl_decode.mojo — KV-cached incremental decode step that takes
# a PRE-COMPUTED embedding row + caller-supplied rope rows: the building block
# for the pure-Mojo Qwen3-VL image CAPTIONER.
#
# Mirrors llm/decoder.decode_step exactly (same ops, same cache discipline,
# sqa_device_par attention) with three deltas the vision-fused stream needs:
#   1. input is an embedding row [1,1,H] (vision pooler rows are NOT tokens;
#      text rows are embedded by the caller via enc._embed) — masked_scatter
#      done caller-side, one row at a time.
#   2. rope cos/sin rows come from the CALLER (3D interleaved M-RoPE for vision
#      rows — lingbot_qwen3vl_fuse._build_mrope_tables; for text rows the equal-
#      axis M-RoPE collapses to plain rope, same tables).
#   3. optional deepstack rows ds0/ds1/ds2 are ADDED after decoder layers 0/1/2
#      (per-position math — identical to fuse_core's batched _add_rows).
# Logits use lm_head when the checkpoint ships one, else the TIED embedding
# table (Qwen3-VL-4B: tie_word_embeddings=true, no lm_head key).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.text_encoder.qwen3_encoder import (
    Qwen3Encoder, _reshape, _add,
)
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.tensor_algebra import concat, zeros_device
from serenitymojo.llm.decoder import KVCache
from serenitymojo.llm.sqa import sqa_device_par


def vl_logits(enc: Qwen3Encoder, hidden: Tensor, ctx: DeviceContext) raises -> Tensor:
    """model.norm + LM head on a [1,1,H] hidden row -> [1,1,vocab]. Uses
    lm_head.weight when present, else the tied embedding table (same [vocab,H]
    orientation, so the same `linear` call)."""
    var normed = rms_norm(
        hidden, enc._w(String("model.norm.weight")), enc.config.rms_norm_eps, ctx
    )
    if enc._has(String("lm_head.weight")):
        return linear(normed, enc._w(String("lm_head.weight")), None, ctx)
    return linear(normed, enc._w(String("model.embed_tokens.weight")), None, ctx)


def vl_decode_step_embed(
    enc: Qwen3Encoder,
    mut cache: KVCache,
    var hidden0: Tensor,        # [1,1,H] embedding row (consumed)
    cos_q: Tensor, sin_q: Tensor,   # [H_heads*half] rope row for q
    cos_k: Tensor, sin_k: Tensor,   # [H_kv*half]   rope row for k
    ds0: Optional[Tensor], ds1: Optional[Tensor], ds2: Optional[Tensor],
    ctx: DeviceContext,
    want_logits: Bool = True,
) raises -> Tensor:
    """One cached decode step from an embedding row. Returns [1,1,vocab] logits
    (or a [1,1,1] dummy when want_logits=False — prompt priming, logits
    discarded). Deepstack rows, when provided, are added after layers 0/1/2
    (vision rows of the fused stream only)."""
    var cfg = enc.config
    var H = cfg.num_heads
    var H_kv = cfg.num_kv_heads
    var dh = cfg.head_dim
    var eps = cfg.rms_norm_eps

    var hidden = hidden0^
    for layer in range(cfg.num_layers):
        var p = String("model.layers.") + String(layer)
        var normed = rms_norm(hidden, enc._w(p + ".input_layernorm.weight"), eps, ctx)
        var q = linear(normed, enc._w(p + ".self_attn.q_proj.weight"), None, ctx)
        var k = linear(normed, enc._w(p + ".self_attn.k_proj.weight"), None, ctx)
        var v = linear(normed, enc._w(p + ".self_attn.v_proj.weight"), None, ctx)
        q = _reshape(q, [1, 1, H, dh], ctx)
        k = _reshape(k, [1, 1, H_kv, dh], ctx)
        v = _reshape(v, [1, 1, H_kv, dh], ctx)
        q = rms_norm(q, enc._w(p + ".self_attn.q_norm.weight"), eps, ctx)
        k = rms_norm(k, enc._w(p + ".self_attn.k_norm.weight"), eps, ctx)
        q = rope_halfsplit(q, cos_q, sin_q, ctx)
        k = rope_halfsplit(k, cos_k, sin_k, ctx)

        if not cache.has[layer]:
            cache.k.append(ArcPointer(k^))
            cache.v.append(ArcPointer(v^))
            cache.has[layer] = True
        else:
            var nk = concat(1, ctx, cache.k[layer][], k)
            var nv = concat(1, ctx, cache.v[layer][], v)
            cache.k[layer] = ArcPointer(nk^)
            cache.v[layer] = ArcPointer(nv^)

        var cur_L = cache.k[layer][].shape()[1]
        var attn = sqa_device_par(
            q, cache.k[layer][], cache.v[layer][], H, H_kv, cur_L, dh, ctx
        )
        var attn_out = linear(attn, enc._w(p + ".self_attn.o_proj.weight"), None, ctx)
        var hidden2 = _add(hidden, attn_out, ctx)

        var normed2 = rms_norm(hidden2, enc._w(p + ".post_attention_layernorm.weight"), eps, ctx)
        var gate = linear(normed2, enc._w(p + ".mlp.gate_proj.weight"), None, ctx)
        var up = linear(normed2, enc._w(p + ".mlp.up_proj.weight"), None, ctx)
        var act = swiglu(gate, up, ctx)
        var mlp_out = linear(act, enc._w(p + ".mlp.down_proj.weight"), None, ctx)
        hidden = _add(hidden2, mlp_out, ctx)

        # deepstack injection (fuse_core layers 0/1/2, per-position add)
        if layer == 0 and ds0:
            hidden = _add(hidden, ds0.value(), ctx)
        elif layer == 1 and ds1:
            hidden = _add(hidden, ds1.value(), ctx)
        elif layer == 2 and ds2:
            hidden = _add(hidden, ds2.value(), ctx)

    if not want_logits:
        return zeros_device([1, 1, 1], STDtype.BF16, ctx)
    return vl_logits(enc, hidden, ctx)
