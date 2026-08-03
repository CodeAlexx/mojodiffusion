# serenitymojo/models/text_encoder/minimax_h3_qwen3vl_streamed.mojo
#
# MiniMax-H3's conditioner ("H3-Encoder"): a text-only forward through the
# first 51 decoder layers (0..50 inclusive) of Qwen3-VL-32B-Instruct's text
# tower, returning `hidden_states[50]` PRE-final-norm — the tensor
# `condition_proj` [5376, 5120] in serenitymojo/models/dit/minimax_h3_frontend.mojo
# consumes. t2va has no conditioning image, so the vision tower
# (`vision_config`, ~depth 27) is never loaded or run.
#
# STATUS 2026-08-02: builds and typechecks; NOT numerically gated. H3's own
# text_encoder/ (14 shards, 62.13 GiB) is still downloading — only
# config.json has landed (see minimax_h3_qwen3vl_streamed_probe.mojo, which
# runs for real if the checkpoint is present and otherwise says so plainly).
# Do not treat a clean build here as parity.
#
# CONFIG is the real, landed
# /home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder/config.json
# (`text_config`), not a guess: hidden_size 5120, num_hidden_layers 64 (only
# 0..50 executed), num_attention_heads 64, num_key_value_heads 8 (GQA n_rep
# 8 — verified by REAL RUN, not inferred, in minimax_h3_repeat_kv_probe.mojo;
# ops/text_encoder/qwen3_encoder.mojo's `_repeat_kv` needed no changes, its
# n_rep math is `head // n_rep` at runtime, not hardcoded to any preset),
# head_dim 128, rms_norm_eps 1e-6, rope_theta 5000000.
#
# ROPE: `rope_scaling.mrope_section: [24,20,20]` (24+20+20 = head_dim/2 = 64)
# with `mrope_interleaved: true` — but t2va is TEXT-ONLY (no image/video
# tokens), so every axis position is identical for every token and MRoPE
# degenerates to ordinary single-axis rope regardless of section boundaries
# or interleave order (the 3 axes only diverge when real spatial/temporal
# positions exist). This repo has TWO independent production precedents for
# exactly this call on this architecture family: ideogram_qwen3vl_streamed.mojo
# (Qwen3-VL-8B text tower, parity-gated) and qwen25vl_encoder.mojo ("RoPE
# half-split (text-only mRoPE collapses to 1D)", used in production for
# Qwen-Image). Reusing plain `_build_rope_tables`/`rope_halfsplit` here
# follows the same precedent — UNVERIFIED against this specific 32B
# checkpoint until real weights land.
#
# WEIGHT KEY PREFIX: the ideogram Qwen3-VL-8B checkpoint stores decoder
# layers under `language_model.layers.N.*` (not plain `model.layers.N.*`,
# which is what `Qwen3Encoder._layer` looks up internally — see
# qwen3_encoder.mojo:628 `p = "model.layers." + String(layer_idx)`). Do NOT
# assume H3's 32B checkpoint uses the same prefix: `_detect_layer_prefix`
# below probes the REAL shard index at load time (`has_tensor`) rather than
# hardcoding a guess, per explicit instruction. `_h3_add` then reads from
# whichever SOURCE prefix was detected but always WRITES into the mini
# encoder's dict under the plain `model.layers.N.*` DESTINATION spelling
# `_layer` expects — same dst/src split ideogram_qwen3vl_streamed.mojo uses.
#
# NO FP8: H3's text_encoder/config.json says `"dtype": "bfloat16"` — this is
# a native-bf16 checkpoint, unlike Ideogram's FP8 Qwen3-VL-8B. `_h3_add`
# below is a plain rename+load (`Tensor.from_view`), NOT
# `ideogram_qwen3vl.mojo`'s `load_fp8_dequant` branch — do not reuse that
# path here.
#
# STREAMING STRATEGY: identical to `ideogram_qwen3vl_streamed.mojo`'s
# `encode_ideogram_taps_streamed` — ONE decoder layer's weights resident at a
# time (~0.95 GiB bf16 at this config: dominated by mlp.gate/up/down at
# intermediate_size 25600), embed table (~1.45 GiB) loaded transiently and
# freed after the embed step, `ctx.synchronize()` between layers so the
# previous layer's buffer is actually reclaimed before the next `_h3_load_layer`
# allocates. Peak VRAM stays in the low single-digit GiB on a 24 GiB card —
# same conclusion as the already-shipped 8B/16GB case.
#
# DISK COST (state this plainly, it is real and not hidden by streaming):
# one encode call reads ~51 layers x 0.95 GiB + 1.45 GiB embed table
# = ~50 GiB from disk. This happens ONCE PER PROMPT (conditioning is computed
# once, outside the denoising loop), not once per diffusion step — a bounded
# ~15-30s one-time tax per generation at a few GiB/s NVMe, not a per-step cost.
#
# Only a single prompt is supported for now (no multi-prompt batching like
# ideogram's N-prompt loop) — that can be added later following the same
# per-layer-shared-weights pattern if H3 needs it.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.qwen3_encoder import (
    Qwen3Encoder,
    Qwen3Config,
    _build_rope_tables,
    _build_causal_mask,
)

comptime TArc = ArcPointer[Tensor]

# text_config, MiniMax-H3/FL2VA/text_encoder/config.json (landed 2026-08-02).
comptime H3_HIDDEN = 5120
comptime H3_LAYERS_TOTAL = 64  # released layer count; only 0..H3_EXTRACT_LAYER run
comptime H3_HEADS = 64
comptime H3_KV_HEADS = 8
comptime H3_HEAD_DIM = 128
comptime H3_THETA = Float64(5000000.0)
comptime H3_EPS = Float32(1.0e-6)

# diffusers PR packing.py MINIMAX_H3_TEXT_ENCODER_LAYER: hidden state AFTER
# decoder layer 50 (0-indexed), pre-final-norm. Layers 51..63 and the LM head
# are dead weight for H3 and must never be loaded.
comptime H3_EXTRACT_LAYER = 50

# Candidate decoder-layer key prefixes to probe, in the order tried. The
# first is the one already OBSERVED on a Qwen3-VL-8B checkpoint of the same
# `Qwen3VLForConditionalGeneration` architecture class (ideogram_qwen3vl.mojo);
# the other two are defensive fallbacks, not assumptions.
def _h3_prefix_candidates() -> List[String]:
    var out = List[String]()
    out.append(String("language_model.layers."))
    out.append(String("model.layers."))
    out.append(String("model.language_model.layers."))
    return out^


def _h3_cfg() -> Qwen3Config:
    return Qwen3Config(
        H3_HIDDEN, H3_LAYERS_TOTAL, H3_HEADS, H3_KV_HEADS, H3_HEAD_DIM,
        H3_EPS, H3_THETA,
    )


def _detect_layer_prefix(st: ShardedSafeTensors) raises -> String:
    """Probe the REAL shard index for which decoder-layer key prefix this
    checkpoint actually uses — do not guess (team-lead instruction). Checked
    against `input_layernorm.weight` of layer 0, which every candidate scheme
    has exactly one of."""
    var candidates = _h3_prefix_candidates()
    for i in range(len(candidates)):
        var probe = candidates[i] + "0.input_layernorm.weight"
        if st.has_tensor(probe):
            return candidates[i]
    raise Error(
        "minimax_h3_qwen3vl_streamed: none of the candidate decoder-layer"
        " prefixes (language_model.layers. / model.layers. /"
        " model.language_model.layers.) are present in the checkpoint index."
        " Read the real key set and add the correct prefix — do not guess."
    )


def _h3_embed_prefix(layer_prefix: String) raises -> String:
    """`embed_tokens` sits at the same nesting level as `layers.` for every
    scheme this file probes — mapped explicitly (not by string-slicing the
    `layers.` suffix off) against the exact candidates `_h3_prefix_candidates`
    lists, so a typo there cannot silently mismatch this one."""
    if layer_prefix == "language_model.layers.":
        return String("language_model.")
    if layer_prefix == "model.layers.":
        return String("model.")
    if layer_prefix == "model.language_model.layers.":
        return String("model.language_model.")
    raise Error(
        "minimax_h3_qwen3vl_streamed: unrecognized layer_prefix "
        + layer_prefix
        + " — add its embed_tokens mapping here"
    )


# Plain rename+load — NO FP8 dequant. `dst` is the plain "model.layers.N.*"
# spelling `Qwen3Encoder._layer` looks up internally; `src` is the REAL
# on-disk key (whatever `_detect_layer_prefix` found).
def _h3_add(
    st: ShardedSafeTensors,
    mut weights: List[TArc],
    mut n2i: Dict[String, Int],
    dst: String, src: String, ctx: DeviceContext,
) raises:
    var t = Tensor.from_view(st.tensor_view(src), ctx)
    n2i[dst] = len(weights)
    weights.append(ArcPointer(t^))


def _h3_load_layer(
    st: ShardedSafeTensors, li: Int, layer_prefix: String, ctx: DeviceContext
) raises -> Qwen3Encoder:
    """One decoder layer's 11 tensors -> mini Qwen3Encoder, native bf16
    (no dequant). Mirrors ideogram_qwen3vl_streamed.mojo's `_load_layer`."""
    var weights = List[TArc]()
    var n2i = Dict[String, Int]()
    var ps = layer_prefix + String(li) + "."
    var pd = String("model.layers.") + String(li) + "."
    _h3_add(st, weights, n2i, pd + "input_layernorm.weight", ps + "input_layernorm.weight", ctx)
    _h3_add(st, weights, n2i, pd + "self_attn.q_proj.weight", ps + "self_attn.q_proj.weight", ctx)
    _h3_add(st, weights, n2i, pd + "self_attn.k_proj.weight", ps + "self_attn.k_proj.weight", ctx)
    _h3_add(st, weights, n2i, pd + "self_attn.v_proj.weight", ps + "self_attn.v_proj.weight", ctx)
    _h3_add(st, weights, n2i, pd + "self_attn.o_proj.weight", ps + "self_attn.o_proj.weight", ctx)
    _h3_add(st, weights, n2i, pd + "self_attn.q_norm.weight", ps + "self_attn.q_norm.weight", ctx)
    _h3_add(st, weights, n2i, pd + "self_attn.k_norm.weight", ps + "self_attn.k_norm.weight", ctx)
    _h3_add(st, weights, n2i, pd + "post_attention_layernorm.weight", ps + "post_attention_layernorm.weight", ctx)
    _h3_add(st, weights, n2i, pd + "mlp.gate_proj.weight", ps + "mlp.gate_proj.weight", ctx)
    _h3_add(st, weights, n2i, pd + "mlp.up_proj.weight", ps + "mlp.up_proj.weight", ctx)
    _h3_add(st, weights, n2i, pd + "mlp.down_proj.weight", ps + "mlp.down_proj.weight", ctx)
    return Qwen3Encoder(weights^, n2i^, _h3_cfg())


def _h3_embed(
    st: ShardedSafeTensors, embed_prefix: String, ids: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """Transient mini-encoder holding ONLY embed_tokens (~1.45 GiB bf16);
    freed on return (mirrors ideogram_qwen3vl_streamed.mojo's `_embed_all`)."""
    var weights = List[TArc]()
    var n2i = Dict[String, Int]()
    _h3_add(st, weights, n2i, String("model.embed_tokens.weight"), embed_prefix + "embed_tokens.weight", ctx)
    var emb = Qwen3Encoder(weights^, n2i^, _h3_cfg())
    return emb._embed(ids, ctx)


def minimax_h3_encode_conditioning_streamed(
    dir_or_file: String, ids: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """H3's conditioner forward: text-only, layers 0..H3_EXTRACT_LAYER,
    pre-final-norm. Returns [1, seq, 5120] — feed straight into
    `minimax_h3_condition_embed` (models/dit/minimax_h3_frontend.mojo).

    No padding/PAD-id handling here (real_len == seq, full causal, no mask
    beyond causality) — that is tokenizer/chat-template wiring, explicitly
    the NEXT step, not this one."""
    if len(ids) == 0:
        raise Error("minimax_h3_encode_conditioning_streamed: empty prompt")
    var seq = len(ids)
    var st = ShardedSafeTensors.open(dir_or_file)
    var layer_prefix = _detect_layer_prefix(st)
    var embed_prefix = _h3_embed_prefix(layer_prefix)

    var hidden = _h3_embed(st, embed_prefix, ids, ctx)  # [1, seq, 5120]
    ctx.synchronize()  # embed table (~1.45 GiB) freed before layer streaming starts

    var dtype = hidden.dtype()
    var q_tables = _build_rope_tables(seq, H3_HEADS, H3_HEAD_DIM, H3_THETA)
    var k_tables = _build_rope_tables(seq, H3_KV_HEADS, H3_HEAD_DIM, H3_THETA)
    comptime half = H3_HEAD_DIM // 2
    var cq_sh = List[Int]()
    cq_sh.append(seq * H3_HEADS * half)
    var ck_sh = List[Int]()
    ck_sh.append(seq * H3_KV_HEADS * half)
    var cos_q = Tensor.from_host(q_tables[0], cq_sh.copy(), dtype, ctx)
    var sin_q = Tensor.from_host(q_tables[1], cq_sh.copy(), dtype, ctx)
    var cos_k = Tensor.from_host(k_tables[0], ck_sh.copy(), dtype, ctx)
    var sin_k = Tensor.from_host(k_tables[1], ck_sh.copy(), dtype, ctx)

    var mask_data = _build_causal_mask(seq, H3_HEADS, seq)  # real_len == seq (no padding here)
    var mask_sh = List[Int]()
    mask_sh.append(1)
    mask_sh.append(H3_HEADS)
    mask_sh.append(seq)
    mask_sh.append(seq)
    var mask = Tensor.from_host(mask_data, mask_sh^, dtype, ctx)

    for li in range(H3_EXTRACT_LAYER + 1):  # 0..50 inclusive, 51 layers
        var lw = _h3_load_layer(st, li, layer_prefix, ctx)
        hidden = lw._layer(li, hidden, cos_q, sin_q, cos_k, sin_k, mask, ctx)
        ctx.synchronize()  # this layer's ~0.95 GiB dropped before the next load

    return hidden^
