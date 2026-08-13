# serenitymojo/models/text_encoder/minimax_h3_qwen3vl_streamed.mojo
#
# MiniMax-H3's conditioner ("H3-Encoder"): a text-only forward through the
# first 50 decoder layers (0..49) of Qwen3-VL-32B-Instruct's text tower,
# returning `hidden_states[50]` PRE-final-norm in HF's own
# `output_hidden_states` indexing — the tensor `condition_proj` [5376, 5120]
# in serenitymojo/models/dit/minimax_h3_frontend.mojo consumes. t2va has no
# conditioning image, so the vision tower (`vision_config`, ~depth 27) is
# never loaded or run.
#
# INDEXING, VERIFIED EMPIRICALLY (not assumed) against the installed
# transformers 4.57.6 Qwen3VLTextModel: `output_hidden_states=True`'s
# `hidden_states` tuple has index 0 = embeddings, index k (k>=1) = the RAW
# (pre-norm) output after running layers 0..k-1 (k layers total), and ONLY
# the LAST tuple entry is overwritten with the post-final-norm
# `last_hidden_state` (`check_model_inputs(tie_last_hidden_states=True)`,
# the default for language models). So `hidden_states[50]` is the state
# after 50 layers have run — layer INDEX 50 itself is never executed. A
# stack truncated to exactly 50 layers would make index 50 the tied
# (post-norm) entry, which is exactly why the diffusers reference code
# raises if `num_hidden_layers <= 50` (encoders.py) — it needs layer 50 to
# exist so hidden_states[50] stays a genuine intermediate entry, but it
# never runs it. Verification script:
# parity/minimax_h3_conditioner_real_weight_oracle.py.
#
# STATUS 2026-08-02: builds and typechecks; NOT numerically gated. H3's own
# text_encoder/ (14 shards, 62.13 GiB) is still downloading — only
# config.json has landed (see minimax_h3_qwen3vl_streamed_probe.mojo, which
# runs for real if the checkpoint is present and otherwise says so plainly).
# Do not treat a clean build here as parity.
#
# DEEPSTACK (2026-08-03): `minimax_h3_encode_conditioning_streamed[_depth]`
# now optionally accept a `MiniMaxH3VisionOutput` (from
# minimax_h3_qwen3vl_vision.mojo) + its `visual_positions`, splicing the
# tower's merged embeds into the embedding stream once and adding its three
# deepstack taps after decoder layers 0/1/2 — see the DEEPSTACK INJECTION
# header immediately above `minimax_h3_mm_token_type_ids` below for the
# vendor citation and exact ordering. Both params default to `None`; t2va
# (text-only, no conditioning image) is unaffected. Gated on CPU-synthesized
# inputs (injection mechanics only, not the vision tower or the 50-layer H3
# conditioner) by parity/minimax_h3_deepstack_gate.mojo +
# scripts/minimax_h3_deepstack_oracle.py.
#
# CONFIG is the real, landed
# /home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder/config.json
# (`text_config`), not a guess: hidden_size 5120, num_hidden_layers 64 (only
# 0..49 executed, see INDEXING above), num_attention_heads 64, num_key_value_heads 8 (GQA n_rep
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
# After each synchronization the source mmap pages are also released with
# `MADV_DONTNEED` via `ShardedSafeTensors.release_to_os()`. The device copy is
# complete at that point, so retaining those clean file-backed pages only grows
# the host cgroup by roughly one checkpoint layer per iteration. This mirrors
# the bounded-residency contract in `gemma3_ltx_streamed.mojo` and is required
# for H3's 62.13 GiB text checkpoint to encode inside a regular user scope.
#
# DISK COST (state this plainly, it is real and not hidden by streaming):
# one encode call reads ~50 layers x 0.95 GiB + 1.45 GiB embed table
# = ~49 GiB from disk. This happens ONCE PER PROMPT (conditioning is computed
# once, outside the denoising loop), not once per diffusion step — a bounded
# ~15-30s one-time tax per generation at a few GiB/s NVMe, not a per-step cost.
#
# Only a single prompt is supported for now (no multi-prompt batching like
# ideogram's N-prompt loop) — that can be added later following the same
# per-layer-shared-weights pattern if H3 needs it.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor, BatchedTensorUploader
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.qwen3_encoder import (
    Qwen3Encoder,
    Qwen3Config,
    _build_rope_tables,
    _build_causal_mask,
)
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    MiniMaxH3VisionOutput,
    minimax_h3_vision_deepstack_lm_layers,
)

comptime TArc = ArcPointer[Tensor]

# text_config, MiniMax-H3/FL2VA/text_encoder/config.json (landed 2026-08-02).
comptime H3_HIDDEN = 5120
comptime H3_LAYERS_TOTAL = 64  # released layer count; only 0..H3_EXTRACT_LAYER-1 run (see INDEXING above)
comptime H3_HEADS = 64
comptime H3_KV_HEADS = 8
comptime H3_HEAD_DIM = 128
comptime H3_THETA = Float64(5000000.0)
comptime H3_EPS = Float32(1.0e-6)

# One persistent pinned-host slab for every checkpoint upload in an encode.
# The largest tensor is embed_tokens [151936,5120] BF16 = 1,555,824,640 B;
# 1536 MiB fits it with 54.8 MiB spare. Every complete decoder layer is
# smaller (~0.95 GiB), so the same allocation is reused for all 50 layers.
# This avoids `Tensor.from_view()`'s one-pinned-buffer-per-tensor churn, whose
# retained CUDA/MAX staging allocations measured 10.0 GiB in the product
# cgroup before layer streaming could finish.
comptime H3_UPLOAD_STAGE_BYTES = 1536 * 1024 * 1024

# diffusers PR packing.py MINIMAX_H3_TEXT_ENCODER_LAYER = 50: the HF
# `hidden_states` index MiniMax-H3 conditions on. Per INDEXING above, that
# is the state after 50 layers run (indices 0..49) — layer index 50 itself
# is NEVER executed. Layers 50..63 and the LM head are dead weight for H3
# and must never be loaded.
comptime H3_EXTRACT_LAYER = 50

# ── partial-checkout shard opening ──────────────────────────────────────────
# `ShardedSafeTensors.open(dir)` eagerly opens EVERY shard the index lists
# and fails on the first missing one (io/sharded.mojo:384-431) — confirmed
# against this exact checkpoint by
# models/dit/parity/minimax_h3_block_real_weight_gate.mojo. H3's
# text_encoder/ is a 62.13 GiB, 14-shard download that lands incrementally,
# so that path is unusable here until every shard exists. Same fix
# models/dit/parity/minimax_h3_modcache_real_weight_gate.mojo uses for the
# transformer checkpoint: build a `ShardedSafeTensors` from its own public
# constructor (`ShardedSafeTensors(shards, name_to_shard)`, sharded.mojo:375)
# over independently-`SafeTensors.open`'d single files — but generalized to
# ALL 14 candidate shard filenames, probed at runtime, rather than a
# hand-picked one or two. Each shard self-reports its own tensor names
# (`SafeTensors.names()`), so no index.json parsing is needed at all: this
# naturally picks up shard 11 and any other shard the moment it lands on
# disk, no code change required — whichever shards hold the tensors for
# layers 0..49 (see INDEXING above: layer 50 itself is never executed, so
# its shard is not actually required, but layers up to 49 may still span
# shard 11 or others not yet measured).
def _h3_shard_filenames() -> List[String]:
    var out = List[String]()
    out.append(String("model-00001-of-00014.safetensors"))
    out.append(String("model-00002-of-00014.safetensors"))
    out.append(String("model-00003-of-00014.safetensors"))
    out.append(String("model-00004-of-00014.safetensors"))
    out.append(String("model-00005-of-00014.safetensors"))
    out.append(String("model-00006-of-00014.safetensors"))
    out.append(String("model-00007-of-00014.safetensors"))
    out.append(String("model-00008-of-00014.safetensors"))
    out.append(String("model-00009-of-00014.safetensors"))
    out.append(String("model-00010-of-00014.safetensors"))
    out.append(String("model-00011-of-00014.safetensors"))
    out.append(String("model-00012-of-00014.safetensors"))
    out.append(String("model-00013-of-00014.safetensors"))
    out.append(String("model-00014-of-00014.safetensors"))
    return out^


def _h3_path_readable(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _h3_open_available_shards(dir: String) raises -> ShardedSafeTensors:
    """Open only the text_encoder/ shard FILES THAT EXIST on disk right
    now. Tensors that live in a not-yet-downloaded shard simply are not in
    the resulting `name_to_shard` map — a lookup for one of them raises a
    clear "missing tensor" error (from `has_tensor`/`tensor_view`) rather
    than this function eagerly failing on shards it never needed to open."""
    var shards = List[ArcPointer[SafeTensors]]()
    var name_to_shard = Dict[String, Int]()
    var filenames = _h3_shard_filenames()
    var opened = 0
    for i in range(len(filenames)):
        var path = dir + "/" + filenames[i]
        if not _h3_path_readable(path):
            continue
        var st = SafeTensors.open(path)
        var idx = len(shards)
        for ref nm in st.names():
            name_to_shard[nm] = idx
        shards.append(ArcPointer(st^))
        opened += 1
    if opened == 0:
        raise Error(
            "minimax_h3_qwen3vl_streamed: no text_encoder shard files (model-000NN-of-00014.safetensors) present in "
            + dir
        )
    return ShardedSafeTensors(shards^, name_to_shard^)


# Candidate decoder-layer key prefixes to probe, in the order tried. The
# first is the one already OBSERVED on a Qwen3-VL-8B checkpoint of the same
# `Qwen3VLForConditionalGeneration` architecture class (ideogram_qwen3vl.mojo);
# the other two are defensive fallbacks, not assumptions. MEASURED against
# H3's own real index (2026-08-02, 704 decoder-layer keys): the THIRD
# candidate, `model.language_model.layers.`, is the one this checkpoint
# actually uses — neither of the first two has any keys at all. Left in
# probed order rather than reordered, so the detection is still real and not
# a hardcoded assumption dressed up as one.
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
    mut uploader: BatchedTensorUploader,
    mut weights: List[TArc],
    mut n2i: Dict[String, Int],
    dst: String, src: String, ctx: DeviceContext,
) raises:
    var t = uploader.from_view(st.tensor_view(src), ctx)
    n2i[dst] = len(weights)
    weights.append(ArcPointer(t^))


def _h3_load_layer(
    st: ShardedSafeTensors,
    mut uploader: BatchedTensorUploader,
    li: Int,
    layer_prefix: String,
    ctx: DeviceContext,
) raises -> Qwen3Encoder:
    """One decoder layer's 11 tensors -> mini Qwen3Encoder, native bf16
    (no dequant). Mirrors ideogram_qwen3vl_streamed.mojo's `_load_layer`."""
    var weights = List[TArc]()
    var n2i = Dict[String, Int]()
    var ps = layer_prefix + String(li) + "."
    var pd = String("model.layers.") + String(li) + "."
    _h3_add(st, uploader, weights, n2i, pd + "input_layernorm.weight", ps + "input_layernorm.weight", ctx)
    _h3_add(st, uploader, weights, n2i, pd + "self_attn.q_proj.weight", ps + "self_attn.q_proj.weight", ctx)
    _h3_add(st, uploader, weights, n2i, pd + "self_attn.k_proj.weight", ps + "self_attn.k_proj.weight", ctx)
    _h3_add(st, uploader, weights, n2i, pd + "self_attn.v_proj.weight", ps + "self_attn.v_proj.weight", ctx)
    _h3_add(st, uploader, weights, n2i, pd + "self_attn.o_proj.weight", ps + "self_attn.o_proj.weight", ctx)
    _h3_add(st, uploader, weights, n2i, pd + "self_attn.q_norm.weight", ps + "self_attn.q_norm.weight", ctx)
    _h3_add(st, uploader, weights, n2i, pd + "self_attn.k_norm.weight", ps + "self_attn.k_norm.weight", ctx)
    _h3_add(st, uploader, weights, n2i, pd + "post_attention_layernorm.weight", ps + "post_attention_layernorm.weight", ctx)
    _h3_add(st, uploader, weights, n2i, pd + "mlp.gate_proj.weight", ps + "mlp.gate_proj.weight", ctx)
    _h3_add(st, uploader, weights, n2i, pd + "mlp.up_proj.weight", ps + "mlp.up_proj.weight", ctx)
    _h3_add(st, uploader, weights, n2i, pd + "mlp.down_proj.weight", ps + "mlp.down_proj.weight", ctx)
    uploader.finish(ctx)
    return Qwen3Encoder(weights^, n2i^, _h3_cfg())


def _h3_embed(
    st: ShardedSafeTensors,
    mut uploader: BatchedTensorUploader,
    embed_prefix: String,
    ids: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    """Transient mini-encoder holding ONLY embed_tokens (~1.45 GiB bf16);
    freed on return (mirrors ideogram_qwen3vl_streamed.mojo's `_embed_all`)."""
    var weights = List[TArc]()
    var n2i = Dict[String, Int]()
    _h3_add(st, uploader, weights, n2i, String("model.embed_tokens.weight"), embed_prefix + "embed_tokens.weight", ctx)
    uploader.finish(ctx)
    var emb = Qwen3Encoder(weights^, n2i^, _h3_cfg())
    return emb._embed(ids, ctx)


# ── DEEPSTACK INJECTION (2026-08-03) ─────────────────────────────────────────
#
# Ground truth: the installed transformers 4.57.6
# `Qwen3VLTextModel.forward`/`_deepstack_process`
# (.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:849-883):
#
#   for layer_idx, decoder_layer in enumerate(self.layers):
#       layer_outputs = decoder_layer(hidden_states, ...)
#       hidden_states = layer_outputs                        # :859
#       if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
#           hidden_states = self._deepstack_process(          # :862-867
#               hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx],
#           )
#   ...
#   def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
#       local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds   # :881
#       hidden_states[visual_pos_masks, :] = local_this                          # :882
#       return hidden_states
#
# i.e. the tap is ADDED, at VISUAL-TOKEN POSITIONS ONLY, AFTER the decoder
# layer's own forward has already completed (line 859's assignment runs
# strictly BEFORE the deepstack check at line 862 — the layer sees the
# PRE-tap state as its input; the tap only ever modifies that layer's
# OUTPUT). `minimax_h3_deepstack_add` below reproduces `_deepstack_process`
# exactly (masked add, nothing else touched); `minimax_h3_splice_vision_embeds`
# reproduces the SEPARATE, one-time substitution of the vision tower's merged
# `image_features` into the embedding stream at the placeholder positions
# (`get_placeholder_mask` + the `inputs_embeds[special_image_mask] = ...`
# assignment pattern, modeling_qwen3_vl.py:1066-1104) that happens once,
# before layer 0, not at every deepstack layer.
#
# `minimax_h3_mm_token_type_ids`/`minimax_h3_visual_positions` reproduce
# `processor.create_mm_token_type_ids` (processing_qwen3_vl.py:241-245:
# `mm_token_type_ids = np.zeros_like(input_ids); mm_token_type_ids[array_ids
# == self.image_token_id] = 1`) generalized to a 3-way text/image/video
# scheme, purely from OUR OWN token stream — no HF processor object needed.
#
# These four functions are plain host-side List[Int]/List[Float32] helpers
# (no Tensor, no DeviceContext) so they are directly unit-testable on CPU —
# see parity/minimax_h3_deepstack_gate.mojo.
def minimax_h3_mm_token_type_ids(
    token_ids: List[Int], image_pad_id: Int, video_pad_id: Int
) -> List[Int]:
    """0 for text, 1 at an `<|image_pad|>` id, 2 at a `<|video_pad|>` id.

    Reproduces `processor.create_mm_token_type_ids` (see header above),
    generalized to also mark video-pad positions as 2 (the installed
    transformers release's own helper only special-cases the image id; H3's
    Mojo side needs the 3-way scheme since keyframe/video pipelines route
    through this same file)."""
    var out = List[Int]()
    out.reserve(len(token_ids))
    for i in range(len(token_ids)):
        var t = token_ids[i]
        if t == image_pad_id:
            out.append(1)
        elif t == video_pad_id:
            out.append(2)
        else:
            out.append(0)
    return out^


def minimax_h3_visual_positions(mm_token_type_ids: List[Int]) -> List[Int]:
    """Sequence indices where `mm_token_type_ids[i] != 0`, in order — the
    `visual_pos_masks` equivalent. Ordered so the k-th entry here corresponds
    to row k of the vision embeds / deepstack tap tensors (the tower emits
    its tokens in the same left-to-right order they appear in the prompt)."""
    var out = List[Int]()
    for i in range(len(mm_token_type_ids)):
        if mm_token_type_ids[i] != 0:
            out.append(i)
    return out^


def minimax_h3_splice_vision_embeds(
    mut inputs_embeds: List[Float32],
    vision_embeds: List[Float32],
    visual_positions: List[Int],
    hidden: Int,
) raises:
    """REPLACE (not add) the embedding rows at `visual_positions` with the
    tower's merged `vision_embeds`, row k -> position `visual_positions[k]`.
    `inputs_embeds` is `[seq*hidden]` row-major, mutated in place. A plain
    row copy (no arithmetic) — bit-exact by construction.

    Raises if `len(vision_embeds) != len(visual_positions) * hidden` (the
    `get_placeholder_mask` count-mismatch check, modeling_qwen3_vl.py:1092-1095,
    reproduced host-side)."""
    var n = len(visual_positions)
    if len(vision_embeds) != n * hidden:
        raise Error(
            "minimax_h3_splice_vision_embeds: vision_embeds has "
            + String(len(vision_embeds))
            + " floats, expected "
            + String(n * hidden)
            + " ("
            + String(n)
            + " visual positions x hidden="
            + String(hidden)
            + ")"
        )
    for k in range(n):
        var pos = visual_positions[k]
        var dst0 = pos * hidden
        var src0 = k * hidden
        for j in range(hidden):
            inputs_embeds[dst0 + j] = vision_embeds[src0 + j]


def minimax_h3_deepstack_add(
    mut hidden_states: List[Float32],
    deepstack_tap: List[Float32],
    visual_positions: List[Int],
    hidden: Int,
) raises:
    """`_deepstack_process`: ADD `deepstack_tap` rows into `hidden_states` at
    `visual_positions` ONLY (row k -> position `visual_positions[k]`); every
    other row of `hidden_states` is left untouched. `hidden_states` is
    `[seq*hidden]` row-major, mutated in place.

    Raises if `len(deepstack_tap) != len(visual_positions) * hidden` (same
    shape-mismatch guard as `minimax_h3_splice_vision_embeds`, mirroring
    `get_placeholder_mask`'s own check)."""
    var n = len(visual_positions)
    if len(deepstack_tap) != n * hidden:
        raise Error(
            "minimax_h3_deepstack_add: deepstack_tap has "
            + String(len(deepstack_tap))
            + " floats, expected "
            + String(n * hidden)
            + " ("
            + String(n)
            + " visual positions x hidden="
            + String(hidden)
            + ")"
        )
    for k in range(n):
        var pos = visual_positions[k]
        var dst0 = pos * hidden
        var src0 = k * hidden
        for j in range(hidden):
            hidden_states[dst0 + j] = hidden_states[dst0 + j] + deepstack_tap[src0 + j]


def minimax_h3_encode_conditioning_streamed_depth(
    dir_or_file: String,
    ids: List[Int],
    num_layers: Int,
    ctx: DeviceContext,
    vision: Optional[MiniMaxH3VisionOutput] = None,
    visual_positions: Optional[List[Int]] = None,
) raises -> Tensor:
    """Run exactly `num_layers` decoder layers (0..num_layers-1) and return
    that raw (pre-norm) hidden state — i.e. HF's `output_hidden_states`
    convention `hidden_states[num_layers]` (index 0 = embeddings, index k
    for k>=1 = raw output after layer k-1, so index k = state after k
    layers have run). VERIFIED against the installed transformers 4.57.6
    Qwen3VLTextModel/Qwen3VLModel (`check_model_inputs`'s capture: it
    records the layer-0 INPUT once, then every layer's OUTPUT, and only the
    LAST entry of the whole tuple is overwritten with the post-final-norm
    `last_hidden_state` — everything before that is the raw per-layer
    output). `minimax_h3_encode_conditioning_streamed` (H3's real
    contract, num_layers=H3_EXTRACT_LAYER=50) is a thin wrapper over this;
    this generic depth parameter exists so parity/minimax_h3_conditioner_
    real_weight_gate.mojo can gate at depths the currently-available shards
    actually support (e.g. 1 and 23) without waiting for shard 11.

    DEEPSTACK (optional, backward-compatible): `vision`/`visual_positions`
    default to `None`, in which case this function behaves EXACTLY as before
    (t2va is text-only and depends on that). When both are supplied (they
    must be supplied TOGETHER — passing only one raises), the vision
    tower's merged embeds are SPLICED into the embedding stream ONCE, before
    layer 0 (`minimax_h3_splice_vision_embeds`), and its three deepstack
    taps are ADDED at `visual_positions` ONLY, immediately AFTER decoder
    layers 0, 1, 2 finish their own forward — matching
    `Qwen3VLTextModel.forward`'s own ordering exactly
    (modeling_qwen3_vl.py:849-867, see the DEEPSTACK INJECTION header above
    `minimax_h3_mm_token_type_ids`). The three taps must already be resident
    in `vision` (a `MiniMaxH3VisionOutput`) before this call begins — they
    are consumed immediately at layers 0-2, so there is no lazy mid-decode
    load path here."""
    if len(ids) == 0:
        raise Error("minimax_h3_encode_conditioning_streamed_depth: empty prompt")
    if num_layers <= 0:
        raise Error("minimax_h3_encode_conditioning_streamed_depth: num_layers must be > 0")
    var has_vision = vision.__bool__()
    if has_vision != visual_positions.__bool__():
        raise Error(
            "minimax_h3_encode_conditioning_streamed_depth: vision and"
            " visual_positions must be supplied together (both Some or both"
            " None) — deepstack injection needs both the tower output and"
            " the positions it substitutes/adds at."
        )
    var seq = len(ids)
    var st = _h3_open_available_shards(dir_or_file)
    var layer_prefix = _detect_layer_prefix(st)
    var embed_prefix = _h3_embed_prefix(layer_prefix)
    var uploader = BatchedTensorUploader(H3_UPLOAD_STAGE_BYTES, ctx)

    var hidden = _h3_embed(st, uploader, embed_prefix, ids, ctx)  # [1, seq, 5120]
    ctx.synchronize()  # embed table (~1.45 GiB) freed before layer streaming starts
    st.release_to_os()  # H2D is complete; drop clean embed-table mmap pages

    if has_vision:
        # Splice ONCE, before layer 0 — layer 0 must consume the embedding
        # stream with the tower's merged embeds already substituted in
        # (get_placeholder_mask's substitution, modeling_qwen3_vl.py:
        # 1066-1104). `hidden` is [1, seq, H3_HIDDEN]; the splice helper is a
        # plain host List[Float32] op, so round-trip it through F32 host
        # memory via the existing `Tensor.to_host`/`Tensor.from_host`
        # boundary (a few hundred KB, negligible next to the multi-GiB
        # per-layer disk streaming this file already pays).
        var vpos = visual_positions.value().copy()
        var host_embeds = hidden.to_host(ctx)
        minimax_h3_splice_vision_embeds(host_embeds, vision.value().embeds, vpos, H3_HIDDEN)
        hidden = Tensor.from_host(host_embeds, hidden.shape(), hidden.dtype(), ctx)

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

    var deepstack_lm_layers = minimax_h3_vision_deepstack_lm_layers()  # [0, 1, 2]

    for li in range(num_layers):  # 0..num_layers-1
        var lw = _h3_load_layer(st, uploader, li, layer_prefix, ctx)
        hidden = lw._layer(li, hidden, cos_q, sin_q, cos_k, sin_k, mask, ctx)
        # Deepstack add happens AFTER the layer's own forward completes,
        # matching modeling_qwen3_vl.py:849-867 EXACTLY: `hidden_states =
        # layer_outputs` (859) runs BEFORE the `layer_idx in
        # range(len(deepstack_visual_embeds))` check (862) — the layer sees
        # the PRE-tap state as its input; the tap is applied to its output.
        if has_vision:
            for k in range(len(deepstack_lm_layers)):
                if deepstack_lm_layers[k] == li:
                    var vpos2 = visual_positions.value().copy()
                    var tap = vision.value().deepstack_block(k)
                    var host_hidden = hidden.to_host(ctx)
                    minimax_h3_deepstack_add(host_hidden, tap, vpos2, H3_HIDDEN)
                    hidden = Tensor.from_host(host_hidden, hidden.shape(), hidden.dtype(), ctx)
        ctx.synchronize()  # this layer's ~0.95 GiB dropped before the next load
        st.release_to_os()  # H2D/compute complete; bound host mmap residency

    print(
        "  H3 conditioner uploads:", uploader.tensors_uploaded, "tensors,",
        uploader.bytes_uploaded, "bytes,", uploader.fence_count,
        "fences through one", H3_UPLOAD_STAGE_BYTES, "byte pinned slab",
    )

    return hidden^


def minimax_h3_encode_conditioning_streamed(
    dir_or_file: String,
    ids: List[Int],
    ctx: DeviceContext,
    vision: Optional[MiniMaxH3VisionOutput] = None,
    visual_positions: Optional[List[Int]] = None,
) raises -> Tensor:
    """H3's real conditioner forward: `hidden_states[H3_EXTRACT_LAYER]` in
    the sense fixed above — i.e. runs layers 0..H3_EXTRACT_LAYER-1 (50
    layers total, NOT including layer index 50 itself; see
    `minimax_h3_encode_conditioning_streamed_depth`'s header for why layer
    50 is never executed). Returns [1, seq, 5120] — feed straight into
    `minimax_h3_condition_embed` (models/dit/minimax_h3_frontend.mojo).

    BUG FIXED 2026-08-02: this function previously ran H3_EXTRACT_LAYER+1
    layers (0..50 inclusive, 51 layers) — one layer too many. Caught before
    any oracle existed, by empirically verifying transformers 4.57.6's
    `output_hidden_states` capture semantics (`check_model_inputs`) rather
    than assuming HF's own indexing convention. See parity/
    minimax_h3_conditioner_real_weight_oracle.py for the verification script.

    No padding/PAD-id handling here (real_len == seq, full causal, no mask
    beyond causality) — that is tokenizer/chat-template wiring, already done
    in minimax_h3_conditioning.mojo.

    DEEPSTACK (optional, backward-compatible): `vision`/`visual_positions`
    default to `None` and are simply forwarded to
    `minimax_h3_encode_conditioning_streamed_depth` — see that function's
    header for the full contract. t2va (text-only, no conditioning image)
    calls this with the 3-positional-argument form and is therefore
    completely unaffected."""
    return minimax_h3_encode_conditioning_streamed_depth(
        dir_or_file, ids, H3_EXTRACT_LAYER, ctx, vision, visual_positions
    )
