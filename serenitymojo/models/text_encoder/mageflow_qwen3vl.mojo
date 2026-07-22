# models/text_encoder/mageflow_qwen3vl.mojo — Mage-Flow Qwen3-VL TEXT conditioning.
#
# Mage-Flow's text_encoder is `Qwen3VLForConditionalGeneration` (text backbone
# keys `model.language_model.*`, vision `model.visual.*`), text_config IDENTICAL
# to Krea-2's Qwen3-VL-4B: hidden 2560, 36 layers, 32 heads / 8 KV GQA,
# head_dim 128, inter 9728, silu, rms 1e-6, rope_theta 5e6, mrope [24,20,20],
# vocab 151936, tied. So the WEIGHT LOADER is reused verbatim from
# krea2_qwen3vl_4b (`load_krea2_qwen3vl_4b` remaps `model.language_model.*` ->
# `model.*` and loads BF16 — Mage's shard keys match byte-for-byte).
#
# The ONLY difference vs Krea-2 is how the conditioning is EXTRACTED. Krea-2
# stacks 12 PRE-final-norm layers -> [1,L',12,2560]. Mage-Flow uses the SINGLE
# last_hidden_state POST model.norm, then drops the leading system-prompt rows:
#   ref: mage_flow/models/modules/text_encoder.py TextEncoder.forward
#     hidden = outputs.last_hidden_state          # [1, L, 2560] POST model.norm
#     hidden = hidden.squeeze(0)                  # [L, 2560]
#     h_valid = hidden[drop_idx:]                 # drop system prompt -> txt
#     vec = h_valid.mean(dim=0)                   # pooled [2560]
#   ref: mage_flow/models/utils.py PROMPT_TEMPLATE
#     "mage-flow":      start_idx = 34   (t2i)
#     "mage-flow-edit": start_idx = 64   (image edit)
# The DiT context IS this `txt` [1, L-drop, 2560] (pipeline.py _encode_texts_packed
# returns res["txt"] straight into _build_pack_ctx / the transformer forward).
#
# RoPE: Mage's mrope_interleaved collapses to plain half-split RoPE for TEXT-ONLY
# inputs (all 3 mrope sections share one 1D position sequence -> identical
# per-position angles), which is exactly what Qwen3Encoder.rope_halfsplit +
# rope_theta 5e6 computes. (Same reduction the Boogu C7 Qwen3-VL gate validated.)
#
# ── EDIT (image-conditioned) path — encode_mageflow_edit ─────────────────────
# The `.edit()` path additionally feeds a reference image through the SAME
# Qwen3-VL: vision tower (model.visual.*) -> deepstack -> the text decoder fuses
# them exactly like Qwen3VLForConditionalGeneration. Ref: mage_flow/pipeline.py
#   generate_edits -> _encode_edits_packed (:396-417):
#     formatted = template("mage-flow-edit").format(
#         "Image 1: <|vision_start|><|image_pad|><|vision_end|>" + instruction)
#     vl = processor(text=[formatted], images=[ref])   # -> input_ids,
#                                                       #    pixel_values, image_grid_thw
#     res = model.txt_enc(input_ids, cu, inputs=vl, drop_idx_override=64)
#     -> DiT context = res["txt"]  (same post-norm last_hidden_state, drop 64)
#   The ref image long edge is capped at 384 (vl_cond_long_edge) BEFORE the
#   processor (pipeline.py:533).
#
# CRITICAL DIVERGENCE from LingBot's fuse_core (spatial M-RoPE): Mage's
# TextEncoder.forward builds `position_ids = arange(length)` per sequence (1D,
# expanded to 3 IDENTICAL mrope axes) and PASSES it to the model. HF's
# Qwen3VLModel.forward only calls get_rope_index (the 2D spatial image positions)
# when `position_ids is None` (modeling_qwen3_vl.py:1335). Since Mage always
# supplies the plain arange, the spatial get_rope_index is SKIPPED -> the
# interleaved M-RoPE degenerates to plain sequential RoPE for EVERY token, image
# tokens included. So encode_mageflow_edit REUSES lingbot's masked_scatter +
# deepstack + decoder helpers, but replaces the spatial position table with a
# plain-arange one (all 3 axes = token index).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder
# Re-export the loader so callers get everything from one module. The loader is
# a byte-for-byte fit for Mage's `model.language_model.*` shard keys.
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import load_krea2_qwen3vl_4b
# EDIT path: reuse the (parity-gated) LingBot vision tower + the fusion helpers.
from serenitymojo.models.text_encoder.lingbot_qwen3vl_vision import (
    Qwen3VLVisionModel, VisionOutput,
)
from serenitymojo.models.text_encoder.lingbot_qwen3vl_fuse import (
    Positions, _build_mrope_tables, _causal_mask_data, _replace_rows, _add_rows,
    _to_rows, _cast, _pad_bucket, IMAGE_TOKEN_ID, FUSE_PAD_ID,
)


# Mage-Flow drop_idx per template (utils.py PROMPT_TEMPLATE start_idx).
comptime MAGEFLOW_T2I_DROP_IDX = 34
comptime MAGEFLOW_EDIT_DROP_IDX = 64

# Qwen3 pad/bos id (151643). Mage template tokens are 151644/151645, so padding
# with 151643 lets encode_layer_states' pad auto-detect recover real_len.
comptime _MAGEFLOW_PAD_ID = 151643


def _mageflow_pad_ids(ids: List[Int]) raises -> List[Int]:
    """Pad the token list UP to the smallest comptime-supported SDPA seq
    (64/128/256/512/1024/2048) with the pad id, so encode_layer_states' pad
    auto-detect (first 151643) recovers real_len = the original L. The encoder
    is causal so pad columns are masked out — numerically identical to unpadded
    L for the kept (pre-pad) rows."""
    var L = len(ids)
    var pad: Int
    if L <= 64:
        pad = 64
    elif L <= 128:
        pad = 128
    elif L <= 256:
        pad = 256
    elif L <= 512:
        pad = 512
    elif L <= 1024:
        pad = 1024
    elif L <= 2048:
        pad = 2048
    else:
        raise Error(
            String("mageflow encode: L=") + String(L)
            + " exceeds 2048 (max supported encoder SDPA seq)"
        )
    var out = List[Int]()
    for i in range(L):
        out.append(ids[i])
    for _i in range(pad - L):
        out.append(_MAGEFLOW_PAD_ID)
    return out^


def encode_mageflow_text(
    enc: Qwen3Encoder, ids: List[Int], drop_idx: Int, ctx: DeviceContext
) raises -> Tensor:
    """Mage-Flow text conditioning: the POST-final-norm last_hidden_state with
    the leading `drop_idx` system-prompt rows dropped -> [1, L-drop_idx, 2560].

    `ids` are the EXACT token ids of the full templated sequence
    (template.format(prompt) tokenized), i.e. what pipeline.py's
    _encode_texts_packed feeds the model. Runs all 36 Qwen3 layers, applies
    model.norm (final_norm) to the last layer output, and slices rows
    [drop_idx, L) — dropping the system prefix AND the SDPA right-padding in one
    narrow. This IS the `txt` the Mage DiT consumes."""
    var L = len(ids)
    if L <= drop_idx:
        raise Error(
            String("mageflow encode: L=") + String(L)
            + " <= drop_idx=" + String(drop_idx)
            + " (prompt produced no post-prefix tokens)"
        )
    var padded = _mageflow_pad_ids(ids)
    # All 36 layer outputs, each [1, L_pad, 2560], PRE-final-norm.
    var states = enc.encode_layer_states(padded, ctx)
    # Last layer output (index num_layers-1), PRE-final-norm. Pass the borrowed
    # deref straight into final_norm (Tensor is not ImplicitlyCopyable).
    # POST model.norm -> Mage's outputs.last_hidden_state.
    var normed = enc.final_norm(states[len(states) - 1][], ctx)  # [1, L_pad, 2560]
    var keep = L - drop_idx
    # Drop the system prefix rows AND the padding: [drop_idx, drop_idx+keep).
    return slice(normed, 1, drop_idx, keep, ctx)  # [1, keep, 2560]


# ── EDIT (image-conditioned) conditioning ────────────────────────────────────
def _mageflow_edit_positions(
    ids: List[Int], L: Int, seq_pad: Int
) raises -> Positions:
    """Mage edit positions: PLAIN sequential arange on all 3 mrope axes (t=h=w=i)
    for EVERY token, image tokens included — the degenerate M-RoPE Mage's
    TextEncoder.forward induces by passing an explicit `arange(length)`
    position_ids (see module header). The contiguous image span (image_pad
    151655) is located for masked_scatter + deepstack; nvis == the image_pad
    count == the vision merged-token count."""
    var pt = List[Int](); pt.resize(seq_pad, 0)
    var ph = List[Int](); ph.resize(seq_pad, 0)
    var pw = List[Int](); pw.resize(seq_pad, 0)
    var vis_start = -1
    var nvis = 0
    for i in range(L):
        pt[i] = i
        ph[i] = i
        pw[i] = i
        if ids[i] == IMAGE_TOKEN_ID:
            if vis_start < 0:
                vis_start = i
            nvis += 1
    # padded rows keep pt=ph=pw=0; they are masked out as causal columns.
    if vis_start < 0:
        raise Error(
            "mageflow edit: no image_pad (151655) token found in input_ids"
        )
    return Positions(pt^, ph^, pw^, vis_start, nvis)


def fuse_mageflow_edit(
    enc: Qwen3Encoder,
    pooler: Tensor,       # [nvis, H]   vision pooler_output (F32 or bf16)
    ds0: Tensor, ds1: Tensor, ds2: Tensor,  # [nvis, H] deepstack features
    ids: List[Int],
    crop_start: Int,
    ctx: DeviceContext,
    f32_stream: Bool = False,
) raises -> Tensor:
    """Mage edit fusion given already-computed vision tokens: masked_scatter the
    pooler rows into the token embeddings at image_pad (151655), run the Qwen3-VL
    text decoder with PLAIN-arange (degenerate) M-RoPE and the deepstack add at
    layers 0/1/2, apply model.norm, drop the leading `crop_start` rows. Returns
    [1, L-crop_start, 2560] = the DiT context (res["txt"]).

    Identical to lingbot fuse_core EXCEPT the position table: Mage passes a plain
    arange position_ids, so the spatial get_rope_index is skipped (see header).

    `f32_stream`: run the decoder activation stream in F32 (weights stay bf16).
    The image-token rows of the fused last_hidden_state are ill-conditioned in
    bf16 (HF's own bf16-vs-f32 cos there ~0.95), so the fusion MATH is only
    gateable at cos>=0.999 in F32. Default False = the native bf16 product path
    the real Mage pipeline runs."""
    var cfg = enc.config
    var H = cfg.hidden_size
    var heads = cfg.num_heads
    var kv = cfg.num_kv_heads
    var dh = cfg.head_dim
    var half = dh // 2
    var L = len(ids)
    var act_dt = STDtype.F32 if f32_stream else STDtype.BF16

    # positions (plain arange) + visual span
    var seq_pad = _pad_bucket(L)
    var pos = _mageflow_edit_positions(ids, L, seq_pad)
    var nvis = pos.nvis
    var vis_start = pos.vis_start

    # padded token ids (image rows overwritten below; pad rows masked)
    var padded = List[Int]()
    for i in range(L):
        padded.append(ids[i])
    for _i in range(seq_pad - L):
        padded.append(FUSE_PAD_ID)

    # 1) token embeddings (bf16 table), then masked_scatter the pooler rows
    var embed = _cast(enc._embed(padded, ctx), act_dt, ctx)   # [1,seq_pad,H]
    var pooler_a = _to_rows(pooler, nvis, H, act_dt, ctx)      # [1,nvis,H]
    var inputs_embeds = _replace_rows(embed, pooler_a, vis_start, ctx)

    # 2) plain-arange M-RoPE tables (act dtype; HF computes cos/sin in f32)
    var q_tab = _build_mrope_tables(pos, seq_pad, heads, dh, cfg.rope_theta)
    var k_tab = _build_mrope_tables(pos, seq_pad, kv, dh, cfg.rope_theta)
    var cos_q = Tensor.from_host(q_tab[0], [seq_pad * heads * half], act_dt, ctx)
    var sin_q = Tensor.from_host(q_tab[1], [seq_pad * heads * half], act_dt, ctx)
    var cos_k = Tensor.from_host(k_tab[0], [seq_pad * kv * half], act_dt, ctx)
    var sin_k = Tensor.from_host(k_tab[1], [seq_pad * kv * half], act_dt, ctx)

    # causal mask (real columns only)
    var mask_data = _causal_mask_data(seq_pad, heads, L)
    var mask = Tensor.from_host(mask_data, [1, heads, seq_pad, seq_pad], act_dt, ctx)

    # deepstack features -> act dtype [1,nvis,H]
    var d0 = _to_rows(ds0, nvis, H, act_dt, ctx)
    var d1 = _to_rows(ds1, nvis, H, act_dt, ctx)
    var d2 = _to_rows(ds2, nvis, H, act_dt, ctx)

    # 3) decoder with deepstack add at layers 0/1/2
    var hidden = inputs_embeds^
    for i in range(cfg.num_layers):
        hidden = enc._layer(i, hidden, cos_q, sin_q, cos_k, sin_k, mask, ctx)
        if i == 0:
            hidden = _add_rows(hidden, d0, vis_start, ctx)
        elif i == 1:
            hidden = _add_rows(hidden, d1, vis_start, ctx)
        elif i == 2:
            hidden = _add_rows(hidden, d2, vis_start, ctx)

    # model.norm -> last_hidden_state, drop prefix + padding
    var normed = enc.final_norm(hidden, ctx)
    var keep = L - crop_start
    return slice(normed, 1, crop_start, keep, ctx)


def encode_mageflow_edit[S: Int](
    vision: Qwen3VLVisionModel,
    enc: Qwen3Encoder,
    ids: List[Int],
    grid_t: Int, grid_h: Int, grid_w: Int,
    pixel_values: Tensor,
    ctx: DeviceContext,
    f32_stream: Bool = False,
) raises -> Tensor:
    """Mage-Flow EDIT conditioning: run the Qwen3-VL vision tower on the
    reference image, then fuse its tokens into the text decoder (fuse_mageflow_edit)
    and drop the leading MAGEFLOW_EDIT_DROP_IDX (64) system-prompt rows. `S` is
    the vision patch seq (grid_t*grid_h*grid_w). `ids` are the EXACT token ids of
    template("mage-flow-edit").format("Image 1: <|vision_start|><|image_pad|>"
    "<|vision_end|>" + instruction) — image_pad already expanded to nvis by the
    Qwen3-VL processor. Returns [1, L-64, 2560] = the DiT context (res["txt"])."""
    var vout = vision.forward[S](pixel_values, grid_t, grid_h, grid_w, ctx)
    return fuse_mageflow_edit(
        enc, vout.pooler, vout.ds0, vout.ds1, vout.ds2,
        ids, MAGEFLOW_EDIT_DROP_IDX, ctx, f32_stream,
    )
