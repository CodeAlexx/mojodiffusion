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
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder
# Re-export the loader so callers get everything from one module. The loader is
# a byte-for-byte fit for Mage's `model.language_model.*` shard keys.
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import load_krea2_qwen3vl_4b


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
