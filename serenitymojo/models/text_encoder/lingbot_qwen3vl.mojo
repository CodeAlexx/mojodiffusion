# models/text_encoder/lingbot_qwen3vl.mojo — LingBot-Video Qwen3-VL-4B TEXT path.
#
# LingBot-Video's text encoder is the Qwen3-VL-4B-Instruct text tower — the SAME
# model Krea-2 uses (config BYTE-IDENTICAL: hidden 2560, 36 layers, 32 heads, 8
# kv, head_dim 128, eps 1e-6, rope_theta 5e6, bf16). So the loader is Krea-2's
# `load_krea2_qwen3vl_4b` VERBATIM (it takes a generic `te_dir`, streams the
# 2-shard snapshot through one reusable pinned host buffer, remaps
# `model.language_model.*` -> `model.*`, skips the vision tower + tied lm_head,
# returns a Qwen3Encoder). We do NOT modify krea2's file — just delegate.
#
# The ONLY behavioral delta vs Krea-2 is the CONDITIONING TAP:
#   * Krea-2 (encode_krea2_stack): a STACK of 12 selected PRE-final-norm layers,
#     drop 34 system-prefix rows -> [1, L', 12, 2560].
#   * LingBot (encode_lingbot_text): HIDDEN_STATE_SKIP_LAYER=0 = the LAST hidden
#     state POST model.norm (HF hidden_states[-1]), crop the system prefix ->
#     [1, L', 2560].
#
# HF layer map (verified against krea2 header + Boogu C7 gate): HF
# all_hidden_states = (emb, layer0_out, ..., layer34_out, model.norm(layer35_out))
# so hidden_states[-1] = model.norm(layer35_out). Qwen3Encoder.encode_layer_states
# returns [layer0_out, ..., layer35_out] (index i = layer i output, NO embedding
# entry, PRE-final-norm), so the Mojo tap is:
#   states[num_layers-1] = states[35] = layer35_out (pre-norm)
#   enc.final_norm(states[35])         = model.norm(layer35_out) = HF[-1]  ✓
#   slice L axis [crop_start : true_len]  (drop system prefix + SDPA padding)
#
# mROPE note: text_config.rope_scaling is mrope_interleaved [24,20,20], but for
# TEXT-ONLY inputs the three mrope sections share one 1D position sequence and
# collapse to plain half-split RoPE — the reduction Qwen3Encoder implements and
# the same one the Boogu/krea2/ideogram Qwen3-VL gates validated at cos>=0.999.
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import load_krea2_qwen3vl_4b


# LingBot-Video Qwen3-VL text_encoder snapshot dir (2-shard bf16 + index.json).
comptime LINGBOT_TE_DIR = "/mnt/disk1/models/lingbot-video-moe/text_encoder"

# Qwen3 pad/bos id (== 151643). LingBot prompt tokens use the chat-template
# markers (151644/151645) and never 151643 in the real span, so padding with
# 151643 lets encode_layer_states' pad auto-detect recover real_len. (Verified by
# oracle_text_ids.py: `151643 not in input_ids`.)
comptime _LINGBOT_PAD_ID = 151643


def load_lingbot_qwen3vl(te_dir: String, ctx: DeviceContext) raises -> Qwen3Encoder:
    """Load the LingBot-Video Qwen3-VL-4B text tower into a Qwen3Encoder.

    Byte-identical to Krea-2's Qwen3-VL text encoder, so this delegates to
    `load_krea2_qwen3vl_4b` (generic in `te_dir`) without change. Point `te_dir`
    at `LINGBOT_TE_DIR` (the LingBot snapshot's text_encoder/)."""
    return load_krea2_qwen3vl_4b(te_dir, ctx)


def _pad_ids(ids: List[Int]) raises -> List[Int]:
    """Pad the token list UP to the smallest comptime-supported SDPA seq
    (64/128/256/512/1024/2048) with _LINGBOT_PAD_ID, so encode_layer_states' pad
    auto-detect (first 151643) recovers real_len = the original L. The encoder is
    causal so pad columns are masked out — numerically identical to the unpadded
    L. (The apple prompt is ~500 tokens -> pads to 512.)"""
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
            String("lingbot encode: L=") + String(L)
            + " exceeds 2048 (max supported encoder SDPA seq)"
        )
    var out = List[Int]()
    for i in range(L):
        out.append(ids[i])
    for _i in range(pad - L):
        out.append(_LINGBOT_PAD_ID)
    return out^


def encode_lingbot_text(
    enc: Qwen3Encoder, ids: List[Int], crop_start: Int, ctx: DeviceContext
) raises -> Tensor:
    """LingBot conditioning: the POST-model.norm last hidden state with the
    leading `crop_start` system-prefix rows dropped -> [1, L-crop_start, 2560].

    `ids` are the EXACT full (pre-crop) token ids the torch Qwen3-VL saw
    (system-prefix + prompt), i.e. the full `input_ids` from the processor.
    `crop_start` is the system-prefix length computed via the
    `<|USER_INPUT_MARKER|>` trick (pipeline.encode_prompt). The tap:
      - encode_layer_states -> all 36 post-layer states (PRE-final-norm)
      - states[num_layers-1] = layer35_out
      - final_norm(states[35]) = model.norm(layer35_out) = HF hidden_states[-1]
      - slice L axis [crop_start : true_len]  (drop system prefix + SDPA padding)
    """
    var L = len(ids)
    if crop_start < 0 or crop_start >= L:
        raise Error(
            String("lingbot encode: crop_start=") + String(crop_start)
            + " out of range for L=" + String(L)
        )
    var padded = _pad_ids(ids)
    # All 36 layer states, each [1, L_pad, 2560], PRE-final-norm.
    var states = enc.encode_layer_states(padded, ctx)
    var last_idx = enc.config.num_layers - 1
    # model.norm(layer35_out) -> post-norm last_hidden_state [1, L_pad, 2560].
    var normed = enc.final_norm(states[last_idx][], ctx)
    # Drop the system prefix AND the SDPA padding in one narrow: [crop_start, L).
    var keep = L - crop_start
    return slice(normed, 1, crop_start, keep, ctx)
