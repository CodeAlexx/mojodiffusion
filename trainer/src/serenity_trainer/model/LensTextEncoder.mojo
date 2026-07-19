# LensTextEncoder.mojo — the Lens text encoder (GPT-OSS, MXFP4, YaRN RoPE),
# producing the multi-layer hidden states at selected_layer_index = [5,11,17,23].
#
# Serenity ref (LensModel.py encode_text + lens/text_encoder.py
# LensGptOssEncoder.encode_layers / forward): the encoder is a GptOssForCausalLM
# subclass that, given input_ids + attention_mask, runs the decoder layer loop and
# CAPTURES the post-residual hidden state (PRE-final-norm) at each selected layer,
# early-exiting after the last selected layer. encode_layers() is used INSTEAD of
# output_hidden_states=True because HF's @capture_outputs replaces hidden[-1] with
# the norm-applied output (LensModel.py comment :213-219). The captured pre-norm
# state at layer 23 (the last) is what Lens needs.
#
# BORROW boundary: the working GPT-OSS forward (streamed per-layer MXFP4 expert
# dequant, YaRN inv_freq + mscale, half-split RoPE, attention-sink SDPA, the layer
# loop with per-layer capture) is serenitymojo's
# models/text_encoder/gpt_oss_encoder.mojo. It is foundation-tier model code; we
# COPY it into the port by re-exporting its GptOssEncoder/GptOssConfig and adding
# the Lens-named seam (encode_layers -> 4 per-layer features). The forward math is
# unchanged; only the namespace/entrypoint is Lens-flavored.
#
# selected_layer_index = [5,11,17,23] (LensTransformer2DModel config; also
# gpt_oss_encoder.lens_extract_layers()). Capture order is ASCENDING (the encoder
# sorts+dedups), matching LensModel's txt_norm.{0..3} -> concat(dim=-1) order.
#
# DTYPE: BF16 hidden states out; MXFP4 experts dequant to transient BF16 on GPU;
# F32 only in the foundation kernel registers (no persistent F32).

from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# Borrow the working GPT-OSS encoder (serenitymojo foundation model code).
from serenitymojo.models.text_encoder.gpt_oss_encoder import (
    GptOssEncoder,
    GptOssConfig,
    lens_extract_layers,
)


comptime TArc = ArcPointer[Tensor]

# Lens conditioning capture layers, 0-indexed (LensTransformer2DModel
# selected_layer_index). Each captured feature is [1,S,enc_hidden_dim=2880] BF16.
comptime LENS_N_TEXT_LAYERS = 4


# ── LensTextEncoder ───────────────────────────────────────────────────────────
# Thin Lens-named wrapper over the borrowed GptOssEncoder. Holds the encoder + its
# config; encode_layers() runs the streamed layer loop and returns the 4 selected
# per-layer hidden states in ascending layer order.
struct LensTextEncoder(Movable):
    var encoder: GptOssEncoder
    var config: GptOssConfig

    def __init__(out self, var encoder: GptOssEncoder, config: GptOssConfig):
        self.encoder = encoder^
        self.config = config

    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> LensTextEncoder:
        # GptOssConfig.lens_default() carries the Lens GPT-OSS dims (24 layers,
        # selected [5,11,17,23], theta=150000, YaRN factor=32, MXFP4 experts).
        var cfg = GptOssConfig.lens_default()
        var enc = GptOssEncoder.load(dir, cfg, ctx)
        return LensTextEncoder(enc^, cfg)

    # ── encode_layers (lens/text_encoder.py:LensGptOssEncoder.encode_layers) ────
    # Source structure (text_encoder.py:124-137 -> forward :76-122):
    #   captured = run decoder loop, capture pre-norm hidden at each selected layer,
    #   early-exit after max(selected). Return list in selected order (ascending).
    # The forward COMBINES the attention_mask with the causal mask
    # (create_causal_mask(attention_mask=...), text_encoder.py:84-93) so padded
    # positions are masked INSIDE the encoder.
    #
    # FINDING 3 — mask contract. The borrowed GptOssEncoder.encode takes no padding
    # mask. GPT-OSS attention is CAUSAL: for a RIGHT-PADDED sequence the valid-prefix
    # outputs are bit-identical whether we feed (full padded ids + mask) or just the
    # valid prefix — a later padded token can never influence an earlier position,
    # and only the prefix features are consumed downstream (LensModel.py:212-216
    # crops the template, then prunes to max_seq_length). We therefore ENFORCE
    # right-padding (mask = 1s then 0s, NO interior 0 / NO left-padding) and feed only
    # the unpadded prefix. This is the faithful equivalent of text_encoder.py:84-93
    # for the right-pad tokenizer config (LensModel.py:191-200 padding='max_length',
    # add_special_tokens=True, i.e. right padding). Each captured feature is
    # [1, valid_len, 2880] BF16, in ascending selected-layer order.
    def encode_layers(
        self, token_ids: List[Int], tokens_mask: List[Int], ctx: DeviceContext
    ) raises -> List[TArc]:
        if len(tokens_mask) != len(token_ids):
            raise Error(
                "LensTextEncoder.encode_layers: tokens_mask length "
                "does not match token_ids length"
            )
        var valid = 0
        var seen_pad = False
        for i in range(len(tokens_mask)):
            if tokens_mask[i] != 0:
                if seen_pad:
                    raise Error(
                        "LensTextEncoder.encode_layers: interior or left padding "
                        "detected — Lens requires right-padded captions (mask is a "
                        "contiguous prefix of 1s)"
                    )
                valid += 1
            else:
                seen_pad = True
        var prefix_ids = List[Int]()
        for i in range(valid):
            prefix_ids.append(token_ids[i])
        var sel = lens_extract_layers()   # [5, 11, 17, 23]
        return self.encoder.encode(prefix_ids, sel, ctx)
