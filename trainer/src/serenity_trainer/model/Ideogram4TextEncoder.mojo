# Ideogram4TextEncoder.mojo — Ideogram-4 Qwen3-VL-8B 13-tap text encoder.
#
# ════════════════════════════════════════════════════════════════════════════
# PORT SPEC — the EXACT ai-toolkit code path reproduced.
# ════════════════════════════════════════════════════════════════════════════
# ai-toolkit ideogram4.py: the text encoder is the frozen, stock
#   Qwen/Qwen3-VL-8B-Instruct language model. Conditioning = the "13-tap"
#   interleaved concat of selected decoder hidden states
#   (QWEN3_VL_ACTIVATION_LAYERS = [0,3,6,9,12,15,18,21,24,27,30,33,35]):
#     stack(taps, 0).permute(1,2,3,0).reshape(1, L, 4096*13 = 53248)
#   (ai-toolkit pipeline _encode_text; reference dump in
#    serenitymojo ideogram4_oracle.py stage_B / stage_E).
#   The stored weights are weight-only FP8 (e4m3 + per-row .weight_scale,
#   keys language_model.*) — dequantized to BF16 at load.
#
# BORROW BOUNDARY: the verified compute lives in serenitymojo
#   (models/text_encoder/ideogram_qwen3vl.mojo, reusing Qwen3Encoder; gated vs
#   torch — chunk7 13-tap cos 0.9999863, 2026-06-07,
#   serenitymojo/models/dit/parity/chunk7_qwen_probe.mojo). This file is the
#   serenity-side seam the dataLoader's "EncodeQwen3VLText" step calls.
#
# NOTE: tokenization (caption JSON -> token ids via the Qwen3-VL chat template)
#   is a SEPARATE concern handled by the dataLoader; this encoder takes token ids
#   and returns the 13-tap conditioning, exactly as the gate does.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder
from serenitymojo.models.text_encoder.ideogram_qwen3vl import (
    load_ideogram_qwen3vl,
    encode_ideogram_taps,
)

comptime IDEOGRAM4_TEXT_FEATURE_DIM = 53248   # 4096 * 13 taps
comptime IDEOGRAM4_QWEN3_VL_HIDDEN = 4096
comptime IDEOGRAM4_TEXT_TAP_COUNT = 13
comptime IDEOGRAM4_TEXT_ENCODER_DEFAULT_PATH = "/home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors"


# Free-function seam: Qwen3Encoder is non-movable (holds device resources), so
# the encoder is held as a local by the caller (exactly as the serenitymojo gate
# does) rather than stored in a struct field.
def ideogram4_load_text_encoder(dir_or_file: String, ctx: DeviceContext) raises -> Qwen3Encoder:
    return load_ideogram_qwen3vl(dir_or_file, ctx)


def ideogram4_load_text_encoder_default(ctx: DeviceContext) raises -> Qwen3Encoder:
    return load_ideogram_qwen3vl(String(IDEOGRAM4_TEXT_ENCODER_DEFAULT_PATH), ctx)


# token ids -> 13-tap interleaved conditioning [1, L, 53248] (BF16).
def ideogram4_encode_text(enc: Qwen3Encoder, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
    return encode_ideogram_taps(enc, ids, ctx)
