# sdxl_lora_ot_save_key_smoke.mojo -- SDXL OneTrainer save-key contract.
#
# Tiny structural gate: build one synthetic SDXL SpatialTransformer LoRA block,
# save through save_sdxl_lora, then inspect the safetensors header and raw OT
# alpha/down/up keys. No UNet, model checkpoint, or sampler run.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import (
    build_sdxl_lora_set,
    load_sdxl_lora_resume,
    save_sdxl_lora,
    sdxl_lora_supported_unet_prefixes,
    sdxl_lora_unsupported_onetrainer_targets,
)
from serenitymojo.training.lora_save import _read_f32


comptime OUT = "/tmp/sdxl_lora_ot_save_key_smoke.safetensors"


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _require_rank2(st: SafeTensors, key: String, dim0: Int, dim1: Int) raises:
    _require(key in st.tensors, String("missing key ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BF16, String("expected BF16 for ") + key)
    _require(len(info.shape) == 2, String("expected rank-2 tensor for ") + key)
    _require(
        info.shape[0] == dim0 and info.shape[1] == dim1,
        String("shape mismatch for ") + key,
    )


def _require_alpha(
    st: SafeTensors, key: String, expected: Float32, ctx: DeviceContext
) raises:
    _require(key in st.tensors, String("missing key ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BF16, String("expected BF16 alpha for ") + key)
    _require(len(info.shape) == 0, String("expected scalar alpha for ") + key)
    var alpha = _read_f32(st, key, ctx)
    _require(len(alpha) == 1, String("expected one alpha value for ") + key)
    var err = alpha[0] - expected
    if err < Float32(0.0):
        err = -err
    _require(err == Float32(0.0), String("alpha value mismatch for ") + key)


def main() raises:
    var ctx = DeviceContext()
    var lora = build_sdxl_lora_set(1, 8, 4, 16, 2, Float32(4.0))
    var supported = sdxl_lora_supported_unet_prefixes(String("input_blocks.4.1"), 1)
    _require(len(supported) == 10, String("expected 10 implemented ST linear OT prefixes"))
    _require(
        supported[0]
        == String("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q"),
        String("supported OT prefix mapping mismatch"),
    )
    var unsupported = sdxl_lora_unsupported_onetrainer_targets()
    _require(len(unsupported) >= 15, String("expected explicit unsupported OT targets"))
    _require(unsupported[0] == String("lora_unet_conv_in"), String("missing conv_in blocker"))
    _require(
        unsupported[3] == String("lora_unet_add_embedding_linear_1"),
        String("missing add_embedding blocker"),
    )
    _require(unsupported[len(unsupported) - 2] == String("lora_te1"), String("missing TE1 blocker"))
    _require(unsupported[len(unsupported) - 1] == String("lora_te2"), String("missing TE2 blocker"))

    var saved = save_sdxl_lora(lora, String("input_blocks.4.1"), String(OUT), ctx)
    _require(saved == 10, String("expected 10 saved SDXL adapters"))

    var st = SafeTensors.open(String(OUT))
    _require(len(st.tensors) == 30, String("expected 30 tensors: 10 modules x alpha/down/up"))

    var q = String("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q")
    _require_alpha(st, q + String(".alpha"), Float32(4.0), ctx)
    _require_rank2(st, q + String(".lora_down.weight"), 2, 8)
    _require_rank2(st, q + String(".lora_up.weight"), 8, 2)

    var cross_k = String("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k")
    _require_rank2(st, cross_k + String(".lora_down.weight"), 2, 4)
    _require_rank2(st, cross_k + String(".lora_up.weight"), 8, 2)

    var ff = String("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_2")
    _require_rank2(st, ff + String(".lora_down.weight"), 2, 16)
    _require_rank2(st, ff + String(".lora_up.weight"), 8, 2)

    var resumed = load_sdxl_lora_resume(
        String("input_blocks.4.1"), 1, 2, Float32(4.0), 8, 4, 16, String(OUT), ctx
    )
    _require(len(resumed.ad) == 10, String("expected 10 resumed adapters"))
    _require(resumed.ad[0].rank == 2, String("resumed rank mismatch"))
    _require(resumed.ad[5].in_f == 4, String("resumed cross-attention key input dim mismatch"))
    _require(resumed.ad[9].in_f == 16, String("resumed FF output adapter input dim mismatch"))

    print("[sdxl-lora-ot-save-key] PASS:", OUT)
