# chroma_lora_ot_save_key_smoke.mojo -- Chroma OneTrainer save-key contract.
#
# Tiny structural gate: build one synthetic double block and one single block
# worth of Chroma block LoRA adapters, save through save_chroma_lora, inspect
# raw OneTrainer alpha/down/up keys, then reload with load_chroma_lora_resume.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.flux.flux_stack_lora import build_flux_lora_set
from serenitymojo.models.chroma.chroma_stack_lora import (
    load_chroma_lora_resume,
    load_chroma_lora_resume_for_layer_filter,
    save_chroma_lora,
    save_chroma_lora_for_layer_filter,
    chroma_lora_prefixes_for_layer_filter,
)
from serenitymojo.training.lora_save import _read_f32


comptime OUT = "/tmp/chroma_lora_ot_save_key_smoke.safetensors"
comptime FILTER_OUT = "/tmp/chroma_lora_ot_save_key_filter_smoke.safetensors"
comptime BASELINE_FILTER = "attn,ff.net"


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
    var lora = build_flux_lora_set(1, 1, 8, 16, 2, Float32(4.0))
    var saved = save_chroma_lora(lora, String(OUT), ctx)
    _require(saved == 17, String("expected 17 saved Chroma adapters"))

    var st = SafeTensors.open(String(OUT))
    _require(len(st.tensors) == 51, String("expected 51 tensors: 17 modules x alpha/down/up"))

    var q = String("lora_transformer_transformer_blocks_0_attn_to_q")
    _require_alpha(st, q + String(".alpha"), Float32(4.0), ctx)
    _require_rank2(st, q + String(".lora_down.weight"), 2, 8)
    _require_rank2(st, q + String(".lora_up.weight"), 8, 2)

    var ctx_ff = String("lora_transformer_transformer_blocks_0_ff_context_net_2")
    _require_rank2(st, ctx_ff + String(".lora_down.weight"), 2, 16)
    _require_rank2(st, ctx_ff + String(".lora_up.weight"), 8, 2)

    var s_out = String("lora_transformer_single_transformer_blocks_0_proj_out")
    _require_rank2(st, s_out + String(".lora_down.weight"), 2, 24)
    _require_rank2(st, s_out + String(".lora_up.weight"), 8, 2)

    var resumed = load_chroma_lora_resume(1, 1, 2, Float32(4.0), String(OUT), ctx)
    _require(len(resumed.ad) == 17, String("expected 17 resumed adapters"))
    _require(resumed.ad[0].rank == 2, String("resumed rank mismatch"))
    _require(resumed.ad[16].in_f == 24, String("resumed single output adapter input dim mismatch"))
    _require(resumed.ad[16].out_f == 8, String("resumed single output adapter output dim mismatch"))

    var baseline_prefixes = chroma_lora_prefixes_for_layer_filter(
        19, 38, String(BASELINE_FILTER)
    )
    _require(
        len(baseline_prefixes) == 304,
        String("expected 304 Chroma baseline layer-filter prefixes"),
    )

    var filter_lora = build_flux_lora_set(19, 38, 8, 16, 2, Float32(4.0))
    var filter_saved = save_chroma_lora_for_layer_filter(
        filter_lora, String(BASELINE_FILTER), String(FILTER_OUT), ctx
    )
    _require(filter_saved == 304, String("expected 304 filtered saved Chroma adapters"))

    var filter_st = SafeTensors.open(String(FILTER_OUT))
    _require(len(filter_st.tensors) == 912, String("expected 912 filtered tensors"))
    _require_rank2(
        filter_st,
        String("lora_transformer_transformer_blocks_0_ff_net_2.lora_down.weight"),
        2,
        16,
    )
    _require(
        String("lora_transformer_transformer_blocks_0_ff_context_net_2.lora_down.weight")
            not in filter_st.tensors,
        String("ff_context.net must not match layer_filter=attn,ff.net"),
    )
    _require(
        String("lora_transformer_single_transformer_blocks_0_proj_mlp.lora_down.weight")
            not in filter_st.tensors,
        String("single proj_mlp must not match layer_filter=attn,ff.net"),
    )

    var filter_resumed = load_chroma_lora_resume_for_layer_filter(
        19, 38, 2, Float32(4.0), String(BASELINE_FILTER), String(FILTER_OUT), ctx
    )
    _require(len(filter_resumed.ad) == 304, String("expected 304 filtered resumed adapters"))
    _require(filter_resumed.ad[0].rank == 2, String("filtered resumed rank mismatch"))
    _require(
        filter_resumed.ad[5].in_f == 16 and filter_resumed.ad[5].out_f == 8,
        String("filtered double ff.net.2 dimensions mismatch"),
    )

    print("[chroma-lora-ot-save-key] PASS:", OUT)
