# flux_lora_ot_save_key_smoke.mojo -- Flux OneTrainer save-key contract.
#
# Tiny structural gate: build one synthetic Flux double block and one single
# block worth of LoRA adapters, save through save_flux_lora, then inspect the
# safetensors header and raw OT alpha/down/up keys. No model weights or denoise.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.flux.flux_stack_lora import (
    build_flux_lora_set,
    flux_lora_missing_ot_transformer_prefixes,
    flux_lora_ot_transformer_prefixes,
    flux_lora_prefixes,
    load_flux_lora_resume,
    require_flux_lora_ot_transformer_complete,
    require_flux_lora_text_encoder_disabled,
    save_flux_lora,
)
from serenitymojo.training.lora_save import _read_f32


comptime OUT = "/tmp/flux_lora_ot_save_key_smoke.safetensors"


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


def _contains(values: List[String], expected: String) -> Bool:
    for i in range(len(values)):
        if values[i] == expected:
            return True
    return False


def _require_prefix(values: List[String], idx: Int, expected: String, label: String) raises:
    _require(idx < len(values), label + String(" missing index ") + String(idx))
    _require(
        values[idx] == expected,
        label + String(" prefix[") + String(idx) + String("] mismatch: ")
        + values[idx] + String(" != ") + expected,
    )


def _require_contains(values: List[String], expected: String, label: String) raises:
    _require(_contains(values, expected), label + String(" missing ") + expected)


def _require_not_contains(values: List[String], expected: String, label: String) raises:
    _require(not _contains(values, expected), label + String(" unexpectedly contains ") + expected)


def _require_exact_supported_prefixes(values: List[String]) raises:
    _require_prefix(values, 0, "lora_transformer_transformer_blocks_0_attn_to_q", "supported")
    _require_prefix(values, 1, "lora_transformer_transformer_blocks_0_attn_to_k", "supported")
    _require_prefix(values, 2, "lora_transformer_transformer_blocks_0_attn_to_v", "supported")
    _require_prefix(values, 3, "lora_transformer_transformer_blocks_0_attn_to_out_0", "supported")
    _require_prefix(values, 4, "lora_transformer_transformer_blocks_0_ff_net_0_proj", "supported")
    _require_prefix(values, 5, "lora_transformer_transformer_blocks_0_ff_net_2", "supported")
    _require_prefix(values, 6, "lora_transformer_transformer_blocks_0_attn_add_q_proj", "supported")
    _require_prefix(values, 7, "lora_transformer_transformer_blocks_0_attn_add_k_proj", "supported")
    _require_prefix(values, 8, "lora_transformer_transformer_blocks_0_attn_add_v_proj", "supported")
    _require_prefix(values, 9, "lora_transformer_transformer_blocks_0_attn_to_add_out", "supported")
    _require_prefix(values, 10, "lora_transformer_transformer_blocks_0_ff_context_net_0_proj", "supported")
    _require_prefix(values, 11, "lora_transformer_transformer_blocks_0_ff_context_net_2", "supported")
    _require_prefix(values, 12, "lora_transformer_single_transformer_blocks_0_attn_to_q", "supported")
    _require_prefix(values, 13, "lora_transformer_single_transformer_blocks_0_attn_to_k", "supported")
    _require_prefix(values, 14, "lora_transformer_single_transformer_blocks_0_attn_to_v", "supported")
    _require_prefix(values, 15, "lora_transformer_single_transformer_blocks_0_proj_mlp", "supported")
    _require_prefix(values, 16, "lora_transformer_single_transformer_blocks_0_proj_out", "supported")


def _require_exact_missing_prefixes(values: List[String]) raises:
    _require_prefix(values, 0, "lora_transformer_context_embedder", "missing")
    _require_prefix(values, 1, "lora_transformer_norm_out_linear", "missing")
    _require_prefix(values, 2, "lora_transformer_proj_out", "missing")
    _require_prefix(
        values,
        3,
        "lora_transformer_time_text_embed_guidance_embedder_linear_1",
        "missing",
    )
    _require_prefix(
        values,
        4,
        "lora_transformer_time_text_embed_guidance_embedder_linear_2",
        "missing",
    )
    _require_prefix(
        values,
        5,
        "lora_transformer_time_text_embed_text_embedder_linear_1",
        "missing",
    )
    _require_prefix(
        values,
        6,
        "lora_transformer_time_text_embed_text_embedder_linear_2",
        "missing",
    )
    _require_prefix(
        values,
        7,
        "lora_transformer_time_text_embed_timestep_embedder_linear_1",
        "missing",
    )
    _require_prefix(
        values,
        8,
        "lora_transformer_time_text_embed_timestep_embedder_linear_2",
        "missing",
    )
    _require_prefix(values, 9, "lora_transformer_x_embedder", "missing")
    _require_prefix(values, 10, "lora_transformer_transformer_blocks_0_norm1_linear", "missing")
    _require_prefix(
        values,
        11,
        "lora_transformer_transformer_blocks_0_norm1_context_linear",
        "missing",
    )
    _require_prefix(values, 12, "lora_transformer_single_transformer_blocks_0_norm_linear", "missing")


def _require_raises_transformer_complete() raises:
    var raised = False
    try:
        require_flux_lora_ot_transformer_complete(1, 1)
    except:
        raised = True
    _require(raised, String("expected incomplete Flux OT transformer surface to raise"))


def _require_raises_te() raises:
    var raised_te1 = False
    try:
        require_flux_lora_text_encoder_disabled(True, False)
    except:
        raised_te1 = True
    _require(raised_te1, String("expected lora_te1 enabled config to raise"))

    var raised_te2 = False
    try:
        require_flux_lora_text_encoder_disabled(False, True)
    except:
        raised_te2 = True
    _require(raised_te2, String("expected lora_te2 enabled config to raise"))

    require_flux_lora_text_encoder_disabled(False, False)


def main() raises:
    var ctx = DeviceContext()
    var lora = build_flux_lora_set(1, 1, 8, 16, 2, Float32(4.0))
    var supported = flux_lora_prefixes(1, 1)
    _require(len(supported) == 17, String("expected 17 supported trained Flux prefixes"))
    _require_exact_supported_prefixes(supported)
    var ot_full = flux_lora_ot_transformer_prefixes(1, 1)
    _require(len(ot_full) == 30, String("expected 30 OT transformer prefixes for 1 double + 1 single"))
    var missing = flux_lora_missing_ot_transformer_prefixes(1, 1)
    _require(len(missing) == 13, String("expected 13 fail-loud missing Flux OT prefixes"))
    _require_exact_missing_prefixes(missing)
    for i in range(len(supported)):
        _require_contains(ot_full, supported[i], "full OT transformer inventory")
        _require_not_contains(missing, supported[i], "fail-loud missing inventory")
    for i in range(len(missing)):
        _require_contains(ot_full, missing[i], "full OT transformer inventory")
    _require_raises_transformer_complete()
    _require_raises_te()

    var saved = save_flux_lora(lora, String(OUT), ctx)
    _require(saved == 17, String("expected 17 saved Flux adapters"))

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

    var resumed = load_flux_lora_resume(1, 1, 2, Float32(4.0), String(OUT), ctx)
    _require(len(resumed.ad) == 17, String("expected 17 resumed adapters"))
    _require(resumed.ad[0].rank == 2, String("resumed rank mismatch"))
    _require(resumed.ad[16].in_f == 24, String("resumed single output adapter input dim mismatch"))
    _require(resumed.ad[16].out_f == 8, String("resumed single output adapter output dim mismatch"))

    print("[flux-lora-ot-save-key] PASS:", OUT)
