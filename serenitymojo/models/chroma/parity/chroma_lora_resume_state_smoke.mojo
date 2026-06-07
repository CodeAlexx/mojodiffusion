# chroma_lora_resume_state_smoke.mojo -- Chroma LoRA resume/state contract.
#
# Tiny structural gate: build one synthetic Chroma block LoRA set, seed nonzero
# A/B weights and AdamW moments, then exercise raw OneTrainer resume and trainer
# state sidecar round-trips. No model weights or denoise run.

from std.builtin.dtype import DType
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet,
    build_flux_lora_set,
)
from serenitymojo.models.chroma.chroma_stack_lora import (
    save_chroma_lora,
    save_chroma_lora_for_layer_filter,
    save_chroma_lora_state,
    save_chroma_lora_state_for_layer_filter,
    load_chroma_lora_resume,
    load_chroma_lora_resume_for_layer_filter,
    load_chroma_lora_state,
    load_chroma_lora_state_for_layer_filter,
    chroma_lora_prefixes_for_layer_filter,
)


comptime LORA_OUT = "/tmp/chroma_lora_resume_smoke.safetensors"
comptime STATE_OUT = "/tmp/chroma_lora_state_smoke.safetensors"
comptime FILTER_LORA_OUT = "/tmp/chroma_lora_resume_filter_smoke.safetensors"
comptime FILTER_STATE_OUT = "/tmp/chroma_lora_state_filter_smoke.safetensors"
comptime BASELINE_FILTER = "attn,ff.net"


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _abs(v: Float32) -> Float32:
    if v < Float32(0.0):
        return -v
    return v


def _bf16(v: BFloat16) -> Float32:
    return v.cast[DType.float32]()


def _require_close(got: Float32, expected: Float32, msg: String) raises:
    _require(_abs(got - expected) <= Float32(0.0001), msg)


def _require_key(st: SafeTensors, key: String, dtype: STDtype) raises:
    _require(key in st.tensors, String("missing key ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == dtype, String("dtype mismatch for ") + key)


def _seed_resume_values(mut lora: FluxLoraSet):
    lora.ad[0].a[0] = BFloat16(Float32(1.25))
    lora.ad[0].b[0] = BFloat16(Float32(-0.5))
    lora.ad[11].b[3] = BFloat16(Float32(0.75))
    lora.ad[16].a[7] = BFloat16(Float32(-1.5))

    lora.ad[0].ma[0] = Float32(0.125)
    lora.ad[0].va[0] = Float32(0.25)
    lora.ad[11].mb[3] = Float32(-0.375)
    lora.ad[16].vb[7] = Float32(0.5)


def main() raises:
    var ctx = DeviceContext()
    var lora = build_flux_lora_set(1, 1, 8, 16, 2, Float32(4.0))
    _seed_resume_values(lora)

    var saved = save_chroma_lora(lora, String(LORA_OUT), ctx)
    _require(saved == 17, String("expected 17 saved Chroma adapters"))

    var raw_st = SafeTensors.open(String(LORA_OUT))
    _require(len(raw_st.tensors) == 51, String("expected 51 raw tensors"))
    _require_key(
        raw_st,
        String("lora_transformer_transformer_blocks_0_attn_to_q.lora_down.weight"),
        STDtype.BF16,
    )
    _require_key(
        raw_st,
        String("lora_transformer_single_transformer_blocks_0_proj_out.lora_up.weight"),
        STDtype.BF16,
    )

    var resumed = load_chroma_lora_resume(1, 1, 2, Float32(4.0), String(LORA_OUT), ctx)
    _require(len(resumed.ad) == 17, String("expected 17 resumed Chroma adapters"))
    _require(resumed.ad[0].rank == 2, String("resumed rank mismatch"))
    _require(resumed.ad[16].in_f == 24, String("resumed single output input dim mismatch"))
    _require(resumed.ad[16].out_f == 8, String("resumed single output output dim mismatch"))
    _require_close(resumed.ad[0].scale, Float32(2.0), String("raw resume scale mismatch"))
    _require_close(_bf16(resumed.ad[0].a[0]), _bf16(lora.ad[0].a[0]), String("raw resume A mismatch"))
    _require_close(_bf16(resumed.ad[0].b[0]), _bf16(lora.ad[0].b[0]), String("raw resume B mismatch"))
    _require_close(_bf16(resumed.ad[11].b[3]), _bf16(lora.ad[11].b[3]), String("raw resume context B mismatch"))
    _require_close(_bf16(resumed.ad[16].a[7]), _bf16(lora.ad[16].a[7]), String("raw resume single A mismatch"))
    _require_close(resumed.ad[0].ma[0], Float32(0.0), String("raw resume should zero A adam_m"))
    _require_close(resumed.ad[0].va[0], Float32(0.0), String("raw resume should zero A adam_v"))
    _require_close(resumed.ad[11].mb[3], Float32(0.0), String("raw resume should zero B adam_m"))
    _require_close(resumed.ad[16].vb[7], Float32(0.0), String("raw resume should zero B adam_v"))

    var state_saved = save_chroma_lora_state(lora, String(STATE_OUT), ctx)
    _require(state_saved == 17, String("expected 17 saved Chroma state adapters"))

    var state_st = SafeTensors.open(String(STATE_OUT))
    _require(len(state_st.tensors) == 102, String("expected 102 state tensors"))
    _require_key(
        state_st,
        String("lora_transformer_transformer_blocks_0_attn_to_q.lora_A.adam_m"),
        STDtype.F32,
    )
    _require_key(
        state_st,
        String("lora_transformer_single_transformer_blocks_0_proj_out.lora_B.adam_v"),
        STDtype.F32,
    )

    var state = load_chroma_lora_state(1, 1, 2, Float32(4.0), String(STATE_OUT), ctx)
    _require(len(state.ad) == 17, String("expected 17 state Chroma adapters"))
    _require_close(state.ad[0].scale, Float32(2.0), String("state resume scale mismatch"))
    _require_close(_bf16(state.ad[0].a[0]), _bf16(lora.ad[0].a[0]), String("state A mismatch"))
    _require_close(_bf16(state.ad[0].b[0]), _bf16(lora.ad[0].b[0]), String("state B mismatch"))
    _require_close(_bf16(state.ad[11].b[3]), _bf16(lora.ad[11].b[3]), String("state context B mismatch"))
    _require_close(_bf16(state.ad[16].a[7]), _bf16(lora.ad[16].a[7]), String("state single A mismatch"))
    _require_close(state.ad[0].ma[0], Float32(0.125), String("state A adam_m mismatch"))
    _require_close(state.ad[0].va[0], Float32(0.25), String("state A adam_v mismatch"))
    _require_close(state.ad[11].mb[3], Float32(-0.375), String("state B adam_m mismatch"))
    _require_close(state.ad[16].vb[7], Float32(0.5), String("state B adam_v mismatch"))

    var baseline_prefixes = chroma_lora_prefixes_for_layer_filter(
        19, 38, String(BASELINE_FILTER)
    )
    _require(len(baseline_prefixes) == 304, String("expected 304 baseline layer-filter prefixes"))

    var filter_lora = build_flux_lora_set(19, 38, 8, 16, 2, Float32(4.0))
    filter_lora.ad[0].a[0] = BFloat16(Float32(0.625))
    filter_lora.ad[0].ma[0] = Float32(0.0625)
    filter_lora.ad[9].b[3] = BFloat16(Float32(-0.875))
    filter_lora.ad[9].mb[3] = Float32(-0.1875)

    var filter_saved = save_chroma_lora_for_layer_filter(
        filter_lora, String(BASELINE_FILTER), String(FILTER_LORA_OUT), ctx
    )
    _require(filter_saved == 304, String("expected 304 filtered raw adapters"))

    var filter_raw = load_chroma_lora_resume_for_layer_filter(
        19, 38, 2, Float32(4.0), String(BASELINE_FILTER), String(FILTER_LORA_OUT), ctx
    )
    _require(len(filter_raw.ad) == 304, String("expected 304 filtered raw resumed adapters"))
    _require_close(_bf16(filter_raw.ad[0].a[0]), Float32(0.625), String("filtered raw A mismatch"))
    _require_close(_bf16(filter_raw.ad[9].b[3]), Float32(-0.875), String("filtered raw B mismatch"))
    _require_close(filter_raw.ad[0].ma[0], Float32(0.0), String("filtered raw A adam_m should be zero"))
    _require_close(filter_raw.ad[9].mb[3], Float32(0.0), String("filtered raw B adam_m should be zero"))

    var filter_state_saved = save_chroma_lora_state_for_layer_filter(
        filter_lora, String(BASELINE_FILTER), String(FILTER_STATE_OUT), ctx
    )
    _require(filter_state_saved == 304, String("expected 304 filtered state adapters"))

    var filter_state = load_chroma_lora_state_for_layer_filter(
        19, 38, 2, Float32(4.0), String(BASELINE_FILTER), String(FILTER_STATE_OUT), ctx
    )
    _require(len(filter_state.ad) == 304, String("expected 304 filtered state resumed adapters"))
    _require_close(_bf16(filter_state.ad[0].a[0]), Float32(0.625), String("filtered state A mismatch"))
    _require_close(_bf16(filter_state.ad[9].b[3]), Float32(-0.875), String("filtered state B mismatch"))
    _require_close(filter_state.ad[0].ma[0], Float32(0.0625), String("filtered state A adam_m mismatch"))
    _require_close(filter_state.ad[9].mb[3], Float32(-0.1875), String("filtered state B adam_m mismatch"))

    print("[chroma-lora-resume-state] PASS:", LORA_OUT, STATE_OUT)
