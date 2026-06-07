# sdxl_lora_resume_state_smoke.mojo -- SDXL LoRA resume/state contract.
#
# Tiny structural gate: build one synthetic SDXL SpatialTransformer LoRA set,
# seed non-zero A/B and AdamW moments, then exercise raw OneTrainer resume and
# trainer state sidecar round-trips. No UNet, model checkpoint, or sampler run.

from std.builtin.dtype import DType
from std.gpu.host import DeviceContext

from serenitymojo.models.sdxl.lora_block import (
    SLOT_A1_Q, SLOT_A2_K, SLOT_FF_PROJ, SLOT_FF_OUT,
)
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import (
    SdxlLoraSet,
    build_sdxl_lora_set,
    save_sdxl_lora,
    save_sdxl_lora_state,
    load_sdxl_lora_resume,
    load_sdxl_lora_state,
)


comptime ST_PREFIX = "input_blocks.4.1"
comptime LORA_OUT = "/tmp/sdxl_lora_resume_smoke.safetensors"
comptime STATE_OUT = "/tmp/sdxl_lora_state_smoke.safetensors"


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


def _seed_resume_values(mut lora: SdxlLoraSet):
    lora.ad[SLOT_A1_Q].b[0] = BFloat16(Float32(1.25))
    lora.ad[SLOT_A2_K].a[7] = BFloat16(Float32(0.75))
    lora.ad[SLOT_FF_OUT].b[3] = BFloat16(Float32(-0.5))
    lora.ad[SLOT_A1_Q].ma[0] = Float32(0.125)
    lora.ad[SLOT_A1_Q].va[0] = Float32(0.25)
    lora.ad[SLOT_FF_PROJ].mb[3] = Float32(-0.375)
    lora.ad[SLOT_A2_K].vb[7] = Float32(0.5)


def main() raises:
    var ctx = DeviceContext()
    var lora = build_sdxl_lora_set(1, 8, 4, 16, 2, Float32(4.0))
    _seed_resume_values(lora)

    var saved = save_sdxl_lora(lora, String(ST_PREFIX), String(LORA_OUT), ctx)
    _require(saved == 10, String("expected 10 saved SDXL adapters"))

    var resumed = load_sdxl_lora_resume(
        String(ST_PREFIX), 1, 2, Float32(4.0), 8, 4, 16, String(LORA_OUT), ctx
    )
    _require(len(resumed.ad) == 10, String("expected 10 resumed SDXL adapters"))
    _require(resumed.ad[SLOT_A1_Q].rank == 2, String("resumed rank mismatch"))
    _require(
        resumed.ad[SLOT_A2_K].in_f == 4 and resumed.ad[SLOT_A2_K].out_f == 8,
        String("resumed cross-attention key shape mismatch"),
    )
    _require(
        resumed.ad[SLOT_FF_OUT].in_f == 16 and resumed.ad[SLOT_FF_OUT].out_f == 8,
        String("resumed FF output adapter shape mismatch"),
    )
    _require_close(resumed.ad[SLOT_A1_Q].scale, Float32(2.0), String("raw resume scale mismatch"))
    _require_close(
        _bf16(resumed.ad[SLOT_A1_Q].b[0]),
        _bf16(lora.ad[SLOT_A1_Q].b[0]),
        String("raw resume B mismatch"),
    )
    _require_close(
        _bf16(resumed.ad[SLOT_A2_K].a[7]),
        _bf16(lora.ad[SLOT_A2_K].a[7]),
        String("raw resume A mismatch"),
    )
    _require_close(resumed.ad[SLOT_A1_Q].ma[0], Float32(0.0), String("raw resume should zero A adam_m"))
    _require_close(resumed.ad[SLOT_A2_K].vb[7], Float32(0.0), String("raw resume should zero B adam_v"))

    var state_saved = save_sdxl_lora_state(lora, String(ST_PREFIX), String(STATE_OUT), ctx)
    _require(state_saved == 10, String("expected 10 saved SDXL state adapters"))

    var state = load_sdxl_lora_state(
        String(ST_PREFIX), 1, 2, Float32(4.0), 8, 4, 16, String(STATE_OUT), ctx
    )
    _require(len(state.ad) == 10, String("expected 10 state SDXL adapters"))
    _require_close(state.ad[SLOT_A1_Q].scale, Float32(2.0), String("state resume scale mismatch"))
    _require_close(
        _bf16(state.ad[SLOT_A1_Q].b[0]),
        _bf16(lora.ad[SLOT_A1_Q].b[0]),
        String("state B mismatch"),
    )
    _require_close(
        _bf16(state.ad[SLOT_A2_K].a[7]),
        _bf16(lora.ad[SLOT_A2_K].a[7]),
        String("state A mismatch"),
    )
    _require_close(
        _bf16(state.ad[SLOT_FF_OUT].b[3]),
        _bf16(lora.ad[SLOT_FF_OUT].b[3]),
        String("state FF out B mismatch"),
    )
    _require_close(state.ad[SLOT_A1_Q].ma[0], Float32(0.125), String("state A adam_m mismatch"))
    _require_close(state.ad[SLOT_A1_Q].va[0], Float32(0.25), String("state A adam_v mismatch"))
    _require_close(state.ad[SLOT_FF_PROJ].mb[3], Float32(-0.375), String("state B adam_m mismatch"))
    _require_close(state.ad[SLOT_A2_K].vb[7], Float32(0.5), String("state B adam_v mismatch"))

    print("[sdxl-lora-resume-state] PASS:", LORA_OUT, STATE_OUT)
