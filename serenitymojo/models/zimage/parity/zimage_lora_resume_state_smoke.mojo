# zimage_lora_resume_state_smoke.mojo -- Z-Image LoRA resume/state contract.
#
# Tiny structural gate: build a synthetic Z-Image LoRA set with refiner + main
# blocks, seed non-zero main-layer A/B weights and AdamW moments, then exercise
# main-only raw resume and trainer state sidecar round-trips. No transformer
# checkpoint, cache, sampler, or oracle tensors.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_lora_resume_state_smoke.mojo

from std.builtin.dtype import DType
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.zimage.lora_block import (
    SLOT_Q, SLOT_W1, SLOT_W2, ZIMAGE_SLOTS,
)
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraSet,
    build_zimage_lora_set,
    save_zimage_lora_main_only,
    save_zimage_lora_main_only_state,
    load_zimage_lora_main_only_resume,
    load_zimage_lora_main_only_state,
)


comptime NUM_NR = 1
comptime NUM_CR = 1
comptime NUM_MAIN = 2
comptime D = 8
comptime F = 16
comptime RANK = 2
comptime ALPHA = Float32(4.0)
comptime MAIN_START = (NUM_NR + NUM_CR) * ZIMAGE_SLOTS
comptime LORA_OUT = "/tmp/zimage_lora_resume_state_smoke.safetensors"
comptime STATE_OUT = "/tmp/zimage_lora_train_state_smoke.safetensors"


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


def _seed_resume_values(mut lora: ZImageLoraSet):
    var main0_q = MAIN_START + SLOT_Q
    var main0_w1 = MAIN_START + SLOT_W1
    var main1_w2 = MAIN_START + ZIMAGE_SLOTS + SLOT_W2

    lora.ad[main0_q].a[0] = BFloat16(Float32(1.25))
    lora.ad[main0_q].b[0] = BFloat16(Float32(-0.5))
    lora.ad[main0_w1].b[3] = BFloat16(Float32(0.75))
    lora.ad[main1_w2].a[7] = BFloat16(Float32(-1.5))

    lora.ad[main0_q].ma[0] = Float32(0.125)
    lora.ad[main0_q].va[0] = Float32(0.25)
    lora.ad[main0_w1].mb[3] = Float32(-0.375)
    lora.ad[main1_w2].vb[7] = Float32(0.5)


def main() raises:
    var ctx = DeviceContext()
    var lora = build_zimage_lora_set(NUM_NR, NUM_CR, NUM_MAIN, D, F, RANK, ALPHA)
    _seed_resume_values(lora)

    var expected_main_adapters = NUM_MAIN * ZIMAGE_SLOTS
    var raw_saved = save_zimage_lora_main_only(lora, String(LORA_OUT), ctx)
    _require(raw_saved == expected_main_adapters, String("expected main-only raw adapter count"))

    var raw_st = SafeTensors.open(String(LORA_OUT))
    _require(len(raw_st.tensors) == expected_main_adapters * 2, String("expected raw A/B tensors only"))
    _require_key(raw_st, String("layers.0.attention.to_q.lora_A.weight"), STDtype.BF16)
    _require_key(raw_st, String("layers.0.feed_forward.w1.lora_B.weight"), STDtype.BF16)
    _require_key(raw_st, String("layers.1.feed_forward.w2.lora_A.weight"), STDtype.BF16)
    _require(
        not (String("noise_refiner.0.attention.to_q.lora_A.weight") in raw_st.tensors),
        String("raw main-only save should exclude noise_refiner"),
    )
    _require(
        not (String("context_refiner.0.attention.to_q.lora_A.weight") in raw_st.tensors),
        String("raw main-only save should exclude context_refiner"),
    )

    var resumed = load_zimage_lora_main_only_resume(
        NUM_NR, NUM_CR, NUM_MAIN, RANK, ALPHA, D, F, String(LORA_OUT), ctx
    )
    _require(len(resumed.ad) == (NUM_NR + NUM_CR + NUM_MAIN) * ZIMAGE_SLOTS, String("resumed full carrier size"))
    _require_close(resumed.ad[MAIN_START + SLOT_Q].scale, Float32(2.0), String("raw resume scale mismatch"))
    _require_close(
        _bf16(resumed.ad[MAIN_START + SLOT_Q].a[0]),
        _bf16(lora.ad[MAIN_START + SLOT_Q].a[0]),
        String("raw resume Q A mismatch"),
    )
    _require_close(
        _bf16(resumed.ad[MAIN_START + SLOT_W1].b[3]),
        _bf16(lora.ad[MAIN_START + SLOT_W1].b[3]),
        String("raw resume W1 B mismatch"),
    )
    _require_close(
        _bf16(resumed.ad[MAIN_START + ZIMAGE_SLOTS + SLOT_W2].a[7]),
        _bf16(lora.ad[MAIN_START + ZIMAGE_SLOTS + SLOT_W2].a[7]),
        String("raw resume W2 A mismatch"),
    )
    _require_close(resumed.ad[MAIN_START + SLOT_Q].ma[0], Float32(0.0), String("raw resume should zero A adam_m"))
    _require_close(resumed.ad[MAIN_START + SLOT_W1].mb[3], Float32(0.0), String("raw resume should zero B adam_m"))

    var state_saved = save_zimage_lora_main_only_state(lora, String(STATE_OUT), ctx)
    _require(state_saved == expected_main_adapters, String("expected main-only state adapter count"))

    var state_st = SafeTensors.open(String(STATE_OUT))
    _require(len(state_st.tensors) == expected_main_adapters * 6, String("expected state A/B + AdamW tensors"))
    _require_key(state_st, String("layers.0.attention.to_q.lora_A.adam_m"), STDtype.F32)
    _require_key(state_st, String("layers.0.feed_forward.w1.lora_B.adam_m"), STDtype.F32)
    _require_key(state_st, String("layers.1.feed_forward.w2.lora_B.adam_v"), STDtype.F32)
    _require(
        not (String("noise_refiner.0.attention.to_q.lora_A.adam_m") in state_st.tensors),
        String("state main-only save should exclude noise_refiner"),
    )

    var state = load_zimage_lora_main_only_state(
        NUM_NR, NUM_CR, NUM_MAIN, RANK, ALPHA, D, F, String(STATE_OUT), ctx
    )
    _require_close(state.ad[MAIN_START + SLOT_Q].scale, Float32(2.0), String("state resume scale mismatch"))
    _require_close(
        _bf16(state.ad[MAIN_START + SLOT_Q].a[0]),
        _bf16(lora.ad[MAIN_START + SLOT_Q].a[0]),
        String("state Q A mismatch"),
    )
    _require_close(
        _bf16(state.ad[MAIN_START + SLOT_Q].b[0]),
        _bf16(lora.ad[MAIN_START + SLOT_Q].b[0]),
        String("state Q B mismatch"),
    )
    _require_close(
        _bf16(state.ad[MAIN_START + SLOT_W1].b[3]),
        _bf16(lora.ad[MAIN_START + SLOT_W1].b[3]),
        String("state W1 B mismatch"),
    )
    _require_close(
        _bf16(state.ad[MAIN_START + ZIMAGE_SLOTS + SLOT_W2].a[7]),
        _bf16(lora.ad[MAIN_START + ZIMAGE_SLOTS + SLOT_W2].a[7]),
        String("state W2 A mismatch"),
    )
    _require_close(state.ad[MAIN_START + SLOT_Q].ma[0], Float32(0.125), String("state A adam_m mismatch"))
    _require_close(state.ad[MAIN_START + SLOT_Q].va[0], Float32(0.25), String("state A adam_v mismatch"))
    _require_close(state.ad[MAIN_START + SLOT_W1].mb[3], Float32(-0.375), String("state B adam_m mismatch"))
    _require_close(state.ad[MAIN_START + ZIMAGE_SLOTS + SLOT_W2].vb[7], Float32(0.5), String("state B adam_v mismatch"))

    print("[zimage-lora-resume-state] PASS:", LORA_OUT, STATE_OUT)
