# qwen_lora_resume_state_smoke.mojo -- Qwen LoRA resume/state contract.
#
# Tiny structural gate: build one synthetic Qwen double block, seed non-zero
# weights and AdamW moments, then exercise raw OneTrainer resume and trainer
# state sidecar round-trips. No model weights or denoise run.

from std.builtin.dtype import DType
from std.gpu.host import DeviceContext

from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    QwenLoraSet,
    build_qwen_lora_set,
    save_qwen_lora,
    save_qwen_lora_state,
    load_qwenimage_lora_resume,
    load_qwenimage_lora_state,
)


comptime LORA_OUT = "/tmp/qwen_lora_resume_smoke.safetensors"
comptime STATE_OUT = "/tmp/qwen_lora_state_smoke.safetensors"


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


def _seed_resume_values(mut lora: QwenLoraSet):
    lora.dbl[0].b[0] = BFloat16(Float32(1.25))
    lora.dbl[4].b[3] = BFloat16(Float32(-0.5))
    lora.dbl[5].a[7] = BFloat16(Float32(0.75))
    lora.dbl[0].ma[0] = Float32(0.125)
    lora.dbl[0].va[0] = Float32(0.25)
    lora.dbl[4].mb[3] = Float32(-0.375)
    lora.dbl[5].vb[7] = Float32(0.5)


def main() raises:
    var ctx = DeviceContext()
    var lora = build_qwen_lora_set(1, 8, 16, 2, Float32(4.0))
    _seed_resume_values(lora)

    var saved = save_qwen_lora(lora, String(LORA_OUT), ctx)
    _require(saved == 12, String("expected 12 saved Qwen adapters"))

    var resumed = load_qwenimage_lora_resume(1, 2, Float32(4.0), String(LORA_OUT), ctx)
    _require(len(resumed.dbl) == 12, String("expected 12 resumed Qwen adapters"))
    _require(resumed.dbl[0].rank == 2, String("resumed rank mismatch"))
    _require(resumed.dbl[4].in_f == 8 and resumed.dbl[4].out_f == 16, String("img ff_up shape mismatch"))
    _require(resumed.dbl[5].in_f == 16 and resumed.dbl[5].out_f == 8, String("img ff_down shape mismatch"))
    _require_close(resumed.dbl[0].scale, Float32(2.0), String("raw resume scale mismatch"))
    _require_close(_bf16(resumed.dbl[0].b[0]), _bf16(lora.dbl[0].b[0]), String("raw resume B mismatch"))
    _require_close(_bf16(resumed.dbl[5].a[7]), _bf16(lora.dbl[5].a[7]), String("raw resume A mismatch"))
    _require_close(resumed.dbl[0].ma[0], Float32(0.0), String("raw resume should zero A moment"))
    _require_close(resumed.dbl[5].vb[7], Float32(0.0), String("raw resume should zero B moment"))

    var state_saved = save_qwen_lora_state(lora, String(STATE_OUT), ctx)
    _require(state_saved == 12, String("expected 12 saved Qwen state adapters"))

    var state = load_qwenimage_lora_state(1, 2, Float32(4.0), String(STATE_OUT), ctx)
    _require(len(state.dbl) == 12, String("expected 12 state Qwen adapters"))
    _require_close(state.dbl[0].scale, Float32(2.0), String("state resume scale mismatch"))
    _require_close(_bf16(state.dbl[0].b[0]), _bf16(lora.dbl[0].b[0]), String("state B mismatch"))
    _require_close(_bf16(state.dbl[4].b[3]), _bf16(lora.dbl[4].b[3]), String("state ff_up B mismatch"))
    _require_close(_bf16(state.dbl[5].a[7]), _bf16(lora.dbl[5].a[7]), String("state ff_down A mismatch"))
    _require_close(state.dbl[0].ma[0], Float32(0.125), String("state A adam_m mismatch"))
    _require_close(state.dbl[0].va[0], Float32(0.25), String("state A adam_v mismatch"))
    _require_close(state.dbl[4].mb[3], Float32(-0.375), String("state B adam_m mismatch"))
    _require_close(state.dbl[5].vb[7], Float32(0.5), String("state B adam_v mismatch"))

    print("[qwen-lora-resume-state] PASS:", LORA_OUT, STATE_OUT)
