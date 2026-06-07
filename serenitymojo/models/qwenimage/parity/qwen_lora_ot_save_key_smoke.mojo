# qwen_lora_ot_save_key_smoke.mojo -- Qwen OneTrainer save-key contract.
#
# This is intentionally tiny: it builds one synthetic Qwen LoRA block, saves it
# through save_qwen_lora, and reopens only the safetensors header/one alpha value.
# It does not run denoise, load model weights, or exercise image generation.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    build_qwen_lora_set,
    save_qwen_lora,
)
from serenitymojo.training.lora_save import _read_f32


comptime OUT = "/tmp/qwen_lora_ot_save_key_smoke.safetensors"


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


def _require_alpha(st: SafeTensors, key: String, expected: Float32, ctx: DeviceContext) raises:
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
    var lora = build_qwen_lora_set(1, 8, 16, 2, Float32(4.0))
    var saved = save_qwen_lora(lora, String(OUT), ctx)
    _require(saved == 12, String("expected 12 saved Qwen adapters"))

    var st = SafeTensors.open(String(OUT))
    _require(len(st.tensors) == 36, String("expected 36 tensors: 12 modules x alpha/down/up"))

    var base = String("transformer.transformer_blocks.0.attn.to_q")
    _require_alpha(st, base + String(".alpha"), Float32(4.0), ctx)
    _require_rank2(st, base + String(".lora_down.weight"), 2, 8)
    _require_rank2(st, base + String(".lora_up.weight"), 8, 2)

    var mlp_up = String("transformer.transformer_blocks.0.img_mlp.net.0.proj")
    _require_rank2(st, mlp_up + String(".lora_down.weight"), 2, 8)
    _require_rank2(st, mlp_up + String(".lora_up.weight"), 16, 2)

    var mlp_down = String("transformer.transformer_blocks.0.txt_mlp.net.2")
    _require_rank2(st, mlp_down + String(".lora_down.weight"), 2, 16)
    _require_rank2(st, mlp_down + String(".lora_up.weight"), 8, 2)

    print("[qwen-lora-ot-save-key] PASS:", OUT)
