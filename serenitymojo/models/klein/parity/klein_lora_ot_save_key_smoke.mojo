# klein_lora_ot_save_key_smoke.mojo -- Klein OneTrainer save-key contract.
#
# Tiny structural gate: build one synthetic Klein double block and one single
# block worth of LoRA adapters, save through save_klein_lora, then inspect the
# safetensors header and alpha/down/up keys. No model weights or denoise run.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.klein.klein_stack_lora import (
    build_klein_lora_set,
    load_klein_lora_resume,
    save_klein_lora,
)
from serenitymojo.training.lora_save import _read_f32


comptime OUT = "/tmp/klein_lora_ot_save_key_smoke.safetensors"


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
    var lora = build_klein_lora_set(1, 1, 8, 16, 2, Float32(4.0))
    var saved = save_klein_lora(lora, String(OUT), ctx)
    _require(saved == 14, String("expected 14 saved Klein adapters"))

    var st = SafeTensors.open(String(OUT))
    _require(len(st.tensors) == 42, String("expected 42 tensors: 14 modules x alpha/down/up"))

    var q = String("transformer.transformer_blocks.0.attn.to_q")
    _require_alpha(st, q + String(".alpha"), Float32(4.0), ctx)
    _require_rank2(st, q + String(".lora_down.weight"), 2, 8)
    _require_rank2(st, q + String(".lora_up.weight"), 8, 2)

    var img_ff = String("transformer.transformer_blocks.0.ff.linear_in")
    _require_rank2(st, img_ff + String(".lora_down.weight"), 2, 8)
    _require_rank2(st, img_ff + String(".lora_up.weight"), 32, 2)

    var txt_ff = String("transformer.transformer_blocks.0.ff_context.linear_out")
    _require_rank2(st, txt_ff + String(".lora_down.weight"), 2, 16)
    _require_rank2(st, txt_ff + String(".lora_up.weight"), 8, 2)

    var s_qkv = String("transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj")
    _require_rank2(st, s_qkv + String(".lora_down.weight"), 2, 8)
    _require_rank2(st, s_qkv + String(".lora_up.weight"), 56, 2)

    var s_out = String("transformer.single_transformer_blocks.0.attn.to_out")
    _require_rank2(st, s_out + String(".lora_down.weight"), 2, 24)
    _require_rank2(st, s_out + String(".lora_up.weight"), 8, 2)

    var resumed = load_klein_lora_resume(1, 1, 2, Float32(4.0), String(OUT), ctx)
    _require(len(resumed.dbl) == 12, String("expected 12 resumed double adapters"))
    _require(len(resumed.sgl) == 2, String("expected 2 resumed single adapters"))
    _require(resumed.dbl[0].rank == 2, String("resumed rank mismatch"))
    _require(resumed.sgl[1].in_f == 24, String("resumed single out adapter input dim mismatch"))
    _require(resumed.sgl[1].out_f == 8, String("resumed single out adapter output dim mismatch"))

    print("[klein-lora-ot-save-key] PASS:", OUT)
