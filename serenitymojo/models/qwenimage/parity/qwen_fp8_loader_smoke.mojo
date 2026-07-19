# qwen_fp8_loader_smoke.mojo -- lightweight gate for Qwen FP8 checkpoint loading.
#
# This intentionally loads one small real checkpoint tensor only. It proves the
# Qwen loader does not route F8_E4M3 checkpoint storage through the generic
# F32/BF16 cast helper, and that the runtime tensor boundary is BF16.

from std.gpu.host import DeviceContext
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.qwenimage.weights import load_qwen_tensor_bf16, load_qwen_host_bf16


comptime CKPT = "/home/alex/.serenity/models/checkpoints/qwen_image_fp8_e4m3fn.safetensors"
comptime KEY = "img_in.weight"


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(CKPT))
    var info = st.tensor_info(String(KEY))
    print("[qwen-fp8-loader] key:", KEY, " header dtype:", info.dtype.name())
    _require(
        info.dtype == STDtype.F8_E4M3,
        String("expected F8_E4M3 checkpoint tensor, got ") + info.dtype.name(),
    )

    var t = load_qwen_tensor_bf16(st, String(KEY), ctx)
    var sh = t.shape()
    _require(t.dtype() == STDtype.BF16, String("expected BF16 runtime tensor"))
    _require(len(sh) == 2, String("expected rank-2 img_in.weight"))
    _require(sh[0] == 3072 and sh[1] == 64, String("unexpected img_in.weight shape"))

    var h = t.to_host_bf16(ctx)
    _require(len(h) == 3072 * 64, String("unexpected host BF16 length"))
    var hb = load_qwen_host_bf16(st, String(KEY), ctx)
    _require(len(hb) == 3072 * 64, String("unexpected public host BF16 length"))
    print("[qwen-fp8-loader] PASS: FP8 checkpoint tensor dequantized to BF16")
