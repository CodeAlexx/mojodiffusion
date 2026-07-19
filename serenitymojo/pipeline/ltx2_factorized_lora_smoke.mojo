# LTX-2 HQ factorized LoRA surface gate.
#
# Verifies the production runtime shape: LoRA adapters attach as low-rank A/B
# factors on LTX2AVBlockWeights instead of materializing full [out,in] deltas.

from std.gpu.host import DeviceContext

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream
from serenitymojo.lora import LoraSet


comptime CKPT_FP8 = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
comptime DISTILLED = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors"
comptime CAMERA = "/home/alex/.serenity/models/loras/ltx-2-19b-lora-camera-control-static.safetensors"
comptime DETAILER = "/home/alex/.serenity/models/loras/ltx-2-19b-ic-lora-detailer.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var stream = LTX2BlockStream.open(String(CKPT_FP8))
    var blk = stream.load_block_bf16(0, ctx)
    var w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx).to_f32(ctx)

    var distilled = LoraSet.load(String(DISTILLED))
    var camera = LoraSet.load(String(CAMERA))
    var detailer = LoraSet.load(String(DETAILER))

    var d = distilled.attach_ltx2_block_factors(0, w, Float32(1.0), ctx)
    var c = camera.attach_ltx2_block_factors(0, w, Float32(0.3), ctx)
    var i = detailer.attach_ltx2_block_factors(0, w, Float32(0.6), ctx)
    var total = d + c + i
    print("factorized block0 attached distilled/camera/detailer:", d, c, i)
    print("factorized block0 total:", total)
    print("factorized block0 stored factors:", len(w.lora_names))
    if d != 34:
        raise Error("factorized smoke: distilled block0 factor count != 34")
    if c != 26:
        raise Error("factorized smoke: camera block0 factor count != 26")
    if i != 10:
        raise Error("factorized smoke: detailer block0 factor count != 10")
    if total != len(w.lora_names):
        raise Error("factorized smoke: stored factor count mismatch")
    print("FACTOR LORA SMOKE PASS")
