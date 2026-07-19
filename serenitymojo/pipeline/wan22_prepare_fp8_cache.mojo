# Quantize the pinned official Wan2.2-TI2V-5B artifact once and atomically save
# the exact Mojo E4M3 resident store. Development/artifact preparation only.

from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.models.dit.wan22_dit import Wan22Config, Wan22DiT


comptime DEFAULT_SOURCE = (
    "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-Mojo"
)
comptime DEFAULT_OUTPUT = (
    "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-Mojo/"
    "wan22_dit_fp8_e4m3_b8fff7315c768468.safetensors"
)


def main() raises:
    var args = argv()
    var source = String(DEFAULT_SOURCE)
    var output = String(DEFAULT_OUTPUT)
    if len(args) >= 2:
        source = String(args[1])
    if len(args) >= 3:
        output = String(args[2])
    print("=== Wan2.2 pinned FP8 cache preparation ===")
    print("  source:", source)
    print("  output:", output)
    var ctx = DeviceContext()
    var model = Wan22DiT.load_fp8_resident(
        source, Wan22Config.ti2v_5b(), ctx
    )
    model.save_fp8_cache(output, ctx)
    print("GATE cache-written", output)
