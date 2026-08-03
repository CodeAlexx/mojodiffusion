# serenitymojo/models/minimax_h3/parity/minimax_h3_video_encoder_parity.mojo
#
# MiniMax-H3 video encoder parity gate: the 3D causal CNN, on a tiny
# random-weight model, with no checkpoint in existence.
#
# Reference: diffusers PR huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_video_encoder_oracle.py against the reference's own test
# fixture config, every parameter re-randomized.
#
# The intermediate `encoder(x)` is compared BEFORE `quant_conv`, so a failure
# lands on one side of the boundary rather than somewhere in the whole stack.
#
# The chosen input, 5 frames of 8x8, is deliberately small enough to reason
# about and still exercises everything that matters: both downsample levels,
# the causal temporal pad (5 -> 3 -> 2 frames), the asymmetric bottom/right
# reflect pad (8 -> 4 -> 2 with a rounding step at each), the channel-changing
# resnet with its 1x1x1 shortcut, and per-frame GroupNorm.
#
# BAR: 2e-5 absolute, as for the other forward gates.
#
# Oracle: python3 scripts/minimax_h3_video_encoder_oracle.py
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_video_encoder_parity.mojo \
#     -o output/checks/minimax_h3_video_encoder_parity \
#   && output/checks/minimax_h3_video_encoder_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.video_encoder import (
    MiniMaxH3VideoEncoderConfig,
    MiniMaxH3VideoEncoderWeights,
    minimax_h3_video_encode,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_video/video_encoder_ref.safetensors"
comptime TOL = Float32(2.0e-5)

comptime FRAMES = 5
comptime HEIGHT = 8
comptime WIDTH = 8


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _max_abs(got: List[Float32], want: List[Float32]) raises -> Float32:
    if len(got) != len(want):
        raise Error("length mismatch")
    var worst = Float32(0.0)
    for i in range(len(got)):
        var diff = got[i] - want[i]
        var mag = -diff if diff < 0.0 else diff
        if mag > worst:
            worst = mag
    return worst


def _names() -> List[String]:
    var names = List[String]()
    names.append(String("encoder.conv_in.weight"))
    names.append(String("encoder.conv_in.bias"))
    names.append(String("encoder.conv_out.weight"))
    names.append(String("encoder.conv_out.bias"))
    names.append(String("encoder.norm_out.weight"))
    names.append(String("encoder.norm_out.bias"))
    names.append(String("quant_conv.weight"))
    names.append(String("quant_conv.bias"))
    for level in range(2):
        var block = String("encoder.down_blocks.") + String(level)
        var resnet = block + ".resnets.0"
        names.append(resnet + ".norm1.weight")
        names.append(resnet + ".norm1.bias")
        names.append(resnet + ".norm2.weight")
        names.append(resnet + ".norm2.bias")
        names.append(resnet + ".conv1.weight")
        names.append(resnet + ".conv1.bias")
        names.append(resnet + ".conv2.weight")
        names.append(resnet + ".conv2.bias")
        if level == 1:
            names.append(resnet + ".conv_shortcut.weight")
            names.append(resnet + ".conv_shortcut.bias")
        names.append(block + ".downsamplers.0.conv.weight")
        names.append(block + ".downsamplers.0.conv.bias")
    return names^


def main() raises:
    print("MiniMax-H3 video encoder parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var checks = 0
    var failures = 0

    var names = _names()
    var values = List[List[Float32]]()
    for i in range(len(names)):
        values.append(_load_f32(st, String("w.") + names[i]))
    var weights = MiniMaxH3VideoEncoderWeights(names^, values^)

    var block_out = [8, 16]
    var spatial = [2, 2]
    var temporal = [2, 2]
    var config = MiniMaxH3VideoEncoderConfig(
        3, 4, block_out^, 1, spatial^, temporal^, 8, Float32(1.0e-6)
    )

    var moments = minimax_h3_video_encode(
        weights, config, _load_f32(st, "in.pixels"), FRAMES, HEIGHT, WIDTH
    )

    print("")
    print("[1] output geometry")
    var want_moments = _load_f32(st, "out.moments")
    checks += 1
    # 5 -> 3 -> 2 frames through the causal pad; 8 -> 4 -> 2 spatially.
    if (
        moments.channels == 8
        and moments.frames == 2
        and moments.height == 2
        and moments.width == 2
    ):
        print("  ok   moments [8, 2, 2, 2] — causal 5->3->2, spatial 8->4->2")
    else:
        failures += 1
        print(
            "  FAIL moments [", moments.channels, moments.frames,
            moments.height, moments.width, "]",
        )

    print("")
    print("[2] moments")
    checks += 1
    if len(moments.data) != len(want_moments):
        failures += 1
        print("  FAIL length", len(moments.data), "!=", len(want_moments))
    else:
        var worst = _max_abs(moments.data, want_moments)
        if worst <= TOL:
            print("  ok   moments", len(moments.data), "values, max_abs", worst)
        else:
            failures += 1
            print("  FAIL moments max_abs", worst)

    print("")
    if failures == 0:
        print("PASS:", checks, "checks")
    else:
        print("FAIL:", failures, "of", checks, "checks")
        raise Error("MiniMax-H3 video encoder parity gate failed")
