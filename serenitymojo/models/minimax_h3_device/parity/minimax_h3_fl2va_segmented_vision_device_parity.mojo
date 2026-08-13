# MiniMax-H3 FL2VA GPU segmentation gate. The single-keyframe 2304-patch
# device path is already staged against Torch BF16. This gate duplicates that
# exact gated input into two independent [1,48,48] segments and proves each
# output half equals the corresponding single-segment result. It catches
# accidental cross-keyframe attention or row reordering without another oracle
# artifact.

from std.collections import List
from std.math import sqrt
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    MiniMaxH3VisionGrid,
)
from serenitymojo.models.minimax_h3_device.vision_tower_device import (
    minimax_h3_vision_device_weights,
    minimax_h3_vision_forward_device,
)

comptime ORACLE = (
    "/home/alex/mojodiffusion/output/minimax_h3_keyframe/"
    "vision_tower_device_ref.safetensors"
)
comptime TEXT_ENCODER_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"
)
comptime HALF_TOKENS = 576
comptime HIDDEN = 5120
comptime COS_BAR = Float64(0.999999)


def _read_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F32:
        raise Error("FL2VA segmented gate: input is not F32")
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _cosine(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("FL2VA segmented gate: length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    return dot / (sqrt(na) * sqrt(nb))


def _half(
    values: List[Float32], half: Int, stride: Int,
) -> List[Float32]:
    var out = List[Float32](capacity=stride)
    var start = half * stride
    for i in range(stride):
        out.append(values[start + i])
    return out^


def _check(label: String, got: List[Float32], want: List[Float32]) raises -> Bool:
    var cosine = _cosine(got, want)
    var passed = cosine >= COS_BAR
    print(" ", "PASS" if passed else "FAIL", label, "cos=", cosine)
    return passed


def main() raises:
    var st = SafeTensors.open(String(ORACLE))
    var rows = _read_f32(st, String("in.pixel_values"))
    if len(rows) != 2304 * 1536:
        raise Error("FL2VA segmented gate: expected 2304x1536 source rows")
    var doubled = List[Float32](capacity=2 * len(rows))
    for i in range(len(rows)):
        doubled.append(rows[i])
    for i in range(len(rows)):
        doubled.append(rows[i])

    var one_grid = List[MiniMaxH3VisionGrid]()
    one_grid.append(MiniMaxH3VisionGrid(1, 48, 48))
    var two_grids = List[MiniMaxH3VisionGrid]()
    two_grids.append(MiniMaxH3VisionGrid(1, 48, 48))
    two_grids.append(MiniMaxH3VisionGrid(1, 48, 48))

    var ctx = DeviceContext()
    var weights = minimax_h3_vision_device_weights(String(TEXT_ENCODER_DIR), ctx)
    var single = minimax_h3_vision_forward_device(weights, rows, one_grid, ctx)
    var paired = minimax_h3_vision_forward_device(weights, doubled, two_grids, ctx)
    if single.num_tokens != HALF_TOKENS or paired.num_tokens != 2 * HALF_TOKENS:
        raise Error("FL2VA segmented gate: token count mismatch")

    var failures = 0
    var embed_stride = HALF_TOKENS * HIDDEN
    for half in range(2):
        if not _check(
            String("embeds half ") + String(half),
            _half(paired.embeds, half, embed_stride),
            single.embeds,
        ):
            failures += 1
    var tap_stride = HALF_TOKENS * HIDDEN
    var paired_tap_stride = 2 * tap_stride
    for tap in range(3):
        var want = List[Float32](capacity=tap_stride)
        for i in range(tap_stride):
            want.append(single.deepstack[tap * tap_stride + i])
        for half in range(2):
            var got = List[Float32](capacity=tap_stride)
            var start = tap * paired_tap_stride + half * tap_stride
            for i in range(tap_stride):
                got.append(paired.deepstack[start + i])
            if not _check(
                String("deepstack[") + String(tap) + String("] half ")
                    + String(half),
                got, want,
            ):
                failures += 1
    if failures != 0:
        raise Error(
            String("minimax_h3_fl2va_segmented_vision_device_parity: ")
                + String(failures) + String(" failure(s)")
        )
    print("PASS: 8 checks")
