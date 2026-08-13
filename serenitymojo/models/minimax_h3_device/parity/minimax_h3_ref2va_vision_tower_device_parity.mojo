# MiniMax-H3 Ref2VA DEVICE vision-tower gate at the released real profile:
# one 768x1344 prepared reference, grid [1,48,84], 4032 patch rows and 1008
# merged tokens. The input is the already-gated vendor processor artifact; the
# reference is transformers' own Qwen3VLVisionModel on CUDA in native BF16.

from std.collections import List
from std.math import sqrt
from std.sys import argv
from std.time import perf_counter_ns
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

comptime DEFAULT_INPUT = (
    "/home/alex/mojodiffusion/output/minimax_h3_ref2va/"
    "conditioning_oracle.safetensors"
)
comptime DEFAULT_ORACLE = (
    "/home/alex/mojodiffusion/output/minimax_h3_ref2va/"
    "vision_tower_device_ref.safetensors"
)
comptime TEXT_ENCODER_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/Ref2VA/text_encoder"
)
comptime COS_BAR = Float64(0.999)
comptime NOISE_MULT = Float64(1.0)


def _read_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    if not st.has_tensor(name):
        raise Error(String("Ref2VA device gate: missing ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F32:
        raise Error(String("Ref2VA device gate: ") + name + " is not F32")
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _read_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    if not st.has_tensor(name):
        raise Error(String("Ref2VA device gate: missing ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.I64:
        raise Error(String("Ref2VA device gate: ") + name + " is not I64")
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _cosine(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("Ref2VA device gate: cosine length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na <= 0.0 or nb <= 0.0:
        raise Error("Ref2VA device gate: zero-vector cosine")
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("Ref2VA device gate: max_abs length mismatch")
    var worst = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > worst:
            worst = d
    return worst


def _noise(ref oracle: SafeTensors, name: String) raises -> Float64:
    var values = _read_f32(oracle, String("noise.") + name)
    if len(values) != 1:
        raise Error("Ref2VA device gate: noise entry is not scalar")
    return Float64(values[0])


def _stage(
    label: String,
    got: List[Float32],
    want: List[Float32],
    ref_deficit: Float64,
) raises -> Bool:
    var cosine = _cosine(got, want)
    var deficit = 1.0 - cosine
    var flat_ok = cosine >= COS_BAR
    var derived_ok = deficit <= NOISE_MULT * ref_deficit
    var passed = flat_ok or derived_ok
    print(
        " ", "PASS" if passed else "FAIL", label,
        "cos=", cosine, " ours=", deficit,
        " torch_self_deficit=", ref_deficit,
        " max_abs=", _max_abs(got, want),
        " arm=", "flat" if flat_ok else ("derived" if derived_ok else "NEITHER"),
    )
    return passed


def main() raises:
    var args = argv()
    var input_path = String(DEFAULT_INPUT)
    var oracle_path = String(DEFAULT_ORACLE)
    if len(args) >= 2:
        input_path = String(args[1])
    if len(args) >= 3:
        oracle_path = String(args[2])

    print("=== MiniMax-H3 Ref2VA vision tower: DEVICE vs GPU-BF16 torch ===")
    print(" input :", input_path)
    print(" oracle:", oracle_path)
    var input = SafeTensors.open(input_path)
    var oracle = SafeTensors.open(oracle_path)
    var grid = _read_i64(input, String("video_grid_thw"))
    if len(grid) != 3 or grid[0] != 1 or grid[1] != 48 or grid[2] != 84:
        raise Error("Ref2VA device gate: expected video grid [1,48,84]")
    var rows = _read_f32(input, String("pixel_values_videos"))
    if len(rows) != 4032 * 1536:
        raise Error("Ref2VA device gate: expected 4032x1536 processor rows")
    var grids = List[MiniMaxH3VisionGrid]()
    grids.append(MiniMaxH3VisionGrid(grid[0], grid[1], grid[2]))

    var ctx = DeviceContext()
    var t0 = perf_counter_ns()
    var weights = minimax_h3_vision_device_weights(String(TEXT_ENCODER_DIR), ctx)
    var t1 = perf_counter_ns()
    var out = minimax_h3_vision_forward_device(weights, rows, grids, ctx)
    var t2 = perf_counter_ns()
    if out.num_tokens != 1008:
        raise Error("Ref2VA device gate: device tower did not return 1008 tokens")

    var failures = 0
    if not _stage(
        String("embeds     "), out.embeds,
        _read_f32(oracle, String("out.embeds")),
        _noise(oracle, String("out.embeds")),
    ):
        failures += 1
    var want_ds = _read_f32(oracle, String("out.deepstack"))
    var stride = 1008 * 5120
    if len(want_ds) != 3 * stride or len(out.deepstack) != 3 * stride:
        raise Error("Ref2VA device gate: deepstack shape mismatch")
    for tap in range(3):
        var got = List[Float32](capacity=stride)
        var want = List[Float32](capacity=stride)
        for i in range(stride):
            got.append(out.deepstack[tap * stride + i])
            want.append(want_ds[tap * stride + i])
        if not _stage(
            String("deepstack[") + String(tap) + String("]"),
            got, want,
            _noise(oracle, String("out.deepstack_") + String(tap)),
        ):
            failures += 1

    print(" upload_s=", Float64(t1 - t0) / 1.0e9)
    print(" forward+readback_s=", Float64(t2 - t1) / 1.0e9)
    if failures != 0:
        raise Error(
            String("minimax_h3_ref2va_vision_tower_device_parity: ")
            + String(failures) + String(" failure(s)")
        )
    print("PASS: 4 checks")
