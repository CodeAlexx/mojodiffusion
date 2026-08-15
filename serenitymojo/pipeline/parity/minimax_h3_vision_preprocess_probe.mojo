# serenitymojo/pipeline/parity/minimax_h3_vision_preprocess_probe.mojo
#
# Gates `pipeline/minimax_h3_vision_preprocess.mojo` BIT-EXACT against the real
# Qwen3-VL image processor's own `pixel_values`. Host only — no GPU, no weights.
#
# BIT-EXACT is the right bar and it is reachable: after the fused constants are
# fixed (127.5 / 127.5) every value is one subtract and one divide on an integer
# input, and everything else is an index permutation. There is no accumulation
# anywhere, so any difference at all is a wrong constant or a wrong index.
#
# The negative control is half the value here. Case 5 is a sub-65,536-pixel
# geometry that the processor WOULD resize; the module's resampler is
# deliberately unimplemented, so the probe requires it to RAISE. A module that
# quietly returned unresized patches would pass every positive case and be
# wrong exactly where a dev geometry is used.
#
# Run:
#   CUDA_VISIBLE_DEVICES="" /home/alex/torchref/venv/bin/python \
#     scripts/minimax_h3_vision_preprocess_oracle.py
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_vision_preprocess_probe.mojo \
#     -o /tmp/h3_vision_preprocess_probe -Xlinker -lm \
#   && /tmp/h3_vision_preprocess_probe \
#        output/minimax_h3_keyframe/vision_preprocess_ref.safetensors

from std.sys import argv
from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage
from serenitymojo.pipeline.minimax_h3_vision_preprocess import (
    MINIMAX_H3_VISION_ROW_WIDTH,
    minimax_h3_vision_patch_grid,
    minimax_h3_vision_patch_rows,
    minimax_h3_vision_resize_is_identity,
)


struct Report(Copyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def ok(mut self, label: String, detail: String):
        self.checks += 1
        print("  ok  ", label, "—", detail)

    def fail(mut self, label: String, detail: String):
        self.checks += 1
        self.failures += 1
        print("  FAIL", label, "—", detail)


def _f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _i32(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    var p = tv.data.unsafe_ptr().bitcast[Int32]()
    var out = List[Int](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _u8(ref st: SafeTensors, name: String) raises -> List[UInt8]:
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    var p = tv.data.unsafe_ptr()
    var out = List[UInt8](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: minimax_h3_vision_preprocess_probe <vision_preprocess_ref.safetensors>")
        return
    print("MiniMax-H3 conditioner IMAGE PREPROCESSOR probe")
    print("  reference:", String(args[1]))
    print("")
    var report = Report()
    var st = SafeTensors.open(String(args[1]))
    var meta = _i32(st, String("meta"))
    var num_cases = len(meta) // 4

    for c in range(num_cases):
        var idx = meta[4 * c]
        var h = meta[4 * c + 1]
        var w = meta[4 * c + 2]
        var identity = meta[4 * c + 3] == 1
        var label = String(w) + "x" + String(h)
        var pixels = _u8(st, String("img_") + String(idx))
        var image = MiniMaxH3RgbImage(pixels^, h, w)

        if minimax_h3_vision_resize_is_identity(h, w) != identity:
            report.fail(
                label + " resize-identity prediction",
                String("module says ") + String(minimax_h3_vision_resize_is_identity(h, w))
                + ", processor says " + String(identity),
            )
            continue

        if not identity:
            # NEGATIVE CONTROL: the module must refuse, not silently skip the
            # resize it cannot do.
            var refused = False
            try:
                var bad = minimax_h3_vision_patch_rows(image)
                _ = len(bad)
            except:
                refused = True
            if refused:
                report.ok(
                    label + " (would be resized)",
                    "REFUSED — resampler unimplemented, and says so",
                )
            else:
                report.fail(
                    label + " (would be resized)",
                    "returned patches without resizing — silently wrong",
                )
            continue

        var got = minimax_h3_vision_patch_rows(image)
        var want = _f32(st, String("px_") + String(idx))
        if len(got) != len(want):
            report.fail(
                label, String("length ") + String(len(got)) + " vs " + String(len(want))
            )
            continue
        var bad = 0
        var first = -1
        for i in range(len(want)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            report.ok(
                label,
                String("bit-exact over ") + String(len(want)) + " values ("
                + String(len(want) // MINIMAX_H3_VISION_ROW_WIDTH) + " rows x "
                + String(MINIMAX_H3_VISION_ROW_WIDTH) + ")",
            )
        else:
            report.fail(
                label,
                String(bad) + " of " + String(len(want)) + " differ, first at "
                + String(first) + ": got " + String(got[first]) + " want "
                + String(want[first]),
            )

        var grid = minimax_h3_vision_patch_grid(image)
        var wgrid = _i32(st, String("grid_") + String(idx))
        if grid[0] == wgrid[0] and grid[1] == wgrid[1] and grid[2] == wgrid[2]:
            report.ok(
                label + " grid",
                String(grid[0]) + "x" + String(grid[1]) + "x" + String(grid[2]),
            )
        else:
            report.fail(label + " grid", "mismatch")

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_vision_preprocess probe FAILED")
