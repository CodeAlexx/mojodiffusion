# serenitymojo/models/minimax_h3/parity/minimax_h3_image_grid_parity.mojo
#
# MiniMax-H3 conditioner image-grid parity gate.
#
# Reference: the REAL Qwen3-VL image processor configured from the released
# `preprocessor_config.json` we fetched with the conditioner, run by
# scripts/minimax_h3_image_grid_oracle.py. The sweep uses `smart_resize`
# directly; a second section runs the processor object END TO END on synthetic
# images, so the sweep is anchored to what the real object produces rather than
# only to the function in isolation.
#
# Also gated: the config the port compiles in must equal the config the
# checkpoint ships. That check is what catches the ComfyUI-style failure of
# inheriting the Qwen2-VL defaults (min_pixels 3136 / max_pixels 12845056)
# instead of this checkpoint's 65536 / 16777216.
#
# Oracle: python3 scripts/minimax_h3_image_grid_oracle.py
# Run:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_image_grid_parity.mojo \
#     -o output/checks/minimax_h3_image_grid_parity \
#   && output/checks/minimax_h3_image_grid_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.image_grid import (
    MINIMAX_H3_VISION_MAX_PIXELS,
    MINIMAX_H3_VISION_MERGE_SIZE,
    MINIMAX_H3_VISION_MIN_PIXELS,
    MINIMAX_H3_VISION_PATCH_SIZE,
    MINIMAX_H3_VISION_TEMPORAL_PATCH,
    minimax_h3_image_grid,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_image_grid/image_grid_ref.safetensors"


def _load_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I64:
        raise Error(String("_load_i64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


struct Report(Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def exact_int(mut self, label: String, got: List[Int], want: List[Int]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var bad = 0
        var first = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            print("  ok  ", label, "exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, bad, "of", len(got), "differ; first at", first,
                "got", got[first], "want", want[first],
            )

    def truth(mut self, label: String, condition: Bool):
        self.checks += 1
        if condition:
            print("  ok  ", label)
        else:
            self.failures += 1
            print("  FAIL", label)


def main() raises:
    print("MiniMax-H3 conditioner image-grid parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var report = Report()

    print("[1] config matches the checkpoint's preprocessor_config.json")
    var config = List[Int]()
    config.append(MINIMAX_H3_VISION_PATCH_SIZE)
    config.append(MINIMAX_H3_VISION_MERGE_SIZE)
    config.append(MINIMAX_H3_VISION_TEMPORAL_PATCH)
    config.append(MINIMAX_H3_VISION_PATCH_SIZE * MINIMAX_H3_VISION_MERGE_SIZE)
    config.append(MINIMAX_H3_VISION_MIN_PIXELS)
    config.append(MINIMAX_H3_VISION_MAX_PIXELS)
    report.exact_int("config", config, _load_i64(st, "config"))

    print("[2] smart_resize sweep + vision token counts")
    var sizes = _load_i64(st, "sizes")
    var resized_want = _load_i64(st, "resized")
    var tokens_want = _load_i64(st, "vision_tokens")
    var resized_got = List[Int]()
    var tokens_got = List[Int]()
    var refusals = 0
    for i in range(len(sizes) // 2):
        var height = sizes[2 * i]
        var width = sizes[2 * i + 1]
        var expected_refusal = tokens_want[i] < 0
        var raised = False
        var grid_h = 0
        var grid_w = 0
        var tokens = -1
        try:
            var grid = minimax_h3_image_grid(height, width)
            grid_h = grid.height
            grid_w = grid.width
            tokens = grid.num_vision_tokens
        except:
            raised = True
            grid_h = -1
            grid_w = -1
            tokens = -1
        if expected_refusal:
            refusals += 1
        if raised != expected_refusal:
            report.truth(
                String("refusal agrees at ") + String(height) + "x" + String(width), False
            )
        resized_got.append(grid_h)
        resized_got.append(grid_w)
        tokens_got.append(tokens)
    report.exact_int("resized", resized_got, resized_want)
    report.exact_int("vision_tokens", tokens_got, tokens_want)
    report.truth("the aspect-ratio guard fired where the reference refused", refusals > 0)

    print("[3] end-to-end grid from the real processor object")
    var e2e_sizes = _load_i64(st, "e2e.sizes")
    var e2e_grid_want = _load_i64(st, "e2e.grid_thw")
    var e2e_tokens_want = _load_i64(st, "e2e.vision_tokens")
    var e2e_grid_got = List[Int]()
    var e2e_tokens_got = List[Int]()
    for i in range(len(e2e_sizes) // 2):
        var grid = minimax_h3_image_grid(e2e_sizes[2 * i], e2e_sizes[2 * i + 1])
        # grid_thw is (t, h, w); a still image has t = 1.
        e2e_grid_got.append(1)
        e2e_grid_got.append(grid.grid_h)
        e2e_grid_got.append(grid.grid_w)
        e2e_tokens_got.append(grid.num_vision_tokens)
    report.exact_int("e2e.grid_thw", e2e_grid_got, e2e_grid_want)
    report.exact_int("e2e.vision_tokens", e2e_tokens_got, e2e_tokens_want)

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks, all exact")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks differ")
        raise Error("MiniMax-H3 image-grid parity gate failed")
