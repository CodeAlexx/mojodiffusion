# serenitymojo/models/text_encoder/parity/minimax_h3_qwen3vl_vision_cpu_gate.mojo
#
# CPU gate for MiniMax-H3's Qwen3-VL vision tower's WEIGHTED forward —
# `minimax_h3_vision_forward_traced` / `minimax_h3_vision_forward` in
# `models/text_encoder/minimax_h3_qwen3vl_vision.mojo`.
#
# Oracle: scripts/minimax_h3_vision_oracle.py, run on the REAL FL2VA
# `text_encoder` weights, CPU, float32, EAGER attention, two references —
# `(1,4,6)` and `(2,2,4)` — 40 patches, 10 merged tokens total. See that
# script's own header for why CPU/float32 isolates the port's arithmetic from
# bf16 rounding (both sides upcast the SAME bf16 bytes the SAME way).
#
# This is the CPU sibling of `minimax_h3_qwen3vl_vision_gate.mojo` (which
# stays PENDING-GPU per its own header/contract — do not edit it here). This
# gate does not touch a GPU or a DeviceContext at all: `minimax_h3_vision_
# forward_traced` is pure `List[Float32]` host arithmetic, and the real FL2VA
# checkpoint on disk is read once, host-side, via `ShardedSafeTensors`.
#
# STAGED, in the oracle's own dump order, so a failure names the stage:
#   out.after_patch          patch_embed + interpolated pos_embed
#   out.block_00/08/16/24/26 the residual stack, at 5 taps across the depth
#   out.deepstack[0..2]      the three POSTSHUFFLE-norm deepstack mergers
#   out.embeds               the final PRESHUFFLE-norm merger
#
# Comparison does NOT early-exit: every stage is checked and reported, and the
# gate raises at the end if any stage missed its bar — so one bad block does
# not hide whether the ones after it also broke.
#
# BARS ARE DERIVED, NOT PICKED. Methodology adopted from h3-keyframe's forward
# oracle (de3ba5c) because it is better than the hand-picked cosine constants
# this gate used before.
#
# The oracle runs the SAME tower a second time in float64 and records how far
# TORCH'S OWN float32 path lands from it, per stage, as `f32err.<stage>`. Each
# bar here is `BAR_MULTIPLE * f32err`, on max_abs_diff. The only way to raise a
# bar is for torch's own f32 error to rise — a property of the model, not of
# this port — so a bar can never be quietly widened until a bug slips through.
#
# WHY A FLAT BAR IS WRONG HERE, concretely: torch's own f32 error is 8.3e-06 at
# after_patch and 4.2e-02 at block_26, five orders of magnitude apart, because
# the activations grow to std ~94 by the last block. A flat `max_abs < 1e-3`
# would FAIL a correct implementation at block 26; a flat loose bar would PASS a
# broken one at block 0. Cosine is still reported, but it is NOT the bar — at
# these magnitudes cosine stays ~0.9999999999 while max_abs moves by orders of
# magnitude, which is exactly how a real defect would hide.
# across a growing-magnitude residual stack):
#   every stage:  max_abs_diff <= 4.0 * f32err.<stage>  (from the oracle dump)
#
# Run:
#   python3 scripts/minimax_h3_vision_oracle.py [out.safetensors]
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/text_encoder/parity/minimax_h3_qwen3vl_vision_cpu_gate.mojo \
#     -o <scratch>/vgate \
#   && <scratch>/vgate [oracle.safetensors]
#
# -O2, not -O0: 27 blocks of host matmul (~115 GMAC at this tiny 40-patch
# scale, dominated by the 4304-wide MLP) is workable at -O0 but noticeably
# slower; -O2 keeps the wall time reasonable for a CPU-only run.

from std.sys import argv
from std.collections import List
from std.math import sqrt
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    H3_VIS_DEEPSTACK_0,
    H3_VIS_DEEPSTACK_1,
    H3_VIS_DEEPSTACK_2,
    H3_VIS_DEPTH,
    H3_VIS_OUT_HIDDEN,
    MiniMaxH3VisionGrid,
    minimax_h3_vision_forward_traced,
    minimax_h3_vision_load_weights,
)

comptime DEFAULT_ORACLE = (
    "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a"
    "/scratchpad/vision_ref.safetensors"
)
comptime TEXT_ENCODER_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"
)

# Bars are `BAR_MULTIPLE * torch's own measured f32 error` per stage. 4x leaves
# room for a different-but-correct summation order without admitting a real
# defect: a genuine arithmetic bug moves max_abs by orders of magnitude, not by
# a factor of four.
comptime BAR_MULTIPLE = Float64(4.0)


struct Report(Movable):
    var checks: Int
    var failures: Int
    var first_failure: String

    def __init__(out self):
        self.checks = 0
        self.failures = 0
        self.first_failure = String("")

    def row(
        mut self, label: String, cos: Float64, max_abs: Float32,
        bar: Float64, f32err: Float64,
    ):
        self.checks += 1
        var passed = Float64(max_abs) <= bar
        var tag = String("ok  ") if passed else String("FAIL")
        var margin = Float64(max_abs) / bar if bar > 0.0 else Float64(0.0)
        print(
            "  ", tag, " ", label, "  max_abs=", max_abs, " bar=", bar,
            " (", BAR_MULTIPLE, "x torch f32err ", f32err, ")  used=",
            margin * 100.0, "% of bar   cos=", cos,
        )
        if not passed:
            self.failures += 1
            if self.first_failure == String(""):
                self.first_failure = label


def _read_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    if not st.has_tensor(name):
        raise Error(String("cpu_gate: oracle has no tensor ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F32:
        raise Error(String("cpu_gate: ") + name + " is not F32")
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _read_i32(ref st: SafeTensors, name: String) raises -> List[Int]:
    if not st.has_tensor(name):
        raise Error(String("cpu_gate: oracle has no tensor ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.I32:
        raise Error(String("cpu_gate: ") + name + " is not I32")
    var p = tv.data.unsafe_ptr().bitcast[Int32]()
    var out = List[Int](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _cosine(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error(
            String("cpu_gate: length mismatch ") + String(len(a)) + " vs " + String(len(b))
        )
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na <= 0.0 or nb <= 0.0:
        return Float64(0.0)
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("cpu_gate: length mismatch in max_abs")
    var worst = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > worst:
            worst = d
    return worst


def _read_f32err(ref st: SafeTensors, stage: String) raises -> Float64:
    """`f32err.<stage>` from the oracle — torch's OWN f32 error at that stage.

    A missing entry is a hard error, not a fallback to some default bar: an
    ungated stage must never look like a passing one."""
    var key = String("f32err.") + stage
    if not st.has_tensor(key):
        raise Error(
            String("cpu_gate: oracle has no ") + key + " — regenerate it with"
            " scripts/minimax_h3_vision_oracle.py, which computes the float64"
            " baseline every bar is derived from"
        )
    var values = _read_f32(st, key)
    if len(values) != 1:
        raise Error(String("cpu_gate: ") + key + " is not a scalar")
    return Float64(values[0])


def _compare(
    mut report: Report, ref st: SafeTensors, label: String, stage: String,
    got: List[Float32], want: List[Float32],
) raises:
    var f32err = _read_f32err(st, stage)
    var bar = BAR_MULTIPLE * f32err
    var cos = _cosine(got, want)
    var mx = _max_abs(got, want)
    report.row(label, cos, mx, bar, f32err)


def main() raises:
    var args = argv()
    var oracle_path = String(DEFAULT_ORACLE)
    if len(args) >= 2:
        oracle_path = String(args[1])

    print("MiniMax-H3 Qwen3-VL VISION TOWER — CPU weighted-forward gate")
    print("  oracle       :", oracle_path)
    print("  text_encoder :", String(TEXT_ENCODER_DIR))
    print("")

    var st = SafeTensors.open(oracle_path)

    var grid_thw = _read_i32(st, String("in.grid_thw"))
    var n_refs = len(grid_thw) // 3
    var grids = List[MiniMaxH3VisionGrid]()
    for i in range(n_refs):
        grids.append(
            MiniMaxH3VisionGrid(grid_thw[3 * i], grid_thw[3 * i + 1], grid_thw[3 * i + 2])
        )
    var num_patches = 0
    for i in range(len(grids)):
        num_patches += grids[i].num_patches()
    print("  grids: ", n_refs, "references,", num_patches, "patches total")
    for i in range(len(grids)):
        print("    grid", i, ": t=", grids[i].t, " h=", grids[i].h, " w=", grids[i].w)

    var pixel_values = _read_f32(st, String("in.pixel_values"))

    var t_load0 = perf_counter_ns()
    var weights = minimax_h3_vision_load_weights(String(TEXT_ENCODER_DIR))
    var t_load1 = perf_counter_ns()
    print("")
    print(
        "  loaded 351 vision tensors (bf16 -> f32) in",
        Float64(t_load1 - t_load0) / 1.0e9, "s",
    )

    var t_fwd0 = perf_counter_ns()
    var trace = minimax_h3_vision_forward_traced(weights, pixel_values, grids)
    var t_fwd1 = perf_counter_ns()
    var fwd_seconds = Float64(t_fwd1 - t_fwd0) / 1.0e9
    print("  forward (27 blocks + mergers) in", fwd_seconds, "s")

    print("")
    print("per-stage comparison (oracle dump order)")
    var report = Report()

    _compare(
        report, st, String("out.after_patch"), String("out.after_patch"), trace.after_patch,
        _read_f32(st, String("out.after_patch")),
    )
    _compare(
        report, st, String("out.block_00"), String("out.block_00"), trace.block_00,
        _read_f32(st, String("out.block_00")),
    )
    _compare(
        report, st, String("out.block_08"), String("out.block_08"), trace.block_08,
        _read_f32(st, String("out.block_08")),
    )
    _compare(
        report, st, String("out.block_16"), String("out.block_16"), trace.block_16,
        _read_f32(st, String("out.block_16")),
    )
    _compare(
        report, st, String("out.block_24"), String("out.block_24"), trace.block_24,
        _read_f32(st, String("out.block_24")),
    )
    _compare(
        report, st, String("out.block_26"), String("out.block_26"), trace.block_26,
        _read_f32(st, String("out.block_26")),
    )

    var want_deepstack = _read_f32(st, String("out.deepstack"))
    var tokens = trace.output.num_tokens
    var tap_stride = tokens * H3_VIS_OUT_HIDDEN
    if len(want_deepstack) != 3 * tap_stride:
        raise Error(
            String("cpu_gate: out.deepstack has ") + String(len(want_deepstack))
            + " values, expected " + String(3 * tap_stride)
        )
    for tap in range(3):
        var want_tap = List[Float32](capacity=tap_stride)
        for i in range(tap_stride):
            want_tap.append(want_deepstack[tap * tap_stride + i])
        var got_tap = trace.output.deepstack_block(tap)
        _compare(
            report, st, String("out.deepstack[") + String(tap) + String("]"),
            String("out.deepstack_") + String(tap), got_tap, want_tap,
        )

    _compare(
        report, st, String("out.embeds"), String("out.embeds"),
        trace.output.embeds, _read_f32(st, String("out.embeds")),
    )

    print("")
    print("wall time: load", Float64(t_load1 - t_load0) / 1.0e9, "s, forward", fwd_seconds, "s")
    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print(
            "FAIL:", report.failures, "of", report.checks,
            "checks; first miss at", report.first_failure,
        )
        raise Error("minimax_h3_qwen3vl_vision_cpu_gate: FAILED")
