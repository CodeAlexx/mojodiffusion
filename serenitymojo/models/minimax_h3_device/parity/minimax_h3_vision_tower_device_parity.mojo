# serenitymojo/models/minimax_h3_device/parity/minimax_h3_vision_tower_device_parity.mojo
#
# GATE: the DEVICE Qwen3-VL vision tower against transformers' OWN
# Qwen3VLVisionModel, run ON THE GPU in the checkpoint's native BF16 —
# matching dtype, matching device, per the repo's no-CPU-parity rule.
#
# Oracle: scripts/minimax_h3_vision_tower_device_oracle.py — the REAL FL2VA
# weights (351 bf16 tensors), the REAL shopping keyframe LANCZOS-stretched to
# the 768x768 canvas (the production first-keyframe law), through the REAL
# AutoProcessor: grid (1, 48, 48), 2304 patches, 576 merged tokens. See that
# script's header for the measured dtype law (bf16 inv_freq, f32 rotary apply,
# f32 softmax) this gate's device side replicates.
#
# STAGED, so a failure names a stage, in the PENDING gate's own contract order
# (models/text_encoder/parity/minimax_h3_qwen3vl_vision_gate.mojo):
#     check  1   input seam    Mojo preprocessor rows vs the processor's own
#                              pixel_values — BIT-exact (both are gated f32
#                              arithmetic on the same u8 canvas)
#     check  2   rope tables   the bf16-replication chain vs the model's own
#                              bf16 cos/sin — <= 1 bf16 ulp
#     check  3   after_patch   patch_embed + interpolated pos_embed
#     checks 4-8 block 0 / 8 / 16 / 24 / 26
#     checks 9-11  deepstack taps 0..2 (each POSTSHUFFLE merger separately —
#                              a preshuffle-normed tap has the right SHAPE and
#                              this is the check that catches it)
#     check 12  embeds         the final PRESHUFFLE merger
#
# BAR — flat where the reference can meet it, DERIVED where it cannot:
#     PASS iff  cos >= 0.999  OR  (1 - cos) <= 1.0 * noise.<stage>
# where `noise.<stage>` is TORCH'S OWN measured bf16 cosine deficit at that
# stage — the same tower run bf16-GPU vs f32-GPU inside the oracle script.
# MEASURED at this real geometry: torch's own noise floor is BELOW cos 0.999
# at the deep stages (block_26 cos(bf16,f32)=0.99539, embeds 0.99719,
# deepstack[2] 0.99864), because the activations reach absmax ~10^4 by block
# 26 where one bf16 ulp is 64 — so a flat 0.999 there would fail TORCH
# ITSELF and can only be met by luck, never by faithfulness. The derived arm
# is STRICTER than it looks: two equally-noisy INDEPENDENT implementations
# would sit at ~2x the reference's own deficit; requiring <= 1.0x demands the
# port be at least as close to torch-bf16 as torch-bf16 is to its own f32
# self. (Cross-checked when this gate was built: the port-vs-f32 cosine also
# BEAT torch-bf16-vs-f32 at every deep stage — 0.99609 vs 0.99539 at
# block_26, 0.99744 vs 0.99719 at embeds — i.e. the port sits BETWEEN the
# two torch references, inside the reference's own quantization noise.)
# A STRUCTURAL error (wrong merger norm width, wrong GELU, raster-order
# rotary, missed bf16 rope rounding) moves cosine far below either arm, not
# to its fourth decimal.
#
# TIMING: the device forward (traced, all stages resident) plus the production
# readback is timed cold and warm and reported against the HOST tower's
# measured 21.5 min at this exact geometry (h3-keyframe lane, 2304 patches) —
# the number this port exists to kill. The torch GPU forward time is echoed
# from the oracle for context.
#
# Run (GPU must be free — nvidia-smi first):
#   python3 scripts/minimax_h3_vision_tower_device_oracle.py
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/minimax_h3_device/parity/minimax_h3_vision_tower_device_parity.mojo \
#     -o output/checks/minimax_h3_vision_tower_device_parity \
#   && output/checks/minimax_h3_vision_tower_device_parity [oracle.safetensors]

from std.collections import List
from std.math import sqrt
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    H3_VIS_OUT_HIDDEN,
    MiniMaxH3VisionGrid,
)
from serenitymojo.models.minimax_h3_device.vision_tower_device import (
    MiniMaxH3VisionDeviceWeights,
    minimax_h3_vision_device_rope_host,
    minimax_h3_vision_device_weights,
    minimax_h3_vision_forward_device,
    minimax_h3_vision_forward_device_traced,
)
from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage
from serenitymojo.pipeline.minimax_h3_vision_preprocess import (
    minimax_h3_vision_patch_rows,
)

comptime DEFAULT_ORACLE = (
    "/home/alex/mojodiffusion/output/minimax_h3_keyframe/vision_tower_device_ref.safetensors"
)
comptime TEXT_ENCODER_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"
)

comptime COS_BAR = Float64(0.999)
# The derived arm's multiple of torch's own measured bf16 deficit. 1.0 is
# STRICT — independent equally-noisy implementations would sit at ~2.0.
comptime NOISE_MULT = Float64(1.0)
# One bf16 ulp at |x| = 1 (2^-7): the rope replication may differ from CUDA's
# cos/sin by a last-ulp f32 disagreement that flips a bf16 rounding on a
# measure-zero set of entries; anything structural is orders beyond this.
comptime ROPE_MAX_ABS_BAR = Float32(0.0079)
# The HOST tower at this exact geometry: 21.5 min (h3-keyframe lane, 2304
# patches) — the number this port exists to kill.
comptime HOST_BASELINE_S = Float64(1290.0)


def _read_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    if not st.has_tensor(name):
        raise Error(String("device gate: oracle has no tensor ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F32:
        raise Error(String("device gate: ") + name + " is not F32")
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _read_i32(ref st: SafeTensors, name: String) raises -> List[Int]:
    if not st.has_tensor(name):
        raise Error(String("device gate: oracle has no tensor ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.I32:
        raise Error(String("device gate: ") + name + " is not I32")
    var p = tv.data.unsafe_ptr().bitcast[Int32]()
    var out = List[Int](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _cosine(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error(
            String("device gate: length mismatch ") + String(len(a))
            + " vs " + String(len(b))
        )
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na <= 0.0 or nb <= 0.0:
        raise Error("device gate: cosine over a zero vector — a stage collapsed")
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("device gate: length mismatch in max_abs")
    var worst = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > worst:
            worst = d
    return worst


def _rms(a: List[Float32]) -> Float64:
    var acc = Float64(0.0)
    for i in range(len(a)):
        acc += Float64(a[i]) * Float64(a[i])
    return sqrt(acc / Float64(len(a)))


def _read_noise(ref st: SafeTensors, stage: String) raises -> Float64:
    """`noise.<stage>` — torch's OWN measured bf16 cosine deficit at that
    stage (bf16 GPU vs f32 GPU inside the oracle script). A missing entry is
    a HARD error, not a fallback: an ungated stage must never look passing."""
    var key = String("noise.") + stage
    if not st.has_tensor(key):
        raise Error(
            String("device gate: oracle has no ") + key + " — regenerate with"
            " scripts/minimax_h3_vision_tower_device_oracle.py, which measures"
            " the torch bf16-vs-f32 noise baseline every derived bar reads"
        )
    var values = _read_f32(st, key)
    if len(values) != 1:
        raise Error(String("device gate: ") + key + " is not a scalar")
    return Float64(values[0])


struct Report(Movable):
    var checks: Int
    var failures: Int
    var first_failure: String

    def __init__(out self):
        self.checks = 0
        self.failures = 0
        self.first_failure = String("")

    def stage(
        mut self, label: String, got: List[Float32], want: List[Float32],
        ref_deficit: Float64,
    ) raises:
        """One gated stage row: PASS iff cos >= COS_BAR (flat arm) OR
        deficit <= NOISE_MULT * ref_deficit (derived arm — torch's own
        measured bf16 deficit at this stage; see file header). max_abs and
        the oracle RMS are reported so max_abs reads relative to the
        signal's own scale."""
        self.checks += 1
        var cos = _cosine(got, want)
        var mx = _max_abs(got, want)
        var scale = _rms(want)
        var deficit = 1.0 - cos
        var flat_ok = cos >= COS_BAR
        var derived_ok = deficit <= NOISE_MULT * ref_deficit
        var passed = flat_ok or derived_ok
        var tag = String("PASS") if passed else String("FAIL")
        var arm = String("flat") if flat_ok else (
            String("derived") if derived_ok else String("NEITHER")
        )
        print(
            "  ", tag, " ", label, "  cos=", cos,
            "  torch_self_deficit=", ref_deficit,
            "  ours=", deficit,
            "  arm=", arm,
            "  max_abs=", mx, "  oracle_rms=", scale,
        )
        if not passed:
            self.failures += 1
            if self.first_failure == String(""):
                self.first_failure = label

    def verdict(mut self, label: String, passed: Bool, detail: String):
        self.checks += 1
        var tag = String("PASS") if passed else String("FAIL")
        print("  ", tag, " ", label, "  ", detail)
        if not passed:
            self.failures += 1
            if self.first_failure == String(""):
                self.first_failure = label


def main() raises:
    var args = argv()
    var oracle_path = String(DEFAULT_ORACLE)
    if len(args) >= 2:
        oracle_path = String(args[1])

    print("=== MiniMax-H3 Qwen3-VL VISION TOWER: DEVICE vs GPU-BF16 TORCH ===")
    print("  oracle       :", oracle_path)
    print("  text_encoder :", String(TEXT_ENCODER_DIR))
    print("  bar          : cos >=", COS_BAR, " per stage (actuals reported)")
    print("")

    var st = SafeTensors.open(oracle_path)
    var report = Report()

    # ── grids ────────────────────────────────────────────────────────────────
    var grid_thw = _read_i32(st, String("in.grid_thw"))
    var n_refs = len(grid_thw) // 3
    var grids = List[MiniMaxH3VisionGrid]()
    var num_patches = 0
    for i in range(n_refs):
        grids.append(MiniMaxH3VisionGrid(
            grid_thw[3 * i], grid_thw[3 * i + 1], grid_thw[3 * i + 2]
        ))
        num_patches += grids[i].num_patches()
    print("  grids:", n_refs, "reference(s),", num_patches, "patches")
    for i in range(len(grids)):
        print("    grid", i, ": t=", grids[i].t, " h=", grids[i].h, " w=", grids[i].w)
    print("")

    # ── check 1: the input seam — OUR preprocessor vs the processor's own ────
    var canvas_f32 = _read_f32(st, String("in.canvas_rgb"))
    var side = 768
    if len(canvas_f32) != side * side * 3:
        raise Error("device gate: in.canvas_rgb is not 768x768x3")
    var pixels = List[UInt8](capacity=len(canvas_f32))
    for i in range(len(canvas_f32)):
        pixels.append(UInt8(Int(canvas_f32[i])))
    var image = MiniMaxH3RgbImage(pixels^, side, side)
    var rows = minimax_h3_vision_patch_rows(image)
    var want_pixels = _read_f32(st, String("in.pixel_values"))
    if len(rows) != len(want_pixels):
        raise Error("device gate: preprocessor row count mismatch")
    var pix_mismatches = 0
    for i in range(len(rows)):
        if rows[i] != want_pixels[i]:
            pix_mismatches += 1
    report.verdict(
        String("input seam (preprocessor rows, BIT-exact)"),
        pix_mismatches == 0,
        String("mismatched values: ") + String(pix_mismatches)
        + " of " + String(len(rows)),
    )

    # ── check 2: rope tables — the bf16-replication chain vs the model's own ─
    var trig = minimax_h3_vision_device_rope_host(grids)
    var want_cos = _read_f32(st, String("out.rope_cos"))
    var want_sin = _read_f32(st, String("out.rope_sin"))
    var cos_mx = _max_abs(trig[0], want_cos)
    var sin_mx = _max_abs(trig[1], want_sin)
    var rope_mx = cos_mx if cos_mx > sin_mx else sin_mx
    report.verdict(
        String("rope tables (bf16-quantization replication)"),
        rope_mx <= ROPE_MAX_ABS_BAR,
        String("max_abs=") + String(rope_mx) + " (bar "
        + String(ROPE_MAX_ABS_BAR) + " = 1 bf16 ulp at 1.0), n="
        + String(len(want_cos)) + " per table",
    )

    # ── load + upload weights ────────────────────────────────────────────────
    var ctx = DeviceContext()
    var t_up0 = perf_counter_ns()
    var w = minimax_h3_vision_device_weights(String(TEXT_ENCODER_DIR), ctx)
    var t_up1 = perf_counter_ns()
    print("")
    print(
        "  uploaded", len(w.names), "device tensors (bf16, qkv split) in",
        Float64(t_up1 - t_up0) / 1.0e9, "s",
    )

    # ── the device forward, timed COLD (includes cublas/first-launch setup) ──
    var t_f0 = perf_counter_ns()
    var trace = minimax_h3_vision_forward_device_traced(w, rows, grids, ctx)
    var after_patch = trace.after_patch.to_host(ctx)   # syncs
    var t_f1 = perf_counter_ns()
    var cold_s = Float64(t_f1 - t_f0) / 1.0e9

    # ── staged comparison, contract order ────────────────────────────────────
    print("")
    print("  -- staged parity, 2304 patches / 576 tokens --")
    print("   rule: PASS iff cos >=", COS_BAR, " OR ours <=", NOISE_MULT,
          "x torch_self_deficit (torch's own bf16-vs-f32 deficit)")
    report.stage(String("after_patch"), after_patch,
                 _read_f32(st, String("out.after_patch")),
                 _read_noise(st, String("out.after_patch")))
    report.stage(String("block_00   "), trace.block_00.to_host(ctx),
                 _read_f32(st, String("out.block_00")),
                 _read_noise(st, String("out.block_00")))
    report.stage(String("block_08   "), trace.block_08.to_host(ctx),
                 _read_f32(st, String("out.block_08")),
                 _read_noise(st, String("out.block_08")))
    report.stage(String("block_16   "), trace.block_16.to_host(ctx),
                 _read_f32(st, String("out.block_16")),
                 _read_noise(st, String("out.block_16")))
    report.stage(String("block_24   "), trace.block_24.to_host(ctx),
                 _read_f32(st, String("out.block_24")),
                 _read_noise(st, String("out.block_24")))
    report.stage(String("block_26   "), trace.block_26.to_host(ctx),
                 _read_f32(st, String("out.block_26")),
                 _read_noise(st, String("out.block_26")))

    var want_deepstack = _read_f32(st, String("out.deepstack"))
    var num_tokens = num_patches // 4
    var tap_stride = num_tokens * H3_VIS_OUT_HIDDEN
    if len(want_deepstack) != 3 * tap_stride:
        raise Error("device gate: out.deepstack has the wrong element count")
    var got_ds0 = trace.ds0.to_host(ctx)
    var got_ds1 = trace.ds1.to_host(ctx)
    var got_ds2 = trace.ds2.to_host(ctx)
    for tap in range(3):
        var want_tap = List[Float32](capacity=tap_stride)
        for i in range(tap_stride):
            want_tap.append(want_deepstack[tap * tap_stride + i])
        var noise = _read_noise(st, String("out.deepstack_") + String(tap))
        if tap == 0:
            report.stage(String("deepstack[0]"), got_ds0, want_tap, noise)
        elif tap == 1:
            report.stage(String("deepstack[1]"), got_ds1, want_tap, noise)
        else:
            report.stage(String("deepstack[2]"), got_ds2, want_tap, noise)

    report.stage(
        String("embeds     "), trace.embeds.to_host(ctx),
        _read_f32(st, String("out.embeds")),
        _read_noise(st, String("out.embeds")),
    )

    # ── timing: WARM production-entry run (what i2va would actually pay) ─────
    var t_w0 = perf_counter_ns()
    var out = minimax_h3_vision_forward_device(w, rows, grids, ctx)
    var t_w1 = perf_counter_ns()
    var warm_s = Float64(t_w1 - t_w0) / 1.0e9
    if out.num_tokens != num_tokens:
        raise Error("device gate: production entry token count mismatch")

    var torch_ms = _read_f32(st, String("meta.forward_ms"))
    print("")
    print("  -- timing, 2304 patches --")
    print("   host tower (measured, h3-keyframe lane) :", HOST_BASELINE_S, "s")
    print("   device forward, cold (first launch)     :", cold_s, "s")
    print("   device forward+readback, warm           :", warm_s, "s")
    print("   torch bf16 GPU forward (oracle, warm)   :", Float64(torch_ms[0]) / 1000.0, "s")
    print("   speedup vs host (warm)                  :", HOST_BASELINE_S / warm_s, "x")

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print(
            "FAIL:", report.failures, "of", report.checks,
            "checks; first miss at", report.first_failure,
        )
        raise Error("minimax_h3_vision_tower_device_parity: FAILED")
