# serenitymojo/models/text_encoder/parity/minimax_h3_qwen3vl_vision_forward_probe.mojo
#
# Gates `models/text_encoder/minimax_h3_qwen3vl_vision_forward.mojo` — the
# WEIGHTED forward of MiniMax-H3's Qwen3-VL vision tower — against transformers'
# own `Qwen3VLVisionModel`, on the REAL FL2VA weights.
#
# HOST float32 on both sides, CPU on both sides. That is not a compromise here,
# it is the point: this Mojo module IS the host oracle of the eventual device
# port, so the comparison is same-dtype/same-device-class and isolates LOGIC.
# The device port owes its own bf16 gate against this; see the module header.
#
# ── THE STACK IS STEPPED, NOT RUN END TO END ────────────────────────────────
# The probe drives the 27 blocks itself and compares at every tap the oracle
# dumps (blocks 0, 8, 16, 24, 26). A single end-to-end comparison of a 27-block
# tower tells you it is wrong and nothing else; per-block, the first failing
# stage names the bug. The stages are, in order:
#     patch_embed -> +pos_embed -> [27 blocks, tapping 8/16/24] -> merger
#
# ── BARS ────────────────────────────────────────────────────────────────────
# BIT-EXACT for what carries no accumulation: cu_seqlens (integers) and the
# merge-block/raster agreement. COSINE + max_abs for everything downstream of a
# dot product, because this module accumulates in SIMD lanes and torch goes
# through BLAS — neither order is the other's and no bar-tightening changes
# that. The absolute tolerance WIDENS with depth on purpose: `after_block_26`
# has std ~103 in the reference, so a 1e-3 max_abs there is ~1e-5 relative,
# while the same number at `after_patch_embed` (std 0.45) would be sloppy. A
# single flat tolerance across a residual stack either passes garbage at the top
# or fails clean arithmetic at the bottom.
#
# ── RUNTIME ─────────────────────────────────────────────────────────────────
# One 280-patch image through 27 blocks is ~115 GMAC in scalar-plus-SIMD host
# float32. Build with -O2. `max_blocks` (argv[3]) runs a prefix — use it to
# localize a failure cheaply before paying for the full depth. The final
# `embeds` check is SKIPPED for a partial run rather than compared against a
# reference it cannot match, and says so.
#
# Run (CPU only; ~2.4 GiB of host weights):
#   CUDA_VISIBLE_DEVICES="" python3 scripts/minimax_h3_vision_forward_oracle.py
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/text_encoder/parity/minimax_h3_qwen3vl_vision_forward_probe.mojo \
#     -o /tmp/h3_vision_forward_probe -Xlinker -lm \
#   && /tmp/h3_vision_forward_probe \
#        output/minimax_h3_keyframe/vision_forward_ref.safetensors \
#        /home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder [max_blocks]

from std.sys import argv
from std.collections import List
from std.math import sqrt
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    H3_VIS_DEPTH,
    minimax_h3_vision_pos_embed_interpolation,
    H3_VIS_HIDDEN,
    H3_VIS_NUM_DEEPSTACK,
    H3_VIS_OUT_HIDDEN,
    MiniMaxH3VisionGrid,
    minimax_h3_vision_cu_seqlens,
    minimax_h3_vision_deepstack_tap_blocks,
)
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision_forward import (
    MiniMaxH3VisionWeights,
    minimax_h3_vision_block_forward,
    minimax_h3_vision_load_weights,
    minimax_h3_vision_merger,
    minimax_h3_vision_patch_embed,
    minimax_h3_vision_pos_embeds,
    minimax_h3_vision_rotary_tables,
)

comptime _VISION_PREFIX = "model.visual."


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


def _read_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    if not st.has_tensor(name):
        raise Error(String("probe: reference has no tensor ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F32:
        raise Error(String("probe: ") + name + " is not F32")
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _read_i32(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.I32:
        raise Error(String("probe: ") + name + " is not I32")
    var p = tv.data.unsafe_ptr().bitcast[Int32]()
    var out = List[Int](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _stage_tol(ref st: SafeTensors, stage: String, factor: Float32) raises -> Float32:
    """The bar for a stage: `factor` x TORCH'S OWN float32 error at that stage.

    The oracle measures `|torch_f32 - torch_f64|` per stage and dumps it. Two
    independent float32 evaluations of the same graph — torch's BLAS order and
    this module's SIMD-lane order — each sit about that far from the true
    answer, so their DIFFERENCE can reach roughly twice it; `factor` is the
    margin above that.

    The point of deriving the bar instead of typing it: a hand-picked constant
    can be nudged upward until a broken port passes, and nothing records that it
    was. This number can only rise if TORCH's own error rises, which is a
    property of the model and the input, not of the port under test."""
    var name = String("f32err_") + stage
    if not st.has_tensor(name):
        raise Error(
            String("probe: the reference carries no f32 baseline for ") + stage
            + " — regenerate it with the current oracle"
        )
    var v = _read_f32(st, name)
    return factor * v[0]


def _compare(
    mut report: Report, label: String,
    got: List[Float32], want: List[Float32], abs_tol: Float32,
) raises:
    if len(got) != len(want):
        report.fail(
            label, String("length ") + String(len(got)) + ", want " + String(len(want))
        )
        return
    var max_abs = Float32(0.0)
    var at = -1
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(want)):
        var d = got[i] - want[i]
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
            at = i
        dot += Float64(got[i]) * Float64(want[i])
        na += Float64(got[i]) * Float64(got[i])
        nb += Float64(want[i]) * Float64(want[i])
    var cosine = Float64(1.0)
    if na > 0.0 and nb > 0.0:
        cosine = dot / (sqrt(na) * sqrt(nb))
    var rms = sqrt(nb / Float64(len(want)))
    if cosine > 0.9999999 and max_abs <= abs_tol:
        report.ok(
            label,
            String("cos=") + String(cosine) + " max_abs=" + String(max_abs)
            + " (ref rms " + String(rms) + ", tol " + String(abs_tol) + ")",
        )
    else:
        report.fail(
            label,
            String("cos=") + String(cosine) + " max_abs=" + String(max_abs)
            + " at " + String(at) + " (ref rms " + String(rms) + ", tol "
            + String(abs_tol) + ")",
        )


def main() raises:
    var args = argv()
    if len(args) < 3:
        print(
            "usage: minimax_h3_qwen3vl_vision_forward_probe <ref.safetensors>"
            " <text_encoder_dir> [max_blocks]"
        )
        return
    var ref_path = String(args[1])
    var te_dir = String(args[2])
    var max_blocks = H3_VIS_DEPTH
    if len(args) >= 4:
        max_blocks = atol(String(args[3]))
    if max_blocks < 1 or max_blocks > H3_VIS_DEPTH:
        raise Error("probe: max_blocks out of range")

    print("MiniMax-H3 Qwen3-VL VISION TOWER — weighted forward probe")
    print("  reference:", ref_path)
    print("  weights  :", te_dir)
    print("  depth    :", max_blocks, "of", H3_VIS_DEPTH, "blocks")
    print("")
    var report = Report()
    var st = SafeTensors.open(ref_path)

    var grid_thw = _read_i32(st, String("grid_thw"))
    var grids = List[MiniMaxH3VisionGrid]()
    var n_ref = len(grid_thw) // 3
    for i in range(n_ref):
        grids.append(
            MiniMaxH3VisionGrid(grid_thw[3 * i], grid_thw[3 * i + 1], grid_thw[3 * i + 2])
        )
    var num_patches = 0
    for i in range(len(grids)):
        num_patches += grids[i].num_patches()
    var tokens = grids[0].num_tokens()
    print("  grid:", grids[0].t, "x", grids[0].h, "x", grids[0].w, "->",
          num_patches, "patches,", tokens, "merged tokens")

    # ── [1] cu_seqlens — integers, BIT-EXACT ────────────────────────────────
    print("")
    print("[1] cu_seqlens (per-FRAME segments)")
    var cu = minimax_h3_vision_cu_seqlens(grids)
    var want_cu = _read_i32(st, String("cu_seqlens"))
    var cu_bad = 0
    if len(cu) != len(want_cu):
        report.fail("cu_seqlens length", String(len(cu)) + " vs " + String(len(want_cu)))
    else:
        for i in range(len(cu)):
            if cu[i] != want_cu[i]:
                cu_bad += 1
        if cu_bad == 0:
            report.ok(
                "cu_seqlens", String(len(cu)) + " boundaries, exact"
            )
        else:
            report.fail("cu_seqlens", String(cu_bad) + " differ")

    # ── [2] rotary tables ───────────────────────────────────────────────────
    print("")
    print("[2] rotary cos/sin (2-D half-width, merge-block order)")
    var rotary = minimax_h3_vision_rotary_tables(grids)
    _compare(report, "rotary_cos", rotary.cos, _read_f32(st, String("rotary_cos")), Float32(2.0e-7))
    _compare(report, "rotary_sin", rotary.sin, _read_f32(st, String("rotary_sin")), Float32(2.0e-7))

    var t0 = perf_counter_ns()
    var weights = minimax_h3_vision_load_weights(te_dir)
    var t1 = perf_counter_ns()
    print("")
    print("  loaded 351 vision tensors to host f32 (",
          Float64(t1 - t0) / 1.0e9, "s )")

    # ── [3] patch embed + position embed ────────────────────────────────────
    print("")
    print("[3] patch_embed (a LINEAR) and the interpolated position embed")
    var pixel_values = _read_f32(st, String("pixel_values"))
    var hidden = minimax_h3_vision_patch_embed(pixel_values, num_patches, weights)
    _compare(
        report, "after_patch_embed", hidden,
        _read_f32(st, String("after_patch_embed")),
        _stage_tol(st, String("after_patch_embed"), Float32(4.0)),
    )
    # The bilinear lookup, decomposed. The four corner INDICES are integers and
    # a single mismatch moves a whole 48x48 grid cell — so they are BIT-EXACT,
    # separately from the weights, which carry torch's float32 linspace
    # rounding. Gating only the summed result would let an index bug hide behind
    # a loose float tolerance, which is exactly how the float64-linspace bug
    # this probe found survived the geometry-only gates.
    var interp = minimax_h3_vision_pos_embed_interpolation(grids)
    var want_idx = _read_i32(st, String("pos_corner_indices"))
    var idx_bad = 0
    var idx_first = -1
    if len(interp.indices) != len(want_idx):
        report.fail("pos corner indices length", String(len(interp.indices)))
    else:
        for i in range(len(want_idx)):
            if interp.indices[i] != want_idx[i]:
                idx_bad += 1
                if idx_first < 0:
                    idx_first = i
        if idx_bad == 0:
            report.ok(
                "pos corner indices",
                String("bit-exact over ") + String(len(want_idx)) + " corners",
            )
        else:
            report.fail(
                "pos corner indices",
                String(idx_bad) + " differ, first at " + String(idx_first)
                + " — a WHOLE grid cell each",
            )
    var want_w = _read_f32(st, String("pos_corner_weights"))
    var got_w = List[Float32]()
    for i in range(len(interp.weights)):
        got_w.append(Float32(interp.weights[i]))
    _compare(report, "pos corner weights", got_w, want_w, Float32(1.0e-7))

    var pos = minimax_h3_vision_pos_embeds(grids, weights)
    _compare(report, "pos_embeds", pos, _read_f32(st, String("pos_embeds")), Float32(1.0e-6))
    for i in range(len(hidden)):
        hidden[i] = hidden[i] + pos[i]
    # after_pos_add has no f64 baseline of its own (it is patch_embed + pos);
    # patch_embed's is the right scale for it.
    _compare(
        report, "after_pos_add", hidden,
        _read_f32(st, String("after_pos_add")),
        _stage_tol(st, String("after_patch_embed"), Float32(4.0)),
    )

    # ── [4] the block stack, compared at every tap the oracle dumped ────────
    print("")
    print("[4] the block stack — compared at blocks 0, 8, 16, 24, 26")
    var taps = minimax_h3_vision_deepstack_tap_blocks()
    # Tolerance grows with depth because the activations do; see the header.
    var block_ids = [0, 8, 16, 24, H3_VIS_DEPTH - 1]

    for layer in range(max_blocks):
        var tb0 = perf_counter_ns()
        hidden = minimax_h3_vision_block_forward(
            hidden, num_patches, layer, rotary, cu, weights
        )
        for t in range(len(taps)):
            if taps[t] == layer:
                var feat = minimax_h3_vision_merger(
                    hidden, num_patches,
                    _VISION_PREFIX + "deepstack_merger_list." + String(t) + ".",
                    True, weights,
                )
                var ds_name = String("deepstack_") + String(t)
                _compare(
                    report, ds_name, feat, _read_f32(st, ds_name),
                    _stage_tol(st, ds_name, Float32(4.0)),
                )
        for b in range(len(block_ids)):
            if block_ids[b] == layer:
                var bname = String("after_block_") + String(layer)
                _compare(
                    report, bname, hidden, _read_f32(st, bname),
                    _stage_tol(st, bname, Float32(4.0)),
                )
        var tb1 = perf_counter_ns()
        if layer == 0:
            print("      (block 0 took", Float64(tb1 - tb0) / 1.0e9, "s; x",
                  max_blocks, "blocks)")

    # ── [5] the final merger ────────────────────────────────────────────────
    print("")
    if max_blocks == H3_VIS_DEPTH:
        print("[5] final merger (PRESHUFFLE norm, exact-erf GELU)")
        var embeds = minimax_h3_vision_merger(
            hidden, num_patches, _VISION_PREFIX + "merger.", False, weights
        )
        _compare(
            report, "embeds", embeds, _read_f32(st, String("embeds")),
            _stage_tol(st, String("embeds"), Float32(4.0)),
        )
        if len(embeds) == tokens * H3_VIS_OUT_HIDDEN:
            report.ok(
                "embeds shape",
                String(tokens) + " x " + String(H3_VIS_OUT_HIDDEN)
                + " (the tower's contract with the conditioner)",
            )
        else:
            report.fail("embeds shape", String(len(embeds)))
    else:
        print("[5] final merger SKIPPED — partial run (", max_blocks, "of",
              H3_VIS_DEPTH, "blocks).")
        print("    A partial stack's merger output is NOT the model's embeds and")
        print("    is deliberately not compared against one.")

    _ = H3_VIS_HIDDEN
    _ = H3_VIS_NUM_DEEPSTACK
    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_qwen3vl_vision_forward probe FAILED")
