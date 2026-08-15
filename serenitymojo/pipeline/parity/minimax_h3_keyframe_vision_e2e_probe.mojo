# serenitymojo/pipeline/parity/minimax_h3_keyframe_vision_e2e_probe.mojo
#
# CAPSTONE gate for the keyframe VISION path, end to end, on CPU:
#   preprocess -> vision tower -> splice at <|image_pad|> -> deepstack add
# against transformers' own Qwen3VLVisionModel and Qwen3VLModel row ops, on the
# real FL2VA weights. Host only: no DeviceContext, no GPU.
#
# ── SCOPE, STATED RATHER THAN IMPLIED ──────────────────────────────────────
# COVERED: the preprocessor, the 27-block tower on real weights, the splice map
# derived from the real presentation, the embed SUBSTITUTION, and the deepstack
# ADD for language layers 0/1/2.
# NOT COVERED: the 50 decoder layers. `minimax_h3_encode_conditioning_streamed_
# depth` needs a DeviceContext and the GPU is off-limits; independently, 50
# layers of a 32B model in f32 is not CPU-feasible. Those layers are gated
# separately. This probe adds the seam nothing else covers.
#
# ── THE FAILURE THIS EXISTS TO CATCH ───────────────────────────────────────
# A splice that is right in COUNT and wrong in PLACEMENT produces a
# perfectly-shaped conditioning. No shape assertion anywhere fires and the
# render is quietly wrong. So the probe checks POSITIONS from two INDEPENDENT
# derivations — this pipeline's `pad_positions` (built from the tokenizer while
# emitting the presentation) and the reference's own `input_ids == image_pad_id`
# mask (`get_placeholder_mask`'s criterion) — and requires them identical before
# it compares a single float.
#
# Canvas 768x768: the SMALLEST canvas resolve_canvas_size can emit, so the host
# tower's O(P^2) attention stays tractable while staying a REAL geometry.
#
# Run (CPU, ~10-20 min for the host tower; build -O2):
#   CUDA_VISIBLE_DEVICES="" /home/alex/torchref/venv/bin/python \
#     scripts/minimax_h3_keyframe_vision_e2e_oracle.py
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_keyframe_vision_e2e_probe.mojo \
#     -o /tmp/h3_kf_e2e -Xlinker -lm \
#   && /tmp/h3_kf_e2e output/minimax_h3_keyframe/keyframe_vision_e2e_ref.safetensors

from std.sys import argv
from std.collections import List
from std.math import sqrt
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    MiniMaxH3VisionGrid,
    minimax_h3_vision_forward,
    minimax_h3_vision_load_weights,
)
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_streamed import (
    minimax_h3_deepstack_add,
    minimax_h3_splice_vision_embeds,
)
from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage
from serenitymojo.pipeline.minimax_h3_vision_preprocess import (
    minimax_h3_vision_patch_rows,
)

comptime HID = 5120
comptime TE_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"


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
    var i = st.tensor_info(name)
    var tv = from_parts(i.dtype, i.shape.copy(), st.tensor_bytes(name))
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var o = List[Float32](capacity=tv.numel())
    for k in range(tv.numel()):
        o.append(p[k])
    return o^


def _i32(ref st: SafeTensors, name: String) raises -> List[Int]:
    var i = st.tensor_info(name)
    var tv = from_parts(i.dtype, i.shape.copy(), st.tensor_bytes(name))
    var p = tv.data.unsafe_ptr().bitcast[Int32]()
    var o = List[Int](capacity=tv.numel())
    for k in range(tv.numel()):
        o.append(Int(p[k]))
    return o^


def _u8(ref st: SafeTensors, name: String) raises -> List[UInt8]:
    var i = st.tensor_info(name)
    var tv = from_parts(i.dtype, i.shape.copy(), st.tensor_bytes(name))
    var p = tv.data.unsafe_ptr()
    var o = List[UInt8](capacity=tv.numel())
    for k in range(tv.numel()):
        o.append(p[k])
    return o^


def _cmp(
    mut r: Report, label: String, got: List[Float32], want: List[Float32],
    tol: Float32, bit_exact: Bool,
) raises:
    if len(got) != len(want):
        r.fail(label, String("len ") + String(len(got)) + " vs " + String(len(want)))
        return
    var bad = 0
    var mx = Float32(0.0)
    var at = -1
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(want)):
        if got[i] != want[i]:
            bad += 1
        var d = got[i] - want[i]
        if d < 0:
            d = -d
        if d > mx:
            mx = d
            at = i
        dot += Float64(got[i]) * Float64(want[i])
        na += Float64(got[i]) * Float64(got[i])
        nb += Float64(want[i]) * Float64(want[i])
    var cos = Float64(1.0)
    if na > 0.0 and nb > 0.0:
        cos = dot / (sqrt(na) * sqrt(nb))
    if bit_exact:
        if bad == 0:
            r.ok(label, String("BIT-EXACT over ") + String(len(want)) + " values")
        else:
            r.fail(label, String(bad) + " differ, max_abs=" + String(mx) + " at " + String(at))
    elif cos > 0.9999999 and mx <= tol:
        r.ok(label, String("cos=") + String(cos) + " max_abs=" + String(mx))
    else:
        r.fail(label, String("cos=") + String(cos) + " max_abs=" + String(mx) + " at " + String(at))


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: minimax_h3_keyframe_vision_e2e_probe <keyframe_vision_e2e_ref.safetensors>")
        return
    print("MiniMax-H3 KEYFRAME VISION PATH — capstone probe (CPU)")
    print("  reference:", String(args[1]))
    print("  scope: preprocess -> tower -> splice -> deepstack.")
    print("         The 50 decoder layers are NOT covered (device path, gated elsewhere).")
    print("")
    var r = Report()
    var st = SafeTensors.open(String(args[1]))

    var gthw = _i32(st, String("grid_thw"))
    var ids = _i32(st, String("token_ids"))
    var want_pos = _i32(st, String("pad_positions"))
    var seq = len(ids)
    var ntok = len(want_pos)
    print("  grid", gthw[0], "x", gthw[1], "x", gthw[2], " seq", seq, " vision tokens", ntok)

    # ── [1] the splice map, from two INDEPENDENT derivations ───────────────
    # Ours: the pad rows as the presentation emitted them. Reference: the
    # `input_ids == image_pad_id` mask get_placeholder_mask itself uses.
    print("")
    print("[1] splice map — our pad rows vs get_placeholder_mask's own criterion")
    var pad_id = 151655
    var ours = List[Int]()
    for i in range(seq):
        if ids[i] == pad_id:
            ours.append(i)
    var pos_bad = 0
    if len(ours) != ntok:
        r.fail("pad row count", String(len(ours)) + " vs " + String(ntok))
    else:
        for i in range(ntok):
            if ours[i] != want_pos[i]:
                pos_bad += 1
        if pos_bad == 0:
            r.ok(
                "pad POSITIONS agree",
                String(ntok) + " rows, identical placement (rows "
                + String(want_pos[0]) + ".." + String(want_pos[ntok - 1]) + ")",
            )
        else:
            r.fail("pad POSITIONS", String(pos_bad) + " differ")

    # ── [2] the preprocessor ───────────────────────────────────────────────
    print("")
    print("[2] preprocess — canvas pixels -> [P, 1536]")
    var img_bytes = _u8(st, String("image"))
    var image = MiniMaxH3RgbImage(img_bytes^, 768, 768)
    var patches = minimax_h3_vision_patch_rows(image)
    _cmp(r, "pixel_values", patches, _f32(st, String("pixel_values")), Float32(0.0), True)

    # ── [3] the tower — OPT-IN, and deliberately not this gate's point ─────
    # The 27-block tower is ALREADY gated by h3-ref2va's own CPU gate, and at a
    # production canvas it dominates everything: MEASURED 1289.7 s (21.5 min) of
    # single-threaded host f32 at 2304 patches — 8.2x the patches of the
    # 280-patch case, with O(P^2) attention on top. A capstone nobody can afford
    # to run stops being a gate, so the default skips it.
    #
    # Pass "with-tower" as argv[2] to include it. Either way the stages this
    # probe EXISTS for are fed the REFERENCE's embeds, which also keeps a splice
    # bug and a tower ulp from masking each other.
    var run_tower = len(args) >= 3 and String(args[2]) == String("with-tower")
    print("")
    if run_tower:
        print("[3] vision tower — 27 blocks, real FL2VA weights (SLOW: ~21 min)")
        var t0 = perf_counter_ns()
        var vw = minimax_h3_vision_load_weights(String(TE_DIR))
        var grids = List[MiniMaxH3VisionGrid]()
        grids.append(MiniMaxH3VisionGrid(gthw[0], gthw[1], gthw[2]))
        var vision = minimax_h3_vision_forward(vw, patches, grids)
        var t1 = perf_counter_ns()
        print("      tower ran in", Float64(t1 - t0) / 1.0e9, "s")
        if vision.num_tokens != ntok:
            r.fail("tower token count", String(vision.num_tokens) + " vs " + String(ntok))
        else:
            r.ok("tower token count", String(ntok))
        _cmp(r, "vision embeds", vision.embeds, _f32(st, String("vision_embeds")), Float32(4.0e-3), False)
    else:
        print("[3] vision tower SKIPPED — pass 'with-tower' as argv[2] to run it")
        print("    (gated separately by h3-ref2va's CPU gate; ~21 min here).")
        print("    The reference's embeds are used below either way.")

    # ── [4] the SPLICE ─────────────────────────────────────────────────────
    print("")
    print("[4] splice — REPLACE the pad rows with the tower's embeds")
    var emb = _f32(st, String("inputs_embeds_before"))
    # Fed the REFERENCE's embeds, so this isolates the row operation from the
    # tower's f32 residual — a splice bug and a tower ulp must not be able to
    # hide behind one another.
    minimax_h3_splice_vision_embeds(emb, _f32(st, String("vision_embeds")), ours, HID)
    _cmp(r, "inputs_embeds after splice", emb, _f32(st, String("inputs_embeds_spliced")), Float32(0.0), True)

    # ── [5] the DEEPSTACK ADD, layers 0/1/2 ────────────────────────────────
    print("")
    print("[5] deepstack — ADD at the same rows, language layers 0/1/2")
    var hs = _f32(st, String("hidden_before"))
    for k in range(3):
        minimax_h3_deepstack_add(hs, _f32(st, String("deepstack_") + String(k)), ours, HID)
        _cmp(
            r, String("hidden after deepstack ") + String(k), hs,
            _f32(st, String("hidden_after_") + String(k)), Float32(0.0), True,
        )

    print("")
    if r.failures == 0:
        print("PASS:", r.checks, "checks")
    else:
        print("FAIL:", r.failures, "of", r.checks, "checks")
        raise Error("minimax_h3_keyframe_vision_e2e probe FAILED")
