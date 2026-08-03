# serenitymojo/models/text_encoder/parity/minimax_h3_deepstack_gate.mojo
#
# Qwen3-VL DEEPSTACK INJECTION mechanics gate — CPU ONLY, no Tensor, no
# DeviceContext, no GPU touched anywhere in this file (it does not even
# import std.gpu.host). Loads the CPU float32 oracle dump
# (scripts/minimax_h3_deepstack_oracle.py, itself run against the REAL
# MiniMax-H3/FL2VA/text_encoder checkpoint on 3 real decoder layers) and
# checks the injection helpers added to minimax_h3_qwen3vl_streamed.mojo:
#   minimax_h3_mm_token_type_ids / minimax_h3_visual_positions /
#   minimax_h3_splice_vision_embeds / minimax_h3_deepstack_add.
#
# Ground truth (see minimax_h3_qwen3vl_streamed.mojo's DEEPSTACK INJECTION
# header for the full vendor citation): transformers 4.57.6
# `Qwen3VLTextModel.forward`/`_deepstack_process`
# (modeling_qwen3_vl.py:849-883) ADDS each deepstack tap at VISUAL-TOKEN
# POSITIONS ONLY, AFTER decoder layers 0/1/2 finish their own forward.
#
# CHECKS (all run, no early exit — every check prints its own result; the
# gate raises at the end if any failed):
#   A. minimax_h3_mm_token_type_ids(in.input_ids, IMAGE_TOKEN_ID,
#      VIDEO_TOKEN_ID) == in.mm_token_type_ids EXACTLY (integers).
#   B. len(minimax_h3_visual_positions(...)) == N, where N is the row count
#      of in.vision_embeds.
#   C. minimax_h3_splice_vision_embeds, applied to a ZERO embedding stream,
#      reproduces in.inputs_embeds at the VISUAL ROWS EXACTLY (bit-exact —
#      it is a copy, not arithmetic; non-visual rows are not compared here,
#      they are the oracle's real text embeddings, which a zero-based host
#      buffer cannot and does not need to reproduce for this check).
#   C2. minimax_h3_deepstack_add, applied to a ZERO hidden-state buffer with
#      in.deepstack's tap-0 block, in COMPLETE ISOLATION (no LM involved at
#      all): the visual rows must equal the tap EXACTLY (0 + tap = tap), and
#      EVERY other row must remain EXACTLY 0.0 untouched. This is the direct
#      unit test of the masked-add itself, independent of any attention
#      propagation questions (which is why it can assert exact-zero at
#      ALL non-visual rows, not just the pre-vision subset D1 uses).
#   D. Host-side comparison of out.hidden_02 vs out.hidden_02_nodeep (both
#      already produced by the oracle's own real-weight 3-layer run):
#        D1. EXACTLY 0.0 at PRE-VISION positions (indices strictly before
#            the first visual position). This is the real, always-valid
#            invariant — a port that (incorrectly) added the deepstack taps
#            to EVERY row, not just the visual ones, would show a nonzero
#            diff here too, which is exactly what this check catches.
#        D2. CLEARLY non-zero at visual positions (the injection itself).
#        Also reported, NOT gated as pass/fail: max_abs at POST-VISION
#        non-visual positions is expected to be clearly non-zero too — that
#        is legitimate causal-self-attention propagation from the modified
#        visual rows in decoder layers 1/2 reading them, not a defect in
#        `_deepstack_process` (which only ever assigns to the rows selected
#        by `visual_pos_masks` — see the oracle's own MEASURED FINDING
#        header in scripts/minimax_h3_deepstack_oracle.py for the full
#        derivation and the same three numbers re-measured independently
#        here).
#   E. Full 3-layer Mojo streamed-decode-on-real-weights vs out.hidden_00/
#      out.hidden_01/out.hidden_02 (cos >= 0.999 each): NOT ATTEMPTED. This
#      is a CPU-only gate by hard requirement (no DeviceContext may be
#      constructed anywhere in this task), and
#      minimax_h3_encode_conditioning_streamed_depth's every Tensor op
#      (embedding gather, RoPE table upload, attention, MLP, the
#      to_host/from_host round-trips the deepstack wiring itself adds)
#      requires a `DeviceContext` to run. This file does not import
#      std.gpu.host at all, so E genuinely cannot run here — the
#      LAYER-ARITHMETIC end-to-end (real Qwen3-VL attention/MLP math wired
#      together with the injection, as opposed to just the injection
#      mechanics A-D above) remains UNGATED. This is stated explicitly, not
#      papered over: A-D fully cover the splice/mask/add MECHANICS this
#      change adds; they do not exercise Qwen3Encoder's own layer forward.
#
# Oracle: CUDA_VISIBLE_DEVICES="" python3 scripts/minimax_h3_deepstack_oracle.py <ref.safetensors>
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/text_encoder/parity/minimax_h3_deepstack_gate.mojo \
#     -o <scratch>/dsgate
#   <scratch>/dsgate <scratch>/deepstack_ref.safetensors

from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_streamed import (
    H3_HIDDEN,
    minimax_h3_mm_token_type_ids,
    minimax_h3_visual_positions,
    minimax_h3_splice_vision_embeds,
    minimax_h3_deepstack_add,
)

comptime DEFAULT_REF = "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/scratchpad/deepstack_ref.safetensors"

# MiniMax-H3/FL2VA/text_encoder/config.json's real special-token ids
# (top-level config: image_token_id / video_token_id), same values the
# oracle script hardcodes.
comptime IMAGE_TOKEN_ID = 151655
comptime VIDEO_TOKEN_ID = 151656


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


def _load_i32(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I32:
        raise Error(String("_load_i32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int32]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _load_shape(ref st: SafeTensors, name: String) raises -> List[Int]:
    return st.tensor_info(name).shape.copy()


def _to_f32(v: List[Int]) -> List[Float32]:
    var out = List[Float32]()
    out.reserve(len(v))
    for i in range(len(v)):
        out.append(Float32(v[i]))
    return out^


def _slice_block(deepstack: List[Float32], block_index: Int, n: Int, hidden: Int) raises -> List[Float32]:
    """One `[n, hidden]` tap block out of `in.deepstack`'s `[3, n, hidden]`
    concatenation (block_index in 0..2)."""
    var stride = n * hidden
    var start = block_index * stride
    if start + stride > len(deepstack):
        raise Error("_slice_block: block_index/n/hidden out of range for deepstack buffer")
    var out = List[Float32]()
    out.reserve(stride)
    for i in range(stride):
        out.append(deepstack[start + i])
    return out^


def _select_rows(v: List[Float32], rows: List[Int], hidden: Int) raises -> List[Float32]:
    var out = List[Float32]()
    out.reserve(len(rows) * hidden)
    for ri in range(len(rows)):
        var base = rows[ri] * hidden
        for j in range(hidden):
            out.append(v[base + j])
    return out^


def _region_stats(
    a: List[Float32], b: List[Float32], rows: List[Int], hidden: Int, cos_threshold: Float64 = 0.999
) raises -> ParityResult:
    """cos + max_abs over `rows` only (degenerate cos=1.0/max_abs=0.0 for an
    empty row set, treated as trivially matching). `cos_threshold` only
    affects the printed `ParityResult.passed` field's own cos-based verdict —
    this file's actual PASS/FAIL decisions are made separately (on max_abs)
    by the `Report` methods below, never on `.passed` directly. Regions that
    are EXPECTED to differ (D2, the post-vision informational note) pass
    threshold=-1.0 so the printed struct does not show a self-contradictory
    'FAIL' next to this file's own 'ok'."""
    if len(rows) == 0:
        return ParityResult(cos=1.0, max_abs=0.0, passed=True, n=0)
    var sa = _select_rows(a, rows, hidden)
    var sb = _select_rows(b, rows, hidden)
    return ParityHarness(cos_threshold).compare_host(sa, sb)


struct Report(Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def exact_int(mut self, label: String, got: List[Int], want: List[Int]) raises:
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
        # Also report cos+max_abs (cast to float) for uniform reporting
        # alongside the exact-integer verdict, per the task's "report cos
        # AND max_abs for each" instruction.
        var stats = ParityHarness(0.999).compare_host(_to_f32(got), _to_f32(want))
        if bad == 0:
            print("  ok  ", label, "exact over", len(got), "values —", stats)
        else:
            self.failures += 1
            print(
                "  FAIL", label, bad, "of", len(got), "differ; first at", first,
                "got", got[first], "want", want[first], "—", stats,
            )

    def count_eq(mut self, label: String, got: Int, want: Int):
        self.checks += 1
        var diff = got - want
        if diff < 0:
            diff = -diff
        if got == want:
            print("  ok  ", label, "=", got, "(max_abs diff =", diff, ")")
        else:
            self.failures += 1
            print("  FAIL", label, "got", got, "want", want, "(max_abs diff =", diff, ")")

    def exact_rows(
        mut self,
        label: String,
        got: List[Float32],
        want: List[Float32],
        rows: List[Int],
        hidden: Int,
    ) raises:
        """Bit-exact comparison restricted to `rows` — check C, which only
        claims exactness at the visual rows (a plain row copy)."""
        self.checks += 1
        var bad = 0
        var first = -1
        for ri in range(len(rows)):
            var base = rows[ri] * hidden
            for j in range(hidden):
                if got[base + j] != want[base + j]:
                    bad += 1
                    if first < 0:
                        first = base + j
        var stats = _region_stats(got, want, rows, hidden)
        if bad == 0:
            print(
                "  ok  ", label, "bit-exact over", len(rows), "rows x", hidden,
                "=", len(rows) * hidden, "values —", stats,
            )
        else:
            self.failures += 1
            print(
                "  FAIL", label, bad, "values differ across", len(rows),
                "rows; first flat index", first, "—", stats,
            )

    def region_eq_zero(mut self, label: String, stats: ParityResult):
        self.checks += 1
        if stats.max_abs == 0.0:
            print("  ok  ", label, "—", stats, "(max_abs exactly 0.0, as required)")
        else:
            self.failures += 1
            print("  FAIL", label, "—", stats, "(expected max_abs exactly 0.0)")

    def region_gt_floor(mut self, label: String, stats: ParityResult, floor: Float64):
        self.checks += 1
        if stats.max_abs > floor:
            print("  ok  ", label, "—", stats, "(clearly >", floor, ")")
        else:
            self.failures += 1
            print("  FAIL", label, "—", stats, "(NOT clearly >", floor, ")")

    def bit_exact_count(mut self, label: String, bad: Int, total: Int, first_bad: Int):
        self.checks += 1
        if bad == 0:
            print("  ok  ", label, "bit-exact over", total, "values")
        else:
            self.failures += 1
            print("  FAIL", label, bad, "of", total, "differ; first flat index", first_bad)

    def informational(self, label: String, stats: ParityResult):
        """Reported, NOT gated pass/fail — see file header for why (D's
        post-vision propagation number)."""
        print("  info", label, "—", stats, "(informational only, not gated)")


def main() raises:
    var args = argv()
    var ref_path = String(DEFAULT_REF)
    if len(args) > 1:
        ref_path = String(args[1])

    print("MiniMax-H3 Qwen3-VL DEEPSTACK INJECTION gate (CPU-only)")
    print("  reference:", ref_path)
    var st = SafeTensors.open(ref_path)
    var report = Report()

    var input_ids = _load_i32(st, "in.input_ids")
    var mm_ref = _load_i32(st, "in.mm_token_type_ids")
    var vision_embeds = _load_f32(st, "in.vision_embeds")
    var vision_embeds_shape = _load_shape(st, "in.vision_embeds")
    var deepstack = _load_f32(st, "in.deepstack")
    var inputs_embeds = _load_f32(st, "in.inputs_embeds")
    var hidden_02 = _load_f32(st, "out.hidden_02")
    var hidden_02_nodeep = _load_f32(st, "out.hidden_02_nodeep")

    var seq = len(input_ids)
    var n_expected = vision_embeds_shape[0]

    # ── A: mm_token_type_ids ────────────────────────────────────────────────
    print("[A] minimax_h3_mm_token_type_ids vs in.mm_token_type_ids")
    var mm_got = minimax_h3_mm_token_type_ids(input_ids, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID)
    report.exact_int("mm_token_type_ids", mm_got, mm_ref)

    # ── B: visual position count ─────────────────────────────────────────────
    print("[B] minimax_h3_visual_positions count vs N (in.vision_embeds rows)")
    var visual_positions = minimax_h3_visual_positions(mm_got)
    report.count_eq("visual_positions.count", len(visual_positions), n_expected)

    # ── C: splice reproduces in.inputs_embeds at the visual rows EXACTLY ────
    print("[C] minimax_h3_splice_vision_embeds (zero stream) vs in.inputs_embeds at visual rows")
    var zero_stream = List[Float32]()
    zero_stream.reserve(seq * H3_HIDDEN)
    for _ in range(seq * H3_HIDDEN):
        zero_stream.append(Float32(0.0))
    minimax_h3_splice_vision_embeds(zero_stream, vision_embeds, visual_positions, H3_HIDDEN)
    report.exact_rows("splice.visual_rows", zero_stream, inputs_embeds, visual_positions, H3_HIDDEN)

    # ── C2: minimax_h3_deepstack_add in complete isolation (no LM) ───────────
    print("[C2] minimax_h3_deepstack_add (zero stream + in.deepstack tap 0), isolated")
    var block0 = _slice_block(deepstack, 0, n_expected, H3_HIDDEN)
    var zero_stream2 = List[Float32]()
    zero_stream2.reserve(seq * H3_HIDDEN)
    for _ in range(seq * H3_HIDDEN):
        zero_stream2.append(Float32(0.0))
    minimax_h3_deepstack_add(zero_stream2, block0, visual_positions, H3_HIDDEN)

    var is_visual = List[Bool]()
    is_visual.resize(seq, False)
    for i in range(len(visual_positions)):
        is_visual[visual_positions[i]] = True
    var all_nonvisual_rows = List[Int]()
    for i in range(seq):
        if not is_visual[i]:
            all_nonvisual_rows.append(i)

    var bad_add = 0
    var first_bad_add = -1
    for k in range(len(visual_positions)):
        var pos = visual_positions[k]
        var dst0 = pos * H3_HIDDEN
        var src0 = k * H3_HIDDEN
        for j in range(H3_HIDDEN):
            if zero_stream2[dst0 + j] != block0[src0 + j]:
                bad_add += 1
                if first_bad_add < 0:
                    first_bad_add = dst0 + j
    report.bit_exact_count(
        "deepstack_add.isolated.visual_rows_eq_tap", bad_add,
        len(visual_positions) * H3_HIDDEN, first_bad_add,
    )

    var bad_untouched = 0
    var first_bad_untouched = -1
    for ri in range(len(all_nonvisual_rows)):
        var base = all_nonvisual_rows[ri] * H3_HIDDEN
        for j in range(H3_HIDDEN):
            if zero_stream2[base + j] != Float32(0.0):
                bad_untouched += 1
                if first_bad_untouched < 0:
                    first_bad_untouched = base + j
    report.bit_exact_count(
        "deepstack_add.isolated.nonvisual_rows_untouched", bad_untouched,
        len(all_nonvisual_rows) * H3_HIDDEN, first_bad_untouched,
    )

    # ── D: host-side hidden_02 vs hidden_02_nodeep, by region ────────────────
    print("[D] out.hidden_02 vs out.hidden_02_nodeep, by region")
    if len(visual_positions) == 0:
        raise Error("minimax_h3_deepstack_gate: no visual positions found — cannot partition regions")
    var first_visual = visual_positions[0]
    var last_visual = visual_positions[0]
    for i in range(len(visual_positions)):
        if visual_positions[i] < first_visual:
            first_visual = visual_positions[i]
        if visual_positions[i] > last_visual:
            last_visual = visual_positions[i]

    var pre_vision_rows = List[Int]()
    for i in range(first_visual):
        pre_vision_rows.append(i)
    var post_vision_rows = List[Int]()
    for i in range(last_visual + 1, seq):
        post_vision_rows.append(i)

    var d1_stats = _region_stats(hidden_02, hidden_02_nodeep, pre_vision_rows, H3_HIDDEN)
    report.region_eq_zero("D1 pre-vision (must be exactly 0.0 — the real invariant)", d1_stats)

    var d2_stats = _region_stats(hidden_02, hidden_02_nodeep, visual_positions, H3_HIDDEN, -1.0)
    report.region_gt_floor("D2 visual (must be clearly non-zero)", d2_stats, 1e-6)

    var post_stats = _region_stats(hidden_02, hidden_02_nodeep, post_vision_rows, H3_HIDDEN, -1.0)
    report.informational(
        "D-note post-vision non-visual (expected non-zero: causal-attention propagation, NOT a bug)",
        post_stats,
    )

    # ── E: full 3-layer end-to-end vs out.hidden_00/01/02 ────────────────────
    print("[E] full 3-layer Mojo streamed-decode-on-real-weights vs out.hidden_00/01/02")
    print(
        "  NOT ATTEMPTED: this gate is CPU-only by hard requirement (no"
        " DeviceContext may be constructed anywhere in this task), and"
        " minimax_h3_encode_conditioning_streamed_depth's every Tensor op"
        " requires one. This file does not import std.gpu.host at all. The"
        " layer-arithmetic end-to-end (real attention/MLP math wired"
        " together with the injection) remains UNGATED — A-D above fully"
        " cover the splice/mask/add MECHANICS, not Qwen3Encoder's own layer"
        " forward."
    )

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks, all as required (E not attempted — see [E] above)")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks failed")
        raise Error("MiniMax-H3 deepstack injection gate failed")
