# serenitymojo/models/text_encoder/parity/minimax_h3_qwen3vl_vision_probe.mojo
#
# HOST probe for the Qwen3-VL vision tower's GEOMETRY — the half of the tower
# that needs no weights and no GPU, and the half most likely to be wrong in a
# way that shapes still line up for.
#
# Gates `models/text_encoder/minimax_h3_qwen3vl_vision.mojo`: cu_seqlens
# (per-FRAME segmentation), the 2-D rotary coordinates in MERGE-BLOCK order,
# the rotary inv_freq table, the bilinear position-embed interpolation
# (indices AND weights), token counts, the deepstack tap/inject index spaces,
# and the 351-name tensor manifest.
#
# Expected values were produced by RUNNING THE VENDOR'S OWN CODE — the exact
# expressions from transformers 4.57.6
# models/qwen3_vl/modeling_qwen3_vl.py (rot_pos_emb :603-641, cu_seqlens :728,
# Qwen3VLVisionRotaryEmbedding :79-91, fast_pos_embed_interpolate :642-679)
# transcribed into a standalone torch script, with no Qwen3VLVisionModel
# constructed (the weights do not exist yet). Comparison is exact.
#
# NOT gated here: the weighted forward. All 351 vision tensors live in
# text_encoder/model-00014-of-00014.safetensors, which has not downloaded.
# Its gate is minimax_h3_qwen3vl_vision_gate.mojo, marked PENDING-GPU.
#
# Run (no GPU, no weights):
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/text_encoder/parity/minimax_h3_qwen3vl_vision_probe.mojo \
#     -o <scratch>/minimax_h3_qwen3vl_vision_probe \
#   && <scratch>/minimax_h3_qwen3vl_vision_probe

from std.collections import List

from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    H3_VIS_DEPTH,
    H3_VIS_GRID_PER_SIDE,
    H3_VIS_HEAD_DIM,
    H3_VIS_HIDDEN,
    H3_VIS_MERGED_WIDTH,
    H3_VIS_NUM_DEEPSTACK,
    H3_VIS_OUT_HIDDEN,
    H3_VIS_PATCH_NUMEL,
    H3_VIS_ROTARY_DIM,
    MiniMaxH3VisionGrid,
    minimax_h3_vision_cu_seqlens,
    minimax_h3_vision_deepstack_lm_layers,
    minimax_h3_vision_deepstack_tap_blocks,
    minimax_h3_vision_param_count,
    minimax_h3_vision_pos_embed_interpolation,
    minimax_h3_vision_rotary_inv_freq,
    minimax_h3_vision_rotary_positions,
    minimax_h3_vision_tensor_names,
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

    def eq_int(mut self, label: String, got: Int, want: Int):
        if got == want:
            self.ok(label, String(got))
        else:
            self.fail(label, String("got ") + String(got) + ", want " + String(want))

    def eq_ints(mut self, label: String, got: List[Int], want: List[Int]):
        if len(got) != len(want):
            self.fail(
                label,
                String("length ") + String(len(got)) + " != " + String(len(want)),
            )
            return
        var bad = 0
        var first = -1
        for i in range(len(want)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            self.ok(label, String("exact over ") + String(len(want)) + " values")
        else:
            self.fail(
                label,
                String(bad) + " differ; first at " + String(first) + ": got "
                + String(got[first]) + ", want " + String(want[first]),
            )

    def eq_f32(mut self, label: String, got: List[Float32], want: List[Float32]):
        if len(got) != len(want):
            self.fail(
                label,
                String("length ") + String(len(got)) + " != " + String(len(want)),
            )
            return
        var bad = 0
        var first = -1
        for i in range(len(want)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            self.ok(
                label, String("bit-exact over ") + String(len(want)) + " float32 values"
            )
        else:
            self.fail(
                label,
                String(bad) + " differ; first at " + String(first) + ": got "
                + String(got[first]) + ", want " + String(want[first]),
            )

    def eq_f64(mut self, label: String, got: List[Float64], want: List[Float64]):
        if len(got) != len(want):
            self.fail(
                label,
                String("length ") + String(len(got)) + " != " + String(len(want)),
            )
            return
        var bad = 0
        var first = -1
        for i in range(len(want)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            self.ok(
                label, String("bit-exact over ") + String(len(want)) + " float64 values"
            )
        else:
            self.fail(
                label,
                String(bad) + " differ; first at " + String(first) + ": got "
                + String(got[first]) + ", want " + String(want[first]),
            )


def main() raises:
    print("MiniMax-H3 Qwen3-VL VISION TOWER geometry probe")
    print("")
    var report = Report()

    # ── [1] config-derived constants ─────────────────────────────────────────
    print("[1] architecture constants (Ref2VA text_encoder/config.json)")
    report.eq_int("depth", H3_VIS_DEPTH, 27)
    report.eq_int("hidden", H3_VIS_HIDDEN, 1152)
    report.eq_int("head_dim", H3_VIS_HEAD_DIM, 72)
    report.eq_int("rotary dim (head_dim // 2)", H3_VIS_ROTARY_DIM, 36)
    report.eq_int("patch numel (3*2*16*16)", H3_VIS_PATCH_NUMEL, 1536)
    report.eq_int("merged width (1152*4)", H3_VIS_MERGED_WIDTH, 4608)
    report.eq_int("out hidden", H3_VIS_OUT_HIDDEN, 5120)
    report.eq_int("grid per side (sqrt 2304)", H3_VIS_GRID_PER_SIDE, 48)

    # ── [2] deepstack: two DIFFERENT index spaces ────────────────────────────
    print("")
    print("[2] deepstack tap blocks vs injection layers (different index spaces)")
    report.eq_ints(
        "vision tap blocks", minimax_h3_vision_deepstack_tap_blocks(), [8, 16, 24]
    )
    report.eq_ints(
        "language inject layers", minimax_h3_vision_deepstack_lm_layers(), [0, 1, 2]
    )
    report.eq_int("deepstack count", H3_VIS_NUM_DEEPSTACK, 3)

    # ── [3] cu_seqlens: ONE SEGMENT PER FRAME ────────────────────────────────
    print("")
    print("[3] cu_seqlens — per-FRAME segmentation, not per-image")
    var grids = List[MiniMaxH3VisionGrid]()
    grids.append(MiniMaxH3VisionGrid(2, 4, 6))
    report.eq_ints("cu_seqlens t=2 h=4 w=6", minimax_h3_vision_cu_seqlens(grids), [0, 24, 48])

    # A realistic 16:9 video reference: 2 blocks at 48x84 patches.
    var real = List[MiniMaxH3VisionGrid]()
    real.append(MiniMaxH3VisionGrid(2, 48, 84))
    report.eq_ints(
        "cu_seqlens t=2 h=48 w=84", minimax_h3_vision_cu_seqlens(real), [0, 4032, 8064]
    )
    report.eq_int("tokens per block (48*84/4)", real[0].tokens_per_block(), 1008)
    report.eq_int("tokens total (2 blocks)", real[0].num_tokens(), 2016)

    # ── [4] rotary coordinates in MERGE-BLOCK order ──────────────────────────
    print("")
    print("[4] rotary (row, col) — merge-block order, repeated per frame")
    var want_coords: List[Int] = [
        0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3, 0, 4, 0, 5, 1, 4, 1, 5,
        2, 0, 2, 1, 3, 0, 3, 1, 2, 2, 2, 3, 3, 2, 3, 3, 2, 4, 2, 5, 3, 4, 3, 5,
        0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3, 0, 4, 0, 5, 1, 4, 1, 5,
        2, 0, 2, 1, 3, 0, 3, 1, 2, 2, 2, 3, 3, 2, 3, 3, 2, 4, 2, 5, 3, 4, 3, 5,
    ]
    report.eq_ints(
        "rotary coords t=2 h=4 w=6", minimax_h3_vision_rotary_positions(grids), want_coords
    )

    # ── [5] rotary inv_freq ──────────────────────────────────────────────────
    print("")
    print("[5] rotary inv_freq — 18 entries, FLOAT32 (vendor dtype=torch.float)")
    var want_inv: List[Float32] = [
        Float32(1.0), Float32(0.5994842052459717), Float32(0.35938137769699097),
        Float32(0.2154434472322464), Float32(0.1291549652814865),
        Float32(0.07742635905742645), Float32(0.04641588404774666),
        Float32(0.027825593948364258), Float32(0.01668100617825985),
        Float32(0.009999999776482582), Float32(0.005994841456413269),
        Float32(0.0035938138607889414), Float32(0.002154434332624078),
        Float32(0.001291549764573574), Float32(0.0007742635789327323),
        Float32(0.00046415894757956266), Float32(0.00027825593133457005),
        Float32(0.00016681010311003774),
    ]
    report.eq_f32("inv_freq (f32, vendor-exact)", minimax_h3_vision_rotary_inv_freq(), want_inv)

    # ── [6] position-embed bilinear interpolation ────────────────────────────
    print("")
    print("[6] fast_pos_embed_interpolate — 4-corner indices and weights, h=3 w=5")
    var interp_grids = List[MiniMaxH3VisionGrid]()
    interp_grids.append(MiniMaxH3VisionGrid(1, 3, 5))
    var interp = minimax_h3_vision_pos_embed_interpolation(interp_grids)
    report.eq_int("interpolation count", interp.count, 15)

    var want_idx: List[Int] = [
        0, 11, 23, 35, 47, 1104, 1115, 1127, 1139, 1151, 2256, 2267, 2279, 2291, 2303,
        1, 12, 24, 36, 47, 1105, 1116, 1128, 1140, 1151, 2257, 2268, 2280, 2292, 2303,
        48, 59, 71, 83, 95, 1152, 1163, 1175, 1187, 1199, 2256, 2267, 2279, 2291, 2303,
        49, 60, 72, 84, 95, 1153, 1164, 1176, 1188, 1199, 2257, 2268, 2280, 2292, 2303,
    ]
    report.eq_ints("corner indices", interp.indices, want_idx)

    var want_wt: List[Float64] = [
        1.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.125, 0.25, 0.375, 0.5, 1.0, 0.25, 0.5, 0.75, 1.0,
        0.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.375, 0.25, 0.125, 0.0, 0.0, 0.75, 0.5, 0.25, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.125, 0.25, 0.375, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.375, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    report.eq_f64("corner weights", interp.weights, want_wt)

    # ── [7] tensor manifest ──────────────────────────────────────────────────
    print("")
    print("[7] tensor manifest and parameter budget")
    var names = minimax_h3_vision_tensor_names()
    report.eq_int("tensor count", len(names), 351)
    # Spot-check the three structural groups.
    var blocks = 0
    var deepstack = 0
    var merger = 0
    for i in range(len(names)):
        if names[i].find(String("visual.blocks.")) >= 0:
            blocks += 1
        elif names[i].find(String("deepstack_merger_list.")) >= 0:
            deepstack += 1
        elif names[i].find(String("visual.merger.")) >= 0:
            merger += 1
    report.eq_int("block tensors (27 x 12)", blocks, 324)
    report.eq_int("deepstack tensors (3 x 6)", deepstack, 18)
    report.eq_int("final merger tensors", merger, 6)

    var params = minimax_h3_vision_param_count()
    print("      parameter count:", params, "(~", params * 2 // (1024 * 1024), "MiB at bf16)")
    if params > 500000000 and params < 700000000:
        report.ok("parameter budget", String(params) + " in the expected 0.5-0.7B band")
    else:
        report.fail("parameter budget", String(params) + " outside the expected band")

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_qwen3vl_vision probe FAILED")
