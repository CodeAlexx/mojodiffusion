# models/dit/parity/minimax_h3_rope_scale_skeptic_probe.mojo — SKEPTIC probe.
#
# Attacks the claim "no F64 range-reduction needed because H3 positions never
# approach ~65536" and the claim that the 6-row synthetic toy in
# `minimax_h3_rope_device_probe.mojo` generalizes to a real packed sequence.
# Builds an actual production-scale packed sequence via the REAL packing
# module (`models/minimax_h3/packing.minimax_h3_build_packed_sequence`) at
# geometry matching the released model's short-edge-cap canvas class
# (768x1344-area, ~124 pixel frames — a several-second clip), not
# hand-picked synthetic numbers, then:
#   1. reports the ACTUAL min/max position magnitude the real packing
#      geometry produces on every axis, so "never approaches 65536" is a
#      measurement instead of an assertion;
#   2. runs the device table builder against the host oracle at this row
#      count (tens of thousands of rows, not 6) and checks for NaN/Inf and
#      elementwise (not aggregate-cosine) divergence;
#   3. exercises the same order of magnitude the real grid_dim computation
#      will see, in case anything only breaks at scale (index overflow,
#      thread/block miscount).
#
#   pixi run mojo run -I . serenitymojo/models/dit/parity/minimax_h3_rope_scale_skeptic_probe.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.minimax_h3.dit_frontend import (
    MINIMAX_H3_ROPE_FREQ_DIM,
    MINIMAX_H3_ROPE_THETA,
    minimax_h3_rope_inv_freq,
    minimax_h3_rope_table,
)
from serenitymojo.models.minimax_h3.packing import (
    minimax_h3_build_packed_sequence,
    minimax_h3_video_latent_num_frames,
    minimax_h3_audio_latent_num_frames,
    minimax_h3_align_num_frames,
    MINIMAX_H3_TEXT_TAG,
)
from serenitymojo.models.dit.minimax_h3_rope import build_minimax_h3_rope_tables


def main() raises:
    var ctx = DeviceContext()

    # Geometry matching the brief's "production sizes (1344x768, 124 frames)":
    # a 16x spatial downsample of a 768x1344-area canvas (the ratio, not the
    # absolute latent size, is what determines position MAGNITUDE — packing's
    # spatial grid normalizes by sqrt(latent_height*latent_width) — so this
    # choice of downsample factor doesn't change the claim-4 answer, only the
    # row count, which is exactly what we ALSO want at production scale).
    var latent_height = 48   # 768 / 16
    var latent_width = 84    # 1344 / 16
    var patch_h = 2
    var patch_w = 2

    var raw_frames = 124
    var aligned_frames = minimax_h3_align_num_frames(raw_frames)  # -> 17n+5
    var num_latent_frames = minimax_h3_video_latent_num_frames(aligned_frames)
    var num_audio_latents = minimax_h3_audio_latent_num_frames(aligned_frames)

    var num_text_tokens = 300  # a full-length prompt after tokenization
    var text_token_tags = List[Int]()
    for _ in range(num_text_tokens):
        text_token_tags.append(MINIMAX_H3_TEXT_TAG)
    var keyframe_anchors = List[Int]()  # t2va: no keyframe conditioning

    var layout = minimax_h3_build_packed_sequence(
        text_token_tags, num_latent_frames, latent_height, latent_width,
        num_audio_latents, patch_h, patch_w, keyframe_anchors,
    )
    var rows = layout.sequence_length
    print("aligned_frames=", aligned_frames, " num_latent_frames=", num_latent_frames,
          " num_audio_latents=", num_audio_latents, " sequence_length(rows)=", rows)

    # 1. ACTUAL position magnitude range on this real production-scale grid.
    var min_t = layout.position_ids[0]
    var max_t = layout.position_ids[0]
    var min_h = layout.position_ids[1]
    var max_h = layout.position_ids[1]
    var min_w = layout.position_ids[2]
    var max_w = layout.position_ids[2]
    for r in range(rows):
        var t = layout.position_ids[3 * r]
        var h = layout.position_ids[3 * r + 1]
        var w = layout.position_ids[3 * r + 2]
        if t < min_t: min_t = t
        if t > max_t: max_t = t
        if h < min_h: min_h = h
        if h > max_h: max_h = h
        if w < min_w: min_w = w
        if w > max_w: max_w = w
    print("t axis range: [", min_t, ",", max_t, "]")
    print("h axis range: [", min_h, ",", max_h, "]")
    print("w axis range: [", min_w, ",", max_w, "]")
    print("(claim under test: none of these approach 65536)")

    # 2. Device table vs host oracle at this row count.
    var freq_dim = MINIMAX_H3_ROPE_FREQ_DIM
    var theta = MINIMAX_H3_ROPE_THETA
    var rotary_dim = 2 * 3 * freq_dim

    var inv_freq = minimax_h3_rope_inv_freq(freq_dim, theta)
    var oracle_table = minimax_h3_rope_table(layout.position_ids, rows, inv_freq)

    var positions_f32 = List[Float32]()
    for i in range(len(layout.position_ids)):
        positions_f32.append(Float32(layout.position_ids[i]))
    var positions_t = Tensor.from_host(positions_f32, [rows * 3], STDtype.F32, ctx)

    var device_table = build_minimax_h3_rope_tables(
        positions_t, ctx, freq_dim, theta, STDtype.F32
    )
    print("device cos/sin shape:", device_table[0].shape(), device_table[1].shape())
    var total_elems = rows * rotary_dim
    print("total table elements:", total_elems, " (grid blocks @256:",
          (total_elems + 255) // 256, ")")

    var device_cos = device_table[0].to_host(ctx)
    var device_sin = device_table[1].to_host(ctx)
    var device_angle = device_table[2].to_host(ctx)

    if len(device_cos) != len(oracle_table.cos):
        raise Error("probe: length mismatch cos")
    if len(device_sin) != len(oracle_table.sin):
        raise Error("probe: length mismatch sin")

    var max_abs_cos = Float64(0.0)
    var max_abs_sin = Float64(0.0)
    var max_abs_angle = Float64(0.0)
    var nan_or_inf = False
    var worst_cos_idx = -1
    for i in range(len(device_cos)):
        var dc = Float64(device_cos[i]) - Float64(oracle_table.cos[i])
        if dc < 0.0: dc = -dc
        if dc > max_abs_cos:
            max_abs_cos = dc
            worst_cos_idx = i
        var ds = Float64(device_sin[i]) - Float64(oracle_table.sin[i])
        if ds < 0.0: ds = -ds
        if ds > max_abs_sin:
            max_abs_sin = ds
        var da = Float64(device_angle[i]) - Float64(oracle_table.angles[i])
        if da < 0.0: da = -da
        if da > max_abs_angle:
            max_abs_angle = da
        # crude NaN/Inf sniff: a finite float compared to itself via bounds;
        # NaN fails every ordering comparison, so dc/ds/da would never exceed
        # max_abs and max_abs would silently stay 0 while raw values are NaN.
        if device_cos[i] != device_cos[i] or device_sin[i] != device_sin[i]:
            nan_or_inf = True

    print("angle max_abs (whole table, elementwise):", max_abs_angle)
    print("cos   max_abs (whole table, elementwise):", max_abs_cos, " worst_idx=", worst_cos_idx,
          " row=", worst_cos_idx // rotary_dim if worst_cos_idx >= 0 else -1)
    print("sin   max_abs (whole table, elementwise):", max_abs_sin)
    print("any NaN in cos/sin:", nan_or_inf)

    if nan_or_inf:
        raise Error("probe: FAIL NaN/Inf detected at production scale")
    if max_abs_angle > 1.0e-3:
        raise Error("probe: FAIL angle arithmetic diverges at production scale")
    if max_abs_cos > 1.0e-3 or max_abs_sin > 1.0e-3:
        raise Error("probe: FAIL cos/sin diverge beyond expected transcendental ULP drift")

    print("minimax_h3_rope_scale_skeptic_probe PASS")
