# models/dit/parity/minimax_h3_rope_device_probe.mojo — compile+run gate for
# `models/dit/minimax_h3_rope.build_minimax_h3_rope_tables`.
#
# Builds the SAME rotary table two ways for one small synthetic packed grid:
#   1. the gated HOST oracle (`models/minimax_h3/dit_frontend.
#      minimax_h3_rope_inv_freq` / `minimax_h3_rope_table`, 15/15 vs
#      diffusers), a per-row CPU loop over List[Float64] positions;
#   2. the new DEVICE builder (`build_minimax_h3_rope_tables`), one GPU
#      thread per (row, column).
# and compares cos / sin / angles. This is a numerical self-check against the
# already-gated oracle, not a fresh diffusers parity gate — the H3 rope math
# itself is proven elsewhere (dit_frontend_parity); what this proves is that
# the GPU kernel reproduces that proven math.
#
# Positions are chosen to span the ranges the real packing module produces
# (models/minimax_h3/packing.mojo): zero, one temporal rotary unit
# (ROPE_FRAME_RESCALE = 5/3), spatial values on the ROPE_SPATIAL_SCALE=32
# grid, and a few larger multi-second/large-canvas magnitudes — never
# anywhere near the ~65536 range where Mojo's float32 trig needs a range
# reduction (see ideogram4_mrope.mojo), which is the point: H3 does not need
# that dance, and the gate should show the two paths already agree tightly.
#
#   pixi run mojo run -I . serenitymojo/models/dit/parity/minimax_h3_rope_device_probe.mojo

from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.models.minimax_h3.dit_frontend import (
    MINIMAX_H3_ROPE_FREQ_DIM,
    MINIMAX_H3_ROPE_THETA,
    minimax_h3_rope_inv_freq,
    minimax_h3_rope_table,
)
from serenitymojo.models.dit.minimax_h3_rope import build_minimax_h3_rope_tables


def _synthetic_positions() -> List[Float64]:
    """Flat `[rows*3]` (t, h, w) grid, row-major. 8 rows spanning small,
    zero, larger magnitudes, and the exact 120-second H3 packed boundary."""
    var p = List[Float64]()
    # row 0: origin.
    p.append(0.0); p.append(0.0); p.append(0.0)
    # row 1: one rotary-time unit forward, still no spatial extent.
    p.append(Float64(5.0) / Float64(3.0)); p.append(0.0); p.append(0.0)
    # row 2: pure spatial, mid-grid.
    p.append(0.0); p.append(16.0); p.append(8.0)
    # row 3: mixed, on the 32-multiple spatial grid plus several temporal units.
    p.append(Float64(100.0) / Float64(3.0)); p.append(32.0); p.append(96.0)
    # row 4: larger canvas extent (768x1344 short-edge-cap class).
    p.append(15.0); p.append(320.0); p.append(320.0)
    # row 5: many rotary-time units in (a several-second clip).
    p.append(200.0); p.append(5.0); p.append(11.0)
    # row 6: final packed text coordinate in the six-anchor 120 s request.
    p.append(1975.0); p.append(0.0); p.append(0.0)
    # row 7: final media coordinate (origin 1976 + 4825-unit span - 1).
    p.append(6800.0); p.append(32.0); p.append(64.0)
    return p^


def main() raises:
    var ctx = DeviceContext()
    var rows = 8
    var freq_dim = MINIMAX_H3_ROPE_FREQ_DIM
    var theta = MINIMAX_H3_ROPE_THETA
    var rotary_dim = 2 * 3 * freq_dim

    # 1. host oracle.
    var position_ids = _synthetic_positions()
    var inv_freq = minimax_h3_rope_inv_freq(freq_dim, theta)
    var oracle = minimax_h3_rope_table(position_ids, rows, inv_freq)
    if oracle.rotary_dim != rotary_dim:
        raise Error(
            String("probe: oracle rotary_dim ") + String(oracle.rotary_dim)
            + " != expected " + String(rotary_dim)
        )
    if len(oracle.cos) != rows * rotary_dim:
        raise Error("probe: oracle cos length mismatch")

    # 2. device builder. Cast Float64 positions to F32 before upload — the
    # same cast the oracle applies internally (dit_frontend.mojo:120-122).
    var positions_f32 = List[Float32]()
    for i in range(len(position_ids)):
        positions_f32.append(Float32(position_ids[i]))
    var positions = Tensor.from_host(
        positions_f32, [rows * 3], STDtype.F32, ctx
    )
    var device = build_minimax_h3_rope_tables(positions, ctx)

    if device[0].shape() != [rows, rotary_dim]:
        raise Error("probe: device cos shape mismatch")
    if device[1].shape() != [rows, rotary_dim]:
        raise Error("probe: device sin shape mismatch")
    if device[2].shape() != [rows, rotary_dim]:
        raise Error("probe: device angle shape mismatch")

    # 3. compare. Angles are pure F32 arithmetic (pos * inv_freq, same order
    # on host and device) — expect this near bit-exact. cos/sin route through
    # host libm vs the GPU trig intrinsic, which can differ by a few ulp on
    # the same input; still expect a very tight cosine similarity given how
    # small these angles are.
    var harness_tight = ParityHarness(cos_threshold=0.999999999)
    var harness_trig = ParityHarness(cos_threshold=0.999999)

    var angle_result = harness_tight.compare(device[2], oracle.angles, ctx)
    print("angles:", angle_result)
    var cos_result = harness_trig.compare(device[0], oracle.cos, ctx)
    print("cos:", cos_result)
    var sin_result = harness_trig.compare(device[1], oracle.sin, ctx)
    print("sin:", sin_result)

    # 4. duplicate-half self-consistency: columns [0,half) and [half,rotary)
    # must be identical in the device output (the whole point of the
    # rotate-half table layout).
    var half = 3 * freq_dim
    var host_cos = device[0].to_host(ctx)
    var host_sin = device[1].to_host(ctx)
    var max_half_diff = Float32(0.0)
    for r in range(rows):
        for c in range(half):
            var a = host_cos[r * rotary_dim + c]
            var b = host_cos[r * rotary_dim + half + c]
            var d = a - b
            if d < Float32(0.0):
                d = -d
            if d > max_half_diff:
                max_half_diff = d
            var sa = host_sin[r * rotary_dim + c]
            var sb = host_sin[r * rotary_dim + half + c]
            var sd = sa - sb
            if sd < Float32(0.0):
                sd = -sd
            if sd > max_half_diff:
                max_half_diff = sd
    print("duplicate-half max_diff:", max_half_diff)

    # 5. BF16 out_dtype dispatch branch (untouched by the checks above, which
    # only exercised F32). Same positions, just the storage-dtype path.
    var device_bf16 = build_minimax_h3_rope_tables(
        positions, ctx, freq_dim, theta, STDtype.BF16
    )
    if device_bf16[0].dtype() != STDtype.BF16:
        raise Error("probe: BF16 cos returned non-BF16 storage")
    if device_bf16[1].dtype() != STDtype.BF16:
        raise Error("probe: BF16 sin returned non-BF16 storage")
    var harness_bf16 = ParityHarness(cos_threshold=0.999)
    var bf16_cos_result = harness_bf16.compare(device_bf16[0], oracle.cos, ctx)
    var bf16_sin_result = harness_bf16.compare(device_bf16[1], oracle.sin, ctx)
    print("bf16 cos:", bf16_cos_result)
    print("bf16 sin:", bf16_sin_result)
    if not bf16_cos_result.passed:
        raise Error("probe: FAIL bf16 cos diverges from the oracle")
    if not bf16_sin_result.passed:
        raise Error("probe: FAIL bf16 sin diverges from the oracle")

    if not angle_result.passed:
        raise Error("probe: FAIL angle arithmetic diverges from the oracle")
    if not cos_result.passed:
        raise Error("probe: FAIL cos diverges from the oracle")
    if not sin_result.passed:
        raise Error("probe: FAIL sin diverges from the oracle")
    if max_half_diff != Float32(0.0):
        raise Error("probe: FAIL the two table halves are not identical")

    # 6. Opt-in RIFLEx gate.  The official one-line rule replaces frequency
    # k-1 with `0.9 * 2*pi / L_test`.  H3's media clock begins after the packed
    # text rows, so the device implementation keeps text untouched and applies
    # that frequency to media-local time with phase continuity at the origin.
    var riflex_k = 10
    var riflex_origin = Float32(1976.0)
    var riflex_span = Float32(4825.0)
    var riflex = build_minimax_h3_rope_tables(
        positions, ctx, freq_dim, theta, STDtype.F32,
        riflex_k, riflex_origin, riflex_span,
    )
    var expected_riflex_angles = oracle.angles.copy()
    var old_frequency = inv_freq[riflex_k - 1]
    var new_frequency = Float32(
        Float64(0.9) * Float64(6.283185307179586476925286766559)
        / Float64(riflex_span)
    )
    var selected = riflex_k - 1
    for r in range(rows):
        var position = Float32(position_ids[3 * r])
        if position >= riflex_origin:
            var expected_angle = (
                riflex_origin * old_frequency
                + (position - riflex_origin) * new_frequency
            )
            expected_riflex_angles[r * rotary_dim + selected] = expected_angle
            expected_riflex_angles[
                r * rotary_dim + half + selected
            ] = expected_angle
    var riflex_angle_result = harness_tight.compare(
        riflex[2], expected_riflex_angles, ctx
    )
    print(
        "riflex k=", riflex_k, " old_frequency=", old_frequency,
        " new_frequency=", new_frequency,
    )
    print("riflex angles:", riflex_angle_result)
    if not riflex_angle_result.passed:
        raise Error("probe: FAIL RIFLEx angles diverge from the reference rule")

    var riflex_host = riflex[2].to_host(ctx)
    var media_cell = 7 * rotary_dim + selected  # row 7 is final media time.
    if riflex_host[media_cell] == oracle.angles[media_cell]:
        raise Error("probe: FAIL RIFLEx negative control did not change media")
    var text_cell = 6 * rotary_dim + selected  # row 6 is final text time.
    if riflex_host[text_cell] != oracle.angles[text_cell]:
        raise Error("probe: FAIL RIFLEx changed packed text RoPE")

    print("minimax_h3_rope_device_probe PASS")
