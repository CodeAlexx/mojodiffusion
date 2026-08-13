# models/dit/parity/minimax_h3_rope_apply_skeptic_probe.mojo — SKEPTIC probe.
#
# Attacks the specific claim that `ops/rope.rope_halfsplit_full` (the device
# kernel `minimax_h3_dit._minimax_h3_apply_partial_rope` actually calls) is
# arithmetically IDENTICAL to the gated host oracle's
# `models/minimax_h3/block_forward._apply_rope_inplace`. That claim was
# checked only by hand (algebra in a comment); this probe RUNS both, on the
# same input, at more than the original 6-row/1-head toy scale, with a
# genuine partial-rope split (rotary_dim=96 of head_dim=128, so 32 channels
# per head must survive untouched), and diffs ELEMENTWISE — not through
# ParityHarness's single aggregate cosine-over-576-elements gate, which
# cannot see an error confined to one axis, one row, or the untouched tail
# once averaged against everything else.
#
# Includes a NEGATIVE CONTROL: the same comparison run against a deliberately
# corrupted table (h/w axes swapped) must FAIL. If it doesn't, this probe is
# not actually testing anything and its PASS on the real table is worthless.
#
#   pixi run mojo run -I . serenitymojo/models/dit/parity/minimax_h3_rope_apply_skeptic_probe.mojo

from max.gpu.host import DeviceContext
from std.math import sin as fsin
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.minimax_h3.dit_frontend import (
    MINIMAX_H3_ROPE_FREQ_DIM,
    MINIMAX_H3_ROPE_THETA,
    minimax_h3_rope_inv_freq,
    minimax_h3_rope_table,
    MiniMaxH3RopeTable,
)
from serenitymojo.models.minimax_h3.block_forward import _apply_rope_inplace
from serenitymojo.models.dit.minimax_h3_rope import build_minimax_h3_rope_tables
from serenitymojo.ops.rope import rope_halfsplit_full


def _pseudo(i: Int) -> Float32:
    """Deterministic non-degenerate synthetic activation in [-1, 1]."""
    return Float32(fsin(Float64(i) * 12.9898123))


def _broadcast_heads(
    table: List[Float32], rows: Int, heads: Int, width: Int
) -> List[Float32]:
    """`[rows, width]` -> `[rows*heads, width]`, replicating row s to every
    head — the host equivalent of `minimax_h3_dit._minimax_h3_expand_rope_per_head`."""
    var out = List[Float32]()
    for s in range(rows):
        for h in range(heads):
            for c in range(width):
                out.append(table[s * width + c])
    return out^


def _max_abs_and_worst(
    a: List[Float32], b: List[Float32]
) -> Tuple[Float32, Int]:
    """Elementwise max|a-b| and the index it occurs at (NOT a cosine
    similarity — a single bad element cannot hide behind many good ones)."""
    var worst = Float32(0.0)
    var worst_idx = -1
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < Float32(0.0):
            d = -d
        if d > worst:
            worst = d
            worst_idx = i
    return (worst, worst_idx)


def main() raises:
    var ctx = DeviceContext()
    var rows = 40
    var heads = 6
    var head_dim = 128
    var freq_dim = MINIMAX_H3_ROPE_FREQ_DIM
    var theta = MINIMAX_H3_ROPE_THETA
    var rotary_dim = 2 * 3 * freq_dim  # 96
    var pass_through = head_dim - rotary_dim  # 32

    print("rows=", rows, " heads=", heads, " head_dim=", head_dim,
          " rotary_dim=", rotary_dim, " pass_through=", pass_through)

    # 1. Realistic-ish (t, h, w) positions: varied magnitude AND sign — the
    # width axis of a wide (>square) canvas genuinely goes slightly negative
    # in the real spatial grid (packing.mojo's `left = (1-ratio)/2` is
    # negative when ratio > 1), so this is not a synthetic edge case.
    var position_ids = List[Float64]()
    for r in range(rows):
        var t = Float64(r) * (Float64(5.0) / Float64(3.0))
        var h = Float64((r * 7) % 25) * (Float64(32.0) / Float64(25.0)) - 3.0
        var w = Float64((r * 13) % 43) * (Float64(32.0) / Float64(43.0)) - 5.0
        position_ids.append(t)
        position_ids.append(h)
        position_ids.append(w)

    # 2. Host oracle table (already gated 15/15 vs diffusers elsewhere).
    var inv_freq = minimax_h3_rope_inv_freq(freq_dim, theta)
    var oracle_table = minimax_h3_rope_table(position_ids, rows, inv_freq)
    if oracle_table.rotary_dim != rotary_dim:
        raise Error("probe: oracle rotary_dim mismatch")

    # 3. Device table (the module under attack).
    var positions_f32 = List[Float32]()
    for i in range(len(position_ids)):
        positions_f32.append(Float32(position_ids[i]))
    var positions_t = Tensor.from_host(positions_f32, [rows * 3], STDtype.F32, ctx)
    var device_table = build_minimax_h3_rope_tables(
        positions_t, ctx, freq_dim, theta, STDtype.F32
    )
    var device_cos_host = device_table[0].to_host(ctx)
    var device_sin_host = device_table[1].to_host(ctx)

    # 4. Synthetic per-head activations, full head_dim=128 width.
    var host_heads = List[Float32]()
    for s in range(rows):
        for h in range(heads):
            for d in range(head_dim):
                host_heads.append(_pseudo((s * heads + h) * head_dim + d))

    # 5. HOST ORACLE apply: mutate a copy in place with the real production
    # function, full head_dim width (rotary_dim rotated, the rest untouched).
    var oracle_heads = host_heads.copy()
    _apply_rope_inplace(
        oracle_heads, rows, heads, head_dim, oracle_table.cos, oracle_table.sin,
        rotary_dim,
    )

    # Sanity check ON THE ORACLE ITSELF: the pass-through tail must be
    # bit-identical to the input (if this fails, the oracle isn't doing what
    # its own docstring claims, and everything below is moot).
    var oracle_tail_diff = Float32(0.0)
    for s in range(rows):
        for h in range(heads):
            for d in range(rotary_dim, head_dim):
                var idx = (s * heads + h) * head_dim + d
                var diff = oracle_heads[idx] - host_heads[idx]
                if diff < Float32(0.0):
                    diff = -diff
                if diff > oracle_tail_diff:
                    oracle_tail_diff = diff
    print("oracle pass-through tail max diff (sanity, expect 0.0):", oracle_tail_diff)

    # Extract the oracle's rotated 96-wide slice for comparison.
    var oracle_rotated = List[Float32]()
    for s in range(rows):
        for h in range(heads):
            for d in range(rotary_dim):
                oracle_rotated.append(host_heads[(s * heads + h) * head_dim + d])
    # placeholder overwritten below with the actual rotated values
    oracle_rotated = List[Float32]()
    for s in range(rows):
        for h in range(heads):
            for d in range(rotary_dim):
                oracle_rotated.append(oracle_heads[(s * heads + h) * head_dim + d])

    # 6. DEVICE apply: build the already-sliced [rows*heads, rotary_dim]
    # input tensor (mirrors what `_minimax_h3_apply_partial_rope`'s `slice`
    # call produces) and run the real `rope_halfsplit_full` kernel — TWICE:
    # (a) against the host oracle's own table (isolates rope_halfsplit_full's
    #     arithmetic alone), and
    # (b) against the device-built table (the full production chain: device
    #     table build -> device rope apply).
    var device_x_host = List[Float32]()
    for s in range(rows):
        for h in range(heads):
            for d in range(rotary_dim):
                device_x_host.append(host_heads[(s * heads + h) * head_dim + d])
    var device_x = Tensor.from_host(
        device_x_host, [rows * heads, rotary_dim], STDtype.F32, ctx
    )

    var cos_exp_a = _broadcast_heads(oracle_table.cos, rows, heads, rotary_dim)
    var sin_exp_a = _broadcast_heads(oracle_table.sin, rows, heads, rotary_dim)
    var cos_a_t = Tensor.from_host(cos_exp_a, [rows * heads, rotary_dim], STDtype.F32, ctx)
    var sin_a_t = Tensor.from_host(sin_exp_a, [rows * heads, rotary_dim], STDtype.F32, ctx)
    var out_a = rope_halfsplit_full(device_x, cos_a_t, sin_a_t, ctx)
    var out_a_host = out_a.to_host(ctx)

    var cos_exp_b = _broadcast_heads(device_cos_host, rows, heads, rotary_dim)
    var sin_exp_b = _broadcast_heads(device_sin_host, rows, heads, rotary_dim)
    var cos_b_t = Tensor.from_host(cos_exp_b, [rows * heads, rotary_dim], STDtype.F32, ctx)
    var sin_b_t = Tensor.from_host(sin_exp_b, [rows * heads, rotary_dim], STDtype.F32, ctx)
    var out_b = rope_halfsplit_full(device_x, cos_b_t, sin_b_t, ctx)
    var out_b_host = out_b.to_host(ctx)

    var result_a = _max_abs_and_worst(out_a_host, oracle_rotated)
    var result_b = _max_abs_and_worst(out_b_host, oracle_rotated)
    print("(a) rope_halfsplit_full[host table]   vs oracle: max_abs=", result_a[0],
          " worst_idx=", result_a[1], " / n=", len(oracle_rotated))
    print("(b) rope_halfsplit_full[device table]  vs oracle: max_abs=", result_b[0],
          " worst_idx=", result_b[1], " / n=", len(oracle_rotated))

    # 7. NEGATIVE CONTROL: swap the h/w axis blocks in the table (columns
    # [freq_dim, 2*freq_dim) <-> [2*freq_dim, 3*freq_dim) within each half)
    # before broadcasting+applying. If `rope_halfsplit_full` genuinely
    # implements the oracle's rotate-half formula, feeding it a WRONG table
    # must produce a large diff. If it doesn't, this probe cannot be trusted
    # to have caught anything above.
    var corrupt_cos = oracle_table.cos.copy()
    var corrupt_sin = oracle_table.sin.copy()
    for s in range(rows):
        var row = s * rotary_dim
        for half_off in range(0, rotary_dim, 3 * freq_dim):
            for i in range(freq_dim):
                var h_idx = row + half_off + freq_dim + i
                var w_idx = row + half_off + 2 * freq_dim + i
                var tmp_c = corrupt_cos[h_idx]
                corrupt_cos[h_idx] = corrupt_cos[w_idx]
                corrupt_cos[w_idx] = tmp_c
                var tmp_s = corrupt_sin[h_idx]
                corrupt_sin[h_idx] = corrupt_sin[w_idx]
                corrupt_sin[w_idx] = tmp_s
    var cos_exp_c = _broadcast_heads(corrupt_cos, rows, heads, rotary_dim)
    var sin_exp_c = _broadcast_heads(corrupt_sin, rows, heads, rotary_dim)
    var cos_c_t = Tensor.from_host(cos_exp_c, [rows * heads, rotary_dim], STDtype.F32, ctx)
    var sin_c_t = Tensor.from_host(sin_exp_c, [rows * heads, rotary_dim], STDtype.F32, ctx)
    var out_c = rope_halfsplit_full(device_x, cos_c_t, sin_c_t, ctx)
    var out_c_host = out_c.to_host(ctx)
    var result_c = _max_abs_and_worst(out_c_host, oracle_rotated)
    print("(c) NEGATIVE CONTROL (h/w swapped table) vs oracle: max_abs=", result_c[0],
          " worst_idx=", result_c[1], " (expect LARGE, this must NOT pass)")

    var TIGHT = Float32(1e-4)
    if oracle_tail_diff != Float32(0.0):
        raise Error("probe: FAIL oracle pass-through tail was mutated")
    if result_a[0] > TIGHT:
        raise Error("probe: FAIL (a) rope_halfsplit_full[host table] diverges from oracle")
    if result_b[0] > TIGHT:
        raise Error("probe: FAIL (b) rope_halfsplit_full[device table] diverges from oracle")
    if result_c[0] <= TIGHT:
        raise Error(
            "probe: FAIL negative control did NOT diverge — this probe is not"
            " discriminative and nothing above should be trusted"
        )
    print("minimax_h3_rope_apply_skeptic_probe PASS (and negative control correctly failed)")
