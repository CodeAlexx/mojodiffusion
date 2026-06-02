# pipeline/ltx2_sampler_smoke.mojo — LTX-2 distilled sampler + guidance parity smoke.
#
# Team 3 self-check for the pure-Mojo LTX-2 distilled sigma schedule, the
# CFG-star guidance, and the STG mask/rescale. The math is small and parity is
# bit-clean in Rust (ltx2_sigma_parity / ltx2_guidance_parity, both max_abs=0.0),
# so this smoke asserts against hand-computed expected values where feasible —
# it is a parity-style self-check, not just a wiring proof.
#
# Checks (PASS/FAIL printed for each):
#   1. build_ltx2_distilled_sigma_schedule(8, 0.025) == first 8 of the distilled
#      table (LTX2_DISTILLED_SIGMAS[:8]); and == build_ltx2_distilled_sigma_schedule
#      with the trailing 0.0 -> the full distilled table.
#   2. cfg_star_rescale on a tiny known case: text=[1,2,3,4], uncond=[2,2,2,2]
#      -> alpha=1.25, rescaled=[2.5,2.5,2.5,2.5].
#   3. cfg_star full combine output stats (no exact oracle for the additive
#      combine — printed for inspection; the rescale piece IS asserted in #2).
#   4. build_skip_layer_mask small case == [[1,1,1],[1,1,0],[1,1,1],[1,1,0]].
#   5. stg_rescale on pos=[1,2,3,4], guided=[2,4,6,8], scale=0.7 -> factor=0.65,
#      out=[1.3,2.6,3.9,5.2].
#   6. One LTX2Scheduler.step (final-step denoise path) sanity.
#
# *** CODE-ONLY: COMPILE-VERIFIED, NOT executed here. A later pass runs it on GPU
# for the actual parity numbers. ***
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.sampling.ltx2_sampling import (
    LTX2Scheduler,
    build_ltx2_distilled_sigma_schedule,
    ltx2_distilled_sigmas,
    ltx2_stage2_distilled_sigmas,
)
from serenitymojo.sampling.ltx2_guidance import (
    cfg_star,
    cfg_star_alpha,
    cfg_star_rescale,
    build_skip_layer_mask,
    single_cond_skip_mask,
    stg_rescale,
    SkipLayerStrategy,
)


comptime _TOL: Float32 = 1e-4


def _abs(x: Float32) -> Float32:
    return x if x >= 0.0 else -x


def _close(a: Float32, b: Float32) -> Bool:
    return _abs(a - b) <= _TOL


def _print_stats(name: String, h: List[Float32]):
    """Print min/max/mean of a host list."""
    var lo = h[0]
    var hi = h[0]
    var s: Float32 = 0.0
    for i in range(len(h)):
        var v = h[i]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
        s += v
    print("    ", name, "min/max/mean =", lo, hi, s / Float32(len(h)))


def main() raises:
    var ctx = DeviceContext()
    var all_pass = True

    print("=== LTX-2 distilled sampler + guidance smoke ===")

    # ── 1. Distilled sigma schedule ───────────────────────────────────────────
    print("\n[1] build_ltx2_distilled_sigma_schedule(8, 0.025)")
    var sched8 = build_ltx2_distilled_sigma_schedule(8, 0.025)
    var expected8 = List[Float32]()
    expected8.append(1.0)
    expected8.append(0.99375)
    expected8.append(0.9875)
    expected8.append(0.98125)
    expected8.append(0.975)
    expected8.append(0.909375)
    expected8.append(0.725)
    expected8.append(0.421875)
    print("    sigmas:")
    for i in range(len(sched8)):
        print("      [", i, "] =", sched8[i], " expected", expected8[i])
    var ok1 = len(sched8) == 8
    if ok1:
        for i in range(8):
            if not _close(sched8[i], expected8[i]):
                ok1 = False
    # Distilled table = sched8 + trailing 0.0.
    var table = ltx2_distilled_sigmas()
    var ok1b = len(table) == 9 and _close(table[8], 0.0)
    if ok1b:
        for i in range(8):
            if not _close(table[i], expected8[i]):
                ok1b = False
    if ok1 and ok1b:
        print("    PASS  (== distilled LTX2_DISTILLED_SIGMAS[:8], +0.0 terminator)")
    else:
        print("    FAIL")
        all_pass = False

    # Stage-2 table.
    var s2 = ltx2_stage2_distilled_sigmas()
    print("    stage2 sigmas:", s2[0], s2[1], s2[2], s2[3])
    var ok1c = (
        len(s2) == 4
        and _close(s2[0], 0.909375)
        and _close(s2[1], 0.725)
        and _close(s2[2], 0.421875)
        and _close(s2[3], 0.0)
    )
    if ok1c:
        print("    stage2 PASS")
    else:
        print("    stage2 FAIL")
        all_pass = False

    # ── 2. CFG-star rescale (exact oracle) ────────────────────────────────────
    print("\n[2] cfg_star_rescale: text=[1,2,3,4], uncond=[2,2,2,2]")
    var text_h = List[Float32]()
    text_h.append(1.0)
    text_h.append(2.0)
    text_h.append(3.0)
    text_h.append(4.0)
    var unc_h = List[Float32]()
    for _ in range(4):
        unc_h.append(2.0)
    var shape4 = List[Int]()
    shape4.append(1)
    shape4.append(4)
    var t_text = Tensor.from_host(text_h.copy(), shape4.copy(), STDtype.F32, ctx)
    var t_unc = Tensor.from_host(unc_h.copy(), shape4.copy(), STDtype.F32, ctx)
    var alphas = cfg_star_alpha(t_text, t_unc, ctx)
    print("    alpha =", alphas[0], " expected 1.25")
    var rescaled = cfg_star_rescale(t_text, t_unc, ctx)
    var resc_h = rescaled.to_host(ctx)
    print("    rescaled =", resc_h[0], resc_h[1], resc_h[2], resc_h[3], " expected 2.5 x4")
    var ok2 = _close(alphas[0], 1.25)
    for i in range(4):
        if not _close(resc_h[i], 2.5):
            ok2 = False
    if ok2:
        print("    PASS")
    else:
        print("    FAIL")
        all_pass = False

    # ── 3. CFG-star full combine (stats only) ─────────────────────────────────
    print("\n[3] cfg_star full combine (scale=3.0) — stats")
    var comb = cfg_star(t_text, t_unc, 3.0, ctx)
    var comb_h = comb.to_host(ctx)
    _print_stats("out", comb_h)
    print("    (rescale piece asserted in [2]; additive combine is informational)")

    # ── 4. STG skip-layer mask (exact oracle) ─────────────────────────────────
    print("\n[4] build_skip_layer_mask(4, 1, 3, skip=[1,3], ptb=2)")
    var skip = List[Int]()
    skip.append(1)
    skip.append(3)
    var mask = build_skip_layer_mask(4, 1, 3, skip.copy(), 2)
    # Expected [[1,1,1],[1,1,0],[1,1,1],[1,1,0]].
    var ok4 = len(mask) == 4
    for r in range(len(mask)):
        print("    row", r, "=", mask[r][0], mask[r][1], mask[r][2])
    if ok4:
        ok4 = (
            _close(mask[0][0], 1.0) and _close(mask[0][1], 1.0) and _close(mask[0][2], 1.0)
            and _close(mask[1][0], 1.0) and _close(mask[1][1], 1.0) and _close(mask[1][2], 0.0)
            and _close(mask[2][0], 1.0) and _close(mask[2][1], 1.0) and _close(mask[2][2], 1.0)
            and _close(mask[3][0], 1.0) and _close(mask[3][1], 1.0) and _close(mask[3][2], 0.0)
        )
    if ok4:
        print("    PASS")
    else:
        print("    FAIL")
        all_pass = False

    # single_cond column variant.
    var col = single_cond_skip_mask(4, skip.copy())
    print("    single_cond mask =", col[0], col[1], col[2], col[3], " expected 1,0,1,0")
    var ok4b = (
        _close(col[0], 1.0) and _close(col[1], 0.0)
        and _close(col[2], 1.0) and _close(col[3], 0.0)
    )
    if not ok4b:
        all_pass = False

    # SkipLayerStrategy parse.
    var strat = SkipLayerStrategy.from_str("stg_av")
    print("    SkipLayerStrategy.from_str('stg_av') ==", strat.tag, "(AttentionValues=1)")
    if strat != SkipLayerStrategy.AttentionValues:
        all_pass = False

    # ── 5. STG std-rescale (exact oracle) ──────────────────────────────────────
    print("\n[5] stg_rescale: pos=[1,2,3,4], guided=[2,4,6,8], scale=0.7")
    var pos_h = List[Float32]()
    pos_h.append(1.0)
    pos_h.append(2.0)
    pos_h.append(3.0)
    pos_h.append(4.0)
    var gd_h = List[Float32]()
    gd_h.append(2.0)
    gd_h.append(4.0)
    gd_h.append(6.0)
    gd_h.append(8.0)
    var t_pos = Tensor.from_host(pos_h.copy(), shape4.copy(), STDtype.F32, ctx)
    var t_gd = Tensor.from_host(gd_h.copy(), shape4.copy(), STDtype.F32, ctx)
    var sr = stg_rescale(t_pos, t_gd, 0.7, ctx)
    var sr_h = sr.to_host(ctx)
    # factor = 0.7*(std_pos/std_guided) + 0.3 = 0.7*0.5 + 0.3 = 0.65.
    print("    out =", sr_h[0], sr_h[1], sr_h[2], sr_h[3], " expected 1.3,2.6,3.9,5.2")
    var ok5 = (
        _close(sr_h[0], 1.3) and _close(sr_h[1], 2.6)
        and _close(sr_h[2], 3.9) and _close(sr_h[3], 5.2)
    )
    if ok5:
        print("    PASS")
    else:
        print("    FAIL")
        all_pass = False

    # ── 6. LTX2Scheduler.step final-step denoise ──────────────────────────────
    print("\n[6] LTX2Scheduler.step final-step denoise (x - v*sigma)")
    var dsched = LTX2Scheduler.distilled()
    print("    num_steps =", dsched.num_steps, " (expected 8)")
    # Make a tiny latent + velocity, run the final step (i = num_steps-1, sigma_next=0).
    var lat_h = List[Float32]()
    lat_h.append(10.0)
    lat_h.append(20.0)
    var vel_h = List[Float32]()
    vel_h.append(1.0)
    vel_h.append(2.0)
    var sh2 = List[Int]()
    sh2.append(1)
    sh2.append(2)
    var lat = Tensor.from_host(lat_h.copy(), sh2.copy(), STDtype.F32, ctx)
    var vel = Tensor.from_host(vel_h.copy(), sh2.copy(), STDtype.F32, ctx)
    var last_i = dsched.num_steps - 1
    var sigma_last = dsched.sigma(last_i)  # 0.421875
    var out6 = dsched.step(lat, vel, last_i, ctx)
    var out6_h = out6.to_host(ctx)
    # Expected: x - v*sigma = [10 - 1*0.421875, 20 - 2*0.421875].
    var e0 = 10.0 - 1.0 * sigma_last
    var e1 = 20.0 - 2.0 * sigma_last
    print("    sigma_last =", sigma_last, " out =", out6_h[0], out6_h[1])
    print("    expected", e0, e1)
    var ok6 = _close(out6_h[0], e0) and _close(out6_h[1], e1)
    if ok6:
        print("    PASS")
    else:
        print("    FAIL")
        all_pass = False

    print("\n=== OVERALL:", "PASS" if all_pass else "FAIL", "===")
