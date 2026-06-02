# Microsoft Lens block-0 image Q/K RMSNorm + RoPE sampled model-math smoke.
#
# CPU-side sampled gate over real transformer weights and existing captures.

from serenitymojo.models.lens.lens_dit_math import validate_lens_block0_qk_rope_sample_gate


def main() raises:
    var stats = validate_lens_block0_qk_rope_sample_gate()
    print("[lens-dit-qk-rope] samples/qk_values:", stats.samples, stats.qk_values)
    print("[lens-dit-qk-rope] finite:", stats.finite_values)
    print(
        "[lens-dit-qk-rope] got mean/std/absmax:",
        stats.got_mean,
        stats.got_std,
        stats.got_absmax,
    )
    print(
        "[lens-dit-qk-rope] ref mean/std/absmax:",
        stats.ref_mean,
        stats.ref_std,
        stats.ref_absmax,
    )
    print(
        "[lens-dit-qk-rope] mean_abs_diff/max_abs_diff:",
        stats.mean_abs_diff,
        stats.max_abs_diff,
    )
    print("Microsoft Lens block-0 image Q/K RoPE sampled smoke PASS")
