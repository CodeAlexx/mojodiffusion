# Microsoft Lens block-0 image QKV sampled model-math smoke.
#
# CPU-side sampled gate over real transformer weights and existing captures.

from serenitymojo.models.lens.lens_dit_math import validate_lens_block0_qkv_sample_gate


def main() raises:
    var stats = validate_lens_block0_qkv_sample_gate()
    print("[lens-dit-qkv] samples/qkv_values:", stats.samples, stats.qkv_values)
    print("[lens-dit-qkv] finite:", stats.finite_values)
    print(
        "[lens-dit-qkv] got mean/std/absmax:",
        stats.got_mean,
        stats.got_std,
        stats.got_absmax,
    )
    print(
        "[lens-dit-qkv] ref mean/std/absmax:",
        stats.ref_mean,
        stats.ref_std,
        stats.ref_absmax,
    )
    print(
        "[lens-dit-qkv] mean_abs_diff/max_abs_diff:",
        stats.mean_abs_diff,
        stats.max_abs_diff,
    )
    print("Microsoft Lens block-0 image QKV sampled smoke PASS")
