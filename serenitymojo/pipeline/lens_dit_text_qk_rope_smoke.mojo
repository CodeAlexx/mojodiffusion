# Microsoft Lens block-0 text Q/K RMSNorm + RoPE sampled model-math smoke.
#
# CPU-side sampled gate over real text-smoke hidden captures and transformer weights.

from serenitymojo.models.lens.lens_dit_math import (
    validate_lens_block0_text_qk_rope_sample_gate,
)


def main() raises:
    var stats = validate_lens_block0_text_qk_rope_sample_gate()
    print("[lens-dit-text-qk-rope] samples/qk_values:", stats.samples, stats.qk_values)
    print("[lens-dit-text-qk-rope] finite:", stats.finite_values)
    print(
        "[lens-dit-text-qk-rope] got mean/std/absmax:",
        stats.got_mean,
        stats.got_std,
        stats.got_absmax,
    )
    print("Microsoft Lens block-0 text Q/K RoPE sampled smoke PASS")
