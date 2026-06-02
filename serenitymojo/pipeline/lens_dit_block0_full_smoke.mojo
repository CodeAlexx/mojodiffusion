# Microsoft Lens block-0 FULL forward (compile-only, bounded) smoke.
#
# Exercises the full dual-stream block-0 forward end-to-end on synthetic
# bounded inputs (N_IMG=64, N_TXT=64, HIDDEN=1536). Uses REAL block-0
# transformer weights via ShardedSafeTensors but does NOT compare against
# any capture. This is a structural/compile gate that extends past the
# existing QKV + QK RoPE smokes to cover joint SDPA, output projections,
# residual+gating, and SwiGLU MLP for both image and text streams.

from serenitymojo.models.lens.lens_dit_math import (
    validate_lens_block0_full_smoke_gate,
)


def main() raises:
    var stats = validate_lens_block0_full_smoke_gate()
    print("[lens-dit-block0-full] n_img/n_txt:", stats.n_img, stats.n_txt)
    print(
        "[lens-dit-block0-full] img_values/img_finite:",
        stats.img_values,
        stats.img_finite,
    )
    print(
        "[lens-dit-block0-full] txt_values/txt_finite:",
        stats.txt_values,
        stats.txt_finite,
    )
    print(
        "[lens-dit-block0-full] img mean/std/absmax:",
        stats.img_mean,
        stats.img_std,
        stats.img_absmax,
    )
    print(
        "[lens-dit-block0-full] txt mean/std/absmax:",
        stats.txt_mean,
        stats.txt_std,
        stats.txt_absmax,
    )
    print("Microsoft Lens block-0 full forward bounded smoke PASS")
