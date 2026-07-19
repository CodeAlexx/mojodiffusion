# models/zimage/train.mojo — Z-Image LoRA training entry point (thin driver).
#
# All logic lives in serenitymojo.training.train_step (shared). Per-model entry:
# pick config, call the shared synthetic driver.
#
# ── sdpa H=30 status (UPDATED 2026-05-30) ────────────────────────────────────
# The earlier "sdpa_backward silently zeros d_q/d_k at H=30" blocker was found to
# be a DEGENERATE-TEST-DATA artifact, NOT a kernel bug (MEASURED: non-degenerate
# gate cos>=0.999 at H=30; the old gate's V-fill aliased mod 9 at H*Dh=3840 making
# V constant across the sequence, so grad_scores=0 was the correct answer and
# torch agreed). See project_mojo_sdpa_h30_blocker_false_2026-05-30 +
# sdpa_bwd_nondegen_parity.mojo. Z-Image is NOT blocked. (One precision watch:
# S=2304 d_k cos 0.9975 — F32 accumulation order, not corruption.)
#
# SCAFFOLD (synthetic): proven step at down-scaled comptime dims. Real run = swap
# comptime dims for cfg dims (H=30) + weight loader (GAP G1) + adaLN modulation
# wiring (GAP G3).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/models/zimage/train.mojo

from std.gpu.host import DeviceContext
from serenitymojo.training.train_step import run_synthetic
from serenitymojo.models.zimage.config import zimage


def main() raises:
    var ctx = DeviceContext()
    print("############################################################")
    print("# Z-Image LoRA training (SCAFFOLD — synthetic, shared pipeline).")
    print("############################################################")
    run_synthetic(zimage(), ctx)
