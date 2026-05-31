# models/klein/train.mojo — Klein LoRA training entry point (thin driver).
#
# All logic lives in serenitymojo.training.train_step (shared). This file is the
# per-model entry: pick the config, call the shared synthetic driver. The full
# split goal — pipeline written once, model files thin — realized here:
# train_klein.mojo went from 595 lines to this.
#
# SCAFFOLD (synthetic): exercises the proven step (dit_block fwd/bwd, flow_match,
# AdamW, LoRA path) at down-scaled comptime dims. Real run = swap comptime dims
# for cfg dims + add weight loader (GAP G1) + Klein double-stream block (GAP G3).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/models/klein/train.mojo

from std.gpu.host import DeviceContext
from serenitymojo.training.train_step import run_synthetic
from serenitymojo.models.klein.config import klein_9b, klein_4b


def main() raises:
    var ctx = DeviceContext()
    print("############################################################")
    print("# Klein LoRA training (SCAFFOLD — synthetic, shared pipeline).")
    print("############################################################")
    run_synthetic(klein_9b(), ctx)
    run_synthetic(klein_4b(), ctx)
