# validation_sampler_smoke.mojo — the L2P sample-shift gate, end to end.
#
# Generates TWO validation images from the SAME seed + caps:
#   (1) baseline   — no LoRA merged,
#   (2) with-LoRA  — the trained LoRA merged into the resident DiT,
# then reports pixel_l1(baseline, with_lora). A diff of ~0 means the LoRA is NOT
# changing the output (the exact bug memory flags as never-checked for L2P); a
# diff > 0 proves the adapter is live at inference.
#
# This is the integration smoke for training/validation_sampler.mojo. It is
# RESOURCE-HEAVY (loads the full resident Klein9B twice) — run it only on a free
# GPU, after the compile lock frees:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/validation_sampler_smoke.mojo
#
# Caps must already exist on disk (produced by a SEPARATE encode process, e.g.
# pipeline/klein9b_encode_smoke.mojo) so the Qwen3 encoder and the 9B DiT never
# co-reside on the 24 GB card.

from std.gpu.host import DeviceContext
from serenitymojo.training.validation_sampler import (
    ValidationCaps,
    load_caps,
    generate_validation,
    pixel_l1,
)


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime CAPS_POS = "/home/alex/mojodiffusion/output/klein9b_caps_pos.bin"
comptime CAPS_NEG = "/home/alex/mojodiffusion/output/klein9b_caps_neg.bin"
# A real trained LoRA (same file the lora wiring smoke uses).
comptime LORA_PATH = "/home/alex/EriDiffusion/EriDiffusion-v2/output/klein_lr3e4_const_b1/klein_lora_step200.safetensors"
comptime OUT_BASE = "/home/alex/mojodiffusion/output/klein_validation_baseline.png"
comptime OUT_LORA = "/home/alex/mojodiffusion/output/klein_validation_with_lora.png"

# Native 1024 grid: 64x64 latent token grid (output/16). Mirrors the multistep
# smoke's grid constants.
comptime N_IMG = 4096
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime LH = 64
comptime LW = 64

comptime NUM_STEPS = 20
comptime CFG = Float32(4.0)
comptime SEED = UInt64(42)
comptime LORA_MULT = Float32(1.0)


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein validation sampler — L2P sample-shift gate ===")
    var caps = load_caps(String(CAPS_POS), String(CAPS_NEG), ctx)

    # (1) baseline — no LoRA.
    print("[1/2] baseline generation (no LoRA)")
    var img_base = generate_validation[N_IMG, N_TXT, S, LH, LW](
        String(KLEIN9B_PATH), String(VAE_PATH), caps,
        String(""), LORA_MULT, NUM_STEPS, CFG, SEED, String(OUT_BASE), ctx,
    )

    # Caps were consumed (move-only) into the first call? No — generate_validation
    # borrows `caps` (its Tensors are read inside forward_full each step, not
    # moved out). But to be safe across move semantics we reload them for pass 2.
    var caps2 = load_caps(String(CAPS_POS), String(CAPS_NEG), ctx)

    # (2) with-LoRA.
    print("[2/2] with-LoRA generation")
    var img_lora = generate_validation[N_IMG, N_TXT, S, LH, LW](
        String(KLEIN9B_PATH), String(VAE_PATH), caps2,
        String(LORA_PATH), LORA_MULT, NUM_STEPS, CFG, SEED, String(OUT_LORA), ctx,
    )

    var d = pixel_l1(img_base, img_lora, ctx)
    print("  pixel_l1(baseline, with_lora) =", d)
    if d == Float32(0.0):
        print("  !!! GATE FAIL: LoRA had ZERO effect on the output (not applied)")
    else:
        print("  GATE PASS: LoRA shifts the sample (diff > 0)")
    print("validation_sampler smoke complete")
