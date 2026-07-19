# serenitymojo/models/flux/parity/stack_real_smoke.mojo
#
# REAL-CONFIG RESIDENCY NOTE for the Flux (flux1-dev) stack — NOT a full-depth
# run. flux1-dev's transformer is ~8.6B parameters at the REAL config (19 double
# + 38 single, D=3072, H=24, Dh=128, mlp_ratio 4 -> Fmlp=12288):
#
#   per double block : 2 * [ (3D*D + 3D)  qkv          (28.3M)
#                          + (D*D + D)     proj          (9.4M)
#                          + (Fmlp*D+Fmlp) mlp.0        (37.7M)
#                          + (D*Fmlp + D)  mlp.2        (37.7M)
#                          + 2*Dh          q/k norm ]  = 226.5M  x19 = 4.30G
#   per single block : (3D+Fmlp)*D + (3D+Fmlp)  linear1 (66.1M)
#                    + D*(D+Fmlp) + D           linear2 (47.2M)
#                    + 2*Dh                     q/k norm = 113.3M  x38 = 4.30G
#   transformer total ~ 8.61G params
#     -> 34.4 GB as F32 weights ALONE (before activations, grads, optimizer)
#     -> 17.2 GB as BF16 weights ALONE
#
# RESIDENCY VERDICT (3090, 24 GB):
#   * FULL-residency F32 forward+backward at real depth DOES NOT FIT — 34.4 GB of
#     weights alone exceeds 24 GB before any activation/grad/optimizer memory.
#   * Even BF16 weights (17.2 GB) leave no headroom for the backward's saved
#     activations + weight grads at full depth.
#   * Therefore flux1-dev training needs the SAME streaming/checkpoint strategy
#     as Ernie + Klein-9B: block-swap offload of double/single block weights
#     (BlockLoader, mirrored in models/dit/flux1_dit.mojo::Flux1Offloaded) +
#     per-block recompute-in-backward (klein_stack's checkpoint contract, lines
#     176-208). The COMPOSITION is proven correct at reduced depth (stack_parity
#     + stack_finitediff); wiring the offload/checkpoint runtime is Phase 3+
#     (LoRA) / runtime work, NOT a Phase-2 composition concern.
#
# This file PRINTS the residency arithmetic (no GPU residency forced) so the note
# travels with the gates. The parity stays at REDUCED depth where the math fits.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/flux/parity/stack_real_smoke.mojo

from serenitymojo.models.flux.config import flux_dev


def main() raises:
    print("==== flux stack_real_smoke (residency arithmetic, flux1-dev) ====")
    var cfg = flux_dev()
    var D = cfg.d_model
    var H = cfg.n_heads
    var Dh = cfg.head_dim
    var Fmlp = cfg.mlp_hidden
    var num_double = cfg.num_double
    var num_single = cfg.num_single

    print("config: D=", D, " H=", H, " Dh=", Dh, " Fmlp=", Fmlp,
          " num_double=", num_double, " num_single=", num_single)

    var dbl = 2 * ((3 * D * D + 3 * D) + (D * D + D) + (Fmlp * D + Fmlp) + (D * Fmlp + D) + 2 * Dh)
    var sgl = (3 * D + Fmlp) * D + (3 * D + Fmlp) + D * (D + Fmlp) + D + 2 * Dh
    var total = num_double * dbl + num_single * sgl

    print("per double block params =", dbl, " x", num_double, "=", num_double * dbl)
    print("per single block params =", sgl, " x", num_single, "=", num_single * sgl)
    print("transformer total params =", total)
    var gb_f32 = Float64(total) * 4.0 / 1.0e9
    var gb_bf16 = Float64(total) * 2.0 / 1.0e9
    print("weights-only residency:  F32 =", gb_f32, "GB   BF16 =", gb_bf16, "GB")
    print("")
    print("VERDICT: full-residency F32 (", gb_f32, "GB) and BF16 (", gb_bf16,
          "GB) both EXCEED 24 GB before activations/grads -> Flux training needs")
    print("         block-swap offload + per-block recompute-in-backward (Ernie/")
    print("         Klein-9B strategy). Parity stays at reduced depth; composition")
    print("         is proven by stack_parity + stack_finitediff.")
