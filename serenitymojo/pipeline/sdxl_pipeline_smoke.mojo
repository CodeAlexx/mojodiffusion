# pipeline/sdxl_pipeline_smoke.mojo — SDXL text→image end-to-end (cached-embed).
#
# Glue: cached CLIP embeddings -> SDXL Euler denoise (UNet) -> LDM VAE decode ->
# PNG. Mirrors inference-flame/src/bin/sdxl_infer.rs exactly (cached-embedding
# path; the CLIP tokenizer is not ported, so embeddings are produced offline by
# the Rust `sdxl_encode` bin and cached as a safetensors with keys:
#   context [1,77,2048], context_uncond [1,77,2048], y [1,2816], y_uncond [1,2816]).
#
# Denoise (sdxl_infer.rs:117-234), eps-prediction Euler:
#   x = noise * sqrt(sigma_max^2 + 1)
#   for i: c_in = 1/sqrt(sigma^2+1); x_in = x * c_in
#          eps_cond   = UNet(x_in, t_i, context,        y)
#          eps_uncond = UNet(x_in, t_i, context_uncond, y_uncond)
#          eps = eps_uncond + CFG*(eps_cond - eps_uncond)
#          x  += (sigma_next - sigma) * eps
#   image = VAE.decode(x)  (scale 0.13025, shift 0.0)
#
# Schedule/CFG constants from sdxl_infer.rs: scaled-linear betas 0.00085->0.012,
# 1000 train steps, leading spacing steps_offset=1, full-quality STEPS=30,
# CFG=7.5, SEED=42, 1024x1024. This smoke runs one denoise step so the
# cache -> UNet -> VAE path can be exercised during normal development.
# The schedule + Euler helpers live in sampling/sdxl_euler.mojo.
#
# Runtime smoke output: `output/sdxl_one_step_1024.png`. This is intentionally
# under-denoised; it proves cached embeddings, UNet forward, VAE decode, and PNG
# writing meet at 1024x1024 without spending a full 30-step quality run.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import default_manifest_by_id
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.models.dit.sdxl_unet import SDXLUNet
from serenitymojo.models.vae.ldm_decoder import load_sdxl_ldm_decoder
from serenitymojo.models.dit.sdxl_contract import (
    sdxl_default_cached_embeddings_path,
    validate_sdxl_pipeline_contract,
)
from serenitymojo.sampling.sdxl_euler import (
    SDXLEulerScheduler,
    sdxl_cfg,
    sdxl_euler_step,
    sdxl_initial_noise_sigma,
    sdxl_input_scale,
)
from serenitymojo.image.png import save_png, ValueRange


comptime OUT = "/home/alex/mojodiffusion/output/sdxl_one_step_1024.png"

comptime WIDTH = 1024
comptime HEIGHT = 1024
comptime LH = HEIGHT // 8  # latent H (128)
comptime LW = WIDTH // 8   # latent W (128)
comptime FULL_QUALITY_STEPS = 30
comptime NUM_STEPS = 1
comptime CFG = Float32(7.5)
comptime SEED = UInt64(42)


# Load a single named tensor from a one-file safetensors (the cached embeddings)
# as a GPU Tensor, preserving its stored dtype.
def _load_named(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def main() raises:
    var manifest = default_manifest_by_id(String("sdxl"))
    var emb_path = sdxl_default_cached_embeddings_path()
    validate_sdxl_pipeline_contract(manifest.denoiser_path, manifest.vae_path, emb_path)
    var ctx = DeviceContext()
    print("=== SDXL pure-Mojo (cached-embedding path) ===")
    print(
        "  ",
        WIDTH,
        "x",
        HEIGHT,
        " smoke steps",
        NUM_STEPS,
        " full steps",
        FULL_QUALITY_STEPS,
        " CFG",
        CFG,
    )
    print("  contract OK")

    # --- Stage 1: cached CLIP embeddings ---
    var emb = ShardedSafeTensors.open(emb_path)
    var context = _load_named(emb, String("context"), ctx)          # [1,77,2048]
    var context_uncond = _load_named(emb, String("context_uncond"), ctx)
    var y = _load_named(emb, String("y"), ctx)                       # [1,2816]
    var y_uncond = _load_named(emb, String("y_uncond"), ctx)
    print("  context", context.shape()[1], context.shape()[2], " y", y.shape()[1])

    # --- Stage 2: UNet ---
    var unet = SDXLUNet[LH, LW].load(manifest.denoiser_path, ctx)
    print("  UNet loaded")

    # --- Stage 3: noise + Euler denoise (eps prediction) ---
    var sched = SDXLEulerScheduler(NUM_STEPS)
    var sigmas = sched.sigmas()

    # GPU Gaussian noise [1,4,LH,LW], then scale by initial-noise sigma.
    var nsh = List[Int]()
    nsh.append(1)
    nsh.append(4)
    nsh.append(LH)
    nsh.append(LW)
    var noise = randn(nsh^, SEED, STDtype.F32, ctx)
    var init_sigma = sdxl_initial_noise_sigma(sigmas[0])
    var x = mul_scalar(noise, init_sigma, ctx)  # [1,4,LH,LW] F32

    for i in range(NUM_STEPS):
        var sigma = sigmas[i]
        var sigma_next = sigmas[i + 1]
        var t_i = sched.timestep(i)

        # scale input: x_in = x / sqrt(sigma^2+1).
        var c_in = sdxl_input_scale(sigma)
        var x_in_f32 = mul_scalar(x, c_in, ctx)
        var x_in = cast_tensor(x_in_f32, STDtype.BF16, ctx)

        # conditional + unconditional eps predictions.
        var eps_cond = cast_tensor(unet.forward(x_in, t_i, context, y, ctx), STDtype.F32, ctx)
        var eps_uncond = cast_tensor(
            unet.forward(x_in, t_i, context_uncond, y_uncond, ctx), STDtype.F32, ctx
        )

        # CFG: eps_uncond + CFG*(eps_cond - eps_uncond).
        var eps = sdxl_cfg(eps_cond, eps_uncond, CFG, ctx)

        # Euler step: x += (sigma_next - sigma)*eps.
        x = sdxl_euler_step(x, eps, sigma, sigma_next, ctx)
        if i == 0 or i == NUM_STEPS - 1:
            print("  step", i + 1, "/", NUM_STEPS, " sigma", sigma)

    # --- Stage 4: VAE decode + PNG ---
    var vae = load_sdxl_ldm_decoder[LH, LW](manifest.vae_path, ctx)
    var image = vae.decode(x, ctx)  # [1,3,8*LH,8*LW] = [1,3,1024,1024]
    print("  decoded", image.shape()[2], "x", image.shape()[3])

    save_png(image, String(OUT), ctx, ValueRange.SIGNED)
    print("IMAGE SAVED:", OUT)
