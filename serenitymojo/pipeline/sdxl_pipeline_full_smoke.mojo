# pipeline/sdxl_pipeline_full_smoke.mojo — SDXL 30-step cached-embed run.
#
# Same path as `sdxl_pipeline_smoke.mojo`, but uses the full 30-step SDXL
# denoise loop and writes a separate PNG artifact. This is intentionally a
# long GPU smoke, not a default quick development check.
# Like the quick smoke, it host-stages only the final latent between denoise and
# VAE so the UNet/embedding tensors drop before tiled 1024 decode.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import default_manifest_by_id
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.models.dit.sdxl_unet import SDXLUNet
from serenitymojo.models.vae.sdxl_tiled_decode import sdxl_tiled_decode
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


comptime OUT = "/home/alex/mojodiffusion/output/sdxl_30step_1024.png"

comptime WIDTH = 1024
comptime HEIGHT = 1024
comptime LH = HEIGHT // 8
comptime LW = WIDTH // 8
comptime NUM_STEPS = 30
comptime CFG = Float32(7.5)
comptime SEED = UInt64(42)


def _load_named(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _latent_shape() -> List[Int]:
    var sh = List[Int]()
    sh.append(1)
    sh.append(4)
    sh.append(LH)
    sh.append(LW)
    return sh^


def _print_vram(tag: String) raises:
    var mem = cu_mem_get_info()
    print("[vram]", tag, "used", mem.used_bytes() // (1024 * 1024), "MiB",
          "free", mem.free_bytes // (1024 * 1024), "MiB")


def denoise_cached(denoiser_path: String, emb_path: String) raises -> List[Float32]:
    # Return only the final latent on host. This lets the UNet and CLIP sidecar
    # tensors drop before the 1024 VAE stage is loaded.
    var ctx = DeviceContext()
    _print_vram("sdxl denoise phase start")
    var emb = ShardedSafeTensors.open(emb_path)
    var context = _load_named(emb, String("context"), ctx)
    var context_uncond = _load_named(emb, String("context_uncond"), ctx)
    var y = _load_named(emb, String("y"), ctx)
    var y_uncond = _load_named(emb, String("y_uncond"), ctx)
    print("  context", context.shape()[1], context.shape()[2], " y", y.shape()[1])

    var unet = SDXLUNet[LH, LW].load(denoiser_path, ctx)
    print("  UNet loaded")

    var sched = SDXLEulerScheduler(NUM_STEPS)
    var sigmas = sched.sigmas()

    var noise = randn(_latent_shape(), SEED, STDtype.F32, ctx)
    var init_sigma = sdxl_initial_noise_sigma(sigmas[0])
    var x = mul_scalar(noise, init_sigma, ctx)

    for i in range(NUM_STEPS):
        var sigma = sigmas[i]
        var sigma_next = sigmas[i + 1]
        var t_i = sched.timestep(i)

        var c_in = sdxl_input_scale(sigma)
        var x_in_f32 = mul_scalar(x, c_in, ctx)
        var x_in = cast_tensor(x_in_f32, STDtype.BF16, ctx)

        var eps_cond = cast_tensor(unet.forward(x_in, t_i, context, y, ctx), STDtype.F32, ctx)
        var eps_uncond = cast_tensor(
            unet.forward(x_in, t_i, context_uncond, y_uncond, ctx), STDtype.F32, ctx
        )
        var eps = sdxl_cfg(eps_cond, eps_uncond, CFG, ctx)
        x = sdxl_euler_step(x, eps, sigma, sigma_next, ctx)
        if i == 0 or (i + 1) % 5 == 0 or i == NUM_STEPS - 1:
            print("  step", i + 1, "/", NUM_STEPS, " sigma", sigma)

    var latent_h = cast_tensor(x, STDtype.F32, ctx).to_host(ctx)
    _print_vram("sdxl denoise phase end")
    return latent_h^


def main() raises:
    var manifest = default_manifest_by_id(String("sdxl"))
    var emb_path = sdxl_default_cached_embeddings_path()
    validate_sdxl_pipeline_contract(manifest.denoiser_path, manifest.vae_path, emb_path)
    var denoiser_path = manifest.denoiser_path.copy()
    var vae_path = manifest.vae_path.copy()
    print("=== SDXL pure-Mojo full cached-embedding run ===")
    print("  ", WIDTH, "x", HEIGHT, " steps", NUM_STEPS, " CFG", CFG)
    print("  contract OK")

    var latent_h = denoise_cached(denoiser_path, emb_path)
    var ctx = DeviceContext()
    ctx.synchronize()
    cu_mempool_trim_current(0)
    ctx.synchronize()
    _print_vram("sdxl decode phase start")
    var x = Tensor.from_host(latent_h, _latent_shape(), STDtype.F32, ctx)
    var image = sdxl_tiled_decode[LH, LW](x, vae_path, ctx)
    _print_vram("sdxl decode phase end")
    print("  decoded", image.shape()[2], "x", image.shape()[3])

    save_png(image, String(OUT), ctx, ValueRange.SIGNED)
    print("IMAGE SAVED:", OUT)
