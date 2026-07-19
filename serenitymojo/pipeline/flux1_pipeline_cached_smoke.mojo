# flux1_pipeline_cached_smoke.mojo - FLUX.1-dev cached-input runtime smoke.
#
# Uses the Rust oracle bundle written by:
#   FLUX1_SAVE_INPUTS=1 cargo run --release --bin flux1_infer
#
# This bypasses Mojo CLIP/T5 placeholder token IDs and starts from the exact
# cached `img_packed`, `t5_hidden`, and `clip_pooled` tensors used by Rust.
#
# The 1024 VAE decode is intentionally a separate DeviceContext phase. A prior
# monolithic path completed all 20 denoise steps but OOMed when the full-frame
# VAE loaded beside post-DiT allocator state. This smoke now host-stages only the
# final packed latent, lets the DiT/text tensors drop, then uses the shared 5x5
# low-memory tiled FLUX decode before writing the PNG.

from std.gpu.host import DeviceContext

from serenitymojo.image.png import ValueRange, save_png
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.models.dit.flux1_contract import (
    flux1_default_cached_inputs_path,
    validate_flux1_cached_inputs_header,
    validate_flux1_pipeline_contract,
)
from serenitymojo.models.dit.flux1_dit import (
    Flux1Config,
    Flux1Offloaded,
    build_flux1_rope_tables,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul_scalar, permute, reshape
from serenitymojo.pipeline.flux_tiled_decode import flux_tiled_decode_5x5_lowmem
from serenitymojo.registry.checkpoints import default_manifest_by_id
from serenitymojo.sampling.flux1_dev import (
    build_flux1_packed_latent_plan,
    build_flux1_sigma_schedule,
)
from serenitymojo.tensor import Tensor


comptime OUT = "/home/alex/mojodiffusion/output/flux1_cached_inputs.png"

comptime HEIGHT = 1024
comptime WIDTH = 1024
comptime AE_IN_CHANNELS = 16
comptime LATENT_H = 2 * ((HEIGHT + 15) // 16)
comptime LATENT_W = 2 * ((WIDTH + 15) // 16)
comptime IMG_H2 = (HEIGHT + 15) // 16
comptime IMG_W2 = (WIDTH + 15) // 16
comptime N_IMG = IMG_H2 * IMG_W2
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime STEPS = 20
comptime GUIDANCE = Float32(3.5)


def _load_named(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _to_bf16(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    if x.dtype() == STDtype.BF16:
        return cast_tensor(x, STDtype.BF16, ctx)
    if x.dtype() == STDtype.F16:
        var x_f32 = cast_tensor(x, STDtype.F32, ctx)
        return cast_tensor(x_f32, STDtype.BF16, ctx)
    return cast_tensor(x, STDtype.BF16, ctx)


def _unpack_latent(packed: Tensor, ctx: DeviceContext) raises -> Tensor:
    var s6 = List[Int]()
    s6.append(1)
    s6.append(IMG_H2)
    s6.append(IMG_W2)
    s6.append(AE_IN_CHANNELS)
    s6.append(2)
    s6.append(2)
    var t6 = reshape(packed, s6^, ctx)
    var p = List[Int]()
    p.append(0)
    p.append(3)
    p.append(1)
    p.append(4)
    p.append(2)
    p.append(5)
    var tp = permute(t6, p^, ctx)
    var sp = List[Int]()
    sp.append(1)
    sp.append(AE_IN_CHANNELS)
    sp.append(LATENT_H)
    sp.append(LATENT_W)
    return reshape(tp, sp^, ctx)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    if len(h) == 0:
        raise Error(String("empty tensor stats: ") + name)
    var min_v = h[0]
    var max_v = h[0]
    var sum_abs: Float64 = 0.0
    var sum_v: Float64 = 0.0
    for i in range(len(h)):
        var v = h[i]
        if v < min_v:
            min_v = v
        if v > max_v:
            max_v = v
        var av = v
        if av < 0.0:
            av = -av
        sum_abs += Float64(av)
        sum_v += Float64(v)
    print(
        "[stats]",
        name,
        "rank",
        len(t.shape()),
        "numel",
        len(h),
        "mean",
        Float32(sum_v / Float64(len(h))),
        "mean_abs",
        Float32(sum_abs / Float64(len(h))),
        "min",
        min_v,
        "max",
        max_v,
    )


def _print_vram(tag: String) raises:
    var mem = cu_mem_get_info()
    print("[vram]", tag, "used", mem.used_bytes() // (1024 * 1024), "MiB",
          "free", mem.free_bytes // (1024 * 1024), "MiB")


def _packed_shape() -> List[Int]:
    var sh = List[Int]()
    sh.append(1)
    sh.append(N_IMG)
    sh.append(AE_IN_CHANNELS * 4)
    return sh^


def denoise_cached(inputs_path: String, denoiser_path: String) raises -> List[Float32]:
    # Return the final packed latent on host. This mirrors flux_sample_cli's
    # staged loading. The DeviceContext is deliberately scoped to this function
    # so all offloaded DiT/text allocations drop before the VAE phase starts.
    var ctx = DeviceContext()
    _print_vram("flux denoise phase start")
    var inputs = ShardedSafeTensors.open(inputs_path)
    var img = _load_named(inputs, String("img_packed"), ctx)
    var txt = _to_bf16(_load_named(inputs, String("t5_hidden"), ctx), ctx)
    var vector = _to_bf16(_load_named(inputs, String("clip_pooled"), ctx), ctx)
    _stats(String("img_packed_initial"), img, ctx)

    print("[dit] FLUX.1 offloaded DiT")
    var model = Flux1Offloaded.load(denoiser_path, Flux1Config.dev(), ctx)
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, 24, 128](
        IMG_H2, IMG_W2, ctx, STDtype.BF16
    )
    var sched = build_flux1_sigma_schedule(STEPS, N_IMG)
    print("[denoise]", STEPS, "steps, guidance", GUIDANCE)
    for i in range(STEPS):
        var t_curr = sched[i]
        var t_prev = sched[i + 1]
        var tvals = List[Float32]()
        tvals.append(t_curr * 1000.0)
        var tsh = List[Int]()
        tsh.append(1)
        var t_vec = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)

        var gvals = List[Float32]()
        gvals.append(GUIDANCE * 1000.0)
        var gsh = List[Int]()
        gsh.append(1)
        var g_vec = Tensor.from_host(gvals, gsh^, STDtype.F32, ctx)

        var img_bf = cast_tensor(img, STDtype.BF16, ctx)
        var pred = cast_tensor(
            model.forward[N_IMG, N_TXT, S](
                img_bf, txt, t_vec, Optional[Tensor](g_vec^), vector, rope[0], rope[1], ctx
            ),
            STDtype.F32,
            ctx,
        )
        var dt = t_prev - t_curr
        img = add(img, mul_scalar(pred, dt, ctx), ctx)
        if i == 0 or i == STEPS - 1:
            _stats(String("pred_step_") + String(i + 1), pred, ctx)

    _stats(String("img_packed_denoised"), img, ctx)
    var packed_h = cast_tensor(img, STDtype.F32, ctx).to_host(ctx)
    _print_vram("flux denoise phase end")
    return packed_h^


def main() raises:
    var manifest = default_manifest_by_id(String("flux1_dev"))
    validate_flux1_pipeline_contract(manifest)
    var inputs_path = flux1_default_cached_inputs_path()
    validate_flux1_cached_inputs_header(inputs_path)
    var plan = build_flux1_packed_latent_plan(WIDTH, HEIGHT, N_TXT)
    plan.validate_dev_1024_contract()

    print("=== FLUX.1 Dev cached-input smoke ===", HEIGHT, "x", WIDTH, STEPS, "steps")
    print("[contract] cached inputs:", inputs_path)

    var packed_h = denoise_cached(inputs_path, manifest.denoiser_path.copy())
    var ctx = DeviceContext()
    ctx.synchronize()
    cu_mempool_trim_current(0)
    ctx.synchronize()

    print("[vae] unpack + tiled decode (5x5 lowmem overlap+blend)")
    _print_vram("flux decode phase start")
    var packed = Tensor.from_host(packed_h, _packed_shape(), STDtype.F32, ctx)
    var latent = _unpack_latent(packed, ctx)
    _stats(String("latent_unpacked"), latent, ctx)
    var rgb = flux_tiled_decode_5x5_lowmem[LATENT_H, LATENT_W](
        latent, manifest.vae_path.copy(), ctx
    )
    _stats(String("rgb"), rgb, ctx)
    _print_vram("flux decode phase end")
    save_png(rgb, OUT, ctx, ValueRange.SIGNED)
    print("[done] saved", OUT)
