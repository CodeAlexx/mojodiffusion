# flux1_pipeline_cached_smoke.mojo - FLUX.1-dev cached-input runtime smoke.
#
# Uses the Rust oracle bundle written by:
#   FLUX1_SAVE_INPUTS=1 cargo run --release --bin flux1_infer
#
# This bypasses Mojo CLIP/T5 placeholder token IDs and starts from the exact
# cached `img_packed`, `t5_hidden`, and `clip_pooled` tensors used by Rust.

from std.gpu.host import DeviceContext

from serenitymojo.image.png import ValueRange, save_png
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
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
from serenitymojo.models.vae.ldm_decoder import load_flux1_ldm_decoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul_scalar, permute, reshape
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


def main() raises:
    var manifest = default_manifest_by_id(String("flux1_dev"))
    validate_flux1_pipeline_contract(manifest)
    var inputs_path = flux1_default_cached_inputs_path()
    validate_flux1_cached_inputs_header(inputs_path)
    var plan = build_flux1_packed_latent_plan(WIDTH, HEIGHT, N_TXT)
    plan.validate_dev_1024_contract()

    var ctx = DeviceContext()
    print("=== FLUX.1 Dev cached-input smoke ===", HEIGHT, "x", WIDTH, STEPS, "steps")
    print("[contract] cached inputs:", inputs_path)

    var inputs = ShardedSafeTensors.open(inputs_path)
    var img = _load_named(inputs, String("img_packed"), ctx)
    var txt = _to_bf16(_load_named(inputs, String("t5_hidden"), ctx), ctx)
    var vector = _to_bf16(_load_named(inputs, String("clip_pooled"), ctx), ctx)
    _stats(String("img_packed_initial"), img, ctx)

    print("[dit] FLUX.1 offloaded DiT")
    var model = Flux1Offloaded.load(manifest.denoiser_path, Flux1Config.dev(), ctx)
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
    print("[vae] unpack + decode")
    var latent = _unpack_latent(img, ctx)
    _stats(String("latent_unpacked"), latent, ctx)
    var vae = load_flux1_ldm_decoder[LATENT_H, LATENT_W](manifest.vae_path, ctx)
    var rgb = vae.decode(latent, ctx)
    _stats(String("rgb"), rgb, ctx)
    save_png(rgb, OUT, ctx, ValueRange.SIGNED)
    print("[done] saved", OUT)
