# lance_t2v_256_9f_dense_probe.mojo - Lance production-shape dense video probe.
#
# This is not a quality target. It runs the 256x256, 9-frame Lance static
# profile with real weights, one denoise step, and all streamed layers to prove
# the production token geometry before the cached/sparse attention path lands.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.components.artifacts import (
    mux_frame_sequence_mp4,
    save_video_frame_sequence_png,
    video_frame_path,
)
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.lance.lance_t2v import (
    LanceT2VOffloaded,
    build_lance_t2v_input,
    build_lance_t2v_padded_uncond_input,
)
from serenitymojo.models.vae.wan22_decoder import Wan22VaeImageDecoder
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.sampling.lance_t2v import (
    build_lance_timestep_schedule,
    lance_cfg,
    lance_cfg_renorm,
    lance_denoise_step,
    lance_timestep_tensor,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime MODEL_DIR = "/home/alex/.serenity/models/lance/Lance_3B_Video"
comptime TOK_JSON = "/home/alex/.serenity/models/lance/Lance_3B_Video/tokenizer.json"
comptime VAE_PATH = "/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors"
comptime PROMPT = "fairy"
comptime LATENT_T = 3
comptime LATENT_H = 16
comptime LATENT_W = 16
comptime L_TOKENS = LATENT_T * LATENT_H * LATENT_W
comptime PATCH_LATENT_DIM = 48
comptime S_TOTAL = 774
comptime MAX_LAYERS = 0
comptime NUM_STEPS = 1
comptime TIMESTEP_SHIFT = Float32(3.5)
comptime CFG_TEXT_SCALE = Float32(4.0)
comptime SEED = UInt64(4242)
comptime OUT_PREFIX = "/home/alex/mojodiffusion/output/lance_t2v_256_9f_dense_frame"
comptime OUT_SUFFIX = "_256.png"
comptime OUT_MP4 = "/home/alex/mojodiffusion/output/lance_t2v_256_9f_dense.mp4"


def _stats(name: String, x: Tensor, ctx: DeviceContext) raises:
    var vals = x.to_host(ctx)
    var n = len(vals)
    if n == 0:
        raise Error(String("empty tensor stats: ") + name)
    var sum = Float64(0.0)
    var absmax = Float32(0.0)
    for i in range(n):
        var v = vals[i]
        sum += Float64(v)
        var av = v
        if av < 0:
            av = -av
        if av > absmax:
            absmax = av
    var mean = Float32(sum / Float64(n))
    var varsum = Float64(0.0)
    for i in range(n):
        var d = vals[i] - mean
        varsum += Float64(d * d)
    print(
        "[lance-256-probe]",
        name,
        "shape",
        x.shape()[0],
        x.shape()[1],
        "mean=",
        mean,
        "std=",
        sqrt(Float32(varsum / Float64(n))),
        "absmax=",
        absmax,
        "n=",
        n,
    )


def _run_lance_denoise(ctx: DeviceContext) raises -> Tensor:
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var input = build_lance_t2v_input(tok, String(PROMPT), LATENT_T, LATENT_H, LATENT_W)
    var uncond = build_lance_t2v_padded_uncond_input(
        input.text_split_len - 2, LATENT_T, LATENT_H, LATENT_W
    )
    if len(input.full_ids) != S_TOTAL:
        raise Error(
            String("lance_t2v_256_9f_dense_probe: S_TOTAL mismatch, got ")
            + String(len(input.full_ids))
        )
    if len(uncond.full_ids) != S_TOTAL:
        raise Error(
            String("lance_t2v_256_9f_dense_probe: uncond S_TOTAL mismatch, got ")
            + String(len(uncond.full_ids))
        )

    var x_shape = List[Int]()
    x_shape.append(L_TOKENS)
    x_shape.append(PATCH_LATENT_DIM)
    var x_t = randn(x_shape.copy(), SEED, STDtype.F32, ctx)
    _stats(String("init_latent"), x_t, ctx)

    print(
        "[lance-256-probe] loading streamed Lance model S=",
        S_TOTAL,
        "tokens=",
        L_TOKENS,
        "layers=",
        MAX_LAYERS,
        "steps=",
        NUM_STEPS,
    )
    var model = LanceT2VOffloaded[S_TOTAL].load(String(MODEL_DIR), ctx)
    var schedule = build_lance_timestep_schedule(NUM_STEPS, TIMESTEP_SHIFT)
    for step in range(NUM_STEPS):
        var t0 = schedule[step]
        var t1 = schedule[step + 1]
        var dt = t0 - t1
        var t_step = lance_timestep_tensor(L_TOKENS, t0, ctx)
        var v_cond = model.forward_velocity(input, x_t, t_step, MAX_LAYERS, ctx)
        var v_uncond = model.forward_velocity(uncond, x_t, t_step, MAX_LAYERS, ctx)
        var v_cfg = lance_cfg(v_uncond, v_cond, CFG_TEXT_SCALE, ctx)
        var v = lance_cfg_renorm(v_cfg, v_cond, Float32(0.0), Float32(1.0), ctx)
        print(
            "[lance-256-probe] step",
            step,
            "cfg velocity shape:",
            v.shape()[0],
            v.shape()[1],
            v.shape()[2],
        )
        _stats(String("velocity"), reshape(v, x_shape.copy(), ctx), ctx)
        var v2 = reshape(v, x_shape.copy(), ctx)
        x_t = lance_denoise_step(x_t, v2, dt, ctx)
    _stats(String("final_latent"), x_t, ctx)
    return x_t^


def main() raises:
    var ctx = DeviceContext()
    var x_t = _run_lance_denoise(ctx)
    print("[lance-256-probe] loading Wan2.2 VAE video decoder")
    var vae = Wan22VaeImageDecoder[LATENT_H, LATENT_W].load(String(VAE_PATH), ctx)
    print("[lance-256-probe] decoding video")
    var video = vae.decode_video_tokens(x_t, LATENT_T, ctx)
    var vs = video.shape()
    print("[lance-256-probe] video shape:", vs[0], vs[1], vs[2], vs[3], vs[4])
    var count = save_video_frame_sequence_png(
        video, String(OUT_PREFIX), String(OUT_SUFFIX), LATENT_H, LATENT_W, ctx
    )
    print("[lance-256-probe] saved frames:", count)
    print("[lance-256-probe] first ->", video_frame_path(String(OUT_PREFIX), 0, String(OUT_SUFFIX)))
    print(
        "[lance-256-probe] last ->",
        video_frame_path(String(OUT_PREFIX), count - 1, String(OUT_SUFFIX)),
    )
    mux_frame_sequence_mp4(String(OUT_PREFIX), String(OUT_SUFFIX), String(OUT_MP4), 8)
    print("[lance-256-probe] mp4 ->", OUT_MP4)
    print("Lance 256x256 9-frame dense video probe PASS")
