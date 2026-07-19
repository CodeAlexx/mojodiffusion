# lance_t2v_video_smoke.mojo - tiny real-weight Lance T2V temporal artifact.
#
# Runs a small Lance 3B Video denoise with real streamed weights and decodes the
# resulting T_lat=3 latent through the cached Wan2.2 video VAE path. The output
# is intentionally tiny: 9 decoded frames at 16x16. It proves native Mojo T2V
# wiring without requiring production-size block-sparse attention.

from std.gpu.host import DeviceContext

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
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors"
comptime PROMPT = "fairy"
comptime LATENT_T = 3
comptime LATENT_H = 1
comptime LATENT_W = 1
comptime L_TOKENS = LATENT_T * LATENT_H * LATENT_W
comptime PATCH_LATENT_DIM = 48
comptime S_TOTAL = 9
comptime MAX_LAYERS = 0
comptime NUM_STEPS = 2
comptime TIMESTEP_SHIFT = Float32(3.5)
comptime CFG_TEXT_SCALE = Float32(4.0)
comptime SEED = UInt64(4242)
comptime OUT_PREFIX = "/home/alex/mojodiffusion/output/lance_t2v_tiny_video_t3_frame"
comptime OUT_SUFFIX = "_16.png"
comptime OUT_MP4 = "/home/alex/mojodiffusion/output/lance_t2v_tiny_video_t3.mp4"


def _run_lance_denoise(ctx: DeviceContext) raises -> Tensor:
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var input = build_lance_t2v_input(tok, String(PROMPT), LATENT_T, LATENT_H, LATENT_W)
    var uncond = build_lance_t2v_padded_uncond_input(
        input.text_split_len - 2, LATENT_T, LATENT_H, LATENT_W
    )
    if len(input.full_ids) != S_TOTAL:
        raise Error(
            String("lance_t2v_video_smoke: S_TOTAL mismatch, got ")
            + String(len(input.full_ids))
        )
    if len(uncond.full_ids) != S_TOTAL:
        raise Error(
            String("lance_t2v_video_smoke: uncond S_TOTAL mismatch, got ")
            + String(len(uncond.full_ids))
        )

    var x_shape = List[Int]()
    x_shape.append(L_TOKENS)
    x_shape.append(PATCH_LATENT_DIM)
    var x_t = randn(x_shape.copy(), SEED, STDtype.F32, ctx)

    print("[lance-video] loading streamed Lance model")
    var model = LanceT2VOffloaded[S_TOTAL].load(String(MODEL_DIR), ctx)
    print("[lance-video] denoise tiny video grid, layers=", MAX_LAYERS, "steps=", NUM_STEPS)
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
        print("[lance-video] step", step, "cfg velocity shape:", v.shape()[0], v.shape()[1], v.shape()[2])
        var v2 = reshape(v, x_shape.copy(), ctx)
        x_t = lance_denoise_step(x_t, v2, dt, ctx)
    return x_t^


def main() raises:
    var ctx = DeviceContext()
    var x_t = _run_lance_denoise(ctx)
    print("[lance-video] loading Wan2.2 VAE video decoder")
    var vae = Wan22VaeImageDecoder[LATENT_H, LATENT_W].load(String(VAE_PATH), ctx)
    print("[lance-video] decoding video")
    var video = vae.decode_video_tokens(x_t, LATENT_T, ctx)
    var vs = video.shape()
    print("[lance-video] video shape:", vs[0], vs[1], vs[2], vs[3], vs[4])
    var count = save_video_frame_sequence_png(
        video, String(OUT_PREFIX), String(OUT_SUFFIX), LATENT_H, LATENT_W, ctx
    )
    print("[lance-video] saved frames:", count)
    print("[lance-video] first ->", video_frame_path(String(OUT_PREFIX), 0, String(OUT_SUFFIX)))
    print("[lance-video] last ->", video_frame_path(String(OUT_PREFIX), count - 1, String(OUT_SUFFIX)))
    mux_frame_sequence_mp4(String(OUT_PREFIX), String(OUT_SUFFIX), String(OUT_MP4), 8)
    print("[lance-video] mp4 ->", OUT_MP4)
