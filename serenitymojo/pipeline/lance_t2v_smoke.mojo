# pipeline/lance_t2v_smoke.mojo — native Lance T2V streamed-weight smoke.
#
# Runs a tiny text_template=false T2V forward on GPU using the real
# Lance_3B_Video checkpoint. This is intentionally small (1x2x2 latent grid,
# one decoder layer by default) so it validates loader/embedding/MoE-gen/mRoPE
# wiring without requiring the still-missing block-sparse video attention.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.sampling.lance_t2v import (
    build_lance_timestep_schedule,
    lance_cfg,
    lance_cfg_renorm,
    lance_denoise_step,
    lance_timestep_tensor,
)
from serenitymojo.models.lance.lance_t2v import (
    LanceT2VOffloaded,
    build_lance_t2v_input,
    build_lance_t2v_padded_uncond_input,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime MODEL_DIR = "/home/alex/.serenity/models/lance/Lance_3B_Video"
comptime TOK_JSON = "/home/alex/.serenity/models/lance/Lance_3B_Video/tokenizer.json"
comptime PROMPT = "fairy"
comptime LATENT_T = 1
comptime LATENT_H = 2
comptime LATENT_W = 2
comptime L_TOKENS = LATENT_T * LATENT_H * LATENT_W
comptime PATCH_LATENT_DIM = 48
comptime S_TOTAL = 10
comptime MAX_LAYERS = 0  # 0 means all Lance decoder layers.
comptime NUM_STEPS = 2
comptime TIMESTEP_SHIFT = Float32(3.5)
comptime CFG_TEXT_SCALE = Float32(4.0)
comptime SEED = UInt64(4242)


def main() raises:
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var input = build_lance_t2v_input(tok, String(PROMPT), LATENT_T, LATENT_H, LATENT_W)
    var uncond = build_lance_t2v_padded_uncond_input(
        input.text_split_len - 2, LATENT_T, LATENT_H, LATENT_W
    )
    if len(input.full_ids) != S_TOTAL:
        raise Error(
            String("lance_t2v_smoke: S_TOTAL mismatch, got ")
            + String(len(input.full_ids))
        )
    if len(uncond.full_ids) != S_TOTAL:
        raise Error(
            String("lance_t2v_smoke: uncond S_TOTAL mismatch, got ")
            + String(len(uncond.full_ids))
        )

    var x_shape = List[Int]()
    x_shape.append(L_TOKENS)
    x_shape.append(PATCH_LATENT_DIM)
    var x_t = randn(x_shape.copy(), SEED, STDtype.F32, ctx)

    print("[lance] loading streamed model")
    var model = LanceT2VOffloaded[S_TOTAL].load(String(MODEL_DIR), ctx)
    print("[lance] denoise tiny grid, layers=", MAX_LAYERS, "steps=", NUM_STEPS)
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
        print("[lance] step", step, "cfg velocity shape:", v.shape()[0], v.shape()[1], v.shape()[2])
        var v2 = reshape(v, x_shape.copy(), ctx)
        x_t = lance_denoise_step(x_t, v2, dt, ctx)

    var vals = x_t.to_host(ctx)
    print("[lance] final latent values:", vals[0], vals[1], vals[2], vals[3])
