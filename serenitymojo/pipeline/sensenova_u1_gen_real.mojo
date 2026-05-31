# pipeline/sensenova_u1_gen_real.mojo — SenseNova-U1-8B-MoT T2I REAL-resolution
# coherence test (not a 2-step smoke).
#
# Differences vs sensenova_u1_gen_smoke.mojo:
#   * Uses the REAL SYSTEM_MESSAGE_FOR_GEN (sensenova_u1_gen.rs:27-46) so the
#     text prefix matches what the model was trained to condition on.
#   * NUM_STEPS = 30 (a real denoise schedule; 2 steps cannot denoise pure noise).
#   * 256x256 resolution -> grid 16x16, token 8x8, L_TOKENS=64. noise_scale=1.0.
#   * TEXT_LEN / L_TOKENS comptime params are only struct tags (attention is
#     runtime-dimensioned); actual seq lengths flow from len(token_ids) /
#     token_h*token_w. We size L_TOKENS to the real value and pin TEXT_LEN to a
#     conservative upper bound; the exact cond/uncond token counts are printed
#     and NOT asserted.
#
# Reference: inference-flame/src/bin/sensenova_u1_gen.rs::run_t2i.
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    reshape, permute, add, sub, mul_scalar, add_scalar,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.dit.sensenova_u1 import (
    SenseNovaU1, SenseNovaU1Config,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime WEIGHTS_DIR = "/home/alex/.serenity/models/sensenova_u1"
comptime VOCAB_JSON = "/home/alex/.serenity/models/sensenova_u1/vocab.json"
comptime MERGES_TXT = "/home/alex/.serenity/models/sensenova_u1/merges.txt"
comptime ADDED_TOKENS_JSON = "/home/alex/.serenity/models/sensenova_u1/added_tokens.json"
comptime OUTPUT = "/home/alex/mojodiffusion/output/sensenova_u1_real_256.png"
comptime PROMPT = "a photo of a cat"

# Real-resolution geometry. 256x256 image.
comptime WIDTH = 256
comptime HEIGHT = 256
comptime PATCH = 16
comptime MERGE = 2
comptime GRID_H = HEIGHT // PATCH        # 16
comptime GRID_W = WIDTH // PATCH         # 16
comptime TOKEN_H = GRID_H // MERGE       # 8
comptime TOKEN_W = GRID_W // MERGE       # 8
comptime L_TOKENS = TOKEN_H * TOKEN_W    # 64
comptime TEXT_LEN = 320                  # struct-tag upper bound (real count printed)
comptime FM_OUT = (PATCH * MERGE) * (PATCH * MERGE) * 3  # 3072

comptime NUM_STEPS = 30
comptime CFG_SCALE = Float32(4.0)
comptime TIMESTEP_SHIFT = Float32(3.0)
comptime SEED = UInt64(42)
comptime T_EPS = Float32(0.05)

# The real system message conditioning the gen prefix (sensenova_u1_gen.rs:27-46).
comptime SYSTEM_MESSAGE_FOR_GEN = String(
    "You are an image generation and editing assistant that accurately understands and executes "
    + "user intent.\n\nYou support two modes:\n\n"
    + "1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. "
    + "Put all reasoning inside the block using plain text. DO NOT include any image tags. "
    + "Keep it reasonable and directly useful for producing the final image.\n\n"
    + "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    + "Task Types:\n\nA. Text-to-Image Generation:\n"
    + "- Generate a high-quality image based on the user's description.\n"
    + "- Ensure visual clarity, semantic consistency, and completeness.\n"
    + "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    + "B. Image Editing:\n"
    + "- Use the provided image(s) as input or reference for modification or transformation.\n"
    + "- The result can be an edited image or a new image based on the reference(s).\n"
    + "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    + "General Rules:\n"
    + "- For any visible text in the image, follow the language specified for the rendered text in "
    + "the user's description, not the language of the prompt. If no language is specified, use the "
    + "user's input language."
)


def _patchify(
    img: Tensor, p: Int, channel_first: Bool, ctx: DeviceContext
) raises -> Tensor:
    var dims = img.shape()
    var b = dims[0]
    var hh = dims[2]
    var ww = dims[3]
    var gh = hh // p
    var gw = ww // p
    var x6 = _reshape6(img, b, 3, gh, p, gw, p, ctx)
    var perm = List[Int]()
    if channel_first:
        perm.append(0); perm.append(2); perm.append(4)
        perm.append(1); perm.append(3); perm.append(5)
    else:
        perm.append(0); perm.append(2); perm.append(4)
        perm.append(3); perm.append(5); perm.append(1)
    var xp = permute(x6, perm, ctx)
    return _reshape3(xp, b, gh * gw, p * p * 3, ctx)


def _unpatchify(
    x: Tensor, p: Int, h: Int, w: Int, ctx: DeviceContext
) raises -> Tensor:
    var dims = x.shape()
    var b = dims[0]
    var gh = h // p
    var gw = w // p
    var x6 = _reshape6(x, b, gh, gw, p, p, 3, ctx)
    var perm = List[Int]()
    perm.append(0); perm.append(5); perm.append(1)
    perm.append(3); perm.append(2); perm.append(4)
    var xp = permute(x6, perm, ctx)
    return _reshape4(xp, b, 3, gh * p, gw * p, ctx)


def _reshape3(x: Tensor, a: Int, b: Int, c: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(a); sh.append(b); sh.append(c)
    return reshape(x, sh^, ctx)


def _reshape4(x: Tensor, a: Int, b: Int, c: Int, d: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(a); sh.append(b); sh.append(c); sh.append(d)
    return reshape(x, sh^, ctx)


def _reshape6(
    x: Tensor, a: Int, b: Int, c: Int, d: Int, e: Int, f: Int, ctx: DeviceContext
) raises -> Tensor:
    var sh = List[Int]()
    sh.append(a); sh.append(b); sh.append(c)
    sh.append(d); sh.append(e); sh.append(f)
    return reshape(x, sh^, ctx)


def _apply_time_schedule(
    t_uniform: List[Float32], shift: Float32
) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(t_uniform)):
        var t = t_uniform[i]
        var sigma = Float32(1.0) - t
        var shifted = shift * sigma / (Float32(1.0) + (shift - Float32(1.0)) * sigma)
        out.append(Float32(1.0) - shifted)
    return out^


def _t2i_query(system: String, user: String, append: String) -> String:
    var q = String("")
    if system.byte_length() > 0:
        q += "<|im_start|>system\n"
        q += system
        q += "<|im_end|>\n"
    q += "<|im_start|>user\n"
    q += user
    q += "<|im_end|>\n<|im_start|>assistant\n"
    q += append
    return q^


def main() raises:
    var ctx = DeviceContext()
    var cfg = SenseNovaU1Config.sensenova_u1_8b()

    print("[sensenova_u1] loading model from", WEIGHTS_DIR)
    var model = SenseNovaU1[L_TOKENS, TEXT_LEN].load(WEIGHTS_DIR, ctx)

    var tok = Qwen3Tokenizer(
        String(VOCAB_JSON), String(MERGES_TXT), String(ADDED_TOKENS_JSON)
    )
    var cond_ids = tok.encode(
        _t2i_query(
            SYSTEM_MESSAGE_FOR_GEN, String(PROMPT),
            String("<think>\n\n</think>\n\n<img>")
        )
    )
    var uncond_ids = tok.encode(_t2i_query(String(""), String(""), String("<img>")))
    print("[sensenova_u1] cond tokens=", len(cond_ids), " uncond tokens=", len(uncond_ids))

    var cond_cache = model.forward_und(cond_ids, ctx)
    var uncond_cache = model.forward_und(uncond_ids, ctx)
    print("[sensenova_u1] prefix forwards done")

    var noise_scale = model.compute_noise_scale(GRID_H, GRID_W)
    print("[sensenova_u1] noise_scale=", noise_scale)
    var noise_sh = List[Int]()
    noise_sh.append(1); noise_sh.append(3); noise_sh.append(HEIGHT); noise_sh.append(WIDTH)
    var img = randn(noise_sh^, SEED, STDtype.BF16, ctx)
    img = mul_scalar(img, noise_scale, ctx)

    var t_uniform = List[Float32]()
    for i in range(NUM_STEPS + 1):
        t_uniform.append(Float32(i) / Float32(NUM_STEPS))
    var tsched = _apply_time_schedule(t_uniform, TIMESTEP_SHIFT)

    var s_norm = noise_scale / cfg.noise_scale_max

    for step in range(NUM_STEPS):
        var t = tsched[step]
        var t_next = tsched[step + 1]

        var z = _patchify(img, PATCH * MERGE, False, ctx)
        var pixel_values = _patchify(img, PATCH, True, ctx)
        var pixel_flat = _reshape_pixel(pixel_values, ctx)

        var image_embeds = model.extract_feature_gen(pixel_flat, GRID_H, GRID_W, ctx)
        var t_vec = List[Float32]()
        for _ in range(L_TOKENS):
            t_vec.append(t)
        var t_sh = List[Int]()
        t_sh.append(L_TOKENS)
        var t_tensor = Tensor.from_host(t_vec, t_sh^, STDtype.F32, ctx)
        var t_emb = model.time_or_scale_embed(t_tensor, "timestep", ctx)
        var t_emb3 = _reshape3(t_emb, 1, L_TOKENS, cfg.hidden_size, ctx)

        var s_vec = List[Float32]()
        for _ in range(L_TOKENS):
            s_vec.append(s_norm)
        var s_sh = List[Int]()
        s_sh.append(L_TOKENS)
        var s_tensor = Tensor.from_host(s_vec, s_sh^, STDtype.F32, ctx)
        var s_emb = model.time_or_scale_embed(s_tensor, "noise", ctx)
        var s_emb3 = _reshape3(s_emb, 1, L_TOKENS, cfg.hidden_size, ctx)

        var additive = add(t_emb3, s_emb3, ctx)
        image_embeds = add(image_embeds, additive, ctx)

        var h_cond = model.forward_gen(
            image_embeds, cond_cache.next_t_index, TOKEN_H, TOKEN_W, cond_cache, ctx
        )
        var h_uncond = model.forward_gen(
            image_embeds, uncond_cache.next_t_index, TOKEN_H, TOKEN_W, uncond_cache, ctx
        )
        var x_cond = model.fm_head_forward(h_cond, ctx)
        var x_uncond = model.fm_head_forward(h_uncond, ctx)

        var denom = Float32(1.0) - t
        if denom < T_EPS:
            denom = T_EPS
        var inv_denom = Float32(1.0) / denom
        var v_cond = mul_scalar(sub(x_cond, z, ctx), inv_denom, ctx)
        var v_uncond = mul_scalar(sub(x_uncond, z, ctx), inv_denom, ctx)

        var v_diff = sub(v_cond, v_uncond, ctx)
        var v = add(v_uncond, mul_scalar(v_diff, CFG_SCALE, ctx), ctx)

        var z_next = add(z, mul_scalar(v, t_next - t, ctx), ctx)
        img = _unpatchify(z_next, PATCH * MERGE, HEIGHT, WIDTH, ctx)
        print("[sensenova_u1] step", step + 1, "/", NUM_STEPS, " t=", t, "->", t_next)

    var final_img = add_scalar(mul_scalar(img, Float32(0.5), ctx), Float32(0.5), ctx)
    var final_f32 = cast_tensor(final_img, STDtype.F32, ctx)
    save_png(final_f32, OUTPUT, ctx, ValueRange.UNIT)
    print("[sensenova_u1] saved ->", OUTPUT)


def _reshape_pixel(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dims = x.shape()
    var sh = List[Int]()
    sh.append(dims[0] * dims[1])
    sh.append(dims[2])
    return reshape(x, sh^, ctx)
