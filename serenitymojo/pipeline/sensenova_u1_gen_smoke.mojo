# pipeline/sensenova_u1_gen_smoke.mojo — SenseNova-U1-8B-MoT T2I glue smoke.
#
# COMPILE-ONLY (GPU wedged). Wires the full T2I path end to end:
#   tokenize* -> forward_und (cond + uncond prefixes -> two KV caches)
#     -> denoise loop { patchify -> extract_feature_gen -> +timestep/noise embed
#        -> forward_gen(cond) + forward_gen(uncond) -> fm_head -> velocity -> CFG
#        -> Euler step -> unpatchify } -> denorm -> save_png.
#
# Reference: inference-flame/src/bin/sensenova_u1_gen.rs::run_t2i (read line by
# line). This file is the Mojo glue; it does NOT execute (no GPU).
#
# *Tokenizer: the SenseNova-U1 dir ships vocab.json + merges.txt +
#  added_tokens.json (NO unified tokenizer.json). The Mojo Qwen3Tokenizer takes
#  a single tokenizer.json, so it CANNOT consume the SenseNova format directly.
#  FLAGGED in the report. This smoke uses a placeholder token list to exercise
#  the model wiring; a real run needs a vocab+merges tokenizer loader.
#
# Small resolution (64x64) chosen so the comptime sequence lengths stay small:
#   patch=16, merge=2 -> grid 4x4 (16 pixel patches), token 2x2 -> L_TOKENS=4.
#   TEXT_LEN is the placeholder prefix length.
#
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
    SenseNovaU1, SenseNovaU1Config, KvCache,
)


comptime WEIGHTS_DIR = "/home/alex/.serenity/models/sensenova_u1"
comptime OUTPUT = "/home/alex/mojodiffusion/output/sensenova_u1_smoke_64.png"

# Small smoke geometry (compile-only). 64x64 image.
comptime WIDTH = 64
comptime HEIGHT = 64
comptime PATCH = 16
comptime MERGE = 2
comptime GRID_H = HEIGHT // PATCH        # 4
comptime GRID_W = WIDTH // PATCH         # 4
comptime TOKEN_H = GRID_H // MERGE       # 2
comptime TOKEN_W = GRID_W // MERGE       # 2
comptime L_TOKENS = TOKEN_H * TOKEN_W    # 4
comptime TEXT_LEN = 8                    # placeholder prefix length
comptime FM_OUT = (PATCH * MERGE) * (PATCH * MERGE) * 3  # 3072

comptime NUM_STEPS = 2
comptime CFG_SCALE = Float32(4.0)
comptime TIMESTEP_SHIFT = Float32(3.0)
comptime SEED = UInt64(42)
comptime T_EPS = Float32(0.05)


# ── patchify / unpatchify (mirror sensenova_u1_gen.rs:300-347) ───────────────
# patchify([B,3,H,W], p, channel_first) -> [B, gh*gw, p*p*3].
#   channel_first=True  : within-patch order (C, kH, kW)  -> matches conv weight
#   channel_first=False : within-patch order (kH, kW, C)  -> z / fm_head 3072
def _patchify(
    img: Tensor, p: Int, channel_first: Bool, ctx: DeviceContext
) raises -> Tensor:
    var dims = img.shape()
    var b = dims[0]
    var hh = dims[2]
    var ww = dims[3]
    var gh = hh // p
    var gw = ww // p
    # reshape [B,3,gh,p,gw,p]
    var x6 = _reshape6(img, b, 3, gh, p, gw, p, ctx)
    # source axes 0=B 1=C 2=gh 3=p 4=gw 5=p
    var perm = List[Int]()
    if channel_first:
        # target B gh gw C p p  -> axes [0,2,4,1,3,5]
        perm.append(0); perm.append(2); perm.append(4)
        perm.append(1); perm.append(3); perm.append(5)
    else:
        # target B gh gw p p C  -> axes [0,2,4,3,5,1]
        perm.append(0); perm.append(2); perm.append(4)
        perm.append(3); perm.append(5); perm.append(1)
    var xp = permute(x6, perm, ctx)
    return _reshape3(xp, b, gh * gw, p * p * 3, ctx)


# unpatchify([B,L,p*p*3], p, h, w) -> [B,3,h,w]; inner order (kH,kW,C).
def _unpatchify(
    x: Tensor, p: Int, h: Int, w: Int, ctx: DeviceContext
) raises -> Tensor:
    var dims = x.shape()
    var b = dims[0]
    var gh = h // p
    var gw = w // p
    # reshape [B,gh,gw,p,p,3]
    var x6 = _reshape6(x, b, gh, gw, p, p, 3, ctx)
    # target B C gh p gw p -> axes [0,5,1,3,2,4]
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


# Apply the standard exponential time-shift schedule to a uniform grid
# (sensenova_u1.rs:1463). timestep_shift != 1 -> "standard" branch.
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


def _placeholder_tokens(n: Int) -> List[Int]:
    # Placeholder token ids (real run needs a vocab+merges tokenizer; flagged).
    var ids = List[Int]()
    for i in range(n):
        ids.append(100 + i)
    return ids^


def main() raises:
    var ctx = DeviceContext()
    var cfg = SenseNovaU1Config.sensenova_u1_8b()

    print("[sensenova_u1] loading model from", WEIGHTS_DIR)
    var model = SenseNovaU1[L_TOKENS, TEXT_LEN].load(WEIGHTS_DIR, ctx)

    # ---- prefix forwards (cond + uncond). Placeholder tokens (see header). ----
    var cond_ids = _placeholder_tokens(TEXT_LEN)
    var uncond_ids = _placeholder_tokens(TEXT_LEN)
    var cond_cache = model.forward_und(cond_ids, ctx)
    var uncond_cache = model.forward_und(uncond_ids, ctx)
    print("[sensenova_u1] prefix forwards done")

    # ---- init noise image [1,3,H,W] BF16, scaled by resolution noise_scale ----
    var noise_scale = model.compute_noise_scale(GRID_H, GRID_W)
    var noise_sh = List[Int]()
    noise_sh.append(1); noise_sh.append(3); noise_sh.append(HEIGHT); noise_sh.append(WIDTH)
    var img = randn(noise_sh^, SEED, STDtype.BF16, ctx)
    img = mul_scalar(img, noise_scale, ctx)

    # ---- timestep grid + schedule ----
    var t_uniform = List[Float32]()
    for i in range(NUM_STEPS + 1):
        t_uniform.append(Float32(i) / Float32(NUM_STEPS))
    var tsched = _apply_time_schedule(t_uniform, TIMESTEP_SHIFT)

    var s_norm = noise_scale / cfg.noise_scale_max

    # ---- step loop ----
    for step in range(NUM_STEPS):
        var t = tsched[step]
        var t_next = tsched[step + 1]

        # z (target, channel-last) and pixel_values (gen embedder input, C-major)
        var z = _patchify(img, PATCH * MERGE, False, ctx)       # [1,L,3072]
        var pixel_values = _patchify(img, PATCH, True, ctx)     # [1,gh*gw,768]
        var pixel_flat = _reshape_pixel(pixel_values, ctx)      # [gh*gw,768]

        # gen embedder + timestep / noise embed
        var image_embeds = model.extract_feature_gen(pixel_flat, GRID_H, GRID_W, ctx)
        var t_vec = List[Float32]()
        for _ in range(L_TOKENS):
            t_vec.append(t)
        var t_sh = List[Int]()
        t_sh.append(L_TOKENS)
        var t_tensor = Tensor.from_host(t_vec, t_sh^, STDtype.F32, ctx)
        var t_emb = model.time_or_scale_embed(t_tensor, "timestep", ctx)  # [L,4096]
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

        # CFG cond + uncond forward_gen
        var h_cond = model.forward_gen(
            image_embeds, cond_cache.next_t_index, TOKEN_H, TOKEN_W, cond_cache, ctx
        )
        var h_uncond = model.forward_gen(
            image_embeds, uncond_cache.next_t_index, TOKEN_H, TOKEN_W, uncond_cache, ctx
        )
        var x_cond = model.fm_head_forward(h_cond, ctx)    # [1,L,3072]
        var x_uncond = model.fm_head_forward(h_uncond, ctx)

        # velocity v = (x_pred - z) / max(1-t, t_eps)
        var denom = Float32(1.0) - t
        if denom < T_EPS:
            denom = T_EPS
        var inv_denom = Float32(1.0) / denom
        var v_cond = mul_scalar(sub(x_cond, z, ctx), inv_denom, ctx)
        var v_uncond = mul_scalar(sub(x_uncond, z, ctx), inv_denom, ctx)

        # CFG (cfg_norm='none'): v = v_uncond + scale*(v_cond - v_uncond)
        var v_diff = sub(v_cond, v_uncond, ctx)
        var v = add(v_uncond, mul_scalar(v_diff, CFG_SCALE, ctx), ctx)

        # Euler step on z, then unpatchify back to image space at p*merge.
        var z_next = add(z, mul_scalar(v, t_next - t, ctx), ctx)
        img = _unpatchify(z_next, PATCH * MERGE, HEIGHT, WIDTH, ctx)
        print("[sensenova_u1] step", step + 1, "/", NUM_STEPS, " t=", t, "->", t_next)

    # ---- denorm ((img*0.5+0.5).clamp(0,1)) and save (UNIT range) ----
    var final_img = add_scalar(mul_scalar(img, Float32(0.5), ctx), Float32(0.5), ctx)
    var final_f32 = cast_tensor(final_img, STDtype.F32, ctx)
    save_png(final_f32, OUTPUT, ctx, ValueRange.UNIT)
    print("[sensenova_u1] saved ->", OUTPUT)


# pixel_values is [1, gh*gw, 768]; flatten to [gh*gw, 768] for extract_feature_gen.
def _reshape_pixel(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dims = x.shape()
    var sh = List[Int]()
    sh.append(dims[0] * dims[1])
    sh.append(dims[2])
    return reshape(x, sh^, ctx)
