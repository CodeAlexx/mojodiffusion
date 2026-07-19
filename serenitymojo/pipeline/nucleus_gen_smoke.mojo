# nucleus_gen_smoke.mojo — Nucleus-Image (17B MoE) T2I end-to-end skeleton.
#
# *** CODE-ONLY: COMPILE-VERIFIED, NOT executed. GPU wedged (awaiting reboot).
#     RUN later. EXIT=0 from `mojo build` is the acceptance gate here. ***
#
# Wires the pure-Mojo Nucleus path:
#   tokenizer -> Qwen3-VL text encoder (hidden_states[-8]) -> Nucleus MoE DiT
#   (flow-match Euler + CFG-Zero* true-CFG) -> Qwen-Image 3D causal VAE -> PNG.
#
# References (read line-by-line):
#   inference-flame/src/bin/nucleus_infer.rs  (the end-to-end path)
#   inference-flame/src/models/nucleus_dit.rs (the DiT forward)
#
# Resolution: 256x256 to keep the comptime sdpa joint-seq small for the smoke.
#   VAE_SCALE_FACTOR=8: latent 32x32. patch_size=2: h_grid=w_grid=16.
#   S_IMG = f(1)*16*16 = 256. S_TXT chosen padded; here a small fixed 64.
#   Joint sdpa seq S = S_IMG + S_TXT = 320 (math-mode, head_dim=128).
#
# Scale to 512/1024 by changing the latent size + S_IMG comptime params.
#
# ─── GAPS / STUBS flagged in report ───────────────────────────────────────────
#   * The 17B model does NOT fit resident; this skeleton uses the all-resident
#     NucleusDiT.load (works only for a small/sliced checkpoint). The streaming
#     runtime (Klein9BOffloaded-style) is a documented STUB — not built here.
#   * Text encoder: Nucleus uses the Qwen3-VL text branch (rope_theta=5e6); the
#     mojo Qwen3Encoder.klein_9b() has matching dims but rope_theta=1e6 — a
#     qwen3_vl_text() config + the right extract layer are needed (FLAGGED).
#   * The NucleusAI/Nucleus-Image snapshot is NOT present on this box; paths are
#     placeholders. RUN requires downloading the checkpoint.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import (
    Qwen3Encoder,
    Qwen3Config,
)
from serenitymojo.models.dit.nucleus_dit import NucleusDiT, NucleusConfig
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    add,
    sub,
    mul,
    mul_scalar,
    div,
    reshape,
    permute,
)
from serenitymojo.image.png import save_png, ValueRange


# ── comptime sequence shapes (256x256 smoke) ─────────────────────────────────
comptime LH = 32            # latent height (256 / 8)
comptime LW = 32            # latent width
comptime PATCH = 2
comptime H_GRID = LH // PATCH   # 16
comptime W_GRID = LW // PATCH   # 16
comptime FRAMES = 1
comptime S_IMG = FRAMES * H_GRID * W_GRID   # 256
comptime S_TXT = 64                          # padded text tokens (small smoke)
comptime IN_CH = 64          # patch²*z_dim = 4*16 = 64 (Nucleus img_in in-dim)
comptime Z_DIM = 16
comptime JOINT = 4096        # Qwen3-VL text dim

# ── paths (placeholders — Nucleus snapshot not on this box) ──────────────────
comptime SNAPSHOT = "/home/alex/.cache/huggingface/hub/models--NucleusAI--Nucleus-Image/snapshots/CHECKPOINT"
comptime TEXT_ENCODER_DIR = SNAPSHOT + "/text_encoder"
comptime TOK_JSON = SNAPSHOT + "/processor/tokenizer.json"
comptime DIT_DIR = SNAPSHOT + "/transformer"
comptime VAE_DIR = SNAPSHOT + "/vae"
comptime OUT = "/home/alex/mojodiffusion/output/nucleus_first_256.png"

comptime SYSTEM_PROMPT = "You are an image generation assistant. Follow the user's prompt literally."
comptime PROMPT = "An orange tabby cat on a wooden table, photorealistic"


# ── Nucleus VAE latent denormalize constants (nucleus_infer.rs) ──────────────
def _vae_latents_mean() -> List[Float32]:
    var v = List[Float32]()
    var data = [
        Float32(-0.7571), Float32(-0.7089), Float32(-0.9113), Float32(0.1075),
        Float32(-0.1745), Float32(0.9653), Float32(-0.1517), Float32(1.5508),
        Float32(0.4134), Float32(-0.0715), Float32(0.5517), Float32(-0.3632),
        Float32(-0.1922), Float32(-0.9497), Float32(0.2503), Float32(-0.2921),
    ]
    for x in data:
        v.append(x)
    return v^


def _vae_latents_std() -> List[Float32]:
    var v = List[Float32]()
    var data = [
        Float32(2.8184), Float32(1.4541), Float32(2.3275), Float32(2.6558),
        Float32(1.2196), Float32(1.7708), Float32(2.6052), Float32(2.0743),
        Float32(3.2687), Float32(2.1526), Float32(2.8652), Float32(1.5579),
        Float32(1.6382), Float32(1.1253), Float32(2.8251), Float32(1.916),
    ]
    for x in data:
        v.append(x)
    return v^


# ── chat template (matches nucleus_infer.rs format_prompt_chat) ──────────────
def format_prompt_chat(prompt: String) -> String:
    return (
        String("<|im_start|>system\n")
        + SYSTEM_PROMPT
        + "<|im_end|>\n<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


# ── latent pack/unpack (nucleus_infer.rs _pack_latents / _unpack_latents) ─────
# pack: [1,Z,H,W] -> [1, S_img, Z*P²]; unpack: inverse.
def pack_latents(latent: Tensor, ctx: DeviceContext) raises -> Tensor:
    # [1, Z, LH, LW] -> view [1, Z, H_GRID, P, W_GRID, P]
    #   -> permute (0,2,4,1,3,5) -> reshape [1, H_GRID*W_GRID, Z*P*P].
    var v = reshape(latent, _sh6(1, Z_DIM, H_GRID, PATCH, W_GRID, PATCH), ctx)
    var perm = List[Int]()
    perm.append(0); perm.append(2); perm.append(4)
    perm.append(1); perm.append(3); perm.append(5)
    var vp = permute(v, perm, ctx)
    return reshape(vp, _sh3(1, S_IMG, Z_DIM * PATCH * PATCH), ctx)


def unpack_latents(latent: Tensor, ctx: DeviceContext) raises -> Tensor:
    # [1, S_img, Z*P²] -> [1, H_GRID, W_GRID, Z, P, P]
    #   -> permute (0,3,1,4,2,5) -> reshape [1, Z, LH, LW].
    var v = reshape(latent, _sh6(1, H_GRID, W_GRID, Z_DIM, PATCH, PATCH), ctx)
    var perm = List[Int]()
    perm.append(0); perm.append(3); perm.append(1)
    perm.append(4); perm.append(2); perm.append(5)
    var vp = permute(v, perm, ctx)
    return reshape(vp, _sh4(1, Z_DIM, LH, LW), ctx)


# ── L2 norm along last dim, keepdim ([..,D]->[..,1]) ─────────────────────────
def norm_last(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var xf = cast_tensor(x, STDtype.F32, ctx)
    var host = xf.to_host(ctx)
    var sh = x.shape()
    var d = sh[len(sh) - 1]
    var rows = 1
    for i in range(len(sh) - 1):
        rows *= sh[i]
    var out = List[Float32]()
    out.resize(rows, Float32(0.0))
    for r in range(rows):
        var acc = Float64(0.0)
        for c in range(d):
            var v = Float64(host[r * d + c])
            acc += v * v
        out[r] = Float32(sqrt(acc))
    # shape [.., 1]
    var osh = List[Int]()
    for i in range(len(sh) - 1):
        osh.append(sh[i])
    osh.append(1)
    return Tensor.from_host(out, osh^, x.dtype(), ctx)


# ── Nucleus sigma schedule: linspace(1.0, 1/N, N) appended with 0 ────────────
def nucleus_sigmas(n: Int) raises -> List[Float32]:
    var sig = List[Float32]()
    var denom = n - 1
    if denom < 1:
        denom = 1
    for i in range(n):
        var t = Float32(i) / Float32(denom)
        var s = Float32(1.0) - t * (Float32(1.0) - Float32(1.0) / Float32(n))
        sig.append(s)
    sig.append(Float32(0.0))
    return sig^


def main() raises:
    var ctx = DeviceContext()
    var steps = 30
    var guidance = Float32(4.0)
    var do_cfg = guidance > Float32(1.0)

    # ── Phase 1: tokenize + encode (cond + uncond) ───────────────────────────
    # FLAG: Qwen3-VL text encoder; rope_theta + extract-layer parity pending.
    var tok = Qwen3Tokenizer(TOK_JSON)
    var ids_cond = tok.encode(format_prompt_chat(PROMPT))
    var ids_uncond = tok.encode(format_prompt_chat(String("")))

    # Qwen3-VL config: 4096 hidden / 36 layers (klein_9b dims). rope_theta is
    # 1e6 here vs Nucleus's 5e6 — FLAGGED; needs a qwen3_vl_text() config.
    var enc_cfg = Qwen3Config.klein_9b()
    var encoder = Qwen3Encoder.load(TEXT_ENCODER_DIR, enc_cfg, ctx)
    # diffusers default_return_index = -8 -> layer 28 of 36 (0-indexed).
    var prompt_embeds = encoder.encode(ids_cond, 28, ctx)   # [1, S, 4096]
    var neg_embeds = encoder.encode(ids_uncond, 28, ctx)

    # ── Phase 2: load DiT + VAE ──────────────────────────────────────────────
    # FLAG: all-resident load — only viable for a sliced/small checkpoint. The
    # full 17B needs the streaming runtime (STUB in nucleus_dit.mojo).
    var dit = NucleusDiT[S_IMG, S_TXT].load(DIT_DIR, ctx)
    var vae = QwenImageVaeDecoder[LH, LW].load(VAE_DIR, ctx)

    # ── Phase 3: initial latent + sigma schedule ─────────────────────────────
    var latent_unpacked = randn(_sh4(1, Z_DIM, LH, LW), 42, STDtype.F32, ctx)
    var latents_f = pack_latents(latent_unpacked, ctx)        # [1, S_IMG, 64] F32
    var latents = cast_tensor(latents_f, STDtype.BF16, ctx)   # model dtype

    var sigmas = nucleus_sigmas(steps)

    # ── Phase 4: denoise loop ────────────────────────────────────────────────
    for i in range(steps):
        var sigma = sigmas[i]
        var sigma_next = sigmas[i + 1]

        var v_cond = dit.forward(
            latents, prompt_embeds, sigma, FRAMES, H_GRID, W_GRID, ctx
        )

        var velocity: Tensor
        if do_cfg:
            var v_uncond = dit.forward(
                latents, neg_embeds, sigma, FRAMES, H_GRID, W_GRID, ctx
            )
            # comb = v_uncond + guidance*(v_cond - v_uncond)   (textbook CFG)
            var diff = sub(v_cond, v_uncond, ctx)
            var comb = add(v_uncond, mul_scalar(diff, guidance, ctx), ctx)
            # CFG-Zero* per-token norm rescale: comb *= ||v_cond|| / ||comb||.
            var cond_n = norm_last(v_cond, ctx)
            var comb_n = norm_last(comb, ctx)
            var ratio = div(cond_n, comb_n, ctx)   # [.., 1] broadcast
            comb = mul(comb, ratio, ctx)
            # noise_pred = -comb.
            velocity = mul_scalar(comb, Float32(-1.0), ctx)
        else:
            velocity = mul_scalar(v_cond, Float32(-1.0), ctx)

        # latents = latents + (sigma_next - sigma) * velocity.
        var dt = sigma_next - sigma
        var dv = mul_scalar(velocity, dt, ctx)
        latents = add(latents, dv, ctx)

    # ── Phase 5: unpack + VAE decode + save PNG ──────────────────────────────
    var unpacked = unpack_latents(latents, ctx)   # [1, Z, LH, LW] BF16
    # VAE expects F32 NCHW [1,16,LH,LW]; it denormalizes internally.
    var unpacked_f = cast_tensor(unpacked, STDtype.F32, ctx)
    var image = vae.decode(unpacked_f, ctx)        # [1,3,8LH,8LW] NCHW [-1,1]
    save_png(image, OUT, ctx, ValueRange.SIGNED)
    print("nucleus smoke wired; output:", OUT)


# ── shape helpers ─────────────────────────────────────────────────────────────
def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c)
    return s^


def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
    return s^


def _sh6(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c)
    s.append(d); s.append(e); s.append(f)
    return s^
