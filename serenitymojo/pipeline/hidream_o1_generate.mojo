# pipeline/hidream_o1_generate.mojo — HiDream-O1-Image T2I real multi-step run.
#
# Modeled on pipeline/hidream_o1_smoke.mojo (the verified 64x64/1-step skeleton)
# but scaled to a real resolution (256x256) and a real multi-step denoise (20
# steps, deterministic Default flow-match Euler). The DiT math is verified clean
# (skeptic doc SKEPTIC_FINDINGS_hidream_2026-05-26.md); this file only wires the
# pipeline glue and CALLS dit.forward — it does NOT modify the DiT.
#
# Reference, read line-by-line:
#   /home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/pipeline.rs
#     build_t2i_input (235-365), generate (386-585), gather_image_rows (657-700),
#     compute_velocity (702-708)
#
# Flow (T2I, batch 1, no ref-image, guidance=1):
#   1) chat template + boi + tms; tokenize -> text_ids (cond len 20 for the test
#      prompt; verified vs HF oracle 3/3 in hidream_tok_check.mojo)
#   2) build full stream [text_ids, vision_start, (L-1)*image_pad]; MRoPE
#      positions over full stream; ar_len = s_text - 1
#   3) z0 = noise_scale_start * randn([1,3,H,W]); patchify p=32 -> [1,L,3072]
#   4) for step in 20: forward (cond); gather image rows (tail L);
#      v = (x_pred - z)/sigma_clamped; model_output = -v; scheduler.step
#   5) unpatchify -> [1,3,H,W] in [-1,1]; save_png (SIGNED)
#
# NO VAE (RGB pixel space), NO separate text encoder (the spine IS the encoder).
#
# CFG: guidance=1 (single forward). cond/uncond have different static seq lengths
# (84 vs 75 at 256x256) which collide with the module-level comptime S; a coherent
# guidance=1 image already demonstrates O1 works (per the task spec). Full CFG
# would need common-S padding/dispatch — flagged, not done this pass.
#
# Module-level comptime S (the 1.0.0b1 instantiator constraint documented in the
# smoke: S MUST be module scope, and the DiT denoise loop must live in its own
# top-level def so it never co-monomorphizes with unpatchify in main).
#
# Mojo 1.0.0b1, NVIDIA GPU. Weights F32-on-disk -> BF16 on load.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.ops.layout import patchify, unpatchify
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import sub, mul_scalar, add, slice as ts_slice
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.dit.hidream_o1 import (
    HiDreamO1Config,
    HiDreamO1Offloaded,
    build_mrope_positions,
)
from serenitymojo.sampling.hidream_o1_scheduler import HiDreamO1Scheduler


# Chat template (pipeline.rs:193-211): user message + assistant prompt + boi +
# one tms token. TIMESTEP_TOKEN_NUM == 1.
def apply_chat_template_t2i(prompt: String) -> String:
    var s = String("<|im_start|>user\n")
    s += prompt
    s += "<|im_end|>\n<|im_start|>assistant\n"
    s += "<|boi_token|><|tms_token|>"
    return s


@fieldwise_init
struct T2ISample(Movable):
    var text_ids: List[Int]
    var t_pos: List[Int]
    var h_pos: List[Int]
    var w_pos: List[Int]
    var s_text: Int
    var ar_len: Int


def build_t2i_input(
    tok: Qwen3Tokenizer,
    cfg: HiDreamO1Config,
    prompt: String,
    h_patches: Int,
    w_patches: Int,
) raises -> T2ISample:
    var template = apply_chat_template_t2i(prompt)
    var text_ids = tok.encode(template)
    var s_text = len(text_ids)
    var image_len = h_patches * w_patches

    # Full stream: text + vision_start + (image_len-1) * image_pad.
    var full_ids = List[Int]()
    for i in range(s_text):
        full_ids.append(text_ids[i])
    full_ids.append(cfg.vision_start_token_id)
    for _ in range(1, image_len):
        full_ids.append(cfg.image_token_id)

    var pos = build_mrope_positions(
        full_ids,
        cfg.image_token_id,
        cfg.vision_start_token_id,
        h_patches,
        w_patches,
        cfg.fix_point,
    )
    var ar_len = s_text - 1
    if ar_len < 0:
        ar_len = 0
    return T2ISample(text_ids^, pos[0].copy(), pos[1].copy(), pos[2].copy(), s_text, ar_len)


# v = (x_pred - z) / sigma. F32 scalar arithmetic happens inside tensor ops;
# the velocity carrier returns in the latent/checkpoint storage dtype.
def compute_velocity(x_pred: Tensor, z: Tensor, sigma: Float32, ctx: DeviceContext) raises -> Tensor:
    if x_pred.dtype() == z.dtype():
        var diff = sub(x_pred, z, ctx)
        return mul_scalar(diff, Float32(1.0) / sigma, ctx)
    var pred = cast_tensor(x_pred, z.dtype(), ctx)
    var diff = sub(pred, z, ctx)
    return mul_scalar(diff, Float32(1.0) / sigma, ctx)


# Gather the L image rows from x_pred [1, S, 3072] — the contiguous tail
# (pipeline.rs:657-700: narrow(dim=1, start=s_text, length=L)).
def gather_image_rows(
    x_pred: Tensor, s_text: Int, image_len: Int, ctx: DeviceContext
) raises -> Tensor:
    return ts_slice(x_pred, 1, s_text, image_len, ctx)


# ── Module-level comptime S (1.0.0b1 instantiator constraint) ────────────────
# 256x256, patch 32 -> 8x8 = 64 image patches. The test prompt
# "a photograph of a red apple on a wooden table" tokenizes to 20 cond tokens
# (HF-oracle verified). S_TOTAL = 20 text + 64 image = 84.
comptime GEN_H = 256
comptime GEN_W = 256
comptime GEN_PATCH = 32
comptime GEN_HP = GEN_H // GEN_PATCH                 # 8
comptime GEN_WP = GEN_W // GEN_PATCH                 # 8
comptime GEN_IMAGE_LEN = GEN_HP * GEN_WP             # 64
comptime GEN_S_TOTAL = 84                            # 20 text + 64 image rows
comptime GEN_STEPS = 20                              # real multi-step denoise


# Full denoise loop (DiT forwards + scheduler steps), isolated in its own
# top-level def at the module-level comptime S so it never co-monomorphizes with
# unpatchify in main. Returns the final patched latent z [1, L, 3072].
# timestep fed to the DiT MUST be t_pixeldit = 1 - step_t/1000, NOT sigma — the
# t_embedder rescales by *1000 internally (pipeline.rs:487, insight #4).
def denoise(
    var z: Tensor,
    cond: T2ISample,
    seed: UInt64,
    ctx: DeviceContext,
) raises -> Tensor:
    var cfg = HiDreamO1Config.dev_8b()
    var image_len = GEN_IMAGE_LEN

    print(
        "[hidream_o1] cond text_len=", cond.s_text,
        " image_len=", image_len,
        " total=", cond.s_text + image_len,
        " fixed=", GEN_S_TOTAL,
    )
    if cond.s_text + image_len != GEN_S_TOTAL:
        raise Error("hidream_o1 generate: prompt seq != fixed S_TOTAL")

    # Load the DiT once at the module-level S (never inside the step loop).
    var dit = HiDreamO1Offloaded[GEN_S_TOTAL].load(
        String("/home/alex/HiDream-O1-Image-Dev-weights"), cfg, ctx
    )
    # Default (deterministic) flow-match Euler, shift 3.0 (Full mode). No CPU
    # host transfer per step (prev = sample + (sigma_next - sigma)*model_output).
    var sched = HiDreamO1Scheduler.full_n_step(GEN_STEPS, Float32(3.0))
    var num_steps = sched.num_inference_steps()

    var noise_scale_start = Float32(7.5)

    for step_idx in range(num_steps):
        var step_t = sched.timestep(step_idx)
        var t_pixeldit = Float32(1.0) - step_t / Float32(1000.0)
        var sigma_clamped = step_t / Float32(1000.0)
        if sigma_clamped < Float32(0.001):
            sigma_clamped = Float32(0.001)

        print("[hidream_o1] step", step_idx + 1, "/", num_steps, " t=", step_t)

        # cond forward -> image-row x_pred -> v_cond = (x_pred - z)/sigma.
        var x_pred_full = dit.forward(
            cond.text_ids, z, cond.t_pos.copy(), cond.h_pos.copy(),
            cond.w_pos.copy(), cond.ar_len, t_pixeldit, ctx,
        )
        var x_pred_cond = gather_image_rows(
            x_pred_full, cond.s_text, image_len, ctx
        )
        var v_cond = compute_velocity(x_pred_cond, z, sigma_clamped, ctx)

        # guidance=1: model_output = -v_cond (sign flip, pipeline.rs:546).
        var model_output = mul_scalar(v_cond, Float32(-1.0), ctx)

        # Deterministic Default scheduler ignores the noise arg; pass zeros.
        var noise_sh = model_output.shape()
        var zeros = randn(noise_sh^, UInt64(0), model_output.dtype(), ctx)
        zeros = mul_scalar(zeros, Float32(0.0), ctx)
        z = sched.step(
            model_output, step_idx, z, zeros, noise_scale_start,
            Float32(0.0), ctx,
        )
    return z^


def hidream_o1_t2i_generate(
    model_dir: String,
    tokenizer_json: String,
    prompt: String,
    out_png: String,
) raises:
    var cfg = HiDreamO1Config.dev_8b()
    if cfg.head_dim != 128:
        raise Error("hidream_o1: unexpected head_dim")

    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(tokenizer_json)
    var p = GEN_PATCH

    var cond = build_t2i_input(tok, cfg, prompt, GEN_HP, GEN_WP)
    var tms_seen = False
    for i in range(len(cond.text_ids)):
        if cond.text_ids[i] == cfg.tms_token_id:
            tms_seen = True
    print("[hidream_o1] tms token seen=", tms_seen)

    var seed = UInt64(0)
    var noise_scale_start = Float32(7.5)
    var z_sh = List[Int]()
    z_sh.append(1); z_sh.append(3); z_sh.append(GEN_H); z_sh.append(GEN_W)
    var z0 = randn(z_sh^, seed, STDtype.BF16, ctx)
    z0 = mul_scalar(z0, noise_scale_start, ctx)
    var z = patchify(z0, p, ctx)  # [1, L, 3072]

    z = denoise(z^, cond, seed, ctx)

    var img = unpatchify(z, 3, GEN_H, GEN_W, p, ctx)
    save_png(img, out_png, ctx, ValueRange.SIGNED)
    print("hidream_o1 generate saved ->", out_png)


def main() raises:
    hidream_o1_t2i_generate(
        String("/home/alex/HiDream-O1-Image-Dev-weights"),
        String("/home/alex/HiDream-O1-Image-Dev-weights/tokenizer.json"),
        String("a photograph of a red apple on a wooden table"),
        String("/home/alex/mojodiffusion/output/hidream_o1_256_20step.png"),
    )
