# pipeline/hidream_o1_cfg.mojo — HiDream-O1-Image T2I with REAL CFG + higher res.
#
# Extends pipeline/hidream_o1_generate.mojo (the working guidance=1 path that
# produced output/hidream_o1_256_20step.png). Adds:
#   1) Classifier-free guidance (CFG) with a COMMON static sequence length S.
#   2) A higher resolution (512x512, patch 32 -> 16x16 = 256 image patches).
#
# Reference, read line-by-line:
#   /home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/pipeline.rs
#     generate (386-585): v_guided = v_uncond + scale*(v_cond - v_uncond)
#                         model_output = -v_guided (pipeline.rs:537-546)
#                         uncond prompt = single space " " (pipeline.rs:425)
#
# ── The CFG common-S problem + fix ──────────────────────────────────────────
# The DiT is comptime-parameterized on S = S_text + image_len. The cond prompt
# tokenizes to 20 text tokens; the uncond prompt " " to 11. image_len is fixed
# by the resolution. So cond/uncond would need DIFFERENT module-level S, which
# the 1.0.0b1 instantiator forbids in one binary.
#
# FIX (this file): build ONE common static S from the COND prompt's S_text (20).
# Pad the uncond text_ids up to that S_text with <|vision_pad|> tokens, and run
# the uncond forward through HiDreamO1Offloaded.forward_padded, which builds a
# pad-aware attention mask (_build_prefix_causal_mask_padded in
# models/dit/hidream_o1.mojo) that blocks the padding KEY columns everywhere.
# The DiT layer math is byte-identical to the cond path; only the mask differs.
# So both cond and uncond run at the SAME comptime S with no DiT-math change.
#
# Keeps guidance<=1 working (single cond forward, model_output = -v_cond) so the
# original 256 image path is preserved by passing guidance=1.
#
# Module-level comptime S (1.0.0b1 instantiator constraint, per the generate
# file): S MUST be module scope and the denoise loop lives in its own top-level
# def so it never co-monomorphizes with unpatchify in main.
#
# SCOPE: pure Mojo+MAX, inference-only, GPU-only. Reuses the verified DiT +
# scheduler unchanged (the only DiT addition is the pad-aware MASK builder +
# forward_padded wrapper — no math touched). Mojo 1.0.0b1, NVIDIA GPU.

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


# A T2I sample: the text portion (fed to embed), MRoPE positions over the FULL
# stream, s_text, ar_len, and a key_valid mask over the FULL stream (1=real,
# 0=padding). For the cond path every slot is valid (key_valid all 1).
@fieldwise_init
struct T2ISample(Movable):
    var text_ids: List[Int]
    var t_pos: List[Int]
    var h_pos: List[Int]
    var w_pos: List[Int]
    var s_text: Int
    var ar_len: Int
    var key_valid: List[Int]


# Build a T2I sample. `target_s_text` >= the raw tokenized text length: if the
# raw text is shorter, it is padded up to `target_s_text` with <|vision_pad|>
# tokens inserted just before the boi/tms tail (so the tms slot stays last and
# is still found by the DiT). key_valid marks padding slots 0 over the FULL
# stream. When target_s_text == raw length, no padding is added (cond path).
def build_t2i_input(
    tok: Qwen3Tokenizer,
    cfg: HiDreamO1Config,
    prompt: String,
    h_patches: Int,
    w_patches: Int,
    target_s_text: Int,
) raises -> T2ISample:
    var template = apply_chat_template_t2i(prompt)
    var raw_ids = tok.encode(template)
    var raw_len = len(raw_ids)
    if raw_len > target_s_text:
        raise Error("hidream_o1 cfg: prompt longer than target_s_text")

    var image_len = h_patches * w_patches

    # Pad raw_ids up to target_s_text by inserting <|vision_pad|> just before
    # the final two structural tokens (boi at raw_len-2, tms at raw_len-1).
    var pad_id = 151654  # <|vision_pad|> — a real embeddable token, masked out.
    var num_pad = target_s_text - raw_len
    var text_ids = List[Int]()
    var pad_text_slots = List[Int]()  # indices (in padded text) that are pads
    if num_pad == 0:
        for i in range(raw_len):
            text_ids.append(raw_ids[i])
    else:
        var tail_start = raw_len - 2  # boi index in raw
        for i in range(tail_start):
            text_ids.append(raw_ids[i])
        for _ in range(num_pad):
            pad_text_slots.append(len(text_ids))
            text_ids.append(pad_id)
        # boi + tms tail.
        text_ids.append(raw_ids[raw_len - 2])
        text_ids.append(raw_ids[raw_len - 1])
    var s_text = len(text_ids)

    # Full stream: padded text + vision_start + (image_len-1) * image_pad.
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

    # key_valid over the FULL stream (s_text + image_len): all 1 except the
    # padding text slots, which are 0 (blocked as attention keys everywhere).
    var s_total = s_text + image_len
    var key_valid = List[Int]()
    for _ in range(s_total):
        key_valid.append(1)
    for i in range(len(pad_text_slots)):
        key_valid[pad_text_slots[i]] = 0

    return T2ISample(
        text_ids^, pos[0].copy(), pos[1].copy(), pos[2].copy(),
        s_text, ar_len, key_valid^,
    )


# v = (x_pred - z) / sigma  (F32).  pipeline.rs:702-708.
def compute_velocity(x_pred: Tensor, z: Tensor, sigma: Float32, ctx: DeviceContext) raises -> Tensor:
    var xf = cast_tensor(x_pred, STDtype.F32, ctx)
    var zf = cast_tensor(z, STDtype.F32, ctx)
    var diff = sub(xf, zf, ctx)
    return mul_scalar(diff, Float32(1.0) / sigma, ctx)


# Gather the L image rows from x_pred [1, S, 3072] (contiguous tail).
def gather_image_rows(
    x_pred: Tensor, s_text: Int, image_len: Int, ctx: DeviceContext
) raises -> Tensor:
    return ts_slice(x_pred, 1, s_text, image_len, ctx)


# ── Module-level comptime S (1.0.0b1 instantiator constraint) ────────────────
# 512x512, patch 32 -> 16x16 = 256 image patches. The cond prompt
# "a photograph of a red apple on a wooden table" tokenizes to 20 text tokens
# (HF-oracle verified); uncond " " tokenizes to 11 and is padded to 20.
# S_TOTAL = 20 text + 256 image = 276 (common to cond + uncond).
comptime GEN_H = 512
comptime GEN_W = 512
comptime GEN_PATCH = 32
comptime GEN_HP = GEN_H // GEN_PATCH                 # 16
comptime GEN_WP = GEN_W // GEN_PATCH                 # 16
comptime GEN_IMAGE_LEN = GEN_HP * GEN_WP             # 256
comptime GEN_S_TEXT = 20                             # cond text len (common)
comptime GEN_S_TOTAL = GEN_S_TEXT + GEN_IMAGE_LEN    # 276
comptime GEN_STEPS = 20                              # real multi-step denoise


# Full denoise loop (DiT forwards + scheduler steps), isolated at the
# module-level comptime S so it never co-monomorphizes with unpatchify in main.
# Runs cond + (if guidance>1) padded-uncond per step, combines via CFG.
def denoise(
    var z: Tensor,
    cond: T2ISample,
    uncond: T2ISample,
    guidance: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var cfg = HiDreamO1Config.dev_8b()
    var image_len = GEN_IMAGE_LEN
    var do_cfg = guidance > Float32(1.0)

    print(
        "[hidream_o1_cfg] cond text_len=", cond.s_text,
        " uncond text_len=", uncond.s_text,
        " image_len=", image_len,
        " S_TOTAL=", GEN_S_TOTAL,
        " guidance=", guidance,
        " cfg=", do_cfg,
    )
    if cond.s_text + image_len != GEN_S_TOTAL:
        raise Error("hidream_o1 cfg: cond seq != fixed S_TOTAL")
    if uncond.s_text + image_len != GEN_S_TOTAL:
        raise Error("hidream_o1 cfg: uncond seq != fixed S_TOTAL (pad mismatch)")

    # Load the DiT once at the module-level S (never inside the step loop).
    var dit = HiDreamO1Offloaded[GEN_S_TOTAL].load(
        String("/home/alex/HiDream-O1-Image-Dev-weights"), cfg, ctx
    )
    var sched = HiDreamO1Scheduler.full_n_step(GEN_STEPS, Float32(3.0))
    var num_steps = sched.num_inference_steps()
    var noise_scale_start = Float32(7.5)

    for step_idx in range(num_steps):
        var step_t = sched.timestep(step_idx)
        var t_pixeldit = Float32(1.0) - step_t / Float32(1000.0)
        var sigma_clamped = step_t / Float32(1000.0)
        if sigma_clamped < Float32(0.001):
            sigma_clamped = Float32(0.001)

        print("[hidream_o1_cfg] step", step_idx + 1, "/", num_steps, " t=", step_t)

        # cond forward -> v_cond.
        var x_pred_full = dit.forward(
            cond.text_ids, z, cond.t_pos.copy(), cond.h_pos.copy(),
            cond.w_pos.copy(), cond.ar_len, t_pixeldit, ctx,
        )
        var x_pred_cond = gather_image_rows(x_pred_full, cond.s_text, image_len, ctx)
        var v_cond = compute_velocity(x_pred_cond, z, sigma_clamped, ctx)

        var v_guided: Tensor
        if do_cfg:
            # padded-uncond forward (pad keys masked) -> v_uncond.
            var x_pred_full_u = dit.forward_padded(
                uncond.text_ids, z, uncond.t_pos.copy(), uncond.h_pos.copy(),
                uncond.w_pos.copy(), uncond.ar_len, uncond.key_valid.copy(),
                t_pixeldit, ctx,
            )
            var x_pred_uncond = gather_image_rows(
                x_pred_full_u, uncond.s_text, image_len, ctx
            )
            var v_uncond = compute_velocity(x_pred_uncond, z, sigma_clamped, ctx)
            # v_guided = v_uncond + guidance*(v_cond - v_uncond)  (pipeline.rs:537)
            var diff = sub(v_cond, v_uncond, ctx)
            var scaled = mul_scalar(diff, guidance, ctx)
            v_guided = add(v_uncond, scaled, ctx)
        else:
            v_guided = v_cond^

        # model_output = -v_guided (pipeline.rs:546).
        var model_output = mul_scalar(v_guided, Float32(-1.0), ctx)

        # Deterministic Default scheduler ignores the noise arg; pass zeros.
        var noise_sh = model_output.shape()
        var zeros = randn(noise_sh^, UInt64(0), STDtype.F32, ctx)
        zeros = mul_scalar(zeros, Float32(0.0), ctx)
        z = sched.step(
            model_output, step_idx, z, zeros, noise_scale_start, Float32(0.0), ctx,
        )
    return z^


def hidream_o1_t2i_generate_cfg(
    model_dir: String,
    tokenizer_json: String,
    prompt: String,
    negative_prompt: String,
    guidance: Float32,
    out_png: String,
) raises:
    var cfg = HiDreamO1Config.dev_8b()
    if cfg.head_dim != 128:
        raise Error("hidream_o1: unexpected head_dim")

    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(tokenizer_json)
    var p = GEN_PATCH

    # cond at its own (common) S_text; uncond padded up to the same S_text.
    var cond = build_t2i_input(tok, cfg, prompt, GEN_HP, GEN_WP, GEN_S_TEXT)
    var unc_prompt = negative_prompt
    if unc_prompt.byte_length() == 0:
        unc_prompt = String(" ")
    var uncond = build_t2i_input(tok, cfg, unc_prompt, GEN_HP, GEN_WP, GEN_S_TEXT)

    print("[hidream_o1_cfg] cond raw s_text=", cond.s_text,
          " uncond padded s_text=", uncond.s_text)

    var seed = UInt64(0)
    var noise_scale_start = Float32(7.5)
    var z_sh = List[Int]()
    z_sh.append(1); z_sh.append(3); z_sh.append(GEN_H); z_sh.append(GEN_W)
    var z0 = randn(z_sh^, seed, STDtype.BF16, ctx)
    z0 = mul_scalar(z0, noise_scale_start, ctx)
    var z = patchify(z0, p, ctx)  # [1, L, 3072]

    z = denoise(z^, cond, uncond, guidance, ctx)

    var img = unpatchify(z, 3, GEN_H, GEN_W, p, ctx)
    save_png(img, out_png, ctx, ValueRange.SIGNED)
    print("hidream_o1 cfg generate saved ->", out_png)


def main() raises:
    hidream_o1_t2i_generate_cfg(
        String("/home/alex/HiDream-O1-Image-Dev-weights"),
        String("/home/alex/HiDream-O1-Image-Dev-weights/tokenizer.json"),
        String("a photograph of a red apple on a wooden table"),
        String(" "),
        Float32(4.0),
        String("/home/alex/mojodiffusion/output/hidream_o1_512_cfg.png"),
    )
