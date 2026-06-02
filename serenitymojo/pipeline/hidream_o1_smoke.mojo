# pipeline/hidream_o1_smoke.mojo — HiDream-O1-Image T2I glue (skeleton, RUN later).
#
# Reference, read line-by-line:
#   /home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/pipeline.rs
#     apply_chat_template_t2i (193-211), build_t2i_input (235-365),
#     generate (386-585), gather_image_rows (657-700),
#     compute_velocity (702-708), save_png (720-...)
#   /home/alex/EriDiffusion/inference-flame/src/bin/hidream_o1_infer.rs (CLI)
#
# Flow (T2I, batch 1, no ref-image):
#   1) chat template + boi + tms; tokenize (add_special_tokens=False) -> text_ids
#   2) build full stream [text_ids, vision_start, (L-1)*image_pad];
#      MRoPE positions over full stream; ar_len = AR-prefix length
#   3) z0 = noise_scale_start * randn([1,3,H,W]); patchify p=32 -> [1,L,3072]
#   4) for step: forward (cond [+uncond]); gather image rows (tail L);
#      v = (x_pred - z)/sigma_clamped; CFG; model_output = -v_guided;
#      scheduler.step (+ per-step noise for Flash)
#   5) unpatchify -> [1,3,H,W] in [-1,1]; save_png (SIGNED)
#
# NO VAE (RGB pixel space), NO separate text encoder (the spine IS the encoder).
#
# Reused: tokenizer/Qwen3Tokenizer, ops/layout.{patchify,unpatchify},
#   ops/random.randn, image/png.save_png, models/dit/hidream_o1,
#   sampling/hidream_o1_scheduler. Mojo 1.0.0b1.
#
# COMPILE-ONLY in this session (GPU wedged). The actual run is gated behind a
# `False` guard so the whole forward/scheduler/decode path typechecks without
# executing. ar_len, S, and the per-step loop wiring are STUBBED for the smoke
# (see report — the AR-prefix length builder needs the token_types_bin logic
# from build_t2i_input/pipeline.rs:317-333, transcribed below).

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


# Build the full token stream + MRoPE positions + ar_len for one T2I sample.
# Returns (full_ids, t_pos, h_pos, w_pos, s_text, ar_len). The model gets the
# text portion (text_ids) for embedding; the L image patches are appended by
# the DiT forward (concat). The full stream is used for positions + mask only.
#
# ar_len (pipeline.rs:317-333, model.rs:671-699): token_types_bin is 1 over the
# L image rows AND the tms row (txt_seq_len-1). The AR prefix is the leading
# run of token_types_bin==0, i.e. everything before the tms row. So
# ar_len = s_text - 1 (the tms slot is the first non-AR row).
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
    # ar_len = s_text - 1 (everything before the tms row is AR-causal).
    var ar_len = s_text - 1
    if ar_len < 0:
        ar_len = 0
    return T2ISample(text_ids^, pos[0].copy(), pos[1].copy(), pos[2].copy(), s_text, ar_len)


# v = (x_pred - z) / sigma  (F32).  pipeline.rs:702-708.
def compute_velocity(x_pred: Tensor, z: Tensor, sigma: Float32, ctx: DeviceContext) raises -> Tensor:
    var xf = cast_tensor(x_pred, STDtype.F32, ctx)
    var zf = cast_tensor(z, STDtype.F32, ctx)
    var diff = sub(xf, zf, ctx)
    return mul_scalar(diff, Float32(1.0) / sigma, ctx)


# Gather the L image rows from x_pred [1, S, 3072] — they are the contiguous
# tail (pipeline.rs:657-700: narrow(dim=1, start=s_text, length=L)).
def gather_image_rows(
    x_pred: Tensor, s_text: Int, image_len: Int, ctx: DeviceContext
) raises -> Tensor:
    return ts_slice(x_pred, 1, s_text, image_len, ctx)


# ── STRUCTURAL WORKAROUND for the 1.0.0b1 comptime-instantiator segfault ─────
# BISECTED (this session): the EXIT=139 segfault is triggered by `dit.forward`
# co-monomorphizing with `unpatchify` *while the comptime sequence length is a
# FUNCTION-LOCAL `comptime`*. Pinned exactly:
#   - DiT.forward alone .......................... EXIT=0
#   - DiT.forward + scheduler.step ............... EXIT=0   (NOT the trigger)
#   - DiT.forward + patchify ..................... EXIT=0
#   - DiT.forward + save_png ..................... EXIT=0
#   - DiT.forward + unpatchify (local comptime S)  EXIT=139 ← toxic pair
#   - same, but S is a MODULE-LEVEL comptime and
#     the denoise loop is its own top-level def ... EXIT=0  ← the fix
# Both `unpatchify` and the DiT's internal `_repeat_kv` use the same dynamic
# `Layout.row_major(-1)` GPU-kernel pattern; instantiating both transitively
# from one `main` with a function-scoped comptime S crashes the instantiator.
# Hoisting S to module scope (qwenimage_pipeline_smoke.mojo style) + putting the
# whole DiT/scheduler denoise loop in its own `def denoise()` (so `main` reaches
# unpatchify and the DiT loop through separate top-level defs, with S fixed at
# module scope) avoids the crash. This is the same structure the working
# qwenimage pipeline uses. NOTE: a mere `def`-split with a *local* comptime S
# does NOT help — the comptime must be at module scope.
#
# S_TOTAL = s_text + image_len is the DiT's static SDPA length. The smoke fixes
# it for the tiny 2x2-patch case (assert s_text + image_len == S_TOTAL). A real
# run picks S from the actual prompt/resolution — a production variant templates
# `denoise` over the supported S values or dispatches on a small table.
comptime SMOKE_H = 64
comptime SMOKE_W = 64
comptime SMOKE_PATCH = 32
comptime SMOKE_HP = SMOKE_H // SMOKE_PATCH            # 2
comptime SMOKE_WP = SMOKE_W // SMOKE_PATCH            # 2
comptime SMOKE_IMAGE_LEN = SMOKE_HP * SMOKE_WP        # 4
comptime SMOKE_S_TOTAL = 20                           # 16 text + 4 image rows
comptime SMOKE_STEPS = 1                              # tiny proof run


# Full denoise loop (DiT forwards + CFG + scheduler steps), isolated in its own
# top-level def at the module-level comptime S so it never co-monomorphizes with
# `unpatchify` in main. Returns the final patched latent z [1, L, 3072].
# timestep fed to the DiT MUST be t_pixeldit = 1 - step_t/1000, NOT sigma — the
# t_embedder rescales by *1000 internally (pipeline.rs:487, insight #4).
def denoise(
    var z: Tensor,
    cond: T2ISample,
    uncond: T2ISample,
    guidance_scale: Float32,
    seed: UInt64,
    ctx: DeviceContext,
) raises -> Tensor:
    var cfg = HiDreamO1Config.dev_8b()
    var do_cfg = guidance_scale > Float32(1.0)
    var image_len = SMOKE_IMAGE_LEN

    # S is fixed at module scope; assert the real seq matches (smoke contract).
    print(
        "[hidream_o1] cond text_len=", cond.s_text,
        " image_len=", image_len,
        " total=", cond.s_text + image_len,
        " fixed=", SMOKE_S_TOTAL,
    )
    print(
        "[hidream_o1] uncond text_len=", uncond.s_text,
        " image_len=", image_len,
        " total=", uncond.s_text + image_len,
    )
    if cond.s_text + image_len != SMOKE_S_TOTAL:
        raise Error("hidream_o1 smoke: prompt seq != fixed S_TOTAL")
    if do_cfg and uncond.s_text + image_len != SMOKE_S_TOTAL:
        raise Error("hidream_o1 smoke: uncond seq != fixed S_TOTAL")

    # Load the DiT once at the module-level S (never inside the step loop).
    var dit = HiDreamO1Offloaded[SMOKE_S_TOTAL].load(
        String("/home/alex/HiDream-O1-Image-Dev-weights"), cfg, ctx
    )
    var sched = HiDreamO1Scheduler.full_n_step(SMOKE_STEPS, Float32(3.0))
    var num_steps = sched.num_inference_steps()

    # noise_scale_start: Dev 7.5 (inference.py:33). noise_clip_std: Dev 2.5
    # (inference.py:35); only Flash injects per-step noise (pipeline.rs:475-481).
    var noise_scale_start = Float32(7.5)
    var noise_clip_std = Float32(2.5)

    # Denoise loop (pipeline.rs:484-577). s_noise is the per-step entry of the
    # linear noise-scale schedule; for Dev both endpoints equal noise_scale_start
    # so it is constant 7.5 across steps (pipeline.rs:455).
    for step_idx in range(num_steps):
        var step_t = sched.timestep(step_idx)
        var t_pixeldit = Float32(1.0) - step_t / Float32(1000.0)
        var sigma_clamped = step_t / Float32(1000.0)
        if sigma_clamped < Float32(0.001):
            sigma_clamped = Float32(0.001)

        # cond forward → image-row x_pred → v_cond = (x_pred - z)/sigma.
        var x_pred_full = dit.forward(
            cond.text_ids, z, cond.t_pos.copy(), cond.h_pos.copy(),
            cond.w_pos.copy(), cond.ar_len, t_pixeldit, ctx,
        )
        # The DiT concatenates [text_emb_with_t, patch_emb] → image rows are the
        # contiguous tail at s_text (pipeline.rs:657-700, vinput_mask run).
        var x_pred_cond = gather_image_rows(
            x_pred_full, cond.s_text, image_len, ctx
        )
        var v_cond = compute_velocity(x_pred_cond, z, sigma_clamped, ctx)

        # CFG: v_guided = v_uncond + s*(v_cond - v_uncond) (pipeline.rs:537-540).
        # Single forward when do_cfg is false (Dev default, guidance 0).
        var v_guided: Tensor
        if do_cfg:
            var x_pred_full_u = dit.forward(
                uncond.text_ids, z, uncond.t_pos.copy(), uncond.h_pos.copy(),
                uncond.w_pos.copy(), uncond.ar_len, t_pixeldit, ctx,
            )
            var x_pred_uncond = gather_image_rows(
                x_pred_full_u, uncond.s_text, image_len, ctx
            )
            var v_uncond = compute_velocity(x_pred_uncond, z, sigma_clamped, ctx)
            var diff = sub(v_cond, v_uncond, ctx)
            var scaled = mul_scalar(diff, guidance_scale, ctx)
            v_guided = add(v_uncond, scaled, ctx)
        else:
            v_guided = v_cond^

        # F3 sign flip before the scheduler (pipeline.rs:546; skeptic OK).
        var model_output = mul_scalar(v_guided, Float32(-1.0), ctx)

        # Per-step noise for Flash (stochastic). The Rust draws from one StdRng
        # stream seeded seed+1 (pipeline.rs:473); the Mojo randn is stateless/
        # seed-only, so we derive a deterministic per-step seed
        # = seed + 1 + step_idx. NOT bit-identical to the Rust StdRng stream
        # (which is itself not bit-identical to the CUDA reference —
        # pipeline.rs:467-471); the N(0,1) statistics match, which is what the
        # schedule consumes.
        var s_noise = noise_scale_start  # constant for Dev (schedule flat)
        if sched.needs_step_noise():
            var noise_sh = model_output.shape()
            var step_seed = seed + UInt64(1) + UInt64(step_idx)
            var step_noise = randn(noise_sh^, step_seed, STDtype.F32, ctx)
            z = sched.step(
                model_output, step_idx, z, step_noise, s_noise,
                noise_clip_std, ctx,
            )
        else:
            # Deterministic schedulers ignore the noise arg; pass zeros.
            var noise_sh = model_output.shape()
            var zeros = randn(noise_sh^, UInt64(0), STDtype.F32, ctx)
            zeros = mul_scalar(zeros, Float32(0.0), ctx)
            z = sched.step(
                model_output, step_idx, z, zeros, s_noise, Float32(0.0), ctx
            )
    return z^


# End-to-end T2I smoke. Loads the offloaded DiT, draws noise, denoises in the
# separate `denoise()` def, then decodes (unpatchify) + saves here.
def hidream_o1_t2i_smoke(
    model_dir: String,
    tokenizer_json: String,
    prompt: String,
    out_png: String,
) raises:
    var cfg = HiDreamO1Config.dev_8b()
    if cfg.head_dim != 128:
        raise Error("hidream_o1: unexpected head_dim")

    # The EXIT=139 segfault is worked around structurally: the denoise loop
    # lives in the top-level `denoise()` def and S is a module-level comptime,
    # so the DiT path and `unpatchify` here reach the instantiator through
    # separate defs (see the bisection note above).
    var run_smoke = True
    if run_smoke:
        var ctx = DeviceContext()
        var tok = Qwen3Tokenizer(tokenizer_json)
        var p = SMOKE_PATCH

        # 64x64 image -> 2x2 patches (L=4) for the smoke. Real run: 1024+ .
        var cond = build_t2i_input(tok, cfg, prompt, SMOKE_HP, SMOKE_WP)
        var uncond = build_t2i_input(tok, cfg, String(" "), SMOKE_HP, SMOKE_WP)
        var tms_seen = False
        for i in range(len(cond.text_ids)):
            if cond.text_ids[i] == cfg.tms_token_id:
                tms_seen = True
        print("[hidream_o1] tms token seen=", tms_seen)

        # Dev default path uses guidance 0 → single forward (pipeline.rs:416).
        var guidance_scale = Float32(0.0)
        var seed = UInt64(0)

        # Initial RGB noise [1,3,H,W] * noise_scale_start. Initial latent draw is
        # seeded by `seed` (pipeline.rs:447); per-step noise by `seed+1`.
        var noise_scale_start = Float32(7.5)
        var z_sh = List[Int]()
        z_sh.append(1); z_sh.append(3); z_sh.append(SMOKE_H); z_sh.append(SMOKE_W)
        var z0 = randn(z_sh^, seed, STDtype.BF16, ctx)
        z0 = mul_scalar(z0, noise_scale_start, ctx)
        var z = patchify(z0, p, ctx)  # [1, L, 3*32*32=3072]

        # Denoise in the isolated top-level def (DiT forwards + CFG + scheduler).
        z = denoise(z^, cond, uncond, guidance_scale, seed, ctx)

        # unpatchify -> [1,3,H,W] in [-1,1]; save (SIGNED -> (v+1)*127.5).
        var img = unpatchify(z, 3, SMOKE_H, SMOKE_W, p, ctx)
        save_png(img, out_png, ctx, ValueRange.SIGNED)
        print("hidream_o1 smoke saved ->", out_png)

    print("hidream_o1 pipeline smoke complete")


def main() raises:
    hidream_o1_t2i_smoke(
        String("/home/alex/HiDream-O1-Image-Dev-weights"),
        String("/home/alex/HiDream-O1-Image-Dev-weights/tokenizer.json"),
        String("a serene mountain lake at dawn"),
        String("/home/alex/mojodiffusion/output/hidream_o1_smoke.png"),
    )
