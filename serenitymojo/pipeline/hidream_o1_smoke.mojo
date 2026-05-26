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


# End-to-end T2I (compile-only skeleton). Builds the whole path; the GPU work
# is behind a `False` guard. The real run will: load DiT, draw noise, denoise,
# decode, save. S (full sequence) is a comptime param of the DiT; the smoke
# uses a small S so the comptime sdpa case stays cheap to instantiate.
def hidream_o1_t2i_smoke(
    model_dir: String,
    tokenizer_json: String,
    prompt: String,
    out_png: String,
) raises:
    var cfg = HiDreamO1Config.dev_8b()
    if cfg.head_dim != 128:
        raise Error("hidream_o1: unexpected head_dim")

    # --- compile-only guard: no GPU ---
    # We typecheck the pipeline-SPECIFIC glue here (tokenizer, build_t2i_input,
    # patchify, velocity, gather, scheduler.step, unpatchify, save_png). The DiT
    # forward is exercised by its own probe (hidream_o1_probe.mojo) — calling
    # .load()+.forward() AND scheduler+patchify in one function crashed the
    # 1.0.0b1 comptime instantiator (segfault, not a source error). Keeping the
    # DiT path in its own translation unit is the workaround. The denoise loop
    # wiring is documented above and transcribed faithfully; the real run wires
    # dit.forward(...) into this loop at S = s_total (comptime).
    if False:
        var ctx = DeviceContext()
        var tok = Qwen3Tokenizer(tokenizer_json)

        # 64x64 image -> 2x2 patches (L=4) for the smoke. Real run: 1024+ .
        comptime H = 64
        comptime W = 64
        var p = cfg.patch_size  # 32
        var h_patches = H // p
        var w_patches = W // p
        var image_len = h_patches * w_patches

        var sample = build_t2i_input(tok, cfg, prompt, h_patches, w_patches)
        var sched = HiDreamO1Scheduler.dev_28step()

        # Initial RGB noise [1,3,H,W] * noise_scale_start (Dev 7.5).
        var noise_scale_start = Float32(7.5)
        var z_sh = List[Int]()
        z_sh.append(1); z_sh.append(3); z_sh.append(H); z_sh.append(W)
        var z0 = randn(z_sh^, UInt64(33), STDtype.BF16, ctx)  # seed+1
        z0 = mul_scalar(z0, noise_scale_start, ctx)
        var z = patchify(z0, p, ctx)  # [1, L, 3*32*32=3072]

        var step_t = sched.timestep(0)
        var sigma_clamped = step_t / Float32(1000.0)
        if sigma_clamped < Float32(0.001):
            sigma_clamped = Float32(0.001)

        # x_pred placeholder = z (the real run supplies dit.forward(...)[image rows]).
        # compute_velocity borrows both args (read), so pass z for both here.
        var v = compute_velocity(z, z, sigma_clamped, ctx)
        var model_output = mul_scalar(v, Float32(-1.0), ctx)  # pipeline F3 sign flip

        var noise_sh = model_output.shape()
        var step_noise = randn(noise_sh^, UInt64(34), STDtype.F32, ctx)
        z = sched.step(
            model_output, 0, z, step_noise, noise_scale_start, Float32(2.5), ctx
        )

        # gather (tail L) — exercised on a full-stream-shaped tensor.
        var gathered = gather_image_rows(z, 0, image_len, ctx)
        _ = gathered

        # unpatchify -> [1,3,H,W] in [-1,1]; save (SIGNED -> (v+1)*127.5).
        var img = unpatchify(z, 3, H, W, p, ctx)
        save_png(img, out_png, ctx, ValueRange.SIGNED)

    print("hidream_o1 pipeline skeleton compiled (run gated; GPU wedged)")


def main() raises:
    hidream_o1_t2i_smoke(
        String("/home/alex/HiDream-O1-Image-Dev-weights"),
        String("/home/alex/HiDream-O1-Image-Dev-weights/tokenizer.json"),
        String("a serene mountain lake at dawn"),
        String("/home/alex/mojodiffusion/output/hidream_o1_smoke.png"),
    )
