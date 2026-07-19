# Ideogram4SampleResident.mojo — sample-during-training denoise for Ideogram-4.
#
# Generates ONE sample latent from the model's CURRENT state — the resident fp8
# base trunk (`Ideogram4Weights`) PLUS the live, in-place-updated LoRA adapters
# (`Ideogram4LoraSet`) — by running the resident LoRA sampler forward inside a
# CFG Euler denoise loop on the SerenityTrainer / inference logit-normal sigma
# schedule. The result is a latent the caller VAE-decodes + writes to PNG.
#
# The sampler forward shares the training forward's packing, embedding, LoRA
# block stack, and final projection, but skips training-only saved activations.
# Its returned velocity is the toolkit convention: -( image rows ) in
# [1,128,GH,GW] F32.
#
# ── SCHEDULE + STEP (1:1 with the gated inference pipeline) ────────────────────
# ideogram4_pipeline.mojo (chunk 9, parity-PASSED) runs:
#   mean = ideogram4_schedule_mean(H, W, 0.5)           # resolution-aware mu
#   si   = make_step_intervals(STEPS)                    # 0..1 in STEPS+1 points
#   for step in range(STEPS-1, -1, -1):
#       t_val = ideogram4_logitnormal(si[step+1], mean)  # current (higher) sigma
#       s_val = ideogram4_logitnormal(si[step],   mean)  # next   (lower)  sigma
#       v_raw = CFG*pos_raw + (1-CFG)*neg_raw            # asymmetric CFG on RAW out
#       z     = z + v_raw*(s_val - t_val)                # Euler, s<t so step is "down"
# In that pipeline pos_raw/neg_raw are the RAW transformer image rows (NOT
# negated). The TRAINING forward returns the NEGATED velocity, vel = -raw, so the
# identical update written in terms of vel is:
#       v   = CFG*pos_vel + (1-CFG)*neg_vel              # = -v_raw
#       z   = z - v*(s_val - t_val)                      # == z + v_raw*(s-t)
# (NEGATED-velocity Euler step, exactly as the team-lead flagged.)
#
# t_flow vs model_t: the inference pipeline passes t_val DIRECTLY to
# ideogram4_forward as model-time (no 1-t flip there). The training forward does
# `model_t = 1 - t_flow` INTERNALLY. So to reproduce the same model-time we pass
#   t_flow = 1 - t_val
# into the training forward (then it computes model_t = 1-(1-t_val) = t_val). ✓
#
# ── CONDITIONING ───────────────────────────────────────────────────────────────
# Inline samples use caller-provided JSON prompt strings. The runner encodes them
# with the same Qwen3-VL chat-template/tokenizer/text-encoder path used by the
# Ideogram4 cache stager, then drops the text encoder before the transformer and
# optimizer state are loaded. The denoise loop receives prompt llm_features plus
# real text_len so pad rows are masked exactly like train/cache conditioning.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import add, mul, mul_scalar, reshape, permute
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.image.png import save_png
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder
from serenitymojo.models.vae.ideogram4_tiled_decode import (
    ideogram4_tiled_decode, ideogram4_tiled_decode_5x5_lowmem,
)
from serenitymojo.sampling.ideogram4_schedule import (
    ideogram4_logitnormal, ideogram4_schedule_mean, make_step_intervals,
)

from serenity_trainer.model.Ideogram4LoRABlock import Ideogram4LoraSet
from serenity_trainer.model.Ideogram4TextEncoder import (
    ideogram4_load_text_encoder_default,
    ideogram4_encode_text,
)
from serenity_trainer.trainer.Ideogram4LoRATrainStep import (
    ideogram4_lora_sample_velocity_resident,
)


# Canonical Ideogram-4 latent-norm + VAE paths — identical to the parity-gated
# inference path (ideogram4_pipeline.mojo:27 / ideogram4_generate_lora.mojo:23,122).
# The latent-norm scale/shift live in the fx fixture; the VAE is the model dir's.
comptime I4_LATENTNORM = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"
comptime I4_VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime I4_TOK_JSON = "/home/alex/.serenity/models/ideogram-4-fp8/tokenizer/tokenizer.json"
comptime I4_PAD_ID = 151643
comptime I4_SAMPLE_TEXT_TOKENS = 1024


def _render_chat_prompt(prompt_json: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt_json.copy()
        + String("<|im_end|>\n<|im_start|>assistant\n")
    )


def _looks_like_json_prompt(prompt: String) -> Bool:
    var bs = prompt.as_bytes()
    for i in range(prompt.byte_length()):
        var ch = bs[i]
        if ch == 0x20 or ch == 0x09 or ch == 0x0A or ch == 0x0D:
            continue
        return ch == 0x7B or ch == 0x5B
    return False


def ideogram4_encode_sample_prompt[NT: Int](
    prompt_json: String,
    ctx: DeviceContext,
    mut text_lens: List[Int],
) raises -> Tensor:
    if not _looks_like_json_prompt(prompt_json):
        raise Error(
            "ideogram4 inline sampler prompts must be JSON objects/arrays; pass"
            " a JSON string or a file containing JSON"
        )

    var tok = Qwen3Tokenizer(String(I4_TOK_JSON))
    var ids = tok.encode(_render_chat_prompt(prompt_json))
    var natural_len = len(ids)
    if natural_len > NT:
        raise Error(
            String("ideogram4 inline sampler prompt tokenized to ")
            + String(natural_len)
            + String(" tokens; max supported by this inline path is ")
            + String(NT)
        )
    while len(ids) < NT:
        ids.append(I4_PAD_ID)

    print("[Ideogram4-lora] encoding inline JSON prompt tokens ", natural_len, "/", NT)
    var enc = ideogram4_load_text_encoder_default(ctx)
    var feats = ideogram4_encode_text(enc, ids, ctx)

    var feats_masked: Tensor
    if natural_len < NT:
        var mask_host = List[Float32]()
        for j in range(NT):
            if j < natural_len:
                mask_host.append(Float32(1.0))
            else:
                mask_host.append(Float32(0.0))
        var mask_f32 = Tensor.from_host(mask_host^, [1, NT, 1], STDtype.F32, ctx)
        var mask = cast_tensor(mask_f32, STDtype.BF16, ctx)
        feats_masked = mul(feats, mask, ctx)
    else:
        feats_masked = feats^
    var llm = cast_tensor(feats_masked, STDtype.BF16, ctx)
    text_lens.append(natural_len)
    return llm^


# ──────────────────────────────────────────────────────────────────────────────
# ideogram4_sample_resident — CFG Euler denoise on the resident base + live LoRA.
#
# Inputs (all on the same DeviceContext as the trainer):
#   weights          resident fp8 base trunk (loaded once at trainer start)
#   loras            live LoRA set (updated in place by the optimizer each step)
#   cond_llm         [1, NT, 53248]  text conditioning for the COND pass
#   uncond_llm       [1, NT, 53248]  text conditioning for the UNCOND pass (zeros)
#   init_noise       [1, 128, GH, GW] F32 — the t=1 latent (pure noise)
#   n_steps          number of Euler steps (inference default 8)
#   cfg              classifier-free guidance scale (inference default 7.0)
#   grid_h, grid_w   latent grid (GH/GW; 16/16 for 256x256)
#
# Returns the denoised latent [1, 128, GH, GW] F32 (still patch-space; the caller
# denorms + unpatches + VAE-decodes exactly as ideogram4_pipeline.mojo does).
#
# NOTE on the comptime split: ideogram4_lora_sample_velocity_resident is
# parameterised [NT, GH, GW] (comptime). This wrapper is therefore also
# parameterised; the driver instantiates it with the SAME NT/GH/GW the trainer
# uses (the cache's fixed sequence length + grid).
# ──────────────────────────────────────────────────────────────────────────────
def ideogram4_sample_resident[NT: Int, GH: Int, GW: Int](
    weights: Ideogram4Weights,
    loras: Ideogram4LoraSet,
    cond_llm: Tensor,        # [1, NT, 53248]
    uncond_llm: Tensor,      # [1, NT, 53248]
    init_noise: Tensor,      # [1, 128, GH, GW] F32
    n_steps: Int,
    cfg: Float32,
    grid_h: Int,
    grid_w: Int,
    ctx: DeviceContext,
    cond_text_len: Int = NT,
    uncond_text_len: Int = 0,
) raises -> Tensor:
    if n_steps < 1:
        raise Error("ideogram4_sample_resident: n_steps must be >= 1")

    # resolution-aware logit-normal mean (mu) — pipeline uses pixel HxW where the
    # latent grid GH/GW maps to 2*GH x 2*GW patches x 8 VAE scale; the inference
    # pipeline passes the LATENT*16 == image px (256x256 for GH=GW=16). We mirror
    # the pipeline call: it used ideogram4_schedule_mean(256, 256, 0.5) for the
    # 16x16 grid, i.e. image px = grid * 16.
    var img_h = grid_h * 16
    var img_w = grid_w * 16
    var mean = ideogram4_schedule_mean(img_h, img_w, 0.5)
    var si = make_step_intervals(n_steps)

    var z = init_noise.clone(ctx)   # [1,128,GH,GW] F32, evolves in place each step
    print("[Ideogram4-lora] sample denoise begin GH=", GH, " GW=", GW, " steps=", n_steps)

    # high -> low sigma (Euler down the schedule), 1:1 with the pipeline loop.
    for step in range(n_steps - 1, -1, -1):
        var t_val = ideogram4_logitnormal(Float64(si[step + 1]), mean)  # current sigma
        var s_val = ideogram4_logitnormal(Float64(si[step]), mean)      # next sigma

        # The training forward flips model_t = 1 - t_flow internally; the pipeline
        # feeds t_val as model-time directly. Pass t_flow = 1 - t_val so the
        # training forward reconstructs model_t = t_val.
        var t_flow = Float32(1.0) - t_val

        # COND pass: resident base + live LoRA, conditioned on cond_llm.
        var cond_vel = ideogram4_lora_sample_velocity_resident[NT, GH, GW](
            weights, z, t_flow, cond_llm, loras, ctx, cond_text_len
        )
        # UNCOND pass: same trunk + LoRA, zeroed text conditioning.
        var uncond_vel = ideogram4_lora_sample_velocity_resident[NT, GH, GW](
            weights, z, t_flow, uncond_llm, loras, ctx, uncond_text_len
        )

        # asymmetric CFG on the (negated) velocities:
        #   v = cfg*pos_vel + (1-cfg)*neg_vel
        var v = add(
            mul_scalar(cond_vel, cfg, ctx),
            mul_scalar(uncond_vel, Float32(1.0) - cfg, ctx),
            ctx,
        )

        # NEGATED-velocity Euler step (== pipeline's z += v_raw*(s-t) with v=-v_raw):
        #   z = z - v*(s_val - t_val)
        z = add(z, mul_scalar(v, -(s_val - t_val), ctx), ctx)

    return z^


# ──────────────────────────────────────────────────────────────────────────────
# ideogram4_decode_latent_to_png — denorm + unpatch + VAE-decode + write PNG.
# 1:1 with the parity-gated decode tail (ideogram4_pipeline.mojo:93-112 /
# ideogram4_generate_lora.mojo:122-131):
#   denorm : zd = z*latent_scale + latent_shift            (128-dim broadcast)
#   unpatch: [1,GH,GW,2,2,32] -> permute(0,5,1,3,2,4) -> [1,32,2GH,2GW]
#   decode : LdmVaeDecoder[2GH,2GW].decode(bf16 latent) -> [1,3,16GH,16GW]
#   write  : save_png (SIGNED [-1,1] range, the VAE output convention)
# The decoder + latentnorm are loaded fresh PER CALL — sample cadence is rare
# (every N steps), so this keeps zero extra resident memory between samples
# (important: the trainer already holds two resident DiT trunks + LoRA + Adam).
# ──────────────────────────────────────────────────────────────────────────────
def ideogram4_decode_latent_to_png[GH: Int, GW: Int](
    z: Tensor,               # [1, 128, GH, GW] F32 (denoised patch-space latent)
    out_path: String,
    ctx: DeviceContext,
) raises:
    print("[Ideogram4-lora] decode begin GH=", GH, " GW=", GW)
    var ln = ShardedSafeTensors.open(I4_LATENTNORM)
    var scale = reshape(Tensor.from_view(ln.tensor_view("latent_scale"), ctx), [1, 1, 128], ctx)
    var shift = reshape(Tensor.from_view(ln.tensor_view("latent_shift"), ctx), [1, 1, 128], ctx)
    print("[Ideogram4-lora] decode latentnorm ready")

    # z is [1,128,GH,GW]; the inference denorm broadcasts over the 128 channel
    # dim with z laid out [1,NIMG,128]. Match the pipeline by permuting to
    # [1,GH,GW,128] -> reshape [1,NIMG,128] before the channel-broadcast multiply,
    # then it folds straight into the [1,GH,GW,2,2,32] unpatch.
    var z_hwc = permute(z, [0, 2, 3, 1], ctx)                 # [1,GH,GW,128]
    var z_tok = reshape(z_hwc, [1, GH * GW, 128], ctx)        # [1,NIMG,128]
    var zd = add(mul(z_tok, scale, ctx), shift, ctx)          # [1,NIMG,128] F32

    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)             # [1,32,GH,2,GW,2]
    var latent = reshape(zp, [1, 32, 2 * GH, 2 * GW], ctx)    # [1,32,2GH,2GW]
    print("[Ideogram4-lora] decode latent unpatch ready")

    comptime if GH >= 128 or GW >= 128:
        print("[Ideogram4-lora] decode tiled VAE 5x5 lowmem")
        var img = ideogram4_tiled_decode_5x5_lowmem[2 * GH, 2 * GW](
            latent, I4_VAE, ctx
        )
        print("[Ideogram4-lora] decode png save")
        save_png(img, out_path, ctx)
    elif GH >= 64 or GW >= 64:
        print("[Ideogram4-lora] decode tiled VAE 3x3")
        var img = ideogram4_tiled_decode[2 * GH, 2 * GW](latent, I4_VAE, ctx)
        print("[Ideogram4-lora] decode png save")
        save_png(img, out_path, ctx)
    else:
        print("[Ideogram4-lora] decode vae load")
        var dec = load_ideogram4_vae_decoder[2 * GH, 2 * GW](I4_VAE, ctx)
        print("[Ideogram4-lora] decode vae load ready")
        print("[Ideogram4-lora] decode vae forward")
        var img = dec.decode(cast_tensor(latent, STDtype.BF16, ctx), ctx)
        print("[Ideogram4-lora] decode png save")
        save_png(img, out_path, ctx)
    print("[Ideogram4-lora] decode png save ready")
