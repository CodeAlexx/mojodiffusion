# validation_sampler.mojo — shared validation helpers for cached prompt caps and
# image comparison.
#
# PURPOSE — the L2P sample-shift gate. After a LoRA training run we need to KNOW
# the LoRA actually changes the output (the "WITH vs WITHOUT pixel diff" check
# that memory flags as never-run for L2P). `generate_validation` returns the
# decoded RGB tensor so a driver can diff WITH-LoRA vs WITHOUT-LoRA; a diff of 0
# means the LoRA is not being applied (a real bug we want to catch BEFORE a long
# run). `pixel_l1` + `save_png` make the gate concrete.
#
# REUSE MAP (every line mirrors a proven module):
#   * LoRA validation path       ← sampling/klein_sampler.mojo
#       Live PEFT adapters are applied by the same stack used during training.
#       The old resident `Klein9BDiT.load_with_config + merge` path is not used
#       for 9B validation because it co-resides a full DiT and VAE and can OOM.
#   * denoise loop               ← pipeline/klein9b_pipeline_multistep_smoke.mojo
#       build_flux2_sigma_schedule → per-step Euler (dt = sigma[i+1]-sigma[i]),
#       timestep pre-scaled *1000 (BFL time_factor), CFG = neg+CFG*(pos-neg),
#       x = x + dt*pred, latent F32 throughout, BF16 only to feed the DiT.
#       The ONE difference vs the offloaded smoke: the resident Klein9BDiT has
#       forward_full (klein_dit.mojo:451) but NOT forward_full_cfg (that is
#       Klein9BOffloaded-only, klein_dit.mojo:708). So we call forward_full
#       TWICE per step (pos caps, then neg caps) and combine with flux2_cfg —
#       the exact math forward_full_cfg performs internally, just unfused.
#   * initial noise + token packing ← klein9b_pipeline_multistep_smoke.initial_tokens
#   * VAE decode + PNG           ← klein9b_pipeline_multistep_smoke main()
#       KleinVaeDecoder[LH,LW].load(...).decode(packed) → save_png(.., SIGNED).
#   * cached caps                ← io/cap_cache.load_tensor_bin (separate-process
#       encode, so the ~16 GB Qwen3 encoder and the 9B DiT never co-reside).
#
# Mojo 1.0.0b1: comptime grid dims (the DiT forward + VAE are comptime-shaped);
# move-only Tensor (x is reassigned each step — never read a stale binding);
# STDtype.F32 value; from_host(values, shape, dtype, ctx).
#
# Run via training/validation_sampler_smoke.mojo (after the compile lock frees).

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.klein_dit import Klein9BDiT, build_klein_rope_tables
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.ops.cast import cast_tensor, cast_tensor_if_needed
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, mul_scalar, reshape, permute
from serenitymojo.sampling.flux2_klein import build_flux2_sigma_schedule, flux2_cfg
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.sampling.klein_sampler import klein_sample


# Positive + negative caption embeddings (produced by a SEPARATE encode process,
# e.g. klein9b_encode_smoke.mojo). Mirrors the multistep smoke's KleinCaps.
@fieldwise_init
struct ValidationCaps(Movable):
    var pos: Tensor
    var neg: Tensor


def load_caps(caps_pos_path: String, caps_neg_path: String, ctx: DeviceContext) raises -> ValidationCaps:
    """Load the two cached caption embeddings from disk (BF16, bit-exact round
    trip via io/cap_cache.load_tensor_bin). No text encoder is loaded here."""
    var pos = load_tensor_bin(caps_pos_path, ctx)
    var neg = load_tensor_bin(caps_neg_path, ctx)
    return ValidationCaps(pos^, neg^)


# Initial latent: NCHW [1,128,LH,LW] randn → token NHWC [1,N_IMG,128]. Byte-for-
# byte the klein9b_pipeline_multistep_smoke.initial_tokens routine.
def _initial_tokens[N_IMG: Int, LH: Int, LW: Int](seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    var nchw_shape = List[Int]()
    nchw_shape.append(1)
    nchw_shape.append(128)
    nchw_shape.append(LH)
    nchw_shape.append(LW)
    var noise_nchw = randn(nchw_shape^, seed, STDtype.F32, ctx)
    var p = List[Int]()
    p.append(0)
    p.append(2)
    p.append(3)
    p.append(1)
    var nhwc = permute(noise_nchw, p^, ctx)
    var sh = List[Int]()
    sh.append(1)
    sh.append(N_IMG)
    sh.append(128)
    return reshape(nhwc, sh^, ctx)


# token NHWC [1,N_IMG,128] → packed NCHW [1,128,LH,LW] for the VAE. Mirrors
# klein9b_pipeline_multistep_smoke.tokens_to_packed_nchw.
def _tokens_to_packed_nchw[LH: Int, LW: Int](tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
    var nhwc_shape = List[Int]()
    nhwc_shape.append(1)
    nhwc_shape.append(LH)
    nhwc_shape.append(LW)
    nhwc_shape.append(128)
    var nhwc = reshape(tokens, nhwc_shape^, ctx)
    var p = List[Int]()
    p.append(0)
    p.append(3)
    p.append(1)
    p.append(2)
    return permute(nhwc, p^, ctx)


# ── Denoise ONE prompt to a latent on a RESIDENT (already-merged) Klein9BDiT ──
# `model` may or may not have a LoRA merged into it — generate_validation merges
# it BEFORE calling this. The loop is the multistep smoke's loop, with the
# pos/neg forward split made explicit (forward_full ×2 → flux2_cfg) because the
# resident model has no fused forward_full_cfg.
def _denoise_resident[
    N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int
](
    model: Klein9BDiT,
    caps: ValidationCaps,
    num_steps: Int,
    cfg_scale: Float32,
    seed: UInt64,
    ctx: DeviceContext,
) raises -> Tensor:
    var rope = build_klein_rope_tables[N_IMG, N_TXT, 32, 128](ctx, STDtype.BF16)
    var sigmas = build_flux2_sigma_schedule(num_steps, N_IMG)
    var x = _initial_tokens[N_IMG, LH, LW](seed, ctx)
    for i in range(num_steps):
        var t_curr = sigmas[i]
        var t_next = sigmas[i + 1]
        var dt = t_next - t_curr  # sigma[i+1] - sigma[i], normally < 0
        # Timestep fed to the DiT: sigma pre-scaled *1000 (BFL time_factor; the
        # Mojo shared t_embedder does NOT scale — multistep smoke does this).
        var tvals = List[Float32]()
        tvals.append(t_curr * 1000.0)
        var tsh = List[Int]()
        tsh.append(1)
        var timestep = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
        var xb = cast_tensor(x, STDtype.BF16, ctx)
        # Positive branch: forward_full(img=xb, txt=caps.pos, ...).
        var pred_pos_bf = model.forward_full[N_IMG, N_TXT, S](
            xb, caps.pos, timestep, rope[0], rope[1], ctx
        )
        # Negative branch: SAME img latent, neg caps. Re-cast xb is not needed —
        # forward_full borrows its inputs (does not consume the move-only xb in a
        # way that prevents reuse here, mirroring forward_full_cfg's two internal
        # passes over the same img tokens). To be safe against move semantics we
        # rebuild a fresh BF16 view for the negative pass.
        var xb2 = cast_tensor(x, STDtype.BF16, ctx)
        var pred_neg_bf = model.forward_full[N_IMG, N_TXT, S](
            xb2, caps.neg, timestep, rope[0], rope[1], ctx
        )
        var pred_pos = cast_tensor(pred_pos_bf, STDtype.F32, ctx)
        var pred_neg = cast_tensor(pred_neg_bf, STDtype.F32, ctx)
        # CFG: neg + CFG*(pos - neg). NO post-CFG sign flip for Klein.
        var pred = flux2_cfg(pred_pos, pred_neg, cfg_scale, ctx)
        # Direct-velocity Euler: x = x + dt*pred (F32 latent in/out).
        x = add(x, mul_scalar(pred, dt, ctx), ctx)
    return x^


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: generate one validation image. Loads the resident Klein DiT, OPTIONALLY
# merges a LoRA, denoises, decodes, and (if out_png != "") saves a PNG. Returns
# the decoded RGB [1,3,16*LH,16*LW] tensor so a caller can pixel-diff WITH vs
# WITHOUT the LoRA (the sample-shift gate).
# ─────────────────────────────────────────────────────────────────────────────
def generate_validation[
    N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int
](
    cfg: TrainConfig,         # config-file-driven arch (block counts, dims)
    model_path: String,
    vae_path: String,
    caps: ValidationCaps,
    lora_path: String,        # "" → no LoRA (baseline pass)
    lora_multiplier: Float32,
    num_steps: Int,
    cfg_scale: Float32,
    seed: UInt64,
    out_png: String,          # "" → don't write a file, just return the tensor
    ctx: DeviceContext,
) raises -> Tensor:
    if model_path != cfg.checkpoint:
        print("[validation] using config checkpoint instead of model_path override:", cfg.checkpoint)
    if vae_path != cfg.vae:
        print("[validation] using config VAE instead of vae_path override:", cfg.vae)
    if lora_multiplier != Float32(1.0):
        print("[validation] live LoRA sampler currently uses multiplier=1.0; requested", lora_multiplier)
    if lora_path == String(""):
        print("[validation] staged baseline pass (no LoRA)")
    else:
        print("[validation] staged live-LoRA pass:", lora_path)

    var txt_sh = List[Int]()
    txt_sh.append(N_TXT)
    txt_sh.append(cfg.joint_attention_dim)
    var pos_txt = cast_tensor_if_needed(
        reshape(caps.pos, txt_sh.copy(), ctx), STDtype.F32, ctx
    )
    var neg_txt = cast_tensor_if_needed(
        reshape(caps.neg, txt_sh^, ctx), STDtype.F32, ctx
    )
    return klein_sample[N_IMG, N_TXT, S, LH, LW, 32, 128](
        cfg, lora_path, pos_txt, neg_txt, cfg_scale, num_steps, seed, out_png, ctx,
    )


# ── Pixel L1 between two decoded RGB tensors — the sample-shift metric ────────
# Returns mean |a - b| over all pixels. 0.0 == identical == the LoRA had NO
# effect on the output (the bug the L2P gate hunts for). > 0 == the LoRA shifts
# the sample.
def pixel_l1(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("pixel_l1: shape mismatch " + String(len(ah)) + " vs " + String(len(bh)))
    if len(ah) == 0:
        return Float32(0.0)
    var s = Float64(0.0)
    for i in range(len(ah)):
        var d = Float64(ah[i] - bh[i])
        s += d if d >= 0.0 else -d
    return Float32(s / Float64(len(ah)))
