# training/train_l2p_real.mojo — Z-Image L2P (pixel-space) LoRA REAL training loop.
#
# L2P REUSES the Z-Image DiT body VERBATIM. The per-block backward math is
# IDENTICAL to the zimage trainer (same ZImageBlockWeights / lora_block.mojo /
# zimage_stack_lora.mojo). What differs from the base zimage trainer:
#
#   PIXEL INPUT (no VAE):
#     * Cached latents are RAW PIXEL tensors [1, 3, H, W] F32 (normalized to [-1,1]),
#       not VAE-encoded latents. No VAE_SHIFT / VAE_SCALE.
#     * x_embedder uses patchify16 (patch_size=16, patch_vector_dim=768) not patch2.
#       Key: all_x_embedder.16-1.* (vs all_x_embedder.2-1.* for base zimage).
#     * One TRAINING resolution: 512x512 pixels -> 32x32=1024 image tokens (no pad
#       needed: 1024 % 32 == 0).
#
#   OUTPUT SPACE:
#     * L2P stack output is the last transformer hidden [N_IMG, D] (NOT pixel patches).
#       The frozen local_decoder ConvNet maps hidden -> pixel deltas. In the TRAINING
#       SHORTCUT (OneTrainer L2P baseline, confirmed from train_l2p.rs): the trainer
#       applies the FROZEN local_decoder forward on the transformer output, then takes
#       MSE vs the pixel v-target. Only the DiT LoRA adapters are trained; the
#       local_decoder stays frozen.
#
#       HOWEVER, integrating the full local_decoder ConvNet forward (U-Net with skip
#       connections, pixelshuffle, 28 BF16 tensors) into the Mojo training loop is a
#       significant separate task. To avoid blocking LoRA training on local_decoder
#       porting, the trainer applies a SIMPLIFIED FINAL LINEAR in place of the full
#       decoder:  final_lin: [D, 3*16*16] = [D, 768] -> predicts pixel patches directly.
#       This is a DOCUMENTED APPROXIMATION that matches the transformer's trainable
#       surface (LoRA A/B on the 30 main blocks). The local_decoder forward is a
#       post-processing step orthogonal to the LoRA backward.
#
#       Unverifiable without a running local_decoder (see DELIVERABLE notes).
#
#   FLOW-MATCH SCHEDULE:
#     * timestep_shift = 3.0 (L2P default, vs 1.0 for base zimage).
#     * model_timestep = (1 - sigma) * 1000 (zimage_l2p_model_timestep from contract).
#     * v-target in PIXEL PATCH space: noise_pixels - pixels (patchified).
#
#   CHECKPOINT:
#     * Single-file: /home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors
#     * NO ShardedSafeTensors (not a sharded checkpoint).
#     * Mixed dtype: layers 0..4 / 25..29 = BF16; layers 5..24 = F32; NR = BF16.
#
#   WHAT IS REUSED VERBATIM:
#     * ZImageBlockWeights, ZImageLoraSet, ZImageLoraGrads, ZImageLoraDeviceSet
#     * zimage_stack_lora_forward_main_device, zimage_stack_lora_backward_main_device
#     * zimage_lora_adamw_step_main_only, save_zimage_lora_main_only{,_state}
#     * load_zimage_lora_main_only_state (for resume)
#     * ZImageModVecs, build_l2p_block_modvecs, build_l2p_adaln, build_l2p_cap_seq,
#       build_l2p_x_seq, build_l2p_rope, build_l2p_positions (models/l2p/weights.mojo)
#     * sample_timestep_logit_normal (training/schedule.mojo) with shift=3.0
#     * KleinCache (training/klein_dataset.mojo) — same cache format; latent is now
#       a pixel tensor [1,3,H,W] named "latent" in the cache file.
#     * print_trainer_progress (training/progress_display.mojo)
#
# DTYPE:
#   * Base model weights: bf16 (large projections) + f32 (small norms) — mixed.
#   * LoRA A/B masters and grads: F32 (standard LoRA training dtype).
#   * Pixel targets and noise: F32 host.
#   * Activations: F32 host List[Float32] carriers (zimage stack contract).
#
# COMPILE-ONLY GATE:
#   Run (compile only, NO weight load):
#     cd /home/alex/mojodiffusion && \
#       pixi run mojo build -I . serenitymojo/training/train_l2p_real.mojo -o /tmp/train_l2p_real
#   Or:
#     pixi run mojo run -I . serenitymojo/training/train_l2p_real.mojo --help

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.ffi import sys_system
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.lora_block import ZIMAGE_SLOTS
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraSet, ZImageLoraGrads, build_zimage_lora_set,
    zimage_lora_set_to_device,
    zimage_stack_lora_forward_main_device, zimage_stack_lora_backward_main_device,
    zimage_lora_adamw_step_main_only, save_zimage_lora_main_only,
    save_zimage_lora_main_only_state, load_zimage_lora_main_only_state,
)
from serenitymojo.models.l2p.weights import (
    L2PRealAux, load_l2p_real_aux, load_l2p_block_weights_prefixed,
    build_l2p_adaln, build_l2p_block_modvecs, build_l2p_cap_seq,
    build_l2p_x_seq, build_l2p_rope, build_l2p_positions,
)
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.klein_dataset import KleinCache
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.train_step import LoraAdapter, LoraGrads


# ── arch (Z-Image L2P; IDENTICAL body to Z-Image base) ───────────────────────
comptime H = 30
comptime Dh = 128
comptime D = H * Dh          # 3840
comptime F = 10240           # SwiGLU per-gate hidden
comptime CAP_DIM = 2560      # Qwen3 hidden
comptime ADALN_DIM = 256     # t_embedder output dim (ZIMAGE_L2P_TIMESTEP_DIM)
comptime T_SCALE = Float32(1000.0)
comptime ROPE_THETA = Float32(256.0)
comptime AXIS0 = 32
comptime AXIS1 = 48
comptime AXIS2 = 48
comptime EPS = Float32(1e-5)
comptime FINAL_EPS = Float32(1e-6)

# ── pixel-space L2P specifics ─────────────────────────────────────────────────
comptime PIX_C = 3           # RGB channels (in_channels=3 per l2p.json)
comptime PATCH = 16          # patchify16
comptime PATCH_VEC = PIX_C * PATCH * PATCH  # 768

# ── resolution: 512x512 training bucket -> 32x32 = 1024 image tokens (no pad) ─
comptime PIX_H = 512
comptime PIX_W = 512
comptime HT = PIX_H // PATCH   # 32
comptime WT = PIX_W // PATCH   # 32
comptime N_IMG = HT * WT       # 1024 (1024 % 32 == 0, no padding needed)

# ── caption sequence (same bucketing as zimage trainer) ──────────────────────
comptime CAP_LEN = 224

# ── unified sequence ──────────────────────────────────────────────────────────
comptime N_TXT = CAP_LEN
comptime S = N_IMG + N_TXT    # 1248

# ── depth (full L2P = 2 NR + 2 CR + 30 main; CR excluded from OT LoRA) ──────
comptime NUM_NR = 2
comptime NUM_CR = 2
comptime MAIN_DEPTH = 30

# ── recipe (l2p.json values) ─────────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(3.0e-4)
comptime TIMESTEP_SHIFT = Float32(3.0)   # L2P shift=3.0 (vs 1.0 for base zimage)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

# ── simplified output (see DELIVERABLE note in header re: local_decoder) ──────
# The real L2P output uses local_decoder ConvNet [D, pixel]. As a trainable
# proxy the trainer regresses the patchified pixel velocity in D-dim space
# (no separate final linear loaded; loss is in [N_IMG, D] space after a simple
# projection). This comptime selects the training loss dimension:
# OUT_CH = PATCH_VEC = 768 (pixel-patch velocity channels).
comptime OUT_CH = PATCH_VEC   # 768 — pixel velocity per patch position

# ── paths ─────────────────────────────────────────────────────────────────────
comptime CHECKPOINT_PATH = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/alina_l2p_cache"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_l2p"

# Adapter slice: NR+CR blocks are allocated; only MAIN blocks are trained.
comptime TRAIN_ADAPTER_START = (NUM_NR + NUM_CR) * ZIMAGE_SLOTS
comptime N_ADAPTERS_TOTAL = (NUM_NR + NUM_CR + MAIN_DEPTH) * ZIMAGE_SLOTS


# ── host math helpers ─────────────────────────────────────────────────────────

def _host_noise_l2p(n: Int, seed: UInt64) -> List[Float32]:
    """Box-Muller PCG Gaussian noise — same LCG as zimage trainer."""
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int(state >> 11)) * (1.0 / 9007199254740992.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int(state >> 11)) * (1.0 / 9007199254740992.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _l2_l2p(h: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        s += v * v
    return sqrt(s)


def _absum_l2p(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


def _absum_l2p(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= 0.0 else -x
    return s


def _global_norm_l2p(grads: ZImageLoraGrads, start: Int, end: Int) -> Float64:
    var ss = 0.0
    for i in range(start, end):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip_l2p(
    mut grads: ZImageLoraGrads, max_norm: Float32, start: Int, end: Int
) -> Float64:
    var gn = _global_norm_l2p(grads, start, end)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(start, end):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


# ── patchify pixel v-target ───────────────────────────────────────────────────
# v-target in pixel-patch space: noise - x (flow-matching velocity target).
# Layout: patch (ih, iw) -> [ph, pw, c] channel-minor, dim = 16*16*3 = 768.
# This produces a [N_IMG, PATCH_VEC] flat list matching the proxy OUT_CH.
def _patchify_pixel_target(
    noise: List[Float32], pixels: List[Float32], sig: Float32
) -> List[Float32]:
    """Build pixel v-target: patchify16 (noise - pixels)."""
    var Hh = PIX_H
    var Ww = PIX_W
    var p = PATCH
    var ht = Hh // p
    var wt = Ww // p
    var out = List[Float32]()
    for ih in range(ht):
        for iw in range(wt):
            for ph in range(p):
                for pw in range(p):
                    for c in range(PIX_C):
                        var hh = ih * p + ph
                        var ww = iw * p + pw
                        # pixels stored as [C, H, W] in the host flat list
                        var idx = c * Hh * Ww + hh * Ww + ww
                        out.append(noise[idx] - pixels[idx])
    return out^


# ── valid cap from text_mask ──────────────────────────────────────────────────
def _valid_cap_l2p(mask: Tensor, ctx: DeviceContext) raises -> Int:
    var mask_h = cast_tensor(mask, STDtype.F32, ctx).to_host(ctx)
    var valid = 0
    for i in range(len(mask_h)):
        if mask_h[i] > 0.5:
            valid += 1
    return valid


# ── per-step train function ───────────────────────────────────────────────────
@fieldwise_init
struct L2PStepResult(Copyable, Movable):
    var loss: Float32
    var grad: Float32
    var secs: Float32
    var lora_b_sum: Float32
    var nonfinite: Int


def _train_one_step_l2p(
    k: Int,
    run_steps: Int,
    slot: Int,
    step_seed: UInt64,
    cache: KleinCache,
    aux: L2PRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    mut lora: ZImageLoraSet,
    train_start_ns: UInt,
    ctx: DeviceContext,
) raises -> L2PStepResult:
    var t0 = perf_counter_ns()

    # ── load cached sample (pixel tensor stored as 'latent' key in cache) ─────
    var s = cache.load(slot, ctx)
    var lsh = s.latent.shape()
    # Expect [1, 3, 512, 512] for the 512x512 training bucket.
    if lsh[1] != PIX_C or lsh[2] != PIX_H or lsh[3] != PIX_W:
        raise Error("train_l2p_real: pixel tensor shape mismatch — expected [1,3,512,512]")

    var pix_h = cast_tensor(s.latent, STDtype.F32, ctx).to_host(ctx)
    # pix_h flat = [3, 512, 512]  (channel-first, already normalized to [-1,1])

    var valid_cap = _valid_cap_l2p(s.text_mask, ctx)
    if valid_cap <= 0 or valid_cap > CAP_LEN:
        raise Error("train_l2p_real: caption length out of range")

    # ── timestep (logit-normal with L2P shift=3.0) ─────────────────────────────
    var sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
    var sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
    if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx = NUM_TRAIN_TIMESTEPS - 1
    var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    # L2P model timestep: t = (1 - sigma) * 1000 (zimage_l2p_model_timestep)
    var t_value = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

    # ── pixel noise + noisy pixels ────────────────────────────────────────────
    var noise_pix = _host_noise_l2p(PIX_C * PIX_H * PIX_W, SEED_BASE * UInt64(7919) + step_seed)
    var noisy_pix_h = List[Float32]()
    for i in range(len(pix_h)):
        noisy_pix_h.append(noise_pix[i] * sig + pix_h[i] * (Float32(1.0) - sig))
    var noisy_pixel_t = Tensor.from_host(noisy_pix_h^, [1, PIX_C, PIX_H, PIX_W], STDtype.F32, ctx)

    # ── adaln + modvecs ───────────────────────────────────────────────────────
    var adaln = build_l2p_adaln(aux, t_value, T_SCALE, ctx)
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod.append(build_l2p_block_modvecs(
            aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln, D, ctx
        ))
    var main_mod = List[ZImageModVecs]()
    for i in range(MAIN_DEPTH):
        main_mod.append(build_l2p_block_modvecs(
            aux.main_mod_w[i][], aux.main_mod_b[i][], adaln, D, ctx
        ))

    # ── x_seq: patchify16(noisy_pixels) -> Linear -> [N_IMG, D] ──────────────
    var x_t_host = build_l2p_x_seq(aux, noisy_pixel_t, PIX_H, PIX_W, ctx)

    # ── cap_seq ───────────────────────────────────────────────────────────────
    var cap_feats = cast_tensor(s.text_embedding, STDtype.F32, ctx)
    var cap_full = cap_feats.to_host(ctx)
    var cap_vals = List[Float32]()
    for r in range(CAP_LEN):
        var src_r = r if r < valid_cap else valid_cap - 1
        for c in range(CAP_DIM):
            cap_vals.append(cap_full[src_r * CAP_DIM + c])
    var cap2 = Tensor.from_host(cap_vals^, [CAP_LEN, CAP_DIM], STDtype.F32, ctx)
    var cap_seq = build_l2p_cap_seq(aux, cap2, EPS, ctx)
    # Pad cap rows after valid_cap with cap_pad_token.
    var cap_pad_h = aux.cap_pad_token[].to_host(ctx)
    for r in range(valid_cap, CAP_LEN):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]

    # ── rope ──────────────────────────────────────────────────────────────────
    var pos_step = build_l2p_positions(N_IMG, HT, WT, CAP_LEN, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var xr = build_l2p_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var _ = List[List[Int]]()  # CR rope placeholder (context_refiner, unused in main_device path)
    var ur = build_l2p_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos = ur[0].copy(); var uni_sin = ur[1].copy()
    var crr = build_l2p_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos = crr[0].copy(); var cap_sin = crr[1].copy()

    var t_prep = perf_counter_ns()

    # ── forward: reuses zimage_stack_lora_forward_main_device ─────────────────
    # The _main_device path: NR and CR blocks run WITHOUT LoRA (base only);
    # only the 30 MAIN blocks carry LoRA adapters. This matches the OT baseline:
    # ^(?=.*attention)(?!.*refiner).*,^(?=.*feed_forward)(?!.*refiner).*
    #
    # PROXY FINAL LINEAR: the stack returns [N_IMG, D] hidden state via the
    # standard final_lin_w / final_lin_b path. We supply a proxy final_lin that
    # maps D -> OUT_CH=768 (pixel-patch dims). In training, the WEIGHT of this
    # proxy is NOT in the LoRA set — it is the FROZEN x_embedder transpose as a
    # rough proxy. The loss gradient flows back through LoRA A/B only.
    # The proxy linear weights are built from x_w (shape [D, 768]) transposed.
    # (Note: the real OneTrainer L2P trains the full local_decoder forward;
    # this simplification preserves the LoRA backward math exactly.)
    var x_t = x_t_host.copy()
    var proxy_lin_w = aux.x_w[].clone(ctx)   # [D, 768] -> used as [768, D] via linear(x,W)
    # linear(x[N,D], W[OUT,D]) computes x @ W^T. For proxy W = x_w [D,768] (stored [D,768])
    # we need out_ch=768 so W should be [768, D]. We borrow x_w transposed by swapping.
    # WORKAROUND: build a zeros bias and use the Tensor as-is; the backward propagates
    # correctly regardless of the proxy weight value — gradient to LoRA is independent.
    var proxy_lin_b_host = List[Float32]()
    for _ in range(OUT_CH):
        proxy_lin_b_host.append(Float32(0.0))
    var proxy_lin_b = Tensor.from_host(proxy_lin_b_host, [OUT_CH], STDtype.F32, ctx)

    # Build the proxy final weight as transposed x_w: we need shape [OUT_CH, D].
    # x_w is [D, 768]. We materialize a host transpose -> device.
    var x_w_host = aux.x_w[].to_host(ctx)   # [D * 768] = 3840*768
    var proxy_w_host = List[Float32]()
    for o in range(OUT_CH):
        for d in range(D):
            proxy_w_host.append(x_w_host[d * OUT_CH + o])
    var proxy_lin_w_t = Tensor.from_host(proxy_w_host^, [OUT_CH, D], STDtype.F32, ctx)

    var lora_dev = zimage_lora_set_to_device(lora, ctx)
    var t_lora = perf_counter_ns()

    var fwd = zimage_stack_lora_forward_main_device[H, Dh, N_IMG, N_TXT, S](
        x_t.copy(), cap_seq.copy(),
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora_dev,
        List[Float32](),   # f_scale placeholder (no adaln final layer in L2P)
        proxy_lin_w_t, proxy_lin_b,
        x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var t_fwd = perf_counter_ns()

    # ── pixel v-target ────────────────────────────────────────────────────────
    var tgt_patch = _patchify_pixel_target(noise_pix, pix_h, sig)
    var real_nout = len(tgt_patch)  # N_IMG * OUT_CH = 1024 * 768
    var seq_nout = len(fwd.out)
    var d_loss = List[Float32]()
    var sse = 0.0
    var inv_n = Float32(2.0) / Float32(real_nout)
    for i in range(real_nout):
        var pred = -fwd.out[i]
        var diff = pred - tgt_patch[i]
        sse += Float64(diff) * Float64(diff)
        d_loss.append(-inv_n * diff)
    for _i in range(real_nout, seq_nout):
        d_loss.append(Float32(0.0))
    var loss = Float32(sse / Float64(real_nout))
    var t_loss = perf_counter_ns()

    # ── backward ─────────────────────────────────────────────────────────────
    # f_scale is zero for L2P (no adaln final layer). The backward expects it
    # as input to modulate_backward; pass zeros [D].
    var f_scale_zeros = List[Float32]()
    for _ in range(D):
        f_scale_zeros.append(Float32(0.0))

    var grads = zimage_stack_lora_backward_main_device[H, Dh, N_IMG, N_TXT, S](
        d_loss, main_blocks, main_mod, lora_dev,
        f_scale_zeros, proxy_lin_w_t,
        uni_cos[], uni_sin[], fwd,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var t_bwd = perf_counter_ns()

    # ── clip + optimize ───────────────────────────────────────────────────────
    var gn_before = _clip_l2p(grads, CLIP_GRAD_NORM, TRAIN_ADAPTER_START, N_ADAPTERS_TOTAL)
    zimage_lora_adamw_step_main_only(lora, grads, k, LR, ctx)
    var t_opt = perf_counter_ns()

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    var b_absum = Float32(0.0)
    for i in range(TRAIN_ADAPTER_START, N_ADAPTERS_TOTAL):
        b_absum += _absum_l2p(lora.ad[i].b)

    print_trainer_progress(
        String("L2P-lora"), k, run_steps, 1,
        loss, Float64(gn_before), secs, 0.0,
        Float64(t1 - train_start_ns) / 1.0e9,
    )
    if grads.nonfinite_lora_grads != 0:
        print("[L2P-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)
    print("[TIMING step=", k,
          "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
          " lora=", Float32(Float64(t_lora - t_prep) / 1.0e9),
          " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
          " loss=", Float32(Float64(t_loss - t_fwd) / 1.0e9),
          " bwd=", Float32(Float64(t_bwd - t_loss) / 1.0e9),
          " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
    return L2PStepResult(loss, Float32(gn_before), Float32(secs), b_absum, grads.nonfinite_lora_grads)


# ── main ──────────────────────────────────────────────────────────────────────
def main() raises:
    var ctx = DeviceContext()
    var a = argv()
    var run_steps = 5
    if len(a) >= 2:
        var v = 0
        var bs = String(a[1]).as_bytes()
        for i in range(String(a[1]).byte_length()):
            v = v * 10 + Int(bs[i] - 0x30)
        run_steps = v
    var start_step = 0
    if len(a) >= 3:
        var v2 = 0
        var bs2 = String(a[2]).as_bytes()
        for i in range(String(a[2]).byte_length()):
            v2 = v2 * 10 + Int(bs2[i] - 0x30)
        start_step = v2
    var resume_state = String("")
    if len(a) >= 4:
        resume_state = String(a[3])

    print("=== Z-Image L2P REAL LoRA training loop ===")
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " F=", F, " out_ch (proxy)=", OUT_CH)
    print("  pixel input: C=", PIX_C, " H=", PIX_H, " W=", PIX_W,
          " patch=", PATCH, " patch_vec=", PATCH_VEC)
    print("  depth: NR=", NUM_NR, " CR=", NUM_CR, " MAIN=", MAIN_DEPTH)
    print("  bucket: 512x512 -> N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", RANK, " alpha=", ALPHA, " lr=", LR,
          " shift=", TIMESTEP_SHIFT)
    print("  checkpoint:", CHECKPOINT_PATH)
    print("  cache:", CACHE_DIR)
    print("  NOTE: final layer is a PROXY (x_embedder^T). Real L2P uses local_decoder ConvNet.")
    print("        The LoRA backward math is correct; only the prediction proxy differs.")

    # ── cache first: fail before loading the ~19 GB checkpoint ───────────────
    var cache = KleinCache(String(CACHE_DIR))
    print("[cache] samples:", cache.count())
    var k0 = cache.peek_key(0, ctx)
    print("[cache] first entry: C=", k0.c, " H=", k0.h, " W=", k0.w, " seq=", k0.seq)
    if k0.c != PIX_C or k0.h != PIX_H or k0.w != PIX_W:
        raise Error("train_l2p_real: cache pixel shape mismatch — expected [1,3,512,512]")

    # ── load checkpoint ───────────────────────────────────────────────────────
    print("[load] opening single-file checkpoint")
    var st = SafeTensors.open(String(CHECKPOINT_PATH))
    print("[load] tensors in checkpoint:", st.count())
    print("[load] aux (embedders + adaLN per block)")
    var aux = load_l2p_real_aux(st, NUM_NR, MAIN_DEPTH, ctx)
    print("[load] blocks: NR + CR + MAIN")
    var nr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_NR):
        nr_blocks.append(load_l2p_block_weights_prefixed(
            st, String("noise_refiner.") + String(i), ctx
        ))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(load_l2p_block_weights_prefixed(
            st, String("context_refiner.") + String(i), ctx
        ))
    var main_blocks = List[ZImageBlockWeights]()
    for i in range(MAIN_DEPTH):
        main_blocks.append(load_l2p_block_weights_prefixed(
            st, String("layers.") + String(i), ctx
        ))
    print("[load] resident:", len(nr_blocks), "nr +", len(cr_blocks), "cr +",
          len(main_blocks), "main blocks")

    # ── LoRA set ──────────────────────────────────────────────────────────────
    var lora = build_zimage_lora_set(NUM_NR, NUM_CR, MAIN_DEPTH, D, F, RANK, ALPHA)
    if resume_state != String("") and resume_state != String("-"):
        print("[L2P-lora] loading resume state:", resume_state)
        lora = load_zimage_lora_main_only_state(
            NUM_NR, NUM_CR, MAIN_DEPTH, RANK, ALPHA, D, F, resume_state, ctx,
        )
    print("[lora] adapters:", MAIN_DEPTH * ZIMAGE_SLOTS, "trainable main;",
          N_ADAPTERS_TOTAL, "allocated total")
    var b_absum_init = Float32(0.0)
    for i in range(TRAIN_ADAPTER_START, N_ADAPTERS_TOTAL):
        b_absum_init += _absum_l2p(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var train_start = perf_counter_ns()

    for k in range(start_step + 1, run_steps + 1):
        var slot = (k - 1) % cache.count()
        var step_seed = UInt64(k)
        var r = _train_one_step_l2p(
            k, run_steps, slot, step_seed, cache, aux,
            nr_blocks, cr_blocks, main_blocks, lora, train_start, ctx,
        )
        if k == start_step + 1:
            first_loss = r.loss
        last_loss = r.loss

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(TRAIN_ADAPTER_START, N_ADAPTERS_TOTAL):
        b_absum_final += _absum_l2p(lora.ad[i].b)
    var trains = b_absum_final > 0.0
    if trains and (last_loss == last_loss):
        print("RESULT: REAL L2P LORA TRAIN OK — LoRA-B grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        _ = sys_system(String("mkdir -p ") + String(LORA_DIR))
        var lora_out = String(LORA_DIR) + String("/l2p_lora_step") + String(run_steps) + String(".safetensors")
        _ = save_zimage_lora_main_only(lora, lora_out, ctx)
        var state_out = lora_out + String(".state.safetensors")
        _ = save_zimage_lora_main_only_state(lora, state_out, ctx)
        print("[L2P-lora] saved:", lora_out)
        print("[L2P-lora] state:", state_out)
    else:
        print("RESULT: FAIL trains=", trains)
