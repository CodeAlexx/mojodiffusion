# train_zimage_real.mojo — Z-Image (NextDiT) LoRA REAL training loop.
#
# Z-Image LoRA stack (models/zimage/zimage_stack_lora.mojo). Real base weights,
# local Mojo-prepared cache; no synthetic tensors and no Rust/Python cache
# dependency. Mirrors train_klein_real.mojo's loop structure (timing, grad clip,
# board, PROG line).
#
# Per step (translated from train_zimage.rs main loop):
#   1. load cached {latent [1,16,72,56], text_embedding [1,512,2560], text_mask}
#   2. latent <- (latent - VAE_SHIFT) * VAE_SCALE         (train_zimage.rs:1051)
#   3. x_seq  = x_embedder(patchify(latent))              (post-embedder tokens)
#      cap_seq= cap_embedder(text_embedding)
#   4. sigma  = logit_normal(shift=1.0) ; sigma_idx = floor(sigma*1000) clamp
#      t_value= (1000 - sigma_idx)/1000                   (train_zimage.rs:1125)
#   5. adaln  = t_embedder(t_value); per-block RAW modvecs + f_scale
#   6. flow-match in LATENT space:
#        noisy_latent = sigma*noise + (1-sigma)*latent
#        target       = patchify(noise - latent)            (v-prediction)
#   7. x_embedder(noisy_latent) -> zimage_stack_lora_forward -> velocity [N_IMG, OUT_CH]
#   8. loss = MSE(-raw_velocity, target_img); d_raw = -(2/N)(-raw_velocity - target_img)
#      (the stack outputs ONLY the N_IMG image rows, so the flow-match target is
#       taken on the IMAGE-token sub-sequence — see _img_target.)
#   9. zimage_stack_lora_backward -> LoRA grads; grad_norm = L2; clip(1.0)
#  10. zimage_lora_adamw_step_main_only ; PRINT PROG step loss grad lr secs
#
# Recipe scalars (train_zimage.rs released-preset defaults):
#   lr=3e-4, rank=16, alpha=1.0, timestep_shift=1.0, clip_grad_norm=1.0,
#   VAE_SHIFT=0.1159, VAE_SCALE=0.3611, NUM_TRAIN_TIMESTEPS=1000.
#
# HARD DTYPE RULE (2026-06-02): Z-Image training is BF16/BP16 for base model
# weights. OneTrainer does not train a full-F32 Z-Image model, and neither
# should this trainer. A full-F32 base/model load will OOM on 24 GB cards.
#
# The current stack still carries activations, scalar reductions, LoRA masters,
# and a few small norm compatibility tensors as F32. That is not a full-F32
# model. Large block projection and MLP weights must stay in checkpoint dtype
# via load_zimage_block_weights_prefixed_mixed until a full mixed/offload stack
# lands.
#
# MEMORY (measured budget): full-depth resident all-F32 base = 24.6 GB > 24 GB.
# Full-depth LoRA training must preserve BF16/BP16 base projections and avoid
# materializing frozen base d_W. If this path OOMs, add block offload; do not
# fall back to all-F32 or reduced-depth and call it a training baseline.
#
# SELF-CONTAINED TRAINER RULE (2026-06-02): Z-Image prepare/train/sample runtime
# must be Mojo-owned. OneTrainer is the read-only source of truth for formulas and
# baselines; Rust/Python may be used only for offline parity evidence, not as the
# training cache producer or runtime dependency.
#
# Run (real 512-bucket LoRA training):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/train_zimage_real.mojo [steps]

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import sys_system
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.zimage.weights import (
    ZImageBlockWeights, load_zimage_block_weights_prefixed_mixed,
)
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.lora_block import ZIMAGE_SLOTS
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraSet, ZImageLoraGrads, build_zimage_lora_set,
    zimage_lora_set_to_device,
    zimage_stack_lora_forward_main_device, zimage_stack_lora_backward_main_device,
    zimage_lora_adamw_step_main_only, save_zimage_lora_main_only,
)
from serenitymojo.models.zimage.real_weights import (
    ZImageRealAux, load_zimage_real_aux, build_adaln, build_block_modvecs,
    build_f_scale, build_cap_seq, build_x_seq, build_rope, build_positions,
)
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.klein_dataset import KleinCache


# ── arch (Z-Image, from transformer config; H/Dh/D fixed comptime) ───────────
comptime H = 30
comptime Dh = 128
comptime D = H * Dh          # 3840
comptime F = 10240           # SwiGLU per-gate hidden
comptime CAP_DIM = 2560      # Qwen3 hidden
comptime ADALN_DIM = 256
comptime T_SCALE = Float32(1000.0)
comptime ROPE_THETA = Float32(256.0)
comptime AXIS0 = 32
comptime AXIS1 = 48
comptime AXIS2 = 48
comptime EPS = Float32(1e-5)
comptime FINAL_EPS = Float32(1e-6)
comptime OUT_CH = 64         # patchified output channels (16ch * 2 * 2)
comptime PATCH = 2

# ── resolution: OneTrainer Alina "512" bucket is 576x448 image -> latent
# [16,72,56] -> patch2 -> 36x28=1008 real image tokens. Diffusers pads image
# tokens to a multiple of 32, so the transformer sees 1024 image rows and loss
# is applied only to the first 1008 rows.
comptime LAT_C = 16
comptime LAT_H = 72
comptime LAT_W = 56
comptime HT = LAT_H // PATCH  # 36
comptime WT = LAT_W // PATCH  # 28
comptime N_IMG_REAL = HT * WT # 1008
comptime IMG_PAD = (32 - (N_IMG_REAL % 32)) % 32
comptime N_IMG = N_IMG_REAL + IMG_PAD # 1024
comptime CAP_LEN = 224        # first-sample OneTrainer cap seq after mask prune + pad32
comptime N_TXT = CAP_LEN
comptime S = N_IMG + N_TXT    # 1248

# ── full Z-Image depth. Reduced-depth runs are smoke-only and not a baseline. ─
comptime NUM_NR = 2
comptime NUM_CR = 2
comptime MAIN_DEPTH = 30
comptime OVERFIT_PROBE = False

# ── recipe (train_zimage.rs released-preset) ─────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)
comptime LR = Float32(3.0e-4)
comptime TIMESTEP_SHIFT = Float32(1.0)
comptime VAE_SHIFT = Float32(0.1159)
comptime VAE_SCALE = Float32(0.3611)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

comptime TRANSFORMER_DIR = "/home/alex/.serenity/models/zimage_base/transformer"
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/alina_zimage_cache"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_zimage"
comptime TRAIN_ADAPTER_START = (NUM_NR + NUM_CR) * ZIMAGE_SLOTS
comptime TRAIN_ADAPTER_COUNT = MAIN_DEPTH * ZIMAGE_SLOTS


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
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


def _l2(h: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        s += v * v
    return sqrt(s)


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


@fieldwise_init
struct FlatStats(Copyable, Movable):
    var mean: Float64
    var std: Float64
    var max_abs: Float32


def _flat_stats(v: List[Float32]) -> FlatStats:
    if len(v) == 0:
        return FlatStats(0.0, 0.0, Float32(0.0))
    var sum = 0.0
    var max_abs = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        sum += Float64(x)
        var ax = x if x >= 0.0 else -x
        if ax > max_abs:
            max_abs = ax
    var mean = sum / Float64(len(v))
    var ss = 0.0
    for i in range(len(v)):
        var d = Float64(v[i]) - mean
        ss += d * d
    return FlatStats(mean, sqrt(ss / Float64(len(v))), max_abs)


def _global_norm(grads: ZImageLoraGrads, start: Int, end: Int) -> Float64:
    var ss = 0.0
    for i in range(start, end):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: ZImageLoraGrads, max_norm: Float32, start: Int, end: Int) -> Float64:
    var gn = _global_norm(grads, start, end)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(start, end):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


@fieldwise_init
struct StepResult(Copyable, Movable):
    var loss: Float32
    var grad: Float32
    var secs: Float32
    var lora_b_sum: Float32
    var lora_b_nonzero: Int
    var nonfinite: Int


def _valid_cap_from_mask(mask: Tensor, ctx: DeviceContext) raises -> Int:
    var mask_h = cast_tensor(mask, STDtype.F32, ctx).to_host(ctx)
    var valid_cap = 0
    for i in range(len(mask_h)):
        if mask_h[i] > 0.5:
            valid_cap += 1
    return valid_cap


def _cache_valid_cap(cache: KleinCache, slot: Int, ctx: DeviceContext) raises -> Int:
    var s = cache.load(slot, ctx)
    return _valid_cap_from_mask(s.text_mask, ctx)


def _train_one_step_bucket[
    LAT_H_B: Int, LAT_W_B: Int, CAP_LEN_B: Int
](
    k: Int,
    run_steps: Int,
    slot: Int,
    step_seed: UInt64,
    cache: KleinCache,
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    mut lora: ZImageLoraSet,
    n_adapters: Int,
    final_lin_w: Tensor,
    final_lin_b: Tensor,
    x_pad_h: List[Float32],
    cap_pad_h: List[Float32],
    ctx: DeviceContext,
) raises -> StepResult:
    comptime HT_B = LAT_H_B // PATCH
    comptime WT_B = LAT_W_B // PATCH
    comptime N_IMG_REAL_B = HT_B * WT_B
    comptime IMG_PAD_B = (32 - (N_IMG_REAL_B % 32)) % 32
    comptime N_IMG_B = N_IMG_REAL_B + IMG_PAD_B
    comptime N_TXT_B = CAP_LEN_B
    comptime S_B = N_IMG_B + N_TXT_B

    var t0 = perf_counter_ns()

    var s = cache.load(slot, ctx)
    var lsh = s.latent.shape()
    if lsh[1] != LAT_C or lsh[2] != LAT_H_B or lsh[3] != LAT_W_B:
        raise Error("train_zimage_real: dispatched sample to wrong latent bucket")

    var lat_h = cast_tensor(s.latent, STDtype.F32, ctx).to_host(ctx)
    for i in range(len(lat_h)):
        lat_h[i] = (lat_h[i] - VAE_SHIFT) * VAE_SCALE

    var valid_cap = _valid_cap_from_mask(s.text_mask, ctx)
    if valid_cap <= 0 or valid_cap > CAP_LEN_B:
        raise Error("train_zimage_real: dispatched sample to wrong text bucket")

    var sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
    var sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
    if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx = NUM_TRAIN_TIMESTEPS - 1
    var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var t_value = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

    var noise_lat = _host_noise(LAT_C * LAT_H_B * LAT_W_B, SEED_BASE * UInt64(7919) + step_seed)
    var noisy_lat_h = List[Float32]()
    for i in range(len(lat_h)):
        noisy_lat_h.append(noise_lat[i] * sig + lat_h[i] * (Float32(1.0) - sig))
    var noisy_latent = Tensor.from_host(noisy_lat_h^, [1, LAT_C, LAT_H_B, LAT_W_B], STDtype.F32, ctx)

    var x_t = build_x_seq(aux, noisy_latent, LAT_C, LAT_H_B, LAT_W_B, PATCH, ctx)
    for _pad in range(IMG_PAD_B):
        for c in range(D):
            x_t.append(x_pad_h[c])

    var cap_feats = cast_tensor(s.text_embedding, STDtype.F32, ctx)
    var cap_full = cap_feats.to_host(ctx)
    var cap_vals = List[Float32]()
    for r in range(CAP_LEN_B):
        var src_r = r if r < valid_cap else valid_cap - 1
        for c in range(CAP_DIM):
            cap_vals.append(cap_full[src_r * CAP_DIM + c])
    var cap2 = Tensor.from_host(cap_vals^, [CAP_LEN_B, CAP_DIM], STDtype.F32, ctx)
    var cap_seq = build_cap_seq(aux, cap2, EPS, ctx)
    for r in range(valid_cap, CAP_LEN_B):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]

    var pos_step = build_positions(N_IMG_B, HT_B, WT_B, CAP_LEN_B, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var xr = build_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var cr = build_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos = cr[0].copy(); var cap_sin = cr[1].copy()
    var ur = build_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos = ur[0].copy(); var uni_sin = ur[1].copy()

    var adaln = build_adaln(aux, t_value, ADALN_DIM, T_SCALE, ctx)
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln, D, ctx))
    var main_mod = List[ZImageModVecs]()
    for i in range(MAIN_DEPTH):
        main_mod.append(build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln, D, ctx))
    var f_scale = build_f_scale(aux, adaln, D, ctx)
    var t_prep = perf_counter_ns()

    var lora_dev = zimage_lora_set_to_device(lora, ctx)
    var t_lora = perf_counter_ns()

    var fwd = zimage_stack_lora_forward_main_device[H, Dh, N_IMG_B, N_TXT_B, S_B](
        x_t.copy(), cap_seq.copy(),
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora_dev,
        f_scale.copy(), final_lin_w, final_lin_b,
        x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var t_fwd = perf_counter_ns()

    var tgt_patch = _patchify_target[LAT_H_B, LAT_W_B](noise_lat, lat_h, sig)
    var real_nout = len(tgt_patch)
    var seq_nout = len(fwd.out)
    var d_loss = List[Float32]()
    var pred_vals = List[Float32]()
    var sse = 0.0
    var inv_n = Float32(2.0) / Float32(real_nout)
    for i in range(real_nout):
        var pred = -fwd.out[i]
        pred_vals.append(pred)
        var diff = pred - tgt_patch[i]
        sse += Float64(diff) * Float64(diff)
        d_loss.append(-inv_n * diff)
    for _i in range(real_nout, seq_nout):
        d_loss.append(Float32(0.0))
    var loss = Float32(sse / Float64(real_nout))
    var t_loss = perf_counter_ns()

    if k == 1:
        var ps = _flat_stats(pred_vals)
        var ts = _flat_stats(tgt_patch)
        print("[DEBUG step=1] bucket=", LAT_H_B, "x", LAT_W_B, " cap=", CAP_LEN_B,
              " sigma_idx=", sigma_idx, " sig=", sig,
              " pred mean=", Float32(ps.mean), " std=", Float32(ps.std),
              " max_abs=", ps.max_abs, " target mean=", Float32(ts.mean),
              " std=", Float32(ts.std), " max_abs=", ts.max_abs)

    var grads = zimage_stack_lora_backward_main_device[H, Dh, N_IMG_B, N_TXT_B, S_B](
        d_loss, main_blocks, main_mod, lora_dev,
        f_scale.copy(), final_lin_w,
        uni_cos[], uni_sin[], fwd,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var t_bwd = perf_counter_ns()

    var gn_before = _clip(grads, CLIP_GRAD_NORM, TRAIN_ADAPTER_START, n_adapters)
    zimage_lora_adamw_step_main_only(lora, grads, k, LR, ctx)
    var t_opt = perf_counter_ns()

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    var b_absum = Float32(0.0)
    var b_nonzero = 0
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        var bs2 = _absum(lora.ad[i].b)
        b_absum += bs2
        if bs2 > 0.0:
            b_nonzero += 1
    print("PROG step=", k, " total=", run_steps, " slot=", slot,
          " bucket=", LAT_H_B, "x", LAT_W_B, " cap=", CAP_LEN_B,
          " loss=", loss, " grad=", Float32(gn_before), " lr=", LR,
          " loraB_sum=", b_absum, " loraB_nonzero=", b_nonzero, "/", TRAIN_ADAPTER_COUNT,
          " nonfinite=", grads.nonfinite_lora_grads, " secs=", Float32(secs))
    print("[TIMING step=", k,
          "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
          " lora_upload=", Float32(Float64(t_lora - t_prep) / 1.0e9),
          " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
          " loss=", Float32(Float64(t_loss - t_fwd) / 1.0e9),
          " bwd=", Float32(Float64(t_bwd - t_loss) / 1.0e9),
          " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
    return StepResult(loss, Float32(gn_before), Float32(secs), b_absum, b_nonzero, grads.nonfinite_lora_grads)


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

    print("=== Z-Image REAL LoRA training loop ===")
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " F=", F, " out_ch=", OUT_CH)
    print("  depth: NR=", NUM_NR, " CR=", NUM_CR, " MAIN=", MAIN_DEPTH, " (full model MAIN=30)")
    print("  buckets: 72x56 cap224/cap256, 88x48 cap224/cap256")
    print("  recipe: rank=", RANK, " alpha=", ALPHA, " lr=", LR, " shift=", TIMESTEP_SHIFT,
          " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  weights:", TRANSFORMER_DIR)
    print("  cache:", CACHE_DIR)

    # ── cache first: fail before loading the 24 GB-class model if prepare has
    # not produced the local Mojo cache yet.
    var cache = KleinCache(String(CACHE_DIR))
    print("[cache] samples:", cache.count())
    var k0 = cache.peek_key(0, ctx)
    print("[cache] first latent: C=", k0.c, " H=", k0.h, " W=", k0.w, " text_seq=", k0.seq)

    # ── load real base weights (frozen) ──────────────────────────────────────
    print("[load] opening sharded transformer dir")
    var st = ShardedSafeTensors.open(String(TRANSFORMER_DIR))
    print("[load] aux (embedders / per-block adaLN / final layer)")
    var aux = load_zimage_real_aux(st, NUM_NR, MAIN_DEPTH, ctx)
    print("[load] blocks: NR + CR + MAIN")
    var nr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_NR):
        nr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("noise_refiner.") + String(i), ctx))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("context_refiner.") + String(i), ctx))
    var main_blocks = List[ZImageBlockWeights]()
    for i in range(MAIN_DEPTH):
        main_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("layers.") + String(i), ctx))
    print("[load] resident blocks:", len(nr_blocks), "nr +", len(cr_blocks), "cr +", len(main_blocks), "main")
    var final_lin_w = aux.final_lin_w[].clone(ctx)
    var final_lin_b = aux.final_lin_b[].clone(ctx)

    var x_pad_h = aux.x_pad_token[].to_host(ctx)
    var cap_pad_h = aux.cap_pad_token[].to_host(ctx)
    print("[load] learned x/cap pad tokens loaded")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_zimage_lora_set(NUM_NR, NUM_CR, MAIN_DEPTH, D, F, RANK, ALPHA)
    var n_adapters = (NUM_NR + NUM_CR + MAIN_DEPTH) * ZIMAGE_SLOTS
    print("[lora] adapters:", TRAIN_ADAPTER_COUNT, "trainable main-layer adapters;",
          n_adapters, "allocated total (refiners frozen/excluded)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var b_absum_init = Float32(0.0)
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    for k in range(1, run_steps + 1):
        var slot = 0 if OVERFIT_PROBE else (k - 1) % cache.count()
        var step_seed = UInt64(1) if OVERFIT_PROBE else UInt64(k)
        var key = cache.peek_key(slot, ctx)
        if key.c != LAT_C:
            raise Error("train_zimage_real: unsupported latent channel count")
        var valid_cap = _cache_valid_cap(cache, slot, ctx)
        var loss: Float32
        if key.h == 72 and key.w == 56:
            if valid_cap <= 224:
                var r_72_224 = _train_one_step_bucket[72, 56, 224](
                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                    lora, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h, ctx,
                )
                loss = r_72_224.loss
            elif valid_cap <= 256:
                var r_72_256 = _train_one_step_bucket[72, 56, 256](
                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                    lora, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h, ctx,
                )
                loss = r_72_256.loss
            else:
                raise Error("train_zimage_real: caption too long for 256-token production bucket")
        elif key.h == 88 and key.w == 48:
            if valid_cap <= 224:
                var r_88_224 = _train_one_step_bucket[88, 48, 224](
                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                    lora, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h, ctx,
                )
                loss = r_88_224.loss
            elif valid_cap <= 256:
                var r_88_256 = _train_one_step_bucket[88, 48, 256](
                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                    lora, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h, ctx,
                )
                loss = r_88_256.loss
            else:
                raise Error("train_zimage_real: caption too long for 256-token production bucket")
        else:
            raise Error("train_zimage_real: unsupported Z-Image production bucket")
        if k == 1:
            first_loss = loss
        last_loss = loss

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        b_absum_final += _absum(lora.ad[i].b)
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        print("RESULT: REAL Z-IMAGE LORA TRAIN OK — LoRA-B grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        _ = sys_system(String("mkdir -p ") + String(LORA_DIR))
        _ = save_zimage_lora_main_only(lora, String(LORA_DIR) + String("/zimage_lora_step") + String(run_steps) + String(".safetensors"), ctx)
    else:
        print("RESULT: FAIL trains=", trains)


# ── helper: patchify the v-target (noise - latent) into OUT_CH channel-minor ──
# Ordering matches build_x_seq's patchify exactly: view [C,Ht,p,Wt,p] ->
# permute (Ht,Wt,p,p,C) -> reshape [Ht*Wt, p*p*C]. v-target = noise - latent.
def _patchify_target[LAT_H_B: Int, LAT_W_B: Int](noise_lat: List[Float32], lat_flat: List[Float32], sig: Float32) -> List[Float32]:
    var C = LAT_C
    var Hh = LAT_H_B
    var Ww = LAT_W_B
    var p = PATCH
    var ht = Hh // p
    var wt = Ww // p
    # token target in [C,H,W]: t[c,h,w] = noise - latent
    # output ordering: token (ih,iw) -> [ph, pw, c] channel-minor (p*p*C=64).
    var out = List[Float32]()
    for ih in range(ht):
        for iw in range(wt):
            for ph in range(p):
                for pw in range(p):
                    for c in range(C):
                        var hh = ih * p + ph
                        var ww = iw * p + pw
                        var idx = c * Hh * Ww + hh * Ww + ww
                        out.append(noise_lat[idx] - lat_flat[idx])
    return out^
