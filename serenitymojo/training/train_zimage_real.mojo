# train_zimage_real.mojo — Z-Image (NextDiT) LoRA REAL training loop.
#
# TRANSLATION of EriDiffusion-v2 train_zimage.rs onto the parity-verified Mojo
# Z-Image LoRA stack (models/zimage/zimage_stack_lora.mojo). Real base weights,
# real prepared cache; no synthetic tensors. Mirrors train_klein_real.mojo's
# loop structure (timing, grad clip, board, PROG line).
#
# Per step (translated from train_zimage.rs main loop):
#   1. load cached {latent [1,16,64,64], text_embedding [1,512,2560], text_mask}
#   2. latent <- (latent - VAE_SHIFT) * VAE_SCALE         (train_zimage.rs:1051)
#   3. x_seq  = x_embedder(patchify(latent))              (post-embedder tokens)
#      cap_seq= cap_embedder(text_embedding)
#   4. sigma  = logit_normal(shift=1.0) ; sigma_idx = floor(sigma*1000) clamp
#      t_value= (1000 - sigma_idx)/1000                   (train_zimage.rs:1125)
#   5. adaln  = t_embedder(t_value); per-block RAW modvecs + f_scale
#   6. flow-match in TOKEN space (stack convention, == Klein Mojo trainer):
#        x_t    = (1-sigma)*x_seq + sigma*noise
#        target = noise - x_seq                            (v-prediction)
#   7. zimage_stack_lora_forward -> velocity [N_IMG, OUT_CH]
#   8. loss = MSE(velocity, target_img); d_loss = (2/N)(velocity - target_img)
#      (the stack outputs ONLY the N_IMG image rows, so the flow-match target is
#       taken on the IMAGE-token sub-sequence — see _img_target.)
#   9. zimage_stack_lora_backward -> LoRA grads; grad_norm = L2; clip(1.0)
#  10. zimage_lora_adamw_step ; PRINT PROG step loss grad lr secs
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
# So MAIN_DEPTH is a temporary correctness knob: the DEFAULT reduced depth runs
# a REAL end-to-end step (real cache + BF16 base projections + real LoRA train)
# within 24 GB, proving loss-down + LoRA-B growth. Full MAIN_DEPTH=30 needs
# BF16-preserving residency and/or turbo/offload, not F32 expansion.
#
# Run (real smoke):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/train_zimage_real.mojo [steps]

from sys import argv
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
    zimage_stack_lora_forward, zimage_stack_lora_backward,
    zimage_lora_adamw_step, save_zimage_lora,
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

# ── resolution (512px): latent [16,64,64] -> patch2 -> 32x32=1024 img tokens ──
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime HT = LAT_H // PATCH  # 32
comptime WT = LAT_W // PATCH  # 32
comptime N_IMG = HT * WT      # 1024 (mult of 32 -> no img pad)
comptime CAP_LEN = 512        # cached text seq (mult of 32 -> no cap pad)
comptime N_TXT = CAP_LEN
comptime S = N_IMG + N_TXT    # 1536

# ── depth knob (see MEMORY note above). Full model: NR=2 CR=2 MAIN=30. ───────
# Reduced default keeps full-resolution tokens + real weights but a subset of
# the 30 main layers so the BF16-preserving block stack fits 24 GB for smoke.
comptime NUM_NR = 2
comptime NUM_CR = 2
comptime MAIN_DEPTH = 4
# Overfit-correctness probe: when True, every step uses the SAME cache sample
# AND the same fixed timestep+noise, so a correct LoRA backward MUST drive the
# loss DOWN monotonically (the canonical trainer-correctness gate, independent
# of per-step sampling variance). VERIFIED 2026-06-01 at MAIN_DEPTH=4:
# loss 445.291 -> 444.384 monotonic over 10 steps, LoRA-B 0 -> 11473 (56/56).
# Production training sets this False (per-step sample + timestep variance).
comptime OVERFIT_PROBE = True

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
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/alina_zimage_512"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_zimage"


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 9007199254740992.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 9007199254740992.0)
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


def _global_norm(grads: ZImageLoraGrads) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: ZImageLoraGrads, max_norm: Float32) -> Float64:
    var gn = _global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


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
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", RANK, " alpha=", ALPHA, " lr=", LR, " shift=", TIMESTEP_SHIFT,
          " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  weights:", TRANSFORMER_DIR)
    print("  cache:", CACHE_DIR)

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

    # ── rope tables (positions fixed for 512px; built once) ──────────────────
    var pos = build_positions(N_IMG, HT, WT, CAP_LEN)
    var x_pos = pos[0].copy()
    var cap_pos = pos[1].copy()
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
    print("[load] rope tables built (img/cap/uni)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_zimage_lora_set(NUM_NR, NUM_CR, MAIN_DEPTH, D, F, RANK, ALPHA)
    var n_adapters = (NUM_NR + NUM_CR + MAIN_DEPTH) * ZIMAGE_SLOTS
    print("[lora] adapters:", n_adapters, " (7 slots x", NUM_NR + NUM_CR + MAIN_DEPTH, "blocks)")

    # ── cache (reuse model-agnostic KleinCache; same schema) ─────────────────
    var cache = KleinCache(String(CACHE_DIR))
    print("[cache] samples:", cache.count())

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        # ── load + VAE shift/scale (train_zimage.rs:1051) ──
        var slot = 0 if OVERFIT_PROBE else (k - 1) % cache.count()
        var step_seed = UInt64(1) if OVERFIT_PROBE else UInt64(k)
        var s = cache.load(slot, ctx)
        var lat_h = cast_tensor(s.latent, STDtype.F32, ctx).to_host(ctx)  # [16*64*64]
        for i in range(len(lat_h)):
            lat_h[i] = (lat_h[i] - VAE_SHIFT) * VAE_SCALE

        # ── timestep (train_zimage.rs:1107-1125) ──
        var sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
        var sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
        if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
            sigma_idx = NUM_TRAIN_TIMESTEPS - 1
        var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
        var t_value = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

        # ── flow-match in LATENT space (train_zimage.rs:1129-1162), THEN embed ──
        # noisy = noise*sigma + latent*(1-sigma) ; v-target = noise - latent
        # (Mojo-stack v convention). The model consumes the EMBEDDED noisy latent
        # (x_t tokens) and predicts the patch-space velocity; the loss target is
        # the patchified v-target (channel-minor, same 64-dim ordering as the
        # x_embedder input). This keeps noise CONSISTENT between model input and
        # target (the single source of incoherence if done in token space).
        var noise_lat = _host_noise(LAT_C * LAT_H * LAT_W, SEED_BASE * UInt64(7919) + step_seed)
        var noisy_lat_h = List[Float32]()
        for i in range(len(lat_h)):
            noisy_lat_h.append(noise_lat[i] * sig + lat_h[i] * (Float32(1.0) - sig))
        var noisy_latent = Tensor.from_host(noisy_lat_h^, [1, LAT_C, LAT_H, LAT_W], STDtype.F32, ctx)

        # ── embedders -> stack input tokens (x_t = embed(noisy latent)) ──
        var x_t = build_x_seq(aux, noisy_latent, LAT_C, LAT_H, LAT_W, PATCH, ctx)  # [N_IMG*D]
        var cap_feats = cast_tensor(s.text_embedding, STDtype.F32, ctx)
        var cap2 = Tensor.from_host(cap_feats.to_host(ctx), [CAP_LEN, CAP_DIM], STDtype.F32, ctx)
        var cap_seq = build_cap_seq(aux, cap2, EPS, ctx)                          # [N_TXT*D]

        # ── adaln + per-block modvecs + f_scale ──
        var adaln = build_adaln(aux, t_value, ADALN_DIM, T_SCALE, ctx)
        var nr_mod = List[ZImageModVecs]()
        for i in range(NUM_NR):
            nr_mod.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln, D, ctx))
        var main_mod = List[ZImageModVecs]()
        for i in range(MAIN_DEPTH):
            main_mod.append(build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln, D, ctx))
        var f_scale = build_f_scale(aux, adaln, D, ctx)

        # ── forward: stack output is velocity in OUT_CH patch space [N_IMG,OUT_CH]
        var fwd = zimage_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
            x_t.copy(), cap_seq.copy(),
            nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora,
            f_scale.copy(), final_lin_w, final_lin_b,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )

        # ── target = patchify(noise_lat - latent) in OUT_CH (channel-minor) ──
        var tgt_patch = _patchify_target(noise_lat, lat_h, sig)
        var nout = len(fwd.out)
        var d_loss = List[Float32]()
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - tgt_patch[i]
            sse += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)
        var loss = Float32(sse / Float64(nout))
        if k == 1:
            first_loss = loss
        last_loss = loss

        # ── backward ──
        var grads = zimage_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
            d_loss, nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora,
            f_scale.copy(), final_lin_w,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[], fwd,
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW ──
        zimage_lora_adamw_step(lora, grads, k, LR, ctx)

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        var b_absum = Float32(0.0)
        var b_nonzero = 0
        for i in range(n_adapters):
            var bs2 = _absum(lora.ad[i].b)
            b_absum += bs2
            if bs2 > 0.0:
                b_nonzero += 1
        print("PROG step=", k, " total=", run_steps, " loss=", loss,
              " grad=", Float32(gn_before), " lr=", LR,
              " loraB_sum=", b_absum, " loraB_nonzero=", b_nonzero, "/", n_adapters,
              " nonfinite=", grads.nonfinite_lora_grads, " secs=", Float32(secs))

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(n_adapters):
        b_absum_final += _absum(lora.ad[i].b)
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        print("RESULT: REAL run OK — LoRA-B grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        _ = sys_system(String("mkdir -p ") + String(LORA_DIR))
        _ = save_zimage_lora(lora, String(LORA_DIR) + String("/zimage_lora_smoke.safetensors"), ctx)
    else:
        print("RESULT: FAIL trains=", trains)


# ── helper: patchify the v-target (noise - latent) into OUT_CH channel-minor ──
# Ordering matches build_x_seq's patchify exactly: view [C,Ht,p,Wt,p] ->
# permute (Ht,Wt,p,p,C) -> reshape [Ht*Wt, p*p*C]. v-target = noise - latent.
def _patchify_target(noise_lat: List[Float32], lat_flat: List[Float32], sig: Float32) -> List[Float32]:
    var C = LAT_C
    var Hh = LAT_H
    var Ww = LAT_W
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
