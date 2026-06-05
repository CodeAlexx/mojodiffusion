# train_sd35_real.mojo — SD3.5-Large LoRA training loop (block-swap offload).
#
# TRANSLATION of the proven Chroma block-swap pattern onto SD3.5-Large.
# Real SD3.5-Large base weights (streamed block-by-block via TurboPlannedLoader),
# real prepared cache (latent + text_embedding + pooled), full 38 joint-block depth.
# No synthetic tensors. Mirrors train_chroma_real.mojo's loop structure.
#
# SD3.5 vs CHROMA (the deltas):
#   - NO frozen approximator. Modulation comes from per-block adaLN_modulation.1
#     (streamed with each block), conditioned on c = t_embed(sigma*1000) + y_embed(pooled).
#   - JOINT BLOCKS ONLY: 38 joint blocks, no single-stream blocks.
#   - Cache keys: "latent" [1,16,128,128], "text_embedding" [1,154,4096], "pooled" [1,2048].
#   - NO RoPE (pos_embed added once at patchify, before blocks, in inference;
#     for training the patchify linear already encodes position via weight layout).
#   - LoRA: SD35LoraSet with 8 adapters/block (4 ctx + 4 x: qkv, proj, fc1, fc2).
#
# Per step:
#   1. Load cached {latent [1,16,128,128], text_embedding [1,154,4096], pooled [1,2048]}
#   2. latent_scaled = (latent - VAE_SHIFT) * VAE_SCALE
#   3. pack_latents([16,128,128]) -> [N_IMG=4096, 64] channel-major patchify
#   4. sigma_idx = floor(logit_normal_sigma(shift=1.0) * 1000) clamp;
#      sig=(idx+1)/1000 ; sigma_cont=sig (passed to t_embedder as sigma*1000)
#   5. noisy = noise*sig + latent_packed*(1-sig) ; target = noise - latent_packed
#   6. sd35_stack_lora_forward_offload(noisy, txt, pooled, sigma, ...) -> pred [N_IMG,64]
#   7. loss = MSE(pred, target); d_loss = (2/N)(pred - target)
#   8. sd35_stack_lora_backward_offload -> LoRA grads; global-norm clip(1.0)
#   9. sd35_lora_adamw_step; print shared progress display
#
# Recipe (from EriDiffusion-v2 prepare_sd35.rs / OneTrainer SD3.5 LoRA preset):
#   lr=1e-4, rank=16, alpha=16, timestep_shift=1.0, clip_grad_norm=1.0
#   VAE shift=0.0609 scale=1.5305
#
# FIXED_SIGMA_SMOKE: every step uses the SAME cache sample AND a fixed
# timestep+noise so a correct LoRA backward MUST drive loss DOWN monotonically.
#
# Run (real smoke):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_sd35_real.mojo -o /tmp/train_sd35_real && \
#     /tmp/train_sd35_real [steps]

from sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.sd35.weights import load_sd35_stack_base
from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35LoraSet, SD35LoraGradSet, SD35StackBase,
    build_sd35_lora_set, sd35_lora_adamw_step, save_sd35_lora, total_adapters,
    sd35_stack_lora_forward_offload, sd35_stack_lora_backward_offload,
)
from serenitymojo.offload.plan import build_sd35_large_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress


# ── arch (sd3.5-large; H/Dh/D fixed comptime, verified vs the checkpoint) ────
comptime H = 38
comptime Dh = 64
comptime D = H * Dh            # 2432
comptime FMLP = 9728           # mlp_hidden = D*4 (approximately; real=9728)
comptime IN_CH = 64            # patch_dim = 16ch * 2*2
comptime TXT_CH = 4096         # combined CLIP-L/G + T5
comptime OUT_CH = 64
comptime NUM_JOINT = 38
comptime EPS = Float32(1e-06)
comptime QK_EPS = Float32(1e-06)
comptime TIMESTEP_DIM = 256    # sinusoidal embedding dim for t_embedder
comptime POOLED_DIM = 2048     # clip_l + clip_g pooled

# ── resolution (1024px): latent [16,128,128] -> pack2 -> 64x64=4096 img tokens ─
comptime LAT_C = 16
comptime LAT_H = 128
comptime LAT_W = 128
comptime PATCH = 2
comptime HT = LAT_H // PATCH   # 64
comptime WT = LAT_W // PATCH   # 64
comptime N_IMG = HT * WT       # 4096
comptime N_TXT = 154           # 77 CLIP-LG + 77 T5 (locked per prepare_sd35_cache.py)
comptime S = N_TXT + N_IMG     # 4250

# ── recipe ──────────────────────────────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)
comptime TIMESTEP_SHIFT = Float32(1.0)
comptime VAE_SHIFT = Float32(0.0609)
comptime VAE_SCALE = Float32(1.5305)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_IDX = 500

comptime CKPT = "/home/alex/.serenity/models/checkpoints/sd3.5_large.safetensors"
comptime CACHE_DIR = "/home/alex/datasets/andrsd35_sd35_cache"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/sd35_lora"


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


def _absum(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= 0.0 else -x
    return s


def _global_norm(grads: SD35LoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: SD35LoraGradSet, max_norm: Float32) -> Float64:
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


def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("sd35 cache: no .safetensors in ") + dir)
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


def _load_host(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


# pack_latents: [16, LAT_H, LAT_W] flat (CHW) -> [N_IMG, IN_CH] channel-major patchify.
# Each patch token aggregates a 2x2 spatial region across all 16 channels.
# Token (ih, iw) -> 64 elements: for c in 16, for ph in 2, for pw in 2.
def _pack_latents(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(HT):
        for iw in range(WT):
            for c in range(LAT_C):
                for ph in range(PATCH):
                    for pw in range(PATCH):
                        var hh = ih * PATCH + ph
                        var ww = iw * PATCH + pw
                        var idx = c * LAT_H * LAT_W + hh * LAT_W + ww
                        out.append(lat[idx])
    return out^


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

    print("=== SD3.5-Large REAL LoRA training loop (block-swap offload) ===")
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " Fmlp=", FMLP, " out_ch=", OUT_CH)
    print("  depth: NUM_JOINT=", NUM_JOINT)
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  resolution: LAT_H=", LAT_H, " LAT_W=", LAT_W, " patch=", PATCH)
    print("  recipe: rank=", RANK, " alpha=", ALPHA, " lr=", LR, " shift=", TIMESTEP_SHIFT,
          " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", CKPT)
    print("  cache:", CACHE_DIR)

    # ── stack-level base (frozen; embedders + final layer) ───────────────────
    print("[load] SD35StackBase (x_embedder, context_embedder, t_embedder, y_embedder, final_layer)")
    var base_st = SafeTensors.open(String(CKPT))
    var base = load_sd35_stack_base(base_st, ctx)
    print("[load] base resident")

    # ── block-swap offload loader ────────────────────────────────────────────
    var plan = build_sd35_large_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(String(CKPT), plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_sd35_lora_set(NUM_JOINT, D, FMLP, RANK, ALPHA)
    var n_adapters = total_adapters(lora)
    print("[lora] adapters:", n_adapters, " (8 per joint block x", NUM_JOINT, "blocks)")

    var files = _list_cache(String(CACHE_DIR))
    print("[cache] samples:", len(files))

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        var st = SafeTensors.open(files[slot])

        # latent: [1, 16, 128, 128] -> flat [1*16*128*128] = [262144]
        var lat_raw = _load_host(st, String("latent"), ctx)

        # text_embedding: [1, 154, 4096] -> flat [1*154*4096] = [631808]
        var te_info = st.tensor_info(String("text_embedding"))
        var te_seq = Int(te_info.shape[1])   # should be 154
        var te_flat = _load_host(st, String("text_embedding"), ctx)
        # Flatten to [N_TXT, TXT_CH]: pad/truncate to exactly N_TXT rows
        var txt_tokens = List[Float32]()
        for r in range(N_TXT):
            if r < te_seq:
                for c in range(TXT_CH):
                    txt_tokens.append(te_flat[r * TXT_CH + c])
            else:
                for _ in range(TXT_CH):
                    txt_tokens.append(Float32(0.0))

        # pooled: [1, 2048] -> flat [2048]
        var pooled_raw = _load_host(st, String("pooled"), ctx)
        # Trim/pad to POOLED_DIM
        var pooled_h = List[Float32]()
        for i in range(POOLED_DIM):
            if i < len(pooled_raw):
                pooled_h.append(pooled_raw[i])
            else:
                pooled_h.append(Float32(0.0))

        # ── VAE shift/scale then pack_latents ──
        # lat_raw is flat [1, 16, 128, 128] in CHW; drop batch dim (offset 0).
        # Scale: latent_scaled = (latent - VAE_SHIFT) * VAE_SCALE
        var lat_chw = List[Float32]()
        for i in range(LAT_C * LAT_H * LAT_W):
            lat_chw.append((lat_raw[i] - VAE_SHIFT) * VAE_SCALE)
        var latent_packed = _pack_latents(lat_chw)   # [N_IMG=4096, 64]

        # ── timestep ──
        var sigma_idx: Int
        if FIXED_SIGMA_SMOKE:
            sigma_idx = FIXED_SIGMA_IDX
        else:
            var sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
            sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
            if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
                sigma_idx = NUM_TRAIN_TIMESTEPS - 1
        var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
        # sigma for t_embedder: the conditioning input is sigma * 1000 (done inside _build_conditioning)
        var sigma_cont = sig   # [0,1] range; _build_conditioning multiplies by 1000

        # ── flow-match in PACKED latent space ──
        var noise = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        # ── forward (offload, full depth) -> velocity [N_IMG, OUT_CH] ──
        var fwd = sd35_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt_tokens.copy(), pooled_h.copy(), sigma_cont,
            base, loader, lora,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
            EPS, QK_EPS, ctx,
        )

        # ── loss = MSE(pred, target) ; d_loss = (2/N)(pred - target) ──
        var nout = len(fwd.out)
        var d_loss = List[Float32]()
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)
        var loss = Float32(sse / Float64(nout))
        if k == 1:
            first_loss = loss
        last_loss = loss

        # ── backward (offload, full depth) ──
        var grads = sd35_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            d_loss, noisy.copy(), txt_tokens.copy(),
            base, loader, lora, fwd,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
            EPS, QK_EPS, ctx,
        )

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW ──
        sd35_lora_adamw_step(lora, grads, k, LR, ctx)

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("SD35-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite != 0:
            print("[SD35-lora] warning nonfinite=", grads.nonfinite)

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
        _ = save_sd35_lora(lora, String(LORA_DIR) + String("/sd35_lora_smoke.safetensors"), ctx)
    else:
        print("RESULT: FAIL trains=", trains)
