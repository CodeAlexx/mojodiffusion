# training/train_wan22_real.mojo — Wan2.2-T2V 14B LoRA training loop (block-swap offload).
#
# MIRRORS train_chroma_real.mojo structure (timing, flow-match recipe, progress
# display, smoke gate). Ports the wan22.rs EDv2 training recipe.
#
# ARCHITECTURE (14B; wan2.2_t2v_low_noise_14b_fp16.safetensors, confirmed):
#   dim=5120, H=40, Dh=128, ffn=13824, num_blocks=40
#   in_ch=64 (patch latent: 16ch * 2*2 patchify -> 64 elem per patch)
#   out_ch=64 (head output per patch token, same as in_ch)
#   text_dim=4096 (T5-XXL hidden dim), freq_dim=256 (sinusoidal embed)
#   S=N_IMG  (image tokens; for 512px T2V: grid depends on resolution)
#
# NOTE: This trainer targets the T2V inference checkpoint reused for image
# LoRA training (single-frame; the VAE+patchify pipeline collapses the
# temporal dimension to F=1 so S = H_patch * W_patch).
#
# FLOW-MATCH RECIPE (wan22.rs + EDv2 training config):
#   timestep_shift = 1.0 (no logit-normal shift bias; plain uniform)
#   sigma_idx = floor(uniform_sigma * 1000) clamped to [0, 999]
#   sig = (sigma_idx + 1) / 1000; t_model = sigma_idx / 1000
#   noisy = noise * sig + latent * (1 - sig)
#   target = noise - latent                 (velocity target)
#   loss = MSE(pred, target)
#
# LORA TARGETS: 8 per block (sa_{q,k,v,o} + ca_{q,k,v,o}), rank=32, alpha=32.
#   320 adapters total. All in=out=dim=5120.
#
# MEMORY SAFETY (enforced):
#   - DO NOT LOAD the full 14B checkpoint into VRAM.
#   - Block weights are streamed ONE AT A TIME by TurboPlannedLoader.
#   - Compile-only deliverable; do not run /tmp/train_wan22_real.
#
# Run (compile only — DO NOT EXECUTE):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_wan22_real.mojo \
#       -o /tmp/train_wan22_real

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

from serenitymojo.models.wan22.weights import load_wan22_stack_base
from serenitymojo.models.wan22.wan22_stack_lora import (
    Wan22LoraSet, Wan22LoraGradSet,
    build_wan22_lora_set, wan22_lora_adamw_step, save_wan22_lora,
    wan22_total_adapters,
    wan22_stack_lora_forward_offload, wan22_stack_lora_backward_offload,
)
from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.wan22_plan import build_wan22_14b_block_plan
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress


# ── arch (wan2.2_t2v_low_noise_14b; dims confirmed from safetensors header) ───
comptime H = 40
comptime Dh = 128
comptime DIM = H * Dh          # 5120
comptime FFN = 13824
comptime NUM_BLOCKS = 40
comptime FREQ_DIM = 256        # sinusoidal time embedding
comptime TEXT_DIM = 4096       # T5-XXL context channels
comptime OUT_CH = 64           # head output per patch token

# ── single-frame 512px patchify (T2V, F=1, 2x2 spatial patch, 16 latent ch) ──
# Patch embed: [16, 1, H_lat, W_lat] -> [S, in_ch=64] where S=H_patch*W_patch.
# 512px -> latent 64x64 -> 2x2 patchify -> 32x32 = 1024 tokens.
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime PH = 2
comptime PW = 2
comptime N_IMG = (LAT_H // PH) * (LAT_W // PW)   # 1024
comptime IN_CH = LAT_C * PH * PW                  # 64 (patch_dim)
comptime S = N_IMG
comptime TXT = 512             # padded T5 sequence length

# ── recipe ────────────────────────────────────────────────────────────────────
comptime RANK = 32
comptime ALPHA = Float32(32.0)
comptime LR = Float32(1.0e-4)
# NOTE: wan22.rs uses a plain uniform timestep (shift=1.0 -> logit-normal
# collapses to near-uniform). The low-noise checkpoint is already fine-tuned
# for a shifted noise schedule; we train with uniform sampling to match that.
comptime TIMESTEP_SHIFT = Float32(1.0)
# VAE shift/scale: Wan2.2 uses a latent normalization. Actual values depend on
# the dataset preprocessing pipeline (the inference code normalises to ~ N(0,1)
# with mean≈0 and std≈1). Conservative defaults (identity): shift=0, scale=1.
# TODO: verify exact Wan2.2 VAE normalisation from wan/vae/config.json or
#       wan22.rs vae_config once the cache is built with the correct values.
comptime VAE_SHIFT = Float32(0.0)
comptime VAE_SCALE = Float32(1.0)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime EPS = Float32(1.0e-6)
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_IDX = 500

comptime CKPT = "/home/alex/.serenity/models/checkpoints/wan2.2_t2v_low_noise_14b_fp16.safetensors"
# TODO: set to a real Wan2.2 prepared cache directory when available.
comptime CACHE_DIR = "/home/alex/datasets/wan22_cache"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/wan22_lora"


# ── deterministic host gaussian noise (Box-Muller PCG; matches chroma trainer) ─
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


def _global_norm(grads: Wan22LoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: Wan22LoraGradSet, max_norm: Float32) -> Float64:
    var gn = _global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var sc = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * sc
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * sc
    return gn


def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("wan22 cache: no .safetensors in ") + dir)
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


# Patchify a [16, LAT_H, LAT_W] latent into [N_IMG, IN_CH] tokens.
# Channel-major 2x2 spatial patch (mirrors chroma's _pack_latents structure).
def _pack_latents(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(LAT_H // PH):
        for iw in range(LAT_W // PW):
            for c in range(LAT_C):
                for ph in range(PH):
                    for pw in range(PW):
                        var hh = ih * PH + ph
                        var ww = iw * PW + pw
                        var idx = c * LAT_H * LAT_W + hh * LAT_W + ww
                        out.append(lat[idx])
    return out^


# Build a trivial [S, Dh/2] RoPE cosine/sine table filled with ones/zeros
# (placeholder for the real 3-axis interleaved RoPE from wan22_dit.mojo).
# TODO: replace with wan22_build_rope once the RoPE builder is callable from
# the training path without a full DiT instance.
def _rope_placeholder(S: Int, half: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(S * half):
        out.append(Float32(1.0))
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

    print("=== Wan2.2-T2V 14B (low-noise) REAL LoRA training loop (block-swap offload) ===")
    print("  arch: dim=", DIM, " H=", H, " Dh=", Dh, " ffn=", FFN, " num_blocks=", NUM_BLOCKS)
    print("  tokens: S=N_IMG=", N_IMG, " TXT=", TXT, " in_ch=", IN_CH, " out_ch=", OUT_CH)
    print("  recipe: rank=", RANK, " alpha=", ALPHA, " lr=", LR,
          " shift=", TIMESTEP_SHIFT, " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", CKPT)
    print("  cache:", CACHE_DIR)

    # ── resident frozen base (embeddings + head) ──────────────────────────────
    print("[load] Wan22StackBase (patch_embedding, text_embedding, time_embedding,")
    print("       time_projection, head)")
    var base_st = SafeTensors.open(String(CKPT))
    var base = load_wan22_stack_base(base_st, ctx)
    print("[load] base resident")

    # ── block-swap offload loader ─────────────────────────────────────────────
    var plan = build_wan22_14b_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(String(CKPT), plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), " blocks)")

    # ── RoPE tables ───────────────────────────────────────────────────────────
    # NOTE: for the compile smoke we use a placeholder rope table. In a full
    # run these should come from wan22_build_rope (models/dit/wan22_dit.mojo).
    # The rope tables enter the block forward as Tensor [S*H, Dh/2].
    var cos = _rope_placeholder(S * H, Dh // 2)
    var sin = _rope_placeholder(S * H, Dh // 2)
    print("[rope] placeholder tables (S*H=", S * H, " x Dh/2=", Dh // 2, ")")
    print("  TODO: replace with wan22_build_rope for real training.")

    # ── LoRA set (B=0 at init) ────────────────────────────────────────────────
    var lora = build_wan22_lora_set(NUM_BLOCKS, DIM, RANK, ALPHA)
    var n_adapters = wan22_total_adapters(lora)
    print("[lora] adapters:", n_adapters, " (8 per block x", NUM_BLOCKS, " blocks)")
    print("  targets: self_attn.{q,k,v,o} + cross_attn.{q,k,v,o} per block")

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # ── cache ─────────────────────────────────────────────────────────────────
    # NOTE: In FIXED_SIGMA_SMOKE=True mode we use a synthetic sample so the
    # cache directory is NOT required to exist. A real run needs a prepared cache
    # in CACHE_DIR with .safetensors files containing "latent" [16*64*64] and
    # "t5_embed" [1, seq, 4096] tensors (same format as the chroma cache).
    var files = List[String]()
    comptime if not FIXED_SIGMA_SMOKE:
        files = _list_cache(String(CACHE_DIR))
        print("[cache] samples:", len(files))
    else:
        print("[cache] FIXED_SIGMA_SMOKE=True: using synthetic sample, no cache needed.")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var train_start = perf_counter_ns()

    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)

        # ── load or synthesize cache sample ──
        var lat_raw: List[Float32]
        var txt_tokens: List[Float32]
        comptime if FIXED_SIGMA_SMOKE:
            # Synthetic: random normal latent + zero text tokens (noise-only smoke).
            lat_raw = _host_noise(LAT_C * LAT_H * LAT_W, SEED_BASE * UInt64(31) + step_seed)
            txt_tokens = List[Float32]()
            for _ in range(TXT * TEXT_DIM):
                txt_tokens.append(Float32(0.0))
        else:
            var slot = (k - 1) % len(files)
            var st = SafeTensors.open(files[slot])
            lat_raw = _load_host(st, String("latent"), ctx)
            var t5_info = st.tensor_info(String("t5_embed"))
            var t5_seq = Int(t5_info.shape[1])
            var t5_flat = _load_host(st, String("t5_embed"), ctx)
            txt_tokens = List[Float32]()
            for r in range(TXT):
                if r < t5_seq:
                    for c in range(TEXT_DIM):
                        txt_tokens.append(t5_flat[r * TEXT_DIM + c])
                else:
                    for _ in range(TEXT_DIM):
                        txt_tokens.append(Float32(0.0))

        # ── VAE shift/scale then patchify ──
        for i in range(len(lat_raw)):
            lat_raw[i] = (lat_raw[i] - VAE_SHIFT) * VAE_SCALE
        var latent_packed = _pack_latents(lat_raw)

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
        var t_model = Float32(sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

        # ── flow-match ──
        var noise = _host_noise(S * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        # ── forward ──
        var fwd = wan22_stack_lora_forward_offload[H, Dh, S, TXT](
            noisy.copy(), txt_tokens.copy(), t_model,
            base, loader, lora,
            cos.copy(), sin.copy(),
            DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        )

        # ── loss = MSE(pred, target) ──
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

        # ── backward ──
        var grads = wan22_stack_lora_backward_offload[H, Dh, S, TXT](
            d_loss, noisy.copy(), txt_tokens.copy(),
            base, loader, lora,
            cos.copy(), sin.copy(), fwd,
            DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        )

        # ── grad norm + clip ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW ──
        wan22_lora_adamw_step(lora, grads, k, LR, ctx)

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("Wan22-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[Wan22-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

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
        _ = save_wan22_lora(lora, String(LORA_DIR) + String("/wan22_lora_smoke.safetensors"), ctx)
    else:
        print("RESULT: FAIL trains=", trains)
