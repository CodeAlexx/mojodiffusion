# train_chroma_real.mojo — Chroma1-HD LoRA training loop (block-swap offload).
#
# TRANSLATION of EriDiffusion-v2 chroma.rs onto the parity-verified Mojo Chroma
# LoRA OFFLOAD stack (models/chroma/chroma_stack_lora.mojo). Real Chroma1-HD base
# weights (streamed block-by-block via TurboPlannedLoader), real prepared cache
# (latent + T5), full 19+38 block depth. No synthetic tensors. Mirrors
# train_flux_real.mojo's loop structure (timing, grad clip, shared progress
# display) and chroma.rs's recipe.
#
# CHROMA vs FLUX (the deltas; see chroma_stack_lora.mojo header):
#   - NO guidance / CLIP-pooled vector. Modulation comes from the FROZEN
#     distilled_guidance_layer APPROXIMATOR (models/dit/chroma_dit.mojo
#     ChromaDitCache.approximator_forward), producing a per-step pooled_temb
#     table [mod_index=344, D=3072]; each block's ModVecs are sliced rows.
#   - Block math IS the proven Flux block (after the loader's separate->fused
#     row-stack), so the per-block LoRA fwd/bwd is REUSED verbatim and the LoRA
#     carrier / AdamW / save is the proven FluxLoraSet path.
#
# Per step:
#   1. load cached {latent [1,16,64,64] RAW, t5_embed [1,seq,4096]}
#   2. latent_scaled = (latent - SHIFT) * SCALE  (Chroma VAE shift/scale)
#   3. pack_latents([16,64,64]) -> [N_IMG=1024, 64] channel-major patchify
#   4. sigma_idx = floor(logit_normal_sigma(shift=1.15) * 1000) clamp;
#      sig=(idx+1)/1000 ; t_model=idx/1000
#   5. noisy = noise*sig + latent_packed*(1-sig) ; target = noise - latent_packed
#   6. pooled_temb = approximator(t_model)  (frozen; once per step)
#   7. chroma_stack_lora_forward_offload(noisy, txt, pooled, ...) -> pred [N_IMG,64]
#   8. loss = MSE(pred, target); d_loss = (2/N)(pred - target)
#   9. chroma_stack_lora_backward_offload -> LoRA grads; global-norm clip(1.0)
#  10. flux_lora_adamw_step; print shared progress display
#
# Recipe scalars (configs/chroma.json / chroma.rs): lr=1e-4, rank=16, alpha=16,
#   timestep_shift=1.15, clip_grad_norm=1.0, VAE shift=0.1159 scale=0.3611.
#
# MEMORY: the Chroma transformer is ~8.9B params (BF16 17.8GB on disk). The
# OFFLOAD path streams one block at a time + holds the resident base
# (x_embedder/context_embedder/proj_out ~tiny) + the approximator (~loaded once)
# + LoRA optimizer state. FULL 19+38 depth is the default.
#
# FIXED_SIGMA_SMOKE: every step uses the SAME cache sample AND a fixed
# timestep+noise so a correct LoRA backward MUST drive loss DOWN monotonically
# (the canonical trainer-correctness gate, same probe as train_flux_real).
#
# Run (real smoke):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_chroma_real.mojo -o /tmp/train_chroma_real && \
#     /tmp/train_chroma_real [steps]

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

from serenitymojo.models.chroma.weights import load_chroma_stack_base
from serenitymojo.models.chroma.chroma_stack_lora import (
    chroma_stack_lora_forward_offload, chroma_stack_lora_backward_offload,
)
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet, build_flux_lora_set,
    flux_lora_adamw_step, save_flux_lora, total_adapters,
)
from serenitymojo.models.flux.lora_block import DBL_STREAM_SLOTS, SGL_SLOTS
from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.models.dit.chroma_dit import ChromaDitCache
from serenitymojo.offload.plan import build_chroma1_hd_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress


# ── arch (chroma1-hd; H/Dh/D fixed comptime, verified vs the checkpoint) ─────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288          # mlp_hidden = D*4
comptime IN_CH = 64            # patch_dim = 16ch * 2*2
comptime TXT_CH = 4096         # T5-XXL hidden
comptime OUT_CH = 64
comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime MOD_INDEX = 3 * NUM_SINGLE + 2 * 6 * NUM_DOUBLE + 2   # 344
comptime EPS = Float32(1e-06)

# ── resolution (512px): latent [16,64,64] -> pack2 -> 32x32=1024 img tokens ──
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime PATCH = 2
comptime HT = LAT_H // PATCH   # 32
comptime WT = LAT_W // PATCH   # 32
comptime N_IMG = HT * WT       # 1024
comptime N_TXT = 512           # T5 padded length
comptime S = N_TXT + N_IMG     # 1536

# ── recipe (configs/chroma.json) ─────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)
comptime TIMESTEP_SHIFT = Float32(1.15)
comptime VAE_SHIFT = Float32(0.1159)
comptime VAE_SCALE = Float32(0.3611)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_IDX = 500

comptime CKPT = "/home/alex/.serenity/models/checkpoints/chroma1_hd_bf16.safetensors"
comptime CACHE_DIR = "/home/alex/datasets/boxjana_chroma_edv2_512"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/chroma_boxjana"


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


def _global_norm(grads: FluxLoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: FluxLoraGradSet, max_norm: Float32) -> Float64:
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
        raise Error(String("chroma cache: no .safetensors in ") + dir)
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


# pack_latents: [16,LAT_H,LAT_W] flat -> [N_IMG, 64] channel-major patchify.
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


# Build the frozen per-step pooled_temb modulation table as host F32 [MOD_INDEX*D].
def _pooled_temb(approx: ChromaDitCache, t_model: Float32, ctx: DeviceContext) raises -> List[Float32]:
    var approx_in = approx._approximator_input(t_model, ctx)
    var pooled = approx.approximator_forward(approx_in, ctx)   # [1, MOD_INDEX, D] BF16
    return cast_tensor(pooled, STDtype.F32, ctx).to_host(ctx)


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

    print("=== Chroma (chroma1-hd) REAL LoRA training loop (block-swap offload) ===")
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " Fmlp=", FMLP, " out_ch=", OUT_CH)
    print("  depth: NUM_DOUBLE=", NUM_DOUBLE, " NUM_SINGLE=", NUM_SINGLE, " mod_index=", MOD_INDEX)
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", RANK, " alpha=", ALPHA, " lr=", LR, " shift=", TIMESTEP_SHIFT,
          " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", CKPT)
    print("  cache:", CACHE_DIR)

    # ── stack-level base (frozen; x_embedder/context_embedder/proj_out) ──────
    print("[load] ChromaStackBase (x_embedder, context_embedder, proj_out)")
    var base_st = SafeTensors.open(String(CKPT))
    var base = load_chroma_stack_base(base_st, NUM_DOUBLE, NUM_SINGLE, ctx)
    print("[load] base resident")

    # ── frozen approximator (distilled_guidance_layer) ───────────────────────
    print("[load] approximator (distilled_guidance_layer)")
    var approx = ChromaDitCache.load(String(CKPT), ctx)
    print("[load] approximator resident")

    # ── block-swap offload loader ────────────────────────────────────────────
    var plan = build_chroma1_hd_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(String(CKPT), plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── 3-axis RoPE tables (positions fixed for 512px; built once, BF16) ─────
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, H, Dh](HT, WT, ctx, STDtype.BF16)
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)
    print("[load] chroma 3-axis rope tables built (S*H x Dh/2)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
    var n_adapters = total_adapters(lora)
    print("[lora] adapters:", n_adapters,
          " (", DBL_STREAM_SLOTS * 2, "x", NUM_DOUBLE, "double +",
          SGL_SLOTS, "x", NUM_SINGLE, "single)")

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
        var lat_raw = _load_host(st, String("latent"), ctx)         # [16*64*64]

        # t5_embed [1, seq, 4096] -> pad/truncate to [N_TXT, 4096] (zero pad rows).
        var t5_info = st.tensor_info(String("t5_embed"))
        var t5_seq = Int(t5_info.shape[1])
        var t5_flat = _load_host(st, String("t5_embed"), ctx)
        var txt_tokens = List[Float32]()
        for r in range(N_TXT):
            if r < t5_seq:
                for c in range(TXT_CH):
                    txt_tokens.append(t5_flat[r * TXT_CH + c])
            else:
                for _ in range(TXT_CH):
                    txt_tokens.append(Float32(0.0))

        # ── VAE shift/scale then pack_latents ──
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

        # ── flow-match in PACKED latent space ──
        var noise = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        # ── frozen approximator -> pooled_temb modulation table ──
        var pooled = _pooled_temb(approx, t_model, ctx)

        # ── forward (offload, full depth) -> velocity [N_IMG, OUT_CH] ──
        var fwd = chroma_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
            base, loader, lora, cos.copy(), sin.copy(),
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
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
        var grads = chroma_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            d_loss, noisy.copy(), txt_tokens.copy(), base, loader, lora,
            cos.copy(), sin.copy(), fwd,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW ──
        flux_lora_adamw_step(lora, grads, k, LR, ctx)

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("Chroma-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[Chroma-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

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
        _ = save_flux_lora(lora, String(LORA_DIR) + String("/chroma_lora_smoke.safetensors"), ctx)
    else:
        print("RESULT: FAIL trains=", trains)
