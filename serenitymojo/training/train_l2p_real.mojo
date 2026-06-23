# training/train_l2p_real.mojo — Z-Image L2P (pixel-space) LoRA REAL training loop.
#
# FAITHFUL to ai-toolkit (authoritative reference) + EDv2 train_l2p.rs (the
# parity-verified Rust port). L2P = Z-Image-Turbo DiT body (reused VERBATIM) +
# 16×16 pixel-space patchify x_embedder + FROZEN MicroDiffusionModel U-Net head
# (the `local_decoder`). LoRA trains ONLY the 30 main DiT `layers` blocks.
#
# Reference (read FULL):
#   ai-toolkit  extensions_built_in/diffusion_models/z_image/z_image_l2p_model.py
#               (MicroDiffusionModel + L2P forward + FakeVAE pixel path)
#   ai-toolkit  toolkit/samplers/custom_flowmatch_sampler.py (add_noise, linear ts)
#   ai-toolkit  jobs/process/BaseSDTrainProcess.py (uniform timestep, noise, loss)
#   EDv2        crates/eridiffusion-cli/src/bin/train_l2p.rs (cross-ref recipe)
#
# RECIPE (ai-toolkit / EDv2, all four prior divergences FIXED here):
#   1. CACHE: reads {pixel [3,512,512] F32, cap_feats [1,seq,2560] F32} via
#      L2PCache (NOT the Klein {latent,text_embedding,text_mask} contract).
#      cap_feats seq VARIES per sample and is ALREADY trimmed to valid tokens —
#      there is NO text_mask; valid_cap := cap_feats.shape[1].
#   2. HEAD: runs the REAL FROZEN local_decoder (MicroDiffusionModel U-Net)
#      forward+backward (models/l2p/local_decoder_train.mojo). The DiT's last
#      image-token hidden [N_IMG, D] IS the feature map (NO final layer-norm /
#      modulate / linear — ai-toolkit has none). pred = local_decoder(noisy, feat).
#   3. TIMESTEP: UNIFORM UNSHIFTED — t_int = randint(0, NUM_TRAIN_TIMESTEPS)+1,
#      sigma = t_int / NUM_TRAIN_TIMESTEPS (ai-toolkit timestep_type='linear').
#      shift=3.0 is the INFERENCE sigma schedule only; it does NOT apply here.
#   4. LoRA: 30 main blocks, 7 Z-Image slots (to_q/to_k/to_v/to_out.0/w1/w3/w2).
#      PEFT save keys via save_zimage_lora_main_only.
#      NOTE (#4 partial): ai-toolkit/EDv2 also LoRA the per-block
#      adaLN_modulation.0 (8th target). The Mojo Z-Image LoRA infra is hardwired
#      to ZIMAGE_SLOTS=7 with no adaLN slot; adding it is a cross-cutting change
#      to the shared Z-Image LoRA stack (struct + fwd + bwd + AdamW + save) that
#      also touches the production zimage trainer. NOT done here — see the
#      BUILD REQUEST / DELIVERABLE notes. This trainer matches the 7-slot set.
#
# FLOW-MATCH (rectified):
#   noisy = (1 - sigma) * pixel + sigma * noise
#   target = noise - pixel                      (v-target in PIXEL space)
#   pred  = local_decoder(noisy, feat)          (returns -v_raw via DiT/decoder)
#   loss  = mean((pred - target)^2)             (F32)
#   We negate the DiT/decoder output (pred = -decoder_out) to match Python's
#   `model_fn_z_image` which returns -DiT(...); target stays noise - pixel.
#
# DTYPE:
#   * DiT base weights: bf16 (large) + f32 (norms) — mixed (as loaded).
#   * LoRA A/B masters/grads: F32.
#   * local_decoder convs: F32 (the conv/pool/silu backward kernels are F32-only).
#   * Pixels, noise, feat, loss: F32 host / device.
#
# COMPILE-ONLY GATE (orchestrator owns the compile):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo build -I . serenitymojo/training/train_l2p_real.mojo -o /tmp/train_l2p_real

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
    zimage_stack_lora_forward_main_device,
    zimage_stack_lora_backward_main_device_nofinal,
    zimage_lora_adamw_step_main_only, save_zimage_lora_main_only,
    save_zimage_lora_main_only_state, load_zimage_lora_main_only_state,
)
from serenitymojo.models.l2p.weights import (
    L2PRealAux, load_l2p_real_aux, load_l2p_block_weights_prefixed,
    build_l2p_adaln, build_l2p_block_modvecs, build_l2p_cap_seq,
    build_l2p_x_seq, build_l2p_rope, build_l2p_positions,
)
from serenitymojo.models.l2p.local_decoder_train import (
    L2PDecoderF32, l2p_decoder_f32_from_gate,
    l2p_decoder_forward, l2p_decoder_backward,
)
from serenitymojo.models.dit.zimage_l2p_local_decoder import ZImageL2PLocalDecoderGate
from serenitymojo.training.klein_dataset import L2PCache
from serenitymojo.training.progress_display import print_trainer_progress


# ── arch (Z-Image L2P; IDENTICAL body to Z-Image base) ───────────────────────
comptime H = 30
comptime Dh = 128
comptime D = H * Dh          # 3840
comptime F = 10240           # SwiGLU per-gate hidden
comptime CAP_DIM = 2560      # Qwen3 hidden
comptime ADALN_DIM = 256     # t_embedder output dim
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
comptime HT = PIX_H // PATCH   # 32  (feat grid H; also p4 grid after 4 pools)
comptime WT = PIX_W // PATCH   # 32
comptime N_IMG = HT * WT       # 1024 (1024 % 32 == 0, no padding needed)

# ── caption sequence: bucketed to CAP_LEN; valid rows from cap_feats.shape[1] ─
comptime CAP_LEN = 224

# ── unified sequence ──────────────────────────────────────────────────────────
comptime N_TXT = CAP_LEN
comptime S = N_IMG + N_TXT    # 1248

# ── depth (full L2P = 2 NR + 2 CR + 30 main; refiners excluded from LoRA) ────
comptime NUM_NR = 2
comptime NUM_CR = 2
comptime MAIN_DEPTH = 30

# ── recipe (ai-toolkit / EDv2 / l2p.json) ────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(3.0e-4)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

# ── paths ─────────────────────────────────────────────────────────────────────
comptime CHECKPOINT_PATH = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_l2p"

# Adapter slice: NR+CR blocks are allocated; only MAIN blocks are trained.
comptime TRAIN_ADAPTER_START = (NUM_NR + NUM_CR) * ZIMAGE_SLOTS
comptime N_ADAPTERS_TOTAL = (NUM_NR + NUM_CR + MAIN_DEPTH) * ZIMAGE_SLOTS


# ── host math helpers ─────────────────────────────────────────────────────────

def _host_noise_l2p(n: Int, seed: UInt64) -> List[Float32]:
    """Box-Muller PCG Gaussian noise N(0,1) — same LCG as zimage trainer."""
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


def _uniform_t_int(seed: UInt64, num_steps: Int) -> Int:
    """Uniform integer in [1, num_steps] (ai-toolkit/EDv2: randint(0,num)+1)."""
    var state = seed * 6364136223846793005 + 1442695040888963407
    var u = UInt64((state >> 11)) % UInt64(num_steps)
    return Int(u) + 1


def _absum_l2p[dt: DType](v: List[Scalar[dt]]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = Float32(v[i])
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


# ── feat map seam: [N_IMG, D] host (token-major, t = ih*WT+iw) <-> NCHW [1,D,HT,WT]
def _tokens_to_feat_nchw(
    x_final_host: List[Float32], ctx: DeviceContext
) raises -> Tensor:
    """Image rows [0,N_IMG) of x_final [S,D] -> feat map NCHW [1,D,HT,WT].
    Token order is row-major (ih,iw): t = ih*WT + iw, matching build_l2p_x_seq."""
    var feat = List[Float32]()
    for _ in range(D * N_IMG):
        feat.append(Float32(0.0))
    for ih in range(HT):
        for iw in range(WT):
            var t = ih * WT + iw
            for d in range(D):
                # NCHW flat: ((0*D + d)*HT + ih)*WT + iw
                feat[(d * HT + ih) * WT + iw] = x_final_host[t * D + d]
    return Tensor.from_host(feat^, [1, D, HT, WT], STDtype.F32, ctx)


def _feat_nchw_to_tokens(d_feat: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    """d_feat NCHW [1,D,HT,WT] -> d_x_full [S,D] (image rows filled, cap rows 0)."""
    var dh = d_feat.to_host(ctx)
    var out = List[Float32]()
    for _ in range(S * D):
        out.append(Float32(0.0))
    for ih in range(HT):
        for iw in range(WT):
            var t = ih * WT + iw
            for d in range(D):
                out[t * D + d] = dh[(d * HT + ih) * WT + iw]
    return out^


# ── per-step result ───────────────────────────────────────────────────────────
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
    cache: L2PCache,
    aux: L2PRealAux,
    dec: L2PDecoderF32,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    mut lora: ZImageLoraSet,
    train_start_ns: UInt,
    ctx: DeviceContext,
) raises -> L2PStepResult:
    var t0 = perf_counter_ns()

    # ── load cached sample: pixel [3,H,W] F32, cap_feats [1,seq,2560] F32 ──────
    var s = cache.load(slot, ctx)
    var psh = s.pixel.shape()
    if len(psh) != 3 or psh[0] != PIX_C or psh[1] != PIX_H or psh[2] != PIX_W:
        raise Error("train_l2p_real: pixel shape mismatch — expected [3,512,512]")
    var csh = s.cap_feats.shape()
    if len(csh) != 3 or csh[0] != 1 or csh[2] != CAP_DIM:
        raise Error("train_l2p_real: cap_feats shape mismatch — expected [1,seq,2560]")
    var valid_cap = csh[1]      # cap_feats already trimmed to valid tokens; NO mask
    if valid_cap <= 0 or valid_cap > CAP_LEN:
        raise Error("train_l2p_real: caption length out of range")

    var pix_h = cast_tensor(s.pixel, STDtype.F32, ctx).to_host(ctx)  # [3,512,512] flat

    # ── timestep: UNIFORM UNSHIFTED (ai-toolkit timestep_type='linear') ───────
    var t_int = _uniform_t_int(SEED_BASE + step_seed, NUM_TRAIN_TIMESTEPS)
    var sigma = Float32(t_int) / Float32(NUM_TRAIN_TIMESTEPS)
    # DiT timestep input: v_in = (1 - sigma). build_l2p_adaln does t_val*T_SCALE
    # with NO internal inversion, and the verified inference contract is
    # zimage_l2p_model_timestep(sigma) = (1-sigma)*1000 (zimage_l2p_contract.mojo:105;
    # ai-toolkit (1000-timestep)/1000; EDv2 dit.rs t=(1-v)*time_scale).
    var t_value = Float32(1.0) - sigma

    # ── pixel noise + noisy pixels (rectified flow) ──────────────────────────
    var noise_pix = _host_noise_l2p(PIX_C * PIX_H * PIX_W, SEED_BASE * UInt64(7919) + step_seed)
    var noisy_pix_h = List[Float32]()
    for i in range(len(pix_h)):
        noisy_pix_h.append(pix_h[i] * (Float32(1.0) - sigma) + noise_pix[i] * sigma)
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

    # ── cap_seq from cap_feats (valid_cap rows, pad rest with cap_pad_token) ──
    var cap_feats = cast_tensor(s.cap_feats, STDtype.F32, ctx)   # [1,seq,2560]
    var cap_full = cap_feats.to_host(ctx)
    var cap_vals = List[Float32]()
    for r in range(CAP_LEN):
        var src_r = r if r < valid_cap else valid_cap - 1
        for c in range(CAP_DIM):
            cap_vals.append(cap_full[src_r * CAP_DIM + c])
    var cap2 = Tensor.from_host(cap_vals^, [CAP_LEN, CAP_DIM], STDtype.F32, ctx)
    var cap_seq = build_l2p_cap_seq(aux, cap2, EPS, ctx)
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
    var ur = build_l2p_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos = ur[0].copy(); var uni_sin = ur[1].copy()
    var crr = build_l2p_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos = crr[0].copy(); var cap_sin = crr[1].copy()

    var t_prep = perf_counter_ns()

    # ── DiT stack forward (last-block hidden = the feature map; NO final layer)
    # We reuse the proven stack forward. ai-toolkit L2P has no final layer-norm/
    # modulate/linear; we pass an IDENTITY-shaped final layer purely so the
    # existing forward runs, but we IGNORE fwd.out and use saved.x_final (the
    # last main-block output) as the feature source. f_scale=zeros, out_ch=D,
    # final_lin_w=identity[D,D], final_lin_b=zeros — these only affect fwd.out,
    # which we discard. The backward we call (nofinal) ignores them entirely.
    var f_scale_zeros = List[Float32]()
    for _ in range(D):
        f_scale_zeros.append(Float32(0.0))
    var ident_host = List[Float32]()
    for _ in range(D * D):
        ident_host.append(Float32(0.0))
    for d in range(D):
        ident_host[d * D + d] = Float32(1.0)
    var ident_w = Tensor.from_host(ident_host^, [D, D], STDtype.F32, ctx)
    var zero_b_host = List[Float32]()
    for _ in range(D):
        zero_b_host.append(Float32(0.0))
    var zero_b = Tensor.from_host(zero_b_host^, [D], STDtype.F32, ctx)

    var lora_dev = zimage_lora_set_to_device(lora, ctx)
    var t_lora = perf_counter_ns()

    var fwd = zimage_stack_lora_forward_main_device[H, Dh, N_IMG, N_TXT, S](
        x_t_host.copy(), cap_seq.copy(),
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora_dev,
        f_scale_zeros.copy(),
        ident_w, zero_b,
        x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
        D, F, D, EPS, FINAL_EPS, ctx,
    )
    var t_fwd = perf_counter_ns()

    # ── feature map [1, D, HT, WT] from the last-block image-token hidden ─────
    var x_final_host = fwd.x_final[].to_host(ctx)   # [S, D]
    var feat_nchw = _tokens_to_feat_nchw(x_final_host, ctx)

    # ── REAL local_decoder forward (FROZEN): pred [1,3,512,512] ───────────────
    var dec_fwd = l2p_decoder_forward[PIX_H, PIX_W, HT, WT](
        dec, noisy_pixel_t, feat_nchw, ctx
    )
    var pred_h = dec_fwd.pred_nchw.to_host(ctx)     # [3,512,512] flat
    var t_dec = perf_counter_ns()

    # ── loss: target = noise - pixel ; pred = -decoder_out ; mean MSE (F32) ──
    var npix = PIX_C * PIX_H * PIX_W
    var d_pred_h = List[Float32]()
    for _ in range(npix):
        d_pred_h.append(Float32(0.0))
    var sse = 0.0
    var inv_n = Float32(2.0) / Float32(npix)
    for i in range(npix):
        var pred = -pred_h[i]
        var target = noise_pix[i] - pix_h[i]
        var diff = pred - target
        sse += Float64(diff) * Float64(diff)
        # dL/d(decoder_out) = dL/dpred * dpred/d(decoder_out) = (2/N)*diff * (-1)
        d_pred_h[i] = -inv_n * diff
    var loss = Float32(sse / Float64(npix))
    var d_pred_t = Tensor.from_host(d_pred_h^, [1, PIX_C, PIX_H, PIX_W], STDtype.F32, ctx)
    var t_loss = perf_counter_ns()

    # ── local_decoder backward (FROZEN): d_pred -> d_feat [1,D,HT,WT] ─────────
    var d_feat = l2p_decoder_backward[PIX_H, PIX_W, HT, WT](
        dec, dec_fwd.acts, d_pred_t, ctx
    )
    var d_x_full = _feat_nchw_to_tokens(d_feat, ctx)   # [S,D], image rows filled
    var t_dbwd = perf_counter_ns()

    # ── DiT stack backward (no final layer): d_x_full -> LoRA grads ───────────
    var grads = zimage_stack_lora_backward_main_device_nofinal[H, Dh, N_IMG, N_TXT, S](
        d_x_full, main_blocks, main_mod, lora_dev,
        uni_cos[], uni_sin[], fwd,
        D, F, EPS, ctx,
    )
    var t_bwd = perf_counter_ns()

    # ── clip + optimize (main adapters only) ──────────────────────────────────
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
          " dec=", Float32(Float64(t_dec - t_fwd) / 1.0e9),
          " loss=", Float32(Float64(t_loss - t_dec) / 1.0e9),
          " dbwd=", Float32(Float64(t_dbwd - t_loss) / 1.0e9),
          " bwd=", Float32(Float64(t_bwd - t_dbwd) / 1.0e9),
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

    print("=== Z-Image L2P REAL LoRA training loop (ai-toolkit faithful) ===")
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " F=", F)
    print("  pixel input: C=", PIX_C, " H=", PIX_H, " W=", PIX_W,
          " patch=", PATCH, " feat grid=", HT, "x", WT)
    print("  depth: NR=", NUM_NR, " CR=", NUM_CR, " MAIN=", MAIN_DEPTH)
    print("  bucket: 512x512 -> N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", RANK, " alpha=", ALPHA, " lr=", LR,
          " timestep=UNIFORM UNSHIFTED")
    print("  checkpoint:", CHECKPOINT_PATH)
    print("  cache:", CACHE_DIR)
    print("  head: REAL FROZEN local_decoder (MicroDiffusionModel U-Net) fwd+bwd")

    # ── cache first: fail before loading the ~19 GB checkpoint ───────────────
    var cache = L2PCache(String(CACHE_DIR))
    print("[cache] samples:", cache.count())
    var k0 = cache.peek_key(0, ctx)
    print("[cache] first entry: C=", k0.c, " H=", k0.h, " W=", k0.w, " cap_seq=", k0.seq)
    if k0.c != PIX_C or k0.h != PIX_H or k0.w != PIX_W:
        raise Error("train_l2p_real: cache pixel shape mismatch — expected [3,512,512]")

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

    # ── FROZEN local_decoder (load BF16 gate, cast convs to F32 once) ─────────
    print("[load] local_decoder (MicroDiffusionModel U-Net, FROZEN)")
    var dec_gate = ZImageL2PLocalDecoderGate.load(String(CHECKPOINT_PATH), ctx)
    var dec = l2p_decoder_f32_from_gate(dec_gate, ctx)

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
            k, run_steps, slot, step_seed, cache, aux, dec,
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
