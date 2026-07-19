# serenitymojo/models/l2p/parity/l2p_batch2_parity.mojo
#
# TRUE BATCH-2 PARITY GATE for the L2P (Z-Image local-decoder) host-grad
# plain-LoRA arm. Loads TWO REAL cache samples (boxjana_l2p_512, 512px, varying
# caption length) and runs the trainer's b2 step logic inline: the PROVEN zimage
# masked B=2 stack (row-stacked [2S,D], per-sample tail mask) + the FROZEN
# local_decoder run TWICE + joint 2N-mean loss + the new masked B=2 nofinal DiT
# backward. Then asserts the two batch-2 invariants:
#
#   (a) loss_B2 == mean(loss_ref(s0), loss_ref(s1))   within 1e-3 relative.  BINDING
#   (b) per-sample decoder pred: cos(b2_pred_i, ref_pred_i) >= 0.999.         BINDING
#   (c) grad cosine B2 vs mean(B1) — INFORMATIONAL (MJ-1073: row-stacked bf16
#       GEMM tiling is M-shape-deterministic; grad-cosine-vs-b1 at depth is the
#       WRONG instrument for a batched trainer with reshaped GEMMs).
#
# MASKED REFERENCE (why not the unmasked b1 trainer forward): boxjana captions
# are 9-10 tokens with CAP_LEN=224, so the b1 trainer's UNMASKED forward attends
# ~214 identical pad rows while the masked B=2 stack masks them. Comparing masked
# b2 to unmasked b1 would spuriously fail the binding bars. The per-sample MASKED
# reference here is the masked B=2 stack run on a DUPLICATED sample (s_i, s_i) and
# split to half-0 — the exact per-sample computation b2 stacks, so (a)+(b) are the
# true batch-semantics invariants. The genuine UNMASKED b1 (the named trainer
# functions) is ALSO run, but only for the INFORMATIONAL grad cosine (c) and an
# unmasked-vs-masked loss-gap print. Self-consistent; no torch oracle.
#
# Build (mem-safe -O2; cuDNN flash shim linked) + run ONE GPU process:
#   cd /home/alex/mojodiffusion
#   MEM_MAX=28G MEM_HIGH=24G SWAP_MAX=2G bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 1 -I . -I /home/alex/MOJO-libs \
#       -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#       serenitymojo/models/l2p/parity/l2p_batch2_parity.mojo \
#       -o output/bin/l2p_batch2_parity
#   output/bin/l2p_batch2_parity

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.zimage_stack import ZImageStackForward
from serenitymojo.models.zimage.lora_block import (
    ZIMAGE_SLOTS, ZImageModVecsDevice, zimage_modvecs_pack2_to_device,
)
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraSet, ZImageLoraGrads, ZImageLoraDeviceSet, ZImageStackForwardB2,
    build_zimage_lora_set, zimage_lora_set_to_device,
    zimage_stack_lora_forward_main_device,
    zimage_stack_lora_backward_main_device_nofinal,
    zimage_stack_lora_forward_main_device_b2_masked,
    zimage_stack_lora_backward_main_device_b2_masked_nofinal,
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


comptime TArc = ArcPointer[Tensor]

# ── arch (identical to train_l2p_real.mojo) ──────────────────────────────────
comptime H = 30
comptime Dh = 128
comptime D = H * Dh
comptime F = 10240
comptime CAP_DIM = 2560
comptime T_SCALE = Float32(1000.0)
comptime ROPE_THETA = Float32(256.0)
comptime AXIS0 = 32
comptime AXIS1 = 48
comptime AXIS2 = 48
comptime EPS = Float32(1e-5)
comptime FINAL_EPS = Float32(1e-6)

comptime PIX_C = 3
comptime PATCH = 16
comptime PIX_H = 512
comptime PIX_W = 512
comptime HT = PIX_H // PATCH
comptime WT = PIX_W // PATCH
comptime N_IMG = HT * WT       # 1024
comptime CAP_LEN = 224
comptime N_TXT = CAP_LEN
comptime S = N_IMG + N_TXT     # 1248

comptime NUM_NR = 2
comptime NUM_CR = 2
# Parity is DEPTH-INDEPENDENT (the b2-vs-per-sample-ref invariant holds per block,
# klein precedent at 4+8), so the gate runs a REDUCED main depth. The full 30-block
# stack does not fit 24GB in a b2 pass: base(34 blocks) + the row-stacked [2S=2496,D]
# tape x depth + two frozen-decoder tapes + the masked-b2 nofinal backward's per-block
# recompute peaks over 24GB (the passes already return host-only numbers so device
# tapes drop BETWEEN passes — this is a single-pass peak, not cross-pass accumulation).
# The gate loads/forwards layers.0..7 only; every count keys off MAIN_DEPTH.
comptime MAIN_DEPTH = 8
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime NPIX = PIX_C * PIX_H * PIX_W

comptime TRAIN_ADAPTER_START = (NUM_NR + NUM_CR) * ZIMAGE_SLOTS
comptime N_ADAPTERS_TOTAL = (NUM_NR + NUM_CR + MAIN_DEPTH) * ZIMAGE_SLOTS

comptime CHECKPOINT_PATH = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512"
comptime COS_BAR = Float64(0.999)
comptime LOSS_REL_BAR = Float64(1.0e-3)


# ── host math helpers ─────────────────────────────────────────────────────────
def _host_noise_l2p(n: Int, seed: UInt64) -> List[Float32]:
    """Box-Muller PCG Gaussian noise N(0,1) — same LCG as the trainer."""
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


def _absf(x: Float64) -> Float64:
    return x if x >= Float64(0.0) else -x


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b) or n == 0:
        return Float64(-2.0)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(n):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na == Float64(0.0) and nb == Float64(0.0):
        return Float64(1.0)
    if na == Float64(0.0) or nb == Float64(0.0):
        return Float64(0.0)
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs_diff(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b):
        return Float64(1.0e9)
    var m = Float64(0.0)
    for i in range(n):
        var d = _absf(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


def _mean2(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(Float32(0.5) * (a[i] + b[i]))
    return out^


def _sumsq(a: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(a)):
        var v = Float64(a[i])
        s += v * v
    return s


# Concatenate the TRAINED main adapters' grads (dA then dB, slot order) into one
# host vector for magnitude-weighted global cosine.
def _concat_trained_grads(grads: ZImageLoraGrads) -> List[Float32]:
    var out = List[Float32]()
    for i in range(TRAIN_ADAPTER_START, N_ADAPTERS_TOTAL):
        for j in range(len(grads.d_a[i])):
            out.append(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            out.append(grads.d_b[i][j])
    return out^


# ── feat map seam (copied verbatim from train_l2p_real.mojo:348-378) ──────────
def _tokens_to_feat_nchw(
    x_final_host: List[Float32], ctx: DeviceContext
) raises -> Tensor:
    var feat = List[Float32]()
    for _ in range(D * N_IMG):
        feat.append(Float32(0.0))
    for ih in range(HT):
        for iw in range(WT):
            var t = ih * WT + iw
            for d in range(D):
                feat[(d * HT + ih) * WT + iw] = x_final_host[t * D + d]
    return Tensor.from_host(feat^, [1, D, HT, WT], STDtype.F32, ctx)


def _feat_nchw_to_tokens(d_feat: Tensor, ctx: DeviceContext) raises -> List[Float32]:
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


# ── per-sample conditioning (built once per sample, reused across runs) ───────
@fieldwise_init
struct _L2PCond(Movable):
    var x_t: List[Float32]                 # [N_IMG, D] host
    var cap_seq: List[Float32]             # [CAP_LEN, D] host
    var nr_mod: List[ZImageModVecs]
    var main_mod: List[ZImageModVecs]
    var x_cos: TArc
    var x_sin: TArc
    var cap_cos: TArc
    var cap_sin: TArc
    var uni_cos: TArc                       # single-sample [x_pos|cap_pos]
    var uni_sin: TArc
    var x_pos: List[List[Int]]
    var cap_pos: List[List[Int]]
    var pix_h: List[Float32]
    var noise_pix: List[Float32]
    var noisy_pixel_t: Tensor
    var valid_cap: Int
    var cap_attn_len: Int
    var main_attn_len: Int
    var sigma: Float32


def _build_cond(
    aux: L2PRealAux, cache: L2PCache, slot: Int, sigma: Float32,
    noise_seed: UInt64, ctx: DeviceContext,
) raises -> _L2PCond:
    var s = cache.load(slot, ctx)
    var psh = s.pixel.shape()
    if len(psh) != 3 or psh[0] != PIX_C or psh[1] != PIX_H or psh[2] != PIX_W:
        raise Error("l2p_batch2_parity: pixel shape mismatch — expected [3,512,512]")
    var csh = s.cap_feats.shape()
    if len(csh) != 3 or csh[0] != 1 or csh[2] != CAP_DIM:
        raise Error("l2p_batch2_parity: cap_feats shape mismatch — expected [1,seq,2560]")
    var valid_cap = csh[1]
    if valid_cap <= 0 or valid_cap > CAP_LEN:
        raise Error("l2p_batch2_parity: caption length out of range")
    var cap_attn_len = ((valid_cap + 31) // 32) * 32
    if cap_attn_len > CAP_LEN:
        cap_attn_len = CAP_LEN
    var main_attn_len = N_IMG + cap_attn_len

    var pix_h = cast_tensor(s.pixel, STDtype.F32, ctx).to_host(ctx)
    var t_value = Float32(1.0) - sigma
    var noise_pix = _host_noise_l2p(NPIX, noise_seed)
    var noisy_h = List[Float32]()
    for i in range(NPIX):
        noisy_h.append(pix_h[i] * (Float32(1.0) - sigma) + noise_pix[i] * sigma)
    var noisy_pixel_t = Tensor.from_host(noisy_h^, [1, PIX_C, PIX_H, PIX_W], STDtype.F32, ctx)

    var adaln = build_l2p_adaln(aux, t_value, T_SCALE, ctx)
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod.append(build_l2p_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln, D, ctx))
    var main_mod = List[ZImageModVecs]()
    for i in range(MAIN_DEPTH):
        main_mod.append(build_l2p_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln, D, ctx))

    var x_t = build_l2p_x_seq(aux, noisy_pixel_t, PIX_H, PIX_W, ctx)

    var cap_feats = cast_tensor(s.cap_feats, STDtype.F32, ctx)
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

    var pos_step = build_l2p_positions(N_IMG, HT, WT, CAP_LEN, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var xr = build_l2p_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var ur = build_l2p_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var crr = build_l2p_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)

    return _L2PCond(
        x_t^, cap_seq^, nr_mod^, main_mod^,
        xr[0].copy(), xr[1].copy(), crr[0].copy(), crr[1].copy(),
        ur[0].copy(), ur[1].copy(),
        x_pos^, cap_pos^, pix_h^, noise_pix^, noisy_pixel_t^,
        valid_cap, cap_attn_len, main_attn_len, sigma,
    )


@fieldwise_init
struct _RunOut(Movable):
    var loss: Float32
    var pred: List[Float32]
    var grads: List[Float32]


# ── genuine UNMASKED b1 (the named trainer functions; informational only) ─────
def _run_b1_unmasked(
    dec: L2PDecoderF32,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    lora_dev: ZImageLoraDeviceSet,
    cond: _L2PCond,
    ident_w: Tensor, zero_b: Tensor, f_scale_zeros: List[Float32],
    ctx: DeviceContext,
) raises -> _RunOut:
    var fwd = zimage_stack_lora_forward_main_device[H, Dh, N_IMG, N_TXT, S](
        cond.x_t.copy(), cond.cap_seq.copy(),
        nr_blocks, cond.nr_mod, cr_blocks, main_blocks, cond.main_mod, lora_dev,
        f_scale_zeros.copy(), ident_w, zero_b,
        cond.x_cos[], cond.x_sin[], cond.cap_cos[], cond.cap_sin[],
        cond.uni_cos[], cond.uni_sin[],
        D, F, D, EPS, FINAL_EPS, ctx,
    )
    var xf = fwd.x_final[].to_host(ctx)     # [S, D]
    var feat = _tokens_to_feat_nchw(xf, ctx)
    var dec_fwd = l2p_decoder_forward[PIX_H, PIX_W, HT, WT](dec, cond.noisy_pixel_t, feat, ctx)
    var pred_h = dec_fwd.pred_nchw.to_host(ctx)
    var inv_n = Float32(2.0) / Float32(NPIX)
    var d_pred_h = List[Float32]()
    for _ in range(NPIX):
        d_pred_h.append(Float32(0.0))
    var sse = 0.0
    for i in range(NPIX):
        var pred = -pred_h[i]
        var target = cond.noise_pix[i] - cond.pix_h[i]
        var diff = pred - target
        sse += Float64(diff) * Float64(diff)
        d_pred_h[i] = -inv_n * diff
    var loss = Float32(sse / Float64(NPIX))
    var d_pred_t = Tensor.from_host(d_pred_h^, [1, PIX_C, PIX_H, PIX_W], STDtype.F32, ctx)
    var d_feat = l2p_decoder_backward[PIX_H, PIX_W, HT, WT](dec, dec_fwd.acts, d_pred_t, ctx)
    var d_x_full = _feat_nchw_to_tokens(d_feat, ctx)
    var grads = zimage_stack_lora_backward_main_device_nofinal[H, Dh, N_IMG, N_TXT, S](
        d_x_full, main_blocks, cond.main_mod, lora_dev,
        cond.uni_cos[], cond.uni_sin[], fwd, D, F, EPS, ctx,
    )
    return _RunOut(loss, pred_h^, _concat_trained_grads(grads))


# ── MASKED per-sample reference: masked B=2 stack on a DUPLICATED sample ──────
# (s_i, s_i) split to half-0 IS the exact per-sample masked computation b2 stacks.
def _run_masked_single(
    dec: L2PDecoderF32,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    lora_dev: ZImageLoraDeviceSet,
    cond: _L2PCond,
    ident_w: Tensor, zero_b: Tensor, f_scale2_zeros: List[Float32],
    ctx: DeviceContext,
) raises -> _RunOut:
    var main_mod_b2 = List[ZImageModVecsDevice]()
    for i in range(MAIN_DEPTH):
        main_mod_b2.append(zimage_modvecs_pack2_to_device(cond.main_mod[i], cond.main_mod[i], D, ctx))
    var uni2_pos = List[List[Int]]()
    for i in range(len(cond.x_pos)):
        uni2_pos.append(cond.x_pos[i].copy())
    for i in range(len(cond.cap_pos)):
        uni2_pos.append(cond.cap_pos[i].copy())
    for i in range(len(cond.x_pos)):
        uni2_pos.append(cond.x_pos[i].copy())
    for i in range(len(cond.cap_pos)):
        uni2_pos.append(cond.cap_pos[i].copy())
    var ur2 = build_l2p_rope(uni2_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)

    var fwd = zimage_stack_lora_forward_main_device_b2_masked[H, Dh, N_IMG, N_TXT, S](
        cond.x_t.copy(), cond.cap_seq.copy(), cond.x_t.copy(), cond.cap_seq.copy(),
        cond.cap_attn_len, cond.cap_attn_len, cond.main_attn_len, cond.main_attn_len,
        nr_blocks, cond.nr_mod, cond.nr_mod, cr_blocks, main_blocks, main_mod_b2, lora_dev,
        f_scale2_zeros.copy(), ident_w, zero_b,
        cond.x_cos[], cond.x_sin[], cond.cap_cos[], cond.cap_sin[],
        cond.cap_cos[], cond.cap_sin[], ur2[0][], ur2[1][],
        D, F, D, EPS, FINAL_EPS, ctx,
    )
    var xf = fwd.x_final[].to_host(ctx)     # [2S, D]
    var xf0 = List[Float32]()
    for i in range(S * D):
        xf0.append(xf[i])
    var feat = _tokens_to_feat_nchw(xf0, ctx)
    var dec_fwd = l2p_decoder_forward[PIX_H, PIX_W, HT, WT](dec, cond.noisy_pixel_t, feat, ctx)
    var pred_h = dec_fwd.pred_nchw.to_host(ctx)

    # single-sample MSE (== the per-sample MSE that b2 averages) for loss parity;
    # dup grad scale = 1/NPIX on BOTH halves => grads == the masked b1 grad.
    var inv_n_b2 = Float32(2.0) / Float32(2 * NPIX)
    var d_pred_h = List[Float32]()
    for _ in range(NPIX):
        d_pred_h.append(Float32(0.0))
    var sse = 0.0
    for i in range(NPIX):
        var pred = -pred_h[i]
        var target = cond.noise_pix[i] - cond.pix_h[i]
        var diff = pred - target
        sse += Float64(diff) * Float64(diff)
        d_pred_h[i] = -inv_n_b2 * diff
    var loss = Float32(sse / Float64(NPIX))
    var d_pred_t = Tensor.from_host(d_pred_h^, [1, PIX_C, PIX_H, PIX_W], STDtype.F32, ctx)
    var d_feat = l2p_decoder_backward[PIX_H, PIX_W, HT, WT](dec, dec_fwd.acts, d_pred_t, ctx)
    var d_x = _feat_nchw_to_tokens(d_feat, ctx)     # [S, D]
    var d_x_full_2s = d_x.copy()
    for i in range(len(d_x)):
        d_x_full_2s.append(d_x[i])                  # both halves = sample i
    var grads = zimage_stack_lora_backward_main_device_b2_masked_nofinal[H, Dh, N_IMG, N_TXT, S](
        d_x_full_2s, cond.main_attn_len, cond.main_attn_len,
        main_blocks, main_mod_b2, lora_dev, ur2[0][], ur2[1][], fwd, D, F, EPS, ctx,
    )
    return _RunOut(loss, pred_h^, _concat_trained_grads(grads))


# ── real MASKED B=2 (the trainer's b2 step, inline) ──────────────────────────
@fieldwise_init
struct _B2Out(Movable):
    var loss: Float32
    var mse0: Float32
    var mse1: Float32
    var pred0: List[Float32]
    var pred1: List[Float32]
    var grads: List[Float32]


def _run_b2(
    dec: L2PDecoderF32,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    lora_dev: ZImageLoraDeviceSet,
    c0: _L2PCond, c1: _L2PCond,
    ident_w: Tensor, zero_b: Tensor, f_scale2_zeros: List[Float32],
    ctx: DeviceContext,
) raises -> _B2Out:
    var main_mod_b2 = List[ZImageModVecsDevice]()
    for i in range(MAIN_DEPTH):
        main_mod_b2.append(zimage_modvecs_pack2_to_device(c0.main_mod[i], c1.main_mod[i], D, ctx))
    var uni2_pos = List[List[Int]]()
    for i in range(len(c0.x_pos)):
        uni2_pos.append(c0.x_pos[i].copy())
    for i in range(len(c0.cap_pos)):
        uni2_pos.append(c0.cap_pos[i].copy())
    for i in range(len(c1.x_pos)):
        uni2_pos.append(c1.x_pos[i].copy())
    for i in range(len(c1.cap_pos)):
        uni2_pos.append(c1.cap_pos[i].copy())
    var ur2 = build_l2p_rope(uni2_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)

    var fwd = zimage_stack_lora_forward_main_device_b2_masked[H, Dh, N_IMG, N_TXT, S](
        c0.x_t.copy(), c0.cap_seq.copy(), c1.x_t.copy(), c1.cap_seq.copy(),
        c0.cap_attn_len, c1.cap_attn_len, c0.main_attn_len, c1.main_attn_len,
        nr_blocks, c0.nr_mod, c1.nr_mod, cr_blocks, main_blocks, main_mod_b2, lora_dev,
        f_scale2_zeros.copy(), ident_w, zero_b,
        c0.x_cos[], c0.x_sin[], c0.cap_cos[], c0.cap_sin[],
        c1.cap_cos[], c1.cap_sin[], ur2[0][], ur2[1][],
        D, F, D, EPS, FINAL_EPS, ctx,
    )
    var xf = fwd.x_final[].to_host(ctx)     # [2S, D]
    var xf0 = List[Float32]()
    for i in range(S * D):
        xf0.append(xf[i])
    var xf1 = List[Float32]()
    var base1 = S * D
    for i in range(S * D):
        xf1.append(xf[base1 + i])
    var feat0 = _tokens_to_feat_nchw(xf0, ctx)
    var feat1 = _tokens_to_feat_nchw(xf1, ctx)
    var dec_fwd0 = l2p_decoder_forward[PIX_H, PIX_W, HT, WT](dec, c0.noisy_pixel_t, feat0, ctx)
    var dec_fwd1 = l2p_decoder_forward[PIX_H, PIX_W, HT, WT](dec, c1.noisy_pixel_t, feat1, ctx)
    var pred_h0 = dec_fwd0.pred_nchw.to_host(ctx)
    var pred_h1 = dec_fwd1.pred_nchw.to_host(ctx)

    var inv_n_b2 = Float32(2.0) / Float32(2 * NPIX)
    var d_pred0_h = List[Float32]()
    var d_pred1_h = List[Float32]()
    for _ in range(NPIX):
        d_pred0_h.append(Float32(0.0))
        d_pred1_h.append(Float32(0.0))
    var sse0 = 0.0
    var sse1 = 0.0
    for i in range(NPIX):
        var pred = -pred_h0[i]
        var target = c0.noise_pix[i] - c0.pix_h[i]
        var diff = pred - target
        sse0 += Float64(diff) * Float64(diff)
        d_pred0_h[i] = -inv_n_b2 * diff
    for i in range(NPIX):
        var pred = -pred_h1[i]
        var target = c1.noise_pix[i] - c1.pix_h[i]
        var diff = pred - target
        sse1 += Float64(diff) * Float64(diff)
        d_pred1_h[i] = -inv_n_b2 * diff
    var mse0 = Float32(sse0 / Float64(NPIX))
    var mse1 = Float32(sse1 / Float64(NPIX))
    var loss = Float32((sse0 + sse1) / Float64(2 * NPIX))
    var d_pred_t0 = Tensor.from_host(d_pred0_h^, [1, PIX_C, PIX_H, PIX_W], STDtype.F32, ctx)
    var d_pred_t1 = Tensor.from_host(d_pred1_h^, [1, PIX_C, PIX_H, PIX_W], STDtype.F32, ctx)
    var d_feat0 = l2p_decoder_backward[PIX_H, PIX_W, HT, WT](dec, dec_fwd0.acts, d_pred_t0, ctx)
    var d_feat1 = l2p_decoder_backward[PIX_H, PIX_W, HT, WT](dec, dec_fwd1.acts, d_pred_t1, ctx)
    var d_x0 = _feat_nchw_to_tokens(d_feat0, ctx)
    var d_x1 = _feat_nchw_to_tokens(d_feat1, ctx)
    var d_x_full_2s = d_x0.copy()
    for i in range(len(d_x1)):
        d_x_full_2s.append(d_x1[i])
    var grads = zimage_stack_lora_backward_main_device_b2_masked_nofinal[H, Dh, N_IMG, N_TXT, S](
        d_x_full_2s, c0.main_attn_len, c1.main_attn_len,
        main_blocks, main_mod_b2, lora_dev, ur2[0][], ur2[1][], fwd, D, F, EPS, ctx,
    )
    return _B2Out(loss, mse0, mse1, pred_h0^, pred_h1^, _concat_trained_grads(grads))


def _perturb_b(mut lora: ZImageLoraSet):
    """Lift trained LoRA-B off the bf16 floor (numerical-vs-bug discriminator)."""
    var s = UInt64(1234567)
    for i in range(TRAIN_ADAPTER_START, N_ADAPTERS_TOTAL):
        var bb = lora.ad[i].b.copy()
        for j in range(len(bb)):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            var r = Float32(Int((s >> UInt64(40)) % UInt64(2000)) - 1000) * Float32(0.00005)
            bb[j] = Float32(r).cast[DType.bfloat16]()
        lora.ad[i].b = bb^


def main() raises:
    var ctx = DeviceContext()
    print("==== l2p_batch2_parity (TRUE batch-2 masked stack vs per-sample masked ref) ====")
    print("  D=", D, " H=", H, " S=", S, " N_IMG=", N_IMG, " CAP_LEN=", CAP_LEN)

    var st = SafeTensors.open(String(CHECKPOINT_PATH))
    var aux = load_l2p_real_aux(st, NUM_NR, MAIN_DEPTH, ctx)
    var nr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_NR):
        nr_blocks.append(load_l2p_block_weights_prefixed(st, String("noise_refiner.") + String(i), ctx))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(load_l2p_block_weights_prefixed(st, String("context_refiner.") + String(i), ctx))
    var main_blocks = List[ZImageBlockWeights]()
    for i in range(MAIN_DEPTH):
        main_blocks.append(load_l2p_block_weights_prefixed(st, String("layers.") + String(i), ctx))
    var dec_gate = ZImageL2PLocalDecoderGate.load(String(CHECKPOINT_PATH), ctx)
    var dec = l2p_decoder_f32_from_gate(dec_gate, ctx)

    var lora = build_zimage_lora_set(NUM_NR, NUM_CR, MAIN_DEPTH, D, F, RANK, ALPHA)
    _perturb_b(lora)
    print("  [discriminator] perturbed trained LoRA-B to small nonzero (off the bf16 floor)")
    var lora_dev = zimage_lora_set_to_device(lora, ctx)

    var cache = L2PCache(String(CACHE_DIR))
    print("  cache samples:", cache.count())

    # ── FIXED sigma + FIXED noise seeds per sample (deterministic) ────────────
    var c0 = _build_cond(aux, cache, 0, Float32(0.35), UInt64(1234567), ctx)
    var c1 = _build_cond(aux, cache, 1, Float32(0.72), UInt64(7654321), ctx)
    print("  sample valid_cap: s0=", c0.valid_cap, " (cap_attn_len=", c0.cap_attn_len,
          ")  s1=", c1.valid_cap, " (cap_attn_len=", c1.cap_attn_len, ")")

    # ── identity final layer (out discarded; features = last-block hidden) ────
    var f_scale_zeros = List[Float32]()
    for _ in range(D):
        f_scale_zeros.append(Float32(0.0))
    var f_scale2_zeros = List[Float32]()
    for _ in range(2 * D):
        f_scale2_zeros.append(Float32(0.0))
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

    # ── per-sample MASKED reference (dup-b2 half-0) ───────────────────────────
    var ref0 = _run_masked_single(dec, nr_blocks, cr_blocks, main_blocks, lora_dev, c0, ident_w, zero_b, f_scale2_zeros, ctx)
    var ref1 = _run_masked_single(dec, nr_blocks, cr_blocks, main_blocks, lora_dev, c1, ident_w, zero_b, f_scale2_zeros, ctx)

    # ── real MASKED B=2 ───────────────────────────────────────────────────────
    var b2 = _run_b2(dec, nr_blocks, cr_blocks, main_blocks, lora_dev, c0, c1, ident_w, zero_b, f_scale2_zeros, ctx)

    # ── genuine UNMASKED b1 (informational only) ──────────────────────────────
    var um0 = _run_b1_unmasked(dec, nr_blocks, cr_blocks, main_blocks, lora_dev, c0, ident_w, zero_b, f_scale_zeros, ctx)
    var um1 = _run_b1_unmasked(dec, nr_blocks, cr_blocks, main_blocks, lora_dev, c1, ident_w, zero_b, f_scale_zeros, ctx)

    var allok = True

    # ── (a) loss parity (BINDING) ─────────────────────────────────────────────
    print("")
    print("---- (a) loss_B2 vs mean(loss_ref(s0), loss_ref(s1))  [BINDING] ----")
    var loss_mean = Float64(0.5) * (Float64(ref0.loss) + Float64(ref1.loss))
    var loss_b2 = Float64(b2.loss)
    var loss_rel = _absf(loss_b2 - loss_mean) / (loss_mean if loss_mean > Float64(1.0e-8) else Float64(1.0e-8))
    print("  loss_ref(s0)=", ref0.loss, "  loss_ref(s1)=", ref1.loss)
    print("  per-sample MSE in B2: s0=", b2.mse0, "  s1=", b2.mse1)
    print("  mean=", loss_mean, "  loss_B2=", loss_b2, "  rel=", loss_rel,
          "  ", "PASS" if loss_rel <= LOSS_REL_BAR else "FAIL")
    if loss_rel > LOSS_REL_BAR:
        allok = False

    # ── (b) forward velocity: per-sample decoder pred (BINDING) ───────────────
    print("")
    print("---- (b) forward velocity: cos(b2_pred_i, ref_pred_i)  [BINDING] ----")
    var cos_p0 = _cos(b2.pred0, ref0.pred)
    var cos_p1 = _cos(b2.pred1, ref1.pred)
    print("  cos(b2_pred0, ref_pred0)=", cos_p0, "  ", "PASS" if cos_p0 >= COS_BAR else "FAIL")
    print("  cos(b2_pred1, ref_pred1)=", cos_p1, "  ", "PASS" if cos_p1 >= COS_BAR else "FAIL")
    if cos_p0 < COS_BAR or cos_p1 < COS_BAR:
        allok = False

    # ── (c) grad cosine — INFORMATIONAL (MJ-1073) ─────────────────────────────
    print("")
    print("---- (c) grad cosine [INFORMATIONAL — MJ-1073] ----")
    var v_b2 = b2.grads.copy()
    var v_ref_mean = _mean2(ref0.grads, ref1.grads)
    var v_um_mean = _mean2(um0.grads, um1.grads)
    if len(v_b2) == len(v_ref_mean):
        print("  B2 vs mean(masked ref) cosine =", _cos(v_b2, v_ref_mean),
              "  max_abs=", _max_abs_diff(v_b2, v_ref_mean),
              "  (clean assembly check; both M=2S)")
    else:
        print("  grad length mismatch (masked ref):", len(v_b2), len(v_ref_mean))
    if len(v_b2) == len(v_um_mean):
        print("  B2 vs mean(UNMASKED b1) cosine =", _cos(v_b2, v_um_mean),
              "  (masking + M=S-vs-2S shape; MJ-1073 says NOT a gate bar)")
    print("  ||grad_B2||=", sqrt(_sumsq(v_b2)), "  ||mean(masked ref)||=", sqrt(_sumsq(v_ref_mean)))

    # ── unmasked-vs-masked forward gap (informational; tiny captions) ─────────
    print("")
    print("---- INFO: unmasked-b1 vs masked-ref per-sample loss (214-pad impact) ----")
    print("  s0: unmasked_b1_loss=", um0.loss, "  masked_ref_loss=", ref0.loss)
    print("  s1: unmasked_b1_loss=", um1.loss, "  masked_ref_loss=", ref1.loss)

    print("")
    if allok:
        print("VERDICT: PASS — loss parity + forward velocity hold (batch-2 masked",
              "stack == mean of two per-sample masked forwards).")
    else:
        print("VERDICT: FAIL — see (a)/(b) above.")
