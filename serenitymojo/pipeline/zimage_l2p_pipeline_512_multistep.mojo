# zimage_l2p_pipeline_512_multistep.mojo — Z-Image L2P 512² multistep inference.
#
# Full end-to-end L2P pixel-space diffusion pipeline:
#   1. Load cached cap_feats / cap_feats_uncond from the sidecar safetensors.
#   2. Embed caption + timestep.
#   3. Run 30-step rectified-flow Euler CFG denoise loop (no VAE).
#   4. Save pixel PNG.
#
# Port of inference-flame/src/bin/l2p_infer.rs + l2p/dit.rs forward_inner.
#
# Targets 512×512 (faster, oracle available). PH=PW=32, N_IMG=1024.
# No VAE — output is already pixel-space BF16.
#
# Sign / timestep contract:
#   - Sampler passes sigma [0,1] as-is to `_forward_inner`.
#   - `_forward_inner` remaps: t = (1 - sigma) * 1000.
#   - Model output is negated once inside `_forward_inner`.
#   - Sampler does NOT negate.
#   - CFG: pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)  [Rust form]
#   - Euler: x_next = x + (sigma_next - sigma) * pred

from std.gpu.host import DeviceContext
from std.math import sqrt, cos as fcos, sin as fsin, exp as fexp, log as flog
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.dit.zimage_l2p_contract import (
    ZIMAGE_L2P_CAP_FEAT_DIM,
    ZIMAGE_L2P_HEAD_DIM,
    ZIMAGE_L2P_HIDDEN,
    ZIMAGE_L2P_NUM_HEADS,
    ZIMAGE_L2P_PATCH_SIZE,
    ZIMAGE_L2P_PATCH_VECTOR_DIM,
    ZIMAGE_L2P_PIXEL_CHANNELS,
    ZIMAGE_L2P_TIMESTEP_DIM,
    ZIMAGE_L2P_PAD_MULTIPLE,
    zimage_l2p_default_checkpoint_path,
    zimage_l2p_default_conditioning_path,
    build_zimage_l2p_sigma_schedule,
)
from serenitymojo.models.dit.zimage_l2p_dit import (
    ZImageL2PBlockWeights,
    ZImageL2PContextBlockWeights,
    ZImageL2PDiTPreBlockGate,
    zimage_l2p_block_forward,
    zimage_l2p_context_block_forward,
    _clone,
)
from serenitymojo.models.dit.zimage_l2p_local_decoder import (
    ZImageL2PLocalDecoderGate,
)
from serenitymojo.models.dit.zimage_l2p_rope import build_zimage_l2p_3d_rope
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    add,
    concat,
    mul_scalar,
    permute,
    reshape,
    slice,
)
from serenitymojo.image.png import save_png, ValueRange


# ─── Pipeline constants ───────────────────────────────────────────────────────

comptime L2P_H = 512
comptime L2P_W = 512
comptime PH = L2P_H // ZIMAGE_L2P_PATCH_SIZE   # 32
comptime PW = L2P_W // ZIMAGE_L2P_PATCH_SIZE   # 32
comptime N_IMG = PH * PW                         # 1024
# From the sidecar: cap_feats=[1,32,2560], cap_feats_uncond=[1,8,2560].
# 32 is already a multiple of 32 → no cap padding.
# 8 is not a multiple of 32 → pad uncond to 32.
comptime CAP_LEN = 32                            # cond cap tokens (sidecar)
comptime UNCOND_LEN = 8                          # uncond cap tokens (sidecar)
comptime CAP_PADDED = 32                         # padded cond (already aligned)
comptime UNCOND_PADDED = 32                      # padded uncond (8 → 32)
# Image tokens: 1024 is already a multiple of 32 → no img padding.
comptime IMG_PADDED = N_IMG                      # 1024 (no padding)
comptime IMG_PAD = 0

# Joint sequence lengths (context + image) for main layers
comptime S_JOINT = CAP_PADDED + IMG_PADDED       # 32 + 1024 = 1056
# Refiner sequence lengths
comptime S_IMG = IMG_PADDED                      # 1024 (noise refiner: image only)
comptime S_CAP = CAP_PADDED                      # 32 (context refiner: cap only)
# Uncond joint
comptime S_JOINT_UNCOND = UNCOND_PADDED + IMG_PADDED  # 32 + 1024 = 1056

comptime NUM_LAYERS = 30
comptime NUM_REFINERS = 2
comptime NUM_STEPS = 30
comptime CFG_SCALE = Float32(2.0)
comptime SHIFT = Float32(3.0)
comptime SEED = UInt64(42)
comptime EPS = Float32(1.0e-5)
comptime CKPT_PATH = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
comptime COND_PATH = "/home/alex/EriDiffusion/inference-flame/output/l2p_embeddings.safetensors"
comptime OUT_PATH = "/home/alex/mojodiffusion/output/zimage_l2p_512_30step.png"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    if n == 0:
        print("  [stat]", name, "EMPTY")
        return
    var s = Float64(0.0)
    var s2 = Float64(0.0)
    var amax = Float64(0.0)
    var any_nonfinite = False
    for i in range(n):
        var v = Float64(h[i])
        if v != v:
            any_nonfinite = True
        var av = v if v >= 0.0 else -v
        if av > Float64(3.4e38):
            any_nonfinite = True
        s += v
        s2 += v * v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  [stat]", name,
        "n=", n,
        "mean=", Float32(mean),
        "std=", Float32(sqrt(Float32(var_))),
        "absmax=", Float32(amax),
    )
    if any_nonfinite:
        print("  WARNING: non-finite values in", name)


def _load_bf16(
    ref st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view_as_bf16(tv, ctx)


# ─── Timestep embedding ───────────────────────────────────────────────────────

def _timestep_embed(
    sigma: Float32,
    t_w0: Tensor,
    t_b0: Tensor,
    t_w2: Tensor,
    t_b2: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Embed sigma -> t_cond [1, 256].

    Rust: t = (1 - sigma) * 1000, then sinusoidal with max_period=10000,
    half_dim=128, then MLP(256->1024->256).
    """
    var dim = ZIMAGE_L2P_TIMESTEP_DIM   # 256
    var half = dim // 2                  # 128
    var scaled = (1.0 - sigma) * 1000.0
    var max_period = Float32(10000.0)
    var log_mp = flog(max_period)
    var emb = List[Float32]()
    for i in range(half):
        var freq = fexp(-log_mp * Float32(i) / Float32(half))
        emb.append(fcos(scaled * freq))
    for i in range(half):
        var freq = fexp(-log_mp * Float32(i) / Float32(half))
        emb.append(fsin(scaled * freq))
    var sh = List[Int]()
    sh.append(1)
    sh.append(dim)
    var t_freq = Tensor.from_host(emb, sh^, STDtype.BF16, ctx)
    var h = linear(t_freq, t_w0, Optional[Tensor](_clone(t_b0, ctx)), ctx)
    h = silu(h, ctx)
    return linear(h, t_w2, Optional[Tensor](_clone(t_b2, ctx)), ctx)


# ─── Caption embedding ────────────────────────────────────────────────────────

def _caption_embed(
    cap_feats: Tensor,
    cap_norm_w: Tensor,
    cap_w: Tensor,
    cap_b: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Embed cap_feats [1, S, 2560] -> [1, S, 3840]."""
    var normed = rms_norm(cap_feats, cap_norm_w, EPS, ctx)
    return linear(normed, cap_w, Optional[Tensor](_clone(cap_b, ctx)), ctx)


# ─── Padding token append ────────────────────────────────────────────────────

def _pad_tokens[PAD_LEN: Int](
    ref tokens: Tensor,
    ref pad_token_w: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Append PAD_LEN copies of pad_token to tokens [1, S, D] -> [1, S+PAD_LEN, D].

    If PAD_LEN == 0 returns a clone of tokens.
    """
    var sh = tokens.shape()
    var d = sh[2]
    comptime if PAD_LEN == 0:
        # No padding needed — copy and return
        var buf = ctx.enqueue_create_buffer[DType.uint8](tokens.nbytes())
        ctx.enqueue_copy(dst_buf=buf, src_buf=tokens.buf)
        ctx.synchronize()
        return Tensor(buf^, tokens.shape(), tokens.dtype())
    else:
        # Build PAD_LEN rows by reading pad_token once and repeating.
        var pad_shape = List[Int]()
        pad_shape.append(1)
        pad_shape.append(PAD_LEN)
        pad_shape.append(d)
        var host = pad_token_w.to_host(ctx)
        var data = List[Float32]()
        for _p in range(PAD_LEN):
            for j in range(d):
                data.append(host[j])
        var pad_tensor = Tensor.from_host(data, pad_shape^, STDtype.BF16, ctx)
        return concat(1, ctx, tokens, pad_tensor)


# ─── Patchify pixel ──────────────────────────────────────────────────────────

def _patchify_pixel[H: Int, W: Int](pixels_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
    """[1,3,H,W] NCHW BF16 -> [1, (H/16)*(W/16), 768] patches."""
    comptime P = ZIMAGE_L2P_PATCH_SIZE  # 16
    comptime ph = H // P
    comptime pw = W // P
    var v = List[Int]()
    v.append(1)
    v.append(ZIMAGE_L2P_PIXEL_CHANNELS)
    v.append(ph)
    v.append(P)
    v.append(pw)
    v.append(P)
    var viewed = reshape(pixels_nchw, v^, ctx)
    var axes = List[Int]()
    axes.append(0)
    axes.append(2)
    axes.append(4)
    axes.append(3)
    axes.append(5)
    axes.append(1)
    var packed = permute(viewed, axes^, ctx)
    var out_shape = List[Int]()
    out_shape.append(1)
    out_shape.append(ph * pw)
    out_shape.append(ZIMAGE_L2P_PATCH_VECTOR_DIM)
    return reshape(packed, out_shape^, ctx)


# ─── Unpatchify (feat_map extraction) ────────────────────────────────────────

def _extract_feat_map[PH_: Int, PW_: Int](
    xc: Tensor,   # [1, S_JOINT, 3840]  (cap + img tokens)
    cap_len: Int, # tokens to skip at front
    img_len: Int, # image token count to extract (= PH_*PW_, no padding)
    ctx: DeviceContext,
) raises -> Tensor:
    """Extract image tokens from joint xc and reshape to [1, 3840, PH, PW] NCHW."""
    # Slice out image tokens: [1, img_len, 3840]
    var x_img = slice(xc, 1, cap_len, img_len, ctx)
    # Reshape to [1, PH, PW, 3840]
    var ph_sh = List[Int]()
    ph_sh.append(1)
    ph_sh.append(PH_)
    ph_sh.append(PW_)
    ph_sh.append(ZIMAGE_L2P_HIDDEN)
    var x_2d = reshape(x_img, ph_sh^, ctx)
    # Permute to NCHW [1, 3840, PH, PW]
    var p = List[Int]()
    p.append(0)
    p.append(3)
    p.append(1)
    p.append(2)
    return permute(x_2d, p^, ctx)


# ─── RoPE slice ──────────────────────────────────────────────────────────────

def _rope_slice(
    rope_full: Tensor,  # [total_seq, 64]
    start: Int,
    length: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Slice [start:start+length, :] from a [total_seq, 64] rope table."""
    return slice(rope_full, 0, start, length, ctx)


# ─── Preloaded model weights ─────────────────────────────────────────────────
# All transformer block weights loaded to GPU once before the denoise loop.
# Uses ArcPointer to store Movable structs in a List.

@fieldwise_init
struct L2PModel(Movable):
    """All L2P DiT weights resident on GPU. Loaded once, reused each step."""
    # context_refiner.0, context_refiner.1 (no adaLN)
    var ctx0: ZImageL2PContextBlockWeights
    var ctx1: ZImageL2PContextBlockWeights
    # noise_refiner.0, noise_refiner.1 (with adaLN)
    var nr0: ZImageL2PBlockWeights
    var nr1: ZImageL2PBlockWeights
    # layers.0..29 — store as list of ArcPointer so we can index at runtime.
    # ArcPointer[ZImageL2PBlockWeights] is Copyable (refcount-bumped copy).
    var layers: List[ArcPointer[ZImageL2PBlockWeights]]

    @staticmethod
    def load(ctx: DeviceContext) raises -> L2PModel:
        print("[model] Loading all L2P DiT block weights to GPU...")
        var ckpt = String(CKPT_PATH)
        var st = SafeTensors.open(ckpt)
        var ctx0 = ZImageL2PContextBlockWeights.load_from_st(
            st, String("context_refiner.0"), ctx
        )
        print("  context_refiner.0 loaded")
        var ctx1 = ZImageL2PContextBlockWeights.load_from_st(
            st, String("context_refiner.1"), ctx
        )
        print("  context_refiner.1 loaded")
        var nr0 = ZImageL2PBlockWeights.load_from_st(
            st, String("noise_refiner.0"), ctx
        )
        print("  noise_refiner.0 loaded")
        var nr1 = ZImageL2PBlockWeights.load_from_st(
            st, String("noise_refiner.1"), ctx
        )
        print("  noise_refiner.1 loaded")
        var layers = List[ArcPointer[ZImageL2PBlockWeights]]()
        for i in range(NUM_LAYERS):
            var prefix = String("layers.") + String(i)
            var bw = ZImageL2PBlockWeights.load_from_st(st, prefix, ctx)
            layers.append(ArcPointer(bw^))
            if i % 5 == 4 or i == 0:
                print("  layers.", i, " loaded")
        print("[model] All weights loaded")
        return L2PModel(ctx0^, ctx1^, nr0^, nr1^, layers^)


# ─── Single forward pass ─────────────────────────────────────────────────────
# Runs the full L2P forward for ONE set of conditioning tokens.
# Returns the negated pixel prediction [1,3,H,W] BF16.
#
# IMPORTANT: this is called separately for cond + uncond; the caller does CFG.

def _forward_inner[H: Int, W: Int](
    ref noisy_pixel: Tensor,         # [1,3,H,W] BF16
    ref cap_feats_embedded: Tensor,  # [1, CAP_PADDED, 3840] BF16 (already embedded+padded)
    ref t_cond: Tensor,              # [1, 256] BF16
    ref rope_cos_full: Tensor,       # [S_JOINT, 64] BF16
    ref rope_sin_full: Tensor,
    ref x_w: Tensor,
    ref x_b: Tensor,
    ref model: L2PModel,
    ref local_decoder: ZImageL2PLocalDecoderGate,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime P = ZIMAGE_L2P_PATCH_SIZE
    comptime ph = H // P   # 32 for 512
    comptime pw = W // P   # 32 for 512
    comptime n_img = ph * pw  # 1024
    comptime cap_p = CAP_PADDED       # 32
    comptime img_p = IMG_PADDED       # 1024
    comptime s_joint = cap_p + img_p  # 1056
    comptime s_img = img_p            # 1024
    comptime s_cap = cap_p            # 32

    # 1. Patchify + x_embedder: [1,3,H,W] -> [1,N_IMG,768] -> [1,N_IMG,3840]
    var patches = _patchify_pixel[H, W](noisy_pixel, ctx)
    var x_emb = linear(patches, x_w, Optional[Tensor](_clone(x_b, ctx)), ctx)

    # 2. RoPE slices for refiners (split from full joint rope)
    var rope_cos_cap = _rope_slice(rope_cos_full, 0, s_cap, ctx)
    var rope_sin_cap = _rope_slice(rope_sin_full, 0, s_cap, ctx)
    var rope_cos_img = _rope_slice(rope_cos_full, s_cap, s_img, ctx)
    var rope_sin_img = _rope_slice(rope_sin_full, s_cap, s_img, ctx)

    # 3. Context refiner: 2 blocks, caption only, NO adaLN (no t_cond)
    # Clone embedded cap so we can mutate it
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](cap_feats_embedded.nbytes())
    ctx.enqueue_copy(dst_buf=c_buf, src_buf=cap_feats_embedded.buf)
    ctx.synchronize()
    var c_emb = Tensor(c_buf^, cap_feats_embedded.shape(), cap_feats_embedded.dtype())

    c_emb = zimage_l2p_context_block_forward[1, s_cap](
        model.ctx0, c_emb, rope_cos_cap, rope_sin_cap,
        ZIMAGE_L2P_NUM_HEADS, ZIMAGE_L2P_HEAD_DIM, EPS, ctx,
    )
    c_emb = zimage_l2p_context_block_forward[1, s_cap](
        model.ctx1, c_emb, rope_cos_cap, rope_sin_cap,
        ZIMAGE_L2P_NUM_HEADS, ZIMAGE_L2P_HEAD_DIM, EPS, ctx,
    )

    # 4. Noise refiner: 2 blocks, image only, WITH adaLN (t_cond)
    var x_r = x_emb^
    x_r = zimage_l2p_block_forward[1, s_img](
        model.nr0, x_r, rope_cos_img, rope_sin_img,
        t_cond,
        ZIMAGE_L2P_NUM_HEADS, ZIMAGE_L2P_HEAD_DIM, EPS, ctx,
    )
    x_r = zimage_l2p_block_forward[1, s_img](
        model.nr1, x_r, rope_cos_img, rope_sin_img,
        t_cond,
        ZIMAGE_L2P_NUM_HEADS, ZIMAGE_L2P_HEAD_DIM, EPS, ctx,
    )

    # 5. Concat [cap, img] -> joint [1, S_JOINT, 3840]
    var xc = concat(1, ctx, c_emb, x_r)

    # 6. Main transformer layers 0..29
    for i in range(NUM_LAYERS):
        # Deref the ArcPointer to get the block weights
        var bw_arc = model.layers[i]
        xc = zimage_l2p_block_forward[1, s_joint](
            bw_arc[], xc, rope_cos_full, rope_sin_full,
            t_cond,
            ZIMAGE_L2P_NUM_HEADS, ZIMAGE_L2P_HEAD_DIM, EPS, ctx,
        )

    # 7. Extract image feat_map from joint output
    var feat_map = _extract_feat_map[ph, pw](xc, cap_p, n_img, ctx)

    # 8. Local U-Net pixel head: (noisy_pixel, feat_map) -> [1,3,H,W]
    var out = local_decoder.full_tiny_forward[H, W](noisy_pixel, feat_map, ctx)

    # 9. Sign flip (Rust: `local_out.mul_scalar(-1.0)`)
    return mul_scalar(out, Float32(-1.0), ctx)


# ─── CFG + Euler step ────────────────────────────────────────────────────────

def _cfg(pred_cond: Tensor, pred_uncond: Tensor, scale: Float32, ctx: DeviceContext) raises -> Tensor:
    """CFG: pred = pred_uncond + scale * (pred_cond - pred_uncond).
    Matches Rust l2p_sampling.rs l2p_euler_step CFG formula.
    """
    var diff = add(pred_cond, mul_scalar(pred_uncond, Float32(-1.0), ctx), ctx)
    var scaled = mul_scalar(diff, scale, ctx)
    return add(pred_uncond, scaled, ctx)


def _euler_step(x: Tensor, pred: Tensor, sigma: Float32, sigma_next: Float32, ctx: DeviceContext) raises -> Tensor:
    """x_next = x + (sigma_next - sigma) * pred."""
    var dsigma = sigma_next - sigma
    return add(x, mul_scalar(pred, dsigma, ctx), ctx)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image L2P 512x512 multistep pipeline ===")
    print("  grid:", PH, "x", PW, "patches, N_IMG=", N_IMG)
    print("  S_JOINT=", S_JOINT, "CAP_PADDED=", CAP_PADDED, "IMG_PADDED=", IMG_PADDED)
    print("  steps=", NUM_STEPS, "cfg=", CFG_SCALE, "shift=", SHIFT, "seed=", SEED)

    var ckpt = zimage_l2p_default_checkpoint_path()
    var cond_path = zimage_l2p_default_conditioning_path()

    # ── Stage 1: Load conditioning ────────────────────────────────────────
    print("\n[cond] Loading L2P conditioning sidecar")
    var cond_st = SafeTensors.open(cond_path)
    var cap_info = cond_st.tensor_info(String("cap_feats"))
    var cap_bytes = cond_st.tensor_bytes(String("cap_feats"))
    var cap_tv = from_parts(cap_info.dtype, cap_info.shape.copy(), cap_bytes)
    var cap_feats = Tensor.from_view_as_bf16(cap_tv, ctx)
    print("  cap_feats:", cap_info.shape[0], "x", cap_info.shape[1], "x", cap_info.shape[2])

    var ucap_info = cond_st.tensor_info(String("cap_feats_uncond"))
    var ucap_bytes = cond_st.tensor_bytes(String("cap_feats_uncond"))
    var ucap_tv = from_parts(ucap_info.dtype, ucap_info.shape.copy(), ucap_bytes)
    var cap_feats_uncond = Tensor.from_view_as_bf16(ucap_tv, ctx)
    print("  cap_feats_uncond:", ucap_info.shape[0], "x", ucap_info.shape[1], "x", ucap_info.shape[2])

    # ── Stage 2: Load pre-block weights (x_embedder, cap_embedder, t_embedder, pad tokens)
    print("\n[weights] Loading pre-block + local decoder weights")
    var st = SafeTensors.open(ckpt)
    var x_w = _load_bf16(st, String("all_x_embedder.16-1.weight"), ctx)
    var x_b = _load_bf16(st, String("all_x_embedder.16-1.bias"), ctx)
    var t_w0 = _load_bf16(st, String("t_embedder.mlp.0.weight"), ctx)
    var t_b0 = _load_bf16(st, String("t_embedder.mlp.0.bias"), ctx)
    var t_w2 = _load_bf16(st, String("t_embedder.mlp.2.weight"), ctx)
    var t_b2 = _load_bf16(st, String("t_embedder.mlp.2.bias"), ctx)
    var cap_norm_w = _load_bf16(st, String("cap_embedder.0.weight"), ctx)
    var cap_emb_w = _load_bf16(st, String("cap_embedder.1.weight"), ctx)
    var cap_emb_b = _load_bf16(st, String("cap_embedder.1.bias"), ctx)
    var x_pad_token = _load_bf16(st, String("x_pad_token"), ctx)
    var cap_pad_token = _load_bf16(st, String("cap_pad_token"), ctx)
    print("  pre-block weights loaded")

    var local_decoder = ZImageL2PLocalDecoderGate.load_default(ctx)
    print("  local decoder loaded")

    # ── Preload all transformer block weights ─────────────────────────────────
    var model = L2PModel.load(ctx)

    # ── Stage 3: Build caption embeddings (shared across steps) ──────────────
    # Cond: cap_feats [1, 32, 2560] → embed → [1, 32, 3840] (no padding needed)
    # Uncond: cap_feats_uncond [1, 8, 2560] → embed → [1, 8, 3840] → pad to [1, 32, 3840]
    print("\n[embed] Building caption embeddings")
    var cap_embedded = _caption_embed(cap_feats, cap_norm_w, cap_emb_w, cap_emb_b, ctx)
    _stats("cap_embedded", cap_embedded, ctx)

    var ucap_embedded = _caption_embed(cap_feats_uncond, cap_norm_w, cap_emb_w, cap_emb_b, ctx)
    # Pad uncond from 8 to 32 tokens
    ucap_embedded = _pad_tokens[UNCOND_PADDED - UNCOND_LEN](ucap_embedded, cap_pad_token, ctx)
    _stats("ucap_embedded_padded", ucap_embedded, ctx)

    # ── Stage 4: Build RoPE tables ────────────────────────────────────────────
    # Full rope for joint sequence: [CAP_PADDED + IMG_PADDED, 64] = [1056, 64]
    print("\n[rope] Building 3-axis L2P RoPE tables for joint S=", S_JOINT)
    var rope = build_zimage_l2p_3d_rope[CAP_PADDED, PH, PW, IMG_PAD](ctx)
    var rope_cos = _clone(rope[0], ctx)
    var rope_sin = _clone(rope[1], ctx)
    print("  rope_cos shape:", rope_cos.shape()[0], "x", rope_cos.shape()[1])

    # Uncond rope: UNCOND_PADDED=32=CAP_PADDED so same position layout but
    # caption positions differ only slightly. For uncond we rebuild with UNCOND_PADDED.
    var rope_uncond = build_zimage_l2p_3d_rope[UNCOND_PADDED, PH, PW, IMG_PAD](ctx)
    var rope_cos_u = _clone(rope_uncond[0], ctx)
    var rope_sin_u = _clone(rope_uncond[1], ctx)
    print("  rope_cos_uncond shape:", rope_cos_u.shape()[0], "x", rope_cos_u.shape()[1])

    # ── Stage 5: Initial noise ────────────────────────────────────────────────
    print("\n[noise] Generating initial pixel noise [1,3,512,512]")
    var nchw_shape = List[Int]()
    nchw_shape.append(1)
    nchw_shape.append(ZIMAGE_L2P_PIXEL_CHANNELS)
    nchw_shape.append(L2P_H)
    nchw_shape.append(L2P_W)
    # Keep the production pixel-noise carrier BF16; tensor ops use F32 math
    # internally where needed.
    var x = randn(nchw_shape^, SEED, STDtype.BF16, ctx)
    _stats("init_pixel_noise", x, ctx)

    # ── Stage 6: Build sigma schedule ────────────────────────────────────────
    var sigmas = build_zimage_l2p_sigma_schedule(NUM_STEPS, SHIFT)
    print("[schedule] sigmas[0]=", sigmas[0], "sigmas[last]=", sigmas[NUM_STEPS])

    # ── Stage 7: Denoise loop ─────────────────────────────────────────────────
    print("\n[denoise]", NUM_STEPS, "steps, CFG=", CFG_SCALE)
    for step in range(NUM_STEPS):
        var sigma = sigmas[step]
        var sigma_next = sigmas[step + 1]
        print("  step", step + 1, "/", NUM_STEPS, "sigma=", sigma, "->", sigma_next)

        # Build t_cond from sigma
        var t_cond = _timestep_embed(sigma, t_w0, t_b0, t_w2, t_b2, ctx)

        # Conditional prediction
        var pred_cond = _forward_inner[L2P_H, L2P_W](
            x, cap_embedded, t_cond,
            rope_cos, rope_sin,
            x_w, x_b,
            model, local_decoder,
            ctx,
        )

        # Unconditional prediction
        var pred_uncond = _forward_inner[L2P_H, L2P_W](
            x, ucap_embedded, t_cond,
            rope_cos_u, rope_sin_u,
            x_w, x_b,
            model, local_decoder,
            ctx,
        )

        # CFG: pred = pred_uncond + cfg * (pred_cond - pred_uncond)
        var pred = _cfg(pred_cond, pred_uncond, CFG_SCALE, ctx)

        # Euler step (x stays BF16 throughout, matching Rust)
        x = _euler_step(x, pred, sigma, sigma_next, ctx)

        if (step + 1) % 5 == 0 or step == 0 or step + 1 == NUM_STEPS:
            _stats("pixel", x, ctx)

    # ── Stage 8: Save PNG ─────────────────────────────────────────────────────
    print("\n[save] Saving", OUT_PATH)
    save_png(x, OUT_PATH, ctx, ValueRange.SIGNED)
    print("[done] saved", OUT_PATH)
