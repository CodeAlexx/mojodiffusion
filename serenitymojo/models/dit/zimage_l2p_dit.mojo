# Z-Image L2P DiT resident slices.
#
# Bounded pre-block model-math gate for the VAE-less L2P DiT. This loads real
# checkpoint weights for pixel patch embedding, timestep embedding, and caption
# embedding. It does not claim transformer block or full denoise coverage.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.math import (
    cos as fcos,
    exp as fexp,
    log as flog,
    sin as fsin,
    sqrt,
    tanh as ftanh,
)
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.dit.zimage_l2p_contract import (
    ZIMAGE_L2P_CAP_FEAT_DIM,
    ZIMAGE_L2P_HIDDEN,
    ZIMAGE_L2P_PATCH_SIZE,
    ZIMAGE_L2P_PATCH_VECTOR_DIM,
    ZIMAGE_L2P_PIXEL_CHANNELS,
    ZIMAGE_L2P_TIMESTEP_DIM,
    zimage_l2p_default_checkpoint_path,
    zimage_l2p_default_conditioning_path,
    validate_zimage_l2p_default_checkpoint_contract,
    zimage_l2p_model_timestep,
)
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import (
    add,
    add_scalar,
    mul,
    permute,
    reshape,
    slice,
)
from serenitymojo.tensor import Tensor


def _load_weight_bf16(
    ref st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view_as_bf16(tv, ctx)


def load_zimage_l2p_conditioning_bf16(
    embeddings_path: String, name: String, ctx: DeviceContext
) raises -> Tensor:
    if name != String("cap_feats") and name != String("cap_feats_uncond"):
        raise Error(
            String("Z-Image L2P conditioning tensor name must be cap_feats/cap_feats_uncond: ")
            + name
        )
    var st = SafeTensors.open(embeddings_path)
    var info = st.tensor_info(name)
    if info.dtype != STDtype.BF16 and info.dtype != STDtype.F32:
        raise Error(String("Z-Image L2P conditioning dtype mismatch for ") + name)
    if len(info.shape) != 3:
        raise Error(String("Z-Image L2P conditioning rank mismatch for ") + name)
    if (
        info.shape[0] != 1
        or info.shape[1] <= 0
        or info.shape[2] != ZIMAGE_L2P_CAP_FEAT_DIM
    ):
        raise Error(String("Z-Image L2P conditioning shape mismatch for ") + name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view_as_bf16(tv, ctx)


def load_zimage_l2p_default_conditioning_bf16(
    name: String, ctx: DeviceContext
) raises -> Tensor:
    return load_zimage_l2p_conditioning_bf16(
        zimage_l2p_default_conditioning_path(), name, ctx
    )


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


@fieldwise_init
struct ZImageL2PDiTPreBlockGate(Movable):
    var x_w: Tensor
    var x_b: Tensor
    var t_w0: Tensor
    var t_b0: Tensor
    var t_w2: Tensor
    var t_b2: Tensor
    var cap_norm_w: Tensor
    var cap_w: Tensor
    var cap_b: Tensor

    @staticmethod
    def load(
        checkpoint_path: String, ctx: DeviceContext
    ) raises -> ZImageL2PDiTPreBlockGate:
        validate_zimage_l2p_default_checkpoint_contract()
        var st = SafeTensors.open(checkpoint_path)
        return ZImageL2PDiTPreBlockGate(
            _load_weight_bf16(st, String("all_x_embedder.16-1.weight"), ctx),
            _load_weight_bf16(st, String("all_x_embedder.16-1.bias"), ctx),
            _load_weight_bf16(st, String("t_embedder.mlp.0.weight"), ctx),
            _load_weight_bf16(st, String("t_embedder.mlp.0.bias"), ctx),
            _load_weight_bf16(st, String("t_embedder.mlp.2.weight"), ctx),
            _load_weight_bf16(st, String("t_embedder.mlp.2.bias"), ctx),
            _load_weight_bf16(st, String("cap_embedder.0.weight"), ctx),
            _load_weight_bf16(st, String("cap_embedder.1.weight"), ctx),
            _load_weight_bf16(st, String("cap_embedder.1.bias"), ctx),
        )

    @staticmethod
    def load_default(ctx: DeviceContext) raises -> ZImageL2PDiTPreBlockGate:
        return ZImageL2PDiTPreBlockGate.load(
            zimage_l2p_default_checkpoint_path(), ctx
        )

    def patchify16_pixel[H: Int, W: Int](
        self, pixels_nchw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        comptime assert H % ZIMAGE_L2P_PATCH_SIZE == 0 and W % ZIMAGE_L2P_PATCH_SIZE == 0, "L2P patchify needs H/W divisible by 16"
        var sh = pixels_nchw.shape()
        if (
            len(sh) != 4
            or sh[0] != 1
            or sh[1] != ZIMAGE_L2P_PIXEL_CHANNELS
            or sh[2] != H
            or sh[3] != W
        ):
            raise Error("Z-Image L2P patchify16_pixel expects [1,3,H,W] NCHW")
        if pixels_nchw.dtype() != STDtype.BF16:
            raise Error("Z-Image L2P patchify16_pixel expects BF16 pixels")
        comptime P = ZIMAGE_L2P_PATCH_SIZE
        var ph = H // P
        var pw = W // P
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

    def pixel_embed[H: Int, W: Int](
        self, pixels_nchw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var patches = self.patchify16_pixel[H, W](pixels_nchw, ctx)
        return linear(
            patches,
            self.x_w,
            Optional[Tensor](_clone(self.x_b, ctx)),
            ctx,
        )

    def timestep_embed(self, sigma: Float32, ctx: DeviceContext) raises -> Tensor:
        var dim = ZIMAGE_L2P_TIMESTEP_DIM
        var half = dim // 2
        var max_period = Float32(10000.0)
        var scaled = zimage_l2p_model_timestep(sigma)
        var emb = List[Float32]()
        var log_mp = flog(max_period)
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
        var h = linear(
            t_freq,
            self.t_w0,
            Optional[Tensor](_clone(self.t_b0, ctx)),
            ctx,
        )
        h = silu(h, ctx)
        return linear(
            h,
            self.t_w2,
            Optional[Tensor](_clone(self.t_b2, ctx)),
            ctx,
        )

    def caption_embed[CAP: Int](
        self, cap_feats: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var sh = cap_feats.shape()
        if (
            len(sh) != 3
            or sh[0] != 1
            or sh[1] != CAP
            or sh[2] != ZIMAGE_L2P_CAP_FEAT_DIM
        ):
            raise Error("Z-Image L2P caption_embed expects [1,CAP,2560]")
        if cap_feats.dtype() != STDtype.BF16:
            raise Error("Z-Image L2P caption_embed expects BF16 cap_feats")
        var normed = rms_norm(cap_feats, self.cap_norm_w, Float32(1e-5), ctx)
        return linear(
            normed,
            self.cap_w,
            Optional[Tensor](_clone(self.cap_b, ctx)),
            ctx,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Block-0 transformer block (Wave 1 chunk 2 / first attention layer).
#
# Port of /home/alex/EriDiffusion/inference-flame/src/models/l2p/dit.rs
# transformer_block + joint_attention + swiglu. INFERENCE-ONLY. Compile gate
# only — not parity-checked yet.
#
# Checkpoint key reality (verified via safetensors header on disk):
#   layers.0.attention.to_q.weight   [3840, 3840] BF16   (NOT fused qkv)
#   layers.0.attention.to_k.weight   [3840, 3840] BF16
#   layers.0.attention.to_v.weight   [3840, 3840] BF16
#   layers.0.attention.to_out.0.weight [3840, 3840] BF16
#   layers.0.attention.norm_q.weight [128] BF16   (per-head Dh, NOT q_norm)
#   layers.0.attention.norm_k.weight [128] BF16   (per-head Dh, NOT k_norm)
#   layers.0.attention_norm1.weight  [3840] BF16
#   layers.0.attention_norm2.weight  [3840] BF16
#   layers.0.feed_forward.w{1,2,3}.weight (10240,3840 / 3840,10240 / 10240,3840) BF16
#   layers.0.ffn_norm1.weight        [3840] BF16
#   layers.0.ffn_norm2.weight        [3840] BF16
#   layers.0.adaLN_modulation.0.weight [15360, 256] BF16  (in: t_embedder dim=256)
#   layers.0.adaLN_modulation.0.bias   [15360]      BF16
#
# This DEVIATES from the Rust source's `qkv` fused naming — flame-core's
# weight loader rewrites split to_q/to_k/to_v into a fused .qkv. on read,
# but the on-disk file is split. The split-QKV layout matches zimage_dit.mojo
# (NextDiT) in this same repo and is the source of truth on disk.
#
# adaLN reads `t_cond` of shape [B, ZIMAGE_L2P_TIMESTEP_DIM=256] (the
# t_embedder MLP output). adaLN.0.weight projects 256 -> 4*dim=15360 which is
# chunked into 4 segments of `dim=3840`:
#     [0]=scale_msa  [1]=gate_msa  [2]=scale_mlp  [3]=gate_mlp
# matching dit.rs:541-543. (NOT 6-way; no shift terms in L2P.)


# Local comptime mirrors of contract head count / head_dim for the sdpa params
# (sdpa requires compile-time H / Dh constants).
comptime ZIMAGE_L2P_NUM_HEADS_C = 30
comptime ZIMAGE_L2P_HEAD_DIM_C = 128


# ─── tanh elementwise (gate activation) ──────────────────────────────────────
comptime _ZL2P_DYN1 = Layout.row_major(-1)
comptime _ZL2P_BLOCK = 256


def _zl2p_tanh_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _ZL2P_DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _ZL2P_DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](ftanh(v).cast[DType.bfloat16]())


def _zl2p_tanh(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Elementwise tanh (BF16-only here — L2P gates are always BF16)."""
    if x.dtype() != STDtype.BF16:
        raise Error("zimage_l2p _zl2p_tanh: expected BF16")
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_ZL2P_DYN1].row_major(IndexList[1](n))
    var grid = (n + _ZL2P_BLOCK - 1) // _ZL2P_BLOCK
    var X = LayoutTensor[DType.bfloat16, _ZL2P_DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var O = LayoutTensor[DType.bfloat16, _ZL2P_DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    ctx.enqueue_function[_zl2p_tanh_kernel_bf16, _zl2p_tanh_kernel_bf16](
        X, O, n, grid_dim=grid, block_dim=_ZL2P_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# Replicate rope cos/sin [S, half] -> [S*H, half] for BSHD rope_interleaved.
#
# Rust applies RoPE in BHSD `[B,H,S,Dh]` with cos/sin reshaped to
# `[1,1,S,half]` which auto-broadcasts across (B,H). Mojo's `rope_interleaved`
# flattens leading dims of `[B,S,H,Dh]` to rows `[B*S*H, Dh]` and indexes
# cos/sin row-by-row, so we materialize cos/sin pre-replicated along H. This
# is a host roundtrip — fine at smoke sizes (e.g. S=96,H=30,half=64 -> 184K
# floats per table). FLAG: replace with a device-side repeat kernel at native
# 1024² (S=4416, ~8.5M floats/table).
def _zl2p_replicate_rope_for_heads(
    rope_2d: Tensor, S: Int, H: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    var sh = rope_2d.shape()
    if len(sh) != 2 or sh[0] != S or sh[1] != half:
        raise Error("zimage_l2p _replicate_rope_for_heads: shape mismatch")
    if rope_2d.dtype() != STDtype.BF16:
        raise Error("zimage_l2p _replicate_rope_for_heads: expected BF16")
    var src = rope_2d.to_host(ctx)  # length S*half (F32 upcast)
    var dst = List[Float32]()
    for s in range(S):
        var base = s * half
        for _h in range(H):
            for i in range(half):
                dst.append(src[base + i])
    var out_sh = List[Int]()
    out_sh.append(S * H)
    out_sh.append(half)
    return Tensor.from_host(dst, out_sh^, STDtype.BF16, ctx)


@fieldwise_init
struct ZImageL2PBlockWeights(Movable):
    """One transformer-block weight bundle (block i of `layers.<i>`).

    Per-block keys are loaded eagerly from the safetensors file. Weights stay
    on device (BF16 throughout). NOTE: the checkpoint stores SPLIT
    `to_q/to_k/to_v` weights — see header comment above for the full key
    list. There is NO fused `attention.qkv.weight` on disk.
    """

    var attention_norm1_w: Tensor       # [dim]
    var to_q_w: Tensor                  # [dim, dim]
    var to_k_w: Tensor                  # [dim, dim]
    var to_v_w: Tensor                  # [dim, dim]
    var attention_norm_q_w: Tensor      # [head_dim]   (per-head RMSNorm)
    var attention_norm_k_w: Tensor      # [head_dim]
    var attention_to_out_w: Tensor      # [dim, dim]
    var attention_norm2_w: Tensor       # [dim]
    var ffn_norm1_w: Tensor             # [dim]
    var ff_w1: Tensor                   # [hidden, dim]
    var ff_w2: Tensor                   # [dim, hidden]
    var ff_w3: Tensor                   # [hidden, dim]
    var ffn_norm2_w: Tensor             # [dim]
    var adaLN_mod_w: Tensor             # [4*dim, t_embedder_dim=256]
    var adaLN_mod_b: Tensor             # [4*dim]

    @staticmethod
    def load(
        checkpoint_path: String, prefix: String, ctx: DeviceContext
    ) raises -> ZImageL2PBlockWeights:
        var st = SafeTensors.open(checkpoint_path)
        return ZImageL2PBlockWeights.load_from_st(st, prefix, ctx)

    @staticmethod
    def load_from_st(
        ref st: SafeTensors, prefix: String, ctx: DeviceContext
    ) raises -> ZImageL2PBlockWeights:
        return ZImageL2PBlockWeights(
            _load_weight_bf16(st, prefix + String(".attention_norm1.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.to_q.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.to_k.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.to_v.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.norm_q.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.norm_k.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.to_out.0.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention_norm2.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".ffn_norm1.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".feed_forward.w1.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".feed_forward.w2.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".feed_forward.w3.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".ffn_norm2.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".adaLN_modulation.0.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".adaLN_modulation.0.bias"), ctx),
        )


# ─── block forward ──────────────────────────────────────────────────────────
# Port of dit.rs:510-629 transformer_block (with adaLN active — block 0 always
# has t_cond available). Math layout:
#
#   mod_out = linear(t_cond, adaLN_mod_w, adaLN_mod_b)   # [B, 4*dim]
#   scale_msa, gate_msa, scale_mlp, gate_mlp = chunk(mod_out, 4)   # [B, dim]
#
#   x_norm   = rms_norm(x, attention_norm1_w, eps)
#   x_norm  *= (1 + scale_msa).unsqueeze(1)              # broadcast over S
#   attn_out = joint_attention(x_norm, rope_cos, rope_sin)
#   attn_out = rms_norm(attn_out, attention_norm2_w, eps)
#   x       += tanh(gate_msa).unsqueeze(1) * attn_out
#
#   ff_in    = rms_norm(x, ffn_norm1_w, eps)
#   ff_in   *= (1 + scale_mlp).unsqueeze(1)
#   ff_out   = w2( silu(w1(ff_in)) * w3(ff_in) )         # SwiGLU
#   ff_out   = rms_norm(ff_out, ffn_norm2_w, eps)
#   x       += tanh(gate_mlp).unsqueeze(1) * ff_out
#
# No negation / sign-flip here — that lives in `forward_inner` of the Rust
# pipeline (out *= -1 and t = (1000 - t_in)/1000), NOT in the block.


def _zl2p_joint_attention[B: Int, S: Int](
    weights: ZImageL2PBlockWeights,
    x: Tensor,                  # [B, S, dim]
    rope_cos_2d: Tensor,        # [S, head_dim/2]
    rope_sin_2d: Tensor,        # [S, head_dim/2]
    num_heads: Int,
    head_dim: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Single-stream joint attention, BSHD via existing serenitymojo ops.

    Port of dit.rs:421-506 `joint_attention`. SPLIT to_q/to_k/to_v weights.
    Output projection via `attention.to_out.0.weight`. RoPE uses the existing
    `rope_interleaved` op (matches Rust `RopeLayout::Interleaved`).
    """
    var dim = num_heads * head_dim
    var half = head_dim // 2
    var scale = Float32(1.0) / sqrt(Float32(head_dim))

    # qkv projections (split): [B, S, dim] each.
    var q = linear(x, weights.to_q_w, None, ctx)
    var k = linear(x, weights.to_k_w, None, ctx)
    var v = linear(x, weights.to_v_w, None, ctx)

    # BSHD reshape: [B, S, H, Dh]
    var sh = List[Int]()
    sh.append(B)
    sh.append(S)
    sh.append(num_heads)
    sh.append(head_dim)
    q = reshape(q, sh.copy(), ctx)
    k = reshape(k, sh.copy(), ctx)
    v = reshape(v, sh.copy(), ctx)

    # Per-head RMSNorm over Dh on Q and K (norm_q / norm_k, eps=1e-5).
    # Note: Rust flattens to [B*S*H, Dh] before RMS; rms_norm here normalizes
    # over the last dim regardless of rank, so a direct call is equivalent.
    q = rms_norm(q, weights.attention_norm_q_w, eps, ctx)
    k = rms_norm(k, weights.attention_norm_k_w, eps, ctx)

    # RoPE — replicate cos/sin across H heads first ([S, half] -> [S*H, half])
    # since rope_interleaved consumes flat rows. Only Q/K are rotated; V is not.
    # FLAG (for skeptic): the host-roundtrip replication is correct but
    # inefficient; replace with a device-side broadcast at native sequence
    # lengths.
    var cos_rep = _zl2p_replicate_rope_for_heads(rope_cos_2d, S, num_heads, half, ctx)
    var sin_rep = _zl2p_replicate_rope_for_heads(rope_sin_2d, S, num_heads, half, ctx)
    q = rope_interleaved(q, cos_rep, sin_rep, ctx)
    k = rope_interleaved(k, cos_rep, sin_rep, ctx)

    # Comptime-shaped sdpa_nomask. Rust passes mask=None — the sdpa_nomask
    # path matches that exactly (no [B,H,S,S] zeros mask materialization).
    var attn = sdpa_nomask[B, S, ZIMAGE_L2P_NUM_HEADS_C, ZIMAGE_L2P_HEAD_DIM_C](
        q, k, v, scale, ctx
    )
    # attn is BSHD [B,S,H,Dh] -> [B,S,dim]
    var asz = List[Int]()
    asz.append(B)
    asz.append(S)
    asz.append(dim)
    attn = reshape(attn, asz^, ctx)
    return linear(attn, weights.attention_to_out_w, None, ctx)


def zimage_l2p_block_forward[B: Int, S: Int](
    weights: ZImageL2PBlockWeights,
    x: Tensor,                  # [B, S, dim] BF16
    rope_cos: Tensor,           # [S, head_dim/2] BF16
    rope_sin: Tensor,           # [S, head_dim/2] BF16
    t_cond: Tensor,             # [B, t_embedder_dim=256] BF16
    num_heads: Int,
    head_dim: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Block 0 forward — single transformer block with adaLN gating.

    Mirrors `transformer_block` in dit.rs:510-629. Returns x updated by both
    the attention and FFN residual branches (gate-residual fused).
    """
    var dim = num_heads * head_dim
    if dim != ZIMAGE_L2P_HIDDEN:
        raise Error("zimage_l2p_block_forward: num_heads*head_dim != hidden")

    # adaLN: t_cond [B, 256] -> mod_out [B, 4*dim].
    var mod_out = linear(
        t_cond,
        weights.adaLN_mod_w,
        Optional[Tensor](_clone(weights.adaLN_mod_b, ctx)),
        ctx,
    )
    # chunk(4, last_dim) along axis 1.
    var scale_msa = slice(mod_out, 1, 0 * dim, dim, ctx)
    var gate_msa = slice(mod_out, 1, 1 * dim, dim, ctx)
    var scale_mlp = slice(mod_out, 1, 2 * dim, dim, ctx)
    var gate_mlp = slice(mod_out, 1, 3 * dim, dim, ctx)

    # tanh on gate_msa / gate_mlp (per dit.rs:582 / :615).
    var g_msa = _zl2p_tanh(gate_msa, ctx)
    var g_mlp = _zl2p_tanh(gate_mlp, ctx)

    # Reshape scale / gate from [B, dim] -> [B, 1, dim] for broadcast over S.
    var b1d = List[Int]()
    b1d.append(B)
    b1d.append(1)
    b1d.append(dim)
    var scale_msa_b = add_scalar(scale_msa, Float32(1.0), ctx)
    scale_msa_b = reshape(scale_msa_b, b1d.copy(), ctx)
    var scale_mlp_b = add_scalar(scale_mlp, Float32(1.0), ctx)
    scale_mlp_b = reshape(scale_mlp_b, b1d.copy(), ctx)
    g_msa = reshape(g_msa, b1d.copy(), ctx)
    g_mlp = reshape(g_mlp, b1d.copy(), ctx)

    # Attention branch.
    var x_norm = rms_norm(x, weights.attention_norm1_w, eps, ctx)
    x_norm = mul(x_norm, scale_msa_b, ctx)
    var attn_out = _zl2p_joint_attention[B, S](
        weights, x_norm, rope_cos, rope_sin, num_heads, head_dim, eps, ctx
    )
    attn_out = rms_norm(attn_out, weights.attention_norm2_w, eps, ctx)
    var gated_attn = mul(g_msa, attn_out, ctx)
    var x_after_attn = add(x, gated_attn, ctx)

    # FFN branch.
    var ff_in = rms_norm(x_after_attn, weights.ffn_norm1_w, eps, ctx)
    ff_in = mul(ff_in, scale_mlp_b, ctx)
    var w1_out = linear(ff_in, weights.ff_w1, None, ctx)
    var w3_out = linear(ff_in, weights.ff_w3, None, ctx)
    var hidden = swiglu(w1_out, w3_out, ctx)  # silu(w1) * w3
    var ff_out = linear(hidden, weights.ff_w2, None, ctx)
    ff_out = rms_norm(ff_out, weights.ffn_norm2_w, eps, ctx)
    var gated_ff = mul(g_mlp, ff_out, ctx)
    return add(x_after_attn, gated_ff, ctx)


# ─── Context-refiner block (no adaLN) ────────────────────────────────────────
# The context_refiner blocks do NOT have adaLN_modulation keys. They have
# identical attention + FFN structure to the main blocks but skip the modulation
# gates entirely (scale=0, gate=0 → equivalent to identity-scale + zero-gate).
# Actually the Rust `has_adaln=false` path just uses `x.add(attn_out)` and
# `x.add(ff_out)` with no scaling/gating at all.

@fieldwise_init
struct ZImageL2PContextBlockWeights(Movable):
    """Context-refiner transformer block — same attention+FFN, no adaLN."""

    var attention_norm1_w: Tensor
    var to_q_w: Tensor
    var to_k_w: Tensor
    var to_v_w: Tensor
    var attention_norm_q_w: Tensor
    var attention_norm_k_w: Tensor
    var attention_to_out_w: Tensor
    var attention_norm2_w: Tensor
    var ffn_norm1_w: Tensor
    var ff_w1: Tensor
    var ff_w2: Tensor
    var ff_w3: Tensor
    var ffn_norm2_w: Tensor

    @staticmethod
    def load(
        checkpoint_path: String, prefix: String, ctx: DeviceContext
    ) raises -> ZImageL2PContextBlockWeights:
        var st = SafeTensors.open(checkpoint_path)
        return ZImageL2PContextBlockWeights.load_from_st(st, prefix, ctx)

    @staticmethod
    def load_from_st(
        ref st: SafeTensors, prefix: String, ctx: DeviceContext
    ) raises -> ZImageL2PContextBlockWeights:
        return ZImageL2PContextBlockWeights(
            _load_weight_bf16(st, prefix + String(".attention_norm1.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.to_q.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.to_k.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.to_v.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.norm_q.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.norm_k.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention.to_out.0.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".attention_norm2.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".ffn_norm1.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".feed_forward.w1.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".feed_forward.w2.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".feed_forward.w3.weight"), ctx),
            _load_weight_bf16(st, prefix + String(".ffn_norm2.weight"), ctx),
        )


def _zl2p_ctx_joint_attention[B: Int, S: Int](
    weights: ZImageL2PContextBlockWeights,
    x: Tensor,
    rope_cos_2d: Tensor,
    rope_sin_2d: Tensor,
    num_heads: Int,
    head_dim: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var dim = num_heads * head_dim
    var half = head_dim // 2
    var scale = Float32(1.0) / sqrt(Float32(head_dim))

    var q = linear(x, weights.to_q_w, None, ctx)
    var k = linear(x, weights.to_k_w, None, ctx)
    var v = linear(x, weights.to_v_w, None, ctx)

    var sh = List[Int]()
    sh.append(B)
    sh.append(S)
    sh.append(num_heads)
    sh.append(head_dim)
    q = reshape(q, sh.copy(), ctx)
    k = reshape(k, sh.copy(), ctx)
    v = reshape(v, sh.copy(), ctx)

    q = rms_norm(q, weights.attention_norm_q_w, eps, ctx)
    k = rms_norm(k, weights.attention_norm_k_w, eps, ctx)

    var cos_rep = _zl2p_replicate_rope_for_heads(rope_cos_2d, S, num_heads, half, ctx)
    var sin_rep = _zl2p_replicate_rope_for_heads(rope_sin_2d, S, num_heads, half, ctx)
    q = rope_interleaved(q, cos_rep, sin_rep, ctx)
    k = rope_interleaved(k, cos_rep, sin_rep, ctx)

    var attn = sdpa_nomask[B, S, ZIMAGE_L2P_NUM_HEADS_C, ZIMAGE_L2P_HEAD_DIM_C](
        q, k, v, scale, ctx
    )
    var asz = List[Int]()
    asz.append(B)
    asz.append(S)
    asz.append(dim)
    attn = reshape(attn, asz^, ctx)
    return linear(attn, weights.attention_to_out_w, None, ctx)


def zimage_l2p_context_block_forward[B: Int, S: Int](
    weights: ZImageL2PContextBlockWeights,
    x: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    num_heads: Int,
    head_dim: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Context-refiner block forward — no adaLN, no gating, pure residual.

    Mirrors `transformer_block` with has_adaln=False (Rust dit.rs:534-544).
    Returns x + attn_residual + ffn_residual.
    """
    # Attention branch (no scale/gate)
    var x_norm = rms_norm(x, weights.attention_norm1_w, eps, ctx)
    var attn_out = _zl2p_ctx_joint_attention[B, S](
        weights, x_norm, rope_cos, rope_sin, num_heads, head_dim, eps, ctx
    )
    attn_out = rms_norm(attn_out, weights.attention_norm2_w, eps, ctx)
    var x_after_attn = add(x, attn_out, ctx)

    # FFN branch (no scale/gate)
    var ff_in = rms_norm(x_after_attn, weights.ffn_norm1_w, eps, ctx)
    var w1_out = linear(ff_in, weights.ff_w1, None, ctx)
    var w3_out = linear(ff_in, weights.ff_w3, None, ctx)
    var hidden = swiglu(w1_out, w3_out, ctx)
    var ff_out = linear(hidden, weights.ff_w2, None, ctx)
    ff_out = rms_norm(ff_out, weights.ffn_norm2_w, eps, ctx)
    return add(x_after_attn, ff_out, ctx)
