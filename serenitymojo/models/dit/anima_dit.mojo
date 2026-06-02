# models/dit/anima_dit.mojo — Anima MiniTrainDIT forward pass (GPU, inference-only).
#
# Port of inference-flame/src/models/anima.rs MiniTrainDIT, faithful line-by-line.
#
# Architecture summary (AnimaConfig::default()):
#   - hidden=2048, 28 blocks, 16 heads, head_dim=128
#   - Patch embed: Linear(68, 2048, no bias)
#     (16 latent + 1 mask) * 2 * 2 = 68 in
#   - Timestep:
#       sinusoidal(2048) -> Linear(2048->2048, no bias) -> SiLU -> hidden
#       hidden -> Linear(2048->6144, no bias) -> base_adaln  [B, 6144]
#       RMSNorm(sinusoidal, weight=t_embedding_norm) -> t_cond  [B, 2048]
#   - Per block (3 AdaLN-LoRA sub-blocks: self-attn, cross-attn, MLP):
#       adaln_mod = SiLU(t_cond) @ W1[2048->256] @ W2[256->6144] + base_adaln
#       chunk3 -> (shift, scale, gate) each [B, 2048]
#       apply_adaln: LayerNorm(x) * (1+scale) + shift   (modulate_pre, eps=1e-6)
#       gate at residual: x = x + gate.unsqueeze(1) * sub_block_output
#   - Self-attn: QKV no-bias, RMSNorm-per-head [B,H,S,D], 3D RoPE halfsplit, SDPA
#   - Cross-attn: Q(2048), K/V(1024->2048) no-bias, RMSNorm-per-head [B,S,H,D], SDPA
#   - MLP: Linear(2048->8192) -> GELU -> Linear(8192->2048), all no-bias
#   - Final layer: 2-output AdaLN, Linear(2048->64)
#   - Patchify: append zero mask channel, permute, flatten -> [B,N,68]
#   - Unpatchify: [B,N,64] -> [B,T,H,W,16]
#
# Data layout throughout: [B, T*nH*nW, 2048] (2048 = hidden)
# F32 residual stream within each block (BF16 for weight ops).
#
# NOTE: head_dim=128, so sdpa_nomask[B,S,16,128] — uses math-mode (flash fails
# on sm_86 for Dh=128, confirmed from attention.mojo comments).

from std.math import exp as fexp, log as flog, cos as fcos, sin as fsin, sqrt as fsqrt, pow as fpow
from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu, gelu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul, mul_scalar, add_scalar, reshape, permute, concat, slice

from serenitymojo.models.dit.anima_contract import (
    ANIMA_HIDDEN,
    ANIMA_DEPTH,
    ANIMA_NUM_HEADS,
    ANIMA_HEAD_DIM,
    ANIMA_MLP_HIDDEN,
    ANIMA_ADALN_LORA_DIM,
    ANIMA_ADALN_DIM,
    ANIMA_ADAPTER_DIM,
    ANIMA_PATCH_IN_DIM,
    ANIMA_PATCH_OUT_DIM,
    ANIMA_LATENT_CHANNELS,
    ANIMA_PATCH_SIZE,
    ANIMA_IMAGE_TOKENS,
    ANIMA_DIT_PATH,
)


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256


# ── Named return types (Tensor is Movable-not-Copyable; Tuple[Tensor,Tensor] ──
# field extraction requires transfer semantics not supported by [i] subscript)

struct _TensorPair(Movable):
    var first: Tensor
    var second: Tensor

    def __init__(out self, var first: Tensor, var second: Tensor):
        self.first = first^
        self.second = second^

    def __del__(deinit self):
        pass


struct _TimestepResult(Movable):
    var t_cond: Tensor
    var base_adaln: Tensor

    def __init__(out self, var t_cond: Tensor, var base_adaln: Tensor):
        self.t_cond = t_cond^
        self.base_adaln = base_adaln^

    def __del__(deinit self):
        pass


struct _AdaLNMods(Movable):
    var shift: Tensor
    var scale: Tensor
    var gate: Tensor

    def __init__(out self, var shift: Tensor, var scale: Tensor, var gate: Tensor):
        self.shift = shift^
        self.scale = scale^
        self.gate = gate^

    def __del__(deinit self):
        pass

# ── seq lengths (compile-time for sdpa_nomask dispatch) ─────────────────────
# 1024x1024 image: latent 128x128, patch 2x2, patch_grid 64x64, T=1
# image_tokens = 1*64*64 = 4096
comptime ANIMA_N = ANIMA_IMAGE_TOKENS  # 4096 image patches
# context sequence length = 256 (adapter dim output)
comptime ANIMA_S_TXT = 256


# ── sinusoidal timestep embedding (COS first, same as anima.rs prepare_timestep) ─
# anima.rs uses cos first then sin (same as zimage_nextdit.rs / embeddings.mojo)
def _anima_sinusoidal_kernel_f32(
    t: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    n: Int,
    half: Int,
    neg_ln_max_period: Float32,
):
    var idx = Int(global_idx.x)
    var total = n * half
    if idx < total:
        var row = idx // half
        var i = idx % half
        var tv = rebind[Scalar[DType.float32]](t[row])
        var freq = fexp(neg_ln_max_period * (Float32(i) / Float32(half)))
        var angle = tv * freq
        o[row, i] = rebind[o.element_type](fcos(angle))
        o[row, half + i] = rebind[o.element_type](fsin(angle))


def _anima_sinusoidal_emb(
    t: Tensor, dim: Int, ctx: DeviceContext
) raises -> Tensor:
    """Sinusoidal timestep embedding [N] -> [N, dim], cos-first (matches Rust)."""
    if t.dtype() != STDtype.F32:
        raise Error("_anima_sinusoidal_emb: t must be F32")
    var n = t.numel()
    var half = dim // 2
    var neg_ln = -flog(Float32(10000.0))
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * dim * 4)
    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var o_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n, dim))
    var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        t.buf.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), o_rl
    )
    var total = n * half
    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_anima_sinusoidal_kernel_f32, _anima_sinusoidal_kernel_f32](
        T, O, n, half, neg_ln, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var sh = List[Int]()
    sh.append(n)
    sh.append(dim)
    return Tensor(out_buf^, sh^, STDtype.F32)


# ── AdaLN: modulate(x, shift, scale) = LayerNorm(x, eps=1e-6) * (1+scale) + shift ──
# x: [B,S,D] BF16, shift/scale: [B,D] BF16 -> output [B,S,D] BF16
# We do: rms_norm(x) on [B*S, D], then per-elem scale/shift with broadcasting.
# Actually Rust uses `modulate_pre_fused_bf16` which is LayerNorm (RMSNorm with eps=1e-6),
# NOT the usual AdaLN with learnable scale/shift — it's shift=(0,scale from modulation).
# From Rust: modulate_pre_fused_bf16(x, shift, scale, eps) = (1+scale)*rms_norm(x,eps) + shift

def _adaln_modulate_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],  # [B*S, D]
    shift: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [B, D] (broadcast over S)
    scale: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [B, D]
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],  # [B*S, D]
    rows: Int,  # B*S
    cols: Int,  # D
    batch: Int,  # B
    seq: Int,    # S
    eps: Float32,
):
    # One block per row; computes LayerNorm (mean + variance) then applies shift+scale.
    # Rust `modulate_pre_fused_bf16` is layernorm (not rms_norm) based on Cosmos DiT convention.
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    # Need 2 shared arrays: one for mean, one for var pass
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var b = row // seq  # batch index

    # Pass 1: compute mean
    var lsum: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        lsum += v
        c += _TPB
    shared[tid] = lsum
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var mean = shared[0] / Float32(cols)
    barrier()

    # Pass 2: compute variance
    var lvar: Float32 = 0.0
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        var d = v - mean
        lvar += d * d
        c += _TPB
    shared[tid] = lvar
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / fsqrt(shared[0] / Float32(cols) + eps)
    barrier()

    # Write: out = (1+scale[b,c]) * layer_normed(x[row,c]) + shift[b,c]
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        var normed = (v - mean) * inv
        var sc = rebind[Scalar[DType.float32]](scale[b, c])
        var sh = rebind[Scalar[DType.float32]](shift[b, c])
        var out_v = (1.0 + sc) * normed + sh
        o[row, c] = rebind[o.element_type](out_v.cast[DType.bfloat16]())
        c += _TPB


def _apply_adaln_modulate(
    x: Tensor,      # [B, S, D] BF16
    shift: Tensor,  # [B, D] BF16
    scale: Tensor,  # [B, D] BF16
    ctx: DeviceContext,
) raises -> Tensor:
    """Apply AdaLN modulation: out = (1+scale)*RMSNorm(x,eps=1e-6) + shift.
    shift/scale are [B,D]; x is [B,S,D]. Broadcast over seq dimension."""
    var xsh = x.shape()
    var b = xsh[0]
    var s = xsh[1]
    var d = xsh[2]
    # Cast shift/scale to F32 for F32 arithmetic
    var shift_f32 = cast_tensor(shift, STDtype.F32, ctx)
    var scale_f32 = cast_tensor(scale, STDtype.F32, ctx)
    var rows = b * s

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](rows * d * 2)  # BF16
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, d))
    var bd_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](b, d))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var Sh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        shift_f32.buf.unsafe_ptr().bitcast[Float32](), bd_rl
    )
    var Sc = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        scale_f32.buf.unsafe_ptr().bitcast[Float32](), bd_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    ctx.enqueue_function[_adaln_modulate_kernel_bf16, _adaln_modulate_kernel_bf16](
        X, Sh, Sc, O, rows, d, b, s, Float32(1e-6),
        grid_dim=rows, block_dim=_TPB
    )
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(b)
    osh.append(s)
    osh.append(d)
    return Tensor(out_buf^, osh^, STDtype.BF16)


# ── 3D RoPE cos/sin construction (matches build_3d_rope_cossin in anima.rs) ───
# Returns (cos, sin) each [1, 1, S, D/2] BF16
# Cosmos Predict2 3D RoPE: halfsplit, axis-split frequencies (t/h/w),
# NTK-scaled thetas (h_extra=4.0, w_extra=4.0, t_extra=1.0 for 16ch model).
def build_anima_3d_rope(
    t_frames: Int,
    nh: Int,
    nw: Int,
    head_dim: Int,
    ctx: DeviceContext,
) raises -> _TensorPair:
    """Build 3D RoPE cos/sin tables for Anima self-attention.
    Returns _TensorPair(cos, sin) each [1,1,S,D/2] in F32.
    Matches build_3d_rope_cossin() in anima.rs exactly.
    """
    var half_d = head_dim // 2  # 64
    var total_seq = t_frames * nh * nw

    var full_d = half_d * 2  # 128
    # Cosmos 3D split: dim_h = full_d//6*2 = 128//6*2 = 42, dim_w=42, dim_t=44
    var dim_h = full_d // 6 * 2   # 42
    var dim_w = dim_h              # 42
    var dim_t = full_d - 2 * dim_h  # 44
    var bins_t = dim_t // 2   # 22
    var bins_h = dim_h // 2   # 21
    var bins_w = dim_w // 2   # 21

    # NTK-scaled thetas (extrapolation_ratio for 16ch model: h/w=4.0, t=1.0)
    # h_ntk = 4.0^(dim_h/(dim_h-2)), computed via exp(log(4)*exponent)
    var base_theta: Float64 = 10000.0
    var h_exp = Float64(dim_h) / (Float64(dim_h) - 2.0)
    var w_exp = Float64(dim_w) / (Float64(dim_w) - 2.0)
    var _ = Float64(dim_t) / (Float64(dim_t) - 2.0)
    # t_extra=1.0 so t_ntk = 1.0^anything = 1.0
    var h_ntk = fexp(flog(Float64(4.0)) * h_exp)
    var w_ntk = fexp(flog(Float64(4.0)) * w_exp)
    var t_ntk = Float64(1.0)  # 1.0^any = 1.0
    var theta_h = Float32(base_theta * h_ntk)
    var theta_w = Float32(base_theta * w_ntk)
    var theta_t = Float32(base_theta * t_ntk)

    # Build frequencies on host: freq = 1 / theta^(2i/dim)
    var freqs_t = List[Float32]()
    for i in range(bins_t):
        var exp_val = Float32(2 * i) / Float32(dim_t)
        freqs_t.append(1.0 / fpow(theta_t, exp_val))
    var freqs_h = List[Float32]()
    for i in range(bins_h):
        var exp_val = Float32(2 * i) / Float32(dim_h)
        freqs_h.append(1.0 / fpow(theta_h, exp_val))
    var freqs_w = List[Float32]()
    for i in range(bins_w):
        var exp_val = Float32(2 * i) / Float32(dim_w)
        freqs_w.append(1.0 / fpow(theta_w, exp_val))

    # Build [total_seq, half_d] cos/sin tables on CPU, then upload
    # The Rust code doubles angles (two identical halves), then takes first half.
    # Result is [S, half_d] where half_d = bins_t + bins_h + bins_w = 64.
    var cos_data = List[Float32]()
    var sin_data = List[Float32]()

    for tf in range(t_frames):
        for ih in range(nh):
            for iw in range(nw):
                var dim_off = 0
                for fi in range(bins_t):
                    var angle = Float32(tf) * freqs_t[fi]
                    cos_data.append(fcos(angle))
                    sin_data.append(fsin(angle))
                dim_off += bins_t
                for fi in range(bins_h):
                    var angle = Float32(ih) * freqs_h[fi]
                    cos_data.append(fcos(angle))
                    sin_data.append(fsin(angle))
                dim_off += bins_h
                for fi in range(bins_w):
                    var angle = Float32(iw) * freqs_w[fi]
                    cos_data.append(fcos(angle))
                    sin_data.append(fsin(angle))

    # Shape [1, 1, total_seq, half_d] for rope_halfsplit_bf16 format
    var cos_sh = List[Int]()
    cos_sh.append(1)
    cos_sh.append(1)
    cos_sh.append(total_seq)
    cos_sh.append(half_d)
    var sin_sh = cos_sh.copy()

    var cos_t = Tensor.from_host(cos_data^, cos_sh^, STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin_data^, sin_sh^, STDtype.F32, ctx)
    return _TensorPair(cos_t^, sin_t^)


# ── RoPE halfsplit on [B, S, H, D] (BF16) ────────────────────────────────────
# rope_cos/sin: [1, 1, S, D/2] F32 (broadcast over B and H).
# out[b,s,h,i]      = x[b,s,h,i]*cos[s,i]     - x[b,s,h,i+D/2]*sin[s,i]
# out[b,s,h,i+D/2]  = x[b,s,h,i+D/2]*cos[s,i] + x[b,s,h,i]*sin[s,i]
def _rope_bshd_halfsplit_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],   # [B*S*H, D]
    c: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],    # [S, D/2]
    s: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],    # [S, D/2]
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],   # [B*S*H, D]
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    half_d: Int,
):
    var idx = Int(global_idx.x)
    var total = B * S * H * half_d
    if idx < total:
        var pair_idx = idx % half_d
        var bsh_idx = idx // half_d
        var si = (bsh_idx // H) % S   # position index -> cos/sin row
        var row = bsh_idx             # [B*S*H] flat row in x
        var x1 = rebind[Scalar[DType.bfloat16]](x[row, pair_idx]).cast[DType.float32]()
        var x2 = rebind[Scalar[DType.bfloat16]](x[row, pair_idx + half_d]).cast[DType.float32]()
        var cv = rebind[Scalar[DType.float32]](c[si, pair_idx])
        var sv = rebind[Scalar[DType.float32]](s[si, pair_idx])
        var out1 = x1 * cv - x2 * sv
        var out2 = x2 * cv + x1 * sv
        o[row, pair_idx] = rebind[o.element_type](out1.cast[DType.bfloat16]())
        o[row, pair_idx + half_d] = rebind[o.element_type](out2.cast[DType.bfloat16]())


def _rope_halfsplit_4d(
    x: Tensor,       # [B, S, H, D] BF16
    rope_cos: Tensor, # [1, 1, S, D/2] F32
    rope_sin: Tensor, # [1, 1, S, D/2] F32
    ctx: DeviceContext,
) raises -> Tensor:
    """Apply halfsplit RoPE to [B,S,H,D] BF16 tensor."""
    var xsh = x.shape()
    var B = xsh[0]
    var S = xsh[1]
    var H = xsh[2]
    var D = xsh[3]
    var half_d = D // 2
    var rows = B * S * H

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](rows * D * 2)  # BF16
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, D))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](S, half_d))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        rope_cos.buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    var Sv = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        rope_sin.buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var ntotal = B * S * H * half_d
    var grid = (ntotal + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _rope_bshd_halfsplit_kernel_bf16, _rope_bshd_halfsplit_kernel_bf16
    ](X, C, Sv, O, B, S, H, D, half_d, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(B)
    osh.append(S)
    osh.append(H)
    osh.append(D)
    return Tensor(out_buf^, osh^, STDtype.BF16)


# ── Patchify kernel: [B,T,H,W,C+1] -> [B,T,nH,nW,C+1,pH,pW] ─────────────────
# We implement patchify as a custom kernel to avoid a sequence of permutes.
# Input: [B,T,H,W,C_pad] where C_pad = C+1 = 17
# Output: [B, T*nH*nW, patch_dim] where patch_dim = C_pad * pH * pW
# Rust permute: reshape to [B,T,nH,pH,nW,pW,C_pad] then permute [0,1,2,4,6,3,5]
# -> [B,T,nH,nW,C_pad,pH,pW], then flatten to [B,T*nH*nW, C_pad*pH*pW]
def _patchify_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # flat [B*T*H*W*C_pad]
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # flat [B*N*patch_dim]
    B: Int, T: Int, nH: Int, nW: Int, pH: Int, pW: Int, Cp: Int,
):
    # Output index: flat index into [B, T*nH*nW, Cp*pH*pW]
    var idx = Int(global_idx.x)
    var patch_dim = Cp * pH * pW
    var total = B * T * nH * nW * patch_dim
    if idx < total:
        # Decode output index [b, t*nH*nW, Cp*pH*pW]
        var c_ph_pw = idx % patch_dim
        var t_patch_idx = idx // patch_dim  # b * T*nH*nW + t*nH*nW + patch_n
        var b = t_patch_idx // (T * nH * nW)
        var t_nhw = t_patch_idx % (T * nH * nW)
        var t_idx = t_nhw // (nH * nW)
        var hw_idx = t_nhw % (nH * nW)
        var h_patch = hw_idx // nW   # ih (which patch row)
        var w_patch = hw_idx % nW    # iw (which patch col)
        # Decode patch dim: [Cp, pH, pW] -> c, ph, pw
        var pw_idx = c_ph_pw % pW
        var c_ph = c_ph_pw // pW
        var ph_idx = c_ph % pH
        var c_idx = c_ph // pH
        # Source index in [B,T,H,W,Cp]:
        # h = h_patch * pH + ph_idx, w = w_patch * pW + pw_idx
        var h = h_patch * pH + ph_idx
        var w = w_patch * pW + pw_idx
        var H = nH * pH
        var W = nW * pW
        var src = ((b * T + t_idx) * H + h) * W * Cp + w * Cp + c_idx
        o[idx] = rebind[o.element_type](x[src])


def _patchify_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    B: Int, T: Int, nH: Int, nW: Int, pH: Int, pW: Int, Cp: Int,
):
    var idx = Int(global_idx.x)
    var patch_dim = Cp * pH * pW
    var total = B * T * nH * nW * patch_dim
    if idx < total:
        var c_ph_pw = idx % patch_dim
        var t_patch_idx = idx // patch_dim
        var b = t_patch_idx // (T * nH * nW)
        var t_nhw = t_patch_idx % (T * nH * nW)
        var t_idx = t_nhw // (nH * nW)
        var hw_idx = t_nhw % (nH * nW)
        var h_patch = hw_idx // nW
        var w_patch = hw_idx % nW
        var pw_idx = c_ph_pw % pW
        var c_ph = c_ph_pw // pW
        var ph_idx = c_ph % pH
        var c_idx = c_ph // pH
        var h = h_patch * pH + ph_idx
        var w = w_patch * pW + pw_idx
        var H = nH * pH
        var W = nW * pW
        var src = ((b * T + t_idx) * H + h) * W * Cp + w * Cp + c_idx
        o[idx] = rebind[o.element_type](x[src])


def _anima_patchify(
    x: Tensor,      # [B, T, H, W, C] BF16
    ctx: DeviceContext,
) raises -> Tensor:
    """Patchify: [B,T,H,W,16] -> [B, T*nH*nW, 68].
    Appends zero mask channel -> [B,T,H,W,17], then rearranges into patches.
    Matches anima.rs patchify permute [0,1,2,4,6,3,5]."""
    var xsh = x.shape()
    var B = xsh[0]
    var T = xsh[1]
    var H = xsh[2]
    var W = xsh[3]
    var C = xsh[4]  # 16
    var pH = ANIMA_PATCH_SIZE  # 2
    var pW = ANIMA_PATCH_SIZE  # 2
    var nH = H // pH
    var nW = W // pW
    var Cp = C + 1  # 17 (with mask channel)
    var N = T * nH * nW
    var patch_dim = Cp * pH * pW  # 68

    # Build [B, T, H, W, Cp] by appending zero mask channel
    # We do this by building the output directly in the patchify kernel,
    # treating the mask (index C) as always zero.
    # Actually, let's build the padded tensor first.
    # Padded tensor: all elements from x, with last dim extended to Cp (zeros at index C).
    var numel_padded = B * T * H * W * Cp
    var padded_buf = ctx.enqueue_create_buffer[DType.uint8](numel_padded * x.dtype().byte_size())

    # Fill padded tensor: copy x data (first C channels), zeros for mask channel.
    # Use a kernel to do this.
    var dt = x.dtype()
    if dt == STDtype.BF16:
        _fill_padded_bf16(x, padded_buf, B, T, H, W, C, ctx)
    else:
        _fill_padded_f32(x, padded_buf, B, T, H, W, C, ctx)

    # Now patchify from padded tensor
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * N * patch_dim * dt.byte_size())
    var padded_n = numel_padded
    var p_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](padded_n))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](B * N * patch_dim))
    var ntotal = B * T * nH * nW * patch_dim
    var grid = (ntotal + _BLOCK - 1) // _BLOCK

    if dt == STDtype.BF16:
        var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            padded_buf.unsafe_ptr().bitcast[BFloat16](), p_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[_patchify_kernel_bf16, _patchify_kernel_bf16](
            P, O, B, T, nH, nW, pH, pW, Cp, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            padded_buf.unsafe_ptr().bitcast[Float32](), p_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_patchify_kernel_f32, _patchify_kernel_f32](
            P, O, B, T, nH, nW, pH, pW, Cp, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()

    var osh = List[Int]()
    osh.append(B)
    osh.append(N)
    osh.append(patch_dim)
    return Tensor(out_buf^, osh^, dt)


def _fill_padded_bf16_kernel(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # [B*T*H*W*C]
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # [B*T*H*W*Cp]
    n_padded: Int,
    C: Int,
    Cp: Int,
):
    var idx = Int(global_idx.x)
    if idx < n_padded:
        var c = idx % Cp
        var spatial_idx = idx // Cp
        if c < C:
            dst[idx] = rebind[dst.element_type](src[spatial_idx * C + c])
        else:
            dst[idx] = rebind[dst.element_type](BFloat16(0.0))


def _fill_padded_f32_kernel(
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n_padded: Int,
    C: Int,
    Cp: Int,
):
    var idx = Int(global_idx.x)
    if idx < n_padded:
        var c = idx % Cp
        var spatial_idx = idx // Cp
        if c < C:
            dst[idx] = rebind[dst.element_type](src[spatial_idx * C + c])
        else:
            dst[idx] = rebind[dst.element_type](Float32(0.0))


def _fill_padded_bf16(
    x: Tensor,
    padded_buf: DeviceBuffer[DType.uint8],
    B: Int, T: Int, H: Int, W: Int, C: Int,
    ctx: DeviceContext,
) raises:
    var Cp = C + 1
    var n_padded = B * T * H * W * Cp
    var n_src = B * T * H * W * C
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_src))
    var dst_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_padded))
    var Src = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    var Dst = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        padded_buf.unsafe_ptr().bitcast[BFloat16](), dst_rl
    )
    var grid = (n_padded + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_fill_padded_bf16_kernel, _fill_padded_bf16_kernel](
        Src, Dst, n_padded, C, Cp, grid_dim=grid, block_dim=_BLOCK
    )


def _fill_padded_f32(
    x: Tensor,
    padded_buf: DeviceBuffer[DType.uint8],
    B: Int, T: Int, H: Int, W: Int, C: Int,
    ctx: DeviceContext,
) raises:
    var Cp = C + 1
    var n_padded = B * T * H * W * Cp
    var n_src = B * T * H * W * C
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_src))
    var dst_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_padded))
    var Src = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), src_rl
    )
    var Dst = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        padded_buf.unsafe_ptr().bitcast[Float32](), dst_rl
    )
    var grid = (n_padded + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_fill_padded_f32_kernel, _fill_padded_f32_kernel](
        Src, Dst, n_padded, C, Cp, grid_dim=grid, block_dim=_BLOCK
    )


# ── Unpatchify kernel: [B, N, 64] -> [B, T, H, W, 16] ───────────────────────
# Inverse of patchify (without the mask channel).
# Rust: [B,T,nH,nW,pH,pW,C] -> permute [0,1,2,4,3,5,6] -> [B,T,nH,pH,nW,pW,C]
# -> reshape [B,T,H,W,C]
def _unpatchify_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # [B*N*patch_out_dim]
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],  # [B*T*H*W*C]
    B: Int, T: Int, nH: Int, nW: Int, pH: Int, pW: Int, C: Int,
):
    # Output index: [B, T, H, W, C] row-major (C varies fastest)
    var idx = Int(global_idx.x)
    var H = nH * pH
    var W = nW * pW
    var total = B * T * H * W * C
    if idx < total:
        # Decode [b, t, h, w, c]
        var c = idx % C
        var hwt = idx // C
        var w = hwt % W
        var ht = hwt // W
        var h = ht % H
        var tb = ht // H
        var t_idx = tb % T
        var b = tb // T
        # Which patch (ih, iw) and local (ph_idx, pw_idx)
        var ih = h // pH
        var ph_idx = h % pH
        var iw = w // pW
        var pw_idx = w % pW
        # Source patch layout: [B, N, patch_out_dim] where patch_out_dim = pH*pW*C
        # Rust unpatchify: reshape [B,N,pH*pW*C] -> [B,T,nH,nW,pH,pW,C], permute->..., reshape
        # patch_out_idx with layout [pH, pW, C] (C fastest): ph_idx * pW * C + pw_idx * C + c
        var patch_out_idx = ph_idx * pW * C + pw_idx * C + c
        var N = T * nH * nW
        var patch_dim = pH * pW * C
        var patch_n = t_idx * nH * nW + ih * nW + iw
        var src = (b * N + patch_n) * patch_dim + patch_out_idx
        o[idx] = rebind[o.element_type](x[src])


def _unpatchify(
    x: Tensor,   # [B, N, patch_out_dim] BF16
    T: Int,
    nH: Int,
    nW: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Unpatchify [B, T*nH*nW, 64] -> [B, T, H, W, 16].
    Matches anima.rs unpatchify."""
    var xsh = x.shape()
    var B = xsh[0]
    var pH = ANIMA_PATCH_SIZE  # 2
    var pW = ANIMA_PATCH_SIZE  # 2
    var C = ANIMA_LATENT_CHANNELS  # 16
    var H = nH * pH
    var W = nW * pW
    var out_numel = B * T * H * W * C

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_numel * 2)  # BF16
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_numel))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
    )
    var grid = (out_numel + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_unpatchify_kernel_bf16, _unpatchify_kernel_bf16](
        X, O, B, T, nH, nW, pH, pW, C, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(B)
    osh.append(T)
    osh.append(H)
    osh.append(W)
    osh.append(C)
    return Tensor(out_buf^, osh^, STDtype.BF16)


# ── slice helper: extract [B, 1, D] from [B, D] by adding seq dim ────────────
def _narrow_cols(x: Tensor, start: Int, length: Int, ctx: DeviceContext) raises -> Tensor:
    """Slice last dim: [B, D] -> [B, length] starting at `start`."""
    return slice(x, 1, start, length, ctx)


# ── AnimaDiT struct ─────────────────────────────────────────────────────────
struct AnimaDiT:
    """Anima MiniTrainDIT. Loads all 28-block weights + resident weights into GPU."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^

    @staticmethod
    def load(dit_path: String, ctx: DeviceContext) raises -> AnimaDiT:
        """Load all Anima DiT weights from safetensors into GPU."""
        print("[AnimaDiT] loading from", dit_path)
        var st = ShardedSafeTensors.open(dit_path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        var names = st.names()
        for i in range(len(names)):
            var nm = names[i]
            var tv = st.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        print("[AnimaDiT] loaded", len(weights), "tensors")
        return AnimaDiT(weights^, name_to_idx^)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("AnimaDiT: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _linear(self, x: Tensor, key: String, ctx: DeviceContext) raises -> Tensor:
        """Linear with no bias: y = x @ weight.T"""
        return linear(x, self._w(key), None, ctx)

    def _rms_norm_w(self, x: Tensor, key: String, ctx: DeviceContext) raises -> Tensor:
        """RMSNorm with weight."""
        return rms_norm(x, self._w(key), Float32(1e-6), ctx)

    # ── Timestep preparation ────────────────────────────────────────────────
    def _prepare_timestep(
        self, t: Tensor, ctx: DeviceContext
    ) raises -> _TimestepResult:
        """Returns _TimestepResult(t_cond [B,2048], base_adaln [B,6144]).
        Faithful to anima.rs prepare_timestep."""
        # 1. Sinusoidal(2048)
        var emb_f32 = _anima_sinusoidal_emb(t, ANIMA_HIDDEN, ctx)   # [B, 2048] F32
        # Cast to BF16 for weight ops
        var emb = cast_tensor(emb_f32, STDtype.BF16, ctx)           # [B, 2048] BF16

        # 2. hidden = SiLU(Linear(emb))  [B, 2048] — no bias
        var h = self._linear(emb, String("net.t_embedder.1.linear_1.weight"), ctx)
        var hidden = silu(h, ctx)  # [B, 2048] BF16

        # 3. base_adaln = Linear(hidden)  [B, 6144]
        var base_adaln = self._linear(hidden, String("net.t_embedder.1.linear_2.weight"), ctx)

        # 4. t_cond = RMSNorm(sinusoidal, t_embedding_norm)  [B, 2048]
        # NOTE: Rust uses the ORIGINAL sinusoidal emb (not the SiLU'd hidden) for t_cond.
        var t_cond = self._rms_norm_w(emb, String("net.t_embedding_norm.weight"), ctx)

        return _TimestepResult(t_cond^, base_adaln^)

    # ── AdaLN-LoRA modulation ────────────────────────────────────────────────
    def _adaln_mod(
        self,
        t_cond: Tensor,    # [B, 2048] BF16
        base_adaln: Tensor, # [B, 6144] BF16
        prefix: String,    # "net.blocks.0.adaln_modulation_self_attn"
        ctx: DeviceContext,
    ) raises -> _AdaLNMods:
        """Compute AdaLN-LoRA: SiLU(t_cond) -> L1(2048->256) -> L2(256->6144) + base_adaln
        -> chunk3 -> _AdaLNMods(shift, scale, gate) each [B, 2048].
        Faithful to anima.rs adaln_modulation."""
        var t_silu = silu(t_cond, ctx)      # [B, 2048]
        var h = self._linear(t_silu, prefix + String(".1.weight"), ctx)  # [B, 256]
        var mod_out = self._linear(h, prefix + String(".2.weight"), ctx)  # [B, 6144]
        # Add base_adaln (broadcast over the 6144 dim)
        var mod_added = add(mod_out, base_adaln, ctx)  # [B, 6144]
        # Chunk into 3 parts: shift, scale, gate each [B, 2048]
        var shift = slice(mod_added, 1, 0, ANIMA_HIDDEN, ctx)
        var scale = slice(mod_added, 1, ANIMA_HIDDEN, ANIMA_HIDDEN, ctx)
        var gate = slice(mod_added, 1, 2 * ANIMA_HIDDEN, ANIMA_HIDDEN, ctx)
        return _AdaLNMods(shift^, scale^, gate^)

    # ── Final-layer AdaLN ────────────────────────────────────────────────────
    def _final_adaln_mod(
        self,
        t_cond: Tensor,
        base_adaln: Tensor,
        ctx: DeviceContext,
    ) raises -> _TensorPair:
        """Final layer: SiLU->L1(2048->256)->L2(256->4096) + base_adaln[:4096]
        -> chunk2 -> _TensorPair(shift, scale)."""
        var t_silu = silu(t_cond, ctx)
        var h = self._linear(
            t_silu, String("net.final_layer.adaln_modulation.1.weight"), ctx
        )
        var mod_out = self._linear(
            h, String("net.final_layer.adaln_modulation.2.weight"), ctx
        )  # [B, 4096]
        # Add base_adaln[:, :2*D] (first 4096 of 6144)
        var adaln_slice = slice(base_adaln, 1, 0, 2 * ANIMA_HIDDEN, ctx)
        var mod_added = add(mod_out, adaln_slice, ctx)
        var shift = slice(mod_added, 1, 0, ANIMA_HIDDEN, ctx)
        var scale = slice(mod_added, 1, ANIMA_HIDDEN, ANIMA_HIDDEN, ctx)
        return _TensorPair(shift^, scale^)

    # ── RMSNorm per-head: x [B,S,H,D] -> [B,S,H,D] ─────────────────────────
    def _rms_norm_per_head_bshd(
        self, x: Tensor, key: String, ctx: DeviceContext
    ) raises -> Tensor:
        """RMSNorm over last dim (D) applied per-head, on [B,S,H,D] tensor."""
        var xsh = x.shape()
        var B = xsh[0]
        var S = xsh[1]
        var H = xsh[2]
        var D = xsh[3]
        # Flatten to [B*S*H, D], norm, unflatten
        var flat_sh = List[Int]()
        flat_sh.append(B * S * H)
        flat_sh.append(D)
        var flat = reshape(x, flat_sh^, ctx)
        var normed = rms_norm(flat, self._w(key), Float32(1e-6), ctx)
        var osh = List[Int]()
        osh.append(B)
        osh.append(S)
        osh.append(H)
        osh.append(D)
        return reshape(normed, osh^, ctx)

    # ── Self-attention with 3D RoPE ──────────────────────────────────────────
    def _self_attn(
        self,
        x: Tensor,         # [B, S, 2048] BF16
        rope_cos: Tensor,  # [1, 1, S, 64] F32
        rope_sin: Tensor,  # [1, 1, S, 64] F32
        prefix: String,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Self-attention with 3D RoPE halfsplit. Matches anima.rs self_attention."""
        var xsh = x.shape()
        var B = xsh[0]
        var S = xsh[1]
        var H = ANIMA_NUM_HEADS  # 16
        var Dh = ANIMA_HEAD_DIM  # 128

        # Q, K, V projections [B, S, 2048] -> [B, S, 2048]
        var q = self._linear(x, prefix + String(".q_proj.weight"), ctx)
        var k = self._linear(x, prefix + String(".k_proj.weight"), ctx)
        var v = self._linear(x, prefix + String(".v_proj.weight"), ctx)

        # Reshape [B, S, H*Dh] -> [B, S, H, Dh]
        var q_sh = List[Int]()
        q_sh.append(B); q_sh.append(S); q_sh.append(H); q_sh.append(Dh)
        var k_sh = q_sh.copy()
        var v_sh = q_sh.copy()
        q = reshape(q, q_sh^, ctx)
        k = reshape(k, k_sh^, ctx)
        v = reshape(v, v_sh^, ctx)

        # RMSNorm per-head (Rust uses [B,H,S,D] format but same result)
        q = self._rms_norm_per_head_bshd(q, prefix + String(".q_norm.weight"), ctx)
        k = self._rms_norm_per_head_bshd(k, prefix + String(".k_norm.weight"), ctx)

        # Apply 3D RoPE (halfsplit) on [B, S, H, Dh] using [1,1,S,Dh/2] cos/sin
        # cos/sin are F32; rope kernel expects F32 cos/sin
        q = _rope_halfsplit_4d(q, rope_cos, rope_sin, ctx)
        k = _rope_halfsplit_4d(k, rope_cos, rope_sin, ctx)

        # SDPA: expects [B, S, H, Dh]. head_dim=128 -> math-mode path.
        var out = sdpa_nomask[1, ANIMA_N, ANIMA_NUM_HEADS, ANIMA_HEAD_DIM](
            q, k, v, Float32(1.0) / fsqrt(Float32(ANIMA_HEAD_DIM)), ctx
        )

        # Flatten [B, S, H, Dh] -> [B, S, 2048]
        var flat_sh = List[Int]()
        flat_sh.append(B); flat_sh.append(S); flat_sh.append(H * Dh)
        var out_flat = reshape(out, flat_sh^, ctx)
        return self._linear(out_flat, prefix + String(".output_proj.weight"), ctx)

    # ── Cross-attention (NO RoPE) ────────────────────────────────────────────
    def _cross_attn(
        self,
        x: Tensor,        # [B, S_img, 2048] BF16
        context: Tensor,  # [B, S_txt, 1024] BF16
        prefix: String,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Cross-attention: Q from image (2048), K/V from text (1024->2048).
        NO RoPE. Matches anima.rs cross_attention."""
        var xsh = x.shape()
        var B = xsh[0]
        var S_img = xsh[1]
        var csh = context.shape()
        var S_txt = csh[1]
        var H = ANIMA_NUM_HEADS  # 16
        var Dh = ANIMA_HEAD_DIM  # 128

        # Q from image [B, S_img, 2048]
        var q = self._linear(x, prefix + String(".q_proj.weight"), ctx)
        # K, V from context [B, S_txt, 1024 -> 2048]
        var k = self._linear(context, prefix + String(".k_proj.weight"), ctx)
        var v = self._linear(context, prefix + String(".v_proj.weight"), ctx)

        # Reshape to [B, S, H, Dh]
        var q_sh = List[Int]()
        q_sh.append(B); q_sh.append(S_img); q_sh.append(H); q_sh.append(Dh)
        var k_sh = List[Int]()
        k_sh.append(B); k_sh.append(S_txt); k_sh.append(H); k_sh.append(Dh)
        var v_sh = k_sh.copy()
        q = reshape(q, q_sh^, ctx)
        k = reshape(k, k_sh^, ctx)
        v = reshape(v, v_sh^, ctx)

        # RMSNorm per-head on [B, S, H, Dh]
        q = self._rms_norm_per_head_bshd(q, prefix + String(".q_norm.weight"), ctx)
        k = self._rms_norm_per_head_bshd(k, prefix + String(".k_norm.weight"), ctx)

        # NO RoPE on cross-attention

        # SDPA for cross-attention: q[B,S_img,H,Dh] × k[B,S_txt,H,Dh]
        # sdpa_nomask expects same S for q and k — we need to handle cross-attention
        # by using the math-mode path directly with different seq lens.
        var out = _cross_sdpa_nomask[1, ANIMA_N, ANIMA_S_TXT, ANIMA_NUM_HEADS, ANIMA_HEAD_DIM](
            q, k, v, Float32(1.0) / fsqrt(Float32(ANIMA_HEAD_DIM)), ctx
        )

        # Flatten [B, S_img, H, Dh] -> [B, S_img, 2048]
        var flat_sh = List[Int]()
        flat_sh.append(B); flat_sh.append(S_img); flat_sh.append(H * Dh)
        var out_flat = reshape(out, flat_sh^, ctx)
        return self._linear(out_flat, prefix + String(".output_proj.weight"), ctx)

    # ── GELU MLP ─────────────────────────────────────────────────────────────
    def _mlp(self, x: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        """MLP: Linear(2048->8192) -> GELU -> Linear(8192->2048), no bias."""
        var h = self._linear(x, prefix + String(".layer1.weight"), ctx)
        var ha = gelu(h, ctx)
        return self._linear(ha, prefix + String(".layer2.weight"), ctx)

    # ── Transformer block ────────────────────────────────────────────────────
    def _transformer_block(
        self,
        x: Tensor,           # [B, S, 2048] BF16 (but residual in F32 inside)
        context: Tensor,     # [B, S_txt, 1024] BF16
        t_cond: Tensor,      # [B, 2048] BF16
        base_adaln: Tensor,  # [B, 6144] BF16
        rope_cos: Tensor,
        rope_sin: Tensor,
        block_idx: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Single MiniTrainDIT block with AdaLN-LoRA modulation.
        Faithful to anima.rs transformer_block.
        Uses F32 residual stream (x_f32) across all 3 sub-blocks."""
        var prefix = String("net.blocks.") + String(block_idx)

        # F32 residual stream (model has large activations ~200+, BF16 loses precision)
        var x_f32 = cast_tensor(x, STDtype.F32, ctx)  # [B, S, 2048] F32

        # --- Self-attention ---
        var sa_mods = self._adaln_mod(
            t_cond, base_adaln,
            prefix + String(".adaln_modulation_self_attn"), ctx
        )
        var x_bf16 = cast_tensor(x_f32, STDtype.BF16, ctx)
        var x_mod = _apply_adaln_modulate(x_bf16, sa_mods.shift, sa_mods.scale, ctx)
        var attn_out = self._self_attn(x_mod, rope_cos, rope_sin,
                                        prefix + String(".self_attn"), ctx)
        # gate: [B, D] -> reshape to [B, 1, D] for broadcasting with [B, S, D]
        var gate_sa_f32 = cast_tensor(sa_mods.gate, STDtype.F32, ctx)
        var gsh = gate_sa_f32.shape()
        var gate_sa_3d_sh = List[Int]()
        gate_sa_3d_sh.append(gsh[0]); gate_sa_3d_sh.append(1); gate_sa_3d_sh.append(gsh[1])
        var gate_sa_3d = reshape(gate_sa_f32, gate_sa_3d_sh^, ctx)
        var attn_f32 = cast_tensor(attn_out, STDtype.F32, ctx)
        var gated_attn = mul(attn_f32, gate_sa_3d, ctx)
        x_f32 = add(x_f32, gated_attn, ctx)

        # --- Cross-attention ---
        var ca_mods = self._adaln_mod(
            t_cond, base_adaln,
            prefix + String(".adaln_modulation_cross_attn"), ctx
        )
        x_bf16 = cast_tensor(x_f32, STDtype.BF16, ctx)
        x_mod = _apply_adaln_modulate(x_bf16, ca_mods.shift, ca_mods.scale, ctx)
        var cross_out = self._cross_attn(x_mod, context,
                                          prefix + String(".cross_attn"), ctx)
        var gate_ca_f32 = cast_tensor(ca_mods.gate, STDtype.F32, ctx)
        var gca_sh = gate_ca_f32.shape()
        var gate_ca_3d_sh = List[Int]()
        gate_ca_3d_sh.append(gca_sh[0]); gate_ca_3d_sh.append(1); gate_ca_3d_sh.append(gca_sh[1])
        var gate_ca_3d = reshape(gate_ca_f32, gate_ca_3d_sh^, ctx)
        var cross_f32 = cast_tensor(cross_out, STDtype.F32, ctx)
        var gated_cross = mul(cross_f32, gate_ca_3d, ctx)
        x_f32 = add(x_f32, gated_cross, ctx)

        # --- MLP ---
        var mlp_mods = self._adaln_mod(
            t_cond, base_adaln,
            prefix + String(".adaln_modulation_mlp"), ctx
        )
        x_bf16 = cast_tensor(x_f32, STDtype.BF16, ctx)
        x_mod = _apply_adaln_modulate(x_bf16, mlp_mods.shift, mlp_mods.scale, ctx)
        var mlp_out = self._mlp(x_mod, prefix + String(".mlp"), ctx)
        var gate_mlp_f32 = cast_tensor(mlp_mods.gate, STDtype.F32, ctx)
        var gmlp_sh = gate_mlp_f32.shape()
        var gate_mlp_3d_sh = List[Int]()
        gate_mlp_3d_sh.append(gmlp_sh[0]); gate_mlp_3d_sh.append(1); gate_mlp_3d_sh.append(gmlp_sh[1])
        var gate_mlp_3d = reshape(gate_mlp_f32, gate_mlp_3d_sh^, ctx)
        var mlp_f32 = cast_tensor(mlp_out, STDtype.F32, ctx)
        var gated_mlp = mul(mlp_f32, gate_mlp_3d, ctx)
        x_f32 = add(x_f32, gated_mlp, ctx)

        # Return BF16 (matches Rust: x_f32.to_dtype(DType::BF16))
        return cast_tensor(x_f32, STDtype.BF16, ctx)

    # ── Full forward pass ────────────────────────────────────────────────────
    def forward_with_context(
        self,
        x: Tensor,          # [B, T, H, W, 16] F32 latent (will cast to BF16 inside)
        timestep: Tensor,   # [B] F32 sigma value (raw, NOT *1000)
        context: Tensor,    # [B, 256, 1024] BF16 pre-computed context
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Full MiniTrainDIT forward. Returns [B, T, H, W, 16] BF16."""
        var xsh = x.shape()
        var T = xsh[1]
        var H = xsh[2]
        var W = xsh[3]
        var pH = ANIMA_PATCH_SIZE  # 2
        var nH = H // pH
        var nW = W // pH

        # Cast latent to BF16 for model computation
        var x_bf16 = cast_tensor(x, STDtype.BF16, ctx)

        # 1. Prepare timestep conditioning (stored in ts_result, used by field access)
        var ts_result = self._prepare_timestep(timestep, ctx)

        # 2. Patchify: [B, T, H, W, 16] -> [B, N, 68]
        var patches = _anima_patchify(x_bf16, ctx)

        # 3. Patch embed: [B, N, 68] -> [B, N, 2048]
        var x_emb = self._linear(patches, String("net.x_embedder.proj.1.weight"), ctx)

        # 4. Build 3D RoPE cos/sin [1, 1, S, 64] F32
        var rope_result = build_anima_3d_rope(T, nH, nW, ANIMA_HEAD_DIM, ctx)

        # 5. Run through 28 transformer blocks
        var x_hidden = x_emb^
        for i in range(ANIMA_DEPTH):
            print("  [block]", i + 1, "/", ANIMA_DEPTH)
            x_hidden = self._transformer_block(
                x_hidden, context,
                ts_result.t_cond, ts_result.base_adaln,
                rope_result.first, rope_result.second,
                i, ctx
            )

        # 6. Final layer
        var final_mods = self._final_adaln_mod(ts_result.t_cond, ts_result.base_adaln, ctx)
        var x_mod = _apply_adaln_modulate(x_hidden, final_mods.first, final_mods.second, ctx)
        var x_out = self._linear(x_mod, String("net.final_layer.linear.weight"), ctx)

        # 7. Unpatchify: [B, N, 64] -> [B, T, H, W, 16]
        return _unpatchify(x_out, T, nH, nW, ctx)


# ── Cross-SDPA with different Q and K/V sequence lengths ─────────────────────
# The standard sdpa_nomask[B,S,H,Dh] requires S_q == S_k.
# For cross-attention: S_q = N_img (4096), S_k = S_txt (256).
# We implement directly via the math-mode path.
def _cross_sdpa_nomask[
    B: Int, S_q: Int, S_k: Int, H: Int, Dh: Int
](
    q: Tensor,  # [B, S_q, H, Dh]
    k: Tensor,  # [B, S_k, H, Dh]
    v: Tensor,  # [B, S_k, H, Dh]
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Cross-attention SDPA: Q[S_q] × K[S_k] -> out[S_q].
    Uses math-mode (matmuls + softmax). No mask."""
    from serenitymojo.ops.attention import _sdpa_math
    comptime BH = B * H
    comptime bhsd_q_rows = B * H * S_q
    comptime bhsd_k_rows = B * H * S_k
    comptime src_q_rows = B * S_q * H
    comptime src_k_rows = B * S_k * H

    # Gather q [B,S_q,H,Dh] -> BHSD-contiguous [B*H,S_q,Dh]
    var q_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_q_rows * Dh)
    var k_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_k_rows * Dh)
    var v_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_k_rows * Dh)

    comptime _DYN2l = Layout.row_major(-1, -1)
    comptime BLOCK = 256

    var q_src_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](src_q_rows, Dh))
    var k_src_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](src_k_rows, Dh))
    var bhsd_q_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](bhsd_q_rows, Dh))
    var bhsd_k_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](bhsd_k_rows, Dh))

    var Qd = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](q_f32.unsafe_ptr(), bhsd_q_rl)
    var Kd = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](k_f32.unsafe_ptr(), bhsd_k_rl)
    var Vd = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](v_f32.unsafe_ptr(), bhsd_k_rl)

    # Gather BF16 BSHD -> F32 BHSD
    from serenitymojo.ops.attention import _gather_bshd_to_bhsd_bf16
    var Qs = LayoutTensor[DType.bfloat16, _DYN2l, MutAnyOrigin](
        q.buf.unsafe_ptr().bitcast[BFloat16](), q_src_rl
    )
    var Ks = LayoutTensor[DType.bfloat16, _DYN2l, MutAnyOrigin](
        k.buf.unsafe_ptr().bitcast[BFloat16](), k_src_rl
    )
    var Vs = LayoutTensor[DType.bfloat16, _DYN2l, MutAnyOrigin](
        v.buf.unsafe_ptr().bitcast[BFloat16](), k_src_rl
    )
    var ngq = B * H * S_q * Dh
    var ngk = B * H * S_k * Dh
    ctx.enqueue_function[_gather_bshd_to_bhsd_bf16, _gather_bshd_to_bhsd_bf16](
        Qs, Qd, B, S_q, H, Dh, grid_dim=(ngq + BLOCK - 1) // BLOCK, block_dim=BLOCK)
    ctx.enqueue_function[_gather_bshd_to_bhsd_bf16, _gather_bshd_to_bhsd_bf16](
        Ks, Kd, B, S_k, H, Dh, grid_dim=(ngk + BLOCK - 1) // BLOCK, block_dim=BLOCK)
    ctx.enqueue_function[_gather_bshd_to_bhsd_bf16, _gather_bshd_to_bhsd_bf16](
        Vs, Vd, B, S_k, H, Dh, grid_dim=(ngk + BLOCK - 1) // BLOCK, block_dim=BLOCK)

    # QKT scores [B*H, S_q, S_k]
    from linalg.matmul.vendor.blas import matmul
    from std.math import exp
    from std.gpu import thread_idx, block_idx, barrier
    from std.gpu.memory import AddressSpace
    from std.memory import stack_allocation

    var scores = ctx.enqueue_create_buffer[DType.float32](BH * S_q * S_k)
    var head_q_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](S_q, Dh))
    var head_k_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](S_k, Dh))
    var sc_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](S_q, S_k))
    var qptr = q_f32.unsafe_ptr()
    var kptr = k_f32.unsafe_ptr()
    var scptr = scores.unsafe_ptr()
    for bh in range(BH):
        var A = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](
            qptr + bh * S_q * Dh, head_q_rl
        )
        var Bt = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](
            kptr + bh * S_k * Dh, head_k_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](
            scptr + bh * S_q * S_k, sc_rl
        )
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)

    # Scale scores
    var sm_rows = BH * S_q
    var sc_full_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](sm_rows, S_k))
    var sc_full = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](scptr, sc_full_rl)
    var nsm = sm_rows * S_k
    var smgrid = (nsm + BLOCK - 1) // BLOCK
    from serenitymojo.ops.attention import _scale_f32
    ctx.enqueue_function[_scale_f32, _scale_f32](
        sc_full, scale, sm_rows, S_k, grid_dim=smgrid, block_dim=BLOCK)

    # Softmax over S_k dim (last dim)
    from serenitymojo.ops.attention import _softmax_rows_f32
    ctx.enqueue_function[_softmax_rows_f32, _softmax_rows_f32](
        sc_full, S_k, grid_dim=sm_rows, block_dim=256)

    # P @ V -> out [B*H, S_q, Dh]
    var out_f32 = ctx.enqueue_create_buffer[DType.float32](bhsd_q_rows * Dh)
    var head_out_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](S_q, Dh))
    var optr = out_f32.unsafe_ptr()
    var vptr = v_f32.unsafe_ptr()
    for bh in range(BH):
        var P = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](
            scptr + bh * S_q * S_k, sc_rl
        )
        var Vh = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](
            vptr + bh * S_k * Dh, head_k_rl
        )
        var Oh = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](
            optr + bh * S_q * Dh, head_out_rl
        )
        matmul(ctx, Oh, P, Vh, transpose_b=False, c_row_major=True)

    # Scatter F32 BHSD -> BF16 BSHD output [B, S_q, H, Dh]
    from serenitymojo.ops.attention import _scatter_bhsd_to_bshd_bf16
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * S_q * H * Dh * 2)
    var out_src_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](bhsd_q_rows, Dh))
    var out_dst_rl = RuntimeLayout[_DYN2l].row_major(IndexList[2](src_q_rows, Dh))
    var out_src = LayoutTensor[DType.float32, _DYN2l, MutAnyOrigin](optr, out_src_rl)
    var Od = LayoutTensor[DType.bfloat16, _DYN2l, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_dst_rl
    )
    var nsc = B * H * S_q * Dh
    var scgrid2 = (nsc + BLOCK - 1) // BLOCK
    ctx.enqueue_function[_scatter_bhsd_to_bshd_bf16, _scatter_bhsd_to_bshd_bf16](
        out_src, Od, B, S_q, H, Dh, grid_dim=scgrid2, block_dim=BLOCK)
    ctx.synchronize()

    var osh = List[Int]()
    osh.append(B); osh.append(S_q); osh.append(H); osh.append(Dh)
    return Tensor(out_buf^, osh^, STDtype.BF16)
