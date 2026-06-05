# models/text_encoder/gpt_oss_encoder.mojo — GPT-OSS text encoder (Lens, GPU).
#
# Pure-Mojo, inference-only port of inference-flame/src/models/gpt_oss_encoder.rs
# + gpt_oss_rope.rs (the microsoft/Lens text encoder, a 24-layer GPT-OSS-20B
# decoder used as a frozen text encoder). HF transformers
# `models/gpt_oss/modeling_gpt_oss.py` is the numerical oracle.
#
# This is the BLOCKER for Lens real-image generation, so the layer loop is
# OOM-AWARE / STREAMED: each layer's weights are loaded from the sharded
# safetensors, run, and FREED before the next layer is loaded. We never hold
# all 24 layers resident. The expert weights are MXFP4-packed (~10 GB on disk);
# they are loaded raw-byte per layer and DEQUANTIZED on the GPU into a transient
# BF16 expert matrix that is freed when the layer finishes.
#
# Architecture per layer (gpt_oss_encoder.rs:687 GptOssLayer::forward):
#   r = x + attn(rms_norm(x, input_layernorm))     # pre-norm
#   r = r + moe(rms_norm(r, post_attention_layernorm))
# attn (gpt_oss_encoder.rs:263):
#   q/k/v Linear(+bias) -> reshape BSHD -> RoPE half-split on q,k (YaRN)
#     -> GQA repeat kv -> SDPA-with-sinks (extra per-head sink logit column)
#     -> o_proj(+bias). Mask = sliding-window-causal (even layers) or
#     full-causal (odd layers).
# moe (gpt_oss_encoder.rs:454, GptOssTopKRouter + GptOssExperts):
#   router Linear(+bias) -> top-k(4)-softmax routing
#   per expert: gate_up = x @ gate_up^T + bias ; interleaved split
#     gate = gate_up[::2], up = gate_up[1::2]
#     gate = clamp(gate, max=limit=7.0); up = clamp(up, [-7,7])
#     glu  = gate * sigmoid(alpha*gate), alpha=1.702
#     act  = (up + 1) * glu
#     y    = act @ down^T + down_bias
#   weighted scatter-add over routed slots -> [T, hidden].
#
# RoPE = HALF-SPLIT (HF rotate_half), YaRN-scaled inv_freq + mscale
# (gpt_oss_rope.rs:278 compute_yarn_inv_freq). theta=150000, factor=32,
# beta_fast=32, beta_slow=1, orig_max_pos=4096.
#
# Config (text_encoder/config.json, GptOssForCausalLM):
#   hidden=2880, layers=24, heads=64, kv_heads=8 (GQA n_rep=8), head_dim=64,
#   intermediate=2880, experts=32, top_k=4, sliding_window=128,
#   rms_norm_eps=1e-5, swiglu_limit=7.0, swiglu_alpha=1.702, attention_bias=true.
#   Lens captures hidden states at layers [5,11,17,23] (pre-final-norm).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.

from std.math import cos as fcos, sin as fsin, log as flog, ldexp
from std.memory import ArcPointer, stack_allocation
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import (
    add as t_add,
    gather_rows,
    mul as t_mul,
    slice as t_slice,
)
from serenitymojo.ops.moe import top_k_router, gated_scatter_add, RouterPlan
from serenitymojo.ops.cast import cast_tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _SDPA_DH_MAX = 128
# FP4 e2m1 lookup table (transformers integrations/mxfp4.py FP4_VALUES).
comptime _FP4_LUT = SIMD[DType.float32, 16](
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct GptOssConfig(Copyable, Movable, ImplicitlyCopyable):
    """GPT-OSS / Lens text-encoder hyperparameters."""

    var hidden_size: Int
    var num_layers: Int
    var num_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var intermediate_size: Int
    var num_experts: Int
    var top_k: Int
    var sliding_window: Int
    var rms_norm_eps: Float32
    var swiglu_limit: Float32
    var swiglu_alpha: Float32
    var rope_theta: Float64
    var rope_factor: Float64
    var rope_beta_fast: Float64
    var rope_beta_slow: Float64
    var rope_orig_max_pos: Int

    @staticmethod
    def lens_default() -> GptOssConfig:
        """microsoft/Lens text encoder (GPT-OSS-20B layout)."""
        return GptOssConfig(
            2880, 24, 64, 8, 64, 2880, 32, 4, 128,
            Float32(1e-5), Float32(7.0), Float32(1.702),
            Float64(150000.0), Float64(32.0), Float64(32.0), Float64(1.0), 4096,
        )

    @staticmethod
    def tiny() -> GptOssConfig:
        """Tiny smoke config (same shape family as Lens, scaled down).
        hidden=64, heads=4, kv=2 (n_rep=2), head_dim=16, inter=64,
        experts=4, top_k=2, window=4."""
        return GptOssConfig(
            64, 2, 4, 2, 16, 64, 4, 2, 4,
            Float32(1e-5), Float32(7.0), Float32(1.702),
            Float64(150000.0), Float64(32.0), Float64(32.0), Float64(1.0), 4096,
        )


def lens_extract_layers() -> List[Int]:
    """Lens conditioning capture layers, 0-indexed: [5, 11, 17, 23]."""
    return [5, 11, 17, 23]


# layer_type: even index -> sliding_attention, odd -> full_attention.
def _is_sliding(layer_idx: Int) -> Bool:
    return (layer_idx % 2) == 0


# ── YaRN RoPE inv_freq (host F32 math, mirrors gpt_oss_rope.rs) ──────────────
from std.math import pi as _PI


def _find_correction_dim(
    num_rot: Float64, dim: Int, base: Float64, max_pos: Float64
) -> Float64:
    var two_pi = 2.0 * _PI
    return (Float64(dim) * flog(max_pos / (num_rot * two_pi))) / (
        2.0 * flog(base)
    )


def _compute_yarn_inv_freq(
    cfg: GptOssConfig,
) -> Tuple[List[Float64], Float64]:
    """Returns (inv_freq[head_dim/2], attention_scaling). Bit-exact port of
    transformers _compute_yarn_parameters for the GPT-OSS config."""
    var dim = cfg.head_dim
    var half = dim // 2
    var theta = cfg.rope_theta
    var factor = cfg.rope_factor
    var orig_max = Float64(cfg.rope_orig_max_pos)

    var inv_freq_extrap = List[Float64]()
    var inv_freq_interp = List[Float64]()
    for k in range(half):
        var exponent = Float64(2 * k) / Float64(dim)
        var pos_freq = theta ** exponent
        inv_freq_extrap.append(1.0 / pos_freq)
        inv_freq_interp.append(1.0 / (factor * pos_freq))

    # find_correction_range(beta_fast, beta_slow): low<-beta_fast, high<-beta_slow.
    var low = _find_correction_dim(cfg.rope_beta_fast, dim, theta, orig_max)
    var high = _find_correction_dim(cfg.rope_beta_slow, dim, theta, orig_max)
    # truncate=false for Lens (no floor/ceil). clamp endpoints.
    if low < 0.0:
        low = 0.0
    var hi_cap = Float64(dim) - 1.0
    if high > hi_cap:
        high = hi_cap

    # linear_ramp_factor(low, high, half) with +0.001 singularity guard.
    var ramp_max = high
    if low == high:
        ramp_max = high + 0.001
    var denom = ramp_max - low

    var inv_freq = List[Float64]()
    for k in range(half):
        var v = (Float64(k) - low) / denom
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        var ramp = v
        var extrap_factor = 1.0 - ramp
        # inv_freq = interp*(1-extrap_factor) + extrap*extrap_factor
        var f = inv_freq_interp[k] * (1.0 - extrap_factor) + inv_freq_extrap[
            k
        ] * extrap_factor
        inv_freq.append(f)

    # get_mscale(factor): 1.0 if factor<=1 else 0.1*ln(factor)+1.
    var mscale = 1.0
    if factor > 1.0:
        mscale = 0.1 * flog(factor) + 1.0
    return (inv_freq^, mscale)


# Build cos/sin tables in row order (position, head): row = t*H + head shares
# position t's angles (RoPE angle depends only on position). Half-split layout:
# cos/sin[row, k] for k in [0, head_dim/2). attention_scaling (mscale) folded in.
# Returns flat F32 list length seq*heads*(head_dim/2).
def _build_yarn_tables(
    cfg: GptOssConfig, seq: Int, heads: Int
) raises -> List[List[Float32]]:
    var pair = _compute_yarn_inv_freq(cfg)
    var inv_freq = pair[0].copy()
    var mscale = pair[1]
    var half = cfg.head_dim // 2
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for t in range(seq):
        for _h in range(heads):
            for k in range(half):
                var angle = Float64(t) * inv_freq[k]
                cos_vals.append(Float32(fcos(angle) * mscale))
                sin_vals.append(Float32(fsin(angle) * mscale))
    return [cos_vals^, sin_vals^]


# ── MXFP4 dequant kernel (transformers convert_moe_packed_tensors) ──────────
# blocks: U8 [R, G, 16] (one expert's 2D row-block grid; R rows, G blocks/row,
#   16 bytes/block, 2 fp4 vals/byte -> 32 cols/block -> G*32 cols total).
# scales: U8 [R, G] (E8M0 exponent, value = byte - 127).
# Output: BF16 [R, G*32] where out[r, g*32 + 2*b + 0] = lut[byte & 0xF] * 2^exp
#         and out[r, g*32 + 2*b + 1] = lut[byte >> 4] * 2^exp.
# One thread per OUTPUT element pair-half; we launch one thread per byte (R*G*16)
# and write the two output columns it produces. This mirrors the CUDA-style
# grid-stride decode in ops/fp8.mojo (one thread per packed unit).
def _mxfp4_dequant_kernel(
    blocks: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    scales: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    r: Int,
    g: Int,
    cols: Int,  # = g * 32
):
    var idx = Int(global_idx.x)
    var total = r * g * 16
    if idx >= total:
        return
    var byte_in_block = idx % 16
    var rest = idx // 16
    var blk = rest % g
    var row = rest // g
    var packed = Int(rebind[Scalar[DType.uint8]](blocks[idx]))
    var sc = Int(rebind[Scalar[DType.uint8]](scales[row * g + blk]))
    var exp = sc - 127
    var lo = packed & 0x0F
    var hi = (packed >> 4) & 0x0F
    var lut = _FP4_LUT
    var v_lo = ldexp(lut[lo], exp)
    var v_hi = ldexp(lut[hi], exp)
    var base_col = blk * 32 + byte_in_block * 2
    var out_lo = row * cols + base_col
    o[out_lo] = rebind[o.element_type](v_lo.cast[DType.bfloat16]())
    o[out_lo + 1] = rebind[o.element_type](v_hi.cast[DType.bfloat16]())


# Dequant one expert's packed matrix -> BF16 [R, cols] (cols = G*32).
def _mxfp4_dequant_expert(
    blocks: Tensor,  # U8 [R, G, 16]  (3D, one expert)
    scales: Tensor,  # U8 [R, G]
    ctx: DeviceContext,
) raises -> Tensor:
    var bs = blocks.shape()
    if len(bs) != 3 or bs[2] != 16:
        raise Error("mxfp4_dequant_expert: blocks must be [R,G,16]")
    var r = bs[0]
    var g = bs[1]
    var cols = g * 32
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](r * cols * 2)
    var blk_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](r * g * 16))
    var sc_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](r * g))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](r * cols))
    var B = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        blocks.buf.unsafe_ptr(), blk_rl
    )
    var S = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        scales.buf.unsafe_ptr(), sc_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )
    var total = r * g * 16
    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_mxfp4_dequant_kernel, _mxfp4_dequant_kernel](
        B, S, O, r, g, cols, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [r, cols], STDtype.BF16)


# ── encoder-local glue kernels ──────────────────────────────────────────────
def _embed_kernel_bf16(
    table: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    ids: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    seq: Int,
    hidden: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * hidden
    if idx < total:
        var t = idx // hidden
        var j = idx % hidden
        var tok = Int(rebind[Scalar[DType.int32]](ids[t]))
        o[idx] = rebind[o.element_type](table[tok * hidden + j])


# GQA repeat BSHD: src [1,N,H_kv,Dh] -> dst [1,N,H,Dh], head h reads kv h//n_rep.
def _repeat_kv_kernel_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    seq: Int,
    h: Int,
    h_kv: Int,
    dh: Int,
    n_rep: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * h * dh
    if idx < total:
        var dh_i = idx % dh
        var rest = idx // dh
        var head = rest % h
        var t = rest // h
        var kvh = head // n_rep
        var src_idx = (t * h_kv + kvh) * dh + dh_i
        dst[idx] = rebind[dst.element_type](src[src_idx])


# Per-row bias add: o[t,j] = x[t,j] + bias[j]. (BF16, F32 math.)
def _row_bias_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    bias: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    rows: Int,
    d: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * d
    if idx < total:
        var j = idx % d
        var xv = rebind[Scalar[DType.bfloat16]](x[idx]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.bfloat16]](bias[j]).cast[DType.float32]()
        o[idx] = rebind[o.element_type]((xv + bv).cast[DType.bfloat16]())


# GPT-OSS gated activation over an interleaved gate_up row [rows, 2*inter]:
#   gate = gate_up[:, ::2], up = gate_up[:, 1::2]
#   gate = min(gate, limit); up = clamp(up, -limit, limit)
#   glu  = gate * sigmoid(alpha*gate); act = (up + 1) * glu
# Output [rows, inter]. One thread per output element. F32 math.
from std.math import exp as fexp


def _gptoss_act_kernel_bf16(
    gate_up: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    rows: Int,
    inter: Int,
    limit: Float32,
    alpha: Float32,
):
    var idx = Int(global_idx.x)
    var total = rows * inter
    if idx >= total:
        return
    var i = idx % inter
    var row = idx // inter
    var base = row * (2 * inter)
    var gate = rebind[Scalar[DType.bfloat16]](gate_up[base + 2 * i]).cast[
        DType.float32
    ]()
    var up = rebind[Scalar[DType.bfloat16]](gate_up[base + 2 * i + 1]).cast[
        DType.float32
    ]()
    # gate.clamp(max=limit); up.clamp(-limit, limit)
    if gate > limit:
        gate = limit
    if up > limit:
        up = limit
    if up < -limit:
        up = -limit
    var sig = Float32(1.0) / (Float32(1.0) + fexp(-alpha * gate))
    var glu = gate * sig
    var act = (up + Float32(1.0)) * glu
    o[idx] = rebind[o.element_type](act.cast[DType.bfloat16]())


# ── SDPA with attention sinks. Storage stays q/k/v dtype; score/value math F32.
# q,k,v are BSHD [1, S, H, Dh] (k,v already GQA-expanded to H). sinks [H] is
# one extra logit per head appended to the K axis BEFORE softmax, then dropped
# after. Causal/sliding masking is scalar control in-kernel, not a tensor
# boundary. Output BSHD [1, S, H, Dh].
def _sdpa_sinks_kernel[dtype: DType](
    q: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    k: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    v: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    sinks: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    seq: Int,
    h: Int,
    dh: Int,
    scale: Float32,
    real_len: Int,
    sliding: Int,
    window: Int,
):
    var idx = Int(global_idx.x)
    var total = h * seq
    if idx >= total:
        return
    var qi = idx % seq
    var head = idx // seq
    # q row offset in BSHD: [1, S, H, Dh] -> (qi*h + head)*dh
    var q_base = (qi * h + head) * dh
    var qreg = stack_allocation[_SDPA_DH_MAX, Scalar[DType.float32]]()
    for d in range(dh):
        qreg[d] = rebind[Scalar[dtype]](q[q_base + d]).cast[DType.float32]()
    # First pass: max over scores (incl sink).
    var sink_logit = rebind[Scalar[dtype]](sinks[head]).cast[DType.float32]()
    var m = sink_logit
    for kj in range(seq):
        var keep = (kj <= qi) and (kj < real_len)
        if keep and sliding != 0:
            if (qi - kj) >= window:
                keep = False
        var score = Float32(-1.0e9)
        if keep:
            var k_base = (kj * h + head) * dh
            var dot = Float32(0.0)
            for d in range(dh):
                dot += qreg[d] * rebind[Scalar[dtype]](
                    k[k_base + d]
                ).cast[DType.float32]()
            score = dot * scale
        if score > m:
            m = score
    # Second pass: denom.
    var denom = fexp(sink_logit - m)
    for kj in range(seq):
        var keep = (kj <= qi) and (kj < real_len)
        if keep and sliding != 0:
            if (qi - kj) >= window:
                keep = False
        if keep:
            var k_base = (kj * h + head) * dh
            var dot = Float32(0.0)
            for d in range(dh):
                dot += qreg[d] * rebind[Scalar[dtype]](
                    k[k_base + d]
                ).cast[DType.float32]()
            var score = dot * scale
            denom += fexp(score - m)
    # Third pass: weighted V sum in private F32 registers (sink contributes 0).
    var o_base = (qi * h + head) * dh
    var acc = stack_allocation[_SDPA_DH_MAX, Scalar[DType.float32]]()
    for d in range(dh):
        acc[d] = 0.0
    for kj in range(seq):
        var keep = (kj <= qi) and (kj < real_len)
        if keep and sliding != 0:
            if (qi - kj) >= window:
                keep = False
        if keep:
            var k_base = (kj * h + head) * dh
            var dot = Float32(0.0)
            for d in range(dh):
                dot += qreg[d] * rebind[Scalar[dtype]](
                    k[k_base + d]
                ).cast[DType.float32]()
            var score = dot * scale
            var w = fexp(score - m) / denom
            var v_base = (kj * h + head) * dh
            for d in range(dh):
                acc[d] += w * rebind[Scalar[dtype]](
                    v[v_base + d]
                ).cast[DType.float32]()
    for d in range(dh):
        o[o_base + d] = rebind[o.element_type](acc[d].cast[dtype]())


# ── host-side dispatch helpers ──────────────────────────────────────────────
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _reshape(x: Tensor, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    var want = 1
    for i in range(len(shape)):
        want *= shape[i]
    if want != x.numel():
        raise Error("reshape: numel mismatch")
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, shape^, x.dtype())


def _add_row_bias(x: Tensor, bias: Tensor, ctx: DeviceContext) raises -> Tensor:
    """o[t,j] = x[t,j] + bias[j]. x: [rows, d] BF16, bias: [d] BF16."""
    var xs = x.shape()
    var d = xs[len(xs) - 1]
    var rows = x.numel() // d
    if bias.numel() != d:
        raise Error("add_row_bias: bias dim mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * d))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var BI = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        bias.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var grid = (rows * d + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_row_bias_kernel_bf16, _row_bias_kernel_bf16](
        X, BI, O, rows, d, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


def _gptoss_act(
    gate_up: Tensor, inter: Int, limit: Float32, alpha: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Interleaved gate/up split + clamp + sigmoid-glu. [rows, 2*inter] BF16 ->
    [rows, inter] BF16."""
    var gs = gate_up.shape()
    var two_inter = gs[len(gs) - 1]
    var rows = gate_up.numel() // two_inter
    if two_inter != 2 * inter:
        raise Error("gptoss_act: last dim != 2*inter")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](rows * inter * 2)
    var gu_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * two_inter))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * inter))
    var GU = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        gate_up.buf.unsafe_ptr().bitcast[BFloat16](), gu_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
    )
    var grid = (rows * inter + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_gptoss_act_kernel_bf16, _gptoss_act_kernel_bf16](
        GU, O, rows, inter, limit, alpha, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [rows, inter], STDtype.BF16)


def _repeat_kv(var x: Tensor, h: Int, h_kv: Int, ctx: DeviceContext) raises -> Tensor:
    var xs = x.shape()
    var seq = xs[1]
    var dh = xs[3]
    var n_rep = h // h_kv
    if n_rep == 1:
        return x^
    var out_n = seq * h * dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_n * 2)
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq * h_kv * dh))
    var dst_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    var D = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), dst_rl
    )
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_repeat_kv_kernel_bf16, _repeat_kv_kernel_bf16](
        S, D, seq, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [1, seq, h, dh], x.dtype())


def _sdpa_with_sinks(
    q: Tensor,  # BSHD [1,S,H,Dh]
    k: Tensor,  # BSHD [1,S,H,Dh] (GQA-expanded)
    v: Tensor,  # BSHD [1,S,H,Dh]
    sinks: Tensor,  # [H], q dtype
    seq: Int,
    h: Int,
    dh: Int,
    scale: Float32,
    real_len: Int,
    sliding: Bool,
    window: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Math-mode SDPA with per-head attention sinks. Returns BSHD [1,S,H,Dh]
    in q's storage dtype. Interior score/value math is F32."""
    if q.dtype() != k.dtype() or q.dtype() != v.dtype():
        raise Error("sdpa_with_sinks: q/k/v dtype mismatch")
    if sinks.dtype() != q.dtype():
        raise Error("sdpa_with_sinks: sinks dtype must match q dtype")
    if dh > _SDPA_DH_MAX:
        raise Error("sdpa_with_sinks: head_dim exceeds scratch limit")
    if sinks.numel() != h:
        raise Error("sdpa_with_sinks: sinks length != num_heads")

    var n = seq * h * dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n * q.dtype().byte_size()
    )
    var n_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](h))
    var total = h * seq
    var grid = (total + _BLOCK - 1) // _BLOCK
    var do_sliding = 1 if sliding else 0
    var dt = q.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var Q = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float32](), n_rl
        )
        var K = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float32](), n_rl
        )
        var V = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float32](), n_rl
        )
        var SK = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            sinks.buf.unsafe_ptr().bitcast[Float32](), s_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), n_rl
        )
        ctx.enqueue_function[
            _sdpa_sinks_kernel[DType.float32],
            _sdpa_sinks_kernel[DType.float32],
        ](
            Q, K, V, SK, O, seq, h, dh, scale, real_len, do_sliding, window,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[BFloat16](), n_rl
        )
        var K = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[BFloat16](), n_rl
        )
        var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[BFloat16](), n_rl
        )
        var SK = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            sinks.buf.unsafe_ptr().bitcast[BFloat16](), s_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), n_rl
        )
        ctx.enqueue_function[
            _sdpa_sinks_kernel[DType.bfloat16],
            _sdpa_sinks_kernel[DType.bfloat16],
        ](
            Q, K, V, SK, O, seq, h, dh, scale, real_len, do_sliding, window,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var Q = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            q.buf.unsafe_ptr().bitcast[Float16](), n_rl
        )
        var K = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            k.buf.unsafe_ptr().bitcast[Float16](), n_rl
        )
        var V = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            v.buf.unsafe_ptr().bitcast[Float16](), n_rl
        )
        var SK = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            sinks.buf.unsafe_ptr().bitcast[Float16](), s_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), n_rl
        )
        ctx.enqueue_function[
            _sdpa_sinks_kernel[DType.float16],
            _sdpa_sinks_kernel[DType.float16],
        ](
            Q, K, V, SK, O, seq, h, dh, scale, real_len, do_sliding, window,
            grid_dim=grid, block_dim=_BLOCK,
        )
    ctx.synchronize()
    return Tensor(out_buf^, [1, seq, h, dh], q.dtype())


def _embed(
    table: Tensor, ids: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var ts = table.shape()
    var hidden = ts[len(ts) - 1]
    var seq = len(ids)
    var id_host = ctx.enqueue_create_host_buffer[DType.uint8](seq * 4)
    var ip = id_host.unsafe_ptr().bitcast[Int32]()
    for i in range(seq):
        ip[i] = Int32(ids[i])
    var id_dev = ctx.enqueue_create_buffer[DType.uint8](seq * 4)
    ctx.enqueue_copy(dst_buf=id_dev, src_buf=id_host)
    ctx.synchronize()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](seq * hidden * 2)
    var tab_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](table.numel()))
    var id_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq * hidden))
    var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        table.buf.unsafe_ptr().bitcast[BFloat16](), tab_rl
    )
    var IDS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        id_dev.unsafe_ptr().bitcast[Int32](), id_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )
    var grid = (seq * hidden + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_embed_kernel_bf16, _embed_kernel_bf16](
        T, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [1, seq, hidden], STDtype.BF16)


# ── one layer's weight block (streamed) ─────────────────────────────────────
comptime LayerBlock = Dict[String, ArcPointer[Tensor]]


# ── GptOssEncoder ───────────────────────────────────────────────────────────
struct GptOssEncoder:
    """GPT-OSS / Lens text encoder. The layer loop is STREAMED: per layer we
    load its weights from the sharded checkpoint, dequant MXFP4 experts on GPU,
    run the layer, and free everything before the next layer. Only the token
    embedding table is held resident across the loop (needed only at the start,
    but kept for re-use). MXFP4 expert dequant happens per layer into transient
    BF16 matrices freed when the layer finishes."""

    var sharded: ShardedSafeTensors
    var config: GptOssConfig

    def __init__(out self, var sharded: ShardedSafeTensors, config: GptOssConfig):
        self.sharded = sharded^
        self.config = config

    @staticmethod
    def load(dir: String, config: GptOssConfig, ctx: DeviceContext) raises -> GptOssEncoder:
        """Open the sharded text_encoder directory (mmap; no full H2D). Weights
        are streamed per layer in encode(). Mirrors the Rust streaming loader's
        intent (gpt_oss_encoder.rs load_from_directory) but defers H2D to the
        per-layer loop so peak VRAM stays bounded."""
        var st = ShardedSafeTensors.open(dir)
        return GptOssEncoder(st^, config)

    def _load_layer(self, layer_idx: Int, ctx: DeviceContext) raises -> LayerBlock:
        """H2D every tensor of one layer. Attention/router/embed weights are
        BF16 (from_view); MoE expert blocks/scales are U8 (from_view_raw) for
        on-GPU dequant; biases are F32 -> kept F32 (added in F32... but our
        add_row_bias is BF16; we convert biases to BF16 on load to match)."""
        var p = String("model.layers.") + String(layer_idx) + "."
        var block = LayerBlock()
        for ref nm in self.sharded.names():
            if not nm.startswith(p):
                continue
            var tv = self.sharded.tensor_view(nm)
            if tv.dtype == STDtype.U8:
                var t = Tensor.from_view_raw(tv, ctx)
                block[nm] = ArcPointer(t^)
            elif tv.dtype == STDtype.F32:
                # gate_up_proj_bias / down_proj_bias: F32 -> BF16.
                var t = Tensor.from_view_as_bf16(tv, ctx)
                block[nm] = ArcPointer(t^)
            else:
                var t = Tensor.from_view(tv, ctx)
                block[nm] = ArcPointer(t^)
        return block^

    def _bw(self, block: LayerBlock, name: String) raises -> ref [block] Tensor:
        if name not in block:
            raise Error(String("missing layer weight: ") + name)
        return block[name][]

    def _attention(
        self,
        block: LayerBlock,
        layer_idx: Int,
        normed: Tensor,  # [1,S,hidden] BF16
        cos_q: Tensor,
        sin_q: Tensor,
        cos_k: Tensor,
        sin_k: Tensor,
        real_len: Int,
        seq: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim
        var scale = Float32(1.0) / Float32(dh) ** 0.5
        var p = String("model.layers.") + String(layer_idx) + ".self_attn."

        ref qw = self._bw(block, p + "q_proj.weight")
        ref qb = self._bw(block, p + "q_proj.bias")
        ref kw = self._bw(block, p + "k_proj.weight")
        ref kb = self._bw(block, p + "k_proj.bias")
        ref vw = self._bw(block, p + "v_proj.weight")
        ref vb = self._bw(block, p + "v_proj.bias")
        var q = linear(normed, qw, Optional[Tensor](_clone(qb, ctx)), ctx)
        var k = linear(normed, kw, Optional[Tensor](_clone(kb, ctx)), ctx)
        var v = linear(normed, vw, Optional[Tensor](_clone(vb, ctx)), ctx)

        q = _reshape(q, [1, seq, h, dh], ctx)
        k = _reshape(k, [1, seq, h_kv, dh], ctx)
        v = _reshape(v, [1, seq, h_kv, dh], ctx)

        # RoPE half-split (NO qk-norm in GPT-OSS).
        q = rope_halfsplit(q, cos_q, sin_q, ctx)
        k = rope_halfsplit(k, cos_k, sin_k, ctx)

        var k_rep = _repeat_kv(k^, h, h_kv, ctx)
        var v_rep = _repeat_kv(v^, h, h_kv, ctx)

        # Sinks are a checkpoint tensor boundary. Keep them in q storage dtype;
        # only scalar reads cast to F32 inside the SDPA kernel.
        ref sinks_w = self._bw(block, p + "sinks")
        var sinks_q: Tensor
        if sinks_w.dtype() == q.dtype():
            sinks_q = _clone(sinks_w, ctx)
        else:
            sinks_q = cast_tensor(_clone(sinks_w, ctx), q.dtype(), ctx)

        var attn = _sdpa_with_sinks(
            q, k_rep, v_rep, sinks_q, seq, h, dh, scale,
            real_len, _is_sliding(layer_idx), cfg.sliding_window, ctx,
        )
        attn = _reshape(attn, [1, seq, h * dh], ctx)

        ref ow = self._bw(block, p + "o_proj.weight")
        ref ob = self._bw(block, p + "o_proj.bias")
        return linear(attn, ow, Optional[Tensor](_clone(ob, ctx)), ctx)

    def _moe(
        self,
        block: LayerBlock,
        layer_idx: Int,
        normed: Tensor,  # [1,S,hidden] BF16
        seq: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var hidden = cfg.hidden_size
        var inter = cfg.intermediate_size
        var e = cfg.num_experts
        var k = cfg.top_k
        var p = String("model.layers.") + String(layer_idx) + ".mlp."

        var t = seq  # B=1
        var x2 = _reshape(normed, [t, hidden], ctx)

        # Router: linear(hidden -> E) + bias.
        ref rw = self._bw(block, p + "router.weight")
        ref rb = self._bw(block, p + "router.bias")
        var logits = linear(x2, rw, Optional[Tensor](_clone(rb, ctx)), ctx)
        # top-k softmax routing (gating = softmax over the k picked logits).
        var plan = top_k_router(logits, k, ctx)

        # Expert weights (MXFP4 packed).
        ref gu_blocks = self._bw(block, p + "experts.gate_up_proj_blocks")  # U8 [E,5760,90,16]
        ref gu_scales = self._bw(block, p + "experts.gate_up_proj_scales")  # U8 [E,5760,90]
        ref gu_bias = self._bw(block, p + "experts.gate_up_proj_bias")  # BF16 [E,5760]
        ref dn_blocks = self._bw(block, p + "experts.down_proj_blocks")  # U8 [E,2880,90,16]
        ref dn_scales = self._bw(block, p + "experts.down_proj_scales")  # U8 [E,2880,90]
        ref dn_bias = self._bw(block, p + "experts.down_proj_bias")  # BF16 [E,2880]

        var two_inter = 2 * inter

        # Accumulator [T, hidden] in model storage dtype. gated_scatter_add uses
        # private F32 atomic scratch for BF16 and writes back before returning.
        var storage = x2.dtype()
        var acc_buf = ctx.enqueue_create_buffer[DType.uint8](
            t * hidden * storage.byte_size()
        )
        ctx.enqueue_memset[DType.uint8](acc_buf, 0)
        ctx.synchronize()
        var accum = Tensor(acc_buf^, [t, hidden], storage)

        # Process each expert: gather its routed tokens, run FFN, scatter.
        # We loop experts (mirrors the Rust looped-expert semantics) and reuse
        # the slot-based router plan: a slot s (token-major, t*k+j) routes
        # plan.expert_ids[s] with gating plan.gating[s].
        var n_slots = len(plan.expert_ids)
        for ei in range(e):
            # Collect tokens routed to expert ei (by source token index) and
            # their slots for scatter.
            var tok_rows = List[Int]()
            var slot_ids = List[Int]()
            for s in range(n_slots):
                if plan.expert_ids[s] == ei:
                    tok_rows.append(s // k)  # source token
                    slot_ids.append(s)
            var ntok = len(tok_rows)
            if ntok == 0:
                continue

            # Dequant this expert's gate_up [5760, hidden] and down [hidden, inter].
            # gate_up_blocks[ei]: [5760, 90, 16] -> BF16 [5760, hidden(=2880)].
            var gu_blk_e = t_slice(gu_blocks, 0, ei, 1, ctx)  # [1,5760,90,16]
            gu_blk_e = _reshape(gu_blk_e, [two_inter, hidden // 32, 16], ctx)
            var gu_sc_e = t_slice(gu_scales, 0, ei, 1, ctx)  # [1,5760,90]
            gu_sc_e = _reshape(gu_sc_e, [two_inter, hidden // 32], ctx)
            var gate_up_w = _mxfp4_dequant_expert(gu_blk_e, gu_sc_e, ctx)  # [5760, hidden]

            var dn_blk_e = t_slice(dn_blocks, 0, ei, 1, ctx)  # [1,2880,90,16]
            dn_blk_e = _reshape(dn_blk_e, [hidden, inter // 32, 16], ctx)
            var dn_sc_e = t_slice(dn_scales, 0, ei, 1, ctx)
            dn_sc_e = _reshape(dn_sc_e, [hidden, inter // 32], ctx)
            var down_w = _mxfp4_dequant_expert(dn_blk_e, dn_sc_e, ctx)  # [hidden, inter]

            # Gather this expert's input rows [ntok, hidden] from x2 on device.
            var xe = gather_rows(x2, tok_rows, ctx)

            # gate_up = xe @ gate_up_w^T + bias  ([ntok, 2*inter]).
            var gu_bias_e = t_slice(gu_bias, 0, ei, 1, ctx)  # [1, 5760]
            gu_bias_e = _reshape(gu_bias_e, [two_inter], ctx)
            var gate_up = linear(xe, gate_up_w, None, ctx)
            gate_up = _add_row_bias(gate_up, gu_bias_e, ctx)

            # Activation -> [ntok, inter].
            var act = _gptoss_act(
                gate_up, inter, cfg.swiglu_limit, cfg.swiglu_alpha, ctx
            )

            # down = act @ down_w^T + bias  ([ntok, hidden]).
            var dn_bias_e = t_slice(dn_bias, 0, ei, 1, ctx)  # [1, 2880]
            dn_bias_e = _reshape(dn_bias_e, [hidden], ctx)
            var down_out = linear(act, down_w, None, ctx)
            down_out = _add_row_bias(down_out, dn_bias_e, ctx)

            # Weighted scatter-add into accum: accum[tok_rows[r]] +=
            #   down_out[r] * gating[slot]. Use the foundation gated_scatter_add
            # with per-row indices = source token, gating = slot gating.
            var gating_e = List[Float32]()
            var idx_e = List[Int]()
            for r in range(ntok):
                gating_e.append(plan.gating[slot_ids[r]])
                idx_e.append(tok_rows[r])
            gated_scatter_add(down_out, gating_e, idx_e, accum, ctx)

        return _reshape(accum, [1, seq, hidden], ctx)

    def _layer(
        self,
        block: LayerBlock,
        layer_idx: Int,
        hidden_in: Tensor,
        cos_q: Tensor,
        sin_q: Tensor,
        cos_k: Tensor,
        sin_k: Tensor,
        real_len: Int,
        seq: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var eps = cfg.rms_norm_eps
        var p = String("model.layers.") + String(layer_idx) + "."

        ref pre_w = self._bw(block, p + "input_layernorm.weight")
        var normed = rms_norm(hidden_in, pre_w, eps, ctx)
        var attn_out = self._attention(
            block, layer_idx, normed, cos_q, sin_q, cos_k, sin_k,
            real_len, seq, ctx,
        )
        var h1 = t_add(hidden_in, attn_out, ctx)

        ref post_w = self._bw(block, p + "post_attention_layernorm.weight")
        var normed2 = rms_norm(h1, post_w, eps, ctx)
        var moe_out = self._moe(block, layer_idx, normed2, seq, ctx)
        return t_add(h1, moe_out, ctx)

    def encode(
        self, token_ids: List[Int], extract_layers: List[Int], ctx: DeviceContext
    ) raises -> List[ArcPointer[Tensor]]:
        """Run the streamed layer loop up to max(extract_layers), capturing the
        post-residual hidden state (PRE-final-norm) at each requested layer.
        Returns one [1,S,hidden] BF16 tensor per requested layer, in ASCENDING
        layer order (capture list is sorted+deduped to match the Rust contract).
        Each layer's weights are loaded, run, and freed before the next."""
        var cfg = self.config
        var seq = len(token_ids)
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim
        var half = dh // 2

        # max layer to run.
        var max_layer = 0
        for i in range(len(extract_layers)):
            if extract_layers[i] > max_layer:
                max_layer = extract_layers[i]

        # Sort + dedup the requested capture layers so output order is
        # deterministic (ascending), matching the Rust contract (GptOssEncoder::new
        # sorts+dedups selected_layers) regardless of caller input order.
        var sorted_layers = List[Int]()
        for i in range(len(extract_layers)):
            var li = extract_layers[i]
            var seen = False
            for j in range(len(sorted_layers)):
                if sorted_layers[j] == li:
                    seen = True
                    break
            if not seen:
                # insertion sort into ascending position
                var pos = len(sorted_layers)
                for j in range(len(sorted_layers)):
                    if li < sorted_layers[j]:
                        pos = j
                        break
                sorted_layers.insert(pos, li)

        # right-pad detection (pad id 199999 per config.json).
        var pad_id = 199999
        var real_len = seq
        for i in range(seq):
            if token_ids[i] == pad_id:
                real_len = i
                break

        # YaRN RoPE tables (BF16, ordered (position, head)).
        var q_tab = _build_yarn_tables(cfg, seq, h)
        var k_tab = _build_yarn_tables(cfg, seq, h_kv)
        var cos_q = Tensor.from_host(q_tab[0], [seq * h * half], STDtype.BF16, ctx)
        var sin_q = Tensor.from_host(q_tab[1], [seq * h * half], STDtype.BF16, ctx)
        var cos_k = Tensor.from_host(k_tab[0], [seq * h_kv * half], STDtype.BF16, ctx)
        var sin_k = Tensor.from_host(k_tab[1], [seq * h_kv * half], STDtype.BF16, ctx)

        # Embedding: load table, gather, then free table.
        var emb_tv = self.sharded.tensor_view(String("model.embed_tokens.weight"))
        var emb_table = Tensor.from_view(emb_tv, ctx)
        var hidden_state = _embed(emb_table, token_ids, ctx)
        _ = emb_table^  # free the 1.1 GB embedding table immediately.

        var captures = List[ArcPointer[Tensor]]()
        # placeholder ordering: capture in extract_layers order at the end.
        var captured_at = Dict[Int, ArcPointer[Tensor]]()

        for li in range(max_layer + 1):
            var block = self._load_layer(li, ctx)
            hidden_state = self._layer(
                block, li, hidden_state, cos_q, sin_q, cos_k, sin_k,
                real_len, seq, ctx,
            )
            # capture if requested.
            for j in range(len(sorted_layers)):
                if sorted_layers[j] == li:
                    captured_at[li] = ArcPointer(_clone(hidden_state, ctx))
            # free the layer block (drops all ArcPointers -> frees VRAM).
            _ = block^

        for j in range(len(sorted_layers)):
            var li = sorted_layers[j]
            if li not in captured_at:
                raise Error("encode: capture missing for a requested layer")
            captures.append(captured_at[li])
        return captures^
