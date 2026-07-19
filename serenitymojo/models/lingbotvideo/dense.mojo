# models/lingbotvideo/dense.mojo — LingBot-Video-Dense-1.3B RESIDENT backbone.
#
# Dense-1.3B == the MoE architecture (models/lingbotvideo/backbone.mojo) EXCEPT:
#   * FFN is a plain DENSE SwiGLU MLP (LingBotVideoMLP), NOT the sparse MoE —
#     down(silu(gate(x)) * up(x)); gate/up: Linear(2048->6144, no bias); down:
#     Linear(6144->2048, no bias). This is EXACTLY the shared-expert SwiGLU in
#     moe.mojo, just at intermediate 6144 not 768.
#   * depth 24 (not 48).
#   * FITS 16GB VRAM (2.6GB bf16) -> RESIDENT: load all 24 blocks to VRAM ONCE,
#     NO per-block stream/free. The 80 pipeline forwards then run FAST.
#
# EVERYTHING ELSE is reused verbatim from backbone.mojo: the attention forward
# (lingbot_attention), the AdaLN 6-way modulation + gated residuals + the four
# block RMSNorms (_rms_norm_bf16_host), the B1 embeddings (lingbot_patchify /
# lingbot_patch_embed / lingbot_text_embed / lingbot_build_rope / lingbot_time_embed),
# and the POST head (norm_out LayerNorm no-affine -> final modulation -> proj_out
# -> unpatchify). Weight names identical to the MoE except the FFN is
# blocks.{i}.ffn.{gate_proj,up_proj,down_proj}.weight (dense — no router/experts).
#
# Mojo 1.0.0b1, NVIDIA GPU. INFERENCE ONLY.

from std.math import sqrt, tanh
from std.memory import ArcPointer
from std.time import perf_counter_ns
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear, linear_bias
from serenitymojo.ops.norm import rms_norm, layer_norm_no_affine
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_flash import sdpa_flash_train_fwd
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne, torch_bf16_rne_value
from serenitymojo.models.lingbotvideo.scheduler import FlowUniPCMultistepScheduler
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig,
    lingbot_patchify,
    lingbot_patch_embed,
    lingbot_build_rope,
    _lw,
)

comptime _TArc = ArcPointer[Tensor]

comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _EW_TPB = 256


# ═════════════════════════════════════════════════════════════════════════════
# GPU AdaLN kernels (perf fix, 2026-07-09). The reference block math is:
#   mod = temb6 + scale_shift_table (FP32); chunk 6; gate=tanh, scale=1+scale
#   attn_in = (norm(x) * scale + shift).to(bf16)
#   x = x + (gate * norm(sub_out)).to(x.dtype)
# The old implementation downloaded temb6 (S×12288 F32) and ran every one of
# these elementwise ops in host loops — ~250M host element-ops PER BLOCK at
# T2V resolution (S≈7.9K), 3.5 s/block. These kernels read temb6 (a single
# modulation row — the timestep is per-batch, so the reference expand(B,S,·)
# makes every token row identical) and the scale_shift_table DIRECTLY with the
# right chunk column offsets, keep all math F32, and round to bf16 only at the
# stores — the exact same F32→bf16 boundaries the host path had.
# ═════════════════════════════════════════════════════════════════════════════


# attn/ffn input modulate: out = bf16( f32(normed) * (1 + (temb[scale_off+j] +
# sst[scale_off+j])) + (temb[shift_off+j] + sst[shift_off+j]) ).
def _adaln_modulate_kernel(
    normed: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],  # [S, H] bf16
    temb: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],     # [K] f32 row
    sst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],      # [K] f32 table
    outp: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],    # [S, H] bf16
    s: Int,
    h: Int,
    shift_off: Int,
    scale_off: Int,
):
    var idx = Int(global_idx.x)
    if idx < s * h:
        var t = idx // h
        var j = idx % h
        var nv = rebind[Scalar[DType.bfloat16]](normed[t, j]).cast[
            DType.float32
        ]()
        var sc = rebind[Scalar[DType.float32]](temb[scale_off + j]) + rebind[
            Scalar[DType.float32]
        ](sst[scale_off + j])
        var shv = rebind[Scalar[DType.float32]](temb[shift_off + j]) + rebind[
            Scalar[DType.float32]
        ](sst[shift_off + j])
        var v = nv * (Float32(1.0) + sc) + shv
        outp[t, j] = rebind[outp.element_type](v.cast[DType.bfloat16]())


def _adaln_modulate(
    normed: Tensor,     # [.., S, H] bf16 (S*H elements)
    temb_row: Tensor,   # [1, K] f32
    sst: Tensor,        # [1, K] or [K] f32 (same K)
    s: Int,
    h: Int,
    shift_off: Int,
    scale_off: Int,
    var out_shape: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](s * h * 2)
    var n_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](s, h))
    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](temb_row.numel()))
    var N = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        normed.buf.unsafe_ptr().bitcast[BFloat16](), n_rl
    )
    var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        temb_row.buf.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        sst.buf.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), n_rl
    )
    var grid = (s * h + _EW_TPB - 1) // _EW_TPB
    ctx.enqueue_function[_adaln_modulate_kernel, _adaln_modulate_kernel](
        N, T, C, O, s, h, shift_off, scale_off,
        grid_dim=grid, block_dim=_EW_TPB,
    )
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


# gated residual: out = bf16( f32(x) + f32(bf16( tanh(temb[gate_off+j] +
# sst[gate_off+j]) * f32(normed) )) ). The inner bf16 round mirrors the host
# `.to(x.dtype)` on the delta; the outer add is F32-accumulate, bf16 store —
# identical boundaries to the old host delta + ops.tensor_algebra.add.
def _adaln_gate_residual_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],       # [S, H] bf16
    normed: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],  # [S, H] bf16
    temb: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],     # [K] f32 row
    sst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],      # [K] f32 table
    outp: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],    # [S, H] bf16
    s: Int,
    h: Int,
    gate_off: Int,
):
    var idx = Int(global_idx.x)
    if idx < s * h:
        var t = idx // h
        var j = idx % h
        var g = tanh(
            rebind[Scalar[DType.float32]](temb[gate_off + j])
            + rebind[Scalar[DType.float32]](sst[gate_off + j])
        )
        var nv = rebind[Scalar[DType.bfloat16]](normed[t, j]).cast[
            DType.float32
        ]()
        var d = (g * nv).cast[DType.bfloat16]()
        var xv = rebind[Scalar[DType.bfloat16]](x[t, j]).cast[DType.float32]()
        var res = xv + d.cast[DType.float32]()
        outp[t, j] = rebind[outp.element_type](res.cast[DType.bfloat16]())


def _adaln_gate_residual(
    x: Tensor,          # [.., S, H] bf16
    normed: Tensor,     # [.., S, H] bf16
    temb_row: Tensor,   # [1, K] f32
    sst: Tensor,        # [1, K] f32
    s: Int,
    h: Int,
    gate_off: Int,
    var out_shape: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](s * h * 2)
    var n_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](s, h))
    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](temb_row.numel()))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), n_rl
    )
    var N = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        normed.buf.unsafe_ptr().bitcast[BFloat16](), n_rl
    )
    var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        temb_row.buf.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        sst.buf.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), n_rl
    )
    var grid = (s * h + _EW_TPB - 1) // _EW_TPB
    ctx.enqueue_function[
        _adaln_gate_residual_kernel, _adaln_gate_residual_kernel
    ](X, N, T, C, O, s, h, gate_off, grid_dim=grid, block_dim=_EW_TPB)
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


# per-head rope tiling on device: out[(t*heads + hh), i] = tbl[t, i]. Replaces
# the host `_tile_rope_perhead` (S*heads*half host loop + up/download) and is
# hoisted to ONCE per forward — the tables are identical across all 24 blocks.
def _tile_rope_kernel(
    tbl: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # [S, half]
    outp: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [S*heads, half]
    s: Int,
    heads: Int,
    half: Int,
):
    var idx = Int(global_idx.x)
    if idx < s * heads * half:
        var i = idx % half
        var row = idx // half
        var t = row // heads
        outp[row, i] = rebind[outp.element_type](
            rebind[Scalar[DType.float32]](tbl[t, i])
        )


def _tile_rope_gpu(
    tbl: Tensor, s: Int, heads: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](s * heads * half * 4)
    var in_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](s, half))
    var out_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](s * heads, half))
    var Ti = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        tbl.buf.unsafe_ptr().bitcast[Float32](), in_rl
    )
    var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), out_rl
    )
    var total = s * heads * half
    var grid = (total + _EW_TPB - 1) // _EW_TPB
    ctx.enqueue_function[_tile_rope_kernel, _tile_rope_kernel](
        Ti, O, s, heads, half, grid_dim=grid, block_dim=_EW_TPB
    )
    return Tensor(out_buf^, [s * heads, half], STDtype.F32)


# device concat of two bf16 row blocks (video tokens ++ text tokens) — replaces
# the host joint-build (two to_host + S*H host writes + re-upload).
def _concat_flat_bf16_kernel(
    a: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    outp: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    na: Int,
    total: Int,
):
    var idx = Int(global_idx.x)
    if idx < total:
        if idx < na:
            outp[idx] = rebind[outp.element_type](
                rebind[Scalar[DType.bfloat16]](a[idx])
            )
        else:
            outp[idx] = rebind[outp.element_type](
                rebind[Scalar[DType.bfloat16]](b[idx - na])
            )


def _concat_rows_bf16(
    a: Tensor, b: Tensor, var out_shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var na = a.numel()
    var nb = b.numel()
    var total = na + nb
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](total * 2)
    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](na))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nb))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[BFloat16](), a_rl
    )
    var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl
    )
    var grid = (total + _EW_TPB - 1) // _EW_TPB
    ctx.enqueue_function[_concat_flat_bf16_kernel, _concat_flat_bf16_kernel](
        A, B, O, na, total, grid_dim=grid, block_dim=_EW_TPB
    )
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


# LingBotVideoRMSNorm on device, mirroring `_rms_norm_bf16_host` EXACTLY minus
# the host round trip: upcast bf16 x -> F32, rms_norm with the F32 gamma (F32
# accumulate — the same `_rms_norm_kernel_f32`), then `.to(bf16)`.
def _rms_norm_bf16_dev(
    x: Tensor, gamma_f32: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    var xf = cast_tensor(x, STDtype.F32, ctx)
    var nf = rms_norm(xf, gamma_f32, eps, ctx)
    return cast_tensor(nf, STDtype.BF16, ctx)


# Minimum comptime S for the cuDNN flash SDPA path. Below this the cuBLAS
# math-mode sdpa_nomask (the oracle-gated path) is used — small-S probes then
# never reference the cshim symbol, so they run under plain `mojo run` with no
# extra link flags. At/above it (T2V, S≈7.9K) attention dispatches to the
# flame cuDNN flash shim (ops/attention_flash): measured 4.9 ms vs 34 ms math
# vs 2.21 s online-softmax tiled per block at S=7860 on the RTX 5080. Flash is
# a different summation order (flash-vs-math cos 0.999998 at this shape) —
# accepted, the pixel gate confirms. Runs needing it must link the shim:
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa ...
comptime _FLASH_MIN_S = 4096


# lingbot_attention (backbone.mojo) with PRE-TILED rope tables [S*heads, 64] —
# the tables are per-forward invariants, tiled once by the caller instead of
# host-tiled twice per block. SDPA dispatch: cuDNN flash at S >= _FLASH_MIN_S,
# cuBLAS math-mode otherwise (the old online-softmax tiled path was the 2.2
# s/block bottleneck and is no longer used here). USE_TILED is retained for
# source compatibility only.
def _lingbot_attention_pretiled[
    S: Int,
    USE_TILED: Bool = True,
](
    x: Tensor,          # [1, S, 2048] bf16
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    b_o: Tensor,
    norm_q_w: Tensor,   # [128] f32
    norm_k_w: Tensor,
    cos_h: Tensor,      # [S*heads, 64] f32 (pre-tiled)
    sin_h: Tensor,
    cfg: LingBotAttnConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    var h = cfg.heads
    var hd = cfg.head_dim
    var q = linear(x, w_q, None, ctx)
    var k = linear(x, w_k, None, ctx)
    var v = linear(x, w_v, None, ctx)
    var q_rows = reshape(q, [S * h, hd], ctx)
    var k_rows = reshape(k, [S * h, hd], ctx)
    var q_n = rms_norm(q_rows, norm_q_w, cfg.eps, ctx)
    var k_n = rms_norm(k_rows, norm_k_w, cfg.eps, ctx)
    var q_r = rope_interleaved(q_n, cos_h, sin_h, ctx)
    var k_r = rope_interleaved(k_n, cos_h, sin_h, ctx)
    var q4 = reshape(q_r, [1, S, h, hd], ctx)
    var k4 = reshape(k_r, [1, S, h, hd], ctx)
    var v4 = reshape(v, [1, S, h, hd], ctx)
    var scale = Float32(1.0) / Float32(sqrt(Float64(hd)))
    var attn: Tensor
    comptime if S >= _FLASH_MIN_S:
        var fwd = sdpa_flash_train_fwd[1, S, 16, 128](q4, k4, v4, scale, ctx)
        # fwd.o is a zero-copy view into fwd.o_pad — own the bytes before the
        # SdpaFlashFwd scratch set is dropped.
        attn = fwd.o.clone(ctx)
    else:
        attn = sdpa_nomask[1, S, 16, 128](q4, k4, v4, scale, ctx)
    var attn_2d = reshape(attn, [S, cfg.hidden], ctx)
    var out = linear_bias(attn_2d, w_o, b_o, ctx)
    return reshape(out, [1, S, cfg.hidden], ctx)


# ═════════════════════════════════════════════════════════════════════════════
# Dense FFN = LingBotVideoMLP: down(silu(gate(x)) * up(x)). Reuses the exact
# shared-expert SwiGLU path from moe.mojo (just intermediate 6144, not 768).
# ═════════════════════════════════════════════════════════════════════════════
def lingbot_dense_ffn(
    ffn_in_bf16: Tensor,   # [S, H] bf16 tokens
    gate_w: Tensor,        # [I, H] bf16 (gate_proj.weight, I=6144)
    up_w: Tensor,          # [I, H] bf16 (up_proj.weight)
    down_w: Tensor,        # [H, I] bf16 (down_proj.weight)
    ctx: DeviceContext,
) raises -> Tensor:
    """Dense SwiGLU MLP forward. Returns [S, H] bf16."""
    var g = linear(ffn_in_bf16, gate_w, None, ctx)  # [S, I]
    var u = linear(ffn_in_bf16, up_w, None, ctx)    # [S, I]
    var mid = swiglu(g, u, ctx)                      # [S, I]  silu(g)*u
    return linear(mid, down_w, None, ctx)           # [S, H]


# ═════════════════════════════════════════════════════════════════════════════
# One full DENSE LingBotVideoBlock forward. AdaLN 6-way modulation + attn + dense
# SwiGLU FFN + two gated residuals. Mirrors LingBotVideoBlock.forward with the
# num_experts=0 branch (self.ffn = LingBotVideoMLP). Numerically identical dtype
# contract to lingbot_video_block (backbone.mojo) — only the FFN differs.
#
# ENTIRELY ON-GPU (2026-07-09 perf fix): the AdaLN modulation, the four block
# RMSNorms, and both gated residuals run as device kernels reading the single
# temb modulation row + scale_shift_table directly (chunk k of 6 = columns
# k*2048..(k+1)*2048). F32 math, bf16 only at stores — same boundaries as the
# old host loops (which took ~3.5 s/block at S≈7.9K; now ~ms).
# ═════════════════════════════════════════════════════════════════════════════
def lingbot_video_dense_block[
    S: Int,
    USE_TILED: Bool = True,
](
    x_in: Tensor,               # [1, S, 2048] bf16
    temb_row: Tensor,           # [1, 12288] f32 (single modulation row; the
                                #   timestep is per-batch so all S token rows
                                #   of the reference temb6 are identical)
    scale_shift_table: Tensor,  # [1, 12288] f32
    w_q: Tensor,                # attn: [2048,2048] bf16
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    b_o: Tensor,                # [2048] bf16
    norm_q_w: Tensor,           # [128] f32
    norm_k_w: Tensor,
    cos_h: Tensor,              # [S*heads, 64] f32 PRE-TILED (per-forward hoist)
    sin_h: Tensor,
    norm1_w: Tensor,            # block norms: [2048] f32
    norm2_w: Tensor,
    norm_post_attn_w: Tensor,
    norm_post_ffn_w: Tensor,
    gate_w: Tensor,             # dense FFN: [6144,2048] bf16
    up_w: Tensor,               # [6144,2048] bf16
    down_w: Tensor,             # [2048,6144] bf16
    cfg: LingBotAttnConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """One full DENSE LingBotVideoBlock forward. Returns [1, S, 2048] bf16."""
    var hdim = cfg.hidden       # 2048
    var eps = cfg.eps

    # ── attention sub-block: attn_in = (norm1(x)*scale_msa + shift_msa).to(bf16)
    var n1 = _rms_norm_bf16_dev(x_in, norm1_w, eps, ctx)
    var attn_in = _adaln_modulate(
        n1, temb_row, scale_shift_table, S, hdim,
        0 * hdim, 1 * hdim, [1, S, hdim], ctx,
    )
    var attn_out = _lingbot_attention_pretiled[S, USE_TILED](
        attn_in, w_q, w_k, w_v, w_o, b_o, norm_q_w, norm_k_w,
        cos_h, sin_h, cfg, ctx,
    )  # [1, S, 2048] bf16

    # x = x + (gate_msa * norm_post_attn(attn_out)).to(x.dtype)
    var npa = _rms_norm_bf16_dev(attn_out, norm_post_attn_w, eps, ctx)
    var x = _adaln_gate_residual(
        x_in, npa, temb_row, scale_shift_table, S, hdim,
        2 * hdim, [1, S, hdim], ctx,
    )

    # ── DENSE FFN sub-block: ffn_in = (norm2(x)*scale_mlp + shift_mlp).to(bf16)
    var n2 = _rms_norm_bf16_dev(x, norm2_w, eps, ctx)
    var ffn_in_bf16 = _adaln_modulate(
        n2, temb_row, scale_shift_table, S, hdim,
        3 * hdim, 4 * hdim, [S, hdim], ctx,
    )
    var ffn_out = lingbot_dense_ffn(ffn_in_bf16, gate_w, up_w, down_w, ctx)  # [S,2048] bf16

    # x = x + (gate_mlp * norm_post_ffn(ffn_out)).to(x.dtype)
    var npf = _rms_norm_bf16_dev(ffn_out, norm_post_ffn_w, eps, ctx)
    return _adaln_gate_residual(
        x, npf, temb_row, scale_shift_table, S, hdim,
        5 * hdim, [1, S, hdim], ctx,
    )  # [1, S, 2048] bf16


# ═════════════════════════════════════════════════════════════════════════════
# RESIDENT dense weights: all 24 blocks (15 weights each) + embedding/post,
# loaded to VRAM ONCE. Dropping the model frees ~2.6GB.
# ═════════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct LingBotDenseBlockW(Movable, Copyable):
    """One dense block's 15 weights (stored dtypes: bf16 bulk / f32 norms)."""

    var scale_shift_table: _TArc      # [1,12288] f32
    var w_q: _TArc                    # [2048,2048] bf16
    var w_k: _TArc
    var w_v: _TArc
    var w_o: _TArc
    var b_o: _TArc                    # [2048] bf16
    var norm_q_w: _TArc               # [128] f32
    var norm_k_w: _TArc
    var norm1_w: _TArc                # [2048] f32
    var norm2_w: _TArc
    var norm_post_attn_w: _TArc
    var norm_post_ffn_w: _TArc
    var gate_w: _TArc                 # dense FFN [6144,2048] bf16
    var up_w: _TArc                   # [6144,2048] bf16
    var down_w: _TArc                 # [2048,6144] bf16


def _load_dense_block(
    model: ShardedSafeTensors, i: Int, ctx: DeviceContext
) raises -> LingBotDenseBlockW:
    """Load dense block-i's 15 weights RESIDENT (stored dtypes)."""
    var p = String("blocks.") + String(i) + String(".")
    return LingBotDenseBlockW(
        _lw(model, p + "scale_shift_table", ctx),
        _lw(model, p + "attn.to_q.weight", ctx),
        _lw(model, p + "attn.to_k.weight", ctx),
        _lw(model, p + "attn.to_v.weight", ctx),
        _lw(model, p + "attn.to_out.weight", ctx),
        _lw(model, p + "attn.to_out.bias", ctx),
        _lw(model, p + "attn.norm_q.weight", ctx),
        _lw(model, p + "attn.norm_k.weight", ctx),
        _lw(model, p + "norm1.weight", ctx),
        _lw(model, p + "norm2.weight", ctx),
        _lw(model, p + "norm_post_attn.weight", ctx),
        _lw(model, p + "norm_post_ffn.weight", ctx),
        _lw(model, p + "ffn.gate_proj.weight", ctx),
        _lw(model, p + "ffn.up_proj.weight", ctx),
        _lw(model, p + "ffn.down_proj.weight", ctx),
    )


@fieldwise_init
struct LingBotDenseModel(Movable):
    """RESIDENT dense transformer: embedding/post weights + all `depth` blocks
    live on the GPU (built once via `load`)."""

    var pe_w: _TArc
    var pe_b: _TArc
    var t_norm_w: _TArc
    var t_l1_w: _TArc
    var t_l1_b: _TArc
    var t_l2_w: _TArc
    var t_l2_b: _TArc
    var ti_l1_w: _TArc
    var ti_l1_b: _TArc
    var ti_l2_w: _TArc
    var ti_l2_b: _TArc
    var tmod_w: _TArc
    var tmod_b: _TArc
    var nom_w: _TArc
    var nom_b: _TArc
    var proj_w: _TArc
    var proj_b: _TArc
    var blocks: List[LingBotDenseBlockW]
    var depth: Int

    @staticmethod
    def load(
        model: ShardedSafeTensors, depth: Int, ctx: DeviceContext
    ) raises -> LingBotDenseModel:
        var blocks = List[LingBotDenseBlockW]()
        for i in range(depth):
            blocks.append(_load_dense_block(model, i, ctx))
        return LingBotDenseModel(
            _lw(model, String("patch_embedder.weight"), ctx),
            _lw(model, String("patch_embedder.bias"), ctx),
            _lw(model, String("text_embedder.norm.weight"), ctx),
            _lw(model, String("text_embedder.linear_1.weight"), ctx),
            _lw(model, String("text_embedder.linear_1.bias"), ctx),
            _lw(model, String("text_embedder.linear_2.weight"), ctx),
            _lw(model, String("text_embedder.linear_2.bias"), ctx),
            _lw(model, String("time_embedder.linear_1.weight"), ctx),
            _lw(model, String("time_embedder.linear_1.bias"), ctx),
            _lw(model, String("time_embedder.linear_2.weight"), ctx),
            _lw(model, String("time_embedder.linear_2.bias"), ctx),
            _lw(model, String("time_modulation.1.weight"), ctx),
            _lw(model, String("time_modulation.1.bias"), ctx),
            _lw(model, String("norm_out_modulation.1.weight"), ctx),
            _lw(model, String("norm_out_modulation.1.bias"), ctx),
            _lw(model, String("proj_out.weight"), ctx),
            _lw(model, String("proj_out.bias"), ctx),
            blocks^,
            depth,
        )


@fieldwise_init
struct LingBotDenseOut(Movable):
    """Dense backbone outputs: pre-block joint (bf16), the requested per-block
    taps (bf16, in tap_indices order), and the final velocity (f32)."""

    var joint_preblock: _TArc     # [1,S,2048] bf16
    var taps: List[_TArc]         # each [1,S,2048] bf16
    var velocity: _TArc           # [1,Cout,T,H,W] f32


def lingbot_dense_backbone[
    S: Int,
    USE_TILED: Bool = True,
](
    latent: Tensor,          # [1,C,T,H,W] f32
    timestep: Tensor,        # [B] f32
    text_embeds: Tensor,     # [1,text_len,2560] f32 (bf16 values)
    m: LingBotDenseModel,    # RESIDENT weights (no streaming)
    grid_t: Int,
    grid_h: Int,
    grid_w: Int,
    text_len: Int,
    patch_f: Int,
    patch_h: Int,
    patch_w: Int,
    out_channels: Int,
    theta: Float32,
    tap_indices: List[Int],
    cfg: LingBotAttnConfig,
    ctx: DeviceContext,
) raises -> LingBotDenseOut:
    """Full dense LingBot-Video transformer forward, weights RESIDENT."""
    var hdim = cfg.hidden       # 2048
    var eps = cfg.eps

    # ── 1. embeddings -> joint (bf16) ─────────────────────────────────────────
    var pt_f32 = lingbot_patchify(latent, patch_f, patch_h, patch_w, ctx)
    var pt_bf16 = cast_tensor(pt_f32, STDtype.BF16, ctx)
    var x = lingbot_patch_embed(pt_bf16, m.pe_w[], m.pe_b[], ctx)   # [n_video,2048] bf16

    var te_bf16 = cast_tensor(text_embeds, STDtype.BF16, ctx)
    var te_f32 = cast_tensor(te_bf16, STDtype.F32, ctx)
    var tn_f32 = rms_norm(te_f32, m.t_norm_w[], eps, ctx)
    var tn_bf16 = cast_tensor(tn_f32, STDtype.BF16, ctx)
    var th1 = linear_bias(tn_bf16, m.t_l1_w[], m.t_l1_b[], ctx)
    var th1a = silu(th1, ctx)
    var text = linear_bias(th1a, m.t_l2_w[], m.t_l2_b[], ctx)       # [1,L,2048] bf16

    # joint = cat([x, text], dim=1) — device concat (byte-identical to the old
    # host round-trip concat).
    var joint = _concat_rows_bf16(x, text, [1, S, hdim], ctx)
    var joint_pre = joint.clone(ctx)

    # ── 2. rope tables (pre-tiled ONCE per forward) + time -> temb row ────────
    var rope = lingbot_build_rope(grid_t, grid_h, grid_w, text_len, theta, ctx)
    var half = cfg.head_dim // 2
    var cos_h = _tile_rope_gpu(rope[0], S, cfg.heads, half, ctx)
    var sin_h = _tile_rope_gpu(rope[1], S, cfg.heads, half, ctx)

    # time path (all F32). The reference expands t_emb to (B,S,2048) before
    # time_modulation, but timestep is per-batch so every token row of temb6 is
    # identical — compute the ONE row and let the block kernels broadcast it.
    var emb = timestep_embedding(timestep, 256, ctx, Float32(10000.0), STDtype.F32)  # [1,256]
    var h1 = linear_bias(emb, m.ti_l1_w[], m.ti_l1_b[], ctx)   # [1,2048]
    var h1a = silu(h1, ctx)
    var t_emb = linear_bias(h1a, m.ti_l2_w[], m.ti_l2_b[], ctx)  # [1,2048] f32
    var sil_row = silu(t_emb, ctx)                               # [1,2048] f32
    var temb_row = linear_bias(sil_row, m.tmod_w[], m.tmod_b[], ctx)  # [1,12288] f32

    # ── 3. run the resident blocks (NO stream/free) ───────────────────────────
    var taps = List[_TArc]()
    for i in range(m.depth):
        var w = m.blocks[i].copy()
        var out = lingbot_video_dense_block[S, USE_TILED](
            joint, temb_row, w.scale_shift_table[],
            w.w_q[], w.w_k[], w.w_v[], w.w_o[], w.b_o[],
            w.norm_q_w[], w.norm_k_w[], cos_h, sin_h,
            w.norm1_w[], w.norm2_w[], w.norm_post_attn_w[], w.norm_post_ffn_w[],
            w.gate_w[], w.up_w[], w.down_w[],
            cfg, ctx,
        )  # [1,S,2048] bf16
        for k in range(len(tap_indices)):
            if tap_indices[k] == i:
                taps.append(_TArc(out.clone(ctx)))
        joint = out^

    # ── 4. POST head (identical math to the MoE backbone, on device) ─────────
    # final_mod = SiLU(t_emb) -> Linear(2048->4096); one row (see temb note).
    var fm_row = linear_bias(sil_row, m.nom_w[], m.nom_b[], ctx)  # [1,4096] f32

    # final_hidden = norm_out(joint)*(1+scale) + shift, FP32 mod -> bf16.
    # Reuses the AdaLN modulate kernel with a zero table: fm layout is
    # [shift(0..H) | scale(H..2H)].
    var ln = layer_norm_no_affine(joint, eps, ctx)  # [1,S,2048] bf16
    var twoh = 2 * hdim
    var zl = List[Float32]()
    zl.resize(twoh, Float32(0.0))
    var ztab = Tensor.from_host(zl, [twoh], STDtype.F32, ctx)
    var fh_bf16 = _adaln_modulate(
        ln, fm_row, ztab, S, hdim, 0, hdim, [S, hdim], ctx
    )
    var projected = linear_bias(fh_bf16, m.proj_w[], m.proj_b[], ctx)  # [S,64] bf16
    var proj_host = projected.to_host(ctx)

    # unpatchify (inverse of lingbot_patchify).
    var Tt = grid_t * patch_f
    var Hh = grid_h * patch_h
    var Ww = grid_w * patch_w
    var pd = patch_f * patch_h * patch_w * out_channels
    var vel = List[Float32]()
    vel.resize(out_channels * Tt * Hh * Ww, Float32(0.0))
    for f in range(grid_t):
        for h in range(grid_h):
            for w in range(grid_w):
                var token = (f * grid_h + h) * grid_w + w
                for pfi in range(patch_f):
                    for phi in range(patch_h):
                        for pwi in range(patch_w):
                            for ci in range(out_channels):
                                var feat = (
                                    ((pfi * patch_h + phi) * patch_w + pwi)
                                    * out_channels + ci
                                )
                                var tt = f * patch_f + pfi
                                var hh = h * patch_h + phi
                                var ww = w * patch_w + pwi
                                var out_off = ((ci * Tt + tt) * Hh + hh) * Ww + ww
                                vel[out_off] = proj_host[token * pd + feat]
    var velocity = Tensor.from_host(
        vel, [1, out_channels, Tt, Hh, Ww], STDtype.F32, ctx
    )

    return LingBotDenseOut(_TArc(joint_pre^), taps^, _TArc(velocity^))


# ═════════════════════════════════════════════════════════════════════════════
# RESIDENT dense T2I generate loop. Mirrors pipeline.lingbot_t2i_generate (same
# CFG FlowUniPC denoise, same f32-latent loop with the ops/torch_bf16 RNE cast at
# the backbone input), but the model is loaded RESIDENT ONCE and the two forwards
# per step reuse it (NO per-block streaming) — so each step runs in ~1s, not
# minutes. Returns per-step + final latents.
# ═════════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct LingBotDenseT2IOut(Movable):
    var step_latents: List[List[Float32]]   # latent ENTERING each step
    var final_latent: List[Float32]         # after the last step
    var lat_shape: List[Int]                # [1,16,1,lh,lw]


def _round_latent_bf16_dense(latent_f32: Tensor, ctx: DeviceContext) raises -> Tensor:
    var bf = torch_f32_to_bf16_rne(latent_f32, ctx)
    return cast_tensor(bf, STDtype.F32, ctx)


def lingbot_t2i_generate_dense[
    S_COND: Int,
    S_UNCOND: Int,
](
    init_latent: Tensor,       # [1,16,1,lh,lw] F32
    prompt_embeds: Tensor,     # [1,L_cond,2560]  F32 (bf16 values)
    neg_embeds: Tensor,        # [1,L_uncond,2560] F32 (bf16 values)
    m: LingBotDenseModel,      # RESIDENT weights (loaded once)
    grid_t: Int,
    grid_h: Int,
    grid_w: Int,
    L_cond: Int,
    L_uncond: Int,
    patch_f: Int,
    patch_h: Int,
    patch_w: Int,
    out_channels: Int,
    theta: Float32,
    num_steps: Int,
    shift: Float64,
    guidance: Float32,
    cfg: LingBotAttnConfig,
    ctx: DeviceContext,
) raises -> LingBotDenseT2IOut:
    """Full CFG T2I denoise with the RESIDENT dense backbone (2 forwards/step)."""
    var sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000)
    sch.set_timesteps(num_steps, shift)

    var no_taps = List[Int]()
    var lat_shape = init_latent.shape().copy()
    var n = init_latent.numel()

    var latent = init_latent.clone(ctx)
    var step_latents = List[List[Float32]]()

    for i in range(num_steps):
        var _t0 = perf_counter_ns()
        step_latents.append(latent.to_host(ctx))

        var t_int = sch.timesteps[i]
        var sig_bf = torch_bf16_rne_value(Float32(t_int) / Float32(1000.0))
        var tb_val = Float32(sig_bf) * Float32(1000.0)
        var tb_host = List[Float32]()
        tb_host.append(tb_val)
        var tb = Tensor.from_host(tb_host, [1], STDtype.F32, ctx)

        var lat_bf = _round_latent_bf16_dense(latent, ctx)

        var cond = lingbot_dense_backbone[S_COND](
            lat_bf, tb, prompt_embeds, m,
            grid_t, grid_h, grid_w, L_cond, patch_f, patch_h, patch_w,
            out_channels, theta, no_taps, cfg, ctx,
        )
        var uncond = lingbot_dense_backbone[S_UNCOND](
            lat_bf, tb, neg_embeds, m,
            grid_t, grid_h, grid_w, L_uncond, patch_f, patch_h, patch_w,
            out_channels, theta, no_taps, cfg, ctx,
        )

        var cond_h = cond.velocity[].to_host(ctx)
        var uncond_h = uncond.velocity[].to_host(ctx)
        var noise = List[Float32]()
        noise.resize(n, Float32(0.0))
        for j in range(n):
            noise[j] = uncond_h[j] + guidance * (cond_h[j] - uncond_h[j])

        var sample = latent.to_host(ctx)
        var nxt = sch.step(noise, sample)
        latent = Tensor.from_host(nxt, lat_shape.copy(), STDtype.F32, ctx)
        var _t1 = perf_counter_ns()
        print("[DPIPE] step", i, "/", num_steps, "  wall =",
              Float64(_t1 - _t0) / 1.0e9, "s")

    var final_host = latent.to_host(ctx)
    return LingBotDenseT2IOut(step_latents^, final_host^, lat_shape^)
