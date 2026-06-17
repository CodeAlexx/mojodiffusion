# models/dit/sensenova_u1.mojo — SenseNova-U1-8B-MoT T2I DiT (GPU, inference).
#
# Pure-Mojo, inference-only port of the SenseNova-U1 T2I path. References, read
# LINE BY LINE (do NOT infer):
#   /home/alex/EriDiffusion/inference-flame/src/models/sensenova_u1.rs   (model)
#   /home/alex/EriDiffusion/inference-flame/src/bin/sensenova_u1_gen.rs  (t2i flow)
#
# ── MoT = "Mixture of TRANSFORMERS", not a top-k MoE router ──────────────────
# SenseNova-U1 carries TWO PARALLEL sets of dense weights per layer:
#   * BASE weights (no suffix): the understanding / text path.
#   * `_mot_gen` weights: the image-generation path.
# Selection is per-token by MODALITY, NOT by a learned router. For T2I the two
# modes are NEVER mixed: the text prefix runs the BASE path (forward_und) which
# populates a per-layer K/V cache; each ODE step runs the `_mot_gen` path
# (forward_gen) over image tokens, concatenating the cached prefix K/V before
# attention (cache is NEVER updated). So `ops/moe` (top_k_router / grouped FFN /
# scatter) does NOT apply — there is no routing arithmetic at all. (Verified vs
# sensenova_u1.rs:74-88 "PER-LAYER ROUTING — TWO MODES, NEVER MIXED FOR T2I".)
#
# ── No separate VAE / no separate text encoder ───────────────────────────────
# SenseNova-U1 is a PIXEL-SPACE flow-matching model: fm_head predicts patch
# pixels directly and patchify/unpatchify operate in RGB pixel space. There is
# NO VAE. The Qwen3 backbone IS the text path (forward_und) interleaved with the
# gen path in the SAME 42-layer transformer — there is no standalone encoder
# model. Both live in this struct. (Verified vs sensenova_u1_gen.rs:493-561.)
#
# ── 3D RoPE (sensenova_u1.rs:43-71) ──────────────────────────────────────────
# head_dim=128 split |t=64|h=32|w=32|. q split into (t=64, hw=64); q_norm on the
# 64-d t half, q_norm_hw on the 64-d hw half; then hw split into (h=32, w=32).
# RoPE half-split, applied separately to each axis:
#   t: theta=5e6 over idx_t   h: theta=1e4 over idx_h   w: theta=1e4 over idx_w
# For text tokens idx_t=position, idx_h=idx_w=0 (identity RoPE — hw rotation
# SKIPPED in the und path, matching the Rust shortcut at sensenova_u1.rs:686-691).
# For image tokens idx_t=text_len (constant), idx_h=row, idx_w=col. V is NOT
# RoPE'd. (Verified vs sensenova_u1.rs:705-717 / 1110-1117.)
#
# ── Per-layer weight keys (verified vs on-disk model.safetensors.index.json) ──
# language_model.model.layers.{i}.{base|_mot_gen}: input_layernorm,
#   post_attention_layernorm, self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm,
#   self_attn.{q,k}_norm_hw, mlp.{gate,up,down}_proj. 26 tensors/layer (13+13).
# q_norm/k_norm are [64] (head_dim/2), NOT [128]. k/v_proj are [1024, 4096]
# (num_kv_heads=8 * 128). All BF16. 1116 total = 26*42 + 24 shared.
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in ops.

from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog, sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit, rope_interleaved
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu, gelu, swiglu
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.tensor_algebra import (
    reshape, permute, slice, concat, add, sub, mul, add_scalar, mul_scalar,
)
from serenitymojo.offload.block_loader import BlockLoader, Block
from serenitymojo.offload.plan import OffloadConfig, build_sensenova_u1_block_plan
from serenitymojo.offload.planned_loader import PlannedBlockLoader


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256
comptime _NEG_BIG = Float32(-3.0e38)


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct SenseNovaU1Config(Copyable, Movable, ImplicitlyCopyable):
    """SenseNova-U1-8B-MoT hyperparameters (config.json + reference defaults)."""

    var vocab_size: Int            # 151936
    var hidden_size: Int           # 4096
    var num_layers: Int            # 42
    var intermediate_size: Int     # 12288
    var num_heads: Int             # 32
    var num_kv_heads: Int          # 8
    var head_dim: Int              # 128
    var rms_norm_eps: Float32      # 1e-6
    var rope_theta: Float64        # 5e6   (t-axis)
    var rope_theta_hw: Float64     # 1e4   (h/w axes)
    var rope_theta_vision: Float64 # 1e4   (vision 2D rope)
    var patch_size: Int            # 16
    var vision_hidden_size: Int    # 1024
    var fm_head_out_dim: Int       # 3072  ((patch*merge)^2 * 3 = 32*32*3)
    var noise_scale_base_seq: Int  # 64
    var noise_scale_max: Float32   # 8.0
    var noise_scale: Float32       # 1.0
    var t_eps: Float32             # 0.05
    var merge_size: Int            # 2  (round(1/downsample_ratio=0.5))

    @staticmethod
    def sensenova_u1_8b() -> SenseNovaU1Config:
        return SenseNovaU1Config(
            151936, 4096, 42, 12288, 32, 8, 128,
            Float32(1e-6), Float64(5_000_000.0), Float64(10_000.0),
            Float64(10_000.0), 16, 1024, 3072, 64,
            Float32(8.0), Float32(1.0), Float32(0.05), 2,
        )

    def rope_dim_t(self) -> Int:
        return self.head_dim // 2  # 64

    def rope_dim_h(self) -> Int:
        return self.head_dim // 4  # 32

    def rope_dim_w(self) -> Int:
        return self.head_dim // 4  # 32


# ─────────────────────────────────────────────────────────────────────────────
# Local glue kernels — NOT foundation ops (the model owns these, like the
# Qwen3Encoder owns its _embed/_add/_repeat_kv glue). We do NOT touch ops/.
# ─────────────────────────────────────────────────────────────────────────────

# Embedding gather: out[t, j] = table[ids[t], j]. One thread per output element.
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


# Residual add: o = a + b, elementwise, F32 math (BF16 storage).
def _add_kernel_bf16(
    a: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.bfloat16]](a[i]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.bfloat16]](b[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((av + bv).cast[DType.bfloat16]())


# GQA repeat in BHSD-flat: src head h reads kv-head (h // n_rep). One thread per
# output element. Layout here is [H, S, Dh] (per-batch; B=1). Pure copy.
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
    var total = h * seq * dh
    if idx < total:
        var dh_i = idx % dh
        var rest = idx // dh
        var s = rest % seq
        var head = rest // seq
        var kvh = head // n_rep
        var src_idx = (kvh * seq + s) * dh + dh_i
        dst[idx] = rebind[dst.element_type](src[src_idx])


# ── non-square SDPA kernels (Sq != Skv) ──────────────────────────────────────
# Faithful generalization of ops/attention._sdpa_math to the gen path where Q
# has Sq image tokens but K/V span Skv = prefix_len + Sq. Inputs are [H, S*, Dh]
# BF16 (already GQA-expanded to H, per-head contiguous; B=1). All interior F32.
# These mirror the verified ops/attention kernels verbatim, only generalized to
# distinct query/key sequence lengths.

# gather [H, S, Dh] BF16 -> F32 [H*S, Dh] (already head-contiguous; just upcast).
def _gather_hs_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    dh: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * dh
    if idx < total:
        var r = idx // dh
        var c = idx % dh
        dst[r, c] = rebind[dst.element_type](
            rebind[Scalar[DType.bfloat16]](src[r, c]).cast[DType.float32]()
        )


# scale + additive mask over scores [H*Sq, Skv]. mask is [H, Sq, Skv] BF16.
def _scale_mask_ns_bf16(
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    mask: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    scale: Float32,
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var v = rebind[Scalar[DType.float32]](scores[r, c]) * scale
        v += rebind[Scalar[DType.bfloat16]](mask[r, c]).cast[DType.float32]()
        scores[r, c] = rebind[scores.element_type](v)


# scale only (no mask) over scores [H*Sq, Skv].
def _scale_only(
    scores: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    scale: Float32,
    rows: Int,
    cols: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * cols
    if idx < total:
        var r = idx // cols
        var c = idx % cols
        var v = rebind[Scalar[DType.float32]](scores[r, c]) * scale
        scores[r, c] = rebind[scores.element_type](v)


# softmax over last dim (Skv) in place, one block per row. Mirrors
# ops/attention._softmax_rows_f32 verbatim.
def _softmax_rows_ns(
    x: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    cols: Int,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var lmax: Float32 = _NEG_BIG
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        if v > lmax:
            lmax = v
        c += _TPB
    shared[tid] = lmax
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            var a = shared[tid]
            var b = shared[tid + active]
            shared[tid] = a if a > b else b
        barrier()
        active //= 2
    var rmax = shared[0]
    barrier()
    var lsum: Float32 = 0.0
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        lsum += fexp(v - rmax)
        c += _TPB
    shared[tid] = lsum
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var rsum = shared[0]
    var inv = 1.0 / rsum
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.float32]](x[row, c])
        x[row, c] = rebind[x.element_type](fexp(v - rmax) * inv)
        c += _TPB


# scatter F32 [H*Sq, Dh] -> BF16 [H*Sq, Dh] (head-contiguous; just downcast).
def _scatter_hs_bf16(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    rows: Int,
    dh: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * dh
    if idx < total:
        var r = idx // dh
        var c = idx % dh
        dst[r, c] = rebind[dst.element_type](
            rebind[Scalar[DType.float32]](src[r, c]).cast[DType.bfloat16]()
        )


# ── host-side glue dispatchers ───────────────────────────────────────────────
def _embed(
    table: Tensor, ids: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """Gather embedding rows table[ids] -> [1, seq, hidden]. BF16 table."""
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

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        seq * hidden * table.dtype().byte_size()
    )
    var tab_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](table.numel()))
    var id_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq * hidden))
    var total = seq * hidden
    var grid = (total + _BLOCK - 1) // _BLOCK
    var IDS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        id_dev.unsafe_ptr().bitcast[Int32](), id_rl
    )
    var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        table.buf.unsafe_ptr().bitcast[BFloat16](), tab_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )
    ctx.enqueue_function[_embed_kernel_bf16, _embed_kernel_bf16](
        T, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var sh = List[Int]()
    sh.append(1)
    sh.append(seq)
    sh.append(hidden)
    return Tensor(out_buf^, sh^, table.dtype())


def _add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    """o = a + b elementwise (same shape/dtype, BF16). F32 math."""
    if a.numel() != b.numel():
        raise Error("sensenova_u1._add: numel mismatch")
    var n = a.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](a.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    ctx.enqueue_function[_add_kernel_bf16, _add_kernel_bf16](
        A, B, O, n, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, a.shape(), a.dtype())


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Deep copy of a Tensor's device buffer."""
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _repeat_kv_bhsd(
    x: Tensor, h: Int, h_kv: Int, ctx: DeviceContext
) raises -> Tensor:
    """GQA repeat in [H_kv, S, Dh] -> [H, S, Dh] (B=1, head-contiguous BHSD)."""
    var xs = x.shape()
    if len(xs) != 3:
        raise Error("sensenova_u1._repeat_kv_bhsd: x must be [H_kv,S,Dh]")
    var seq = xs[1]
    var dh = xs[2]
    if xs[0] != h_kv:
        raise Error("sensenova_u1._repeat_kv_bhsd: head dim != h_kv")
    var n_rep = h // h_kv
    if n_rep == 1:
        return _clone(x, ctx)
    var out_n = h * seq * dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](h_kv * seq * dh))
    var dst_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    var D = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), dst_rl
    )
    ctx.enqueue_function[_repeat_kv_kernel_bf16, _repeat_kv_kernel_bf16](
        S, D, seq, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(h)
    osh.append(seq)
    osh.append(dh)
    return Tensor(out_buf^, osh^, x.dtype())


# ── BSHD [1,S,H,Dh] -> head-contiguous BHSD-flat [H,S,Dh] (a permute) ────────
def _to_bhsd(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """[1, S, H, Dh] -> [H, S, Dh] via permute(0,2,1,3) collapsed (B=1)."""
    var perm = List[Int]()
    perm.append(0)
    perm.append(2)
    perm.append(1)
    perm.append(3)
    var p = permute(x, perm, ctx)  # [1, H, S, Dh]
    var sh = List[Int]()
    sh.append(p.shape()[1])
    sh.append(p.shape()[2])
    sh.append(p.shape()[3])
    return reshape(p, sh^, ctx)


# ── non-square attention helper (model-local; mirrors ops/attention math mode) ─
def _attention_nonsquare(
    q_hsd: Tensor,   # [H, Sq, Dh] BF16 (already GQA-expanded)
    k_hsd: Tensor,   # [H, Skv, Dh] BF16
    v_hsd: Tensor,   # [H, Skv, Dh] BF16
    mask: Tensor,    # [H, Sq, Skv] BF16 additive, or empty (use_mask=False)
    use_mask: Bool,
    scale: Float32,
    h: Int,
    sq: Int,
    skv: Int,
    dh: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Per-head SDPA with Sq != Skv. Returns [H, Sq, Dh] BF16. F32 interior.
    Faithful generalization of ops/attention._sdpa_math (Dh=128 -> math mode)."""
    # 1) upcast q/k/v to F32 (already head-contiguous).
    var q_f32 = ctx.enqueue_create_buffer[DType.float32](h * sq * dh)
    var k_f32 = ctx.enqueue_create_buffer[DType.float32](h * skv * dh)
    var v_f32 = ctx.enqueue_create_buffer[DType.float32](h * skv * dh)
    var q_rows = h * sq
    var kv_rows = h * skv
    var q_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](q_rows, dh))
    var kv_src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](kv_rows, dh))
    var qf_dst = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        q_f32.unsafe_ptr(), q_src_rl
    )
    var kf_dst = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        k_f32.unsafe_ptr(), kv_src_rl
    )
    var vf_dst = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        v_f32.unsafe_ptr(), kv_src_rl
    )
    var qs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        q_hsd.buf.unsafe_ptr().bitcast[BFloat16](), q_src_rl
    )
    var ks = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        k_hsd.buf.unsafe_ptr().bitcast[BFloat16](), kv_src_rl
    )
    var vs = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        v_hsd.buf.unsafe_ptr().bitcast[BFloat16](), kv_src_rl
    )
    var qgrid = (q_rows * dh + _BLOCK - 1) // _BLOCK
    var kvgrid = (kv_rows * dh + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_gather_hs_bf16, _gather_hs_bf16](
        qs, qf_dst, q_rows, dh, grid_dim=qgrid, block_dim=_BLOCK
    )
    ctx.enqueue_function[_gather_hs_bf16, _gather_hs_bf16](
        ks, kf_dst, kv_rows, dh, grid_dim=kvgrid, block_dim=_BLOCK
    )
    ctx.enqueue_function[_gather_hs_bf16, _gather_hs_bf16](
        vs, vf_dst, kv_rows, dh, grid_dim=kvgrid, block_dim=_BLOCK
    )

    # 2) QKᵀ per head -> scores F32 [H, Sq, Skv]
    var scores = ctx.enqueue_create_buffer[DType.float32](h * sq * skv)
    var q_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sq, dh))
    var k_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](skv, dh))
    var sc_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sq, skv))
    var qptr = q_f32.unsafe_ptr()
    var kptr = k_f32.unsafe_ptr()
    var scptr = scores.unsafe_ptr()
    for head in range(h):
        var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            qptr + head * sq * dh, q_head_rl
        )
        var Bt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            kptr + head * skv * dh, k_head_rl
        )
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            scptr + head * sq * skv, sc_head_rl
        )
        matmul(ctx, C, A, Bt, transpose_b=True, c_row_major=True)

    # 3) scale (+ optional mask) over [H*Sq, Skv]
    var sm_rows = h * sq
    var sc_full_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sm_rows, skv))
    var sc_full = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        scptr, sc_full_rl
    )
    var nsm = sm_rows * skv
    var smgrid = (nsm + _BLOCK - 1) // _BLOCK
    if use_mask:
        var Mf = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            mask.buf.unsafe_ptr().bitcast[BFloat16](), sc_full_rl
        )
        ctx.enqueue_function[_scale_mask_ns_bf16, _scale_mask_ns_bf16](
            sc_full, Mf, scale, sm_rows, skv, grid_dim=smgrid, block_dim=_BLOCK
        )
    else:
        ctx.enqueue_function[_scale_only, _scale_only](
            sc_full, scale, sm_rows, skv, grid_dim=smgrid, block_dim=_BLOCK
        )

    # 4) softmax over last dim (Skv) in place, one block per row.
    ctx.enqueue_function[_softmax_rows_ns, _softmax_rows_ns](
        sc_full, skv, grid_dim=sm_rows, block_dim=_TPB
    )

    # 5) P @ V per head -> F32 [H, Sq, Dh]
    var out_f32 = ctx.enqueue_create_buffer[DType.float32](h * sq * dh)
    var v_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](skv, dh))
    var o_head_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](sq, dh))
    var optr = out_f32.unsafe_ptr()
    var vptr = v_f32.unsafe_ptr()
    for head in range(h):
        var P = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            scptr + head * sq * skv, sc_head_rl
        )
        var Vh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            vptr + head * skv * dh, v_head_rl
        )
        var Oh = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            optr + head * sq * dh, o_head_rl
        )
        matmul(ctx, Oh, P, Vh, transpose_b=False, c_row_major=True)

    # 6) downcast F32 [H*Sq, Dh] -> BF16 [H, Sq, Dh]
    var out_rows = h * sq
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_rows * dh * 2)
    var out_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](out_rows, dh))
    var src_o = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_f32.unsafe_ptr(), out_rl
    )
    var dst_o = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )
    var ogrid = (out_rows * dh + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_scatter_hs_bf16, _scatter_hs_bf16](
        src_o, dst_o, out_rows, dh, grid_dim=ogrid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(h)
    osh.append(sq)
    osh.append(dh)
    return Tensor(out_buf^, osh^, STDtype.BF16)


# ── host-side RoPE / sinusoidal / mask table builders ────────────────────────
# Half-split cos/sin for explicit integer positions, laid out per-(position) in
# row order matching how rope_halfsplit flattens leading dims to rows. The data
# tensor is BSHD [1, S, H, Dh_axis] (never permuted to BHSD before RoPE), so
# rope_halfsplit flattens leading dims [1,S,H] -> rows in SEQ-MAJOR order:
# row r = s*H + h. We must build the table in the same order — outer loop over
# seq, inner loop over heads — so table row r and data row r share a position.
# (Mirrors zimage_dit.mojo:546-561 "row order: token t, head head".) The angle
# depends ONLY on the position, so the per-position angle block is tiled across
# heads within each token.
def _build_rope_for_positions_hs(
    positions: List[Int], heads: Int, axis_dim: Int, theta: Float64
) raises -> List[List[Float32]]:
    """Returns [cos, sin], each flat length heads*len(positions)*(axis_dim/2)."""
    var half = axis_dim // 2
    var seq = len(positions)
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(theta))
    for s in range(seq):
        var pos = Float32(positions[s])
        for _hh in range(heads):
            for i in range(half):
                var exponent = -log_theta * Float32(2 * i) / Float32(axis_dim)
                var inv_freq = fexp(exponent)
                var angle = pos * inv_freq
                cos_vals.append(fcos(angle))
                sin_vals.append(fsin(angle))
    var out = List[List[Float32]]()
    out.append(cos_vals^)
    out.append(sin_vals^)
    return out^


# Interleaved cos/sin tables for the vision 2D RoPE. rope_interleaved consumes
# cos/sin [rows, D/2]; rows here = number of patch tokens (B*N). Angle depends on
# the patch coord only. layout [rows, half].
def _build_rope_interleaved(
    positions: List[Int], axis_dim: Int, theta: Float64
) raises -> List[List[Float32]]:
    var half = axis_dim // 2
    var n = len(positions)
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(theta))
    for s in range(n):
        var pos = Float32(positions[s])
        for i in range(half):
            var exponent = -log_theta * Float32(2 * i) / Float32(axis_dim)
            var inv_freq = fexp(exponent)
            var angle = pos * inv_freq
            cos_vals.append(fcos(angle))
            sin_vals.append(fsin(angle))
    var out = List[List[Float32]]()
    out.append(cos_vals^)
    out.append(sin_vals^)
    return out^


# Causal mask [H, S, S] additive BF16: 0 where j<=i, -1e4 otherwise. real_len
# masks right padding (cols >= real_len blocked).
def _build_causal_mask(seq: Int, heads: Int, real_len: Int) raises -> List[Float32]:
    var neg = Float32(-1.0e4)
    var data = List[Float32]()
    for _hh in range(heads):
        for i in range(seq):
            for j in range(seq):
                if j <= i and j < real_len:
                    data.append(Float32(0.0))
                else:
                    data.append(neg)
    return data^


# ── KV cache (populated by forward_und, read by forward_gen, never updated) ───
# One (K, V) per layer at [H_kv, prefix_len, Dh] BF16 (BEFORE GQA repeat).
struct KvCache(Movable):
    var k_layers: List[ArcPointer[Tensor]]
    var v_layers: List[ArcPointer[Tensor]]
    var prefix_len: Int
    var next_t_index: Int

    def __init__(
        out self,
        var k_layers: List[ArcPointer[Tensor]],
        var v_layers: List[ArcPointer[Tensor]],
        prefix_len: Int,
        next_t_index: Int,
    ):
        self.k_layers = k_layers^
        self.v_layers = v_layers^
        self.prefix_len = prefix_len
        self.next_t_index = next_t_index


def _is_sensenova_t2i_shared(name: String) -> Bool:
    # T2I-only resident set. The checkpoint also contains language_model.lm_head
    # (think-mode autoregressive decode) and vision_model.embeddings.* (VQA /
    # understanding-side image input); neither is used by this Mojo T2I smoke,
    # and keeping them resident costs ~1.22 GiB of GPU memory.
    if name == "language_model.model.embed_tokens.weight":
        return True
    if name == "language_model.model.norm.weight":
        return True
    if name == "language_model.model.norm_mot_gen.weight":
        return True
    if name.startswith("fm_modules."):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SenseNovaU1 model
# ─────────────────────────────────────────────────────────────────────────────
struct SenseNovaU1[L_TOKENS: Int, TEXT_LEN: Int](Movable):
    """SenseNova-U1-8B-MoT T2I model. Per-layer transformer weights stream from
    the sharded checkpoint via PlannedBlockLoader; shared weights (embed_tokens, final
    norms, fm_modules, vision_model_mot_gen embedder) are resident.

    Comptime params:
      L_TOKENS : number of image gen tokens (token_h * token_w). The gen-path
                 query sequence length (Sq). Compile-time so the per-head matmul
                 / softmax shapes are fixed.
      TEXT_LEN : prefix text token count (the und-path causal seq length, and
                 the t-axis RoPE index for image tokens). Compile-time.
    These are the only runtime-variable sequence lengths SDPA needs pinned; the
    caller picks the SenseNovaU1[L, T] specialization matching its resolution +
    prompt (see the smoke pipeline). FLAGGED: a production run needs a comptime
    dispatch enumerating (L_TOKENS, TEXT_LEN) cases (like qwen3_encoder's
    _sdpa_dispatch); here we parameterize the struct instead."""

    var config: SenseNovaU1Config
    var shared: Dict[String, ArcPointer[Tensor]]
    var loader: PlannedBlockLoader

    def __init__(
        out self,
        config: SenseNovaU1Config,
        var shared: Dict[String, ArcPointer[Tensor]],
        var loader: PlannedBlockLoader,
    ):
        self.config = config
        self.shared = shared^
        self.loader = loader^

    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> SenseNovaU1[Self.L_TOKENS, Self.TEXT_LEN]:
        """Open the sharded checkpoint. Per-layer weights stream via PlannedBlockLoader;
        T2I-required shared tensors are loaded resident into a Dict."""
        var raw_loader = BlockLoader.open(dir)
        var shared = Dict[String, ArcPointer[Tensor]]()
        for ref nm in raw_loader.sharded.names():
            if _is_sensenova_t2i_shared(nm):
                var tv = raw_loader.sharded.tensor_view(nm)
                var t = Tensor.from_view(tv, ctx)
                shared[nm] = ArcPointer(t^)
        var plan = build_sensenova_u1_block_plan()
        var loader = PlannedBlockLoader(
            raw_loader^, plan^, OffloadConfig.single_pass()
        )
        return SenseNovaU1[Self.L_TOKENS, Self.TEXT_LEN](
            SenseNovaU1Config.sensenova_u1_8b(), shared^, loader^
        )

    def _shared(self, name: String) raises -> ref [self.shared] Tensor:
        if name not in self.shared:
            raise Error(String("sensenova_u1: missing shared weight: ") + name)
        return self.shared[name][]

    # ── one base (understanding / text-prefix) layer ─────────────────────────
    # Returns (new_hidden, k_cache, v_cache). K/V are [H_kv, S, Dh] BF16 BEFORE
    # GQA repeat (what forward_gen will concat with). Mirrors und_layer
    # (sensenova_u1.rs:628-761): t-axis RoPE only, hw rotation skipped for text.
    def _und_layer(
        self,
        block: Block,
        i: Int,
        hidden: Tensor,
        cos_t: Tensor,
        sin_t: Tensor,
        cos_t_kv: Tensor,
        sin_t_kv: Tensor,
        mask: Tensor,
        mut k_layers: List[ArcPointer[Tensor]],
        mut v_layers: List[ArcPointer[Tensor]],
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim
        var d_t = cfg.rope_dim_t()  # 64
        var eps = cfg.rms_norm_eps
        var scale = Float32(1.0) / sqrt(Float32(dh))
        var seq = hidden.shape()[1]
        var p = String("language_model.model.layers.") + String(i)


        var normed = rms_norm(hidden, _bget(block, p, ".input_layernorm.weight"), eps, ctx)
        var q = linear(normed, _bget(block, p, ".self_attn.q_proj.weight"), None, ctx)
        var k = linear(normed, _bget(block, p, ".self_attn.k_proj.weight"), None, ctx)
        var v = linear(normed, _bget(block, p, ".self_attn.v_proj.weight"), None, ctx)

        # reshape to BSHD [1, S, H, Dh] / [1, S, H_kv, Dh]
        q = _reshape4(q, 1, seq, h, dh, ctx)
        k = _reshape4(k, 1, seq, h_kv, dh, ctx)
        v = _reshape4(v, 1, seq, h_kv, dh, ctx)

        # split last dim into (t=64, hw=64), norm each half, RoPE the t half only
        # (text hw positions are 0 -> identity rotation, skipped).
        var q_t = slice(q, 3, 0, d_t, ctx)        # [1,S,H,64]
        var q_hw = slice(q, 3, d_t, dh - d_t, ctx) # [1,S,H,64]
        q_t = rms_norm(q_t, _bget(block, p, ".self_attn.q_norm.weight"), eps, ctx)
        q_hw = rms_norm(q_hw, _bget(block, p, ".self_attn.q_norm_hw.weight"), eps, ctx)
        q_t = rope_halfsplit(q_t, cos_t, sin_t, ctx)
        q = concat(3, ctx, q_t, q_hw)

        var k_t = slice(k, 3, 0, d_t, ctx)
        var k_hw = slice(k, 3, d_t, dh - d_t, ctx)
        k_t = rms_norm(k_t, _bget(block, p, ".self_attn.k_norm.weight"), eps, ctx)
        k_hw = rms_norm(k_hw, _bget(block, p, ".self_attn.k_norm_hw.weight"), eps, ctx)
        k_t = rope_halfsplit(k_t, cos_t_kv, sin_t_kv, ctx)
        k = concat(3, ctx, k_t, k_hw)

        # cache K/V at [H_kv, S, Dh] (BHSD-flat, BEFORE GQA repeat)
        var k_bhsd = _to_bhsd(k, ctx)  # [H_kv, S, Dh]
        var v_bhsd = _to_bhsd(v, ctx)
        var k_cache = _clone(k_bhsd, ctx)
        var v_cache = _clone(v_bhsd, ctx)

        # GQA repeat, attention (square: Sq=Skv=seq), causal mask.
        var k_g = _repeat_kv_bhsd(k_bhsd, h, h_kv, ctx)  # [H,S,Dh]
        var q_bhsd = _to_bhsd(q, ctx)                    # [H,S,Dh]
        var v_g = _repeat_kv_bhsd(v_bhsd, h, h_kv, ctx)
        var attn = _attention_nonsquare(
            q_bhsd, k_g, v_g, mask, True, scale, h, seq, seq, dh, ctx
        )  # [H,S,Dh]

        # [H,S,Dh] -> [1,S,H*Dh]
        var attn_perm = _bhsd_to_bsh(attn, h, seq, dh, ctx)  # [1,S,H,Dh]
        var attn_flat = _reshape3(attn_perm, 1, seq, h * dh, ctx)
        var attn_out = linear(attn_flat, _bget(block, p, ".self_attn.o_proj.weight"), None, ctx)
        var hidden2 = _add(hidden, attn_out, ctx)

        # SwiGLU MLP (base).
        var normed2 = rms_norm(
            hidden2, _bget(block, p, ".post_attention_layernorm.weight"), eps, ctx
        )
        var gate = linear(normed2, _bget(block, p, ".mlp.gate_proj.weight"), None, ctx)
        var up = linear(normed2, _bget(block, p, ".mlp.up_proj.weight"), None, ctx)
        var act = swiglu(gate, up, ctx)
        var mlp_out = linear(act, _bget(block, p, ".mlp.down_proj.weight"), None, ctx)
        var out = _add(hidden2, mlp_out, ctx)

        # Append the per-layer K/V cache entries (move into the lists).
        k_layers.append(ArcPointer(k_cache^))
        v_layers.append(ArcPointer(v_cache^))
        return out^

    # forward_und: text prefix path -> KV cache + final hidden state.
    # Mirrors sensenova_u1.rs:547-619. Uses BASE weights, causal mask, t-axis
    # RoPE on positions [0..seq). The final norm is the BASE norm.weight.
    # NOTE: returns ONLY the KvCache (KvCache is Movable-not-Copyable, and a
    # Tuple[KvCache, Tensor] can't be index-extracted by move in 1.0.0b1). The
    # final BASE-norm hidden state is used only by think-mode autoregression
    # (OUT OF SCOPE for T2I), so we compute it for fidelity but do not return it.
    def forward_und(
        mut self, token_ids: List[Int], ctx: DeviceContext
    ) raises -> KvCache:
        var cfg = self.config
        var seq = len(token_ids)
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var d_t = cfg.rope_dim_t()

        # embed tokens -> [1, seq, hidden]
        ref table = self._shared("language_model.model.embed_tokens.weight")
        var hidden = _embed(table, token_ids, ctx)

        # t-axis RoPE tables: positions 0..seq, half-split over 64 dims.
        var pos_t = List[Int]()
        for t in range(seq):
            pos_t.append(t)
        var q_tab = _build_rope_for_positions_hs(pos_t, h, d_t, cfg.rope_theta)
        var k_tab = _build_rope_for_positions_hs(pos_t.copy(), h_kv, d_t, cfg.rope_theta)
        var half = d_t // 2
        var cq_sh = List[Int]()
        cq_sh.append(seq * h * half)
        var ck_sh = List[Int]()
        ck_sh.append(seq * h_kv * half)
        var cos_t = Tensor.from_host(q_tab[0], cq_sh.copy(), STDtype.BF16, ctx)
        var sin_t = Tensor.from_host(q_tab[1], cq_sh.copy(), STDtype.BF16, ctx)
        var cos_t_kv = Tensor.from_host(k_tab[0], ck_sh.copy(), STDtype.BF16, ctx)
        var sin_t_kv = Tensor.from_host(k_tab[1], ck_sh.copy(), STDtype.BF16, ctx)

        # causal mask [H, seq, seq] (real_len = seq, no padding in our prefix).
        var mask_data = _build_causal_mask(seq, h, seq)
        var mask_sh = List[Int]()
        mask_sh.append(h)
        mask_sh.append(seq)
        mask_sh.append(seq)
        var mask = Tensor.from_host(mask_data, mask_sh^, STDtype.BF16, ctx)

        var k_layers = List[ArcPointer[Tensor]]()
        var v_layers = List[ArcPointer[Tensor]]()
        var total = cfg.num_layers
        self.loader.config = OffloadConfig.single_pass()
        self.loader.prefetch_with_ctx(0, ctx)
        for i in range(total):
            var handle = self.loader.await_block(i, ctx)
            self.loader.prefetch_next_with_ctx(i, ctx)
            hidden = self._und_layer(
                handle.block, i, hidden, cos_t, sin_t, cos_t_kv, sin_t_kv, mask,
                k_layers, v_layers, ctx
            )
            self.loader.mark_active_block_done(ctx)

        # final BASE norm (computed for fidelity; consumed only by think-mode).
        ref final_w = self._shared("language_model.model.norm.weight")
        hidden = rms_norm(hidden, final_w, cfg.rms_norm_eps, ctx)
        _ = hidden^

        var cache = KvCache(k_layers^, v_layers^, seq, seq)
        return cache^

    # ── one gen (image) layer ────────────────────────────────────────────────
    # Mirrors gen_layer (sensenova_u1.rs:968-1090): _mot_gen weights, FULL 3D
    # RoPE, K/V concat with the cached prefix along seq (no update), full
    # (non-causal) attention. image_embeds [1, L, hidden].
    def _gen_layer(
        self,
        block: Block,
        i: Int,
        hidden: Tensor,
        cos_t: Tensor, sin_t: Tensor,
        cos_h: Tensor, sin_h: Tensor,
        cos_w: Tensor, sin_w: Tensor,
        cos_t_kv: Tensor, sin_t_kv: Tensor,
        cos_h_kv: Tensor, sin_h_kv: Tensor,
        cos_w_kv: Tensor, sin_w_kv: Tensor,
        past_k: Tensor, past_v: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim
        var eps = cfg.rms_norm_eps
        var scale = Float32(1.0) / sqrt(Float32(dh))
        var l = hidden.shape()[1]
        var prefix_len = past_k.shape()[1]
        var skv = prefix_len + l
        var p = String("language_model.model.layers.") + String(i)

        var normed = rms_norm(hidden, _bget(block, p, ".input_layernorm_mot_gen.weight"), eps, ctx)
        var q = linear(normed, _bget(block, p, ".self_attn.q_proj_mot_gen.weight"), None, ctx)
        var k = linear(normed, _bget(block, p, ".self_attn.k_proj_mot_gen.weight"), None, ctx)
        var v = linear(normed, _bget(block, p, ".self_attn.v_proj_mot_gen.weight"), None, ctx)

        q = _reshape4(q, 1, l, h, dh, ctx)
        k = _reshape4(k, 1, l, h_kv, dh, ctx)
        v = _reshape4(v, 1, l, h_kv, dh, ctx)

        # FULL 3D RoPE on q and k.
        q = self._apply_3d_rope(
            block, p, q, True, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, ctx
        )
        k = self._apply_3d_rope(
            block, p, k, False, cos_t_kv, sin_t_kv, cos_h_kv, sin_h_kv,
            cos_w_kv, sin_w_kv, ctx
        )

        # to BHSD-flat, concat the cached prefix K/V along seq (axis 1), repeat.
        var k_bhsd = _to_bhsd(k, ctx)  # [H_kv, L, Dh]
        var v_bhsd = _to_bhsd(v, ctx)
        var k_full = concat(1, ctx, past_k, k_bhsd)  # [H_kv, prefix+L, Dh]
        var v_full = concat(1, ctx, past_v, v_bhsd)
        var k_g = _repeat_kv_bhsd(k_full, h, h_kv, ctx)  # [H, skv, Dh]
        var v_g = _repeat_kv_bhsd(v_full, h, h_kv, ctx)
        var q_bhsd = _to_bhsd(q, ctx)  # [H, L, Dh]

        # full attention (no mask): gen tokens see prefix + each other. The mask
        # arg is ignored when use_mask=False; pass a resident tensor by borrow as
        # a placeholder (no allocation, no copy).
        ref dummy_mask = self._shared("language_model.model.norm_mot_gen.weight")
        var attn = _attention_nonsquare(
            q_bhsd, k_g, v_g, dummy_mask, False, scale, h, l, skv, dh, ctx
        )  # [H, L, Dh]

        var attn_perm = _bhsd_to_bsh(attn, h, l, dh, ctx)  # [1,L,H,Dh]
        var attn_flat = _reshape3(attn_perm, 1, l, h * dh, ctx)
        var attn_out = linear(attn_flat, _bget(block, p, ".self_attn.o_proj_mot_gen.weight"), None, ctx)
        var hidden2 = _add(hidden, attn_out, ctx)

        var normed2 = rms_norm(
            hidden2, _bget(block, p, ".post_attention_layernorm_mot_gen.weight"), eps, ctx
        )
        var gate = linear(normed2, _bget(block, p, ".mlp_mot_gen.gate_proj.weight"), None, ctx)
        var up = linear(normed2, _bget(block, p, ".mlp_mot_gen.up_proj.weight"), None, ctx)
        var act = swiglu(gate, up, ctx)
        var mlp_out = linear(act, _bget(block, p, ".mlp_mot_gen.down_proj.weight"), None, ctx)
        return _add(hidden2, mlp_out, ctx)

    # apply 3-axis RoPE-with-norms to [1, S, H, 128]. Splits (t=64, hw=64),
    # norms each, splits hw -> (h=32, w=32), RoPE each, concat. is_q selects the
    # *_mot_gen q vs k norm weights. Mirrors apply_3d_rope (sensenova_u1.rs:1098).
    def _apply_3d_rope(
        self,
        block: Block,
        p: String,
        x: Tensor,
        is_q: Bool,
        cos_t: Tensor, sin_t: Tensor,
        cos_h: Tensor, sin_h: Tensor,
        cos_w: Tensor, sin_w: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var dh = cfg.head_dim
        var d_t = cfg.rope_dim_t()  # 64
        var d_h = cfg.rope_dim_h()  # 32
        var eps = cfg.rms_norm_eps
        var nm_t: String
        var nm_hw: String
        if is_q:
            nm_t = ".self_attn.q_norm_mot_gen.weight"
            nm_hw = ".self_attn.q_norm_hw_mot_gen.weight"
        else:
            nm_t = ".self_attn.k_norm_mot_gen.weight"
            nm_hw = ".self_attn.k_norm_hw_mot_gen.weight"

        var x_t = slice(x, 3, 0, d_t, ctx)           # [1,S,H,64]
        var x_hw = slice(x, 3, d_t, dh - d_t, ctx)   # [1,S,H,64]
        x_t = rms_norm(x_t, _bget(block, p, nm_t), eps, ctx)
        x_hw = rms_norm(x_hw, _bget(block, p, nm_hw), eps, ctx)
        var x_h = slice(x_hw, 3, 0, d_h, ctx)        # [1,S,H,32]
        var x_w = slice(x_hw, 3, d_h, d_h, ctx)      # [1,S,H,32]
        x_t = rope_halfsplit(x_t, cos_t, sin_t, ctx)
        x_h = rope_halfsplit(x_h, cos_h, sin_h, ctx)
        x_w = rope_halfsplit(x_w, cos_w, sin_w, ctx)
        return concat(3, ctx, x_t, x_h, x_w)

    # forward_gen: per-step image path -> hidden [1, L, hidden]. Mirrors
    # sensenova_u1.rs:890-959. Reads the KV cache, never updates it. text_len is
    # the t-axis RoPE index assigned to ALL image tokens.
    def forward_gen(
        mut self,
        image_embeds: Tensor,
        text_len: Int,
        token_h: Int,
        token_w: Int,
        cache: KvCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var d_t = cfg.rope_dim_t()
        var d_h = cfg.rope_dim_h()
        var d_w = cfg.rope_dim_w()
        var l = token_h * token_w

        # positional indices for gen tokens.
        var idx_t = List[Int]()
        var idx_h = List[Int]()
        var idx_w = List[Int]()
        for i in range(l):
            idx_t.append(text_len)
            idx_h.append(i // token_w)
            idx_w.append(i % token_w)

        # RoPE tables for q (H heads) and k (H_kv heads), three axes each.
        var qt = _build_rope_for_positions_hs(idx_t, h, d_t, cfg.rope_theta)
        var qh = _build_rope_for_positions_hs(idx_h, h, d_h, cfg.rope_theta_hw)
        var qw = _build_rope_for_positions_hs(idx_w, h, d_w, cfg.rope_theta_hw)
        var kt = _build_rope_for_positions_hs(idx_t.copy(), h_kv, d_t, cfg.rope_theta)
        var kh = _build_rope_for_positions_hs(idx_h.copy(), h_kv, d_h, cfg.rope_theta_hw)
        var kw = _build_rope_for_positions_hs(idx_w.copy(), h_kv, d_w, cfg.rope_theta_hw)

        var cos_t = self._mk_tab(qt[0].copy(), l, h, d_t, ctx)
        var sin_t = self._mk_tab(qt[1].copy(), l, h, d_t, ctx)
        var cos_h = self._mk_tab(qh[0].copy(), l, h, d_h, ctx)
        var sin_h = self._mk_tab(qh[1].copy(), l, h, d_h, ctx)
        var cos_w = self._mk_tab(qw[0].copy(), l, h, d_w, ctx)
        var sin_w = self._mk_tab(qw[1].copy(), l, h, d_w, ctx)
        var cos_t_kv = self._mk_tab(kt[0].copy(), l, h_kv, d_t, ctx)
        var sin_t_kv = self._mk_tab(kt[1].copy(), l, h_kv, d_t, ctx)
        var cos_h_kv = self._mk_tab(kh[0].copy(), l, h_kv, d_h, ctx)
        var sin_h_kv = self._mk_tab(kh[1].copy(), l, h_kv, d_h, ctx)
        var cos_w_kv = self._mk_tab(kw[0].copy(), l, h_kv, d_w, ctx)
        var sin_w_kv = self._mk_tab(kw[1].copy(), l, h_kv, d_w, ctx)

        var hidden = _clone(image_embeds, ctx)
        var total = cfg.num_layers
        self.loader.config = OffloadConfig.single_pass()
        self.loader.prefetch_with_ctx(0, ctx)
        for i in range(total):
            var handle = self.loader.await_block(i, ctx)
            self.loader.prefetch_next_with_ctx(i, ctx)
            ref pk = cache.k_layers[i][]
            ref pv = cache.v_layers[i][]
            hidden = self._gen_layer(
                handle.block, i, hidden,
                cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
                cos_t_kv, sin_t_kv, cos_h_kv, sin_h_kv, cos_w_kv, sin_w_kv,
                pk, pv, ctx,
            )
            self.loader.mark_active_block_done(ctx)

        # final GEN norm.
        ref final_w = self._shared("language_model.model.norm_mot_gen.weight")
        return rms_norm(hidden, final_w, cfg.rms_norm_eps, ctx)

    def _mk_tab(
        self, var data: List[Float32], seq: Int, heads: Int, axis_dim: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var sh = List[Int]()
        sh.append(seq * heads * (axis_dim // 2))
        return Tensor.from_host(data, sh^, STDtype.BF16, ctx)

    # ── gen-side patch + 2x2 merge embedder ──────────────────────────────────
    # Mirrors extract_feature_gen (sensenova_u1.rs:1166-1273). pixel_values
    # [B*N, 768] (already 16x16-patchified, C-major). Returns [B, L, 4096].
    #   1. patch_embedding Conv2d k=s=16 as matmul (weight [1024,3,16,16]->[1024,768]).
    #   2. GELU.
    #   3. 2D INTERLEAVED RoPE: first 512 dims rotated by x-coord, last 512 by
    #      y-coord (theta=1e4). (rope_interleaved == flame_core rope_fused_bf16.)
    #   4. dense_embedding Conv2d k=s=2 as matmul (weight [4096,1024,2,2]->[4096,4096])
    #      with spatial 2x2 pack permute [0,1,3,5,2,4].
    def extract_feature_gen(
        self,
        pixel_values: Tensor,  # [B*N, 768] BF16
        grid_h: Int,
        grid_w: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var patch = cfg.patch_size
        var vh = cfg.vision_hidden_size  # 1024
        var hidden = cfg.hidden_size     # 4096
        var merge = cfg.merge_size       # 2
        var patch_flat = 3 * patch * patch  # 768
        var n = grid_h * grid_w
        var bn = pixel_values.shape()[0]
        var b = bn // n
        var token_h = grid_h // merge
        var token_w = grid_w // merge

        ref pe_w = self._shared(
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight"
        )
        ref pe_b = self._shared(
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias"
        )
        ref de_w = self._shared(
            "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight"
        )
        ref de_b = self._shared(
            "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.bias"
        )

        # (1) patch embed: [B*N,768] @ [1024,768]ᵀ + b -> [B*N,1024]
        var pe_w_flat = _reshape2(_clone(pe_w, ctx), vh, patch_flat, ctx)
        var x = linear(pixel_values, pe_w_flat, Optional(_clone(pe_b, ctx)), ctx)

        # (2) GELU
        x = gelu(x, ctx)

        # (3) 2D interleaved RoPE: split [B*N,1024] -> two [B*N,512]; x-coord on
        # first, y-coord on second; theta=1e4.
        var half = vh // 2  # 512
        var pos_x = List[Int]()
        var pos_y = List[Int]()
        for _bb in range(b):
            for i in range(n):
                pos_x.append(i % grid_w)
                pos_y.append(i // grid_w)
        var tx = _build_rope_interleaved(pos_x, half, cfg.rope_theta_vision)
        var ty = _build_rope_interleaved(pos_y, half, cfg.rope_theta_vision)
        var rl_sh = List[Int]()
        rl_sh.append(bn)
        rl_sh.append(half // 2)
        var cos_x = Tensor.from_host(tx[0], rl_sh.copy(), STDtype.BF16, ctx)
        var sin_x = Tensor.from_host(tx[1], rl_sh.copy(), STDtype.BF16, ctx)
        var cos_y = Tensor.from_host(ty[0], rl_sh.copy(), STDtype.BF16, ctx)
        var sin_y = Tensor.from_host(ty[1], rl_sh.copy(), STDtype.BF16, ctx)

        var x_x = slice(x, 1, 0, half, ctx)      # [B*N,512]
        var x_y = slice(x, 1, half, half, ctx)   # [B*N,512]
        x_x = rope_interleaved(x_x, cos_x, sin_x, ctx)
        x_y = rope_interleaved(x_y, cos_y, sin_y, ctx)
        x = concat(1, ctx, x_x, x_y)             # [B*N,1024]

        # (4) 2x2 spatial merge: reshape [B,gh,gw,1024] -> [B,th,2,tw,2,1024],
        # permute [0,1,3,5,2,4] -> [B,th,tw,1024,2,2] -> [1, B*th*tw, 4096], then
        # dense matmul (weight [4096,1024,2,2]->[4096,4096]).
        x = _reshape4(x, b, grid_h, grid_w, vh, ctx)
        var x6 = _reshape6(x, b, token_h, merge, token_w, merge, vh, ctx)
        var perm = List[Int]()
        perm.append(0)
        perm.append(1)
        perm.append(3)
        perm.append(5)
        perm.append(2)
        perm.append(4)
        var xp = permute(x6, perm, ctx)  # [B, th, tw, vh, 2, 2]
        var merge_flat = vh * merge * merge  # 4096
        var xm = _reshape3(xp, 1, b * token_h * token_w, merge_flat, ctx)
        var de_w_flat = _reshape2(_clone(de_w, ctx), hidden, merge_flat, ctx)
        var out = linear(xm, de_w_flat, Optional(_clone(de_b, ctx)), ctx)
        return _reshape3(out, b, token_h * token_w, hidden, ctx)

    # ── fm_modules: timestep / noise-scale embed + fm_head ───────────────────
    # time_or_scale_embed: sinusoidal(256, COS-first) -> Linear(256,4096) -> SiLU
    # -> Linear(4096,4096). (sensenova_u1.rs:1390.) which: "timestep" | "noise".
    def time_or_scale_embed(
        self, t: Tensor, which: String, ctx: DeviceContext
    ) raises -> Tensor:
        var prefix: String
        if which == "timestep":
            prefix = "fm_modules.timestep_embedder"
        else:
            prefix = "fm_modules.noise_scale_embedder"
        ref w0 = self._shared(prefix + ".mlp.0.weight")
        ref b0 = self._shared(prefix + ".mlp.0.bias")
        ref w2 = self._shared(prefix + ".mlp.2.weight")
        ref b2 = self._shared(prefix + ".mlp.2.bias")
        # sinusoidal freq embed [N,256] in MLP weight dtype (cos-first).
        var fe = timestep_embedding(
            t, 256, ctx, Float32(10000.0), w0.dtype()
        )
        var n = fe.shape()[0]
        var f3d = _reshape3(fe, 1, n, 256, ctx)
        var h0 = linear(f3d, w0, Optional(_clone(b0, ctx)), ctx)
        h0 = silu(h0, ctx)
        var h2 = linear(h0, w2, Optional(_clone(b2, ctx)), ctx)
        var hidden = h2.shape()[2]
        return _reshape2(h2, n, hidden, ctx)

    # fm_head: Linear(4096,4096) -> GELU -> Linear(4096,3072). (rs:1430.)
    def fm_head_forward(self, hidden: Tensor, ctx: DeviceContext) raises -> Tensor:
        ref w0 = self._shared("fm_modules.fm_head.0.weight")
        ref b0 = self._shared("fm_modules.fm_head.0.bias")
        ref w2 = self._shared("fm_modules.fm_head.2.weight")
        ref b2 = self._shared("fm_modules.fm_head.2.bias")
        var h0 = linear(hidden, w0, Optional(_clone(b0, ctx)), ctx)
        h0 = gelu(h0, ctx)
        return linear(h0, w2, Optional(_clone(b2, ctx)), ctx)

    # resolution-aware noise scale (rs:1507):
    #   scale = sqrt((gh*gw)/merge^2/base) * noise_scale, capped at max.
    def compute_noise_scale(self, grid_h: Int, grid_w: Int) -> Float32:
        var merge = Float32(self.config.merge_size)
        var base = Float32(self.config.noise_scale_base_seq)
        var n_tok = Float32(grid_h * grid_w)
        var raw = sqrt(n_tok / (merge * merge) / base) * self.config.noise_scale
        if raw > self.config.noise_scale_max:
            return self.config.noise_scale_max
        return raw

# ── block weight lookup (borrows the loaded block; no copy) ──────────────────
# Resolves `prefix + suffix` against the loaded Block and returns a borrowed
# reference to the Tensor (origin bound to `block`).
def _bget[
    mut: Bool, //, origin: Origin[mut=mut]
](ref [origin] block: Block, p: String, suffix: String) raises -> ref [block] Tensor:
    var full = p + suffix
    if full not in block:
        raise Error(String("sensenova_u1: missing block weight: ") + full)
    return block[full][]


# ── reshape helpers (clone + new shape; Tensor uniquely owns its buffer) ─────
def _reshape2(x: Tensor, a: Int, b: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    return reshape(x, sh^, ctx)


def _reshape3(x: Tensor, a: Int, b: Int, c: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    return reshape(x, sh^, ctx)


def _reshape4(x: Tensor, a: Int, b: Int, c: Int, d: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    sh.append(d)
    return reshape(x, sh^, ctx)


def _reshape6(
    x: Tensor, a: Int, b: Int, c: Int, d: Int, e: Int, f: Int, ctx: DeviceContext
) raises -> Tensor:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    sh.append(d)
    sh.append(e)
    sh.append(f)
    return reshape(x, sh^, ctx)


# [H, S, Dh] (head-contiguous) -> [1, S, H, Dh] (BSHD) via permute(1,0,2)+unsqueeze.
def _bhsd_to_bsh(x: Tensor, h: Int, s: Int, dh: Int, ctx: DeviceContext) raises -> Tensor:
    # x is [H,S,Dh]; permute to [S,H,Dh] then prepend batch=1.
    var perm = List[Int]()
    perm.append(1)
    perm.append(0)
    perm.append(2)
    var p = permute(x, perm, ctx)  # [S,H,Dh]
    return _reshape4(p, 1, s, h, dh, ctx)


# cast F32 tensor -> BF16 (uses ops/cast indirectly to avoid touching ops/).
def _cast_f32_to_bf16(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    from serenitymojo.ops.cast import cast_tensor
    return cast_tensor(x, STDtype.BF16, ctx)
