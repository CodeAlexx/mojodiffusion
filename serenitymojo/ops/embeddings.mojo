# ops/embeddings.mojo — DiT timestep embedding + RoPE freq-table construction.
#
# Two DiT-foundation pieces, EXACT formulas mirrored from the Z-Image NextDiT
# reference (read line-by-line, do NOT infer):
#   /home/alex/EriDiffusion/inference-flame/src/models/zimage_nextdit.rs
#
# ── timestep_embedding (zimage_nextdit.rs:411-428) ──────────────────────────
#   Sinusoidal embedding of a scalar timestep, COS first then SIN:
#     half   = dim / 2
#     freq_i = exp(-ln(max_period) * i / half)        for i in [0, half)
#     angle  = t * freq_i
#     emb[n, i]        = cos(angle)                    (first half)
#     emb[n, half + i] = sin(angle)                    (second half)
#   Output shape [N, dim].  (Rust uses freq_dim = config.min_mod = 256,
#   max_period = 10000.0 — getting the cos/sin order wrong silently breaks DiT.)
#
# ── t_embedder (zimage_nextdit.rs:437-439) ──────────────────────────────────
#   timestep_embedding -> Linear(t_embedder.mlp.0) -> SiLU -> Linear(t_embedder.mlp.2)
#   (the DiT timestep MLP; biases on both linears in the reference.)
#
# ── build_rope_tables (zimage_nextdit.rs:693-709) ───────────────────────────
#   Per-axis RoPE inv_freq (single-axis form):
#     half     = head_dim / 2
#     inv_freq_i = 1 / theta^(i / half)               for i in [0, half)
#     angle      = position * inv_freq_i
#     cos[pos, i] = cos(angle);  sin[pos, i] = sin(angle)
#   cos/sin tables shaped [num_positions, head_dim/2] — EXACTLY the layout
#   `serenitymojo/ops/rope.rope_halfsplit` consumes (cos/sin = [rows, D/2], the
#   row index shared between data tensor and freq tensor; Z-Image rope_theta=256).
#
# Compute is F32; outputs stored as F32 here (so the parity gate isolates op
# correctness from BF16 quantization). Only two new kernels are introduced — the
# sinusoidal compute and the freq-table build; everything else reuses
# ops/linear + ops/activations.silu.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import exp, log, cos, sin
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256


# ── sinusoidal timestep-embedding kernel (F32 in / F32 out) ─────────────────
def _timestep_embed_kernel_f32(
    t: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    n: Int,
    dim: Int,
    half: Int,
    neg_ln_max_period: Float32,  # -ln(max_period), precomputed on host
):
    # One thread per (row, i) pair: row in [0, n), i in [0, half).
    var idx = Int(global_idx.x)
    var total = n * half
    if idx < total:
        var row = idx // half
        var i = idx % half
        var tv = rebind[Scalar[DType.float32]](t[row])
        # freq = exp(-ln(max_period) * i / half)
        var freq = exp(neg_ln_max_period * (Float32(i) / Float32(half)))
        var angle = tv * freq
        # COS first (cols [0, half)), SIN second (cols [half, dim)).
        o[row, i] = rebind[o.element_type](cos(angle))
        o[row, half + i] = rebind[o.element_type](sin(angle))


# ── sinusoidal timestep-embedding kernel — SIN-FIRST variant (ERNIE) ────────
# ERNIE-Image's `time_embed` (ernie_image.rs:588-609) builds the sinusoidal
# vector as `cat([sin_part, cos_part], dim=1)` — the OPPOSITE order from
# Z-Image NextDiT (zimage_nextdit.rs:425-426, cos-first). The trained
# `time_embedding.linear_1.weight` rows assume the SIN block sits in cols
# [0, half) and the COS block sits in cols [half, dim); feeding the cos-first
# kernel's output drives downstream activations into BF16 overflow after the
# shared AdaLN linear (4096 -> 24576). This kernel is identical to the
# cos-first kernel above except the two writes are swapped. The new
# `timestep_embedding_sin_first` host function below dispatches it. Z-Image,
# FLUX, Klein, Qwen, HiDream, SenseNova and SDXL must continue to call the
# cos-first `timestep_embedding`.
def _timestep_embed_kernel_f32_sin_first(
    t: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    n: Int,
    dim: Int,
    half: Int,
    neg_ln_max_period: Float32,
):
    var idx = Int(global_idx.x)
    var total = n * half
    if idx < total:
        var row = idx // half
        var i = idx % half
        var tv = rebind[Scalar[DType.float32]](t[row])
        var freq = exp(neg_ln_max_period * (Float32(i) / Float32(half)))
        var angle = tv * freq
        # SIN first (cols [0, half)), COS second (cols [half, dim)) — ERNIE.
        o[row, i] = rebind[o.element_type](sin(angle))
        o[row, half + i] = rebind[o.element_type](cos(angle))


def timestep_embedding(
    t: Tensor, dim: Int, ctx: DeviceContext, max_period: Float32 = 10000.0
) raises -> Tensor:
    """Sinusoidal timestep embedding (Z-Image NextDiT order: COS then SIN).

    t:   [N]            scalar timesteps (1-D; flattened length = N). F32.
    dim: embedding dim  (must be even; half = dim/2 cos + half sin).
    returns [N, dim]    F32 storage; F32 math.
    """
    if dim % 2 != 0:
        raise Error("timestep_embedding: dim must be even")
    if t.dtype() != STDtype.F32:
        raise Error("timestep_embedding: t must be F32")
    var n = t.numel()
    var half = dim // 2
    var neg_ln_mp = -log(max_period)

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
    ctx.enqueue_function[
        _timestep_embed_kernel_f32, _timestep_embed_kernel_f32
    ](T, O, n, dim, half, neg_ln_mp, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var out_shape = List[Int]()
    out_shape.append(n)
    out_shape.append(dim)
    return Tensor(out_buf^, out_shape^, STDtype.F32)


def timestep_embedding_sin_first(
    t: Tensor, dim: Int, ctx: DeviceContext, max_period: Float32 = 10000.0
) raises -> Tensor:
    """Sinusoidal timestep embedding (ERNIE order: SIN then COS).

    Equivalent to `timestep_embedding` except the two halves are swapped:
        emb[n, i]        = sin(angle)        cols [0, half)
        emb[n, half + i] = cos(angle)        cols [half, dim)
    This matches ERNIE-Image (`ernie_image.rs:603` `cat([sin, cos], 1)`).
    Z-Image and all other DiTs in this repo train against cos-first and MUST
    keep calling `timestep_embedding`. Confirmed via skeptic finding
    `serenitymojo/parity/SKEPTIC_FINDINGS_ernie_block0_2026-05-28.md` (A2/A5).

    t:   [N]            scalar timesteps (1-D; flattened length = N). F32.
    dim: embedding dim  (must be even).
    returns [N, dim]    F32 storage; F32 math.
    """
    if dim % 2 != 0:
        raise Error("timestep_embedding_sin_first: dim must be even")
    if t.dtype() != STDtype.F32:
        raise Error("timestep_embedding_sin_first: t must be F32")
    var n = t.numel()
    var half = dim // 2
    var neg_ln_mp = -log(max_period)

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
    ctx.enqueue_function[
        _timestep_embed_kernel_f32_sin_first,
        _timestep_embed_kernel_f32_sin_first,
    ](T, O, n, dim, half, neg_ln_mp, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var out_shape = List[Int]()
    out_shape.append(n)
    out_shape.append(dim)
    return Tensor(out_buf^, out_shape^, STDtype.F32)


def t_embedder(
    t: Tensor,
    dim: Int,
    mlp0_weight: Tensor,
    mlp0_bias: Optional[Tensor],
    mlp2_weight: Tensor,
    mlp2_bias: Optional[Tensor],
    ctx: DeviceContext,
    max_period: Float32 = 10000.0,
) raises -> Tensor:
    """DiT timestep MLP: timestep_embedding -> Linear -> SiLU -> Linear.

    t:           [N]                scalar timesteps (F32).
    dim:         sinusoidal embed dim (== mlp.0 weight in-dim).
    mlp0_weight: [hidden, dim]      PyTorch row-major (t_embedder.mlp.0.weight).
    mlp0_bias:   [hidden] or None   (t_embedder.mlp.0.bias).
    mlp2_weight: [out, hidden]      PyTorch row-major (t_embedder.mlp.2.weight).
    mlp2_bias:   [out] or None      (t_embedder.mlp.2.bias).
    returns [N, out].
    The sinusoidal embedding is F32; it is cast to the MLP weights' dtype before
    the first Linear so the GEMM dtype matches (mirrors the Rust BF16 t-embed).
    """
    var emb = timestep_embedding(t, dim, ctx, max_period)  # [N, dim] F32
    # Cast the F32 embedding to the MLP weights' compute dtype (BF16 in the
    # reference) on GPU. Keep inference activations off the CPU path.
    var w_dtype = mlp0_weight.dtype()
    var emb_in: Tensor
    if w_dtype == STDtype.F32:
        emb_in = emb^
    else:
        emb_in = cast_tensor(emb, w_dtype, ctx)
    var h = linear(emb_in, mlp0_weight, mlp0_bias, ctx)    # [N, hidden]
    var ha = silu(h, ctx)
    return linear(ha, mlp2_weight, mlp2_bias, ctx)         # [N, out]


# ── RoPE freq-table build kernel (F32 in / F32 out) ─────────────────────────
def _rope_tables_kernel_f32(
    positions: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cos_t: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    sin_t: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    rows: Int,
    half: Int,  # head_dim/2
    theta: Float32,
):
    # One thread per (row, i): row in [0, rows), i in [0, half).
    var idx = Int(global_idx.x)
    var total = rows * half
    if idx < total:
        var row = idx // half
        var i = idx % half
        var pos = rebind[Scalar[DType.float32]](positions[row])
        # inv_freq = 1 / theta^(i / half) = theta^(-i/half)
        var inv_freq = exp(-log(theta) * (Float32(i) / Float32(half)))
        var angle = pos * inv_freq
        cos_t[row, i] = rebind[cos_t.element_type](cos(angle))
        sin_t[row, i] = rebind[sin_t.element_type](sin(angle))


def build_rope_tables(
    positions: Tensor, head_dim: Int, theta: Float32, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """RoPE cos/sin tables in the half-split layout `rope_halfsplit` consumes.

    positions: [rows]            position indices (F32).
    head_dim:  rotary head dim   (must be even; half = head_dim/2 angles).
    theta:     rope_theta        (Z-Image = 256.0).
    returns (cos, sin), each [rows, head_dim/2], F32. Feed straight into
    ops/rope.rope_halfsplit with an x of shape [rows, head_dim].
    """
    if head_dim % 2 != 0:
        raise Error("build_rope_tables: head_dim must be even")
    if positions.dtype() != STDtype.F32:
        raise Error("build_rope_tables: positions must be F32")
    var rows = positions.numel()
    var half = head_dim // 2

    var cos_buf = ctx.enqueue_create_buffer[DType.uint8](rows * half * 4)
    var sin_buf = ctx.enqueue_create_buffer[DType.uint8](rows * half * 4)
    var p_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, half))
    var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        positions.buf.unsafe_ptr().bitcast[Float32](), p_rl
    )
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        cos_buf.unsafe_ptr().bitcast[Float32](), f_rl
    )
    var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        sin_buf.unsafe_ptr().bitcast[Float32](), f_rl
    )
    var total = rows * half
    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_rope_tables_kernel_f32, _rope_tables_kernel_f32](
        P, C, S, rows, half, theta, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()

    var cos_shape = List[Int]()
    cos_shape.append(rows)
    cos_shape.append(half)
    var sin_shape = List[Int]()
    sin_shape.append(rows)
    sin_shape.append(half)
    var cos_out = Tensor(cos_buf^, cos_shape^, STDtype.F32)
    var sin_out = Tensor(sin_buf^, sin_shape^, STDtype.F32)
    return (cos_out^, sin_out^)
