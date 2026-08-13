# serenitymojo/models/dit/minimax_h3_rope.mojo — device (GPU) 3-axis MM-RoPE
# table builder for MiniMax-H3.
#
# GPU counterpart to the host-float32 oracle `models/minimax_h3/dit_frontend.
# mojo` (`minimax_h3_rope_inv_freq` / `minimax_h3_rope_table`, gated 15/15 vs
# the diffusers reference). The oracle's table build is a per-row HOST loop
# over `List[Float64]` — correct, and fine for its own 15-row gate fixture,
# but it cannot scale to a real packed sequence (thousands of rows) or feed a
# device forward without a host round trip every step. This module reproduces
# the SAME math as one GPU thread per (row, output-column):
#
#   cos/sin[row, col] = trig(pos[row, axis(col)] * inv_freq[local_i(col)])
#
# where axis(col)/local_i(col) fold the `[S, 2*3*rope_freq_dim]` output the
# same way the oracle's per-row loop does: the first `3*rope_freq_dim`
# columns are the concatenation of the (t, h, w) angle blocks, and the second
# `3*rope_freq_dim` columns are a LITERAL duplicate of the first — the
# rotate-half convention needs both halves (dit_frontend.mojo:24-27, 108-110,
# 130-139).
#
# inv_freq (16 entries, independent of sequence length) is computed ONCE on
# the HOST by calling the oracle's OWN `minimax_h3_rope_inv_freq` — reused,
# not reimplemented. This is not "device code calling the oracle": it is
# host driver code preparing a 16-float kernel input before the launch,
# exactly the way `ops/rope_tables.build_multiaxis_rope_tables_per_axis`
# uploads a host-computed `thetas: List[Float32]` before its own kernel runs
# (ops/rope_tables.mojo:271-320), and exactly the way
# `models/minimax_h3/block_forward.mojo` (unit 9, the block-forward oracle)
# already imports the same function (block_forward.mojo:42-48). Reusing the
# gated function is safer than a second hand-transcription of the double-
# rounding trap it encodes:
#
#   power = Float32(rope_theta ** exponent)   # round the f64 pow to f32 FIRST
#   inv_freq = Float32(1.0) / power           # then divide, ALSO at f32
#
# Doing the whole computation in f64 and rounding once, or using this
# toolchain's float32 `pow`, both measurably diverge from the diffusers
# reference (dit_frontend.mojo:56-74). This is exactly why the existing
# generic multi-axis builder (`ops/rope_tables.build_multiaxis_rope_tables`,
# which derives inv_freq IN-KERNEL via `exp(-log(theta) * i/half)`) is NOT
# reused here for the frequency generation: that is a different — and for
# THIS model wrong — arithmetic path. It also can't express "one shared
# inv_freq re-used across all three axes", which is H3's convention (contrast
# wan22/cosmos, where each axis owns a disjoint slice of head_dim).
#
# cos/sin are evaluated directly at float32 (no float64 range-reduction, and
# no F64 promotion of the angle before calling trig): the oracle's own
# docstring says promoting to float64 first is MORE accurate and therefore
# WRONG here (dit_frontend.mojo:131-134). Unlike Ideogram-4's MRoPE
# (positions up to ~65536, needing the F64-reduce-then-F32-trig dance in
# `models/dit/ideogram4_mrope.mojo`), H3's rotary positions stay small — a
# `ROPE_FRAME_RESCALE=5/3`-per-latent-frame temporal clock and a
# `ROPE_SPATIAL_SCALE=32`-normalized spatial grid
# (`models/minimax_h3/packing.mojo:77-81`), nowhere near the magnitude where
# Mojo's float32 trig goes wrong.
#
# LAYOUT: this table is FULL WIDTH `[rows, rotary_dim=96]`, matching the
# oracle's `MiniMaxH3RopeTable.cos`/`.sin` shape exactly (dit_frontend.mojo's
# own header: "position_ids [S,3] -> cos/sin [S, 2*3*rope_freq_dim] = [S,96]").
# It feeds `ops/rope.rope_halfsplit_full` directly. Verified by hand against
# `models/minimax_h3/block_forward.mojo::_apply_rope_inplace` (the gated
# block oracle, lines 160-189): that function's rotate-half indexing —
# `partner = -original[i+half]` for `i<half`, `original[i-half]` otherwise,
# combined via `original[i]*cos[row*rotary_dim+i] + partner*sin[...]` — is
# arithmetically identical to `rope_halfsplit_full`'s `cv0/sv0` (first half,
# `o[i]=x0*cv0-x1*sv0`) and `cv1/sv1` (second half, `o[i+half]=x1*cv1+x0*sv1`)
# once the table is duplicated across both halves, which is exactly what
# `_minimax_h3_rope_kernel` below writes. (`rope_halfsplit`, the half-width
# variant, would give byte-identical results too since cos[i]==cos[i+half] —
# `rope_halfsplit_full` is used because it matches the oracle's already-
# committed full-width `MiniMaxH3RopeTable` shape without a second table.)
#
# `angles` (pos * inv_freq, pre-trig) is carried alongside cos/sin so a gate
# can isolate the ARITHMETIC (bit-exact, ours to get right) from the
# TRANSCENDENTAL (device libm intrinsic vs host libm — a few ulp apart on
# large arguments, not ours to fix) — the same split the oracle's own
# `MiniMaxH3RopeTable.angles` field exists for (dit_frontend.mojo:88-92,
# 96-97). Only the first half (`3*rope_freq_dim` columns) is independent
# information; the second half duplicates it, same as cos/sin.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import cos as fcos, sin as fsin
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.minimax_h3.dit_frontend import (
    MINIMAX_H3_ROPE_FREQ_DIM,
    MINIMAX_H3_ROPE_THETA,
    minimax_h3_rope_inv_freq,
)

comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256

# Rotary axes (t, h, w). Deliberately its OWN constant, not a reuse of
# `dit_frontend.MINIMAX_H3_MODALITY_NUM` — both happen to be 3, but one counts
# rotary axes and the other counts AdaLN modality tags (video/text/audio).
# Conflating them is exactly the kind of silent, plausible-looking bug this
# port's docs keep flagging (see minimax_h3_dit.mojo's AdaLN section).
comptime MINIMAX_H3_NUM_AXES = 3


def _minimax_h3_rope_kernel[out_dtype: DType](
    positions: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [rows*3]
    inv_freq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [freq_dim]
    cos_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],  # [rows, rotary_dim]
    sin_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],  # [rows, rotary_dim]
    angle_t: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],  # [rows, rotary_dim]
    rows_w: Int32,
    freq_dim_w: Int32,
    half_w: Int32,  # 3 * freq_dim
    rotary_dim_w: Int32,  # 2 * half
):
    """One thread per (row, output column) of the full `[rows, rotary_dim]`
    table. `col` folds onto `[0, half)` via `col if col < half else col -
    half` — the two halves are a literal duplicate, matching the oracle's
    `for _ in range(2): ...` loop (dit_frontend.mojo:135-139). Within a half,
    `d // freq_dim` selects the axis (0=t, 1=h, 2=w) and `d % freq_dim` the
    local frequency index, matching the oracle's `t-block, then h-block, then
    w-block` concatenation (dit_frontend.mojo:124-129)."""
    var rows = Int(rows_w)
    var freq_dim = Int(freq_dim_w)
    var half = Int(half_w)
    var rotary_dim = Int(rotary_dim_w)
    var idx = Int(global_idx.x)
    var total = rows * rotary_dim
    if idx >= total:
        return
    var row = idx // rotary_dim
    var col = idx % rotary_dim
    var d = col if col < half else col - half
    var axis = d // freq_dim
    var local_i = d % freq_dim
    var pos = rebind[Scalar[DType.float32]](
        positions[row * MINIMAX_H3_NUM_AXES + axis]
    )
    var freq = rebind[Scalar[DType.float32]](inv_freq[local_i])
    var angle = pos * freq
    angle_t[row, col] = rebind[angle_t.element_type](angle)
    cos_t[row, col] = rebind[cos_t.element_type](fcos(angle).cast[out_dtype]())
    sin_t[row, col] = rebind[sin_t.element_type](fsin(angle).cast[out_dtype]())


def build_minimax_h3_rope_tables(
    positions: Tensor,
    ctx: DeviceContext,
    rope_freq_dim: Int = MINIMAX_H3_ROPE_FREQ_DIM,
    rope_theta: Float64 = MINIMAX_H3_ROPE_THETA,
    out_dtype: STDtype = STDtype.F32,
) raises -> Tuple[Tensor, Tensor, Tensor]:
    """Device 3-axis MM-RoPE cos/sin/angle tables for MiniMax-H3.

    positions: F32 `[rows * 3]`, row-major (t, h, w) per row — the caller
               casts the packing module's `List[Float64]` position_ids to F32
               BEFORE upload (e.g. via `Tensor.from_host` on a staged
               `List[Float32]`), mirroring the oracle's own
               `Float32(position_ids[3*row + axis])` cast
               (dit_frontend.mojo:120-122).
    rope_freq_dim / rope_theta: the released config's `rope_inv_freq_len=16`
               (models/dit/minimax_h3_dit.mojo:16) and the diffusers default
               `theta=10000.0` (dit_frontend.mojo:42). Both default to the
               oracle's own constants so a bare call matches production.
    out_dtype: storage dtype for the RETURNED cos/sin (F32 or BF16 — H3's own
               rope math is F32-only per the checkpoint's mixed-precision
               layout, but the apply kernels accept either as a table dtype
               distinct from the activation dtype). `angles` is always F32
               (a diagnostic tap, not a runtime tensor).

    returns (cos, sin, angles), each `[rows, rotary_dim]` where
            `rotary_dim = 2 * 3 * rope_freq_dim` (96 for the released config).
            Feed cos/sin straight into `ops/rope.rope_halfsplit_full` — see
            this file's header for why that op is the byte-for-byte match for
            `models/minimax_h3/block_forward.mojo::_apply_rope_inplace`
            (only the leading `rotary_dim` of each `attention_head_dim=128`
            head rotates; the remaining 32 channels pass through untouched —
            slicing that is the caller's job, same as the oracle).
    """
    if positions.dtype() != STDtype.F32:
        raise Error("build_minimax_h3_rope_tables: positions must be F32")
    if rope_freq_dim <= 0:
        raise Error(
            "build_minimax_h3_rope_tables: rope_freq_dim must be positive"
        )
    var pn = positions.numel()
    if pn % MINIMAX_H3_NUM_AXES != 0:
        raise Error(
            "build_minimax_h3_rope_tables: positions numel must be rows*3"
        )
    var rows = pn // MINIMAX_H3_NUM_AXES

    # inv_freq: reuse the gated oracle function (16 host floats, independent
    # of rows) rather than re-deriving the double-rounding trap — see header.
    var inv_freq_host = minimax_h3_rope_inv_freq(rope_freq_dim, rope_theta)
    var inv_freq_tensor = Tensor.from_host(
        inv_freq_host, [rope_freq_dim], STDtype.F32, ctx
    )

    var half = MINIMAX_H3_NUM_AXES * rope_freq_dim
    var rotary_dim = 2 * half

    var cos_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * rotary_dim * out_dtype.byte_size()
    )
    var sin_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * rotary_dim * out_dtype.byte_size()
    )
    var angle_buf = ctx.enqueue_create_buffer[DType.uint8](
        rows * rotary_dim * STDtype.F32.byte_size()
    )

    var p_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](pn))
    var if_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rope_freq_dim))
    var f_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, rotary_dim))

    var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(positions.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=p_rl,
    )
    var IF = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(inv_freq_tensor.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=if_rl,
    )
    var ANG = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(angle_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )

    var total = rows * rotary_dim
    var grid = (total + _BLOCK - 1) // _BLOCK
    var odt = out_dtype.to_mojo_dtype()
    if odt == DType.float32:
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=f_rl,
    )
        ctx.enqueue_function[_minimax_h3_rope_kernel[DType.float32]](
            P, IF, C, S, ANG, Int32(rows), Int32(rope_freq_dim), Int32(half), Int32(rotary_dim),
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif odt == DType.bfloat16:
        var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
        var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=f_rl,
    )
        ctx.enqueue_function[_minimax_h3_rope_kernel[DType.bfloat16]](
            P, IF, C, S, ANG, Int32(rows), Int32(rope_freq_dim), Int32(half), Int32(rotary_dim),
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        raise Error(
            "build_minimax_h3_rope_tables: out_dtype must be F32 or BF16"
        )
    ctx.synchronize()

    var cos_shape: List[Int] = [rows, rotary_dim]
    var sin_shape: List[Int] = [rows, rotary_dim]
    var angle_shape: List[Int] = [rows, rotary_dim]
    return (
        Tensor(cos_buf^, cos_shape^, out_dtype),
        Tensor(sin_buf^, sin_shape^, out_dtype),
        Tensor(angle_buf^, angle_shape^, STDtype.F32),
    )
