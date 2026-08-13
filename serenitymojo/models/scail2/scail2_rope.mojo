# SCAIL-2 sequence layout and custom interleaved RoPE tables.
#
# Oracle: zai-org/SCAIL-2, wan-scail2 commit
# 5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5,
# wan/modules/model_scail2.py:
#   rope_apply_ref, rope_apply_video, rope_apply_pose, rope_apply_scail,
#   SCAIL2Model.forward.
#
# SCAIL-2 concatenates tokens in this exact order:
#   [additional refs?] [primary ref] [video] [downsampled pose]
# The primary/additional reference and video regions use ordinary 3-axis Wan
# interleaved RoPE.  Pose is different: the official model constructs the
# high-resolution complex frequency grid, then average-pools its real and
# imaginary components 2x2.  The averaged complex values are not unit length,
# so pose cannot be represented by a single fractional position passed to the
# shared multiaxis table builder.  This module builds the final cos/sin tables
# directly and preserves that complex average exactly (up to F32 trig/storage).

from std.math import exp, log, cos, sin
from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime SCAIL2_HEAD_DIM = 128
comptime SCAIL2_ROPE_HALF = 64
comptime SCAIL2_ROPE_THETA = Float32(10000.0)
comptime SCAIL2_ROPE_TABLE_LENGTH = 8192
comptime SCAIL2_SPATIAL_SHIFT = 120


@fieldwise_init
struct Scail2SequencePlan(Copyable, Movable, ImplicitlyCopyable):
    """Patched-grid geometry for the exact SCAIL-2 attention sequence."""

    var video_t: Int
    var grid_h: Int
    var grid_w: Int
    var additional_ref_count: Int
    var replace_flag: Bool

    def validate(self) raises:
        if self.video_t <= 0:
            raise Error("SCAIL-2 video_t must be positive")
        if self.grid_h <= 0 or self.grid_w <= 0:
            raise Error("SCAIL-2 patched H/W must be positive")
        if self.grid_h % 2 != 0 or self.grid_w % 2 != 0:
            raise Error("SCAIL-2 patched H/W must be even for pose pooling")
        if self.additional_ref_count < 0:
            raise Error("SCAIL-2 additional_ref_count must be non-negative")
        # The pinned official model materializes exactly 8192 complex
        # frequencies per axis and slices that table after applying these
        # shifts. Reject any geometry that the official implementation cannot
        # represent instead of analytically extending it here.
        if self.video_t > SCAIL2_ROPE_TABLE_LENGTH:
            raise Error("SCAIL-2 video_t exceeds the official RoPE table")
        if self.additional_ref_count > SCAIL2_ROPE_TABLE_LENGTH:
            raise Error(
                "SCAIL-2 additional_ref_count exceeds the official RoPE table"
            )
        var base_video_shift = 0 if self.replace_flag else 1
        if (
            self.additional_ref_count + base_video_shift + self.video_t
            > SCAIL2_ROPE_TABLE_LENGTH
        ):
            raise Error("SCAIL-2 shifted temporal range exceeds RoPE table")
        if self.grid_w > SCAIL2_ROPE_TABLE_LENGTH - SCAIL2_SPATIAL_SHIFT:
            raise Error("SCAIL-2 pose width shift exceeds RoPE table")
        var max_grid_h = SCAIL2_ROPE_TABLE_LENGTH
        if self.replace_flag:
            max_grid_h -= SCAIL2_SPATIAL_SHIFT
        if self.grid_h > max_grid_h:
            raise Error("SCAIL-2 height shift exceeds RoPE table")

    def spatial_tokens(self) -> Int:
        return self.grid_h * self.grid_w

    def additional_ref_length(self) -> Int:
        return self.additional_ref_count * self.spatial_tokens()

    def ref_length(self) -> Int:
        return self.spatial_tokens()

    def video_length(self) -> Int:
        return self.video_t * self.spatial_tokens()

    def pose_length(self) -> Int:
        return self.video_t * (self.grid_h // 2) * (self.grid_w // 2)

    def sequence_length(self) -> Int:
        return (
            self.additional_ref_length() + self.ref_length()
            + self.video_length() + self.pose_length()
        )


def build_scail2_position_descriptors(
    plan: Scail2SequencePlan, ctx: DeviceContext
) raises -> Tensor:
    """Return F32 [S,4] rows `[region,t,h,w]` in exact sequence order.

    Region ids are 0=additional-ref, 1=primary-ref, 2=video, 3=pose.
    Coordinates include the official forward's shifts.  Pose h/w are the
    top-left coordinates of the high-resolution 2x2 cell whose complex
    frequencies are averaged; consumers must not treat them as a lone RoPE
    position.
    """
    plan.validate()
    var out = List[Float32]()
    var ref_spatial_shift = 120 if plan.replace_flag else 0
    var base_video_shift = 0 if plan.replace_flag else 1

    for fi in range(plan.additional_ref_count):
        for hi in range(plan.grid_h):
            for wi in range(plan.grid_w):
                out.append(0.0)
                out.append(Float32(fi))
                out.append(Float32(hi + ref_spatial_shift))
                out.append(Float32(wi))

    for hi in range(plan.grid_h):
        for wi in range(plan.grid_w):
            out.append(1.0)
            out.append(Float32(plan.additional_ref_count))
            out.append(Float32(hi + ref_spatial_shift))
            out.append(Float32(wi))

    for fi in range(plan.video_t):
        for hi in range(plan.grid_h):
            for wi in range(plan.grid_w):
                out.append(2.0)
                out.append(Float32(fi + base_video_shift + plan.additional_ref_count))
                out.append(Float32(hi))
                out.append(Float32(wi))

    for fi in range(plan.video_t):
        for hi in range(plan.grid_h // 2):
            for wi in range(plan.grid_w // 2):
                out.append(3.0)
                out.append(Float32(fi + base_video_shift + plan.additional_ref_count))
                out.append(Float32(2 * hi))
                out.append(Float32(2 * wi + 120))

    return Tensor.from_host(out^, [plan.sequence_length(), 4], STDtype.F32, ctx)


def _scail2_rope_kernel[out_dtype: DType](
    cos_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],
    sin_t: LayoutTensor[out_dtype, _DYN2, MutAnyOrigin],
    total_rows_w: Int32,
    video_t_w: Int32,
    grid_h_w: Int32,
    grid_w_w: Int32,
    additional_ref_count_w: Int32,
    replace_mode_w: Int32,
):
    var total_rows = Int(total_rows_w)
    var video_t = Int(video_t_w)
    var grid_h = Int(grid_h_w)
    var grid_w = Int(grid_w_w)
    var additional_ref_count = Int(additional_ref_count_w)
    var replace_mode = Int(replace_mode_w)
    var idx = Int(global_idx.x)
    var total = total_rows * SCAIL2_ROPE_HALF
    if idx >= total:
        return

    var row = idx // SCAIL2_ROPE_HALF
    var col = idx % SCAIL2_ROPE_HALF

    # Official head_dim=128 split: full dimensions [44,42,42], hence complex
    # pair widths [22,21,21].
    var axis = 0
    var local_i = col
    var axis_half = 22
    if col >= 22 and col < 43:
        axis = 1
        local_i = col - 22
        axis_half = 21
    elif col >= 43:
        axis = 2
        local_i = col - 43
        axis_half = 21

    var spatial = grid_h * grid_w
    var add_len = additional_ref_count * spatial
    var ref_end = add_len + spatial
    var video_end = ref_end + video_t * spatial
    var half_h = grid_h // 2
    var half_w = grid_w // 2
    var base_video_shift = 1
    var ref_spatial_shift = 0
    if replace_mode != 0:
        base_video_shift = 0
        ref_spatial_shift = 120

    var pos: Float32
    var pose_region = False
    if row < add_len:
        var fi = row // spatial
        var rem = row % spatial
        var hi = rem // grid_w
        var wi = rem % grid_w
        if axis == 0:
            pos = Float32(fi)
        elif axis == 1:
            pos = Float32(hi + ref_spatial_shift)
        else:
            pos = Float32(wi)
    elif row < ref_end:
        var rem = row - add_len
        var hi = rem // grid_w
        var wi = rem % grid_w
        if axis == 0:
            pos = Float32(additional_ref_count)
        elif axis == 1:
            pos = Float32(hi + ref_spatial_shift)
        else:
            pos = Float32(wi)
    elif row < video_end:
        var rem0 = row - ref_end
        var fi = rem0 // spatial
        var rem = rem0 % spatial
        var hi = rem // grid_w
        var wi = rem % grid_w
        if axis == 0:
            pos = Float32(fi + base_video_shift + additional_ref_count)
        elif axis == 1:
            pos = Float32(hi)
        else:
            pos = Float32(wi)
    else:
        pose_region = True
        var pose_spatial = half_h * half_w
        var rem0 = row - video_end
        var fi = rem0 // pose_spatial
        var rem = rem0 % pose_spatial
        var hi = rem // half_w
        var wi = rem % half_w
        if axis == 0:
            pos = Float32(fi + base_video_shift + additional_ref_count)
        elif axis == 1:
            pos = Float32(2 * hi)
        else:
            pos = Float32(2 * wi + 120)

    var inv_freq = exp(
        -log(SCAIL2_ROPE_THETA)
        * (Float32(local_i) / Float32(axis_half))
    )
    var angle = pos * inv_freq
    var cv = cos(angle)
    var sv = sin(angle)

    # avg_pool2d(real/imag, kernel=2, stride=2) in the official pose path.
    # Temporal values are identical across the 2x2 cell. H and W values each
    # average two adjacent coordinates (the orthogonal duplicate cancels).
    if pose_region and axis != 0:
        var angle_next = (pos + 1.0) * inv_freq
        cv = 0.5 * (cv + cos(angle_next))
        sv = 0.5 * (sv + sin(angle_next))

    cos_t[row, col] = rebind[cos_t.element_type](cv.cast[out_dtype]())
    sin_t[row, col] = rebind[sin_t.element_type](sv.cast[out_dtype]())


def build_scail2_rope_tables(
    plan: Scail2SequencePlan,
    ctx: DeviceContext,
    out_dtype: STDtype,
) raises -> Tuple[Tensor, Tensor]:
    """Build final interleaved `(cos,sin)` tables, each `[S,64]`."""
    plan.validate()
    var rows = plan.sequence_length()
    var n = rows * SCAIL2_ROPE_HALF
    var cos_buf = ctx.enqueue_create_buffer[DType.uint8](n * out_dtype.byte_size())
    var sin_buf = ctx.enqueue_create_buffer[DType.uint8](n * out_dtype.byte_size())
    var rl = RuntimeLayout[_DYN2].row_major(
        IndexList[2](rows, SCAIL2_ROPE_HALF)
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    var replace_mode = 1 if plan.replace_flag else 0
    var odt = out_dtype.to_mojo_dtype()
    if odt == DType.float32:
        var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        var S = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_scail2_rope_kernel[DType.float32]](
            C, S, Int32(rows), Int32(plan.video_t), Int32(plan.grid_h), Int32(plan.grid_w),
            Int32(plan.additional_ref_count), Int32(replace_mode),
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif odt == DType.bfloat16:
        var C = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        var S = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_scail2_rope_kernel[DType.bfloat16]](
            C, S, Int32(rows), Int32(plan.video_t), Int32(plan.grid_h), Int32(plan.grid_w),
            Int32(plan.additional_ref_count), Int32(replace_mode),
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif odt == DType.float16:
        var C = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cos_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        var S = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(sin_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_scail2_rope_kernel[DType.float16]](
            C, S, Int32(rows), Int32(plan.video_t), Int32(plan.grid_h), Int32(plan.grid_w),
            Int32(plan.additional_ref_count), Int32(replace_mode),
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        raise Error("SCAIL-2 RoPE output dtype must be F32, BF16, or F16")

    var shape = List[Int]()
    shape.append(rows)
    shape.append(SCAIL2_ROPE_HALF)
    var shape2 = shape.copy()
    return (
        Tensor(cos_buf^, shape^, out_dtype),
        Tensor(sin_buf^, shape2^, out_dtype),
    )
