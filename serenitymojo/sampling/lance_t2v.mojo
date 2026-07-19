# sampling/lance_t2v.mojo - Lance shifted-flow scheduler and CFG helpers.
#
# Reference:
#   /home/alex/EriDiffusion/inference-flame/src/models/lance.rs
#   /home/alex/Lance/modeling/lance/lance.py::validation_gen_KVcache
#
# Lance uses a decreasing shifted-flow schedule and an Euler update
#   x_next = x - (t_i - t_{i+1}) * v
# where v points from data toward noise. T2V CFG applies textbook text CFG,
# then global norm renorm against the conditional velocity.

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, barrier, global_idx
from std.gpu.memory import AddressSpace
from std.math import sqrt
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import add, mul_scalar, sub


comptime _DYN1 = Layout.row_major(-1)
comptime _TPB = 256
comptime _BLOCK = 256


def lance_shifted_t(index: Int, num_steps: Int, shift: Float32) raises -> Float32:
    if num_steps <= 0:
        raise Error("lance_shifted_t: num_steps must be > 0")
    if shift <= Float32(0.0):
        raise Error("lance_shifted_t: shift must be > 0")
    if index < 0 or index > num_steps:
        raise Error("lance_shifted_t: index out of range")
    var t = Float32(1.0) - Float32(index) / Float32(num_steps)
    return shift * t / (Float32(1.0) + (shift - Float32(1.0)) * t)


def build_lance_timestep_schedule(num_steps: Int, shift: Float32) raises -> List[Float32]:
    """Host scalar schedule of length num_steps+1.

    Host scalar setup is intentional; tensor math stays on GPU.
    """
    var out = List[Float32]()
    for i in range(num_steps + 1):
        out.append(lance_shifted_t(i, num_steps, shift))
    return out^


def lance_timestep_tensor(n: Int, t: Float32, ctx: DeviceContext) raises -> Tensor:
    if n <= 0:
        raise Error("lance_timestep_tensor: n must be > 0")
    var vals = List[Float32]()
    for _ in range(n):
        vals.append(t)
    var sh = List[Int]()
    sh.append(n)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


def lance_cfg(
    v_uncond: Tensor, v_cond: Tensor, guidance_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Text CFG: uncond + scale * (cond - uncond)."""
    if v_uncond.numel() != v_cond.numel():
        raise Error("lance_cfg: shape mismatch")
    if v_uncond.dtype() != v_cond.dtype():
        raise Error("lance_cfg: dtype mismatch")
    var diff = sub(v_cond, v_uncond, ctx)
    var scaled = mul_scalar(diff, guidance_scale, ctx)
    return add(v_uncond, scaled, ctx)


def lance_denoise_step(
    x_t: Tensor, v_pred: Tensor, dt: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Euler update x_next = x_t - dt * v_pred with positive dt."""
    if x_t.numel() != v_pred.numel():
        raise Error("lance_denoise_step: shape mismatch")
    if x_t.dtype() != v_pred.dtype():
        raise Error("lance_denoise_step: dtype mismatch")
    var delta = mul_scalar(v_pred, dt, ctx)
    return sub(x_t, delta, ctx)


def _sumsq_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local: Float32 = 0.0
    var i = tid
    while i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        local += v * v
        i += _TPB
    shared[tid] = local
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        o[0] = shared[0]


def _sumsq_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local: Float32 = 0.0
    var i = tid
    while i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        local += v * v
        i += _TPB
    shared[tid] = local
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        o[0] = shared[0]


def _sumsq_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local: Float32 = 0.0
    var i = tid
    while i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        local += v * v
        i += _TPB
    shared[tid] = local
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    if tid == 0:
        o[0] = shared[0]


def _global_sumsq(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](4)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](1))
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_sumsq_kernel_f32, _sumsq_kernel_f32](
            X, O, n, grid_dim=1, block_dim=_TPB
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_sumsq_kernel_bf16, _sumsq_kernel_bf16](
            X, O, n, grid_dim=1, block_dim=_TPB
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[_sumsq_kernel_f16, _sumsq_kernel_f16](
            X, O, n, grid_dim=1, block_dim=_TPB
        )
    ctx.synchronize()
    return Tensor(out_buf^, [1], STDtype.F32)


def _renorm_kernel_f32(
    v_cfg: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cond_sumsq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cfg_sumsq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    renorm_min: Float32,
    renorm_max: Float32,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var cfg_norm = sqrt(rebind[Scalar[DType.float32]](cfg_sumsq[0]))
        var scale = renorm_max
        if cfg_norm > Float32(0.0):
            scale = sqrt(rebind[Scalar[DType.float32]](cond_sumsq[0])) / (
                cfg_norm + Float32(1.0e-8)
            )
        if scale < renorm_min:
            scale = renorm_min
        if scale > renorm_max:
            scale = renorm_max
        o[idx] = rebind[o.element_type](
            rebind[Scalar[DType.float32]](v_cfg[idx]) * scale
        )


def _renorm_kernel_bf16(
    v_cfg: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    cond_sumsq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cfg_sumsq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    renorm_min: Float32,
    renorm_max: Float32,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var cfg_norm = sqrt(rebind[Scalar[DType.float32]](cfg_sumsq[0]))
        var scale = renorm_max
        if cfg_norm > Float32(0.0):
            scale = sqrt(rebind[Scalar[DType.float32]](cond_sumsq[0])) / (
                cfg_norm + Float32(1.0e-8)
            )
        if scale < renorm_min:
            scale = renorm_min
        if scale > renorm_max:
            scale = renorm_max
        var v = rebind[Scalar[DType.bfloat16]](v_cfg[idx]).cast[DType.float32]()
        o[idx] = rebind[o.element_type]((v * scale).cast[DType.bfloat16]())


def _renorm_kernel_f16(
    v_cfg: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    cond_sumsq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    cfg_sumsq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    renorm_min: Float32,
    renorm_max: Float32,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var cfg_norm = sqrt(rebind[Scalar[DType.float32]](cfg_sumsq[0]))
        var scale = renorm_max
        if cfg_norm > Float32(0.0):
            scale = sqrt(rebind[Scalar[DType.float32]](cond_sumsq[0])) / (
                cfg_norm + Float32(1.0e-8)
            )
        if scale < renorm_min:
            scale = renorm_min
        if scale > renorm_max:
            scale = renorm_max
        var v = rebind[Scalar[DType.float16]](v_cfg[idx]).cast[DType.float32]()
        o[idx] = rebind[o.element_type]((v * scale).cast[DType.float16]())


def lance_cfg_renorm(
    v_cfg: Tensor,
    v_cond: Tensor,
    renorm_min: Float32,
    renorm_max: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Global CFG renorm used by Lance T2V.

    Computes clamp(norm(v_cond)/(norm(v_cfg)+1e-8), min, max) on GPU and
    scales v_cfg without host tensor readback.
    """
    if v_cfg.numel() != v_cond.numel():
        raise Error("lance_cfg_renorm: shape mismatch")
    if v_cfg.dtype() != v_cond.dtype():
        raise Error("lance_cfg_renorm: dtype mismatch")
    if renorm_min > renorm_max:
        raise Error("lance_cfg_renorm: min > max")
    var cond_sumsq = _global_sumsq(v_cond, ctx)
    var cfg_sumsq = _global_sumsq(v_cfg, ctx)
    var n = v_cfg.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](v_cfg.nbytes())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](1))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = v_cfg.dtype().to_mojo_dtype()
    var CS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        cond_sumsq.buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    var GS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        cfg_sumsq.buf.unsafe_ptr().bitcast[Float32](), s_rl
    )
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            v_cfg.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        ctx.enqueue_function[_renorm_kernel_f32, _renorm_kernel_f32](
            X, CS, GS, O, renorm_min, renorm_max, n,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            v_cfg.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        ctx.enqueue_function[_renorm_kernel_bf16, _renorm_kernel_bf16](
            X, CS, GS, O, renorm_min, renorm_max, n,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            v_cfg.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        ctx.enqueue_function[_renorm_kernel_f16, _renorm_kernel_f16](
            X, CS, GS, O, renorm_min, renorm_max, n,
            grid_dim=grid, block_dim=_BLOCK,
        )
    ctx.synchronize()
    return Tensor(out_buf^, v_cfg.shape(), v_cfg.dtype())
