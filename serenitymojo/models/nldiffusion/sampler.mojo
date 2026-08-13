# models/nldiffusion/sampler.mojo — CHUNK E: NL-Diffusion-Image DETERMINISTIC
# unmask-sampler ops (INFERENCE-only, GPU-only where it matters).
#
# The production sampler (text_to_image, modeling_nemotron_labs_diffusion_image.py
# L557-856) is STOCHASTIC: sample_policy='multinomial' (Categorical.sample) +
# confidence_policy in {'mmada','mask_git'} (gumbel / multinomial). Those use
# torch RNG and are NOT bit-reproducible in Mojo. We therefore port + gate ONLY
# the DETERMINISTIC config: sample_policy='argmax' + confidence_policy='stratified'
# (no RNG). This file provides the deterministic building blocks; wiring the full
# generation loop is CHUNK F (not done here).
#
# Ops (reference line numbers are into modeling_nemotron_labs_diffusion_image.py):
#   1. nldiff_num_transfer_tokens  — _get_num_transfer_tokens (L206-233), the
#      shift (logit-normal) per-step token-budget schedule. Host Int math (tiny).
#   2. nldiff_temp_schedule_linear — linear temp schedule (L671-673). Host math.
#   3. nldiff_cfg_combine          — CFG combine (L773): (1+gs)*cond - gs*uncond,
#      preserving -inf where cond is -inf. GPU elementwise (logit-sized).
#   4. nldiff_argmax_lastdim       — logits.argmax(-1) (L797). GPU row reduction.
#   5. nldiff_stratified_select    — stratified confidence policy (L815-821):
#      unmask_order[start:start+num_transfer], start = tokens committed so far.
#      NOTE: the unmask_order table itself (a fixed PMJ permutation of the 64x64
#      grid from _stratified_random(n=64,seed=42,shuffle_blocks=True)) is LOADED
#      from the oracle; regenerating _stratified_random in Mojo is an optional
#      later follow-up (documented, not implemented here).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import isinf
from max.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ─────────────────────────────────────────────────────────────────────────────
# 1. num_transfer_tokens — _get_num_transfer_tokens (L206-233)
#
# Reference (mask_num == n_mask, a single batch row):
#   steps  = int(min(steps, n_mask))
#   t      = linspace(0, 1, steps+1)                 # torch float32
#   sigmas = shift * t / (1 + (shift-1) * t)         # _logit_normal_schedule
#   s      = (sigmas * n_mask).to(int64)             # truncate toward zero
#   s      = s[1:] - s[:-1]                           # per-step deltas
#   s      = clamp(s, 1, None)                        # each step >= 1
#   delta  = s.sum() - n_mask                         # overflow from clamping
#   # redistribute overflow: walk decrementing entries that are > 1
#   j = 0
#   while delta > 0:
#       j = j % steps
#       if s[j] == 1: j += 1; continue
#       delta -= 1; s[j] -= 1; j += 1
#   return s.flip(-1)                                 # returns int[steps], sum==n_mask
#
# Float math mirrors torch float32 EXACTLY: t[i] = i/steps is exact in f32 when
# steps is a power of two (64 -> exact), n_mask=4096=2^12 scaling is exact, so
# the truncation boundary matches torch bit-for-bit.
# ─────────────────────────────────────────────────────────────────────────────
def nldiff_num_transfer_tokens(
    n_mask: Int, n_steps: Int, shift: Int
) raises -> List[Int]:
    """Per-step masked-token budget (shift/logit-normal schedule). Returns an
    int list of length `steps = min(n_steps, n_mask)` summing to `n_mask`,
    time-reversed (flip). Exact match to `_get_num_transfer_tokens`."""
    var steps = n_steps
    if steps > n_mask:
        steps = n_mask
    if steps <= 0:
        raise Error("nldiff_num_transfer_tokens: steps must be > 0")

    var shift_f = Float32(shift)
    var nmask_f = Float32(n_mask)

    # sigmas*(n_mask) truncated to int, over the steps+1 grid points.
    var s_int = List[Int]()
    for i in range(steps + 1):
        var t = Float32(i) / Float32(steps)
        var sigma = shift_f * t / (Float32(1.0) + (shift_f - Float32(1.0)) * t)
        s_int.append(Int(sigma * nmask_f))  # truncates toward zero (>= 0)

    # per-step deltas, clamped to >= 1
    var s = List[Int]()
    var total = 0
    for i in range(steps):
        var d = s_int[i + 1] - s_int[i]
        if d < 1:
            d = 1
        s.append(d)
        total += d

    # redistribute overflow so the budget sums to exactly n_mask
    var delta = total - n_mask
    if delta < 0:
        raise Error("nldiff_num_transfer_tokens: negative delta (invalid schedule)")
    var j = 0
    var guard = 0
    var guard_max = (steps + 1) * (n_mask + 1)
    while delta > 0:
        j = j % steps
        if s[j] == 1:
            j += 1
            guard += 1
            if guard > guard_max:
                raise Error("nldiff_num_transfer_tokens: redistribution did not converge")
            continue
        delta -= 1
        s[j] -= 1
        j += 1

    # verify + flip(-1)
    var check = 0
    for i in range(steps):
        check += s[i]
    if check != n_mask:
        raise Error("nldiff_num_transfer_tokens: budget sum != n_mask")
    var out = List[Int]()
    for i in range(steps - 1, -1, -1):
        out.append(s[i])
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# 2. temp_schedule_linear — (L671-673)
#   sch_t = np.linspace(0, 1, n_steps)               # numpy float64
#   temps = (1 - sch_t) * (1 - min_temp) + min_temp
# Computed in Float64 to match the numpy float64 oracle (cos gate, not exact).
# ─────────────────────────────────────────────────────────────────────────────
def nldiff_temp_schedule_linear(
    n_steps: Int, min_temp: Float64 = 0.01
) raises -> List[Float32]:
    """Linear per-step temperature schedule, length n_steps, from 1.0 down to
    min_temp. Mirrors the schedule_temp=='linear' branch."""
    if n_steps <= 0:
        raise Error("nldiff_temp_schedule_linear: n_steps must be > 0")
    var out = List[Float32]()
    for i in range(n_steps):
        var t: Float64
        if n_steps == 1:
            t = 0.0
        else:
            t = Float64(i) / Float64(n_steps - 1)
        var v = (1.0 - t) * (1.0 - min_temp) + min_temp
        out.append(Float32(v))
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# 3. cfg_combine — (L772-774)
#   logits_is_ninf = (cond == -inf)
#   logits = (1 + gs) * cond - gs * uncond
#   logits[logits_is_ninf] = -inf                    # avoid inf-inf -> nan
# GPU elementwise, one thread per element. -inf is preserved by re-using `cond`
# itself (which is -inf) rather than a literal, so no inf-inf nan can appear.
# (For the deterministic gate there are no -inf logits — top_k/top_p unused.)
# ─────────────────────────────────────────────────────────────────────────────
def _cfg_kernel[dtype: DType](
    cond: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    uncond: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n_w: Int64,
    gs: Float32,
):
    var n = Int(n_w)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var c = rebind[Scalar[dtype]](cond[i]).cast[DType.float32]()
    var u = rebind[Scalar[dtype]](uncond[i]).cast[DType.float32]()
    var r = (Float32(1.0) + gs) * c - gs * u
    if isinf(c) and c < Float32(0.0):
        r = c  # -inf preserved (avoids inf - inf = nan)
    o[i] = rebind[o.element_type](r.cast[dtype]())


def nldiff_cfg_combine(
    cond: Tensor, uncond: Tensor, gs: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Classifier-free-guidance logit combine: (1+gs)*cond - gs*uncond,
    elementwise, preserving -inf where cond is -inf. cond/uncond must share
    shape+dtype; output is a fresh Tensor of the same shape+dtype."""
    var shp = cond.shape()
    var ushp = uncond.shape()
    if len(shp) != len(ushp):
        raise Error("nldiff_cfg_combine: rank mismatch")
    for i in range(len(shp)):
        if shp[i] != ushp[i]:
            raise Error("nldiff_cfg_combine: shape mismatch")
    if cond.dtype() != uncond.dtype():
        raise Error("nldiff_cfg_combine: dtype mismatch")

    var n = cond.numel()
    var dt = cond.dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * dt.byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var mdt = dt.to_mojo_dtype()

    if mdt == DType.float32:
        var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(cond.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        var U = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(uncond.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_cfg_kernel[DType.float32]](C, U, O, Int64(n), gs, grid_dim=grid, block_dim=_BLOCK)
    elif mdt == DType.bfloat16:
        var C = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(cond.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        var U = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(uncond.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_cfg_kernel[DType.bfloat16]](C, U, O, Int64(n), gs, grid_dim=grid, block_dim=_BLOCK)
    else:
        var C = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(cond.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        var U = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(uncond.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl,
    )
        ctx.enqueue_function[_cfg_kernel[DType.float16]](C, U, O, Int64(n), gs, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    return Tensor(out_buf^, shp^, dt)


# ─────────────────────────────────────────────────────────────────────────────
# 4. argmax over last dim — logits.argmax(-1) (L797)
# GPU row reduction: one thread per row scans V classes, ties -> FIRST index
# (matches torch.argmax). Returns host token ids (List[Int]) for the sampler
# grid logic; the reduction itself runs on GPU (proves argmax over 131072).
# ─────────────────────────────────────────────────────────────────────────────
def _argmax_row_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    n_rows_w: Int32,
    v_w: Int32,
):
    var n_rows = Int(n_rows_w)
    var v = Int(v_w)
    var r = Int(global_idx.x)
    if r >= n_rows:
        return
    var base = r * v
    var best = rebind[Scalar[dtype]](x[base]).cast[DType.float32]()
    var best_i = 0
    for i in range(1, v):
        var val = rebind[Scalar[dtype]](x[base + i]).cast[DType.float32]()
        if val > best:  # strict > keeps FIRST max (torch semantics)
            best = val
            best_i = i
    o[r] = Int32(best_i)


def nldiff_argmax_lastdim(logits: Tensor, ctx: DeviceContext) raises -> List[Int]:
    """Argmax over the last dim of `logits` -> one token id per row. Rows =
    numel / last_dim. Ties resolve to the FIRST index (torch.argmax)."""
    var shp = logits.shape()
    if len(shp) < 1:
        raise Error("nldiff_argmax_lastdim: needs rank >= 1")
    var v = shp[len(shp) - 1]
    if v <= 0:
        raise Error("nldiff_argmax_lastdim: empty last dim")
    var n_rows = logits.numel() // v

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n_rows * 4)
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](logits.numel()))
    var rl_out = RuntimeLayout[_DYN1].row_major(IndexList[1](n_rows))
    var grid = (n_rows + _BLOCK - 1) // _BLOCK
    var mdt = logits.dtype().to_mojo_dtype()

    var O = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Int32]())
        ),
        runtime_layout=rl_out,
    )
    if mdt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(logits.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl_in,
    )
        ctx.enqueue_function[_argmax_row_kernel[DType.float32]](X, O, Int32(n_rows), Int32(v), grid_dim=grid, block_dim=_BLOCK)
    elif mdt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(logits.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=rl_in,
    )
        ctx.enqueue_function[_argmax_row_kernel[DType.bfloat16]](X, O, Int32(n_rows), Int32(v), grid_dim=grid, block_dim=_BLOCK)
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float16], MutAnyOrigin](
            unsafe_from_address=Int(logits.buf.unsafe_ptr().bitcast[Float16]())
        ),
        runtime_layout=rl_in,
    )
        ctx.enqueue_function[_argmax_row_kernel[DType.float16]](X, O, Int32(n_rows), Int32(v), grid_dim=grid, block_dim=_BLOCK)

    var host = ctx.enqueue_create_host_buffer[DType.uint8](n_rows * 4)
    ctx.enqueue_copy(dst_buf=host, src_buf=out_buf)
    ctx.synchronize()
    var ip = host.unsafe_ptr().bitcast[Int32]()
    var out = List[Int]()
    for r in range(n_rows):
        out.append(Int(ip[r]))
    _ = out_buf^
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# 5. stratified select — confidence_policy=='stratified' (L815-821)
#   start = n_tokens - n_mask   (== tokens committed so far)
#   select_index = unmask_order[start : start + num_transfer]
# The unmask_order table is a fixed permutation (PMJ / _stratified_random,
# seed=42) LOADED from the oracle; we only slice it here.
# ─────────────────────────────────────────────────────────────────────────────
def nldiff_stratified_select(
    unmask_order: List[Int], start: Int, num_transfer: Int
) raises -> List[Int]:
    """The next `num_transfer` grid indices to commit: the contiguous slice
    unmask_order[start:start+num_transfer]. `start` == tokens committed so far
    (== n_tokens - n_mask). Deterministic — no RNG."""
    if start < 0 or num_transfer < 0:
        raise Error("nldiff_stratified_select: negative start/num_transfer")
    if start + num_transfer > len(unmask_order):
        raise Error("nldiff_stratified_select: slice out of range of unmask_order")
    var out = List[Int]()
    for i in range(start, start + num_transfer):
        out.append(unmask_order[i])
    return out^
