# training/adamw8bit.mojo — host-math 8-bit AdamW, bitsandbytes 0.49.2
# block-wise parity (T2.A).
#
# Reference (the proven Rust-stack oracle chain):
#   * algorithm: /home/alex/EriDiffusion/flame-core/src/adam8bit_kernel.rs —
#     pure port of the on-device math of bnb `optim.AdamW8bit`
#     (block_wise=True), byte-parity-gated against bnb 0.49.2 by
#     crates/eridiffusion-cli/src/bin/parity_adam8bit_bnb{,_wd,_tail,
#     _bf16grad,_multistep}.rs.
#   * oracle dumps: /home/alex/EriDiffusion/EriDiffusion-v2/tests/parity/
#     adam8bit_data/ — bnb 0.49.2 F.optimizer_update_8bit_blockwise("adam",
#     ...) before/after snapshots produced by tests/parity/
#     adam8bit_bnb_python_ref*.py (bnb 0.49.2 + torch 2.10.0+cu128).
#
# State layout per parameter (n elements):
#   m_codes  [n]            u8 codes into the SIGNED dynamic LUT (qmap1)
#   v_codes  [n]            u8 codes into the UNSIGNED dynamic LUT (qmap2)
#   m_absmax [ceil(n/256)]  per-256-block F32 scale for m
#   v_absmax [ceil(n/256)]  per-256-block F32 scale for v
# Dequant of element i: qmap[code[i]] * absmax[i / 256]. Block size is
# HARDCODED to 256 in bnb (optim/optimizer.py:478) — do not change.
#
# Per-step math (bnb "adam" with weight_decay — bnb's C++ blockwise kernel
# applies wd DECOUPLED, proven by the Rust _wd parity bin):
#   m_old = qmap_s[m_code] * m_absmax[blk]
#   v_old = qmap_u[v_code] * v_absmax[blk]
#   m_new = beta1*m_old + (1-beta1)*g
#   v_new = beta2*v_old + (1-beta2)*g*g
#   p    -= lr * (m_new/bc1) / (sqrt(v_new/bc2) + eps)     bc = 1 - beta^t
#   if wd != 0:  p -= lr*wd*p                              (decoupled, AFTER)
#   absmax' = max_block |new|  (fallback 1e-12 if a block is all-zero)
#   code'   = argmin_c |qmap[c] - new/absmax'|             (first-wins ties)
# ALL element math is Float32 (the bnb kernel is f32) — measured on the bnb
# dumps (adamw8bit_parity.mojo, 2026-06-11): codes EXACT-EQUAL on every f32
# scenario incl. all 10 multistep steps; bf16grad has 2 signed-code
# mismatches (LUT-tiebreak boundary cases, inside the Rust ref bin's own
# <=5 allowance); param max|Δ| <= 2.4e-7 (~1.5 f32 ulp of the updated
# params, FAR below one 8-bit quantum); absmax max|Δ| = 0.0 (bit-exact).
#
# LUT (adam8bit_create_dynamic_map): port of bnb functional.py
# create_dynamic_map(signed, 7, 8). BIT-EXACT vs bnb's dump requires
# reproducing torch.linspace(0.1, 1, k, f32)'s CPU kernel exactly:
#   step = (1.0f - 0.1f) / f32(k-1)
#   j <  k//2 : fmaf(step,  f32(j),       0.1f)   (single-rounding FMA)
#   j >= k//2 : fmaf(-step, f32(k-1-j),   1.0f)   (filled BACK from `end`)
# then means = (b[j]+b[j+1])/2 in f32, scaled by the f32 power of ten.
# (Empirically verified bit-exact against before.qmap1/qmap2 — plain
# `0.1+j*step` in either f32 or f64 is NOT bit-exact at k>=17.)
#
# Host-math first (List[Float32] params/grads, the adafactor.mojo T1.C
# pattern); a fused GPU kernel can land later behind the same levers
# dispatch. bf16 grads: upcast bf16->f32 BEFORE calling (bnb's host dispatch
# does g.float(); the upcast is exact).
#
# Parity gate: training/tests/adamw8bit_parity.mojo vs the bnb dumps
# (5 scenarios: basic, weight-decay, tail block, bf16 grads, 10-step).
#
# Mojo 1.0.0b1.

from std.math import fma, sqrt
from std.memory import ArcPointer, stack_allocation
from std.gpu import thread_idx, block_idx
from max.gpu import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import full_device

comptime ADAMW8BIT_BLOCK_SIZE = 256
"""bnb block size for the blockwise 8-bit optimizer state
(optim/optimizer.py:478). Hardcoded in bnb's kernels; do not change."""


def _linspace_f32_torch(k: Int) -> List[Float32]:
    """torch.linspace(0.1, 1, k, dtype=f32) CPU-kernel bit-exact:
    f32 step, FMA fill, second half filled backwards from `end`."""
    var out = List[Float32](capacity=k)
    var step = (Float32(1.0) - Float32(0.1)) / Float32(k - 1)
    var half = k // 2
    for j in range(k):
        if j < half:
            out.append(fma(step, Float32(j), Float32(0.1)))
        else:
            out.append(fma(-step, Float32(k - 1 - j), Float32(1.0)))
    return out^


def adam8bit_create_dynamic_map(signed: Bool) raises -> List[Float32]:
    """256-entry dynamic-exponent qmap, bnb 0.49.2 create_dynamic_map(signed,
    max_exponent_bits=7, total_bits=8). signed=True -> the m LUT ("dynamic"),
    signed=False -> the v LUT ("udynamic"). Sorted ascending; bit-exact vs
    bnb's torch output (gated in adamw8bit_parity.mojo at 0.0 max|Δ|)."""
    # f32 powers of ten 1e-6..1e0 (literal rounding == torch's f32 cast of
    # the python float 10**(-6+i)).
    var scales = List[Float32]()
    scales.append(Float32(1.0e-6))
    scales.append(Float32(1.0e-5))
    scales.append(Float32(1.0e-4))
    scales.append(Float32(1.0e-3))
    scales.append(Float32(1.0e-2))
    scales.append(Float32(1.0e-1))
    scales.append(Float32(1.0))

    var data = List[Float32]()
    for i in range(7):
        var fraction_items: Int
        if signed:
            fraction_items = (1 << i) + 1
        else:
            fraction_items = (1 << (i + 1)) + 1
        var bd = _linspace_f32_torch(fraction_items)
        var scale = scales[i]
        for j in range(fraction_items - 1):
            var mean = (bd[j] + bd[j + 1]) / Float32(2.0)
            data.append(scale * mean)
        if signed:
            for j in range(fraction_items - 1):
                var mean = (bd[j] + bd[j + 1]) / Float32(2.0)
                data.append(-(scale * mean))
    data.append(Float32(0.0))
    data.append(Float32(1.0))
    if len(data) != 256:
        # bnb asserts len == 2**total_bits with these constants (254 means
        # + 0 + 1 for both signed and unsigned).
        raise Error(
            String("adam8bit_create_dynamic_map: ")
            + String(len(data))
            + String(" entries != 256")
        )
    # Ascending insertion sort (256 entries; avoids stdlib sort API churn).
    for i in range(1, 256):
        var v = data[i]
        var j = i - 1
        while j >= 0 and data[j] > v:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = v
    return data^


struct Adam8bitState(Copyable, Movable):
    """Per-parameter block-wise 8-bit AdamW moment state. Zero-init matches
    bnb Optimizer8bit.init_state (optim/optimizer.py:497-519): all codes 0,
    all absmax 0.0 -> initial dequant is exactly 0 regardless of LUT."""

    var m_codes: List[UInt8]
    var v_codes: List[UInt8]
    var m_absmax: List[Float32]
    var v_absmax: List[Float32]
    var n: Int
    var step: Int  # completed steps (0 before the first step; bnb is 1-based)

    def __init__(out self, n: Int):
        self.m_codes = List[UInt8](capacity=n)
        self.v_codes = List[UInt8](capacity=n)
        for _ in range(n):
            self.m_codes.append(UInt8(0))
            self.v_codes.append(UInt8(0))
        var blocks = (n + ADAMW8BIT_BLOCK_SIZE - 1) // ADAMW8BIT_BLOCK_SIZE
        self.m_absmax = List[Float32](capacity=blocks)
        self.v_absmax = List[Float32](capacity=blocks)
        for _ in range(blocks):
            self.m_absmax.append(Float32(0.0))
            self.v_absmax.append(Float32(0.0))
        self.n = n
        self.step = 0


def _pow_f32(base: Float32, e: Int) -> Float32:
    # f32 repeated multiply (bias correction beta^t; t small, 1-ulp class
    # differences vs powi are ~1e-11 on the param — far inside the gate).
    var out = Float32(1.0)
    for _ in range(e):
        out = out * base
    return out


def adam8bit_step_bnb(
    mut p: List[Float32],
    g: List[Float32],
    mut state: Adam8bitState,
    qmap_signed: List[Float32],
    qmap_unsigned: List[Float32],
    step: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
) raises:
    """One bnb-parity blockwise 8-bit AdamW step at 1-based `step` (the bias
    correction exponent). Mutates p and state in place; does NOT touch
    state.step (the gate drives arbitrary before-states; trainers use
    adamw8bit_step below). g must be F32 (upcast bf16 grads BEFORE calling
    — exact, mirrors bnb's host g.float())."""
    var n = state.n
    if len(p) != n or len(g) != n:
        raise Error("adam8bit_step_bnb: p/g length != state.n")
    if len(qmap_signed) != 256 or len(qmap_unsigned) != 256:
        raise Error("adam8bit_step_bnb: qmap length != 256")
    if step < 1:
        raise Error("adam8bit_step_bnb: step must be >= 1")
    var n_blocks = (n + ADAMW8BIT_BLOCK_SIZE - 1) // ADAMW8BIT_BLOCK_SIZE
    if len(state.m_absmax) < n_blocks or len(state.v_absmax) < n_blocks:
        raise Error("adam8bit_step_bnb: absmax buffers too small")

    var bc1 = Float32(1.0) - _pow_f32(beta1, step)
    var bc2 = Float32(1.0) - _pow_f32(beta2, step)
    var one_m_b1 = Float32(1.0) - beta1
    var one_m_b2 = Float32(1.0) - beta2

    var m_new = List[Float32](capacity=ADAMW8BIT_BLOCK_SIZE)
    var v_new = List[Float32](capacity=ADAMW8BIT_BLOCK_SIZE)
    for _ in range(ADAMW8BIT_BLOCK_SIZE):
        m_new.append(Float32(0.0))
        v_new.append(Float32(0.0))

    for blk in range(n_blocks):
        var base = blk * ADAMW8BIT_BLOCK_SIZE
        var cnt = min(ADAMW8BIT_BLOCK_SIZE, n - base)
        var am_prev = state.m_absmax[blk]
        var av_prev = state.v_absmax[blk]

        # Pass 1: dequant, AdamW math, param update, stash new moments.
        for t in range(cnt):
            var i = base + t
            var gv = g[i]
            var m_old = qmap_signed[Int(state.m_codes[i])] * am_prev
            var v_old = qmap_unsigned[Int(state.v_codes[i])] * av_prev
            var mn = beta1 * m_old + one_m_b1 * gv
            var vn = beta2 * v_old + one_m_b2 * gv * gv
            m_new[t] = mn
            v_new[t] = vn
            var m_hat = mn / bc1
            var v_hat = vn / bc2
            var upd = lr * m_hat / (sqrt(v_hat) + eps)
            var pv = p[i] - upd
            if weight_decay != Float32(0.0):
                pv = pv - lr * weight_decay * pv  # decoupled, AFTER the update
            p[i] = pv

        # Block max-abs reduction (max is order-independent — exact).
        var amm = Float32(0.0)
        var amv = Float32(0.0)
        for t in range(cnt):
            var am = m_new[t] if m_new[t] >= Float32(0.0) else -m_new[t]
            var av = v_new[t] if v_new[t] >= Float32(0.0) else -v_new[t]
            if am > amm:
                amm = am
            if av > amv:
                amv = av
        if amm == Float32(0.0):
            amm = Float32(1.0e-12)  # all-zero-block guard (kernel parity)
        if amv == Float32(0.0):
            amv = Float32(1.0e-12)
        state.m_absmax[blk] = amm
        state.v_absmax[blk] = amv

        # Pass 2: requant — linear argmin scan, strict < (first/lowest code
        # wins ties — the kernel's tiebreak).
        for t in range(cnt):
            var i = base + t
            var m_norm = m_new[t] / amm
            var v_norm = v_new[t] / amv
            var best_m = 0
            var d0m = qmap_signed[0] - m_norm
            var best_m_d = d0m if d0m >= Float32(0.0) else -d0m
            var best_v = 0
            var d0v = qmap_unsigned[0] - v_norm
            var best_v_d = d0v if d0v >= Float32(0.0) else -d0v
            for c in range(1, 256):
                var dm = qmap_signed[c] - m_norm
                if dm < Float32(0.0):
                    dm = -dm
                if dm < best_m_d:
                    best_m_d = dm
                    best_m = c
                var dv = qmap_unsigned[c] - v_norm
                if dv < Float32(0.0):
                    dv = -dv
                if dv < best_v_d:
                    best_v_d = dv
                    best_v = c
            state.m_codes[i] = UInt8(best_m)
            state.v_codes[i] = UInt8(best_v)


def adamw8bit_step(
    mut p: List[Float32],
    g: List[Float32],
    mut state: Adam8bitState,
    qmap_signed: List[Float32],
    qmap_unsigned: List[Float32],
    k: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
) raises:
    """Trainer entry: one step at trainer step k (1-based, the same t the
    AdamW path passes). Fails loud on a step-count desync (no save/resume
    sidecar yet — the levers contract)."""
    if state.step != k - 1:
        raise Error(
            String("adamw8bit_step: step desync (state.step=")
            + String(state.step)
            + String(", trainer step=")
            + String(k)
            + String(")")
        )
    adam8bit_step_bnb(
        p, g, state, qmap_signed, qmap_unsigned, k,
        lr, beta1, beta2, eps, weight_decay,
    )
    state.step = k


# ---------------------------------------------------------------------------
# Device-resident multi-tensor AdamW8bit
# ---------------------------------------------------------------------------
#
# The original API above remains the byte-parity host oracle. Production H3
# has roughly 140M rank-32 adapter parameters, so downloading those gradients
# and running the 256-code search on the CPU would dominate every step. This
# sibling executes the same bnb blockwise algorithm on the GPU while keeping
# the quantized moments resident. Parameter tensors remain separate; the
# optimizer state is one padded flat slab. Each parameter starts on a fresh
# 256-element block, exactly preserving bnb's per-parameter block boundary.

comptime _TArc = ArcPointer[Tensor]
comptime _DYN1 = Layout.row_major(-1)


struct Adam8bitDeviceState(Copyable, Movable):
    var m_codes: _TArc       # U8 [total padded parameter elements]
    var v_codes: _TArc       # U8 [total padded parameter elements]
    var m_absmax: _TArc      # F32 [total 256-element blocks]
    var v_absmax: _TArc      # F32 [total 256-element blocks]
    var qmap_signed: _TArc   # F32 [256]
    var qmap_unsigned: _TArc # F32 [256]
    var total_padded: Int
    var total_blocks: Int

    def __init__(
        out self,
        var m_codes: _TArc,
        var v_codes: _TArc,
        var m_absmax: _TArc,
        var v_absmax: _TArc,
        var qmap_signed: _TArc,
        var qmap_unsigned: _TArc,
        total_padded: Int,
        total_blocks: Int,
    ):
        self.m_codes = m_codes^
        self.v_codes = v_codes^
        self.m_absmax = m_absmax^
        self.v_absmax = v_absmax^
        self.qmap_signed = qmap_signed^
        self.qmap_unsigned = qmap_unsigned^
        self.total_padded = total_padded
        self.total_blocks = total_blocks


def _zero_u8_tensor(n: Int, ctx: DeviceContext) raises -> Tensor:
    if n <= 0:
        raise Error("AdamW8bit device state must contain at least one byte")
    var buf = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_memset(buf, UInt8(0))
    var shape: List[Int] = [n]
    return Tensor(buf^, shape^, STDtype.U8)


def _adam8bit_total_blocks(params: List[_TArc]) raises -> Int:
    var total = 0
    for i in range(len(params)):
        var n = params[i][].numel()
        if n <= 0:
            raise Error("AdamW8bit device parameter tensor is empty")
        total += (n + ADAMW8BIT_BLOCK_SIZE - 1) // ADAMW8BIT_BLOCK_SIZE
    return total


def adamw8bit_device_state(
    params: List[_TArc], ctx: DeviceContext
) raises -> Adam8bitDeviceState:
    var blocks = _adam8bit_total_blocks(params)
    var padded = blocks * ADAMW8BIT_BLOCK_SIZE
    var m_codes = _zero_u8_tensor(padded, ctx)
    var v_codes = _zero_u8_tensor(padded, ctx)
    var bshape: List[Int] = [blocks]
    var m_absmax = full_device(bshape.copy(), Float32(0.0), STDtype.F32, ctx)
    var v_absmax = full_device(bshape.copy(), Float32(0.0), STDtype.F32, ctx)
    var qshape: List[Int] = [256]
    var q_signed = Tensor.from_host(
        adam8bit_create_dynamic_map(True), qshape.copy(), STDtype.F32, ctx
    )
    var q_unsigned = Tensor.from_host(
        adam8bit_create_dynamic_map(False), qshape^, STDtype.F32, ctx
    )
    return Adam8bitDeviceState(
        _TArc(m_codes^), _TArc(v_codes^),
        _TArc(m_absmax^), _TArc(v_absmax^),
        _TArc(q_signed^), _TArc(q_unsigned^), padded, blocks,
    )


def adamw8bit_device_state_from_tensors(
    params: List[_TArc],
    var m_codes: _TArc,
    var v_codes: _TArc,
    var m_absmax: _TArc,
    var v_absmax: _TArc,
    ctx: DeviceContext,
) raises -> Adam8bitDeviceState:
    var blocks = _adam8bit_total_blocks(params)
    var padded = blocks * ADAMW8BIT_BLOCK_SIZE
    if m_codes[].dtype() != STDtype.U8 or v_codes[].dtype() != STDtype.U8:
        raise Error("AdamW8bit device code tensors must be U8")
    if m_absmax[].dtype() != STDtype.F32 or v_absmax[].dtype() != STDtype.F32:
        raise Error("AdamW8bit device absmax tensors must be F32")
    if m_codes[].numel() != padded or v_codes[].numel() != padded:
        raise Error("AdamW8bit device code-state size mismatch")
    if m_absmax[].numel() != blocks or v_absmax[].numel() != blocks:
        raise Error("AdamW8bit device absmax-state size mismatch")
    var qshape: List[Int] = [256]
    var q_signed = Tensor.from_host(
        adam8bit_create_dynamic_map(True), qshape.copy(), STDtype.F32, ctx
    )
    var q_unsigned = Tensor.from_host(
        adam8bit_create_dynamic_map(False), qshape^, STDtype.F32, ctx
    )
    return Adam8bitDeviceState(
        m_codes^, v_codes^, m_absmax^, v_absmax^,
        _TArc(q_signed^), _TArc(q_unsigned^), padded, blocks,
    )


def _adamw8bit_device_kernel[g_dtype: DType](
    p_addr: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    g_addr: LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin],
    param_numel: LayoutTensor[DType.int64, _DYN1, MutAnyOrigin],
    block_offsets: LayoutTensor[DType.int64, _DYN1, MutAnyOrigin],
    m_codes: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    v_codes: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    m_absmax: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    v_absmax: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    qmap_signed: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    qmap_unsigned: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ntensors_w: Int32,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    bc1: Float32,
    bc2: Float32,
    clip_scale: Float32,
):
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var ntensors = Int(ntensors_w)
    var sh_signed = stack_allocation[
        ADAMW8BIT_BLOCK_SIZE, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var sh_unsigned = stack_allocation[
        ADAMW8BIT_BLOCK_SIZE, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var sh_m = stack_allocation[
        ADAMW8BIT_BLOCK_SIZE, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var sh_v = stack_allocation[
        ADAMW8BIT_BLOCK_SIZE, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var sh_tensor = stack_allocation[
        1, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()

    sh_signed[tid] = rebind[Scalar[DType.float32]](qmap_signed[tid])
    sh_unsigned[tid] = rebind[Scalar[DType.float32]](qmap_unsigned[tid])
    if tid == 0:
        var ti = 0
        while ti + 1 < ntensors and Int(rebind[Scalar[DType.int64]](block_offsets[ti + 1])) <= bid:
            ti += 1
        sh_tensor[0] = Int32(ti)
    barrier()

    var ti = Int(sh_tensor[0])
    var local_block = bid - Int(rebind[Scalar[DType.int64]](block_offsets[ti]))
    var local_idx = local_block * ADAMW8BIT_BLOCK_SIZE + tid
    var n = Int(rebind[Scalar[DType.int64]](param_numel[ti]))
    var active = local_idx < n
    var state_idx = bid * ADAMW8BIT_BLOCK_SIZE + tid

    var pa = rebind[Scalar[DType.uint64]](p_addr[ti])
    var ga = rebind[Scalar[DType.uint64]](g_addr[ti])
    var pp = UnsafePointer[Scalar[DType.float32], MutExternalOrigin](
        unsafe_from_address=Int(pa)
    )
    var gp = UnsafePointer[Scalar[g_dtype], MutExternalOrigin](
        unsafe_from_address=Int(ga)
    )

    var gv = Float32(0.0)
    var pv = Float32(0.0)
    var m_old = Float32(0.0)
    var v_old = Float32(0.0)
    if active:
        gv = gp[local_idx].cast[DType.float32]() * clip_scale
        pv = pp[local_idx].cast[DType.float32]()
        var mc = Int(rebind[Scalar[DType.uint8]](m_codes[state_idx]))
        var vc = Int(rebind[Scalar[DType.uint8]](v_codes[state_idx]))
        m_old = sh_signed[mc] * rebind[Scalar[DType.float32]](m_absmax[bid])
        v_old = sh_unsigned[vc] * rebind[Scalar[DType.float32]](v_absmax[bid])

    var m_new = beta1 * m_old + (Float32(1.0) - beta1) * gv
    var v_new = beta2 * v_old + (Float32(1.0) - beta2) * gv * gv
    if active:
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        pv -= lr * m_hat / (sqrt(v_hat) + eps)
        if weight_decay != Float32(0.0):
            pv -= lr * weight_decay * pv
        pp[local_idx] = pv

    var am = m_new if m_new >= Float32(0.0) else -m_new
    var av = v_new if v_new >= Float32(0.0) else -v_new
    sh_m[tid] = am if active else Float32(0.0)
    sh_v[tid] = av if active else Float32(0.0)
    barrier()
    var width = ADAMW8BIT_BLOCK_SIZE // 2
    while width > 0:
        if tid < width:
            if sh_m[tid + width] > sh_m[tid]:
                sh_m[tid] = sh_m[tid + width]
            if sh_v[tid + width] > sh_v[tid]:
                sh_v[tid] = sh_v[tid + width]
        barrier()
        width //= 2

    var new_m_abs = sh_m[0]
    var new_v_abs = sh_v[0]
    if new_m_abs == Float32(0.0):
        new_m_abs = Float32(1.0e-12)
    if new_v_abs == Float32(0.0):
        new_v_abs = Float32(1.0e-12)
    if tid == 0:
        m_absmax[bid] = new_m_abs
        v_absmax[bid] = new_v_abs
    barrier()

    if active:
        var m_norm = m_new / new_m_abs
        var v_norm = v_new / new_v_abs
        var best_m = 0
        var dm0 = sh_signed[0] - m_norm
        var best_dm = dm0 if dm0 >= Float32(0.0) else -dm0
        var best_v = 0
        var dv0 = sh_unsigned[0] - v_norm
        var best_dv = dv0 if dv0 >= Float32(0.0) else -dv0
        for code in range(1, 256):
            var dm = sh_signed[code] - m_norm
            if dm < Float32(0.0):
                dm = -dm
            if dm < best_dm:
                best_dm = dm
                best_m = code
            var dv = sh_unsigned[code] - v_norm
            if dv < Float32(0.0):
                dv = -dv
            if dv < best_dv:
                best_dv = dv
                best_v = code
        m_codes[state_idx] = UInt8(best_m)
        v_codes[state_idx] = UInt8(best_v)


def adamw8bit_device_step(
    params: List[_TArc],
    grads: List[_TArc],
    state: Adam8bitDeviceState,
    step: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
    clip_scale: Float32 = Float32(1.0),
) raises:
    var nt = len(params)
    if nt == 0 or len(grads) != nt:
        raise Error("AdamW8bit device params/grads must be non-empty and matched")
    if step < 1:
        raise Error("AdamW8bit device step must be >= 1")
    var blocks = _adam8bit_total_blocks(params)
    if blocks != state.total_blocks:
        raise Error("AdamW8bit device state no longer matches parameter geometry")
    var grad_dtype = grads[0][].dtype()
    if grad_dtype != STDtype.F32 and grad_dtype != STDtype.BF16 and grad_dtype != STDtype.F16:
        raise Error("AdamW8bit device gradients must be F32, BF16, or F16")

    var p_host = ctx.enqueue_create_host_buffer[DType.uint8](nt * 8)
    var g_host = ctx.enqueue_create_host_buffer[DType.uint8](nt * 8)
    var n_host = ctx.enqueue_create_host_buffer[DType.uint8](nt * 8)
    var bo_host = ctx.enqueue_create_host_buffer[DType.uint8]((nt + 1) * 8)
    var pp = p_host.unsafe_ptr().bitcast[UInt64]()
    var gp = g_host.unsafe_ptr().bitcast[UInt64]()
    var np = n_host.unsafe_ptr().bitcast[Int64]()
    var bp = bo_host.unsafe_ptr().bitcast[Int64]()
    var block_cursor = 0
    bp[0] = Int64(0)
    for i in range(nt):
        if params[i][].dtype() != STDtype.F32:
            raise Error("AdamW8bit device master parameters must be F32")
        if grads[i][].dtype() != grad_dtype:
            raise Error("AdamW8bit device gradients must share one dtype")
        var n = params[i][].numel()
        if grads[i][].numel() != n:
            raise Error("AdamW8bit device per-tensor param/grad mismatch")
        pp[i] = UInt64(Int(params[i][].buf.unsafe_ptr().bitcast[Float32]()))
        gp[i] = UInt64(Int(grads[i][].buf.unsafe_ptr()))
        np[i] = Int64(n)
        block_cursor += (n + ADAMW8BIT_BLOCK_SIZE - 1) // ADAMW8BIT_BLOCK_SIZE
        bp[i + 1] = Int64(block_cursor)

    var p_dev = ctx.enqueue_create_buffer[DType.uint8](nt * 8)
    var g_dev = ctx.enqueue_create_buffer[DType.uint8](nt * 8)
    var n_dev = ctx.enqueue_create_buffer[DType.uint8](nt * 8)
    var bo_dev = ctx.enqueue_create_buffer[DType.uint8]((nt + 1) * 8)
    ctx.enqueue_copy(dst_buf=p_dev, src_buf=p_host)
    ctx.enqueue_copy(dst_buf=g_dev, src_buf=g_host)
    ctx.enqueue_copy(dst_buf=n_dev, src_buf=n_host)
    ctx.enqueue_copy(dst_buf=bo_dev, src_buf=bo_host)

    var a_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nt))
    var bo_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nt + 1))
    var code_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](state.total_padded))
    var block_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](state.total_blocks))
    var q_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](256))
    var PA = LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint64], MutAnyOrigin](
            unsafe_from_address=Int(p_dev.unsafe_ptr().bitcast[UInt64]())
        ), runtime_layout=a_rl,
    )
    var GA = LayoutTensor[DType.uint64, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint64], MutAnyOrigin](
            unsafe_from_address=Int(g_dev.unsafe_ptr().bitcast[UInt64]())
        ), runtime_layout=a_rl,
    )
    var NN = LayoutTensor[DType.int64, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int64], MutAnyOrigin](
            unsafe_from_address=Int(n_dev.unsafe_ptr().bitcast[Int64]())
        ), runtime_layout=a_rl,
    )
    var BO = LayoutTensor[DType.int64, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int64], MutAnyOrigin](
            unsafe_from_address=Int(bo_dev.unsafe_ptr().bitcast[Int64]())
        ), runtime_layout=bo_rl,
    )
    var MC = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(state.m_codes[].buf.unsafe_ptr())
        ), runtime_layout=code_rl,
    )
    var VC = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(state.v_codes[].buf.unsafe_ptr())
        ), runtime_layout=code_rl,
    )
    var MA = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(state.m_absmax[].buf.unsafe_ptr().bitcast[Float32]())
        ), runtime_layout=block_rl,
    )
    var VA = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(state.v_absmax[].buf.unsafe_ptr().bitcast[Float32]())
        ), runtime_layout=block_rl,
    )
    var QS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(state.qmap_signed[].buf.unsafe_ptr().bitcast[Float32]())
        ), runtime_layout=q_rl,
    )
    var QU = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(state.qmap_unsigned[].buf.unsafe_ptr().bitcast[Float32]())
        ), runtime_layout=q_rl,
    )

    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(step):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    if grad_dtype == STDtype.F32:
        ctx.enqueue_function[_adamw8bit_device_kernel[DType.float32]](
            PA, GA, NN, BO, MC, VC, MA, VA, QS, QU, Int32(nt),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2, clip_scale,
            grid_dim=state.total_blocks, block_dim=ADAMW8BIT_BLOCK_SIZE,
        )
    elif grad_dtype == STDtype.BF16:
        ctx.enqueue_function[_adamw8bit_device_kernel[DType.bfloat16]](
            PA, GA, NN, BO, MC, VC, MA, VA, QS, QU, Int32(nt),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2, clip_scale,
            grid_dim=state.total_blocks, block_dim=ADAMW8BIT_BLOCK_SIZE,
        )
    else:
        ctx.enqueue_function[_adamw8bit_device_kernel[DType.float16]](
            PA, GA, NN, BO, MC, VC, MA, VA, QS, QU, Int32(nt),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2, clip_scale,
            grid_dim=state.total_blocks, block_dim=ADAMW8BIT_BLOCK_SIZE,
        )
