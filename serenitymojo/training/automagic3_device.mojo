# automagic3_device.mojo — GPU port of the factored-2D Automagic3 step.
#
# WHY: the levers host path (levers_optimizer_step_host -> automagic3_step_2d)
# runs the WHOLE optimizer on the CPU as List[Float32] loops over ~54M LoRA
# elements/step (~10s on krea2). ai-toolkit's automagic3 is a torch optimizer =
# GPU. This module moves the per-matrix math onto the device so automagic3 is
# fast, mirroring the fused_lora_adamw_plain_step pack->upload->kernel pattern.
#
# SCOPE: factored 2D only (LoRA A/B are always 2D — automagic3_step_1d is never
# constructed for LoRA). Bit-faithful target: the host automagic3 on identical
# inputs (gate: automagic3_device_parity). Reductions accumulate in Float64 to
# match the host's Float64 sums (the F32 oracle's rel<=4e-8 bar is unforgiving of
# reduction-order drift, so per-row/col sums are sequential in one thread, and the
# rmean/RMS block reductions use a single-thread serial pass for determinism).
#
# ONE BLOCK PER MATRIX. The block walks its matrix with grid-stride loops; the
# small serial reductions (rmean over rows, RMS over numel) run on thread 0 into
# shared memory. The group vote (sum w*up-w*down, sum w) atomic-adds two f64
# accumulators shared by ALL matrices -> the host reads them after and nudges the
# single shared lr (apply_vote). State (row_var/col_var + the H-plane sign ring +
# hist_idx/fill) is device-resident across steps.
#
# Build/gate: serenitymojo/training/parity/automagic3_device_parity.mojo

from std.gpu import thread_idx, block_idx, block_dim
from max.gpu import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.atomic import Atomic
from std.math import sqrt, exp
from std.memory import ArcPointer, stack_allocation
from std.time import perf_counter_ns
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

comptime _DYN1 = Layout.row_major(-1)   # dynamic-size flat 1D buffer layout


# Per-matrix descriptor packed host-side: offsets into the flat buffers + shape.
# rows = p.rows, cols = p.cols, numel = rows*cols. goff = grad/param flat offset,
# roff = row_var offset, coff = col_var offset, soff = sign-ring flat offset
# (H*numel bytes from this base), hidx/hfill = ring cursor (device-updated).
@fieldwise_init
struct A3MatDesc(Copyable, Movable):
    var rows: Int32
    var cols: Int32
    var goff: Int32   # flat element offset into p_f32 / g_f32 / update scratch
    var roff: Int32   # element offset into row_var
    var coff: Int32   # element offset into col_var
    var soff: Int32   # element offset into sign_ring (UInt8 per element per plane)


comptime _TPB = 256          # threads per block (one block per matrix)
comptime _A3_H = 8           # sign-history planes (controller window; cfg default)


# ─────────────────────────────────────────────────────────────────────────────
# The one fused per-matrix kernel. Phases separated by barrier():
#   1) sanitize grad (NaN/inf->0) into g scratch; per-row & per-col sq sums ->
#      EMA update row_var/col_var (one thread per row, one per col, serial inner).
#   2) thread 0: rmean = mean(row_var) (serial f64).  -> shared
#   3) update[i] = rsqrt(row_var[r]/rmean)*rsqrt(col_var[c])*g[i]   (per element)
#   4) thread 0: u_sq = sum(update^2) (serial f64) -> RMS -> inv_scale  -> shared
#   5) update[i] *= inv_scale ; clamp +/-clip ; record sign bit into the ring
#   6) thread 0: if window full, walk the ring per element, accumulate the matrix
#      vote into the GROUP f64 atomics (num,den).
#   7) update[i] += wd*p[i] ; p[i] -= lr*update[i]   (lr passed in; host-adapted)
# SR writeback bf16 is a SEPARATE concern handled by the existing
# automagic3_writeback_bf16_sr on the host after download (keeps the verified SR
# path unchanged); this kernel produces the F32 master update only.
def automagic3_factored_kernel(
    p: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # flat f32 master params (in/out)
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # flat f32 grads (in)
    upd: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], # flat f32 update scratch
    row_var: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    col_var: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    sign_ring: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin], # H planes * numel
    descs: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],     # 6 int32 per matrix
    hist_idx: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],  # per matrix
    hist_fill: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin], # per matrix
    group_num: LayoutTensor[DType.float64, _DYN1, MutAnyOrigin], # 1 elt (group)
    group_den: LayoutTensor[DType.float64, _DYN1, MutAnyOrigin], # 1 elt
    pb: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],       # bf16 a/b out (device SR)
    beta2: Float32,
    one_minus_b2: Float32,
    eps: Float32,
    clip: Float32,
    lr: Float32,
    weight_decay: Float32,
    grad_scale: Float32,
    step_w: Int32,
    seed: UInt64,
):
    var step = Int(step_w)
    var m = Int(block_idx.x)
    var base = m * 6
    var rows = Int(descs[base + 0])
    var cols = Int(descs[base + 1])
    var goff = Int(descs[base + 2])
    var roff = Int(descs[base + 3])
    var coff = Int(descs[base + 4])
    var soff = Int(descs[base + 5])
    var numel = rows * cols
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)

    var sh = stack_allocation[2, Scalar[DType.float64], address_space=AddressSpace.SHARED]()
    var shr = stack_allocation[256, Scalar[DType.float64], address_space=AddressSpace.SHARED]()  # tree-reduce scratch (TPB)

    # ── phase 1: sanitize grad into g scratch (reuse g in place), EMA row/col ──
    var i = tid
    while i < numel:
        var gv = rebind[Scalar[DType.float32]](g[goff + i]) * grad_scale
        if gv != gv or gv > Float32(3.0e38) or gv < Float32(-3.0e38):
            g[goff + i] = Float32(0.0)
        i += nt
    barrier()
    # row EMA: one thread per row, serial over cols (f64 sum to match host)
    var r = tid
    while r < rows:
        var s = Float64(0.0)
        for c in range(cols):
            var gv = Float64(rebind[Scalar[DType.float32]](g[goff + r * cols + c]))
            s += gv * gv
        var row_mean = Float32(s / Float64(cols))
        var rv = rebind[Scalar[DType.float32]](row_var[roff + r])
        row_var[roff + r] = beta2 * rv + one_minus_b2 * (row_mean + eps)
        r += nt
    # col EMA: one thread per col, serial over rows
    var cc = tid
    while cc < cols:
        var s = Float64(0.0)
        for rr in range(rows):
            var gv = Float64(rebind[Scalar[DType.float32]](g[goff + rr * cols + cc]))
            s += gv * gv
        var col_mean = Float32(s / Float64(rows))
        var cv = rebind[Scalar[DType.float32]](col_var[coff + cc])
        col_var[coff + cc] = beta2 * cv + one_minus_b2 * (col_mean + eps)
        cc += nt
    barrier()

    # ── phase 2: rmean = mean(row_var) over rows (parallel block reduce, f64) ──
    var rsl = Float64(0.0)
    var rr0 = tid
    while rr0 < rows:
        rsl += Float64(rebind[Scalar[DType.float32]](row_var[roff + rr0]))
        rr0 += nt
    shr[tid] = rsl
    barrier()
    var act2 = nt // 2
    while act2 > 0:
        if tid < act2:
            shr[tid] = shr[tid] + shr[tid + act2]
        barrier()
        act2 //= 2
    if tid == 0:
        sh[0] = shr[0] / Float64(rows)
    barrier()
    var rmean = Float32(sh[0])

    # ── phase 3: update = rsqrt(row/rmean) * rsqrt(col) * g ──
    var j = tid
    while j < numel:
        var rr = j // cols
        var cci = j % cols
        var rfac = Float32(1.0) / sqrt(rebind[Scalar[DType.float32]](row_var[roff + rr]) / rmean)
        var cfac = Float32(1.0) / sqrt(rebind[Scalar[DType.float32]](col_var[coff + cci]))
        var gv = rebind[Scalar[DType.float32]](g[goff + j])
        upd[goff + j] = rfac * cfac * gv
        j += nt
    barrier()

    # ── phase 4: RMS = sqrt(mean(update^2)) over numel (parallel block reduce) ──
    var usl = Float64(0.0)
    var k0 = tid
    while k0 < numel:
        var uv = Float64(rebind[Scalar[DType.float32]](upd[goff + k0]))
        usl += uv * uv
        k0 += nt
    shr[tid] = usl
    barrier()
    var act4 = nt // 2
    while act4 > 0:
        if tid < act4:
            shr[tid] = shr[tid] + shr[tid + act4]
        barrier()
        act4 //= 2
    if tid == 0:
        var u_rms = Float64(sqrt(Float32(shr[0] / Float64(numel))))  # F32 sqrt (GPU has no F64 sqrt)
        var scale_div = u_rms / Float64(clip)
        if scale_div < 1.0:
            scale_div = 1.0
        sh[1] = 1.0 / scale_div
    barrier()
    var inv_scale = Float32(sh[1])

    # ── phase 5: scale, clamp, record sign bit into the ring plane ──
    var plane = Int(hist_idx[m])     # plane to overwrite (oldest)
    var k2 = tid
    while k2 < numel:
        var u = rebind[Scalar[DType.float32]](upd[goff + k2]) * inv_scale
        if u > clip:
            u = clip
        elif u < -clip:
            u = -clip
        upd[goff + k2] = u
        var bit = UInt8(1) if u > Float32(0.0) else UInt8(0)
        sign_ring[soff + plane * numel + k2] = bit
        k2 += nt
    barrier()

    # ── phase 6: ring bookkeeping (thread 0) + PARALLEL vote (block reduce) ──
    # `plane` (= hist_idx[m], read in phase 5) and fill are uniform across the
    # block, so every thread takes the same branch — no divergent-barrier hang.
    var fill6 = Int(hist_fill[m]) + 1
    if fill6 > _A3_H:
        fill6 = _A3_H
    var newidx = (plane + 1) % _A3_H
    barrier()   # all threads read hist_fill before thread 0 overwrites it
    if tid == 0:
        hist_fill[m] = Int32(fill6)
        hist_idx[m] = Int32(newidx)
    if fill6 == _A3_H:
        var start_plane = newidx   # oldest after advance (== host hist_idx)
        var mnum = Float64(0.0)
        var mden = Float64(0.0)
        var e0 = tid
        while e0 < numel:
            var s1 = 0
            var flips = 0
            var prev = False
            for kk in range(_A3_H):
                var b = sign_ring[soff + ((start_plane + kk) % _A3_H) * numel + e0] != UInt8(0)
                if b:
                    s1 += 1
                if kk > 0 and (b != prev):
                    flips += 1
                prev = b
            var w = Float64(rebind[Scalar[DType.float32]](upd[goff + e0]))
            if w < 0.0:
                w = -w
            if s1 == _A3_H or s1 == 0:
                mnum += w
            elif flips == (_A3_H - 1):
                mnum -= w
            mden += w
            e0 += nt
        # block-reduce mnum -> group_num atomic
        shr[tid] = mnum
        barrier()
        var actn = nt // 2
        while actn > 0:
            if tid < actn:
                shr[tid] = shr[tid] + shr[tid + actn]
            barrier()
            actn //= 2
        if tid == 0:
            _ = Atomic[DType.float64].fetch_add(group_num.ptr, shr[0])
        barrier()
        # block-reduce mden -> group_den atomic
        shr[tid] = mden
        barrier()
        var actd = nt // 2
        while actd > 0:
            if tid < actd:
                shr[tid] = shr[tid] + shr[tid + actd]
            barrier()
            actd //= 2
        if tid == 0:
            _ = Atomic[DType.float64].fetch_add(group_den.ptr, shr[0])
        barrier()
    else:
        barrier()

    # ── phase 7: decoupled WD + param update (f32 master) + device SR -> bf16 ──
    var k3 = tid
    while k3 < numel:
        var u = rebind[Scalar[DType.float32]](upd[goff + k3])
        if weight_decay != Float32(0.0):
            u = u + weight_decay * rebind[Scalar[DType.float32]](p[goff + k3])
        var new_p = rebind[Scalar[DType.float32]](p[goff + k3]) - lr * u
        p[goff + k3] = new_p
        # device stochastic-rounding bf16 writeback: same bit-trick as
        # sr_truncate_f32_to_bits (add a uniform [0,2^16) into the dropped 16
        # mantissa bits, mask, narrow). splitmix64 over a per-(elem,step) counter
        # gives a decorrelated uniform; unbiased (SR property is RNG-independent,
        # gated by automagic3_sr_parity_gate). low 16 zeroed -> BFloat16() exact.
        var cnt = UInt64(goff + k3) + UInt64(step) * UInt64(0x9E3779B97F4A7C15) + seed
        cnt = cnt + UInt64(0x9E3779B97F4A7C15)
        var z = cnt
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        var rnd = UInt32(z >> 32) & UInt32(0x0000FFFF)
        var asi = new_p.to_bits[DType.uint32]() + rnd
        asi = asi & UInt32(0xFFFF0000)
        pb[goff + k3] = BFloat16(Float32(from_bits=asi))
        k3 += nt


# ─────────────────────────────────────────────────────────────────────────────
# Persistent device state + the host-facing step. Built once from host_lora; the
# F32 master + factored state + sign ring live on device across steps. Mirrors
# the PROVEN parity-harness dispatch (automagic3_device_parity.mojo). The bf16 a/b
# the forward consumes are produced by the VERIFIED host SR writeback
# (automagic3_writeback_bf16_sr) from the downloaded F32 master each step.
# ─────────────────────────────────────────────────────────────────────────────
from max.gpu.host import DeviceBuffer, HostBuffer
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.automagic3 import (
    Automagic3Rng, automagic3_writeback_bf16_sr,
    AUTOMAGIC3_DEFAULT_LR, AUTOMAGIC3_LR_MIN, AUTOMAGIC3_LR_MAX,
)
from serenitymojo.training.device_train_step import TrainStepDeviceResult
from serenitymojo.training.on_device_global_norm import on_device_grad_stats
from serenitymojo.training.perf_record import (
    PERF_FAST_PATH_DEVICE,
    PERF_FAST_PATH_HOST_GRAD_COMPAT,
    TrainingPhaseTimings,
)
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor


comptime TArc = ArcPointer[Tensor]


struct Automagic3DeviceState(Movable):
    var inited: Bool
    var nmat: Int          # 2 * n_adapters (A,B per adapter)
    var np: Int            # total param elements
    var nr: Int            # total row_var elements
    var ncv: Int           # total col_var elements
    var lr: Float64
    var rng: Automagic3Rng
    var p_dev: DeviceBuffer[DType.float32]    # F32 master (persistent)
    var g_dev: DeviceBuffer[DType.float32]
    var u_dev: DeviceBuffer[DType.float32]
    var rv_dev: DeviceBuffer[DType.float32]
    var cv_dev: DeviceBuffer[DType.float32]
    var sr_dev: DeviceBuffer[DType.uint8]
    var dsc_dev: DeviceBuffer[DType.int32]
    var hidx_dev: DeviceBuffer[DType.int32]
    var hfill_dev: DeviceBuffer[DType.int32]
    var gnum_dev: DeviceBuffer[DType.float64]
    var gden_dev: DeviceBuffer[DType.float64]
    var pb_dev: DeviceBuffer[DType.uint8]      # raw BF16 a/b bytes (device-SR output)
    var seg_len: List[Int]                     # A,B element counts per adapter
    var step: Int

    def __init__(out self, ctx: DeviceContext) raises:
        self.inited = False
        self.nmat = 0; self.np = 0; self.nr = 0; self.ncv = 0
        self.lr = AUTOMAGIC3_DEFAULT_LR
        self.rng = Automagic3Rng(UInt64(0x5EED_A3D0))
        # placeholder 1-elt buffers; real ones built in lazy_init
        self.p_dev = ctx.enqueue_create_buffer[DType.float32](1)
        self.g_dev = ctx.enqueue_create_buffer[DType.float32](1)
        self.u_dev = ctx.enqueue_create_buffer[DType.float32](1)
        self.rv_dev = ctx.enqueue_create_buffer[DType.float32](1)
        self.cv_dev = ctx.enqueue_create_buffer[DType.float32](1)
        self.sr_dev = ctx.enqueue_create_buffer[DType.uint8](1)
        self.dsc_dev = ctx.enqueue_create_buffer[DType.int32](1)
        self.hidx_dev = ctx.enqueue_create_buffer[DType.int32](1)
        self.hfill_dev = ctx.enqueue_create_buffer[DType.int32](1)
        self.gnum_dev = ctx.enqueue_create_buffer[DType.float64](1)
        self.gden_dev = ctx.enqueue_create_buffer[DType.float64](1)
        self.pb_dev = ctx.enqueue_create_buffer[DType.uint8](2)
        self.seg_len = List[Int]()
        self.step = 0

    def elem_offset(self, adapter_idx: Int, b_side: Bool) -> Int:
        var off = 0
        for i in range(adapter_idx):
            off += self.seg_len[2 * i]
            off += self.seg_len[2 * i + 1]
        if b_side:
            off += self.seg_len[2 * adapter_idx]
        return off


def _dynf(p: UnsafePointer[Float32, MutAnyOrigin], n: Int) -> LayoutTensor[DType.float32, _DYN1, MutAnyOrigin]:
    return LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(p)
        ),
        runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](n)),
    )


def automagic3_device_state_init_from_adapters(
    mut state: Automagic3DeviceState,
    adapters: List[LoraAdapter],
    start_lr: Float64,
    ctx: DeviceContext,
) raises:
    """Initialize persistent A3 state and BF16 device param mirror from host LoRA."""
    if state.inited:
        return
    var na = len(adapters)
    if na == 0:
        raise Error("automagic3_device_state_init_from_adapters: empty adapter set")
    var nmat = 2 * na
    var np = 0
    var nr = 0
    var ncv = 0
    var seg_len = List[Int]()
    for i in range(na):
        var n_a = len(adapters[i].a)
        var n_b = len(adapters[i].b)
        seg_len.append(n_a)
        seg_len.append(n_b)
        np += n_a + n_b
        nr += adapters[i].rank + adapters[i].out_f
        ncv += adapters[i].in_f + adapters[i].rank
    state.nmat = nmat
    state.np = np
    state.nr = nr
    state.ncv = ncv
    state.lr = start_lr
    state.seg_len = seg_len^
    state.p_dev = ctx.enqueue_create_buffer[DType.float32](np)
    state.g_dev = ctx.enqueue_create_buffer[DType.float32](np)
    state.u_dev = ctx.enqueue_create_buffer[DType.float32](np)
    state.rv_dev = ctx.enqueue_create_buffer[DType.float32](nr)
    state.cv_dev = ctx.enqueue_create_buffer[DType.float32](ncv)
    state.sr_dev = ctx.enqueue_create_buffer[DType.uint8](_A3_H * np)
    state.dsc_dev = ctx.enqueue_create_buffer[DType.int32](nmat * 6)
    state.hidx_dev = ctx.enqueue_create_buffer[DType.int32](nmat)
    state.hfill_dev = ctx.enqueue_create_buffer[DType.int32](nmat)
    state.pb_dev = ctx.enqueue_create_buffer[DType.uint8](np * 2)

    var ph = ctx.enqueue_create_host_buffer[DType.float32](np)
    var pbh = ctx.enqueue_create_host_buffer[DType.uint8](np * 2)
    var dh = ctx.enqueue_create_host_buffer[DType.int32](nmat * 6)
    var pp = ph.unsafe_ptr()
    var pbp = pbh.unsafe_ptr().bitcast[BFloat16]()
    var dp = dh.unsafe_ptr()
    var goff = 0
    var roff = 0
    var coff = 0
    var soff = 0
    var mi = 0
    for i in range(na):
        var na_e = len(adapters[i].a)
        for e in range(na_e):
            var v = adapters[i].a[e]
            pp[goff + e] = Float32(v)
            pbp[goff + e] = v
        dp[mi * 6 + 0] = Int32(adapters[i].rank)
        dp[mi * 6 + 1] = Int32(adapters[i].in_f)
        dp[mi * 6 + 2] = Int32(goff)
        dp[mi * 6 + 3] = Int32(roff)
        dp[mi * 6 + 4] = Int32(coff)
        dp[mi * 6 + 5] = Int32(soff)
        goff += na_e
        roff += adapters[i].rank
        coff += adapters[i].in_f
        soff += _A3_H * na_e
        mi += 1

        var nb_e = len(adapters[i].b)
        for e in range(nb_e):
            var v = adapters[i].b[e]
            pp[goff + e] = Float32(v)
            pbp[goff + e] = v
        dp[mi * 6 + 0] = Int32(adapters[i].out_f)
        dp[mi * 6 + 1] = Int32(adapters[i].rank)
        dp[mi * 6 + 2] = Int32(goff)
        dp[mi * 6 + 3] = Int32(roff)
        dp[mi * 6 + 4] = Int32(coff)
        dp[mi * 6 + 5] = Int32(soff)
        goff += nb_e
        roff += adapters[i].out_f
        coff += adapters[i].rank
        soff += _A3_H * nb_e
        mi += 1
    ctx.enqueue_copy(dst_buf=state.p_dev, src_buf=ph)
    ctx.enqueue_copy(dst_buf=state.pb_dev, src_buf=pbh)
    ctx.enqueue_copy(dst_buf=state.dsc_dev, src_buf=dh)
    state.rv_dev.enqueue_fill(Float32(0.0))
    state.cv_dev.enqueue_fill(Float32(0.0))
    state.sr_dev.enqueue_fill(UInt8(0))
    state.hidx_dev.enqueue_fill(Int32(0))
    state.hfill_dev.enqueue_fill(Int32(0))
    state.inited = True


def _a3_grad_as_f32(
    t: TArc, mut keepalive: List[TArc], ctx: DeviceContext
) raises -> TArc:
    if t[].dtype() == STDtype.F32:
        return t.copy()
    var tc = TArc(cast_tensor(t[], STDtype.F32, ctx))
    keepalive.append(tc.copy())
    return tc^


def automagic3_device_state_copy_device_grad_pair(
    mut state: Automagic3DeviceState,
    adapter_idx: Int,
    d_a: TArc,
    d_b: TArc,
    mut keepalive: List[TArc],
    ctx: DeviceContext,
) raises:
    """Copy one transient device dA/dB pair into A3's persistent flat grad buffer."""
    if not state.inited:
        raise Error("automagic3_device_state_copy_device_grad_pair: state is not initialized")
    if adapter_idx < 0 or adapter_idx * 2 + 1 >= len(state.seg_len):
        raise Error(
            "automagic3_device_state_copy_device_grad_pair: adapter index outside state "
            + String(adapter_idx)
        )
    var n_a = state.seg_len[2 * adapter_idx]
    var n_b = state.seg_len[2 * adapter_idx + 1]
    var da = _a3_grad_as_f32(d_a, keepalive, ctx)
    var db = _a3_grad_as_f32(d_b, keepalive, ctx)
    if da[].numel() != n_a or db[].numel() != n_b:
        raise Error(
            "automagic3_device_state_copy_device_grad_pair: grad numel mismatch at "
            + String(adapter_idx)
        )
    var a_off = state.elem_offset(adapter_idx, False)
    var b_off = state.elem_offset(adapter_idx, True)
    var dst_a = state.g_dev.create_sub_buffer[DType.float32](a_off, n_a)
    var dst_b = state.g_dev.create_sub_buffer[DType.float32](b_off, n_b)
    var src_a = da[].buf.create_sub_buffer[DType.float32](0, n_a)
    var src_b = db[].buf.create_sub_buffer[DType.float32](0, n_b)
    ctx.enqueue_copy(dst_buf=dst_a, src_buf=src_a)
    ctx.enqueue_copy(dst_buf=dst_b, src_buf=src_b)


def _automagic3_device_step_preloaded(
    mut state: Automagic3DeviceState,
    beta2: Float64,
    eps: Float64,
    clip: Float64,
    weight_decay: Float64,
    grad_scale: Float32,
    ctx: DeviceContext,
) raises -> Float32:
    if not state.inited:
        raise Error("_automagic3_device_step_preloaded: state is not initialized")
    state.gnum_dev.enqueue_fill(Float64(0.0))
    state.gden_dev.enqueue_fill(Float64(0.0))

    var P = _dynf(state.p_dev.unsafe_ptr(), state.np)
    var G = _dynf(state.g_dev.unsafe_ptr(), state.np)
    var U = _dynf(state.u_dev.unsafe_ptr(), state.np)
    var RV = _dynf(state.rv_dev.unsafe_ptr(), state.nr)
    var CV = _dynf(state.cv_dev.unsafe_ptr(), state.ncv)
    var SR = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=Int(state.sr_dev.unsafe_ptr())
        ),
        runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](_A3_H * state.np)),
    )
    var DSC = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(state.dsc_dev.unsafe_ptr())
        ),
        runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](state.nmat * 6)),
    )
    var HIDX = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(state.hidx_dev.unsafe_ptr())
        ),
        runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](state.nmat)),
    )
    var HFILL = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(state.hfill_dev.unsafe_ptr())
        ),
        runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](state.nmat)),
    )
    var GNUM = LayoutTensor[DType.float64, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float64], MutAnyOrigin](
            unsafe_from_address=Int(state.gnum_dev.unsafe_ptr())
        ),
        runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](1)),
    )
    var GDEN = LayoutTensor[DType.float64, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float64], MutAnyOrigin](
            unsafe_from_address=Int(state.gden_dev.unsafe_ptr())
        ),
        runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](1)),
    )
    var PB = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(state.pb_dev.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=RuntimeLayout[_DYN1].row_major(IndexList[1](state.np)),
    )

    ctx.enqueue_function[automagic3_factored_kernel](
        P, G, U, RV, CV, SR, DSC, HIDX, HFILL, GNUM, GDEN, PB,
        Float32(beta2), Float32(1.0 - beta2), Float32(eps), Float32(clip),
        Float32(state.lr), Float32(weight_decay), grad_scale,
        Int32(state.step), UInt64(0x5EED_A3D0),
        grid_dim=state.nmat, block_dim=256,
    )
    state.step += 1

    var num_h = ctx.enqueue_create_host_buffer[DType.float64](1)
    var den_h = ctx.enqueue_create_host_buffer[DType.float64](1)
    ctx.enqueue_copy(dst_buf=num_h, src_buf=state.gnum_dev)
    ctx.enqueue_copy(dst_buf=den_h, src_buf=state.gden_dev)
    ctx.synchronize()
    var gn = num_h.unsafe_ptr()[0]
    var gd = den_h.unsafe_ptr()[0]
    if gd > 0.0:
        var den = gd
        if den < 1.0e-30:
            den = 1.0e-30
        var sig = gn / den
        if sig > 1.0:
            sig = 1.0
        elif sig < -1.0:
            sig = -1.0
        var nl = state.lr * exp(sig)
        if nl < AUTOMAGIC3_LR_MIN:
            nl = AUTOMAGIC3_LR_MIN
        elif nl > AUTOMAGIC3_LR_MAX:
            nl = AUTOMAGIC3_LR_MAX
        state.lr = nl
    return Float32(state.lr)


def automagic3_device_state_sync_params(
    state: Automagic3DeviceState,
    mut adapters: List[LoraAdapter],
    ctx: DeviceContext,
) raises:
    if not state.inited:
        raise Error("automagic3_device_state_sync_params: state is not initialized")
    var pbres = ctx.enqueue_create_host_buffer[DType.uint8](state.np * 2)
    ctx.enqueue_copy(dst_buf=pbres, src_buf=state.pb_dev)
    ctx.synchronize()
    var pbp = pbres.unsafe_ptr().bitcast[BFloat16]()
    var off = 0
    for i in range(len(adapters)):
        var n_a = len(adapters[i].a)
        for e in range(n_a):
            adapters[i].a[e] = pbp[off + e]
        off += n_a
        var n_b = len(adapters[i].b)
        for e in range(n_b):
            adapters[i].b[e] = pbp[off + e]
        off += n_b


def automagic3_device_step(
    mut state: Automagic3DeviceState,
    mut adapters: List[LoraAdapter],
    d_a: List[List[Float32]],
    d_b: List[List[Float32]],
    start_lr: Float64,
    beta2: Float64,
    eps: Float64,
    clip: Float64,
    weight_decay: Float64,
    ctx: DeviceContext,
) raises -> Float32:
    """One GPU automagic3 step over ALL adapters (A,B). Math on device (gated
    vs host == ai-toolkit); SR bf16 writeback on host (verified fn). Returns the
    self-adapted lr after this step's vote."""
    var na = len(adapters)
    if na == 0:
        return Float32(state.lr)
    automagic3_device_state_init_from_adapters(state, adapters, start_lr, ctx)

    # ── upload grads (matrix order: A then B per adapter) ──
    var gh = ctx.enqueue_create_host_buffer[DType.float32](state.np)
    var gp = gh.unsafe_ptr()
    var off = 0
    for i in range(na):
        for e in range(len(d_a[i])): gp[off + e] = d_a[i][e]
        off += len(d_a[i])
        for e in range(len(d_b[i])): gp[off + e] = d_b[i][e]
        off += len(d_b[i])
    ctx.enqueue_copy(dst_buf=state.g_dev, src_buf=gh)
    var lr_now = _automagic3_device_step_preloaded(
        state, beta2, eps, clip, weight_decay, Float32(1.0), ctx,
    )
    automagic3_device_state_sync_params(state, adapters, ctx)
    return lr_now


def automagic3_device_preloaded_step_result(
    mut state: Automagic3DeviceState,
    loss: Float32,
    beta2: Float64,
    eps: Float64,
    clip: Float64,
    weight_decay: Float64,
    max_grad_norm: Float32,
    ctx: DeviceContext,
) raises -> TrainStepDeviceResult:
    """Device-fast Automagic3 over state.g_dev already filled with F32 grads."""
    if not state.inited:
        raise Error("automagic3_device_preloaded_step_result: state is not initialized")
    var t0 = perf_counter_ns()
    var sh: List[Int] = [state.np]
    var stat_grads = List[TArc]()
    stat_grads.append(TArc(Tensor(
        state.g_dev.create_sub_buffer[DType.uint8](0, state.np * 4),
        sh.copy(),
        STDtype.F32,
    )))
    var stats = on_device_grad_stats(stat_grads, ctx)
    if stats.nonfinite_count != 0:
        raise Error("automagic3_device_preloaded_step_result: nonfinite device grads")
    var grad_scale = Float32(1.0)
    if max_grad_norm > Float32(0.0) and stats.grad_norm > max_grad_norm:
        grad_scale = max_grad_norm / stats.grad_norm
    var lr_now = _automagic3_device_step_preloaded(
        state, beta2, eps, clip, weight_decay, grad_scale, ctx,
    )
    var t1 = perf_counter_ns()
    var phases = TrainingPhaseTimings(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        Float64(t1 - t0) / 1.0e9,
        0.0,
        0.0,
    )
    var result = TrainStepDeviceResult(
        loss,
        stats.grad_norm,
        grad_scale,
        phases^,
        stats.scalar_readback_count + 2,
        0,
        stats.sync_count + 1,
        stats.nonfinite_count,
        PERF_FAST_PATH_DEVICE,
        String("device-automagic3-preloaded-grads"),
        String("automagic3_lr=") + String(lr_now),
    )
    result.validate()
    return result^


def automagic3_device_step_result(
    mut state: Automagic3DeviceState,
    mut adapters: List[LoraAdapter],
    d_a: List[List[Float32]],
    d_b: List[List[Float32]],
    loss: Float32,
    grad_norm: Float32,
    start_lr: Float64,
    beta2: Float64,
    eps: Float64,
    clip: Float64,
    weight_decay: Float64,
    ctx: DeviceContext,
) raises -> TrainStepDeviceResult:
    """Host-grad-compatible Automagic3 device optimizer step.

    Grad lists are still host-resident compatibility inputs, and the step reads
    the bf16 param mirror back for the next forward/save path. The result is
    therefore deliberately not a device-fast claim even though the optimizer
    math runs in the GPU kernel.
    """
    var t0 = perf_counter_ns()
    var lr_now = automagic3_device_step(
        state,
        adapters,
        d_a,
        d_b,
        start_lr,
        beta2,
        eps,
        clip,
        weight_decay,
        ctx,
    )
    var t1 = perf_counter_ns()
    var phases = TrainingPhaseTimings(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        Float64(t1 - t0) / 1.0e9,
        0.0,
        0.0,
    )
    var result = TrainStepDeviceResult(
        loss,
        grad_norm,
        Float32(1.0),
        phases^,
        2,
        1,  # full_tensor_readback_count: bf16 param mirror for next forward/save
        1,
        0,
        PERF_FAST_PATH_HOST_GRAD_COMPAT,
        String("device-automagic3-host-grad-compat"),
        String("automagic3_lr=") + String(lr_now),
    )
    result.validate()
    return result^


# ── hi-res inline-sample offload (zimage free-then-render pattern) ────────────
# The persistent A3 state (~4.8GB at krea2 rank64: p/g/u F32×np + sr 8·np u8 +
# pb 2·np u8) does NOT co-fit with a 1024 inline render on a 24GB card
# (MEASURED 2026-07-01: render-then-init OOMs at A3 init; init-then-render OOMs
# in the sampler = the original cadence-sample crash). Offload snapshots every
# persistent buffer to host, swaps the device buffers to 1-elt placeholders
# (frees the VRAM for the render), and restore rebuilds + re-uploads.
# gnum/gden are per-step scalar accumulators (re-filled each step) — skipped.
struct Automagic3HostSnapshot(Movable):
    var p: HostBuffer[DType.float32]
    var g: HostBuffer[DType.float32]
    var u: HostBuffer[DType.float32]
    var rv: HostBuffer[DType.float32]
    var cv: HostBuffer[DType.float32]
    var sr: HostBuffer[DType.uint8]
    var dsc: HostBuffer[DType.int32]
    var hidx: HostBuffer[DType.int32]
    var hfill: HostBuffer[DType.int32]
    var pb: HostBuffer[DType.uint8]

    def __init__(
        out self,
        var p: HostBuffer[DType.float32], var g: HostBuffer[DType.float32],
        var u: HostBuffer[DType.float32], var rv: HostBuffer[DType.float32],
        var cv: HostBuffer[DType.float32], var sr: HostBuffer[DType.uint8],
        var dsc: HostBuffer[DType.int32], var hidx: HostBuffer[DType.int32],
        var hfill: HostBuffer[DType.int32], var pb: HostBuffer[DType.uint8],
    ):
        self.p = p^; self.g = g^; self.u = u^; self.rv = rv^; self.cv = cv^
        self.sr = sr^; self.dsc = dsc^; self.hidx = hidx^; self.hfill = hfill^
        self.pb = pb^


def automagic3_device_state_offload_to_host(
    mut state: Automagic3DeviceState, ctx: DeviceContext
) raises -> Automagic3HostSnapshot:
    """D2H-snapshot ALL persistent A3 buffers and free them on device.

    After this call state.inited is False; the ONLY legal next operations are
    automagic3_device_state_restore_from_host (below) or drop. Metadata
    (np/nr/ncv/nmat/seg_len/step/lr/rng) stays in `state` so restore is exact.
    """
    if not state.inited:
        raise Error("automagic3_device_state_offload_to_host: state not initialized")
    var hp = ctx.enqueue_create_host_buffer[DType.float32](state.np)
    var hg = ctx.enqueue_create_host_buffer[DType.float32](state.np)
    var hu = ctx.enqueue_create_host_buffer[DType.float32](state.np)
    var hrv = ctx.enqueue_create_host_buffer[DType.float32](state.nr)
    var hcv = ctx.enqueue_create_host_buffer[DType.float32](state.ncv)
    var hsr = ctx.enqueue_create_host_buffer[DType.uint8](_A3_H * state.np)
    var hdsc = ctx.enqueue_create_host_buffer[DType.int32](state.nmat * 6)
    var hhidx = ctx.enqueue_create_host_buffer[DType.int32](state.nmat)
    var hhfill = ctx.enqueue_create_host_buffer[DType.int32](state.nmat)
    var hpb = ctx.enqueue_create_host_buffer[DType.uint8](state.np * 2)
    ctx.enqueue_copy(dst_buf=hp, src_buf=state.p_dev)
    ctx.enqueue_copy(dst_buf=hg, src_buf=state.g_dev)
    ctx.enqueue_copy(dst_buf=hu, src_buf=state.u_dev)
    ctx.enqueue_copy(dst_buf=hrv, src_buf=state.rv_dev)
    ctx.enqueue_copy(dst_buf=hcv, src_buf=state.cv_dev)
    ctx.enqueue_copy(dst_buf=hsr, src_buf=state.sr_dev)
    ctx.enqueue_copy(dst_buf=hdsc, src_buf=state.dsc_dev)
    ctx.enqueue_copy(dst_buf=hhidx, src_buf=state.hidx_dev)
    ctx.enqueue_copy(dst_buf=hhfill, src_buf=state.hfill_dev)
    ctx.enqueue_copy(dst_buf=hpb, src_buf=state.pb_dev)
    ctx.synchronize()
    # Swap the big device buffers for 1-elt placeholders so they actually free.
    state.p_dev = ctx.enqueue_create_buffer[DType.float32](1)
    state.g_dev = ctx.enqueue_create_buffer[DType.float32](1)
    state.u_dev = ctx.enqueue_create_buffer[DType.float32](1)
    state.rv_dev = ctx.enqueue_create_buffer[DType.float32](1)
    state.cv_dev = ctx.enqueue_create_buffer[DType.float32](1)
    state.sr_dev = ctx.enqueue_create_buffer[DType.uint8](1)
    state.dsc_dev = ctx.enqueue_create_buffer[DType.int32](1)
    state.hidx_dev = ctx.enqueue_create_buffer[DType.int32](1)
    state.hfill_dev = ctx.enqueue_create_buffer[DType.int32](1)
    state.pb_dev = ctx.enqueue_create_buffer[DType.uint8](2)
    state.inited = False
    ctx.synchronize()
    return Automagic3HostSnapshot(
        hp^, hg^, hu^, hrv^, hcv^, hsr^, hdsc^, hhidx^, hhfill^, hpb^
    )


def automagic3_device_state_restore_from_host(
    mut state: Automagic3DeviceState,
    var snap: Automagic3HostSnapshot,
    ctx: DeviceContext,
) raises:
    """Rebuild the device buffers from an offload snapshot (inverse of offload)."""
    if state.inited:
        raise Error("automagic3_device_state_restore_from_host: state already initialized")
    state.p_dev = ctx.enqueue_create_buffer[DType.float32](state.np)
    state.g_dev = ctx.enqueue_create_buffer[DType.float32](state.np)
    state.u_dev = ctx.enqueue_create_buffer[DType.float32](state.np)
    state.rv_dev = ctx.enqueue_create_buffer[DType.float32](state.nr)
    state.cv_dev = ctx.enqueue_create_buffer[DType.float32](state.ncv)
    state.sr_dev = ctx.enqueue_create_buffer[DType.uint8](_A3_H * state.np)
    state.dsc_dev = ctx.enqueue_create_buffer[DType.int32](state.nmat * 6)
    state.hidx_dev = ctx.enqueue_create_buffer[DType.int32](state.nmat)
    state.hfill_dev = ctx.enqueue_create_buffer[DType.int32](state.nmat)
    state.pb_dev = ctx.enqueue_create_buffer[DType.uint8](state.np * 2)
    ctx.enqueue_copy(dst_buf=state.p_dev, src_buf=snap.p)
    ctx.enqueue_copy(dst_buf=state.g_dev, src_buf=snap.g)
    ctx.enqueue_copy(dst_buf=state.u_dev, src_buf=snap.u)
    ctx.enqueue_copy(dst_buf=state.rv_dev, src_buf=snap.rv)
    ctx.enqueue_copy(dst_buf=state.cv_dev, src_buf=snap.cv)
    ctx.enqueue_copy(dst_buf=state.sr_dev, src_buf=snap.sr)
    ctx.enqueue_copy(dst_buf=state.dsc_dev, src_buf=snap.dsc)
    ctx.enqueue_copy(dst_buf=state.hidx_dev, src_buf=snap.hidx)
    ctx.enqueue_copy(dst_buf=state.hfill_dev, src_buf=snap.hfill)
    ctx.enqueue_copy(dst_buf=state.pb_dev, src_buf=snap.pb)
    ctx.synchronize()
    state.inited = True


# ═════════════════════════════════════════════════════════════════════════════
# DISK serialization (MJ-1081): full A3 optimizer state as ONE safetensors
# sidecar (`<ckpt>.a3state.safetensors`) so resume is EXACT, not warm.
# SAVE reuses the gate-proven offload->restore round trip (MJ-1043 render
# path): D2H-snapshot, write the host buffers, restore. LOAD requires a
# freshly-initialized state (same adapter geometry) and overwrites every
# buffer + step/lr/rng from the file — geometry mismatch FAILS LOUD.
# ═════════════════════════════════════════════════════════════════════════════

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.ffi import BytePtr, sys_memcpy

comptime A3STATE_VERSION = 1


def _a3_i64_tensor(v: Int, ctx: DeviceContext) raises -> Tensor:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](8)
    host.unsafe_ptr().bitcast[Int64]()[0] = Int64(v)
    var dev = ctx.enqueue_create_buffer[DType.uint8](8)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    var shape: List[Int] = [1]
    return Tensor(dev^, shape^, STDtype.I64)


def _a3_meta_i64(st: SafeTensors, name: String, default: Int) raises -> Int:
    if name not in st.tensors:
        return default
    var b = st.tensor_bytes(name)
    if len(b) < 8:
        return default
    var v = 0
    for i in range(8):
        v = v | (Int(b[i]) << (8 * i))
    return v


def _a3_host_u8_tensor[
    T: DType
](hb: HostBuffer[T], nbytes: Int, var shape: List[Int], dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    """Wrap a typed host buffer's raw bytes as a device tensor (for the writer)."""
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var u8h = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var dst = BytePtr(unsafe_from_address=Int(u8h.unsafe_ptr()))
    var src = BytePtr(unsafe_from_address=Int(hb.unsafe_ptr()))
    _ = sys_memcpy(dst, src, nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=u8h)
    ctx.synchronize()
    return Tensor(dev^, shape^, dt)


def _a3_load_into[
    T: DType
](st: SafeTensors, name: String, dst_buf: DeviceBuffer[T], nelems: Int, elem_bytes: Int, ctx: DeviceContext) raises:
    """H2D a sidecar tensor's raw bytes into an existing typed device buffer."""
    var nbytes = nelems * elem_bytes
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.nbytes() != nbytes:
        raise Error(
            String("a3state: size mismatch for ") + name + ": file "
            + String(tv.nbytes()) + " vs state " + String(nbytes)
        )
    var host = ctx.enqueue_create_host_buffer[T](nelems)
    var dst = BytePtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var srcp = BytePtr(unsafe_from_address=Int(tv.data.unsafe_ptr()))
    _ = sys_memcpy(dst, srcp, nbytes)
    ctx.enqueue_copy(dst_buf=dst_buf, src_buf=host)
    ctx.synchronize()


def automagic3_device_state_save(
    mut state: Automagic3DeviceState, path: String, ctx: DeviceContext
) raises:
    """Write the FULL A3 state to `path` (safetensors). Non-destructive:
    offload->write->restore (the gate-proven render round trip)."""
    if not state.inited:
        raise Error("automagic3_device_state_save: state not initialized")
    var np = state.np
    var nr = state.nr
    var ncv = state.ncv
    var nmat = state.nmat
    var snap = automagic3_device_state_offload_to_host(state, ctx)

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String("a3.p"))
    tensors.append(ArcPointer(_a3_host_u8_tensor(snap.p, np * 4, [np], STDtype.F32, ctx)))
    names.append(String("a3.g"))
    tensors.append(ArcPointer(_a3_host_u8_tensor(snap.g, np * 4, [np], STDtype.F32, ctx)))
    names.append(String("a3.u"))
    tensors.append(ArcPointer(_a3_host_u8_tensor(snap.u, np * 4, [np], STDtype.F32, ctx)))
    names.append(String("a3.rv"))
    tensors.append(ArcPointer(_a3_host_u8_tensor(snap.rv, nr * 4, [nr], STDtype.F32, ctx)))
    names.append(String("a3.cv"))
    tensors.append(ArcPointer(_a3_host_u8_tensor(snap.cv, ncv * 4, [ncv], STDtype.F32, ctx)))
    names.append(String("a3.sr"))
    tensors.append(ArcPointer(_a3_host_u8_tensor(snap.sr, _A3_H * np, [_A3_H * np], STDtype.U8, ctx)))
    names.append(String("a3.dsc"))
    tensors.append(ArcPointer(_a3_host_u8_tensor(snap.dsc, nmat * 6 * 4, [nmat * 6], STDtype.I32, ctx)))
    names.append(String("a3.hidx"))
    tensors.append(ArcPointer(_a3_host_u8_tensor(snap.hidx, nmat * 4, [nmat], STDtype.I32, ctx)))
    names.append(String("a3.hfill"))
    tensors.append(ArcPointer(_a3_host_u8_tensor(snap.hfill, nmat * 4, [nmat], STDtype.I32, ctx)))
    names.append(String("a3.pb"))
    tensors.append(ArcPointer(_a3_host_u8_tensor(snap.pb, np * 2, [np * 2], STDtype.U8, ctx)))

    # seg_len as I64 list tensor
    var seg_host = ctx.enqueue_create_host_buffer[DType.uint8](len(state.seg_len) * 8)
    var sp = seg_host.unsafe_ptr().bitcast[Int64]()
    for i in range(len(state.seg_len)):
        sp[i] = Int64(state.seg_len[i])
    var seg_dev = ctx.enqueue_create_buffer[DType.uint8](len(state.seg_len) * 8)
    ctx.enqueue_copy(dst_buf=seg_dev, src_buf=seg_host)
    ctx.synchronize()
    var seg_shape: List[Int] = [len(state.seg_len)]
    names.append(String("a3.seg_len"))
    tensors.append(ArcPointer(Tensor(seg_dev^, seg_shape^, STDtype.I64)))

    names.append(String("__meta__.version"))
    tensors.append(ArcPointer(_a3_i64_tensor(A3STATE_VERSION, ctx)))
    names.append(String("__meta__.np"))
    tensors.append(ArcPointer(_a3_i64_tensor(np, ctx)))
    names.append(String("__meta__.nr"))
    tensors.append(ArcPointer(_a3_i64_tensor(nr, ctx)))
    names.append(String("__meta__.ncv"))
    tensors.append(ArcPointer(_a3_i64_tensor(ncv, ctx)))
    names.append(String("__meta__.nmat"))
    tensors.append(ArcPointer(_a3_i64_tensor(nmat, ctx)))
    names.append(String("__meta__.step"))
    tensors.append(ArcPointer(_a3_i64_tensor(state.step, ctx)))
    # f64 lr + u64 rng state as raw bit patterns
    var lr_bits = UnsafePointer(to=state.lr).bitcast[Int64]()[0]
    names.append(String("__meta__.lr_bits"))
    tensors.append(ArcPointer(_a3_i64_tensor(Int(lr_bits), ctx)))
    var rng_bits = UnsafePointer(to=state.rng.state).bitcast[Int64]()[0]
    names.append(String("__meta__.rng_bits"))
    tensors.append(ArcPointer(_a3_i64_tensor(Int(rng_bits), ctx)))

    save_safetensors(names, tensors, path, ctx)
    automagic3_device_state_restore_from_host(state, snap^, ctx)


def automagic3_device_state_load(
    mut state: Automagic3DeviceState, path: String, ctx: DeviceContext
) raises:
    """Overwrite a freshly-initialized state from a sidecar. FAILS LOUD on any
    geometry/version mismatch (wrong file for these adapters)."""
    if not state.inited:
        raise Error("automagic3_device_state_load: init from adapters first")
    var st = SafeTensors.open(path)
    if _a3_meta_i64(st, String("__meta__.version"), -1) != A3STATE_VERSION:
        raise Error("a3state: version mismatch: " + path)
    if _a3_meta_i64(st, String("__meta__.np"), -1) != state.np:
        raise Error("a3state: np mismatch (different adapter geometry): " + path)
    if _a3_meta_i64(st, String("__meta__.nr"), -1) != state.nr:
        raise Error("a3state: nr mismatch: " + path)
    if _a3_meta_i64(st, String("__meta__.ncv"), -1) != state.ncv:
        raise Error("a3state: ncv mismatch: " + path)
    if _a3_meta_i64(st, String("__meta__.nmat"), -1) != state.nmat:
        raise Error("a3state: nmat mismatch: " + path)

    _a3_load_into(st, String("a3.p"), state.p_dev, state.np, 4, ctx)
    _a3_load_into(st, String("a3.g"), state.g_dev, state.np, 4, ctx)
    _a3_load_into(st, String("a3.u"), state.u_dev, state.np, 4, ctx)
    _a3_load_into(st, String("a3.rv"), state.rv_dev, state.nr, 4, ctx)
    _a3_load_into(st, String("a3.cv"), state.cv_dev, state.ncv, 4, ctx)
    _a3_load_into(st, String("a3.sr"), state.sr_dev, _A3_H * state.np, 1, ctx)
    _a3_load_into(st, String("a3.dsc"), state.dsc_dev, state.nmat * 6, 4, ctx)
    _a3_load_into(st, String("a3.hidx"), state.hidx_dev, state.nmat, 4, ctx)
    _a3_load_into(st, String("a3.hfill"), state.hfill_dev, state.nmat, 4, ctx)
    _a3_load_into(st, String("a3.pb"), state.pb_dev, state.np * 2, 1, ctx)

    state.step = _a3_meta_i64(st, String("__meta__.step"), state.step)
    var lr_bits = Int64(_a3_meta_i64(st, String("__meta__.lr_bits"), 0))
    state.lr = UnsafePointer(to=lr_bits).bitcast[Float64]()[0]
    var rng_bits = Int64(_a3_meta_i64(st, String("__meta__.rng_bits"), 0))
    state.rng.state = UnsafePointer(to=rng_bits).bitcast[UInt64]()[0]
