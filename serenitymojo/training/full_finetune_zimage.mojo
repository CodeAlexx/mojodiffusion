# training/full_finetune_zimage.mojo — T2.C full-rank finetune (Z-Image first).
#
# Host-offloaded optimizer-state strategy (the ONLY one that fits 24 GB —
# VRAM math in the T2.C report / TIER2_PARITY_CAMPAIGN_2026-06-11.md):
#   * device: bf16 base weights stay resident (12.31 GB) and ARE the live
#     training weights; per step the full-FT backward materializes ONE block's
#     d_W set at a time (~354 MB bf16 transient) and D2H's it (the 10.6 GB
#     grad set never lives on device).
#   * host: F32 masters (21.2 GB) + bnb block-wise 8-bit Adam moments
#     (training/adamw8bit.mojo state, ~10.8 GB) + the pinned bf16 grad
#     staging the stack backward returns (10.6 GB).
#   * per step: host 8-bit AdamW on the masters (grads read bf16->F32 with
#     the global-clip scale folded in), then RNE bf16 write-back H2D into the
#     SAME resident device weight buffers — the device model is always the
#     bf16 image of the masters (the OneTrainer fine-tune master/weight
#     contract; full_finetune_contract.mojo pins F32 masters/moments).
#
# Trainable surface v1: the 30 MAIN-block slot projections (attention
# to_q/to_k/to_v/to_out.0 + feed_forward w1/w3/w2) = 5.309B of 6.155B params
# (86%). Refiners / embedders / norms / adaLN / final layer stay frozen —
# documented delta vs OneTrainer ZImageFineTuneSetup (which trains all
# transformer params); extending grads to those surfaces is the follow-up.
#
# 8-bit Adam math: training/adamw8bit.mojo adam8bit_step_bnb is the
# parity-gated oracle (bnb 0.49.2). The 256-entry linear argmin requant scan
# is O(256/element) — untenable at 5.31B elements/step — so this module
# carries _adam8bit_step_fast: the SAME per-element math with the requant
# argmin replaced by a binary search over the sorted qmap (equivalent
# first-wins tie semantics; nonfinite values code to 0 exactly like the
# linear scan). Equivalence is gated AT RUNTIME on every full-FT run start:
# adam8bit_fast_equivalence_gate runs both implementations on random +
# tail + zero-block data and requires BIT-EQUAL codes/absmax/params.
#
# Mojo 1.0.0b1.

from std.algorithm import parallelize
from std.collections import List, Dict
from std.gpu.host import DeviceContext, HostBuffer
from std.math import sqrt
from std.memory import ArcPointer, alloc
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import (
    sys_system, sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.lora_block import (
    ZIMAGE_SLOTS, SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_W1, SLOT_W3, SLOT_W2,
)
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageFullFTGrads, zimage_fullft_slot_numels,
)
from serenitymojo.training.adamw8bit import (
    Adam8bitState, adam8bit_create_dynamic_map, adam8bit_step_bnb,
    ADAMW8BIT_BLOCK_SIZE,
)

comptime TArc = ArcPointer[Tensor]


# ── the 7 slot weight tensors of one block, in slot order ─────────────────────
def zimage_fullft_slot_tensor(w: ZImageBlockWeights, s: Int) raises -> TArc:
    if s == SLOT_Q:
        return w.wq.copy()
    if s == SLOT_K:
        return w.wk.copy()
    if s == SLOT_V:
        return w.wv.copy()
    if s == SLOT_O:
        return w.wo.copy()
    if s == SLOT_W1:
        return w.w1.copy()
    if s == SLOT_W3:
        return w.w3.copy()
    if s == SLOT_W2:
        return w.w2.copy()
    raise Error("zimage_fullft_slot_tensor: bad slot")


# checkpoint key for (main block, slot) — the SOURCE schema (diffusers
# transformer dir; load_zimage_block_weights_prefixed_mixed reads these).
def zimage_fullft_slot_key(bi: Int, s: Int) raises -> String:
    var p = String("layers.") + String(bi)
    if s == SLOT_Q:
        return p + String(".attention.to_q.weight")
    if s == SLOT_K:
        return p + String(".attention.to_k.weight")
    if s == SLOT_V:
        return p + String(".attention.to_v.weight")
    if s == SLOT_O:
        return p + String(".attention.to_out.0.weight")
    if s == SLOT_W1:
        return p + String(".feed_forward.w1.weight")
    if s == SLOT_W3:
        return p + String(".feed_forward.w3.weight")
    if s == SLOT_W2:
        return p + String(".feed_forward.w2.weight")
    raise Error("zimage_fullft_slot_key: bad slot")


# ── fast requant: binary search over the SORTED qmap ─────────────────────────
# Equivalent to adamw8bit.mojo's linear argmin scan: lowest code wins ties
# (the ascending scan's strict-< keeps the first/lowest index); nonfinite
# values (NaN/±inf) code to 0 (the linear scan's comparisons are all-false
# for NaN and never-strictly-better for inf, leaving best=0).
def _q_code_fast(qmap: List[Float32], x: Float32) -> Int:
    if not (x - x == Float32(0.0)):
        return 0
    var lo = 0
    var hi = 255
    while lo < hi:
        var mid = (lo + hi) // 2
        if qmap[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return 0
    var d1 = x - qmap[lo - 1]
    var d2 = qmap[lo] - x
    if d2 < d1:
        return lo
    return lo - 1


def _pow_f32(base: Float32, e: Int) -> Float32:
    var out = Float32(1.0)
    for _ in range(e):
        out = out * base
    return out


@fieldwise_init
struct FullFTStepStats(Copyable, Movable):
    var upd_l1: Float64    # Σ|p_new - p_old| over the slot
    var upd_max: Float32   # max|p_new - p_old|


# One bnb-parity 8-bit AdamW step over a master param list, grads read from a
# bf16 pointer with `gscale` (the global-clip factor) folded in. Per-element
# math is a verbatim copy of adam8bit_step_bnb; ONLY the requant argmin is the
# binary search (equivalence gated at run start).
# @no_inline: the full-FT step calls this from a parallel closure; pinning
# ONE compiled body guarantees the run-start equivalence gate (which calls it
# from a plain def) exercises the same code the training loop runs. (Measured:
# a pointer/closure re-body of this loop picked up FMA contraction and broke
# bit-equality vs adam8bit_step_bnb at step 2 — m_absmax 1-ulp shifts.)
@no_inline
def _adam8bit_step_fast(
    mut p: List[Float32],
    gp: UnsafePointer[BFloat16, MutAnyOrigin],
    n: Int,
    gscale: Float32,
    mut state: Adam8bitState,
    qmap_signed: List[Float32],
    qmap_unsigned: List[Float32],
    step: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
) raises -> FullFTStepStats:
    if len(p) != n or state.n != n:
        raise Error("_adam8bit_step_fast: p/state length != n")
    if step < 1:
        raise Error("_adam8bit_step_fast: step must be >= 1")
    var n_blocks = (n + ADAMW8BIT_BLOCK_SIZE - 1) // ADAMW8BIT_BLOCK_SIZE
    if len(state.m_absmax) < n_blocks or len(state.v_absmax) < n_blocks:
        raise Error("_adam8bit_step_fast: absmax buffers too small")

    var bc1 = Float32(1.0) - _pow_f32(beta1, step)
    var bc2 = Float32(1.0) - _pow_f32(beta2, step)
    var one_m_b1 = Float32(1.0) - beta1
    var one_m_b2 = Float32(1.0) - beta2

    var upd_l1 = 0.0
    var upd_max = Float32(0.0)

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

        for t in range(cnt):
            var i = base + t
            var gv = gp[i].cast[DType.float32]() * gscale
            var m_old = qmap_signed[Int(state.m_codes[i])] * am_prev
            var v_old = qmap_unsigned[Int(state.v_codes[i])] * av_prev
            var mn = beta1 * m_old + one_m_b1 * gv
            var vn = beta2 * v_old + one_m_b2 * gv * gv
            m_new[t] = mn
            v_new[t] = vn
            var m_hat = mn / bc1
            var v_hat = vn / bc2
            var upd = lr * m_hat / (sqrt(v_hat) + eps)
            var p_old = p[i]
            var pv = p_old - upd
            if weight_decay != Float32(0.0):
                pv = pv - lr * weight_decay * pv  # decoupled, AFTER the update
            p[i] = pv
            var d = pv - p_old
            if d < Float32(0.0):
                d = -d
            upd_l1 += Float64(d)
            if d > upd_max:
                upd_max = d

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
            amm = Float32(1.0e-12)
        if amv == Float32(0.0):
            amv = Float32(1.0e-12)
        state.m_absmax[blk] = amm
        state.v_absmax[blk] = amv

        for t in range(cnt):
            var i = base + t
            state.m_codes[i] = UInt8(_q_code_fast(qmap_signed, m_new[t] / amm))
            state.v_codes[i] = UInt8(_q_code_fast(qmap_unsigned, v_new[t] / amv))

    return FullFTStepStats(upd_l1, upd_max)


# ── run-start equivalence gate: fast step == parity-gated reference, BIT-EQUAL.
def adam8bit_fast_equivalence_gate() raises:
    var qs = adam8bit_create_dynamic_map(True)
    var qu = adam8bit_create_dynamic_map(False)
    var n = 4099  # tail block included
    var gscale = Float32(0.73)

    # deterministic LCG data: params, bf16 grads (incl. a zero block).
    var p_ref = List[Float32](capacity=n)
    var g_bf = List[BFloat16](capacity=n)
    var state_lcg = UInt64(0x9E3779B97F4A7C15)
    for i in range(n):
        state_lcg = state_lcg * 6364136223846793005 + 1442695040888963407
        var u = Float64(Int(state_lcg >> 11)) * (1.0 / 9007199254740992.0)
        p_ref.append(Float32((u - 0.5) * 4.0))
        state_lcg = state_lcg * 6364136223846793005 + 1442695040888963407
        var v = Float64(Int(state_lcg >> 11)) * (1.0 / 9007199254740992.0)
        if i >= 512 and i < 768:
            g_bf.append(BFloat16(Float32(0.0)))  # all-zero block
        else:
            g_bf.append(BFloat16(Float32((v - 0.5) * 0.02)))
    var p_fast = p_ref.copy()

    var st_ref = Adam8bitState(n)
    var st_fast = Adam8bitState(n)

    # two steps to exercise nonzero before-state.
    for k in range(1, 3):
        var g_f32 = List[Float32](capacity=n)
        for i in range(n):
            g_f32.append(g_bf[i].cast[DType.float32]() * gscale)
        adam8bit_step_bnb(
            p_ref, g_f32, st_ref, qs, qu, k,
            Float32(1.0e-5), Float32(0.9), Float32(0.999),
            Float32(1.0e-8), Float32(0.01),
        )
        _ = _adam8bit_step_fast(
            p_fast, g_bf.unsafe_ptr(), n, gscale, st_fast, qs, qu, k,
            Float32(1.0e-5), Float32(0.9), Float32(0.999),
            Float32(1.0e-8), Float32(0.01),
        )
        for i in range(n):
            if st_ref.m_codes[i] != st_fast.m_codes[i]:
                raise Error("adam8bit fast gate: m_code mismatch")
            if st_ref.v_codes[i] != st_fast.v_codes[i]:
                raise Error("adam8bit fast gate: v_code mismatch")
            if p_ref[i] != p_fast[i]:
                raise Error("adam8bit fast gate: param mismatch")
        var nb = (n + ADAMW8BIT_BLOCK_SIZE - 1) // ADAMW8BIT_BLOCK_SIZE
        for b in range(nb):
            if st_ref.m_absmax[b] != st_fast.m_absmax[b]:
                raise Error("adam8bit fast gate: m_absmax mismatch")
            if st_ref.v_absmax[b] != st_fast.v_absmax[b]:
                raise Error("adam8bit fast gate: v_absmax mismatch")
    print("[fullft] adam8bit fast-requant equivalence gate: PASS (codes/absmax/params bit-equal, 2 steps, n=4099)")


# ── full-FT optimizer state (host F32 masters + 8-bit moments per slot) ──────
struct ZImageFullFTOpt(Movable):
    var masters: List[List[Float32]]   # num_main * ZIMAGE_SLOTS, F32
    var states: List[Adam8bitState]
    var qmap_signed: List[Float32]
    var qmap_unsigned: List[Float32]
    var slot_numels: List[Int]
    var num_main: Int
    var staging: HostBuffer[DType.uint8]   # bf16 write-back staging (max slot)

    def __init__(
        out self,
        var masters: List[List[Float32]],
        var states: List[Adam8bitState],
        var qmap_signed: List[Float32], var qmap_unsigned: List[Float32],
        var slot_numels: List[Int], num_main: Int,
        var staging: HostBuffer[DType.uint8],
    ):
        self.masters = masters^
        self.states = states^
        self.qmap_signed = qmap_signed^
        self.qmap_unsigned = qmap_unsigned^
        self.slot_numels = slot_numels^
        self.num_main = num_main
        self.staging = staging^


def zimage_full_ft_opt_init(
    main_blocks: List[ZImageBlockWeights], D: Int, F: Int, ctx: DeviceContext
) raises -> ZImageFullFTOpt:
    adam8bit_fast_equivalence_gate()
    var t0 = perf_counter_ns()
    var slot_numels = zimage_fullft_slot_numels(D, F)
    var masters = List[List[Float32]]()
    var states = List[Adam8bitState]()
    var num_main = len(main_blocks)
    var total = 0
    var max_n = 0
    for bi in range(num_main):
        for s in range(ZIMAGE_SLOTS):
            var w_t = zimage_fullft_slot_tensor(main_blocks[bi], s)
            if w_t[].dtype() != STDtype.BF16:
                raise Error("fullft opt init: base weight must be BF16-resident")
            var n = w_t[].numel()
            if n != slot_numels[s]:
                raise Error("fullft opt init: slot numel mismatch")
            masters.append(w_t[].to_host(ctx))   # bf16 -> F32 exact upcast
            states.append(Adam8bitState(n))
            total += n
            if n > max_n:
                max_n = n
    var staging = ctx.enqueue_create_host_buffer[DType.uint8](max_n * 2)
    var secs = Float64(perf_counter_ns() - t0) / 1.0e9
    print(
        "[fullft] optimizer init: ", num_main * ZIMAGE_SLOTS,
        " slots, ", total, " trained params (",
        Float32(Float64(total) * 4.0 / 1.0e9), " GB F32 masters + ",
        Float32(Float64(total) * 2.0 / 1.0e9), " GB 8-bit moment codes host), ",
        Float32(secs), " s",
    )
    return ZImageFullFTOpt(
        masters^, states^,
        adam8bit_create_dynamic_map(True), adam8bit_create_dynamic_map(False),
        slot_numels^, num_main, staging,
    )


# global grad L2 norm over every trained d_W (F64 accumulate). Fails loud on a
# nonfinite norm (NaN/inf grads).
def zimage_full_ft_grad_norm(grads: ZImageFullFTGrads) raises -> Float64:
    var ss = 0.0
    var per_block_elems = 0
    for s in range(ZIMAGE_SLOTS):
        per_block_elems += grads.slot_numels[s]
    for j in range(len(grads.bufs_rev)):
        var gp = grads.bufs_rev[j].unsafe_ptr().bitcast[BFloat16]()
        for i in range(per_block_elems):
            var g = Float64(gp[i].cast[DType.float32]())
            ss += g * g
    var gn = sqrt(ss)
    if not (gn - gn == 0.0):
        raise Error("zimage_full_ft_grad_norm: nonfinite grad norm")
    return gn


# One full-FT optimizer step: per slot — host 8-bit AdamW on the F32 master
# (clip scale folded into the grad read), then RNE bf16 write-back into the
# resident device weight buffer (the next step's forward weights).
def zimage_full_ft_step(
    mut opt: ZImageFullFTOpt,
    grads: ZImageFullFTGrads,
    gscale: Float32,
    k: Int,
    lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
    main_blocks: List[ZImageBlockWeights],
    ctx: DeviceContext,
) raises -> FullFTStepStats:
    if grads.num_main != opt.num_main:
        raise Error("zimage_full_ft_step: block count mismatch")
    var n_slots = opt.num_main * ZIMAGE_SLOTS

    # Phase 1 — host 8-bit AdamW, PARALLEL over the 210 independent slots
    # (disjoint masters/moment-state/grad ranges; no device interaction).
    # Each worker calls the SAME @no_inline _adam8bit_step_fast body the
    # run-start equivalence gate bit-compares against adam8bit_step_bnb, so
    # the math is bit-identical to the sequential loop. (Sequential measured
    # 384.8 s/step on 5.31B params — the v1 smoke blocker.)
    var gaddrs = List[Int](capacity=n_slots)
    var l1s = List[Float64](capacity=n_slots)
    var mxs = List[Float32](capacity=n_slots)
    var errs = List[Int](capacity=n_slots)
    for bi in range(opt.num_main):
        var bj = grads.buf_index(bi)
        for s in range(ZIMAGE_SLOTS):
            var idx = bi * ZIMAGE_SLOTS + s
            if opt.states[idx].step != k - 1:
                raise Error("zimage_full_ft_step: optimizer step desync")
            gaddrs.append(
                Int(grads.bufs_rev[bj].unsafe_ptr() + grads.slot_offsets[s])
            )
            l1s.append(0.0)
            mxs.append(Float32(0.0))
            errs.append(0)
    var masters_p = opt.masters.unsafe_ptr()
    var states_p = opt.states.unsafe_ptr()
    var slotn_p = opt.slot_numels.unsafe_ptr()
    var ga_p = gaddrs.unsafe_ptr()
    var l1_p = l1s.unsafe_ptr()
    var mx_p = mxs.unsafe_ptr()
    var er_p = errs.unsafe_ptr()
    var qsl = opt.qmap_signed.copy()
    var qul = opt.qmap_unsigned.copy()

    @parameter
    fn _slot_worker(w: Int):
        var s = w % ZIMAGE_SLOTS
        var n = slotn_p[s]
        var gp = UnsafePointer[BFloat16, MutAnyOrigin](
            unsafe_from_address=ga_p[w]
        )
        try:
            var st = _adam8bit_step_fast(
                masters_p[w], gp, n, gscale, states_p[w],
                qsl, qul, k, lr, beta1, beta2, eps, weight_decay,
            )
            states_p[w].step = k
            l1_p[w] = st.upd_l1
            mx_p[w] = st.upd_max
        except:
            er_p[w] = 1

    parallelize[_slot_worker](n_slots)

    var upd_l1 = 0.0
    var upd_max = Float32(0.0)
    for w in range(n_slots):
        if errs[w] != 0:
            raise Error("zimage_full_ft_step: slot optimizer step failed")
        upd_l1 += l1s[w]
        if mxs[w] > upd_max:
            upd_max = mxs[w]

    # Phase 2 — RNE bf16 write-back into the live device weight buffers.
    # Sequential per slot (single DeviceContext), parallel chunked f32->bf16
    # conversion (independent per element, bit-safe).
    var sp = opt.staging.unsafe_ptr().bitcast[BFloat16]()
    for bi in range(opt.num_main):
        for s in range(ZIMAGE_SLOTS):
            var idx = bi * ZIMAGE_SLOTS + s
            var n = opt.slot_numels[s]
            var mp = opt.masters[idx].unsafe_ptr()

            @parameter
            fn _conv_worker(c: Int):
                var s0 = c * 262144
                var e0 = min(s0 + 262144, n)
                for i in range(s0, e0):
                    sp[i] = BFloat16(mp[i])

            parallelize[_conv_worker]((n + 262143) // 262144)
            var w_t = zimage_fullft_slot_tensor(main_blocks[bi], s)
            if w_t[].nbytes() != n * 2:
                raise Error("zimage_full_ft_step: device weight size mismatch")
            var src = opt.staging.create_sub_buffer[DType.uint8](0, n * 2)
            ctx.enqueue_copy(dst_buf=w_t[].buf, src_buf=src)
            ctx.synchronize()
    return FullFTStepStats(upd_l1, upd_max)


# ── full-checkpoint save (SOURCE key schema) ─────────────────────────────────
def _sorted_strings(var names: List[String]) -> List[String]:
    for i in range(1, len(names)):
        var v = names[i].copy()
        var j = i - 1
        while j >= 0 and names[j] > v:
            names[j + 1] = names[j].copy()
            j -= 1
        names[j + 1] = v.copy()
    return names^


def _pwrite_all(fd: Int, addr: Int, count: Int, offset: Int) raises:
    var done = 0
    while done < count:
        var chunk = count - done
        if chunk > 1 << 30:
            chunk = 1 << 30
        var w = sys_pwrite(
            fd, BytePtr(unsafe_from_address=addr + done), chunk, offset + done
        )
        if w != chunk:
            raise Error("fullft save: short write")
        done += chunk


# Write the full transformer checkpoint (ALL source keys, source dtypes/shapes;
# trained keys = bf16 image of the masters, frozen keys byte-copied from the
# source shards) as out_dir/diffusion_pytorch_model.safetensors + index.json +
# config.json — ShardedSafeTensors.open(out_dir) / the zimage loaders open it
# directly. Prints the trained-weight delta vs the source (gate b evidence).
def zimage_full_ft_save_checkpoint(
    src_dir: String, out_dir: String, opt: ZImageFullFTOpt, ctx: DeviceContext
) raises:
    var t0 = perf_counter_ns()
    var st = ShardedSafeTensors.open(src_dir)
    var names = _sorted_strings(st.names())
    if len(names) == 0:
        raise Error("fullft save: source has no tensors")

    var trained = Dict[String, Int]()   # key -> master index
    for bi in range(opt.num_main):
        for s in range(ZIMAGE_SLOTS):
            trained[zimage_fullft_slot_key(bi, s)] = bi * ZIMAGE_SLOTS + s
    var n_trained_found = 0

    # header JSON + offsets (data laid out in sorted-name order).
    var header = String("{")
    var offsets = List[Int]()
    var cursor = 0
    for i in range(len(names)):
        var info = st.tensor_info(names[i])
        var nbytes = info.size
        offsets.append(cursor)
        if i > 0:
            header += String(",")
        header += String('"') + names[i] + String('":{"dtype":"')
        header += info.dtype.name()
        header += String('","shape":[')
        for d in range(len(info.shape)):
            if d > 0:
                header += String(",")
            header += String(info.shape[d])
        header += String('],"data_offsets":[')
        header += String(cursor) + String(",") + String(cursor + nbytes)
        header += String("]}")
        cursor += nbytes
    header += String("}")
    var total_data = cursor

    _ = sys_system(String("mkdir -p ") + out_dir)
    var out_path = out_dir + String("/diffusion_pytorch_model.safetensors")
    var fd = sys_open(out_path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("fullft save: cannot create ") + out_path)

    # u64 LE header length + header JSON.
    var hlen = header.byte_length()
    var head = alloc[UInt8](8 + hlen)
    var hl = UInt64(hlen)
    for b in range(8):
        head[b] = UInt8((hl >> (8 * b)) & 0xFF)
    var hb = header.as_bytes()
    for b in range(hlen):
        head[8 + b] = hb[b]
    _pwrite_all(fd, Int(head), 8 + hlen, 0)
    head.free()
    var data_base = 8 + hlen

    # delta stats vs source (trained keys only).
    var delta_l1 = 0.0
    var delta_max = Float32(0.0)
    var delta_nonzero = 0
    var trained_elems = 0

    var max_n = 0
    for s in range(ZIMAGE_SLOTS):
        if opt.slot_numels[s] > max_n:
            max_n = opt.slot_numels[s]
    var stage = alloc[UInt8](max_n * 2)

    for i in range(len(names)):
        var name = names[i]
        var src = st.tensor_bytes(name)
        var nbytes = len(src)
        if name in trained:
            var idx = trained[name]
            n_trained_found += 1
            var n = opt.slot_numels[
                idx % ZIMAGE_SLOTS
            ]
            if nbytes != n * 2:
                raise Error(String("fullft save: trained key size mismatch: ") + name)
            var sp = stage.bitcast[BFloat16]()
            var op = src.unsafe_ptr().bitcast[BFloat16]()
            for e in range(n):
                var nv = BFloat16(opt.masters[idx][e])
                sp[e] = nv
                var d = nv.cast[DType.float32]() - op[e].cast[DType.float32]()
                if d < Float32(0.0):
                    d = -d
                delta_l1 += Float64(d)
                if d > delta_max:
                    delta_max = d
                if d > Float32(0.0):
                    delta_nonzero += 1
            trained_elems += n
            _pwrite_all(fd, Int(stage), nbytes, data_base + offsets[i])
        else:
            _pwrite_all(
                fd, Int(src.unsafe_ptr()), nbytes, data_base + offsets[i]
            )
    stage.free()
    _ = sys_close(fd)
    if n_trained_found != opt.num_main * ZIMAGE_SLOTS:
        raise Error("fullft save: not every trained key found in source schema")

    # index.json (all keys -> the single shard) + config.json copy.
    var idxj = String('{"metadata":{"total_size":') + String(total_data)
    idxj += String('},"weight_map":{')
    for i in range(len(names)):
        if i > 0:
            idxj += String(",")
        idxj += String('"') + names[i]
        idxj += String('":"diffusion_pytorch_model.safetensors"')
    idxj += String("}}")
    var idx_path = out_dir + String("/diffusion_pytorch_model.safetensors.index.json")
    var fdi = sys_open(idx_path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fdi < 0:
        raise Error("fullft save: cannot create index.json")
    var ib = idxj.as_bytes()
    var ibuf = alloc[UInt8](idxj.byte_length())
    for b in range(idxj.byte_length()):
        ibuf[b] = ib[b]
    _pwrite_all(fdi, Int(ibuf), idxj.byte_length(), 0)
    ibuf.free()
    _ = sys_close(fdi)
    _ = sys_system(
        String("cp -f ") + src_dir + String("/config.json ") + out_dir
        + String("/config.json")
    )

    var secs = Float64(perf_counter_ns() - t0) / 1.0e9
    print(
        "[fullft] checkpoint saved: ", out_path, " (", len(names), " keys, ",
        total_data, " data bytes, ", Float32(secs), " s)",
    )
    print(
        "[fullft] trained-weight delta vs source: nonzero_elems=", delta_nonzero,
        " /", trained_elems, " |Δ|_1=", Float32(delta_l1),
        " max|Δ|=", delta_max,
    )
