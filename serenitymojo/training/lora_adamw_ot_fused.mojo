# training/lora_adamw_ot_fused.mojo — GPU fused LoRA AdamW with OneTrainer
# semantics (BF16-quantized moments + stochastic-rounding writeback).
#
# WHY: the Klein product trainer's optimizer stage is a HOST scalar loop
# (`_adamw_host_list_precomputed` in models/klein/lora_adapter.mojo) over all
# adapter elements — MEASURED 2026-06-10 at 4.3-6.0 s of a 10.5 s step (57%),
# PROG_STAGE phase=optim. This file moves the identical per-element math to ONE
# GPU launch over all segments.
#
# CONTRACT (MEASURED 2026-06-10, see lora_adamw_ot_fused_parity.mojo):
#   * params (a/b): BIT-EQUAL to the host loop — zero mismatches over 3.1M
#     elements incl. adversarial binade-boundary values.
#   * m/v moments: equal except RNE midpoint ties flipped by device-vs-host
#     1-ulp arithmetic (codegen contraction/reassociation, NOT controllable
#     from source — explicit-fma host probe reproduced plain host, refuting
#     simple FMA contraction). Measured rate ~5e-6, ALWAYS ±1 bf16 quantum —
#     i.e. inside the noise floor OneTrainer accepts by quantizing moments to
#     bf16. Gate: params strict, m/v bounded ±1 quantum at rate < 1e-4.
# Per-element math mirrored from `_adamw_host_list_precomputed`:
#   pf = p[i](f32) * decay                      # decoupled WD, precomputed decay
#   mf = mf + (1-beta1)*(gv - mf)
#   vf = beta2*vf + (1-beta2)*gv*gv
#   m[i] = RNE_bf16(mf) as f32                  # OneTrainer BF16 moment storage
#   v[i] = RNE_bf16(vf) as f32
#   denom = sqrt(v_q)/bc2_sqrt + eps
#   newp = pf - step_size * m_q / denom
#   p[i] = SR_bf16(newp, sr_uniform(seed, i))   # i = INTRA-segment index
# The helpers are the very same defs the host path uses
# (ops/torch_bf16.torch_bf16_rne_value, util/bf16_stochastic_rounding._sr_bf16 /
# sr_uniform); torch_bf16_rne_value is already device-proven
# (ops/torch_bf16.mojo `_torch_f32_to_bf16_rne_kernel`). The SR uniform depends
# only on (seed, intra-segment index), so segment packing order cannot change
# results.
#
# GATE: training/lora_adamw_ot_fused_parity.mojo — host loop vs this kernel on
# identical data must be BIT-EQUAL (params, m, v) before any trainer wires this
# in. Plus the trainer's own deterministic loss regression (Klein step-2 loss
# reproduces exactly when the optimizer is bit-equal).
#
# Data movement (increment 1): adapters live in host Lists, so each step packs
# p/g/m/v into pinned staging via sys_memcpy (no per-element host math), ONE
# H2D per role, ONE launch, ONE D2H per role, memcpy back. Device-resident
# adapters (killing the round-trip entirely) is the next increment.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import floor, ldexp, sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.util.bf16_stochastic_rounding import sr_uniform


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── exact-exponent BF16 quantizers ───────────────────────────────────────────
# Same structure as ops/torch_bf16.torch_bf16_rne_value and
# util/bf16_stochastic_rounding._sr_bf16, but the binade exponent comes from an
# EXACT halving/doubling loop and step from ldexp (pure exponent bit-ops) —
# NOT from F64 log/pow. Reason (MEASURED 2026-06-10, parity gate v1): device
# libdevice log/pow are not correctly rounded like glibc's, which flipped RNE
# midpoint ties on ~4.5e-6 of realistic values (±1 bf16 quantum on v). glibc's
# correctly-rounded log/pow always yield the true binade + exact power-of-two
# step, so these exact versions are bit-equal to the HOST helpers — proven by
# the parity gate, which compares against the host originals.


def _binade_e(av: Float64) -> Int:
    # e with 2^e <= av < 2^(e+1). *0.5 / *2.0 are exact exponent shifts in F64.
    var e = 0
    var x = av
    while x >= Float64(2.0):
        x *= Float64(0.5)
        e += 1
    while x < Float64(1.0):
        x *= Float64(2.0)
        e -= 1
    return e


def _rne_bf16_exact(v: Float32) -> BFloat16:
    if not (v == v):
        return v.cast[DType.bfloat16]()
    if v == Float32(0.0):
        return BFloat16(0.0)
    var sign = Float32(1.0)
    var a = v
    if a < Float32(0.0):
        sign = Float32(-1.0)
        a = -a
    var av = Float64(a)
    if a < Float32(1.0e-38):
        return v.cast[DType.bfloat16]()
    var e = _binade_e(av)
    var step = ldexp(Float64(1.0), e - 7)
    var y = av / step
    var kf = floor(y)
    var frac = y - kf
    var k = Int(kf)
    if frac > Float64(0.5) or (frac == Float64(0.5) and (k & 1) != 0):
        k += 1
    var q = Float32(Float64(k) * step)
    if sign < Float32(0.0):
        q = -q
    return q.cast[DType.bfloat16]()


def _sr_bf16_exact(v: Float32, u: Float32) -> BFloat16:
    if not (v == v):
        return v.cast[DType.bfloat16]()
    if v == Float32(0.0):
        return BFloat16(0.0)
    var sign = Float32(1.0)
    var a = v
    if a < Float32(0.0):
        sign = Float32(-1.0)
        a = -a
    if a < Float32(1.0e-38):
        return v.cast[DType.bfloat16]()
    var av = Float64(a)
    var e = _binade_e(av)
    var step = ldexp(Float64(1.0), e - 7)
    var y = av / step
    var kf = floor(y)
    var frac = y - kf
    var k = Int(kf)
    if Float64(u) < frac:
        k += 1
    var q = Float32(Float64(k) * step)
    if sign < Float32(0.0):
        q = -q
    return q.cast[DType.bfloat16]()


def _lora_adamw_ot_kernel(
    p: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    m: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    offs: LayoutTensor[DType.int64, _DYN1, MutAnyOrigin],
    nseg: Int,
    total: Int,
    step_size: Float32,
    bc2_sqrt: Float32,
    decay: Float32,
    one_minus_beta1: Float32,
    beta2: Float32,
    one_minus_beta2: Float32,
    eps: Float32,
    seed_i: Int,
):
    var gid = Int(global_idx.x)
    if gid >= total:
        return
    # locate segment ti (largest ti with offs[ti] <= gid) for the SR stream's
    # intra-segment index. nseg is small (hundreds); linear scan like
    # fused_adamw_multitensor.mojo.
    var ti = 0
    while ti + 1 < nseg and Int(rebind[Scalar[DType.int64]](offs[ti + 1])) <= gid:
        ti += 1
    var j = gid - Int(rebind[Scalar[DType.int64]](offs[ti]))

    var pf = rebind[Scalar[DType.bfloat16]](p[gid]).cast[DType.float32]()
    var gv = rebind[Scalar[DType.float32]](g[gid])
    var mf = rebind[Scalar[DType.float32]](m[gid])
    var vf = rebind[Scalar[DType.float32]](v[gid])

    pf = pf * decay
    mf = mf + one_minus_beta1 * (gv - mf)
    vf = beta2 * vf + one_minus_beta2 * gv * gv

    var m_q = _rne_bf16_exact(mf)
    var v_q = _rne_bf16_exact(vf)
    var mfq = m_q.cast[DType.float32]()
    var vfq = v_q.cast[DType.float32]()
    m[gid] = rebind[m.element_type](mfq)
    v[gid] = rebind[v.element_type](vfq)

    var denom = sqrt(vfq) / bc2_sqrt + eps
    var newp = pf - step_size * mfq / denom

    var u = sr_uniform(UInt32(seed_i), j)
    p[gid] = rebind[p.element_type](_sr_bf16_exact(newp, u))


def fused_lora_adamw_ot_step(
    mut adapters: List[LoraAdapter],
    d_a: List[List[Float32]],
    d_b: List[List[Float32]],
    step_size: Float32,
    bc2_sqrt: Float32,
    decay: Float32,
    one_minus_beta1: Float32,
    beta2: Float32,
    one_minus_beta2: Float32,
    eps: Float32,
    seed: UInt32,
    ctx: DeviceContext,
) raises:
    """One fused OT-semantics AdamW step over ALL adapters' A and B params.
    Mutates adapters' a/b (BF16 SR writeback) and ma/va/mb/vb (BF16-quantized
    F32 moments) in place — bit-equal to looping `_lora_adamw_precomputed`."""
    var na = len(adapters)
    if na == 0:
        return
    if len(d_a) != na or len(d_b) != na:
        raise Error("fused_lora_adamw_ot_step: adapters/d_a/d_b length mismatch")

    # ── segment table: 2 segments per adapter (a then b) ─────────────────────
    var nseg = 2 * na
    var seg_len = List[Int]()
    var total = 0
    for i in range(na):
        var n_a = len(adapters[i].a)
        var n_b = len(adapters[i].b)
        if len(d_a[i]) != n_a or len(adapters[i].ma) != n_a or len(adapters[i].va) != n_a:
            raise Error(
                "fused_lora_adamw_ot_step: A-side len mismatch at adapter "
                + String(i)
            )
        if len(d_b[i]) != n_b or len(adapters[i].mb) != n_b or len(adapters[i].vb) != n_b:
            raise Error(
                "fused_lora_adamw_ot_step: B-side len mismatch at adapter "
                + String(i)
            )
        seg_len.append(n_a)
        seg_len.append(n_b)
        total += n_a + n_b
    if total == 0:
        return

    # ── pinned staging (pack via raw memcpy — NO per-element host math) ──────
    var host_p = ctx.enqueue_create_host_buffer[DType.uint8](total * 2)
    var host_g = ctx.enqueue_create_host_buffer[DType.uint8](total * 4)
    var host_m = ctx.enqueue_create_host_buffer[DType.uint8](total * 4)
    var host_v = ctx.enqueue_create_host_buffer[DType.uint8](total * 4)
    var host_off = ctx.enqueue_create_host_buffer[DType.uint8]((nseg + 1) * 8)

    var hp = Int(host_p.unsafe_ptr())
    var hg = Int(host_g.unsafe_ptr())
    var hm = Int(host_m.unsafe_ptr())
    var hv = Int(host_v.unsafe_ptr())
    var op = host_off.unsafe_ptr().bitcast[Int64]()

    var off = 0
    op[0] = Int64(0)
    for i in range(na):
        var n_a = seg_len[2 * i]
        var n_b = seg_len[2 * i + 1]
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].a.unsafe_ptr())),
            n_a * 2,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hg + off * 4),
            BytePtr(unsafe_from_address=Int(d_a[i].unsafe_ptr())),
            n_a * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hm + off * 4),
            BytePtr(unsafe_from_address=Int(adapters[i].ma.unsafe_ptr())),
            n_a * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hv + off * 4),
            BytePtr(unsafe_from_address=Int(adapters[i].va.unsafe_ptr())),
            n_a * 4,
        )
        off += n_a
        op[2 * i + 1] = Int64(off)
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].b.unsafe_ptr())),
            n_b * 2,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hg + off * 4),
            BytePtr(unsafe_from_address=Int(d_b[i].unsafe_ptr())),
            n_b * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hm + off * 4),
            BytePtr(unsafe_from_address=Int(adapters[i].mb.unsafe_ptr())),
            n_b * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hv + off * 4),
            BytePtr(unsafe_from_address=Int(adapters[i].vb.unsafe_ptr())),
            n_b * 4,
        )
        off += n_b
        op[2 * i + 2] = Int64(off)

    # ── H2D, launch, D2H ──────────────────────────────────────────────────────
    var dev_p = ctx.enqueue_create_buffer[DType.uint8](total * 2)
    var dev_g = ctx.enqueue_create_buffer[DType.uint8](total * 4)
    var dev_m = ctx.enqueue_create_buffer[DType.uint8](total * 4)
    var dev_v = ctx.enqueue_create_buffer[DType.uint8](total * 4)
    var dev_off = ctx.enqueue_create_buffer[DType.uint8]((nseg + 1) * 8)
    ctx.enqueue_copy(dst_buf=dev_p, src_buf=host_p)
    ctx.enqueue_copy(dst_buf=dev_g, src_buf=host_g)
    ctx.enqueue_copy(dst_buf=dev_m, src_buf=host_m)
    ctx.enqueue_copy(dst_buf=dev_v, src_buf=host_v)
    ctx.enqueue_copy(dst_buf=dev_off, src_buf=host_off)

    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nseg + 1))
    var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        dev_p.unsafe_ptr().bitcast[BFloat16](), t_rl
    )
    var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dev_g.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dev_m.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var V = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dev_v.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var OFF = LayoutTensor[DType.int64, _DYN1, MutAnyOrigin](
        dev_off.unsafe_ptr().bitcast[Int64](), o_rl
    )

    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_lora_adamw_ot_kernel, _lora_adamw_ot_kernel](
        P, G, M, V, OFF, nseg, total, step_size, bc2_sqrt, decay,
        one_minus_beta1, beta2, one_minus_beta2, eps, Int(seed),
        grid_dim=grid, block_dim=_BLOCK,
    )

    ctx.enqueue_copy(dst_buf=host_p, src_buf=dev_p)
    ctx.enqueue_copy(dst_buf=host_m, src_buf=dev_m)
    ctx.enqueue_copy(dst_buf=host_v, src_buf=dev_v)
    ctx.synchronize()

    # ── unpack back into the adapters' host lists ─────────────────────────────
    off = 0
    for i in range(na):
        var n_a = seg_len[2 * i]
        var n_b = seg_len[2 * i + 1]
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].a.unsafe_ptr())),
            BytePtr(unsafe_from_address=hp + off * 2),
            n_a * 2,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].ma.unsafe_ptr())),
            BytePtr(unsafe_from_address=hm + off * 4),
            n_a * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].va.unsafe_ptr())),
            BytePtr(unsafe_from_address=hv + off * 4),
            n_a * 4,
        )
        off += n_a
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].b.unsafe_ptr())),
            BytePtr(unsafe_from_address=hp + off * 2),
            n_b * 2,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].mb.unsafe_ptr())),
            BytePtr(unsafe_from_address=hm + off * 4),
            n_b * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].vb.unsafe_ptr())),
            BytePtr(unsafe_from_address=hv + off * 4),
            n_b * 4,
        )
        off += n_b
