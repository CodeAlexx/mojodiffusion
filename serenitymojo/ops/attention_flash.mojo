# ops/attention_flash.mojo — cuDNN v9 Flash SDPA (training fwd + bwd) via the
# flame-core shim, ported 2026-06-11 after Alex's numerics sign-off (memory:
# sdpa-flash-signoff; HANDOFF_2026-06-11_OVERNIGHT_OT_PARITY.md §3.5).
#
# C side: serenitymojo/ops/cshim/cudnn_sdpa{,_bwd}.cpp — byte-copies of
# flame-core/src/cuda/cudnn_sdpa{,_bwd}.cpp (the production kernels flame
# trains with), compiled to ops/cshim/lib/libserenity_cudnn_sdpa.so by
# ops/cshim/build.sh and linked into binaries with
#   -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
# ("C for the gaps, Mojo for the rest" — the httpserver cshim pattern).
#
# Layout bridge: serenitymojo SDPA tensors are [B, S, H, Dh] contiguous BF16;
# the shim wants logical [B, H, N, D] + per-tensor 4-element ELEMENT-stride
# vectors. The same memory is described with strides {S_eff*H*Dh, Dh, H*Dh, 1}
# — no transpose copies. Stats is FP32 [B, H, N_q, 1], layout FIXED at strides
# {H*N_q, N_q, 1, 1} (the shim contract, flame ffi.rs:836-845).
#
# Padding (flame sdpa.rs maybe_pad_for_cudnn, ALIGN=128 — the post-misalign
# guard, BACKLOG_qwen_cudnn_sdpa_bwd_misalign.md): cuDNN's flash bwd wants
# 128-aligned sequence lengths. S not 128-aligned (zimage S=1248) is padded to
# S_PAD with ZERO rows per batch; the REAL length goes to the shim's
# real_N_q/real_N_kv, which build padding masks (SEQ_LEN tensors) inside the
# graph, so padded rows never contribute. Klein S=1536 is already aligned
# (S_PAD == S: no copies, no masks).
#
# NUMERICS: flash is a DIFFERENT summation order than the math-mode
# sdpa_nomask path — bit-equality with the old anchors is impossible BY
# DESIGN (approved). Gate: tests/sdpa_flash_parity.mojo measures flash-vs-math
# agreement AND both paths against an F32 reference on real shapes.
#
# Mojo 1.0.0b1, Linux x86-64, NVIDIA sm_86+, cuDNN v9 (pip wheel libs).

from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUDA, CUstream
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr


def _dev_ptr(t: Tensor) -> BytePtr:
    return BytePtr(unsafe_from_address=Int(t.buf.unsafe_ptr()))


def _strides_bhnd(n_eff: Int, h: Int, dh: Int) -> UnsafePointer[Int64, MutAnyOrigin]:
    """Element strides describing our [B, n_eff, H, Dh] contiguous memory as
    the shim's logical [B, H, N, D]: {n_eff*H*Dh, Dh, H*Dh, 1}."""
    var s = alloc[Int64](4)
    s[0] = Int64(n_eff * h * dh)
    s[1] = Int64(dh)
    s[2] = Int64(h * dh)
    s[3] = Int64(1)
    return s


def _stats_strides(n_q: Int, h: Int) -> UnsafePointer[Int64, MutAnyOrigin]:
    """Stats layout fixed by the shim: [B, H, N_q, 1] strides
    {H*N_q, N_q, 1, 1}."""
    var s = alloc[Int64](4)
    s[0] = Int64(h * n_q)
    s[1] = Int64(n_q)
    s[2] = Int64(1)
    s[3] = Int64(1)
    return s


def _pad_seq[
    B: Int, S: Int, S_PAD: Int
](t: Tensor, h: Int, dh: Int, ctx: DeviceContext) raises -> Tensor:
    """[B,S,H,Dh] bf16 -> [B,S_PAD,H,Dh] bf16 with ZERO pad rows per batch.
    One memset + B sub-buffer d2d copies; identity (refcount view of the same
    tensor) when S_PAD == S."""
    comptime if S_PAD == S:
        return Tensor(t.buf.copy(), t.shape(), t.dtype())
    var row = h * dh * 2  # bf16 bytes per sequence row
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * S_PAD * row)
    ctx.enqueue_memset[DType.uint8](out_buf, 0)
    for b in range(B):
        var src = t.buf.create_sub_buffer[DType.uint8](b * S * row, S * row)
        var dst = out_buf.create_sub_buffer[DType.uint8](b * S_PAD * row, S * row)
        ctx.enqueue_copy(dst_buf=dst, src_buf=src)
    var shape: List[Int] = [B, S_PAD, h, dh]
    return Tensor(out_buf^, shape^, STDtype.BF16)


def _unpad_seq[
    B: Int, S: Int, S_PAD: Int
](t: Tensor, h: Int, dh: Int, ctx: DeviceContext) raises -> Tensor:
    """[B,S_PAD,H,Dh] -> [B,S,H,Dh]: refcount view when no padding; B=1 is a
    zero-copy sub-buffer view (caller must keep the padded owner alive —
    scratch-ring view rule); B>1 compacts via B d2d copies."""
    comptime if S_PAD == S:
        return Tensor(t.buf.copy(), t.shape(), t.dtype())
    var row = h * dh * 2
    var shape: List[Int] = [B, S, h, dh]
    comptime if B == 1:
        var view = t.buf.create_sub_buffer[DType.uint8](0, S * row)
        return Tensor(view^, shape^, STDtype.BF16)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](B * S * row)
    for b in range(B):
        var src = t.buf.create_sub_buffer[DType.uint8](b * S_PAD * row, S * row)
        var dst = out_buf.create_sub_buffer[DType.uint8](b * S * row, S * row)
        ctx.enqueue_copy(dst_buf=dst, src_buf=src)
    return Tensor(out_buf^, shape^, STDtype.BF16)


struct SdpaFlashFwd(Movable):
    """Train-forward result. `o` is the consumer-facing [B,S,H,Dh] output
    (a zero-copy view into `o_pad` when padding was needed at B=1 — keep this
    struct alive while `o` is in use). The padded q/k/v/o + stats are the
    saved set the backward consumes (no re-padding in bwd)."""

    var o: Tensor
    var o_pad: Tensor
    var q_pad: Tensor
    var k_pad: Tensor
    var v_pad: Tensor
    var stats: Tensor

    def __init__(
        out self,
        var o: Tensor,
        var o_pad: Tensor,
        var q_pad: Tensor,
        var k_pad: Tensor,
        var v_pad: Tensor,
        var stats: Tensor,
    ):
        self.o = o^
        self.o_pad = o_pad^
        self.q_pad = q_pad^
        self.k_pad = k_pad^
        self.v_pad = v_pad^
        self.stats = stats^


struct SdpaFlashGrads(Movable):
    var d_q: Tensor
    var d_k: Tensor
    var d_v: Tensor

    def __init__(out self, var d_q: Tensor, var d_k: Tensor, var d_v: Tensor):
        self.d_q = d_q^
        self.d_k = d_k^
        self.d_v = d_v^


def sdpa_flash_train_fwd[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor, k: Tensor, v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaFlashFwd:
    """cuDNN flash SDPA training forward on [B,S,H,Dh] BF16 (no mask,
    non-causal — the DiT trainer call). Returns O + the softmax Stats (LSE)
    the flash backward needs. FAIL-LOUD on any nonzero shim rc."""
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16 or v.dtype() != STDtype.BF16:
        raise Error("sdpa_flash_train_fwd: q/k/v must be BF16")
    comptime S_PAD = ((S + 127) // 128) * 128

    var q_pad = _pad_seq[B, S, S_PAD](q, H, Dh, ctx)
    var k_pad = _pad_seq[B, S, S_PAD](k, H, Dh, ctx)
    var v_pad = _pad_seq[B, S, S_PAD](v, H, Dh, ctx)

    var o_buf = ctx.enqueue_create_buffer[DType.uint8](B * S_PAD * H * Dh * 2)
    ctx.enqueue_memset[DType.uint8](o_buf, 0)
    var o_shape: List[Int] = [B, S_PAD, H, Dh]
    var o_pad = Tensor(o_buf^, o_shape^, STDtype.BF16)

    var stats_buf = ctx.enqueue_create_buffer[DType.uint8](B * H * S_PAD * 4)
    ctx.enqueue_memset[DType.uint8](stats_buf, 0)
    var stats_shape: List[Int] = [B, H, S_PAD, 1]
    var stats = Tensor(stats_buf^, stats_shape^, STDtype.F32)

    var qs = _strides_bhnd(S_PAD, H, Dh)
    var ks = _strides_bhnd(S_PAD, H, Dh)
    var vs = _strides_bhnd(S_PAD, H, Dh)
    var os_ = _strides_bhnd(S_PAD, H, Dh)
    var stream = CUDA(ctx.stream())
    var rc = Int(external_call["flame_cudnn_sdpa_bf16_train_fwd", Int32](
        _dev_ptr(q_pad), _dev_ptr(k_pad), _dev_ptr(v_pad),
        _dev_ptr(o_pad), _dev_ptr(stats),
        Int32(B), Int32(H), Int32(S_PAD), Int32(S_PAD), Int32(Dh),
        scale,
        qs, ks, vs, os_,
        Int64(0), Int64(0), Int64(0), Int64(0), Int64(0),
        Int32(0),            # causal = false
        Int32(S), Int32(S),  # real lengths -> padding mask when S_PAD > S
        stream,
    ))
    qs.free(); ks.free(); vs.free(); os_.free()
    if rc != 0:
        raise Error(
            String("sdpa_flash_train_fwd: shim rc=") + String(rc)
            + " (B=" + String(B) + " S=" + String(S) + " S_PAD=" + String(S_PAD)
            + " H=" + String(H) + " Dh=" + String(Dh) + ")"
        )

    var o = _unpad_seq[B, S, S_PAD](o_pad, H, Dh, ctx)
    return SdpaFlashFwd(o^, o_pad^, q_pad^, k_pad^, v_pad^, stats^)


def sdpa_flash_backward[
    B: Int, S: Int, H: Int, Dh: Int
](
    fwd: SdpaFlashFwd,
    d_out: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaFlashGrads:
    """cuDNN flash SDPA backward: dQ/dK/dV [B,S,H,Dh] BF16 from the saved
    padded train-forward set + upstream d_out [B,S,H,Dh]. FAIL-LOUD."""
    comptime S_PAD = ((S + 127) // 128) * 128
    var do_pad = _pad_seq[B, S, S_PAD](d_out, H, Dh, ctx)

    var nbytes = B * S_PAD * H * Dh * 2
    var dq_buf = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var dk_buf = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var dv_buf = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_memset[DType.uint8](dq_buf, 0)
    ctx.enqueue_memset[DType.uint8](dk_buf, 0)
    ctx.enqueue_memset[DType.uint8](dv_buf, 0)
    var g_shape: List[Int] = [B, S_PAD, H, Dh]
    var dq_pad = Tensor(dq_buf^, g_shape.copy(), STDtype.BF16)
    var dk_pad = Tensor(dk_buf^, g_shape.copy(), STDtype.BF16)
    var dv_pad = Tensor(dv_buf^, g_shape^, STDtype.BF16)

    var qs = _strides_bhnd(S_PAD, H, Dh)
    var ks = _strides_bhnd(S_PAD, H, Dh)
    var vs = _strides_bhnd(S_PAD, H, Dh)
    var os_ = _strides_bhnd(S_PAD, H, Dh)
    var dos = _strides_bhnd(S_PAD, H, Dh)
    var dqs = _strides_bhnd(S_PAD, H, Dh)
    var dks = _strides_bhnd(S_PAD, H, Dh)
    var dvs = _strides_bhnd(S_PAD, H, Dh)
    var stream = CUDA(ctx.stream())
    var rc = Int(external_call["flame_cudnn_sdpa_bwd_bf16", Int32](
        _dev_ptr(fwd.q_pad), _dev_ptr(fwd.k_pad), _dev_ptr(fwd.v_pad),
        _dev_ptr(fwd.o_pad), _dev_ptr(do_pad), _dev_ptr(fwd.stats),
        _dev_ptr(dq_pad), _dev_ptr(dk_pad), _dev_ptr(dv_pad),
        Int32(B), Int32(H), Int32(S_PAD), Int32(S_PAD), Int32(Dh),
        scale,
        qs, ks, vs, os_, dos, dqs, dks, dvs,
        Int64(0), Int64(0), Int64(0), Int64(0), Int64(0),
        Int64(0), Int64(0), Int64(0), Int64(0),
        Int32(0),            # causal = false
        Int32(S), Int32(S),  # real lengths
        stream,
    ))
    qs.free(); ks.free(); vs.free(); os_.free()
    dos.free(); dqs.free(); dks.free(); dvs.free()
    if rc != 0:
        raise Error(
            String("sdpa_flash_backward: shim rc=") + String(rc)
            + " (B=" + String(B) + " S=" + String(S) + " S_PAD=" + String(S_PAD)
            + " H=" + String(H) + " Dh=" + String(Dh) + ")"
        )

    var d_q = _unpad_seq[B, S, S_PAD](dq_pad, H, Dh, ctx)
    var d_k = _unpad_seq[B, S, S_PAD](dk_pad, H, Dh, ctx)
    var d_v = _unpad_seq[B, S, S_PAD](dv_pad, H, Dh, ctx)
    # B=1 unpad returns VIEWS into the padded grads; re-box as owning copies
    # so the result outlives the padded buffers (one d2d each, pad case only).
    comptime if B == 1 and S_PAD != S:
        var dq_o = ctx.enqueue_create_buffer[DType.uint8](B * S * H * Dh * 2)
        var dk_o = ctx.enqueue_create_buffer[DType.uint8](B * S * H * Dh * 2)
        var dv_o = ctx.enqueue_create_buffer[DType.uint8](B * S * H * Dh * 2)
        ctx.enqueue_copy(dst_buf=dq_o, src_buf=d_q.buf)
        ctx.enqueue_copy(dst_buf=dk_o, src_buf=d_k.buf)
        ctx.enqueue_copy(dst_buf=dv_o, src_buf=d_v.buf)
        var sh: List[Int] = [B, S, H, Dh]
        return SdpaFlashGrads(
            Tensor(dq_o^, sh.copy(), STDtype.BF16),
            Tensor(dk_o^, sh.copy(), STDtype.BF16),
            Tensor(dv_o^, sh^, STDtype.BF16),
        )
    return SdpaFlashGrads(d_q^, d_k^, d_v^)


# ─── F32-boundary helpers (Klein: F32 activations, approved bf16 flash) ──────
# Klein's trainer SDPA runs on F32 activations (the "9,216 F32 sgemms"
# attribution). The approved flash path casts q/k/v -> bf16 at the SDPA
# boundary, runs cuDNN flash, and casts O / dQ/dK/dV back to F32. ALIGNED
# SHAPES ONLY (S % 128 == 0 — Klein S=1536; fail-loud otherwise: no padding
# logic on this path). The bf16 q/k/v/o + stats are returned as TArcs for the
# saved tape; the backward consumes them without re-casting.

from std.memory import ArcPointer
from serenitymojo.ops.cast import cast_tensor

comptime TArc = ArcPointer[Tensor]


struct SdpaFlashF32Fwd(Movable):
    var att: Tensor      # [B,S,H,Dh] F32 — the math-path drop-in output
    var q_bf: TArc
    var k_bf: TArc
    var v_bf: TArc
    var o_bf: TArc
    var stats: TArc

    def __init__(
        out self,
        var att: Tensor, var q_bf: TArc, var k_bf: TArc, var v_bf: TArc,
        var o_bf: TArc, var stats: TArc,
    ):
        self.att = att^
        self.q_bf = q_bf^
        self.k_bf = k_bf^
        self.v_bf = v_bf^
        self.o_bf = o_bf^
        self.stats = stats^


def sdpa_flash_train_fwd_f32[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor, k: Tensor, v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaFlashF32Fwd:
    comptime if (S % 128) != 0:
        raise Error("sdpa_flash_train_fwd_f32: S must be 128-aligned")
    var q_bf = cast_tensor(q, STDtype.BF16, ctx)
    var k_bf = cast_tensor(k, STDtype.BF16, ctx)
    var v_bf = cast_tensor(v, STDtype.BF16, ctx)
    var fwd = sdpa_flash_train_fwd[B, S, H, Dh](q_bf, k_bf, v_bf, scale, ctx)
    var att = cast_tensor(fwd.o, STDtype.F32, ctx)
    return SdpaFlashF32Fwd(
        att^, TArc(q_bf^), TArc(k_bf^), TArc(v_bf^),
        TArc(Tensor(fwd.o_pad.buf.copy(), fwd.o_pad.shape(), fwd.o_pad.dtype())),
        TArc(Tensor(fwd.stats.buf.copy(), fwd.stats.shape(), fwd.stats.dtype())),
    )


def sdpa_flash_backward_f32[
    B: Int, S: Int, H: Int, Dh: Int
](
    q_bf: TArc, k_bf: TArc, v_bf: TArc, o_bf: TArc, stats: TArc,
    d_att: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> SdpaFlashGrads:
    """Aligned-only flash backward with F32 grads in/out. d_att F32 ->
    bf16; dQ/dK/dV bf16 -> F32 (the hand-chain consumes F32)."""
    comptime if (S % 128) != 0:
        raise Error("sdpa_flash_backward_f32: S must be 128-aligned")
    var do_bf = cast_tensor(d_att, STDtype.BF16, ctx)

    var nbytes = B * S * H * Dh * 2
    var dq_buf = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var dk_buf = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var dv_buf = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var g_shape: List[Int] = [B, S, H, Dh]
    var dq_bf = Tensor(dq_buf^, g_shape.copy(), STDtype.BF16)
    var dk_bf = Tensor(dk_buf^, g_shape.copy(), STDtype.BF16)
    var dv_bf = Tensor(dv_buf^, g_shape^, STDtype.BF16)

    var qs = _strides_bhnd(S, H, Dh)
    var ks = _strides_bhnd(S, H, Dh)
    var vs = _strides_bhnd(S, H, Dh)
    var os_ = _strides_bhnd(S, H, Dh)
    var dos = _strides_bhnd(S, H, Dh)
    var dqs = _strides_bhnd(S, H, Dh)
    var dks = _strides_bhnd(S, H, Dh)
    var dvs = _strides_bhnd(S, H, Dh)
    var stream = CUDA(ctx.stream())
    var rc = Int(external_call["flame_cudnn_sdpa_bwd_bf16", Int32](
        _dev_ptr(q_bf[]), _dev_ptr(k_bf[]), _dev_ptr(v_bf[]),
        _dev_ptr(o_bf[]), _dev_ptr(do_bf), _dev_ptr(stats[]),
        _dev_ptr(dq_bf), _dev_ptr(dk_bf), _dev_ptr(dv_bf),
        Int32(B), Int32(H), Int32(S), Int32(S), Int32(Dh),
        scale,
        qs, ks, vs, os_, dos, dqs, dks, dvs,
        Int64(0), Int64(0), Int64(0), Int64(0), Int64(0),
        Int64(0), Int64(0), Int64(0), Int64(0),
        Int32(0), Int32(S), Int32(S),
        stream,
    ))
    qs.free(); ks.free(); vs.free(); os_.free()
    dos.free(); dqs.free(); dks.free(); dvs.free()
    if rc != 0:
        raise Error(
            String("sdpa_flash_backward_f32: shim rc=") + String(rc)
            + " (B=" + String(B) + " S=" + String(S)
            + " H=" + String(H) + " Dh=" + String(Dh) + ")"
        )
    return SdpaFlashGrads(
        cast_tensor(dq_bf, STDtype.F32, ctx),
        cast_tensor(dk_bf, STDtype.F32, ctx),
        cast_tensor(dv_bf, STDtype.F32, ctx),
    )
