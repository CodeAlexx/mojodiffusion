# serenitymojo/models/sd35/sd35_block_device.mojo
#
# DEVICE-RESIDENT SD3.5 JOINT LoRA block, fwd + bwd (MJ-1069 port, rung 1).
#
# Device transcription of the HOST sd35 joint block (sd35_block.mojo — the
# torch-gated oracle whose EVERY op is a host round trip and whose modulate/
# gated-residual/qkv-split/joint-concat are HOST SCALAR LOOPS; the stack casts
# every block weight to F32 HOST per visit — MJ-1069's ~55GB appetite, Large
# NEVER completed a step). Template: qwenimage_block_device.mojo (28/28
# max_abs=0.0) + SD35_LARGE_DEVICE_PORT_MAP_2026-07-06.md.
#
# NUMERICS (map contract):
#   * F32 device chain end-to-end (host is all-F32; elementwise host loops ==
#     the shared device ops' per-element F32 expressions).
#   * LoRA fwd/bwd are BF16 GEMMs inside the F32 chain (host _sd35_lora_fwd/
#     _sd35_lora_bwd upload F32 lists as BF16 => activations bf16-ROUND at the
#     LoRA boundary only): device mirrors with _bf16(x_f32) inputs, F32 scale,
#     F32 folds into the F32 base (host folds via _add_lists, no re-round).
#   * NO rope. qk RMS uses qk_eps (separate from ln eps). Joint attention
#     concat CTX FIRST. Backward recomputes q_rms/k_rms from saved q_pre/k_pre
#     (host does the same — sd35_block.mojo:1608-1611).
#   * frozen base/mod: device backward returns ONLY d_x per stream + the 16
#     LoRA d_A/d_B slots (host's d_w/d_b/mod grads dropped; dx-only ops).
#
# Mojo current beta: `def`, `comptime`, `var`, explicit raises.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear, linear_bias_scratch
from serenitymojo.ops.linalg_backward import (
    linear_backward_dx, linear_backward_dw, linear_backward_dx_scratch,
)

from serenitymojo.training.train_step import LoraAdapter

from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu, gelu_scratch
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_padmask_bf16, sdpa_flash_backward_padmask_bf16,
)
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, reshape_in_place, slice, concat, add, mul_scalar,
)
from serenitymojo.ops.norm_backward import (
    rms_norm_backward_dx, layer_norm_backward_dx,
)
from serenitymojo.ops.activation_backward import gelu_backward, gelu_backward_scratch
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import gate_residual_backward_dxdy
from serenitymojo.scratch_ring import ScratchRingAllocator

from serenitymojo.models.sd35.sd35_block import (
    ModVecs, StreamWeights, JointBlockWeights,
)


comptime TArc = ArcPointer[Tensor]

# ── scratch-ring MLP routing (audit item 5, MJ-1090) ─────────────────────────
# SD3.5-Large's MLP path allocates [N_IMG, MLP] F32 tensors (~159MB at 1024px)
# every block, every step — the measured ~495 driver allocs ≥128MB/step storm.
# When True, the *_scratch block variants route the X-stream (large-N) MLP
# transients through a caller-owned ScratchRingAllocator instead of per-op
# driver mallocs: fwd h1_base (linear_bias_scratch) + hg (gelu_scratch); bwd fc2
# base_dx (linear_backward_dx_scratch) + d_h1 (gelu_backward_scratch). Every
# routed op is BIT-IDENTICAL to its plain counterpart (only the allocation
# source changes). `hg` is the sole SAVED tensor placed in the ring; it survives
# the recompute-forward -> backward span because the conductor holds ONE ring
# frame per block iteration (see sd35_stack_lora.mojo). The plain block
# functions are retained unchanged and used by the parity gate; flip this to
# False to make the *_scratch variants fall back to the plain path verbatim.
comptime SD35_SCRATCH_MLP = True

# per-block LoRA slot order (ctx stream then x stream)
comptime SD35_SLOTS = 8
comptime SD_CQKV = 0
comptime SD_CPROJ = 1
comptime SD_CFC1 = 2
comptime SD_CFC2 = 3
comptime SD_XQKV = 4
comptime SD_XPROJ = 5
comptime SD_XFC1 = 6
comptime SD_XFC2 = 7


def _ones(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(1.0)
    return o^


def _zeros(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(0.0)
    return o^


def _t_f32(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _f32(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    if t.dtype() == STDtype.F32:
        return t.clone(ctx)
    return cast_tensor(t, STDtype.F32, ctx)


def _bf16(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    if t.dtype() == STDtype.BF16:
        return t.clone(ctx)
    return cast_tensor(t, STDtype.BF16, ctx)


# ═══════════════════════════════════════════════════════════════════════════
# Device carriers (weights F32 like the host's per-visit F32 casts; LoRA bf16)
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct SD35StreamWeightsDevice(Copyable, Movable):
    var wqkv: TArc     # [3D, D] F32
    var bqkv: TArc     # [3D]
    var wproj: TArc    # [D, D]
    var bproj: TArc    # [D]
    var wfc1: TArc     # [MLP, D]
    var bfc1: TArc     # [MLP]
    var wfc2: TArc     # [D, MLP]
    var bfc2: TArc     # [D]
    var q_norm: TArc   # [Dh]
    var k_norm: TArc   # [Dh]


def sd35_stream_weights_to_device(
    w: StreamWeights, D: Int, MLP: Int, Dh: Int, ctx: DeviceContext
) raises -> SD35StreamWeightsDevice:
    return SD35StreamWeightsDevice(
        TArc(_t_f32(w.wqkv.copy(), [3 * D, D], ctx)),
        TArc(_t_f32(w.bqkv.copy(), [3 * D], ctx)),
        TArc(_t_f32(w.wproj.copy(), [D, D], ctx)),
        TArc(_t_f32(w.bproj.copy(), [D], ctx)),
        TArc(_t_f32(w.wfc1.copy(), [MLP, D], ctx)),
        TArc(_t_f32(w.bfc1.copy(), [MLP], ctx)),
        TArc(_t_f32(w.wfc2.copy(), [D, MLP], ctx)),
        TArc(_t_f32(w.bfc2.copy(), [D], ctx)),
        TArc(_t_f32(w.q_norm.copy(), [Dh], ctx)),
        TArc(_t_f32(w.k_norm.copy(), [Dh], ctx)),
    )


@fieldwise_init
struct SD35JointBlockWeightsDevice(Copyable, Movable):
    var ctxw: SD35StreamWeightsDevice
    var xw: SD35StreamWeightsDevice


def sd35_joint_block_weights_to_device(
    w: JointBlockWeights, D: Int, MLP: Int, Dh: Int, ctx: DeviceContext
) raises -> SD35JointBlockWeightsDevice:
    return SD35JointBlockWeightsDevice(
        sd35_stream_weights_to_device(w.ctxw, D, MLP, Dh, ctx),
        sd35_stream_weights_to_device(w.xw, D, MLP, Dh, ctx),
    )


@fieldwise_init
struct SD35ModVecsDevice(Copyable, Movable):
    var shift_msa: TArc   # [D] F32
    var scale_msa: TArc
    var gate_msa: TArc
    var shift_mlp: TArc
    var scale_mlp: TArc
    var gate_mlp: TArc


def sd35_modvecs_to_device(mv: ModVecs, D: Int, ctx: DeviceContext) raises -> SD35ModVecsDevice:
    return SD35ModVecsDevice(
        TArc(_t_f32(mv.shift_msa.copy(), [D], ctx)),
        TArc(_t_f32(mv.scale_msa.copy(), [D], ctx)),
        TArc(_t_f32(mv.gate_msa.copy(), [D], ctx)),
        TArc(_t_f32(mv.shift_mlp.copy(), [D], ctx)),
        TArc(_t_f32(mv.scale_mlp.copy(), [D], ctx)),
        TArc(_t_f32(mv.gate_mlp.copy(), [D], ctx)),
    )


@fieldwise_init
struct SD35LoraAdapterDevice(Copyable, Movable):
    var a: TArc        # [rank, in_f]  bf16
    var b: TArc        # [out_f, rank] bf16
    var rank: Int
    var in_f: Int
    var out_f: Int
    var scale: Float32


def sd35_lora_adapter_to_device(
    lo: LoraAdapter, ctx: DeviceContext
) raises -> SD35LoraAdapterDevice:
    return SD35LoraAdapterDevice(
        TArc(Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)),
        TArc(Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)),
        lo.rank, lo.in_f, lo.out_f, lo.scale,
    )


def sd35_opt_lora_to_device(
    lo: Optional[LoraAdapter], ctx: DeviceContext
) raises -> Optional[SD35LoraAdapterDevice]:
    if lo:
        return Optional[SD35LoraAdapterDevice](sd35_lora_adapter_to_device(lo.value(), ctx))
    return Optional[SD35LoraAdapterDevice](None)


@fieldwise_init
struct SD35StreamLoraDevice(Copyable, Movable):
    var qkv: Optional[SD35LoraAdapterDevice]
    var proj: Optional[SD35LoraAdapterDevice]
    var fc1: Optional[SD35LoraAdapterDevice]
    var fc2: Optional[SD35LoraAdapterDevice]


@fieldwise_init
struct SD35JointBlockLoraDevice(Copyable, Movable):
    var ctxl: SD35StreamLoraDevice
    var xl: SD35StreamLoraDevice


# ═══════════════════════════════════════════════════════════════════════════
# Device saved activations (F32 TArcs; mirror host StreamSaved fields)
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct SD35StreamSavedDevice(Copyable, Movable):
    var s: TArc        # [N,D] block input
    var ln1: TArc      # [N,D]
    var norm: TArc     # [N,D]
    var q_pre: TArc    # [1,N,H,Dh]
    var k_pre: TArc    # [1,N,H,Dh]
    var v: TArc        # [1,N,H,Dh]
    var att: TArc      # [N,D]  per-stream attention slice
    var attn_res: TArc # [N,D]
    var ln2: TArc      # [N,D]
    var mlp_in: TArc   # [N,D]
    var h1: TArc       # [N,MLP] gelu input
    var hg: TArc       # [N,MLP] gelu output


@fieldwise_init
struct SD35JointBlockSavedDevice(Copyable, Movable):
    var ctx_saved: SD35StreamSavedDevice
    var x_saved: SD35StreamSavedDevice


@fieldwise_init
struct SD35JointBlockDeviceForward(Copyable, Movable):
    var ctx_out: TArc   # [N_CTX, D] F32 device-resident
    var x_out: TArc     # [N_IMG, D]
    var saved: SD35JointBlockSavedDevice


@fieldwise_init
struct SD35JointBlockLoraDeviceGrads(Copyable, Movable):
    var d_ctx: TArc                  # [N_CTX,D] F32
    var d_x: TArc                    # [N_IMG,D] F32
    var d_a: List[Optional[TArc]]    # SD35_SLOTS entries
    var d_b: List[Optional[TArc]]


# ═══════════════════════════════════════════════════════════════════════════
# LoRA device primitives — mirror _sd35_lora_fwd/_sd35_lora_bwd EXACTLY:
# bf16 GEMMs on bf16-rounded F32 activations; F32 scale; F32 folds.
# ═══════════════════════════════════════════════════════════════════════════
def _lora_apply_f32(
    var base_y: Tensor, x_f32: Tensor, lo: Optional[SD35LoraAdapterDevice],
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    if not lo:
        return base_y^
    ref l = lo.value()
    var xb = _bf16(x_f32, ctx)                          # host: F32->BF16 upload (round)
    var nb1 = Optional[Tensor](None)
    var t = linear(xb, l.a[], nb1^, ctx)                # bf16 (host t roundtrips exactly)
    var nb2 = Optional[Tensor](None)
    var dy = linear(t, l.b[], nb2^, ctx)                # bf16
    var delta = mul_scalar(_f32(dy, ctx), l.scale, ctx) # F32 (host: scale in F32)
    return add(base_y, delta, ctx)                      # F32 add, F32 store


@fieldwise_init
struct _SD35LoraBwd(Copyable, Movable):
    var d_a: TArc      # [rank,in_f] bf16
    var d_b: TArc      # [out_f,rank] bf16
    var d_x_lo: TArc   # [M,in_f] bf16


def _lora_bwd_f32(
    d_contrib_f32: Tensor, x_f32: Tensor, lo: SD35LoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> _SD35LoraBwd:
    var xb = _bf16(x_f32, ctx)
    var nb = Optional[Tensor](None)
    var t = linear(xb, lo.a[], nb^, ctx)                                     # bf16
    # host: d_dy = scale * d_contrib in F32, then BF16 upload (round)
    var d_dy = _bf16(mul_scalar(d_contrib_f32, lo.scale, ctx), ctx)
    var d_t = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)    # bf16
    var d_b = linear_backward_dw(d_dy, t, M, lo.rank, lo.out_f, ctx)
    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)
    var d_a = linear_backward_dw(d_t, xb, M, lo.in_f, lo.rank, ctx)
    return _SD35LoraBwd(TArc(d_a^), TArc(d_b^), TArc(d_x_lo^))


def _proj_bwd_lora_f32(
    d_y_f32: Tensor, x_f32: Tensor, w: Tensor,
    lo: Optional[SD35LoraAdapterDevice], slot: Int,
    M: Int, in_f: Int, out_f: Int,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
) raises -> Tensor:
    var base_dx = linear_backward_dx(d_y_f32, w, M, in_f, out_f, ctx)   # F32
    if lo:
        var lg = _lora_bwd_f32(d_y_f32, x_f32, lo.value(), M, ctx)
        d_a_slots[slot] = Optional[TArc](lg.d_a)
        d_b_slots[slot] = Optional[TArc](lg.d_b)
        # host folds base F32 + lora F32 (exact upcast) with _add_lists, no re-round
        return add(base_dx, _f32(lg.d_x_lo[], ctx), ctx)
    return base_dx^




# ── head-chunked math SDPA (S=4250-scale memory fix) ─────────────────────────
# Math-mode scores are [1,H,S,S] F32 (~2.7GB at Large 1024²). Attention is
# PER-HEAD independent, so computing head-halves separately is BIT-IDENTICAL
# while halving peak score memory. H must be even (Large H=38 -> 2x19).


# ── FLASH vs MATH attention dispatch (audit item 2, 2026-07-06) ──────────────
# MEASURED: per-head math SDPA = 8.85s of the ~15s sd35-Large step (61%).
# FLASH=True (default): cuDNN padmask flash — q/k/v bf16-cast, padded to the
# next 128-multiple, real_len masks the pad; backward REGENERATES o/stats by
# re-running the (deterministic-input) flash fwd, so the saved-tape contract
# is unchanged. VALUE-CLASS numerics vs the F32 math oracle (bf16 flash —
# same class as the klein/zimage flash sign-off); FLASH=False = the per-head
# math arm, BIT-IDENTICAL to the host oracle (33/33 gate).
def _pad_seq_zeros[H: Int, Dh: Int](t_bf: Tensor, S: Int, S_PAD: Int, ctx: DeviceContext) raises -> Tensor:
    if S_PAD == S:
        return t_bf.clone(ctx)
    var zeros_h = List[Float32]()
    for _ in range((S_PAD - S) * H * Dh):
        zeros_h.append(0.0)
    var zpad = cast_tensor(
        Tensor.from_host(zeros_h^, [1, S_PAD - S, H, Dh], STDtype.F32, ctx),
        STDtype.BF16, ctx,
    )
    return concat(1, ctx, t_bf, zpad)


def _sd35_sdpa_fwd[
    H: Int, Dh: Int, S: Int, FLASH: Bool
](q: Tensor, k: Tensor, v: Tensor, scale: Float32, ctx: DeviceContext) raises -> Tensor:
    comptime if FLASH:
        comptime S_PAD = ((S + 127) // 128) * 128
        var qp = _pad_seq_zeros[H, Dh](_bf16(q, ctx), S, S_PAD, ctx)
        var kp = _pad_seq_zeros[H, Dh](_bf16(k, ctx), S, S_PAD, ctx)
        var vp = _pad_seq_zeros[H, Dh](_bf16(v, ctx), S, S_PAD, ctx)
        var f = sdpa_flash_train_fwd_padmask_bf16[1, S_PAD, H, Dh](qp, kp, vp, S, scale, ctx)
        var o_real = slice(f.att, 1, 0, S, ctx)
        return _f32(o_real, ctx)
    else:
        return _sdpa_fwd_hchunk[H, Dh, S](q, k, v, scale, ctx)


def _sd35_sdpa_bwd[
    H: Int, Dh: Int, S: Int, FLASH: Bool
](q: Tensor, k: Tensor, v: Tensor, d_att: Tensor, scale: Float32, ctx: DeviceContext) raises -> _SdpaChunkGrads:
    comptime if FLASH:
        comptime S_PAD = ((S + 127) // 128) * 128
        var qp = _pad_seq_zeros[H, Dh](_bf16(q, ctx), S, S_PAD, ctx)
        var kp = _pad_seq_zeros[H, Dh](_bf16(k, ctx), S, S_PAD, ctx)
        var vp = _pad_seq_zeros[H, Dh](_bf16(v, ctx), S, S_PAD, ctx)
        # regenerate o/stats (flash fwd on identical padded inputs)
        var f = sdpa_flash_train_fwd_padmask_bf16[1, S_PAD, H, Dh](qp, kp, vp, S, scale, ctx)
        var dop = _pad_seq_zeros[H, Dh](_bf16(d_att, ctx), S, S_PAD, ctx)
        var g = sdpa_flash_backward_padmask_bf16[1, S_PAD, H, Dh](
            f.q_bf, f.k_bf, f.v_bf, f.o_bf, f.stats, dop, S, scale, ctx,
        )
        var dq_r = slice(g.d_q, 1, 0, S, ctx)
        var dk_r = slice(g.d_k, 1, 0, S, ctx)
        var dv_r = slice(g.d_v, 1, 0, S, ctx)
        return _SdpaChunkGrads(_f32(dq_r, ctx), _f32(dk_r, ctx), _f32(dv_r, ctx))
    else:
        return _sdpa_bwd_hchunk[H, Dh, S](q, k, v, d_att, scale, ctx)


def _sdpa_fwd_hchunk[
    H: Int, Dh: Int, S: Int
](q: Tensor, k: Tensor, v: Tensor, scale: Float32, ctx: DeviceContext) raises -> Tensor:

    # PER-HEAD chunks: math scores are [1,h,S,S] F32 (~2.7GB at H=38, S=4250);
    # per-head ~72MB. Heads are independent => BIT-IDENTICAL. Rolling concat
    # (head outputs are [1,S,1,Dh] ~1MB — cheap).
    var qh0 = slice(q, 2, 0, 1, ctx)
    var kh0 = slice(k, 2, 0, 1, ctx)
    var vh0 = slice(v, 2, 0, 1, ctx)
    var out = sdpa_nomask[1, S, 1, Dh](qh0, kh0, vh0, scale, ctx)
    for h in range(1, H):
        var qh = slice(q, 2, h, 1, ctx)
        var kh = slice(k, 2, h, 1, ctx)
        var vh = slice(v, 2, h, 1, ctx)
        var ah = sdpa_nomask[1, S, 1, Dh](qh, kh, vh, scale, ctx)
        out = concat(2, ctx, out, ah)
    return out^


@fieldwise_init
struct _SdpaChunkGrads(Movable):
    var d_q: Tensor
    var d_k: Tensor
    var d_v: Tensor


def _sdpa_bwd_hchunk[
    H: Int, Dh: Int, S: Int
](q: Tensor, k: Tensor, v: Tensor, d_att: Tensor, scale: Float32, ctx: DeviceContext) raises -> _SdpaChunkGrads:

    var q0 = slice(q, 2, 0, 1, ctx)
    var k0 = slice(k, 2, 0, 1, ctx)
    var v0 = slice(v, 2, 0, 1, ctx)
    var d0 = slice(d_att, 2, 0, 1, ctx)
    var g0 = sdpa_backward[1, S, 1, Dh](q0, k0, v0, d0, scale, ctx)
    var dq = g0.d_q.clone(ctx)
    var dk = g0.d_k.clone(ctx)
    var dv = g0.d_v.clone(ctx)
    for h in range(1, H):
        var qh = slice(q, 2, h, 1, ctx)
        var kh = slice(k, 2, h, 1, ctx)
        var vh = slice(v, 2, h, 1, ctx)
        var dh = slice(d_att, 2, h, 1, ctx)
        var g = sdpa_backward[1, S, 1, Dh](qh, kh, vh, dh, scale, ctx)
        dq = concat(2, ctx, dq, g.d_q)
        dk = concat(2, ctx, dk, g.d_k)
        dv = concat(2, ctx, dv, g.d_v)
    return _SdpaChunkGrads(dq^, dk^, dv^)


# ═══════════════════════════════════════════════════════════════════════════
# Forward
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct _SD35StreamPreDevice(Copyable, Movable):
    var ln1: TArc
    var norm: TArc
    var q_pre: TArc    # [1,N,H,Dh]
    var k_pre: TArc
    var q_rms: TArc
    var k_rms: TArc
    var v: TArc


def _sd35_stream_pre_device[
    H: Int, Dh: Int
](
    s: TArc, w: SD35StreamWeightsDevice, mv: SD35ModVecsDevice,
    lo_qkv: Optional[SD35LoraAdapterDevice],
    N: Int, D: Int, eps: Float32, qk_eps: Float32,
    ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _SD35StreamPreDevice:
    var ln1 = layer_norm(s[], ones, zeros, eps, ctx)                # F32
    var norm = modulate(ln1, mv.scale_msa[], mv.shift_msa[], ctx)   # F32
    var b = Optional[Tensor](w.bqkv[].clone(ctx))
    var qkv_base = linear(norm, w.wqkv[], b, ctx)                   # [N,3D] F32
    var qkv = _lora_apply_f32(qkv_base^, norm, lo_qkv, N, ctx)
    var q_flat = slice(qkv, 1, 0, D, ctx)
    var k_flat = slice(qkv, 1, D, D, ctx)
    var v_flat = slice(qkv, 1, 2 * D, D, ctx)
    var q_pre = reshape_owned(q_flat^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, N, H, Dh])
    var v = reshape_owned(v_flat^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], qk_eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], qk_eps, ctx)
    return _SD35StreamPreDevice(
        TArc(ln1^), TArc(norm^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
    )


@fieldwise_init
struct _SD35StreamPostDevice(Copyable, Movable):
    var out: TArc
    var attn_res: TArc
    var ln2: TArc
    var mlp_in: TArc
    var h1: TArc
    var hg: TArc


def _sd35_stream_post_device(
    s: TArc, att: TArc, w: SD35StreamWeightsDevice, mv: SD35ModVecsDevice,
    lo: SD35StreamLoraDevice,
    N: Int, D: Int, MLP: Int, eps: Float32,
    ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _SD35StreamPostDevice:
    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var proj_base = linear(att[], w.wproj[], bp, ctx)               # [N,D] F32
    var proj = _lora_apply_f32(proj_base^, att[], lo.proj, N, ctx)
    var attn_res = residual_gate(s[], mv.gate_msa[], proj, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, mv.scale_mlp[], mv.shift_mlp[], ctx)
    var b1 = Optional[Tensor](w.bfc1[].clone(ctx))
    var h1_base = linear(mlp_in, w.wfc1[], b1, ctx)                 # [N,MLP]
    var h1 = _lora_apply_f32(h1_base^, mlp_in, lo.fc1, N, ctx)
    var hg = gelu(h1, ctx)
    var b2 = Optional[Tensor](w.bfc2[].clone(ctx))
    var mlp_base = linear(hg, w.wfc2[], b2, ctx)                    # [N,D]
    var mlp_out = _lora_apply_f32(mlp_base^, hg, lo.fc2, N, ctx)
    var out = residual_gate(attn_res, mv.gate_mlp[], mlp_out, ctx)
    return _SD35StreamPostDevice(
        TArc(out^), TArc(attn_res^), TArc(ln2^), TArc(mlp_in^), TArc(h1^), TArc(hg^),
    )


def _sd35_stream_post_device_scratch(
    s: TArc, att: TArc, w: SD35StreamWeightsDevice, mv: SD35ModVecsDevice,
    lo: SD35StreamLoraDevice,
    N: Int, D: Int, MLP: Int, eps: Float32,
    ones: Tensor, zeros: Tensor, ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> _SD35StreamPostDevice:
    """Scratch-ring sibling of `_sd35_stream_post_device` for the large-N
    (image) stream. Routes the two ≥128MB [N,MLP] F32 forward transients
    through `scratch`: `h1_base` (linear_bias_scratch, rewound before `hg`) and
    the SAVED `hg` (gelu_scratch, which stays live at the ring top for the whole
    block iteration). Everything else is identical to the plain helper, so the
    saved struct is bit-identical. Callers MUST hold ONE ring frame across the
    recompute-forward -> backward span so `hg` survives (conductor contract)."""
    comptime if not SD35_SCRATCH_MLP:
        return _sd35_stream_post_device(s, att, w, mv, lo, N, D, MLP, eps, ones, zeros, ctx)

    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var proj_base = linear(att[], w.wproj[], bp, ctx)               # [N,D] F32
    var proj = _lora_apply_f32(proj_base^, att[], lo.proj, N, ctx)
    var attn_res = residual_gate(s[], mv.gate_msa[], proj, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, mv.scale_mlp[], mv.shift_mlp[], ctx)

    # fc1: route the h1_base [N,MLP] transient through the ring, reclaim it
    # before `hg` (which reuses its slot and becomes the block-iteration
    # survivor). `h1` (the LoRA-add result, or the plain linear when no adapter)
    # stays a normal alloc — it is a SAVED tensor read by the backward.
    var mark_hb = scratch.mark()
    var h1: Tensor
    if lo.fc1:
        var h1_base = linear_bias_scratch(
            mlp_in, w.wfc1[], Optional[Tensor](w.bfc1[].clone(ctx)), ctx, scratch,
        )                                                          # ring [N,MLP]
        h1 = _lora_apply_f32(h1_base^, mlp_in, lo.fc1, N, ctx)     # plain
    else:
        h1 = linear(mlp_in, w.wfc1[], Optional[Tensor](w.bfc1[].clone(ctx)), ctx)
    scratch.rewind(mark_hb)
    var hg = gelu_scratch(h1, ctx, scratch)                        # ring [N,MLP] SAVED

    var b2 = Optional[Tensor](w.bfc2[].clone(ctx))
    var mlp_base = linear(hg, w.wfc2[], b2, ctx)                    # [N,D]
    var mlp_out = _lora_apply_f32(mlp_base^, hg, lo.fc2, N, ctx)
    var out = residual_gate(attn_res, mv.gate_mlp[], mlp_out, ctx)
    return _SD35StreamPostDevice(
        TArc(out^), TArc(attn_res^), TArc(ln2^), TArc(mlp_in^), TArc(h1^), TArc(hg^),
    )


def sd35_joint_block_forward_device[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    context: TArc, x: TArc,
    w: SD35JointBlockWeightsDevice,
    ctx_mod: SD35ModVecsDevice, x_mod: SD35ModVecsDevice,
    lora: SD35JointBlockLoraDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> SD35JointBlockDeviceForward:
    var ones_t = _t_f32(_ones(D), [D], ctx)
    var zeros_t = _t_f32(_zeros(D), [D], ctx)

    var cp = _sd35_stream_pre_device[H, Dh](
        context, w.ctxw, ctx_mod, lora.ctxl.qkv, N_CTX, D, eps, qk_eps, ones_t, zeros_t, ctx)
    var xp = _sd35_stream_pre_device[H, Dh](
        x, w.xw, x_mod, lora.xl.qkv, N_IMG, D, eps, qk_eps, ones_t, zeros_t, ctx)

    # JOINT attention: concat CTX FIRST along the sequence (dim 1)
    var q = concat(1, ctx, cp.q_rms[], xp.q_rms[])   # [1,S,H,Dh]
    var k = concat(1, ctx, cp.k_rms[], xp.k_rms[])
    var v = concat(1, ctx, cp.v[], xp.v[])
    var att = _sd35_sdpa_fwd[H, Dh, S, FLASH](q, k, v, scale, ctx)

    var ctx_att_4d = slice(att, 1, 0, N_CTX, ctx)
    var x_att_4d = slice(att, 1, N_CTX, N_IMG, ctx)
    var ctx_att = TArc(reshape_owned(ctx_att_4d^, [N_CTX, D]))
    var x_att = TArc(reshape_owned(x_att_4d^, [N_IMG, D]))

    var cpost = _sd35_stream_post_device(
        context, ctx_att, w.ctxw, ctx_mod, lora.ctxl, N_CTX, D, MLP, eps, ones_t, zeros_t, ctx)
    var xpost = _sd35_stream_post_device(
        x, x_att, w.xw, x_mod, lora.xl, N_IMG, D, MLP, eps, ones_t, zeros_t, ctx)

    var ctx_saved = SD35StreamSavedDevice(
        context.copy(), cp.ln1.copy(), cp.norm.copy(), cp.q_pre.copy(), cp.k_pre.copy(),
        cp.v.copy(), ctx_att.copy(), cpost.attn_res.copy(), cpost.ln2.copy(),
        cpost.mlp_in.copy(), cpost.h1.copy(), cpost.hg.copy(),
    )
    var x_saved = SD35StreamSavedDevice(
        x.copy(), xp.ln1.copy(), xp.norm.copy(), xp.q_pre.copy(), xp.k_pre.copy(),
        xp.v.copy(), x_att.copy(), xpost.attn_res.copy(), xpost.ln2.copy(),
        xpost.mlp_in.copy(), xpost.h1.copy(), xpost.hg.copy(),
    )
    return SD35JointBlockDeviceForward(
        cpost.out.copy(), xpost.out.copy(),
        SD35JointBlockSavedDevice(ctx_saved^, x_saved^),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Backward (dx-only frozen base; 16 LoRA slots)
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct _SD35StreamPostBackDevice(Copyable, Movable):
    var d_s: TArc      # [N,D] F32 (residual partial)
    var d_att: TArc    # [N,D] F32


def _sd35_stream_post_backward_device(
    d_out: TArc, sv: SD35StreamSavedDevice, w: SD35StreamWeightsDevice,
    mv: SD35ModVecsDevice, lo: SD35StreamLoraDevice,
    slot_base: Int,
    N: Int, D: Int, MLP: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
) raises -> _SD35StreamPostBackDevice:
    # out = attn_res + gate_mlp * mlp
    var grb2 = gate_residual_backward_dxdy(d_out[], mv.gate_mlp[], ctx)
    # mlp = linear(hg, Wfc2)[+LoRA]
    var d_hg = _proj_bwd_lora_f32(
        grb2.d_y, sv.hg[], w.wfc2[], lo.fc2, slot_base + 3, N, MLP, D,
        d_a_slots, d_b_slots, ctx,
    )
    var d_h1 = gelu_backward(d_hg, sv.h1[], ctx)
    var d_mlp_in = _proj_bwd_lora_f32(
        d_h1, sv.mlp_in[], w.wfc1[], lo.fc1, slot_base + 2, N, D, MLP,
        d_a_slots, d_b_slots, ctx,
    )
    var mb2 = modulate_backward(d_mlp_in, sv.ln2[], mv.scale_mlp[], ctx, False)
    var d_attn_res_ln = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    var d_attn_res = add(grb2.d_x, d_attn_res_ln, ctx)      # F32

    # attn_res = s + gate_msa * proj
    var grb1 = gate_residual_backward_dxdy(d_attn_res, mv.gate_msa[], ctx)
    var d_att = _proj_bwd_lora_f32(
        grb1.d_y, sv.att[], w.wproj[], lo.proj, slot_base + 1, N, D, D,
        d_a_slots, d_b_slots, ctx,
    )
    return _SD35StreamPostBackDevice(TArc(grb1.d_x.clone(ctx)), TArc(d_att^))


def _sd35_stream_post_backward_device_scratch(
    d_out: TArc, sv: SD35StreamSavedDevice, w: SD35StreamWeightsDevice,
    mv: SD35ModVecsDevice, lo: SD35StreamLoraDevice,
    slot_base: Int,
    N: Int, D: Int, MLP: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext, mut scratch: ScratchRingAllocator,
) raises -> _SD35StreamPostBackDevice:
    """Scratch-ring sibling of `_sd35_stream_post_backward_device` (large-N
    stream). Routes the two ≥128MB [N,MLP] F32 backward transients through
    `scratch`: the fc2 `base_dx` (linear_backward_dx_scratch, rewound after the
    LoRA fold) and `d_h1` (gelu_backward_scratch, rewound after fc1's dx). Both
    are allocated ABOVE the recompute-forward's surviving `hg` slot and reclaimed
    within this call, so ring peak stays at ~2 classes. Math is bit-identical to
    the plain helper; `sv.hg`/`sv.h1` are read exactly as before."""
    comptime if not SD35_SCRATCH_MLP:
        return _sd35_stream_post_backward_device(
            d_out, sv, w, mv, lo, slot_base, N, D, MLP, eps, ones,
            d_a_slots, d_b_slots, ctx,
        )

    var grb2 = gate_residual_backward_dxdy(d_out[], mv.gate_mlp[], ctx)
    # d_hg = fc2 backward (+LoRA fold). base_dx [N,MLP] routed through the ring;
    # d_hg (the LoRA add survivor, consumed next by gelu_backward) stays plain.
    var d_hg: Tensor
    if lo.fc2:
        var mark_t = scratch.mark()
        var base_dx = linear_backward_dx_scratch(
            grb2.d_y, w.wfc2[], N, MLP, D, ctx, scratch,
        )                                                          # ring [N,MLP]
        var lg = _lora_bwd_f32(grb2.d_y, sv.hg[], lo.fc2.value(), N, ctx)
        d_a_slots[slot_base + 3] = Optional[TArc](lg.d_a)
        d_b_slots[slot_base + 3] = Optional[TArc](lg.d_b)
        d_hg = add(base_dx, _f32(lg.d_x_lo[], ctx), ctx)           # plain
        scratch.rewind(mark_t)
    else:
        d_hg = linear_backward_dx(grb2.d_y, w.wfc2[], N, MLP, D, ctx)  # plain

    var mark_d = scratch.mark()
    var d_h1 = gelu_backward_scratch(d_hg, sv.h1[], ctx, scratch)   # ring [N,MLP]
    var d_mlp_in = _proj_bwd_lora_f32(
        d_h1, sv.mlp_in[], w.wfc1[], lo.fc1, slot_base + 2, N, D, MLP,
        d_a_slots, d_b_slots, ctx,
    )                                                              # plain [N,D]
    scratch.rewind(mark_d)

    var mb2 = modulate_backward(d_mlp_in, sv.ln2[], mv.scale_mlp[], ctx, False)
    var d_attn_res_ln = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    var d_attn_res = add(grb2.d_x, d_attn_res_ln, ctx)             # F32

    var grb1 = gate_residual_backward_dxdy(d_attn_res, mv.gate_msa[], ctx)
    var d_att = _proj_bwd_lora_f32(
        grb1.d_y, sv.att[], w.wproj[], lo.proj, slot_base + 1, N, D, D,
        d_a_slots, d_b_slots, ctx,
    )
    return _SD35StreamPostBackDevice(TArc(grb1.d_x.clone(ctx)), TArc(d_att^))


def _sd35_stream_pre_backward_device[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    d_s_partial: TArc,
    sv: SD35StreamSavedDevice, w: SD35StreamWeightsDevice,
    mv: SD35ModVecsDevice, lo_qkv: Optional[SD35LoraAdapterDevice],
    slot_base: Int,
    N: Int, D: Int, eps: Float32, qk_eps: Float32, ones: Tensor,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
) raises -> Tensor:
    var d_q_pre = rms_norm_backward_dx(d_q_rms, sv.q_pre[], w.q_norm[], qk_eps, ctx)
    var d_k_pre = rms_norm_backward_dx(d_k_rms, sv.k_pre[], w.k_norm[], qk_eps, ctx)
    var d_q_flat = reshape(d_q_pre, [N, D], ctx)
    var d_k_flat = reshape(d_k_pre, [N, D], ctx)
    var d_v_flat = reshape(d_v, [N, D], ctx)
    var d_qkv = concat(1, ctx, d_q_flat, d_k_flat, d_v_flat)   # [N,3D]

    var d_norm = _proj_bwd_lora_f32(
        d_qkv, sv.norm[], w.wqkv[], lo_qkv, slot_base, N, D, 3 * D,
        d_a_slots, d_b_slots, ctx,
    )
    var mb1 = modulate_backward(d_norm, sv.ln1[], mv.scale_msa[], ctx, False)
    var d_s_norm = layer_norm_backward_dx(mb1.d_x, sv.s[], ones, eps, ctx)
    return add(d_s_partial[], d_s_norm, ctx)   # F32 (host _add_lists)


def sd35_joint_block_backward_device[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    d_ctx_out: TArc, d_x_out: TArc,
    w: SD35JointBlockWeightsDevice,
    ctx_mod: SD35ModVecsDevice, x_mod: SD35ModVecsDevice,
    lora: SD35JointBlockLoraDevice,
    saved: SD35JointBlockSavedDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> SD35JointBlockLoraDeviceGrads:
    var ones_t = _t_f32(_ones(D), [D], ctx)

    var d_a = List[Optional[TArc]]()
    var d_b = List[Optional[TArc]]()
    for _ in range(SD35_SLOTS):
        d_a.append(Optional[TArc](None))
        d_b.append(Optional[TArc](None))

    var cpb = _sd35_stream_post_backward_device(
        d_ctx_out, saved.ctx_saved, w.ctxw, ctx_mod, lora.ctxl, SD_CQKV,
        N_CTX, D, MLP, eps, ones_t, d_a, d_b, ctx,
    )
    var xpb = _sd35_stream_post_backward_device(
        d_x_out, saved.x_saved, w.xw, x_mod, lora.xl, SD_XQKV,
        N_IMG, D, MLP, eps, ones_t, d_a, d_b, ctx,
    )

    # joint d_att (ctx FIRST) + recompute joint q_rms/k_rms from saved q/k_pre
    var d_catt_4d = reshape(cpb.d_att[], [1, N_CTX, H, Dh], ctx)
    var d_xatt_4d = reshape(xpb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_catt_4d, d_xatt_4d)     # [1,S,H,Dh]

    var cq = rms_norm(saved.ctx_saved.q_pre[], w.ctxw.q_norm[], qk_eps, ctx)
    var ck = rms_norm(saved.ctx_saved.k_pre[], w.ctxw.k_norm[], qk_eps, ctx)
    var xq = rms_norm(saved.x_saved.q_pre[], w.xw.q_norm[], qk_eps, ctx)
    var xk = rms_norm(saved.x_saved.k_pre[], w.xw.k_norm[], qk_eps, ctx)
    var jq = concat(1, ctx, cq, xq)
    var jk = concat(1, ctx, ck, xk)
    var jv = concat(1, ctx, saved.ctx_saved.v[], saved.x_saved.v[])

    var sb = _sd35_sdpa_bwd[H, Dh, S, FLASH](jq, jk, jv, d_att_joint, scale, ctx)

    var c_dq = slice(sb.d_q, 1, 0, N_CTX, ctx)
    var x_dq = slice(sb.d_q, 1, N_CTX, N_IMG, ctx)
    var c_dk = slice(sb.d_k, 1, 0, N_CTX, ctx)
    var x_dk = slice(sb.d_k, 1, N_CTX, N_IMG, ctx)
    var c_dv = slice(sb.d_v, 1, 0, N_CTX, ctx)
    var x_dv = slice(sb.d_v, 1, N_CTX, N_IMG, ctx)

    var d_ctx = _sd35_stream_pre_backward_device[H, Dh](
        c_dq, c_dk, c_dv, cpb.d_s, saved.ctx_saved, w.ctxw, ctx_mod,
        lora.ctxl.qkv, SD_CQKV, N_CTX, D, eps, qk_eps, ones_t, d_a, d_b, ctx,
    )
    var d_x = _sd35_stream_pre_backward_device[H, Dh](
        x_dq, x_dk, x_dv, xpb.d_s, saved.x_saved, w.xw, x_mod,
        lora.xl.qkv, SD_XQKV, N_IMG, D, eps, qk_eps, ones_t, d_a, d_b, ctx,
    )

    return SD35JointBlockLoraDeviceGrads(TArc(d_ctx^), TArc(d_x^), d_a^, d_b^)


# ═══════════════════════════════════════════════════════════════════════════
# Scratch-ring joint block (fwd + bwd). The ctx (small-N) stream keeps the plain
# post helpers; the x (large-N) stream routes its MLP transients through the
# caller-owned ring. The conductor holds ONE ring frame per block iteration so
# the forward's ring-backed `hg` survives into the recompute-driven backward.
# ═══════════════════════════════════════════════════════════════════════════
def sd35_joint_block_forward_device_scratch[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    context: TArc, x: TArc,
    w: SD35JointBlockWeightsDevice,
    ctx_mod: SD35ModVecsDevice, x_mod: SD35ModVecsDevice,
    lora: SD35JointBlockLoraDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext, mut scratch: ScratchRingAllocator,
) raises -> SD35JointBlockDeviceForward:
    var ones_t = _t_f32(_ones(D), [D], ctx)
    var zeros_t = _t_f32(_zeros(D), [D], ctx)

    var cp = _sd35_stream_pre_device[H, Dh](
        context, w.ctxw, ctx_mod, lora.ctxl.qkv, N_CTX, D, eps, qk_eps, ones_t, zeros_t, ctx)
    var xp = _sd35_stream_pre_device[H, Dh](
        x, w.xw, x_mod, lora.xl.qkv, N_IMG, D, eps, qk_eps, ones_t, zeros_t, ctx)

    var q = concat(1, ctx, cp.q_rms[], xp.q_rms[])   # [1,S,H,Dh]
    var k = concat(1, ctx, cp.k_rms[], xp.k_rms[])
    var v = concat(1, ctx, cp.v[], xp.v[])
    var att = _sd35_sdpa_fwd[H, Dh, S, FLASH](q, k, v, scale, ctx)

    var ctx_att_4d = slice(att, 1, 0, N_CTX, ctx)
    var x_att_4d = slice(att, 1, N_CTX, N_IMG, ctx)
    var ctx_att = TArc(reshape_owned(ctx_att_4d^, [N_CTX, D]))
    var x_att = TArc(reshape_owned(x_att_4d^, [N_IMG, D]))

    var cpost = _sd35_stream_post_device(
        context, ctx_att, w.ctxw, ctx_mod, lora.ctxl, N_CTX, D, MLP, eps, ones_t, zeros_t, ctx)
    var xpost = _sd35_stream_post_device_scratch(
        x, x_att, w.xw, x_mod, lora.xl, N_IMG, D, MLP, eps, ones_t, zeros_t, ctx, scratch)

    var ctx_saved = SD35StreamSavedDevice(
        context.copy(), cp.ln1.copy(), cp.norm.copy(), cp.q_pre.copy(), cp.k_pre.copy(),
        cp.v.copy(), ctx_att.copy(), cpost.attn_res.copy(), cpost.ln2.copy(),
        cpost.mlp_in.copy(), cpost.h1.copy(), cpost.hg.copy(),
    )
    var x_saved = SD35StreamSavedDevice(
        x.copy(), xp.ln1.copy(), xp.norm.copy(), xp.q_pre.copy(), xp.k_pre.copy(),
        xp.v.copy(), x_att.copy(), xpost.attn_res.copy(), xpost.ln2.copy(),
        xpost.mlp_in.copy(), xpost.h1.copy(), xpost.hg.copy(),
    )
    return SD35JointBlockDeviceForward(
        cpost.out.copy(), xpost.out.copy(),
        SD35JointBlockSavedDevice(ctx_saved^, x_saved^),
    )


def sd35_joint_block_backward_device_scratch[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    d_ctx_out: TArc, d_x_out: TArc,
    w: SD35JointBlockWeightsDevice,
    ctx_mod: SD35ModVecsDevice, x_mod: SD35ModVecsDevice,
    lora: SD35JointBlockLoraDevice,
    saved: SD35JointBlockSavedDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext, mut scratch: ScratchRingAllocator,
) raises -> SD35JointBlockLoraDeviceGrads:
    var ones_t = _t_f32(_ones(D), [D], ctx)

    var d_a = List[Optional[TArc]]()
    var d_b = List[Optional[TArc]]()
    for _ in range(SD35_SLOTS):
        d_a.append(Optional[TArc](None))
        d_b.append(Optional[TArc](None))

    var cpb = _sd35_stream_post_backward_device(
        d_ctx_out, saved.ctx_saved, w.ctxw, ctx_mod, lora.ctxl, SD_CQKV,
        N_CTX, D, MLP, eps, ones_t, d_a, d_b, ctx,
    )
    var xpb = _sd35_stream_post_backward_device_scratch(
        d_x_out, saved.x_saved, w.xw, x_mod, lora.xl, SD_XQKV,
        N_IMG, D, MLP, eps, ones_t, d_a, d_b, ctx, scratch,
    )

    var d_catt_4d = reshape(cpb.d_att[], [1, N_CTX, H, Dh], ctx)
    var d_xatt_4d = reshape(xpb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_catt_4d, d_xatt_4d)     # [1,S,H,Dh]

    var cq = rms_norm(saved.ctx_saved.q_pre[], w.ctxw.q_norm[], qk_eps, ctx)
    var ck = rms_norm(saved.ctx_saved.k_pre[], w.ctxw.k_norm[], qk_eps, ctx)
    var xq = rms_norm(saved.x_saved.q_pre[], w.xw.q_norm[], qk_eps, ctx)
    var xk = rms_norm(saved.x_saved.k_pre[], w.xw.k_norm[], qk_eps, ctx)
    var jq = concat(1, ctx, cq, xq)
    var jk = concat(1, ctx, ck, xk)
    var jv = concat(1, ctx, saved.ctx_saved.v[], saved.x_saved.v[])

    var sb = _sd35_sdpa_bwd[H, Dh, S, FLASH](jq, jk, jv, d_att_joint, scale, ctx)

    var c_dq = slice(sb.d_q, 1, 0, N_CTX, ctx)
    var x_dq = slice(sb.d_q, 1, N_CTX, N_IMG, ctx)
    var c_dk = slice(sb.d_k, 1, 0, N_CTX, ctx)
    var x_dk = slice(sb.d_k, 1, N_CTX, N_IMG, ctx)
    var c_dv = slice(sb.d_v, 1, 0, N_CTX, ctx)
    var x_dv = slice(sb.d_v, 1, N_CTX, N_IMG, ctx)

    var d_ctx = _sd35_stream_pre_backward_device[H, Dh](
        c_dq, c_dk, c_dv, cpb.d_s, saved.ctx_saved, w.ctxw, ctx_mod,
        lora.ctxl.qkv, SD_CQKV, N_CTX, D, eps, qk_eps, ones_t, d_a, d_b, ctx,
    )
    var d_x = _sd35_stream_pre_backward_device[H, Dh](
        x_dq, x_dk, x_dv, xpb.d_s, saved.x_saved, w.xw, x_mod,
        lora.xl.qkv, SD_XQKV, N_IMG, D, eps, qk_eps, ones_t, d_a, d_b, ctx,
    )

    return SD35JointBlockLoraDeviceGrads(TArc(d_ctx^), TArc(d_x^), d_a^, d_b^)


# ═══════════════════════════════════════════════════════════════════════════
# context_pre_only final block (Large joint_blocks.37): ctx stream = qkv ONLY
# (AdaLayerNormContinuous scale/shift modulate, no gates, no post); joint
# attention ctx-first; the ctx attention slice is DISCARDED (backward feeds
# ZERO d_att rows for ctx); only x gets post + output. 5 LoRA slots:
# ctx_qkv + x{qkv,proj,fc1,fc2}. Mirror of sd35_block.mojo 2306/2433.
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct SD35CtxPreSavedDevice(Copyable, Movable):
    var ctx_input: TArc   # [N_CTX,D]
    var ctx_ln: TArc
    var ctx_norm: TArc
    var ctx_q_pre: TArc   # [1,N_CTX,H,Dh]
    var ctx_k_pre: TArc
    var ctx_v: TArc
    var x_saved: SD35StreamSavedDevice


@fieldwise_init
struct SD35CtxPreDeviceForward(Copyable, Movable):
    var x_out: TArc       # [N_IMG,D] F32
    var saved: SD35CtxPreSavedDevice


@fieldwise_init
struct SD35CtxPreDeviceGrads(Copyable, Movable):
    var d_x: TArc                    # [N_IMG,D] F32
    var d_ctx: TArc                  # [N_CTX,D] F32
    var d_a: List[Optional[TArc]]    # SD35_SLOTS layout (SD_CQKV + x slots used)
    var d_b: List[Optional[TArc]]


def sd35_context_preonly_forward_device[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    context: TArc, x: TArc,
    ctx_qkv_w: TArc, ctx_qkv_b: TArc,       # [3D,D] / [3D] F32
    ctx_qnorm: TArc, ctx_knorm: TArc,       # [Dh] F32
    ctx_scale: TArc, ctx_shift: TArc,       # [D] F32
    xw: SD35StreamWeightsDevice, x_mod: SD35ModVecsDevice,
    ctx_qkv_lora: Optional[SD35LoraAdapterDevice],
    x_lora: SD35StreamLoraDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> SD35CtxPreDeviceForward:
    var ones_t = _t_f32(_ones(D), [D], ctx)
    var zeros_t = _t_f32(_zeros(D), [D], ctx)

    # ctx pre (qkv only)
    var ctx_ln = layer_norm(context[], ones_t, zeros_t, eps, ctx)
    var ctx_norm = modulate(ctx_ln, ctx_scale[], ctx_shift[], ctx)
    var cb = Optional[Tensor](ctx_qkv_b[].clone(ctx))
    var cqkv_base = linear(ctx_norm, ctx_qkv_w[], cb, ctx)
    var cqkv = _lora_apply_f32(cqkv_base^, ctx_norm, ctx_qkv_lora, N_CTX, ctx)
    var cq_flat = slice(cqkv, 1, 0, D, ctx)
    var ck_flat = slice(cqkv, 1, D, D, ctx)
    var cv_flat = slice(cqkv, 1, 2 * D, D, ctx)
    var cq_pre = reshape_owned(cq_flat^, [1, N_CTX, H, Dh])
    var ck_pre = reshape_owned(ck_flat^, [1, N_CTX, H, Dh])
    var cv = reshape_owned(cv_flat^, [1, N_CTX, H, Dh])
    var cq_rms = rms_norm(cq_pre, ctx_qnorm[], qk_eps, ctx)
    var ck_rms = rms_norm(ck_pre, ctx_knorm[], qk_eps, ctx)

    # x pre (standard)
    var xp = _sd35_stream_pre_device[H, Dh](
        x, xw, x_mod, x_lora.qkv, N_IMG, D, eps, qk_eps, ones_t, zeros_t, ctx)

    var q = concat(1, ctx, cq_rms, xp.q_rms[])
    var k = concat(1, ctx, ck_rms, xp.k_rms[])
    var v = concat(1, ctx, cv, xp.v[])
    var att = _sd35_sdpa_fwd[H, Dh, S, FLASH](q, k, v, scale, ctx)

    var x_att_4d = slice(att, 1, N_CTX, N_IMG, ctx)
    var x_att = TArc(reshape_owned(x_att_4d^, [N_IMG, D]))

    var xpost = _sd35_stream_post_device(
        x, x_att, xw, x_mod, x_lora, N_IMG, D, MLP, eps, ones_t, zeros_t, ctx)

    var x_saved = SD35StreamSavedDevice(
        x.copy(), xp.ln1.copy(), xp.norm.copy(), xp.q_pre.copy(), xp.k_pre.copy(),
        xp.v.copy(), x_att.copy(), xpost.attn_res.copy(), xpost.ln2.copy(),
        xpost.mlp_in.copy(), xpost.h1.copy(), xpost.hg.copy(),
    )
    var saved = SD35CtxPreSavedDevice(
        context.copy(), TArc(ctx_ln^), TArc(ctx_norm^),
        TArc(cq_pre^), TArc(ck_pre^), TArc(cv^), x_saved^,
    )
    return SD35CtxPreDeviceForward(xpost.out.copy(), saved^)


def sd35_context_preonly_backward_device[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    d_x_out: TArc,
    ctx_qkv_w: TArc, ctx_qnorm: TArc, ctx_knorm: TArc, ctx_scale: TArc,
    xw: SD35StreamWeightsDevice, x_mod: SD35ModVecsDevice,
    ctx_qkv_lora: Optional[SD35LoraAdapterDevice],
    x_lora: SD35StreamLoraDevice,
    saved: SD35CtxPreSavedDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> SD35CtxPreDeviceGrads:
    var ones_t = _t_f32(_ones(D), [D], ctx)

    var d_a = List[Optional[TArc]]()
    var d_b = List[Optional[TArc]]()
    for _ in range(SD35_SLOTS):
        d_a.append(Optional[TArc](None))
        d_b.append(Optional[TArc](None))

    # x post backward (standard)
    var xpb = _sd35_stream_post_backward_device(
        d_x_out, saved.x_saved, xw, x_mod, x_lora, SD_XQKV,
        N_IMG, D, MLP, eps, ones_t, d_a, d_b, ctx,
    )

    # joint d_att: ZERO rows for ctx (its attention output was discarded)
    var d_catt_zero = Tensor.from_host(
        _zeros(N_CTX * H * Dh), [1, N_CTX, H, Dh], STDtype.F32, ctx)
    var d_xatt_4d = reshape(xpb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_catt_zero, d_xatt_4d)

    var cq = rms_norm(saved.ctx_q_pre[], ctx_qnorm[], qk_eps, ctx)
    var ck = rms_norm(saved.ctx_k_pre[], ctx_knorm[], qk_eps, ctx)
    var xq = rms_norm(saved.x_saved.q_pre[], xw.q_norm[], qk_eps, ctx)
    var xk = rms_norm(saved.x_saved.k_pre[], xw.k_norm[], qk_eps, ctx)
    var jq = concat(1, ctx, cq, xq)
    var jk = concat(1, ctx, ck, xk)
    var jv = concat(1, ctx, saved.ctx_v[], saved.x_saved.v[])

    var sb = _sd35_sdpa_bwd[H, Dh, S, FLASH](jq, jk, jv, d_att_joint, scale, ctx)

    var c_dq = slice(sb.d_q, 1, 0, N_CTX, ctx)
    var x_dq = slice(sb.d_q, 1, N_CTX, N_IMG, ctx)
    var c_dk = slice(sb.d_k, 1, 0, N_CTX, ctx)
    var x_dk = slice(sb.d_k, 1, N_CTX, N_IMG, ctx)
    var c_dv = slice(sb.d_v, 1, 0, N_CTX, ctx)
    var x_dv = slice(sb.d_v, 1, N_CTX, N_IMG, ctx)

    # x pre backward (standard)
    var d_x = _sd35_stream_pre_backward_device[H, Dh](
        x_dq, x_dk, x_dv, xpb.d_s, saved.x_saved, xw, x_mod,
        x_lora.qkv, SD_XQKV, N_IMG, D, eps, qk_eps, ones_t, d_a, d_b, ctx,
    )

    # ctx pre backward (qkv only; frozen base dx-only)
    var d_cq_pre = rms_norm_backward_dx(c_dq, saved.ctx_q_pre[], ctx_qnorm[], qk_eps, ctx)
    var d_ck_pre = rms_norm_backward_dx(c_dk, saved.ctx_k_pre[], ctx_knorm[], qk_eps, ctx)
    var d_cq_flat = reshape(d_cq_pre, [N_CTX, D], ctx)
    var d_ck_flat = reshape(d_ck_pre, [N_CTX, D], ctx)
    var d_cv_flat = reshape(c_dv, [N_CTX, D], ctx)
    var d_cqkv = concat(1, ctx, d_cq_flat, d_ck_flat, d_cv_flat)

    var d_ctx_norm = _proj_bwd_lora_f32(
        d_cqkv, saved.ctx_norm[], ctx_qkv_w[], ctx_qkv_lora, SD_CQKV,
        N_CTX, D, 3 * D, d_a, d_b, ctx,
    )
    var mb_c = modulate_backward(d_ctx_norm, saved.ctx_ln[], ctx_scale[], ctx, False)
    var d_context = layer_norm_backward_dx(mb_c.d_x, saved.ctx_input[], ones_t, eps, ctx)

    return SD35CtxPreDeviceGrads(TArc(d_x^), TArc(d_context^), d_a^, d_b^)


# ═══════════════════════════════════════════════════════════════════════════
# Scratch-ring context_pre_only block (fwd + bwd). Only the x stream has a post
# section here, so its MLP transients route through the ring exactly as the
# joint block; the ctx qkv-only path is unchanged.
# ═══════════════════════════════════════════════════════════════════════════
def sd35_context_preonly_forward_device_scratch[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    context: TArc, x: TArc,
    ctx_qkv_w: TArc, ctx_qkv_b: TArc,
    ctx_qnorm: TArc, ctx_knorm: TArc,
    ctx_scale: TArc, ctx_shift: TArc,
    xw: SD35StreamWeightsDevice, x_mod: SD35ModVecsDevice,
    ctx_qkv_lora: Optional[SD35LoraAdapterDevice],
    x_lora: SD35StreamLoraDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext, mut scratch: ScratchRingAllocator,
) raises -> SD35CtxPreDeviceForward:
    var ones_t = _t_f32(_ones(D), [D], ctx)
    var zeros_t = _t_f32(_zeros(D), [D], ctx)

    var ctx_ln = layer_norm(context[], ones_t, zeros_t, eps, ctx)
    var ctx_norm = modulate(ctx_ln, ctx_scale[], ctx_shift[], ctx)
    var cb = Optional[Tensor](ctx_qkv_b[].clone(ctx))
    var cqkv_base = linear(ctx_norm, ctx_qkv_w[], cb, ctx)
    var cqkv = _lora_apply_f32(cqkv_base^, ctx_norm, ctx_qkv_lora, N_CTX, ctx)
    var cq_flat = slice(cqkv, 1, 0, D, ctx)
    var ck_flat = slice(cqkv, 1, D, D, ctx)
    var cv_flat = slice(cqkv, 1, 2 * D, D, ctx)
    var cq_pre = reshape_owned(cq_flat^, [1, N_CTX, H, Dh])
    var ck_pre = reshape_owned(ck_flat^, [1, N_CTX, H, Dh])
    var cv = reshape_owned(cv_flat^, [1, N_CTX, H, Dh])
    var cq_rms = rms_norm(cq_pre, ctx_qnorm[], qk_eps, ctx)
    var ck_rms = rms_norm(ck_pre, ctx_knorm[], qk_eps, ctx)

    var xp = _sd35_stream_pre_device[H, Dh](
        x, xw, x_mod, x_lora.qkv, N_IMG, D, eps, qk_eps, ones_t, zeros_t, ctx)

    var q = concat(1, ctx, cq_rms, xp.q_rms[])
    var k = concat(1, ctx, ck_rms, xp.k_rms[])
    var v = concat(1, ctx, cv, xp.v[])
    var att = _sd35_sdpa_fwd[H, Dh, S, FLASH](q, k, v, scale, ctx)

    var x_att_4d = slice(att, 1, N_CTX, N_IMG, ctx)
    var x_att = TArc(reshape_owned(x_att_4d^, [N_IMG, D]))

    var xpost = _sd35_stream_post_device_scratch(
        x, x_att, xw, x_mod, x_lora, N_IMG, D, MLP, eps, ones_t, zeros_t, ctx, scratch)

    var x_saved = SD35StreamSavedDevice(
        x.copy(), xp.ln1.copy(), xp.norm.copy(), xp.q_pre.copy(), xp.k_pre.copy(),
        xp.v.copy(), x_att.copy(), xpost.attn_res.copy(), xpost.ln2.copy(),
        xpost.mlp_in.copy(), xpost.h1.copy(), xpost.hg.copy(),
    )
    var saved = SD35CtxPreSavedDevice(
        context.copy(), TArc(ctx_ln^), TArc(ctx_norm^),
        TArc(cq_pre^), TArc(ck_pre^), TArc(cv^), x_saved^,
    )
    return SD35CtxPreDeviceForward(xpost.out.copy(), saved^)


def sd35_context_preonly_backward_device_scratch[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    d_x_out: TArc,
    ctx_qkv_w: TArc, ctx_qnorm: TArc, ctx_knorm: TArc, ctx_scale: TArc,
    xw: SD35StreamWeightsDevice, x_mod: SD35ModVecsDevice,
    ctx_qkv_lora: Optional[SD35LoraAdapterDevice],
    x_lora: SD35StreamLoraDevice,
    saved: SD35CtxPreSavedDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext, mut scratch: ScratchRingAllocator,
) raises -> SD35CtxPreDeviceGrads:
    var ones_t = _t_f32(_ones(D), [D], ctx)

    var d_a = List[Optional[TArc]]()
    var d_b = List[Optional[TArc]]()
    for _ in range(SD35_SLOTS):
        d_a.append(Optional[TArc](None))
        d_b.append(Optional[TArc](None))

    var xpb = _sd35_stream_post_backward_device_scratch(
        d_x_out, saved.x_saved, xw, x_mod, x_lora, SD_XQKV,
        N_IMG, D, MLP, eps, ones_t, d_a, d_b, ctx, scratch,
    )

    var d_catt_zero = Tensor.from_host(
        _zeros(N_CTX * H * Dh), [1, N_CTX, H, Dh], STDtype.F32, ctx)
    var d_xatt_4d = reshape(xpb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_catt_zero, d_xatt_4d)

    var cq = rms_norm(saved.ctx_q_pre[], ctx_qnorm[], qk_eps, ctx)
    var ck = rms_norm(saved.ctx_k_pre[], ctx_knorm[], qk_eps, ctx)
    var xq = rms_norm(saved.x_saved.q_pre[], xw.q_norm[], qk_eps, ctx)
    var xk = rms_norm(saved.x_saved.k_pre[], xw.k_norm[], qk_eps, ctx)
    var jq = concat(1, ctx, cq, xq)
    var jk = concat(1, ctx, ck, xk)
    var jv = concat(1, ctx, saved.ctx_v[], saved.x_saved.v[])

    var sb = _sd35_sdpa_bwd[H, Dh, S, FLASH](jq, jk, jv, d_att_joint, scale, ctx)

    var c_dq = slice(sb.d_q, 1, 0, N_CTX, ctx)
    var x_dq = slice(sb.d_q, 1, N_CTX, N_IMG, ctx)
    var c_dk = slice(sb.d_k, 1, 0, N_CTX, ctx)
    var x_dk = slice(sb.d_k, 1, N_CTX, N_IMG, ctx)
    var c_dv = slice(sb.d_v, 1, 0, N_CTX, ctx)
    var x_dv = slice(sb.d_v, 1, N_CTX, N_IMG, ctx)

    var d_x = _sd35_stream_pre_backward_device[H, Dh](
        x_dq, x_dk, x_dv, xpb.d_s, saved.x_saved, xw, x_mod,
        x_lora.qkv, SD_XQKV, N_IMG, D, eps, qk_eps, ones_t, d_a, d_b, ctx,
    )

    var d_cq_pre = rms_norm_backward_dx(c_dq, saved.ctx_q_pre[], ctx_qnorm[], qk_eps, ctx)
    var d_ck_pre = rms_norm_backward_dx(c_dk, saved.ctx_k_pre[], ctx_knorm[], qk_eps, ctx)
    var d_cq_flat = reshape(d_cq_pre, [N_CTX, D], ctx)
    var d_ck_flat = reshape(d_ck_pre, [N_CTX, D], ctx)
    var d_cv_flat = reshape(c_dv, [N_CTX, D], ctx)
    var d_cqkv = concat(1, ctx, d_cq_flat, d_ck_flat, d_cv_flat)

    var d_ctx_norm = _proj_bwd_lora_f32(
        d_cqkv, saved.ctx_norm[], ctx_qkv_w[], ctx_qkv_lora, SD_CQKV,
        N_CTX, D, 3 * D, d_a, d_b, ctx,
    )
    var mb_c = modulate_backward(d_ctx_norm, saved.ctx_ln[], ctx_scale[], ctx, False)
    var d_context = layer_norm_backward_dx(mb_c.d_x, saved.ctx_input[], ones_t, eps, ctx)

    return SD35CtxPreDeviceGrads(TArc(d_x^), TArc(d_context^), d_a^, d_b^)


# ═══════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 (row-stacked) joint block, fwd + bwd (MJ-1073, sd35 rung).
#
# Template: chroma_block_device.mojo *_b2 (the flux-family b2 shape) mapped onto
# sd35's F32 chain + ctx-first joint attention. context [2*N_CTX,D], x
# [2*N_IMG,D]; each stream's two samples are ROW-STACKED (sample0 rows first).
# Per-block mods are PER-SAMPLE (sigma-dependent adaLN) → packed [2,D] and the
# shared modulate/residual_gate/modulate_backward/gate_residual_backward_dxdy ops
# broadcast them across each sample's row range (rows split evenly; elementwise.
# mojo:122 / rope_struct_backward.mojo:1112). The shared stream pre/post helpers
# run at M=2N: LoRA fwd applies per-row, and the backward's d_a/d_b GEMM SUMS
# both samples in-GEMM (M=2N) — the grad-accum=2 contract (caller pre-scales each
# sample's d_out by 0.5). Attention is PER-SAMPLE: each sample's joint (ctx FIRST)
# flash over S=N_CTX+N_IMG is BIT-IDENTICAL to b1 (padmask real_len=S).
#
# ALLOCATION: the b2 variants use PLAIN allocations (the non-scratch stream
# helpers). The SD35_SCRATCH_MLP ring is sized to the b1 [N_IMG,MLP] F32 class;
# the b2 x-stream MLP transients are [2*N_IMG,MLP] (2x that class) and MUST NOT
# reuse the b1-sized ring, so the b2 arm accepts the per-op alloc storm for now
# (correctness first — a dedicated 2x ring is a later VRAM tuning step).
# ═══════════════════════════════════════════════════════════════════════════
def _stack2_f32(a: List[Float32], b: List[Float32], D: Int, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    for i in range(D):
        v.append(a[i])
    for i in range(D):
        v.append(b[i])
    return Tensor.from_host(v^, [2, D], STDtype.F32, ctx)


# pack two per-sample host ModVecs into an SD35ModVecsDevice whose six fields are
# each [2, D] F32 (row 0 = sample0) — the per-sample adaLN pack for the b2 path.
def sd35_modvecs_pack_b2(mv0: ModVecs, mv1: ModVecs, D: Int, ctx: DeviceContext) raises -> SD35ModVecsDevice:
    return SD35ModVecsDevice(
        TArc(_stack2_f32(mv0.shift_msa, mv1.shift_msa, D, ctx)),
        TArc(_stack2_f32(mv0.scale_msa, mv1.scale_msa, D, ctx)),
        TArc(_stack2_f32(mv0.gate_msa, mv1.gate_msa, D, ctx)),
        TArc(_stack2_f32(mv0.shift_mlp, mv1.shift_mlp, D, ctx)),
        TArc(_stack2_f32(mv0.scale_mlp, mv1.scale_mlp, D, ctx)),
        TArc(_stack2_f32(mv0.gate_mlp, mv1.gate_mlp, D, ctx)),
    )


def sd35_joint_block_forward_device_b2[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    context: TArc, x: TArc,
    w: SD35JointBlockWeightsDevice,
    ctx_mod: SD35ModVecsDevice, x_mod: SD35ModVecsDevice,
    lora: SD35JointBlockLoraDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> SD35JointBlockDeviceForward:
    comptime N_CTX2 = 2 * N_CTX
    comptime N_IMG2 = 2 * N_IMG
    var ones_t = _t_f32(_ones(D), [D], ctx)
    var zeros_t = _t_f32(_zeros(D), [D], ctx)

    var cp = _sd35_stream_pre_device[H, Dh](
        context, w.ctxw, ctx_mod, lora.ctxl.qkv, N_CTX2, D, eps, qk_eps, ones_t, zeros_t, ctx)
    var xp = _sd35_stream_pre_device[H, Dh](
        x, w.xw, x_mod, lora.xl.qkv, N_IMG2, D, eps, qk_eps, ones_t, zeros_t, ctx)

    # per-sample joint attention (ctx FIRST). cp/xp.* are row-stacked [1,2N,H,Dh].
    var q0 = concat(1, ctx, slice(cp.q_rms[], 1, 0, N_CTX, ctx), slice(xp.q_rms[], 1, 0, N_IMG, ctx))
    var k0 = concat(1, ctx, slice(cp.k_rms[], 1, 0, N_CTX, ctx), slice(xp.k_rms[], 1, 0, N_IMG, ctx))
    var v0 = concat(1, ctx, slice(cp.v[], 1, 0, N_CTX, ctx), slice(xp.v[], 1, 0, N_IMG, ctx))
    var q1 = concat(1, ctx, slice(cp.q_rms[], 1, N_CTX, N_CTX, ctx), slice(xp.q_rms[], 1, N_IMG, N_IMG, ctx))
    var k1 = concat(1, ctx, slice(cp.k_rms[], 1, N_CTX, N_CTX, ctx), slice(xp.k_rms[], 1, N_IMG, N_IMG, ctx))
    var v1 = concat(1, ctx, slice(cp.v[], 1, N_CTX, N_CTX, ctx), slice(xp.v[], 1, N_IMG, N_IMG, ctx))

    var att0 = _sd35_sdpa_fwd[H, Dh, S, FLASH](q0, k0, v0, scale, ctx)   # [1,S,H,Dh]
    var att1 = _sd35_sdpa_fwd[H, Dh, S, FLASH](q1, k1, v1, scale, ctx)

    # split each per-sample att into ctx/x, re-stack row-wise per stream.
    var ctx_att_4d = concat(1, ctx,
        slice(att0, 1, 0, N_CTX, ctx), slice(att1, 1, 0, N_CTX, ctx))       # [1,N_CTX2,H,Dh]
    var x_att_4d = concat(1, ctx,
        slice(att0, 1, N_CTX, N_IMG, ctx), slice(att1, 1, N_CTX, N_IMG, ctx))  # [1,N_IMG2,H,Dh]
    var ctx_att = TArc(reshape_owned(ctx_att_4d^, [N_CTX2, D]))
    var x_att = TArc(reshape_owned(x_att_4d^, [N_IMG2, D]))

    var cpost = _sd35_stream_post_device(
        context, ctx_att, w.ctxw, ctx_mod, lora.ctxl, N_CTX2, D, MLP, eps, ones_t, zeros_t, ctx)
    var xpost = _sd35_stream_post_device(
        x, x_att, w.xw, x_mod, lora.xl, N_IMG2, D, MLP, eps, ones_t, zeros_t, ctx)

    var ctx_saved = SD35StreamSavedDevice(
        context.copy(), cp.ln1.copy(), cp.norm.copy(), cp.q_pre.copy(), cp.k_pre.copy(),
        cp.v.copy(), ctx_att.copy(), cpost.attn_res.copy(), cpost.ln2.copy(),
        cpost.mlp_in.copy(), cpost.h1.copy(), cpost.hg.copy(),
    )
    var x_saved = SD35StreamSavedDevice(
        x.copy(), xp.ln1.copy(), xp.norm.copy(), xp.q_pre.copy(), xp.k_pre.copy(),
        xp.v.copy(), x_att.copy(), xpost.attn_res.copy(), xpost.ln2.copy(),
        xpost.mlp_in.copy(), xpost.h1.copy(), xpost.hg.copy(),
    )
    return SD35JointBlockDeviceForward(
        cpost.out.copy(), xpost.out.copy(),
        SD35JointBlockSavedDevice(ctx_saved^, x_saved^),
    )


def sd35_joint_block_backward_device_b2[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    d_ctx_out: TArc, d_x_out: TArc,
    w: SD35JointBlockWeightsDevice,
    ctx_mod: SD35ModVecsDevice, x_mod: SD35ModVecsDevice,
    lora: SD35JointBlockLoraDevice,
    saved: SD35JointBlockSavedDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> SD35JointBlockLoraDeviceGrads:
    comptime N_CTX2 = 2 * N_CTX
    comptime N_IMG2 = 2 * N_IMG
    var ones_t = _t_f32(_ones(D), [D], ctx)

    var d_a = List[Optional[TArc]]()
    var d_b = List[Optional[TArc]]()
    for _ in range(SD35_SLOTS):
        d_a.append(Optional[TArc](None))
        d_b.append(Optional[TArc](None))

    var cpb = _sd35_stream_post_backward_device(
        d_ctx_out, saved.ctx_saved, w.ctxw, ctx_mod, lora.ctxl, SD_CQKV,
        N_CTX2, D, MLP, eps, ones_t, d_a, d_b, ctx,
    )
    var xpb = _sd35_stream_post_backward_device(
        d_x_out, saved.x_saved, w.xw, x_mod, lora.xl, SD_XQKV,
        N_IMG2, D, MLP, eps, ones_t, d_a, d_b, ctx,
    )

    # recompute row-stacked q_rms/k_rms from saved q/k_pre (per-row == per-sample
    # bit-identical); joint v from saved per-stream v.
    var cq = rms_norm(saved.ctx_saved.q_pre[], w.ctxw.q_norm[], qk_eps, ctx)   # [1,N_CTX2,H,Dh]
    var ck = rms_norm(saved.ctx_saved.k_pre[], w.ctxw.k_norm[], qk_eps, ctx)
    var xq = rms_norm(saved.x_saved.q_pre[], w.xw.q_norm[], qk_eps, ctx)       # [1,N_IMG2,H,Dh]
    var xk = rms_norm(saved.x_saved.k_pre[], w.xw.k_norm[], qk_eps, ctx)

    var d_catt_4d = reshape(cpb.d_att[], [1, N_CTX2, H, Dh], ctx)
    var d_xatt_4d = reshape(xpb.d_att[], [1, N_IMG2, H, Dh], ctx)

    # per-sample joint attention backward (ctx FIRST).
    var jq0 = concat(1, ctx, slice(cq, 1, 0, N_CTX, ctx), slice(xq, 1, 0, N_IMG, ctx))
    var jk0 = concat(1, ctx, slice(ck, 1, 0, N_CTX, ctx), slice(xk, 1, 0, N_IMG, ctx))
    var jv0 = concat(1, ctx, slice(saved.ctx_saved.v[], 1, 0, N_CTX, ctx), slice(saved.x_saved.v[], 1, 0, N_IMG, ctx))
    var da0 = concat(1, ctx, slice(d_catt_4d, 1, 0, N_CTX, ctx), slice(d_xatt_4d, 1, 0, N_IMG, ctx))
    var sb0 = _sd35_sdpa_bwd[H, Dh, S, FLASH](jq0, jk0, jv0, da0, scale, ctx)

    var jq1 = concat(1, ctx, slice(cq, 1, N_CTX, N_CTX, ctx), slice(xq, 1, N_IMG, N_IMG, ctx))
    var jk1 = concat(1, ctx, slice(ck, 1, N_CTX, N_CTX, ctx), slice(xk, 1, N_IMG, N_IMG, ctx))
    var jv1 = concat(1, ctx, slice(saved.ctx_saved.v[], 1, N_CTX, N_CTX, ctx), slice(saved.x_saved.v[], 1, N_IMG, N_IMG, ctx))
    var da1 = concat(1, ctx, slice(d_catt_4d, 1, N_CTX, N_CTX, ctx), slice(d_xatt_4d, 1, N_IMG, N_IMG, ctx))
    var sb1 = _sd35_sdpa_bwd[H, Dh, S, FLASH](jq1, jk1, jv1, da1, scale, ctx)

    # split ctx/x per sample, re-stack row-wise per stream.
    var c_dq = concat(1, ctx, slice(sb0.d_q, 1, 0, N_CTX, ctx), slice(sb1.d_q, 1, 0, N_CTX, ctx))
    var x_dq = concat(1, ctx, slice(sb0.d_q, 1, N_CTX, N_IMG, ctx), slice(sb1.d_q, 1, N_CTX, N_IMG, ctx))
    var c_dk = concat(1, ctx, slice(sb0.d_k, 1, 0, N_CTX, ctx), slice(sb1.d_k, 1, 0, N_CTX, ctx))
    var x_dk = concat(1, ctx, slice(sb0.d_k, 1, N_CTX, N_IMG, ctx), slice(sb1.d_k, 1, N_CTX, N_IMG, ctx))
    var c_dv = concat(1, ctx, slice(sb0.d_v, 1, 0, N_CTX, ctx), slice(sb1.d_v, 1, 0, N_CTX, ctx))
    var x_dv = concat(1, ctx, slice(sb0.d_v, 1, N_CTX, N_IMG, ctx), slice(sb1.d_v, 1, N_CTX, N_IMG, ctx))

    var d_ctx = _sd35_stream_pre_backward_device[H, Dh](
        c_dq, c_dk, c_dv, cpb.d_s, saved.ctx_saved, w.ctxw, ctx_mod,
        lora.ctxl.qkv, SD_CQKV, N_CTX2, D, eps, qk_eps, ones_t, d_a, d_b, ctx,
    )
    var d_x = _sd35_stream_pre_backward_device[H, Dh](
        x_dq, x_dk, x_dv, xpb.d_s, saved.x_saved, w.xw, x_mod,
        lora.xl.qkv, SD_XQKV, N_IMG2, D, eps, qk_eps, ones_t, d_a, d_b, ctx,
    )
    return SD35JointBlockLoraDeviceGrads(TArc(d_ctx^), TArc(d_x^), d_a^, d_b^)


# ── context_pre_only final block, TRUE BATCH-2 (row-stacked). ctx qkv-only with
#    per-sample continuous adaLN (scale/shift [2,D]); ctx attention DISCARDED (b2
#    backward feeds ZERO d_att rows for ctx per sample); only x has post+output. ─
def sd35_context_preonly_forward_device_b2[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    context: TArc, x: TArc,
    ctx_qkv_w: TArc, ctx_qkv_b: TArc,
    ctx_qnorm: TArc, ctx_knorm: TArc,
    ctx_scale: TArc, ctx_shift: TArc,       # [2,D] packed per-sample
    xw: SD35StreamWeightsDevice, x_mod: SD35ModVecsDevice,   # x_mod packed [2,D]
    ctx_qkv_lora: Optional[SD35LoraAdapterDevice],
    x_lora: SD35StreamLoraDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> SD35CtxPreDeviceForward:
    comptime N_CTX2 = 2 * N_CTX
    comptime N_IMG2 = 2 * N_IMG
    var ones_t = _t_f32(_ones(D), [D], ctx)
    var zeros_t = _t_f32(_zeros(D), [D], ctx)

    # ctx pre (qkv only), row-stacked; per-sample scale/shift broadcast.
    var ctx_ln = layer_norm(context[], ones_t, zeros_t, eps, ctx)          # [N_CTX2,D]
    var ctx_norm = modulate(ctx_ln, ctx_scale[], ctx_shift[], ctx)         # [2,D] broadcast
    var cb = Optional[Tensor](ctx_qkv_b[].clone(ctx))
    var cqkv_base = linear(ctx_norm, ctx_qkv_w[], cb, ctx)
    var cqkv = _lora_apply_f32(cqkv_base^, ctx_norm, ctx_qkv_lora, N_CTX2, ctx)
    var cq_flat = slice(cqkv, 1, 0, D, ctx)
    var ck_flat = slice(cqkv, 1, D, D, ctx)
    var cv_flat = slice(cqkv, 1, 2 * D, D, ctx)
    var cq_pre = reshape_owned(cq_flat^, [1, N_CTX2, H, Dh])
    var ck_pre = reshape_owned(ck_flat^, [1, N_CTX2, H, Dh])
    var cv = reshape_owned(cv_flat^, [1, N_CTX2, H, Dh])
    var cq_rms = rms_norm(cq_pre, ctx_qnorm[], qk_eps, ctx)
    var ck_rms = rms_norm(ck_pre, ctx_knorm[], qk_eps, ctx)

    var xp = _sd35_stream_pre_device[H, Dh](
        x, xw, x_mod, x_lora.qkv, N_IMG2, D, eps, qk_eps, ones_t, zeros_t, ctx)

    # per-sample joint attention; the ctx attention slice is DISCARDED.
    var q0 = concat(1, ctx, slice(cq_rms, 1, 0, N_CTX, ctx), slice(xp.q_rms[], 1, 0, N_IMG, ctx))
    var k0 = concat(1, ctx, slice(ck_rms, 1, 0, N_CTX, ctx), slice(xp.k_rms[], 1, 0, N_IMG, ctx))
    var v0 = concat(1, ctx, slice(cv, 1, 0, N_CTX, ctx), slice(xp.v[], 1, 0, N_IMG, ctx))
    var att0 = _sd35_sdpa_fwd[H, Dh, S, FLASH](q0, k0, v0, scale, ctx)
    var q1 = concat(1, ctx, slice(cq_rms, 1, N_CTX, N_CTX, ctx), slice(xp.q_rms[], 1, N_IMG, N_IMG, ctx))
    var k1 = concat(1, ctx, slice(ck_rms, 1, N_CTX, N_CTX, ctx), slice(xp.k_rms[], 1, N_IMG, N_IMG, ctx))
    var v1 = concat(1, ctx, slice(cv, 1, N_CTX, N_CTX, ctx), slice(xp.v[], 1, N_IMG, N_IMG, ctx))
    var att1 = _sd35_sdpa_fwd[H, Dh, S, FLASH](q1, k1, v1, scale, ctx)

    var x_att_4d = concat(1, ctx,
        slice(att0, 1, N_CTX, N_IMG, ctx), slice(att1, 1, N_CTX, N_IMG, ctx))   # [1,N_IMG2,H,Dh]
    var x_att = TArc(reshape_owned(x_att_4d^, [N_IMG2, D]))

    var xpost = _sd35_stream_post_device(
        x, x_att, xw, x_mod, x_lora, N_IMG2, D, MLP, eps, ones_t, zeros_t, ctx)

    var x_saved = SD35StreamSavedDevice(
        x.copy(), xp.ln1.copy(), xp.norm.copy(), xp.q_pre.copy(), xp.k_pre.copy(),
        xp.v.copy(), x_att.copy(), xpost.attn_res.copy(), xpost.ln2.copy(),
        xpost.mlp_in.copy(), xpost.h1.copy(), xpost.hg.copy(),
    )
    var saved = SD35CtxPreSavedDevice(
        context.copy(), TArc(ctx_ln^), TArc(ctx_norm^),
        TArc(cq_pre^), TArc(ck_pre^), TArc(cv^), x_saved^,
    )
    return SD35CtxPreDeviceForward(xpost.out.copy(), saved^)


def sd35_context_preonly_backward_device_b2[
    H: Int, Dh: Int, N_CTX: Int, N_IMG: Int, S: Int, FLASH: Bool = True
](
    d_x_out: TArc,
    ctx_qkv_w: TArc, ctx_qnorm: TArc, ctx_knorm: TArc, ctx_scale: TArc,   # ctx_scale [2,D]
    xw: SD35StreamWeightsDevice, x_mod: SD35ModVecsDevice,                # x_mod [2,D]
    ctx_qkv_lora: Optional[SD35LoraAdapterDevice],
    x_lora: SD35StreamLoraDevice,
    saved: SD35CtxPreSavedDevice,
    D: Int, MLP: Int, eps: Float32, qk_eps: Float32, scale: Float32,
    ctx: DeviceContext,
) raises -> SD35CtxPreDeviceGrads:
    comptime N_CTX2 = 2 * N_CTX
    comptime N_IMG2 = 2 * N_IMG
    var ones_t = _t_f32(_ones(D), [D], ctx)

    var d_a = List[Optional[TArc]]()
    var d_b = List[Optional[TArc]]()
    for _ in range(SD35_SLOTS):
        d_a.append(Optional[TArc](None))
        d_b.append(Optional[TArc](None))

    var xpb = _sd35_stream_post_backward_device(
        d_x_out, saved.x_saved, xw, x_mod, x_lora, SD_XQKV,
        N_IMG2, D, MLP, eps, ones_t, d_a, d_b, ctx,
    )

    var cq = rms_norm(saved.ctx_q_pre[], ctx_qnorm[], qk_eps, ctx)   # [1,N_CTX2,H,Dh]
    var ck = rms_norm(saved.ctx_k_pre[], ctx_knorm[], qk_eps, ctx)
    var xq = rms_norm(saved.x_saved.q_pre[], xw.q_norm[], qk_eps, ctx)   # [1,N_IMG2,H,Dh]
    var xk = rms_norm(saved.x_saved.k_pre[], xw.k_norm[], qk_eps, ctx)

    var d_xatt_4d = reshape(xpb.d_att[], [1, N_IMG2, H, Dh], ctx)
    # ZERO ctx d_att rows (per sample); the ctx attention output was discarded.
    var zero_catt = Tensor.from_host(
        _zeros(N_CTX * H * Dh), [1, N_CTX, H, Dh], STDtype.F32, ctx)

    var jq0 = concat(1, ctx, slice(cq, 1, 0, N_CTX, ctx), slice(xq, 1, 0, N_IMG, ctx))
    var jk0 = concat(1, ctx, slice(ck, 1, 0, N_CTX, ctx), slice(xk, 1, 0, N_IMG, ctx))
    var jv0 = concat(1, ctx, slice(saved.ctx_v[], 1, 0, N_CTX, ctx), slice(saved.x_saved.v[], 1, 0, N_IMG, ctx))
    var da0 = concat(1, ctx, zero_catt, slice(d_xatt_4d, 1, 0, N_IMG, ctx))
    var sb0 = _sd35_sdpa_bwd[H, Dh, S, FLASH](jq0, jk0, jv0, da0, scale, ctx)

    var jq1 = concat(1, ctx, slice(cq, 1, N_CTX, N_CTX, ctx), slice(xq, 1, N_IMG, N_IMG, ctx))
    var jk1 = concat(1, ctx, slice(ck, 1, N_CTX, N_CTX, ctx), slice(xk, 1, N_IMG, N_IMG, ctx))
    var jv1 = concat(1, ctx, slice(saved.ctx_v[], 1, N_CTX, N_CTX, ctx), slice(saved.x_saved.v[], 1, N_IMG, N_IMG, ctx))
    var da1 = concat(1, ctx, zero_catt, slice(d_xatt_4d, 1, N_IMG, N_IMG, ctx))
    var sb1 = _sd35_sdpa_bwd[H, Dh, S, FLASH](jq1, jk1, jv1, da1, scale, ctx)

    var c_dq = concat(1, ctx, slice(sb0.d_q, 1, 0, N_CTX, ctx), slice(sb1.d_q, 1, 0, N_CTX, ctx))
    var x_dq = concat(1, ctx, slice(sb0.d_q, 1, N_CTX, N_IMG, ctx), slice(sb1.d_q, 1, N_CTX, N_IMG, ctx))
    var c_dk = concat(1, ctx, slice(sb0.d_k, 1, 0, N_CTX, ctx), slice(sb1.d_k, 1, 0, N_CTX, ctx))
    var x_dk = concat(1, ctx, slice(sb0.d_k, 1, N_CTX, N_IMG, ctx), slice(sb1.d_k, 1, N_CTX, N_IMG, ctx))
    var c_dv = concat(1, ctx, slice(sb0.d_v, 1, 0, N_CTX, ctx), slice(sb1.d_v, 1, 0, N_CTX, ctx))
    var x_dv = concat(1, ctx, slice(sb0.d_v, 1, N_CTX, N_IMG, ctx), slice(sb1.d_v, 1, N_CTX, N_IMG, ctx))

    var d_x = _sd35_stream_pre_backward_device[H, Dh](
        x_dq, x_dk, x_dv, xpb.d_s, saved.x_saved, xw, x_mod,
        x_lora.qkv, SD_XQKV, N_IMG2, D, eps, qk_eps, ones_t, d_a, d_b, ctx,
    )

    # ctx pre backward (qkv only; frozen base dx-only), row-stacked M=2N.
    var d_cq_pre = rms_norm_backward_dx(c_dq, saved.ctx_q_pre[], ctx_qnorm[], qk_eps, ctx)
    var d_ck_pre = rms_norm_backward_dx(c_dk, saved.ctx_k_pre[], ctx_knorm[], qk_eps, ctx)
    var d_cq_flat = reshape(d_cq_pre, [N_CTX2, D], ctx)
    var d_ck_flat = reshape(d_ck_pre, [N_CTX2, D], ctx)
    var d_cv_flat = reshape(c_dv, [N_CTX2, D], ctx)
    var d_cqkv = concat(1, ctx, d_cq_flat, d_ck_flat, d_cv_flat)   # [N_CTX2,3D]

    var d_ctx_norm = _proj_bwd_lora_f32(
        d_cqkv, saved.ctx_norm[], ctx_qkv_w[], ctx_qkv_lora, SD_CQKV,
        N_CTX2, D, 3 * D, d_a, d_b, ctx,
    )
    var mb_c = modulate_backward(d_ctx_norm, saved.ctx_ln[], ctx_scale[], ctx, False)
    var d_context = layer_norm_backward_dx(mb_c.d_x, saved.ctx_input[], ones_t, eps, ctx)

    return SD35CtxPreDeviceGrads(TArc(d_x^), TArc(d_context^), d_a^, d_b^)
