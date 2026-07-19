# serenitymojo/models/flux/flux_block_device.mojo
#
# DEVICE-RESIDENT FLUX DOUBLE + SINGLE LoRA block BACKWARD that ALSO emits the
# modulation-vector grads (d_shift/d_scale/d_gate).
#
# The FLUX (= Chroma) block MATH is identical between the two models, so the
# device FORWARD (chroma_double_block_lora_forward_device / _single_..) and every
# low-level LoRA/proj helper are REUSED verbatim from chroma_block_device.mojo —
# they are byte-for-byte the flux block (that file's header: "DEVICE-RESIDENT
# Chroma (= Flux)"). The ONLY thing flux needs on top is the BACKWARD emission of
# the per-block modulation-vector gradients, because flux TRAINS the modulation
# Linears (86 stack adapters) whereas chroma FREEZES them (distilled approximator)
# and DISCARDS these grads.
#
# The mod grads come from the SAME modulate_backward / gate_residual_backward
# calls chroma already makes: chroma passes compute_param_grads=False and drops
# d_scale/d_shift; gate_residual_backward already computes d_g (default True) and
# chroma drops it. Here we flip the modulate flag to True and READ the four/two
# param grads, pack them into the host [6D]/[3D] flat in the exact host block
# order (shift1,scale1,gate1,shift2,scale2,gate2 / shift,scale,gate — see
# flux_stack._modvec6 / _single_modvec3), and return them alongside d_x + the
# block LoRA grads. Because every op + input is byte-identical to chroma's proven
# bit-identical d_x path, the mod grads are byte-identical to the HOST flux block
# backward (double_block_lora_backward / single_block_lora_backward)'s d_im/d_tm/
# d_sm — validated by flux_block_device_parity.
#
# Mojo current beta: `def`, `comptime`, `var`, explicit `raises`.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_in_place, slice, concat, add,
)
from serenitymojo.ops.norm_backward import (
    rms_norm_backward_dx, layer_norm_backward_dx,
)
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import gate_residual_backward, rope_backward
from serenitymojo.ops.shape_backward import cat_backward

from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.lora_block import (
    DBL_STREAM_SLOTS, D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    SGL_SLOTS, S_SQ, S_SK, S_SV, S_PMLP, S_L2,
)

# REUSE the chroma device block: structs, low-level LoRA/proj/dtype helpers.
from serenitymojo.models.chroma.chroma_block_device import (
    StreamLoraDevice, DoubleBlockLoraDevice, SingleBlockLoraDevice,
    ModVecsDevice, SingleModVecsDevice,
    StreamSavedDevice, DoubleBlockSavedDevice, SingleBlockSavedDevice,
    _lora_apply_device, _lora_bwd_device, _proj_bwd_with_lora_device,
    _f32, _bf16, _t_bf16, _ones,
    _chroma_sdpa_bwd, _ChromaSdpaGrads,
)


comptime TArc = ArcPointer[Tensor]


# ── flux mod-vec grad packing (host [6D]/[3D], host block order) ──────────────
def _cat6(
    a: List[Float32], b: List[Float32], c: List[Float32],
    d: List[Float32], e: List[Float32], f: List[Float32],
) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)): o.append(a[i])
    for i in range(len(b)): o.append(b[i])
    for i in range(len(c)): o.append(c[i])
    for i in range(len(d)): o.append(d[i])
    for i in range(len(e)): o.append(e[i])
    for i in range(len(f)): o.append(f[i])
    return o^


def _cat3(a: List[Float32], b: List[Float32], c: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)): o.append(a[i])
    for i in range(len(b)): o.append(b[i])
    for i in range(len(c)): o.append(c[i])
    return o^


# ── flux per-stream grads: chroma StreamLoraDeviceGrads + [6D] mod flat ───────
@fieldwise_init
struct FluxStreamLoraDeviceGrads(Copyable, Movable):
    var d_x: TArc                    # [N,D]  F32
    var d_a: List[Optional[TArc]]    # DBL_STREAM_SLOTS entries
    var d_b: List[Optional[TArc]]
    var d_modvec6: List[Float32]     # [6D] host: shift1,scale1,gate1,shift2,scale2,gate2


@fieldwise_init
struct FluxDoubleBlockDeviceGrads(Copyable, Movable):
    var img: FluxStreamLoraDeviceGrads
    var txt: FluxStreamLoraDeviceGrads


@fieldwise_init
struct FluxSingleBlockDeviceGrads(Copyable, Movable):
    var d_x: TArc                    # [S,D]  F32
    var d_a: List[Optional[TArc]]    # SGL_SLOTS entries
    var d_b: List[Optional[TArc]]
    var d_modvec3: List[Float32]     # [3D] host: shift,scale,gate


# ── per-stream post backward (mod grads: d_gate1, d_shift2, d_scale2, d_gate2) ─
@fieldwise_init
struct _FluxStreamPostBackDevice(Copyable, Movable):
    var d_x: TArc          # [N,D] F32 (stream input grad via gate1 residual)
    var d_att: TArc        # [N,D] bf16 (grad into joint-attention slice)
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]


def _flux_stream_post_backward_device(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecsDevice, sv: StreamSavedDevice, lo: StreamLoraDevice,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
) raises -> _FluxStreamPostBackDevice:
    # recompute mlp output WITH LoRA(mlp2) so gate_residual_backward y matches fwd.
    var b2 = Optional[Tensor](w.bmlp2[].clone(ctx))
    var mlp_base = linear(sv.mlp_h[], w.wmlp2[], b2, ctx)
    var mlp_y = _lora_apply_device(mlp_base^, sv.mlp_h[], lo.mlp2, N, ctx)   # bf16
    # gate2: FULL-F32 (host `_tf32`); d_out F32. d_g = d_gate2.
    var grg2 = gate_residual_backward(d_out[], sv.attn_res[], mv.gate2_f32[], mlp_y, ctx)
    var d_gate2 = grg2.d_g.to_host(ctx)

    var pm2_dx = _proj_bwd_with_lora_device(
        _bf16(grg2.d_y, ctx), sv.mlp_h[], w.wmlp2[], lo.mlp2, D_MLP2, N, Fmlp, D,
        d_a_slots, d_b_slots, ctx,
    )                                                                       # [N,Fmlp] bf16
    var d_mlp_pre = gelu_backward(pm2_dx, sv.mlp_pre[], ctx)                # bf16

    var pm0_dx = _proj_bwd_with_lora_device(
        d_mlp_pre, sv.mlp_in[], w.wmlp0[], lo.mlp0, D_MLP0, N, D, Fmlp,
        d_a_slots, d_b_slots, ctx,
    )                                                                       # [N,D] bf16

    # mlp_in = modulate(ln2, scale2, shift2); TRAINABLE mod -> emit d_scale2/d_shift2.
    var mb2 = modulate_backward(pm0_dx, sv.ln2[], mv.scale2[], ctx, True)
    var d_scale2 = mb2.d_scale.to_host(ctx)
    var d_shift2 = mb2.d_shift.to_host(ctx)
    var lnb2_dx = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)   # bf16
    var d_attn_res_total = add(_bf16(grg2.d_x, ctx), lnb2_dx, ctx)          # bf16
    var d_attn_res_f32 = _f32(d_attn_res_total, ctx)

    # attn_res = residual_gate(x, gate1, proj_out): recompute proj WITH LoRA(proj).
    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var proj_base = linear(att[], w.wproj[], bp, ctx)
    var proj_out = _lora_apply_device(proj_base^, att[], lo.proj, N, ctx)   # bf16
    var grg1 = gate_residual_backward(d_attn_res_f32, x[], mv.gate1_f32[], proj_out, ctx)
    var d_gate1 = grg1.d_g.to_host(ctx)

    var d_att = _proj_bwd_with_lora_device(
        _bf16(grg1.d_y, ctx), att[], w.wproj[], lo.proj, D_PROJ, N, D, D,
        d_a_slots, d_b_slots, ctx,
    )                                                                       # [N,D] bf16
    return _FluxStreamPostBackDevice(
        TArc(grg1.d_x.clone(ctx)), TArc(d_att^),
        d_gate1^, d_shift2^, d_scale2^, d_gate2^,
    )


# ── per-stream pre backward (mod grads: d_shift1, d_scale1) ───────────────────
@fieldwise_init
struct _FluxStreamPreBackDevice(Copyable, Movable):
    var d_x: TArc          # [N,D] F32 (stream input grad via norm path)
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]


def _flux_stream_pre_backward_device[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecsDevice, sv: StreamSavedDevice, lo: StreamLoraDevice,
    N: Int, D: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
) raises -> _FluxStreamPreBackDevice:
    var dq_b = _bf16(d_q_rms, ctx)
    var dk_b = _bf16(d_k_rms, ctx)
    var dv_b = _bf16(d_v, ctx)
    var rb_q_dx = rms_norm_backward_dx(dq_b, sv.q_pre[], w.q_norm[], eps, ctx)   # [1,N,H,Dh]
    var rb_k_dx = rms_norm_backward_dx(dk_b, sv.k_pre[], w.k_norm[], eps, ctx)
    reshape_in_place(rb_q_dx, [N, D])
    reshape_in_place(rb_k_dx, [N, D])
    var d_v_flat = reshape(dv_b, [N, D], ctx)

    var d_qkv = concat(1, ctx, rb_q_dx, rb_k_dx, d_v_flat)   # [N,3D] bf16
    var base_d_norm = linear_backward_dx(d_qkv, w.wqkv[], N, D, 3 * D, ctx)   # [N,D] bf16
    var d_norm_f32 = _f32(base_d_norm, ctx)

    if lo.to_q:
        var lg = _lora_bwd_device(rb_q_dx, sv.norm[], lo.to_q.value(), N, ctx)
        d_a_slots[D_SQ] = Optional[TArc](lg.d_a)
        d_b_slots[D_SQ] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lo.to_k:
        var lg = _lora_bwd_device(rb_k_dx, sv.norm[], lo.to_k.value(), N, ctx)
        d_a_slots[D_SK] = Optional[TArc](lg.d_a)
        d_b_slots[D_SK] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lo.to_v:
        var lg = _lora_bwd_device(d_v_flat, sv.norm[], lo.to_v.value(), N, ctx)
        d_a_slots[D_SV] = Optional[TArc](lg.d_a)
        d_b_slots[D_SV] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)

    # norm = modulate(ln1, scale1, shift1); TRAINABLE mod -> emit d_scale1/d_shift1.
    var mb1 = modulate_backward(_bf16(d_norm_f32, ctx), sv.ln1[], mv.scale1[], ctx, True)
    var d_scale1 = mb1.d_scale.to_host(ctx)
    var d_shift1 = mb1.d_shift.to_host(ctx)
    var lnb1_dx = layer_norm_backward_dx(mb1.d_x, sv.x[], ones, eps, ctx)   # bf16
    return _FluxStreamPreBackDevice(TArc(_f32(lnb1_dx, ctx)), d_shift1^, d_scale1^)


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block — device backward (LoRA d_a/d_b + d_x + [6D] mod grads per stream)
# ═══════════════════════════════════════════════════════════════════════════
def flux_double_block_lora_backward_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, FLASH: Bool = True
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    lora: DoubleBlockLoraDevice,
    saved: DoubleBlockSavedDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxDoubleBlockDeviceGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)

    var ia = List[Optional[TArc]]()
    var ib = List[Optional[TArc]]()
    var ta = List[Optional[TArc]]()
    var tb = List[Optional[TArc]]()
    for _ in range(DBL_STREAM_SLOTS):
        ia.append(Optional[TArc](None)); ib.append(Optional[TArc](None))
        ta.append(Optional[TArc](None)); tb.append(Optional[TArc](None))

    var ipb = _flux_stream_post_backward_device(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, saved.img, lora.img,
        N_IMG, D, Fmlp, eps, ones_t, ia, ib, ctx,
    )
    var tpb = _flux_stream_post_backward_device(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT, D, Fmlp, eps, ones_t, ta, tb, ctx,
    )

    # join per-stream d_att into joint d_att (txt FIRST); reshape [N,D]->[1,N,H,Dh].
    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh] bf16

    var sb = _chroma_sdpa_bwd[H, Dh, S, FLASH](
        saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx)

    var d_q_joint = rope_backward(_f32(sb.d_q, ctx), cos, sin, True, ctx)
    var d_k_joint = rope_backward(_f32(sb.d_k, ctx), cos, sin, True, ctx)

    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(_f32(sb.d_v, ctx), N_TXT, N_IMG, 1, ctx)

    var iprb = _flux_stream_pre_backward_device[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, saved.img, lora.img,
        N_IMG, D, eps, ones_t, ia, ib, ctx,
    )
    var tprb = _flux_stream_pre_backward_device[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT, D, eps, ones_t, ta, tb, ctx,
    )

    var d_img_x = add(ipb.d_x[], iprb.d_x[], ctx)   # F32
    var d_txt_x = add(tpb.d_x[], tprb.d_x[], ctx)

    # pack [6D] mod grads in host block order: shift1,scale1,gate1,shift2,scale2,gate2.
    var img_mv6 = _cat6(
        iprb.d_shift1, iprb.d_scale1, ipb.d_gate1,
        ipb.d_shift2, ipb.d_scale2, ipb.d_gate2,
    )
    var txt_mv6 = _cat6(
        tprb.d_shift1, tprb.d_scale1, tpb.d_gate1,
        tpb.d_shift2, tpb.d_scale2, tpb.d_gate2,
    )

    var img_g = FluxStreamLoraDeviceGrads(TArc(d_img_x^), ia^, ib^, img_mv6^)
    var txt_g = FluxStreamLoraDeviceGrads(TArc(d_txt_x^), ta^, tb^, txt_mv6^)
    return FluxDoubleBlockDeviceGrads(img_g^, txt_g^)


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE block — device backward (LoRA d_a/d_b + d_x + [3D] mod grads)
# ═══════════════════════════════════════════════════════════════════════════
def flux_single_block_lora_backward_device[
    H: Int, Dh: Int, S: Int, FLASH: Bool = True
](
    d_out: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    saved: SingleBlockSavedDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxSingleBlockDeviceGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)

    var d_a_slots = List[Optional[TArc]]()
    var d_b_slots = List[Optional[TArc]]()
    for _ in range(SGL_SLOTS):
        d_a_slots.append(Optional[TArc](None)); d_b_slots.append(Optional[TArc](None))

    # result = residual_gate(x, gate, out): recompute out WITH LoRA(linear2).
    var b2 = Optional[Tensor](w.b2[].clone(ctx))
    var out_base = linear(saved.out_in[], w.w2[], b2, ctx)
    var out_y = _lora_apply_device(out_base^, saved.out_in[], lora.linear2, S, ctx)   # bf16
    var grg = gate_residual_backward(d_out[], saved.x[], mv.gate_f32[], out_y, ctx)
    var d_gate = grg.d_g.to_host(ctx)

    var pl2_dx = _proj_bwd_with_lora_device(
        _bf16(grg.d_y, ctx), saved.out_in[], w.w2[], lora.linear2, S_L2,
        S, D + Fmlp, D, d_a_slots, d_b_slots, ctx,
    )                                                            # [S, D+Fmlp] bf16

    var dx_w2_f32 = _f32(pl2_dx, ctx)
    reshape_in_place(dx_w2_f32, [1, S, D + Fmlp])
    var cb = cat_backward(dx_w2_f32, D, Fmlp, 2, ctx)
    reshape_in_place(cb.d_0, [1, S, H, Dh])
    reshape_in_place(cb.d_1, [S, Fmlp])
    var d_att_flat_b = _bf16(cb.d_0, ctx)
    var d_mlp_h_b = _bf16(cb.d_1, ctx)

    var d_mlp_in = gelu_backward(d_mlp_h_b, saved.mlp_in[], ctx)   # [S,Fmlp] bf16

    var sb = _chroma_sdpa_bwd[H, Dh, S, FLASH](
        saved.q_rope[], saved.k_rope[], saved.v[], d_att_flat_b, scale, ctx)

    var d_q_rms = rope_backward(_f32(sb.d_q, ctx), cos, sin, True, ctx)
    var d_k_rms = rope_backward(_f32(sb.d_k, ctx), cos, sin, True, ctx)

    var rb_q_dx = rms_norm_backward_dx(_bf16(d_q_rms, ctx), saved.q_pre[], w.q_norm[], eps, ctx)
    var rb_k_dx = rms_norm_backward_dx(_bf16(d_k_rms, ctx), saved.k_pre[], w.k_norm[], eps, ctx)
    reshape_in_place(rb_q_dx, [S, D])
    reshape_in_place(rb_k_dx, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    var d_qkv = concat(1, ctx, rb_q_dx, rb_k_dx, sb.d_v)   # [S,3D]
    var d_fused = concat(1, ctx, d_qkv, d_mlp_in)          # [S,3D+Fmlp]

    var base_d_norm = linear_backward_dx(d_fused, w.w1[], S, D, 3 * D + Fmlp, ctx)   # [S,D]
    var d_norm_f32 = _f32(base_d_norm, ctx)

    if lora.to_q:
        var lg = _lora_bwd_device(rb_q_dx, saved.norm[], lora.to_q.value(), S, ctx)
        d_a_slots[S_SQ] = Optional[TArc](lg.d_a); d_b_slots[S_SQ] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lora.to_k:
        var lg = _lora_bwd_device(rb_k_dx, saved.norm[], lora.to_k.value(), S, ctx)
        d_a_slots[S_SK] = Optional[TArc](lg.d_a); d_b_slots[S_SK] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lora.to_v:
        var lg = _lora_bwd_device(sb.d_v, saved.norm[], lora.to_v.value(), S, ctx)
        d_a_slots[S_SV] = Optional[TArc](lg.d_a); d_b_slots[S_SV] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lora.proj_mlp:
        var lg = _lora_bwd_device(d_mlp_in, saved.norm[], lora.proj_mlp.value(), S, ctx)
        d_a_slots[S_PMLP] = Optional[TArc](lg.d_a); d_b_slots[S_PMLP] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)

    # norm = modulate(ln, scale, shift); TRAINABLE mod -> emit d_scale/d_shift.
    var mb = modulate_backward(_bf16(d_norm_f32, ctx), saved.ln[], mv.scale[], ctx, True)
    var d_scale = mb.d_scale.to_host(ctx)
    var d_shift = mb.d_shift.to_host(ctx)
    var lnb_dx = layer_norm_backward_dx(mb.d_x, saved.x[], ones_t, eps, ctx)   # bf16
    var d_x = add(grg.d_x, _f32(lnb_dx, ctx), ctx)   # F32

    var mv3 = _cat3(d_shift, d_scale, d_gate)
    return FluxSingleBlockDeviceGrads(TArc(d_x^), d_a_slots^, d_b_slots^, mv3^)


# ═══════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 (row-stacked) — flux b2 REUSES the chroma device b2 blocks VERBATIM.
#
# Why no flux-specific b2 block lives here: the ONLY thing that makes the flux
# device b1 BACKWARD differ from chroma's (this file's b1 code above) is the
# emission of the per-block modulation-vector grads (d_shift/d_scale/d_gate, the
# 86 stack-adapter surface) — flux TRAINS the modulation Linears, chroma freezes
# them. TRUE batch-2 packs the mods per-sample as [2, D] adaLN, and the shared
# modulate_backward/gate_residual_backward PARAM-grad reductions are unsupported
# for [B, D] scales/gates with B>1 (elementwise_backward.modulate_backward raises;
# gate_residual_backward's d_g reduction is a single-vector reduction). So flux b2
# — exactly like the landed HOST flux b2 (models/flux/lora_block.double_block_lora_
# backward_b2, which passes compute_param_grads=False / compute_gate_grad=False) —
# is BLOCK-PROJECTION LoRA ONLY: the mod grads are DISCARDED. With mod grads gone,
# the flux b2 double/single fwd+bwd are BYTE-IDENTICAL to chroma's device b2 blocks
# (models/chroma/chroma_block_device.chroma_{double,single}_block_lora_{forward,
# backward}_device_b2 — the block MATH is the same flux block). The flux b2 device
# STACK conductors (models/flux/flux_stack_lora.flux_stack_lora_*_device_offload_
# full_b2) therefore call those chroma b2 blocks directly (FLASH threaded), just as
# the flux b1 device conductor already calls chroma_double_block_lora_forward_device
# for the b1 FORWARD. No new flux b2 block code is needed or wanted (adding thin
# wrappers would be dead duplication). Production b2 + mod-adapter training is a
# recorded follow-up gated on a [B, D] param-grad reduction in the shared ops.
# ═══════════════════════════════════════════════════════════════════════════
