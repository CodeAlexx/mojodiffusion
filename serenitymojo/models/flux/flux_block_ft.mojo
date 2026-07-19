# serenitymojo/models/flux/flux_block_ft.mojo
#
# FULL-FINETUNE device block backwards for the FLUX-family block (flux = chroma:
# chroma_block_device.mojo's header — "DEVICE-RESIDENT Chroma (= Flux)"; the two
# models share the block math BYTE-FOR-BYTE, only the weight-key layout differs
# and the chroma loader fuses that away). Chroma full-FT rollout item 3
# (docs/FULL_FINETUNE_ROLLOUT_PLAN_2026-07-07.md); because the math is 100%
# flux-shared these live FLUX-side and chroma RE-EXPORTS them (mirroring how
# chroma_block.mojo already imports flux's host backwards) — this IS flux P1
# (the flux FT block backward) at the same time.
#
# Same hand-chain as the chroma device-resident LoRA backwards
# (chroma_double_block_lora_backward_device / chroma_single_block_lora_
# backward_device), but the base matmul WEIGHTS are TRAINABLE: each projection
# site emits d_W = d_yᵀ @ x_saved via ops/linalg_backward.linear_backward_dw
# with EXPLICIT STDtype.F32 output (the BOOL default sentinel means "match
# input dtype" -> bf16 grads the F32 optimizer rejects — krea2 trap), alongside
# the plain bf16 d_x carry (linear_backward_dx). NO LoRA anywhere, NO int8
# (reference trainer full-FT forbids quantized linears — quantized forwards detach the weight).
#
# Trainable surface is GATED (don't fork) by the comptime param SURFACE_V2:
#
# SURFACE_V2=False (the v1 default — flux's CURRENT 10-arm behavior, unchanged):
#   double, per stream: wqkv [3D,D], wproj [D,D], wmlp0 [Fmlp,D], wmlp2 [D,Fmlp]
#   single:             w1 [3D+Fmlp,D], w2 [D,D+Fmlp]
#   biases / q_norm/k_norm rms scales frozen (dx-only arms).
#
# SURFACE_V2=True (FULL_SURFACE_PLAN_2026-07-08 Phase B rows 3/4; chroma's v2
# arm consumes this TODAY, flux's own Phase B will follow): the SAME matmul dW
# arms PLUS, appended to the SAME dw list in the FT_DBL_SLOT_*/FT_SGL_SLOT_*
# fixed order below:
#   every linear BIAS grad d_b = d_y row-sum over the tokens
#     (ops/linalg_backward.linear_backward_db — F32-accumulated colsum; taken
#     from the F32 pre-cast d_y where the chain has one, else bf16 + cast F32),
#   q_norm/k_norm rms-scale grads d_g via the FULL ops/norm_backward.
#     rms_norm_backward (F32-accumulated d_g kernel, store-dtype out, cast F32
#     for the optimizer — the krea2 dW-dtype trap).
# STILL frozen under SURFACE_V2: the modulation vectors — for chroma they are
# genuinely frozen (distilled approximator rows, a Phase C candidate); for
# flux the per-block mod.lin Linears are flux's OWN Phase B delta (adds its
# own arms; not emitted here).
#
# dX fold order (C15) matches the chroma LoRA _device backwards EXACTLY:
#   double post-stream: gate2-dxdy -> wmlp2-dx -> gelu -> wmlp0-dx -> modulate
#     -> ln2 -> d_attn_res_total = add(_bf16(grg2.d_x), norm_branch) ->
#     gate1-dxdy -> wproj-dx; joint d_att concat (txt FIRST) -> sdpa -> rope ->
#     cat_backward splits -> pre-stream (rms-dx -> d_qkv concat(q,k,v) ->
#     wqkv-dx -> modulate -> ln1) -> d_x = add(post_residual, pre_norm) [F32].
#   single: gate-dxdy -> w2-dx -> cat_backward(D,Fmlp) -> gelu/sdpa arms ->
#     d_qkv concat(q,k,v) -> d_fused concat(d_qkv, d_mlp_in) -> w1-dx ->
#     modulate -> ln -> d_x = add(residual_branch, norm_branch) [F32].
# (gate_residual_backward_dxdy's d_x/d_y are identical to the full
# gate_residual_backward's — its docstring — so dropping the y recompute
# changes nothing in the dX chain.)
#
# SDPA arm: same _chroma_sdpa_bwd dispatch as the LoRA backwards (FLASH=True
# default = cuDNN padmask flash regenerating o/stats; FLASH=False = the math
# arm, the parity-gate mode).
#
# Oracle: chroma_block_oracle.py CHROMA_FT_ORACLE=1 (base-only mode — the
# base-block oracle refs must exclude the LoRA dX contribution; dW alone would
# also differ because LoRA shifts downstream activations). Gate:
# models/chroma/parity/chroma_block_ft_parity.mojo, cos >= 0.999. Flux runs the
# SAME weights layout at the SAME real dims (D=3072, H=24, Dh=128), so that
# gate is the flux FT math gate too.
#
# Mojo current beta: `def`, `comptime`, `var`, explicit `raises`.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import (
    linear_backward_dx, linear_backward_dw, linear_backward_db,
)
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_in_place, concat, add,
)
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, rms_norm_backward_dx, layer_norm_backward_dx,
)
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, gate_residual_backward_dxdy, rope_backward,
)
from serenitymojo.ops.shape_backward import cat_backward

from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)

# REUSE the chroma device block substrate (byte-for-byte the flux block; the
# same reuse direction flux_block_device.mojo already established).
from serenitymojo.models.chroma.chroma_block_device import (
    ModVecsDevice, SingleModVecsDevice,
    StreamSavedDevice, DoubleBlockSavedDevice, SingleBlockSavedDevice,
    _f32, _bf16, _t_bf16, _ones,
    _chroma_sdpa_bwd,
)


comptime TArc = ArcPointer[Tensor]

# v2 FULL-SURFACE slot indices (FULL_SURFACE_PLAN_2026-07-08 Phase B rows 3/4;
# slots 0-7 double / 0-1 single are the v1 matmuls in the dw-list order below).
# FIXED order — host store tails, af_states flat index, sidecar and the parity
# gate all key off it. Only emitted when SURFACE_V2=True.
comptime FT_DBL_SLOT_IMG_BQKV = 8    # img_attn.qkv.bias   [3D]
comptime FT_DBL_SLOT_IMG_BPROJ = 9   # img_attn.proj.bias  [D]
comptime FT_DBL_SLOT_IMG_BMLP0 = 10  # img_mlp.0.bias      [Fmlp]
comptime FT_DBL_SLOT_IMG_BMLP2 = 11  # img_mlp.2.bias      [D]
comptime FT_DBL_SLOT_TXT_BQKV = 12   # txt_attn.qkv.bias   [3D]
comptime FT_DBL_SLOT_TXT_BPROJ = 13  # txt_attn.proj.bias  [D]
comptime FT_DBL_SLOT_TXT_BMLP0 = 14  # txt_mlp.0.bias      [Fmlp]
comptime FT_DBL_SLOT_TXT_BMLP2 = 15  # txt_mlp.2.bias      [D]
comptime FT_DBL_SLOT_IMG_QNORM = 16  # img q rms scale     [Dh]
comptime FT_DBL_SLOT_IMG_KNORM = 17  # img k rms scale     [Dh]
comptime FT_DBL_SLOT_TXT_QNORM = 18  # txt q rms scale     [Dh]
comptime FT_DBL_SLOT_TXT_KNORM = 19  # txt k rms scale     [Dh]
comptime FT_DBL_SLOTS_V2 = 20
comptime FT_SGL_SLOT_B1 = 2          # linear1.bias        [3D+Fmlp]
comptime FT_SGL_SLOT_B2 = 3          # linear2.bias        [D]
comptime FT_SGL_SLOT_QNORM = 4       # q rms scale         [Dh]
comptime FT_SGL_SLOT_KNORM = 5       # k rms scale         [Dh]
comptime FT_SGL_SLOTS_V2 = 6


# ═══════════════════════════════════════════════════════════════════════════
# FT grad carriers (dW device F32; d_x device F32 inter-block carries)
# ═══════════════════════════════════════════════════════════════════════════
struct FluxDoubleBlockFTGrads(Movable):
    var d_img_x: TArc      # [N_IMG,D] F32 img input grad (inter-block carry)
    var d_txt_x: TArc      # [N_TXT,D] F32 txt input grad (inter-block carry)
    # F32 device, FIXED order. len 8 (SURFACE_V2=False) / 20 (=True):
    #   [0] img wqkv  [3D,D]     [1] img wproj [D,D]
    #   [2] img wmlp0 [Fmlp,D]   [3] img wmlp2 [D,Fmlp]
    #   [4] txt wqkv  [3D,D]     [5] txt wproj [D,D]
    #   [6] txt wmlp0 [Fmlp,D]   [7] txt wmlp2 [D,Fmlp]
    #   [8-15] biases, [16-19] q/k rms scales (the FT_DBL_SLOT_* map above).
    var dw: List[TArc]
    # FLUX mod-grad chain (MOD_GRADS=True; EMPTY otherwise — chroma freezes
    # mods). Per-stream modulation FLAT grads, [1,6D] F32 in _modvecs_from_flat
    # chunk order [shift1,scale1,gate1,shift2,scale2,gate2]:
    #   [0] = d_img_mod_flat, [1] = d_txt_mod_flat.
    # The stack routes each through its mod.lin: d_W = dwᵀ@silu(vec), d_b=colsum.
    var mod_flat: List[TArc]

    def __init__(
        out self, var d_img_x: TArc, var d_txt_x: TArc, var dw: List[TArc],
        var mod_flat: List[TArc],
    ):
        self.d_img_x = d_img_x^
        self.d_txt_x = d_txt_x^
        self.dw = dw^
        self.mod_flat = mod_flat^


struct FluxSingleBlockFTGrads(Movable):
    var d_x: TArc          # [S,D] F32 input grad (the inter-block carry)
    # F32 device, FIXED order. len 2 (SURFACE_V2=False) / 6 (=True):
    #   [0]=d_w1 [3D+Fmlp,D], [1]=d_w2 [D,D+Fmlp],
    #   [2]=d_b1, [3]=d_b2, [4]=d_q_norm, [5]=d_k_norm (FT_SGL_SLOT_* map).
    var dw: List[TArc]
    # FLUX mod-grad chain (MOD_GRADS=True; EMPTY otherwise). [0] = d_mod_flat
    # [1,3D] F32 in _single_modvecs_from_flat order [shift,scale,gate].
    var mod_flat: List[TArc]

    def __init__(out self, var d_x: TArc, var dw: List[TArc], var mod_flat: List[TArc]):
        self.d_x = d_x^
        self.dw = dw^
        self.mod_flat = mod_flat^


# ── per-stream post backward (dW for wproj/wmlp0/wmlp2; mods frozen; biases
# trainable under SURFACE_V2) ─────────────────────────────────────────────────
struct _FluxStreamPostBackFT(Movable):
    var d_x: TArc        # [N,D] F32 residual branch into the stream input
    var d_att: TArc      # [N,D] bf16 grad into the joint-attention slice
    var dw_proj: TArc    # F32 [D,D]
    var dw_mlp0: TArc    # F32 [Fmlp,D]
    var dw_mlp2: TArc    # F32 [D,Fmlp]
    # SURFACE_V2 bias grads, COMPUTE order: [d_bmlp2 [D], d_bmlp0 [Fmlp],
    # d_bproj [D]] (F32). EMPTY on the v1 path (flux's current behavior).
    var db: List[TArc]
    # MOD_GRADS modulation-flat grads, COMPUTE (append) order:
    #   [0]=d_gate2 [D], [1]=d_shift2 [D], [2]=d_scale2 [D], [3]=d_gate1 [D]
    # (F32). EMPTY unless MOD_GRADS (flux only).
    var modp: List[TArc]

    def __init__(
        out self, var d_x: TArc, var d_att: TArc,
        var dw_proj: TArc, var dw_mlp0: TArc, var dw_mlp2: TArc,
        var db: List[TArc], var modp: List[TArc],
    ):
        self.d_x = d_x^
        self.d_att = d_att^
        self.dw_proj = dw_proj^
        self.dw_mlp0 = dw_mlp0^
        self.dw_mlp2 = dw_mlp2^
        self.db = db^
        self.modp = modp^


def _flux_stream_post_backward_ft_dev[
    SURFACE_V2: Bool = False, MOD_GRADS: Bool = False
](
    d_out: TArc,
    w: StreamWeights, mv: ModVecsDevice, sv: StreamSavedDevice,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor,
    ctx: DeviceContext,
) raises -> _FluxStreamPostBackFT:
    var db = List[TArc]()
    var modp = List[TArc]()

    # final = residual_gate(attn_res, gate2, mlp); mod grads frozen -> dxdy only.
    # gate2_f32 matches d_out's F32 (the same dtypes the LoRA path feeds
    # gate_residual_backward; dxdy's d_x/d_y are identical by contract).
    var grg2 = gate_residual_backward_dxdy(d_out[], mv.gate2_f32[], ctx)
    var d_y2_b = _bf16(grg2.d_y, ctx)

    comptime if MOD_GRADS:
        # d_gate2 = sum_rows d_out * mlp_output; mlp = linear(mlp_h, Wmlp2,
        # bmlp2) is not saved -> recompute it (WITH bias, the gated value), then
        # the FULL gate_residual_backward d_g (grad_out+gate F32; d_x/d_y match
        # the dxdy chain above so the dX path is unchanged).
        var mlp_y = linear(sv.mlp_h[], w.wmlp2[], Optional[Tensor](w.bmlp2[].clone(ctx)), ctx)
        var grg2f = gate_residual_backward(d_out[], sv.attn_res[], mv.gate2_f32[], mlp_y, ctx, True)
        modp.append(TArc(grg2f.d_g.clone(ctx)))   # [0] d_gate2 (F32)

    comptime if SURFACE_V2:
        # d_bmlp2 = d_y2 row-sum over the N tokens; colsum the F32 pre-cast
        # grad (F32 in -> F32 out, no extra rounding).
        db.append(TArc(linear_backward_db(grg2.d_y, N, D, ctx)))

    # mlp = linear(mlp_h, Wmlp2, bmlp2): TRAINABLE weight -> d_W (F32) + d_x carry.
    var dw_mlp2 = linear_backward_dw(d_y2_b, sv.mlp_h[], N, Fmlp, D, ctx, STDtype.F32)
    var pm2_dx = linear_backward_dx(d_y2_b, w.wmlp2[], N, Fmlp, D, ctx)   # [N,Fmlp] bf16

    var d_mlp_pre = gelu_backward(pm2_dx, sv.mlp_pre[], ctx)              # bf16

    comptime if SURFACE_V2:
        # d_bmlp0 = d_mlp_pre row-sum; the gelu chain is bf16-native -> cast
        # the grad F32 FIRST so the F32-accumulated colsum lands F32 directly
        # (no output bf16 rounding — measured parity win).
        db.append(TArc(linear_backward_db(_f32(d_mlp_pre, ctx), N, Fmlp, ctx)))

    # mlp_pre = linear(mlp_in, Wmlp0, bmlp0): TRAINABLE weight.
    var dw_mlp0 = linear_backward_dw(d_mlp_pre, sv.mlp_in[], N, D, Fmlp, ctx, STDtype.F32)
    var pm0_dx = linear_backward_dx(d_mlp_pre, w.wmlp0[], N, D, Fmlp, ctx)   # [N,D] bf16

    # mlp_in = modulate(ln2, scale2, shift2); mod frozen (v1/chroma) OR
    # trainable (flux MOD_GRADS: same dx kernel, + d_scale2/d_shift2 [D]).
    var mb2 = modulate_backward(pm0_dx, sv.ln2[], mv.scale2[], ctx, MOD_GRADS)
    comptime if MOD_GRADS:
        modp.append(TArc(cast_tensor(mb2.d_shift, STDtype.F32, ctx)))   # [1] d_shift2
        modp.append(TArc(cast_tensor(mb2.d_scale, STDtype.F32, ctx)))   # [2] d_scale2
    var lnb2_dx = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)   # bf16
    # attn_res feeds residual (grg2.d_x F32) AND ln2 (bf16): host sums bf16, then F32.
    var d_attn_res_total = add(_bf16(grg2.d_x, ctx), lnb2_dx, ctx)        # bf16
    var d_attn_res_f32 = _f32(d_attn_res_total, ctx)

    # attn_res = residual_gate(x, gate1, proj_out); mod grads frozen -> the
    # residual branch d_x IS d_attn_res (dxdy passthrough).
    var grg1 = gate_residual_backward_dxdy(d_attn_res_f32, mv.gate1_f32[], ctx)
    var d_y1_b = _bf16(grg1.d_y, ctx)

    comptime if MOD_GRADS:
        # d_gate1 = sum_rows d_attn_res * proj_out; proj_out = linear(att,
        # Wproj, bproj) not saved -> recompute (WITH bias); FULL d_g.
        var proj_y = linear(sv.att[], w.wproj[], Optional[Tensor](w.bproj[].clone(ctx)), ctx)
        var grg1f = gate_residual_backward(d_attn_res_f32, sv.x[], mv.gate1_f32[], proj_y, ctx, True)
        modp.append(TArc(grg1f.d_g.clone(ctx)))   # [3] d_gate1 (F32)

    comptime if SURFACE_V2:
        # d_bproj = d_y1 row-sum (F32 pre-cast grad -> F32 out).
        db.append(TArc(linear_backward_db(grg1.d_y, N, D, ctx)))

    # proj_out = linear(att, Wproj, bproj): TRAINABLE weight.
    var dw_proj = linear_backward_dw(d_y1_b, sv.att[], N, D, D, ctx, STDtype.F32)
    var d_att = linear_backward_dx(d_y1_b, w.wproj[], N, D, D, ctx)       # [N,D] bf16

    return _FluxStreamPostBackFT(
        TArc(grg1.d_x.clone(ctx)), TArc(d_att^),
        TArc(dw_proj^), TArc(dw_mlp0^), TArc(dw_mlp2^),
        db^, modp^,
    )


# ── per-stream pre backward (dW for wqkv; mods frozen; q/k rms scales + bqkv
# trainable under SURFACE_V2) ─────────────────────────────────────────────────
struct _FluxStreamPreBackFT(Movable):
    var d_x: TArc        # [N,D] F32 layer_norm branch into the stream input
    var dw_qkv: TArc     # F32 [3D,D]
    # SURFACE_V2 extras, COMPUTE order: [d_q_norm [Dh], d_k_norm [Dh],
    # d_bqkv [3D]] (F32). EMPTY on the v1 path (flux's current behavior).
    var extra: List[TArc]
    # MOD_GRADS modulation-flat grads, COMPUTE order: [d_shift1 [D],
    # d_scale1 [D]] (F32). EMPTY unless MOD_GRADS (flux only).
    var modp: List[TArc]

    def __init__(
        out self, var d_x: TArc, var dw_qkv: TArc, var extra: List[TArc],
        var modp: List[TArc],
    ):
        self.d_x = d_x^
        self.dw_qkv = dw_qkv^
        self.extra = extra^
        self.modp = modp^


def _flux_stream_pre_backward_ft_dev[
    H: Int, Dh: Int, SURFACE_V2: Bool = False, MOD_GRADS: Bool = False
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecsDevice, sv: StreamSavedDevice,
    N: Int, D: Int, eps: Float32, ones: Tensor,
    ctx: DeviceContext,
) raises -> _FluxStreamPreBackFT:
    var extra = List[TArc]()
    var modp = List[TArc]()
    # incoming grads (cat_backward outputs) F32 -> bf16 for the bf16-native chain.
    var dq_b = _bf16(d_q_rms, ctx)
    var dk_b = _bf16(d_k_rms, ctx)
    var dv_b = _bf16(d_v, ctx)
    var d_v_flat = reshape(dv_b, [N, D], ctx)
    var d_qkv: Tensor   # [N,3D] bf16
    comptime if SURFACE_V2:
        # q/k rms scales TRAINABLE (v2) -> full rms backward (d_x + d_g) over
        # q_pre/k_pre [1,N,H,Dh] (d_g [Dh], rows = N*H); d_g arrives bf16
        # (store dtype, F32-accumulated dg kernel) -> cast F32 for the opt.
        # The result structs stay whole (Tensor field moves are disallowed —
        # explicit-destroy buffer handle); d_x is reshaped/consumed in place.
        var rb_q = rms_norm_backward(dq_b, sv.q_pre[], w.q_norm[], eps, ctx)
        var rb_k = rms_norm_backward(dk_b, sv.k_pre[], w.k_norm[], eps, ctx)
        extra.append(TArc(cast_tensor(rb_q.d_g, STDtype.F32, ctx)))
        extra.append(TArc(cast_tensor(rb_k.d_g, STDtype.F32, ctx)))
        reshape_in_place(rb_q.d_x, [N, D])
        reshape_in_place(rb_k.d_x, [N, D])
        d_qkv = concat(1, ctx, rb_q.d_x, rb_k.d_x, d_v_flat)
    else:
        # q/k rms scales FROZEN in v1 -> dx-only rms backward.
        var rb_q_dx = rms_norm_backward_dx(dq_b, sv.q_pre[], w.q_norm[], eps, ctx)   # [1,N,H,Dh]
        var rb_k_dx = rms_norm_backward_dx(dk_b, sv.k_pre[], w.k_norm[], eps, ctx)
        reshape_in_place(rb_q_dx, [N, D])
        reshape_in_place(rb_k_dx, [N, D])
        d_qkv = concat(1, ctx, rb_q_dx, rb_k_dx, d_v_flat)

    comptime if SURFACE_V2:
        # d_bqkv = d_qkv row-sum over the N tokens (bf16 chain -> cast the
        # grad F32 FIRST: F32-accumulated colsum lands F32 directly, no
        # output bf16 rounding).
        extra.append(TArc(linear_backward_db(_f32(d_qkv, ctx), N, 3 * D, ctx)))

    # qkv = linear(norm, Wqkv, bqkv): TRAINABLE weight -> d_W (F32) + d_x carry.
    var dw_qkv = linear_backward_dw(d_qkv, sv.norm[], N, D, 3 * D, ctx, STDtype.F32)
    var base_d_norm = linear_backward_dx(d_qkv, w.wqkv[], N, D, 3 * D, ctx)   # [N,D] bf16

    # norm = modulate(ln1, scale1, shift1); mod frozen (v1/chroma) OR trainable
    # (flux MOD_GRADS: same dx kernel, + d_scale1/d_shift1 [D]).
    var mb1 = modulate_backward(base_d_norm, sv.ln1[], mv.scale1[], ctx, MOD_GRADS)
    comptime if MOD_GRADS:
        modp.append(TArc(cast_tensor(mb1.d_shift, STDtype.F32, ctx)))   # [0] d_shift1
        modp.append(TArc(cast_tensor(mb1.d_scale, STDtype.F32, ctx)))   # [1] d_scale1
    var lnb1_dx = layer_norm_backward_dx(mb1.d_x, sv.x[], ones, eps, ctx)   # bf16
    return _FluxStreamPreBackFT(TArc(_f32(lnb1_dx, ctx)), TArc(dw_qkv^), extra^, modp^)


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block — FT device backward (8 dW F32 + d_img_x/d_txt_x F32)
# ═══════════════════════════════════════════════════════════════════════════
def flux_double_block_ft_backward_dev[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, FLASH: Bool = True,
    SURFACE_V2: Bool = False, MOD_GRADS: Bool = False
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    saved: DoubleBlockSavedDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxDoubleBlockFTGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)

    var ipb = _flux_stream_post_backward_ft_dev[SURFACE_V2, MOD_GRADS](
        d_io_t, w.img, img_mod, saved.img, N_IMG, D, Fmlp, eps, ones_t, ctx,
    )
    var tpb = _flux_stream_post_backward_ft_dev[SURFACE_V2, MOD_GRADS](
        d_to_t, w.txt, txt_mod, saved.txt, N_TXT, D, Fmlp, eps, ones_t, ctx,
    )

    # join per-stream d_att into joint d_att (txt FIRST); reshape [N,D]->[1,N,H,Dh].
    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh] bf16

    var sb = _chroma_sdpa_bwd[H, Dh, S, FLASH](
        saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx)

    # F32-ONLY: rope_backward grad + cos/sin F32. sb.d_q/d_k bf16 -> cast up.
    var d_q_joint = rope_backward(_f32(sb.d_q, ctx), cos, sin, True, ctx)
    var d_k_joint = rope_backward(_f32(sb.d_k, ctx), cos, sin, True, ctx)

    # F32-ONLY: cat_backward grad F32. sb.d_v bf16 -> cast up.
    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(_f32(sb.d_v, ctx), N_TXT, N_IMG, 1, ctx)

    var iprb = _flux_stream_pre_backward_ft_dev[H, Dh, SURFACE_V2, MOD_GRADS](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, saved.img, N_IMG, D, eps, ones_t, ctx,
    )
    var tprb = _flux_stream_pre_backward_ft_dev[H, Dh, SURFACE_V2, MOD_GRADS](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, saved.txt, N_TXT, D, eps, ones_t, ctx,
    )

    # C15: post residual branch FIRST, pre norm branch second (the LoRA order).
    var d_img_x = add(ipb.d_x[], iprb.d_x[], ctx)   # F32
    var d_txt_x = add(tpb.d_x[], tprb.d_x[], ctx)

    var dw = List[TArc]()
    dw.append(iprb.dw_qkv.copy())
    dw.append(ipb.dw_proj.copy())
    dw.append(ipb.dw_mlp0.copy())
    dw.append(ipb.dw_mlp2.copy())
    dw.append(tprb.dw_qkv.copy())
    dw.append(tpb.dw_proj.copy())
    dw.append(tpb.dw_mlp0.copy())
    dw.append(tpb.dw_mlp2.copy())
    comptime if SURFACE_V2:
        # slots 8-19 (FT_DBL_SLOT_* map): 8 biases then 4 q/k rms scales.
        # pre.extra = [d_q_norm, d_k_norm, d_bqkv] (compute order);
        # post.db   = [d_bmlp2, d_bmlp0, d_bproj] (compute order).
        dw.append(iprb.extra[2].copy())   # 8  img bqkv
        dw.append(ipb.db[2].copy())       # 9  img bproj
        dw.append(ipb.db[1].copy())       # 10 img bmlp0
        dw.append(ipb.db[0].copy())       # 11 img bmlp2
        dw.append(tprb.extra[2].copy())   # 12 txt bqkv
        dw.append(tpb.db[2].copy())       # 13 txt bproj
        dw.append(tpb.db[1].copy())       # 14 txt bmlp0
        dw.append(tpb.db[0].copy())       # 15 txt bmlp2
        dw.append(iprb.extra[0].copy())   # 16 img q_norm
        dw.append(iprb.extra[1].copy())   # 17 img k_norm
        dw.append(tprb.extra[0].copy())   # 18 txt q_norm
        dw.append(tprb.extra[1].copy())   # 19 txt k_norm

    var mod_flat = List[TArc]()
    comptime if MOD_GRADS:
        # per-stream [6D] mod.lin FLAT grad in _modvecs_from_flat chunk order
        # [shift1, scale1, gate1, shift2, scale2, gate2]; pre.modp =
        # [d_shift1, d_scale1], post.modp = [d_gate2, d_shift2, d_scale2, d_gate1].
        var im_flat = concat(
            0, ctx,
            iprb.modp[0][], iprb.modp[1][], ipb.modp[3][],
            ipb.modp[1][], ipb.modp[2][], ipb.modp[0][],
        )
        reshape_in_place(im_flat, [1, 6 * D])
        var tm_flat = concat(
            0, ctx,
            tprb.modp[0][], tprb.modp[1][], tpb.modp[3][],
            tpb.modp[1][], tpb.modp[2][], tpb.modp[0][],
        )
        reshape_in_place(tm_flat, [1, 6 * D])
        mod_flat.append(TArc(im_flat^))
        mod_flat.append(TArc(tm_flat^))
    return FluxDoubleBlockFTGrads(TArc(d_img_x^), TArc(d_txt_x^), dw^, mod_flat^)


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE block — FT device backward (2 dW F32 + d_x F32)
# ═══════════════════════════════════════════════════════════════════════════
def flux_single_block_ft_backward_dev[
    H: Int, Dh: Int, S: Int, FLASH: Bool = True, SURFACE_V2: Bool = False,
    MOD_GRADS: Bool = False
](
    d_out: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice,
    saved: SingleBlockSavedDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxSingleBlockFTGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)
    # SURFACE_V2 grads, filled at their compute sites (fixed vars, slot-mapped
    # in the dw assembly at the bottom).
    var v2 = List[TArc]()   # compute order: [d_b2, d_q_norm, d_k_norm, d_b1]
    var modp = List[TArc]()  # MOD_GRADS compute order: [d_gate, d_shift, d_scale]

    # result = residual_gate(x, gate, out); mod grads frozen -> dxdy only.
    var grg = gate_residual_backward_dxdy(d_out[], mv.gate_f32[], ctx)
    var d_y_b = _bf16(grg.d_y, ctx)

    comptime if MOD_GRADS:
        # d_gate = sum_rows d_out * out; out = linear(out_in, W2, b2) not saved
        # -> recompute (WITH bias, the gated value); FULL gate d_g (F32).
        var out_y = linear(saved.out_in[], w.w2[], Optional[Tensor](w.b2[].clone(ctx)), ctx)
        var grgf = gate_residual_backward(d_out[], saved.x[], mv.gate_f32[], out_y, ctx, True)
        modp.append(TArc(grgf.d_g.clone(ctx)))   # [0] d_gate (F32)

    comptime if SURFACE_V2:
        # d_b2 = d_y row-sum over the S tokens (F32 pre-cast grad -> F32 out).
        v2.append(TArc(linear_backward_db(grg.d_y, S, D, ctx)))

    # out = linear(out_in, W2, b2): TRAINABLE weight -> d_W (F32) + d_x carry.
    var dw_w2 = linear_backward_dw(d_y_b, saved.out_in[], S, D + Fmlp, D, ctx, STDtype.F32)
    var pl2_dx = linear_backward_dx(d_y_b, w.w2[], S, D + Fmlp, D, ctx)   # [S,D+Fmlp] bf16

    # out_in = concat(att_flat, mlp_h). F32-ONLY cat_backward.
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

    reshape_in_place(sb.d_v, [S, D])
    var d_qkv: Tensor   # [S,3D] bf16
    comptime if SURFACE_V2:
        # q/k rms scales TRAINABLE (v2) -> full rms backward (d_x + d_g) over
        # q_pre/k_pre [1,S,H,Dh] (d_g [Dh], rows = S*H); d_g arrives bf16 ->
        # cast F32 for the optimizer. The result structs stay whole (Tensor
        # field moves are disallowed — explicit-destroy buffer handle); d_x is
        # reshaped/consumed in place.
        var rb_q = rms_norm_backward(_bf16(d_q_rms, ctx), saved.q_pre[], w.q_norm[], eps, ctx)
        var rb_k = rms_norm_backward(_bf16(d_k_rms, ctx), saved.k_pre[], w.k_norm[], eps, ctx)
        v2.append(TArc(cast_tensor(rb_q.d_g, STDtype.F32, ctx)))
        v2.append(TArc(cast_tensor(rb_k.d_g, STDtype.F32, ctx)))
        reshape_in_place(rb_q.d_x, [S, D])
        reshape_in_place(rb_k.d_x, [S, D])
        d_qkv = concat(1, ctx, rb_q.d_x, rb_k.d_x, sb.d_v)
    else:
        # q/k rms scales FROZEN in v1 -> dx-only rms backward.
        var rb_q_dx = rms_norm_backward_dx(_bf16(d_q_rms, ctx), saved.q_pre[], w.q_norm[], eps, ctx)
        var rb_k_dx = rms_norm_backward_dx(_bf16(d_k_rms, ctx), saved.k_pre[], w.k_norm[], eps, ctx)
        reshape_in_place(rb_q_dx, [S, D])
        reshape_in_place(rb_k_dx, [S, D])
        d_qkv = concat(1, ctx, rb_q_dx, rb_k_dx, sb.d_v)
    var d_fused = concat(1, ctx, d_qkv, d_mlp_in)          # [S,3D+Fmlp]

    comptime if SURFACE_V2:
        # d_b1 = d_fused row-sum over the S tokens (bf16 chain -> cast the
        # grad F32 FIRST: F32-accumulated colsum lands F32 directly, no
        # output bf16 rounding).
        v2.append(TArc(linear_backward_db(_f32(d_fused, ctx), S, 3 * D + Fmlp, ctx)))

    # fused = linear(norm, W1, b1): TRAINABLE weight -> d_W (F32) + d_x carry.
    var dw_w1 = linear_backward_dw(d_fused, saved.norm[], S, D, 3 * D + Fmlp, ctx, STDtype.F32)
    var base_d_norm = linear_backward_dx(d_fused, w.w1[], S, D, 3 * D + Fmlp, ctx)   # [S,D] bf16

    # norm = modulate(ln, scale, shift); mod frozen (v1/chroma) OR trainable
    # (flux MOD_GRADS: same dx kernel, + d_scale/d_shift [D]).
    var mb = modulate_backward(base_d_norm, saved.ln[], mv.scale[], ctx, MOD_GRADS)
    comptime if MOD_GRADS:
        modp.append(TArc(cast_tensor(mb.d_shift, STDtype.F32, ctx)))   # [1] d_shift
        modp.append(TArc(cast_tensor(mb.d_scale, STDtype.F32, ctx)))   # [2] d_scale
    var lnb_dx = layer_norm_backward_dx(mb.d_x, saved.x[], ones_t, eps, ctx)   # bf16

    # x feeds BOTH the residual (grg.d_x) AND layer_norm(x) -> SUM (C15 order).
    var d_x = add(grg.d_x, _f32(lnb_dx, ctx), ctx)   # F32

    var dw = List[TArc]()
    dw.append(TArc(dw_w1^))
    dw.append(TArc(dw_w2^))
    comptime if SURFACE_V2:
        # slots 2-5 (FT_SGL_SLOT_* map); v2 compute order was
        # [d_b2, d_q_norm, d_k_norm, d_b1].
        dw.append(v2[3].copy())   # 2 b1
        dw.append(v2[0].copy())   # 3 b2
        dw.append(v2[1].copy())   # 4 q_norm
        dw.append(v2[2].copy())   # 5 k_norm

    var mod_flat = List[TArc]()
    comptime if MOD_GRADS:
        # [3D] mod.lin FLAT grad in _single_modvecs_from_flat order
        # [shift, scale, gate]; modp = [d_gate, d_shift, d_scale].
        var sm_flat = concat(0, ctx, modp[1][], modp[2][], modp[0][])
        reshape_in_place(sm_flat, [1, 3 * D])
        mod_flat.append(TArc(sm_flat^))
    return FluxSingleBlockFTGrads(TArc(d_x^), dw^, mod_flat^)
