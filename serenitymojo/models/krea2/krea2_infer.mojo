# models/krea2/krea2_infer.mojo — Krea-2 INFERENCE sampler stack (worker-facing).
#
# The inference-only subset hoisted from the trainer's proven inline sampler
# (train_krea2.mojo::_krea2_sample_resident_latent + _build_conditioning +
# _krea2_inline_cond_* + load_krea2_resident_cond), plus the LoRA overlay loader.
# It imports ONLY inference deps (krea2_dit / krea2_stack / krea2_cache_reader /
# klein.lora_block / training.lora_save reader) — NOT the trainer's autograd_v2 /
# optimizer stack — so the serenity_worker_krea2 binary stays lean.
#
# Fixed-LTMAX length-bucket path (NOT the exact-LT comptime krea2_pipeline path):
# every prompt (natural LT <= LTMAX) is padded to LTMAX and reordered to
# [TXT_real(lt) | IMG(imglen) | TXT_pad(tail)] with real_len = lt+imglen for the
# cuDNN flash-padmask. ONE comptime arm (LFULL = LTMAX + imglen) serves all
# prompts — no per-prompt rebuild. This is exactly the contract the krea2 LoRA
# trainer used (so a trained LoRA overlays faithfully).
#
# Mojo 1.0.0b1, NVIDIA GPU. Inference-only.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.io.env import env_or
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    add, sub, mul_scalar, reshape, concat, slice, permute, zeros_device,
)
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.random import randn
from serenitymojo.ops.random_torch import randn_torch
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.sampling.krea2_sampler import (
    krea2_packed_seq_len, krea2_timesteps, krea2_cfg, krea2_euler_step,
)
from serenitymojo.sampling.inpaint import (
    mask_blend, lanpaint_coef_c, lanpaint_flow_score,
    lanpaint_overdamped_advance, LanPaintDampedStep,
    lanpaint_damped_advance,
)
from serenitymojo.models.krea2.krea2_cache_reader import (
    krea2_patchify, krea2_build_pos, krea2_build_refiner_mask,
    krea2_build_pos_cond, krea2_build_pos_cond_grid,
    krea2_reorder_combined_edit, krea2_edit_real_len,
    krea2_build_edit_attn_bias, krea2_edit_cond_bias,
    KREA2_TXT_LAYERS, KREA2_TXT_DIM, KREA2_HEADS,
)
from serenitymojo.training.krea2_omini_layout import (
    Krea2OminiLayout, krea2_omini_mod_split,
)
from serenitymojo.models.dit.krea2_dit import (
    krea2_first, krea2_temb, krea2_tmlp, krea2_tproj, krea2_txtmlp,
    krea2_text_fusion, build_krea2_rope,
    _wb, _scale, _txtf_bundle, Krea2TextFusionWeights, Krea2SharedResident,
)
from serenitymojo.models.krea2.krea2_block import Krea2BlockLora
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackLora, Krea2StreamFinal, KREA2_SLOTS_PER_BLOCK,
    krea2_stack_lora_forward_streamed,
    Krea2ResidentFp8, build_krea2_resident_fp8,
    Krea2ResidentInt8, build_krea2_resident_int8,
    Krea2ResidentSquareq, build_krea2_resident_squareq,
)
from serenitymojo.models.dit.krea2_dit import Krea2HostInt8Inf
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice, lora_adapter_to_device,
    klein_lora_fwd_device_resident_unfused,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_save import NamedLora, load_lora_for_resume
from serenitymojo.serve.proc_ipc import (
    write_msg, wait_sampling_ack, prefault_self_executable,
    prefault_mapped_shared_libraries,
)

comptime TArc = ArcPointer[Tensor]

# ── arch constants (single_mmdit_large_wide) — match train_krea2 / krea2_dit ──
comptime FEATURES = 6144
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime TXTHEADS = 20
comptime TXTHD = 128
comptime NLAYERS_TXT = 12
comptime TDIM = 256
comptime NBLOCKS = 28
comptime EPS = Float32(1.0e-5)
comptime THETA = Float32(1.0e3)


# ══════════════════════════════════════════════════════════════════════════════
# RESIDENT CONDITIONING WEIGHTS (bf16, loaded once). The conditioning FORWARD runs
# per step (per-sample context) but reads from THIS resident set (no per-step disk
# read). Byte-identical to the per-step _wb/_scale/_txtf_bundle loads.
# ══════════════════════════════════════════════════════════════════════════════
struct Krea2ResidentCond(Copyable, Movable):
    var first_w: TArc
    var first_b: TArc
    var tmlp0_w: TArc
    var tmlp0_b: TArc
    var tmlp2_w: TArc
    var tmlp2_b: TArc
    var tproj1_w: TArc
    var tproj1_b: TArc
    var lw0: Krea2TextFusionWeights
    var lw1: Krea2TextFusionWeights
    var rf0: Krea2TextFusionWeights
    var rf1: Krea2TextFusionWeights
    var projector_w: TArc
    var txtmlp0_scale: TArc
    var txtmlp1_w: TArc
    var txtmlp1_b: TArc
    var txtmlp3_w: TArc
    var txtmlp3_b: TArc

    def __init__(
        out self,
        var first_w: TArc, var first_b: TArc,
        var tmlp0_w: TArc, var tmlp0_b: TArc, var tmlp2_w: TArc, var tmlp2_b: TArc,
        var tproj1_w: TArc, var tproj1_b: TArc,
        var lw0: Krea2TextFusionWeights, var lw1: Krea2TextFusionWeights,
        var rf0: Krea2TextFusionWeights, var rf1: Krea2TextFusionWeights,
        var projector_w: TArc,
        var txtmlp0_scale: TArc, var txtmlp1_w: TArc, var txtmlp1_b: TArc,
        var txtmlp3_w: TArc, var txtmlp3_b: TArc,
    ):
        self.first_w = first_w^
        self.first_b = first_b^
        self.tmlp0_w = tmlp0_w^
        self.tmlp0_b = tmlp0_b^
        self.tmlp2_w = tmlp2_w^
        self.tmlp2_b = tmlp2_b^
        self.tproj1_w = tproj1_w^
        self.tproj1_b = tproj1_b^
        self.lw0 = lw0^
        self.lw1 = lw1^
        self.rf0 = rf0^
        self.rf1 = rf1^
        self.projector_w = projector_w^
        self.txtmlp0_scale = txtmlp0_scale^
        self.txtmlp1_w = txtmlp1_w^
        self.txtmlp1_b = txtmlp1_b^
        self.txtmlp3_w = txtmlp3_w^
        self.txtmlp3_b = txtmlp3_b^


def load_krea2_resident_cond(
    st: ShardedSafeTensors, key_prefix: String, ctx: DeviceContext
) raises -> Krea2ResidentCond:
    return Krea2ResidentCond(
        TArc(_wb(st, key_prefix + "first.weight", ctx)),
        TArc(_wb(st, key_prefix + "first.bias", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.0.weight", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.0.bias", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.2.weight", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.2.bias", ctx)),
        TArc(_wb(st, key_prefix + "tproj.1.weight", ctx)),
        TArc(_wb(st, key_prefix + "tproj.1.bias", ctx)),
        _txtf_bundle(st, key_prefix + "txtfusion.layerwise_blocks.0", ctx),
        _txtf_bundle(st, key_prefix + "txtfusion.layerwise_blocks.1", ctx),
        _txtf_bundle(st, key_prefix + "txtfusion.refiner_blocks.0", ctx),
        _txtf_bundle(st, key_prefix + "txtfusion.refiner_blocks.1", ctx),
        TArc(_wb(st, key_prefix + "txtfusion.projector.weight", ctx)),
        TArc(_scale(st, key_prefix + "txtmlp.0.scale", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.1.weight", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.1.bias", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.3.weight", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.3.bias", ctx)),
    )


def krea2_resident_cond_from_shared(
    shared: Krea2SharedResident,
) raises -> Krea2ResidentCond:
    """View the v2 sidecar's resident shared store as T2I conditioning weights."""
    if len(shared.t) != 18 or len(shared.txtf) != 4:
        raise Error("krea2 shared sidecar has an invalid conditioning layout")
    return Krea2ResidentCond(
        shared.t[0].copy(), shared.t[1].copy(),
        shared.t[2].copy(), shared.t[3].copy(),
        shared.t[4].copy(), shared.t[5].copy(),
        shared.t[6].copy(), shared.t[7].copy(),
        shared.txtf[0].copy(), shared.txtf[1].copy(),
        shared.txtf[2].copy(), shared.txtf[3].copy(),
        shared.t[8].copy(), shared.t[9].copy(),
        shared.t[10].copy(), shared.t[11].copy(),
        shared.t[12].copy(), shared.t[13].copy(),
    )


def krea2_stream_final_from_shared(
    shared: Krea2SharedResident,
) raises -> Krea2StreamFinal:
    """View the v2 sidecar's resident shared store as the T2I final layer."""
    if len(shared.t) != 18:
        raise Error("krea2 shared sidecar has an invalid final-layer layout")
    return Krea2StreamFinal(
        shared.t[14].copy(), shared.t[15].copy(),
        shared.t[16].copy(), shared.t[17].copy(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONING — the frozen krea2_forward prefix (steps 1-11) up to the block
# stack: produce combined [1,LFULL,F], blk_vec, tmlp_out, and per-token rope.
# The length-bucket REORDER makes combined = [TXT_real | IMG | TXT_pad].
# ══════════════════════════════════════════════════════════════════════════════
struct _Cond(Movable):
    var combined: TArc
    var blk_vec: Tensor
    var tmlp_out: Tensor
    var cos: Tensor
    var sin: Tensor
    # OminiControl EDIT (C8): mods(t=0) for the COND rows. Present ONLY on a
    # CONDL > 0 build — the twin of the trainer's _CondEdit.blk_vec_cond.
    var blk_vec_cond: Optional[Tensor]

    def __init__(
        out self, var combined: TArc, var blk_vec: Tensor, var tmlp_out: Tensor,
        var cos: Tensor, var sin: Tensor,
        var blk_vec_cond: Optional[Tensor] = Optional[Tensor](None),
    ):
        self.combined = combined^
        self.blk_vec = blk_vec^
        self.tmlp_out = tmlp_out^
        self.cos = cos^
        self.sin = sin^
        self.blk_vec_cond = blk_vec_cond^


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONING — extended in C8 with an OPTIONAL OminiControl condition segment.
#
# CONDL == 0 (the default, every pre-C8 caller): the statements below are EXACTLY
# the pre-C8 ones — every EDIT statement sits behind `comptime if CONDL > 0`, the
# same shape-of-change C2..C6 used in the trainer. CONDL > 0 turns the sequence
# into [TXT_real(lt) | IMG | COND | TXT_pad], adds the second (t = 0) modulation
# vector, and reorders the SOURCE-order pos table [TXT(LT) | IMG | COND] with the
# READER's own krea2_reorder_combined_edit — the very function the trainer calls,
# so there is no second copy of the gather math (layout seam A.3).
#
# WHY THIS IS THE TRAINER'S SEQUENCE, NOT A LOOKALIKE: cond_e is the shared
# frozen krea2_first(cond_tokens) plus the condition-only `first` LoRA, while
# image rows receive only the frozen projection. The second temb chain is the SAME
# temb->tmlp->tproj on t = 0, and the concat order is the Krea2OminiLayout one.
# Those are the three things train_krea2._build_conditioning_edit does.
#
# ⚠ ONE KNOWN TRAIN/INFER DIVERGENCE, PRE-EXISTING AND NOT INTRODUCED HERE: the
# TRAINER passes Optional[Tensor](None) for the txtfusion refiner mask (it never
# builds one, edit or not) while this inference builder masks the text-pad
# columns by default — the fix that stopped long prompts from blowing up CFG.
# `use_refiner_mask=False` reproduces the trainer's text conditioning exactly, so
# the two can be A/B'd; it is NOT the default because every krea2 render in this
# repo has been made with the mask on.
def build_conditioning[LT: Int, LFULL: Int, CONDL: Int = 0](
    cond_w: Krea2ResidentCond,
    img: Tensor,            # [1, IMGLEN, 64] BF16 patchified noised latent
    context: Tensor,        # [1, LT, 12, 2560] BF16 (LT == LTMAX)
    pos: Tensor,            # [1, LFULL, 3] F32 (CONDL>0: SOURCE order TXT|IMG|COND)
    t: Tensor,              # [1] F32 timestep
    real_text_len: Int,     # natural caption length lt (<= LT); real_len = lt+IMGLEN
    ctx: DeviceContext,
    cond_img: Optional[Tensor] = Optional[Tensor](None),
        # [1, CONDL, 64] BF16 patchified CLEAN condition latent. REQUIRED when
        # CONDL > 0, ignored (and refused) when CONDL == 0.
    use_refiner_mask: Bool = True,
    first_lora: Optional[LoraAdapterDevice] = Optional[LoraAdapterDevice](None),
) raises -> _Cond:
    comptime IMGL = LFULL - LT - CONDL
    comptime assert IMGL > 0, "build_conditioning: LFULL must exceed LT + CONDL"
    comptime if CONDL > 0:
        if not cond_img:
            raise Error(
                "build_conditioning: a CONDL > 0 build needs cond_img"
                " [1, CONDL, 64] (the patchified CLEAN condition latent)"
            )
    else:
        if cond_img:
            raise Error(
                "build_conditioning: cond_img supplied to a CONDL == 0 build"
                " — rebuild with the OminiControl condition length"
            )
    var img_bf = cast_tensor(img, STDtype.BF16, ctx)
    var img_e = krea2_first(img_bf, cond_w.first_w[], cond_w.first_b[], ctx)

    var te = krea2_temb(t, TDIM, ctx, STDtype.BF16)
    var t_vec = krea2_tmlp(
        te, cond_w.tmlp0_w[], cond_w.tmlp0_b[], cond_w.tmlp2_w[], cond_w.tmlp2_b[], ctx,
    )
    var t3 = reshape(t_vec, [1, 1, FEATURES], ctx)

    var blk_vec = krea2_tproj(t3, cond_w.tproj1_w[], cond_w.tproj1_b[], ctx)
    var blk_vec2 = reshape(blk_vec, [1, 6 * FEATURES], ctx)

    # REFINER MASK (fix): the txtfusion refiner attends over the LTMAX text slots;
    # our fixed-LTMAX pad fills [real_text_len, LT) with ZEROS. Without a mask those
    # zero-pad columns contaminate the text conditioning ∝ pad count, so cond (long,
    # little pad) and uncond (short, lots of pad) diverge and CFG blows up → black.
    # Mask the pad columns to match ai-toolkit / krea-ai/krea-2 (mmdit.py:288).
    # NOTE: the masked sdpa (ops/attention.sdpa) requires mask.dtype == q.dtype;
    # the refiner runs BF16 at inference, so cast the F32 additive mask to BF16.
    var refiner_mask: Optional[Tensor]
    if real_text_len < LT and use_refiner_mask:
        var rm_f32 = krea2_build_refiner_mask(real_text_len, LT, TXTHEADS, ctx)
        refiner_mask = Optional[Tensor](cast_tensor(rm_f32, STDtype.BF16, ctx))
    else:
        refiner_mask = Optional[Tensor](None)
    var ctx_fused = krea2_text_fusion[LT, NLAYERS_TXT, TXTHEADS, TXTHD](
        context, cond_w.lw0, cond_w.lw1,
        cond_w.projector_w[],
        cond_w.rf0, cond_w.rf1, refiner_mask^, ctx,
    )

    var ctx_proj = krea2_txtmlp(
        ctx_fused,
        cond_w.txtmlp0_scale[],
        cond_w.txtmlp1_w[], cond_w.txtmlp1_b[],
        cond_w.txtmlp3_w[], cond_w.txtmlp3_b[],
        ctx,
    )

    # length-bucket reorder → [TXT_real(0:lt) | IMG | TXT_pad(tail)], or, with an
    # OminiControl condition segment, [TXT_real | IMG | COND | TXT_pad].
    var combined: Tensor
    var blk_vec_cond = Optional[Tensor](None)
    comptime if CONDL > 0:
        # COND rows: shared frozen `first` plus the Omini condition-only LoRA.
        # Target image rows above deliberately do not receive this delta.
        var cond_bf = cast_tensor(cond_img.value(), STDtype.BF16, ctx)
        var cond_e = krea2_first(cond_bf, cond_w.first_w[], cond_w.first_b[], ctx)
        if first_lora:
            var cond_delta = klein_lora_fwd_device_resident_unfused(
                cond_bf, first_lora.value(), CONDL, ctx
            )
            cond_e = add(cond_e, cond_delta, ctx)
        # chain #2: temb->tmlp->tproj on t = 0, the condition-row modulation.
        var t_zero = _t_scalar(Float32(0.0), ctx)
        var te_c = krea2_temb(t_zero, TDIM, ctx, STDtype.BF16)
        var t_vec_c = krea2_tmlp(
            te_c, cond_w.tmlp0_w[], cond_w.tmlp0_b[],
            cond_w.tmlp2_w[], cond_w.tmlp2_b[], ctx,
        )
        var t3_c = reshape(t_vec_c, [1, 1, FEATURES], ctx)
        var bvc = krea2_tproj(t3_c, cond_w.tproj1_w[], cond_w.tproj1_b[], ctx)
        blk_vec_cond = Optional[Tensor](reshape(bvc, [1, 6 * FEATURES], ctx))
        # BOUNDARY: a zero-length slice launches grid_dim 0 and aborts, so
        # neither empty text segment is materialized (same guards the reader's
        # krea2_reorder_combined_edit carries).
        if real_text_len <= 0:
            var h0 = concat(1, ctx, img_e, cond_e)
            combined = concat(1, ctx, h0, ctx_proj)
        elif real_text_len < LT:
            var real_text = slice(ctx_proj, 1, 0, real_text_len, ctx)
            var pad_text = slice(
                ctx_proj, 1, real_text_len, LT - real_text_len, ctx
            )
            var head = concat(1, ctx, real_text, img_e)
            var head2 = concat(1, ctx, head, cond_e)
            combined = concat(1, ctx, head2, pad_text)
        else:
            var head3 = concat(1, ctx, ctx_proj, img_e)
            combined = concat(1, ctx, head3, cond_e)
    else:
        if real_text_len < LT:
            var real_text = slice(ctx_proj, 1, 0, real_text_len, ctx)
            var pad_text = slice(ctx_proj, 1, real_text_len, LT - real_text_len, ctx)
            var head = concat(1, ctx, real_text, img_e)
            combined = concat(1, ctx, head, pad_text)
        else:
            combined = concat(1, ctx, ctx_proj, img_e)

    # rope: reorder pos to match combined (text positions are all-zero → no rotation change).
    var pos_re: Tensor
    comptime if CONDL > 0:
        # THE READER'S OWN GATHER — not a copy of it. krea2_reorder_combined_edit
        # is the function train_krea2 calls and the C6 gate compares element-for-
        # element against krea2_omini_pos_combined.
        pos_re = krea2_reorder_combined_edit[LT, LFULL, CONDL](
            pos, real_text_len, ctx
        )
    else:
        if real_text_len < LT:
            var pos_real = slice(pos, 1, 0, real_text_len, ctx)
            var pos_img = slice(pos, 1, LT, LFULL - LT, ctx)
            var pos_pad = slice(pos, 1, real_text_len, LT - real_text_len, ctx)
            var pos_head = concat(1, ctx, pos_real, pos_img)
            pos_re = concat(1, ctx, pos_head, pos_pad)
        else:
            pos_re = pos.clone(ctx)
    var pos_flat = reshape(pos_re, [LFULL * 3], ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, THETA, ctx, STDtype.F32)
    var rcos = rope[0].clone(ctx)
    var rsin = rope[1].clone(ctx)

    return _Cond(
        TArc(combined^), blk_vec2^, t3^, rcos^, rsin^, blk_vec_cond^
    )


# ══════════════════════════════════════════════════════════════════════════════
# INLINE COND — context tensor (from an encode-child .bin) → padded LTMAX context
# + pos + natural text length.
# ══════════════════════════════════════════════════════════════════════════════
struct Krea2InlineCond(Copyable, Movable):
    var context: TArc      # [1, LTMAX, 12, 2560] BF16
    var pos: TArc          # [1, LTMAX+imglen, 3] F32
    var text_len: Int      # natural LT before padding

    def __init__(out self, var context: TArc, var pos: TArc, text_len: Int):
        self.context = context^
        self.pos = pos^
        self.text_len = text_len


def _pad_context_to_ltmax[LTMAX: Int](
    context: Tensor, lt: Int, ctx: DeviceContext,
) raises -> Tensor:
    var sh = context.shape()
    if (
        len(sh) != 4 or sh[0] != 1
        or sh[2] != KREA2_TXT_LAYERS or sh[3] != KREA2_TXT_DIM
    ):
        raise Error("krea2_infer: expected context [1, LT, 12, 2560]")
    if lt > LTMAX:
        raise Error(
            String("krea2_infer: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise the compiled Krea2 LTMAX bucket)"
        )
    var ctx_bf = cast_tensor(context, STDtype.BF16, ctx)
    if lt == LTMAX:
        return ctx_bf^
    var pad = zeros_device(
        [1, LTMAX - lt, KREA2_TXT_LAYERS, KREA2_TXT_DIM], STDtype.BF16, ctx
    )
    return concat(1, ctx, ctx_bf, pad)


def inline_cond_from_context[LH: Int, LW: Int, LTMAX: Int](
    var context: Tensor, ctx: DeviceContext,
) raises -> Krea2InlineCond:
    var sh = context.shape()
    if (
        len(sh) != 4 or sh[0] != 1
        or sh[2] != KREA2_TXT_LAYERS or sh[3] != KREA2_TXT_DIM
    ):
        raise Error("krea2_infer: expected context [1, LT, 12, 2560]")
    var lt = sh[1]
    var padded = _pad_context_to_ltmax[LTMAX](context^, lt, ctx)
    var pos = krea2_build_pos[LH, LW](LTMAX, ctx)
    return Krea2InlineCond(TArc(padded^), TArc(pos^), lt)


def inline_cond_from_context_edit[LH: Int, LW: Int, LTMAX: Int, CONDL: Int](
    var context: Tensor,
    dh: Int, dw: Int, pos_scale: Float32,
    ctx: DeviceContext,
) raises -> Krea2InlineCond:
    """The EDIT twin of inline_cond_from_context: same LTMAX text padding, but
    `pos` is the SOURCE-order table [TXT zeros(LTMAX) | IMG grid | COND grid]
    from krea2_build_pos_cond — the reader's own builder, which the C5 gate
    compares element-for-element against the layout module's host twin. For the
    EDIT condition type (dh=dw=0, pos_scale=1.0) the COND grid EQUALS the IMG
    grid, i.e. the condition overlaps the target canvas."""
    var sh = context.shape()
    if (
        len(sh) != 4 or sh[0] != 1
        or sh[2] != KREA2_TXT_LAYERS or sh[3] != KREA2_TXT_DIM
    ):
        raise Error("krea2_infer: expected context [1, LT, 12, 2560]")
    var lt = sh[1]
    var padded = _pad_context_to_ltmax[LTMAX](context^, lt, ctx)
    var pos = krea2_build_pos_cond[LH, LW](LTMAX, CONDL, dh, dw, pos_scale, ctx)
    return Krea2InlineCond(TArc(padded^), TArc(pos^), lt)


def inline_cond_from_context_edit_grid[
    LH: Int, LW: Int, COND_LH: Int, COND_LW: Int, LTMAX: Int, CONDL: Int
](
    var context: Tensor,
    dh: Int, dw: Int, pos_scale: Float32,
    ctx: DeviceContext,
) raises -> Krea2InlineCond:
    """EDIT conditioning with an independently sized condition-token grid.
    A compact condition grid plus position scale can span a larger target grid
    without changing the adapter's condition-row count."""
    var sh = context.shape()
    if (
        len(sh) != 4 or sh[0] != 1
        or sh[2] != KREA2_TXT_LAYERS or sh[3] != KREA2_TXT_DIM
    ):
        raise Error("krea2_infer: expected context [1, LT, 12, 2560]")
    var lt = sh[1]
    var padded = _pad_context_to_ltmax[LTMAX](context^, lt, ctx)
    var pos = krea2_build_pos_cond_grid[LH, LW, COND_LH, COND_LW](
        LTMAX, CONDL, dh, dw, pos_scale, ctx
    )
    return Krea2InlineCond(TArc(padded^), TArc(pos^), lt)


def inline_cond_from_bin[LH: Int, LW: Int, LTMAX: Int](
    path: String, ctx: DeviceContext,
) raises -> Krea2InlineCond:
    if path == String(""):
        raise Error("krea2_infer: empty context bin path")
    var context = load_tensor_bin(path, ctx)
    return inline_cond_from_context[LH, LW, LTMAX](context^, ctx)


def _unpatch[LH: Int, LW: Int](tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Inverse patchify: [1,imglen,64] -> [1,16,LH,LW]."""
    comptime gh = LH // 2
    comptime gw = LW // 2
    var x6 = reshape(tokens, [1, gh, gw, 16, 2, 2], ctx)
    var xp = permute(x6, [0, 3, 1, 4, 2, 5], ctx)
    return reshape(xp, [1, 16, gh * 2, gw * 2], ctx)


def _t_scalar(v: Float32, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    h.append(v)
    return Tensor.from_host(h^, [1], STDtype.F32, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# LoRA OVERLAY loader — a saved PEFT/ai-toolkit .safetensors (diffusion_model.*
# .lora_A/.lora_B, 224 adapters) → Krea2StackLora (ADDED overlay, never fused).
# Composes the generic reader (moments zeroed) + the host→device grouping.
# ══════════════════════════════════════════════════════════════════════════════
def _krea2_lora_prefix(bi: Int, slot: Int) raises -> String:
    var b = String("diffusion_model.blocks.") + String(bi)
    if slot == 0:
        return b + ".attn.wq"
    elif slot == 1:
        return b + ".attn.wk"
    elif slot == 2:
        return b + ".attn.wv"
    elif slot == 3:
        return b + ".attn.gate"
    elif slot == 4:
        return b + ".attn.wo"
    elif slot == 5:
        return b + ".mlp.gate"
    elif slot == 6:
        return b + ".mlp.up"
    elif slot == 7:
        return b + ".mlp.down"
    raise Error(String("_krea2_lora_prefix: bad slot ") + String(slot))


def empty_krea2_stack_lora() raises -> Krea2StackLora:
    """Identity overlay: every slot None → pure frozen base (base-only render)."""
    var blocks = List[Krea2BlockLora]()
    for _bi in range(NBLOCKS):
        blocks.append(Krea2BlockLora(
            Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        ))
    return Krea2StackLora(blocks^)


def load_krea2_stack_lora(
    path: String, scale: Float32, ctx: DeviceContext
) raises -> Krea2StackLora:
    """Load a trained krea2 LoRA (PEFT diffusion_model.* layout) as an inference
    overlay. `scale` folds (alpha/rank) * strength_model (moments are ignored)."""
    var prefixes = List[String]()
    for bi in range(NBLOCKS):
        for s in range(KREA2_SLOTS_PER_BLOCK):
            prefixes.append(_krea2_lora_prefix(bi, s))
    var loaded = load_lora_for_resume(prefixes, scale, path, ctx)
    if len(loaded) != NBLOCKS * KREA2_SLOTS_PER_BLOCK:
        raise Error(
            String("krea2_infer: LoRA adapter count ") + String(len(loaded))
            + " != " + String(NBLOCKS * KREA2_SLOTS_PER_BLOCK) + " (bad file/layout)"
        )
    var blocks = List[Krea2BlockLora]()
    for bi in range(NBLOCKS):
        var base = bi * KREA2_SLOTS_PER_BLOCK
        blocks.append(Krea2BlockLora(
            Optional[LoraAdapterDevice](lora_adapter_to_device(loaded[base + 0].adapter, ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(loaded[base + 1].adapter, ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(loaded[base + 2].adapter, ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(loaded[base + 3].adapter, ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(loaded[base + 4].adapter, ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(loaded[base + 5].adapter, ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(loaded[base + 6].adapter, ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(loaded[base + 7].adapter, ctx)),
        ))
    print("[krea2_infer] LoRA overlay loaded:", NBLOCKS * KREA2_SLOTS_PER_BLOCK,
          "adapters  scale=", scale, "  ", path)
    return Krea2StackLora(blocks^)


def load_krea2_omini_first_lora(
    path: String, scale: Float32, ctx: DeviceContext
) raises -> LoraAdapterDevice:
    """Load the mandatory condition-only Krea `first`/Omini x_embedder LoRA."""
    var prefixes = List[String]()
    prefixes.append(String("diffusion_model.first"))
    var loaded = load_lora_for_resume(prefixes, scale, path, ctx)
    if len(loaded) != 1:
        raise Error("krea2 edit: corrected LoRA is missing diffusion_model.first")
    if loaded[0].adapter.in_f != 64 or loaded[0].adapter.out_f != FEATURES:
        raise Error("krea2 edit: diffusion_model.first LoRA shape must be 64->6144")
    return lora_adapter_to_device(loaded[0].adapter, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE — the full resident-base + LoRA + CFG + euler denoise loop → latent →
# VAE decode → PNG. One blocking call (the worker's step()); prints per-step
# progress. Hoisted verbatim from train_krea2._krea2_sample_resident_latent +
# _krea2_decode_latent_to_png.
# ══════════════════════════════════════════════════════════════════════════════
struct Krea2ComfyVelocityPair(Movable):
    """Normal and BIG-guidance flow velocities from one cond/uncond pair.

    LanPaint's Comfy sampler uses textbook CFG (`uncond + cfg*(cond-uncond)`),
    which intentionally differs from Krea's creator T2I guidance helper.
    """

    var normal: Tensor
    var big: Tensor

    def __init__(out self, var normal: Tensor, var big: Tensor):
        self.normal = normal^
        self.big = big^


def _krea2_comfy_cfg(
    v_cond: Tensor, v_uncond: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    var diff = sub(v_cond, v_uncond, ctx)
    return add(v_uncond, mul_scalar(diff, scale, ctx), ctx)


def krea2_predict_comfy_velocity_pair[
    LH: Int, LW: Int, LTMAX: Int, LFULL: Int
](
    st: ShardedSafeTensors,
    key_prefix: String,
    cond_w: Krea2ResidentCond,
    fin: Krea2StreamFinal,
    lora: Krea2StackLora,
    cond: Krea2InlineCond,
    uncond: Krea2InlineCond,
    latent: Tensor,
    timestep: Float32,
    cfg_scale: Float32,
    cfg_big: Float32,
    resident: Optional[Krea2ResidentFp8],
    resident_i8: Optional[Krea2ResidentInt8],
    host_i8: Optional[Krea2HostInt8Inf],
    ctx: DeviceContext,
) raises -> Krea2ComfyVelocityPair:
    """One Krea2 flow-model evaluation for the LanPaint inner loop.

    The unconditional stack is skipped when both Comfy guidance scales are 1.
    That is the verified Krea2-Turbo/Image-First profile, so each LanPaint
    iteration costs one DiT forward instead of two.
    """
    comptime imglen = (LH // 2) * (LW // 2)
    comptime assert LFULL == LTMAX + imglen, "Krea2 LanPaint LFULL mismatch"

    var img_tokens_f32 = krea2_patchify[LH, LW](latent, ctx)
    var img_tokens = torch_f32_to_bf16_rne(img_tokens_f32, ctx)
    var t_t = _t_scalar(timestep, ctx)
    var c = build_conditioning[LTMAX, LFULL](
        cond_w, img_tokens, cond.context[], cond.pos[], t_t, cond.text_len, ctx,
    )
    var real_len_c = Optional[Int](cond.text_len + imglen)
    var pred_c = krea2_stack_lora_forward_streamed[
        LFULL, HEADS, KVHEADS, HEADDIM
    ](
        c.combined, c.blk_vec, c.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin,
        c.cos, c.sin, EPS, cond.text_len, imglen, ctx, real_len_c, resident,
        resident_i8=resident_i8, host_i8_inf=host_i8,
    )
    var v_c = pred_c.velocity[].clone(ctx)

    if cfg_scale == Float32(1.0) and cfg_big == Float32(1.0):
        var v_c_f32 = cast_tensor(v_c, STDtype.F32, ctx)
        var v_c_latent = _unpatch[LH, LW](v_c_f32, ctx)
        return Krea2ComfyVelocityPair(v_c_latent.clone(ctx), v_c_latent^)

    _ = pred_c^
    ctx.synchronize()
    var u = build_conditioning[LTMAX, LFULL](
        cond_w, img_tokens, uncond.context[], uncond.pos[], t_t,
        uncond.text_len, ctx,
    )
    var real_len_u = Optional[Int](uncond.text_len + imglen)
    var pred_u = krea2_stack_lora_forward_streamed[
        LFULL, HEADS, KVHEADS, HEADDIM
    ](
        u.combined, u.blk_vec, u.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin,
        u.cos, u.sin, EPS, uncond.text_len, imglen, ctx, real_len_u, resident,
        resident_i8=resident_i8, host_i8_inf=host_i8,
    )
    var v_u = pred_u.velocity[].clone(ctx)
    var v_normal_bf16 = _krea2_comfy_cfg(v_c, v_u, cfg_scale, ctx)
    var v_big_bf16 = _krea2_comfy_cfg(v_c, v_u, cfg_big, ctx)
    var v_normal_f32 = cast_tensor(v_normal_bf16, STDtype.F32, ctx)
    var v_big_f32 = cast_tensor(v_big_bf16, STDtype.F32, ctx)
    var v_normal = _unpatch[LH, LW](v_normal_f32, ctx)
    var v_big = _unpatch[LH, LW](v_big_f32, ctx)
    return Krea2ComfyVelocityPair(v_normal^, v_big^)


# ── ONE OminiControl EDIT forward (C8) ────────────────────────────────────────
# Everything numeric here is SHARED with the trainer: build_conditioning (the
# CONDL > 0 arm above), Krea2OminiLayout for the offsets, krea2_omini_mod_split
# for the modulation boundary, krea2_edit_real_len for the flash prefix, and
# krea2_stack_lora_forward_streamed with the SAME three EDIT switches the C6
# trainer step passes. The ONLY thing this adds is the optional attn_bias.
#
# The velocity returned is final[:, lt : lt+imglen, :] — the IMAGE rows. The COND
# rows and the TXT_pad tail are sliced away inside the stack (txtlen=lt,
# imglen=imglen), which is exactly why the C6 loss-mask gate found the cond rows
# bit-invisible to the loss. THE COND ROWS ARE NEVER DENOISED: the euler update
# below only ever touches `latent`, which is unpatchified from this image-row
# slice, and the condition tokens are rebuilt from the CLEAN source every step.
def _krea2_edit_velocity[
    LH: Int, LW: Int, LTMAX: Int, LFULL: Int, CONDL: Int
](
    st: ShardedSafeTensors,
    key_prefix: String,
    cond_w: Krea2ResidentCond,
    fin: Krea2StreamFinal,
    lora: Krea2StackLora,
    first_lora: Optional[LoraAdapterDevice],
    text: Krea2InlineCond,      # text conditioning (positive OR negative)
    img_tokens: Tensor,         # [1, imglen, 64] BF16 patchified NOISED latent
    cond_tokens: Tensor,        # [1, CONDL, 64] BF16 patchified CLEAN condition
    t_t: Tensor,                # [1] F32 timestep
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8Inf] = Optional[Krea2HostInt8Inf](None),
    attn_bias: Optional[TArc] = Optional[TArc](None),
    use_refiner_mask: Bool = True,
) raises -> Tensor:
    comptime imglen = (LH // 2) * (LW // 2)
    comptime assert LFULL == LTMAX + imglen + CONDL, "edit LFULL mismatch"
    comptime assert CONDL > 0, "_krea2_edit_velocity needs a condition segment"
    var lt = text.text_len
    var lay = Krea2OminiLayout(LTMAX, imglen, CONDL, lt)
    lay.check_flash_prefix()
    var c = build_conditioning[LTMAX, LFULL, CONDL](
        cond_w, img_tokens, text.context[], text.pos[], t_t, lt, ctx,
        cond_img=Optional[Tensor](cond_tokens.clone(ctx)),
        use_refiner_mask=use_refiner_mask,
        first_lora=first_lora.copy(),
    )
    var real_len = Optional[Int](krea2_edit_real_len(lt, imglen, CONDL))
    var pred = krea2_stack_lora_forward_streamed[
        LFULL, HEADS, KVHEADS, HEADDIM
    ](
        c.combined, c.blk_vec, c.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin,
        c.cos, c.sin, EPS, lt, imglen, ctx, real_len, resident,
        resident_sq=resident_sq, resident_i8=resident_i8, host_i8_inf=host_i8,
        vec_cond=Optional[TArc](TArc(c.blk_vec_cond.value().clone(ctx))),
        cond_off=Optional[Int](krea2_omini_mod_split(lay)),
        cond_len=Optional[Int](CONDL),
        attn_bias=attn_bias.copy(),
    )
    return pred.velocity[].clone(ctx)


def krea2_sample_latent[
    LH: Int, LW: Int, LTMAX: Int, LFULL: Int, CONDL: Int = 0
](
    st: ShardedSafeTensors,
    key_prefix: String,
    cond_w: Krea2ResidentCond,
    fin: Krea2StreamFinal,
    lora: Krea2StackLora,
    cond: Krea2InlineCond,
    uncond: Krea2InlineCond,
    sample_steps: Int,
    cfg_scale: Float32,
    seed: UInt64,
    resident: Optional[Krea2ResidentFp8],
    ctx: DeviceContext,
    use_fixed_mu_1_15: Bool = False,
    progress_fd: Int32 = Int32(-1),
    # ── OminiControl EDIT (C8). ALL of these are inert on a CONDL == 0 build:
    # every statement that reads them is behind `comptime if CONDL > 0`, so the
    # text-to-image sampler every existing caller compiles is unchanged. ──────
    cond_tokens: Optional[Tensor] = Optional[Tensor](None),
        # [1, CONDL, 64] BF16 patchified CLEAN condition latent. REQUIRED when
        # CONDL > 0.
    first_lora: Optional[LoraAdapterDevice] = Optional[LoraAdapterDevice](None),
        # Corrected Omini `first`/x_embedder adapter. None is valid only for an
        # explicit frozen-base (--no-lora) render.
    cond_tokens_black: Optional[Tensor] = Optional[Tensor](None),
        # [1, CONDL, 64] BF16 patchified latent of a BLACK image — the
        # unconditional branch of OminiControl's image-CFG (Condition.encode(
        # empty=True), flux_omini.py:125-126, 657-658). REQUIRED when
        # image_guidance_scale != 1.0.
    image_guidance_scale: Float32 = Float32(1.0),
        # flux_omini.py:781  pred = unc + s*(cond - unc). 1.0 = OFF (one forward).
    condition_scale: Float32 = Float32(1.0),
        # flux_omini.py:286-292  additive log(scale) attention bias between COND
        # and non-COND tokens. 1.0 = THE IDENTITY (no bias tensor is built and the
        # attention takes the untouched flash-padmask path).
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8Inf] = Optional[Krea2HostInt8Inf](None),
    use_refiner_mask: Bool = True,
) raises -> Tensor:
    comptime imglen = (LH // 2) * (LW // 2)
    comptime assert LFULL == LTMAX + imglen + CONDL, "Krea2 sample LFULL mismatch"
    if sample_steps < 1:
        raise Error("krea2_infer: sample steps must be >= 1")
    comptime if CONDL > 0:
        if not cond_tokens:
            raise Error(
                "krea2_sample_latent: a CONDL > 0 (OminiControl EDIT) build needs"
                " cond_tokens [1, CONDL, 64]"
            )
        if image_guidance_scale != Float32(1.0) and not cond_tokens_black:
            raise Error(
                "krea2_sample_latent: image_guidance_scale != 1.0 needs"
                " cond_tokens_black (the BLACK-image condition,"
                " flux_omini.py Condition.encode(empty=True))"
            )
    else:
        if Bool(cond_tokens) or Bool(cond_tokens_black):
            raise Error(
                "krea2_sample_latent: condition tokens supplied to a CONDL == 0"
                " build — the sequence has no COND rows to put them in"
            )
        if condition_scale != Float32(1.0) or image_guidance_scale != Float32(1.0):
            raise Error(
                "krea2_sample_latent: condition_scale / image_guidance_scale are"
                " OminiControl EDIT knobs and do nothing on a CONDL == 0 build"
            )

    # DIAGNOSTIC: inject a fixed noise latent (e.g. a torch-reference dump) instead of
    # generating from `seed`. Isolates whether the noise stream drives composition.
    var _inject = env_or("KREA2_INJECT_NOISE", String(""))
    var latent: Tensor
    if _inject.byte_length() > 0:
        latent = load_tensor_bin(_inject, ctx)
        print("[krea2-infer] INJECTED noise latent from ", _inject)
    else:
        # torch-parity Philox randn (bit-compatible with torch.randn on this GPU) so
        # a given seed reproduces the reference/ai-toolkit composition. [[project_krea2_noise_letterbox]]
        latent = randn_torch([1, 16, LH, LW], UInt64(seed), ctx)
    var seq = krea2_packed_seq_len(LH * 8, LW * 8)
    # Creator contract: Raw derives mu from resolution; the distilled Turbo
    # checkpoint is sampled with a fixed mu=1.15 at every admitted resolution.
    var ts = krea2_timesteps(
        seq, sample_steps,
        mu_override=Float32(1.15),
        use_mu_override=use_fixed_mu_1_15,
    )
    print("[krea2-infer] steps=", sample_steps, " cfg=", cfg_scale, " seed=", seed,
          " LT(cond)=", cond.text_len, " LT(uncond)=", uncond.text_len)

    # ── OminiControl condition_scale bias, built ONCE (lt is fixed for the run).
    # 1.0 is THE IDENTITY: log(1) == 0, so no tensor is built and every block
    # keeps the flash-padmask arm it was trained with.
    var attn_bias_c = Optional[TArc](None)
    var attn_bias_u = Optional[TArc](None)
    comptime if CONDL > 0:
        if condition_scale != Float32(1.0):
            print(
                "[krea2-omini] condition_scale=", condition_scale,
                " -> additive attention bias log(scale)=",
                krea2_edit_cond_bias(condition_scale),
                " on COND<->non-COND logits (flux_omini.py:280-341).",
            )
            print(
                "[krea2-omini]   THIS ARM IS UNVERIFIED ON DEVICE (authored in",
                "C8 with the GPU busy). It also swaps the cuDNN flash-padmask",
                "for the masked math SDPA, which is NOT the attention the LoRA",
                "was trained under. Use condition_scale=1.0 for the honest",
                "render; treat anything else as an experiment.",
            )
            attn_bias_c = Optional[TArc](TArc(krea2_build_edit_attn_bias(
                cond.text_len, LTMAX, imglen, CONDL, condition_scale,
                KREA2_HEADS, STDtype.BF16, ctx,
            )))
            if cfg_scale > Float32(0.0):
                attn_bias_u = Optional[TArc](TArc(krea2_build_edit_attn_bias(
                    uncond.text_len, LTMAX, imglen, CONDL, condition_scale,
                    KREA2_HEADS, STDtype.BF16, ctx,
                )))

    # Activate lazy CUDA/cuDNN kernels without advancing the latent or schedule.
    # The first real forward otherwise loads NVVM/GPU compiler DSOs and CUDA
    # ComputeCache entries after the hardware I/O fence.
    var warm_img_f32 = krea2_patchify[LH, LW](latent, ctx)
    var warm_img = torch_f32_to_bf16_rne(warm_img_f32, ctx)
    var warm_t = _t_scalar(ts[0], ctx)
    comptime if CONDL > 0:
        var warm_v = _krea2_edit_velocity[LH, LW, LTMAX, LFULL, CONDL](
            st, key_prefix, cond_w, fin, lora, first_lora.copy(), cond,
            warm_img, cond_tokens.value(), warm_t, ctx,
            resident, resident_sq, resident_i8, host_i8,
            attn_bias_c.copy(), use_refiner_mask,
        )
        _ = warm_v^
    else:
        var warm_c = build_conditioning[LTMAX, LFULL](
            cond_w, warm_img, cond.context[], cond.pos[], warm_t,
            cond.text_len, ctx,
        )
        var warm_real_len = Optional[Int](cond.text_len + imglen)
        var warm_pred = krea2_stack_lora_forward_streamed[
            LFULL, HEADS, KVHEADS, HEADDIM
        ](
            warm_c.combined, warm_c.blk_vec, warm_c.tmlp_out,
            st, key_prefix, NBLOCKS, lora, fin,
            warm_c.cos, warm_c.sin, EPS, cond.text_len, imglen, ctx,
            warm_real_len, resident,
            resident_i8=resident_i8, host_i8_inf=host_i8,
        )
        _ = warm_pred^
    ctx.synchronize()
    print("[krea2-infer] pre-step-0 denoiser kernel warm-up complete")
    print("[krea2-infer] prefaulted worker image bytes=", prefault_self_executable())
    print(
        "[krea2-infer] prefaulted mapped shared-library bytes=",
        prefault_mapped_shared_libraries(),
    )

    # This is the hard boundary for the no-disk-during-denoise gate.  Model
    # activation, sidecar validation, and host/GPU population are complete;
    # no checkpoint-backed path may be touched after this event.
    if progress_fd >= 0:
        try:
            write_msg(
                progress_fd,
                String("{\"ev\":\"progress\",\"step\":0,\"total\":")
                + String(sample_steps)
                + String(",\"phase\":\"sampling\",\"preview\":\"\"}"),
            )
        except e:
            print("[krea2-infer] progress IPC skipped:", String(e))

    for si in range(sample_steps):
        var t_cur = ts[si]
        var t_prev = ts[si + 1]
        var img_tokens_f32 = krea2_patchify[LH, LW](latent, ctx)
        var img_tokens = torch_f32_to_bf16_rne(img_tokens_f32, ctx)
        var t_t = _t_scalar(t_cur, ctx)

        var v_bf16: Tensor
        comptime if CONDL > 0:
            # ── EDIT step. TWO guidance axes, evaluated in this order:
            #   1. IMAGE-CFG (OminiControl's own): a second forward whose ONLY
            #      difference is a BLACK-image condition, then
            #      unc + s*(cond - unc)  (flux_omini.py:758-781).
            #   2. TEXT-CFG (krea2's; OminiControl has none — FLUX-dev is
            #      distilled): the negative-prompt forward keeps the REAL
            #      condition, and krea2_cfg composes it with the result of (1).
            # Composing the two is OUR choice, not something the reference does;
            # with cfg_scale <= 0 (the krea2 Turbo profile) only axis 1 runs.
            var v_img = _krea2_edit_velocity[LH, LW, LTMAX, LFULL, CONDL](
                st, key_prefix, cond_w, fin, lora, first_lora.copy(), cond,
                img_tokens, cond_tokens.value(), t_t, ctx,
                resident, resident_sq, resident_i8, host_i8,
                attn_bias_c.copy(), use_refiner_mask,
            )
            if image_guidance_scale != Float32(1.0):
                ctx.synchronize()
                var v_black = _krea2_edit_velocity[LH, LW, LTMAX, LFULL, CONDL](
                    st, key_prefix, cond_w, fin, lora, first_lora.copy(), cond,
                    img_tokens, cond_tokens_black.value(), t_t, ctx,
                    resident, resident_sq, resident_i8, host_i8,
                    attn_bias_c.copy(), use_refiner_mask,
                )
                v_img = _krea2_comfy_cfg(
                    v_img, v_black, image_guidance_scale, ctx
                )
            if cfg_scale <= Float32(0.0):
                v_bf16 = v_img^
            else:
                ctx.synchronize()
                var v_txt_u = _krea2_edit_velocity[LH, LW, LTMAX, LFULL, CONDL](
                    st, key_prefix, cond_w, fin, lora, first_lora.copy(), uncond,
                    img_tokens, cond_tokens.value(), t_t, ctx,
                    resident, resident_sq, resident_i8, host_i8,
                    attn_bias_u.copy(), use_refiner_mask,
                )
                v_bf16 = krea2_cfg(v_img, v_txt_u, cfg_scale, ctx)
        else:
            var c = build_conditioning[LTMAX, LFULL](
                cond_w, img_tokens, cond.context[], cond.pos[], t_t, cond.text_len, ctx,
            )
            var real_len_c = Optional[Int](cond.text_len + imglen)
            var pred_c = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
                c.combined, c.blk_vec, c.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin,
                c.cos, c.sin, EPS, cond.text_len, imglen, ctx, real_len_c, resident,
                resident_i8=resident_i8, host_i8_inf=host_i8,
            )

            # Creator sampling.py enables CFG only when guidance > 0. Turbo's
            # recommended guidance=0 therefore executes one conditional forward.
            if cfg_scale <= Float32(0.0):
                v_bf16 = pred_c.velocity[].clone(ctx)
            else:
                var v_c = pred_c.velocity[].clone(ctx)
                _ = pred_c^
                ctx.synchronize()
                var u = build_conditioning[LTMAX, LFULL](
                    cond_w, img_tokens, uncond.context[], uncond.pos[], t_t,
                    uncond.text_len, ctx,
                )
                var real_len_u = Optional[Int](uncond.text_len + imglen)
                var pred_u = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
                    u.combined, u.blk_vec, u.tmlp_out,
                    st, key_prefix, NBLOCKS, lora, fin,
                    u.cos, u.sin, EPS, uncond.text_len, imglen, ctx, real_len_u, resident,
                    resident_i8=resident_i8, host_i8_inf=host_i8,
                )
                v_bf16 = krea2_cfg(v_c, pred_u.velocity[], cfg_scale, ctx)

        var v_f32 = cast_tensor(v_bf16, STDtype.F32, ctx)
        var v_latent = _unpatch[LH, LW](v_f32, ctx)
        latent = krea2_euler_step(latent, v_latent, t_cur, t_prev, ctx)
        ctx.synchronize()
        # Krea runs the complete denoise loop inside one blocking backend tick.
        # Emit the same newline-JSON progress event used by the Klein worker so
        # the server can update the live job while this call is still running.
        if progress_fd >= 0:
            try:
                write_msg(
                    progress_fd,
                    String("{\"ev\":\"progress\",\"step\":")
                    + String(si + 1)
                    + String(",\"total\":")
                    + String(sample_steps)
                    + String(",\"phase\":\"sampling\",\"preview\":\"\"}"),
                )
                if si + 1 == sample_steps:
                    wait_sampling_ack(progress_fd)
            except e:
                print("[krea2-infer] progress IPC skipped:", String(e))
        if si == 0 or si + 1 == sample_steps or (si + 1) % 5 == 0:
            print("[krea2-infer] step", si + 1, "/", sample_steps, " t=", t_cur)
    return latent^


def _lanpaint_combined_c(
    x_t: Tensor,
    score: Tensor,
    preserve_mask: Tensor,
    a_unknown: Float32,
    a_known: Float32,
    abt: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var c_unknown = lanpaint_coef_c(x_t, score, a_unknown, abt, ctx)
    var c_known = lanpaint_coef_c(x_t, score, a_known, abt, ctx)
    return mask_blend(preserve_mask, c_known, c_unknown, ctx)


def _lanpaint_branch_advance(
    x_t: Tensor,
    c: Tensor,
    noise: Tensor,
    preserve_mask: Tensor,
    a_unknown: Float32,
    dt_unknown: Float32,
    a_known: Float32,
    dt_known: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var x_unknown = lanpaint_overdamped_advance(
        x_t, c, noise, a_unknown, dt_unknown, ctx
    )
    var x_known = lanpaint_overdamped_advance(
        x_t, c, noise, a_known, dt_known, ctx
    )
    return mask_blend(preserve_mask, x_known, x_unknown, ctx)


def _lanpaint_branch_c_kick(
    x_t: Tensor,
    c_new: Tensor,
    c_previous: Tensor,
    preserve_mask: Tensor,
    dt_unknown: Float32,
    dt_known: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var delta = sub(c_new, c_previous, ctx)
    var x_unknown = add(x_t, mul_scalar(delta, dt_unknown, ctx), ctx)
    var x_known = add(x_t, mul_scalar(delta, dt_known, ctx), ctx)
    return mask_blend(preserve_mask, x_known, x_unknown, ctx)


struct Krea2LanPaintDampedBranch(Movable):
    var x: Tensor
    var velocity: Tensor

    def __init__(out self, var x: Tensor, var velocity: Tensor):
        self.x = x^
        self.velocity = velocity^


def _lanpaint_branch_damped_advance(
    x_t: Tensor,
    velocity: Tensor,
    c: Tensor,
    noise_y: Tensor,
    noise_v: Tensor,
    preserve_mask: Tensor,
    gamma: Float32,
    a_unknown: Float32,
    dt_unknown: Float32,
    a_known: Float32,
    dt_known: Float32,
    ctx: DeviceContext,
) raises -> Krea2LanPaintDampedBranch:
    var unknown = lanpaint_damped_advance(
        x_t, velocity, c, noise_y, noise_v,
        gamma, a_unknown, dt_unknown, ctx,
    )
    var known = lanpaint_damped_advance(
        x_t, velocity, c, noise_y, noise_v,
        gamma, a_known, dt_known, ctx,
    )
    var x_next = mask_blend(preserve_mask, known.x, unknown.x, ctx)
    var v_next = mask_blend(
        preserve_mask, known.velocity, unknown.velocity, ctx
    )
    return Krea2LanPaintDampedBranch(x_next^, v_next^)


def _lanpaint_branch_velocity_kick(
    velocity: Tensor,
    c_new: Tensor,
    c_previous: Tensor,
    preserve_mask: Tensor,
    gamma: Float32,
    dt_unknown: Float32,
    dt_known: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    # Upstream run_damped midpoint correction:
    #   v += sqrt(Gamma) * (C_new - C_previous) * dt
    var delta = sub(c_new, c_previous, ctx)
    var sqrt_gamma = gamma ** Float32(0.5)
    var v_unknown = add(
        velocity, mul_scalar(delta, sqrt_gamma * dt_unknown, ctx), ctx
    )
    var v_known = add(
        velocity, mul_scalar(delta, sqrt_gamma * dt_known, ctx), ctx
    )
    return mask_blend(preserve_mask, v_known, v_unknown, ctx)


def krea2_sample_lanpaint_latent[
    LH: Int, LW: Int, LTMAX: Int, LFULL: Int
](
    st: ShardedSafeTensors,
    key_prefix: String,
    cond_w: Krea2ResidentCond,
    fin: Krea2StreamFinal,
    lora: Krea2StackLora,
    cond: Krea2InlineCond,
    uncond: Krea2InlineCond,
    source_latent: Tensor,
    preserve_mask: Tensor,
    sample_steps: Int,
    cfg_scale: Float32,
    seed: UInt64,
    lanpaint_num_steps: Int,
    lanpaint_lambda: Float32,
    lanpaint_step_size: Float32,
    lanpaint_beta: Float32,
    lanpaint_friction: Float32,
    lanpaint_prompt_mode: String,
    lanpaint_early_stop: Int,
    resident: Optional[Krea2ResidentFp8],
    resident_i8: Optional[Krea2ResidentInt8],
    host_i8: Optional[Krea2HostInt8Inf],
    ctx: DeviceContext,
    use_fixed_mu_1_15: Bool = False,
    progress_fd: Int32 = Int32(-1),
) raises -> Tensor:
    """Krea2 flow sampling with LanPaint's damped SHO inner integrator.

    This follows upstream LanPaint's run_damped trajectory: exact stochastic
    harmonic-oscillator advances, midpoint C refresh, velocity kick, known-mask
    replacement, flow/VP conversion, and the outer Euler update. Model calls,
    sampler math, noise tensors, and latent state all execute in Mojo.
    """
    comptime imglen = (LH // 2) * (LW // 2)
    comptime assert LFULL == LTMAX + imglen, "Krea2 LanPaint LFULL mismatch"
    if sample_steps < 1:
        raise Error("krea2 LanPaint: sample steps must be >= 1")
    if lanpaint_num_steps < 0:
        raise Error("krea2 LanPaint: inner steps must be >= 0")
    if lanpaint_step_size <= Float32(0.0):
        raise Error("krea2 LanPaint: step size must be > 0")
    if lanpaint_beta <= Float32(0.0):
        raise Error("krea2 LanPaint: beta must be > 0")
    if lanpaint_friction <= Float32(0.0):
        raise Error("krea2 LanPaint: friction must be > 0")

    var base_noise = randn_torch([1, 16, LH, LW], seed, ctx)
    var latent = base_noise.clone(ctx)
    var seq = krea2_packed_seq_len(LH * 8, LW * 8)
    var ts = krea2_timesteps(
        seq, sample_steps,
        mu_override=Float32(1.15),
        use_mu_override=use_fixed_mu_1_15,
    )
    # Upstream LanPaint's KSAMPLER forces cfg_BIG=1.0 for every FLUX model.
    # Krea2 uses Comfy's FLUX model type, so the UI prompt-mode value must not
    # override the known-region guidance branch here.
    var cfg_big = Float32(1.0)
    print(
        "[krea2-lanpaint] damped steps=", sample_steps,
        " inner=", lanpaint_num_steps, " cfg=", cfg_scale,
        " cfg_big=", cfg_big, " lambda=", lanpaint_lambda,
        " step_size=", lanpaint_step_size, " beta=", lanpaint_beta,
        " friction=", lanpaint_friction, " mode=", lanpaint_prompt_mode,
    )
    print("[krea2-lanpaint] prefaulted worker image bytes=", prefault_self_executable())
    print(
        "[krea2-lanpaint] prefaulted mapped shared-library bytes=",
        prefault_mapped_shared_libraries(),
    )

    # All checkpoint/sidecar population must be finished before this boundary.
    # The product I/O gate treats any physical read after this event as fatal.
    if progress_fd >= 0:
        try:
            write_msg(
                progress_fd,
                String("{\"ev\":\"progress\",\"step\":0,\"total\":")
                + String(sample_steps)
                + String(",\"phase\":\"sampling\",\"preview\":\"\"}"),
            )
        except e:
            print("[krea2-lanpaint] progress IPC skipped:", String(e))

    for si in range(sample_steps):
        var t_cur = ts[si]
        var t_prev = ts[si + 1]
        var one_minus_t = Float32(1.0) - t_cur
        var denom = one_minus_t * one_minus_t + t_cur * t_cur
        var abt = (one_minus_t * one_minus_t) / denom
        var vp_factor = abt ** Float32(0.5) + (Float32(1.0) - abt) ** Float32(0.5)

        # Comfy flow noise_scaling(t, noise, source), then replace only known
        # pixels before switching to LanPaint's VP notation.
        var known_noised = add(
            mul_scalar(source_latent, one_minus_t, ctx),
            mul_scalar(base_noise, t_cur, ctx),
            ctx,
        )
        latent = mask_blend(preserve_mask, known_noised, latent, ctx)
        var x_t = mul_scalar(latent, vp_factor, ctx)

        var a_unknown = Float32(1.0) / (Float32(1.0) - abt)
        var a_known = (Float32(1.0) + lanpaint_lambda) / (Float32(1.0) - abt)
        var dt_unknown = lanpaint_step_size * (Float32(1.0) - abt)
        var dt_known = dt_unknown * lanpaint_beta
        # prepare_step_size() in upstream LanPaint reduces both branches to the
        # same Gamma; beta changes dt_known, not this friction coefficient.
        var gamma = (
            lanpaint_friction * lanpaint_friction
            / (Float32(0.2) * (Float32(1.0) - abt))
        )
        var inner_steps = lanpaint_num_steps
        if sample_steps - si <= lanpaint_early_stop:
            inner_steps = 0
        var c_state = Optional[Tensor](None)
        var velocity_state = Optional[Tensor](None)

        for li in range(inner_steps):
            if li > 0:
                var c_previous = c_state[].clone(ctx)
                var velocity_previous = velocity_state[].clone(ctx)
                var noise_y_half = randn_torch(
                    [1, 16, LH, LW],
                    seed + UInt64(1000000 + si * 65536 + li * 8),
                    ctx,
                )
                var noise_v_half = randn_torch(
                    [1, 16, LH, LW],
                    seed + UInt64(1000001 + si * 65536 + li * 8),
                    ctx,
                )
                var first_half = _lanpaint_branch_damped_advance(
                    x_t, velocity_previous, c_previous,
                    noise_y_half, noise_v_half, preserve_mask, gamma,
                    a_unknown, dt_unknown * 0.5,
                    a_known, dt_known * 0.5, ctx,
                )
                # Clone both fields so the move-only result remains whole for
                # destruction; each latent copy is tiny beside one DiT call.
                x_t = first_half.x.clone(ctx)
                velocity_state = Optional[Tensor](
                    first_half.velocity.clone(ctx)
                )

            var x_model = mul_scalar(x_t, Float32(1.0) / vp_factor, ctx)
            var pair = krea2_predict_comfy_velocity_pair[
                LH, LW, LTMAX, LFULL
            ](
                st, key_prefix, cond_w, fin, lora, cond, uncond,
                x_model, t_cur, cfg_scale, cfg_big,
                resident, resident_i8, host_i8, ctx,
            )
            var x0 = sub(x_model, mul_scalar(pair.normal, t_cur, ctx), ctx)
            var x0_big = sub(x_model, mul_scalar(pair.big, t_cur, ctx), ctx)
            var score = lanpaint_flow_score(
                x_t, x0, x0_big, source_latent, preserve_mask,
                lanpaint_lambda, ctx,
            )
            var c_new = _lanpaint_combined_c(
                x_t, score, preserve_mask, a_unknown, a_known, abt, ctx
            )

            if li == 0:
                # D=sqrt(2), so upstream's v0=randn*D/sqrt(2) is randn.
                var velocity_initial = randn_torch(
                    [1, 16, LH, LW],
                    seed + UInt64(1000000 + si * 65536),
                    ctx,
                )
                var noise_y_full = randn_torch(
                    [1, 16, LH, LW],
                    seed + UInt64(1000001 + si * 65536),
                    ctx,
                )
                var noise_v_full = randn_torch(
                    [1, 16, LH, LW],
                    seed + UInt64(1000002 + si * 65536),
                    ctx,
                )
                var full = _lanpaint_branch_damped_advance(
                    x_t, velocity_initial, c_new,
                    noise_y_full, noise_v_full, preserve_mask, gamma,
                    a_unknown, dt_unknown, a_known, dt_known, ctx,
                )
                x_t = full.x.clone(ctx)
                velocity_state = Optional[Tensor](full.velocity.clone(ctx))
            else:
                var c_previous = c_state[].clone(ctx)
                var velocity_kicked = _lanpaint_branch_velocity_kick(
                    velocity_state[], c_new, c_previous, preserve_mask,
                    gamma, dt_unknown, dt_known, ctx,
                )
                var noise_y_half = randn_torch(
                    [1, 16, LH, LW],
                    seed + UInt64(1000002 + si * 65536 + li * 8),
                    ctx,
                )
                var noise_v_half = randn_torch(
                    [1, 16, LH, LW],
                    seed + UInt64(1000003 + si * 65536 + li * 8),
                    ctx,
                )
                var second_half = _lanpaint_branch_damped_advance(
                    x_t, velocity_kicked, c_previous,
                    noise_y_half, noise_v_half, preserve_mask, gamma,
                    a_unknown, dt_unknown * 0.5,
                    a_known, dt_known * 0.5, ctx,
                )
                x_t = second_half.x.clone(ctx)
                velocity_state = Optional[Tensor](
                    second_half.velocity.clone(ctx)
                )
            c_state = Optional[Tensor](c_new^)

        # Match LanPaint's current Comfy Krea sampler contract. PaintMethod
        # mutates the sampler input to this advanced x, predicts denoised x0,
        # restores clean source x0 in the known region, and lets Euler derive
        # the effective velocity from that (advanced x, denoised x0) pair.
        var advanced_x = mul_scalar(x_t, Float32(1.0) / vp_factor, ctx)
        var final_pair = krea2_predict_comfy_velocity_pair[
            LH, LW, LTMAX, LFULL
        ](
            st, key_prefix, cond_w, fin, lora, cond, uncond,
            advanced_x, t_cur, cfg_scale, cfg_scale,
            resident, resident_i8, host_i8, ctx,
        )
        var final_x0 = sub(
            advanced_x, mul_scalar(final_pair.normal, t_cur, ctx), ctx
        )
        var denoised = mask_blend(
            preserve_mask, source_latent, final_x0, ctx
        )
        if t_cur <= Float32(1.0e-6):
            latent = denoised^
        else:
            var effective_v = mul_scalar(
                sub(advanced_x, denoised, ctx), Float32(1.0) / t_cur, ctx
            )
            latent = krea2_euler_step(
                advanced_x, effective_v, t_cur, t_prev, ctx
            )
        ctx.synchronize()

        if progress_fd >= 0:
            try:
                write_msg(
                    progress_fd,
                    String("{\"ev\":\"progress\",\"step\":")
                    + String(si + 1)
                    + String(",\"total\":")
                    + String(sample_steps)
                    + String(",\"phase\":\"sampling\",\"preview\":\"\"}"),
                )
                if si + 1 == sample_steps:
                    wait_sampling_ack(progress_fd)
            except e:
                print("[krea2-lanpaint] progress IPC skipped:", String(e))
        print("[krea2-lanpaint] step", si + 1, "/", sample_steps, " t=", t_cur)
    return latent^


def krea2_decode_latent_to_png[LH: Int, LW: Int](
    latent: Tensor, vae_dir: String, out_png: String, ctx: DeviceContext,
) raises:
    print("[krea2-infer] decoding ->", out_png)
    var dec = QwenImageVaeDecoder[LH, LW].load(vae_dir, ctx)
    var latent_bf16 = torch_f32_to_bf16_rne(latent, ctx)
    var img = dec.decode(latent_bf16, ctx)
    save_png(img, out_png, ctx, ValueRange.SIGNED)
