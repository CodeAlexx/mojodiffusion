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

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.io.env import env_or
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape, concat, slice, permute, zeros_device
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.random import randn
from serenitymojo.ops.random_torch import randn_torch
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.sampling.krea2_sampler import (
    krea2_packed_seq_len, krea2_timesteps, krea2_cfg, krea2_euler_step,
)
from serenitymojo.models.krea2.krea2_cache_reader import (
    krea2_patchify, krea2_build_pos, krea2_build_refiner_mask,
    KREA2_TXT_LAYERS, KREA2_TXT_DIM,
)
from serenitymojo.models.dit.krea2_dit import (
    krea2_first, krea2_temb, krea2_tmlp, krea2_tproj, krea2_txtmlp,
    krea2_text_fusion, build_krea2_rope,
    _wb, _scale, _txtf_bundle, Krea2TextFusionWeights,
)
from serenitymojo.models.krea2.krea2_block import Krea2BlockLora
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackLora, Krea2StreamFinal, KREA2_SLOTS_PER_BLOCK,
    krea2_stack_lora_forward_streamed,
    Krea2ResidentFp8, build_krea2_resident_fp8,
    Krea2ResidentInt8, build_krea2_resident_int8,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice, lora_adapter_to_device
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_save import NamedLora, load_lora_for_resume
from serenitymojo.serve.proc_ipc import write_msg

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

    def __init__(
        out self, var combined: TArc, var blk_vec: Tensor, var tmlp_out: Tensor,
        var cos: Tensor, var sin: Tensor,
    ):
        self.combined = combined^
        self.blk_vec = blk_vec^
        self.tmlp_out = tmlp_out^
        self.cos = cos^
        self.sin = sin^


def build_conditioning[LT: Int, LFULL: Int](
    cond_w: Krea2ResidentCond,
    img: Tensor,            # [1, IMGLEN, 64] BF16 patchified noised latent
    context: Tensor,        # [1, LT, 12, 2560] BF16 (LT == LTMAX)
    pos: Tensor,            # [1, LFULL, 3] F32
    t: Tensor,              # [1] F32 timestep
    real_text_len: Int,     # natural caption length lt (<= LT); real_len = lt+IMGLEN
    ctx: DeviceContext,
) raises -> _Cond:
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
    if real_text_len < LT:
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

    # length-bucket reorder → [TXT_real(0:lt) | IMG | TXT_pad(tail)].
    var combined: Tensor
    if real_text_len < LT:
        var real_text = slice(ctx_proj, 1, 0, real_text_len, ctx)
        var pad_text = slice(ctx_proj, 1, real_text_len, LT - real_text_len, ctx)
        var head = concat(1, ctx, real_text, img_e)
        combined = concat(1, ctx, head, pad_text)
    else:
        combined = concat(1, ctx, ctx_proj, img_e)

    # rope: reorder pos to match combined (text positions are all-zero → no rotation change).
    var pos_re: Tensor
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

    return _Cond(TArc(combined^), blk_vec2^, t3^, rcos^, rsin^)


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


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE — the full resident-base + LoRA + CFG + euler denoise loop → latent →
# VAE decode → PNG. One blocking call (the worker's step()); prints per-step
# progress. Hoisted verbatim from train_krea2._krea2_sample_resident_latent +
# _krea2_decode_latent_to_png.
# ══════════════════════════════════════════════════════════════════════════════
def krea2_sample_latent[LH: Int, LW: Int, LTMAX: Int, LFULL: Int](
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
) raises -> Tensor:
    comptime imglen = (LH // 2) * (LW // 2)
    comptime assert LFULL == LTMAX + imglen, "Krea2 sample LFULL mismatch"
    if sample_steps < 1:
        raise Error("krea2_infer: sample steps must be >= 1")

    # DIAGNOSTIC: inject a fixed noise latent (e.g. a torch-reference dump) instead of
    # generating from `seed`. Isolates whether the noise stream drives composition.
    var _inject = env_or("KREA2_INJECT_NOISE", String(""))
    var latent: Tensor
    if len(_inject) > 0:
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

    for si in range(sample_steps):
        var t_cur = ts[si]
        var t_prev = ts[si + 1]
        var img_tokens_f32 = krea2_patchify[LH, LW](latent, ctx)
        var img_tokens = torch_f32_to_bf16_rne(img_tokens_f32, ctx)
        var t_t = _t_scalar(t_cur, ctx)

        var c = build_conditioning[LTMAX, LFULL](
            cond_w, img_tokens, cond.context[], cond.pos[], t_t, cond.text_len, ctx,
        )
        var real_len_c = Optional[Int](cond.text_len + imglen)
        var pred_c = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
            c.combined, c.blk_vec, c.tmlp_out,
            st, key_prefix, NBLOCKS, lora, fin,
            c.cos, c.sin, EPS, cond.text_len, imglen, ctx, real_len_c, resident,
        )

        var v_bf16: Tensor
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
            except e:
                print("[krea2-infer] progress IPC skipped:", String(e))
        if si == 0 or si + 1 == sample_steps or (si + 1) % 5 == 0:
            print("[krea2-infer] step", si + 1, "/", sample_steps, " t=", t_cur)
    return latent^


def krea2_decode_latent_to_png[LH: Int, LW: Int](
    latent: Tensor, vae_dir: String, out_png: String, ctx: DeviceContext,
) raises:
    print("[krea2-infer] decoding ->", out_png)
    var dec = QwenImageVaeDecoder[LH, LW].load(vae_dir, ctx)
    var latent_bf16 = torch_f32_to_bf16_rne(latent, ctx)
    var img = dec.decode(latent_bf16, ctx)
    save_png(img, out_png, ctx, ValueRange.SIGNED)
