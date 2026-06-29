# models/chroma/chroma_stack_lora.mojo
#
# Chroma1-HD FULL DiT STACK *WITH LoRA*, BLOCK-SWAP OFFLOAD: forward (saving
# ckpt-inputs) + full-depth backward (training). Mirrors
# models/flux/flux_stack_lora.mojo::flux_stack_lora_{forward,backward}_offload
# EXACTLY for the block math — Chroma's per-block compute IS the proven Flux
# block (after the separate->fused row-stack the loader does), so this file
# REUSES the verified per-block LoRA fwd/bwd (models/flux/lora_block.mojo) and
# the FluxLoraSet carrier / optimizer while using Chroma-owned OneTrainer
# raw-key save/resume wrappers.
#
# WHAT DIFFERS FROM FLUX (and why this is a Chroma-specific stack, not a direct
# reuse of flux_stack_lora_*_offload):
#   (1) MODULATION SOURCE. Flux derives per-block ModVecs from
#       silu(t_embed+guidance+clip) -> per-block modulation.lin (FluxStackBase +
#       _embed_vec_forward + _mod_proj). Chroma has NO timestep/guidance/CLIP
#       embed chain and NO per-block modulation linears. Instead a frozen
#       distilled_guidance_layer APPROXIMATOR produces a pooled_temb table
#       [mod_index=344, D=3072] once per step (chroma_dit.mojo
#       approximator_forward), and each block's ModVecs are SLICED ROWS of that
#       table (chroma_dit.mojo double_block_smoke_forward:341-357 /
#       single_block_smoke_forward:447-450). Row layout (mod_index=344):
#         single blocks bi : rows 3*bi + {0:shift,1:scale,2:gate}      (0..113)
#         double img blocks: img_mod_start=3*38=114 ; bi -> 114+6*bi+{shift1,
#           scale1,gate1,shift2,scale2,gate2}                          (114..227)
#         double txt blocks: txt_mod_start=114+6*19=228 ; same 6-layout (228..341)
#         final layer       : rows 342:shift 343:scale
#   (2) NO guidance/vector vec, so there is no vec backward; the approximator is
#       FROZEN (LoRA scope) so the per-block mod-vec grads the block backward
#       returns are simply DISCARDED (they would flow into the frozen
#       approximator). Only the LoRA d_A/d_B are collected for the optimizer —
#       exactly the Klein/Ernie/Flux LoRA-scope contract.
#   (3) BLOCK WEIGHTS are streamed with DIFFUSERS keys (transformer_blocks.bi.* /
#       single_transformer_blocks.bi.*) and FUSED on the fly by row-stacking
#       to_q/to_k/to_v(/proj_mlp) (mirrors models/chroma/weights.mojo). Flux
#       streams pre-fused BFL keys (img_attn.qkv.weight). So this file provides
#       chroma _double/_single_weights_from_block that read Chroma's separate
#       projections and build the SAME StreamWeights/SingleBlockWeights the
#       proven block consumes.
#   (4) FINAL LAYER: layer_norm(no affine) -> modulate(scale,shift from rows
#       342/343) -> proj_out [out_ch, D]. Flux uses a final_adaln linear off
#       vec_silu + a final_lin. Chroma's proj_out is a plain linear (x_embedder's
#       inverse), shift/scale come straight from the approximator rows.
#
# Tenet 1 (no new block math): every block fwd/bwd arm is the proven Flux LoRA
#   block. Tenet 2: same offload streaming contract as flux_stack_lora_*_offload
#   (build plan -> TurboPlannedLoader -> await/prefetch/mark_done per block).
#
# Mojo 0.26.x+: def not fn; Tensor move-only; host List[Float32] carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

# proven per-block LoRA block (re-exported under chroma_* names in chroma_block).
from serenitymojo.models.chroma.chroma_block import (
    ChromaModVecs, ChromaSingleModVecs,
    ChromaStreamWeights, ChromaDoubleBlockWeights, ChromaSingleBlockWeights,
    ChromaDoubleBlockSaved, ChromaSingleBlockSaved,
    ChromaDoubleBlockLora, ChromaSingleBlockLora,
    chroma_double_block_lora_forward, chroma_double_block_lora_backward,
    chroma_single_block_lora_forward, chroma_single_block_lora_backward,
    DBL_STREAM_SLOTS, SGL_SLOTS,
    D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    S_SQ, S_SK, S_SV, S_PMLP, S_L2,
)

# Reuse the proven Flux LoRA carrier + optimizer. Chroma save/load is owned in
# this file because OneTrainer Chroma has its own target inventory and filters.
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet,
    FluxDirectDoRAGradSet, FluxDirectOFTGradSet,
    build_flux_lora_set, total_adapters,
    flux_lora_adamw_step,
    _dbl_base, _sgl_base, _double_lora_for, _single_lora_for,
    _flux_direct_dbl_slot_targeted, _flux_direct_sgl_slot_targeted,
    _flux_direct_expected_slots,
    _flux_direct_dora_zero_grads, _flux_direct_oft_zero_grads,
    _flux_direct_dora_double_for, _flux_direct_oft_double_for,
    _flux_direct_dora_single_for, _flux_direct_oft_single_for,
    _scatter_flux_dora_double, _scatter_flux_oft_double,
    _scatter_flux_dora_single, _scatter_flux_oft_single,
    _nonfinite_flux_direct_double, _nonfinite_flux_direct_single,
)
from serenitymojo.models.flux.lora_block import (
    FLUX_DIRECT_ALGO_DORA, FLUX_DIRECT_ALGO_OFT,
    double_block_direct_lycoris_forward, double_block_direct_lycoris_backward,
    single_block_direct_lycoris_forward, single_block_direct_lycoris_backward,
)
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectOFTSet,
    empty_flat_direct_dora_set, empty_flat_direct_oft_set,
    flat_direct_dora_append_from_weight, flat_direct_oft_append,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_save import (
    NamedLora,
    save_lora_onetrainer, load_lora_for_resume,
    save_lora_train_state, load_lora_train_state,
)
from serenitymojo.models.flux.flux_stack import (
    _add_lists, _zeros, _ones, _t, _concat_seq, _split_seq, _chunk,
)

from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.tensor_algebra import concat
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm_backward import layer_norm_backward
from serenitymojo.ops.elementwise_backward import modulate_backward


comptime TArc = ArcPointer[Tensor]


def _t_like(
    vals: List[Float32], var shape: List[Int], ref_weight: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var t = _t(vals, shape^, ctx)
    if t.dtype() == ref_weight.dtype():
        return t^
    return cast_tensor(t, ref_weight.dtype(), ctx)


# ── frozen stack-level base: x_embedder / context_embedder / proj_out + the
#    per-step pooled_temb modulation table (built by the approximator). ─────────
struct ChromaStackBase(Movable):
    var x_embedder_w: TArc          # [D, in_ch]
    var x_embedder_b: TArc          # [D]
    var context_embedder_w: TArc    # [D, txt_ch]
    var context_embedder_b: TArc    # [D]
    var proj_out_w: TArc            # [out_ch, D]
    var proj_out_b: TArc            # [out_ch]
    var num_double: Int
    var num_single: Int

    def __init__(
        out self,
        var x_embedder_w: TArc, var x_embedder_b: TArc,
        var context_embedder_w: TArc, var context_embedder_b: TArc,
        var proj_out_w: TArc, var proj_out_b: TArc,
        num_double: Int, num_single: Int,
    ):
        self.x_embedder_w = x_embedder_w^
        self.x_embedder_b = x_embedder_b^
        self.context_embedder_w = context_embedder_w^
        self.context_embedder_b = context_embedder_b^
        self.proj_out_w = proj_out_w^
        self.proj_out_b = proj_out_b^
        self.num_double = num_double
        self.num_single = num_single


# ── Chroma forward tape (lean; no embed-MLP acts — Chroma has none) ──────────
struct ChromaStackForward(Movable):
    var out: List[Float32]                 # [N_IMG, out_ch]
    var dbl_saved: List[ChromaDoubleBlockSaved]
    var sgl_saved: List[ChromaSingleBlockSaved]
    var dbl_img_mod: List[List[Float32]]   # num_double x [6D] (rows packed)
    var dbl_txt_mod: List[List[Float32]]
    var sgl_mod_flat: List[List[Float32]]  # num_single x [3D]
    var img_out: TArc                      # [N_IMG, D]
    var ln_img_out: TArc                   # [N_IMG, D]
    var final_shift: List[Float32]         # [D]
    var final_scale: List[Float32]         # [D]

    def __init__(
        out self,
        var out: List[Float32],
        var dbl_saved: List[ChromaDoubleBlockSaved],
        var sgl_saved: List[ChromaSingleBlockSaved],
        var dbl_img_mod: List[List[Float32]], var dbl_txt_mod: List[List[Float32]],
        var sgl_mod_flat: List[List[Float32]],
        var img_out: TArc, var ln_img_out: TArc,
        var final_shift: List[Float32], var final_scale: List[Float32],
    ):
        self.out = out^
        self.dbl_saved = dbl_saved^
        self.sgl_saved = sgl_saved^
        self.dbl_img_mod = dbl_img_mod^
        self.dbl_txt_mod = dbl_txt_mod^
        self.sgl_mod_flat = sgl_mod_flat^
        self.img_out = img_out^
        self.ln_img_out = ln_img_out^
        self.final_shift = final_shift^
        self.final_scale = final_scale^


# ── modulation-row indexing (chroma_dit.mojo layout; see header) ─────────────
def _dbl_img_mod_flat(pooled: List[Float32], bi: Int, num_double: Int, num_single: Int, D: Int) -> List[Float32]:
    var img_mod_start = 3 * num_single
    var base_row = img_mod_start + 6 * bi
    var out = List[Float32]()
    for r in range(6):
        var off = (base_row + r) * D
        for c in range(D):
            out.append(pooled[off + c])
    return out^


def _dbl_txt_mod_flat(pooled: List[Float32], bi: Int, num_double: Int, num_single: Int, D: Int) -> List[Float32]:
    var img_mod_start = 3 * num_single
    var txt_mod_start = img_mod_start + 6 * num_double
    var base_row = txt_mod_start + 6 * bi
    var out = List[Float32]()
    for r in range(6):
        var off = (base_row + r) * D
        for c in range(D):
            out.append(pooled[off + c])
    return out^


def _sgl_mod_flat(pooled: List[Float32], bi: Int, D: Int) -> List[Float32]:
    var base_row = 3 * bi
    var out = List[Float32]()
    for r in range(3):
        var off = (base_row + r) * D
        for c in range(D):
            out.append(pooled[off + c])
    return out^


def _final_shift_scale(pooled: List[Float32], mod_index: Int, D: Int) -> List[List[Float32]]:
    # rows mod_index-2 (shift), mod_index-1 (scale).
    var shift = List[Float32]()
    var scale = List[Float32]()
    var soff = (mod_index - 2) * D
    var coff = (mod_index - 1) * D
    for c in range(D):
        shift.append(pooled[soff + c])
    for c in range(D):
        scale.append(pooled[coff + c])
    var out = List[List[Float32]]()
    out.append(shift^); out.append(scale^)
    return out^


def _modvecs_from_flat6(flat: List[Float32], D: Int) -> ChromaModVecs:
    # flat is [6D]: shift1,scale1,gate1,shift2,scale2,gate2.
    return ChromaModVecs(
        _chunk(flat, 0, D), _chunk(flat, 1, D), _chunk(flat, 2, D),
        _chunk(flat, 3, D), _chunk(flat, 4, D), _chunk(flat, 5, D),
    )


def _single_modvecs_from_flat3(flat: List[Float32], D: Int) -> ChromaSingleModVecs:
    return ChromaSingleModVecs(_chunk(flat, 0, D), _chunk(flat, 1, D), _chunk(flat, 2, D))


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ── OneTrainer raw-key save/resume surface for Chroma block adapters ─────────
# The default save/resume path keeps the full current block-projection surface.
# The `_for_layer_filter` variants below apply OneTrainer's substring
# layer_filter contract at save/resume time, so the local Chroma baseline
# `attn,ff.net` can save its 304-adapter inventory without narrowing the
# broader full-surface carrier.
def _chroma_dbl_stream_prefix(bi: Int, stream_img: Bool, slot: Int) -> String:
    var b = String("lora_transformer_transformer_blocks_") + String(bi) + "_"
    if stream_img:
        if slot == D_SQ:
            return b + "attn_to_q"
        elif slot == D_SK:
            return b + "attn_to_k"
        elif slot == D_SV:
            return b + "attn_to_v"
        elif slot == D_PROJ:
            return b + "attn_to_out_0"
        elif slot == D_MLP0:
            return b + "ff_net_0_proj"
        return b + "ff_net_2"
    else:
        if slot == D_SQ:
            return b + "attn_add_q_proj"
        elif slot == D_SK:
            return b + "attn_add_k_proj"
        elif slot == D_SV:
            return b + "attn_add_v_proj"
        elif slot == D_PROJ:
            return b + "attn_to_add_out"
        elif slot == D_MLP0:
            return b + "ff_context_net_0_proj"
        return b + "ff_context_net_2"


def _chroma_sgl_prefix(bi: Int, slot: Int) -> String:
    var b = String("lora_transformer_single_transformer_blocks_") + String(bi) + "_"
    if slot == S_SQ:
        return b + "attn_to_q"
    elif slot == S_SK:
        return b + "attn_to_k"
    elif slot == S_SV:
        return b + "attn_to_v"
    elif slot == S_PMLP:
        return b + "proj_mlp"
    return b + "proj_out"


def _chroma_dbl_stream_raw_name(bi: Int, stream_img: Bool, slot: Int) -> String:
    var b = String("transformer_blocks.") + String(bi) + "."
    if stream_img:
        if slot == D_SQ:
            return b + "attn.to_q"
        elif slot == D_SK:
            return b + "attn.to_k"
        elif slot == D_SV:
            return b + "attn.to_v"
        elif slot == D_PROJ:
            return b + "attn.to_out.0"
        elif slot == D_MLP0:
            return b + "ff.net.0.proj"
        return b + "ff.net.2"
    else:
        if slot == D_SQ:
            return b + "attn.add_q_proj"
        elif slot == D_SK:
            return b + "attn.add_k_proj"
        elif slot == D_SV:
            return b + "attn.add_v_proj"
        elif slot == D_PROJ:
            return b + "attn.to_add_out"
        elif slot == D_MLP0:
            return b + "ff_context.net.0.proj"
        return b + "ff_context.net.2"


def _chroma_sgl_raw_name(bi: Int, slot: Int) -> String:
    var b = String("single_transformer_blocks.") + String(bi) + "."
    if slot == S_SQ:
        return b + "attn.to_q"
    elif slot == S_SK:
        return b + "attn.to_k"
    elif slot == S_SV:
        return b + "attn.to_v"
    elif slot == S_PMLP:
        return b + "proj_mlp"
    return b + "proj_out"


def _chroma_layer_filter_matches(raw_name: String, layer_filter: String) -> Bool:
    # Mirrors OneTrainer's non-regex layer_filter: split on comma and match
    # selected substrings against the raw Diffusers module name.
    var parts = layer_filter.split(",")
    if len(parts) == 0:
        return True
    for i in range(len(parts)):
        var part = parts[i].strip()
        if part == String("") or raw_name.find(part) >= 0:
            return True
    return False


def chroma_lora_prefixes(num_double: Int, num_single: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_double):
        for s in range(DBL_STREAM_SLOTS):
            out.append(_chroma_dbl_stream_prefix(bi, True, s))
        for s in range(DBL_STREAM_SLOTS):
            out.append(_chroma_dbl_stream_prefix(bi, False, s))
    for bi in range(num_single):
        for s in range(SGL_SLOTS):
            out.append(_chroma_sgl_prefix(bi, s))
    return out^


def chroma_lora_prefixes_for_layer_filter(
    num_double: Int, num_single: Int, layer_filter: String
) -> List[String]:
    var out = List[String]()
    for bi in range(num_double):
        for s in range(DBL_STREAM_SLOTS):
            if _chroma_layer_filter_matches(
                _chroma_dbl_stream_raw_name(bi, True, s), layer_filter
            ):
                out.append(_chroma_dbl_stream_prefix(bi, True, s))
        for s in range(DBL_STREAM_SLOTS):
            if _chroma_layer_filter_matches(
                _chroma_dbl_stream_raw_name(bi, False, s), layer_filter
            ):
                out.append(_chroma_dbl_stream_prefix(bi, False, s))
    for bi in range(num_single):
        for s in range(SGL_SLOTS):
            if _chroma_layer_filter_matches(_chroma_sgl_raw_name(bi, s), layer_filter):
                out.append(_chroma_sgl_prefix(bi, s))
    return out^


def _chroma_set_from_named(
    named: List[NamedLora], num_double: Int, num_single: Int, rank: Int,
) -> FluxLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(named)):
        ad.append(named[i].adapter.copy())
    return FluxLoraSet(ad^, num_double, num_single, rank)


def _chroma_named_loras(set: FluxLoraSet) -> List[NamedLora]:
    var prefixes = chroma_lora_prefixes(set.num_double, set.num_single)
    var named = List[NamedLora]()
    var n = total_adapters(set)
    for i in range(n):
        named.append(NamedLora(prefixes[i], set.ad[i].copy()))
    return named^


def _chroma_named_loras_for_layer_filter(
    set: FluxLoraSet, layer_filter: String
) -> List[NamedLora]:
    var named = List[NamedLora]()
    var idx = 0
    for bi in range(set.num_double):
        for s in range(DBL_STREAM_SLOTS):
            if _chroma_layer_filter_matches(
                _chroma_dbl_stream_raw_name(bi, True, s), layer_filter
            ):
                named.append(
                    NamedLora(_chroma_dbl_stream_prefix(bi, True, s), set.ad[idx].copy())
                )
            idx += 1
        for s in range(DBL_STREAM_SLOTS):
            if _chroma_layer_filter_matches(
                _chroma_dbl_stream_raw_name(bi, False, s), layer_filter
            ):
                named.append(
                    NamedLora(_chroma_dbl_stream_prefix(bi, False, s), set.ad[idx].copy())
                )
            idx += 1
    for bi in range(set.num_single):
        for s in range(SGL_SLOTS):
            if _chroma_layer_filter_matches(_chroma_sgl_raw_name(bi, s), layer_filter):
                named.append(NamedLora(_chroma_sgl_prefix(bi, s), set.ad[idx].copy()))
            idx += 1
    return named^


def save_chroma_lora(set: FluxLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    return save_lora_onetrainer(_chroma_named_loras(set), path, ctx)


def save_chroma_lora_for_layer_filter(
    set: FluxLoraSet, layer_filter: String, path: String, ctx: DeviceContext
) raises -> Int:
    return save_lora_onetrainer(_chroma_named_loras_for_layer_filter(set, layer_filter), path, ctx)


def save_chroma_lora_state(set: FluxLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    return save_lora_train_state(_chroma_named_loras(set), path, ctx)


def save_chroma_lora_state_for_layer_filter(
    set: FluxLoraSet, layer_filter: String, path: String, ctx: DeviceContext
) raises -> Int:
    return save_lora_train_state(_chroma_named_loras_for_layer_filter(set, layer_filter), path, ctx)


def load_chroma_lora_resume(
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> FluxLoraSet:
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(
        chroma_lora_prefixes(num_double, num_single), scale, path, ctx
    )
    return _chroma_set_from_named(named, num_double, num_single, rank)


def load_chroma_lora_resume_for_layer_filter(
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    layer_filter: String, path: String, ctx: DeviceContext,
) raises -> FluxLoraSet:
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(
        chroma_lora_prefixes_for_layer_filter(num_double, num_single, layer_filter),
        scale, path, ctx,
    )
    return _chroma_set_from_named(named, num_double, num_single, rank)


def load_chroma_lora_state(
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> FluxLoraSet:
    var scale = alpha / Float32(rank)
    var named = load_lora_train_state(
        chroma_lora_prefixes(num_double, num_single), scale, path, ctx
    )
    return _chroma_set_from_named(named, num_double, num_single, rank)


def load_chroma_lora_state_for_layer_filter(
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    layer_filter: String, path: String, ctx: DeviceContext,
) raises -> FluxLoraSet:
    var scale = alpha / Float32(rank)
    var named = load_lora_train_state(
        chroma_lora_prefixes_for_layer_filter(num_double, num_single, layer_filter),
        scale, path, ctx,
    )
    return _chroma_set_from_named(named, num_double, num_single, rank)


# ── streamed-block -> fused weight structs (diffusers keys; row-stack q;k;v) ──
# Keep checkpoint tensors device-resident in their stored dtype. Only the fused
# qkv/w1 tensors are materialized, because the proven Flux block consumes fused
# projections; unfused block tensors are passed by ArcPointer handle.
def _block_tensor(block: Block, key: String) raises -> TArc:
    if not (key in block):
        raise Error(String("Chroma offload block missing tensor: ") + key)
    return block[key].copy()


def _block_clone(block: Block, key: String, ctx: DeviceContext) raises -> Tensor:
    if not (key in block):
        raise Error(String("Chroma offload block missing tensor: ") + key)
    return block[key][].clone(ctx)


def _row_stack3_block(
    block: Block, ka: String, kb: String, kc: String, ctx: DeviceContext
) raises -> TArc:
    var a = _block_clone(block, ka, ctx)
    var b = _block_clone(block, kb, ctx)
    var c = _block_clone(block, kc, ctx)
    return TArc(concat(0, ctx, a, b, c))


def _row_stack4_block(
    block: Block, ka: String, kb: String, kc: String, kd: String, ctx: DeviceContext
) raises -> TArc:
    var a = _block_clone(block, ka, ctx)
    var b = _block_clone(block, kb, ctx)
    var c = _block_clone(block, kc, ctx)
    var d = _block_clone(block, kd, ctx)
    return TArc(concat(0, ctx, a, b, c, d))


def _chroma_stream_from_block(
    block: Block, bp: String,
    qk: String, kk: String, vk: String, outk: String,
    mlp0k: String, mlp2k: String, nqk: String, nkk: String,
    D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
) raises -> ChromaStreamWeights:
    var wqkv = _row_stack3_block(
        block, bp + qk + String(".weight"), bp + kk + String(".weight"),
        bp + vk + String(".weight"), ctx,
    )
    var bqkv = _row_stack3_block(
        block, bp + qk + String(".bias"), bp + kk + String(".bias"),
        bp + vk + String(".bias"), ctx,
    )
    return ChromaStreamWeights(
        wqkv^, bqkv^,
        _block_tensor(block, bp + outk + String(".weight")),
        _block_tensor(block, bp + outk + String(".bias")),
        _block_tensor(block, bp + mlp0k + String(".weight")),
        _block_tensor(block, bp + mlp0k + String(".bias")),
        _block_tensor(block, bp + mlp2k + String(".weight")),
        _block_tensor(block, bp + mlp2k + String(".bias")),
        _block_tensor(block, bp + nqk + String(".weight")),
        _block_tensor(block, bp + nkk + String(".weight")),
    )


def _chroma_double_from_block(
    block: Block, bp: String, D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
) raises -> ChromaDoubleBlockWeights:
    var img = _chroma_stream_from_block(
        block, bp,
        String("attn.to_q"), String("attn.to_k"), String("attn.to_v"),
        String("attn.to_out.0"), String("ff.net.0.proj"), String("ff.net.2"),
        String("attn.norm_q"), String("attn.norm_k"), D, Fmlp, Dh, ctx,
    )
    var txt = _chroma_stream_from_block(
        block, bp,
        String("attn.add_q_proj"), String("attn.add_k_proj"), String("attn.add_v_proj"),
        String("attn.to_add_out"), String("ff_context.net.0.proj"), String("ff_context.net.2"),
        String("attn.norm_added_q"), String("attn.norm_added_k"), D, Fmlp, Dh, ctx,
    )
    return ChromaDoubleBlockWeights(img^, txt^)


def _chroma_single_from_block(
    block: Block, sp: String, D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
) raises -> ChromaSingleBlockWeights:
    var w1 = _row_stack4_block(
        block,
        sp + String("attn.to_q.weight"), sp + String("attn.to_k.weight"),
        sp + String("attn.to_v.weight"), sp + String("proj_mlp.weight"), ctx,
    )
    var b1 = _row_stack4_block(
        block,
        sp + String("attn.to_q.bias"), sp + String("attn.to_k.bias"),
        sp + String("attn.to_v.bias"), sp + String("proj_mlp.bias"), ctx,
    )
    return ChromaSingleBlockWeights(
        w1^, b1^,
        _block_tensor(block, sp + String("proj_out.weight")),
        _block_tensor(block, sp + String("proj_out.bias")),
        _block_tensor(block, sp + String("attn.norm_q.weight")),
        _block_tensor(block, sp + String("attn.norm_k.weight")),
    )


def _block_weight_host(block: Block, key: String, ctx: DeviceContext) raises -> List[Float32]:
    return _block_tensor(block, key)[].to_host(ctx)


def _chroma_dbl_stream_local_name(stream_img: Bool, slot: Int) -> String:
    if stream_img:
        if slot == D_SQ:
            return String("attn.to_q")
        if slot == D_SK:
            return String("attn.to_k")
        if slot == D_SV:
            return String("attn.to_v")
        if slot == D_PROJ:
            return String("attn.to_out.0")
        if slot == D_MLP0:
            return String("ff.net.0.proj")
        return String("ff.net.2")
    if slot == D_SQ:
        return String("attn.add_q_proj")
    if slot == D_SK:
        return String("attn.add_k_proj")
    if slot == D_SV:
        return String("attn.add_v_proj")
    if slot == D_PROJ:
        return String("attn.to_add_out")
    if slot == D_MLP0:
        return String("ff_context.net.0.proj")
    return String("ff_context.net.2")


def _append_chroma_direct_dora_weight(
    mut set: FlatDirectDoRASet, block: Block, key: String, prefix: String,
    in_f: Int, out_f: Int, rank: Int, alpha: Float32, seed: UInt64,
    wd_on_out: Bool, ctx: DeviceContext,
) raises:
    var w = _block_weight_host(block, key + String(".weight"), ctx)
    flat_direct_dora_append_from_weight(
        set, w^, in_f, out_f, rank, alpha, prefix, seed, wd_on_out,
    )


def _append_chroma_direct_dora_stream(
    mut set: FlatDirectDoRASet, block: Block, bp: String,
    bi: Int, stream_img: Bool, targets: Int,
    D: Int, Fmlp: Int, rank: Int, alpha: Float32, seed: UInt64,
    wd_on_out: Bool, ctx: DeviceContext,
) raises:
    var stream_off = 0 if stream_img else DBL_STREAM_SLOTS
    if _flux_direct_dbl_slot_targeted(stream_off + D_SQ, targets):
        _append_chroma_direct_dora_weight(
            set, block, bp + _chroma_dbl_stream_local_name(stream_img, D_SQ),
            _chroma_dbl_stream_prefix(bi, stream_img, D_SQ), D, D, rank, alpha,
            seed + UInt64(bi * (2 * DBL_STREAM_SLOTS) + stream_off + D_SQ), wd_on_out, ctx,
        )
    if _flux_direct_dbl_slot_targeted(stream_off + D_SK, targets):
        _append_chroma_direct_dora_weight(
            set, block, bp + _chroma_dbl_stream_local_name(stream_img, D_SK),
            _chroma_dbl_stream_prefix(bi, stream_img, D_SK), D, D, rank, alpha,
            seed + UInt64(bi * (2 * DBL_STREAM_SLOTS) + stream_off + D_SK), wd_on_out, ctx,
        )
    if _flux_direct_dbl_slot_targeted(stream_off + D_SV, targets):
        _append_chroma_direct_dora_weight(
            set, block, bp + _chroma_dbl_stream_local_name(stream_img, D_SV),
            _chroma_dbl_stream_prefix(bi, stream_img, D_SV), D, D, rank, alpha,
            seed + UInt64(bi * (2 * DBL_STREAM_SLOTS) + stream_off + D_SV), wd_on_out, ctx,
        )
    if _flux_direct_dbl_slot_targeted(stream_off + D_PROJ, targets):
        _append_chroma_direct_dora_weight(
            set, block, bp + _chroma_dbl_stream_local_name(stream_img, D_PROJ),
            _chroma_dbl_stream_prefix(bi, stream_img, D_PROJ), D, D, rank, alpha,
            seed + UInt64(bi * (2 * DBL_STREAM_SLOTS) + stream_off + D_PROJ), wd_on_out, ctx,
        )
    if _flux_direct_dbl_slot_targeted(stream_off + D_MLP0, targets):
        _append_chroma_direct_dora_weight(
            set, block, bp + _chroma_dbl_stream_local_name(stream_img, D_MLP0),
            _chroma_dbl_stream_prefix(bi, stream_img, D_MLP0), D, Fmlp, rank, alpha,
            seed + UInt64(bi * (2 * DBL_STREAM_SLOTS) + stream_off + D_MLP0), wd_on_out, ctx,
        )
    if _flux_direct_dbl_slot_targeted(stream_off + D_MLP2, targets):
        _append_chroma_direct_dora_weight(
            set, block, bp + _chroma_dbl_stream_local_name(stream_img, D_MLP2),
            _chroma_dbl_stream_prefix(bi, stream_img, D_MLP2), Fmlp, D, rank, alpha,
            seed + UInt64(bi * (2 * DBL_STREAM_SLOTS) + stream_off + D_MLP2), wd_on_out, ctx,
        )


def _append_chroma_direct_dora_single(
    mut set: FlatDirectDoRASet, block: Block, sp: String,
    num_double: Int, bi: Int, targets: Int,
    D: Int, Fmlp: Int, rank: Int, alpha: Float32, seed: UInt64,
    wd_on_out: Bool, ctx: DeviceContext,
) raises:
    var base = num_double * (2 * DBL_STREAM_SLOTS) + bi * SGL_SLOTS
    if _flux_direct_sgl_slot_targeted(S_SQ, targets):
        _append_chroma_direct_dora_weight(
            set, block, sp + String("attn.to_q"), _chroma_sgl_prefix(bi, S_SQ),
            D, D, rank, alpha, seed + UInt64(base + S_SQ), wd_on_out, ctx,
        )
    if _flux_direct_sgl_slot_targeted(S_SK, targets):
        _append_chroma_direct_dora_weight(
            set, block, sp + String("attn.to_k"), _chroma_sgl_prefix(bi, S_SK),
            D, D, rank, alpha, seed + UInt64(base + S_SK), wd_on_out, ctx,
        )
    if _flux_direct_sgl_slot_targeted(S_SV, targets):
        _append_chroma_direct_dora_weight(
            set, block, sp + String("attn.to_v"), _chroma_sgl_prefix(bi, S_SV),
            D, D, rank, alpha, seed + UInt64(base + S_SV), wd_on_out, ctx,
        )
    if _flux_direct_sgl_slot_targeted(S_PMLP, targets):
        _append_chroma_direct_dora_weight(
            set, block, sp + String("proj_mlp"), _chroma_sgl_prefix(bi, S_PMLP),
            D, Fmlp, rank, alpha, seed + UInt64(base + S_PMLP), wd_on_out, ctx,
        )
    if _flux_direct_sgl_slot_targeted(S_L2, targets):
        _append_chroma_direct_dora_weight(
            set, block, sp + String("proj_out"), _chroma_sgl_prefix(bi, S_L2),
            D + Fmlp, D, rank, alpha, seed + UInt64(base + S_L2), wd_on_out, ctx,
        )


def build_chroma_direct_dora_set_from_offload(
    mut loader: TurboPlannedLoader,
    num_double: Int, num_single: Int,
    D: Int, Fmlp: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool, ctx: DeviceContext,
) raises -> FlatDirectDoRASet:
    if loader.block_count() < num_double + num_single:
        raise Error("build_chroma_direct_dora_set_from_offload: loader depth too small")
    var set = empty_flat_direct_dora_set()
    if num_double + num_single > 0:
        loader.prefetch_with_ctx(0, ctx)
    for bi in range(num_double):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var bp = handle.prefix + String(".")
        _append_chroma_direct_dora_stream(
            set, handle.block, bp, bi, True, targets,
            D, Fmlp, rank, alpha, seed, wd_on_out, ctx,
        )
        _append_chroma_direct_dora_stream(
            set, handle.block, bp, bi, False, targets,
            D, Fmlp, rank, alpha, seed, wd_on_out, ctx,
        )
        loader.mark_active_block_done(ctx)
    for bi in range(num_single):
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        _append_chroma_direct_dora_single(
            set, handle.block, handle.prefix + String("."),
            num_double, bi, targets, D, Fmlp, rank, alpha, seed, wd_on_out, ctx,
        )
        loader.mark_active_block_done(ctx)
    if len(set.ad) != _flux_direct_expected_slots(num_double, num_single, targets):
        raise Error("build_chroma_direct_dora_set_from_offload: direct slot count mismatch")
    return set^


def build_chroma_direct_oft_set_for_stack(
    num_double: Int, num_single: Int,
    D: Int, Fmlp: Int, block_size: Int, targets: Int,
) raises -> FlatDirectOFTSet:
    var set = empty_flat_direct_oft_set()
    for bi in range(num_double):
        for slot in range(DBL_STREAM_SLOTS):
            if _flux_direct_dbl_slot_targeted(slot, targets):
                if slot == D_MLP0:
                    flat_direct_oft_append(set, D, Fmlp, block_size, _chroma_dbl_stream_prefix(bi, True, slot))
                elif slot == D_MLP2:
                    flat_direct_oft_append(set, Fmlp, D, block_size, _chroma_dbl_stream_prefix(bi, True, slot))
                else:
                    flat_direct_oft_append(set, D, D, block_size, _chroma_dbl_stream_prefix(bi, True, slot))
        for slot in range(DBL_STREAM_SLOTS):
            if _flux_direct_dbl_slot_targeted(DBL_STREAM_SLOTS + slot, targets):
                if slot == D_MLP0:
                    flat_direct_oft_append(set, D, Fmlp, block_size, _chroma_dbl_stream_prefix(bi, False, slot))
                elif slot == D_MLP2:
                    flat_direct_oft_append(set, Fmlp, D, block_size, _chroma_dbl_stream_prefix(bi, False, slot))
                else:
                    flat_direct_oft_append(set, D, D, block_size, _chroma_dbl_stream_prefix(bi, False, slot))
    for bi in range(num_single):
        if _flux_direct_sgl_slot_targeted(S_SQ, targets):
            flat_direct_oft_append(set, D, D, block_size, _chroma_sgl_prefix(bi, S_SQ))
        if _flux_direct_sgl_slot_targeted(S_SK, targets):
            flat_direct_oft_append(set, D, D, block_size, _chroma_sgl_prefix(bi, S_SK))
        if _flux_direct_sgl_slot_targeted(S_SV, targets):
            flat_direct_oft_append(set, D, D, block_size, _chroma_sgl_prefix(bi, S_SV))
        if _flux_direct_sgl_slot_targeted(S_PMLP, targets):
            flat_direct_oft_append(set, D, Fmlp, block_size, _chroma_sgl_prefix(bi, S_PMLP))
        if _flux_direct_sgl_slot_targeted(S_L2, targets):
            flat_direct_oft_append(set, D + Fmlp, D, block_size, _chroma_sgl_prefix(bi, S_L2))
    if len(set.ad) != _flux_direct_expected_slots(num_double, num_single, targets):
        raise Error("build_chroma_direct_oft_set_for_stack: direct slot count mismatch")
    return set^


# ═════════════════════════════════════════════════════════════════════════════
# FULL FORWARD WITH LoRA, BLOCK-SWAP OFFLOAD.
#   Inputs: img_tokens [N_IMG,in_ch], txt_tokens [N_TXT,txt_ch], the per-step
#   pooled_temb table [mod_index*D] (built by the frozen approximator), cos/sin
#   rope flats. Streams 19 double then 38 single blocks one at a time.
# ═════════════════════════════════════════════════════════════════════════════
def chroma_stack_lora_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    pooled: List[Float32], mod_index: Int,
    base: ChromaStackBase,
    mut loader: TurboPlannedLoader, lora: FluxLoraSet,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ChromaStackForward:
    var num_double = lora.num_double
    var num_single = lora.num_single

    loader.prefetch_with_ctx(0, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # input projections (frozen base linears).
    var bi_img = Optional[Tensor](base.x_embedder_b[].clone(ctx))
    var img = linear(
        _t_like(img_tokens.copy(), [N_IMG, in_ch], base.x_embedder_w[], ctx),
        base.x_embedder_w[], bi_img, ctx,
    ).to_host(ctx)
    var bi_txt = Optional[Tensor](base.context_embedder_b[].clone(ctx))
    var txt = linear(
        _t_like(txt_tokens.copy(), [N_TXT, txt_ch], base.context_embedder_w[], ctx),
        base.context_embedder_w[], bi_txt, ctx,
    ).to_host(ctx)

    var dbl_img_mod = List[List[Float32]]()
    var dbl_txt_mod = List[List[Float32]]()
    var dbl_saved = List[ChromaDoubleBlockSaved]()
    for bi in range(num_double):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _chroma_double_from_block(handle.block, handle.prefix + String("."), D, Fmlp, Dh, ctx)
        var im_flat = _dbl_img_mod_flat(pooled, bi, num_double, num_single, D)
        var tm_flat = _dbl_txt_mod_flat(pooled, bi, num_double, num_single, D)
        var im = _modvecs_from_flat6(im_flat, D)
        var tm = _modvecs_from_flat6(tm_flat, D)
        var bl = _double_lora_for(lora, bi)
        var fwd = chroma_double_block_lora_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), w, im, tm, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        dbl_saved.append(fwd.saved.copy())
        dbl_img_mod.append(im_flat^)
        dbl_txt_mod.append(tm_flat^)
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)

    # joint sequence: txt FIRST then img (Chroma/Flux convention).
    var x = _concat_seq(txt, img)

    var sgl_mod_flat = List[List[Float32]]()
    var sgl_saved = List[ChromaSingleBlockSaved]()
    for bi in range(num_single):
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w = _chroma_single_from_block(handle.block, handle.prefix + String("."), D, Fmlp, Dh, ctx)
        var sm_flat = _sgl_mod_flat(pooled, bi, D)
        var sm = _single_modvecs_from_flat3(sm_flat, D)
        var bl = _single_lora_for(lora, bi)
        var fwd = chroma_single_block_lora_forward[H, Dh, S](
            x.copy(), w, sm, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        sgl_saved.append(fwd.saved.copy())
        sgl_mod_flat.append(sm_flat^)
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)

    var parts = _split_seq(x, N_TXT, N_IMG, D)
    var img_out = parts[1].copy()

    # final layer: layer_norm(no affine) -> modulate(scale,shift) -> proj_out.
    var ss = _final_shift_scale(pooled, mod_index, D)
    var final_shift = ss[0].copy()
    var final_scale = ss[1].copy()

    var ln_img_out = layer_norm(
        _t(img_out.copy(), [N_IMG, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), eps, ctx,
    ).to_host(ctx)
    var normed = modulate(
        _t(ln_img_out.copy(), [N_IMG, D], ctx),
        _t(final_scale.copy(), [D], ctx), _t(final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var pb = Optional[Tensor](base.proj_out_b[].clone(ctx))
    var out = linear(
        _t_like(normed, [N_IMG, D], base.proj_out_w[], ctx),
        base.proj_out_w[], pb, ctx,
    ).to_host(ctx)

    return ChromaStackForward(
        out^, dbl_saved^, sgl_saved^,
        dbl_img_mod^, dbl_txt_mod^, sgl_mod_flat^,
        TArc(_t(img_out^, [N_IMG, D], ctx)), TArc(_t(ln_img_out^, [N_IMG, D], ctx)),
        final_shift^, final_scale^,
    )


def _chroma_stack_direct_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    pooled: List[Float32], mod_index: Int,
    base: ChromaStackBase,
    mut loader: TurboPlannedLoader,
    dora: FlatDirectDoRASet, oft: FlatDirectOFTSet, algo: Int,
    num_double: Int, num_single: Int, targets: Int,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ChromaStackForward:
    loader.prefetch_with_ctx(0, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    var bi_img = Optional[Tensor](base.x_embedder_b[].clone(ctx))
    var img = linear(
        _t_like(img_tokens.copy(), [N_IMG, in_ch], base.x_embedder_w[], ctx),
        base.x_embedder_w[], bi_img, ctx,
    ).to_host(ctx)
    var bi_txt = Optional[Tensor](base.context_embedder_b[].clone(ctx))
    var txt = linear(
        _t_like(txt_tokens.copy(), [N_TXT, txt_ch], base.context_embedder_w[], ctx),
        base.context_embedder_w[], bi_txt, ctx,
    ).to_host(ctx)

    var dbl_img_mod = List[List[Float32]]()
    var dbl_txt_mod = List[List[Float32]]()
    var dbl_saved = List[ChromaDoubleBlockSaved]()
    for bi in range(num_double):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _chroma_double_from_block(handle.block, handle.prefix + String("."), D, Fmlp, Dh, ctx)
        var im_flat = _dbl_img_mod_flat(pooled, bi, num_double, num_single, D)
        var tm_flat = _dbl_txt_mod_flat(pooled, bi, num_double, num_single, D)
        var im = _modvecs_from_flat6(im_flat, D)
        var tm = _modvecs_from_flat6(tm_flat, D)
        var direct = _flux_direct_dora_double_for(dora, bi, targets) if algo == FLUX_DIRECT_ALGO_DORA else _flux_direct_oft_double_for(oft, bi, targets)
        var fwd = double_block_direct_lycoris_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), w, im, tm, direct, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        dbl_saved.append(fwd.saved.copy())
        dbl_img_mod.append(im_flat^)
        dbl_txt_mod.append(tm_flat^)
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)

    var x = _concat_seq(txt, img)

    var sgl_mod_flat = List[List[Float32]]()
    var sgl_saved = List[ChromaSingleBlockSaved]()
    for bi in range(num_single):
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w = _chroma_single_from_block(handle.block, handle.prefix + String("."), D, Fmlp, Dh, ctx)
        var sm_flat = _sgl_mod_flat(pooled, bi, D)
        var sm = _single_modvecs_from_flat3(sm_flat, D)
        var direct = _flux_direct_dora_single_for(dora, num_double, bi, targets) if algo == FLUX_DIRECT_ALGO_DORA else _flux_direct_oft_single_for(oft, num_double, bi, targets)
        var fwd = single_block_direct_lycoris_forward[H, Dh, S](
            x.copy(), w, sm, direct, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        sgl_saved.append(fwd.saved.copy())
        sgl_mod_flat.append(sm_flat^)
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)

    var parts = _split_seq(x, N_TXT, N_IMG, D)
    var img_out = parts[1].copy()

    var ss = _final_shift_scale(pooled, mod_index, D)
    var final_shift = ss[0].copy()
    var final_scale = ss[1].copy()

    var ln_img_out = layer_norm(
        _t(img_out.copy(), [N_IMG, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), eps, ctx,
    ).to_host(ctx)
    var normed = modulate(
        _t(ln_img_out.copy(), [N_IMG, D], ctx),
        _t(final_scale.copy(), [D], ctx), _t(final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var pb = Optional[Tensor](base.proj_out_b[].clone(ctx))
    var out = linear(
        _t_like(normed, [N_IMG, D], base.proj_out_w[], ctx),
        base.proj_out_w[], pb, ctx,
    ).to_host(ctx)

    return ChromaStackForward(
        out^, dbl_saved^, sgl_saved^,
        dbl_img_mod^, dbl_txt_mod^, sgl_mod_flat^,
        TArc(_t(img_out^, [N_IMG, D], ctx)), TArc(_t(ln_img_out^, [N_IMG, D], ctx)),
        final_shift^, final_scale^,
    )


def chroma_stack_direct_dora_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    pooled: List[Float32], mod_index: Int,
    base: ChromaStackBase,
    mut loader: TurboPlannedLoader, dora: FlatDirectDoRASet,
    num_double: Int, num_single: Int, targets: Int,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ChromaStackForward:
    var expected = _flux_direct_expected_slots(num_double, num_single, targets)
    if len(dora.ad) != expected:
        raise Error("chroma_stack_direct_dora_forward_offload: direct slot count mismatch")
    return _chroma_stack_direct_forward_offload[H, Dh, N_IMG, N_TXT, S](
        img_tokens, txt_tokens, pooled, mod_index, base, loader,
        dora, empty_flat_direct_oft_set(),
        FLUX_DIRECT_ALGO_DORA, num_double, num_single, targets, cos, sin,
        D, Fmlp, in_ch, txt_ch, out_ch, eps, ctx,
    )


def chroma_stack_direct_oft_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    pooled: List[Float32], mod_index: Int,
    base: ChromaStackBase,
    mut loader: TurboPlannedLoader, oft: FlatDirectOFTSet,
    num_double: Int, num_single: Int, targets: Int,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ChromaStackForward:
    var expected = _flux_direct_expected_slots(num_double, num_single, targets)
    if len(oft.ad) != expected:
        raise Error("chroma_stack_direct_oft_forward_offload: direct slot count mismatch")
    return _chroma_stack_direct_forward_offload[H, Dh, N_IMG, N_TXT, S](
        img_tokens, txt_tokens, pooled, mod_index, base, loader,
        empty_flat_direct_dora_set(),
        oft, FLUX_DIRECT_ALGO_OFT, num_double, num_single, targets, cos, sin,
        D, Fmlp, in_ch, txt_ch, out_ch, eps, ctx,
    )


# ═════════════════════════════════════════════════════════════════════════════
# FULL BACKWARD WITH LoRA, BLOCK-SWAP OFFLOAD (REVERSE block stream).
#   Frozen-approximator scope: per-block mod-vec grads are DISCARDED (they would
#   flow into the frozen distilled_guidance_layer). Only LoRA d_A/d_B collected.
# ═════════════════════════════════════════════════════════════════════════════
def chroma_stack_lora_backward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ChromaStackBase,
    mut loader: TurboPlannedLoader, lora: FluxLoraSet,
    cos: List[Float32], sin: List[Float32],
    saved: ChromaStackForward,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxLoraGradSet:
    var num_double = lora.num_double
    var num_single = lora.num_single

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    var n_adapters = total_adapters(lora)
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a_flat.append(List[Float32]()); d_b_flat.append(List[Float32]())
    var nonfinite = 0

    # ── final layer backward (proj_out -> modulate -> layer_norm) ──
    var normed = modulate(
        saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), _t(saved.final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var lbf = linear_backward(
        _t_like(d_out, [N_IMG, out_ch], base.proj_out_w[], ctx),
        _t_like(normed, [N_IMG, D], base.proj_out_w[], ctx), base.proj_out_w[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)
    var mbf = modulate_backward(
        _t(d_normed, [N_IMG, D], ctx), saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)
    var lnbf = layer_norm_backward(
        _t(d_ln_img_out, [N_IMG, D], ctx), saved.img_out[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_img_out = lnbf.d_x.to_host(ctx)

    var d_x = _concat_seq(_zeros(N_TXT * D), d_img_out)

    # ── single-stream backward (REVERSE; LoRA; streamed weights) ──
    var bi = num_single - 1
    while bi >= 0:
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)
        var w = _chroma_single_from_block(handle.block, handle.prefix + String("."), D, Fmlp, Dh, ctx)
        var sm = _single_modvecs_from_flat3(saved.sgl_mod_flat[bi].copy(), D)
        var bl = _single_lora_for(lora, bi)
        var bg = chroma_single_block_lora_backward[H, Dh, S](
            d_x.copy(), w, sm, bl, saved.sgl_saved[bi], cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_x = bg.base.d_x.copy()
        var sbase = _sgl_base(lora, bi)
        for s in range(SGL_SLOTS):
            d_a_flat[sbase + s] = bg.lora.d_a[s].copy()
            d_b_flat[sbase + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        # mod-vec grads (bg.base shift/scale/gate) DISCARDED (frozen approximator).
        loader.mark_active_block_done(ctx)
        bi -= 1

    var seam = _split_seq(d_x, N_TXT, N_IMG, D)
    var d_to = seam[0].copy()
    var d_io = seam[1].copy()

    # ── double-stream backward (REVERSE; LoRA; streamed weights) ──
    var di = num_double - 1
    while di >= 0:
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var w = _chroma_double_from_block(handle.block, handle.prefix + String("."), D, Fmlp, Dh, ctx)
        var im = _modvecs_from_flat6(saved.dbl_img_mod[di].copy(), D)
        var tm = _modvecs_from_flat6(saved.dbl_txt_mod[di].copy(), D)
        var bl = _double_lora_for(lora, di)
        var bg = chroma_double_block_lora_backward[H, Dh, N_IMG, N_TXT, S](
            d_io.copy(), d_to.copy(), w, im, tm, bl, saved.dbl_saved[di],
            cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_io = bg.base.img.d_x.copy()
        d_to = bg.base.txt.d_x.copy()
        var dbase = _dbl_base(di)
        for s in range(DBL_STREAM_SLOTS):
            d_a_flat[dbase + s] = bg.lora.img.d_a[s].copy()
            d_b_flat[dbase + s] = bg.lora.img.d_b[s].copy()
            d_a_flat[dbase + DBL_STREAM_SLOTS + s] = bg.lora.txt.d_a[s].copy()
            d_b_flat[dbase + DBL_STREAM_SLOTS + s] = bg.lora.txt.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.img.d_a[s]) + _nonfinite(bg.lora.img.d_b[s])
            nonfinite += _nonfinite(bg.lora.txt.d_a[s]) + _nonfinite(bg.lora.txt.d_b[s])
        # mod-vec grads DISCARDED (frozen approximator).
        loader.mark_active_block_done(ctx)
        di -= 1

    # input-projection backward (frozen base; grads discarded, arms exercised).
    var lbi = linear_backward(
        _t_like(d_io, [N_IMG, D], base.x_embedder_w[], ctx),
        _t_like(img_tokens, [N_IMG, in_ch], base.x_embedder_w[], ctx), base.x_embedder_w[],
        N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var lbt = linear_backward(
        _t_like(d_to, [N_TXT, D], base.context_embedder_w[], ctx),
        _t_like(txt_tokens, [N_TXT, txt_ch], base.context_embedder_w[], ctx), base.context_embedder_w[],
        N_TXT, txt_ch, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)

    return FluxLoraGradSet(
        d_a_flat^, d_b_flat^,
        d_img_tokens^, d_txt_tokens^, _zeros(D),
        _zeros(1), _zeros(1), _zeros(1),
        nonfinite,
    )


def chroma_stack_direct_dora_backward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ChromaStackBase,
    mut loader: TurboPlannedLoader, dora: FlatDirectDoRASet,
    num_double: Int, num_single: Int, targets: Int,
    cos: List[Float32], sin: List[Float32],
    saved: ChromaStackForward,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxDirectDoRAGradSet:
    var expected = _flux_direct_expected_slots(num_double, num_single, targets)
    if len(dora.ad) != expected:
        raise Error("chroma_stack_direct_dora_backward_offload: direct slot count mismatch")
    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var dora_grads = _flux_direct_dora_zero_grads(dora)
    var nonfinite = 0

    var normed = modulate(
        saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), _t(saved.final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var lbf = linear_backward(
        _t_like(d_out, [N_IMG, out_ch], base.proj_out_w[], ctx),
        _t_like(normed, [N_IMG, D], base.proj_out_w[], ctx), base.proj_out_w[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)
    var mbf = modulate_backward(
        _t(d_normed, [N_IMG, D], ctx), saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)
    var lnbf = layer_norm_backward(
        _t(d_ln_img_out, [N_IMG, D], ctx), saved.img_out[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_x = _concat_seq(_zeros(N_TXT * D), lnbf.d_x.to_host(ctx))

    var bi = num_single - 1
    while bi >= 0:
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)
        var w = _chroma_single_from_block(handle.block, handle.prefix + String("."), D, Fmlp, Dh, ctx)
        var sm = _single_modvecs_from_flat3(saved.sgl_mod_flat[bi].copy(), D)
        var direct = _flux_direct_dora_single_for(dora, num_double, bi, targets)
        var bg = single_block_direct_lycoris_backward[H, Dh, S](
            d_x.copy(), w, sm, direct, saved.sgl_saved[bi], cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_x = bg.d_x.copy()
        _scatter_flux_dora_single(dora_grads, targets, num_double, bi, bg)
        nonfinite += _nonfinite_flux_direct_single(bg)
        loader.mark_active_block_done(ctx)
        bi -= 1

    var seam = _split_seq(d_x, N_TXT, N_IMG, D)
    var d_to = seam[0].copy()
    var d_io = seam[1].copy()

    var di = num_double - 1
    while di >= 0:
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var w = _chroma_double_from_block(handle.block, handle.prefix + String("."), D, Fmlp, Dh, ctx)
        var im = _modvecs_from_flat6(saved.dbl_img_mod[di].copy(), D)
        var tm = _modvecs_from_flat6(saved.dbl_txt_mod[di].copy(), D)
        var direct = _flux_direct_dora_double_for(dora, di, targets)
        var bg = double_block_direct_lycoris_backward[H, Dh, N_IMG, N_TXT, S](
            d_io.copy(), d_to.copy(), w, im, tm, direct, saved.dbl_saved[di],
            cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_io = bg.img.d_x.copy()
        d_to = bg.txt.d_x.copy()
        _scatter_flux_dora_double(dora_grads, targets, di, bg)
        nonfinite += _nonfinite_flux_direct_double(bg)
        loader.mark_active_block_done(ctx)
        di -= 1

    var lbi = linear_backward(
        _t_like(d_io, [N_IMG, D], base.x_embedder_w[], ctx),
        _t_like(img_tokens, [N_IMG, in_ch], base.x_embedder_w[], ctx), base.x_embedder_w[],
        N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var lbt = linear_backward(
        _t_like(d_to, [N_TXT, D], base.context_embedder_w[], ctx),
        _t_like(txt_tokens, [N_TXT, txt_ch], base.context_embedder_w[], ctx), base.context_embedder_w[],
        N_TXT, txt_ch, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)

    return FluxDirectDoRAGradSet(
        dora_grads^, d_img_tokens^, d_txt_tokens^, _zeros(D),
        _zeros(1), _zeros(1), _zeros(1), nonfinite,
    )


def chroma_stack_direct_oft_backward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ChromaStackBase,
    mut loader: TurboPlannedLoader, oft: FlatDirectOFTSet,
    num_double: Int, num_single: Int, targets: Int,
    cos: List[Float32], sin: List[Float32],
    saved: ChromaStackForward,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxDirectOFTGradSet:
    var expected = _flux_direct_expected_slots(num_double, num_single, targets)
    if len(oft.ad) != expected:
        raise Error("chroma_stack_direct_oft_backward_offload: direct slot count mismatch")
    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var oft_grads = _flux_direct_oft_zero_grads(oft)
    var nonfinite = 0

    var normed = modulate(
        saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), _t(saved.final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var lbf = linear_backward(
        _t_like(d_out, [N_IMG, out_ch], base.proj_out_w[], ctx),
        _t_like(normed, [N_IMG, D], base.proj_out_w[], ctx), base.proj_out_w[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)
    var mbf = modulate_backward(
        _t(d_normed, [N_IMG, D], ctx), saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)
    var lnbf = layer_norm_backward(
        _t(d_ln_img_out, [N_IMG, D], ctx), saved.img_out[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_x = _concat_seq(_zeros(N_TXT * D), lnbf.d_x.to_host(ctx))

    var bi = num_single - 1
    while bi >= 0:
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)
        var w = _chroma_single_from_block(handle.block, handle.prefix + String("."), D, Fmlp, Dh, ctx)
        var sm = _single_modvecs_from_flat3(saved.sgl_mod_flat[bi].copy(), D)
        var direct = _flux_direct_oft_single_for(oft, num_double, bi, targets)
        var bg = single_block_direct_lycoris_backward[H, Dh, S](
            d_x.copy(), w, sm, direct, saved.sgl_saved[bi], cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_x = bg.d_x.copy()
        _scatter_flux_oft_single(oft_grads, targets, num_double, bi, bg)
        nonfinite += _nonfinite_flux_direct_single(bg)
        loader.mark_active_block_done(ctx)
        bi -= 1

    var seam = _split_seq(d_x, N_TXT, N_IMG, D)
    var d_to = seam[0].copy()
    var d_io = seam[1].copy()

    var di = num_double - 1
    while di >= 0:
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var w = _chroma_double_from_block(handle.block, handle.prefix + String("."), D, Fmlp, Dh, ctx)
        var im = _modvecs_from_flat6(saved.dbl_img_mod[di].copy(), D)
        var tm = _modvecs_from_flat6(saved.dbl_txt_mod[di].copy(), D)
        var direct = _flux_direct_oft_double_for(oft, di, targets)
        var bg = double_block_direct_lycoris_backward[H, Dh, N_IMG, N_TXT, S](
            d_io.copy(), d_to.copy(), w, im, tm, direct, saved.dbl_saved[di],
            cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_io = bg.img.d_x.copy()
        d_to = bg.txt.d_x.copy()
        _scatter_flux_oft_double(oft_grads, targets, di, bg)
        nonfinite += _nonfinite_flux_direct_double(bg)
        loader.mark_active_block_done(ctx)
        di -= 1

    var lbi = linear_backward(
        _t_like(d_io, [N_IMG, D], base.x_embedder_w[], ctx),
        _t_like(img_tokens, [N_IMG, in_ch], base.x_embedder_w[], ctx), base.x_embedder_w[],
        N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var lbt = linear_backward(
        _t_like(d_to, [N_TXT, D], base.context_embedder_w[], ctx),
        _t_like(txt_tokens, [N_TXT, txt_ch], base.context_embedder_w[], ctx), base.context_embedder_w[],
        N_TXT, txt_ch, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)

    return FluxDirectOFTGradSet(
        oft_grads^, d_img_tokens^, d_txt_tokens^, _zeros(D),
        _zeros(1), _zeros(1), _zeros(1), nonfinite,
    )
