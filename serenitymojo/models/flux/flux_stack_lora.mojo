# serenitymojo/models/flux/flux_stack_lora.mojo
#
# Flux (flux1-dev) FULL DiT STACK *WITH LoRA* on every trained block projection:
# forward (saving ckpt-inputs) + full-depth backward (training) that uses the
# parity-verified per-block LoRA variants (models/flux/lora_block.mojo), COLLECTS
# every adapter's d_A/d_B, and supports an AdamW step + a OneTrainer raw-key save
# across the currently implemented adapters. This file COMPOSES; it rebuilds NOTHING.
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/flux/block.mojo : base double/single block fwd+bwd (Phase-1 GREEN).
#   * models/flux/flux_stack.mojo : the BASE full-stack composition (Phase-2
#     GREEN). THIS FILE is that file with the base per-block calls swapped for the
#     LoRA variants + LoRA-grad collection. The embed/vec chain + per-block
#     modulation projections + final layer are reused VERBATIM (frozen base).
#   * models/flux/lora_block.mojo : double/single_block_lora_forward/backward
#     (reduce to base when adapters absent; LoRA d_x summed into the proj-input
#     grad). Slot order per the OneTrainer convert_flux_lora.py target set.
#   * training/{lora_save, train_step} : LoraAdapter, _lora_adamw,
#     save_lora_onetrainer.
#
# CARRIER DESIGN (Tenet-2: make the right thing easy) — mirrors KleinLoraSet /
#   ErnieLoraSet. FluxLoraSet holds ONE flat List[LoraAdapter]:
#     doubles first: block bi (0..num_double-1), 12 slots/block
#       flat = bi*DBL_SLOTS_PER_BLOCK + (stream*6 + slot)
#       stream 0=img 1=txt ; slot order {to_q,to_k,to_v,proj,mlp0,mlp2}
#     singles next: block bi (0..num_single-1), 5 slots/block
#       flat = num_double*DBL_SLOTS_PER_BLOCK + bi*SGL_SLOTS + slot
#       slot order {to_q,to_k,to_v,proj_mlp,linear2}
#   The optimizer walks this flat list; the backward SCATTERS the returned per-
#   block slot d_A/d_B into the matching flat FluxLoraGrads.
#
# SCOPE: LoRA-on-block-projection training. Base weights (input/text proj, the
#   embed MLPs, per-block modulation linears, the block linears, final layer) are
#   FROZEN — their grads are computed by the base path and discarded for the
#   optimizer; only block-projection d_A/d_B are trained. This matches Klein/Ernie
#   scope. The OneTrainer parity inventory below makes the not-yet-trained
#   stack-level and modulation Linear targets explicit and fail-loud.
#
# RESIDENCY: full F32 residency of 19+38 real-depth blocks is ~34GB (Phase-2
#   finding) and does NOT fit a 3090 — real-depth runs need block-swap offload +
#   per-block recompute (Ernie/Klein-9B strategy). The PARITY gate runs at REDUCED
#   depth (fits); composition is proven there. A streamed variant mirrors the
#   ernie_stack_lora_*_streamed path and is deferred to the runtime increment.
#
# Mojo 1.0.0b1: def not fn; Tensor move-only; host List[Float32] carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.models.flux.block import (
    ModVecs, SingleModVecs,
    StreamWeights,
    DoubleBlockWeights, SingleBlockWeights,
    DoubleBlockSaved, SingleBlockSaved, DoubleBlockGrads, SingleBlockGrads,
    StreamGrads,
)
from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.flux.lora_block import (
    StreamLora, DoubleBlockLora, SingleBlockLora,
    DoubleBlockLoraGrads, SingleBlockLoraGrads, StreamLoraGrads,
    DBL_STREAM_SLOTS, SGL_SLOTS,
    D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    S_SQ, S_SK, S_SV, S_PMLP, S_L2,
    double_block_lora_forward, double_block_lora_backward,
    single_block_lora_forward, single_block_lora_backward,
    flux_lora_apply, flux_lora_bwd, FluxLoraGrads,
)
from serenitymojo.models.flux.flux_stack import (
    FluxStackBase, FluxStackForward, FluxStackGrads, EmbedMlp, ModLin,
    _add_lists, _zeros, _ones, _t, _concat_seq, _split_seq, _chunk,
    _modvecs_from_flat, _single_modvecs_from_flat, _modvec6, _single_modvec3,
    _embed_vec_forward, _mod_proj, _embed_vec_backward,
)
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm_backward import layer_norm_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.activation_backward import silu_backward

from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_save import (
    NamedLora,
    save_lora_onetrainer, load_lora_for_resume,
    save_lora_train_state, load_lora_train_state,
)


comptime TArc = ArcPointer[Tensor]

# 6 slots per stream x 2 streams = 12 slots per double block.
comptime DBL_SLOTS_PER_BLOCK = 2 * DBL_STREAM_SLOTS


# ── adapter init (A kaiming-uniform, B=0 — PEFT identity at step 0) ───────────
# OneTrainer LoRAModule.initialize_weights:
#   nn.init.kaiming_uniform_(lora_down.weight, a=sqrt(5)); nn.init.zeros_(lora_up.weight)
# lora_down is nn.Linear(in_features, rank) so weight shape is [rank, in_features],
# fan_in = in_features. kaiming_uniform_ with a=sqrt(5) gives
#   gain  = sqrt(2 / (1 + a^2)) = sqrt(2/6) = sqrt(1/3)
#   bound = gain * sqrt(3 / fan_in) = sqrt(1/3) * sqrt(3) / sqrt(fan_in) = 1/sqrt(fan_in)
# i.e. A ~ U(-1/sqrt(in_f), +1/sqrt(in_f)). (B stays exactly 0 -> identity at step 0.)
def _kaiming_uniform_a(n: Int, seed: UInt64, fan_in: Int) -> List[Float32]:
    var bound = Float32(1.0) / sqrt(Float32(fan_in))   # kaiming_uniform_(a=sqrt5) bound
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)  # u in [0,1)
        out.append((u * Float32(2.0) - Float32(1.0)) * bound)         # -> U(-bound,+bound)
    return out^


def make_lora_adapter(
    rank: Int, alpha: Float32, in_f: Int, out_f: Int, seed: UInt64
) -> LoraAdapter:
    var scale = alpha / Float32(rank)
    return LoraAdapter(
        _kaiming_uniform_a(rank * in_f, seed, in_f),   # A ~ U(-1/sqrt(in_f), +1/sqrt(in_f))
        _zeros(out_f * rank),              # B = 0 (adapter identity at init)
        rank, in_f, out_f, scale,
        _zeros(rank * in_f), _zeros(rank * in_f),    # ma / va
        _zeros(out_f * rank), _zeros(out_f * rank),  # mb / vb
    )


# ── the LoRA carrier: every trained adapter, flat-indexed ─────────────────────
struct FluxLoraSet(Copyable, Movable):
    var ad: List[LoraAdapter]
    var num_double: Int
    var num_single: Int
    var rank: Int

    def __init__(out self, var ad: List[LoraAdapter], num_double: Int, num_single: Int, rank: Int):
        self.ad = ad^
        self.num_double = num_double
        self.num_single = num_single
        self.rank = rank


def _dbl_base(bi: Int) -> Int:
    return bi * DBL_SLOTS_PER_BLOCK


def _sgl_base(set: FluxLoraSet, bi: Int) -> Int:
    return set.num_double * DBL_SLOTS_PER_BLOCK + bi * SGL_SLOTS


# Build the full LoRA set for a Flux stack. Per-projection in/out shapes:
#   double to_q/to_k/to_v : in=D out=D ; proj : in=D out=D ;
#     mlp0 : in=D out=Fmlp ; mlp2 : in=Fmlp out=D.
#   single to_q/to_k/to_v : in=D out=D ; proj_mlp : in=D out=Fmlp ;
#     linear2 : in=D+Fmlp out=D.
def build_flux_lora_set(
    num_double: Int, num_single: Int, D: Int, Fmlp: Int, rank: Int, alpha: Float32
) -> FluxLoraSet:
    var ad = List[LoraAdapter]()
    var seed = UInt64(3000)
    for _ in range(num_double):
        for _stream in range(2):   # img then txt
            ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1     # to_q
            ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1     # to_k
            ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1     # to_v
            ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1     # proj
            ad.append(make_lora_adapter(rank, alpha, D, Fmlp, seed)); seed += 1  # mlp0
            ad.append(make_lora_adapter(rank, alpha, Fmlp, D, seed)); seed += 1  # mlp2
    for _ in range(num_single):
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1         # to_q
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1         # to_k
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1         # to_v
        ad.append(make_lora_adapter(rank, alpha, D, Fmlp, seed)); seed += 1      # proj_mlp
        ad.append(make_lora_adapter(rank, alpha, D + Fmlp, D, seed)); seed += 1  # linear2
    return FluxLoraSet(ad^, num_double, num_single, rank)


def total_adapters(set: FluxLoraSet) -> Int:
    return set.num_double * DBL_SLOTS_PER_BLOCK + set.num_single * SGL_SLOTS


# ═════════════════════════════════════════════════════════════════════════════
# STACK-LEVEL LoRA — the OneTrainer "#flux LoRA.json" default (empty layer_filter)
# trains a LoRA on EVERY transformer Linear, not only the per-block attn+ff
# projections above. The remaining targets (frozen in the block-projection-only
# path) are the per-block MODULATION linears + the transformer-level embedder /
# input-projection / final linears. They are ALL stack-level base linears that
# FluxStackBase already runs through linear()/linear_backward() with their d_x
# threaded; this carrier adds the LoRA delta (fwd) + grad (bwd) on each, EXACTLY
# like the block-projection adapters (same rank/alpha/scale/kaiming-A/B-zero
# convention via make_lora_adapter), with OT-matching saved key names.
#
# OT target inventory (convert_flux_lora.py:44-62, default empty filter):
#   per DOUBLE block:  norm1.linear (img_mod.lin [6D,D]),
#                      norm1_context.linear (txt_mod.lin [6D,D])
#   per SINGLE block:  norm.linear  (modulation.lin [3D,D])
#   transformer level: context_embedder (txt_in [D,txt_ch]),
#                      x_embedder       (img_in [D,in_ch]),
#                      time_text_embed.timestep_embedder.linear_1 ([D,T_DIM]),
#                                       .timestep_embedder.linear_2 ([D,D]),
#                      time_text_embed.text_embedder.linear_1 ([D,VEC_DIM]),
#                                       .text_embedder.linear_2 ([D,D]),
#                      time_text_embed.guidance_embedder.linear_1 ([D,T_DIM]),
#                                       .guidance_embedder.linear_2 ([D,D]),
#                      norm_out.linear  (final_adaln_w [2D,D]),
#                      proj_out         (final_lin [out_ch,D]).
#
# The 10 transformer-level adapters are held in a FLAT List[Optional[LoraAdapter]]
# in the canonical slot order below; the per-block modulation adapters are flat
# lists parallel to the double/single blocks. An EMPTY set (all-None or zero-len
# block lists) makes every flux_stack_lora_apply a no-op, so the block-projection
# path stays BIT-IDENTICAL when stack-level LoRA is not requested.
# ═════════════════════════════════════════════════════════════════════════════

# transformer-level slot order (canonical; matches _stack_ot_prefixes below).
comptime ST_CTX_EMB = 0    # context_embedder (txt_in)   in=txt_ch out=D
comptime ST_X_EMB = 1      # x_embedder       (img_in)   in=in_ch  out=D
comptime ST_TIME_1 = 2     # timestep_embedder.linear_1  in=T_DIM  out=D
comptime ST_TIME_2 = 3     # timestep_embedder.linear_2  in=D      out=D
comptime ST_TEXT_1 = 4     # text_embedder.linear_1      in=VEC_DIM out=D
comptime ST_TEXT_2 = 5     # text_embedder.linear_2      in=D      out=D
comptime ST_GUID_1 = 6     # guidance_embedder.linear_1  in=T_DIM  out=D
comptime ST_GUID_2 = 7     # guidance_embedder.linear_2  in=D      out=D
comptime ST_NORM_OUT = 8   # norm_out.linear (final_adaln) in=D    out=2D
comptime ST_PROJ_OUT = 9   # proj_out (final_lin)         in=D     out=out_ch
comptime ST_LEVEL_SLOTS = 10


struct FluxStackLoraSet(Copyable, Movable):
    # transformer-level adapters (flat, ST_* slot order; None if absent).
    var level: List[Optional[LoraAdapter]]
    # per-double-block modulation linears (norm1.linear / norm1_context.linear).
    var dbl_img_mod: List[Optional[LoraAdapter]]   # len = num_double
    var dbl_txt_mod: List[Optional[LoraAdapter]]   # len = num_double
    # per-single-block modulation linear (norm.linear).
    var sgl_mod: List[Optional[LoraAdapter]]       # len = num_single
    var num_double: Int
    var num_single: Int
    var rank: Int
    var enabled: Bool

    def __init__(
        out self,
        var level: List[Optional[LoraAdapter]],
        var dbl_img_mod: List[Optional[LoraAdapter]],
        var dbl_txt_mod: List[Optional[LoraAdapter]],
        var sgl_mod: List[Optional[LoraAdapter]],
        num_double: Int, num_single: Int, rank: Int, enabled: Bool,
    ):
        self.level = level^
        self.dbl_img_mod = dbl_img_mod^
        self.dbl_txt_mod = dbl_txt_mod^
        self.sgl_mod = sgl_mod^
        self.num_double = num_double
        self.num_single = num_single
        self.rank = rank
        self.enabled = enabled


# ── stack-level grad scatter buffers (filled during backward, assembled last) ─
# Parallel to FluxStackLoraSet: per-group d_a/d_b lists (empty entry == absent).
struct _StackGradBuf(Movable):
    var lvl_d_a: List[List[Float32]]   # ST_LEVEL_SLOTS entries
    var lvl_d_b: List[List[Float32]]
    var dimg_d_a: List[List[Float32]]  # num_double
    var dimg_d_b: List[List[Float32]]
    var dtxt_d_a: List[List[Float32]]
    var dtxt_d_b: List[List[Float32]]
    var sgl_d_a: List[List[Float32]]   # num_single
    var sgl_d_b: List[List[Float32]]

    def __init__(out self, num_double: Int, num_single: Int):
        self.lvl_d_a = List[List[Float32]]()
        self.lvl_d_b = List[List[Float32]]()
        for _ in range(ST_LEVEL_SLOTS):
            self.lvl_d_a.append(List[Float32]()); self.lvl_d_b.append(List[Float32]())
        self.dimg_d_a = List[List[Float32]]()
        self.dimg_d_b = List[List[Float32]]()
        self.dtxt_d_a = List[List[Float32]]()
        self.dtxt_d_b = List[List[Float32]]()
        for _ in range(num_double):
            self.dimg_d_a.append(List[Float32]()); self.dimg_d_b.append(List[Float32]())
            self.dtxt_d_a.append(List[Float32]()); self.dtxt_d_b.append(List[Float32]())
        self.sgl_d_a = List[List[Float32]]()
        self.sgl_d_b = List[List[Float32]]()
        for _ in range(num_single):
            self.sgl_d_a.append(List[Float32]()); self.sgl_d_b.append(List[Float32]())


# Assemble the scatter buffers into the FLAT (d_a, d_b) the optimizer + saver
# walk: populated adapters ONLY, in the SAME order as the AdamW walk and
# _flux_stack_named_loras (level ST_* slots, then per-double img+txt mod, then
# per-single mod). Writes into the caller's `d_a`/`d_b` (mut out-params, assumed
# empty) and RETURNS the nonfinite count. No struct return — the caller's lists
# are plain locals moved ONCE into FluxLoraGradSet (no partial-move hazard).
def _assemble_stack_grads_into(
    sset: FluxStackLoraSet, buf: _StackGradBuf,
    mut d_a: List[List[Float32]], mut d_b: List[List[Float32]],
) -> Int:
    var nonfinite = 0
    if not sset.enabled:
        return 0
    for slot in range(ST_LEVEL_SLOTS):
        if sset.level[slot]:
            d_a.append(buf.lvl_d_a[slot].copy())
            d_b.append(buf.lvl_d_b[slot].copy())
            nonfinite += _nonfinite(buf.lvl_d_a[slot]) + _nonfinite(buf.lvl_d_b[slot])
    for bi in range(sset.num_double):
        if sset.dbl_img_mod[bi]:
            d_a.append(buf.dimg_d_a[bi].copy())
            d_b.append(buf.dimg_d_b[bi].copy())
            nonfinite += _nonfinite(buf.dimg_d_a[bi]) + _nonfinite(buf.dimg_d_b[bi])
        if sset.dbl_txt_mod[bi]:
            d_a.append(buf.dtxt_d_a[bi].copy())
            d_b.append(buf.dtxt_d_b[bi].copy())
            nonfinite += _nonfinite(buf.dtxt_d_a[bi]) + _nonfinite(buf.dtxt_d_b[bi])
    for bi in range(sset.num_single):
        if sset.sgl_mod[bi]:
            d_a.append(buf.sgl_d_a[bi].copy())
            d_b.append(buf.sgl_d_b[bi].copy())
            nonfinite += _nonfinite(buf.sgl_d_a[bi]) + _nonfinite(buf.sgl_d_b[bi])
    return nonfinite


# An empty set: all-None level slots + empty block lists. Every apply is a no-op,
# so passing this preserves the block-projection path BIT-FOR-BIT.
def empty_flux_stack_lora_set(num_double: Int, num_single: Int, rank: Int) -> FluxStackLoraSet:
    var level = List[Optional[LoraAdapter]]()
    for _ in range(ST_LEVEL_SLOTS):
        level.append(Optional[LoraAdapter](None))
    var dbl_img = List[Optional[LoraAdapter]]()
    var dbl_txt = List[Optional[LoraAdapter]]()
    var sgl = List[Optional[LoraAdapter]]()
    return FluxStackLoraSet(level^, dbl_img^, dbl_txt^, sgl^, num_double, num_single, rank, False)


# Build the FULL stack-level LoRA set (all 13 OT target groups). Seeds continue
# from the block-projection set's seed space (build_flux_lora_set ends below
# 3000+total) so the two sets never share an A-init seed.
def build_flux_stack_lora_set(
    num_double: Int, num_single: Int,
    D: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, has_guidance: Bool,
    rank: Int, alpha: Float32,
) -> FluxStackLoraSet:
    var seed = UInt64(900000)   # disjoint from build_flux_lora_set's 3000-range
    var level = List[Optional[LoraAdapter]]()
    # ST_* order; in/out shapes match the base linears (FluxStackBase).
    level.append(_opt(make_lora_adapter(rank, alpha, txt_ch, D, seed))); seed += 1   # context_embedder
    level.append(_opt(make_lora_adapter(rank, alpha, in_ch, D, seed))); seed += 1    # x_embedder
    level.append(_opt(make_lora_adapter(rank, alpha, T_DIM, D, seed))); seed += 1    # timestep linear_1
    level.append(_opt(make_lora_adapter(rank, alpha, D, D, seed))); seed += 1        # timestep linear_2
    level.append(_opt(make_lora_adapter(rank, alpha, VEC_DIM, D, seed))); seed += 1  # text linear_1
    level.append(_opt(make_lora_adapter(rank, alpha, D, D, seed))); seed += 1        # text linear_2
    # guidance embedder: present in OT only when the model has a guidance_in MLP.
    if has_guidance:
        level.append(_opt(make_lora_adapter(rank, alpha, T_DIM, D, seed))); seed += 1  # guidance linear_1
        level.append(_opt(make_lora_adapter(rank, alpha, D, D, seed))); seed += 1      # guidance linear_2
    else:
        level.append(Optional[LoraAdapter](None))
        level.append(Optional[LoraAdapter](None))
    level.append(_opt(make_lora_adapter(rank, alpha, D, 2 * D, seed))); seed += 1    # norm_out.linear
    level.append(_opt(make_lora_adapter(rank, alpha, D, out_ch, seed))); seed += 1   # proj_out

    var dbl_img = List[Optional[LoraAdapter]]()
    var dbl_txt = List[Optional[LoraAdapter]]()
    for _ in range(num_double):
        dbl_img.append(_opt(make_lora_adapter(rank, alpha, D, 6 * D, seed))); seed += 1
        dbl_txt.append(_opt(make_lora_adapter(rank, alpha, D, 6 * D, seed))); seed += 1
    var sgl = List[Optional[LoraAdapter]]()
    for _ in range(num_single):
        sgl.append(_opt(make_lora_adapter(rank, alpha, D, 3 * D, seed))); seed += 1

    return FluxStackLoraSet(level^, dbl_img^, dbl_txt^, sgl^, num_double, num_single, rank, True)


# Number of populated (non-None) stack-level adapters in `sset`.
def total_stack_adapters(sset: FluxStackLoraSet) -> Int:
    if not sset.enabled:
        return 0
    var n = 0
    for i in range(len(sset.level)):
        if sset.level[i]:
            n += 1
    for i in range(len(sset.dbl_img_mod)):
        if sset.dbl_img_mod[i]:
            n += 1
    for i in range(len(sset.dbl_txt_mod)):
        if sset.dbl_txt_mod[i]:
            n += 1
    for i in range(len(sset.sgl_mod)):
        if sset.sgl_mod[i]:
            n += 1
    return n


# accessor for a level slot -> Optional copy (None if absent or set disabled).
def _level_lo(sset: FluxStackLoraSet, slot: Int) -> Optional[LoraAdapter]:
    if not sset.enabled:
        return Optional[LoraAdapter](None)
    return sset.level[slot].copy()


# per-double-block modulation adapter (img_mod=norm1.linear / txt_mod=norm1_context.linear).
def _dbl_mod_lo(sset: FluxStackLoraSet, bi: Int, stream_img: Bool) -> Optional[LoraAdapter]:
    if not sset.enabled:
        return Optional[LoraAdapter](None)
    if stream_img:
        return sset.dbl_img_mod[bi].copy()
    return sset.dbl_txt_mod[bi].copy()


# per-single-block modulation adapter (norm.linear).
def _sgl_mod_lo(sset: FluxStackLoraSet, bi: Int) -> Optional[LoraAdapter]:
    if not sset.enabled:
        return Optional[LoraAdapter](None)
    return sset.sgl_mod[bi].copy()


# ── LoRA-aware base-linear forward delta (host) ──────────────────────────────
# base_y is the FROZEN base linear output; add scale*((x@Aᵀ)@Bᵀ) if present.
# Identical contract to lora_block.flux_lora_apply (REUSED here) — kept as a
# thin alias so the stack-level call sites read clearly.
def _stack_lora_apply(
    base_y: List[Float32], x_h: List[Float32], lo: Optional[LoraAdapter],
    M: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    return flux_lora_apply(base_y, x_h, lo, M, ctx)


# silu(vec) -> mod.lin -> [chunk] flat, WITH a LoRA delta on mod.lin (if present).
# Mirrors flux_stack._mod_proj; input to the LoRA is the SAME vec_silu [1,D].
def _mod_proj_lora(
    vec_silu: List[Float32], ml: ModLin, lo: Optional[LoraAdapter],
    chunk: Int, D: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var base_y = _mod_proj(vec_silu.copy(), ml, chunk, D, ctx)   # [1,chunk]
    return _stack_lora_apply(base_y, vec_silu.copy(), lo, 1, ctx)


# ── LoRA-aware MLPEmbedder forward (linear_1 -> silu -> linear_2) ─────────────
# Mirrors flux_stack._mlp_embed_fwd but adds LoRA on BOTH linears. Returns
#   [0] = mlp output [1,D]   [1] = in_layer hidden (silu input, LoRA-modified)
# The hidden is LoRA-modified because lin1's LoRA delta changes the silu input.
def _mlp_embed_fwd_lora(
    emb_in: List[Float32], mlp: EmbedMlp, lo1: Optional[LoraAdapter], lo2: Optional[LoraAdapter],
    in_dim: Int, D: Int, ctx: DeviceContext,
) raises -> List[List[Float32]]:
    var x = _t(emb_in.copy(), [1, in_dim], ctx)
    var b_in = Optional[Tensor](mlp.in_b[].clone(ctx))
    var hid_base = linear(x, mlp.in_w[], b_in, ctx).to_host(ctx)         # [1,D] base
    var hid_h = _stack_lora_apply(hid_base, emb_in.copy(), lo1, 1, ctx)  # +LoRA(lin1)
    var act = silu(_t(hid_h.copy(), [1, D], ctx), ctx).to_host(ctx)
    var b_out = Optional[Tensor](mlp.out_b[].clone(ctx))
    var out_base = linear(_t(act.copy(), [1, D], ctx), mlp.out_w[], b_out, ctx).to_host(ctx)  # base
    var out_h = _stack_lora_apply(out_base, act.copy(), lo2, 1, ctx)     # +LoRA(lin2)
    var o = List[List[Float32]]()
    o.append(out_h^)     # [0] = mlp output (LoRA-modified)
    o.append(hid_h^)     # [1] = in_layer hidden (silu input, LoRA-modified)
    return o^


# LoRA-aware embed-vec forward: vec = time + guidance + vector, each MLPEmbedder
# carrying its OWN linear_1/linear_2 LoRA. Mirrors flux_stack._embed_vec_forward.
# Returns a _VecFwdL with the SAME fields _embed_vec_forward exposes (so the
# saved tape stores the LoRA-modified t_hid/g_hid/v_hid for backward).
struct _VecFwdL(Copyable, Movable):
    var vec: List[Float32]
    var t_emb: List[Float32]
    var t_hid: List[Float32]
    var g_emb: List[Float32]
    var g_hid: List[Float32]
    var v_hid: List[Float32]

    def __init__(
        out self, var vec: List[Float32], var t_emb: List[Float32], var t_hid: List[Float32],
        var g_emb: List[Float32], var g_hid: List[Float32], var v_hid: List[Float32],
    ):
        self.vec = vec^
        self.t_emb = t_emb^
        self.t_hid = t_hid^
        self.g_emb = g_emb^
        self.g_hid = g_hid^
        self.v_hid = v_hid^


def _embed_vec_forward_lora(
    timestep: List[Float32], guidance: Optional[List[Float32]], vector: List[Float32],
    base: FluxStackBase, sset: FluxStackLoraSet,
    D: Int, T_DIM: Int, VEC_DIM: Int, ctx: DeviceContext,
) raises -> _VecFwdL:
    var t_emb = timestep_embedding(
        Tensor.from_host(timestep.copy(), [1], STDtype.F32, ctx), T_DIM, ctx,
        Float32(10000.0), STDtype.F32,
    ).to_host(ctx)
    var tr = _mlp_embed_fwd_lora(
        t_emb.copy(), base.time_in, _level_lo(sset, ST_TIME_1), _level_lo(sset, ST_TIME_2),
        T_DIM, D, ctx,
    )
    var vec = tr[0].copy()
    var t_hid = tr[1].copy()

    var g_emb = _zeros(T_DIM)
    var g_hid = _zeros(D)
    if base.has_guidance and guidance:
        g_emb = timestep_embedding(
            Tensor.from_host(guidance.value().copy(), [1], STDtype.F32, ctx), T_DIM, ctx,
            Float32(10000.0), STDtype.F32,
        ).to_host(ctx)
        var gr = _mlp_embed_fwd_lora(
            g_emb.copy(), base.guidance_in, _level_lo(sset, ST_GUID_1), _level_lo(sset, ST_GUID_2),
            T_DIM, D, ctx,
        )
        vec = _add_lists(vec, gr[0])
        g_hid = gr[1].copy()

    var vr = _mlp_embed_fwd_lora(
        vector.copy(), base.vector_in, _level_lo(sset, ST_TEXT_1), _level_lo(sset, ST_TEXT_2),
        VEC_DIM, D, ctx,
    )
    vec = _add_lists(vec, vr[0])
    var v_hid = vr[1].copy()

    return _VecFwdL(vec^, t_emb^, t_hid^, g_emb^, g_hid^, v_hid^)


# ── LoRA-aware MLPEmbedder backward ──────────────────────────────────────────
# Returns d wrt the embedder input [1,in_dim] PLUS the lin1/lin2 LoRA d_a/d_b
# (empty lists when the slot is absent). Mirrors flux_stack._mlp_embed_backward,
# adding the two LoRA branches. `hid` is the LoRA-MODIFIED silu input (saved by
# the LoRA forward); `emb_in` is the embedder input (the real CLIP pooled / the
# saved sinusoid) — REQUIRED non-zero for the lin1 LoRA d_a.
struct _MlpEmbedBackL(Movable):
    var d_in: List[Float32]
    var d_a1: List[Float32]
    var d_b1: List[Float32]
    var d_a2: List[Float32]
    var d_b2: List[Float32]

    def __init__(
        out self, var d_in: List[Float32],
        var d_a1: List[Float32], var d_b1: List[Float32],
        var d_a2: List[Float32], var d_b2: List[Float32],
    ):
        self.d_in = d_in^
        self.d_a1 = d_a1^
        self.d_b1 = d_b1^
        self.d_a2 = d_a2^
        self.d_b2 = d_b2^


def _mlp_embed_backward_lora(
    d_out_mlp: List[Float32], emb_in: List[Float32], hid: List[Float32],
    mlp: EmbedMlp, lo1: Optional[LoraAdapter], lo2: Optional[LoraAdapter],
    in_dim: Int, D: Int, ctx: DeviceContext,
) raises -> _MlpEmbedBackL:
    # out = base_lin2(act) [+ LoRA2(act)] ; act = silu(hid).
    var act = silu(_t(hid.copy(), [1, D], ctx), ctx).to_host(ctx)
    var lb_out = linear_backward(
        _t(d_out_mlp.copy(), [1, D], ctx), _t(act.copy(), [1, D], ctx), mlp.out_w[], 1, D, D, ctx,
    )
    var d_act = lb_out.d_x.to_host(ctx)
    var d_a2 = List[Float32]()
    var d_b2 = List[Float32]()
    if lo2:
        var lg = flux_lora_bwd(d_out_mlp.copy(), act.copy(), lo2.value(), 1, ctx)
        d_a2 = lg.d_a.copy()
        d_b2 = lg.d_b.copy()
        d_act = _add_lists(d_act, lg.d_x)
    # act = silu(hid)
    var d_hid = silu_backward(_t(d_act, [1, D], ctx), _t(hid.copy(), [1, D], ctx), ctx).to_host(ctx)
    # hid = base_lin1(emb_in) [+ LoRA1(emb_in)]
    var lb_in = linear_backward(
        _t(d_hid.copy(), [1, D], ctx), _t(emb_in.copy(), [1, in_dim], ctx), mlp.in_w[], 1, in_dim, D, ctx,
    )
    var d_in = lb_in.d_x.to_host(ctx)
    var d_a1 = List[Float32]()
    var d_b1 = List[Float32]()
    if lo1:
        var lg = flux_lora_bwd(d_hid.copy(), emb_in.copy(), lo1.value(), 1, ctx)
        d_a1 = lg.d_a.copy()
        d_b1 = lg.d_b.copy()
        d_in = _add_lists(d_in, lg.d_x)
    return _MlpEmbedBackL(d_in^, d_a1^, d_b1^, d_a2^, d_b2^)


# ── LoRA-aware embed-vec backward ────────────────────────────────────────────
# Mirrors flux_stack._embed_vec_backward but also collects the 6 embedder LoRA
# grads (timestep/text/guidance lin1/lin2). `vector` is the REAL CLIP-pooled
# input (REQUIRED for the text_embedder lin1 d_a — the base path could pass a
# zero placeholder because it discards d_w, but the LoRA d_a needs the true
# input). Returns d_timestep/d_guidance/d_vector PLUS the 6 LoRA grad pairs in
# ST_* order: [TIME_1, TIME_2, TEXT_1, TEXT_2, GUID_1, GUID_2].
struct _EmbedBackL(Movable):
    var d_timestep: List[Float32]
    var d_guidance: List[Float32]
    var d_vector: List[Float32]
    var lo_d_a: List[List[Float32]]   # 6 entries (ST_* embedder order)
    var lo_d_b: List[List[Float32]]

    def __init__(
        out self, var d_timestep: List[Float32], var d_guidance: List[Float32],
        var d_vector: List[Float32],
        var lo_d_a: List[List[Float32]], var lo_d_b: List[List[Float32]],
    ):
        self.d_timestep = d_timestep^
        self.d_guidance = d_guidance^
        self.d_vector = d_vector^
        self.lo_d_a = lo_d_a^
        self.lo_d_b = lo_d_b^


# sinusoid-scalar grad helper (same analytic jacobian as flux_stack._sinusoid_t_grad).
def _sinusoid_t_grad_local(d_emb: List[Float32], emb: List[Float32], T_DIM: Int, max_period: Float32) -> Float32:
    from std.math import log as flog, exp as fexp
    var half = T_DIM // 2
    var neg_ln = -flog(max_period)
    var acc = Float32(0.0)
    for i in range(half):
        var f = fexp(neg_ln * (Float32(i) / Float32(half)))
        acc += d_emb[i] * (-f) * emb[half + i]
        acc += d_emb[half + i] * (f) * emb[i]
    return acc


def _embed_vec_backward_lora(
    d_vec: List[Float32], saved: FluxStackForward, base: FluxStackBase, sset: FluxStackLoraSet,
    vector: List[Float32],
    has_guidance: Bool, D: Int, T_DIM: Int, VEC_DIM: Int, max_period: Float32, ctx: DeviceContext,
) raises -> _EmbedBackL:
    var lo_d_a = List[List[Float32]]()
    var lo_d_b = List[List[Float32]]()
    for _ in range(6):
        lo_d_a.append(List[Float32]()); lo_d_b.append(List[Float32]())

    # time branch (lin1/lin2 = embedder slots 0/1 here -> ST_TIME_1/2).
    var tb = _mlp_embed_backward_lora(
        d_vec.copy(), saved.t_emb, saved.t_hid, base.time_in,
        _level_lo(sset, ST_TIME_1), _level_lo(sset, ST_TIME_2), T_DIM, D, ctx,
    )
    lo_d_a[0] = tb.d_a1.copy(); lo_d_b[0] = tb.d_b1.copy()
    lo_d_a[1] = tb.d_a2.copy(); lo_d_b[1] = tb.d_b2.copy()
    var d_timestep = List[Float32]()
    d_timestep.append(_sinusoid_t_grad_local(tb.d_in, saved.t_emb, T_DIM, max_period))

    # text branch (vector_in; ST_TEXT_1/2). Pass the REAL CLIP input for lin1 d_a.
    var vb = _mlp_embed_backward_lora(
        d_vec.copy(), vector.copy(), saved.v_hid, base.vector_in,
        _level_lo(sset, ST_TEXT_1), _level_lo(sset, ST_TEXT_2), VEC_DIM, D, ctx,
    )
    lo_d_a[2] = vb.d_a1.copy(); lo_d_b[2] = vb.d_b1.copy()
    lo_d_a[3] = vb.d_a2.copy(); lo_d_b[3] = vb.d_b2.copy()
    var d_vector = vb.d_in.copy()

    # guidance branch (ST_GUID_1/2).
    var d_guidance = List[Float32]()
    d_guidance.append(0.0)
    if has_guidance:
        var gb = _mlp_embed_backward_lora(
            d_vec.copy(), saved.g_emb, saved.g_hid, base.guidance_in,
            _level_lo(sset, ST_GUID_1), _level_lo(sset, ST_GUID_2), T_DIM, D, ctx,
        )
        lo_d_a[4] = gb.d_a1.copy(); lo_d_b[4] = gb.d_b1.copy()
        lo_d_a[5] = gb.d_a2.copy(); lo_d_b[5] = gb.d_b2.copy()
        d_guidance[0] = _sinusoid_t_grad_local(gb.d_in, saved.g_emb, T_DIM, max_period)

    return _EmbedBackL(d_timestep^, d_guidance^, d_vector^, lo_d_a^, lo_d_b^)


# accessor by flat index -> a COPY.
def flux_lora_get(set: FluxLoraSet, idx: Int) -> LoraAdapter:
    return set.ad[idx].copy()


def _opt(ad: LoraAdapter) -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](ad.copy())


def _stream_lora_for(set: FluxLoraSet, base: Int) -> StreamLora:
    return StreamLora(
        _opt(set.ad[base + D_SQ]), _opt(set.ad[base + D_SK]), _opt(set.ad[base + D_SV]),
        _opt(set.ad[base + D_PROJ]), _opt(set.ad[base + D_MLP0]), _opt(set.ad[base + D_MLP2]),
    )


def _double_lora_for(set: FluxLoraSet, bi: Int) -> DoubleBlockLora:
    var base = _dbl_base(bi)
    var img = _stream_lora_for(set, base)
    var txt = _stream_lora_for(set, base + DBL_STREAM_SLOTS)
    return DoubleBlockLora(img^, txt^)


def _single_lora_for(set: FluxLoraSet, bi: Int) -> SingleBlockLora:
    var base = _sgl_base(set, bi)
    return SingleBlockLora(
        _opt(set.ad[base + S_SQ]), _opt(set.ad[base + S_SK]), _opt(set.ad[base + S_SV]),
        _opt(set.ad[base + S_PMLP]), _opt(set.ad[base + S_L2]),
    )


# ── collected LoRA grads (flat, parallel to FluxLoraSet) ─────────────────────
struct FluxLoraGradSet(Movable):
    var d_a: List[List[Float32]]
    var d_b: List[List[Float32]]
    # load-bearing arms (prove the chain composes through the stack).
    var d_img_tokens: List[Float32]
    var d_txt_tokens: List[Float32]
    var d_vec: List[Float32]
    var d_timestep: List[Float32]
    var d_guidance: List[Float32]
    var d_vector: List[Float32]
    var nonfinite_lora_grads: Int
    # STACK-LEVEL LoRA grads (parallel to FluxStackLoraSet's populated adapters,
    # in scatter order: level slots (ST_*), then per-double img/txt mod, then per-
    # single mod — only POPULATED adapters get an entry, matching the AdamW walk
    # in flux_stack_lora_adamw_step). Empty when stack-level LoRA is disabled, so
    # the block-projection trainer path is unaffected.
    var st_d_a: List[List[Float32]]
    var st_d_b: List[List[Float32]]

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        var d_vec: List[Float32],
        var d_timestep: List[Float32], var d_guidance: List[Float32], var d_vector: List[Float32],
        nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.d_vec = d_vec^
        self.d_timestep = d_timestep^
        self.d_guidance = d_guidance^
        self.d_vector = d_vector^
        self.nonfinite_lora_grads = nonfinite_lora_grads
        self.st_d_a = List[List[Float32]]()
        self.st_d_b = List[List[Float32]]()

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        var d_vec: List[Float32],
        var d_timestep: List[Float32], var d_guidance: List[Float32], var d_vector: List[Float32],
        nonfinite_lora_grads: Int,
        var st_d_a: List[List[Float32]], var st_d_b: List[List[Float32]],
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.d_vec = d_vec^
        self.d_timestep = d_timestep^
        self.d_guidance = d_guidance^
        self.d_vector = d_vector^
        self.nonfinite_lora_grads = nonfinite_lora_grads
        self.st_d_a = st_d_a^
        self.st_d_b = st_d_b^


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# RESIDENT LoRA stack (reduced depth, for the COMPOSITION parity gate). Mirrors
# flux_stack_forward/backward exactly, swapping per-block calls for LoRA ones.
# ─────────────────────────────────────────────────────────────────────────────
# Thin wrapper: block-projection ONLY (stack-level LoRA disabled). Preserves the
# EXACT pre-existing forward — passes an empty FluxStackLoraSet so every stack-
# level apply is a no-op and the original code path runs unchanged.
def flux_stack_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    timestep: List[Float32], guidance: Optional[List[Float32]], vector: List[Float32],
    base: FluxStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights], lora: FluxLoraSet,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxStackForward:
    var sset = empty_flux_stack_lora_set(len(dbw), len(sbw), lora.rank)
    return flux_stack_lora_forward_full[H, Dh, N_IMG, N_TXT, S](
        img_tokens, txt_tokens, timestep, guidance, vector, base,
        dbw, sbw, lora, sset, cos, sin,
        D, Fmlp, in_ch, txt_ch, out_ch, T_DIM, VEC_DIM, eps, ctx,
    )


# FULL forward: block-projection LoRA (`lora`) PLUS stack-level LoRA (`sset`) on
# the embedders / input projections / per-block modulation linears / final layer
# — the complete OneTrainer default surface. When `sset.enabled` is False this is
# numerically identical to the block-projection-only path (it takes the original
# branch at every stack-level site).
def flux_stack_lora_forward_full[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    timestep: List[Float32], guidance: Optional[List[Float32]], vector: List[Float32],
    base: FluxStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights], lora: FluxLoraSet,
    sset: FluxStackLoraSet,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxStackForward:
    var num_double = len(dbw)
    var num_single = len(sbw)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    # embed-vec: LoRA-aware path (6 embedder linears) when enabled, else original.
    var vec = List[Float32]()
    var t_emb = List[Float32]()
    var t_hid = List[Float32]()
    var g_emb = List[Float32]()
    var g_hid = List[Float32]()
    var v_hid = List[Float32]()
    if sset.enabled:
        var vf = _embed_vec_forward_lora(timestep, guidance, vector, base, sset, D, T_DIM, VEC_DIM, ctx)
        vec = vf.vec.copy(); t_emb = vf.t_emb.copy(); t_hid = vf.t_hid.copy()
        g_emb = vf.g_emb.copy(); g_hid = vf.g_hid.copy(); v_hid = vf.v_hid.copy()
    else:
        var vf = _embed_vec_forward(timestep, guidance, vector, base, D, T_DIM, VEC_DIM, ctx)
        vec = vf.vec.copy(); t_emb = vf.t_emb.copy(); t_hid = vf.t_hid.copy()
        g_emb = vf.g_emb.copy(); g_hid = vf.g_hid.copy(); v_hid = vf.v_hid.copy()
    var vec_silu = silu(_t(vec.copy(), [1, D], ctx), ctx).to_host(ctx)

    # input projections (+ LoRA on x_embedder / context_embedder when enabled).
    var bi_img = Optional[Tensor](base.img_in_b[].clone(ctx))
    var img_base = linear(_t(img_tokens.copy(), [N_IMG, in_ch], ctx), base.img_in[], bi_img, ctx).to_host(ctx)
    var img = _stack_lora_apply(img_base, img_tokens.copy(), _level_lo(sset, ST_X_EMB), N_IMG, ctx)
    var bi_txt = Optional[Tensor](base.txt_in_b[].clone(ctx))
    var txt_base = linear(_t(txt_tokens.copy(), [N_TXT, txt_ch], ctx), base.txt_in[], bi_txt, ctx).to_host(ctx)
    var txt = _stack_lora_apply(txt_base, txt_tokens.copy(), _level_lo(sset, ST_CTX_EMB), N_TXT, ctx)

    var dbl_img_mod = List[List[Float32]]()
    var dbl_txt_mod = List[List[Float32]]()
    var dbl_saved = List[DoubleBlockSaved]()
    for bi in range(num_double):
        var im_flat = _mod_proj_lora(
            vec_silu.copy(), base.dbl_mod[bi].img, _dbl_mod_lo(sset, bi, True), 6 * D, D, ctx
        )
        var tm_flat = _mod_proj_lora(
            vec_silu.copy(), base.dbl_mod[bi].txt, _dbl_mod_lo(sset, bi, False), 6 * D, D, ctx
        )
        var im = _modvecs_from_flat(im_flat, D)
        var tm = _modvecs_from_flat(tm_flat, D)
        var bl = _double_lora_for(lora, bi)
        var fwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), dbw[bi], im, tm, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        dbl_saved.append(fwd.saved.copy())
        dbl_img_mod.append(im_flat^)
        dbl_txt_mod.append(tm_flat^)
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()

    var x = _concat_seq(txt, img)

    var sgl_mod_flat = List[List[Float32]]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        var sm_flat = _mod_proj_lora(
            vec_silu.copy(), base.sgl_mod[bi], _sgl_mod_lo(sset, bi), 3 * D, D, ctx
        )
        var sm = _single_modvecs_from_flat(sm_flat, D)
        var bl = _single_lora_for(lora, bi)
        var fwd = single_block_lora_forward[H, Dh, S](
            x.copy(), sbw[bi], sm, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        sgl_saved.append(fwd.saved.copy())
        sgl_mod_flat.append(sm_flat^)
        x = fwd.out.copy()

    var parts = _split_seq(x, N_TXT, N_IMG, D)
    var img_out = parts[1].copy()

    var fb = Optional[Tensor](base.final_adaln_b[].clone(ctx))
    var fmods_base = linear(_t(vec_silu.copy(), [1, D], ctx), base.final_adaln_w[], fb, ctx).to_host(ctx)
    var fmods = _stack_lora_apply(fmods_base, vec_silu.copy(), _level_lo(sset, ST_NORM_OUT), 1, ctx)
    var final_shift = _chunk(fmods, 0, D)
    var final_scale = _chunk(fmods, 1, D)

    var ln_img_out = layer_norm(
        _t(img_out.copy(), [N_IMG, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), eps, ctx,
    ).to_host(ctx)
    var normed = modulate(
        _t(ln_img_out.copy(), [N_IMG, D], ctx),
        _t(final_scale.copy(), [D], ctx), _t(final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var flb = Optional[Tensor](base.final_lin_b[].clone(ctx))
    var out_base = linear(_t(normed.copy(), [N_IMG, D], ctx), base.final_lin[], flb, ctx).to_host(ctx)
    var out = _stack_lora_apply(out_base, normed.copy(), _level_lo(sset, ST_PROJ_OUT), N_IMG, ctx)

    return FluxStackForward(
        out^, vec^, vec_silu^,
        dbl_img_mod^, dbl_txt_mod^, sgl_mod_flat^,
        dbl_saved^, sgl_saved^,
        TArc(_t(img_out^, [N_IMG, D], ctx)), TArc(_t(ln_img_out^, [N_IMG, D], ctx)),
        final_shift^, final_scale^,
        t_emb^, t_hid^, g_emb^, g_hid^, v_hid^,
    )


# Thin wrapper: block-projection ONLY (stack-level LoRA disabled). Passes an
# empty FluxStackLoraSet so the original backward runs unchanged.
def flux_stack_lora_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: FluxStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights], lora: FluxLoraSet,
    cos: List[Float32], sin: List[Float32],
    saved: FluxStackForward,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, max_period: Float32, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxLoraGradSet:
    var sset = empty_flux_stack_lora_set(len(dbw), len(sbw), lora.rank)
    return flux_stack_lora_backward_full[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens, txt_tokens, base, dbw, sbw, lora, sset, _zeros(VEC_DIM),
        cos, sin, saved, D, Fmlp, in_ch, txt_ch, out_ch, T_DIM, VEC_DIM, max_period, eps, ctx,
    )


# FULL backward: block-projection LoRA grads PLUS stack-level LoRA grads. `vector`
# is the real CLIP-pooled input (used ONLY by the text_embedder lin1 d_a; ignored
# when sset disabled). Stack grads land in the returned FluxLoraGradSet.st_d_a /
# .st_d_b in the canonical order the AdamW + save walk uses.
def flux_stack_lora_backward_full[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: FluxStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights], lora: FluxLoraSet,
    sset: FluxStackLoraSet, vector: List[Float32],
    cos: List[Float32], sin: List[Float32],
    saved: FluxStackForward,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, max_period: Float32, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxLoraGradSet:
    var num_double = len(dbw)
    var num_single = len(sbw)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    var n_adapters = total_adapters(lora)
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a_flat.append(List[Float32]()); d_b_flat.append(List[Float32]())
    var nonfinite = 0
    var sbuf = _StackGradBuf(num_double, num_single)

    var d_vec_silu = _zeros(D)

    # ── final layer backward (frozen base, same as flux_stack_backward) ──
    var normed = modulate(
        saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), _t(saved.final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    # proj_out = linear(normed, final_lin) [+ LoRA(proj_out)]
    var lbf = linear_backward(
        _t(d_out.copy(), [N_IMG, out_ch], ctx), _t(normed.copy(), [N_IMG, D], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)
    var lo_proj_out = _level_lo(sset, ST_PROJ_OUT)
    if lo_proj_out:
        var lg = flux_lora_bwd(d_out.copy(), normed.copy(), lo_proj_out.value(), N_IMG, ctx)
        sbuf.lvl_d_a[ST_PROJ_OUT] = lg.d_a.copy()
        sbuf.lvl_d_b[ST_PROJ_OUT] = lg.d_b.copy()
        d_normed = _add_lists(d_normed, lg.d_x)
    var mbf = modulate_backward(
        _t(d_normed, [N_IMG, D], ctx), saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)
    var d_final_scale = mbf.d_scale.to_host(ctx)
    var d_final_shift = mbf.d_shift.to_host(ctx)
    var d_fmods = _concat_seq(d_final_shift, d_final_scale)
    # norm_out = linear(vec_silu, final_adaln_w) [+ LoRA(norm_out)]
    var lb_fmods = linear_backward(
        _t(d_fmods.copy(), [1, 2 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
        base.final_adaln_w[], 1, D, 2 * D, ctx,
    )
    d_vec_silu = _add_lists(d_vec_silu, lb_fmods.d_x.to_host(ctx))
    var lo_norm_out = _level_lo(sset, ST_NORM_OUT)
    if lo_norm_out:
        var lg = flux_lora_bwd(d_fmods.copy(), saved.vec_silu.copy(), lo_norm_out.value(), 1, ctx)
        sbuf.lvl_d_a[ST_NORM_OUT] = lg.d_a.copy()
        sbuf.lvl_d_b[ST_NORM_OUT] = lg.d_b.copy()
        d_vec_silu = _add_lists(d_vec_silu, lg.d_x)

    var lnbf = layer_norm_backward(
        _t(d_ln_img_out, [N_IMG, D], ctx), saved.img_out[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_img_out = lnbf.d_x.to_host(ctx)

    var d_x = _concat_seq(_zeros(N_TXT * D), d_img_out)

    # ── single-stream backward (REVERSE; LoRA) ──
    var bi = num_single - 1
    while bi >= 0:
        var sm = _single_modvecs_from_flat(saved.sgl_mod_flat[bi].copy(), D)
        var bl = _single_lora_for(lora, bi)
        var bg = single_block_lora_backward[H, Dh, S](
            d_x.copy(), sbw[bi], sm, bl, saved.sgl_saved[bi], cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_x = bg.base.d_x.copy()
        # scatter single-block LoRA grads
        var sbase = _sgl_base(lora, bi)
        for s in range(SGL_SLOTS):
            d_a_flat[sbase + s] = bg.lora.d_a[s].copy()
            d_b_flat[sbase + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        # modulation.lin grad -> d_vec_silu (frozen base mod.lin) [+ LoRA(norm.linear)]
        var d_sm = _single_modvec3(bg.base)
        var lb_sm = linear_backward(
            _t(d_sm.copy(), [1, 3 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.sgl_mod[bi].w[], 1, D, 3 * D, ctx,
        )
        d_vec_silu = _add_lists(d_vec_silu, lb_sm.d_x.to_host(ctx))
        var lo_sm = _sgl_mod_lo(sset, bi)
        if lo_sm:
            var lg = flux_lora_bwd(d_sm.copy(), saved.vec_silu.copy(), lo_sm.value(), 1, ctx)
            sbuf.sgl_d_a[bi] = lg.d_a.copy()
            sbuf.sgl_d_b[bi] = lg.d_b.copy()
            d_vec_silu = _add_lists(d_vec_silu, lg.d_x)
        bi -= 1

    var seam = _split_seq(d_x, N_TXT, N_IMG, D)
    var d_to = seam[0].copy()
    var d_io = seam[1].copy()

    # ── double-stream backward (REVERSE; LoRA) ──
    var di = num_double - 1
    while di >= 0:
        var im = _modvecs_from_flat(saved.dbl_img_mod[di].copy(), D)
        var tm = _modvecs_from_flat(saved.dbl_txt_mod[di].copy(), D)
        var bl = _double_lora_for(lora, di)
        var bg = double_block_lora_backward[H, Dh, N_IMG, N_TXT, S](
            d_io.copy(), d_to.copy(), dbw[di], im, tm, bl, saved.dbl_saved[di],
            cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_io = bg.base.img.d_x.copy()
        d_to = bg.base.txt.d_x.copy()
        # scatter double-block LoRA grads (img slots then txt slots)
        var dbase = _dbl_base(di)
        for s in range(DBL_STREAM_SLOTS):
            d_a_flat[dbase + s] = bg.lora.img.d_a[s].copy()
            d_b_flat[dbase + s] = bg.lora.img.d_b[s].copy()
            d_a_flat[dbase + DBL_STREAM_SLOTS + s] = bg.lora.txt.d_a[s].copy()
            d_b_flat[dbase + DBL_STREAM_SLOTS + s] = bg.lora.txt.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.img.d_a[s]) + _nonfinite(bg.lora.img.d_b[s])
            nonfinite += _nonfinite(bg.lora.txt.d_a[s]) + _nonfinite(bg.lora.txt.d_b[s])
        # img_mod/txt_mod grads -> their mod.lin backward -> d_vec_silu [+ LoRA]
        var d_im = _modvec6(bg.base.img)
        var d_tm = _modvec6(bg.base.txt)
        var lb_im = linear_backward(
            _t(d_im.copy(), [1, 6 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.dbl_mod[di].img.w[], 1, D, 6 * D, ctx,
        )
        var lb_tm = linear_backward(
            _t(d_tm.copy(), [1, 6 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.dbl_mod[di].txt.w[], 1, D, 6 * D, ctx,
        )
        d_vec_silu = _add_lists(d_vec_silu, lb_im.d_x.to_host(ctx))
        d_vec_silu = _add_lists(d_vec_silu, lb_tm.d_x.to_host(ctx))
        var lo_im = _dbl_mod_lo(sset, di, True)
        if lo_im:
            var lg = flux_lora_bwd(d_im.copy(), saved.vec_silu.copy(), lo_im.value(), 1, ctx)
            sbuf.dimg_d_a[di] = lg.d_a.copy()
            sbuf.dimg_d_b[di] = lg.d_b.copy()
            d_vec_silu = _add_lists(d_vec_silu, lg.d_x)
        var lo_tm = _dbl_mod_lo(sset, di, False)
        if lo_tm:
            var lg = flux_lora_bwd(d_tm.copy(), saved.vec_silu.copy(), lo_tm.value(), 1, ctx)
            sbuf.dtxt_d_a[di] = lg.d_a.copy()
            sbuf.dtxt_d_b[di] = lg.d_b.copy()
            d_vec_silu = _add_lists(d_vec_silu, lg.d_x)
        di -= 1

    # ── input-projection backward (+ LoRA on x_embedder / context_embedder) ──
    var lbi = linear_backward(
        _t(d_io.copy(), [N_IMG, D], ctx), _t(img_tokens.copy(), [N_IMG, in_ch], ctx), base.img_in[],
        N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var lo_x_emb = _level_lo(sset, ST_X_EMB)
    if lo_x_emb:
        var lg = flux_lora_bwd(d_io.copy(), img_tokens.copy(), lo_x_emb.value(), N_IMG, ctx)
        sbuf.lvl_d_a[ST_X_EMB] = lg.d_a.copy()
        sbuf.lvl_d_b[ST_X_EMB] = lg.d_b.copy()
        d_img_tokens = _add_lists(d_img_tokens, lg.d_x)
    var lbt = linear_backward(
        _t(d_to.copy(), [N_TXT, D], ctx), _t(txt_tokens.copy(), [N_TXT, txt_ch], ctx), base.txt_in[],
        N_TXT, txt_ch, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)
    var lo_ctx_emb = _level_lo(sset, ST_CTX_EMB)
    if lo_ctx_emb:
        var lg = flux_lora_bwd(d_to.copy(), txt_tokens.copy(), lo_ctx_emb.value(), N_TXT, ctx)
        sbuf.lvl_d_a[ST_CTX_EMB] = lg.d_a.copy()
        sbuf.lvl_d_b[ST_CTX_EMB] = lg.d_b.copy()
        d_txt_tokens = _add_lists(d_txt_tokens, lg.d_x)

    # ── vec backward: silu(vec) -> d_vec -> embeds ──
    var d_vec = silu_backward(_t(d_vec_silu.copy(), [1, D], ctx), _t(saved.vec.copy(), [1, D], ctx), ctx).to_host(ctx)
    var d_timestep = List[Float32]()
    var d_guidance = List[Float32]()
    var d_vector = List[Float32]()
    if sset.enabled:
        var eb = _embed_vec_backward_lora(
            d_vec.copy(), saved, base, sset, vector.copy(), base.has_guidance,
            D, T_DIM, VEC_DIM, max_period, ctx,
        )
        d_timestep = eb.d_timestep.copy()
        d_guidance = eb.d_guidance.copy()
        d_vector = eb.d_vector.copy()
        # embedder LoRA grads (ST_* order: TIME_1,TIME_2,TEXT_1,TEXT_2,GUID_1,GUID_2)
        sbuf.lvl_d_a[ST_TIME_1] = eb.lo_d_a[0].copy(); sbuf.lvl_d_b[ST_TIME_1] = eb.lo_d_b[0].copy()
        sbuf.lvl_d_a[ST_TIME_2] = eb.lo_d_a[1].copy(); sbuf.lvl_d_b[ST_TIME_2] = eb.lo_d_b[1].copy()
        sbuf.lvl_d_a[ST_TEXT_1] = eb.lo_d_a[2].copy(); sbuf.lvl_d_b[ST_TEXT_1] = eb.lo_d_b[2].copy()
        sbuf.lvl_d_a[ST_TEXT_2] = eb.lo_d_a[3].copy(); sbuf.lvl_d_b[ST_TEXT_2] = eb.lo_d_b[3].copy()
        sbuf.lvl_d_a[ST_GUID_1] = eb.lo_d_a[4].copy(); sbuf.lvl_d_b[ST_GUID_1] = eb.lo_d_b[4].copy()
        sbuf.lvl_d_a[ST_GUID_2] = eb.lo_d_a[5].copy(); sbuf.lvl_d_b[ST_GUID_2] = eb.lo_d_b[5].copy()
    else:
        var eb = _embed_vec_backward(d_vec.copy(), saved, base, base.has_guidance, D, T_DIM, VEC_DIM, max_period, ctx)
        d_timestep = eb.d_timestep.copy()
        d_guidance = eb.d_guidance.copy()
        d_vector = eb.d_vector.copy()

    # Stack-level grads land in plain locals (mut out-params), moved ONCE into the
    # result — no struct return, so no partial-move hazard.
    var st_d_a = List[List[Float32]]()
    var st_d_b = List[List[Float32]]()
    nonfinite += _assemble_stack_grads_into(sset, sbuf, st_d_a, st_d_b)

    return FluxLoraGradSet(
        d_a_flat^, d_b_flat^,
        d_img_tokens^, d_txt_tokens^, d_vec^,
        d_timestep^, d_guidance^, d_vector^,
        nonfinite,
        st_d_a^, st_d_b^,
    )


# ── AdamW step on EVERY adapter (reuses the proven per-adapter _lora_adamw) ───
def flux_lora_adamw_step(
    mut set: FluxLoraSet, grads: FluxLoraGradSet, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var n = total_adapters(set)
    for i in range(n):
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(set.ad[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay)


# AdamW on the STACK-LEVEL adapters, walked in the SAME canonical order
# _assemble_stack_grads used to build grads.st_d_a/.st_d_b (level ST_* slots,
# then per-double img/txt mod, then per-single mod — populated only). Updates the
# Optional adapters in `sset` in place. No-op when stack-level LoRA is disabled.
def flux_stack_lora_adamw_step(
    mut sset: FluxStackLoraSet, grads: FluxLoraGradSet, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    if not sset.enabled:
        return
    var idx = 0
    for slot in range(ST_LEVEL_SLOTS):
        if sset.level[slot]:
            var ad = sset.level[slot].value().copy()
            var lg = LoraGrads(grads.st_d_a[idx].copy(), grads.st_d_b[idx].copy())
            _lora_adamw(ad, lg, t, lr, ctx, beta1, beta2, eps, weight_decay)
            sset.level[slot] = Optional[LoraAdapter](ad^)
            idx += 1
    for bi in range(sset.num_double):
        if sset.dbl_img_mod[bi]:
            var ad = sset.dbl_img_mod[bi].value().copy()
            var lg = LoraGrads(grads.st_d_a[idx].copy(), grads.st_d_b[idx].copy())
            _lora_adamw(ad, lg, t, lr, ctx, beta1, beta2, eps, weight_decay)
            sset.dbl_img_mod[bi] = Optional[LoraAdapter](ad^)
            idx += 1
        if sset.dbl_txt_mod[bi]:
            var ad = sset.dbl_txt_mod[bi].value().copy()
            var lg = LoraGrads(grads.st_d_a[idx].copy(), grads.st_d_b[idx].copy())
            _lora_adamw(ad, lg, t, lr, ctx, beta1, beta2, eps, weight_decay)
            sset.dbl_txt_mod[bi] = Optional[LoraAdapter](ad^)
            idx += 1
    for bi in range(sset.num_single):
        if sset.sgl_mod[bi]:
            var ad = sset.sgl_mod[bi].value().copy()
            var lg = LoraGrads(grads.st_d_a[idx].copy(), grads.st_d_b[idx].copy())
            _lora_adamw(ad, lg, t, lr, ctx, beta1, beta2, eps, weight_decay)
            sset.sgl_mod[bi] = Optional[LoraAdapter](ad^)
            idx += 1


# ── OneTrainer legacy/raw save-key scheme for the adapters Flux trains now ──
# Prefixes match OneTrainer's default Flux LoRA safetensors export for the
# current block-projection surface. save_lora_onetrainer appends .alpha /
# .lora_down.weight / .lora_up.weight.
def _dbl_stream_prefix(bi: Int, stream_img: Bool, slot: Int) -> String:
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


def _sgl_prefix(bi: Int, slot: Int) -> String:
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


# Old PEFT/BFL prefixes are retained only for resume compatibility with files
# saved before the product LoRA save moved to OneTrainer raw suffixes.
def _legacy_peft_dbl_stream_prefix(bi: Int, stream_img: Bool, slot: Int) -> String:
    var b = String("double") + String("_blocks.") + String(bi) + "."
    var s = "img" if stream_img else "txt"
    if slot == D_SQ:
        return b + s + "_attn.qkv.0"
    elif slot == D_SK:
        return b + s + "_attn.qkv.1"
    elif slot == D_SV:
        return b + s + "_attn.qkv.2"
    elif slot == D_PROJ:
        return b + s + "_attn.proj"
    elif slot == D_MLP0:
        return b + s + "_mlp.0"
    return b + s + "_mlp.2"


def _legacy_peft_sgl_prefix(bi: Int, slot: Int) -> String:
    var b = String("single") + String("_blocks.") + String(bi) + "."
    if slot == S_SQ:
        return b + "linear1.0"
    elif slot == S_SK:
        return b + "linear1.1"
    elif slot == S_SV:
        return b + "linear1.2"
    elif slot == S_PMLP:
        return b + "linear1.3"
    return b + "linear2"


def flux_lora_prefixes(num_double: Int, num_single: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_double):
        for s in range(DBL_STREAM_SLOTS):   # img stream
            out.append(_dbl_stream_prefix(bi, True, s))
        for s in range(DBL_STREAM_SLOTS):   # txt stream
            out.append(_dbl_stream_prefix(bi, False, s))
    for bi in range(num_single):
        for s in range(SGL_SLOTS):
            out.append(_sgl_prefix(bi, s))
    return out^


def _dbl_ot_norm_prefix(bi: Int, stream_img: Bool) -> String:
    var b = String("lora_transformer_transformer_blocks_") + String(bi) + "_"
    if stream_img:
        return b + "norm1_linear"
    return b + "norm1_context_linear"


def _sgl_ot_norm_prefix(bi: Int) -> String:
    return (
        String("lora_transformer_single_transformer_blocks_") + String(bi)
        + "_norm_linear"
    )


def _append_flux_ot_stack_prefixes(mut out: List[String]):
    out.append("lora_transformer_context_embedder")
    out.append("lora_transformer_norm_out_linear")
    out.append("lora_transformer_proj_out")
    out.append("lora_transformer_time_text_embed_guidance_embedder_linear_1")
    out.append("lora_transformer_time_text_embed_guidance_embedder_linear_2")
    out.append("lora_transformer_time_text_embed_text_embedder_linear_1")
    out.append("lora_transformer_time_text_embed_text_embedder_linear_2")
    out.append("lora_transformer_time_text_embed_timestep_embedder_linear_1")
    out.append("lora_transformer_time_text_embed_timestep_embedder_linear_2")
    out.append("lora_transformer_x_embedder")


def flux_lora_ot_transformer_prefixes(num_double: Int, num_single: Int) -> List[String]:
    """OneTrainer Flux transformer LoRA inventory for current default filters.

    This is an inventory contract, not a math claim. `flux_lora_prefixes` is the
    supported trained/saveable subset. `flux_lora_missing_ot_transformer_prefixes`
    enumerates the still-frozen targets that must be wired before claiming full
    OneTrainer transformer parity.
    """
    var out = List[String]()
    _append_flux_ot_stack_prefixes(out)
    for bi in range(num_double):
        for s in range(DBL_STREAM_SLOTS):
            out.append(_dbl_stream_prefix(bi, True, s))
        out.append(_dbl_ot_norm_prefix(bi, True))
        for s in range(DBL_STREAM_SLOTS):
            out.append(_dbl_stream_prefix(bi, False, s))
        out.append(_dbl_ot_norm_prefix(bi, False))
    for bi in range(num_single):
        for s in range(SGL_SLOTS):
            out.append(_sgl_prefix(bi, s))
        out.append(_sgl_ot_norm_prefix(bi))
    return out^


def flux_lora_missing_ot_transformer_prefixes(num_double: Int, num_single: Int) -> List[String]:
    var out = List[String]()
    _append_flux_ot_stack_prefixes(out)
    for bi in range(num_double):
        out.append(_dbl_ot_norm_prefix(bi, True))
        out.append(_dbl_ot_norm_prefix(bi, False))
    for bi in range(num_single):
        out.append(_sgl_ot_norm_prefix(bi))
    return out^


def require_flux_lora_ot_transformer_complete(num_double: Int, num_single: Int) raises:
    var missing = flux_lora_missing_ot_transformer_prefixes(num_double, num_single)
    if len(missing) != 0:
        raise Error(
            String("Flux LoRA transformer surface is not full OneTrainer parity: ")
            + String(len(missing)) + " target groups remain frozen; first missing "
            + missing[0]
        )


def require_flux_lora_text_encoder_disabled(train_te1: Bool, train_te2: Bool) raises:
    if train_te1:
        raise Error("Flux LoRA lora_te1 save/resume surface is not implemented")
    if train_te2:
        raise Error("Flux LoRA lora_te2 save/resume surface is not implemented")


def _legacy_peft_flux_lora_prefixes(num_double: Int, num_single: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_double):
        for s in range(DBL_STREAM_SLOTS):   # img stream
            out.append(_legacy_peft_dbl_stream_prefix(bi, True, s))
        for s in range(DBL_STREAM_SLOTS):   # txt stream
            out.append(_legacy_peft_dbl_stream_prefix(bi, False, s))
    for bi in range(num_single):
        for s in range(SGL_SLOTS):
            out.append(_legacy_peft_sgl_prefix(bi, s))
    return out^


def _flux_set_from_named(
    named: List[NamedLora], num_double: Int, num_single: Int, rank: Int,
) -> FluxLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(named)):
        ad.append(named[i].adapter.copy())
    return FluxLoraSet(ad^, num_double, num_single, rank)


def _flux_named_loras(set: FluxLoraSet) -> List[NamedLora]:
    var prefixes = flux_lora_prefixes(set.num_double, set.num_single)
    var named = List[NamedLora]()
    var n = total_adapters(set)
    for i in range(n):
        named.append(NamedLora(prefixes[i], set.ad[i].copy()))
    return named^


# OT key for one transformer-level adapter slot (ST_* order matches the AdamW /
# scatter walk). Reuses the canonical OT names from _append_flux_ot_stack_prefixes.
def _st_level_prefix(slot: Int) -> String:
    if slot == ST_CTX_EMB:
        return String("lora_transformer_context_embedder")
    elif slot == ST_X_EMB:
        return String("lora_transformer_x_embedder")
    elif slot == ST_TIME_1:
        return String("lora_transformer_time_text_embed_timestep_embedder_linear_1")
    elif slot == ST_TIME_2:
        return String("lora_transformer_time_text_embed_timestep_embedder_linear_2")
    elif slot == ST_TEXT_1:
        return String("lora_transformer_time_text_embed_text_embedder_linear_1")
    elif slot == ST_TEXT_2:
        return String("lora_transformer_time_text_embed_text_embedder_linear_2")
    elif slot == ST_GUID_1:
        return String("lora_transformer_time_text_embed_guidance_embedder_linear_1")
    elif slot == ST_GUID_2:
        return String("lora_transformer_time_text_embed_guidance_embedder_linear_2")
    elif slot == ST_NORM_OUT:
        return String("lora_transformer_norm_out_linear")
    return String("lora_transformer_proj_out")


# Named LoRAs for the POPULATED stack-level adapters, in the SAME canonical order
# the AdamW walk + _assemble_stack_grads use (level ST_* slots, then per-double
# img/txt mod, then per-single mod). Empty when stack-level LoRA is disabled.
def _flux_stack_named_loras(sset: FluxStackLoraSet) -> List[NamedLora]:
    var named = List[NamedLora]()
    if not sset.enabled:
        return named^
    for slot in range(ST_LEVEL_SLOTS):
        if sset.level[slot]:
            named.append(NamedLora(_st_level_prefix(slot), sset.level[slot].value().copy()))
    for bi in range(sset.num_double):
        if sset.dbl_img_mod[bi]:
            named.append(NamedLora(_dbl_ot_norm_prefix(bi, True), sset.dbl_img_mod[bi].value().copy()))
        if sset.dbl_txt_mod[bi]:
            named.append(NamedLora(_dbl_ot_norm_prefix(bi, False), sset.dbl_txt_mod[bi].value().copy()))
    for bi in range(sset.num_single):
        if sset.sgl_mod[bi]:
            named.append(NamedLora(_sgl_ot_norm_prefix(bi), sset.sgl_mod[bi].value().copy()))
    return named^


def save_flux_lora(set: FluxLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = _flux_named_loras(set)
    return save_lora_onetrainer(named, path, ctx)


def save_flux_lora_state(set: FluxLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = _flux_named_loras(set)
    return save_lora_train_state(named, path, ctx)


# Combined save: block-projection adapters (`set`) AND stack-level adapters
# (`sset`) into ONE OneTrainer-keyed safetensors — the full default surface OT
# exports. Block keys first (existing order), then the populated stack keys.
def save_flux_lora_combined(
    set: FluxLoraSet, sset: FluxStackLoraSet, path: String, ctx: DeviceContext
) raises -> Int:
    var named = _flux_named_loras(set)
    var st_named = _flux_stack_named_loras(sset)
    for i in range(len(st_named)):
        named.append(st_named[i].copy())
    return save_lora_onetrainer(named, path, ctx)


def save_flux_lora_state_combined(
    set: FluxLoraSet, sset: FluxStackLoraSet, path: String, ctx: DeviceContext
) raises -> Int:
    var named = _flux_named_loras(set)
    var st_named = _flux_stack_named_loras(sset)
    for i in range(len(st_named)):
        named.append(st_named[i].copy())
    return save_lora_train_state(named, path, ctx)


def load_flux_lora_resume(
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> FluxLoraSet:
    var prefixes = flux_lora_prefixes(num_double, num_single)
    var scale = alpha / Float32(rank)
    try:
        var named = load_lora_for_resume(prefixes, scale, path, ctx)   # flat order
        return _flux_set_from_named(named, num_double, num_single, rank)
    except:
        var legacy_prefixes = _legacy_peft_flux_lora_prefixes(num_double, num_single)
        var named = load_lora_for_resume(legacy_prefixes, scale, path, ctx)
        return _flux_set_from_named(named, num_double, num_single, rank)


def load_flux_lora_state(
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> FluxLoraSet:
    var prefixes = flux_lora_prefixes(num_double, num_single)
    var scale = alpha / Float32(rank)
    var named = load_lora_train_state(prefixes, scale, path, ctx)
    return _flux_set_from_named(named, num_double, num_single, rank)


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK-SWAP OFFLOAD LoRA stack — streams ONE transformer block at a time.
#
# WHY THIS PATH EXISTS (RESIDENCY note above): full F32 residency of 19+38
# real-depth blocks is ~34GB and does NOT fit a 24GB GPU. The RESIDENT stack
# (flux_stack_lora_forward/backward) holds every DoubleBlockWeights/
# SingleBlockWeights at once. The offload stack mirrors it EXACTLY — same math,
# same checkpoint contract, same LoRA-grad collection — but pulls each block's
# base weights from a `TurboPlannedLoader` (block-swap), builds the block weight
# struct, runs the ALREADY-VERIFIED per-block LoRA fwd/bwd (lora_block.mojo),
# then DROPS the block weights before the next block. The LoRA adapters + the
# checkpoint activations (host List[Float32], cheap) stay resident.
#
# CONTRACT (mirrors klein_stack_lora's offload_turbo path, lines 699-786 +
# 1457-1629): the streamed block plan is build_flux1_dev_block_plan() (19 double
# then 38 single), so block index bi (0..num_double-1) maps to the corresponding
# double block and num_double+bi maps to the corresponding single block — the
# SAME flat order the resident stack and the LoRA carrier use. The forward keeps
# the FULL FluxStackForward
# tape (per-block host saved activations) so the backward is recompute-FREE for
# the block math (it only re-streams the block WEIGHTS). This matches the
# resident stack's no-recompute backward: flux block forward already saves every
# activation it needs, so the offload backward reuses saved.dbl_saved[bi] /
# saved.sgl_saved[bi] verbatim, just re-streaming the weights to run the bwd
# arms. No double_block_lora_forward recompute is needed (unlike Klein, whose
# device-resident blocks discard most saved tensors).
#
# Tenet 1: NO new block math. The block weight struct is built from streamed
# device tensors by copying ArcPointer handles, preserving checkpoint dtype and
# avoiding D2H/H2D round-trips. Tenet 2: the offload entry points take the same
# args as the resident ones with `dbw`/`sbw` replaced by `mut loader`.
# ═════════════════════════════════════════════════════════════════════════════

# Borrow/copy one streamed block tensor as a device-resident Arc. The copied Arc
# keeps the tensor alive after the loader's active block handle is released.
def _block_tensor(block: Block, key: String) raises -> TArc:
    if not (key in block):
        raise Error(String("Flux offload block missing tensor: ") + key)
    return block[key].copy()


def _stream_weights_from_block(
    block: Block, dp: String, stream: String,
    D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
) raises -> StreamWeights:
    var ap = dp + String(".") + stream + String("_attn")
    var mp = dp + String(".") + stream + String("_mlp")
    return StreamWeights(
        _block_tensor(block, ap + String(".qkv.weight")),
        _block_tensor(block, ap + String(".qkv.bias")),
        _block_tensor(block, ap + String(".proj.weight")),
        _block_tensor(block, ap + String(".proj.bias")),
        _block_tensor(block, mp + String(".0.weight")),
        _block_tensor(block, mp + String(".0.bias")),
        _block_tensor(block, mp + String(".2.weight")),
        _block_tensor(block, mp + String(".2.bias")),
        _block_tensor(block, ap + String(".norm.query_norm.scale")),
        _block_tensor(block, ap + String(".norm.key_norm.scale")),
    )


def _double_weights_from_block(
    block: Block, dp: String, D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
) raises -> DoubleBlockWeights:
    return DoubleBlockWeights(
        _stream_weights_from_block(block, dp, String("img"), D, Fmlp, Dh, ctx),
        _stream_weights_from_block(block, dp, String("txt"), D, Fmlp, Dh, ctx),
    )


def _single_weights_from_block(
    block: Block, sp: String, D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _block_tensor(block, sp + String(".linear1.weight")),
        _block_tensor(block, sp + String(".linear1.bias")),
        _block_tensor(block, sp + String(".linear2.weight")),
        _block_tensor(block, sp + String(".linear2.bias")),
        _block_tensor(block, sp + String(".norm.query_norm.scale")),
        _block_tensor(block, sp + String(".norm.key_norm.scale")),
    )


# ── FULL FORWARD WITH LoRA, BLOCK-SWAP OFFLOAD ───────────────────────────────
# Identical to flux_stack_lora_forward except double/single block weights are
# streamed per-block via `loader` (built from build_flux1_dev_block_plan) instead
# of held in resident `dbw`/`sbw` lists. Dh is needed to build the streamed
# weight structs (resident path derives it from the passed structs).
# Thin wrapper: block-projection ONLY (stack-level LoRA disabled).
def flux_stack_lora_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    timestep: List[Float32], guidance: Optional[List[Float32]], vector: List[Float32],
    base: FluxStackBase,
    mut loader: TurboPlannedLoader, lora: FluxLoraSet,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxStackForward:
    var sset = empty_flux_stack_lora_set(lora.num_double, lora.num_single, lora.rank)
    return flux_stack_lora_forward_offload_full[H, Dh, N_IMG, N_TXT, S](
        img_tokens, txt_tokens, timestep, guidance, vector, base,
        loader, lora, sset, cos, sin,
        D, Fmlp, in_ch, txt_ch, out_ch, T_DIM, VEC_DIM, eps, ctx,
    )


# FULL offload forward: block-projection LoRA PLUS stack-level LoRA (`sset`).
# Mirrors flux_stack_lora_forward_full but streams block weights per-block.
def flux_stack_lora_forward_offload_full[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    timestep: List[Float32], guidance: Optional[List[Float32]], vector: List[Float32],
    base: FluxStackBase,
    mut loader: TurboPlannedLoader, lora: FluxLoraSet, sset: FluxStackLoraSet,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxStackForward:
    var num_double = lora.num_double
    var num_single = lora.num_single

    loader.prefetch_with_ctx(0, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    var vec = List[Float32]()
    var t_emb = List[Float32]()
    var t_hid = List[Float32]()
    var g_emb = List[Float32]()
    var g_hid = List[Float32]()
    var v_hid = List[Float32]()
    if sset.enabled:
        var vf = _embed_vec_forward_lora(timestep, guidance, vector, base, sset, D, T_DIM, VEC_DIM, ctx)
        vec = vf.vec.copy(); t_emb = vf.t_emb.copy(); t_hid = vf.t_hid.copy()
        g_emb = vf.g_emb.copy(); g_hid = vf.g_hid.copy(); v_hid = vf.v_hid.copy()
    else:
        var vf = _embed_vec_forward(timestep, guidance, vector, base, D, T_DIM, VEC_DIM, ctx)
        vec = vf.vec.copy(); t_emb = vf.t_emb.copy(); t_hid = vf.t_hid.copy()
        g_emb = vf.g_emb.copy(); g_hid = vf.g_hid.copy(); v_hid = vf.v_hid.copy()
    var vec_silu = silu(_t(vec.copy(), [1, D], ctx), ctx).to_host(ctx)

    var bi_img = Optional[Tensor](base.img_in_b[].clone(ctx))
    var img_base = linear(_t(img_tokens.copy(), [N_IMG, in_ch], ctx), base.img_in[], bi_img, ctx).to_host(ctx)
    var img = _stack_lora_apply(img_base, img_tokens.copy(), _level_lo(sset, ST_X_EMB), N_IMG, ctx)
    var bi_txt = Optional[Tensor](base.txt_in_b[].clone(ctx))
    var txt_base = linear(_t(txt_tokens.copy(), [N_TXT, txt_ch], ctx), base.txt_in[], bi_txt, ctx).to_host(ctx)
    var txt = _stack_lora_apply(txt_base, txt_tokens.copy(), _level_lo(sset, ST_CTX_EMB), N_TXT, ctx)

    var dbl_img_mod = List[List[Float32]]()
    var dbl_txt_mod = List[List[Float32]]()
    var dbl_saved = List[DoubleBlockSaved]()
    for bi in range(num_double):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, D, Fmlp, Dh, ctx)
        var im_flat = _mod_proj_lora(
            vec_silu.copy(), base.dbl_mod[bi].img, _dbl_mod_lo(sset, bi, True), 6 * D, D, ctx
        )
        var tm_flat = _mod_proj_lora(
            vec_silu.copy(), base.dbl_mod[bi].txt, _dbl_mod_lo(sset, bi, False), 6 * D, D, ctx
        )
        var im = _modvecs_from_flat(im_flat, D)
        var tm = _modvecs_from_flat(tm_flat, D)
        var bl = _double_lora_for(lora, bi)
        var fwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), w, im, tm, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        dbl_saved.append(fwd.saved.copy())
        dbl_img_mod.append(im_flat^)
        dbl_txt_mod.append(tm_flat^)
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)

    var x = _concat_seq(txt, img)

    var sgl_mod_flat = List[List[Float32]]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, Fmlp, Dh, ctx)
        var sm_flat = _mod_proj_lora(
            vec_silu.copy(), base.sgl_mod[bi], _sgl_mod_lo(sset, bi), 3 * D, D, ctx
        )
        var sm = _single_modvecs_from_flat(sm_flat, D)
        var bl = _single_lora_for(lora, bi)
        var fwd = single_block_lora_forward[H, Dh, S](
            x.copy(), w, sm, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        sgl_saved.append(fwd.saved.copy())
        sgl_mod_flat.append(sm_flat^)
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)

    var parts = _split_seq(x, N_TXT, N_IMG, D)
    var img_out = parts[1].copy()

    var fb = Optional[Tensor](base.final_adaln_b[].clone(ctx))
    var fmods_base = linear(_t(vec_silu.copy(), [1, D], ctx), base.final_adaln_w[], fb, ctx).to_host(ctx)
    var fmods = _stack_lora_apply(fmods_base, vec_silu.copy(), _level_lo(sset, ST_NORM_OUT), 1, ctx)
    var final_shift = _chunk(fmods, 0, D)
    var final_scale = _chunk(fmods, 1, D)

    var ln_img_out = layer_norm(
        _t(img_out.copy(), [N_IMG, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), eps, ctx,
    ).to_host(ctx)
    var normed = modulate(
        _t(ln_img_out.copy(), [N_IMG, D], ctx),
        _t(final_scale.copy(), [D], ctx), _t(final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var flb = Optional[Tensor](base.final_lin_b[].clone(ctx))
    var out_base = linear(_t(normed.copy(), [N_IMG, D], ctx), base.final_lin[], flb, ctx).to_host(ctx)
    var out = _stack_lora_apply(out_base, normed.copy(), _level_lo(sset, ST_PROJ_OUT), N_IMG, ctx)

    return FluxStackForward(
        out^, vec^, vec_silu^,
        dbl_img_mod^, dbl_txt_mod^, sgl_mod_flat^,
        dbl_saved^, sgl_saved^,
        TArc(_t(img_out^, [N_IMG, D], ctx)), TArc(_t(ln_img_out^, [N_IMG, D], ctx)),
        final_shift^, final_scale^,
        t_emb^, t_hid^, g_emb^, g_hid^, v_hid^,
    )


# ── FULL BACKWARD WITH LoRA, BLOCK-SWAP OFFLOAD ──────────────────────────────
# Identical to flux_stack_lora_backward except block weights are streamed per
# block in REVERSE (singles last..0, then doubles last..0). Saved activations are
# reused from the forward tape (recompute-free block math); only the WEIGHTS are
# re-streamed. The plan is streamed back-to-front: prefetch the last block, then
# step down.
# Thin wrapper: block-projection ONLY (stack-level LoRA disabled).
def flux_stack_lora_backward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: FluxStackBase,
    mut loader: TurboPlannedLoader, lora: FluxLoraSet,
    cos: List[Float32], sin: List[Float32],
    saved: FluxStackForward,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, max_period: Float32, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxLoraGradSet:
    var sset = empty_flux_stack_lora_set(lora.num_double, lora.num_single, lora.rank)
    return flux_stack_lora_backward_offload_full[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens, txt_tokens, base, loader, lora, sset, _zeros(VEC_DIM),
        cos, sin, saved, D, Fmlp, in_ch, txt_ch, out_ch, T_DIM, VEC_DIM, max_period, eps, ctx,
    )


# FULL offload backward: block-projection grads PLUS stack-level grads. `vector`
# is the real CLIP-pooled input (text_embedder lin1 d_a). Mirrors
# flux_stack_lora_backward_full, streaming block weights per-block in REVERSE.
def flux_stack_lora_backward_offload_full[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: FluxStackBase,
    mut loader: TurboPlannedLoader, lora: FluxLoraSet,
    sset: FluxStackLoraSet, vector: List[Float32],
    cos: List[Float32], sin: List[Float32],
    saved: FluxStackForward,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, max_period: Float32, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxLoraGradSet:
    var num_double = lora.num_double
    var num_single = lora.num_single

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    var n_adapters = total_adapters(lora)
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a_flat.append(List[Float32]()); d_b_flat.append(List[Float32]())
    var nonfinite = 0
    var sbuf = _StackGradBuf(num_double, num_single)

    var d_vec_silu = _zeros(D)

    # ── final layer backward (frozen base, same as flux_stack_backward) ──
    var normed = modulate(
        saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), _t(saved.final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var lbf = linear_backward(
        _t(d_out.copy(), [N_IMG, out_ch], ctx), _t(normed.copy(), [N_IMG, D], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)
    var lo_proj_out = _level_lo(sset, ST_PROJ_OUT)
    if lo_proj_out:
        var lg = flux_lora_bwd(d_out.copy(), normed.copy(), lo_proj_out.value(), N_IMG, ctx)
        sbuf.lvl_d_a[ST_PROJ_OUT] = lg.d_a.copy()
        sbuf.lvl_d_b[ST_PROJ_OUT] = lg.d_b.copy()
        d_normed = _add_lists(d_normed, lg.d_x)
    var mbf = modulate_backward(
        _t(d_normed, [N_IMG, D], ctx), saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)
    var d_final_scale = mbf.d_scale.to_host(ctx)
    var d_final_shift = mbf.d_shift.to_host(ctx)
    var d_fmods = _concat_seq(d_final_shift, d_final_scale)
    var lb_fmods = linear_backward(
        _t(d_fmods.copy(), [1, 2 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
        base.final_adaln_w[], 1, D, 2 * D, ctx,
    )
    d_vec_silu = _add_lists(d_vec_silu, lb_fmods.d_x.to_host(ctx))
    var lo_norm_out = _level_lo(sset, ST_NORM_OUT)
    if lo_norm_out:
        var lg = flux_lora_bwd(d_fmods.copy(), saved.vec_silu.copy(), lo_norm_out.value(), 1, ctx)
        sbuf.lvl_d_a[ST_NORM_OUT] = lg.d_a.copy()
        sbuf.lvl_d_b[ST_NORM_OUT] = lg.d_b.copy()
        d_vec_silu = _add_lists(d_vec_silu, lg.d_x)

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
        var w = _single_weights_from_block(handle.block, handle.prefix, D, Fmlp, Dh, ctx)
        var sm = _single_modvecs_from_flat(saved.sgl_mod_flat[bi].copy(), D)
        var bl = _single_lora_for(lora, bi)
        var bg = single_block_lora_backward[H, Dh, S](
            d_x.copy(), w, sm, bl, saved.sgl_saved[bi], cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_x = bg.base.d_x.copy()
        var sbase = _sgl_base(lora, bi)
        for s in range(SGL_SLOTS):
            d_a_flat[sbase + s] = bg.lora.d_a[s].copy()
            d_b_flat[sbase + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        var d_sm = _single_modvec3(bg.base)
        var lb_sm = linear_backward(
            _t(d_sm.copy(), [1, 3 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.sgl_mod[bi].w[], 1, D, 3 * D, ctx,
        )
        d_vec_silu = _add_lists(d_vec_silu, lb_sm.d_x.to_host(ctx))
        var lo_sm = _sgl_mod_lo(sset, bi)
        if lo_sm:
            var lg = flux_lora_bwd(d_sm.copy(), saved.vec_silu.copy(), lo_sm.value(), 1, ctx)
            sbuf.sgl_d_a[bi] = lg.d_a.copy()
            sbuf.sgl_d_b[bi] = lg.d_b.copy()
            d_vec_silu = _add_lists(d_vec_silu, lg.d_x)
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
        var w = _double_weights_from_block(handle.block, handle.prefix, D, Fmlp, Dh, ctx)
        var im = _modvecs_from_flat(saved.dbl_img_mod[di].copy(), D)
        var tm = _modvecs_from_flat(saved.dbl_txt_mod[di].copy(), D)
        var bl = _double_lora_for(lora, di)
        var bg = double_block_lora_backward[H, Dh, N_IMG, N_TXT, S](
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
        var d_im = _modvec6(bg.base.img)
        var d_tm = _modvec6(bg.base.txt)
        var lb_im = linear_backward(
            _t(d_im.copy(), [1, 6 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.dbl_mod[di].img.w[], 1, D, 6 * D, ctx,
        )
        var lb_tm = linear_backward(
            _t(d_tm.copy(), [1, 6 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.dbl_mod[di].txt.w[], 1, D, 6 * D, ctx,
        )
        d_vec_silu = _add_lists(d_vec_silu, lb_im.d_x.to_host(ctx))
        d_vec_silu = _add_lists(d_vec_silu, lb_tm.d_x.to_host(ctx))
        var lo_im = _dbl_mod_lo(sset, di, True)
        if lo_im:
            var lg = flux_lora_bwd(d_im.copy(), saved.vec_silu.copy(), lo_im.value(), 1, ctx)
            sbuf.dimg_d_a[di] = lg.d_a.copy()
            sbuf.dimg_d_b[di] = lg.d_b.copy()
            d_vec_silu = _add_lists(d_vec_silu, lg.d_x)
        var lo_tm = _dbl_mod_lo(sset, di, False)
        if lo_tm:
            var lg = flux_lora_bwd(d_tm.copy(), saved.vec_silu.copy(), lo_tm.value(), 1, ctx)
            sbuf.dtxt_d_a[di] = lg.d_a.copy()
            sbuf.dtxt_d_b[di] = lg.d_b.copy()
            d_vec_silu = _add_lists(d_vec_silu, lg.d_x)
        loader.mark_active_block_done(ctx)
        di -= 1

    # ── input-projection backward (+ LoRA on x_embedder / context_embedder) ──
    var lbi = linear_backward(
        _t(d_io.copy(), [N_IMG, D], ctx), _t(img_tokens.copy(), [N_IMG, in_ch], ctx), base.img_in[],
        N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var lo_x_emb = _level_lo(sset, ST_X_EMB)
    if lo_x_emb:
        var lg = flux_lora_bwd(d_io.copy(), img_tokens.copy(), lo_x_emb.value(), N_IMG, ctx)
        sbuf.lvl_d_a[ST_X_EMB] = lg.d_a.copy()
        sbuf.lvl_d_b[ST_X_EMB] = lg.d_b.copy()
        d_img_tokens = _add_lists(d_img_tokens, lg.d_x)
    var lbt = linear_backward(
        _t(d_to.copy(), [N_TXT, D], ctx), _t(txt_tokens.copy(), [N_TXT, txt_ch], ctx), base.txt_in[],
        N_TXT, txt_ch, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)
    var lo_ctx_emb = _level_lo(sset, ST_CTX_EMB)
    if lo_ctx_emb:
        var lg = flux_lora_bwd(d_to.copy(), txt_tokens.copy(), lo_ctx_emb.value(), N_TXT, ctx)
        sbuf.lvl_d_a[ST_CTX_EMB] = lg.d_a.copy()
        sbuf.lvl_d_b[ST_CTX_EMB] = lg.d_b.copy()
        d_txt_tokens = _add_lists(d_txt_tokens, lg.d_x)

    # ── vec backward: silu(vec) -> d_vec -> embeds ──
    var d_vec = silu_backward(_t(d_vec_silu.copy(), [1, D], ctx), _t(saved.vec.copy(), [1, D], ctx), ctx).to_host(ctx)
    var d_timestep = List[Float32]()
    var d_guidance = List[Float32]()
    var d_vector = List[Float32]()
    if sset.enabled:
        var eb = _embed_vec_backward_lora(
            d_vec.copy(), saved, base, sset, vector.copy(), base.has_guidance,
            D, T_DIM, VEC_DIM, max_period, ctx,
        )
        d_timestep = eb.d_timestep.copy()
        d_guidance = eb.d_guidance.copy()
        d_vector = eb.d_vector.copy()
        sbuf.lvl_d_a[ST_TIME_1] = eb.lo_d_a[0].copy(); sbuf.lvl_d_b[ST_TIME_1] = eb.lo_d_b[0].copy()
        sbuf.lvl_d_a[ST_TIME_2] = eb.lo_d_a[1].copy(); sbuf.lvl_d_b[ST_TIME_2] = eb.lo_d_b[1].copy()
        sbuf.lvl_d_a[ST_TEXT_1] = eb.lo_d_a[2].copy(); sbuf.lvl_d_b[ST_TEXT_1] = eb.lo_d_b[2].copy()
        sbuf.lvl_d_a[ST_TEXT_2] = eb.lo_d_a[3].copy(); sbuf.lvl_d_b[ST_TEXT_2] = eb.lo_d_b[3].copy()
        sbuf.lvl_d_a[ST_GUID_1] = eb.lo_d_a[4].copy(); sbuf.lvl_d_b[ST_GUID_1] = eb.lo_d_b[4].copy()
        sbuf.lvl_d_a[ST_GUID_2] = eb.lo_d_a[5].copy(); sbuf.lvl_d_b[ST_GUID_2] = eb.lo_d_b[5].copy()
    else:
        var eb = _embed_vec_backward(d_vec.copy(), saved, base, base.has_guidance, D, T_DIM, VEC_DIM, max_period, ctx)
        d_timestep = eb.d_timestep.copy()
        d_guidance = eb.d_guidance.copy()
        d_vector = eb.d_vector.copy()

    # Stack-level grads land in plain locals (mut out-params), moved ONCE into the
    # result — no struct return, so no partial-move hazard.
    var st_d_a = List[List[Float32]]()
    var st_d_b = List[List[Float32]]()
    nonfinite += _assemble_stack_grads_into(sset, sbuf, st_d_a, st_d_b)

    return FluxLoraGradSet(
        d_a_flat^, d_b_flat^,
        d_img_tokens^, d_txt_tokens^, d_vec^,
        d_timestep^, d_guidance^, d_vector^,
        nonfinite,
        st_d_a^, st_d_b^,
    )
