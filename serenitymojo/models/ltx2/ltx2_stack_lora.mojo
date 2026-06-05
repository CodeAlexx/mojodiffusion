# models/ltx2/ltx2_stack_lora.mojo
#
# LTX-2 (video DiT) FULL STACK *WITH LoRA*, BLOCK-SWAP OFFLOAD: forward (saving
# per-block activations) + full-depth backward (training). Streams 48 identical
# transformer_blocks one at a time via TurboPlannedLoader and drives the proven
# per-block fwd/bwd in models/ltx2/ltx2_block.mojo (DO NOT MODIFY that file).
#
# WHAT DIFFERS FROM CHROMA (the offload template):
#   (1) BLOCK KIND. LTX-2 has NO double/single split — all 48 blocks are the
#       same BasicAVTransformerBlock (video-only LoRA surface attn1 self-attn +
#       FFN). plan = build_ltx2_block_plan(48) (offload/ltx2_plan.mojo).
#   (2) MODULATION SOURCE. Each block has a LEARNABLE scale_shift_table[9,D]
#       (frozen in LoRA training; rows 0-5 = shift/scale/gate for msa+mlp). The
#       global adaln_single conditioning network turns sigma -> a 6*D modulation
#       delta that is ADDED on top of the per-block table (LTX-2 AdaLN-single).
#       The combined modvec = sst_rows[r] + adaln_delta[r] per block. Both the
#       table AND the adaln_single network are FROZEN (LoRA scope) so all
#       returned modvec grads are DISCARDED.
#   (3) LEGACY NARROW LoRA TARGET SET:
#       attn1.to_q, attn1.to_k, attn1.to_v, attn1.to_out.0 -> 4 adapters/block,
#       4*48 = 192 total. Flat-indexed bi*4 + {0:q,1:k,2:v,3:o}.
#       This is NOT musubi's production T2V preset, which targets all AV
#       attention modules. See training/ltx2_av_training_readiness.mojo.
#   (4) RoPE is SPLIT type (rope_halfsplit). cos/sin built via
#       models/dit/ltx2_rope.build_ltx2_rope, flattened to [S*H, Dh//2] in the
#       split layout the block expects (NOT interleaved).
#
# BF16 contract (flame-core): weights/activations/saved-acts BF16 (the block does
#   to_host_bf16/from_host_bf16 internally); modvecs + returned LoRA grads F32 for
#   the optimizer. No F32 carrier detours through the compute path.
#
# Mojo 0.26.x+: def not fn; comptime not alias; move-only Tensor; host carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

# proven per-block LoRA block (DO NOT MODIFY ltx2_block.mojo).
from serenitymojo.models.ltx2.ltx2_block import (
    LTX2ModVecs, LTX2BlockWeights, LTX2BlockSaved, LTX2Lora,
    LTX2BlockForward, LTX2BlockGrads,
    ltx2_block_forward, ltx2_block_backward,
)
from serenitymojo.models.ltx2.weights import (
    LTX2StackBase, LTX2BlockOffloadWeights, load_ltx2_block_offload_from_block,
)

# reuse the proven host-list LoRA carrier + AdamW + PEFT save.
from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_save import NamedLora, save_lora_peft

from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.activations import silu

comptime TArc = ArcPointer[Tensor]

# 4 LoRA slots per block: to_q, to_k, to_v, to_out.0.
comptime LTX2_SLOTS = 4


# ── host helpers ─────────────────────────────────────────────────────────────
def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ── LoRA carrier: 192 adapters flat-indexed (bi*4 + slot) ────────────────────
struct LTX2LoraSet(Copyable, Movable):
    var ad: List[LoraAdapter]
    var num_layers: Int
    var rank: Int

    def __init__(out self, var ad: List[LoraAdapter], num_layers: Int, rank: Int):
        self.ad = ad^
        self.num_layers = num_layers
        self.rank = rank


def _lora_base(bi: Int) -> Int:
    return bi * LTX2_SLOTS


def _make_lora_adapter(rank: Int, alpha: Float32, in_f: Int, out_f: Int, seed: UInt64) -> LoraAdapter:
    var scale = alpha / Float32(rank)
    return LoraAdapter(
        _randn(rank * in_f, seed, 0.01),   # A small randn
        _zeros(out_f * rank),              # B = 0 (PEFT identity at step 0)
        rank, in_f, out_f, scale,
        _zeros(rank * in_f), _zeros(rank * in_f),    # ma / va
        _zeros(out_f * rank), _zeros(out_f * rank),  # mb / vb
    )


# Build the full LoRA set: 4 adapters (q,k,v,o), each [in=D,out=D], per block.
def build_ltx2_lora_set(num_layers: Int, D: Int, rank: Int, alpha: Float32) -> LTX2LoraSet:
    var ad = List[LoraAdapter]()
    var seed = UInt64(7000)
    for _ in range(num_layers):
        ad.append(_make_lora_adapter(rank, alpha, D, D, seed)); seed += 1  # to_q
        ad.append(_make_lora_adapter(rank, alpha, D, D, seed)); seed += 1  # to_k
        ad.append(_make_lora_adapter(rank, alpha, D, D, seed)); seed += 1  # to_v
        ad.append(_make_lora_adapter(rank, alpha, D, D, seed)); seed += 1  # to_out.0
    return LTX2LoraSet(ad^, num_layers, rank)


def total_ltx2_adapters(set: LTX2LoraSet) -> Int:
    return set.num_layers * LTX2_SLOTS


# Build the 4 LTX2Lora views (block-facing struct) for block bi from the carrier.
def _block_loras(lora: LTX2LoraSet, bi: Int) -> List[LTX2Lora]:
    var base = _lora_base(bi)
    var out = List[LTX2Lora]()
    for s in range(LTX2_SLOTS):
        ref a = lora.ad[base + s]
        out.append(LTX2Lora(a.a.copy(), a.b.copy(), a.rank, a.in_f, a.out_f, a.scale))
    return out^


# ── collected LoRA grads (flat, parallel to LTX2LoraSet) ─────────────────────
struct LTX2LoraGradSet(Movable):
    var d_a: List[List[Float32]]   # [n_adapters][rank*in]
    var d_b: List[List[Float32]]   # [n_adapters][out*rank]
    var d_in: List[Float32]        # block-input grad at stack entry (exercised)
    var nonfinite_lora_grads: Int

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_in: List[Float32], nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_in = d_in^
        self.nonfinite_lora_grads = nonfinite_lora_grads


# ── adaln_single conditioning: sigma -> 6*D modulation delta (host F32) ──────
# adaln_single = timestep_embedder(sigma) [256] -> silu -> linear[6*D,256].
# In LTX-2 this delta is shared across all blocks (AdaLN-single) and ADDED on
# top of each block's scale_shift_table rows. Built once per step.
#   layout of the 6*D output: [shift_msa, scale_msa, gate_msa,
#                              shift_mlp, scale_mlp, gate_mlp] each [D].
def ltx2_adaln_delta(
    base: LTX2StackBase, timestep_embed_256: List[Float32], ctx: DeviceContext,
) raises -> List[Float32]:
    # timestep_embed_256 is the [256] sinusoidal+MLP timestep embedding from the
    # trainer (built once per step). We run the two adaln timestep linears then
    # the 6*D projection. silu between matches diffusers AdaLayerNormSingle.
    var emb = Tensor.from_host(timestep_embed_256.copy(), [1, 256], STDtype.BF16, ctx)
    var b1 = Optional[Tensor](base.adaln_lin1_b[].clone(ctx))
    var h1 = linear(emb, base.adaln_lin1_w[], b1, ctx)        # [1,256]
    var h1a = silu(h1, ctx)
    var b2 = Optional[Tensor](base.adaln_lin2_b[].clone(ctx))
    var h2 = linear(h1a, base.adaln_lin2_w[], b2, ctx)        # [1,256]
    var h2a = silu(h2, ctx)
    var bo = Optional[Tensor](base.adaln_out_b[].clone(ctx))
    var out = linear(h2a, base.adaln_out_w[], bo, ctx)        # [1, 6*D]
    return out.to_host(ctx)


# Build per-block combined modvecs: sst_base[r] + adaln_delta[r] for r in 0..5.
def _block_modvecs(h: LTX2BlockOffloadWeights, adaln_delta: List[Float32], D: Int) -> LTX2ModVecs:
    # adaln_delta is [6*D]: rows shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp.
    var d_shift_msa = List[Float32]()
    var d_scale_msa = List[Float32]()
    var d_gate_msa = List[Float32]()
    var d_shift_mlp = List[Float32]()
    var d_scale_mlp = List[Float32]()
    var d_gate_mlp = List[Float32]()
    for c in range(D):
        d_shift_msa.append(adaln_delta[0 * D + c])
        d_scale_msa.append(adaln_delta[1 * D + c])
        d_gate_msa.append(adaln_delta[2 * D + c])
        d_shift_mlp.append(adaln_delta[3 * D + c])
        d_scale_mlp.append(adaln_delta[4 * D + c])
        d_gate_mlp.append(adaln_delta[5 * D + c])
    return LTX2ModVecs(
        _add_lists(h.sst_shift_msa, d_shift_msa^),
        _add_lists(h.sst_scale_msa, d_scale_msa^),
        _add_lists(h.sst_gate_msa, d_gate_msa^),
        _add_lists(h.sst_shift_mlp, d_shift_mlp^),
        _add_lists(h.sst_scale_mlp, d_scale_mlp^),
        _add_lists(h.sst_gate_mlp, d_gate_mlp^),
    )


# LTX2BlockSaved is Movable-only (NOT Copyable) and ltx2_block.mojo is frozen, so
# it cannot live in a List (List[T] requires T: Copyable). We flatten each block's
# 20 BF16 activation fields into a List[List[BFloat16]] (Copyable) for the tape and
# rebuild the LTX2BlockSaved struct verbatim in backward. Field order MUST match
# the LTX2BlockSaved ctor (ltx2_block.mojo:304).
def _flatten_saved(s: LTX2BlockSaved) -> List[List[BFloat16]]:
    var o = List[List[BFloat16]]()
    o.append(s.hidden.copy()); o.append(s.norm_h.copy()); o.append(s.mod_h.copy())
    o.append(s.q_pre.copy()); o.append(s.k_pre.copy())
    o.append(s.q_rms.copy()); o.append(s.k_rms.copy()); o.append(s.v.copy())
    o.append(s.q_rope.copy()); o.append(s.k_rope.copy()); o.append(s.att_flat.copy())
    o.append(s.gl.copy()); o.append(s.gates.copy())
    o.append(s.att_g.copy()); o.append(s.hs.copy())
    o.append(s.norm_ff.copy()); o.append(s.mod_ff.copy())
    o.append(s.h1.copy()); o.append(s.h1g.copy()); o.append(s.ff.copy())
    return o^


def _rebuild_saved(f: List[List[BFloat16]]) -> LTX2BlockSaved:
    return LTX2BlockSaved(
        f[0].copy(), f[1].copy(), f[2].copy(),
        f[3].copy(), f[4].copy(),
        f[5].copy(), f[6].copy(), f[7].copy(),
        f[8].copy(), f[9].copy(), f[10].copy(),
        f[11].copy(), f[12].copy(),
        f[13].copy(), f[14].copy(),
        f[15].copy(), f[16].copy(),
        f[17].copy(), f[18].copy(), f[19].copy(),
    )


# ── forward tape ─────────────────────────────────────────────────────────────
struct LTX2StackForward(Movable):
    var out: List[Float32]                    # [N, out_ch] de-patchified output
    var saved: List[List[List[BFloat16]]]     # num_layers x 20 BF16 act fields
    var modvecs: List[List[Float32]]          # num_layers x [6*D] (combined modvecs flat)
    var x_in: List[BFloat16]                  # [N, D] post-patchify BF16 block-stream input

    def __init__(
        out self,
        var out: List[Float32], var saved: List[List[List[BFloat16]]],
        var modvecs: List[List[Float32]], var x_in: List[BFloat16],
    ):
        self.out = out^
        self.saved = saved^
        self.modvecs = modvecs^
        self.x_in = x_in^


# flatten an LTX2ModVecs [6*D] for the tape (so backward reconstructs verbatim).
def _flatten_modvecs(mv: LTX2ModVecs) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(mv.shift_msa)):
        o.append(mv.shift_msa[i])
    for i in range(len(mv.scale_msa)):
        o.append(mv.scale_msa[i])
    for i in range(len(mv.gate_msa)):
        o.append(mv.gate_msa[i])
    for i in range(len(mv.shift_mlp)):
        o.append(mv.shift_mlp[i])
    for i in range(len(mv.scale_mlp)):
        o.append(mv.scale_mlp[i])
    for i in range(len(mv.gate_mlp)):
        o.append(mv.gate_mlp[i])
    return o^


def _modvecs_from_flat(flat: List[Float32], D: Int) -> LTX2ModVecs:
    var shift_msa = List[Float32](); var scale_msa = List[Float32](); var gate_msa = List[Float32]()
    var shift_mlp = List[Float32](); var scale_mlp = List[Float32](); var gate_mlp = List[Float32]()
    for c in range(D):
        shift_msa.append(flat[0 * D + c])
        scale_msa.append(flat[1 * D + c])
        gate_msa.append(flat[2 * D + c])
        shift_mlp.append(flat[3 * D + c])
        scale_mlp.append(flat[4 * D + c])
        gate_mlp.append(flat[5 * D + c])
    return LTX2ModVecs(shift_msa^, scale_msa^, gate_msa^, shift_mlp^, scale_mlp^, gate_mlp^)


# ═════════════════════════════════════════════════════════════════════════════
# FULL FORWARD WITH LoRA, BLOCK-SWAP OFFLOAD.
#   Inputs: x_tokens [N, in_ch] (post-patchify-ready latent tokens), the per-step
#   adaln_delta [6*D] (from ltx2_adaln_delta), cos/sin split rope flats.
#   Streams all num_layers blocks one at a time. The frozen patchify_proj /
#   proj_out are applied at the stack boundary.
# ═════════════════════════════════════════════════════════════════════════════
def ltx2_stack_lora_forward_offload[
    H: Int, Dh: Int, S: Int
](
    x_tokens: List[Float32],
    adaln_delta: List[Float32],
    base: LTX2StackBase,
    mut loader: TurboPlannedLoader, lora: LTX2LoraSet,
    cos: List[Float32], sin: List[Float32],
    D: Int, FF: Int, in_ch: Int, out_ch: Int, eps: Float32,
    use_lora: Bool,
    ctx: DeviceContext,
) raises -> LTX2StackForward:
    var num_layers = lora.num_layers

    loader.prefetch_with_ctx(0, ctx)

    # split-rope tables: [S*H, Dh//2] (built by ltx2_rope, already split layout).
    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    # patchify_proj (frozen base linear): [N,in_ch] -> [N,D]
    var pb = Optional[Tensor](base.patchify_b[].clone(ctx))
    var x = linear(
        Tensor.from_host(x_tokens.copy(), [S, in_ch], STDtype.BF16, ctx),
        base.patchify_w[], pb, ctx,
    ).to_host_bf16(ctx)
    var x_in = x.copy()

    var saved = List[List[List[BFloat16]]]()
    var modvecs_flat = List[List[Float32]]()
    for bi in range(num_layers):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var h = load_ltx2_block_offload_from_block(
            handle.block, handle.prefix + String("."), D, H, ctx,
        )
        var mv = _block_modvecs(h, adaln_delta, D)
        var bl = _block_loras(lora, bi)
        var fwd = ltx2_block_forward[H, Dh, S](
            x.copy(), h.weights.copy(), mv, cos_t, sin_t,
            bl[0], bl[1], bl[2], bl[3], use_lora,
            D, FF, eps, ctx,
        )
        saved.append(_flatten_saved(fwd.saved))
        modvecs_flat.append(_flatten_modvecs(mv))
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)

    # proj_out (frozen base): [N,D] -> [N,out_ch]
    var ob = Optional[Tensor](base.proj_out_b[].clone(ctx))
    var out = linear(
        Tensor.from_host_bf16(x.copy(), [S, D], ctx),
        base.proj_out_w[], ob, ctx,
    ).to_host(ctx)

    return LTX2StackForward(out^, saved^, modvecs_flat^, x_in^)


# ═════════════════════════════════════════════════════════════════════════════
# FULL BACKWARD WITH LoRA, BLOCK-SWAP OFFLOAD (REVERSE block stream).
#   Frozen-base scope: per-block modvec grads + base weight grads are DISCARDED
#   (the scale_shift_table and adaln_single are frozen). Only LoRA d_A/d_B
#   collected. proj_out / patchify base arms are exercised (grads discarded).
# ═════════════════════════════════════════════════════════════════════════════
def ltx2_stack_lora_backward_offload[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    x_tokens: List[Float32],
    base: LTX2StackBase,
    mut loader: TurboPlannedLoader, lora: LTX2LoraSet,
    cos: List[Float32], sin: List[Float32],
    saved: LTX2StackForward,
    D: Int, FF: Int, in_ch: Int, out_ch: Int, eps: Float32,
    use_lora: Bool,
    ctx: DeviceContext,
) raises -> LTX2LoraGradSet:
    var num_layers = lora.num_layers

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    var n_adapters = total_ltx2_adapters(lora)
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a_flat.append(List[Float32]()); d_b_flat.append(List[Float32]())
    var nonfinite = 0

    # ── proj_out backward (frozen; grads discarded, arm exercised) ──
    var lbo = linear_backward(
        Tensor.from_host(d_out.copy(), [S, out_ch], STDtype.BF16, ctx),
        Tensor.from_host_bf16(saved.x_in.copy(), [S, D], ctx),  # placeholder x (unused for d_x scale)
        base.proj_out_w[], S, D, out_ch, ctx,
    )
    var d_x = lbo.d_x.to_host(ctx)   # [N,D] grad into the last block output

    # ── block-stream backward (REVERSE; LoRA; streamed weights) ──
    var bi = num_layers - 1
    while bi >= 0:
        var handle = loader.await_block(bi, ctx)
        if bi > 0:
            loader.prefetch_with_ctx(bi - 1, ctx)
        var h = load_ltx2_block_offload_from_block(
            handle.block, handle.prefix + String("."), D, H, ctx,
        )
        var mv = _modvecs_from_flat(saved.modvecs[bi].copy(), D)
        var bl = _block_loras(lora, bi)
        var blk_saved = _rebuild_saved(saved.saved[bi])
        var bg = ltx2_block_backward[H, Dh, S](
            d_x.copy(), h.weights.copy(), mv, blk_saved, cos_t, sin_t,
            bl[0], bl[1], bl[2], bl[3], use_lora,
            D, FF, eps, ctx,
        )
        d_x = bg.d_hidden.copy()
        var base_idx = _lora_base(bi)
        if use_lora:
            # slot 0:q, 1:k, 2:v, 3:o
            d_a_flat[base_idx + 0] = bg.d_lq_a.copy(); d_b_flat[base_idx + 0] = bg.d_lq_b.copy()
            d_a_flat[base_idx + 1] = bg.d_lk_a.copy(); d_b_flat[base_idx + 1] = bg.d_lk_b.copy()
            d_a_flat[base_idx + 2] = bg.d_lv_a.copy(); d_b_flat[base_idx + 2] = bg.d_lv_b.copy()
            d_a_flat[base_idx + 3] = bg.d_lo_a.copy(); d_b_flat[base_idx + 3] = bg.d_lo_b.copy()
            for s in range(LTX2_SLOTS):
                nonfinite += _nonfinite(d_a_flat[base_idx + s]) + _nonfinite(d_b_flat[base_idx + s])
        # modvec grads + base weight grads DISCARDED (frozen).
        loader.mark_active_block_done(ctx)
        bi -= 1

    # ── patchify backward (frozen; grads discarded, arm exercised) ──
    var lbi = linear_backward(
        Tensor.from_host(d_x.copy(), [S, D], STDtype.BF16, ctx),
        Tensor.from_host(x_tokens.copy(), [S, in_ch], STDtype.BF16, ctx),
        base.patchify_w[], S, in_ch, D, ctx,
    )
    var d_in = lbi.d_x.to_host(ctx)

    return LTX2LoraGradSet(d_a_flat^, d_b_flat^, d_in^, nonfinite)


# ── optimizer + save ─────────────────────────────────────────────────────────
def ltx2_lora_adamw_step(
    mut lora: LTX2LoraSet, grads: LTX2LoraGradSet, t: Int, lr: Float32, ctx: DeviceContext,
) raises:
    var n = total_ltx2_adapters(lora)
    for i in range(n):
        var g = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(lora.ad[i], g, t, lr, ctx)


# PEFT-keyed save. Module prefix mirrors musubi lora_ltx2.py naming:
#   transformer_blocks.{bi}.attn1.{to_q,to_k,to_v,to_out.0}
def save_ltx2_lora(lora: LTX2LoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var slot_names = List[String]()
    slot_names.append(String("attn1.to_q"))
    slot_names.append(String("attn1.to_k"))
    slot_names.append(String("attn1.to_v"))
    slot_names.append(String("attn1.to_out.0"))
    var named = List[NamedLora]()
    for bi in range(lora.num_layers):
        var base = _lora_base(bi)
        for s in range(LTX2_SLOTS):
            var prefix = String("transformer_blocks.") + String(bi) + String(".") + slot_names[s]
            named.append(NamedLora(prefix, lora.ad[base + s].copy()))
    return save_lora_peft(named, path, ctx)
