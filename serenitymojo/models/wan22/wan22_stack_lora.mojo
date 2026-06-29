# serenitymojo/models/wan22/wan22_stack_lora.mojo
#
# Wan2.2-T2V FULL BLOCK STACK *WITH LoRA*, BLOCK-SWAP OFFLOAD.
# forward (saving ckpt-inputs) + backward (training). Mirrors the chroma/flux
# offload stack pattern: one block streamed at a time via TurboPlannedLoader.
#
# ARCHITECTURE DELTAS vs. Chroma/Flux:
#   - SINGLE IMAGE STREAM. Only x [S,dim] — no two-stream join. Context [TXT,dim]
#     enters via cross-attn only.
#   - PER-TOKEN AdaLN (not per-channel). Modulation vectors are [S,dim], computed
#     from time_embedding -> time_projection -> per-block modulation add:
#       e0 = time_projection(SiLU(time_embedding(sin_emb(t)))) [S,6*dim]
#            reshaped [S,6,dim]
#       block_mod = blocks.{i}.modulation [1,6,dim] (learnable, frozen for LoRA)
#       e = (block_mod + e0)  [S,6,dim]  then chunk(6) -> shift_sa, scale_sa,
#           gate_sa, shift_ffn, scale_ffn, gate_ffn  (each [S,dim])
#   - head.modulation [1,2,dim] + e_head [1,S,dim] -> head shift/scale [1,S,dim]
#   - LoRA targets: 8 per block — self_attn.{q,k,v,o} + cross_attn.{q,k,v,o}
#     (wan22.rs:199-206). in=out=dim each. 40 blocks -> 320 adapters.
#   - CHECKPOINT KEYS: blocks.{i}.self_attn.q.weight etc. (NOT fused).
#
# CARRIER DESIGN: Wan22LoraSet has a flat List[LoraAdapter], 8 slots per block.
# Slot order per block: sa_q, sa_k, sa_v, sa_o, ca_q, ca_k, ca_v, ca_o.
#
# MODULATION PROVIDER: the frozen time embedding chain + per-block modulation
# weights. In training scope the modulation chain IS frozen (LoRA only adapts
# the 8 attention projections). The per-block `modulation` tensor is loaded from
# the streamed block at forward time, not from a resident approximator.
#
# DTYPE: native BF16 carriers (the block uses BF16 weights + saved activations
# in BF16). Grads stay F32 in LoraAdapter (host master precision).
#
# Mojo 0.26.x+: def not fn; Tensor move-only; TArc carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt, sin as _fsin, cos as _fcos

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.activations import silu, gelu
from serenitymojo.ops.tensor_algebra import (
    reshape, add, mul, mul_scalar, slice, concat, add_scalar,
)
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm_backward import layer_norm_backward
from serenitymojo.ops.elementwise_backward import modulate_backward

from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.wan22.wan22_block import (
    WanModVecs, WanBlockWeights, WanSaved, WanBlockForward,
    WanBlockLora, WanBlockLoraGrads,
    WanBlockDirectProjectionWeights, WanBlockDirectLycoris,
    WanBlockDirectLycorisGrads, WanDirectProjectionGrad,
    WAN_DIRECT_ALGO_DORA, WAN_DIRECT_ALGO_OFT,
    wan22_block_lora_forward, wan22_block_lora_backward,
    wan22_block_direct_lycoris_forward, wan22_block_direct_lycoris_backward,
)
from serenitymojo.models.klein.lora_block import KleinLoraGrads
from serenitymojo.training.dora_adapter import DoRAGrads
from serenitymojo.training.oft_onetrainer import OFTOTGrads
from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_save import NamedLora, save_lora_peft
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectDoRAGrads, FlatDirectOFTSet, FlatDirectOFTGrads,
)
from serenitymojo.models.wan22.wan22_direct_lycoris_stack import (
    empty_wan22_direct_dora_set, empty_wan22_direct_oft_set,
    wan22_direct_dora_append_block_weights, build_wan22_direct_oft_set,
    wan22_direct_dora_zero_grads, wan22_direct_dora_scatter_slot_grad,
    wan22_direct_oft_zero_grads, wan22_direct_oft_scatter_slot_grad,
)


comptime TArc = ArcPointer[Tensor]

# LoRA slots per block (8 attention projections).
comptime WAN_SLOTS = 8
# Slot indices within a block's 8-slot window.
comptime W_SA_Q = 0
comptime W_SA_K = 1
comptime W_SA_V = 2
comptime W_SA_O = 3
comptime W_CA_Q = 4
comptime W_CA_K = 5
comptime W_CA_V = 6
comptime W_CA_O = 7


# ── host helpers ──────────────────────────────────────────────────────────────
def _zeros_f32(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _randn_f32(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ── upload helpers ────────────────────────────────────────────────────────────
def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _t16(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


def _t_like(
    vals: List[Float32], var shape: List[Int], ref_weight: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var t = _t(vals, shape^, ctx)
    if t.dtype() == ref_weight.dtype():
        return t^
    return cast_tensor(t, ref_weight.dtype(), ctx)


def _cast_like(x: Tensor, ref_weight: Tensor, ctx: DeviceContext) raises -> Tensor:
    if x.dtype() == ref_weight.dtype():
        return x.clone(ctx)
    return cast_tensor(x, ref_weight.dtype(), ctx)


def _clone_t(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ── RESIDENT (frozen) stack base: embeddings + head ──────────────────────────
struct Wan22StackBase(Movable):
    # patch embedding (patchify output -> dim)
    var pe_w: TArc           # [dim, in_patch_dim] = [5120, 64]
    var pe_b: TArc           # [dim]
    # text embedding MLP (T5 context -> dim)
    var te0_w: TArc          # [dim, text_dim] = [5120, 4096]
    var te0_b: TArc          # [dim]
    var te2_w: TArc          # [dim, dim]
    var te2_b: TArc          # [dim]
    # time embedding MLP (sinusoidal -> dim)
    var tme0_w: TArc         # [dim, freq_dim] = [5120, 256]
    var tme0_b: TArc         # [dim]
    var tme2_w: TArc         # [dim, dim]
    var tme2_b: TArc         # [dim]
    # time projection MLP (SiLU -> Linear(dim -> 6*dim))
    var tp1_w: TArc          # [6*dim, dim] = [30720, 5120]
    var tp1_b: TArc          # [6*dim]
    # head
    var hh_w: TArc           # [out_ch, dim] = [64, 5120]
    var hh_b: TArc           # [out_ch]
    var head_mod: TArc       # [1, 2, dim]

    def __init__(
        out self,
        var pe_w: TArc, var pe_b: TArc,
        var te0_w: TArc, var te0_b: TArc, var te2_w: TArc, var te2_b: TArc,
        var tme0_w: TArc, var tme0_b: TArc, var tme2_w: TArc, var tme2_b: TArc,
        var tp1_w: TArc, var tp1_b: TArc,
        var hh_w: TArc, var hh_b: TArc, var head_mod: TArc,
    ):
        self.pe_w = pe_w^
        self.pe_b = pe_b^
        self.te0_w = te0_w^
        self.te0_b = te0_b^
        self.te2_w = te2_w^
        self.te2_b = te2_b^
        self.tme0_w = tme0_w^
        self.tme0_b = tme0_b^
        self.tme2_w = tme2_w^
        self.tme2_b = tme2_b^
        self.tp1_w = tp1_w^
        self.tp1_b = tp1_b^
        self.hh_w = hh_w^
        self.hh_b = hh_b^
        self.head_mod = head_mod^


# ── LoRA carrier (flat, 8 slots/block) ────────────────────────────────────────
struct Wan22LoraSet(Movable):
    var ad: List[LoraAdapter]  # flat: block bi -> ad[bi*8 + slot]
    var num_blocks: Int
    var rank: Int

    def __init__(out self, var ad: List[LoraAdapter], num_blocks: Int, rank: Int):
        self.ad = ad^
        self.num_blocks = num_blocks
        self.rank = rank


def _make_adapter(rank: Int, alpha: Float32, in_f: Int, out_f: Int, seed: UInt64) -> LoraAdapter:
    var scale = alpha / Float32(rank)
    return LoraAdapter(
        _randn_f32(rank * in_f, seed, Float32(0.01)),
        _zeros_f32(out_f * rank),
        rank, in_f, out_f, scale,
        _zeros_f32(rank * in_f), _zeros_f32(rank * in_f),
        _zeros_f32(out_f * rank), _zeros_f32(out_f * rank),
    )


def build_wan22_lora_set(
    num_blocks: Int, dim: Int, rank: Int, alpha: Float32,
) -> Wan22LoraSet:
    """Build a full Wan22LoraSet (A=randn, B=0 -> identity at init).
    8 adapters per block: sa_{q,k,v,o} + ca_{q,k,v,o}, all in=out=dim.
    """
    var ad = List[LoraAdapter]()
    var seed = UInt64(9001)
    for _ in range(num_blocks):
        for _ in range(8):
            ad.append(_make_adapter(rank, alpha, dim, dim, seed))
            seed += 1
    return Wan22LoraSet(ad^, num_blocks, rank)


def wan22_total_adapters(lora: Wan22LoraSet) -> Int:
    return lora.num_blocks * WAN_SLOTS


def _block_base(bi: Int) -> Int:
    return bi * WAN_SLOTS


# Build the WanBlockLora struct for block bi from the flat carrier.
def _wan_block_lora_for(lora: Wan22LoraSet, bi: Int) -> WanBlockLora:
    var base = _block_base(bi)
    return WanBlockLora(
        Optional[LoraAdapter](lora.ad[base + W_SA_Q].copy()),
        Optional[LoraAdapter](lora.ad[base + W_SA_K].copy()),
        Optional[LoraAdapter](lora.ad[base + W_SA_V].copy()),
        Optional[LoraAdapter](lora.ad[base + W_SA_O].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_Q].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_K].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_V].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_O].copy()),
    )


def _wan_block_direct_dora_for(dora: FlatDirectDoRASet, bi: Int) -> WanBlockDirectLycoris:
    return WanBlockDirectLycoris(
        WAN_DIRECT_ALGO_DORA, dora.copy(), empty_wan22_direct_oft_set(), _block_base(bi),
    )


def _wan_block_direct_oft_for(oft: FlatDirectOFTSet, bi: Int) -> WanBlockDirectLycoris:
    return WanBlockDirectLycoris(
        WAN_DIRECT_ALGO_OFT, empty_wan22_direct_dora_set(), oft.copy(), _block_base(bi),
    )


# ── LoRA grad carrier ─────────────────────────────────────────────────────────
struct Wan22LoraGradSet(Movable):
    var d_a: List[List[Float32]]   # [num_blocks*8][rank*dim]
    var d_b: List[List[Float32]]   # [num_blocks*8][dim*rank]
    var d_x_tokens: List[Float32]  # [S,dim] input grad (load-bearing arm)
    var d_context: List[Float32]   # [TXT,dim] context input grad
    var nonfinite_lora_grads: Int

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_x_tokens: List[Float32], var d_context: List[Float32],
        nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x_tokens = d_x_tokens^
        self.d_context = d_context^
        self.nonfinite_lora_grads = nonfinite_lora_grads


struct Wan22DirectDoRAGradSet(Movable):
    var grads: FlatDirectDoRAGrads
    var d_x_tokens: List[Float32]
    var d_context: List[Float32]
    var nonfinite_grads: Int

    def __init__(
        out self, var grads: FlatDirectDoRAGrads,
        var d_x_tokens: List[Float32], var d_context: List[Float32],
        nonfinite_grads: Int,
    ):
        self.grads = grads^
        self.d_x_tokens = d_x_tokens^
        self.d_context = d_context^
        self.nonfinite_grads = nonfinite_grads


struct Wan22DirectOFTGradSet(Movable):
    var grads: FlatDirectOFTGrads
    var d_x_tokens: List[Float32]
    var d_context: List[Float32]
    var nonfinite_grads: Int

    def __init__(
        out self, var grads: FlatDirectOFTGrads,
        var d_x_tokens: List[Float32], var d_context: List[Float32],
        nonfinite_grads: Int,
    ):
        self.grads = grads^
        self.d_x_tokens = d_x_tokens^
        self.d_context = d_context^
        self.nonfinite_grads = nonfinite_grads


# ── Forward tape ──────────────────────────────────────────────────────────────
struct Wan22StackForward(Movable):
    var out: List[Float32]                # [S, out_ch] final velocity prediction
    var block_saved: List[WanSaved]       # per-block activations (BF16 host)
    var block_modvecs: List[WanModVecs]   # per-block modulation vectors (F32 host)
    var x_img: List[Float32]              # [S, dim] image tokens before head
    var context_emb: List[Float32]        # [TXT, dim] embedded context (frozen)
    # sinusoidal time embedding (scalar -> [S, dim]) — saved for head backward
    var e_head_f32: List[Float32]         # [S, dim] (F32 host)
    var head_shift: List[Float32]         # [S, dim]
    var head_scale: List[Float32]         # [S, dim]

    def __init__(
        out self,
        var out: List[Float32],
        var block_saved: List[WanSaved],
        var block_modvecs: List[WanModVecs],
        var x_img: List[Float32],
        var context_emb: List[Float32],
        var e_head_f32: List[Float32],
        var head_shift: List[Float32],
        var head_scale: List[Float32],
    ):
        self.out = out^
        self.block_saved = block_saved^
        self.block_modvecs = block_modvecs^
        self.x_img = x_img^
        self.context_emb = context_emb^
        self.e_head_f32 = e_head_f32^
        self.head_shift = head_shift^
        self.head_scale = head_scale^


# ── Sinusoidal timestep embedding (matches wan22_dit.mojo::timestep_embedding) ─
# Produces F32 [S, freq_dim] where each row is the same sin/cos embedding of `t`.
def _sin_embed(t: Float32, S: Int, freq_dim: Int, theta: Float32 = Float32(10000.0)) -> List[Float32]:
    var half = freq_dim // 2
    var out = List[Float32]()
    for _ in range(S):
        for k in range(half):
            var freq = Float32(1.0) / (theta ** (Float32(k) / Float32(half)))
            var arg = t * freq
            out.append(Float32(_fsin(Float64(arg))))
        for k in range(half):
            var freq = Float32(1.0) / (theta ** (Float32(k) / Float32(half)))
            var arg = t * freq
            out.append(Float32(_fcos(Float64(arg))))
    return out^


# ── time feature chain (frozen; produces e0 [S, 6, dim] and e_head [S, dim]) ──
# Returns (e0_flat [S*6*dim], e_head_flat [S*dim]).
def _time_features(
    t_model: Float32, S: Int, dim: Int, freq_dim: Int,
    base: Wan22StackBase, ctx: DeviceContext,
) raises -> List[List[Float32]]:
    # sinusoidal [S, freq_dim]
    var sin_emb = _sin_embed(t_model, S, freq_dim)
    # time_embedding: Linear -> SiLU -> Linear  [S, freq_dim] -> [S, dim]
    var nb0 = Optional[Tensor](base.tme0_b[].clone(ctx))
    var e = linear(
        _t_like(sin_emb^, [S, freq_dim], base.tme0_w[], ctx),
        base.tme0_w[], nb0, ctx,
    )
    e = silu(e, ctx)
    var nb2 = Optional[Tensor](base.tme2_b[].clone(ctx))
    var e2_in = _cast_like(e, base.tme2_w[], ctx)
    e = linear(e2_in, base.tme2_w[], nb2, ctx)   # [S, dim]
    # e_head = F32 copy of e  [S, dim]
    var e_head = cast_tensor(e, STDtype.F32, ctx).to_host(ctx)
    # time_projection: SiLU(e) -> Linear(dim -> 6*dim)
    var e_silu = silu(e, ctx)
    var ntp = Optional[Tensor](base.tp1_b[].clone(ctx))
    var e_proj_in = _cast_like(e_silu, base.tp1_w[], ctx)
    var e0_flat_t = linear(e_proj_in, base.tp1_w[], ntp, ctx)    # [S, 6*dim]
    var e0_f32 = cast_tensor(e0_flat_t, STDtype.F32, ctx).to_host(ctx)
    var res = List[List[Float32]]()
    res.append(e0_f32^)     # index 0: e0_flat F32 [S*6*dim]
    res.append(e_head^)     # index 1: e_head F32 [S*dim]
    return res^


# ── Per-block modulation vectors from e0_flat + blocks.{i}.modulation ─────────
# e0_flat: F32 [S * 6 * dim] from time chain.
# block_mod: F32 [1 * 6 * dim] from the streamed block (cast from BF16).
# Result: WanModVecs (each field is F32 [S*dim]).
def _block_modvecs(
    e0_flat: List[Float32], block_mod_h: List[Float32],
    bi: Int, S: Int, dim: Int,
) -> WanModVecs:
    # e0_flat is [S, 6, dim] in row-major: index [s, j, d] = s*(6*dim) + j*dim + d
    # block_mod_h is [1, 6, dim]: index [0, j, d] = j*dim + d
    # e[s, j, d] = e0_flat[s,j,d] + block_mod_h[j,d]  (broadcast over S).
    var shift_sa = List[Float32]()
    var scale_sa = List[Float32]()
    var gate_sa = List[Float32]()
    var shift_ffn = List[Float32]()
    var scale_ffn = List[Float32]()
    var gate_ffn = List[Float32]()
    for s in range(S):
        for d in range(dim):
            shift_sa.append(e0_flat[s * 6 * dim + 0 * dim + d] + block_mod_h[0 * dim + d])
            scale_sa.append(e0_flat[s * 6 * dim + 1 * dim + d] + block_mod_h[1 * dim + d])
            gate_sa.append(e0_flat[s * 6 * dim + 2 * dim + d] + block_mod_h[2 * dim + d])
            shift_ffn.append(e0_flat[s * 6 * dim + 3 * dim + d] + block_mod_h[3 * dim + d])
            scale_ffn.append(e0_flat[s * 6 * dim + 4 * dim + d] + block_mod_h[4 * dim + d])
            gate_ffn.append(e0_flat[s * 6 * dim + 5 * dim + d] + block_mod_h[5 * dim + d])
    return WanModVecs(shift_sa^, scale_sa^, gate_sa^, shift_ffn^, scale_ffn^, gate_ffn^)


# ── Head modulation (head.modulation [1,2,dim] + e_head [S,dim]) ───────────────
# Returns (head_shift [S,dim], head_scale [S,dim]).
def _head_modvecs(
    head_mod_h: List[Float32], e_head_h: List[Float32], S: Int, dim: Int,
) -> List[List[Float32]]:
    # head_mod_h: [1, 2, dim] -> [2, dim] (first dim=1 is broadcast).
    # head shift: chunk 0 of axis 1; head scale: chunk 1.
    # e_head_h: [S, dim].
    # result[s,d] = head_mod_h[chunk, d] + e_head_h[s, d].
    var shift = List[Float32]()
    var scale = List[Float32]()
    for s in range(S):
        for d in range(dim):
            shift.append(head_mod_h[0 * dim + d] + e_head_h[s * dim + d])
            scale.append(head_mod_h[1 * dim + d] + e_head_h[s * dim + d])
    var res = List[List[Float32]]()
    res.append(shift^)
    res.append(scale^)
    return res^


# ── Load per-block weights from the streamed Block ─────────────────────────────
def _block_f32(block: Block, key: String, ctx: DeviceContext) raises -> List[Float32]:
    if not (key in block):
        raise Error(String("wan22 offload block missing tensor: ") + key)
    return cast_tensor(block[key][], STDtype.F32, ctx).to_host(ctx)


def _wan22_block_weights_from_block(
    block: Block, prefix: String, dim: Int, ffn: Int, hd: Int, ctx: DeviceContext,
) raises -> WanBlockWeights:
    var bp = prefix + "."
    return WanBlockWeights(
        # self-attn weights
        _block_f32(block, bp + String("self_attn.q.weight"), ctx),
        _block_f32(block, bp + String("self_attn.k.weight"), ctx),
        _block_f32(block, bp + String("self_attn.v.weight"), ctx),
        _block_f32(block, bp + String("self_attn.o.weight"), ctx),
        _block_f32(block, bp + String("self_attn.q.bias"), ctx),
        _block_f32(block, bp + String("self_attn.k.bias"), ctx),
        _block_f32(block, bp + String("self_attn.v.bias"), ctx),
        _block_f32(block, bp + String("self_attn.o.bias"), ctx),
        _block_f32(block, bp + String("self_attn.norm_q.weight"), ctx),
        _block_f32(block, bp + String("self_attn.norm_k.weight"), ctx),
        # cross-attn weights
        _block_f32(block, bp + String("cross_attn.q.weight"), ctx),
        _block_f32(block, bp + String("cross_attn.k.weight"), ctx),
        _block_f32(block, bp + String("cross_attn.v.weight"), ctx),
        _block_f32(block, bp + String("cross_attn.o.weight"), ctx),
        _block_f32(block, bp + String("cross_attn.q.bias"), ctx),
        _block_f32(block, bp + String("cross_attn.k.bias"), ctx),
        _block_f32(block, bp + String("cross_attn.v.bias"), ctx),
        _block_f32(block, bp + String("cross_attn.o.bias"), ctx),
        _block_f32(block, bp + String("cross_attn.norm_q.weight"), ctx),
        _block_f32(block, bp + String("cross_attn.norm_k.weight"), ctx),
        # norm3 (affine LN before cross-attn)
        _block_f32(block, bp + String("norm3.weight"), ctx),
        _block_f32(block, bp + String("norm3.bias"), ctx),
        # ffn
        _block_f32(block, bp + String("ffn.0.weight"), ctx),
        _block_f32(block, bp + String("ffn.0.bias"), ctx),
        _block_f32(block, bp + String("ffn.2.weight"), ctx),
        _block_f32(block, bp + String("ffn.2.bias"), ctx),
        dim, ffn, hd, ctx,
    )


def _wan22_direct_attention_weights_from_block(
    block: Block, prefix: String, ctx: DeviceContext,
) raises -> List[List[Float32]]:
    var bp = prefix + "."
    var out = List[List[Float32]]()
    out.append(_block_f32(block, bp + String("self_attn.q.weight"), ctx))
    out.append(_block_f32(block, bp + String("self_attn.k.weight"), ctx))
    out.append(_block_f32(block, bp + String("self_attn.v.weight"), ctx))
    out.append(_block_f32(block, bp + String("self_attn.o.weight"), ctx))
    out.append(_block_f32(block, bp + String("cross_attn.q.weight"), ctx))
    out.append(_block_f32(block, bp + String("cross_attn.k.weight"), ctx))
    out.append(_block_f32(block, bp + String("cross_attn.v.weight"), ctx))
    out.append(_block_f32(block, bp + String("cross_attn.o.weight"), ctx))
    return out^


def _wan22_direct_projection_weights_from_block(
    block: Block, prefix: String, ctx: DeviceContext,
) raises -> WanBlockDirectProjectionWeights:
    var bp = prefix + "."
    return WanBlockDirectProjectionWeights(
        _block_f32(block, bp + String("self_attn.q.weight"), ctx),
        _block_f32(block, bp + String("self_attn.k.weight"), ctx),
        _block_f32(block, bp + String("self_attn.v.weight"), ctx),
        _block_f32(block, bp + String("self_attn.o.weight"), ctx),
        _block_f32(block, bp + String("self_attn.q.bias"), ctx),
        _block_f32(block, bp + String("self_attn.k.bias"), ctx),
        _block_f32(block, bp + String("self_attn.v.bias"), ctx),
        _block_f32(block, bp + String("self_attn.o.bias"), ctx),
        _block_f32(block, bp + String("cross_attn.q.weight"), ctx),
        _block_f32(block, bp + String("cross_attn.k.weight"), ctx),
        _block_f32(block, bp + String("cross_attn.v.weight"), ctx),
        _block_f32(block, bp + String("cross_attn.o.weight"), ctx),
        _block_f32(block, bp + String("cross_attn.q.bias"), ctx),
        _block_f32(block, bp + String("cross_attn.k.bias"), ctx),
        _block_f32(block, bp + String("cross_attn.v.bias"), ctx),
        _block_f32(block, bp + String("cross_attn.o.bias"), ctx),
    )


def build_wan22_direct_dora_set_from_offload(
    mut loader: TurboPlannedLoader, num_blocks: Int, dim: Int,
    rank: Int, alpha: Float32, seed: UInt64, wd_on_out: Bool,
    ctx: DeviceContext,
) raises -> FlatDirectDoRASet:
    var set = empty_wan22_direct_dora_set()
    if num_blocks > 0:
        loader.prefetch_with_ctx(0, ctx)
    for bi in range(num_blocks):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var weights = _wan22_direct_attention_weights_from_block(handle.block, handle.prefix, ctx)
        wan22_direct_dora_append_block_weights(
            set, bi, weights^, dim, rank, alpha,
            seed + UInt64(bi * WAN_SLOTS), wd_on_out,
        )
        loader.mark_active_block_done(ctx)
    return set^


def _scatter_dora_direct_grad(
    mut grads: FlatDirectDoRAGrads, slot: Int, g: WanDirectProjectionGrad,
) raises:
    var dg = DoRAGrads(
        g.d_a.copy(), g.d_b.copy(), g.d_m.copy(), List[Float32](),
    )
    wan22_direct_dora_scatter_slot_grad(grads, slot, dg^)


def _scatter_oft_direct_grad(
    mut grads: FlatDirectOFTGrads, slot: Int, g: WanDirectProjectionGrad,
) raises:
    var og = OFTOTGrads(g.d_vec.copy(), List[Float32]())
    wan22_direct_oft_scatter_slot_grad(grads, slot, og^)


def _nonfinite_dora_projection(g: WanDirectProjectionGrad) -> Int:
    return _nonfinite(g.d_a) + _nonfinite(g.d_b) + _nonfinite(g.d_m)


def _nonfinite_oft_projection(g: WanDirectProjectionGrad) -> Int:
    return _nonfinite(g.d_vec)


# ── Text context embedding (frozen; T5 [TXT, text_dim] -> [TXT, dim]) ─────────
def _embed_context(
    txt_tokens: List[Float32], TXT: Int, text_dim: Int, dim: Int,
    base: Wan22StackBase, ctx: DeviceContext,
) raises -> List[Float32]:
    var nb0 = Optional[Tensor](base.te0_b[].clone(ctx))
    var h = linear(_t_like(txt_tokens, [TXT, text_dim], base.te0_w[], ctx), base.te0_w[], nb0, ctx)
    h = gelu(h, ctx)
    var nb2 = Optional[Tensor](base.te2_b[].clone(ctx))
    var h2_in = _cast_like(h, base.te2_w[], ctx)
    h = linear(h2_in, base.te2_w[], nb2, ctx)   # [TXT, dim]
    return cast_tensor(h, STDtype.F32, ctx).to_host(ctx)


# ── Patchify: [S, in_ch] image tokens -> [S, dim] via patch_embedding linear ──
def _embed_image(
    img_tokens: List[Float32], S: Int, in_ch: Int, dim: Int,
    base: Wan22StackBase, ctx: DeviceContext,
) raises -> List[Float32]:
    var nb = Optional[Tensor](base.pe_b[].clone(ctx))
    var h = linear(_t_like(img_tokens, [S, in_ch], base.pe_w[], ctx), base.pe_w[], nb, ctx)
    return cast_tensor(h, STDtype.F32, ctx).to_host(ctx)


# ── Head final layer (no-affine LN -> modulate -> linear) ─────────────────────
# head_shift/head_scale are [S, dim] F32 host.
def _head_forward(
    x: List[Float32], head_shift: List[Float32], head_scale: List[Float32],
    S: Int, dim: Int, out_ch: Int, eps: Float32,
    base: Wan22StackBase, ctx: DeviceContext,
) raises -> List[Float32]:
    var ones = List[Float32]()
    for _ in range(dim):
        ones.append(Float32(1.0))
    var zeros = List[Float32]()
    for _ in range(dim):
        zeros.append(Float32(0.0))
    from serenitymojo.ops.norm import layer_norm
    var ln_x = layer_norm(
        _t(x, [S, dim], ctx),
        _t(ones^, [dim], ctx), _t(zeros^, [dim], ctx), eps, ctx,
    )   # [S, dim]
    # modulate: x_mod = ln_x * (1 + scale) + shift
    var scale_d = _t(head_scale, [S, dim], ctx)
    var shift_d = _t(head_shift, [S, dim], ctx)
    var sc1 = add_scalar(scale_d, Float32(1.0), ctx)
    var prod = mul(ln_x, sc1, ctx)
    from serenitymojo.ops.tensor_algebra import add as _tadd
    var modulated = _tadd(prod, shift_d, ctx)
    var nb = Optional[Tensor](base.hh_b[].clone(ctx))
    var head_in = _cast_like(modulated, base.hh_w[], ctx)
    var out = linear(head_in, base.hh_w[], nb, ctx)   # [S, out_ch]
    return cast_tensor(out, STDtype.F32, ctx).to_host(ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL FORWARD WITH LoRA, BLOCK-SWAP OFFLOAD.
#   img_tokens: [S, in_ch]  (patchified image latent, pre-patch-embed)
#   txt_tokens: [TXT, text_dim]  (raw T5 embeddings)
#   t_model: Float32 scalar timestep in [0,1]
#   cos/sin: RoPE tables [S, Dh/2] (precomputed 3-axis interleaved)
# ═══════════════════════════════════════════════════════════════════════════════
def wan22_stack_lora_forward_offload[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    t_model: Float32,
    base: Wan22StackBase,
    mut loader: TurboPlannedLoader, lora: Wan22LoraSet,
    cos: List[Float32], sin: List[Float32],
    dim: Int, ffn: Int, in_ch: Int, text_dim: Int, out_ch: Int,
    freq_dim: Int, eps: Float32, ctx: DeviceContext,
) raises -> Wan22StackForward:
    var num_blocks = lora.num_blocks

    loader.prefetch_with_ctx(0, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # ── frozen embeddings ──
    var img = _embed_image(img_tokens, S, in_ch, dim, base, ctx)
    var context_emb = _embed_context(txt_tokens, TXT, text_dim, dim, base, ctx)

    # ── frozen time feature chain (produces e0_flat [S*6*dim] and e_head [S*dim]) ──
    var tfeats = _time_features(t_model, S, dim, freq_dim, base, ctx)
    var e0_flat = tfeats[0].copy()   # [S * 6 * dim]
    var e_head = tfeats[1].copy()    # [S * dim]

    # ── stream 40 WanAttentionBlocks ──
    var block_saved = List[WanSaved]()
    var block_modvecs = List[WanModVecs]()

    for bi in range(num_blocks):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)

        # Load block modulation [1, 6, dim] BF16 -> F32 host.
        var bp = handle.prefix + "."
        var mod_key = bp + String("modulation")
        if not (mod_key in handle.block):
            raise Error(String("wan22 block missing modulation: ") + mod_key)
        var block_mod_t = cast_tensor(handle.block[mod_key][], STDtype.F32, ctx)
        var block_mod_h = block_mod_t.to_host(ctx)   # [1*6*dim]

        # Build per-token modulation vectors.
        var mv = _block_modvecs(e0_flat, block_mod_h, bi, S, dim)

        # Load block weights.
        var w = _wan22_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)

        # LoRA-augmented forward.
        var bl = _wan_block_lora_for(lora, bi)
        var fwd = wan22_block_lora_forward[H, Dh, S, TXT](
            img.copy(), context_emb.copy(), mv, w, bl, cos_t, sin_t, dim, ffn, eps, ctx,
        )

        block_saved.append(fwd.saved.copy())
        block_modvecs.append(mv.copy())
        img = fwd.x_out.copy()
        loader.mark_active_block_done(ctx)

    # ── head modulation + final linear ──
    var head_mod_h = base.head_mod[].to_host(ctx)   # [1*2*dim]
    var hmod = _head_modvecs(head_mod_h, e_head, S, dim)
    var head_shift = hmod[0].copy()
    var head_scale = hmod[1].copy()

    var out = _head_forward(img, head_shift, head_scale, S, dim, out_ch, eps, base, ctx)

    return Wan22StackForward(
        out^, block_saved^, block_modvecs^,
        img^, context_emb^,
        e_head^, head_shift^, head_scale^,
    )


def wan22_stack_direct_dora_forward_offload[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    t_model: Float32,
    base: Wan22StackBase,
    mut loader: TurboPlannedLoader, dora: FlatDirectDoRASet,
    cos: List[Float32], sin: List[Float32],
    dim: Int, ffn: Int, in_ch: Int, text_dim: Int, out_ch: Int,
    freq_dim: Int, eps: Float32, ctx: DeviceContext,
) raises -> Wan22StackForward:
    var num_blocks = len(dora.ad) // WAN_SLOTS

    loader.prefetch_with_ctx(0, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    var img = _embed_image(img_tokens, S, in_ch, dim, base, ctx)
    var context_emb = _embed_context(txt_tokens, TXT, text_dim, dim, base, ctx)

    var tfeats = _time_features(t_model, S, dim, freq_dim, base, ctx)
    var e0_flat = tfeats[0].copy()
    var e_head = tfeats[1].copy()

    var block_saved = List[WanSaved]()
    var block_modvecs = List[WanModVecs]()

    for bi in range(num_blocks):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)

        var bp = handle.prefix + "."
        var mod_key = bp + String("modulation")
        if not (mod_key in handle.block):
            raise Error(String("wan22 block missing modulation: ") + mod_key)
        var block_mod_t = cast_tensor(handle.block[mod_key][], STDtype.F32, ctx)
        var block_mod_h = block_mod_t.to_host(ctx)

        var mv = _block_modvecs(e0_flat, block_mod_h, bi, S, dim)
        var w = _wan22_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)
        var direct_w = _wan22_direct_projection_weights_from_block(handle.block, handle.prefix, ctx)
        var direct = _wan_block_direct_dora_for(dora, bi)
        var fwd = wan22_block_direct_lycoris_forward[H, Dh, S, TXT](
            img.copy(), context_emb.copy(), mv, w, direct_w, direct,
            cos_t, sin_t, dim, ffn, eps, ctx,
        )

        block_saved.append(fwd.saved.copy())
        block_modvecs.append(mv.copy())
        img = fwd.x_out.copy()
        loader.mark_active_block_done(ctx)

    var head_mod_h = base.head_mod[].to_host(ctx)
    var hmod = _head_modvecs(head_mod_h, e_head, S, dim)
    var head_shift = hmod[0].copy()
    var head_scale = hmod[1].copy()
    var out = _head_forward(img, head_shift, head_scale, S, dim, out_ch, eps, base, ctx)

    return Wan22StackForward(
        out^, block_saved^, block_modvecs^,
        img^, context_emb^,
        e_head^, head_shift^, head_scale^,
    )


def wan22_stack_direct_oft_forward_offload[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    t_model: Float32,
    base: Wan22StackBase,
    mut loader: TurboPlannedLoader, oft: FlatDirectOFTSet,
    cos: List[Float32], sin: List[Float32],
    dim: Int, ffn: Int, in_ch: Int, text_dim: Int, out_ch: Int,
    freq_dim: Int, eps: Float32, ctx: DeviceContext,
) raises -> Wan22StackForward:
    var num_blocks = len(oft.ad) // WAN_SLOTS

    loader.prefetch_with_ctx(0, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    var img = _embed_image(img_tokens, S, in_ch, dim, base, ctx)
    var context_emb = _embed_context(txt_tokens, TXT, text_dim, dim, base, ctx)

    var tfeats = _time_features(t_model, S, dim, freq_dim, base, ctx)
    var e0_flat = tfeats[0].copy()
    var e_head = tfeats[1].copy()

    var block_saved = List[WanSaved]()
    var block_modvecs = List[WanModVecs]()

    for bi in range(num_blocks):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)

        var bp = handle.prefix + "."
        var mod_key = bp + String("modulation")
        if not (mod_key in handle.block):
            raise Error(String("wan22 block missing modulation: ") + mod_key)
        var block_mod_t = cast_tensor(handle.block[mod_key][], STDtype.F32, ctx)
        var block_mod_h = block_mod_t.to_host(ctx)

        var mv = _block_modvecs(e0_flat, block_mod_h, bi, S, dim)
        var w = _wan22_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)
        var direct_w = _wan22_direct_projection_weights_from_block(handle.block, handle.prefix, ctx)
        var direct = _wan_block_direct_oft_for(oft, bi)
        var fwd = wan22_block_direct_lycoris_forward[H, Dh, S, TXT](
            img.copy(), context_emb.copy(), mv, w, direct_w, direct,
            cos_t, sin_t, dim, ffn, eps, ctx,
        )

        block_saved.append(fwd.saved.copy())
        block_modvecs.append(mv.copy())
        img = fwd.x_out.copy()
        loader.mark_active_block_done(ctx)

    var head_mod_h = base.head_mod[].to_host(ctx)
    var hmod = _head_modvecs(head_mod_h, e_head, S, dim)
    var head_shift = hmod[0].copy()
    var head_scale = hmod[1].copy()
    var out = _head_forward(img, head_shift, head_scale, S, dim, out_ch, eps, base, ctx)

    return Wan22StackForward(
        out^, block_saved^, block_modvecs^,
        img^, context_emb^,
        e_head^, head_shift^, head_scale^,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FULL BACKWARD WITH LoRA, BLOCK-SWAP OFFLOAD (REVERSE block stream).
# The frozen modulation chain grads (d_e0, d_block_mod) are DISCARDED.
# Only LoRA d_A/d_B are collected for the optimizer.
# ═══════════════════════════════════════════════════════════════════════════════
def wan22_stack_lora_backward_offload[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: Wan22StackBase,
    mut loader: TurboPlannedLoader, lora: Wan22LoraSet,
    cos: List[Float32], sin: List[Float32],
    saved: Wan22StackForward,
    dim: Int, ffn: Int, in_ch: Int, text_dim: Int, out_ch: Int,
    freq_dim: Int, eps: Float32, ctx: DeviceContext,
) raises -> Wan22LoraGradSet:
    var num_blocks = lora.num_blocks
    var n_adapters = wan22_total_adapters(lora)

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # ── head backward (proj_out -> modulate -> LN_no_affine) ──
    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.norm_backward import layer_norm_backward
    from serenitymojo.ops.elementwise_backward import modulate_backward

    var ones = List[Float32]()
    for _ in range(dim):
        ones.append(Float32(1.0))
    var zeros = List[Float32]()
    for _ in range(dim):
        zeros.append(Float32(0.0))
    var ln_x_img = layer_norm(
        _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), _t(zeros^, [dim], ctx), eps, ctx,
    ).to_host(ctx)
    # modulate: x_mod = ln_x * (1+scale) + shift
    var scale_d = _t(saved.head_scale.copy(), [S, dim], ctx)
    var shift_d = _t(saved.head_shift.copy(), [S, dim], ctx)
    from serenitymojo.ops.tensor_algebra import add_scalar as _add_scalar
    var sc1 = _add_scalar(scale_d, Float32(1.0), ctx)
    var ln_x_t = _t(ln_x_img.copy(), [S, dim], ctx)
    var modulated = mul(ln_x_t, sc1, ctx)
    modulated = add(modulated, shift_d, ctx)
    # linear backward through head.head
    var lbh = linear_backward(
        _t_like(d_out, [S, out_ch], base.hh_w[], ctx),
        _cast_like(modulated, base.hh_w[], ctx), base.hh_w[],
        S, dim, out_ch, ctx,
    )
    var d_modulated = lbh.d_x.to_host(ctx)
    # modulate backward
    var mbh = modulate_backward(
        _t(d_modulated, [S, dim], ctx), _t(ln_x_img^, [S, dim], ctx),
        _t(saved.head_scale.copy(), [S, dim], ctx), ctx,
    )
    var d_ln_img = mbh.d_x.to_host(ctx)
    # LN_no_affine backward
    var lnbh = layer_norm_backward(
        _t(d_ln_img, [S, dim], ctx), _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), eps, ctx,
    )
    var d_x_img = lnbh.d_x.to_host(ctx)   # [S, dim] — enters last block output

    # ── init grad accumulators ──
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    var nonfinite = 0

    # ── stream blocks in REVERSE ──
    var bi = num_blocks - 1
    while bi >= 0:
        var block_idx = bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)

        var mv = saved.block_modvecs[bi].copy()
        var w = _wan22_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)
        var bl = _wan_block_lora_for(lora, bi)

        var bg = wan22_block_lora_backward[H, Dh, S, TXT](
            d_x_img.copy(), mv, w, bl, saved.block_saved[bi],
            cos_t, sin_t, dim, ffn, eps, ctx,
        )

        d_x_img = bg.base.d_x.copy()
        # context grad: accumulated (cross-attn; frozen context_emb discarded)

        # Scatter LoRA grads into flat arrays.
        var bbase = _block_base(bi)
        d_a_flat[bbase + W_SA_Q] = bg.sa_q_da.copy()
        d_b_flat[bbase + W_SA_Q] = bg.sa_q_db.copy()
        d_a_flat[bbase + W_SA_K] = bg.sa_k_da.copy()
        d_b_flat[bbase + W_SA_K] = bg.sa_k_db.copy()
        d_a_flat[bbase + W_SA_V] = bg.sa_v_da.copy()
        d_b_flat[bbase + W_SA_V] = bg.sa_v_db.copy()
        d_a_flat[bbase + W_SA_O] = bg.sa_o_da.copy()
        d_b_flat[bbase + W_SA_O] = bg.sa_o_db.copy()
        d_a_flat[bbase + W_CA_Q] = bg.ca_q_da.copy()
        d_b_flat[bbase + W_CA_Q] = bg.ca_q_db.copy()
        d_a_flat[bbase + W_CA_K] = bg.ca_k_da.copy()
        d_b_flat[bbase + W_CA_K] = bg.ca_k_db.copy()
        d_a_flat[bbase + W_CA_V] = bg.ca_v_da.copy()
        d_b_flat[bbase + W_CA_V] = bg.ca_v_db.copy()
        d_a_flat[bbase + W_CA_O] = bg.ca_o_da.copy()
        d_b_flat[bbase + W_CA_O] = bg.ca_o_db.copy()

        nonfinite += _nonfinite(bg.sa_q_da) + _nonfinite(bg.sa_q_db)
        nonfinite += _nonfinite(bg.sa_k_da) + _nonfinite(bg.sa_k_db)
        nonfinite += _nonfinite(bg.sa_v_da) + _nonfinite(bg.sa_v_db)
        nonfinite += _nonfinite(bg.sa_o_da) + _nonfinite(bg.sa_o_db)
        nonfinite += _nonfinite(bg.ca_q_da) + _nonfinite(bg.ca_q_db)
        nonfinite += _nonfinite(bg.ca_k_da) + _nonfinite(bg.ca_k_db)
        nonfinite += _nonfinite(bg.ca_v_da) + _nonfinite(bg.ca_v_db)
        nonfinite += _nonfinite(bg.ca_o_da) + _nonfinite(bg.ca_o_db)

        loader.mark_active_block_done(ctx)
        bi -= 1

    # ── input projection backward (frozen; grads discarded) ──
    var lbi = linear_backward(
        _t_like(d_x_img, [S, dim], base.pe_w[], ctx),
        _t_like(img_tokens, [S, in_ch], base.pe_w[], ctx), base.pe_w[],
        S, in_ch, dim, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    # Text context backward: use zero grad (context is frozen — no backward needed).
    var d_txt_tokens = _zeros_f32(TXT * text_dim)

    return Wan22LoraGradSet(
        d_a_flat^, d_b_flat^,
        d_img_tokens^, d_txt_tokens^,
        nonfinite,
    )


def wan22_stack_direct_dora_backward_offload[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: Wan22StackBase,
    mut loader: TurboPlannedLoader, dora: FlatDirectDoRASet,
    cos: List[Float32], sin: List[Float32],
    saved: Wan22StackForward,
    dim: Int, ffn: Int, in_ch: Int, text_dim: Int, out_ch: Int,
    freq_dim: Int, eps: Float32, ctx: DeviceContext,
) raises -> Wan22DirectDoRAGradSet:
    var num_blocks = len(dora.ad) // WAN_SLOTS

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.norm_backward import layer_norm_backward
    from serenitymojo.ops.elementwise_backward import modulate_backward

    var ones = List[Float32]()
    for _ in range(dim):
        ones.append(Float32(1.0))
    var zeros = List[Float32]()
    for _ in range(dim):
        zeros.append(Float32(0.0))
    var ln_x_img = layer_norm(
        _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), _t(zeros^, [dim], ctx), eps, ctx,
    ).to_host(ctx)
    var scale_d = _t(saved.head_scale.copy(), [S, dim], ctx)
    var shift_d = _t(saved.head_shift.copy(), [S, dim], ctx)
    from serenitymojo.ops.tensor_algebra import add_scalar as _add_scalar
    var sc1 = _add_scalar(scale_d, Float32(1.0), ctx)
    var ln_x_t = _t(ln_x_img.copy(), [S, dim], ctx)
    var modulated = mul(ln_x_t, sc1, ctx)
    modulated = add(modulated, shift_d, ctx)
    var lbh = linear_backward(
        _t_like(d_out, [S, out_ch], base.hh_w[], ctx),
        _cast_like(modulated, base.hh_w[], ctx), base.hh_w[],
        S, dim, out_ch, ctx,
    )
    var d_modulated = lbh.d_x.to_host(ctx)
    var mbh = modulate_backward(
        _t(d_modulated, [S, dim], ctx), _t(ln_x_img^, [S, dim], ctx),
        _t(saved.head_scale.copy(), [S, dim], ctx), ctx,
    )
    var d_ln_img = mbh.d_x.to_host(ctx)
    var lnbh = layer_norm_backward(
        _t(d_ln_img, [S, dim], ctx), _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), eps, ctx,
    )
    var d_x_img = lnbh.d_x.to_host(ctx)

    var dora_grads = wan22_direct_dora_zero_grads(dora)
    var nonfinite = 0

    var bi = num_blocks - 1
    while bi >= 0:
        var block_idx = bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)

        var mv = saved.block_modvecs[bi].copy()
        var w = _wan22_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)
        var direct_w = _wan22_direct_projection_weights_from_block(handle.block, handle.prefix, ctx)
        var direct = _wan_block_direct_dora_for(dora, bi)

        var bg = wan22_block_direct_lycoris_backward[H, Dh, S, TXT](
            d_x_img.copy(), mv, w, direct_w, direct, saved.block_saved[bi],
            cos_t, sin_t, dim, ffn, eps, ctx,
        )

        d_x_img = bg.d_x.copy()
        var bbase = _block_base(bi)
        _scatter_dora_direct_grad(dora_grads, bbase + W_SA_Q, bg.sa_q)
        _scatter_dora_direct_grad(dora_grads, bbase + W_SA_K, bg.sa_k)
        _scatter_dora_direct_grad(dora_grads, bbase + W_SA_V, bg.sa_v)
        _scatter_dora_direct_grad(dora_grads, bbase + W_SA_O, bg.sa_o)
        _scatter_dora_direct_grad(dora_grads, bbase + W_CA_Q, bg.ca_q)
        _scatter_dora_direct_grad(dora_grads, bbase + W_CA_K, bg.ca_k)
        _scatter_dora_direct_grad(dora_grads, bbase + W_CA_V, bg.ca_v)
        _scatter_dora_direct_grad(dora_grads, bbase + W_CA_O, bg.ca_o)

        nonfinite += _nonfinite_dora_projection(bg.sa_q)
        nonfinite += _nonfinite_dora_projection(bg.sa_k)
        nonfinite += _nonfinite_dora_projection(bg.sa_v)
        nonfinite += _nonfinite_dora_projection(bg.sa_o)
        nonfinite += _nonfinite_dora_projection(bg.ca_q)
        nonfinite += _nonfinite_dora_projection(bg.ca_k)
        nonfinite += _nonfinite_dora_projection(bg.ca_v)
        nonfinite += _nonfinite_dora_projection(bg.ca_o)

        loader.mark_active_block_done(ctx)
        bi -= 1

    var lbi = linear_backward(
        _t_like(d_x_img, [S, dim], base.pe_w[], ctx),
        _t_like(img_tokens, [S, in_ch], base.pe_w[], ctx), base.pe_w[],
        S, in_ch, dim, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var d_txt_tokens = _zeros_f32(TXT * text_dim)

    return Wan22DirectDoRAGradSet(dora_grads^, d_img_tokens^, d_txt_tokens^, nonfinite)


def wan22_stack_direct_oft_backward_offload[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: Wan22StackBase,
    mut loader: TurboPlannedLoader, oft: FlatDirectOFTSet,
    cos: List[Float32], sin: List[Float32],
    saved: Wan22StackForward,
    dim: Int, ffn: Int, in_ch: Int, text_dim: Int, out_ch: Int,
    freq_dim: Int, eps: Float32, ctx: DeviceContext,
) raises -> Wan22DirectOFTGradSet:
    var num_blocks = len(oft.ad) // WAN_SLOTS

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.norm_backward import layer_norm_backward
    from serenitymojo.ops.elementwise_backward import modulate_backward

    var ones = List[Float32]()
    for _ in range(dim):
        ones.append(Float32(1.0))
    var zeros = List[Float32]()
    for _ in range(dim):
        zeros.append(Float32(0.0))
    var ln_x_img = layer_norm(
        _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), _t(zeros^, [dim], ctx), eps, ctx,
    ).to_host(ctx)
    var scale_d = _t(saved.head_scale.copy(), [S, dim], ctx)
    var shift_d = _t(saved.head_shift.copy(), [S, dim], ctx)
    from serenitymojo.ops.tensor_algebra import add_scalar as _add_scalar
    var sc1 = _add_scalar(scale_d, Float32(1.0), ctx)
    var ln_x_t = _t(ln_x_img.copy(), [S, dim], ctx)
    var modulated = mul(ln_x_t, sc1, ctx)
    modulated = add(modulated, shift_d, ctx)
    var lbh = linear_backward(
        _t_like(d_out, [S, out_ch], base.hh_w[], ctx),
        _cast_like(modulated, base.hh_w[], ctx), base.hh_w[],
        S, dim, out_ch, ctx,
    )
    var d_modulated = lbh.d_x.to_host(ctx)
    var mbh = modulate_backward(
        _t(d_modulated, [S, dim], ctx), _t(ln_x_img^, [S, dim], ctx),
        _t(saved.head_scale.copy(), [S, dim], ctx), ctx,
    )
    var d_ln_img = mbh.d_x.to_host(ctx)
    var lnbh = layer_norm_backward(
        _t(d_ln_img, [S, dim], ctx), _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), eps, ctx,
    )
    var d_x_img = lnbh.d_x.to_host(ctx)

    var oft_grads = wan22_direct_oft_zero_grads(oft)
    var nonfinite = 0

    var bi = num_blocks - 1
    while bi >= 0:
        var block_idx = bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)

        var mv = saved.block_modvecs[bi].copy()
        var w = _wan22_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)
        var direct_w = _wan22_direct_projection_weights_from_block(handle.block, handle.prefix, ctx)
        var direct = _wan_block_direct_oft_for(oft, bi)

        var bg = wan22_block_direct_lycoris_backward[H, Dh, S, TXT](
            d_x_img.copy(), mv, w, direct_w, direct, saved.block_saved[bi],
            cos_t, sin_t, dim, ffn, eps, ctx,
        )

        d_x_img = bg.d_x.copy()
        var bbase = _block_base(bi)
        _scatter_oft_direct_grad(oft_grads, bbase + W_SA_Q, bg.sa_q)
        _scatter_oft_direct_grad(oft_grads, bbase + W_SA_K, bg.sa_k)
        _scatter_oft_direct_grad(oft_grads, bbase + W_SA_V, bg.sa_v)
        _scatter_oft_direct_grad(oft_grads, bbase + W_SA_O, bg.sa_o)
        _scatter_oft_direct_grad(oft_grads, bbase + W_CA_Q, bg.ca_q)
        _scatter_oft_direct_grad(oft_grads, bbase + W_CA_K, bg.ca_k)
        _scatter_oft_direct_grad(oft_grads, bbase + W_CA_V, bg.ca_v)
        _scatter_oft_direct_grad(oft_grads, bbase + W_CA_O, bg.ca_o)

        nonfinite += _nonfinite_oft_projection(bg.sa_q)
        nonfinite += _nonfinite_oft_projection(bg.sa_k)
        nonfinite += _nonfinite_oft_projection(bg.sa_v)
        nonfinite += _nonfinite_oft_projection(bg.sa_o)
        nonfinite += _nonfinite_oft_projection(bg.ca_q)
        nonfinite += _nonfinite_oft_projection(bg.ca_k)
        nonfinite += _nonfinite_oft_projection(bg.ca_v)
        nonfinite += _nonfinite_oft_projection(bg.ca_o)

        loader.mark_active_block_done(ctx)
        bi -= 1

    var lbi = linear_backward(
        _t_like(d_x_img, [S, dim], base.pe_w[], ctx),
        _t_like(img_tokens, [S, in_ch], base.pe_w[], ctx), base.pe_w[],
        S, in_ch, dim, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var d_txt_tokens = _zeros_f32(TXT * text_dim)

    return Wan22DirectOFTGradSet(oft_grads^, d_img_tokens^, d_txt_tokens^, nonfinite)


# ── AdamW step on all adapters ────────────────────────────────────────────────
def wan22_lora_adamw_step(
    mut lora: Wan22LoraSet, grads: Wan22LoraGradSet, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps_opt: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var n = wan22_total_adapters(lora)
    for i in range(n):
        if len(grads.d_a[i]) == 0:
            continue
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(lora.ad[i], lg, t, lr, ctx, beta1, beta2, eps_opt, weight_decay)


# ── PEFT-keyed save ───────────────────────────────────────────────────────────
# Key format: "blocks.{i}.self_attn.q.lora_A.weight" etc.
def _wan22_lora_prefixes(num_blocks: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_blocks):
        var b = String("blocks.") + String(bi) + "."
        out.append(b + String("self_attn.q"))
        out.append(b + String("self_attn.k"))
        out.append(b + String("self_attn.v"))
        out.append(b + String("self_attn.o"))
        out.append(b + String("cross_attn.q"))
        out.append(b + String("cross_attn.k"))
        out.append(b + String("cross_attn.v"))
        out.append(b + String("cross_attn.o"))
    return out^


def save_wan22_lora(lora: Wan22LoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var prefixes = _wan22_lora_prefixes(lora.num_blocks)
    var named = List[NamedLora]()
    var n = wan22_total_adapters(lora)
    for i in range(n):
        named.append(NamedLora(prefixes[i], lora.ad[i].copy()))
    return save_lora_peft(named, path, ctx)
