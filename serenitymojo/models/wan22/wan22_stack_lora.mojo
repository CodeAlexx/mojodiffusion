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
#   - LoRA targets: 10 per block — self_attn.{q,k,v,o} + cross_attn.{q,k,v,o}
#     (in=out=dim each) + ffn.0 (dim->ffn) + ffn.2 (ffn->dim), matching Musubi's
#     wrapped-linear set. 40 blocks -> 400 adapters.
#   - CHECKPOINT KEYS: blocks.{i}.self_attn.q.weight etc. (NOT fused).
#
# CARRIER DESIGN: Wan22LoraSet has a flat List[LoraAdapter], 10 slots per block.
# Slot order per block: sa_q, sa_k, sa_v, sa_o, ca_q, ca_k, ca_v, ca_o, ffn0, ffn2
# (the ffn slots are non-square: ffn0 A=[r,dim]/B=[ffn,r]; ffn2 A=[r,ffn]/B=[dim,r]).
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
from std.time import perf_counter_ns
from std.ffi import external_call
from std.memory import alloc
from std.builtin.type_aliases import MutExternalOrigin
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt, sin as _fsin, cos as _fcos

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
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
    wan22_block_lora_backward_devnative, WanBlockLoraDeviceGrads,
    wan22_block_direct_lycoris_forward, wan22_block_direct_lycoris_backward,
    WanI2VBlockWeights, WanI2VBlockLora, WanI2VBlockLoraGrads, WanI2VSaved,
    wan22_i2v_block_lora_forward, wan22_i2v_block_lora_backward,
)
from serenitymojo.autograd_v2.wan22_block_graph import (
    wan22_block_lora_graph_backward, WanBlockGraphGrads,
)
from serenitymojo.models.klein.lora_block import KleinLoraGrads
from serenitymojo.training.dora_adapter import DoRAGrads
from serenitymojo.training.oft_serenity_trainer import OFTOTGrads
from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_adamw_plain_fused import (
    fused_lora_adamw_plain_step,
    LoraAdamWPlainDeviceState, lora_adamw_plain_device_state_init,
    fused_lora_adamw_plain_step_resident,
    fused_lora_adamw_plain_step_resident_device_grads,
)
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_peft, save_lora_train_state, load_lora_train_state,
    load_lora_for_resume,
)
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

# Attention-only slots per block (8 projections). This constant governs the
# DIRECT DoRA/OFT paths (FlatDirectDoRASet/FlatDirectOFTSet), which wrap ONLY the
# 8 attention linears — do NOT widen it, or their `len(ad)//WAN_SLOTS` block
# counts + _block_base offsets break.
comptime WAN_SLOTS = 8
# LoRA carrier slots per block (10 linears = 8 attention + ffn.0 + ffn.2). The
# plain-LoRA Wan22LoraSet uses THIS count. Deterministic order: attn first, then
# ffn.0, ffn.2.
comptime WAN_LORA_SLOTS = 10
# Slot indices within a block's 8-slot window.
comptime W_SA_Q = 0
comptime W_SA_K = 1
comptime W_SA_V = 2
comptime W_SA_O = 3
comptime W_CA_Q = 4
comptime W_CA_K = 5
comptime W_CA_V = 6
comptime W_CA_O = 7
# The 2 extra LoRA-only slots (attn+ffn = 10-slot window).
comptime W_FFN0 = 8   # ffn.0: in=dim, out=ffn
comptime W_FFN2 = 9   # ffn.2: in=ffn, out=dim


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


# ── LoRA carrier (flat, 10 slots/block) ───────────────────────────────────────
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
    num_blocks: Int, dim: Int, ffn: Int, rank: Int, alpha: Float32,
) -> Wan22LoraSet:
    """Build a full Wan22LoraSet (A=randn, B=0 -> identity at init).
    10 adapters per block, deterministic order: sa_{q,k,v,o} + ca_{q,k,v,o}
    (all in=out=dim), then ffn.0 (in=dim,out=ffn) + ffn.2 (in=ffn,out=dim).
    """
    var ad = List[LoraAdapter]()
    var seed = UInt64(9001)
    for _ in range(num_blocks):
        # 8 square attention adapters
        for _ in range(8):
            ad.append(_make_adapter(rank, alpha, dim, dim, seed))
            seed += 1
        # ffn.0: dim -> ffn ; ffn.2: ffn -> dim (non-square)
        ad.append(_make_adapter(rank, alpha, dim, ffn, seed))
        seed += 1
        ad.append(_make_adapter(rank, alpha, ffn, dim, seed))
        seed += 1
    return Wan22LoraSet(ad^, num_blocks, rank)


def wan22_total_adapters(lora: Wan22LoraSet) -> Int:
    return lora.num_blocks * WAN_LORA_SLOTS


def _block_base(bi: Int) -> Int:
    # DIRECT DoRA/OFT (8 attention slots/block).
    return bi * WAN_SLOTS


def _lora_block_base(bi: Int) -> Int:
    # Plain-LoRA carrier (10 slots/block).
    return bi * WAN_LORA_SLOTS


# Build the WanBlockLora struct for block bi from the flat carrier (10 slots).
def _wan_block_lora_for(lora: Wan22LoraSet, bi: Int) -> WanBlockLora:
    var base = _lora_block_base(bi)
    return WanBlockLora(
        Optional[LoraAdapter](lora.ad[base + W_SA_Q].copy()),
        Optional[LoraAdapter](lora.ad[base + W_SA_K].copy()),
        Optional[LoraAdapter](lora.ad[base + W_SA_V].copy()),
        Optional[LoraAdapter](lora.ad[base + W_SA_O].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_Q].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_K].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_V].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_O].copy()),
        Optional[LoraAdapter](lora.ad[base + W_FFN0].copy()),
        Optional[LoraAdapter](lora.ad[base + W_FFN2].copy()),
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
    var d_a: List[List[Float32]]   # [num_blocks*10][rank*in]  (ffn slots non-square)
    var d_b: List[List[Float32]]   # [num_blocks*10][out*rank]
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
    # per-block F32 INPUT tape (block-0 input = embedded img seq; block i = block
    # i-1's F32 output). Populated by the plain-LoRA forward for the WAN_DEVNATIVE
    # RECOMPUTE backward; empty on the direct-DoRA/OFT forwards (they don't use it).
    var block_inputs: List[List[Float32]]

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
        var block_inputs: List[List[Float32]] = List[List[Float32]](),
    ):
        self.out = out^
        self.block_saved = block_saved^
        self.block_modvecs = block_modvecs^
        self.x_img = x_img^
        self.context_emb = context_emb^
        self.e_head_f32 = e_head_f32^
        self.head_shift = head_shift^
        self.head_scale = head_scale^
        self.block_inputs = block_inputs^


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
comptime _WanEnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _wan_phase_dbg() -> Bool:
    """WAN22_PHASE_TIMERS=1 — per-block host/GPU split inside the stack loops.
    Same external_call getenv the trainer uses (train_wan22_real.mojo:248); the
    std.os getenv overload does not lower in this build."""
    var name = String("WAN22_PHASE_TIMERS")
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _WanEnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _WanEnvPtr](cname)
    buf.free()
    return Int(ret) != 0


def _block_modvecs_dev(
    e0_t: Tensor, bmod_t: Tensor, S: Int, dim: Int, ctx: DeviceContext,
) raises -> WanModVecs:
    """DEVICE sibling of `_block_modvecs` (below) — same result, same return type,
    drop-in at any call site.

    `_block_modvecs` is a pure-CPU nested loop (S*dim iterations × 6 `append`s)
    computing what is just a broadcast add, `e[s,j,d] = e0[s,j,d] + bmod[j,d]`.
    MEASURED at the real dims (S=256, dim=5120,
    models/wan22/parity/wan22_host_cost_bench.mojo): **64.66 ms per call × 40
    blocks = 2.586 s/step**, i.e. ~32% of the (post-LoRA-fix) 8.4 s step — the
    largest single remaining host cost, and a direct violation of the
    host-is-a-parking-lot-not-a-CPU rule.

    Here the same arithmetic is one broadcast `add` on device ([1,6,dim] over
    [S,6,dim]) plus 6 slices. BIT-EQUAL by construction: both sides do the SAME
    F32 elementwise add in the SAME order (IEEE add is deterministic elementwise);
    only WHERE it runs differs. Gated in
    `models/wan22/parity/wan22_modvecs_dev_parity.mojo`.

    Still returns host `List[Float32]`s because every consumer signature takes
    `WanModVecs` — the to_host of 6 × [S,dim] costs ~0.59 ms each (same bench),
    so ~3.5 ms/block vs the 64.66 ms loop. Making WanModVecs hold device tensors
    (deleting the readback AND the consumers' re-uploads) is the follow-up."""
    var e_all = add(e0_t, bmod_t, ctx)            # [S,6,dim] F32, broadcast over S
    var out = List[List[Float32]]()
    for j in range(6):
        var sl = slice(e_all, 1, j, 1, ctx)       # [S,1,dim]
        out.append(reshape(sl, [S, dim], ctx).to_host(ctx))
    # oracle field order: shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn
    return WanModVecs(
        out[0].copy(), out[1].copy(), out[2].copy(),
        out[3].copy(), out[4].copy(), out[5].copy(),
    )


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
    # Bernini-R fp8 per-block cache: the 10 2-D matrices stream as E4M3 bytes
    # alongside a sibling per-output-row F32 scale "<key>.__fp8_scale" [out].
    # Reconstruct the weight per-row (out = e4m3_decode * scale[out]) to BF16 —
    # the EXACT dequant the certified Bernini inference stream uses
    # (wan22_fp8_stream.load_block_bf16 / bernini_r_block_parity, E4M3>=.99) —
    # then cast to the F32 host list the block fwd/bwd already consumes. BF16/F16
    # checkpoints (no fp8) fall through to the plain cast unchanged (back-compat).
    if block[key][].dtype() == STDtype.F8_E4M3:
        var sname = key + String(".__fp8_scale")
        if not (sname in block):
            raise Error(String("wan22 fp8 block missing per-row scale: ") + sname)
        var bf16 = fp8_e4m3_dequant_perrow_to_bf16(block[key][], block[sname][], ctx)
        return cast_tensor(bf16, STDtype.F32, ctx).to_host(ctx)
    return cast_tensor(block[key][], STDtype.F32, ctx).to_host(ctx)


# Device-native per-block loader (LEVER 1). Returns the frozen base weight as a
# bf16 DEVICE TArc WITHOUT the device->F32->host round-trip _block_f32 does (which
# then re-`from_host`s to device bf16 in the WanBlockWeights ctor — a full per-step
# host round-trip per tensor). fp8 path: the EXACT per-row dequant fp8->bf16 the
# certified Bernini stream uses (wan22_fp8_stream / bernini_r_block_parity), kept
# on device. non-fp8 path: cast to bf16 on device (cast_tensor always materializes
# a fresh device buffer — no aliasing with the streamed block). The result is
# BIT-IDENTICAL to TArc(from_host(_block_f32(...), BF16)) because bf16->F32->bf16
# is lossless. Feeds WanBlockWeights.from_device / WanI2VBlockWeights.from_device.
def _block_bf16_dev(block: Block, key: String, ctx: DeviceContext) raises -> TArc:
    if not (key in block):
        raise Error(String("wan22 offload block missing tensor: ") + key)
    if block[key][].dtype() == STDtype.F8_E4M3:
        var sname = key + String(".__fp8_scale")
        if not (sname in block):
            raise Error(String("wan22 fp8 block missing per-row scale: ") + sname)
        return TArc(fp8_e4m3_dequant_perrow_to_bf16(block[key][], block[sname][], ctx))
    return TArc(cast_tensor(block[key][], STDtype.BF16, ctx))


def _wan22_block_weights_from_block(
    block: Block, prefix: String, dim: Int, ffn: Int, hd: Int, ctx: DeviceContext,
) raises -> WanBlockWeights:
    var bp = prefix + "."
    # LEVER 1: device-native base load (no F32 host round-trip). from_device stores
    # the already-bf16-device tensors directly; bit-identical to the from_host path.
    # dim/ffn/hd unused here (tensors carry their own device shapes) — kept for the
    # caller signature / back-compat.
    _ = dim
    _ = ffn
    _ = hd
    return WanBlockWeights.from_device(
        # self-attn weights
        _block_bf16_dev(block, bp + String("self_attn.q.weight"), ctx),
        _block_bf16_dev(block, bp + String("self_attn.k.weight"), ctx),
        _block_bf16_dev(block, bp + String("self_attn.v.weight"), ctx),
        _block_bf16_dev(block, bp + String("self_attn.o.weight"), ctx),
        _block_bf16_dev(block, bp + String("self_attn.q.bias"), ctx),
        _block_bf16_dev(block, bp + String("self_attn.k.bias"), ctx),
        _block_bf16_dev(block, bp + String("self_attn.v.bias"), ctx),
        _block_bf16_dev(block, bp + String("self_attn.o.bias"), ctx),
        _block_bf16_dev(block, bp + String("self_attn.norm_q.weight"), ctx),
        _block_bf16_dev(block, bp + String("self_attn.norm_k.weight"), ctx),
        # cross-attn weights
        _block_bf16_dev(block, bp + String("cross_attn.q.weight"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.k.weight"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.v.weight"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.o.weight"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.q.bias"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.k.bias"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.v.bias"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.o.bias"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.norm_q.weight"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.norm_k.weight"), ctx),
        # norm3 (affine LN before cross-attn)
        _block_bf16_dev(block, bp + String("norm3.weight"), ctx),
        _block_bf16_dev(block, bp + String("norm3.bias"), ctx),
        # ffn
        _block_bf16_dev(block, bp + String("ffn.0.weight"), ctx),
        _block_bf16_dev(block, bp + String("ffn.0.bias"), ctx),
        _block_bf16_dev(block, bp + String("ffn.2.weight"), ctx),
        _block_bf16_dev(block, bp + String("ffn.2.bias"), ctx),
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
    save_acts: Bool = True,
    device_lora: Bool = False,
    batched_cross: Bool = False,
    device_modvecs: Bool = False,
) raises -> Wan22StackForward:
    """`save_acts=False` omits the per-block save-all HOST activation tape
    (`block_saved`). ONLY the host offload backward (this file, `…_backward_offload`)
    reads it; the devnative and autograd_v2 graph drivers RECOMPUTE each block from
    `block_inputs`, so for them the tape is ~5.5 GB/step of host list traffic
    (23 lists/block built, then `.copy()`d) for data nobody reads. Default True =
    byte-identical to before (C13)."""
    var num_blocks = lora.num_blocks

    loader.prefetch_with_ctx(0, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S, Dh // 2], STDtype.F32, ctx)

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
    # per-block F32 INPUT tape for the WAN_DEVNATIVE recompute backward.
    var block_inputs = List[List[Float32]]()

    # e0 uploaded ONCE per step for the device modvec builder ([S,6,dim] F32,
    # ~31 MB, one HtoD) — it originates on device inside _time_features anyway.
    var e0_t = Tensor.from_host(e0_flat.copy(), [S, 6, dim], STDtype.F32, ctx)

    for bi in range(num_blocks):
        var _t_pre = perf_counter_ns()
        var handle = loader.await_block(bi, ctx)
        var _t_aw = perf_counter_ns()
        loader.prefetch_next_with_ctx(bi, ctx)
        var _t_blk0 = perf_counter_ns()

        # Load block modulation [1, 6, dim] BF16 -> F32 host.
        var bp = handle.prefix + "."
        var mod_key = bp + String("modulation")
        if not (mod_key in handle.block):
            raise Error(String("wan22 block missing modulation: ") + mod_key)
        var _t_await = perf_counter_ns()
        var block_mod_t = cast_tensor(handle.block[mod_key][], STDtype.F32, ctx)

        # Build per-token modulation vectors. The HOST builder is a pure-CPU
        # broadcast-add loop measured at 64.66 ms/block = 2.586 s/step (32% of the
        # step); the device builder is the same arithmetic on GPU, bit-equal.
        var mv: WanModVecs
        if device_modvecs:
            mv = _block_modvecs_dev(
                e0_t, reshape(block_mod_t, [1, 6, dim], ctx), S, dim, ctx,
            )
        else:
            mv = _block_modvecs(e0_flat, block_mod_t.to_host(ctx), bi, S, dim)

        var _t_mv = perf_counter_ns()
        # Load block weights.
        var w = _wan22_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)
        var _t_w = perf_counter_ns()

        # LoRA-augmented forward. `x_h`/`context_h` are READ (borrowed) args — the
        # old `.copy()`s were pure waste: [S,dim]=1.31M + [TXT,dim]=2.6M host floats
        # per block, ~157M element copies/step at 40 blocks.
        var bl = _wan_block_lora_for(lora, bi)
        block_inputs.append(img.copy())
        var fwd = wan22_block_lora_forward[H, Dh, S, TXT](
            img, context_emb, mv, w, bl, cos_t, sin_t, dim, ffn, eps, ctx,
            save_acts, device_lora, batched_cross,
        )

        # save_acts=False (recompute backward drivers): the 23-list per-block host
        # tape is never read, so it is neither BUILT (above) nor COPIED here.
        var _t_fwd = perf_counter_ns()
        if _wan_phase_dbg() and bi < 3:
            print("    [blk", bi, "] AWAIT=", Float64(_t_aw - _t_pre) / 1.0e6,
                  " prefetch=", Float64(_t_blk0 - _t_aw) / 1.0e6,
                  " mod=", Float64(_t_await - _t_blk0) / 1.0e6,
                  " modvecs=", Float64(_t_mv - _t_await) / 1.0e6,
                  " weights=", Float64(_t_w - _t_mv) / 1.0e6,
                  " blockfwd=", Float64(_t_fwd - _t_w) / 1.0e6, "ms")
        if save_acts:
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
        block_inputs^,
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

    var cos_t = Tensor.from_host(cos.copy(), [S, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S, Dh // 2], STDtype.F32, ctx)

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

    var cos_t = Tensor.from_host(cos.copy(), [S, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S, Dh // 2], STDtype.F32, ctx)

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

    var cos_t = Tensor.from_host(cos.copy(), [S, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S, Dh // 2], STDtype.F32, ctx)

    # ── head backward (proj_out -> modulate -> LN_no_affine) ──
    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.norm_backward import layer_norm_backward_dx
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
        compute_param_grads=False,  # head modulation frozen (LoRA): only d_x needed; scale is per-token [S,dim]
    )
    var d_ln_img = mbh.d_x.to_host(ctx)
    # LN_no_affine backward
    var lnbh = layer_norm_backward_dx(
        _t(d_ln_img, [S, dim], ctx), _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), eps, ctx,
    )
    var d_x_img = lnbh.to_host(ctx)   # [S, dim] — enters last block output

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

        # Scatter LoRA grads into flat arrays (10 slots/block).
        var bbase = _lora_block_base(bi)
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
        d_a_flat[bbase + W_FFN0] = bg.ffn0_da.copy()
        d_b_flat[bbase + W_FFN0] = bg.ffn0_db.copy()
        d_a_flat[bbase + W_FFN2] = bg.ffn2_da.copy()
        d_b_flat[bbase + W_FFN2] = bg.ffn2_db.copy()

        nonfinite += _nonfinite(bg.sa_q_da) + _nonfinite(bg.sa_q_db)
        nonfinite += _nonfinite(bg.sa_k_da) + _nonfinite(bg.sa_k_db)
        nonfinite += _nonfinite(bg.sa_v_da) + _nonfinite(bg.sa_v_db)
        nonfinite += _nonfinite(bg.sa_o_da) + _nonfinite(bg.sa_o_db)
        nonfinite += _nonfinite(bg.ca_q_da) + _nonfinite(bg.ca_q_db)
        nonfinite += _nonfinite(bg.ca_k_da) + _nonfinite(bg.ca_k_db)
        nonfinite += _nonfinite(bg.ca_v_da) + _nonfinite(bg.ca_v_db)
        nonfinite += _nonfinite(bg.ca_o_da) + _nonfinite(bg.ca_o_db)
        nonfinite += _nonfinite(bg.ffn0_da) + _nonfinite(bg.ffn0_db)
        nonfinite += _nonfinite(bg.ffn2_da) + _nonfinite(bg.ffn2_db)

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


# ══════════════════════════════════════════════════════════════════════════════
# WAN_DEVNATIVE loader-based backward: same head-backward + streamed conductor
# seam as wan22_stack_lora_backward_offload, but the reverse BLOCK loop RECOMPUTES
# each block from its saved F32 input (saved.block_inputs) and runs the DEVICE-
# NATIVE block backward (wan22_block_lora_backward_devnative). The block d_x stays
# a device BF16 Tensor threaded block->block (no to_host in the loop), and each
# block's 10 device LoRA grads (F32 TArc) accumulate into a device grad set that
# feeds the resident device-grad AdamW directly (no host grad list). The input-
# projection backward is skipped (frozen, discarded in LoRA training).
# ══════════════════════════════════════════════════════════════════════════════
def wan22_stack_lora_backward_offload_devnative[
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
) raises -> Wan22LoraDeviceGradSet:
    var num_blocks = lora.num_blocks
    var n_adapters = wan22_total_adapters(lora)

    if len(saved.block_inputs) != num_blocks:
        raise Error(
            "wan22_stack_lora_backward_offload_devnative: forward tape missing block_inputs"
            " (use wan22_stack_lora_forward_offload)"
        )

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S, Dh // 2], STDtype.F32, ctx)

    # ── head backward (proj_out -> modulate -> LN_no_affine) — verbatim clone of
    #    the offload backward's head chain, producing d_x_img [S,dim] host F32. ──
    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.norm_backward import layer_norm_backward_dx
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
        compute_param_grads=False,
    )
    var d_ln_img = mbh.d_x.to_host(ctx)
    var lnbh = layer_norm_backward_dx(
        _t(d_ln_img, [S, dim], ctx), _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), eps, ctx,
    )
    var d_x_img = lnbh.to_host(ctx)   # [S, dim] — enters last block output

    # ── dense per-adapter device grad slots (all 10/block filled) ──
    var da_opt = List[Optional[TArc]]()
    var db_opt = List[Optional[TArc]]()
    for _ in range(n_adapters):
        da_opt.append(Optional[TArc]())
        db_opt.append(Optional[TArc]())

    # seed the reverse chain with the head-grad as a device BF16 tensor.
    var d_out_dev = Tensor.from_host(d_x_img.copy(), [S, dim], STDtype.BF16, ctx)

    # ── stream blocks in REVERSE (recompute + device-native) ──
    var bi = num_blocks - 1
    while bi >= 0:
        var handle = loader.await_block(bi, ctx)
        if bi > 0:
            loader.prefetch_with_ctx(bi - 1, ctx)

        var mv = saved.block_modvecs[bi].copy()
        var w = _wan22_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)
        var bl = _wan_block_lora_for(lora, bi)

        # RECOMPUTE this block's forward from its F32 input (one device tape live).
        var rc = wan22_block_lora_forward[H, Dh, S, TXT](
            saved.block_inputs[bi].copy(), saved.context_emb.copy(),
            mv, w, bl, cos_t, sin_t, dim, ffn, eps, ctx,
        )
        var gd = wan22_block_lora_backward_devnative[H, Dh, S, TXT](
            d_out_dev, mv, w, bl, rc.saved, cos_t, sin_t, dim, ffn, eps, ctx,
        )
        d_out_dev = gd.d_x.clone(ctx)
        var bbase = _lora_block_base(bi)
        for slot in range(WAN_LORA_SLOTS):
            if gd.d_a[slot]:
                da_opt[bbase + slot] = Optional[TArc](gd.d_a[slot].value().copy())
                db_opt[bbase + slot] = Optional[TArc](gd.d_b[slot].value().copy())

        loader.mark_active_block_done(ctx)
        bi -= 1

    # flatten dense slots -> ascending device grad lists + dense grad indices.
    var d_a = List[TArc]()
    var d_b = List[TArc]()
    var grad_indices = List[Int]()
    for i in range(n_adapters):
        if not da_opt[i]:
            raise Error(
                "wan22_stack_lora_backward_offload_devnative: missing device grad slot "
                + String(i)
            )
        d_a.append(da_opt[i].value().copy())
        d_b.append(db_opt[i].value().copy())
        grad_indices.append(i)

    var d_x_tokens = d_out_dev.to_host(ctx)
    return Wan22LoraDeviceGradSet(
        d_a^, d_b^, grad_indices^, d_x_tokens^, n_adapters,
    )


def wan22_stack_lora_backward_offload_graph[
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
    batched_cross: Bool = False,
) raises -> Wan22LoraDeviceGradSet:
    """autograd_v2 GRAPH twin of `wan22_stack_lora_backward_offload_devnative`
    (this file :1113) — same signature, same conductor seam, same return type;
    the per-block call is the ONLY difference (Phase 2, contract C13: the
    devnative and host drivers stay compiled and reachable).

    Per block the devnative driver does TWO calls — `wan22_block_lora_forward`
    (a save-all host tape) then `wan22_block_lora_backward_devnative` (which
    re-uploads those saved acts). The graph driver replaces BOTH with one
    `wan22_block_lora_graph_backward`, which recomputes the block forward
    THROUGH the record_* wrappers and drives the dependency-counted engine
    backward — so the save-all host tape and its re-upload round trip are gone,
    and the step becomes StepSlab-allocatable / CUDA-graph capturable (the
    actual 15→~1.6 s/step lever; the ~52k host launches/step are what capture
    collapses).

    Gated BIT-EXACT per block against the devnative oracle on all 10 LoRA slots
    (autograd_v2/tests/wan_block_graph_parity.mojo; d_x/d_context are a 4dp
    value class by the C15 fan-in reassociation — see that gate's header).

    MEMORY (16 GB): exactly ONE device tape is live at a time — the graph is
    built, executed, and dropped inside each loop iteration, matching the
    devnative driver's recompute discipline. `d_context` is computed by the
    block graph but DISCARDED here, as the devnative driver does: this is
    LoRA-only training and the text-encoder path is frozen.
    """
    var num_blocks = lora.num_blocks
    var n_adapters = wan22_total_adapters(lora)

    if len(saved.block_inputs) != num_blocks:
        raise Error(
            "wan22_stack_lora_backward_offload_graph: forward tape missing block_inputs"
            " (use wan22_stack_lora_forward_offload)"
        )

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S, Dh // 2], STDtype.F32, ctx)

    # ── head backward (proj_out → modulate → LN_no_affine) — verbatim clone of
    #    the devnative driver's head chain (:1140-1179). ──
    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.norm_backward import layer_norm_backward_dx
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
        compute_param_grads=False,
    )
    var d_ln_img = mbh.d_x.to_host(ctx)
    var lnbh = layer_norm_backward_dx(
        _t(d_ln_img, [S, dim], ctx), _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), eps, ctx,
    )
    var d_x_img = lnbh.to_host(ctx)   # [S, dim] — enters last block output

    # ── dense per-adapter device grad slots (all 10/block filled) ──
    var da_opt = List[Optional[TArc]]()
    var db_opt = List[Optional[TArc]]()
    for _ in range(n_adapters):
        da_opt.append(Optional[TArc]())
        db_opt.append(Optional[TArc]())

    var d_out_dev = Tensor.from_host(d_x_img.copy(), [S, dim], STDtype.BF16, ctx)

    # ── stream blocks in REVERSE (graph recompute, one device tape live) ──
    var bi = num_blocks - 1
    while bi >= 0:
        var handle = loader.await_block(bi, ctx)
        if bi > 0:
            loader.prefetch_with_ctx(bi - 1, ctx)

        # mv is READ-ONLY here (handed straight to the graph fn) — no .copy():
        # 6 x [S,dim] host lists per block = ~314M element copies/step at 40 blocks.
        ref mv = saved.block_modvecs[bi]
        var w = _wan22_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)
        var bl = _wan_block_lora_for(lora, bi)

        # the block's saved INPUT is the graph's recompute source (bf16 upload,
        # identical rounding to wan22_block_lora_forward's own _ta16, :1458).
        var x_in = TArc(_t16(saved.block_inputs[bi].copy(), [S, dim], ctx))
        var context_in = TArc(_t16(saved.context_emb.copy(), [TXT, dim], ctx))
        var gg = wan22_block_lora_graph_backward[H, Dh, S, TXT](
            d_out_dev, x_in, context_in, mv, w, bl, cos_t, sin_t,
            dim, ffn, eps, ctx, batched_cross,
        )
        d_out_dev = gg.d_x[].clone(ctx)
        var bbase = _lora_block_base(bi)
        for slot in range(WAN_LORA_SLOTS):
            if gg.d_a[slot]:
                da_opt[bbase + slot] = Optional[TArc](gg.d_a[slot].value().copy())
                db_opt[bbase + slot] = Optional[TArc](gg.d_b[slot].value().copy())

        loader.mark_active_block_done(ctx)
        bi -= 1

    # flatten dense slots → ascending device grad lists + dense grad indices.
    var d_a = List[TArc]()
    var d_b = List[TArc]()
    var grad_indices = List[Int]()
    for i in range(n_adapters):
        if not da_opt[i]:
            raise Error(
                "wan22_stack_lora_backward_offload_graph: missing device grad slot "
                + String(i)
            )
        d_a.append(da_opt[i].value().copy())
        d_b.append(db_opt[i].value().copy())
        grad_indices.append(i)

    var d_x_tokens = d_out_dev.to_host(ctx)
    return Wan22LoraDeviceGradSet(
        d_a^, d_b^, grad_indices^, d_x_tokens^, n_adapters,
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

    var cos_t = Tensor.from_host(cos.copy(), [S, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S, Dh // 2], STDtype.F32, ctx)

    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.norm_backward import layer_norm_backward_dx
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
        compute_param_grads=False,  # head modulation frozen (LoRA): only d_x needed; scale is per-token [S,dim]
    )
    var d_ln_img = mbh.d_x.to_host(ctx)
    var lnbh = layer_norm_backward_dx(
        _t(d_ln_img, [S, dim], ctx), _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), eps, ctx,
    )
    var d_x_img = lnbh.to_host(ctx)

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

    var cos_t = Tensor.from_host(cos.copy(), [S, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S, Dh // 2], STDtype.F32, ctx)

    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.norm_backward import layer_norm_backward_dx
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
        compute_param_grads=False,  # head modulation frozen (LoRA): only d_x needed; scale is per-token [S,dim]
    )
    var d_ln_img = mbh.d_x.to_host(ctx)
    var lnbh = layer_norm_backward_dx(
        _t(d_ln_img, [S, dim], ctx), _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), eps, ctx,
    )
    var d_x_img = lnbh.to_host(ctx)

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
# WAN22_GPU_ADAMW (default True) routes the update through the fused GPU
# PLAIN-AdamW kernel (training/lora_adamw_plain_fused.mojo::
# fused_lora_adamw_plain_step — per-element math mirrored from the SAME
# _adamw_host_list the host loop runs), replacing 320 host scalar passes with
# one packed launch. False = the retained host scalar loop
# (wan22_lora_adamw_step_host), the parity oracle. GPU FMA contraction vs the
# host F32 chain differs at ulp level (klein fused-AdamW class, ledger
# MJ-1017) — gate with
# models/wan22/parity/wan22_gpu_adamw_update_parity_smoke.mojo.
comptime WAN22_GPU_ADAMW = True


def wan22_lora_adamw_step_host(
    mut lora: Wan22LoraSet, grads: Wan22LoraGradSet, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps_opt: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    """Retained host scalar AdamW loop (the pre-WAN22_GPU_ADAMW path; kept as
    the fallback + the parity oracle for the fused GPU kernel)."""
    var n = wan22_total_adapters(lora)
    for i in range(n):
        if len(grads.d_a[i]) == 0:
            continue
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(lora.ad[i], lg, t, lr, ctx, beta1, beta2, eps_opt, weight_decay)


def wan22_lora_adamw_step(
    mut lora: Wan22LoraSet, grads: Wan22LoraGradSet, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps_opt: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    comptime if WAN22_GPU_ADAMW:
        # One fused GPU launch per contiguous run of adapters WITH grads —
        # the host loop's skip-empty semantics kept (adapters whose grad
        # lists are empty stay untouched). The plain-LoRA backward fills all
        # WAN_SLOTS per block, so this is normally ONE call over [0, n).
        var n = wan22_total_adapters(lora)
        var i = 0
        while i < n:
            if len(grads.d_a[i]) == 0:
                i += 1
                continue
            var run_start = i
            while i < n and len(grads.d_a[i]) != 0:
                i += 1
            fused_lora_adamw_plain_step(
                lora.ad, grads.d_a, grads.d_b, run_start, i,
                t, lr, beta1, beta2, eps_opt, weight_decay, ctx,
            )
    else:
        wan22_lora_adamw_step_host(
            lora, grads, t, lr, ctx, beta1, beta2, eps_opt, weight_decay
        )


# ── MJ-1085 resident host-grad AdamW (persistent device P/M/V) ────────────────
# Sibling of wan22_lora_adamw_step (WAN22_GPU_ADAMW per-step fused arm): the
# per-step fused arm allocates fresh PINNED host staging every step and can hit
# the MJ-1070 unmapped-buffer segfault under pinned pressure. The resident arm
# uploads P/M/V once (wan22_lora_adamw_state_init) then per step only grads go up
# and params come back — same kernel, bit-identical values.
#
# The plain-LoRA backward fills ALL WAN_SLOTS per block (the per-step fused arm's
# "normally ONE call over [0, n)" case), so the resident step covers the whole
# adapter range. If a caller ever produces empty grad lists for some adapters,
# fused_lora_adamw_plain_step_resident fails loud on the length check rather than
# skipping (a fail-loud divergence from the fused arm's skip-empty runs).
def wan22_lora_adamw_state_init(
    lora: Wan22LoraSet, ctx: DeviceContext
) raises -> LoraAdamWPlainDeviceState:
    return lora_adamw_plain_device_state_init(
        lora.ad, 0, wan22_total_adapters(lora), ctx
    )


def wan22_lora_adamw_step_resident(
    mut state: LoraAdamWPlainDeviceState,
    mut lora: Wan22LoraSet, grads: Wan22LoraGradSet, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps_opt: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    fused_lora_adamw_plain_step_resident(
        state, lora.ad, grads.d_a, grads.d_b,
        t, lr, beta1, beta2, eps_opt, weight_decay, ctx,
    )


# ── PEFT-keyed save ───────────────────────────────────────────────────────────
# Key format: "diffusion_model.blocks.{i}.self_attn.q.lora_A.weight" etc.
# ai-toolkit / ComfyUI Wan LoRA convention: `diffusion_model.` prefix + lora_A/lora_B
# (ai-toolkit stores internally as `transformer.` and rewrites to `diffusion_model.`
# on save for ComfyUI — see ai-toolkit base_audio_model.py:73-84 / network_mixins.py).
# Save + resume both use this function, so keys stay internally consistent.
def _wan22_lora_prefixes(num_blocks: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_blocks):
        var b = String("diffusion_model.blocks.") + String(bi) + "."
        out.append(b + String("self_attn.q"))
        out.append(b + String("self_attn.k"))
        out.append(b + String("self_attn.v"))
        out.append(b + String("self_attn.o"))
        out.append(b + String("cross_attn.q"))
        out.append(b + String("cross_attn.k"))
        out.append(b + String("cross_attn.v"))
        out.append(b + String("cross_attn.o"))
        out.append(b + String("ffn.0"))
        out.append(b + String("ffn.2"))
    return out^


def save_wan22_lora(lora: Wan22LoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var prefixes = _wan22_lora_prefixes(lora.num_blocks)
    var named = List[NamedLora]()
    var n = wan22_total_adapters(lora)
    for i in range(n):
        named.append(NamedLora(prefixes[i], lora.ad[i].copy()))
    return save_lora_peft(named, path, ctx)


# ── FULL train-state save / resume (A/B + AdamW moments) — MJ-1088 ────────────
# The PEFT save above is inference-only. This `.state` variant additionally carries
# adam_m/adam_v via the shared training/lora_save.mojo plumbing so a resumed run does
# NOT zero AdamW momentum (the MJ-1077 / MJ-1088 warm-restart class). Same block
# prefixes and flat 10×num_blocks adapter order as save_wan22_lora.
def save_wan22_lora_state(
    lora: Wan22LoraSet, path: String, ctx: DeviceContext,
    var meta: List[Float32] = List[Float32](),
) raises -> Int:
    """Write the trainer-only RESUME state (A/B + AdamW moments). `meta` (optional)
    is the resume-scope guard vector (e.g. [step, seed])."""
    var prefixes = _wan22_lora_prefixes(lora.num_blocks)
    var named = List[NamedLora]()
    var n = wan22_total_adapters(lora)
    for i in range(n):
        named.append(NamedLora(prefixes[i], lora.ad[i].copy()))
    return save_lora_train_state(named, path, ctx, meta^)


def load_wan22_lora_state(
    num_blocks: Int, D: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> Wan22LoraSet:
    """Reload the LoRA set (A/B + AdamW moments) written by save_wan22_lora_state.
    The restored moments ride the LoraAdapter host lists into the resident device
    AdamW state (wan22_lora_adamw_state_init seeds dev_m/dev_v from them)."""
    var scale = alpha / Float32(rank)
    var prefixes = _wan22_lora_prefixes(num_blocks)
    var loaded = load_lora_train_state(prefixes, scale, path, ctx)
    var ad = List[LoraAdapter]()
    for ref nl in loaded:
        ad.append(nl.adapter.copy())
    return Wan22LoraSet(ad^, num_blocks, rank)


def load_wan22_lora_resume(
    num_blocks: Int, D: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> Wan22LoraSet:
    """WARM resume: reload A/B weights only (from a PEFT file with no AdamW moments).
    Moments are ZEROED — the caller must warn loudly. Same prefix order as
    save_wan22_lora / load_wan22_lora_state."""
    var scale = alpha / Float32(rank)
    var prefixes = _wan22_lora_prefixes(num_blocks)
    var loaded = load_lora_for_resume(prefixes, scale, path, ctx)
    var ad = List[LoraAdapter]()
    for ref nl in loaded:
        ad.append(nl.adapter.copy())
    return Wan22LoraSet(ad^, num_blocks, rank)


# ═══════════════════════════════════════════════════════════════════════════════
# WAN 2.1 I2V-14B — DUAL CROSS-ATTENTION offload stack (12-slot LoRA/block).
#
# Mirrors the base T2V offload fwd/bwd above, but per block calls the CERTIFIED
# wan22_i2v_block_lora_forward/backward (models/wan22/wan22_block.mojo, cos>=0.999
# vs Musubi). Deltas vs the T2V stack:
#   (a) TWO frozen contexts — computed ONCE before the block loop, threaded
#       (copied) into every block: context_txt = text_embedding(umt5) [TXT,dim]
#       (frozen, embedded here) and context_img [IMG,dim] (frozen CLIP already
#       projected by MLPProj in the trainer, passed in). NOT re-derived per block.
#   (b) Each block streams the 5 extra image-branch weights alongside the base 27:
#       cross_attn.{k_img,v_img}.{weight,bias} + cross_attn.norm_k_img.weight.
#   (c) 12 LoRA adapters/block (base 10 + cross_attn.k_img + cross_attn.v_img).
#   (d) d_context (umt5 text / CLIP image) is NOT propagated out — umt5, CLIP, and
#       MLPProj are all FROZEN; only the per-block LoRA A/B grads matter.
# The certified block math + the T2V stack above are UNCHANGED.
# ═══════════════════════════════════════════════════════════════════════════════

# LoRA carrier slots per I2V block (12 = base 10 + img_k + img_v). Slot order:
# sa_{q,k,v,o}, ca_{q,k,v,o}, ffn.0, ffn.2, cross_attn.k_img, cross_attn.v_img.
comptime WAN_I2V_LORA_SLOTS = 12
comptime W_IMG_K = 10   # cross_attn.k_img: in=dim, out=dim
comptime W_IMG_V = 11   # cross_attn.v_img: in=dim, out=dim


struct Wan22I2VLoraSet(Movable):
    var ad: List[LoraAdapter]  # flat: block bi -> ad[bi*12 + slot]
    var num_blocks: Int
    var rank: Int

    def __init__(out self, var ad: List[LoraAdapter], num_blocks: Int, rank: Int):
        self.ad = ad^
        self.num_blocks = num_blocks
        self.rank = rank


def build_wan22_i2v_lora_set(
    num_blocks: Int, dim: Int, ffn: Int, rank: Int, alpha: Float32,
) -> Wan22I2VLoraSet:
    """Build a full Wan22I2VLoraSet (A=randn, B=0 -> identity at init).
    12 adapters per block, deterministic order: sa_{q,k,v,o} + ca_{q,k,v,o}
    (all in=out=dim), ffn.0 (dim->ffn), ffn.2 (ffn->dim), then cross_attn.k_img
    + cross_attn.v_img (both in=out=dim)."""
    var ad = List[LoraAdapter]()
    var seed = UInt64(31337)
    for _ in range(num_blocks):
        for _ in range(8):
            ad.append(_make_adapter(rank, alpha, dim, dim, seed))
            seed += 1
        ad.append(_make_adapter(rank, alpha, dim, ffn, seed))
        seed += 1
        ad.append(_make_adapter(rank, alpha, ffn, dim, seed))
        seed += 1
        # image-branch cross-attn: k_img + v_img (both square dim->dim)
        ad.append(_make_adapter(rank, alpha, dim, dim, seed))
        seed += 1
        ad.append(_make_adapter(rank, alpha, dim, dim, seed))
        seed += 1
    return Wan22I2VLoraSet(ad^, num_blocks, rank)


def wan22_i2v_total_adapters(lora: Wan22I2VLoraSet) -> Int:
    return lora.num_blocks * WAN_I2V_LORA_SLOTS


def _i2v_lora_block_base(bi: Int) -> Int:
    return bi * WAN_I2V_LORA_SLOTS


def _wan_i2v_block_lora_for(lora: Wan22I2VLoraSet, bi: Int) -> WanI2VBlockLora:
    var base = _i2v_lora_block_base(bi)
    var base_lora = WanBlockLora(
        Optional[LoraAdapter](lora.ad[base + W_SA_Q].copy()),
        Optional[LoraAdapter](lora.ad[base + W_SA_K].copy()),
        Optional[LoraAdapter](lora.ad[base + W_SA_V].copy()),
        Optional[LoraAdapter](lora.ad[base + W_SA_O].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_Q].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_K].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_V].copy()),
        Optional[LoraAdapter](lora.ad[base + W_CA_O].copy()),
        Optional[LoraAdapter](lora.ad[base + W_FFN0].copy()),
        Optional[LoraAdapter](lora.ad[base + W_FFN2].copy()),
    )
    return WanI2VBlockLora(
        base_lora^,
        Optional[LoraAdapter](lora.ad[base + W_IMG_K].copy()),
        Optional[LoraAdapter](lora.ad[base + W_IMG_V].copy()),
    )


struct Wan22I2VLoraGradSet(Movable):
    var d_a: List[List[Float32]]   # [num_blocks*12][rank*in]
    var d_b: List[List[Float32]]   # [num_blocks*12][out*rank]
    var d_x_tokens: List[Float32]  # [S,in_ch] input grad (frozen patch-embed backward)
    var nonfinite_lora_grads: Int

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_x_tokens: List[Float32], nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x_tokens = d_x_tokens^
        self.nonfinite_lora_grads = nonfinite_lora_grads


# Load per-block I2V weights: the base 27 tensors PLUS the 5 image-branch weights
# (cross_attn.k_img/v_img weight+bias, cross_attn.norm_k_img.weight). Streamed
# under the SAME "blocks.{i}." prefix as the base tensors.
def _wan22_i2v_block_weights_from_block(
    block: Block, prefix: String, dim: Int, ffn: Int, hd: Int, ctx: DeviceContext,
) raises -> WanI2VBlockWeights:
    var base = _wan22_block_weights_from_block(block, prefix, dim, ffn, hd, ctx)
    var bp = prefix + "."
    # LEVER 1: device-native base load (no F32 host round-trip). dim unused here.
    _ = dim
    return WanI2VBlockWeights.from_device(
        base^,
        _block_bf16_dev(block, bp + String("cross_attn.k_img.weight"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.k_img.bias"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.v_img.weight"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.v_img.bias"), ctx),
        _block_bf16_dev(block, bp + String("cross_attn.norm_k_img.weight"), ctx),
    )


# ── I2V forward tape ──────────────────────────────────────────────────────────
struct Wan22I2VStackForward(Movable):
    var out: List[Float32]                   # [S, out_ch] velocity prediction
    var block_saved: List[WanI2VSaved]       # per-block activations (dual cross-attn)
    var block_modvecs: List[WanModVecs]      # per-block modulation vectors (F32)
    var x_img: List[Float32]                 # [S, dim] image tokens before head
    var context_txt: List[Float32]           # [TXT, dim] embedded text context (frozen)
    var context_img: List[Float32]           # [IMG, dim] MLPProj image context (frozen)
    var e_head_f32: List[Float32]            # [S, dim]
    var head_shift: List[Float32]            # [S, dim]
    var head_scale: List[Float32]            # [S, dim]

    def __init__(
        out self,
        var out: List[Float32],
        var block_saved: List[WanI2VSaved],
        var block_modvecs: List[WanModVecs],
        var x_img: List[Float32],
        var context_txt: List[Float32],
        var context_img: List[Float32],
        var e_head_f32: List[Float32],
        var head_shift: List[Float32],
        var head_scale: List[Float32],
    ):
        self.out = out^
        self.block_saved = block_saved^
        self.block_modvecs = block_modvecs^
        self.x_img = x_img^
        self.context_txt = context_txt^
        self.context_img = context_img^
        self.e_head_f32 = e_head_f32^
        self.head_shift = head_shift^
        self.head_scale = head_scale^


# ═══════════════════════════════════════════════════════════════════════════════
# I2V FULL FORWARD WITH LoRA, BLOCK-SWAP OFFLOAD.
#   img_tokens:     [S, in_ch]      (I2V packed patch: noisy16->64 ++ cond_y20->80 = 144)
#   txt_tokens:     [TXT, text_dim] (raw umt5 embeddings — embedded here, frozen)
#   context_img_h:  [IMG, dim]      (CLIP already projected by frozen MLPProj)
#   t_model:        Float32 scalar timestep
#   cos/sin:        RoPE tables [S, Dh/2]
# ═══════════════════════════════════════════════════════════════════════════════
def wan22_i2v_stack_lora_forward_offload[
    H: Int, Dh: Int, S: Int, TXT: Int, IMG: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    context_img_h: List[Float32],
    t_model: Float32,
    base: Wan22StackBase,
    mut loader: TurboPlannedLoader, lora: Wan22I2VLoraSet,
    cos: List[Float32], sin: List[Float32],
    dim: Int, ffn: Int, in_ch: Int, text_dim: Int, out_ch: Int,
    freq_dim: Int, eps: Float32, ctx: DeviceContext,
) raises -> Wan22I2VStackForward:
    var num_blocks = lora.num_blocks

    loader.prefetch_with_ctx(0, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S, Dh // 2], STDtype.F32, ctx)

    # ── frozen embeddings — context assembled ONCE, shared by every block ──
    var img = _embed_image(img_tokens, S, in_ch, dim, base, ctx)
    var context_txt = _embed_context(txt_tokens, TXT, text_dim, dim, base, ctx)
    var context_img = context_img_h.copy()   # already [IMG,dim] via MLPProj (frozen)

    # ── frozen time feature chain ──
    var tfeats = _time_features(t_model, S, dim, freq_dim, base, ctx)
    var e0_flat = tfeats[0].copy()
    var e_head = tfeats[1].copy()

    var block_saved = List[WanI2VSaved]()
    var block_modvecs = List[WanModVecs]()

    for bi in range(num_blocks):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)

        var bp = handle.prefix + "."
        var mod_key = bp + String("modulation")
        if not (mod_key in handle.block):
            raise Error(String("wan21-i2v block missing modulation: ") + mod_key)
        var block_mod_t = cast_tensor(handle.block[mod_key][], STDtype.F32, ctx)
        var block_mod_h = block_mod_t.to_host(ctx)

        var mv = _block_modvecs(e0_flat, block_mod_h, bi, S, dim)
        var w = _wan22_i2v_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)
        var bl = _wan_i2v_block_lora_for(lora, bi)
        var fwd = wan22_i2v_block_lora_forward[H, Dh, S, TXT, IMG](
            img.copy(), context_txt.copy(), context_img.copy(),
            mv, w, bl, cos_t, sin_t, dim, ffn, eps, ctx,
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

    return Wan22I2VStackForward(
        out^, block_saved^, block_modvecs^,
        img^, context_txt^, context_img^,
        e_head^, head_shift^, head_scale^,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# I2V FULL BACKWARD WITH LoRA, BLOCK-SWAP OFFLOAD (REVERSE block stream).
# Only the 12 LoRA d_A/d_B per block are collected (all base/img frozen-weight
# grads + d_context are discarded — umt5/CLIP/MLPProj frozen).
# ═══════════════════════════════════════════════════════════════════════════════
def wan22_i2v_stack_lora_backward_offload[
    H: Int, Dh: Int, S: Int, TXT: Int, IMG: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: Wan22StackBase,
    mut loader: TurboPlannedLoader, lora: Wan22I2VLoraSet,
    cos: List[Float32], sin: List[Float32],
    saved: Wan22I2VStackForward,
    dim: Int, ffn: Int, in_ch: Int, text_dim: Int, out_ch: Int,
    freq_dim: Int, eps: Float32, ctx: DeviceContext,
) raises -> Wan22I2VLoraGradSet:
    var num_blocks = lora.num_blocks
    var n_adapters = wan22_i2v_total_adapters(lora)

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S, Dh // 2], STDtype.F32, ctx)

    # ── head backward (identical to the base T2V stack) ──
    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.norm_backward import layer_norm_backward_dx
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
        compute_param_grads=False,
    )
    var d_ln_img = mbh.d_x.to_host(ctx)
    var lnbh = layer_norm_backward_dx(
        _t(d_ln_img, [S, dim], ctx), _t(saved.x_img.copy(), [S, dim], ctx),
        _t(ones.copy(), [dim], ctx), eps, ctx,
    )
    var d_x_img = lnbh.to_host(ctx)

    # ── init grad accumulators (12 slots/block) ──
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
        var w = _wan22_i2v_block_weights_from_block(handle.block, handle.prefix, dim, ffn, Dh, ctx)
        var bl = _wan_i2v_block_lora_for(lora, bi)

        var bg = wan22_i2v_block_lora_backward[H, Dh, S, TXT, IMG](
            d_x_img.copy(), mv, w, bl, saved.block_saved[bi],
            cos_t, sin_t, dim, ffn, eps, ctx,
        )

        d_x_img = bg.base.base.d_x.copy()

        var bbase = _i2v_lora_block_base(bi)
        d_a_flat[bbase + W_SA_Q] = bg.base.sa_q_da.copy()
        d_b_flat[bbase + W_SA_Q] = bg.base.sa_q_db.copy()
        d_a_flat[bbase + W_SA_K] = bg.base.sa_k_da.copy()
        d_b_flat[bbase + W_SA_K] = bg.base.sa_k_db.copy()
        d_a_flat[bbase + W_SA_V] = bg.base.sa_v_da.copy()
        d_b_flat[bbase + W_SA_V] = bg.base.sa_v_db.copy()
        d_a_flat[bbase + W_SA_O] = bg.base.sa_o_da.copy()
        d_b_flat[bbase + W_SA_O] = bg.base.sa_o_db.copy()
        d_a_flat[bbase + W_CA_Q] = bg.base.ca_q_da.copy()
        d_b_flat[bbase + W_CA_Q] = bg.base.ca_q_db.copy()
        d_a_flat[bbase + W_CA_K] = bg.base.ca_k_da.copy()
        d_b_flat[bbase + W_CA_K] = bg.base.ca_k_db.copy()
        d_a_flat[bbase + W_CA_V] = bg.base.ca_v_da.copy()
        d_b_flat[bbase + W_CA_V] = bg.base.ca_v_db.copy()
        d_a_flat[bbase + W_CA_O] = bg.base.ca_o_da.copy()
        d_b_flat[bbase + W_CA_O] = bg.base.ca_o_db.copy()
        d_a_flat[bbase + W_FFN0] = bg.base.ffn0_da.copy()
        d_b_flat[bbase + W_FFN0] = bg.base.ffn0_db.copy()
        d_a_flat[bbase + W_FFN2] = bg.base.ffn2_da.copy()
        d_b_flat[bbase + W_FFN2] = bg.base.ffn2_db.copy()
        d_a_flat[bbase + W_IMG_K] = bg.img_k_da.copy()
        d_b_flat[bbase + W_IMG_K] = bg.img_k_db.copy()
        d_a_flat[bbase + W_IMG_V] = bg.img_v_da.copy()
        d_b_flat[bbase + W_IMG_V] = bg.img_v_db.copy()

        nonfinite += _nonfinite(bg.base.sa_q_da) + _nonfinite(bg.base.sa_q_db)
        nonfinite += _nonfinite(bg.base.sa_k_da) + _nonfinite(bg.base.sa_k_db)
        nonfinite += _nonfinite(bg.base.sa_v_da) + _nonfinite(bg.base.sa_v_db)
        nonfinite += _nonfinite(bg.base.sa_o_da) + _nonfinite(bg.base.sa_o_db)
        nonfinite += _nonfinite(bg.base.ca_q_da) + _nonfinite(bg.base.ca_q_db)
        nonfinite += _nonfinite(bg.base.ca_k_da) + _nonfinite(bg.base.ca_k_db)
        nonfinite += _nonfinite(bg.base.ca_v_da) + _nonfinite(bg.base.ca_v_db)
        nonfinite += _nonfinite(bg.base.ca_o_da) + _nonfinite(bg.base.ca_o_db)
        nonfinite += _nonfinite(bg.base.ffn0_da) + _nonfinite(bg.base.ffn0_db)
        nonfinite += _nonfinite(bg.base.ffn2_da) + _nonfinite(bg.base.ffn2_db)
        nonfinite += _nonfinite(bg.img_k_da) + _nonfinite(bg.img_k_db)
        nonfinite += _nonfinite(bg.img_v_da) + _nonfinite(bg.img_v_db)

        loader.mark_active_block_done(ctx)
        bi -= 1

    # ── input projection backward (frozen patch-embed; grad reported not used) ──
    var lbi = linear_backward(
        _t_like(d_x_img, [S, dim], base.pe_w[], ctx),
        _t_like(img_tokens, [S, in_ch], base.pe_w[], ctx), base.pe_w[],
        S, in_ch, dim, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)

    return Wan22I2VLoraGradSet(d_a_flat^, d_b_flat^, d_img_tokens^, nonfinite)


# ── AdamW step on all 12/block I2V adapters (fused GPU plain-AdamW; same kernel
#    as the T2V path). The I2V backward fills ALL 12 slots per block, so this is
#    normally ONE fused call over [0, n). ─────────────────────────────────────
def wan22_i2v_lora_adamw_step(
    mut lora: Wan22I2VLoraSet, grads: Wan22I2VLoraGradSet, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps_opt: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    comptime if WAN22_GPU_ADAMW:
        var n = wan22_i2v_total_adapters(lora)
        var i = 0
        while i < n:
            if len(grads.d_a[i]) == 0:
                i += 1
                continue
            var run_start = i
            while i < n and len(grads.d_a[i]) != 0:
                i += 1
            fused_lora_adamw_plain_step(
                lora.ad, grads.d_a, grads.d_b, run_start, i,
                t, lr, beta1, beta2, eps_opt, weight_decay, ctx,
            )
    else:
        var n = wan22_i2v_total_adapters(lora)
        for i in range(n):
            if len(grads.d_a[i]) == 0:
                continue
            var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
            _lora_adamw(lora.ad[i], lg, t, lr, ctx, beta1, beta2, eps_opt, weight_decay)


# ── PEFT-keyed save (12 targets/block; diffusion_model.-prefixed) ─────────────
# The 2 extra keys vs T2V are cross_attn.k_img and cross_attn.v_img.
def _wan22_i2v_lora_prefixes(num_blocks: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_blocks):
        var b = String("diffusion_model.blocks.") + String(bi) + "."
        out.append(b + String("self_attn.q"))
        out.append(b + String("self_attn.k"))
        out.append(b + String("self_attn.v"))
        out.append(b + String("self_attn.o"))
        out.append(b + String("cross_attn.q"))
        out.append(b + String("cross_attn.k"))
        out.append(b + String("cross_attn.v"))
        out.append(b + String("cross_attn.o"))
        out.append(b + String("ffn.0"))
        out.append(b + String("ffn.2"))
        out.append(b + String("cross_attn.k_img"))
        out.append(b + String("cross_attn.v_img"))
    return out^


def save_wan22_i2v_lora(lora: Wan22I2VLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var prefixes = _wan22_i2v_lora_prefixes(lora.num_blocks)
    var named = List[NamedLora]()
    var n = wan22_i2v_total_adapters(lora)
    for i in range(n):
        named.append(NamedLora(prefixes[i], lora.ad[i].copy()))
    return save_lora_peft(named, path, ctx)


def save_wan22_i2v_lora_state(
    lora: Wan22I2VLoraSet, path: String, ctx: DeviceContext,
    var meta: List[Float32] = List[Float32](),
) raises -> Int:
    """RESUME state (A/B + AdamW moments) for the 12/block I2V LoRA."""
    var prefixes = _wan22_i2v_lora_prefixes(lora.num_blocks)
    var named = List[NamedLora]()
    var n = wan22_i2v_total_adapters(lora)
    for i in range(n):
        named.append(NamedLora(prefixes[i], lora.ad[i].copy()))
    return save_lora_train_state(named, path, ctx, meta^)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE-1b — device-native, LOADER-FREE block-stack (zimage-style rebuild).
#
# These mirror the SCAIL-2 loader-free stack (models/scail2/scail2_stack_lora.mojo)
# for the Wan2.2 T2V block (10-slot plain LoRA, no img_k/img_v). They operate on an
# explicit List[WanBlockWeights] + per-block WanModVecs (NO TurboPlannedLoader / no
# head / no embed), so a small same-process gate can compare:
#   (1) host recompute  ==  host save-all   (bit) — proves the recompute discipline;
#   (2) device-native   ==  host recompute  (bit) — proves the device tape.
#
# Everything here is behind the WAN_DEVNATIVE comptime flag at the trainer seam; the
# functions themselves are always compiled (gate-don't-delete, C13) and are pure
# additions — no existing path is touched.
# ══════════════════════════════════════════════════════════════════════════════

# ── loader-free forward tape: keep BOTH the per-block F32 INPUT tape (recompute)
#    and the save-all WanSaved tape (so one process can gate recompute==save-all). ─
struct Wan22BlockStackForward(Movable):
    var x_out: List[Float32]                 # [S,dim] F32 final block-stack output
    var block_inputs: List[List[Float32]]    # per-block F32 input sequence (recompute tape)
    var block_saved: List[WanSaved]          # per-block saved activations (save-all tape)

    def __init__(
        out self, var x_out: List[Float32],
        var block_inputs: List[List[Float32]], var block_saved: List[WanSaved],
    ):
        self.x_out = x_out^
        self.block_inputs = block_inputs^
        self.block_saved = block_saved^


# ── device-resident LoRA grad set (10 slots/block, F32 device tensors) ────────
# d_a[i]/d_b[i] are the flat adapter-order (block bi -> i = bi*10 + slot) device
# grads; grad_indices are the matching absolute adapter indices (0..n-1, dense).
# Feeds fused_lora_adamw_plain_step_resident_device_grads with NO host round-trip.
struct Wan22LoraDeviceGradSet(Movable):
    var d_a: List[TArc]            # [n_adapters] F32 [rank,in]  device
    var d_b: List[TArc]            # [n_adapters] F32 [out,rank] device
    var grad_indices: List[Int]    # [n_adapters] absolute adapter indices (dense)
    var d_x_tokens: List[Float32]  # [S,dim] F32 input-sequence grad (host readback)
    var num_adapters: Int

    def __init__(
        out self, var d_a: List[TArc], var d_b: List[TArc],
        var grad_indices: List[Int], var d_x_tokens: List[Float32],
        num_adapters: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.grad_indices = grad_indices^
        self.d_x_tokens = d_x_tokens^
        self.num_adapters = num_adapters


# ── loader-free forward (records both tapes) ──────────────────────────────────
def wan22_blockstack_lora_forward[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    var sequence: List[Float32],
    modvecs: List[WanModVecs],
    context: List[Float32],
    weights: List[WanBlockWeights], lora: Wan22LoraSet,
    cos: Tensor, sin: Tensor,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
) raises -> Wan22BlockStackForward:
    var num_layers = lora.num_blocks
    var x = sequence^
    var block_inputs = List[List[Float32]]()
    var block_saved = List[WanSaved]()
    for bi in range(num_layers):
        block_inputs.append(x.copy())
        var bl = _wan_block_lora_for(lora, bi)
        var fwd = wan22_block_lora_forward[H, Dh, S, TXT](
            x.copy(), context.copy(), modvecs[bi].copy(), weights[bi], bl,
            cos, sin, dim, ffn, eps, ctx,
        )
        block_saved.append(fwd.saved.copy())
        x = fwd.x_out.copy()
    return Wan22BlockStackForward(x^, block_inputs^, block_saved^)


# ── scatter one block's 10 host LoRA grads into the flat accumulators ──────────
def _wan_scatter_block_host_grads(
    mut d_a: List[List[Float32]], mut d_b: List[List[Float32]],
    bbase: Int, bg: WanBlockLoraGrads,
) raises -> Int:
    d_a[bbase + W_SA_Q] = bg.sa_q_da.copy(); d_b[bbase + W_SA_Q] = bg.sa_q_db.copy()
    d_a[bbase + W_SA_K] = bg.sa_k_da.copy(); d_b[bbase + W_SA_K] = bg.sa_k_db.copy()
    d_a[bbase + W_SA_V] = bg.sa_v_da.copy(); d_b[bbase + W_SA_V] = bg.sa_v_db.copy()
    d_a[bbase + W_SA_O] = bg.sa_o_da.copy(); d_b[bbase + W_SA_O] = bg.sa_o_db.copy()
    d_a[bbase + W_CA_Q] = bg.ca_q_da.copy(); d_b[bbase + W_CA_Q] = bg.ca_q_db.copy()
    d_a[bbase + W_CA_K] = bg.ca_k_da.copy(); d_b[bbase + W_CA_K] = bg.ca_k_db.copy()
    d_a[bbase + W_CA_V] = bg.ca_v_da.copy(); d_b[bbase + W_CA_V] = bg.ca_v_db.copy()
    d_a[bbase + W_CA_O] = bg.ca_o_da.copy(); d_b[bbase + W_CA_O] = bg.ca_o_db.copy()
    d_a[bbase + W_FFN0] = bg.ffn0_da.copy(); d_b[bbase + W_FFN0] = bg.ffn0_db.copy()
    d_a[bbase + W_FFN2] = bg.ffn2_da.copy(); d_b[bbase + W_FFN2] = bg.ffn2_db.copy()
    var nf = 0
    nf += _nonfinite(bg.sa_q_da) + _nonfinite(bg.sa_q_db)
    nf += _nonfinite(bg.sa_k_da) + _nonfinite(bg.sa_k_db)
    nf += _nonfinite(bg.sa_v_da) + _nonfinite(bg.sa_v_db)
    nf += _nonfinite(bg.sa_o_da) + _nonfinite(bg.sa_o_db)
    nf += _nonfinite(bg.ca_q_da) + _nonfinite(bg.ca_q_db)
    nf += _nonfinite(bg.ca_k_da) + _nonfinite(bg.ca_k_db)
    nf += _nonfinite(bg.ca_v_da) + _nonfinite(bg.ca_v_db)
    nf += _nonfinite(bg.ca_o_da) + _nonfinite(bg.ca_o_db)
    nf += _nonfinite(bg.ffn0_da) + _nonfinite(bg.ffn0_db)
    nf += _nonfinite(bg.ffn2_da) + _nonfinite(bg.ffn2_db)
    return nf


# ── loader-free HOST backward (recompute flag; save-all when False) ───────────
# The oracle for both device-native and the recompute-discipline gate. `recompute`
# = True regenerates each block's WanSaved from its F32 input; False consumes the
# save-all tape. Returns the same Wan22LoraGradSet the offload path returns.
def wan22_blockstack_lora_backward[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    var d_out: List[Float32],
    modvecs: List[WanModVecs],
    context: List[Float32],
    weights: List[WanBlockWeights], lora: Wan22LoraSet,
    cos: Tensor, sin: Tensor,
    fwd_state: Wan22BlockStackForward,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
    recompute: Bool = True,
) raises -> Wan22LoraGradSet:
    var num_layers = lora.num_blocks
    var n_adapters = wan22_total_adapters(lora)
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a.append(List[Float32]())
        d_b.append(List[Float32]())
    var nonfinite = 0
    var d_cur = d_out^

    var bi = num_layers - 1
    while bi >= 0:
        var bl = _wan_block_lora_for(lora, bi)
        if recompute:
            var rc = wan22_block_lora_forward[H, Dh, S, TXT](
                fwd_state.block_inputs[bi].copy(), context.copy(),
                modvecs[bi].copy(), weights[bi], bl, cos, sin,
                dim, ffn, eps, ctx,
            )
            var bg = wan22_block_lora_backward[H, Dh, S, TXT](
                d_cur.copy(), modvecs[bi].copy(), weights[bi], bl, rc.saved,
                cos, sin, dim, ffn, eps, ctx,
            )
            d_cur = bg.base.d_x.copy()
            nonfinite += _wan_scatter_block_host_grads(d_a, d_b, _lora_block_base(bi), bg)
        else:
            var bg = wan22_block_lora_backward[H, Dh, S, TXT](
                d_cur.copy(), modvecs[bi].copy(), weights[bi], bl,
                fwd_state.block_saved[bi], cos, sin, dim, ffn, eps, ctx,
            )
            d_cur = bg.base.d_x.copy()
            nonfinite += _wan_scatter_block_host_grads(d_a, d_b, _lora_block_base(bi), bg)
        bi -= 1

    return Wan22LoraGradSet(d_a^, d_b^, d_cur^, _zeros_f32(TXT * dim), nonfinite)


# ── loader-free DEVICE-NATIVE backward (recompute; device grads threaded) ─────
# Mirrors wan22_blockstack_lora_backward (recompute=True) but calls the device-
# native block backward wan22_block_lora_backward_devnative: d_x stays a device
# BF16 Tensor threaded block->block (no to_host inside the loop), and each block's
# 10 device LoRA grads (F32 TArc) accumulate into the flat device grad set. Only
# the final input-sequence grad + per-adapter grads leave as device tensors; the
# single d_x_tokens readback at the end is for parity/inspection.
def wan22_blockstack_lora_backward_devnative[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    d_out: List[Float32],
    modvecs: List[WanModVecs],
    context: List[Float32],
    weights: List[WanBlockWeights], lora: Wan22LoraSet,
    cos: Tensor, sin: Tensor,
    fwd_state: Wan22BlockStackForward,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
) raises -> Wan22LoraDeviceGradSet:
    var num_layers = lora.num_blocks
    var n_adapters = wan22_total_adapters(lora)

    # dense per-adapter device grad slots (all 10/block are filled by plain LoRA).
    var da_opt = List[Optional[TArc]]()
    var db_opt = List[Optional[TArc]]()
    for _ in range(n_adapters):
        da_opt.append(Optional[TArc]())
        db_opt.append(Optional[TArc]())

    # seed the reverse chain with the output grad as a device BF16 tensor (== the
    # oracle's _ta16 upload of d_out_h).
    var d_out_dev = Tensor.from_host(d_out.copy(), [S, dim], STDtype.BF16, ctx)

    var bi = num_layers - 1
    while bi >= 0:
        var bl = _wan_block_lora_for(lora, bi)
        # RECOMPUTE this block's forward from its F32 input (one device tape live).
        var rc = wan22_block_lora_forward[H, Dh, S, TXT](
            fwd_state.block_inputs[bi].copy(), context.copy(),
            modvecs[bi].copy(), weights[bi], bl, cos, sin, dim, ffn, eps, ctx,
        )
        var gd = wan22_block_lora_backward_devnative[H, Dh, S, TXT](
            d_out_dev, modvecs[bi].copy(), weights[bi], bl, rc.saved,
            cos, sin, dim, ffn, eps, ctx,
        )
        # d_x(bi) -> d_out(bi-1), device BF16, no host round-trip.
        d_out_dev = gd.d_x.clone(ctx)
        # scatter this block's 10 device grads into flat adapter order.
        var bbase = _lora_block_base(bi)
        for slot in range(WAN_LORA_SLOTS):
            if gd.d_a[slot]:
                da_opt[bbase + slot] = Optional[TArc](gd.d_a[slot].value().copy())
                db_opt[bbase + slot] = Optional[TArc](gd.d_b[slot].value().copy())
        bi -= 1

    # flatten dense slots -> ascending device grad lists + dense grad indices.
    var d_a = List[TArc]()
    var d_b = List[TArc]()
    var grad_indices = List[Int]()
    for i in range(n_adapters):
        if not da_opt[i]:
            raise Error(
                "wan22_blockstack_lora_backward_devnative: missing device grad slot "
                + String(i)
            )
        d_a.append(da_opt[i].value().copy())
        d_b.append(db_opt[i].value().copy())
        grad_indices.append(i)

    var d_x_tokens = d_out_dev.to_host(ctx)
    return Wan22LoraDeviceGradSet(
        d_a^, d_b^, grad_indices^, d_x_tokens^, n_adapters,
    )
