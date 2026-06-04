# serenitymojo/models/qwenimage/qwenimage_stack_lora.mojo
#
# Qwen-Image MMDiT stack WITH LoRA on every trained projection: a flat LoRA set
# across all 60 double blocks (12 adapters/block: img/txt x q/k/v/out/ff_up/ff_down),
# the LoRA stack fwd+bwd (per-block recompute), grad scatter, AdamW step, and PEFT
# save. Mirrors models/klein/klein_stack_lora.mojo, specialized to Qwen-Image's
# 12-target double block. REUSES the shared training/ LoRA math (LoraAdapter,
# _lora_adamw) + lora_save.save_lora_peft — does NOT fork it.
#
# SLOT order per block (DBL_SLOTS = 12): img q,k,v,out,ff_up,ff_down then
# txt q,k,v,out,ff_up,ff_down. LoRA save prefixes use the diffusers transformer
# key layout (transformer_blocks.{i}.attn.to_q, ...; .img_mlp.net.0.proj, ...)
# matching EDv2 qwenimage.rs::lora_module_path (qwenimage.rs:419-430).
#
# 2026-06-04: Added qwenimage_stack_lora_forward_offload +
#   qwenimage_stack_lora_backward_offload — TurboPlannedLoader block-swap offload
#   (mirrors chroma_stack_lora.mojo). Per-block mod-MLP weights (img_mod.1 /
#   txt_mod.1) are read from the streamed Block each iteration (cast F32 host).
#   Frozen mod-MLP grads are discarded (LoRA-scope: only d_A/d_B collected).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.linalg_backward import linear_backward

from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

from serenitymojo.models.klein.lora_block import LoraAdapter
from serenitymojo.training.train_step import _lora_adamw, LoraGrads
from serenitymojo.training.lora_save import NamedLora, save_lora_peft

from serenitymojo.models.qwenimage.qwenimage_block import (
    DoubleBlockWeights, ModVecs, StreamLora, DoubleBlockLora,
    StreamLoraGrads, DoubleBlockLoraGrads,
    double_block_lora_forward, double_block_lora_backward,
    DoubleBlockLoraForward, DoubleBlockSaved,
    StreamWeights,
)
from serenitymojo.models.qwenimage.qwenimage_stack import (
    QwenStackBase, QwenStackForward, QwenStackGrads,
    _zeros as _qstack_zeros, _ones as _qstack_ones, _t as _qstack_t,
    _linear_b,
)


comptime TArc = ArcPointer[Tensor]
comptime DBL_SLOTS = 12   # img{q,k,v,out,ffu,ffd} + txt{q,k,v,out,ffu,ffd}


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def make_lora_adapter(
    rank: Int, alpha: Float32, in_f: Int, out_f: Int, seed: UInt64
) -> LoraAdapter:
    var scale = alpha / Float32(rank)
    return LoraAdapter(
        _randn(rank * in_f, seed, 0.01),   # A small randn
        _zeros(out_f * rank),              # B = 0 (PEFT identity at init)
        rank, in_f, out_f, scale,
        _zeros(rank * in_f), _zeros(rank * in_f),
        _zeros(out_f * rank), _zeros(out_f * rank),
    )


# ── the LoRA carrier: every trained adapter, flat-indexed ────────────────────
struct QwenLoraSet(Copyable, Movable):
    var dbl: List[LoraAdapter]   # num_double * DBL_SLOTS
    var num_double: Int
    var rank: Int

    def __init__(out self, var dbl: List[LoraAdapter], num_double: Int, rank: Int):
        self.dbl = dbl^
        self.num_double = num_double
        self.rank = rank


# Build the full LoRA set. Slot in/out:
#   q/k/v/out: in=D out=D ; ff_up: in=D out=F ; ff_down: in=F out=D.
def build_qwen_lora_set(
    num_double: Int, D: Int, F: Int, rank: Int, alpha: Float32
) -> QwenLoraSet:
    var dbl = List[LoraAdapter]()
    var seed = UInt64(7000)
    for _ in range(num_double):
        # img: q,k,v,out (D,D) ; ff_up (D,F) ; ff_down (F,D)
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
        dbl.append(make_lora_adapter(rank, alpha, D, F, seed)); seed += 1
        dbl.append(make_lora_adapter(rank, alpha, F, D, seed)); seed += 1
        # txt: same six
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
        dbl.append(make_lora_adapter(rank, alpha, D, F, seed)); seed += 1
        dbl.append(make_lora_adapter(rank, alpha, F, D, seed)); seed += 1
    return QwenLoraSet(dbl^, num_double, rank)


# build a transient DoubleBlockLora for block bi from the flat set.
def double_lora_for(set: QwenLoraSet, bi: Int) -> DoubleBlockLora:
    var base = bi * DBL_SLOTS
    var img = StreamLora(
        Optional[LoraAdapter](set.dbl[base + 0].copy()),
        Optional[LoraAdapter](set.dbl[base + 1].copy()),
        Optional[LoraAdapter](set.dbl[base + 2].copy()),
        Optional[LoraAdapter](set.dbl[base + 3].copy()),
        Optional[LoraAdapter](set.dbl[base + 4].copy()),
        Optional[LoraAdapter](set.dbl[base + 5].copy()),
    )
    var txt = StreamLora(
        Optional[LoraAdapter](set.dbl[base + 6].copy()),
        Optional[LoraAdapter](set.dbl[base + 7].copy()),
        Optional[LoraAdapter](set.dbl[base + 8].copy()),
        Optional[LoraAdapter](set.dbl[base + 9].copy()),
        Optional[LoraAdapter](set.dbl[base + 10].copy()),
        Optional[LoraAdapter](set.dbl[base + 11].copy()),
    )
    return DoubleBlockLora(img^, txt^)


# transient list of per-block DoubleBlockLora for the LoRA stack forward/backward.
def lora_list_from_set(set: QwenLoraSet) -> List[DoubleBlockLora]:
    var out = List[DoubleBlockLora]()
    for bi in range(set.num_double):
        out.append(double_lora_for(set, bi))
    return out^


# ── grad scatter: stack-lora backward grads -> flat per-adapter LoraGrads ─────
# Per slot, build a LoraGrads (d_a, d_b) for the optimizer.
def _slot_grads(da: List[Float32], db: List[Float32]) -> LoraGrads:
    return LoraGrads(da.copy(), db.copy())


# Apply AdamW to every adapter in the set using the per-block stream grads.
# dbl_lora_grads: num_double StreamLoraGrads pairs (img, txt) packed as
# DoubleBlockLoraGrads.img / .txt.
def qwen_lora_adamw_step(
    mut set: QwenLoraSet,
    img_lora_grads: List[StreamLoraGrads], txt_lora_grads: List[StreamLoraGrads],
    t: Int, lr: Float32, ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    for bi in range(set.num_double):
        var base = bi * DBL_SLOTS
        var ig = img_lora_grads[bi].copy()
        var tg = txt_lora_grads[bi].copy()
        # img slots 0..5
        _lora_adamw(set.dbl[base + 0], _slot_grads(ig.q_d_a, ig.q_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        _lora_adamw(set.dbl[base + 1], _slot_grads(ig.k_d_a, ig.k_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        _lora_adamw(set.dbl[base + 2], _slot_grads(ig.v_d_a, ig.v_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        _lora_adamw(set.dbl[base + 3], _slot_grads(ig.out_d_a, ig.out_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        _lora_adamw(set.dbl[base + 4], _slot_grads(ig.ff_up_d_a, ig.ff_up_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        _lora_adamw(set.dbl[base + 5], _slot_grads(ig.ff_down_d_a, ig.ff_down_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        # txt slots 6..11
        _lora_adamw(set.dbl[base + 6], _slot_grads(tg.q_d_a, tg.q_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        _lora_adamw(set.dbl[base + 7], _slot_grads(tg.k_d_a, tg.k_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        _lora_adamw(set.dbl[base + 8], _slot_grads(tg.v_d_a, tg.v_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        _lora_adamw(set.dbl[base + 9], _slot_grads(tg.out_d_a, tg.out_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        _lora_adamw(set.dbl[base + 10], _slot_grads(tg.ff_up_d_a, tg.ff_up_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)
        _lora_adamw(set.dbl[base + 11], _slot_grads(tg.ff_down_d_a, tg.ff_down_d_b), t, lr, ctx, beta1, beta2, eps, weight_decay)


# ── PEFT save (diffusers transformer key layout) ─────────────────────────────
# slot -> module suffix; prefix = "transformer_blocks.{bi}.<suffix>".
def _slot_suffix(slot: Int) -> String:
    # img stream
    if slot == 0: return String("attn.to_q")
    if slot == 1: return String("attn.to_k")
    if slot == 2: return String("attn.to_v")
    if slot == 3: return String("attn.to_out.0")
    if slot == 4: return String("img_mlp.net.0.proj")
    if slot == 5: return String("img_mlp.net.2")
    # txt stream
    if slot == 6: return String("attn.add_q_proj")
    if slot == 7: return String("attn.add_k_proj")
    if slot == 8: return String("attn.add_v_proj")
    if slot == 9: return String("attn.to_add_out")
    if slot == 10: return String("txt_mlp.net.0.proj")
    return String("txt_mlp.net.2")


def _qwen_lora_prefix(block_idx: Int, slot: Int) -> String:
    return String("transformer_blocks.") + String(block_idx) + "." + _slot_suffix(slot)


def save_qwen_lora(set: QwenLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.num_double):
        for s in range(DBL_SLOTS):
            named.append(NamedLora(
                _qwen_lora_prefix(bi, s),
                set.dbl[bi * DBL_SLOTS + s].copy(),
            ))
    return save_lora_peft(named^, path, ctx)


# ── LoRA grad accumulator (flat d_a/d_b per adapter + nonfinite counter) ─────
struct QwenLoraGradSet(Movable):
    var d_a: List[List[Float32]]    # num_double * DBL_SLOTS
    var d_b: List[List[Float32]]
    var d_img_tokens: List[Float32]
    var d_txt_tokens: List[Float32]
    var nonfinite_lora_grads: Int

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.nonfinite_lora_grads = nonfinite_lora_grads


# ── Qwen offload forward tape (checkpoint inputs + saved activations) ─────────
# Stores each block's (img,txt) INPUT (host F32) and the DoubleBlockLoraForward
# saved activations (device-resident TArc tensors). The backward uses the saved
# activations directly — no per-block recompute needed under offload (the block
# weights are re-streamed anyway).
struct QwenOffloadForward(Movable):
    var out: List[Float32]                    # [N_IMG, out_ch] final output
    var dbl_img_in: List[List[Float32]]       # num_double x [N_IMG, D] block inputs
    var dbl_txt_in: List[List[Float32]]       # num_double x [N_TXT, D] block inputs
    var dbl_saved: List[DoubleBlockSaved]     # num_double saved activations
    var img_out: ArcPointer[Tensor]           # [N_IMG, D] last block img output
    var ln_img_out: ArcPointer[Tensor]        # [N_IMG, D] layer_norm(img_out)
    var final_scale: List[Float32]            # [D] from norm_out.linear
    var final_shift: List[Float32]            # [D]

    def __init__(
        out self,
        var out: List[Float32],
        var dbl_img_in: List[List[Float32]], var dbl_txt_in: List[List[Float32]],
        var dbl_saved: List[DoubleBlockSaved],
        var img_out: ArcPointer[Tensor], var ln_img_out: ArcPointer[Tensor],
        var final_scale: List[Float32], var final_shift: List[Float32],
    ):
        self.out = out^
        self.dbl_img_in = dbl_img_in^
        self.dbl_txt_in = dbl_txt_in^
        self.dbl_saved = dbl_saved^
        self.img_out = img_out^
        self.ln_img_out = ln_img_out^
        self.final_scale = final_scale^
        self.final_shift = final_shift^


# ── helpers for reading from Block ────────────────────────────────────────────

def _block_f32_host(block: Block, key: String, ctx: DeviceContext) raises -> List[Float32]:
    if not (key in block):
        raise Error(String("QwenImage offload block missing tensor: ") + key)
    return cast_tensor(block[key][], STDtype.F32, ctx).to_host(ctx)


def _nonfinite_check(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# Build StreamWeights for one stream (img or txt) from a block.
def _stream_weights_from_block_offload(
    block: Block, bp: String, is_img: Bool,
    D: Int, F: Int, Dh: Int, ctx: DeviceContext,
) raises -> StreamWeights:
    var qp: String
    var kp: String
    var vp: String
    var op: String
    var nqp: String
    var nkp: String
    var mlp: String
    if is_img:
        qp = ".attn.to_q"
        kp = ".attn.to_k"
        vp = ".attn.to_v"
        op = ".attn.to_out.0"
        nqp = ".attn.norm_q"
        nkp = ".attn.norm_k"
        mlp = ".img_mlp"
    else:
        qp = ".attn.add_q_proj"
        kp = ".attn.add_k_proj"
        vp = ".attn.add_v_proj"
        op = ".attn.to_add_out"
        nqp = ".attn.norm_added_q"
        nkp = ".attn.norm_added_k"
        mlp = ".txt_mlp"
    return StreamWeights(
        _block_f32_host(block, bp + qp + ".weight", ctx),
        _block_f32_host(block, bp + kp + ".weight", ctx),
        _block_f32_host(block, bp + vp + ".weight", ctx),
        _block_f32_host(block, bp + qp + ".bias", ctx),
        _block_f32_host(block, bp + kp + ".bias", ctx),
        _block_f32_host(block, bp + vp + ".bias", ctx),
        _block_f32_host(block, bp + op + ".weight", ctx),
        _block_f32_host(block, bp + op + ".bias", ctx),
        _block_f32_host(block, bp + mlp + ".net.0.proj.weight", ctx),
        _block_f32_host(block, bp + mlp + ".net.0.proj.bias", ctx),
        _block_f32_host(block, bp + mlp + ".net.2.weight", ctx),
        _block_f32_host(block, bp + mlp + ".net.2.bias", ctx),
        _block_f32_host(block, bp + nqp + ".weight", ctx),
        _block_f32_host(block, bp + nkp + ".weight", ctx),
        D, F, Dh, ctx,
    )


def _double_block_weights_from_block(
    block: Block, bp: String, D: Int, F: Int, Dh: Int, ctx: DeviceContext,
) raises -> DoubleBlockWeights:
    var img = _stream_weights_from_block_offload(block, bp, True, D, F, Dh, ctx)
    var txt = _stream_weights_from_block_offload(block, bp, False, D, F, Dh, ctx)
    return DoubleBlockWeights(img^, txt^)


# Compute per-block ModVecs from the streamed block's frozen mod-MLP.
# temb_h [1, D] silu-activated timestep embedding.
def _modvecs_from_block(
    block: Block, bp: String, is_img: Bool,
    temb_h: List[Float32], D: Int, ctx: DeviceContext,
) raises -> ModVecs:
    var mk: String
    if is_img:
        mk = ".img_mod.1"
    else:
        mk = ".txt_mod.1"
    var mw = _block_f32_host(block, bp + mk + ".weight", ctx)  # [6D, D]
    var mb = _block_f32_host(block, bp + mk + ".bias", ctx)    # [6D]
    var temb = Tensor.from_host(temb_h.copy(), [1, D], STDtype.F32, ctx)
    # temb_h is already silu-activated (caller pre-activates once)
    var bt = Tensor.from_host(mb^, [6 * D], STDtype.F32, ctx)
    var mods = linear(
        temb,
        Tensor.from_host(mw^, [6 * D, D], STDtype.F32, ctx),
        Optional[Tensor](bt^), ctx,
    ).to_host(ctx)   # [1, 6D]
    # chunk order: shift1, scale1, gate1, shift2, scale2, gate2
    var out = List[List[Float32]]()
    for c in range(6):
        var chunk = List[Float32]()
        for i in range(D):
            chunk.append(mods[c * D + i])
        out.append(chunk^)
    return ModVecs(
        out[0].copy(), out[1].copy(), out[2].copy(),
        out[3].copy(), out[4].copy(), out[5].copy(),
    )


# Compute final scale/shift from norm_out.linear in the base stack.
# base_st: ShardedSafeTensors wrapping the checkpoint (passed as List[Float32]
# already loaded by the trainer from base).
# We receive pre-loaded final_scale/final_shift from the trainer, so this
# function takes the temb and the loaded norm_out weights.
def _compute_final_modvecs(
    silu_temb_h: List[Float32],
    norm_out_w: List[Float32], norm_out_b: List[Float32],
    D: Int, ctx: DeviceContext,
) raises -> List[List[Float32]]:
    var temb = Tensor.from_host(silu_temb_h.copy(), [1, D], STDtype.F32, ctx)
    var bt = Tensor.from_host(norm_out_b.copy(), [2 * D], STDtype.F32, ctx)
    var fmods = linear(
        temb,
        Tensor.from_host(norm_out_w.copy(), [2 * D, D], STDtype.F32, ctx),
        Optional[Tensor](bt^), ctx,
    ).to_host(ctx)   # [1, 2D]
    var fscale = List[Float32]()
    var fshift = List[Float32]()
    for i in range(D):
        fscale.append(fmods[i])
        fshift.append(fmods[D + i])
    var out = List[List[Float32]]()
    out.append(fscale^)
    out.append(fshift^)
    return out^


# ── Qwen-Image offload base (frozen non-block weights + norm_out MLP) ─────────
# Extended from QwenStackBase to carry the norm_out linear for final modvec.
struct QwenOffloadBase(Movable):
    var stack: QwenStackBase
    var norm_out_w: List[Float32]    # [2D, D]  norm_out.linear.weight
    var norm_out_b: List[Float32]    # [2D]     norm_out.linear.bias
    var te_lin1_w: List[Float32]     # [D, timestep_dim]  timestep MLP linear_1
    var te_lin1_b: List[Float32]     # [D]
    var te_lin2_w: List[Float32]     # [D, D]   timestep MLP linear_2
    var te_lin2_b: List[Float32]     # [D]

    def __init__(
        out self,
        var stack: QwenStackBase,
        var norm_out_w: List[Float32], var norm_out_b: List[Float32],
        var te_lin1_w: List[Float32], var te_lin1_b: List[Float32],
        var te_lin2_w: List[Float32], var te_lin2_b: List[Float32],
    ):
        self.stack = stack^
        self.norm_out_w = norm_out_w^
        self.norm_out_b = norm_out_b^
        self.te_lin1_w = te_lin1_w^
        self.te_lin1_b = te_lin1_b^
        self.te_lin2_w = te_lin2_w^
        self.te_lin2_b = te_lin2_b^


# Compute silu(temb) from a sinusoidal embedding input t (already pre-embedded).
# temb_sinusoidal_h: [1, timestep_dim] sinusoidal embedding (host F32).
# Returns silu(MLP(temb)) as host [1, D] for per-block modvec compute.
def compute_silu_temb(
    base: QwenOffloadBase, temb_sinusoidal_h: List[Float32],
    timestep_dim: Int, D: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var t_emb = Tensor.from_host(temb_sinusoidal_h.copy(), [1, timestep_dim], STDtype.F32, ctx)
    var b1 = Tensor.from_host(base.te_lin1_b.copy(), [D], STDtype.F32, ctx)
    var h1 = linear(
        t_emb,
        Tensor.from_host(base.te_lin1_w.copy(), [D, timestep_dim], STDtype.F32, ctx),
        Optional[Tensor](b1^), ctx,
    )
    var h1_silu = silu(h1, ctx)
    var b2 = Tensor.from_host(base.te_lin2_b.copy(), [D], STDtype.F32, ctx)
    var temb_out = linear(
        h1_silu,
        Tensor.from_host(base.te_lin2_w.copy(), [D, D], STDtype.F32, ctx),
        Optional[Tensor](b2^), ctx,
    )
    # silu the output for use as per-block mod MLP input
    return silu(temb_out, ctx).to_host(ctx)


# ═════════════════════════════════════════════════════════════════════════════
# FULL FORWARD WITH LoRA, BLOCK-SWAP OFFLOAD (60 double blocks).
#
#   img_tokens [N_IMG, in_ch], txt_tokens [N_TXT, txt_ch].
#   silu_temb_h [1, D]: silu(time_text_embed(t)), pre-computed once per step.
#   cos/sin [S*H, Dh//2]: 3-axis interleaved RoPE tables (host F32).
#   norm_out_w/norm_out_b: top-level norm_out.linear weights (pre-loaded).
#   Streams 60 double blocks one at a time via TurboPlannedLoader.
# ═════════════════════════════════════════════════════════════════════════════
def qwenimage_stack_lora_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    silu_temb_h: List[Float32],
    base: QwenOffloadBase,
    mut loader: TurboPlannedLoader, lora: QwenLoraSet,
    cos: List[Float32], sin: List[Float32],
    norm_out_w: List[Float32], norm_out_b: List[Float32],
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> QwenOffloadForward:
    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.elementwise import modulate

    var num_double = lora.num_double

    loader.prefetch_with_ctx(0, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # input projections (frozen base)
    var img = _linear_b(img_tokens, base.stack.img_in_w[], base.stack.img_in_b[], N_IMG, in_ch, ctx)
    var txt = _linear_b(txt_tokens, base.stack.txt_in_w[], base.stack.txt_in_b[], N_TXT, txt_ch, ctx)

    var dbl_img_in = List[List[Float32]]()
    var dbl_txt_in = List[List[Float32]]()
    var dbl_saved = List[DoubleBlockSaved]()

    for bi in range(num_double):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var bp = handle.prefix
        var w = _double_block_weights_from_block(handle.block, bp + String("."), D, F, Dh, ctx)
        var img_mod = _modvecs_from_block(handle.block, bp, True, silu_temb_h, D, ctx)
        var txt_mod = _modvecs_from_block(handle.block, bp, False, silu_temb_h, D, ctx)
        var bl = double_lora_for(lora, bi)

        dbl_img_in.append(img.copy())
        dbl_txt_in.append(txt.copy())

        var fwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), w, img_mod, txt_mod, bl,
            cos_t, sin_t, D, F, eps, ctx,
        )
        dbl_saved.append(fwd.saved.copy())
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)

    # final layer: layer_norm -> modulate(final_scale, final_shift) -> proj_out
    var final_mods = _compute_final_modvecs(silu_temb_h, norm_out_w, norm_out_b, D, ctx)
    var final_scale = final_mods[0].copy()
    var final_shift = final_mods[1].copy()

    var img_t = Tensor.from_host(img.copy(), [N_IMG, D], STDtype.F32, ctx)
    var ln_img_out_t = layer_norm(
        img_t,
        Tensor.from_host(_qstack_ones(D), [D], STDtype.F32, ctx),
        Tensor.from_host(_qstack_zeros(D), [D], STDtype.F32, ctx),
        eps, ctx,
    )
    var ln_img_out_h = ln_img_out_t.to_host(ctx)
    var normed = modulate(
        Tensor.from_host(ln_img_out_h.copy(), [N_IMG, D], STDtype.F32, ctx),
        Tensor.from_host(final_scale.copy(), [D], STDtype.F32, ctx),
        Tensor.from_host(final_shift.copy(), [D], STDtype.F32, ctx),
        ctx,
    ).to_host(ctx)
    var out = _linear_b(normed, base.stack.proj_out_w[], base.stack.proj_out_b[], N_IMG, D, ctx)

    var img_arc = ArcPointer[Tensor](Tensor.from_host(img^, [N_IMG, D], STDtype.F32, ctx))
    var ln_arc = ArcPointer[Tensor](Tensor.from_host(ln_img_out_h^, [N_IMG, D], STDtype.F32, ctx))

    return QwenOffloadForward(
        out^, dbl_img_in^, dbl_txt_in^, dbl_saved^,
        img_arc^, ln_arc^, final_scale^, final_shift^,
    )


# ═════════════════════════════════════════════════════════════════════════════
# FULL BACKWARD WITH LoRA, BLOCK-SWAP OFFLOAD (REVERSE block stream).
#   Frozen mod-MLP scope: per-block modvec grads are DISCARDED.
#   Only LoRA d_A/d_B are collected (12 slots x 60 blocks = 720 adapters).
# ═════════════════════════════════════════════════════════════════════════════
def qwenimage_stack_lora_backward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    silu_temb_h: List[Float32],
    base: QwenOffloadBase,
    mut loader: TurboPlannedLoader, lora: QwenLoraSet,
    cos: List[Float32], sin: List[Float32],
    norm_out_w: List[Float32], norm_out_b: List[Float32],
    saved: QwenOffloadForward,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> QwenLoraGradSet:
    from serenitymojo.ops.norm import layer_norm
    from serenitymojo.ops.elementwise import modulate
    from serenitymojo.ops.norm_backward import layer_norm_backward
    from serenitymojo.ops.elementwise_backward import modulate_backward

    var num_double = lora.num_double
    var n_adapters = num_double * DBL_SLOTS

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # flat grad accumulators (one entry per adapter)
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    var nonfinite = 0

    # ── final layer backward: proj_out -> modulate -> layer_norm ──────────────
    var final_scale = saved.final_scale.copy()
    var final_shift = saved.final_shift.copy()

    var normed = modulate(
        saved.ln_img_out[],
        Tensor.from_host(final_scale.copy(), [D], STDtype.F32, ctx),
        Tensor.from_host(final_shift.copy(), [D], STDtype.F32, ctx),
        ctx,
    ).to_host(ctx)

    var lbf = linear_backward(
        Tensor.from_host(d_out, [N_IMG, out_ch], STDtype.F32, ctx),
        Tensor.from_host(normed, [N_IMG, D], STDtype.F32, ctx),
        base.stack.proj_out_w[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)

    var mbf = modulate_backward(
        Tensor.from_host(d_normed, [N_IMG, D], STDtype.F32, ctx),
        saved.ln_img_out[],
        Tensor.from_host(final_scale.copy(), [D], STDtype.F32, ctx),
        ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)

    var lnbf = layer_norm_backward(
        Tensor.from_host(d_ln_img_out, [N_IMG, D], STDtype.F32, ctx),
        saved.img_out[],
        Tensor.from_host(_qstack_ones(D), [D], STDtype.F32, ctx),
        eps, ctx,
    )
    var d_img_out = lnbf.d_x.to_host(ctx)
    var d_txt_out = _qstack_zeros(N_TXT * D)

    # ── double-stream backward (REVERSE; LoRA; streamed weights) ───────────────
    var di = num_double - 1
    while di >= 0:
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var bp = handle.prefix
        var w = _double_block_weights_from_block(handle.block, bp + String("."), D, F, Dh, ctx)
        var img_mod = _modvecs_from_block(handle.block, bp, True, silu_temb_h, D, ctx)
        var txt_mod = _modvecs_from_block(handle.block, bp, False, silu_temb_h, D, ctx)
        var bl = double_lora_for(lora, di)

        var bg = double_block_lora_backward[H, Dh, N_IMG, N_TXT, S](
            d_img_out.copy(), d_txt_out.copy(), w, img_mod, txt_mod,
            bl, saved.dbl_saved[di], cos_t, sin_t, D, F, eps, ctx,
        )
        d_img_out = bg.base.img.d_x.copy()
        d_txt_out = bg.base.txt.d_x.copy()

        # scatter LoRA grads into flat arrays (slot layout: img 0..5, txt 6..11)
        var base_slot = di * DBL_SLOTS
        # img slots 0..5: q,k,v,out,ff_up,ff_down
        d_a_flat[base_slot + 0] = bg.img.q_d_a.copy()
        d_b_flat[base_slot + 0] = bg.img.q_d_b.copy()
        d_a_flat[base_slot + 1] = bg.img.k_d_a.copy()
        d_b_flat[base_slot + 1] = bg.img.k_d_b.copy()
        d_a_flat[base_slot + 2] = bg.img.v_d_a.copy()
        d_b_flat[base_slot + 2] = bg.img.v_d_b.copy()
        d_a_flat[base_slot + 3] = bg.img.out_d_a.copy()
        d_b_flat[base_slot + 3] = bg.img.out_d_b.copy()
        d_a_flat[base_slot + 4] = bg.img.ff_up_d_a.copy()
        d_b_flat[base_slot + 4] = bg.img.ff_up_d_b.copy()
        d_a_flat[base_slot + 5] = bg.img.ff_down_d_a.copy()
        d_b_flat[base_slot + 5] = bg.img.ff_down_d_b.copy()
        # txt slots 6..11: q,k,v,out,ff_up,ff_down
        d_a_flat[base_slot + 6] = bg.txt.q_d_a.copy()
        d_b_flat[base_slot + 6] = bg.txt.q_d_b.copy()
        d_a_flat[base_slot + 7] = bg.txt.k_d_a.copy()
        d_b_flat[base_slot + 7] = bg.txt.k_d_b.copy()
        d_a_flat[base_slot + 8] = bg.txt.v_d_a.copy()
        d_b_flat[base_slot + 8] = bg.txt.v_d_b.copy()
        d_a_flat[base_slot + 9] = bg.txt.out_d_a.copy()
        d_b_flat[base_slot + 9] = bg.txt.out_d_b.copy()
        d_a_flat[base_slot + 10] = bg.txt.ff_up_d_a.copy()
        d_b_flat[base_slot + 10] = bg.txt.ff_up_d_b.copy()
        d_a_flat[base_slot + 11] = bg.txt.ff_down_d_a.copy()
        d_b_flat[base_slot + 11] = bg.txt.ff_down_d_b.copy()

        nonfinite += (
            _nonfinite_check(bg.img.q_d_a) + _nonfinite_check(bg.img.q_d_b) +
            _nonfinite_check(bg.txt.q_d_a) + _nonfinite_check(bg.txt.q_d_b)
        )

        # mod-vec grads (bg.base) discarded — frozen mod-MLP not a LoRA target.
        loader.mark_active_block_done(ctx)
        di -= 1

    # input-projection backward (frozen; grads exercised but discarded)
    var lbi = linear_backward(
        Tensor.from_host(d_img_out, [N_IMG, D], STDtype.F32, ctx),
        Tensor.from_host(img_tokens, [N_IMG, in_ch], STDtype.F32, ctx),
        base.stack.img_in_w[], N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)

    var lbt = linear_backward(
        Tensor.from_host(d_txt_out, [N_TXT, D], STDtype.F32, ctx),
        Tensor.from_host(txt_tokens, [N_TXT, txt_ch], STDtype.F32, ctx),
        base.stack.txt_in_w[], N_TXT, txt_ch, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)

    return QwenLoraGradSet(d_a_flat^, d_b_flat^, d_img_tokens^, d_txt_tokens^, nonfinite)


# ── AdamW step for the offload grad set ──────────────────────────────────────
def qwen_offload_lora_adamw_step(
    mut set: QwenLoraSet,
    grads: QwenLoraGradSet,
    t: Int, lr: Float32, ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    for bi in range(set.num_double):
        var base_slot = bi * DBL_SLOTS
        for s in range(DBL_SLOTS):
            var idx = base_slot + s
            if len(grads.d_a[idx]) > 0:
                _lora_adamw(
                    set.dbl[idx],
                    LoraGrads(grads.d_a[idx].copy(), grads.d_b[idx].copy()),
                    t, lr, ctx, beta1, beta2, eps, weight_decay,
                )
