# serenitymojo/models/scail2/scail2_stack_lora.mojo
#
# SCAIL-2 i2v FULL-DEPTH (40-block) LoRA BLOCK-STACK — composition layer (CHUNK 2a).
# forward (save-acts + per-block input tape) + hand-chained backward (per-block
# RECOMPUTE, with an optional save-all path) + the LoRA set / AdamW / save.
#
# THIN FORK of the certified Wan2.1 i2v offload stack
#   serenitymojo/models/wan22/wan22_stack_lora.mojo::
#     wan22_i2v_stack_lora_{forward,backward}_offload  (+ its 12-slot LoRA set /
#     adamw_step / save), with the per-block call swapped to the SCAIL-2 F32-
#     residual block  models/scail2/scail2_block_train.mojo::
#     scail2_block_lora_{forward,backward}.
#
# SCOPE (CHUNK 2a). This is the BLOCK STACK ONLY. It takes an ALREADY-EMBEDDED
# F32 sequence + the (frozen, shared) text/image conditioning tensors + per-block
# modulation vectors + per-block frozen weights, and returns the block-stack
# output PRE-HEAD. The frozen embed (patchify/mask/concat), the time/text/image
# conditioning chain, the RoPE-table build, the head fwd/bwd, and FP8-stream base
# loading are CHUNK 2b — NOT wired here.
#
# INTER-BLOCK CONTRACT (verified against scail2_block_train):
#   - The block carries its residual stream in F32 (SCAIL-2 pinned residual). The
#     block forward returns x_out as List[Float32] and consumes x_h as
#     List[Float32]; hand-chaining just threads x_out(bi) -> x_h(bi+1) verbatim
#     (F32 host list). The block internally recasts x_h F32 -> bf16 for norm1 and
#     re-reads the F32 residual base — that round-trip is INSIDE the block, so the
#     stack sees a clean F32 seam.
#   - d_x(block N) == d_y(block N-1): the block backward returns d_x w.r.t. its
#     F32 input; the reverse loop feeds it as the previous block's d_out.
#   - Context (text + image) is SHARED + FROZEN across all blocks: assembled once
#     upstream, copied into every block, d_context discarded (umt5/CLIP/MLPProj
#     frozen).
#
# LoRA carrier: flat List[LoraAdapter], 12 slots/block (identical to the wan22 i2v
# set). Slot order: sa_{q,k,v,o}, ca_{q,k,v,o}, ffn.0, ffn.2, cross_attn.k_img,
# cross_attn.v_img. 40 blocks -> 480 adapters.
#
# Mojo 0.26.x+: def not fn; Tensor move-only; TArc carriers; structs .copy()/^.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.models.wan22.wan22_block import (
    WanModVecs, WanBlockLora, WanI2VBlockWeights, WanI2VBlockLora,
    WanI2VBlockLoraGrads,
)
from serenitymojo.models.scail2.scail2_block_train import (
    Scail2Saved, Scail2BlockForward,
    scail2_block_lora_forward, scail2_block_lora_backward,
)
from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_adamw_plain_fused import fused_lora_adamw_plain_step
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_peft, save_lora_train_state,
)


comptime TArc = ArcPointer[Tensor]

# ── LoRA carrier slots per SCAIL-2 i2v block (12 = base 10 + img_k + img_v). ──
comptime SC_SLOTS = 12
comptime SC_SA_Q = 0
comptime SC_SA_K = 1
comptime SC_SA_V = 2
comptime SC_SA_O = 3
comptime SC_CA_Q = 4
comptime SC_CA_K = 5
comptime SC_CA_V = 6
comptime SC_CA_O = 7
comptime SC_FFN0 = 8    # ffn.0: in=dim, out=ffn
comptime SC_FFN2 = 9    # ffn.2: in=ffn, out=dim
comptime SC_IMG_K = 10  # cross_attn.k_img: in=dim, out=dim
comptime SC_IMG_V = 11  # cross_attn.v_img: in=dim, out=dim

# GPU fused plain-AdamW (mirror of the wan22 WAN22_GPU_ADAMW default).
comptime SCAIL2_GPU_ADAMW = True


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


# ── LoRA carrier (flat, 12 slots/block) — fork of Wan22I2VLoraSet ─────────────
struct Scail2LoraSet(Movable):
    var ad: List[LoraAdapter]  # flat: block bi -> ad[bi*12 + slot]
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


def build_scail2_lora_set(
    num_blocks: Int, dim: Int, ffn: Int, rank: Int, alpha: Float32,
) -> Scail2LoraSet:
    """Build a full Scail2LoraSet (A=randn, B=0 -> identity at init).
    12 adapters per block, deterministic order: sa_{q,k,v,o} + ca_{q,k,v,o}
    (all in=out=dim), ffn.0 (dim->ffn), ffn.2 (ffn->dim), then cross_attn.k_img
    + cross_attn.v_img (both in=out=dim)."""
    var ad = List[LoraAdapter]()
    var seed = UInt64(52022)
    for _ in range(num_blocks):
        for _ in range(8):
            ad.append(_make_adapter(rank, alpha, dim, dim, seed))
            seed += 1
        ad.append(_make_adapter(rank, alpha, dim, ffn, seed))
        seed += 1
        ad.append(_make_adapter(rank, alpha, ffn, dim, seed))
        seed += 1
        ad.append(_make_adapter(rank, alpha, dim, dim, seed))   # k_img
        seed += 1
        ad.append(_make_adapter(rank, alpha, dim, dim, seed))   # v_img
        seed += 1
    return Scail2LoraSet(ad^, num_blocks, rank)


def scail2_total_adapters(lora: Scail2LoraSet) -> Int:
    return lora.num_blocks * SC_SLOTS


def _sc_block_base(bi: Int) -> Int:
    return bi * SC_SLOTS


def _scail2_block_lora_for(lora: Scail2LoraSet, bi: Int) -> WanI2VBlockLora:
    var base = _sc_block_base(bi)
    var base_lora = WanBlockLora(
        Optional[LoraAdapter](lora.ad[base + SC_SA_Q].copy()),
        Optional[LoraAdapter](lora.ad[base + SC_SA_K].copy()),
        Optional[LoraAdapter](lora.ad[base + SC_SA_V].copy()),
        Optional[LoraAdapter](lora.ad[base + SC_SA_O].copy()),
        Optional[LoraAdapter](lora.ad[base + SC_CA_Q].copy()),
        Optional[LoraAdapter](lora.ad[base + SC_CA_K].copy()),
        Optional[LoraAdapter](lora.ad[base + SC_CA_V].copy()),
        Optional[LoraAdapter](lora.ad[base + SC_CA_O].copy()),
        Optional[LoraAdapter](lora.ad[base + SC_FFN0].copy()),
        Optional[LoraAdapter](lora.ad[base + SC_FFN2].copy()),
    )
    return WanI2VBlockLora(
        base_lora^,
        Optional[LoraAdapter](lora.ad[base + SC_IMG_K].copy()),
        Optional[LoraAdapter](lora.ad[base + SC_IMG_V].copy()),
    )


# ── LoRA grad carrier — fork of Wan22I2VLoraGradSet ───────────────────────────
struct Scail2LoraGradSet(Movable):
    var d_a: List[List[Float32]]   # [num_blocks*12][rank*in]
    var d_b: List[List[Float32]]   # [num_blocks*12][out*rank]
    var nonfinite_lora_grads: Int

    def __init__(
        out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.nonfinite_lora_grads = nonfinite_lora_grads


# ── Forward tape ──────────────────────────────────────────────────────────────
# block_inputs[bi] is the F32 input sequence to block bi (block 0's = the embedded
# sequence; block i's = block i-1's F32 output). block_saved[bi] is the block's
# saved activations (the save-all backward path). x_out is the final F32 block-
# stack output (pre-head).
struct Scail2StackForward(Movable):
    var x_out: List[Float32]                 # [S, dim] F32
    var block_inputs: List[List[Float32]]    # per-block F32 input sequence
    var block_saved: List[Scail2Saved]       # per-block saved activations

    def __init__(
        out self, var x_out: List[Float32],
        var block_inputs: List[List[Float32]], var block_saved: List[Scail2Saved],
    ):
        self.x_out = x_out^
        self.block_inputs = block_inputs^
        self.block_saved = block_saved^


# ── Backward result (d_x w.r.t. the input sequence + the LoRA grad set) ───────
struct Scail2StackBackward(Movable):
    var d_x: List[Float32]         # [S, dim] F32
    var grads: Scail2LoraGradSet

    def __init__(out self, var d_x: List[Float32], var grads: Scail2LoraGradSet):
        self.d_x = d_x^
        self.grads = grads^


# ═══════════════════════════════════════════════════════════════════════════════
# FULL-DEPTH FORWARD WITH LoRA (hand-chained; save-acts + per-block input tape).
#   sequence:     [S, dim]  ALREADY-EMBEDDED F32 sequence (block-0 input)
#   modvecs:      List[WanModVecs] (len = num_layers) per-block AdaLN vectors [S,dim]
#   context_txt:  [TXT, dim] shared frozen text context
#   context_img:  [IMG, dim] shared frozen image context (CLIP MLPProj)
#   cos/sin:      RoPE tables [S, DH/2] (F32 tensors)
#   base_blocks:  List[WanI2VBlockWeights] (len = num_layers) per-block frozen weights
# ═══════════════════════════════════════════════════════════════════════════════
def scail2_stack_lora_forward[
    H: Int, DH: Int, S: Int, TXT: Int, IMG: Int
](
    var sequence: List[Float32],
    modvecs: List[WanModVecs],
    context_txt: List[Float32], context_img: List[Float32],
    cos: Tensor, sin: Tensor,
    base_blocks: List[WanI2VBlockWeights], lora_set: Scail2LoraSet,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
) raises -> Scail2StackForward:
    var num_layers = lora_set.num_blocks

    var x = sequence^
    var block_inputs = List[List[Float32]]()
    # NOTE (device tape): Scail2Saved is now DEVICE-resident (ArcPointer[Tensor]),
    # so this save-all `block_saved` holds num_layers × one-block of ACTIVATION
    # VRAM simultaneously. That is fine for the small compositions this function
    # serves (the 3-block parity's recompute-vs-save-all gate). The real 40-block
    # trainer does NOT use this path — it uses the STREAMING driver in
    # scail2_stack_train_full.mojo, whose backward RECOMPUTES each block from a
    # host F32 input tape so only ONE block's device tape is live at a time
    # (no VRAM blowup). Do not run this non-streamed save-all path at full depth.
    var block_saved = List[Scail2Saved]()

    for bi in range(num_layers):
        block_inputs.append(x.copy())
        var bl = _scail2_block_lora_for(lora_set, bi)
        var fwd = scail2_block_lora_forward[S, TXT, IMG, H, DH](
            x.copy(), context_txt.copy(), context_img.copy(),
            modvecs[bi].copy(), base_blocks[bi], bl, cos, sin, dim, ffn, eps, ctx,
        )
        block_saved.append(fwd.saved.copy())
        x = fwd.x_out.copy()

    return Scail2StackForward(x^, block_inputs^, block_saved^)


# ── scatter one block's 12 LoRA grads into the flat accumulators ──────────────
def _scatter_block_grads(
    mut d_a: List[List[Float32]], mut d_b: List[List[Float32]],
    bbase: Int, bg: WanI2VBlockLoraGrads,
) raises -> Int:
    d_a[bbase + SC_SA_Q] = bg.base.sa_q_da.copy()
    d_b[bbase + SC_SA_Q] = bg.base.sa_q_db.copy()
    d_a[bbase + SC_SA_K] = bg.base.sa_k_da.copy()
    d_b[bbase + SC_SA_K] = bg.base.sa_k_db.copy()
    d_a[bbase + SC_SA_V] = bg.base.sa_v_da.copy()
    d_b[bbase + SC_SA_V] = bg.base.sa_v_db.copy()
    d_a[bbase + SC_SA_O] = bg.base.sa_o_da.copy()
    d_b[bbase + SC_SA_O] = bg.base.sa_o_db.copy()
    d_a[bbase + SC_CA_Q] = bg.base.ca_q_da.copy()
    d_b[bbase + SC_CA_Q] = bg.base.ca_q_db.copy()
    d_a[bbase + SC_CA_K] = bg.base.ca_k_da.copy()
    d_b[bbase + SC_CA_K] = bg.base.ca_k_db.copy()
    d_a[bbase + SC_CA_V] = bg.base.ca_v_da.copy()
    d_b[bbase + SC_CA_V] = bg.base.ca_v_db.copy()
    d_a[bbase + SC_CA_O] = bg.base.ca_o_da.copy()
    d_b[bbase + SC_CA_O] = bg.base.ca_o_db.copy()
    d_a[bbase + SC_FFN0] = bg.base.ffn0_da.copy()
    d_b[bbase + SC_FFN0] = bg.base.ffn0_db.copy()
    d_a[bbase + SC_FFN2] = bg.base.ffn2_da.copy()
    d_b[bbase + SC_FFN2] = bg.base.ffn2_db.copy()
    d_a[bbase + SC_IMG_K] = bg.img_k_da.copy()
    d_b[bbase + SC_IMG_K] = bg.img_k_db.copy()
    d_a[bbase + SC_IMG_V] = bg.img_v_da.copy()
    d_b[bbase + SC_IMG_V] = bg.img_v_db.copy()

    var nf = 0
    nf += _nonfinite(bg.base.sa_q_da) + _nonfinite(bg.base.sa_q_db)
    nf += _nonfinite(bg.base.sa_k_da) + _nonfinite(bg.base.sa_k_db)
    nf += _nonfinite(bg.base.sa_v_da) + _nonfinite(bg.base.sa_v_db)
    nf += _nonfinite(bg.base.sa_o_da) + _nonfinite(bg.base.sa_o_db)
    nf += _nonfinite(bg.base.ca_q_da) + _nonfinite(bg.base.ca_q_db)
    nf += _nonfinite(bg.base.ca_k_da) + _nonfinite(bg.base.ca_k_db)
    nf += _nonfinite(bg.base.ca_v_da) + _nonfinite(bg.base.ca_v_db)
    nf += _nonfinite(bg.base.ca_o_da) + _nonfinite(bg.base.ca_o_db)
    nf += _nonfinite(bg.base.ffn0_da) + _nonfinite(bg.base.ffn0_db)
    nf += _nonfinite(bg.base.ffn2_da) + _nonfinite(bg.base.ffn2_db)
    nf += _nonfinite(bg.img_k_da) + _nonfinite(bg.img_k_db)
    nf += _nonfinite(bg.img_v_da) + _nonfinite(bg.img_v_db)
    return nf


# ═══════════════════════════════════════════════════════════════════════════════
# FULL-DEPTH BACKWARD WITH LoRA (REVERSE hand-chain).
# `recompute` = True (default): per block, RECOMPUTE its forward from the saved
# F32 input to regenerate the block's activations (bounds live memory to one
# block, matching the offload/checkpoint contract), then run its backward.
# `recompute` = False: use the save-all tape (fwd_state.block_saved[bi]) directly.
# Only the 12 LoRA d_A/d_B per block are collected; all base/img frozen-weight
# grads + d_context are discarded (conditioning frozen).
# ═══════════════════════════════════════════════════════════════════════════════
def scail2_stack_lora_backward[
    H: Int, DH: Int, S: Int, TXT: Int, IMG: Int
](
    var d_out: List[Float32],
    modvecs: List[WanModVecs],
    context_txt: List[Float32], context_img: List[Float32],
    cos: Tensor, sin: Tensor,
    base_blocks: List[WanI2VBlockWeights], lora_set: Scail2LoraSet,
    fwd_state: Scail2StackForward,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
    recompute: Bool = True,
) raises -> Scail2StackBackward:
    var num_layers = lora_set.num_blocks
    var n_adapters = scail2_total_adapters(lora_set)

    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a.append(List[Float32]())
        d_b.append(List[Float32]())
    var nonfinite = 0

    var d_cur = d_out^   # F32 grad w.r.t. the last block's output

    var bi = num_layers - 1
    while bi >= 0:
        var bl = _scail2_block_lora_for(lora_set, bi)

        if recompute:
            # RECOMPUTE this block's forward from its saved F32 input.
            var rc = scail2_block_lora_forward[S, TXT, IMG, H, DH](
                fwd_state.block_inputs[bi].copy(), context_txt.copy(), context_img.copy(),
                modvecs[bi].copy(), base_blocks[bi], bl, cos, sin, dim, ffn, eps, ctx,
            )
            var bg = scail2_block_lora_backward[S, TXT, IMG, H, DH](
                d_cur.copy(), modvecs[bi].copy(), base_blocks[bi], bl, rc.saved,
                cos, sin, dim, ffn, eps, ctx,
            )
            d_cur = bg.base.base.d_x.copy()
            nonfinite += _scatter_block_grads(d_a, d_b, _sc_block_base(bi), bg)
        else:
            var bg = scail2_block_lora_backward[S, TXT, IMG, H, DH](
                d_cur.copy(), modvecs[bi].copy(), base_blocks[bi], bl,
                fwd_state.block_saved[bi], cos, sin, dim, ffn, eps, ctx,
            )
            d_cur = bg.base.base.d_x.copy()
            nonfinite += _scatter_block_grads(d_a, d_b, _sc_block_base(bi), bg)

        bi -= 1

    var grads = Scail2LoraGradSet(d_a^, d_b^, nonfinite)
    return Scail2StackBackward(d_cur^, grads^)


# ── AdamW step over all 12/block adapters (fused GPU plain-AdamW; same kernel as
#    the wan22 i2v path). The backward fills ALL 12 slots per block, so this is
#    normally ONE fused call over [0, n). ─────────────────────────────────────
def scail2_lora_adamw_step(
    mut lora: Scail2LoraSet, grads: Scail2LoraGradSet, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps_opt: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    comptime if SCAIL2_GPU_ADAMW:
        var n = scail2_total_adapters(lora)
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
        var n = scail2_total_adapters(lora)
        for i in range(n):
            if len(grads.d_a[i]) == 0:
                continue
            var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
            _lora_adamw(lora.ad[i], lg, t, lr, ctx, beta1, beta2, eps_opt, weight_decay)


# ── PEFT-keyed save (12 targets/block; diffusion_model.-prefixed) ─────────────
# ai-toolkit / ComfyUI Wan LoRA convention: `diffusion_model.` prefix + lora_A /
# lora_B keys. Maps each of the 12 targets to its checkpoint key:
#   blocks.{i}.self_attn.{q,k,v,o}, blocks.{i}.cross_attn.{q,k,v,o,k_img,v_img},
#   blocks.{i}.ffn.{0,2}.
def _scail2_lora_prefixes(num_blocks: Int) -> List[String]:
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


def save_scail2_lora(lora: Scail2LoraSet, path: String, ctx: DeviceContext) raises -> Int:
    """Inference-only PEFT save (A/B, diffusion_model.-prefixed lora_A/lora_B)."""
    var prefixes = _scail2_lora_prefixes(lora.num_blocks)
    var named = List[NamedLora]()
    var n = scail2_total_adapters(lora)
    for i in range(n):
        named.append(NamedLora(prefixes[i], lora.ad[i].copy()))
    return save_lora_peft(named, path, ctx)


def save_scail2_lora_state(
    lora: Scail2LoraSet, path: String, ctx: DeviceContext,
    var meta: List[Float32] = List[Float32](),
) raises -> Int:
    """Trainer RESUME state (A/B + AdamW moments) for the 12/block SCAIL-2 LoRA."""
    var prefixes = _scail2_lora_prefixes(lora.num_blocks)
    var named = List[NamedLora]()
    var n = scail2_total_adapters(lora)
    for i in range(n):
        named.append(NamedLora(prefixes[i], lora.ad[i].copy()))
    return save_lora_train_state(named, path, ctx, meta^)
