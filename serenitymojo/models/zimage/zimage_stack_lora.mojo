# serenitymojo/models/zimage/zimage_stack_lora.mojo
#
# Z-IMAGE (NextDiT) FULL DiT STACK *WITH LoRA* on every trained projection:
# forward (saving ckpt-inputs) + reduced/full-depth backward (training) using the
# parity-verified per-block LoRA variants (models/zimage/lora_block.mojo), COLLECTS
# every adapter's d_A/d_B, and supports an AdamW step + a PEFT/ai-toolkit save
# across all 7 × (num_nr + num_cr + num_main) adapters. This file COMPOSES; it
# rebuilds NOTHING. Mirrors models/ernie/ernie_stack_lora.mojo (the PROVEN pattern).
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/zimage/block.mojo : base block fwd+bwd (modulated 19/19 + refiner 15/15).
#   * models/zimage/zimage_stack.mojo : the BASE full-stack composition (VERDICT
#     PASS, all token/weight/mod grads cos>=0.999). THIS FILE is that file with the
#     base per-block calls swapped for the LoRA variants + LoRA-grad collection.
#   * models/zimage/lora_block.mojo : zimage_block_lora_forward/backward (modulated)
#     and zimage_refiner_lora_forward/backward (unmodulated context refiner); each
#     reduces to base when adapters absent; LoRA d_x summed into the proj-input grad.
#   * training/{lora_save, train_step, optim} : LoraAdapter, _lora_adamw, save_lora_peft.
#
# TARGET SET (OneTrainer baseline filter) — read line-by-line from OT source:
#   ZImageLoRASetup.py:57 -> LoRAModuleWrapper(model.transformer, "transformer",
#   config, config.layer_filter.split(",")). With an empty filter,
#   LoRAModule.py:638-656 (__create_modules) adapts EVERY nn.Linear/Conv2d child of
#   the transformer. The diffusers ZImageTransformer2DModel
#   (transformer_z_image.py:184-224, 359+) gives each block:
#       attention (diffusers Attention -> to_q, to_k, to_v, to_out.0)  [4 Linear]
#       feed_forward (FeedForward -> w1, w3, w2)                        [3 Linear]
#   in noise_refiner.<i>, context_refiner.<i>, and layers.<i>.
#
#   The active OneTrainer Z-Image baseline does NOT use the empty filter. It uses:
#     ^(?=.*attention)(?!.*refiner).*,^(?=.*feed_forward)(?!.*refiner).*
#   so only main `layers.<i>.{attention,feed_forward}` are trainable; noise/context
#   refiners are excluded. This file can carry refiner adapters for parity smokes,
#   but production train uses the `*_main_only` optimizer/save helpers below.
#
# KEY NAMING (round-trip with OT / inference-flame) — slot order Q,K,V,O,w1,w3,w2:
#   OT saves transformer_lora.state_dict() (ZImageLoRASaver.py:24-25), whose keys
#   are the diffusers submodule paths under prefix "transformer.":
#       transformer.<stream>.<i>.attention.{to_q,to_k,to_v,to_out.0}
#       transformer.<stream>.<i>.feed_forward.{w1,w3,w2}
#   with kohya lora_down/lora_up (PeftBase LoRAModule.py:143-144). We REUSE
#   save_lora_peft, which emits PEFT "<prefix>.lora_A.weight"/".lora_B.weight" — the
#   convention train_klein and the inference lora.mojo loader use (lora_save.mojo
#   header) — using the diffusers module path as <prefix> (the Ernie precedent:
#   omit the OT wrapper "transformer." prefix; lora.mojo detects DiffusionModel by
#   the ".lora_A.weight" suffix). A = lora_down [rank,in], B = lora_up [out,rank].
#   The inference-flame zimage_nextdit.rs fuses qkv -> attention.qkv.weight; for the
#   pure-Mojo path the un-fused diffusers names are canonical (zimage/weights.mojo
#   loads to_q/to_k/to_v/to_out.0/feed_forward.{w1,w3,w2} directly).
#
# CARRIER DESIGN (Tenet-2) — mirrors ErnieLoraSet but THREE flat segments:
#   ZImageLoraSet holds ONE flat List[LoraAdapter] laid out as
#       [ nr blocks ][ cr blocks ][ main blocks ]   (each block = 7 slots)
#   indexed by a deterministic scheme: flat = (segment_base + block)*ZIMAGE_SLOTS +
#   slot. The optimizer walks the flat list; the backward SCATTERS each per-block
#   7-slot d_A/d_B into the matching flat entry.
#
# SCOPE: LoRA-on-projection training. Base weights are FROZEN — their grads come
#   from the base path and are discarded for the optimizer; only d_A/d_B are
#   trained. Per-block RAW mod-vec grads are returned (each block backprops them into
#   its OWN adaLN_modulation.0 at step end; Z-Image mod is PER-BLOCK, NOT shared —
#   so NO summation across blocks, unlike Ernie).
#
# Mojo 1.0.0b1: def not fn; Tensor move-only; host List[Float32] carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward
from serenitymojo.ops.elementwise_backward import modulate_backward

from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import (
    ZImageModVecs, ZImageBlockGrads, ZImageRefinerGrads,
)
from serenitymojo.models.zimage.lora_block import (
    ZImageBlockLora, ZImageBlockLoraGrads, ZIMAGE_SLOTS,
    SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_W1, SLOT_W3, SLOT_W2,
    zimage_block_lora_forward, zimage_block_lora_backward,
    zimage_refiner_lora_forward, zimage_refiner_lora_backward,
)
from serenitymojo.models.zimage.zimage_stack import (
    ZImageStackForward, _zeros, _ones, _t, _add_lists,
    _concat_img_cap, _split_img_cap, _linear_wdev_bias, _saved_x_out,
)

from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_peft, load_lora_for_resume,
)


comptime TArc = ArcPointer[Tensor]


# ── adapter init (A small randn, B=0 — PEFT identity at step 0) ───────────────
# LCG randn byte-identical to ernie_stack_lora._randn / train_step._randn.
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
        _zeros(out_f * rank),              # B = 0 (adapter identity at init)
        rank, in_f, out_f, scale,
        _zeros(rank * in_f), _zeros(rank * in_f),    # ma / va
        _zeros(out_f * rank), _zeros(out_f * rank),  # mb / vb
    )


# ── the LoRA carrier: every trained adapter, flat-indexed across 3 segments ───
struct ZImageLoraSet(Copyable, Movable):
    var ad: List[LoraAdapter]   # (num_nr + num_cr + num_main) * ZIMAGE_SLOTS
    var num_nr: Int
    var num_cr: Int
    var num_main: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoraAdapter],
        num_nr: Int, num_cr: Int, num_main: Int, rank: Int,
    ):
        self.ad = ad^
        self.num_nr = num_nr
        self.num_cr = num_cr
        self.num_main = num_main
        self.rank = rank

    # segment base (in BLOCKS) for the three streams.
    def nr_base(self) -> Int:
        return 0

    def cr_base(self) -> Int:
        return self.num_nr

    def main_base(self) -> Int:
        return self.num_nr + self.num_cr

    def num_blocks(self) -> Int:
        return self.num_nr + self.num_cr + self.num_main


# ── per-block adapter slot shapes (in, out) for slot s, given D (hidden), F (ffn) ─
def _slot_in(s: Int, D: Int, F: Int) -> Int:
    if s == SLOT_W2:   # feed_forward.w2: in = F
        return F
    return D           # to_q/k/v/out: in=D ; w1/w3: in=D


def _slot_out(s: Int, D: Int, F: Int) -> Int:
    if s == SLOT_W1 or s == SLOT_W3:   # w1/w3: out = F
        return F
    return D           # to_q/k/v/out: out=D ; w2: out=D


# Append one block's 7 adapters to `ad` (advances `seed`).
def _append_block_adapters(
    mut ad: List[LoraAdapter], mut seed: UInt64, rank: Int, alpha: Float32, D: Int, F: Int
):
    for s in range(ZIMAGE_SLOTS):
        var in_f = _slot_in(s, D, F)
        var out_f = _slot_out(s, D, F)
        ad.append(make_lora_adapter(rank, alpha, in_f, out_f, seed))
        seed += 1


# ── build the full LoRA set for a Z-Image stack (nr | cr | main segments) ─────
def build_zimage_lora_set(
    num_nr: Int, num_cr: Int, num_main: Int, D: Int, F: Int, rank: Int, alpha: Float32
) -> ZImageLoraSet:
    var ad = List[LoraAdapter]()
    var seed = UInt64(3000)
    for _ in range(num_nr):
        _append_block_adapters(ad, seed, rank, alpha, D, F)
    for _ in range(num_cr):
        _append_block_adapters(ad, seed, rank, alpha, D, F)
    for _ in range(num_main):
        _append_block_adapters(ad, seed, rank, alpha, D, F)
    return ZImageLoraSet(ad^, num_nr, num_cr, num_main, rank)


# Build a transient ZImageBlockLora for block `block_idx` (in flat-block space).
def _block_lora_for(set: ZImageLoraSet, block_idx: Int) -> ZImageBlockLora:
    var base = block_idx * ZIMAGE_SLOTS
    return ZImageBlockLora(
        Optional[LoraAdapter](set.ad[base + SLOT_Q].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_K].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_V].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_O].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_W1].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_W3].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_W2].copy()),
    )


# ── collected LoRA grads (flat, parallel to ZImageLoraSet) ───────────────────
struct ZImageLoraGrads(Movable):
    var d_a: List[List[Float32]]   # num_blocks * ZIMAGE_SLOTS
    var d_b: List[List[Float32]]
    # load-bearing input-token grads (prove the full chain back to embedder outs).
    var d_x_seq: List[Float32]        # [N_IMG, D]
    var d_cap_seq: List[Float32]      # [N_TXT, D]
    # per-block RAW mod-vec grads (Z-Image mod is PER-BLOCK — not summed).
    var nr_mod: List[List[Float32]]   # num_nr   x [4D]
    var main_mod: List[List[Float32]] # num_main x [4D]
    var d_f_scale: List[Float32]      # [D]
    var d_final_lin: List[Float32]    # [out_ch, D] (base, discarded by AdamW)
    var nonfinite_lora_grads: Int

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_x_seq: List[Float32], var d_cap_seq: List[Float32],
        var nr_mod: List[List[Float32]], var main_mod: List[List[Float32]],
        var d_f_scale: List[Float32], var d_final_lin: List[Float32],
        nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x_seq = d_x_seq^
        self.d_cap_seq = d_cap_seq^
        self.nr_mod = nr_mod^
        self.main_mod = main_mod^
        self.d_f_scale = d_f_scale^
        self.d_final_lin = d_final_lin^
        self.nonfinite_lora_grads = nonfinite_lora_grads


def _modvec4(g: ZImageBlockGrads) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(g.d_scale_msa)):
        o.append(g.d_scale_msa[i])
    for i in range(len(g.d_gate_msa)):
        o.append(g.d_gate_msa[i])
    for i in range(len(g.d_scale_mlp)):
        o.append(g.d_scale_mlp[i])
    for i in range(len(g.d_gate_mlp)):
        o.append(g.d_gate_mlp[i])
    return o^


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# RESIDENT LoRA stack (small depth, for the COMPOSITION parity gate). Mirrors
# zimage_stack_forward/backward exactly, swapping per-block calls for LoRA ones.
# Blocks are passed resident (the gate uses NR=1/CR=1/MAIN=2 so they fit 24 GB).
# ─────────────────────────────────────────────────────────────────────────────
def zimage_stack_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    x_seq: List[Float32], cap_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    lora: ZImageLoraSet,
    f_scale: List[Float32],
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageStackForward:
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)

    # ── noise refiner (MODULATED + LoRA) on x_seq [N_IMG,D] ──
    var nr_x_in = List[TArc]()
    var xs = x_seq.copy()
    for i in range(num_nr):
        nr_x_in.append(TArc(_t(xs.copy(), [N_IMG, D], ctx)))
        var bl = _block_lora_for(lora, lora.nr_base() + i)
        var fwd = zimage_block_lora_forward[H, Dh, N_IMG](
            xs.copy(), nr_blocks[i], nr_mod[i], bl, x_cos, x_sin, D, F, eps, ctx,
        )
        xs = fwd.out.copy()

    # ── context refiner (UNMODULATED + LoRA) on cap_seq [N_TXT,D] ──
    var cr_x_in = List[TArc]()
    var cs = cap_seq.copy()
    for i in range(num_cr):
        cr_x_in.append(TArc(_t(cs.copy(), [N_TXT, D], ctx)))
        var bl = _block_lora_for(lora, lora.cr_base() + i)
        var fwd = zimage_refiner_lora_forward[H, Dh, N_TXT](
            cs.copy(), cr_blocks[i], bl, cap_cos, cap_sin, D, F, eps, ctx,
        )
        cs = fwd.out.copy()

    # ── unified = concat([x, cap]) -> [S,D] ──
    var x = _concat_img_cap(xs, cs)

    # ── main layers (MODULATED + LoRA) ──
    var main_x_in = List[TArc]()
    for i in range(num_main):
        main_x_in.append(TArc(_t(x.copy(), [S, D], ctx)))
        var bl = _block_lora_for(lora, lora.main_base() + i)
        var fwd = zimage_block_lora_forward[H, Dh, S](
            x.copy(), main_blocks[i], main_mod[i], bl, uni_cos, uni_sin, D, F, eps, ctx,
        )
        x = fwd.out.copy()

    # ── final layer ──
    var ln_x = layer_norm(
        _t(x.copy(), [S, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), final_eps, ctx,
    ).to_host(ctx)
    var x_out = modulate(
        _t(ln_x.copy(), [S, D], ctx),
        _t(f_scale.copy(), [D], ctx), _t(_zeros(D), [D], ctx), ctx,
    ).to_host(ctx)
    var patches = _linear_wdev_bias(x_out, final_lin_w, final_lin_b, S, D, ctx)
    var parts = _split_img_cap(patches, N_IMG, N_TXT, out_ch)
    var out = parts[0].copy()

    return ZImageStackForward(
        out^, x_seq.copy(), cap_seq.copy(),
        nr_x_in^, cr_x_in^, main_x_in^,
        TArc(_t(x^, [S, D], ctx)), TArc(_t(ln_x^, [S, D], ctx)),
    )


def zimage_stack_lora_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    lora: ZImageLoraSet,
    f_scale: List[Float32],
    final_lin_w: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    saved: ZImageStackForward,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageLoraGrads:
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)
    var num_blocks = num_nr + num_cr + num_main

    # ── final-layer backward (identical to base stack) ──
    var d_patches = _concat_img_cap(d_out, _zeros(N_TXT * out_ch))
    var x_out = _saved_x_out(saved, f_scale, D, S, ctx)
    var final_dx = linear_backward_dx(
        _t(d_patches, [S, out_ch], ctx), final_lin_w, S, D, out_ch, ctx,
    )
    var d_x_out = final_dx.to_host(ctx)
    var d_final_lin = List[Float32]()
    var mbf = modulate_backward(
        _t(d_x_out, [S, D], ctx), saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_x = mbf.d_x.to_host(ctx)
    var d_f_scale = mbf.d_scale.to_host(ctx)
    var lnbf = layer_norm_backward(
        _t(d_ln_x, [S, D], ctx), saved.x_final[], _t(_ones(D), [D], ctx), final_eps, ctx,
    )
    var d_x = lnbf.d_x.to_host(ctx)

    # flat LoRA grad slots.
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(num_blocks * ZIMAGE_SLOTS):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    var nonfinite = 0

    # ── main layers backward (REVERSE; per-block recompute) ──
    var main_mod_rev = List[List[Float32]]()
    var bi = num_main - 1
    while bi >= 0:
        var bl = _block_lora_for(lora, lora.main_base() + bi)
        var refwd = zimage_block_lora_forward[H, Dh, S](
            saved.main_x_in[bi][].to_host(ctx), main_blocks[bi], main_mod[bi], bl,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        var bg = zimage_block_lora_backward[H, Dh, S](
            d_x.copy(), main_blocks[bi], main_mod[bi], bl, refwd.saved,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        d_x = bg.base.d_x.copy()
        main_mod_rev.append(_modvec4(bg.base))
        var base_idx = (lora.main_base() + bi) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            d_a_flat[base_idx + s] = bg.lora.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        bi -= 1
    var main_mod_grads = List[List[Float32]]()
    var jm = len(main_mod_rev) - 1
    while jm >= 0:
        main_mod_grads.append(main_mod_rev[jm].copy())
        jm -= 1

    # ── unified seam: split d_x [S,D] -> d_xs (first), d_cs (rest) ──
    var seam = _split_img_cap(d_x, N_IMG, N_TXT, D)
    var d_xs = seam[0].copy()
    var d_cs = seam[1].copy()

    # ── context refiner backward (UNMODULATED + LoRA; REVERSE) ──
    var ci = num_cr - 1
    while ci >= 0:
        var bl = _block_lora_for(lora, lora.cr_base() + ci)
        var refwd = zimage_refiner_lora_forward[H, Dh, N_TXT](
            saved.cr_x_in[ci][].to_host(ctx), cr_blocks[ci], bl, cap_cos, cap_sin, D, F, eps, ctx,
        )
        var bg = zimage_refiner_lora_backward[H, Dh, N_TXT](
            d_cs.copy(), cr_blocks[ci], bl, refwd.saved, cap_cos, cap_sin, D, F, eps, ctx,
        )
        d_cs = bg.base.d_x.copy()
        var base_idx = (lora.cr_base() + ci) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            d_a_flat[base_idx + s] = bg.lora.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        ci -= 1

    # ── noise refiner backward (MODULATED + LoRA; REVERSE) ──
    var nr_mod_rev = List[List[Float32]]()
    var ni = num_nr - 1
    while ni >= 0:
        var bl = _block_lora_for(lora, lora.nr_base() + ni)
        var refwd = zimage_block_lora_forward[H, Dh, N_IMG](
            saved.nr_x_in[ni][].to_host(ctx), nr_blocks[ni], nr_mod[ni], bl, x_cos, x_sin, D, F, eps, ctx,
        )
        var bg = zimage_block_lora_backward[H, Dh, N_IMG](
            d_xs.copy(), nr_blocks[ni], nr_mod[ni], bl, refwd.saved, x_cos, x_sin, D, F, eps, ctx,
        )
        d_xs = bg.base.d_x.copy()
        nr_mod_rev.append(_modvec4(bg.base))
        var base_idx = (lora.nr_base() + ni) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            d_a_flat[base_idx + s] = bg.lora.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        ni -= 1
    var nr_mod_grads = List[List[Float32]]()
    var jn = len(nr_mod_rev) - 1
    while jn >= 0:
        nr_mod_grads.append(nr_mod_rev[jn].copy())
        jn -= 1

    return ZImageLoraGrads(
        d_a_flat^, d_b_flat^,
        d_xs^, d_cs^,
        nr_mod_grads^, main_mod_grads^,
        d_f_scale^, d_final_lin^,
        nonfinite,
    )


# ── AdamW step on EVERY adapter (reuses the proven per-adapter _lora_adamw) ───
def zimage_lora_adamw_step(
    mut set: ZImageLoraSet, grads: ZImageLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var n = set.num_blocks() * ZIMAGE_SLOTS
    for i in range(n):
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(set.ad[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay)


# ── AdamW step on OneTrainer baseline trainable adapters only: main layers. ──
def zimage_lora_adamw_step_main_only(
    mut set: ZImageLoraSet, grads: ZImageLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var start = set.main_base() * ZIMAGE_SLOTS
    var end = set.num_blocks() * ZIMAGE_SLOTS
    for i in range(start, end):
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(set.ad[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay)


# ── per-block PEFT/kohya prefix scheme (the INVERSE of the inference target map) ─
# slot -> diffusers module suffix (transformer_z_image.py + zimage/weights.mojo).
def _slot_suffix(slot: Int) -> String:
    if slot == SLOT_Q:
        return String(".attention.to_q")
    elif slot == SLOT_K:
        return String(".attention.to_k")
    elif slot == SLOT_V:
        return String(".attention.to_v")
    elif slot == SLOT_O:
        return String(".attention.to_out.0")
    elif slot == SLOT_W1:
        return String(".feed_forward.w1")
    elif slot == SLOT_W3:
        return String(".feed_forward.w3")
    return String(".feed_forward.w2")


# stream prefix for a flat block index (nr | cr | main). Matches inference-flame
# zimage_nextdit.rs: noise_refiner.{i} / context_refiner.{i} / layers.{i}.
def _stream_prefix(set: ZImageLoraSet, block_idx: Int) -> String:
    if block_idx < set.cr_base():
        return String("noise_refiner.") + String(block_idx - set.nr_base())
    elif block_idx < set.main_base():
        return String("context_refiner.") + String(block_idx - set.cr_base())
    return String("layers.") + String(block_idx - set.main_base())


def _zimage_lora_prefix(set: ZImageLoraSet, block_idx: Int, slot: Int) -> String:
    return _stream_prefix(set, block_idx) + _slot_suffix(slot)


def zimage_lora_prefixes(set: ZImageLoraSet) -> List[String]:
    var out = List[String]()
    for bi in range(set.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            out.append(_zimage_lora_prefix(set, bi, s))
    return out^


# ── SAVE every adapter as a PEFT/ai-toolkit safetensors ──────────────────────
def save_zimage_lora(set: ZImageLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            named.append(NamedLora(
                _zimage_lora_prefix(set, bi, s),
                set.ad[bi * ZIMAGE_SLOTS + s].copy(),
            ))
    return save_lora_peft(named, path, ctx)


def save_zimage_lora_main_only(set: ZImageLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.main_base(), set.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            named.append(NamedLora(
                _zimage_lora_prefix(set, bi, s),
                set.ad[bi * ZIMAGE_SLOTS + s].copy(),
            ))
    return save_lora_peft(named, path, ctx)


# ── RESUME: load the adapter A/B back from a save_zimage_lora file ───────────
# AdamW moments are ZEROED (resume them from a loop TrainState checkpoint). The
# returned set carries the SAME flat order build_zimage_lora_set produces.
def load_zimage_lora_resume(
    num_nr: Int, num_cr: Int, num_main: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> ZImageLoraSet:
    # build a transient set to derive the flat prefix order, then overwrite A/B.
    var template = build_zimage_lora_set(num_nr, num_cr, num_main, 1, 1, rank, alpha)
    var prefixes = zimage_lora_prefixes(template)
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(prefixes, scale, path, ctx)   # flat order
    var ad = List[LoraAdapter]()
    for i in range(template.num_blocks() * ZIMAGE_SLOTS):
        ad.append(named[i].adapter.copy())
    return ZImageLoraSet(ad^, num_nr, num_cr, num_main, rank)
