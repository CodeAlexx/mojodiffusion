# serenitymojo/models/mageflow/mageflow_stack_lora.mojo
#
# Mage-Flow-Base 12-block LoRA stack: full-depth fwd+bwd with block-swap
# OFFLOAD streaming (TurboPlannedLoader), AdamW step, and ai-toolkit-keyed
# save/resume. 12 LoRA targets/block x 12 blocks = 144 adapters.
#
# The block math is BANKED and NOT reimplemented: qwenimage_block.mojo::
# double_block_lora_forward/backward — GATED vs the REAL mage_flow block
# (incl. the text-identity rope) by parity/mageflow_block_lora_parity.mojo
# (PASS, cos>=0.999). The ONE MageFlow delta — text tokens NOT roped — enters
# as data through build_mageflow_rope_tables (models/dit/mageflow_dit.mojo:
# 79-152), passed straight through to the block exactly as the parity gate
# does. Two stack-level deltas vs the qwen offload stack:
#   1. txt path = RMSNorm(txt_norm.weight) -> txt_in Linear(2560->3072)
#      (mageflow_dit.mojo::_txt_in; the qwen stack takes pre-normed text).
#   2. temb = the mage bf16-rounded sinusoid pipeline (weights.mojo::
#      compute_mageflow_silu_temb) — the caller computes silu(temb) once per
#      step and this stack feeds it to the SAME frozen mod readers the qwen
#      stack uses (`_modvecs_from_block`, `_compute_final_modvecs`).
#
# MEMORY DECISION (16GB card): OFFLOAD, not resident-12. 12 blocks x
# 679,662,592 B ≈ 8.2 GB bf16 resident would leave <8 GB for saved activations
# + workspace; the qwenimage engine's offload path is proven, keeps ≤2 blocks
# (~1.4 GB) in flight, and needs NO new resident-stack surface. Under offload
# the backward re-streams weights and uses the per-block DoubleBlockSaved
# activations saved by the forward — no per-block recompute needed
# (qwenimage_stack_lora.mojo:663-667 contract). The inter-block handoff is the
# AUTOGRAD_INTERNALS §8.1 d_x(block N) = d_y(block N+1) chain, walked in
# reverse.
#
# REUSED (imported, not forked): QwenLoraSet/build_qwen_lora_set/double_lora_for
# (kaiming-A/zero-B init via make_lora_adapter, qwenimage_stack_lora.mojo:
# 106-124), the offload block/mod readers, _lora_adamw, lora_save.
#
# SAVE KEYS (ai-toolkit / PEFT):
#   diffusion_model.transformer_blocks.{i}.<suffix>.lora_A.weight / .lora_B.weight
# suffixes: attn.to_q/to_k/to_v/to_out.0, img_mlp.net.0.proj, img_mlp.net.2,
# attn.add_q_proj/add_k_proj/add_v_proj/to_add_out, txt_mlp.net.0.proj,
# txt_mlp.net.2 (slot order == qwen _slot_suffix, reused).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.norm_backward import layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.linalg_backward import linear_backward

from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

from serenitymojo.training.train_step import _lora_adamw, LoraGrads
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_peft, load_lora_for_resume,
    save_lora_train_state, load_lora_train_state,
)

from serenitymojo.models.klein.lora_block import LoraAdapter
from serenitymojo.models.qwenimage.qwenimage_block import (
    DoubleBlockSaved,
    double_block_lora_forward, double_block_lora_backward,
)
from serenitymojo.models.qwenimage.qwenimage_stack import (
    _zeros as _qstack_zeros, _ones as _qstack_ones, _linear_b,
)
from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    DBL_SLOTS, QwenLoraSet, build_qwen_lora_set, double_lora_for,
    QwenOffloadForward,
    _double_block_weights_from_block, _modvecs_from_block,
    _compute_final_modvecs, _slot_suffix, _nonfinite_check,
)

from serenitymojo.models.mageflow.weights import MageFlowBase


comptime TArc = ArcPointer[Tensor]


# ── the LoRA set: the qwen 12-slot carrier at MageFlow depth/dims ─────────────
comptime MageFlowLoraSet = QwenLoraSet


def build_mageflow_lora_set(
    depth: Int, D: Int, F: Int, rank: Int, alpha: Float32
) -> MageFlowLoraSet:
    # A: kaiming_uniform(a=sqrt5), B: zeros — make_lora_adapter
    # (qwenimage_stack_lora.mojo:106-124) via build_qwen_lora_set.
    return build_qwen_lora_set(depth, D, F, rank, alpha)


def mageflow_total_adapters(set: MageFlowLoraSet) -> Int:
    return set.num_double * DBL_SLOTS


# ── flat grad set (LoRA-only scope: d_A/d_B per adapter + nonfinite count) ────
struct MageFlowLoraGradSet(Movable):
    var d_a: List[List[Float32]]    # depth * DBL_SLOTS
    var d_b: List[List[Float32]]
    var nonfinite_lora_grads: Int

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.nonfinite_lora_grads = nonfinite_lora_grads


# ═════════════════════════════════════════════════════════════════════════════
# FULL FORWARD, BLOCK-SWAP OFFLOAD (12 double blocks).
#   img_tokens [N_IMG, in_ch=128]  patchified latent tokens (host F32).
#   txt_raw    [N_TXT, txt_ch=2560] RAW Qwen3-VL context (un-normed; the
#              txt_norm RMSNorm is applied HERE, mageflow_dit.mojo::_txt_in).
#   silu_temb_h [1, D]: silu(time_text_embed(sigma)) via
#              weights.mojo::compute_mageflow_silu_temb (mage bf16 sinusoid).
#   pe_cos/pe_sin: build_mageflow_rope_tables output, [(N_TXT+N_IMG)*H, Dh//2]
#              — txt rows identity, img rows msrope (parity-gated).
# Mirrors qwenimage_stack_lora_forward_offload (qwenimage_stack_lora.mojo:
# 923-1001) with the two mageflow deltas from the header.
# ═════════════════════════════════════════════════════════════════════════════
def mageflow_stack_lora_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_raw: List[Float32],
    silu_temb_h: List[Float32],
    base: MageFlowBase,
    mut loader: TurboPlannedLoader, lora: MageFlowLoraSet,
    pe_cos: Tensor, pe_sin: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> QwenOffloadForward:
    comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
    var num_double = lora.num_double

    loader.prefetch_with_ctx(0, ctx)

    # input projections (frozen base):
    #   img_in Linear(128 -> D); txt: RMSNorm(txt_norm) -> txt_in Linear(2560 -> D)
    var img = _linear_b(
        img_tokens, base.stack.img_in_w[], base.stack.img_in_b[], N_IMG, in_ch, ctx
    )
    var txt_t = Tensor.from_host(txt_raw.copy(), [N_TXT, txt_ch], STDtype.BF16, ctx)
    var txt_normed = rms_norm(txt_t, base.txt_norm_w[], eps, ctx).to_host(ctx)
    var txt = _linear_b(
        txt_normed, base.stack.txt_in_w[], base.stack.txt_in_b[], N_TXT, txt_ch, ctx
    )

    var dbl_img_in = List[List[Float32]]()
    var dbl_txt_in = List[List[Float32]]()
    var dbl_saved = List[DoubleBlockSaved]()

    for bi in range(num_double):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var bp = handle.prefix
        var w = _double_block_weights_from_block(handle.block, bp, D, F, Dh, ctx)
        var img_mod = _modvecs_from_block(handle.block, bp, True, silu_temb_h, D, ctx)
        var txt_mod = _modvecs_from_block(handle.block, bp, False, silu_temb_h, D, ctx)
        var bl = double_lora_for(lora, bi)

        dbl_img_in.append(img.copy())
        dbl_txt_in.append(txt.copy())

        var fwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), w, img_mod, txt_mod, bl,
            pe_cos, pe_sin, D, F, eps, ctx,
        )
        dbl_saved.append(fwd.saved.copy())
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)

    # final layer: AdaLayerNormContinuous — layer_norm(no affine) ->
    # modulate(scale=chunk0, shift=chunk1 of norm_out.linear(silu_temb)) ->
    # proj_out (mageflow_dit.mojo:350-362 == the qwen final path).
    var final_mods = _compute_final_modvecs(
        silu_temb_h, base.norm_out_w, base.norm_out_b, D, ctx
    )
    var final_scale = final_mods[0].copy()
    var final_shift = final_mods[1].copy()

    var img_t = Tensor.from_host(img.copy(), [N_IMG, D], STDtype.BF16, ctx)
    var ln_img_out_t = layer_norm(
        img_t,
        Tensor.from_host(_qstack_ones(D), [D], STDtype.BF16, ctx),
        Tensor.from_host(_qstack_zeros(D), [D], STDtype.BF16, ctx),
        eps, ctx,
    )
    var ln_img_out_h = ln_img_out_t.to_host(ctx)
    var normed = modulate(
        Tensor.from_host(ln_img_out_h.copy(), [N_IMG, D], STDtype.BF16, ctx),
        Tensor.from_host(final_scale.copy(), [D], STDtype.BF16, ctx),
        Tensor.from_host(final_shift.copy(), [D], STDtype.BF16, ctx),
        ctx,
    ).to_host(ctx)
    var out = _linear_b(
        normed, base.stack.proj_out_w[], base.stack.proj_out_b[], N_IMG, D, ctx
    )

    var img_arc = ArcPointer[Tensor](Tensor.from_host(img^, [N_IMG, D], STDtype.BF16, ctx))
    var ln_arc = ArcPointer[Tensor](Tensor.from_host(ln_img_out_h^, [N_IMG, D], STDtype.BF16, ctx))

    return QwenOffloadForward(
        out^, dbl_img_in^, dbl_txt_in^, dbl_saved^,
        img_arc^, ln_arc^, final_scale^, final_shift^,
    )


# ═════════════════════════════════════════════════════════════════════════════
# FULL BACKWARD, BLOCK-SWAP OFFLOAD (REVERSE block stream).
#   LoRA-only scope: frozen mod-MLP grads discarded; the frozen img_in/txt_in
#   input-projection backward is SKIPPED entirely (nothing trainable upstream —
#   inputs are data, txt_norm/txt_in/img_in are frozen), so the chain stops at
#   block 0's d_x. 12 slots x 12 blocks = 144 (d_A, d_B) pairs collected.
# Mirrors qwenimage_stack_lora_backward_offload (qwenimage_stack_lora.mojo:
# 1009-1153) minus the discarded input-projection tail.
# ═════════════════════════════════════════════════════════════════════════════
def mageflow_stack_lora_backward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    silu_temb_h: List[Float32],
    base: MageFlowBase,
    mut loader: TurboPlannedLoader, lora: MageFlowLoraSet,
    pe_cos: Tensor, pe_sin: Tensor,
    saved: QwenOffloadForward,
    D: Int, F: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> MageFlowLoraGradSet:
    comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
    var num_double = lora.num_double
    var n_adapters = num_double * DBL_SLOTS

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

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
        Tensor.from_host(final_scale.copy(), [D], STDtype.BF16, ctx),
        Tensor.from_host(final_shift.copy(), [D], STDtype.BF16, ctx),
        ctx,
    ).to_host(ctx)

    var lbf = linear_backward(
        Tensor.from_host(d_out, [N_IMG, out_ch], STDtype.BF16, ctx),
        Tensor.from_host(normed, [N_IMG, D], STDtype.BF16, ctx),
        base.stack.proj_out_w[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)

    var mbf = modulate_backward(
        Tensor.from_host(d_normed, [N_IMG, D], STDtype.BF16, ctx),
        saved.ln_img_out[],
        Tensor.from_host(final_scale.copy(), [D], STDtype.BF16, ctx),
        ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)

    var lnbf_dx = layer_norm_backward_dx(
        Tensor.from_host(d_ln_img_out, [N_IMG, D], STDtype.BF16, ctx),
        saved.img_out[],
        Tensor.from_host(_qstack_ones(D), [D], STDtype.BF16, ctx),
        eps, ctx,
    )
    var d_img_out = lnbf_dx.to_host(ctx)
    var d_txt_out = _qstack_zeros(N_TXT * D)

    # ── double-stream backward (REVERSE; d_x(N) = d_y(N+1) handoff) ───────────
    var di = num_double - 1
    while di >= 0:
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var bp = handle.prefix
        var w = _double_block_weights_from_block(handle.block, bp, D, F, Dh, ctx)
        var img_mod = _modvecs_from_block(handle.block, bp, True, silu_temb_h, D, ctx)
        var txt_mod = _modvecs_from_block(handle.block, bp, False, silu_temb_h, D, ctx)
        var bl = double_lora_for(lora, di)

        var bg = double_block_lora_backward[H, Dh, N_IMG, N_TXT, S](
            d_img_out.copy(), d_txt_out.copy(), w, img_mod, txt_mod,
            bl, saved.dbl_saved[di], pe_cos, pe_sin, D, F, eps, ctx,
        )
        d_img_out = bg.base.img.d_x.copy()
        d_txt_out = bg.base.txt.d_x.copy()

        # scatter LoRA grads (slot layout: img 0..5, txt 6..11)
        var base_slot = di * DBL_SLOTS
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

        # bg.base modvec grads discarded — frozen mod-MLP not a LoRA target.
        loader.mark_active_block_done(ctx)
        di -= 1

    # frozen input-projection backward intentionally skipped (header).
    return MageFlowLoraGradSet(d_a_flat^, d_b_flat^, nonfinite)


# ── AdamW over the flat adapter set (reuses the shared _lora_adamw) ───────────
def mageflow_lora_adamw_step(
    mut set: MageFlowLoraSet, grads: MageFlowLoraGradSet,
    t: Int, lr: Float32, ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    if len(grads.d_a) != len(set.dbl) or len(grads.d_b) != len(set.dbl):
        raise Error("mageflow_lora_adamw_step: grad/adapter count mismatch")
    for i in range(len(set.dbl)):
        _lora_adamw(
            set.dbl[i],
            LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy()),
            t, lr, ctx, beta1, beta2, eps, weight_decay,
        )


# ── ai-toolkit save keys: diffusion_model.transformer_blocks.{i}.<suffix> ─────
def _mageflow_lora_prefix(block_idx: Int, slot: Int) -> String:
    return String("diffusion_model.transformer_blocks.") + String(block_idx) \
        + "." + _slot_suffix(slot)


def mageflow_lora_prefixes(num_double: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_double):
        for s in range(DBL_SLOTS):
            out.append(_mageflow_lora_prefix(bi, s))
    return out^


def _mageflow_named_loras(set: MageFlowLoraSet) -> List[NamedLora]:
    var named = List[NamedLora]()
    for bi in range(set.num_double):
        for s in range(DBL_SLOTS):
            named.append(NamedLora(
                _mageflow_lora_prefix(bi, s),
                set.dbl[bi * DBL_SLOTS + s].copy(),
            ))
    return named^


def save_mageflow_lora(
    set: MageFlowLoraSet, path: String, ctx: DeviceContext
) raises -> Int:
    # PEFT/ai-toolkit pairs: <prefix>.lora_A.weight / <prefix>.lora_B.weight.
    return save_lora_peft(_mageflow_named_loras(set), path, ctx)


def save_mageflow_lora_state(
    set: MageFlowLoraSet, path: String, ctx: DeviceContext,
    var meta: List[Float32] = List[Float32](),
) raises -> Int:
    return save_lora_train_state(_mageflow_named_loras(set), path, ctx, meta^)


def load_mageflow_lora_resume(
    num_double: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> MageFlowLoraSet:
    var prefixes = mageflow_lora_prefixes(num_double)
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(prefixes, scale, path, ctx)
    var dbl = List[LoraAdapter]()
    for i in range(num_double * DBL_SLOTS):
        dbl.append(named[i].adapter.copy())
    return MageFlowLoraSet(dbl^, num_double, rank)


def load_mageflow_lora_state(
    num_double: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> MageFlowLoraSet:
    var prefixes = mageflow_lora_prefixes(num_double)
    var scale = alpha / Float32(rank)
    var named = load_lora_train_state(prefixes, scale, path, ctx)
    var dbl = List[LoraAdapter]()
    for i in range(num_double * DBL_SLOTS):
        dbl.append(named[i].adapter.copy())
    return MageFlowLoraSet(dbl^, num_double, rank)
