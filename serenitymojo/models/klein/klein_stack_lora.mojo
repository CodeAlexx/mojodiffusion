# serenitymojo/models/klein/klein_stack_lora.mojo
#
# Klein (FLUX.2) FULL DiT STACK *WITH LoRA* on every trained attention
# projection: forward (saving acts) + full-depth backward (training) that uses
# the already-parity-verified per-block LoRA variants for EVERY block, COLLECTS
# every adapter's d_A/d_B, and supports an AdamW step + a PEFT/ai-toolkit save
# across all ~80 adapters. This file COMPOSES; it rebuilds NOTHING.
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/klein/double_block.mojo : double_block_lora_forward/backward
#       (img/txt × qkv/proj LoRA, d_A/d_B vs torch cos>=0.999).
#   * models/klein/single_block.mojo : single_block_lora_forward/backward
#       (qkv-rows on w1 + cols on w2 LoRA, d_A/d_B vs torch cos>=0.999).
#   * models/klein/klein_stack.mojo : the BASE full-stack fwd+bwd composition
#       (input proj → modulation → N double → concat → N single → final layer;
#       per-block recompute backward). THIS FILE IS THAT FILE with the base
#       per-block calls swapped for the LoRA variants + LoRA-grad collection.
#   * models/klein/lora_block.mojo + training/train_step.mojo : LoraAdapter,
#       _make_lora init (A small randn, B=0), _lora_adamw (the per-adapter AdamW).
#
# CARRIER DESIGN (Tenet-2: make the right thing easy)
#   With 8 double + 24 single blocks the trained-adapter count is large
#   (8×4 + 24×2 = 80 at real depth). Rather than 80 named fields, KleinLoraSet
#   holds ONE flat `List[LoraAdapter]` indexed by a deterministic scheme:
#       doubles first: block bi, slot s in {img_qkv,img_proj,txt_qkv,txt_proj}
#           flat = bi*4 + s                       (s = 0..3)
#       singles next : block bi, slot s in {qkv,out}
#           flat = num_double*4 + bi*2 + s        (s = 0..1)
#   The host optimizer/save path keeps this as the source of truth. The hot
#   trainer path builds a parallel `KleinLoraDeviceSet` once per step; transient
#   DoubleBlockLoraDevice / SingleBlockLoraDevice wrappers then borrow the same
#   resident A/B tensors across forward, backward recompute, and LoRA backward.
#   The backward SCATTERS the returned per-block d_A/d_B back into a matching
#   flat KleinLoraGrads. klein_lora_adamw_step walks the two flat lists in
#   lockstep and runs _lora_adamw on every adapter.
#
# SCOPE: LoRA-on-attention-projection training. Base weights (input/output proj,
#   modulation, the qkv/proj/w1/w2 linears) are FROZEN — their grads are computed
#   by the base path and discarded for the optimizer; only d_A/d_B are trained.
#   The shared modvec grads are still summed-across-blocks and returned (NOT
#   backpropped into the modulation MLP — that link is a later finetune phase).
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor move-only (return Movable structs, never
# store Tensor in a collection); `ArcPointer[Tensor]` is the Copyable device
# carrier; no-bias linear = linear(x, w, Optional[Tensor](None), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import concat, slice, zeros_device

from serenitymojo.models.klein.double_block import (
    DoubleBlockWeights, ModVecs, ModVecsDevice, modvecs_to_device,
    DoubleBlockSaved, DoubleBlockGrads,
    StreamLora, StreamLoraDevice, DoubleBlockLora, DoubleBlockLoraDevice, DoubleBlockLoraGrads,
    double_block_lora_forward, double_block_lora_backward,
    double_block_lora_forward_device, double_block_lora_backward_device,
    double_block_lora_forward_device_resident, double_block_lora_backward_device_resident,
    double_block_lora_forward_device_resident_scratch,
    double_block_lora_backward_device_resident_scratch,
)
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleModVecsDevice, single_modvecs_to_device,
    SingleBlockSaved, SingleBlockGrads,
    SingleBlockLora, SingleBlockLoraDevice, SingleBlockLoraGrads,
    single_block_lora_forward, single_block_lora_backward,
    single_block_lora_forward_device, single_block_lora_backward_device,
    single_block_lora_forward_device_resident,
    single_block_lora_forward_device_resident_scratch,
    single_block_lora_recompute_saved_device_resident,
    single_block_lora_recompute_saved_device_resident_scratch,
    single_block_lora_backward_device_resident,
    single_block_lora_backward_device_resident_scratch,
)
from serenitymojo.models.klein.klein_stack import (
    KleinStackBase, KleinStackForward,
    _add_lists, _zeros, _ones, _t, _linear_fwd, _linear_fwd_wdev,
    _concat_seq, _split_seq, _concat3, _modvec6,
)
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward

from serenitymojo.models.klein.lora_block import LoraAdapter, LoraAdapterDevice, lora_adapter_to_device
from serenitymojo.training.train_step import LoraGrads, _lora_adamw
from serenitymojo.training.lora_save import NamedLora, save_lora_peft, load_lora_for_resume


comptime TArc = ArcPointer[Tensor]


# ── flat-index slot scheme (the carrier's contract) ──────────────────────────
# Double slots (per block): 0=img_qkv 1=img_proj 2=txt_qkv 3=txt_proj.
# Single slots (per block): 0=qkv 1=out.
comptime DBL_SLOTS = 4
comptime SGL_SLOTS = 2
comptime BK_DOUBLE = 0
comptime BK_SINGLE = 1
comptime SGL_SAVE_TAIL = 9


# ── adapter init (A small randn, B=0 — PEFT identity at step 0) ───────────────
# Standalone of train_step._make_lora that takes rank/alpha directly (this file
# is config-agnostic; the loop-builder supplies rank/alpha).
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


# ── the LoRA carrier: every trained adapter, flat-indexed ────────────────────
struct KleinLoraSet(Copyable, Movable):
    var dbl: List[LoraAdapter]   # num_double * DBL_SLOTS, slot order img_qkv,img_proj,txt_qkv,txt_proj
    var sgl: List[LoraAdapter]   # num_single * SGL_SLOTS, slot order qkv,out
    var num_double: Int
    var num_single: Int
    var rank: Int

    def __init__(
        out self, var dbl: List[LoraAdapter], var sgl: List[LoraAdapter],
        num_double: Int, num_single: Int, rank: Int,
    ):
        self.dbl = dbl^
        self.sgl = sgl^
        self.num_double = num_double
        self.num_single = num_single
        self.rank = rank


struct KleinLoraDeviceSet(Copyable, Movable):
    var dbl: List[LoraAdapterDevice]   # same flat order as KleinLoraSet.dbl
    var sgl: List[LoraAdapterDevice]   # same flat order as KleinLoraSet.sgl
    var num_double: Int
    var num_single: Int
    var rank: Int

    def __init__(
        out self, var dbl: List[LoraAdapterDevice], var sgl: List[LoraAdapterDevice],
        num_double: Int, num_single: Int, rank: Int,
    ):
        self.dbl = dbl^
        self.sgl = sgl^
        self.num_double = num_double
        self.num_single = num_single
        self.rank = rank


def klein_lora_set_to_device(
    set: KleinLoraSet, ctx: DeviceContext
) raises -> KleinLoraDeviceSet:
    var dbl = List[LoraAdapterDevice]()
    var nd = set.num_double * DBL_SLOTS
    for i in range(nd):
        dbl.append(lora_adapter_to_device(set.dbl[i], ctx))
    var sgl = List[LoraAdapterDevice]()
    var ns = set.num_single * SGL_SLOTS
    for i in range(ns):
        sgl.append(lora_adapter_to_device(set.sgl[i], ctx))
    return KleinLoraDeviceSet(dbl^, sgl^, set.num_double, set.num_single, set.rank)


# Accessor by (block_kind, block_idx, slot) → a COPY of the adapter. (LoraAdapter
# is Copyable; this is the read accessor the task asks for.)
def klein_lora_get(
    set: KleinLoraSet, block_kind: Int, block_idx: Int, slot: Int
) -> LoraAdapter:
    if block_kind == BK_DOUBLE:
        return set.dbl[block_idx * DBL_SLOTS + slot].copy()
    return set.sgl[block_idx * SGL_SLOTS + slot].copy()


# ── build the full LoRA set for a Klein stack ────────────────────────────────
# dims: D (model dim) for the projection in/out shapes:
#   double img/txt qkv : in=D out=3D ; double img/txt proj : in=D out=D.
#   single qkv (w1 rows): in=D out=3D ; single out (w2 cols): in=D out=D.
# Each adapter gets a distinct seed so A is non-degenerate per slot.
def build_klein_lora_set(
    num_double: Int, num_single: Int, D: Int, rank: Int, alpha: Float32
) -> KleinLoraSet:
    var dbl = List[LoraAdapter]()
    var seed = UInt64(1000)
    for _ in range(num_double):
        # slot 0 img_qkv (in=D,out=3D)
        dbl.append(make_lora_adapter(rank, alpha, D, 3 * D, seed)); seed += 1
        # slot 1 img_proj (in=D,out=D)
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
        # slot 2 txt_qkv (in=D,out=3D)
        dbl.append(make_lora_adapter(rank, alpha, D, 3 * D, seed)); seed += 1
        # slot 3 txt_proj (in=D,out=D)
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
    var sgl = List[LoraAdapter]()
    for _ in range(num_single):
        # slot 0 qkv on w1 rows (in=D,out=3D)
        sgl.append(make_lora_adapter(rank, alpha, D, 3 * D, seed)); seed += 1
        # slot 1 out on w2 cols (in=D,out=D)
        sgl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1
    return KleinLoraSet(dbl^, sgl^, num_double, num_single, rank)


# build a transient DoubleBlockLora for block bi from the flat set.
def _dbl_lora_for(set: KleinLoraSet, bi: Int) -> DoubleBlockLora:
    var base = bi * DBL_SLOTS
    var img = StreamLora(
        Optional[LoraAdapter](set.dbl[base + 0].copy()),
        Optional[LoraAdapter](set.dbl[base + 1].copy()),
    )
    var txt = StreamLora(
        Optional[LoraAdapter](set.dbl[base + 2].copy()),
        Optional[LoraAdapter](set.dbl[base + 3].copy()),
    )
    return DoubleBlockLora(img^, txt^)


# build a transient SingleBlockLora for block bi from the flat set.
def _sgl_lora_for(set: KleinLoraSet, bi: Int) -> SingleBlockLora:
    var base = bi * SGL_SLOTS
    return SingleBlockLora(
        Optional[LoraAdapter](set.sgl[base + 0].copy()),
        Optional[LoraAdapter](set.sgl[base + 1].copy()),
    )


def _dbl_lora_dev_for(set: KleinLoraDeviceSet, bi: Int) -> DoubleBlockLoraDevice:
    var base = bi * DBL_SLOTS
    var img = StreamLoraDevice(
        Optional[LoraAdapterDevice](set.dbl[base + 0].copy()),
        Optional[LoraAdapterDevice](set.dbl[base + 1].copy()),
    )
    var txt = StreamLoraDevice(
        Optional[LoraAdapterDevice](set.dbl[base + 2].copy()),
        Optional[LoraAdapterDevice](set.dbl[base + 3].copy()),
    )
    return DoubleBlockLoraDevice(img^, txt^)


def _sgl_lora_dev_for(set: KleinLoraDeviceSet, bi: Int) -> SingleBlockLoraDevice:
    var base = bi * SGL_SLOTS
    return SingleBlockLoraDevice(
        Optional[LoraAdapterDevice](set.sgl[base + 0].copy()),
        Optional[LoraAdapterDevice](set.sgl[base + 1].copy()),
    )


def _concat6(
    a: List[Float32], b: List[Float32], c: List[Float32],
    d: List[Float32], e: List[Float32], f: List[Float32],
) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i])
    for i in range(len(b)):
        o.append(b[i])
    for i in range(len(c)):
        o.append(c[i])
    for i in range(len(d)):
        o.append(d[i])
    for i in range(len(e)):
        o.append(e[i])
    for i in range(len(f)):
        o.append(f[i])
    return o^


# ── the collected LoRA grads (flat, parallel to KleinLoraSet) ────────────────
# Plus the base-weight grads (computed, discarded for the optimizer) and the
# load-bearing input-token grads + shared modvec grads (same as KleinStackGrads).
struct KleinLoraGrads(Copyable, Movable):
    # flat LoRA grads: d_a/d_b per adapter, SAME flat order as KleinLoraSet.dbl/sgl.
    var dbl_d_a: List[List[Float32]]   # num_double*DBL_SLOTS
    var dbl_d_b: List[List[Float32]]
    var sgl_d_a: List[List[Float32]]   # num_single*SGL_SLOTS
    var sgl_d_b: List[List[Float32]]
    # load-bearing input-token grads (prove the whole chain).
    var d_img_tokens: List[Float32]
    var d_txt_tokens: List[Float32]
    # shared modulation-vector grads (summed across blocks; NOT into the mod MLP).
    var d_img_mod: List[Float32]       # [6D]
    var d_txt_mod: List[Float32]       # [6D]
    var d_single_mod: List[Float32]    # [3D]
    # base-weight grads (optional to consume; FROZEN params, discarded by AdamW).
    var d_img_in: List[Float32]
    var d_txt_in: List[Float32]
    var d_final_lin: List[Float32]
    var d_final_shift: List[Float32]
    var d_final_scale: List[Float32]

    def __init__(
        out self,
        var dbl_d_a: List[List[Float32]], var dbl_d_b: List[List[Float32]],
        var sgl_d_a: List[List[Float32]], var sgl_d_b: List[List[Float32]],
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        var d_img_mod: List[Float32], var d_txt_mod: List[Float32],
        var d_single_mod: List[Float32],
        var d_img_in: List[Float32], var d_txt_in: List[Float32],
        var d_final_lin: List[Float32],
        var d_final_shift: List[Float32], var d_final_scale: List[Float32],
    ):
        self.dbl_d_a = dbl_d_a^
        self.dbl_d_b = dbl_d_b^
        self.sgl_d_a = sgl_d_a^
        self.sgl_d_b = sgl_d_b^
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.d_img_mod = d_img_mod^
        self.d_txt_mod = d_txt_mod^
        self.d_single_mod = d_single_mod^
        self.d_img_in = d_img_in^
        self.d_txt_in = d_txt_in^
        self.d_final_lin = d_final_lin^
        self.d_final_shift = d_final_shift^
        self.d_final_scale = d_final_scale^


# ── FULL FORWARD WITH LoRA (checkpoint inputs only retained) ─────────────────
# Mirrors klein_stack_forward exactly, swapping the per-block calls for the
# LoRA variants. `saved` carries the LoRA-MODIFIED activations so the backward
# recompute regenerates them identically.
def klein_stack_lora_forward_device_inputs_resident_moddev_rope[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> KleinStackForward:
    var num_double = len(dbw)
    var num_single = len(sbw)

    var no_bias = Optional[Tensor](None)
    var img = TArc(linear(img_tokens_t[], base.img_in[], no_bias^, ctx))
    var no_bias_txt = Optional[Tensor](None)
    var txt = TArc(linear(txt_tokens_t[], base.txt_in[], no_bias_txt^, ctx))
    var img_in_act = img.copy()
    var txt_in_act = txt.copy()

    var dbl_img_in = List[TArc]()
    var dbl_txt_in = List[TArc]()
    var dbl_saved = List[DoubleBlockSaved]()
    for bi in range(num_double):
        dbl_img_in.append(img.copy())
        dbl_txt_in.append(txt.copy())
        var bl = _dbl_lora_dev_for(lora, bi)
        var fwd = double_block_lora_forward_device_resident[H, Dh, N_IMG, N_TXT, S](
            img, txt, dbw[bi], img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, ctx,
        )
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()

    var x = TArc(concat(0, ctx, txt[], img[]))

    var sgl_x_in = List[TArc]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        sgl_x_in.append(x.copy())
        var sl = _sgl_lora_dev_for(lora, bi)
        var fwd = single_block_lora_forward_device_resident[H, Dh, S](
            x, sbw[bi], single_mod_dev, sl, cos_t, sin_t, D, F, eps, ctx,
        )
        if bi >= num_single - SGL_SAVE_TAIL:
            sgl_saved.append(fwd.saved.copy())
        x = fwd.out.copy()

    var img_out = TArc(slice(x[], 0, N_TXT, N_IMG, ctx))

    var ln_img_out = TArc(layer_norm(
        img_out[], _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), eps, ctx,
    ))
    var normed = modulate(
        ln_img_out[],
        base.final_scale[], base.final_shift[], ctx,
    )
    var no_bias_out = Optional[Tensor](None)
    var out = linear(normed, base.final_lin[], no_bias_out^, ctx).to_host(ctx)

    return KleinStackForward(
        out^, img_in_act^, txt_in_act^,
        dbl_img_in^, dbl_txt_in^, sgl_x_in^,
        dbl_saved^, sgl_saved^,
        img_out^, ln_img_out^,
    )


def klein_stack_lora_forward_device_inputs_resident_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> KleinStackForward:
    var num_double = len(dbw)
    var num_single = len(sbw)

    var no_bias = Optional[Tensor](None)
    var img = TArc(linear(img_tokens_t[], base.img_in[], no_bias^, ctx))
    var no_bias_txt = Optional[Tensor](None)
    var txt = TArc(linear(txt_tokens_t[], base.txt_in[], no_bias_txt^, ctx))
    var img_in_act = img.copy()
    var txt_in_act = txt.copy()
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    var dbl_img_in = List[TArc]()
    var dbl_txt_in = List[TArc]()
    var dbl_saved = List[DoubleBlockSaved]()
    for bi in range(num_double):
        dbl_img_in.append(img.copy())
        dbl_txt_in.append(txt.copy())
        var bl = _dbl_lora_dev_for(lora, bi)
        var fwd = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            img, txt, dbw[bi], img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()

    var x = TArc(concat(0, ctx, txt[], img[]))

    var sgl_x_in = List[TArc]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        sgl_x_in.append(x.copy())
        var sl = _sgl_lora_dev_for(lora, bi)
        var fwd = single_block_lora_forward_device_resident_scratch[H, Dh, S](
            x, sbw[bi], single_mod_dev, sl, cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        if bi >= num_single - SGL_SAVE_TAIL:
            sgl_saved.append(fwd.saved.copy())
        x = fwd.out.copy()

    var img_out = TArc(slice(x[], 0, N_TXT, N_IMG, ctx))

    var ln_img_out = TArc(layer_norm(
        img_out[], norm_ones[], norm_zeros[], eps, ctx,
    ))
    var normed = modulate(
        ln_img_out[],
        base.final_scale[], base.final_shift[], ctx,
    )
    var no_bias_out = Optional[Tensor](None)
    var out = linear(normed, base.final_lin[], no_bias_out^, ctx).to_host(ctx)

    return KleinStackForward(
        out^, img_in_act^, txt_in_act^,
        dbl_img_in^, dbl_txt_in^, sgl_x_in^,
        dbl_saved^, sgl_saved^,
        img_out^, ln_img_out^,
    )


def klein_stack_lora_forward_device_inputs_resident_moddev[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos: List[Float32], sin: List[Float32],
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> KleinStackForward:
    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    return klein_stack_lora_forward_device_inputs_resident_moddev_rope[
        H, Dh, N_IMG, N_TXT, S
    ](
        img_tokens_t, txt_tokens_t, base, dbw, sbw, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t,
        D, F, in_ch, txt_ch, out_ch, eps, ctx,
    )


def klein_stack_lora_forward_device_inputs_resident[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod: ModVecs, txt_mod: ModVecs, single_mod: SingleModVecs,
    cos: List[Float32], sin: List[Float32],
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> KleinStackForward:
    var img_mod_dev = modvecs_to_device(img_mod, D, ctx)
    var txt_mod_dev = modvecs_to_device(txt_mod, D, ctx)
    var single_mod_dev = single_modvecs_to_device(single_mod, D, ctx)
    return klein_stack_lora_forward_device_inputs_resident_moddev[
        H, Dh, N_IMG, N_TXT, S
    ](
        img_tokens_t, txt_tokens_t, base, dbw, sbw, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos, sin,
        D, F, in_ch, txt_ch, out_ch, eps, ctx,
    )


def klein_stack_lora_forward_device_inputs[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraSet,
    img_mod: ModVecs, txt_mod: ModVecs, single_mod: SingleModVecs,
    cos: List[Float32], sin: List[Float32],
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> KleinStackForward:
    var lora_dev = klein_lora_set_to_device(lora, ctx)
    return klein_stack_lora_forward_device_inputs_resident[H, Dh, N_IMG, N_TXT, S](
        img_tokens_t, txt_tokens_t, base, dbw, sbw, lora_dev,
        img_mod, txt_mod, single_mod, cos, sin,
        D, F, in_ch, txt_ch, out_ch, eps, ctx,
    )


def klein_stack_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraSet,
    img_mod: ModVecs, txt_mod: ModVecs, single_mod: SingleModVecs,
    cos: List[Float32], sin: List[Float32],
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> KleinStackForward:
    return klein_stack_lora_forward_device_inputs[H, Dh, N_IMG, N_TXT, S](
        TArc(_t(img_tokens, [N_IMG, in_ch], ctx)),
        TArc(_t(txt_tokens, [N_TXT, txt_ch], ctx)),
        base, dbw, sbw, lora, img_mod, txt_mod, single_mod, cos, sin,
        D, F, in_ch, txt_ch, out_ch, eps, ctx,
    )


# ── FULL BACKWARD WITH LoRA (full-depth; per-block recompute) ────────────────
# Mirrors klein_stack_backward, calling the LoRA per-block backward and
# COLLECTING every adapter's d_A/d_B into the flat KleinLoraGrads.
def klein_stack_lora_backward_resident_moddev_rope[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    saved: KleinStackForward,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    compute_input_grads: Bool = True,
    compute_aux_grads: Bool = True,
) raises -> KleinLoraGrads:
    var num_double = len(dbw)
    var num_single = len(sbw)

    # ── final layer backward (frozen base) ──
    # final_lin is frozen in LoRA training; only d_x flows into the final norm.
    var d_normed_t = linear_backward_dx(
        _t(d_out, [N_IMG, out_ch], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )

    var mbf = modulate_backward(
        d_normed_t, saved.ln_img_out[],
        base.final_scale[], ctx, compute_aux_grads,
    )
    var d_final_scale = List[Float32]()
    var d_final_shift = List[Float32]()
    if compute_aux_grads:
        d_final_scale = mbf.d_scale.to_host(ctx)
        d_final_shift = mbf.d_shift.to_host(ctx)

    var d_img_out_t = layer_norm_backward_dx(
        mbf.d_x, saved.img_out[],
        _t(_ones(D), [D], ctx), eps, ctx,
    )

    var d_txt_zero = zeros_device([N_TXT, D], STDtype.F32, ctx)
    var d_x = TArc(concat(0, ctx, d_txt_zero, d_img_out_t))

    # flat single LoRA grads collected in FORWARD order (block 0..num_single-1).
    var sgl_d_a = List[List[Float32]]()
    var sgl_d_b = List[List[Float32]]()
    for _ in range(num_single * SGL_SLOTS):
        sgl_d_a.append(List[Float32]())
        sgl_d_b.append(List[Float32]())

    # ── single-stream backward (REVERSE; per-block recompute) ──
    var d_single_mod = _zeros(3 * D)
    var bi = num_single - 1
    var saved_single_start = num_single - len(saved.sgl_saved)
    while bi >= 0:
        var sl = _sgl_lora_dev_for(lora, bi)
        var block_saved: SingleBlockSaved
        if bi >= saved_single_start:
            block_saved = saved.sgl_saved[bi - saved_single_start].copy()
        else:
            block_saved = single_block_lora_recompute_saved_device_resident[H, Dh, S](
                saved.sgl_x_in[bi], sbw[bi], single_mod_dev, sl,
                cos_t, sin_t, D, F, eps, ctx,
            )
        var bg = single_block_lora_backward_device_resident[H, Dh, S](
            d_x, sbw[bi], single_mod_dev, sl, block_saved, cos_t, sin_t,
            D, F, eps, ctx, compute_aux_grads,
        )
        d_x = bg.d_x.copy()
        if compute_aux_grads:
            d_single_mod = _add_lists(
                d_single_mod,
                _concat3(bg.d_shift, bg.d_scale, bg.d_gate),
            )
        # scatter into the flat slots (qkv=slot0, out=slot1).
        var sbase = bi * SGL_SLOTS
        sgl_d_a[sbase + 0] = bg.qkv_d_a.copy()
        sgl_d_b[sbase + 0] = bg.qkv_d_b.copy()
        sgl_d_a[sbase + 1] = bg.out_d_a.copy()
        sgl_d_b[sbase + 1] = bg.out_d_b.copy()
        bi -= 1

    # double→single seam: split d_x [S,D] back into d_txt_out, d_img_out.
    var d_txt_out = TArc(slice(d_x[], 0, 0, N_TXT, ctx))
    var d_img_out2 = TArc(slice(d_x[], 0, N_TXT, N_IMG, ctx))

    # flat double LoRA grads (slot order img_qkv,img_proj,txt_qkv,txt_proj).
    var dbl_d_a = List[List[Float32]]()
    var dbl_d_b = List[List[Float32]]()
    for _ in range(num_double * DBL_SLOTS):
        dbl_d_a.append(List[Float32]())
        dbl_d_b.append(List[Float32]())

    # ── double-stream backward (REVERSE; per-block recompute) ──
    var d_img_mod = _zeros(6 * D)
    var d_txt_mod = _zeros(6 * D)
    var di = num_double - 1
    var d_io = d_img_out2.copy()
    var d_to = d_txt_out.copy()
    while di >= 0:
        var bl = _dbl_lora_dev_for(lora, di)
        var fwd = double_block_lora_forward_device_resident[H, Dh, N_IMG, N_TXT, S](
            saved.dbl_img_in[di], saved.dbl_txt_in[di],
            dbw[di], img_mod_dev, txt_mod_dev, bl, cos_t, sin_t, D, F, eps, ctx,
        )
        var bg = double_block_lora_backward_device_resident[H, Dh, N_IMG, N_TXT, S](
            d_io, d_to, dbw[di], img_mod_dev, txt_mod_dev, bl, fwd.saved,
            cos_t, sin_t, D, F, eps, ctx, compute_aux_grads,
        )
        d_io = bg.img.d_x.copy()
        d_to = bg.txt.d_x.copy()
        if compute_aux_grads:
            d_img_mod = _add_lists(
                d_img_mod,
                _concat6(bg.img.d_shift1, bg.img.d_scale1, bg.img.d_gate1,
                         bg.img.d_shift2, bg.img.d_scale2, bg.img.d_gate2),
            )
            d_txt_mod = _add_lists(
                d_txt_mod,
                _concat6(bg.txt.d_shift1, bg.txt.d_scale1, bg.txt.d_gate1,
                         bg.txt.d_shift2, bg.txt.d_scale2, bg.txt.d_gate2),
            )
        # scatter into the flat slots.
        var dbase = di * DBL_SLOTS
        dbl_d_a[dbase + 0] = bg.img.qkv_d_a.copy()
        dbl_d_b[dbase + 0] = bg.img.qkv_d_b.copy()
        dbl_d_a[dbase + 1] = bg.img.proj_d_a.copy()
        dbl_d_b[dbase + 1] = bg.img.proj_d_b.copy()
        dbl_d_a[dbase + 2] = bg.txt.qkv_d_a.copy()
        dbl_d_b[dbase + 2] = bg.txt.qkv_d_b.copy()
        dbl_d_a[dbase + 3] = bg.txt.proj_d_a.copy()
        dbl_d_b[dbase + 3] = bg.txt.proj_d_b.copy()
        di -= 1

    var d_img_in = List[Float32]()
    var d_txt_in = List[Float32]()
    var d_img_tokens = List[Float32]()
    var d_txt_tokens = List[Float32]()
    if compute_input_grads:
        # ── input-projection backward (frozen base; d_tokens load-bearing for
        # parity, but unused by the real LoRA trainer).
        var d_img_tokens_t = linear_backward_dx(
            d_io[], base.img_in[], N_IMG, in_ch, D, ctx,
        )
        d_img_tokens = d_img_tokens_t.to_host(ctx)

        var d_txt_tokens_t = linear_backward_dx(
            d_to[], base.txt_in[], N_TXT, txt_ch, D, ctx,
        )
        d_txt_tokens = d_txt_tokens_t.to_host(ctx)

    return KleinLoraGrads(
        dbl_d_a^, dbl_d_b^, sgl_d_a^, sgl_d_b^,
        d_img_tokens^, d_txt_tokens^,
        d_img_mod^, d_txt_mod^, d_single_mod^,
        d_img_in^, d_txt_in^, List[Float32](), d_final_shift^, d_final_scale^,
    )


def klein_stack_lora_backward_resident_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    saved: KleinStackForward,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_input_grads: Bool = True,
    compute_aux_grads: Bool = True,
) raises -> KleinLoraGrads:
    var num_double = len(dbw)
    var num_single = len(sbw)
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    var d_normed_t = linear_backward_dx(
        _t(d_out, [N_IMG, out_ch], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )

    var mbf = modulate_backward(
        d_normed_t, saved.ln_img_out[],
        base.final_scale[], ctx, compute_aux_grads,
    )
    var d_final_scale = List[Float32]()
    var d_final_shift = List[Float32]()
    if compute_aux_grads:
        d_final_scale = mbf.d_scale.to_host(ctx)
        d_final_shift = mbf.d_shift.to_host(ctx)

    var d_img_out_t = layer_norm_backward_dx(
        mbf.d_x, saved.img_out[],
        norm_ones[], eps, ctx,
    )

    var d_txt_zero = zeros_device([N_TXT, D], STDtype.F32, ctx)
    var d_x = TArc(concat(0, ctx, d_txt_zero, d_img_out_t))

    var sgl_d_a = List[List[Float32]]()
    var sgl_d_b = List[List[Float32]]()
    for _ in range(num_single * SGL_SLOTS):
        sgl_d_a.append(List[Float32]())
        sgl_d_b.append(List[Float32]())

    var d_single_mod = _zeros(3 * D)
    var bi = num_single - 1
    var saved_single_start = num_single - len(saved.sgl_saved)
    while bi >= 0:
        var sl = _sgl_lora_dev_for(lora, bi)
        var block_saved: SingleBlockSaved
        if bi >= saved_single_start:
            block_saved = saved.sgl_saved[bi - saved_single_start].copy()
        else:
            block_saved = single_block_lora_recompute_saved_device_resident_scratch[H, Dh, S](
                saved.sgl_x_in[bi], sbw[bi], single_mod_dev, sl,
                cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
            )
        var bg = single_block_lora_backward_device_resident_scratch[H, Dh, S](
            d_x, sbw[bi], single_mod_dev, sl, block_saved, cos_t, sin_t,
            D, F, eps, norm_ones[], ctx, scratch, compute_aux_grads,
        )
        d_x = bg.d_x.copy()
        if compute_aux_grads:
            d_single_mod = _add_lists(
                d_single_mod,
                _concat3(bg.d_shift, bg.d_scale, bg.d_gate),
            )
        var sbase = bi * SGL_SLOTS
        sgl_d_a[sbase + 0] = bg.qkv_d_a.copy()
        sgl_d_b[sbase + 0] = bg.qkv_d_b.copy()
        sgl_d_a[sbase + 1] = bg.out_d_a.copy()
        sgl_d_b[sbase + 1] = bg.out_d_b.copy()
        bi -= 1

    var d_txt_out = TArc(slice(d_x[], 0, 0, N_TXT, ctx))
    var d_img_out2 = TArc(slice(d_x[], 0, N_TXT, N_IMG, ctx))

    var dbl_d_a = List[List[Float32]]()
    var dbl_d_b = List[List[Float32]]()
    for _ in range(num_double * DBL_SLOTS):
        dbl_d_a.append(List[Float32]())
        dbl_d_b.append(List[Float32]())

    var d_img_mod = _zeros(6 * D)
    var d_txt_mod = _zeros(6 * D)
    var di = num_double - 1
    var d_io = d_img_out2.copy()
    var d_to = d_txt_out.copy()
    while di >= 0:
        var bl = _dbl_lora_dev_for(lora, di)
        var fwd = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            saved.dbl_img_in[di], saved.dbl_txt_in[di],
            dbw[di], img_mod_dev, txt_mod_dev, bl, cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        var bg = double_block_lora_backward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            d_io, d_to, dbw[di], img_mod_dev, txt_mod_dev, bl, fwd.saved,
            cos_t, sin_t, D, F, eps, norm_ones[], ctx, scratch, compute_aux_grads,
        )
        d_io = bg.img.d_x.copy()
        d_to = bg.txt.d_x.copy()
        if compute_aux_grads:
            d_img_mod = _add_lists(
                d_img_mod,
                _concat6(bg.img.d_shift1, bg.img.d_scale1, bg.img.d_gate1,
                         bg.img.d_shift2, bg.img.d_scale2, bg.img.d_gate2),
            )
            d_txt_mod = _add_lists(
                d_txt_mod,
                _concat6(bg.txt.d_shift1, bg.txt.d_scale1, bg.txt.d_gate1,
                         bg.txt.d_shift2, bg.txt.d_scale2, bg.txt.d_gate2),
            )
        var dbase = di * DBL_SLOTS
        dbl_d_a[dbase + 0] = bg.img.qkv_d_a.copy()
        dbl_d_b[dbase + 0] = bg.img.qkv_d_b.copy()
        dbl_d_a[dbase + 1] = bg.img.proj_d_a.copy()
        dbl_d_b[dbase + 1] = bg.img.proj_d_b.copy()
        dbl_d_a[dbase + 2] = bg.txt.qkv_d_a.copy()
        dbl_d_b[dbase + 2] = bg.txt.qkv_d_b.copy()
        dbl_d_a[dbase + 3] = bg.txt.proj_d_a.copy()
        dbl_d_b[dbase + 3] = bg.txt.proj_d_b.copy()
        di -= 1

    var d_img_in = List[Float32]()
    var d_txt_in = List[Float32]()
    var d_img_tokens = List[Float32]()
    var d_txt_tokens = List[Float32]()
    if compute_input_grads:
        var d_img_tokens_t = linear_backward_dx(
            d_io[], base.img_in[], N_IMG, in_ch, D, ctx,
        )
        d_img_tokens = d_img_tokens_t.to_host(ctx)

        var d_txt_tokens_t = linear_backward_dx(
            d_to[], base.txt_in[], N_TXT, txt_ch, D, ctx,
        )
        d_txt_tokens = d_txt_tokens_t.to_host(ctx)

    return KleinLoraGrads(
        dbl_d_a^, dbl_d_b^, sgl_d_a^, sgl_d_b^,
        d_img_tokens^, d_txt_tokens^,
        d_img_mod^, d_txt_mod^, d_single_mod^,
        d_img_in^, d_txt_in^, List[Float32](), d_final_shift^, d_final_scale^,
    )


def klein_stack_lora_backward_resident_moddev[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos: List[Float32], sin: List[Float32],
    saved: KleinStackForward,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    compute_input_grads: Bool = True,
    compute_aux_grads: Bool = True,
) raises -> KleinLoraGrads:
    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    return klein_stack_lora_backward_resident_moddev_rope[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens, txt_tokens, base, dbw, sbw, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t, saved,
        D, F, in_ch, txt_ch, out_ch, eps, ctx, compute_input_grads,
        compute_aux_grads,
    )


def klein_stack_lora_backward_resident[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod: ModVecs, txt_mod: ModVecs, single_mod: SingleModVecs,
    cos: List[Float32], sin: List[Float32],
    saved: KleinStackForward,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    compute_input_grads: Bool = True,
    compute_aux_grads: Bool = True,
) raises -> KleinLoraGrads:
    var img_mod_dev = modvecs_to_device(img_mod, D, ctx)
    var txt_mod_dev = modvecs_to_device(txt_mod, D, ctx)
    var single_mod_dev = single_modvecs_to_device(single_mod, D, ctx)
    return klein_stack_lora_backward_resident_moddev[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens, txt_tokens, base, dbw, sbw, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos, sin, saved,
        D, F, in_ch, txt_ch, out_ch, eps, ctx, compute_input_grads,
        compute_aux_grads,
    )


def klein_stack_lora_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    lora: KleinLoraSet,
    img_mod: ModVecs, txt_mod: ModVecs, single_mod: SingleModVecs,
    cos: List[Float32], sin: List[Float32],
    saved: KleinStackForward,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    compute_input_grads: Bool = True,
    compute_aux_grads: Bool = True,
) raises -> KleinLoraGrads:
    var lora_dev = klein_lora_set_to_device(lora, ctx)
    return klein_stack_lora_backward_resident[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens, txt_tokens, base, dbw, sbw, lora_dev,
        img_mod, txt_mod, single_mod, cos, sin, saved,
        D, F, in_ch, txt_ch, out_ch, eps, ctx, compute_input_grads, compute_aux_grads,
    )


# ── AdamW step on EVERY adapter (reuses the proven per-adapter _lora_adamw) ───
# Walks the flat dbl/sgl adapter lists in lockstep with the flat grads and
# mutates A/B (and the carried ma/va/mb/vb) in place. `t` is the 1-based step.
def klein_lora_adamw_step(
    mut set: KleinLoraSet, grads: KleinLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
) raises:
    var nd = set.num_double * DBL_SLOTS
    for i in range(nd):
        var lg = LoraGrads(grads.dbl_d_a[i].copy(), grads.dbl_d_b[i].copy())
        _lora_adamw(set.dbl[i], lg, t, lr, ctx)
    var ns = set.num_single * SGL_SLOTS
    for i in range(ns):
        var lg = LoraGrads(grads.sgl_d_a[i].copy(), grads.sgl_d_b[i].copy())
        _lora_adamw(set.sgl[i], lg, t, lr, ctx)


# ── per-block PEFT/ai-toolkit prefix scheme (the INVERSE of lora.mojo loader) ─
# lora.mojo::_map_klein_trainer accepts exactly these prefixes (lora.mojo:215-237):
#   double_blocks.<i>.img_attn.qkv_proj   (FULL slot on .img_attn.qkv.weight)
#   double_blocks.<i>.img_attn.out_proj   (FULL slot on .img_attn.proj.weight)
#   double_blocks.<i>.txt_attn.qkv_proj   (FULL slot on .txt_attn.qkv.weight)
#   double_blocks.<i>.txt_attn.out_proj   (FULL slot on .txt_attn.proj.weight)
#   single_blocks.<i>.qkv_proj            (ROWS slot on .linear1.weight)
#   single_blocks.<i>.out_proj            (COLS slot on .linear2.weight)
# save_lora_peft appends .lora_A.weight / .lora_B.weight, so the saved file is
# byte-exact loadable by lora.mojo (DiffusionModel format) and ai-toolkit.
def _klein_lora_prefix(block_kind: Int, block_idx: Int, slot: Int) -> String:
    if block_kind == BK_DOUBLE:
        var b = String("double_blocks.") + String(block_idx)
        if slot == 0:
            return b + ".img_attn.qkv_proj"
        elif slot == 1:
            return b + ".img_attn.out_proj"
        elif slot == 2:
            return b + ".txt_attn.qkv_proj"
        else:
            return b + ".txt_attn.out_proj"
    var s = String("single_blocks.") + String(block_idx)
    if slot == 0:
        return s + ".qkv_proj"
    return s + ".out_proj"


# Build the ordered prefix list for the whole set (doubles first, then singles),
# matching the flat KleinLoraSet index order. Exposed so the resume loader and
# the save use the SAME order.
def klein_lora_prefixes(num_double: Int, num_single: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_double):
        for s in range(DBL_SLOTS):
            out.append(_klein_lora_prefix(BK_DOUBLE, bi, s))
    for bi in range(num_single):
        for s in range(SGL_SLOTS):
            out.append(_klein_lora_prefix(BK_SINGLE, bi, s))
    return out^


# ── SAVE every adapter as a PEFT/ai-toolkit safetensors ──────────────────────
# Returns the number of (A,B) pairs written (== total adapters).
def save_klein_lora(set: KleinLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.num_double):
        for s in range(DBL_SLOTS):
            named.append(NamedLora(
                _klein_lora_prefix(BK_DOUBLE, bi, s),
                set.dbl[bi * DBL_SLOTS + s].copy(),
            ))
    for bi in range(set.num_single):
        for s in range(SGL_SLOTS):
            named.append(NamedLora(
                _klein_lora_prefix(BK_SINGLE, bi, s),
                set.sgl[bi * SGL_SLOTS + s].copy(),
            ))
    return save_lora_peft(named, path, ctx)


# ── RESUME: load the adapter A/B back from a save_klein_lora file ────────────
# AdamW moments are ZEROED (resume them from a loop TrainState checkpoint, the
# same contract as load_lora_for_resume). The returned set carries the SAME flat
# order build_klein_lora_set produces, so the AdamW step + save round-trip.
def load_klein_lora_resume(
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> KleinLoraSet:
    var prefixes = klein_lora_prefixes(num_double, num_single)
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(prefixes, scale, path, ctx)   # flat order
    # split the flat NamedLora list back into dbl/sgl flat adapter lists.
    var dbl = List[LoraAdapter]()
    var sgl = List[LoraAdapter]()
    var n_dbl = num_double * DBL_SLOTS
    for i in range(n_dbl):
        dbl.append(named[i].adapter.copy())
    var n_sgl = num_single * SGL_SLOTS
    for i in range(n_sgl):
        sgl.append(named[n_dbl + i].adapter.copy())
    return KleinLoraSet(dbl^, sgl^, num_double, num_single, rank)
