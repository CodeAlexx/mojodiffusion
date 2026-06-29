# serenitymojo/models/klein/klein_stack_lora.mojo
#
# Klein (FLUX.2) FULL DiT STACK *WITH LoRA* on every OneTrainer-wrapped
# projection: forward (saving acts) + full-depth backward (training) that uses
# the already-parity-verified per-block LoRA variants for EVERY block, COLLECTS
# every adapter's d_A/d_B, and supports an AdamW step + a PEFT/ai-toolkit save
# across all adapters. This file COMPOSES; it rebuilds NOTHING.
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/klein/double_block.mojo : double_block_lora_forward/backward
#       (img/txt × q/k/v/out/ff_in/ff_out LoRA slots).
#   * models/klein/single_block.mojo : single_block_lora_forward/backward
#       (full to_qkv_mlp_proj + full to_out LoRA slots).
#   * models/klein/klein_stack.mojo : the BASE full-stack fwd+bwd composition
#       (input proj → modulation → N double → concat → N single → final layer;
#       per-block recompute backward). THIS FILE IS THAT FILE with the base
#       per-block calls swapped for the LoRA variants + LoRA-grad collection.
#   * models/klein/lora_block.mojo + models/klein/lora_adapter.mojo :
#       LoraAdapter, _make_lora init (A small randn, B=0), _lora_adamw
#       (the per-adapter OneTrainer-style AdamW).
#
# CARRIER DESIGN (Tenet-2: make the right thing easy)
#   With 8 double + 24 single blocks the trained-adapter count is large
#   (8*12 + 24*2 = 144 at real depth). Rather than 144 named fields, KleinLoraSet
#   holds ONE flat `List[LoraAdapter]` indexed by a deterministic scheme:
#       doubles first: block bi, slot s in
#           img {q,k,v,out,ff_in,ff_out}, txt {q,k,v,out,ff_in,ff_out}
#           flat = bi*12 + s                      (s = 0..11)
#       singles next : block bi, slot s in {qkv,out}
#           flat = num_double*12 + bi*2 + s       (s = 0..1)
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

from std.gpu.host import DeviceContext, HostBuffer
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import concat, slice, zeros_device

from serenitymojo.models.klein.double_block import (
    StreamWeights, DoubleBlockWeights, ModVecs, ModVecsDevice, modvecs_to_device,
    DoubleBlockSaved, DoubleBlockGrads,
    StreamLora, StreamLoraDevice, DoubleBlockLora, DoubleBlockLoraDevice, DoubleBlockLoraGrads,
    double_block_lora_forward, double_block_lora_backward,
    double_block_lora_forward_device, double_block_lora_backward_device,
    double_block_lora_to_device,
    double_block_lora_forward_device_resident, double_block_lora_backward_device_resident,
    double_block_lora_forward_device_resident_scratch,
    double_block_lora_predict_device_resident_scratch,
    double_block_lora_backward_device_resident_scratch,
    double_block_lora_backward_device_resident_scratch_tensors,
    double_block_direct_dora_forward_device_resident_scratch,
    double_block_direct_dora_backward_device_resident_scratch,
    double_block_direct_oft_forward_device_resident_scratch,
    double_block_direct_oft_backward_device_resident_scratch,
    DoubleBlockDirectDoRAGradsT, DoubleBlockDirectOFTGradsT,
    StreamDirectDoRAGradsT, StreamDirectOFTGradsT,
)
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleModVecsDevice, single_modvecs_to_device,
    SingleBlockSaved, SingleBlockGrads,
    SingleBlockLora, SingleBlockLoraDevice, SingleBlockLoraGrads,
    single_block_lora_forward, single_block_lora_backward,
    single_block_lora_forward_device, single_block_lora_backward_device,
    single_block_lora_to_device,
    single_block_lora_forward_device_resident,
    single_block_lora_forward_device_resident_scratch,
    single_block_lora_predict_device_resident_scratch,
    single_block_lora_recompute_saved_device_resident,
    single_block_lora_recompute_saved_device_resident_scratch,
    single_block_lora_backward_device_resident,
    single_block_lora_backward_device_resident_scratch,
    single_block_lora_backward_device_resident_scratch_tensors,
    single_block_direct_dora_forward_device_resident_scratch,
    single_block_direct_dora_backward_device_resident_scratch,
    single_block_direct_oft_forward_device_resident_scratch,
    single_block_direct_oft_backward_device_resident_scratch,
    SingleBlockDirectDoRAGradsT, SingleBlockDirectOFTGradsT,
)
from serenitymojo.models.klein.klein_direct_lycoris_stack import (
    KleinStackDirectDoRA, KleinStackDirectOFT,
    KleinDirectDoRAGradT, KleinDirectOFTGradT,
    build_klein_direct_dora_set_from_checkpoint,
    build_klein_direct_oft_set_from_checkpoint,
    empty_klein_direct_dora_set, empty_klein_direct_oft_set,
    klein_direct_dense_carrier_bytes,
    klein_direct_dora_preflight, klein_direct_oft_preflight,
    klein_direct_dora_blocks_to_device, klein_direct_oft_blocks_to_device,
    klein_direct_dora_zero_grads, klein_direct_oft_zero_grads,
    klein_direct_dora_scatter_slot_grad, klein_direct_oft_scatter_slot_grad,
    klein_direct_dora_grad_norm, klein_direct_dora_clip_grads,
    klein_direct_dora_adamw_step, klein_direct_dora_zero_leg_l1,
    klein_direct_dora_trainable_bytes, save_klein_direct_dora,
    klein_direct_oft_grad_norm, klein_direct_oft_clip_grads,
    klein_direct_oft_adamw_step, klein_direct_oft_vec_l1,
    klein_direct_oft_trainable_bytes, save_klein_direct_oft,
)
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectDoRAGrads, FlatDirectOFTSet, FlatDirectOFTGrads,
)
from serenitymojo.training.dora_adapter import DoRAGrads
from serenitymojo.training.oft_onetrainer import OFTOTGrads
from serenitymojo.training.lokr_stack import LOKR_TGT_ALL, _slot_targeted
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

from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice, lora_adapter_to_device, _tensor_to_host_f32,
)
# P6 graph engine (AUTOGRAD_V2_MOJO_DESIGN.md): per-block recompute+backward
# driven by autograd_v2 (klein_stack_lora_backward_graph below).
from serenitymojo.autograd_v2.klein_block_graph import (
    klein_double_block_graph_backward,
    klein_single_block_graph_backward,
)
from serenitymojo.models.klein.lora_adapter import LoraAdapter, _lora_adamw_precomputed
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_onetrainer, load_lora_for_resume,
    save_lora_train_state, load_lora_train_state, _read_f32,
)
from serenitymojo.training.checkpoint import HostOffload, offload_to_host, restore_to_device
from serenitymojo.training.lora_adamw_ot_fused import (
    fused_lora_adamw_ot_step,
    LoraAdamWOTDeviceState, lora_adamw_ot_device_state_init,
    fused_lora_adamw_ot_step_resident, lora_adamw_ot_device_state_sync_moments,
)
from serenitymojo.models.klein.activation_tape import KleinStackLoraOffloadedTape
from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.ops.cast import cast_tensor


comptime TArc = ArcPointer[Tensor]


@fieldwise_init
struct KleinLoraCfgPreds(Movable):
    var pos: Tensor
    var neg: Tensor


# ── flat-index slot scheme (the carrier's contract) ──────────────────────────
# Double slots (per block, 12):
#   0=img_q 1=img_k 2=img_v 3=img_out 4=img_ff_in 5=img_ff_out
#   6=txt_q 7=txt_k 8=txt_v 9=txt_out 10=txt_ff_in 11=txt_ff_out
# Single slots (per block): 0=qkv 1=out.
comptime DBL_SLOTS = 12
comptime SGL_SLOTS = 2
comptime BK_DOUBLE = 0
comptime BK_SINGLE = 1
# Saving block activations reduces backward recompute but is expensive at real
# Klein sequence length. Keep production parity default at full recompute until
# activation offload is wired through the accepted train replay.
comptime DBL_SAVE_TAIL = 0
comptime SGL_SAVE_TAIL = 0


# ── adapter init (A = kaiming_uniform(a=sqrt(5)), B=0) ───────────────────────
# Matches OneTrainer LoRAModule.initialize_weights for lora_down/lora_up shape.
def _kaiming_uniform_a_sqrt5(n: Int, in_f: Int, seed: UInt64) -> List[Float32]:
    var bound = Float32(1.0) / sqrt(Float32(in_f))
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u * Float32(2.0) - Float32(1.0)) * bound)
    return out^


def make_lora_adapter(
    rank: Int, alpha: Float32, in_f: Int, out_f: Int, seed: UInt64
) -> LoraAdapter:
    var scale = alpha / Float32(rank)
    return LoraAdapter(
        _kaiming_uniform_a_sqrt5(rank * in_f, in_f, seed),
        _zeros(out_f * rank),              # B = 0 (adapter identity at init)
        rank, in_f, out_f, scale,
        _zeros(rank * in_f), _zeros(rank * in_f),    # ma / va
        _zeros(out_f * rank), _zeros(out_f * rank),  # mb / vb
    )


# ── the LoRA carrier: every trained adapter, flat-indexed ────────────────────
struct KleinLoraSet(Copyable, Movable):
    var dbl: List[LoraAdapter]   # num_double * DBL_SLOTS, slots 0-5 img, 6-11 txt
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


def _klein_resident_adapter(
    lo: LoraAdapter, state: LoraAdamWOTDeviceState, idx: Int,
) raises -> LoraAdapterDevice:
    var n_a = len(lo.a)
    var n_b = len(lo.b)
    var a_off = state.elem_offset(idx, False)
    var b_off = state.elem_offset(idx, True)
    return LoraAdapterDevice(
        TArc(Tensor(
            state.dev_p.create_sub_buffer[DType.uint8](a_off * 2, n_a * 2),
            [lo.rank, lo.in_f], STDtype.BF16,
        )),
        TArc(Tensor(
            state.dev_p.create_sub_buffer[DType.uint8](b_off * 2, n_b * 2),
            [lo.out_f, lo.rank], STDtype.BF16,
        )),
        lo.rank, lo.in_f, lo.out_f, lo.scale,
    )


def klein_lora_set_to_device_resident(
    set: KleinLoraSet,
    dbl_state: LoraAdamWOTDeviceState,
    sgl_state: LoraAdamWOTDeviceState,
    ctx: DeviceContext,
) raises -> KleinLoraDeviceSet:
    """v2 engine (resident-set): device LoRA set built ONCE, every adapter a
    zero-copy sub-buffer view into the persistent OT-AdamW parameter buffers
    (dbl/sgl). The in-place optimizer update IS the next step\'s weights; the
    per-step klein_lora_set_to_device upload disappears. States must outlive
    the returned set (scratch_ring view discipline)."""
    var dbl = List[LoraAdapterDevice]()
    var nd = set.num_double * DBL_SLOTS
    for i in range(nd):
        dbl.append(_klein_resident_adapter(set.dbl[i], dbl_state, i))
    var sgl = List[LoraAdapterDevice]()
    var ns = set.num_single * SGL_SLOTS
    for i in range(ns):
        sgl.append(_klein_resident_adapter(set.sgl[i], sgl_state, i))
    return KleinLoraDeviceSet(dbl^, sgl^, set.num_double, set.num_single, set.rank)


def klein_lora_adamw_step_resident(
    mut dbl_state: LoraAdamWOTDeviceState,
    mut sgl_state: LoraAdamWOTDeviceState,
    mut set: KleinLoraSet, grads: KleinLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    """klein_lora_adamw_step on the persistent device state — same scalars,
    same kernel, same SR stream (seed, intra-segment index): bit-identical."""
    if t < 1:
        raise Error("klein_lora_adamw_step_resident: t must be >= 1")
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    var step_size = lr / bc1
    var bc2_sqrt = sqrt(bc2)
    var decay = Float32(1.0) - lr * weight_decay
    var one_minus_beta1 = Float32(1.0) - beta1
    var one_minus_beta2 = Float32(1.0) - beta2
    var seed = UInt32(t)
    fused_lora_adamw_ot_step_resident(
        dbl_state, set.dbl, grads.dbl_d_a, grads.dbl_d_b,
        step_size, bc2_sqrt, decay, one_minus_beta1, beta2,
        one_minus_beta2, eps, seed, ctx,
    )
    fused_lora_adamw_ot_step_resident(
        sgl_state, set.sgl, grads.sgl_d_a, grads.sgl_d_b,
        step_size, bc2_sqrt, decay, one_minus_beta1, beta2,
        one_minus_beta2, eps, seed, ctx,
    )


def scale_klein_lora_set(mut set: KleinLoraSet, multiplier: Float32):
    """Apply a runtime LoRA multiplier by scaling adapter contribution only.

    This intentionally does not touch BF16 A/B storage or AdamW moments. The
    live Klein forward multiplies every LoRA contribution by `adapter.scale`.
    """
    if multiplier == Float32(1.0):
        return
    var nd = set.num_double * DBL_SLOTS
    for i in range(nd):
        set.dbl[i].scale *= multiplier
    var ns = set.num_single * SGL_SLOTS
    for i in range(ns):
        set.sgl[i].scale *= multiplier


# Accessor by (block_kind, block_idx, slot) → a COPY of the adapter. (LoraAdapter
# is Copyable; this is the read accessor the task asks for.)
def klein_lora_get(
    set: KleinLoraSet, block_kind: Int, block_idx: Int, slot: Int
) -> LoraAdapter:
    if block_kind == BK_DOUBLE:
        return set.dbl[block_idx * DBL_SLOTS + slot].copy()
    return set.sgl[block_idx * SGL_SLOTS + slot].copy()


# ── build the full LoRA set for a Klein stack ────────────────────────────────
# dims: D (model dim) and F (MLP hidden dim) for the projection in/out shapes:
#   double img/txt q,k,v,out: in=D out=D ; ff_in: in=D out=2F ; ff_out: in=F out=D.
#   single to_qkv_mlp: in=D out=3D+2F ; single to_out: in=D+F out=D.
# Each adapter gets a distinct seed so A is non-degenerate per slot.
def build_klein_lora_set(
    num_double: Int, num_single: Int, D: Int, F: Int, rank: Int, alpha: Float32
) -> KleinLoraSet:
    var dbl = List[LoraAdapter]()
    var seed = UInt64(1000)
    for _ in range(num_double):
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 0 img_q
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 1 img_k
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 2 img_v
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 3 img_out
        dbl.append(make_lora_adapter(rank, alpha, D, 2 * F, seed)); seed += 1    # 4 img_ff_in
        dbl.append(make_lora_adapter(rank, alpha, F, D, seed)); seed += 1        # 5 img_ff_out
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 6 txt_q
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 7 txt_k
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 8 txt_v
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 9 txt_out
        dbl.append(make_lora_adapter(rank, alpha, D, 2 * F, seed)); seed += 1    # 10 txt_ff_in
        dbl.append(make_lora_adapter(rank, alpha, F, D, seed)); seed += 1        # 11 txt_ff_out
    var sgl = List[LoraAdapter]()
    for _ in range(num_single):
        # slot 0 to_qkv_mlp on w1 (in=D,out=3D+2F)
        sgl.append(make_lora_adapter(rank, alpha, D, 3 * D + 2 * F, seed)); seed += 1
        # slot 1 to_out on w2 (in=D+F,out=D)
        sgl.append(make_lora_adapter(rank, alpha, D + F, D, seed)); seed += 1
    return KleinLoraSet(dbl^, sgl^, num_double, num_single, rank)


# build a transient DoubleBlockLora for block bi from the flat set.
def _dbl_lora_for(set: KleinLoraSet, bi: Int) -> DoubleBlockLora:
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
        Optional[LoraAdapterDevice](set.dbl[base + 2].copy()),
        Optional[LoraAdapterDevice](set.dbl[base + 3].copy()),
        Optional[LoraAdapterDevice](set.dbl[base + 4].copy()),
        Optional[LoraAdapterDevice](set.dbl[base + 5].copy()),
    )
    var txt = StreamLoraDevice(
        Optional[LoraAdapterDevice](set.dbl[base + 6].copy()),
        Optional[LoraAdapterDevice](set.dbl[base + 7].copy()),
        Optional[LoraAdapterDevice](set.dbl[base + 8].copy()),
        Optional[LoraAdapterDevice](set.dbl[base + 9].copy()),
        Optional[LoraAdapterDevice](set.dbl[base + 10].copy()),
        Optional[LoraAdapterDevice](set.dbl[base + 11].copy()),
    )
    return DoubleBlockLoraDevice(img^, txt^)


def _sgl_lora_dev_for(set: KleinLoraDeviceSet, bi: Int) -> SingleBlockLoraDevice:
    var base = bi * SGL_SLOTS
    return SingleBlockLoraDevice(
        Optional[LoraAdapterDevice](set.sgl[base + 0].copy()),
        Optional[LoraAdapterDevice](set.sgl[base + 1].copy()),
    )


def klein_double_lora_device_for(set: KleinLoraDeviceSet, bi: Int) -> DoubleBlockLoraDevice:
    return _dbl_lora_dev_for(set, bi)


def klein_single_lora_device_for(set: KleinLoraDeviceSet, bi: Int) -> SingleBlockLoraDevice:
    return _sgl_lora_dev_for(set, bi)


def _block_tensor_base(block: Block, key: String) raises -> TArc:
    if not (key in block):
        raise Error(String("Klein offload block missing tensor: ") + key)
    # Klein transformer-block matrices are BF16 on disk. Keep them in the
    # turbo-loader slab dtype and let mixed GEMM consume F32 activations with
    # BF16 weights, instead of promoting every streamed block to F32.
    return block[key].copy()


def _stream_weights_from_block(
    block: Block, prefix: String, stream: String, ctx: DeviceContext,
) raises -> StreamWeights:
    var ap = prefix + String(".") + stream + String("_attn")
    var mp = prefix + String(".") + stream + String("_mlp")
    return StreamWeights(
        _block_tensor_base(block, ap + String(".qkv.weight")),
        _block_tensor_base(block, ap + String(".proj.weight")),
        _block_tensor_base(block, mp + String(".0.weight")),
        _block_tensor_base(block, mp + String(".2.weight")),
        _block_tensor_base(block, ap + String(".norm.query_norm.scale")),
        _block_tensor_base(block, ap + String(".norm.key_norm.scale")),
    )


def _double_weights_from_block(
    block: Block, prefix: String, ctx: DeviceContext,
) raises -> DoubleBlockWeights:
    return DoubleBlockWeights(
        _stream_weights_from_block(block, prefix, String("img"), ctx),
        _stream_weights_from_block(block, prefix, String("txt"), ctx),
    )


def _single_weights_from_block(
    block: Block, prefix: String, D: Int, F: Int, ctx: DeviceContext,
) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _block_tensor_base(block, prefix + String(".linear1.weight")),
        _block_tensor_base(block, prefix + String(".linear2.weight")),
        _block_tensor_base(block, prefix + String(".norm.query_norm.scale")),
        _block_tensor_base(block, prefix + String(".norm.key_norm.scale")),
        D, F, ctx, False,
    )


def _single_weights_full_from_block(
    block: Block, prefix: String, D: Int, F: Int, ctx: DeviceContext,
) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _block_tensor_base(block, prefix + String(".linear1.weight")),
        _block_tensor_base(block, prefix + String(".linear2.weight")),
        _block_tensor_base(block, prefix + String(".norm.query_norm.scale")),
        _block_tensor_base(block, prefix + String(".norm.key_norm.scale")),
        D, F, ctx, True,
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


struct KleinLoraTensorGrads(Copyable, Movable):
    var dbl_d_a: List[TArc]
    var dbl_d_b: List[TArc]
    var sgl_d_a: List[TArc]
    var sgl_d_b: List[TArc]
    var d_img_tokens: List[Float32]
    var d_txt_tokens: List[Float32]
    var d_img_mod: List[Float32]
    var d_txt_mod: List[Float32]
    var d_single_mod: List[Float32]
    var d_img_in: List[Float32]
    var d_txt_in: List[Float32]
    var d_final_lin: List[Float32]
    var d_final_shift: List[Float32]
    var d_final_scale: List[Float32]

    def __init__(
        out self,
        var dbl_d_a: List[TArc], var dbl_d_b: List[TArc],
        var sgl_d_a: List[TArc], var sgl_d_b: List[TArc],
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


def _host_grad_slice_to_list(
    host: HostBuffer[DType.uint8], offset: Int, numel: Int
) -> List[Float32]:
    var out = List[Float32]()
    var fp = (host.unsafe_ptr() + offset).bitcast[Float32]()
    for i in range(numel):
        out.append(fp[i])
    return out^


def _grad_arc_f32(t: TArc, ctx: DeviceContext) raises -> TArc:
    if t[].dtype() == STDtype.F32:
        return t.copy()
    # Host AdamW stores master params and moments as F32; device grads may be BF16.
    var t32 = cast_tensor(t[], STDtype.F32, ctx)
    return TArc(t32^)


def _host_grad_group_to_lists(
    host: HostBuffer[DType.uint8],
    offsets: List[Int],
    numels: List[Int],
    start: Int,
    count: Int,
) -> List[List[Float32]]:
    var out = List[List[Float32]]()
    for i in range(count):
        var idx = start + i
        out.append(_host_grad_slice_to_list(host, offsets[idx], numels[idx]))
    return out^


def _required_tarc(opt: Optional[TArc], name: String) raises -> TArc:
    if opt:
        return opt.value().copy()
    raise Error(String("missing LoRA grad tensor: ") + name)


def klein_lora_tensor_grads_to_host(
    tg: KleinLoraTensorGrads, ctx: DeviceContext
) raises -> KleinLoraGrads:
    var dbl_a_f32 = List[TArc]()
    var dbl_b_f32 = List[TArc]()
    var sgl_a_f32 = List[TArc]()
    var sgl_b_f32 = List[TArc]()
    for i in range(len(tg.dbl_d_a)):
        dbl_a_f32.append(_grad_arc_f32(tg.dbl_d_a[i], ctx))
    for i in range(len(tg.dbl_d_b)):
        dbl_b_f32.append(_grad_arc_f32(tg.dbl_d_b[i], ctx))
    for i in range(len(tg.sgl_d_a)):
        sgl_a_f32.append(_grad_arc_f32(tg.sgl_d_a[i], ctx))
    for i in range(len(tg.sgl_d_b)):
        sgl_b_f32.append(_grad_arc_f32(tg.sgl_d_b[i], ctx))

    var total_bytes = 0
    for i in range(len(dbl_a_f32)):
        total_bytes += dbl_a_f32[i][].nbytes()
    for i in range(len(dbl_b_f32)):
        total_bytes += dbl_b_f32[i][].nbytes()
    for i in range(len(sgl_a_f32)):
        total_bytes += sgl_a_f32[i][].nbytes()
    for i in range(len(sgl_b_f32)):
        total_bytes += sgl_b_f32[i][].nbytes()

    var host = ctx.enqueue_create_host_buffer[DType.uint8](total_bytes)
    var offsets = List[Int]()
    var numels = List[Int]()
    var cursor = 0
    for i in range(len(dbl_a_f32)):
        offsets.append(cursor)
        numels.append(dbl_a_f32[i][].numel())
        var dst = host.create_sub_buffer[DType.uint8](cursor, dbl_a_f32[i][].nbytes())
        ctx.enqueue_copy(dst_buf=dst, src_buf=dbl_a_f32[i][].buf)
        cursor += dbl_a_f32[i][].nbytes()
    for i in range(len(dbl_b_f32)):
        offsets.append(cursor)
        numels.append(dbl_b_f32[i][].numel())
        var dst = host.create_sub_buffer[DType.uint8](cursor, dbl_b_f32[i][].nbytes())
        ctx.enqueue_copy(dst_buf=dst, src_buf=dbl_b_f32[i][].buf)
        cursor += dbl_b_f32[i][].nbytes()
    for i in range(len(sgl_a_f32)):
        offsets.append(cursor)
        numels.append(sgl_a_f32[i][].numel())
        var dst = host.create_sub_buffer[DType.uint8](cursor, sgl_a_f32[i][].nbytes())
        ctx.enqueue_copy(dst_buf=dst, src_buf=sgl_a_f32[i][].buf)
        cursor += sgl_a_f32[i][].nbytes()
    for i in range(len(sgl_b_f32)):
        offsets.append(cursor)
        numels.append(sgl_b_f32[i][].numel())
        var dst = host.create_sub_buffer[DType.uint8](cursor, sgl_b_f32[i][].nbytes())
        ctx.enqueue_copy(dst_buf=dst, src_buf=sgl_b_f32[i][].buf)
        cursor += sgl_b_f32[i][].nbytes()
    ctx.synchronize()

    var dbl_a_start = 0
    var dbl_b_start = dbl_a_start + len(tg.dbl_d_a)
    var sgl_a_start = dbl_b_start + len(tg.dbl_d_b)
    var sgl_b_start = sgl_a_start + len(tg.sgl_d_a)
    return KleinLoraGrads(
        _host_grad_group_to_lists(host, offsets, numels, dbl_a_start, len(tg.dbl_d_a)),
        _host_grad_group_to_lists(host, offsets, numels, dbl_b_start, len(tg.dbl_d_b)),
        _host_grad_group_to_lists(host, offsets, numels, sgl_a_start, len(tg.sgl_d_a)),
        _host_grad_group_to_lists(host, offsets, numels, sgl_b_start, len(tg.sgl_d_b)),
        tg.d_img_tokens.copy(), tg.d_txt_tokens.copy(),
        tg.d_img_mod.copy(), tg.d_txt_mod.copy(), tg.d_single_mod.copy(),
        tg.d_img_in.copy(), tg.d_txt_in.copy(), tg.d_final_lin.copy(),
        tg.d_final_shift.copy(), tg.d_final_scale.copy(),
    )


struct KleinStackDirectDoRAGradsT(Movable):
    var grads: FlatDirectDoRAGrads
    var d_img_tokens: List[Float32]
    var d_txt_tokens: List[Float32]

    def __init__(
        out self, var grads: FlatDirectDoRAGrads,
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
    ):
        self.grads = grads^
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^


struct KleinStackDirectOFTGradsT(Movable):
    var grads: FlatDirectOFTGrads
    var d_img_tokens: List[Float32]
    var d_txt_tokens: List[Float32]

    def __init__(
        out self, var grads: FlatDirectOFTGrads,
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
    ):
        self.grads = grads^
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^


def _klein_direct_slots_per_double(targets: Int) raises -> Int:
    var n = 0
    for slot in range(DBL_SLOTS):
        if _slot_targeted(True, slot, targets):
            n += 1
    return n


def _klein_direct_slots_per_single(targets: Int) raises -> Int:
    var n = 0
    for slot in range(SGL_SLOTS):
        if _slot_targeted(False, slot, targets):
            n += 1
    return n


def _klein_direct_double_base(bi: Int, targets: Int) raises -> Int:
    return bi * _klein_direct_slots_per_double(targets)


def _klein_direct_single_base(num_double: Int, bi: Int, targets: Int) raises -> Int:
    return (
        num_double * _klein_direct_slots_per_double(targets)
        + bi * _klein_direct_slots_per_single(targets)
    )


def _klein_direct_expected_slots(
    num_double: Int, num_single: Int, targets: Int,
) raises -> Int:
    return (
        num_double * _klein_direct_slots_per_double(targets)
        + num_single * _klein_direct_slots_per_single(targets)
    )


def _dora_grad_to_host(g: KleinDirectDoRAGradT, ctx: DeviceContext) raises -> DoRAGrads:
    if not g.d_a or not g.d_b or not g.d_m:
        raise Error("_dora_grad_to_host: missing direct DoRA grad tensor")
    return DoRAGrads(
        g.d_a.value()[].to_host(ctx),
        g.d_b.value()[].to_host(ctx),
        g.d_m.value()[].to_host(ctx),
        List[Float32](),
    )


def _oft_grad_to_host(g: KleinDirectOFTGradT, ctx: DeviceContext) raises -> OFTOTGrads:
    if not g.d_vec:
        raise Error("_oft_grad_to_host: missing direct OFT grad tensor")
    return OFTOTGrads(g.d_vec.value()[].to_host(ctx), List[Float32]())


def _klein_dora_stream_grad(g: StreamDirectDoRAGradsT, slot: Int) -> KleinDirectDoRAGradT:
    if slot == 0:
        return g.q.copy()
    if slot == 1:
        return g.k.copy()
    if slot == 2:
        return g.v.copy()
    if slot == 3:
        return g.out.copy()
    if slot == 4:
        return g.ff_in.copy()
    return g.ff_out.copy()


def _klein_oft_stream_grad(g: StreamDirectOFTGradsT, slot: Int) -> KleinDirectOFTGradT:
    if slot == 0:
        return g.q.copy()
    if slot == 1:
        return g.k.copy()
    if slot == 2:
        return g.v.copy()
    if slot == 3:
        return g.out.copy()
    if slot == 4:
        return g.ff_in.copy()
    return g.ff_out.copy()


def _scatter_klein_dora_double(
    mut grads: FlatDirectDoRAGrads, targets: Int, bi: Int,
    bg: DoubleBlockDirectDoRAGradsT, ctx: DeviceContext,
) raises:
    var compact = _klein_direct_double_base(bi, targets)
    for slot in range(DBL_SLOTS):
        if not _slot_targeted(True, slot, targets):
            continue
        if slot < 6:
            var hg = _dora_grad_to_host(_klein_dora_stream_grad(bg.img, slot), ctx)
            klein_direct_dora_scatter_slot_grad(grads, compact, hg)
        else:
            var hg = _dora_grad_to_host(_klein_dora_stream_grad(bg.txt, slot - 6), ctx)
            klein_direct_dora_scatter_slot_grad(grads, compact, hg)
        compact += 1
    if compact != _klein_direct_double_base(bi + 1, targets):
        raise Error("_scatter_klein_dora_double: compact slot mismatch")


def _scatter_klein_oft_double(
    mut grads: FlatDirectOFTGrads, targets: Int, bi: Int,
    bg: DoubleBlockDirectOFTGradsT, ctx: DeviceContext,
) raises:
    var compact = _klein_direct_double_base(bi, targets)
    for slot in range(DBL_SLOTS):
        if not _slot_targeted(True, slot, targets):
            continue
        if slot < 6:
            var hg = _oft_grad_to_host(_klein_oft_stream_grad(bg.img, slot), ctx)
            klein_direct_oft_scatter_slot_grad(grads, compact, hg)
        else:
            var hg = _oft_grad_to_host(_klein_oft_stream_grad(bg.txt, slot - 6), ctx)
            klein_direct_oft_scatter_slot_grad(grads, compact, hg)
        compact += 1
    if compact != _klein_direct_double_base(bi + 1, targets):
        raise Error("_scatter_klein_oft_double: compact slot mismatch")


def _scatter_klein_dora_single(
    mut grads: FlatDirectDoRAGrads, targets: Int, num_double: Int, bi: Int,
    bg: SingleBlockDirectDoRAGradsT, ctx: DeviceContext,
) raises:
    var compact = _klein_direct_single_base(num_double, bi, targets)
    if _slot_targeted(False, 0, targets):
        var hg0 = _dora_grad_to_host(bg.qkv, ctx)
        klein_direct_dora_scatter_slot_grad(grads, compact, hg0)
        compact += 1
    if _slot_targeted(False, 1, targets):
        var hg1 = _dora_grad_to_host(bg.out, ctx)
        klein_direct_dora_scatter_slot_grad(grads, compact, hg1)
        compact += 1
    if compact != _klein_direct_single_base(num_double, bi + 1, targets):
        raise Error("_scatter_klein_dora_single: compact slot mismatch")


def _scatter_klein_oft_single(
    mut grads: FlatDirectOFTGrads, targets: Int, num_double: Int, bi: Int,
    bg: SingleBlockDirectOFTGradsT, ctx: DeviceContext,
) raises:
    var compact = _klein_direct_single_base(num_double, bi, targets)
    if _slot_targeted(False, 0, targets):
        var hg0 = _oft_grad_to_host(bg.qkv, ctx)
        klein_direct_oft_scatter_slot_grad(grads, compact, hg0)
        compact += 1
    if _slot_targeted(False, 1, targets):
        var hg1 = _oft_grad_to_host(bg.out, ctx)
        klein_direct_oft_scatter_slot_grad(grads, compact, hg1)
        compact += 1
    if compact != _klein_direct_single_base(num_double, bi + 1, targets):
        raise Error("_scatter_klein_oft_single: compact slot mismatch")


def klein_stack_direct_dora_forward_offload_turbo_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    dora: FlatDirectDoRASet,
    num_double: Int, num_single: Int, targets: Int,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> KleinStackForward:
    if len(dora.ad) != _klein_direct_expected_slots(num_double, num_single, targets):
        raise Error("klein_stack_direct_dora_forward: direct slot count mismatch")
    var dora_dev = klein_direct_dora_blocks_to_device(
        dora, num_double, num_single, targets, ctx,
    )
    loader.prefetch_with_ctx(0, ctx)

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
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var fwd = double_block_direct_dora_forward_device_resident_scratch[
            H, Dh, N_IMG, N_TXT, S
        ](
            img, txt, w, img_mod_dev, txt_mod_dev, dora_dev.dbl[bi],
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        if bi >= num_double - DBL_SAVE_TAIL:
            dbl_saved.append(fwd.saved.copy())
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)
    var x = TArc(concat(0, ctx, txt[], img[]))

    var sgl_x_in = List[TArc]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        sgl_x_in.append(x.copy())
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w: SingleBlockWeights
        if targets == LOKR_TGT_ALL:
            w = _single_weights_full_from_block(handle.block, handle.prefix, D, F, ctx)
        else:
            w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var fwd = single_block_direct_dora_forward_device_resident_scratch[H, Dh, S](
            x, w, single_mod_dev, dora_dev.sgl[bi], cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        if bi >= num_single - SGL_SAVE_TAIL:
            sgl_saved.append(fwd.saved.copy())
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)

    var img_out = TArc(slice(x[], 0, N_TXT, N_IMG, ctx))
    var ln_img_out = TArc(layer_norm(img_out[], norm_ones[], norm_zeros[], eps, ctx))
    var normed = modulate(ln_img_out[], base.final_scale[], base.final_shift[], ctx)
    var no_bias_out = Optional[Tensor](None)
    var out = linear(normed, base.final_lin[], no_bias_out^, ctx).to_host(ctx)

    return KleinStackForward(
        out^, img_in_act^, txt_in_act^,
        dbl_img_in^, dbl_txt_in^, sgl_x_in^,
        dbl_saved^, sgl_saved^,
        img_out^, ln_img_out^,
    )


def klein_stack_direct_oft_forward_offload_turbo_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    oft: FlatDirectOFTSet,
    num_double: Int, num_single: Int, targets: Int,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> KleinStackForward:
    if len(oft.ad) != _klein_direct_expected_slots(num_double, num_single, targets):
        raise Error("klein_stack_direct_oft_forward: direct slot count mismatch")
    var oft_dev = klein_direct_oft_blocks_to_device(
        oft, num_double, num_single, targets, ctx,
    )
    loader.prefetch_with_ctx(0, ctx)

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
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var fwd = double_block_direct_oft_forward_device_resident_scratch[
            H, Dh, N_IMG, N_TXT, S
        ](
            img, txt, w, img_mod_dev, txt_mod_dev, oft_dev.dbl[bi],
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        if bi >= num_double - DBL_SAVE_TAIL:
            dbl_saved.append(fwd.saved.copy())
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)
    var x = TArc(concat(0, ctx, txt[], img[]))

    var sgl_x_in = List[TArc]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        sgl_x_in.append(x.copy())
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w: SingleBlockWeights
        if targets == LOKR_TGT_ALL:
            w = _single_weights_full_from_block(handle.block, handle.prefix, D, F, ctx)
        else:
            w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var fwd = single_block_direct_oft_forward_device_resident_scratch[H, Dh, S](
            x, w, single_mod_dev, oft_dev.sgl[bi], cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        if bi >= num_single - SGL_SAVE_TAIL:
            sgl_saved.append(fwd.saved.copy())
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)

    var img_out = TArc(slice(x[], 0, N_TXT, N_IMG, ctx))
    var ln_img_out = TArc(layer_norm(img_out[], norm_ones[], norm_zeros[], eps, ctx))
    var normed = modulate(ln_img_out[], base.final_scale[], base.final_shift[], ctx)
    var no_bias_out = Optional[Tensor](None)
    var out = linear(normed, base.final_lin[], no_bias_out^, ctx).to_host(ctx)

    return KleinStackForward(
        out^, img_in_act^, txt_in_act^,
        dbl_img_in^, dbl_txt_in^, sgl_x_in^,
        dbl_saved^, sgl_saved^,
        img_out^, ln_img_out^,
    )


def klein_stack_direct_dora_backward_offload_turbo_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: TArc, txt_tokens: TArc,
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    dora: FlatDirectDoRASet,
    num_double: Int, num_single: Int, targets: Int,
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
) raises -> KleinStackDirectDoRAGradsT:
    if len(dora.ad) != _klein_direct_expected_slots(num_double, num_single, targets):
        raise Error("klein_stack_direct_dora_backward: direct slot count mismatch")
    var dora_dev = klein_direct_dora_blocks_to_device(
        dora, num_double, num_single, targets, ctx,
    )
    var direct_grads = klein_direct_dora_zero_grads(dora)
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var d_normed_t = linear_backward_dx(
        _t(d_out, [N_IMG, out_ch], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )
    var mbf = modulate_backward(
        d_normed_t, saved.ln_img_out[], base.final_scale[], ctx, compute_aux_grads,
    )
    var d_img_out_t = layer_norm_backward_dx(
        mbf.d_x, saved.img_out[], norm_ones[], eps, ctx,
    )
    var d_txt_zero = zeros_device([N_TXT, D], STDtype.F32, ctx)
    var d_x = TArc(concat(0, ctx, d_txt_zero, d_img_out_t))

    var bi = num_single - 1
    var saved_single_start = num_single - len(saved.sgl_saved)
    while bi >= 0:
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)
        var w: SingleBlockWeights
        if targets == LOKR_TGT_ALL:
            w = _single_weights_full_from_block(handle.block, handle.prefix, D, F, ctx)
        else:
            w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var block_saved: SingleBlockSaved
        if bi >= saved_single_start:
            block_saved = saved.sgl_saved[bi - saved_single_start].copy()
        else:
            var fwd = single_block_direct_dora_forward_device_resident_scratch[H, Dh, S](
                saved.sgl_x_in[bi], w, single_mod_dev, dora_dev.sgl[bi],
                cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
            )
            block_saved = fwd.saved.copy()
        var bg = single_block_direct_dora_backward_device_resident_scratch[H, Dh, S](
            d_x, w, single_mod_dev, dora_dev.sgl[bi], block_saved,
            cos_t, sin_t, D, F, eps, norm_ones[], ctx, scratch, compute_aux_grads,
        )
        d_x = bg.d_x.copy()
        _scatter_klein_dora_single(direct_grads, targets, num_double, bi, bg, ctx)
        loader.mark_active_block_done(ctx)
        bi -= 1

    var d_txt_out = TArc(slice(d_x[], 0, 0, N_TXT, ctx))
    var d_img_out2 = TArc(slice(d_x[], 0, N_TXT, N_IMG, ctx))
    var di = num_double - 1
    var saved_double_start = num_double - len(saved.dbl_saved)
    var d_io = d_img_out2.copy()
    var d_to = d_txt_out.copy()
    while di >= 0:
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var block_saved: DoubleBlockSaved
        if di >= saved_double_start:
            block_saved = saved.dbl_saved[di - saved_double_start].copy()
        else:
            var fwd = double_block_direct_dora_forward_device_resident_scratch[
                H, Dh, N_IMG, N_TXT, S
            ](
                saved.dbl_img_in[di], saved.dbl_txt_in[di],
                w, img_mod_dev, txt_mod_dev, dora_dev.dbl[di],
                cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
            )
            block_saved = fwd.saved.copy()
        var bg = double_block_direct_dora_backward_device_resident_scratch[
            H, Dh, N_IMG, N_TXT, S
        ](
            d_io, d_to, w, img_mod_dev, txt_mod_dev, dora_dev.dbl[di], block_saved,
            cos_t, sin_t, D, F, eps, norm_ones[], ctx, scratch, compute_aux_grads,
        )
        d_io = bg.img.d_x.copy()
        d_to = bg.txt.d_x.copy()
        _scatter_klein_dora_double(direct_grads, targets, di, bg, ctx)
        loader.mark_active_block_done(ctx)
        di -= 1

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

    return KleinStackDirectDoRAGradsT(direct_grads^, d_img_tokens^, d_txt_tokens^)


def klein_stack_direct_oft_backward_offload_turbo_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: TArc, txt_tokens: TArc,
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    oft: FlatDirectOFTSet,
    num_double: Int, num_single: Int, targets: Int,
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
) raises -> KleinStackDirectOFTGradsT:
    if len(oft.ad) != _klein_direct_expected_slots(num_double, num_single, targets):
        raise Error("klein_stack_direct_oft_backward: direct slot count mismatch")
    var oft_dev = klein_direct_oft_blocks_to_device(
        oft, num_double, num_single, targets, ctx,
    )
    var direct_grads = klein_direct_oft_zero_grads(oft)
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var d_normed_t = linear_backward_dx(
        _t(d_out, [N_IMG, out_ch], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )
    var mbf = modulate_backward(
        d_normed_t, saved.ln_img_out[], base.final_scale[], ctx, compute_aux_grads,
    )
    var d_img_out_t = layer_norm_backward_dx(
        mbf.d_x, saved.img_out[], norm_ones[], eps, ctx,
    )
    var d_txt_zero = zeros_device([N_TXT, D], STDtype.F32, ctx)
    var d_x = TArc(concat(0, ctx, d_txt_zero, d_img_out_t))

    var bi = num_single - 1
    var saved_single_start = num_single - len(saved.sgl_saved)
    while bi >= 0:
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)
        var w: SingleBlockWeights
        if targets == LOKR_TGT_ALL:
            w = _single_weights_full_from_block(handle.block, handle.prefix, D, F, ctx)
        else:
            w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var block_saved: SingleBlockSaved
        if bi >= saved_single_start:
            block_saved = saved.sgl_saved[bi - saved_single_start].copy()
        else:
            var fwd = single_block_direct_oft_forward_device_resident_scratch[H, Dh, S](
                saved.sgl_x_in[bi], w, single_mod_dev, oft_dev.sgl[bi],
                cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
            )
            block_saved = fwd.saved.copy()
        var bg = single_block_direct_oft_backward_device_resident_scratch[H, Dh, S](
            d_x, w, single_mod_dev, oft_dev.sgl[bi], block_saved,
            cos_t, sin_t, D, F, eps, norm_ones[], ctx, scratch, compute_aux_grads,
        )
        d_x = bg.d_x.copy()
        _scatter_klein_oft_single(direct_grads, targets, num_double, bi, bg, ctx)
        loader.mark_active_block_done(ctx)
        bi -= 1

    var d_txt_out = TArc(slice(d_x[], 0, 0, N_TXT, ctx))
    var d_img_out2 = TArc(slice(d_x[], 0, N_TXT, N_IMG, ctx))
    var di = num_double - 1
    var saved_double_start = num_double - len(saved.dbl_saved)
    var d_io = d_img_out2.copy()
    var d_to = d_txt_out.copy()
    while di >= 0:
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var block_saved: DoubleBlockSaved
        if di >= saved_double_start:
            block_saved = saved.dbl_saved[di - saved_double_start].copy()
        else:
            var fwd = double_block_direct_oft_forward_device_resident_scratch[
                H, Dh, N_IMG, N_TXT, S
            ](
                saved.dbl_img_in[di], saved.dbl_txt_in[di],
                w, img_mod_dev, txt_mod_dev, oft_dev.dbl[di],
                cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
            )
            block_saved = fwd.saved.copy()
        var bg = double_block_direct_oft_backward_device_resident_scratch[
            H, Dh, N_IMG, N_TXT, S
        ](
            d_io, d_to, w, img_mod_dev, txt_mod_dev, oft_dev.dbl[di], block_saved,
            cos_t, sin_t, D, F, eps, norm_ones[], ctx, scratch, compute_aux_grads,
        )
        d_io = bg.img.d_x.copy()
        d_to = bg.txt.d_x.copy()
        _scatter_klein_oft_double(direct_grads, targets, di, bg, ctx)
        loader.mark_active_block_done(ctx)
        di -= 1

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

    return KleinStackDirectOFTGradsT(direct_grads^, d_img_tokens^, d_txt_tokens^)


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


def klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> KleinStackForward:
    var num_double = lora.num_double
    var num_single = lora.num_single

    loader.prefetch_with_ctx(0, ctx)

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
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var bl = _dbl_lora_dev_for(lora, bi)
        var fwd = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            img, txt, w, img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        if bi >= num_double - DBL_SAVE_TAIL:
            dbl_saved.append(fwd.saved.copy())
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)

    var x = TArc(concat(0, ctx, txt[], img[]))

    var sgl_x_in = List[TArc]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        sgl_x_in.append(x.copy())
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var sl = _sgl_lora_dev_for(lora, bi)
        var fwd = single_block_lora_forward_device_resident_scratch[H, Dh, S](
            x, w, single_mod_dev, sl, cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        if bi >= num_single - SGL_SAVE_TAIL:
            sgl_saved.append(fwd.saved.copy())
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)

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


def klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_offloaded_tape[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> KleinStackLoraOffloadedTape:
    """Forward for OneTrainer CPU_OFFLOADED-style Klein LoRA training.

    Base weights still stream through TurboPlannedLoader. Unlike the normal
    streamed forward, this path parks the checkpoint/backward inputs in
    HostOffload raw-byte carriers as soon as each block boundary is reached.
    BF16/F16 storage stays BF16/F16; only compute internals use F32.
    """
    var num_double = lora.num_double
    var num_single = lora.num_single

    loader.prefetch_with_ctx(0, ctx)

    var no_bias = Optional[Tensor](None)
    var img = TArc(linear(img_tokens_t[], base.img_in[], no_bias^, ctx))
    var no_bias_txt = Optional[Tensor](None)
    var txt = TArc(linear(txt_tokens_t[], base.txt_in[], no_bias_txt^, ctx))
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    var dbl_img_in = List[HostOffload]()
    var dbl_txt_in = List[HostOffload]()
    for bi in range(num_double):
        var off_img = offload_to_host(img[], ctx)
        dbl_img_in.append(off_img^)
        var off_txt = offload_to_host(txt[], ctx)
        dbl_txt_in.append(off_txt^)
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var bl = _dbl_lora_dev_for(lora, bi)
        var fwd = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            img, txt, w, img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)
    var x = TArc(concat(0, ctx, txt[], img[]))

    var sgl_x_in = List[HostOffload]()
    for bi in range(num_single):
        var off_x = offload_to_host(x[], ctx)
        sgl_x_in.append(off_x^)
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var sl = _sgl_lora_dev_for(lora, bi)
        var fwd = single_block_lora_forward_device_resident_scratch[H, Dh, S](
            x, w, single_mod_dev, sl, cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)

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
    var off_img_out = offload_to_host(img_out[], ctx)
    var off_ln_img_out = offload_to_host(ln_img_out[], ctx)

    return KleinStackLoraOffloadedTape(
        out^,
        dbl_img_in^,
        dbl_txt_in^,
        sgl_x_in^,
        off_img_out^,
        off_ln_img_out^,
    )


def klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> Tensor:
    """Inference-only Klein LoRA forward.

    The training forward returns a KleinStackForward tape with per-block inputs
    and saved activations for backward. Validation sampling must not keep that
    tape, especially at 1024 where S=4608. This path runs the same math but only
    returns the final velocity prediction.
    """
    var num_double = lora.num_double
    var num_single = lora.num_single

    loader.prefetch_with_ctx(0, ctx)

    var no_bias = Optional[Tensor](None)
    var img = TArc(linear(img_tokens_t[], base.img_in[], no_bias^, ctx))
    var no_bias_txt = Optional[Tensor](None)
    var txt = TArc(linear(txt_tokens_t[], base.txt_in[], no_bias_txt^, ctx))
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    for bi in range(num_double):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var bl = _dbl_lora_dev_for(lora, bi)
        var fwd = double_block_lora_predict_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            img, txt, w, img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)

    var x = TArc(concat(0, ctx, txt[], img[]))

    for bi in range(num_single):
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var sl = _sgl_lora_dev_for(lora, bi)
        var fwd = single_block_lora_predict_device_resident_scratch[H, Dh, S](
            x, w, single_mod_dev, sl, cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)

    var img_out = TArc(slice(x[], 0, N_TXT, N_IMG, ctx))
    var ln_img_out = TArc(layer_norm(
        img_out[], norm_ones[], norm_zeros[], eps, ctx,
    ))
    var normed = modulate(
        ln_img_out[],
        base.final_scale[], base.final_shift[], ctx,
    )
    var no_bias_out = Optional[Tensor](None)
    var out = linear(normed, base.final_lin[], no_bias_out^, ctx)
    return out^


def klein_stack_lora_predict_offload_turbo_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> List[Float32]:
    var out = klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch[
        H, Dh, N_IMG, N_TXT, S
    ](
        img_tokens_t, txt_tokens_t, base, loader, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t,
        D, F, in_ch, txt_ch, out_ch, eps, ctx, scratch,
    )
    return out.to_host(ctx)


def klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc,
    txt_pos_tokens_t: TArc,
    txt_neg_tokens_t: TArc,
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> KleinLoraCfgPreds:
    """CFG-paired inference-only Klein LoRA forward.

    Validation sampling needs both positive and negative predictions at each
    denoise step. The naive path ran the whole offloaded stack twice. This path
    keeps each streamed block resident, runs positive and negative branches
    through it, records compute completion once both branches are queued, and
    returns device-resident predictions so CFG stays on GPU.
    """
    var num_double = lora.num_double
    var num_single = lora.num_single

    loader.prefetch_with_ctx(0, ctx)

    var no_bias_pos = Optional[Tensor](None)
    var img_pos = TArc(linear(img_tokens_t[], base.img_in[], no_bias_pos^, ctx))
    var no_bias_neg = Optional[Tensor](None)
    var img_neg = TArc(linear(img_tokens_t[], base.img_in[], no_bias_neg^, ctx))
    var no_bias_txt_pos = Optional[Tensor](None)
    var txt_pos = TArc(linear(txt_pos_tokens_t[], base.txt_in[], no_bias_txt_pos^, ctx))
    var no_bias_txt_neg = Optional[Tensor](None)
    var txt_neg = TArc(linear(txt_neg_tokens_t[], base.txt_in[], no_bias_txt_neg^, ctx))
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    for bi in range(num_double):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var bl = _dbl_lora_dev_for(lora, bi)
        var fwd_pos = double_block_lora_predict_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            img_pos, txt_pos, w, img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        ctx.synchronize()
        var fwd_neg = double_block_lora_predict_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            img_neg, txt_neg, w, img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        img_pos = fwd_pos.img_out.copy()
        txt_pos = fwd_pos.txt_out.copy()
        img_neg = fwd_neg.img_out.copy()
        txt_neg = fwd_neg.txt_out.copy()
        loader.mark_active_block_done(ctx)
        ctx.synchronize()

    var x_pos = TArc(concat(0, ctx, txt_pos[], img_pos[]))
    var x_neg = TArc(concat(0, ctx, txt_neg[], img_neg[]))

    for bi in range(num_single):
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var sl = _sgl_lora_dev_for(lora, bi)
        var fwd_pos = single_block_lora_predict_device_resident_scratch[H, Dh, S](
            x_pos, w, single_mod_dev, sl, cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        ctx.synchronize()
        var fwd_neg = single_block_lora_predict_device_resident_scratch[H, Dh, S](
            x_neg, w, single_mod_dev, sl, cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        x_pos = fwd_pos.out.copy()
        x_neg = fwd_neg.out.copy()
        loader.mark_active_block_done(ctx)
        ctx.synchronize()

    var img_out_pos = TArc(slice(x_pos[], 0, N_TXT, N_IMG, ctx))
    var img_out_neg = TArc(slice(x_neg[], 0, N_TXT, N_IMG, ctx))
    var ln_img_out_pos = layer_norm(
        img_out_pos[], norm_ones[], norm_zeros[], eps, ctx,
    )
    var ln_img_out_neg = layer_norm(
        img_out_neg[], norm_ones[], norm_zeros[], eps, ctx,
    )
    var normed_pos = modulate(
        ln_img_out_pos, base.final_scale[], base.final_shift[], ctx,
    )
    var normed_neg = modulate(
        ln_img_out_neg, base.final_scale[], base.final_shift[], ctx,
    )
    var no_bias_out_pos = Optional[Tensor](None)
    var pred_pos = linear(normed_pos, base.final_lin[], no_bias_out_pos^, ctx)
    var no_bias_out_neg = Optional[Tensor](None)
    var pred_neg = linear(normed_neg, base.final_lin[], no_bias_out_neg^, ctx)
    return KleinLoraCfgPreds(pred_pos^, pred_neg^)


def klein_stack_lora_predict_offload_turbo_hostlora_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora: KleinLoraSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> List[Float32]:
    """Inference-only offload path that streams LoRA adapters per block.

    1024 validation with CFG can be close to 24GB when all 80 LoRA adapters are
    resident. This variant keeps the PEFT LoRA host-side and uploads only the
    current block's adapters, matching the block-swapped base-weight lifetime.
    """
    var num_double = lora.num_double
    var num_single = lora.num_single

    var no_bias = Optional[Tensor](None)
    var img = TArc(linear(img_tokens_t[], base.img_in[], no_bias^, ctx))
    var no_bias_txt = Optional[Tensor](None)
    var txt = TArc(linear(txt_tokens_t[], base.txt_in[], no_bias_txt^, ctx))
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    loader.prefetch_with_ctx(0, ctx)
    for bi in range(num_double):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var bl_host = _dbl_lora_for(lora, bi)
        var bl = double_block_lora_to_device(bl_host, ctx)
        var fwd = double_block_lora_predict_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            img, txt, w, img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)
        ctx.synchronize()

    var x = TArc(concat(0, ctx, txt[], img[]))

    for bi in range(num_single):
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var sl_host = _sgl_lora_for(lora, bi)
        var sl = single_block_lora_to_device(sl_host, ctx)
        var fwd = single_block_lora_predict_device_resident_scratch[H, Dh, S](
            x, w, single_mod_dev, sl, cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)
        ctx.synchronize()

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
    return out^


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

    # flat double LoRA grads (slots 0-5 img q,k,v,out,ff_in,ff_out; 6-11 txt).
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
        # scatter into the flat slots (0-5 img, 6-11 txt; order q,k,v,out,ff_in,ff_out).
        var dbase = di * DBL_SLOTS
        dbl_d_a[dbase + 0] = bg.img.q_d_a.copy()
        dbl_d_b[dbase + 0] = bg.img.q_d_b.copy()
        dbl_d_a[dbase + 1] = bg.img.k_d_a.copy()
        dbl_d_b[dbase + 1] = bg.img.k_d_b.copy()
        dbl_d_a[dbase + 2] = bg.img.v_d_a.copy()
        dbl_d_b[dbase + 2] = bg.img.v_d_b.copy()
        dbl_d_a[dbase + 3] = bg.img.out_d_a.copy()
        dbl_d_b[dbase + 3] = bg.img.out_d_b.copy()
        dbl_d_a[dbase + 4] = bg.img.ff_in_d_a.copy()
        dbl_d_b[dbase + 4] = bg.img.ff_in_d_b.copy()
        dbl_d_a[dbase + 5] = bg.img.ff_out_d_a.copy()
        dbl_d_b[dbase + 5] = bg.img.ff_out_d_b.copy()
        dbl_d_a[dbase + 6] = bg.txt.q_d_a.copy()
        dbl_d_b[dbase + 6] = bg.txt.q_d_b.copy()
        dbl_d_a[dbase + 7] = bg.txt.k_d_a.copy()
        dbl_d_b[dbase + 7] = bg.txt.k_d_b.copy()
        dbl_d_a[dbase + 8] = bg.txt.v_d_a.copy()
        dbl_d_b[dbase + 8] = bg.txt.v_d_b.copy()
        dbl_d_a[dbase + 9] = bg.txt.out_d_a.copy()
        dbl_d_b[dbase + 9] = bg.txt.out_d_b.copy()
        dbl_d_a[dbase + 10] = bg.txt.ff_in_d_a.copy()
        dbl_d_b[dbase + 10] = bg.txt.ff_in_d_b.copy()
        dbl_d_a[dbase + 11] = bg.txt.ff_out_d_a.copy()
        dbl_d_b[dbase + 11] = bg.txt.ff_out_d_b.copy()
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
        dbl_d_a[dbase + 0] = bg.img.q_d_a.copy()
        dbl_d_b[dbase + 0] = bg.img.q_d_b.copy()
        dbl_d_a[dbase + 1] = bg.img.k_d_a.copy()
        dbl_d_b[dbase + 1] = bg.img.k_d_b.copy()
        dbl_d_a[dbase + 2] = bg.img.v_d_a.copy()
        dbl_d_b[dbase + 2] = bg.img.v_d_b.copy()
        dbl_d_a[dbase + 3] = bg.img.out_d_a.copy()
        dbl_d_b[dbase + 3] = bg.img.out_d_b.copy()
        dbl_d_a[dbase + 4] = bg.img.ff_in_d_a.copy()
        dbl_d_b[dbase + 4] = bg.img.ff_in_d_b.copy()
        dbl_d_a[dbase + 5] = bg.img.ff_out_d_a.copy()
        dbl_d_b[dbase + 5] = bg.img.ff_out_d_b.copy()
        dbl_d_a[dbase + 6] = bg.txt.q_d_a.copy()
        dbl_d_b[dbase + 6] = bg.txt.q_d_b.copy()
        dbl_d_a[dbase + 7] = bg.txt.k_d_a.copy()
        dbl_d_b[dbase + 7] = bg.txt.k_d_b.copy()
        dbl_d_a[dbase + 8] = bg.txt.v_d_a.copy()
        dbl_d_b[dbase + 8] = bg.txt.v_d_b.copy()
        dbl_d_a[dbase + 9] = bg.txt.out_d_a.copy()
        dbl_d_b[dbase + 9] = bg.txt.out_d_b.copy()
        dbl_d_a[dbase + 10] = bg.txt.ff_in_d_a.copy()
        dbl_d_b[dbase + 10] = bg.txt.ff_in_d_b.copy()
        dbl_d_a[dbase + 11] = bg.txt.ff_out_d_a.copy()
        dbl_d_b[dbase + 11] = bg.txt.ff_out_d_b.copy()
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


def klein_stack_lora_backward_offload_turbo_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
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
    var num_double = lora.num_double
    var num_single = lora.num_single
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

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
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var sl = _sgl_lora_dev_for(lora, bi)
        var block_saved: SingleBlockSaved
        if bi >= saved_single_start:
            block_saved = saved.sgl_saved[bi - saved_single_start].copy()
        else:
            block_saved = single_block_lora_recompute_saved_device_resident_scratch[H, Dh, S](
                saved.sgl_x_in[bi], w, single_mod_dev, sl,
                cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
            )
        var bg = single_block_lora_backward_device_resident_scratch[H, Dh, S](
            d_x, w, single_mod_dev, sl, block_saved, cos_t, sin_t,
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
        loader.mark_active_block_done(ctx)
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
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var bl = _dbl_lora_dev_for(lora, di)
        var dbl_block_saved: DoubleBlockSaved
        var saved_double_start = num_double - len(saved.dbl_saved)
        if di >= saved_double_start:
            dbl_block_saved = saved.dbl_saved[di - saved_double_start].copy()
        else:
            var fwd = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
                saved.dbl_img_in[di], saved.dbl_txt_in[di],
                w, img_mod_dev, txt_mod_dev, bl, cos_t, sin_t, D, F, eps,
                norm_ones[], norm_zeros[], ctx, scratch,
            )
            dbl_block_saved = fwd.saved.copy()
        var bg = double_block_lora_backward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            d_io, d_to, w, img_mod_dev, txt_mod_dev, bl, dbl_block_saved,
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
        dbl_d_a[dbase + 0] = bg.img.q_d_a.copy()
        dbl_d_b[dbase + 0] = bg.img.q_d_b.copy()
        dbl_d_a[dbase + 1] = bg.img.k_d_a.copy()
        dbl_d_b[dbase + 1] = bg.img.k_d_b.copy()
        dbl_d_a[dbase + 2] = bg.img.v_d_a.copy()
        dbl_d_b[dbase + 2] = bg.img.v_d_b.copy()
        dbl_d_a[dbase + 3] = bg.img.out_d_a.copy()
        dbl_d_b[dbase + 3] = bg.img.out_d_b.copy()
        dbl_d_a[dbase + 4] = bg.img.ff_in_d_a.copy()
        dbl_d_b[dbase + 4] = bg.img.ff_in_d_b.copy()
        dbl_d_a[dbase + 5] = bg.img.ff_out_d_a.copy()
        dbl_d_b[dbase + 5] = bg.img.ff_out_d_b.copy()
        dbl_d_a[dbase + 6] = bg.txt.q_d_a.copy()
        dbl_d_b[dbase + 6] = bg.txt.q_d_b.copy()
        dbl_d_a[dbase + 7] = bg.txt.k_d_a.copy()
        dbl_d_b[dbase + 7] = bg.txt.k_d_b.copy()
        dbl_d_a[dbase + 8] = bg.txt.v_d_a.copy()
        dbl_d_b[dbase + 8] = bg.txt.v_d_b.copy()
        dbl_d_a[dbase + 9] = bg.txt.out_d_a.copy()
        dbl_d_b[dbase + 9] = bg.txt.out_d_b.copy()
        dbl_d_a[dbase + 10] = bg.txt.ff_in_d_a.copy()
        dbl_d_b[dbase + 10] = bg.txt.ff_in_d_b.copy()
        dbl_d_a[dbase + 11] = bg.txt.ff_out_d_a.copy()
        dbl_d_b[dbase + 11] = bg.txt.ff_out_d_b.copy()
        loader.mark_active_block_done(ctx)
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


def _graph_grad_to_host(
    o: Optional[TArc], name: String, ctx: DeviceContext
) raises -> List[Float32]:
    """Device adapter grad -> host F32 list via the SAME conversion the
    hand-chain path applies (lora_block.mojo _to_host_pair_f32 ->
    _tensor_to_host_f32: F32 direct copy, else cast_tensor to F32 then copy) —
    bit-equal host lists whenever the device tensors are bit-equal."""
    var t = _required_tarc(o, name)
    return _tensor_to_host_f32(t[], ctx)


def klein_stack_lora_backward_graph[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
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
    """P6 graph-engine variant of
    klein_stack_lora_backward_offload_turbo_moddev_rope_scratch (above): a
    COPY of that backward with each per-block recompute+hand-chain pair
    swapped for the autograd_v2 per-block graph call
    (klein_double/single_block_graph_backward). The conductor loop shape is
    PRESERVED VERBATIM (contract C10): the same loader.await_block /
    prefetch_with_ctx / mark_active_block_done calls bracket each block, the
    same ScratchRingAllocator threads through, and the head (final-layer
    backward) + tail (input-token grads) are byte-identical copies.

    Restrictions (fail loud, contract C13 keeps the hand-chain reachable):
      * compute_aux_grads must be False (the trainer's production call; the
        graph arms are aux-off — mod-vec grads need the hand-chain path);
      * full per-block recompute only (DBL/SGL_SAVE_TAIL == 0, the production
        parity default — a non-empty saved tail would silently skip the graph
        recompute, so it raises)."""
    if compute_aux_grads:
        raise Error(
            "klein_stack_lora_backward_graph: compute_aux_grads=True is not"
            " wired through the graph path (aux mod-vec grads); use the"
            " hand-chain klein_stack_lora_backward_offload_turbo_moddev_rope_"
            "scratch"
        )
    if len(saved.dbl_saved) > 0 or len(saved.sgl_saved) > 0:
        raise Error(
            "klein_stack_lora_backward_graph: saved-tail checkpoints present;"
            " the graph path is full-recompute only (DBL/SGL_SAVE_TAIL == 0)"
        )
    var num_double = lora.num_double
    var num_single = lora.num_single
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

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
    while bi >= 0:
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var sl = _sgl_lora_dev_for(lora, bi)
        # graph engine: recompute-forward + backward in ONE call (replaces the
        # recompute_saved + backward_device_resident_scratch pair above).
        var bg = klein_single_block_graph_backward[H, Dh, S](
            d_x, w, single_mod_dev, sl, saved.sgl_x_in[bi],
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        d_x = bg.d_x.copy()
        var sbase = bi * SGL_SLOTS
        sgl_d_a[sbase + 0] = _graph_grad_to_host(bg.qkv_d_a, String("sgl qkv_d_a"), ctx)
        sgl_d_b[sbase + 0] = _graph_grad_to_host(bg.qkv_d_b, String("sgl qkv_d_b"), ctx)
        sgl_d_a[sbase + 1] = _graph_grad_to_host(bg.out_d_a, String("sgl out_d_a"), ctx)
        sgl_d_b[sbase + 1] = _graph_grad_to_host(bg.out_d_b, String("sgl out_d_b"), ctx)
        loader.mark_active_block_done(ctx)
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
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var bl = _dbl_lora_dev_for(lora, di)
        # graph engine: recompute-forward + backward in ONE call (replaces the
        # forward_device_resident_scratch + backward_..._scratch pair above).
        var bg = klein_double_block_graph_backward[H, Dh, N_IMG, N_TXT, S](
            d_io, d_to, w, img_mod_dev, txt_mod_dev, bl,
            saved.dbl_img_in[di], saved.dbl_txt_in[di],
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        d_io = bg.img.d_x.copy()
        d_to = bg.txt.d_x.copy()
        var dbase = di * DBL_SLOTS
        dbl_d_a[dbase + 0] = _graph_grad_to_host(bg.img.q_d_a, String("dbl q_d_a"), ctx)
        dbl_d_b[dbase + 0] = _graph_grad_to_host(bg.img.q_d_b, String("dbl q_d_b"), ctx)
        dbl_d_a[dbase + 1] = _graph_grad_to_host(bg.img.k_d_a, String("dbl k_d_a"), ctx)
        dbl_d_b[dbase + 1] = _graph_grad_to_host(bg.img.k_d_b, String("dbl k_d_b"), ctx)
        dbl_d_a[dbase + 2] = _graph_grad_to_host(bg.img.v_d_a, String("dbl v_d_a"), ctx)
        dbl_d_b[dbase + 2] = _graph_grad_to_host(bg.img.v_d_b, String("dbl v_d_b"), ctx)
        dbl_d_a[dbase + 3] = _graph_grad_to_host(bg.img.out_d_a, String("dbl out_d_a"), ctx)
        dbl_d_b[dbase + 3] = _graph_grad_to_host(bg.img.out_d_b, String("dbl out_d_b"), ctx)
        dbl_d_a[dbase + 4] = _graph_grad_to_host(bg.img.ff_in_d_a, String("dbl ff_in_d_a"), ctx)
        dbl_d_b[dbase + 4] = _graph_grad_to_host(bg.img.ff_in_d_b, String("dbl ff_in_d_b"), ctx)
        dbl_d_a[dbase + 5] = _graph_grad_to_host(bg.img.ff_out_d_a, String("dbl ff_out_d_a"), ctx)
        dbl_d_b[dbase + 5] = _graph_grad_to_host(bg.img.ff_out_d_b, String("dbl ff_out_d_b"), ctx)
        dbl_d_a[dbase + 6] = _graph_grad_to_host(bg.txt.q_d_a, String("dbl txt q_d_a"), ctx)
        dbl_d_b[dbase + 6] = _graph_grad_to_host(bg.txt.q_d_b, String("dbl txt q_d_b"), ctx)
        dbl_d_a[dbase + 7] = _graph_grad_to_host(bg.txt.k_d_a, String("dbl txt k_d_a"), ctx)
        dbl_d_b[dbase + 7] = _graph_grad_to_host(bg.txt.k_d_b, String("dbl txt k_d_b"), ctx)
        dbl_d_a[dbase + 8] = _graph_grad_to_host(bg.txt.v_d_a, String("dbl txt v_d_a"), ctx)
        dbl_d_b[dbase + 8] = _graph_grad_to_host(bg.txt.v_d_b, String("dbl txt v_d_b"), ctx)
        dbl_d_a[dbase + 9] = _graph_grad_to_host(bg.txt.out_d_a, String("dbl txt out_d_a"), ctx)
        dbl_d_b[dbase + 9] = _graph_grad_to_host(bg.txt.out_d_b, String("dbl txt out_d_b"), ctx)
        dbl_d_a[dbase + 10] = _graph_grad_to_host(bg.txt.ff_in_d_a, String("dbl txt ff_in_d_a"), ctx)
        dbl_d_b[dbase + 10] = _graph_grad_to_host(bg.txt.ff_in_d_b, String("dbl txt ff_in_d_b"), ctx)
        dbl_d_a[dbase + 11] = _graph_grad_to_host(bg.txt.ff_out_d_a, String("dbl txt ff_out_d_a"), ctx)
        dbl_d_b[dbase + 11] = _graph_grad_to_host(bg.txt.ff_out_d_b, String("dbl txt ff_out_d_b"), ctx)
        loader.mark_active_block_done(ctx)
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


def klein_stack_lora_backward_offloaded_tape_turbo_moddev_rope_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    mut loader: TurboPlannedLoader,
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    tape: KleinStackLoraOffloadedTape,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_input_grads: Bool = True,
    compute_aux_grads: Bool = True,
) raises -> KleinLoraGrads:
    """Backward over a raw-byte offloaded Klein LoRA activation tape.

    This is the product-shaped CPU_OFFLOADED path: base weights stream in
    reverse through TurboPlannedLoader, and each saved activation boundary is
    restored from host only when its block is recomputed/backpropped.
    """
    var num_double = lora.num_double
    var num_single = lora.num_single
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var d_normed_t = linear_backward_dx(
        _t(d_out, [N_IMG, out_ch], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )

    var ln_img_out_t = restore_to_device(tape.ln_img_out, ctx)
    var mbf = modulate_backward(
        d_normed_t, ln_img_out_t,
        base.final_scale[], ctx, compute_aux_grads,
    )
    var d_final_scale = List[Float32]()
    var d_final_shift = List[Float32]()
    if compute_aux_grads:
        d_final_scale = mbf.d_scale.to_host(ctx)
        d_final_shift = mbf.d_shift.to_host(ctx)

    var img_out_t = restore_to_device(tape.img_out, ctx)
    var d_img_out_t = layer_norm_backward_dx(
        mbf.d_x, img_out_t,
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
    while bi >= 0:
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, F, ctx)
        var sl = _sgl_lora_dev_for(lora, bi)
        var x_t = restore_to_device(tape.sgl_x_in[bi], ctx)
        var x_arc = TArc(x_t^)
        var block_saved = single_block_lora_recompute_saved_device_resident_scratch[H, Dh, S](
            x_arc, w, single_mod_dev, sl,
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        var bg = single_block_lora_backward_device_resident_scratch[H, Dh, S](
            d_x, w, single_mod_dev, sl, block_saved, cos_t, sin_t,
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
        loader.mark_active_block_done(ctx)
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
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, ctx)
        var bl = _dbl_lora_dev_for(lora, di)
        var img_t = restore_to_device(tape.dbl_img_in[di], ctx)
        var txt_t = restore_to_device(tape.dbl_txt_in[di], ctx)
        var img_arc = TArc(img_t^)
        var txt_arc = TArc(txt_t^)
        var fwd = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            img_arc, txt_arc,
            w, img_mod_dev, txt_mod_dev, bl, cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        var bg = double_block_lora_backward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            d_io, d_to, w, img_mod_dev, txt_mod_dev, bl, fwd.saved,
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
        dbl_d_a[dbase + 0] = bg.img.q_d_a.copy()
        dbl_d_b[dbase + 0] = bg.img.q_d_b.copy()
        dbl_d_a[dbase + 1] = bg.img.k_d_a.copy()
        dbl_d_b[dbase + 1] = bg.img.k_d_b.copy()
        dbl_d_a[dbase + 2] = bg.img.v_d_a.copy()
        dbl_d_b[dbase + 2] = bg.img.v_d_b.copy()
        dbl_d_a[dbase + 3] = bg.img.out_d_a.copy()
        dbl_d_b[dbase + 3] = bg.img.out_d_b.copy()
        dbl_d_a[dbase + 4] = bg.img.ff_in_d_a.copy()
        dbl_d_b[dbase + 4] = bg.img.ff_in_d_b.copy()
        dbl_d_a[dbase + 5] = bg.img.ff_out_d_a.copy()
        dbl_d_b[dbase + 5] = bg.img.ff_out_d_b.copy()
        dbl_d_a[dbase + 6] = bg.txt.q_d_a.copy()
        dbl_d_b[dbase + 6] = bg.txt.q_d_b.copy()
        dbl_d_a[dbase + 7] = bg.txt.k_d_a.copy()
        dbl_d_b[dbase + 7] = bg.txt.k_d_b.copy()
        dbl_d_a[dbase + 8] = bg.txt.v_d_a.copy()
        dbl_d_b[dbase + 8] = bg.txt.v_d_b.copy()
        dbl_d_a[dbase + 9] = bg.txt.out_d_a.copy()
        dbl_d_b[dbase + 9] = bg.txt.out_d_b.copy()
        dbl_d_a[dbase + 10] = bg.txt.ff_in_d_a.copy()
        dbl_d_b[dbase + 10] = bg.txt.ff_in_d_b.copy()
        dbl_d_a[dbase + 11] = bg.txt.ff_out_d_a.copy()
        dbl_d_b[dbase + 11] = bg.txt.ff_out_d_b.copy()
        loader.mark_active_block_done(ctx)
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


# ── AdamW step on EVERY adapter (reuses the proven per-adapter AdamW math) ────
# Walks the flat dbl/sgl adapter lists in lockstep with the flat grads and
# mutates A/B (and the carried ma/va/mb/vb) in place. `t` is the 1-based step.
def klein_lora_adamw_step(
    mut set: KleinLoraSet, grads: KleinLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    if t < 1:
        raise Error("klein_lora_adamw_step: t must be >= 1")

    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    var step_size = lr / bc1
    var bc2_sqrt = sqrt(bc2)
    var decay = Float32(1.0) - lr * weight_decay
    var one_minus_beta1 = Float32(1.0) - beta1
    var one_minus_beta2 = Float32(1.0) - beta2
    var seed = UInt32(t)

    # GPU fused step (was a host scalar loop measured at 4.3-6.0 s/step on
    # Klein-9B — PROG_STAGE phase=optim 2026-06-10). Gate:
    # training/lora_adamw_ot_fused_parity.mojo — params bit-equal to the host
    # loop, m/v within ±1 bf16 quantum at ~3e-6 (RNE midpoint ties from
    # device-vs-host 1-ulp arithmetic). Host loop kept in
    # models/klein/lora_adapter.mojo as the gate reference.
    fused_lora_adamw_ot_step(
        set.dbl, grads.dbl_d_a, grads.dbl_d_b,
        step_size, bc2_sqrt, decay, one_minus_beta1, beta2,
        one_minus_beta2, eps, seed, ctx,
    )
    fused_lora_adamw_ot_step(
        set.sgl, grads.sgl_d_a, grads.sgl_d_b,
        step_size, bc2_sqrt, decay, one_minus_beta1, beta2,
        one_minus_beta2, eps, seed, ctx,
    )


# ── per-block prefix scheme in OneTrainer target order ───────────────────────
# The generic PEFT writer appends .lora_A.weight / .lora_B.weight, but the prefix
# is the OneTrainer Flux2 LoRAModule target path. Keep this block-major order in
# lockstep with build_klein_lora_set and Flux2LoRASetup.
def _klein_lora_prefix(block_kind: Int, block_idx: Int, slot: Int) -> String:
    if block_kind == BK_DOUBLE:
        var b = String("transformer.transformer_blocks.") + String(block_idx)
        if slot == 0:
            return b + ".attn.to_q"
        elif slot == 1:
            return b + ".attn.to_k"
        elif slot == 2:
            return b + ".attn.to_v"
        elif slot == 3:
            return b + ".attn.to_out.0"
        elif slot == 4:
            return b + ".ff.linear_in"
        elif slot == 5:
            return b + ".ff.linear_out"
        elif slot == 6:
            return b + ".attn.add_q_proj"
        elif slot == 7:
            return b + ".attn.add_k_proj"
        elif slot == 8:
            return b + ".attn.add_v_proj"
        elif slot == 9:
            return b + ".attn.to_add_out"
        elif slot == 10:
            return b + ".ff_context.linear_in"
        return b + ".ff_context.linear_out"
    var s = String("transformer.single_transformer_blocks.") + String(block_idx)
    if slot == 0:
        return s + ".attn.to_qkv_mlp_proj"
    return s + ".attn.to_out"


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


def _zero_f32(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _slice_row_major(
    values: List[Float32], row_start: Int, row_count: Int, cols: Int
) raises -> List[Float32]:
    if row_start < 0 or row_count < 0 or cols <= 0:
        raise Error("klein LoRA row slice: invalid shape")
    var start = row_start * cols
    var end = start + row_count * cols
    if end > len(values):
        raise Error("klein LoRA row slice: range exceeds tensor data")
    var out = List[Float32]()
    for i in range(start, end):
        out.append(values[i])
    return out^


def _read_lora_adapter_pair(
    st: SafeTensors, prefix: String, default_scale: Float32, ctx: DeviceContext
) raises -> LoraAdapter:
    var key_a = prefix + String(".lora_A.weight")
    var key_b = prefix + String(".lora_B.weight")
    if key_a not in st.tensors:
        raise Error(String("load_klein_flux2_lora: missing ") + key_a)
    if key_b not in st.tensors:
        raise Error(String("load_klein_flux2_lora: missing ") + key_b)

    var a_info = st.tensor_info(key_a)
    var b_info = st.tensor_info(key_b)
    if len(a_info.shape) != 2 or len(b_info.shape) != 2:
        raise Error(String("load_klein_flux2_lora: A/B must be 2-D for ") + prefix)
    var rank = a_info.shape[0]
    var in_f = a_info.shape[1]
    var out_f = b_info.shape[0]
    if b_info.shape[1] != rank:
        raise Error(
            String("load_klein_flux2_lora: B.shape[1]=") + String(b_info.shape[1])
            + String(" != rank ") + String(rank) + String(" for '") + prefix + String("'")
        )

    var adapter_scale = default_scale
    var key_alpha = prefix + String(".alpha")
    if key_alpha in st.tensors:
        var alpha_h = _read_f32(st, key_alpha, ctx)
        if len(alpha_h) != 1:
            raise Error(String("load_klein_flux2_lora: .alpha must have one value for ") + prefix)
        adapter_scale = alpha_h[0] / Float32(rank)

    return LoraAdapter(
        _read_f32(st, key_a, ctx),
        _read_f32(st, key_b, ctx),
        rank,
        in_f,
        out_f,
        adapter_scale,
        _zero_f32(rank * in_f),
        _zero_f32(rank * in_f),
        _zero_f32(out_f * rank),
        _zero_f32(out_f * rank),
    )


def _read_lora_adapter_row_range(
    st: SafeTensors, prefix: String, row_start: Int, row_count: Int,
    default_scale: Float32, ctx: DeviceContext,
) raises -> LoraAdapter:
    var key_a = prefix + String(".lora_A.weight")
    var key_b = prefix + String(".lora_B.weight")
    if key_a not in st.tensors:
        raise Error(String("load_klein_flux2_lora: missing ") + key_a)
    if key_b not in st.tensors:
        raise Error(String("load_klein_flux2_lora: missing ") + key_b)

    var a_info = st.tensor_info(key_a)
    var b_info = st.tensor_info(key_b)
    if len(a_info.shape) != 2 or len(b_info.shape) != 2:
        raise Error(String("load_klein_flux2_lora: A/B must be 2-D for ") + prefix)
    var rank = a_info.shape[0]
    var in_f = a_info.shape[1]
    var out_f = b_info.shape[0]
    if b_info.shape[1] != rank:
        raise Error(
            String("load_klein_flux2_lora: B.shape[1]=") + String(b_info.shape[1])
            + String(" != rank ") + String(rank) + String(" for '") + prefix + String("'")
        )
    if out_f < row_start + row_count:
        raise Error(
            String("load_klein_flux2_lora: row range exceeds B rows for ")
            + prefix
        )

    var adapter_scale = default_scale
    var key_alpha = prefix + String(".alpha")
    if key_alpha in st.tensors:
        var alpha_h = _read_f32(st, key_alpha, ctx)
        if len(alpha_h) != 1:
            raise Error(String("load_klein_flux2_lora: .alpha must have one value for ") + prefix)
        adapter_scale = alpha_h[0] / Float32(rank)

    var a = _read_f32(st, key_a, ctx)
    var b = _slice_row_major(_read_f32(st, key_b, ctx), row_start, row_count, rank)
    return LoraAdapter(
        a^,
        b^,
        rank,
        in_f,
        row_count,
        adapter_scale,
        _zero_f32(rank * in_f),
        _zero_f32(rank * in_f),
        _zero_f32(row_count * rank),
        _zero_f32(row_count * rank),
    )


def _check_adapter_shape(
    ad: LoraAdapter, in_f: Int, out_f: Int, label: String
) raises:
    if ad.in_f != in_f or ad.out_f != out_f:
        raise Error(
            String("load_klein_flux2_lora: shape mismatch for ")
            + label
            + String(" got in=") + String(ad.in_f)
            + String(" out=") + String(ad.out_f)
            + String(" expected in=") + String(in_f)
            + String(" out=") + String(out_f)
        )


def _has_lora_pair(st: SafeTensors, prefix: String) -> Bool:
    return (
        prefix + String(".lora_A.weight") in st.tensors
        and prefix + String(".lora_B.weight") in st.tensors
    )


def _load_klein_flux2_double_blocks_lora(
    var st: SafeTensors,
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    D: Int, F: Int, ctx: DeviceContext,
) raises -> KleinLoraSet:
    # AI Toolkit / Comfy Flux2-Klein exports use model-weight keys directly:
    # diffusion_model.double_blocks.{i}.img_attn.qkv/proj and *_mlp.{0,2}.
    # The live Klein sampler wants per-slot q/k/v/out/ff_in/ff_out adapters.
    var set = build_klein_lora_set(num_double, num_single, D, F, rank, alpha)
    var default_scale = alpha / Float32(rank)
    var mapped = 0
    for bi in range(num_double):
        var base = bi * DBL_SLOTS
        var dp = String("diffusion_model.double_blocks.") + String(bi) + String(".")

        var p = dp + String("img_attn.qkv")
        if _has_lora_pair(st, p):
            var ad0 = _read_lora_adapter_row_range(st, p, 0, D, default_scale, ctx)
            _check_adapter_shape(ad0, D, D, p)
            set.dbl[base + 0] = ad0^
            var ad1 = _read_lora_adapter_row_range(st, p, D, D, default_scale, ctx)
            _check_adapter_shape(ad1, D, D, p)
            set.dbl[base + 1] = ad1^
            var ad2 = _read_lora_adapter_row_range(st, p, 2 * D, D, default_scale, ctx)
            _check_adapter_shape(ad2, D, D, p)
            set.dbl[base + 2] = ad2^
            mapped += 3
        p = dp + String("img_attn.proj")
        if _has_lora_pair(st, p):
            var ad = _read_lora_adapter_pair(st, p, default_scale, ctx)
            _check_adapter_shape(ad, D, D, p)
            set.dbl[base + 3] = ad^
            mapped += 1
        p = dp + String("img_mlp.0")
        if _has_lora_pair(st, p):
            var ad = _read_lora_adapter_pair(st, p, default_scale, ctx)
            _check_adapter_shape(ad, D, 2 * F, p)
            set.dbl[base + 4] = ad^
            mapped += 1
        p = dp + String("img_mlp.2")
        if _has_lora_pair(st, p):
            var ad = _read_lora_adapter_pair(st, p, default_scale, ctx)
            _check_adapter_shape(ad, F, D, p)
            set.dbl[base + 5] = ad^
            mapped += 1

        p = dp + String("txt_attn.qkv")
        if _has_lora_pair(st, p):
            var ad0 = _read_lora_adapter_row_range(st, p, 0, D, default_scale, ctx)
            _check_adapter_shape(ad0, D, D, p)
            set.dbl[base + 6] = ad0^
            var ad1 = _read_lora_adapter_row_range(st, p, D, D, default_scale, ctx)
            _check_adapter_shape(ad1, D, D, p)
            set.dbl[base + 7] = ad1^
            var ad2 = _read_lora_adapter_row_range(st, p, 2 * D, D, default_scale, ctx)
            _check_adapter_shape(ad2, D, D, p)
            set.dbl[base + 8] = ad2^
            mapped += 3
        p = dp + String("txt_attn.proj")
        if _has_lora_pair(st, p):
            var ad = _read_lora_adapter_pair(st, p, default_scale, ctx)
            _check_adapter_shape(ad, D, D, p)
            set.dbl[base + 9] = ad^
            mapped += 1
        p = dp + String("txt_mlp.0")
        if _has_lora_pair(st, p):
            var ad = _read_lora_adapter_pair(st, p, default_scale, ctx)
            _check_adapter_shape(ad, D, 2 * F, p)
            set.dbl[base + 10] = ad^
            mapped += 1
        p = dp + String("txt_mlp.2")
        if _has_lora_pair(st, p):
            var ad = _read_lora_adapter_pair(st, p, default_scale, ctx)
            _check_adapter_shape(ad, F, D, p)
            set.dbl[base + 11] = ad^
            mapped += 1

    for bi in range(num_single):
        var base = bi * SGL_SLOTS
        var sp = String("diffusion_model.single_blocks.") + String(bi) + String(".")
        var p = sp + String("linear1")
        if _has_lora_pair(st, p):
            var ad = _read_lora_adapter_pair(st, p, default_scale, ctx)
            _check_adapter_shape(ad, D, 3 * D + 2 * F, p)
            set.sgl[base + 0] = ad^
            mapped += 1
        p = sp + String("linear2")
        if _has_lora_pair(st, p):
            var ad = _read_lora_adapter_pair(st, p, default_scale, ctx)
            _check_adapter_shape(ad, D + F, D, p)
            set.sgl[base + 1] = ad^
            mapped += 1

    if mapped == 0:
        raise Error("load_klein_flux2_lora: no Flux2/Klein LoRA pairs mapped")
    print("[klein][lora] loaded Flux2/Klein double_blocks adapters:", mapped)
    return set^


# ── SAVE every adapter as a OneTrainer raw LoRA safetensors ──────────────────
# Returns the number of (A,B) pairs written (== total adapters). The train-state
# saver below stays PEFT-like plus AdamW moments; this product LoRA file is for
# OneTrainer parity and validation/sampler loading.
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
    return save_lora_onetrainer(named, path, ctx)


# ── EMA SAVE (Wave 2B item 2i): write the EMA SHADOW params, not the live ones ─
# Mirrors save_klein_lora EXACTLY (same flat iteration order + prefix mapping),
# but substitutes each adapter's A/B with the shadow buffers. The shadow Lists
# are flat in the SAME order the trainer allocated them (range(len(set.dbl)) then
# range(len(set.sgl))), so shadow_dbl_a[bi*DBL_SLOTS+s] aligns with
# set.dbl[bi*DBL_SLOTS+s]. We copy the live adapter for shapes/scale/prefix and
# overwrite a/b only — the optimizer moments are irrelevant to the product LoRA
# save.
def save_klein_lora_ema(
    set: KleinLoraSet,
    shadow_dbl_a: List[List[BFloat16]], shadow_dbl_b: List[List[BFloat16]],
    shadow_sgl_a: List[List[BFloat16]], shadow_sgl_b: List[List[BFloat16]],
    path: String, ctx: DeviceContext,
) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.num_double):
        for s in range(DBL_SLOTS):
            var flat = bi * DBL_SLOTS + s
            var ad = set.dbl[flat].copy()
            ad.a = shadow_dbl_a[flat].copy()
            ad.b = shadow_dbl_b[flat].copy()
            named.append(NamedLora(_klein_lora_prefix(BK_DOUBLE, bi, s), ad^))
    for bi in range(set.num_single):
        for s in range(SGL_SLOTS):
            var flat = bi * SGL_SLOTS + s
            var ad = set.sgl[flat].copy()
            ad.a = shadow_sgl_a[flat].copy()
            ad.b = shadow_sgl_b[flat].copy()
            named.append(NamedLora(_klein_lora_prefix(BK_SINGLE, bi, s), ad^))
    return save_lora_onetrainer(named, path, ctx)


def save_klein_lora_state(set: KleinLoraSet, path: String, ctx: DeviceContext) raises -> Int:
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
    return save_lora_train_state(named, path, ctx)


# ── RESUME: load the adapter A/B back from a save_klein_lora file ────────────
# AdamW moments are ZEROED (resume them from a loop TrainState checkpoint, the
# same contract as load_lora_for_resume). The returned set carries the SAME flat
# order build_klein_lora_set produces, so the AdamW step + save round-trip.
def load_klein_lora_resume(
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
    D: Int = 4096, F: Int = 12288,
) raises -> KleinLoraSet:
    var st = SafeTensors.open(path)
    if (
        String("diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight")
        in st.tensors
    ):
        return _load_klein_flux2_double_blocks_lora(
            st^, num_double, num_single, rank, alpha, D, F, ctx,
        )
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


def load_klein_lora_state(
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> KleinLoraSet:
    var prefixes = klein_lora_prefixes(num_double, num_single)
    var scale = alpha / Float32(rank)
    var named = load_lora_train_state(prefixes, scale, path, ctx)
    var dbl = List[LoraAdapter]()
    var sgl = List[LoraAdapter]()
    var n_dbl = num_double * DBL_SLOTS
    for i in range(n_dbl):
        dbl.append(named[i].adapter.copy())
    var n_sgl = num_single * SGL_SLOTS
    for i in range(n_sgl):
        sgl.append(named[n_dbl + i].adapter.copy())
    return KleinLoraSet(dbl^, sgl^, num_double, num_single, rank)
