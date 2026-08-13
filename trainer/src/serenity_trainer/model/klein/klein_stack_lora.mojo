# model/klein/klein_stack_lora.mojo — Klein (FLUX.2) full DiT LoRA forward/backward.
#
# BORROWED VERBATIM (resident-path functions only) FROM
#   serenitymojo/models/klein/klein_stack_lora.mojo. COPIED into the
#   serenity_trainer namespace per the port rule (serenitymojo/models + /training
#   are NOT reuse sources; only serenitymojo/{tensor,io,ops,scratch_ring}
#   foundation is imported unchanged). The offload-turbo streaming loaders and the
#   safetensors save/load helpers are DROPPED here — block streaming lives in the
#   port's modelLoader, LoRA save in modelSaver/flux2. What remains is the math:
#   the LoRA adapter set, the device-resident training forward (saves a
#   KleinStackForward tape), the device-resident hand-chained backward (returns
#   per-adapter d_A/d_B as KleinLoraGrads), and the AdamW step over every adapter.
#
# This is the Klein equivalent of Serenity's diffusers Flux2Transformer2DModel
# forward: img_in/txt_in projections, 8 double-stream + 24 single-stream blocks
# (9B), the final adaLN + linear, all driven by the shared timestep modulation
# vectors (ModVecs/SingleModVecs from model/klein/weights.mojo).
#
#   double slots (per block, 12): 0-5 img (q,k,v,out,ff_in,ff_out),
#                                  6-11 txt (q,k,v,out,ff_in,ff_out)
#   single slots (per block): 0=qkv 1=out
#   9B: num_double=8, num_single=24 -> 8*12 + 24*2 = 144 LoRA adapters
#       (1:1 with Serenity SEPARATE nn.Linear wrapping, Flux2LoRASetup.py:57).

from max.gpu.host import DeviceContext, HostBuffer
from std.collections import List, Optional
from std.os import getenv
from std.memory import ArcPointer
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import concat, slice, zeros_device

from serenity_trainer.model.klein.double_block import (
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
)
from serenity_trainer.model.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleModVecsDevice, single_modvecs_to_device,
    SingleBlockSaved, SingleBlockGrads,
    SingleBlockLora, SingleBlockLoraDevice, SingleBlockLoraGrads,
    SingleBlockLoraDeviceGrads,
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
)
from serenity_trainer.model.klein.klein_stack import (
    KleinStackBase, KleinStackForward,
    _add_lists, _zeros, _ones, _t, _t_dtype, _linear_fwd, _linear_fwd_wdev,
    _concat_seq, _split_seq, _concat3, _modvec6,
)
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward

from serenity_trainer.model.klein.lora_block import LoraAdapter, LoraAdapterDevice, lora_adapter_to_device
from serenity_trainer.model.klein.lora_adapter import LoraGrads, _lora_adamw


comptime TArc = ArcPointer[Tensor]


@fieldwise_init
struct KleinLoraCfgPreds(Movable):
    var pos: Tensor
    var neg: Tensor


# ── flat-index slot scheme (the carrier's contract) ──────────────────────────
# Double slots (per block, 12 — 1:1 with Serenity's SEPARATE nn.Linear wrapping,
# Flux2LoRASetup.py:57 / Flux2Model.py:52-71, transformer_flux2.py:526-544,314-316):
#   0=img_q 1=img_k 2=img_v 3=img_out 4=img_ff_in 5=img_ff_out
#   6=txt_q 7=txt_k 8=txt_v 9=txt_out 10=txt_ff_in 11=txt_ff_out
#   img_q/k/v   = attn.to_q/to_k/to_v        (in=D, out=D)
#   img_out     = attn.to_out.0              (in=D, out=D)
#   img_ff_in   = ff.linear_in               (in=D, out=2F)
#   img_ff_out  = ff.linear_out              (in=F, out=D)
#   txt_q/k/v   = attn.add_q_proj/add_k_proj/add_v_proj (in=D, out=D)
#   txt_out     = attn.to_add_out            (in=D, out=D)
#   txt_ff_in   = ff_context.linear_in       (in=D, out=2F)
#   txt_ff_out  = ff_context.linear_out      (in=F, out=D)
# Single slots (per block): 0=qkv 1=out.
comptime DBL_SLOTS = 12
comptime SGL_SLOTS = 2
comptime BK_DOUBLE = 0
comptime BK_SINGLE = 1
# Saving block activations reduces backward recompute but is expensive. The real
# Klein cached step001 replay is S=1632 on a 24GB 3090 Ti; saving any double
# blocks, or all 24 single blocks, OOMs before the training forward completes.
# Keep the default at full recompute for the parity gate until offload is added.
comptime DBL_SAVE_TAIL = 0
comptime SGL_SAVE_TAIL = 0


# ── adapter init (A = kaiming_uniform(a=√5), B=0 — PEFT identity at step 0) ────
# 1:1 with Serenity LoRAModule.initialize_weights (LoRAModule.py:312-315):
#   nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
#   nn.init.zeros_(lora_up.weight)
# lora_down is nn.Linear(in_features, rank) → weight shape [rank, in_f], i.e. our
# `A` array. For a 2D weight, torch kaiming_uniform_(a=√5) reduces to a uniform on
# [-bound, +bound] with:
#   gain  = sqrt(2/(1+a²)) = sqrt(2/6) = sqrt(1/3)
#   std   = gain / sqrt(fan_in)              (fan_in = in_features)
#   bound = sqrt(3) * std = sqrt(3)*sqrt(1/3)/sqrt(in_f) = 1/sqrt(in_f)
# So A ~ Uniform(-1/√in_f, +1/√in_f). This MATCHES torch's kaiming_uniform_ exactly
# for the 2D case (the only case here). The RNG STREAM differs from torch's
# Generator (documented divergence, same as the timestep/noise RNG note), but the
# DISTRIBUTION is now identical to Serenity's — not the old fixed-0.01 uniform.
def _kaiming_uniform_a_sqrt5(n: Int, in_f: Int, seed: UInt64) -> List[Float32]:
    var bound = Float32(1.0) / sqrt(Float32(in_f))
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)  # [0,1)
        out.append((u * Float32(2.0) - Float32(1.0)) * bound)          # [-bound,+bound]
    return out^


def make_lora_adapter(
    rank: Int, alpha: Float32, in_f: Int, out_f: Int, seed: UInt64
) -> LoraAdapter:
    var scale = alpha / Float32(rank)
    return LoraAdapter(
        _kaiming_uniform_a_sqrt5(rank * in_f, in_f, seed),  # A = kaiming_uniform(a=√5)
        _zeros(out_f * rank),              # B = 0 (adapter identity at init)
        rank, in_f, out_f, scale,
        _zeros(rank * in_f), _zeros(rank * in_f),    # ma / va
        _zeros(out_f * rank), _zeros(out_f * rank),  # mb / vb
    )


# ── the LoRA carrier: every trained adapter, flat-indexed ────────────────────
struct KleinLoraSet(Copyable, Movable):
    var dbl: List[LoraAdapter]   # num_double * DBL_SLOTS, slots 0-5 img (q,k,v,out,ff_in,ff_out), 6-11 txt
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
# dims: D (model dim), F (mlp_hidden). Per double block, Serenity wraps 12
# SEPARATE nn.Linear (Flux2LoRASetup.py:57, transformer_flux2.py:526-544,314-316):
#   img/txt q,k,v,out : in=D out=D ; ff_in : in=D out=2F ; ff_out : in=F out=D.
#   single to_qkv_mlp_proj: in=D out=3D+2F ; single to_out: in=D+F out=D
#   (the two FULL fused Linears of the parallel single block, incl. mlp columns).
# Each adapter gets a distinct seed so A is non-degenerate per slot.
def build_klein_lora_set(
    num_double: Int, num_single: Int, D: Int, F: Int, rank: Int, alpha: Float32
) -> KleinLoraSet:
    var dbl = List[LoraAdapter]()
    var seed = UInt64(1000)
    for _ in range(num_double):
        # img stream: q,k,v,out (in=D,out=D), ff_in (in=D,out=2F), ff_out (in=F,out=D)
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 0 img_q
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 1 img_k
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 2 img_v
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 3 img_out
        dbl.append(make_lora_adapter(rank, alpha, D, 2 * F, seed)); seed += 1    # 4 img_ff_in
        dbl.append(make_lora_adapter(rank, alpha, F, D, seed)); seed += 1        # 5 img_ff_out
        # txt stream: q,k,v,out (in=D,out=D), ff_in (in=D,out=2F), ff_out (in=F,out=D)
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 6 txt_q
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 7 txt_k
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 8 txt_v
        dbl.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1        # 9 txt_out
        dbl.append(make_lora_adapter(rank, alpha, D, 2 * F, seed)); seed += 1    # 10 txt_ff_in
        dbl.append(make_lora_adapter(rank, alpha, F, D, seed)); seed += 1        # 11 txt_ff_out
    var sgl = List[LoraAdapter]()
    for _ in range(num_single):
        # Serenity wraps the two FULL fused Linears of the parallel single block
        # (Flux2LoRASetup.py:57, transformer_flux2.py:752-763):
        #   slot 0 to_qkv_mlp_proj : ONE Linear [3D+2F, D] → in=D, out=3D+2F
        #     (covers BOTH the q/k/v rows AND the gate_up/mlp rows).
        #   slot 1 to_out          : ONE Linear [D, D+F]   → in=D+F, out=D
        #     (input is the FULL cat([attn, mlp]) [D+F]).
        sgl.append(make_lora_adapter(rank, alpha, D, 3 * D + 2 * F, seed)); seed += 1
        sgl.append(make_lora_adapter(rank, alpha, D + F, D, seed)); seed += 1
    return KleinLoraSet(dbl^, sgl^, num_double, num_single, rank)


# build a transient DoubleBlockLora for block bi from the flat set.
# slot order: 0-5 img (q,k,v,out,ff_in,ff_out), 6-11 txt (q,k,v,out,ff_in,ff_out).
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


# F32-FIX EXPERIMENT (KLEIN_F32_BWD): build F32 copies of a single block's weights/
# saved-activations/mod-vecs so single_block_lora_backward runs the WHOLE chain in
# F32 storage (bf16->F32 is exact). Tests whether the text-grad divergence is a
# bf16-accumulation/conditioning artifact (Mojo-F32 vs Mojo-bf16 decorrelates) or a
# real discrete bug (they agree). LoRA skipped: B=0 => LoRA contributes 0 to d_x.
def _sbw_f32(w: SingleBlockWeights, ctx: DeviceContext) raises -> SingleBlockWeights:
    # SingleBlockWeights ctor builds from host data; copy then overwrite TArc fields.
    var o = w.copy()
    o.w1 = _grad_arc_f32(w.w1, ctx)
    o.w2 = _grad_arc_f32(w.w2, ctx)
    o.w2_att = _grad_arc_f32(w.w2_att, ctx)
    o.w2_mlp = _grad_arc_f32(w.w2_mlp, ctx)
    o.q_norm = _grad_arc_f32(w.q_norm, ctx)
    o.k_norm = _grad_arc_f32(w.k_norm, ctx)
    return o^


def _sbsaved_f32(s: SingleBlockSaved, ctx: DeviceContext) raises -> SingleBlockSaved:
    return SingleBlockSaved(
        _grad_arc_f32(s.x, ctx), _grad_arc_f32(s.ln, ctx), _grad_arc_f32(s.norm, ctx),
        _grad_arc_f32(s.q_pre, ctx), _grad_arc_f32(s.k_pre, ctx),
        _grad_arc_f32(s.q_rms, ctx), _grad_arc_f32(s.k_rms, ctx), _grad_arc_f32(s.v, ctx),
        _grad_arc_f32(s.q_rope, ctx), _grad_arc_f32(s.k_rope, ctx),
        _grad_arc_f32(s.att_flat, ctx),
        _grad_arc_f32(s.mlp_gate, ctx), _grad_arc_f32(s.mlp_up, ctx), _grad_arc_f32(s.mlp, ctx),
        _grad_arc_f32(s.out_in, ctx),
    )


def _smv_f32(mv: SingleModVecsDevice, ctx: DeviceContext) raises -> SingleModVecsDevice:
    return SingleModVecsDevice(
        _grad_arc_f32(mv.shift, ctx), _grad_arc_f32(mv.scale, ctx), _grad_arc_f32(mv.gate, ctx),
    )


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
    var norm_dtype = img[].dtype()
    var norm_ones = TArc(_t_dtype(_ones(D), [D], norm_dtype, ctx))
    var norm_zeros = TArc(_t_dtype(_zeros(D), [D], norm_dtype, ctx))

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
    var norm_dtype = img[].dtype()
    var norm_ones = TArc(_t_dtype(_ones(D), [D], norm_dtype, ctx))
    var norm_zeros = TArc(_t_dtype(_zeros(D), [D], norm_dtype, ctx))

    var dbl_img_in = List[TArc]()
    var dbl_txt_in = List[TArc]()
    var dbl_saved = List[DoubleBlockSaved]()
    # FORWARD-BISECTION: dump each double block's txt/img OUTPUT vs reference trainer per-double-block
    # output hooks -> find where the structured ~0.5% txt forward diff is born.
    var _dbn = List[String]()
    var _dbt = List[ArcPointer[Tensor]]()
    var _db_dump = getenv("KLEIN_DBL_DUMP")
    if _db_dump != String(""):
        _dbn.append(String("din_txt")); _dbt.append(ArcPointer(txt[].clone(ctx)))
        _dbn.append(String("din_img")); _dbt.append(ArcPointer(img[].clone(ctx)))
    for bi in range(num_double):
        dbl_img_in.append(img.copy())
        dbl_txt_in.append(txt.copy())
        var bl = _dbl_lora_dev_for(lora, bi)
        var _dblf = String("")
        if bi == 0 and _db_dump != String(""):
            _dblf = _db_dump + String(".dblfwd")
        var fwd = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            img, txt, dbw[bi], img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
            dump_path=_dblf,
        )
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        if _db_dump != String(""):
            _dbn.append(String("dtxt_") + String(bi))
            _dbt.append(ArcPointer(txt[].clone(ctx)))
            _dbn.append(String("dimg_") + String(bi))
            _dbt.append(ArcPointer(img[].clone(ctx)))
    if _db_dump != String(""):
        save_safetensors(_dbn, _dbt, _db_dump + String(".dbl"), ctx)
        print("[klein-dbl-dump] wrote per-double-block txt/img out (", num_double, "blocks)")

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
    var norm_dtype = saved.img_out[].dtype()
    var norm_ones = TArc(_t_dtype(_ones(D), [D], norm_dtype, ctx))
    var d_normed_t = linear_backward_dx(
        _t_dtype(d_out, [N_IMG, out_ch], base.final_lin[].dtype(), ctx), base.final_lin[],
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

    var d_txt_zero = zeros_device([N_TXT, D], d_img_out_t.dtype(), ctx)
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
        dbl_d_a[dbase + 0] = bg.img.q_d_a.copy();      dbl_d_b[dbase + 0] = bg.img.q_d_b.copy()
        dbl_d_a[dbase + 1] = bg.img.k_d_a.copy();      dbl_d_b[dbase + 1] = bg.img.k_d_b.copy()
        dbl_d_a[dbase + 2] = bg.img.v_d_a.copy();      dbl_d_b[dbase + 2] = bg.img.v_d_b.copy()
        dbl_d_a[dbase + 3] = bg.img.out_d_a.copy();    dbl_d_b[dbase + 3] = bg.img.out_d_b.copy()
        dbl_d_a[dbase + 4] = bg.img.ff_in_d_a.copy();  dbl_d_b[dbase + 4] = bg.img.ff_in_d_b.copy()
        dbl_d_a[dbase + 5] = bg.img.ff_out_d_a.copy(); dbl_d_b[dbase + 5] = bg.img.ff_out_d_b.copy()
        dbl_d_a[dbase + 6] = bg.txt.q_d_a.copy();      dbl_d_b[dbase + 6] = bg.txt.q_d_b.copy()
        dbl_d_a[dbase + 7] = bg.txt.k_d_a.copy();      dbl_d_b[dbase + 7] = bg.txt.k_d_b.copy()
        dbl_d_a[dbase + 8] = bg.txt.v_d_a.copy();      dbl_d_b[dbase + 8] = bg.txt.v_d_b.copy()
        dbl_d_a[dbase + 9] = bg.txt.out_d_a.copy();    dbl_d_b[dbase + 9] = bg.txt.out_d_b.copy()
        dbl_d_a[dbase + 10] = bg.txt.ff_in_d_a.copy(); dbl_d_b[dbase + 10] = bg.txt.ff_in_d_b.copy()
        dbl_d_a[dbase + 11] = bg.txt.ff_out_d_a.copy(); dbl_d_b[dbase + 11] = bg.txt.ff_out_d_b.copy()
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
    var norm_dtype = saved.img_out[].dtype()
    var norm_ones = TArc(_t_dtype(_ones(D), [D], norm_dtype, ctx))
    var norm_zeros = TArc(_t_dtype(_zeros(D), [D], norm_dtype, ctx))

    var d_normed_t = linear_backward_dx(
        _t_dtype(d_out, [N_IMG, out_ch], base.final_lin[].dtype(), ctx), base.final_lin[],
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

    var d_txt_zero = zeros_device([N_TXT, D], d_img_out_t.dtype(), ctx)
    var d_x = TArc(concat(0, ctx, d_txt_zero, d_img_out_t))

    var sgl_d_a = List[List[Float32]]()
    var sgl_d_b = List[List[Float32]]()
    for _ in range(num_single * SGL_SLOTS):
        sgl_d_a.append(List[Float32]())
        sgl_d_b.append(List[Float32]())

    var d_single_mod = _zeros(3 * D)
    var bi = num_single - 1
    var saved_single_start = num_single - len(saved.sgl_saved)
    # PER-BLOCK localization: collect running d_x AFTER each single block backward
    # (== grad at that block's INPUT). vs reference trainer per-block input-grad hooks -> finds the
    # block index where txt grad first diverges.
    var _dxn = List[String]()
    var _dxt = List[ArcPointer[Tensor]]()
    var _dx_dump = getenv("KLEIN_DY_DUMP")
    var norm_ones_f32 = TArc(_t_dtype(_ones(D), [D], STDtype.F32, ctx))
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
        # LOCALIZATION self-check: on the LAST single block (a SAVED block), feed its
        # own saved input x back through the RECOMPUTE path and dump both. If saved vs
        # recompute diverge, the saved/recompute activations the backward consumes are
        # corrupted (loss stays exact) — the documented scratch-lifetime footgun.
        if bi == num_single - 1 and getenv("KLEIN_DY_DUMP") != String(""):
            var rc = single_block_lora_recompute_saved_device_resident_scratch[H, Dh, S](
                block_saved.x, sbw[bi], single_mod_dev, sl,
                cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
            )
            var sn = List[String]()
            var st = List[ArcPointer[Tensor]]()
            sn.append("saved_q_pre");  st.append(ArcPointer(block_saved.q_pre[].clone(ctx)))
            sn.append("recomp_q_pre"); st.append(ArcPointer(rc.q_pre[].clone(ctx)))
            sn.append("saved_v");      st.append(ArcPointer(block_saved.v[].clone(ctx)))
            sn.append("recomp_v");     st.append(ArcPointer(rc.v[].clone(ctx)))
            sn.append("saved_q_rope"); st.append(ArcPointer(block_saved.q_rope[].clone(ctx)))
            sn.append("recomp_q_rope");st.append(ArcPointer(rc.q_rope[].clone(ctx)))
            sn.append("saved_norm");   st.append(ArcPointer(block_saved.norm[].clone(ctx)))
            sn.append("recomp_norm");  st.append(ArcPointer(rc.norm[].clone(ctx)))
            sn.append("saved_out_in"); st.append(ArcPointer(block_saved.out_in[].clone(ctx)))
            sn.append("recomp_out_in");st.append(ArcPointer(rc.out_in[].clone(ctx)))
            save_safetensors(sn, st, getenv("KLEIN_DY_DUMP") + String(".sblk"), ctx)
            print("[klein-dy-dump] wrote saved-vs-recompute single-block activations (block", bi, ")")
        var _sdpa_dump = String("")
        var _probe_blk = 0
        if getenv("KLEIN_INJECT_BLK") != String(""):
            _probe_blk = atol(getenv("KLEIN_INJECT_BLK"))
        if bi == _probe_blk and getenv("KLEIN_DY_DUMP") != String(""):
            _sdpa_dump = getenv("KLEIN_DY_DUMP") + String(".sdpa")
        # DELTA-BISECTION inject: replace this block's INPUT grad with reference trainer's correct
        # grad-at-block-output (matched input), so dx_{bi} measures THIS block's
        # backward in isolation vs reference trainer sdx_{bi}. Isolates the per-block deterministic
        # delta. Env KLEIN_INJECT_BLK=<idx>, KLEIN_INJECT=<path to {"inj":[S,D]}>.
        if getenv("KLEIN_INJECT_BLK") != String("") and bi == atol(getenv("KLEIN_INJECT_BLK")):
            var inj_st = ShardedSafeTensors.open(getenv("KLEIN_INJECT"))
            var inj_dev = Tensor.from_view(inj_st.tensor_view(String("inj")), ctx)
            d_x = TArc(cast_tensor(inj_dev, d_x[].dtype(), ctx))
            print("[klein-inject] block", bi, "d_x <- reference trainer grad-at-block-output")
        var _f32_bwd = getenv("KLEIN_F32_BWD") != String("")
        var bg: SingleBlockLoraDeviceGrads
        if _f32_bwd:
            # F32-FIX EXPERIMENT: run this block's backward fully in F32 (no LoRA;
            # B=0 so LoRA adds nothing to d_x). d_x carrier stays F32 across blocks.
            var w32 = _sbw_f32(sbw[bi], ctx)
            var s32 = _sbsaved_f32(block_saved, ctx)
            var m32 = _smv_f32(single_mod_dev, ctx)
            var dx32 = _grad_arc_f32(d_x, ctx)
            var no_lora = SingleBlockLoraDevice(None, None)
            bg = single_block_lora_backward_device_resident_scratch[H, Dh, S](
                dx32, w32, m32, no_lora, s32, cos_t, sin_t,
                D, F, eps, norm_ones_f32[], ctx, scratch, False,
                dump_path=_sdpa_dump,
            )
        else:
            bg = single_block_lora_backward_device_resident_scratch[H, Dh, S](
                d_x, sbw[bi], single_mod_dev, sl, block_saved, cos_t, sin_t,
                D, F, eps, norm_ones[], ctx, scratch, compute_aux_grads,
                dump_path=_sdpa_dump,
            )
        d_x = bg.d_x.copy()
        if _dx_dump != String(""):
            _dxn.append(String("dx_") + String(bi))
            _dxt.append(ArcPointer(d_x[].clone(ctx)))
            # FORWARD-DIVERGENCE: also dump each block's FORWARD INPUT (block_saved.x)
            # vs reference trainer sfx_{bi} -> finds where forward text activations first diverge.
            _dxn.append(String("xfwd_") + String(bi))
            _dxt.append(ArcPointer(block_saved.x[].clone(ctx)))
        if compute_aux_grads and not _f32_bwd:
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

    if _dx_dump != String(""):
        save_safetensors(_dxn, _dxt, _dx_dump + String(".dxall"), ctx)
        print("[klein-dy-dump] wrote per-block running d_x (", len(_dxn), "blocks)")

    var d_txt_out = TArc(slice(d_x[], 0, 0, N_TXT, ctx))
    var d_img_out2 = TArc(slice(d_x[], 0, N_TXT, N_IMG, ctx))

    # LOCALIZATION: dump the single<->double boundary grad (joint stream grad AFTER
    # all single-block backward, BEFORE any double-block backward). vs reference trainer grad at
    # single_transformer_blocks[0] input: match => error is in DOUBLE blocks; differ
    # => error is in head/output or SINGLE blocks.
    if getenv("KLEIN_DY_DUMP") != String(""):
        var bn = List[String]()
        bn.append("d_img_bnd")
        bn.append("d_txt_bnd")
        var bt = List[ArcPointer[Tensor]]()
        bt.append(ArcPointer(d_img_out2[].clone(ctx)))
        bt.append(ArcPointer(d_txt_out[].clone(ctx)))
        save_safetensors(bn, bt, getenv("KLEIN_DY_DUMP") + String(".bnd"), ctx)
        print("[klein-dy-dump] wrote single<->double boundary d_img_bnd/d_txt_bnd")

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
        # DECISIVE-TEST (Klein d_B bug): dump block-0 image-stream per-projection
        # d_y so it can be diffed vs SerenityTrainer's to_q/k/v module-output grads.
        # Inert unless KLEIN_DY_DUMP is set in the env.
        var _dy_dump = String("")
        if di == 0 and getenv("KLEIN_DY_DUMP") != String(""):
            _dy_dump = getenv("KLEIN_DY_DUMP")
            # LOCALIZATION: dump d_io/d_to ENTERING block-0 backward (== grad at
            # block-0 OUTPUT). If already wrong vs reference trainer, error is UPSTREAM (singles /
            # doubles 1..N); if correct, error is born inside block-0 backward.
            var dion = List[String]()
            dion.append("d_io")
            dion.append("d_to")
            var diot = List[ArcPointer[Tensor]]()
            diot.append(ArcPointer(d_io[].clone(ctx)))
            diot.append(ArcPointer(d_to[].clone(ctx)))
            save_safetensors(dion, diot, _dy_dump + String(".dio"), ctx)
            print("[klein-dy-dump] wrote d_io/d_to -> ", _dy_dump, ".dio")
        var bg = double_block_lora_backward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            d_io, d_to, dbw[di], img_mod_dev, txt_mod_dev, bl, fwd.saved,
            cos_t, sin_t, D, F, eps, norm_ones[], ctx, scratch, compute_aux_grads,
            dump_path=_dy_dump,
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
        dbl_d_a[dbase + 0] = bg.img.q_d_a.copy();      dbl_d_b[dbase + 0] = bg.img.q_d_b.copy()
        dbl_d_a[dbase + 1] = bg.img.k_d_a.copy();      dbl_d_b[dbase + 1] = bg.img.k_d_b.copy()
        dbl_d_a[dbase + 2] = bg.img.v_d_a.copy();      dbl_d_b[dbase + 2] = bg.img.v_d_b.copy()
        dbl_d_a[dbase + 3] = bg.img.out_d_a.copy();    dbl_d_b[dbase + 3] = bg.img.out_d_b.copy()
        dbl_d_a[dbase + 4] = bg.img.ff_in_d_a.copy();  dbl_d_b[dbase + 4] = bg.img.ff_in_d_b.copy()
        dbl_d_a[dbase + 5] = bg.img.ff_out_d_a.copy(); dbl_d_b[dbase + 5] = bg.img.ff_out_d_b.copy()
        dbl_d_a[dbase + 6] = bg.txt.q_d_a.copy();      dbl_d_b[dbase + 6] = bg.txt.q_d_b.copy()
        dbl_d_a[dbase + 7] = bg.txt.k_d_a.copy();      dbl_d_b[dbase + 7] = bg.txt.k_d_b.copy()
        dbl_d_a[dbase + 8] = bg.txt.v_d_a.copy();      dbl_d_b[dbase + 8] = bg.txt.v_d_b.copy()
        dbl_d_a[dbase + 9] = bg.txt.out_d_a.copy();    dbl_d_b[dbase + 9] = bg.txt.out_d_b.copy()
        dbl_d_a[dbase + 10] = bg.txt.ff_in_d_a.copy(); dbl_d_b[dbase + 10] = bg.txt.ff_in_d_b.copy()
        dbl_d_a[dbase + 11] = bg.txt.ff_out_d_a.copy(); dbl_d_b[dbase + 11] = bg.txt.ff_out_d_b.copy()
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
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
    stochastic_rounding: Bool = True,
) raises:
    var nd = set.num_double * DBL_SLOTS
    for i in range(nd):
        var lg = LoraGrads(grads.dbl_d_a[i].copy(), grads.dbl_d_b[i].copy())
        _lora_adamw(set.dbl[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay, stochastic_rounding)
    var ns = set.num_single * SGL_SLOTS
    for i in range(ns):
        var lg = LoraGrads(grads.sgl_d_a[i].copy(), grads.sgl_d_b[i].copy())
        _lora_adamw(set.sgl[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay, stochastic_rounding)


# ── INFERENCE-ONLY resident forward (NO saved tape; for the sampler) ─────────
# Serenity's Flux2Sampler.__sample_base runs the transformer under
# @torch.no_grad() — NO saved activations, NO recompute checkpoints. This is the
# resident-weight analogue of the source's offload predict path
# (klein_stack_lora_predict_offload_turbo_moddev_rope_scratch): same math as the
# training forward but it threads the per-block `predict_*_resident_scratch`
# variants (which skip _make_saved / DoubleBlockSaved) and returns ONLY the final
# velocity prediction [N_IMG, out_ch] as a host list. The sampler must NOT reuse
# the training forward (which builds a KleinStackForward tape sized S=4608 at
# 1024 res). Mirrors klein_stack_lora.mojo:813-887 with resident weights.
def klein_stack_lora_predict_resident_moddev_rope_scratch[
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
) raises -> List[Float32]:
    var num_double = len(dbw)
    var num_single = len(sbw)

    var no_bias = Optional[Tensor](None)
    var img = TArc(linear(img_tokens_t[], base.img_in[], no_bias^, ctx))
    var no_bias_txt = Optional[Tensor](None)
    var txt = TArc(linear(txt_tokens_t[], base.txt_in[], no_bias_txt^, ctx))
    var norm_dtype = img[].dtype()
    var norm_ones = TArc(_t_dtype(_ones(D), [D], norm_dtype, ctx))
    var norm_zeros = TArc(_t_dtype(_zeros(D), [D], norm_dtype, ctx))

    for bi in range(num_double):
        var bl = _dbl_lora_dev_for(lora, bi)
        var fwd = double_block_lora_predict_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
            img, txt, dbw[bi], img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, norm_ones[], norm_zeros[], ctx, scratch,
        )
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()

    var x = TArc(concat(0, ctx, txt[], img[]))

    for bi in range(num_single):
        var sl = _sgl_lora_dev_for(lora, bi)
        var fwd = single_block_lora_predict_device_resident_scratch[H, Dh, S](
            x, sbw[bi], single_mod_dev, sl, cos_t, sin_t, D, F, eps,
            norm_ones[], norm_zeros[], ctx, scratch,
        )
        x = fwd.out.copy()

    var img_out = TArc(slice(x[], 0, N_TXT, N_IMG, ctx))
    var ln_img_out = TArc(layer_norm(
        img_out[], norm_ones[], norm_zeros[], eps, ctx,
    ))
    var normed = modulate(
        ln_img_out[], base.final_scale[], base.final_shift[], ctx,
    )
    var no_bias_out = Optional[Tensor](None)
    var out = linear(normed, base.final_lin[], no_bias_out^, ctx).to_host(ctx)
    return out^
