# serenitymojo/models/flux/lora_block.mojo
#
# LoRA-ON-PROJECTION for the Flux (flux1-dev) DOUBLE + SINGLE blocks. Mirrors the
# PROVEN Ernie/Klein LoRA template (models/ernie/lora_block.mojo,
# models/klein/lora_block.mojo) specialized to FLUX's OneTrainer target set
# (verified line-by-line against /home/alex/OneTrainer/modules/util/convert/lora/
# convert_flux_lora.py:6-41):
#
#   DOUBLE block, per stream s in {img, txt} — 6 trained projections each:
#     img_attn.qkv.0/.1/.2  -> to_q / to_k / to_v   (the 3 D-slices of wqkv [3D,D])
#     img_attn.proj         -> proj                  (wproj [D,D])
#     img_mlp.0             -> mlp0                  (wmlp0 [Fmlp,D])
#     img_mlp.2             -> mlp2                  (wmlp2 [D,Fmlp])
#     (txt_* mirror; img_mod.lin/txt_mod.lin are STACK-level base linears, wired
#      at the stack layer, NOT here — same scope as Klein/Ernie which keep the
#      modulation/embedder linears frozen.)
#   SINGLE block — 5 trained projections:
#     linear1.0/.1/.2  -> to_q / to_k / to_v   (the 3 D-slices of w1's first 3D rows)
#     linear1.3        -> proj_mlp              (the Fmlp-slice of w1's rows)
#     linear2          -> linear2               (w2 [D, D+Fmlp])
#     (modulation.lin is STACK-level, wired at the stack layer.)
#
# WHY 3 SEPARATE q/k/v ADAPTERS (not one fused qkv adapter like Klein): OT/
# diffusers train to_q/to_k/to_v as SEPARATE Linears (convert map emits distinct
# keys img_attn.qkv.0/.1/.2). A rank-r adapter on a fused [3D,D] weight is NOT
# the same low-rank family as three independent rank-r adapters on the 3 D-slices.
# Modelling them separately is the OT-faithful recipe AND makes the saved keys
# round-trip with OT-trained / ai-toolkit LoRAs.
#
# WHY THE TRAINER MATH IS THE AUTHORITY (identical to Ernie/Klein lora_block):
#   For a projection y = linear(x, W) (W [out,in]), LoRA-adapted output is
#       y' = linear(x, W) + scale*((x @ Aᵀ) @ Bᵀ)
#   A=[rank,in], B=[out,rank], scale = alpha/rank.
#   BACKWARD (given d_y' at the projection output, x the saved projection input):
#       d_dy = scale*d_y' ; d_B = d_dyᵀ@t (t=x@Aᵀ) ; d_t = d_dy@B ;
#       d_A = d_tᵀ@x ; d_x = d_t@A  (the LoRA branch's contribution to the
#       projection INPUT grad, SUMMED into the base d_x).
#   The two helpers below are byte-identical to train_step._lora_fwd / _lora_bwd
#   plus the d_x term that file drops.
#
# NO NEW ops/ PRIMITIVE: forward = two linear()s; backward = two linear_backward()s
# plus the existing slice/concat the base block already uses. Tenet 1 honored.
#
# Bit-exact base when adapters absent: each flux_lora_apply returns base_y
# unchanged when the Optional is empty, so the LoRA forward reduces to the
# verified base forward (saved activations are the LoRA-modified ones, so the
# backward recompute regenerates them identically — same checkpoint contract).
#
# Mojo 1.0.0b1: def not fn; Tensor move-only crosses API as host List[Float32].

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx

# REUSE the trainer's LoRA structs (the target authority).
from serenitymojo.training.train_step import LoraAdapter, LoraGrads

# Forward + backward ops shared with the base block (Tenet 1: nothing new here).
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, reshape_in_place, slice, concat, add,
)
from serenitymojo.ops.norm_backward import rms_norm_backward, layer_norm_backward
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import gate_residual_backward, rope_backward
from serenitymojo.ops.shape_backward import cat_backward

from serenitymojo.models.flux.block import (
    ModVecs, SingleModVecs,
    StreamWeights, DoubleBlockWeights, StreamSaved, DoubleBlockSaved,
    DoubleBlockForward, StreamGrads, DoubleBlockGrads,
    SingleBlockWeights, SingleBlockSaved, SingleBlockForward, SingleBlockGrads,
)
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectOFTSet,
)
from serenitymojo.training.dora_substitution_device import (
    dora_device_from_host, dora_substitution_forward_device,
    dora_substitution_backward_device,
)
from serenitymojo.training.oft_onetrainer_device import (
    oft_ot_rotate_b4, oft_ot_rotate_backward_b4,
)


comptime TArc = ArcPointer[Tensor]


# ── host helpers ─────────────────────────────────────────────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _ones(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(1.0)
    return o^


def _zeros(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(0.0)
    return o^


# NATIVE BF16 COMPUTE: F32-host carriers → BF16 device tensors so matmuls hit
# linear's native bf16·bf16 path (F32 accumulate inside the GEMM, like flame-core).
def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


# F32 upload — ONLY at F32-only op boundaries (rope_backward cos/sin,
# gate_residual_backward grad_out/gate, cat_backward grad).
def _tf32(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# Re-upload a saved BF16 host activation → BF16 *device* tensor (native, no F32
# detour) so the backward matmuls run on bf16 saved acts.
def _tb16(vals: List[BFloat16], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host_bf16(vals, shape^, ctx)


# Convert F32 host list → BFloat16 (for saving the x input copy as BF16).
def _f32_to_bf16(v: List[Float32]) -> List[BFloat16]:
    var o = List[BFloat16]()
    for i in range(len(v)):
        o.append(BFloat16(v[i]))
    return o^


# Local bf16↔F32 device casts for single F32-only op calls (no shared-op edits).
def _to_f32(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(t, STDtype.F32, ctx)


def _to_bf16(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(t, STDtype.BF16, ctx)


# ── LoRA fwd/bwd (host list; byte-identical to train_step._lora_fwd/_lora_bwd) ─
def flux_lora_fwd(
    x_h: List[Float32], lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> List[Float32]:
    # NATIVE BF16: adapter A/B matmuls upload bf16 → bf16·bf16 GEMM (F32 accumulate).
    var nb1 = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb1^, ctx,
    ).to_host(ctx)                                   # [M,rank] (upcast bf16→F32)
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        nb2^, ctx,
    ).to_host(ctx)                                   # [M,out] (upcast bf16→F32)
    var out = List[Float32]()
    for i in range(len(dy)):
        out.append(lo.scale * dy[i])
    return out^


# base_y + LoRA(x) if present; else base_y unchanged (bit-exact base no-regress).
def flux_lora_apply(
    base_y: List[Float32], x_h: List[Float32], lo: Optional[LoraAdapter],
    M: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    if not lo:
        return base_y.copy()
    var contrib = flux_lora_fwd(x_h, lo.value(), M, ctx)
    var out = List[Float32]()
    for i in range(len(base_y)):
        out.append(base_y[i] + contrib[i])
    return out^


struct FluxLoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]   # LoRA contribution to the projection INPUT grad [M,in]

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def flux_lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> FluxLoraGrads:
    # NATIVE BF16: adapter A/B matmuls + their backward upload bf16 (F32 accumulate
    # inside the GEMM). Returned grads upcast bf16→F32 via to_host for the optimizer.
    var nb_t = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb_t^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var d_dy = List[Float32]()
    for i in range(len(d_contrib_h)):
        d_dy.append(lo.scale * d_contrib_h[i])       # [M,out]
    var lbB = linear_backward(
        Tensor.from_host(d_dy^, [M, lo.out_f], STDtype.BF16, ctx),
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.to_host(ctx)                   # [M,rank]
    var d_b = lbB.d_w.to_host(ctx)                   # [out_f,rank]
    var lbA = linear_backward(
        Tensor.from_host(d_t^, [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_x_lo = lbA.d_x.to_host(ctx)                # [M,in_f]
    var d_a = lbA.d_w.to_host(ctx)                   # [rank,in_f]
    return FluxLoraGrads(d_a^, d_b^, d_x_lo^)


# take cols [c0,c0+w) from a [rows,total] row-major host list -> [rows,w].
def _take_cols(src: List[Float32], rows: Int, total: Int, c0: Int, w: Int) -> List[Float32]:
    var o = List[Float32]()
    for r in range(rows):
        var base = r * total
        for c in range(w):
            o.append(src[base + c0 + c])
    return o^


# add a [rows,w] delta into cols [c0,c0+w) of a [rows,total] buffer (others kept).
def _add_into_cols(
    dst: List[Float32], delta: List[Float32], rows: Int, total: Int, c0: Int, w: Int
) -> List[Float32]:
    var o = dst.copy()
    for r in range(rows):
        var base = r * total
        for c in range(w):
            o[base + c0 + c] = o[base + c0 + c] + delta[r * w + c]
    return o^


# ═══════════════════════════════════════════════════════════════════════════
# Per-block LoRA carriers (Optional slots; canonical slot order below).
# ═══════════════════════════════════════════════════════════════════════════
# Double-stream slot order (per stream): to_q, to_k, to_v, proj, mlp0, mlp2.
comptime DBL_STREAM_SLOTS = 6
comptime D_SQ = 0    # to_q
comptime D_SK = 1    # to_k
comptime D_SV = 2    # to_v
comptime D_PROJ = 3  # img/txt_attn.proj
comptime D_MLP0 = 4  # img/txt_mlp.0
comptime D_MLP2 = 5  # img/txt_mlp.2

# Single-block slot order: to_q, to_k, to_v, proj_mlp, linear2.
comptime SGL_SLOTS = 5
comptime S_SQ = 0
comptime S_SK = 1
comptime S_SV = 2
comptime S_PMLP = 3   # linear1.3 (proj_mlp)
comptime S_L2 = 4     # linear2


struct StreamLora(Copyable, Movable):
    var to_q: Optional[LoraAdapter]
    var to_k: Optional[LoraAdapter]
    var to_v: Optional[LoraAdapter]
    var proj: Optional[LoraAdapter]
    var mlp0: Optional[LoraAdapter]
    var mlp2: Optional[LoraAdapter]

    def __init__(
        out self,
        var to_q: Optional[LoraAdapter], var to_k: Optional[LoraAdapter],
        var to_v: Optional[LoraAdapter], var proj: Optional[LoraAdapter],
        var mlp0: Optional[LoraAdapter], var mlp2: Optional[LoraAdapter],
    ):
        self.to_q = to_q^
        self.to_k = to_k^
        self.to_v = to_v^
        self.proj = proj^
        self.mlp0 = mlp0^
        self.mlp2 = mlp2^


struct DoubleBlockLora(Copyable, Movable):
    var img: StreamLora
    var txt: StreamLora

    def __init__(out self, var img: StreamLora, var txt: StreamLora):
        self.img = img^
        self.txt = txt^


struct SingleBlockLora(Copyable, Movable):
    var to_q: Optional[LoraAdapter]
    var to_k: Optional[LoraAdapter]
    var to_v: Optional[LoraAdapter]
    var proj_mlp: Optional[LoraAdapter]
    var linear2: Optional[LoraAdapter]

    def __init__(
        out self,
        var to_q: Optional[LoraAdapter], var to_k: Optional[LoraAdapter],
        var to_v: Optional[LoraAdapter], var proj_mlp: Optional[LoraAdapter],
        var linear2: Optional[LoraAdapter],
    ):
        self.to_q = to_q^
        self.to_k = to_k^
        self.to_v = to_v^
        self.proj_mlp = proj_mlp^
        self.linear2 = linear2^


# ── per-stream / per-block LoRA grads (parallel to slots; empty if absent) ────
struct StreamLoraGrads(Copyable, Movable):
    var d_a: List[List[Float32]]   # DBL_STREAM_SLOTS entries
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


struct DoubleBlockLoraGrads(Copyable, Movable):
    var img: StreamLoraGrads
    var txt: StreamLoraGrads

    def __init__(out self, var img: StreamLoraGrads, var txt: StreamLoraGrads):
        self.img = img^
        self.txt = txt^


struct SingleBlockLoraGrads(Copyable, Movable):
    var d_a: List[List[Float32]]   # SGL_SLOTS entries
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


struct DoubleBlockLoraBackward(Movable):
    var base: DoubleBlockGrads
    var lora: DoubleBlockLoraGrads

    def __init__(out self, var base: DoubleBlockGrads, var lora: DoubleBlockLoraGrads):
        self.base = base^
        self.lora = lora^


struct SingleBlockLoraBackward(Movable):
    var base: SingleBlockGrads
    var lora: SingleBlockLoraGrads

    def __init__(out self, var base: SingleBlockGrads, var lora: SingleBlockLoraGrads):
        self.base = base^
        self.lora = lora^


def _empty_f32() -> List[Float32]:
    return List[Float32]()


comptime FLUX_DIRECT_ALGO_DORA = 1
comptime FLUX_DIRECT_ALGO_OFT = 2
comptime FLUX_DIRECT_TGT_ATTN = 1
comptime FLUX_DIRECT_TGT_ALL = 2


def _flux_direct_dbl_slot_targeted(slot: Int, targets: Int) raises -> Bool:
    if targets < FLUX_DIRECT_TGT_ATTN or targets > FLUX_DIRECT_TGT_ALL:
        raise Error("FluxBlockDirectLycoris: targets must be 1(attn)|2(all)")
    var s = slot % DBL_STREAM_SLOTS
    if s == D_SQ or s == D_SK or s == D_SV or s == D_PROJ:
        return targets >= FLUX_DIRECT_TGT_ATTN
    return targets >= FLUX_DIRECT_TGT_ALL


def _flux_direct_sgl_slot_targeted(slot: Int, targets: Int) raises -> Bool:
    if targets < FLUX_DIRECT_TGT_ATTN or targets > FLUX_DIRECT_TGT_ALL:
        raise Error("FluxBlockDirectLycoris: targets must be 1(attn)|2(all)")
    var s = slot % SGL_SLOTS
    if s == S_SQ or s == S_SK or s == S_SV:
        return targets >= FLUX_DIRECT_TGT_ATTN
    return targets >= FLUX_DIRECT_TGT_ALL


struct FluxDoubleBlockDirectLycoris(Copyable, Movable):
    var algo: Int
    var dora: FlatDirectDoRASet
    var oft: FlatDirectOFTSet
    var img_q_slot: Int
    var img_k_slot: Int
    var img_v_slot: Int
    var img_proj_slot: Int
    var img_mlp0_slot: Int
    var img_mlp2_slot: Int
    var txt_q_slot: Int
    var txt_k_slot: Int
    var txt_v_slot: Int
    var txt_proj_slot: Int
    var txt_mlp0_slot: Int
    var txt_mlp2_slot: Int

    def __init__(
        out self, algo: Int, var dora: FlatDirectDoRASet,
        var oft: FlatDirectOFTSet, base_slot: Int, targets: Int,
    ) raises:
        var iq = -1
        var ik = -1
        var iv = -1
        var ip = -1
        var im0 = -1
        var im2 = -1
        var tq = -1
        var tk = -1
        var tv = -1
        var tp = -1
        var tm0 = -1
        var tm2 = -1
        var compact = base_slot
        for slot in range(2 * DBL_STREAM_SLOTS):
            if not _flux_direct_dbl_slot_targeted(slot, targets):
                continue
            var stream_slot = slot % DBL_STREAM_SLOTS
            var is_img = slot < DBL_STREAM_SLOTS
            if is_img:
                if stream_slot == D_SQ:
                    iq = compact
                elif stream_slot == D_SK:
                    ik = compact
                elif stream_slot == D_SV:
                    iv = compact
                elif stream_slot == D_PROJ:
                    ip = compact
                elif stream_slot == D_MLP0:
                    im0 = compact
                else:
                    im2 = compact
            else:
                if stream_slot == D_SQ:
                    tq = compact
                elif stream_slot == D_SK:
                    tk = compact
                elif stream_slot == D_SV:
                    tv = compact
                elif stream_slot == D_PROJ:
                    tp = compact
                elif stream_slot == D_MLP0:
                    tm0 = compact
                else:
                    tm2 = compact
            compact += 1
        self.algo = algo
        self.dora = dora^
        self.oft = oft^
        self.img_q_slot = iq
        self.img_k_slot = ik
        self.img_v_slot = iv
        self.img_proj_slot = ip
        self.img_mlp0_slot = im0
        self.img_mlp2_slot = im2
        self.txt_q_slot = tq
        self.txt_k_slot = tk
        self.txt_v_slot = tv
        self.txt_proj_slot = tp
        self.txt_mlp0_slot = tm0
        self.txt_mlp2_slot = tm2


struct FluxSingleBlockDirectLycoris(Copyable, Movable):
    var algo: Int
    var dora: FlatDirectDoRASet
    var oft: FlatDirectOFTSet
    var q_slot: Int
    var k_slot: Int
    var v_slot: Int
    var proj_mlp_slot: Int
    var linear2_slot: Int

    def __init__(
        out self, algo: Int, var dora: FlatDirectDoRASet,
        var oft: FlatDirectOFTSet, base_slot: Int, targets: Int,
    ) raises:
        var q = -1
        var k = -1
        var v = -1
        var pm = -1
        var l2 = -1
        var compact = base_slot
        for slot in range(SGL_SLOTS):
            if not _flux_direct_sgl_slot_targeted(slot, targets):
                continue
            if slot == S_SQ:
                q = compact
            elif slot == S_SK:
                k = compact
            elif slot == S_SV:
                v = compact
            elif slot == S_PMLP:
                pm = compact
            else:
                l2 = compact
            compact += 1
        self.algo = algo
        self.dora = dora^
        self.oft = oft^
        self.q_slot = q
        self.k_slot = k
        self.v_slot = v
        self.proj_mlp_slot = pm
        self.linear2_slot = l2


struct FluxDirectProjectionGrad(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_m: List[Float32]
    var d_vec: List[Float32]

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32],
        var d_m: List[Float32], var d_vec: List[Float32],
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_m = d_m^
        self.d_vec = d_vec^


struct _FluxDirectProjectionGradDev(Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_m: List[Float32]
    var d_vec: List[Float32]
    var d_x: TArc

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32],
        var d_m: List[Float32], var d_vec: List[Float32], var d_x: TArc,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_m = d_m^
        self.d_vec = d_vec^
        self.d_x = d_x^


def _flux_direct_grad_public(g: _FluxDirectProjectionGradDev) -> FluxDirectProjectionGrad:
    return FluxDirectProjectionGrad(
        g.d_a.copy(), g.d_b.copy(), g.d_m.copy(), g.d_vec.copy(),
    )


def _flux_empty_direct_grad() -> FluxDirectProjectionGrad:
    return FluxDirectProjectionGrad(_empty_f32(), _empty_f32(), _empty_f32(), _empty_f32())


struct FluxStreamDirectLycorisGrads(Copyable, Movable):
    var d_x: List[Float32]
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]
    var q: FluxDirectProjectionGrad
    var k: FluxDirectProjectionGrad
    var v: FluxDirectProjectionGrad
    var proj: FluxDirectProjectionGrad
    var mlp0: FluxDirectProjectionGrad
    var mlp2: FluxDirectProjectionGrad

    def __init__(
        out self, var d_x: List[Float32],
        var d_shift1: List[Float32], var d_scale1: List[Float32],
        var d_gate1: List[Float32], var d_shift2: List[Float32],
        var d_scale2: List[Float32], var d_gate2: List[Float32],
        var q: FluxDirectProjectionGrad, var k: FluxDirectProjectionGrad,
        var v: FluxDirectProjectionGrad, var proj: FluxDirectProjectionGrad,
        var mlp0: FluxDirectProjectionGrad, var mlp2: FluxDirectProjectionGrad,
    ):
        self.d_x = d_x^
        self.d_shift1 = d_shift1^
        self.d_scale1 = d_scale1^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^
        self.q = q^
        self.k = k^
        self.v = v^
        self.proj = proj^
        self.mlp0 = mlp0^
        self.mlp2 = mlp2^


struct FluxDoubleBlockDirectLycorisGrads(Copyable, Movable):
    var img: FluxStreamDirectLycorisGrads
    var txt: FluxStreamDirectLycorisGrads

    def __init__(
        out self,
        var img: FluxStreamDirectLycorisGrads,
        var txt: FluxStreamDirectLycorisGrads,
    ):
        self.img = img^
        self.txt = txt^


struct FluxSingleBlockDirectLycorisGrads(Copyable, Movable):
    var d_x: List[Float32]
    var d_shift: List[Float32]
    var d_scale: List[Float32]
    var d_gate: List[Float32]
    var q: FluxDirectProjectionGrad
    var k: FluxDirectProjectionGrad
    var v: FluxDirectProjectionGrad
    var proj_mlp: FluxDirectProjectionGrad
    var linear2: FluxDirectProjectionGrad

    def __init__(
        out self, var d_x: List[Float32],
        var d_shift: List[Float32], var d_scale: List[Float32],
        var d_gate: List[Float32],
        var q: FluxDirectProjectionGrad, var k: FluxDirectProjectionGrad,
        var v: FluxDirectProjectionGrad, var proj_mlp: FluxDirectProjectionGrad,
        var linear2: FluxDirectProjectionGrad,
    ):
        self.d_x = d_x^
        self.d_shift = d_shift^
        self.d_scale = d_scale^
        self.d_gate = d_gate^
        self.q = q^
        self.k = k^
        self.v = v^
        self.proj_mlp = proj_mlp^
        self.linear2 = linear2^


struct FluxDoubleBlockDirectLycorisForward(Copyable, Movable):
    var img_out: List[Float32]
    var txt_out: List[Float32]
    var saved: DoubleBlockSaved

    def __init__(
        out self, var img_out: List[Float32], var txt_out: List[Float32],
        var saved: DoubleBlockSaved,
    ):
        self.img_out = img_out^
        self.txt_out = txt_out^
        self.saved = saved^


struct FluxSingleBlockDirectLycorisForward(Copyable, Movable):
    var out: List[Float32]
    var saved: SingleBlockSaved

    def __init__(out self, var out: List[Float32], var saved: SingleBlockSaved):
        self.out = out^
        self.saved = saved^


def _flux_oft_vec_tensor(
    set: FlatDirectOFTSet, slot: Int, ctx: DeviceContext,
) raises -> Tensor:
    if slot < 0 or slot >= len(set.ad):
        raise Error("FluxBlockDirectLycoris OFT: slot out of range")
    if not set.active[slot]:
        raise Error("FluxBlockDirectLycoris OFT: inactive slot")
    ref sl = set.ad[slot]
    if sl.b != 4:
        raise Error("FluxBlockDirectLycoris OFT: only block_size=4 is wired on GPU")
    return Tensor.from_host(sl.vec.copy(), [sl.r, 6], STDtype.F32, ctx)


def _direct_proj_fwd_device(
    algo: Int, dora: FlatDirectDoRASet, oft: FlatDirectOFTSet, slot: Int,
    x: Tensor, w_orig: Tensor, bias: Tensor,
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> Tensor:
    if slot < 0:
        return linear(x, w_orig, Optional[Tensor](bias.clone(ctx)), ctx)
    if algo == FLUX_DIRECT_ALGO_DORA:
        if slot >= len(dora.ad):
            raise Error("FluxBlockDirectLycoris DoRA: slot out of range")
        if not dora.active[slot]:
            raise Error("FluxBlockDirectLycoris DoRA: inactive slot")
        if dora.ad[slot].in_f != in_f or dora.ad[slot].out_f != out_f:
            raise Error("FluxBlockDirectLycoris DoRA: slot shape mismatch")
        var dev = dora_device_from_host(dora.ad[slot], ctx)
        # Streamed synthetic/legacy checkpoints can present F32 block weights
        # while Flux block activations are BF16/F16. Direct DoRA kernels require
        # a supported x/w storage pair; projection compute follows x dtype.
        if x.dtype() != STDtype.F32 and w_orig.dtype() != x.dtype():
            var wc = cast_tensor(w_orig, x.dtype(), ctx)
            var y = dora_substitution_forward_device(x, wc, dev, ctx)
            if bias.dtype() != y.dtype():
                var bc = cast_tensor(bias, y.dtype(), ctx)
                return add(y^, bc, ctx)
            return add(y^, bias, ctx)
        var y = dora_substitution_forward_device(x, w_orig, dev, ctx)
        if bias.dtype() != y.dtype():
            var bc = cast_tensor(bias, y.dtype(), ctx)
            return add(y^, bc, ctx)
        return add(y^, bias, ctx)
    if algo == FLUX_DIRECT_ALGO_OFT:
        if slot >= len(oft.ad):
            raise Error("FluxBlockDirectLycoris OFT: slot out of range")
        if oft.ad[slot].in_f != in_f or oft.ad[slot].out_f != out_f:
            raise Error("FluxBlockDirectLycoris OFT: slot shape mismatch")
        var vec = _flux_oft_vec_tensor(oft, slot, ctx)
        var x_rot = oft_ot_rotate_b4(x, vec, ctx)
        return linear(x_rot, w_orig, Optional[Tensor](bias.clone(ctx)), ctx)
    raise Error("FluxBlockDirectLycoris: unsupported direct algorithm")


def _direct_proj_bwd_device(
    algo: Int, dora: FlatDirectDoRASet, oft: FlatDirectOFTSet, slot: Int,
    d_y: Tensor, x: Tensor, w_orig: Tensor,
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> _FluxDirectProjectionGradDev:
    if slot < 0:
        var dx = linear_backward_dx(d_y, w_orig, M, in_f, out_f, ctx)
        return _FluxDirectProjectionGradDev(
            _empty_f32(), _empty_f32(), _empty_f32(), _empty_f32(), TArc(dx^),
        )
    if algo == FLUX_DIRECT_ALGO_DORA:
        if slot >= len(dora.ad):
            raise Error("FluxBlockDirectLycoris DoRA backward: slot out of range")
        if not dora.active[slot]:
            raise Error("FluxBlockDirectLycoris DoRA backward: inactive slot")
        if dora.ad[slot].in_f != in_f or dora.ad[slot].out_f != out_f:
            raise Error("FluxBlockDirectLycoris DoRA backward: slot shape mismatch")
        var dev = dora_device_from_host(dora.ad[slot], ctx)
        if x.dtype() != STDtype.F32 and w_orig.dtype() != x.dtype():
            var wc = cast_tensor(w_orig, x.dtype(), ctx)
            var g = dora_substitution_backward_device(d_y, x, wc, dev, ctx)
            return _FluxDirectProjectionGradDev(
                g.d_a.to_host(ctx), g.d_b.to_host(ctx), g.d_m.to_host(ctx),
                _empty_f32(), TArc(g.d_x.clone(ctx)),
            )
        var g = dora_substitution_backward_device(d_y, x, w_orig, dev, ctx)
        return _FluxDirectProjectionGradDev(
            g.d_a.to_host(ctx), g.d_b.to_host(ctx), g.d_m.to_host(ctx),
            _empty_f32(), TArc(g.d_x.clone(ctx)),
        )
    if algo == FLUX_DIRECT_ALGO_OFT:
        if slot >= len(oft.ad):
            raise Error("FluxBlockDirectLycoris OFT backward: slot out of range")
        if oft.ad[slot].in_f != in_f or oft.ad[slot].out_f != out_f:
            raise Error("FluxBlockDirectLycoris OFT backward: slot shape mismatch")
        var vec = _flux_oft_vec_tensor(oft, slot, ctx)
        var d_x_rot = linear_backward_dx(d_y, w_orig, M, in_f, out_f, ctx)
        if d_x_rot.dtype() != x.dtype():
            d_x_rot = cast_tensor(d_x_rot^, x.dtype(), ctx)
        if d_x_rot.shape() != x.shape():
            d_x_rot = reshape_owned(d_x_rot^, x.shape())
        var g = oft_ot_rotate_backward_b4(d_x_rot^, x, vec, ctx)
        return _FluxDirectProjectionGradDev(
            _empty_f32(), _empty_f32(), _empty_f32(), g.d_vec.to_host(ctx),
            TArc(g.d_x.clone(ctx)),
        )
    raise Error("FluxBlockDirectLycoris: unsupported direct algorithm")


# proj-backward helper: base linear_backward d_x then add the LoRA branch's d_x
# (if present), collecting d_a/d_b into the slot lists. Returns the SUMMED d_x.
struct _ProjBwd(Movable):
    var d_x: Tensor
    var d_w: Tensor
    var d_b: Tensor

    def __init__(out self, var d_x: Tensor, var d_w: Tensor, var d_b: Tensor):
        self.d_x = d_x^
        self.d_w = d_w^
        self.d_b = d_b^


def _proj_bwd_with_lora(
    d_y: Tensor, x_in: Tensor, w: Tensor, x_in_h: List[Float32],
    lo: Optional[LoraAdapter], slot: Int,
    M: Int, in_f: Int, out_f: Int,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> _ProjBwd:
    var lb = linear_backward(d_y, x_in, w, M, in_f, out_f, ctx)
    var d_w = lb.d_w.clone(ctx)
    var d_b = lb.d_b.clone(ctx)
    if lo:
        var d_y_h = d_y.to_host(ctx)
        var lg = flux_lora_bwd(d_y_h, x_in_h, lo.value(), M, ctx)
        d_a_slots[slot] = lg.d_a.copy()
        d_b_slots[slot] = lg.d_b.copy()
        var base_dx = lb.d_x.to_host(ctx)
        var summed = _add_lists(base_dx, lg.d_x)
        return _ProjBwd(_t(summed, [M, in_f], ctx), d_w^, d_b^)
    var d_x = lb.d_x.clone(ctx)
    return _ProjBwd(d_x^, d_w^, d_b^)


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block LoRA forward — mirrors double_block_forward, injecting LoRA on
# each stream's q/k/v slices + proj + mlp0 + mlp2 BEFORE the downstream op.
# ═══════════════════════════════════════════════════════════════════════════
struct _StreamPreH(Movable):
    var q_rms: Tensor
    var k_rms: Tensor
    var v: Tensor
    var ln1_h: List[BFloat16]
    var norm_h: List[BFloat16]
    var q_pre_h: List[BFloat16]
    var k_pre_h: List[BFloat16]

    def __init__(
        out self, var q_rms: Tensor, var k_rms: Tensor, var v: Tensor,
        var ln1_h: List[BFloat16], var norm_h: List[BFloat16],
        var q_pre_h: List[BFloat16], var k_pre_h: List[BFloat16],
    ):
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^
        self.ln1_h = ln1_h^
        self.norm_h = norm_h^
        self.q_pre_h = q_pre_h^
        self.k_pre_h = k_pre_h^


def _stream_pre_lora[
    H: Int, Dh: Int
](
    x: Tensor, w: StreamWeights, mv: ModVecs, lo: StreamLora,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPreH:
    var ln1 = layer_norm(x, ones, zeros, eps, ctx)
    var norm = modulate(ln1, _t(mv.scale1.copy(), [D], ctx), _t(mv.shift1.copy(), [D], ctx), ctx)
    var norm_h_f32 = norm.to_host(ctx)                      # F32 for LoRA computation
    var b = Optional[Tensor](w.bqkv[].clone(ctx))
    var qkv = linear(norm, w.wqkv[], b, ctx)                # [N,3D]
    # base q/k/v slices
    var q_base = slice(qkv, 1, 0, D, ctx).to_host(ctx)
    var k_base = slice(qkv, 1, D, D, ctx).to_host(ctx)
    var v_base = slice(qkv, 1, 2 * D, D, ctx).to_host(ctx)
    # LoRA on each (separate to_q/to_k/to_v adapters; shared norm input)
    var q_h = flux_lora_apply(q_base, norm_h_f32, lo.to_q, N, ctx)
    var k_h = flux_lora_apply(k_base, norm_h_f32, lo.to_k, N, ctx)
    var v_h = flux_lora_apply(v_base, norm_h_f32, lo.to_v, N, ctx)
    var q_pre = reshape_owned(_t(q_h, [N, D], ctx)^, [1, N, H, Dh])
    var k_pre = reshape_owned(_t(k_h, [N, D], ctx)^, [1, N, H, Dh])
    var v = reshape_owned(_t(v_h, [N, D], ctx)^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPreH(
        q_rms^, k_rms^, v^,
        ln1.to_host_bf16(ctx), _f32_to_bf16(norm_h_f32),
        q_pre.to_host_bf16(ctx), k_pre.to_host_bf16(ctx),
    )


struct _StreamPostH(Movable):
    var out: Tensor
    var attn_res_h: List[BFloat16]
    var ln2_h: List[BFloat16]
    var mlp_in_h: List[BFloat16]
    var mlp_pre_h: List[BFloat16]
    var mlp_h_h: List[BFloat16]

    def __init__(
        out self, var out: Tensor, var attn_res_h: List[BFloat16],
        var ln2_h: List[BFloat16], var mlp_in_h: List[BFloat16],
        var mlp_pre_h: List[BFloat16], var mlp_h_h: List[BFloat16],
    ):
        self.out = out^
        self.attn_res_h = attn_res_h^
        self.ln2_h = ln2_h^
        self.mlp_in_h = mlp_in_h^
        self.mlp_pre_h = mlp_pre_h^
        self.mlp_h_h = mlp_h_h^


def _stream_post_lora(
    x: Tensor, att: Tensor, att_h_f32: List[Float32],
    w: StreamWeights, mv: ModVecs, lo: StreamLora,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor, zeros: Tensor,
    ctx: DeviceContext,
) raises -> _StreamPostH:
    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var out_base = linear(att, w.wproj[], bp, ctx).to_host(ctx)   # [N,D]
    var out_h = flux_lora_apply(out_base, att_h_f32, lo.proj, N, ctx)
    var out = _t(out_h, [N, D], ctx)
    var attn_res = residual_gate(x, _t(mv.gate1.copy(), [D], ctx), out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, _t(mv.scale2.copy(), [D], ctx), _t(mv.shift2.copy(), [D], ctx), ctx)
    var mlp_in_h_f32 = mlp_in.to_host(ctx)                       # F32 for LoRA computation
    var b0 = Optional[Tensor](w.bmlp0[].clone(ctx))
    var mlp_pre_base = linear(mlp_in, w.wmlp0[], b0, ctx).to_host(ctx)   # [N,Fmlp]
    var mlp_pre_h = flux_lora_apply(mlp_pre_base, mlp_in_h_f32, lo.mlp0, N, ctx)
    var mlp_pre = _t(mlp_pre_h, [N, Fmlp], ctx)
    var mlp_h = gelu(mlp_pre, ctx)                               # [N,Fmlp]
    var mlp_h_h_f32 = mlp_h.to_host(ctx)                         # F32 for LoRA computation
    var b2 = Optional[Tensor](w.bmlp2[].clone(ctx))
    var mlp_base = linear(mlp_h, w.wmlp2[], b2, ctx).to_host(ctx)   # [N,D]
    var mlp_out_h = flux_lora_apply(mlp_base, mlp_h_h_f32, lo.mlp2, N, ctx)
    var mlp = _t(mlp_out_h, [N, D], ctx)
    var final = residual_gate(attn_res, _t(mv.gate2.copy(), [D], ctx), mlp, ctx)
    return _StreamPostH(
        final^, attn_res.to_host_bf16(ctx), ln2.to_host_bf16(ctx),
        _f32_to_bf16(mlp_in_h_f32), _f32_to_bf16(mlp_pre_h), _f32_to_bf16(mlp_h_h_f32),
    )


def _stream_pre_direct[
    H: Int, Dh: Int
](
    x: Tensor, w: StreamWeights, mv: ModVecs,
    direct: FluxDoubleBlockDirectLycoris, q_slot: Int, k_slot: Int, v_slot: Int,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPreH:
    var ln1 = layer_norm(x, ones, zeros, eps, ctx)
    var norm = modulate(ln1, _t(mv.scale1.copy(), [D], ctx), _t(mv.shift1.copy(), [D], ctx), ctx)
    var ln1_h = ln1.to_host_bf16(ctx)
    var norm_h = norm.to_host_bf16(ctx)

    var q_w = slice(w.wqkv[], 0, 0, D, ctx)
    var k_w = slice(w.wqkv[], 0, D, D, ctx)
    var v_w = slice(w.wqkv[], 0, 2 * D, D, ctx)
    var q_b = slice(w.bqkv[], 0, 0, D, ctx)
    var k_b = slice(w.bqkv[], 0, D, D, ctx)
    var v_b = slice(w.bqkv[], 0, 2 * D, D, ctx)
    var q_flat = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, q_slot, norm, q_w, q_b, N, D, D, ctx,
    )
    var k_flat = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, k_slot, norm, k_w, k_b, N, D, D, ctx,
    )
    var v_flat = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, v_slot, norm, v_w, v_b, N, D, D, ctx,
    )
    var q_pre = reshape_owned(q_flat^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, N, H, Dh])
    var v = reshape_owned(v_flat^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPreH(
        q_rms^, k_rms^, v^,
        ln1_h^, norm_h^, q_pre.to_host_bf16(ctx), k_pre.to_host_bf16(ctx),
    )


def _stream_post_direct(
    x: Tensor, att: Tensor,
    w: StreamWeights, mv: ModVecs, direct: FluxDoubleBlockDirectLycoris,
    proj_slot: Int, mlp0_slot: Int, mlp2_slot: Int,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor, zeros: Tensor,
    ctx: DeviceContext,
) raises -> _StreamPostH:
    var out = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, proj_slot,
        att, w.wproj[], w.bproj[], N, D, D, ctx,
    )
    var attn_res = residual_gate(x, _t(mv.gate1.copy(), [D], ctx), out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(
        ln2, _t(mv.scale2.copy(), [D], ctx), _t(mv.shift2.copy(), [D], ctx), ctx
    )
    var mlp_pre = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, mlp0_slot,
        mlp_in, w.wmlp0[], w.bmlp0[], N, D, Fmlp, ctx,
    )
    var mlp_h = gelu(mlp_pre, ctx)
    var mlp_h_h = mlp_h.to_host_bf16(ctx)
    var mlp = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, mlp2_slot,
        mlp_h, w.wmlp2[], w.bmlp2[], N, Fmlp, D, ctx,
    )
    var final = residual_gate(attn_res, _t(mv.gate2.copy(), [D], ctx), mlp, ctx)
    return _StreamPostH(
        final^, attn_res.to_host_bf16(ctx), ln2.to_host_bf16(ctx),
        mlp_in.to_host_bf16(ctx), mlp_pre.to_host_bf16(ctx), mlp_h_h^,
    )


def double_block_direct_lycoris_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img: List[Float32], txt: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs,
    direct: FluxDoubleBlockDirectLycoris,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxDoubleBlockDirectLycorisForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)
    var cos_b = _to_bf16(cos, ctx)
    var sin_b = _to_bf16(sin, ctx)

    var img_x = _t(img, [N_IMG, D], ctx)
    var txt_x = _t(txt, [N_TXT, D], ctx)

    var ip = _stream_pre_direct[H, Dh](
        img_x, w.img, img_mod, direct,
        direct.img_q_slot, direct.img_k_slot, direct.img_v_slot,
        N_IMG, D, eps, ones_t, zeros_t, ctx,
    )
    var tp = _stream_pre_direct[H, Dh](
        txt_x, w.txt, txt_mod, direct,
        direct.txt_q_slot, direct.txt_k_slot, direct.txt_v_slot,
        N_TXT, D, eps, ones_t, zeros_t, ctx,
    )

    var q = concat(1, ctx, tp.q_rms, ip.q_rms)
    var k = concat(1, ctx, tp.k_rms, ip.k_rms)
    var v = concat(1, ctx, tp.v, ip.v)

    var q_rope = rope_interleaved(q, cos_b, sin_b, ctx)
    var k_rope = rope_interleaved(k, cos_b, sin_b, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = reshape_owned(txt_att_4d^, [N_TXT, D])
    var img_att = reshape_owned(img_att_4d^, [N_IMG, D])

    var ipost = _stream_post_direct(
        img_x, img_att, w.img, img_mod, direct,
        direct.img_proj_slot, direct.img_mlp0_slot, direct.img_mlp2_slot,
        N_IMG, D, Fmlp, eps, ones_t, zeros_t, ctx,
    )
    var tpost = _stream_post_direct(
        txt_x, txt_att, w.txt, txt_mod, direct,
        direct.txt_proj_slot, direct.txt_mlp0_slot, direct.txt_mlp2_slot,
        N_TXT, D, Fmlp, eps, ones_t, zeros_t, ctx,
    )

    var img_saved = StreamSaved(
        _f32_to_bf16(img), ip.ln1_h.copy(), ip.norm_h.copy(),
        ip.q_pre_h.copy(), ip.k_pre_h.copy(),
        img_att.to_host_bf16(ctx), ipost.attn_res_h.copy(),
        ipost.ln2_h.copy(), ipost.mlp_in_h.copy(),
        ipost.mlp_pre_h.copy(), ipost.mlp_h_h.copy(),
    )
    var txt_saved = StreamSaved(
        _f32_to_bf16(txt), tp.ln1_h.copy(), tp.norm_h.copy(),
        tp.q_pre_h.copy(), tp.k_pre_h.copy(),
        txt_att.to_host_bf16(ctx), tpost.attn_res_h.copy(),
        tpost.ln2_h.copy(), tpost.mlp_in_h.copy(),
        tpost.mlp_pre_h.copy(), tpost.mlp_h_h.copy(),
    )
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^,
        q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx), v.to_host_bf16(ctx),
    )

    var img_out = ipost.out.to_host(ctx)
    var txt_out = tpost.out.to_host(ctx)
    return FluxDoubleBlockDirectLycorisForward(img_out^, txt_out^, saved^)


def double_block_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img: List[Float32], txt: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs, lora: DoubleBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)
    var cos_b = _to_bf16(cos, ctx)
    var sin_b = _to_bf16(sin, ctx)

    var img_x = _t(img, [N_IMG, D], ctx)
    var txt_x = _t(txt, [N_TXT, D], ctx)

    var ip = _stream_pre_lora[H, Dh](img_x, w.img, img_mod, lora.img, N_IMG, D, eps, ones_t, zeros_t, ctx)
    var tp = _stream_pre_lora[H, Dh](txt_x, w.txt, txt_mod, lora.txt, N_TXT, D, eps, ones_t, zeros_t, ctx)

    var q = concat(1, ctx, tp.q_rms, ip.q_rms)   # [1,S,H,Dh] txt FIRST
    var k = concat(1, ctx, tp.k_rms, ip.k_rms)
    var v = concat(1, ctx, tp.v, ip.v)

    var q_rope = rope_interleaved(q, cos_b, sin_b, ctx)
    var k_rope = rope_interleaved(k, cos_b, sin_b, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = reshape_owned(txt_att_4d^, [N_TXT, D])
    var img_att = reshape_owned(img_att_4d^, [N_IMG, D])
    var img_att_h_f32 = img_att.to_host(ctx)                     # F32 for proj LoRA computation
    var txt_att_h_f32 = txt_att.to_host(ctx)

    var ipost = _stream_post_lora(img_x, img_att, img_att_h_f32, w.img, img_mod, lora.img, N_IMG, D, Fmlp, eps, ones_t, zeros_t, ctx)
    var tpost = _stream_post_lora(txt_x, txt_att, txt_att_h_f32, w.txt, txt_mod, lora.txt, N_TXT, D, Fmlp, eps, ones_t, zeros_t, ctx)

    var img_saved = StreamSaved(
        _f32_to_bf16(img), ip.ln1_h.copy(), ip.norm_h.copy(),
        ip.q_pre_h.copy(), ip.k_pre_h.copy(),
        _f32_to_bf16(img_att_h_f32), ipost.attn_res_h.copy(),
        ipost.ln2_h.copy(), ipost.mlp_in_h.copy(),
        ipost.mlp_pre_h.copy(), ipost.mlp_h_h.copy(),
    )
    var txt_saved = StreamSaved(
        _f32_to_bf16(txt), tp.ln1_h.copy(), tp.norm_h.copy(),
        tp.q_pre_h.copy(), tp.k_pre_h.copy(),
        _f32_to_bf16(txt_att_h_f32), tpost.attn_res_h.copy(),
        tpost.ln2_h.copy(), tpost.mlp_in_h.copy(),
        tpost.mlp_pre_h.copy(), tpost.mlp_h_h.copy(),
    )
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^,
        q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx), v.to_host_bf16(ctx),
    )

    var img_out = ipost.out.to_host(ctx)
    var txt_out = tpost.out.to_host(ctx)
    return DoubleBlockForward(img_out^, txt_out^, saved^)


# ── per-stream post backward (LoRA-aware: proj, mlp0, mlp2) ──────────────────
struct _StreamPostBackL(Movable):
    var d_x: List[Float32]
    var d_att: List[Float32]
    var d_wproj: List[Float32]
    var d_bproj: List[Float32]
    var d_wmlp0: List[Float32]
    var d_bmlp0: List[Float32]
    var d_wmlp2: List[Float32]
    var d_bmlp2: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]

    def __init__(
        out self, var d_x: List[Float32], var d_att: List[Float32],
        var d_wproj: List[Float32], var d_bproj: List[Float32],
        var d_wmlp0: List[Float32], var d_bmlp0: List[Float32],
        var d_wmlp2: List[Float32], var d_bmlp2: List[Float32],
        var d_gate1: List[Float32],
        var d_shift2: List[Float32], var d_scale2: List[Float32], var d_gate2: List[Float32],
    ):
        self.d_x = d_x^
        self.d_att = d_att^
        self.d_wproj = d_wproj^
        self.d_bproj = d_bproj^
        self.d_wmlp0 = d_wmlp0^
        self.d_bmlp0 = d_bmlp0^
        self.d_wmlp2 = d_wmlp2^
        self.d_bmlp2 = d_bmlp2^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^


def _stream_post_backward_lora(
    d_out: Tensor, x: Tensor, att: Tensor,
    w: StreamWeights, mv: ModVecs, sv: StreamSaved, lo: StreamLora,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> _StreamPostBackL:
    # Saved acts re-uploaded BF16 (native bf16 backward matmuls).
    # d_out is F32 (upstream grad); gate_residual_backward needs grad_out + gate F32.
    var attn_res_t = _tb16(sv.attn_res.copy(), [N, D], ctx)
    var mlp_h_b = _tb16(sv.mlp_h.copy(), [N, Fmlp], ctx)
    # recompute mlp output WITH LoRA(mlp2) so gate_residual_backward y matches fwd.
    var b2 = Optional[Tensor](w.bmlp2[].clone(ctx))
    var mlp_base = linear(mlp_h_b, w.wmlp2[], b2, ctx).to_host(ctx)
    var mlp_h_vals_f32 = List[Float32]()
    for i in range(len(sv.mlp_h)):
        mlp_h_vals_f32.append(Float32(sv.mlp_h[i]))
    var mlp_y_h = flux_lora_apply(mlp_base, mlp_h_vals_f32, lo.mlp2, N, ctx)
    var mlp_y = _t(mlp_y_h, [N, D], ctx)              # bf16
    # F32-ONLY: gate_residual_backward grad_out/gate F32 (x/y may be bf16).
    var grg2 = gate_residual_backward(
        d_out, attn_res_t, _tf32(mv.gate2.copy(), [D], ctx), mlp_y, ctx
    )
    var d_gate2 = grg2.d_g.to_host(ctx)

    # mlp = linear(mlp_h, Wmlp2)[+LoRA(mlp2)]  W [D, Fmlp]. grg2.d_y F32 → bf16.
    var pm2 = _proj_bwd_with_lora(
        _to_bf16(grg2.d_y, ctx), mlp_h_b, w.wmlp2[], mlp_h_vals_f32, lo.mlp2, D_MLP2, N, Fmlp, D,
        d_a_slots, d_b_slots, ctx,
    )
    var d_wmlp2 = pm2.d_w.to_host(ctx)
    var d_bmlp2 = pm2.d_b.to_host(ctx)

    # mlp_h = gelu(mlp_pre)
    var mlp_pre_t = _tb16(sv.mlp_pre.copy(), [N, Fmlp], ctx)
    var d_mlp_pre = gelu_backward(pm2.d_x, mlp_pre_t, ctx)  # bf16

    # mlp_pre = linear(mlp_in, Wmlp0)[+LoRA(mlp0)]  W [Fmlp, D]
    var mlp_in_t = _tb16(sv.mlp_in.copy(), [N, D], ctx)
    var mlp_in_vals_f32 = List[Float32]()
    for i in range(len(sv.mlp_in)):
        mlp_in_vals_f32.append(Float32(sv.mlp_in[i]))
    var pm0 = _proj_bwd_with_lora(
        d_mlp_pre, mlp_in_t, w.wmlp0[], mlp_in_vals_f32, lo.mlp0, D_MLP0, N, D, Fmlp,
        d_a_slots, d_b_slots, ctx,
    )
    var d_wmlp0 = pm0.d_w.to_host(ctx)
    var d_bmlp0 = pm0.d_b.to_host(ctx)

    # mlp_in = modulate(ln2, scale2, shift2)  (pm0.d_x, ln2_t, scale2 all bf16)
    var ln2_t = _tb16(sv.ln2.copy(), [N, D], ctx)
    var mb2 = modulate_backward(pm0.d_x, ln2_t, _t(mv.scale2.copy(), [D], ctx), ctx)
    var d_scale2 = mb2.d_scale.to_host(ctx)
    var d_shift2 = mb2.d_shift.to_host(ctx)

    var lnb2 = layer_norm_backward(mb2.d_x, attn_res_t, ones, eps, ctx)  # bf16
    # attn_res feeds residual (grg2.d_x F32) AND ln2 (bf16) → SUM bf16, then F32 for gate bwd.
    var d_attn_res_total = add(_to_bf16(grg2.d_x, ctx), lnb2.d_x, ctx)
    var d_attn_res_f32 = _to_f32(d_attn_res_total, ctx)

    # attn_res = residual_gate(x, gate1, proj_out): recompute proj WITH LoRA(proj)
    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var proj_base = linear(att, w.wproj[], bp, ctx).to_host(ctx)
    var att_h = att.to_host(ctx)
    var proj_y_h = flux_lora_apply(proj_base, att_h, lo.proj, N, ctx)
    var proj_out = _t(proj_y_h, [N, D], ctx)         # bf16
    var grg1 = gate_residual_backward(
        d_attn_res_f32, x, _tf32(mv.gate1.copy(), [D], ctx), proj_out, ctx
    )
    var d_gate1 = grg1.d_g.to_host(ctx)
    var d_x_res = grg1.d_x.to_host(ctx)

    # proj_out = linear(att, Wproj)[+LoRA(proj)]  W [D, D]. grg1.d_y F32 → bf16.
    var pproj = _proj_bwd_with_lora(
        _to_bf16(grg1.d_y, ctx), att, w.wproj[], att_h, lo.proj, D_PROJ, N, D, D,
        d_a_slots, d_b_slots, ctx,
    )
    var d_wproj = pproj.d_w.to_host(ctx)
    var d_bproj = pproj.d_b.to_host(ctx)
    var d_att = pproj.d_x.to_host(ctx)

    return _StreamPostBackL(
        d_x_res^, d_att^, d_wproj^, d_bproj^,
        d_wmlp0^, d_bmlp0^, d_wmlp2^, d_bmlp2^,
        d_gate1^, d_shift2^, d_scale2^, d_gate2^,
    )


# ── per-stream pre backward (LoRA-aware: to_q, to_k, to_v) ───────────────────
struct _StreamPreBackL(Movable):
    var d_x: List[Float32]
    var d_wqkv: List[Float32]
    var d_bqkv: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]

    def __init__(
        out self, var d_x: List[Float32],
        var d_wqkv: List[Float32], var d_bqkv: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_shift1: List[Float32], var d_scale1: List[Float32],
    ):
        self.d_x = d_x^
        self.d_wqkv = d_wqkv^
        self.d_bqkv = d_bqkv^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_shift1 = d_shift1^
        self.d_scale1 = d_scale1^


def _stream_pre_backward_lora[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecs, sv: StreamSaved, lo: StreamLora,
    N: Int, D: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> _StreamPreBackL:
    # Incoming grads (cat_backward outputs) F32 → bf16 for the bf16-native chain.
    var dq_b = _to_bf16(d_q_rms, ctx)
    var dk_b = _to_bf16(d_k_rms, ctx)
    var dv_b = _to_bf16(d_v, ctx)
    var q_pre_t = _tb16(sv.q_pre.copy(), [1, N, H, Dh], ctx)
    var k_pre_t = _tb16(sv.k_pre.copy(), [1, N, H, Dh], ctx)
    var rb_q = rms_norm_backward(dq_b, q_pre_t, w.q_norm[], eps, ctx)   # bf16
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(dk_b, k_pre_t, w.k_norm[], eps, ctx)   # bf16
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [N, D])
    reshape_in_place(rb_k.d_x, [N, D])
    var d_v_flat = reshape(dv_b, [N, D], ctx)

    # The fused qkv linear's base d_w/d_b come from the joined d_qkv [N,3D]; the
    # LoRA on to_q/to_k/to_v consumes the per-slice d_y (rb_q.d_x / rb_k.d_x /
    # d_v_flat) against the SHARED input `norm`. d_x_lo from all three slices SUMS
    # into the base norm grad (LoRA contribution to the projection input).
    var norm_t = _tb16(sv.norm.copy(), [N, D], ctx)
    var d_qkv = concat(1, ctx, rb_q.d_x, rb_k.d_x, d_v_flat)   # [N,3D]
    var lb_qkv = linear_backward(d_qkv, norm_t, w.wqkv[], N, D, 3 * D, ctx)
    var d_wqkv = lb_qkv.d_w.to_host(ctx)
    var d_bqkv = lb_qkv.d_b.to_host(ctx)
    var d_norm = lb_qkv.d_x.to_host(ctx)   # base norm grad [N,D]

    # to_q / to_k / to_v LoRA: each consumes its own d_y slice, input = norm (F32).
    var norm_vals_f32 = List[Float32]()
    for i in range(len(sv.norm)):
        norm_vals_f32.append(Float32(sv.norm[i]))
    var d_q_h = rb_q.d_x.to_host(ctx)
    var d_k_h = rb_k.d_x.to_host(ctx)
    var d_v_h = d_v_flat.to_host(ctx)
    if lo.to_q:
        var lg = flux_lora_bwd(d_q_h, norm_vals_f32, lo.to_q.value(), N, ctx)
        d_a_slots[D_SQ] = lg.d_a.copy()
        d_b_slots[D_SQ] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)
    if lo.to_k:
        var lg = flux_lora_bwd(d_k_h, norm_vals_f32, lo.to_k.value(), N, ctx)
        d_a_slots[D_SK] = lg.d_a.copy()
        d_b_slots[D_SK] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)
    if lo.to_v:
        var lg = flux_lora_bwd(d_v_h, norm_vals_f32, lo.to_v.value(), N, ctx)
        d_a_slots[D_SV] = lg.d_a.copy()
        d_b_slots[D_SV] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)

    # norm = modulate(ln1, scale1, shift1)
    var ln1_t = _tb16(sv.ln1.copy(), [N, D], ctx)
    var mb1 = modulate_backward(_t(d_norm, [N, D], ctx), ln1_t, _t(mv.scale1.copy(), [D], ctx), ctx)
    var d_scale1 = mb1.d_scale.to_host(ctx)
    var d_shift1 = mb1.d_shift.to_host(ctx)

    var x_t = _tb16(sv.x.copy(), [N, D], ctx)
    var lnb1 = layer_norm_backward(mb1.d_x, x_t, ones, eps, ctx)
    var d_x_norm = lnb1.d_x.to_host(ctx)
    return _StreamPreBackL(d_x_norm^, d_wqkv^, d_bqkv^, d_q_norm^, d_k_norm^, d_shift1^, d_scale1^)


def double_block_lora_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_img_out: List[Float32], d_txt_out: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs, lora: DoubleBlockLora,
    saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockLoraBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)

    # Upstream output grads stay F32 — gate_residual_backward needs grad_out F32.
    var d_io_t = _tf32(d_img_out, [N_IMG, D], ctx)
    var d_to_t = _tf32(d_txt_out, [N_TXT, D], ctx)

    var img_x = _tb16(saved.img.x.copy(), [N_IMG, D], ctx)
    var txt_x = _tb16(saved.txt.x.copy(), [N_TXT, D], ctx)
    var img_att = _tb16(saved.img.att.copy(), [N_IMG, D], ctx)
    var txt_att = _tb16(saved.txt.att.copy(), [N_TXT, D], ctx)

    # slot lists per stream
    var ia = List[List[Float32]]()
    var ib = List[List[Float32]]()
    var ta = List[List[Float32]]()
    var tb = List[List[Float32]]()
    for _ in range(DBL_STREAM_SLOTS):
        ia.append(List[Float32]()); ib.append(List[Float32]())
        ta.append(List[Float32]()); tb.append(List[Float32]())

    var ipb = _stream_post_backward_lora(
        d_io_t, img_x, img_att, w.img, img_mod, saved.img, lora.img,
        N_IMG, D, Fmlp, eps, ones_t, ia, ib, ctx,
    )
    var tpb = _stream_post_backward_lora(
        d_to_t, txt_x, txt_att, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT, D, Fmlp, eps, ones_t, ta, tb, ctx,
    )

    # join per-stream attention-slice grads into joint d_att (txt FIRST).
    # _t uploads bf16 → matches bf16 q_rope/k_rope/v for sdpa_backward.
    var d_tatt_4d = _t(tpb.d_att.copy(), [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = _t(ipb.d_att.copy(), [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh] bf16

    # sdpa backward (JOINT) — bf16-native.
    var q_rope_t = _tb16(saved.q_rope.copy(), [1, S, H, Dh], ctx)
    var k_rope_t = _tb16(saved.k_rope.copy(), [1, S, H, Dh], ctx)
    var v_joint_t = _tb16(saved.v_joint.copy(), [1, S, H, Dh], ctx)
    var sb = sdpa_backward[1, S, H, Dh](q_rope_t, k_rope_t, v_joint_t, d_att_joint, scale, ctx)

    # F32-ONLY: rope_backward grad + cos/sin F32. sb.d_q/d_k bf16 → cast up.
    var d_q_joint = rope_backward(_to_f32(sb.d_q, ctx), cos, sin, True, ctx)
    var d_k_joint = rope_backward(_to_f32(sb.d_k, ctx), cos, sin, True, ctx)

    # F32-ONLY: cat_backward grad F32. sb.d_v bf16 → cast up.
    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(_to_f32(sb.d_v, ctx), N_TXT, N_IMG, 1, ctx)

    var iprb = _stream_pre_backward_lora[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, saved.img, lora.img,
        N_IMG, D, eps, ones_t, ia, ib, ctx,
    )
    var tprb = _stream_pre_backward_lora[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT, D, eps, ones_t, ta, tb, ctx,
    )

    var d_img_x = _add_lists(ipb.d_x, iprb.d_x)
    var d_txt_x = _add_lists(tpb.d_x, tprb.d_x)

    var img_grads = StreamGrads(
        d_img_x^,
        iprb.d_wqkv.copy(), iprb.d_bqkv.copy(),
        ipb.d_wproj.copy(), ipb.d_bproj.copy(),
        ipb.d_wmlp0.copy(), ipb.d_bmlp0.copy(),
        ipb.d_wmlp2.copy(), ipb.d_bmlp2.copy(),
        iprb.d_q_norm.copy(), iprb.d_k_norm.copy(),
        iprb.d_shift1.copy(), iprb.d_scale1.copy(), ipb.d_gate1.copy(),
        ipb.d_shift2.copy(), ipb.d_scale2.copy(), ipb.d_gate2.copy(),
    )
    var txt_grads = StreamGrads(
        d_txt_x^,
        tprb.d_wqkv.copy(), tprb.d_bqkv.copy(),
        tpb.d_wproj.copy(), tpb.d_bproj.copy(),
        tpb.d_wmlp0.copy(), tpb.d_bmlp0.copy(),
        tpb.d_wmlp2.copy(), tpb.d_bmlp2.copy(),
        tprb.d_q_norm.copy(), tprb.d_k_norm.copy(),
        tprb.d_shift1.copy(), tprb.d_scale1.copy(), tpb.d_gate1.copy(),
        tpb.d_shift2.copy(), tpb.d_scale2.copy(), tpb.d_gate2.copy(),
    )
    return DoubleBlockLoraBackward(
        DoubleBlockGrads(img_grads^, txt_grads^),
        DoubleBlockLoraGrads(StreamLoraGrads(ia^, ib^), StreamLoraGrads(ta^, tb^)),
    )


struct _StreamPostBackDirect(Movable):
    var d_x: List[Float32]
    var d_att: Tensor
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]
    var proj_g: _FluxDirectProjectionGradDev
    var mlp0_g: _FluxDirectProjectionGradDev
    var mlp2_g: _FluxDirectProjectionGradDev

    def __init__(
        out self, var d_x: List[Float32], var d_att: Tensor,
        var d_gate1: List[Float32], var d_shift2: List[Float32],
        var d_scale2: List[Float32], var d_gate2: List[Float32],
        var proj_g: _FluxDirectProjectionGradDev,
        var mlp0_g: _FluxDirectProjectionGradDev,
        var mlp2_g: _FluxDirectProjectionGradDev,
    ):
        self.d_x = d_x^
        self.d_att = d_att^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^
        self.proj_g = proj_g^
        self.mlp0_g = mlp0_g^
        self.mlp2_g = mlp2_g^


def _stream_post_backward_direct(
    d_out: Tensor, w: StreamWeights, mv: ModVecs,
    direct: FluxDoubleBlockDirectLycoris, sv: StreamSaved,
    proj_slot: Int, mlp0_slot: Int, mlp2_slot: Int,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor,
    ctx: DeviceContext,
) raises -> _StreamPostBackDirect:
    var x_t = _tb16(sv.x.copy(), [N, D], ctx)
    var att_t = _tb16(sv.att.copy(), [N, D], ctx)
    var attn_res_t = _tb16(sv.attn_res.copy(), [N, D], ctx)
    var mlp_h_b = _tb16(sv.mlp_h.copy(), [N, Fmlp], ctx)
    var mlp_pre_t = _tb16(sv.mlp_pre.copy(), [N, Fmlp], ctx)
    var mlp_in_t = _tb16(sv.mlp_in.copy(), [N, D], ctx)

    var mlp_y = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, mlp2_slot,
        mlp_h_b, w.wmlp2[], w.bmlp2[], N, Fmlp, D, ctx,
    )
    var grg2 = gate_residual_backward(
        d_out, attn_res_t, _tf32(mv.gate2.copy(), [D], ctx), mlp_y, ctx,
    )
    var d_gate2 = grg2.d_g.to_host(ctx)
    var mlp2_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, mlp2_slot,
        _to_bf16(grg2.d_y, ctx), mlp_h_b, w.wmlp2[], N, Fmlp, D, ctx,
    )
    var d_mlp_pre = gelu_backward(mlp2_g.d_x[], mlp_pre_t, ctx)
    var mlp0_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, mlp0_slot,
        d_mlp_pre, mlp_in_t, w.wmlp0[], N, D, Fmlp, ctx,
    )

    var mb2 = modulate_backward(
        mlp0_g.d_x[], _tb16(sv.ln2.copy(), [N, D], ctx),
        _t(mv.scale2.copy(), [D], ctx), ctx,
    )
    var d_scale2 = mb2.d_scale.to_host(ctx)
    var d_shift2 = mb2.d_shift.to_host(ctx)
    var lnb2 = layer_norm_backward(mb2.d_x, attn_res_t, ones, eps, ctx)
    var d_attn_res_total = add(_to_bf16(grg2.d_x, ctx), lnb2.d_x, ctx)

    var proj_y = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, proj_slot,
        att_t, w.wproj[], w.bproj[], N, D, D, ctx,
    )
    var grg1 = gate_residual_backward(
        _to_f32(d_attn_res_total, ctx), x_t, _tf32(mv.gate1.copy(), [D], ctx),
        proj_y, ctx,
    )
    var d_gate1 = grg1.d_g.to_host(ctx)
    var proj_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, proj_slot,
        _to_bf16(grg1.d_y, ctx), att_t, w.wproj[], N, D, D, ctx,
    )
    return _StreamPostBackDirect(
        grg1.d_x.to_host(ctx), proj_g.d_x[].clone(ctx),
        d_gate1^, d_shift2^, d_scale2^, d_gate2^,
        proj_g^, mlp0_g^, mlp2_g^,
    )


struct _StreamPreBackDirect(Movable):
    var d_x: List[Float32]
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]
    var q_g: _FluxDirectProjectionGradDev
    var k_g: _FluxDirectProjectionGradDev
    var v_g: _FluxDirectProjectionGradDev

    def __init__(
        out self, var d_x: List[Float32],
        var d_shift1: List[Float32], var d_scale1: List[Float32],
        var q_g: _FluxDirectProjectionGradDev,
        var k_g: _FluxDirectProjectionGradDev,
        var v_g: _FluxDirectProjectionGradDev,
    ):
        self.d_x = d_x^
        self.d_shift1 = d_shift1^
        self.d_scale1 = d_scale1^
        self.q_g = q_g^
        self.k_g = k_g^
        self.v_g = v_g^


def _stream_pre_backward_direct[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecs, direct: FluxDoubleBlockDirectLycoris,
    sv: StreamSaved, q_slot: Int, k_slot: Int, v_slot: Int,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPreBackDirect:
    var dq_b = _to_bf16(d_q_rms, ctx)
    var dk_b = _to_bf16(d_k_rms, ctx)
    var dv_b = _to_bf16(d_v, ctx)
    var q_pre_t = _tb16(sv.q_pre.copy(), [1, N, H, Dh], ctx)
    var k_pre_t = _tb16(sv.k_pre.copy(), [1, N, H, Dh], ctx)
    var rb_q = rms_norm_backward(dq_b, q_pre_t, w.q_norm[], eps, ctx)
    var rb_k = rms_norm_backward(dk_b, k_pre_t, w.k_norm[], eps, ctx)

    reshape_in_place(rb_q.d_x, [N, D])
    reshape_in_place(rb_k.d_x, [N, D])
    var d_v_flat = reshape(dv_b, [N, D], ctx)

    var norm_t = _tb16(sv.norm.copy(), [N, D], ctx)
    var q_w = slice(w.wqkv[], 0, 0, D, ctx)
    var k_w = slice(w.wqkv[], 0, D, D, ctx)
    var v_w = slice(w.wqkv[], 0, 2 * D, D, ctx)
    var q_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, q_slot,
        rb_q.d_x, norm_t, q_w, N, D, D, ctx,
    )
    var k_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, k_slot,
        rb_k.d_x, norm_t, k_w, N, D, D, ctx,
    )
    var v_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, v_slot,
        d_v_flat, norm_t, v_w, N, D, D, ctx,
    )
    var d_norm = add(add(q_g.d_x[], k_g.d_x[], ctx), v_g.d_x[], ctx)
    var mb1 = modulate_backward(
        d_norm, _tb16(sv.ln1.copy(), [N, D], ctx),
        _t(mv.scale1.copy(), [D], ctx), ctx,
    )
    var d_scale1 = mb1.d_scale.to_host(ctx)
    var d_shift1 = mb1.d_shift.to_host(ctx)
    var lnb1 = layer_norm_backward(
        mb1.d_x, _tb16(sv.x.copy(), [N, D], ctx), ones, eps, ctx,
    )
    return _StreamPreBackDirect(
        lnb1.d_x.to_host(ctx), d_shift1^, d_scale1^, q_g^, k_g^, v_g^,
    )


def double_block_direct_lycoris_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_img_out: List[Float32], d_txt_out: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs,
    direct: FluxDoubleBlockDirectLycoris, saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxDoubleBlockDirectLycorisGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)

    var d_io_t = _tf32(d_img_out, [N_IMG, D], ctx)
    var d_to_t = _tf32(d_txt_out, [N_TXT, D], ctx)

    var ipb = _stream_post_backward_direct(
        d_io_t, w.img, img_mod, direct, saved.img,
        direct.img_proj_slot, direct.img_mlp0_slot, direct.img_mlp2_slot,
        N_IMG, D, Fmlp, eps, ones_t, ctx,
    )
    var tpb = _stream_post_backward_direct(
        d_to_t, w.txt, txt_mod, direct, saved.txt,
        direct.txt_proj_slot, direct.txt_mlp0_slot, direct.txt_mlp2_slot,
        N_TXT, D, Fmlp, eps, ones_t, ctx,
    )

    var d_tatt_4d = reshape(tpb.d_att, [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att, [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)

    var q_rope_t = _tb16(saved.q_rope.copy(), [1, S, H, Dh], ctx)
    var k_rope_t = _tb16(saved.k_rope.copy(), [1, S, H, Dh], ctx)
    var v_joint_t = _tb16(saved.v_joint.copy(), [1, S, H, Dh], ctx)
    var sb = sdpa_backward[1, S, H, Dh](q_rope_t, k_rope_t, v_joint_t, d_att_joint, scale, ctx)
    var d_q_joint = rope_backward(_to_f32(sb.d_q, ctx), cos, sin, True, ctx)
    var d_k_joint = rope_backward(_to_f32(sb.d_k, ctx), cos, sin, True, ctx)
    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(_to_f32(sb.d_v, ctx), N_TXT, N_IMG, 1, ctx)

    var iprb = _stream_pre_backward_direct[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, direct, saved.img,
        direct.img_q_slot, direct.img_k_slot, direct.img_v_slot,
        N_IMG, D, eps, ones_t, ctx,
    )
    var tprb = _stream_pre_backward_direct[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, direct, saved.txt,
        direct.txt_q_slot, direct.txt_k_slot, direct.txt_v_slot,
        N_TXT, D, eps, ones_t, ctx,
    )

    var d_img_x = _add_lists(ipb.d_x, iprb.d_x)
    var d_txt_x = _add_lists(tpb.d_x, tprb.d_x)
    var img = FluxStreamDirectLycorisGrads(
        d_img_x^,
        iprb.d_shift1.copy(), iprb.d_scale1.copy(), ipb.d_gate1.copy(),
        ipb.d_shift2.copy(), ipb.d_scale2.copy(), ipb.d_gate2.copy(),
        _flux_direct_grad_public(iprb.q_g^),
        _flux_direct_grad_public(iprb.k_g^),
        _flux_direct_grad_public(iprb.v_g^),
        _flux_direct_grad_public(ipb.proj_g^),
        _flux_direct_grad_public(ipb.mlp0_g^),
        _flux_direct_grad_public(ipb.mlp2_g^),
    )
    var txt = FluxStreamDirectLycorisGrads(
        d_txt_x^,
        tprb.d_shift1.copy(), tprb.d_scale1.copy(), tpb.d_gate1.copy(),
        tpb.d_shift2.copy(), tpb.d_scale2.copy(), tpb.d_gate2.copy(),
        _flux_direct_grad_public(tprb.q_g^),
        _flux_direct_grad_public(tprb.k_g^),
        _flux_direct_grad_public(tprb.v_g^),
        _flux_direct_grad_public(tpb.proj_g^),
        _flux_direct_grad_public(tpb.mlp0_g^),
        _flux_direct_grad_public(tpb.mlp2_g^),
    )
    return FluxDoubleBlockDirectLycorisGrads(img^, txt^)


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE block LoRA forward/backward.
#   fused = linear(norm, W1, b1)   [S, 3D+Fmlp]
#   qkv = fused[:, :3D] ; mlp_in = fused[:, 3D:3D+Fmlp]
#   LoRA on to_q/to_k/to_v (3 D-slices of qkv) + proj_mlp (the Fmlp slice).
#   out = linear(out_in, W2, b2) ; LoRA on linear2 (input = out_in [S,D+Fmlp]).
# ═══════════════════════════════════════════════════════════════════════════
def single_block_lora_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs, lora: SingleBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)
    var cos_b = _to_bf16(cos, ctx)
    var sin_b = _to_bf16(sin, ctx)

    var x_t = _t(x, [S, D], ctx)
    var ln_t = layer_norm(x_t, ones_t, zeros_t, eps, ctx)
    var norm_t = modulate(ln_t, _t(mv.scale.copy(), [D], ctx), _t(mv.shift.copy(), [D], ctx), ctx)
    var norm_h = norm_t.to_host(ctx)                       # [S,D] F32 for LoRA computation

    var b1 = Optional[Tensor](w.b1[].clone(ctx))
    var fused = linear(norm_t, w.w1[], b1, ctx)            # [S, 3D+Fmlp]

    var q_base = slice(fused, 1, 0, D, ctx).to_host(ctx)
    var k_base = slice(fused, 1, D, D, ctx).to_host(ctx)
    var v_base = slice(fused, 1, 2 * D, D, ctx).to_host(ctx)
    var mlp_base = slice(fused, 1, 3 * D, Fmlp, ctx).to_host(ctx)   # [S,Fmlp]

    var q_h = flux_lora_apply(q_base, norm_h, lora.to_q, S, ctx)
    var k_h = flux_lora_apply(k_base, norm_h, lora.to_k, S, ctx)
    var v_h = flux_lora_apply(v_base, norm_h, lora.to_v, S, ctx)
    var mlp_in_h = flux_lora_apply(mlp_base, norm_h, lora.proj_mlp, S, ctx)

    var q_pre = reshape_owned(_t(q_h, [S, D], ctx)^, [1, S, H, Dh])
    var k_pre = reshape_owned(_t(k_h, [S, D], ctx)^, [1, S, H, Dh])
    var v = reshape_owned(_t(v_h, [S, D], ctx)^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos_b, sin_b, ctx)
    var k_rope = rope_interleaved(k_rms, cos_b, sin_b, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var mlp_in = _t(mlp_in_h, [S, Fmlp], ctx)
    var mlp_h = gelu(mlp_in, ctx)                          # [S,Fmlp]

    var out_in = concat(1, ctx, att_flat, mlp_h)           # [S, D+Fmlp]
    var out_in_h_f32 = out_in.to_host(ctx)                 # F32 for LoRA computation

    var b2 = Optional[Tensor](w.b2[].clone(ctx))
    var out_base = linear(out_in, w.w2[], b2, ctx).to_host(ctx)   # [S,D]
    var out_h = flux_lora_apply(out_base, out_in_h_f32, lora.linear2, S, ctx)
    var out_proj = _t(out_h, [S, D], ctx)

    var result = residual_gate(x_t, _t(mv.gate.copy(), [D], ctx), out_proj, ctx)

    var saved = SingleBlockSaved(
        _f32_to_bf16(x), ln_t.to_host_bf16(ctx), _f32_to_bf16(norm_h),
        q_pre.to_host_bf16(ctx), k_pre.to_host_bf16(ctx),
        q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx), v.to_host_bf16(ctx),
        att_flat.to_host_bf16(ctx),
        _f32_to_bf16(mlp_in_h), mlp_h.to_host_bf16(ctx), _f32_to_bf16(out_in_h_f32),
    )
    return SingleBlockForward(result.to_host(ctx), saved^)


def single_block_direct_lycoris_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs,
    direct: FluxSingleBlockDirectLycoris,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxSingleBlockDirectLycorisForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)
    var cos_b = _to_bf16(cos, ctx)
    var sin_b = _to_bf16(sin, ctx)

    var x_t = _t(x, [S, D], ctx)
    var ln_t = layer_norm(x_t, ones_t, zeros_t, eps, ctx)
    var norm_t = modulate(ln_t, _t(mv.scale.copy(), [D], ctx), _t(mv.shift.copy(), [D], ctx), ctx)
    var norm_h = norm_t.to_host_bf16(ctx)

    var q_w = slice(w.w1[], 0, 0, D, ctx)
    var k_w = slice(w.w1[], 0, D, D, ctx)
    var v_w = slice(w.w1[], 0, 2 * D, D, ctx)
    var pm_w = slice(w.w1[], 0, 3 * D, Fmlp, ctx)
    var q_b = slice(w.b1[], 0, 0, D, ctx)
    var k_b = slice(w.b1[], 0, D, D, ctx)
    var v_b = slice(w.b1[], 0, 2 * D, D, ctx)
    var pm_b = slice(w.b1[], 0, 3 * D, Fmlp, ctx)

    var q_flat = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, direct.q_slot,
        norm_t, q_w, q_b, S, D, D, ctx,
    )
    var k_flat = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, direct.k_slot,
        norm_t, k_w, k_b, S, D, D, ctx,
    )
    var v_flat = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, direct.v_slot,
        norm_t, v_w, v_b, S, D, D, ctx,
    )
    var mlp_in = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, direct.proj_mlp_slot,
        norm_t, pm_w, pm_b, S, D, Fmlp, ctx,
    )

    var q_pre = reshape_owned(q_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos_b, sin_b, ctx)
    var k_rope = rope_interleaved(k_rms, cos_b, sin_b, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var mlp_h = gelu(mlp_in, ctx)
    var out_in = concat(1, ctx, att_flat, mlp_h)
    var out_in_h = out_in.to_host_bf16(ctx)
    var out_proj = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, direct.linear2_slot,
        out_in, w.w2[], w.b2[], S, D + Fmlp, D, ctx,
    )

    var result = residual_gate(x_t, _t(mv.gate.copy(), [D], ctx), out_proj, ctx)

    var saved = SingleBlockSaved(
        _f32_to_bf16(x), ln_t.to_host_bf16(ctx), norm_h^,
        q_pre.to_host_bf16(ctx), k_pre.to_host_bf16(ctx),
        q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx), v.to_host_bf16(ctx),
        att_flat.to_host_bf16(ctx),
        mlp_in.to_host_bf16(ctx), mlp_h.to_host_bf16(ctx), out_in_h^,
    )
    return FluxSingleBlockDirectLycorisForward(result.to_host(ctx), saved^)


def single_block_direct_lycoris_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs,
    direct: FluxSingleBlockDirectLycoris, saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxSingleBlockDirectLycorisGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var scale_t = _t(mv.scale.copy(), [D], ctx)
    var gate_t_f32 = _tf32(mv.gate.copy(), [D], ctx)

    var d_out_t = _tf32(d_out, [S, D], ctx)
    var x_t = _tb16(saved.x.copy(), [S, D], ctx)
    var out_in_t = _tb16(saved.out_in.copy(), [S, D + Fmlp], ctx)

    var out_y = _direct_proj_fwd_device(
        direct.algo, direct.dora, direct.oft, direct.linear2_slot,
        out_in_t, w.w2[], w.b2[], S, D + Fmlp, D, ctx,
    )
    var grg = gate_residual_backward(d_out_t, x_t, gate_t_f32, out_y, ctx)
    var d_gate = grg.d_g.to_host(ctx)
    var l2_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, direct.linear2_slot,
        _to_bf16(grg.d_y, ctx), out_in_t, w.w2[], S, D + Fmlp, D, ctx,
    )

    var dx_w2_f32 = _to_f32(l2_g.d_x[], ctx)
    reshape_in_place(dx_w2_f32, [1, S, D + Fmlp])
    var cb = cat_backward(dx_w2_f32, D, Fmlp, 2, ctx)
    reshape_in_place(cb.d_0, [1, S, H, Dh])
    reshape_in_place(cb.d_1, [S, Fmlp])
    var d_att_flat_b = _to_bf16(cb.d_0, ctx)
    var d_mlp_h_b = _to_bf16(cb.d_1, ctx)

    var mlp_in_t = _tb16(saved.mlp_in.copy(), [S, Fmlp], ctx)
    var d_mlp_in = gelu_backward(d_mlp_h_b, mlp_in_t, ctx)

    var q_rope_t = _tb16(saved.q_rope.copy(), [1, S, H, Dh], ctx)
    var k_rope_t = _tb16(saved.k_rope.copy(), [1, S, H, Dh], ctx)
    var v_t = _tb16(saved.v.copy(), [1, S, H, Dh], ctx)
    var sb = sdpa_backward[1, S, H, Dh](q_rope_t, k_rope_t, v_t, d_att_flat_b, scale, ctx)

    var d_q_rms = rope_backward(_to_f32(sb.d_q, ctx), cos, sin, True, ctx)
    var d_k_rms = rope_backward(_to_f32(sb.d_k, ctx), cos, sin, True, ctx)

    var q_pre_t = _tb16(saved.q_pre.copy(), [1, S, H, Dh], ctx)
    var k_pre_t = _tb16(saved.k_pre.copy(), [1, S, H, Dh], ctx)
    var rb_q = rms_norm_backward(_to_bf16(d_q_rms, ctx), q_pre_t, w.q_norm[], eps, ctx)
    var rb_k = rms_norm_backward(_to_bf16(d_k_rms, ctx), k_pre_t, w.k_norm[], eps, ctx)

    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    var norm_t = _tb16(saved.norm.copy(), [S, D], ctx)
    var q_w = slice(w.w1[], 0, 0, D, ctx)
    var k_w = slice(w.w1[], 0, D, D, ctx)
    var v_w = slice(w.w1[], 0, 2 * D, D, ctx)
    var pm_w = slice(w.w1[], 0, 3 * D, Fmlp, ctx)
    var q_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, direct.q_slot,
        rb_q.d_x, norm_t, q_w, S, D, D, ctx,
    )
    var k_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, direct.k_slot,
        rb_k.d_x, norm_t, k_w, S, D, D, ctx,
    )
    var v_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, direct.v_slot,
        sb.d_v, norm_t, v_w, S, D, D, ctx,
    )
    var pm_g = _direct_proj_bwd_device(
        direct.algo, direct.dora, direct.oft, direct.proj_mlp_slot,
        d_mlp_in, norm_t, pm_w, S, D, Fmlp, ctx,
    )

    var d_norm = add(add(add(q_g.d_x[], k_g.d_x[], ctx), v_g.d_x[], ctx), pm_g.d_x[], ctx)
    var mb = modulate_backward(d_norm, _tb16(saved.ln.copy(), [S, D], ctx), scale_t, ctx)
    var d_scale = mb.d_scale.to_host(ctx)
    var d_shift = mb.d_shift.to_host(ctx)
    var lnb = layer_norm_backward(mb.d_x, x_t, ones_t, eps, ctx)
    var d_x = _add_lists(grg.d_x.to_host(ctx), lnb.d_x.to_host(ctx))

    return FluxSingleBlockDirectLycorisGrads(
        d_x^, d_shift^, d_scale^, d_gate^,
        _flux_direct_grad_public(q_g^),
        _flux_direct_grad_public(k_g^),
        _flux_direct_grad_public(v_g^),
        _flux_direct_grad_public(pm_g^),
        _flux_direct_grad_public(l2_g^),
    )


def single_block_lora_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs, lora: SingleBlockLora, saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockLoraBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var scale_t = _t(mv.scale.copy(), [D], ctx)        # bf16 for modulate_backward
    var gate_t_f32 = _tf32(mv.gate.copy(), [D], ctx)   # F32 for gate_residual_backward

    var d_out_t = _tf32(d_out, [S, D], ctx)            # F32 (gate_residual_backward grad_out)
    var x_t = _tb16(saved.x.copy(), [S, D], ctx)
    var out_in_t = _tb16(saved.out_in.copy(), [S, D + Fmlp], ctx)

    var d_a_slots = List[List[Float32]]()
    var d_b_slots = List[List[Float32]]()
    for _ in range(SGL_SLOTS):
        d_a_slots.append(List[Float32]()); d_b_slots.append(List[Float32]())

    # result = residual_gate(x, gate, out): recompute out WITH LoRA(linear2)
    # F32-ONLY: gate_residual_backward grad_out/gate F32 (x/y may be bf16).
    var b2 = Optional[Tensor](w.b2[].clone(ctx))
    var out_base = linear(out_in_t, w.w2[], b2, ctx).to_host(ctx)
    var out_in_vals_f32 = List[Float32]()
    for i in range(len(saved.out_in)):
        out_in_vals_f32.append(Float32(saved.out_in[i]))
    var out_y_h = flux_lora_apply(out_base, out_in_vals_f32, lora.linear2, S, ctx)
    var out_y = _t(out_y_h, [S, D], ctx)               # bf16
    var grg = gate_residual_backward(d_out_t, x_t, gate_t_f32, out_y, ctx)
    var d_gate = grg.d_g.to_host(ctx)

    # out = linear(out_in, W2)[+LoRA(linear2)]  W2 [D, D+Fmlp]. grg.d_y F32 → bf16.
    var pl2 = _proj_bwd_with_lora(
        _to_bf16(grg.d_y, ctx), out_in_t, w.w2[], out_in_vals_f32, lora.linear2, S_L2,
        S, D + Fmlp, D, d_a_slots, d_b_slots, ctx,
    )
    var d_w2 = pl2.d_w.to_host(ctx)
    var d_b2 = pl2.d_b.to_host(ctx)

    # out_in = concat(att_flat, mlp_h). F32-ONLY: cat_backward needs F32 grad.
    var dx_w2_f32 = _to_f32(pl2.d_x, ctx)
    reshape_in_place(dx_w2_f32, [1, S, D + Fmlp])
    var cb = cat_backward(dx_w2_f32, D, Fmlp, 2, ctx)
    reshape_in_place(cb.d_0, [1, S, H, Dh])
    reshape_in_place(cb.d_1, [S, Fmlp])
    var d_att_flat_b = _to_bf16(cb.d_0, ctx)
    var d_mlp_h_b = _to_bf16(cb.d_1, ctx)

    # mlp_h = gelu(mlp_in)
    var mlp_in_t = _tb16(saved.mlp_in.copy(), [S, Fmlp], ctx)
    var d_mlp_in = gelu_backward(d_mlp_h_b, mlp_in_t, ctx)   # [S,Fmlp] bf16

    # attention branch (bf16-native sdpa).
    var q_rope_t = _tb16(saved.q_rope.copy(), [1, S, H, Dh], ctx)
    var k_rope_t = _tb16(saved.k_rope.copy(), [1, S, H, Dh], ctx)
    var v_t = _tb16(saved.v.copy(), [1, S, H, Dh], ctx)
    var sb = sdpa_backward[1, S, H, Dh](q_rope_t, k_rope_t, v_t, d_att_flat_b, scale, ctx)

    # F32-ONLY: rope_backward grad + cos/sin F32. sb.d_q/d_k bf16 → cast up.
    var d_q_rms = rope_backward(_to_f32(sb.d_q, ctx), cos, sin, True, ctx)
    var d_k_rms = rope_backward(_to_f32(sb.d_k, ctx), cos, sin, True, ctx)

    var q_pre_t = _tb16(saved.q_pre.copy(), [1, S, H, Dh], ctx)
    var k_pre_t = _tb16(saved.k_pre.copy(), [1, S, H, Dh], ctx)
    # back to bf16 for the bf16-native rms_norm_backward chain.
    var rb_q = rms_norm_backward(_to_bf16(d_q_rms, ctx), q_pre_t, w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(_to_bf16(d_k_rms, ctx), k_pre_t, w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # join the per-slice d_y into d_fused [S, 3D+Fmlp] (all bf16). sb.d_v bf16.
    var d_qkv = concat(1, ctx, rb_q.d_x, rb_k.d_x, sb.d_v)   # [S,3D] bf16
    var d_fused = concat(1, ctx, d_qkv, d_mlp_in)            # [S,3D+Fmlp] bf16

    # fused = linear(norm, W1, b1)
    var norm_t = _tb16(saved.norm.copy(), [S, D], ctx)
    var lb_w1 = linear_backward(d_fused, norm_t, w.w1[], S, D, 3 * D + Fmlp, ctx)
    var d_w1 = lb_w1.d_w.to_host(ctx)
    var d_b1 = lb_w1.d_b.to_host(ctx)
    var d_norm = lb_w1.d_x.to_host(ctx)   # base norm grad [S,D]

    # LoRA on to_q/to_k/to_v (input = norm F32) + proj_mlp.
    var norm_vals_f32 = List[Float32]()
    for i in range(len(saved.norm)):
        norm_vals_f32.append(Float32(saved.norm[i]))
    var d_q_h = rb_q.d_x.to_host(ctx)
    var d_k_h = rb_k.d_x.to_host(ctx)
    var d_v_h = sb.d_v.to_host(ctx)
    var d_mlp_in_h = d_mlp_in.to_host(ctx)
    if lora.to_q:
        var lg = flux_lora_bwd(d_q_h, norm_vals_f32, lora.to_q.value(), S, ctx)
        d_a_slots[S_SQ] = lg.d_a.copy(); d_b_slots[S_SQ] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)
    if lora.to_k:
        var lg = flux_lora_bwd(d_k_h, norm_vals_f32, lora.to_k.value(), S, ctx)
        d_a_slots[S_SK] = lg.d_a.copy(); d_b_slots[S_SK] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)
    if lora.to_v:
        var lg = flux_lora_bwd(d_v_h, norm_vals_f32, lora.to_v.value(), S, ctx)
        d_a_slots[S_SV] = lg.d_a.copy(); d_b_slots[S_SV] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)
    if lora.proj_mlp:
        var lg = flux_lora_bwd(d_mlp_in_h, norm_vals_f32, lora.proj_mlp.value(), S, ctx)
        d_a_slots[S_PMLP] = lg.d_a.copy(); d_b_slots[S_PMLP] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)

    # norm = modulate(ln, scale, shift)
    var ln_t = _tb16(saved.ln.copy(), [S, D], ctx)
    var mb = modulate_backward(_t(d_norm, [S, D], ctx), ln_t, scale_t, ctx)
    var d_scale = mb.d_scale.to_host(ctx)
    var d_shift = mb.d_shift.to_host(ctx)

    var lnb = layer_norm_backward(mb.d_x, x_t, ones_t, eps, ctx)
    var d_x_res = grg.d_x.to_host(ctx)
    var d_x_norm = lnb.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    var base = SingleBlockGrads(
        d_x^, d_w1^, d_b1^, d_w2^, d_b2^, d_q_norm^, d_k_norm^,
        d_shift^, d_scale^, d_gate^,
    )
    return SingleBlockLoraBackward(base^, SingleBlockLoraGrads(d_a_slots^, d_b_slots^))
