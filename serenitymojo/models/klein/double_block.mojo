# serenitymojo/models/klein/double_block.mojo
#
# Klein (FLUX.2) DOUBLE-STREAM DiT block: forward (saving activations) +
# hand-chained backward (training), packaged as a reusable unit in the EXACT
# style proven by serenitymojo/training/dit_block.mojo (the single-stream unit,
# verdict "PACKAGED UNIT MATCHES INLINE COMPOSITION" vs torch). This file is
# that pattern DOUBLED: two streams (img + txt) coupled by ONE joint attention.
#
# DEVICE-RESIDENT INTERIOR (Increment 2 perf refactor, 2026-05-31)
#   The PUBLIC API is unchanged: `img`/`txt` enter as host `List[Float32]`,
#   `img_out`/`txt_out` and every grad leave as host `List[Float32]` (so
#   klein_stack.mojo / klein_stack_lora.mojo and the parity gates compile with
#   ZERO changes). But INTERNALLY the chain now threads device `Tensor`s op-to-op:
#   `from_host(img)`/`from_host(txt)` run ONCE at the forward entry, `.to_host`
#   runs ONCE per output / per grad leaving the block. Saved activations are
#   device Tensors carried by `TArc` (= ArcPointer[Tensor], autograd.mojo:50) —
#   a TArc copy is a refcount bump of the SAME buffer (no D2D copy, no sync). The
#   OLD code bounced EVERY intermediate op's output to host (`.to_host` →
#   `from_host`) and did host split/join row-loops, forcing ~hundreds of
#   host-stall syncs per block. Removing the per-INTERMEDIATE bounce is the win;
#   the boundary from_host/to_host stay.
#
#   WHY TArc (and not Movable-only Tensor like single_block.mojo)
#   The double block REUSES per-stream activations at multiple consumers that a
#   plain move can't serve: `ip.q_rms`/`ip.k_rms`/`ip.v` feed BOTH the joint
#   concat AND are stored into `StreamSaved`; the `_StreamPre`/`_StreamPost`
#   helper structs have destructors, so a partial `^`-move of ONE field out of a
#   struct is illegal in Mojo 1.0.0b1 (the file's own original comment warned of
#   this). TArc dissolves that: every field is Copyable-by-refcount, so a value
#   can be borrowed (`arc[]`) by an op AND copied (refcount bump) into the saved
#   struct, with the buffer shared, freed once by the last Arc owner. No clone(),
#   no to_host() per intermediate.
#
#   The qkv|gate_up split, the q/k/v split, and their backward scatters stay
#   DEVICE slice/concat (ops/tensor_algebra.slice + .concat) — same as the joint
#   section already did. reshape [N,D]<->[1,N,H,Dh] is a row-major byte no-op, so
#   on flat row-major buffers the [N,3D] qkv slice into three [N,D] blocks is a
#   contiguous column cut == slice(dim=1).
#
# WHY HOST List[Float32] STILL AT THE API BOUNDARY
#   The boundary contract is fixed by the callers (stack + gates pass host lists).
#   The LoRA-delta helpers have device-resident siblings, so LoRA activations and
#   adapter A/B tensors stay on device in the hot trainer path; only d_A/d_B
#   leave for the existing host optimizer state. The base chain is fully
#   device-resident.
#
# FORWARD GRAPH (mirrors models/dit/klein_dit.mojo `_double_block`, lines 267-352)
#   For stream s in {img, txt}, with precomputed AdaLN vectors
#   (shift1,scale1,gate1,shift2,scale2,gate2) each [D]:
#     s_ln1   = layer_norm(s, 1, 0, eps)                     # ones/zeros weight
#     s_norm  = modulate(s_ln1, scale1, shift1)              # (1+scale)*x+shift
#     s_qkv   = linear(s_norm, Wqkv_s)                       # [1,N,3D]
#     s_q/k/v = qkv_part(s_qkv)  -> reshape [1,N,H,Dh]
#     s_q     = rms_norm(s_q, query_norm_s)                  # per-head Dh rms
#     s_k     = rms_norm(s_k, key_norm_s)
#   JOINT (txt FIRST, then img):
#     q = concat(axis=1, txt_q, img_q)   k,v likewise        # [1,S,H,Dh]
#     qr = rope_interleaved(q, cos, sin) ; kr = rope_interleaved(k, cos, sin)
#     att = sdpa_nomask(qr, kr, v, 1/sqrt(Dh))               # [1,S,H,Dh]
#     txt_att = slice(att, 1, 0, N_TXT) ; img_att = slice(att, 1, N_TXT, N_IMG)
#     reshape each -> [1,N,D]
#   Per stream again:
#     s_out      = linear(s_att, Wproj_s)                    # [1,N,D]
#     s_attn_res = residual_gate(s, gate1, s_out)            # s + gate1*s_out
#     s_ln2      = layer_norm(s_attn_res, 1, 0, eps)
#     s_mlp_in   = modulate(s_ln2, scale2, shift2)
#     s_mlp      = swiglu_linear(s_mlp_in, Wgu_s, Wd_s)
#     s_final    = residual_gate(s_attn_res, gate2, s_mlp)
#   output = (txt_final, img_final)
#
# swiglu_linear(x, Wgu, Wd): gu=linear(x,Wgu)[1,N,2F]; gate=gu[:,:,:F];
#   up=gu[:,:,F:]; act=swiglu(gate,up); return linear(act, Wd).
#
# BACKWARD: every arm is an EXISTING, VERIFIED kernel — this file only composes
# them. The joint-attention coupling means d for txt and img both flow OUT of the
# SAME sdpa_backward, then split via the concat/slice backward (txt FIRST).
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor move-only (carried via TArc in the saved
# structs); no-bias linear = linear(x, w, Optional[Tensor](None), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator

# ── forward ops (GPU) ────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
# cuDNN flash SDPA for the joint attention (approved 2026-06-11; ONE flag
# for all Klein blocks, defined in single_block.mojo — see its header).
from serenitymojo.models.klein.single_block import KLEIN_SDPA_FLASH
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_f32, sdpa_flash_backward_f32,
)
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, reshape_in_place, slice, concat, add,
)
from serenitymojo.ops.tensor_algebra_scratch import (
    concat2_scratch, concat3_scratch, slice_scratch,
)

# ── backward arms (GPU; all pre-built + gated) ───────────────────────────────
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx, linear_backward_dx_scratch, LinearGrads,
)
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, rms_norm_backward_dx, RmsNormBackward,
    layer_norm_backward, layer_norm_backward_dx, LayerNormBackward,
)
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward, SwigluGrads
from serenitymojo.ops.attention_backward import (
    sdpa_backward, sdpa_backward_scratch, SdpaGrads,
)
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, gate_residual_backward_dxdy, GateResidualGrads,
    rope_backward,
)
from serenitymojo.ops.shape_backward import (
    cat_backward, CatGrads2, slice_backward, reshape_backward,
)


# TArc = the Copyable device carrier (ArcPointer[Tensor]); a copy is a refcount
# bump of the SAME device buffer (no D2D, no sync). Mirrors autograd.mojo:50.
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


# F32 host-list -> device Tensor (boundary / weight upload only).
def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _t_dtype(
    vals: List[Float32], var shape: List[Int], dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    return Tensor.from_host(vals, shape^, dtype, ctx)


# F32 host-list -> resident device Tensor boxed as a Copyable carrier.
def _ta(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, shape^, STDtype.F32, ctx))


# ── per-stream modulation vectors (each [D]) ─────────────────────────────────
struct ModVecs(Copyable, Movable):
    var shift1: List[Float32]
    var scale1: List[Float32]
    var gate1: List[Float32]
    var shift2: List[Float32]
    var scale2: List[Float32]
    var gate2: List[Float32]

    def __init__(
        out self,
        var shift1: List[Float32], var scale1: List[Float32], var gate1: List[Float32],
        var shift2: List[Float32], var scale2: List[Float32], var gate2: List[Float32],
    ):
        self.shift1 = shift1^
        self.scale1 = scale1^
        self.gate1 = gate1^
        self.shift2 = shift2^
        self.scale2 = scale2^
        self.gate2 = gate2^


struct ModVecsDevice(Copyable, Movable):
    var shift1: TArc
    var scale1: TArc
    var gate1: TArc
    var shift2: TArc
    var scale2: TArc
    var gate2: TArc

    def __init__(
        out self,
        var shift1: TArc, var scale1: TArc, var gate1: TArc,
        var shift2: TArc, var scale2: TArc, var gate2: TArc,
    ):
        self.shift1 = shift1^
        self.scale1 = scale1^
        self.gate1 = gate1^
        self.shift2 = shift2^
        self.scale2 = scale2^
        self.gate2 = gate2^


def modvecs_to_device(mv: ModVecs, D: Int, ctx: DeviceContext) raises -> ModVecsDevice:
    return ModVecsDevice(
        TArc(_t(mv.shift1.copy(), [D], ctx)),
        TArc(_t(mv.scale1.copy(), [D], ctx)),
        TArc(_t(mv.gate1.copy(), [D], ctx)),
        TArc(_t(mv.shift2.copy(), [D], ctx)),
        TArc(_t(mv.scale2.copy(), [D], ctx)),
        TArc(_t(mv.gate2.copy(), [D], ctx)),
    )


# ── per-stream trainable weights (A2: DEVICE-RESIDENT, uploaded ONCE) ─────────
#   wqkv: [3D, D]   wproj: [D, D]
#   wgu : [2F, D]   wd:    [D, F]
#   q_norm/k_norm: [Dh]  (per-head rms scale)
#
# A2 PERF (2026-05-31): the FROZEN base weight matrices are now device-resident
# `TArc` carriers uploaded EXACTLY ONCE at struct construction (load time),
# instead of host `List[Float32]` re-uploaded by `_t(w.field.copy(), ...)` on
# EVERY linear/rms_norm call EVERY step (the measured 73.7 GB H2D/step). The
# `__init__` accepts the host lists (so the loader + parity gates pass the same
# byte-identical data) plus the dims + ctx needed to upload each at its real
# shape; the upload runs once here. Use-sites pass `w.field[]` (a borrow of the
# SAME resident buffer) — no per-op `from_host`, no per-op sync.
struct StreamWeights(Copyable, Movable):
    var wqkv: TArc      # [3D, D]
    var wproj: TArc     # [D, D]
    var wgu: TArc       # [2F, D]
    var wd: TArc        # [D, F]
    var q_norm: TArc    # [Dh]
    var k_norm: TArc    # [Dh]

    def __init__(
        out self,
        var wqkv: List[Float32], var wproj: List[Float32],
        var wgu: List[Float32], var wd: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
        D: Int, F: Int, Dh: Int, ctx: DeviceContext,
    ) raises:
        self.wqkv = TArc(Tensor.from_host(wqkv^, [3 * D, D], STDtype.F32, ctx))
        self.wproj = TArc(Tensor.from_host(wproj^, [D, D], STDtype.F32, ctx))
        self.wgu = TArc(Tensor.from_host(wgu^, [2 * F, D], STDtype.F32, ctx))
        self.wd = TArc(Tensor.from_host(wd^, [D, F], STDtype.F32, ctx))
        self.q_norm = TArc(Tensor.from_host(q_norm^, [Dh], STDtype.F32, ctx))
        self.k_norm = TArc(Tensor.from_host(k_norm^, [Dh], STDtype.F32, ctx))

    def __init__(
        out self,
        var wqkv: TArc, var wproj: TArc,
        var wgu: TArc, var wd: TArc,
        var q_norm: TArc, var k_norm: TArc,
    ):
        self.wqkv = wqkv^
        self.wproj = wproj^
        self.wgu = wgu^
        self.wd = wd^
        self.q_norm = q_norm^
        self.k_norm = k_norm^


struct DoubleBlockWeights(Copyable, Movable):
    var img: StreamWeights
    var txt: StreamWeights

    def __init__(out self, var img: StreamWeights, var txt: StreamWeights):
        self.img = img^
        self.txt = txt^


# ── saved activations (per stream, DEVICE-RESIDENT via TArc) ──────────────────
# Each field is the fresh device Tensor returned by the producing op, carried by
# a Copyable refcount handle (TArc). Consumed by-borrow in the backward
# (`sv.field[]`); callers only borrow a StreamSaved, never copy the struct, so
# the SAME device buffers are reused — no clone(), no to_host() per intermediate.
struct StreamSaved(Copyable, Movable):
    var x: TArc        # [N,D]   block input
    var ln1: TArc      # [N,D]   layer_norm(x)
    var norm: TArc     # [N,D]   modulate(ln1, scale1, shift1)
    var q_pre: TArc    # [1,N,H,Dh]  q before rms (post-qkv split)
    var k_pre: TArc    # [1,N,H,Dh]
    var q_rms: TArc    # [1,N,H,Dh]  rms_norm(q_pre, q_norm)
    var k_rms: TArc    # [1,N,H,Dh]
    var v: TArc        # [1,N,H,Dh]
    var att: TArc      # [N,D]   per-stream attention slice (reshaped)
    var attn_res: TArc # [N,D]   residual_gate(x, gate1, proj(att))
    var ln2: TArc      # [N,D]   layer_norm(attn_res)
    var mlp_in: TArc   # [N,D]   modulate(ln2, scale2, shift2)
    var gu: TArc       # [N,2F]  linear(mlp_in, Wgu)
    var gate: TArc     # [N,F]
    var up: TArc       # [N,F]
    var act: TArc      # [N,F]   swiglu(gate, up)

    def __init__(
        out self,
        var x: TArc, var ln1: TArc, var norm: TArc,
        var q_pre: TArc, var k_pre: TArc,
        var q_rms: TArc, var k_rms: TArc, var v: TArc,
        var att: TArc, var attn_res: TArc,
        var ln2: TArc, var mlp_in: TArc,
        var gu: TArc, var gate: TArc, var up: TArc,
        var act: TArc,
    ):
        self.x = x^
        self.ln1 = ln1^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^
        self.att = att^
        self.attn_res = attn_res^
        self.ln2 = ln2^
        self.mlp_in = mlp_in^
        self.gu = gu^
        self.gate = gate^
        self.up = up^
        self.act = act^


struct DoubleBlockSaved(Copyable, Movable):
    var img: StreamSaved
    var txt: StreamSaved
    # joint attention saved (shared across both streams), DEVICE-RESIDENT via TArc
    var q_rope: TArc   # [1,S,H,Dh]  rope(concat q)
    var k_rope: TArc   # [1,S,H,Dh]  rope(concat k)
    var v_joint: TArc  # [1,S,H,Dh]  concat v
    # cos/sin are NOT saved here — resident rope tables passed by borrow to the
    # backward (uploaded ONCE per run instead of re-saved/re-uploaded per block).
    # Flash-SDPA saved set (Optional: only the KLEIN_SDPA_FLASH production
    # fwd fills these; other constructor sites pass nothing).
    var flash_q: Optional[TArc]
    var flash_k: Optional[TArc]
    var flash_v: Optional[TArc]
    var flash_o: Optional[TArc]
    var flash_stats: Optional[TArc]

    def __init__(
        out self, var img: StreamSaved, var txt: StreamSaved,
        var q_rope: TArc, var k_rope: TArc, var v_joint: TArc,
        var flash_q: Optional[TArc] = None,
        var flash_k: Optional[TArc] = None,
        var flash_v: Optional[TArc] = None,
        var flash_o: Optional[TArc] = None,
        var flash_stats: Optional[TArc] = None,
    ):
        self.img = img^
        self.txt = txt^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.v_joint = v_joint^
        self.flash_q = flash_q^
        self.flash_k = flash_k^
        self.flash_v = flash_v^
        self.flash_o = flash_o^
        self.flash_stats = flash_stats^


struct DoubleBlockForward(Copyable, Movable):
    var img_out: List[Float32]   # [N_IMG, D]  (host: boundary readback)
    var txt_out: List[Float32]   # [N_TXT, D]
    var saved: DoubleBlockSaved

    def __init__(
        out self, var img_out: List[Float32], var txt_out: List[Float32],
        var saved: DoubleBlockSaved,
    ):
        self.img_out = img_out^
        self.txt_out = txt_out^
        self.saved = saved^


struct DoubleBlockDeviceForward(Copyable, Movable):
    var img_out: TArc
    var txt_out: TArc
    var saved: DoubleBlockSaved

    def __init__(
        out self, var img_out: TArc, var txt_out: TArc,
        var saved: DoubleBlockSaved,
    ):
        self.img_out = img_out^
        self.txt_out = txt_out^
        self.saved = saved^


struct DoubleBlockDeviceOutput(Copyable, Movable):
    var img_out: TArc
    var txt_out: TArc

    def __init__(out self, var img_out: TArc, var txt_out: TArc):
        self.img_out = img_out^
        self.txt_out = txt_out^


# ── backward result: stream input grads + all trainable weight grads ─────────
struct StreamGrads(Copyable, Movable):
    var d_x: List[Float32]
    var d_wqkv: List[Float32]
    var d_wproj: List[Float32]
    var d_wgu: List[Float32]
    var d_wd: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    # modulation-vector grads (block outputs; not backproped into mod MLP)
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32], var d_wqkv: List[Float32], var d_wproj: List[Float32],
        var d_wgu: List[Float32], var d_wd: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_shift1: List[Float32], var d_scale1: List[Float32], var d_gate1: List[Float32],
        var d_shift2: List[Float32], var d_scale2: List[Float32], var d_gate2: List[Float32],
    ):
        self.d_x = d_x^
        self.d_wqkv = d_wqkv^
        self.d_wproj = d_wproj^
        self.d_wgu = d_wgu^
        self.d_wd = d_wd^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_shift1 = d_shift1^
        self.d_scale1 = d_scale1^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^


struct DoubleBlockGrads(Copyable, Movable):
    var img: StreamGrads
    var txt: StreamGrads

    def __init__(out self, var img: StreamGrads, var txt: StreamGrads):
        self.img = img^
        self.txt = txt^


# ── per-stream FORWARD up to the per-stream q/k/v (pre-join), DEVICE-RESIDENT ──
# Fields are TArc so q_rms/k_rms/v can be BOTH borrowed by the joint concat AND
# copied (refcount bump) into StreamSaved without a partial-move out of a
# struct-with-destructor.
struct _StreamPre(Copyable, Movable):
    var ln1: TArc
    var norm: TArc
    var q_pre: TArc
    var k_pre: TArc
    var q_rms: TArc
    var k_rms: TArc
    var v: TArc

    def __init__(
        out self, var ln1: TArc, var norm: TArc,
        var q_pre: TArc, var k_pre: TArc,
        var q_rms: TArc, var k_rms: TArc, var v: TArc,
    ):
        self.ln1 = ln1^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^


def _stream_pre[
    H: Int, Dh: Int
](
    x: TArc, w: StreamWeights, mv: ModVecs,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPre:
    var ln1 = layer_norm(x[], ones, zeros, eps, ctx)
    var norm = modulate(ln1, _t(mv.scale1.copy(), [D], ctx), _t(mv.shift1.copy(), [D], ctx), ctx)
    var no_bias = Optional[Tensor](None)
    var qkv = linear(norm, w.wqkv[], no_bias^, ctx)   # [N,3D]
    # qkv channel split [N,3D] -> 3x [N,D] (contiguous column cut == slice dim=1).
    var q_pre_flat = slice(qkv, 1, 0, D, ctx)
    var k_pre_flat = slice(qkv, 1, D, D, ctx)
    var v_flat = slice(qkv, 1, 2 * D, D, ctx)
    # reshape [N,D] -> [1,N,H,Dh] is a row-major byte no-op.
    var q_pre = reshape_owned(q_pre_flat^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, N, H, Dh])
    var v = reshape_owned(v_flat^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPre(
        TArc(ln1^), TArc(norm^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
    )


# per-stream FORWARD from the joint attention slice to the stream output.
struct _StreamPost(Copyable, Movable):
    var out: TArc        # [N,D]  the stream final (device; readback at the block boundary)
    var attn_res: TArc
    var ln2: TArc
    var mlp_in: TArc
    var gu: TArc
    var gate: TArc
    var up: TArc
    var act: TArc

    def __init__(
        out self, var out: TArc, var attn_res: TArc,
        var ln2: TArc, var mlp_in: TArc, var gu: TArc,
        var gate: TArc, var up: TArc, var act: TArc,
    ):
        self.out = out^
        self.attn_res = attn_res^
        self.ln2 = ln2^
        self.mlp_in = mlp_in^
        self.gu = gu^
        self.gate = gate^
        self.up = up^
        self.act = act^


def _stream_post(
    x: TArc, att: TArc, w: StreamWeights, mv: ModVecs,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, zeros: Tensor,
    ctx: DeviceContext,
) raises -> _StreamPost:
    var no_bias = Optional[Tensor](None)
    var out = linear(att[], w.wproj[], no_bias^, ctx)   # [N,D]
    var attn_res = residual_gate(x[], _t(mv.gate1.copy(), [D], ctx), out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, _t(mv.scale2.copy(), [D], ctx), _t(mv.shift2.copy(), [D], ctx), ctx)
    var no_bias2 = Optional[Tensor](None)
    var gu = linear(mlp_in, w.wgu[], no_bias2^, ctx)   # [N,2F]
    # gate/up channel split [N,2F] -> 2x [N,F] (contiguous column cut).
    var gate = slice(gu, 1, 0, F, ctx)
    var up = slice(gu, 1, F, F, ctx)
    var act = swiglu(gate, up, ctx)
    var no_bias3 = Optional[Tensor](None)
    var mlp = linear(act, w.wd[], no_bias3^, ctx)   # [N,D]
    var final = residual_gate(attn_res, _t(mv.gate2.copy(), [D], ctx), mlp, ctx)
    return _StreamPost(
        TArc(final^), TArc(attn_res^), TArc(ln2^), TArc(mlp_in^),
        TArc(gu^), TArc(gate^), TArc(up^), TArc(act^),
    )


# build a StreamSaved (refcount-bump copies of the device carriers) from the pre/
# post helpers + the per-stream attention slice. No data movement, no sync.
def _make_saved(
    x: TArc, pre: _StreamPre, att: TArc, post: _StreamPost
) -> StreamSaved:
    return StreamSaved(
        x.copy(), pre.ln1.copy(), pre.norm.copy(), pre.q_pre.copy(), pre.k_pre.copy(),
        pre.q_rms.copy(), pre.k_rms.copy(), pre.v.copy(),
        att.copy(), post.attn_res.copy(), post.ln2.copy(), post.mlp_in.copy(),
        post.gu.copy(), post.gate.copy(), post.up.copy(), post.act.copy(),
    )


# ── FORWARD of one DOUBLE block ──────────────────────────────────────────────
# cos/sin: precomputed rope tables for the JOINT sequence, [S*H, Dh/2].
def double_block_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img: List[Float32], txt: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockForward:
    # S == N_TXT + N_IMG (caller-asserted; S is comptime for sdpa).
    # cos/sin are RESIDENT rope tables borrowed from the caller (uploaded once).
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    # resident layer_norm ones[D]/zeros[D] (uploaded once, borrowed by every LN).
    var norm_dtype = STDtype.F32
    if w.img.wqkv[].dtype() == STDtype.BF16:
        norm_dtype = STDtype.BF16
    var ones_t = _t_dtype(_ones(D), [D], norm_dtype, ctx)
    var zeros_t = _t_dtype(_zeros(D), [D], norm_dtype, ctx)

    # img/txt ENTER host -> ONE from_host each (boxed as TArc carriers so the
    # input can feed BOTH the pre AND the post residual AND the saved struct).
    var img_x = _ta(img, [N_IMG, D], ctx)
    var txt_x = _ta(txt, [N_TXT, D], ctx)

    # per-stream pre-attention
    var ip = _stream_pre[H, Dh](img_x, w.img, img_mod, N_IMG, D, eps, ones_t, zeros_t, ctx)
    var tp = _stream_pre[H, Dh](txt_x, w.txt, txt_mod, N_TXT, D, eps, ones_t, zeros_t, ctx)

    # JOINT concat (txt FIRST, then img), per [1,N,H,Dh] tensors along axis=1.
    var q = concat(1, ctx, tp.q_rms[], ip.q_rms[])   # [1,S,H,Dh]
    var k = concat(1, ctx, tp.k_rms[], ip.k_rms[])
    var v = concat(1, ctx, tp.v[], ip.v[])

    # rope then sdpa (joint), all device-resident
    var q_rope = rope_interleaved(q, cos, sin, ctx)
    var k_rope = rope_interleaved(k, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)   # [1,S,H,Dh]

    # split joint attention back per stream (txt FIRST); reshape [1,N,H,Dh]->[N,D]
    # is a byte no-op so slice dim=1 then reshape to [N,D].
    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG, D]))

    # per-stream post-attention
    var ipost = _stream_post(img_x, img_att, w.img, img_mod, N_IMG, D, F, eps, ones_t, zeros_t, ctx)
    var tpost = _stream_post(txt_x, txt_att, w.txt, txt_mod, N_TXT, D, F, eps, ones_t, zeros_t, ctx)

    # save (refcount-bump carriers; no data movement)
    var img_saved = _make_saved(img_x, ip, img_att, ipost)
    var txt_saved = _make_saved(txt_x, tp, txt_att, tpost)
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^)
    )

    # ONE boundary readback per stream output.
    var img_out = ipost.out[].to_host(ctx)
    var txt_out = tpost.out[].to_host(ctx)
    return DoubleBlockForward(img_out^, txt_out^, saved^)


# ── per-stream BACKWARD: from d_out (stream output grad) down to the per-stream
#    joint-attention-slice grad d_att, plus all post-attention weight grads.
# d_x and d_att stay DEVICE-RESIDENT (TArc) so the joint section consumes them on
# device; the weight/mod-vec grads are host lists (they leave the block).
struct _StreamPostBack(Copyable, Movable):
    # d_x and d_att are HOST List[Float32]. Op-result grad structs
    # (GateResidualGrads / LinearGrads) carry move-only `Tensor` fields with a
    # synthesized destructor; Mojo 1.0.0b1 forbids moving an individual field out
    # of such a value (partial move leaves it undestroyable) and `Tensor` is not
    # Copyable, so the only legal way to get these surviving grads out of the
    # producing struct is `.to_host()` (a borrow). Both grads leave this helper
    # for the joint section: d_x is summed on host by the caller anyway; d_att is
    # re-uploaded ONCE per stream in the caller (2 from_host total — negligible vs
    # the per-intermediate bounces this refactor removes). The rest of the chain
    # stays device-resident.
    var d_x: List[Float32]      # grad into stream input via the gate1 residual [N,D]
    var d_att: List[Float32]    # grad into the joint-attention slice [N,D]
    var d_wproj: List[Float32]
    var d_wgu: List[Float32]
    var d_wd: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]

    def __init__(
        out self, var d_x: List[Float32], var d_att: List[Float32],
        var d_wproj: List[Float32], var d_wgu: List[Float32], var d_wd: List[Float32],
        var d_gate1: List[Float32],
        var d_shift2: List[Float32], var d_scale2: List[Float32], var d_gate2: List[Float32],
    ):
        self.d_x = d_x^
        self.d_att = d_att^
        self.d_wproj = d_wproj^
        self.d_wgu = d_wgu^
        self.d_wd = d_wd^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^


# NOTE: op-result grad structs (GateResidualGrads / LinearGrads) hold move-only
# `Tensor` fields with a synthesized destructor. Mojo 1.0.0b1 forbids moving an
# individual field out of such a value (partial move → "destroyed out of the
# middle of a value"), and `Tensor` is not Copyable. So everywhere below, those
# struct fields are consumed ONLY by-borrow (into the next op) or by `.to_host()`
# — never `^`-moved out. Grads that must survive past the producing scope leave as
# host `List[Float32]` (re-uploaded where the joint section needs them on device).


def _stream_post_backward(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecs, sv: StreamSaved,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPostBack:
    # final = residual_gate(attn_res, gate2, mlp): o = attn_res + gate2*mlp
    # `mlp` (the gated `y`) is recomputed = linear(act, Wd) inline (keep every
    # `sv` read local to this function, mirroring _stream_pre_backward).
    var no_bias_mlp = Optional[Tensor](None)
    var mlp_y = linear(sv.act[], w.wd[], no_bias_mlp^, ctx)
    var grg2 = gate_residual_backward(
        d_out[], sv.attn_res[], _t(mv.gate2.copy(), [D], ctx), mlp_y, ctx
    )
    # grg2.d_x = attn_res's residual branch (device); grg2.d_y = d_mlp (device)
    var d_gate2 = grg2.d_g.to_host(ctx)

    # mlp = linear(act, Wd)
    var lb_d = linear_backward(grg2.d_y, sv.act[], w.wd[], N, F, D, ctx)
    var d_wd = lb_d.d_w.to_host(ctx)

    # act = swiglu(gate, up)
    var sgb = swiglu_backward(lb_d.d_x, sv.gate[], sv.up[], ctx)
    # join gate/up grads back into d_gu [N,2F] (device concat on dim 1)
    var d_gu = concat(1, ctx, sgb.d_gate, sgb.d_up)

    # gu = linear(mlp_in, Wgu)
    var lb_gu = linear_backward(d_gu, sv.mlp_in[], w.wgu[], N, D, 2 * F, ctx)
    var d_wgu = lb_gu.d_w.to_host(ctx)

    # mlp_in = modulate(ln2, scale2, shift2)
    var mb2 = modulate_backward(lb_gu.d_x, sv.ln2[], _t(mv.scale2.copy(), [D], ctx), ctx)
    var d_scale2 = mb2.d_scale.to_host(ctx)
    var d_shift2 = mb2.d_shift.to_host(ctx)

    # ln2 = layer_norm(attn_res, 1, 0)
    var lnb2 = layer_norm_backward(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    # attn_res feeds BOTH the residual (grg2.d_x) AND ln2 -> SUM on device
    # (tensor_algebra.add; both [N,D], no host bounce).
    var d_attn_res_total = TArc(add(grg2.d_x, lnb2.d_x, ctx))

    # attn_res = residual_gate(x, gate1, proj_out): o = x + gate1*proj_out
    var no_bias = Optional[Tensor](None)
    var proj_out = linear(att[], w.wproj[], no_bias^, ctx)   # recompute proj output
    var grg1 = gate_residual_backward(
        d_attn_res_total[], x[], _t(mv.gate1.copy(), [D], ctx), proj_out, ctx
    )
    # Consume grg1 by-borrow / to_host only (no field `^`-move out of the struct).
    # grg1.d_g -> host (gate grad); grg1.d_x = residual branch -> host (caller sums
    # it with the pre branch anyway); grg1.d_y = d_proj_out borrowed into the
    # proj linear_backward below.
    var d_gate1 = grg1.d_g.to_host(ctx)
    var d_x_res = grg1.d_x.to_host(ctx)

    # proj_out = linear(att, Wproj); consume lb_p by to_host only.
    var lb_p = linear_backward(grg1.d_y, att[], w.wproj[], N, D, D, ctx)
    var d_wproj = lb_p.d_w.to_host(ctx)
    var d_att = lb_p.d_x.to_host(ctx)   # re-uploaded by the caller for the joint section

    return _StreamPostBack(
        d_x_res^, d_att^, d_wproj^, d_wgu^, d_wd^,
        d_gate1=d_gate1^,
        d_shift2=d_shift2^, d_scale2=d_scale2^, d_gate2=d_gate2^,
    )


# ── per-stream BACKWARD: from d_q/d_k/d_v (post-rms joint slices, [1,N,H,Dh])
#    down to the stream input grad d_x, plus pre-attention weight grads.
# d_x is host (it leaves the block as part of StreamGrads.d_x); the pre/post d_x
# branches are summed at the boundary.
struct _StreamPreBack(Copyable, Movable):
    var d_x: List[Float32]
    var d_wqkv: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]

    def __init__(
        out self, var d_x: List[Float32], var d_wqkv: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_shift1: List[Float32], var d_scale1: List[Float32],
    ):
        self.d_x = d_x^
        self.d_wqkv = d_wqkv^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_shift1 = d_shift1^
        self.d_scale1 = d_scale1^


def _stream_pre_backward[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecs, sv: StreamSaved,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPreBack:
    # q_rms = rms_norm(q_pre, q_norm) over last dim Dh; d_q_rms is [1,N,H,Dh].
    var rb_q = rms_norm_backward(d_q_rms, sv.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, sv.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    # join d_q_pre|d_k_pre|d_v back into d_qkv [N,3D] (reshape [1,N,H,Dh]->[N,D]
    # is a byte no-op; then device concat on dim 1).
    reshape_in_place(rb_q.d_x, [N, D])
    reshape_in_place(rb_k.d_x, [N, D])
    var d_v_flat = reshape(d_v, [N, D], ctx)
    var d_qkv = concat(1, ctx, rb_q.d_x, rb_k.d_x, d_v_flat)   # [N,3D]

    # qkv = linear(norm, Wqkv)
    var lb_qkv = linear_backward(d_qkv, sv.norm[], w.wqkv[], N, D, 3 * D, ctx)
    var d_wqkv = lb_qkv.d_w.to_host(ctx)

    # norm = modulate(ln1, scale1, shift1)
    var mb1 = modulate_backward(lb_qkv.d_x, sv.ln1[], _t(mv.scale1.copy(), [D], ctx), ctx)
    var d_scale1 = mb1.d_scale.to_host(ctx)
    var d_shift1 = mb1.d_shift.to_host(ctx)

    # ln1 = layer_norm(x, 1, 0)
    var lnb1 = layer_norm_backward(mb1.d_x, sv.x[], ones, eps, ctx)
    var d_x_norm = lnb1.d_x.to_host(ctx)
    return _StreamPreBack(
        d_x_norm^, d_wqkv^, d_q_norm^, d_k_norm^, d_shift1^, d_scale1^
    )


# ── BACKWARD of one DOUBLE block (hand-chained; the joint coupling) ──────────
# d_img_out / d_txt_out: upstream grads of the two stream outputs ([N,D] each).
def double_block_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_img_out: List[Float32], d_txt_out: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs, saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockGrads:
    # S == N_TXT + N_IMG (caller-asserted; S is comptime for sdpa backward).
    # cos/sin resident rope tables, borrowed (uploaded once).
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)

    # d_img_out / d_txt_out ENTER host -> ONE from_host each.
    var d_io_t = _ta(d_img_out, [N_IMG, D], ctx)
    var d_to_t = _ta(d_txt_out, [N_TXT, D], ctx)

    # ── post-attention backward per stream ──
    var ipb = _stream_post_backward(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, saved.img,
        N_IMG, D, F, eps, ones_t, ctx,
    )
    var tpb = _stream_post_backward(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, saved.txt,
        N_TXT, D, F, eps, ones_t, ctx,
    )
    # d_att per stream is [N,D]; reshape to [1,N,H,Dh] is a byte no-op.

    # ── join the two per-stream attention-slice grads back into joint d_att
    #    (txt FIRST, then img) along axis=1. The forward was slice(att, txt|img);
    #    the backward of a slice is scatter into zeros, and txt|img tile the full
    #    S, so a plain concat (txt then img) reconstructs d_att [1,S,H,Dh]. ──
    # d_att per stream comes back as host List (only legal exit from the
    # producing LinearGrads); re-upload ONCE per stream as [1,N,H,Dh] (byte-
    # identical to [N,D]) for the device joint section.
    var d_tatt_4d = _t(tpb.d_att.copy(), [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = _t(ipb.d_att.copy(), [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh]

    # ── sdpa backward (JOINT) -> d_q_rope, d_k_rope, d_v_joint ──
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx,
    )

    # ── rope backward (cos/sin non-learnable -> only d_x) ──
    var d_q_joint = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_joint = rope_backward(sb.d_k, cos, sin, True, ctx)

    # ── split joint q/k/v grads back per stream (txt FIRST) along axis=1 ──
    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(sb.d_v, N_TXT, N_IMG, 1, ctx)
    # cq.d_0 = d_txt_q, cq.d_1 = d_img_q ; likewise k, v. All [1,N,H,Dh] device.

    # ── pre-attention backward per stream ──
    var iprb = _stream_pre_backward[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, saved.img, N_IMG, D, eps, ones_t, ctx,
    )
    var tprb = _stream_pre_backward[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, saved.txt, N_TXT, D, eps, ones_t, ctx,
    )

    # ── combine: stream input grad = residual branch (from post, device) + norm
    #    branch (from pre, host). Read the post residual branch ONCE to host and
    #    sum with the pre's host d_x. ──
    var d_img_x = _add_lists(ipb.d_x, iprb.d_x)
    var d_txt_x = _add_lists(tpb.d_x, tprb.d_x)

    var img_grads = StreamGrads(
        d_img_x^, iprb.d_wqkv.copy(), ipb.d_wproj.copy(), ipb.d_wgu.copy(), ipb.d_wd.copy(),
        iprb.d_q_norm.copy(), iprb.d_k_norm.copy(),
        iprb.d_shift1.copy(), iprb.d_scale1.copy(), ipb.d_gate1.copy(),
        ipb.d_shift2.copy(), ipb.d_scale2.copy(), ipb.d_gate2.copy(),
    )
    var txt_grads = StreamGrads(
        d_txt_x^, tprb.d_wqkv.copy(), tpb.d_wproj.copy(), tpb.d_wgu.copy(), tpb.d_wd.copy(),
        tprb.d_q_norm.copy(), tprb.d_k_norm.copy(),
        tprb.d_shift1.copy(), tprb.d_scale1.copy(), tpb.d_gate1.copy(),
        tpb.d_shift2.copy(), tpb.d_scale2.copy(), tpb.d_gate2.copy(),
    )
    return DoubleBlockGrads(img_grads^, txt_grads^)


# ═══════════════════════════════════════════════════════════════════════════
# LoRA-ON-PROJECTION VARIANT
#
# Targets (per stream, matching OneTrainer Flux2LoRASetup):
#   q, k, v, attention out, ff linear_in, and ff linear_out. Base weights remain
# fused where the checkpoint stores them fused, but LoRA adapters are separate
# OneTrainer modules with separate A/B tensors and gradients.
#
# NOTE: the LoRA-delta helpers have resident device variants. The host-list
# helpers remain for compatibility/parity, while the hot trainer path passes
# `DoubleBlockLoraDevice` and avoids per-use A/B uploads.
# ═══════════════════════════════════════════════════════════════════════════

from serenitymojo.models.klein.lora_block import (
    LoraAdapter, LoraAdapterDevice, lora_adapter_to_device,
    klein_lora_fwd_device, klein_lora_bwd_device,
    klein_lora_fwd_device_resident, klein_lora_bwd_device_resident,
    klein_lora_bwd_device_resident_tensors,
    KleinLoraDeviceGrads, KleinLoraDeviceGradTensors,
)
# LoRA dropout primitives (LoRAModule.py:328, applied to EVERY LoRAModule via
# set_dropout, Flux2LoRASetup.py:65). Identical machinery to single_block; the
# double block reuses it 1:1 so all 96 double-block adapters honor
# config.dropout_probability. p==0 -> mask all-ones -> reduces EXACTLY to the
# verified no-dropout klein_lora_*_device_resident path (parity-byte-identical).
from serenitymojo.models.klein.single_block import (
    LoraDropout,
    _klein_lora_fwd_dropout,
    _klein_lora_bwd_dropout,
    _klein_lora_bwd_dropout_tensors,
)
from serenitymojo.models.klein.klein_direct_lycoris_stack import (
    KleinStreamDirectDoRA, KleinDoubleDirectDoRA,
    KleinStreamDirectOFT, KleinDoubleDirectOFT,
    KleinDirectDoRAGradT, KleinDirectOFTGradT,
    klein_direct_dora_projection_forward_optional,
    klein_direct_dora_projection_backward_optional,
    klein_direct_oft_projection_forward_optional,
    klein_direct_oft_projection_backward_optional,
)


# Per-stream LoRA dropout salts for the 6 SEPARATE OneTrainer adapters
# (q,k,v,out,ff_in,ff_out). OneTrainer gives every LoRAModule its own dropout
# (Flux2LoRASetup.py:65 -> LoRAModule.py:806-813,328). Each slot carries a
# distinct salt so q/k/v/out/ff adapters draw independent masks; default p==0
# keeps every existing call byte-identical to the no-dropout base.
struct StreamLoraDropout(ImplicitlyCopyable, Movable):
    var q: LoraDropout
    var k: LoraDropout
    var v: LoraDropout
    var out: LoraDropout
    var ff_in: LoraDropout
    var ff_out: LoraDropout

    def __init__(
        out self,
        q: LoraDropout = LoraDropout(), k: LoraDropout = LoraDropout(),
        v: LoraDropout = LoraDropout(), out_: LoraDropout = LoraDropout(),
        ff_in: LoraDropout = LoraDropout(), ff_out: LoraDropout = LoraDropout(),
    ):
        self.q = q
        self.k = k
        self.v = v
        self.out = out_
        self.ff_in = ff_in
        self.ff_out = ff_out


struct DoubleBlockLoraDropout(ImplicitlyCopyable, Movable):
    var img: StreamLoraDropout
    var txt: StreamLoraDropout

    def __init__(
        out self,
        img: StreamLoraDropout = StreamLoraDropout(),
        txt: StreamLoraDropout = StreamLoraDropout(),
    ):
        self.img = img
        self.txt = txt


# ── 1:1 OneTrainer SEPARATE-Linear LoRA adapters per stream ──────────────────
# OneTrainer/Flux2LoRASetup.py:57 wraps SEPARATE diffusers nn.Linear modules
# (LoRAModuleWrapper, LoRAModule.py:648-650 wraps EVERY nn.Linear child). The
# `qkv_fusion` in Flux2Model.py:35-72 is a CHECKPOINT-CONVERSION concern (mapping
# diffusers weight NAMES → original fused names) — it does NOT fuse the live LoRA
# adapters. So each separate nn.Linear gets its OWN lora_down/lora_up/alpha:
#
#   diffusers (transformer_flux2.py)         OneTrainer wraps  → our slot
#   attn.to_q        (Linear D→D)  526        img q             q
#   attn.to_k        (Linear D→D)  527        img k             k
#   attn.to_v        (Linear D→D)  528        img v             v
#   attn.to_out.0    (Linear D→D)  535        img to_out.0      out
#   ff.linear_in     (Linear D→2F) 314        img ff_in         ff_in
#   ff.linear_out    (Linear F→D)  316        img ff_out        ff_out
#   attn.add_q_proj  (Linear D→D)  541        txt q             q
#   attn.add_k_proj  (Linear D→D)  542        txt k             k
#   attn.add_v_proj  (Linear D→D)  543        txt v             v
#   attn.to_add_out  (Linear D→D)  544        txt to_add_out    out
#   ff_context.linear_in  (D→2F)   314        txt ff_in         ff_in
#   ff_context.linear_out (F→D)    316        txt ff_out        ff_out
#
# 6 adapters per stream × 2 streams = 12 per double block (vs the old fused 2/2).
# Klein 9B: 8 double blocks × 12 = 96 double adapters (was 32). With 24×2 single
# = 48, the total is 96 + 48 = 144 — matching OneTrainer exactly (was 80).
#
# KEY 1:1 DIFFERENCE FROM THE OLD FUSED qkv ADAPTER
#   A fused qkv adapter [D→3D] with ONE shared lora_down [rank,D] and lora_up
#   [3D,rank] is NOT equivalent to three SEPARATE q/k/v adapters each [D→D] with
#   their OWN lora_down/lora_up. OneTrainer has three separate down-projections,
#   so the q/k/v deltas occupy disjoint rank subspaces. We now apply each
#   separately and add the delta into the matching column band of the fused base
#   qkv output (q→[0:D], k→[D:2D], v→[2D:3D]); the base WEIGHTS stay fused
#   ([3D,D] etc.) since fused-base = column-concat of the separate bases (the
#   forward math is identical), only the LoRA deltas are now separate.
struct StreamLora(Copyable, Movable):
    var q: Optional[LoraAdapter]       # attn.to_q / attn.add_q_proj   (in=D,  out=D)
    var k: Optional[LoraAdapter]       # attn.to_k / attn.add_k_proj   (in=D,  out=D)
    var v: Optional[LoraAdapter]       # attn.to_v / attn.add_v_proj   (in=D,  out=D)
    var out: Optional[LoraAdapter]     # attn.to_out.0 / to_add_out    (in=D,  out=D)
    var ff_in: Optional[LoraAdapter]   # ff.linear_in / ff_context.linear_in  (in=D, out=2F)
    var ff_out: Optional[LoraAdapter]  # ff.linear_out / ff_context.linear_out (in=F, out=D)

    def __init__(
        out self,
        var q: Optional[LoraAdapter], var k: Optional[LoraAdapter],
        var v: Optional[LoraAdapter], var out: Optional[LoraAdapter],
        var ff_in: Optional[LoraAdapter], var ff_out: Optional[LoraAdapter],
    ):
        self.q = q^
        self.k = k^
        self.v = v^
        self.out = out^
        self.ff_in = ff_in^
        self.ff_out = ff_out^


struct StreamLoraDevice(Copyable, Movable):
    var q: Optional[LoraAdapterDevice]
    var k: Optional[LoraAdapterDevice]
    var v: Optional[LoraAdapterDevice]
    var out: Optional[LoraAdapterDevice]
    var ff_in: Optional[LoraAdapterDevice]
    var ff_out: Optional[LoraAdapterDevice]

    def __init__(
        out self,
        var q: Optional[LoraAdapterDevice], var k: Optional[LoraAdapterDevice],
        var v: Optional[LoraAdapterDevice], var out: Optional[LoraAdapterDevice],
        var ff_in: Optional[LoraAdapterDevice], var ff_out: Optional[LoraAdapterDevice],
    ):
        self.q = q^
        self.k = k^
        self.v = v^
        self.out = out^
        self.ff_in = ff_in^
        self.ff_out = ff_out^


def _optional_lora_to_device(
    lo: Optional[LoraAdapter], ctx: DeviceContext
) raises -> Optional[LoraAdapterDevice]:
    if lo:
        return Optional[LoraAdapterDevice](lora_adapter_to_device(lo.value(), ctx))
    return Optional[LoraAdapterDevice](None)


def stream_lora_to_device(lo: StreamLora, ctx: DeviceContext) raises -> StreamLoraDevice:
    return StreamLoraDevice(
        _optional_lora_to_device(lo.q, ctx),
        _optional_lora_to_device(lo.k, ctx),
        _optional_lora_to_device(lo.v, ctx),
        _optional_lora_to_device(lo.out, ctx),
        _optional_lora_to_device(lo.ff_in, ctx),
        _optional_lora_to_device(lo.ff_out, ctx),
    )


struct DoubleBlockLora(Copyable, Movable):
    var img: StreamLora
    var txt: StreamLora

    def __init__(out self, var img: StreamLora, var txt: StreamLora):
        self.img = img^
        self.txt = txt^


struct DoubleBlockLoraDevice(Copyable, Movable):
    var img: StreamLoraDevice
    var txt: StreamLoraDevice

    def __init__(out self, var img: StreamLoraDevice, var txt: StreamLoraDevice):
        self.img = img^
        self.txt = txt^


def double_block_lora_to_device(
    lora: DoubleBlockLora, ctx: DeviceContext
) raises -> DoubleBlockLoraDevice:
    return DoubleBlockLoraDevice(
        stream_lora_to_device(lora.img, ctx),
        stream_lora_to_device(lora.txt, ctx),
    )


# Per-stream LoRA grads: d_A/d_B for the 6 SEPARATE OneTrainer adapters
# (q,k,v,out,ff_in,ff_out). Empty list when the corresponding adapter is absent.
struct StreamLoraGrads(Copyable, Movable):
    var q_d_a: List[Float32]
    var q_d_b: List[Float32]
    var k_d_a: List[Float32]
    var k_d_b: List[Float32]
    var v_d_a: List[Float32]
    var v_d_b: List[Float32]
    var out_d_a: List[Float32]
    var out_d_b: List[Float32]
    var ff_in_d_a: List[Float32]
    var ff_in_d_b: List[Float32]
    var ff_out_d_a: List[Float32]
    var ff_out_d_b: List[Float32]

    def __init__(
        out self,
        var q_d_a: List[Float32], var q_d_b: List[Float32],
        var k_d_a: List[Float32], var k_d_b: List[Float32],
        var v_d_a: List[Float32], var v_d_b: List[Float32],
        var out_d_a: List[Float32], var out_d_b: List[Float32],
        var ff_in_d_a: List[Float32], var ff_in_d_b: List[Float32],
        var ff_out_d_a: List[Float32], var ff_out_d_b: List[Float32],
    ):
        self.q_d_a = q_d_a^
        self.q_d_b = q_d_b^
        self.k_d_a = k_d_a^
        self.k_d_b = k_d_b^
        self.v_d_a = v_d_a^
        self.v_d_b = v_d_b^
        self.out_d_a = out_d_a^
        self.out_d_b = out_d_b^
        self.ff_in_d_a = ff_in_d_a^
        self.ff_in_d_b = ff_in_d_b^
        self.ff_out_d_a = ff_out_d_a^
        self.ff_out_d_b = ff_out_d_b^


struct DoubleBlockLoraGrads(Copyable, Movable):
    var base: DoubleBlockGrads
    var img: StreamLoraGrads
    var txt: StreamLoraGrads

    def __init__(
        out self, var base: DoubleBlockGrads,
        var img: StreamLoraGrads, var txt: StreamLoraGrads,
    ):
        self.base = base^
        self.img = img^
        self.txt = txt^


struct StreamLoraDeviceGrads(Copyable, Movable):
    var d_x: TArc
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]
    var q_d_a: List[Float32]
    var q_d_b: List[Float32]
    var k_d_a: List[Float32]
    var k_d_b: List[Float32]
    var v_d_a: List[Float32]
    var v_d_b: List[Float32]
    var out_d_a: List[Float32]
    var out_d_b: List[Float32]
    var ff_in_d_a: List[Float32]
    var ff_in_d_b: List[Float32]
    var ff_out_d_a: List[Float32]
    var ff_out_d_b: List[Float32]

    def __init__(
        out self,
        var d_x: TArc,
        var d_shift1: List[Float32], var d_scale1: List[Float32], var d_gate1: List[Float32],
        var d_shift2: List[Float32], var d_scale2: List[Float32], var d_gate2: List[Float32],
        var q_d_a: List[Float32], var q_d_b: List[Float32],
        var k_d_a: List[Float32], var k_d_b: List[Float32],
        var v_d_a: List[Float32], var v_d_b: List[Float32],
        var out_d_a: List[Float32], var out_d_b: List[Float32],
        var ff_in_d_a: List[Float32], var ff_in_d_b: List[Float32],
        var ff_out_d_a: List[Float32], var ff_out_d_b: List[Float32],
    ):
        self.d_x = d_x^
        self.d_shift1 = d_shift1^
        self.d_scale1 = d_scale1^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^
        self.q_d_a = q_d_a^
        self.q_d_b = q_d_b^
        self.k_d_a = k_d_a^
        self.k_d_b = k_d_b^
        self.v_d_a = v_d_a^
        self.v_d_b = v_d_b^
        self.out_d_a = out_d_a^
        self.out_d_b = out_d_b^
        self.ff_in_d_a = ff_in_d_a^
        self.ff_in_d_b = ff_in_d_b^
        self.ff_out_d_a = ff_out_d_a^
        self.ff_out_d_b = ff_out_d_b^


struct DoubleBlockLoraDeviceGrads(Copyable, Movable):
    var img: StreamLoraDeviceGrads
    var txt: StreamLoraDeviceGrads

    def __init__(
        out self, var img: StreamLoraDeviceGrads, var txt: StreamLoraDeviceGrads,
    ):
        self.img = img^
        self.txt = txt^


struct StreamLoraDeviceGradTensors(Copyable, Movable):
    var d_x: TArc
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]
    var q_d_a: Optional[TArc]
    var q_d_b: Optional[TArc]
    var k_d_a: Optional[TArc]
    var k_d_b: Optional[TArc]
    var v_d_a: Optional[TArc]
    var v_d_b: Optional[TArc]
    var out_d_a: Optional[TArc]
    var out_d_b: Optional[TArc]
    var ff_in_d_a: Optional[TArc]
    var ff_in_d_b: Optional[TArc]
    var ff_out_d_a: Optional[TArc]
    var ff_out_d_b: Optional[TArc]

    def __init__(
        out self,
        var d_x: TArc,
        var d_shift1: List[Float32], var d_scale1: List[Float32], var d_gate1: List[Float32],
        var d_shift2: List[Float32], var d_scale2: List[Float32], var d_gate2: List[Float32],
        var q_d_a: Optional[TArc], var q_d_b: Optional[TArc],
        var k_d_a: Optional[TArc], var k_d_b: Optional[TArc],
        var v_d_a: Optional[TArc], var v_d_b: Optional[TArc],
        var out_d_a: Optional[TArc], var out_d_b: Optional[TArc],
        var ff_in_d_a: Optional[TArc], var ff_in_d_b: Optional[TArc],
        var ff_out_d_a: Optional[TArc], var ff_out_d_b: Optional[TArc],
    ):
        self.d_x = d_x^
        self.d_shift1 = d_shift1^
        self.d_scale1 = d_scale1^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^
        self.q_d_a = q_d_a^
        self.q_d_b = q_d_b^
        self.k_d_a = k_d_a^
        self.k_d_b = k_d_b^
        self.v_d_a = v_d_a^
        self.v_d_b = v_d_b^
        self.out_d_a = out_d_a^
        self.out_d_b = out_d_b^
        self.ff_in_d_a = ff_in_d_a^
        self.ff_in_d_b = ff_in_d_b^
        self.ff_out_d_a = ff_out_d_a^
        self.ff_out_d_b = ff_out_d_b^


struct DoubleBlockLoraDeviceGradTensors(Copyable, Movable):
    var img: StreamLoraDeviceGradTensors
    var txt: StreamLoraDeviceGradTensors

    def __init__(
        out self, var img: StreamLoraDeviceGradTensors, var txt: StreamLoraDeviceGradTensors,
    ):
        self.img = img^
        self.txt = txt^


# ── LoRA-aware per-stream pre (wqkv delta applied to the qkv output) ─────────
def _stream_pre_lora_resident[
    H: Int, Dh: Int
](
    x: TArc, w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
    drop: StreamLoraDropout = StreamLoraDropout(),
) raises -> _StreamPre:
    var ln1 = layer_norm(x[], ones, zeros, eps, ctx)
    var norm = modulate(ln1, mv.scale1[], mv.shift1[], ctx)
    var no_bias = Optional[Tensor](None)
    var qkv = linear(norm, w.wqkv[], no_bias^, ctx)   # [N,3D]
    # SEPARATE q/k/v LoRA (OneTrainer attn.to_q/to_k/to_v, transformer_flux2.py
    # :526-528; txt add_q/k/v_proj :541-543). Each adapter has its OWN lora_down
    # (in=D) → distinct rank subspace, so we split the base qkv into the three
    # [N,D] column bands and add the matching per-adapter delta. NOT equivalent to
    # one fused [D→3D] adapter with a shared lora_down. Each adapter also carries
    # its own dropout (LoRAModule.py:328); p==0 -> identity.
    var q_pre_flat = slice(qkv, 1, 0, D, ctx)
    var k_pre_flat = slice(qkv, 1, D, D, ctx)
    var v_flat = slice(qkv, 1, 2 * D, D, ctx)
    if lo.q:
        q_pre_flat = add(q_pre_flat, _klein_lora_fwd_dropout(norm, lo.q.value(), N, drop.q, ctx), ctx)
    if lo.k:
        k_pre_flat = add(k_pre_flat, _klein_lora_fwd_dropout(norm, lo.k.value(), N, drop.k, ctx), ctx)
    if lo.v:
        v_flat = add(v_flat, _klein_lora_fwd_dropout(norm, lo.v.value(), N, drop.v, ctx), ctx)
    var q_pre = reshape_owned(q_pre_flat^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, N, H, Dh])
    var v = reshape_owned(v_flat^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPre(
        TArc(ln1^), TArc(norm^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
    )


# ── LoRA-aware per-stream post (wproj delta applied to the out projection) ───
def _stream_post_lora_resident(
    x: TArc, att: TArc, w: StreamWeights, mv: ModVecsDevice,
    lo: StreamLoraDevice, N: Int, D: Int, F: Int, eps: Float32,
    ones: Tensor, zeros: Tensor, ctx: DeviceContext,
    drop: StreamLoraDropout = StreamLoraDropout(),
) raises -> _StreamPost:
    var no_bias = Optional[Tensor](None)
    var out = linear(att[], w.wproj[], no_bias^, ctx)   # [N,D]
    # LoRA on attn out-proj (OneTrainer attn.to_out.0 / attn.to_add_out,
    # transformer_flux2.py:535/544): input att [N,D]; delta [N,D] added to out.
    # Own dropout (LoRAModule.py:328); p==0 -> identity.
    if lo.out:
        out = add(out, _klein_lora_fwd_dropout(att[], lo.out.value(), N, drop.out, ctx), ctx)
    var attn_res = residual_gate(x[], mv.gate1[], out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, mv.scale2[], mv.shift2[], ctx)
    var no_bias2 = Optional[Tensor](None)
    var gu = linear(mlp_in, w.wgu[], no_bias2^, ctx)   # [N,2F]
    # LoRA on ff.linear_in / ff_context.linear_in (transformer_flux2.py:314,
    # Flux2Model.py:58/63): one adapter in=D out=2F added to the FULL gu output.
    if lo.ff_in:
        gu = add(gu, _klein_lora_fwd_dropout(mlp_in, lo.ff_in.value(), N, drop.ff_in, ctx), ctx)
    var gate = slice(gu, 1, 0, F, ctx)
    var up = slice(gu, 1, F, F, ctx)
    var act = swiglu(gate, up, ctx)
    var no_bias3 = Optional[Tensor](None)
    var mlp = linear(act, w.wd[], no_bias3^, ctx)   # [N,D]
    # LoRA on ff.linear_out / ff_context.linear_out (transformer_flux2.py:316,
    # Flux2Model.py:59/64): adapter in=F out=D added to mlp (input is act [N,F]).
    if lo.ff_out:
        mlp = add(mlp, _klein_lora_fwd_dropout(act, lo.ff_out.value(), N, drop.ff_out, ctx), ctx)
    var final = residual_gate(attn_res, mv.gate2[], mlp, ctx)
    return _StreamPost(
        TArc(final^), TArc(attn_res^), TArc(ln2^), TArc(mlp_in^),
        TArc(gu^), TArc(gate^), TArc(up^), TArc(act^),
    )


# ── FORWARD of one DOUBLE block WITH LoRA on the attention projections ───────
# Identical graph to double_block_forward, but the two stream qkv/proj linears
# carry an optional LoRA delta. `saved` is the LoRA-MODIFIED activations (so the
# backward sees the correct q/k/v/att/attn_res etc.).
def double_block_lora_forward_device_resident[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_x: TArc, txt_x: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice, lora: DoubleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    drop: DoubleBlockLoraDropout = DoubleBlockLoraDropout(),
) raises -> DoubleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var norm_dtype = img_x[].dtype()
    var ones_t = _t_dtype(_ones(D), [D], norm_dtype, ctx)
    var zeros_t = _t_dtype(_zeros(D), [D], norm_dtype, ctx)

    var ip = _stream_pre_lora_resident[H, Dh](img_x, w.img, img_mod, lora.img, N_IMG, D, eps, ones_t, zeros_t, ctx, drop.img)
    var tp = _stream_pre_lora_resident[H, Dh](txt_x, w.txt, txt_mod, lora.txt, N_TXT, D, eps, ones_t, zeros_t, ctx, drop.txt)

    var q = concat(1, ctx, tp.q_rms[], ip.q_rms[])
    var k = concat(1, ctx, tp.k_rms[], ip.k_rms[])
    var v = concat(1, ctx, tp.v[], ip.v[])

    var q_rope = rope_interleaved(q, cos, sin, ctx)
    var k_rope = rope_interleaved(k, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG, D]))

    var ipost = _stream_post_lora_resident(
        img_x, img_att, w.img, img_mod, lora.img, N_IMG, D, F, eps, ones_t, zeros_t, ctx, drop.img)
    var tpost = _stream_post_lora_resident(
        txt_x, txt_att, w.txt, txt_mod, lora.txt, N_TXT, D, F, eps, ones_t, zeros_t, ctx, drop.txt)

    var img_saved = _make_saved(img_x, ip, img_att, ipost)
    var txt_saved = _make_saved(txt_x, tp, txt_att, tpost)
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^)
    )

    return DoubleBlockDeviceForward(ipost.out.copy(), tpost.out.copy(), saved^)


def double_block_lora_forward_device_resident_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_x: TArc, txt_x: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice, lora: DoubleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    drop: DoubleBlockLoraDropout = DoubleBlockLoraDropout(),
) raises -> DoubleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ip = _stream_pre_lora_resident[H, Dh](
        img_x, w.img, img_mod, lora.img, N_IMG, D, eps, norm_ones, norm_zeros, ctx, drop.img)
    var tp = _stream_pre_lora_resident[H, Dh](
        txt_x, w.txt, txt_mod, lora.txt, N_TXT, D, eps, norm_ones, norm_zeros, ctx, drop.txt)

    var qk_mark = scratch.mark()
    var q = concat2_scratch(1, ctx, scratch, tp.q_rms[], ip.q_rms[])
    var k = concat2_scratch(1, ctx, scratch, tp.k_rms[], ip.k_rms[])
    var v = concat(1, ctx, tp.v[], ip.v[])

    var q_rope = rope_interleaved(q, cos, sin, ctx)
    var k_rope = rope_interleaved(k, cos, sin, ctx)
    scratch.rewind(qk_mark)
    var att: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG, D]))

    var ipost = _stream_post_lora_resident(
        img_x, img_att, w.img, img_mod, lora.img, N_IMG, D, F, eps, norm_ones, norm_zeros, ctx, drop.img)
    var tpost = _stream_post_lora_resident(
        txt_x, txt_att, w.txt, txt_mod, lora.txt, N_TXT, D, F, eps, norm_ones, norm_zeros, ctx, drop.txt)

    var img_saved = _make_saved(img_x, ip, img_att, ipost)
    var txt_saved = _make_saved(txt_x, tp, txt_att, tpost)
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )

    return DoubleBlockDeviceForward(ipost.out.copy(), tpost.out.copy(), saved^)


def double_block_lora_predict_device_resident_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_x: TArc, txt_x: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice, lora: DoubleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> DoubleBlockDeviceOutput:
    """Inference-only LoRA double block: same math, no backward tape."""
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ip = _stream_pre_lora_resident[H, Dh](
        img_x, w.img, img_mod, lora.img, N_IMG, D, eps, norm_ones, norm_zeros, ctx)
    var tp = _stream_pre_lora_resident[H, Dh](
        txt_x, w.txt, txt_mod, lora.txt, N_TXT, D, eps, norm_ones, norm_zeros, ctx)

    var qk_mark = scratch.mark()
    var q = concat2_scratch(1, ctx, scratch, tp.q_rms[], ip.q_rms[])
    var k = concat2_scratch(1, ctx, scratch, tp.k_rms[], ip.k_rms[])
    var v = concat(1, ctx, tp.v[], ip.v[])

    var q_rope = rope_interleaved(q, cos, sin, ctx)
    var k_rope = rope_interleaved(k, cos, sin, ctx)
    scratch.rewind(qk_mark)
    var att: Tensor
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
    else:
        att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG, D]))

    var ipost = _stream_post_lora_resident(
        img_x, img_att, w.img, img_mod, lora.img, N_IMG, D, F, eps, norm_ones, norm_zeros, ctx)
    var tpost = _stream_post_lora_resident(
        txt_x, txt_att, w.txt, txt_mod, lora.txt, N_TXT, D, F, eps, norm_ones, norm_zeros, ctx)

    return DoubleBlockDeviceOutput(ipost.out.copy(), tpost.out.copy())


def _weight_rows(w: Tensor, row_start: Int, row_count: Int, ctx: DeviceContext) raises -> Tensor:
    return slice(w, 0, row_start, row_count, ctx)


def _stream_pre_direct_dora_resident[
    H: Int, Dh: Int
](
    x: TArc, w: StreamWeights, mv: ModVecsDevice, ad: KleinStreamDirectDoRA,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPre:
    var ln1 = layer_norm(x[], ones, zeros, eps, ctx)
    var norm = modulate(ln1, mv.scale1[], mv.shift1[], ctx)
    var wq = _weight_rows(w.wqkv[], 0, D, ctx)
    var wk = _weight_rows(w.wqkv[], D, D, ctx)
    var wv = _weight_rows(w.wqkv[], 2 * D, D, ctx)
    var q_pre_flat = klein_direct_dora_projection_forward_optional(
        norm, wq, ad.q, N, ctx,
    )
    var k_pre_flat = klein_direct_dora_projection_forward_optional(
        norm, wk, ad.k, N, ctx,
    )
    var v_flat = klein_direct_dora_projection_forward_optional(
        norm, wv, ad.v, N, ctx,
    )
    var q_pre = reshape_owned(q_pre_flat^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, N, H, Dh])
    var v = reshape_owned(v_flat^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPre(
        TArc(ln1^), TArc(norm^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
    )


def _stream_post_direct_dora_resident(
    x: TArc, att: TArc, w: StreamWeights, mv: ModVecsDevice,
    ad: KleinStreamDirectDoRA, N: Int, D: Int, F: Int, eps: Float32,
    ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPost:
    var out = klein_direct_dora_projection_forward_optional(
        att[], w.wproj[], ad.out, N, ctx,
    )
    var attn_res = residual_gate(x[], mv.gate1[], out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, mv.scale2[], mv.shift2[], ctx)
    var gu = klein_direct_dora_projection_forward_optional(
        mlp_in, w.wgu[], ad.ff_in, N, ctx,
    )
    var gate = slice(gu, 1, 0, F, ctx)
    var up = slice(gu, 1, F, F, ctx)
    var act = swiglu(gate, up, ctx)
    var mlp = klein_direct_dora_projection_forward_optional(
        act, w.wd[], ad.ff_out, N, ctx,
    )
    var final = residual_gate(attn_res, mv.gate2[], mlp, ctx)
    return _StreamPost(
        TArc(final^), TArc(attn_res^), TArc(ln2^), TArc(mlp_in^),
        TArc(gu^), TArc(gate^), TArc(up^), TArc(act^),
    )


def _stream_pre_direct_oft_resident[
    H: Int, Dh: Int
](
    x: TArc, w: StreamWeights, mv: ModVecsDevice, ad: KleinStreamDirectOFT,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPre:
    var ln1 = layer_norm(x[], ones, zeros, eps, ctx)
    var norm = modulate(ln1, mv.scale1[], mv.shift1[], ctx)
    var wq = _weight_rows(w.wqkv[], 0, D, ctx)
    var wk = _weight_rows(w.wqkv[], D, D, ctx)
    var wv = _weight_rows(w.wqkv[], 2 * D, D, ctx)
    var q_pre_flat = klein_direct_oft_projection_forward_optional(
        norm, wq, ad.q, N, ctx,
    )
    var k_pre_flat = klein_direct_oft_projection_forward_optional(
        norm, wk, ad.k, N, ctx,
    )
    var v_flat = klein_direct_oft_projection_forward_optional(
        norm, wv, ad.v, N, ctx,
    )
    var q_pre = reshape_owned(q_pre_flat^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, N, H, Dh])
    var v = reshape_owned(v_flat^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPre(
        TArc(ln1^), TArc(norm^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
    )


def _stream_post_direct_oft_resident(
    x: TArc, att: TArc, w: StreamWeights, mv: ModVecsDevice,
    ad: KleinStreamDirectOFT, N: Int, D: Int, F: Int, eps: Float32,
    ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPost:
    var out = klein_direct_oft_projection_forward_optional(
        att[], w.wproj[], ad.out, N, ctx,
    )
    var attn_res = residual_gate(x[], mv.gate1[], out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, mv.scale2[], mv.shift2[], ctx)
    var gu = klein_direct_oft_projection_forward_optional(
        mlp_in, w.wgu[], ad.ff_in, N, ctx,
    )
    var gate = slice(gu, 1, 0, F, ctx)
    var up = slice(gu, 1, F, F, ctx)
    var act = swiglu(gate, up, ctx)
    var mlp = klein_direct_oft_projection_forward_optional(
        act, w.wd[], ad.ff_out, N, ctx,
    )
    var final = residual_gate(attn_res, mv.gate2[], mlp, ctx)
    return _StreamPost(
        TArc(final^), TArc(attn_res^), TArc(ln2^), TArc(mlp_in^),
        TArc(gu^), TArc(gate^), TArc(up^), TArc(act^),
    )


def double_block_direct_dora_forward_device_resident_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_x: TArc, txt_x: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    dora: KleinDoubleDirectDoRA,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> DoubleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ip = _stream_pre_direct_dora_resident[H, Dh](
        img_x, w.img, img_mod, dora.img, N_IMG, D, eps, norm_ones, norm_zeros, ctx,
    )
    var tp = _stream_pre_direct_dora_resident[H, Dh](
        txt_x, w.txt, txt_mod, dora.txt, N_TXT, D, eps, norm_ones, norm_zeros, ctx,
    )

    var qk_mark = scratch.mark()
    var q = concat2_scratch(1, ctx, scratch, tp.q_rms[], ip.q_rms[])
    var k = concat2_scratch(1, ctx, scratch, tp.k_rms[], ip.k_rms[])
    var v = concat(1, ctx, tp.v[], ip.v[])

    var q_rope = rope_interleaved(q, cos, sin, ctx)
    var k_rope = rope_interleaved(k, cos, sin, ctx)
    scratch.rewind(qk_mark)
    var att: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG, D]))

    var ipost = _stream_post_direct_dora_resident(
        img_x, img_att, w.img, img_mod, dora.img, N_IMG, D, F, eps, norm_ones, norm_zeros, ctx)
    var tpost = _stream_post_direct_dora_resident(
        txt_x, txt_att, w.txt, txt_mod, dora.txt, N_TXT, D, F, eps, norm_ones, norm_zeros, ctx)

    var img_saved = _make_saved(img_x, ip, img_att, ipost)
    var txt_saved = _make_saved(txt_x, tp, txt_att, tpost)
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )

    return DoubleBlockDeviceForward(ipost.out.copy(), tpost.out.copy(), saved^)


def double_block_direct_oft_forward_device_resident_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_x: TArc, txt_x: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    oft: KleinDoubleDirectOFT,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor, norm_zeros: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
) raises -> DoubleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ip = _stream_pre_direct_oft_resident[H, Dh](
        img_x, w.img, img_mod, oft.img, N_IMG, D, eps, norm_ones, norm_zeros, ctx,
    )
    var tp = _stream_pre_direct_oft_resident[H, Dh](
        txt_x, w.txt, txt_mod, oft.txt, N_TXT, D, eps, norm_ones, norm_zeros, ctx,
    )

    var qk_mark = scratch.mark()
    var q = concat2_scratch(1, ctx, scratch, tp.q_rms[], ip.q_rms[])
    var k = concat2_scratch(1, ctx, scratch, tp.k_rms[], ip.k_rms[])
    var v = concat(1, ctx, tp.v[], ip.v[])

    var q_rope = rope_interleaved(q, cos, sin, ctx)
    var k_rope = rope_interleaved(k, cos, sin, ctx)
    scratch.rewind(qk_mark)
    var att: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    comptime if KLEIN_SDPA_FLASH:
        var ff = sdpa_flash_train_fwd_f32[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG, D]))

    var ipost = _stream_post_direct_oft_resident(
        img_x, img_att, w.img, img_mod, oft.img, N_IMG, D, F, eps, norm_ones, norm_zeros, ctx)
    var tpost = _stream_post_direct_oft_resident(
        txt_x, txt_att, w.txt, txt_mod, oft.txt, N_TXT, D, F, eps, norm_ones, norm_zeros, ctx)

    var img_saved = _make_saved(img_x, ip, img_att, ipost)
    var txt_saved = _make_saved(txt_x, tp, txt_att, tpost)
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )

    return DoubleBlockDeviceForward(ipost.out.copy(), tpost.out.copy(), saved^)


struct StreamDirectDoRAGradsT(Copyable, Movable):
    var d_x: TArc
    var q: KleinDirectDoRAGradT
    var k: KleinDirectDoRAGradT
    var v: KleinDirectDoRAGradT
    var out: KleinDirectDoRAGradT
    var ff_in: KleinDirectDoRAGradT
    var ff_out: KleinDirectDoRAGradT

    def __init__(
        out self, var d_x: TArc,
        var q: KleinDirectDoRAGradT, var k: KleinDirectDoRAGradT,
        var v: KleinDirectDoRAGradT, var out: KleinDirectDoRAGradT,
        var ff_in: KleinDirectDoRAGradT, var ff_out: KleinDirectDoRAGradT,
    ):
        self.d_x = d_x^
        self.q = q^
        self.k = k^
        self.v = v^
        self.out = out^
        self.ff_in = ff_in^
        self.ff_out = ff_out^


struct DoubleBlockDirectDoRAGradsT(Copyable, Movable):
    var img: StreamDirectDoRAGradsT
    var txt: StreamDirectDoRAGradsT

    def __init__(
        out self, var img: StreamDirectDoRAGradsT, var txt: StreamDirectDoRAGradsT,
    ):
        self.img = img^
        self.txt = txt^


struct StreamDirectOFTGradsT(Copyable, Movable):
    var d_x: TArc
    var q: KleinDirectOFTGradT
    var k: KleinDirectOFTGradT
    var v: KleinDirectOFTGradT
    var out: KleinDirectOFTGradT
    var ff_in: KleinDirectOFTGradT
    var ff_out: KleinDirectOFTGradT

    def __init__(
        out self, var d_x: TArc,
        var q: KleinDirectOFTGradT, var k: KleinDirectOFTGradT,
        var v: KleinDirectOFTGradT, var out: KleinDirectOFTGradT,
        var ff_in: KleinDirectOFTGradT, var ff_out: KleinDirectOFTGradT,
    ):
        self.d_x = d_x^
        self.q = q^
        self.k = k^
        self.v = v^
        self.out = out^
        self.ff_in = ff_in^
        self.ff_out = ff_out^


struct DoubleBlockDirectOFTGradsT(Copyable, Movable):
    var img: StreamDirectOFTGradsT
    var txt: StreamDirectOFTGradsT

    def __init__(
        out self, var img: StreamDirectOFTGradsT, var txt: StreamDirectOFTGradsT,
    ):
        self.img = img^
        self.txt = txt^


struct _StreamPostDirectDoRABack(Copyable, Movable):
    var d_x: TArc
    var d_att: TArc
    var out: KleinDirectDoRAGradT
    var ff_in: KleinDirectDoRAGradT
    var ff_out: KleinDirectDoRAGradT

    def __init__(
        out self, var d_x: TArc, var d_att: TArc,
        var out_g: KleinDirectDoRAGradT,
        var ff_in: KleinDirectDoRAGradT, var ff_out: KleinDirectDoRAGradT,
    ):
        self.d_x = d_x^
        self.d_att = d_att^
        self.out = out_g^
        self.ff_in = ff_in^
        self.ff_out = ff_out^


struct _StreamPreDirectDoRABack(Copyable, Movable):
    var d_x: TArc
    var q: KleinDirectDoRAGradT
    var k: KleinDirectDoRAGradT
    var v: KleinDirectDoRAGradT

    def __init__(
        out self, var d_x: TArc,
        var q: KleinDirectDoRAGradT, var k: KleinDirectDoRAGradT,
        var v: KleinDirectDoRAGradT,
    ):
        self.d_x = d_x^
        self.q = q^
        self.k = k^
        self.v = v^


struct _StreamPostDirectOFTBack(Copyable, Movable):
    var d_x: TArc
    var d_att: TArc
    var out: KleinDirectOFTGradT
    var ff_in: KleinDirectOFTGradT
    var ff_out: KleinDirectOFTGradT

    def __init__(
        out self, var d_x: TArc, var d_att: TArc,
        var out_g: KleinDirectOFTGradT,
        var ff_in: KleinDirectOFTGradT, var ff_out: KleinDirectOFTGradT,
    ):
        self.d_x = d_x^
        self.d_att = d_att^
        self.out = out_g^
        self.ff_in = ff_in^
        self.ff_out = ff_out^


struct _StreamPreDirectOFTBack(Copyable, Movable):
    var d_x: TArc
    var q: KleinDirectOFTGradT
    var k: KleinDirectOFTGradT
    var v: KleinDirectOFTGradT

    def __init__(
        out self, var d_x: TArc,
        var q: KleinDirectOFTGradT, var k: KleinDirectOFTGradT,
        var v: KleinDirectOFTGradT,
    ):
        self.d_x = d_x^
        self.q = q^
        self.k = k^
        self.v = v^


def _stream_post_backward_direct_dora_resident_scratch(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecsDevice, ad: KleinStreamDirectDoRA, sv: StreamSaved,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
) raises -> _StreamPostDirectDoRABack:
    var grg2: GateResidualGrads
    if compute_aux_grads:
        var mlp_y = klein_direct_dora_projection_forward_optional(
            sv.act[], w.wd[], ad.ff_out, N, ctx,
        )
        grg2 = gate_residual_backward(
            d_out[], sv.attn_res[], mv.gate2[], mlp_y, ctx
        )
    else:
        grg2 = gate_residual_backward_dxdy(d_out[], mv.gate2[], ctx)

    var post_mark = scratch.mark()
    var bw_ff_out = klein_direct_dora_projection_backward_optional(
        grg2.d_y, sv.act[], w.wd[], ad.ff_out, N, F, D, ctx,
    )
    var sgb = swiglu_backward(bw_ff_out.d_x, sv.gate[], sv.up[], ctx)
    var d_gu = concat2_scratch(1, ctx, scratch, sgb.d_gate, sgb.d_up)
    var bw_ff_in = klein_direct_dora_projection_backward_optional(
        d_gu, sv.mlp_in[], w.wgu[], ad.ff_in, N, D, 2 * F, ctx,
    )

    var mb2 = modulate_backward(bw_ff_in.d_x, sv.ln2[], mv.scale2[], ctx, compute_aux_grads)
    scratch.rewind(post_mark)
    var d_attn_res_norm = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    var d_attn_res_total = TArc(add(grg2.d_x, d_attn_res_norm, ctx))

    var grg1: GateResidualGrads
    if compute_aux_grads:
        var proj_out = klein_direct_dora_projection_forward_optional(
            att[], w.wproj[], ad.out, N, ctx,
        )
        grg1 = gate_residual_backward(
            d_attn_res_total[], x[], mv.gate1[], proj_out, ctx
        )
    else:
        grg1 = gate_residual_backward_dxdy(d_attn_res_total[], mv.gate1[], ctx)

    var bw_out = klein_direct_dora_projection_backward_optional(
        grg1.d_y, att[], w.wproj[], ad.out, N, D, D, ctx,
    )
    return _StreamPostDirectDoRABack(
        d_attn_res_total^, TArc(bw_out.d_x.clone(ctx)),
        bw_out.dora.copy(), bw_ff_in.dora.copy(), bw_ff_out.dora.copy(),
    )


def _stream_pre_backward_direct_dora_resident_scratch[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v_flat: Tensor,
    w: StreamWeights, mv: ModVecsDevice, ad: KleinStreamDirectDoRA, sv: StreamSaved,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
) raises -> _StreamPreDirectDoRABack:
    var scratch_mark = scratch.mark()
    var d_q_pre = rms_norm_backward_dx(d_q_rms, sv.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre = rms_norm_backward_dx(d_k_rms, sv.k_pre[], w.k_norm[], eps, ctx)
    var d_q_pre_flat = reshape_owned(d_q_pre^, [N, D])
    var d_k_pre_flat = reshape_owned(d_k_pre^, [N, D])
    var wq = _weight_rows(w.wqkv[], 0, D, ctx)
    var wk = _weight_rows(w.wqkv[], D, D, ctx)
    var wv = _weight_rows(w.wqkv[], 2 * D, D, ctx)
    var bw_q = klein_direct_dora_projection_backward_optional(
        d_q_pre_flat, sv.norm[], wq, ad.q, N, D, D, ctx,
    )
    var bw_k = klein_direct_dora_projection_backward_optional(
        d_k_pre_flat, sv.norm[], wk, ad.k, N, D, D, ctx,
    )
    var bw_v = klein_direct_dora_projection_backward_optional(
        d_v_flat, sv.norm[], wv, ad.v, N, D, D, ctx,
    )
    var d_norm_t = add(add(bw_q.d_x, bw_k.d_x, ctx), bw_v.d_x, ctx)
    var mb1 = modulate_backward(d_norm_t, sv.ln1[], mv.scale1[], ctx, compute_aux_grads)
    var d_x_norm_t = layer_norm_backward_dx(mb1.d_x, sv.x[], ones, eps, ctx)
    var out = _StreamPreDirectDoRABack(
        TArc(d_x_norm_t^), bw_q.dora.copy(), bw_k.dora.copy(), bw_v.dora.copy(),
    )
    scratch.rewind(scratch_mark)
    return out^


def _stream_post_backward_direct_oft_resident_scratch(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecsDevice, ad: KleinStreamDirectOFT, sv: StreamSaved,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
) raises -> _StreamPostDirectOFTBack:
    var grg2: GateResidualGrads
    if compute_aux_grads:
        var mlp_y = klein_direct_oft_projection_forward_optional(
            sv.act[], w.wd[], ad.ff_out, N, ctx,
        )
        grg2 = gate_residual_backward(
            d_out[], sv.attn_res[], mv.gate2[], mlp_y, ctx
        )
    else:
        grg2 = gate_residual_backward_dxdy(d_out[], mv.gate2[], ctx)

    var post_mark = scratch.mark()
    var bw_ff_out = klein_direct_oft_projection_backward_optional(
        grg2.d_y, sv.act[], w.wd[], ad.ff_out, N, F, D, ctx,
    )
    var sgb = swiglu_backward(bw_ff_out.d_x, sv.gate[], sv.up[], ctx)
    var d_gu = concat2_scratch(1, ctx, scratch, sgb.d_gate, sgb.d_up)
    var bw_ff_in = klein_direct_oft_projection_backward_optional(
        d_gu, sv.mlp_in[], w.wgu[], ad.ff_in, N, D, 2 * F, ctx,
    )

    var mb2 = modulate_backward(bw_ff_in.d_x, sv.ln2[], mv.scale2[], ctx, compute_aux_grads)
    scratch.rewind(post_mark)
    var d_attn_res_norm = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    var d_attn_res_total = TArc(add(grg2.d_x, d_attn_res_norm, ctx))

    var grg1: GateResidualGrads
    if compute_aux_grads:
        var proj_out = klein_direct_oft_projection_forward_optional(
            att[], w.wproj[], ad.out, N, ctx,
        )
        grg1 = gate_residual_backward(
            d_attn_res_total[], x[], mv.gate1[], proj_out, ctx
        )
    else:
        grg1 = gate_residual_backward_dxdy(d_attn_res_total[], mv.gate1[], ctx)

    var bw_out = klein_direct_oft_projection_backward_optional(
        grg1.d_y, att[], w.wproj[], ad.out, N, D, D, ctx,
    )
    return _StreamPostDirectOFTBack(
        d_attn_res_total^, TArc(bw_out.d_x.clone(ctx)),
        bw_out.oft.copy(), bw_ff_in.oft.copy(), bw_ff_out.oft.copy(),
    )


def _stream_pre_backward_direct_oft_resident_scratch[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v_flat: Tensor,
    w: StreamWeights, mv: ModVecsDevice, ad: KleinStreamDirectOFT, sv: StreamSaved,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
) raises -> _StreamPreDirectOFTBack:
    var scratch_mark = scratch.mark()
    var d_q_pre = rms_norm_backward_dx(d_q_rms, sv.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre = rms_norm_backward_dx(d_k_rms, sv.k_pre[], w.k_norm[], eps, ctx)
    var d_q_pre_flat = reshape_owned(d_q_pre^, [N, D])
    var d_k_pre_flat = reshape_owned(d_k_pre^, [N, D])
    var wq = _weight_rows(w.wqkv[], 0, D, ctx)
    var wk = _weight_rows(w.wqkv[], D, D, ctx)
    var wv = _weight_rows(w.wqkv[], 2 * D, D, ctx)
    var bw_q = klein_direct_oft_projection_backward_optional(
        d_q_pre_flat, sv.norm[], wq, ad.q, N, D, D, ctx,
    )
    var bw_k = klein_direct_oft_projection_backward_optional(
        d_k_pre_flat, sv.norm[], wk, ad.k, N, D, D, ctx,
    )
    var bw_v = klein_direct_oft_projection_backward_optional(
        d_v_flat, sv.norm[], wv, ad.v, N, D, D, ctx,
    )
    var d_norm_t = add(add(bw_q.d_x, bw_k.d_x, ctx), bw_v.d_x, ctx)
    var mb1 = modulate_backward(d_norm_t, sv.ln1[], mv.scale1[], ctx, compute_aux_grads)
    var d_x_norm_t = layer_norm_backward_dx(mb1.d_x, sv.x[], ones, eps, ctx)
    var out = _StreamPreDirectOFTBack(
        TArc(d_x_norm_t^), bw_q.oft.copy(), bw_k.oft.copy(), bw_v.oft.copy(),
    )
    scratch.rewind(scratch_mark)
    return out^


def double_block_direct_dora_backward_device_resident_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    dora: KleinDoubleDirectDoRA,
    saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
) raises -> DoubleBlockDirectDoRAGradsT:
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ipb = _stream_post_backward_direct_dora_resident_scratch(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, dora.img, saved.img,
        N_IMG, D, F, eps, norm_ones, ctx, scratch, compute_aux_grads,
    )
    var tpb = _stream_post_backward_direct_dora_resident_scratch(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, dora.txt, saved.txt,
        N_TXT, D, F, eps, norm_ones, ctx, scratch, compute_aux_grads,
    )

    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat2_scratch(1, ctx, scratch, d_tatt_4d, d_iatt_4d)

    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    comptime if KLEIN_SDPA_FLASH:
        if not saved.flash_stats:
            raise Error("double direct DoRA bwd: missing flash tape")
        var fb = sdpa_flash_backward_f32[1, S, H, Dh](
            saved.flash_q.value(), saved.flash_k.value(),
            saved.flash_v.value(), saved.flash_o.value(),
            saved.flash_stats.value(), d_att_joint, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward_scratch[1, S, H, Dh](
            saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx, scratch,
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_q_joint = rope_backward(d_q_sb, cos, sin, True, ctx)
    var d_k_joint = rope_backward(d_k_sb, cos, sin, True, ctx)

    var d_txt_q = slice_scratch(d_q_joint, 1, 0, N_TXT, ctx, scratch)
    var d_img_q = slice_scratch(d_q_joint, 1, N_TXT, N_IMG, ctx, scratch)
    var d_txt_k = slice_scratch(d_k_joint, 1, 0, N_TXT, ctx, scratch)
    var d_img_k = slice_scratch(d_k_joint, 1, N_TXT, N_IMG, ctx, scratch)
    var d_txt_v = slice_scratch(d_v_sb, 1, 0, N_TXT, ctx, scratch)
    var d_img_v = slice_scratch(d_v_sb, 1, N_TXT, N_IMG, ctx, scratch)
    reshape_in_place(d_img_v, [N_IMG, D])
    reshape_in_place(d_txt_v, [N_TXT, D])

    var iprb = _stream_pre_backward_direct_dora_resident_scratch[H, Dh](
        d_img_q, d_img_k, d_img_v, w.img, img_mod, dora.img, saved.img,
        N_IMG, D, eps, norm_ones, ctx, scratch, compute_aux_grads,
    )
    var tprb = _stream_pre_backward_direct_dora_resident_scratch[H, Dh](
        d_txt_q, d_txt_k, d_txt_v, w.txt, txt_mod, dora.txt, saved.txt,
        N_TXT, D, eps, norm_ones, ctx, scratch, compute_aux_grads,
    )

    var d_img_x_t = TArc(add(ipb.d_x[], iprb.d_x[], ctx))
    var d_txt_x_t = TArc(add(tpb.d_x[], tprb.d_x[], ctx))
    var img_grads = StreamDirectDoRAGradsT(
        d_img_x_t^, iprb.q.copy(), iprb.k.copy(), iprb.v.copy(),
        ipb.out.copy(), ipb.ff_in.copy(), ipb.ff_out.copy(),
    )
    var txt_grads = StreamDirectDoRAGradsT(
        d_txt_x_t^, tprb.q.copy(), tprb.k.copy(), tprb.v.copy(),
        tpb.out.copy(), tpb.ff_in.copy(), tpb.ff_out.copy(),
    )
    var out = DoubleBlockDirectDoRAGradsT(img_grads^, txt_grads^)
    scratch.rewind(scratch_mark)
    return out^


def double_block_direct_oft_backward_device_resident_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    oft: KleinDoubleDirectOFT,
    saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
) raises -> DoubleBlockDirectOFTGradsT:
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ipb = _stream_post_backward_direct_oft_resident_scratch(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, oft.img, saved.img,
        N_IMG, D, F, eps, norm_ones, ctx, scratch, compute_aux_grads,
    )
    var tpb = _stream_post_backward_direct_oft_resident_scratch(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, oft.txt, saved.txt,
        N_TXT, D, F, eps, norm_ones, ctx, scratch, compute_aux_grads,
    )

    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat2_scratch(1, ctx, scratch, d_tatt_4d, d_iatt_4d)

    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    comptime if KLEIN_SDPA_FLASH:
        if not saved.flash_stats:
            raise Error("double direct OFT bwd: missing flash tape")
        var fb = sdpa_flash_backward_f32[1, S, H, Dh](
            saved.flash_q.value(), saved.flash_k.value(),
            saved.flash_v.value(), saved.flash_o.value(),
            saved.flash_stats.value(), d_att_joint, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward_scratch[1, S, H, Dh](
            saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx, scratch,
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_q_joint = rope_backward(d_q_sb, cos, sin, True, ctx)
    var d_k_joint = rope_backward(d_k_sb, cos, sin, True, ctx)

    var d_txt_q = slice_scratch(d_q_joint, 1, 0, N_TXT, ctx, scratch)
    var d_img_q = slice_scratch(d_q_joint, 1, N_TXT, N_IMG, ctx, scratch)
    var d_txt_k = slice_scratch(d_k_joint, 1, 0, N_TXT, ctx, scratch)
    var d_img_k = slice_scratch(d_k_joint, 1, N_TXT, N_IMG, ctx, scratch)
    var d_txt_v = slice_scratch(d_v_sb, 1, 0, N_TXT, ctx, scratch)
    var d_img_v = slice_scratch(d_v_sb, 1, N_TXT, N_IMG, ctx, scratch)
    reshape_in_place(d_img_v, [N_IMG, D])
    reshape_in_place(d_txt_v, [N_TXT, D])

    var iprb = _stream_pre_backward_direct_oft_resident_scratch[H, Dh](
        d_img_q, d_img_k, d_img_v, w.img, img_mod, oft.img, saved.img,
        N_IMG, D, eps, norm_ones, ctx, scratch, compute_aux_grads,
    )
    var tprb = _stream_pre_backward_direct_oft_resident_scratch[H, Dh](
        d_txt_q, d_txt_k, d_txt_v, w.txt, txt_mod, oft.txt, saved.txt,
        N_TXT, D, eps, norm_ones, ctx, scratch, compute_aux_grads,
    )

    var d_img_x_t = TArc(add(ipb.d_x[], iprb.d_x[], ctx))
    var d_txt_x_t = TArc(add(tpb.d_x[], tprb.d_x[], ctx))
    var img_grads = StreamDirectOFTGradsT(
        d_img_x_t^, iprb.q.copy(), iprb.k.copy(), iprb.v.copy(),
        ipb.out.copy(), ipb.ff_in.copy(), ipb.ff_out.copy(),
    )
    var txt_grads = StreamDirectOFTGradsT(
        d_txt_x_t^, tprb.q.copy(), tprb.k.copy(), tprb.v.copy(),
        tpb.out.copy(), tpb.ff_in.copy(), tpb.ff_out.copy(),
    )
    var out = DoubleBlockDirectOFTGradsT(img_grads^, txt_grads^)
    scratch.rewind(scratch_mark)
    return out^


def double_block_lora_forward_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_x: TArc, txt_x: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice, lora: DoubleBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    drop: DoubleBlockLoraDropout = DoubleBlockLoraDropout(),
) raises -> DoubleBlockDeviceForward:
    var lora_dev = double_block_lora_to_device(lora, ctx)
    return double_block_lora_forward_device_resident[H, Dh, N_IMG, N_TXT, S](
        img_x, txt_x, w, img_mod, txt_mod, lora_dev, cos, sin, D, F, eps, ctx, drop,
    )


def double_block_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img: List[Float32], txt: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs, lora: DoubleBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockForward:
    var img_mod_dev = modvecs_to_device(img_mod, D, ctx)
    var txt_mod_dev = modvecs_to_device(txt_mod, D, ctx)
    var fwd = double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S](
        _ta(img, [N_IMG, D], ctx), _ta(txt, [N_TXT, D], ctx),
        w, img_mod_dev, txt_mod_dev, lora, cos, sin, D, F, eps, ctx,
    )
    var img_out = fwd.img_out[].to_host(ctx)
    var txt_out = fwd.txt_out[].to_host(ctx)
    return DoubleBlockForward(img_out^, txt_out^, fwd.saved.copy())


# ── LoRA-aware per-stream POST backward ──────────────────────────────────────
# Mirrors _stream_post_backward; additionally runs SEPARATE LoRA backward for the
# out-proj (att, d_proj_out → out_d_a/_b; folds LoRA d_x into d_att), ff_in (mlp_in,
# d_gu → ff_in_d_a/_b), and ff_out (act, d_mlp → ff_out_d_a/_b), each folding its
# LoRA d_x into the matching base grad so the upstream chain stays correct.
struct _StreamPostBackLora(Copyable, Movable):
    var base: _StreamPostBack
    var d_x: TArc
    var d_att: TArc
    var out_d_a: List[Float32]
    var out_d_b: List[Float32]
    var ff_in_d_a: List[Float32]
    var ff_in_d_b: List[Float32]
    var ff_out_d_a: List[Float32]
    var ff_out_d_b: List[Float32]

    def __init__(
        out self, var base: _StreamPostBack,
        var d_x: TArc, var d_att: TArc,
        var out_d_a: List[Float32], var out_d_b: List[Float32],
        var ff_in_d_a: List[Float32], var ff_in_d_b: List[Float32],
        var ff_out_d_a: List[Float32], var ff_out_d_b: List[Float32],
    ):
        self.base = base^
        self.d_x = d_x^
        self.d_att = d_att^
        self.out_d_a = out_d_a^
        self.out_d_b = out_d_b^
        self.ff_in_d_a = ff_in_d_a^
        self.ff_in_d_b = ff_in_d_b^
        self.ff_out_d_a = ff_out_d_a^
        self.ff_out_d_b = ff_out_d_b^


struct _StreamPostBackLoraTensors(Copyable, Movable):
    var base: _StreamPostBack
    var d_x: TArc
    var d_att: TArc
    var out_d_a: Optional[TArc]
    var out_d_b: Optional[TArc]
    var ff_in_d_a: Optional[TArc]
    var ff_in_d_b: Optional[TArc]
    var ff_out_d_a: Optional[TArc]
    var ff_out_d_b: Optional[TArc]

    def __init__(
        out self, var base: _StreamPostBack,
        var d_x: TArc, var d_att: TArc,
        var out_d_a: Optional[TArc], var out_d_b: Optional[TArc],
        var ff_in_d_a: Optional[TArc], var ff_in_d_b: Optional[TArc],
        var ff_out_d_a: Optional[TArc], var ff_out_d_b: Optional[TArc],
    ):
        self.base = base^
        self.d_x = d_x^
        self.d_att = d_att^
        self.out_d_a = out_d_a^
        self.out_d_b = out_d_b^
        self.ff_in_d_a = ff_in_d_a^
        self.ff_in_d_b = ff_in_d_b^
        self.ff_out_d_a = ff_out_d_a^
        self.ff_out_d_b = ff_out_d_b^


def _stream_post_backward_lora_resident(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice, sv: StreamSaved,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
    compute_aux_grads: Bool = True,
    drop: StreamLoraDropout = StreamLoraDropout(),
) raises -> _StreamPostBackLora:
    var grg2: GateResidualGrads
    var d_gate2 = List[Float32]()
    if compute_aux_grads:
        var no_bias_mlp = Optional[Tensor](None)
        var mlp_y = linear(sv.act[], w.wd[], no_bias_mlp^, ctx)
        grg2 = gate_residual_backward(
            d_out[], sv.attn_res[], mv.gate2[], mlp_y, ctx
        )
        d_gate2 = grg2.d_g.to_host(ctx)
    else:
        grg2 = gate_residual_backward_dxdy(d_out[], mv.gate2[], ctx)

    # frozen wd backward: d_x ONLY (base d_wd computed-then-discarded by trainer).
    var d_d_dx = linear_backward_dx(grg2.d_y, w.wd[], N, F, D, ctx)

    # LoRA on ff.linear_out / ff_context.linear_out (input=act, d_y=grg2.d_y=d_mlp):
    # ff_out_d_a/_b + d_act_lo (folded into the grad into act). Dropout-aware: same
    # mask as forward regenerated bit-identically (LoRAModule.py:328); p==0 -> base.
    var ff_out_d_a = List[Float32]()
    var ff_out_d_b = List[Float32]()
    if lo.ff_out:
        var lg = _klein_lora_bwd_dropout(grg2.d_y, sv.act[], lo.ff_out.value(), N, drop.ff_out, ctx)
        d_d_dx = add(d_d_dx, lg.d_x, ctx)
        ff_out_d_a = lg.d_a.copy()
        ff_out_d_b = lg.d_b.copy()

    var sgb = swiglu_backward(d_d_dx, sv.gate[], sv.up[], ctx)
    var d_gu = concat(1, ctx, sgb.d_gate, sgb.d_up)

    # frozen wgu backward: d_x ONLY (base d_wgu computed-then-discarded).
    var d_gu_dx = linear_backward_dx(d_gu, w.wgu[], N, D, 2 * F, ctx)

    # LoRA on ff.linear_in / ff_context.linear_in (input=mlp_in, d_y=d_gu [N,2F]):
    # ff_in_d_a/_b + d_mlp_in_lo (folded into the grad into mlp_in).
    var ff_in_d_a = List[Float32]()
    var ff_in_d_b = List[Float32]()
    if lo.ff_in:
        var lg = _klein_lora_bwd_dropout(d_gu, sv.mlp_in[], lo.ff_in.value(), N, drop.ff_in, ctx)
        d_gu_dx = add(d_gu_dx, lg.d_x, ctx)
        ff_in_d_a = lg.d_a.copy()
        ff_in_d_b = lg.d_b.copy()

    var mb2 = modulate_backward(d_gu_dx, sv.ln2[], mv.scale2[], ctx, compute_aux_grads)
    var d_scale2 = List[Float32]()
    var d_shift2 = List[Float32]()
    if compute_aux_grads:
        d_scale2 = mb2.d_scale.to_host(ctx)
        d_shift2 = mb2.d_shift.to_host(ctx)

    var d_attn_res_norm = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    var d_attn_res_total = TArc(add(grg2.d_x, d_attn_res_norm, ctx))

    # proj_out = linear(att, Wproj) [+ LoRA]. Recompute it only when d_gate1 is
    # requested; d_x/d_y do not depend on the gated y value. Dropout-aware so the
    # recomputed proj_out matches the forward value (same mask, p==0 -> identity).
    var grg1: GateResidualGrads
    # Consume grg1 by to_host only (no field `^`-move out of the struct). d_y ->
    # host (LoRA bridge needs host); d_g -> host. d_x is just the incoming
    # residual grad, so the LoRA device path carries it by TArc and leaves the
    # legacy host slot empty.
    var d_gate1 = List[Float32]()
    if compute_aux_grads:
        var no_bias = Optional[Tensor](None)
        var proj_out = linear(att[], w.wproj[], no_bias^, ctx)
        if lo.out:
            var dlt = _klein_lora_fwd_dropout(att[], lo.out.value(), N, drop.out, ctx)
            proj_out = add(proj_out, dlt, ctx)
        grg1 = gate_residual_backward(
            d_attn_res_total[], x[], mv.gate1[], proj_out, ctx
        )
        d_gate1 = grg1.d_g.to_host(ctx)
    else:
        grg1 = gate_residual_backward_dxdy(d_attn_res_total[], mv.gate1[], ctx)
    var d_x_res_t = d_attn_res_total.copy()
    var d_x_res = List[Float32]()

    # frozen proj backward: d_x ONLY (base d_wproj computed-then-discarded).
    var d_att_t = linear_backward_dx(grg1.d_y, w.wproj[], N, D, D, ctx)

    # LoRA backward on attn out-proj (input=att, d_y=d_proj_out): out_d_a/_b + d_att_lo
    var out_d_a = List[Float32]()
    var out_d_b = List[Float32]()
    if lo.out:
        var lg = _klein_lora_bwd_dropout(grg1.d_y, att[], lo.out.value(), N, drop.out, ctx)
        d_att_t = add(d_att_t, lg.d_x, ctx)   # LoRA contribution to projection input
        out_d_a = lg.d_a.copy()
        out_d_b = lg.d_b.copy()

    # d_wproj/d_wgu/d_wd are frozen-base grads (stripped above; LoRA path discards
    # them, base double gate still validates that exact d_w math) — empty placeholders.
    var base = _StreamPostBack(
        d_x_res^, List[Float32](), List[Float32](), List[Float32](), List[Float32](),
        d_gate1=d_gate1^,
        d_shift2=d_shift2^, d_scale2=d_scale2^, d_gate2=d_gate2^,
    )
    return _StreamPostBackLora(
        base^, d_x_res_t^, TArc(d_att_t^),
        out_d_a^, out_d_b^, ff_in_d_a^, ff_in_d_b^, ff_out_d_a^, ff_out_d_b^,
    )


def _stream_post_backward_lora_resident_scratch(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice, sv: StreamSaved,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
    drop: StreamLoraDropout = StreamLoraDropout(),
) raises -> _StreamPostBackLora:
    var grg2: GateResidualGrads
    var d_gate2 = List[Float32]()
    if compute_aux_grads:
        var no_bias_mlp = Optional[Tensor](None)
        var mlp_y = linear(sv.act[], w.wd[], no_bias_mlp^, ctx)
        grg2 = gate_residual_backward(
            d_out[], sv.attn_res[], mv.gate2[], mlp_y, ctx
        )
        d_gate2 = grg2.d_g.to_host(ctx)
    else:
        grg2 = gate_residual_backward_dxdy(d_out[], mv.gate2[], ctx)

    var post_mark = scratch.mark()
    var d_d_dx = linear_backward_dx_scratch(grg2.d_y, w.wd[], N, F, D, ctx, scratch)

    var ff_out_d_a = List[Float32]()
    var ff_out_d_b = List[Float32]()
    if lo.ff_out:
        var lg = _klein_lora_bwd_dropout(grg2.d_y, sv.act[], lo.ff_out.value(), N, drop.ff_out, ctx)
        d_d_dx = add(d_d_dx, lg.d_x, ctx)
        ff_out_d_a = lg.d_a.copy()
        ff_out_d_b = lg.d_b.copy()

    var sgb = swiglu_backward(d_d_dx, sv.gate[], sv.up[], ctx)
    var d_gu = concat2_scratch(1, ctx, scratch, sgb.d_gate, sgb.d_up)

    var d_gu_dx = linear_backward_dx_scratch(
        d_gu, w.wgu[], N, D, 2 * F, ctx, scratch,
    )

    var ff_in_d_a = List[Float32]()
    var ff_in_d_b = List[Float32]()
    if lo.ff_in:
        var lg = _klein_lora_bwd_dropout(d_gu, sv.mlp_in[], lo.ff_in.value(), N, drop.ff_in, ctx)
        d_gu_dx = add(d_gu_dx, lg.d_x, ctx)
        ff_in_d_a = lg.d_a.copy()
        ff_in_d_b = lg.d_b.copy()

    var mb2 = modulate_backward(d_gu_dx, sv.ln2[], mv.scale2[], ctx, compute_aux_grads)
    var d_scale2 = List[Float32]()
    var d_shift2 = List[Float32]()
    if compute_aux_grads:
        d_scale2 = mb2.d_scale.to_host(ctx)
        d_shift2 = mb2.d_shift.to_host(ctx)
    scratch.rewind(post_mark)

    var d_attn_res_norm = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    var d_attn_res_total = TArc(add(grg2.d_x, d_attn_res_norm, ctx))

    var grg1: GateResidualGrads
    var d_gate1 = List[Float32]()
    if compute_aux_grads:
        var no_bias = Optional[Tensor](None)
        var proj_out = linear(att[], w.wproj[], no_bias^, ctx)
        if lo.out:
            var dlt = _klein_lora_fwd_dropout(att[], lo.out.value(), N, drop.out, ctx)
            proj_out = add(proj_out, dlt, ctx)
        grg1 = gate_residual_backward(
            d_attn_res_total[], x[], mv.gate1[], proj_out, ctx
        )
        d_gate1 = grg1.d_g.to_host(ctx)
    else:
        grg1 = gate_residual_backward_dxdy(d_attn_res_total[], mv.gate1[], ctx)
    var d_x_res_t = d_attn_res_total.copy()
    var d_x_res = List[Float32]()

    var d_att_t = linear_backward_dx(grg1.d_y, w.wproj[], N, D, D, ctx)

    var out_d_a = List[Float32]()
    var out_d_b = List[Float32]()
    if lo.out:
        var lg = _klein_lora_bwd_dropout(grg1.d_y, att[], lo.out.value(), N, drop.out, ctx)
        d_att_t = add(d_att_t, lg.d_x, ctx)
        out_d_a = lg.d_a.copy()
        out_d_b = lg.d_b.copy()

    var base = _StreamPostBack(
        d_x_res^, List[Float32](), List[Float32](), List[Float32](), List[Float32](),
        d_gate1=d_gate1^,
        d_shift2=d_shift2^, d_scale2=d_scale2^, d_gate2=d_gate2^,
    )
    return _StreamPostBackLora(
        base^, d_x_res_t^, TArc(d_att_t^),
        out_d_a^, out_d_b^, ff_in_d_a^, ff_in_d_b^, ff_out_d_a^, ff_out_d_b^,
    )


def _stream_post_backward_lora_resident_scratch_tensors(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice, sv: StreamSaved,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
    drop: StreamLoraDropout = StreamLoraDropout(),
) raises -> _StreamPostBackLoraTensors:
    var grg2: GateResidualGrads
    var d_gate2 = List[Float32]()
    if compute_aux_grads:
        var no_bias_mlp = Optional[Tensor](None)
        var mlp_y = linear(sv.act[], w.wd[], no_bias_mlp^, ctx)
        grg2 = gate_residual_backward(
            d_out[], sv.attn_res[], mv.gate2[], mlp_y, ctx
        )
        d_gate2 = grg2.d_g.to_host(ctx)
    else:
        grg2 = gate_residual_backward_dxdy(d_out[], mv.gate2[], ctx)

    var post_mark = scratch.mark()
    var d_d_dx = linear_backward_dx_scratch(grg2.d_y, w.wd[], N, F, D, ctx, scratch)

    var ff_out_d_a = Optional[TArc](None)
    var ff_out_d_b = Optional[TArc](None)
    if lo.ff_out:
        var lg = _klein_lora_bwd_dropout_tensors(grg2.d_y, sv.act[], lo.ff_out.value(), N, drop.ff_out, ctx)
        d_d_dx = add(d_d_dx, lg.d_x[], ctx)
        ff_out_d_a = Optional[TArc](lg.d_a.copy())
        ff_out_d_b = Optional[TArc](lg.d_b.copy())

    var sgb = swiglu_backward(d_d_dx, sv.gate[], sv.up[], ctx)
    var d_gu = concat2_scratch(1, ctx, scratch, sgb.d_gate, sgb.d_up)

    var d_gu_dx = linear_backward_dx_scratch(
        d_gu, w.wgu[], N, D, 2 * F, ctx, scratch,
    )

    var ff_in_d_a = Optional[TArc](None)
    var ff_in_d_b = Optional[TArc](None)
    if lo.ff_in:
        var lg = _klein_lora_bwd_dropout_tensors(d_gu, sv.mlp_in[], lo.ff_in.value(), N, drop.ff_in, ctx)
        d_gu_dx = add(d_gu_dx, lg.d_x[], ctx)
        ff_in_d_a = Optional[TArc](lg.d_a.copy())
        ff_in_d_b = Optional[TArc](lg.d_b.copy())

    var mb2 = modulate_backward(d_gu_dx, sv.ln2[], mv.scale2[], ctx, compute_aux_grads)
    var d_scale2 = List[Float32]()
    var d_shift2 = List[Float32]()
    if compute_aux_grads:
        d_scale2 = mb2.d_scale.to_host(ctx)
        d_shift2 = mb2.d_shift.to_host(ctx)
    scratch.rewind(post_mark)

    var d_attn_res_norm = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    var d_attn_res_total = TArc(add(grg2.d_x, d_attn_res_norm, ctx))

    var grg1: GateResidualGrads
    var d_gate1 = List[Float32]()
    if compute_aux_grads:
        var no_bias = Optional[Tensor](None)
        var proj_out = linear(att[], w.wproj[], no_bias^, ctx)
        if lo.out:
            var dlt = _klein_lora_fwd_dropout(att[], lo.out.value(), N, drop.out, ctx)
            proj_out = add(proj_out, dlt, ctx)
        grg1 = gate_residual_backward(
            d_attn_res_total[], x[], mv.gate1[], proj_out, ctx
        )
        d_gate1 = grg1.d_g.to_host(ctx)
    else:
        grg1 = gate_residual_backward_dxdy(d_attn_res_total[], mv.gate1[], ctx)
    var d_x_res_t = d_attn_res_total.copy()
    var d_x_res = List[Float32]()

    var d_att_t = linear_backward_dx(grg1.d_y, w.wproj[], N, D, D, ctx)

    var out_d_a = Optional[TArc](None)
    var out_d_b = Optional[TArc](None)
    if lo.out:
        var lg = _klein_lora_bwd_dropout_tensors(
            grg1.d_y, att[], lo.out.value(), N, drop.out, ctx
        )
        d_att_t = add(d_att_t, lg.d_x[], ctx)
        out_d_a = Optional[TArc](lg.d_a.copy())
        out_d_b = Optional[TArc](lg.d_b.copy())

    var base = _StreamPostBack(
        d_x_res^, List[Float32](), List[Float32](), List[Float32](), List[Float32](),
        d_gate1=d_gate1^,
        d_shift2=d_shift2^, d_scale2=d_scale2^, d_gate2=d_gate2^,
    )
    return _StreamPostBackLoraTensors(
        base^, d_x_res_t^, TArc(d_att_t^),
        out_d_a^, out_d_b^, ff_in_d_a^, ff_in_d_b^, ff_out_d_a^, ff_out_d_b^,
    )


# ── LoRA-aware per-stream PRE backward ───────────────────────────────────────
# Mirrors _stream_pre_backward; additionally runs SEPARATE q/k/v LoRA backward,
# each at its own column band of the wqkv-output grad (q→d_q_pre_flat,
# k→d_k_pre_flat, v→d_v_flat, input=norm) → q/k/v_d_a/_b, folding each LoRA d_x
# into d_norm (so d_x via layer_norm is correct).
struct _StreamPreBackLora(Copyable, Movable):
    var base: _StreamPreBack
    var d_x: TArc
    var q_d_a: List[Float32]
    var q_d_b: List[Float32]
    var k_d_a: List[Float32]
    var k_d_b: List[Float32]
    var v_d_a: List[Float32]
    var v_d_b: List[Float32]

    def __init__(
        out self, var base: _StreamPreBack,
        var d_x: TArc,
        var q_d_a: List[Float32], var q_d_b: List[Float32],
        var k_d_a: List[Float32], var k_d_b: List[Float32],
        var v_d_a: List[Float32], var v_d_b: List[Float32],
    ):
        self.base = base^
        self.d_x = d_x^
        self.q_d_a = q_d_a^
        self.q_d_b = q_d_b^
        self.k_d_a = k_d_a^
        self.k_d_b = k_d_b^
        self.v_d_a = v_d_a^
        self.v_d_b = v_d_b^


struct _StreamPreBackLoraTensors(Copyable, Movable):
    var base: _StreamPreBack
    var d_x: TArc
    var q_d_a: Optional[TArc]
    var q_d_b: Optional[TArc]
    var k_d_a: Optional[TArc]
    var k_d_b: Optional[TArc]
    var v_d_a: Optional[TArc]
    var v_d_b: Optional[TArc]

    def __init__(
        out self, var base: _StreamPreBack,
        var d_x: TArc,
        var q_d_a: Optional[TArc], var q_d_b: Optional[TArc],
        var k_d_a: Optional[TArc], var k_d_b: Optional[TArc],
        var v_d_a: Optional[TArc], var v_d_b: Optional[TArc],
    ):
        self.base = base^
        self.d_x = d_x^
        self.q_d_a = q_d_a^
        self.q_d_b = q_d_b^
        self.k_d_a = k_d_a^
        self.k_d_b = k_d_b^
        self.v_d_a = v_d_a^
        self.v_d_b = v_d_b^


def _stream_pre_backward_lora_resident[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v_flat: Tensor,
    w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice, sv: StreamSaved,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
    compute_aux_grads: Bool = True,
    drop: StreamLoraDropout = StreamLoraDropout(),
) raises -> _StreamPreBackLora:
    var d_q_pre = rms_norm_backward_dx(d_q_rms, sv.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = List[Float32]()
    var d_k_pre = rms_norm_backward_dx(d_k_rms, sv.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = List[Float32]()

    var d_q_pre_flat = reshape_owned(d_q_pre^, [N, D])
    var d_k_pre_flat = reshape_owned(d_k_pre^, [N, D])
    var d_qkv = concat(1, ctx, d_q_pre_flat, d_k_pre_flat, d_v_flat)   # grad at wqkv OUTPUT [N,3D]

    # frozen qkv backward: d_x ONLY (base d_wqkv computed-then-discarded).
    var d_norm_t = linear_backward_dx(
        d_qkv, w.wqkv[], N, D, 3 * D, ctx,
    )

    # SEPARATE q/k/v LoRA backward (OneTrainer to_q/to_k/to_v, add_q/k/v_proj).
    # Each adapter's d_y is its OWN column band of d_qkv (q→[N,D] d_q_pre_flat,
    # k→d_k_pre_flat, v→d_v_flat), input=sv.norm; each LoRA d_x folds into d_norm.
    # Dropout-aware (LoRAModule.py:328); p==0 -> base klein_lora_bwd_device_resident.
    var q_d_a = List[Float32]()
    var q_d_b = List[Float32]()
    var k_d_a = List[Float32]()
    var k_d_b = List[Float32]()
    var v_d_a = List[Float32]()
    var v_d_b = List[Float32]()
    if lo.q:
        var lg = _klein_lora_bwd_dropout(d_q_pre_flat, sv.norm[], lo.q.value(), N, drop.q, ctx)
        d_norm_t = add(d_norm_t, lg.d_x, ctx)
        q_d_a = lg.d_a.copy()
        q_d_b = lg.d_b.copy()
    if lo.k:
        var lg = _klein_lora_bwd_dropout(d_k_pre_flat, sv.norm[], lo.k.value(), N, drop.k, ctx)
        d_norm_t = add(d_norm_t, lg.d_x, ctx)
        k_d_a = lg.d_a.copy()
        k_d_b = lg.d_b.copy()
    if lo.v:
        var lg = _klein_lora_bwd_dropout(d_v_flat, sv.norm[], lo.v.value(), N, drop.v, ctx)
        d_norm_t = add(d_norm_t, lg.d_x, ctx)
        v_d_a = lg.d_a.copy()
        v_d_b = lg.d_b.copy()

    var mb1 = modulate_backward(d_norm_t, sv.ln1[], mv.scale1[], ctx, compute_aux_grads)
    var d_scale1 = List[Float32]()
    var d_shift1 = List[Float32]()
    if compute_aux_grads:
        d_scale1 = mb1.d_scale.to_host(ctx)
        d_shift1 = mb1.d_shift.to_host(ctx)

    var d_x_norm_t = layer_norm_backward_dx(mb1.d_x, sv.x[], ones, eps, ctx)
    var d_x_norm_arc = TArc(d_x_norm_t^)
    var d_x_norm = List[Float32]()
    # d_wqkv is a frozen-base grad (stripped above) — empty placeholder.
    var base = _StreamPreBack(
        d_x_norm^, List[Float32](), d_q_norm^, d_k_norm^, d_shift1^, d_scale1^
    )
    return _StreamPreBackLora(base^, d_x_norm_arc^, q_d_a^, q_d_b^, k_d_a^, k_d_b^, v_d_a^, v_d_b^)


def _stream_pre_backward_lora_resident_scratch[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v_flat: Tensor,
    w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice, sv: StreamSaved,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
    drop: StreamLoraDropout = StreamLoraDropout(),
) raises -> _StreamPreBackLora:
    var scratch_mark = scratch.mark()
    var d_q_pre = rms_norm_backward_dx(d_q_rms, sv.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = List[Float32]()
    var d_k_pre = rms_norm_backward_dx(d_k_rms, sv.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = List[Float32]()

    var d_q_pre_flat = reshape_owned(d_q_pre^, [N, D])
    var d_k_pre_flat = reshape_owned(d_k_pre^, [N, D])
    var d_qkv = concat3_scratch(1, ctx, scratch, d_q_pre_flat, d_k_pre_flat, d_v_flat, True)

    var d_norm_t = linear_backward_dx_scratch(
        d_qkv, w.wqkv[], N, D, 3 * D, ctx, scratch,
    )

    # SEPARATE q/k/v LoRA backward (see resident variant for the OneTrainer mapping).
    # Dropout-aware (LoRAModule.py:328); p==0 -> base klein_lora_bwd_device_resident.
    var q_d_a = List[Float32]()
    var q_d_b = List[Float32]()
    var k_d_a = List[Float32]()
    var k_d_b = List[Float32]()
    var v_d_a = List[Float32]()
    var v_d_b = List[Float32]()
    if lo.q:
        var lg = _klein_lora_bwd_dropout(d_q_pre_flat, sv.norm[], lo.q.value(), N, drop.q, ctx)
        d_norm_t = add(d_norm_t, lg.d_x, ctx)
        q_d_a = lg.d_a.copy()
        q_d_b = lg.d_b.copy()
    if lo.k:
        var lg = _klein_lora_bwd_dropout(d_k_pre_flat, sv.norm[], lo.k.value(), N, drop.k, ctx)
        d_norm_t = add(d_norm_t, lg.d_x, ctx)
        k_d_a = lg.d_a.copy()
        k_d_b = lg.d_b.copy()
    if lo.v:
        var lg = _klein_lora_bwd_dropout(d_v_flat, sv.norm[], lo.v.value(), N, drop.v, ctx)
        d_norm_t = add(d_norm_t, lg.d_x, ctx)
        v_d_a = lg.d_a.copy()
        v_d_b = lg.d_b.copy()

    var mb1 = modulate_backward(d_norm_t, sv.ln1[], mv.scale1[], ctx, compute_aux_grads)
    var d_scale1 = List[Float32]()
    var d_shift1 = List[Float32]()
    if compute_aux_grads:
        d_scale1 = mb1.d_scale.to_host(ctx)
        d_shift1 = mb1.d_shift.to_host(ctx)

    var d_x_norm_t = layer_norm_backward_dx(mb1.d_x, sv.x[], ones, eps, ctx)
    var d_x_norm_arc = TArc(d_x_norm_t^)
    var d_x_norm = List[Float32]()
    var base = _StreamPreBack(
        d_x_norm^, List[Float32](), d_q_norm^, d_k_norm^, d_shift1^, d_scale1^
    )
    var out = _StreamPreBackLora(
        base^, d_x_norm_arc^, q_d_a^, q_d_b^, k_d_a^, k_d_b^, v_d_a^, v_d_b^,
    )
    scratch.rewind(scratch_mark)
    return out^


def _stream_pre_backward_lora_resident_scratch_tensors[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v_flat: Tensor,
    w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice, sv: StreamSaved,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
    drop: StreamLoraDropout = StreamLoraDropout(),
) raises -> _StreamPreBackLoraTensors:
    var scratch_mark = scratch.mark()
    var d_q_pre = rms_norm_backward_dx(d_q_rms, sv.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = List[Float32]()
    var d_k_pre = rms_norm_backward_dx(d_k_rms, sv.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = List[Float32]()

    var d_q_pre_flat = reshape_owned(d_q_pre^, [N, D])
    var d_k_pre_flat = reshape_owned(d_k_pre^, [N, D])
    var d_qkv = concat3_scratch(1, ctx, scratch, d_q_pre_flat, d_k_pre_flat, d_v_flat, True)

    var d_norm_t = linear_backward_dx_scratch(
        d_qkv, w.wqkv[], N, D, 3 * D, ctx, scratch,
    )

    # SEPARATE q/k/v LoRA backward (device-tensor grads kept on device).
    # Dropout-aware (LoRAModule.py:328); p==0 -> base klein_lora_bwd_device_resident_tensors.
    var q_d_a = Optional[TArc](None)
    var q_d_b = Optional[TArc](None)
    var k_d_a = Optional[TArc](None)
    var k_d_b = Optional[TArc](None)
    var v_d_a = Optional[TArc](None)
    var v_d_b = Optional[TArc](None)
    if lo.q:
        var lg = _klein_lora_bwd_dropout_tensors(d_q_pre_flat, sv.norm[], lo.q.value(), N, drop.q, ctx)
        d_norm_t = add(d_norm_t, lg.d_x[], ctx)
        q_d_a = Optional[TArc](lg.d_a.copy())
        q_d_b = Optional[TArc](lg.d_b.copy())
    if lo.k:
        var lg = _klein_lora_bwd_dropout_tensors(d_k_pre_flat, sv.norm[], lo.k.value(), N, drop.k, ctx)
        d_norm_t = add(d_norm_t, lg.d_x[], ctx)
        k_d_a = Optional[TArc](lg.d_a.copy())
        k_d_b = Optional[TArc](lg.d_b.copy())
    if lo.v:
        var lg = _klein_lora_bwd_dropout_tensors(d_v_flat, sv.norm[], lo.v.value(), N, drop.v, ctx)
        d_norm_t = add(d_norm_t, lg.d_x[], ctx)
        v_d_a = Optional[TArc](lg.d_a.copy())
        v_d_b = Optional[TArc](lg.d_b.copy())

    var mb1 = modulate_backward(d_norm_t, sv.ln1[], mv.scale1[], ctx, compute_aux_grads)
    var d_scale1 = List[Float32]()
    var d_shift1 = List[Float32]()
    if compute_aux_grads:
        d_scale1 = mb1.d_scale.to_host(ctx)
        d_shift1 = mb1.d_shift.to_host(ctx)

    var d_x_norm_t = layer_norm_backward_dx(mb1.d_x, sv.x[], ones, eps, ctx)
    var d_x_norm_arc = TArc(d_x_norm_t^)
    var d_x_norm = List[Float32]()
    var base = _StreamPreBack(
        d_x_norm^, List[Float32](), d_q_norm^, d_k_norm^, d_shift1^, d_scale1^
    )
    var out = _StreamPreBackLoraTensors(
        base^, d_x_norm_arc^, q_d_a^, q_d_b^, k_d_a^, k_d_b^, v_d_a^, v_d_b^,
    )
    scratch.rewind(scratch_mark)
    return out^


# ── BACKWARD of one DOUBLE block WITH LoRA on the attention projections ──────
def double_block_lora_backward_device_resident[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice, lora: DoubleBlockLoraDevice,
    saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    compute_aux_grads: Bool = True,
    drop: DoubleBlockLoraDropout = DoubleBlockLoraDropout(),
) raises -> DoubleBlockLoraDeviceGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_dtype(_ones(D), [D], saved.img.x[].dtype(), ctx)

    var ipb = _stream_post_backward_lora_resident(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, lora.img, saved.img,
        N_IMG, D, F, eps, ones_t, ctx, compute_aux_grads, drop.img,
    )
    var tpb = _stream_post_backward_lora_resident(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, lora.txt, saved.txt,
        N_TXT, D, F, eps, ones_t, ctx, compute_aux_grads, drop.txt,
    )

    # d_att per stream stays device-resident; reshape [N,D] -> [1,N,H,Dh] is
    # byte-identical and avoids the old readback/re-upload bridge.
    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh]

    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx,
    )

    var d_q_joint = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_joint = rope_backward(sb.d_k, cos, sin, True, ctx)

    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(sb.d_v, N_TXT, N_IMG, 1, ctx)
    reshape_in_place(cv.d_1, [N_IMG, D])
    reshape_in_place(cv.d_0, [N_TXT, D])

    var iprb = _stream_pre_backward_lora_resident[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, lora.img, saved.img,
        N_IMG, D, eps, ones_t, ctx, compute_aux_grads, drop.img,
    )
    var tprb = _stream_pre_backward_lora_resident[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, lora.txt, saved.txt,
        N_TXT, D, eps, ones_t, ctx, compute_aux_grads, drop.txt,
    )

    var d_img_x_t = TArc(add(ipb.d_x[], iprb.d_x[], ctx))
    var d_txt_x_t = TArc(add(tpb.d_x[], tprb.d_x[], ctx))

    var img_grads = StreamLoraDeviceGrads(
        d_img_x_t^,
        iprb.base.d_shift1.copy(), iprb.base.d_scale1.copy(), ipb.base.d_gate1.copy(),
        ipb.base.d_shift2.copy(), ipb.base.d_scale2.copy(), ipb.base.d_gate2.copy(),
        iprb.q_d_a.copy(), iprb.q_d_b.copy(), iprb.k_d_a.copy(), iprb.k_d_b.copy(),
        iprb.v_d_a.copy(), iprb.v_d_b.copy(), ipb.out_d_a.copy(), ipb.out_d_b.copy(),
        ipb.ff_in_d_a.copy(), ipb.ff_in_d_b.copy(), ipb.ff_out_d_a.copy(), ipb.ff_out_d_b.copy(),
    )
    var txt_grads = StreamLoraDeviceGrads(
        d_txt_x_t^,
        tprb.base.d_shift1.copy(), tprb.base.d_scale1.copy(), tpb.base.d_gate1.copy(),
        tpb.base.d_shift2.copy(), tpb.base.d_scale2.copy(), tpb.base.d_gate2.copy(),
        tprb.q_d_a.copy(), tprb.q_d_b.copy(), tprb.k_d_a.copy(), tprb.k_d_b.copy(),
        tprb.v_d_a.copy(), tprb.v_d_b.copy(), tpb.out_d_a.copy(), tpb.out_d_b.copy(),
        tpb.ff_in_d_a.copy(), tpb.ff_in_d_b.copy(), tpb.ff_out_d_a.copy(), tpb.ff_out_d_b.copy(),
    )
    return DoubleBlockLoraDeviceGrads(img_grads^, txt_grads^)


def double_block_lora_backward_device_resident_scratch[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice, lora: DoubleBlockLoraDevice,
    saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
    drop: DoubleBlockLoraDropout = DoubleBlockLoraDropout(),
) raises -> DoubleBlockLoraDeviceGrads:
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ipb = _stream_post_backward_lora_resident_scratch(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, lora.img, saved.img,
        N_IMG, D, F, eps, norm_ones, ctx, scratch, compute_aux_grads, drop.img,
    )
    var tpb = _stream_post_backward_lora_resident_scratch(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, lora.txt, saved.txt,
        N_TXT, D, F, eps, norm_ones, ctx, scratch, compute_aux_grads, drop.txt,
    )

    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat2_scratch(1, ctx, scratch, d_tatt_4d, d_iatt_4d)

    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    comptime if KLEIN_SDPA_FLASH:
        if not saved.flash_stats:
            raise Error(
                "double_block bwd: KLEIN_SDPA_FLASH on but saved tape has"
                " no flash stats (fwd/bwd flag mismatch)"
            )
        var fb = sdpa_flash_backward_f32[1, S, H, Dh](
            saved.flash_q.value(), saved.flash_k.value(),
            saved.flash_v.value(), saved.flash_o.value(),
            saved.flash_stats.value(), d_att_joint, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward_scratch[1, S, H, Dh](
            saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx, scratch,
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_q_joint = rope_backward(d_q_sb, cos, sin, True, ctx)
    var d_k_joint = rope_backward(d_k_sb, cos, sin, True, ctx)

    var d_txt_q = slice_scratch(d_q_joint, 1, 0, N_TXT, ctx, scratch)
    var d_img_q = slice_scratch(d_q_joint, 1, N_TXT, N_IMG, ctx, scratch)
    var d_txt_k = slice_scratch(d_k_joint, 1, 0, N_TXT, ctx, scratch)
    var d_img_k = slice_scratch(d_k_joint, 1, N_TXT, N_IMG, ctx, scratch)
    var d_txt_v = slice_scratch(d_v_sb, 1, 0, N_TXT, ctx, scratch)
    var d_img_v = slice_scratch(d_v_sb, 1, N_TXT, N_IMG, ctx, scratch)
    reshape_in_place(d_img_v, [N_IMG, D])
    reshape_in_place(d_txt_v, [N_TXT, D])

    var iprb = _stream_pre_backward_lora_resident_scratch[H, Dh](
        d_img_q, d_img_k, d_img_v, w.img, img_mod, lora.img, saved.img,
        N_IMG, D, eps, norm_ones, ctx, scratch, compute_aux_grads, drop.img,
    )
    var tprb = _stream_pre_backward_lora_resident_scratch[H, Dh](
        d_txt_q, d_txt_k, d_txt_v, w.txt, txt_mod, lora.txt, saved.txt,
        N_TXT, D, eps, norm_ones, ctx, scratch, compute_aux_grads, drop.txt,
    )

    var d_img_x_t = TArc(add(ipb.d_x[], iprb.d_x[], ctx))
    var d_txt_x_t = TArc(add(tpb.d_x[], tprb.d_x[], ctx))

    var img_grads = StreamLoraDeviceGrads(
        d_img_x_t^,
        iprb.base.d_shift1.copy(), iprb.base.d_scale1.copy(), ipb.base.d_gate1.copy(),
        ipb.base.d_shift2.copy(), ipb.base.d_scale2.copy(), ipb.base.d_gate2.copy(),
        iprb.q_d_a.copy(), iprb.q_d_b.copy(), iprb.k_d_a.copy(), iprb.k_d_b.copy(),
        iprb.v_d_a.copy(), iprb.v_d_b.copy(), ipb.out_d_a.copy(), ipb.out_d_b.copy(),
        ipb.ff_in_d_a.copy(), ipb.ff_in_d_b.copy(), ipb.ff_out_d_a.copy(), ipb.ff_out_d_b.copy(),
    )
    var txt_grads = StreamLoraDeviceGrads(
        d_txt_x_t^,
        tprb.base.d_shift1.copy(), tprb.base.d_scale1.copy(), tpb.base.d_gate1.copy(),
        tpb.base.d_shift2.copy(), tpb.base.d_scale2.copy(), tpb.base.d_gate2.copy(),
        tprb.q_d_a.copy(), tprb.q_d_b.copy(), tprb.k_d_a.copy(), tprb.k_d_b.copy(),
        tprb.v_d_a.copy(), tprb.v_d_b.copy(), tpb.out_d_a.copy(), tpb.out_d_b.copy(),
        tpb.ff_in_d_a.copy(), tpb.ff_in_d_b.copy(), tpb.ff_out_d_a.copy(), tpb.ff_out_d_b.copy(),
    )
    var out = DoubleBlockLoraDeviceGrads(img_grads^, txt_grads^)
    scratch.rewind(scratch_mark)
    return out^


def double_block_lora_backward_device_resident_scratch_tensors[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice, lora: DoubleBlockLoraDevice,
    saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    norm_ones: Tensor,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    compute_aux_grads: Bool = True,
    drop: DoubleBlockLoraDropout = DoubleBlockLoraDropout(),
) raises -> DoubleBlockLoraDeviceGradTensors:
    var scratch_mark = scratch.mark()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ipb = _stream_post_backward_lora_resident_scratch_tensors(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, lora.img, saved.img,
        N_IMG, D, F, eps, norm_ones, ctx, scratch, compute_aux_grads, drop.img,
    )
    var tpb = _stream_post_backward_lora_resident_scratch_tensors(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, lora.txt, saved.txt,
        N_TXT, D, F, eps, norm_ones, ctx, scratch, compute_aux_grads, drop.txt,
    )

    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat2_scratch(1, ctx, scratch, d_tatt_4d, d_iatt_4d)

    var sb = sdpa_backward_scratch[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx, scratch,
    )

    var d_q_joint = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_joint = rope_backward(sb.d_k, cos, sin, True, ctx)

    var d_txt_q = slice_scratch(d_q_joint, 1, 0, N_TXT, ctx, scratch)
    var d_img_q = slice_scratch(d_q_joint, 1, N_TXT, N_IMG, ctx, scratch)
    var d_txt_k = slice_scratch(d_k_joint, 1, 0, N_TXT, ctx, scratch)
    var d_img_k = slice_scratch(d_k_joint, 1, N_TXT, N_IMG, ctx, scratch)
    var d_txt_v = slice_scratch(sb.d_v, 1, 0, N_TXT, ctx, scratch)
    var d_img_v = slice_scratch(sb.d_v, 1, N_TXT, N_IMG, ctx, scratch)
    reshape_in_place(d_img_v, [N_IMG, D])
    reshape_in_place(d_txt_v, [N_TXT, D])

    var iprb = _stream_pre_backward_lora_resident_scratch_tensors[H, Dh](
        d_img_q, d_img_k, d_img_v, w.img, img_mod, lora.img, saved.img,
        N_IMG, D, eps, norm_ones, ctx, scratch, compute_aux_grads, drop.img,
    )
    var tprb = _stream_pre_backward_lora_resident_scratch_tensors[H, Dh](
        d_txt_q, d_txt_k, d_txt_v, w.txt, txt_mod, lora.txt, saved.txt,
        N_TXT, D, eps, norm_ones, ctx, scratch, compute_aux_grads, drop.txt,
    )

    var d_img_x_t = TArc(add(ipb.d_x[], iprb.d_x[], ctx))
    var d_txt_x_t = TArc(add(tpb.d_x[], tprb.d_x[], ctx))

    var img_grads = StreamLoraDeviceGradTensors(
        d_img_x_t^,
        iprb.base.d_shift1.copy(), iprb.base.d_scale1.copy(), ipb.base.d_gate1.copy(),
        ipb.base.d_shift2.copy(), ipb.base.d_scale2.copy(), ipb.base.d_gate2.copy(),
        iprb.q_d_a.copy(), iprb.q_d_b.copy(), iprb.k_d_a.copy(), iprb.k_d_b.copy(),
        iprb.v_d_a.copy(), iprb.v_d_b.copy(), ipb.out_d_a.copy(), ipb.out_d_b.copy(),
        ipb.ff_in_d_a.copy(), ipb.ff_in_d_b.copy(), ipb.ff_out_d_a.copy(), ipb.ff_out_d_b.copy(),
    )
    var txt_grads = StreamLoraDeviceGradTensors(
        d_txt_x_t^,
        tprb.base.d_shift1.copy(), tprb.base.d_scale1.copy(), tpb.base.d_gate1.copy(),
        tpb.base.d_shift2.copy(), tpb.base.d_scale2.copy(), tpb.base.d_gate2.copy(),
        tprb.q_d_a.copy(), tprb.q_d_b.copy(), tprb.k_d_a.copy(), tprb.k_d_b.copy(),
        tprb.v_d_a.copy(), tprb.v_d_b.copy(), tpb.out_d_a.copy(), tpb.out_d_b.copy(),
        tpb.ff_in_d_a.copy(), tpb.ff_in_d_b.copy(), tpb.ff_out_d_a.copy(), tpb.ff_out_d_b.copy(),
    )
    var out = DoubleBlockLoraDeviceGradTensors(img_grads^, txt_grads^)
    scratch.rewind(scratch_mark)
    return out^


def double_block_lora_backward_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice, lora: DoubleBlockLora,
    saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    compute_aux_grads: Bool = True,
    drop: DoubleBlockLoraDropout = DoubleBlockLoraDropout(),
) raises -> DoubleBlockLoraDeviceGrads:
    var lora_dev = double_block_lora_to_device(lora, ctx)
    return double_block_lora_backward_device_resident[H, Dh, N_IMG, N_TXT, S](
        d_io_t, d_to_t, w, img_mod, txt_mod, lora_dev, saved,
        cos, sin, D, F, eps, ctx, compute_aux_grads, drop,
    )


def double_block_lora_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_img_out: List[Float32], d_txt_out: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs, lora: DoubleBlockLora,
    saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockLoraGrads:
    var img_mod_dev = modvecs_to_device(img_mod, D, ctx)
    var txt_mod_dev = modvecs_to_device(txt_mod, D, ctx)
    var dg = double_block_lora_backward_device[H, Dh, N_IMG, N_TXT, S](
        _ta(d_img_out, [N_IMG, D], ctx), _ta(d_txt_out, [N_TXT, D], ctx),
        w, img_mod_dev, txt_mod_dev, lora, saved, cos, sin, D, F, eps, ctx,
    )
    var d_img_x = dg.img.d_x[].to_host(ctx)
    var d_txt_x = dg.txt.d_x[].to_host(ctx)
    var img_grads = StreamGrads(
        d_img_x^, List[Float32](), List[Float32](), List[Float32](), List[Float32](),
        List[Float32](), List[Float32](),
        dg.img.d_shift1.copy(), dg.img.d_scale1.copy(), dg.img.d_gate1.copy(),
        dg.img.d_shift2.copy(), dg.img.d_scale2.copy(), dg.img.d_gate2.copy(),
    )
    var txt_grads = StreamGrads(
        d_txt_x^, List[Float32](), List[Float32](), List[Float32](), List[Float32](),
        List[Float32](), List[Float32](),
        dg.txt.d_shift1.copy(), dg.txt.d_scale1.copy(), dg.txt.d_gate1.copy(),
        dg.txt.d_shift2.copy(), dg.txt.d_scale2.copy(), dg.txt.d_gate2.copy(),
    )
    var base_grads = DoubleBlockGrads(img_grads^, txt_grads^)
    var img_lora = StreamLoraGrads(
        dg.img.q_d_a.copy(), dg.img.q_d_b.copy(),
        dg.img.k_d_a.copy(), dg.img.k_d_b.copy(),
        dg.img.v_d_a.copy(), dg.img.v_d_b.copy(),
        dg.img.out_d_a.copy(), dg.img.out_d_b.copy(),
        dg.img.ff_in_d_a.copy(), dg.img.ff_in_d_b.copy(),
        dg.img.ff_out_d_a.copy(), dg.img.ff_out_d_b.copy(),
    )
    var txt_lora = StreamLoraGrads(
        dg.txt.q_d_a.copy(), dg.txt.q_d_b.copy(),
        dg.txt.k_d_a.copy(), dg.txt.k_d_b.copy(),
        dg.txt.v_d_a.copy(), dg.txt.v_d_b.copy(),
        dg.txt.out_d_a.copy(), dg.txt.out_d_b.copy(),
        dg.txt.ff_in_d_a.copy(), dg.txt.ff_in_d_b.copy(),
        dg.txt.ff_out_d_a.copy(), dg.txt.ff_out_d_b.copy(),
    )
    return DoubleBlockLoraGrads(base_grads^, img_lora^, txt_lora^)
