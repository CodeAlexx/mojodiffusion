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
#   The LoRA-delta helpers (lora_block.mojo klein_lora_fwd/bwd) are host-list-
#   typed and OUT OF SCOPE for this increment, so the LoRA branches bridge
#   device<->host AT THOSE CALLS ONLY (small, present only when an adapter is
#   set). The dominant base chain is fully device-resident.
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

# ── forward ops (GPU) ────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import reshape, slice, concat, add

# ── backward arms (GPU; all pre-built + gated) ───────────────────────────────
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx, LinearGrads
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, RmsNormBackward,
    layer_norm_backward, LayerNormBackward,
)
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward, SwigluGrads
from serenitymojo.ops.attention_backward import sdpa_backward, SdpaGrads
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, GateResidualGrads, rope_backward,
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

    def __init__(
        out self, var img: StreamSaved, var txt: StreamSaved,
        var q_rope: TArc, var k_rope: TArc, var v_joint: TArc,
    ):
        self.img = img^
        self.txt = txt^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.v_joint = v_joint^


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
    var q_pre = reshape(q_pre_flat, [1, N, H, Dh], ctx)
    var k_pre = reshape(k_pre_flat, [1, N, H, Dh], ctx)
    var v = reshape(v_flat, [1, N, H, Dh], ctx)
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
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)

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
    var txt_att = TArc(reshape(txt_att_4d, [N_TXT, D], ctx))
    var img_att = TArc(reshape(img_att_4d, [N_IMG, D], ctx))

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
    var d_q_pre_flat = reshape(rb_q.d_x, [N, D], ctx)
    var d_k_pre_flat = reshape(rb_k.d_x, [N, D], ctx)
    var d_v_flat = reshape(d_v, [N, D], ctx)
    var d_qkv = concat(1, ctx, d_q_pre_flat, d_k_pre_flat, d_v_flat)   # [N,3D]

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
# Targets (per stream, attention-only — matches lora.mojo _map_klein_trainer):
#   wqkv (qkv_proj, FULL [3D,D])  and  wproj (out_proj, FULL [D,D]).
# Forward adds the LoRA delta at those two linears; backward returns d_A/d_B for
# each and folds the LoRA d_x contribution back into the projection-input grad.
#
# NOTE: the LoRA-delta helpers (klein_lora_fwd/bwd) are host-list-typed
# (lora_block.mojo, out of scope here), so the LoRA branches bridge device<->host
# AT THOSE CALLS ONLY. The base chain stays device-resident.
# ═══════════════════════════════════════════════════════════════════════════

from serenitymojo.models.klein.lora_block import (
    LoraAdapter, klein_lora_fwd, klein_lora_bwd, KleinLoraGrads,
)


# Optional LoRA adapters for one stream's two attention projections.
struct StreamLora(Copyable, Movable):
    var qkv: Optional[LoraAdapter]    # on wqkv  (in=D, out=3D)
    var proj: Optional[LoraAdapter]   # on wproj (in=D, out=D)

    def __init__(
        out self, var qkv: Optional[LoraAdapter], var proj: Optional[LoraAdapter]
    ):
        self.qkv = qkv^
        self.proj = proj^


struct DoubleBlockLora(Copyable, Movable):
    var img: StreamLora
    var txt: StreamLora

    def __init__(out self, var img: StreamLora, var txt: StreamLora):
        self.img = img^
        self.txt = txt^


# Per-stream LoRA grads: d_A/d_B for the qkv and proj adapters (empty when the
# corresponding adapter is absent).
struct StreamLoraGrads(Copyable, Movable):
    var qkv_d_a: List[Float32]
    var qkv_d_b: List[Float32]
    var proj_d_a: List[Float32]
    var proj_d_b: List[Float32]

    def __init__(
        out self,
        var qkv_d_a: List[Float32], var qkv_d_b: List[Float32],
        var proj_d_a: List[Float32], var proj_d_b: List[Float32],
    ):
        self.qkv_d_a = qkv_d_a^
        self.qkv_d_b = qkv_d_b^
        self.proj_d_a = proj_d_a^
        self.proj_d_b = proj_d_b^


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


# ── LoRA-aware per-stream pre (wqkv delta applied to the qkv output) ─────────
def _stream_pre_lora[
    H: Int, Dh: Int
](
    x: TArc, w: StreamWeights, mv: ModVecs, lo: StreamLora,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPre:
    var ln1 = layer_norm(x[], ones, zeros, eps, ctx)
    var norm = modulate(ln1, _t(mv.scale1.copy(), [D], ctx), _t(mv.shift1.copy(), [D], ctx), ctx)
    var no_bias = Optional[Tensor](None)
    var qkv = linear(norm, w.wqkv[], no_bias^, ctx)   # [N,3D]
    # LoRA on wqkv: delta [N,3D] added to qkv. Host-bridge at the helper boundary.
    if lo.qkv:
        var norm_h = norm.to_host(ctx)
        var dlt = klein_lora_fwd(norm_h, lo.qkv.value(), N, ctx)   # [N,3D]
        qkv = _t(_add_lists(qkv.to_host(ctx), dlt), [N, 3 * D], ctx)
    var q_pre_flat = slice(qkv, 1, 0, D, ctx)
    var k_pre_flat = slice(qkv, 1, D, D, ctx)
    var v_flat = slice(qkv, 1, 2 * D, D, ctx)
    var q_pre = reshape(q_pre_flat, [1, N, H, Dh], ctx)
    var k_pre = reshape(k_pre_flat, [1, N, H, Dh], ctx)
    var v = reshape(v_flat, [1, N, H, Dh], ctx)
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPre(
        TArc(ln1^), TArc(norm^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
    )


# ── LoRA-aware per-stream post (wproj delta applied to the out projection) ───
def _stream_post_lora(
    x: TArc, att: TArc, w: StreamWeights, mv: ModVecs,
    lo: StreamLora, N: Int, D: Int, F: Int, eps: Float32,
    ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPost:
    var no_bias = Optional[Tensor](None)
    var out = linear(att[], w.wproj[], no_bias^, ctx)   # [N,D]
    # LoRA on wproj: input is att [N,D]; delta [N,D] added to out.
    if lo.proj:
        var att_h = att[].to_host(ctx)
        var dlt = klein_lora_fwd(att_h, lo.proj.value(), N, ctx)
        out = _t(_add_lists(out.to_host(ctx), dlt), [N, D], ctx)
    var attn_res = residual_gate(x[], _t(mv.gate1.copy(), [D], ctx), out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, _t(mv.scale2.copy(), [D], ctx), _t(mv.shift2.copy(), [D], ctx), ctx)
    var no_bias2 = Optional[Tensor](None)
    var gu = linear(mlp_in, w.wgu[], no_bias2^, ctx)   # [N,2F]
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


# ── FORWARD of one DOUBLE block WITH LoRA on the attention projections ───────
# Identical graph to double_block_forward, but the two stream qkv/proj linears
# carry an optional LoRA delta. `saved` is the LoRA-MODIFIED activations (so the
# backward sees the correct q/k/v/att/attn_res etc.).
def double_block_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img: List[Float32], txt: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs, lora: DoubleBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)

    var img_x = _ta(img, [N_IMG, D], ctx)
    var txt_x = _ta(txt, [N_TXT, D], ctx)

    var ip = _stream_pre_lora[H, Dh](img_x, w.img, img_mod, lora.img, N_IMG, D, eps, ones_t, zeros_t, ctx)
    var tp = _stream_pre_lora[H, Dh](txt_x, w.txt, txt_mod, lora.txt, N_TXT, D, eps, ones_t, zeros_t, ctx)

    var q = concat(1, ctx, tp.q_rms[], ip.q_rms[])
    var k = concat(1, ctx, tp.k_rms[], ip.k_rms[])
    var v = concat(1, ctx, tp.v[], ip.v[])

    var q_rope = rope_interleaved(q, cos, sin, ctx)
    var k_rope = rope_interleaved(k, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = TArc(reshape(txt_att_4d, [N_TXT, D], ctx))
    var img_att = TArc(reshape(img_att_4d, [N_IMG, D], ctx))

    var ipost = _stream_post_lora(
        img_x, img_att, w.img, img_mod, lora.img, N_IMG, D, F, eps, ones_t, zeros_t, ctx)
    var tpost = _stream_post_lora(
        txt_x, txt_att, w.txt, txt_mod, lora.txt, N_TXT, D, F, eps, ones_t, zeros_t, ctx)

    var img_saved = _make_saved(img_x, ip, img_att, ipost)
    var txt_saved = _make_saved(txt_x, tp, txt_att, tpost)
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^)
    )

    var img_out = ipost.out[].to_host(ctx)
    var txt_out = tpost.out[].to_host(ctx)
    return DoubleBlockForward(img_out^, txt_out^, saved^)


# ── LoRA-aware per-stream POST backward ──────────────────────────────────────
# Mirrors _stream_post_backward; additionally, when the proj adapter is present,
# runs klein_lora_bwd at the wproj output (d_proj_out, input=att) → proj_d_a/_b
# and adds the LoRA d_x into d_att (so d_att flowing into sdpa is correct).
struct _StreamPostBackLora(Copyable, Movable):
    var base: _StreamPostBack
    var proj_d_a: List[Float32]
    var proj_d_b: List[Float32]

    def __init__(
        out self, var base: _StreamPostBack,
        var proj_d_a: List[Float32], var proj_d_b: List[Float32],
    ):
        self.base = base^
        self.proj_d_a = proj_d_a^
        self.proj_d_b = proj_d_b^


def _stream_post_backward_lora(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecs, lo: StreamLora, sv: StreamSaved,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPostBackLora:
    var no_bias_mlp = Optional[Tensor](None)
    var mlp_y = linear(sv.act[], w.wd[], no_bias_mlp^, ctx)
    var grg2 = gate_residual_backward(
        d_out[], sv.attn_res[], _t(mv.gate2.copy(), [D], ctx), mlp_y, ctx
    )
    var d_gate2 = grg2.d_g.to_host(ctx)

    # frozen wd backward: d_x ONLY (base d_wd computed-then-discarded by trainer).
    var d_d_dx = linear_backward_dx(grg2.d_y, w.wd[], N, F, D, ctx)

    var sgb = swiglu_backward(d_d_dx, sv.gate[], sv.up[], ctx)
    var d_gu = concat(1, ctx, sgb.d_gate, sgb.d_up)

    # frozen wgu backward: d_x ONLY (base d_wgu computed-then-discarded).
    var d_gu_dx = linear_backward_dx(d_gu, w.wgu[], N, D, 2 * F, ctx)

    var mb2 = modulate_backward(d_gu_dx, sv.ln2[], _t(mv.scale2.copy(), [D], ctx), ctx)
    var d_scale2 = mb2.d_scale.to_host(ctx)
    var d_shift2 = mb2.d_shift.to_host(ctx)

    var lnb2 = layer_norm_backward(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    var d_attn_res_total = TArc(add(grg2.d_x, lnb2.d_x, ctx))

    # proj_out = linear(att, Wproj) [+ LoRA]; recompute the LoRA-modified proj.
    var no_bias = Optional[Tensor](None)
    var proj_out = linear(att[], w.wproj[], no_bias^, ctx)
    if lo.proj:
        var att_h = att[].to_host(ctx)
        var dlt = klein_lora_fwd(att_h, lo.proj.value(), N, ctx)
        proj_out = _t(_add_lists(proj_out.to_host(ctx), dlt), [N, D], ctx)
    var grg1 = gate_residual_backward(
        d_attn_res_total[], x[], _t(mv.gate1.copy(), [D], ctx), proj_out, ctx
    )
    # Consume grg1 by to_host only (no field `^`-move out of the struct). d_y ->
    # host (LoRA bridge needs host); d_g -> host; d_x = residual branch -> host
    # (caller sums it with the pre branch).
    var d_gate1 = grg1.d_g.to_host(ctx)
    var d_proj_out_h = grg1.d_y.to_host(ctx)   # grad at the wproj OUTPUT (LoRA needs host)
    var d_x_res = grg1.d_x.to_host(ctx)

    # frozen proj backward: d_x ONLY (base d_wproj computed-then-discarded).
    var d_p_dx = linear_backward_dx(
        _t(d_proj_out_h.copy(), [N, D], ctx), w.wproj[], N, D, D, ctx
    )
    var d_att_h = d_p_dx.to_host(ctx)   # LoRA folds into it below

    # LoRA backward on wproj (input=att, d_y=d_proj_out): proj_d_a/_b + d_att_lo
    var proj_d_a = List[Float32]()
    var proj_d_b = List[Float32]()
    if lo.proj:
        var att_h2 = att[].to_host(ctx)
        var lg = klein_lora_bwd(d_proj_out_h, att_h2, lo.proj.value(), N, ctx)
        d_att_h = _add_lists(d_att_h, lg.d_x)   # LoRA contribution to projection input
        proj_d_a = lg.d_a.copy()
        proj_d_b = lg.d_b.copy()

    # d_wproj/d_wgu/d_wd are frozen-base grads (stripped above; LoRA path discards
    # them, base double gate still validates that exact d_w math) — empty placeholders.
    var base = _StreamPostBack(
        d_x_res^, d_att_h^, List[Float32](), List[Float32](), List[Float32](),
        d_gate1=d_gate1^,
        d_shift2=d_shift2^, d_scale2=d_scale2^, d_gate2=d_gate2^,
    )
    return _StreamPostBackLora(base^, proj_d_a^, proj_d_b^)


# ── LoRA-aware per-stream PRE backward ───────────────────────────────────────
# Mirrors _stream_pre_backward; additionally, when the qkv adapter is present,
# runs klein_lora_bwd at the wqkv output (d_qkv, input=norm) → qkv_d_a/_b and
# adds the LoRA d_x into d_norm (so d_x via layer_norm is correct).
struct _StreamPreBackLora(Copyable, Movable):
    var base: _StreamPreBack
    var qkv_d_a: List[Float32]
    var qkv_d_b: List[Float32]

    def __init__(
        out self, var base: _StreamPreBack,
        var qkv_d_a: List[Float32], var qkv_d_b: List[Float32],
    ):
        self.base = base^
        self.qkv_d_a = qkv_d_a^
        self.qkv_d_b = qkv_d_b^


def _stream_pre_backward_lora[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecs, lo: StreamLora, sv: StreamSaved,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPreBackLora:
    var rb_q = rms_norm_backward(d_q_rms, sv.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, sv.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    var d_q_pre_flat = reshape(rb_q.d_x, [N, D], ctx)
    var d_k_pre_flat = reshape(rb_k.d_x, [N, D], ctx)
    var d_v_flat = reshape(d_v, [N, D], ctx)
    var d_qkv = concat(1, ctx, d_q_pre_flat, d_k_pre_flat, d_v_flat)   # grad at wqkv OUTPUT [N,3D]

    # frozen qkv backward: d_x ONLY (base d_wqkv computed-then-discarded).
    var d_qkv_dx = linear_backward_dx(d_qkv, w.wqkv[], N, D, 3 * D, ctx)
    var d_norm_h = d_qkv_dx.to_host(ctx)

    # LoRA backward on wqkv (input=norm, d_y=d_qkv): qkv_d_a/_b + d_norm_lo
    var qkv_d_a = List[Float32]()
    var qkv_d_b = List[Float32]()
    if lo.qkv:
        var d_qkv_h = d_qkv.to_host(ctx)
        var norm_h = sv.norm[].to_host(ctx)
        var lg = klein_lora_bwd(d_qkv_h, norm_h, lo.qkv.value(), N, ctx)
        d_norm_h = _add_lists(d_norm_h, lg.d_x)
        qkv_d_a = lg.d_a.copy()
        qkv_d_b = lg.d_b.copy()

    var d_norm_t = _t(d_norm_h^, [N, D], ctx)
    var mb1 = modulate_backward(d_norm_t, sv.ln1[], _t(mv.scale1.copy(), [D], ctx), ctx)
    var d_scale1 = mb1.d_scale.to_host(ctx)
    var d_shift1 = mb1.d_shift.to_host(ctx)

    var lnb1 = layer_norm_backward(mb1.d_x, sv.x[], ones, eps, ctx)
    var d_x_norm = lnb1.d_x.to_host(ctx)
    # d_wqkv is a frozen-base grad (stripped above) — empty placeholder.
    var base = _StreamPreBack(
        d_x_norm^, List[Float32](), d_q_norm^, d_k_norm^, d_shift1^, d_scale1^
    )
    return _StreamPreBackLora(base^, qkv_d_a^, qkv_d_b^)


# ── BACKWARD of one DOUBLE block WITH LoRA on the attention projections ──────
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
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)

    var d_io_t = _ta(d_img_out, [N_IMG, D], ctx)
    var d_to_t = _ta(d_txt_out, [N_TXT, D], ctx)

    var ipb = _stream_post_backward_lora(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, lora.img, saved.img,
        N_IMG, D, F, eps, ones_t, ctx,
    )
    var tpb = _stream_post_backward_lora(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, lora.txt, saved.txt,
        N_TXT, D, F, eps, ones_t, ctx,
    )

    # d_att per stream comes back as host List; re-upload ONCE per stream as
    # [1,N,H,Dh] (byte-identical to [N,D]) for the device joint section.
    var d_tatt_4d = _t(tpb.base.d_att.copy(), [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = _t(ipb.base.d_att.copy(), [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh]

    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx,
    )

    var d_q_joint = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_joint = rope_backward(sb.d_k, cos, sin, True, ctx)

    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(sb.d_v, N_TXT, N_IMG, 1, ctx)

    var iprb = _stream_pre_backward_lora[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, lora.img, saved.img, N_IMG, D, eps, ones_t, ctx,
    )
    var tprb = _stream_pre_backward_lora[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, lora.txt, saved.txt, N_TXT, D, eps, ones_t, ctx,
    )

    var d_img_x = _add_lists(ipb.base.d_x, iprb.base.d_x)
    var d_txt_x = _add_lists(tpb.base.d_x, tprb.base.d_x)

    var img_grads = StreamGrads(
        d_img_x^, iprb.base.d_wqkv.copy(), ipb.base.d_wproj.copy(),
        ipb.base.d_wgu.copy(), ipb.base.d_wd.copy(),
        iprb.base.d_q_norm.copy(), iprb.base.d_k_norm.copy(),
        iprb.base.d_shift1.copy(), iprb.base.d_scale1.copy(), ipb.base.d_gate1.copy(),
        ipb.base.d_shift2.copy(), ipb.base.d_scale2.copy(), ipb.base.d_gate2.copy(),
    )
    var txt_grads = StreamGrads(
        d_txt_x^, tprb.base.d_wqkv.copy(), tpb.base.d_wproj.copy(),
        tpb.base.d_wgu.copy(), tpb.base.d_wd.copy(),
        tprb.base.d_q_norm.copy(), tprb.base.d_k_norm.copy(),
        tprb.base.d_shift1.copy(), tprb.base.d_scale1.copy(), tpb.base.d_gate1.copy(),
        tpb.base.d_shift2.copy(), tpb.base.d_scale2.copy(), tpb.base.d_gate2.copy(),
    )
    var base_grads = DoubleBlockGrads(img_grads^, txt_grads^)

    var img_lora = StreamLoraGrads(
        iprb.qkv_d_a.copy(), iprb.qkv_d_b.copy(), ipb.proj_d_a.copy(), ipb.proj_d_b.copy(),
    )
    var txt_lora = StreamLoraGrads(
        tprb.qkv_d_a.copy(), tprb.qkv_d_b.copy(), tpb.proj_d_a.copy(), tpb.proj_d_b.copy(),
    )
    return DoubleBlockLoraGrads(base_grads^, img_lora^, txt_lora^)
