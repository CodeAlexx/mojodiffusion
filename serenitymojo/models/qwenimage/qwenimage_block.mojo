# serenitymojo/models/qwenimage/qwenimage_block.mojo
#
# Qwen-Image MMDiT DOUBLE-STREAM block: forward (saving activations) +
# hand-chained backward (training) + LoRA variants. Mirrors the proven
# Klein double_block.mojo pattern, adapted to Qwen-Image's block math (the exact
# inference forward in models/dit/qwenimage_dit.mojo::_block_forward, lines
# 490-632, which read inference-flame/src/models/qwenimage_dit.rs line-by-line;
# recipe + LoRA targets cited from EriDiffusion-v2
# crates/eridiffusion-core/src/models/qwenimage.rs:1677-1907 + :70-82).
#
# DIFFERENCES vs Klein (what makes this a new compute, not a copy):
#   - SPLIT Q/K/V: six biased linears per block (to_q/to_k/to_v on img_normed,
#     add_q/k/v_proj on txt_normed), NOT one fused [3D,D] qkv. (Klein: fused wqkv.)
#   - GELU(tanh) MLP: ff_in -> linear(net.0.proj)+bias -> gelu -> linear(net.2)+bias.
#     (Klein: SwiGLU gate/up split.)
#   - BIASED linears throughout (q/k/v/out + ff_up/ff_down), so every linear
#     contributes a d_b grad. (Klein: no-bias linears.)
#   - LayerNorm (parameter-free, ones/zeros affine) on each stream pre-attn and
#     pre-mlp; QK-RMSNorm over head_dim (norm_q/norm_k/norm_added_q/norm_added_k).
#   - SEPARATE img_mod / txt_mod (each its own [6D] modulation), TXT-FIRST concat,
#     3-axis INTERLEAVED RoPE. (Same modulate / residual_gate AdaLN primitives.)
#
# FORWARD GRAPH (per block, streams s in {img, txt}, mod vecs from img_mod/txt_mod):
#   s_ln1     = layer_norm(s, 1, 0, eps)
#   s_normed  = modulate(s_ln1, scale1, shift1)        # (1+scale)*x + shift
#   s_q = linear(s_normed, Wq_s, bq_s) ; s_k = ... ; s_v = ...   # each [N,D]
#   s_q/s_k reshaped [1,N,H,Dh]; rms_norm(s_q, qnorm_s) ; rms_norm(s_k, knorm_s)
#   s_v reshaped [1,N,H,Dh]
#   JOINT (txt FIRST then img):
#     q = concat(1, txt_q, img_q) ; k,v likewise         # [1,S,H,Dh]
#     qr = rope_interleaved(q, cos, sin) ; kr = rope_interleaved(k, cos, sin)
#     att = sdpa(qr, kr, v, 1/sqrt(Dh))                  # [1,S,H,Dh]
#     txt_att = slice(att,1,0,N_TXT) ; img_att = slice(att,1,N_TXT,N_IMG)
#     reshape each [1,N,H,Dh] -> [N,D]
#   s_o       = linear(s_att, Wout_s, bout_s)            # [N,D]
#   s_attn_res= residual_gate(s, gate1, s_o)             # s + gate1*s_o
#   s_ln2     = layer_norm(s_attn_res, 1, 0, eps)
#   s_ff_in   = modulate(s_ln2, scale2, shift2)
#   s_ff_up   = linear(s_ff_in, Wup_s, bup_s)            # [N,F]
#   s_ff_act  = gelu(s_ff_up)
#   s_ff_down = linear(s_ff_act, Wdn_s, bdn_s)           # [N,D]
#   s_final   = residual_gate(s_attn_res, gate2, s_ff_down)
#
# BACKWARD: every arm is an EXISTING, gated kernel (linear_backward,
# layer_norm_backward, rms_norm_backward, gelu_backward, sdpa_backward,
# rope_backward, gate_residual_backward, modulate_backward, cat_backward,
# slice/reshape). This file only COMPOSES them in reverse. The joint-attention
# coupling means txt+img both flow OUT of the SAME sdpa_backward, then split via
# the cat/slice backward (txt FIRST).
#
# API boundary: img/txt enter + img_out/txt_out + every grad leave as host
# List[Float32] for the current stack/parity callers. Device tensors use BF16
# storage; kernels accumulate internally in F32.
#
# Mojo 1.0.0b1, NVIDIA GPU. `def` not `fn`; Tensor move-only (TArc in saved
# structs); biased linear = linear(x, w, Optional[Tensor](b), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor

# ── forward ops (GPU) ────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, reshape_in_place, slice, concat, add,
)

# ── backward arms (GPU; all pre-built + gated) ───────────────────────────────
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx, LinearGrads
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, RmsNormBackward,
    layer_norm_backward, LayerNormBackward,
)
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.attention_backward import sdpa_backward, SdpaGrads
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, GateResidualGrads, rope_backward,
)
from serenitymojo.ops.shape_backward import cat_backward, CatGrads2


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


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


def _ta(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, shape^, STDtype.BF16, ctx))


# ── per-stream AdaLN modulation vectors (each [D]) ───────────────────────────
# (shift1, scale1, gate1, shift2, scale2, gate2) — diffusers chunk order for the
# 6-chunk img_mod / txt_mod outputs (qwenimage.rs:1683 chunk(6); chunks[0]=shift1,
# [1]=scale1, [2]=gate1, [3]=shift2, [4]=scale2, [5]=gate2).
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


# ── per-stream trainable weights (DEVICE-RESIDENT TArc, uploaded ONCE) ────────
#   wq/wk/wv: [D,D]  bq/bk/bv: [D]      (split q/k/v projections)
#   wout: [D,D]  bout: [D]              (attention output projection)
#   wup: [F,D]   bup: [F]               (net.0.proj)
#   wdn: [D,F]   bdn: [D]               (net.2)
#   q_norm/k_norm: [Dh]                 (per-head QK rms scale)
struct StreamWeights(Copyable, Movable):
    var wq: TArc
    var wk: TArc
    var wv: TArc
    var bq: TArc
    var bk: TArc
    var bv: TArc
    var wout: TArc
    var bout: TArc
    var wup: TArc
    var bup: TArc
    var wdn: TArc
    var bdn: TArc
    var q_norm: TArc
    var k_norm: TArc

    def __init__(
        out self,
        var wq: List[Float32], var wk: List[Float32], var wv: List[Float32],
        var bq: List[Float32], var bk: List[Float32], var bv: List[Float32],
        var wout: List[Float32], var bout: List[Float32],
        var wup: List[Float32], var bup: List[Float32],
        var wdn: List[Float32], var bdn: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
        D: Int, F: Int, Dh: Int, ctx: DeviceContext,
    ) raises:
        self.wq = TArc(Tensor.from_host(wq^, [D, D], STDtype.BF16, ctx))
        self.wk = TArc(Tensor.from_host(wk^, [D, D], STDtype.BF16, ctx))
        self.wv = TArc(Tensor.from_host(wv^, [D, D], STDtype.BF16, ctx))
        self.bq = TArc(Tensor.from_host(bq^, [D], STDtype.BF16, ctx))
        self.bk = TArc(Tensor.from_host(bk^, [D], STDtype.BF16, ctx))
        self.bv = TArc(Tensor.from_host(bv^, [D], STDtype.BF16, ctx))
        self.wout = TArc(Tensor.from_host(wout^, [D, D], STDtype.BF16, ctx))
        self.bout = TArc(Tensor.from_host(bout^, [D], STDtype.BF16, ctx))
        self.wup = TArc(Tensor.from_host(wup^, [F, D], STDtype.BF16, ctx))
        self.bup = TArc(Tensor.from_host(bup^, [F], STDtype.BF16, ctx))
        self.wdn = TArc(Tensor.from_host(wdn^, [D, F], STDtype.BF16, ctx))
        self.bdn = TArc(Tensor.from_host(bdn^, [D], STDtype.BF16, ctx))
        self.q_norm = TArc(Tensor.from_host(q_norm^, [Dh], STDtype.BF16, ctx))
        self.k_norm = TArc(Tensor.from_host(k_norm^, [Dh], STDtype.BF16, ctx))

    def __init__(
        out self,
        var wq: TArc, var wk: TArc, var wv: TArc,
        var bq: TArc, var bk: TArc, var bv: TArc,
        var wout: TArc, var bout: TArc,
        var wup: TArc, var bup: TArc,
        var wdn: TArc, var bdn: TArc,
        var q_norm: TArc, var k_norm: TArc,
    ):
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.bq = bq^
        self.bk = bk^
        self.bv = bv^
        self.wout = wout^
        self.bout = bout^
        self.wup = wup^
        self.bup = bup^
        self.wdn = wdn^
        self.bdn = bdn^
        self.q_norm = q_norm^
        self.k_norm = k_norm^


struct DoubleBlockWeights(Copyable, Movable):
    var img: StreamWeights
    var txt: StreamWeights

    def __init__(out self, var img: StreamWeights, var txt: StreamWeights):
        self.img = img^
        self.txt = txt^


# ── saved activations per stream (DEVICE-RESIDENT via TArc) ───────────────────
struct StreamSaved(Copyable, Movable):
    var x: TArc        # [N,D]   block input
    var ln1: TArc      # [N,D]   layer_norm(x)
    var normed: TArc   # [N,D]   modulate(ln1, scale1, shift1)
    var q_pre: TArc    # [1,N,H,Dh]  q before rms
    var k_pre: TArc    # [1,N,H,Dh]
    var att: TArc      # [N,D]   per-stream attention slice (reshaped)
    var attn_res: TArc # [N,D]   residual_gate(x, gate1, out_proj(att))
    var ln2: TArc      # [N,D]   layer_norm(attn_res)
    var ff_in: TArc    # [N,D]   modulate(ln2, scale2, shift2)
    var ff_up: TArc    # [N,F]   linear(ff_in, Wup)+bup  (pre-gelu)
    var ff_act: TArc   # [N,F]   gelu(ff_up)

    def __init__(
        out self,
        var x: TArc, var ln1: TArc, var normed: TArc,
        var q_pre: TArc, var k_pre: TArc,
        var att: TArc, var attn_res: TArc,
        var ln2: TArc, var ff_in: TArc, var ff_up: TArc, var ff_act: TArc,
    ):
        self.x = x^
        self.ln1 = ln1^
        self.normed = normed^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.att = att^
        self.attn_res = attn_res^
        self.ln2 = ln2^
        self.ff_in = ff_in^
        self.ff_up = ff_up^
        self.ff_act = ff_act^


struct DoubleBlockSaved(Copyable, Movable):
    var img: StreamSaved
    var txt: StreamSaved
    var q_rope: TArc   # [1,S,H,Dh]  rope(concat q)
    var k_rope: TArc   # [1,S,H,Dh]  rope(concat k)
    var v_joint: TArc  # [1,S,H,Dh]  concat v

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


# ── backward result: stream input grad + every trainable weight/bias grad ─────
struct StreamGrads(Copyable, Movable):
    var d_x: List[Float32]
    var d_wq: List[Float32]
    var d_wk: List[Float32]
    var d_wv: List[Float32]
    var d_bq: List[Float32]
    var d_bk: List[Float32]
    var d_bv: List[Float32]
    var d_wout: List[Float32]
    var d_bout: List[Float32]
    var d_wup: List[Float32]
    var d_bup: List[Float32]
    var d_wdn: List[Float32]
    var d_bdn: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    # modulation-vector grads (block outputs; backproped into mod MLP by stack)
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32],
        var d_wq: List[Float32], var d_wk: List[Float32], var d_wv: List[Float32],
        var d_bq: List[Float32], var d_bk: List[Float32], var d_bv: List[Float32],
        var d_wout: List[Float32], var d_bout: List[Float32],
        var d_wup: List[Float32], var d_bup: List[Float32],
        var d_wdn: List[Float32], var d_bdn: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_shift1: List[Float32], var d_scale1: List[Float32], var d_gate1: List[Float32],
        var d_shift2: List[Float32], var d_scale2: List[Float32], var d_gate2: List[Float32],
    ):
        self.d_x = d_x^
        self.d_wq = d_wq^
        self.d_wk = d_wk^
        self.d_wv = d_wv^
        self.d_bq = d_bq^
        self.d_bk = d_bk^
        self.d_bv = d_bv^
        self.d_wout = d_wout^
        self.d_bout = d_bout^
        self.d_wup = d_wup^
        self.d_bup = d_bup^
        self.d_wdn = d_wdn^
        self.d_bdn = d_bdn^
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


# ── per-stream FORWARD up to per-stream q/k/v (pre-join), DEVICE-RESIDENT ──────
struct _StreamPre(Copyable, Movable):
    var ln1: TArc
    var normed: TArc
    var q_pre: TArc     # [1,N,H,Dh]  pre-rms
    var k_pre: TArc
    var q_rms: TArc     # [1,N,H,Dh]  rms_norm(q_pre, q_norm)
    var k_rms: TArc
    var v: TArc

    def __init__(
        out self, var ln1: TArc, var normed: TArc,
        var q_pre: TArc, var k_pre: TArc,
        var q_rms: TArc, var k_rms: TArc, var v: TArc,
    ):
        self.ln1 = ln1^
        self.normed = normed^
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
    var normed = modulate(
        ln1, _t(mv.scale1.copy(), [D], ctx), _t(mv.shift1.copy(), [D], ctx), ctx
    )
    # split q/k/v: three biased linears on the SAME normed input.
    var q_flat = linear(normed, w.wq[], Optional[Tensor](_clone_t(w.bq[], ctx)), ctx)
    var k_flat = linear(normed, w.wk[], Optional[Tensor](_clone_t(w.bk[], ctx)), ctx)
    var v_flat = linear(normed, w.wv[], Optional[Tensor](_clone_t(w.bv[], ctx)), ctx)
    var q_pre = reshape_owned(q_flat^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, N, H, Dh])
    var v = reshape_owned(v_flat^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPre(
        TArc(ln1^), TArc(normed^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
    )


# per-stream FORWARD from the joint attention slice to the stream output.
struct _StreamPost(Copyable, Movable):
    var out: TArc        # [N,D]  the stream final
    var attn_res: TArc
    var ln2: TArc
    var ff_in: TArc
    var ff_up: TArc
    var ff_act: TArc

    def __init__(
        out self, var out: TArc, var attn_res: TArc,
        var ln2: TArc, var ff_in: TArc, var ff_up: TArc, var ff_act: TArc,
    ):
        self.out = out^
        self.attn_res = attn_res^
        self.ln2 = ln2^
        self.ff_in = ff_in^
        self.ff_up = ff_up^
        self.ff_act = ff_act^


def _stream_post(
    x: TArc, att: TArc, w: StreamWeights, mv: ModVecs,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, zeros: Tensor,
    ctx: DeviceContext,
) raises -> _StreamPost:
    var out = linear(att[], w.wout[], Optional[Tensor](_clone_t(w.bout[], ctx)), ctx)  # [N,D]
    var attn_res = residual_gate(x[], _t(mv.gate1.copy(), [D], ctx), out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var ff_in = modulate(
        ln2, _t(mv.scale2.copy(), [D], ctx), _t(mv.shift2.copy(), [D], ctx), ctx
    )
    var ff_up = linear(ff_in, w.wup[], Optional[Tensor](_clone_t(w.bup[], ctx)), ctx)  # [N,F]
    var ff_act = gelu(ff_up, ctx)
    var ff_down = linear(ff_act, w.wdn[], Optional[Tensor](_clone_t(w.bdn[], ctx)), ctx)  # [N,D]
    var final = residual_gate(attn_res, _t(mv.gate2.copy(), [D], ctx), ff_down, ctx)
    return _StreamPost(
        TArc(final^), TArc(attn_res^), TArc(ln2^), TArc(ff_in^),
        TArc(ff_up^), TArc(ff_act^),
    )


# clone a small device tensor (bias) so it can pass as Optional[Tensor] (Tensor
# is move-only -> cannot borrow into Optional directly).
def _clone_t(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _make_saved(
    x: TArc, pre: _StreamPre, att: TArc, post: _StreamPost
) -> StreamSaved:
    return StreamSaved(
        x.copy(), pre.ln1.copy(), pre.normed.copy(), pre.q_pre.copy(), pre.k_pre.copy(),
        att.copy(), post.attn_res.copy(), post.ln2.copy(),
        post.ff_in.copy(), post.ff_up.copy(), post.ff_act.copy(),
    )


# ── FORWARD of one Qwen-Image double block ───────────────────────────────────
# cos/sin: precomputed 3-axis interleaved rope tables for the JOINT sequence,
# [S*H, Dh/2].
def double_block_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img: List[Float32], txt: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)

    var img_x = _ta(img, [N_IMG, D], ctx)
    var txt_x = _ta(txt, [N_TXT, D], ctx)

    var ip = _stream_pre[H, Dh](img_x, w.img, img_mod, N_IMG, D, eps, ones_t, zeros_t, ctx)
    var tp = _stream_pre[H, Dh](txt_x, w.txt, txt_mod, N_TXT, D, eps, ones_t, zeros_t, ctx)

    # JOINT concat TXT FIRST, then IMG (qwenimage_dit.mojo:578 cat txt|img).
    var q = concat(1, ctx, tp.q_rms[], ip.q_rms[])   # [1,S,H,Dh]
    var k = concat(1, ctx, tp.k_rms[], ip.k_rms[])
    var v = concat(1, ctx, tp.v[], ip.v[])

    var q_rope = rope_interleaved(q, cos, sin, ctx)
    var k_rope = rope_interleaved(k, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)   # [1,S,H,Dh]

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG, D]))

    var ipost = _stream_post(img_x, img_att, w.img, img_mod, N_IMG, D, F, eps, ones_t, zeros_t, ctx)
    var tpost = _stream_post(txt_x, txt_att, w.txt, txt_mod, N_TXT, D, F, eps, ones_t, zeros_t, ctx)

    var img_saved = _make_saved(img_x, ip, img_att, ipost)
    var txt_saved = _make_saved(txt_x, tp, txt_att, tpost)
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^)
    )

    var img_out = ipost.out[].to_host(ctx)
    var txt_out = tpost.out[].to_host(ctx)
    return DoubleBlockForward(img_out^, txt_out^, saved^)


# ── per-stream POST-attention backward: d_out -> (d_x via residual, d_att) + grads
struct _StreamPostBack(Copyable, Movable):
    var d_x: List[Float32]      # grad into stream input via gate1 residual [N,D]
    var d_att: List[Float32]    # grad into the joint-attention slice [N,D]
    var d_wout: List[Float32]
    var d_bout: List[Float32]
    var d_wup: List[Float32]
    var d_bup: List[Float32]
    var d_wdn: List[Float32]
    var d_bdn: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]

    def __init__(
        out self, var d_x: List[Float32], var d_att: List[Float32],
        var d_wout: List[Float32], var d_bout: List[Float32],
        var d_wup: List[Float32], var d_bup: List[Float32],
        var d_wdn: List[Float32], var d_bdn: List[Float32],
        var d_gate1: List[Float32],
        var d_shift2: List[Float32], var d_scale2: List[Float32], var d_gate2: List[Float32],
    ):
        self.d_x = d_x^
        self.d_att = d_att^
        self.d_wout = d_wout^
        self.d_bout = d_bout^
        self.d_wup = d_wup^
        self.d_bup = d_bup^
        self.d_wdn = d_wdn^
        self.d_bdn = d_bdn^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^


def _stream_post_backward(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecs, sv: StreamSaved,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPostBack:
    # final = residual_gate(attn_res, gate2, ff_down): recompute ff_down = linear(ff_act, Wdn)+bdn
    var ff_down = linear(sv.ff_act[], w.wdn[], Optional[Tensor](_clone_t(w.bdn[], ctx)), ctx)
    var grg2 = gate_residual_backward(
        d_out[], sv.attn_res[], _t(mv.gate2.copy(), [D], ctx), ff_down, ctx
    )
    var d_gate2 = grg2.d_g.to_host(ctx)
    # grg2.d_x = attn_res residual branch (device); grg2.d_y = d_ff_down (device)

    # ff_down = linear(ff_act, Wdn, bdn)
    var lb_dn = linear_backward(grg2.d_y, sv.ff_act[], w.wdn[], N, F, D, ctx)
    var d_wdn = lb_dn.d_w.to_host(ctx)
    var d_bdn = lb_dn.d_b.to_host(ctx)

    # ff_act = gelu(ff_up)
    var d_ff_up = gelu_backward(lb_dn.d_x, sv.ff_up[], ctx)

    # ff_up = linear(ff_in, Wup, bup)
    var lb_up = linear_backward(d_ff_up, sv.ff_in[], w.wup[], N, D, F, ctx)
    var d_wup = lb_up.d_w.to_host(ctx)
    var d_bup = lb_up.d_b.to_host(ctx)

    # ff_in = modulate(ln2, scale2, shift2)
    var mb2 = modulate_backward(lb_up.d_x, sv.ln2[], _t(mv.scale2.copy(), [D], ctx), ctx)
    var d_scale2 = mb2.d_scale.to_host(ctx)
    var d_shift2 = mb2.d_shift.to_host(ctx)

    # ln2 = layer_norm(attn_res, 1, 0)
    var lnb2 = layer_norm_backward(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    # attn_res feeds BOTH the residual (grg2.d_x) AND ln2 -> SUM on device
    var d_attn_res_total = TArc(add(grg2.d_x, lnb2.d_x, ctx))

    # attn_res = residual_gate(x, gate1, out): recompute out = linear(att, Wout)+bout
    var out_proj = linear(att[], w.wout[], Optional[Tensor](_clone_t(w.bout[], ctx)), ctx)
    var grg1 = gate_residual_backward(
        d_attn_res_total[], x[], _t(mv.gate1.copy(), [D], ctx), out_proj, ctx
    )
    var d_gate1 = grg1.d_g.to_host(ctx)
    var d_x_res = grg1.d_x.to_host(ctx)

    # out = linear(att, Wout, bout)
    var lb_out = linear_backward(grg1.d_y, att[], w.wout[], N, D, D, ctx)
    var d_wout = lb_out.d_w.to_host(ctx)
    var d_bout = lb_out.d_b.to_host(ctx)
    var d_att = lb_out.d_x.to_host(ctx)

    return _StreamPostBack(
        d_x_res^, d_att^, d_wout^, d_bout^, d_wup^, d_bup^, d_wdn^, d_bdn^,
        d_gate1=d_gate1^,
        d_shift2=d_shift2^, d_scale2=d_scale2^, d_gate2=d_gate2^,
    )


# ── per-stream PRE-attention backward: (d_q_rms, d_k_rms, d_v) -> d_x + grads ──
struct _StreamPreBack(Copyable, Movable):
    var d_x: List[Float32]
    var d_wq: List[Float32]
    var d_wk: List[Float32]
    var d_wv: List[Float32]
    var d_bq: List[Float32]
    var d_bk: List[Float32]
    var d_bv: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]

    def __init__(
        out self, var d_x: List[Float32],
        var d_wq: List[Float32], var d_wk: List[Float32], var d_wv: List[Float32],
        var d_bq: List[Float32], var d_bk: List[Float32], var d_bv: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_shift1: List[Float32], var d_scale1: List[Float32],
    ):
        self.d_x = d_x^
        self.d_wq = d_wq^
        self.d_wk = d_wk^
        self.d_wv = d_wv^
        self.d_bq = d_bq^
        self.d_bk = d_bk^
        self.d_bv = d_bv^
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
    # q_rms = rms_norm(q_pre, q_norm) over last dim Dh; d_q_rms [1,N,H,Dh].
    var rb_q = rms_norm_backward(d_q_rms, sv.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, sv.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    # reshape [1,N,H,Dh] -> [N,D] byte no-op for the linear backward.
    reshape_in_place(rb_q.d_x, [N, D])
    reshape_in_place(rb_k.d_x, [N, D])
    var d_v_flat = reshape(d_v, [N, D], ctx)

    # q = linear(normed, Wq, bq) ; k = linear(normed, Wk, bk) ; v = linear(normed, Wv, bv)
    var lb_q = linear_backward(rb_q.d_x, sv.normed[], w.wq[], N, D, D, ctx)
    var lb_k = linear_backward(rb_k.d_x, sv.normed[], w.wk[], N, D, D, ctx)
    var lb_v = linear_backward(d_v_flat, sv.normed[], w.wv[], N, D, D, ctx)
    var d_wq = lb_q.d_w.to_host(ctx)
    var d_wk = lb_k.d_w.to_host(ctx)
    var d_wv = lb_v.d_w.to_host(ctx)
    var d_bq = lb_q.d_b.to_host(ctx)
    var d_bk = lb_k.d_b.to_host(ctx)
    var d_bv = lb_v.d_b.to_host(ctx)

    # normed feeds all three q/k/v -> SUM the three d_x on device.
    var d_normed = TArc(add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx))

    # normed = modulate(ln1, scale1, shift1)
    var mb1 = modulate_backward(d_normed[], sv.ln1[], _t(mv.scale1.copy(), [D], ctx), ctx)
    var d_scale1 = mb1.d_scale.to_host(ctx)
    var d_shift1 = mb1.d_shift.to_host(ctx)

    # ln1 = layer_norm(x, 1, 0)
    var lnb1 = layer_norm_backward(mb1.d_x, sv.x[], ones, eps, ctx)
    var d_x_norm = lnb1.d_x.to_host(ctx)
    return _StreamPreBack(
        d_x_norm^, d_wq^, d_wk^, d_wv^, d_bq^, d_bk^, d_bv^,
        d_q_norm^, d_k_norm^, d_shift1^, d_scale1^,
    )


# ── BACKWARD of one Qwen-Image double block (hand-chained; joint coupling) ────
def double_block_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_img_out: List[Float32], d_txt_out: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs, saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)

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

    # ── join per-stream d_att back into joint d_att (txt FIRST) ──
    var d_tatt_4d = _t(tpb.d_att.copy(), [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = _t(ipb.d_att.copy(), [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh]

    # ── joint sdpa backward ──
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx,
    )

    # ── rope backward (interleaved; cos/sin non-learnable -> d_x only) ──
    var d_q_joint = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_joint = rope_backward(sb.d_k, cos, sin, True, ctx)

    # ── split joint q/k/v grads per stream (txt FIRST) along axis=1 ──
    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(sb.d_v, N_TXT, N_IMG, 1, ctx)

    # ── pre-attention backward per stream ──
    var iprb = _stream_pre_backward[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, saved.img, N_IMG, D, eps, ones_t, ctx,
    )
    var tprb = _stream_pre_backward[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, saved.txt, N_TXT, D, eps, ones_t, ctx,
    )

    # ── stream input grad = residual branch (post) + norm branch (pre) ──
    var d_img_x = _add_lists(ipb.d_x, iprb.d_x)
    var d_txt_x = _add_lists(tpb.d_x, tprb.d_x)

    var img_grads = StreamGrads(
        d_img_x^,
        iprb.d_wq.copy(), iprb.d_wk.copy(), iprb.d_wv.copy(),
        iprb.d_bq.copy(), iprb.d_bk.copy(), iprb.d_bv.copy(),
        ipb.d_wout.copy(), ipb.d_bout.copy(),
        ipb.d_wup.copy(), ipb.d_bup.copy(), ipb.d_wdn.copy(), ipb.d_bdn.copy(),
        iprb.d_q_norm.copy(), iprb.d_k_norm.copy(),
        iprb.d_shift1.copy(), iprb.d_scale1.copy(), ipb.d_gate1.copy(),
        ipb.d_shift2.copy(), ipb.d_scale2.copy(), ipb.d_gate2.copy(),
    )
    var txt_grads = StreamGrads(
        d_txt_x^,
        tprb.d_wq.copy(), tprb.d_wk.copy(), tprb.d_wv.copy(),
        tprb.d_bq.copy(), tprb.d_bk.copy(), tprb.d_bv.copy(),
        tpb.d_wout.copy(), tpb.d_bout.copy(),
        tpb.d_wup.copy(), tpb.d_bup.copy(), tpb.d_wdn.copy(), tpb.d_bdn.copy(),
        tprb.d_q_norm.copy(), tprb.d_k_norm.copy(),
        tprb.d_shift1.copy(), tprb.d_scale1.copy(), tpb.d_gate1.copy(),
        tpb.d_shift2.copy(), tpb.d_scale2.copy(), tpb.d_gate2.copy(),
    )
    return DoubleBlockGrads(img_grads^, txt_grads^)


# ═══════════════════════════════════════════════════════════════════════════
# LoRA-ON-PROJECTION VARIANT
#
# Targets (per stream — matches EDv2 qwenimage.rs:70-82 QWENIMAGE_TARGETS, 12/block):
#   q (to_q/add_q_proj), k, v, out (to_out.0/to_add_out)  [in=D,out=D each]
#   ff_up (img/txt_mlp.net.0.proj)  [in=D,out=F]
#   ff_down (img/txt_mlp.net.2)     [in=F,out=D]
# Forward adds the LoRA delta at each projection's linear output; backward returns
# d_A/d_B for each and folds the LoRA d_x contribution back into the projection-
# input grad. REUSES klein_lora_fwd / klein_lora_bwd (the model-agnostic
# y=linear(x,W) LoRA math = train_step._lora_fwd/_lora_bwd).
# ═══════════════════════════════════════════════════════════════════════════

from serenitymojo.models.klein.lora_block import (
    LoraAdapter, klein_lora_fwd, klein_lora_bwd, KleinLoraGrads,
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


# Optional LoRA adapters for one stream's six trained projections.
struct StreamLora(Copyable, Movable):
    var q: Optional[LoraAdapter]
    var k: Optional[LoraAdapter]
    var v: Optional[LoraAdapter]
    var out: Optional[LoraAdapter]
    var ff_up: Optional[LoraAdapter]
    var ff_down: Optional[LoraAdapter]

    def __init__(
        out self,
        var q: Optional[LoraAdapter], var k: Optional[LoraAdapter],
        var v: Optional[LoraAdapter], var out: Optional[LoraAdapter],
        var ff_up: Optional[LoraAdapter], var ff_down: Optional[LoraAdapter],
    ):
        self.q = q^
        self.k = k^
        self.v = v^
        self.out = out^
        self.ff_up = ff_up^
        self.ff_down = ff_down^


struct DoubleBlockLora(Copyable, Movable):
    var img: StreamLora
    var txt: StreamLora

    def __init__(out self, var img: StreamLora, var txt: StreamLora):
        self.img = img^
        self.txt = txt^


# Per-stream LoRA grads: d_A/d_B for the 6 adapters (empty when absent).
struct StreamLoraGrads(Copyable, Movable):
    var q_d_a: List[Float32]
    var q_d_b: List[Float32]
    var k_d_a: List[Float32]
    var k_d_b: List[Float32]
    var v_d_a: List[Float32]
    var v_d_b: List[Float32]
    var out_d_a: List[Float32]
    var out_d_b: List[Float32]
    var ff_up_d_a: List[Float32]
    var ff_up_d_b: List[Float32]
    var ff_down_d_a: List[Float32]
    var ff_down_d_b: List[Float32]

    def __init__(
        out self,
        var q_d_a: List[Float32], var q_d_b: List[Float32],
        var k_d_a: List[Float32], var k_d_b: List[Float32],
        var v_d_a: List[Float32], var v_d_b: List[Float32],
        var out_d_a: List[Float32], var out_d_b: List[Float32],
        var ff_up_d_a: List[Float32], var ff_up_d_b: List[Float32],
        var ff_down_d_a: List[Float32], var ff_down_d_b: List[Float32],
    ):
        self.q_d_a = q_d_a^
        self.q_d_b = q_d_b^
        self.k_d_a = k_d_a^
        self.k_d_b = k_d_b^
        self.v_d_a = v_d_a^
        self.v_d_b = v_d_b^
        self.out_d_a = out_d_a^
        self.out_d_b = out_d_b^
        self.ff_up_d_a = ff_up_d_a^
        self.ff_up_d_b = ff_up_d_b^
        self.ff_down_d_a = ff_down_d_a^
        self.ff_down_d_b = ff_down_d_b^


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


struct DoubleBlockLoraForward(Copyable, Movable):
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


def _empty() -> List[Float32]:
    return List[Float32]()


comptime QWEN_DIRECT_ALGO_DORA = 1
comptime QWEN_DIRECT_ALGO_OFT = 2
comptime QWEN_DIRECT_TGT_ATTN = 1
comptime QWEN_DIRECT_TGT_ALL = 2
comptime QD_IMG_Q = 0
comptime QD_IMG_K = 1
comptime QD_IMG_V = 2
comptime QD_IMG_OUT = 3
comptime QD_IMG_FF_UP = 4
comptime QD_IMG_FF_DOWN = 5
comptime QD_TXT_Q = 6
comptime QD_TXT_K = 7
comptime QD_TXT_V = 8
comptime QD_TXT_OUT = 9
comptime QD_TXT_FF_UP = 10
comptime QD_TXT_FF_DOWN = 11


def _qwen_direct_slot_targeted(slot: Int, targets: Int) raises -> Bool:
    if targets < QWEN_DIRECT_TGT_ATTN or targets > QWEN_DIRECT_TGT_ALL:
        raise Error("QwenBlockDirectLycoris: targets must be 1(attn)|2(all)")
    var s = slot % 12
    if s <= QD_IMG_OUT or (s >= QD_TXT_Q and s <= QD_TXT_OUT):
        return targets >= QWEN_DIRECT_TGT_ATTN
    return targets >= QWEN_DIRECT_TGT_ALL


struct QwenBlockDirectLycoris(Copyable, Movable):
    var algo: Int
    var dora: FlatDirectDoRASet
    var oft: FlatDirectOFTSet
    var img_q_slot: Int
    var img_k_slot: Int
    var img_v_slot: Int
    var img_out_slot: Int
    var img_ff_up_slot: Int
    var img_ff_down_slot: Int
    var txt_q_slot: Int
    var txt_k_slot: Int
    var txt_v_slot: Int
    var txt_out_slot: Int
    var txt_ff_up_slot: Int
    var txt_ff_down_slot: Int

    def __init__(
        out self, algo: Int, var dora: FlatDirectDoRASet,
        var oft: FlatDirectOFTSet, base_slot: Int, targets: Int,
    ) raises:
        var img_q = -1
        var img_k = -1
        var img_v = -1
        var img_out = -1
        var img_ff_up = -1
        var img_ff_down = -1
        var txt_q = -1
        var txt_k = -1
        var txt_v = -1
        var txt_out = -1
        var txt_ff_up = -1
        var txt_ff_down = -1
        var compact = base_slot
        for slot in range(12):
            if not _qwen_direct_slot_targeted(slot, targets):
                continue
            if slot == QD_IMG_Q:
                img_q = compact
            elif slot == QD_IMG_K:
                img_k = compact
            elif slot == QD_IMG_V:
                img_v = compact
            elif slot == QD_IMG_OUT:
                img_out = compact
            elif slot == QD_IMG_FF_UP:
                img_ff_up = compact
            elif slot == QD_IMG_FF_DOWN:
                img_ff_down = compact
            elif slot == QD_TXT_Q:
                txt_q = compact
            elif slot == QD_TXT_K:
                txt_k = compact
            elif slot == QD_TXT_V:
                txt_v = compact
            elif slot == QD_TXT_OUT:
                txt_out = compact
            elif slot == QD_TXT_FF_UP:
                txt_ff_up = compact
            elif slot == QD_TXT_FF_DOWN:
                txt_ff_down = compact
            compact += 1
        self.algo = algo
        self.dora = dora^
        self.oft = oft^
        self.img_q_slot = img_q
        self.img_k_slot = img_k
        self.img_v_slot = img_v
        self.img_out_slot = img_out
        self.img_ff_up_slot = img_ff_up
        self.img_ff_down_slot = img_ff_down
        self.txt_q_slot = txt_q
        self.txt_k_slot = txt_k
        self.txt_v_slot = txt_v
        self.txt_out_slot = txt_out
        self.txt_ff_up_slot = txt_ff_up
        self.txt_ff_down_slot = txt_ff_down


def _qwen_direct_flat_slot(direct: QwenBlockDirectLycoris, slot: Int) raises -> Int:
    if slot == QD_IMG_Q:
        return direct.img_q_slot
    if slot == QD_IMG_K:
        return direct.img_k_slot
    if slot == QD_IMG_V:
        return direct.img_v_slot
    if slot == QD_IMG_OUT:
        return direct.img_out_slot
    if slot == QD_IMG_FF_UP:
        return direct.img_ff_up_slot
    if slot == QD_IMG_FF_DOWN:
        return direct.img_ff_down_slot
    if slot == QD_TXT_Q:
        return direct.txt_q_slot
    if slot == QD_TXT_K:
        return direct.txt_k_slot
    if slot == QD_TXT_V:
        return direct.txt_v_slot
    if slot == QD_TXT_OUT:
        return direct.txt_out_slot
    if slot == QD_TXT_FF_UP:
        return direct.txt_ff_up_slot
    if slot == QD_TXT_FF_DOWN:
        return direct.txt_ff_down_slot
    raise Error("QwenBlockDirectLycoris: bad direct slot")


struct QwenDirectProjectionGrad(Copyable, Movable):
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


struct _QwenDirectProjectionGradDev(Movable):
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


def _qwen_direct_grad_public(g: _QwenDirectProjectionGradDev) -> QwenDirectProjectionGrad:
    return QwenDirectProjectionGrad(
        g.d_a.copy(), g.d_b.copy(), g.d_m.copy(), g.d_vec.copy(),
    )


struct QwenStreamDirectLycorisGrads(Copyable, Movable):
    var d_x: List[Float32]
    var q: QwenDirectProjectionGrad
    var k: QwenDirectProjectionGrad
    var v: QwenDirectProjectionGrad
    var out_proj: QwenDirectProjectionGrad
    var ff_up: QwenDirectProjectionGrad
    var ff_down: QwenDirectProjectionGrad

    def __init__(
        out self, var d_x: List[Float32],
        var q: QwenDirectProjectionGrad, var k: QwenDirectProjectionGrad,
        var v: QwenDirectProjectionGrad, var out_g: QwenDirectProjectionGrad,
        var ff_up: QwenDirectProjectionGrad, var ff_down: QwenDirectProjectionGrad,
    ):
        self.d_x = d_x^
        self.q = q^
        self.k = k^
        self.v = v^
        self.out_proj = out_g^
        self.ff_up = ff_up^
        self.ff_down = ff_down^


struct QwenBlockDirectLycorisGrads(Copyable, Movable):
    var img: QwenStreamDirectLycorisGrads
    var txt: QwenStreamDirectLycorisGrads

    def __init__(
        out self,
        var img: QwenStreamDirectLycorisGrads,
        var txt: QwenStreamDirectLycorisGrads,
    ):
        self.img = img^
        self.txt = txt^


struct DoubleBlockDirectLycorisForward(Copyable, Movable):
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


def _add_optional_bias(y: Tensor, bias: Tensor, ctx: DeviceContext) raises -> Tensor:
    if bias.dtype() != y.dtype():
        var bc = cast_tensor(bias, y.dtype(), ctx)
        return add(y, bc, ctx)
    return add(y, bias, ctx)


def _qwen_oft_vec_tensor(
    set: FlatDirectOFTSet, slot: Int, ctx: DeviceContext,
) raises -> Tensor:
    if slot < 0 or slot >= len(set.ad):
        raise Error("QwenBlockDirectLycoris OFT: slot out of range")
    if not set.active[slot]:
        raise Error("QwenBlockDirectLycoris OFT: inactive slot")
    ref sl = set.ad[slot]
    if sl.b != 4:
        raise Error("QwenBlockDirectLycoris OFT: only block_size=4 is wired on GPU")
    return Tensor.from_host(sl.vec.copy(), [sl.r, 6], STDtype.F32, ctx)


def _direct_proj_fwd_device(
    direct: QwenBlockDirectLycoris, slot: Int,
    x: Tensor, w_orig: Tensor, bias: Tensor,
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> Tensor:
    var flat_slot = _qwen_direct_flat_slot(direct, slot)
    if flat_slot < 0:
        return linear(x, w_orig, Optional[Tensor](_clone_t(bias, ctx)), ctx)
    if direct.algo == QWEN_DIRECT_ALGO_DORA:
        if flat_slot >= len(direct.dora.ad):
            raise Error("QwenBlockDirectLycoris DoRA: slot out of range")
        if not direct.dora.active[flat_slot]:
            raise Error("QwenBlockDirectLycoris DoRA: inactive slot")
        if direct.dora.ad[flat_slot].in_f != in_f or direct.dora.ad[flat_slot].out_f != out_f:
            raise Error("QwenBlockDirectLycoris DoRA: slot shape mismatch")
        var dev = dora_device_from_host(direct.dora.ad[flat_slot], ctx)
        var y = dora_substitution_forward_device(x, w_orig, dev, ctx)
        return _add_optional_bias(y^, bias, ctx)
    if direct.algo == QWEN_DIRECT_ALGO_OFT:
        if flat_slot >= len(direct.oft.ad):
            raise Error("QwenBlockDirectLycoris OFT: slot out of range")
        if direct.oft.ad[flat_slot].in_f != in_f or direct.oft.ad[flat_slot].out_f != out_f:
            raise Error("QwenBlockDirectLycoris OFT: slot shape mismatch")
        var vec = _qwen_oft_vec_tensor(direct.oft, flat_slot, ctx)
        var x_rot = oft_ot_rotate_b4(x, vec, ctx)
        return linear(x_rot, w_orig, Optional[Tensor](_clone_t(bias, ctx)), ctx)
    raise Error("QwenBlockDirectLycoris: unsupported direct algorithm")


def _direct_proj_bwd_device(
    direct: QwenBlockDirectLycoris, slot: Int,
    d_y: Tensor, x: Tensor, w_orig: Tensor,
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> _QwenDirectProjectionGradDev:
    var flat_slot = _qwen_direct_flat_slot(direct, slot)
    if flat_slot < 0:
        var dx = linear_backward_dx(d_y, w_orig, M, in_f, out_f, ctx)
        return _QwenDirectProjectionGradDev(
            _empty(), _empty(), _empty(), _empty(), TArc(dx^),
        )
    if direct.algo == QWEN_DIRECT_ALGO_DORA:
        if flat_slot >= len(direct.dora.ad):
            raise Error("QwenBlockDirectLycoris DoRA backward: slot out of range")
        if not direct.dora.active[flat_slot]:
            raise Error("QwenBlockDirectLycoris DoRA backward: inactive slot")
        if direct.dora.ad[flat_slot].in_f != in_f or direct.dora.ad[flat_slot].out_f != out_f:
            raise Error("QwenBlockDirectLycoris DoRA backward: slot shape mismatch")
        var dev = dora_device_from_host(direct.dora.ad[flat_slot], ctx)
        var g = dora_substitution_backward_device(d_y, x, w_orig, dev, ctx)
        return _QwenDirectProjectionGradDev(
            g.d_a.to_host(ctx), g.d_b.to_host(ctx), g.d_m.to_host(ctx),
            _empty(), TArc(g.d_x.clone(ctx)),
        )
    if direct.algo == QWEN_DIRECT_ALGO_OFT:
        if flat_slot >= len(direct.oft.ad):
            raise Error("QwenBlockDirectLycoris OFT backward: slot out of range")
        if direct.oft.ad[flat_slot].in_f != in_f or direct.oft.ad[flat_slot].out_f != out_f:
            raise Error("QwenBlockDirectLycoris OFT backward: slot shape mismatch")
        var vec = _qwen_oft_vec_tensor(direct.oft, flat_slot, ctx)
        var d_x_rot = linear_backward_dx(d_y, w_orig, M, in_f, out_f, ctx)
        if d_x_rot.dtype() != x.dtype():
            d_x_rot = cast_tensor(d_x_rot^, x.dtype(), ctx)
        if d_x_rot.shape() != x.shape():
            d_x_rot = reshape_owned(d_x_rot^, x.shape())
        var g = oft_ot_rotate_backward_b4(d_x_rot^, x, vec, ctx)
        return _QwenDirectProjectionGradDev(
            _empty(), _empty(), _empty(), g.d_vec.to_host(ctx), TArc(g.d_x.clone(ctx)),
        )
    raise Error("QwenBlockDirectLycoris: unsupported direct algorithm")


def _stream_pre_direct[
    H: Int, Dh: Int
](
    x: TArc, w: StreamWeights, mv: ModVecs, direct: QwenBlockDirectLycoris,
    q_slot: Int, k_slot: Int, v_slot: Int,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPre:
    var ln1 = layer_norm(x[], ones, zeros, eps, ctx)
    var normed = modulate(
        ln1, _t(mv.scale1.copy(), [D], ctx), _t(mv.shift1.copy(), [D], ctx), ctx
    )
    var q_flat = _direct_proj_fwd_device(
        direct, q_slot, normed, w.wq[], w.bq[], N, D, D, ctx,
    )
    var k_flat = _direct_proj_fwd_device(
        direct, k_slot, normed, w.wk[], w.bk[], N, D, D, ctx,
    )
    var v_flat = _direct_proj_fwd_device(
        direct, v_slot, normed, w.wv[], w.bv[], N, D, D, ctx,
    )
    var q_pre = reshape_owned(q_flat^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, N, H, Dh])
    var v = reshape_owned(v_flat^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPre(
        TArc(ln1^), TArc(normed^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
    )


def _stream_post_direct(
    x: TArc, att: TArc, w: StreamWeights, mv: ModVecs,
    direct: QwenBlockDirectLycoris, out_slot: Int, ff_up_slot: Int, ff_down_slot: Int,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, zeros: Tensor,
    ctx: DeviceContext,
) raises -> _StreamPost:
    var out = _direct_proj_fwd_device(
        direct, out_slot, att[], w.wout[], w.bout[], N, D, D, ctx,
    )
    var attn_res = residual_gate(x[], _t(mv.gate1.copy(), [D], ctx), out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var ff_in = modulate(
        ln2, _t(mv.scale2.copy(), [D], ctx), _t(mv.shift2.copy(), [D], ctx), ctx
    )
    var ff_up = _direct_proj_fwd_device(
        direct, ff_up_slot, ff_in, w.wup[], w.bup[], N, D, F, ctx,
    )
    var ff_act = gelu(ff_up, ctx)
    var ff_down = _direct_proj_fwd_device(
        direct, ff_down_slot, ff_act, w.wdn[], w.bdn[], N, F, D, ctx,
    )
    var final = residual_gate(attn_res, _t(mv.gate2.copy(), [D], ctx), ff_down, ctx)
    return _StreamPost(
        TArc(final^), TArc(attn_res^), TArc(ln2^), TArc(ff_in^),
        TArc(ff_up^), TArc(ff_act^),
    )


def double_block_direct_lycoris_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img: List[Float32], txt: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs,
    direct: QwenBlockDirectLycoris,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockDirectLycorisForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)

    var img_x = _ta(img, [N_IMG, D], ctx)
    var txt_x = _ta(txt, [N_TXT, D], ctx)

    var ip = _stream_pre_direct[H, Dh](
        img_x, w.img, img_mod, direct,
        QD_IMG_Q, QD_IMG_K, QD_IMG_V,
        N_IMG, D, eps, ones_t, zeros_t, ctx,
    )
    var tp = _stream_pre_direct[H, Dh](
        txt_x, w.txt, txt_mod, direct,
        QD_TXT_Q, QD_TXT_K, QD_TXT_V,
        N_TXT, D, eps, ones_t, zeros_t, ctx,
    )

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

    var ipost = _stream_post_direct(
        img_x, img_att, w.img, img_mod, direct,
        QD_IMG_OUT, QD_IMG_FF_UP, QD_IMG_FF_DOWN,
        N_IMG, D, F, eps, ones_t, zeros_t, ctx,
    )
    var tpost = _stream_post_direct(
        txt_x, txt_att, w.txt, txt_mod, direct,
        QD_TXT_OUT, QD_TXT_FF_UP, QD_TXT_FF_DOWN,
        N_TXT, D, F, eps, ones_t, zeros_t, ctx,
    )

    var img_saved = _make_saved(img_x, ip, img_att, ipost)
    var txt_saved = _make_saved(txt_x, tp, txt_att, tpost)
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^)
    )

    var img_out = ipost.out[].to_host(ctx)
    var txt_out = tpost.out[].to_host(ctx)
    return DoubleBlockDirectLycorisForward(img_out^, txt_out^, saved^)


struct _StreamPostBackDirect(Movable):
    var d_x: TArc
    var d_att: TArc
    var out_g: _QwenDirectProjectionGradDev
    var ff_up_g: _QwenDirectProjectionGradDev
    var ff_down_g: _QwenDirectProjectionGradDev

    def __init__(
        out self, var d_x: TArc, var d_att: TArc,
        var out_g: _QwenDirectProjectionGradDev,
        var ff_up_g: _QwenDirectProjectionGradDev,
        var ff_down_g: _QwenDirectProjectionGradDev,
    ):
        self.d_x = d_x^
        self.d_att = d_att^
        self.out_g = out_g^
        self.ff_up_g = ff_up_g^
        self.ff_down_g = ff_down_g^


def _stream_post_backward_direct(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecs, direct: QwenBlockDirectLycoris,
    sv: StreamSaved,
    out_slot: Int, ff_up_slot: Int, ff_down_slot: Int,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPostBackDirect:
    var ff_down = _direct_proj_fwd_device(
        direct, ff_down_slot, sv.ff_act[], w.wdn[], w.bdn[], N, F, D, ctx,
    )
    var grg2 = gate_residual_backward(
        d_out[], sv.attn_res[], _t(mv.gate2.copy(), [D], ctx), ff_down, ctx
    )

    var ff_down_g = _direct_proj_bwd_device(
        direct, ff_down_slot, grg2.d_y, sv.ff_act[], w.wdn[], N, F, D, ctx,
    )
    var d_ff_up = gelu_backward(ff_down_g.d_x[], sv.ff_up[], ctx)

    var ff_up_g = _direct_proj_bwd_device(
        direct, ff_up_slot, d_ff_up, sv.ff_in[], w.wup[], N, D, F, ctx,
    )

    var mb2 = modulate_backward(ff_up_g.d_x[], sv.ln2[], _t(mv.scale2.copy(), [D], ctx), ctx)
    var lnb2 = layer_norm_backward(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    var d_attn_res_total = TArc(add(grg2.d_x, lnb2.d_x, ctx))

    var out_proj = _direct_proj_fwd_device(
        direct, out_slot, att[], w.wout[], w.bout[], N, D, D, ctx,
    )
    var grg1 = gate_residual_backward(
        d_attn_res_total[], x[], _t(mv.gate1.copy(), [D], ctx), out_proj, ctx
    )
    var out_g = _direct_proj_bwd_device(
        direct, out_slot, grg1.d_y, att[], w.wout[], N, D, D, ctx,
    )
    return _StreamPostBackDirect(
        TArc(grg1.d_x.clone(ctx)), out_g.d_x.copy(),
        out_g^, ff_up_g^, ff_down_g^,
    )


struct _StreamPreBackDirect(Movable):
    var d_x: TArc
    var q_g: _QwenDirectProjectionGradDev
    var k_g: _QwenDirectProjectionGradDev
    var v_g: _QwenDirectProjectionGradDev

    def __init__(
        out self, var d_x: TArc,
        var q_g: _QwenDirectProjectionGradDev,
        var k_g: _QwenDirectProjectionGradDev,
        var v_g: _QwenDirectProjectionGradDev,
    ):
        self.d_x = d_x^
        self.q_g = q_g^
        self.k_g = k_g^
        self.v_g = v_g^


def _stream_pre_backward_direct[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecs, direct: QwenBlockDirectLycoris,
    sv: StreamSaved, q_slot: Int, k_slot: Int, v_slot: Int,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPreBackDirect:
    var rb_q = rms_norm_backward(d_q_rms, sv.q_pre[], w.q_norm[], eps, ctx)
    var rb_k = rms_norm_backward(d_k_rms, sv.k_pre[], w.k_norm[], eps, ctx)

    reshape_in_place(rb_q.d_x, [N, D])
    reshape_in_place(rb_k.d_x, [N, D])
    var d_v_flat = reshape(d_v, [N, D], ctx)

    var q_g = _direct_proj_bwd_device(
        direct, q_slot, rb_q.d_x, sv.normed[], w.wq[], N, D, D, ctx,
    )
    var k_g = _direct_proj_bwd_device(
        direct, k_slot, rb_k.d_x, sv.normed[], w.wk[], N, D, D, ctx,
    )
    var v_g = _direct_proj_bwd_device(
        direct, v_slot, d_v_flat, sv.normed[], w.wv[], N, D, D, ctx,
    )
    var d_normed = TArc(add(add(q_g.d_x[], k_g.d_x[], ctx), v_g.d_x[], ctx))

    var mb1 = modulate_backward(d_normed[], sv.ln1[], _t(mv.scale1.copy(), [D], ctx), ctx)
    var lnb1 = layer_norm_backward(mb1.d_x, sv.x[], ones, eps, ctx)
    return _StreamPreBackDirect(
        TArc(lnb1.d_x.clone(ctx)), q_g^, k_g^, v_g^,
    )


def double_block_direct_lycoris_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_img_out: List[Float32], d_txt_out: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs,
    direct: QwenBlockDirectLycoris, saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> QwenBlockDirectLycorisGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)

    var d_io_t = _ta(d_img_out, [N_IMG, D], ctx)
    var d_to_t = _ta(d_txt_out, [N_TXT, D], ctx)

    var ipb = _stream_post_backward_direct(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, direct, saved.img,
        QD_IMG_OUT, QD_IMG_FF_UP, QD_IMG_FF_DOWN,
        N_IMG, D, F, eps, ones_t, ctx,
    )
    var tpb = _stream_post_backward_direct(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, direct, saved.txt,
        QD_TXT_OUT, QD_TXT_FF_UP, QD_TXT_FF_DOWN,
        N_TXT, D, F, eps, ones_t, ctx,
    )

    # Qwen concatenates txt first in forward, so rebuild the joint grad in the
    # same order after reshaping the direct per-stream device grads.
    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)

    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx,
    )
    var d_q_joint = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_joint = rope_backward(sb.d_k, cos, sin, True, ctx)
    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(sb.d_v, N_TXT, N_IMG, 1, ctx)

    var iprb = _stream_pre_backward_direct[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, direct, saved.img,
        QD_IMG_Q, QD_IMG_K, QD_IMG_V, N_IMG, D, eps, ones_t, ctx,
    )
    var tprb = _stream_pre_backward_direct[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, direct, saved.txt,
        QD_TXT_Q, QD_TXT_K, QD_TXT_V, N_TXT, D, eps, ones_t, ctx,
    )

    var d_img_t = add(ipb.d_x[], iprb.d_x[], ctx)
    var d_txt_t = add(tpb.d_x[], tprb.d_x[], ctx)
    var d_img_x = d_img_t.to_host(ctx)
    var d_txt_x = d_txt_t.to_host(ctx)

    var img = QwenStreamDirectLycorisGrads(
        d_img_x^,
        _qwen_direct_grad_public(iprb.q_g^),
        _qwen_direct_grad_public(iprb.k_g^),
        _qwen_direct_grad_public(iprb.v_g^),
        _qwen_direct_grad_public(ipb.out_g^),
        _qwen_direct_grad_public(ipb.ff_up_g^),
        _qwen_direct_grad_public(ipb.ff_down_g^),
    )
    var txt = QwenStreamDirectLycorisGrads(
        d_txt_x^,
        _qwen_direct_grad_public(tprb.q_g^),
        _qwen_direct_grad_public(tprb.k_g^),
        _qwen_direct_grad_public(tprb.v_g^),
        _qwen_direct_grad_public(tpb.out_g^),
        _qwen_direct_grad_public(tpb.ff_up_g^),
        _qwen_direct_grad_public(tpb.ff_down_g^),
    )
    return QwenBlockDirectLycorisGrads(img^, txt^)


# Add the LoRA contribution of a projection (on host input x_h [M,in]) into a
# device output Tensor y [M,out]. Returns y + delta as a fresh device Tensor.
def _add_lora_delta(
    y: Tensor, x_h: List[Float32], lo: Optional[LoraAdapter], M: Int, ctx: DeviceContext,
) raises -> Tensor:
    if not lo:
        return _clone_t(y, ctx)
    var delta_h = klein_lora_fwd(x_h, lo.value(), M, ctx)
    var delta = _t(delta_h^, y.shape().copy(), ctx)
    return add(y, delta, ctx)


# ── per-stream LoRA FORWARD up to q/k/v (deltas added at q/k/v linears) ───────
def _stream_pre_lora[
    H: Int, Dh: Int
](
    x: TArc, w: StreamWeights, mv: ModVecs, slo: StreamLora,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPre:
    var ln1 = layer_norm(x[], ones, zeros, eps, ctx)
    var normed = modulate(
        ln1, _t(mv.scale1.copy(), [D], ctx), _t(mv.shift1.copy(), [D], ctx), ctx
    )
    var normed_h = normed.to_host(ctx)
    var q_base = linear(normed, w.wq[], Optional[Tensor](_clone_t(w.bq[], ctx)), ctx)
    var k_base = linear(normed, w.wk[], Optional[Tensor](_clone_t(w.bk[], ctx)), ctx)
    var v_base = linear(normed, w.wv[], Optional[Tensor](_clone_t(w.bv[], ctx)), ctx)
    var q_flat = _add_lora_delta(q_base, normed_h, slo.q, N, ctx)
    var k_flat = _add_lora_delta(k_base, normed_h, slo.k, N, ctx)
    var v_flat = _add_lora_delta(v_base, normed_h, slo.v, N, ctx)
    var q_pre = reshape_owned(q_flat^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, N, H, Dh])
    var v = reshape_owned(v_flat^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPre(
        TArc(ln1^), TArc(normed^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
    )


# ── per-stream LoRA FORWARD from attention slice to output (deltas at out/ff) ──
def _stream_post_lora(
    x: TArc, att: TArc, w: StreamWeights, mv: ModVecs, slo: StreamLora,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, zeros: Tensor,
    ctx: DeviceContext,
) raises -> _StreamPost:
    var att_h = att[].to_host(ctx)
    var out_base = linear(att[], w.wout[], Optional[Tensor](_clone_t(w.bout[], ctx)), ctx)
    var out = _add_lora_delta(out_base, att_h, slo.out, N, ctx)
    var attn_res = residual_gate(x[], _t(mv.gate1.copy(), [D], ctx), out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var ff_in = modulate(
        ln2, _t(mv.scale2.copy(), [D], ctx), _t(mv.shift2.copy(), [D], ctx), ctx
    )
    var ff_in_h = ff_in.to_host(ctx)
    var ff_up_base = linear(ff_in, w.wup[], Optional[Tensor](_clone_t(w.bup[], ctx)), ctx)
    var ff_up = _add_lora_delta(ff_up_base, ff_in_h, slo.ff_up, N, ctx)
    var ff_act = gelu(ff_up, ctx)
    var ff_act_h = ff_act.to_host(ctx)
    var ff_down_base = linear(ff_act, w.wdn[], Optional[Tensor](_clone_t(w.bdn[], ctx)), ctx)
    var ff_down = _add_lora_delta(ff_down_base, ff_act_h, slo.ff_down, N, ctx)
    var final = residual_gate(attn_res, _t(mv.gate2.copy(), [D], ctx), ff_down, ctx)
    return _StreamPost(
        TArc(final^), TArc(attn_res^), TArc(ln2^), TArc(ff_in^),
        TArc(ff_up^), TArc(ff_act^),
    )


# ── FORWARD of one Qwen-Image double block WITH LoRA ──────────────────────────
def double_block_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img: List[Float32], txt: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs,
    lora: DoubleBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockLoraForward:
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
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG, D]))

    var ipost = _stream_post_lora(img_x, img_att, w.img, img_mod, lora.img, N_IMG, D, F, eps, ones_t, zeros_t, ctx)
    var tpost = _stream_post_lora(txt_x, txt_att, w.txt, txt_mod, lora.txt, N_TXT, D, F, eps, ones_t, zeros_t, ctx)

    var img_saved = _make_saved(img_x, ip, img_att, ipost)
    var txt_saved = _make_saved(txt_x, tp, txt_att, tpost)
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^)
    )

    var img_out = ipost.out[].to_host(ctx)
    var txt_out = tpost.out[].to_host(ctx)
    return DoubleBlockLoraForward(img_out^, txt_out^, saved^)


# ── per-stream LoRA POST-attention backward (base grads + LoRA d_A/d_B) ───────
# Returns the base _StreamPostBack plus the out/ff_up/ff_down adapter grads, with
# each LoRA d_x_lo already folded into the relevant projection-input grad.
struct _StreamPostBackLora(Copyable, Movable):
    var base: _StreamPostBack
    var out_d_a: List[Float32]
    var out_d_b: List[Float32]
    var ff_up_d_a: List[Float32]
    var ff_up_d_b: List[Float32]
    var ff_down_d_a: List[Float32]
    var ff_down_d_b: List[Float32]

    def __init__(
        out self, var base: _StreamPostBack,
        var out_d_a: List[Float32], var out_d_b: List[Float32],
        var ff_up_d_a: List[Float32], var ff_up_d_b: List[Float32],
        var ff_down_d_a: List[Float32], var ff_down_d_b: List[Float32],
    ):
        self.base = base^
        self.out_d_a = out_d_a^
        self.out_d_b = out_d_b^
        self.ff_up_d_a = ff_up_d_a^
        self.ff_up_d_b = ff_up_d_b^
        self.ff_down_d_a = ff_down_d_a^
        self.ff_down_d_b = ff_down_d_b^


def _stream_post_backward_lora(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecs, slo: StreamLora, sv: StreamSaved,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPostBackLora:
    # ff_down = base + LoRA: recompute base ff_down (+ LoRA delta) for the gate residual.
    var ff_act_h = sv.ff_act[].to_host(ctx)
    var ff_down_base = linear(sv.ff_act[], w.wdn[], Optional[Tensor](_clone_t(w.bdn[], ctx)), ctx)
    var ff_down = _add_lora_delta(ff_down_base, ff_act_h, slo.ff_down, N, ctx)
    var grg2 = gate_residual_backward(
        d_out[], sv.attn_res[], _t(mv.gate2.copy(), [D], ctx), ff_down, ctx
    )
    var d_gate2 = grg2.d_g.to_host(ctx)
    # d_ff_down = grg2.d_y (grad at the ff_down OUTPUT, base+LoRA share it).
    var d_ff_down_h = grg2.d_y.to_host(ctx)

    # base ff_down = linear(ff_act, Wdn, bdn)
    var lb_dn = linear_backward(grg2.d_y, sv.ff_act[], w.wdn[], N, F, D, ctx)
    var d_wdn = lb_dn.d_w.to_host(ctx)
    var d_bdn = lb_dn.d_b.to_host(ctx)
    var d_ff_act_h = lb_dn.d_x.to_host(ctx)
    # LoRA ff_down: d_A/d_B + fold d_x_lo into d_ff_act
    var ffd_da = _empty()
    var ffd_db = _empty()
    if slo.ff_down:
        var lg = klein_lora_bwd(d_ff_down_h, ff_act_h, slo.ff_down.value(), N, ctx)
        d_ff_act_h = _add_lists(d_ff_act_h, lg.d_x)
        ffd_da = lg.d_a.copy()
        ffd_db = lg.d_b.copy()
    var d_ff_act = _t(d_ff_act_h.copy(), [N, F], ctx)

    # ff_act = gelu(ff_up)
    var d_ff_up = gelu_backward(d_ff_act, sv.ff_up[], ctx)
    var d_ff_up_h = d_ff_up.to_host(ctx)

    # base ff_up = linear(ff_in, Wup, bup)
    var lb_up = linear_backward(d_ff_up, sv.ff_in[], w.wup[], N, D, F, ctx)
    var d_wup = lb_up.d_w.to_host(ctx)
    var d_bup = lb_up.d_b.to_host(ctx)
    var d_ff_in_h = lb_up.d_x.to_host(ctx)
    var ff_in_h = sv.ff_in[].to_host(ctx)
    var ffu_da = _empty()
    var ffu_db = _empty()
    if slo.ff_up:
        var lg = klein_lora_bwd(d_ff_up_h, ff_in_h, slo.ff_up.value(), N, ctx)
        d_ff_in_h = _add_lists(d_ff_in_h, lg.d_x)
        ffu_da = lg.d_a.copy()
        ffu_db = lg.d_b.copy()
    var d_ff_in = _t(d_ff_in_h.copy(), [N, D], ctx)

    # ff_in = modulate(ln2, scale2, shift2)
    var mb2 = modulate_backward(d_ff_in, sv.ln2[], _t(mv.scale2.copy(), [D], ctx), ctx)
    var d_scale2 = mb2.d_scale.to_host(ctx)
    var d_shift2 = mb2.d_shift.to_host(ctx)

    var lnb2 = layer_norm_backward(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    var d_attn_res_total = TArc(add(grg2.d_x, lnb2.d_x, ctx))

    # attn_res = residual_gate(x, gate1, out); recompute base+LoRA out
    var att_h = att[].to_host(ctx)
    var out_base = linear(att[], w.wout[], Optional[Tensor](_clone_t(w.bout[], ctx)), ctx)
    var out_proj = _add_lora_delta(out_base, att_h, slo.out, N, ctx)
    var grg1 = gate_residual_backward(
        d_attn_res_total[], x[], _t(mv.gate1.copy(), [D], ctx), out_proj, ctx
    )
    var d_gate1 = grg1.d_g.to_host(ctx)
    var d_x_res = grg1.d_x.to_host(ctx)
    var d_out_h = grg1.d_y.to_host(ctx)

    # base out = linear(att, Wout, bout)
    var lb_out = linear_backward(grg1.d_y, att[], w.wout[], N, D, D, ctx)
    var d_wout = lb_out.d_w.to_host(ctx)
    var d_bout = lb_out.d_b.to_host(ctx)
    var d_att_h = lb_out.d_x.to_host(ctx)
    var out_da = _empty()
    var out_db = _empty()
    if slo.out:
        var lg = klein_lora_bwd(d_out_h, att_h, slo.out.value(), N, ctx)
        d_att_h = _add_lists(d_att_h, lg.d_x)
        out_da = lg.d_a.copy()
        out_db = lg.d_b.copy()

    var base = _StreamPostBack(
        d_x_res^, d_att_h^, d_wout^, d_bout^, d_wup^, d_bup^, d_wdn^, d_bdn^,
        d_gate1=d_gate1^,
        d_shift2=d_shift2^, d_scale2=d_scale2^, d_gate2=d_gate2^,
    )
    return _StreamPostBackLora(
        base^, out_da^, out_db^, ffu_da^, ffu_db^, ffd_da^, ffd_db^,
    )


# ── per-stream LoRA PRE-attention backward (base grads + q/k/v LoRA d_A/d_B) ──
struct _StreamPreBackLora(Copyable, Movable):
    var base: _StreamPreBack
    var q_d_a: List[Float32]
    var q_d_b: List[Float32]
    var k_d_a: List[Float32]
    var k_d_b: List[Float32]
    var v_d_a: List[Float32]
    var v_d_b: List[Float32]

    def __init__(
        out self, var base: _StreamPreBack,
        var q_d_a: List[Float32], var q_d_b: List[Float32],
        var k_d_a: List[Float32], var k_d_b: List[Float32],
        var v_d_a: List[Float32], var v_d_b: List[Float32],
    ):
        self.base = base^
        self.q_d_a = q_d_a^
        self.q_d_b = q_d_b^
        self.k_d_a = k_d_a^
        self.k_d_b = k_d_b^
        self.v_d_a = v_d_a^
        self.v_d_b = v_d_b^


def _stream_pre_backward_lora[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecs, slo: StreamLora, sv: StreamSaved,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPreBackLora:
    var rb_q = rms_norm_backward(d_q_rms, sv.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, sv.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [N, D])
    reshape_in_place(rb_k.d_x, [N, D])
    var d_v_flat = reshape(d_v, [N, D], ctx)

    # grads at the q/k/v projection OUTPUTS (base + LoRA share them).
    var d_q_out_h = rb_q.d_x.to_host(ctx)
    var d_k_out_h = rb_k.d_x.to_host(ctx)
    var d_v_out_h = d_v_flat.to_host(ctx)
    var normed_h = sv.normed[].to_host(ctx)

    var lb_q = linear_backward(rb_q.d_x, sv.normed[], w.wq[], N, D, D, ctx)
    var lb_k = linear_backward(rb_k.d_x, sv.normed[], w.wk[], N, D, D, ctx)
    var lb_v = linear_backward(d_v_flat, sv.normed[], w.wv[], N, D, D, ctx)
    var d_wq = lb_q.d_w.to_host(ctx)
    var d_wk = lb_k.d_w.to_host(ctx)
    var d_wv = lb_v.d_w.to_host(ctx)
    var d_bq = lb_q.d_b.to_host(ctx)
    var d_bk = lb_k.d_b.to_host(ctx)
    var d_bv = lb_v.d_b.to_host(ctx)

    var d_nq_h = lb_q.d_x.to_host(ctx)
    var d_nk_h = lb_k.d_x.to_host(ctx)
    var d_nv_h = lb_v.d_x.to_host(ctx)

    var q_da = _empty()
    var q_db = _empty()
    if slo.q:
        var lg = klein_lora_bwd(d_q_out_h, normed_h, slo.q.value(), N, ctx)
        d_nq_h = _add_lists(d_nq_h, lg.d_x)
        q_da = lg.d_a.copy()
        q_db = lg.d_b.copy()
    var k_da = _empty()
    var k_db = _empty()
    if slo.k:
        var lg = klein_lora_bwd(d_k_out_h, normed_h, slo.k.value(), N, ctx)
        d_nk_h = _add_lists(d_nk_h, lg.d_x)
        k_da = lg.d_a.copy()
        k_db = lg.d_b.copy()
    var v_da = _empty()
    var v_db = _empty()
    if slo.v:
        var lg = klein_lora_bwd(d_v_out_h, normed_h, slo.v.value(), N, ctx)
        d_nv_h = _add_lists(d_nv_h, lg.d_x)
        v_da = lg.d_a.copy()
        v_db = lg.d_b.copy()

    # normed feeds all three q/k/v -> sum the three input grads (host).
    var d_normed_h = _add_lists(_add_lists(d_nq_h, d_nk_h), d_nv_h)
    var d_normed = _t(d_normed_h.copy(), [N, D], ctx)

    var mb1 = modulate_backward(d_normed, sv.ln1[], _t(mv.scale1.copy(), [D], ctx), ctx)
    var d_scale1 = mb1.d_scale.to_host(ctx)
    var d_shift1 = mb1.d_shift.to_host(ctx)

    var lnb1 = layer_norm_backward(mb1.d_x, sv.x[], ones, eps, ctx)
    var d_x_norm = lnb1.d_x.to_host(ctx)
    var base = _StreamPreBack(
        d_x_norm^, d_wq^, d_wk^, d_wv^, d_bq^, d_bk^, d_bv^,
        d_q_norm^, d_k_norm^, d_shift1^, d_scale1^,
    )
    return _StreamPreBackLora(base^, q_da^, q_db^, k_da^, k_db^, v_da^, v_db^)


# ── BACKWARD of one Qwen-Image double block WITH LoRA ─────────────────────────
def double_block_lora_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_img_out: List[Float32], d_txt_out: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs,
    lora: DoubleBlockLora, saved: DoubleBlockSaved,
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

    var d_tatt_4d = _t(tpb.base.d_att.copy(), [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = _t(ipb.base.d_att.copy(), [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)

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

    var img_base = StreamGrads(
        d_img_x^,
        iprb.base.d_wq.copy(), iprb.base.d_wk.copy(), iprb.base.d_wv.copy(),
        iprb.base.d_bq.copy(), iprb.base.d_bk.copy(), iprb.base.d_bv.copy(),
        ipb.base.d_wout.copy(), ipb.base.d_bout.copy(),
        ipb.base.d_wup.copy(), ipb.base.d_bup.copy(), ipb.base.d_wdn.copy(), ipb.base.d_bdn.copy(),
        iprb.base.d_q_norm.copy(), iprb.base.d_k_norm.copy(),
        iprb.base.d_shift1.copy(), iprb.base.d_scale1.copy(), ipb.base.d_gate1.copy(),
        ipb.base.d_shift2.copy(), ipb.base.d_scale2.copy(), ipb.base.d_gate2.copy(),
    )
    var txt_base = StreamGrads(
        d_txt_x^,
        tprb.base.d_wq.copy(), tprb.base.d_wk.copy(), tprb.base.d_wv.copy(),
        tprb.base.d_bq.copy(), tprb.base.d_bk.copy(), tprb.base.d_bv.copy(),
        tpb.base.d_wout.copy(), tpb.base.d_bout.copy(),
        tpb.base.d_wup.copy(), tpb.base.d_bup.copy(), tpb.base.d_wdn.copy(), tpb.base.d_bdn.copy(),
        tprb.base.d_q_norm.copy(), tprb.base.d_k_norm.copy(),
        tprb.base.d_shift1.copy(), tprb.base.d_scale1.copy(), tpb.base.d_gate1.copy(),
        tpb.base.d_shift2.copy(), tpb.base.d_scale2.copy(), tpb.base.d_gate2.copy(),
    )
    var base = DoubleBlockGrads(img_base^, txt_base^)

    var img_lora = StreamLoraGrads(
        iprb.q_d_a.copy(), iprb.q_d_b.copy(), iprb.k_d_a.copy(), iprb.k_d_b.copy(),
        iprb.v_d_a.copy(), iprb.v_d_b.copy(), ipb.out_d_a.copy(), ipb.out_d_b.copy(),
        ipb.ff_up_d_a.copy(), ipb.ff_up_d_b.copy(), ipb.ff_down_d_a.copy(), ipb.ff_down_d_b.copy(),
    )
    var txt_lora = StreamLoraGrads(
        tprb.q_d_a.copy(), tprb.q_d_b.copy(), tprb.k_d_a.copy(), tprb.k_d_b.copy(),
        tprb.v_d_a.copy(), tprb.v_d_b.copy(), tpb.out_d_a.copy(), tpb.out_d_b.copy(),
        tpb.ff_up_d_a.copy(), tpb.ff_up_d_b.copy(), tpb.ff_down_d_a.copy(), tpb.ff_down_d_b.copy(),
    )
    return DoubleBlockLoraGrads(base^, img_lora^, txt_lora^)
