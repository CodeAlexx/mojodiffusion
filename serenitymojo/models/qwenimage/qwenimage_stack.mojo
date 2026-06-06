# serenitymojo/models/qwenimage/qwenimage_stack.mojo
#
# FULL Qwen-Image MMDiT stack: composes N (=60) parity-verified double-stream
# blocks (qwenimage_block.mojo) into the complete model with per-block RECOMPUTE
# checkpointing (the §6.2 checkpoint_block idea applied per-block, so real depth
# fits 24 GB without OOM). Mirrors models/klein/klein_stack.mojo (the proven
# stack composition), specialized to Qwen-Image:
#
#   - ALL double-stream (num_double=60, num_single=0): no single blocks.
#   - Modulation is PER-BLOCK (each block has its own img_mod.1 / txt_mod.1 MLP),
#     NOT shared. The stack computes each block's ModVecs from the shared temb via
#     that block's (frozen) mod-MLP weights. The per-block modvec grads come back
#     from the block backward; they are NOT backpropped into the mod-MLP (frozen,
#     not a LoRA target — the deferred finetune link, same contract as Klein).
#   - Final layer = norm_out (AdaLayerNormContinuous): scale/shift come from a
#     SHARED final mod vector; out = linear(modulate(layer_norm(img), scale, shift),
#     proj_out). (The caller precomputes final scale/shift from temb.)
#
# INTER-BLOCK HANDOFF: d_x(block N) = d_y(block N+1) — each double_block_backward
# returns (d_img_x, d_txt_x) fed as the previous block's (d_img_out, d_txt_out).
# TXT-FIRST throughout. Per-block recompute: the forward retains ONLY each block's
# (img,txt) INPUT; the backward RE-RUNS that block's forward to regenerate saved,
# then runs the verified per-block backward.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import layer_norm_backward
from serenitymojo.ops.elementwise_backward import modulate_backward

from serenitymojo.models.qwenimage.qwenimage_block import (
    DoubleBlockWeights, ModVecs, DoubleBlockSaved, DoubleBlockGrads, StreamGrads,
    double_block_forward, double_block_backward,
)


comptime TArc = ArcPointer[Tensor]


def _zeros(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(0.0)
    return o^


def _ones(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(1.0)
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


# biased linear forward (host list in/out): y = x @ W.T + b
def _linear_b(
    x_h: List[Float32], w: Tensor, b: Tensor, M: Int, in_f: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var bclone = ctx.enqueue_create_buffer[DType.uint8](b.nbytes())
    ctx.enqueue_copy(dst_buf=bclone, src_buf=b.buf)
    ctx.synchronize()
    var bt = Tensor(bclone^, b.shape(), b.dtype())
    return linear(
        Tensor.from_host(x_h.copy(), [M, in_f], STDtype.BF16, ctx),
        w, Optional[Tensor](bt^), ctx,
    ).to_host(ctx)


# ── stack base (frozen non-block weights) ────────────────────────────────────
struct QwenStackBase(Copyable, Movable):
    var img_in_w: TArc     # [D, in_ch]    img_in.weight
    var img_in_b: TArc     # [D]           img_in.bias
    var txt_in_w: TArc     # [D, txt_ch]   txt_in.weight
    var txt_in_b: TArc     # [D]           txt_in.bias
    var proj_out_w: TArc   # [out_ch, D]   proj_out.weight
    var proj_out_b: TArc   # [out_ch]      proj_out.bias

    def __init__(
        out self,
        var img_in_w: List[Float32], var img_in_b: List[Float32],
        var txt_in_w: List[Float32], var txt_in_b: List[Float32],
        var proj_out_w: List[Float32], var proj_out_b: List[Float32],
        D: Int, in_ch: Int, txt_ch: Int, out_ch: Int, ctx: DeviceContext,
    ) raises:
        self.img_in_w = TArc(Tensor.from_host(img_in_w^, [D, in_ch], STDtype.BF16, ctx))
        self.img_in_b = TArc(Tensor.from_host(img_in_b^, [D], STDtype.BF16, ctx))
        self.txt_in_w = TArc(Tensor.from_host(txt_in_w^, [D, txt_ch], STDtype.BF16, ctx))
        self.txt_in_b = TArc(Tensor.from_host(txt_in_b^, [D], STDtype.BF16, ctx))
        self.proj_out_w = TArc(Tensor.from_host(proj_out_w^, [out_ch, D], STDtype.BF16, ctx))
        self.proj_out_b = TArc(Tensor.from_host(proj_out_b^, [out_ch], STDtype.BF16, ctx))

    def __init__(
        out self,
        var img_in_w: TArc, var img_in_b: TArc,
        var txt_in_w: TArc, var txt_in_b: TArc,
        var proj_out_w: TArc, var proj_out_b: TArc,
    ):
        self.img_in_w = img_in_w^
        self.img_in_b = img_in_b^
        self.txt_in_w = txt_in_w^
        self.txt_in_b = txt_in_b^
        self.proj_out_w = proj_out_w^
        self.proj_out_b = proj_out_b^


# ── forward result (CHECKPOINT INPUTS ONLY — true per-block recompute) ────────
# To bound memory at real depth (60 blocks x F=12288) we retain ONLY each block's
# (img,txt) INPUT — host lists, cheap — NOT the full per-block saved activations.
# The backward RE-RUNS each block's forward from its saved input to regenerate
# `saved`, then runs that block's verified backward. Peak memory = one block's
# activation footprint + the resident inter-block stream tensors.
struct QwenStackForward(Copyable, Movable):
    var out: List[Float32]              # [N_IMG, out_ch]
    var dbl_img_in: List[List[Float32]] # num_double x [N_IMG,D]  (checkpoint inputs)
    var dbl_txt_in: List[List[Float32]] # num_double x [N_TXT,D]
    var img_out: TArc                   # [N_IMG, D]  (last block img output)
    var ln_img_out: TArc                # [N_IMG, D]  layer_norm(img_out)

    def __init__(
        out self, var out: List[Float32],
        var dbl_img_in: List[List[Float32]], var dbl_txt_in: List[List[Float32]],
        var img_out: TArc, var ln_img_out: TArc,
    ):
        self.out = out^
        self.dbl_img_in = dbl_img_in^
        self.dbl_txt_in = dbl_txt_in^
        self.img_out = img_out^
        self.ln_img_out = ln_img_out^


# ── backward result ──────────────────────────────────────────────────────────
struct QwenStackGrads(Copyable, Movable):
    var d_img_tokens: List[Float32]   # [N_IMG, in_ch]
    var d_txt_tokens: List[Float32]   # [N_TXT, txt_ch]
    var dbl_grads: List[DoubleBlockGrads]
    var d_proj_out_w: List[Float32]
    var d_proj_out_b: List[Float32]
    var d_final_scale: List[Float32]  # [D]
    var d_final_shift: List[Float32]  # [D]

    def __init__(
        out self,
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        var dbl_grads: List[DoubleBlockGrads],
        var d_proj_out_w: List[Float32], var d_proj_out_b: List[Float32],
        var d_final_scale: List[Float32], var d_final_shift: List[Float32],
    ):
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.dbl_grads = dbl_grads^
        self.d_proj_out_w = d_proj_out_w^
        self.d_proj_out_b = d_proj_out_b^
        self.d_final_scale = d_final_scale^
        self.d_final_shift = d_final_shift^


# ── FULL FORWARD (per-block recompute: only block inputs retained) ───────────
# img_tokens [N_IMG,in_ch], txt_tokens [N_TXT,txt_ch].
# dbw: num_double DoubleBlockWeights. img_mods/txt_mods: PER-BLOCK ModVecs.
# final_scale/final_shift [D]: the norm_out AdaLN vectors (precomputed from temb).
def qwenimage_stack_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: QwenStackBase,
    dbw: List[DoubleBlockWeights],
    img_mods: List[ModVecs], txt_mods: List[ModVecs],
    final_scale: List[Float32], final_shift: List[Float32],
    cos: List[Float32], sin: List[Float32],
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> QwenStackForward:
    var num_double = len(dbw)
    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    var img = _linear_b(img_tokens, base.img_in_w[], base.img_in_b[], N_IMG, in_ch, ctx)
    var txt = _linear_b(txt_tokens, base.txt_in_w[], base.txt_in_b[], N_TXT, txt_ch, ctx)

    var dbl_img_in = List[List[Float32]]()
    var dbl_txt_in = List[List[Float32]]()
    for bi in range(num_double):
        dbl_img_in.append(img.copy())
        dbl_txt_in.append(txt.copy())
        # forward; discard the saved activations (recomputed in backward).
        var fwd = double_block_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), dbw[bi], img_mods[bi], txt_mods[bi],
            cos_t, sin_t, D, F, eps, ctx,
        )
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()

    # final layer (norm_out + proj_out) on img only
    var ln_img_out = layer_norm(
        _t(img.copy(), [N_IMG, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), eps, ctx,
    ).to_host(ctx)
    var normed = modulate(
        _t(ln_img_out.copy(), [N_IMG, D], ctx),
        _t(final_scale.copy(), [D], ctx), _t(final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var out = _linear_b(normed, base.proj_out_w[], base.proj_out_b[], N_IMG, D, ctx)

    return QwenStackForward(
        out^, dbl_img_in^, dbl_txt_in^,
        TArc(_t(img^, [N_IMG, D], ctx)), TArc(_t(ln_img_out^, [N_IMG, D], ctx)),
    )


# ── FULL BACKWARD (full-depth; per-block recompute) ──────────────────────────
def qwenimage_stack_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: QwenStackBase,
    dbw: List[DoubleBlockWeights],
    img_mods: List[ModVecs], txt_mods: List[ModVecs],
    final_scale: List[Float32], final_shift: List[Float32],
    cos: List[Float32], sin: List[Float32],
    saved: QwenStackForward,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> QwenStackGrads:
    var num_double = len(dbw)
    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    # ── final layer backward ──
    var normed = modulate(
        saved.ln_img_out[],
        _t(final_scale.copy(), [D], ctx), _t(final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var lbf = linear_backward(
        _t(d_out, [N_IMG, out_ch], ctx), _t(normed, [N_IMG, D], ctx),
        base.proj_out_w[], N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)
    var d_proj_out_w = lbf.d_w.to_host(ctx)
    var d_proj_out_b = lbf.d_b.to_host(ctx)

    var mbf = modulate_backward(
        _t(d_normed, [N_IMG, D], ctx), saved.ln_img_out[],
        _t(final_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)
    var d_final_scale = mbf.d_scale.to_host(ctx)
    var d_final_shift = mbf.d_shift.to_host(ctx)

    var lnbf = layer_norm_backward(
        _t(d_ln_img_out, [N_IMG, D], ctx), saved.img_out[],
        _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_img_out = lnbf.d_x.to_host(ctx)   # [N_IMG,D] = grad of last block img output

    # ── double-stream backward (REVERSE; PER-BLOCK RECOMPUTE) ──
    # For each block (deepest first) RE-RUN its forward from the saved input to
    # regenerate `saved`, then run the verified per-block backward. Peak memory
    # stays at ~one block's activations -> fits real depth in 24 GB.
    var dbl_grads_rev = List[DoubleBlockGrads]()
    var di = num_double - 1
    var d_io = d_img_out.copy()
    var d_to = _zeros(N_TXT * D)   # txt output of last block not read by final layer
    while di >= 0:
        var rf = double_block_forward[H, Dh, N_IMG, N_TXT, S](
            saved.dbl_img_in[di].copy(), saved.dbl_txt_in[di].copy(),
            dbw[di], img_mods[di], txt_mods[di], cos_t, sin_t, D, F, eps, ctx,
        )
        var bg = double_block_backward[H, Dh, N_IMG, N_TXT, S](
            d_io.copy(), d_to.copy(), dbw[di], img_mods[di], txt_mods[di],
            rf.saved.copy(), cos_t, sin_t, D, F, eps, ctx,
        )
        d_io = bg.img.d_x.copy()
        d_to = bg.txt.d_x.copy()
        dbl_grads_rev.append(bg^)
        di -= 1
    var dbl_grads = List[DoubleBlockGrads]()
    var k = len(dbl_grads_rev) - 1
    while k >= 0:
        dbl_grads.append(dbl_grads_rev[k].copy())
        k -= 1

    # ── input-projection backward ──
    var lbi = linear_backward(
        _t(d_io, [N_IMG, D], ctx), _t(img_tokens, [N_IMG, in_ch], ctx),
        base.img_in_w[], N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)

    var lbt = linear_backward(
        _t(d_to, [N_TXT, D], ctx), _t(txt_tokens, [N_TXT, txt_ch], ctx),
        base.txt_in_w[], N_TXT, txt_ch, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)

    return QwenStackGrads(
        d_img_tokens^, d_txt_tokens^, dbl_grads^,
        d_proj_out_w^, d_proj_out_b^, d_final_scale^, d_final_shift^,
    )
