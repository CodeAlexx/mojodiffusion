# serenitymojo/models/klein/img_in_ref.mojo
#
# Klein image-EDIT reference-conditioning compute unit — a PARALLEL ADDITIVE
# PROJECTION on the input seam. The sequence-invariant alternative to token-concat
# editing.
#
# THE MATH (mirrors klein_stack.mojo:327 `img = _linear_fwd_wdev(img_tokens, img_in, ...)`):
#     img = linear(noise_tokens, img_in) + linear(ref_tokens, img_in_ref)
#   where
#     noise_tokens : [N_IMG, in_ch]   the target latent, VAE-encoded + packed
#     img_in       : [D, in_ch]       FROZEN base input projection (klein_stack.mojo:184)
#     ref_tokens   : [N_IMG, in_ch]   the reference image, VAE-encoded + packed
#                                       (SAME shape as noise_tokens)
#     img_in_ref   : [D, in_ch]       NEW zero-init TRAINABLE projection — the only
#                                       new trained parameter here
#
# ZERO-INIT / FLAGS-OFF CONTRACT (hard requirement):
#   img_in_ref is zero-initialized -> linear(ref_tokens, 0) == 0 -> the summed
#   projection is BYTE-IDENTICAL to today's text-to-image path
#   img = linear(noise_tokens, img_in). Proven by the parity gate's
#   `img_in_ref = 0` identity check (max_abs 0).
#
# BACKWARD is a plain linear backward on the ref branch (ops/linalg_backward.mojo
# `linear_backward`, this is the SAME helper klein_stack.mojo:502 uses for the
# frozen img_in path):
#     y[M,out] = x[M,in] @ W[out,in]ᵀ   with  M=N_IMG, in=in_ch, out=D
#     d_ref_tokens   = d_img_act @ img_in_ref            [N_IMG, in_ch]   (linear_backward.d_x)
#     d_img_in_ref_w = d_img_actᵀ @ ref_tokens           [D, in_ch]       (linear_backward.d_w)
#   img_in_ref_w is the ONLY new trained parameter -> d_img_in_ref_w is the
#   load-bearing grad. d_ref_tokens is also computed so the unit is self-contained
#   and testable. The noise branch (d_noise_tokens / d_img_in) is the existing
#   FROZEN path already covered by klein_stack — NOT recomputed here.
#
# API CROSSING: host List[Float32] per the stack's conventions (klein_stack.mojo);
# the forward reuses the SAME `_linear_fwd` + `_add_lists` klein_stack uses so the
# summed projection matches the stack numerics exactly. Multi-return via a Movable
# struct (NOT a tuple).
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor move-only; host List[Float32] carriers.

from max.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.tensor import Tensor
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
# Reuse the EXACT host helpers klein_stack uses (identical numerics on the sum).
from serenitymojo.models.klein.klein_stack import _linear_fwd, _add_lists, _t


# ── forward result: the summed image activation + the ref acts saved for bwd ──
# For a plain linear backward the only activation the ref branch needs is
# ref_tokens itself (d_w = d_yᵀ @ ref_tokens); img_in_ref_w is passed to the
# backward directly (a frozen-shape param handle). We SAVE ref_tokens so the
# backward is self-contained from `saved` alone.
struct ImgInRefForward(Movable):
    var img_act: List[Float32]      # [N_IMG, D]   summed projection
    var ref_tokens: List[Float32]   # [N_IMG, in_ch]  saved for d_img_in_ref_w

    def __init__(
        out self, var img_act: List[Float32], var ref_tokens: List[Float32]
    ):
        self.img_act = img_act^
        self.ref_tokens = ref_tokens^


# ── backward result: grad of the NEW trained param + grad of the ref tokens ──
struct ImgInRefGrads(Movable):
    var d_img_in_ref_w: List[Float32]   # [D, in_ch]      the load-bearing grad
    var d_ref_tokens: List[Float32]     # [N_IMG, in_ch]  self-contained extra

    def __init__(
        out self,
        var d_img_in_ref_w: List[Float32], var d_ref_tokens: List[Float32],
    ):
        self.d_img_in_ref_w = d_img_in_ref_w^
        self.d_ref_tokens = d_ref_tokens^


# ── FORWARD ───────────────────────────────────────────────────────────────────
#   img_act = linear(noise_tokens, img_in_w) + linear(ref_tokens, img_in_ref_w)
# noise_tokens/ref_tokens: [N_IMG, in_ch] ; img_in_w/img_in_ref_w: [D, in_ch].
# Reuses `_linear_fwd` (klein_stack) for BOTH projections and `_add_lists` for the
# sum -> byte-matches the stack's input-projection numerics; adding the
# zero-projection when img_in_ref_w == 0 leaves img_act byte-identical.
def img_in_ref_forward(
    noise_tokens: List[Float32], img_in_w: List[Float32],
    ref_tokens: List[Float32], img_in_ref_w: List[Float32],
    N_IMG: Int, in_ch: Int, D: Int, ctx: DeviceContext,
) raises -> ImgInRefForward:
    var noise_proj = _linear_fwd(noise_tokens, img_in_w, N_IMG, in_ch, D, ctx)
    var ref_proj = _linear_fwd(ref_tokens, img_in_ref_w, N_IMG, in_ch, D, ctx)
    var img_act = _add_lists(noise_proj, ref_proj)
    return ImgInRefForward(img_act^, ref_tokens.copy())


# ── BACKWARD ──────────────────────────────────────────────────────────────────
# d_img_act: upstream grad of img_act [N_IMG, D].
# Returns d_img_in_ref_w [D, in_ch] (the only new trained param) and d_ref_tokens
# [N_IMG, in_ch]. Both come from a SINGLE linear_backward on the ref branch:
#   linear_backward(grad_y=d_img_act, x=ref_tokens, weight=img_in_ref_w,
#                   M=N_IMG, in_features=in_ch, out_features=D)
#     -> d_x = d_ref_tokens ; d_w = d_img_in_ref_w.
def img_in_ref_backward(
    d_img_act: List[Float32], saved: ImgInRefForward, img_in_ref_w: List[Float32],
    N_IMG: Int, in_ch: Int, D: Int, ctx: DeviceContext,
) raises -> ImgInRefGrads:
    var lg = linear_backward(
        _t(d_img_act, [N_IMG, D], ctx),
        _t(saved.ref_tokens, [N_IMG, in_ch], ctx),
        _t(img_in_ref_w, [D, in_ch], ctx),
        N_IMG, in_ch, D, ctx,
    )
    var d_img_in_ref_w = lg.d_w.to_host(ctx)
    var d_ref_tokens = lg.d_x.to_host(ctx)
    return ImgInRefGrads(d_img_in_ref_w^, d_ref_tokens^)
