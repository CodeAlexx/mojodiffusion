# models/dit/ltx2_nag.mojo — Normalized Attention Guidance (NAG) for LTX-2.
#
# Pure-Mojo port of /home/alex/ltx2-app/archive_pre_lightricks_20260411/nag.py.
# NAG applies CFG-style guidance in cross-attention OUTPUT space using a NULL
# text encoding as the negative baseline — so NO real negative prompt is needed.
#
# Python reference (_nag_combine, nag.py:21-46), per the authoritative spec:
#   guidance = x_pos * scale - x_neg * (scale - 1)
#   norm_pos  = ||x_pos||_1   over last dim (keepdim)         (torch.norm p=1)
#   norm_guid = ||guidance||_1 over last dim (keepdim)
#   ratio     = norm_guid / (norm_pos + 1e-7)
#   if ratio > tau:  adjustment = (norm_pos * tau) / (norm_guid + 1e-7)
#                    guidance   = guidance * adjustment        (row-broadcast)
#   output = guidance * alpha + x_pos * (1 - alpha)
# Defaults (all community workflows): scale=11, alpha=0.25, tau=2.5.
#
# The patch (NAGPatch.apply, nag.py:94-119) wraps attn2 / audio_attn2 forward:
#   out_pos = attn2(x, real_context, ...)        # positive prompt KV
#   out_neg = attn2(x, null_context,  ...)        # NULL prompt KV  (no mask)
#   return _nag_combine(out_pos, out_neg, scale, alpha, tau)
#
# In serenitymojo the AV block (ltx2_dit.mojo ltx2_block_forward_av) computes the
# cross-attn output via `_av_attention(..., "attn2", mod_q, mv_ctx, ...)`. NAG is
# wired by computing a SECOND `_av_attention` call with the null-context KV and
# feeding both outputs through `nag_combine`. This file provides:
#   * nag_combine(pos, neg, scale, alpha, tau, ctx)  — the fused L1-clip blend.
#   * NAGContext — carries null video/audio KV-context + (scale, alpha, tau) and
#     the predicate `enabled` so the block can branch the combine in/out without
#     touching its non-NAG path.
#
# *** CODE-ONLY: compile-targeted; the combine has a unit gate in
#     serenitymojo/models/dit/parity/ltx2_nag_parity.mojo (cos>=0.999 vs nag.py).
#
# Mojo 1.0.0b1, NVIDIA GPU. F32 math in the combine kernel (matches the Python
# torch.float32 reference); storage dtype preserved on output.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from std.math import sqrt
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── NAG combine: one thread per (row) over the leading dims; the thread loops
# the last `dim` to accumulate both L1 norms in F32, then writes the blended
# row. Matches nag.py exactly: guidance, conditional rescale, alpha-blend.
#
# Layout: pos/neg are [.., dim] contiguous (row-major). `rows = numel/dim`.
# Each row r occupies pos[r*dim : (r+1)*dim].
def _nag_combine_kernel_f32(
    pos: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    neg: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int,
    dim: Int,
    scale: Float32,
    alpha: Float32,
    tau: Float32,
):
    var r = Int(global_idx.x)
    if r >= rows:
        return
    var base = r * dim
    var sm1 = scale - Float32(1.0)
    # Pass 1: L1 norms of pos and guidance.
    var norm_pos = Float32(0.0)
    var norm_guid = Float32(0.0)
    for j in range(dim):
        var p = rebind[Scalar[DType.float32]](pos[base + j])
        var ng = rebind[Scalar[DType.float32]](neg[base + j])
        var g = p * scale - ng * sm1
        norm_pos += abs(p)
        norm_guid += abs(g)
    # Conditional rescale of guidance (row-uniform factor).
    var ratio = norm_guid / (norm_pos + Float32(1e-7))
    var factor = Float32(1.0)
    if ratio > tau:
        factor = (norm_pos * tau) / (norm_guid + Float32(1e-7))
    # Pass 2: blended output = guidance*factor*alpha + pos*(1-alpha).
    var one_minus_a = Float32(1.0) - alpha
    for j in range(dim):
        var p = rebind[Scalar[DType.float32]](pos[base + j])
        var ng = rebind[Scalar[DType.float32]](neg[base + j])
        var g = (p * scale - ng * sm1) * factor
        o[base + j] = rebind[o.element_type](g * alpha + p * one_minus_a)


def _nag_combine_kernel_bf16(
    pos: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    neg: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    rows: Int,
    dim: Int,
    scale: Float32,
    alpha: Float32,
    tau: Float32,
):
    var r = Int(global_idx.x)
    if r >= rows:
        return
    var base = r * dim
    var sm1 = scale - Float32(1.0)
    var norm_pos = Float32(0.0)
    var norm_guid = Float32(0.0)
    for j in range(dim):
        var p = rebind[Scalar[DType.bfloat16]](pos[base + j]).cast[DType.float32]()
        var ng = rebind[Scalar[DType.bfloat16]](neg[base + j]).cast[DType.float32]()
        var g = p * scale - ng * sm1
        norm_pos += abs(p)
        norm_guid += abs(g)
    var ratio = norm_guid / (norm_pos + Float32(1e-7))
    var factor = Float32(1.0)
    if ratio > tau:
        factor = (norm_pos * tau) / (norm_guid + Float32(1e-7))
    var one_minus_a = Float32(1.0) - alpha
    for j in range(dim):
        var p = rebind[Scalar[DType.bfloat16]](pos[base + j]).cast[DType.float32]()
        var ng = rebind[Scalar[DType.bfloat16]](neg[base + j]).cast[DType.float32]()
        var g = (p * scale - ng * sm1) * factor
        var outv = g * alpha + p * one_minus_a
        o[base + j] = rebind[o.element_type](outv.cast[DType.bfloat16]())


def _nag_combine_kernel_f16(
    pos: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    neg: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    rows: Int,
    dim: Int,
    scale: Float32,
    alpha: Float32,
    tau: Float32,
):
    var r = Int(global_idx.x)
    if r >= rows:
        return
    var base = r * dim
    var sm1 = scale - Float32(1.0)
    var norm_pos = Float32(0.0)
    var norm_guid = Float32(0.0)
    for j in range(dim):
        var p = rebind[Scalar[DType.float16]](pos[base + j]).cast[DType.float32]()
        var ng = rebind[Scalar[DType.float16]](neg[base + j]).cast[DType.float32]()
        var g = p * scale - ng * sm1
        norm_pos += abs(p)
        norm_guid += abs(g)
    var ratio = norm_guid / (norm_pos + Float32(1e-7))
    var factor = Float32(1.0)
    if ratio > tau:
        factor = (norm_pos * tau) / (norm_guid + Float32(1e-7))
    var one_minus_a = Float32(1.0) - alpha
    for j in range(dim):
        var p = rebind[Scalar[DType.float16]](pos[base + j]).cast[DType.float32]()
        var ng = rebind[Scalar[DType.float16]](neg[base + j]).cast[DType.float32]()
        var g = (p * scale - ng * sm1) * factor
        var outv = g * alpha + p * one_minus_a
        o[base + j] = rebind[o.element_type](outv.cast[DType.float16]())


# ── public: NAG combine ──────────────────────────────────────────────────────
# pos / neg: same shape [.., dim], same dtype. Output: same shape/dtype as pos.
# Implements nag.py:_nag_combine exactly with F32 accumulation of the L1 norms.
def nag_combine(
    pos: Tensor,
    neg: Tensor,
    scale: Float32,
    alpha: Float32,
    tau: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """CFG in attention-output space with L1-norm clipping and alpha blend.

    guidance = pos*scale - neg*(scale-1); if ||guidance||_1/||pos||_1 > tau,
    rescale guidance by (||pos||_1*tau)/||guidance||_1; out = guidance*alpha +
    pos*(1-alpha). L1 norms are over the last dim, F32-accumulated."""
    var ps = pos.shape()
    var ns = neg.shape()
    if len(ps) != len(ns):
        raise Error("nag_combine: rank mismatch between pos and neg")
    for i in range(len(ps)):
        if ps[i] != ns[i]:
            raise Error("nag_combine: shape mismatch between pos and neg")
    if pos.dtype() != neg.dtype():
        raise Error("nag_combine: dtype mismatch between pos and neg")

    var dim = ps[len(ps) - 1]
    var numel = pos.numel()
    if dim == 0:
        raise Error("nag_combine: zero last dim")
    var rows = numel // dim

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](pos.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](numel))
    var grid = (rows + _BLOCK - 1) // _BLOCK
    var dt = pos.dtype().to_mojo_dtype()

    if dt == DType.float32:
        var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            pos.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var N = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            neg.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_nag_combine_kernel_f32, _nag_combine_kernel_f32](
            P, N, O, rows, dim, scale, alpha, tau,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            pos.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var N = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            neg.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_nag_combine_kernel_bf16, _nag_combine_kernel_bf16](
            P, N, O, rows, dim, scale, alpha, tau,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var P = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            pos.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var N = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            neg.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_nag_combine_kernel_f16, _nag_combine_kernel_f16](
            P, N, O, rows, dim, scale, alpha, tau,
            grid_dim=grid, block_dim=_BLOCK,
        )
    ctx.synchronize()
    return Tensor(out_buf^, pos.shape(), pos.dtype())


# ── NAGContext: carries the null-text KV-context for video (+ optional audio)
# and the (scale, alpha, tau) hyperparameters. The DiT block consults `enabled`
# to decide whether to run the second (null) cross-attn and combine. ──────────
#
# Port of NAGPatch.__init__ (nag.py:52-65). `nag_v_ctx` / `nag_a_ctx` are the
# NULL text encoder hidden states (no real negative prompt). They are already
# KV-modulated-or-raw to match what `_av_attention` feeds as its KV input — i.e.
# the caller passes the SAME representation it would for the positive context
# (post `_kv_modulate` or raw clone), but built from the null encoding.
struct NAGContext(Movable):
    var enabled: Bool
    var has_audio: Bool
    var nag_v_ctx: Tensor      # null video text KV-context [1, N_TXT, 4096]
    var nag_a_ctx: Tensor      # null audio text KV-context [1, N_TXT, 2048]
    var scale: Float32
    var alpha: Float32
    var tau: Float32

    def __init__(
        out self,
        enabled: Bool,
        has_audio: Bool,
        var nag_v_ctx: Tensor,
        var nag_a_ctx: Tensor,
        scale: Float32,
        alpha: Float32,
        tau: Float32,
    ):
        self.enabled = enabled
        self.has_audio = has_audio
        self.nag_v_ctx = nag_v_ctx^
        self.nag_a_ctx = nag_a_ctx^
        self.scale = scale
        self.alpha = alpha
        self.tau = tau

    @staticmethod
    def defaults(
        var nag_v_ctx: Tensor,
        var nag_a_ctx: Tensor,
        has_audio: Bool,
    ) -> NAGContext:
        """Community-default NAG (scale=11, alpha=0.25, tau=2.5), enabled."""
        return NAGContext(
            True, has_audio, nag_v_ctx^, nag_a_ctx^,
            Float32(11.0), Float32(0.25), Float32(2.5),
        )

    @staticmethod
    def disabled(ctx: DeviceContext) raises -> NAGContext:
        """An off NAGContext with placeholder 1-element tensors (never read)."""
        var z = List[Float32]()
        z.append(Float32(0.0))
        var sh = List[Int]()
        sh.append(1)
        var t0 = Tensor.from_host(z, sh.copy(), STDtype.BF16, ctx)
        var t1 = Tensor.from_host(z, sh.copy(), STDtype.BF16, ctx)
        return NAGContext(
            False, False, t0^, t1^,
            Float32(11.0), Float32(0.25), Float32(2.5),
        )

    def combine_video(
        self, pos: Tensor, neg: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """nag_combine on the VIDEO attn2 output pair."""
        return nag_combine(pos, neg, self.scale, self.alpha, self.tau, ctx)

    def combine_audio(
        self, pos: Tensor, neg: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """nag_combine on the AUDIO audio_attn2 output pair."""
        return nag_combine(pos, neg, self.scale, self.alpha, self.tau, ctx)
