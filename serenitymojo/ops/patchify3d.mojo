# ops/patchify3d.mojo — 3D (video) DiT patch-embed unfold + its inverse.
#
# RANK-5 gap from ops/parity/OPS_GAP_AUDIT_2026-06-03.md: the video-DiT
# "Conv3d patch embed" (wan22, wan_vace, hunyuan15, cosmos, nava-video) is NOT a
# sliding conv — it is `kernel == stride == patch_size`, i.e. a per-(t,h,w) patch
# LINEAR projection over non-overlapping cubes. Mathematically:
#     unfold([C,F,H,W]) -> [n_patches, C*pt*ph*pw]   (this op, patchify3d)
#     tokens = unfold @ Wᵀ + b                        (caller: ops/linear.linear)
# where the conv kernel weight stored as [out, C, pt, ph, pw] flattens to
# [out, C*pt*ph*pw] with NO transpose — its C-major row-major memory order is
# exactly the within-patch flatten this op emits. EQUIVALENCE PROVEN (see audit
# + parity oracle: torch.conv3d(stride=kernel) == this unfold + matmul).
#
# REFERENCES (read line by line, Rust READ-ONLY):
#   wan22_dit.rs:667-703 `patchify`:
#     input  [C, F, H, W]  (no batch; C-major contiguous; pf,ph,pw = 1,2,2)
#     token order   patch_idx = fi*ho*wo + hi*wo + wi      (F-major, then H, W)
#     within-patch  dst_ch = ci*(pf*ph*pw) + pfi*(ph*pw) + phi*pw + pwi
#                   => order (c, pf, ph, pw)  with c SLOWEST, pw FASTEST.
#     out  [n_patches, C*pf*ph*pw].
#   wan22_dit.rs:705-745 `unpatchify` (einsum 'fhwpqrc->cfphqwr'):
#     input  [n_patches, C_out*pf*ph*pw], grid (fo,ho,wo)
#     within-patch READ order  src_ch = pfi*(ph*pw*c) + phi*(pw*c) + pwi*c + ci
#                   => order (pf, ph, pw, c)  with c FASTEST (DIFFERENT from
#                   patchify's c-slowest! the FinalLayer linear is trained to this
#                   layout, so unpatchify MUST mirror its own ref, not be a literal
#                   transpose-inverse of patchify). out [C_out, F, H, W].
#   cosmos_predict25_dit.rs:1544-1616: same structure ([B,C,T,H,W], patchify inner
#     "(c r m n)" c-slowest; unpatchify "(p1 p2 t' c)" c-fastest) — confirms the
#     scheme and the documented patchify/unpatchify asymmetry. We expose the wan22
#     single-sample [C,F,H,W] form (the primary ref); a [B,...] caller loops B or
#     pre-flattens, same as the rope_tables op leaves position-build to the caller.
#
# POSITIONAL EMBEDDING IS NOT FOLDED IN (caller responsibility, like rope_tables):
# cosmos's learnable abs-pos add and any sinusoidal pos-emb stay OUT of this op.
#
# Kernel style mirrors ops/layout.mojo (2D patchify): runtime _DYN1 layout, three
# dtype branches, one thread per OUTPUT element, pure index gather (cast-on-store,
# no reduction => no F32 accumulation needed; values relocate unchanged). The
# F32-accumulate happens in the downstream `linear` GEMM, not here.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ─────────────────────────────────────────────────────────────────────────────
# patchify3d. One thread per OUTPUT element. Output [n_patches, C*pf*ph*pw];
# decode the flat output index into (patch, f) where
#   patch = fi*HO*WO + hi*WO + wi   (F-major token order)
#   f     = ((ci*pf + pfi)*ph + phi)*pw + pwi   (c-slowest within-patch)
# then map to input offset in [C, F, H, W] (C-major contiguous):
#   src_f = fi*pf + pfi,  src_h = hi*ph + phi,  src_w = wi*pw + pwi
#   in_off = ((ci*F + src_f)*H + src_h)*W + src_w
# ─────────────────────────────────────────────────────────────────────────────
def _patchify3d_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    C: Int, F: Int, H: Int, W: Int,
    pf: Int, ph: Int, pw: Int,
    FO: Int, HO: Int, WO: Int,
):
    var idx = Int(global_idx.x)
    var PD = C * pf * ph * pw
    var L = FO * HO * WO
    var total = L * PD
    if idx < total:
        var f = idx % PD
        var patch = idx // PD
        var wi = patch % WO
        var rem = patch // WO
        var hi = rem % HO
        var fi = rem // HO
        var pwi = f % pw
        var t1 = f // pw
        var phi = t1 % ph
        var t2 = t1 // ph
        var pfi = t2 % pf
        var ci = t2 // pf
        var src_f = fi * pf + pfi
        var src_h = hi * ph + phi
        var src_w = wi * pw + pwi
        var in_off = ((ci * F + src_f) * H + src_h) * W + src_w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[in_off]))


def _patchify3d_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    C: Int, F: Int, H: Int, W: Int,
    pf: Int, ph: Int, pw: Int,
    FO: Int, HO: Int, WO: Int,
):
    var idx = Int(global_idx.x)
    var PD = C * pf * ph * pw
    var L = FO * HO * WO
    var total = L * PD
    if idx < total:
        var f = idx % PD
        var patch = idx // PD
        var wi = patch % WO
        var rem = patch // WO
        var hi = rem % HO
        var fi = rem // HO
        var pwi = f % pw
        var t1 = f // pw
        var phi = t1 % ph
        var t2 = t1 // ph
        var pfi = t2 % pf
        var ci = t2 // pf
        var src_f = fi * pf + pfi
        var src_h = hi * ph + phi
        var src_w = wi * pw + pwi
        var in_off = ((ci * F + src_f) * H + src_h) * W + src_w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[in_off]))


def _patchify3d_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    C: Int, F: Int, H: Int, W: Int,
    pf: Int, ph: Int, pw: Int,
    FO: Int, HO: Int, WO: Int,
):
    var idx = Int(global_idx.x)
    var PD = C * pf * ph * pw
    var L = FO * HO * WO
    var total = L * PD
    if idx < total:
        var f = idx % PD
        var patch = idx // PD
        var wi = patch % WO
        var rem = patch // WO
        var hi = rem % HO
        var fi = rem // HO
        var pwi = f % pw
        var t1 = f // pw
        var phi = t1 % ph
        var t2 = t1 // ph
        var pfi = t2 % pf
        var ci = t2 // pf
        var src_f = fi * pf + pfi
        var src_h = hi * ph + phi
        var src_w = wi * pw + pwi
        var in_off = ((ci * F + src_f) * H + src_h) * W + src_w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](x[in_off]))


def patchify3d(
    x: Tensor, patch_f: Int, patch_h: Int, patch_w: Int, ctx: DeviceContext
) raises -> Tensor:
    """Video-DiT 3D patch unfold: [C,F,H,W] -> [n_patches, C*pf*ph*pw].

    Non-overlapping (pf,ph,pw) cubes (kernel==stride==patch). Token order is
    F-major then H then W (`patch = fi*HO*WO + hi*WO + wi`). Within-patch flatten
    is (c, pf, ph, pw) with channel SLOWEST — exactly the row-major memory order
    of a torch Conv3d weight [out, C, pf, ph, pw], so the caller's patch-embed is
    `linear(patchify3d(x,...), pe_w.reshape([out, C*pf*ph*pw]), pe_bias)`. Matches
    wan22_dit.rs:667 / cosmos_predict25_dit.rs:1544 `patchify`. pf,ph,pw must
    divide F,H,W. Positional embedding is the caller's job (kept out of this op)."""
    var xshape = x.shape()
    if len(xshape) != 4:
        raise Error("patchify3d: x must be rank-4 [C,F,H,W]")
    var C = xshape[0]
    var F = xshape[1]
    var H = xshape[2]
    var W = xshape[3]
    if patch_f <= 0 or patch_h <= 0 or patch_w <= 0:
        raise Error("patchify3d: patch sizes must be positive")
    if F % patch_f != 0 or H % patch_h != 0 or W % patch_w != 0:
        raise Error("patchify3d: patch sizes must divide F,H,W")
    var FO = F // patch_f
    var HO = H // patch_h
    var WO = W // patch_w
    var L = FO * HO * WO
    var PD = C * patch_f * patch_h * patch_w
    var total = L * PD

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var grid = (total + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_patchify3d_kernel_f32, _patchify3d_kernel_f32](
            X, O, C, F, H, W, patch_f, patch_h, patch_w, FO, HO, WO,
            grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_patchify3d_kernel_bf16, _patchify3d_kernel_bf16](
            X, O, C, F, H, W, patch_f, patch_h, patch_w, FO, HO, WO,
            grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_patchify3d_kernel_f16, _patchify3d_kernel_f16](
            X, O, C, F, H, W, patch_f, patch_h, patch_w, FO, HO, WO,
            grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    var out_shape = List[Int]()
    out_shape.append(L)
    out_shape.append(PD)
    return Tensor(out_buf^, out_shape^, x.dtype())


# ─────────────────────────────────────────────────────────────────────────────
# unpatchify3d. Inverse fold AFTER the output Linear. Mirrors wan22_dit.rs:705
# einsum 'fhwpqrc->cfphqwr' (and cosmos:1588 "(p1 p2 t' c)"): the within-patch
# READ order is (pf, ph, pw, c) with c FASTEST — DIFFERENT from patchify's
# c-slowest. One thread per OUTPUT element [C,F,H,W]; decode (ci, src_f, src_h,
# src_w), recover (fi,pfi)/(hi,phi)/(wi,pwi), token patch = fi*HO*WO+hi*WO+wi,
# read src_ch = ((pfi*ph + phi)*pw + pwi)*C + ci, seq_off = patch*PD + src_ch.
# ─────────────────────────────────────────────────────────────────────────────
def _unpatchify3d_kernel_f32(
    seq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    C: Int, F: Int, H: Int, W: Int,
    pf: Int, ph: Int, pw: Int,
    FO: Int, HO: Int, WO: Int,
):
    var idx = Int(global_idx.x)
    var total = C * F * H * W
    if idx < total:
        var src_w = idx % W
        var rem = idx // W
        var src_h = rem % H
        rem = rem // H
        var src_f = rem % F
        var ci = rem // F
        var fi = src_f // pf
        var pfi = src_f % pf
        var hi = src_h // ph
        var phi = src_h % ph
        var wi = src_w // pw
        var pwi = src_w % pw
        var PD = C * pf * ph * pw
        var patch = (fi * HO + hi) * WO + wi
        var src_ch = ((pfi * ph + phi) * pw + pwi) * C + ci
        var seq_off = patch * PD + src_ch
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](seq[seq_off]))


def _unpatchify3d_kernel_bf16(
    seq: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    C: Int, F: Int, H: Int, W: Int,
    pf: Int, ph: Int, pw: Int,
    FO: Int, HO: Int, WO: Int,
):
    var idx = Int(global_idx.x)
    var total = C * F * H * W
    if idx < total:
        var src_w = idx % W
        var rem = idx // W
        var src_h = rem % H
        rem = rem // H
        var src_f = rem % F
        var ci = rem // F
        var fi = src_f // pf
        var pfi = src_f % pf
        var hi = src_h // ph
        var phi = src_h % ph
        var wi = src_w // pw
        var pwi = src_w % pw
        var PD = C * pf * ph * pw
        var patch = (fi * HO + hi) * WO + wi
        var src_ch = ((pfi * ph + phi) * pw + pwi) * C + ci
        var seq_off = patch * PD + src_ch
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](seq[seq_off]))


def _unpatchify3d_kernel_f16(
    seq: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    C: Int, F: Int, H: Int, W: Int,
    pf: Int, ph: Int, pw: Int,
    FO: Int, HO: Int, WO: Int,
):
    var idx = Int(global_idx.x)
    var total = C * F * H * W
    if idx < total:
        var src_w = idx % W
        var rem = idx // W
        var src_h = rem % H
        rem = rem // H
        var src_f = rem % F
        var ci = rem // F
        var fi = src_f // pf
        var pfi = src_f % pf
        var hi = src_h // ph
        var phi = src_h % ph
        var wi = src_w // pw
        var pwi = src_w % pw
        var PD = C * pf * ph * pw
        var patch = (fi * HO + hi) * WO + wi
        var src_ch = ((pfi * ph + phi) * pw + pwi) * C + ci
        var seq_off = patch * PD + src_ch
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](seq[seq_off]))


def unpatchify3d(
    seq: Tensor,
    out_channels: Int, frames: Int, height: Int, width: Int,
    patch_f: Int, patch_h: Int, patch_w: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Inverse 3D fold (post output-Linear): [n_patches, C_out*pf*ph*pw] ->
    [C_out, F, H, W]. The output geometry can't be inferred from the sequence, so
    C_out/F/H/W and the patch sizes are passed explicitly. Within-patch READ order
    is (pf, ph, pw, c) — channel FASTEST — mirroring wan22_dit.rs:705 einsum
    'fhwpqrc->cfphqwr' (and cosmos:1588 "(p1 p2 t' c)"). This is INTENTIONALLY
    different from patchify3d's c-slowest order (the model's FinalLayer linear is
    trained to this layout); it is NOT a literal transpose-inverse of patchify3d.
    Requires L == (F/pf)*(H/ph)*(W/pw) and last dim == C_out*pf*ph*pw."""
    var sshape = seq.shape()
    if len(sshape) != 2:
        raise Error("unpatchify3d: seq must be rank-2 [n_patches, C*pf*ph*pw]")
    var L = sshape[0]
    var PDin = sshape[1]
    var C = out_channels
    if patch_f <= 0 or patch_h <= 0 or patch_w <= 0:
        raise Error("unpatchify3d: patch sizes must be positive")
    if frames % patch_f != 0 or height % patch_h != 0 or width % patch_w != 0:
        raise Error("unpatchify3d: patch sizes must divide F,H,W")
    var FO = frames // patch_f
    var HO = height // patch_h
    var WO = width // patch_w
    if L != FO * HO * WO:
        raise Error("unpatchify3d: n_patches != (F/pf)*(H/ph)*(W/pw)")
    var PD = C * patch_f * patch_h * patch_w
    if PDin != PD:
        raise Error("unpatchify3d: last dim != C_out*pf*ph*pw")
    var total = C * frames * height * width

    var dt = seq.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](total * seq.dtype().byte_size())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var s_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](L * PD))
    var grid = (total + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            seq.buf.unsafe_ptr().bitcast[Float32](), s_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_unpatchify3d_kernel_f32, _unpatchify3d_kernel_f32](
            S, O, C, frames, height, width, patch_f, patch_h, patch_w, FO, HO, WO,
            grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            seq.buf.unsafe_ptr().bitcast[BFloat16](), s_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_unpatchify3d_kernel_bf16, _unpatchify3d_kernel_bf16](
            S, O, C, frames, height, width, patch_f, patch_h, patch_w, FO, HO, WO,
            grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var S = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            seq.buf.unsafe_ptr().bitcast[Float16](), s_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_unpatchify3d_kernel_f16, _unpatchify3d_kernel_f16](
            S, O, C, frames, height, width, patch_f, patch_h, patch_w, FO, HO, WO,
            grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    var out_shape = List[Int]()
    out_shape.append(C)
    out_shape.append(frames)
    out_shape.append(height)
    out_shape.append(width)
    return Tensor(out_buf^, out_shape^, seq.dtype())
