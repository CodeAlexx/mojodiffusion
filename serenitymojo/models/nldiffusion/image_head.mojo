# models/nldiffusion/image_head.mojo — CHUNK C: NL-Diffusion-Image image head.
#
# Pure-Mojo + MAX (GPU-only, INFERENCE) port of the four image-head pieces of
# NVIDIA NL-Diffusion-Image (reference `modeling_nemotron_labs_diffusion_image.py`
# L344-383 SimpleUVitBlock, L428-444 call_gen_embedding / call_gen_predictor).
# See serenitymojo/models/nldiffusion/PORT_SPEC.md §CHUNK C.
#
# Data flow (1024^2, gen_shape=(64,64)):
#   gen_embedding(ids[1,4096]) -> [1,4096,4096]
#     downsample_gen (SimpleUVitBlock, 2x down) -> [1,1024,4096]
#       [backbone — CHUNK A/B, done]
#     upsample_gen  (SimpleUVitBlock, 2x up)   -> [1,4096,4096]
#   gen_predictor(.) -> codebook logits [.,131072]
#
# The two SimpleUVitBlocks wrap diffusers Downsample2D / Upsample2D. The exact
# forward math reproduced here (from diffusers models/{downsampling,upsampling}.py):
#   Downsample2D(use_conv, k=2, stride=2, pad=0, norm=rms_norm eps 1e-6, bias=False):
#     norm(x.permute NCHW->NHWC).permute back  ; F.pad (0,1,0,1)  ; Conv2d(k2,s2,p0)
#     -> the (0,1,0,1) bottom/right zero-pad is NEVER read for an even (64x64) input
#        (last k2/s2 window starts at index 62, covers 62,63) — so a plain k2/s2/p0
#        conv over the 64x64 map is numerically identical. Verified in the analysis.
#   Upsample2D(use_conv_transpose, k=2, stride=2, pad=0, norm=rms_norm eps 1e-6,
#              bias=False, interpolate=False):
#     norm(x NCHW->NHWC).permute back ; return ConvTranspose2d(k2,s2,p0)(x)
#     -> k==stride==2 conv-transpose has NO overlap: each input cell (i,j) writes a
#        clean 2x2 output block out[2i+ki,2j+kj,co] = sum_ci in[i,j,ci]*W[ci,co,ki,kj].
#        HAND-ROLLED here (no transpose-conv op in the foundation) as 4 matmuls
#        (one per (ki,kj), reusing `linear`) + a scatter/interleave kernel.
#
# NHWC realization: the reference does "b (h w) d -> b d h w" (NCHW) then permutes
# to NHWC for the RMSNorm; our [1, h*w, d] tensor IS already the NHWC map
# [1, h, w, d] (position index h*w == (i,j), channel == d) — a pure reshape, no
# permute. RMSNorm is over the last (channel) dim either way. conv2d is NHWC.
#
# The RMSNorm weight (downsample.norm / upsample.norm) has elementwise_affine=True
# so it is a genuine learned [4096] scale (NOT identity) — loaded from the shards.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.memory import ArcPointer
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import reshape, permute, slice, gather_rows

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime UVIT_EPS = Float32(1.0e-6)  # diffusers Downsample2D/Upsample2D norm eps


def _load_bf16(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> ArcPointer[Tensor]:
    """Load a bf16 weight tensor from the shards (stored dtype preserved)."""
    var tv = st.tensor_view(name)
    var t = Tensor.from_view(tv, ctx)
    return ArcPointer(t^)


# ── piece 1: gen_embedding ────────────────────────────────────────────────────
def nldiff_gen_embedding(
    ids: List[Int], model: ShardedSafeTensors, ctx: DeviceContext
) raises -> Tensor:
    """`nn.Embedding(codebook_size+256=131328, 4096)` lookup.

    `ids` are the image token ids (length N). Returns [1, N, 4096] (bf16 — the
    embedding table's storage dtype; a lookup is exact, no arithmetic).
    Reference: call_gen_embedding L430 `self.gen_embedding(token_ids)`.
    """
    var w = _load_bf16(model, String("encoder.gen_embedding.weight"), ctx)
    var emb = gather_rows(w[], ids, ctx)  # [N, 4096]
    var n = len(ids)
    var d = w[].shape()[1]
    return reshape(emb, [1, n, d], ctx)


# ── piece 2: downsample_gen (SimpleUVitBlock, 2x down) ─────────────────────────
def nldiff_downsample_gen[
    H: Int, W: Int, D: Int
](
    x: Tensor,
    conv_w: Tensor,
    norm_w: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """SimpleUVitBlock(downsample=True): rearrange -> Downsample2D -> rearrange.

    x:      [1, H*W, D]                         (bf16; H=W=64, D=4096 for 1024^2)
    conv_w: [Cout, Cin, 2, 2] = [D, D, 2, 2]    (PyTorch Conv2d weight, bf16)
    norm_w: [D]                                 (RMSNorm scale, bf16)
    returns [1, (H/2)*(W/2), D]                 (bf16; [1,1024,4096]).

    Downsample2D.forward: norm(NHWC) over channels (eps 1e-6) then Conv2d(k2,s2,p0).
    The external F.pad (0,1,0,1) is a no-op for even H,W (see file header) so a
    plain k2/s2/p0 conv over the H*W map is identical.
    """
    # [1, H*W, D] IS the NHWC map [1, H, W, D] (a reshape, not a permute).
    var nhwc = reshape(x, [1, H, W, D], ctx)
    var normed = rms_norm(nhwc, norm_w, UVIT_EPS, ctx)  # over channel dim
    # PyTorch Conv2d weight [Cout,Cin,Kh,Kw] -> conv2d RSCF [Kh,Kw,Cin,Cout].
    var wrscf = permute(conv_w, [2, 3, 1, 0], ctx)  # [2,2,D,D]
    var y = conv2d[
        1, H, W, D, 2, 2, D, 2, 2, 0, 0
    ](normed, wrscf, None, ctx)  # [1, H/2, W/2, D]
    return reshape(y, [1, (H // 2) * (W // 2), D], ctx)


# ── conv-transpose (k2/s2/p0) scatter — writes each input cell's 2x2 output block
def _scatter_interleave_kernel_bf16(
    y: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    total: Int,
    C: Int,
    Ws: Int,
    Wo: Int,
    ki: Int,
    kj: Int,
):
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var row = idx // C
    var c = idx % C
    var i = row // Ws
    var j = row % Ws
    var orow = (2 * i + ki) * Wo + (2 * j + kj)
    dst[orow * C + c] = y[idx]


def _scatter_interleave_kernel_f32(
    y: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    total: Int,
    C: Int,
    Ws: Int,
    Wo: Int,
    ki: Int,
    kj: Int,
):
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var row = idx // C
    var c = idx % C
    var i = row // Ws
    var j = row % Ws
    var orow = (2 * i + ki) * Wo + (2 * j + kj)
    dst[orow * C + c] = y[idx]


def _scatter_into(
    mut dst: Tensor,
    y: Tensor,
    Hs: Int,
    Ws: Int,
    Wo: Int,
    ki: Int,
    kj: Int,
    ctx: DeviceContext,
) raises:
    """dst[(2i+ki)*Wo + (2j+kj), c] = y[i*Ws+j, c] for all (i,j,c). y:[Hs*Ws,C]."""
    var c = y.shape()[1]
    var total = Hs * Ws * c
    var dt = y.dtype().to_mojo_dtype()
    var y_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](dst.numel()))
    var grid = (total + _BLOCK - 1) // _BLOCK
    if dt == DType.bfloat16:
        var Y = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            y.buf.unsafe_ptr().bitcast[BFloat16](), y_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            dst.buf.unsafe_ptr().bitcast[BFloat16](), o_rl
        )
        ctx.enqueue_function[
            _scatter_interleave_kernel_bf16, _scatter_interleave_kernel_bf16
        ](Y, O, total, c, Ws, Wo, ki, kj, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.float32:
        var Y = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            y.buf.unsafe_ptr().bitcast[Float32](), y_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            dst.buf.unsafe_ptr().bitcast[Float32](), o_rl
        )
        ctx.enqueue_function[
            _scatter_interleave_kernel_f32, _scatter_interleave_kernel_f32
        ](Y, O, total, c, Ws, Wo, ki, kj, grid_dim=grid, block_dim=_BLOCK)
    else:
        raise Error("_scatter_into: dtype must be bf16 or f32")


# ── piece 3: upsample_gen (SimpleUVitBlock, 2x up) ─────────────────────────────
def nldiff_upsample_gen[
    H: Int, W: Int
](
    x: Tensor,
    conv_w: Tensor,
    norm_w: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """SimpleUVitBlock(upsample=True): rearrange -> Upsample2D -> rearrange.

    x:      [1, H*W, D]                         (bf16; H=W=32, the (64/2,64/2) map)
    conv_w: [Cin, Cout, 2, 2] = [D, D, 2, 2]    (PyTorch ConvTranspose2d weight, bf16)
    norm_w: [D]                                 (RMSNorm scale, bf16)
    returns [1, (2H)*(2W), D]                   (bf16; [1,4096,4096]).

    Upsample2D.forward (use_conv_transpose): norm(NHWC) over channels (eps 1e-6)
    then ConvTranspose2d(k2,s2,p0,bias=False). Hand-rolled transpose conv (k==stride
    == 2, no overlap): 4 `linear` matmuls (one per output-block offset (ki,kj), using
    the weight slice W[:,:,ki,kj] as [Cin,Cout]) + a scatter into the 2x-larger map.
    """
    var d = x.shape()[2]
    var nhwc = reshape(x, [1, H, W, d], ctx)
    var normed = rms_norm(nhwc, norm_w, UVIT_EPS, ctx)  # [1,H,W,D]
    var inp = reshape(normed, [H * W, d], ctx)  # [H*W, Cin]

    # ConvTranspose weight [Cin,Cout,kH,kW] -> [kH,kW,Cin,Cout] so each (ki,kj)
    # slice is a contiguous [Cin,Cout] matrix.
    var wp = permute(conv_w, [2, 3, 0, 1], ctx)  # [2,2,D,D]
    var w4 = reshape(wp, [4, d, d], ctx)  # [(ki*2+kj), Cin, Cout]

    var Ho = 2 * H
    var Wo = 2 * W
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        Ho * Wo * d * x.dtype().byte_size()
    )
    var result = Tensor(out_buf^, [1, Ho * Wo, d], x.dtype())

    for kk in range(4):
        var ki = kk // 2
        var kj = kk % 2
        var wslice = slice(w4, 0, kk, 1, ctx)  # [1, Cin, Cout]
        var wslice2 = reshape(wslice^, [d, d], ctx)  # [Cin, Cout]
        # out_block = inp @ W[:,:,ki,kj]  (transpose_b=False: w is [in,out]).
        var yk = linear(inp, wslice2, None, ctx, transpose_b=False)  # [H*W, Cout]
        _scatter_into(result, yk, H, W, Wo, ki, kj, ctx)
        _ = wslice2^
        _ = yk^
    return result^


# ── piece 4: gen_predictor ────────────────────────────────────────────────────
def nldiff_gen_predictor(
    x: Tensor, pred_w: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """`nn.Linear(4096, codebook_size=131072, bias=False)`.

    x:      [M, 4096]           (bf16 hidden states)
    pred_w: [131072, 4096]      (PyTorch Linear weight, bf16)
    returns [M, 131072]         (codebook logits; y = x @ pred_wᵀ).
    Reference: call_gen_predictor L444 `self.gen_predictor(gen_hidden_states)`.
    """
    return linear(x, pred_w, None, ctx)  # transpose_b=True (default)
