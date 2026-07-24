# models/vae/seedvr2_vae.mojo — SeedVR2-3B video VAE decoder (pure-Mojo, INFERENCE).
#
# Chunk 1: the inflated CAUSAL-3D conv — the atomic foundation of the VAE.
#
# Reference semantics (SeedVR `InflatedCausalConv3d`,
#   /home/alex/SeedVR/models/video_vae_v3/modules/inflated_layers.py +
#   causal_inflation_lib.py `extend_head`):
# A causal 3D conv over NCDHW video (D = temporal/frame axis). For a kernel
# (kd,kh,kw) with symmetric spatial pad:
#   1. The temporal (D) axis is padded LEFT-ONLY by (kd-1) frames, and the pad
#      content is the FIRST FRAME REPLICATED (`extend_head` with memory=None,
#      `times = 2*temporal_padding`; for pad=1, k=3 that is 2 copies of frame[0]
#      prepended along D). This keeps the conv causal (output D length preserved).
#   2. Spatial H,W use SYMMETRIC zero-pad = (kh//2, kw//2).
#   3. Then a standard Conv3d (stride 1, temporal pad removed, symmetric spatial
#      pad). In D=4 -> out D=4.
#
# We reuse the verified NDHWC/FCQRS cuDNN conv (models/vae/conv3d.mojo,
# `conv3d_fcqrs_cudnn`) with pad_d=0 and do the causal first-frame-replicate
# left-pad manually via the existing `slice`/`concat` ops (same pattern the LTX2
# decoder uses for its temporal replicate-pad). Layout bridging NCDHW<->NDHWC
# reuses the existing `permute` op (no new kernel).
#
# conv3d_fcqrs_cudnn wants NDHWC [N,D,H,W,Cin] input and FCQRS filter
# [Cout,Cin,Q,R,S] (== checkpoint OIDHW, loaded as-is, NO transpose).
#
# Mojo 1.0.0b1, NVIDIA GPU. Pure Mojo + MAX. No autograd, no Python at runtime.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import (
    permute, slice, concat, add, reshape, transpose, mul_scalar,
)
from serenitymojo.ops.pixelshuffle import depth_to_space_3d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.linear import linear
from serenitymojo.ops.softmax import softmax_lastdim
from serenitymojo.models.vae.conv3d import conv3d_fcqrs_cudnn
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from math import sqrt


# ── causal_conv3d ─────────────────────────────────────────────────────────────
# x_ncdhw: NCDHW [N, C, D, H, W]  (D = temporal/frame axis)
# w_oidhw: checkpoint OIDHW [Cout, Cin, KD, KH, KW]  (== FCQRS for cuDNN, as-is)
# bias:    [Cout]
# returns: NCDHW [N, Cout, D, H, W]  (D preserved by the causal left-pad)
def causal_conv3d[
    KD: Int, KH: Int, KW: Int
](x_ncdhw: Tensor, w_oidhw: Tensor, var bias: Tensor, ctx: DeviceContext) raises -> Tensor:
    var xs = x_ncdhw.shape()
    if len(xs) != 5:
        raise Error("seedvr2 causal_conv3d: x must be rank-5 NCDHW [N,C,D,H,W]")

    # NCDHW [N,C,D,H,W] -> NDHWC [N,D,H,W,C]: permute axes (0,2,3,4,1).
    var to_ndhwc = List[Int]()
    to_ndhwc.append(0); to_ndhwc.append(2); to_ndhwc.append(3)
    to_ndhwc.append(4); to_ndhwc.append(1)
    var x = permute(x_ncdhw, to_ndhwc^, ctx)  # [N,D,H,W,C]

    # Causal temporal left-pad: prepend (KD-1) copies of frame[0] along D (axis 1
    # in NDHWC). extend_head(memory=None) duplicates the FIRST frame features.
    var first = slice(x, 1, 0, 1, ctx)  # [N,1,H,W,C]
    var x_pad = x^
    comptime for _unused in range(KD - 1):
        x_pad = concat(1, ctx, first, x_pad)  # prepend one first-frame copy

    # Standard conv3d: temporal pad already applied (pad_d=0), symmetric spatial
    # pad = (KH//2, KW//2). Bias added inside the cuDNN helper.
    var y = conv3d_fcqrs_cudnn(
        x_pad, w_oidhw, Optional[Tensor](bias^),
        1, 1, 1, 0, KH // 2, KW // 2, ctx,
    )  # NDHWC [N,D,H,W,Cout]

    # NDHWC [N,D,H,W,Cout] -> NCDHW [N,Cout,D,H,W]: permute axes (0,4,1,2,3).
    var to_ncdhw = List[Int]()
    to_ncdhw.append(0); to_ncdhw.append(4)
    to_ncdhw.append(1); to_ncdhw.append(2); to_ncdhw.append(3)
    return permute(y, to_ncdhw^, ctx)


# ── PER-FRAME GroupNorm over NCDHW ────────────────────────────────────────────
# SeedVR `causal_norm_wrapper` folds the temporal axis D into the batch, runs a
# 2D GroupNorm(num_groups=32, eps=1e-6) over (C,H,W) PER FRAME, then unfolds.
# (This is DIFFERENT from the LTX2 upsampler's `_group_norm_ndhwc`, which reduces
# over the whole (D,H,W) slab per group — NOT per-frame. Do not reuse that here.)
#
# Bridge: NCDHW [N,C,D,H,W] -> NDHWC [N,D,H,W,C] (permute 0,2,3,4,1) -> merge N,D
# into the batch giving NHWC [N*D, H, W, C] (contiguous, zero-reorder reshape) ->
# group_norm(32, 1e-6) reduces per (n*d, group) over (H,W,cpg) == per-frame ->
# reshape back to [N,D,H,W,C] -> NCDHW (permute 0,4,1,2,3).
def _group_norm_per_frame(
    x_ncdhw: Tensor, weight: Tensor, bias: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var to_ndhwc = List[Int]()
    to_ndhwc.append(0); to_ndhwc.append(2); to_ndhwc.append(3)
    to_ndhwc.append(4); to_ndhwc.append(1)
    var ndhwc = permute(x_ncdhw, to_ndhwc^, ctx)  # [N,D,H,W,C]
    var sh = ndhwc.shape()
    var N = sh[0]; var D = sh[1]; var H = sh[2]; var W = sh[3]; var C = sh[4]

    # merge (N,D) -> batch: NHWC [N*D, H, W, C] (row-major bytes unchanged).
    var m4 = List[Int]()
    m4.append(N * D); m4.append(H); m4.append(W); m4.append(C)
    var x4 = reshape(ndhwc, m4^, ctx)

    var gn = group_norm(x4, weight, bias, 32, Float32(1e-6), ctx)  # per-frame

    # unfold back to NDHWC then NCDHW.
    var s5 = List[Int]()
    s5.append(N); s5.append(D); s5.append(H); s5.append(W); s5.append(C)
    var gn5 = reshape(gn, s5^, ctx)
    var to_ncdhw = List[Int]()
    to_ncdhw.append(0); to_ncdhw.append(4)
    to_ncdhw.append(1); to_ncdhw.append(2); to_ncdhw.append(3)
    return permute(gn5, to_ncdhw^, ctx)


# ── resnet_block3d ────────────────────────────────────────────────────────────
# SeedVR `ResnetBlock3D.forward` (pre-activation residual, temb=None,
# output_scale_factor=1.0, nonlinearity=SiLU), INFERENCE:
#   h = SiLU(GroupNorm(norm1, input))              # per-frame GN
#   h = causal_conv3d[3,3,3](conv1, h)
#   h = SiLU(GroupNorm(norm2, h))                  # per-frame GN
#   h = causal_conv3d[3,3,3](conv2, h)
#   if has_shortcut: input = causal_conv3d[1,1,1](conv_shortcut, input)
#   return input + h
#
# x_ncdhw: NCDHW [N, Cin, D, H, W]. Weights are checkpoint layout: norms [C],
# conv{1,2} OIDHW [Cout,Cin,3,3,3], conv_shortcut OIDHW [Cout,Cin,1,1,1].
# has_shortcut selects the channel-changing path; when False sc_w/sc_b are unused
# placeholders. Fixed num_groups=32, eps=1e-6. Reuses chunk-1 `causal_conv3d`
# (VERIFIED bit-exact); conv biases are `.clone(ctx)`'d because it consumes bias.
def resnet_block3d(
    x_ncdhw: Tensor,
    norm1_w: Tensor, norm1_b: Tensor,
    conv1_w: Tensor, conv1_b: Tensor,
    norm2_w: Tensor, norm2_b: Tensor,
    conv2_w: Tensor, conv2_b: Tensor,
    has_shortcut: Bool,
    sc_w: Tensor, sc_b: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var h = _group_norm_per_frame(x_ncdhw, norm1_w, norm1_b, ctx)
    h = silu(h, ctx)
    h = causal_conv3d[3, 3, 3](h, conv1_w, conv1_b.clone(ctx), ctx)

    h = _group_norm_per_frame(h, norm2_w, norm2_b, ctx)
    h = silu(h, ctx)
    h = causal_conv3d[3, 3, 3](h, conv2_w, conv2_b.clone(ctx), ctx)

    if has_shortcut:
        var sc = causal_conv3d[1, 1, 1](x_ncdhw, sc_w, sc_b.clone(ctx), ctx)
        return add(sc, h, ctx)
    return add(x_ncdhw, h, ctx)


# ── upsample3d ────────────────────────────────────────────────────────────────
# SeedVR `Upsample3D.forward` (MAGViT-v2 pixel-shuffle upsampler, no norm,
# INFERENCE, non-slicing), NCDHW in/out:
#   h = upscale_conv(h)                       # plain Conv3d 1x1x1, C -> C*ratio
#   h = rearrange(h, "b (x y z c) f h w -> b c (f z) (h x) (w y)",
#                 x=SR, y=SR, z=TR)           # depth-to-space 3D
#   if TR == 2: h = remove_head(h)            # causal-dup frame removal (see below)
#   h = causal_conv3d[3,3,3](conv, h)         # learned smoothing
#
# ratio = SR*SR*TR (spatial_ratio² · temporal_ratio). SR=spatial_ratio (2),
# TR=temporal_ratio (1 or 2). remove_head is applied iff TR > 1.
#
# CRITICAL — the einops channel decomposition `(x y z c)` is channel-MINOR (c is
# the FASTEST index, spatial x the SLOWEST): flat channel = ((x*Y+y)*Z+z)*C+c,
# with X=Y=SR, Z=TR, C=out_channels. Output coords: fo=f*Z+z, ho=h*X+x, wo=w*Y+y.
# The verified `depth_to_space_3d` (ops/pixelshuffle) uses the DIFFERENT
# channel-MAJOR order ct=((c*p1+i1)*p2+i2)*p3+i3 with fo=f*p1+i1, ho=h*p2+i2,
# wo=w*p3+i3. We reconcile by (option c) a channel-only REPACK of the upscale_conv
# output from SeedVR's (x,y,z,c) packing into the (c,z,x,y) packing that
# depth_to_space_3d(p1=Z, p2=X, p3=Y) consumes — a rank-6 reshape+permute+reshape
# (F,H,W kept MERGED so we stay within permute's rank-6 limit). Then p1=Z(temporal)
# → fo=f*Z+z, p2=X → ho=h*X+x, p3=Y → wo=w*Y+y == SeedVR's mapping exactly.
#
# remove_head (SeedVR causal_inflation_lib.remove_head, times=1, sp_rank=0):
#   cat(h[:,:,:1], h[:,:,2:], dim=2) — KEEPS frame 0, DROPS frame index 1, keeps
#   the rest. (This is NOT a plain drop-first-frame; it removes the duplicated
#   causal head that d2s created at temporal position 1.) D -> D-1 along axis 2.
#
# x_ncdhw: NCDHW [N, C, D, H, W]. up_w/up_b: upscale Conv3d 1x1x1
# [C*ratio, C, 1,1,1]/[C*ratio]. conv_w/conv_b: causal 3x3x3 [C, C, 3,3,3]/[C].
# Reuses chunk-1 causal_conv3d (consumes its bias → `.clone(ctx)`). All F32.
def upsample3d[
    SR: Int, TR: Int
](
    x_ncdhw: Tensor,
    up_w: Tensor, up_b: Tensor,
    conv_w: Tensor, conv_b: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var xs = x_ncdhw.shape()
    if len(xs) != 5:
        raise Error("seedvr2 upsample3d: x must be rank-5 NCDHW [N,C,D,H,W]")
    var N = xs[0]; var C = xs[1]; var D = xs[2]; var H = xs[3]; var W = xs[4]

    # 1) upscale_conv: plain Conv3d 1x1x1 (pad 0). causal_conv3d[1,1,1]'s temporal
    #    left-pad loop runs 0 times (KD-1==0) → identical to a plain 1x1x1 conv.
    var hup = causal_conv3d[1, 1, 1](x_ncdhw, up_w, up_b.clone(ctx), ctx)
    # hup: [N, C*SR*SR*TR, D, H, W]

    # 2) channel REPACK (x,y,z,c) -> (c,z,x,y), rank-6 (F,H,W merged).
    var FHW = D * H * W
    var m3 = List[Int]()
    m3.append(N); m3.append(C * SR * SR * TR); m3.append(FHW)
    var h3 = reshape(hup, m3^, ctx)                 # [N, Ctot, FHW]
    var m6 = List[Int]()
    m6.append(N); m6.append(SR); m6.append(SR); m6.append(TR)
    m6.append(C); m6.append(FHW)
    var h6 = reshape(h3, m6^, ctx)                  # [N, X, Y, Z, C, FHW]
    var perm = List[Int]()
    perm.append(0); perm.append(4); perm.append(3)  # B, C, Z
    perm.append(1); perm.append(2); perm.append(5)  # X, Y, FHW
    var hp = permute(h6, perm^, ctx)                # [N, C, Z, X, Y, FHW]
    var m5 = List[Int]()
    m5.append(N); m5.append(C * SR * SR * TR)
    m5.append(D); m5.append(H); m5.append(W)
    var hrep = reshape(hp, m5^, ctx)                # [N, Ctot(c,z,x,y), D, H, W]

    # 3) depth_to_space_3d with p1=Z=TR (temporal), p2=X=SR, p3=Y=SR. No frame
    #    drop here (remove_head is the SeedVR variant below, not drop-first).
    var hds = depth_to_space_3d(hrep, TR, SR, SR, False, ctx)
    # hds: [N, C, D*TR, H*SR, W*SR]

    # 4) remove_head iff TR>1: cat(hds[:,:,:1], hds[:,:,2:]) along D (axis 2).
    var hh = hds^
    comptime if TR > 1:
        var Dds = hh.shape()[2]
        var head = slice(hh, 2, 0, 1, ctx)          # frame 0
        var tail = slice(hh, 2, 2, Dds - 2, ctx)    # frames 2..Dds-1
        hh = concat(2, ctx, head, tail)             # D*TR -> D*TR - 1

    # 5) learned causal 3x3x3 smoothing.
    return causal_conv3d[3, 3, 3](hh, conv_w, conv_b.clone(ctx), ctx)


# ── mid_attention ─────────────────────────────────────────────────────────────
# SeedVR `UNetMidBlock3D` spatial self-attention (diffusers `Attention`, SINGLE
# HEAD, per-frame). Input is ALREADY per-frame folded to 4-D NCHW [B,C,H,W]
# (the frames are the batch); heads=1, dim_head=C (full attention),
# residual_connection=True, rescale_output_factor=1.0, norm_num_groups=32.
#
# Forward (diffusers AttnProcessor2_0, single head H=1, Dh=C):
#   residual = x                                       # [B,C,H,W]
#   h = GroupNorm(group_norm, x)                       # per-frame, 32 groups
#   h = h.view(B,C,H*W).transpose(1,2)                 # [B, S, C]  (S=H*W)
#   q = Linear(to_q,h); k = Linear(to_k,h); v = Linear(to_v,h)   # [B,S,C]
#   attn = softmax((q @ kᵀ) * 1/sqrt(C), -1) @ v       # per-frame [B,S,C]
#   out = Linear(to_out.0, attn)                       # [B,S,C]
#   out = out.transpose(1,2).view(B,C,H,W)
#   return out + residual                              # rescale=1.0
#
# ALL F32 (matmuls via `linear` = F32-accumulated GEMM; softmax F32). The
# batched per-frame Q@Kᵀ@V loops the B frames (each has its own K/V) using
# `linear` for both matmuls: scores = linear(q,k,transpose_b=True) = q @ kᵀ,
# out = linear(attn,v,transpose_b=False) = attn @ v. `linear` handles the
# token-batched [B,S,C] q/k/v projections directly (leading dims flatten to M).
# GroupNorm reuses chunk-2 `_group_norm_per_frame` (eps=1e-6) by adding a
# singleton D axis (NCHW -> NCDHW [B,C,1,H,W]).
def mid_attention(
    x_nchw: Tensor,
    gn_w: Tensor, gn_b: Tensor,
    q_w: Tensor, q_b: Tensor,
    k_w: Tensor, k_b: Tensor,
    v_w: Tensor, v_b: Tensor,
    o_w: Tensor, o_b: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var xs = x_nchw.shape()
    if len(xs) != 4:
        raise Error("seedvr2 mid_attention: x must be rank-4 NCHW [B,C,H,W]")
    var B = xs[0]; var C = xs[1]; var H = xs[2]; var W = xs[3]
    var S = H * W

    # GroupNorm per-frame: NCHW [B,C,H,W] -> NCDHW [B,C,1,H,W] -> reuse chunk-2 GN.
    var g5 = List[Int]()
    g5.append(B); g5.append(C); g5.append(1); g5.append(H); g5.append(W)
    var x5 = reshape(x_nchw, g5^, ctx)
    var gn5 = _group_norm_per_frame(x5, gn_w, gn_b, ctx)  # eps=1e-6
    var gn4 = List[Int]()
    gn4.append(B); gn4.append(C); gn4.append(H); gn4.append(W)
    var hn = reshape(gn5, gn4^, ctx)                      # [B,C,H,W]

    # view(B,C,S).transpose(1,2) -> [B,S,C] (channels-last tokens).
    var hcs = List[Int]()
    hcs.append(B); hcs.append(C); hcs.append(S)
    var hbcs = reshape(hn, hcs^, ctx)                     # [B,C,S]
    var h = transpose(hbcs, 1, 2, ctx)                    # [B,S,C]

    # Q/K/V projections (token-batched; `linear` flattens leading dims).
    var q = linear(h, q_w, Optional[Tensor](q_b.clone(ctx)), ctx, True)   # [B,S,C]
    var k = linear(h, k_w, Optional[Tensor](k_b.clone(ctx)), ctx, True)
    var v = linear(h, v_w, Optional[Tensor](v_b.clone(ctx)), ctx, True)

    var scale = Float32(1.0) / sqrt(Float32(C))
    var frame2 = List[Int]()  # [S,C] target for a single frame
    frame2.append(S); frame2.append(C)
    var back3 = List[Int]()   # [1,S,C] for re-concat along batch
    back3.append(1); back3.append(S); back3.append(C)

    # Per-frame single-head attention, concatenated back along the batch axis.
    var attn_out = x_nchw.clone(ctx)  # placeholder, overwritten below
    for b in range(B):
        var qf2 = frame2.copy()
        var qb = reshape(slice(q, 0, b, 1, ctx), qf2^, ctx)    # [S,C]
        var kf2 = frame2.copy()
        var kb = reshape(slice(k, 0, b, 1, ctx), kf2^, ctx)    # [S,C]
        var vf2 = frame2.copy()
        var vb = reshape(slice(v, 0, b, 1, ctx), vf2^, ctx)    # [S,C]

        var scores = linear(qb, kb, None, ctx, True)           # q @ kᵀ  [S,S]
        scores = mul_scalar(scores, scale, ctx)
        var attn = softmax_lastdim(scores, ctx)                # [S,S]
        var ob = linear(attn, vb, None, ctx, False)            # attn @ v [S,C]
        var b3 = back3.copy()
        var ob3 = reshape(ob, b3^, ctx)                        # [1,S,C]
        if b == 0:
            attn_out = ob3^
        else:
            attn_out = concat(0, ctx, attn_out, ob3)           # [b+1,S,C]

    # to_out.0 projection, then back to NCHW and residual add.
    var out_bsc = linear(attn_out, o_w, Optional[Tensor](o_b.clone(ctx)), ctx, True)  # [B,S,C]
    var out_bcs = transpose(out_bsc, 1, 2, ctx)                # [B,C,S]
    var o4 = List[Int]()
    o4.append(B); o4.append(C); o4.append(H); o4.append(W)
    var out_nchw = reshape(out_bcs, o4^, ctx)                  # [B,C,H,W]
    return add(out_nchw, x_nchw, ctx)                          # rescale=1.0


# ── ASSEMBLY (Chunk 5+6): full Decoder3D forward ──────────────────────────────
# Composes the VERIFIED primitives above into SeedVR `Decoder3D.forward`
# (/home/alex/SeedVR/models/video_vae_v3/modules/attn_video_vae.py:966):
#   h = causal_conv3d[3,3,3](conv_in, latent)   # 16 -> 512
#   h = mid_block(h)
#   h = up_block(h, 0..3)
#   h = SiLU(GroupNorm-32(conv_norm_out, h))    # 128 ch, eps 1e-6, per-frame
#   h = causal_conv3d[3,3,3](conv_out, h)       # 128 -> 3
# Weights are streamed from the checkpoint SafeTensors handle by name (no big
# holder struct needed; load_bias moves each tensor straight to device).


# _resnet3d_st: load a ResnetBlock3D's weights from `st` under `prefix` and run
# the verified `resnet_block3d`. Placeholder shortcut tensors (norm1) when the
# block has no channel change — never touched by resnet_block3d in that case.
def _resnet3d_st(
    h: Tensor, st: SafeTensors, prefix: String, has_shortcut: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    var norm1_w = load_bias(st, prefix + ".norm1.weight", ctx)
    var norm1_b = load_bias(st, prefix + ".norm1.bias", ctx)
    var conv1_w = load_bias(st, prefix + ".conv1.weight", ctx)
    var conv1_b = load_bias(st, prefix + ".conv1.bias", ctx)
    var norm2_w = load_bias(st, prefix + ".norm2.weight", ctx)
    var norm2_b = load_bias(st, prefix + ".norm2.bias", ctx)
    var conv2_w = load_bias(st, prefix + ".conv2.weight", ctx)
    var conv2_b = load_bias(st, prefix + ".conv2.bias", ctx)
    var sc_w = norm1_w.clone(ctx)
    var sc_b = norm1_b.clone(ctx)
    if has_shortcut:
        sc_w = load_bias(st, prefix + ".conv_shortcut.weight", ctx)
        sc_b = load_bias(st, prefix + ".conv_shortcut.bias", ctx)
    return resnet_block3d(
        h, norm1_w, norm1_b, conv1_w, conv1_b, norm2_w, norm2_b,
        conv2_w, conv2_b, has_shortcut, sc_w, sc_b, ctx,
    )


# ── mid_block3d (UNetMidBlock3D.forward) ──────────────────────────────────────
#   h = resnet_block3d(resnets.0)(h)                     # 512->512 no shortcut
#   FOLD  NCDHW[N,C,D,H,W] -> NCHW[N*D,C,H,W]  (permute 0,2,1,3,4 then merge N,D)
#   h = mid_attention(attentions.0)(h)                   # per-frame single-head
#   UNFOLD NCHW[N*D,C,H,W] -> NCDHW[N,C,D,H,W] (reshape then permute 0,2,1,3,4)
#   h = resnet_block3d(resnets.1)(h)                     # 512->512 no shortcut
def mid_block3d(h_in: Tensor, st: SafeTensors, ctx: DeviceContext) raises -> Tensor:
    var h = _resnet3d_st(h_in, st, "decoder.mid_block.resnets.0", False, ctx)

    var sh = h.shape()
    var N = sh[0]; var C = sh[1]; var D = sh[2]; var H = sh[3]; var W = sh[4]

    # FOLD: permute(0,2,1,3,4) -> [N,D,C,H,W]; reshape merge (N,D) -> [N*D,C,H,W].
    var to_ndchw = List[Int]()
    to_ndchw.append(0); to_ndchw.append(2); to_ndchw.append(1)
    to_ndchw.append(3); to_ndchw.append(4)
    var hndchw = permute(h, to_ndchw^, ctx)              # [N,D,C,H,W]
    var mf = List[Int]()
    mf.append(N * D); mf.append(C); mf.append(H); mf.append(W)
    var hf = reshape(hndchw, mf^, ctx)                   # [N*D,C,H,W]

    var ap = String("decoder.mid_block.attentions.0")
    var gn_w = load_bias(st, ap + ".group_norm.weight", ctx)
    var gn_b = load_bias(st, ap + ".group_norm.bias", ctx)
    var q_w = load_bias(st, ap + ".to_q.weight", ctx)
    var q_b = load_bias(st, ap + ".to_q.bias", ctx)
    var k_w = load_bias(st, ap + ".to_k.weight", ctx)
    var k_b = load_bias(st, ap + ".to_k.bias", ctx)
    var v_w = load_bias(st, ap + ".to_v.weight", ctx)
    var v_b = load_bias(st, ap + ".to_v.bias", ctx)
    var o_w = load_bias(st, ap + ".to_out.0.weight", ctx)
    var o_b = load_bias(st, ap + ".to_out.0.bias", ctx)
    var ha = mid_attention(hf, gn_w, gn_b, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, ctx)

    # UNFOLD: reshape [N*D,C,H,W] -> [N,D,C,H,W]; permute(0,2,1,3,4) -> [N,C,D,H,W].
    var mu = List[Int]()
    mu.append(N); mu.append(D); mu.append(C); mu.append(H); mu.append(W)
    var hu5 = reshape(ha, mu^, ctx)                      # [N,D,C,H,W]
    var to_ncdhw = List[Int]()
    to_ncdhw.append(0); to_ncdhw.append(2); to_ncdhw.append(1)
    to_ncdhw.append(3); to_ncdhw.append(4)
    var hback = permute(hu5, to_ncdhw^, ctx)             # [N,C,D,H,W]

    return _resnet3d_st(hback, st, "decoder.mid_block.resnets.1", False, ctx)


# ── up_block3d (UpDecoderBlock3D.forward) ─────────────────────────────────────
# 3 resnets in sequence (temporal_modules = Identity -> skipped), then ONE
# upsampler if present. HAS_SHORTCUT gates resnet 0's channel change; HAS_UP
# gates the upsampler (SR,TR its spatial/temporal ratios).
def up_block3d[
    HAS_SHORTCUT: Bool, HAS_UP: Bool, SR: Int, TR: Int
](h_in: Tensor, st: SafeTensors, block_idx: Int, ctx: DeviceContext) raises -> Tensor:
    var base = String("decoder.up_blocks.") + String(block_idx)
    var h = _resnet3d_st(h_in, st, base + ".resnets.0", HAS_SHORTCUT, ctx)
    h = _resnet3d_st(h, st, base + ".resnets.1", False, ctx)
    h = _resnet3d_st(h, st, base + ".resnets.2", False, ctx)

    comptime if HAS_UP:
        var up_w = load_bias(st, base + ".upsamplers.0.upscale_conv.weight", ctx)
        var up_b = load_bias(st, base + ".upsamplers.0.upscale_conv.bias", ctx)
        var conv_w = load_bias(st, base + ".upsamplers.0.conv.weight", ctx)
        var conv_b = load_bias(st, base + ".upsamplers.0.conv.bias", ctx)
        h = upsample3d[SR, TR](h, up_w, up_b, conv_w, conv_b, ctx)

    return h^


# ── decode_seedvr2_vae: full Decoder3D e2e ────────────────────────────────────
# latent: NCDHW [1,16,4,16,16] -> returns NCDHW [1,3,13,128,128].
def decode_seedvr2_vae(latent: Tensor, st: SafeTensors, ctx: DeviceContext) raises -> Tensor:
    var ci_w = load_bias(st, "decoder.conv_in.weight", ctx)
    var ci_b = load_bias(st, "decoder.conv_in.bias", ctx)
    var h = causal_conv3d[3, 3, 3](latent, ci_w, ci_b^, ctx)   # 16 -> 512

    h = mid_block3d(h, st, ctx)
    h = up_block3d[False, True, 2, 2](h, st, 0, ctx)   # 512->512, up SR2 TR2
    h = up_block3d[False, True, 2, 2](h, st, 1, ctx)   # 512->512, up SR2 TR2
    h = up_block3d[True, True, 2, 1](h, st, 2, ctx)    # 512->256, up SR2 TR1
    h = up_block3d[True, False, 2, 1](h, st, 3, ctx)   # 256->128, no upsampler

    var no_w = load_bias(st, "decoder.conv_norm_out.weight", ctx)
    var no_b = load_bias(st, "decoder.conv_norm_out.bias", ctx)
    h = _group_norm_per_frame(h, no_w, no_b, ctx)      # GN-32, 128 ch, eps 1e-6
    h = silu(h, ctx)

    var co_w = load_bias(st, "decoder.conv_out.weight", ctx)
    var co_b = load_bias(st, "decoder.conv_out.bias", ctx)
    return causal_conv3d[3, 3, 3](h, co_w, co_b^, ctx)         # 128 -> 3
