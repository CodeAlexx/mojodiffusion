# models/dit/nava_embed.mojo — NAVA chunk-3 pre-embedding paths.
#
# Four independent sub-paths that transform raw latents + text + timestep into
# transformer-block inputs. Mirrors prepare_transformer_block_kwargs in
# /home/alex/EriDiffusion/inference-flame/ports/nava/nava_src/models/nava/modules/model_mm.py
#
# (A) video patch-embed   → x_vid   [1, 320, 3072]
# (B) audio patch-embed   → x_audio [1, 34, 3072]
# (C) text embed          → context [1, 512, 3072]
# (D) time embed          → e0      [1, 1, 6, 3072]  (single token; broadcast to
#                                                      full seq by caller)
#
# All weights are BF16 in NAVA_fp8.safetensors under prefix "backbone.".
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.conv1d import conv1d
from serenitymojo.ops.patchify3d import patchify3d
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.activations import silu, gelu
from serenitymojo.ops.tensor_algebra import (
    reshape, permute, concat, slice, zeros_device, mul,
)


# ── Weight store ─────────────────────────────────────────────────────────────
# Dict[String, ArcPointer[Tensor]] — same pattern as nava_block.mojo.
# Access weight directly:    w["key"][]          (borrowed reference to Tensor)
# Access bias as Optional:   Optional(w["key"][].clone(ctx))


def load_nava_embed_weights(
    st: ShardedSafeTensors,
    prefix: String,
    ctx: DeviceContext,
) raises -> Dict[String, ArcPointer[Tensor]]:
    """Load all BF16 embed weights from the safetensors file.

    prefix: e.g. "backbone."  — all keys below are relative to this prefix.
    Stores the following short keys:
      "pe.weight"      [3072, 48, 1, 2, 2]
      "pe.bias"        [3072]
      "pe_aud0.weight" [3072, 128, 7]
      "pe_aud0.bias"   [3072]
      "pe_w1.weight"   [8192, 3072, 7]
      "pe_w2.weight"   [3072, 8192, 7]
      "pe_w3.weight"   [8192, 3072, 7]
      "te0.weight"     [3072, 4096]
      "te0.bias"       [3072]
      "te2.weight"     [3072, 3072]
      "te2.bias"       [3072]
      "tme0.weight"    [3072, 256]
      "tme0.bias"      [3072]
      "tme2.weight"    [3072, 3072]
      "tme2.bias"      [3072]
      "tp1.weight"     [18432, 3072]
      "tp1.bias"       [18432]
    """
    var w = Dict[String, ArcPointer[Tensor]]()

    var t0 = Tensor.from_view(st.tensor_view(prefix + "patch_embedding.weight"), ctx)
    w["pe.weight"] = ArcPointer(t0^)
    var t1 = Tensor.from_view(st.tensor_view(prefix + "patch_embedding.bias"), ctx)
    w["pe.bias"] = ArcPointer(t1^)

    var t2 = Tensor.from_view(st.tensor_view(prefix + "patch_embedding_audio.0.weight"), ctx)
    w["pe_aud0.weight"] = ArcPointer(t2^)
    var t3 = Tensor.from_view(st.tensor_view(prefix + "patch_embedding_audio.0.bias"), ctx)
    w["pe_aud0.bias"] = ArcPointer(t3^)

    var t4 = Tensor.from_view(st.tensor_view(prefix + "patch_embedding_audio.2.w1.weight"), ctx)
    w["pe_w1.weight"] = ArcPointer(t4^)
    var t5 = Tensor.from_view(st.tensor_view(prefix + "patch_embedding_audio.2.w2.weight"), ctx)
    w["pe_w2.weight"] = ArcPointer(t5^)
    var t6 = Tensor.from_view(st.tensor_view(prefix + "patch_embedding_audio.2.w3.weight"), ctx)
    w["pe_w3.weight"] = ArcPointer(t6^)

    var t7 = Tensor.from_view(st.tensor_view(prefix + "text_embedding.0.weight"), ctx)
    w["te0.weight"] = ArcPointer(t7^)
    var t8 = Tensor.from_view(st.tensor_view(prefix + "text_embedding.0.bias"), ctx)
    w["te0.bias"] = ArcPointer(t8^)
    var t9 = Tensor.from_view(st.tensor_view(prefix + "text_embedding.2.weight"), ctx)
    w["te2.weight"] = ArcPointer(t9^)
    var t10 = Tensor.from_view(st.tensor_view(prefix + "text_embedding.2.bias"), ctx)
    w["te2.bias"] = ArcPointer(t10^)

    var t11 = Tensor.from_view(st.tensor_view(prefix + "time_embedding.0.weight"), ctx)
    w["tme0.weight"] = ArcPointer(t11^)
    var t12 = Tensor.from_view(st.tensor_view(prefix + "time_embedding.0.bias"), ctx)
    w["tme0.bias"] = ArcPointer(t12^)
    var t13 = Tensor.from_view(st.tensor_view(prefix + "time_embedding.2.weight"), ctx)
    w["tme2.weight"] = ArcPointer(t13^)
    var t14 = Tensor.from_view(st.tensor_view(prefix + "time_embedding.2.bias"), ctx)
    w["tme2.bias"] = ArcPointer(t14^)

    var t15 = Tensor.from_view(st.tensor_view(prefix + "time_projection.1.weight"), ctx)
    w["tp1.weight"] = ArcPointer(t15^)
    var t16 = Tensor.from_view(st.tensor_view(prefix + "time_projection.1.bias"), ctx)
    w["tp1.bias"] = ArcPointer(t16^)

    return w^


# ── (A) Video patch-embed ─────────────────────────────────────────────────────
def nava_video_patch_embed(
    in_lat_vid: Tensor,  # [1280, 48] BF16
    w: Dict[String, ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises -> Tensor:
    """Conv3d patch-embed: [1280,48] → [1,320,3072].

    in_lat_vid must be BF16. Shape is [F*H*W, C] = [1280, 48] with F=5,H=16,W=16.
    Reshapes → [5,16,16,48] → permute(3,0,1,2)=[48,5,16,16] (CFHW) →
    patchify3d(1,2,2) → [320,192] → linear(pe_w[3072,192]) → [320,3072] →
    reshape → [1,320,3072].
    """
    # [1280,48] → [5,16,16,48]
    var shape_fhwc = List[Int]()
    shape_fhwc.append(5)
    shape_fhwc.append(16)
    shape_fhwc.append(16)
    shape_fhwc.append(48)
    var lat_fhwc = reshape(in_lat_vid, shape_fhwc^, ctx)  # [5,16,16,48]

    # permute (F,H,W,C)→(C,F,H,W): perm [3,0,1,2]
    var perm_cfhw = List[Int]()
    perm_cfhw.append(3)
    perm_cfhw.append(0)
    perm_cfhw.append(1)
    perm_cfhw.append(2)
    var lat_cfhw = permute(lat_fhwc, perm_cfhw, ctx)  # [48,5,16,16]

    # patchify3d: [48,5,16,16] → [320, 192]  (192 = 48*1*2*2)
    var patches = patchify3d(lat_cfhw, 1, 2, 2, ctx)  # [320,192]

    # Flatten conv weight [3072,48,1,2,2] → [3072,192]
    var pe_w_shape = List[Int]()
    pe_w_shape.append(3072)
    pe_w_shape.append(192)
    var pe_w = reshape(w["pe.weight"][], pe_w_shape^, ctx)  # [3072,192]

    # linear: [320,192] @ [3072,192]^T + bias[3072] → [320,3072]
    var out_flat = linear(patches, pe_w, Optional(w["pe.bias"][].clone(ctx)), ctx)  # [320,3072]

    # reshape → [1,320,3072]
    var out_shape = List[Int]()
    out_shape.append(1)
    out_shape.append(320)
    out_shape.append(3072)
    return reshape(out_flat, out_shape^, ctx)


# ── (A2) Video patch-embed hi-res (832×480) ──────────────────────────────────
def nava_video_patch_embed_hires(
    in_lat_vid: Tensor,  # [7800, 48] BF16
    w: Dict[String, ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises -> Tensor:
    """Conv3d patch-embed hi-res: [7800,48] → [1,1950,3072].

    in_lat_vid must be BF16. Shape is [F*Hlat*Wlat, C] = [7800, 48] with
    F=5, Hlat=30, Wlat=52. Patch size [1,2,2] → Hp=15, Wp=26 → VID=1950.
    Reshapes → [5,30,52,48] → permute(3,0,1,2)=[48,5,30,52] (CFHW) →
    patchify3d(1,2,2) → [1950,192] → linear(pe_w[3072,192]) → [1950,3072] →
    reshape → [1,1950,3072].
    """
    # [7800,48] → [5,30,52,48]
    var shape_fhwc = List[Int]()
    shape_fhwc.append(5)
    shape_fhwc.append(30)
    shape_fhwc.append(52)
    shape_fhwc.append(48)
    var lat_fhwc = reshape(in_lat_vid, shape_fhwc^, ctx)  # [5,30,52,48]

    # permute (F,H,W,C)→(C,F,H,W): perm [3,0,1,2]
    var perm_cfhw = List[Int]()
    perm_cfhw.append(3)
    perm_cfhw.append(0)
    perm_cfhw.append(1)
    perm_cfhw.append(2)
    var lat_cfhw = permute(lat_fhwc, perm_cfhw, ctx)  # [48,5,30,52]

    # patchify3d: [48,5,30,52] → [1950, 192]  (192 = 48*1*2*2)
    var patches = patchify3d(lat_cfhw, 1, 2, 2, ctx)  # [1950,192]

    # Flatten conv weight [3072,48,1,2,2] → [3072,192]
    var pe_w_shape = List[Int]()
    pe_w_shape.append(3072)
    pe_w_shape.append(192)
    var pe_w = reshape(w["pe.weight"][], pe_w_shape^, ctx)  # [3072,192]

    # linear: [1950,192] @ [3072,192]^T + bias[3072] → [1950,3072]
    var out_flat = linear(patches, pe_w, Optional(w["pe.bias"][].clone(ctx)), ctx)  # [1950,3072]

    # reshape → [1,1950,3072]
    var out_shape = List[Int]()
    out_shape.append(1)
    out_shape.append(1950)
    out_shape.append(3072)
    return reshape(out_flat, out_shape^, ctx)


# ── (B) Audio patch-embed ─────────────────────────────────────────────────────
def nava_audio_patch_embed(
    in_lat_aud: Tensor,  # [34, 128] BF16
    w: Dict[String, ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises -> Tensor:
    """Audio Sequential patch-embed: [34,128] → [1,34,3072].

    Reference: patch_embedding_audio = Sequential(
      ChannelLastConv1d(128,3072,k=7,pad=3), SiLU,
      ConvMLP(3072, 3072*4=8192, k=7,pad=3)  [SwiGLU conv: w2(silu(w1(x))*w3(x))]
    )
    ChannelLastConv1d: permute [B,L,C]→[B,C,L], Conv1d, permute back.
    ConvMLP: w2( silu(w1(x)) * w3(x) ) where w1/w2/w3 are ChannelLastConv1d no bias.
    """
    # [34,128] → [1,34,128]
    var shape1 = List[Int]()
    shape1.append(1)
    shape1.append(34)
    shape1.append(128)
    var x = reshape(in_lat_aud, shape1^, ctx)  # [1,34,128]

    # ChannelLastConv1d: permute [B,L,C]→[B,C,L]
    var perm_to_ncl = List[Int]()
    perm_to_ncl.append(0)
    perm_to_ncl.append(2)
    perm_to_ncl.append(1)
    var x_ncl = permute(x, perm_to_ncl, ctx)  # [1,128,34]

    # Conv1d(128,3072,k=7,pad=3,stride=1,dil=1,groups=1) → [1,3072,34]
    var h_ncl = conv1d(
        x_ncl, w["pe_aud0.weight"][], Optional(w["pe_aud0.bias"][].clone(ctx)),
        1, 3, 1, 1, ctx
    )  # [1,3072,34]

    # permute back [B,C,L]→[B,L,C]
    var perm_to_nlc = List[Int]()
    perm_to_nlc.append(0)
    perm_to_nlc.append(2)
    perm_to_nlc.append(1)
    var h = permute(h_ncl, perm_to_nlc, ctx)  # [1,34,3072]

    # SiLU
    var h_silu = silu(h, ctx)  # [1,34,3072]

    # ConvMLP: permute h → [1,3072,34] for conv1d
    var perm_to_ncl2 = List[Int]()
    perm_to_ncl2.append(0)
    perm_to_ncl2.append(2)
    perm_to_ncl2.append(1)
    var h_clt = permute(h_silu, perm_to_ncl2, ctx)  # [1,3072,34]

    # a = conv1d(h_clt, w1, None, 1,3,1,1) → [1,8192,34]
    var a = conv1d(
        h_clt, w["pe_w1.weight"][], Optional[Tensor](None),
        1, 3, 1, 1, ctx
    )  # [1,8192,34]

    # c = conv1d(h_clt, w3, None, 1,3,1,1) → [1,8192,34]
    var c = conv1d(
        h_clt, w["pe_w3.weight"][], Optional[Tensor](None),
        1, 3, 1, 1, ctx
    )  # [1,8192,34]

    # g = silu(a) * c → [1,8192,34]
    var g = mul(silu(a, ctx), c, ctx)  # [1,8192,34]

    # out = conv1d(g, w2, None, 1,3,1,1) → [1,3072,34]
    var out_ncl = conv1d(
        g, w["pe_w2.weight"][], Optional[Tensor](None),
        1, 3, 1, 1, ctx
    )  # [1,3072,34]

    # permute → [1,34,3072]
    var perm_to_nlc2 = List[Int]()
    perm_to_nlc2.append(0)
    perm_to_nlc2.append(2)
    perm_to_nlc2.append(1)
    return permute(out_ncl, perm_to_nlc2, ctx)  # [1,34,3072]


# ── (C) Text embed ────────────────────────────────────────────────────────────
def nava_text_embed(
    in_text: Tensor,  # [actual_len, 4096] BF16
    w: Dict[String, ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises -> Tensor:
    """Text embedding: pad to [512,4096] → Linear → GELU(tanh) → Linear → [1,512,3072]."""
    comptime text_len = 512
    var actual_len = in_text.shape()[0]
    var pad_len = text_len - actual_len

    # Pad rows [actual_len..511] with zeros → [512,4096]
    var padded: Tensor
    if pad_len > 0:
        var pad_shape = List[Int]()
        pad_shape.append(pad_len)
        pad_shape.append(4096)
        var zeros = zeros_device(pad_shape^, in_text.dtype(), ctx)
        padded = concat(0, ctx, in_text, zeros)  # [512,4096]
    else:
        # No padding needed; clone to own a fresh tensor
        var same_shape = List[Int]()
        same_shape.append(text_len)
        same_shape.append(4096)
        padded = reshape(in_text, same_shape^, ctx)

    # Reshape to [1,512,4096]
    var shape_3d = List[Int]()
    shape_3d.append(1)
    shape_3d.append(text_len)
    shape_3d.append(4096)
    var x = reshape(padded, shape_3d^, ctx)  # [1,512,4096]

    # Linear(4096,3072) → [1,512,3072]
    var h = linear(x, w["te0.weight"][], Optional(w["te0.bias"][].clone(ctx)), ctx)

    # GELU (tanh-approx)
    var h_act = gelu(h, ctx)

    # Linear(3072,3072) → [1,512,3072]
    return linear(h_act, w["te2.weight"][], Optional(w["te2.bias"][].clone(ctx)), ctx)


# ── (D) Time embed ────────────────────────────────────────────────────────────
def nava_time_embed(
    in_t: Tensor,  # [1] F32  (scalar timestep, e.g. 500.0)
    w: Dict[String, ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises -> Tensor:
    """Time embedding → time-projection: t[1] → e0 [1,1,6,3072].

    Reference:
      e = time_embedding( sinusoidal_embedding_1d(256, t) )
            = Linear(256,3072) → SiLU → Linear(3072,3072)
      e0 = time_projection(e)    # Sequential: SiLU → Linear(3072,18432)
            .unflatten(2,(6,3072)) → [1,1,6,3072]
    We return the single-token e0 [1,1,6,3072]; caller broadcasts to full seq.
    """
    # sinusoidal_embedding_1d(256, t): COS first, matches timestep_embedding → [1,256] BF16
    var se = timestep_embedding(in_t, 256, ctx, Float32(10000.0), STDtype.BF16)  # [1,256]

    # time_embedding: Linear(256,3072) → SiLU → Linear(3072,3072)
    var e = linear(se, w["tme0.weight"][], Optional(w["tme0.bias"][].clone(ctx)), ctx)  # [1,3072]
    var e_silu = silu(e, ctx)  # [1,3072]
    var e2 = linear(e_silu, w["tme2.weight"][], Optional(w["tme2.bias"][].clone(ctx)), ctx)  # [1,3072]

    # time_projection: SiLU → Linear(3072,18432)
    var e2_silu = silu(e2, ctx)  # [1,3072]
    var e_proj = linear(e2_silu, w["tp1.weight"][], Optional(w["tp1.bias"][].clone(ctx)), ctx)  # [1,18432]

    # reshape [1,18432] → [1,1,6,3072]
    var e0_shape = List[Int]()
    e0_shape.append(1)
    e0_shape.append(1)
    e0_shape.append(6)
    e0_shape.append(3072)
    return reshape(e_proj, e0_shape^, ctx)  # [1,1,6,3072]
