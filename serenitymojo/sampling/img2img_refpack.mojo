# sampling/img2img_refpack.mojo — reference-latent packing for edit / img2img.
#
# NEW standalone module (does not touch any existing sampler/pipeline). Ports
# the WEIGHT-FREE token + position-id packing of the Klein edit path:
#   inference-flame/src/sampling/klein_sampling.rs::prepare_reference_ids
#   inference-flame/src/bin/klein_edit_infer.rs   (noise_img_ids + cat packing)
#
# Convention (Klein/Flux2 latent family):
#   * spatial latent is [1, 128, LH, LW]; "pack" → permute [0,2,3,1] →
#     reshape [1, LH*LW, 128]  (the encode_raw passthrough: latent is already
#     produced by the VAE; packing is the token layout the DiT consumes).
#   * noise_img_ids: [N_img, 4], row r*LW+c = [0,        row, col, 0]
#   * ref_ids:       [N_img, 4], row r*LW+c = [t_offset, row, col, 0]
#       The distinct T-coordinate (typically 10.0) makes RoPE separate the
#       reference tokens from the noise tokens (klein_sampling.rs:155-176).
#   * combined_img_ids = cat([noise_img_ids ; ref_ids], dim 0) → [2*N_img, 4].
#   * combined tokens   = cat([noise_tokens ; ref_tokens], dim 1) → [1, 2*N_img, 128]
#     (reference image latent tokens APPENDED after the noise tokens).
#
# AGENT-DEFAULT (flagged): the ref-id packing convention here is Klein/Flux2's
# [t,row,col,0] 4-vector with reference T-offset and noise T=0, reference tokens
# appended AFTER the noise tokens along the sequence axis. Other arches (e.g.
# Qwen-Image-Edit) use a 3-axis [t,h,w] id and may prepend; this module targets
# the Klein edit convention exactly as in klein_edit_infer.rs.
#
# Mojo 1.0.0b1: `def` not `fn`; move-only Tensor.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import permute, reshape, concat


# --------------------------------------------------------------------------
# encode_raw passthrough: pack a spatial latent [1, C, LH, LW] into the DiT
# token layout [1, LH*LW, C]. (The VAE encode itself is weight-bearing; the
# *packing* is the token-layout transform the edit pipeline applies to the
# already-encoded reference latent.)
# --------------------------------------------------------------------------
def pack_latent_tokens(latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = latent_nchw.shape()
    if len(sh) != 4:
        raise Error("pack_latent_tokens: expected [1,C,LH,LW]")
    var c = sh[1]
    var lh = sh[2]
    var lw = sh[3]
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(1)   # NCHW -> NHWC
    var nhwc = permute(latent_nchw, p^, ctx)
    var out_shape = List[Int]()
    out_shape.append(1); out_shape.append(lh * lw); out_shape.append(c)
    return reshape(nhwc, out_shape^, ctx)


# --------------------------------------------------------------------------
# noise_img_ids: [N_img, 4], row r*LW+c = [0, row, col, 0].
# --------------------------------------------------------------------------
def build_noise_img_ids(
    latent_h: Int, latent_w: Int, ctx: DeviceContext
) raises -> Tensor:
    return _build_grid_ids(latent_h, latent_w, 0.0, ctx)


# --------------------------------------------------------------------------
# prepare_reference_ids: [N_img, 4], row r*LW+c = [t_offset, row, col, 0].
# Direct port of klein_sampling.rs::prepare_reference_ids.
# --------------------------------------------------------------------------
def prepare_reference_ids(
    latent_h: Int, latent_w: Int, t_offset: Float32, ctx: DeviceContext
) raises -> Tensor:
    return _build_grid_ids(latent_h, latent_w, t_offset, ctx)


# Shared grid-id builder: row r*LW+c = [t_coord, row, col, 0].
def _build_grid_ids(
    latent_h: Int, latent_w: Int, t_coord: Float32, ctx: DeviceContext
) raises -> Tensor:
    var n = latent_h * latent_w
    var data = List[Float32]()
    for row in range(latent_h):
        for col in range(latent_w):
            data.append(t_coord)
            data.append(Float32(row))
            data.append(Float32(col))
            data.append(0.0)
    var shape = List[Int]()
    shape.append(n); shape.append(4)
    return Tensor.from_host(data, shape^, STDtype.F32, ctx)


# --------------------------------------------------------------------------
# combined_img_ids = cat([noise_ids ; ref_ids], dim 0) → [2*N_img, 4].
# --------------------------------------------------------------------------
def prepare_combined_img_ids(
    latent_h: Int, latent_w: Int, t_offset: Float32, ctx: DeviceContext
) raises -> Tensor:
    var noise_ids = build_noise_img_ids(latent_h, latent_w, ctx)
    var ref_ids = prepare_reference_ids(latent_h, latent_w, t_offset, ctx)
    return concat(0, ctx, noise_ids, ref_ids)


# --------------------------------------------------------------------------
# combined tokens = cat([noise_tokens ; ref_tokens], dim 1) → [1, 2*N_img, C].
# Reference latent tokens appended AFTER the noise tokens (klein_edit_infer.rs).
# --------------------------------------------------------------------------
def prepare_combined_tokens(
    noise_tokens: Tensor, ref_tokens: Tensor, ctx: DeviceContext
) raises -> Tensor:
    return concat(1, ctx, noise_tokens, ref_tokens)
