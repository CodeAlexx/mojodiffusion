# SDXL ADM conditioning shared by inference/training-facing callers.
# Contract: SerenityTrainer/diffusers added_cond_kwargs concatenate projected
# CLIP-G pooled text [1280] with six COS-first 256-d embeddings for
# [height, width, crop_top, crop_left, target_height, target_width].

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.tensor_algebra import concat, reshape


comptime SDXL_POOLED_DIM = 1280
comptime SDXL_TIME_ID_COUNT = 6
comptime SDXL_TIME_ID_EMBED_DIM = 256
comptime SDXL_ADM_DIM = 2816


def sdxl_adm_y(
    pooled_clip_g: Tensor, height: Int, width: Int, ctx: DeviceContext
) raises -> Tensor:
    var ps = pooled_clip_g.shape()
    if len(ps) != 2 or ps[0] != 1 or ps[1] != SDXL_POOLED_DIM:
        raise Error("SDXL ADM: projected CLIP-G pooled tensor must be [1,1280]")
    if height <= 0 or width <= 0:
        raise Error("SDXL ADM: height and width must be positive")

    var time_ids = List[Float32]()
    time_ids.append(Float32(height))
    time_ids.append(Float32(width))
    time_ids.append(Float32(0.0))
    time_ids.append(Float32(0.0))
    time_ids.append(Float32(height))
    time_ids.append(Float32(width))
    var time_tensor = Tensor.from_host(
        time_ids, [SDXL_TIME_ID_COUNT], STDtype.F32, ctx
    )
    var time_emb = timestep_embedding(
        time_tensor, SDXL_TIME_ID_EMBED_DIM, ctx, Float32(10000.0), STDtype.BF16
    )
    time_emb = reshape(
        time_emb, [1, SDXL_TIME_ID_COUNT * SDXL_TIME_ID_EMBED_DIM], ctx
    )
    var y = concat(1, ctx, pooled_clip_g, time_emb)
    var ys = y.shape()
    if len(ys) != 2 or ys[0] != 1 or ys[1] != SDXL_ADM_DIM:
        raise Error("SDXL ADM: assembled conditioning must be [1,2816]")
    return y^
