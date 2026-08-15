# serenitymojo/models/minimax_h3/h3_final_train.mojo
#
# MiniMax-H3 FINAL LAYER training twin: forward + hand-chained backward for
# the d_x handoff into block 49. Heads stay FROZEN (no weight grads) — the
# backward returns only d_hidden. Oracle: musubi akane/minimax-h3 @ 04324c28
# model.py:400-433 (MiniMaxH3FinalLayer.forward):
#
#   shift, scale = adaln_proj(temb).chunk(2)      (mod TABLE precomputed here
#                                                  — the modcache contract;
#                                                  cols [0,D)=shift, [D,2D)=scale)
#   media   = h[media_indices]                     media = video ++ audio rows
#   n       = rms_norm(media, norm_w)
#   m       = n * (1 + scale[ts_idx]) + shift[ts_idx]
#   video   = video_out(m[:Sv]) + b                [Sv, 24*patch]
#   audio   = audio_out(m[Sv:]) + b                [Sa, audio_ch]
#
# Text rows never reach the loss: d_hidden scatters media grads back to
# their sequence positions and leaves text rows exactly zero.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.tensor_algebra import (
    mul, add, add_scalar, slice, concat, gather_rows,
)
from serenitymojo.ops.shape_backward import index_select_backward

comptime TArc = ArcPointer[Tensor]


struct H3FinalTrainWeights(Copyable, Movable):
    var norm_w: TArc        # [D]
    var video_out_w: TArc   # [Pv, D]
    var video_out_b: TArc   # [Pv]
    var audio_out_w: TArc   # [Pa, D]
    var audio_out_b: TArc   # [Pa]

    def __init__(
        out self,
        var norm_w: TArc,
        var video_out_w: TArc, var video_out_b: TArc,
        var audio_out_w: TArc, var audio_out_b: TArc,
    ):
        self.norm_w = norm_w^
        self.video_out_w = video_out_w^
        self.video_out_b = video_out_b^
        self.audio_out_w = audio_out_w^
        self.audio_out_b = audio_out_b^


struct H3FinalTrainSaved(Copyable, Movable):
    var media: TArc        # [Sm, D] gathered pre-norm rows
    var scale_rows: TArc   # [Sm, D] gathered scale
    var m: TArc            # [Sm, D] modulated rows (linear inputs)

    def __init__(out self, var media: TArc, var scale_rows: TArc, var m: TArc):
        self.media = media^
        self.scale_rows = scale_rows^
        self.m = m^


struct H3FinalTrainForward(Copyable, Movable):
    var video: TArc        # [Sv, Pv]
    var audio: TArc        # [Sa, Pa]
    var saved: H3FinalTrainSaved

    def __init__(
        out self, var video: TArc, var audio: TArc, var saved: H3FinalTrainSaved
    ):
        self.video = video^
        self.audio = audio^
        self.saved = saved^


def _media_ts_indices(
    ts_idx: List[Int], video_idx: List[Int], audio_idx: List[Int]
) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(video_idx)):
        out.append(ts_idx[video_idx[i]])
    for i in range(len(audio_idx)):
        out.append(ts_idx[audio_idx[i]])
    return out^


def _media_indices(
    video_idx: List[Int], audio_idx: List[Int]
) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(video_idx)):
        out.append(video_idx[i])
    for i in range(len(audio_idx)):
        out.append(audio_idx[i])
    return out^


def h3_final_train_forward(
    h: Tensor,              # [S, D]
    w: H3FinalTrainWeights,
    mod: Tensor,            # [rows, 2D] adaln table: [0,D)=shift, [D,2D)=scale
    ts_idx: List[Int],      # [S] timestep-table row per sequence position
    video_idx: List[Int],
    audio_idx: List[Int],
    eps: Float32,
    ctx: DeviceContext,
) raises -> H3FinalTrainForward:
    var D = h.shape()[1]
    var sv = len(video_idx)
    var sa = len(audio_idx)
    var midx = _media_indices(video_idx, audio_idx)
    var mts = _media_ts_indices(ts_idx, video_idx, audio_idx)

    var media = gather_rows(h, midx, ctx)                  # [Sm, D]
    var n = rms_norm(media, w.norm_w[], eps, ctx)
    var mod_rows = gather_rows(mod, mts, ctx)              # [Sm, 2D]
    var shift = slice(mod_rows, 1, 0, D, ctx)
    var scale = slice(mod_rows, 1, D, D, ctx)
    var sp1 = add_scalar(scale, Float32(1.0), ctx)
    var m = add(mul(n, sp1, ctx), shift, ctx)              # [Sm, D]

    var mv = slice(m, 0, 0, sv, ctx)
    var ma = slice(m, 0, sv, sa, ctx)
    var video = linear_bias(mv, w.video_out_w[], w.video_out_b[], ctx)
    var audio = linear_bias(ma, w.audio_out_w[], w.audio_out_b[], ctx)

    return H3FinalTrainForward(
        TArc(video^), TArc(audio^),
        H3FinalTrainSaved(TArc(media^), TArc(scale^), TArc(m^)),
    )


def h3_final_train_backward(
    d_video: Tensor,        # [Sv, Pv]
    d_audio: Tensor,        # [Sa, Pa]
    saved: H3FinalTrainSaved,
    w: H3FinalTrainWeights,
    ts_idx: List[Int],
    video_idx: List[Int],
    audio_idx: List[Int],
    S: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """d_hidden [S, D]. Heads + norm + adaln FROZEN: weight grads dropped."""
    var D = saved.media[].shape()[1]
    var sv = len(video_idx)
    var sa = len(audio_idx)
    var pv = d_video.shape()[1]
    var pa = d_audio.shape()[1]

    var mv = slice(saved.m[], 0, 0, sv, ctx)
    var ma = slice(saved.m[], 0, sv, sa, ctx)
    var gv = linear_backward(d_video, mv, w.video_out_w[], sv, D, pv, ctx)
    var ga = linear_backward(d_audio, ma, w.audio_out_w[], sa, D, pa, ctx)
    var d_m = concat(0, ctx, gv.d_x, ga.d_x)               # [Sm, D]

    # m = n*(1+scale) + shift  =>  d_n = d_m * (1+scale)
    var sp1 = add_scalar(saved.scale_rows[], Float32(1.0), ctx)
    var d_n = mul(d_m, sp1, ctx)
    var d_media = rms_norm_backward_dx(d_n, saved.media[], w.norm_w[], eps, ctx)

    var midx = _media_indices(video_idx, audio_idx)
    var in_shape: List[Int] = [S, D]
    return index_select_backward(d_media, midx, 0, in_shape^, ctx)
