# GPU-only comparison for two saved MiniMax-H3 row-space latent files.
#
# Tensor subtraction and all full-tensor reductions run on the GPU.  Only L2
# norms, non-finite counts, and the cosine reconstructed from those scalar
# norms are copied to the host.

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.tensor_algebra import sub
from serenitymojo.tensor import Tensor
from serenitymojo.training.on_device_global_norm import (
    DeviceGradStats,
    on_device_grad_stats,
)


def _load(
    ref st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _stats(var tensor: Tensor, ctx: DeviceContext) raises -> DeviceGradStats:
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer[Tensor](tensor^))
    return on_device_grad_stats(tensors, ctx)


def _compare(
    name: String,
    var reference: Tensor,
    var candidate: Tensor,
    ctx: DeviceContext,
) raises:
    if reference.shape() != candidate.shape():
        raise Error(name + ": shape mismatch")
    var shape = reference.shape()
    var difference = sub(candidate, reference, ctx)
    var ref_stats = _stats(reference^, ctx)
    var got_stats = _stats(candidate^, ctx)
    var diff_stats = _stats(difference^, ctx)
    var nr = Float64(ref_stats.grad_norm)
    var ng = Float64(got_stats.grad_norm)
    var nd = Float64(diff_stats.grad_norm)
    var dot = (nr * nr + ng * ng - nd * nd) * 0.5
    var cosine = dot / (nr * ng + 1.0e-30)
    var rel_l2 = nd / (nr + 1.0e-30)
    print(
        name,
        "shape=", shape,
        "cos=", cosine,
        "rel_l2=", rel_l2,
        "ref_l2=", nr,
        "got_l2=", ng,
        "diff_l2=", nd,
        "ref_nonfinite=", ref_stats.nonfinite_count,
        "got_nonfinite=", got_stats.nonfinite_count,
        "diff_nonfinite=", diff_stats.nonfinite_count,
    )
    if (
        ref_stats.nonfinite_count != 0
        or got_stats.nonfinite_count != 0
        or diff_stats.nonfinite_count != 0
    ):
        raise Error(name + ": finite gate failed")


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error(
            "usage: minimax_h3_latent_compare_probe"
            " <reference.safetensors> <candidate.safetensors>"
        )
    var ctx = DeviceContext()
    var ref_st = SafeTensors.open(String(args[1]))
    var got_st = SafeTensors.open(String(args[2]))
    _compare(
        "video_state_rows",
        _load(ref_st, "video_state_rows", ctx),
        _load(got_st, "video_state_rows", ctx),
        ctx,
    )
    _compare(
        "audio_state_rows",
        _load(ref_st, "audio_state_rows", ctx),
        _load(got_st, "audio_state_rows", ctx),
        ctx,
    )
