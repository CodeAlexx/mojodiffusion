# GPU-only finite-value probe for saved MiniMax-H3 row-space latents.
#
# The tensor scan and sum-of-squares reduction run on the GPU.  Only the two
# scalar summaries (norm and non-finite count) are copied back to the host.

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.training.on_device_global_norm import on_device_grad_stats


def _load(
    ref st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _probe(name: String, var tensor: Tensor, ctx: DeviceContext) raises:
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer[Tensor](tensor^))
    var stats = on_device_grad_stats(tensors, ctx)
    print(
        name,
        "shape=", tensors[0][].shape(),
        "dtype=", tensors[0][].dtype().name(),
        "l2=", stats.grad_norm,
        "nonfinite=", stats.nonfinite_count,
    )


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: minimax_h3_latent_finite_probe <latents.safetensors>")
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(args[1]))
    _probe("video_state_rows", _load(st, "video_state_rows", ctx), ctx)
    _probe("audio_state_rows", _load(st, "audio_state_rows", ctx), ctx)
