# Real official-checkpoint SCAIL-2 visual-conditioning runner.
#
# argv: scail2_clip_real_run <visual.safetensors> <stage.safetensors>
#                              <output.safetensors>
# Input key: clip_pixel [1,3,224,224] F32, already creator-preprocessed.
# Output key: clip_context [1,257,1280] F16.

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.scail2.scail2_clip_vision import Scail2ClipVision


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error(
            "usage: scail2_clip_real_run <visual.safetensors> "
            "<stage.safetensors> <output.safetensors>"
        )
    var ctx = DeviceContext()
    var staged = ShardedSafeTensors.open(args[2])
    var pixel = Tensor.from_view(staged.tensor_view(String("clip_pixel")), ctx)
    var pixel_shape = pixel.shape()
    if (
        len(pixel_shape) != 4 or pixel_shape[0] != 1 or pixel_shape[1] != 3
        or pixel_shape[2] != 224 or pixel_shape[3] != 224
    ):
        raise Error("staged clip_pixel must be [1,3,224,224]")

    print("SCAIL-2 CLIP: loading official visual checkpoint")
    var model = Scail2ClipVision.load(args[1], ctx)
    print("SCAIL-2 CLIP: running 31-block visual forward")
    var output = model.forward(pixel, ctx)
    var shape = output.shape()
    if len(shape) != 3 or shape[0] != 1 or shape[1] != 257 or shape[2] != 1280:
        raise Error("SCAIL-2 CLIP output must be [1,257,1280]")

    var host = output.to_host(ctx)
    var minimum = host[0]
    var maximum = host[0]
    for value in host:
        if not (value == value):
            raise Error("SCAIL-2 CLIP output contains NaN")
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value

    var names = List[String]()
    names.append(String("clip_context"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(output^))
    save_safetensors(names, tensors, args[3], ctx)
    print("SCAIL-2 CLIP real run PASS")
    print("  shape=[1,257,1280] dtype=F16")
    print("  range=[", minimum, ",", maximum, "]")
    print("  output=", args[3])
