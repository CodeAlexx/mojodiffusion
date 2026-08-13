# Compile-time reachability gate for the full 31-block SCAIL-2 visual tower.
# Running it requires a converted official visual safetensors file and executes
# one zero-image forward; `mojo build` is sufficient for the bounded build gate.

from max.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.scail2.scail2_clip_vision import Scail2ClipVision


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: scail2_clip_compile <official_visual.safetensors>")
    var ctx = DeviceContext()
    var values = List[Float32]()
    values.resize(1 * 3 * 224 * 224, Float32(0.0))
    var image = Tensor.from_host(values, [1, 3, 224, 224], STDtype.F32, ctx)
    var model = Scail2ClipVision.load(args[1], ctx)
    var output = model.forward(image, ctx)
    var shape = output.shape()
    if len(shape) != 3 or shape[0] != 1 or shape[1] != 257 or shape[2] != 1280:
        raise Error("SCAIL-2 CLIP output must be [1,257,1280]")
    print("SCAIL-2 CLIP full visual forward shape PASS [1,257,1280]")
