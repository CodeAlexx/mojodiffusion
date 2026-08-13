# SCAIL-2 visual conditioning (standalone production process).
#
# argv:
#   scail2_encode_clip <visual_checkpoint.safetensors>
#                      <stage.safetensors> <output.safetensors>
# Input: clip_pixel [1,3,224,224] F32, already creator-preprocessed.
# Output: clip_context [1,257,1280] F16.

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.scail2.scail2_clip_vision import Scail2ClipVision
from serenitymojo.models.scail2.scail2_manifest import (
    write_scail2_condition_manifest,
)
from serenitymojo.tensor import Tensor


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error(
            "usage: scail2_encode_clip <visual_checkpoint.safetensors> "
            "<stage.safetensors> <output.safetensors>"
        )
    var visual_path = String(args[1])
    var stage_path = String(args[2])
    var output_path = String(args[3])
    if output_path == visual_path or output_path == stage_path:
        raise Error("SCAIL-2 CLIP output must not overwrite an input artifact")

    var staged = ShardedSafeTensors.open(stage_path)
    var input_key = String("clip_pixel")
    if not staged.has_tensor(input_key):
        raise Error("SCAIL-2 staged input is missing clip_pixel")
    var view = staged.tensor_view(input_key)
    if view.dtype != STDtype.F32:
        raise Error("SCAIL-2 clip_pixel must be F32")
    if view.shape != [1, 3, 224, 224]:
        raise Error("SCAIL-2 clip_pixel must be [1,3,224,224]")

    var ctx = DeviceContext()
    var pixel = Tensor.from_view(view, ctx)
    var model = Scail2ClipVision.load(visual_path, ctx)
    var output = model.forward(pixel, ctx)
    if output.dtype() != STDtype.F16:
        raise Error("SCAIL-2 clip_context must be F16")
    if output.shape() != [1, 257, 1280]:
        raise Error("SCAIL-2 clip_context must be [1,257,1280]")
    var host = output.to_host(ctx)
    var minimum = host[0]
    var maximum = host[0]
    for value in host:
        if not (value == value) or value > Float32(65000.0) or value < Float32(-65000.0):
            raise Error("SCAIL-2 clip_context contains a non-finite value")
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value

    var names = List[String]()
    names.append(String("clip_context"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(output^))
    save_safetensors(names, tensors, output_path, ctx)
    _ = write_scail2_condition_manifest(
        String(args[0]), String("clip"), stage_path, visual_path, output_path
    )
    print("SCAIL-2 CLIP conditioning PASS")
    print("  shape=[1,257,1280] dtype=F16 range=[", minimum, ",", maximum, "]")
    print("  output=", output_path)
