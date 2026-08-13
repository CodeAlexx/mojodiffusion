# Production-Mojo resize parity dump for the pinned creator inputs.

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.scail2.scail2_preprocess import (
    scail2_hwc_f32_to_chw,
    scail2_resize_center_f32,
    scail2_resize_center_u8,
)
from serenitymojo.image.decode import decode_image
from serenitymojo.tensor import Tensor


comptime H = 512
comptime W = 896


def main() raises:
    var args = argv()
    if len(args) != 6:
        raise Error(
            "usage: scail2_preprocess_parity "
            "<reference> <frame0> <frame32> <frame64> <output>"
        )
    var reference_hwc = scail2_resize_center_f32(
        decode_image(String(args[1]), drop_alpha=True), H, W
    )
    var reference = scail2_hwc_f32_to_chw(reference_hwc, H, W)
    var frames = List[Float32]()
    frames.resize(3 * 3 * H * W, Float32(0.0))
    for fi in range(3):
        var pixel = scail2_resize_center_u8(
            decode_image(String(args[2 + fi]), drop_alpha=True), H, W
        )
        for y in range(H):
            for x in range(W):
                for c in range(3):
                    frames[((fi * 3 + c) * H + y) * W + x] = (
                        Float32(Int(pixel[(y * W + x) * 3 + c])) - 127.5
                    ) / 127.5
    var ctx = DeviceContext()
    var reference_tensor = Tensor.from_host(
        reference, [1, 3, 1, H, W], STDtype.F32, ctx
    )
    var frame_tensor = Tensor.from_host(
        frames, [3, 3, H, W], STDtype.F32, ctx
    )
    var names = List[String]()
    names.append(String("reference_pixel"))
    names.append(String("pose_signed"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(reference_tensor^))
    tensors.append(ArcPointer(frame_tensor^))
    save_safetensors(names, tensors, String(args[5]), ctx)
    print("GATE wrote SCAIL-2 Mojo preprocess parity:", String(args[5]))
