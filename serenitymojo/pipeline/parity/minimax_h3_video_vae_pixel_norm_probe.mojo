# pipeline/parity/minimax_h3_video_vae_pixel_norm_probe.mojo — compile+run
# gate for pipeline/minimax_h3_video_vae_pixel_norm.mojo.
#
#   1. Hand-computed ImageNet normalize: known RGB values, checked against
#      (x-mean)/std computed by hand.
#   2. Round-trip: normalize then denormalize must return the ORIGINAL
#      pixel values (bit-close, not just "some value") — this is the
#      property that matters (denormalize is claimed to be the exact
#      inverse; prove it, don't assert it).
#
#   pixi run mojo run -I . pipeline/parity/minimax_h3_video_vae_pixel_norm_probe.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.pipeline.minimax_h3_video_vae_pixel_norm import (
    minimax_h3_pixel_norm_constants, minimax_h3_video_pixel_denormalize,
    minimax_h3_video_pixel_normalize,
)


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("probe: _max_abs_diff length mismatch")
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < Float32(0.0):
            d = -d
        if d > m:
            m = d
    return m


def main() raises:
    var ctx = DeviceContext()
    var constants = minimax_h3_pixel_norm_constants("imagenet")

    # one RGB pixel, [1,1,1,1,3] NDHWC.
    var host = List[Float32]()
    host.append(Float32(0.7))
    host.append(Float32(0.2))
    host.append(Float32(0.9))
    var pixels = Tensor.from_host(host, [1, 1, 1, 1, 3], STDtype.F32, ctx)

    var normed = minimax_h3_video_pixel_normalize(pixels, constants, ctx)
    var normed_host = normed.to_host(ctx)
    var expected = List[Float32]()
    expected.append((Float32(0.7) - Float32(0.485)) / Float32(0.229))
    expected.append((Float32(0.2) - Float32(0.456)) / Float32(0.224))
    expected.append((Float32(0.9) - Float32(0.406)) / Float32(0.225))
    var diff = _max_abs_diff(normed_host, expected)
    print("normalize max_abs vs hand-computed:", diff)
    if diff > Float32(1.0e-5):
        raise Error("probe: FAIL normalize does not match hand-computed ImageNet transform")

    var denormed = minimax_h3_video_pixel_denormalize(normed, constants, ctx)
    var roundtrip_diff = _max_abs_diff(denormed.to_host(ctx), host)
    print("round-trip max_abs (denormalize(normalize(x)) vs x):", roundtrip_diff)
    if roundtrip_diff > Float32(1.0e-4):
        raise Error("probe: FAIL round trip does not recover the original pixel values")

    print("minimax_h3_video_vae_pixel_norm_probe PASS")
