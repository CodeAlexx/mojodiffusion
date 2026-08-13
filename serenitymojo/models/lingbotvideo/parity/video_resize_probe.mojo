# video_resize_probe.mojo — smoke for resize_video_bicubic.
# Upscales a synthetic [1,3,3,8,8] clip in [0,1] to [1,3,3,16,16] and prints
# in/out shapes + that all values stay in [0,1].
#
# Run:
#   rm -f serenitymojo.mojopkg && pixi run mojo run -I . \
#     serenitymojo/models/lingbotvideo/parity/video_resize_probe.mojo

from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.lingbotvideo.lingbot_image_preprocess import (
    resize_video_bicubic,
)


def main() raises:
    var ctx = DeviceContext()
    comptime T = 3
    comptime IH = 8
    comptime IW = 8
    comptime OH = 16
    comptime OW = 16

    # synthetic clip in [0,1]: smooth ramp per (c,t,h,w)
    var vals = List[Float32]()
    vals.resize(3 * T * IH * IW, Float32(0.0))
    for c in range(3):
        for t in range(T):
            for h in range(IH):
                for w in range(IW):
                    var idx = ((c * T + t) * IH + h) * IW + w
                    var f = Float32(c + t + h + w) / Float32(2 + T + IH + IW)
                    vals[idx] = f
    var clip = Tensor.from_host(vals, [1, 3, T, IH, IW], STDtype.F32, ctx)
    var cs = clip.shape()
    print("[VRESIZE] in shape [", cs[0], cs[1], cs[2], cs[3], cs[4], "]")

    var up = resize_video_bicubic(clip, OH, OW, ctx)
    var us = up.shape()
    print("[VRESIZE] out shape [", us[0], us[1], us[2], us[3], us[4], "]")

    var host = up.to_host(ctx)
    var mn = Float32(1.0e30)
    var mx = Float32(-1.0e30)
    var in_range = True
    for i in range(len(host)):
        var v = host[i]
        if v < mn:
            mn = v
        if v > mx:
            mx = v
        if v < 0.0 or v > 1.0:
            in_range = False
    print("[VRESIZE] min", mn, "max", mx, "all_in_[0,1]", in_range)
    if not in_range:
        raise Error("resize_video_bicubic produced values outside [0,1]")
    print("[VRESIZE] OK")
