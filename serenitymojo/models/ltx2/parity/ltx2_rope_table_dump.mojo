# ltx2_rope_table_dump.mojo — dump the TRAINING head's video-RoPE cos/sin
# tables for the parity comparison vs musubi's own rope functions
# (scripts/ltx2_rope_table_compare.py). Stage-isolation probe for the
# 2026-07-16 video-arm fwd-parity FAIL (cos 0.9958-0.9991 vs bar 0.999).
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/ltx2/parity/ltx2_rope_table_dump.mojo \
#     -o /tmp/ltx2_rope_dump
#   /tmp/ltx2_rope_dump <out.safetensors>
#
# Emits: video arm (4,9,16)@fps25 and image arm (1,16,16)@fps25 tables,
# F32, halfsplit (token,head)-row layout [P*32, 64].

from std.sys import argv
from std.collections import List
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.ltx2.ltx2_video_stack import _build_video_rope


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: <out.safetensors>")
    var out_path = String(args[1])

    var ctx = DeviceContext()
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    var vid = _build_video_rope(4, 9, 16, Float64(25.0), STDtype.F32, ctx)
    names.append(String("video_cos"))
    tensors.append(ArcPointer[Tensor](vid[0].clone(ctx)))
    names.append(String("video_sin"))
    tensors.append(ArcPointer[Tensor](vid[1].clone(ctx)))

    var img = _build_video_rope(1, 16, 16, Float64(25.0), STDtype.F32, ctx)
    names.append(String("image_cos"))
    tensors.append(ArcPointer[Tensor](img[0].clone(ctx)))
    names.append(String("image_sin"))
    tensors.append(ArcPointer[Tensor](img[1].clone(ctx)))

    save_safetensors(names, tensors, out_path, ctx)
    print("[rope-dump] DONE ->", out_path)
