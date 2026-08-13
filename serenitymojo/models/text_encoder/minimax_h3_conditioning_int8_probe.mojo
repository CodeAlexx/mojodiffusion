# Real-prompt, full-depth MiniMax-H3 INT8 conditioning probe.

from max.gpu.host import DeviceContext
from std.collections import List
from std.memory import ArcPointer, alloc
from std.sys import argv

from serenitymojo.io.ffi import (
    O_RDONLY, file_size, sys_close, sys_open, sys_pread,
)
from serenitymojo.models.text_encoder.minimax_h3_conditioning import (
    MINIMAX_H3_ENCODER_INT8,
    minimax_h3_encode_conditioning,
)
from serenitymojo.tensor import Tensor
from serenitymojo.training.on_device_global_norm import on_device_grad_stats


def _read_text(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open prompt file: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error("prompt file is empty")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            buf.free()
            _ = sys_close(fd)
            raise Error("failed reading prompt file")
        done += got
    _ = sys_close(fd)
    var out = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=buf, length=n)))
    buf.free()
    return out^


def main() raises:
    var args = argv()
    if len(args) != 4:
        print(
            "usage: minimax_h3_conditioning_int8_probe"
            " <processor_dir> <text_encoder_dir> <prompt_file>"
        )
        return
    var prompt = _read_text(String(args[3]))
    var ctx = DeviceContext()
    var out = minimax_h3_encode_conditioning(
        String(args[1]), String(args[2]), prompt, ctx,
        MINIMAX_H3_ENCODER_INT8,
    )
    var token_count = len(out.token_tags)
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer[Tensor](out.embeds.clone(ctx)))
    var stats = on_device_grad_stats(tensors, ctx)
    ctx.synchronize()
    if stats.nonfinite_count != 0:
        raise Error(
            String("MiniMax-H3 INT8 conditioning produced non-finite values: ")
            + String(stats.nonfinite_count)
        )
    print("PASS: MiniMax-H3 real-prompt INT8 conditioning finite gate")
    print("  tokens:", token_count)
    print(
        "  output:", tensors[0][].shape(), tensors[0][].dtype().name(),
        "l2=", stats.grad_norm, "nonfinite=", stats.nonfinite_count,
    )
