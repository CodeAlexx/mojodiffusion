# minimax_h3_cond_probe — dump OUR runtime conditioner's embeds for a prompt
# (argv[1] = prompt file, argv[2] = output safetensors) so they can be
# cos-compared against the torchref-generated mmh3 TE caches. Diagnoses the
# train-vs-gen conditioning-consistency question directly.
from std.memory import ArcPointer
from std.sys import argv
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import sys_open, sys_pread, sys_close, BytePtr, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.models.text_encoder.minimax_h3_conditioning import (
    minimax_h3_encode_conditioning,
)
from serenitymojo.models.dit.minimax_h3_dit import minimax_h3_released_config
from std.memory import alloc

comptime PROCESSOR_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/processor"
comptime TEXT_ENCODER_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/text_encoder"


def _read_text(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var bytes = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var n = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if n <= 0:
            break
        for i in range(n):
            bytes.append(buf[i])
        offset += n
        if n < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    while len(bytes) > 0 and (bytes[len(bytes) - 1] == 10 or bytes[len(bytes) - 1] == 13):
        _ = bytes.pop()
    return String(unsafe_from_utf8=bytes)


def main() raises:
    var a = argv()
    if len(a) < 3:
        raise Error("usage: minimax_h3_cond_probe <prompt.txt> <out.safetensors>")
    var prompt = _read_text(String(a[1]))
    var out_path = String(a[2])
    var ctx = DeviceContext()
    var cond = minimax_h3_encode_conditioning(
        String(PROCESSOR_DIR), String(TEXT_ENCODER_DIR), prompt, ctx,
    )
    print("tokens:", cond.embeds.shape()[1])
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String("embeds"))
    tensors.append(ArcPointer(cond.embeds.clone(ctx)))
    save_safetensors(names, tensors, out_path, ctx)
    print("saved:", out_path)
