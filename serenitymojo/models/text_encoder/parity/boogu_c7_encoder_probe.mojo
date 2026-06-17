# boogu_c7_encoder_probe.mojo — C7 (Boogu Qwen3-VL TEXT encoder) probe.
#
# Reads the oracle's c7_input_ids (45 ints, dumped as raw F32) from
# boogu_dumps/c7_input_ids.bin, loads the Boogu mllm Qwen3-VL language-model
# stack, runs BOTH boogu_encode (RAW last-layer, pre-final-norm) and
# boogu_encode_normed (final_norm'd), and prints both output shapes [1,45,4096]
# + stds. Compare to the oracle's last_hidden std 16.77 (boogu_c7_oracle.py).
#
# This probe does NOT claim a parity cosine — the orchestrator gates vs
# boogu_dumps/c7_last_hidden.bin. Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . \
#     serenitymojo/models/text_encoder/parity/boogu_c7_encoder_probe.mojo
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.text_encoder.boogu_qwen3vl import (
    load_boogu_qwen3vl,
    boogu_encode,
    boogu_encode_normed,
)


comptime MLLM_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/mllm"
comptime IDS_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/"
    "boogu_dumps/c7_input_ids.bin"
)


def _read_ids_f32(path: String) raises -> List[Int]:
    """Read raw little-endian F32 token ids and cast to Int."""
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ids: ") + path)
    var nbytes = file_size(fd)
    if nbytes <= 0 or (nbytes % 4) != 0:
        _ = sys_close(fd)
        raise Error("bad ids file size")
    var buf = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, buf + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var n = done // 4
    var fp = buf.bitcast[Float32]()
    var ids = List[Int]()
    for i in range(n):
        ids.append(Int(fp[i]))
    buf.free()
    return ids^


def _std(vals: List[Float32]) -> Float32:
    var n = len(vals)
    if n == 0:
        return Float32(0.0)
    var mean = Float32(0.0)
    for i in range(n):
        mean += vals[i]
    mean /= Float32(n)
    var var_acc = Float32(0.0)
    for i in range(n):
        var d = vals[i] - mean
        var_acc += d * d
    var_acc /= Float32(n)
    return sqrt(var_acc)


def main() raises:
    var ctx = DeviceContext()

    var ids = _read_ids_f32(IDS_PATH)
    print("[c7-probe] loaded", len(ids), "token ids; first/last =",
          ids[0], "/", ids[len(ids) - 1])

    print("[c7-probe] loading Boogu mllm Qwen3-VL text stack (bf16) ...")
    var enc = load_boogu_qwen3vl(MLLM_DIR, ctx)
    print("[c7-probe] encoder loaded.")

    var raw = boogu_encode(enc, ids, ctx)
    var raw_sh = raw.shape()
    var raw_host = raw.to_host(ctx)
    var raw_std = _std(raw_host)
    print("[c7-probe] RAW  (pre-final-norm) shape = [",
          raw_sh[0], ",", raw_sh[1], ",", raw_sh[2], "]  std =", raw_std)

    var normed = boogu_encode_normed(enc, ids, ctx)
    var normed_sh = normed.shape()
    var normed_host = normed.to_host(ctx)
    var normed_std = _std(normed_host)
    print("[c7-probe] NORM (final_norm'd)   shape = [",
          normed_sh[0], ",", normed_sh[1], ",", normed_sh[2], "]  std =", normed_std)

    print("[c7-probe] oracle hidden_states[-1] std = 16.77463 (boogu_c7_oracle.py)")
    print("[c7-probe] done.")
