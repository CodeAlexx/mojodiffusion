# krea2_te_parity_probe.mojo — Krea-2 Qwen3-VL-4B TE 12-layer-stack parity gate vs torch.
#
# Reads the oracle's EXACT input_ids (full prefix+prompt+suffix) + the
# (1, L', 12, 2560) stacked-and-prefix-dropped hidden states from gen_krea2_te.py,
# loads the real Qwen3-VL-4B text stack, runs encode_krea2_stack (12 SELECT_LAYERS
# stacked, leading 34 system-prefix rows dropped), and compares vs torch
# (cos >= 0.999). FAIL-LOUD: a length mismatch or cos < 0.999 raises.
#
# Run (oracle FIRST — dumps the byte-identical token ids + reference stack):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/text_encoder/parity/gen_krea2_te.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#     serenitymojo/models/text_encoder/parity/krea2_te_parity_probe.mojo
from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.parity import ParityHarness
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import (
    load_krea2_qwen3vl_4b,
    encode_krea2_stack,
    KREA2_DROP_IDX,
)


comptime DUMP = (
    "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/krea2_dumps/"
)
comptime TE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
comptime HIDDEN = 2560
comptime N_LAYERS_STACKED = 12


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(
            String("cannot open (run gen_krea2_te.py first): ") + path
        )
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle dump: ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var cnt = done // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(cnt):
        out.append(fp[i])
    buf.free()
    return out^


def _read_ids(path: String) raises -> List[Int]:
    """Token ids dumped as raw little-endian F32 -> Int."""
    var f = _read_bin_f32(path)
    var ids = List[Int]()
    for i in range(len(f)):
        ids.append(Int(f[i]))
    return ids^


def _std(vals: List[Float32]) -> Float32:
    var n = len(vals)
    if n == 0:
        return Float32(0.0)
    var mean = Float32(0.0)
    for i in range(n):
        mean += vals[i]
    mean /= Float32(n)
    var acc = Float32(0.0)
    for i in range(n):
        var d = vals[i] - mean
        acc += d * d
    acc /= Float32(n)
    return sqrt(acc)


def _vram(ctx: DeviceContext, tag: String) raises:
    ctx.synchronize()
    var mi = ctx.get_memory_info()
    print(
        "[krea2-vram]", tag, ": free=",
        Int(Float64(mi[0]) / 1048576.0), "MiB used=",
        Int(Float64(mi[1] - mi[0]) / 1048576.0), "MiB",
    )


def main() raises:
    var ctx = DeviceContext()
    _vram(ctx, "start")

    var ids = _read_ids(DUMP + "krea2_input_ids.bin")
    var L_full = len(ids)
    var keep = L_full - KREA2_DROP_IDX
    print(
        "[krea2-te-probe] loaded", L_full, "token ids (full prefix+prompt+suffix);",
        "first/last =", ids[0], "/", ids[L_full - 1],
        "| drop_idx =", KREA2_DROP_IDX, "-> keep =", keep,
    )

    print("[krea2-te-probe] loading Qwen3-VL-4B text stack (bf16) ...")
    var enc = load_krea2_qwen3vl_4b(TE_DIR, ctx)
    print("[krea2-te-probe] encoder loaded.")
    _vram(ctx, "after load")

    var stack = encode_krea2_stack(enc, ids, ctx)
    _vram(ctx, "after encode")
    var sh = stack.shape()
    print(
        "[krea2-te-probe] stack shape = [",
        sh[0], ",", sh[1], ",", sh[2], ",", sh[3], "]",
    )

    # Shape gate: must be [1, keep, 12, 2560].
    if sh[0] != 1 or sh[1] != keep or sh[2] != N_LAYERS_STACKED or sh[3] != HIDDEN:
        raise Error(
            String("krea2-te-probe: shape mismatch, got [")
            + String(sh[0]) + "," + String(sh[1]) + "," + String(sh[2])
            + "," + String(sh[3]) + "] expected [1,"
            + String(keep) + ",12,2560]"
        )

    var reference = _read_bin_f32(DUMP + "krea2_te_stack.bin")
    var host = stack.to_host(ctx)
    print(
        "[krea2-te-probe] mojo std =", _std(host),
        "| n_mojo =", len(host), "n_ref =", len(reference),
    )

    var harness = ParityHarness(0.999)
    var res = harness.compare(stack, reference, ctx)
    print("[krea2-te-probe]", res)

    if not res.passed:
        raise Error(
            String("krea2-te-probe FAIL: cos=") + String(res.cos)
            + " < 0.999 (max_abs=" + String(res.max_abs) + ")"
        )
    print("[krea2-te-probe] PASS — Qwen3-VL-4B 12-layer stack matches torch.")
