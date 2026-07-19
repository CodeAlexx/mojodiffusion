# qwen3_preattn_probe.mojo — parity for the encoder path UP TO attention.
#
# The SDK flash_attention fails to instantiate at head_dim=128 (see report:
# "no valid implementation of mma" for depth=128, ANY head count, BF16/F32).
# This probe validates EVERY OTHER foundation op the encoder uses — embedding
# gather, rms_norm (hidden + per-head qk-norm), linear (q/k proj), and
# rope_halfsplit — against the numpy oracle, isolating the blocker to SDPA.
#
# Run AFTER oracle.py:
#   pixi run mojo run -I . \
#     serenitymojo/models/text_encoder/parity/qwen3_preattn_probe.mojo

from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.text_encoder.qwen3_encoder import (
    Qwen3Config,
    Qwen3Encoder,
)
from serenitymojo.io.ffi import (
    sys_open,
    sys_close,
    sys_pread,
    file_size,
    O_RDONLY,
)


comptime TE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/text_encoder"
)
comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/"
    "ref.jsonl"
)


def _read_text(path: String) raises -> List[UInt8]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ref: ") + path)
    var n = file_size(fd)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var out = List[UInt8]()
    for i in range(done):
        out.append(buf[i])
    buf.free()
    return out^


def _find(bytes: List[UInt8], start: Int, needle: String) raises -> Int:
    var nl = needle.byte_length()
    var i = start
    while i + nl <= len(bytes):
        var ok = True
        for j in range(nl):
            if Int(bytes[i + j]) != ord(needle[byte=j]):
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1


def _ref_for(tag: String) raises -> List[Float32]:
    """Find the JSONL line whose tag == `tag`, parse its data array."""
    var bytes = _read_text(String(REF_PATH))
    var n = len(bytes)
    var key = String("\"tag\": \"") + tag + "\""
    var pos = _find(bytes, 0, key)
    if pos < 0:
        raise Error(String("tag not found: ") + tag)
    # find this line's bounds
    var ls = pos
    while ls > 0 and Int(bytes[ls - 1]) != 0x0A:
        ls -= 1
    var le = pos
    while le < n and Int(bytes[le]) != 0x0A:
        le += 1
    var dk = _find(bytes, ls, String("\"data\": ["))
    if dk < 0 or dk >= le:
        raise Error("data missing")
    var ds = dk + 9
    var de = ds
    while de < le and Int(bytes[de]) != 0x5D:
        de += 1
    var data = List[Float32]()
    var i = ds
    while i < de:
        var c = Int(bytes[i])
        if c == 0x20 or c == 0x2C:
            i += 1
            continue
        var ne = i
        while ne < de and Int(bytes[ne]) != 0x2C:
            ne += 1
        var num = List[UInt8]()
        for j in range(i, ne):
            if Int(bytes[j]) != 0x20:
                num.append(bytes[j])
        data.append(Float32(atof(String(from_utf8=num))))
        i = ne + 1
    return data^


def main() raises:
    var ctx = DeviceContext()
    var cfg = Qwen3Config.zimage()

    var ids = List[Int]()
    for t in [9707, 11, 1879, 0, 358, 1079, 264, 1467]:
        ids.append(t)

    print("loading text_encoder weights...")
    var enc = Qwen3Encoder.load(String(TE_DIR), cfg, ctx)
    print("  loaded.")

    print("running pre-attention path (embed/norm/proj/qk-norm/rope)...")
    var dbg = enc.debug_pre_attn(ids, ctx)

    var harness = ParityHarness(0.99)
    print("=== pre-attention parity (cos + max_abs) ===")
    var r0 = harness.compare(dbg[0][], _ref_for(String("embed")), ctx)
    print("embed          :", r0)
    var r1 = harness.compare(dbg[1][], _ref_for(String("l0_input_norm")), ctx)
    print("l0_input_norm  :", r1)
    var r2 = harness.compare(dbg[2][], _ref_for(String("l0_q_rope")), ctx)
    print("l0_q_rope      :", r2)
    var r3 = harness.compare(dbg[3][], _ref_for(String("l0_k_rope")), ctx)
    print("l0_k_rope      :", r3)
