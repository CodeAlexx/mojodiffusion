# qwen3_parity.mojo — GPU parity driver for the Qwen3 text encoder.
#
# Loads the REAL Z-Image text_encoder weights, runs the Mojo encoder for a
# FIXED token-id list (must match parity/oracle.py TOKEN_IDS), and compares
# per-layer hidden states + the final last_hidden_state against the numpy
# oracle dump (parity/ref.jsonl) using ParityHarness (cos + max_abs in F64).
#
# Per-layer GPU streaming comparison (the Mojo side is BF16-storage/F32-accum;
# the reference is clean-f32 numpy). Run AFTER oracle.py.
#
# Run: pixi run mojo run -I . \
#        serenitymojo/models/text_encoder/parity/qwen3_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.text_encoder.qwen3_encoder import (
    Qwen3Config,
    Qwen3Encoder,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime TE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/text_encoder"
)
comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/"
    "ref.jsonl"
)


# ── tiny JSONL field readers (only what we need: "tag" string, "data" array) ──
def _read_text(path: String) raises -> List[UInt8]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ref: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error("empty ref file")
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


@fieldwise_init
struct _Ref(Copyable, Movable):
    var tag: String
    var data: List[Float32]


def _find(bytes: List[UInt8], start: Int, needle: String) raises -> Int:
    """Index of `needle` in bytes at/after start, or -1."""
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


def _parse_line(bytes: List[UInt8], line_start: Int, line_end: Int) raises -> _Ref:
    """Parse one JSONL object's "tag" and "data" array (numbers only)."""
    # tag
    var tag_key = _find(bytes, line_start, String("\"tag\": \""))
    if tag_key < 0 or tag_key >= line_end:
        raise Error("ref line missing tag")
    var ts = tag_key + 8
    var te = ts
    while te < line_end and Int(bytes[te]) != 0x22:  # closing quote
        te += 1
    var tag_chars = List[UInt8]()
    for i in range(ts, te):
        tag_chars.append(bytes[i])
    var tag = String(from_utf8=tag_chars)
    # data array
    var data_key = _find(bytes, line_start, String("\"data\": ["))
    if data_key < 0 or data_key >= line_end:
        raise Error("ref line missing data")
    var ds = data_key + 9
    var de = ds
    while de < line_end and Int(bytes[de]) != 0x5D:  # ']'
        de += 1
    # parse comma-separated floats in [ds, de)
    var data = List[Float32]()
    var i = ds
    while i < de:
        # skip spaces / commas
        var c = Int(bytes[i])
        if c == 0x20 or c == 0x2C:
            i += 1
            continue
        var ne = i
        while ne < de:
            var cc = Int(bytes[ne])
            if cc == 0x2C:
                break
            ne += 1
        var num_chars = List[UInt8]()
        for j in range(i, ne):
            var ch = Int(bytes[j])
            if ch != 0x20:
                num_chars.append(bytes[j])
        var s = String(from_utf8=num_chars)
        data.append(Float32(atof(s)))
        i = ne + 1
    return _Ref(tag, data^)


def _load_refs() raises -> List[_Ref]:
    var bytes = _read_text(String(REF_PATH))
    var refs = List[_Ref]()
    var i = 0
    var n = len(bytes)
    while i < n:
        var le = i
        while le < n and Int(bytes[le]) != 0x0A:  # newline
            le += 1
        if le > i + 2:  # non-empty line
            refs.append(_parse_line(bytes, i, le))
        i = le + 1
    return refs^


def _get(refs: List[_Ref], tag: String) raises -> List[Float32]:
    for ref r in refs:
        if r.tag == tag:
            return r.data.copy()
    raise Error(String("ref tag not found: ") + tag)


def main() raises:
    var ctx = DeviceContext()
    var cfg = Qwen3Config.zimage()

    # FIXED token ids — MUST match parity/oracle.py TOKEN_IDS.
    var ids = List[Int]()
    for t in [9707, 11, 1879, 0, 358, 1079, 264, 1467]:
        ids.append(t)

    print("loading text_encoder weights (398 tensors, 3 shards)...")
    var enc = Qwen3Encoder.load(String(TE_DIR), cfg, ctx)
    print("  loaded.")

    var refs = _load_refs()

    print("running encoder forward (36 layers, seq=", len(ids), ")...")
    var states = enc.encode_layer_states(ids, ctx)
    print("  forward done. collected", len(states), "layer states.")

    var harness = ParityHarness(0.99)

    # per-layer (the oracle emitted layer0..3 + layer35)
    var check_layers = List[Int]()
    for li in [0, 1, 2, 3, 35]:
        check_layers.append(li)

    print("=== per-layer parity (cos + max_abs) ===")
    for ref li in check_layers:
        var tag = String("layer") + String(li)
        var refv = _get(refs, tag)
        var res = harness.compare(states[li][], refv, ctx)
        print("layer", li, ":", res)

    # final last_hidden_state = model.norm(last layer state)
    var final = enc.final_norm(states[len(states) - 1][], ctx)
    var ref_final = _get(refs, String("last_hidden_state"))
    var res_f = harness.compare(final, ref_final, ctx)
    print("=== final last_hidden_state ===")
    print("last_hidden_state :", res_f)
