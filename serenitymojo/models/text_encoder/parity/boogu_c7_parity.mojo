# boogu_c7_parity.mojo — C7 (Qwen3-VL text encoder) REAL-weight parity gate vs torch.
#
# Reads the oracle's EXACT input_ids (45 tokens) + last_hidden_state from
# boogu_c7_oracle.py, loads the Boogu Qwen3-VL LM encoder, runs boogu_encode (RAW
# pre-final-norm last-layer output — the convention the C7 build measured as the
# match), and compares [1,45,4096] vs torch (cos >= 0.999 + magnitude).
#
# Run (oracle FIRST):
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/text_encoder/parity/boogu_c7_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/text_encoder/parity/boogu_c7_parity.mojo
from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.parity import ParityHarness
from serenitymojo.models.text_encoder.boogu_qwen3vl import load_boogu_qwen3vl, boogu_encode

comptime DUMP = "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/boogu_dumps/"
comptime MLLM_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/mllm"
comptime L = 45
comptime HIDDEN = 4096


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run boogu_c7_oracle.py first): ") + path)
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
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("=== C7 (Boogu Qwen3-VL text encoder) REAL-weight parity vs torch ===")

    var ids_f = _read_bin_f32(DUMP + "c7_input_ids.bin")
    if len(ids_f) != L:
        raise Error("input_ids len wrong: " + String(len(ids_f)))
    var ids = List[Int]()
    for i in range(L):
        ids.append(Int(ids_f[i] + 0.5))   # F32 token id -> Int (round)
    print("  loaded", len(ids), "token ids")

    var ref_hidden = _read_bin_f32(DUMP + "c7_last_hidden.bin")
    if len(ref_hidden) != L * HIDDEN:
        raise Error("last_hidden len wrong: " + String(len(ref_hidden)))

    print("  loading Qwen3-VL LM encoder (bf16, ~16GB)…")
    var enc = load_boogu_qwen3vl(String(MLLM_DIR), ctx)
    var hidden = boogu_encode(enc, ids, ctx)   # [1,45,4096] RAW pre-final-norm

    var sh = hidden.shape()
    if len(sh) != 3 or sh[1] != L or sh[2] != HIDDEN:
        raise Error("encoder output shape wrong")

    var h = ParityHarness()
    var r = h.compare(hidden, ref_hidden, ctx)
    print("  last_hidden", r)
    if not r.passed:
        raise Error("C7 encoder parity FAIL (cos < 0.999) — suspect mROPE vs half-split RoPE")
    print("VERDICT: C7 PASS — Qwen3-VL text encoder last_hidden matches torch (cos >= 0.999)")
