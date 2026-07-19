# flux_t5_parity.mojo — GATE D (T5): REAL-weight FLUX T5-XXL encoder parity vs
# the HF torch oracle (flux_t5_oracle.py). Validates this session's fp16->bf16
# T5 fix numerically (the conditioning that drives the image).
#
# Reads the EXACT 512 token ids the oracle fed HF T5, runs the Mojo T5Encoder
# (REAL t5xxl_fp16.safetensors, weights cast fp16->bf16 at load — the fix), and
# compares the [1,512,4096] hidden state. No attention mask (BFL=None), so the
# full padded sequence is compared on both sides.
#
# Metric: cosine over [1,512,4096]. Bar: cos >= 0.99 (bf16-compute floor over 24
# T5 layers; the oracle uses bf16 weights + fp32 accumulation = the Mojo recipe).
#
# Run (oracle FIRST):
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/models/flux/parity/flux_t5_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/flux/parity/flux_t5_parity.mojo

from std.math import sqrt
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.text_encoder.t5_encoder import T5Encoder, T5Config


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/"
comptime T5_PATH = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors"
comptime S = 512
comptime D = 4096


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle (run flux_t5_oracle.py first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(n // 4):
        out.append(fp[i])
    buf.free()
    return out^


def _read_bin_i32(path: String) raises -> List[Int]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ids (run oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var ip = buf.bitcast[Int32]()
    var out = List[Int]()
    for i in range(n // 4):
        out.append(Int(ip[i]))
    buf.free()
    return out^


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def main() raises:
    var ctx = DeviceContext()
    print("=== GATE D (T5): FLUX T5-XXL encoder REAL-weight parity vs HF torch ===")
    print("  t5:", T5_PATH)

    var ids = _read_bin_i32(REF_DIR + "flux_t5_ids.bin")
    if len(ids) != S:
        raise Error("ids count wrong: " + String(len(ids)))
    print("  ids:", len(ids), " first:", ids[0], ids[1], ids[2], ids[3], "... eos@9:", ids[9])

    var t5 = T5Encoder[S].load(String(T5_PATH), T5Config.t5_xxl(), ctx)
    var hidden = cast_tensor(t5.encode(ids^, ctx), STDtype.F32, ctx)
    var hsh = hidden.shape()
    if len(hsh) != 3 or hsh[0] != 1 or hsh[1] != S or hsh[2] != D:
        raise Error("hidden shape wrong: expected [1,512,4096]")
    print("  hidden shape OK: [1,", hsh[1], ",", hsh[2], "]")

    var h = hidden.to_host(ctx)
    for i in range(len(h)):
        var v = h[i]
        if not (v == v) or _abs(v) > 1.0e30:
            raise Error("hidden non-finite at " + String(i) + " (fp16->bf16 fix regressed?)")
    print("  hidden all finite OK (no NaN — fp16->bf16 fix holds)")

    var oracle = _read_bin_f32(REF_DIR + "flux_t5_hidden.bin")
    if len(oracle) != len(h):
        raise Error("oracle/mojo size mismatch " + String(len(oracle)) + " vs " + String(len(h)))

    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    var sad: Float32 = 0.0
    for i in range(len(oracle)):
        var a = h[i]
        var b = oracle[i]
        dot += a * b
        na += a * a
        nb += b * b
        sad += _abs(a - b)
    var cos = dot / (sqrt(na) * sqrt(nb)) if (na > 0.0 and nb > 0.0) else Float32(1.0)
    var mad = sad / Float32(len(oracle))

    print("  cos vs HF oracle =", cos)
    print("  mean-abs-diff    =", mad)
    if cos < 0.99:
        raise Error("FLUX T5 encoder parity FAIL: cos " + String(cos) + " < 0.99")

    print("VERDICT: PASS — FLUX T5-XXL encoder REAL weights, hidden [1,512,4096]",
          "finite (no NaN), cos vs HF =", cos, "(>= 0.99)")
