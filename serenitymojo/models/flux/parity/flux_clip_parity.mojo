# flux_clip_parity.mojo — GATE D (CLIP): REAL-weight FLUX CLIP-L pooled parity vs
# the HF torch oracle (flux_clip_oracle.py). Validates the `y`/vector_in
# conditioning ([1,768]) — and catches any pooled projection/position mismatch.
#
# Reads the EXACT 77 token ids the oracle fed HF CLIP, runs the Mojo ClipEncoder
# (REAL clip_l.safetensors), takes pooled = post-LN hidden at first EOS (no
# projection — matches HF CLIPTextModel.pooler_output), compares [1,768].
#
# Metric: cosine. Bar: cos >= 0.99 (bf16-weight floor; CLIP-L is shallow so
# expect high). Run (oracle FIRST):
#   python3 serenitymojo/models/flux/parity/flux_clip_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/flux/parity/flux_clip_parity.mojo

from std.math import sqrt
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.text_encoder.clip_encoder import ClipEncoder, ClipConfig


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/"
comptime CLIP_PATH = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
comptime S = 77
comptime H = 768


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle (run flux_clip_oracle.py first): ") + path)
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
    print("=== GATE D (CLIP): FLUX CLIP-L pooled REAL-weight parity vs HF torch ===")
    print("  clip:", CLIP_PATH)

    var ids = _read_bin_i32(REF_DIR + "flux_clip_ids.bin")
    if len(ids) != S:
        raise Error("ids count wrong: " + String(len(ids)))
    print("  ids:", len(ids), " bos:", ids[0], " ids[1..3]:", ids[1], ids[2], ids[3])

    var clip = ClipEncoder.load(String(CLIP_PATH), ClipConfig.clip_l(), ctx)
    var out = clip.encode_sdxl[S](ids^, ctx)
    var pooled = cast_tensor(out[1], STDtype.F32, ctx)   # [1,768]
    var psh = pooled.shape()
    if len(psh) != 2 or psh[0] != 1 or psh[1] != H:
        raise Error("pooled shape wrong: expected [1,768]")
    print("  pooled shape OK: [1,", psh[1], "]")

    var h = pooled.to_host(ctx)
    for i in range(len(h)):
        var v = h[i]
        if not (v == v) or _abs(v) > 1.0e30:
            raise Error("pooled non-finite at " + String(i))
    print("  pooled all finite OK")

    var oracle = _read_bin_f32(REF_DIR + "flux_clip_pooled.bin")
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
        raise Error("FLUX CLIP pooled parity FAIL: cos " + String(cos) + " < 0.99")

    print("VERDICT: PASS — FLUX CLIP-L pooled REAL weights, [1,768] finite,",
          "cos vs HF =", cos, "(>= 0.99)")
