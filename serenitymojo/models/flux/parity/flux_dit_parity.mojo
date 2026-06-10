# flux_dit_parity.mojo — GATE B: REAL-weight Flux.1-dev DiT forward parity vs
# the BFL torch oracle (flux_dit_oracle.py).
#
# Reads the deterministic inputs the oracle dumped (img, txt, vector, t, guidance)
# on a tiny 4x4 token grid (N_IMG=16, N_TXT=16, S=32 — the DiT is sequence-length
# agnostic so this runs the FULL 19+38 block stack with REAL flux1-dev weights),
# runs Flux1Offloaded.forward, and compares the predicted velocity against the
# torch reference.
#
# Conventions (must match the CLI exactly):
#   * inputs cast to BF16 (img/txt/vector are BF16 in the real CLI caps path).
#   * t_vec = t*1000, g_vec = guidance*1000 (BFL time_factor; Mojo embedder f=1).
#   * rope built locally via build_flux1_rope_tables (BFL EmbedND convention,
#     identical & deterministic on both sides).
#
# Metric: cosine over the [1,16,64] velocity. Bar: cos >= 0.99 (bf16-compute floor
# across 57 blocks; >=0.999 is exact-level). Also asserts finite + correct shape.
#
# Run (oracle FIRST, separate command):
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/models/flux/parity/flux_dit_oracle.py 4 4 16
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . -Xlinker -lcuda serenitymojo/models/flux/parity/flux_dit_parity.mojo

from std.math import sqrt
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.flux1_dit import (
    Flux1Config,
    Flux1Offloaded,
    build_flux1_rope_tables,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/"
comptime DIT_PATH = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"
comptime H2 = 4
comptime W2 = 4
comptime N_IMG = H2 * W2          # 16
comptime N_TXT = 16
comptime S = N_IMG + N_TXT        # 32


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle (run flux_dit_oracle.py first): ") + path)
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


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def main() raises:
    var ctx = DeviceContext()
    print("=== GATE B: Flux.1-dev DiT forward REAL-weight parity vs BFL torch ===")
    print("  dit:", DIT_PATH)
    print("  grid H2=", H2, "W2=", W2, "N_IMG=", N_IMG, "N_TXT=", N_TXT, "S=", S)

    var img_h = _read_bin_f32(REF_DIR + "flux_dit_img.bin")
    if len(img_h) != N_IMG * 64:
        raise Error("img bin size wrong: " + String(len(img_h)))
    var txt_h = _read_bin_f32(REF_DIR + "flux_dit_txt.bin")
    if len(txt_h) != N_TXT * 4096:
        raise Error("txt bin size wrong: " + String(len(txt_h)))
    var vec_h = _read_bin_f32(REF_DIR + "flux_dit_vec.bin")
    if len(vec_h) != 768:
        raise Error("vec bin size wrong: " + String(len(vec_h)))
    var tg = _read_bin_f32(REF_DIR + "flux_dit_tg.bin")
    if len(tg) != 2:
        raise Error("tg bin size wrong: " + String(len(tg)))

    var img = cast_tensor(Tensor.from_host(img_h, [1, N_IMG, 64], STDtype.F32, ctx), STDtype.BF16, ctx)
    var txt = cast_tensor(Tensor.from_host(txt_h, [1, N_TXT, 4096], STDtype.F32, ctx), STDtype.BF16, ctx)
    var vector = cast_tensor(Tensor.from_host(vec_h, [1, 768], STDtype.F32, ctx), STDtype.BF16, ctx)

    # t / guidance pre-scaled by 1000 (BFL time_factor; oracle fed raw, BFL x1000).
    var tvals = List[Float32]()
    tvals.append(tg[0] * 1000.0)
    var t_vec = Tensor.from_host(tvals, [1], STDtype.F32, ctx)
    var gvals = List[Float32]()
    gvals.append(tg[1] * 1000.0)
    var g_vec = Tensor.from_host(gvals, [1], STDtype.F32, ctx)

    var rope = build_flux1_rope_tables[N_IMG, N_TXT, 24, 128](H2, W2, ctx, STDtype.BF16)

    print("  loading FLUX.1-dev DiT (offloaded)")
    var model = Flux1Offloaded.load(DIT_PATH, Flux1Config.dev(), ctx)
    var pred = cast_tensor(
        model.forward[N_IMG, N_TXT, S](
            img, txt, t_vec, Optional[Tensor](g_vec^), vector, rope[0], rope[1], ctx,
        ),
        STDtype.F32, ctx,
    )
    var psh = pred.shape()
    if len(psh) != 3 or psh[0] != 1 or psh[1] != N_IMG or psh[2] != 64:
        raise Error("pred shape wrong: expected [1,16,64]")
    print("  pred shape OK: [1,", psh[1], ",", psh[2], "]")

    var pred_h = pred.to_host(ctx)
    for i in range(len(pred_h)):
        var v = pred_h[i]
        if not (v == v) or _abs(v) > 1.0e30:
            raise Error("pred non-finite at " + String(i))
    print("  pred all finite OK")

    var oracle = _read_bin_f32(REF_DIR + "flux_dit_pred.bin")
    if len(oracle) != len(pred_h):
        raise Error("oracle/mojo size mismatch " + String(len(oracle)) + " vs " + String(len(pred_h)))

    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    var sad: Float32 = 0.0
    for i in range(len(oracle)):
        var a = pred_h[i]
        var b = oracle[i]
        dot += a * b
        na += a * a
        nb += b * b
        sad += _abs(a - b)
    var cos = dot / (sqrt(na) * sqrt(nb)) if (na > 0.0 and nb > 0.0) else Float32(1.0)
    var mad = sad / Float32(len(oracle))

    print("  cos vs BFL oracle =", cos)
    print("  mean-abs-diff     =", mad)
    if cos < 0.99:
        raise Error("Flux DiT forward parity FAIL: cos " + String(cos) + " < 0.99")

    print("VERDICT: PASS — Flux.1-dev DiT forward REAL weights, pred [1,16,64]",
          "finite, cos vs BFL torch =", cos, "(>= 0.99)")
