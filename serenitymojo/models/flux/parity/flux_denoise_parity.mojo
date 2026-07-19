# flux_denoise_parity.mojo — GATE C: FULL FLUX.1-dev denoise integration parity
# vs the BFL torch oracle (flux_denoise_oracle.py). Capstone gate: ties the DiT,
# schedule, and Euler loop together over 20 steps.
#
# Reads the pinned initial noise + text embeds the oracle used (RNG streams
# differ between the Mojo custom randn and torch, so noise is PINNED — see the
# oracle header), runs the SAME 20-step flow-match Euler loop the CLI runs with
# the REAL flux1-dev DiT, and compares the FINAL latent.
#
# Conventions match the CLI denoise(): t/guidance pre-scaled by 1000; rope via
# build_flux1_rope_tables; schedule via build_flux1_sigma_schedule(STEPS, N_IMG);
# Euler img = img + (t_prev - t_curr) * pred. Tiny 4x4 grid (N_IMG=16, N_TXT=16).
#
# Metric: cosine over the [1,16,64] final latent. Bar: cos >= 0.99 (bf16-compute
# floor compounded over 20 steps; per-step DiT is cos 0.9994 -> bounded growth).
#
# Run (oracle FIRST):
#   python3 serenitymojo/models/flux/parity/flux_denoise_oracle.py 4 4 16 20
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . -Xlinker -lcuda serenitymojo/models/flux/parity/flux_denoise_parity.mojo

from std.math import sqrt
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul_scalar
from serenitymojo.models.dit.flux1_dit import (
    Flux1Config,
    Flux1Offloaded,
    build_flux1_rope_tables,
)
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/"
comptime DIT_PATH = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"
comptime H2 = 4
comptime W2 = 4
comptime N_IMG = H2 * W2          # 16
comptime N_TXT = 16
comptime S = N_IMG + N_TXT        # 32
comptime STEPS = 20
comptime GUIDANCE = Float32(3.5)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle (run flux_denoise_oracle.py first): ") + path)
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


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def main() raises:
    var ctx = DeviceContext()
    print("=== GATE C: FLUX.1-dev FULL DENOISE integration parity vs BFL torch ===")
    print("  dit:", DIT_PATH, " grid", N_IMG, "img +", N_TXT, "txt, STEPS", STEPS)

    var noise_h = _read_bin_f32(REF_DIR + "flux_dn_noise.bin")
    if len(noise_h) != N_IMG * 64:
        raise Error("noise bin size wrong: " + String(len(noise_h)))
    var txt_h = _read_bin_f32(REF_DIR + "flux_dn_txt.bin")
    if len(txt_h) != N_TXT * 4096:
        raise Error("txt bin size wrong: " + String(len(txt_h)))
    var vec_h = _read_bin_f32(REF_DIR + "flux_dn_vec.bin")
    if len(vec_h) != 768:
        raise Error("vec bin size wrong: " + String(len(vec_h)))

    var txt = cast_tensor(Tensor.from_host(txt_h, [1, N_TXT, 4096], STDtype.F32, ctx), STDtype.BF16, ctx)
    var vector = cast_tensor(Tensor.from_host(vec_h, [1, 768], STDtype.F32, ctx), STDtype.BF16, ctx)
    # latent kept in F32 between steps (matches CLI: img is F32, cast to BF16 per step).
    var img = Tensor.from_host(noise_h, [1, N_IMG, 64], STDtype.F32, ctx)

    var rope = build_flux1_rope_tables[N_IMG, N_TXT, 24, 128](H2, W2, ctx, STDtype.BF16)
    var sched = build_flux1_sigma_schedule(STEPS, N_IMG)

    print("  loading FLUX.1-dev DiT (offloaded)")
    var model = Flux1Offloaded.load(DIT_PATH, Flux1Config.dev(), ctx)

    for i in range(STEPS):
        var t_curr = sched[i]
        var t_prev = sched[i + 1]
        var tvals = List[Float32]()
        tvals.append(t_curr * 1000.0)
        var t_vec = Tensor.from_host(tvals, [1], STDtype.F32, ctx)
        var gvals = List[Float32]()
        gvals.append(GUIDANCE * 1000.0)
        var g_vec = Tensor.from_host(gvals, [1], STDtype.F32, ctx)

        var img_bf = cast_tensor(img, STDtype.BF16, ctx)
        var pred = cast_tensor(
            model.forward[N_IMG, N_TXT, S](
                img_bf, txt, t_vec, Optional[Tensor](g_vec^), vector, rope[0], rope[1], ctx,
            ),
            STDtype.F32, ctx,
        )
        var dt = t_prev - t_curr
        img = add(img, mul_scalar(pred, dt, ctx), ctx)

    var psh = img.shape()
    if len(psh) != 3 or psh[0] != 1 or psh[1] != N_IMG or psh[2] != 64:
        raise Error("final shape wrong: expected [1,16,64]")
    print("  final latent shape OK: [1,", psh[1], ",", psh[2], "]")

    var h = img.to_host(ctx)
    for i in range(len(h)):
        var v = h[i]
        if not (v == v) or _abs(v) > 1.0e30:
            raise Error("final latent non-finite at " + String(i))
    print("  final latent all finite OK")

    var oracle = _read_bin_f32(REF_DIR + "flux_dn_final.bin")
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

    print("  cos vs BFL oracle (final latent) =", cos)
    print("  mean-abs-diff                    =", mad)
    if cos < 0.99:
        raise Error("FLUX denoise integration parity FAIL: cos " + String(cos) + " < 0.99")

    print("VERDICT: PASS — FLUX.1-dev full 20-step denoise, final latent [1,16,64]",
          "finite, cos vs BFL torch =", cos, "(>= 0.99)")
