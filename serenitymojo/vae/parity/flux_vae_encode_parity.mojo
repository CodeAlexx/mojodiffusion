# flux_vae_encode_parity.mojo — Blocker C GATE: REAL-weight Flux.1 AE encoder
# parity vs the torch oracle (flux_vae_encode_oracle.py).
#
# Loads the deterministic input image dumped by the oracle, runs the Mojo
# FluxVaeEncoder (REAL ae.safetensors weights), and compares the MEAN latent
# (mu) against the torch reference at cos >= 0.99 (BF16-conv-floor tolerant; the
# conv weights are stored BF16 so the latent carries a BF16 precision floor).
# Also asserts: latent shape [1,16,LH,LW], all finite.
#
# Run (oracle FIRST, separate command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/vae/parity/flux_vae_encode_oracle.py 8 8
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/vae/parity/flux_vae_encode_parity.mojo

from std.math import sqrt
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.vae.flux_vae_encoder import FluxVaeEncoder, FLUX_ZC


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/vae/parity/"
comptime AE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"
comptime LH = 8
comptime LW = 8
comptime IH = 8 * LH
comptime IW = 8 * LW


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
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


def _cos(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("cos: length mismatch " + String(len(a)) + " vs " + String(len(b)))
    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    var denom = sqrt(na) * sqrt(nb)
    if denom == 0.0:
        return 1.0
    return dot / denom


def main() raises:
    var ctx = DeviceContext()
    print("=== Blocker C GATE: Flux.1 AE encoder REAL-weight parity vs torch ===")
    print("  ae:", AE_PATH)
    print("  input [1,3,", IH, ",", IW, "] -> latent [1,16,", LH, ",", LW, "]")

    # Input image dumped by the oracle.
    var img_h = _read_bin_f32(REF_DIR + "flux_vae_in.bin")
    if len(img_h) != 3 * IH * IW:
        raise Error("input bin size wrong: " + String(len(img_h)))
    var img = Tensor.from_host(img_h, [1, 3, IH, IW], STDtype.F32, ctx)

    # Load REAL encoder weights and run.
    var enc = FluxVaeEncoder[LH, LW].load(String(AE_PATH), ctx)
    var mu = enc.encode_mean(img, ctx)            # [1,16,LH,LW]
    var msh = mu.shape()
    if len(msh) != 4 or msh[0] != 1 or msh[1] != FLUX_ZC or msh[2] != LH or msh[3] != LW:
        raise Error("latent shape wrong: expected [1,16,8,8]")
    print("  latent shape OK: [1,", msh[1], ",", msh[2], ",", msh[3], "]")

    var mu_h = mu.to_host(ctx)
    for i in range(len(mu_h)):
        var v = mu_h[i]
        if not (v == v) or _abs(v) > 1.0e30:
            raise Error("latent non-finite at " + String(i))
    print("  latent all finite OK")

    # Compare vs torch reference mu.
    var oracle_mu = _read_bin_f32(REF_DIR + "flux_vae_mu.bin")
    var cos = _cos(mu_h, oracle_mu)
    print("  mu cos vs torch oracle =", cos)
    if cos < 0.99:
        raise Error("Flux VAE encode parity FAIL: cos " + String(cos) + " < 0.99")

    print("VERDICT: PASS — Flux.1 AE encoder loads REAL ae.safetensors, latent",
          "[1,16,8,8] finite, cos vs torch =", cos, "(>= 0.99)")
