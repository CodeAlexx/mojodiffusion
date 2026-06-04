# acestep_vae_probe.mojo — ACE-Step Oobleck VAE decoder parity probe.
#
# GATE: decode the fixed oracle latent [1,64,8] through the Mojo Oobleck decoder
# and compare the waveform [1,2,15360] against the canonical diffusers
# AutoencoderOobleck bf16-GPU oracle (cos>=0.999 + magnitude ratio).
#
# Oracle built from the REAL checkpoint (NOT a transcription):
#   /tmp/ace_lat.f32  — raw F32 latent  [1,64,8]   (512 elems)
#   /tmp/ace_out.f32  — raw F32 waveform [1,2,15360] (30720 elems)
# produced by:  AutoencoderOobleck.from_pretrained(...).to(bf16).decoder(lat)
#
# Run: pixi run mojo run -I . serenitymojo/acestep_vae_probe.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, BytePtr, alloc
from serenitymojo.parity import ParityHarness
from serenitymojo.models.vae.acestep_vae import OobleckVaeDecoder


comptime O_RDONLY = Int32(0)
comptime CKPT = "/home/alex/ACE-Step-1.5/checkpoints/vae/diffusion_pytorch_model.safetensors"


# Read `n` F32 values from a raw little-endian f32 file via pread.
def _read_f32(path: String, n: Int) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        raise Error("could not open " + path)
    var nbytes = n * 4
    var buf = alloc[UInt8](nbytes)
    var got = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), nbytes, 0)
    _ = sys_close(fd)
    if got != nbytes:
        buf.free()
        raise Error("short read on " + path)
    var fptr = buf.bitcast[Float32]()
    var out = List[Float32]()
    out.resize(n, Float32(0.0))
    for i in range(n):
        out[i] = fptr[i]
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("ACE-Step Oobleck VAE decoder parity probe")

    # ── load oracle latent + reference waveform ───────────────────────────────
    var lat_h = _read_f32("/tmp/ace_lat.f32", 1 * 64 * 8)
    var ref_wave = _read_f32("/tmp/ace_out.f32", 1 * 2 * 15360)
    print("loaded oracle: latent", len(lat_h), "elems, ref waveform", len(ref_wave), "elems")

    var lsh = List[Int]()
    lsh.append(1); lsh.append(64); lsh.append(8)
    var latent = Tensor.from_host(lat_h, lsh^, STDtype.F32, ctx)

    # ── load decoder + decode ─────────────────────────────────────────────────
    print("loading decoder weights from checkpoint ...")
    var dec = OobleckVaeDecoder.load(CKPT, ctx)
    print("decoding ...")
    var wave = dec.decode(latent, ctx)
    var ws = wave.shape()
    print("Mojo waveform shape: [", ws[0], ",", ws[1], ",", ws[2], "]")

    # ── magnitude ratio (RMS) ─────────────────────────────────────────────────
    var got = wave.to_host(ctx)
    var sa = Float64(0.0)
    var sr = Float64(0.0)
    for i in range(len(got)):
        sa += Float64(got[i]) * Float64(got[i])
    for i in range(len(ref_wave)):
        sr += Float64(ref_wave[i]) * Float64(ref_wave[i])
    var rms_a = (sa / Float64(len(got))) ** 0.5
    var rms_r = (sr / Float64(len(ref_wave))) ** 0.5
    var mag_ratio = rms_a / rms_r if rms_r > 0.0 else 0.0

    # ── cosine parity ─────────────────────────────────────────────────────────
    var harness = ParityHarness(0.999)
    var res = harness.compare_host(got, ref_wave)
    print("---")
    print("cos      =", res.cos)
    print("max_abs  =", res.max_abs)
    print("mag_ratio=", mag_ratio, " (rms_mojo=", rms_a, " rms_ref=", rms_r, ")")
    print("---")
    if res.cos >= 0.999:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
