# NAVA audio resample parity: Mojo resample_hann(wav48, 48000, 16000) vs torch
# torchaudio.functional.resample (sinc_interp_hann). wav48 [1,2,63840] -> [1,2,21280].
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.resample import resample_hann

comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity/nava_audio_vae_fx.safetensors"


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA audio resample 48k->16k parity (sinc_interp_hann) ===")
    var fx = ShardedSafeTensors.open(FX)
    var wav48 = Tensor.from_view_as_f32(fx.tensor_view("wav48"), ctx)  # [1,2,63840]
    var ws = wav48.shape()
    print("  wav48 in: [", ws[0], ",", ws[1], ",", ws[2], "]")

    var wav16 = resample_hann(wav48, 48000, 16000, ctx)  # [1,2,21280]
    var os = wav16.shape()
    print("  wav16 out:[", os[0], ",", os[1], ",", os[2], "]")

    var ref_host = Tensor.from_view(fx.tensor_view("wav16"), ctx).to_host(ctx)
    print("NAVA resample 48->16 vs torchaudio:", ParityHarness(0.999).compare(wav16, ref_host, ctx))
