# NAVA audio vocoder parity: Mojo LTX2VocoderWithBWE.forward on NAVA's
# ltx-2.3-22b-dev_audio_vae.safetensors vs torch (F32 production) raw 48 kHz wav.
# mel [1,2,133,64] -> wav48 [1,2,63840].  Gate cos>=0.999 (forward F32-computes).
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.vocoder.ltx2_vocoder import LTX2VocoderWithBWE

comptime CKPT = "/home/alex/.serenity/models/checkpoints/NAVA/params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity/nava_audio_vae_fx.safetensors"


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA audio vocoder parity (WithBWE, 48kHz, F32 compute) ===")
    var voc = LTX2VocoderWithBWE.from_file(CKPT, ctx)

    var fx = ShardedSafeTensors.open(FX)
    var mel = Tensor.from_view_as_f32(fx.tensor_view("mel"), ctx)  # [1,2,133,64]
    var ms = mel.shape()
    print("  mel in:  [", ms[0], ",", ms[1], ",", ms[2], ",", ms[3], "]")

    var wav = voc.forward(mel, ctx)  # [1,2,63840]
    var ws = wav.shape()
    print("  wav out: [", ws[0], ",", ws[1], ",", ws[2], "]")

    var ref_host = Tensor.from_view(fx.tensor_view("wav48"), ctx).to_host(ctx)
    print("NAVA vocoder wav48 vs torch:", ParityHarness(0.999).compare(wav, ref_host, ctx))
