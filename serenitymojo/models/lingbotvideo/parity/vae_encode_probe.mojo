# Gate: LingBotWanVaeEncoder (Wan2.1-style 3D causal VAE), image mode T=1.
#
# Loads the REAL encoder.* + quant_conv.* weights from the LingBot VAE
# safetensors, plus the deterministic input pixel + normalized latent from
# oracle_vae_encode.safetensors, runs encode, and compares to the oracle latent
# with ParityHarness (gate cos >= 0.999) plus magnitude ratio |mine|/|ref| and
# max_abs.
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/vae_encode_probe.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.vae_encoder import LingBotWanVaeEncoder

comptime VAE_WEIGHTS = "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime H = 256
comptime W = 256


def _sh(s: List[Int]) -> String:
    var out = String("[")
    for i in range(len(s)):
        if i > 0:
            out += ", "
        out += String(s[i])
    out += "]"
    return out


def _l2(v: List[Float32]) -> Float64:
    var acc: Float64 = 0.0
    for i in range(len(v)):
        acc += Float64(v[i]) * Float64(v[i])
    return sqrt(acc)


def main() raises:
    var ctx = DeviceContext()

    print("[vae-enc] loading LingBot Wan VAE encoder weights from", VAE_WEIGHTS)
    var enc = LingBotWanVaeEncoder[H, W].load(String(VAE_WEIGHTS), ctx)

    var oracle_path = String(PARITY_DIR) + "/oracle_vae_encode.safetensors"
    print("[vae-enc] loading oracle pixel / latent from", oracle_path)
    var st = ShardedSafeTensors.open(oracle_path)
    var pixel = Tensor.from_view_as_f32(st.tensor_view("pixel"), ctx)
    var lat_ref = Tensor.from_view_as_f32(st.tensor_view("latent"), ctx)
    var ref_host = lat_ref.to_host(ctx)
    print("  pixel  shape =", _sh(pixel.shape()))
    print("  latent shape =", _sh(lat_ref.shape()))

    print("[vae-enc] running encode")
    var out = enc.encode(pixel, ctx)
    print("  encoded shape =", _sh(out.shape()))

    var mine_host = out.to_host(ctx)
    var harness = ParityHarness(0.999)
    var res = harness.compare_host(mine_host, ref_host)

    var mag_mine = _l2(mine_host)
    var mag_ref = _l2(ref_host)
    var ratio = mag_mine / mag_ref if mag_ref > 0.0 else 0.0

    print("[vae-enc] encode parity:", res)
    print("[vae-enc] cos           =", res.cos)
    print("[vae-enc] magnitude     |mine|/|ref| =", ratio, "(|mine|=", mag_mine, "|ref|=", mag_ref, ")")
    print("[vae-enc] max_abs_diff  =", res.max_abs)
    if res.passed:
        print("[vae-enc] GATE PASS (cos >= 0.999)")
    else:
        print("[vae-enc] GATE FAIL (cos < 0.999)")
