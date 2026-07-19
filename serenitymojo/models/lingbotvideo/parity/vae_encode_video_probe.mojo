# Probe: LingBotWanVaeEncoder.encode_video (Wan2.1-style 3D causal VAE),
# TEMPORAL (video) mode. Loads the REAL encoder.* + quant_conv.* weights plus a
# deterministic input clip from oracle_vae_encode_video.safetensors (key `pixel`,
# [1,3,13,256,256]), runs encode_video, and prints the output latent shape +
# mean/std/absmax. (No parity gate here — the orchestrator owns that.)
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . \
#       serenitymojo/models/lingbotvideo/parity/vae_encode_video_probe.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.lingbotvideo.vae_encoder import LingBotWanVaeEncoder
from serenitymojo.parity import ParityHarness

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


def _stats(v: List[Float32]) raises:
    var n = len(v)
    var mean: Float64 = 0.0
    var amax: Float64 = 0.0
    for i in range(n):
        var x = Float64(v[i])
        mean += x
        var a = x if x >= 0.0 else -x
        if a > amax:
            amax = a
    mean /= Float64(n)
    var var_: Float64 = 0.0
    for i in range(n):
        var d = Float64(v[i]) - mean
        var_ += d * d
    var_ /= Float64(n)
    print("  count  =", n)
    print("  mean   =", mean)
    print("  std    =", sqrt(var_))
    print("  absmax =", amax)


def main() raises:
    var ctx = DeviceContext()

    print("[vae-enc-vid] loading LingBot Wan VAE encoder weights from", VAE_WEIGHTS)
    var enc = LingBotWanVaeEncoder[H, W].load(String(VAE_WEIGHTS), ctx)

    var oracle_path = String(PARITY_DIR) + "/oracle_vae_encode_video.safetensors"
    print("[vae-enc-vid] loading oracle clip from", oracle_path)
    var st = ShardedSafeTensors.open(oracle_path)
    var clip = Tensor.from_view_as_f32(st.tensor_view("pixel"), ctx)
    print("  clip shape =", _sh(clip.shape()))

    print("[vae-enc-vid] running encode_video")
    var out = enc.encode_video(clip, ctx)
    print("  encoded shape =", _sh(out.shape()))

    var mine_host = out.to_host(ctx)
    _stats(mine_host)

    # ── parity gate: cos >= 0.999 vs the diffusers oracle latent + magnitude ──
    var lat_ref = Tensor.from_view_as_f32(st.tensor_view("latent"), ctx)
    print("  oracle latent shape =", _sh(lat_ref.shape()))
    var ref_host = lat_ref.to_host(ctx)
    var harness = ParityHarness(0.999)
    var res = harness.compare_host(mine_host, ref_host)
    # magnitude ratio |mine|/|ref| (cos is magnitude-blind — check it too).
    var nm: Float64 = 0.0
    var nr: Float64 = 0.0
    for i in range(len(mine_host)):
        nm += Float64(mine_host[i]) * Float64(mine_host[i])
        nr += Float64(ref_host[i]) * Float64(ref_host[i])
    var mag_ratio = sqrt(nm) / sqrt(nr) if nr > 0.0 else 0.0
    print("[vae-enc-vid] cos           =", res.cos)
    print("[vae-enc-vid] max abs diff  =", res.max_abs)
    print("[vae-enc-vid] mag ratio     =", mag_ratio)
    if res.cos >= 0.999:
        print("[vae-enc-vid] GATE PASS (cos >= 0.999)")
    else:
        print("[vae-enc-vid] GATE FAIL (cos < 0.999)")
