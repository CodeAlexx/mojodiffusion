# Probe: LingBotWanVaeEncoder.encode_video_tiled (Wan2.1-style 3D causal VAE),
# SPATIALLY-TILED temporal mode. Loads the REAL encoder.* + quant_conv.* weights
# plus a deterministic input clip from oracle_vae_encode_tiled.safetensors (key
# `pixel`, [1,3,13,384,384]), runs encode_video_tiled (2x2 overlapping 256px
# tiles, stride 192, latent blend 8), and compares to oracle key `latent`
# [1,16,4,48,48] with ParityHarness(0.999). Prints cos + max_abs + mag ratio.
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . \
#       serenitymojo/models/lingbotvideo/parity/vae_encode_tiled_probe.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.lingbotvideo.vae_encoder import LingBotWanVaeEncoder
from serenitymojo.parity import ParityHarness

comptime VAE_WEIGHTS = "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime H = 384
comptime W = 384
comptime T = 13


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

    print("[vae-enc-tiled] loading LingBot Wan VAE encoder weights from", VAE_WEIGHTS)
    var enc = LingBotWanVaeEncoder[H, W].load(String(VAE_WEIGHTS), ctx)

    var oracle_path = String(PARITY_DIR) + "/oracle_vae_encode_tiled.safetensors"
    print("[vae-enc-tiled] loading oracle clip from", oracle_path)
    var clip: Tensor
    var have_oracle = True
    try:
        var st0 = ShardedSafeTensors.open(oracle_path)
        clip = Tensor.from_view_as_f32(st0.tensor_view("pixel"), ctx)
    except e:
        print("[vae-enc-tiled] WARNING: oracle missing (", e, "); synthesizing")
        have_oracle = False
        # deterministic clip in [0,1], NCTHW [1,3,T,H,W].
        var vals = List[Float32]()
        var cnt = 1 * 3 * T * H * W
        var seed: Int = 12345
        for _i in range(cnt):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            vals.append(Float32(seed % 1000) / Float32(1000.0))
        var shp = List[Int]()
        shp.append(1); shp.append(3); shp.append(T); shp.append(H); shp.append(W)
        clip = Tensor.from_host(vals, shp^, STDtype.F32, ctx)
    print("  clip shape =", _sh(clip.shape()))

    print("[vae-enc-tiled] running encode_video_tiled (2x2 tiles, stride 192)")
    var out = enc.encode_video_tiled(clip, ctx)
    print("  encoded shape =", _sh(out.shape()))

    var mine_host = out.to_host(ctx)
    _stats(mine_host)

    if not have_oracle:
        print("[vae-enc-tiled] no oracle -> shape/stats only (no gate)")
        return

    # ── parity gate: cos >= 0.999 vs the diffusers oracle latent + magnitude ──
    var st = ShardedSafeTensors.open(oracle_path)
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
    print("[vae-enc-tiled] cos           =", res.cos)
    print("[vae-enc-tiled] max abs diff  =", res.max_abs)
    print("[vae-enc-tiled] mag ratio     =", mag_ratio)
    if res.cos >= 0.999:
        print("[vae-enc-tiled] GATE PASS (cos >= 0.999)")
    else:
        print("[vae-enc-tiled] GATE FAIL (cos < 0.999)")
