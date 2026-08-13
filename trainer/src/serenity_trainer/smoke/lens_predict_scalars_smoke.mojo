# lens_predict_scalars_smoke.mojo — self-contained parity gate for the Lens
# predict-path SCALAR math (no DiT, no SLICE-A weights):
#   (1) scale_latents  — LensModel.scale_latents (LensModel.py:795-800):
#         latents_bn_mean = vae.bn.running_mean.view(1,-1,1,1)
#         latents_bn_std  = sqrt(vae.bn.running_var + vae.config.batch_norm_eps)
#         return (latents - mean) / std
#       Reference DATA: parity/lens/vae_bn.safetensors (running_mean/var [128]),
#       eps = batch_norm_eps from parity/lens/meta.json (0.0001).
#   (2) calculate_timestep_shift — LensModel.calculate_timestep_shift (LensModel.py
#         :754-765), using the checkpoint scheduler config (scheduler_config.json):
#         base_image_seq_len=256, max_image_seq_len=4096, base_shift=0.5,
#         max_shift=1.15, patch_size=2.
#         image_seq_len = (W//patch)*(H//patch)
#         m  = (max_shift-base_shift)/(max_seq-base_seq)
#         b  = base_shift - m*base_seq
#         mu = image_seq_len*m + b ; return exp(mu)
#   (3) compute_empirical_mu — the SAMPLER-path mu (lens_flowmatch.mojo), printed
#       for cross-check (LensSampler uses this, NOT calculate_timestep_shift).
#
# GATES:
#   * scale_latents output must be finite for every element (no NaN/Inf) — proves
#     the bn buffers + eps are read correctly and the divide is well-formed.
#   * calculate_timestep_shift must be finite and > 0 (it is exp(mu)).
# The mean/std/shift values are printed so the orchestrator can diff vs Serenity.

from max.gpu.host import DeviceContext
from std.math import sqrt, exp, isfinite
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors


# parity fixtures
comptime VAE_BN_PATH   = "trainer/parity/lens/vae_bn.safetensors"
comptime HIDDEN_PATH   = "trainer/parity/lens/dit_fwd_in_hidden.safetensors"

# meta.json batch_norm_eps (LensModel.scale_latents vae.config.batch_norm_eps).
comptime BATCH_NORM_EPS = Float32(0.0001)
# scheduler_config.json (FlowMatchEulerDiscreteScheduler).
comptime BASE_IMG_SEQ_LEN = 256
comptime MAX_IMG_SEQ_LEN  = 4096
comptime BASE_SHIFT       = Float32(0.5)
comptime MAX_SHIFT        = Float32(1.15)
comptime PATCH_SIZE       = 2
# oracle geometry (meta.json): patchified latent grid is s_img_h x s_img_w.
comptime LAT_H = 8
comptime LAT_W = 8
comptime BN_CHANNELS = 128   # patchified latent channels (vae_bn running_mean len)


# LensModel.calculate_timestep_shift (LensModel.py:754-765).
def calculate_timestep_shift(latent_height: Int, latent_width: Int) -> Float32:
    var image_seq_len = (latent_width // PATCH_SIZE) * (latent_height // PATCH_SIZE)
    var m = (MAX_SHIFT - BASE_SHIFT) / Float32(MAX_IMG_SEQ_LEN - BASE_IMG_SEQ_LEN)
    var b = BASE_SHIFT - m * Float32(BASE_IMG_SEQ_LEN)
    var mu = Float32(image_seq_len) * m + b
    return exp(mu)


def main() raises:
    var ctx = DeviceContext()
    print("=== Lens predict-scalars smoke (scale_latents + timestep_shift) ===")

    # ── load reference bn buffers + the fixed latent ──────────────────────────
    var bn = ShardedSafeTensors.open(String(VAE_BN_PATH))
    var mean_h = Tensor.from_view(bn.tensor_view(String("running_mean")), ctx).to_host(ctx)
    var var_h  = Tensor.from_view(bn.tensor_view(String("running_var")), ctx).to_host(ctx)
    if len(mean_h) != BN_CHANNELS or len(var_h) != BN_CHANNELS:
        raise Error(
            String("bn channel mismatch: mean=") + String(len(mean_h))
            + String(" var=") + String(len(var_h)) + String(" expected ")
            + String(BN_CHANNELS)
        )

    var hid = ShardedSafeTensors.open(String(HIDDEN_PATH))
    var lat_h = Tensor.from_view(hid.tensor_view(String("x")), ctx).to_host(ctx)  # [1,64,128]
    var n_tok = len(lat_h) // BN_CHANNELS    # 64

    # ── GATE 1: scale_latents = (z - mean) / sqrt(var + eps), per channel ──────
    # layout [1, n_tok, 128]: channel index = elem % 128.
    var out_min = Float32(1.0e30)
    var out_max = Float32(-1.0e30)
    var s = Float64(0.0)
    var s2 = Float64(0.0)
    var total = n_tok * BN_CHANNELS
    for i in range(total):
        var c = i % BN_CHANNELS
        var std = sqrt(var_h[c] + BATCH_NORM_EPS)
        var v = (lat_h[i] - mean_h[c]) / std
        if not isfinite(v):
            raise Error(String("scale_latents produced non-finite at i=") + String(i))
        if v < out_min: out_min = v
        if v > out_max: out_max = v
        s += Float64(v)
        s2 += Float64(v) * Float64(v)
    var mean = Float32(s / Float64(total))
    var variance = Float32(s2 / Float64(total)) - mean * mean
    var stdev = sqrt(variance) if variance > 0.0 else Float32(0.0)
    print("  GATE 1 scale_latents OK (all finite): mean=", mean, " std=", stdev,
          " min=", out_min, " max=", out_max)

    # ── GATE 2: calculate_timestep_shift (predict path) ───────────────────────
    var shift = calculate_timestep_shift(LAT_H, LAT_W)
    if not isfinite(shift) or shift <= 0.0:
        raise Error(String("calculate_timestep_shift non-finite/<=0: ") + String(shift))
    print("  GATE 2 calculate_timestep_shift(", LAT_H, ",", LAT_W, ") =", shift,
          " (finite, >0)")

    print("=== smoke complete ===")
