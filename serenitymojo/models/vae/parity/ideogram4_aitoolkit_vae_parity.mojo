# Parity: Ideogram-4 VAE *encoder* (training data path) vs **ai-toolkit** oracle.
#
# Oracle = ideogram4_aitoolkit_vae_oracle.py (ai-toolkit encode_images:556-578):
#   moments = vae.encoder(img)        # incl quant_conv
#   mean    = moments[:, :32]         # deterministic mode
#   patched = patchify_latents(mean, 2)
#   latents = (patched - shift) / scale   # per-128-ch get_latent_norm()
# ai-toolkit runs the WHOLE encode (incl the norm divide) in bf16 (vae dtype).
# The mojo data path (ldm_encoder.ideogram4_normalize_latents) does the divide in
# F32. The oracle dumps both: `latents` (prod bf16-norm) and `latents_f32norm`.
# We gate the mojo F32-path output against BOTH and FAIL LOUD on the prod ref.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg &&
#   pixi run mojo run -I . serenitymojo/models/vae/parity/ideogram4_aitoolkit_vae_parity.mojo
from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.vae.ldm_encoder import (
    load_ideogram4_vae_encoder,
    encode_ideogram4_latents,
)

comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity/ideogram4_aitoolkit_vae.safetensors"
comptime COS_BAR = 0.999


def _std(h: List[Float32]) raises -> Float64:
    var n = len(h)
    var s = Float64(0.0)
    var ss = Float64(0.0)
    for i in range(n):
        var v = Float64(h[i])
        s += v
        ss += v * v
    var m = s / Float64(n)
    var var_ = ss / Float64(n) - m * m
    return sqrt(var_ if var_ > 0.0 else 0.0)


def main() raises:
    var ctx = DeviceContext()
    # image 256x256 -> latent 32x32 -> LH=LW=32; packed gh=gw=16.
    var enc = load_ideogram4_vae_encoder[32, 32](VAE, ctx)
    var fx = ShardedSafeTensors.open(FX)

    var img = cast_tensor(Tensor.from_view(fx.tensor_view("image"), ctx), STDtype.BF16, ctx)
    var shift = Tensor.from_view(fx.tensor_view("latent_shift"), ctx)  # [128] F32
    var scale = Tensor.from_view(fx.tensor_view("latent_scale"), ctx)  # [128] F32

    # stage 1: raw moments [1,64,32,32] (encode_moments == ae.encoder incl quant_conv).
    # DIAGNOSTIC ONLY (not gated): the second 32 channels are logvar, a near-constant
    # band around -3 (std~0.33) that is UNUSED by the data path (mean-only mode) and
    # whose cosine is structurally fragile in bf16. We gate on `mean` + final latents.
    var moments = enc.encode_moments(img, ctx)
    var mom_exp = Tensor.from_view(fx.tensor_view("moments"), ctx).to_host(ctx)
    var r_mom = ParityHarness(COS_BAR).compare(moments, mom_exp, ctx)
    print("moments parity (full 64ch, mean+logvar, diagnostic):", r_mom)

    # stage 2: deterministic mean latent [1,32,32,32]
    var mean = enc.encode_mean(img, ctx)
    var mean_exp = Tensor.from_view(fx.tensor_view("mean"), ctx).to_host(ctx)
    var r_mean = ParityHarness(COS_BAR).compare(mean, mean_exp, ctx)
    print("mean parity:", r_mean)

    # stage 3: full normalized packed latents [1,128,16,16] (mojo F32 norm)
    var lat = encode_ideogram4_latents[32, 32](enc, img, shift, scale, ctx)
    var ls = lat.shape()
    print("latents shape:", ls[0], ls[1], ls[2], ls[3], "(expect 1 128 16 16)")
    if ls[0] != 1 or ls[1] != 128 or ls[2] != 16 or ls[3] != 16:
        raise Error("latents shape mismatch")

    var lat_host = lat.to_host(ctx)
    var lat_std = _std(lat_host)

    # PROD reference (ai-toolkit bf16-norm) — the gate that matters.
    var lat_prod = Tensor.from_view(fx.tensor_view("latents"), ctx).to_host(ctx)
    var lat_prod_std = _std(lat_prod)
    var r_prod = ParityHarness(COS_BAR).compare(lat, lat_prod, ctx)
    print("latents vs PROD (ai-toolkit bf16-norm):", r_prod)

    # F32-norm reference (matches mojo's norm dtype) — diagnostic.
    var lat_f32 = Tensor.from_view(fx.tensor_view("latents_f32norm"), ctx).to_host(ctx)
    var r_f32 = ParityHarness(COS_BAR).compare(lat, lat_f32, ctx)
    print("latents vs f32norm (mojo-matching norm):", r_f32)

    print("STD  mojo:", lat_std, " ai-toolkit prod:", lat_prod_std)
    # ~0.83 is the TRUE ideogram4 latent-norm std (per-ch scale 1.6-1.9 over-divides
    # slightly). A drastically lower std (~0.5) would flag an HWC<->CHW scramble.
    if lat_std < 0.70:
        raise Error(
            "latent std too low ("
            + String(lat_std)
            + ") — possible HWC<->CHW scramble in the data path"
        )

    # FAIL LOUD on the production data-path references: mean (the only moments half
    # used) and the final normalized latents vs ai-toolkit's bf16-norm production.
    if not r_mean.passed:
        raise Error(
            "ideogram4 VAE encode MEAN parity FAILED: cos="
            + String(r_mean.cos)
            + " < "
            + String(COS_BAR)
        )
    if not r_prod.passed:
        raise Error(
            "ideogram4 ai-toolkit VAE encode parity FAILED: cos="
            + String(r_prod.cos)
            + " < "
            + String(COS_BAR)
        )

    print("ideogram4 ai-toolkit VAE encode parity OK")
