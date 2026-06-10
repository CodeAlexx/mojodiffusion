# DEV-ONLY parity oracle for the Ideogram-4 VAE *encoder* (training path).
# Mirrors ai-toolkit ideogram4.py::encode_images (lines 455-476):
#   moments = vae.encoder(image)            # [1,64,H/8,W/8]  (NO quant_conv — encoder() is called directly)
#   mean    = moments[:, :32]               # DiagonalGaussian mode (deterministic; training uses mean)
#   patched = patchify_latents(mean, 2)     # [1,128,gh,gw]
#   latents = (patched - shift) / scale     # per-128-channel latent norm
# Dumps a fixed image + every stage so the Mojo encoder can be gated element-wise.
# Reference AutoEncoder = the SAME ideogram4-ref module the decoder gate (chunk8) uses.
import sys, os, json, torch
sys.path.insert(0, "/home/alex/ideogram4-ref/src")
from safetensors.torch import load_file, save_file
from ideogram4.autoencoder import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict
from ideogram4.latent_norm import get_latent_norm

ROOT = "/home/alex/.serenity/models/ideogram-4-fp8"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"
dev = torch.device("cuda")
dt = torch.bfloat16
PATCH = 2


def patchify_latents(z, patch_size=2):
    # ai-toolkit src/pipeline.py:40 — (B,ae_ch,H8,W8) -> (B, ae_ch*patch^2, gh, gw)
    b, ae_ch, h8, w8 = z.shape
    ph = pw = patch_size
    gh, gw = h8 // ph, w8 // pw
    z = z.view(b, ae_ch, gh, ph, gw, pw)
    z = z.permute(0, 3, 5, 1, 2, 4).reshape(b, ph * pw * ae_ch, gh, gw)
    return z


def main():
    H = W = 256  # -> H/8 = 32, gh = gw = 16, 256 image tokens
    ae = AutoEncoder(AutoEncoderParams())
    ae.load_state_dict(convert_diffusers_state_dict(load_file(f"{ROOT}/vae/diffusion_pytorch_model.safetensors")))
    ae.to(device=dev, dtype=dt)
    ae.eval()
    ae_ch = ae.params.z_channels
    print(f"[V] AutoEncoder loaded, z_channels={ae_ch}")

    torch.manual_seed(7)
    img = (torch.rand(1, 3, H, W, device=dev, dtype=torch.float32) * 2.0 - 1.0)  # [-1,1]
    with torch.no_grad():
        moments = ae.encoder(img.to(dt))            # [1,64,32,32]
    mean = moments[:, :ae_ch]                        # [1,32,32,32]
    patched = patchify_latents(mean.float(), PATCH)  # [1,128,16,16]
    shift, scale = get_latent_norm()
    shift = shift.view(1, -1, 1, 1).to(patched.device, torch.float32)
    scale = scale.view(1, -1, 1, 1).to(patched.device, torch.float32)
    latents = (patched - shift) / scale              # [1,128,16,16]

    fx = {
        "image": img.float().cpu(),                  # [1,3,256,256] in [-1,1]
        "moments": moments.float().cpu(),            # [1,64,32,32]
        "mean": mean.float().cpu(),                  # [1,32,32,32]
        "patched": patched.float().cpu(),            # [1,128,16,16]
        "latents": latents.float().cpu(),            # [1,128,16,16] normalized (== batch.latents)
        "latent_shift": shift.reshape(-1).cpu(),     # [128]
        "latent_scale": scale.reshape(-1).cpu(),     # [128]
    }
    save_file(fx, f"{OUT}/ideogram4_fx_vae_encode.safetensors")
    json.dump(
        {"H": H, "W": W, "ae_ch": int(ae_ch), "patch": PATCH,
         "gh": H // 8 // PATCH, "gw": W // 8 // PATCH,
         "moments_std": float(moments.float().std()),
         "mean_std": float(mean.float().std()),
         "latents_std": float(latents.std()),
         "img_seed": 7},
        open(f"{OUT}/ideogram4_fx_vae_encode_meta.json", "w"), indent=2,
    )
    print(f"[V] saved encode fixture: moments{tuple(moments.shape)} mean_std={mean.float().std():.4f} "
          f"latents{tuple(latents.shape)} latents_std={latents.std():.4f}")


if __name__ == "__main__":
    main()
