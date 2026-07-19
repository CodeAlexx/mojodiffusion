# DEV-ONLY parity oracle for the Ideogram-4 VAE *encoder* (training data path),
# oracled against **ai-toolkit** (NOT ideogram4-ref).
#
# Mirrors EXACTLY the ai-toolkit production encode used to build training latents:
#   ai-toolkit/extensions_built_in/diffusion_models/ideogram4/ideogram4.py
#     encode_images() lines 556-578:
#       ae_channels = vae.params.z_channels            # 32
#       moments = self.vae.encoder(images)             # incl. quant_conv (vae.py:225)
#       mean    = moments[:, :ae_channels]             # deterministic mode (training)
#       patched = patchify_latents(mean, patch_size=2) # [B,128,gh,gw]  (src/pipeline.py:80)
#       shift   = self._latent_shift.to(patched.dtype) # get_latent_norm() -> [128]
#       scale   = self._latent_scale.to(patched.dtype)
#       latents = (patched - shift) / scale            # IN VAE DTYPE (bf16 in prod)
#
# Production dtype: _load_vae() loads the VAE with self.torch_dtype, and
# encode_images() casts images to self.vae_torch_dtype (defaults to model dtype).
# Training runs bf16, so the ENTIRE encode incl. the latent-norm divide runs in bf16.
# We dump the true bf16 production latents AND an f32-norm variant so the gate can
# quantify the mojo F32-norm choice (ldm_encoder.mojo:626 ideogram4_normalize_latents
# casts to F32 before the divide).
#
# Uses ai-toolkit's OWN vae.py + latent_norm.py + pipeline.patchify_latents.
import sys, os, json, importlib.util, torch
from safetensors.torch import load_file, save_file

# Load ai-toolkit's ideogram4 src modules DIRECTLY by file path. We cannot
# `import extensions_built_in...` because that package's __init__ chain pulls in
# chroma -> torchao (not installed in this venv). vae.py / latent_norm.py only
# need torch+einops; patchify_latents only needs torch. So load each file as an
# isolated module — same source ai-toolkit runs, no package side effects.
SRC = "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4/src"


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_vae = _load("aitk_i4_vae", f"{SRC}/vae.py")
_ln = _load("aitk_i4_latent_norm", f"{SRC}/latent_norm.py")
AutoEncoder = _vae.AutoEncoder
AutoEncoderParams = _vae.AutoEncoderParams
convert_diffusers_state_dict = _vae.convert_diffusers_state_dict
get_latent_norm = _ln.get_latent_norm


# Inlined verbatim from ai-toolkit src/pipeline.py:80-88 (patch=2). pipeline.py
# itself has a relative `.transformer` import + transformers.masking_utils that
# won't resolve when loaded standalone; this fn is pure torch and self-contained.
def patchify_latents(z: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """(B, ae_ch, H8, W8) -> (B, ae_ch * patch**2, gh, gw)."""
    b, ae_ch, h8, w8 = z.shape
    ph = pw = patch_size
    gh, gw = h8 // ph, w8 // pw
    z = z.view(b, ae_ch, gh, ph, gw, pw)
    z = z.permute(0, 3, 5, 1, 2, 4).reshape(b, ph * pw * ae_ch, gh, gw)
    return z

ROOT = "/home/alex/.serenity/models/ideogram-4-fp8"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"
dev = torch.device("cuda")
DT = torch.bfloat16   # production vae dtype (model dtype == bf16)
PATCH = 2


def main():
    H = W = 256  # -> H/8 = 32, gh = gw = 16, 256 image tokens
    ae = AutoEncoder(AutoEncoderParams())
    ae.load_state_dict(
        convert_diffusers_state_dict(load_file(f"{ROOT}/vae/diffusion_pytorch_model.safetensors"))
    )
    ae.to(device=dev, dtype=DT)
    ae.eval()
    ae.requires_grad_(False)
    ae_ch = ae.params.z_channels
    print(f"[V] ai-toolkit AutoEncoder loaded, z_channels={ae_ch}, dtype={DT}")

    # Same deterministic input the prior oracle (and mojo probe) used: seed 7,
    # randn-uniform [-1,1], so the mojo gate can reuse one fixed image tensor.
    torch.manual_seed(7)
    img = (torch.rand(1, 3, H, W, device=dev, dtype=torch.float32) * 2.0 - 1.0)  # [-1,1]

    with torch.no_grad():
        # PRODUCTION encode_images path, exact dtype flow.
        images = img.to(dev, dtype=DT)
        moments = ae.encoder(images)          # [1,64,32,32] bf16 (incl. quant_conv)
        mean = moments[:, :ae_ch]             # [1,32,32,32] bf16 (deterministic mode)
        patched = patchify_latents(mean, PATCH)  # [1,128,16,16] bf16

        shift_f32, scale_f32 = get_latent_norm()                 # [128] f32
        shift_b = shift_f32.view(1, -1, 1, 1).to(patched.device, patched.dtype)  # bf16
        scale_b = scale_f32.view(1, -1, 1, 1).to(patched.device, patched.dtype)  # bf16
        latents_prod = (patched - shift_b) / scale_b             # [1,128,16,16] bf16 (PROD)

        # F32-norm variant (matches mojo ideogram4_normalize_latents: F32 divide).
        patched_f = patched.float()
        shift_f = shift_f32.view(1, -1, 1, 1).to(patched.device, torch.float32)
        scale_f = scale_f32.view(1, -1, 1, 1).to(patched.device, torch.float32)
        latents_f32 = (patched_f - shift_f) / scale_f            # [1,128,16,16] f32

    fx = {
        "image": img.float().cpu(),                      # [1,3,256,256] in [-1,1] (F32, fed as bf16)
        "moments": moments.float().cpu(),                # [1,64,32,32]
        "mean": mean.float().cpu(),                      # [1,32,32,32]
        "patched": patched.float().cpu(),                # [1,128,16,16] (bf16 patchify, stored f32)
        "latents": latents_prod.float().cpu(),           # [1,128,16,16] PROD bf16-norm == batch.latents
        "latents_f32norm": latents_f32.float().cpu(),    # [1,128,16,16] f32-norm variant
        "latent_shift": shift_f32.reshape(-1).cpu(),     # [128] f32
        "latent_scale": scale_f32.reshape(-1).cpu(),     # [128] f32
    }
    save_file(fx, f"{OUT}/ideogram4_aitoolkit_vae.safetensors")
    json.dump(
        {
            "oracle": "ai-toolkit",
            "H": H, "W": W, "ae_ch": int(ae_ch), "patch": PATCH,
            "gh": H // 8 // PATCH, "gw": W // 8 // PATCH,
            "vae_dtype": "bfloat16",
            "norm_dtype_prod": "bfloat16",
            "norm_dtype_f32variant": "float32",
            "moments_std": float(moments.float().std()),
            "mean_std": float(mean.float().std()),
            "latents_std_prod": float(latents_prod.float().std()),
            "latents_std_f32norm": float(latents_f32.std()),
            "img_seed": 7,
            "uses_self_bn": False,  # BatchNorm2d self.bn is defined but NOT called in encode_images
        },
        open(f"{OUT}/ideogram4_aitoolkit_vae_meta.json", "w"), indent=2,
    )
    print(
        f"[V] saved ai-toolkit encode fixture: moments{tuple(moments.shape)} "
        f"mean_std={mean.float().std():.4f} latents{tuple(latents_prod.shape)} "
        f"latents_std_prod={latents_prod.float().std():.4f} "
        f"latents_std_f32norm={latents_f32.std():.4f}"
    )


if __name__ == "__main__":
    main()
