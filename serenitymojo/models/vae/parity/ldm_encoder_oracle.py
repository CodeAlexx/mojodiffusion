# ldm_encoder_oracle.py — SDXL/LDM AutoencoderKL ENCODER bf16 GPU oracle.
#
# Dumps the reference encoder MOMENTS (mean|logvar, the conv_out+quant_conv
# output, UNSCALED) and the MODE (mean, first latent_ch channels) for a FIXED
# deterministic [1,3,64,64] input image in [-1,1], computed by diffusers
# AutoencoderKL on the GPU in bfloat16 — matching the faithful Mojo BF16 path
# (Rust ldm_encoder.rs casts weights+input to BF16; rs:566-576).
#
# Weights: madebyollin sdxl-vae-fp16-fix (the fp16-safe SDXL VAE).
#
# DEV TOOL ONLY — never imported by shipped Mojo. Outputs:
#   parity/ldmenc_img_64x64.bin       [1,3,64,64] f32, the fixed input (so the
#                                      Mojo side reads the EXACT same pixels)
#   parity/ldmenc_moments_64x64.bin   [1,8,8,8]  f32, NHWC moments (mean|logvar)
#   parity/ldmenc_mode_64x64.bin      [1,4,8,8]  f32 NCHW, the mode (mean) latent
#   parity/ldmenc_meta_64x64.json     shapes + dtype note
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/models/vae/parity/ldm_encoder_oracle.py

import os
import json
import numpy as np
import torch
from diffusers import AutoencoderKL

HERE = os.path.dirname(os.path.abspath(__file__))
VAE_DIR = "/home/alex/madebyollin_sdxl-vae-fp16-fix"
IH = IW = 64
LH = LW = IH // 8


def main():
    assert torch.cuda.is_available(), "GPU required"
    dev = torch.device("cuda")

    # Fixed deterministic input EXACTLY matching the Mojo probe ramp:
    #   x = (i % 251) / 125 - 1, row-major over [1,3,64,64].
    n = 1 * 3 * IH * IW
    ramp = np.array([(i % 251) / 125.0 - 1.0 for i in range(n)], dtype=np.float32)
    img_f32 = ramp.reshape(1, 3, IH, IW)
    img_f32.tofile(os.path.join(HERE, "ldmenc_img_64x64.bin"))

    # ONE model, bf16, on GPU.
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.bfloat16)
    vae = vae.to(dev).eval()

    img = torch.from_numpy(img_f32).to(dev, dtype=torch.bfloat16)

    with torch.no_grad():
        # AutoencoderKL.encode returns AutoencoderKLOutput.latent_dist
        # (DiagonalGaussianDistribution); .parameters is the raw moments
        # (mean|logvar) = conv_out -> quant_conv output, UNSCALED. .mode() is the
        # mean (first latent_ch channels).
        posterior = vae.encode(img).latent_dist
        moments = posterior.parameters  # [1, 2*latent_ch, LH, LW] NCHW
        mode = posterior.mode()         # [1, latent_ch, LH, LW] NCHW

    # moments NCHW [1,8,LH,LW] -> NHWC [1,LH,LW,8] to match the Mojo encoder's
    # encode_moments output layout (NHWC, channel-last).
    moments_f32 = moments.float().cpu().numpy()        # [1,8,LH,LW]
    moments_nhwc = np.transpose(moments_f32, (0, 2, 3, 1)).copy()  # [1,LH,LW,8]
    moments_nhwc.astype(np.float32).tofile(
        os.path.join(HERE, "ldmenc_moments_64x64.bin")
    )

    mode_f32 = mode.float().cpu().numpy().astype(np.float32)  # NCHW [1,4,LH,LW]
    mode_f32.tofile(os.path.join(HERE, "ldmenc_mode_64x64.bin"))

    meta = {
        "img_shape": [1, 3, IH, IW],
        "moments_shape_nhwc": [1, LH, LW, 8],
        "mode_shape_nchw": [1, 4, LH, LW],
        "dtype": "bf16-on-gpu, dumped as f32",
        "vae": VAE_DIR,
        "unscaled": True,
    }
    with open(os.path.join(HERE, "ldmenc_meta_64x64.json"), "w") as f:
        json.dump(meta, f, indent=2)

    def stats(name, a):
        a = a.reshape(-1)
        print(
            f"[oracle] {name} n={a.size} min={a.min():.4f} max={a.max():.4f} "
            f"mean={a.mean():.4f} std={a.std():.4f}"
        )

    stats("moments", moments_nhwc)
    stats("mode", mode_f32)
    print("[oracle] dumped moments + mode + img to", HERE)

    del vae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
