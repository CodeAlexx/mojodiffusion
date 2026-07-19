# sd3_encoder_oracle.py — SD3.5 embedded VAE ENCODER bf16 GPU oracle (16ch).
#
# Builds the diffusers AutoencoderKL ENCODER straight from the SD3.5 Medium
# single-file checkpoint (the SAME embedded first_stage_model VAE weights the
# Mojo encoder loads) and dumps the reference MOMENTS (mean|logvar, the
# conv_out output, UNSCALED — SD3 VAE has no quant_conv) and the MODE (mean,
# first 16 channels) for a FIXED deterministic [1,3,IH,IW] input image in
# [-1,1], computed on the GPU in bfloat16 (matching the faithful Mojo BF16
# path; SD3.5 ships BF16 VAE weights on disk).
#
# DEV TOOL ONLY — never imported by shipped Mojo. Outputs (256x256 -> 32x32):
#   parity/sd3enc_img_256x256.bin        [1,3,256,256] f32, the fixed input
#   parity/sd3enc_moments_256x256.bin    [1,32,32,32]  f32 NHWC moments (mu|logvar)
#   parity/sd3enc_mode_256x256.bin       [1,16,32,32]  f32 NCHW, the mode (mean)
#   parity/sd3enc_moments_f32_256x256.bin[1,32,32,32]  f32 NHWC, F32 ceiling oracle
#   parity/sd3enc_meta_256x256.json      shapes + dtype note
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/models/vae/parity/sd3_encoder_oracle.py

import os
import json
import numpy as np
import torch
from diffusers import AutoencoderKL

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/home/alex/.serenity/models/checkpoints/stablediffusion35_medium.safetensors"
IH = IW = 256
LH = LW = IH // 8          # 32
LATENT_CH = 16
ZC2 = 2 * LATENT_CH        # 32


def _encode_moments_mode(vae, img):
    with torch.no_grad():
        posterior = vae.encode(img).latent_dist
        moments = posterior.parameters  # [1, 32, LH, LW] NCHW (UNSCALED)
        mode = posterior.mode()         # [1, 16, LH, LW] NCHW
    return moments, mode


def main():
    assert torch.cuda.is_available(), "GPU required"
    dev = torch.device("cuda")

    # Fixed deterministic input EXACTLY matching the Mojo probe ramp:
    #   x = (i % 251) / 125 - 1, row-major over [1,3,IH,IW].
    n = 1 * 3 * IH * IW
    ramp = np.array([(i % 251) / 125.0 - 1.0 for i in range(n)], dtype=np.float32)
    img_f32 = ramp.reshape(1, 3, IH, IW)
    img_f32.tofile(os.path.join(HERE, "sd3enc_img_256x256.bin"))

    # Build the diffusers AutoencoderKL from the SAME single-file SD3.5 checkpoint
    # (embedded first_stage_model VAE). from_single_file remaps the LDM keys to
    # diffusers naming, so these are the identical weights the Mojo encoder loads.
    vae = AutoencoderKL.from_single_file(CKPT)
    assert vae.config.latent_channels == LATENT_CH, vae.config.latent_channels
    # SD3 VAE has no quant_conv.
    assert getattr(vae, "quant_conv", None) is None, "unexpected quant_conv"

    # BF16-on-GPU oracle (the faithful path).
    vae_bf16 = vae.to(dev, dtype=torch.bfloat16).eval()
    img_bf16 = torch.from_numpy(img_f32).to(dev, dtype=torch.bfloat16)
    moments_bf16, mode_bf16 = _encode_moments_mode(vae_bf16, img_bf16)

    # F32-on-GPU ceiling oracle (for the BF16-vs-F32 self-delta reference).
    vae_f32 = vae.to(dev, dtype=torch.float32).eval()
    img_f32t = torch.from_numpy(img_f32).to(dev, dtype=torch.float32)
    moments_f32o, _ = _encode_moments_mode(vae_f32, img_f32t)

    def to_nhwc(t):
        a = t.float().cpu().numpy()                 # [1,C,LH,LW]
        return np.transpose(a, (0, 2, 3, 1)).copy()  # [1,LH,LW,C]

    moments_nhwc = to_nhwc(moments_bf16)            # [1,LH,LW,32]
    moments_nhwc.astype(np.float32).tofile(
        os.path.join(HERE, "sd3enc_moments_256x256.bin")
    )
    moments_f32_nhwc = to_nhwc(moments_f32o)
    moments_f32_nhwc.astype(np.float32).tofile(
        os.path.join(HERE, "sd3enc_moments_f32_256x256.bin")
    )

    mode_f32 = mode_bf16.float().cpu().numpy().astype(np.float32)  # NCHW [1,16,LH,LW]
    mode_f32.tofile(os.path.join(HERE, "sd3enc_mode_256x256.bin"))

    meta = {
        "img_shape": [1, 3, IH, IW],
        "moments_shape_nhwc": [1, LH, LW, ZC2],
        "mode_shape_nchw": [1, LATENT_CH, LH, LW],
        "dtype": "bf16-on-gpu, dumped as f32",
        "checkpoint": CKPT,
        "embedded_vae": "first_stage_model (via from_single_file)",
        "unscaled": True,
        "latent_channels": LATENT_CH,
    }
    with open(os.path.join(HERE, "sd3enc_meta_256x256.json"), "w") as f:
        json.dump(meta, f, indent=2)

    def stats(name, a):
        a = a.reshape(-1)
        print(
            f"[oracle] {name} n={a.size} min={a.min():.4f} max={a.max():.4f} "
            f"mean={a.mean():.4f} std={a.std():.4f}"
        )

    # BF16-vs-F32 oracle self-delta (the realistic ceiling for the Mojo BF16 port).
    mb = moments_nhwc.reshape(-1).astype(np.float64)
    mf = moments_f32_nhwc.reshape(-1).astype(np.float64)
    cos = float(mb @ mf / (np.linalg.norm(mb) * np.linalg.norm(mf)))
    print(f"[oracle] diffusers BF16-vs-F32 moments self-cos = {cos:.6f}")

    stats("moments(bf16)", moments_nhwc)
    stats("mode(bf16)", mode_f32)
    print("[oracle] dumped moments + mode + img to", HERE)

    del vae, vae_bf16, vae_f32
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
