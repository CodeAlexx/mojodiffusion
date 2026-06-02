# qwenimage_encoder_oracle.py — torch reference for the Qwen-Image VAE ENCODER.
#
# Builds an AutoencoderKLQwenImage from the qwen-image-2512 config, loads the
# ANIMA weights (qwen_image_vae.safetensors, native Wan key names) into it by
# renaming Wan keys -> diffusers keys, then encodes a FIXED non-degenerate RGB
# image (in [-1,1], OT RescaleImageChannels convention) and dumps:
#   img_<HxW>.bin     [1,3,H,W]   the fixed input image (F32)
#   latmean_<HxW>.bin [1,16,1,H/8,W/8] the diagonal-gaussian MEAN latent (F32)
#
# WEIGHT RECONCILIATION (proven by qwenimage_encoder_oracle.py's loader):
#   The anima file and the qwen-image-2512 file are BYTE-IDENTICAL tensors under
#   two naming schemes (194 tensors each). We load the ANIMA weights here so the
#   Mojo encoder (which reads the SAME anima .safetensors with Wan keys) and this
#   torch oracle use literally the same bytes.
#
# DEV-ONLY parity oracle. Never shipped. Run with system python3 (has diffusers
# 0.38 AutoencoderKLQwenImage + torch cu128).

import json
import struct
import sys
import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage

ANIMA_VAE = "/home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors"
CFG = "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/config.json"
OUTDIR = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = f.tell()
        blob = f.read()
    hdr.pop("__metadata__", None)
    out = {}
    for k, v in hdr.items():
        s, e = v["data_offsets"]
        dt = v["dtype"]
        assert dt == "BF16", dt
        arr = np.frombuffer(blob[s:e], dtype=np.uint16).copy()
        t = torch.from_numpy(arr).view(torch.bfloat16).reshape(v["shape"]).float()
        out[k] = t
    return out


def wan_key_to_diffusers(k):
    """Map an anima (Wan) state-dict key to the diffusers AutoencoderKLQwenImage
    key. Only ENCODER + quant_conv keys are handled here (decoder ignored)."""
    if k == "conv1.weight":
        return "quant_conv.weight"
    if k == "conv1.bias":
        return "quant_conv.bias"
    if not k.startswith("encoder."):
        return None  # decoder.* / conv2.* — not needed for encode
    rest = k[len("encoder."):]
    # init / head
    if rest.startswith("conv1."):
        return "encoder.conv_in." + rest[len("conv1."):]
    if rest.startswith("head.0."):
        return "encoder.norm_out." + rest[len("head.0."):]
    if rest.startswith("head.2."):
        return "encoder.conv_out." + rest[len("head.2."):]
    # middle: encoder.middle.0/1/2 -> mid_block.resnets.0 / attentions.0 / resnets.1
    if rest.startswith("middle.0."):
        return "encoder.mid_block.resnets.0." + _res_sub(rest[len("middle.0."):])
    if rest.startswith("middle.1."):
        return "encoder.mid_block.attentions.0." + rest[len("middle.1."):]
    if rest.startswith("middle.2."):
        return "encoder.mid_block.resnets.1." + _res_sub(rest[len("middle.2."):])
    # downsamples.{n}.* -> down_blocks.{n}.*
    if rest.startswith("downsamples."):
        tail = rest[len("downsamples."):]
        idx, sub = tail.split(".", 1)
        return f"encoder.down_blocks.{idx}." + _down_sub(sub)
    raise KeyError(k)


def _res_sub(sub):
    """residual.0.gamma->norm1.gamma, residual.2.*->conv1.*, residual.3.gamma->
    norm2.gamma, residual.6.*->conv2.*, shortcut.*->conv_shortcut.*."""
    if sub.startswith("residual.0."):
        return "norm1." + sub[len("residual.0."):]
    if sub.startswith("residual.2."):
        return "conv1." + sub[len("residual.2."):]
    if sub.startswith("residual.3."):
        return "norm2." + sub[len("residual.3."):]
    if sub.startswith("residual.6."):
        return "conv2." + sub[len("residual.6."):]
    if sub.startswith("shortcut."):
        return "conv_shortcut." + sub[len("shortcut."):]
    raise KeyError(sub)


def _down_sub(sub):
    # resample.* and time_conv.* pass through unchanged; residual/shortcut remap
    if sub.startswith("resample.") or sub.startswith("time_conv."):
        return sub
    return _res_sub(sub)


def main():
    cfg = json.load(open(CFG))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    model = AutoencoderKLQwenImage.from_config(cfg)
    model = model.eval()

    raw = read_safetensors(ANIMA_VAE)
    sd = model.state_dict()
    mapped = {}
    used = 0
    for k, v in raw.items():
        dk = wan_key_to_diffusers(k)
        if dk is None:
            continue
        assert dk in sd, f"target missing: {dk} (from {k})"
        assert tuple(sd[dk].shape) == tuple(v.shape), f"shape {k}->{dk}: {v.shape} vs {sd[dk].shape}"
        mapped[dk] = v
        used += 1
    # load encoder + quant_conv; leave decoder as-is (unused)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    enc_missing = [m for m in missing if m.startswith("encoder.") or m.startswith("quant_conv")]
    assert not enc_missing, f"encoder keys not filled: {enc_missing[:5]}"
    print(f"[oracle] mapped {used} encoder/quant tensors; encoder fully covered")

    model = model.to("cuda", dtype=torch.float32)

    H = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    g = torch.Generator(device="cpu").manual_seed(1234)
    # fixed non-degenerate image, then map [0,1)->[-1,1] (OT RescaleImageChannels)
    img01 = torch.rand(1, 3, H, W, generator=g, dtype=torch.float32)
    img = img01 * 2.0 - 1.0  # [-1,1]
    x = img.to("cuda")
    # lift to 1 video frame: [1,3,1,H,W]
    x5 = x.unsqueeze(2)
    with torch.no_grad():
        posterior = model.encode(x5).latent_dist
        mean = posterior.mode()  # [1,16,1,H/8,W/8]  (mode == mean)
    mean = mean.float().cpu().contiguous()
    std = posterior.mode().float().std().item()
    print(f"[oracle] img {tuple(img.shape)} latent_mean {tuple(mean.shape)} latent_std={mean.std().item():.4f}")

    tag = f"{H}x{W}"
    img.cpu().contiguous().numpy().astype("<f4").tofile(f"{OUTDIR}/qie_img_{tag}.bin")
    mean.numpy().astype("<f4").tofile(f"{OUTDIR}/qie_latmean_{tag}.bin")
    with open(f"{OUTDIR}/qie_meta_{tag}.json", "w") as f:
        json.dump({"H": H, "W": W, "lat_shape": list(mean.shape),
                   "latent_std": float(mean.std().item())}, f)
    print(f"[oracle] wrote qie_img_{tag}.bin, qie_latmean_{tag}.bin")


if __name__ == "__main__":
    main()
