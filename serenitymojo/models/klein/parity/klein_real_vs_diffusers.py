#!/usr/bin/env python3
# klein_real_vs_diffusers.py — REAL-dim forward parity: Mojo Klein vs diffusers Flux2.
#
# Purpose: the Mojo toy-dim gates only validate the Mojo math against a hand-
# transcription of itself. They CANNOT catch a divergence between the Mojo
# forward and the REAL Flux2 model. This loads the authoritative diffusers
# Flux2Transformer2DModel (real Klein-4B weights via from_single_file), runs it
# on the SAME inputs the Mojo trainer used (dumped to a safetensors by the Mojo
# side), and compares the velocity output — to localize the systematic
# mean-offset / under-magnitude bias seen in training (loss ~2 vs baseline ~0.75).
#
# INPUT: a dump safetensors written by the Mojo trainer (see the dump block added
# to train_klein_real.mojo) containing:
#   x_t        [1024, 128]  F32   the noised img tokens fed to the forward
#   text       [512, 7680]  F32   encoder_hidden_states (cached Qwen3 embedding)
#   velocity   [1024, 128]  F32   the Mojo forward output (to compare against)
#   sigma      [1]          F32   the timestep (in [0,1]; diffusers ×1000 internally)
#
# Run (SEPARATE command, never chained after a mojo build):
#   /home/alex/serenity/venv/bin/python \
#     serenitymojo/models/klein/parity/klein_real_vs_diffusers.py /tmp/klein_dump.safetensors

import sys
import json
import struct
import math
import numpy as np
import torch

CKPT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors"
DUMP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/klein_dump.safetensors"

N_IMG = 1024
N_TXT = 512
IMG_W = 32  # 32x32 latent token grid


def load_safetensors(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            fh.seek(base + s)
            raw = fh.read(e - s)
            dt = v["dtype"]
            if dt == "F32":
                a = np.frombuffer(raw, np.float32)
            elif dt == "BF16":
                u = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16)
                a = u.view(np.float32)
            elif dt == "F16":
                a = np.frombuffer(raw, np.float16).astype(np.float32)
            else:
                raise ValueError(f"dtype {dt}")
            out[k] = a.reshape(v["shape"])
        return out


def stats(name, a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"  {name}: cos={cos:.6f}  "
          f"a(mean={a.mean():+.4f},std={a.std():.4f})  "
          f"b(mean={b.mean():+.4f},std={b.std():.4f})  "
          f"maxabsdiff={np.abs(a-b).max():.4f}")
    return cos


def main():
    print(f"=== loading dump {DUMP} ===")
    d = load_safetensors(DUMP)
    for k in d:
        print(f"   {k} {d[k].shape}")
    x_t = torch.from_numpy(d["x_t"].copy()).float().cuda().unsqueeze(0)        # [1,1024,128]
    text = torch.from_numpy(d["text"].copy()).float().cuda().unsqueeze(0)      # [1,512,7680]
    mojo_v = d["velocity"].reshape(N_IMG, -1)
    sigma = float(d["sigma"].ravel()[0])
    print(f"   sigma={sigma:.6f}")

    # position ids — match the fixed Mojo/diffusers convention:
    #   img: [0, row, col, 0] ;  txt: [0,0,0,k]
    img_ids = torch.zeros(N_IMG, 4)
    for idx in range(N_IMG):
        img_ids[idx, 1] = idx // IMG_W
        img_ids[idx, 2] = idx % IMG_W
    txt_ids = torch.zeros(N_TXT, 4)
    for k in range(N_TXT):
        txt_ids[k, 3] = k
    img_ids = img_ids.cuda()
    txt_ids = txt_ids.cuda()

    print("=== loading diffusers Flux2Transformer2DModel (explicit 4B config) ===")
    from safetensors.torch import load_file
    from diffusers import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import (
        convert_flux2_transformer_checkpoint_to_diffusers,
    )
    # 4B config derived from the checkpoint (heads=24→inner 3072, 5+20 blocks,
    # joint_dim=7680, no guidance). The diffusers default is a larger variant.
    cfg = dict(
        patch_size=1, in_channels=128, out_channels=128,
        num_layers=5, num_single_layers=20,
        attention_head_dim=128, num_attention_heads=24,
        joint_attention_dim=7680, timestep_guidance_channels=256,
        mlp_ratio=3.0, axes_dims_rope=(32, 32, 32, 32),
        rope_theta=2000, eps=1e-6, guidance_embeds=False,
    )
    raw = load_file(CKPT)
    diff_sd = convert_flux2_transformer_checkpoint_to_diffusers(raw)
    model = Flux2Transformer2DModel(**cfg).to(torch.bfloat16).cuda().eval()
    missing, unexpected = model.load_state_dict(diff_sd, strict=False)
    print(f"   load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        print("   missing[:5]:", missing[:5])
    if unexpected[:5]:
        print("   unexpected[:5]:", unexpected[:5])

    # hooks: x_embedder out (= Mojo img_in_act), norm_out input (= Mojo img_out)
    caught = {}
    model.x_embedder.register_forward_hook(
        lambda m, i, o: caught.__setitem__("img_in", o.detach())
    )
    model.norm_out.register_forward_hook(
        lambda m, i, o: caught.__setitem__("img_out", i[0].detach())
    )

    ts = torch.tensor([sigma], dtype=torch.float32).cuda()
    # guidance: try None first; Klein-base may be guidance-distilled.
    with torch.no_grad():
        out = model(
            hidden_states=x_t.to(torch.bfloat16),
            encoder_hidden_states=text.to(torch.bfloat16),
            timestep=ts,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=True,
        )
    diff_v = out.sample if hasattr(out, "sample") else out[0]
    diff_v = diff_v.float().squeeze(0).cpu().numpy().reshape(N_IMG, -1)

    print("=== per-stage parity: Mojo vs diffusers ===")
    di = caught["img_in"].float().squeeze(0).cpu().numpy().reshape(N_IMG, -1)
    do = caught["img_out"].float().squeeze(0).cpu().numpy().reshape(N_IMG, -1)
    stats("img_in_act (after input proj)", d["img_in_act"].reshape(N_IMG, -1), di)
    stats("img_out    (after 25 blocks) ", d["img_out"].reshape(N_IMG, -1), do)
    stats("velocity   (final)           ", mojo_v, diff_v)
    print("\nFirst stage where cos drops / std diverges localizes the bug:")
    print("  img_in bad -> input projection; img_out bad (img_in ok) -> the 25 blocks;")
    print("  velocity bad (img_out ok) -> final norm_out/proj_out (modulation).")


if __name__ == "__main__":
    main()
