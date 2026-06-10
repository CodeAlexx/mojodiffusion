#!/usr/bin/env python3
# flux_dit_lora_oracle.py — BFL torch reference for the FLUX DiT forward WITH a
# Kohya-BFL LoRA additively applied. Verifies the Mojo runtime overlay
# (flux_lora_overlay.mojo) is numerically correct — not just "changes output".
#
# Reuses the EXACT Gate B inputs (reads flux_dit_img/txt/vec/tg.bin) and the same
# tiny 4x4 grid, so the only delta vs Gate B is the LoRA. Applies the SAME math
# the Mojo overlay applies: for each Kohya target, delta = (up @ down)*(alpha/r),
# added onto the BFL base weight. Dtype faithful to the Mojo: base fp16->bf16->fp32,
# up/down bf16->fp32 (the Mojo computes the delta in bf16); fp32 accumulation.
#
# Dumps: flux_dit_lora_pred.bin [1,16,64].
# Usage: python3 flux_dit_lora_oracle.py [lora_path]

import sys, os, re
import numpy as np
import torch

sys.path.insert(0, "/home/alex/black-forest-labs-flux/src")
from flux.model import Flux, FluxParams
from safetensors.torch import load_file
from safetensors import safe_open

DIT_PATH = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"
DEFAULT_LORA = "/home/alex/.serenity/models/loras/Fluxass (1).safetensors"
DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity"

PARAMS = FluxParams(
    in_channels=64, out_channels=64, vec_in_dim=768, context_in_dim=4096,
    hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19,
    depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10_000,
    qkv_bias=True, guidance_embed=True,
)

DOUBLE = [("img_attn_qkv", "img_attn.qkv"), ("img_attn_proj", "img_attn.proj"),
          ("img_mlp_0", "img_mlp.0"), ("img_mlp_2", "img_mlp.2"),
          ("img_mod_lin", "img_mod.lin"), ("txt_attn_qkv", "txt_attn.qkv"),
          ("txt_attn_proj", "txt_attn.proj"), ("txt_mlp_0", "txt_mlp.0"),
          ("txt_mlp_2", "txt_mlp.2"), ("txt_mod_lin", "txt_mod.lin")]
SINGLE = [("linear1", "linear1"), ("linear2", "linear2"),
          ("modulation_lin", "modulation.lin")]


def rd(name, shape):
    a = np.fromfile(f"{DIR}/{name}", dtype="<f4")
    return torch.from_numpy(a.reshape(shape).copy()).to(torch.float32)


def main():
    lora_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LORA
    H2 = W2 = 4
    N_IMG = 16
    N_TXT = 16

    sd = load_file(DIT_PATH)
    for k in list(sd.keys()):
        sd[k] = sd[k].to(torch.float32)
    with torch.device("meta"):
        model = Flux(PARAMS)
    model.load_state_dict(sd, strict=False, assign=True)
    model = model.eval()

    # Apply LoRA additively onto the fp32 params (bf16-faithful delta).
    lw = load_file(lora_path)
    names = set(lw.keys())
    sd_model = dict(model.named_parameters())
    n_applied = 0

    def apply(stem, bfl):
        nonlocal n_applied
        dk, uk = f"{stem}.lora_down.weight", f"{stem}.lora_up.weight"
        if dk not in names or uk not in names:
            return
        down = lw[dk].to(torch.bfloat16).to(torch.float32)   # [r,in]
        up = lw[uk].to(torch.bfloat16).to(torch.float32)      # [out,r]
        r = down.shape[0]
        scale = 1.0
        ak = f"{stem}.alpha"
        if ak in names:
            scale = float(lw[ak].to(torch.float32).item()) / r
        delta = (up @ down) * scale                           # [out,in]
        p = sd_model[bfl]
        if tuple(p.shape) != tuple(delta.shape):
            raise SystemExit(f"shape mismatch {bfl}: {tuple(p.shape)} vs {tuple(delta.shape)}")
        with torch.no_grad():
            p.add_(delta)
        n_applied += 1

    for bi in range(PARAMS.depth):
        for ks, bs in DOUBLE:
            apply(f"lora_unet_double_blocks_{bi}_{ks}", f"double_blocks.{bi}.{bs}.weight")
    for bi in range(PARAMS.depth_single_blocks):
        for ks, bs in SINGLE:
            apply(f"lora_unet_single_blocks_{bi}_{ks}", f"single_blocks.{bi}.{bs}.weight")
    print(f"[oracle] applied {n_applied} LoRA deltas from {lora_path}")

    img = rd("flux_dit_img.bin", (1, N_IMG, 64))
    txt = rd("flux_dit_txt.bin", (1, N_TXT, 4096))
    y = rd("flux_dit_vec.bin", (1, 768))
    tg = rd("flux_dit_tg.bin", (2,))
    t_raw, guid_raw = torch.tensor([tg[0]]), torch.tensor([tg[1]])

    img_ids = torch.zeros(1, N_IMG, 3, dtype=torch.float32)
    rows = torch.arange(H2).view(H2, 1).expand(H2, W2).reshape(-1)
    cols = torch.arange(W2).view(1, W2).expand(H2, W2).reshape(-1)
    img_ids[0, :, 1] = rows.float()
    img_ids[0, :, 2] = cols.float()
    txt_ids = torch.zeros(1, N_TXT, 3, dtype=torch.float32)

    with torch.no_grad():
        pred = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                     timesteps=t_raw, y=y, guidance=guid_raw)
    print(f"[oracle] LoRA pred {list(pred.shape)} mean={pred.mean().item():.6f} "
          f"std={pred.std().item():.6f}")
    pred.detach().to(torch.float32).contiguous().numpy().ravel().astype("<f4").tofile(
        f"{DIR}/flux_dit_lora_pred.bin")
    print("[oracle] dumped flux_dit_lora_pred.bin")


if __name__ == "__main__":
    main()
