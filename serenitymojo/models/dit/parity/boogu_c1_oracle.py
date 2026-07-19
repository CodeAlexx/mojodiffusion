#!/usr/bin/env python
# boogu_c1_oracle.py — C1 (embedders) parity oracle. Dev tool, NOT shipped.
#
# Loads ONLY the Boogu transformer (the mllm/encoder is not needed for C1 and
# may still be downloading) on GPU bf16, builds DETERMINISTIC inputs, runs the
# real x_embedder + time_caption_embed submodules, and dumps inputs + outputs as
# raw little-endian F32 .bin (NCHW/row-major) — same convention as
# flux_vae_decode_oracle.py so the Mojo gate reads them via io/ffi.
#
# Inputs are cast to bf16 (the model dtype) BEFORE dumping, so the dumped F32 is
# the exact value the model saw and the Mojo side can re-cast idempotently.
#
# Run (separate command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/dit/parity/boogu_c1_oracle.py
import os
os.environ.setdefault("device", "cuda:0")

import numpy as np
import torch

TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boogu_dumps")
os.makedirs(OUT, exist_ok=True)


def dump(name, t):
    v = t.detach().float().cpu().numpy().ravel()
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(v.astype("<f4").tobytes())
    return list(t.shape)


def main():
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    print(f"[c1-oracle] loading transformer (bf16) from {TF_DIR}")
    model = BooguImageTransformer2DModel.from_pretrained(
        TF_DIR, torch_dtype=torch.bfloat16
    ).to("cuda:0").eval()

    dev, dt = "cuda:0", torch.bfloat16
    torch.manual_seed(0)
    # deterministic inputs, cast to model dtype then back so dumped F32 == model input
    timestep = torch.tensor([0.25], device=dev, dtype=torch.float32)
    instr = torch.randn(1, 16, 4096, device=dev).to(dt)          # instruction feats
    L_img = 256                                                   # 32x32 patch grid (a 64x64 latent / p=2)
    tokens = torch.randn(1, L_img, 2 * 2 * 16, device=dev).to(dt)  # patchified [B,L,64]

    with torch.no_grad():
        temb, caption = model.time_caption_embed(timestep, instr, dt)
        xembed = model.x_embedder(tokens)

    shapes = {}
    shapes["c1_in_timestep.bin"] = dump("c1_in_timestep.bin", timestep)
    shapes["c1_in_instr.bin"] = dump("c1_in_instr.bin", instr)
    shapes["c1_in_tokens.bin"] = dump("c1_in_tokens.bin", tokens)
    shapes["c1_out_temb.bin"] = dump("c1_out_temb.bin", temb)
    shapes["c1_out_caption.bin"] = dump("c1_out_caption.bin", caption)
    shapes["c1_out_xembed.bin"] = dump("c1_out_xembed.bin", xembed)

    with open(os.path.join(OUT, "c1_meta.txt"), "w") as f:
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
        f.write(f"temb std={temb.float().std().item():.6f} "
                f"caption std={caption.float().std().item():.6f} "
                f"xembed std={xembed.float().std().item():.6f}\n")
    print("[c1-oracle] dumped:")
    for k, v in shapes.items():
        print(f"  {k:24s} {v}")
    print(f"[c1-oracle] temb.std={temb.float().std():.5f} "
          f"caption.std={caption.float().std():.5f} xembed.std={xembed.float().std():.5f}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
