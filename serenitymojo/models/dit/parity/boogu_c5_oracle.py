#!/usr/bin/env python
# boogu_c5_oracle.py — C5 (output norm / norm_out) parity oracle. Dev tool, NOT shipped.
#
# Runs the REAL model.norm_out (LuminaLayerNormContinuous) on deterministic
# synthetic (hidden[1,272,3360], temb[1,1024]) and dumps inputs+output as raw F32.
# norm_out: scale = linear_1(silu(temb)); x = LayerNorm(x, elementwise_affine=False,
# eps=1e-6) * (1+scale)[:,None,:]; out = linear_2(x) -> [1,272, patch*patch*out_ch=64].
# (Unpatchify is a deterministic rearrange of the IMAGE token rows [16:272] -> [16,32,32];
#  the numeric gate target is norm_out's [1,272,64] output.)
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c5_oracle.py
import os
os.environ.setdefault("device", "cuda:0")

import numpy as np
import torch

TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boogu_dumps")
os.makedirs(OUT, exist_ok=True)

SEQ, HIDDEN, OUTDIM = 272, 3360, 64


def dump(name, t):
    v = t.detach().float().cpu().numpy().ravel().astype("<f4")
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(v.tobytes())
    return list(t.shape)


def main():
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    print(f"[c5-oracle] loading transformer (bf16) from {TF_DIR}")
    model = BooguImageTransformer2DModel.from_pretrained(
        TF_DIR, torch_dtype=torch.bfloat16
    ).to("cuda:0").eval()
    dev, dt = "cuda:0", torch.bfloat16

    torch.manual_seed(5)
    hidden = torch.randn(1, SEQ, HIDDEN, device=dev).to(dt)
    temb = torch.randn(1, 1024, device=dev).to(dt)
    with torch.no_grad():
        y = model.norm_out(hidden, temb)         # [1,272,64]

    shapes = {}
    shapes["c5_in_hidden.bin"] = dump("c5_in_hidden.bin", hidden)
    shapes["c5_in_temb.bin"] = dump("c5_in_temb.bin", temb)
    shapes["c5_out.bin"] = dump("c5_out.bin", y)
    with open(os.path.join(OUT, "c5_meta.txt"), "w") as f:
        f.write(f"seq={SEQ} hidden={HIDDEN} out_dim={OUTDIM}\n")
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
        f.write(f"out.std={y.float().std():.5f}\n")
    print("[c5-oracle] dumped:", shapes)
    print(f"[c5-oracle] out shape {list(y.shape)} std={y.float().std():.5f}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
