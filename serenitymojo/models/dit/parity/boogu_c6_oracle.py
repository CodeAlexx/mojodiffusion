#!/usr/bin/env python
# boogu_c6_oracle.py — C6 (full DiT forward) integration parity oracle. Dev tool.
#
# Runs the REAL BooguImageTransformer2DModel.forward end-to-end (T2I no-ref,
# batch=1) on deterministic synthetic inputs and dumps inputs + the velocity
# output. This gates the full wiring: patchify -> embed -> context/noise refiners
# -> 8 double-stream -> fuse -> 32 single-stream -> norm_out -> unpatchify.
#
# forward(hidden_states[1,16,32,32], timestep[1], instruction_hidden_states[1,16,4096],
#         freqs_cis(get_freqs_cis), instruction_attention_mask[1,16], ref=None) -> velocity[1,16,32,32]
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c6_oracle.py
import os
os.environ.setdefault("device", "cuda:0")

import numpy as np
import torch

TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boogu_dumps")
os.makedirs(OUT, exist_ok=True)

CAP_LEN, H_LAT, W_LAT = 16, 32, 32         # latent 16x32x32 -> 16x16 token grid -> 256 img tokens
INCH = 16
AXES_DIM, AXES_LENS, THETA = [40, 40, 40], [2048, 1664, 1664], 10000


def dump(name, t):
    v = t.detach().float().cpu().numpy().ravel().astype("<f4")
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(v.tobytes())
    return list(t.shape)


def main():
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    from boogu.models.transformers.rope import BooguImageDoubleStreamRotaryPosEmbed
    print(f"[c6-oracle] loading transformer (bf16) from {TF_DIR}")
    model = BooguImageTransformer2DModel.from_pretrained(
        TF_DIR, torch_dtype=torch.bfloat16
    ).to("cuda:0").eval()
    dev, dt = "cuda:0", torch.bfloat16

    freqs_cis = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(AXES_DIM, AXES_LENS, THETA)

    torch.manual_seed(6)
    latent = torch.randn(1, INCH, H_LAT, W_LAT, device=dev).to(dt)
    timestep = torch.tensor([0.25], device=dev, dtype=torch.float32)
    instr = torch.randn(1, CAP_LEN, 4096, device=dev).to(dt)
    instr_mask = torch.ones(1, CAP_LEN, dtype=torch.bool, device=dev)

    with torch.no_grad():
        vel = model.forward(
            latent, timestep, instr, freqs_cis, instr_mask,
            ref_image_hidden_states=None, return_dict=False,
        )
    if isinstance(vel, (list, tuple)):
        vel = vel[0] if not torch.is_tensor(vel) else vel
    vel = torch.as_tensor(vel) if not torch.is_tensor(vel) else vel
    print("[c6-oracle] velocity shape:", tuple(vel.shape), vel.dtype)

    shapes = {}
    shapes["c6_in_latent.bin"] = dump("c6_in_latent.bin", latent)
    shapes["c6_in_timestep.bin"] = dump("c6_in_timestep.bin", timestep)
    shapes["c6_in_instr.bin"] = dump("c6_in_instr.bin", instr)
    shapes["c6_out_velocity.bin"] = dump("c6_out_velocity.bin", vel)
    with open(os.path.join(OUT, "c6_meta.txt"), "w") as f:
        f.write(f"cap_len={CAP_LEN} H_lat={H_LAT} W_lat={W_LAT} in_ch={INCH} "
                f"h_tok={H_LAT//2} w_tok={W_LAT//2} img_tokens={(H_LAT//2)*(W_LAT//2)}\n")
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
        f.write(f"vel.std={vel.float().std():.5f} vel.shape={list(vel.shape)}\n")
    print("[c6-oracle] dumped:", shapes)
    print(f"[c6-oracle] velocity std={vel.float().std():.5f}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
