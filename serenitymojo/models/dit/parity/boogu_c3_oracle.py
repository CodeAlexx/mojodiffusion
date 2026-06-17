#!/usr/bin/env python
# boogu_c3_oracle.py — C3 (single-stream block) parity oracle. Dev tool, NOT shipped.
#
# Loads the REAL transformer, runs single_stream_layers[0] (BooguImageTransformerBlock,
# modulation=True) on a deterministic synthetic (hidden_states, temb) + the REAL
# joint RoPE freqs_cis (rebuilt via the rope embedder, same cap_len/h_tok/w_tok as
# the C2 gate), all-True attention mask (full-valid b=1). Dumps inputs + output as
# raw F32 .bin so the Mojo C3 block gates against byte-identical inputs.
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c3_oracle.py
import os
os.environ.setdefault("device", "cuda:0")  # blocks pick the SDPA processor (flash_attn absent)

import numpy as np
import torch

TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boogu_dumps")
os.makedirs(OUT, exist_ok=True)

CAP_LEN, H_LAT, W_LAT, P = 16, 32, 32, 2
H_TOK, W_TOK = H_LAT // P, W_LAT // P     # 16,16
IMG_LEN = H_TOK * W_TOK                    # 256
SEQ = CAP_LEN + IMG_LEN                     # 272
HIDDEN = 3360
AXES_DIM, AXES_LENS, THETA = [40, 40, 40], [2048, 1664, 1664], 10000


def dump(name, t):
    v = t.detach().float().cpu().numpy().ravel().astype("<f4")
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(v.tobytes())
    return list(t.shape)


def main():
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    from boogu.models.transformers.rope import BooguImageDoubleStreamRotaryPosEmbed
    print(f"[c3-oracle] loading transformer (bf16) from {TF_DIR}")
    model = BooguImageTransformer2DModel.from_pretrained(
        TF_DIR, torch_dtype=torch.bfloat16
    ).to("cuda:0").eval()

    dev, dt = "cuda:0", torch.bfloat16
    # REAL joint freqs_cis for this T2I no-ref case (same as C2).
    freqs_cis = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(AXES_DIM, AXES_LENS, THETA)
    rope = BooguImageDoubleStreamRotaryPosEmbed(theta=THETA, axes_dim=AXES_DIM,
                                                axes_lens=AXES_LENS, patch_size=P)
    attn_mask_cap = torch.ones(1, CAP_LEN, dtype=torch.bool)
    out = rope.forward(freqs_cis, attn_mask_cap, [[0]], [IMG_LEN], [None], [(H_LAT, W_LAT)],
                       torch.device("cuda:0"))
    joint_freqs = out[3]            # [1,272,60] complex, the rotary_emb for the joint seq

    torch.manual_seed(3)
    hidden = torch.randn(1, SEQ, HIDDEN, device=dev).to(dt)
    temb = torch.randn(1, 1024, device=dev).to(dt)
    joint_mask = torch.ones(1, SEQ, dtype=torch.bool, device=dev)

    block = model.single_stream_layers[0]
    with torch.no_grad():
        y = block(hidden, joint_mask, joint_freqs, temb)   # [1,272,3360]

    shapes = {}
    shapes["c3_in_hidden.bin"] = dump("c3_in_hidden.bin", hidden)
    shapes["c3_in_temb.bin"] = dump("c3_in_temb.bin", temb)
    shapes["c3_out.bin"] = dump("c3_out.bin", y)
    with open(os.path.join(OUT, "c3_meta.txt"), "w") as f:
        f.write(f"cap_len={CAP_LEN} h_tok={H_TOK} w_tok={W_TOK} seq={SEQ} hidden={HIDDEN}\n")
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
        f.write(f"in.std={hidden.float().std():.5f} out.std={y.float().std():.5f}\n")
    print("[c3-oracle] dumped:", shapes)
    print(f"[c3-oracle] hidden.std={hidden.float().std():.5f} out.std={y.float().std():.5f}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
