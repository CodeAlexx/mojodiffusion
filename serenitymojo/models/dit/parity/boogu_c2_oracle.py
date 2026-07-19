#!/usr/bin/env python
# boogu_c2_oracle.py — C2 (3-axis RoPE embedder) parity oracle. Dev tool, NOT shipped.
#
# Runs the REAL BooguImageDoubleStreamRotaryPosEmbed standalone (no big model
# load) for a T2I no-ref, batch=1 case and dumps the produced freqs_cis as
# real/imag raw F32 .bin (the freqs_cis is complex e^{iθ}; real=cos θ, imag=sin θ).
# The Mojo C2 builds cos/sin tables via ops/rope_tables.build_multiaxis_rope_tables
# and we gate cos↔real, sin↔imag.
#
# T2I case: cap_len=16, latent 32x32 (img tokens = (32/2)*(32/2) = 256), no ref.
# joint seq = 16 + 256 = 272.  axes_dim=[40,40,40], axes_lens=[2048,1664,1664], theta=10000, p=2.
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c2_oracle.py
import os
os.environ.setdefault("device", "cpu")  # rope embedder is tiny; CPU is fine

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boogu_dumps")
os.makedirs(OUT, exist_ok=True)

CAP_LEN = 16
H_LAT, W_LAT = 32, 32
P = 2
H_TOK, W_TOK = H_LAT // P, W_LAT // P          # 16, 16
IMG_LEN = H_TOK * W_TOK                          # 256
AXES_DIM = [40, 40, 40]
AXES_LENS = [2048, 1664, 1664]
THETA = 10000


def dump(name, t):
    v = np.asarray(t).ravel().astype("<f4")
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(v.tobytes())
    return list(np.asarray(t).shape)


def main():
    from boogu.models.transformers.rope import BooguImageDoubleStreamRotaryPosEmbed
    rope = BooguImageDoubleStreamRotaryPosEmbed(
        theta=THETA, axes_dim=AXES_DIM, axes_lens=AXES_LENS, patch_size=P
    )
    freqs_cis = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(AXES_DIM, AXES_LENS, THETA)
    device = torch.device("cpu")
    attn_mask = torch.ones(1, CAP_LEN, dtype=torch.bool)
    l_eff_ref = [[0]]
    l_eff_img = [IMG_LEN]
    ref_sizes = [None]
    img_sizes = [(H_LAT, W_LAT)]

    out = rope.forward(freqs_cis, attn_mask, l_eff_ref, l_eff_img, ref_sizes, img_sizes, device)
    (cap_f, ref_f, img_f, joint_f, cap_lens, seq_lens, comb_f, comb_lens) = out
    # joint_f: [1, seq, 60] complex
    print("[c2-oracle] joint freqs_cis:", tuple(joint_f.shape), joint_f.dtype,
          "| cap_lens", cap_lens, "seq_lens", seq_lens, "comb_lens", comb_lens)

    jf = joint_f[0]                              # [seq, 60] complex
    shapes = {}
    shapes["c2_joint_real.bin"] = dump("c2_joint_real.bin", jf.real)
    shapes["c2_joint_imag.bin"] = dump("c2_joint_imag.bin", jf.imag)
    shapes["c2_img_real.bin"] = dump("c2_img_real.bin", img_f[0].real)
    shapes["c2_img_imag.bin"] = dump("c2_img_imag.bin", img_f[0].imag)
    shapes["c2_cap_real.bin"] = dump("c2_cap_real.bin", cap_f[0].real)
    shapes["c2_cap_imag.bin"] = dump("c2_cap_imag.bin", cap_f[0].imag)

    with open(os.path.join(OUT, "c2_meta.txt"), "w") as f:
        f.write(f"cap_len={CAP_LEN} img_len={IMG_LEN} seq_len={CAP_LEN+IMG_LEN} "
                f"H_tok={H_TOK} W_tok={W_TOK} half=60 axes_dim={AXES_DIM} theta={THETA}\n")
        f.write(f"cap_lens={cap_lens} seq_lens={seq_lens} comb_lens={comb_lens}\n")
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
    print("[c2-oracle] dumped:", {k: v for k, v in shapes.items()})
    # sanity: |freqs_cis| should be 1 (unit complex)
    mag = jf.abs()
    print(f"[c2-oracle] |joint| min={mag.min():.6f} max={mag.max():.6f} (should be ~1.0)")


if __name__ == "__main__":
    main()
