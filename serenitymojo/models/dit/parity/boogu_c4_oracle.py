#!/usr/bin/env python
# boogu_c4_oracle.py — C4 (double-stream block) parity oracle. Dev tool, NOT shipped.
#
# Loads the REAL transformer, runs double_stream_layers[0]
# (BooguImageDoubleStreamTransformerBlock, modulation=True) on deterministic
# synthetic (img_hidden, instruct_hidden, temb) + the REAL joint & combined-img
# RoPE freqs_cis (rebuilt via the rope embedder, T2I no-ref batch=1: cap_len=16,
# img 256 = 16x16 token grid), all-True masks. Dumps inputs + the two outputs
# (img_out, instruct_out) as raw F32 .bin.
#
# Block.forward(img_hidden, instruct_hidden, img_attention_mask, joint_attention_mask,
#               image_rotary_emb(=combined_img,256), rotary_emb(=joint,272), temb,
#               encoder_seq_lengths=[16], seq_lengths=[272]) -> (img_out, instruct_out)
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c4_oracle.py
import os
os.environ.setdefault("device", "cuda:0")

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
    print(f"[c4-oracle] loading transformer (bf16) from {TF_DIR}")
    model = BooguImageTransformer2DModel.from_pretrained(
        TF_DIR, torch_dtype=torch.bfloat16
    ).to("cuda:0").eval()
    dev, dt = "cuda:0", torch.bfloat16

    freqs_cis = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(AXES_DIM, AXES_LENS, THETA)
    rope = BooguImageDoubleStreamRotaryPosEmbed(theta=THETA, axes_dim=AXES_DIM,
                                                axes_lens=AXES_LENS, patch_size=P)
    rout = rope.forward(freqs_cis, torch.ones(1, CAP_LEN, dtype=torch.bool),
                        [[0]], [IMG_LEN], [None], [(H_LAT, W_LAT)], torch.device("cuda:0"))
    joint_freqs = rout[3]              # [1,272,60] complex  -> rotary_emb (joint attn)
    combined_img_freqs = rout[6]       # [1,256,60] complex  -> image_rotary_emb (img self-attn)

    torch.manual_seed(4)
    img_hidden = torch.randn(1, IMG_LEN, HIDDEN, device=dev).to(dt)
    instruct_hidden = torch.randn(1, CAP_LEN, HIDDEN, device=dev).to(dt)
    temb = torch.randn(1, 1024, device=dev).to(dt)
    img_mask = torch.ones(1, IMG_LEN, dtype=torch.bool, device=dev)
    joint_mask = torch.ones(1, SEQ, dtype=torch.bool, device=dev)

    block = model.double_stream_layers[0]
    with torch.no_grad():
        img_out, instruct_out = block(
            img_hidden, instruct_hidden, img_mask, joint_mask,
            combined_img_freqs, joint_freqs, temb, [CAP_LEN], [SEQ],
        )

    shapes = {}
    shapes["c4_in_img.bin"] = dump("c4_in_img.bin", img_hidden)
    shapes["c4_in_instruct.bin"] = dump("c4_in_instruct.bin", instruct_hidden)
    shapes["c4_in_temb.bin"] = dump("c4_in_temb.bin", temb)
    shapes["c4_out_img.bin"] = dump("c4_out_img.bin", img_out)
    shapes["c4_out_instruct.bin"] = dump("c4_out_instruct.bin", instruct_out)
    with open(os.path.join(OUT, "c4_meta.txt"), "w") as f:
        f.write(f"cap_len={CAP_LEN} img_len={IMG_LEN} seq={SEQ} hidden={HIDDEN} "
                f"h_tok={H_TOK} w_tok={W_TOK}\n")
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
        f.write(f"img_out.std={img_out.float().std():.5f} "
                f"instruct_out.std={instruct_out.float().std():.5f}\n")
    print("[c4-oracle] dumped:", shapes)
    print(f"[c4-oracle] img_out.std={img_out.float().std():.5f} "
          f"instruct_out.std={instruct_out.float().std():.5f}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
