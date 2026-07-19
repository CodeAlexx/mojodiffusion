#!/usr/bin/env python3
# gen_qwenimage_oracle.py — diffusers oracle for the Qwen-Image MMDiT forward.
#
# DEV-ONLY oracle (pure-Mojo runtime rule: Python never runs in the shipped
# path). Loads QwenImageTransformer2DModel with diffusers, runs ONE forward with
# FIXED-seed inputs, and dumps the transformer output plus per-block(0)
# intermediates for localization. Each dump is a flat float32 .bin with a
# sidecar .shape text file (mirrors the Z-Image / VAE oracle format).
#
# Run with the scratch venv:
#   /tmp/vae_oracle_venv/bin/python gen_qwenimage_oracle.py [HL WL TXTLEN]
#
# We run the model in float32 so the oracle is CPU/dtype-stable (PyTorch CPU vs
# CUDA BF16 diverge per-layer). The Mojo side runs BF16 storage / F32 accum;
# the parity threshold (cos>=0.99) accounts for that.
#
# Inputs to QwenImageTransformer2DModel.forward:
#   hidden_states:         [B, N_img, in_channels=64]   (already-patchified latent)
#   encoder_hidden_states: [B, N_txt, joint_attention_dim=3584]
#   timestep:              [B]                            (in [0,1] * 1000 inside)
#   img_shapes:            [(frame, h, w)]                (latent patch grid)
#   txt_seq_lens:          [N_txt]                        (RoPE text length)
# Output: [B, N_img, out_channels*?]  -> proj_out gives [B, N_img, 64].
#
# IMPORTANT: generate at the SAME (HL,WL,TXTLEN) you test in Mojo — do not reuse
# a different-size fixture.

import os
import sys
import numpy as np
import torch

from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# NOTE: the Qwen-Image-2512 snapshot on this box has only the index file (the 9
# weight shards were never fully downloaded). Qwen-Image-Edit-2511 has the
# SAME MMDiT architecture (60 double-stream blocks, dim=3072, heads=24,
# head_dim=128, axes_dims_rope=(16,56,56), identical weight keys) AND all 5
# shards present on disk, so we use it as the parity oracle. We disable its
# `zero_cond_t` flag so it runs the plain T2I double-stream forward that the
# Mojo port implements (zero_cond_t adds per-region timestep machinery that is
# an Edit-only path, not part of the base MMDiT being ported).
XFMR_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/"
    "snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/transformer"
)

# Latent patch grid (frame, h, w) and text token count.
FRAME = 1
HL = int(sys.argv[1]) if len(sys.argv) > 1 else 4
WL = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TXTLEN = int(sys.argv[3]) if len(sys.argv) > 3 else 16
OUT_DIR = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "qwenimage")
if not os.path.isabs(OUT_DIR):
    OUT_DIR = os.path.join(HERE, OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 1234


def dump(name, t):
    if isinstance(t, (list, tuple)):
        t = t[0]
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    arr.astype("<f4").tofile(os.path.join(OUT_DIR, name + ".bin"))
    with open(os.path.join(OUT_DIR, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  dumped {name:34s} shape={tuple(t.shape)} "
          f"mean={float(t.mean()):+.5f} std={float(t.std()):+.5f}")


def main():
    torch.manual_seed(SEED)
    model = QwenImageTransformer2DModel.from_pretrained(XFMR_DIR, torch_dtype=torch.float32)
    model.eval()
    # Force the plain T2I double-stream path (see XFMR_DIR note).
    if getattr(model, "zero_cond_t", False):
        model.zero_cond_t = False
        for blk in model.transformer_blocks:
            blk.zero_cond_t = False
    cfg = model.config
    in_ch = cfg.in_channels
    dim = cfg.num_attention_heads * cfg.attention_head_dim
    jad = cfg.joint_attention_dim
    print(f"num_layers={cfg.num_layers} heads={cfg.num_attention_heads} "
          f"head_dim={cfg.attention_head_dim} dim={dim} in_ch={in_ch} "
          f"joint_attention_dim={jad} axes_dims_rope={cfg.axes_dims_rope}")

    n_img = FRAME * HL * WL
    # Fixed random inputs.
    hidden = torch.randn(1, n_img, in_ch, dtype=torch.float32) * 0.5
    cap = torch.randn(1, TXTLEN, jad, dtype=torch.float32) * 0.5
    t = torch.tensor([0.7], dtype=torch.float32)

    dump("in_img", hidden)
    dump("in_cap", cap)
    dump("in_t", t)

    img_shapes = [(FRAME, HL, WL)]
    txt_seq_lens = [TXTLEN]

    captures = {}
    with torch.no_grad():
        # ── Reproduce the model.forward boundaries so we can localize. ──
        h = model.img_in(hidden)                            # [1, n_img, dim]
        captures["after_img_in"] = h.clone()
        ehs = model.txt_norm(cap)
        captures["after_txt_norm"] = ehs.clone()
        ehs = model.txt_in(ehs)                             # [1, txt, dim]
        captures["after_txt_in"] = ehs.clone()

        temb = model.time_text_embed(t, h)                  # [1, dim]
        captures["temb"] = temb.clone()

        rope = model.pos_embed(img_shapes, max_txt_seq_len=TXTLEN, device=h.device)
        img_freqs, txt_freqs = rope                          # complex [n_img, Dh/2], [txt, Dh/2]
        # Dump cos/sin (real/imag) for img + txt RoPE so Mojo can match exactly.
        captures["rope_img_cos"] = torch.view_as_real(img_freqs)[..., 0].clone()  # [n_img, Dh/2]
        captures["rope_img_sin"] = torch.view_as_real(img_freqs)[..., 1].clone()
        captures["rope_txt_cos"] = torch.view_as_real(txt_freqs)[..., 0].clone()  # [txt, Dh/2]
        captures["rope_txt_sin"] = torch.view_as_real(txt_freqs)[..., 1].clone()

        # Walk all blocks; capture after block 0 + after full stack.
        eh, hs = ehs, h
        for li, block in enumerate(model.transformer_blocks):
            eh, hs = block(
                hidden_states=hs,
                encoder_hidden_states=eh,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=rope,
            )
            if li == 0:
                captures["block0_img"] = hs.clone()         # [1, n_img, dim]
                captures["block0_txt"] = eh.clone()         # [1, txt, dim]
        captures["after_blocks_img"] = hs.clone()
        captures["after_blocks_txt"] = eh.clone()

        hs = model.norm_out(hs, temb)                       # AdaLayerNormContinuous
        captures["after_norm_out"] = hs.clone()
        out = model.proj_out(hs)                            # [1, n_img, 64]
        captures["out"] = out.clone()

        # Cross-check vs the public forward.
        full = model(
            hidden_states=hidden,
            encoder_hidden_states=cap,
            encoder_hidden_states_mask=None,
            timestep=t,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=True,
        ).sample
        d = float((full - out).abs().max())
        print(f"  manual-walk vs forward() max_abs_diff = {d:.3e}")

    for k, v in captures.items():
        dump(k, v)

    with open(os.path.join(OUT_DIR, "dims.txt"), "w") as f:
        f.write(f"FRAME={FRAME}\nHL={HL}\nWL={WL}\nTXTLEN={TXTLEN}\n")
        f.write(f"n_img={n_img}\ndim={dim}\nhead_dim={cfg.attention_head_dim}\n")
    print(f"  dims: n_img={n_img} txt={TXTLEN} dim={dim}")
    print("oracle dump complete ->", OUT_DIR)


if __name__ == "__main__":
    main()
