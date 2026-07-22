#!/usr/bin/env python
# mageflow_block_oracle.py — MageFlow DiT double-stream block parity oracle.
# Dev tool, NOT shipped.
#
# Instantiates ONE real MageFlow double block (transformer_blocks[0]) from the
# downloaded Mage-Flow-Edit-Turbo transformer weights, feeds deterministic
# non-degenerate img/txt/temb + the real msrope tables (image-only; text tokens
# NOT roped, apply_text_rotary_emb=false), runs the block, and dumps
# inputs + a few taps + the (img, txt) block outputs as a single f32
# safetensors so the Mojo probe gates against byte-identical inputs.
#
# GPU bf16. flash-attn is absent in the current Python environment, so we force the
# functionally-equivalent SDPA varlen fallback (softmax_scale=None => 1/sqrt(d),
# dense/non-causal — exactly what MageFlow uses).
#
# Run:
#   pixi run python \
#     serenitymojo/models/dit/parity/mageflow_block_oracle.py
import os
import numpy as np
import torch
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/transformer"
SFT = os.path.join(CKPT, "diffusion_pytorch_model.safetensors")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mageflow_dumps")
os.makedirs(OUT, exist_ok=True)

# Small non-degenerate grid: frame=1, h=4, w=4 -> 16 image tokens; 8 text tokens.
FRAME, H_TOK, W_TOK = 1, 4, 4
N_IMG = FRAME * H_TOK * W_TOK   # 16
N_TXT = 8
HIDDEN = 3072

# MageFlowParams from config.json (Mage-Flow-Edit-Turbo).
DIT_STRUCTURE = dict(
    in_channels=128,
    out_channels=128,
    context_in_dim=2560,
    hidden_size=3072,
    num_heads=24,
    depth=12,
    axes_dim=[16, 56, 56],
    checkpoint=False,
    patch_size=1,
)


def main():
    from mage_flow.models.modules._attn_backend import set_attn_backend
    from mage_flow.models.utils import load_model

    set_attn_backend("sdpa")  # flash-attn absent; SDPA varlen fallback (equivalent)

    dev, dt = "cuda:0", torch.bfloat16
    print(f"[mf-oracle] building MageFlow + loading weights from {SFT}")
    model = load_model(DIT_STRUCTURE, pretrain_path=SFT)
    model = model.to(dev, dtype=dt).eval()

    block = model.transformer_blocks[0]
    pos_embed = model.pos_embed  # MageFlowEmbedRope(theta=10000, axes=[16,56,56], scale_rope=True)

    # ── real image msrope (text tokens are NOT rotated) ──────────────────────
    ms_pe = pos_embed([(FRAME, H_TOK, W_TOK)], device=torch.device(dev))  # [16,64] complex
    print(f"[mf-oracle] ms_pe: shape={tuple(ms_pe.shape)} dtype={ms_pe.dtype}")

    # ── deterministic non-degenerate inputs ─────────────────────────────────
    g = torch.Generator(device=dev).manual_seed(0)
    img = (torch.randn(1, N_IMG, HIDDEN, generator=g, device=dev) * 0.5).to(dt)
    txt = (torch.randn(1, N_TXT, HIDDEN, generator=g, device=dev) * 0.5).to(dt)
    temb = (torch.randn(1, HIDDEN, generator=g, device=dev) * 0.5).to(dt)

    img_cu = torch.tensor([0, N_IMG], dtype=torch.int32, device=dev)
    txt_cu = torch.tensor([0, N_TXT], dtype=torch.int32, device=dev)

    dumps = {}

    def dump(name, t):
        dumps[name] = t.detach().float().cpu().contiguous()
        return list(t.shape)

    dump("mf_in_img", img[0])       # [16,3072]
    dump("mf_in_txt", txt[0])       # [8,3072]
    dump("mf_in_temb", temb)        # [1,3072]

    # ── taps computed with the REAL submodules (faithful internal steps) ─────
    with torch.no_grad():
        img_mod_params = block.img_mod(temb)      # [1,6*dim]
        txt_mod_params = block.txt_mod(temb)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
        img_normed = block.img_norm1(img)
        img_modulated, _ = block._modulate(img_normed, img_mod1, cu_lens=img_cu)
        txt_normed = block.txt_norm1(txt)
        txt_modulated, _ = block._modulate(txt_normed, txt_mod1, cu_lens=txt_cu)
        dump("tap_img_modulated", img_modulated[0])   # [16,3072]
        dump("tap_txt_modulated", txt_modulated[0])   # [8,3072]

        img_attn_out, txt_attn_out = block.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=ms_pe,
            txt_cu_lens=txt_cu,
            img_cu_lens=img_cu,
        )
        dump("tap_img_attn", img_attn_out.reshape(-1, HIDDEN))  # [16,3072]
        dump("tap_txt_attn", txt_attn_out.reshape(-1, HIDDEN))  # [8,3072]

        # ── full real block forward (the parity reference) ──
        txt_out, img_out = block(
            hidden_states=img,
            encoder_hidden_states=txt,
            temb=temb,
            image_rotary_emb=ms_pe,
            txt_cu_lens=txt_cu,
            img_cu_lens=img_cu,
        )
    dump("mf_out_img", img_out[0])  # [16,3072]
    dump("mf_out_txt", txt_out[0])  # [8,3072]

    out_path = os.path.join(OUT, "mageflow_block0.safetensors")
    save_file(dumps, out_path)
    print(f"[mf-oracle] dumped {len(dumps)} tensors -> {out_path}")
    for k, v in dumps.items():
        print(f"    {k:20s} {tuple(v.shape)}  std={v.std().item():.5f}")

    # free GPU promptly (shared 16GB card)
    del model, block, pos_embed
    torch.cuda.empty_cache()
    print("[mf-oracle] done; GPU freed")


if __name__ == "__main__":
    main()
