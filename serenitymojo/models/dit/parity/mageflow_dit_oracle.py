#!/usr/bin/env python
# mageflow_dit_oracle.py — FULL MageFlow DiT forward parity oracle.
# Dev tool, NOT shipped.
#
# Loads the REAL Mage-Flow-Edit-Turbo transformer (12 double blocks) from the
# downloaded weights, feeds a fixed non-degenerate latent [1,128,4,4], a fixed
# Qwen3-VL context [1,8,2560] and a fixed timestep, and dumps the input +
# output velocity plus taps (post img_in, post txt_in, temb, post-block0) as a
# single f32 safetensors so the Mojo probe gates against byte-identical inputs.
#
# GPU bf16. flash-attn is absent in the current Python environment, so we force the
# functionally-equivalent SDPA varlen fallback (dense, non-causal, 1/sqrt(d)).
#
# Run:
#   pixi run python \
#     serenitymojo/models/dit/parity/mageflow_dit_oracle.py
import os

import torch
from einops import rearrange
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/transformer"
SFT = os.path.join(CKPT, "diffusion_pytorch_model.safetensors")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mageflow_dumps")
os.makedirs(OUT, exist_ok=True)

FRAME, H_TOK, W_TOK = 1, 4, 4
N_IMG = FRAME * H_TOK * W_TOK   # 16
N_TXT = 8
CIN = 128
CTX_DIM = 2560
HIDDEN = 3072
TIMESTEP = 0.75                 # must match the probe's comptime TIMESTEP

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
    print(f"[mf-dit-oracle] building MageFlow + loading weights from {SFT}")
    model = load_model(DIT_STRUCTURE, pretrain_path=SFT)
    model = model.to(dev, dtype=dt).eval()

    # ── deterministic non-degenerate inputs ──────────────────────────────────
    g = torch.Generator(device=dev).manual_seed(0)
    latent = (torch.randn(1, CIN, H_TOK, W_TOK, generator=g, device=dev) * 0.5).to(dt)
    img = rearrange(latent, "b c h w -> b (h w) c")           # [1,16,128] packed
    txt = (torch.randn(1, N_TXT, CTX_DIM, generator=g, device=dev) * 0.5).to(dt)  # RAW context
    timesteps = torch.tensor([TIMESTEP], device=dev, dtype=torch.float32)

    img_cu = torch.tensor([0, N_IMG], dtype=torch.int32, device=dev)
    txt_cu = torch.tensor([0, N_TXT], dtype=torch.int32, device=dev)
    img_shapes = [(FRAME, H_TOK, W_TOK)]

    dumps = {}

    def dump(name, t):
        dumps[name] = t.detach().float().cpu().contiguous()
        return list(t.shape)

    dump("in_img", img[0])   # [16,128]  patchified latent tokens
    dump("in_txt", txt[0])   # [8,2560]  raw Qwen3-VL context

    with torch.no_grad():
        # ── taps via the real submodules ──
        ms_pe = model.pos_embed(img_shapes, device=torch.device(dev))
        img_proj = model.img_in(img)                          # [1,16,3072]
        txt_normed = model.txt_norm(txt)
        txt_proj = model.txt_in(txt_normed)                   # [1,8,3072]
        temb = model.time_text_embed(timesteps.to(dt), img_proj)  # [1,3072]
        dump("tap_img_in", img_proj[0])   # [16,3072]
        dump("tap_txt_in", txt_proj[0])   # [8,3072]
        dump("tap_temb", temb)            # [1,3072]

        # ── block stack, tapping the accumulation trajectory at 0/5/11 ──
        img_i, txt_i = img_proj, txt_proj
        for idx, block in enumerate(model.transformer_blocks):
            txt_i, img_i = block(
                hidden_states=img_i,
                encoder_hidden_states=txt_i,
                temb=temb,
                image_rotary_emb=ms_pe,
                txt_cu_lens=txt_cu,
                img_cu_lens=img_cu,
            )
            if idx in (0, 5, 11):
                dump(f"tap_block{idx}_img", img_i[0])  # [16,3072]
                dump(f"tap_block{idx}_txt", txt_i[0])  # [8,3072]

        # ── full real forward: the parity reference velocity ──
        vel = model(
            img, txt, timesteps,
            img_shapes=img_shapes,
            img_cu_seqlens=img_cu,
            txt_cu_seqlens=txt_cu,
        )
    dump("velocity", vel[0])  # [16,128]

    out_path = os.path.join(OUT, "mageflow_dit.safetensors")
    save_file(dumps, out_path)
    print(f"[mf-dit-oracle] dumped {len(dumps)} tensors -> {out_path}")
    for k, v in dumps.items():
        print(f"    {k:16s} {tuple(v.shape)}  std={v.std().item():.5f}")

    del model
    torch.cuda.empty_cache()
    print("[mf-dit-oracle] done; GPU freed")


if __name__ == "__main__":
    main()
