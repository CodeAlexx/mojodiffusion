#!/usr/bin/env python3
# gen_oracle.py — diffusers oracle for the Z-Image NextDiT transformer forward.
#
# DEV-ONLY oracle (pure-Mojo runtime rule: Python never runs in the shipped
# path). Loads ZImageTransformer2DModel with diffusers, runs ONE forward with
# FIXED-seed inputs, and dumps the transformer output plus a handful of named
# intermediates for per-block localization. Each dump is a flat float32 .bin
# with a sidecar .shape text file (mirrors the VAE oracle format).
#
# Run with the scratch venv (extend it if diffusers/torch missing):
#   /tmp/vae_oracle_venv/bin/python gen_oracle.py [HL WL CAPLEN]
#
# We run the model in float32 (NOT bf16) so the oracle is CPU/dtype-stable
# (see memory: PyTorch CPU vs CUDA BF16 diverge per-layer). The Mojo side runs
# BF16 storage / F32 accumulation; parity threshold accounts for that.
#
# The forward signature in diffusers is list-based:
#   x: list[Tensor (C,F,H,W)]   — here F=1 (image), so each item is (16,1,Hl,Wl)
#   t: Tensor [bsz]             — timestep in [0,1]
#   cap_feats: list[Tensor (cap_len, cap_feat_dim)]
# Basic (non-omni) mode. Output: list of (out_channels, F, H, W) -> we take [0].

import os
import sys
import struct
import numpy as np
import torch

from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel,
    ZImageTransformerBlock,
    RopeEmbedder,
)

HERE = os.path.dirname(os.path.abspath(__file__))
XFMR_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)

HL = int(sys.argv[1]) if len(sys.argv) > 1 else 8
WL = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CAPLEN = int(sys.argv[3]) if len(sys.argv) > 3 else 32  # multiple of 32 -> no pad
# Optional 4th arg: output dir (relative to HERE or absolute). Default = HERE
# (the 8x8 set). Pass e.g. "parity64" to keep the larger set side-by-side.
OUT_DIR = sys.argv[4] if len(sys.argv) > 4 else HERE
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
    print(f"  dumped {name:30s} shape={tuple(t.shape)} "
          f"mean={float(t.mean()):+.5f} std={float(t.std()):+.5f}")


def main():
    torch.manual_seed(SEED)
    model = ZImageTransformer2DModel.from_pretrained(XFMR_DIR, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    print(f"dim={cfg.dim} n_heads={cfg.n_heads} n_layers={cfg.n_layers} "
          f"n_refiner={cfg.n_refiner_layers} cap_feat_dim={cfg.cap_feat_dim} "
          f"rope_theta={cfg.rope_theta} t_scale={cfg.t_scale} "
          f"axes_dims={cfg.axes_dims} axes_lens={cfg.axes_lens} eps={cfg.norm_eps}")
    in_ch = cfg.in_channels
    dim = cfg.dim
    cap_feat_dim = cfg.cap_feat_dim

    # Fixed inputs (image latent, timestep, caption features).
    img = torch.randn(in_ch, 1, HL, WL, dtype=torch.float32)          # (C,F,H,W)
    cap = torch.randn(CAPLEN, cap_feat_dim, dtype=torch.float32)       # (cap_len, D)
    t = torch.tensor([0.7], dtype=torch.float32)                       # timestep in [0,1]

    dump("in_img", img)
    dump("in_cap", cap)
    dump("in_t", t)

    # ── Capture intermediates via the SAME boundaries the Mojo forward uses. ──
    captures = {}

    # adaln_input = t_embedder(t * t_scale)
    with torch.no_grad():
        adaln_input = model.t_embedder(t * cfg.t_scale)
    captures["t_emb"] = adaln_input.clone()

    patch_size, f_patch_size = 2, 1

    with torch.no_grad():
        # Patchify + embed (single image).
        (x_p, cap_p, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask) = \
            model.patchify_and_embed([img], [cap], patch_size, f_patch_size)
        x_seqlens = [len(xi) for xi in x_p]
        x = model.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x_p, dim=0))
        captures["x_after_embedder"] = x.clone()  # [img_padded_len, dim]

        x, x_freqs, x_mask, _, x_noise_tensor = model._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask,
            model.x_pad_token, None, img.device,
        )
        captures["x_after_prepare"] = x.clone()   # [1, img_padded_len, dim]

        # Noise refiner. Also dump the layer-0 modulation chunks for localization.
        nr0 = model.noise_refiner[0]
        mod0 = nr0.adaLN_modulation(adaln_input)
        sm, gm, smlp, gmlp = mod0.unsqueeze(1).chunk(4, dim=2)
        captures["nr0_scale_msa"] = (1.0 + sm).clone()
        captures["nr0_gate_msa"] = gm.tanh().clone()
        captures["nr0_scale_mlp"] = (1.0 + smlp).clone()
        captures["nr0_gate_mlp"] = gmlp.tanh().clone()
        # Sub-step localization of noise_refiner.0 (modulated attention branch).
        scale_msa = 1.0 + sm
        nr0_n1 = nr0.attention_norm1(x)                 # RMSNorm(x)
        captures["nr0_norm1"] = nr0_n1.clone()
        nr0_n1s = nr0_n1 * scale_msa                    # * scale_msa
        captures["nr0_norm1_scaled"] = nr0_n1s.clone()
        nr0_attn = nr0.attention(nr0_n1s, attention_mask=x_mask, freqs_cis=x_freqs)
        captures["nr0_attn_out"] = nr0_attn.clone()
        img_tokens = (HL // 2) * (WL // 2)
        for li, layer in enumerate(model.noise_refiner):
            x = layer(x, x_mask, x_freqs, adaln_input, None, None, None)
            captures[f"x_after_noise_refiner_{li}"] = x.clone()
            if li == 0:
                # only the real image tokens (first img_tokens rows). Mirrors the
                # Mojo stage-12 slice. x is [1, img_padded, dim].
                captures["x_after_noise_refiner_0_real"] = \
                    x[:, :img_tokens, :].clone()

        # Cap embed + refine.
        cap_seqlens = [len(ci) for ci in cap_p]
        capf = model.cap_embedder(torch.cat(cap_p, dim=0))
        captures["cap_after_embedder"] = capf.clone()
        capf, cap_freqs, cap_mask, _, _ = model._prepare_sequence(
            list(capf.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask,
            model.cap_pad_token, None, img.device,
        )
        captures["cap_after_prepare"] = capf.clone()
        for li, layer in enumerate(model.context_refiner):
            capf = layer(capf, cap_mask, cap_freqs)
            captures[f"cap_after_context_refiner_{li}"] = capf.clone()

        # Unified [x, cap] (basic mode order).
        unified, unified_freqs, unified_mask, unified_noise_tensor = \
            model._build_unified_sequence(
                x, x_freqs, x_seqlens, None,
                capf, cap_freqs, cap_seqlens, None,
                None, None, None, None,
                False, img.device,
            )
        captures["unified_initial"] = unified.clone()

        # Dump unified RoPE freqs_cis as real/imag so the Mojo side can build
        # the identical interleaved cos/sin tables. unified_freqs: [1, S, Dh/2]
        # complex -> we store cos (real) and sin (imag), each [1, S, Dh/2].
        uf = unified_freqs  # complex64 [1, S, 64]
        captures["unified_rope_cos"] = torch.view_as_real(uf)[..., 0].clone()
        captures["unified_rope_sin"] = torch.view_as_real(uf)[..., 1].clone()
        # head-0 RoPE cos table. Z-Image RoPE freqs are shared across heads, so
        # head 0 == the full per-position table. Mirrors the Mojo stage-13 slice
        # (reshape [S,H,half] -> head 0). Stored as [1, S, half] to match.
        captures["unified_rope_cos_h0"] = torch.view_as_real(uf)[..., 0].clone()
        # Also dump position ids for x and cap (int -> float) for diagnostics.
        captures["x_pos_ids"] = torch.cat(x_pos_ids, dim=0).float().clone()
        captures["cap_pos_ids"] = torch.cat(cap_pos_ids, dim=0).float().clone()

        # Main layers (capture block 0 + after full stack).
        for li, layer in enumerate(model.layers):
            unified = layer(unified, unified_mask, unified_freqs, adaln_input,
                            unified_noise_tensor, None, None)
            if li == 0:
                captures["unified_after_layer_0"] = unified.clone()
        captures["unified_after_main"] = unified.clone()

        # Final layer.
        final = model.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
        captures["after_final_layer"] = final.clone()

        # Unpatchify -> (C, F, H, W); take [0].
        out = model.unpatchify(list(final.unbind(dim=0)), x_size, patch_size, f_patch_size, None)
        out0 = out[0]                          # (C, F, H, W)
        captures["out"] = out0.clone()

        # Cross-check vs the public forward.
        full = model(x=[img], t=t, cap_feats=[cap], return_dict=True).sample[0]
        d = float((full - out0).abs().max())
        print(f"  manual-walk vs forward() max_abs_diff = {d:.3e}")

    for k, v in captures.items():
        dump(k, v)

    # Record key dims for the Mojo side.
    img_tokens = (HL // 2) * (WL // 2)
    img_pad = ((-img_tokens) % 32)
    img_padded = img_tokens + img_pad
    cap_pad = ((-CAPLEN) % 32)
    cap_padded = CAPLEN + cap_pad
    with open(os.path.join(OUT_DIR, "dims.txt"), "w") as f:
        f.write(f"HL={HL}\nWL={WL}\nCAPLEN={CAPLEN}\n")
        f.write(f"img_tokens={img_tokens}\nimg_padded={img_padded}\n")
        f.write(f"cap_padded={cap_padded}\nunified={img_padded + cap_padded}\n")
    print(f"  dims: img_tokens={img_tokens} img_padded={img_padded} "
          f"cap_padded={cap_padded} unified={img_padded + cap_padded}")
    print("oracle dump complete ->", OUT_DIR)


if __name__ == "__main__":
    main()
