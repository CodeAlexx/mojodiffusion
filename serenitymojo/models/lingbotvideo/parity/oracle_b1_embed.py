#!/usr/bin/env python
# Oracle-B1: the transformer's PRE-BLOCK path — patchify, patch_embedder,
# text_embedder, time (Timesteps->TimestepEmbedding->time_modulation)->temb6,
# and the 3D RoPE BUILD (make_joint_position_ids + LingBotVideoRotaryEmbedding).
# Standalone sub-modules, SEEDED synthetic weights (small). Closes skeptic
# FRAGILE-2: gates build_multiaxis_rope_tables + position ids, which A1/A3 skipped.
import os, sys, math
import torch, torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")
from lingbot_video.transformer_lingbot_video import (
    LingBotVideoTextEmbedder, LingBotVideoRotaryEmbedding, make_joint_position_ids,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
H, TEXT_DIM, FREQ_DIM, PATCH = 2048, 2560, 256, (1, 2, 2)
IN_CH = 16
AXES_DIMS, AXES_LENS, THETA = (32, 48, 48), (4096, 512, 512), 256.0
GT, GH, GW, TEXT_LEN = 1, 8, 8, 8              # latent 16x16 -> patch -> 8x8 video tokens; S=72
DEV = "cuda"


def main():
    torch.manual_seed(909)
    patch_embedder = nn.Linear(IN_CH * math.prod(PATCH), H, bias=True).to(DEV)        # 64->2048
    text_embedder = LingBotVideoTextEmbedder(TEXT_DIM, H).to(DEV)
    time_proj = Timesteps(FREQ_DIM, flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedder = TimestepEmbedding(FREQ_DIM, H, act_fn="silu", sample_proj_bias=True).to(DEV)
    time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(H, 6 * H)).to(DEV)
    for m in (patch_embedder, text_embedder, time_embedder, time_modulation):
        m.float().eval()
    rope = LingBotVideoRotaryEmbedding(AXES_DIMS, AXES_LENS, THETA)

    n_video = GT * GH * GW
    S = n_video + TEXT_LEN
    latent = torch.randn(1, IN_CH, GT, GH * PATCH[1], GW * PATCH[2], device=DEV, dtype=torch.float32)
    timestep = torch.tensor([500.0], device=DEV)
    text_embeds = torch.randn(1, TEXT_LEN, TEXT_DIM, device=DEV, dtype=torch.float32)

    with torch.no_grad():
        B, C, T, Hh, Ww = latent.shape
        pF, pH, pW = PATCH
        gt, gh, gw = T // pF, Hh // pH, Ww // pW
        patch_tokens = latent.reshape(B, C, gt, pF, gh, pH, gw, pW)
        patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, gt * gh * gw, pF * pH * pW * C)
        x = patch_embedder(patch_tokens)                              # (1,64,2048)
        text = text_embedder(text_embeds)                            # (1,8,2048)
        joint = torch.cat([x, text], dim=1)                          # (1,72,2048)
        pos_ids = make_joint_position_ids(TEXT_LEN, gt, gh, gw, torch.device(DEV))
        freqs = rope(pos_ids)                                        # (72,64) complex
        t_emb = time_embedder(time_proj(timestep.float()))           # (1,2048)
        temb_input = t_emb.unsqueeze(1).expand(B, S, -1)
        temb6 = time_modulation(temb_input.reshape(B * S, -1)).reshape(B, S, -1)  # (1,72,12288)

    caps = {
        "latent": latent.cpu(), "timestep": timestep.cpu(), "text_embeds": text_embeds.cpu(),
        "patch_tokens": patch_tokens.cpu(), "x": x.cpu(), "text": text.cpu(), "joint": joint.cpu(),
        "pos_ids": pos_ids.float().cpu(), "freqs_cos": freqs.real.float().cpu(), "freqs_sin": freqs.imag.float().cpu(),
        "t_emb": t_emb.cpu(), "temb6": temb6.cpu(),
        "patch_embedder_w": patch_embedder.weight.cpu(), "patch_embedder_b": patch_embedder.bias.cpu(),
        "text_norm_w": text_embedder.norm.weight.cpu(),
        "text_lin1_w": text_embedder.linear_1.weight.cpu(), "text_lin1_b": text_embedder.linear_1.bias.cpu(),
        "text_lin2_w": text_embedder.linear_2.weight.cpu(), "text_lin2_b": text_embedder.linear_2.bias.cpu(),
        "time_lin1_w": time_embedder.linear_1.weight.cpu(), "time_lin1_b": time_embedder.linear_1.bias.cpu(),
        "time_lin2_w": time_embedder.linear_2.weight.cpu(), "time_lin2_b": time_embedder.linear_2.bias.cpu(),
        "time_mod_w": time_modulation[1].weight.cpu(), "time_mod_b": time_modulation[1].bias.cpu(),
    }
    save_file(caps, os.path.join(OUT, "oracle_b1.safetensors"))
    print(f"SAVED oracle_b1.safetensors  S={S} n_video={n_video}")
    print(f"  x std {x.std():.5f} | text std {text.std():.5f} | temb6 std {temb6.std():.5f}")
    print(f"  freqs_cos[0,:3]={freqs.real[0,:3].tolist()}  pos_ids[0]={pos_ids[0].tolist()} pos_ids[{n_video}]={pos_ids[n_video].tolist()}")


if __name__ == "__main__":
    main()
