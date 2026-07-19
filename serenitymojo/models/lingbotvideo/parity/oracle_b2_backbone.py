#!/usr/bin/env python
# Oracle-B2: FULL 48-layer backbone reference forward, MEMORY-LEAN via manual
# per-block streaming (load one block ~1.2GB from the shards, run, free) so the
# 60GB model never resides — fits 16GB VRAM / 60GB RAM. REAL weights. Forces the
# eager MoE path. Captures pre-block joint/temb6/rope, every block output, and the
# final velocity. Small inputs (latent 16x16 -> 64 video tokens + 8 text = S 72).
import os, sys, json, math, time, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")
from lingbot_video.transformer_lingbot_video import (
    LingBotVideoBlock, LingBotVideoTextEmbedder, LingBotVideoRotaryEmbedding,
    make_joint_position_ids,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

MDIR = "/mnt/disk1/models/lingbot-video-moe/transformer"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
DEV = "cuda"
GT, GH, GW, TEXT_LEN = 1, 8, 8, 8                    # latent 16x16 -> 8x8 video toks; S=72


def main():
    torch.manual_seed(31415)
    cfg = json.load(open(os.path.join(MDIR, "config.json")))
    H = cfg["hidden_size"]; DEPTH = cfg["depth"]; PATCH = tuple(cfg["patch_size"])
    IN_CH = cfg["in_channels"]; TEXT_DIM = cfg["text_dim"]; FREQ = cfg["freq_dim"]
    EPS = cfg["norm_eps"]
    wm = json.load(open(os.path.join(MDIR, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    handles = {}
    def W(name):
        shard = wm[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(MDIR, shard), framework="pt", device="cpu")
        return handles[shard].get_tensor(name)     # CPU load; block moved to GPU per-iter

    # ---- embedding modules (real weights) ----
    patch_embedder = nn.Linear(IN_CH * math.prod(PATCH), H, bias=True).to(DEV)
    patch_embedder.weight.data = W("patch_embedder.weight"); patch_embedder.bias.data = W("patch_embedder.bias")
    text_embedder = LingBotVideoTextEmbedder(TEXT_DIM, H).to(DEV)
    text_embedder.norm.weight.data = W("text_embedder.norm.weight")
    text_embedder.linear_1.weight.data = W("text_embedder.linear_1.weight"); text_embedder.linear_1.bias.data = W("text_embedder.linear_1.bias")
    text_embedder.linear_2.weight.data = W("text_embedder.linear_2.weight"); text_embedder.linear_2.bias.data = W("text_embedder.linear_2.bias")
    time_proj = Timesteps(FREQ, flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedder = TimestepEmbedding(FREQ, H, act_fn="silu", sample_proj_bias=True).to(DEV)
    time_embedder.linear_1.weight.data = W("time_embedder.linear_1.weight"); time_embedder.linear_1.bias.data = W("time_embedder.linear_1.bias")
    time_embedder.linear_2.weight.data = W("time_embedder.linear_2.weight"); time_embedder.linear_2.bias.data = W("time_embedder.linear_2.bias")
    time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(H, 6 * H)).to(DEV)
    time_modulation[1].weight.data = W("time_modulation.1.weight"); time_modulation[1].bias.data = W("time_modulation.1.bias")
    rope = LingBotVideoRotaryEmbedding(tuple(cfg["axes_dims"]), tuple(cfg["axes_lens"]), cfg["rope_theta"])
    norm_out = nn.LayerNorm(H, elementwise_affine=False, eps=EPS).to(DEV)
    norm_out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(H, 2 * H)).to(DEV)
    norm_out_modulation[1].weight.data = W("norm_out_modulation.1.weight"); norm_out_modulation[1].bias.data = W("norm_out_modulation.1.bias")
    proj_out = nn.Linear(H, math.prod(PATCH) * cfg["out_channels"]).to(DEV)
    proj_out.weight.data = W("proj_out.weight"); proj_out.bias.data = W("proj_out.bias")
    for m in (patch_embedder, text_embedder, time_embedder, time_modulation, norm_out_modulation, proj_out):
        m.to(DEV).eval()                            # W() now loads CPU -> move embedders to GPU

    n_video = GT * GH * GW
    S = n_video + TEXT_LEN
    latent = torch.randn(1, IN_CH, GT, GH * PATCH[1], GW * PATCH[2], device=DEV, dtype=torch.float32)
    timestep = torch.tensor([500.0], device=DEV)
    text_embeds = torch.randn(1, TEXT_LEN, TEXT_DIM, device=DEV, dtype=torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        B, C, T, Hh, Ww = latent.shape
        pF, pH, pW = PATCH
        gt, gh, gw = T // pF, Hh // pH, Ww // pW
        pt = latent.reshape(B, C, gt, pF, gh, pH, gw, pW).permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, gt*gh*gw, pF*pH*pW*C)
        x = patch_embedder(pt.to(patch_embedder.weight.dtype))
        text = text_embedder(text_embeds.to(text_embedder.linear_1.weight.dtype))
        joint = torch.cat([x, text], dim=1)
        pos_ids = make_joint_position_ids(TEXT_LEN, gt, gh, gw, torch.device(DEV))
        freqs = rope(pos_ids); rotary = freqs.unsqueeze(0)
        t_emb = time_embedder(time_proj(timestep.float()))
        temb_input = t_emb.unsqueeze(1).expand(B, S, -1)
        temb6 = time_modulation(temb_input.reshape(B * S, -1))     # (S, 12288)

    caps = {"latent": latent.cpu(), "timestep": timestep.cpu(), "text_embeds": text_embeds.float().cpu(),
            "joint_preblock": joint.float().cpu(), "temb6": temb6.float().cpu(),
            "freqs_cos": freqs.real.float().cpu(), "freqs_sin": freqs.imag.float().cpu()}

    # ---- stream the 48 blocks ----
    t0 = time.time()
    for i in range(DEPTH):
        blk = LingBotVideoBlock(
            hidden_size=H, num_attention_heads=cfg["num_attention_heads"], intermediate_size=cfg["intermediate_size"],
            norm_eps=EPS, qkv_bias=cfg["qkv_bias"], out_bias=cfg["out_bias"], num_experts=cfg["num_experts"],
            num_experts_per_tok=cfg["num_experts_per_tok"], moe_intermediate_size=cfg["moe_intermediate_size"],
            decoder_sparse_step=cfg["decoder_sparse_step"], mlp_only_layers=tuple(cfg["mlp_only_layers"]),
            n_shared_experts=cfg["n_shared_experts"], score_func=cfg["score_func"], norm_topk_prob=cfg["norm_topk_prob"],
            n_group=cfg["n_group"], topk_group=cfg["topk_group"], routed_scaling_factor=cfg["routed_scaling_factor"], layer_idx=i)
        pref = f"blocks.{i}."
        sd = {k[len(pref):]: W(k) for k in wm if k.startswith(pref)}
        blk.load_state_dict(sd, assign=True)            # assign -> keep stored dtypes (bf16 bulk / fp32 norms)
        blk.to(DEV).eval()
        blk.ffn._run_grouped_experts = blk.ffn._run_experts_for_loop.__get__(blk.ffn, type(blk.ffn))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            joint = blk(joint, temb6, rotary)
        caps[f"block_{i}"] = joint.float().cpu()
        blk.ffn._run_grouped_experts = None         # break the self-referential bound-method cycle
        del blk, sd
        gc.collect()                                # collect the cyclic block NOW (else GPU fills ~block 28)
        torch.cuda.empty_cache()
        if i % 12 == 0:
            print(f"  block {i}/{DEPTH} done ({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        final_mod = norm_out_modulation(temb_input.reshape(B * S, -1))
        shift, scale = final_mod.reshape(B, S, -1).chunk(2, dim=-1)
        final_hidden = norm_out(joint) * (1.0 + scale) + shift
        projected = proj_out(final_hidden.to(proj_out.weight.dtype))
        xv = projected[:, :n_video]
        Cout = cfg["out_channels"]
        xv = xv.reshape(B, gt, gh, gw, pF, pH, pW, Cout).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, Hh, Ww)

    caps["velocity"] = xv.float().cpu()
    save_file(caps, os.path.join(OUT, "oracle_b2.safetensors"))
    print(f"SAVED oracle_b2.safetensors  S={S} depth={DEPTH}  ({time.time()-t0:.0f}s stream)")
    print(f"  velocity {list(xv.shape)} mean {xv.float().mean():.5f} std {xv.float().std():.5f} absmax {xv.float().abs().max():.4f}")


if __name__ == "__main__":
    main()
