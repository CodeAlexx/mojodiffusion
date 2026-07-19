#!/usr/bin/env python
# Oracle-A2: LingBotVideoSparseMoeBlock (grouped-sigmoid token-choice router +
# 128 SwiGLU experts top-8 + 1 shared expert). Standalone, SEEDED synthetic
# weights. Forces the EAGER per-expert path (reproducible parity ref matching the
# Mojo grouped_expert_ffn). e_score_correction_bias seeded NON-ZERO so the
# selection(bias-added)-vs-gating(bias-free) asymmetry is exercised.
import os, sys
import torch
from safetensors.torch import save_file

sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")
from lingbot_video.transformer_lingbot_video import LingBotVideoSparseMoeBlock

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
H, INTER_DENSE, E, TOPK, MOE_I = 2048, 6144, 128, 8, 768
SCORE, NORM_TOPK, NGROUP, TOPKGROUP, RSCALE, NSHARED = "sigmoid", True, 4, 2, 2.5, 1
S = 72                                     # tokens (matches A1: 64 video + 8 text)
DEV = "cuda"


def main():
    torch.manual_seed(4321)
    blk = LingBotVideoSparseMoeBlock(
        H, INTER_DENSE, E, TOPK, MOE_I, SCORE, NORM_TOPK,
        NGROUP, TOPKGROUP, RSCALE, NSHARED,
    ).to(DEV)
    # init the empty-Parameter weights (constructor uses torch.empty)
    with torch.no_grad():
        blk.router.weight.data = (0.02 * torch.randn(E, H, device=DEV)).float()
        # NON-ZERO correction bias -> selection != gating (exercise the asymmetry)
        blk.router.e_score_correction_bias.data = (0.1 * torch.randn(E, device=DEV)).float()
        blk.experts.w1.data = (0.02 * torch.randn(E, MOE_I, H, device=DEV)).bfloat16()
        blk.experts.w2.data = (0.02 * torch.randn(E, H, MOE_I, device=DEV)).bfloat16()
        blk.experts.w3.data = (0.02 * torch.randn(E, MOE_I, H, device=DEV)).bfloat16()
        blk.shared_experts.gate_proj.weight.data = (0.02 * torch.randn(MOE_I, H, device=DEV)).bfloat16()
        blk.shared_experts.up_proj.weight.data = (0.02 * torch.randn(MOE_I, H, device=DEV)).bfloat16()
        blk.shared_experts.down_proj.weight.data = (0.02 * torch.randn(H, MOE_I, device=DEV)).bfloat16()
    blk.eval()

    # Force EAGER per-expert path (grouped_mm and for-loop share the (tokens,counts)
    # signature) so the reference matches the Mojo grouped_expert_ffn per-expert math.
    blk._run_grouped_experts = blk._run_experts_for_loop.__get__(blk, type(blk))

    ffn_in = torch.randn(1, S, H, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        tokens = ffn_in.view(-1, H)
        top_indices, top_scores, logits, scores, sfc = blk.router(tokens)
        out = blk(ffn_in)                       # (1,S,H) — routed + shared

    caps = {
        "ffn_in": ffn_in.float().cpu(),
        "router_weight": blk.router.weight.float().cpu(),
        "e_score_correction_bias": blk.router.e_score_correction_bias.float().cpu(),
        "w1": blk.experts.w1.float().cpu(), "w2": blk.experts.w2.float().cpu(),
        "w3": blk.experts.w3.float().cpu(),
        "shared_gate": blk.shared_experts.gate_proj.weight.float().cpu(),
        "shared_up": blk.shared_experts.up_proj.weight.float().cpu(),
        "shared_down": blk.shared_experts.down_proj.weight.float().cpu(),
        "top_indices": top_indices.int().cpu(),      # (S,8) selected expert ids
        "top_scores": top_scores.float().cpu(),      # (S,8) bias-free,norm,×2.5 gates
        "out": out.float().cpu(),
    }
    save_file(caps, os.path.join(OUT, "oracle_a2.safetensors"))
    print(f"SAVED oracle_a2.safetensors  S={S} E={E} topk={TOPK}")
    print(f"  top_indices[0]={top_indices[0].tolist()}")
    print(f"  top_scores[0]={[round(v,4) for v in top_scores[0].tolist()]}  (sum={top_scores[0].sum():.4f}, ~=2.5)")
    print(f"  out mean {out.float().mean():.5f} std {out.float().std():.5f} absmax {out.float().abs().max():.4f}")


if __name__ == "__main__":
    main()
