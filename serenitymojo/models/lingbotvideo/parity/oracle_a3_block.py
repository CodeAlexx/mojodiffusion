#!/usr/bin/env python
# Oracle-A3: full LingBotVideoBlock (AdaLN 6-way modulation + attn + MoE + the two
# gated residuals). Standalone, seeded. scale_shift_table seeded NON-ZERO so the
# modulation path is exercised. temb6 synthesized directly (fp32) — the time
# embedder is gated separately in chunk B. Forces eager MoE path.
import os, sys
import torch
from safetensors.torch import save_file

sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")
from lingbot_video.transformer_lingbot_video import (
    LingBotVideoBlock, LingBotVideoRotaryEmbedding, make_joint_position_ids,
)

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
H, HEADS, INTER, EPS = 2048, 16, 6144, 1e-6
E, TOPK, MOE_I, NSHARED = 128, 8, 768, 1
NGROUP, TOPKGROUP, RSCALE = 4, 2, 2.5
AXES_DIMS, AXES_LENS, THETA = (32, 48, 48), (4096, 512, 512), 256.0
GT, GH, GW, TEXT_LEN = 1, 8, 8, 8
DEV = "cuda"


def main():
    torch.manual_seed(2468)
    blk = LingBotVideoBlock(
        hidden_size=H, num_attention_heads=HEADS, intermediate_size=INTER, norm_eps=EPS,
        qkv_bias=False, out_bias=True, num_experts=E, num_experts_per_tok=TOPK,
        moe_intermediate_size=MOE_I, decoder_sparse_step=1, mlp_only_layers=(),
        n_shared_experts=NSHARED, score_func="sigmoid", norm_topk_prob=True,
        n_group=NGROUP, topk_group=TOPKGROUP, routed_scaling_factor=RSCALE, layer_idx=0,
    ).to(DEV)

    with torch.no_grad():
        # AdaLN table: NON-ZERO (constructor zeros it)
        blk.scale_shift_table.data = (0.05 * torch.randn(1, 6 * H, device=DEV)).float()
        for lin in (blk.attn.to_q, blk.attn.to_k, blk.attn.to_v, blk.attn.to_out):
            lin.weight.data = (0.02 * torch.randn_like(lin.weight)).bfloat16()
        blk.attn.to_out.bias.data = (0.01 * torch.randn(H, device=DEV)).bfloat16()
        blk.attn.norm_q.weight.data = (1.0 + 0.02 * torch.randn(H // HEADS, device=DEV)).float()
        blk.attn.norm_k.weight.data = (1.0 + 0.02 * torch.randn(H // HEADS, device=DEV)).float()
        for nm in (blk.norm1, blk.norm2, blk.norm_post_attn, blk.norm_post_ffn):
            nm.weight.data = (1.0 + 0.02 * torch.randn(H, device=DEV)).float()
        blk.ffn.router.weight.data = (0.02 * torch.randn(E, H, device=DEV)).float()
        blk.ffn.router.e_score_correction_bias.data = (0.1 * torch.randn(E, device=DEV)).float()
        blk.ffn.experts.w1.data = (0.02 * torch.randn(E, MOE_I, H, device=DEV)).bfloat16()
        blk.ffn.experts.w2.data = (0.02 * torch.randn(E, H, MOE_I, device=DEV)).bfloat16()
        blk.ffn.experts.w3.data = (0.02 * torch.randn(E, MOE_I, H, device=DEV)).bfloat16()
        blk.ffn.shared_experts.gate_proj.weight.data = (0.02 * torch.randn(MOE_I, H, device=DEV)).bfloat16()
        blk.ffn.shared_experts.up_proj.weight.data = (0.02 * torch.randn(MOE_I, H, device=DEV)).bfloat16()
        blk.ffn.shared_experts.down_proj.weight.data = (0.02 * torch.randn(H, MOE_I, device=DEV)).bfloat16()
    blk.eval()
    blk.ffn._run_grouped_experts = blk.ffn._run_experts_for_loop.__get__(blk.ffn, type(blk.ffn))

    n_video = GT * GH * GW
    S = n_video + TEXT_LEN
    x = torch.randn(1, S, H, device=DEV, dtype=torch.bfloat16)
    temb6 = (0.5 * torch.randn(S, 6 * H, device=DEV)).float()          # (B*S, 6D), fp32

    rope = LingBotVideoRotaryEmbedding(AXES_DIMS, AXES_LENS, THETA)
    pos_ids = make_joint_position_ids(TEXT_LEN, GT, GH, GW, torch.device(DEV))
    freqs = rope(pos_ids)
    rotary = freqs.unsqueeze(0)

    with torch.no_grad():
        out = blk(x, temb6, rotary)

    caps = {
        "x": x.float().cpu(), "temb6": temb6.cpu(),
        "freqs_cos": freqs.real.float().cpu(), "freqs_sin": freqs.imag.float().cpu(),
        "scale_shift_table": blk.scale_shift_table.float().cpu(),
        "w_q": blk.attn.to_q.weight.float().cpu(), "w_k": blk.attn.to_k.weight.float().cpu(),
        "w_v": blk.attn.to_v.weight.float().cpu(), "w_o": blk.attn.to_out.weight.float().cpu(),
        "b_o": blk.attn.to_out.bias.float().cpu(),
        "norm_q_w": blk.attn.norm_q.weight.float().cpu(), "norm_k_w": blk.attn.norm_k.weight.float().cpu(),
        "norm1_w": blk.norm1.weight.float().cpu(), "norm2_w": blk.norm2.weight.float().cpu(),
        "norm_post_attn_w": blk.norm_post_attn.weight.float().cpu(),
        "norm_post_ffn_w": blk.norm_post_ffn.weight.float().cpu(),
        "router_weight": blk.ffn.router.weight.float().cpu(),
        "e_score_correction_bias": blk.ffn.router.e_score_correction_bias.float().cpu(),
        "w1": blk.ffn.experts.w1.float().cpu(), "w2": blk.ffn.experts.w2.float().cpu(),
        "w3": blk.ffn.experts.w3.float().cpu(),
        "shared_gate": blk.ffn.shared_experts.gate_proj.weight.float().cpu(),
        "shared_up": blk.ffn.shared_experts.up_proj.weight.float().cpu(),
        "shared_down": blk.ffn.shared_experts.down_proj.weight.float().cpu(),
        "out": out.float().cpu(),
    }
    save_file(caps, os.path.join(OUT, "oracle_a3.safetensors"))
    print(f"SAVED oracle_a3.safetensors  S={S}")
    print(f"  out mean {out.float().mean():.5f} std {out.float().std():.5f} absmax {out.float().abs().max():.4f}")


if __name__ == "__main__":
    main()
