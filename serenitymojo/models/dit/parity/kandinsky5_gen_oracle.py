#!/usr/bin/env python
# kandinsky5_gen_oracle.py — oracle for the Kandinsky-5 DiT block-0 + full-stack
# parity. Instantiates ONE canonical TransformerDecoderBlock (visual: self-attn +
# cross-attn + FFN, the full decoder block) from the canonical musubi-tuner
# kandinsky5 nn.py, runs it bf16 on GPU, and dumps byte-identical f32
# inputs/outputs + the block weights (bf16 / f32 modulation) as a safetensors so
# the Mojo path loads the EXACT same weights.
#
# Config: Kandinsky-5.0-T2V-Lite-sft-5s — model_dim=1792, ff_dim=7168,
# time_dim=512, axes_dims=[16,24,24] -> head_dim=64, num_heads=28, patch (1,2,2).
#
# Grid: F=1, H=4, W=4 (patched d_out=1,h_out=2,w_out=2 -> S=4); TXT=8.
#
# Run:
#   /home/alex/musubi-tuner/.venv/bin/python \
#     serenitymojo/models/dit/parity/kandinsky5_gen_oracle.py

import os, sys, struct
import numpy as np
import torch

sys.path.insert(0, "/home/alex/musubi-tuner/src")
from musubi_tuner.kandinsky5.models.nn import (
    Modulation, MultiheadSelfAttentionDec, MultiheadCrossAttention,
    FeedForward, RoPE3D, apply_scale_shift_norm, apply_gate_sum,
)
from torch import nn

DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
DEV = "cuda"
torch.manual_seed(1234)

MODEL_DIM = 1792
TIME_DIM = 512
FF_DIM = 7168
AXES = (16, 24, 24)
HEAD_DIM = sum(AXES)          # 64
NUM_HEADS = MODEL_DIM // HEAD_DIM  # 28
EPS = 1e-6

# patched grid
D_OUT, H_OUT, W_OUT = 1, 2, 2
S = D_OUT * H_OUT * W_OUT     # 4
TXT = 8


def w_f32_bin(name, t):
    a = t.detach().to(torch.float32).contiguous().cpu().numpy().astype("<f4")
    a.tofile(DIR + name + ".bin")
    with open(DIR + name + ".shape", "w") as f:
        f.write(",".join(str(x) for x in t.shape))


class DecoderBlock(nn.Module):
    """Mirror of TransformerDecoderBlock (dit.py) without the offloader import."""
    def __init__(self):
        super().__init__()
        self.visual_modulation = Modulation(TIME_DIM, MODEL_DIM, 9)
        self.self_attention_norm = nn.LayerNorm(MODEL_DIM, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionDec(MODEL_DIM, HEAD_DIM, "sdpa")
        self.cross_attention_norm = nn.LayerNorm(MODEL_DIM, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(MODEL_DIM, HEAD_DIM, "sdpa")
        self.feed_forward_norm = nn.LayerNorm(MODEL_DIM, elementwise_affine=False)
        self.feed_forward = FeedForward(MODEL_DIM, FF_DIM)

    def forward(self, visual_embed, text_embed, time_embed, rope):
        self_p, cross_p, ff_p = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)
        shift, scale, gate = torch.chunk(self_p, 3, dim=-1)
        o = apply_scale_shift_norm(self.self_attention_norm, visual_embed, scale, shift)
        o = self.self_attention(o, rope, None)
        visual_embed = apply_gate_sum(visual_embed, o, gate)
        self._after_sa = visual_embed.clone()

        shift, scale, gate = torch.chunk(cross_p, 3, dim=-1)
        o = apply_scale_shift_norm(self.cross_attention_norm, visual_embed, scale, shift)
        o = self.cross_attention(o, text_embed, None)
        visual_embed = apply_gate_sum(visual_embed, o, gate)
        self._after_ca = visual_embed.clone()

        shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
        o = apply_scale_shift_norm(self.feed_forward_norm, visual_embed, scale, shift)
        o = self.feed_forward(o)
        visual_embed = apply_gate_sum(visual_embed, o, gate)
        return visual_embed


def main():
    # Modulation out_layer is zero-initialized by Modulation.__init__; re-randomize
    # so AdaLN params are non-degenerate (the real checkpoint is trained non-zero).
    blk = DecoderBlock().to(DEV)
    blk = blk.to(torch.bfloat16)
    for m in [blk.self_attention, blk.cross_attention]:
        for nm in ["to_query", "to_key", "to_value", "out_layer"]:
            lin = getattr(m, nm)
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.normal_(lin.bias, std=0.02)
        for nm in ["query_norm", "key_norm"]:
            nn.init.normal_(getattr(m, nm).weight, mean=1.0, std=0.02)
    nn.init.normal_(blk.feed_forward.in_layer.weight, std=0.02)
    nn.init.normal_(blk.feed_forward.out_layer.weight, std=0.02)
    # modulation: keep weights in F32 (matches checkpoint dtype), non-zero.
    blk.visual_modulation = blk.visual_modulation.to(torch.float32)
    nn.init.normal_(blk.visual_modulation.out_layer.weight, std=0.02)
    nn.init.normal_(blk.visual_modulation.out_layer.bias, std=0.02)

    # ── inputs ──
    # REAL model: dit.py before_visual_transformer_blocks -> fractal_flatten ->
    # x.flatten(0,2) -> RANK-2 (S, dim). The decoder self-attn's unsqueeze(0) then
    # makes F.sdpa contract over S (STANDARD spatial attention, 28 heads). Feeding
    # rank-3 (1,S,dim) here would silently switch sdpa to the HEAD axis — that was
    # the oracle bug. Feed rank-2 to reflect the real model. (Cross-attn forward
    # re-adds/strips a batch dim internally, so rank-2 text is fine too.)
    visual = torch.randn(S, MODEL_DIM, device=DEV, dtype=torch.bfloat16)
    text = torch.randn(TXT, MODEL_DIM, device=DEV, dtype=torch.bfloat16)
    time_embed = torch.randn(1, TIME_DIM, device=DEV, dtype=torch.float32)

    # ── 3D RoPE table for the (D_OUT,H_OUT,W_OUT) grid ──
    rope3d = RoPE3D(AXES).to(DEV)
    pos = [
        torch.arange(D_OUT, device=DEV),
        torch.arange(H_OUT, device=DEV),
        torch.arange(W_OUT, device=DEV),
    ]
    rope = rope3d((D_OUT, H_OUT, W_OUT), pos, (1.0, 1.0, 1.0))
    # rope shape: [D,H,W, 1, head/2, 2, 2]; flatten DHW -> [S,1,head/2,2,2]
    rope_flat = rope.reshape(S, *rope.shape[3:])

    with torch.no_grad():
        out = blk(visual, text, time_embed, rope_flat)

    # ── isolate self-attn: dump the modulated input, rope'd q, and raw attn out ──
    from musubi_tuner.kandinsky5.models.nn import apply_rotary
    with torch.no_grad():
        self_p, _, _ = torch.chunk(blk.visual_modulation(time_embed), 3, dim=-1)
        sh, sc, ga = torch.chunk(self_p, 3, dim=-1)
        sa_in = apply_scale_shift_norm(blk.self_attention_norm, visual, sc, sh)
        q, k, v = blk.self_attention.get_qkv(sa_in)   # rank-2 in -> (S,H,DH)
        q, k = blk.self_attention.norm_qk(q, k)
        w_f32_bin("k5_sa_q_prerope", q)        # [S,H,DH]
        q_r = apply_rotary(q, rope_flat).type_as(q)
        k_r = apply_rotary(k, rope_flat).type_as(k)
        w_f32_bin("k5_sa_q_postrope", q_r)     # [S,H,DH]
        sa_raw = blk.self_attention(sa_in, rope_flat, None)
        w_f32_bin("k5_sa_in", sa_in)           # [S,dim]
        w_f32_bin("k5_sa_raw_out", sa_raw)     # [S,dim]

    # ── dump inputs / output (f32; all rank-2 (S,dim)/(TXT,dim) now) ──
    w_f32_bin("k5_block0_visual_in", visual)        # [S, dim]
    w_f32_bin("k5_block0_text_in", text)            # [TXT, dim]
    w_f32_bin("k5_block0_time_in", time_embed[0])   # [time_dim]
    w_f32_bin("k5_block0_out", out)                 # [S, dim]
    w_f32_bin("k5_block0_after_sa", blk._after_sa)
    w_f32_bin("k5_block0_after_ca", blk._after_ca)

    # ── dump block weights (f32; Mojo casts to bf16, modulation stays f32) ──
    sd = blk.state_dict()
    for k, v in sd.items():
        w_f32_bin("w_" + k.replace(".", "_"), v)

    with open(DIR + "k5_grid.txt", "w") as f:
        f.write(f"S={S} TXT={TXT} DIM={MODEL_DIM} TIME={TIME_DIM} "
                f"HEAD={HEAD_DIM} HEADS={NUM_HEADS} D={D_OUT} H={H_OUT} W={W_OUT}\n")
    print(f"oracle done: S={S} TXT={TXT} dim={MODEL_DIM} head={HEAD_DIM} heads={NUM_HEADS}")
    print("out stats: mean=%.5f std=%.5f" % (out.float().mean().item(), out.float().std().item()))


if __name__ == "__main__":
    main()
