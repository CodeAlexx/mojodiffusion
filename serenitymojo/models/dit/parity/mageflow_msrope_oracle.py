#!/usr/bin/env python3
"""OFFLINE parity ORACLE for MageFlow msrope (multi-axis RoPE).

Drives the REAL mage_flow msrope code:
  mage_flow.models.modules.mage_layers.MageFlowEmbedRope  (table builder)
  mage_flow.models.modules.mage_layers.apply_rotary_emb_mageflow  (apply)

MageFlow config: rope_type "msrope", axes_dim [16,56,56] (sum=128=head_dim),
theta 10000, scale_rope=True (native-resolution centering), applied to the
IMAGE/vision tokens ONLY (text tokens are not rotated).

We use a SMALL fixed even grid (frame=1, h=4, w=4 -> 16 image tokens), head_dim
128, a small number of heads, and fixed non-degenerate randn q/k so the Mojo
probe can gate BOTH the cos/sin tables AND the rope-applied q/k.

Run:  /home/alex/OneTrainer/venv/bin/python .../mageflow_msrope_oracle.py
Output: safetensors next to this file, read by mageflow_msrope_probe.mojo.
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

from mage_flow.models.modules.mage_layers import (
    MageFlowEmbedRope,
    apply_rotary_emb_mageflow,
)

OUT = Path(__file__).with_name("mageflow_fx_msrope.safetensors")

THETA = 10000
AXES_DIM = [16, 56, 56]  # sum == 128 == head_dim
FRAME = 1
HEIGHT = 4               # even (scale_rope centering is symmetric for even dims)
WIDTH = 4                # even
HEADS = 24               # production num_heads
HEAD_DIM = 128


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    # --- REAL msrope table builder ---------------------------------------
    rope = MageFlowEmbedRope(theta=THETA, axes_dim=AXES_DIM, scale_rope=True)
    vid_freqs = rope([(FRAME, HEIGHT, WIDTH)], device=device)  # [S_img, 64] complex
    s_img = FRAME * HEIGHT * WIDTH
    assert vid_freqs.shape == (s_img, sum(AXES_DIM) // 2), vid_freqs.shape

    # Reconstruct the per-token, per-axis positions the builder actually used
    # under scale_rope=True (frame idx; height/width centered around 0), so the
    # Mojo probe can feed the SAME positions to build_multiaxis_rope_tables.
    hh = HEIGHT // 2
    ww = WIDTH // 2
    h_positions = list(range(-(HEIGHT - hh), 0)) + list(range(0, hh))
    w_positions = list(range(-(WIDTH - ww), 0)) + list(range(0, ww))
    positions = torch.zeros(s_img, 3, dtype=torch.float32)
    for t in range(s_img):
        f_idx = t // (HEIGHT * WIDTH)
        rem = t % (HEIGHT * WIDTH)
        h_idx = rem // WIDTH
        w_idx = rem % WIDTH
        positions[t, 0] = float(f_idx)
        positions[t, 1] = float(h_positions[h_idx])
        positions[t, 2] = float(w_positions[w_idx])

    # --- fixed non-degenerate q,k, then REAL apply -----------------------
    # apply_rotary_emb_mageflow expects x [B, S, H, D]; freqs [S, 64] complex.
    q = torch.randn(1, s_img, HEADS, HEAD_DIM, dtype=torch.float32)
    k = torch.randn(1, s_img, HEADS, HEAD_DIM, dtype=torch.float32)
    q_out = apply_rotary_emb_mageflow(q, vid_freqs)  # [1, S, H, D]
    k_out = apply_rotary_emb_mageflow(k, vid_freqs)

    # --- lay out for the Mojo probe --------------------------------------
    # rope_interleaved flattens leading dims to rows; Mojo row order is
    # token-major then head: row = t*HEADS + h. Build [S*H, ...] tensors.
    def to_rows(x):  # [1,S,H,D] -> [S*H, D]
        return x[0].reshape(s_img * HEADS, HEAD_DIM).contiguous()

    # cos/sin tables per row (each token's table repeated across heads).
    cos_tok = vid_freqs.real.float().contiguous()        # [S, 64]
    sin_tok = vid_freqs.imag.float().contiguous()         # [S, 64]
    cos_rows = cos_tok.repeat_interleave(HEADS, dim=0).contiguous()  # [S*H,64]
    sin_rows = sin_tok.repeat_interleave(HEADS, dim=0).contiguous()
    pos_rows = positions.repeat_interleave(HEADS, dim=0).contiguous()  # [S*H,3]

    save_file(
        {
            "cos_rows": cos_rows,
            "sin_rows": sin_rows,
            "pos_rows": pos_rows.contiguous(),
            "q": to_rows(q),
            "k": to_rows(k),
            "q_out": to_rows(q_out),
            "k_out": to_rows(k_out),
        },
        str(OUT),
    )
    print(
        f"wrote {OUT}\n  grid f={FRAME} h={HEIGHT} w={WIDTH} -> {s_img} img tokens, "
        f"heads={HEADS}, head_dim={HEAD_DIM}\n  rows={s_img*HEADS} half={cos_tok.shape[1]} "
        f"axes_dim={AXES_DIM} theta={THETA} scale_rope=True\n"
        f"  apply convention: view_as_complex(reshape(...,-1,2)) == INTERLEAVED "
        f"(adjacent-pair)"
    )


if __name__ == "__main__":
    main()
