#!/usr/bin/env python3
# serenitymojo/models/wan22/parity/bernini_src_id_rope_oracle.py
#
# Torch oracle for the BERNINI-R source-id RoPE
# (models/wan22/bernini_src_id_rope.mojo). Replicates the EXACT math of
# Bernini/bernini/models/transformer_wan.py::WanRotaryPosEmbed.forward
# (the standard Wan 3D rope + the use_src_id_rotary_emb complex phase), using the
# SAME diffusers.get_1d_rotary_pos_embed Bernini itself calls.
#
# REAL head_dim = 128 (Wan2.2 A14B). Small patch grid (ppf,pph,ppw) so S is tiny.
# Several source_ids: 0 (identity), 2 (integer), 1.5 (FRACTIONAL — Bernini's key
# feature). Dumps the final complex freqs' real (cos) / imag (sin) per source_id
# as [S, head_dim/2] F32 .bin files the Mojo gate reads at cos>=0.999.
#
# Run (SEPARATE command, torchref venv):
#   /home/alex/torchref-image/venv/bin/python \
#       serenitymojo/models/wan22/parity/bernini_src_id_rope_oracle.py

import os
import struct
import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed

REF_DIR = os.path.dirname(os.path.abspath(__file__))

# ── dims (must match the Mojo gate) ──
HEAD_DIM = 128
THETA = 10000.0
PPF, PPH, PPW = 1, 2, 3          # patch grid -> S = 6 tokens
S = PPF * PPH * PPW
HALF = HEAD_DIM // 2             # 64
MAX_SEQ_LEN = 64                 # >= max(PPF,PPH,PPW)

SOURCE_IDS = [("0", 0.0), ("2", 2.0), ("1p5", 1.5)]


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


def build_standard_freqs():
    """Stock Wan 3D rope freqs [1,1,S,head_dim/2] complex (WanRotaryPosEmbed)."""
    h_dim = w_dim = 2 * (HEAD_DIM // 6)      # 42
    t_dim = HEAD_DIM - h_dim - w_dim         # 44
    freqs = []
    for dim in [t_dim, h_dim, w_dim]:
        freq = get_1d_rotary_pos_embed(
            dim, MAX_SEQ_LEN, THETA,
            use_real=False, repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        )
        freqs.append(freq)
    freqs_all = torch.cat(freqs, dim=1)      # [MAX_SEQ_LEN, 64]

    parts = freqs_all.split_with_sizes(
        [HEAD_DIM // 2 - 2 * (HEAD_DIM // 6), HEAD_DIM // 6, HEAD_DIM // 6],
        dim=1,
    )                                        # [22, 21, 21]
    freqs_f = parts[0][:PPF].view(PPF, 1, 1, -1).expand(PPF, PPH, PPW, -1)
    freqs_h = parts[1][:PPH].view(1, PPH, 1, -1).expand(PPF, PPH, PPW, -1)
    freqs_w = parts[2][:PPW].view(1, 1, PPW, -1).expand(PPF, PPH, PPW, -1)
    freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, S, -1)
    return freqs                             # complex128 [1,1,S,64]


def apply_source_id(freqs, source_id):
    """freqs * freqs_visual_id (WanRotaryPosEmbed.forward :274-289)."""
    pos = torch.tensor([float(source_id)], dtype=torch.float64)
    freqs_visual_id = get_1d_rotary_pos_embed(
        HEAD_DIM, pos, THETA,
        use_real=False, repeat_interleave_real=False,
        freqs_dtype=torch.float64,
    )                                        # [1, 64] complex
    freqs_visual_id = (
        freqs_visual_id.view(1, 1, 1, -1)
        .expand(PPF, PPH, PPW, -1)
        .reshape(1, 1, S, -1)
    )
    return freqs * freqs_visual_id


def main():
    freqs_std = build_standard_freqs()       # [1,1,S,64]

    # Stock (no source-id) cos/sin for the Mojo identity cross-check.
    W("ref_stock_cos", freqs_std.real.reshape(S, HALF))
    W("ref_stock_sin", freqs_std.imag.reshape(S, HALF))

    for tag, sid in SOURCE_IDS:
        freqs = apply_source_id(freqs_std, sid)     # [1,1,S,64] complex
        cos = freqs.real.reshape(S, HALF)
        sin = freqs.imag.reshape(S, HALF)
        W("ref_cos_" + tag, cos)
        W("ref_sin_" + tag, sin)

    print("S =", S, " head_dim =", HEAD_DIM, " half =", HALF, " theta =", THETA)
    print("done.")


if __name__ == "__main__":
    main()
