#!/usr/bin/env python3
# serenitymojo/models/wan22/parity/bernini_cond_forward_oracle.py
#
# Torch oracle for BERNINI-R Tier-2b — the reference-CONDITIONING PACKED sequence.
# Gates the ONE genuinely-new assembly Tier-2b adds on top of the already-certified
# block/stack: the MULTI-SEGMENT source-id RoPE concatenated along the token axis
# for a packed  [conditioning | target]  sequence (real head_dim=128).
#
# Reproduces, per Bernini/bernini/models/transformer_wan.py::WanRotaryPosEmbed
# (the SAME diffusers.get_1d_rotary_pos_embed the model itself calls), the packed
# rope that pack_vae_latents (:268 input_vae_rope.append per source, :281
# torch.cat) produces:
#   segment 0 = CONDITIONING ref: patch grid (1,8,8) -> 64 tokens, source_id = 1
#   segment 1 = TARGET         : patch grid (1,16,16) -> 256 tokens, source_id = 0
#   packed cos/sin = cat([seg0, seg1], tokens) -> [320, head_dim/2]
# Target source_id=0 => its region is BIT-IDENTICAL to the stock Wan 3D rope
# (identity property); the conditioning region (source_id=1) is rotated.
#
# Dumps the packed cos/sin + each segment's stock (no-src-id) cos/sin so the Mojo
# gate (bernini_cond_forward_parity.mojo) can verify at cos>=0.999 AND check the
# identity/rotation split + the target-region slice offset.
#
# Run (SEPARATE command, ai-toolkit venv):
#   /home/alex/ai-toolkit/venv/bin/python \
#       serenitymojo/models/wan22/parity/bernini_cond_forward_oracle.py

import os
import struct
import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed

REF_DIR = os.path.dirname(os.path.abspath(__file__))

HEAD_DIM = 128                    # REAL Wan2.2 A14B head dim
THETA = 10000.0
HALF = HEAD_DIM // 2              # 64
MAX_SEQ_LEN = 64

# Packed segments: (name, ppf, pph, ppw, source_id)
SEGMENTS = [
    ("cond", 1, 8, 8, 1.0),      # conditioning reference, src_id=1, 64 tokens
    ("tgt", 1, 16, 16, 0.0),     # target, src_id=0 (== stock wan rope), 256 tokens
]


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


def build_standard_freqs(ppf, pph, ppw):
    """Stock Wan 3D rope freqs [S, head_dim/2] complex (WanRotaryPosEmbed)."""
    s = ppf * pph * ppw
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
    )
    freqs_f = parts[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_h = parts[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_w = parts[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(s, -1)
    return freqs                             # complex128 [S, 64]


def apply_source_id(freqs, ppf, pph, ppw, source_id):
    """freqs * freqs_visual_id (WanRotaryPosEmbed.forward :274-289)."""
    s = ppf * pph * ppw
    pos = torch.tensor([float(source_id)], dtype=torch.float64)
    fvi = get_1d_rotary_pos_embed(
        HEAD_DIM, pos, THETA,
        use_real=False, repeat_interleave_real=False,
        freqs_dtype=torch.float64,
    )                                        # [1, 64] complex
    fvi = fvi.view(1, -1).expand(s, -1)
    return freqs * fvi


def main():
    packed_cos = []
    packed_sin = []
    for (name, ppf, pph, ppw, sid) in SEGMENTS:
        std = build_standard_freqs(ppf, pph, ppw)          # [S,64]
        W("cf_stock_cos_" + name, std.real.reshape(-1, HALF))
        W("cf_stock_sin_" + name, std.imag.reshape(-1, HALF))
        rot = apply_source_id(std, ppf, pph, ppw, sid)     # [S,64]
        packed_cos.append(rot.real.reshape(-1, HALF))
        packed_sin.append(rot.imag.reshape(-1, HALF))

    cos = torch.cat(packed_cos, dim=0)   # [320,64]
    sin = torch.cat(packed_sin, dim=0)
    W("cf_packed_cos", cos)
    W("cf_packed_sin", sin)

    s_cond = SEGMENTS[0][1] * SEGMENTS[0][2] * SEGMENTS[0][3]
    s_tgt = SEGMENTS[1][1] * SEGMENTS[1][2] * SEGMENTS[1][3]
    print("packed S =", cos.shape[0], " (cond", s_cond, "+ target", s_tgt, ")",
          " head_dim =", HEAD_DIM, " half =", HALF)
    print("done.")


if __name__ == "__main__":
    main()
