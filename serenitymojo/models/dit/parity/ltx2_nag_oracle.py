#!/usr/bin/env python3
"""Oracle for the LTX-2 NAG combine gate (ltx2_nag_parity.mojo).

Emits reference outputs of _nag_combine — a VERBATIM copy of
/home/alex/ltx2-app/archive_pre_lightricks_20260411/nag.py:_nag_combine
(lines 21-46) — for deterministic pos/neg inputs that the Mojo gate fills with
the SAME formula. Only the OUTPUT is read back by the gate.

  guidance = pos*scale - neg*(scale-1)
  norm_pos  = ||pos||_1   over last dim (keepdim)
  norm_guid = ||guidance||_1 over last dim (keepdim)
  ratio = norm_guid / (norm_pos + 1e-7)
  mask = ratio > tau
  adjustment = (norm_pos*tau)/(norm_guid + 1e-7)
  guidance = where(mask, guidance*adjustment, guidance)
  out = guidance*alpha + pos*(1-alpha)

Run:  python3 serenitymojo/models/dit/parity/ltx2_nag_oracle.py
Writes: serenitymojo/models/dit/parity/ltx2_nag_ref.txt
"""

import os

import torch

OUT = os.path.join(os.path.dirname(__file__), "ltx2_nag_ref.txt")

# Community defaults (nag.py:8).
SCALE = 11.0
ALPHA = 0.25
TAU = 2.5


def _nag_combine(x_pos, x_neg, scale, alpha, tau):
    guidance = x_pos * scale - x_neg * (scale - 1)
    norm_pos = torch.norm(x_pos, p=1, dim=-1, keepdim=True)
    norm_guid = torch.norm(guidance, p=1, dim=-1, keepdim=True)
    ratio = norm_guid / (norm_pos + 1e-7)
    mask = ratio > tau
    adjustment = (norm_pos * tau) / (norm_guid + 1e-7)
    guidance = torch.where(mask, guidance * adjustment, guidance)
    return guidance * alpha + x_pos * (1 - alpha)


# Deterministic fills — MUST match the Mojo gate's _fill_pos / _fill_neg.
def fill_pos(n):
    return torch.tensor(
        [((i * 7) % 13 - 6) * 0.05 for i in range(n)], dtype=torch.float32
    )


def fill_neg(n):
    return torch.tensor(
        [((i * 5) % 11 - 5) * 0.05 for i in range(n)], dtype=torch.float32
    )


# A second fill set scaled large so SOME rows exceed tau (exercise the clip).
def fill_pos_big(n):
    return torch.tensor(
        [((i * 7) % 13 - 6) * 0.5 for i in range(n)], dtype=torch.float32
    )


def fill_neg_big(n):
    return torch.tensor(
        [((i * 5) % 11 - 5) * 0.5 for i in range(n)], dtype=torch.float32
    )


def line(tag, t):
    flat = t.reshape(-1).tolist()
    return tag + " " + " ".join(repr(float(v)) for v in flat)


def main():
    lines = []

    # Case A: small magnitude (most rows BELOW tau -> no clip).
    S, D = 6, 16
    n = S * D
    pos = fill_pos(n).reshape(1, S, D)
    neg = fill_neg(n).reshape(1, S, D)
    outA = _nag_combine(pos, neg, SCALE, ALPHA, TAU)
    lines.append(line("caseA", outA))

    # Case B: large magnitude (rows ABOVE tau -> L1 clip active).
    pos_b = fill_pos_big(n).reshape(1, S, D)
    neg_b = fill_neg_big(n).reshape(1, S, D)
    outB = _nag_combine(pos_b, neg_b, SCALE, ALPHA, TAU)
    lines.append(line("caseB", outB))

    # Case C: video-like width (D=128, S=4) small magnitude.
    S2, D2 = 4, 128
    n2 = S2 * D2
    posC = fill_pos(n2).reshape(1, S2, D2)
    negC = fill_neg(n2).reshape(1, S2, D2)
    outC = _nag_combine(posC, negC, SCALE, ALPHA, TAU)
    lines.append(line("caseC", outC))

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)
    # Report how many rows triggered the clip in case B (sanity).
    g = pos_b * SCALE - neg_b * (SCALE - 1)
    np = torch.norm(pos_b, p=1, dim=-1)
    ng = torch.norm(g, p=1, dim=-1)
    print("caseB rows over tau:", int((ng / (np + 1e-7) > TAU).sum()), "/", S)


if __name__ == "__main__":
    main()
