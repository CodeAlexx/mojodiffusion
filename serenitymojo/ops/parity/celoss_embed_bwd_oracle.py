#!/usr/bin/env python3
# celoss_embed_bwd_oracle.py — PyTorch reference for CrossEntropy / NLL / BCE /
# Embedding BACKWARD (serenitymojo/ops/celoss_embed_backward.mojo).
#
# Oracle = PyTorch (DEV-ONLY, per the parity convention). F64 interior for a
# clean reference; the Mojo path is F32 (cos>=0.999 gate).
#
# Emits a tagged space-separated text file (same format the proven
# sdpa_bwd_parity._read_ref consumes). Inputs are written too so the Mojo
# driver feeds byte-identical data; only the GRADIENT tags are gated:
#   ce_dlogits   — d_logits from F.cross_entropy (reduction="mean")
#   nll_dlp      — d_log_probs from F.nll_loss (reduction="mean")
#   bce_dpred    — d_pred from F.binary_cross_entropy (PLAIN prob form, mean)
#   embed_dtable — d_weight from nn.Embedding
# Plus input tags: ce_logits, ce_target, nll_logprobs, nll_target,
#   bce_pred, bce_target, embed_idx, embed_gradout.
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/ops/parity/celoss_embed_bwd_oracle.py

import os
import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "celoss_embed_bwd_ref.txt")


def emit(lines, tag, vals):
    lines.append(tag + " " + " ".join(f"{float(x):.8f}" for x in vals))


def main():
    torch.manual_seed(0)
    lines = []

    # ── CrossEntropy: [N=5, C=7] ──────────────────────────────────────────────
    N, C = 5, 7
    logits = torch.randn(N, C, dtype=torch.float64, requires_grad=True)
    ce_target = torch.tensor([3, 0, 6, 1, 4], dtype=torch.long)
    ce_loss = F.cross_entropy(logits, ce_target, reduction="mean")
    (ce_dlogits,) = torch.autograd.grad(ce_loss, logits)
    emit(lines, "ce_logits", logits.detach().reshape(-1).tolist())
    emit(lines, "ce_target", ce_target.tolist())
    emit(lines, "ce_dlogits", ce_dlogits.reshape(-1).tolist())

    # ── NLLLoss: input is log-probabilities [N=4, C=6] ────────────────────────
    Nn, Cn = 4, 6
    raw = torch.randn(Nn, Cn, dtype=torch.float64)
    log_probs = F.log_softmax(raw, dim=1).detach().requires_grad_(True)
    nll_target = torch.tensor([2, 5, 0, 3], dtype=torch.long)
    nll_loss = F.nll_loss(log_probs, nll_target, reduction="mean")
    (nll_dlp,) = torch.autograd.grad(nll_loss, log_probs)
    emit(lines, "nll_logprobs", log_probs.detach().reshape(-1).tolist())
    emit(lines, "nll_target", nll_target.tolist())
    emit(lines, "nll_dlp", nll_dlp.reshape(-1).tolist())

    # ── BCELoss (PLAIN probability form): [12] ────────────────────────────────
    Nb = 12
    pred = torch.sigmoid(torch.randn(Nb, dtype=torch.float64)).detach().requires_grad_(True)
    bce_target = (torch.rand(Nb, dtype=torch.float64) > 0.5).double()
    bce_loss = F.binary_cross_entropy(pred, bce_target, reduction="mean")
    (bce_dpred,) = torch.autograd.grad(bce_loss, pred)
    emit(lines, "bce_pred", pred.detach().reshape(-1).tolist())
    emit(lines, "bce_target", bce_target.reshape(-1).tolist())
    emit(lines, "bce_dpred", bce_dpred.reshape(-1).tolist())

    # ── Embedding: table [10, 8], repeated indices accumulate ─────────────────
    num_embeddings, dim = 10, 8
    emb = torch.nn.Embedding(num_embeddings, dim).double()
    emb_idx = torch.tensor([2, 5, 2, 9, 0, 5], dtype=torch.long)
    out = emb(emb_idx)
    g = torch.randn_like(out)
    out.backward(g)
    emit(lines, "embed_idx", emb_idx.tolist())
    emit(lines, "embed_gradout", g.detach().reshape(-1).tolist())
    emit(lines, "embed_dtable", emb.weight.grad.reshape(-1).tolist())

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
