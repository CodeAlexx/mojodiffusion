#!/usr/bin/env python3
# serenitymojo/models/sdxl/parity/embed_lora_oracle.py
#
# Torch-autograd oracle for the SDXL embed/proj LINEAR-LoRA family that OneTrainer
# trains on the UNet but that does NOT live inside the SpatialTransformer:
#   lora_unet_time_embedding_linear_1 / linear_2,
#   lora_unet_add_embedding_linear_1 / linear_2,
#   lora_unet_*_resnets_*_time_emb_proj
# Every one is a plain Linear adapted by y' = base + scale*((x@A.T)@B.T)
# (A=[rank,in], B=[out,rank], scale=alpha/rank) — the SAME linear-LoRA primitive
# (train_step._lora_fwd/_lora_bwd via sdxl_lora_apply/sdxl_lora_bwd) the
# SpatialTransformer slots use, just at rectangular in!=out embed dims. This gate
# dumps a torch.autograd reference for d_A / d_B / d_x at representative embed dims
# so the family has its own explicit PASS (not only "covered by the ST gate").
#
# Layouts (.bin = flat host f32): x[M,in], A[rank,in], B[out,rank],
#   Wbase[out,in], go[M,out]; refs y_full[M,out], d_A[rank,in], d_B[out,rank],
#   d_x[M,in] (LoRA-branch contribution, delta-only graph).
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#          serenitymojo/models/sdxl/parity/embed_lora_oracle.py

import os
import struct
import torch

DT = torch.float64
REF = os.path.dirname(os.path.abspath(__file__))
RANK = 4
ALPHA = 8.0
LSCALE = ALPHA / RANK


def fill(n, a, b, c, scale=0.05):
    return torch.tensor([(float((i * a) % b) - c) * scale for i in range(n)], dtype=DT)


def WB(name, t):
    arr = t.detach().reshape(-1).tolist()
    with open(os.path.join(REF, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % len(arr), *[float(v) for v in arr]))


def run(tag, M, in_f, out_f):
    x = fill(M * in_f, 7, 13, 6.0, 0.05).reshape(M, in_f).clone().requires_grad_(True)
    A = fill(RANK * in_f, 5, 11, 5.0, 0.02).reshape(RANK, in_f).clone().requires_grad_(True)
    B = fill(out_f * RANK, 3, 7, 3.0, 0.05).reshape(out_f, RANK).clone().requires_grad_(True)  # LIVE B
    Wbase = fill(out_f * in_f, 6, 17, 8.0, 0.02).reshape(out_f, in_f)
    go = fill(M * out_f, 2, 9, 4.0, 0.05).reshape(M, out_f)

    delta = LSCALE * ((x @ A.T) @ B.T)            # [M,out] LoRA branch
    y_full = (x @ Wbase.T + delta).detach()       # forward ref (base + lora)
    loss = (delta * go).sum()                     # delta-only -> isolates LoRA grads
    loss.backward()

    WB("emb_%s_x" % tag, x)
    WB("emb_%s_A" % tag, A)
    WB("emb_%s_B" % tag, B)
    WB("emb_%s_Wbase" % tag, Wbase)
    WB("emb_%s_go" % tag, go)
    WB("emb_%s_yfull" % tag, y_full)
    WB("emb_%s_dA" % tag, A.grad)
    WB("emb_%s_dB" % tag, B.grad)
    WB("emb_%s_dx" % tag, x.grad)
    print("[%s] M=%d in=%d out=%d rank=%d scale=%.3f loss=%.6f" % (tag, M, in_f, out_f, RANK, LSCALE, float(loss)))


def main():
    # representative embed dims (small synthetic; gates MATH, in!=out exercised):
    run("te1", 4, 12, 20)   # time_embedding.linear_1 (in<out)
    run("te2", 3, 20, 16)   # time_embedding.linear_2 / time_emb_proj (in>out)
    run("add1", 2, 28, 20)  # add_embedding.linear_1 (wide in)
    print("wrote embed-LoRA oracle refs to", REF)


if __name__ == "__main__":
    main()
