#!/usr/bin/env python3
# Phase 4a: HF Qwen3-4B reference for the zimage worker _encode_text_fixed.
# Worker cap [256,2560] = encode(ids, EXTRACT_LAYER=34) sliced to 256 rows, PRE-final-norm.
# EXTRACT_LAYER=34 (Mojo states[34] = output of layer 34) == HF hidden_states[35] (penultimate).
# Feed BYTE-IDENTICAL dumped ids; compare real rows.
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

DEV="cuda"
DUMP="output/checks/phase4a/zimage_worker_cond.safetensors"
ENC="/home/alex/.serenity/models/zimage_base/text_encoder"
PAD=151643

def rowcos(a,b): return torch.nn.functional.cosine_similarity(a.float(),b.float(),dim=1)
def ocos(a,b):
    a=a.reshape(-1).float(); b=b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(a,b,dim=0))

d=load_file(DUMP)
cap=d["cap"].to(DEV)                  # [256,2560] bf16
ids=d["ids"].to(torch.int64)
real=int((ids!=PAD).sum().item())
print("real tokens =", real)
ids_t=ids.view(1,-1).to(DEV)          # [1,512]

m=AutoModelForCausalLM.from_pretrained(ENC, dtype=torch.bfloat16).to(DEV).eval()
with torch.no_grad():
    o=m(input_ids=ids_t, output_hidden_states=True)
hs=o.hidden_states
print("num hidden_states =", len(hs), "(expect 37)")
penult = hs[35][0][:256, :]           # states[34] == hs[35]
# also grab neighbors to confirm the exact layer index is right
h34 = hs[34][0][:256, :]
h36 = hs[36][0][:256, :]

def rep(name, ref):
    rc=rowcos(cap, ref)
    print(f"{name:26s} overall={ocos(cap,ref):.6f}  real[0:{real}] min={rc[:real].min():.6f} mean={rc[:real].mean():.6f}  allmin={rc.min():.6f}")

print("--- worker cap (bf16) vs HF Qwen3-4B (bf16) ---")
rep("cap vs hs[35] (EXTRACT=34)", penult)
rep("cap vs hs[34] (sanity -1)", h34)
rep("cap vs hs[36] post-norm  ", h36)
rc=rowcos(cap, penult)
print("cap-vs-hs35 per-row real cos:", " ".join(f"{v:.4f}" for v in rc[:real].tolist()))
