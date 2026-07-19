#!/usr/bin/env python3
# Phase 4a: HF Qwen3-8B reference for the klein worker encode_klein.
# Worker joint [1,512,12288] = cat(states[8], states[17], states[26]) pre-final-norm.
# HF mapping (embedding-indexed): states[k] = hidden_states[k+1].
#   states[8]->hs[9]  states[17]->hs[18]  states[26]->hs[27]
# Feed the BYTE-IDENTICAL dumped ids; compare each 4096-slice on real rows.
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

DEV="cuda"
DUMP="output/checks/phase4a/klein_worker_cond.safetensors"
QDIR="/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
PAD=151643

def rowcos(a,b): return torch.nn.functional.cosine_similarity(a.float(),b.float(),dim=1)
def ocos(a,b):
    a=a.reshape(-1).float(); b=b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(a,b,dim=0))

d=load_file(DUMP)
joint=d["joint"][0].to(DEV)           # [512,12288] bf16
ids=d["ids"].to(torch.int64)
real=int((ids!=PAD).sum().item())
print("real tokens =", real)
ids_t=ids.view(1,-1).to(DEV)

w8  = joint[:, 0:4096]
w17 = joint[:, 4096:8192]
w26 = joint[:, 8192:12288]

m=AutoModelForCausalLM.from_pretrained(QDIR, torch_dtype=torch.bfloat16).to(DEV).eval()
with torch.no_grad():
    o=m(input_ids=ids_t, output_hidden_states=True)
hs=o.hidden_states
print("num hidden_states =", len(hs))
r9, r18, r27 = hs[9][0], hs[18][0], hs[27][0]

def rep(name, w, ref):
    rc=rowcos(w,ref)
    print(f"{name:22s} overall={ocos(w,ref):.6f}  real[0:{real}] min={rc[:real].min():.6f} mean={rc[:real].mean():.6f}  allmin={rc.min():.6f}")

print("--- worker (bf16) vs HF Qwen3-8B (bf16) ---")
rep("states[8]  vs hs[9]",  w8,  r9)
rep("states[17] vs hs[18]", w17, r18)
rep("states[26] vs hs[26]", w26, r27)
rc=rowcos(w26,r27)
print("states26 per-row real cos:", " ".join(f"{v:.4f}" for v in rc[:real].tolist()))
