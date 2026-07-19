#!/usr/bin/env python3
# Disambiguate the SD3 T5 ~0.995 cos: is it the T5-XXL bf16 noise floor, or a
# real divergence? Run HF T5 in fp32 AND bf16 on the byte-identical dumped ids
# (sequentially to fit 24GB), measure HF-bf16-vs-HF-fp32 (the noise floor), and
# compare the worker (bf16) against both.
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import T5Config, T5EncoderModel

DEV="cuda"
DUMP="output/checks/phase4a/sd3_worker_cond.safetensors"
T5P="/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors"

def rowcos(a,b): return torch.nn.functional.cosine_similarity(a.float(),b.float(),dim=1)
def ocos(a,b):
    a=a.reshape(-1).float(); b=b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(a,b,dim=0))

d=load_file(DUMP)
ctx=d["context"][0]
w_t5=ctx[154:410,0:4096].to(DEV)               # worker T5 (bf16)
t5_ids=d["t5_ids"].to(torch.int64).view(1,-1).to(DEV)
real=int((t5_ids[0]!=0).sum().item())

cfg=T5Config(vocab_size=32128,d_model=4096,d_kv=64,d_ff=10240,num_layers=24,
    num_heads=64,relative_attention_num_buckets=32,relative_attention_max_distance=128,
    feed_forward_proj="gated-gelu",is_encoder_decoder=False,use_cache=False,
    tie_word_embeddings=False,layer_norm_epsilon=1e-6)

def run(dt):
    m=T5EncoderModel(cfg).to(DEV).to(dt).eval()
    sd={}
    with safe_open(T5P,"pt") as f:
        for k in f.keys(): sd[k]=f.get_tensor(k).to(dt)
    m.load_state_dict(sd,strict=False)
    with torch.no_grad():
        o=m(input_ids=t5_ids).last_hidden_state[0].float().cpu()
    del m,sd; torch.cuda.empty_cache()
    return o

o32=run(torch.float32)
o16=run(torch.bfloat16)
o32=o32.to(DEV); o16=o16.to(DEV)

def rep(tag,a,b):
    rc=rowcos(a,b)
    print(f"{tag:34s} overall={ocos(a,b):.6f}  real[0:{real}] min={rc[:real].min():.6f} mean={rc[:real].mean():.6f}")

print("--- T5-XXL bf16 NOISE FLOOR (HF bf16 vs HF fp32) ---")
rep("HF-bf16 vs HF-fp32", o16, o32)
print("--- worker (bf16) vs HF ---")
rep("worker vs HF-fp32", w_t5, o32)
rep("worker vs HF-bf16", w_t5, o16)
rc=rowcos(w_t5,o32)
print("worker-vs-fp32 per-row real cos:", " ".join(f"{v:.4f}" for v in rc[:real].tolist()))
