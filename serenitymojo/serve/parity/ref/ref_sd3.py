#!/usr/bin/env python3
# Phase 4a: HF reference for the SD3 worker triple encoder.
# Worker context [1,410,4096] decomposes as:
#   rows [0:77]    = pad(CLIP-L penultimate hidden_states[-2], 768->4096)
#   rows [77:154]  = pad(CLIP-G penultimate hidden_states[-2], 1280->4096)
#   rows [154:410] = T5-XXL encoder last_hidden_state [256,4096]
# pooled [1,2048] = [ CLIP-L pooler(768) | CLIP-G text_embeds projected(1280) ]
# Compared against HF on the BYTE-IDENTICAL dumped token ids.
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import (CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection,
                          T5Config, T5EncoderModel)

DEV = "cuda"
DUMP = "output/checks/phase4a/sd3_worker_cond.safetensors"
CLIP_L = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
CLIP_G = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors"
T5P = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors"

def cos(a,b):
    a=a.reshape(-1).float(); b=b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(a,b,dim=0))
def rowcos(a,b):
    return torch.nn.functional.cosine_similarity(a.float(),b.float(),dim=1)
def load_sd(path):
    sd={}
    with safe_open(path,"pt") as f:
        for k in f.keys(): sd[k]=f.get_tensor(k).float()
    return sd

d = load_file(DUMP)
ctx = d["context"][0].to(DEV)      # [410,4096] bf16
pooled = d["pooled"][0].to(DEV)    # [2048]
l_ids = d["l_ids"].to(torch.int64).view(1,-1).to(DEV)
g_ids = d["g_ids"].to(torch.int64).view(1,-1).to(DEV)
t5_ids = d["t5_ids"].to(torch.int64).view(1,-1).to(DEV)

w_l  = ctx[0:77,   0:768]
w_g  = ctx[77:154, 0:1280]
w_t5 = ctx[154:410,0:4096]
w_lp = pooled[0:768]
w_gp = pooled[768:2048]

CLIP_REAL = 11        # BOS + 8 content + EOS@10
t5_real = int((t5_ids[0]!=0).sum().item())  # content + double EOS
print(f"t5 real rows = {t5_real}")

cfg_l = CLIPTextConfig(vocab_size=49408,hidden_size=768,intermediate_size=3072,
    num_hidden_layers=12,num_attention_heads=12,max_position_embeddings=77,
    hidden_act="quick_gelu",layer_norm_eps=1e-5)
cfg_g = CLIPTextConfig(vocab_size=49408,hidden_size=1280,intermediate_size=5120,
    num_hidden_layers=32,num_attention_heads=20,max_position_embeddings=77,
    hidden_act="gelu",layer_norm_eps=1e-5,projection_dim=1280)

ml = CLIPTextModel(cfg_l).to(DEV).float().eval();  ml.load_state_dict(load_sd(CLIP_L),strict=False)
mg = CLIPTextModelWithProjection(cfg_g).to(DEV).float().eval(); mg.load_state_dict(load_sd(CLIP_G),strict=False)
with torch.no_grad():
    ol = ml(input_ids=l_ids, output_hidden_states=True)
    og = mg(input_ids=g_ids, output_hidden_states=True)
l_penult = ol.hidden_states[-2][0]
g_penult = og.hidden_states[-2][0]
l_pool = ol.pooler_output[0]
g_embed = og.text_embeds[0]         # projected [1280]
del ml, mg; torch.cuda.empty_cache()

# T5-XXL v1.1 encoder, bf16 (worker stores bf16; T5 fp16 is unstable).
cfg_t5 = T5Config(vocab_size=32128, d_model=4096, d_kv=64, d_ff=10240,
    num_layers=24, num_heads=64, relative_attention_num_buckets=32,
    relative_attention_max_distance=128, feed_forward_proj="gated-gelu",
    is_encoder_decoder=False, use_cache=False, tie_word_embeddings=False,
    layer_norm_epsilon=1e-6)
mt = T5EncoderModel(cfg_t5).to(DEV).bfloat16().eval()
sd = {}
with safe_open(T5P,"pt") as f:
    for k in f.keys(): sd[k]=f.get_tensor(k).bfloat16()
miss,unexp = mt.load_state_dict(sd, strict=False)
print("[t5] missing:", [m for m in miss if 'embed_tokens' not in m][:4], "unexpected:", unexp[:4])
with torch.no_grad():
    reference = mt(input_ids=t5_ids).last_hidden_state[0]   # [256,4096] bf16

def rep(name, w, ref, real):
    rc = rowcos(w, ref)
    print(f"\n=== {name} ===  overall cos = {cos(w,ref):.6f}")
    print(f"  REAL rows[0:{real}] min row-cos = {rc[:real].min():.6f}   mean = {rc[:real].mean():.6f}")
    print(f"  ALL rows min row-cos = {rc.min():.6f}")

rep("CLIP-L penultimate", w_l, l_penult, CLIP_REAL)
rep("CLIP-G penultimate", w_g, g_penult, CLIP_REAL)
rep("T5-XXL hidden", w_t5, reference, t5_real)
print(f"\n=== pooled ===")
print(f"  CLIP-L pooled worker-vs-HF pooler        = {cos(w_lp,l_pool):.6f}")
print(f"  CLIP-G pooled(proj) worker-vs-HF text_embeds = {cos(w_gp,g_embed):.6f}")
