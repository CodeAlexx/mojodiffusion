#!/usr/bin/env python3
# HF CLIP reference for the SDXL worker runtime CLIP encode (gate:
# serenitymojo/serve/parity/sdxl_worker_encode_gate.mojo).
# Loads the SAME standalone clip_l/clip_g safetensors the worker uses, runs HF
# CLIPTextModel on the BYTE-IDENTICAL token ids the worker dumped, and compares
# the worker's context tensor against:
#   (a) HF hidden_states[-2]  = penultimate = the diffusers/SDXL context tensor
#   (b) HF last_hidden_state  = post-final-LN last hidden
# MJ-1061 fix: the gate now calls encode_sdxl(..., penultimate_context=True), so
# the worker returns the PENULTIMATE context. PASS = comparison (a) high
# (measured min real-row cos 0.999797); (b) is now the LOW one (~0.68-0.73).
# Before MJ-1061 (post-final-LN return, MJ-1052) the two were swapped and (a)
# FAILED at 0.727/0.680 — that is the bug this gate now guards against.
import torch, numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

DEV = "cuda"
DUMP = "output/checks/phase4a/sdxl_worker_clip.safetensors"
CLIP_L = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
CLIP_G = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors"
REAL_ROWS = 11  # BOS + 8 content + EOS at idx10 => rows 0..10 are "real"

def cos(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

def rowcos(a, b):  # a,b [S,H]
    a = a.float(); b = b.float()
    return torch.nn.functional.cosine_similarity(a, b, dim=1)  # [S]

d = load_file(DUMP)
l_ids = d["l_ids"].to(torch.int64).view(1, -1).to(DEV)
g_ids = d["g_ids"].to(torch.int64).view(1, -1).to(DEV)
w_l = d["l_hidden"][0].to(DEV)   # [77,768] worker (post-final-LN)
w_g = d["g_hidden"][0].to(DEV)   # [77,1280]
w_lp = d["l_pooled"][0].to(DEV)  # [768]
w_gp = d["g_pooled"][0].to(DEV)  # [1280] (raw, pre text_projection)

cfg_l = CLIPTextConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072,
                       num_hidden_layers=12, num_attention_heads=12,
                       max_position_embeddings=77, hidden_act="quick_gelu",
                       layer_norm_eps=1e-5)
cfg_g = CLIPTextConfig(vocab_size=49408, hidden_size=1280, intermediate_size=5120,
                       num_hidden_layers=32, num_attention_heads=20,
                       max_position_embeddings=77, hidden_act="gelu",
                       layer_norm_eps=1e-5, projection_dim=1280)

def load_sd(path):
    sd = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).float()
    return sd

ml = CLIPTextModel(cfg_l).to(DEV).float().eval()
sl = load_sd(CLIP_L)
missing, unexpected = ml.load_state_dict(sl, strict=False)
print("[clip_l] missing:", [m for m in missing if "position_ids" not in m][:6], "unexpected:", unexpected[:6])

mg = CLIPTextModelWithProjection(cfg_g).to(DEV).float().eval()
sg = load_sd(CLIP_G)
missing_g, unexpected_g = mg.load_state_dict(sg, strict=False)
print("[clip_g] missing:", [m for m in missing_g if "position_ids" not in m][:6], "unexpected:", unexpected_g[:6])

with torch.no_grad():
    ol = ml(input_ids=l_ids, output_hidden_states=True)
    og = mg(input_ids=g_ids, output_hidden_states=True)

# clip_l
l_penult = ol.hidden_states[-2][0]      # diffusers SDXL context
l_lastLN = ol.last_hidden_state[0]      # post-final-LN
l_pool = ol.pooler_output[0]
# clip_g
g_penult = og.hidden_states[-2][0]
g_lastLN = og.last_hidden_state[0]
# pooler for G (raw, pre projection): HF text_model.pooler_output; encode_sdxl g_pooled is raw pooled (pre proj)
g_pool_raw = og.text_model_output.pooler_output[0] if hasattr(og, "text_model_output") else None

def report(name, w, penult, lastLN):
    print(f"\n=== {name} ===")
    print(f"  overall cos  worker-vs-PENULTIMATE(hs[-2], diffusers) = {cos(w, penult):.6f}")
    print(f"  overall cos  worker-vs-POSTLN(last_hidden_state)      = {cos(w, lastLN):.6f}")
    rp = rowcos(w, penult); rl = rowcos(w, lastLN)
    print(f"  REAL rows[0:{REAL_ROWS}] min row-cos vs PENULT = {rp[:REAL_ROWS].min():.6f}  vs POSTLN = {rl[:REAL_ROWS].min():.6f}")
    print(f"  ALL rows[0:77]     min row-cos vs PENULT = {rp.min():.6f}  vs POSTLN = {rl.min():.6f}")
    print(f"  per-row vs POSTLN (rows 0..12): " + " ".join(f"{v:.4f}" for v in rl[:13].tolist()))
    print(f"  per-row vs PENULT (rows 0..12): " + " ".join(f"{v:.4f}" for v in rp[:13].tolist()))

report("CLIP-L hidden", w_l, l_penult, l_lastLN)
report("CLIP-G hidden", w_g, g_penult, g_lastLN)

print("\n=== pooled ===")
print(f"  CLIP-L pooled worker-vs-HF pooler = {cos(w_lp, l_pool):.6f}")
if g_pool_raw is not None:
    print(f"  CLIP-G pooled(raw) worker-vs-HF text_model pooler = {cos(w_gp, g_pool_raw):.6f}")
