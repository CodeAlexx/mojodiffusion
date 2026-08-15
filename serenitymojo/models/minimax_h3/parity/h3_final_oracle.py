# h3_final_oracle.py — torch-autograd oracle for the H3 FINAL LAYER training
# twin (musubi akane/minimax-h3 @ 04324c28, model.py:400-433) on REAL FL2VA
# final_layer weights. The adaln table is precomputed and dumped (modcache
# contract); backward drives d_hidden through a weighted-sum loss over both
# heads.
#
# Run:  /home/alex/musubi-tuner/.venv/bin/python h3_final_oracle.py
import glob
import json
import sys

sys.path.insert(0, "/home/alex/musubi-h3/src")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
OUT = "/home/alex/mojodiffusion/output/checks/h3_final_oracle.safetensors"

D = 5376
N_TS = 3
S = 384
SV = 256          # video rows
SA = 96           # audio rows (rest of the sequence is text: 32 rows)
EPS_KEY = "final_norm_eps"

with open(f"{CKPT}/model.safetensors.index.json") as f:
    weight_map = json.load(f)["weight_map"]

need = {
    "final_layer.norm.weight",
    "final_layer.adaln_proj.linear.weight",
    "final_layer.adaln_proj.linear.bias",
    "final_layer.video_out.weight",
    "final_layer.video_out.bias",
    "final_layer.audio_out.weight",
    "final_layer.audio_out.bias",
}
tensors = {}
for shard in sorted({weight_map[k] for k in need}):
    with safe_open(f"{CKPT}/{shard}", framework="pt") as f:
        for k in need:
            if weight_map[k] == shard:
                tensors[k] = f.get_tensor(k)

with open(f"{CKPT}/config.json") as f:
    cfg = json.load(f)
eps = float(cfg.get(EPS_KEY, 1e-5))
print("final_norm_eps:", eps, "| adaln_t_table_size:", cfg.get("adaln_t_table_size"))
for k, v in tensors.items():
    print(f"  {k}: {tuple(v.shape)} {v.dtype}")

norm_w = tensors["final_layer.norm.weight"].to(torch.bfloat16)
video_w = tensors["final_layer.video_out.weight"].to(torch.bfloat16)
video_b = tensors["final_layer.video_out.bias"].to(torch.bfloat16)
audio_w = tensors["final_layer.audio_out.weight"].to(torch.bfloat16)
audio_b = tensors["final_layer.audio_out.bias"].to(torch.bfloat16)
PV = video_w.shape[0]
PA = audio_w.shape[0]

torch.manual_seed(11)
hidden = (torch.randn(1, S, D, dtype=torch.float32) * 0.5).to(torch.bfloat16).requires_grad_(True)
# adaln table: what adaln_proj(temb) would produce, bf16 like the .to(dtype)
mod = (torch.randn(N_TS, 2 * D, dtype=torch.float32) * 0.3).to(torch.bfloat16)
ts_idx = torch.arange(S, dtype=torch.long) % N_TS
perm = torch.randperm(S)
video_idx = perm[:SV].sort().values
audio_idx = perm[SV : SV + SA].sort().values

shift, scale = mod.chunk(2, dim=-1)
media_indices = torch.cat((video_idx, audio_idx))
media_ts = ts_idx.index_select(0, media_indices)

norm = torch.nn.RMSNorm(D, eps=eps, dtype=torch.bfloat16)
with torch.no_grad():
    norm.weight.copy_(norm_w)
media = norm(hidden.index_select(1, media_indices))
media = media * (1.0 + scale.index_select(0, media_ts))
media = media + shift.index_select(0, media_ts)
video_hidden, audio_hidden = media.split((SV, SA), dim=1)
video = torch.nn.functional.linear(video_hidden, video_w, video_b)
audio = torch.nn.functional.linear(audio_hidden, audio_w, audio_b)

d_video = (torch.randn(1, SV, PV, dtype=torch.float32) * 0.1).to(torch.bfloat16)
d_audio = (torch.randn(1, SA, PA, dtype=torch.float32) * 0.1).to(torch.bfloat16)
loss = (video.float() * d_video.float()).sum() + (audio.float() * d_audio.float()).sum()
loss.backward()

save_file(
    {
        "hidden": hidden.detach().squeeze(0).contiguous(),
        "mod": mod.contiguous(),
        "ts_idx": ts_idx.to(torch.float32).contiguous(),
        "video_idx": video_idx.to(torch.float32).contiguous(),
        "audio_idx": audio_idx.to(torch.float32).contiguous(),
        "video_out": video.detach().squeeze(0).contiguous(),
        "audio_out": audio.detach().squeeze(0).contiguous(),
        "d_video": d_video.squeeze(0).contiguous(),
        "d_audio": d_audio.squeeze(0).contiguous(),
        "d_hidden": hidden.grad.detach().squeeze(0).float().contiguous(),
        "eps": torch.tensor([eps], dtype=torch.float32),
    },
    OUT,
)
print("wrote", OUT)
print("  video_out std", float(video.float().std()), "audio_out std", float(audio.float().std()))
print("  d_hidden std", float(hidden.grad.float().std()))
