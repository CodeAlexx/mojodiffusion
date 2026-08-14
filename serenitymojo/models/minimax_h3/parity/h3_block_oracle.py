# h3_block_oracle.py — torch-autograd oracle for the MiniMax-H3 transformer
# block backward (musubi-tuner akane/minimax-h3 @ 04324c28 is THE oracle).
#
# Loads block-0 REAL weights from the FL2VA diffusers shards into musubi's
# MiniMaxH3TransformerBlock, runs fwd + autograd on seeded NON-DEGENERATE
# inputs (randn — house rule: never modular fills), and dumps input/weight
# grads to a safetensors bundle for the Mojo parity gate (cos >= 0.999 per
# tensor). Real head count H=56, Dh=128, hidden=5376, packed-SwiGLU ffn
# 14336; small S keeps torch cheap while every op runs at real width.
#
# Run:  /home/alex/musubi-tuner/.venv/bin/python h3_block_oracle.py
import sys, json, struct, glob
sys.path.insert(0, "/home/alex/musubi-h3/src")

import torch
from safetensors.torch import load_file, save_file
from musubi_tuner.minimax_h3.model import (
    MiniMaxH3TransformerBlock,
    MiniMaxH3TransformerConfig,
)

CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
OUT = "/home/alex/mojodiffusion/output/checks/h3_block0_oracle.safetensors"
S = 384          # packed rows (video+audio mix), small for oracle speed
N_TS = 3         # distinct timestep rows (video/audio/observed classes)

cfg_json = json.load(open(f"{CKPT}/config.json"))
cfg = MiniMaxH3TransformerConfig(
    hidden_size=cfg_json["hidden_size"],
    num_attention_heads=cfg_json["num_attention_heads"],
    attention_head_dim=cfg_json["attention_head_dim"],
    ffn_dim=cfg_json["ffn_hidden_size"],
    time_embed_dim=cfg_json["time_embed_dim"],
    norm_eps=cfg_json["norm_eps"],
    qk_norm_eps=cfg_json["qk_norm_eps"],
    # adaln table: None -> continuous adaln_proj with SiLU (matches ckpt keys)
    adaln_t_table_size=None,
)

# gather block-0 tensors across shards
want = {}
for shard in sorted(glob.glob(f"{CKPT}/model-*.safetensors")):
    with open(shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    hit = [k for k in header if k.startswith("blocks.0.")]
    if hit:
        sd = load_file(shard)
        for k in hit:
            want[k[len("blocks.0."):]] = sd[k]
        del sd
print("block-0 tensors:", len(want))

block = MiniMaxH3TransformerBlock(cfg)
missing, unexpected = block.load_state_dict(want, strict=False)
assert not unexpected, f"unexpected keys: {unexpected}"
print("missing (expected none):", missing)
block = block.cuda().to(torch.bfloat16)
block.train()
for p in block.parameters():
    p.requires_grad_(True)

g = torch.Generator(device="cpu").manual_seed(1234)
x = torch.randn(1, S, cfg.hidden_size, generator=g).cuda().bfloat16().requires_grad_(True)
# scale 0.02: raw randn through the real 2688->32256 adaln_proj explodes
# scales/gates (out.std ~1e5) and washes out bf16 precision for the gate;
# a small-amplitude non-degenerate temb keeps outputs O(10).
temb = (0.02 * torch.randn(N_TS, cfg.time_embed_dim, generator=g)).cuda().bfloat16().requires_grad_(True)
idx = torch.arange(S, device="cuda") % N_TS       # per-row modulation index
# rope contract (_apply_rotary_emb): 2-D [S, rotary_dim], rotary_dim=96
# (6 spatial/temporal axes x rope_freq_dim 16), passthrough = 128-96=32.
ROTARY = 96
cos = torch.randn(S, ROTARY, generator=g).cuda().bfloat16()
sin = torch.randn(S, ROTARY, generator=g).cuda().bfloat16()

# expose the modulation intermediate: our trainer consumes the precomputed
# per-layer mod table (modcache contract), so the gate compares d_modulation
# (scattered per-row grads), not d_temb.
modulation = block.adaln_proj(temb).view(-1, 6 * block.hidden_size).to(x.dtype)
modulation.retain_grad()
shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
norm_h = block._norm_and_modulate(block.norm1, x, shift_attn, scale_attn, idx)
attention = block.attn(norm_h, (cos, sin), None)
h_mid = x + gate_attn.index_select(0, idx) * attention
norm_h2 = block._norm_and_modulate(block.norm2, h_mid, shift_mlp, scale_mlp, idx)
ff = block.mlp(norm_h2)
out = h_mid + gate_mlp.index_select(0, idx) * ff
# scalar loss: seeded projection so every output element carries gradient
w = torch.randn(out.shape, generator=g).cuda().bfloat16()
loss = (out.float() * w.float()).sum()
loss.backward()

bundle = {
    "in_x": x.detach().float().cpu(),
    "in_temb": temb.detach().float().cpu(),
    "in_idx": idx.float().cpu(),
    "in_cos": cos.float().cpu(),
    "in_sin": sin.float().cpu(),
    "loss_w": w.float().cpu(),
    "out": out.detach().float().cpu(),
    "d_x": x.grad.float().cpu(),
    "d_temb": temb.grad.float().cpu(),
    "modulation": modulation.detach().float().cpu(),
    "d_modulation": modulation.grad.float().cpu(),
}
for name, p in block.named_parameters():
    if p.grad is not None:
        bundle["d_" + name.replace(".", "_")] = p.grad.float().cpu()
save_file(bundle, OUT)
print("wrote", OUT, "tensors:", len(bundle))
print("out.std", out.float().std().item(), "d_x.std", x.grad.float().std().item())
