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

# ── LoRA arm: rank-16 adapters on the four musubi targets. RANDOM A and B
# (zeros-B is the canonical init but zeroes the d_A signal — degenerate for
# a gate); scale = alpha/rank = 1.0 (alpha 16). y += scale * (x@A^T)@B^T.
RANK, LORA_SCALE = 16, 1.0
def _mk(out_f, in_f, seed):
    gl = torch.Generator(device="cpu").manual_seed(seed)
    a = (0.05 * torch.randn(RANK, in_f, generator=gl)).cuda().bfloat16().requires_grad_(True)
    b = (0.05 * torch.randn(out_f, RANK, generator=gl)).cuda().bfloat16().requires_grad_(True)
    return a, b
qkv_a, qkv_b = _mk(block.attn.qkv_proj.out_features, block.attn.qkv_proj.in_features, 71)
out_a, out_b = _mk(block.attn.out_proj.out_features, block.attn.out_proj.in_features, 72)
fc1_a, fc1_b = _mk(block.mlp.fc1.out_features, block.mlp.fc1.in_features, 73)
fc2_a, fc2_b = _mk(block.mlp.fc2.out_features, block.mlp.fc2.in_features, 74)

def _lora(x, a, b):
    return LORA_SCALE * ((x @ a.T) @ b.T)

# rebuild the forward WITH lora contributions (same math + lora adds)
norm_h_l = block._norm_and_modulate(block.norm1, x, shift_attn, scale_attn, idx)
qkv_l = block.attn.qkv_proj(norm_h_l) + _lora(norm_h_l, qkv_a, qkv_b)
q_l, k_l, v_l = qkv_l.chunk(3, dim=-1)
bsz, seq, _ = q_l.shape
hd = block.attn.head_dim
q_l = q_l.view(bsz, seq, -1, hd); k_l = k_l.view(bsz, seq, -1, hd); v_l = v_l.view(bsz, seq, -1, hd)
q_l = block.attn.q_norm(q_l); k_l = block.attn.k_norm(k_l)
import musubi_tuner.minimax_h3.model as _m
q_l = _m._apply_rotary_emb(q_l, cos, sin); k_l = _m._apply_rotary_emb(k_l, cos, sin)
att_l = torch.nn.functional.scaled_dot_product_attention(
    q_l.transpose(1, 2), k_l.transpose(1, 2), v_l.transpose(1, 2)
).transpose(1, 2).reshape(bsz, seq, -1)
attn_y_l = block.attn.out_proj(att_l) + _lora(att_l, out_a, out_b)
h_mid_l = x + gate_attn.index_select(0, idx) * attn_y_l
norm_h2_l = block._norm_and_modulate(block.norm2, h_mid_l, shift_mlp, scale_mlp, idx)
fc1_out_l = block.mlp.fc1(norm_h2_l) + _lora(norm_h2_l, fc1_a, fc1_b)
gate_l, value_l = fc1_out_l.chunk(2, dim=-1)
swi_l = torch.nn.functional.silu(gate_l) * value_l
ff_l = block.mlp.fc2(swi_l) + _lora(swi_l, fc2_a, fc2_b)
out_lora = h_mid_l + gate_mlp.index_select(0, idx) * ff_l
# scalar loss: seeded projection so every output element carries gradient
w = torch.randn(out.shape, generator=g).cuda().bfloat16()
loss = (out.float() * w.float()).sum()
loss.backward(retain_graph=True)

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
# ── LoRA arm backward: base grads captured above; zero everything shared,
# then backprop the lora-arm loss so d_lora_* are pure second-pass grads.
block.zero_grad(set_to_none=True)
loss_l = (out_lora.float() * w.float()).sum()
loss_l.backward()
bundle["lora_out"] = out_lora.detach().float().cpu()
for nm2, t2 in [("qkv_a",qkv_a),("qkv_b",qkv_b),("out_a",out_a),("out_b",out_b),
               ("fc1_a",fc1_a),("fc1_b",fc1_b),("fc2_a",fc2_a),("fc2_b",fc2_b)]:
    bundle["lora_" + nm2] = t2.detach().float().cpu()
    bundle["d_lora_" + nm2] = t2.grad.float().cpu()
save_file(bundle, OUT)
print("wrote", OUT, "tensors:", len(bundle))
print("out.std", out.float().std().item(), "d_x.std", x.grad.float().std().item())

# ── 2-BLOCK CHAIN ARM: blocks 0+1 composed (no LoRA), for the Mojo stack
# driver's recompute + d_x-handoff gate. Same inputs; fresh grads.
want1 = {}
for shard in sorted(glob.glob(f"{CKPT}/model-*.safetensors")):
    with open(shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    hit = [k for k in header if k.startswith("blocks.1.")]
    if hit:
        sd = load_file(shard)
        for k in hit:
            want1[k[len("blocks.1."):]] = sd[k]
        del sd
block1 = MiniMaxH3TransformerBlock(cfg)
m1, u1 = block1.load_state_dict(want1, strict=False)
assert not m1 and not u1, (m1, u1)
block1 = block1.cuda().to(torch.bfloat16)
block1.train()

block.zero_grad(set_to_none=True)
block1.zero_grad(set_to_none=True)
x2 = x.detach().clone().requires_grad_(True)
temb2 = temb.detach().clone().requires_grad_(True)
mod0 = block.adaln_proj(temb2).view(-1, 6 * block.hidden_size).to(x2.dtype)
mod1 = block1.adaln_proj(temb2).view(-1, 6 * block1.hidden_size).to(x2.dtype)
mod0.retain_grad(); mod1.retain_grad()

def _run_block(b, xin, m):
    sa, sca, ga, sm, scm, gm = m.chunk(6, dim=-1)
    nh = b._norm_and_modulate(b.norm1, xin, sa, sca, idx)
    at = b.attn(nh, (cos, sin), None)
    hm = xin + ga.index_select(0, idx) * at
    n2 = b._norm_and_modulate(b.norm2, hm, sm, scm, idx)
    return hm + gm.index_select(0, idx) * b.mlp(n2)

h1 = _run_block(block, x2, mod0)
h2 = _run_block(block1, h1, mod1)
loss2 = (h2.float() * w.float()).sum()
loss2.backward()

chain = {
    "chain_h1": h1.detach().float().cpu(),
    "chain_out": h2.detach().float().cpu(),
    "chain_d_x": x2.grad.float().cpu(),
    "chain_mod1": mod1.detach().float().cpu(),
    "chain_d_mod0": mod0.grad.float().cpu(),
    "chain_d_mod1": mod1.grad.float().cpu(),
}
for name, p in block.named_parameters():
    if p.grad is not None:
        chain["chain_b0_d_" + name.replace(".", "_")] = p.grad.float().cpu()
for name, p in block1.named_parameters():
    if p.grad is not None:
        chain["chain_b1_d_" + name.replace(".", "_")] = p.grad.float().cpu()
bundle2 = dict(load_file(OUT))
bundle2.update(chain)
save_file(bundle2, OUT)
print("chain arm added:", len(chain), "tensors; chain_out.std", h2.float().std().item())
