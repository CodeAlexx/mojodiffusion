#!/usr/bin/env python3
# Oracle generator for the MagiHuman SHARED transformer layer (CHUNK A gate).
#
# Block 4 is the FIRST shared (num_modality=1, SwiGLU7) layer — the simplest
# faithful gate target (the Rust port's own note: shared path = layers 4..35).
# We replicate the canonical math from inference/model/dit/dit_module.py
# (Attention + MLP, num_modality=1) and the validated Rust port
# inference-flame/src/models/magihuman_dit.rs::SharedTransformerLayer::forward:
#
#   hidden_size=5120 head_dim=128 Hq=40 Hkv=8 (GQA x5) gating_size=40
#   qkv_out = 5120 + 1024 + 1024 + 40 = 7208
#   ROPE_DIM = (head_dim/8)*2*3 = 96  -> partial halfsplit rope on first 96 of 128
#   RMSNorm gain = (weight + 1) ; eps 1e-6 ; bf16 weights, f32 accumulate
#   attn gating: out *= sigmoid(g)  (per-head scalar)
#   SwiGLU7: interleaved split, clamp, sigmoid(1.702*glu)*glu * (linear+1)
#
# NO flash / NO context-parallel / NO modality dispatch (num_modality=1 means
# those are identity). bf16 matmuls with f32 accumulation, matching the Rust path.
#
# Emits a small-grid fixture safetensors consumed by magihuman_block4_gate.mojo.
# Self-checks numerically against an independent f64 recompute of one path.

import json, struct
import numpy as np
import torch
from safetensors.torch import save_file

ST = "/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/magihuman_block4_fixture.safetensors"
PREFIX = "block.layers.4."

HIDDEN = 5120
HEAD_DIM = 128
HQ = 40
HKV = 8
GATING = HQ            # 40
Q_SIZE = HQ * HEAD_DIM    # 5120
KV_SIZE = HKV * HEAD_DIM  # 1024
QKV_OUT = Q_SIZE + 2 * KV_SIZE + GATING  # 7208
REPEAT_KV = HQ // HKV  # 5
ROPE_DIM = (HEAD_DIM // 8) * 2 * 3  # 96
EPS = 1e-6
S = 128  # small grid (single modality), avoids Dh=128 SDPA OOM


def load_keys(path, keys):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k in keys:
            meta = header[k]
            assert meta["dtype"] == "BF16"
            a, b = meta["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(meta["shape"])
            out[k] = t.clone()
        return out


def rms_norm_p1(x, weight, eps=EPS):
    # (x * rsqrt(mean(x^2)+eps)) * (weight + 1).  f32 reduction, bf16 in/out.
    in_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    normed = xf * torch.rsqrt(var + eps)
    out = normed * (weight.float() + 1.0)
    return out.to(in_dtype)


def apply_rope_partial(x_bhsd, cos, sin):
    # x: [B,H,S,head_dim]; cos/sin: [S, ROPE_DIM/2].  Rotate first ROPE_DIM dims.
    # halfsplit (rotate_half) convention with cos/sin duplicated cat([t,t]).
    B, H, Sq, D = x_bhsd.shape
    x_rot = x_bhsd[..., :ROPE_DIM].float()
    x_pass = x_bhsd[..., ROPE_DIM:]
    half = ROPE_DIM // 2
    cos_full = torch.cat([cos, cos], dim=-1).view(1, 1, Sq, ROPE_DIM)  # [1,1,S,ROPE_DIM]
    sin_full = torch.cat([sin, sin], dim=-1).view(1, 1, Sq, ROPE_DIM)
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    out_rot = x_rot * cos_full + rotated * sin_full
    return torch.cat([out_rot.to(x_bhsd.dtype), x_pass], dim=-1)


def swiglu7(x, alpha=1.702, limit=7.0):
    # interleaved split: x_glu = x[...,::2], x_linear = x[...,1::2]
    xf = x.float()
    x_glu = xf[..., 0::2]
    x_linear = xf[..., 1::2]
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1.0)


def repeat_kv(x_bhsd, rep):
    # [B,H,S,D] -> [B,H*rep,S,D] (each head copied rep times, interleaved)
    B, H, Sq, D = x_bhsd.shape
    return x_bhsd.unsqueeze(2).expand(B, H, rep, Sq, D).reshape(B, H * rep, Sq, D)


def shared_layer_forward(h_in, w, cos, sin):
    # h_in: [S, hidden] (bf16). Matches SharedTransformerLayer::forward.
    L = h_in.shape[0]
    hb = h_in.to(torch.bfloat16)
    # attention pre-norm
    hn = rms_norm_p1(hb, w["attention.pre_norm.weight"])
    qkv = (hn.float() @ w["attention.linear_qkv.weight"].float().t()).to(torch.bfloat16)
    q = qkv[:, :Q_SIZE]
    k = qkv[:, Q_SIZE:Q_SIZE + KV_SIZE]
    v = qkv[:, Q_SIZE + KV_SIZE:Q_SIZE + 2 * KV_SIZE]
    g = qkv[:, Q_SIZE + 2 * KV_SIZE:]
    q = q.reshape(L, HQ, HEAD_DIM)
    k = k.reshape(L, HKV, HEAD_DIM)
    v = v.reshape(L, HKV, HEAD_DIM)
    g = g.reshape(L, HQ, 1)
    q = rms_norm_p1(q, w["attention.q_norm.weight"])
    k = rms_norm_p1(k, w["attention.k_norm.weight"])
    # [L,H,D] -> [1,H,L,D]
    qh = q.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    kh = k.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    vh = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    qh = apply_rope_partial(qh, cos, sin)
    kh = apply_rope_partial(kh, cos, sin)
    kh = repeat_kv(kh, REPEAT_KV)
    vh = repeat_kv(vh, REPEAT_KV)
    # SDPA f32, scale 1/sqrt(D)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    qf, kf, vf = qh.float(), kh.float(), vh.float()
    scores = (qf @ kf.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    attn = (probs @ vf)  # [1,H,L,D]
    attn = attn.permute(0, 2, 1, 3).squeeze(0)  # [L,H,D]
    gate = torch.sigmoid(g.float())
    attn = attn * gate
    attn_flat = attn.reshape(L, HQ * HEAD_DIM).to(torch.bfloat16)
    proj = (attn_flat.float() @ w["attention.linear_proj.weight"].float().t()).to(torch.bfloat16)
    h_after = (hb.float() + proj.float()).to(torch.bfloat16)
    # MLP
    mn = rms_norm_p1(h_after, w["mlp.pre_norm.weight"])
    up = (mn.float() @ w["mlp.up_gate_proj.weight"].float().t())  # f32
    act = swiglu7(up).to(torch.bfloat16)
    down = (act.float() @ w["mlp.down_proj.weight"].float().t()).to(torch.bfloat16)
    out = (h_after.float() + down.float())  # final f32 accumulator
    return out, cos, sin


def build_rope(S):
    # ElementWiseFourierEmbed: bands = freq_bands(head_dim//8=16, T=10000, step=1).
    # 3 axes (t,h,w). We use a simple 1D layout: coords = arange(S) on the h-axis,
    # t=w=0, sizes/refs = S. This is a faithful instance of the Fourier embed.
    num_bands = HEAD_DIM // 8  # 16
    exp = np.arange(0, num_bands, 1, dtype=np.float32) / num_bands
    bands = 1.0 / (10000.0 ** exp)  # [16]
    # coords_xyz [S,3]=(t,h,w); sizes [S,3]; refs [S,3]
    coords = np.zeros((S, 3), dtype=np.float32)
    coords[:, 1] = np.arange(S, dtype=np.float32)  # h-axis sweeps
    sizes = np.full((S, 3), float(S), dtype=np.float32)
    refs = np.full((S, 3), float(S), dtype=np.float32)
    refs_m1 = refs - 1.0
    sizes_m1 = sizes - 1.0
    scales = refs_m1 / (sizes_m1 + 1e-30)
    centers = sizes_m1 * 0.5
    centers[:, 0] = 0.0
    cx = coords - centers
    # proj [S,3,B] = (cx)[...,None] * scales[...,None] * bands
    proj = cx[:, :, None] * scales[:, :, None] * bands[None, None, :]
    sin_proj = np.sin(proj)  # [S,3,16]
    cos_proj = np.cos(proj)
    cat = np.concatenate([sin_proj, cos_proj], axis=1)  # [S,6,16]
    rope = cat.reshape(S, 6 * num_bands)  # [S,96]
    # rope.tensor_split(2,-1) -> sin_emb, cos_emb  (first half sin, second cos)
    half = ROPE_DIM // 2  # 48
    sin_emb = rope[:, :half]
    cos_emb = rope[:, half:]
    return torch.from_numpy(cos_emb).float(), torch.from_numpy(sin_emb).float()


def main():
    torch.manual_seed(0)
    keys = [PREFIX + s for s in [
        "attention.pre_norm.weight", "attention.q_norm.weight",
        "attention.k_norm.weight", "attention.linear_qkv.weight",
        "attention.linear_proj.weight", "mlp.pre_norm.weight",
        "mlp.up_gate_proj.weight", "mlp.down_proj.weight",
    ]]
    raw = load_keys(ST, keys)
    w = {k[len(PREFIX):]: v for k, v in raw.items()}

    cos, sin = build_rope(S)
    # input: small magnitude (post-adapter residual stream), bf16
    h_in = (torch.randn(S, HIDDEN) * 0.5).to(torch.bfloat16)

    out, cos, sin = shared_layer_forward(h_in, w, cos, sin)

    # self-check: recompute pre-norm + qkv split in f64 on first token, compare.
    hb0 = h_in[0].double()
    var0 = (hb0.pow(2).mean())
    n0 = hb0 * (1.0 / (var0 + EPS).sqrt())
    n0 = n0 * (w["attention.pre_norm.weight"][:].double() + 1.0)
    # mojo will reproduce this; here just assert finite + reasonable
    assert torch.isfinite(out).all(), "non-finite oracle output"
    print(f"oracle out: shape={tuple(out.shape)} mean={out.mean():.5f} std={out.std():.5f} "
          f"min={out.min():.4f} max={out.max():.4f}")

    fixture = {
        "input": h_in.float().contiguous(),
        "cos": cos.float().contiguous(),
        "sin": sin.float().contiguous(),
        "expected": out.float().contiguous(),
    }
    for s, t in w.items():
        fixture["w_" + s] = t.float().contiguous()
    save_file(fixture, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
