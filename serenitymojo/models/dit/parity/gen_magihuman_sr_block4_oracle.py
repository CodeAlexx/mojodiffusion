#!/usr/bin/env python3
# Oracle generator for the MagiHuman SR (super-res) SHARED transformer layer.
#
# SR block 4 is the first SHARED (num_modality=1, SwiGLU7) layer. Architecture is
# IDENTICAL to the base distill shared layer (see gen_magihuman_block4_oracle.py)
# EXCEPT the weight layout: SR stores SEPARATE linear_q/k/v/g instead of a fused
# linear_qkv. We load the REAL SR1080 bf16 checkpoint's split weights, run the
# canonical SharedTransformerLayer forward (4 separate Q/K/V/G matmuls), and emit
# a small-grid fixture. The mojo gate fuses the split weights into linear_qkv and
# calls the base block forward — this oracle confirms that fuse is numerically
# faithful to the true 4-matmul path.

import json, struct
import numpy as np
import torch
from safetensors.torch import save_file

ST = "/home/alex/.serenity/models/dits/magi_human_sr1080_bf16.safetensors"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/magihuman_sr_block4_fixture.safetensors"
PREFIX = "block.layers.4."

HIDDEN = 5120
HEAD_DIM = 128
HQ = 40
HKV = 8
GATING = HQ
Q_SIZE = HQ * HEAD_DIM
KV_SIZE = HKV * HEAD_DIM
REPEAT_KV = HQ // HKV
ROPE_DIM = (HEAD_DIM // 8) * 2 * 3  # 96
EPS = 1e-6
S = 128


def load_keys(path, keys):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k in keys:
            meta = header[k]
            assert meta["dtype"] == "BF16", (k, meta["dtype"])
            a, b = meta["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(meta["shape"])
            out[k] = t.clone()
        return out


def rms_norm_p1(x, weight, eps=EPS):
    in_dtype = x.dtype
    xf = x.float()
    v = xf.pow(2).mean(-1, keepdim=True)
    normed = xf * torch.rsqrt(v + eps)
    out = normed * (weight.float() + 1.0)
    return out.to(in_dtype)


def apply_rope_partial(x_bhsd, cos, sin):
    B, H, Sq, D = x_bhsd.shape
    x_rot = x_bhsd[..., :ROPE_DIM].float()
    x_pass = x_bhsd[..., ROPE_DIM:]
    half = ROPE_DIM // 2
    cos_full = torch.cat([cos, cos], dim=-1).view(1, 1, Sq, ROPE_DIM)
    sin_full = torch.cat([sin, sin], dim=-1).view(1, 1, Sq, ROPE_DIM)
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    out_rot = x_rot * cos_full + rotated * sin_full
    return torch.cat([out_rot.to(x_bhsd.dtype), x_pass], dim=-1)


def swiglu7(x, alpha=1.702, limit=7.0):
    xf = x.float()
    x_glu = xf[..., 0::2].clamp(max=limit)
    x_linear = xf[..., 1::2].clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1.0)


def repeat_kv(x_bhsd, rep):
    B, H, Sq, D = x_bhsd.shape
    return x_bhsd.unsqueeze(2).expand(B, H, rep, Sq, D).reshape(B, H * rep, Sq, D)


def sr_shared_forward(h_in, w, cos, sin):
    # SR: separate linear_q/k/v/g (4 matmuls), then identical to base.
    L = h_in.shape[0]
    hb = h_in.to(torch.bfloat16)
    hn = rms_norm_p1(hb, w["attention.pre_norm.weight"])
    hnf = hn.float()
    q = (hnf @ w["attention.linear_q.weight"].float().t()).to(torch.bfloat16)
    k = (hnf @ w["attention.linear_k.weight"].float().t()).to(torch.bfloat16)
    v = (hnf @ w["attention.linear_v.weight"].float().t()).to(torch.bfloat16)
    g = (hnf @ w["attention.linear_g.weight"].float().t()).to(torch.bfloat16)
    q = q.reshape(L, HQ, HEAD_DIM)
    k = k.reshape(L, HKV, HEAD_DIM)
    v = v.reshape(L, HKV, HEAD_DIM)
    g = g.reshape(L, HQ, 1)
    q = rms_norm_p1(q, w["attention.q_norm.weight"])
    k = rms_norm_p1(k, w["attention.k_norm.weight"])
    qh = q.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    kh = k.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    vh = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    qh = apply_rope_partial(qh, cos, sin)
    kh = apply_rope_partial(kh, cos, sin)
    kh = repeat_kv(kh, REPEAT_KV)
    vh = repeat_kv(vh, REPEAT_KV)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    qf, kf, vf = qh.float(), kh.float(), vh.float()
    scores = (qf @ kf.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    attn = (probs @ vf).permute(0, 2, 1, 3).squeeze(0)
    gate = torch.sigmoid(g.float())
    attn = attn * gate
    attn_flat = attn.reshape(L, HQ * HEAD_DIM).to(torch.bfloat16)
    proj = (attn_flat.float() @ w["attention.linear_proj.weight"].float().t()).to(torch.bfloat16)
    h_after = (hb.float() + proj.float()).to(torch.bfloat16)
    mn = rms_norm_p1(h_after, w["mlp.pre_norm.weight"])
    up = (mn.float() @ w["mlp.up_gate_proj.weight"].float().t())
    act = swiglu7(up).to(torch.bfloat16)
    down = (act.float() @ w["mlp.down_proj.weight"].float().t()).to(torch.bfloat16)
    out = (h_after.float() + down.float())
    return out


def build_rope(S):
    num_bands = HEAD_DIM // 8
    exp = np.arange(0, num_bands, 1, dtype=np.float32) / num_bands
    bands = 1.0 / (10000.0 ** exp)
    coords = np.zeros((S, 3), dtype=np.float32)
    coords[:, 1] = np.arange(S, dtype=np.float32)
    sizes = np.full((S, 3), float(S), dtype=np.float32)
    refs = np.full((S, 3), float(S), dtype=np.float32)
    scales = (refs - 1.0) / (sizes - 1.0 + 1e-30)
    centers = (sizes - 1.0) * 0.5
    centers[:, 0] = 0.0
    cx = coords - centers
    proj = cx[:, :, None] * scales[:, :, None] * bands[None, None, :]
    cat = np.concatenate([np.sin(proj), np.cos(proj)], axis=1).reshape(S, 6 * num_bands)
    half = ROPE_DIM // 2
    return torch.from_numpy(cat[:, half:]).float(), torch.from_numpy(cat[:, :half]).float()


def main():
    torch.manual_seed(0)
    split_names = [
        "attention.pre_norm.weight", "attention.q_norm.weight",
        "attention.k_norm.weight",
        "attention.linear_q.weight", "attention.linear_k.weight",
        "attention.linear_v.weight", "attention.linear_g.weight",
        "attention.linear_proj.weight",
        "mlp.pre_norm.weight", "mlp.up_gate_proj.weight", "mlp.down_proj.weight",
    ]
    raw = load_keys(ST, [PREFIX + s for s in split_names])
    w = {k[len(PREFIX):]: v for k, v in raw.items()}

    cos, sin = build_rope(S)
    h_in = (torch.randn(S, HIDDEN) * 0.5).to(torch.bfloat16)
    out = sr_shared_forward(h_in, w, cos, sin)
    assert torch.isfinite(out).all(), "non-finite oracle output"
    print(f"SR oracle out: shape={tuple(out.shape)} mean={out.mean():.5f} std={out.std():.5f} "
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
