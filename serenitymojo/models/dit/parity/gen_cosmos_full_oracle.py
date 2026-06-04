#!/usr/bin/env python3
# Full-forward (all 28 blocks) oracle for the serenitymojo cosmos_predict25_dit
# port, at a LARGE token count (N=8192) that the math-mode Dh=128 SDPA cannot
# fit in 24GB (proves the tiled SDPA path). Mirrors the Mojo `forward` EXACTLY
# (cosmos_predict25_dit.mojo:forward) with the REAL post-trained checkpoint and
# the SAME deterministic LCG inputs as pipeline/cosmos_dit_full_smoke.mojo, then
# writes a flat safetensors the Mojo full-forward driver compares against.
#
# The per-block math here is the same as gen_cosmos_block0_oracle.py (which
# self-checks at cos 0.999987 vs the real captured block_0_output); this file
# adds the LVG/padding-mask concat, crossattn_proj, patchify, x_embedder,
# timestep MLP, the 28-block loop, FinalLayer and cosmos unpatchify.
#
# bf16 sub-blocks, f32 residual, REAL flash/efficient attention for self-attn.

import struct, json, math, os
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

CKPT = "/home/alex/.cosmos-predict25/base/post-trained/cosmos_predict25_2b_dit.safetensors"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/cosmos_full_fixture.safetensors"

dev = "cuda"
D = 2048
H = 16
DH = 128
EPS = 1e-6
NUM_BLOCKS = 28
PS = 2          # patch_spatial
PT = 1          # patch_temporal
IN_CH = 16
OUT_CH = 16
CROSS_PROJ_IN = 100352
CROSS_IN = 1024

# ── resolution (matches the Mojo driver) ─────────────────────────────────────
TG = int(os.environ.get("COSMOS_TG", "2"))
HG = int(os.environ.get("COSMOS_HG", "128"))
WG = int(os.environ.get("COSMOS_WG", "128"))
TXTRAW = int(os.environ.get("COSMOS_TXT", "16"))
TIMESTEP = float(os.environ.get("COSMOS_T", "700.0"))

Tp = TG // PT
Hp = HG // PS
Wp = WG // PS
N = Tp * Hp * Wp
TXT = TXTRAW
print(f"[oracle] grid TG={TG} HG={HG} WG={WG} -> Tp={Tp} Hp={Hp} Wp={Wp} N={N} TXT={TXT}")


# ── load full checkpoint (only the keys we need) ─────────────────────────────
def st_header(path):
    with open(path, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        hdr = json.loads(fp.read(n))
    return hdr, 8 + n


def st_get(path, hdr, base, key):
    e = hdr[key]
    s, t = e["data_offsets"]
    with open(path, "rb") as fp:
        fp.seek(base + s)
        raw = fp.read(t - s)
    assert e["dtype"] == "BF16", (key, e["dtype"])
    arr = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()
    return arr.reshape(e["shape"])


hdr, base = st_header(CKPT)
W = {}


def load(key):
    if key not in W:
        W[key] = st_get(CKPT, hdr, base, key).to(dev)
    return W[key]


def lin(x, w, b=None):
    out = F.linear(x, w.to(x.dtype))
    if b is not None:
        out = out + b.to(x.dtype)
    return out


def rms(x, w):
    xf = x.float()
    n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (n * w.float()).to(x.dtype)


# ── deterministic LCG inputs, byte-identical to the Mojo driver ──────────────
# _rand4 in the driver: seed=99, scale 0.2, x_lat [IN_CH,TG,HG,WG]
def rand4(C, Fd, Hd, Wd, seed, scale):
    n = C * Fd * Hd * Wd
    out = torch.empty(n, dtype=torch.float32)
    s = seed
    for i in range(n):
        s = (s * 1103515245 + 12345) % 2147483648
        out[i] = (s / 2147483648.0 - 0.5) * scale
    return out.reshape(C, Fd, Hd, Wd)


def rand2(R, C, seed, scale):
    n = R * C
    out = torch.empty(n, dtype=torch.float32)
    s = seed
    for i in range(n):
        s = (s * 1103515245 + 12345) % 2147483648
        out[i] = (s / 2147483648.0 - 0.5) * scale
    return out.reshape(R, C)


x_lat = rand4(IN_CH, TG, HG, WG, 99, 0.2).to(dev).to(torch.bfloat16)   # [16,T,H,W]
text_raw = rand2(TXTRAW, CROSS_PROJ_IN, 7, 0.05).to(dev).to(torch.bfloat16)  # [16,100352]


# ── 1+2. LVG + padding-mask concat (image mode -> zeros) -> [18,T,H,W] ───────
zeros1 = torch.zeros(1, TG, HG, WG, device=dev, dtype=torch.bfloat16)
x_in = torch.cat([x_lat, zeros1, zeros1], dim=0)   # [18,T,H,W]

# ── 3. crossattn_proj 100352->1024 (Linear+bias) ─────────────────────────────
text_proj = lin(text_raw, load("crossattn_proj.0.weight"), load("crossattn_proj.0.bias"))
text_ctx = text_proj.to(torch.bfloat16)            # [TXT,1024]

# ── 4. patchify [18,T,H,W] -> [N, 18*PT*PS*PS=72], then x_embedder Linear ─────
# patchify3d order (matches ops/patchify3d): token = fi*Hp*Wp + hi*Wp + wi (F-major),
# within-patch (c, pf, ph, pw) c-SLOWEST.
C18 = x_in.shape[0]
# [C, Tp, PT, Hp, PS, Wp, PS]
xv = x_in.reshape(C18, Tp, PT, Hp, PS, Wp, PS)
# token order (Tp, Hp, Wp), within-patch (C, PT, PS, PS) c-slowest
patched = xv.permute(1, 3, 5, 0, 2, 4, 6).reshape(N, C18 * PT * PS * PS)  # [N,72]
x_seq = lin(patched, load("x_embedder.proj.1.weight"))  # [N, D] (no bias key used by mojo _lin_nobias_t)
# NOTE: mojo uses _lin_nobias_t (no bias) for x_embedder.proj.1.weight.

# ── 5. timestep conditioning ─────────────────────────────────────────────────
def timestep_embedding(t_vec, dim, max_period=10000.0):
    # cos-first: emb[:, :half]=cos(angle), emb[:, half:]=sin(angle)
    half = dim // 2
    i = torch.arange(half, device=t_vec.device, dtype=torch.float32)
    freq = torch.exp(-math.log(max_period) * (i / half))
    angle = t_vec[:, None].float() * freq[None, :]
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)  # [Tp,dim]


t_vec = torch.full((Tp,), TIMESTEP, device=dev, dtype=torch.float32)
sin_emb = timestep_embedding(t_vec, D)             # [Tp,D] f32
sample = sin_emb.to(torch.bfloat16)
h1 = F.silu(lin(sample, load("t_embedder.1.linear_1.weight")))
adaln_lora = lin(h1, load("t_embedder.1.linear_2.weight"))   # [Tp,3D] bf16
emb = rms(sample, load("t_embedding_norm.weight"))           # [Tp,D] bf16


# ── 6. RoPE tables (half-split, per-axis NTK theta) ──────────────────────────
def cosmos_rope(tp, hp, wp):
    dim_h = DH // 6 * 2          # 42
    dim_w = dim_h               # 42
    dim_t = DH - 2 * dim_h      # 44
    # per-axis theta = 10000 * ratio^(dim/(dim-2)); ratios t=1,h=3,w=3
    th_t = 10000.0 * (1.0 ** (dim_t / (dim_t - 2)))
    th_h = 10000.0 * (3.0 ** (dim_h / (dim_h - 2)))
    th_w = 10000.0 * (3.0 ** (dim_w / (dim_w - 2)))
    halves = [dim_t // 2, dim_h // 2, dim_w // 2]   # [22,21,21] sum=64
    thetas = [th_t, th_h, th_w]
    rows = tp * hp * wp
    # positions token-major (t,h,w)
    pos = torch.empty(rows, 3, dtype=torch.float32)
    idx = 0
    for ti in range(tp):
        for hi in range(hp):
            for wi in range(wp):
                pos[idx, 0] = ti
                pos[idx, 1] = hi
                pos[idx, 2] = wi
                idx += 1
    pos = pos.to(dev)
    cols = []
    for a in range(3):
        ha = halves[a]
        i = torch.arange(ha, device=dev, dtype=torch.float32)
        inv_freq = torch.exp(-math.log(thetas[a]) * (i / ha))
        angle = pos[:, a][:, None] * inv_freq[None, :]   # [rows, ha]
        cols.append(angle)
    angle_full = torch.cat(cols, dim=-1)                 # [rows, 64]
    return torch.cos(angle_full), torch.sin(angle_full)  # f32


cos_t, sin_t = cosmos_rope(Tp, Hp, Wp)   # [N,64] f32


def halfsplit_rope(x_nhd, cos_n, sin_n):
    # x: [N,H,DH]; cos/sin: [N,DH/2]. pair (x[d], x[d+half]).
    half = x_nhd.shape[-1] // 2
    c = cos_n[:, None, :].to(x_nhd.dtype)
    s = sin_n[:, None, :].to(x_nhd.dtype)
    x1 = x_nhd[..., :half]
    x2 = x_nhd[..., half:]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.cat([o1, o2], dim=-1)


def adaln_chunk(blk, emb, lora, sub):
    pre = blk + "adaln_modulation_" + sub
    h0 = F.silu(emb.to(torch.bfloat16))
    h1 = lin(h0, load(pre + ".1.weight"))
    h2 = lin(h1, load(pre + ".2.weight"))
    summed = (h2 + lora).float()
    return summed[..., :D], summed[..., D:2 * D], summed[..., 2 * D:]


def ln_mod_tokens(x_f32, sc_t, sh_t, tp, hpwp):
    # x_f32 [N,D]; sc_t/sh_t [Tp,D] broadcast over hpwp. LN no-affine eps=1e-6.
    xb = x_f32.to(torch.bfloat16).float()
    mu = xb.mean(-1, keepdim=True)
    var = xb.var(-1, keepdim=True, unbiased=False)
    ln = (xb - mu) / torch.sqrt(var + EPS)
    sc = sc_t.repeat_interleave(hpwp, dim=0)   # [N,D]
    sh = sh_t.repeat_interleave(hpwp, dim=0)
    return (ln * (1 + sc) + sh).to(torch.bfloat16)


# ── 7. 28 transformer blocks (f32 residual) ──────────────────────────────────
x_f32 = x_seq.float()   # [N,D]
hpwp = Hp * Wp
tx = text_ctx.reshape(TXT, CROSS_IN)

for bi in range(NUM_BLOCKS):
    blk = f"blocks.{bi}."

    # self-attn
    sh, sc, ga = adaln_chunk(blk, emb, adaln_lora, "self_attn")
    x_mod = ln_mod_tokens(x_f32, sc, sh, Tp, hpwp)   # [N,D] bf16
    q = lin(x_mod, load(blk + "self_attn.q_proj.weight")).reshape(N, H, DH)
    k = lin(x_mod, load(blk + "self_attn.k_proj.weight")).reshape(N, H, DH)
    v = lin(x_mod, load(blk + "self_attn.v_proj.weight")).reshape(N, H, DH)
    q = rms(q, load(blk + "self_attn.q_norm.weight"))
    k = rms(k, load(blk + "self_attn.k_norm.weight"))
    q = halfsplit_rope(q, cos_t, sin_t)
    k = halfsplit_rope(k, cos_t, sin_t)
    qh = q.permute(1, 0, 2)[None].bfloat16()   # [1,H,N,DH]
    kh = k.permute(1, 0, 2)[None].bfloat16()
    vh = v.permute(1, 0, 2)[None].bfloat16()
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        att = F.scaled_dot_product_attention(qh, kh, vh)[0]   # [H,N,DH]
    att = att.permute(1, 0, 2).reshape(N, H * DH)
    sa = lin(att, load(blk + "self_attn.output_proj.weight"))
    ga_n = ga.repeat_interleave(hpwp, dim=0)
    x_f32 = x_f32 + ga_n * sa.float()

    # cross-attn (text only, no rope, f32 attention to match mojo softmax path)
    sh, sc, ga = adaln_chunk(blk, emb, adaln_lora, "cross_attn")
    x_mod = ln_mod_tokens(x_f32, sc, sh, Tp, hpwp)
    q = lin(x_mod, load(blk + "cross_attn.q_proj.weight")).reshape(N, H, DH)
    k = lin(tx, load(blk + "cross_attn.k_proj.weight")).reshape(TXT, H, DH)
    v = lin(tx, load(blk + "cross_attn.v_proj.weight")).reshape(TXT, H, DH)
    q = rms(q, load(blk + "cross_attn.q_norm.weight"))
    k = rms(k, load(blk + "cross_attn.k_norm.weight"))
    qh = q.permute(1, 0, 2); kh = k.permute(1, 0, 2); vh = v.permute(1, 0, 2)
    att = F.scaled_dot_product_attention(qh.float(), kh.float(), vh.float()).to(torch.bfloat16)
    att = att.permute(1, 0, 2).reshape(N, H * DH)
    ca = lin(att, load(blk + "cross_attn.output_proj.weight"))
    ga_n = ga.repeat_interleave(hpwp, dim=0)
    x_f32 = x_f32 + ga_n * ca.float()

    # mlp
    sh, sc, ga = adaln_chunk(blk, emb, adaln_lora, "mlp")
    x_mod = ln_mod_tokens(x_f32, sc, sh, Tp, hpwp)
    mh = lin(x_mod, load(blk + "mlp.layer1.weight"))
    mh = F.gelu(mh)
    mo = lin(mh, load(blk + "mlp.layer2.weight"))
    ga_n = ga.repeat_interleave(hpwp, dim=0)
    x_f32 = x_f32 + ga_n * mo.float()
    if bi % 7 == 0:
        print(f"  block {bi} done; x abs-mean={x_f32.abs().mean().item():.4f}")

# ── 8. FinalLayer: LN_no_affine + 2-chunk adaln + Linear[D->64] ──────────────
x_final = x_f32.to(torch.bfloat16)
fh0 = F.silu(emb.to(torch.bfloat16))
fh1 = lin(fh0, load("final_layer.adaln_modulation.1.weight"))
fh2 = lin(fh1, load("final_layer.adaln_modulation.2.weight"))   # [Tp,2D]
adaln_2d = adaln_lora[:, :2 * D]
fsum = (fh2 + adaln_2d).float()
f_sh = fsum[:, :D]
f_sc = fsum[:, D:2 * D]
x_mod = ln_mod_tokens(x_final.float(), f_sc, f_sh, Tp, hpwp)
head_out = lin(x_mod, load("final_layer.linear.weight"))         # [N,64]

# ── 9. cosmos unpatchify [N, O] -> [OUT_CH, T, H, W]  (mojo cosmos_unpatchify) ─
# pt==1: reshape [N,O] -> [Tp,Hp,Wp,p1,p2,c] (c-fastest), permute axes [5,0,1,3,2,4]
# -> [c,Tp,Hp,p1,Wp,p2], reshape [out_c, Tp, Hp*ps, Wp*ps].
assert PT == 1
x6 = head_out.reshape(Tp, Hp, Wp, PS, PS, OUT_CH)        # 0=Tp 1=Hp 2=Wp 3=p1 4=p2 5=c
xp = x6.permute(5, 0, 1, 3, 2, 4)                        # [c,Tp,Hp,p1,Wp,p2]
out = xp.reshape(OUT_CH, Tp, Hp * PS, Wp * PS).contiguous()   # [OUT_CH, T, H, W]

print(f"[oracle] full forward done. out shape={list(out.shape)} abs-mean={out.float().abs().mean().item():.5f}")

save_file({
    "expected": out.reshape(OUT_CH, TG, HG, WG).float().contiguous().cpu(),
    "_meta": torch.tensor([N, Tp, Hp, Wp, TXT, OUT_CH, TG, HG, WG], dtype=torch.float32),
}, OUT)
print("wrote", OUT)
