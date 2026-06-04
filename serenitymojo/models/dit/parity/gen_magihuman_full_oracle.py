#!/usr/bin/env python3
# Oracle generator for the MagiHuman DiT FULL FORWARD (CHUNK B gate).
#
# Replicates the validated Rust port inference-flame/src/models/magihuman_dit.rs
# ::MagiHumanDiT::forward end-to-end:
#   adapter.embed (per-modality video/audio/text linears + bias)
#   adapter.rope_from_coords (ElementWiseFourier, REAL adapter.rope.bands tensor)
#   40 layers: MM (num_modality=3) for [0,1,2,3,36,37,38,39] (GELU7 for 0..3,
#              SwiGLU7 for 36..39); SHARED (num_modality=1, SwiGLU7) for 4..35.
#   final video/audio heads (mm_rms_norm_single + linear, audio padded to 192).
#
# Reads the REAL distill bf16 checkpoint. Runs in F32 on CPU (the 30.6 GB ckpt
# does not fit GPU here). Tokens sorted V then A then T; group_sizes=[V,A,T].
# Emits a small fixture safetensors consumed by magihuman_full_gate.mojo:
#   xv,xa,xt (adapter inputs), coords, expected (full forward output).
# Weights are NOT emitted — the Mojo gate reads them from the checkpoint mmap.

import json, struct
import numpy as np
import torch
from safetensors.torch import save_file

ST = "/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/magihuman_full_fixture.safetensors"

HIDDEN = 5120
HEAD_DIM = 128
HQ = 40
HKV = 8
GATING = HQ
Q_SIZE = HQ * HEAD_DIM       # 5120
KV_SIZE = HKV * HEAD_DIM     # 1024
QKV_OUT = Q_SIZE + 2 * KV_SIZE + GATING  # 7208
REPEAT_KV = HQ // HKV        # 5
ROPE_DIM = (HEAD_DIM // 8) * 2 * 3  # 96
ROPE_BANDS = HEAD_DIM // 8   # 16
EPS = 1e-6
NUM_LAYERS = 40
MM_LAYERS = {0, 1, 2, 3, 36, 37, 38, 39}
GELU7_LAYERS = {0, 1, 2, 3}
VIDEO_IN = 192
AUDIO_IN = 64
TEXT_IN = 3584

# Small but genuine multimodal layout (exercises all 3 groups + MM/shared/heads).
V, A, T = 64, 32, 32
GS = [V, A, T]
L = V + A + T  # 128


def _hdr():
    with open(ST, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


HEADER, BASE = _hdr()


def load(k):
    meta = HEADER[k]
    assert meta["dtype"] == "BF16", (k, meta["dtype"])
    a, b = meta["data_offsets"]
    with open(ST, "rb") as f:
        f.seek(BASE + a)
        raw = f.read(b - a)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(meta["shape"]).float()


def rms_norm_single(x, weight, eps=EPS):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    normed = xf * torch.rsqrt(var + eps)
    return normed * (weight.float() + 1.0)


def mm_rms_norm_multi(x, weight_full, gs, eps=EPS):
    # x: [.., last]; weight_full: [last*3]. Per-modality (weight_chunk+1) gain.
    last = x.shape[-1]
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    normed = xf * torch.rsqrt(var + eps)
    pieces = []
    off = 0
    for i in range(3):
        nrows = gs[i]
        if nrows == 0:
            continue
        chunk_x = normed[off:off + nrows]
        chunk_w = weight_full[i * last:(i + 1) * last].float()
        pieces.append(chunk_x * (chunk_w + 1.0))
        off += nrows
    return torch.cat(pieces, 0)


def mm_linear(x, weight_full, gs, out_per):
    # weight_full: [out_per*3, in]; per-modality x @ wᵀ.
    pieces = []
    off = 0
    for i in range(3):
        nrows = gs[i]
        if nrows == 0:
            continue
        chunk_x = x[off:off + nrows].float()
        chunk_w = weight_full[i * out_per:(i + 1) * out_per].float()
        pieces.append(chunk_x @ chunk_w.t())
        off += nrows
    return torch.cat(pieces, 0)


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
    return torch.cat([out_rot, x_pass.float()], dim=-1)


def swiglu7(x, alpha=1.702, limit=7.0):
    xf = x.float()
    x_glu = xf[..., 0::2].clamp(max=limit)
    x_linear = xf[..., 1::2].clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1.0)


def gelu7(x, alpha=1.702, limit=7.0):
    xf = x.float().clamp(max=limit)
    return xf * torch.sigmoid(alpha * xf)


def repeat_kv(x_bhsd, rep):
    B, H, Sq, D = x_bhsd.shape
    return x_bhsd.unsqueeze(2).expand(B, H, rep, Sq, D).reshape(B, H * rep, Sq, D)


def attn_mlp_core(h_in, prefix, cos, sin, is_mm, use_swiglu, gs):
    # bf16 round-trip matching the Mojo path. h_in: [L,hidden] f32.
    hb = h_in.to(torch.bfloat16).float()
    if is_mm:
        hn = mm_rms_norm_multi(hb, load(prefix + "attention.pre_norm.weight"), gs)
        qkv = mm_linear(hn.to(torch.bfloat16), load(prefix + "attention.linear_qkv.weight"), gs, QKV_OUT)
    else:
        hn = rms_norm_single(hb, load(prefix + "attention.pre_norm.weight"))
        w = load(prefix + "attention.linear_qkv.weight")
        qkv = (hn.to(torch.bfloat16).float() @ w.t())
    qkv = qkv.to(torch.bfloat16).float()
    q = qkv[:, :Q_SIZE]
    k = qkv[:, Q_SIZE:Q_SIZE + KV_SIZE]
    v = qkv[:, Q_SIZE + KV_SIZE:Q_SIZE + 2 * KV_SIZE]
    g = qkv[:, Q_SIZE + 2 * KV_SIZE:].reshape(L, HQ, 1)
    q = q.reshape(L, HQ, HEAD_DIM)
    k = k.reshape(L, HKV, HEAD_DIM)
    v = v.reshape(L, HKV, HEAD_DIM)
    if is_mm:
        q = mm_rms_norm_multi(q, load(prefix + "attention.q_norm.weight"), gs)
        k = mm_rms_norm_multi(k, load(prefix + "attention.k_norm.weight"), gs)
    else:
        q = rms_norm_single(q, load(prefix + "attention.q_norm.weight"))
        k = rms_norm_single(k, load(prefix + "attention.k_norm.weight"))
    qh = q.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    kh = k.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    vh = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    qh = apply_rope_partial(qh, cos, sin)
    kh = apply_rope_partial(kh, cos, sin)
    kh = repeat_kv(kh, REPEAT_KV)
    vh = repeat_kv(vh, REPEAT_KV)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    scores = (qh @ kh.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    attn = (probs @ vh).permute(0, 2, 1, 3).squeeze(0)  # [L,H,D]
    gate = torch.sigmoid(g.float())
    attn = attn * gate
    attn_flat = attn.reshape(L, HQ * HEAD_DIM).to(torch.bfloat16)
    if is_mm:
        proj = mm_linear(attn_flat, load(prefix + "attention.linear_proj.weight"), gs, HIDDEN)
    else:
        proj = (attn_flat.float() @ load(prefix + "attention.linear_proj.weight").t())
    proj = proj.to(torch.bfloat16).float()
    h_after = (hb + proj)
    h_after_bf = h_after.to(torch.bfloat16).float()
    # MLP
    if is_mm:
        mn = mm_rms_norm_multi(h_after_bf, load(prefix + "mlp.pre_norm.weight"), gs)
        up_w = load(prefix + "mlp.up_gate_proj.weight")
        up_per = up_w.shape[0] // 3
        up = mm_linear(mn.to(torch.bfloat16), up_w, gs, up_per)
    else:
        mn = rms_norm_single(h_after_bf, load(prefix + "mlp.pre_norm.weight"))
        up = (mn.to(torch.bfloat16).float() @ load(prefix + "mlp.up_gate_proj.weight").t())
    act = (swiglu7(up) if use_swiglu else gelu7(up)).to(torch.bfloat16)
    if is_mm:
        down = mm_linear(act, load(prefix + "mlp.down_proj.weight"), gs, HIDDEN)
    else:
        down = (act.float() @ load(prefix + "mlp.down_proj.weight").t())
    down = down.to(torch.bfloat16).float()
    return h_after + down  # f32 accumulate


def rope_from_coords(coords, bands):
    # coords: [L,9]; bands: [16]. Returns (cos_emb[L,48], sin_emb[L,48]).
    c = coords.astype(np.float64)
    xyz = c[:, 0:3]
    sizes = c[:, 3:6]
    refs = c[:, 6:9]
    scales = (refs - 1.0) / (sizes - 1.0 + 1e-30)
    centers = (sizes - 1.0) * 0.5
    centers[:, 0] = 0.0
    cx = (xyz - centers) * scales  # [L,3]
    bb = bands.astype(np.float64)  # [16]
    proj = cx[:, :, None] * bb[None, None, :]  # [L,3,16]
    sin_proj = np.sin(proj)
    cos_proj = np.cos(proj)
    cat = np.concatenate([sin_proj, cos_proj], axis=1)  # [L,6,16]
    rope = cat.reshape(L, 6 * ROPE_BANDS)  # [L,96]
    half = ROPE_DIM // 2
    sin_emb = rope[:, :half]
    cos_emb = rope[:, half:]
    return torch.from_numpy(cos_emb).float(), torch.from_numpy(sin_emb).float()


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Adapter inputs: per-modality raw features (already grouped V,A,T).
    xv = (torch.randn(V, VIDEO_IN) * 0.5)
    xa = (torch.randn(A, AUDIO_IN) * 0.5)
    xt = (torch.randn(T, TEXT_IN) * 0.5)

    # 1. Adapter embed (video/audio/text linears + bias) -> [L, hidden] f32
    pv = xv @ load("adapter.video_embedder.weight").t() + load("adapter.video_embedder.bias")
    pa = xa @ load("adapter.audio_embedder.weight").t() + load("adapter.audio_embedder.bias")
    pt = xt @ load("adapter.text_embedder.weight").t() + load("adapter.text_embedder.bias")
    h = torch.cat([pv, pa, pt], 0)  # [L, hidden]

    # Coords [L,9] = (t,h,w,T,H,W,refT,refH,refW). Synthetic but real-formula:
    # sweep h-axis within each modality group; sizes=refs=L (scale=1, time-center 0).
    coords = np.zeros((L, 9), dtype=np.float32)
    coords[:, 1] = np.arange(L, dtype=np.float32)  # h sweeps
    coords[:, 3:6] = float(L)
    coords[:, 6:9] = float(L)
    bands = load("adapter.rope.bands").numpy()  # REAL checkpoint bands [16]
    cos, sin = rope_from_coords(coords, bands)

    # 2. 40 layers
    h = h.to(torch.bfloat16).float()  # entry cast
    for i in range(NUM_LAYERS):
        prefix = f"block.layers.{i}."
        is_mm = i in MM_LAYERS
        use_swiglu = i not in GELU7_LAYERS  # 36..39 swiglu, 0..3 gelu, 4..35 swiglu
        h = attn_mlp_core(h, prefix, cos, sin, is_mm, use_swiglu, GS)
        print(f"  layer {i:2d} {'MM ' if is_mm else 'SH '}{'swiglu' if use_swiglu else 'gelu7 '} -> mean {h.mean():.4f} std {h.std():.4f}")

    # 3. Final heads
    out = torch.zeros(L, VIDEO_IN)
    xv_h = rms_norm_single(h[:V].to(torch.bfloat16).float(), load("final_norm_video.weight"))
    out[:V] = (xv_h.to(torch.bfloat16).float() @ load("final_linear_video.weight").t())
    xa_h = rms_norm_single(h[V:V + A].to(torch.bfloat16).float(), load("final_norm_audio.weight"))
    pa_out = (xa_h.to(torch.bfloat16).float() @ load("final_linear_audio.weight").t())  # [A,64]
    out[V:V + A, :AUDIO_IN] = pa_out
    # text rows stay zero

    assert torch.isfinite(out).all()
    print(f"full forward out: shape={tuple(out.shape)} mean={out.mean():.5f} std={out.std():.5f} "
          f"min={out.min():.4f} max={out.max():.4f}")

    fixture = {
        "xv": xv.float().contiguous(),
        "xa": xa.float().contiguous(),
        "xt": xt.float().contiguous(),
        "coords": torch.from_numpy(coords).float().contiguous(),
        "bands": torch.from_numpy(bands).float().contiguous(),
        "expected": out.float().contiguous(),
        "group_sizes": torch.tensor([V, A, T], dtype=torch.int32),
    }
    save_file(fixture, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
