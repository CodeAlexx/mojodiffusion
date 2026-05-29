#!/usr/bin/env python3
"""LTX-2 joint-AV transformer block-0 parity oracle (Plan P3).

Faithful Python port of the Rust dual-stream block forward
`LTX2TransformerBlock::forward_with_skip`
(inference-flame/src/models/ltx2_model.rs:1148-1465) for BLOCK 0, loading the
REAL block-0 weights from the distilled-fp8 checkpoint (block 0 is a *boundary*
block stored in BF16 — no FP8 dequant needed; verified: block-0 dtypes are
{BF16, F32}).

The block runs SIX attention paths:
  attn1            video self-attn   (Q/KV=video 4096, 32 heads x 128)
  audio_attn1      audio self-attn   (Q/KV=audio 2048, 32 heads x 64)
  attn2            video<->text      (Q=video, KV=video_context)
  audio_attn2      audio<->text      (Q=audio, KV=audio_context)
  audio_to_video   a2v cross-modal   (Q=video 4096, KV=audio 2048, to_out->4096)
  video_to_audio   v2a cross-modal   (Q=audio 2048, KV=video 4096, to_out->2048)

This oracle:
  1. generates DETERMINISTIC inputs (seeded) at a small shape (S_v=16, S_a=8,
     N_txt=12),
  2. computes the per-token timestep modulation tensors, the AV-cross-attn
     global modulation tensors, the prompt-timestep KV-modulation tensors, and
     all RoPE cos/sin tables internally (faithful ports of the model-level
     forward_audio_video_inner that wraps the block),
  3. runs the full block-0 forward,
  4. dumps EVERY tensor the Mojo smoke needs to ingest (inputs, contexts,
     modulation params, rope tables) PLUS the gate targets (video_out,
     audio_out), all as F32, to
     output/ltx2_av_block0/av_block0_ref.safetensors.

The Mojo smoke `serenitymojo/pipeline/ltx2_av_block_parity_smoke.mojo` loads
this file, runs the Mojo AV block forward on the SAME inputs, and gates
cosine_similarity >= 0.999 on BOTH video_out and audio_out.

Run:
  python3 scripts/ltx2_av_block0_parity.py            # dump ref tensors
  python3 scripts/ltx2_av_block0_parity.py --self      # self-check (python vs python re-run)

All block math runs in F32 (op-identical to the Rust BF16 path, strictly more
accurate) so the gate is not BF16-GEMM-jitter limited.
"""

import json
import math
import os
import struct
import sys

import torch
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_av_block0"
OUT = os.path.join(OUT_DIR, "av_block0_ref.safetensors")

PREFIX = "model.diffusion_model.transformer_blocks.0."
EPS = 1e-6
ROPE_THETA = 10000.0

# LTX2Config.default (ltx2_model.rs:97-133)
NUM_HEADS = 32
HEAD_DIM = 128
INNER_DIM = NUM_HEADS * HEAD_DIM            # 4096
AUDIO_HEADS = 32
AUDIO_HEAD_DIM = 64
AUDIO_INNER_DIM = AUDIO_HEADS * AUDIO_HEAD_DIM  # 2048
AUDIO_CROSS_ATTN_DIM = 2048
AUDIO_SCALE_FACTOR = 4
POS_EMBED_MAX_POS = 20
CAUSAL_OFFSET = 0
VAE_SF = (8, 32, 32)        # vae_scale_factors (temporal, h, w) — LTX2Config default
FRAME_RATE = 25.0
TIMESTEP_SCALE_MULTIPLIER = 1000.0
CROSS_ATTN_TS_SCALE_MULTIPLIER = 1000.0     # cross_attn_timestep_scale_multiplier default

# Smoke shape (small, deterministic). Video grid 4x2x2 = 16 tokens; audio 8 tok.
NF, NH, NW = 4, 2, 2
S_V = NF * NH * NW          # 16 video tokens
S_A = 8                     # audio tokens (frames)
N_TXT = 12                  # text-context length (video + audio share length here)
SEED = 20260528
INPUT_SCALE = 0.1
SIGMA = 0.7                 # single-sample timestep (sigma)

# Oracle math is deterministic F32; run on CPU (avoids a torch+CUDA FPE on the
# tiny BF16 frombuffer tensors, and the dumped tensors are framework-portable).
DEV = "cpu"


# ---------------------------------------------------------------------------
# safetensors partial load (header + selected tensors), -> torch on DEV
# ---------------------------------------------------------------------------
def _read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
       "F8_E4M3": torch.float8_e4m3fn}


def load_block0(path):
    hdr, data_off = _read_header(path)
    out = {}
    with open(path, "rb") as f:
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            if not k.startswith(PREFIX):
                continue
            dt = _DT[meta["dtype"]]
            s, e = meta["data_offsets"]
            f.seek(data_off + s)
            raw = f.read(e - s)
            t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta["shape"])
            # block 0 is BF16/F32 — cast everything to F32 for the oracle math.
            out[k[len(PREFIX):]] = t.to(torch.float32).to(DEV)
    return out


# ---------------------------------------------------------------------------
# ops (faithful F32 port of ltx2_model.rs)
# ---------------------------------------------------------------------------
def rms_norm(x, weight, eps):
    xf = x.to(torch.float32)
    ms = xf.mul(xf).mean(dim=-1, keepdim=True)
    normed = xf * torch.rsqrt(ms + eps)
    if weight is not None:
        normed = normed * weight.to(torch.float32)
    return normed


def gelu_approximate(x):
    return torch.nn.functional.gelu(x.to(torch.float32), approximate="tanh")


def linear3d(x, w, b):
    y = x.to(torch.float32) @ w.to(torch.float32).t()
    if b is not None:
        y = y + b.to(torch.float32)
    return y


# ---------------------------------------------------------------------------
# RoPE (faithful port of compute_rope_frequencies, ltx2_model.rs:373)
# ---------------------------------------------------------------------------
def compute_rope(coords, dim, max_positions, theta, num_heads, device):
    """coords: [1, num_pos_dims, P, 2] (start,end). Returns (cos,sin) each
    [num_heads, P, head_rope_dim] (B=1 dropped). F32."""
    coords = coords.to(torch.float64)
    num_pos_dims = coords.shape[1]
    P = coords.shape[2]
    starts = coords[..., 0]                 # [1, D, P]
    ends = coords[..., 1]
    midpoints = (starts + ends) * 0.5       # [1, D, P]

    grid = []
    for d in range(num_pos_dims):
        normed = midpoints[:, d, :] / max_positions[d]   # [1, P]
        grid.append(normed.unsqueeze(2))                 # [1, P, 1]
    grid = torch.cat(grid, dim=2)                        # [1, P, D]

    num_rope_elems = num_pos_dims * 2
    freq_count = dim // num_rope_elems
    denom = max(freq_count - 1, 1)
    i = torch.arange(freq_count, dtype=torch.float64, device=device)
    freq = (theta ** (i / denom)) * (math.pi / 2.0)      # [freq_count]

    grid_4d = grid.unsqueeze(3)                          # [1, P, D, 1]
    scaled = grid_4d * 2.0 - 1.0
    angles = scaled * freq.view(1, 1, 1, freq_count)     # [1, P, D, freq_count]
    angles_t = angles.permute(0, 1, 3, 2)                # [1, P, freq_count, D]
    rope_freqs = freq_count * num_pos_dims
    angles_flat = angles_t.reshape(1, P, rope_freqs)     # [1, P, rope_freqs]

    cos_raw = torch.cos(angles_flat)
    sin_raw = torch.sin(angles_flat)
    half_dim = dim // 2
    if rope_freqs < half_dim:
        pad = half_dim - rope_freqs
        cos_pad = torch.ones(1, P, pad, dtype=torch.float64, device=device)
        sin_pad = torch.zeros(1, P, pad, dtype=torch.float64, device=device)
        cos_out = torch.cat([cos_pad, cos_raw], dim=2)
        sin_out = torch.cat([sin_pad, sin_raw], dim=2)
    else:
        cos_out, sin_out = cos_raw, sin_raw

    head_rope_dim = half_dim // num_heads
    cos_h = cos_out.reshape(1, P, num_heads, head_rope_dim).permute(0, 2, 1, 3)
    sin_h = sin_out.reshape(1, P, num_heads, head_rope_dim).permute(0, 2, 1, 3)
    return cos_h[0].to(torch.float32), sin_h[0].to(torch.float32)  # [H, P, hrd]


def apply_rope(x, cos, sin):
    """x: [H, S, head_dim]. cos/sin: [H, S, head_dim/2]. Half-split.
       first = first*cos - second*sin ; second = second*cos + first*sin."""
    hd = x.shape[-1]
    half = hd // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2], dim=-1)


# ---------------------------------------------------------------------------
# coords (faithful ports of build_video_coords / build_audio_coords)
# ---------------------------------------------------------------------------
def build_video_coords(nf, nh, nw, vae_sf, causal_offset, frame_rate, device):
    P = nf * nh * nw
    data = torch.zeros(1, 3, P, 2, dtype=torch.float32, device=device)
    vae_t = float(vae_sf[0])
    for f in range(nf):
        for h in range(nh):
            for w in range(nw):
                tok = f * nh * nw + h * nw + w
                fs = f * vae_sf[0]
                fe = (f + 1) * vae_sf[0]
                fs_c = max(fs + causal_offset - vae_t, 0.0) / frame_rate
                fe_c = max(fe + causal_offset - vae_t, 0.0) / frame_rate
                data[0, 0, tok, 0] = fs_c
                data[0, 0, tok, 1] = fe_c
                data[0, 1, tok, 0] = h * vae_sf[1]
                data[0, 1, tok, 1] = (h + 1) * vae_sf[1]
                data[0, 2, tok, 0] = w * vae_sf[2]
                data[0, 2, tok, 1] = (w + 1) * vae_sf[2]
    return data


def build_audio_coords(nframes, scale_factor, causal_offset, device):
    P = nframes
    data = torch.zeros(1, 1, P, 2, dtype=torch.float32, device=device)
    mel_to_sec = 16000.0 / 160.0   # 100
    scale = float(scale_factor)
    for t in range(nframes):
        ms = t * scale
        me = (t + 1) * scale
        ms_c = max(ms + causal_offset - scale, 0.0) / mel_to_sec
        me_c = max(me + causal_offset - scale, 0.0) / mel_to_sec
        data[0, 0, t, 0] = ms_c
        data[0, 0, t, 1] = me_c
    return data


# ---------------------------------------------------------------------------
# attention (faithful F32 port of LTX2Attention::forward, ltx2_model.rs:739)
# Q/KV may differ in seq-len AND modality width. num_heads/head_dim are the
# *loaded* attn config (audio=32x64 for cross-modal). Per-head gate from Q-input.
# ---------------------------------------------------------------------------
def attention(w, prefix, hidden, kv, num_heads, head_dim,
              q_rope=None, k_rope=None, eps=EPS):
    q = linear3d(hidden, w[prefix + ".to_q.weight"], w[prefix + ".to_q.bias"])
    k = linear3d(kv, w[prefix + ".to_k.weight"], w[prefix + ".to_k.bias"])
    v = linear3d(kv, w[prefix + ".to_v.weight"], w[prefix + ".to_v.bias"])
    q = rms_norm(q, w[prefix + ".q_norm.weight"], eps)
    k = rms_norm(k, w[prefix + ".k_norm.weight"], eps)

    b, sq, _ = q.shape
    skv = k.shape[1]
    inner = num_heads * head_dim
    # [B, S, H, hd] -> [H, S, hd]  (B=1)
    qh = q.reshape(b, sq, num_heads, head_dim)[0].permute(1, 0, 2)
    kh = k.reshape(b, skv, num_heads, head_dim)[0].permute(1, 0, 2)
    vh = v.reshape(b, skv, num_heads, head_dim)[0].permute(1, 0, 2)

    if q_rope is not None:
        qh = apply_rope(qh, q_rope[0], q_rope[1])
    krope = k_rope if k_rope is not None else q_rope
    if krope is not None:
        kh = apply_rope(kh, krope[0], krope[1])

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale   # [H, Sq, Skv]
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vh)                              # [H, Sq, hd]
    out = out.permute(1, 0, 2).reshape(1, sq, inner)          # [1, Sq, inner]

    gw = w.get(prefix + ".to_gate_logits.weight")
    if gw is not None:
        gb = w.get(prefix + ".to_gate_logits.bias")
        gate_logits = linear3d(hidden, gw, gb)                # [1, Sq, H]
        gates = torch.sigmoid(gate_logits) * 2.0
        out4 = out.reshape(1, sq, num_heads, head_dim)
        out = (out4 * gates.unsqueeze(3)).reshape(1, sq, inner)

    out = linear3d(out, w[prefix + ".to_out.0.weight"], w[prefix + ".to_out.0.bias"])
    return out


def fused_modulate(x, scale, shift):
    # x * (1 + scale) + shift. scale/shift broadcast over tokens if [1,1,dim].
    return x.to(torch.float32) * (scale.to(torch.float32) + 1.0) + shift.to(torch.float32)


def compute_ada6(table, temb, dim):
    """table [>=6, dim], temb [1, N, 6*dim]. Returns 6x [1, N, dim]."""
    N = temb.shape[1]
    t6 = table[:6].reshape(1, 1, 6, dim)
    temb6 = temb[..., :6 * dim].reshape(1, N, 6, dim)
    ada = t6 + temb6
    return [ada[:, :, i, :] for i in range(6)]


def compute_ada_ca(table, temb, dim):
    """rows 6-8 of [9,dim] + last 3*dim of temb. Returns shift,scale,gate."""
    N = temb.shape[1]
    tca = table[6:9].reshape(1, 1, 3, dim)
    tembca = temb[..., 6 * dim:9 * dim].reshape(1, N, 3, dim)
    ada = tca + tembca
    return ada[:, :, 0, :], ada[:, :, 1, :], ada[:, :, 2, :]


def compute_cross_attn_params(v_table, a_table, temb_ca_ss, temb_ca_a_ss,
                              temb_ca_gate, temb_ca_a_gate, vdim, adim):
    """v_table/a_table: [5, dim]. temb_ca_ss: [1,1,4*dim], gate [1,1,dim].
       Returns (a2v_gate, v2a_gate, v_a2v(sc,sh), v_v2a(sc,sh),
                a_a2v(sc,sh), a_v2a(sc,sh)). All [1,1,dim] (broadcast)."""
    v_ss = v_table[:4].reshape(1, 1, 4, vdim)
    v_gate = v_table[4:5].reshape(1, 1, 1, vdim)
    a_ss = a_table[:4].reshape(1, 1, 4, adim)
    a_gate = a_table[4:5].reshape(1, 1, 1, adim)

    v_comb = v_ss + temb_ca_ss.reshape(1, 1, 4, vdim)
    video_a2v_scale = v_comb[:, :, 0, :]
    video_a2v_shift = v_comb[:, :, 1, :]
    video_v2a_scale = v_comb[:, :, 2, :]
    video_v2a_shift = v_comb[:, :, 3, :]
    a2v_gate = (v_gate + temb_ca_gate.reshape(1, 1, 1, vdim))[:, :, 0, :]

    a_comb = a_ss + temb_ca_a_ss.reshape(1, 1, 4, adim)
    audio_a2v_scale = a_comb[:, :, 0, :]
    audio_a2v_shift = a_comb[:, :, 1, :]
    audio_v2a_scale = a_comb[:, :, 2, :]
    audio_v2a_shift = a_comb[:, :, 3, :]
    v2a_gate = (a_gate + temb_ca_a_gate.reshape(1, 1, 1, adim))[:, :, 0, :]

    return (a2v_gate, v2a_gate,
            (video_a2v_scale, video_a2v_shift),
            (video_v2a_scale, video_v2a_shift),
            (audio_a2v_scale, audio_a2v_shift),
            (audio_v2a_scale, audio_v2a_shift))


def kv_modulate(context, psst, prompt_ts, dim):
    """context*(1+scale_kv)+shift_kv where combined = psst[2,dim]+prompt_ts.
       psst: [2,dim]; prompt_ts: [1, seq, 2*dim]."""
    seq = prompt_ts.shape[1]
    psst_bc = psst.reshape(1, 1, 2, dim)
    pt4 = prompt_ts.reshape(1, seq, 2, dim)
    combined = psst_bc + pt4
    shift_kv = combined[:, :, 0, :]
    scale_kv = combined[:, :, 1, :]
    return fused_modulate(context, scale_kv, shift_kv)


# ---------------------------------------------------------------------------
# AdaLayerNormSingle (timestep_embedder + linear) — faithful port.
# sinusoidal embed -> SiLU(linear_1) -> linear_2 produces num_mod_params*dim.
# We load the per-block prompt/ca adaln only via tables; the per-TOKEN video/
# audio timestep modulation (temb) and the AV-cross global modulation are
# produced by the model-level AdaLayerNormSingle layers, which live OUTSIDE the
# block. To keep this oracle self-contained AND faithful, we synthesize those
# temb tensors deterministically: the block forward is exercised with
# *arbitrary but reproducible* temb tensors (the block math doesn't depend on
# how temb was produced — it just adds scale_shift_table + temb). The Mojo smoke
# ingests the SAME temb tensors dumped here. This isolates the BLOCK forward
# (the P3 deliverable) from the model-level timestep MLP (a separate phase).
# ---------------------------------------------------------------------------
def synth(shape, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, generator=g) * INPUT_SCALE).to(torch.float32).to(device)


def run_block(w, inp, device, dump=None):  # noqa: ARG001 (dump used below)
    hs = inp["hs"].clone()
    ahs = inp["ahs"].clone()
    enc = inp["enc_hs"]
    aenc = inp["audio_enc_hs"]
    temb = inp["v_timestep"]
    a_temb = inp["a_timestep"]
    v_ca_ss = inp["v_ca_ss"]
    a_ca_ss = inp["a_ca_ss"]
    v_ca_gate = inp["v_ca_gate"]
    a_ca_gate = inp["a_ca_gate"]
    vrope = (inp["v_cos"], inp["v_sin"])
    arope = (inp["a_cos"], inp["a_sin"])
    cavrope = (inp["ca_v_cos"], inp["ca_v_sin"])
    caarope = (inp["ca_a_cos"], inp["ca_a_sin"])
    vpt = inp.get("video_prompt_ts")
    apt = inp.get("audio_prompt_ts")

    vdim, adim = INNER_DIM, AUDIO_INNER_DIM

    # ---- 1. Video self-attn ----
    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = compute_ada6(
        w["scale_shift_table"], temb, vdim)
    mod_h = fused_modulate(rms_norm(hs, w.get("norm1.weight"), EPS), sc_msa, sh_msa)
    attn_out = attention(w, "attn1", mod_h, mod_h, NUM_HEADS, HEAD_DIM,
                         q_rope=vrope)
    hs = hs + attn_out * g_msa

    # ---- Audio self-attn ----
    a_sh_msa, a_sc_msa, a_g_msa, a_sh_mlp, a_sc_mlp, a_g_mlp = compute_ada6(
        w["audio_scale_shift_table"], a_temb, adim)
    mod_a = fused_modulate(rms_norm(ahs, w.get("audio_norm1.weight"), EPS),
                           a_sc_msa, a_sh_msa)
    attn_a = attention(w, "audio_attn1", mod_a, mod_a, AUDIO_HEADS, AUDIO_HEAD_DIM,
                       q_rope=arope)
    ahs = ahs + attn_a * a_g_msa

    # ---- 2. Video cross-attn (text) ----
    v_sh_ca, v_sc_ca, v_g_ca = compute_ada_ca(w["scale_shift_table"], temb, vdim)
    mod_h2 = fused_modulate(rms_norm(hs, w.get("norm2.weight"), EPS), v_sc_ca, v_sh_ca)
    if w.get("prompt_scale_shift_table") is not None and vpt is not None:
        mv_ctx = kv_modulate(enc, w["prompt_scale_shift_table"], vpt, vdim)
    else:
        mv_ctx = enc
    ca_out = attention(w, "attn2", mod_h2, mv_ctx, NUM_HEADS, HEAD_DIM)
    hs = hs + ca_out * v_g_ca

    # ---- Audio cross-attn (text) ----
    a_sh_ca, a_sc_ca, a_g_ca = compute_ada_ca(w["audio_scale_shift_table"], a_temb, adim)
    mod_a2 = fused_modulate(rms_norm(ahs, w.get("audio_norm2.weight"), EPS),
                            a_sc_ca, a_sh_ca)
    if w.get("audio_prompt_scale_shift_table") is not None and apt is not None:
        ma_ctx = kv_modulate(aenc, w["audio_prompt_scale_shift_table"], apt, adim)
    else:
        ma_ctx = aenc
    ca_a_out = attention(w, "audio_attn2", mod_a2, ma_ctx, AUDIO_HEADS, AUDIO_HEAD_DIM)
    ahs = ahs + ca_a_out * a_g_ca

    # ---- 3. A2V / V2A cross-modal ----
    norm_a2v = rms_norm(hs, w.get("audio_to_video_norm.weight"), EPS)   # video stream
    norm_v2a = rms_norm(ahs, w.get("video_to_audio_norm.weight"), EPS)  # audio stream
    (a2v_gate, v2a_gate, v_a2v, v_v2a, a_a2v, a_v2a) = compute_cross_attn_params(
        w["scale_shift_table_a2v_ca_video"], w["scale_shift_table_a2v_ca_audio"],
        v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, vdim, adim)

    # A2V: Q=video (mod by v_a2v), KV=audio (mod by a_a2v). to_out -> 4096.
    mod_video_a2v = norm_a2v * (v_a2v[0] + 1.0) + v_a2v[1]
    mod_audio_a2v = norm_v2a * (a_a2v[0] + 1.0) + a_a2v[1]
    a2v_out = attention(w, "audio_to_video_attn", mod_video_a2v, mod_audio_a2v,
                        AUDIO_HEADS, AUDIO_HEAD_DIM,
                        q_rope=cavrope, k_rope=caarope)
    hs = hs + a2v_out * a2v_gate

    # V2A: Q=audio (mod by a_v2a), KV=video (mod by v_v2a). to_out -> 2048.
    mod_video_v2a = norm_a2v * (v_v2a[0] + 1.0) + v_v2a[1]
    mod_audio_v2a = norm_v2a * (a_v2a[0] + 1.0) + a_v2a[1]
    ahs_pre_v2a = ahs.clone()   # snapshot before v2a residual (for delta gate)
    v2a_out = attention(w, "video_to_audio_attn", mod_audio_v2a, mod_video_v2a,
                        AUDIO_HEADS, AUDIO_HEAD_DIM,
                        q_rope=caarope, k_rope=cavrope)
    v2a_delta = v2a_out * v2a_gate   # the exact addend applied to the audio stream
    ahs = ahs + v2a_delta
    if dump is not None:
        dump["ahs_pre_v2a"] = ahs_pre_v2a.detach().to(torch.float32).cpu().contiguous()
        dump["v2a_delta"] = v2a_delta.detach().to(torch.float32).cpu().contiguous()
        dump["v2a_raw_out"] = v2a_out.detach().to(torch.float32).cpu().contiguous()

    # ---- 4. FFN ----
    mod_ff = fused_modulate(rms_norm(hs, w.get("norm3.weight"), EPS), sc_mlp, sh_mlp)
    ff_out = linear3d(gelu_approximate(linear3d(mod_ff, w["ff.net.0.proj.weight"],
                                                w["ff.net.0.proj.bias"])),
                      w["ff.net.2.weight"], w["ff.net.2.bias"])
    hs = hs + ff_out * g_mlp

    mod_aff = fused_modulate(rms_norm(ahs, w.get("audio_norm3.weight"), EPS),
                             a_sc_mlp, a_sh_mlp)
    aff_out = linear3d(gelu_approximate(linear3d(mod_aff, w["audio_ff.net.0.proj.weight"],
                                                 w["audio_ff.net.0.proj.bias"])),
                       w["audio_ff.net.2.weight"], w["audio_ff.net.2.bias"])
    ahs = ahs + aff_out * a_g_mlp

    return hs, ahs


def build_inputs(device):
    """Deterministic block-0 inputs + all precomputed modulation/rope tensors."""
    inp = {}
    inp["hs"] = synth((1, S_V, INNER_DIM), SEED + 0, device)
    inp["ahs"] = synth((1, S_A, AUDIO_INNER_DIM), SEED + 1, device)
    inp["enc_hs"] = synth((1, N_TXT, INNER_DIM), SEED + 2, device)
    inp["audio_enc_hs"] = synth((1, N_TXT, AUDIO_INNER_DIM), SEED + 3, device)
    # per-token timestep modulation (9-param for video; 6-param used + 3 ca)
    inp["v_timestep"] = synth((1, S_V, 9 * INNER_DIM), SEED + 4, device)
    inp["a_timestep"] = synth((1, S_A, 9 * AUDIO_INNER_DIM), SEED + 5, device)
    # AV-cross global modulation (broadcast over tokens) [1,1,*]
    inp["v_ca_ss"] = synth((1, 1, 4 * INNER_DIM), SEED + 6, device)
    inp["a_ca_ss"] = synth((1, 1, 4 * AUDIO_INNER_DIM), SEED + 7, device)
    inp["v_ca_gate"] = synth((1, 1, INNER_DIM), SEED + 8, device)
    inp["a_ca_gate"] = synth((1, 1, AUDIO_INNER_DIM), SEED + 9, device)
    # prompt-timestep KV modulation [1, N_TXT, 2*dim]
    inp["video_prompt_ts"] = synth((1, N_TXT, 2 * INNER_DIM), SEED + 10, device)
    inp["audio_prompt_ts"] = synth((1, N_TXT, 2 * AUDIO_INNER_DIM), SEED + 11, device)

    # RoPE tables — computed faithfully from coords (the real model path).
    vcoords = build_video_coords(NF, NH, NW, VAE_SF, CAUSAL_OFFSET, FRAME_RATE, device)
    acoords = build_audio_coords(S_A, AUDIO_SCALE_FACTOR, CAUSAL_OFFSET, device)
    v_cos, v_sin = compute_rope(vcoords, INNER_DIM, [POS_EMBED_MAX_POS,
                                VAE_SF[1] * NH, VAE_SF[2] * NW], ROPE_THETA,
                                NUM_HEADS, device)
    # NOTE video max_pos uses base_height/base_width in the model; we use the
    # grid extent here (the block forward consumes the dumped table verbatim, so
    # the exact max_pos only needs to MATCH between this dump and itself).
    a_cos, a_sin = compute_rope(acoords, AUDIO_INNER_DIM, [POS_EMBED_MAX_POS],
                                ROPE_THETA, AUDIO_HEADS, device)
    # cross-attn rope: temporal-only, dim = audio_cross_attention_dim, heads=audio.
    v_tcoords = vcoords[:, 0:1, :, :]    # [1,1,Sv,2]
    a_tcoords = acoords[:, 0:1, :, :]    # [1,1,Sa,2]
    ca_v_cos, ca_v_sin = compute_rope(v_tcoords, AUDIO_CROSS_ATTN_DIM,
                                      [POS_EMBED_MAX_POS], ROPE_THETA, AUDIO_HEADS, device)
    ca_a_cos, ca_a_sin = compute_rope(a_tcoords, AUDIO_CROSS_ATTN_DIM,
                                      [POS_EMBED_MAX_POS], ROPE_THETA, AUDIO_HEADS, device)
    inp["v_cos"], inp["v_sin"] = v_cos, v_sin
    inp["a_cos"], inp["a_sin"] = a_cos, a_sin
    inp["ca_v_cos"], inp["ca_v_sin"] = ca_v_cos, ca_v_sin
    inp["ca_a_cos"], inp["ca_a_sin"] = ca_a_cos, ca_a_sin
    return inp


def cos_sim(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[oracle] loading block-0 weights from {os.path.basename(CKPT)}")
    w = load_block0(CKPT)
    print(f"[oracle] loaded {len(w)} block-0 tensors")
    inp = build_inputs(DEV)
    print(f"[oracle] shapes: S_V={S_V} S_A={S_A} N_TXT={N_TXT}")

    # Prepare the output dict early so run_block can stash v2a intermediates.
    out = {}
    hs_out, ahs_out = run_block(w, inp, DEV, dump=out)
    print(f"[oracle] video_out {tuple(hs_out.shape)} "
          f"mean={hs_out.mean():.4f} std={hs_out.std():.4f} "
          f"absmax={hs_out.abs().max():.4f}")
    print(f"[oracle] audio_out {tuple(ahs_out.shape)} "
          f"mean={ahs_out.mean():.4f} std={ahs_out.std():.4f} "
          f"absmax={ahs_out.abs().max():.4f}")
    if "v2a_delta" in out:
        d = out["v2a_delta"]
        print(f"[oracle] v2a_delta {tuple(d.shape)} "
              f"mean={d.mean():.6f} std={d.std():.6f} "
              f"absmax={d.abs().max():.6f}")

    if "--self" in sys.argv:
        hs2, ahs2 = run_block(w, inp, DEV)
        print(f"[self-check] video cos={cos_sim(hs_out, hs2):.6f} "
              f"audio cos={cos_sim(ahs_out, ahs2):.6f}")
        return

    # Dump everything the Mojo smoke ingests + gate targets, all F32, CPU.
    for k, v in inp.items():
        out[k] = v.detach().to(torch.float32).cpu().contiguous()
    out["video_out"] = hs_out.detach().to(torch.float32).cpu().contiguous()
    out["audio_out"] = ahs_out.detach().to(torch.float32).cpu().contiguous()
    save_file(out, OUT)
    print(f"[oracle] dumped {len(out)} tensors -> {OUT}")


if __name__ == "__main__":
    main()
