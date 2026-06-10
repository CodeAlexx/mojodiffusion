#!/usr/bin/env python3
"""LTX-2 joint-AV transformer block-0 BACKWARD parity oracle (trainer stage 1).

torch.autograd reference for the hand-chained Mojo backward of the FULL AV
block (serenitymojo/models/ltx2/ltx2_av_backward.mojo).

Block math = faithful F32 port of musubi-tuner BasicAVTransformerBlock._forward
  /home/alex/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/transformer.py
    _forward                      :466   (sublayer order + residual gating)
    get_ada_values                :195   (table[idx] + temb chunk)
    get_av_ca_ada_values          :245   (5-row a2v/v2a tables, 4 ss + 1 gate)
    _apply_text_cross_attention   :267   (rows 6:9 Q-mod + prompt KV-mod)
    apply_cross_attention_adaln   :853
The same math was already gate-proven against the Mojo inference spine
(scripts/ltx2_av_block0_parity.py -> pipeline/ltx2_av_block_parity_smoke.mojo,
video cos 0.9999943); this file reuses that exact forward, EXTENDED with:
  * REAL block-0 weights from the dequant-bf16 export (block-0 keys ONLY —
    never the whole 42 GB),
  * factorized LoRA y = Wx + b + scale*B(A(x)) on the 24 production targets
    (musubi LTX2_INCLUDE_PATTERNS_T2V): {to_q,to_k,to_v,to_out.0} x
    {attn1, attn2, audio_attn1, audio_attn2, audio_to_video_attn,
     video_to_audio_attn}.  A AND B both random nonzero (B=0 would produce no
    d_B... actually no d_A signal through B; both nonzero gives full-rank
    gradient signal on both factors),
  * torch.autograd grads from seeded d_video/d_audio cotangents:
    d_hs, d_ahs (stream input grads) + d_A/d_B for all 24 pairs.

musubi's FFN clamp(+-60000) is identity at these magnitudes (|ff|<<60000) and
the Mojo inference spine omits it; omitted here too (grad-identical in-range).

Dump -> output/ltx2_av_bwd/av_block0_bwd_ref.safetensors (all F32):
  inputs (same keys as the forward oracle), lora.<mod>.<proj>.A/.B,
  d_video, d_audio, video_out, audio_out (fwd cross-check),
  g_d_hidden, g_d_ahs, g_dA.<mod>.<proj>, g_dB.<mod>.<proj>.

Run:  /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_av_block_bwd_oracle.py
"""

import json
import math
import os
import struct
import sys

import torch
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_av_bwd"
OUT = os.path.join(OUT_DIR, "av_block0_bwd_ref.safetensors")

PREFIX = "model.diffusion_model.transformer_blocks.0."
EPS = 1e-6
ROPE_THETA = 10000.0

NUM_HEADS = 32
HEAD_DIM = 128
INNER_DIM = NUM_HEADS * HEAD_DIM                 # 4096
AUDIO_HEADS = 32
AUDIO_HEAD_DIM = 64
AUDIO_INNER_DIM = AUDIO_HEADS * AUDIO_HEAD_DIM   # 2048
AUDIO_CROSS_ATTN_DIM = 2048
AUDIO_SCALE_FACTOR = 4
POS_EMBED_MAX_POS = 20
CAUSAL_OFFSET = 0
VAE_SF = (8, 32, 32)
FRAME_RATE = 25.0

# Gate dims (brief): S_V=128, S_A=16, N_TXT=128, REAL head counts (32x128 video,
# 32x64 audio/cross-modal). Video grid 8x4x4 = 128 tokens.
NF, NH, NW = 8, 4, 4
S_V = NF * NH * NW          # 128
S_A = 16
N_TXT = 128
SEED = 20260610
INPUT_SCALE = 0.1

LORA_RANK = 16
LORA_SCALE = 0.5            # alpha/rank = 8/16
LORA_MODULES = ["attn1", "attn2", "audio_attn1", "audio_attn2",
                "audio_to_video_attn", "video_to_audio_attn"]
LORA_PROJS = ["to_q", "to_k", "to_v", "to_out.0"]

DEV = "cpu"


# ---------------------------------------------------------------------------
# safetensors partial load (block-0 keys only)
# ---------------------------------------------------------------------------
def _read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def load_block0(path):
    hdr, data_off = _read_header(path)
    out = {}
    with open(path, "rb") as f:
        for k, meta in hdr.items():
            if k == "__metadata__" or not k.startswith(PREFIX):
                continue
            dt = _DT[meta["dtype"]]
            s, e = meta["data_offsets"]
            f.seek(data_off + s)
            raw = f.read(e - s)
            t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta["shape"])
            # F32-table -> BF16 -> F32 round-trip to MATCH the Mojo loader
            # (LTX2AVBlockWeights.load uploads everything via from_view_as_bf16,
            # then .to_f32 for the gate). BF16 sources are exact either way.
            out[k[len(PREFIX):]] = (
                t.to(torch.bfloat16).to(torch.float32).to(DEV)
            )
    return out


# ---------------------------------------------------------------------------
# ops (identical to scripts/ltx2_av_block0_parity.py — autograd-transparent)
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


def compute_rope(coords, dim, max_positions, theta, num_heads, device):
    coords = coords.to(torch.float64)
    num_pos_dims = coords.shape[1]
    P = coords.shape[2]
    starts = coords[..., 0]
    ends = coords[..., 1]
    midpoints = (starts + ends) * 0.5

    grid = []
    for d in range(num_pos_dims):
        normed = midpoints[:, d, :] / max_positions[d]
        grid.append(normed.unsqueeze(2))
    grid = torch.cat(grid, dim=2)

    num_rope_elems = num_pos_dims * 2
    freq_count = dim // num_rope_elems
    denom = max(freq_count - 1, 1)
    i = torch.arange(freq_count, dtype=torch.float64, device=device)
    freq = (theta ** (i / denom)) * (math.pi / 2.0)

    grid_4d = grid.unsqueeze(3)
    scaled = grid_4d * 2.0 - 1.0
    angles = scaled * freq.view(1, 1, 1, freq_count)
    angles_t = angles.permute(0, 1, 3, 2)
    rope_freqs = freq_count * num_pos_dims
    angles_flat = angles_t.reshape(1, P, rope_freqs)

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
    hd = x.shape[-1]
    half = hd // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2], dim=-1)


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
    data = torch.zeros(1, 1, nframes, 2, dtype=torch.float32, device=device)
    mel_to_sec = 16000.0 / 160.0
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
# attention WITH factorized LoRA on to_q/to_k/to_v/to_out.0
# lora: dict "module.proj" -> (A [r,in], B [out,r], scale)
# ---------------------------------------------------------------------------
def _lora_add(y, x, lora, key):
    ent = lora.get(key)
    if ent is None:
        return y
    A, B, s = ent
    return y + s * ((x.to(torch.float32) @ A.t()) @ B.t())


def attention(w, prefix, hidden, kv, num_heads, head_dim, lora,
              q_rope=None, k_rope=None, eps=EPS):
    q = linear3d(hidden, w[prefix + ".to_q.weight"], w[prefix + ".to_q.bias"])
    q = _lora_add(q, hidden, lora, prefix + ".to_q")
    k = linear3d(kv, w[prefix + ".to_k.weight"], w[prefix + ".to_k.bias"])
    k = _lora_add(k, kv, lora, prefix + ".to_k")
    v = linear3d(kv, w[prefix + ".to_v.weight"], w[prefix + ".to_v.bias"])
    v = _lora_add(v, kv, lora, prefix + ".to_v")
    q = rms_norm(q, w[prefix + ".q_norm.weight"], eps)
    k = rms_norm(k, w[prefix + ".k_norm.weight"], eps)

    b, sq, _ = q.shape
    skv = k.shape[1]
    inner = num_heads * head_dim
    qh = q.reshape(b, sq, num_heads, head_dim)[0].permute(1, 0, 2)
    kh = k.reshape(b, skv, num_heads, head_dim)[0].permute(1, 0, 2)
    vh = v.reshape(b, skv, num_heads, head_dim)[0].permute(1, 0, 2)

    if q_rope is not None:
        qh = apply_rope(qh, q_rope[0], q_rope[1])
    krope = k_rope if k_rope is not None else q_rope
    if krope is not None:
        kh = apply_rope(kh, krope[0], krope[1])

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vh)
    out = out.permute(1, 0, 2).reshape(1, sq, inner)

    gw = w.get(prefix + ".to_gate_logits.weight")
    if gw is not None:
        gb = w.get(prefix + ".to_gate_logits.bias")
        gate_logits = linear3d(hidden, gw, gb)
        gates = torch.sigmoid(gate_logits) * 2.0
        out4 = out.reshape(1, sq, num_heads, head_dim)
        out = (out4 * gates.unsqueeze(3)).reshape(1, sq, inner)

    out_proj = linear3d(out, w[prefix + ".to_out.0.weight"],
                        w[prefix + ".to_out.0.bias"])
    out_proj = _lora_add(out_proj, out, lora, prefix + ".to_out.0")
    return out_proj


def fused_modulate(x, scale, shift):
    return x.to(torch.float32) * (scale.to(torch.float32) + 1.0) + shift.to(torch.float32)


def compute_ada6(table, temb, dim):
    N = temb.shape[1]
    t6 = table[:6].reshape(1, 1, 6, dim)
    temb6 = temb[..., :6 * dim].reshape(1, N, 6, dim)
    ada = t6 + temb6
    return [ada[:, :, i, :] for i in range(6)]


def compute_ada_ca(table, temb, dim):
    N = temb.shape[1]
    tca = table[6:9].reshape(1, 1, 3, dim)
    tembca = temb[..., 6 * dim:9 * dim].reshape(1, N, 3, dim)
    ada = tca + tembca
    return ada[:, :, 0, :], ada[:, :, 1, :], ada[:, :, 2, :]


def compute_cross_attn_params(v_table, a_table, temb_ca_ss, temb_ca_a_ss,
                              temb_ca_gate, temb_ca_a_gate, vdim, adim):
    v_ss = v_table[:4].reshape(1, 1, 4, vdim)
    v_gate = v_table[4:5].reshape(1, 1, 1, vdim)
    a_ss = a_table[:4].reshape(1, 1, 4, adim)
    a_gate = a_table[4:5].reshape(1, 1, 1, adim)

    v_comb = v_ss + temb_ca_ss.reshape(1, 1, 4, vdim)
    a2v_gate = (v_gate + temb_ca_gate.reshape(1, 1, 1, vdim))[:, :, 0, :]
    a_comb = a_ss + temb_ca_a_ss.reshape(1, 1, 4, adim)
    v2a_gate = (a_gate + temb_ca_a_gate.reshape(1, 1, 1, adim))[:, :, 0, :]

    return (a2v_gate, v2a_gate,
            (v_comb[:, :, 0, :], v_comb[:, :, 1, :]),
            (v_comb[:, :, 2, :], v_comb[:, :, 3, :]),
            (a_comb[:, :, 0, :], a_comb[:, :, 1, :]),
            (a_comb[:, :, 2, :], a_comb[:, :, 3, :]))


def kv_modulate(context, psst, prompt_ts, dim):
    seq = prompt_ts.shape[1]
    psst_bc = psst.reshape(1, 1, 2, dim)
    pt4 = prompt_ts.reshape(1, seq, 2, dim)
    combined = psst_bc + pt4
    shift_kv = combined[:, :, 0, :]
    scale_kv = combined[:, :, 1, :]
    return fused_modulate(context, scale_kv, shift_kv)


def synth(shape, seed, device, scale=INPUT_SCALE):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.float32).to(device)


# ---------------------------------------------------------------------------
# full block forward (musubi _forward order; video/audio independent until
# the cross-modal stage, so the spine's interleaved order is identical math)
# ---------------------------------------------------------------------------
def run_block(w, inp, lora):
    hs = inp["hs"]
    ahs = inp["ahs"]
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
    vpt = inp["video_prompt_ts"]
    apt = inp["audio_prompt_ts"]

    vdim, adim = INNER_DIM, AUDIO_INNER_DIM

    # 1. video self-attn (transformer.py:600-614)
    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = compute_ada6(
        w["scale_shift_table"], temb, vdim)
    mod_h = fused_modulate(rms_norm(hs, w.get("norm1.weight"), EPS), sc_msa, sh_msa)
    attn_out = attention(w, "attn1", mod_h, mod_h, NUM_HEADS, HEAD_DIM, lora,
                         q_rope=vrope)
    hs = hs + attn_out * g_msa

    # audio self-attn (transformer.py:633-649)
    a_sh_msa, a_sc_msa, a_g_msa, a_sh_mlp, a_sc_mlp, a_g_mlp = compute_ada6(
        w["audio_scale_shift_table"], a_temb, adim)
    mod_a = fused_modulate(rms_norm(ahs, w.get("audio_norm1.weight"), EPS),
                           a_sc_msa, a_sh_msa)
    attn_a = attention(w, "audio_attn1", mod_a, mod_a, AUDIO_HEADS,
                       AUDIO_HEAD_DIM, lora, q_rope=arope)
    ahs = ahs + attn_a * a_g_msa

    # 2. video text cross-attn (:616-631, _apply_text_cross_attention :267)
    v_sh_ca, v_sc_ca, v_g_ca = compute_ada_ca(w["scale_shift_table"], temb, vdim)
    mod_h2 = fused_modulate(rms_norm(hs, w.get("norm2.weight"), EPS), v_sc_ca, v_sh_ca)
    mv_ctx = kv_modulate(enc, w["prompt_scale_shift_table"], vpt, vdim)
    ca_out = attention(w, "attn2", mod_h2, mv_ctx, NUM_HEADS, HEAD_DIM, lora)
    hs = hs + ca_out * v_g_ca

    # audio text cross-attn (:651-668)
    a_sh_ca, a_sc_ca, a_g_ca = compute_ada_ca(w["audio_scale_shift_table"], a_temb, adim)
    mod_a2 = fused_modulate(rms_norm(ahs, w.get("audio_norm2.weight"), EPS),
                            a_sc_ca, a_sh_ca)
    ma_ctx = kv_modulate(aenc, w["audio_prompt_scale_shift_table"], apt, adim)
    ca_a_out = attention(w, "audio_attn2", mod_a2, ma_ctx, AUDIO_HEADS,
                         AUDIO_HEAD_DIM, lora)
    ahs = ahs + ca_a_out * a_g_ca

    # 3. cross-modal a2v / v2a (:672-750)
    norm_a2v = rms_norm(hs, w.get("audio_to_video_norm.weight"), EPS)
    norm_v2a = rms_norm(ahs, w.get("video_to_audio_norm.weight"), EPS)
    (a2v_gate, v2a_gate, v_a2v, v_v2a, a_a2v, a_v2a) = compute_cross_attn_params(
        w["scale_shift_table_a2v_ca_video"], w["scale_shift_table_a2v_ca_audio"],
        v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, vdim, adim)

    mod_video_a2v = norm_a2v * (v_a2v[0] + 1.0) + v_a2v[1]
    mod_audio_a2v = norm_v2a * (a_a2v[0] + 1.0) + a_a2v[1]
    a2v_out = attention(w, "audio_to_video_attn", mod_video_a2v, mod_audio_a2v,
                        AUDIO_HEADS, AUDIO_HEAD_DIM, lora,
                        q_rope=cavrope, k_rope=caarope)
    hs = hs + a2v_out * a2v_gate

    mod_video_v2a = norm_a2v * (v_v2a[0] + 1.0) + v_v2a[1]
    mod_audio_v2a = norm_v2a * (a_v2a[0] + 1.0) + a_v2a[1]
    v2a_out = attention(w, "video_to_audio_attn", mod_audio_v2a, mod_video_v2a,
                        AUDIO_HEADS, AUDIO_HEAD_DIM, lora,
                        q_rope=caarope, k_rope=cavrope)
    ahs = ahs + v2a_out * v2a_gate

    # 4. FFNs (:766-790); clamp omitted (identity in-range, spine has none)
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
    inp = {}
    inp["hs"] = synth((1, S_V, INNER_DIM), SEED + 0, device)
    inp["ahs"] = synth((1, S_A, AUDIO_INNER_DIM), SEED + 1, device)
    inp["enc_hs"] = synth((1, N_TXT, INNER_DIM), SEED + 2, device)
    inp["audio_enc_hs"] = synth((1, N_TXT, AUDIO_INNER_DIM), SEED + 3, device)
    inp["v_timestep"] = synth((1, S_V, 9 * INNER_DIM), SEED + 4, device)
    inp["a_timestep"] = synth((1, S_A, 9 * AUDIO_INNER_DIM), SEED + 5, device)
    inp["v_ca_ss"] = synth((1, 1, 4 * INNER_DIM), SEED + 6, device)
    inp["a_ca_ss"] = synth((1, 1, 4 * AUDIO_INNER_DIM), SEED + 7, device)
    inp["v_ca_gate"] = synth((1, 1, INNER_DIM), SEED + 8, device)
    inp["a_ca_gate"] = synth((1, 1, AUDIO_INNER_DIM), SEED + 9, device)
    inp["video_prompt_ts"] = synth((1, N_TXT, 2 * INNER_DIM), SEED + 10, device)
    inp["audio_prompt_ts"] = synth((1, N_TXT, 2 * AUDIO_INNER_DIM), SEED + 11, device)

    vcoords = build_video_coords(NF, NH, NW, VAE_SF, CAUSAL_OFFSET, FRAME_RATE, device)
    acoords = build_audio_coords(S_A, AUDIO_SCALE_FACTOR, CAUSAL_OFFSET, device)
    v_cos, v_sin = compute_rope(vcoords, INNER_DIM,
                                [POS_EMBED_MAX_POS, VAE_SF[1] * NH, VAE_SF[2] * NW],
                                ROPE_THETA, NUM_HEADS, device)
    a_cos, a_sin = compute_rope(acoords, AUDIO_INNER_DIM, [POS_EMBED_MAX_POS],
                                ROPE_THETA, AUDIO_HEADS, device)
    ca_v_cos, ca_v_sin = compute_rope(vcoords[:, 0:1, :, :], AUDIO_CROSS_ATTN_DIM,
                                      [POS_EMBED_MAX_POS], ROPE_THETA, AUDIO_HEADS, device)
    ca_a_cos, ca_a_sin = compute_rope(acoords[:, 0:1, :, :], AUDIO_CROSS_ATTN_DIM,
                                      [POS_EMBED_MAX_POS], ROPE_THETA, AUDIO_HEADS, device)
    inp["v_cos"], inp["v_sin"] = v_cos, v_sin
    inp["a_cos"], inp["a_sin"] = a_cos, a_sin
    inp["ca_v_cos"], inp["ca_v_sin"] = ca_v_cos, ca_v_sin
    inp["ca_a_cos"], inp["ca_a_sin"] = ca_a_cos, ca_a_sin
    return inp


def build_lora(w, device):
    """24 factorized adapters, A AND B random nonzero (B=0 kills d_A signal)."""
    lora = {}
    seed = SEED + 100
    for mod in LORA_MODULES:
        for proj in LORA_PROJS:
            wkey = f"{mod}.{proj}.weight"
            out_f, in_f = w[wkey].shape
            A = synth((LORA_RANK, in_f), seed, device,
                      scale=1.0 / math.sqrt(in_f))      # kaiming-ish
            B = synth((out_f, LORA_RANK), seed + 1, device, scale=0.02)
            seed += 2
            A.requires_grad_(True)
            B.requires_grad_(True)
            lora[f"{mod}.{proj}"] = (A, B, LORA_SCALE)
    return lora


def cos_sim(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[bwd-oracle] loading block-0 from {os.path.basename(CKPT)}")
    w = load_block0(CKPT)
    print(f"[bwd-oracle] {len(w)} block-0 tensors (bf16-roundtripped F32)")
    inp = build_inputs(DEV)
    lora = build_lora(w, DEV)
    print(f"[bwd-oracle] S_V={S_V} S_A={S_A} N_TXT={N_TXT} rank={LORA_RANK} "
          f"scale={LORA_SCALE} adapters={len(lora)}")

    hs_leaf = inp["hs"].clone().requires_grad_(True)
    ahs_leaf = inp["ahs"].clone().requires_grad_(True)
    fwd_inp = dict(inp)
    fwd_inp["hs"] = hs_leaf
    fwd_inp["ahs"] = ahs_leaf

    hs_out, ahs_out = run_block(w, fwd_inp, lora)
    print(f"[bwd-oracle] video_out mean={hs_out.mean():.5f} std={hs_out.std():.5f}")
    print(f"[bwd-oracle] audio_out mean={ahs_out.mean():.5f} std={ahs_out.std():.5f}")

    d_video = synth((1, S_V, INNER_DIM), SEED + 500, DEV)
    d_audio = synth((1, S_A, AUDIO_INNER_DIM), SEED + 501, DEV)
    loss = (hs_out * d_video).sum() + (ahs_out * d_audio).sum()

    leaves = [hs_leaf, ahs_leaf]
    names = ["g_d_hidden", "g_d_ahs"]
    for mod in LORA_MODULES:
        for proj in LORA_PROJS:
            A, B, _ = lora[f"{mod}.{proj}"]
            leaves += [A, B]
            names += [f"g_dA.{mod}.{proj}", f"g_dB.{mod}.{proj}"]
    grads = torch.autograd.grad(loss, leaves)

    out = {}
    for k, v in inp.items():
        out[k] = v.detach().to(torch.float32).cpu().contiguous()
    for mod in LORA_MODULES:
        for proj in LORA_PROJS:
            A, B, _ = lora[f"{mod}.{proj}"]
            out[f"lora.{mod}.{proj}.A"] = A.detach().to(torch.float32).cpu().contiguous()
            out[f"lora.{mod}.{proj}.B"] = B.detach().to(torch.float32).cpu().contiguous()
    out["d_video"] = d_video.cpu().contiguous()
    out["d_audio"] = d_audio.cpu().contiguous()
    out["video_out"] = hs_out.detach().to(torch.float32).cpu().contiguous()
    out["audio_out"] = ahs_out.detach().to(torch.float32).cpu().contiguous()
    for name, g in zip(names, grads):
        gn = float(g.norm())
        if gn == 0.0:
            raise RuntimeError(f"degenerate (zero) reference grad: {name}")
        out[name] = g.detach().to(torch.float32).cpu().contiguous()
    print(f"[bwd-oracle] d_hidden |g|={grads[0].norm():.5f} "
          f"d_ahs |g|={grads[1].norm():.5f}")

    if "--self" in sys.argv:
        hs2, ahs2 = run_block(w, fwd_inp, lora)
        print(f"[self] video cos={cos_sim(hs_out, hs2):.7f} "
              f"audio cos={cos_sim(ahs_out, ahs2):.7f}")
        return

    save_file(out, OUT)
    print(f"[bwd-oracle] dumped {len(out)} tensors -> {OUT}")


if __name__ == "__main__":
    main()
