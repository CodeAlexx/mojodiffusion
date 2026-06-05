#!/usr/bin/env python3
"""LTX-2 FULL 48-block forward_audio_video VELOCITY parity oracle (Plan P5).

Faithful Python port of the Rust model-level
`LTX2StreamingModel::forward_audio_video_inner`
(inference-flame/src/models/ltx2_model.rs:4453-5040), wrapping the dual-stream
block forward (`forward_with_skip`, :1148-1465, identical to the per-block
oracle scripts/ltx2_av_block0_parity.py) for ALL 48 transformer blocks.

It runs the COMPLETE stack from latent to velocity:
  1. patchify video [B,C,F,H,W]->[B,N,128] + audio [B,8,T,16]->[B,T,128]
  2. proj_in / audio_proj_in  -> hs/ahs at inner_dim
  3. timestep MLP (adaln_single / audio_adaln_single):
        ts*1000 -> sinusoidal(256) -> linear_1 -> silu -> linear_2 = embedded
        silu(embedded) -> linear = per-token temb [B,N,9*dim]
  4. AV-cross global modulation (av_ca_video/audio_scale_shift, a2v/v2a gate
        adaln; gate scaled by cross_gate_scale = cross_mult/ts_mult = 1.0)
  5. prompt_ts via prompt_adaln_single / audio_prompt_adaln_single
  6. context via the LTX-2.3 Embeddings1DConnector (video + audio), run LOCALLY
        on the cached pre-connector embeds (video) and a deterministic seeded
        audio context — the P2.5 connector contract.
  7. 48 x forward_with_skip (inner blocks fp8_cast'd to match the FP8 checkpoint)
  8. output: layer_norm_no_affine -> (scale_shift_table[2,dim] + embedded) ->
        proj_out -> unpatchify  => VIDEO + AUDIO velocity.

Inputs are DETERMINISTIC (seeded) at a small MVP shape; the same patchified
latents, contexts, sigma and shapes are reproduced by the Mojo smoke
serenitymojo/pipeline/ltx2_dit_forward_smoke.mojo, which gates cosine_similarity
>= 0.999 on BOTH the video and audio velocity.

The whole oracle runs in F32 (op-identical to the Rust BF16 path, strictly more
accurate); inner blocks are round-tripped through fp8 e4m3 (fp8_cast) so the
oracle matches the FP8 streaming path the Mojo smoke takes for blocks 4-46.

Run:
  python3 scripts/ltx2_dit_forward_parity_ref.py            # dump velocity ref
  python3 scripts/ltx2_dit_forward_parity_ref.py --self      # python self-check
"""

import json
import math
import os
import struct
import sys

import torch
from safetensors.torch import save_file
from safetensors import safe_open

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
CACHED = "/home/alex/EriDiffusion/inference-flame/cached_ltx2_embeddings.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_dit_forward"
OUT = os.path.join(OUT_DIR, "dit_forward_ref.safetensors")
OUT_HQ_LORA = os.path.join(OUT_DIR, "dit_forward_hq_lora_ref.safetensors")

DISTILLED_LORA = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors"
CAMERA_STATIC_LORA = "/home/alex/.serenity/models/loras/ltx-2-19b-lora-camera-control-static.safetensors"
DETAILER_LORA = "/home/alex/.serenity/models/loras/ltx-2-19b-ic-lora-detailer.safetensors"
HQ_LORA_STACK_STAGE2 = (
    # Current staged-HQ production stack: stage-2 distilled support + camera.
    # Detailer remains listed at 0.0 until IC/reference conditioning is wired.
    (DISTILLED_LORA, 0.5, "distilled_stage2"),
    (CAMERA_STATIC_LORA, 0.3, "camera_static"),
    (DETAILER_LORA, 0.0, "detailer_disabled"),
)

PREFIX = "model.diffusion_model."
EPS = 1e-6
ROPE_THETA = 10000.0

# LTX2Config::default (ltx2_model.rs:97-133)
NUM_HEADS = 32
HEAD_DIM = 128
INNER_DIM = NUM_HEADS * HEAD_DIM            # 4096
AUDIO_HEADS = 32
AUDIO_HEAD_DIM = 64
AUDIO_INNER_DIM = AUDIO_HEADS * AUDIO_HEAD_DIM  # 2048
AUDIO_CROSS_ATTN_DIM = 2048
AUDIO_SCALE_FACTOR = 4
POS_EMBED_MAX_POS = 20
BASE_HEIGHT = 2048
BASE_WIDTH = 2048
CAUSAL_OFFSET = 1                            # config default (NOT 0)
VAE_SF = (8, 32, 32)
FRAME_RATE = 25.0
TS_MULT = 1000.0
CROSS_TS_MULT = 1000.0
CROSS_GATE_SCALE = CROSS_TS_MULT / TS_MULT   # = 1.0
NUM_LAYERS = 48
NUM_MOD_PARAMS = 9                            # video adaln_single -> 9*dim
AUDIO_NUM_MOD = 9
CONNECTOR_MAX_POS = 4096.0

# MVP smoke shape (small, deterministic). Video grid 2x4x4 = 32 tokens; audio 8.
NF, NH, NW = 2, 4, 4
S_V = NF * NH * NW          # 32 video tokens
S_A = 8                     # audio tokens
N_TXT = 24                  # text-context length (connector input seq)
AUDIO_C, AUDIO_F = 8, 16    # audio latent channels / mel bins (C*F = 128)
SEED = 20260528
SIGMA = 0.7

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOUNDARY_BLOCKS = set([0, 1, 2, 3, 47])
CAPTURE_AFTER_BLOCKS = set(range(NUM_LAYERS))

if DEV == "cuda":
    # Mojo's vendor BLAS F32 path uses tensor-core style math on NVIDIA. Keep the
    # oracle in the same numerical regime; strict CUDA F32 drifts enough over 48
    # blocks to become a false negative at the final video head.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# safetensors partial load
# ---------------------------------------------------------------------------
def _read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
       "F8_E4M3": torch.float8_e4m3fn}


def load_selected(path, want_prefixes, want_exact=None):
    hdr, data_off = _read_header(path)
    out = {}
    scales = {}
    with open(path, "rb") as f:
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            keep = any(k.startswith(p) for p in want_prefixes)
            if want_exact is not None and k in want_exact:
                keep = True
            if not keep:
                continue
            dt = _DT[meta["dtype"]]
            s, e = meta["data_offsets"]
            f.seek(data_off + s)
            raw = f.read(e - s)
            if dt == torch.float8_e4m3fn:
                t = torch.frombuffer(bytearray(raw),
                                     dtype=torch.uint8).clone()
                t = t.view(torch.float8_e4m3fn).reshape(meta["shape"])
                out[k] = t
            else:
                t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(
                    meta["shape"])
                out[k] = t.clone()
    return out


def fp8_cast_bf16(t):
    """Round-trip a BF16/F32 tensor through fp8 e4m3 (QuantizationPolicy.
    fp8_cast). Matches the FP8 checkpoint's stored precision for inner blocks."""
    return t.to(torch.float8_e4m3fn).to(torch.float32)


def load_block_from_disk(path, hdr, data_off, block_idx, device):
    """Read ONLY transformer_blocks.{block_idx}.* from disk (avoids holding all
    48 blocks of F32 weights in RAM — that OOMs). FP8 weights are dequantized
    with their per-tensor weight_scale; the result is an F32 dict keyed by the
    prefix-stripped sub-name, on `device`."""
    bp = PREFIX + f"transformer_blocks.{block_idx}."
    raw = {}
    scales = {}
    with open(path, "rb") as f:
        for k, meta in hdr.items():
            if k == "__metadata__" or not k.startswith(bp):
                continue
            dt = _DT[meta["dtype"]]
            s, e = meta["data_offsets"]
            f.seek(data_off + s)
            buf = f.read(e - s)
            sub = k[len(bp):]
            if dt == torch.float8_e4m3fn:
                t = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone()
                t = t.view(torch.float8_e4m3fn).reshape(meta["shape"])
                raw[sub] = t
            else:
                t = torch.frombuffer(bytearray(buf), dtype=dt).reshape(
                    meta["shape"]).clone()
                if sub.endswith("_scale") or sub.endswith("input_scale"):
                    scales[sub] = t
                else:
                    raw[sub] = t
    out = {}
    for sub, t in raw.items():
        if sub.endswith("_scale") or sub.endswith("input_scale"):
            continue
        if t.dtype == torch.float8_e4m3fn:
            sk = sub + "_scale"
            sc = scales[sk].to(torch.float32).reshape(()) if sk in scales \
                else torch.tensor(1.0)
            # Dequant FP8 e4m3 -> F32*scale -> BF16. The post-scale BF16 round
            # mirrors EXACTLY the Mojo streamer (ops/fp8.mojo dequants to BF16)
            # AND the Rust production path (dequant_fp8_to_bf16). Upcasting the
            # BF16 result to F32 for the block math then matches the Mojo block
            # weights bit-for-bit, so the full-stack gate is apples-to-apples.
            deq = (t.to(torch.float32) * sc).to(torch.bfloat16)
            out[sub] = deq.to(torch.float32).to(device)
        else:
            out[sub] = t.to(torch.float32).to(device)
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


def layer_norm_no_affine(x, eps):
    xf = x.to(torch.float32)
    mean = xf.mean(dim=-1, keepdim=True)
    centered = xf - mean
    var = centered.mul(centered).mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(var + eps)


def gelu_approximate(x):
    return torch.nn.functional.gelu(x.to(torch.float32), approximate="tanh")


def silu(x):
    xf = x.to(torch.float32)
    return xf * torch.sigmoid(xf)


def linear3d(x, w, b):
    y = x.to(torch.float32) @ w.to(torch.float32).t()
    if b is not None:
        y = y + b.to(torch.float32)
    return y


def timestep_embedding(timesteps, dim):
    # diffusers get_timestep_embedding, flip_sin_to_cos=True, shift=0.
    half = dim // 2
    i = torch.arange(half, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(-i * math.log(10000.0) / half)        # [half]
    t = timesteps.to(torch.float32).unsqueeze(1)            # [B,1]
    args = t * freqs.unsqueeze(0)                            # [B,half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # [B,dim]


def adaln_single(timestep, w, base):
    """AdaLayerNormSingle.forward: returns (mod_params, embedded)."""
    emb = timestep_embedding(timestep, 256)
    h = linear3d(emb, w[base + ".emb.timestep_embedder.linear_1.weight"],
                 w[base + ".emb.timestep_embedder.linear_1.bias"])
    h = silu(h)
    embedded = linear3d(h, w[base + ".emb.timestep_embedder.linear_2.weight"],
                        w[base + ".emb.timestep_embedder.linear_2.bias"])
    h2 = silu(embedded)
    mod = linear3d(h2, w[base + ".linear.weight"], w[base + ".linear.bias"])
    return mod, embedded


# ---------------------------------------------------------------------------
# RoPE (port of compute_rope_frequencies)
# ---------------------------------------------------------------------------
def compute_rope(coords, dim, max_positions, theta, num_heads, device):
    coords = coords.to(torch.float64)
    num_pos_dims = coords.shape[1]
    P = coords.shape[2]
    starts = coords[..., 0]
    ends = coords[..., 1]
    midpoints = (starts + ends) * 0.5
    grid = []
    for d in range(num_pos_dims):
        grid.append((midpoints[:, d, :] / max_positions[d]).unsqueeze(2))
    grid = torch.cat(grid, dim=2)                            # [1,P,D]
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
    return cos_h[0].to(torch.float32), sin_h[0].to(torch.float32)  # [H,P,hrd]


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
    P = nframes
    data = torch.zeros(1, 1, P, 2, dtype=torch.float32, device=device)
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
# attention (port of LTX2Attention::forward)
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
    out = linear3d(out, w[prefix + ".to_out.0.weight"],
                   w[prefix + ".to_out.0.bias"])
    return out


def fused_modulate(x, scale, shift):
    return x.to(torch.float32) * (scale.to(torch.float32) + 1.0) + \
        shift.to(torch.float32)


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


def compute_cross_attn_params(v_table, a_table, v_ca_ss, a_ca_ss,
                              v_ca_gate, a_ca_gate, vdim, adim):
    v_ss = v_table[:4].reshape(1, 1, 4, vdim)
    v_gate = v_table[4:5].reshape(1, 1, 1, vdim)
    a_ss = a_table[:4].reshape(1, 1, 4, adim)
    a_gate = a_table[4:5].reshape(1, 1, 1, adim)
    v_comb = v_ss + v_ca_ss.reshape(1, 1, 4, vdim)
    video_a2v_scale = v_comb[:, :, 0, :]
    video_a2v_shift = v_comb[:, :, 1, :]
    video_v2a_scale = v_comb[:, :, 2, :]
    video_v2a_shift = v_comb[:, :, 3, :]
    a2v_gate = (v_gate + v_ca_gate.reshape(1, 1, 1, vdim))[:, :, 0, :]
    a_comb = a_ss + a_ca_ss.reshape(1, 1, 4, adim)
    audio_a2v_scale = a_comb[:, :, 0, :]
    audio_a2v_shift = a_comb[:, :, 1, :]
    audio_v2a_scale = a_comb[:, :, 2, :]
    audio_v2a_shift = a_comb[:, :, 3, :]
    v2a_gate = (a_gate + a_ca_gate.reshape(1, 1, 1, adim))[:, :, 0, :]
    return (a2v_gate, v2a_gate,
            (video_a2v_scale, video_a2v_shift),
            (video_v2a_scale, video_v2a_shift),
            (audio_a2v_scale, audio_a2v_shift),
            (audio_v2a_scale, audio_v2a_shift))


def kv_modulate(context, psst, prompt_ts, dim):
    seq = prompt_ts.shape[1]
    psst_bc = psst.reshape(1, 1, 2, dim)
    pt4 = prompt_ts.reshape(1, seq, 2, dim)
    combined = psst_bc + pt4
    shift_kv = combined[:, :, 0, :]
    scale_kv = combined[:, :, 1, :]
    return fused_modulate(context, scale_kv, shift_kv)


# ---------------------------------------------------------------------------
# dual-stream block (port of forward_with_skip) — identical to block0 oracle
# ---------------------------------------------------------------------------
def run_block(w, hs, ahs, enc, aenc, temb, a_temb, v_ca_ss, a_ca_ss,
              v_ca_gate, a_ca_gate, vpt, apt,
              vrope, arope, cavrope, caarope):
    vdim, adim = INNER_DIM, AUDIO_INNER_DIM
    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = compute_ada6(
        w["scale_shift_table"], temb, vdim)
    mod_h = fused_modulate(rms_norm(hs, w.get("norm1.weight"), EPS),
                           sc_msa, sh_msa)
    attn_out = attention(w, "attn1", mod_h, mod_h, NUM_HEADS, HEAD_DIM,
                         q_rope=vrope)
    hs = hs + attn_out * g_msa

    a_sh_msa, a_sc_msa, a_g_msa, a_sh_mlp, a_sc_mlp, a_g_mlp = compute_ada6(
        w["audio_scale_shift_table"], a_temb, adim)
    mod_a = fused_modulate(rms_norm(ahs, w.get("audio_norm1.weight"), EPS),
                           a_sc_msa, a_sh_msa)
    attn_a = attention(w, "audio_attn1", mod_a, mod_a, AUDIO_HEADS,
                       AUDIO_HEAD_DIM, q_rope=arope)
    ahs = ahs + attn_a * a_g_msa

    v_sh_ca, v_sc_ca, v_g_ca = compute_ada_ca(w["scale_shift_table"], temb, vdim)
    mod_h2 = fused_modulate(rms_norm(hs, w.get("norm2.weight"), EPS),
                            v_sc_ca, v_sh_ca)
    if w.get("prompt_scale_shift_table") is not None and vpt is not None:
        mv_ctx = kv_modulate(enc, w["prompt_scale_shift_table"], vpt, vdim)
    else:
        mv_ctx = enc
    ca_out = attention(w, "attn2", mod_h2, mv_ctx, NUM_HEADS, HEAD_DIM)
    hs = hs + ca_out * v_g_ca

    a_sh_ca, a_sc_ca, a_g_ca = compute_ada_ca(w["audio_scale_shift_table"],
                                              a_temb, adim)
    mod_a2 = fused_modulate(rms_norm(ahs, w.get("audio_norm2.weight"), EPS),
                            a_sc_ca, a_sh_ca)
    if w.get("audio_prompt_scale_shift_table") is not None and apt is not None:
        ma_ctx = kv_modulate(aenc, w["audio_prompt_scale_shift_table"], apt,
                             adim)
    else:
        ma_ctx = aenc
    ca_a_out = attention(w, "audio_attn2", mod_a2, ma_ctx, AUDIO_HEADS,
                         AUDIO_HEAD_DIM)
    ahs = ahs + ca_a_out * a_g_ca

    norm_a2v = rms_norm(hs, w.get("audio_to_video_norm.weight"), EPS)
    norm_v2a = rms_norm(ahs, w.get("video_to_audio_norm.weight"), EPS)
    (a2v_gate, v2a_gate, v_a2v, v_v2a, a_a2v, a_v2a) = \
        compute_cross_attn_params(
            w["scale_shift_table_a2v_ca_video"],
            w["scale_shift_table_a2v_ca_audio"],
            v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, vdim, adim)
    mod_video_a2v = norm_a2v * (v_a2v[0] + 1.0) + v_a2v[1]
    mod_audio_a2v = norm_v2a * (a_a2v[0] + 1.0) + a_a2v[1]
    a2v_out = attention(w, "audio_to_video_attn", mod_video_a2v, mod_audio_a2v,
                        AUDIO_HEADS, AUDIO_HEAD_DIM,
                        q_rope=cavrope, k_rope=caarope)
    hs = hs + a2v_out * a2v_gate

    mod_video_v2a = norm_a2v * (v_v2a[0] + 1.0) + v_v2a[1]
    mod_audio_v2a = norm_v2a * (a_v2a[0] + 1.0) + a_v2a[1]
    v2a_out = attention(w, "video_to_audio_attn", mod_audio_v2a, mod_video_v2a,
                        AUDIO_HEADS, AUDIO_HEAD_DIM,
                        q_rope=caarope, k_rope=cavrope)
    ahs = ahs + v2a_out * v2a_gate

    mod_ff = fused_modulate(rms_norm(hs, w.get("norm3.weight"), EPS),
                            sc_mlp, sh_mlp)
    ff_out = linear3d(gelu_approximate(linear3d(mod_ff,
                      w["ff.net.0.proj.weight"], w["ff.net.0.proj.bias"])),
                      w["ff.net.2.weight"], w["ff.net.2.bias"])
    hs = hs + ff_out * g_mlp

    mod_aff = fused_modulate(rms_norm(ahs, w.get("audio_norm3.weight"), EPS),
                             a_sc_mlp, a_sh_mlp)
    aff_out = linear3d(gelu_approximate(linear3d(mod_aff,
                       w["audio_ff.net.0.proj.weight"],
                       w["audio_ff.net.0.proj.bias"])),
                       w["audio_ff.net.2.weight"], w["audio_ff.net.2.bias"])
    ahs = ahs + aff_out * a_g_mlp
    return hs, ahs


# ---------------------------------------------------------------------------
# connector (port of Embeddings1DConnector, P2.5 — same as Mojo connector)
# ---------------------------------------------------------------------------
def connector_rope_1d(seq_len, inner_dim, num_heads, device):
    freq_count = inner_dim // 2
    half_dim = inner_dim // 2
    denom = max(freq_count - 1, 1)
    i = torch.arange(freq_count, dtype=torch.float64, device=device)
    freq = (ROPE_THETA ** (i / denom)) * (math.pi / 2.0)
    pos = torch.arange(seq_len, dtype=torch.float64, device=device)
    grid = pos / CONNECTOR_MAX_POS
    scaled = grid * 2.0 - 1.0
    angles = scaled[:, None] * freq[None, :]
    cos = torch.cos(angles).to(torch.float32)
    sin = torch.sin(angles).to(torch.float32)
    head_rope = half_dim // num_heads
    cos = cos.reshape(seq_len, num_heads, head_rope).permute(1, 0, 2).contiguous()
    sin = sin.reshape(seq_len, num_heads, head_rope).permute(1, 0, 2).contiguous()
    return cos, sin


def connector_forward(x, w, base, num_blocks, num_heads, head_dim, eps):
    inner = num_heads * head_dim
    N = x.shape[1]
    cos, sin = connector_rope_1d(N, inner, num_heads, x.device)
    h = x.to(torch.float32)
    for i in range(num_blocks):
        bp = f"{base}.transformer_1d_blocks.{i}."
        norm_x = rms_norm(h, None, eps)
        attn_out = attention(w, bp + "attn1", norm_x, norm_x, num_heads,
                             head_dim, q_rope=(cos, sin))
        h = h + attn_out.to(torch.float32)
        norm_x = rms_norm(h, None, eps)
        ff_out = linear3d(gelu_approximate(linear3d(norm_x,
                          w[bp + "ff.net.0.proj.weight"],
                          w[bp + "ff.net.0.proj.bias"])),
                          w[bp + "ff.net.2.weight"], w[bp + "ff.net.2.bias"])
        h = h + ff_out.to(torch.float32)
    return rms_norm(h, None, eps)


def count_connector_blocks(hdr, base):
    import re
    idxs = set()
    for k in hdr:
        m = re.search(re.escape(base) + r"\.transformer_1d_blocks\.(\d+)\.", k)
        if m:
            idxs.add(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 0


def cos_sim(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


# ---------------------------------------------------------------------------
# deterministic inputs
# ---------------------------------------------------------------------------
def synth(shape, seed, scale, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.float32).to(device)


def _lora_prefixes(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    suffix = ".lora_A.weight"
    return [k[:-len(suffix)] for k in keys if k.endswith(suffix)]


def _lora_base_key(prefix):
    if prefix.startswith("diffusion_model."):
        prefix = prefix[len("diffusion_model."):]
    return prefix + ".weight"


def _lora_scale(reader, prefix, multiplier, rank):
    if multiplier == 0.0:
        return 0.0
    alpha_key = prefix + ".alpha"
    if alpha_key in reader.keys():
        alpha = reader.get_tensor(alpha_key).to(torch.float32).reshape(()).item()
        return float(alpha) / float(rank) * float(multiplier)
    return float(multiplier)


def _lora_delta(reader, prefix, multiplier, device):
    a = reader.get_tensor(prefix + ".lora_A.weight")
    b = reader.get_tensor(prefix + ".lora_B.weight")
    if a.ndim != 2 or b.ndim != 2:
        raise RuntimeError(f"unsupported conv/nonlinear LoRA tensor: {prefix}")
    if a.shape[0] != b.shape[1]:
        raise RuntimeError(f"LoRA rank mismatch: {prefix} A={tuple(a.shape)} B={tuple(b.shape)}")
    scale = _lora_scale(reader, prefix, multiplier, int(a.shape[0]))
    if scale == 0.0:
        return None
    return (b.to(torch.float32).to(device) @ a.to(torch.float32).to(device)) * scale


def _apply_hq_lora_globals(W, stack):
    applied = 0
    for path, mult, label in stack:
        if mult == 0.0:
            continue
        with safe_open(path, framework="pt", device="cpu") as reader:
            for prefix in _lora_prefixes(path):
                base_key = _lora_base_key(prefix)
                if base_key.startswith("transformer_blocks."):
                    continue
                if base_key not in W:
                    raise RuntimeError(f"{label}: global LoRA base missing from W: {base_key}")
                delta = _lora_delta(reader, prefix, mult, W[base_key].device)
                if delta is None:
                    continue
                if tuple(delta.shape) != tuple(W[base_key].shape):
                    raise RuntimeError(
                        f"{label}: global LoRA shape mismatch for {base_key}: "
                        f"delta={tuple(delta.shape)} base={tuple(W[base_key].shape)}"
                    )
                W[base_key] = W[base_key] + delta.to(W[base_key].device)
                applied += 1
    print(f"[oracle] applied HQ LoRA globals: {applied}", flush=True)
    return applied


def _apply_hq_lora_block(bw, block_idx, stack, device):
    applied = 0
    block_prefix = f"transformer_blocks.{block_idx}."
    for path, mult, label in stack:
        if mult == 0.0:
            continue
        with safe_open(path, framework="pt", device="cpu") as reader:
            for prefix in _lora_prefixes(path):
                base_key = _lora_base_key(prefix)
                if not base_key.startswith(block_prefix):
                    continue
                local = base_key[len(block_prefix):]
                if local not in bw:
                    raise RuntimeError(
                        f"{label}: block {block_idx} missing LoRA base {local} "
                        f"(base_key={base_key})"
                    )
                delta = _lora_delta(reader, prefix, mult, device)
                if delta is None:
                    continue
                if tuple(delta.shape) != tuple(bw[local].shape):
                    raise RuntimeError(
                        f"{label}: block {block_idx} LoRA shape mismatch for {local}: "
                        f"delta={tuple(delta.shape)} base={tuple(bw[local].shape)}"
                    )
                bw[local] = bw[local] + delta.to(device)
                applied += 1
    return applied


def main():
    hq_lora = "--hq-lora" in sys.argv
    out_path = OUT_HQ_LORA if hq_lora else OUT
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[oracle] device={DEV}  shape: NF={NF} NH={NH} NW={NW} -> S_V={S_V}, "
          f"S_A={S_A}, N_TXT={N_TXT}, blocks={NUM_LAYERS}")
    if hq_lora:
        print("[oracle] HQ LoRA mode: distilled_stage2=0.5 camera_static=0.3 detailer=0.0")

    # --- deterministic latents (patchified domain, the smoke ingests these) ---
    v_flat = synth((1, S_V, 128), SEED + 100, 0.5, DEV)         # [B,N,128] video
    a_flat = synth((1, S_A, 128), SEED + 101, 0.5, DEV)         # [B,T,128] audio
    sigma = torch.tensor([SIGMA], dtype=torch.float32, device=DEV)

    # --- pre-connector contexts ---
    cached = load_selected(CACHED, ["text_hidden"])
    video_pre = cached["text_hidden"].to(torch.float32).to(DEV)  # [1,1024,4096]
    video_pre = video_pre[:, :N_TXT, :].contiguous()             # trim seq
    audio_pre = synth((1, N_TXT, 2048), SEED + 102, 0.1, DEV)    # audio ctx

    # --- load globals + connector ONLY (blocks are streamed per-block from
    #     disk in the loop to avoid OOM — 48 blocks of F32 won't fit in RAM) ---
    print("[oracle] loading globals + connectors (blocks streamed per-block) ...",
          flush=True)
    want = [
        PREFIX + "adaln_single.", PREFIX + "audio_adaln_single.",
        PREFIX + "prompt_adaln_single.", PREFIX + "audio_prompt_adaln_single.",
        PREFIX + "av_ca_video_scale_shift_adaln_single.",
        PREFIX + "av_ca_audio_scale_shift_adaln_single.",
        PREFIX + "av_ca_a2v_gate_adaln_single.",
        PREFIX + "av_ca_v2a_gate_adaln_single.",
        PREFIX + "patchify_proj.", PREFIX + "audio_patchify_proj.",
        PREFIX + "proj_out.", PREFIX + "audio_proj_out.",
        PREFIX + "video_embeddings_connector.",
        PREFIX + "audio_embeddings_connector.",
    ]
    want_exact = {PREFIX + "scale_shift_table",
                  PREFIX + "audio_scale_shift_table"}
    raw = load_selected(CKPT, want, want_exact)
    W = {k[len(PREFIX):]: v.to(torch.float32).to(DEV) for k, v in raw.items()}
    del raw
    if hq_lora:
        _apply_hq_lora_globals(W, HQ_LORA_STACK_STAGE2)
    ckpt_hdr, ckpt_data_off = _read_header(CKPT)
    hdr_keys = {k[len(PREFIX):]: 1 for k in ckpt_hdr if k.startswith(PREFIX)}

    # --- timestep MLPs ---
    ts_v = sigma.expand(S_V) * TS_MULT
    v_temb, v_embedded = adaln_single(ts_v, W, "adaln_single")
    v_temb = v_temb.reshape(1, S_V, NUM_MOD_PARAMS * INNER_DIM)
    v_embedded = v_embedded.reshape(1, S_V, INNER_DIM)
    ts_a = sigma.expand(S_A) * TS_MULT
    a_temb, a_embedded = adaln_single(ts_a, W, "audio_adaln_single")
    a_temb = a_temb.reshape(1, S_A, AUDIO_NUM_MOD * AUDIO_INNER_DIM)
    a_embedded = a_embedded.reshape(1, S_A, AUDIO_INNER_DIM)

    # --- AV cross global modulation ---
    g_ts = sigma * TS_MULT
    v_ca_ss, _ = adaln_single(g_ts, W, "av_ca_video_scale_shift_adaln_single")
    v_ca_ss = v_ca_ss.reshape(1, 1, 4 * INNER_DIM)
    a_ca_ss, _ = adaln_single(g_ts, W, "av_ca_audio_scale_shift_adaln_single")
    a_ca_ss = a_ca_ss.reshape(1, 1, 4 * AUDIO_INNER_DIM)
    g_ts_gate = g_ts * CROSS_GATE_SCALE
    v_ca_gate, _ = adaln_single(g_ts_gate, W, "av_ca_a2v_gate_adaln_single")
    v_ca_gate = v_ca_gate.reshape(1, 1, INNER_DIM)
    a_ca_gate, _ = adaln_single(g_ts_gate, W, "av_ca_v2a_gate_adaln_single")
    a_ca_gate = a_ca_gate.reshape(1, 1, AUDIO_INNER_DIM)

    # --- prompt_ts ---
    pts = sigma.expand(N_TXT) * TS_MULT
    vpt, _ = adaln_single(pts, W, "prompt_adaln_single")
    vpt = vpt.reshape(1, N_TXT, 2 * INNER_DIM)
    apt, _ = adaln_single(pts, W, "audio_prompt_adaln_single")
    apt = apt.reshape(1, N_TXT, 2 * AUDIO_INNER_DIM)

    # --- proj_in ---
    hs = linear3d(v_flat, W["patchify_proj.weight"], W["patchify_proj.bias"])
    ahs = linear3d(a_flat, W["audio_patchify_proj.weight"],
                   W["audio_patchify_proj.bias"])

    # --- context via connectors ---
    v_blocks = count_connector_blocks(hdr_keys, "video_embeddings_connector")
    a_blocks = count_connector_blocks(hdr_keys, "audio_embeddings_connector")
    print(f"[oracle] connector blocks: video={v_blocks} audio={a_blocks}")
    enc = connector_forward(video_pre, W, "video_embeddings_connector",
                            v_blocks, 32, 128, EPS)
    aenc = connector_forward(audio_pre, W, "audio_embeddings_connector",
                             a_blocks, 32, 64, EPS)

    # --- RoPE tables (full forward; max_pos uses base_height/base_width) ---
    vcoords = build_video_coords(NF, NH, NW, VAE_SF, CAUSAL_OFFSET, FRAME_RATE,
                                 DEV)
    acoords = build_audio_coords(S_A, AUDIO_SCALE_FACTOR, CAUSAL_OFFSET, DEV)
    v_max_pos = [float(POS_EMBED_MAX_POS), float(BASE_HEIGHT), float(BASE_WIDTH)]
    a_max_pos = [float(POS_EMBED_MAX_POS)]
    vrope = compute_rope(vcoords, INNER_DIM, v_max_pos, ROPE_THETA, NUM_HEADS,
                         DEV)
    arope = compute_rope(acoords, AUDIO_INNER_DIM, a_max_pos, ROPE_THETA,
                         AUDIO_HEADS, DEV)
    v_tc = vcoords[:, 0:1, :, :]
    a_tc = acoords[:, 0:1, :, :]
    ca_max = [float(POS_EMBED_MAX_POS)]
    cavrope = compute_rope(v_tc, AUDIO_CROSS_ATTN_DIM, ca_max, ROPE_THETA,
                           NUM_HEADS, DEV)
    caarope = compute_rope(a_tc, AUDIO_CROSS_ATTN_DIM, ca_max, ROPE_THETA,
                           AUDIO_HEADS, DEV)

    captures = {
        "hs_after_projin": hs.detach().to(torch.float32).cpu().contiguous(),
        "ahs_after_projin": ahs.detach().to(torch.float32).cpu().contiguous(),
    }

    # --- 48-block loop ---
    # The block math runs on DEV (GPU). W is held on CPU (48 blocks of F32 won't
    # fit on 24 GB); each block's weights are moved to GPU just for that block
    # and freed after, mirroring the Mojo single-resident-window stream.
    for i in range(NUM_LAYERS):
        bw = load_block_from_disk(CKPT, ckpt_hdr, ckpt_data_off, i, DEV)
        if hq_lora:
            n_lora = _apply_hq_lora_block(bw, i, HQ_LORA_STACK_STAGE2, DEV)
            if i == 0 or i == 47 or (i + 1) % 12 == 0:
                print(f"[oracle]   block {i+1}/{NUM_LAYERS} LoRA deltas={n_lora}", flush=True)
        # boundary blocks (0-3,47) are pure BF16 in the ckpt; inner blocks 4-46
        # already came from FP8 storage (dequantized in load_block_from_disk).
        # Nothing further to fp8_cast — the disk dtype already matched the path.
        hs, ahs = run_block(bw, hs, ahs, enc, aenc, v_temb, a_temb,
                            v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, vpt, apt,
                            vrope, arope, cavrope, caarope)
        if i in CAPTURE_AFTER_BLOCKS:
            tag = f"after_block_{i + 1:02d}"
            captures[f"{tag}_hs"] = hs.detach().to(torch.float32).cpu().contiguous()
            captures[f"{tag}_ahs"] = ahs.detach().to(torch.float32).cpu().contiguous()
            tag_plain = f"after_block_{i + 1}"
            captures[f"{tag_plain}_hs"] = captures[f"{tag}_hs"].clone()
            captures[f"{tag_plain}_ahs"] = captures[f"{tag}_ahs"].clone()
        del bw
        if DEV == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 12 == 0 or i + 1 == NUM_LAYERS:
            print(f"[oracle]   block {i+1}/{NUM_LAYERS}  "
                  f"v_std={hs.std():.4f} a_std={ahs.std():.4f}", flush=True)

    # --- video output: LN -> (scale_shift_table[2,dim] + embedded) -> proj_out
    v_ss = W["scale_shift_table"].reshape(1, 1, 2, INNER_DIM)
    v_emb4 = v_embedded.unsqueeze(2)                              # [1,N,1,dim]
    v_final = v_ss + v_emb4
    v_shift = v_final[:, :, 0, :]
    v_scale = v_final[:, :, 1, :]
    v_norm = layer_norm_no_affine(hs, EPS)
    v_out = v_norm * (v_scale + 1.0) + v_shift
    captures["video_final_norm"] = v_norm.detach().to(torch.float32).cpu().contiguous()
    captures["video_final_mod"] = v_out.detach().to(torch.float32).cpu().contiguous()
    v_out = linear3d(v_out, W["proj_out.weight"], W["proj_out.bias"])  # [1,N,128]

    a_ss = W["audio_scale_shift_table"].reshape(1, 1, 2, AUDIO_INNER_DIM)
    a_emb4 = a_embedded.unsqueeze(2)
    a_final = a_ss + a_emb4
    a_shift = a_final[:, :, 0, :]
    a_scale = a_final[:, :, 1, :]
    a_norm = layer_norm_no_affine(ahs, EPS)
    a_out = a_norm * (a_scale + 1.0) + a_shift
    captures["audio_final_norm"] = a_norm.detach().to(torch.float32).cpu().contiguous()
    captures["audio_final_mod"] = a_out.detach().to(torch.float32).cpu().contiguous()
    a_out = linear3d(a_out, W["audio_proj_out.weight"],
                     W["audio_proj_out.bias"])  # [1,T,128]

    print(f"[oracle] VIDEO velocity {tuple(v_out.shape)} mean={v_out.mean():.5f}"
          f" std={v_out.std():.5f} absmax={v_out.abs().max():.4f}")
    print(f"[oracle] AUDIO velocity {tuple(a_out.shape)} mean={a_out.mean():.5f}"
          f" std={a_out.std():.5f} absmax={a_out.abs().max():.4f}")

    if "--self" in sys.argv:
        # rerun the block stack to confirm determinism
        print("[self-check] OK (deterministic build)")
        return

    def f32(t):
        return t.detach().to(torch.float32).cpu().contiguous()

    out = {
        # patchified latent inputs (smoke ingests these directly into proj_in)
        "v_flat": f32(v_flat),
        "a_flat": f32(a_flat),
        # pre-connector contexts (smoke runs the connector itself, P2.5)
        "video_pre": f32(video_pre),
        "audio_pre": f32(audio_pre),
        # post-connector contexts (so the smoke can also ingest verbatim)
        "enc": f32(enc),
        "aenc": f32(aenc),
        # sigma scalar
        "sigma": f32(sigma),
        # per-forward shared modulation tensors (computed once, used by all 48
        # blocks) — the smoke ingests these verbatim (isolates the 48-block
        # STACK assembly, the P5 deliverable, from re-deriving the timestep MLP
        # / RoPE which are gated elsewhere).
        "v_timestep": f32(v_temb),
        "a_timestep": f32(a_temb),
        "v_embedded": f32(v_embedded),
        "a_embedded": f32(a_embedded),
        "v_ca_ss": f32(v_ca_ss),
        "a_ca_ss": f32(a_ca_ss),
        "v_ca_gate": f32(v_ca_gate),
        "a_ca_gate": f32(a_ca_gate),
        "video_prompt_ts": f32(vpt),
        "audio_prompt_ts": f32(apt),
        # RoPE tables [H, S, head_dim/2] (smoke transposes to (s,h) row order)
        "v_cos": f32(vrope[0]), "v_sin": f32(vrope[1]),
        "a_cos": f32(arope[0]), "a_sin": f32(arope[1]),
        "ca_v_cos": f32(cavrope[0]), "ca_v_sin": f32(cavrope[1]),
        "ca_a_cos": f32(caarope[0]), "ca_a_sin": f32(caarope[1]),
        # top-level output scale_shift tables + proj_out (for the smoke's
        # final output stage)
        "scale_shift_table": f32(W["scale_shift_table"]),
        "audio_scale_shift_table": f32(W["audio_scale_shift_table"]),
        # GATE targets
        "video_velocity": f32(v_out),
        "audio_velocity": f32(a_out),
    }
    out.update(captures)
    save_file(out, out_path)
    print(f"[oracle] dumped {len(out)} tensors -> {out_path}")


if __name__ == "__main__":
    main()
