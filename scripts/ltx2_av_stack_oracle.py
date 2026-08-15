#!/usr/bin/env python3
"""LTX-2 joint-AV 2-block STACK forward parity oracle (trainer P6.1 gate a).

Extends the gated block oracle (scripts/ltx2_av_block_bwd_oracle.py) from ONE
block to head + 2 blocks + tail — the composition the Mojo AV training STACK
(models/ltx2/ltx2_av_stack.mojo) must reproduce:

  HEAD  : patchify (patchify_proj / audio_patchify_proj) of noisy video/audio
          latents -> hidden/ahs; modulation FROM ONE SIGMA via the 8 global
          AdaLayerNormSingle MLPs (adaln_single / audio_adaln_single /
          av_ca_video|audio_scale_shift / av_ca_a2v|v2a_gate /
          prompt|audio_prompt_adaln_single) -> the block's temb/ca/prompt inputs.
  BLOCKS: block-0 then block-1 (REAL weights), SHARED head modulation + rope +
          contexts, PER-BLOCK scale_shift_table; reuses run_block from the gated
          block oracle (proven block math, no transcription).  NONZERO LoRA (rank
          16, scale 0.5, A and B random) on the 24-pair surface, PER BLOCK.
  TAIL  : layer_norm + (global scale_shift_table row0/1 + embedded) + proj_out
          / audio_proj_out -> video/audio velocity pred [1,S,128].

Full S_A=16, no attention padding mask (scout verdict 2026-07-18: audio padding
is LOSS-ONLY; no attention-level masking exists in torchref).

Dumps the byte-identical inputs + block-0/1 weight refs + reference outputs
(video/audio velocity AND pre-tail hidden for stage isolation) for the Mojo gate
`parity/ltx2_av_stack_parity.mojo`.

GPU/CPU note: F32 CPU compute (repo synthetic-dims convention). Run:
  /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_av_stack_oracle.py
"""
import json
import math
import os
import struct
import sys

import torch
from safetensors.torch import save_file

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Reuse the PROVEN, gated block math + helpers + dims from the block oracle.
from ltx2_av_block_bwd_oracle import (  # noqa: E402
    run_block,
    attention,
    _lora_add,
    rms_norm,
    fused_modulate,
    gelu_approximate,
    compute_ada6,
    compute_ada_ca,
    compute_cross_attn_params,
    kv_modulate,
    build_video_coords,
    build_audio_coords,
    compute_rope,
    linear3d,
    synth,
    cos_sim,
    EPS,
    ROPE_THETA,
    INNER_DIM,
    AUDIO_INNER_DIM,
    NUM_HEADS,
    HEAD_DIM,
    AUDIO_HEADS,
    AUDIO_HEAD_DIM,
    AUDIO_CROSS_ATTN_DIM,
    AUDIO_SCALE_FACTOR,
    POS_EMBED_MAX_POS,
    CAUSAL_OFFSET,
    VAE_SF,
    FRAME_RATE,
    NF,
    NH,
    NW,
    S_V,
    S_A,
    N_TXT,
    SEED,
    INPUT_SCALE,
    LORA_RANK,
    LORA_SCALE,
    LORA_MODULES,
    LORA_PROJS,
)

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_av_stack"
OUT = os.path.join(OUT_DIR, "av_stack2_ref.safetensors")
BWD_OUT = os.path.join(OUT_DIR, "av_stack2_bwd_ref.safetensors")  # --backward
DEV = "cpu"
N_BLOCKS = 2
DIFF = "model.diffusion_model."
SIGMA = 0.7                 # fixed forward sigma (the block-gate convention)
TS_MULT = 1000.0            # sigma -> timestep scale (MVP _build_mod)
# FFN LoRA surface (P6.2 (a) TRUE-672 + rider): audio_ff (672 preset) + video ff
# (1344 v2v-full reconcile). 4 single-linear adapters/block, appended after the 24
# attention pairs. Order MUST match ltx2_av_stack.mojo _av_lora_slots().
FFN_MODULES = ["audio_ff.net.0.proj", "audio_ff.net.2", "ff.net.0.proj", "ff.net.2"]


# ---------------------------------------------------------------------------
# checkpoint loading (bf16-roundtripped to F32, mirrors load_block0)
# ---------------------------------------------------------------------------
_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def _read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def _load_keys(path, want_prefix, strip):
    """Load every tensor whose key starts with `want_prefix`, bf16->F32, key
    stripped of `strip`."""
    hdr, data_off = _read_header(path)
    out = {}
    with open(path, "rb") as f:
        for k, meta in hdr.items():
            if k == "__metadata__" or not k.startswith(want_prefix):
                continue
            dt = _DT[meta["dtype"]]
            s, e = meta["data_offsets"]
            f.seek(data_off + s)
            raw = f.read(e - s)
            t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta["shape"])
            out[k[len(strip):]] = t.to(torch.float32).contiguous()
    return out


def load_block(path, idx):
    p = f"{DIFF}transformer_blocks.{idx}."
    return _load_keys(path, p, p)


ADALN_BASES = [
    "adaln_single", "audio_adaln_single",
    "av_ca_video_scale_shift_adaln_single", "av_ca_audio_scale_shift_adaln_single",
    "av_ca_a2v_gate_adaln_single", "av_ca_v2a_gate_adaln_single",
    "prompt_adaln_single", "audio_prompt_adaln_single",
]


def load_globals(path):
    """Head/tail globals ONLY (targeted — avoids loading the connectors/embedders):
    the 8 adaln MLPs + patchify / proj_out / scale_shift tables."""
    want = set()
    for k in ["patchify_proj", "audio_patchify_proj", "proj_out", "audio_proj_out"]:
        want.add(k + ".weight"); want.add(k + ".bias")
    want.add("scale_shift_table"); want.add("audio_scale_shift_table")
    for base in ADALN_BASES:
        for sub in [".emb.timestep_embedder.linear_1", ".emb.timestep_embedder.linear_2", ".linear"]:
            want.add(base + sub + ".weight"); want.add(base + sub + ".bias")
    hdr, data_off = _read_header(path)
    out = {}
    with open(path, "rb") as f:
        for k, meta in hdr.items():
            if k == "__metadata__" or not k.startswith(DIFF):
                continue
            short = k[len(DIFF):]
            if short not in want:
                continue
            dt = _DT[meta["dtype"]]
            s, e = meta["data_offsets"]
            f.seek(data_off + s)
            raw = f.read(e - s)
            t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta["shape"])
            out[short] = t.to(torch.float32).contiguous()
    return out


# ---------------------------------------------------------------------------
# head: AdaLayerNormSingle.forward (MVP _adaln_single / _timestep_embedding)
# ---------------------------------------------------------------------------
def silu(x):
    return x * torch.sigmoid(x)


def timestep_embedding(ts, dim, device):
    """Sinusoidal, cos-first then sin (MVP _timestep_embedding)."""
    n = len(ts)
    half = dim // 2
    out = torch.zeros(n, dim, dtype=torch.float32, device=device)
    for r in range(n):
        t = float(ts[r])
        for i in range(half):
            freq = math.exp(-i * math.log(10000.0) / half)
            arg = t * freq
            out[r, i] = math.cos(arg)
            out[r, half + i] = math.sin(arg)
    return out


def adaln_single(gw, base, ts_vals, device):
    """AdaLayerNormSingle: (mod [N, n*dim], embedded [N, dim])."""
    emb = timestep_embedding(ts_vals, 256, device)
    h = linear3d(emb, gw[base + ".emb.timestep_embedder.linear_1.weight"],
                 gw[base + ".emb.timestep_embedder.linear_1.bias"])
    h = silu(h)
    embedded = linear3d(h, gw[base + ".emb.timestep_embedder.linear_2.weight"],
                        gw[base + ".emb.timestep_embedder.linear_2.bias"])
    mod = linear3d(silu(embedded), gw[base + ".linear.weight"], gw[base + ".linear.bias"])
    return mod, embedded


def build_head(gw, sigma, device):
    """From ONE sigma -> the block's shared modulation inputs + tail embedded."""
    st = sigma * TS_MULT
    vt_mod, v_emb = adaln_single(gw, "adaln_single", [st] * S_V, device)
    at_mod, a_emb = adaln_single(gw, "audio_adaln_single", [st] * S_A, device)
    vcs, _ = adaln_single(gw, "av_ca_video_scale_shift_adaln_single", [st], device)
    acs, _ = adaln_single(gw, "av_ca_audio_scale_shift_adaln_single", [st], device)
    vcg, _ = adaln_single(gw, "av_ca_a2v_gate_adaln_single", [st], device)
    acg, _ = adaln_single(gw, "av_ca_v2a_gate_adaln_single", [st], device)
    vpt, _ = adaln_single(gw, "prompt_adaln_single", [st] * N_TXT, device)
    apt, _ = adaln_single(gw, "audio_prompt_adaln_single", [st] * N_TXT, device)
    return {
        "v_timestep": vt_mod.reshape(1, S_V, 9 * INNER_DIM),
        "a_timestep": at_mod.reshape(1, S_A, 9 * AUDIO_INNER_DIM),
        "v_embedded": v_emb.reshape(1, S_V, INNER_DIM),
        "a_embedded": a_emb.reshape(1, S_A, AUDIO_INNER_DIM),
        "v_ca_ss": vcs.reshape(1, 1, 4 * INNER_DIM),
        "a_ca_ss": acs.reshape(1, 1, 4 * AUDIO_INNER_DIM),
        "v_ca_gate": vcg.reshape(1, 1, INNER_DIM),
        "a_ca_gate": acg.reshape(1, 1, AUDIO_INNER_DIM),
        "video_prompt_ts": vpt.reshape(1, N_TXT, 2 * INNER_DIM),
        "audio_prompt_ts": apt.reshape(1, N_TXT, 2 * AUDIO_INNER_DIM),
    }


def output_stage(hs, sst, embedded, proj_w, proj_b, dim):
    """Tail: layer_norm + (sst[0]+embedded shift, sst[1]+embedded scale) + proj."""
    normed = torch.nn.functional.layer_norm(hs, (dim,), None, None, EPS)
    shift = sst[0].reshape(1, 1, dim) + embedded
    scale = sst[1].reshape(1, 1, dim) + embedded
    out = normed * (scale + 1.0) + shift
    return linear3d(out, proj_w, proj_b)


def build_lora(w, device, seed0, requires_grad=False):
    lora = {}
    seed = seed0
    for mod in LORA_MODULES:
        for proj in LORA_PROJS:
            out_f, in_f = w[f"{mod}.{proj}.weight"].shape
            A = synth((LORA_RANK, in_f), seed, device, scale=1.0 / math.sqrt(in_f))
            B = synth((out_f, LORA_RANK), seed + 1, device, scale=0.02)  # NONZERO B -> d_A non-degenerate
            seed += 2
            if requires_grad:
                A.requires_grad_(True)
                B.requires_grad_(True)
            lora[f"{mod}.{proj}"] = (A, B, LORA_SCALE)
    for ffn in FFN_MODULES:   # 4 FFN pairs (audio_ff + video ff, single linears)
        out_f, in_f = w[f"{ffn}.weight"].shape
        A = synth((LORA_RANK, in_f), seed, device, scale=1.0 / math.sqrt(in_f))
        B = synth((out_f, LORA_RANK), seed + 1, device, scale=0.02)  # NONZERO B
        seed += 2
        if requires_grad:
            A.requires_grad_(True)
            B.requires_grad_(True)
        lora[ffn] = (A, B, LORA_SCALE)
    return lora


def build_rope(device):
    vcoords = build_video_coords(NF, NH, NW, VAE_SF, CAUSAL_OFFSET, FRAME_RATE, device)
    acoords = build_audio_coords(S_A, AUDIO_SCALE_FACTOR, CAUSAL_OFFSET, device)
    # video h/w max_pos = FIXED [20,2048,2048] in torchref (rope.py:190) — not the
    # geometry-dependent VAE_SF*grid (torchref-WRONG; corrected 2026-07-18, rope gate).
    v_cos, v_sin = compute_rope(vcoords, INNER_DIM,
                                [POS_EMBED_MAX_POS, 2048.0, 2048.0],
                                ROPE_THETA, NUM_HEADS, device)
    a_cos, a_sin = compute_rope(acoords, AUDIO_INNER_DIM, [POS_EMBED_MAX_POS],
                                ROPE_THETA, AUDIO_HEADS, device)
    ca_v_cos, ca_v_sin = compute_rope(vcoords[:, 0:1, :, :], AUDIO_CROSS_ATTN_DIM,
                                      [POS_EMBED_MAX_POS], ROPE_THETA, AUDIO_HEADS, device)
    ca_a_cos, ca_a_sin = compute_rope(acoords[:, 0:1, :, :], AUDIO_CROSS_ATTN_DIM,
                                      [POS_EMBED_MAX_POS], ROPE_THETA, AUDIO_HEADS, device)
    return {
        "v_cos": v_cos, "v_sin": v_sin, "a_cos": a_cos, "a_sin": a_sin,
        "ca_v_cos": ca_v_cos, "ca_v_sin": ca_v_sin,
        "ca_a_cos": ca_a_cos, "ca_a_sin": ca_a_sin,
    }


def run_block_acts(w, inp, lora):
    """Byte-identical call sequence to the imported run_block, but captures the
    16 LTX2AVBlockActs FIELD tensors (hidden/ahs inputs, the 6 attention .out,
    the 6 residual points, the 2 FFN pre-gelu). Uses the SAME proven imported
    helpers (attention/compute_ada6/...) — no math transcription. main() cross-
    checks (hs,ahs) against run_block so a call-sequence slip is caught."""
    hs = inp["hs"]; ahs = inp["ahs"]
    enc = inp["enc_hs"]; aenc = inp["audio_enc_hs"]
    temb = inp["v_timestep"]; a_temb = inp["a_timestep"]
    v_ca_ss = inp["v_ca_ss"]; a_ca_ss = inp["a_ca_ss"]
    v_ca_gate = inp["v_ca_gate"]; a_ca_gate = inp["a_ca_gate"]
    vrope = (inp["v_cos"], inp["v_sin"]); arope = (inp["a_cos"], inp["a_sin"])
    cavrope = (inp["ca_v_cos"], inp["ca_v_sin"]); caarope = (inp["ca_a_cos"], inp["ca_a_sin"])
    vpt = inp["video_prompt_ts"]; apt = inp["audio_prompt_ts"]
    vdim, adim = INNER_DIM, AUDIO_INNER_DIM
    a = {"hidden": hs, "ahs": ahs}

    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = compute_ada6(w["scale_shift_table"], temb, vdim)
    mod_h = fused_modulate(rms_norm(hs, w.get("norm1.weight"), EPS), sc_msa, sh_msa)
    a["at1"] = attention(w, "attn1", mod_h, mod_h, NUM_HEADS, HEAD_DIM, lora, q_rope=vrope)
    hs = hs + a["at1"] * g_msa; a["hs1"] = hs

    a_sh_msa, a_sc_msa, a_g_msa, a_sh_mlp, a_sc_mlp, a_g_mlp = compute_ada6(w["audio_scale_shift_table"], a_temb, adim)
    mod_a = fused_modulate(rms_norm(ahs, w.get("audio_norm1.weight"), EPS), a_sc_msa, a_sh_msa)
    a["aat1"] = attention(w, "audio_attn1", mod_a, mod_a, AUDIO_HEADS, AUDIO_HEAD_DIM, lora, q_rope=arope)
    ahs = ahs + a["aat1"] * a_g_msa; a["ahss1"] = ahs

    v_sh_ca, v_sc_ca, v_g_ca = compute_ada_ca(w["scale_shift_table"], temb, vdim)
    mod_h2 = fused_modulate(rms_norm(hs, w.get("norm2.weight"), EPS), v_sc_ca, v_sh_ca)
    mv_ctx = kv_modulate(enc, w["prompt_scale_shift_table"], vpt, vdim)
    a["at2"] = attention(w, "attn2", mod_h2, mv_ctx, NUM_HEADS, HEAD_DIM, lora)
    hs = hs + a["at2"] * v_g_ca; a["hs2"] = hs

    a_sh_ca, a_sc_ca, a_g_ca = compute_ada_ca(w["audio_scale_shift_table"], a_temb, adim)
    mod_a2 = fused_modulate(rms_norm(ahs, w.get("audio_norm2.weight"), EPS), a_sc_ca, a_sh_ca)
    ma_ctx = kv_modulate(aenc, w["audio_prompt_scale_shift_table"], apt, adim)
    a["aat2"] = attention(w, "audio_attn2", mod_a2, ma_ctx, AUDIO_HEADS, AUDIO_HEAD_DIM, lora)
    ahs = ahs + a["aat2"] * a_g_ca; a["ahss2"] = ahs

    norm_a2v = rms_norm(hs, w.get("audio_to_video_norm.weight"), EPS)
    norm_v2a = rms_norm(ahs, w.get("video_to_audio_norm.weight"), EPS)
    (a2v_gate, v2a_gate, v_a2v, v_v2a, a_a2v, a_v2a) = compute_cross_attn_params(
        w["scale_shift_table_a2v_ca_video"], w["scale_shift_table_a2v_ca_audio"],
        v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, vdim, adim)
    mod_video_a2v = norm_a2v * (v_a2v[0] + 1.0) + v_a2v[1]
    mod_audio_a2v = norm_v2a * (a_a2v[0] + 1.0) + a_a2v[1]
    a["a2v"] = attention(w, "audio_to_video_attn", mod_video_a2v, mod_audio_a2v,
                         AUDIO_HEADS, AUDIO_HEAD_DIM, lora, q_rope=cavrope, k_rope=caarope)
    hs = hs + a["a2v"] * a2v_gate; a["hs3"] = hs
    mod_video_v2a = norm_a2v * (v_v2a[0] + 1.0) + v_v2a[1]
    mod_audio_v2a = norm_v2a * (a_v2a[0] + 1.0) + a_v2a[1]
    a["v2a"] = attention(w, "video_to_audio_attn", mod_audio_v2a, mod_video_v2a,
                         AUDIO_HEADS, AUDIO_HEAD_DIM, lora, q_rope=caarope, k_rope=cavrope)
    ahs = ahs + a["v2a"] * v2a_gate; a["ahss3"] = ahs

    mod_ff = fused_modulate(rms_norm(hs, w.get("norm3.weight"), EPS), sc_mlp, sh_mlp)
    a["h1_v"] = _lora_add(linear3d(mod_ff, w["ff.net.0.proj.weight"], w["ff.net.0.proj.bias"]),
                          mod_ff, lora, "ff.net.0.proj")
    h1g_v = gelu_approximate(a["h1_v"])
    ff_out = _lora_add(linear3d(h1g_v, w["ff.net.2.weight"], w["ff.net.2.bias"]),
                       h1g_v, lora, "ff.net.2")
    hs = hs + ff_out * g_mlp

    mod_aff = fused_modulate(rms_norm(ahs, w.get("audio_norm3.weight"), EPS), a_sc_mlp, a_sh_mlp)
    a["h1_a"] = _lora_add(linear3d(mod_aff, w["audio_ff.net.0.proj.weight"], w["audio_ff.net.0.proj.bias"]),
                          mod_aff, lora, "audio_ff.net.0.proj")
    h1g_a = gelu_approximate(a["h1_a"])
    aff_out = _lora_add(linear3d(h1g_a, w["audio_ff.net.2.weight"], w["audio_ff.net.2.bias"]),
                        h1g_a, lora, "audio_ff.net.2")
    ahs = ahs + aff_out * a_g_mlp

    return hs, ahs, a


# the 16 LTX2AVBlockActs fields, in struct order (ltx2_av_backward.mojo:1003).
ACT_FIELDS = ["hidden", "ahs", "at1", "aat1", "hs1", "ahss1", "at2", "aat2",
              "hs2", "ahss2", "a2v", "v2a", "hs3", "ahss3", "h1_v", "h1_a"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[stack-oracle] loading globals + {N_BLOCKS} blocks from {os.path.basename(CKPT)}")
    gw = load_globals(CKPT)
    blocks = [load_block(CKPT, i) for i in range(N_BLOCKS)]
    print(f"[stack-oracle] {len(gw)} globals, {len(blocks[0])} tensors/block; "
          f"S_V={S_V} S_A={S_A} N_TXT={N_TXT} sigma={SIGMA}")

    # noisy latents at the PATCHIFY-INPUT shape (channels=128 both streams) + contexts.
    v_lat = synth((1, S_V, 128), SEED + 0, DEV, scale=INPUT_SCALE)
    a_lat = synth((1, S_A, 128), SEED + 1, DEV, scale=INPUT_SCALE)
    enc = synth((1, N_TXT, INNER_DIM), SEED + 2, DEV, scale=INPUT_SCALE)
    aenc = synth((1, N_TXT, AUDIO_INNER_DIM), SEED + 3, DEV, scale=INPUT_SCALE)

    # HEAD: patchify + modulation-from-sigma + rope.
    backward = "--backward" in sys.argv
    hidden = linear3d(v_lat, gw["patchify_proj.weight"], gw["patchify_proj.bias"])
    ahs = linear3d(a_lat, gw["audio_patchify_proj.weight"], gw["audio_patchify_proj.bias"])
    if backward:
        # grad flows to the block-0 hidden/ahs; the HEAD is FROZEN (LoRA scope), so
        # the chain stops here — matches the Mojo stack backward's d_input.
        hidden = hidden.detach().requires_grad_(True)
        ahs = ahs.detach().requires_grad_(True)
    head = build_head(gw, SIGMA, DEV)
    rope = build_rope(DEV)

    shared = dict(head)
    del shared["v_embedded"], shared["a_embedded"]
    shared.update(rope)
    shared["enc_hs"] = enc
    shared["audio_enc_hs"] = aenc

    loras = [build_lora(blocks[i], DEV, SEED + 100 + i * 1000, requires_grad=backward)
             for i in range(N_BLOCKS)]

    # BLOCK LOOP (shared modulation, per-block weights + LoRA). Forward mode also
    # captures the 16 acts + cross-checks against run_block; backward mode uses the
    # clean run_block path (autograd tracks the graph to the loss).
    hs, ah = hidden, ahs
    block_acts = []
    for i in range(N_BLOCKS):
        inp = dict(shared); inp["hs"] = hs; inp["ahs"] = ah
        if backward:
            hs, ah = run_block(blocks[i], inp, loras[i])
        else:
            hs_a, ah_a, acts = run_block_acts(blocks[i], inp, loras[i])
            hs_r, ah_r = run_block(blocks[i], inp, loras[i])
            cv, ca = cos_sim(hs_a, hs_r), cos_sim(ah_a, ah_r)
            if cv < 0.9999999 or ca < 0.9999999:
                raise RuntimeError(f"block{i} run_block_acts != run_block (cos v={cv} a={ca})")
            block_acts.append(acts)
            hs, ah = hs_a, ah_a
            print(f"[stack-oracle] block{i} video std={hs.std():.5f} audio std={ah.std():.5f} "
                  f"(acts-vs-runblock {cv:.7f}/{ca:.7f})")

    v_hidden, a_hidden = hs, ah  # pre-tail (stage isolation)

    # TAIL.
    v_vel = output_stage(v_hidden, gw["scale_shift_table"], head["v_embedded"],
                         gw["proj_out.weight"], gw["proj_out.bias"], INNER_DIM)
    a_vel = output_stage(a_hidden, gw["audio_scale_shift_table"], head["a_embedded"],
                         gw["audio_proj_out.weight"], gw["audio_proj_out.bias"], AUDIO_INNER_DIM)
    print(f"[stack-oracle] video_vel std={v_vel.std():.5f} audio_vel std={a_vel.std():.5f}")

    if backward:
        # random tail cotangents (like the block bwd oracle) + autograd over the
        # 2 stream inputs and every LoRA A/B (24 pairs x N_BLOCKS). Proves the
        # COMPOSITION: tail-grad seeding, per-block d_x->d_y across BOTH streams.
        d_video = synth((1, S_V, 128), SEED + 500, DEV)
        d_audio = synth((1, S_A, 128), SEED + 501, DEV)
        loss = (v_vel * d_video).sum() + (a_vel * d_audio).sum()
        leaves = [hidden, ahs]
        gnames = ["d_hidden", "d_ahs"]
        for i in range(N_BLOCKS):
            for mod in LORA_MODULES:
                for proj in LORA_PROJS:
                    A, B, _ = loras[i][f"{mod}.{proj}"]
                    leaves += [A, B]
                    gnames += [f"b{i}.dA.{mod}.{proj}", f"b{i}.dB.{mod}.{proj}"]
            for ffn in FFN_MODULES:
                A, B, _ = loras[i][ffn]
                leaves += [A, B]
                gnames += [f"b{i}.dA.{ffn}", f"b{i}.dB.{ffn}"]
        grads = torch.autograd.grad(loss, leaves)
        out = {"v_lat": v_lat, "a_lat": a_lat, "enc": enc, "aenc": aenc,
               "d_video": d_video, "d_audio": d_audio}
        for k, v in rope.items():
            out[k] = v
        for i in range(N_BLOCKS):
            for mod in LORA_MODULES:
                for proj in LORA_PROJS:
                    A, B, _ = loras[i][f"{mod}.{proj}"]
                    out[f"b{i}.lora.{mod}.{proj}.A"] = A
                    out[f"b{i}.lora.{mod}.{proj}.B"] = B
            for ffn in FFN_MODULES:
                A, B, _ = loras[i][ffn]
                out[f"b{i}.lora.{ffn}.A"] = A
                out[f"b{i}.lora.{ffn}.B"] = B
        for name, gv in zip(gnames, grads):
            if float(gv.norm()) == 0.0:
                raise RuntimeError(f"degenerate (zero) reference grad: {name}")
            out[f"g.{name}"] = gv
        out = {k: v.detach().to(torch.float32).cpu().contiguous() for k, v in out.items()}
        save_file(out, BWD_OUT)
        print(f"[stack-oracle] BWD dumped {len(out)} tensors "
              f"(d_hidden |g|={grads[0].norm():.5f}, d_ahs |g|={grads[1].norm():.5f}) -> {BWD_OUT}")
        return

    if "--self" in sys.argv:
        hs2, ah2 = hidden, ahs
        for i in range(N_BLOCKS):
            inp = dict(shared); inp["hs"] = hs2; inp["ahs"] = ah2
            hs2, ah2 = run_block(blocks[i], inp, loras[i])
        print(f"[self] video cos={cos_sim(v_hidden, hs2):.7f} audio cos={cos_sim(a_hidden, ah2):.7f}")
        return

    # dump inputs + weight refs + reference outputs.
    out = {
        "v_lat": v_lat, "a_lat": a_lat, "enc": enc, "aenc": aenc,
        "video_hidden": v_hidden.contiguous(), "audio_hidden": a_hidden.contiguous(),
        "video_vel": v_vel.contiguous(), "audio_vel": a_vel.contiguous(),
    }
    for k, v in rope.items():
        out[k] = v.contiguous()
    for i in range(N_BLOCKS):
        for mod in LORA_MODULES:
            for proj in LORA_PROJS:
                A, B, _ = loras[i][f"{mod}.{proj}"]
                out[f"b{i}.lora.{mod}.{proj}.A"] = A.contiguous()
                out[f"b{i}.lora.{mod}.{proj}.B"] = B.contiguous()
        for ffn in FFN_MODULES:
            A, B, _ = loras[i][ffn]
            out[f"b{i}.lora.{ffn}.A"] = A.contiguous()
            out[f"b{i}.lora.{ffn}.B"] = B.contiguous()
    # all 16 LTX2AVBlockActs field tensors per block (the full acts contract the
    # gated backward consumes; pin (a)).
    for i in range(N_BLOCKS):
        for f in ACT_FIELDS:
            out[f"b{i}.act.{f}"] = block_acts[i][f].contiguous()
    out = {k: v.detach().to(torch.float32).cpu().contiguous() for k, v in out.items()}
    save_file(out, OUT)
    print(f"[stack-oracle] dumped {len(out)} tensors -> {OUT}")


if __name__ == "__main__":
    main()
