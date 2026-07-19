#!/usr/bin/env python3
"""LTX-2 VIDEO-ONLY (audio=None) transformer block-0 BACKWARD parity oracle.

torch.autograd reference for the hand-chained Mojo VIDEO-ONLY backward
(serenitymojo/models/ltx2/ltx2_video_backward.mojo) — the video-mode training
arm musubi runs when the audio modality is absent.

Video-only sublayer set (musubi BasicAVTransformerBlock._forward, audio=None):
  /home/alex/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/transformer.py
    run_vx   :589   video is not None and enabled and numel>0   -> True
    run_ax   :590   audio is None                                -> False
    run_a2v  :592   run_vx and (audio is not None ...)           -> False
    run_v2a  :593   run_ax and (...)                             -> False
  So ONLY these run:
    video self-attn (attn1, video rope)      :595-611  (if run_vx)
    video text cross-attn (attn2)            :613-628  (inside run_vx)
    video FFN                                :763-776  (if run_vx)
  Skipped entirely (audio=None): audio self-attn / audio text cross-attn
  (:632-670, if run_ax), cross-modal a2v/v2a (:673-761, if run_a2v or run_v2a),
  audio FFN (:778-791, if run_ax).

This oracle REUSES the green AV oracle's helper math verbatim
(scripts/ltx2_av_block_bwd_oracle.py — imported as `av`, its main() is guarded
so import is side-effect free) and just runs the video-only forward + backward:
  * REAL block-0 weights from the dequant-bf16 export (block-0 keys only),
  * factorized LoRA y = Wx + b + scale*B(A(x)) on the 8 video-mode targets:
    {to_q,to_k,to_v,to_out.0} x {attn1, attn2}. A AND B both random nonzero.
  * torch.autograd grads from a seeded d_video cotangent:
    d_hidden (video input grad) + d_A/d_B for all 8 pairs.

Dump -> output/ltx2_av_bwd/video_block0_bwd_ref.safetensors (all F32):
  inputs hs, enc_hs, v_timestep, video_prompt_ts, v_cos, v_sin,
  lora.<mod>.<proj>.A/.B, d_video, video_out (fwd cross-check),
  g_d_hidden, g_dA.<mod>.<proj>, g_dB.<mod>.<proj>.

Run:
  /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_video_block_bwd_oracle.py
"""

import math
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ltx2_av_block_bwd_oracle as av  # noqa: E402  (shared helper math)

OUT_DIR = av.OUT_DIR
OUT = os.path.join(OUT_DIR, "video_block0_bwd_ref.safetensors")
OUT_V2V = os.path.join(OUT_DIR, "video_block0_bwd_v2v_ref.safetensors")

# Video-mode LoRA-active surface = 8 pairs/block (t2v); v2v adds 2 FFN pairs.
LORA_MODULES = ["attn1", "attn2"]
LORA_PROJS = ["to_q", "to_k", "to_v", "to_out.0"]
LORA_FFN = ["ff.net.0.proj", "ff.net.2"]   # v2v-only (P3): the two video FFN linears

DEV = av.DEV


def _linear3d_lora(x, W, b, lora, key):
    """av.linear3d(x,W,b) + scale*B(A(x)) when a factor is attached under `key`."""
    out = av.linear3d(x, W, b)
    if key in lora:
        A, B, scale = lora[key]
        out = out + ((x @ A.t()) @ B.t()) * scale
    return out


# ---------------------------------------------------------------------------
# video-only forward (musubi _forward audio=None path)
# ---------------------------------------------------------------------------
def run_video_block(w, inp, lora):
    hs = inp["hs"]
    enc = inp["enc_hs"]
    temb = inp["v_timestep"]
    vrope = (inp["v_cos"], inp["v_sin"])
    vpt = inp["video_prompt_ts"]
    vdim = av.INNER_DIM

    # 1. video self-attn (transformer.py:595-611)
    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = av.compute_ada6(
        w["scale_shift_table"], temb, vdim)
    mod_h = av.fused_modulate(
        av.rms_norm(hs, w.get("norm1.weight"), av.EPS), sc_msa, sh_msa)
    attn_out = av.attention(w, "attn1", mod_h, mod_h, av.NUM_HEADS, av.HEAD_DIM,
                            lora, q_rope=vrope)
    hs = hs + attn_out * g_msa

    # 2. video text cross-attn (transformer.py:613-628, _apply_text_cross_attention)
    v_sh_ca, v_sc_ca, v_g_ca = av.compute_ada_ca(w["scale_shift_table"], temb, vdim)
    mod_h2 = av.fused_modulate(
        av.rms_norm(hs, w.get("norm2.weight"), av.EPS), v_sc_ca, v_sh_ca)
    mv_ctx = av.kv_modulate(enc, w["prompt_scale_shift_table"], vpt, vdim)
    ca_out = av.attention(w, "attn2", mod_h2, mv_ctx, av.NUM_HEADS, av.HEAD_DIM, lora)
    hs = hs + ca_out * v_g_ca

    # (audio self-attn, audio text cross-attn, cross-modal a2v/v2a all skipped)

    # 3. video FFN (transformer.py:763-776); clamp omitted (identity in-range).
    # v2v attaches LoRA on ff.net.0.proj + ff.net.2 (no-op when absent from lora).
    mod_ff = av.fused_modulate(
        av.rms_norm(hs, w.get("norm3.weight"), av.EPS), sc_mlp, sh_mlp)
    h1 = _linear3d_lora(
        mod_ff, w["ff.net.0.proj.weight"], w["ff.net.0.proj.bias"], lora, "ff.net.0.proj")
    ff_out = _linear3d_lora(
        av.gelu_approximate(h1), w["ff.net.2.weight"], w["ff.net.2.bias"], lora, "ff.net.2")
    hs = hs + ff_out * g_mlp
    return hs


def build_video_inputs(device):
    """Only the inputs the video-only path consumes (same seeds/scales as AV)."""
    inp = {}
    inp["hs"] = av.synth((1, av.S_V, av.INNER_DIM), av.SEED + 0, device)
    inp["enc_hs"] = av.synth((1, av.N_TXT, av.INNER_DIM), av.SEED + 2, device)
    inp["v_timestep"] = av.synth((1, av.S_V, 9 * av.INNER_DIM), av.SEED + 4, device)
    inp["video_prompt_ts"] = av.synth(
        (1, av.N_TXT, 2 * av.INNER_DIM), av.SEED + 10, device)

    vcoords = av.build_video_coords(
        av.NF, av.NH, av.NW, av.VAE_SF, av.CAUSAL_OFFSET, av.FRAME_RATE, device)
    v_cos, v_sin = av.compute_rope(
        vcoords, av.INNER_DIM,
        [av.POS_EMBED_MAX_POS, av.VAE_SF[1] * av.NH, av.VAE_SF[2] * av.NW],
        av.ROPE_THETA, av.NUM_HEADS, device)
    inp["v_cos"], inp["v_sin"] = v_cos, v_sin
    return inp


def _add_factor(lora, key, w, device, seed):
    out_f, in_f = w[f"{key}.weight"].shape
    A = av.synth((av.LORA_RANK, in_f), seed, device,
                 scale=1.0 / math.sqrt(in_f))      # kaiming-ish
    B = av.synth((out_f, av.LORA_RANK), seed + 1, device, scale=0.02)
    A.requires_grad_(True)
    B.requires_grad_(True)
    lora[key] = (A, B, av.LORA_SCALE)


def lora_key_list(v2v):
    keys = [f"{m}.{p}" for m in LORA_MODULES for p in LORA_PROJS]
    if v2v:
        keys += LORA_FFN
    return keys


def build_video_lora(w, device, v2v=False):
    """8 factorized adapters ({attn1,attn2} x 4 projs), A and B random nonzero;
    v2v adds 2 FFN adapters (ff.net.0.proj 4096->16384, ff.net.2 16384->4096)."""
    lora = {}
    seed = av.SEED + 200
    for key in lora_key_list(v2v):
        _add_factor(lora, key, w, device, seed)
        seed += 2
    return lora


def main():
    v2v = "--v2v" in sys.argv
    out_path = OUT_V2V if v2v else OUT
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[vid-bwd-oracle] loading block-0 from {os.path.basename(av.CKPT)}"
          f" (preset={'v2v' if v2v else 't2v'})")
    w = av.load_block0(av.CKPT)
    print(f"[vid-bwd-oracle] {len(w)} block-0 tensors (bf16-roundtripped F32)")
    inp = build_video_inputs(DEV)
    lora = build_video_lora(w, DEV, v2v)
    keys = lora_key_list(v2v)
    print(f"[vid-bwd-oracle] S_V={av.S_V} N_TXT={av.N_TXT} rank={av.LORA_RANK} "
          f"scale={av.LORA_SCALE} adapters={len(lora)} (video-only)")

    hs_leaf = inp["hs"].clone().requires_grad_(True)
    fwd_inp = dict(inp)
    fwd_inp["hs"] = hs_leaf

    hs_out = run_video_block(w, fwd_inp, lora)
    print(f"[vid-bwd-oracle] video_out mean={hs_out.mean():.5f} std={hs_out.std():.5f}")

    d_video = av.synth((1, av.S_V, av.INNER_DIM), av.SEED + 500, DEV)
    loss = (hs_out * d_video).sum()

    leaves = [hs_leaf]
    names = ["g_d_hidden"]
    for key in keys:
        A, B, _ = lora[key]
        leaves += [A, B]
        names += [f"g_dA.{key}", f"g_dB.{key}"]
    grads = torch.autograd.grad(loss, leaves)

    out = {}
    for k, v in inp.items():
        out[k] = v.detach().to(torch.float32).cpu().contiguous()
    for key in keys:
        A, B, _ = lora[key]
        out[f"lora.{key}.A"] = A.detach().to(torch.float32).cpu().contiguous()
        out[f"lora.{key}.B"] = B.detach().to(torch.float32).cpu().contiguous()
    out["d_video"] = d_video.cpu().contiguous()
    out["video_out"] = hs_out.detach().to(torch.float32).cpu().contiguous()
    for name, g in zip(names, grads):
        gn = float(g.norm())
        if gn == 0.0:
            raise RuntimeError(f"degenerate (zero) reference grad: {name}")
        out[name] = g.detach().to(torch.float32).cpu().contiguous()
    print(f"[vid-bwd-oracle] d_hidden |g|={grads[0].norm():.5f}")

    if "--self" in sys.argv:
        hs2 = run_video_block(w, fwd_inp, lora)
        print(f"[self] video cos={av.cos_sim(hs_out, hs2):.7f}")
        return

    save_file(out, out_path)
    print(f"[vid-bwd-oracle] dumped {len(out)} tensors -> {out_path}")


if __name__ == "__main__":
    main()
