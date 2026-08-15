#!/usr/bin/env python3
"""LTX-2.3 VIDEO-MODE STACK (head + N blocks + tail) BACKWARD parity oracle.

torch.autograd reference for the composed Mojo video-mode training stack
(serenitymojo/models/ltx2/ltx2_video_stack.mojo):

    latent -> patchify -> patchify_proj -> [block 0 .. block N-1] -> tail -> pred

REDUCED DEPTH (2 real blocks) at a small geometry so the whole autograd graph
fits on CPU. Everything is REAL:
  * blocks 0 and 1: real weights from the distilled dequant-bf16 export
    (block keys only; bf16-roundtripped F32 to match the Mojo BF16 loader),
  * head: real patchify_proj + real adaln_single + real prompt_adaln_single for
    ONE fixed sigma (the AdaLN-single conditioning the DiT actually consumes),
  * tail: real top-level scale_shift_table[2,4096] + real proj_out, the torchref
    _process_output "AdaLN Structural Fix" done in F32
    (torchref model.py:_process_output:797).
  * factorized LoRA y = Wx + b + scale*B(A x) on the 8 video-mode targets per
    block ({to_q,to_k,to_v,to_out.0} x {attn1, attn2}), A AND B random nonzero.

The forward reuses the GREEN per-block video oracle verbatim
(scripts/ltx2_video_block_bwd_oracle.py::run_video_block, imported as `vid`,
which itself imports the AV helper math as `av`). We add ONLY the frozen head
(patchify + adaln) and frozen tail (F32 modulate + no-affine layernorm +
proj_out), then autograd through hidden0 (the block-0 input leaf = d_input) and
every LoRA A/B.

The head is FROZEN: gradient stops at hidden0 (block-0 input). We expose hidden0
as the leaf and report its grad as `g_d_input` — the tensor the Mojo stack
backward returns (grad into block-0 hidden; patchify/adaln not backpropagated).

Dump -> output/ltx2_video_stack/video_stack_bwd_ref.safetensors (all F32):
  hidden0, enc_hs, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
  lora.<bi>.<mod>.<proj>.A/.B, d_pred, pred (fwd cross-check),
  g_d_input, g_dA.<bi>.<mod>.<proj>, g_dB.<bi>.<mod>.<proj>.

Run:
  /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_video_stack_oracle.py
"""

import json
import math
import os
import struct
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ltx2_av_block_bwd_oracle as av       # noqa: E402  (helper math)
import ltx2_video_block_bwd_oracle as vid   # noqa: E402  (per-block video fwd)

CKPT = av.CKPT   # distilled dequant-bf16 export (blocks + head + tail, video path)
# Real torchref TE cache row source for the context (POST-connector embeds); real
# distribution beats synthetic. First N_TXT rows sliced below.
CACHE_TE = "/home/alex/datasets/ltx2_ref_v3/cache/0288f3d69c08e816d81b014da620db49_ltx2_te.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_video_stack"
OUT = os.path.join(OUT_DIR, "video_stack_bwd_ref.safetensors")

DEV = av.DEV
EPS = av.EPS
VDIM = av.INNER_DIM              # 4096

# ── REDUCED stack geometry (2 blocks, tiny latent) ──────────────────────────
NUM_BLOCKS = 2
NF, NH, NW = 2, 3, 4            # latent grid -> S_V = 24 tokens
S_V = NF * NH * NW              # 24
N_TXT = 32
LATENT_C = 128                 # patchify channel = per-token in_ch
SIGMA = 0.7                    # one fixed timestep for the head AdaLN path
TS_MULT = 1000.0

LORA_MODULES = ["attn1", "attn2"]
LORA_PROJS = ["to_q", "to_k", "to_v", "to_out.0"]

DM = "model.diffusion_model."


# ── partial safetensors loads (bf16-roundtripped F32, matches Mojo loader) ──
def _read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def load_block(path, idx):
    """Load one transformer block's tensors (prefix-stripped, bf16-roundtripped)."""
    prefix = f"{DM}transformer_blocks.{idx}."
    hdr, data_off = _read_header(path)
    out = {}
    with open(path, "rb") as f:
        for k, meta in hdr.items():
            if k == "__metadata__" or not k.startswith(prefix):
                continue
            dt = _DT[meta["dtype"]]
            s, e = meta["data_offsets"]
            f.seek(data_off + s)
            raw = f.read(e - s)
            t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta["shape"])
            out[k[len(prefix):]] = t.to(torch.bfloat16).to(torch.float32).to(DEV)
    return out


def load_named(path, names):
    """Load specific top-level keys (head/tail), bf16-roundtripped F32."""
    hdr, data_off = _read_header(path)
    want = {DM + n: n for n in names}
    out = {}
    with open(path, "rb") as f:
        for full, meta in hdr.items():
            if full not in want:
                continue
            dt = _DT[meta["dtype"]]
            s, e = meta["data_offsets"]
            f.seek(data_off + s)
            raw = f.read(e - s)
            t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta["shape"])
            out[want[full]] = t.to(torch.bfloat16).to(torch.float32).to(DEV)
    missing = [n for n in names if n not in out]
    if missing:
        raise RuntimeError(f"load_named: missing head/tail keys {missing}")
    return out


# ── head: sinusoidal timestep embedding (MVP _timestep_embedding: cos-first,
#    downscale_freq_shift=0) + AdaLayerNormSingle MLP ─────────────────────────
def timestep_embedding(ts, dim=256):
    half = dim // 2
    i = torch.arange(half, dtype=torch.float64, device=DEV)
    freq = torch.exp(-i * math.log(10000.0) / half)            # [half]
    arg = ts.view(-1, 1).double() * freq.view(1, -1)           # [N,half]
    emb = torch.cat([torch.cos(arg), torch.sin(arg)], dim=1)   # [N,dim] cos|sin
    return emb.to(torch.float32)


def adaln_single(hw, base, ts_vals):
    """embedded = linear_2(silu(linear_1(sinusoidal(ts)))); mod = linear(silu(embedded))."""
    emb = timestep_embedding(ts_vals, 256)                      # [N,256]
    w1 = hw[base + ".emb.timestep_embedder.linear_1.weight"]
    b1 = hw[base + ".emb.timestep_embedder.linear_1.bias"]
    h = torch.nn.functional.silu(emb @ w1.t() + b1)            # [N,4096]
    w2 = hw[base + ".emb.timestep_embedder.linear_2.weight"]
    b2 = hw[base + ".emb.timestep_embedder.linear_2.bias"]
    embedded = h @ w2.t() + b2                                  # [N,4096]
    wl = hw[base + ".linear.weight"]
    bl = hw[base + ".linear.bias"]
    mod = torch.nn.functional.silu(embedded) @ wl.t() + bl      # [N, n*4096]
    return mod, embedded


# ── tail: torchref _process_output (F32 AdaLN Structural Fix) ──────────────────
def tail_process_output(x, v_embedded, sst, proj_w, proj_b, eps):
    x32 = x.to(torch.float32)
    emb32 = v_embedded.to(torch.float32)                       # [1,S_V,4096]
    sst32 = sst.to(torch.float32)                              # [2,4096]
    shift = sst32[0].view(1, 1, -1) + emb32                    # [1,S_V,4096]
    scale = sst32[1].view(1, 1, -1) + emb32
    normed = torch.nn.functional.layer_norm(x32, (x32.shape[-1],), eps=eps)
    xm = normed * (1.0 + scale) + shift
    pred = xm @ proj_w.to(torch.float32).t() + proj_b.to(torch.float32)
    return pred                                                # [1,S_V,128]


def load_real_context(n_txt):
    """First n_txt rows of a REAL POST-connector video_prompt_embeds cache row
    (bf16-roundtripped F32). torchref's cache mask is all-ones so every row is a
    valid token — slicing the head rows lands entirely in real context."""
    with safe_open(CACHE_TE, framework="pt") as st:
        m = st.get_tensor("prompt_attention_mask")
        if int(m.min()) != 1 or int(m.max()) != 1:
            raise RuntimeError("cache prompt_attention_mask not all-ones")
        emb = st.get_tensor("video_prompt_embeds_bfloat16")   # [1024,4096] bf16
    emb = emb[:n_txt].to(torch.bfloat16).to(torch.float32)
    return emb.reshape(1, n_txt, VDIM).to(DEV)


def build_stack_inputs():
    inp = {}
    # latent [1,C,F,H,W]; patchify -> [1,S_V,C]
    latent = av.synth((1, LATENT_C, NF, NH, NW), av.SEED + 700, DEV, scale=0.5)
    inp["latent"] = latent
    inp["enc_hs"] = load_real_context(N_TXT)   # real POST-connector context rows
    # 3D video rope — TORCHREF-FAITHFUL params so the Mojo head's rope can be gated
    # against this: causal_offset=1 (get_pixel_coords causal_fix) + max_pos
    # [20, 2048, 2048] (POS_EMBED_MAX_POS / BASE_HW), matching the AV MVP spine
    # and ltx2_video_stack.mojo::_build_video_rope. rope freqs stay F64 (no bf16
    # rounding of inv_freq, MJ-0815).
    vcoords = av.build_video_coords(
        NF, NH, NW, av.VAE_SF, 1, av.FRAME_RATE, DEV)   # causal_offset=1
    v_cos, v_sin = av.compute_rope(
        vcoords, VDIM, [20.0, 2048.0, 2048.0],
        av.ROPE_THETA, av.NUM_HEADS, DEV)
    inp["v_cos"], inp["v_sin"] = v_cos, v_sin
    return inp


def build_block_lora(w, bi):
    lora = {}
    seed = av.SEED + 900 + bi * 100
    for mod in LORA_MODULES:
        for proj in LORA_PROJS:
            out_f, in_f = w[f"{mod}.{proj}.weight"].shape
            A = av.synth((av.LORA_RANK, in_f), seed, DEV, scale=1.0 / math.sqrt(in_f))
            B = av.synth((out_f, av.LORA_RANK), seed + 1, DEV, scale=0.02)
            seed += 2
            A.requires_grad_(True)
            B.requires_grad_(True)
            lora[f"{mod}.{proj}"] = (A, B, av.LORA_SCALE)
    return lora


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[stack-oracle] loading {NUM_BLOCKS} blocks + head/tail from "
          f"{os.path.basename(CKPT)}")
    blocks = [load_block(CKPT, bi) for bi in range(NUM_BLOCKS)]
    head_keys = []
    for base in ["adaln_single", "prompt_adaln_single"]:
        head_keys += [
            base + ".emb.timestep_embedder.linear_1.weight",
            base + ".emb.timestep_embedder.linear_1.bias",
            base + ".emb.timestep_embedder.linear_2.weight",
            base + ".emb.timestep_embedder.linear_2.bias",
            base + ".linear.weight", base + ".linear.bias",
        ]
    head_keys += ["patchify_proj.weight", "patchify_proj.bias"]
    tail_keys = ["scale_shift_table", "proj_out.weight", "proj_out.bias"]
    hw = load_named(CKPT, head_keys + tail_keys)
    print(f"[stack-oracle] S_V={S_V} N_TXT={N_TXT} sigma={SIGMA} "
          f"rank={av.LORA_RANK} scale={av.LORA_SCALE}")

    inp = build_stack_inputs()

    # ── head (FROZEN) ───────────────────────────────────────────────────────
    # patchify: [1,C,F,H,W] -> [1,C,S_V] -> [1,S_V,C]   (torchref einops p=1)
    tokens = inp["latent"].reshape(1, LATENT_C, S_V).permute(0, 2, 1)
    hidden0 = tokens @ hw["patchify_proj.weight"].t() + hw["patchify_proj.bias"]
    ts_v = torch.full((S_V,), SIGMA * TS_MULT, dtype=torch.float32, device=DEV)
    v_temb, v_embedded = adaln_single(hw, "adaln_single", ts_v)   # [S_V,9*4096],[S_V,4096]
    v_temb = v_temb.reshape(1, S_V, 9 * VDIM)
    v_embedded = v_embedded.reshape(1, S_V, VDIM)
    ts_p = torch.full((N_TXT,), SIGMA * TS_MULT, dtype=torch.float32, device=DEV)
    v_prompt_ts, _ = adaln_single(hw, "prompt_adaln_single", ts_p)  # [N_TXT,2*4096]
    v_prompt_ts = v_prompt_ts.reshape(1, N_TXT, 2 * VDIM)

    # hidden0 is the STACK input leaf (head frozen -> grad stops here = d_input).
    hidden0_leaf = hidden0.detach().clone().requires_grad_(True)

    loras = [build_block_lora(blocks[bi], bi) for bi in range(NUM_BLOCKS)]

    # ── blocks (reuse the green per-block video forward) ────────────────────
    x = hidden0_leaf
    for bi in range(NUM_BLOCKS):
        blk_inp = {
            "hs": x, "enc_hs": inp["enc_hs"],
            "v_timestep": v_temb, "video_prompt_ts": v_prompt_ts,
            "v_cos": inp["v_cos"], "v_sin": inp["v_sin"],
        }
        x = vid.run_video_block(blocks[bi], blk_inp, loras[bi])

    # ── tail (FROZEN, F32 interior) ─────────────────────────────────────────
    pred = tail_process_output(
        x, v_embedded, hw["scale_shift_table"],
        hw["proj_out.weight"], hw["proj_out.bias"], EPS)
    print(f"[stack-oracle] pred mean={pred.mean():.6f} std={pred.std():.6f} "
          f"shape={tuple(pred.shape)}")

    d_pred = av.synth((1, S_V, pred.shape[-1]), av.SEED + 800, DEV)
    loss = (pred * d_pred).sum()

    leaves = [hidden0_leaf]
    names = ["g_d_input"]
    for bi in range(NUM_BLOCKS):
        for mod in LORA_MODULES:
            for proj in LORA_PROJS:
                A, B, _ = loras[bi][f"{mod}.{proj}"]
                leaves += [A, B]
                names += [f"g_dA.{bi}.{mod}.{proj}", f"g_dB.{bi}.{mod}.{proj}"]
    grads = torch.autograd.grad(loss, leaves)

    out = {}
    # head-gate primitives: latent + sigma let the Mojo head recompute
    # hidden0/v_temb/v_embedded/v_prompt_ts/v_cos/v_sin and gate them.
    out["latent"] = inp["latent"].detach().to(torch.float32).cpu().contiguous()
    out["sigma"] = torch.tensor([SIGMA], dtype=torch.float32).cpu().contiguous()
    out["hidden0"] = hidden0.detach().to(torch.float32).cpu().contiguous()
    out["enc_hs"] = inp["enc_hs"].detach().to(torch.float32).cpu().contiguous()
    out["v_temb"] = v_temb.detach().to(torch.float32).cpu().contiguous()
    out["v_embedded"] = v_embedded.detach().to(torch.float32).cpu().contiguous()
    out["v_prompt_ts"] = v_prompt_ts.detach().to(torch.float32).cpu().contiguous()
    out["v_cos"] = inp["v_cos"].detach().to(torch.float32).cpu().contiguous()
    out["v_sin"] = inp["v_sin"].detach().to(torch.float32).cpu().contiguous()
    for bi in range(NUM_BLOCKS):
        for mod in LORA_MODULES:
            for proj in LORA_PROJS:
                A, B, _ = loras[bi][f"{mod}.{proj}"]
                out[f"lora.{bi}.{mod}.{proj}.A"] = A.detach().to(torch.float32).cpu().contiguous()
                out[f"lora.{bi}.{mod}.{proj}.B"] = B.detach().to(torch.float32).cpu().contiguous()
    out["d_pred"] = d_pred.cpu().contiguous()
    out["pred"] = pred.detach().to(torch.float32).cpu().contiguous()
    for name, g in zip(names, grads):
        gn = float(g.norm())
        if gn == 0.0:
            raise RuntimeError(f"degenerate (zero) reference grad: {name}")
        out[name] = g.detach().to(torch.float32).cpu().contiguous()
    print(f"[stack-oracle] d_input |g|={grads[0].norm():.6f}")

    if "--self" in sys.argv:
        x2 = hidden0_leaf
        for bi in range(NUM_BLOCKS):
            blk_inp = {
                "hs": x2, "enc_hs": inp["enc_hs"],
                "v_timestep": v_temb, "video_prompt_ts": v_prompt_ts,
                "v_cos": inp["v_cos"], "v_sin": inp["v_sin"],
            }
            x2 = vid.run_video_block(blocks[bi], blk_inp, loras[bi])
        pred2 = tail_process_output(
            x2, v_embedded, hw["scale_shift_table"],
            hw["proj_out.weight"], hw["proj_out.bias"], EPS)
        print(f"[self] pred cos={av.cos_sim(pred, pred2):.7f}")
        return

    save_file(out, OUT)
    print(f"[stack-oracle] dumped {len(out)} tensors -> {OUT}")


if __name__ == "__main__":
    main()
