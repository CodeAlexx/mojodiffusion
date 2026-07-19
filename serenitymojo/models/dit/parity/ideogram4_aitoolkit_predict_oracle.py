#!/usr/bin/env python3
# DEV-ONLY parity oracle for the serenitymojo Ideogram-4 FULL FORWARD STACK
# ("predict": input_proj -> 34 blocks -> final_layer -> velocity) + MRoPE.
#
# Source of truth: ai-toolkit's PRODUCTION Ideogram-4 implementation
#   /home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4/src/transformer.py
#   /home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4/src/pipeline.py
#   /home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4/ideogram4.py
# This REPLACES the earlier (INVALID) ideogram4-ref forward oracle
#   serenitymojo/models/dit/parity/ideogram4_predict_oracle.py (imports
#   /home/alex/ideogram4-ref). It is the FULL-composition sibling of the verified
#   per-block ai-toolkit oracle (ideogram4_aitoolkit_oracle.py): same checkpoint,
#   same fp8->bf16 fold, same geometry, same packing scheme.
#
# What it captures (so the mojo full forward can be gated element-wise):
#   * velocity          : predict_velocity(...) toolkit velocity = -image_velocity
#   * transformer_out   : the raw transformer(...) output (f32, B,L,128)
#   * block{0,16,33}_out: per-block hidden states (localizes embedders vs blocks)
#   * mrope_cos/sin     : Ideogram4MRoPE.forward(position_ids) (f32)
# plus the inputs (x, llm_full, t, model_t, indicator/segment/position f32) so the
# mojo gate feeds byte-identical tensors.
#
# Dtype: bf16 — the production dtype. _dequantize_fp8_state_dict folds the fp8
# .weight + sibling .weight_scale to bf16 (w.float()*scale[:,None] -> bf16) and
# casts everything else to bf16; the model runs bf16. This is NOT "lowering to a
# bf16 port" — bf16 IS production here.
#
# IMPORTANT MRoPE provenance: ai-toolkit's _load_transformer re-registers
#   inv_freq = 1/theta^(arange(0,head_dim,2)/head_dim)  in torch.float32
# (transformer.py:95-98 builds it f32 too) and NEVER casts the rotary_emb buffer
# to bf16. So production MRoPE runs FULLY float32. The earlier ideogram4-ref mrope
# oracle cast the whole module .to(bf16) (bf16-rounded inv_freq). At the
# IMAGE_POSITION_OFFSET=65536 positions this f32-vs-bf16 inv_freq choice changes
# cos/sin materially. This oracle captures the FAITHFUL ai-toolkit f32 cos/sin.
#
# Inputs: NON-DEGENERATE (randn, fixed seed). GH=GW=16, NTEXT=4 -> L=260, t=0.7.
# All tokens share segment_id=1 (single packed sample) so the native SDPA block
# mask is all-True; native (SDPA) backend only (flash_attn not installed, and the
# production default backend is "native").
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/dit/parity/ideogram4_aitoolkit_predict_oracle.py
#
# Output: serenitymojo/models/dit/parity/ideogram4_aitoolkit_predict.safetensors

import os
import sys
import gc

import torch
from safetensors import safe_open
from safetensors.torch import save_file

AITK = "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4/src"
sys.path.insert(0, AITK)
# pipeline.py uses package-relative imports (`from .transformer import ...`), so
# import the package and pull both modules from it (transformer == i4 alias).
import importlib  # noqa: E402

_PKG_PARENT = "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4"
sys.path.insert(0, _PKG_PARENT)

# ai-toolkit PRODUCTION transformer + the predict_velocity packing (authoritative).
i4 = importlib.import_module("src.transformer")
predict_velocity = importlib.import_module("src.pipeline").predict_velocity

ROOT = "/home/alex/.serenity/models/ideogram-4-fp8/transformer"
CKPT = os.path.join(ROOT, "diffusion_pytorch_model.safetensors")
OUT_DIR = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity"
OUT = os.path.join(OUT_DIR, "ideogram4_aitoolkit_predict.safetensors")

DEV = torch.device("cuda")
DT = torch.bfloat16  # production dtype (fp8 -> bf16)

# Parity input geometry (mirrors the block oracle + chunk6 inputs so the gate
# inputs line up). GH=GW=16 -> NIMG=256, NTEXT=4 -> L=260.
GH = GW = 16
NIMG = GH * GW          # 256
NTEXT = 4
L = NTEXT + NIMG        # 260
TVAL = 0.7
SEED = 1234

CAPTURE_BLOCKS = (0, 16, 33)


def _dequant_fp8(w_fp8, scale):
    # Production fold (_dequantize_fp8_state_dict): runs on the work device (GPU),
    # w_fp8.float()*scale[:,None] -> bf16.
    w = w_fp8.to(DEV, torch.float32)
    s = scale.to(DEV, torch.float32)
    return (w * s.unsqueeze(1)).to(DT)


def load_transformer():
    """Load the FULL transformer from the fp8 checkpoint, dequantized to bf16
    exactly as the ai-toolkit production loader (_dequantize_fp8_state_dict ->
    load_state_dict(assign=True)); then re-register inv_freq in float32 (the
    production _load_transformer step)."""
    cfg = i4.Ideogram4Config()
    with torch.device("meta"):
        m = i4.Ideogram4Transformer2DModel(cfg)

    sd = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        raw = {k: f.get_tensor(k) for k in f.keys()}

    for k, v in raw.items():
        if k.endswith(".weight_scale"):
            continue
        scale_key = k + "_scale"
        if k.endswith(".weight") and scale_key in raw:
            sd[k] = _dequant_fp8(v, raw[scale_key])
        elif v.is_floating_point():
            sd[k] = v.to(DEV, DT)
        else:
            sd[k] = v.to(DEV)

    # assign=True swaps the (meta) params for the GPU-resident dequantized tensors.
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, f"unexpected keys: {unexpected}"
    # inv_freq is the only non-persistent buffer absent from the checkpoint (and
    # is still a meta tensor until re-registered below).
    assert all("inv_freq" in mk for mk in missing), f"missing (non inv_freq): {missing}"

    # Production _load_transformer: rebuild inv_freq in FLOAT32 (NOT bf16).
    head_dim = cfg.emb_dim // cfg.num_heads
    inv_freq = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    m.rotary_emb.register_buffer("inv_freq", inv_freq.to(DEV), persistent=False)
    m.eval()
    m.set_attention_backend("native")
    print(f"[oracle] full transformer loaded ({DT}, native); "
          f"inv_freq dtype {m.rotary_emb.inv_freq.dtype}; "
          f"VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return cfg, m


def build_inputs(cfg):
    """NON-DEGENERATE flow inputs at the parity geometry. latents/noise/t in the
    ai-toolkit (B,128,gh,gw) latent layout that predict_velocity consumes."""
    g = torch.Generator(device=DEV).manual_seed(SEED)
    # latents (B,128,gh,gw) and noise (same shape); flow add_noise + target.
    latents = torch.randn(1, cfg.in_channels, GH, GW, device=DEV,
                          dtype=torch.float32, generator=g)
    noise = torch.randn(1, cfg.in_channels, GH, GW, device=DEV,
                        dtype=torch.float32, generator=g)
    t = torch.full((1,), TVAL, device=DEV, dtype=torch.float32)
    t01 = t.view(-1, 1, 1, 1)
    noisy = (1.0 - t01) * latents + t01 * noise          # toolkit flow add_noise
    target = noise - latents                              # get_loss_target

    # Qwen3-VL features (B, Lt, llm_dim) — non-degenerate randn (a real run feeds
    # the encoder taps; the transformer forward only sees these features).
    llm_features = torch.randn(1, NTEXT, cfg.llm_features_dim, device=DEV,
                               dtype=torch.float32, generator=g).to(DT)
    text_mask = torch.ones(1, NTEXT, dtype=torch.long, device=DEV)
    return latents, noise, t, noisy, target, llm_features, text_mask


def main():
    torch.manual_seed(SEED)
    cfg, m = load_transformer()
    latents, noise, t, noisy, target, llm_features, text_mask = build_inputs(cfg)
    print(f"[oracle] inputs: latents{tuple(latents.shape)} llm{tuple(llm_features.shape)} "
          f"t={TVAL} L={L} (NTEXT={NTEXT}+NIMG={NIMG})")

    # ── per-block capture hooks (block 0/16/33 hidden-state outputs) ──
    cap = {}

    def mk_hook(idx):
        def hook(mod, inp, out):
            cap[idx] = out.detach().float().cpu()
        return hook

    handles = [m.layers[i].register_forward_hook(mk_hook(i)) for i in CAPTURE_BLOCKS]

    # ── capture the raw transformer(...) output too (transformer_out) ──
    # predict_velocity calls transformer(...); hook the top-level module forward.
    top_out = {}

    def top_hook(mod, inp, out):
        top_out["out"] = out.detach().float().cpu()

    th = m.register_forward_hook(top_hook)

    # ── run predict_velocity (the production packing + velocity) ──
    # t is the TOOLKIT flow time (1=noise). predict_velocity feeds model_t=1-t and
    # negates the model output -> toolkit velocity (noise - clean).
    with torch.no_grad():
        velocity = predict_velocity(
            transformer=m,
            latents=noisy,            # (B,128,gh,gw) noisy latent
            t=t,                      # toolkit flow time
            llm_features=llm_features,
            text_mask=text_mask,
        )  # (B,128,gh,gw) toolkit velocity

    for h in handles:
        h.remove()
    th.remove()

    transformer_out = top_out["out"]   # (B, L, 128) raw model output (f32)
    print(f"[oracle] velocity{tuple(velocity.shape)} std {velocity.float().std():.5f}  "
          f"transformer_out{tuple(transformer_out.shape)} std {transformer_out.std():.5f}")
    for i in CAPTURE_BLOCKS:
        print(f"  block{i}_out{tuple(cap[i].shape)} std {cap[i].std():.5f}")

    # ── MRoPE cos/sin (rebuild the EXACT packed-sequence position_ids the model
    # used inside predict_velocity, then call the production f32 MRoPE) ──
    # (predict_velocity builds these internally; reconstruct to dump cos/sin.)
    num_image = GH * GW
    text_pos = (text_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    text_pos_3d = text_pos.unsqueeze(-1).expand(-1, -1, 3)
    h_idx = torch.arange(GH, device=DEV).view(-1, 1).expand(GH, GW).reshape(-1)
    w_idx = torch.arange(GW, device=DEV).view(1, -1).expand(GH, GW).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + i4.IMAGE_POSITION_OFFSET
    image_pos_3d = image_pos.unsqueeze(0).expand(1, -1, -1)
    position_ids = torch.cat([text_pos_3d, image_pos_3d], dim=1)   # (1, L, 3)

    # rebuild indicator/segment too (for the mojo gate inputs).
    indicator = torch.zeros(1, L, dtype=torch.long, device=DEV)
    indicator[:, :NTEXT] = text_mask.long() * i4.LLM_TOKEN_INDICATOR
    indicator[:, NTEXT:] = i4.OUTPUT_IMAGE_INDICATOR
    segment_ids = torch.ones(1, L, dtype=torch.long, device=DEV)

    with torch.no_grad():
        cos, sin = m.rotary_emb(position_ids)             # (1, L, head_dim) f32
    print(f"[oracle] mrope cos{tuple(cos.shape)} cos_std {float(cos.std()):.5f} "
          f"sin_std {float(sin.std()):.5f}  (inv_freq dtype {m.rotary_emb.inv_freq.dtype})")

    # ── also build the packed x / llm_full the transformer consumed (for the mojo
    # full-forward inputs), matching predict_velocity's packing exactly. ──
    image_tokens = noisy.permute(0, 2, 3, 1).reshape(1, num_image, cfg.in_channels)
    x_packed = torch.cat(
        [torch.zeros(1, NTEXT, cfg.in_channels, device=DEV, dtype=image_tokens.dtype),
         image_tokens], dim=1)
    llm_full = torch.cat(
        [llm_features,
         torch.zeros(1, num_image, llm_features.shape[-1], device=DEV, dtype=DT)], dim=1)
    model_t = 1.0 - t

    fx = {
        # inputs (so the mojo gate feeds byte-identical tensors)
        "in.latents": latents.float().cpu(),
        "in.noise": noise.float().cpu(),
        "in.noisy": noisy.float().cpu(),
        "in.target": target.float().cpu(),
        "in.t": t.float().cpu(),
        "in.model_t": model_t.float().cpu(),
        "in.llm_features": llm_features.float().cpu(),
        "in.llm_full": llm_full.float().cpu(),
        "in.x_packed": x_packed.float().cpu(),
        "in.text_mask": text_mask.to(torch.int32).cpu(),
        "in.position_ids": position_ids.to(torch.int32).cpu(),
        "in.position_ids_f32": position_ids.to(torch.float32).cpu(),
        "in.indicator": indicator.to(torch.int32).cpu(),
        "in.indicator_f32": indicator.to(torch.float32).cpu(),
        "in.segment_ids": segment_ids.to(torch.int32).cpu(),
        "in.segment_ids_f32": segment_ids.to(torch.float32).cpu(),
        # captures
        "out.velocity": velocity.float().cpu(),
        "out.transformer_out": transformer_out,
        "out.mrope_cos": cos.float().cpu(),
        "out.mrope_sin": sin.float().cpu(),
    }
    for i in CAPTURE_BLOCKS:
        fx[f"out.block{i}_out"] = cap[i]

    fx = {k: v.contiguous() for k, v in fx.items()}
    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(fx, OUT)
    print(f"[oracle] saved {len(fx)} tensors -> {OUT}")
    print(f"  geometry: L={L} (NTEXT={NTEXT}+NIMG={NIMG}), t={TVAL}, "
          f"velocity_std={velocity.float().std():.4f}, target_std={target.std():.4f}")

    del m
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
