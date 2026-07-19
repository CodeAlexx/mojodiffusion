#!/usr/bin/env python3
"""Generate ACE-Step-1.5 turbo DiT parity oracles (block-0 + full-forward).

Runs the REAL canonical model (acestep-v15-turbo checkpoint's
modeling_acestep_v15_turbo.py) eager-mode bf16 on GPU and dumps inputs +
expected outputs + decoder weights into safetensors fixtures the Mojo gates
load.

Block-0 at S=64 and full-forward at T=200 (patched seq SP=100) keep every
self-attn sliding window (128) a no-op so sdpa_nomask is exact.

Usage:
    cd /home/alex/ACE-Step-1.5/checkpoints/acestep-v15-turbo
    python3 /home/alex/mojodiffusion/serenitymojo/models/dit/parity/gen_acestep_oracle.py
"""
import sys, importlib.util
import torch
from safetensors.torch import load_file, save_file

CKPT = "/home/alex/ACE-Step-1.5/checkpoints/acestep-v15-turbo"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity"
sys.path.insert(0, CKPT)


def _load_mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, f"{CKPT}/{fname}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


cfg_mod = _load_mod("ace_cfg", "configuration_acestep_v15.py")
mdl = _load_mod("ace_mdl", "modeling_acestep_v15_turbo.py")
config = cfg_mod.AceStepConfig.from_pretrained(CKPT)
config._attn_implementation = "eager"
dev, dt = "cuda", torch.bfloat16
all_w = load_file(f"{CKPT}/model.safetensors")
dec = {k[len("decoder."):]: v.to(dev).to(dt)
       for k, v in all_w.items() if k.startswith("decoder.")}


def gen_block0():
    torch.manual_seed(0)
    layer = mdl.AceStepDiTLayer(config, 0).to(dev).to(dt).eval()
    sd = {k[len("layers.0."):]: v for k, v in dec.items()
          if k.startswith("layers.0.")}
    layer.load_state_dict(sd, strict=False)
    B, S, H, L = 1, 64, config.hidden_size, 48
    half = config.head_dim // 2
    torch.manual_seed(1)
    hidden = torch.randn(B, S, H, device=dev, dtype=dt)
    enc = torch.randn(B, L, H, device=dev, dtype=dt)
    temb = torch.randn(B, 6, H, device=dev, dtype=dt)
    rot = mdl.Qwen3RotaryEmbedding(config).to(dev)
    pos = torch.arange(S, device=dev).unsqueeze(0)
    cos, sin = rot(hidden, pos)
    cm = mdl.create_4d_mask
    sm = cm(seq_len=S, dtype=dt, device=dev, attention_mask=None,
            sliding_window=config.sliding_window, is_sliding_window=True,
            is_causal=False)
    em = cm(seq_len=max(S, L), dtype=dt, device=dev, attention_mask=None,
            sliding_window=None, is_sliding_window=False,
            is_causal=False)[:, :, :S, :L]
    with torch.no_grad():
        out = layer(hidden, (cos, sin), temb, attention_mask=sm,
                    encoder_hidden_states=enc, encoder_attention_mask=em)[0]
    fx = {"hidden": hidden.float().cpu(), "enc": enc.float().cpu(),
          "temb": temb.float().cpu(),
          "cos": cos[0, :, :half].float().cpu(),
          "sin": sin[0, :, :half].float().cpu(),
          "expected": out.float().cpu()}
    for k, v in sd.items():
        fx["w_" + k] = v.float().cpu()
    fx = {k: v.contiguous() for k, v in fx.items()}
    save_file(fx, f"{OUT}/acestep_block0_fixture.safetensors")
    print("block0 saved; out std", out.float().std().item())


def gen_full():
    model = mdl.AceStepDiTModel(config).to(dev).to(dt).eval()
    model.load_state_dict(dec, strict=False)
    B, T, L = 1, 200, 48
    torch.manual_seed(7)
    hidden = torch.randn(B, T, config.audio_acoustic_hidden_dim,
                         device=dev, dtype=dt)
    context = torch.randn(B, T, 128, device=dev, dtype=dt)
    enc = torch.randn(B, L, config.hidden_size, device=dev, dtype=dt)
    ts = torch.rand(B, device=dev, dtype=dt) * 0.8 + 0.1
    with torch.no_grad():
        out = model(hidden_states=hidden, timestep=ts, timestep_r=ts.clone(),
                    attention_mask=None, encoder_hidden_states=enc,
                    encoder_attention_mask=None, context_latents=context,
                    use_cache=False)[0]
    fx = {"hidden": hidden.float().cpu(), "context": context.float().cpu(),
          "enc": enc.float().cpu(), "timestep": ts.float().cpu(),
          "timestep_r": ts.float().cpu(), "expected": out.float().cpu()}
    for k, v in dec.items():
        fx["w_" + k] = v.float().cpu()
    fx = {k: v.contiguous() for k, v in fx.items()}
    save_file(fx, f"{OUT}/acestep_full_fixture.safetensors")
    print("full saved; out std", out.float().std().item())


if __name__ == "__main__":
    gen_block0()
    gen_full()
