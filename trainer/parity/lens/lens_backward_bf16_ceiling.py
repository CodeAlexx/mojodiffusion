#!/usr/bin/env python
"""Is the txt-stream backward gap a Mojo bug or just the BF16 ceiling?

The Mojo backward runs BF16; backward_grad_ref.safetensors is F32. If BF16 itself
(torch, same math) drops the low-magnitude txt d_B direction to ~0.98 vs F32, then
the Mojo matching F32 at ~0.98 is CORRECT and the cos>=0.999-vs-F32 bar is the wrong
reference (dtype bad-reference trap). If torch-BF16-vs-F32 stays ~0.999 on txt, the
Mojo's 0.98 is a real txt-path bug.

This recomputes d_B in BF16 (same hooks/formula as lens_backward_oracle.py, same A
from the F32 dump) and reports per-slot cos(bf16_dB, f32_dB).

Run: /home/alex/Serenity/venv/bin/python parity/lens/lens_backward_bf16_ceiling.py
"""
import importlib.util
import json
import os

import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/home/alex/.serenity/models/microsoft_lens"
LENS_TF_PY = "/home/alex/vendor-refs/Lens/lens/transformer.py"
SLOT_MODULES = ["attn.img_qkv","attn.txt_qkv","attn.to_out.0","attn.to_add_out",
                "img_mlp.w1","img_mlp.w2","img_mlp.w3","txt_mlp.w1","txt_mlp.w2","txt_mlp.w3"]


def cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    d = (a.norm() * b.norm())
    return float((a @ b) / d) if d > 0 else (1.0 if a.norm()==0 and b.norm()==0 else 0.0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16
    spec = importlib.util.spec_from_file_location("lens_transformer", LENS_TF_PY)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.LensTransformer2DModel.from_pretrained(CKPT, subfolder="transformer", torch_dtype=dt).to(dev).eval()
    nblk = m.config.num_layers
    meta = json.load(open(os.path.join(HERE, "meta.json")))
    bmeta = json.load(open(os.path.join(HERE, "backward_grad_meta.json")))
    SCALE = bmeta["scale"]
    ref = load_file(os.path.join(HERE, "backward_grad_ref.safetensors"))

    hidden = load_file(os.path.join(HERE,"dit_fwd_in_hidden.safetensors"))["x"].to(dev, dt)
    txt = [load_file(os.path.join(HERE,f"dit_fwd_in_txt_{i}.safetensors"))["x"].to(dev, dt) for i in range(4)]
    mask = load_file(os.path.join(HERE,"dit_fwd_in_mask.safetensors"))["x"].bool().to(dev)
    timestep = load_file(os.path.join(HERE,"dit_fwd_in_timestep.safetensors"))["x"].to(dev, dt)
    img_shapes = [tuple(meta["img_shapes"][0])]
    target_v = ref["target_v"].to(dev, dt)

    mods = dict(m.named_modules())
    order = [f"transformer_blocks.{b}.{rel}" for b in range(nblk) for rel in SLOT_MODULES]
    cap_x, cap_g, handles = {}, {}, []
    for name in order:
        lin = mods[name]
        handles.append(lin.register_forward_hook(lambda mo,i,o,nm=name: cap_x.__setitem__(nm, i[0].detach().reshape(-1,i[0].shape[-1]).float())))
        handles.append(lin.register_full_backward_hook(lambda mo,gi,go,nm=name: cap_g.__setitem__(nm, go[0].detach().reshape(-1,go[0].shape[-1]).float())))

    out = m(hidden_states=hidden, encoder_hidden_states=txt, encoder_hidden_states_mask=mask,
            timestep=timestep, img_shapes=img_shapes)
    loss = ((out.float() - target_v.float())**2).mean()
    loss.backward()
    for h in handles: h.remove()

    per_slot = {s: [] for s in range(10)}
    dump = {}
    ceiling = []   # per-adapter cos(bf16,f32) — the best a BF16 impl can do vs F32
    for idx, name in enumerate(order):
        slot = idx % 10
        A = ref[f"A_{idx}"].float()
        f32_dB = ref[f"dB_{idx}"].float()
        gl = cap_g.get(name)
        if gl is None:
            bf16_dB = torch.zeros_like(f32_dB)
        else:
            x = cap_x[name]
            A = A.to(x.device)
            bf16_dB = (SCALE * (gl.t() @ (x @ A.t()))).cpu()
        c = cos(bf16_dB, f32_dB)
        per_slot[slot].append(c)
        ceiling.append(c)
        dump[f"dB_bf16_{idx}"] = bf16_dB.float().contiguous()
    from safetensors.torch import save_file
    save_file(dump, os.path.join(HERE, "backward_grad_ref_bf16.safetensors"))
    json.dump({"ceiling_cos": ceiling, "n_adapters": len(order),
               "note": "per-adapter cos(torch_bf16_dB, torch_f32_dB); the BF16 accuracy ceiling vs F32"},
              open(os.path.join(HERE, "backward_grad_bf16_meta.json"), "w"))

    print(f"BF16 CEILING (torch bf16 d_B vs torch f32 d_B), device={dev}")
    names = ["img_qkv","txt_qkv","to_out","to_add_out","img_w1","img_w2","img_w3","txt_w1","txt_w2","txt_w3"]
    for s in range(10):
        vals = [v for v in per_slot[s] if v==v]
        mn = min(vals); mean = sum(vals)/len(vals)
        tag = "IMG " if s in (0,2,4,5,6) else "TXT "
        print(f"  {tag}slot {s} {names[s]:11s} mean cos(bf16,f32) = {mean:.5f}  min = {mn:.5f}")


if __name__ == "__main__":
    main()
