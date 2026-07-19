#!/usr/bin/env python
"""Lens backward grad oracle (Serenity-only) — torch-autograd reference for the
hand-chained Mojo LoRA backward.

At B=0 the LoRA forward delta is 0 (forward == base, already cos-0.9998 verified),
and the only nonzero LoRA grad is d_B. For target Linear l:
    d_B_l = scale * g_l^T @ (x_l @ A_l^T)
where x_l = the Linear's input activation, g_l = grad wrt the Linear's output.
This is EXACTLY Mojo lora_backward's d_B (linear_backward(d_y*scale, x@A^T, B=0).d_w).
We capture x_l (forward hook) and g_l (full backward hook) from torch autograd on
the real-weights base transformer, compute d_B_l analytically, and dump A_l + d_B_l
+ the fixed MSE target so the Mojo grad-parity smoke uses byte-identical A/inputs.

Comparing Mojo's per-adapter d_B to this validates the whole hand-chained backward
chain AND the (block,slot) -> Linear mapping.

Run: /home/alex/Serenity/venv/bin/python parity/lens/lens_backward_oracle.py
"""
import importlib.util
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/home/alex/.serenity/models/microsoft_lens"
LENS_TF_PY = "/home/alex/vendor-refs/Lens/lens/transformer.py"

RANK = 16
ALPHA = 16.0
SCALE = ALPHA / RANK

# (block, slot) order — MUST match lensLoraTargets.mojo block-major/slot-minor.
SLOT_MODULES = [
    "attn.img_qkv", "attn.txt_qkv", "attn.to_out.0", "attn.to_add_out",
    "img_mlp.w1", "img_mlp.w2", "img_mlp.w3",
    "txt_mlp.w1", "txt_mlp.w2", "txt_mlp.w3",
]


def load_dit():
    spec = importlib.util.spec_from_file_location("lens_transformer", LENS_TF_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.LensTransformer2DModel.from_pretrained(CKPT, subfolder="transformer", torch_dtype=torch.float32)
    return m.eval()


def main():
    m = load_dit()
    nblk = m.config.num_layers

    # reuse the forward oracle's fixed inputs
    hidden = load_file(os.path.join(HERE, "dit_fwd_in_hidden.safetensors"))["x"].float()
    txt = [load_file(os.path.join(HERE, f"dit_fwd_in_txt_{i}.safetensors"))["x"].float() for i in range(4)]
    mask = load_file(os.path.join(HERE, "dit_fwd_in_mask.safetensors"))["x"].bool()
    timestep = load_file(os.path.join(HERE, "dit_fwd_in_timestep.safetensors"))["x"].float()
    meta = json.load(open(os.path.join(HERE, "meta.json")))
    img_shapes = [tuple(meta["img_shapes"][0])]

    # enumerate the 480 target Linears in canonical order; build per-adapter A and hooks
    order, A_by_name, in_by_name, out_by_name = [], {}, {}, {}
    mod_by_name = dict(m.named_modules())
    for b in range(nblk):
        for slot, rel in enumerate(SLOT_MODULES):
            name = f"transformer_blocks.{b}.{rel}"
            lin = mod_by_name[name]
            in_f, out_f = lin.in_features, lin.out_features
            idx = b * len(SLOT_MODULES) + slot
            rng = np.random.default_rng(1000 + idx)
            A = (rng.standard_normal((RANK, in_f)) * (1.0 / np.sqrt(in_f))).astype(np.float32)
            order.append(name); A_by_name[name] = A
            in_by_name[name] = in_f; out_by_name[name] = out_f

    cap_x, cap_g = {}, {}
    handles = []
    for name in order:
        lin = mod_by_name[name]
        def fwd_hook(mod, inp, out, nm=name):
            cap_x[nm] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        def bwd_hook(mod, gin, gout, nm=name):
            cap_g[nm] = gout[0].detach().reshape(-1, gout[0].shape[-1]).float()
        handles.append(lin.register_forward_hook(fwd_hook))
        handles.append(lin.register_full_backward_hook(bwd_hook))

    out = m(hidden_states=hidden, encoder_hidden_states=txt, encoder_hidden_states_mask=mask,
            timestep=timestep, img_shapes=img_shapes)
    g = torch.Generator().manual_seed(777)
    target_v = torch.randn(out.shape, generator=g, dtype=torch.float32)
    loss = ((out - target_v) ** 2).mean()
    loss.backward()
    for h in handles:
        h.remove()

    # d_B_l = scale * g_l^T @ (x_l @ A_l^T)
    dump = {"target_v": target_v.contiguous()}
    dB_stats = []
    zero_adapters = []
    for idx, name in enumerate(order):
        A = torch.from_numpy(A_by_name[name])               # [rank, in]
        out_f = out_by_name[name]
        gl = cap_g.get(name)                                  # [M, out] or None (no grad path)
        if gl is None:
            # last-block txt-post adapters: txt output discarded by the img-only head
            dB = torch.zeros((out_f, RANK), dtype=torch.float32)
            zero_adapters.append(name)
        else:
            x = cap_x[name]                                  # [M, in]
            xa = x @ A.t()                                   # [M, rank]
            dB = SCALE * (gl.t() @ xa)                       # [out, rank]
        dump[f"A_{idx}"] = A.contiguous()
        dump[f"dB_{idx}"] = dB.contiguous()
        dB_stats.append(float(dB.abs().max()))

    save_file(dump, os.path.join(HERE, "backward_grad_ref.safetensors"))
    bmeta = {
        "rank": RANK, "alpha": ALPHA, "scale": SCALE, "n_adapters": len(order),
        "slot_modules": SLOT_MODULES, "n_blocks": nblk,
        "target_seed": 777, "loss": float(loss),
        "in_features": [in_by_name[n] for n in order],
        "out_features": [out_by_name[n] for n in order],
        "dB_absmax_min": min(dB_stats), "dB_absmax_max": max(dB_stats),
        "n_zero_dB": int(sum(1 for s in dB_stats if s == 0.0)),
        "zero_adapters": zero_adapters,
    }
    json.dump(bmeta, open(os.path.join(HERE, "backward_grad_meta.json"), "w"), indent=2)
    print("BACKWARD ORACLE OK")
    print(f"  loss {bmeta['loss']:.6f}  adapters {len(order)}  "
          f"dB_absmax range [{bmeta['dB_absmax_min']:.3e}, {bmeta['dB_absmax_max']:.3e}]  "
          f"zero-dB adapters {bmeta['n_zero_dB']}")


if __name__ == "__main__":
    main()
