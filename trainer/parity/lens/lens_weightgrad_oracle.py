#!/usr/bin/env python
"""Lens full-finetune weight-grad oracle (Serenity-only).

For full fine-tune the grad wrt each base Linear weight is dW_l = g_l^T @ x_l
(g_l = grad wrt the Linear's output, x_l = its input) — the SAME g_l/x_l the LoRA
backward parity already validated. This dumps torch-autograd dW for the per-block
target Linears so the Mojo finetune backward (dW accumulation) can be gated.

Uses the SAME fixed forward inputs as the LoRA backward oracle (dit_fwd_in_*),
same MSE(out, target_v) loss → identical g_l/x_l. dW_l computed via fwd/bwd hooks.

Run: /home/alex/ai-toolkit/venv/bin/python parity/lens/lens_weightgrad_oracle.py
"""
import importlib.util, json, os
import torch
from safetensors.torch import load_file, save_file

HERE=os.path.dirname(os.path.abspath(__file__)); CKPT="/home/alex/.serenity/models/microsoft_lens"
TF_PY="/home/alex/vendor-refs/Lens/lens/transformer.py"
SLOTS=["attn.img_qkv","attn.txt_qkv","attn.to_out.0","attn.to_add_out",
       "img_mlp.w1","img_mlp.w2","img_mlp.w3","txt_mlp.w1","txt_mlp.w2","txt_mlp.w3"]


def main():
    s=importlib.util.spec_from_file_location("tf",TF_PY); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    mod=m.LensTransformer2DModel.from_pretrained(CKPT+"/transformer",torch_dtype=torch.float32)
    nblk=mod.config.num_layers
    meta=json.load(open(HERE+"/meta.json"))
    hidden=load_file(HERE+"/dit_fwd_in_hidden.safetensors")["x"].float()
    txt=[load_file(HERE+f"/dit_fwd_in_txt_{i}.safetensors")["x"].float() for i in range(4)]
    mask=load_file(HERE+"/dit_fwd_in_mask.safetensors")["x"].bool()
    ts=load_file(HERE+"/dit_fwd_in_timestep.safetensors")["x"].float()
    img_shapes=[tuple(meta["img_shapes"][0])]
    ref=load_file(HERE+"/backward_grad_ref.safetensors")  # reuse same target_v
    target_v=ref["target_v"].float()

    names=[f"transformer_blocks.{b}.{r}" for b in range(nblk) for r in SLOTS]
    md=dict(mod.named_modules()); capx={}; capg={}; hs=[]
    for n in names:
        lin=md[n]
        hs.append(lin.register_forward_hook(lambda mo,i,o,nm=n: capx.__setitem__(nm,i[0].detach().reshape(-1,i[0].shape[-1]).float())))
        hs.append(lin.register_full_backward_hook(lambda mo,gi,go,nm=n: capg.__setitem__(nm,go[0].detach().reshape(-1,go[0].shape[-1]).float())))
    out=mod(hidden_states=hidden,encoder_hidden_states=txt,encoder_hidden_states_mask=mask,timestep=ts,img_shapes=img_shapes)
    loss=((out-target_v)**2).mean(); loss.backward()
    for h in hs: h.remove()

    dump={}; stats=[]; zero=[]
    for idx,n in enumerate(names):
        g=capg.get(n)
        if g is None:
            # last-block txt-post: no grad path (img-only head)
            w=md[n].weight; dW=torch.zeros_like(w).float(); zero.append(n)
        else:
            x=capx[n]; dW=g.t() @ x        # [out,in] = dL/dW
        dump[f"dW_{idx}"]=dW.contiguous(); stats.append(float(dW.abs().max()))
    save_file(dump,HERE+"/weightgrad_ref.safetensors")
    json.dump({"n":len(names),"slots":SLOTS,"n_blocks":nblk,"loss":float(loss),
               "dW_absmax_min":min(stats),"dW_absmax_max":max(stats),"zero_adapters":zero,
               "note":"dW = g^T @ x per target Linear (full-finetune weight grad); same inputs as backward_grad_ref"},
              open(HERE+"/weightgrad_meta.json","w"),indent=2)
    print("WEIGHTGRAD ORACLE OK")
    print(f"  loss {float(loss):.6f}  targets {len(names)}  dW_absmax [{min(stats):.3e},{max(stats):.3e}]  zero {len(zero)}")


if __name__=="__main__":
    main()
