#!/usr/bin/env python
"""Lens real-data predict->loss oracle (Serenity-only). REQUIRED parity gate.

Replicates BaseLensSetup.predict (pr-1510) EXACTLY on a real image+caption, with
the deterministic path (timestep=499) and a fixed noise tensor, then dumps the
byte-identical inputs the Mojo needs + the reference flow-matching loss. The Mojo
predict on the SAME (scaled_noisy_packed, text_features, target) must match.

Sequential to fit 24GB: encode caption (free) -> vae-encode image (free) ->
transformer forward (bf16, matches Mojo storage dtype).

Run: /home/alex/ai-toolkit/venv/bin/python parity/lens/lens_loss_oracle.py
"""
import importlib.util, json, os, gc
import torch
from safetensors.torch import save_file
from PIL import Image
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/home/alex/.serenity/models/microsoft_lens"
LENS_TF_PY = "/home/alex/vendor-refs/Lens/lens/transformer.py"
LENS_TE_PY = "/home/alex/vendor-refs/Lens/lens/text_encoder.py"
IMG = os.path.join(HERE, "gen", "lens_mojo.png")   # a real image (pixels)
SEL = [5, 11, 17, 23]; CROP = 97; MAXLEN = 512
NUM_TRAIN_TS = 1000; TIMESTEP = 499                 # int(1000*0.5)-1, deterministic
SEED = 7777


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    dev = "cuda"
    dt = torch.bfloat16
    # ---- 1) caption -> 4 layer features (crop 97) ----
    cap = open("/tmp/lens_prompt.txt").read().strip()
    from transformers import PreTrainedTokenizerFast
    tk = PreTrainedTokenizerFast.from_pretrained(CKPT + "/tokenizer")
    CHAT_SYSTEM=("Describe the image by detailing the color, shape, size, texture, "
      "quantity, text, spatial relationships of the objects and background.")
    CHAT_AT="Need to generate one image according to the description."
    conv=[{"role":"system","content":CHAT_SYSTEM,"thinking":None},
          {"role":"user","content":cap,"thinking":None},
          {"role":"assistant","thinking":CHAT_AT,"content":""}]
    rendered=tk.apply_chat_template(conv, tokenize=False, add_generation_prompt=False).split("<|return|>")[0]
    enc_in=tk(rendered, max_length=MAXLEN+CROP, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True)
    TE=load_mod(LENS_TE_PY,"lens_te").LensGptOssEncoder
    enc=TE.from_pretrained(CKPT+"/text_encoder", torch_dtype=dt, low_cpu_mem_usage=True, device_map="auto").eval()
    enc.set_selected_layers(SEL)
    ids=enc_in.input_ids; am=enc_in.attention_mask
    with torch.no_grad():
        feats=enc.encode_layers(ids, am)               # 4 x [1,609,2880]
    real=int(am.sum().item())
    feats=[f[:, CROP:real, :].float().cpu() for f in feats]   # crop chat prefix, drop pad
    del enc; gc.collect(); torch.cuda.empty_cache()
    S_txt=feats[0].shape[1]

    # ---- 2) real image -> VAE latent ----
    from diffusers import AutoencoderKLFlux2
    vae=AutoencoderKLFlux2.from_pretrained(CKPT+"/vae", torch_dtype=torch.float32).to(dev).eval()
    im=Image.open(IMG).convert("RGB")
    x=torch.from_numpy(np.asarray(im)).float().permute(2,0,1)[None]/127.5-1.0   # [1,3,H,W] in [-1,1]
    with torch.no_grad():
        post=vae.encode(x.to(dev)).latent_dist
        latent=post.mode().float().cpu()                # [1,32,H/8,W/8]
    bn_mean=vae.bn.running_mean.float().cpu()
    bn_std=torch.sqrt(vae.bn.running_var.float()+float(vae.config.batch_norm_eps)).cpu()
    del vae; gc.collect(); torch.cuda.empty_cache()

    # ---- 3) predict (BaseLensSetup.predict, deterministic) ----
    def patchify(z):
        b,c,h,w=z.shape
        z=z.view(b,c,h//2,2,w//2,2).permute(0,1,3,5,2,4).reshape(b,c*4,h//2,w//2); return z
    def unpatchify(z):
        b,c,h,w=z.shape
        z=z.reshape(b,c//4,2,2,h,w).permute(0,1,4,2,5,3).reshape(b,c//4,h*2,w*2); return z
    def pack(z):
        b,c,h,w=z.shape; return z.reshape(b,c,h*w).permute(0,2,1)
    def unpack(z,h,w):
        b,s,c=z.shape; return z.reshape(b,h,w,c).permute(0,3,1,2)

    lat_p=patchify(latent)                              # [1,128,h/2,w/2]
    H,W=lat_p.shape[-2], lat_p.shape[-1]
    scaled=(lat_p - bn_mean.view(1,-1,1,1))/bn_std.view(1,-1,1,1)
    g=torch.Generator().manual_seed(SEED)
    noise=torch.randn(scaled.shape, generator=g, dtype=torch.float32)
    sigma=(TIMESTEP+1)/NUM_TRAIN_TS                     # 0.5
    scaled_noisy=noise*sigma + scaled*(1.0-sigma)
    target=noise - scaled                               # flow target
    packed_in=pack(scaled_noisy)                        # [1, H*W, 128]

    TF=load_mod(LENS_TF_PY,"lens_tf").LensTransformer2DModel
    m=TF.from_pretrained(CKPT+"/transformer", torch_dtype=dt).to(dev).eval()
    ts=torch.tensor([TIMESTEP/1000.0], dtype=torch.float32)
    with torch.no_grad():
        pred=m(hidden_states=packed_in.to(dev,dt),
               encoder_hidden_states=[f.to(dev,dt) for f in feats],
               encoder_hidden_states_mask=torch.ones(1,S_txt,dtype=torch.bool,device=dev),
               timestep=ts.to(dev), img_shapes=[(1,H,W)])
    pred_unpacked=unpatchify(unpack(pred.float().cpu(), H, W))
    target_unpacked=unpatchify(target)
    loss=float(((pred_unpacked - target_unpacked)**2).mean())

    save_file({"packed_in": packed_in.contiguous(),
               "target": target.contiguous(),
               "feat_0": feats[0].contiguous(),"feat_1": feats[1].contiguous(),
               "feat_2": feats[2].contiguous(),"feat_3": feats[3].contiguous()},
              os.path.join(HERE,"loss_ref.safetensors"))
    json.dump({"loss": loss, "timestep": TIMESTEP, "sigma": sigma, "seed": SEED,
               "H_packed": H, "W_packed": W, "S_txt": S_txt,
               "latent_shape": list(latent.shape), "img": IMG,
               "note": "BaseLensSetup.predict, deterministic t=499, real-image latent + real caption"},
              open(os.path.join(HERE,"loss_ref_meta.json"),"w"), indent=2)
    print("LOSS ORACLE OK")
    print(f"  Serenity predict loss = {loss:.6f}   (t={TIMESTEP} sigma={sigma} S_txt={S_txt} packed {H}x{W})")


if __name__ == "__main__":
    main()
