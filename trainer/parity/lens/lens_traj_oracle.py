#!/usr/bin/env python
"""Lens denoise-trajectory + full-image oracle (Serenity-only).

Runs Serenity's sampler math (LensSampler.__sample_base / lens.pipeline) from a
FIXED latent0 with FIXED text features, dumping latent0, latent_final, the features,
and the decoded image — so the Mojo sampler runs from byte-identical latent0+features
and is gated: latent_final cos>=0.999 (trajectory) + decoded-image PSNR (full sampler).

Isolates RNG: latent0 is dumped (no generator mismatch). Sequential to fit 24GB:
encode (free) -> transformer denoise (bf16) -> vae decode.

Run: /home/alex/ai-toolkit/venv/bin/python parity/lens/lens_traj_oracle.py
"""
import importlib.util, json, os, gc
import numpy as np, torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/home/alex/.serenity/models/microsoft_lens"
TF_PY = "/home/alex/vendor-refs/Lens/lens/transformer.py"
TE_PY = "/home/alex/vendor-refs/Lens/lens/text_encoder.py"
SEL=[5,11,17,23]; CROP=97; MAXLEN=512
H=W=512; VAE_SF=8; NCH=32; PATCH=2
STEPS=8; CFG=4.0; SEED=42


def load_mod(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def compute_empirical_mu(seq,n):
    a1,b1=8.73809524e-05,1.89833333; a2,b2=0.00016927,0.45666666
    if seq>4300: return float(a2*seq+b2)
    m200=a2*seq+b2; m10=a1*seq+b1; a=(m200-m10)/190.0; b=m200-200.0*a; return float(a*n+b)

def encode_caption(enc, tk, caption):
    CS=("Describe the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background.")
    AT="Need to generate one image according to the description."
    conv=[{"role":"system","content":CS,"thinking":None},{"role":"user","content":caption,"thinking":None},
          {"role":"assistant","thinking":AT,"content":""}]
    r=tk.apply_chat_template(conv, tokenize=False, add_generation_prompt=False).split("<|return|>")[0]
    o=tk(r, max_length=MAXLEN+CROP, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True)
    real=int(o.attention_mask.sum())
    with torch.no_grad():
        outs=enc.encode_layers(o.input_ids, o.attention_mask)
    return [f[:,CROP:real,:].float().cpu() for f in outs]


def main():
    dev="cuda"; dt=torch.bfloat16
    cap=open("/tmp/lens_prompt.txt").read().strip()
    from transformers import PreTrainedTokenizerFast
    tk=PreTrainedTokenizerFast.from_pretrained(CKPT+"/tokenizer")
    enc=load_mod(TE_PY,"te").LensGptOssEncoder.from_pretrained(CKPT+"/text_encoder",torch_dtype=dt,low_cpu_mem_usage=True,device_map="auto").eval()
    enc.set_selected_layers(SEL)
    pos=encode_caption(enc,tk,cap)
    neg=encode_caption(enc,tk,"")
    # pad/truncate neg to pos length for a fixed S_txt (sampler pads); use pos S_txt
    S=pos[0].shape[1]
    def fit(feats):
        out=[]
        for f in feats:
            if f.shape[1]>=S: out.append(f[:,:S,:])
            else: out.append(torch.cat([f, f.new_zeros(1,S-f.shape[1],f.shape[2])],dim=1))
        return out
    neg=fit(neg)
    del enc; gc.collect(); torch.cuda.empty_cache()

    h_lat,w_lat=H//VAE_SF//PATCH, W//VAE_SF//PATCH    # 32,32
    seq=h_lat*w_lat                                     # 1024
    def patchify(z):
        b,c,hh,ww=z.shape; return z.view(b,c,hh//2,2,ww//2,2).permute(0,1,3,5,2,4).reshape(b,c*4,hh//2,ww//2)
    def pack(z):
        b,c,hh,ww=z.shape; return z.reshape(b,c,hh*ww).permute(0,2,1)
    g=torch.Generator().manual_seed(SEED)
    lat0=torch.randn(1,NCH,H//VAE_SF,W//VAE_SF,generator=g,dtype=torch.float32)
    packed0=pack(patchify(lat0)).to(dev,dt)            # [1,1024,128]

    from diffusers import FlowMatchEulerDiscreteScheduler
    sched=FlowMatchEulerDiscreteScheduler.from_pretrained(CKPT+"/scheduler")
    mu=compute_empirical_mu(seq,STEPS)
    sigmas=np.linspace(1.0,1.0/STEPS,STEPS)
    sched.set_timesteps(sigmas=sigmas, device=dev, mu=mu)
    timesteps=sched.timesteps

    TF=load_mod(TF_PY,"tf").LensTransformer2DModel.from_pretrained(CKPT+"/transformer",torch_dtype=dt).to(dev).eval()
    pf=[f.to(dev,dt) for f in pos]; nf=[f.to(dev,dt) for f in neg]
    pm=torch.ones(1,S,dtype=torch.bool,device=dev); nm=torch.ones(1,S,dtype=torch.bool,device=dev)
    img_shapes=[(1,h_lat,w_lat)]
    latent=packed0.clone()
    with torch.no_grad():
        for i,t in enumerate(timesteps):
            ts=(t/1000.0).expand(1).to(dev)
            cond=TF(hidden_states=latent, encoder_hidden_states=pf, encoder_hidden_states_mask=pm, timestep=ts, img_shapes=img_shapes)
            unc =TF(hidden_states=latent, encoder_hidden_states=nf, encoder_hidden_states_mask=nm, timestep=ts, img_shapes=img_shapes)
            comb=unc+CFG*(cond-unc)
            cn=torch.norm(cond,dim=-1,keepdim=True); mn=torch.norm(comb,dim=-1,keepdim=True).clamp_min(1e-12)
            pred=comb*(cn/mn)
            latent=sched.step(pred, t, latent, return_dict=False)[0]
    latent_final=latent.float().cpu()
    del TF; gc.collect(); torch.cuda.empty_cache()

    # decode tail (unpack -> unscale -> unpatchify -> vae.decode) for the image
    from diffusers import AutoencoderKLFlux2
    vae=AutoencoderKLFlux2.from_pretrained(CKPT+"/vae",torch_dtype=torch.float32).to(dev).eval()
    bn_m=vae.bn.running_mean.float().view(1,-1,1,1); bn_s=torch.sqrt(vae.bn.running_var.float()+float(vae.config.batch_norm_eps)).view(1,-1,1,1)
    def unpack(z,hh,ww):
        b,s,c=z.shape; return z.reshape(b,hh,ww,c).permute(0,3,1,2)
    def unpatchify(z):
        b,c,hh,ww=z.shape; return z.reshape(b,c//4,2,2,hh,ww).permute(0,1,4,2,5,3).reshape(b,c//4,hh*2,ww*2)
    with torch.no_grad():
        z=unpack(latent_final.to(dev),h_lat,w_lat); z=z*bn_s+bn_m; z=unpatchify(z)
        img=vae.decode(z,return_dict=False)[0].float().cpu()

    save_file({"packed0":packed0.float().cpu().contiguous(),"latent_final":latent_final.contiguous(),
               "image":img.contiguous(),
               "pf_0":pos[0].contiguous(),"pf_1":pos[1].contiguous(),"pf_2":pos[2].contiguous(),"pf_3":pos[3].contiguous(),
               "nf_0":neg[0].contiguous(),"nf_1":neg[1].contiguous(),"nf_2":neg[2].contiguous(),"nf_3":neg[3].contiguous()},
              os.path.join(HERE,"traj_ref.safetensors"))
    json.dump({"H":H,"W":W,"steps":STEPS,"cfg":CFG,"seed":SEED,"S_txt":S,"h_lat":h_lat,"w_lat":w_lat,
               "image_shape":list(img.shape),"image_mean":float(img.mean()),"image_std":float(img.std()),
               "latent_final_mean":float(latent_final.mean()),"latent_final_std":float(latent_final.std())},
              open(os.path.join(HERE,"traj_ref_meta.json"),"w"),indent=2)
    print("TRAJ ORACLE OK")
    print(f"  steps={STEPS} cfg={CFG} S_txt={S}  latent_final mean {float(latent_final.mean()):.5f} std {float(latent_final.std()):.5f}")
    print(f"  image {list(img.shape)} mean {float(img.mean()):.4f} std {float(img.std()):.4f}")


if __name__=="__main__":
    main()
