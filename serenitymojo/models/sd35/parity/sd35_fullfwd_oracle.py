#!/usr/bin/env python3
# Full-model FORWARD parity oracle for the sd3.5-medium Mojo stack.
# Loads the REAL diffusers SD3Transformer2DModel (single_file), runs ONE forward
# on the real cached latent/text/pooled at a fixed sigma, and dumps:
#   ff_ref      [N_IMG,64]  = diffusers velocity in OUTPUT order (ph,pw,c) =
#                             diffusers proj_out/unpatchify == Mojo final_layer order
#   ff_noisy    [N_IMG,64]  = packed noisy input in INPUT order (c,ph,pw) = conv order
# NOTE the input/output feature orders DIFFER (conv (c,ph,pw) vs proj_out (ph,pw,c)).
#   ff_txt/ff_pooled        = encoder + pooled inputs
# The Mojo gate (sd35_fullfwd_parity via the smoke) compares its stack forward
# output to ff_ref. bf16 floor expected -> PASS at cos >= 0.99.
#
# Run:
#   /home/alex/OneTrainer/venv/bin/python serenitymojo/models/sd35/parity/sd35_fullfwd_oracle.py
import struct, os, torch
from safetensors import safe_open
from diffusers import SD3Transformer2DModel

CK="/home/alex/.serenity/models/checkpoints/sd3.5_medium.safetensors"
CACHE="/home/alex/datasets/andrsd35_sd35_cache/10.safetensors"
REF=os.path.dirname(os.path.abspath(__file__))
LAT_C, LAT_H, LAT_W, PATCH = 16, 128, 128, 2
HT, WT = LAT_H//PATCH, LAT_W//PATCH      # 64,64
N_IMG = HT*WT                             # 4096
VAE_SHIFT, VAE_SCALE = 0.0609, 1.5305
SIGMA = 0.5

# INPUT pack: (c,ph,pw) feature order — matches x_embedder Conv2d weight [D,C,kh,kw].
def pack_in(lat):
    out=torch.empty(N_IMG, LAT_C*PATCH*PATCH, dtype=lat.dtype)
    t=0
    for ih in range(HT):
        for iw in range(WT):
            f=0
            for c in range(LAT_C):
                for ph in range(PATCH):
                    for pw in range(PATCH):
                        out[t,f]=lat[c, ih*PATCH+ph, iw*PATCH+pw]; f+=1
            t+=1
    return out

# OUTPUT pack: (ph,pw,c) feature order — matches diffusers proj_out / unpatchify
# (reshape(...,p,p,C); einsum nhwpqc->nchpwq), == the Mojo final_layer.linear order.
def pack_out(lat):
    out=torch.empty(N_IMG, PATCH*PATCH*LAT_C, dtype=lat.dtype)
    t=0
    for ih in range(HT):
        for iw in range(WT):
            f=0
            for ph in range(PATCH):
                for pw in range(PATCH):
                    for c in range(LAT_C):
                        out[t,f]=lat[c, ih*PATCH+ph, iw*PATCH+pw]; f+=1
            t+=1
    return out

def W(name,t):
    flat=t.detach().reshape(-1).to(torch.float32).numpy()
    open(os.path.join(REF,name+".bin"),"wb").write(struct.pack("<%df"%flat.size,*flat.tolist()))
    print("wrote",name,tuple(t.shape))

def main():
    torch.manual_seed(0)
    with safe_open(CACHE,"pt") as f:
        latent=f.get_tensor("latent").float()        # [1,16,128,128]
        txt=f.get_tensor("text_embedding").float()   # [1,154,4096]
        pooled=f.get_tensor("pooled").float()        # [1,2048]
    lat_scaled=(latent - VAE_SHIFT)*VAE_SCALE
    # deterministic uniform[-2,2) noise matching the Mojo smoke's _noise (LCG)
    st=20260609 & 0xFFFFFFFFFFFFFFFF
    noise=torch.empty(N_IMG*LAT_C*PATCH*PATCH)
    for i in range(noise.numel()):
        st=(st*6364136223846793005+1442695040888963407)&0xFFFFFFFFFFFFFFFF
        noise[i]=((st>>40)*(1.0/16777216.0)-0.5)*2.0
    # noise is in PACKED order; unpack to [16,128,128] to build the full noisy latent
    noise_full=torch.zeros(LAT_C,LAT_H,LAT_W); t=0
    for ih in range(HT):
        for iw in range(WT):
            f=0
            for c in range(LAT_C):
                for ph in range(PATCH):
                    for pw in range(PATCH):
                        noise_full[c,ih*PATCH+ph,iw*PATCH+pw]=noise[t*64+f]; f+=1
            t+=1
    noisy_full=(noise_full*SIGMA + lat_scaled[0]*(1.0-SIGMA)).unsqueeze(0)  # [1,16,128,128]

    print("loading diffusers SD3.5-medium ...")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    m=SD3Transformer2DModel.from_single_file(CK, torch_dtype=torch.float32).eval().to(dev)
    timestep=torch.tensor([SIGMA*1000.0], device=dev)   # Mojo feeds sigma*1000 to t_embedder
    with torch.no_grad():
        out=m(hidden_states=noisy_full.to(dev), encoder_hidden_states=txt.to(dev),
              pooled_projections=pooled.to(dev), timestep=timestep, return_dict=True).sample
    out=out.float().cpu()[0]    # [16,128,128] velocity
    W("ff_ref", pack_out(out))      # [N_IMG,64]
    W("ff_noisy", pack_in(noisy_full[0]))
    W("ff_txt", txt[0])         # [154,4096]
    W("ff_pooled", pooled[0])   # [2048]
    print("DONE  (sigma=%.3f, timestep=%.1f)"%(SIGMA, SIGMA*1000.0))

if __name__=="__main__": main()
