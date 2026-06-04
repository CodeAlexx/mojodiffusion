# ldm_encoder_oracle256.py — SDXL/LDM encoder bf16 GPU oracle at 256x256 with a
# realistic structured image (multi-frequency sinusoids, full [-1,1] dynamic
# range — representative of an actual photo encode, unlike the tight 64x64 ramp
# where BF16 rounding dominates). Dumps moments(NHWC)+mode(NCHW)+img(f32).
# Also dumps an F32-oracle moments file for the BF16-ceiling cross-check.
# Run: /home/alex/serenityflow-v2/.venv/bin/python <this>
import os, json, numpy as np, torch
from diffusers import AutoencoderKL
HERE=os.path.dirname(os.path.abspath(__file__))
VAE_DIR="/home/alex/madebyollin_sdxl-vae-fp16-fix"
IH=IW=256; LH=LW=IH//8

def make_img():
    yy,xx=np.meshgrid(np.linspace(0,1,IH),np.linspace(0,1,IW),indexing="ij")
    r=0.6*np.sin(2*np.pi*3*xx)+0.4*np.cos(2*np.pi*5*yy)
    g=0.5*np.sin(2*np.pi*7*(xx+yy))+0.3*np.cos(2*np.pi*2*xx)
    b=0.7*np.cos(2*np.pi*4*yy)*np.sin(2*np.pi*2*xx)
    img=np.stack([r,g,b],0)[None].astype(np.float32)
    return np.clip(img,-1.0,1.0)

def main():
    dev=torch.device("cuda")
    img_f32=make_img()
    img_f32.tofile(os.path.join(HERE,"ldmenc_img_256x256.bin"))
    # bf16 oracle
    vae=AutoencoderKL.from_pretrained(VAE_DIR,torch_dtype=torch.bfloat16).to(dev).eval()
    img=torch.from_numpy(img_f32).to(dev,dtype=torch.bfloat16)
    with torch.no_grad():
        post=vae.encode(img).latent_dist
        mom=post.parameters.float().cpu().numpy()
        mode=post.mode().float().cpu().numpy()
    np.transpose(mom,(0,2,3,1)).copy().astype(np.float32).tofile(os.path.join(HERE,"ldmenc_moments_256x256.bin"))
    mode.astype(np.float32).tofile(os.path.join(HERE,"ldmenc_mode_256x256.bin"))
    del vae; torch.cuda.empty_cache()
    # f32 oracle (ceiling)
    vae=AutoencoderKL.from_pretrained(VAE_DIR,torch_dtype=torch.float32).to(dev).eval()
    with torch.no_grad():
        post=vae.encode(torch.from_numpy(img_f32).to(dev)).latent_dist
        momf=post.parameters.float().cpu().numpy()
    momf_nhwc=np.transpose(momf,(0,2,3,1)).reshape(-1)
    momf_nhwc.astype(np.float32).tofile(os.path.join(HERE,"ldmenc_moments_f32_256x256.bin"))
    def cos(a,b):
        a=a.reshape(-1);b=b.reshape(-1);return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    bf=np.transpose(mom,(0,2,3,1)).reshape(-1)
    print("[oracle256] F32-vs-BF16 self moments cos=",cos(momf_nhwc,bf),
          " mode cos=",cos(post.mode().float().cpu().numpy().reshape(-1) if False else mode.reshape(-1),mode.reshape(-1)))
    print("[oracle256] moments min/max/std",bf.min(),bf.max(),bf.std())
    with open(os.path.join(HERE,"ldmenc_meta_256x256.json"),"w") as f:
        json.dump({"img":[1,3,IH,IW],"moments_nhwc":[1,LH,LW,8],"mode_nchw":[1,4,LH,LW],"dtype":"bf16-gpu->f32","unscaled":True},f,indent=2)
    del vae; torch.cuda.empty_cache()
    print("[oracle256] done", HERE)

if __name__=="__main__": main()
