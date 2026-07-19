# DEV-ONLY Wave-0 parity oracle for the Ideogram-4 Mojo port. NEVER shipped.
# Dumps per-chunk fixtures from /home/alex/ideogram4-ref on LOCAL .serenity fp8
# weights, GPU bf16, one model at a time. Saves each stage to its own file so a
# later-stage OOM cannot lose earlier fixtures. No network (offline env enforced).
import sys, os, json, gc, torch
sys.path.insert(0, "/home/alex/ideogram4-ref/src")
from safetensors.torch import load_file, save_file
from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from ideogram4.quantized_loading import (is_fp8_state_dict, swap_linears_to_fp8, load_fp8_state_dict)
from ideogram4.constants import (LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR, IMAGE_POSITION_OFFSET)

ROOT="/home/alex/.serenity/models/ideogram-4-fp8"
OUT="/home/alex/mojodiffusion/serenitymojo/models/dit/parity"
dev=torch.device("cuda"); dt=torch.bfloat16
def vram(): return torch.cuda.memory_allocated()/1e9
def free():
    gc.collect(); torch.cuda.empty_cache()

# shared tiny packed seq (grid 16x16 = 256 img tokens, 4 text tokens)
GH=GW=16; NIMG=GH*GW; NTEXT=4; L=NTEXT+NIMG; TVAL=0.7
def build_inputs(cfg):
    torch.manual_seed(0)
    llm=torch.randn(1,L,cfg.llm_features_dim,device=dev,dtype=torch.float32)
    x=torch.randn(1,L,cfg.in_channels,device=dev,dtype=torch.float32)
    t=torch.full((1,),TVAL,device=dev,dtype=torch.float32)
    pos=torch.zeros(1,L,3,dtype=torch.long,device=dev)
    seg=torch.full((1,L),SEQUENCE_PADDING_INDICATOR,dtype=torch.long,device=dev)
    ind=torch.zeros(1,L,dtype=torch.long,device=dev)
    tp=torch.arange(NTEXT,device=dev); pos[0,:NTEXT]=torch.stack([tp,tp,tp],1)
    hi=torch.arange(GH,device=dev).view(-1,1).expand(GH,GW).reshape(-1)
    wi=torch.arange(GW,device=dev).view(1,-1).expand(GH,GW).reshape(-1)
    ti=torch.zeros_like(hi); ipos=torch.stack([ti,hi,wi],1)+IMAGE_POSITION_OFFSET
    pos[0,NTEXT:]=ipos; ind[0,:NTEXT]=LLM_TOKEN_INDICATOR; ind[0,NTEXT:]=OUTPUT_IMAGE_INDICATOR; seg[0,:]=1
    return llm,x,t,pos,seg,ind

# ── Stage A: transformer (chunks 1,2,3,5,6) ──────────────────────────────────
def stage_A():
    sd=load_file(f"{ROOT}/transformer/diffusion_pytorch_model.safetensors")
    cfg=Ideogram4Config()
    m=Ideogram4Transformer(cfg); m.to(dt)
    swap_linears_to_fp8(m, sd, compute_dtype=dt); load_fp8_state_dict(m, sd, device=dev, dtype=dt); m.eval()
    print(f"[A] transformer loaded, VRAM {vram():.2f}GB")
    fx={}
    # chunk1: fp8 per-row dequant of a representative Linear
    lin=dict(m.named_modules())["layers.0.attention.qkv"]
    fx["chunk1.qkv_dequant_expected"]=(lin.weight.to(torch.float32)*lin.weight_scale.to(torch.float32).unsqueeze(1)).cpu()
    llm,x,t,pos,seg,ind=build_inputs(cfg)
    # chunk2: MRoPE cos/sin
    with torch.no_grad(): cos,sin=m.rotary_emb(pos)
    fx["chunk2.mrope_cos"]=cos.float().cpu(); fx["chunk2.mrope_sin"]=sin.float().cpu()
    # chunk3: t-embedding (EmbedScalar) for several t
    tvec=torch.tensor([0.0,0.25,0.5,0.7,1.0],device=dev,dtype=torch.float32)
    with torch.no_grad(): temb=m.t_embedding(tvec)
    fx["chunk3.t_values"]=tvec.float().cpu(); fx["chunk3.t_embed"]=temb.float().cpu()
    # chunk5: one block I/O via forward hook on layers[0]
    cap={}
    def hook(mod,inp,out):
        cap["in_x"]=inp[0].detach().float().cpu()
        # block forward kwargs: segment_ids, cos, sin, adaln_input (positional after x)
        cap["out"]=out.detach().float().cpu()
    h=m.layers[0].register_forward_hook(hook)
    with torch.no_grad():
        vel=m(llm_features=llm,x=x,t=t,position_ids=pos,segment_ids=seg,indicator=ind)
    h.remove()
    fx["chunk5.block0_in_x"]=cap["in_x"]; fx["chunk5.block0_out"]=cap["out"]
    # also dump the block's other inputs (recompute adaln_input + cos/sin already have)
    with torch.no_grad():
        tc=m.t_embedding(t); tc=tc.unsqueeze(1) if t.dim()==1 else tc
        adaln_input=torch.nn.functional.silu(m.adaln_proj(tc))
    fx["chunk5.adaln_input"]=adaln_input.float().cpu()
    # chunk6: full DiT inputs + velocity
    fx["chunk6.in_llm"]=llm.float().cpu(); fx["chunk6.in_x"]=x.float().cpu(); fx["chunk6.in_t"]=t.float().cpu()
    fx["chunk6.in_position_ids"]=pos.to(torch.int32).cpu()
    fx["chunk6.in_segment_ids"]=seg.to(torch.int32).cpu()
    fx["chunk6.in_indicator"]=ind.to(torch.int32).cpu()
    fx["chunk6.out_velocity"]=vel.float().cpu()
    save_file(fx, f"{OUT}/ideogram4_fx_transformer.safetensors")
    print(f"[A] saved {len(fx)} tensors: {list(fx.keys())}")
    del m, sd; free(); print(f"[A] freed, VRAM {vram():.2f}GB")

def stage_P():
    # Re-dump the deterministic packed-seq position_ids as F32 (model-free).
    cfg=Ideogram4Config()
    _,_,_,pos,seg,ind=build_inputs(cfg)
    fx={"chunk6.in_position_ids_f32":pos.to(torch.float32).cpu(),
        "chunk6.in_segment_ids_f32":seg.to(torch.float32).cpu(),
        "chunk6.in_indicator_f32":ind.to(torch.float32).cpu()}
    save_file(fx, f"{OUT}/ideogram4_fx_inputs_f32.safetensors")
    print(f"[P] saved f32 position/segment/indicator, L={pos.shape[1]}")


def stage_E():
    import gc as _gc
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from transformers.masking_utils import create_causal_mask
    from ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
    from ideogram4.latent_norm import get_latent_norm
    from ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals
    from ideogram4.autoencoder import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict
    H=W=256; STEPS=8; CFG=7.0; SEED=0; PROMPT="a red cube on a white table"
    patch=16; gh=H//patch; gw=W//patch; nimg=gh*gw
    # --- Qwen 13-tap ---
    te=f"{ROOT}/text_encoder"
    tok=AutoTokenizer.from_pretrained(f"{ROOT}/tokenizer")
    qcfg=AutoConfig.from_pretrained(te); qm=AutoModel.from_config(qcfg)
    qsd=load_file(f"{te}/model.safetensors"); swap_linears_to_fp8(qm,qsd,compute_dtype=dt)
    load_fp8_state_dict(qm,qsd,device=dev,dtype=dt,assign=True,strict=False); qm.eval()
    rendered=tok.apply_chat_template([{"role":"user","content":[{"type":"text","text":PROMPT}]}],add_generation_prompt=True,tokenize=False)
    ids=tok(rendered,return_tensors="pt",add_special_tokens=False)["input_ids"].to(dev); nt=ids.shape[1]
    lm=qm.language_model; ie=lm.embed_tokens(ids)
    p2=torch.arange(nt,device=dev).unsqueeze(0); p4=p2[None].expand(4,1,-1); tpos=p4[0]; mpos=p4[1:]
    cm=create_causal_mask(config=lm.config,inputs_embeds=ie,attention_mask=torch.ones_like(ids),past_key_values=None,position_ids=tpos)
    pe=lm.rotary_emb(ie,mpos); taps=set(QWEN3_VL_ACTIVATION_LAYERS); cap={}; hs=ie
    with torch.no_grad():
        for i,ly in enumerate(lm.layers):
            hs=ly(hs,attention_mask=cm,position_ids=tpos,past_key_values=None,position_embeddings=pe)
            if isinstance(hs,tuple): hs=hs[0]
            if i in taps: cap[i]=hs
    sel=[cap[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    text_features=torch.permute(torch.stack(sel,0),(1,2,3,0)).reshape(1,nt,-1).float()
    del qm,qsd; _gc.collect(); torch.cuda.empty_cache()
    # --- build inputs (single prompt, left-pad=0): [text][image] ---
    from ideogram4.constants import IMAGE_POSITION_OFFSET as OFF, LLM_TOKEN_INDICATOR as LLM, OUTPUT_IMAGE_INDICATOR as IMG, SEQUENCE_PADDING_INDICATOR as PAD
    total=nt+nimg
    pos=torch.zeros(1,total,3,dtype=torch.long,device=dev); seg=torch.full((1,total),PAD,dtype=torch.long,device=dev); ind=torch.zeros(1,total,dtype=torch.long,device=dev)
    tp=torch.arange(nt,device=dev); pos[0,:nt]=torch.stack([tp,tp,tp],1)
    hi=torch.arange(gh,device=dev).view(-1,1).expand(gh,gw).reshape(-1); wi=torch.arange(gw,device=dev).view(1,-1).expand(gh,gw).reshape(-1)
    ipos=torch.stack([torch.zeros_like(hi),hi,wi],1)+OFF; pos[0,nt:]=ipos
    ind[0,:nt]=LLM; ind[0,nt:]=IMG; seg[0,:]=1
    dtt=dt
    llm_full=torch.cat([text_features.to(dtt),torch.zeros(1,nimg,text_features.shape[-1],device=dev,dtype=dtt)],1)
    neg_pos=pos[:,nt:]; neg_seg=seg[:,nt:]; neg_ind=ind[:,nt:]
    neg_llm=torch.zeros(1,nimg,text_features.shape[-1],device=dev,dtype=dtt)
    # --- transformers ---
    def load_tf(sub):
        sd=load_file(f"{ROOT}/{sub}/diffusion_pytorch_model.safetensors")
        from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
        m=Ideogram4Transformer(Ideogram4Config()); m.to(dt)
        swap_linears_to_fp8(m,sd,compute_dtype=dt); load_fp8_state_dict(m,sd,device="cpu",dtype=dt); m.eval(); return m
    cond=load_tf("transformer"); uncond=load_tf("unconditional_transformer")
    g=torch.Generator(device=dev); g.manual_seed(SEED)
    z0=torch.randn(1,nimg,128,dtype=torch.float32,device=dev,generator=g); z=z0.clone()
    tzpad=torch.zeros(1,nt,128,dtype=torch.float32,device=dev)
    sched=get_schedule_for_resolution((H,W),known_mean=0.5,std=1.0); si=make_step_intervals(STEPS).to(dev)
    with torch.no_grad():
        for i in range(STEPS-1,-1,-1):
            tv=float(sched(si[i+1].unsqueeze(0)).item()); sv=float(sched(si[i].unsqueeze(0)).item())
            t=torch.full((1,),tv,dtype=dt,device=dev)
            pz=torch.cat([tzpad,z],1)
            cond.to(dev)
            pv=cond(llm_features=llm_full,x=pz.to(dt),t=t,position_ids=pos,segment_ids=seg,indicator=ind)[:,nt:].clone()
            cond.to("cpu"); torch.cuda.empty_cache()
            uncond.to(dev)
            nv=uncond(llm_features=neg_llm,x=z.to(dt),t=t,position_ids=neg_pos,segment_ids=neg_seg,indicator=neg_ind).clone()
            uncond.to("cpu"); torch.cuda.empty_cache()
            v=CFG*pv+(1.0-CFG)*nv
            z=z+v.to(torch.float32)*(sv-tv)
    del cond,uncond; _gc.collect(); torch.cuda.empty_cache()
    # --- decode ---
    shift,scale=get_latent_norm(); shift=shift.to(dev); scale=scale.to(dev)
    zf=z*scale+shift
    ae_ch=zf.shape[-1]//(2*2)
    zr=zf.view(1,gh,gw,2,2,ae_ch).permute(0,5,1,3,2,4).contiguous().view(1,ae_ch,gh*2,gw*2)
    ae=AutoEncoder(AutoEncoderParams()); ae.load_state_dict(convert_diffusers_state_dict(load_file(f"{ROOT}/vae/diffusion_pytorch_model.safetensors"))); ae.to(device=dev,dtype=dt); ae.eval()
    with torch.no_grad(): dec=ae.decoder(zr.to(dt))
    img=dec.float().clamp(-1,1)
    fx={"z0":z0.cpu(),"final_z":z.cpu(),"llm_full":llm_full.float().cpu(),"pos_f32":pos.to(torch.float32).cpu(),
        "ind_f32":ind.to(torch.float32).cpu(),"neg_pos_f32":neg_pos.to(torch.float32).cpu(),"neg_ind_f32":neg_ind.to(torch.float32).cpu(),
        "final_latent":zr.float().cpu(),"decoded":img.cpu()}
    save_file(fx,f"{OUT}/ideogram4_fx_sampler.safetensors")
    json.dump({"H":H,"W":W,"steps":STEPS,"cfg":CFG,"seed":SEED,"prompt":PROMPT,"nt":int(nt),"nimg":int(nimg),
        "ids":ids[0].tolist(),"gh":gh,"gw":gw},open(f"{OUT}/ideogram4_fx_sampler_meta.json","w"),indent=2)
    print(f"[E] saved sampler fixture nt={nt} nimg={nimg} final_z std={float(z.std()):.4f} img std={float(img.std()):.4f}")


if __name__=="__main__":
    stage=sys.argv[1] if len(sys.argv)>1 else "A"
    if stage=="A": stage_A()

# ── Stage B: Qwen3-VL 13-tap (chunk7) ────────────────────────────────────────
def stage_B():
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from transformers.masking_utils import create_causal_mask
    from ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
    te=f"{ROOT}/text_encoder"
    tok=AutoTokenizer.from_pretrained(f"{ROOT}/tokenizer")
    cfg=AutoConfig.from_pretrained(te)
    model=AutoModel.from_config(cfg)
    sd=load_file(f"{te}/model.safetensors")
    swap_linears_to_fp8(model, sd, compute_dtype=dt)
    load_fp8_state_dict(model, sd, device=dev, dtype=dt, assign=True, strict=False)
    model.eval()
    print(f"[B] qwen3-vl loaded, VRAM {vram():.2f}GB")
    prompt="a red cube on a white table"
    rendered=tok.apply_chat_template([{"role":"user","content":[{"type":"text","text":prompt}]}],
        add_generation_prompt=True, tokenize=False)
    enc=tok(rendered, return_tensors="pt", add_special_tokens=False)
    ids=enc["input_ids"].to(dev); seq=ids.shape[1]
    lm=model.language_model
    inputs_embeds=lm.embed_tokens(ids)
    pos_2d=torch.arange(seq,device=dev).unsqueeze(0)
    p4=pos_2d[None,...].expand(4,pos_2d.shape[0],-1); text_pos=p4[0]; mrope_pos=p4[1:]
    attn=torch.ones_like(ids)
    cmask=create_causal_mask(config=lm.config, inputs_embeds=inputs_embeds, attention_mask=attn,
        past_key_values=None, position_ids=text_pos)
    pe=lm.rotary_emb(inputs_embeds, mrope_pos)
    taps=set(QWEN3_VL_ACTIVATION_LAYERS); cap={}; hs=inputs_embeds
    with torch.no_grad():
        for i,layer in enumerate(lm.layers):
            hs=layer(hs, attention_mask=cmask, position_ids=text_pos, past_key_values=None, position_embeddings=pe)
            if isinstance(hs,tuple): hs=hs[0]
            if i in taps: cap[i]=hs
    selected=[cap[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    stacked=torch.stack(selected,0)                  # (13,1,L,4096)
    feats=torch.permute(stacked,(1,2,3,0)).reshape(1,seq,-1)  # (1,L,53248)
    fx={"chunk7.token_ids":ids.to(torch.int32).cpu(), "chunk7.llm_features":feats.float().cpu()}
    for j,i in enumerate(QWEN3_VL_ACTIVATION_LAYERS):
        fx[f"chunk7.tap_{i}"]=selected[j].float().cpu()
    save_file(fx, f"{OUT}/ideogram4_fx_qwen.safetensors")
    json.dump({"prompt":prompt,"seq":int(seq),"taps":list(QWEN3_VL_ACTIVATION_LAYERS),
        "rendered":rendered}, open(f"{OUT}/ideogram4_fx_qwen_meta.json","w"), indent=2)
    print(f"[B] saved taps={len(selected)} seq={seq} feats={tuple(feats.shape)}")
    del model, sd; free(); print(f"[B] freed, VRAM {vram():.2f}GB")

# ── Stage C: Flux2 VAE decode (chunk8) ───────────────────────────────────────
def stage_C():
    from ideogram4.autoencoder import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict
    ae=AutoEncoder(AutoEncoderParams())
    sd=convert_diffusers_state_dict(load_file(f"{ROOT}/vae/diffusion_pytorch_model.safetensors"))
    ae.load_state_dict(sd); ae.to(device=dev,dtype=dt); ae.eval()
    print(f"[C] vae loaded, VRAM {vram():.2f}GB")
    torch.manual_seed(7)
    z=torch.randn(1,32,32,32,device=dev,dtype=torch.float32)   # [1,32,32,32] -> [1,3,256,256]
    with torch.no_grad():
        dec=ae.decoder(z.to(dt))
    fx={"chunk8.latent":z.float().cpu(), "chunk8.decoded":dec.float().cpu()}
    save_file(fx, f"{OUT}/ideogram4_fx_vae.safetensors")
    print(f"[C] decoded {tuple(dec.shape)} mean {dec.float().mean():.4f} std {dec.float().std():.4f}")
    del ae, sd; free(); print(f"[C] freed, VRAM {vram():.2f}GB")

# ── Stage D: logit-normal schedule scalars (chunk4) ──────────────────────────
def stage_D():
    from ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals
    H=W=1024; STEPS=8; MU=0.5; STD=1.0
    sched=get_schedule_for_resolution((H,W), known_mean=MU, std=STD)
    si=make_step_intervals(STEPS)
    vals=[float(sched(si[i].unsqueeze(0)).item()) for i in range(STEPS+1)]
    out={"height":H,"width":W,"num_steps":STEPS,"mu":MU,"std":STD,"known_resolution":[512,512],
        "schedule_mean":sched.mean,"logsnr_min":sched.logsnr_min,"logsnr_max":sched.logsnr_max,
        "step_intervals":[float(v) for v in si.tolist()],"schedule_values":vals}
    json.dump(out, open(f"{OUT}/ideogram4_fx_schedule.json","w"), indent=2)
    print(f"[D] schedule mean={sched.mean:.6f} vals[0..3]={vals[:4]}")

if __name__=="__main__":
    stage=sys.argv[1] if len(sys.argv)>1 else "A"
    {"A":stage_A,"B":stage_B,"C":stage_C,"D":stage_D,"P":stage_P,"E":stage_E}[stage]()


# ── Stage E: full end-to-end reference (256^2, CFG) for chunk 9 ───────────────