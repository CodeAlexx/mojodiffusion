"""CAPSTONE oracle — the full keyframe VISION path, CPU, real weights.

preprocess -> vision tower -> splice at <|image_pad|> -> deepstack add.

── WHAT IS AND IS NOT COVERED, STATED UP FRONT ─────────────────────────────
COVERED (all CPU): the image preprocessor, the 27-block vision tower on the real
FL2VA weights, the derivation of the splice map from the real presentation, the
embed SUBSTITUTION at the pad rows, and the deepstack ADD at those same rows for
language layers 0/1/2.

NOT COVERED: the 50 decoder layers themselves. `minimax_h3_encode_conditioning_
streamed_depth` takes a DeviceContext, and the GPU is off-limits; separately, 50
layers of a 32B model at f32 is not CPU-feasible at any useful speed. Those
layers are gated elsewhere. What this adds is the seam nothing else covers: that
the tower's rows land on the RIGHT sequence positions.

That is the failure this exists to catch. A splice that is right in COUNT and
wrong in PLACEMENT yields a perfectly-shaped conditioning, no shape assertion
anywhere fires, and the render is quietly wrong.

Canvas 768x768 — the SMALLEST canvas `resolve_canvas_size` can emit (768 short
edge, aspect 1), chosen so the Mojo host tower's O(P^2) attention stays tractable
while remaining a REAL production geometry, not a toy.

Run:  CUDA_VISIBLE_DEVICES="" python3 scripts/minimax_h3_keyframe_vision_e2e_oracle.py
Writes: output/minimax_h3_keyframe/keyframe_vision_e2e_ref.safetensors
"""
import json, os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

H3="/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA"
TE=os.path.join(H3,"text_encoder"); OUT="/home/alex/mojodiffusion/output/minimax_h3_keyframe"
PREFIX="model.visual."; HID=5120
CANVAS_H, CANVAS_W = 768, 768
PROMPT=("For the target video, at 0.00 seconds into the target video, <Picture 1>"
        " (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description:"
        " [Shot 1] Live-action, cinematic, a test.\n\noverall_soundscape: Room tone."
        "\n\nnon_diegetic_music: N/A")

def load_tower():
    cfg=json.load(open(os.path.join(TE,"config.json")))
    v=Qwen3VLVisionConfig(**cfg["vision_config"]); mdl=Qwen3VLVisionModel(v)
    wm=json.load(open(os.path.join(TE,"model.safetensors.index.json")))["weight_map"]
    vis={k:s for k,s in wm.items() if k.startswith(PREFIX)}
    st={}
    for shard in sorted(set(vis.values())):
        with safe_open(os.path.join(TE,shard),framework="pt",device="cpu") as f:
            for k,s in vis.items():
                if s==shard: st[k[len(PREFIX):]]=f.get_tensor(k).to(torch.float32)
    mdl.load_state_dict(st,strict=True)
    return mdl.eval().to(torch.float32), v

def main():
    os.makedirs(OUT,exist_ok=True)
    proc=AutoProcessor.from_pretrained(os.path.join(H3,"processor"))
    tok=AutoTokenizer.from_pretrained(os.path.join(H3,"processor"))
    ip=proc.image_processor; M=ip.merge_size
    pad_id=tok.convert_tokens_to_ids("<|image_pad|>")

    rng=np.random.default_rng(4242)
    arr=rng.integers(0,256,(CANVAS_H,CANVAS_W,3),dtype=np.uint8)
    img=Image.fromarray(arr,"RGB")
    vis=ip(images=[img],return_tensors="pt")
    px=vis["pixel_values"].to(torch.float32); gthw=vis["image_grid_thw"]
    t,h,w=gthw[0].tolist(); ntok=t*h*w//(M*M)
    print(f"canvas {CANVAS_W}x{CANVAS_H} -> grid {t}x{h}x{w}, {t*h*w} patches, {ntok} tokens")

    # presentation ids (the vendor's t2va/fl2va loop, encoders.py:152-169)
    ids=tok(f"<Picture 1>: ",add_special_tokens=False)["input_ids"]
    ids+= ([tok.convert_tokens_to_ids("<|vision_start|>")]+[pad_id]*ntok
           +[tok.convert_tokens_to_ids("<|vision_end|>")])
    ids+= tok(PROMPT,add_special_tokens=False)["input_ids"]
    ids_t=torch.tensor(ids)
    # get_placeholder_mask's own criterion: input_ids == image_token_id
    mask=(ids_t==pad_id)
    positions=torch.nonzero(mask).flatten()
    print(f"presentation: {len(ids)} rows, {int(mask.sum())} image_pad rows")
    assert int(mask.sum())==ntok, (int(mask.sum()), ntok)

    tower,vcfg=load_tower()
    with torch.no_grad():
        embeds,deep=tower(px,gthw)
    print(f"tower: embeds {tuple(embeds.shape)}, {len(deep)} deepstack taps")

    # SPLICE: masked_scatter, exactly what Qwen3VLModel does with the mask above.
    g=torch.Generator().manual_seed(99)
    inputs_embeds=torch.randn(len(ids),HID,generator=g,dtype=torch.float32)
    before=inputs_embeds.clone()
    spliced=before.clone()
    spliced[mask,:]=embeds.to(spliced.dtype)

    # DEEPSTACK: _deepstack_process — ADD at the visual rows only.
    hs=torch.randn(len(ids),HID,generator=g,dtype=torch.float32)
    ds_before=hs.clone()
    ds_after=[]
    cur=ds_before.clone()
    for k in range(len(deep)):
        cur=cur.clone()
        cur[mask,:]=cur[mask,:].clone()+deep[k].to(cur.dtype)
        ds_after.append(cur.clone())

    ten={
        "image":torch.from_numpy(arr.copy()),
        "grid_thw":gthw.to(torch.int32),
        "token_ids":ids_t.to(torch.int32),
        "pad_positions":positions.to(torch.int32),
        "pixel_values":px,
        "vision_embeds":embeds.float(),
        "inputs_embeds_before":before,
        "inputs_embeds_spliced":spliced,
        "hidden_before":ds_before,
    }
    for k in range(len(deep)):
        ten[f"deepstack_{k}"]=deep[k].float()
        ten[f"hidden_after_{k}"]=ds_after[k]
    p=os.path.join(OUT,"keyframe_vision_e2e_ref.safetensors")
    save_file({k:v.contiguous() for k,v in ten.items()},p)
    print("wrote",p)
    for k,v in ten.items(): print(f"  {k:24s} {tuple(v.shape)}")

if __name__=="__main__": main()
