#!/usr/bin/env python
# Oracle-1 for the NL-Diffusion-Image Mojo port (INFERENCE ONLY).
# Reference authority = NVIDIA creator source (trust_remote_code). NOT SerenityTrainer.
#
# Runs text_to_image(n_steps=1, guidance_scale=0, res=256) — a single-sample,
# no-CFG forward — with forward hooks on every T2I component, and dumps exact
# inputs+outputs so each Mojo chunk (A backbone layer, B full backbone, C image
# head, D VQ decoder) can be gated at cos>=0.999 against these tensors.
#
# Env: /home/alex/SerenityTrainer/venv/bin/python (torch 2.12 cu130, transformers
# 5.5.4, einops). flash_attn absent on sm_120 -> attn_implementation=sdpa.
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DEBUG_FIX_PADDING", "1")
os.environ.setdefault("NOT_ALWASY_DO_2DPOOL", "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerFast
from safetensors.torch import save_file

DIR = "/mnt/disk1/models/NL-Diffusion-Image"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"
SEED = 42

captured = {}   # name -> tensor (first occurrence only)
meta = {}       # name -> shape/dtype/notes

def _cpu(t):
    # move to CPU FIRST, then float32 (float32 on GPU would spike VRAM -> OOM)
    return t.detach().cpu().float().contiguous()

def save_once(name, t, note=""):
    if name in captured:
        return
    if not isinstance(t, torch.Tensor):
        return
    full_shape = list(t.shape)
    # cap huge tensors (e.g. gen_predictor logits [4096,131072]) to a row-slice for parity
    if t.dim() >= 2 and t.numel() > 50_000_000:
        t = t.reshape(-1, t.shape[-1])[:128].contiguous()
        note = (note + f" [SLICED first128 rows of {full_shape}]").strip()
    captured[name] = _cpu(t)
    meta[name] = {"shape": list(t.shape), "full_shape": full_shape, "dtype": str(t.dtype), "note": note}
    print(f"  [tap] {name:38s} {list(t.shape)} {t.dtype} {note}")

def mk_pre_hook(name):
    # capture the module's input (args[0]) and selected kwargs (position_embeddings)
    def hook(mod, args, kwargs):
        if args and isinstance(args[0], torch.Tensor):
            save_once(f"{name}.in", args[0])
        pe = kwargs.get("position_embeddings", None)
        if isinstance(pe, (tuple, list)) and len(pe) == 2:
            save_once("rope.cos", pe[0], "yarn cos fed to layers")
            save_once("rope.sin", pe[1], "yarn sin fed to layers")
        am = kwargs.get("attention_mask", None)
        if am is None:
            meta.setdefault(f"{name}.attn_mask", "None (bidirectional)")
        elif isinstance(am, torch.Tensor):
            save_once(f"{name}.attn_mask", am)
        return None
    return hook

def mk_post_hook(name):
    def hook(mod, args, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(out, torch.Tensor):
            save_once(f"{name}.out", out)
    return hook

def main():
    torch.manual_seed(SEED)
    cfg = AutoConfig.from_pretrained(DIR, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    print("loading model (device_map=auto, 13GiB GPU cap, bf16, sdpa)...")
    model = AutoModelForCausalLM.from_pretrained(
        DIR, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", max_memory={0: "13GiB", "cpu": "120GiB"},
        low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.eval()
    model.config.dlm_paradigm = "bidirectional"
    tok = PreTrainedTokenizerFast.from_pretrained(DIR)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        tok.pad_token = tok.eos_token

    enc = model.encoder
    hooks = []
    # backbone layers: 0,16,33 (chunk A + B)
    for li in (0, 16, 33):
        lyr = enc.layers[li]
        hooks.append(lyr.register_forward_pre_hook(mk_pre_hook(f"layer{li}"), with_kwargs=True))
        hooks.append(lyr.register_forward_hook(mk_post_hook(f"layer{li}")))
    # embeds / head / norm (chunk B + C)
    for nm, mod in [("embed_tokens", enc.embed_tokens), ("gen_embedding", enc.gen_embedding),
                    ("downsample_gen", enc.downsample_gen), ("upsample_gen", enc.upsample_gen),
                    ("gen_predictor", enc.gen_predictor), ("norm", enc.norm)]:
        if mod is None:
            continue
        hooks.append(mod.register_forward_pre_hook(mk_pre_hook(nm), with_kwargs=True))
        hooks.append(mod.register_forward_hook(mk_post_hook(nm)))

    prompt = "A red apple on a wooden table, studio lighting, photorealistic"
    micro = "ORIGINAL WIDTH : 1024; ORIGINAL HEIGHT : 1024; TOP : 0; LEFT : 0; SCORE : 6.5"
    # RELEASE SUPPORTS 1024 ONLY (downsample=True: n_tokens_txt reserve slots are
    # hardcoded 1024 for res==1024; 256/512 mismatch the 2x UViT downsample).
    res = 1024
    n_tokens = (res // 16) * (res // 16)   # 64x64 = 4096 latent tokens
    print(f"running text_to_image (1 step, gs=0, res={res}, n_tokens={n_tokens})...")
    # We only need the FORWARD taps (all captured before the sampler step). The
    # model's own Categorical.sample() over [1,4096,131072] logits OOMs at 1024^2
    # on 16GB (a chunk-E sampler concern) -> catch it and keep the captured taps.
    try:
        with torch.no_grad(), torch.inference_mode():
            model.text_to_image(
                prompt, tokenizer=tok, guidance_scale=0.0, n_steps=1, shift=5,
                schedule="shift", confidence_policy="mmada", schedule_temp="linear",
                temperature=0.86, alg_temp=1.0, dynamic_temperature=False,
                min_temperature=0.01, edit_threshold=0.6, micro_cond=micro,
                template="Generate an image with the caption:\n <prompt>",
                image_resolution=res, n_tokens=n_tokens, is_legacy=False,
                use_cache=False, disable_tqdm=True, return_intermediate_steps=False)
    except Exception as e:
        print(f"  [note] generation stopped after forward taps ({type(e).__name__}: {str(e)[:80]}) -- expected; taps captured")
    for h in hooks:
        h.remove()
    torch.cuda.empty_cache()

    # --- Oracle for chunk D: VQ decode of a FIXED deterministic token grid ---
    print("dumping VQ-decode oracle on a fixed token grid...")
    torch.manual_seed(SEED)
    cbk = enc.vqvae.config.codebook_size
    h16 = w16 = 16   # decode a small 16x16 grid -> 256x256 img (decoder is conv/res-agnostic; avoids 1024^2 OOM)
    fixed_ids = torch.randint(0, cbk, (1, h16 * w16), dtype=torch.long)
    save_once("vqdec.ids", fixed_ids.float(), "input token ids [1,256]")
    try:
        with torch.no_grad(), torch.inference_mode():
            # replicate decode_image_gen internals to tap intermediate + final
            vqdev = next(enc.vqvae.parameters()).device
            cb_entry = enc.vqvae.quantize.get_codebook_entry(fixed_ids.to(vqdev))
            save_once("vqdec.codebook_entry", cb_entry, "get_codebook_entry out")
            from einops import rearrange
            z = rearrange(cb_entry, "b (h w) d -> b d h w", h=h16, w=w16)
            dec = enc.vqvae.decode(z).float()
            save_once("vqdec.decoded_raw", dec, "vqvae.decode raw")
            save_once("vqdec.decoded_clamped", dec.clamp(-1, 1), "clamped [-1,1]")
    except Exception as e:
        print(f"  [note] VQ-decode oracle failed ({type(e).__name__}: {str(e)[:80]})")

    # save everything
    if captured:
        save_file(captured, os.path.join(OUT, "oracle1.safetensors"))
    with open(os.path.join(OUT, "oracle1_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nSAVED {len(captured)} tensors -> {OUT}/oracle1.safetensors")
    print("keys:", sorted(captured.keys()))

if __name__ == "__main__":
    main()
