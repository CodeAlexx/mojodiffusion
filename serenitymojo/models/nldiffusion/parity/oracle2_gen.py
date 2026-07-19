#!/usr/bin/env python
# Oracle-2: DETERMINISTIC end-to-end reference token grid (chunks E/F gate).
# Uses sample_policy=argmax + confidence_policy=stratified => NO torch RNG in the
# sampler => the Mojo port can reproduce the token grid (given matching logits).
# Memory-lean: lower device_map GPU cap so the [1,4096,131072] logits fit on 16GB.
# Captures x0_img (final token grid) by monkeypatching decode_image_gen (avoids the
# 1024^2 decode OOM inside the resident-model process; decode separately later).
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DEBUG_FIX_PADDING", "1")
os.environ.setdefault("NOT_ALWASY_DO_2DPOOL", "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerFast
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from safetensors.torch import save_file

DIR = "/mnt/disk1/models/NL-Diffusion-Image"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"
N_STEPS = int(os.environ.get("N_STEPS", "64"))
GPU_CAP = os.environ.get("GPU_CAP", "6GiB")
GS = float(os.environ.get("GS", "0.0"))  # 0 = no CFG (single-batch logits, fits 16GB; still deterministic)
PROMPT = "A red apple on a wooden table, studio lighting, photorealistic"

def main():
    torch.manual_seed(42); np.random.seed(42)
    cfg = AutoConfig.from_pretrained(DIR, trust_remote_code=True); cfg._attn_implementation = "sdpa"
    print(f"loading (GPU cap {GPU_CAP}, bf16, sdpa)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        DIR, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory={0: GPU_CAP, "cpu": "120GiB"}, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.eval(); model.config.dlm_paradigm = "bidirectional"
    tok = PreTrainedTokenizerFast.from_pretrained(DIR)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id; tok.pad_token = tok.eos_token
    mod = sys.modules[type(model).__module__]

    # deterministic stratified order for the 64x64 grid (seed 42), passed in + saved
    order = mod._stratified_random(n=64, seed=42, shuffle_blocks=True)

    grabbed = {}
    orig_decode = model.decode_image_gen
    def patched_decode(images_to_decode, h, w):
        grabbed["x0_img"] = images_to_decode.detach().cpu().long().clone()
        print(f"[capture] token grid {list(images_to_decode.shape)} captured; skipping in-proc decode", flush=True)
        # return a dummy 1x1 image array so text_to_image's Image.fromarray works
        return np.zeros((1, 1, 1, 3), dtype=np.uint8)
    model.decode_image_gen = patched_decode

    micro = "ORIGINAL WIDTH : 1024; ORIGINAL HEIGHT : 1024; TOP : 0; LEFT : 0; SCORE : 6.5"
    t0 = time.time()
    print(f"generating (argmax+stratified, gs=5, n_steps={N_STEPS}, 1024^2)...", flush=True)
    with torch.no_grad(), torch.inference_mode():
        model.text_to_image(
            PROMPT, tokenizer=tok, sample_policy="argmax", confidence_policy="stratified",
            guidance_scale=GS, n_steps=N_STEPS, shift=5, schedule="shift",
            schedule_temp="linear", temperature=0.86, alg_temp=1.0, dynamic_temperature=False,
            min_temperature=0.01, edit_threshold=-1, micro_cond=micro,
            template="Generate an image with the caption:\n <prompt>",
            image_resolution=1024, n_tokens=4096, unmask_order=order,
            is_legacy=False, use_cache=False, disable_tqdm=False, return_intermediate_steps=False)
    dt = time.time() - t0
    x0 = grabbed["x0_img"]  # [1,4096]
    caps = {"x0_img": x0.float(), "unmask_order": torch.as_tensor(np.asarray(order, dtype=np.int64)).float()}
    save_file(caps, os.path.join(OUT, "oracle2_tokens.safetensors"))
    json.dump({"n_steps": N_STEPS, "gs": GS, "policy": "argmax+stratified", "prompt": PROMPT,
               "gen_time_s": dt, "sec_per_step": dt / N_STEPS, "token_grid": list(x0.shape),
               "n_unique_tokens": int(x0.unique().numel())},
              open(os.path.join(OUT, "oracle2_meta.json"), "w"), indent=2)
    print(f"DONE in {dt:.0f}s ({dt/N_STEPS:.1f}s/step). token grid {list(x0.shape)}, "
          f"{int(x0.unique().numel())} unique ids. SAVED oracle2_tokens.safetensors", flush=True)

if __name__ == "__main__":
    main()
