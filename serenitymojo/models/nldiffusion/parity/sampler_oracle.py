#!/usr/bin/env python
# Deterministic sampler oracle for chunk E (fast, CPU, no model).
# Dumps the deterministic sampler pieces so the Mojo sampler can gate exactly:
#  - _get_num_transfer_tokens (shift schedule) for n_tokens=4096
#  - sch_temperatures (linear schedule)
#  - stratified unmask_order (_stratified_random seed 42)
#  - argmax over the captured gen_predictor logits (oracle1 gen_predictor.out)
#  - a CFG-combine reference on synthetic logits
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from safetensors.torch import save_file, load_file

DIR = "/mnt/disk1/models/NL-Diffusion-Image"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"
N_TOKENS = 4096
N_STEPS = 64
SHIFT = 5
MIN_TEMP = 0.01

def load_fns():
    import sys
    cls = get_class_from_dynamic_module(
        "modeling_nemotron_labs_diffusion_image.NemotronLabsDiffusionImageForMaskedDiffusion", DIR)
    return sys.modules[cls.__module__]   # the loaded module (package set up correctly)

def main():
    torch.manual_seed(42); np.random.seed(42)
    mod = load_fns()
    caps, meta = {}, {}
    def sv(n, t, note=""):
        t = torch.as_tensor(t).detach().cpu().float().contiguous()
        caps[n] = t; meta[n] = {"shape": list(t.shape), "note": note}
        print(f"  [tap] {n:22s} {list(t.shape)} {note}")

    # 1) num_transfer_tokens (shift schedule)
    mask_index = torch.ones(1, N_TOKENS, dtype=torch.bool)
    ntt = mod._get_num_transfer_tokens(mask_index, N_STEPS, schedule='shift', shift=SHIFT)
    sv("num_transfer", ntt[0], f"per-step token budget, sum={int(ntt.sum())}")

    # 2) sch_temperatures (linear)
    sch_t = np.linspace(0, 1, N_STEPS)
    sch_temps = (1.0 - sch_t) * (1.0 - MIN_TEMP) + MIN_TEMP
    sv("sch_temps_linear", sch_temps, "linear temp schedule")

    # 3) stratified unmask_order (seed 42)
    dim = int(np.sqrt(N_TOKENS))  # 64
    order = mod._stratified_random(n=dim, seed=42, shuffle_blocks=True)
    sv("unmask_order", np.asarray(order, dtype=np.int64), f"stratified order len={len(order)}")

    # 4) argmax over captured gen_predictor logits
    o1 = load_file(os.path.join(OUT, "oracle1.safetensors"))
    logits = o1["gen_predictor.out"]  # [128,131072] f32
    am = logits.argmax(dim=-1)
    sv("gen_predictor_argmax", am.float(), "argmax token id per row of gen_predictor.out[128]")

    # 5) CFG combine reference on synthetic logits (elementwise math check)
    torch.manual_seed(7)
    cond = torch.randn(4, 16); uncond = torch.randn(4, 16); gs = 5.0
    combined = (1.0 + gs) * cond - gs * uncond
    sv("cfg_cond", cond); sv("cfg_uncond", uncond); sv("cfg_combined", combined, "(1+gs)cond-gs*uncond, gs=5")

    save_file(caps, os.path.join(OUT, "sampler_oracle.safetensors"))
    meta["config"] = {"n_tokens": N_TOKENS, "n_steps": N_STEPS, "shift": SHIFT, "min_temp": MIN_TEMP, "gs": 5.0}
    json.dump(meta, open(os.path.join(OUT, "sampler_oracle_meta.json"), "w"), indent=2, default=str)
    print("num_transfer[:6] =", ntt[0][:6].tolist(), " sum =", int(ntt.sum()), "(== n_tokens)")
    print("unmask_order[:8] =", list(order[:8]))
    print(f"SAVED -> {OUT}/sampler_oracle.safetensors")

if __name__ == "__main__":
    main()
