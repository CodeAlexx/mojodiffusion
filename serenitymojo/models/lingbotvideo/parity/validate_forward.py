#!/usr/bin/env python
# Ground-truth check: run the REAL diffusers LingBotVideoTransformer3DModel.forward
# (via accelerate disk-offload — independent of my manual per-block reconstruction)
# on oracle_b2's exact inputs, and compare velocity to oracle_b2["velocity"] (my
# streamed reconstruction). If cos~1.0 -> reconstruction faithful (blob = settings).
import os, sys, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["LINGBOT_MOE_EXPERT_BACKEND"] = "grouped_mm"
import torch
from safetensors.torch import load_file
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")
from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

MDIR = "/mnt/disk1/models/lingbot-video-moe/transformer"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"


def main():
    cfg = json.load(open(os.path.join(MDIR, "config.json")))
    cfg_clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with init_empty_weights():
        model = LingBotVideoTransformer3DModel(**cfg_clean)
    model = load_checkpoint_and_dispatch(
        model, checkpoint=MDIR, device_map="auto",
        max_memory={0: "11GiB", "cpu": "40GiB"},
        offload_folder="/mnt/disk2/lingbot_offload",
        no_split_module_classes=["LingBotVideoBlock"], dtype=None)
    model.eval()
    # DECISIVE TEST: force the REAL model onto the eager MoE path (identical math to
    # grouped_mm). If it now matches my eager reconstruction, grouped_mm-under-offload
    # is the corrupt one; if it still diverges, my reconstruction has the bug.
    if os.environ.get("FORCE_EAGER") == "1":
        for blk in model.blocks:
            blk.ffn._run_grouped_experts = blk.ffn._run_experts_for_loop.__get__(blk.ffn, type(blk.ffn))
        print("[forced eager MoE on the real model]")

    o = load_file(os.path.join(OUT, "oracle_b2.safetensors"))
    latent = o["latent"].cuda().float()
    timestep = o["timestep"].cuda().float()
    text_embeds = o["text_embeds"].cuda().to(torch.bfloat16)
    ref_recon = o["velocity"].float()               # my manual streamed reconstruction

    # hook every block to localize where real vs reconstruction diverges
    taps = {}
    def mk(i):
        def h(m, inp, output):
            taps[i] = (output[0] if isinstance(output, tuple) else output).detach().float().cpu()
        return h
    for i, blk in enumerate(model.blocks):
        blk.register_forward_hook(mk(i))

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # matches pipeline _transformer_autocast
        out = model(latent, timestep, text_embeds, return_dict=False)[0].float().cpu()

    print("=== per-block: REAL-diffusers vs my-reconstruction (oracle_b2 block_i) ===")
    for i in [0, 1, 12, 24, 47]:
        if i in taps and f"block_{i}" in o:
            ra, rb = taps[i].flatten(), o[f"block_{i}"].float().flatten()
            c = torch.nn.functional.cosine_similarity(ra, rb, dim=0).item()
            print(f"  block_{i}: cos={c:.6f}  |real|/|recon|={ra.norm()/rb.norm():.4f}  real_std={taps[i].std():.4f} recon_std={o[f'block_{i}'].float().std():.4f}")

    a, b = out.flatten(), ref_recon.flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(f"REAL-diffusers vs my-reconstruction velocity:")
    print(f"  cos = {cos:.7f}   |real|/|recon| = {a.norm()/b.norm():.5f}   max_abs = {(a-b).abs().max():.5f}")
    print(f"  real: mean {out.mean():.5f} std {out.std():.5f}  recon: mean {ref_recon.mean():.5f} std {ref_recon.std():.5f}")
    if cos > 0.999:
        print("  => RECONSTRUCTION FAITHFUL. The E blob is SETTINGS (res/steps/plain-prompt), not a forward bug.")
    else:
        print("  => RECONSTRUCTION DIVERGES from real diffusers -> bug in the manual streamed forward (check post/patchify/temb).")


if __name__ == "__main__":
    main()
