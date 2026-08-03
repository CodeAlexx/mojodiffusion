"""MiniMax-H3 Qwen3-VL VISION TOWER oracle — REAL FL2VA weights, CPU, float32.

Reference for serenitymojo/models/text_encoder/parity/
minimax_h3_qwen3vl_vision_forward_probe.mojo.

── WHY CPU float32, AND WHY THAT IS NOT THE "CPU PARITY" ANTI-PATTERN ────────
This repo's rule is that a GPU bf16 implementation must be gated against a
GPU-generated bf16 oracle, because a CPU f32 reference conflates dtype error
with logic error. That rule is about a DEVICE port. What is being gated here is
the HOST float32 forward — the oracle half of this repo's usual
host-oracle/device-port pair (models/minimax_h3/video_encoder.mojo vs
models/vae/minimax_h3_video_encoder_device.mojo is the same split). So both
sides here are CPU float32: same device class, same dtype, apples to apples.
The eventual device port of this tower needs its OWN bf16 gate against this
host implementation; that is stated in the probe, not assumed away.

The released weights are BF16. They are upcast to float32 ONCE, before the
forward, so the reference is a clean f32 evaluation of the exact stored values
rather than a bf16 forward the Mojo side would have to imitate.

── WHAT IT DUMPS ────────────────────────────────────────────────────────────
Every stage boundary the Mojo forward has to hit, so a failure localizes:
    pixel_values          the tower's input, straight from the H3 processor
    pos_embeds            the bilinear 48x48 interpolation alone
    after_patch_embed     the LINEAR (trap 1), before the position add
    after_pos_add         patch_embed + pos_embeds
    after_block_{0,8,16,24,26}
    deepstack_{0,1,2}     the POSTSHUFFLE-norm mergers (trap 3)
    embeds                the final PRESHUFFLE-norm merger (traps 3 + 4)
    rotary_cos/rotary_sin the 2-D half-width tables (trap 5)
    cu_seqlens            the per-frame segments (trap 2)

Run (CPU only, ~1.1 GiB of weights, no CUDA):
    CUDA_VISIBLE_DEVICES="" python3 scripts/minimax_h3_vision_forward_oracle.py \
        [image.png] [out.safetensors]
Defaults to output/minimax_h3_keyframe/prepared_keyframe_192x128.png and
output/minimax_h3_keyframe/vision_forward_ref.safetensors.
"""

import glob
import json
import os
import struct
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

H3 = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA"
TEXT_ENCODER = os.path.join(H3, "text_encoder")
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_keyframe"
PREFIX = "model.visual."


def load_vision_tower():
    """Build the tower from FL2VA's own vision_config and load ONLY the 351
    `model.visual.*` tensors — not the 62 GiB text tower they share a checkpoint
    with. They all live in one shard, so this reads one file."""
    cfg_all = json.load(open(os.path.join(TEXT_ENCODER, "config.json")))
    vcfg = Qwen3VLVisionConfig(**cfg_all["vision_config"])
    print(f"vision_config: depth={vcfg.depth} hidden={vcfg.hidden_size} "
          f"heads={vcfg.num_heads} deepstack={vcfg.deepstack_visual_indexes} "
          f"out={vcfg.out_hidden_size}")
    model = Qwen3VLVisionModel(vcfg)

    wm = json.load(open(os.path.join(TEXT_ENCODER, "model.safetensors.index.json")))["weight_map"]
    visual = {k: v for k, v in wm.items() if k.startswith(PREFIX)}
    shards = sorted({v for v in visual.values()})
    print(f"{len(visual)} visual tensors in {len(shards)} shard(s): {shards}")

    state = {}
    for shard in shards:
        with safe_open(os.path.join(TEXT_ENCODER, shard), framework="pt", device="cpu") as f:
            for k in visual:
                if visual[k] == shard:
                    # BF16 on disk -> float32 ONCE, before anything runs.
                    state[k[len(PREFIX):]] = f.get_tensor(k).to(torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=True), None
    print(f"loaded {len(state)} tensors, strict=True OK")
    return model.eval().to(torch.float32), vcfg


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        OUT_DIR, "prepared_keyframe_192x128.png")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        OUT_DIR, "vision_forward_ref.safetensors")
    os.makedirs(OUT_DIR, exist_ok=True)

    proc = AutoProcessor.from_pretrained(os.path.join(H3, "processor"))
    image = Image.open(image_path).convert("RGB")
    vision = proc.image_processor(images=[image], return_tensors="pt")
    pixel_values = vision["pixel_values"].to(torch.float32)
    grid_thw = vision["image_grid_thw"]
    t, h, w = grid_thw[0].tolist()
    print(f"{image_path}: pixel_values {tuple(pixel_values.shape)} grid {t}x{h}x{w} "
          f"-> {t*h*w} patches, {t*h*w//4} merged tokens")

    model, vcfg = load_vision_tower()

    taps = {}

    def grab(name):
        def hook(_mod, _inp, out):
            taps[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return hook

    handles = [model.patch_embed.register_forward_hook(grab("after_patch_embed"))]
    for i in (0, 8, 16, 24, vcfg.depth - 1):
        handles.append(model.blocks[i].register_forward_hook(grab(f"after_block_{i}")))
    for i in range(len(vcfg.deepstack_visual_indexes)):
        handles.append(model.deepstack_merger_list[i].register_forward_hook(grab(f"deepstack_{i}")))

    with torch.no_grad():
        # The geometry pieces, captured directly off the model so the Mojo
        # module's own versions are compared against the tower's, not against a
        # transcription of it.
        pos_embeds = model.fast_pos_embed_interpolate(grid_thw)
        rotary = model.rot_pos_emb(grid_thw)
        seq = pixel_values.shape[0]
        emb = torch.cat((rotary.reshape(seq, -1), rotary.reshape(seq, -1)), dim=-1)
        cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32)
        cu = torch.nn.functional.pad(cu, (1, 0), value=0)

        embeds, deepstack_list = model(pixel_values, grid_thw)

        # The bilinear lookup decomposed: the four CORNER INDICES are integers
        # and a mismatch moves a whole grid cell, so they are gated bit-exact
        # separately from the weights, which carry torch's float32 linspace
        # rounding. Rebuilt here with the model's own num_grid_per_side.
        side = model.num_grid_per_side
        idx_list = [[] for _ in range(4)]
        wt_list = [[] for _ in range(4)]
        for t_, h_, w_ in zip(grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]):
            h_idxs = torch.linspace(0, side - 1, h_)
            w_idxs = torch.linspace(0, side - 1, w_)
            hf, wf = h_idxs.int(), w_idxs.int()
            hc = (h_idxs.int() + 1).clip(max=side - 1)
            wc = (w_idxs.int() + 1).clip(max=side - 1)
            dh, dw = h_idxs - hf, w_idxs - wf
            bh, bhc = hf * side, hc * side
            for i, (a, b) in enumerate([(bh, wf), (bh, wc), (bhc, wf), (bhc, wc)]):
                idx_list[i].extend((a[None].T + b[None]).flatten().tolist())
            for i, (a, b) in enumerate([((1 - dh), (1 - dw)), ((1 - dh), dw),
                                        (dh, (1 - dw)), (dh, dw)]):
                wt_list[i].extend((a[None].T * b[None]).flatten().tolist())
        tensors_extra = {
            "pos_corner_indices": torch.tensor(idx_list, dtype=torch.int32),
            "pos_corner_weights": torch.tensor(wt_list, dtype=torch.float32),
        }

    for hd in handles:
        hd.remove()

    tensors = {
        "pixel_values": pixel_values,
        "grid_thw": grid_thw.to(torch.int32),
        "cu_seqlens": cu.to(torch.int32),
        "pos_embeds": pos_embeds.float(),
        "rotary_cos": emb.cos().float(),
        "rotary_sin": emb.sin().float(),
        "after_pos_add": (taps["after_patch_embed"] + pos_embeds).float(),
        "embeds": embeds.float(),
    }
    for k, v in taps.items():
        tensors[k] = v.float()
    for i, d in enumerate(deepstack_list):
        tensors[f"deepstack_{i}"] = d.float()
    tensors.update(tensors_extra)

    # ── THE TOLERANCE BASELINE ──────────────────────────────────────────────
    # Run the SAME tower in float64 and record how far torch's own float32 path
    # lands from it, per stage. The probe sets its bars as a multiple of THESE
    # numbers instead of hand-picked constants, so a bar can never be quietly
    # widened until a bug passes: the only way to raise it is for torch's own
    # f32 error to rise, which is a property of the model, not of the port.
    #
    # This matters most at depth. By block 26 the activations reach std ~103 and
    # torch's own f32 answer is already ~0.7 from the true one — a "max_abs <
    # 1e-3" bar there would fail a CORRECT implementation, and a flat loose bar
    # would pass a broken one at block 0.
    print("computing the float64 baseline (torch f32 vs torch f64)...")
    taps64 = {}

    def grab64(name):
        def hook(_mod, _inp, out):
            taps64[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return hook

    model64 = model.to(torch.float64)
    h64 = [model64.patch_embed.register_forward_hook(grab64("after_patch_embed"))]
    for i in (0, 8, 16, 24, vcfg.depth - 1):
        h64.append(model64.blocks[i].register_forward_hook(grab64(f"after_block_{i}")))
    for i in range(len(vcfg.deepstack_visual_indexes)):
        h64.append(model64.deepstack_merger_list[i].register_forward_hook(grab64(f"deepstack_{i}")))
    with torch.no_grad():
        embeds64, _ = model64(pixel_values.to(torch.float64), grid_thw)
    for hd in h64:
        hd.remove()
    taps64["embeds"] = embeds64

    baseline = {}
    for name, ref64 in taps64.items():
        if name not in tensors:
            continue
        a = tensors[name].double().flatten()
        b = ref64.double().flatten()
        if a.numel() != b.numel():
            continue
        baseline[name] = float((a - b).abs().max())
    for name, err in sorted(baseline.items()):
        tensors["f32err_" + name] = torch.tensor([err], dtype=torch.float32)
        print(f"  f32err_{name:20s} {err:.6g}")

    save_file({k: v.contiguous() for k, v in tensors.items()}, out_path)
    print("wrote", out_path)
    for k in sorted(tensors):
        v = tensors[k]
        extra = ""
        if v.dtype.is_floating_point:
            extra = f" mean={v.mean().item():+.6f} std={v.std().item():.6f}"
        print(f"  {k:22s} {tuple(v.shape)}{extra}")
    print("cuda_initialized", torch.cuda.is_initialized())


if __name__ == "__main__":
    main()
