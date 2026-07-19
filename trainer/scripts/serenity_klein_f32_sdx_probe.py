#!/usr/bin/env python3
"""
PRECISION DISCRIMINATOR (F32, GPU block-swap, NO LoRA).

Runs the Klein/Flux2 transformer backward in F32 on GPU via diffusers block-level
group offloading (CPU<->GPU stream), capturing the per-single-block running input
gradient (== Mojo running d_x). Compared against the bf16 reference
(serenity_klein_block0_sdx.safetensors) this answers: is the joint-attention TEXT grad
ill-conditioned (reference trainer-F32 vs reference trainer-bf16 also decorrelates -> precision-inherent, F32 is
the fix) or precision-robust (Mojo has a real discrete bug)?

No LoRA: with B=0 the LoRA contributes nothing to the forward, and we capture grad
at BLOCK INPUTS (not LoRA grads), so the base transformer suffices. We seed grad by
making the latent input require grad. Same frozen step001 inputs as the gate.

Run: KLEIN_SERENITY_F32=1 already implied; just:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/alex/SerenityTrainer/venv/bin/python scripts/serenity_klein_f32_sdx_probe.py
"""

import sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

reference trainer = "/home/alex/SerenityTrainer"
sys.path.insert(0, reference trainer)
from diffusers import Flux2Transformer2DModel  # noqa: E402

PARITY = "/home/alex/serenity-trainer/parity"
STEP = "/tmp/klein_train_ref_2step_step001.safetensors"
TRANSFORMER_DIR = (
    "/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/"
    "snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3/transformer"
)
OUT_SDX = f"{PARITY}/serenity_klein_block0_sdx_f32.safetensors"
H_LAT, W_LAT = 40, 28
F32 = torch.float32


def unpack_latents(latents, height, width):
    b, seq, c = latents.shape
    return latents.reshape(b, height, width, c).permute(0, 3, 1, 2)


def unpatchify_latents(latents):
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def main():
    torch.manual_seed(0)
    d = load_file(STEP, device="cpu")
    packed = d["trace.packed_latent_input"].to(F32).cuda().requires_grad_(True)
    ts = d["trace.transformer_timestep"].to(F32).cuda()
    ehs = d["trace.encoder_hidden_states"].to(F32).cuda()
    text_ids = d["trace.text_ids"].cuda()
    image_ids = d["trace.image_ids"].cuda()
    target = d["output.target"].float().cuda()
    loss_weight = d["batch.loss_weight"].float().cuda()
    H, W = H_LAT, W_LAT

    print(f"[load] Flux2Transformer2DModel F32 from {TRANSFORMER_DIR}")
    transformer = Flux2Transformer2DModel.from_pretrained(TRANSFORMER_DIR, torch_dtype=F32)
    transformer.eval()
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=True,
    )
    print("[load] leaf-level group offload (F32, stream) enabled")

    guidance = None
    if bool(transformer.config.guidance_embeds):
        guidance = torch.tensor([1.0], device="cuda", dtype=F32).expand(packed.shape[0])

    captured = {}

    def mk_sdx(i):
        def h(module, args):
            x = args[0]
            if isinstance(x, torch.Tensor) and x.requires_grad and x.dim() == 3:
                x.register_hook(lambda g, i=i: captured.__setitem__(f"sdx_{i}", g.detach().float().cpu()[0]))
        return h

    handles = []
    for i, sb_i in enumerate(transformer.single_transformer_blocks):
        handles.append(sb_i.register_forward_pre_hook(mk_sdx(i)))

    packed_predicted_flow = transformer(
        hidden_states=packed,
        timestep=ts,
        guidance=guidance,
        encoder_hidden_states=ehs,
        txt_ids=text_ids,
        img_ids=image_ids,
        joint_attention_kwargs=None,
        return_dict=True,
    ).sample

    predicted = unpatchify_latents(unpack_latents(packed_predicted_flow, H, W))
    mean_dim = list(range(1, predicted.ndim))
    losses = F.mse_loss(predicted.float(), target, reduction="none").mean(mean_dim)
    loss = (losses * loss_weight).mean()
    print(f"[fwd] F32 loss={loss.item():.8f}")
    loss.backward()
    for h in handles:
        h.remove()

    sdx = {k: v.contiguous() for k, v in captured.items() if k.startswith("sdx_")}
    if not sdx:
        print("[ERR] no per-block grads captured")
        sys.exit(1)
    save_file(sdx, OUT_SDX)
    print(f"[saved] {OUT_SDX} ({len(sdx)} blocks)")
    print("[done]")


if __name__ == "__main__":
    main()
