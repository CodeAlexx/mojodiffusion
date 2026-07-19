#!/usr/bin/env python3
"""
SerenityTrainer Klein (Flux2) training-parity ORACLE.

Replaces the deleted /home/alex/Serenity fork as the authoritative reference for
the Mojo Klein/Flux2 LoRA training-parity gate. It reuses SerenityTrainer's OWN
transformer call + LoRA wrapper + flow-matching loss, fed the SAME frozen trace
tensors the Mojo gate consumes, and emits SerenityTrainer's predicted flow, loss, and
per-LoRA gradients.

Faithfulness (cited SerenityTrainer source, all under /home/alex/SerenityTrainer):
  - Transformer forward call:  modules/modelSetup/BaseFlux2Setup.py:142-151  (predict)
  - unpack / unpatchify:       modules/model/Flux2Model.py:259-262, 304-310
  - target/flow + predicted:   modules/modelSetup/BaseFlux2Setup.py:153-166
  - loss (MSE, unmasked):      modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py:139-197
                               (_flow_matching_losses :307-343 -> __unmasked_losses)
  - calculate_loss .mean():    modules/modelSetup/BaseFlux2Setup.py:181-194
  - LoRA wrapper construction: modules/modelSetup/Flux2LoRASetup.py:52-71  (setup_model)
  - LoRAModule forward/scale:  modules/module/LoRAModule.py:323-329  (out + ld*(alpha/rank))

We DO NOT use predict's internal noise/timestep/seed generation; instead we feed
the frozen trace.* tensors so inputs are byte-identical to the Mojo gate.
guidance_embeds=False for klein-base-9B -> guidance=None (matches BaseFlux2Setup.py:132-136).

Run:  /home/alex/SerenityTrainer/venv/bin/python /home/alex/serenity-trainer/scripts/serenity_klein_oracle.py
"""

import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

reference trainer = "/home/alex/SerenityTrainer"
sys.path.insert(0, reference trainer)

from diffusers import Flux2Transformer2DModel  # noqa: E402
from modules.module.LoRAModule import LoRAModuleWrapper  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402

PARITY = "/home/alex/serenity-trainer/parity"
STEP = f"{PARITY}/klein_train_ref_step000.safetensors"
ADAPTERS = f"{PARITY}/klein_train_ref_step000_adapters.safetensors"
META = f"{PARITY}/klein_train_ref_meta.json"
TRANSFORMER_DIR = (
    "/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/"
    "snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3/transformer"
)
OUT_GRADS = f"{PARITY}/serenity_klein_grads.safetensors"

DEV = "cuda"
BF16 = torch.bfloat16


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-30)


# SerenityTrainer Flux2Model.unpack_latents / unpatchify_latents (static, pure reshape/permute)
def unpack_latents(latents, height, width):  # Flux2Model.py:259-262
    b, seq, c = latents.shape
    return latents.reshape(b, height, width, c).permute(0, 3, 1, 2)


def unpatchify_latents(latents):  # Flux2Model.py:304-310
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def main():
    torch.manual_seed(0)
    meta = json.load(open(META))
    rc = meta["runtime_config"]
    lora_rank = int(rc["lora_rank"])
    lora_alpha = float(rc["lora_alpha"])
    print(f"[cfg] model={rc['base_model_name']} rank={lora_rank} alpha={lora_alpha} "
          f"layer_filter={rc['layer_filter_preset']} lora_dtype={rc['lora_weight_dtype']}")

    # ---- frozen inputs (byte-identical to Mojo gate) ----
    d = load_file(STEP, device=DEV)
    packed_latent_input = d["trace.packed_latent_input"]      # [1,1024,128] bf16
    transformer_timestep = d["trace.transformer_timestep"]    # [1] f32  == timestep/1000
    encoder_hidden_states = d["trace.encoder_hidden_states"]  # [1,512,12288] bf16
    text_ids = d["trace.text_ids"]                            # [1,512,4] int64
    image_ids = d["trace.image_ids"]                          # [1,1024,4] int64
    ref_packed_flow = d["trace.packed_predicted_flow"]        # [1,1024,128] bf16
    target = d["output.target"].float()                      # [1,32,64,64] f32 (unpatchified flow)
    loss_weight = d["batch.loss_weight"].float()             # [1]
    dump_loss = d["output.loss_for_backward"].item()
    H = W = 32  # latent_input.shape[2], shape[3] (scaled_noisy_latent_image [1,128,32,32])

    # ---- transformer (SerenityTrainer loads the same diffusers Flux2Transformer2DModel) ----
    print(f"[load] Flux2Transformer2DModel from {TRANSFORMER_DIR}")
    transformer = Flux2Transformer2DModel.from_pretrained(
        TRANSFORMER_DIR, torch_dtype=BF16
    ).to(DEV)
    transformer.eval()  # base frozen; LoRA provides the only trainable params
    transformer.requires_grad_(False)
    # SerenityTrainer enables gradient checkpointing for the flux2 transformer
    # (BaseFlux2Setup.setup_optimizations -> enable_checkpointing_for_flux2_transformer);
    # recompute-on-backward, numerically identical, needed to fit 9B bf16 on 24GB.
    transformer.enable_gradient_checkpointing()
    guidance_embeds = bool(transformer.config.guidance_embeds)
    guidance = None
    if guidance_embeds:  # BaseFlux2Setup.py:132-136
        gs = 1.0
        guidance = torch.tensor([gs], device=DEV, dtype=BF16).expand(packed_latent_input.shape[0])
    print(f"[load] guidance_embeds={guidance_embeds} -> guidance={'None' if guidance is None else gs}")

    # ---- LoRA via SerenityTrainer's Flux2LoRASetup.setup_model (Flux2LoRASetup.py:52-71) ----
    config = TrainConfig.default_values()
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    config.layer_filter = "transformer_block"   # LAYER_PRESETS["blocks"] (BaseFlux2Setup.py:38-41)
    config.layer_filter_regex = False
    config.train_device = DEV
    config.dropout_probability = 0.0
    # peft_type=LORA, lora_decompose=False, lora_weight_dtype=FLOAT_32 are defaults (verified)

    wrapper = LoRAModuleWrapper(
        transformer, "transformer", config, config.layer_filter.split(",")
    )

    # ---- load adapter_before.* (initial LoRA: B=0) -> SerenityTrainer key space "transformer.*" ----
    adapters = load_file(ADAPTERS, device=DEV)
    init_sd = {}
    for k, v in adapters.items():
        if k.startswith("adapter_before."):
            mod = k[len("adapter_before."):]                  # e.g. single_transformer_blocks.0.attn.to_out.lora_down.weight
            init_sd["transformer." + mod] = v                 # wrapper.prefix == "transformer"
    # dump omits the per-module `alpha` buffer; inject it (== lora_alpha) so strict load validates the full set
    for mod_name in wrapper.lora_modules:
        init_sd[f"transformer.{mod_name}.alpha"] = torch.tensor(lora_alpha)
    wrapper.load_state_dict(init_sd, strict=True)             # LoRAModule.py:721-749 (strict -> validates layer set)
    print(f"[lora] loaded {len(init_sd)} adapter_before tensors into {len(wrapper.lora_modules)} LoRA modules")

    wrapper.set_dropout(0.0)
    wrapper.to(dtype=torch.float32)                          # lora_weight_dtype FLOAT_32 (Flux2LoRASetup.py:66)
    wrapper.hook_to_module()                                 # Flux2LoRASetup.py:67
    wrapper.requires_grad_(True)

    # ---- forward (mirror BaseFlux2Setup.predict :142-151) under bf16 autocast ----
    with torch.autocast(device_type="cuda", dtype=BF16):
        packed_predicted_flow = transformer(
            hidden_states=packed_latent_input.to(BF16),
            timestep=transformer_timestep,                  # already timestep/1000 == 0.545
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states.to(BF16),
            txt_ids=text_ids,
            img_ids=image_ids,
            joint_attention_kwargs=None,
            return_dict=True,
        ).sample

    predicted_flow = unpack_latents(packed_predicted_flow, H, W)   # [1,128,32,32]
    predicted = unpatchify_latents(predicted_flow)                 # [1,32,64,64]

    # ---- GATE 1: forward vs frozen dump ----
    fwd_cos_packed = cos(packed_predicted_flow.detach(), ref_packed_flow)
    fwd_cos_pred = cos(predicted.detach(), d["output.predicted"])
    print("\n==== GATE 1: SerenityTrainer forward vs frozen Serenity dump ====")
    print(f"  cos(packed_predicted_flow, trace.packed_predicted_flow) = {fwd_cos_packed:.6f}")
    print(f"  cos(predicted,            output.predicted)            = {fwd_cos_pred:.6f}")

    # ---- loss (unmasked MSE -> .mean(); ModelSetupDiffusionLossMixin.py:150-155 + BaseFlux2Setup.py:181-194) ----
    mean_dim = list(range(1, predicted.ndim))
    losses = F.mse_loss(predicted.float(), target, reduction="none").mean(mean_dim)  # mse_strength=1
    losses = losses * 1.0          # loss_scaler NONE -> scale 1
    losses = losses * loss_weight  # loss_weight==1
    # loss_weight_fn CONSTANT -> no timestep weighting
    loss = losses.mean()
    print(f"  SerenityTrainer loss = {loss.item():.8f}")
    print(f"  dump loss       = {dump_loss:.8f}")
    print(f"  abs diff        = {abs(loss.item() - dump_loss):.3e}")
    verdict = "MATCH (SerenityTrainer == old Serenity reference)" if (
        fwd_cos_packed > 0.999 and abs(loss.item() - dump_loss) < 5e-3
    ) else "DIVERGE (old reference differs from SerenityTrainer)"
    print(f"  VERDICT: {verdict}")

    # ---- backward + capture per-LoRA grads ----
    loss.backward()

    grads = {}
    name_to_param = {}
    for mod_name, module in wrapper.lora_modules.items():
        for pname, p in module.named_parameters():
            if "lora_down" in pname or "lora_up" in pname:
                # pname like "lora_down.weight"; key = <module path>.<pname>
                key = f"{mod_name}.{pname}"
                name_to_param[key] = p
    n_grad = 0
    for key, p in name_to_param.items():
        if p.grad is not None:
            grads[key] = p.grad.detach().to(torch.float32).cpu().contiguous()
            n_grad += 1
        else:
            grads[key] = torch.zeros_like(p, dtype=torch.float32, device="cpu")
    save_file(grads, OUT_GRADS)
    print(f"\n[grads] wrote {len(grads)} tensors ({n_grad} with non-None .grad) -> {OUT_GRADS}")

    # ---- report sample grad L2s + compare to dump pre_clip grads ----
    print("\n==== GATE 2: per-LoRA grad L2 (SerenityTrainer) vs dump adapter_pre_clip_grad ====")
    samples = [
        "transformer_blocks.0.attn.to_q.lora_up.weight",
        "transformer_blocks.0.attn.to_k.lora_up.weight",
        "transformer_blocks.0.attn.to_v.lora_up.weight",
        "transformer_blocks.0.attn.to_q.lora_down.weight",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_up.weight",
        "single_transformer_blocks.0.attn.to_out.lora_up.weight",
    ]
    for s in samples:
        serenity_l2 = grads[s].float().norm().item() if s in grads else float("nan")
        dk = "adapter_pre_clip_grad." + s
        dump_l2 = adapters[dk].float().norm().item() if dk in adapters else float("nan")
        cc = cos(grads[s], adapters[dk].cpu()) if (s in grads and dk in adapters) else float("nan")
        print(f"  {s}\n      SERENITY_L2={serenity_l2:.6f}  dump_L2={dump_l2:.6f}  cos={cc:.6f}")

    print("\n[done]")


if __name__ == "__main__":
    main()
