#!/usr/bin/env python3
"""
SerenityTrainer SDXL (Stable Diffusion XL 1.0 base) training-parity ORACLE.

Authoritative reference for the Mojo SDXL LoRA training-parity gate. It reuses
SerenityTrainer's OWN UNet call + LoRA wrapper + eps-pred (target) MSE loss, fed the
SAME frozen trace tensors the Mojo gate / Serenity dump consume, and emits
SerenityTrainer's predicted noise, loss, and per-LoRA gradients.

SDXL is an eps-prediction UNet (NOT a flow-match DiT): there is no
unpack/unpatchify -- the UNet output IS the [1,4,H,W] predicted noise.

Faithfulness (cited SerenityTrainer source, all under /home/alex/SerenityTrainer):
  - UNet forward call:        modules/modelSetup/BaseStableDiffusionXLSetup.py:282-288 (predict)
      model.unet(sample=latent_input, timestep=timestep,
                 encoder_hidden_states=text_encoder_output,
                 added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids}).sample
  - eps target / predicted:   BaseStableDiffusionXLSetup.py:292-298
      prediction_type 'epsilon' -> predicted = predicted_latent_noise; target = latent_noise
  - loss (MSE, unmasked):     modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py:139-197
                              (__unmasked_losses) + _diffusion_losses :262-305
  - calculate_loss .mean():   BaseStableDiffusionXLSetup.py:375-388
  - LoRA wrapper construction: modules/modelSetup/StableDiffusionXLLoRASetup.py:90-92,116-117
      LoRAModuleWrapper(model.unet, "lora_unet", config, config.layer_filter.split(","))
      (layer_filter "" -> [""] -> empty ModuleFilter matches ALL layers => full UNet,
       ModuleFilter.py:27 "empty patterns ... match all layers, resulting in full training")
  - LoRAModule forward/scale: modules/module/LoRAModule.py:329 (orig + ld*(alpha/rank))
  - autocast bf16:            setup_optimizations create_autocast_context (train_dtype BFLOAT_16)

We DO NOT use predict's internal noise/timestep/seed generation; instead we feed
the frozen trace.* tensors so inputs are byte-identical to the Serenity dump:
  trace.latent_input            (1,4,168,96) bf16 -> sample
  trace.unet_timestep           (1,) i32 (=399)   -> timestep
  trace.encoder_hidden_states   (1,77,2048) bf16  -> encoder_hidden_states (CLIP-L 768 || CLIP-G 1280)
  trace.added_cond_text_embeds  (1,1280)          -> added_cond_kwargs["text_embeds"] (pooled CLIP-G)
  trace.added_cond_time_ids     (1,6) bf16         -> added_cond_kwargs["time_ids"]
forward ref: output.predicted / trace.predicted_latent_noise (1,4,168,96)
target ref : output.target (= latent_noise)
loss ref   : output.loss_for_backward (0.135332...)

NOTE on TE LoRA: the frozen dump's adapter set is UNet-only
(adapter_before.* are ALL "lora_unet.*"; 1588 tensors == 794 modules x {down,up};
NO lora_te1/lora_te2 keys). The original Serenity run trained UNet LoRA only
(text_encoder.train=False, no lora_te1 in init state -> StableDiffusionXLLoRASetup
does NOT create TE LoRA, lines 79-88). So this oracle mirrors UNet LoRA only,
exactly matching the dump's adapter set. Text encoders are NOT loaded.

Run:
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    /home/alex/SerenityTrainer/venv/bin/python \
    /home/alex/serenity-trainer/scripts/serenity_sdxl_oracle.py
"""

import json
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

reference trainer = "/home/alex/SerenityTrainer"
sys.path.insert(0, reference trainer)

from diffusers import UNet2DConditionModel  # noqa: E402
from modules.module.LoRAModule import LoRAModuleWrapper  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402

PARITY = "/home/alex/serenity-trainer/parity"
STEP = f"{PARITY}/sdxl_train_ref_step000.safetensors"
ADAPTERS = f"{PARITY}/sdxl_train_ref_step000_adapters.safetensors"
META = f"{PARITY}/sdxl_train_ref_meta.json"
# Dump's authoritative base model (meta.runtime_config.base_model_name); single-file
# SDXL -> diffusers UNet2DConditionModel via the standard SDXL conversion (same one
# SerenityTrainer's StableDiffusionXLModelLoader targets).
BASE_MODEL = "/home/alex/.serenity/models/checkpoints/sd_xl_base_1.0.safetensors"
OUT_GRADS = f"{PARITY}/serenity_sdxl_grads.safetensors"

# dump's full-UNet grad L2 (no clip) for cross-check (from task/meta)
DUMP_GRAD_NORM_NO_CLIP = 0.010624590

DEV = "cuda"
BF16 = torch.bfloat16


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-30)


def main():
    torch.manual_seed(0)
    meta = json.load(open(META))
    rc = meta["runtime_config"]
    lora_rank = int(rc["lora_rank"])
    lora_alpha = float(rc["lora_alpha"])
    layer_filter = rc["layer_filter"]              # "" (preset "full")
    print(f"[cfg] model={rc['model_type']} rank={lora_rank} alpha={lora_alpha} "
          f"layer_filter={layer_filter!r} preset={rc['layer_filter_preset']} "
          f"lora_dtype={rc['lora_weight_dtype']} train_dtype={rc['train_dtype']} "
          f"prediction_type={rc['prediction_type']} (SDXL default scheduler -> epsilon)")

    # ---- frozen inputs (byte-identical to Serenity dump / Mojo gate) ----
    d = load_file(STEP, device=DEV)
    latent_input = d["trace.latent_input"]                 # [1,4,168,96] bf16 (== scaled_noisy_latent_image)
    unet_timestep = d["trace.unet_timestep"]               # [1] i32 (=399)
    encoder_hidden_states = d["trace.encoder_hidden_states"]  # [1,77,2048] bf16
    added_text_embeds = d["trace.added_cond_text_embeds"]  # [1,1280] f32 (pooled CLIP-G)
    added_time_ids = d["trace.added_cond_time_ids"]        # [1,6] bf16
    ref_predicted = d["output.predicted"]                  # [1,4,168,96] bf16
    ref_pred_latent_noise = d["trace.predicted_latent_noise"]  # [1,4,168,96] bf16
    target = d["output.target"].float()                    # [1,4,168,96] (latent_noise)
    loss_weight = d["batch.loss_weight"].float()           # [1]
    dump_loss = d["output.loss_for_backward"].item()
    print(f"[inputs] sample={tuple(latent_input.shape)} timestep={unet_timestep.tolist()} "
          f"ehs={tuple(encoder_hidden_states.shape)} pooled={tuple(added_text_embeds.shape)} "
          f"time_ids={added_time_ids.float().tolist()}")

    # ---- UNet (diffusers UNet2DConditionModel from the dump's base single-file) ----
    print(f"[load] UNet2DConditionModel.from_single_file {BASE_MODEL}")
    unet = UNet2DConditionModel.from_single_file(BASE_MODEL, torch_dtype=BF16).to(DEV)
    unet.eval()                       # base frozen; LoRA provides the only trainable params
    unet.requires_grad_(False)
    # SerenityTrainer enables gradient checkpointing for the SDXL UNet
    # (BaseStableDiffusionXLSetup.setup_optimizations:51-53); recompute-on-backward,
    # numerically identical.
    unet.enable_gradient_checkpointing()

    # ---- LoRA via SerenityTrainer's StableDiffusionXLLoRASetup.setup_model (lines 90-92,116-117) ----
    config = TrainConfig.default_values()
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    config.layer_filter = layer_filter             # "" -> full UNet
    config.layer_filter_regex = False
    config.train_device = DEV
    config.dropout_probability = 0.0
    # peft_type=LORA, lora_decompose=False, lora_weight_dtype=FLOAT_32 are defaults

    wrapper = LoRAModuleWrapper(
        unet, "lora_unet", config, config.layer_filter.split(",")
    )
    print(f"[lora] {len(wrapper.lora_modules)} LoRA modules created "
          f"(expect 794 -> 1588 down/up tensors)")

    # ---- load adapter_before.* (initial LoRA: B=0); keys carry "lora_unet." prefix ----
    adapters = load_file(ADAPTERS, device=DEV)
    init_sd = {}
    for k, v in adapters.items():
        if k.startswith("adapter_before."):
            init_sd[k[len("adapter_before."):]] = v        # -> "lora_unet.<mod>...."
    # dump omits the per-module `alpha` buffer; inject it (== lora_alpha) for strict load
    for mod_name in wrapper.lora_modules:
        init_sd[f"lora_unet.{mod_name}.alpha"] = torch.tensor(lora_alpha)
    wrapper.load_state_dict(init_sd, strict=True)
    print(f"[lora] loaded {len(init_sd)} adapter_before tensors (strict)")

    wrapper.set_dropout(0.0)
    wrapper.to(dtype=torch.float32)                        # lora_weight_dtype FLOAT_32
    wrapper.hook_to_module()                               # StableDiffusionXLLoRASetup.py:117
    wrapper.requires_grad_(True)

    # ---- forward (mirror BaseStableDiffusionXLSetup.predict :282-288) under bf16 autocast ----
    added_cond_kwargs = {
        "text_embeds": added_text_embeds.to(BF16),
        "time_ids": added_time_ids.to(BF16),
    }
    with torch.autocast(device_type="cuda", dtype=BF16):
        predicted = unet(
            sample=latent_input.to(BF16),
            timestep=unet_timestep,
            encoder_hidden_states=encoder_hidden_states.to(BF16),
            added_cond_kwargs=added_cond_kwargs,
        ).sample                                            # [1,4,168,96] -- already predicted noise

    # ---- GATE 1: forward vs frozen dump ----
    fwd_cos_pred = cos(predicted.detach(), ref_predicted)
    fwd_cos_pln = cos(predicted.detach(), ref_pred_latent_noise)
    print("\n==== GATE 1: SerenityTrainer forward vs frozen Serenity dump ====")
    print(f"  cos(predicted, output.predicted)            = {fwd_cos_pred:.6f}")
    print(f"  cos(predicted, trace.predicted_latent_noise)= {fwd_cos_pln:.6f}")

    # ---- loss (eps target MSE -> .mean(); __unmasked_losses + _diffusion_losses + calculate_loss) ----
    mean_dim = list(range(1, predicted.ndim))
    losses = F.mse_loss(predicted.float(), target, reduction="none").mean(mean_dim)  # mse_strength=1
    losses = losses * 1.0          # loss_scaler NONE -> scale 1
    losses = losses * loss_weight  # loss_weight==1
    # loss_weight_fn CONSTANT -> no timestep weighting (epsilon)
    loss = losses.mean()
    print(f"  SerenityTrainer loss = {loss.item():.8f}")
    print(f"  dump loss       = {dump_loss:.8f}")
    print(f"  abs diff        = {abs(loss.item() - dump_loss):.3e}")
    verdict = "MATCH (SerenityTrainer == frozen Serenity SDXL dump)" if (
        fwd_cos_pred > 0.999 and abs(loss.item() - dump_loss) < 5e-3
    ) else "DIVERGE (frozen dump differs from SerenityTrainer)"
    print(f"  VERDICT: {verdict}")

    # ---- backward + capture per-LoRA grads ----
    loss.backward()

    grads = {}
    name_to_param = {}
    for mod_name, module in wrapper.lora_modules.items():
        for pname, p in module.named_parameters():
            if "lora_down" in pname or "lora_up" in pname:
                key = f"lora_unet.{mod_name}.{pname}"        # == dump adapter namespace
                name_to_param[key] = p
    n_grad = 0
    total_sq = 0.0
    for key, p in name_to_param.items():
        if p.grad is not None:
            g = p.grad.detach().to(torch.float32).cpu().contiguous()
            grads[key] = g
            total_sq += g.double().pow(2).sum().item()
            n_grad += 1
        else:
            grads[key] = torch.zeros_like(p, dtype=torch.float32, device="cpu")
    save_file(grads, OUT_GRADS)
    total_grad_norm = total_sq ** 0.5
    print(f"\n[grads] wrote {len(grads)} tensors ({n_grad} with non-None .grad) -> {OUT_GRADS}")

    # ---- GATE 2: total grad-norm vs dump + sample L2s across families ----
    print("\n==== GATE 2: full-UNet grad norm + per-family sample L2 (SerenityTrainer) ====")
    print(f"  SerenityTrainer total grad L2 (no clip) = {total_grad_norm:.9f}")
    print(f"  dump grad_norm_no_clip             = {DUMP_GRAD_NORM_NO_CLIP:.9f}")
    print(f"  abs diff                           = {abs(total_grad_norm - DUMP_GRAD_NORM_NO_CLIP):.3e}")
    print(f"  rel diff                           = "
          f"{abs(total_grad_norm - DUMP_GRAD_NORM_NO_CLIP) / DUMP_GRAD_NORM_NO_CLIP:.3e}")

    samples = [
        # conv adapter (resnet conv2d)
        "lora_unet.conv_in.lora_up.weight",
        "lora_unet.down_blocks.0.resnets.0.conv1.lora_up.weight",
        "lora_unet.down_blocks.0.resnets.0.conv1.lora_down.weight",
        # embed adapters (Linear in time/add embedding)
        "lora_unet.time_embedding.linear_1.lora_up.weight",
        "lora_unet.add_embedding.linear_1.lora_up.weight",
        # spatial-transformer (ST) attention adapters
        "lora_unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_up.weight",
        "lora_unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_up.weight",
        "lora_unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight",
        "lora_unet.down_blocks.1.attentions.0.proj_in.lora_up.weight",
        "lora_unet.down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.lora_up.weight",
    ]
    for s in samples:
        serenity_l2 = grads[s].float().norm().item() if s in grads else float("nan")
        bk = "adapter_before." + s
        ak = "adapter_after." + s
        if bk in adapters and ak in adapters:
            dw_l2 = (adapters[ak] - adapters[bk]).float().norm().item()
        else:
            dw_l2 = float("nan")
        print(f"  {s}\n      SERENITY_grad_L2={serenity_l2:.6f}  (adapter_after-before)_L2={dw_l2:.6e}")

    print("\n[adapter key mapping]")
    print("  dump weight keys : adapter_before.<NAME> / adapter_after.<NAME>")
    print("                     (also adapter_pre.<NAME>, adapter_post.<NAME>); ALL <NAME> are lora_unet.*")
    print("  dump grad keys   : NONE (adapter_dump='step'; no adapter_pre_clip_grad.*)")
    print("  reference trainer grads file    : <NAME> where")
    print("    NAME = lora_unet.{conv_in,conv_out,time_embedding.linear_{1,2},add_embedding.linear_{1,2},")
    print("           {down,mid,up}_blocks...resnets.<i>.{conv1,conv2,time_emb_proj},")
    print("           ...downsamplers/upsamplers.0.conv,")
    print("           ...attentions.<a>.{proj_in,proj_out},")
    print("           ...attentions.<a>.transformer_blocks.<t>.{attn1,attn2}.{to_q,to_k,to_v,to_out.0},")
    print("           ...transformer_blocks.<t>.ff.net.{0.proj,2}}.{lora_down,lora_up}.weight")
    print("  => Mojo compares reference trainer grads '<NAME>' to its own grad for the same NAME.")
    print("  TE LoRA: SKIPPED (dump adapter set is UNet-only; no lora_te1/lora_te2 in dump).")

    print("\n[done]")


if __name__ == "__main__":
    main()
