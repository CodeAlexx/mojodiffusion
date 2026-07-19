#!/usr/bin/env python3
"""
SerenityTrainer Ernie (ERNIE-Image) training-parity ORACLE.

Authoritative reference for the Mojo Ernie LoRA training-parity gate. It reuses
SerenityTrainer's OWN transformer call + LoRA wrapper + flow-matching loss, fed the
SAME frozen trace tensors the Mojo gate consumes, and emits SerenityTrainer's
predicted flow, loss, and per-LoRA gradients.

Faithfulness (cited SerenityTrainer source, all under /home/alex/SerenityTrainer):
  - Transformer forward call:  modules/modelSetup/BaseErnieSetup.py:122-128 (predict)
  - unpatchify_latents:        modules/model/ErnieModel.py:164-169
  - target/flow + predicted:   modules/modelSetup/BaseErnieSetup.py:130-137
  - loss (MSE, unmasked):      modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py:139-197
                               (_flow_matching_losses :307-343 -> __unmasked_losses)
  - calculate_loss .mean():    modules/modelSetup/BaseErnieSetup.py:152-165
  - LoRA wrapper construction: modules/modelSetup/ErnieLoRASetup.py:50-65 (setup_model)
  - layer presets attn-mlp:    modules/modelSetup/BaseErnieSetup.py:33-38 -> ["self_attention","mlp"]
  - LoRAModule forward/scale:  modules/module/LoRAModule.py:323-329 (out + ld*(alpha/rank))
  - text_lens (tokens_mask):   modules/model/ErnieModel.py:153 (tokens_mask.sum(dim=1).long())

We DO NOT use predict's internal noise/timestep/seed generation; instead we feed
the frozen trace.* tensors so inputs are byte-identical to the Mojo gate.
Ernie's transformer signature (diffusers ErnieImageTransformer2DModel.forward):
  (hidden_states, timestep, text_bth, text_lens, return_dict)  -- 3D RoPE + text
  masking are handled internally given text_lens.

Run:
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    /home/alex/SerenityTrainer/venv/bin/python \
    /home/alex/serenity-trainer/scripts/serenity_ernie_oracle.py
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

reference trainer = "/home/alex/SerenityTrainer"
sys.path.insert(0, reference trainer)

from diffusers import ErnieImageTransformer2DModel  # noqa: E402
from modules.module.LoRAModule import LoRAModuleWrapper  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402

PARITY = "/home/alex/serenity-trainer/parity"
STEP = f"{PARITY}/ernie_train_ref_step000.safetensors"
ADAPTERS = f"{PARITY}/ernie_train_ref_step000_adapters.safetensors"
META = f"{PARITY}/ernie_train_ref_meta.json"
TRANSFORMER_DIR = "/home/alex/models/ERNIE-Image/transformer"
OUT_GRADS = f"{PARITY}/serenity_ernie_grads.safetensors"

DEV = "cuda"
BF16 = torch.bfloat16


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-30)


# SerenityTrainer ErnieModel.unpatchify_latents (static, pure reshape/permute) :164-169
def unpatchify_latents(latents):
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
    layer_filter = rc["layer_filter"]  # "self_attention,mlp"
    print(f"[cfg] model={rc['base_model_name']} rank={lora_rank} alpha={lora_alpha} "
          f"layer_filter={layer_filter} preset={rc['layer_filter_preset']} "
          f"lora_dtype={rc['lora_weight_dtype']} train_dtype={rc['train_dtype']}")

    # ---- frozen inputs (byte-identical to Mojo gate) ----
    d = load_file(STEP, device=DEV)
    hidden_states = d["trace.transformer_hidden_states"]        # [2,128,40,28] bf16 (scaled_noisy_latent_image, patchified)
    transformer_timestep = d["trace.transformer_timestep"]     # [2] i32  (discrete, 545/365)
    encoder_hidden_states = d["trace.encoder_hidden_states"]   # [2,201,3072] bf16 (text_encoder_output, trimmed)
    tokens_mask = d["trace.encode_text.tokens_mask"]           # [2,512] i64
    ref_packed_flow = d["trace.packed_predicted_flow"]         # [2,128,40,28] bf16
    target = d["output.target"].float()                       # [2,32,80,56] f32 (unpatchified flow)
    loss_weight = d["batch.loss_weight"].float()              # [2]
    dump_loss = d["output.loss_for_backward"].item()

    # text_lens = tokens_mask.sum(dim=1).long()  (ErnieModel.py:153)
    text_lens = tokens_mask.sum(dim=1).long()                 # [201, 157]
    print(f"[inputs] hidden_states={tuple(hidden_states.shape)} timestep={transformer_timestep.tolist()} "
          f"text_bth={tuple(encoder_hidden_states.shape)} text_lens={text_lens.tolist()}")

    # ---- transformer (SerenityTrainer loads diffusers ErnieImageTransformer2DModel) ----
    print(f"[load] ErnieImageTransformer2DModel from {TRANSFORMER_DIR}")
    transformer = ErnieImageTransformer2DModel.from_pretrained(
        TRANSFORMER_DIR, torch_dtype=BF16
    ).to(DEV)
    transformer.eval()  # base frozen; LoRA provides the only trainable params
    transformer.requires_grad_(False)
    # SerenityTrainer enables gradient checkpointing for the ernie transformer
    # (BaseErnieSetup.setup_optimizations -> enable_checkpointing_for_ernie_transformer);
    # recompute-on-backward, numerically identical, needed to fit on 24GB.
    transformer.enable_gradient_checkpointing()

    # ---- LoRA via SerenityTrainer's ErnieLoRASetup.setup_model (ErnieLoRASetup.py:50-65) ----
    config = TrainConfig.default_values()
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    config.layer_filter = layer_filter             # "self_attention,mlp"
    config.layer_filter_regex = False
    config.train_device = DEV
    config.dropout_probability = 0.0
    # peft_type=LORA, lora_decompose=False, lora_weight_dtype=FLOAT_32 defaults

    wrapper = LoRAModuleWrapper(
        transformer, "transformer", config, config.layer_filter.split(",")
    )
    print(f"[lora] {len(wrapper.lora_modules)} LoRA modules created")

    # ---- load adapter_before.* (initial LoRA: B=0); keys already contain "transformer." ----
    adapters = load_file(ADAPTERS, device=DEV)
    init_sd = {}
    for k, v in adapters.items():
        if k.startswith("adapter_before."):
            init_sd[k[len("adapter_before."):]] = v        # -> "transformer.layers.N...."
    # dump omits the per-module `alpha` buffer; inject it (== lora_alpha) for strict load
    for mod_name in wrapper.lora_modules:
        init_sd[f"transformer.{mod_name}.alpha"] = torch.tensor(lora_alpha)
    wrapper.load_state_dict(init_sd, strict=True)
    print(f"[lora] loaded {len(init_sd)} adapter_before tensors")

    wrapper.set_dropout(0.0)
    wrapper.to(dtype=torch.float32)                          # lora_weight_dtype FLOAT_32
    wrapper.hook_to_module()                                 # ErnieLoRASetup.py:65
    wrapper.requires_grad_(True)

    # ---- forward (mirror BaseErnieSetup.predict :122-128) under bf16 autocast ----
    with torch.autocast(device_type="cuda", dtype=BF16):
        predicted_flow = transformer(
            hidden_states=hidden_states.to(BF16),
            timestep=transformer_timestep,
            text_bth=encoder_hidden_states.to(BF16),
            text_lens=text_lens,
            return_dict=False,
        )[0]

    predicted = unpatchify_latents(predicted_flow)           # [2,32,80,56]

    # ---- GATE 1: forward vs frozen dump ----
    fwd_cos_packed = cos(predicted_flow.detach(), ref_packed_flow)
    fwd_cos_pred = cos(predicted.detach(), d["output.predicted"])
    print("\n==== GATE 1: SerenityTrainer forward vs frozen Serenity dump ====")
    print(f"  cos(predicted_flow, trace.packed_predicted_flow) = {fwd_cos_packed:.6f}")
    print(f"  cos(predicted,      output.predicted)            = {fwd_cos_pred:.6f}")

    # ---- loss (unmasked MSE -> .mean(); ModelSetupDiffusionLossMixin __unmasked_losses
    #      + _flow_matching_losses; calculate_loss .mean()) ----
    mean_dim = list(range(1, predicted.ndim))
    losses = F.mse_loss(predicted.float(), target, reduction="none").mean(mean_dim)  # mse_strength=1
    losses = losses * 1.0          # loss_scaler NONE -> scale 1
    losses = losses * loss_weight  # loss_weight==1
    # loss_weight_fn CONSTANT -> no timestep weighting (BaseErnieSetup uses flow matching SIGMA path,
    # but config.loss_weight_fn default CONSTANT -> pass)
    loss = losses.mean()
    print(f"  SerenityTrainer loss = {loss.item():.8f}")
    print(f"  dump loss       = {dump_loss:.8f}")
    print(f"  abs diff        = {abs(loss.item() - dump_loss):.3e}")
    verdict = "MATCH (SerenityTrainer == frozen Serenity Ernie dump)" if (
        fwd_cos_packed > 0.999 and abs(loss.item() - dump_loss) < 5e-3
    ) else "DIVERGE (frozen dump differs from SerenityTrainer)"
    print(f"  VERDICT: {verdict}")

    # ---- backward + capture per-LoRA grads ----
    loss.backward()

    grads = {}
    name_to_param = {}
    for mod_name, module in wrapper.lora_modules.items():
        for pname, p in module.named_parameters():
            if "lora_down" in pname or "lora_up" in pname:
                # full param name == dump adapter name space: transformer.<mod_name>.<pname>
                key = f"transformer.{mod_name}.{pname}"
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

    # ---- report sample grad L2s ----
    print("\n==== GATE 2: per-LoRA grad L2 (SerenityTrainer) ====")
    samples = [
        "transformer.layers.0.self_attention.to_q.lora_up.weight",
        "transformer.layers.0.self_attention.to_k.lora_up.weight",
        "transformer.layers.0.self_attention.to_v.lora_up.weight",
        "transformer.layers.0.self_attention.to_q.lora_down.weight",
        "transformer.layers.0.self_attention.to_out.0.lora_up.weight",
        "transformer.layers.0.mlp.gate_proj.lora_up.weight",
        "transformer.layers.0.mlp.up_proj.lora_up.weight",
        "transformer.layers.0.mlp.linear_fc2.lora_up.weight",
    ]
    for s in samples:
        serenity_l2 = grads[s].float().norm().item() if s in grads else float("nan")
        # cross-check the weight delta (adapter_after - adapter_before) direction if lr>0
        bk = "adapter_before." + s
        ak = "adapter_after." + s
        if bk in adapters and ak in adapters:
            dw = (adapters[ak] - adapters[bk]).float()
            dw_l2 = dw.norm().item()
        else:
            dw_l2 = float("nan")
        print(f"  {s}\n      SERENITY_grad_L2={serenity_l2:.6f}  (adapter_after-before)_L2={dw_l2:.6e}")

    print("\n[adapter key mapping]")
    print("  dump weight keys : adapter_before.<NAME> / adapter_after.<NAME>")
    print("  dump grad keys   : adapter_pre_clip_grad.<NAME>  (step-with-grads mode; absent in this 'step' dump)")
    print("  reference trainer grads file    : <NAME>   where NAME = transformer.layers.<i>.{self_attention.{to_q,to_k,to_v,to_out.0},mlp.{gate_proj,up_proj,linear_fc2}}.{lora_down,lora_up}.weight")
    print("  => Mojo compares reference trainer '<NAME>' to its own grad for the same NAME (or dump 'adapter_pre_clip_grad.<NAME>').")

    print("\n[done]")


if __name__ == "__main__":
    main()
