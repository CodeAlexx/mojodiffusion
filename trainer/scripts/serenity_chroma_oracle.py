#!/usr/bin/env python3
"""
SerenityTrainer Chroma (Chroma1-HD) training-parity ORACLE.

Authoritative reference for the Mojo Chroma LoRA training-parity gate. It reuses
SerenityTrainer's OWN transformer call + LoRA wrapper + flow-matching loss, fed the
SAME frozen trace tensors the Mojo gate consumes, and emits SerenityTrainer's
predicted flow, loss, and per-LoRA gradients.

Faithfulness (cited SerenityTrainer source, all under /home/alex/SerenityTrainer):
  - Transformer forward call:  modules/modelSetup/BaseChromaSetup.py:227-236 (predict)
      (Chroma has NO `guidance` kwarg: the distilled approximator builds the
       modulation internally from `timestep`; guidance_embeds=false in config.)
  - unpack_latents:            modules/model/ChromaModel.py:259-270
  - target/flow + predicted:   modules/modelSetup/BaseChromaSetup.py:238-250
                               (predicted = unpack(packed_flow); target = noise - scaled_latent)
  - loss (MSE, unmasked):      modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py:139-197
                               (__unmasked_losses) + _flow_matching_losses :307-343
  - calculate_loss .mean():    modules/modelSetup/BaseChromaSetup.py:265-278
  - LoRA wrapper construction: modules/modelSetup/ChromaLoRASetup.py:78-95 (setup_model)
      prefix "lora_transformer", filter config.layer_filter.split(",")
  - layer presets attn-mlp:    modules/modelSetup/BaseChromaSetup.py:40-45 -> ["attn","ff.net"]
  - LoRAModule forward/scale:  modules/module/LoRAModule.py (out + ld*(alpha/rank))

We DO NOT use predict's internal noise/timestep/seed generation; instead we feed
the frozen trace.* tensors so inputs are byte-identical to the Mojo gate:
  trace.packed_latent_input (2,1024,64)  -> hidden_states
  trace.transformer_timestep (2,) f32    -> timestep   (ALREADY /1000: 0.907,0.709)
  trace.encoder_hidden_states (2,224,4096)-> encoder_hidden_states (trimmed text)
  trace.text_ids (224,3)                 -> txt_ids
  trace.image_ids_forward (1024,3)       -> img_ids
  trace.attention_mask (2,1248)          -> attention_mask  (text||image; not all-True)

Run:
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    /home/alex/SerenityTrainer/venv/bin/python \
    /home/alex/serenity-trainer/scripts/serenity_chroma_oracle.py
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

reference trainer = "/home/alex/SerenityTrainer"
sys.path.insert(0, reference trainer)

from diffusers import ChromaTransformer2DModel  # noqa: E402
from modules.module.LoRAModule import LoRAModuleWrapper  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402

PARITY = "/home/alex/serenity-trainer/parity"
STEP = f"{PARITY}/chroma_train_ref_step000.safetensors"
ADAPTERS = f"{PARITY}/chroma_train_ref_step000_adapters.safetensors"
META = f"{PARITY}/chroma_train_ref_meta.json"
TRANSFORMER_DIR = (
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/"
    "snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer"
)
OUT_GRADS = f"{PARITY}/serenity_chroma_grads.safetensors"

DEV = "cuda"
BF16 = torch.bfloat16


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-30)


# SerenityTrainer ChromaModel.unpack_latents (static, pure reshape/permute) :259-270
def unpack_latents(latents, height: int, width: int):
    batch_size, _, channels = latents.shape
    height = height // 2
    width = width // 2
    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)
    return latents


def main():
    torch.manual_seed(0)
    meta = json.load(open(META))
    rc = meta["runtime_config"]
    lora_rank = int(rc["lora_rank"])
    lora_alpha = float(rc["lora_alpha"])
    layer_filter = rc["layer_filter"]  # "attn,ff.net"
    print(f"[cfg] model=Chroma1-HD rank={lora_rank} alpha={lora_alpha} "
          f"layer_filter={layer_filter} preset={rc['layer_filter_preset']} "
          f"lora_dtype={rc['lora_weight_dtype']} train_dtype={rc['train_dtype']}")

    # ---- frozen inputs (byte-identical to Mojo gate) ----
    d = load_file(STEP, device=DEV)
    packed_latent_input = d["trace.packed_latent_input"]      # [2,1024,64] bf16
    transformer_timestep = d["trace.transformer_timestep"]    # [2] f32  (== timestep/1000)
    encoder_hidden_states = d["trace.encoder_hidden_states"]  # [2,224,4096] bf16
    text_ids = d["trace.text_ids"]                            # [224,3] bf16
    image_ids = d["trace.image_ids_forward"]                  # [1024,3] bf16
    attention_mask = d["trace.attention_mask"]                # [2,1248] bool
    ref_packed_flow = d["trace.packed_predicted_flow"]        # [2,1024,64] bf16
    target = d["trace.flow"].float()                          # [2,16,64,64] f32 (noise - scaled_latent)
    loss_weight = d["batch.loss_weight"].float()              # [2]
    dump_loss = d["output.loss_for_backward"].item()
    # latent_input.shape[2], shape[3]  (latent_image [2,16,64,64])
    H = W = 64

    print(f"[inputs] hidden_states={tuple(packed_latent_input.shape)} "
          f"timestep={transformer_timestep.tolist()} "
          f"txt={tuple(encoder_hidden_states.shape)} txt_ids={tuple(text_ids.shape)} "
          f"img_ids={tuple(image_ids.shape)} attn_mask={tuple(attention_mask.shape)} "
          f"(all_true={bool(attention_mask.all())})")

    # ---- transformer (SerenityTrainer loads diffusers ChromaTransformer2DModel) ----
    print(f"[load] ChromaTransformer2DModel from {TRANSFORMER_DIR}")
    transformer = ChromaTransformer2DModel.from_pretrained(
        TRANSFORMER_DIR, torch_dtype=BF16
    ).to(DEV)
    transformer.eval()  # base frozen; LoRA provides the only trainable params
    transformer.requires_grad_(False)
    # SerenityTrainer enables gradient checkpointing for the chroma transformer
    # (BaseChromaSetup.setup_optimizations -> enable_checkpointing_for_chroma_transformer);
    # recompute-on-backward, numerically identical, needed to fit ~9B bf16 on 24GB.
    transformer.enable_gradient_checkpointing()
    print(f"[load] guidance_embeds={transformer.config.guidance_embeds} "
          f"num_layers={transformer.config.num_layers} "
          f"num_single_layers={transformer.config.num_single_layers}")

    # ---- LoRA via SerenityTrainer's ChromaLoRASetup.setup_model (ChromaLoRASetup.py:78-95) ----
    config = TrainConfig.default_values()
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    config.layer_filter = layer_filter             # "attn,ff.net"
    config.layer_filter_regex = False
    config.train_device = DEV
    config.dropout_probability = 0.0
    # peft_type=LORA, lora_decompose=False, lora_weight_dtype=FLOAT_32 defaults

    wrapper = LoRAModuleWrapper(
        transformer, "lora_transformer", config, config.layer_filter.split(",")
    )
    print(f"[lora] {len(wrapper.lora_modules)} LoRA modules created (expect 304)")

    # ---- load adapter_before.* (initial LoRA: B=0); keys carry "lora_transformer." prefix ----
    adapters = load_file(ADAPTERS, device=DEV)
    init_sd = {}
    for k, v in adapters.items():
        if k.startswith("adapter_before."):
            init_sd[k[len("adapter_before."):]] = v        # -> "lora_transformer.<mod>...."
    # dump omits the per-module `alpha` buffer; inject it (== lora_alpha) for strict load
    for mod_name in wrapper.lora_modules:
        init_sd[f"lora_transformer.{mod_name}.alpha"] = torch.tensor(lora_alpha)
    wrapper.load_state_dict(init_sd, strict=True)
    print(f"[lora] loaded {len(init_sd)} adapter_before tensors (strict)")

    wrapper.set_dropout(0.0)
    wrapper.to(dtype=torch.float32)                          # lora_weight_dtype FLOAT_32
    wrapper.hook_to_module()                                 # ChromaLoRASetup.py:95
    wrapper.requires_grad_(True)

    # ---- forward (mirror BaseChromaSetup.predict :227-236) under bf16 autocast ----
    with torch.autocast(device_type="cuda", dtype=BF16):
        packed_predicted_flow = transformer(
            hidden_states=packed_latent_input.to(BF16),
            timestep=transformer_timestep,                  # already timestep/1000
            encoder_hidden_states=encoder_hidden_states.to(BF16),
            txt_ids=text_ids.to(BF16),
            img_ids=image_ids.to(BF16),
            attention_mask=attention_mask,
            joint_attention_kwargs=None,
            return_dict=True,
        ).sample

    predicted = unpack_latents(packed_predicted_flow, H, W)   # [2,16,64,64]

    # ---- GATE 1: forward vs frozen dump ----
    fwd_cos_packed = cos(packed_predicted_flow.detach(), ref_packed_flow)
    fwd_cos_pred = cos(predicted.detach(), d["output.predicted"])
    print("\n==== GATE 1: SerenityTrainer forward vs frozen Serenity dump ====")
    print(f"  cos(packed_predicted_flow, trace.packed_predicted_flow) = {fwd_cos_packed:.6f}")
    print(f"  cos(predicted,             output.predicted)            = {fwd_cos_pred:.6f}")

    # ---- loss (unmasked MSE -> .mean(); __unmasked_losses + _flow_matching_losses
    #      + calculate_loss .mean()) ----
    mean_dim = list(range(1, predicted.ndim))
    losses = F.mse_loss(predicted.float(), target, reduction="none").mean(mean_dim)  # mse_strength=1
    losses = losses * 1.0          # loss_scaler NONE -> scale 1
    losses = losses * loss_weight  # loss_weight==1
    # loss_weight_fn CONSTANT -> no timestep weighting
    loss = losses.mean()
    print(f"  SerenityTrainer loss = {loss.item():.10f}")
    print(f"  dump loss       = {dump_loss:.10f}")
    print(f"  abs diff        = {abs(loss.item() - dump_loss):.3e}")
    verdict = "MATCH (SerenityTrainer == frozen Serenity Chroma dump)" if (
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
                # full name == dump adapter namespace (minus adapter_before./after. prefix):
                #   lora_transformer.<mod_name>.<pname>
                key = f"lora_transformer.{mod_name}.{pname}"
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

    # ---- GATE 2: report sample grad L2s + adapter delta cross-check ----
    print("\n==== GATE 2: per-LoRA grad L2 (SerenityTrainer) ====")
    samples = [
        "lora_transformer.transformer_blocks.0.attn.to_q.lora_up.weight",
        "lora_transformer.transformer_blocks.0.attn.to_k.lora_up.weight",
        "lora_transformer.transformer_blocks.0.attn.to_v.lora_up.weight",
        "lora_transformer.transformer_blocks.0.attn.to_q.lora_down.weight",
        "lora_transformer.transformer_blocks.0.attn.to_k.lora_down.weight",
        "lora_transformer.transformer_blocks.0.attn.to_v.lora_down.weight",
        "lora_transformer.single_transformer_blocks.0.attn.to_q.lora_up.weight",
        "lora_transformer.single_transformer_blocks.0.attn.to_v.lora_up.weight",
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
    print("                     (also adapter_pre.<NAME>, adapter_post.<NAME>)")
    print("  dump grad keys   : NONE in this dump (adapter_dump='step'; no adapter_pre_clip_grad.*)")
    print("  reference trainer grads file    : <NAME>  where")
    print("    NAME = lora_transformer.{transformer_blocks.<i>.{attn.{to_q,to_k,to_v,to_out.0,"
          "add_q_proj,add_k_proj,add_v_proj,to_add_out},ff.net.{0.proj,2}},"
          "single_transformer_blocks.<j>.attn.{to_q,to_k,to_v}}.{lora_down,lora_up}.weight")
    print("  => Mojo compares reference trainer grads '<NAME>' to its own grad for the same NAME.")

    print("\n[done]")


if __name__ == "__main__":
    main()
