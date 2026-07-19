#!/usr/bin/env python3
"""
Anima (Anima-Base-v1.0 / Cosmos-Predict2-family video DiT) training-parity ORACLE.

IMPORTANT — THIS IS A diffusers-DIRECT ORACLE, *NOT* an SerenityTrainer model setup.
  Stock SerenityTrainer (/home/alex/SerenityTrainer, Nerogar) has NO AnimaModel /
  BaseAnimaSetup (verified: `ls modules/model | grep -i anima` -> empty,
  `ls modules/modelSetup | grep -i Anima` -> empty). The frozen dump was produced
  by the now-DELETED /home/alex/Serenity-anima-ref fork, whose AnimaModel wrapped
  the diffusers `CosmosTransformer3DModel` (config `_class_name:
  "CosmosTransformer3DModel"` in the base snapshot transformer/config.json).
  We therefore reconstruct the reference forward + flow-match loss directly from
  the diffusers `CosmosTransformer3DModel` (the same class the deleted fork
  wrapped) plus SerenityTrainer's OWN `LoRAModuleWrapper` for faithful LoRA semantics
  and key namespace. Label: diffusers-not-SerenityTrainer oracle.

Faithfulness:
  - Transformer:      diffusers CosmosTransformer3DModel.forward
                      (venv/src/diffusers/.../transformers/transformer_cosmos.py)
                      loaded from the dump's base_model_name transformer subfolder.
  - LoRA wrapper:     SerenityTrainer modules/module/LoRAModule.LoRAModuleWrapper
                      prefix "transformer", filter "attn1,attn2,ff"
                      -> 280 modules == exact dump module set (verified).
  - Forward inputs:   FROZEN trace.* tensors (byte-identical to the Mojo gate):
                        hidden_states = trace.transformer_hidden_states [2,16,1,64,64]
                        timestep      = trace.transformer_timestep [2]  (0.545,0.365)
                        encoder_hidden_states = trace.encoder_hidden_states [2,512,1024]
                        padding_mask  = trace.padding_mask [1,1,512,512] (all-zero)
                        attention_mask = None  (Mojo bug-fixer proved cross-attn
                        masking degrades cos; 3D RoPE only — model output [2,16,1,64,64])
  - predicted:        transformer(...).sample directly (3D latent shape; NO unpack).
  - flow-match loss:  target = trace.flow = noise - scaled_latent (verified == output.target);
                      loss = MSE(predicted, target, none).mean(per-sample dims) * loss_weight,
                      then .mean()  (== dump output.loss_for_backward; loss_weight==1).

We do NOT regenerate noise/timestep/seed: frozen inputs only -> deterministic.

Run:
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    /home/alex/SerenityTrainer/venv/bin/python \
    /home/alex/serenity-trainer/scripts/serenity_anima_oracle.py
"""

import json
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

reference trainer = "/home/alex/SerenityTrainer"
sys.path.insert(0, reference trainer)

from diffusers import CosmosTransformer3DModel  # noqa: E402
from modules.module.LoRAModule import LoRAModuleWrapper  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402

PARITY = "/home/alex/serenity-trainer/parity"
STEP = f"{PARITY}/anima_train_ref_step000.safetensors"
ADAPTERS = f"{PARITY}/anima_train_ref_step000_adapters.safetensors"
META = f"{PARITY}/anima_train_ref_meta.json"
OUT_GRADS = f"{PARITY}/serenity_anima_grads.safetensors"

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
    layer_filter = rc["layer_filter"]  # "attn1,attn2,ff"
    base_model = rc["base_model_name"]
    transformer_dir = f"{base_model}/transformer"
    print(f"[cfg] model=Anima-Base-v1.0 (CosmosTransformer3DModel) rank={lora_rank} "
          f"alpha={lora_alpha} layer_filter={layer_filter} "
          f"preset={rc['layer_filter_preset']} lora_dtype={rc['lora_weight_dtype']} "
          f"train_dtype={rc['train_dtype']}")
    print("[note] diffusers-DIRECT oracle: stock SerenityTrainer has NO Anima model setup; "
          "reconstructed from diffusers CosmosTransformer3DModel (the class the deleted "
          "Serenity-anima-ref fork wrapped) + SerenityTrainer LoRAModuleWrapper.")

    # ---- frozen inputs (byte-identical to Mojo gate) ----
    d = load_file(STEP, device=DEV)
    hidden_states = d["trace.transformer_hidden_states"]      # [2,16,1,64,64] bf16
    transformer_timestep = d["trace.transformer_timestep"]    # [2] f32 (0.545,0.365)
    encoder_hidden_states = d["trace.encoder_hidden_states"]  # [2,512,1024] bf16
    padding_mask = d["trace.padding_mask"]                    # [1,1,512,512] bf16 (all-zero)
    ref_predicted_flow = d["trace.predicted_flow"]            # [2,16,1,64,64] bf16
    target = d["output.target"].float()                      # [2,16,1,64,64] f32 (== trace.flow)
    loss_weight = d["batch.loss_weight"].float()             # [2]
    dump_loss = d["output.loss_for_backward"].item()
    print(f"[inputs] hidden={tuple(hidden_states.shape)} ts={transformer_timestep.tolist()} "
          f"enc={tuple(encoder_hidden_states.shape)} pad={tuple(padding_mask.shape)} "
          f"attn_mask=None")

    # ---- transformer (the same diffusers class the deleted fork wrapped) ----
    print(f"[load] CosmosTransformer3DModel from {transformer_dir}")
    transformer = CosmosTransformer3DModel.from_pretrained(
        transformer_dir, torch_dtype=BF16
    ).to(DEV)
    transformer.eval()
    transformer.requires_grad_(False)
    # gradient checkpointing: recompute-on-backward, numerically identical, fits 24GB.
    transformer.enable_gradient_checkpointing()
    print(f"[load] heads={transformer.config.num_attention_heads} "
          f"head_dim={transformer.config.attention_head_dim} "
          f"layers={transformer.config.num_layers} "
          f"extra_pos={transformer.config.extra_pos_embed_type} "
          f"concat_pad={transformer.config.concat_padding_mask}")

    # ---- LoRA via SerenityTrainer's LoRAModuleWrapper (prefix transformer, filter attn1,attn2,ff) ----
    config = TrainConfig.default_values()
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    config.layer_filter = layer_filter
    config.layer_filter_regex = False
    config.train_device = DEV
    config.dropout_probability = 0.0

    wrapper = LoRAModuleWrapper(
        transformer, "transformer", config, config.layer_filter.split(",")
    )
    print(f"[lora] {len(wrapper.lora_modules)} LoRA modules created (expect 280)")

    # ---- load adapter_before.* (initial LoRA: B=0); keys carry "transformer." prefix ----
    adapters = load_file(ADAPTERS, device=DEV)
    init_sd = {}
    for k, v in adapters.items():
        if k.startswith("adapter_before."):
            init_sd[k[len("adapter_before."):]] = v          # -> "transformer.<mod>...."
    # dump omits per-module `alpha` buffer; inject (== lora_alpha) for strict load
    for mod_name in wrapper.lora_modules:
        init_sd[f"transformer.{mod_name}.alpha"] = torch.tensor(lora_alpha)
    wrapper.load_state_dict(init_sd, strict=True)
    print(f"[lora] loaded {len(init_sd)} adapter_before tensors (strict)")

    wrapper.set_dropout(0.0)
    wrapper.to(dtype=torch.float32)                          # lora_weight_dtype FLOAT_32
    wrapper.hook_to_module()
    wrapper.requires_grad_(True)

    # ---- forward under bf16 autocast (train_dtype BFLOAT_16) ----
    with torch.autocast(device_type="cuda", dtype=BF16):
        predicted = transformer(
            hidden_states=hidden_states.to(BF16),
            timestep=transformer_timestep,                  # exact value fed to ref (0.545,0.365)
            encoder_hidden_states=encoder_hidden_states.to(BF16),
            attention_mask=None,                            # NO cross-attn mask (3D RoPE only)
            padding_mask=padding_mask.to(BF16),
            return_dict=True,
        ).sample                                            # [2,16,1,64,64]

    # ---- GATE 1: forward vs frozen dump ----
    fwd_cos = cos(predicted.detach(), ref_predicted_flow)
    fwd_cos_out = cos(predicted.detach(), d["output.predicted"])
    print("\n==== GATE 1: diffusers-Cosmos forward vs frozen Serenity-Anima dump ====")
    print(f"  cos(predicted, trace.predicted_flow) = {fwd_cos:.6f}")
    print(f"  cos(predicted, output.predicted)     = {fwd_cos_out:.6f}")

    # ---- flow-match loss: MSE(predicted, target).mean(sample dims) * loss_weight -> .mean() ----
    mean_dim = list(range(1, predicted.ndim))
    losses = F.mse_loss(predicted.float(), target, reduction="none").mean(mean_dim)
    losses = losses * loss_weight  # loss_weight==1
    loss = losses.mean()
    print(f"  oracle loss = {loss.item():.10f}")
    print(f"  dump loss   = {dump_loss:.10f}")
    print(f"  abs diff    = {abs(loss.item() - dump_loss):.3e}")
    verdict = "MATCH" if (fwd_cos > 0.999 and abs(loss.item() - dump_loss) < 5e-3) else \
        ("SKEWED" if fwd_cos > 0.99 else "DIVERGE")
    print(f"  VERDICT: {verdict}")

    # ---- backward + capture per-LoRA grads ----
    loss.backward()

    grads = {}
    name_to_param = {}
    for mod_name, module in wrapper.lora_modules.items():
        for pname, p in module.named_parameters():
            if "lora_down" in pname or "lora_up" in pname:
                # key == dump adapter namespace (minus adapter_before./after. prefix):
                #   transformer.<mod_name>.<pname>
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

    # ---- GATE 2: sample grad L2 + adapter-delta cross-check ----
    print("\n==== GATE 2: per-LoRA grad L2 (oracle) vs (adapter_after - adapter_before) ====")
    samples = [
        "transformer.transformer_blocks.0.attn1.to_q.lora_up.weight",
        "transformer.transformer_blocks.0.attn1.to_k.lora_up.weight",
        "transformer.transformer_blocks.0.attn1.to_v.lora_up.weight",
        "transformer.transformer_blocks.0.attn1.to_q.lora_down.weight",
        "transformer.transformer_blocks.0.attn2.to_q.lora_up.weight",
        "transformer.transformer_blocks.0.attn2.to_k.lora_up.weight",
        "transformer.transformer_blocks.0.ff.net.0.proj.lora_up.weight",
        "transformer.transformer_blocks.0.ff.net.2.lora_up.weight",
        "transformer.transformer_blocks.27.attn1.to_q.lora_up.weight",
    ]
    for s in samples:
        serenity_l2 = grads[s].float().norm().item() if s in grads else float("nan")
        bk = "adapter_before." + s
        ak = "adapter_after." + s
        if bk in adapters and ak in adapters:
            dw_l2 = (adapters[ak] - adapters[bk]).float().norm().item()
        else:
            dw_l2 = float("nan")
        print(f"  {s}\n      oracle_grad_L2={serenity_l2:.6f}  (after-before)_L2={dw_l2:.6e}")

    print("\n[adapter key mapping]")
    print("  dump weight keys : adapter_before.<NAME> / adapter_after.<NAME>")
    print("                     (also adapter_pre.<NAME>, adapter_post.<NAME>)")
    print("  dump grad keys   : NONE in this dump (no adapter_pre_clip_grad.*)")
    print("  oracle grads file: <NAME>  where")
    print("    NAME = transformer.{transformer_blocks.<i>.{attn1,attn2}."
          "{to_q,to_k,to_v,to_out.0},ff.net.{0.proj,2}}.{lora_down,lora_up}.weight")
    print("  => Mojo compares oracle grad '<NAME>' to its own grad for the same NAME.")

    print("\n[done]")


if __name__ == "__main__":
    main()
