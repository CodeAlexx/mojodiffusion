#!/usr/bin/env python3
"""
Anima (Anima-Base-v1.0 / CosmosTransformer3DModel) training-parity ORACLE — REAL SETUP.

DIFFERENCE FROM serenity_anima_oracle.py (the diffusers-DIRECT oracle):
  This oracle runs inside the cloned SerenityTrainer-Anima PR venv
  (/home/alex/SerenityTrainer-anima/venv) which pins
  diffusers @ git commit b003a47  ->  diffusers 0.39.0.dev0  AND
  transformers 5.9.0. The Anima base-model snapshot transformer/config.json
  carries `_diffusers_version: "0.39.0.dev0"` — i.e. the FROZEN dump was
  produced by THIS exact CosmosTransformer3DModel class, NOT the working
  venv's diffusers 0.38.0.dev0 that serenity_anima_oracle.py used (which produced
  cos=0.999099 — a version-skew residual).

  We mirror the PR's real training path:
    - LoRA via the clone's modules.module.LoRAModule.LoRAModuleWrapper
      (the same wrapper AnimaLoRASetup.setup_model uses,
       /home/alex/SerenityTrainer-anima/modules/modelSetup/AnimaLoRASetup.py:61).
    - Transformer forward + flow target mirror
      BaseAnimaSetup.predict EXACTLY (cited line numbers below)
      /home/alex/SerenityTrainer-anima/modules/modelSetup/BaseAnimaSetup.py
    - Loss mirrors BaseAnimaSetup.calculate_loss -> _flow_matching_losses
      -> __unmasked_losses (default config: mse_strength=1, all others 0,
       loss_scaler scale=1, loss_weight=1, LossWeight.CONSTANT).
      Verified vs dump: output.loss_pre_scale == output.loss_for_backward
      == 0.0667838  => scale==1, weight==1.

  The transformer here is loaded directly as CosmosTransformer3DModel
  (== AnimaModel.transformer's class; AnimaModel.py:18,74) from the dump's
  base_model_name transformer subfolder. Feeding FROZEN trace.* inputs means
  encode_text / noise / timestep generation (predict:96-126) is bypassed
  byte-identically — only the transformer call (predict:135-141) and flow
  target (predict:143-149) + loss matter, which we reproduce verbatim.

Run:
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=/home/alex/SerenityTrainer-anima \
    /home/alex/SerenityTrainer-anima/venv/bin/python \
    /home/alex/serenity-trainer/scripts/serenity_anima_real_oracle.py
"""

import json
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

# clone path so LoRAModuleWrapper / TrainConfig come from the PR, not stock reference trainer.
CLONE = "/home/alex/SerenityTrainer-anima"
if CLONE not in sys.path:
    sys.path.insert(0, CLONE)

import diffusers  # noqa: E402
from diffusers import CosmosTransformer3DModel  # noqa: E402
from modules.module.LoRAModule import LoRAModuleWrapper  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402

PARITY = "/home/alex/serenity-trainer/parity"
STEP = f"{PARITY}/anima_train_ref_step000.safetensors"
ADAPTERS = f"{PARITY}/anima_train_ref_step000_adapters.safetensors"
META = f"{PARITY}/anima_train_ref_meta.json"
OUT_GRADS = f"{PARITY}/serenity_anima_real_grads.safetensors"

DEV = "cuda"
BF16 = torch.bfloat16


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-30)


def main():
    torch.manual_seed(0)
    print(f"[env] diffusers={diffusers.__version__}  torch={torch.__version__}")
    import transformers
    print(f"[env] transformers={transformers.__version__}")

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
    print("[note] REAL-setup oracle: clone LoRAModuleWrapper + diffusers@b003a47 "
          "CosmosTransformer3DModel (matches snapshot _diffusers_version 0.39.0.dev0).")

    # ---- frozen inputs (byte-identical to Mojo gate) ----
    d = load_file(STEP, device=DEV)
    hidden_states = d["trace.transformer_hidden_states"]      # [2,16,1,64,64] bf16
    transformer_timestep = d["trace.transformer_timestep"]    # [2] f32 (0.545,0.365) == timestep/1000 (predict:137)
    encoder_hidden_states = d["trace.encoder_hidden_states"]  # [2,512,1024] bf16
    padding_mask = d["trace.padding_mask"]                    # [1,1,512,512] bf16 (all-zero, predict:131-133)
    ref_predicted_flow = d["trace.predicted_flow"]            # [2,16,1,64,64] bf16
    target = d["output.target"].float()                      # [2,16,1,64,64] f32 (== trace.flow, predict:143)
    loss_weight = d["batch.loss_weight"].float()             # [2]
    dump_loss = d["output.loss_for_backward"].item()
    print(f"[inputs] hidden={tuple(hidden_states.shape)} ts={transformer_timestep.tolist()} "
          f"enc={tuple(encoder_hidden_states.shape)} pad={tuple(padding_mask.shape)} attn_mask=None")

    # ---- transformer: the PR's CosmosTransformer3DModel (AnimaModel.transformer class) ----
    print(f"[load] CosmosTransformer3DModel from {transformer_dir}")
    transformer = CosmosTransformer3DModel.from_pretrained(
        transformer_dir, torch_dtype=BF16
    ).to(DEV)
    transformer.eval()
    transformer.requires_grad_(False)
    # gradient checkpointing: recompute-on-backward, numerically identical, fits 24GB (3090Ti).
    transformer.enable_gradient_checkpointing()
    print(f"[load] heads={transformer.config.num_attention_heads} "
          f"head_dim={transformer.config.attention_head_dim} "
          f"layers={transformer.config.num_layers} "
          f"extra_pos={transformer.config.extra_pos_embed_type} "
          f"concat_pad={transformer.config.concat_padding_mask}")

    # ---- LoRA via the clone's LoRAModuleWrapper ----
    # mirrors AnimaLoRASetup.setup_model: LoRAModuleWrapper(transformer,"transformer",config,filter)
    #   (AnimaLoRASetup.py:61)
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

    # ---- load adapter_before.* (initial LoRA state, B=0) ----
    adapters = load_file(ADAPTERS, device=DEV)
    init_sd = {}
    for k, v in adapters.items():
        if k.startswith("adapter_before."):
            init_sd[k[len("adapter_before."):]] = v          # -> "transformer.<mod>...."
    # dump omits per-module alpha buffer; inject for strict load
    for mod_name in wrapper.lora_modules:
        init_sd[f"transformer.{mod_name}.alpha"] = torch.tensor(lora_alpha)
    wrapper.load_state_dict(init_sd, strict=True)
    print(f"[lora] loaded {len(init_sd)} adapter_before tensors (strict)")

    wrapper.set_dropout(0.0)
    wrapper.to(dtype=torch.float32)                          # lora_weight_dtype FLOAT_32 (AnimaLoRASetup.py:70)
    wrapper.hook_to_module()                                 # (AnimaLoRASetup.py:71)
    wrapper.requires_grad_(True)

    # ---- forward: mirror BaseAnimaSetup.predict:135-141 EXACTLY ----
    # predict: model.transformer(hidden_states=scaled_noisy_latent.to(bf16),
    #            timestep=timestep/1000, encoder_hidden_states=text_out.to(bf16),
    #            padding_mask=padding_mask, return_dict=False)[0]
    # under model.autocast_context (bf16 autocast, setup_optimizations:57).
    # NO attention_mask is passed (predict:135-141 omits it).
    with torch.autocast(device_type="cuda", dtype=BF16):
        predicted_flow = transformer(
            hidden_states=hidden_states.to(BF16),               # predict:136
            timestep=transformer_timestep,                      # predict:137 (already /1000)
            encoder_hidden_states=encoder_hidden_states.to(BF16),  # predict:138
            padding_mask=padding_mask.to(BF16),                 # predict:139
            return_dict=False,                                  # predict:140
        )[0]                                                    # predict:141

    # ---- GATE 1: forward vs frozen dump ----
    fwd_cos = cos(predicted_flow.detach(), ref_predicted_flow)
    fwd_cos_out = cos(predicted_flow.detach(), d["output.predicted"])
    print("\n==== GATE 1: REAL-setup (diffusers@b003a47) forward vs frozen dump ====")
    print(f"  cos(predicted, trace.predicted_flow) = {fwd_cos:.6f}")
    print(f"  cos(predicted, output.predicted)     = {fwd_cos_out:.6f}")
    print(f"  [prior diffusers-direct (0.38.0.dev0) oracle: 0.999099]")

    # ---- loss: mirror calculate_loss -> _flow_matching_losses -> __unmasked_losses ----
    # flow = latent_noise - scaled_latent_image (predict:143) == output.target == trace.flow.
    # __unmasked_losses(:150-155): MSE(predicted.f32, target.f32, none).mean(sample_dims)*mse_strength(1)
    # _flow_matching_losses(:330-331): *loss_scaler.get_scale (1) * loss_weight (1); CONSTANT weight_fn.
    # calculate_loss(:171-177): .mean().
    mean_dim = list(range(1, predicted_flow.ndim))
    losses = F.mse_loss(predicted_flow.float(), target, reduction="none").mean(mean_dim)
    losses = losses * loss_weight  # loss_weight==1
    loss = losses.mean()
    print(f"  oracle loss = {loss.item():.10f}")
    print(f"  dump loss   = {dump_loss:.10f}")
    print(f"  abs diff    = {abs(loss.item() - dump_loss):.3e}")
    verdict = "MATCH" if (fwd_cos > 0.9999 and abs(loss.item() - dump_loss) < 5e-3) else \
        ("CLOSE" if fwd_cos > 0.999 else ("SKEWED" if fwd_cos > 0.99 else "DIVERGE"))
    print(f"  VERDICT: {verdict}")

    # ---- backward + capture per-LoRA grads ----
    loss.backward()

    grads = {}
    name_to_param = {}
    for mod_name, module in wrapper.lora_modules.items():
        for pname, p in module.named_parameters():
            if "lora_down" in pname or "lora_up" in pname:
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
    print("  oracle grads file: <NAME>  where")
    print("    NAME = transformer.{transformer_blocks.<i>.{attn1,attn2}."
          "{to_q,to_k,to_v,to_out.0},ff.net.{0.proj,2}}.{lora_down,lora_up}.weight")
    print("  => Mojo compares oracle grad '<NAME>' to its own grad for the same NAME.")
    print("\n[done]")


if __name__ == "__main__":
    main()
