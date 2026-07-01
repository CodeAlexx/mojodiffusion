# Krea2 Trainable Surface Blocker

Date: 2026-06-30

Evidence level: trainable-surface blocker only. This compares an existing
ai-toolkit Krea2 LoRA output against the current Mojo Krea2 real-cache smoke
LoRA output. It is not gradient parity, optimizer parity, loss replay,
save/resume equivalence, speed evidence, or convergence evidence.

Update: this document is retained for the historical default-path real-cache
artifact listed below. The opt-in `KREA2_TXTFUSION_LORA` real-cache smoke now
saves the full 256-adapter surface and passes exact key/shape/dtype comparison
against the same ai-toolkit output. See
`artifacts/training_perf/krea2_txtfusion_devicegrad_realcache_smoke_2026-06-30.md`.

Inputs:

- ai-toolkit output:
  `/home/alex/ai-toolkit/output/my_first_lora_v1/my_first_lora_v1_000002994.safetensors`
- Mojo real-cache smoke output:
  `/tmp/krea2_devicegrad_realcache_smoke/krea2_devicegrad_realcache_smoke_2.safetensors`

Validation command:

```bash
python3 scripts/check_krea2_trainable_surface.py --expect-known-mismatch
```

Current result:

```text
[krea2-surface] ai_toolkit: path=/home/alex/ai-toolkit/output/my_first_lora_v1/my_first_lora_v1_000002994.safetensors total=512 blocks=448 txtfusion=64 non_block=64 target_prefixes=256 dtypes=['torch.bfloat16']
[krea2-surface] mojo: path=/tmp/krea2_devicegrad_realcache_smoke/krea2_devicegrad_realcache_smoke_2.safetensors total=448 blocks=448 txtfusion=0 non_block=0 target_prefixes=224 dtypes=['torch.bfloat16']
[krea2-surface] delta: common_keys=448 missing_from_mojo=64 missing_txtfusion=64 missing_non_txtfusion=0 extra_in_mojo=0 block_key_delta=0 shape_mismatch=0 dtype_mismatch=0
[krea2-surface] missing_sample=['diffusion_model.txtfusion.layerwise_blocks.0.attn.gate.lora_A.weight', 'diffusion_model.txtfusion.layerwise_blocks.0.attn.gate.lora_B.weight', 'diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_A.weight', 'diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_B.weight', 'diffusion_model.txtfusion.layerwise_blocks.0.attn.wo.lora_A.weight', 'diffusion_model.txtfusion.layerwise_blocks.0.attn.wo.lora_B.weight', 'diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_A.weight', 'diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_B.weight']
[krea2-surface] PASS known_mismatch ai_toolkit_total=512 mojo_total=448 common_keys=448 missing_txtfusion=64 block_key_delta=0 shape_mismatch=0 dtype_mismatch=0 ai_target_prefixes=256 mojo_target_prefixes=224
[krea2-surface] scope=trainable-surface blocker only; not gradient, optimizer, loss, save/resume, speed, or convergence parity
```

Claim boundary:

- The main block LoRA key surface matches between the inspected ai-toolkit output
  and Mojo real-cache smoke output: `448` block tensors on both sides and
  `block_key_delta=0`.
- The common main-block LoRA tensor names, shapes, and dtypes now match:
  `shape_mismatch=0`, `dtype_mismatch=0`, and both outputs are BF16. The
  inspected ai-toolkit config and Mojo smoke config are both rank/alpha
  `32` / `32`.
- ai-toolkit also saved `64` BF16 `diffusion_model.txtfusion.*` LoRA tensors.
- Mojo saved no `txtfusion` LoRA tensors in the current Krea2 real-cache smoke.
- Full Krea2 parity remains blocked until the txtfusion LoRA surface is either
  trained/saved in Mojo with matching ai-toolkit semantics or explicitly
  excluded by a source-backed reference configuration that does not train it.
