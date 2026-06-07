# Mojo OneTrainer Presets

Status: product-layer preset catalog with resolved config materialization, not
full OneTrainer preset parity.

`serenitymojo/training/onetrainer_preset_catalog.mojo` maps the target
OneTrainer-style preset names to repo-local Mojo config files and the local
OneTrainer reference preset/config paths. Each entry now carries explicit recipe
family, variant kind, and VRAM tier metadata. The product dry-run materializes a
resolved config under `/tmp/mojo-ot-presets/<preset>.json` so concept/sample
overrides are present in the config path printed for the model runner.
The product entrypoint and real model loops now resolve sampling cadence through
the same shared OneTrainer loop policy, so `validation_prompts_file`,
`sample_definition_file_name`, and disabled step sampling fail or pass the same
way in dry-runs and train loops.

Current LoRA preset aliases:

| Alias | Mojo model type | Local config | OneTrainer reference |
|---|---|---|---|
| `qwen_lora_16gb`, `qwen_lora_24gb` | `qwenimage` | `serenitymojo/configs/qwenimage.json` | `/home/alex/OneTrainer/training_presets/#qwen LoRA {16,24}GB.json` |
| `ernie_lora_8gb`, `ernie_lora_16gb` | `ernie_image` | `serenitymojo/configs/ernie_image.json` | `/home/alex/OneTrainer/training_presets/#ernie LoRA {8,16}GB.json` |
| `anima_lora` | `anima` | `serenitymojo/configs/anima.json` | `/home/alex/OneTrainer-anima-ref/training_presets/#anima LoRA.json` |
| `sd35_lora`, `sd3-5` | `STABLE_DIFFUSION_35` | `serenitymojo/configs/sd35.json` | `/home/alex/OneTrainer/configs/sd35m_100step_baseline.json` |
| `sdxl_1_0_lora` | `sdxl` | `serenitymojo/configs/sdxl.json` | `/home/alex/OneTrainer/training_presets/#sdxl 1.0 LoRA.json` |
| `flux1_dev`, `flux1_dev_lora`, `flux1_lora` | `flux` | `serenitymojo/configs/flux.json` | `/home/alex/OneTrainer/training_presets/#flux LoRA.json` |
| `flux2_lora_8gb`, `flux2_lora_16gb`, `klein9b` | `klein` | `serenitymojo/configs/klein9b.json` | `/home/alex/OneTrainer/training_presets/#flux2 LoRA {8,16}GB.json` |
| `chroma_lora_8gb`, `chroma_lora_16gb`, `chroma_lora_24gb` | `chroma` | `serenitymojo/configs/chroma.json` | `/home/alex/OneTrainer/training_presets/#chroma LoRA {8,16,24}GB.json` |
| `zimage_lora_8gb`, `zimage_lora_16gb` | `zimage` | `serenitymojo/configs/zimage.json` | `/home/alex/OneTrainer/training_presets/#z-image LoRA {8,16}GB.json` |

Important limits:

- A concept file is still user input. The catalog requires a `concept_file_name`
  override instead of pretending there is a production dataset baked in.
- 8GB/16GB/24GB entries are visible as separate preset IDs and metadata, but
  they still share the current repo-local model config unless a model-specific
  local config file exists. The OneTrainer reference path is recorded for each
  variant so the remaining import work is explicit.
- Full-finetune presets are cataloged for Qwen, Anima, Flux2/Klein, Chroma, and
  Z-Image, but product full-finetune loops are not wired yet and fail loud.
- `klein4b` and Z-Image DeTurbo presets are cataloged as not product-wired:
  they must not silently route through the Klein 9B or base Z-Image runners.
- SDXL finetune and SDXL inpaint LoRA reference presets are cataloged, but they
  are not product-wired until the matching full/inpaint product loops exist.
- Plain SD3 is blocked. OneTrainer has `STABLE_DIFFUSION_3` presets, but the
  active target here is SD3.5 (`STABLE_DIFFUSION_35`).
- This catalog proves no-CUDA product input readiness only. It is not a claim
  of loss/gradient/speed parity or production sample quality.

Verification:

```bash
timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_train_loop_policy_smoke.mojo
timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_product_run_smoke.mojo
pixi run mojo run -I . serenitymojo/training/onetrainer_train_dry_run.mojo --preset qwen_lora_16gb /tmp/ot_product_named_concepts.json /tmp/ot_product_named_samples.json
python3 scripts/check_onetrainer_cache_preflight_contract.py
python3 scripts/check_onetrainer_conditioning_contracts.py
python3 scripts/check_vae_sampler_contracts.py
python3 scripts/check_onetrainer_preset_catalog_contract.py
python3 scripts/check_train_loop_cache_contract_bindings.py
```
