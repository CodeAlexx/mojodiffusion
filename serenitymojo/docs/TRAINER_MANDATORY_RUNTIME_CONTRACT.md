# Trainer Mandatory Runtime Contract

Status: binding for every Mojo trainer.

## Output Line

Every trainer must route normal operator progress through
`serenitymojo/training/progress_display.mojo::print_trainer_progress`.

Required display format:

```text
[Klein-lora] step 1613/2000 | epoch 14/17 | loss 0.5909 | grad_norm 0.1527 | 2.1s/step | elapsed 0:55:37 | ETA 0:13:20
```

Use the model's own display name, for example `Klein-lora`, `Ernie-lora`,
`Anima-lora`, `ZImage-lora`, or `LTX2-lora`. Machine-readable `PROG` lines may
exist behind explicit tooling/debug paths, but they must not replace this human
display line.

## Sampling

Every trainer must have a built-in validation sampler path. The sampler must
read prompts, seeds, size, steps, CFG, and cap-cache paths from the shared
`serenity.sample_prompts.v1` JSON read by
`serenitymojo/training/sample_prompt_config.mojo`.

If a model's sampler owns the text encoder at runtime, its prompt JSON may set
`precache_required=false`; it still must read prompt text, negative prompt, seed,
size, steps, and CFG from that shared JSON.

Image validation samples default to `1024x1024`. Image prompt entries must be
`1024x1024` or larger. Do not silently lower validation resolution to avoid OOM;
fix process separation, offload, or sampler memory instead.

## Save And Resume

Every trainer checkpoint cadence must write both files:

- PEFT/ai-toolkit LoRA safetensors for inference and external tools.
- Trainer-state safetensors carrying LoRA A/B plus AdamW `m/v` moments for exact
  Mojo resume.

Resume must load the trainer-state file, not only the PEFT LoRA file. Reloading
PEFT alone resets optimizer moments and is not a valid resume proof.

## Smoke Test

New trainers must support this smoke before being treated as integrated:

1. Sample once from step `0`.
2. Train `10` steps.
3. Save PEFT LoRA plus trainer state.
4. Sample from the step-10 checkpoint.
5. Resume from trainer state and train another `15` steps to total step `25`.
6. Save PEFT LoRA plus trainer state.
7. Sample at step `25`.

The smoke is a wiring check for sampler, save, resume, progress display, finite
loss, finite grad norm, and visible LoRA application. It is not a quality verdict.
