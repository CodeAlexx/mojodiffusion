# Trainer Sample Prompts + SerenityBoard Contract

Status: binding trainer runtime contract for Klein and future Mojo trainers.
See also `serenitymojo/docs/TRAINER_MANDATORY_RUNTIME_CONTRACT.md`.

## Shared Sample Prompt JSON

Trainers do not hardcode prompt text, sample size, or cap-cache paths. A train
config points at one shared prompt file:

```json
{
  "validation_prompts_file": "/home/alex/mojodiffusion/serenitymojo/configs/klein9b_alina_samples.json"
}
```

Prompt files use `serenity.sample_prompts.v1`; see
`serenitymojo/configs/sample_prompts.example.json`.

Top-level `defaults` apply to every prompt unless the prompt overrides them:

- `sample_every`: normally `500`
- `sample_at_start`: `true` for baseline samples before step 1
- `save_before_sample`: `true`
- `precache_required`: `true`
- `width`, `height`: image sample size. Image validation defaults to
  `1024x1024`; prompt entries must be `1024x1024` or larger unless the model is
  a non-image/video path with its own explicit frame contract.
- `frames`, `fps`: video trainers can use these; image trainers require `frames=1`
- `steps`, `cfg`, `seed`, `negative`

Each prompt has:

- `id`: stable board/output label
- `prompt`: visible validation prompt
- `negative`: optional override
- `caps.positive`, `caps.negative`: precomputed text-conditioning cache files

## Precache Rule

Training must not load text encoders. For Klein, run the separate Mojo process:

```bash
/tmp/klein9b_precache_sample_prompts /home/alex/mojodiffusion/serenitymojo/configs/klein9b_alina_samples.json
```

That process loads Qwen3-8B, writes every prompt/negative cap listed in the JSON,
then exits so encoder GPU memory is released before training or sampling.

## Sampling Cadence

Production Klein LoRA cadence:

- Step `0`: baseline samples for the configured prompts, before trainer weights load.
- Every `500` steps: save PEFT LoRA, save trainer state, exit the worker process,
  sample configured prompts in standalone sampler processes, then launch the next
  worker process.
- Final `2000`: save final PEFT LoRA/state and sample.

Do not lower validation sample size to avoid OOM. If a large model cannot
sample `1024x1024` while live trainer allocations remain resident, use
process-separated cadence/offload. For Klein 9B, use:

```bash
/home/alex/mojodiffusion/output/bin/train_klein_cadence \
  /home/alex/mojodiffusion/serenitymojo/configs/klein9b.json \
  /home/alex/mojodiffusion/output/bin/train_klein_real \
  /home/alex/mojodiffusion/output/bin/klein_sample_cli
```

To register existing cadence samples without continuing training:

```bash
/home/alex/mojodiffusion/output/bin/train_klein_cadence \
  /home/alex/mojodiffusion/serenitymojo/configs/klein9b.json \
  /home/alex/mojodiffusion/output/bin/train_klein_real \
  /home/alex/mojodiffusion/output/bin/klein_sample_cli \
  500 sampleonly
```

Short smoke runs may label samples as wiring checks, but image outputs are still
`1024x1024` or larger. A 50-step LoRA image is not treated as a useful visual
validation artifact.

## LoRA Saves

Klein writes two files at production checkpoints:

- `alina_lora_step<N>.safetensors`: plain PEFT-style A/B LoRA for external tools
  and ComfyUI-compatible loaders.
- `alina_lora_step<N>.safetensors.state.safetensors`: trainer-only A/B plus AdamW
  `m/v` moments for exact Mojo cadence resume.

Do not reload the PEFT file into a live trainer just to prove save/load. PEFT
files do not carry optimizer moments, so doing that resets AdamW and harms
convergence.

Every trainer must prove the save/resume path with the standard smoke: sample at
step 0, train 10 steps, save PEFT+state, sample, resume trainer state, train to
25 total steps, save PEFT+state, and sample again.

## SerenityBoard

Mojo trainers write `/home/alex/mojodiffusion/output/alina_train/board.db` using
`serenitymojo/training/serenityboard.mojo`. Runtime board writes are pure Mojo
SQLite FFI, not Python.

Scalar tags:

- `loss/train`
- `grad_norm`
- `lr/default`
- `perf/steps_per_sec`
- `perf/sec_per_step`
- `perf/noise_elems_per_sec`

Artifacts/text:

- `samples/<prompt_id>`: PNG artifacts copied to `output/alina_train/blobs/`
- `prompts/<prompt_id>`: prompt text
- `events/save`, `events/save_state`, `events/resume`
- `config/train`, `config/sample_prompts`

## Current Klein State

Verified on 2026-06-02 after the rank-2 concat/slice fast paths and the final
2000-step run:

- The trainer is pure Mojo at runtime. Python is allowed only for parity,
  baselines, and old-log replay.
- Full Klein 9B LoRA training reached step `2000/2000` and saved:
  `alina_lora_final.safetensors`,
  `alina_lora_final.safetensors.state.safetensors`,
  `sample_step2000_alina_garden.png`, and
  `sample_step2000_alina_evening.png`.
- Final line: loss `0.8455`, grad_norm `0.1116`, speed `2.1s/step`.
- Warm training speed held around `2.0-2.1s/step`; only post-sample restarts
  showed `3.1-3.2s/step`.
- Offload fallbacks stayed at `0`.
- Earlier in-process full-run validation lowered sample resolution because 1024
  validation could OOM when trainer allocations remained resident. That is no
  longer an accepted normal-output workaround; use the process-separated 1024+
  sampler/cadence path instead.
- Standalone 1024 sampling from the final LoRA succeeded for
  `sample_step2000_alina_garden_1024.png`; it used `N_IMG=4096`, denoised at
  about `4.8-4.9s/step`, and reported `fallbacks 0`.

Speed changes that are kept:

- Pinned block store: mmap bytes are copied to pinned host memory once; hot
  blockswap prefetches copy pinned host slabs to GPU.
- Training cache tensors are preloaded for compatible 512 latent samples.
- Progress display is shared pure Mojo through
  `serenitymojo/training/progress_display.mojo`.
- Single-block q/k/v LoRA forward now computes the rank projection once and
  applies q/k/v B row ranges from that shared rank tensor.
- LoRA A/B are uploaded to device as BF16 for matmul parity with the Rust path;
  host optimizer/save state remains F32.

Speed experiments that regressed and should not be repeated blindly:

- Saving double-block activation tapes: `DBL_SAVE_TAIL=8` and `4` OOMed during
  forward; `DBL_SAVE_TAIL=2` fit but slowed the step to about `5.1s`.
- Default-stream host-to-device copies in the turbo loader: correct but slower
  (`4.5-4.7s/step`) because it removes overlap.
- Combining single-block base q/k/v into one `[S,3D]` projection plus slices:
  slower (`4.8-4.9s/step`) due to materialized slice copies.
- Precasting the single-block normalized activation once to BF16 and reusing it
  across q/k/v/gate-up: slower (`5.0s/step`) from added memory pressure and
  scheduling cost.
- Batched LoRA grad D2H readback using the early tensor-grad path regressed to
  about `4.8-4.9s/step`; do not re-enable without a smaller proof.

Speed change that worked:

- `ops/tensor_algebra.mojo` now has rank-2, `dim=1` concat/slice kernels for
  F32/BF16/F16. This removed the tiny D2D copy storm from Klein q/k/v and gate
  splits. Nsight D2D copies dropped from `321626` to `1013`, cutting D2D GPU
  time from `424.10 ms` to `30.49 ms`.

Current speed bottlenecks to attack next:

- True copy-engine blockswap is still missing. `DeviceStream` can launch kernels
  and record/wait events, but Mojo `DeviceContext.enqueue_copy` is default-stream
  only in the current API. The existing turbo path uses a side-stream GPU copy
  kernel, so it overlaps but still consumes SM time. A real async DMA bridge is
  the likely route to the Rust target.
- LoRA grad readback still performs many host sync points in the active path.
- Attention backward is still the decomposed math path; compare against the Rust
  kernels and the inference-side Klein kernels before changing the trainer.
- Loss and `d_loss` are host-side; add device MSE/loss-grad to remove the final
  forward host round-trip before backward.
- Single-block W2 slicing and scratch allocations still allocate outside the
  ideal Rust-style path.
