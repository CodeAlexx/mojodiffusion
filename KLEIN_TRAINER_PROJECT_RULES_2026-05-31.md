# Klein Trainer Project Rules - 2026-05-31

- This is the pure Mojo trainer/runtime path for Klein LoRA training.
- Python is acceptable for local development inspection and parity tests only.
- Do not add Python to the final Mojo trainer, sampler, or runtime path.
- Keep the Rust side pure Rust and the Mojo side pure Mojo.
- Do not make the Mojo runtime depend on Rust, and do not make Rust runtime code depend on Mojo.
- LoRA saves should stay plain PEFT-style safetensors compatible with ai-toolkit/diffusers-style consumers.
- Follow OneTrainer preset training parameters where applicable, but use block swapping/offload for Klein memory instead of quantization.
