# Mojo Diffusion Numeric API

Shared runtime helpers for PyTorch-compatible diffusion phase math live in
`serenitymojo.ops`. These are common ops, not LTX2-only code.

## BF16 PyTorch-Eager Helpers

Import path:

```mojo
from serenitymojo.ops import (
    torch_bf16_eager_add_scaled,
    torch_bf16_eager_blend_with_f32_mask,
    torch_bf16_eager_velocity_from_x0,
    torch_f32_to_bf16_rne,
)
```

Source:

`serenitymojo/ops/torch_bf16.mojo`

API:

| Helper | Contract |
|---|---|
| `torch_f32_to_bf16_rne(x, ctx)` | F32 tensor to BF16 using PyTorch/CUDA round-to-nearest-even semantics. |
| `torch_bf16_eager_blend_with_f32_mask(noise, clean, mask, ctx)` | Matches `(noise * mask + clean * (1 - mask)).to(bfloat16)` with BF16 `noise`/`clean` and F32 broadcast mask. |
| `torch_bf16_eager_velocity_from_x0(sample, denoised, sigma, ctx)` | Matches `((sample.float() - denoised.float()) / sigma).to(bfloat16)`. |
| `torch_bf16_eager_add_scaled(x, velocity, scale, ctx)` | Matches `(x.float() + velocity.float() * scale).to(bfloat16)`. |

Why this API exists:

PyTorch eager materializes F32 temporaries between operations before the final
BF16 cast. A single fused Mojo kernel can land exactly on a BF16 tie and choose
the other BF16 value. These helpers preserve the PyTorch operation boundaries
and final BF16 round-to-nearest-even behavior for noiser and scheduler handoffs.

Use these helpers when a model’s reference path is PyTorch eager and the tensor
crosses a BF16 storage boundary after noising, velocity reconstruction, or Euler
updates. Do not use them as a blanket replacement for faster fused kernels when
the reference is not PyTorch-eager or bit parity is not required.

Verified by:

```bash
pixi run mojo run -I . serenitymojo/pipeline/ltx2_creator_phase_parity_smoke.mojo
python3 scripts/ltx2_parity_gate.py --only creator_fast_phase_handoff --timeout 120 --hide-gaps
python3 scripts/check_ltx2_dtype_contract.py --scope sidecar
```

Current evidence: the LTX2 creator fast-distilled phase gate is exact for
stage schedules, stage-1/stage-2 GaussianNoiser handoffs, transformer latent
handoff, velocity reconstruction, and Euler `next_latent` at the sampled
first/last steps. The creator dump also records raw transformer velocity at
those steps. Raw velocity is not bit-identical to velocity reconstructed from
the denoised x0 because `X0Model` round-trips through a BF16 x0 boundary; the
gate checks that relationship with a bounded `0.03125` max-abs BF16 tolerance
while keeping noiser/Euler phase math exact.

## LTX2 Sampling API

LTX2 creator-fast users should import through `serenitymojo.sampling`:

```mojo
from serenitymojo.sampling import (
    LTX2Scheduler,
    ltx2_creator_noiser_from_noise,
    ltx2_distilled_sigmas,
    ltx2_stage2_distilled_sigmas,
)
```

`LTX2Scheduler.step(...)` routes BF16 latent/velocity tensors through
`torch_bf16_eager_add_scaled`, so distilled Euler updates preserve the
creator/PyTorch BF16 boundary. Non-BF16 tensors keep the existing generic tensor
algebra path.

`ltx2_creator_noiser_from_noise(clean_latent, torch_noise, scaled_mask, ctx)`
is the public noiser handoff for parity-sensitive runs. It deliberately takes
noise as an input tensor. For bit parity, that noise must come from the
creator/PyTorch oracle or another proven matching RNG contract; Mojo-native
`randn` is not a same-seed replacement for `torch.Generator`.
