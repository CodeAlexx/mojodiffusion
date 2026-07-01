# ZImage Batched-Mask SDPA Backward Gate

Date: 2026-06-30

Evidence level: shared-op gate. This is not ZImage product-loop transformer
forward/backward parity. The all-trainable streamed replay that consumes this
mask path is tracked separately in the selected-grad replay artifacts.

Purpose:

- unblock the shared training attention ABI for B=2 samples with different
  padded sequence lengths
- preserve the legacy broadcast masked backward path
- prove the named training wrapper accepts full per-sample additive masks

Implementation:

- `serenitymojo/ops/attention_backward.mojo`
  - `sdpa_backward_masked` now accepts either legacy F32 `[H*S,S]` masks or
    full batched F32 `[B,H,S,S]` / `[B*H*S,S]` masks.
  - `sdpa_backward_masked_batched` is the fail-loud named entry point for full
    per-sample masks.
- `serenitymojo/ops/attention_train.mojo`
  - exposes `training_sdpa_backward_masked_batched_strict`.
- `serenitymojo/ops/parity/sdpa_bwd_batched_mask_parity.mojo`
  - compares the new batched path against the legacy broadcast path when the
    mask is equivalent.
  - runs a ZImage-like differing per-sample key-tail mask smoke.

Command:

```bash
pixi run mojo run -I . serenitymojo/ops/parity/sdpa_bwd_batched_mask_parity.mojo
```

Observed output:

```text
same-mask batched vs broadcast d_q: ParityResult(cos=1.0, max_abs=0.0, n=192, PASS)
same-mask batched vs broadcast d_k: ParityResult(cos=1.0, max_abs=0.0, n=192, PASS)
same-mask batched vs broadcast d_v: ParityResult(cos=1.0000000000000002, max_abs=0.0, n=192, PASS)
differing per-sample key-tail mask: PASS
ALL PASS
```

Remaining ZImage work:

- build or pass the exact OneTrainer cap-refiner mask for cap rows `160/128`
- build or pass the exact unified main mask for sequence rows `1184/1152`
- keep selected replay on the non-graph streamed masked B2 stack:
  `zimage_stack_lora_forward_main_device_b2_masked_streamed` and
  `zimage_stack_lora_backward_main_device_b2_masked_streamed`
- add a masked graph/slab route or fail-loud exclusion for graph B2, because the
  current autograd-v2 SDPA record path remains no-mask
- all-trainable streamed replay is now collected against
  `adapter_post_clip_grad.*`; product-loop integration and graph/slab masked
  support or fail-loud exclusion remain.
