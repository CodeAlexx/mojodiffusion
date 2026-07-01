# Krea2 Reduced-Depth ai-toolkit AdamW Update Replay

Date: 2026-06-30

Evidence level: reduced-depth ai-toolkit `SingleStreamDiT` block-stack gradient
plus shared-device AdamW update replay. This uses the existing NBLOCKS=4
`krea2_stack_oracle.py` artifact and is not real-cache product parity,
full-28-block parity, txtfusion parity, save/resume equivalence, or convergence
evidence.

Source:

- Oracle producer:
  `serenitymojo/models/krea2/parity/krea2_stack_oracle.py`
- Oracle artifact:
  `serenitymojo/models/krea2/parity/krea2_stack_oracle.safetensors`
- Mojo forward/backward consumer:
  `serenitymojo/models/krea2/parity/krea2_stack_parity.mojo`
- Mojo optimizer consumer:
  `serenitymojo/models/krea2/parity/krea2_stack_adamw_update_replay.mojo`

Validation command:

```bash
/home/alex/serenityflow-v2/.venv/bin/python \
  serenitymojo/models/krea2/parity/krea2_stack_oracle.py
pixi run mojo run -I . \
  -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
  serenitymojo/models/krea2/parity/krea2_stack_parity.mojo
pixi run mojo run -I . \
  serenitymojo/models/krea2/parity/krea2_stack_adamw_update_replay.mojo
```

Current result:

```text
adamw update tensors=64 lr=0.001 betas=(0.9,0.999) eps=1e-08 weight_decay=0.01
OK dumped 522 tensors -> /home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_stack_oracle.safetensors  (7062.2 MB)
VERDICT: PASS - krea2 single-stream stack LoRA fwd+bwd matches torch (cos>=0.999)
[krea2-stack-adamw-update-mojo] reduced_depth_shared_device_abi_replay PASS tensors= 64  numel= 3833856  nonzero_update= 3833856  nonzero_param_error= 602011  max_param_abs= 7.450581e-09  param_l2= 6.767480359105244e-07  nonzero_state_error= 7667712  max_state_abs= 3.7252903e-09  state_l2= 1.3120294196390754e-07  grad_norm= 5.863462  clip_scale= 1.0  syncs= 2
[krea2-stack-adamw-update-mojo] scope=reduced-depth ai-toolkit SingleStreamDiT block-stack gradient plus shared-device AdamW update replay; not real-cache, full-28-block, txtfusion, or convergence parity
```

Claim boundary:

- The existing stack parity still proves reduced-depth ai-toolkit block-stack
  forward output and all `4 * 8 * 2` LoRA gradients against torch autograd.
- The AdamW replay opens the same oracle artifact, uploads the `64` dumped F32
  LoRA before/grad tensors into `DeviceTrainableSet` and `DeviceGradSet`,
  starts `DeviceAdamWState` from zero moments, and runs
  `device_adamw_train_step_update` at step `1`.
- The replay covers `3833856` elements with `3833856` nonzero update elements,
  max param abs `7.450581e-09`, max state abs `3.7252903e-09`, grad norm
  `5.863462`, clip scale `1.0`, and sync count `2`.
- This does not cover the real 512px cache, the full Krea2 adapter surface, or
  txtfusion LoRA modules. The retained default/non-txtfusion Mojo smoke
  trains/saves only main-block LoRA tensors; the newer opt-in
  `KREA2_TXTFUSION_LORA` smoke covers the full 256-adapter key/dtype surface
  but is still not this ai-toolkit numeric oracle.
