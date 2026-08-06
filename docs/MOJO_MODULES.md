# serenitymojo module map (training-autograd port)

> One row/paragraph per public module. Read this once at session start to know
> where things live. Modeled on flame-core/docs/FLAME_MODULES.md.
>
> **Status legend**
> - **PROVEN** — has a parity gate (`*_parity.mojo` / `*_bwd_parity.mojo`)
>   that the master handoff records the lead re-ran on a clean serial build,
>   cos ≥ 0.999 vs PyTorch.
> - **PROVEN-TOY** — gate is green but only at toy/32-aligned shapes; NOT
>   verified at real model dims. (The sdpa-bwd H=30 trap, §below, is the
>   cautionary tale.)
> - **SCAFFOLD** — built + imports, used by a smoke/assembly file, but no
>   cos-gate against torch yet.
> - **INFERENCE** — pre-existing inference-only forward path (the bulk of the
>   tree); inert to training unless a backward partner is wired.
>
> Every claim below cites a `file:line` actually read in serenitymojo/.
> Read files that may contain NUL display artifacts via
> `python3 -c "print(open('PATH','rb').read().decode('utf-8','replace'))"`.

flame-core is a 100K-line Rust+CUDA library. serenitymojo (package name in
`serenitymojo/__init__.mojo`) is a ~237K-LOC pure-Mojo tree that started as an
**inference** port (Tensor, model forwards, VAEs, samplers, safetensors reader)
and has had a **training-autograd spine** grafted on top (the 2026-05-30 port).
The training spine is a thin, sharply-bounded set of modules; the rest of the
tree is inference forwards that the training path consumes as fixed kernels.

Operational trainer/runtime guidance lives in
`docs/MOJO_TRAINER_RUNTIME_API_GUIDE.md`. Read it before changing offload,
scratch-ring, Z-Image, Klein, or production LoRA/full-finetune loops; this module
map catalogs APIs, while that guide records the production usage rules and
lessons from real runs.

SerenityTrainer-port core/runtime changes from the current trainer port are summarized
in `serenitymojo/docs/SERENITY_TRAINER_CORE_PORT_STATUS_2026-06-06.md`. Use that short
doc for the current shared config, policy, optimizer, dtype, artifact-consumer,
and "not production-ready yet" boundaries.

---

## Core types (TRAINING-critical)

### `tensor.mojo` — the central `Tensor`
`struct Tensor(Movable)` (`tensor.mojo:32`). Holds a
`DeviceBuffer[DType.uint8]` of raw element bytes, host-side `_shape: List[Int]`,
a runtime `_dtype: STDtype`, and — the only training field — `var id: Int`
(`tensor.mojo` field block; `0 = untracked`, the inference default; a `Tape`
stamps a fresh nonzero id when a tensor enters the graph). **Move-only**: it
uniquely owns its device buffer, mirroring the loader's ownership discipline,
so it CANNOT be a `List`/`Dict` element directly — collections box it as
`ArcPointer[Tensor]` (see autograd). Storage is monomorphic uint8 bytes,
`.bitcast` to the concrete element `DType` at each op boundary. Construction is
H2D: `from_host(values, shape, dtype, ctx)` (values first, ctx last) or
`from_view` (weight-load path). `to_host` reads back for parity. Has a
`clone(ctx)` (per master handoff §2) used by the tape to save graph tensors.
**Status: PROVEN** (every training gate stands on it).

### `io/dtype.mojo` — `STDtype`
The runtime dtype tag (`STDtype.F32`, `STDtype.BF16`, …) carried by every
`Tensor`. `name()` / `byte_size()` / `to_mojo_dtype()`. Note the idiom:
`STDtype.F32` is a value, **not** `STDtype.f32()` (handoff §4). Status:
**PROVEN** (used everywhere).

### `parity.mojo` — `ParityHarness`
`struct ParityHarness` (`parity.mojo`) + `@fieldwise_init struct ParityResult`
(`cos`, `max_abs`, `passed`, `n`). Compares a GPU `Tensor` against a host
`List[Float32]` reference: cosine similarity + max-abs-diff, both computed in
**F64 on the host** after `to_host()` so the comparison never loses precision
vs the device data. `DEFAULT_COS_THRESHOLD = 0.999` (`parity.mojo`). The Python
(numpy/torch) oracle is DEV-ONLY — references are generated offline and passed
in as host lists; nothing in this module touches Python. This is the Mojo
analog of flame-core's `parity::ParityHarness`. **Status: PROVEN** (the gate
mechanism itself).

---

## The autograd engine (TRAINING spine)

### `autograd.mojo` — reverse-mode tape engine
Port of flame-core `src/autograd.rs` tape (`autograd.mojo:1-2` header). Key
design (USER decision, header lines 4-7): **EXPLICIT threaded `Tape` struct, no
globals** — `Tape` is passed `mut`; serenitymojo has no global mutable state and
Mojo 1.0.0b1 globals are unreliable.

| Symbol | Where | Role |
|---|---|---|
| `comptime TArc = ArcPointer[Tensor]` | `autograd.mojo:48` | the box that lets move-only `Tensor` live in collections |
| `OP_ADD..OP_MSE` (9 codes 0–8) | `autograd.mojo:52-60` | tape op-kind constants: ADD/SUB/MUL/MATMUL/LINEAR/RMSNORM/SILU/SWIGLU/MSE |
| `struct TapeEntry(Copyable, Movable)` | `autograd.mojo:244` | `out_id`, `op_kind`, `lhs_id`, `rhs_id`, `saved0/1: TArc`, `dim_m/n/k`, `third_id`, `saved2: Optional[TArc]` (the 3-input/linear slot) |
| `struct Tape(Movable)` | `autograd.mojo:278` | the tape itself; entries + id→grad accumulation |
| `def backward(...)` | `autograd.mojo:441` | reverse walk → id→grad map |
| `def ones_like` / `_accum` | `:210` / `:427` | seed grad / grad accumulation |

Imports its per-op backward kernels from the `ops/*_backward.mojo` family
(`autograd.mojo:33-42`): `mm_backward`, `linear_backward`, `rms_norm_backward`,
`silu_backward`, `swiglu_backward`, `mse_backward`. **9 ops are wired into
`tape.backward()`** (master handoff §2); the remaining ~68 backward arms exist
as kernels but are hand-chained, not yet tape-dispatched.
**Status: PROVEN** (tape gates: `autograd_*_smoke.mojo`, all cos ≥ 0.99999999).

### `autograd_*_smoke.mojo` — tape gates (7 files)
`autograd_smoke` (add/sub/mul), `autograd_matmul_smoke`,
`autograd_linear_smoke`, `autograd_rmsnorm_smoke`, `autograd_silu_smoke`,
`autograd_swiglu_smoke`, `autograd_mse_smoke`. Each defines `main()` and drives
one op through `tape.backward()` vs a torch reference. **Status: PROVEN** (the
op→gate table is master handoff §2).

---

## Backward op kernels (TRAINING) — `ops/*_backward.mojo`

These are the hand-written GPU backward partners of the inference forwards.
All-F32 interior (the master-precision training path); BF16/F16 only at the
storage boundary. All parity-gated under `ops/parity/`.

| Module | Public backward defs (per header) | Forward partner | Status |
|---|---|---|---|
| `ops/attention_backward.mojo` | `sdpa_backward[B,S,H,Dh]` and opt-in `sdpa_backward_scratch[B,S,H,Dh]` → (d_q,d_k,d_v); `_softmax_bwd_rows_f32` kernel | `ops/attention.mojo` (math-mode SDPA fwd) | **PROVEN** for non-degenerate parity; scratch variant is **PROVEN** against the normal path by `scratch_ring_smoke` d_q/d_k/d_v equality. |
| `ops/linalg_backward.mojo` | `mm_backward`, `bmm`, `linear_backward`, `addbias` grads, plus opt-in scratch d_x-only helpers including row-split accumulation (transposed GEMMs via vendor BLAS) | `ops/linear.mojo` | **PROVEN** (`linalg_bwd_parity`; scratch row-split helper covered by `scratch_ring_smoke`) |
| `ops/norm_backward.mojo` | `rms_norm_backward` (+`RmsNormBackward` struct), layer_norm, group_norm (NHWC) d_x/d_g/d_b | `ops/norm.mojo` | **PROVEN** (`norm_bwd_parity`, 8 grads) |
| `ops/activation_backward.mojo` | `silu_backward`, relu/sigmoid/tanh/gelu(tanh-approx, verbatim from flame-core `gelu_backward.cu`) | `ops/activations.mojo` | **PROVEN** (`activation_bwd_parity`, 5 arms) |
| `ops/reduce_backward.mojo` | sqrt/square/log/softmax/logsoftmax/sum/mean backward | `ops/reduce.mojo` | **PROVEN** (`reduce_bwd_parity`, 7 incl softmax@1024) |
| `ops/loss_swiglu_backward.mojo` | `mse_backward`, `huber`, `swiglu_backward` (+`SwigluGrads`) | `ops/activations.mojo` / loss | **PROVEN** (`loss_swiglu_bwd_parity`). Caveat: `mse_backward` has had transient false-"unimportable" reports (handoff §4 — a serial-build cache artifact, imports fine clean) |
| `ops/rope_struct_backward.mojo` | `rope_backward` (Interleaved + Halfsplit), `qkv_split_permute_backward`, `gate_residual_backward` | `ops/rope.mojo`, `ops/elementwise.mojo` | **PROVEN** (`rope_struct_bwd_parity`) |
| `ops/shape_backward.mojo` | Cat/Split/Slice/Reshape/Transpose/Permute/Broadcast/Repeat/Where/Clamp/Maximum/Minimum/Cast/IndexSelect (grad routing) | various shape ops | **PROVEN** (`shape_bwd_parity`, 18 Tier-0 arms) |
| `ops/conv2d_backward.mojo` | d_x/d_w/d_b naive NHWC/RSCF (no SDK conv-backward) | `ops/conv.mojo` (SDK fwd) | **PROVEN** (`conv2d_bwd_parity`) |
| `ops/pool_backward.mojo` | maxpool2d + upsample-nearest2d backward (NHWC, VAE path) | `ops/conv.mojo`, `models/vae/upsample.mojo` | **PROVEN** (`pool_bwd_parity`) |
| `ops/celoss_embed_backward.mojo` | CrossEntropy / NLL / BCE / Embedding backward | — | **PROVEN** (`celoss_embed_bwd_parity`) |

Master handoff §2 totals this as **~68 backward arms cos ≥ 0.999 vs torch**
(+BF16 variants ≥ 0.99).

---

## The modular trainer split (2026-05-30) — `training/` (shared) vs `models/<m>/`

> Design doc: `docs/architecture/RECOMMENDED_TRAINER_STRUCTURE.md` (Stage 1 IMPLEMENTED + compile-
> verified, RC=0). The SerenityTrainer-style seam: the SHARED training pipeline is
> written once in `training/`, and the only per-model surface is `models/<m>/`
> (block fwd/bwd, σ-map, LoRA-target map, weight-key layout). This replaces the
> EDv2 pattern where ~60–70% of each of 17 trainers was duplicated boilerplate.
> A placement correction (header lines 7–18): the shared pipeline lives in
> **`training/`** (NOT a new `pipeline/` dir — that name already means INFERENCE
> smokes here). The agnostic primitives (optim/schedule/loop/dit_block/checkpoint)
> were LEFT in `training/` so the proven cos≥0.999 gate suite was untouched.

### Shared pipeline — `training/train_config.mojo`, `training/train_step.mojo`
- **`training/train_config.mojo`** — `struct TrainConfig(Copyable, Movable)`
  (`train_config.mojo:15`): the ONE training-config descriptor. Unifies the old
  per-file KleinTrainConfig/ZImageTrainConfig (identical modulo n_layers). Carries
  the RUNTIME recipe scalars (`lr`, `timestep_shift`, `lora_rank`, `lora_alpha`,
  `eps`, `adapter_algo`) + nominal model dims (`d_model`/`n_heads`/`head_dim`/
  `mlp_hidden`/
  `n_layers`). Per-model constructors return this type. **Mojo constraint** (header
  8–10): attention SHAPE (B,S,H,Dh) is a COMPTIME param of the step, NOT carried
  here. **Status: SCAFFOLD** (plumbing struct; exercised by the gated step).
- **`training/train_step.mojo`** — the SHARED, model-agnostic LoRA training step
  (the ~85% klein/zimage used to duplicate verbatim). Key defs: `train_step[Bp,Sp,
  Hp,Dhp](...)` (`:261`) = one step (flow-match v-target → LoRA delta on block
  input → `dit_block_forward` → MSE → `dit_block_backward` → LoRA backward → AdamW
  on LoRA params only; base block weights FROZEN); `struct LoraAdapter` (`:120`,
  `a:[rank,in]`, `b:[out,rank]`, `scale=alpha/rank`, +AdamW m/v moments);
  `_lora_fwd`/`_lora_bwd`/`_lora_adamw` (proven `linear`/`linear_backward`/
  `adamw_step` path); `run_synthetic(cfg, ctx)` (`:306`) = the generic short LoRA
  loop on synthetic data shared by every model entry point. **Status: SCAFFOLD**
  (asserts loss finite / grads nonzero / loss decreases at down-scaled comptime
  dims `_M=4,_D=8,_H=2,_Dh=4,_FF=16`; composes only PROVEN primitives — dit_block
  fwd/bwd, flow_match, AdamW. Real run = swap comptime synth dims for cfg dims +
  weight loader. NOT a cos-gate against torch by itself).

### Per-model — `models/zimage/`, `models/klein/`
- **`models/zimage/config.mojo`** — `def zimage() -> TrainConfig` (`:12`):
  dim 3840, n_heads 30, head_dim 128, mlp_hidden 2560(*placeholder), 30 single
  blocks; lr 3e-4, alpha 1.0, eps 1e-5. **`models/zimage/train.mojo`** — thin
  `main()` calls `run_synthetic(zimage(), ctx)`. **Status: SCAFFOLD** (synthetic).
- **`models/klein/config.mojo`** — `klein_9b()` (`:17`, inner 4096 / 32 heads /
  head_dim 128 / mlp 12288 / 8 double+24 single=32 blocks; lr 4e-4, shift 1.8,
  rank/alpha 16, eps 1e-6 — dims CONFIRMED from real safetensors headers) +
  `klein_4b()` (`:24`, inner 3072 / 24 heads / mlp 9216 / 5 double+20 single=25
  blocks). **`models/klein/train.mojo`** — thin `main()` → `run_synthetic` on both.
  **Status: SCAFFOLD** (synthetic).
- **`models/klein/double_block.mojo`** (1437 L) — Klein FLUX.2 DOUBLE-stream DiT
  block fwd (saving acts) + hand-chained bwd, packaged as a reusable unit (the
  `dit_block.mojo` pattern DOUBLED: img+txt streams coupled by ONE joint attention;
  mirrors `klein_dit.mojo` `_double_block`). Host `List[Float32]` API boundary.
  Also `double_block_lora_forward/backward` (img/txt × qkv/proj LoRA). **Status:
  PROVEN per source header** ("gated 28/28 vs torch"; gates `parity/double_block_
  parity.mojo` + `double_block_lora_parity.mojo` with full `ref_*`/`lref_*` `.bin`
  reference sets present). NOTE: built THIS session, not yet in the master handoff's
  lead-verified table — reconcile.
- **`models/klein/single_block.mojo`** (828 L) — Klein FLUX.2 SINGLE-stream block
  fwd+bwd (the double pattern HALVED+flattened: parallel attn+MLP, fused linear1
  qkv+gate_up channel split, linear2 join). + `single_block_lora_*` (qkv-rows on w1
  + output LoRA on the attention half of w2). The scratch training path now
  stores packed W2 column blocks (`w2_att`, `w2_mlp`) and avoids materializing
  `out_in` / full `d_out_in`; reference callers still keep the original W2 by
  default. **Status: PROVEN per source header** (gates `single_block_parity`
  + `single_block_lora_parity`, `slref_*` refs present). Same session caveat.
- **`models/klein/klein_stack.mojo`** — FULL Klein DiT stack: COMPOSES the
  parity-verified double+single blocks into the complete model (input proj →
  modulation → N double → concat → N single → final layer), per-block recompute in
  backward (gradient checkpoint at block granularity, fits 8+24 blocks in 24 GB).
  Mirrors `klein_dit.mojo` `forward_full`. The training structs now carry saved
  activations with `ArcPointer[Tensor]` device carriers so inter-block handoff
  avoids host-list traffic.
- **`models/klein/klein_stack_lora.mojo`** (835 L) — the stack WITH LoRA on every trained
  projection: per-block LoRA variants for every block + collects adapter d_A/d_B
  into one flat `KleinLoraSet`, supports AdamW step + PEFT save. The hot trainer
  path uses device input tokens, resident block/modulation tensors, checkpoint-tail
  single-block saves (`SGL_SAVE_TAIL = 9`), and can skip unused input-token/aux
  modulation grads in the real LoRA optimizer path. It also exposes
  `KleinLoraDeviceSet` / `klein_lora_set_to_device`, so the trainer uploads LoRA
  A/B once per step and reuses them across forward, backward recompute, and LoRA
  backward. The real trainer uses the `_moddev_rope` entry points so per-step
  modulation chunks and RoPE tables stay device-resident. Single-block LoRA
  backward also reuses the saved attention-flat tensor instead of slicing it
  back out of `out_in` twice. The no-aux real trainer path skips gate-residual
  `y` recomputes for discarded gate/modulation grads, and checkpointed
  single-block backward recompute uses a save-only path that stops at the saved
  attention/MLP activations instead of producing a discarded block output or
  concatenated `out_in`. The real trainer now routes
  scratch-aware stack wrappers through a shared two-slab `ScratchRingAllocator`
  (512 MiB x 2) for block-local concat/slice temporaries, scratch-backed frozen
  linear dx outputs, scratch-backed SDPA backward work buffers, direct row-split
  W1 single-block forward/backward, and direct fresh q/k/v row outputs with
  scratch LoRA row deltas. Packed W2 scratch projection removes the remaining
  single-block `out_in`/full-dx materialization; the real trainer loads packed
  W2 only (`keep_w2=False`) to avoid GPU duplication. Together with the shared
  F32 no-bias `linear` fast path, latest clean 4B timing band is `2.0461085`,
  `2.067908`, loss `2.734082`, grad `0.17687473`; this meets the few-seconds
  target.
- **`models/klein/lora_block.mojo`** (306 L) — LoRA-on-projection helpers shared by the
  double/single LoRA variants; SAME math as `train_step.mojo` plus the projection
  input-grad contribution `d_x_lo`. The hot `*_device` helpers keep activation and
  `d_x_lo` tensors on device and batch `d_A`/`d_B` readback into one sync.
  `LoraAdapterDevice` boxes A/B as `ArcPointer[Tensor]`; legacy host-list helpers
  remain for compatibility/parity.
- **`models/klein/weights.mojo`** — G1 real-safetensors → training weight
  structs. Loads the 12-tensor-per-double-block + 4-per-single-block key layout
  (same keys the inference `klein_dit.mojo` reads) into the host `List[Float32]`
  weight structs the verified block fwd/bwd consume. Also exposes
  `KleinStepModWeights`, `load_klein_step_mod_weights`,
  `build_klein_step_mods_cached`, and `build_klein_step_mods_device_cached`, so
  frozen timestep/modulation weights are loaded once before timed training steps
  and reused device-resident. The device-cached variant returns `ModVecsDevice`
  / `SingleModVecsDevice` chunks for the hot trainer path. `load_single_block_weights`
  accepts `keep_w2`; reference/parity callers keep the original full W2 by
  default, while the real scratch trainer keeps only packed W2 column blocks to
  avoid duplicating GPU memory. **Status: PROVEN for
  cached mods** (`klein_step_mod_cache_smoke`: host and device chunks all
  max_abs 0.0).
- `models/klein/parity/load_{double,single}_block_smoke.mojo` — real-weight load
  smokes for the block weight structs.

## Scratch allocation — shared opt-in memory

| Module | Purpose / key defs | Status |
|---|---|---|
| `scratch_ring.mojo` | `ScratchRingAllocator`: SerenityTrainer-style fixed GPU scratch slabs (`DType.uint8`), 16-byte aligned sub-buffer allocation, forward allocation from the head, reverse allocation from the tail for backward/recompute frames, explicit `mark`/`rewind`/`reset`, and Tensor wrappers over `create_sub_buffer`. Matches the local SerenityTrainer pattern in `docs/RamOffloading.md` and `modules/util/LayerOffloadConductor.py`: persistent int8 cache tensors, typed slice/view reinterpretation, and ordered forward/backward allocation. The allocator is shared infrastructure for any model, but callers must opt in and own the frame lifetime; it is not a global Tensor allocator. | **PROVEN** (`scratch_ring_smoke`: clone, alignment, mark/rewind, reset, forward+reverse allocation) |
| `ops/tensor_algebra_scratch.mojo` | Opt-in scratch-backed hot shape helpers: `concat2_scratch`, `concat3_scratch`, `slice_scratch`. The F32 rank-2 dim-1 path keeps specialized kernels; other valid ranks/dims use copy-backed scratch output. Each helper can allocate from the ring head or tail (`reverse=True`) for backward/recompute frames. Kept separate from `ops/tensor_algebra.mojo` so normal model imports do not compile or use scratch kernels unless explicitly requested. | **PROVEN** (`scratch_ring_smoke`: concat2/slice/concat3 plus rank-4 generic concat/slice parity) |
| `ops/linear.mojo` | `linear_scratch`: opt-in F32 no-bias linear forward whose output storage comes from `ScratchRingAllocator`; `linear_rows` / `linear_rows_scratch`: fresh or scratch output over a contiguous row range of row-major `[out,in]` weights; `linear_two_inputs_scratch`: `x0@w0.T + x1@w1.T` with BLAS `beta=1`, used when a model pre-packs weights by input block. Bias and non-F32 full-linear paths fall back to normal `linear`; scratch helpers are F32-only. | **PROVEN** (`scratch_ring_smoke`: `scratch linear fwd`, `fresh linear rows`, `scratch linear rows`, `scratch linear two`) |
| `ops/tensor_algebra.mojo` | `add_in_place_f32`: owned-buffer F32 in-place accumulation helper for paths where allocating a fresh add output would just be copied forward. Also owns device-side runtime constants: `zeros_device`, `full_device`, and `scalar_f32_device`, which avoid synchronized `Tensor.from_host([scalar])` uploads in denoise/train hot loops. | **PROVEN** (`scratch_ring_smoke`: `scratch add in place`; `algebra_smoke`: zero/full/scalar device constructors) |
| `ops/linalg_backward.mojo` | `linear_backward_dx_split_scratch`: opt-in frozen-weight d_x helper for two contiguous output-row grad blocks; uses BLAS `beta=1` to accumulate without materializing a concat. | **PROVEN** (`scratch_ring_smoke`: `scratch linear split`) |
| `ops/attention_backward.mojo` | `sdpa_backward_scratch`: opt-in decomposed SDPA backward that keeps the large recompute/work buffers in a nested scratch frame, rewinds them before return, and returns normal fresh d_q/d_k/d_v tensors. | **PROVEN** (`scratch_ring_smoke`: scratch d_q/d_k/d_v equal normal `sdpa_backward`) |

## Training orchestration — `training/`

| Module | Purpose / key defs | Status |
|---|---|---|
| `training/optim.mojo` | `adamw_step` (decoupled WD, ported to match torch.optim.AdamW / SerenityTrainer exactly: WD on `p` before the adaptive Adam subtraction, NOT folded into `g`), SGD+momentum, global-norm grad clip | **PROVEN** (`optim_converge_parity`: AdamW bowl ratio 9.8e-15) |
| `training/schedule.mojo` | loop POLICY: `flow_match_noise_target` (the real Z-Image v-target: `x_t=(1-σ)·latent+σ·noise`, `target=noise-latent`), logit-normal timestep + qwen shift, EMA, grad-accum. Ported to match EDv2 `train_qwenimage.rs` | **PROVEN** (`schedule_parity`) |
| `training/checkpoint.mojo` | gradient checkpointing + activation offload (toy linear→silu). **Critical finding**: Mojo 1.0.0b1 cannot store a heterogeneous captured closure in a struct field → no `recompute_fn` closure like flame-core; the recompute is open-coded instead | **PROVEN** (`checkpoint_parity` toy) |
| `training/checkpoint_block.mojo` | gradient checkpointing for a FULL DiT block (saves only block input x, recomputes the whole block in backward) | **PROVEN** (`checkpoint_block_parity`: full-block dx cos 0.99999999, offload round-trip max_abs 0) |
| `training/dit_block.mojo` | reusable `dit_block_forward` / `dit_block_backward` unit (2 residual branch points, 2 fan-out points). Data crosses the API as host `List[Float32]`, not GPU Tensors (the move-only constraint; matches the proven inline gate) | **PROVEN** (`dit_block_unit_parity` + `block_composed_parity` "BLOCK COMPOSITION SOUND" at H=2) |
| `training/loop.mojo` | reusable F32-master / BF16-compute training-loop harness. **grads-as-input, NOT callbacks** (no storable closures): caller runs its own fwd+bwd, hands BF16 grads back; harness owns F32 masters + AdamW (m,v) + step `t` + grad accumulators; resumable via the byte-exact safetensors writer | **PROVEN** (`loop_parity`: trains, checkpoint round-trip byte-exact, resume continues) |
| `training/full_finetune_contract.mojo` | no-CUDA SerenityTrainer full-finetune readiness map. Pins the local target families with `TrainingMethod.FINE_TUNE` registrations, keeps Anima pointed at `/home/alex/SerenityTrainer-anima-ref`, separates shared full-weight save/load plus F32 optimizer sidecar scaffolding from unsupported real product loops, and names required resume sidecar keys: `param.N`, `adam_m.N`, `adam_v.N`, `__meta__=[t_step, accum_count]`, plus the tensor-name manifest | **CONTRACT** (`full_finetune_contract_smoke`: target map, unsupported-loop boundary, sidecar schema failures) |
| `training/serenity_trainer_train_loop_policy.mojo` | shared SerenityTrainer real-loop policy used by Qwen, Ernie, Anima, SD3.5, SDXL, Flux.1, Klein, Chroma, and Z-Image. Centralizes LoRA/AdamW validation, checkpoint/offload policy, sample cadence, save cadence, output paths, step paths, final-vs-step paths, and `.state.safetensors` sidecar naming without creating CUDA context. | **CONTRACT** (`serenity_trainer_train_loop_policy_smoke` plus all nine train-control wiring smokes) |
| `training/serenity_trainer_train_dry_run.mojo` | no-CUDA product entrypoint dry run. Reads a SerenityTrainer-style config, validates concept/sample/product-run requirements through `serenity_trainer_product_run.mojo`, preflights `continue_last_backup` internal backup sidecars, prints workspace dirs and the resolved `serenitymojo/training/<runner>.mojo <config> <max_steps>` command without spawning the model loop or creating `DeviceContext()` | **CONTRACT** (`serenity_trainer_product_run_smoke` plus direct dry-run command) |
| `training/serenity_trainer_cache_preflight.mojo` | no-CUDA product cache preflight. Binds each parsed `model_type` to the SerenityTrainer text-conditioning cache contract and VAE encode/cache contract before product run, exposes required train/sample cache fields, and fails loudly when `only_cache` is requested for a model whose current Mojo encoder cannot emit SerenityTrainer raw VAE cache. | **CONTRACT** (`serenity_trainer_cache_preflight_smoke`: all current target model contracts, Qwen/SD3.5 raw-cache pass, plain SD3 block, Klein/Flux/Chroma raw-cache block) |
| `scripts/check_chroma_sdxl_mojo_update_consumers.py` | strict Chroma/SDXL update-consumer guard. Separates update-delta artifact consumers from full AdamW parity. Current Chroma/SDXL Mojo smokes open the same step/adapters/meta files and compare sampled `adapter_post -> adapter_after` deltas, but `--require-mojo-parity` still exits `2` until gradients/backward replay and model AdamW execution are present. | **CONTRACT / BLOCKED FOR FULL PARITY** |
| `scripts/check_zero_lr_mojo_state_init_consumers.py` | strict Qwen/Ernie/Anima state-init guard. Requires Mojo artifact gates to open real SerenityTrainer step/adapters/meta files and validate zero-lr optimizer state initialization. This is accepted state-init consumption only, not nonzero update parity. | **CONTRACT** |
| `scripts/check_train_loop_cache_contract_bindings.py` | report-only train-loop inventory. Scans Qwen, Ernie, Anima, Klein, Z-Image, Chroma, Flux.1, SD3.5, and SDXL loops for `TrainConfig` read/validate, `only_cache` before `DeviceContext()`, sample cadence helpers, explicit text/VAE cache preflight binding, and host-F32 cache markers. `--strict` fails known gaps; default exits 0 for port tracking. | **REPORT** (`check_train_loop_cache_contract_bindings.py --marker-limit 1`: all nine real loops now have text/VAE preflight binding; all nine still report host-F32 cache/activation markers) |
| `training/zimage_train_step.mojo` | ONE training step on a single Z-Image DiT-block FFN sub-path (synthetic), manual chained backward (NOT the 5-op tape) + AdamW; asserts loss finite, grads nonzero, loss decreases | **SCAFFOLD** (asserts only, no cos-gate vs torch; T5 assembly piece) |
| `training/ltx2/v2v_cache.mojo` + `training/ltx2/v2v_loss.mojo` + `models/ltx2/parity/ltx2_ic_v2v_*` (2026-07-17) | LTX2 IC-LoRA/V2V (musubi P5) units 1-2: reference-cache pairing (musubi TRAINING route — separate reference_cache_directory, same basename, standard latents_* key; CacheItem.ref_lat_path + --reference_cache_dir), ref-slice target-only masked-MSE loss + cotangent seam, two-grid v2v RoPE (`_build_v2v_coords`/`_build_v2v_rope` in ltx2_video_stack — ref PREPENDED, ref H/W ×downscale CO-LOCATION, one shared coord source with t2v), S=320 image512 comptime arm, torch oracle mirroring musubi line-by-line. Trainer wiring landed (forward_v2v + image512_v2v/video_v2v arms + first-frame Bernoulli, comptime-eliminated on base arms). | **GPU-GATED (image512 core)** (`ltx2_v2v_cache_roundtrip` byte-exact + 3 negatives; `ltx2_v2v_rope_coords_gate`; `ltx2_v2v_loss_cotangent_gate` FD 9.8e-6; oracle 30/30 both grids; `ltx2_ic_v2v_block_parity` S=320 cos 1.0 vs oracle; anchors byte-identical; live v2v smoke 4 steps @6.8s/step; S=608 video gate = next window) |
| `models/ltx2/ltx2_video_stack_capture.mojo` + `offload/ltx2_block_stream.mojo` stages (2026-07-17) | per-block CUDA-graph capture for the LTX2 trainer: `LTX2CaptureStage` (persistent standalone/offset-0 buffers — weight stage, LoRA slots, per-step staging, fixed dxA/dhB, resident grad store, graph handle) + `ltx2_video_stack_lora_backward_graph_capture` (warmup step0 / capture step1 / replay 48×). Loader gains `LTX2StandaloneStage`/`load_block_bf16_standalone` (86 standalone per-tensor buffers; residency dispatch dequants resident blocks device-only from the VRAM fp8 pool; `sync:Bool` opt-out for the replay loop). HARD RULE (MJ-1114, measured both ways): externally-refilled capture inputs must be STANDALONE buffers or offset-0 views — packed non-zero-offset sub-buffer views are layout-fragile under replay. `LTX2_V2_CAPTURE` (requires `LTX2_V2_SLAB`) default OFF: correct (3-way anchor identity, replay from step 2) but measured ~6.9s/step vs 6.3 uncaptured — residual GPU-side, unattributed (nsys records no kernel rows under graph replay). | **PROVEN-CORRECT / DEFAULT-OFF** (`ltx2_capture_wiring_parity`, `ltx2_standalone_stream_parity` incl. resident arm, `ltx2_block_capture_smoke`, `ltx2_block_capture_refill_smoke`, `ltx2_capture_external_refill_constraint`) |

---

## Data path (TRAINING) — image→latent, caption→embedding, cache, batch

The EDv2 PRECOMPUTE model: encode once to a disk cache, then the loop reads
(latent, text_embedding, mask) batches. The HEAVY encoders (~16 GB Qwen3 + VAE)
are imported only by the prepare driver, never by the loop.

| Module | Purpose / key defs | Status |
|---|---|---|
| `sampling/vae_encoder_contract.mojo` | no-CUDA SerenityTrainer VAE encode/cache readiness map. Pins image range, raw cached latent shape, prepared latent shape, SerenityTrainer `SampleVAEDistribution(mode="mean")`, scale/patch/token handoff, and fail-loud raw encode readiness for SDXL, SD3.5, Qwen, Ernie, Anima, Flux.1-dev, Flux2/Klein, Chroma, and Z-Image. Flux2/Klein is intentionally marked prepared-only for the current Mojo encoder because it emits the 128-channel patchified+BN latent instead of SerenityTrainer's raw 32-channel cached mean. | **CONTRACT** (`vae_encoder_contract_smoke`: target map, shape math, Flux2/Klein raw-cache block) |
| `models/vae/klein_encoder.mojo` (439 L) | FLUX.2/Klein VAE ENCODER (image→packed latent). `struct KleinVaeEncoder` + `.encode` → `[1,128,H/16,W/16]`. Mirrors Rust `KleinVaeEncoder::encode` (`klein_vae.rs:706-872`): conv_in → 4 down_blocks (ch_mult 1,2,4,4) → mid (resnet+attn+resnet) → GroupNorm/silu → conv_out(512→64) → quant_conv → mu=first 32 ch (deterministic, NO sampling) → patchify 2×2 (inverse of decoder `_unpatchify_packed`, `pc=((c*2+ph)*2+pw)`) → BatchNorm (eps 1e-4). INFERENCE-only (VAE frozen in LoRA training); F32 end-to-end. | **PROVEN (verified-finite)** — encoded real-Alina-image latent **std 0.962** (gate target ~0.96; a HWC→CHW scramble gives ~0.85). Smoke: `pipeline/klein_encode_smoke.mojo` |
| `training/klein_dataset.mojo` (224 L) | The cache write + read path. `write_sample(latent, text_embedding, text_mask, path, ctx)` (`:52`) → one single-file safetensors (keys `latent`[1,128,h,w] / `text_embedding`[1,512,D] / `text_mask`[1,512], byte-exact, storage dtype preserved). `struct KleinCache` (`:126`): `__init__(dir)` enumerates+sorts `.safetensors` (reproducible order, mirrors `LatentDataset::new`), `count`, `peek_key` (header-only bucket key), `load(index)`, `load_batch(indices)` (concat dim-0 same-bucket samples). `BucketKey`/`KleinSample` structs. Does NOT import the heavy encoders. | **PROVEN (byte-exact)** — write/read round-trip byte-exact (master handoff writer property). |
| `io/cap_cache.mojo` | bit-exact tensor cache `save_tensor_bin(t,path,ctx):71` / `load_tensor_bin(path,ctx):113`. The separate-process encode↔train handoff for caption embeddings (so 16 GB Qwen3 + 9B DiT never co-reside). | **PROVEN** (bit-exact round trip; INFERENCE-origin, reused by training) |
| `io/parquet/` (snappy/thrift/reader/extract) | 100%-Mojo Apache Parquet reader for the dataset shards ML datasets ship as (BYTE_ARRAY; SNAPPY; dictionary-encoded). `reader.read_byte_array_column` → `List[List[UInt8]]` (present-only, raises on nulls); `read_byte_array_column_aligned` → row-aligned values + null mask (for nullable metadata columns). `extract` = `parquet_extract` CLI with **two shard shapes (SimpleTuner-parity)**: (a) **inline-blob** (BYTE_ARRAY holds PNG/JPEG/MP4 bytes → `<out>/<NNNNN>.<ext>`+`.txt`); (b) **metadata** (`--filename-column` + caption cols, images already on disk → `<stem>.txt` sidecars in `--image-dir`). Config column mapping (`--caption-column` default caption/text/prompt + loud report; `--fallback-caption-column` on null/empty; empty-after-fallback dropped like ST), UTF-8-safe captions, resumable (skip existing unless `--overwrite`), `manifest.jsonl`. Feeds the `prepare_*_cache.py` builders. Scope: flat schemas, BYTE_ARRAY, Snappy/uncompressed (raises otherwise). File I/O via `io/ffi` (no builtin `open`). Tests: `tests/snappy_smoke.mojo` (inline, self-contained), `tests/parquet_smoke.mojo` + `parquet_oracle.py` (argv `<parquet> <col>`, FNV vs pyarrow). | **PROVEN byte-identical to pyarrow** — every column shape (captions, JSON, 35 MB JPEG dict, 485 MB PNG dict, 485 MB col in 6.8 s); inline 256-row extract byte-exact; metadata fixture (null/empty/CJK/emoji/fallback/skip/resume) byte-exact |

---

## Validation + persistence harness (TRAINING) — sample-shift gate, LoRA save, config read

| Module | Purpose / key defs | Status |
|---|---|---|
| `training/validation_sampler.mojo` (224 L) | The L2P sample-shift gate. `generate_validation[N_IMG,N_TXT,S,LH,LW](...)` (`:169`): load resident `Klein9BDiT.load_full`, OPTIONALLY merge a LoRA (`LoraSet.load.merge_into_indexed`), denoise via the proven Flux-2 sigma schedule + Euler (forward_full ×2 pos/neg → `flux2_cfg`, since the resident model has no fused `forward_full_cfg`), VAE-decode, save PNG, RETURN the RGB tensor. `pixel_l1(a,b,ctx)` (`:213`) = mean abs diff = the WITH-vs-WITHOUT-LoRA metric (0 ⇒ LoRA not applied — the bug it hunts). `load_caps` reads cached pos/neg embeddings (no encoder loaded). | **SCAFFOLD (compiles)** — reuses only proven modules (klein DiT forward, VAE decode, LoRA merge, sigma schedule); compiled, not yet lead-run on real weights. Smoke: `validation_sampler_smoke.mojo` |
| `training/lora_save.mojo` (201 L) | `save_lora_peft(adapters, path, ctx)` (`:83`) → PEFT/ai-toolkit-keyed safetensors: `<prefix>.lora_A.weight` [rank,in] + `<prefix>.lora_B.weight` [out,rank], F32, the EXACT inverse of `lora.mojo`'s load (so `LoraSet.load` / the validation sampler / ai-toolkit/diffusers open it). `load_lora_for_resume(prefixes, scale, path)` (`:146`) reads A/B back (AdamW moments zeroed — resume those from the `loop.mojo` TrainState). Deliberately writes NO `.alpha` (matches train_klein convention; caller re-supplies alpha/rank as the merge multiplier). `struct NamedLora`. Also the **F32-exact trainer-state family** (2026-07-16): `struct F32LoraState` + `save_lora_train_state_f32` (bf16 `.weight` compat copies + F32 `.lora_A/B.master` + F32 AdamW moments + `__meta__`, written HOST-DIRECT via `save_safetensors_host` — never stages through VRAM) / `load_lora_train_state_f32` (bit-exact, host-only reads) / `lora_train_state_has_f32_masters` (probe: F32-exact resume vs warm bf16 fallback). Old-era `save/load_lora_train_state` (bf16 A/B) untouched for existing callers — but bf16-only state RE-ROUNDS F32 masters on resume (MJ-1108 class); new trainers use the `_f32` family. | **PROVEN (byte-exact)** — round-trips byte-exact (F32, no BF16 truncation). Smoke: `lora_save_smoke.mojo`. F32 family: `models/ltx2/parity/ltx2_lora_state_f32_roundtrip.mojo` (bit-exact + sub-bf16 tripwire) + GPU continuation gate byte-exact (`scripts/check_ltx2_resume_continuation.py`) |
| `io/train_config_reader.mojo` | `read_train_config(json_path)` -> `TrainConfig`. Reads the model config JSON as the single source of truth for arch, paths, LoRA recipe, adapter algorithm, optimizer, cadence, offload/checkpoint flags, and `continue_last_backup`. The reader is pure Mojo: hand-rolled general JSON scalar parsing over the proven `io/json_header.mojo` cursor helpers; no Python/runtime reflection. It pulls checkpoint/vae paths, model dims, `learning_rate` -> `lr`, `lora_rank`, `lora_alpha`, `timestep_shift`, `network_algorithm`/`adapter_algo`/`algo` (`lora`, `locon`/`lycoris`, `loha`, `lokr`, `full`, `dora`, `oft`, `boft`), and nested `optimizer.{eps,weight_decay,beta1,beta2}`. Defaults live in `training/train_config.mojo`; current SerenityTrainer-style defaults are timestep shift `1.0`, AdamW eps `1e-8`, weight decay `0.01`, betas `0.9/0.999`, adapter algorithm plain LoRA. | **PROVEN (verified-finite)** — `train_config_reader_smoke.mojo`, `serenity_trainer_policy_config_smoke.mojo`, and `reader_levers_reachability_smoke.mojo`; `serenitymojo/configs/klein9b.json` currently sets `learning_rate: 4e-4`. |

---

## In-progress / mid-write (reconcile next session)

> Keep this section conservative. The Klein 9B LoRA loop is now real enough to
> sample and step, but not yet at the Rust trainer's few-seconds speed target.

| Module | Intended role (per header, may change) | Status |
|---|---|---|
| `pipeline/klein_prepare_alina.mojo` (present) | REAL prepare driver for the Alina LoRA dataset: for 4 staged 512² images + captions, `KleinVaeEncoder.encode` (assert std≈0.96) + Qwen3-8B `encode_klein` (512 tok) → `write_sample` to `output/alina_cache/`. Qwen3+VAE co-reside ONLY in this process; the train process never imports Qwen3. | **IN PROGRESS** — file exists, header complete; not yet lead-run end-to-end. |
| `training/train_klein_real.mojo` (the integrated loop) | Real Klein LoRA loop: `KleinCache` reader -> block-streamed `klein_stack_lora` fwd/bwd -> AdamW -> `save_lora_peft`. Current run target is Klein 9B at 512 (`N_IMG=1024`, latent `32 x 32`, `N_TXT=512`) with config-driven arch/paths and `learning_rate = 0.0004`. It uses `TurboPlannedLoader`, staged CFG sampling, resident RoPE tables, pure-Mojo Rust-style progress display, PEFT-style LoRA saving, and separate forward/backward `ScratchRingAllocator` arenas. This block-weight streaming is not SerenityTrainer `CPU_OFFLOADED` activation/layer parity. | **PARTIAL REAL RUN** — 50-step smoke trains without OOM and reaches the target band after early high loss (`step 18 loss 0.2721` observed). Current speed is ~`8.3s/step` before the next speed pass; target remains Rust-like `2.xs/step`. This is not full SerenityTrainer predict/backward/AdamW replay parity. |
| `training/progress_display.mojo` | Shared pure-Mojo trainer/sample UI formatter. Provides `print_trainer_progress` plus sample setup/step/saved helpers so Klein, Z-Image, Anima, and later trainers use one display contract. It prints Rust-style operator lines with step, epoch, loss, grad_norm, seconds/step, elapsed, and ETA. | **ACTIVE CONTRACT** — trainer runtime UI must call this Mojo module directly. Python wrappers are dev/replay only, not final trainer UI. |

### Klein 9B Trainer Notes - 2026-05-31

- User constraints: final trainer/runtime code stays pure Mojo; Python is allowed only for development/parity/log wrapping. Rust-side code stays pure Rust. Do not introduce Rust-from-Python or Python runtime dependencies.
- Project context: most of the Mojo stack is a pure-Mojo port/proving ground for the new Rust-stack tech developed for two goals: speed and functionality. Treat Rust as the design/performance reference, but keep the Mojo implementation native.
- Config: follow SerenityTrainer presets where possible, but do not quantize Klein for training. Use block swapping/offload instead. Current Klein 9B config sets `learning_rate = 4e-4`, timestep shift `1.0`, AdamW `eps = 1e-8`, weight decay `0.01`, betas `0.9/0.999`.
- Cache buckets: real SerenityBoard/Alina cache may contain mixed latent sizes. The trainer must filter with `KleinCache.peek_key` and only train on samples matching the compile-time shape (`c=128`, `h=32`, `w=32`, text seq `512` for the current 512 run). Observed cache state: `40 of 118` samples compatible. Without this filter, step 1 hit `reshape_owned: numel mismatch 131072 != 143360`.
- Scratch/ring allocator: yes, the trainer uses `ScratchRingAllocator`. A shared forward/backward ring exhausted during the first real backward pass; keep separate forward and backward scratch arenas unless the lifetimes are reworked.
- Sampling: staged validation sampling now mirrors the known-good inference loop: cached positive and negative caps, CFG via `flux2_cfg`, 20-step Flux2/Klein sigma schedule, live PEFT LoRA loaded through `load_klein_lora_resume`, then VAE decode. `sampling/klein_sample_cli.mojo` is a single runtime-dispatch entry for supported resolutions (currently 512/1024); do not add one CLI file per resolution. The quick fix runs positive and negative branches separately, so sample denoise is currently about `7.3s/step` at 512 and will be slower at 1024.
- Telemetry/UI: trainer runtime display is pure Mojo through `training/progress_display.mojo`, not Python. The screen line must look like `[Klein-lora] step k/total | epoch e/E | loss ... | grad_norm ... | ...s/step | elapsed ... | ETA ...`. `scripts/train_progress.py` is only an optional dev/replay helper for old raw logs; do not make final trainer UI depend on Python.
- Current speed diagnosis: noising is healthy (`~60M elems/sec`) and optimizer is negligible (`~0.07s`). The bottleneck is block streaming/staging/casting in the transformer path, especially per-block clone/cast to F32 and the current turbo prefetch overlap behavior. Fixing this should be done in shared offload code where possible so Klein, LTX, HiDream, SenseNova, etc. benefit.
- Save format: keep LoRA output plain PEFT-style safetensors via `save_lora_peft`, matching ai-toolkit/diffusers/ComfyUI expectations. Do not save a private-only adapter format.
- Next model after Klein: Z-Image. Local SerenityTrainer reference: `modules/model/ZImageModel.py::calculate_timestep_shift` computes dynamic shift from latent size with `patch_size = 2`, and `modules/modelSetup/BaseZImageSetup.py` passes either that value or fixed `config.timestep_shift`. With FlowMatch defaults, 512 training is about `1.88` and 1024 is about `3.16`; use fixed practical shifts `1.8` for 512 and `3.0` for 1024+ unless a config explicitly says otherwise.

---

## Persistence (TRAINING) — `io/`

### `io/safetensors_writer.mojo` — the WRITER (training output)
Pure-Mojo inverse of `io/safetensors.mojo` (the reader). Byte-exact format:
8-byte LE header_len, JSON header, concatenated tensor bytes in header order.
Emits the canonical compact form the Python `safetensors` lib produces (so
external tools open it). dtype/byte-size from `STDtype.name()/byte_size()`.
This is the only way trained weights / LoRA adapters get SAVED — the rest of
`io/` reads. **Status: PROVEN** (round-trips byte-exact F32+BF16; Python
`safetensors` opens it — master handoff §2).
Also exposes a **HOST-DIRECT path** (2026-07-16): `struct HostTensorDesc`
(dtype/shape/raw LE bytes) + `save_safetensors_host(names, descs, path)` —
identical file format, ZERO DeviceContext. Use it for any state that already
lives in host lists: the device path stages host→device→host before pwrite,
which for the LTX2 384-adapter F32 state was ~1.4 GB of transient VRAM on a
trainer peaking 22.5/24 GiB. **Status: PROVEN** (bit-exact round-trip via
`ltx2_lora_state_f32_roundtrip`; Python `safetensors` opens the host-written
file).

### `io/` reader stack (INFERENCE, consumed by training)
`io/safetensors.mojo` (reader), `io/sharded.mojo`, `io/mmap.mojo`,
`io/json_header.mojo`, `io/tensor_view.mojo` (`TensorView` / `from_parts`),
`io/dtype.mojo`, `io/ffi.mojo`, `io/cap_cache.mojo`. The weight-load path the
training run uses to bring real Z-Image weights onto the device. Status:
**INFERENCE** (pre-existing, parity-probed under `io/parity/`).

---

## LoRA — `lora.mojo`
**INFERENCE-only** LoRA loader (`LoraSet`). Opens a LoRA `.safetensors`, detects
its key format, computes `delta = scale·(B@A)`, and either **merges** into a
resident base weight Dict/List (`merge_into` / `merge_into_indexed`) or **applies
at-dequant** onto FP8-streamed LTX-2 blocks (re-added every dequant, never fused
to disk). **NOT a training LoRA** — `training/lora_save.mojo` is the exact inverse
(save side). `lora_probe.mojo` is its smoke.

**Accepted key formats** (`_detect_format`, `lora.mojo:192-237`):
- `FMT_KOHYA_SDXL` — `lora_unet_….lora_down/.lora_up` + `.alpha` (TE `lora_te*` skipped)
- `FMT_LTX2_DISTILLED` — `diffusion_model.…lora_A/lora_B.weight` with the cross-modal
  AV attention families (`audio_to_video_attn`/`video_to_audio_attn`/`audio_attn1`);
  matched BEFORE the generic DM branch (`lora.mojo:229-232`)
- `FMT_DIFFUSION_MODEL` — **`diffusion_model.<module>.lora_A/lora_B.weight`** (peft /
  ai-toolkit / ComfyUI); `_map_diffusion_model` strips `diffusion_model.`+`.default`,
  appends `.weight` (`lora.mojo:411-426`). **This is the exact format the Wan2.2/2.1
  (and Klein/LTX-2) trainers save — trained LoRAs load back with NO conversion.**
- `FMT_ZIMAGE_TRAINER` — split `attention.to_q/k/v` → fused `attention.qkv` row-ranges
- `FMT_KLEIN_TRAINER` — bare `<prefix>.lora_A`; EDv2 `train_klein` split→fused QKV
  remap (`_map_klein_split_qkv`, `lora.mojo:429-467`)

**Scale** (`_module_scale`, `lora.mojo:656-681`): per-module `scale=(alpha/rank)·multiplier`;
**when `.alpha` is absent, alpha defaults to module_rank ⇒ `scale=multiplier`** (so a
train_klein/Wan-style file that ships no `.alpha` needs the caller to pass `alpha/rank`
as `multiplier`). LTX-2 uses `strength·(B@A)` (no alpha/rank division).

**LTX-2 at-dequant runtime hooks** (distinct from resident `merge_into*`):
`apply_to_av_block`, `attach_ltx2_block_factors*`, `accumulate_ltx2_block_deltas*`,
`apply_to_globals*`. These are fail-closed on unmatched keys; the resident `merge_into`
path SKIPS an absent base key (does not fail-loud, `lora.mojo:751-752`).

**Wan2.2 TI2V-5B BF16-streamed and resident-FP8 hooks** (2026-07-29):
`Wan22DiTOffloaded.load_with_lora` resolves the generic
`FMT_DIFFUSION_MODEL` mappings, merges global tensors once, and applies each
block delta to that block's freshly loaded exact-BF16 weights. The adapter does
not accumulate across steps. `Wan22DiT.merge_lora_fp8_resident` dequantizes each
targeted row-scaled E4M3 matrix to BF16, applies `scale·(B@A)` once, and
requantizes only the in-memory resident tensor. Neither path rewrites the base
checkpoint/cache. The Rust `/v1/video` preflight checks every admitted A/B pair
against the 5B 3072/14336 dimensions before GPU work, so a Wan 14B adapter
cannot silently run on the 5B base. The 300-mapping BF16 product smoke completes
one denoise step and is required by the Wan product gate.

**KJNodes `LTX2LoraLoaderAdvanced` — per-stream strengths** (commit `4706f99`):
`LoraStreamMults` POD (`lora.mojo:101-142`) = five multipliers
`video / video_to_audio / audio / audio_to_video / other`, matched by key substring
with KJ most-specific precedence (`mult_for_key`); a `0.0` slider drops the module
after fail-closed validation. `_streamed` overloads on all four LTX-2 apply seams
(`lora.mojo:865-1112`); non-streamed methods delegate via `LoraStreamMults.identity()`.
Exposed via env `LTX2_TRAINED_LORA_STREAMS_{i}` (five comma-joined floats), the
`ltx2_request_cli` per-row `video/video_to_audio/audio/audio_to_video/other` fields
(validated `[0,1]`; default rows stay byte-identical), and the Rust serve
`LTX2LoraLoaderAdvanced` node (`serve/workflow_graph.mojo` + `graph/execute.rs`).
The request product additionally binds special adapters through stable IDs in
`configs/ltx2_feature_adapters.json`. Foley/V2A is admitted only with
`video=0`, `video_to_audio=1`, `audio=1`, `audio_to_video=0`, `other=1`,
source strength `1.0`, and generated audio. Cinemagraph is admitted only as an
I2V overlay with its exact trigger and explicit bounded weight. IC-LoRA
artifacts are not ordinary overlays and remain rejected until their
reference-token feature runner is admitted.

**Product inference callers today**: SDXL, Ideogram 4, SD 3/3.5, Qwen Image,
Anima, Flux.1, Chroma, Klein/Flux.2, Krea 2, Z-Image, and LTX-2. The Rust model
registry discovers LoRAs recursively and case-insensitively, resolves the
selected registry identity to the exact file, preserves the user's multiplier,
and publishes each family's real composition limit. Unknown/custom adapters
are allowed to reach the selected family worker; the Mojo target loader fails
loudly only when its tensors do not match that model. Cross-family adapters
with positive architecture metadata are rejected before GPU load.

The model-specific product overlays preserve the creator key topology rather
than pretending one universal mapping: `models/dit/sdxl_unet.mojo` handles
kohya/Diffusers SDXL names; `models/dit/ideogram4_resident.mojo` accepts
A/B and down/up pairs; Qwen uses canonical split/fused projection attachment;
`models/sd35/sd3_lokr_overlay.mojo` implements the SwarmUI/Comfy efficient
Kronecker carrier; `models/flux/flux_lora_overlay.mojo` and
`models/chroma/chroma_lora_overlay.mojo` split fused creator projections into
their runtime slots. NOTE: no Wan *inference* pipeline calls `LoraSet` yet —
Wan trained LoRAs are loadable (`FMT_DIFFUSION_MODEL`) but not yet wired into a
Wan inference run. Status: **INFERENCE**.

## SwarmUI-compatible sampling schedules

`sampling/swarmui_schedules.mojo` ports the creator's Flux model-sampling
contract at shift 1.15. It implements `normal`, `karras`, `exponential`,
`simple`, `ddim_uniform`, `sgm_uniform`, `beta`, `linear_quadratic`, and
`kl_optimal`, including the creator's DDIM interval behavior when the requested
step count is not a divisor of 999. `sampling/swarmui_schedules_smoke.mojo`
pins exact scalar answers and the requested-versus-executed step contract.

Flux and Chroma expose genuine Euler/flow-match Euler and DPM++ 2M
data-prediction updates across those nine schedules. The server publishes the
per-family executable set through one capability document consumed by both
Canvas and Generate. The separate public discovery catalog mirrors SwarmUI's
44 sampler and 16 scheduler IDs for workflow compatibility; catalog presence
does not claim that all 44 algorithms are implemented for every model.

---

## Model forwards (INFERENCE) — the fixed kernels training composes over

The `models/` subtree is the inference forward library. Training reuses these
as fixed forward kernels and supplies the backward via `ops/*_backward.mojo`.
Distinguish: these are NOT training modules; only their forward shapes/configs
matter to the training path.

### `models/dit/` — DiT/transformer forwards
`zimage_dit.mojo` is **the** training target — `NextDiTConfig.zimage()` at
`zimage_dit.mojo:96-98` is `dim=3840, n_heads=30, head_dim=128` (the H=30 that
the sdpa-bwd toy gate missed). Also: `klein_dit`, `flux1_dit`, `chroma_dit`,
`anima_dit`, `ernie_image`, `qwenimage_dit`, `sd3_mmdit`, `sdxl_unet`,
`hidream_o1`, `nucleus_dit`/`nucleus_moe`, `ltx2_dit`, `seedvr2_dit`,
`seedvr2_sampler`, `sensenova_u1`,
`zimage_l2p_dit` (+ `_contract.mojo` shape contracts and `_probe.mojo` smokes
per model). Status: **INFERENCE**.

### `models/text_encoder/` — conditioning
`t5_encoder.mojo` (the real Z-Image T5 run depends on this), `clip_encoder`,
`qwen3_encoder`, `qwen25vl_encoder`, and `gemma3_ltx_streamed` (the
layer-streamed Gemma-3-12B FP8 LTX2 conditioner).
`serenity_trainer_conditioning_contract.mojo`
is the no-CUDA SerenityTrainer conditioning/cache-readiness map for the target image
models: tokenizer lengths, prompt/chat mode, mask/crop mode, hidden/pooled dims,
cache-field names, runtime text/img ids, and dtype-boundary policy. Parity under
`models/text_encoder/parity/`. Status: **INFERENCE / CONTRACT**.

### `models/vae/` — encode/decode
`zimage_decoder`, `klein_decoder`, `ldm_decoder`, `conv3d`, `wan22_decoder`,
`qwenimage_decoder`, `ltx2_*`, `seedvr2_vae`, `vae_ops`, `upsample`,
`decoder2d`. These are
the forwards whose `conv2d`/`pool`/`upsample` backward partners live in
`ops/conv2d_backward.mojo` + `ops/pool_backward.mojo`. Status: **INFERENCE**.

### Other inference subtrees
`models/pid/`, `models/lens/`, `models/lance/`, `models/upsampler/`,
`models/vocoder/`, `models/realesrgan/` (RRDBNet x4plus and compact SRVGG
x4v3 forwards); plus `pipeline/` (end-to-end inference smokes),
`sampling/` (schedulers/flow-match samplers), `tokenizer/`, `offload/`
(block-streaming loaders for 24 GB fit), `runtime/`, `registry/`,
`components/`, `image/`. All **INFERENCE** — the training run borrows the
offloaders (`offload/`) and samplers' schedule math but does not backprop
through them.

`sampling/ltx2_sampling.mojo` exposes the creator-fast LTX2 public surface
through `serenitymojo.sampling`: `LTX2Scheduler`, distilled sigma tables, and
`ltx2_creator_noiser_from_noise`, plus the Sulphur creator stage-one shifted
schedule, stage-two LCM sigmas, LCM update, and Euler ancestral CFG++ step.
BF16 `LTX2Scheduler.step` uses the shared PyTorch-eager BF16 API so
noiser/scheduler phase parity is not lost in fused tensor algebra.

`sampling/ltx2_request_cli.mojo` is the pure-Mojo product adapter for canonical
`serenity.genparams.v1` video requests. It preserves the exact authored JSON,
checks conditioning sidecar prompt identity, resolves an arbitrary LoRA list
under the shared model root, and carries the request's `model_quant`. LTX uses
one `output/bin/ltx2_serenity_runtime` executable. The old per-geometry binary
naming, lookup fallback, and 31-binary build task were removed; profile metadata
may inform the UI but cannot select an alternate executable. The server fails
closed while the runtime-shape Mojo path is absent or stale.

`configs/ltx2_checkpoint_workflows.json` separately binds checkpoint identities
to publisher-authored inference recipes without hard-coding those recipes into
the UI. The initial Sulphur profile follows its published workflow: eight
stage-one Euler ancestral CFG++ evaluations with the LTX shifted schedule,
three stage-two LCM evaluations with fixed sigmas/seed, and its CondSafe
distillation adapter at `0.7` then `0.5`. An empty workflow ID still
auto-detects the registered creator profile. The published Qwen3.5 prompt
enhancer files and raw-text/optional-image/no-system-prompt contract are
advertised, but execution stays fail-closed until both local artifacts and a
real llama.cpp route exist.

Ordinary I2V uses the LTX Desktop preprocessing contract: Lanczos fit/fill at
the authored conditioning size, then
`scripts/ltx2_creator_image_preprocess.py` performs the creator's one-frame
PyAV/libx264 CRF-33 `veryfast` yuv420p round trip. The product check matched
the creator output pixel-for-pixel (MAE `0.0`, maximum difference `0`).
Frame-zero conditioning uses source strength 1.0 and separate
half/full-resolution VAE encodes. An optional final image uses Lightricks'
clean keyframe-token contract at the final FPS-normalized frame coordinate,
including a per-guide denoise mask and independent encoding in both stages;
the guide tokens are removed before VAE decode. I2V and ordinary V2V use
clean-latent clamping plus exact per-token model timesteps. Painted V2V masks
lower to the same latent grid with white/edit and black/preserve semantics at
both stages. The deterministic
`sampling/parity/ltx2_conditioning_mask_parity.mojo` gate checks I2V, uniform
V2V, painted V2V, and the T2V broadcast control. Audio is never inferred from
an input's mere presence: the canonical request carries an explicit generated,
source-preserving, or no-audio policy.

Retake and Extend instead use the LTX Desktop one-stage full-resolution
pipeline with the complete distilled BF16 checkpoint and no two-stage support
LoRA or spatial upscaler. `models/vae/ltx2_audio_processor.mojo` implements the
creator waveform-to-log-mel frontend, while `models/vae/ltx2_audio_vae.mojo`
now exposes source-audio encode as well as decode. Source video VAE encode uses
the creator's 256/64-pixel spatial and 24/16-frame temporal tiles. Retake uses a
binary temporal mask and freezes source audio in replace-video mode. Extend
zero-pads video/audio latents at the selected edge, regenerates audio, and
includes the creator's 0.5-second seam. The Rust boundary stages creator-exact
PyAV audio samples with `scripts/ltx2_decode_source_audio.py` before entering
the Mojo AudioProcessor/AudioVAE path.

`configs/ltx2_feature_adapters.json` is the feature-adapter product registry.
The server embeds the resolved document in the immutable request and the Mojo
CLI revalidates the input kind, trigger, weight range, mask restrictions, and
per-stream strengths. Cinemagraph and Foley/V2A have real product evidence.
The installed reference-conditioned IC-LoRAs are published as runtime-pending,
not silently treated as trained LoRAs.

When manual conditioning sidecars are blank,
`pipeline/ltx2_encode_prompt.mojo` tokenizes both prompts and runs the streamed
`models/text_encoder/gemma3_ltx_streamed.mojo` encoder once per layer pair. It
writes the six prompt-matched pre-connector tensors and reports tokenization,
all 48 Gemma layers, projections, and save progress to the server. The server
caches those tensors by prompt pair and conditioner digest. Exact tokenizer
IDs passed and real context cosine measured 0.99923-0.99973; the optimized
reference prompt completed in 17.19 seconds without Python in the product path.
After each synchronized Gemma layer and aggregate projection upload, the
conditioner releases clean mmap-backed checkpoint pages to the OS. A real
512x768, 121-frame Canvas V2V product run measured a 17.8GB cgroup peak with no
swap; the prior unreleased-page path reached 54.9GB and triggered desktop OOM.

The admitted `int4` route keeps the 48-block SVD-int4 base slab resident and
streams factorized LoRA A/B matrices per block without allocating dense
deltas. Atomic `status.json` and `result.json` publish the executed
geometry/schedule, timings, frame count/duration, dtype contract, and sampled
peak VRAM. Creator fast-distilled requests use Euler plus `ltx2_distilled`
with eight stage-1 evaluations; bounded dev requests use `res2s` plus `ltx2`
for 1-20 steps.

Full LTX 2.3 fine-tunes are registered in the same JSON under `checkpoints`.
Each entry names the UI/API `id`, model-root-relative `path`, accepted aliases,
guidance and quant modes, and whether `support_lora` is `official` (dev base
needs the support LoRA) or `baked` (a distilled full checkpoint must not stack
it again). The server publishes only installed entries to Generate, resolves
only registered paths below `SERENITY_MODEL_ROOT`, and passes the resolved
checkpoint/support contract to each fresh Mojo process. New entries remain
experimental until their own sampled artifact and parity evidence is accepted.

Long and high-resolution decode uses the Desktop tiled VAE contract and streams
completed PNG chunks instead of retaining the whole movie tensor. The
960x544/481f product gate produced a coherent 20.041667-second H.264 video and
an exact 48 kHz stereo A/V mux, but its 19,621 MiB peak is not accepted on a
16 GB RTX 5080. `sampling/realesrgan_x4_cli.mojo` exposes the separately
admitted 2x/4x post-upscale request: RRDB x4plus is functional but
experimental-slow, while SRVGG x4v3 and the imported SeedVR2 source remain
disabled when their local weights or complete user-video route are absent.
Build the single runtime-geometry request runner with
`pixi run build-ltx2-request`.

The 2026-07-27 creator-parity product gate used the same checkpoint, source,
prompt, and locked seed in LTX Desktop and Serenity. Paired Retake measured
0.962804 mean SSIM over 121 frames and 0.987185 over the protected region.
Paired three-second Extend measured 0.986503 over the protected 108 frames;
the seam and generated extension followed different numerical trajectories but
both remained visually clean. Result manifests intentionally keep sampler and
speed parity unaccepted until those independent gates pass.

The 2026-07-28 real Sulphur BF16 first+last-frame gate (`video-0016`) produced
704x1280, 121 frames at 24 FPS in 192.20 seconds with a 14,583 MiB sampled peak.
Frames 0/30/60/90/120 retained coherent eyes, lips, hair, facial identity,
wires, and metal-panel geometry. Both staged keyframes were pixel-identical to
the creator PyAV preprocessing output.

---

## Forward op library (INFERENCE) — `ops/*.mojo` (non-`_backward`)

The forward kernels the backward partners pair with: `ops/attention.mojo`
(math-mode SDPA fwd), `ops/linear.mojo`, `ops/norm.mojo`, `ops/activations.mojo`
(`silu`, `swiglu`), `ops/reduce.mojo`, `ops/rope.mojo`, `ops/conv.mojo`
(SDK conv2d fwd wrapper), `ops/conv1d.mojo`, `ops/elementwise.mojo`,
`ops/tensor_algebra.mojo` (transpose/concat/slice/add/mul_scalar/zeros_device/full_device/scalar_f32_device),
`ops/tensor_algebra_scratch.mojo` (opt-in scratch-backed shape helpers),
`ops/softmax.mojo`, `ops/cast.mojo`, `ops/embeddings.mojo`, `ops/layout.mojo`,
`ops/moe.mojo`, `ops/fp8.mojo`, `ops/mxfp4.mojo`, `ops/snake.mojo`,
`ops/pixelshuffle.mojo`, `ops/random.mojo`, `ops/unary.mojo`,
`ops/activation1d.mojo`, `ops/torch_bf16.mojo`. Status: **INFERENCE** (each has
a `*_smoke.mojo` or an end-to-end parity gate).

`ops/torch_bf16.mojo` is the cross-model PyTorch-eager BF16 numeric API:
`torch_f32_to_bf16_rne`, `torch_bf16_eager_blend_with_f32_mask`,
`torch_bf16_eager_velocity_from_x0`, and `torch_bf16_eager_add_scaled`.
Import these through `from serenitymojo.ops import ...` for noiser/scheduler
handoffs that must match PyTorch eager F32 temporaries before a BF16 storage
boundary. The LTX2 creator phase gate also records raw transformer velocity at
the first/last stage steps; noiser and Euler phase math are exact, while raw
velocity vs x0-derived velocity is gated with the documented BF16 x0 round-trip
tolerance. See `docs/MOJO_DIFFUSION_NUMERIC_API.md`.

---

## THE sdpa-bwd H=30 SCARE — RESOLVED as degenerate test data (2026-05-30)

Earlier in the session `ops/parity/sdpa_bwd_realseq_parity.mojo` reported
`sdpa_backward` producing **numerically-zero d_q/d_k at H=30** (Z-Image's real
head count) while d_v passed (cos 0.99999999), flagged as a silent half-learning
blocker (bug doc `BUG_sdpa_backward_H30_dq_dk_zero.md`, master handoff §1).

**Source now records this was NOT a kernel bug.** `models/zimage/train.mojo:6-14`:
the H=30 zero was a DEGENERATE-TEST-DATA artifact — the old gate's V-fill aliased
mod 9 at H·Dh=3840, making V constant across the sequence, so `grad_scores=0` was
the CORRECT answer and torch agreed. The non-degenerate gate
`ops/parity/sdpa_bwd_nondegen_parity.mojo` (+`_oracle.py`) measures cos≥0.999 at
H=30. **Z-Image is NOT blocked.** One precision watch remains: at S=2304, d_k
cos 0.9975 (F32 accumulation order, not corruption). See
`project_mojo_sdpa_h30_blocker_false_2026-05-30`.

> Reconcile note: `HANDOFF_2026-05-30_MOJO_TRAINING_PORT_MASTER.md` §1 still
> describes the H=30 zero as an OPEN blocker. The newer `zimage/train.mojo`
> source supersedes it. If you trust only the master handoff you'll chase a
> non-bug — the in-tree `sdpa_bwd_nondegen_parity.mojo` is the authority.

---

## Where to start

- **Add a backward op**: write the kernel in `ops/<x>_backward.mojo` mirroring
  the nearest sibling's NHWC/F32/uint8-bitcast scaffolding, add a gate
  `ops/parity/<x>_bwd_parity.mojo` (cos ≥ 0.999 vs a torch host reference).
- **Wire it into the tape**: add an `OP_<X>` `comptime` const, a `TapeEntry`
  shape, and a `backward()` dispatch arm in `autograd.mojo`, then a
  `autograd_<x>_smoke.mojo`. **After wiring, `grep -c "elif ek == OP_X"`** —
  silent Edit failures dropped arms twice (handoff §4).
- **Assemble a model step**: compose `training/dit_block.mojo` × N +
  `training/checkpoint_block.mojo` (24 GB fit) + `training/loop.mojo`
  (F32-master/BF16) + `training/schedule.mojo` (v-target) +
  `io/safetensors_writer.mojo` (save). The Z-Image op→backward map is
  `docs/maps/T5_ZIMAGE_TRAINING_MAP.md`.
- **Find the convention/idiom that's biting you**: `docs/MOJO_CONVENTIONS.md`.
- **Debug a dead/wrong gradient**: `docs/MOJO_DIAGNOSTICS.md`.
## Trainer Runtime Contracts

- `serenitymojo/training/progress_display.mojo` is the shared pure-Mojo screen
  display for trainer and sampler progress.
- `serenitymojo/training/sample_prompt_config.mojo` reads shared
  `serenity.sample_prompts.v1` JSON files. Trainers read prompt text, sample
  parameters, and precomputed cap-cache paths from this file.
- `serenitymojo/docs/TRAINER_MANDATORY_RUNTIME_CONTRACT.md` is binding for all
  model trainers: shared progress display, shared prompt JSON, image samples
  defaulting to 1024x1024, PEFT plus optimizer-state resume checkpoints, and the
  sample/train/save/resume smoke.
- `serenitymojo/training/serenityboard.mojo` writes SerenityBoard `board.db`
  directly from Mojo via SQLite FFI. It records train scalars, prompt text,
  save/resume events, and PNG artifacts.
- `serenitymojo/docs/TRAINER_SAMPLE_PROMPTS_AND_BOARD_2026-05-31.md` documents
  the production sampling cadence and board tags.

## 2026-06-01 — flame-core parity port (new standalone modules)

New standalone, parity-gated modules porting missing flame-core/EDv2 tools. **None are
wired into any trainer/sampler/config** — each matches its Rust reference at parity.
Full inventory + gates + scope caveats: `docs/FLAMECORE_PARITY_PORTED_2026-06-01.md`.

- Levers (modules, not wired): `training/{lr_schedule,loss_weight,timestep_bias,caption_dropout,noise_modifiers,grad_accum,ema_schedule}.mojo`
- Optimizers: `training/opt_{lion,stableadamw,adafactor,prodigy,schedulefree}.mojo`
- LyCORIS adapters: `training/{loha,dora,lokr,oft,boft,locon_conv,tucker_conv,full}_adapter.mojo` (+ `*_save.mojo`). Primitive gates cover the family; production trainers accept LoCon only on linear LoRA-compatible targets, Klein also carries the proven LoKr end-to-end path, and unsupported variants fail before checkpoint/GPU-heavy work.
- LyCORIS carrier-dispatch (additive families, 2026-06-27 — `docs/LYCORIS_CARRIER_DISPATCH_2026-06-27.md`): `training/lokr_stack.mojo` (LoKr Kronecker→`(a,b)` carrier) + `training/loha_stack.mojo` (LoHa Hadamard→`r_eff=r²` carrier + klein orchestration). Both materialize a LyCORIS master into the plain-LoRA `(a,b)` the model stack already trains, chain the carrier `d_a/d_b` back to the master factor grads, host-AdamW the masters, save in lycoris keys — NO stack/kernel change. Proven e2e on real Klein-9B: `models/klein/parity/klein_stack_{lokr,loha}_real_smoke.mojo`; carrier algebra: `training/tests/{lokr_st_parity,loha_carrier_parity}.mojo`. OFT/BOFT/DoRA are NOT carrier-compatible (multiplicative/renorm) — need new stack hooks. `training/dora_adapter.mojo` has a `wd_on_out` axis flag (default True=lycoris per-output; False=SerenityTrainer per-input, gated `tests/dora_serenity_trainer_parity.mojo` cos=0.99999998). ai-toolkit ships lycoris 1.8.3 (verified == Mojo formula, MJ-1020).
- Diagnostics: `training/grad_coverage.mojo`
- Samplers: `sampling/{dpmpp_2m,unipc,inpaint,img2img_refpack}.mojo`; encoder `vae/vae_encode_general.mojo` (weight-gated)
- Perf (new sibling kernels): `ops/vec_{permute0213,transpose,rms_norm,swiglu,modulate}.mojo`, `training/{fused_adamw_multitensor,on_device_global_norm}.mojo`
- Infra: `offload/transfer_benchmark.mojo`, `io/disk_check.mojo`

NOT ported (in-place-edit-only / NO_MOJO_PATH / user-excluded) + the parked trainer-file
edits: see sections C and D of the parity-ported doc.

## 2026-07-13 — ACE-Step-1.5 LoRA backward (autograd_v2 graph engine)

- `autograd_v2/acestep_block_graph.mojo` (NEW) — the ACE-Step DiT LoRA backward.
  `acestep_block_lora_graph_backward[S,L,NH]` records ONE layer's forward op-for-op
  (flat [S,D] token space, zimage discipline) through the `record_*` wrappers and
  runs `engine.execute` → d_x + 8 LoRA A/B device grads (self+cross q/k/v/o).
  `acestep_stack_lora_graph_backward[SP,L,NH,LAYERS]` = forward conductor (saves each
  block's input) → MSE-loss grad → proj_out ConvTranspose1d bwd → final-AdaLN bwd →
  32× block bwd → 512 grads. Gated: `models/dit/parity/acestep_{block0,full}_bwd_gate.mojo`
  (16/16 + 512/512 PASS; block0_d_out 0.982 vs oracle).
- **New engine op kinds** (`autograd_v2/node.mojo` OPK table + `ops_record.mojo`
  record wrapper + `engine.mojo` apply arm — additive, existing kinds untouched):
  - `OPK_ROPE_HALFSPLIT` — `record_rope_halfsplit(g,x,cos,sin,ctx)`; fwd
    `rope_halfsplit`, bwd `rope_backward(...,interleaved=False)` (Qwen3 rotate_half;
    the interleaved `OPK_ROPE` silently aliases the wrong angle — do NOT reuse it).
  - `OPK_LINEAR_DX` — `record_linear_dx(g,x,w,M,in_f,out_f,ctx)`; frozen linear,
    dx-only via `linear_backward_dx` (no d_w). For the acestep MLP gate/up/down
    (no LoRA) — routes d_x so upstream cross/self LoRA grads are complete.
  - `sdpa_backward_dispatch` gained acestep buckets (1,64,32,128)/(1,64,16,128).
- dQ/dK note: self-attn q_proj + k_proj LoRA grads are softmax-jacobian-derived and
  implementation-variant (measured: torch F.sdpa vs manual-softmax disagree 0.89–0.99
  on those tensors, loss identical). Our math sdpa_backward lands in that band —
  value-tolerance, not a bug. See `serenitymojo/MAP.md` 2026-07-13 entry.
- `models/acestep/acestep_train_step.mojo` (NEW) — the training step wrapping the
  full-stack backward: `acestep_sample_t` (logit-normal timestep, r=t), `acestep_flow_noise`
  (xt=t·x1+(1−t)·x0, flow=x1−x0), `acestep_apply_cfg_dropout` (bs=1 coin flip),
  `acestep_loss_mse`, `acestep_train_step[SP,L,NH,LAYERS]` → AcestepStackLoraGrads (512
  grads + loss). Gate `models/acestep/parity/acestep_train_step_gate.mojo` 4/4 PASS. FIXED
  recipe = plain MSE (not masked); CFG dropout untested (null_cond not in decoder-only load).
- `models/acestep/train_acestep.mojo` (NEW) — the LoRA trainer DRIVER: AceStepTrainConfig +
  _build_adapters (256 train_step.LoraAdapter) + acestep_train loop (GRAD ACCUM: SUM `grad_accum`
  FULLY DEVICE-RESIDENT AdamW: params+moments on device; `full` LoRA = dev_p sub-buffer VIEWS
  (`_acestep_devp_views` → no re-upload); `grad_accum` micro-grads accumulated ON DEVICE in F32
  (no to_host) → `copy_device_grad_pair` → `fused_lora_adamw_plain_step_resident_preloaded_grads`
  (norm+clip on device; sync-to-host only at save) → `save_lora_peft`. Streams samples by path;
  `$ACESTEP_STEPS`/`$ACESTEP_ACCUM`/`$ACESTEP_CACHE`/`$ACESTEP_CFG_PCT` (CLI) OR positional argv (serenity-trainer UI config-runner `acestep` shape: checkpoint/cache/out/run/steps/accum/lr/rank/alpha/save_every/cfg_pct/seed) overrides. GATE: accum=4 Σ|B| 26.383848
  (byte-identical to host-list); accum=1 142.3154 (within 0.012% — on-device F32 vs F64 host norm).
- `models/acestep/acestep_cache_reader.mojo` + `scripts/acestep_pt_to_cache.py` (NEW, #10) — the
  train-cache path. Converter: upstream `.pt` (5 keys) → per-sample BF16 safetensors + manifest.txt
  (--from-oracle builds one from the parity dump). Reader: `AcestepTrainSample` + `acestep_load_sample`
  (5 keys by name; works on cache samples AND the raw dump) + `acestep_read_manifest`. `$ACESTEP_CACHE`
  selects the cache dir else the oracle dump; cache run is BYTE-IDENTICAL to dump-direct. **★ ACE-Step
  trainer vertical COMPLETE: fwd+bwd+train-step+driver+cache, all gated — trains a LoRA end-to-end.**

## MageFlow image generation and editing (2026-07-22)

- `models/dit/mageflow_dit.mojo` implements the 12-block MageFlow DiT using the
  shared Qwen-Image block math plus image-only multi-axis RoPE. The component
  gates report block cosine 0.99999 and full velocity cosine 0.99898.
- `models/text_encoder/mageflow_qwen3vl.mojo` reuses the Qwen3-VL loader and
  exposes both the T2I post-norm context path (`drop_idx=34`) and the edit
  vision/deepstack fusion path (`drop_idx=64`).
- `models/vae/mageflow_vae.mojo` provides one-step MageVAE encode and decode,
  including non-square aspect-preserving edit geometry. The recorded full
  encode mean cosine is 0.99999976; square decode remains cosine 1.0.
- `pipeline/mageflow_pipeline.mojo` is the sequentially offloaded four-step
  Turbo T2I capstone. Its recorded fixed-fixture final-latent cosine is 0.9942
  and the image was visually matched to the oracle.
- `pipeline/mageflow_edit_pipeline.mojo` VAE-encodes a clean reference,
  concatenates its latent tokens after the pure-noise target, steps only the
  target tokens, and preserves source aspect ratio. The recorded edit gate is
  0.99979 for the reference latent and 0.99934 for the final target latent,
  with a visually matching edited image.
- These are direct pure-Mojo pipeline entrypoints and parity gates. They are not
  yet registered as Serenity server workers or Canvas engines; the UI must not
  route to them until capability, request, and worker lifecycle wiring exists.

## Serenity Canvas editing runtime (2026-07-23)

- `serve/klein_runtime_backend.mojo` is the resident pure-Mojo product runtime
  for FLUX.2 Klein 9B and 4B. In addition to text-to-image it now accepts one
  native `ReferenceLatent` source at 512x512 or 1024x1024, VAE-encodes that
  source in the worker, and dispatches the matching compiled 4B/9B edit shape.
  Ordinary img2img remains rejected, and the older two-reference path remains
  bounded to 512x512.
- `serve/image_io.mojo` keeps LanPaint's authored final-blend mask distinct from
  its optional expanded sampler-context mask. This lets structural edits expose
  surrounding geometry during denoising without committing generated pixels
  outside the user's painted region. The same module owns crop, hard-mask,
  preserve-mask, and source-preserving blend primitives.
- `serve/krea2_backend.mojo` forwards the explicit image-pixel context expansion
  to the latent preserve-mask loader. `models/krea2/krea2_infer.mojo` uses
  upstream LanPaint's FLUX `cfg_BIG=1.0` contract instead of changing that
  branch based on the UI prompt-mode label.
- Rust remains the capability/request control plane; image editing, reference
  VAE encode, LanPaint sampling, and pixel output remain in the Mojo workers.

## MiniMax-H3 audio-video inference runtime (2026-08-05)

- `models/dit/minimax_h3_runtime_cache.mojo` owns the versioned conditioning,
  modulation, and resident-weight sidecars used by the product runners. Cache
  reload is dtype/shape validated and byte preserving; CPU work is limited to
  file I/O and staging, while model execution remains on the GPU.
- `models/dit/minimax_h3_fp8_resident.mojo` is the historically named,
  scheme-selectable resident store. The admitted product schemes are group-wise
  INT8 weights for the Quality arm and direct W8A8 weights for the Fast arm.
  `models/dit/minimax_h3_int8_linear.mojo` plus the model-scoped cshim implement
  GPU activation quantization, INT8 GEMM, and BF16 output scaling.
- `models/text_encoder/minimax_h3_qwen3vl_int8.mojo` and
  `minimax_h3_qwen3vl_int8_cache_cli.mojo` provide the shared per-row INT8
  Qwen3-VL text-encoder cache and GPU forward. The BF16 conditioning output is
  shared by BF16, INT8 Quality, and INT8 Fast DiT runners.
- `ops/sage_attention_int8.mojo` is an opt-in Sage-style INT8-QK attention
  backend with F32 accumulation and explicit tail masking. It is experimental
  and off by default: cU-DNN remains the accepted product backend, and Fast
  rejects Sage because the measured end-to-end path was slower and below the
  audio parity bar.
- `pipeline/minimax_h3_t2va.mojo` is the AOT-specialized product CLI. The
  tracked registry selects five geometries at 24 FPS/20 steps inside one
  `minimax_h3_serenity_runtime` executable: 512x320x175, 832x480x73,
  960x544x56, plus 15.08-second/362-frame profiles at 512x320 and 832x480.
  The 512x320 long profile admits BF16, INT8 Quality, and INT8 Fast; both INT8
  modes keep zero blocks resident and stream their quantized cache one block at
  a time so the longer attention activation retains VRAM headroom. The
  832x480 long profile is BF16-only. Shorter profiles retain their measured
  resident prefixes. Denoise and fresh-process GPU VAE decode/NVENC mux remain
  phase isolated.
- `serenity-server/crates/server/src/video.rs` and the Canvas Generate/Workflow
  modules resolve exact profile geometry and precision before acquiring the GPU
  lease. Unsupported geometry, missing runners/caches, stale machine gates, and
  invalid attention combinations fail queue admission closed. Canvas treats
  duration as the protected selector: it opens T2VA on the longest admitted
  profile, labels each resolution with its available seconds, and disables a
  resolution that would silently shorten the selected duration. Choosing a
  different supported duration explicitly re-resolves the compatible
  resolution and exact frame count.

**Status: INFERENCE / PRODUCT-GATED.** The three modes are deliberately
separate product choices: INT8 Fast is perceptually accepted, INT8 Quality and
streamed BF16 preserve the quality choices, and no claim is made that their
full denoise trajectories are numerically identical. The 512x320x362 Fast gate
completed all 19 evaluations with finite video/audio latents, decoded 362
frames plus 15.075 seconds of stereo audio, and peaked at 13,884 MiB during
denoise and 11,398 MiB during decode. The long Quality/BF16 arms passed finite
one-evaluation gates; 832x480x362 BF16 also has a prior complete 15.08-second
artifact and a current-run finite gate.

## serve/parity — worker runtime gates (Phase-5 worker-fix campaign)

Gates for the process-isolated **worker** runtime (`serve/`): they exercise the
EXACT code path a live worker runs, not a re-implementation. Two kinds:

- **Encode-dump gates** (GPU + real models): each `main()` runs the worker's text
  encode for one fixed prompt (`"a photorealistic red fox sitting in autumn
  leaves"`) and DUMPS the produced conditioning + the token ids into
  `output/checks/phase4a/<model>_worker_*.safetensors`. A python reference under
  `serve/parity/ref/` then loads the dump and compares on BYTE-IDENTICAL ids. The
  Mojo binary produces the dump; the python ref renders the PASS/FAIL verdict
  (recorded in `output/checks/phase4a/<model>_verdict.txt`). MJ-1052/MJ-1061.

  | gate | worker path | reference | verdict |
  |------|-------------|-----------|---------|
  | `serve/parity/sdxl_worker_encode_gate.mojo` | `sdxl_backend::_encode_one` → `ClipEncoder.encode_sdxl(penultimate_context=True)` | `ref/ref_sdxl_clip.py` (HF CLIPTextModel) | PASS post-MJ-1061 (min real-row cos **0.999797**; was 0.727/0.680 FAIL when it returned post-final-LN, MJ-1052) |
  | `serve/parity/sd3_worker_encode_gate.mojo` | `sd3_backend::_assemble_one` (CLIP-L∥CLIP-G penult ∥ T5-XXL) | `ref/ref_sd3.py`, `ref/ref_sd3_t5_noise.py` | PASS (CLIP-L 0.999997, CLIP-G 0.999792, T5 0.999488 vs **fp32** ref) |
  | `serve/parity/klein_worker_encode_gate.mojo` | `klein_runtime_backend::_encode_text_pair` (Qwen3-8B, layers 8/17/26) | `ref/ref_klein.py` (HF Qwen3-8B) | PASS (min real-row cos 0.997114) |
  | `serve/parity/zimage_worker_encode_gate.mojo` | `pipeline/zimage_generate::_encode_text_fixed` (Qwen3-4B layer-34 penult) | `ref/ref_zimage.py` (HF Qwen3-4B) | PASS (min real-row cos 0.999752) |

  Pad rows diverge benignly in klein/zimage (worker uses the Rust key-padding mask
  at `real_len`; HF runs mask-less) — real-row cos is the gate, overall cos is not.

- **`serve/parity/jobparams_roundtrip_smoke.mojo`** (pure-CPU, self-verifying —
  prints `ALL PASS` / raises on mismatch): round-trips a populated `JobParams`
  through the worker wire (`ipc_codec.encode_start` → `json.loads` →
  `decode_start`, the exact parent→child path), asserts the `JobParams` global
  defaults (steps 20 / seed 0 / cfg 4.5), and direct-imports + asserts the frozen
  per-backend default constants (`SD3_DEFAULT_STEPS=28`, `SD3_DEFAULT_SEED=42`,
  `QWENIMAGE_DEFAULT_CFG=4.0`) plus replicates their guard branches
  (steps≤0→28, cfg≤0→4.0, sensenova steps<1→raise) as pure logic. It does NOT
  execute each backend's `start()`/admission (needs GPU + models), and flux's
  cfg≤0→3.5 lives in `flux_backend.start` (not a constant), so that is documented
  not asserted.

## Training-free editing — the FlowEdit family (2026-07-14)

Inference-only instruction editing (FlowEdit ODE, arXiv 2412.08629: Euler on the
velocity difference `V_tgt − V_src` between a source-prompt and target-prompt
forward, no inversion, no training) + a velocity-difference AUTO-MASK
(per-token/voxel L2 saliency → quantile threshold → dilate → hard source-latent
copy outside the mask). One pipeline per model vertical; each documents its own
schedule/CFG/comptime-S conventions in its header:

| pipeline | model | notes |
|---|---|---|
| `pipeline/krea2_flowedit.mojo` | krea2 512² | int8 W8A8 hybrid base + int8 disk sidecar (`models/krea2/krea2_int8_cache.mojo`); 2D auto-mask; variable-LT via LT_SHARED pad + length-bucket reorder |
| `pipeline/lingbot_flowedit.mojo` | LingBot Dense-1.3B T2V 13f | UniPC sigma grid, UniPC step bypassed; 3D spatiotemporal auto-mask (3×3×3 dilate incl. temporal); comptime L_SRC/L_TGT/L_NEG per prompt (fail-loud prints the recompile value) |
| `pipeline/ideogram4_flowedit.mojo` | ideogram4 1024² | model-time τ runs 0→1 (inverted); single-trunk CFG (uncond = cond trunk + zero text) for 16GB; streamed TE `models/text_encoder/ideogram_qwen3vl_streamed.mojo`; JSON-caption src/tgt pairs |
| `pipeline/klein_edit_mojo.mojo` | Klein-9B (native edit) | NOT FlowEdit — native in-context reference conditioning, 1–2 refs (multi-ref T-offsets 10.0+r) via generalized `sampling/klein_sampler.mojo` |

Style edits: run WITHOUT `--auto-mask` (global change), `--nmax` = strength.
Insertion: add an element to the target prompt (layout control is weak).
User-image subjects: Klein multi-ref (native), or pixel composite + FlowEdit
harmonization pass.

Operating guide (recipes, gotchas, avoid-list, learnings ledger) for this whole
surface: `docs/MOJO_TRAINING_FREE_EDITING.md`.
