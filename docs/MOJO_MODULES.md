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
| `ops/linalg_backward.mojo` | `mm_backward`, `bmm`, `linear_backward`, `addbias` grads (transposed GEMMs via vendor BLAS) | `ops/linear.mojo` | **PROVEN** (`linalg_bwd_parity`) |
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

> Design doc: `RECOMMENDED_TRAINER_STRUCTURE.md` (Stage 1 IMPLEMENTED + compile-
> verified, RC=0). The OneTrainer-style seam: the SHARED training pipeline is
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
  `eps`) + nominal model dims (`d_model`/`n_heads`/`head_dim`/`mlp_hidden`/
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
  + cols on w2). **Status: PROVEN per source header** (gates `single_block_parity`
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
  single-block backward recompute uses a save-only path that stops at `out_in`
  instead of producing a discarded block output. The real trainer now routes
  scratch-aware stack wrappers through a shared two-slab `ScratchRingAllocator`
  (512 MiB x 2) for block-local concat/slice temporaries, scratch-backed frozen
  linear dx outputs, and scratch-backed SDPA backward work buffers. Together with
  the shared F32 no-bias `linear` fast path, latest clean 4B timing band is
  `3.5673797`, `3.6506753`, `3.6517751`, `3.5703504`,
  loss `2.734082`, grad `0.17687473`; still above the 2-3s target.
- **`models/klein/lora_block.mojo`** (289 L) — LoRA-on-projection helpers shared by the
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
  / `SingleModVecsDevice` chunks for the hot trainer path. **Status: PROVEN for
  cached mods** (`klein_step_mod_cache_smoke`: host and device chunks all
  max_abs 0.0).
- `models/klein/parity/load_{double,single}_block_smoke.mojo` — real-weight load
  smokes for the block weight structs.

## Scratch allocation — shared opt-in memory

| Module | Purpose / key defs | Status |
|---|---|---|
| `scratch_ring.mojo` | `ScratchRingAllocator`: OneTrainer-style fixed GPU scratch slabs (`DType.uint8`), 16-byte aligned sub-buffer allocation, forward allocation from the head, reverse allocation from the tail for backward/recompute frames, explicit `mark`/`rewind`/`reset`, and Tensor wrappers over `create_sub_buffer`. Matches the local OneTrainer pattern in `docs/RamOffloading.md` and `modules/util/LayerOffloadConductor.py`: persistent int8 cache tensors, typed slice/view reinterpretation, and ordered forward/backward allocation. The allocator is shared infrastructure for any model, but callers must opt in and own the frame lifetime; it is not a global Tensor allocator. | **PROVEN** (`scratch_ring_smoke`: clone, alignment, mark/rewind, reset, forward+reverse allocation) |
| `ops/tensor_algebra_scratch.mojo` | Opt-in scratch-backed hot shape helpers: `concat2_scratch`, `concat3_scratch`, `slice_scratch`. The F32 rank-2 dim-1 path keeps specialized kernels; other valid ranks/dims use copy-backed scratch output. Each helper can allocate from the ring head or tail (`reverse=True`) for backward/recompute frames. Kept separate from `ops/tensor_algebra.mojo` so normal model imports do not compile or use scratch kernels unless explicitly requested. | **PROVEN** (`scratch_ring_smoke`: concat2/slice/concat3 plus rank-4 generic concat/slice parity) |
| `ops/linear.mojo` | `linear_scratch`: opt-in F32 no-bias linear forward whose output storage comes from `ScratchRingAllocator`; bias and non-F32 paths fall back to normal `linear`. | **PROVEN** (`scratch_ring_smoke`: `scratch linear fwd`) |
| `ops/attention_backward.mojo` | `sdpa_backward_scratch`: opt-in decomposed SDPA backward that keeps the large recompute/work buffers in a nested scratch frame, rewinds them before return, and returns normal fresh d_q/d_k/d_v tensors. | **PROVEN** (`scratch_ring_smoke`: scratch d_q/d_k/d_v equal normal `sdpa_backward`) |

## Training orchestration — `training/`

| Module | Purpose / key defs | Status |
|---|---|---|
| `training/optim.mojo` | `adamw_step` (decoupled WD, ported to match flame-core `adam.rs` exactly — WD on `p` after the Adam step, NOT folded into `g`), SGD+momentum, global-norm grad clip | **PROVEN** (`optim_converge_parity`: AdamW bowl ratio 9.8e-15) |
| `training/schedule.mojo` | loop POLICY: `flow_match_noise_target` (the real Z-Image v-target: `x_t=(1-σ)·latent+σ·noise`, `target=noise-latent`), logit-normal timestep + qwen shift, EMA, grad-accum. Ported to match EDv2 `train_qwenimage.rs` | **PROVEN** (`schedule_parity`) |
| `training/checkpoint.mojo` | gradient checkpointing + activation offload (toy linear→silu). **Critical finding**: Mojo 1.0.0b1 cannot store a heterogeneous captured closure in a struct field → no `recompute_fn` closure like flame-core; the recompute is open-coded instead | **PROVEN** (`checkpoint_parity` toy) |
| `training/checkpoint_block.mojo` | gradient checkpointing for a FULL DiT block (saves only block input x, recomputes the whole block in backward) | **PROVEN** (`checkpoint_block_parity`: full-block dx cos 0.99999999, offload round-trip max_abs 0) |
| `training/dit_block.mojo` | reusable `dit_block_forward` / `dit_block_backward` unit (2 residual branch points, 2 fan-out points). Data crosses the API as host `List[Float32]`, not GPU Tensors (the move-only constraint; matches the proven inline gate) | **PROVEN** (`dit_block_unit_parity` + `block_composed_parity` "BLOCK COMPOSITION SOUND" at H=2) |
| `training/loop.mojo` | reusable F32-master / BF16-compute training-loop harness. **grads-as-input, NOT callbacks** (no storable closures): caller runs its own fwd+bwd, hands BF16 grads back; harness owns F32 masters + AdamW (m,v) + step `t` + grad accumulators; resumable via the byte-exact safetensors writer | **PROVEN** (`loop_parity`: trains, checkpoint round-trip byte-exact, resume continues) |
| `training/zimage_train_step.mojo` | ONE training step on a single Z-Image DiT-block FFN sub-path (synthetic), manual chained backward (NOT the 5-op tape) + AdamW; asserts loss finite, grads nonzero, loss decreases | **SCAFFOLD** (asserts only, no cos-gate vs torch; T5 assembly piece) |

---

## Data path (TRAINING) — image→latent, caption→embedding, cache, batch

The EDv2 PRECOMPUTE model: encode once to a disk cache, then the loop reads
(latent, text_embedding, mask) batches. The HEAVY encoders (~16 GB Qwen3 + VAE)
are imported only by the prepare driver, never by the loop.

| Module | Purpose / key defs | Status |
|---|---|---|
| `models/vae/klein_encoder.mojo` (439 L) | FLUX.2/Klein VAE ENCODER (image→packed latent). `struct KleinVaeEncoder` + `.encode` → `[1,128,H/16,W/16]`. Mirrors Rust `KleinVaeEncoder::encode` (`klein_vae.rs:706-872`): conv_in → 4 down_blocks (ch_mult 1,2,4,4) → mid (resnet+attn+resnet) → GroupNorm/silu → conv_out(512→64) → quant_conv → mu=first 32 ch (deterministic, NO sampling) → patchify 2×2 (inverse of decoder `_unpatchify_packed`, `pc=((c*2+ph)*2+pw)`) → BatchNorm (eps 1e-4). INFERENCE-only (VAE frozen in LoRA training); F32 end-to-end. | **PROVEN (verified-finite)** — encoded real-Alina-image latent **std 0.962** (gate target ~0.96; a HWC→CHW scramble gives ~0.85). Smoke: `pipeline/klein_encode_smoke.mojo` |
| `training/klein_dataset.mojo` (224 L) | The cache write + read path. `write_sample(latent, text_embedding, text_mask, path, ctx)` (`:52`) → one single-file safetensors (keys `latent`[1,128,h,w] / `text_embedding`[1,512,D] / `text_mask`[1,512], byte-exact, storage dtype preserved). `struct KleinCache` (`:126`): `__init__(dir)` enumerates+sorts `.safetensors` (reproducible order, mirrors `LatentDataset::new`), `count`, `peek_key` (header-only bucket key), `load(index)`, `load_batch(indices)` (concat dim-0 same-bucket samples). `BucketKey`/`KleinSample` structs. Does NOT import the heavy encoders. | **PROVEN (byte-exact)** — write/read round-trip byte-exact (master handoff writer property). |
| `io/cap_cache.mojo` | bit-exact tensor cache `save_tensor_bin(t,path,ctx):71` / `load_tensor_bin(path,ctx):113`. The separate-process encode↔train handoff for caption embeddings (so 16 GB Qwen3 + 9B DiT never co-reside). | **PROVEN** (bit-exact round trip; INFERENCE-origin, reused by training) |

---

## Validation + persistence harness (TRAINING) — sample-shift gate, LoRA save, config read

| Module | Purpose / key defs | Status |
|---|---|---|
| `training/validation_sampler.mojo` (224 L) | The L2P sample-shift gate. `generate_validation[N_IMG,N_TXT,S,LH,LW](...)` (`:169`): load resident `Klein9BDiT.load_full`, OPTIONALLY merge a LoRA (`LoraSet.load.merge_into_indexed`), denoise via the proven Flux-2 sigma schedule + Euler (forward_full ×2 pos/neg → `flux2_cfg`, since the resident model has no fused `forward_full_cfg`), VAE-decode, save PNG, RETURN the RGB tensor. `pixel_l1(a,b,ctx)` (`:213`) = mean abs diff = the WITH-vs-WITHOUT-LoRA metric (0 ⇒ LoRA not applied — the bug it hunts). `load_caps` reads cached pos/neg embeddings (no encoder loaded). | **SCAFFOLD (compiles)** — reuses only proven modules (klein DiT forward, VAE decode, LoRA merge, sigma schedule); compiled, not yet lead-run on real weights. Smoke: `validation_sampler_smoke.mojo` |
| `training/lora_save.mojo` (201 L) | `save_lora_peft(adapters, path, ctx)` (`:83`) → PEFT/ai-toolkit-keyed safetensors: `<prefix>.lora_A.weight` [rank,in] + `<prefix>.lora_B.weight` [out,rank], F32, the EXACT inverse of `lora.mojo`'s load (so `LoraSet.load` / the validation sampler / ai-toolkit/diffusers open it). `load_lora_for_resume(prefixes, scale, path)` (`:146`) reads A/B back (AdamW moments zeroed — resume those from the `loop.mojo` TrainState). Deliberately writes NO `.alpha` (matches train_klein convention; caller re-supplies alpha/rank as the merge multiplier). `struct NamedLora`. | **PROVEN (byte-exact)** — round-trips byte-exact (F32, no BF16 truncation). Smoke: `lora_save_smoke.mojo` |
| `io/train_config_reader.mojo` (350 L) | `read_train_config(json_path, d_model, n_heads, head_dim, mlp_hidden, n_layers, timestep_shift=1.8)` (`:274`) → `ReadConfigResult(TrainConfig, OptimExtra)`. Reads a OneTrainer-style JSON config (a NEW general-JSON scalar parser — float/sci-notation/bool/null + one-level `optimizer` descent — REUSING `io/json_header.mojo`'s proven `_Cursor`/`_parse_string`/`_skip_value`). Pulls `learning_rate`→lr, `lora_rank`, `lora_alpha`, `model_type`→name, nested `optimizer.eps`→eps + `weight_decay`/`beta1`/`beta2`→OptimExtra. Model dims + timestep_shift are caller-supplied (JSON doesn't carry them). Pure-Mojo file read via `io/ffi` syscalls. | **PROVEN (verified-finite)** — header (`:18-19`) records it reads `/home/alex/OneTrainer/configs/klein9b_loss_compare.json`. Smoke: `train_config_reader_smoke.mojo` |

---

## In-progress / mid-write (reconcile next session)

> Another agent is building the integrated real-data loop NOW. Documented as
> in-progress; do NOT over-claim. Reconcile against source when stable.

| Module | Intended role (per header, may change) | Status |
|---|---|---|
| `pipeline/klein_prepare_alina.mojo` (present) | REAL prepare driver for the Alina LoRA dataset: for 4 staged 512² images + captions, `KleinVaeEncoder.encode` (assert std≈0.96) + Qwen3-8B `encode_klein` (512 tok) → `write_sample` to `output/alina_cache/`. Qwen3+VAE co-reside ONLY in this process; the train process never imports Qwen3. | **IN PROGRESS** — file exists, header complete; not yet lead-run end-to-end. |
| `training/train_klein_real.mojo` (the integrated loop) | The integrated real-dim Klein LoRA timing loop (`KleinCache` reader → real `klein_stack_lora` fwd/bwd → AdamW → `save_lora_peft`). It uses cached per-step modulation weights, device modulation chunks, resident RoPE tables loaded before timing, the no-aux gate-residual d_x/d_y path, save-only checkpoint recompute for unsaved single blocks, `SGL_SAVE_TAIL = 9`, one two-slab 512 MiB scratch ring reset per step for scratch-aware Klein block temporaries, scratch-backed frozen linear dx outputs, scratch-backed SDPA backward work buffers, and the shared F32 no-bias `linear` fast path. Latest measured one-step runs: `3.5673797`, `3.6506753`, `3.6517751`, `3.5703504`, loss `2.734082`, grad `0.17687473`. | **PROVEN timing/smoke** — parity is covered by block/stack LoRA gates plus `klein_step_mod_cache_smoke`; still above the 2-3s target. |

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

### `io/` reader stack (INFERENCE, consumed by training)
`io/safetensors.mojo` (reader), `io/sharded.mojo`, `io/mmap.mojo`,
`io/json_header.mojo`, `io/tensor_view.mojo` (`TensorView` / `from_parts`),
`io/dtype.mojo`, `io/ffi.mojo`, `io/cap_cache.mojo`. The weight-load path the
training run uses to bring real Z-Image weights onto the device. Status:
**INFERENCE** (pre-existing, parity-probed under `io/parity/`).

---

## LoRA — `lora.mojo`
**INFERENCE-only** merge-at-load LoRA (`lora.mojo:1-2` header). Loads a LoRA
`.safetensors`, computes `delta_W = scale·(B@A)`, `scale=(alpha/rank)·multiplier`,
and merges in place into the base weight Dict (`W[target] += delta_W`) with
row/col/row-range slotting (`SLOT_FULL/ROWS/COLS/...`, `lora.mojo:~70+`). The
runtime `forward_lora` overlay path is **NOT** ported (needs model-forward
changes). Port of inference-flame `lora_merge.rs`. **NOT a training LoRA** — it
fuses an already-trained adapter for inference. `lora_probe.mojo` is its smoke.
Status: **INFERENCE**.

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
`hidream_o1`, `nucleus_dit`/`nucleus_moe`, `ltx2_dit`, `sensenova_u1`,
`zimage_l2p_dit` (+ `_contract.mojo` shape contracts and `_probe.mojo` smokes
per model). Status: **INFERENCE**.

### `models/text_encoder/` — conditioning
`t5_encoder.mojo` (the real Z-Image T5 run depends on this), `clip_encoder`,
`qwen3_encoder`, `qwen25vl_encoder`. Parity under
`models/text_encoder/parity/`. Status: **INFERENCE**.

### `models/vae/` — encode/decode
`zimage_decoder`, `klein_decoder`, `ldm_decoder`, `conv3d`, `wan22_decoder`,
`qwenimage_decoder`, `ltx2_*`, `vae_ops`, `upsample`, `decoder2d`. These are
the forwards whose `conv2d`/`pool`/`upsample` backward partners live in
`ops/conv2d_backward.mojo` + `ops/pool_backward.mojo`. Status: **INFERENCE**.

### Other inference subtrees
`models/pid/`, `models/lens/`, `models/lance/`, `models/upsampler/`,
`models/vocoder/`; plus `pipeline/` (end-to-end inference smokes),
`sampling/` (schedulers/flow-match samplers), `tokenizer/`, `offload/`
(block-streaming loaders for 24 GB fit), `runtime/`, `registry/`,
`components/`, `image/`. All **INFERENCE** — the training run borrows the
offloaders (`offload/`) and samplers' schedule math but does not backprop
through them.

---

## Forward op library (INFERENCE) — `ops/*.mojo` (non-`_backward`)

The forward kernels the backward partners pair with: `ops/attention.mojo`
(math-mode SDPA fwd), `ops/linear.mojo`, `ops/norm.mojo`, `ops/activations.mojo`
(`silu`, `swiglu`), `ops/reduce.mojo`, `ops/rope.mojo`, `ops/conv.mojo`
(SDK conv2d fwd wrapper), `ops/conv1d.mojo`, `ops/elementwise.mojo`,
`ops/tensor_algebra.mojo` (transpose/concat/slice/add/mul_scalar/zeros_device),
`ops/tensor_algebra_scratch.mojo` (opt-in scratch-backed shape helpers),
`ops/softmax.mojo`, `ops/cast.mojo`, `ops/embeddings.mojo`, `ops/layout.mojo`,
`ops/moe.mojo`, `ops/fp8.mojo`, `ops/mxfp4.mojo`, `ops/snake.mojo`,
`ops/pixelshuffle.mojo`, `ops/random.mojo`, `ops/unary.mojo`,
`ops/activation1d.mojo`. Status: **INFERENCE** (each has a `*_smoke.mojo`).

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
  `T5_ZIMAGE_TRAINING_MAP.md`.
- **Find the convention/idiom that's biting you**: `docs/MOJO_CONVENTIONS.md`.
- **Debug a dead/wrong gradient**: `docs/MOJO_DIAGNOSTICS.md`.
