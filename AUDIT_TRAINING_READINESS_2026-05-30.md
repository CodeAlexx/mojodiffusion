# Training-Readiness Audit — Mojo training port (serenitymojo) vs flame-core

> Date: 2026-05-30. READ-ONLY audit (no edits, no compile — builder holds the
> compile lock). Every claim cites a `file:line` read directly this session.
> Question answered: **"What stands between this and actually training Klein
> 4b/9b and Z-Image LoRA for real (not synthetic)?"**
>
> Tenet-4 discipline: structural claims (`file:line`, struct fields, what is /
> isn't present) were READ this session. Numeric cos-vs-torch values are
> **MEASURED-by-prior-lead** (master handoff §2, re-run on a clean serial
> build) — NOT re-run here. The audit does not assert any number it could not
> trace to a source.

---

## TL;DR — the actual finish line

The training **engine** (tape, per-op backward kernels, optimizer, schedule,
checkpoint, resumable loop) is real and parity-gated. But **no real model can
train today**, for reasons that have nothing to do with the engine:

1. There is **no weight loader** (safetensors → `BlockWeights`). A reader
   exists; nothing maps real Klein/Z-Image keys onto the block-weight struct.
2. The proven block (`dit_block.mojo`) is an **un-modulated** transformer
   block. Real Klein and Z-Image blocks are **AdaLN-modulated** (timestep
   conditioning: shift/scale/gate). That modulation — forward AND backward —
   is **not assembled** into a block.
3. The Klein **double-stream block** (joint txt+img attention) does not exist
   as code. Klein = 5/8 double-stream + 20/24 single-stream blocks.
4. Everything proven is proven at **toy dims** (`D=8, H=2, Dh=4, M=4`) on
   **synthetic data**. The shared `train_step` literally substitutes toy
   comptime dims and runs random tensors. No real-dim run, no dataset.
5. The trained "LoRA" is a single **synthetic adapter on the block input** —
   not the real per-projection (wq/wk/wv/wo) LoRA-target map, and there is no
   LoRA save in PEFT/ai-toolkit format.

The **SDPA H=30 "blocker" from the master handoff is FALSE** — it was
degenerate test data, now disproven by a non-degenerate gate (see §2). So the
engine's headline risk is retired; the remaining work is **assembly + loaders +
real-dim re-gating**, not autograd correctness.

---

## 1. READINESS GAP TABLE

| # | Gap | Sev | Category | Evidence (`file:line`) | flame-core comparison |
|---|-----|-----|----------|------------------------|------------------------|
| G1 | **No safetensors→BlockWeights loader.** Reader exists (`SafeTensors.open`, `tensor_bytes`, mmap) but nothing builds a `BlockWeights`/per-block tensor set from real Klein/Z-Image keys (fused-qkv split, key-name mapping). | **BLOCKER** | reader: `io/safetensors.mojo:64,76,172`; consumer absent — `models/klein/train.mojo:9-13` ("add weight loader (GAP G1)"); `train_step.mojo:_make_block_weights` synthesizes via `_randn` | flame-core trainers load base weights + auto-detect variant from `img_in.weight` first dim (`eridiffusion-core/src/models/klein.rs:24`); base model resident, LoRA on top. |
| G2 | **AdaLN modulation not assembled (fwd+bwd).** `dit_block.mojo` is rms→qkv→sdpa→out→res→rms→swiglu→res with **no** timestep modulation. Real blocks do `scale_msa/gate_msa/scale_mlp/gate_mlp = chunk4(adaLN_modulation(adaln_input))`, `gate=tanh(gate)`, `scale=1+scale`. | **BLOCKER** | absent in `training/dit_block.mojo` (grep: only comment hits, no modulate code); required by inference block `models/dit/zimage_dit.mojo:21-26,407-428` | flame-core `modulate_pre`/`fused_residual_gate` are tape ops inside `double_block_forward`/`single_block_forward`; backward flows through `Op::Mul`/`AddBias`/gate-residual (FLAME_AUTOGRAD §8.2). |
| G3 | **Klein double-stream block missing entirely.** Klein 9B = 8 double + 24 single; 4B = 5 double + 20 single. The proven block is single-stream only. Double-stream = joint txt+img attention with two modulation sets — a genuinely new compute unit, fwd+bwd, ungated. | **BLOCKER** (Klein only) | only a comment: `models/klein/train.mojo` ("Klein double-stream block (GAP G3)"); no `double_stream` code (grep training/, models/klein) | flame-core `double_block_forward` (`model.rs:935-1089`, ~210 tape entries/block, FLAME_AUTOGRAD §8.2). |
| G4 | **Proven only at toy dims H=2/D=8/M=4 on synthetic data.** The shared step substitutes `_D=8,_H=2,_Dh=4,_M=4` and runs `_randn` tensors; real cfg dims (Klein D=4096/H=32, Z-Image D=3840/H=30) are carried in `TrainConfig` for documentation only. | **HIGH** | `train_step.mojo:48-53` (toy comptime), `:run_synthetic` overrides cfg dims with `_D/_H/_Dh`; `:_randn` synthetic latent/noise | flame-core trains at real dims with real cached latents + text-encoder conditioning (train_klein.rs pipeline header :3-13). |
| G5 | **No data path.** No latent-cache reader, no dataset, no conditioning (text-encoder embeddings) wired into the training step. | **BLOCKER** | grep training/ + models/{klein,zimage}: only `loop_parity.mojo` references a cache; train step takes `_randn` latent/noise | flame-core: cached-latent dirs, caption dropout, multires noise, timestep bias, masked loss (train_klein.rs imports :26-35). |
| G6 | **No LoRA training target-module map; no LoRA save.** `lora.mojo` is **inference-only merge-at-load** (overlay/backward NOT ported, `lora.mojo:11-14`). `train_step` trains ONE synthetic adapter on the **block input**, not wq/wk/wv/wo/etc. No `lora_targets.mojo`/`weights.mojo` in models/. No PEFT-format LoRA writer. | **BLOCKER** | `lora.mojo:1-14`; `train_step.mojo:_make_lora` (single adapter, in_f=out_f=D); `models/{klein,zimage}/` lack lora_targets/weights | flame-core: LoRA on img_qkv/img_out/txt_qkv/txt_out (double), qkv/out (single) (FLAME_AUTOGRAD §8.1), PEFT/ai-toolkit save format (project memory). |
| G7 | **9 ops wired to the tape; the model is hand-chained, not tape-driven.** `tape.backward()` dispatches add/sub/mul/matmul/linear/rmsnorm/silu/swiglu/mse only. sdpa/rope/qkv-split/gate-residual/shape/conv/pool are standalone kernels hand-threaded in `dit_block.mojo` via host `List[Float32]`. | **MED** | `autograd.mojo:52-60` (OP_ADD..OP_MSE); `MOJO_AUTOGRAD_INTERNALS §1.6`; hand-chaining `dit_block.mojo:20-34` | flame-core: 47-variant `Op` enum, full block recorded to one tape (FLAME_AUTOGRAD §5.1). Mojo's choice is viable for assembly but every new block kind must be hand-chained correctly (G2/G3 risk lives here). |
| G8 | **Host round-trip per op.** `dit_block.mojo` crosses every op boundary as host `List[Float32]` (`.to_host(ctx)` after each op) because Tensor is move-only and branch points need Copyable carriers. Correct, but a real 30–32-block forward+backward at S~384–2304 will pay enormous H2D/D2H + sync cost. | **MED** (perf, not correctness) | `dit_block.mojo:21-34` (data-flow contract); `train_step._lora_fwd` round-trips every linear | flame-core keeps activations on-device; saved tensors are Arc bumps (FLAME_AUTOGRAD §6.1). Mojo's per-op `Tensor.clone` is a **deep device copy** (`tensor.mojo:69-80`, MOJO_AUTOGRAD §1.5). |
| G9 | **Grad-clip is a hardcoded 2-tensor case.** `clip_grad_global_norm(g1,g2,...)` does global-L2 over exactly two grads via host readback. Real LoRA has many adapters (2 tensors × N target modules × N blocks). | **MED** | `optim.mojo:234-282` (two params, `to_host` sum-of-squares) | flame-core: multi-tensor global-norm clip across all params, single kernel (train_klein.rs:1678 "multi-tensor kernel launch"). |
| G10 | **Klein 4B config dims are PLACEHOLDER and the block-split is wrong.** `klein_4b()` n_layers=32; flame-core says 4B = **5 double + 20 single = 25 blocks**, inner=3072, heads=24, mlp_hidden=9216. `TrainConfig` has no double/single split field at all. | **HIGH** | `models/klein/config.mojo:klein_4b` (n_layers=32, "PLACEHOLDER dims (GAP G2)"); authoritative dims `eridiffusion-core/src/models/klein.rs:4-5` | flame-core klein.rs lists exact per-variant double/single counts + auto-detect. |
| G11 | **EMA exists but is not wired into a model loop; no EMA persistence.** `ema_update` is a proven kernel; no training loop calls it, and loop.mojo checkpoint saves masters/m/v only (no shadow). | **MED** | `schedule.mojo:ema_update` (kernel only); `loop.mojo:34-37` (checkpoint keys: param/adam_m/adam_v/__meta__ — no ema) | flame-core: full `ParameterEma` with inv-gamma/power/min/max decay, validation-swap (train_klein.rs:156-176). (flame-core also does NOT persist shadow across resume — parity, not a gap.) |
| G12 | **mlp_hidden for Z-Image is a placeholder.** `zimage()` mlp_hidden=2560 flagged "placeholder pending real weight header." Wrong FF width → wrong weight shapes at load. | **MED** | `models/zimage/config.mojo` ("(*) mlp_hidden placeholder") | Resolved at load by reading the real header (which G1's loader must do). |

---

## 2. "PROVEN AT TOY, UNVERIFIED AT REAL" list (must re-gate at real dims)

Every composition/op below is green ONLY at toy/synthetic shapes. Each must be
re-gated at real Klein (D=4096, H=32, Dh=128) / Z-Image (D=3840, H=30, Dh=128)
dims and real sequence lengths (S∈{384,1152,2304}) before a real run.

| Proven artifact | Gate file | Proven at | Must re-gate at | Why it matters |
|---|---|---|---|---|
| Full single-stream DiT block fwd+bwd ("BLOCK COMPOSITION SOUND") | `training/parity/block_composed_parity.mojo` | **H=2** | H=30 (Z-Image), H=32 (Klein single) | Composition correctness (residuals/fan-outs) was the headline proof — but at H=2. MOJO_AUTOGRAD §5.3 CAVEAT says so explicitly. |
| 3-block stack train (inter-block d_x→d_y handoff) | `training/parity/stack_train_parity.mojo` | **H=2, 3 blocks** | 30–32 blocks at real H | Depth + dtype interplay at real width is untested. |
| `dit_block_unit` (reusable fwd/bwd) | `training/parity/dit_block_unit_parity.mojo` | **H=2** | real H | The unit the model assembly will instantiate ×30–32. |
| Gradient checkpointing (full block) | `training/parity/checkpoint_block_parity.mojo` | toy block dims | real block dims (and a 24 GB memory check) | Required to fit Klein 9B / Z-Image on 24 GB; the offload round-trip must hold at real tensor sizes. |
| SDPA backward d_q/d_k/d_v | `ops/parity/sdpa_bwd_nondegen_parity.mojo` (H=30/6/32, S=384) | **non-degenerate H=30 PASS** (handoff-stale "FAIL" disproven) | larger S (1152/2304); note S=2304 d_k cos 0.9975 watch | The disproven blocker. One precision watch remains (F32 accumulation order at S=2304), not corruption. |
| Mixed-precision (F32 master / BF16 compute) one step | `training/parity/mixed_precision_parity.mojo` | toy | real-dim block (BF16 base + F32 LoRA master) | The real run is BF16-base; BF16 compute through a real block is unverified. |
| Resumable loop (TrainState) | `training/parity/loop_parity.mojo` | toy params | real param count (thousands of LoRA tensors) | Resume byte-exactness at real param scale + real checkpoint size. |
| AdamW / SGD / grad-clip convergence | `ops/parity/optim_converge_parity.mojo` | toy bowl + 2-tensor clip | many-tensor clip (G9) | Optimizer math proven; multi-tensor clip is not. |
| ~68 per-op backward arms | `ops/parity/*_bwd_parity.mojo` | shapes ≤1024, mostly 32-aligned | real attention/MLP shapes | Per-op parity is the foundation; real-shape regressions (alignment, S-length) are the class the H=30 false-positive came from. |

**The cautionary tale (Tenet 4):** the H=30 "silent-zero" was a FALSE GREEN's
mirror image — a FALSE RED from degenerate oracle data (V constant across seq ⇒
grad_scores genuinely 0 ⇒ cos-of-two-zeros = noise). Caught only by a
real-dimension non-degenerate gate (`sdpa_bwd_nondegen_parity.mojo` header;
`models/zimage/train.mojo:11-18`). **Same discipline must gate every item
above before "done."**

---

## 3. WEIGHT-LOADING gap (explicit answer)

- **Does a safetensors READER exist?** YES.
  `io/safetensors.mojo` — `SafeTensors.open(path)` (`:76`), mmap-backed
  (`MmapRegion`), `tensor_bytes(name)` (`:172`), `tensor_info`/`names`/`count`
  (`:211/:217/:224`), prefetch + release-to-OS. Plus a sharded reader
  (`io/sharded.mojo`) and `TensorView`/`from_parts` (`io/tensor_view.mojo`)
  per MOJO_MODULES.
- **Does a WRITER exist?** YES.
  `io/safetensors_writer.mojo` (`:1-278`), byte-exact, opens in Python
  `safetensors` (MEASURED-by-prior-lead).
- **Can real Klein/Z-Image weights → `BlockWeights` today?** **NO.**
  - `train_step.mojo` builds `BlockWeights` ONLY via `_make_block_weights`
    (`_randn` synthetic), not from any file.
  - No code maps real weight keys → the 9-field `BlockWeights`
    (wq/wk/wv/wo/wg/wu/wd/g1/g2). The fused-qkv→wq/wk/wv split, the AdaLN
    modulation weights, the t_embedder, and the double-stream weight set are
    all unaddressed.
  - `models/klein/train.mojo:9-13` and `models/zimage/train.mojo:20-22` both
    flag this as "GAP G1 (weight loader)."
  - **Reusable asset:** `lora.mojo` already has the Klein/Z-Image key-naming
    + format-detection logic (`FMT_KLEIN_TRAINER`/`FMT_ZIMAGE_TRAINER`,
    `_detect_format`, fused-QKV slot constants, `lora.mojo:60-155`) — the
    save-key convention and QKV slicing for a loader/writer can be lifted from
    there rather than re-derived.

**Net:** the byte-level I/O primitives are done; the **model-specific mapping
layer** (keys→struct, fused-QKV split, modulation/t-embedder weights, variant
auto-detect) is the entire G1 gap.

---

## 4. flame-core PARITY-OF-APPROACH check

Does the Mojo step match flame-core's training contract? Per-element:

| Contract element | flame-core | Mojo port | Verdict |
|---|---|---|---|
| **Flow-match target** | `x_t=(1-σ)·latent+σ·noise`, `target=noise-latent` (train_qwenimage.rs:1093-1099) | `flow_match_noise_target` byte-for-byte same (`schedule.mojo` :309 region, kernel `_flow_match_kernel`) | ✅ MATCH (cites the exact Rust lines) |
| **Timestep sampling** | LogitNormal(weight=0)→`sigmoid(N(0,1))`, then qwen-shift clamp [1/1000,1] | `sample_timestep_logit_normal` (`schedule.mojo:~253`) — same RNG→sigmoid→shift→clamp, cites timestep_dist.rs:181-189 + train_qwenimage.rs:411-414 | ✅ MATCH. **Note:** Klein's real σ-map is `(floor(t)+1)/1000` over discrete t (train_klein.rs:8), a per-model σ-map — the per-model `sigma_map` seam (RECOMMENDED_TRAINER_STRUCTURE) is **not yet implemented**; the shared step uses the qwen logit-normal path for both. |
| **Optimizer (AdamW, decoupled WD)** | WD applied to `p` after the Adam step, not folded into `g`; FP32 state; 1-based bias correction | `adamw_step` (`optim.mojo:142`) decoupled WD, F32, host `1-β^t` bias correction — explicitly "matches flame-core adam.rs" | ✅ MATCH (math); ⚠️ single-tensor F32 only (no 8-bit, no stochastic-round, no multi-tensor — MOJO_KERNELS §12). Klein/Z-Image don't need 8-bit, so acceptable. |
| **Grad clipping** | multi-tensor global-L2, max_norm=1.0 | `clip_grad_global_norm` correct math but **2-tensor hardcoded**, host readback (G9) | ⚠️ PARTIAL — math parity, scale gap. |
| **EMA** | `ParameterEma` inv-gamma/power/min/max decay, validation-swap | `ema_update` kernel correct (decay·shadow+(1-decay)·live) but **not wired**, no decay schedule, no validation-swap (G11) | ⚠️ PARTIAL — primitive present, policy absent. |
| **Checkpoint / resume** | LoRA + AdamW(m,v,t) + step counter; full vs lora resume modes | `loop.mojo` TrainState saves param/adam_m/adam_v/__meta__(t,accum); resume = max_abs 0 on masters + t restored (`loop.mojo:34-41`) | ✅ MATCH (mechanism). Klein's "full resume refuses if rank/alpha differ" guard is trainer policy, not yet present. |
| **Loss** | mean MSE in F32 | `_mse_loss`/`_mse_grad` inline in `train_step.mojo` (the tape's `mse_backward` carries 2/N) | ✅ MATCH (math). ⚠️ inline host loss, not the tape leaf, in the shared step. |
| **Frozen base / LoRA-only grads** | `needed_grad_ids` filter drops frozen-weight grads (~5 GB saving) | No frozen-weight filter; only the id-0 sentinel drops grads (MOJO_AUTOGRAD §2.5). Base block weights are simply never given a LoRA/never optimized | ⚠️ DIVERGES — correctness OK for LoRA (base never updated), but memory: every saved base activation/grad is materialized; at 30–32 real blocks this is a 24 GB risk (ties to G8). |
| **Conditioning (text-encoder)** | cached text embeddings drive cross/joint attention | Not wired into the step (G5) | ❌ ABSENT |

**Summary:** the **recipe math** (flow-match, timestep, AdamW, loss, checkpoint)
is faithfully ported and cites the exact flame-core/EDv2 lines. The **policy
layer** (multi-tensor clip, EMA schedule, frozen-grad memory filter, per-model
σ-map, conditioning) and the **memory strategy** for a real 24 GB run are the
divergences.

---

## 5. ORDERED PHASE SUGGESTION → real Klein-9B LoRA run

Dependencies noted. Each phase ends in a **real-dim parity gate** (Tenet 4).

**Phase A — Retire the toy ceiling (gates only, no new compute).**
A1. Re-run `block_composed_parity` + `dit_block_unit_parity` at **H=32 / H=30**
    (currently H=2). Depends on: nothing. Proves the proven block actually
    composes at real width before anything is built on it.
A2. Re-run `stack_train_parity` at real H with ≥4 blocks; confirm checkpointing
    round-trip at real block tensor size. → unblocks trusting the assembly.
*(If A1/A2 regress, fix before any assembly — this is the cheapest place to
catch a real-dim composition bug.)*

**Phase B — Z-Image first (single-stream, NOT double-stream; H=30 disproven).**
B1. **G2 — AdaLN modulation block.** Build modulated single-stream fwd+bwd:
    `scale=1+chunk`, `gate=tanh(chunk)`, two modulation sets, residual-gate.
    Reuse existing standalone kernels (gate_residual_backward, broadcast-mul,
    tanh) — hand-chain like `dit_block.mojo`. Gate vs torch at H=30. Depends: A1.
B2. **G1 — Z-Image weight loader.** safetensors→per-block weights incl.
    adaLN_modulation, t_embedder, norms; resolve mlp_hidden (G12) from the real
    header; fused-QKV split. Reuse `lora.mojo` key logic. Depends: B1 (defines
    the target struct).
B3. **G6 — real LoRA targets + PEFT writer.** Map Z-Image LoRA modules (split
    attention/feed_forward), wire real per-projection adapters (replace the
    single synthetic block-input adapter), save via safetensors_writer in
    ai-toolkit format. Depends: B2.
B4. **G5 — data path.** Latent-cache reader + cached text embeddings →
    conditioning. Depends: nothing structurally, but needed for B5.
B5. **Assemble 30-layer Z-Image** = modulated block ×30 + noise/context refiners
    + checkpoint_block (24 GB fit) + loop (F32-master/BF16) + schedule (v-target)
    + writer. Add G9 (multi-tensor clip) + G11 (EMA wiring) here. Real-dim
    smoke: 1→5→50 steps, monitor loss + grad-norm + LoRA-B nonzero ratio.
B6. **Real run verdict** = loss drops **AND a sample shifts on the trigger**
    (the L2P lesson — never loss/no-crash alone). NOT agent-completable.

**Phase C — Klein (adds the genuinely-new double-stream unit).**
C1. **G3 — double-stream block** fwd+bwd (joint txt+img attention, two
    modulation sets). Its own parity gate before use. Depends: B1 (modulation
    machinery), A1.
C2. **G10/G12 — fix Klein 4B dims** (5 double + 20 single = 25 blocks, inner
    3072/heads 24/mlp 9216) and add a double/single split to `TrainConfig`.
    Depends: nothing.
C3. **G1 (Klein) loader** — fused-QKV, variant auto-detect from `img_in.weight`
    first dim. Depends: C1.
C4. **Assemble Klein** = 8 double + 24 single (9B) / 5+20 (4B) + offload +
    loop. Reuse B3/B4/B5 LoRA/data/loop infra. Real-dim smoke + run verdict.

**Critical-path to "real Klein-9B LoRA run":** A1 → B1 → C1 → C3 → C4. Z-Image
(Phase B) is the faster first real run (no double-stream, H=30 already
disproven) and de-risks every shared piece (loader pattern, LoRA targets, data,
assembly) that Klein then reuses.

---

## TOP 5 BLOCKERS (ranked)

1. **G1 — No weight loader (safetensors → BlockWeights).** Nothing maps real
   model weights onto the block struct; training runs on `_randn`. Reader
   exists; the model-specific mapping layer is entirely absent.
   (`train_step.mojo:_make_block_weights`, `io/safetensors.mojo:76`.)
2. **G2 — AdaLN modulation block not assembled (fwd+bwd).** The proven block is
   un-modulated; real Klein/Z-Image blocks are timestep-modulated. Backward
   kernels exist but the modulated block is not hand-chained or gated.
   (`training/dit_block.mojo` vs `models/dit/zimage_dit.mojo:21-26`.)
3. **G3 — Klein double-stream block missing (code, not comment).** Klein needs
   5/8 double-stream joint-attention blocks; only single-stream exists.
   (`models/klein/train.mojo` GAP note; no double_stream code.)
4. **G6 — No real LoRA target map + no LoRA save.** Trains a single synthetic
   adapter on the block input; `lora.mojo` is inference-merge-only; no PEFT
   writer. A trained adapter today is neither on the right modules nor savable
   for inference. (`lora.mojo:11-14`, `train_step.mojo:_make_lora`.)
5. **G5 — No data path / conditioning.** No latent cache, no dataset, no
   text-encoder embeddings in the step. (grep training/ + models/.)

(Severity-adjacent: **G4** — everything is toy-dim synthetic — is the umbrella
the above sit under; **G10** Klein-4B placeholder dims is a correctness landmine
once G1 lands. The much-feared **SDPA H=30** is NOT on this list: disproven.)
