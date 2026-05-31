# SPEEDUP QUICK-WINS — serenitymojo Klein LoRA training step

**Date:** 2026-05-30
**Scope:** READ-ONLY audit. Low-effort, low-risk changes that shrink the
**measured** transfer/sync count **without** the block-API residency redesign
(lever A1 / `TArc`) described in `AUDIT_FUSION_SPEEDUP_PLAN_2026-05-30.md`.
**Measured baseline:** ~236 s/step, ~3 GB / 24 GB resident, GPU SM idle ~77 %,
~6000+ H2D + ~3000+ D2H + many ctx.synchronize()/step.

---

## Root mechanism (one sentence)

Every op is `Tensor.from_host(...) → GPU kernel → .to_host(...)`, and **both
`from_host` and `to_host` call `ctx.synchronize()`** (`serenitymojo/tensor.mojo:145`
upload, `:320` download — also `:175,:203,:257,:309`). So each op pays
*(uploads + 1) blocking syncs*, and every frozen weight / rope table is
re-uploaded on every op of every pass. The redesign (A1) removes the host
boundary entirely; the items below remove **redundant** and **avoidable**
crossings while keeping the host-`List[Float32]` boundary intact.

---

## RANKED TABLE — ships today, no API change

| # | Fix | Site (file:line) | Effort | Risk | Reduction in copies/syncs | Parity re-gate? |
|---|-----|------------------|--------|------|---------------------------|------------------|
| **Q1** | **Stop recomputing forward in backward** — `klein_stack_lora_backward` re-runs `single_block_lora_forward` / `double_block_lora_forward` per block solely to regenerate `saved`. The forward already computed it; save the full per-block `BlockSaved`/`DoubleBlockSaved`/`SingleBlockSaved` and feed it to backward. | `klein_stack_lora.mojo:370-373` (single recompute), `:410-413` (double recompute); enabled by `KleinStackForward` saving only block *inputs* `klein_stack.mojo:288-309` | **M (1–2 d)** | **MED** (must re-gate; ~3 GB→~6–9 GB resident, still ≤24) | **~halves the whole step.** A full block-forward is ~30 syncs + ~60 H2D + ~30 D2H (per `AUDIT_FUSION_SPEEDUP_PLAN:60-90`). 32 blocks × removing one redundant forward each ≈ **−~960 syncs, −~1900 H2D, −~960 D2H per step** | **YES** — re-gate `klein_stack_real_smoke` / `block_composed_parity` (cos 0.99999999). Lower risk than it looks: backward already trusts `fwd.saved`; we just save more fields. |
| **Q2** | **Upload cos/sin rope tables ONCE, reuse the device handle.** Built once in the loop (`_build_klein_rope_host`, `train_klein_real.mojo:303`) but re-uploaded as fresh `Tensor.from_host` on every rope call: 4 uploads/double-fwd (`double_block.mojo:552-557`), 4/single-fwd (`single_block.mojo:406-411`), same in each backward (`:651-656`, `:1032-1037`), AND again in every Q1-recompute. They are constant for the entire run. | `double_block.mojo:552-557, 1032-1037`; `single_block.mojo:406-411, 651-656` | **S (0.5–1 d)** if a resident-`Tensor` rope arg is threaded; **near-zero** if combined with Q1 (recompute already gone) | **LOW** (read-only constant; bit-identical) | cos+sin uploaded ~8×/block (fwd+bwd) × 32 blocks = **~512 H2D + ~512 syncs/step** today → **2 uploads total for the run**. Even just removing the *backward-side* rope uploads (subsumed by Q1) is **~256 H2D/step**. | NO (constant data, bit-identical) — but if threaded as a `Tensor` it touches signatures, so smoke-run once. |
| **Q3** | **Hoist per-step modvec rebuild out where constant; it is correctly per-step here, but the seed vec_silu + base struct are rebuilt needlessly.** `_build_step_mods` is legitimately per-step (sigma varies). BUT `build_klein_vec_silu`/`load_klein_stack_base` re-read `time_in.*` / `img_in` / `final_layer.*` weights from the safetensors via `_load_host_f32` every call — those weights are constant. The base struct is built once (good), but `_build_step_mods` re-loads `double_stream_modulation_*` / `single_stream_modulation` weights from disk **every step** (`weights.mojo:149,162` via `_load_host_f32`). | `train_klein_real.mojo:347` → `weights.mojo:146-164` (`_load_host_f32` of mod weights per step); `_build_step_mods` `train_klein_real.mojo:228-240` | **S (0.5–1 d)** | **LOW** | Removes per-step **safetensors re-read + cast + H2D** of the 3 modulation weight matrices (each `[6D,D]`/`[3D,D]`). ~**3 re-reads + 3 H2D + 3 syncs/step** plus disk I/O. Hoist the weight `List[Float32]` load out of the loop; keep only the cheap `_linear_row` per step. | NO (same math, cached input) — smoke-run to confirm. |
| **Q4** | **`_ones(D)`/`_zeros(D)` layer-norm weights rebuilt + uploaded every layer_norm.** `_layer_norm_fwd` builds a fresh `[D]` ones and `[D]` zeros host list and uploads BOTH on every call (`double_block.mojo:121-129`); layer_norm_backward uploads `_ones(D)` again. Klein has 2 layer_norms/stream/double-block + 1/single-block. | `double_block.mojo:126` (`_t(_ones(d)...)`, `_t(_zeros(d)...)`), `:679,:777` (bwd ones); `single_block.mojo` analogous; `klein_stack.mojo:294` (final ln) | **S (0.5 d)** | **LOW** | Constant `[D]` vectors re-uploaded ~**6–8×/double-block + 2×/single-block** → ~**(8×8 + 2×24) ≈ 112 H2D + 112 syncs/step** just for ones/zeros. Upload one resident ones[D]/zeros[D] at load, reuse. | NO (bit-identical constants). |
| **Q5** | **LoRA A/B re-uploaded per op in fwd AND bwd recompute.** `klein_lora_fwd`/`klein_lora_bwd` upload `lo.a`/`lo.b` fresh every call (`lora_block.mojo:51,57,89,99,108`). A/B change once/step (after AdamW), not per op. With Q1 the bwd recompute uploads vanish; the remaining fwd uploads of A/B are small ([rank,in]) but still 1 sync each. | `lora_block.mojo:51,57,89,99,108` | **S (0.5 d)** | **LOW** | 80 adapters × (A+B) × (fwd + bwd-recompute). Q1 removes the recompute copies; resident-A/B (refreshed post-AdamW) removes the rest: ~**160 H2D + 160 syncs/step**. Lowest byte-count of the set (rank=16). | NO (data identical within a step). |

### Combined effect of Q1+Q2 (the two that matter)

Q1 alone roughly **halves** every count below by deleting the redundant
backward-side forward pass. Q2 then strips the rope re-uploads from what
remains. Together they target the **largest contributors** to the
~6000 H2D / ~3000 D2H / many-sync budget without touching the proven host-list
composition math — only *which* tensors get re-uploaded and *whether* the
forward runs twice.

---

## NEEDS THE RESIDENCY REDESIGN (out of scope here — see AUDIT_FUSION_SPEEDUP_PLAN A1)

| Item | Why it needs A1 |
|------|------------------|
| Eliminate the **activation** host round-trip between ops (the `_add_lists` residual sums, qkv/gate-up split, concat/slice on host) | These are the host-side branch carriers; moving them on-device is exactly the `TArc`/`ArcPointer[Tensor]` rewrite (`AUDIT_FUSION_SPEEDUP_PLAN` A1). Changing them changes the gated composition path → full redesign + re-gate. |
| Make **frozen base block weights** (`wqkv/wproj/wgu/wd/q_norm/k_norm`, `w1/w2`) resident instead of `from_host` per op | This is lever **A2** (3–5 d). Bigger than the items above because the per-op weight upload is wired through `_linear_fwd`/`_rms_fwd_4d` etc. and the weight structs are `List[Float32]`. Real win but not "hours." |
| Collapse the per-op `to_host` sync to one-per-block | Lever **A3** — only possible once activations stay on-device (subsumed by A1). |
| Device-side `_add_lists` / qkv-split / gate-up split / `_concat_seq`/`_split_seq` (host CPU loops today: `klein_stack.mojo:135,145`; `double_block.mojo:398,490`; `_latent_to_img_tokens` pack `train_klein_real.mojo:130-137`) | Each is a CPU loop on host lists that *only exists because the data is already on host*. They become free/device once A1 keeps tensors resident; rewriting them standalone would add device round-trips, net-negative before A1. |
| dtype: activations round-trip as **F32** host lists (`STDtype.F32` everywhere in the block path) | BF16-on-device would halve transfer bytes, but flipping dtype on the host-list boundary changes the numerics that were gated to cos 0.99999999 and interacts with A1. Defer to A1 (which is BF16-first per `tensor.mojo:9`). |

---

## TOP 3 QUICK WINS (effort in hours)

1. **Q1 — Save forward activations, delete the backward's recompute forward.**
   ~8–16 h. `klein_stack_lora_backward` (`klein_stack_lora.mojo:370-373, 410-413`)
   currently calls the block *forward* again per block purely to rebuild `saved`.
   Extend `KleinStackForward` to retain each block's full `BlockSaved` (it already
   retains the block *inputs* at `klein_stack.mojo:288-309`) and pass it straight
   to backward. **~Halves the entire step** (−~960 syncs, −~1900 H2D, −~960 D2H).
   Memory headroom exists (3 GB → ~6–9 GB of 24 GB). Re-gate `klein_stack_real_smoke`.

2. **Q2 — Upload cos/sin rope tables once for the run.** ~4–8 h (near-zero if
   done with Q1). Built once (`train_klein_real.mojo:303`) yet re-uploaded as
   fresh `from_host` on every rope call (`double_block.mojo:552-557`;
   `single_block.mojo:406-411` and the backward twins). Constant for the whole
   run → **~512 H2D + ~512 syncs/step** removed (≥~256 of them die for free with Q1).

3. **Q4 — Resident `ones[D]`/`zeros[D]` for layer_norm.** ~4 h. `_layer_norm_fwd`
   rebuilds and uploads two constant `[D]` vectors on every layer_norm call
   (`double_block.mojo:126`, plus the `_ones(D)` in every `layer_norm_backward`).
   Upload once at load, reuse → **~112 H2D + ~112 syncs/step** removed, zero
   numeric change.

(Q3 hoist-mod-weight-load and Q5 resident-LoRA-A/B are the next two; both LOW
risk, ~4–8 h each, smaller byte counts.)

---

## Verification notes for whoever implements

- Every count above is derived from `tensor.mojo:145/:320` (each `from_host`/
  `to_host` = 1 sync) × the per-op site counts (`double_block.mojo` has 104
  `from_host`/`to_host` refs, `single_block.mojo` 62) × the block counts (8
  double + 24 single = 32, `AUDIT_TRAINING_READINESS_2026-05-30.md:29`). They are
  order-of-magnitude, not profiled per-line — **re-measure with the existing sync
  counter before/after each change** (Tenet 4: measurement beats assertion).
- Q1 is the only MED-risk item because it touches the gated composition path.
  It does **not** change the math — backward already consumes `fwd.saved`; we
  only change `saved` from "block inputs, recompute the rest" to "full saved,
  no recompute." Re-gate after.
- Do **not** attempt the activation-boundary or base-weight-residency changes as
  "quick wins" — they are lever A and will fight the borrow checker and the
  parity gates. They belong in the planned A1/A2 work.
