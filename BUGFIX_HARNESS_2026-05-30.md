# BUGFIX_HARNESS_2026-05-30 — Source-fidelity audit of the Mojo training harness

Round: READ-ONLY skeptic pass over the builder's three harness pieces.
Tenet 4 reminder: **every patch below is a HYPOTHESIS until the LEAD
compiles + runs it.** Nothing here was compiled (builder holds the write
lock; lead compiles after). Cite-both-sides line refs are exact as of the
files' state on 2026-05-30.

## 0. What actually exists (scope correction)

| Piece (task) | Builder file | Status |
|---|---|---|
| Config reader | `serenitymojo/training/train_config.mojo` + `models/klein/config.mojo` + `models/zimage/config.mojo` | Present — **but HARDCODED Mojo constructors, not a JSON reader.** No OneTrainer JSON is parsed at runtime. |
| LoRA save / resume | `serenitymojo/training/lora_save.mojo` (adapter export) + `training/loop.mojo` (`save_checkpoint`/`load_checkpoint`, generic optimizer-state) | Present. |
| Validation sampler | `serenitymojo/training/validation_sampler.mojo` | **MISSING.** Referenced only as a comment in `lora_save.mojo:13`. Not written this round. |

Consequence for the task framing:
- "config key → TrainConfig field map" is a **build-time transcription check**
  (are the hardcoded constants faithful to the JSON?), not a parser-mapping
  check. Done in §B.
- "sampler L2P trap" cannot be audited — the sampler does not exist yet. Note
  in §D is a forward-looking contract the sampler MUST honor.

---

## A. LoRA SAVE/LOAD key + orientation contract

### A.1 The loader contract (lora.mojo — the authority the save must invert)

| Field | Value | lora.mojo ref |
|---|---|---|
| A key suffix (DiffusionModel fmt) | `.lora_A.weight` | `lora.mojo:152` (`_suffix_a`) |
| B key suffix (DiffusionModel fmt) | `.lora_B.weight` | `lora.mojo:161` (`_suffix_b`) |
| Format detect from `.lora_A.weight` suffix | → `FMT_DIFFUSION_MODEL` | `lora.mojo:120` (suffix match) → `lora.mojo:143` (return) |
| A ("down") orientation | `[rank, in_features]` | `lora.mojo:15`, reload `lora.mojo:496` |
| B ("up") orientation | `[out_features, rank]` | `lora.mojo:16`, reload `lora.mojo:497` |
| delta_W | `scale * (B @ A)` → `[out, in]` | `lora.mojo:17`, `_compute_delta` `lora.mojo:489-505` |
| B@A compute | `linear(B, transpose(A))` (foundation `linear=x@wᵀ`) | `lora.mojo:505` |
| scale | `(alpha / module_rank) * multiplier` | `lora.mojo:18`, `_module_scale` `lora.mojo:462-486` |
| `module_rank` source | `A.shape[0]` (read from file, not hardcoded) | `lora.mojo:479` |
| alpha source | `<prefix>.alpha` scalar tensor (shape `[]`) **if present**, else defaults to `module_rank` (→ scale=multiplier) | `lora.mojo:482-486`, `_read_scalar_alpha` `lora.mojo:357` |

### A.2 The builder's save (lora_save.mojo) — orientation VERDICT: FAITHFUL

| Emitted | lora_save.mojo ref | Matches loader? |
|---|---|---|
| `<prefix>.lora_A.weight` shape `[rank, in]` (`= LoraAdapter.a`) | `lora_save.mojo:118-119` | ✅ matches `lora.mojo:15/496` |
| `<prefix>.lora_B.weight` shape `[out, rank]` (`= LoraAdapter.b`) | `lora_save.mojo:120-121` | ✅ matches `lora.mojo:16/497` |
| F32 storage, byte-exact reload | `lora_save.mojo:71-75` | ✅ reader upcasts via `to_host`; round-trip safe |
| In-memory orientation it serializes | `train_step.mojo:121-122` (`a:[rank,in], b:[out,rank]`) | ✅ same as PEFT |
| PEFT init A=randn / B=0 | `train_step.mojo:150-157` | ✅ matches `eridiffusion-core/lora.rs:41-56` |

The orientation round-trip is correct. The `safetensors_writer.mojo` itself
is a faithful generic inverse of the reader (8-byte LE header_len, compact
JSON header in insertion order, contiguous `data_offsets`, raw D2H byte copy
with no F32 cast — `safetensors_writer.mojo:1-29, 220-278`). No orientation
bug.

### A.3 DIVERGENCE D1 — save omits the per-module `.alpha` scalar

The canonical EriDiffusion-v2 writer **does** emit `<prefix>.alpha`:

> `eridiffusion-core/src/lora.rs:131-139` `save_tensors`:
> ```
> out.insert(format!("{prefix}.lora_A.weight"), a);
> out.insert(format!("{prefix}.lora_B.weight"), b);
> out.insert(format!("{prefix}.alpha"), alpha);   // <- scalar shape []
> ```
> with the doc (`lora.rs:128-130`): "The `.alpha` scalar prevents loaders
> from falling back to `scale = 1.0`, which would over-apply adapters trained
> with `alpha < rank`."

The Mojo save deliberately skips it (`lora_save.mojo:31-38`), relying on the
caller re-supplying `multiplier = alpha/rank` at merge time.

Why this matters (severity: **medium, latent**): for the klein9b recipe
`alpha == rank == 16` (OneTrainer `klein9b_loss_compare.json:23-24`), so
`alpha/rank == 1` and the omission is harmless — the loader's
default-to-module_rank path (`lora.mojo:486`) reproduces the right scale with
`multiplier=1.0`. **But** the moment anyone trains `alpha != rank` (e.g.
`alpha=8, rank=16`), a file written by `lora_save.mojo` and loaded by ANY
external tool (ai-toolkit, diffusers PEFT) or by `lora.mojo` with the default
`multiplier=1.0` will silently over-apply the adapter 2×. This is exactly the
`alpha < rank over-applies` failure mode in memory
(`project_klein_grad_runaway_proven_2026-05-26`). It is a fidelity gap vs the
named reference (`lora.rs:138`), not just a style choice.

**PATCH P1 (HYPOTHESIS — write `.alpha` to match lora.rs:138).** Make the save
the full inverse of the loader's alpha path. Requires plumbing `alpha` into
`NamedLora` (the `LoraAdapter` carries only `scale = alpha/rank`, not alpha &
rank separately — `train_step.mojo:126,153`). Two sub-changes:

P1a — carry alpha on the saved struct (`lora_save.mojo:59-62`):
```
# OLD
@fieldwise_init
struct NamedLora(Copyable, Movable):
    var prefix: String
    var adapter: LoraAdapter
```
```
# NEW
@fieldwise_init
struct NamedLora(Copyable, Movable):
    var prefix: String
    var adapter: LoraAdapter
    var alpha: Float32      # raw alpha (= scale * rank); written as <prefix>.alpha
```

P1b — emit the scalar in the save loop (`lora_save.mojo:121`, after the B append):
```
# OLD (after line 121)
        names.append(nl.prefix + ".lora_B.weight")
        tensors.append(ArcPointer(_f32_2d(a.b.copy(), a.out_f, a.rank, ctx)))
```
```
# NEW
        names.append(nl.prefix + ".lora_B.weight")
        tensors.append(ArcPointer(_f32_2d(a.b.copy(), a.out_f, a.rank, ctx)))
        # <prefix>.alpha — scalar shape [], matching eridiffusion-core/lora.rs:135-138.
        var alpha_vals = List[Float32]()
        alpha_vals.append(nl.alpha)
        var alpha_sh = List[Int]()          # shape [] = empty dims (scalar)
        names.append(nl.prefix + ".alpha")
        tensors.append(ArcPointer(Tensor.from_host(alpha_vals^, alpha_sh^, STDtype.F32, ctx)))
```
CAVEAT for the lead: confirm `Tensor.from_host` + the safetensors writer accept
a **rank-0** shape (`shape:[]`). The reader's `_read_scalar_alpha`
(`lora.mojo:357`) expects shape `[]`. If rank-0 tensors aren't supported by the
Mojo `Tensor`/writer yet, fall back to shape `[1]` — `_read_scalar_alpha` reads
element 0 either way (verify against `lora.mojo:357-380`). This is the one spot
most likely to fail compile; flagged as hypothesis.

If P1 is judged out-of-scope for this round, the **minimum** fix is a loud
doc/runtime guard: `save_lora_peft` should refuse (or warn) when the caller's
`alpha != rank` and no `.alpha` is being written, so a non-unit ratio can never
silently ship. That guard is itself a hypothesis.

### A.4 NON-issue (checked, no patch): loop.mojo `param.N` checkpoint

`training/loop.mojo:183-204` (`save_checkpoint`) writes generic
`param.<i>`/`adam_m.<i>`/`adam_v.<i>`/`__meta__` keys, NOT PEFT LoRA keys. It is
the **resumable optimizer-state** half and is self-consistent with its own
`load_checkpoint` (`loop.mojo:226-258`) — F32 masters reload byte-for-byte.
This is correct and intentional (`lora_save.mojo:1-14` documents the
two-file split). No round-trip bug; do NOT "fix" it to emit LoRA keys.

---

## B. Config-key → TrainConfig field map (transcription check)

Source JSONs: `/home/alex/OneTrainer/configs/klein9b_loss_compare.json` and
`klein4b_benchmark.json`. Target: hardcoded `klein_9b()` / `klein_4b()` in
`serenitymojo/models/klein/config.mojo` and the AdamW constants in
`training/train_step.mojo`.

### B.1 klein9b map

| JSON key | JSON value | json line (klein9b) | TrainConfig field / const | mojo ref | OK? |
|---|---|---|---|---|---|
| `learning_rate` | `4e-4` | :20 | `lr` = `4.0e-4` | `config.mojo:20` | ✅ |
| `lora_rank` | `16` | :23 | `lora_rank` = `16` | `config.mojo:20` | ✅ |
| `lora_alpha` | `16` | :24 | `lora_alpha` = `16.0` | `config.mojo:20` | ✅ |
| `optimizer.optimizer` | `ADAMW` | :33 | (AdamW path hardcoded) | `train_step.mojo:226-237` | ✅ |
| `optimizer.beta1` | `0.9` | :34 | `b1` = `0.9` | `train_step.mojo:229` | ✅ |
| `optimizer.beta2` | `0.999` | :35 | `b2` = `0.999` | `train_step.mojo:230` | ✅ |
| `optimizer.eps` | `1e-8` | :36 | `aeps` = `1.0e-8` | `train_step.mojo:231` | ✅ (AdamW eps; see B.3) |
| `optimizer.weight_decay` | `0.01` | :37 | `wd` = `0.01` | `train_step.mojo:232` | ✅ |
| `model_type` | `FLUX_2` | :3 | (Klein==FLUX.2; dims in config) | `config.mojo:17-21` | ✅ |
| `learning_rate_scheduler` | `CONSTANT` | :55 | constant LR (no decay in `_lora_adamw`) | `train_step.mojo:226` | ✅ |
| `epochs` | `2` | :18 | — (synthetic scaffold; not wired) | — | ⚠️ B.4 |
| `learning_rate_warmup_steps` | `100` | :21 | — (not wired; scaffold) | — | ⚠️ B.4 |
| `clip_grad_norm` | `1.0` | (:69) | — (not applied in `_lora_adamw`) | — | ⚠️ B.4 |
| (no `timestep_shift` key) | — | — | `timestep_shift` = `1.8` | `config.mojo:20` | see B.2 |

### B.2 `timestep_shift = 1.8` is NOT from the JSON — verified honest

Neither config has a `timestep_shift` / `shift` key (only
`timestep_distribution: LOGIT_NORMAL`, klein9b :63). The `1.8` is the
project-validated value from memory (`feedback_klein9b_timestep_shift_1.8`),
and `config.mojo:8` documents it as "project-validated", not config-sourced.
**Not a mis-map** — correctly sourced and labeled. klein4b uses `1.0`
(`config.mojo:29`), also not in JSON; acceptable as the 4B default.

### B.3 `cfg.eps = 1e-6` is a DIFFERENT eps — verified NOT a bug

`TrainConfig.eps` (`config.mojo:20` = `1.0e-6`) is the **RMSNorm/layernorm
epsilon**, threaded into the dit-block forward/backward at
`train_step.mojo:283` and `:291` — NOT the AdamW eps. The AdamW eps is a
separate hardcoded `1.0e-8` (`train_step.mojo:231`) and DOES match
`optimizer.eps` (json :36). So there is **no eps collision**; the two 1e-6 /
1e-8 values are correctly distinct. (A reviewer skimming would mistake this for
a config drift — it is not.) No patch.

### B.4 DIVERGENCE D2 (informational, not a wrong-default) — nested-optimizer
trap is avoided, but several JSON keys are dropped because the scaffold is
synthetic

The nested `optimizer.{beta1,beta2,eps,weight_decay}` object — the trap the
task called out — is transcribed CORRECTLY (B.1). But `TrainConfig` does not
*carry* the betas/wd/optimizer-eps at all; they live as hardcoded constants in
`train_step.mojo:229-232`. And `epochs`, `learning_rate_warmup_steps`,
`clip_grad_norm` are not represented anywhere. For the current synthetic
scaffold (`models/klein/train.mojo:23` calls `run_synthetic`) this is fine —
no real loop consumes them yet. **But the values being right today is by
luck of the hardcode, not by mapping**: change the JSON and Mojo won't follow.

This is a structural fidelity gap, not a wrong value. Two options for the lead
(both hypotheses):
- **Recommended (defer):** leave hardcoded for the scaffold; when the real
  `run` loop lands (GAP G1), extend `TrainConfig` with
  `beta1, beta2, adam_eps, weight_decay, warmup_steps, grad_clip` and have the
  per-model constructor pass the JSON-sourced values, with `train_step.mojo`
  reading `cfg.*` instead of the literals. No patch this round.
- **Now (if the lead wants the constants centralized):** move the four AdamW
  literals from `train_step.mojo:229-232` onto `TrainConfig` so the single
  source of truth is `config.mojo`. Larger change; defer unless requested.

No incorrect-default was found — flagging the **brittleness**, per the task's
"does NOT silently drop or mis-default" clause: it drops `epochs`,
`warmup_steps`, `clip_grad_norm` (D2), but only because no loop reads them yet.

---

## C. safetensors writer fidelity (supporting check) — FAITHFUL

`io/safetensors_writer.mojo` is a clean inverse of `io/safetensors.mojo`:
- 8-byte LE header length (`:220-232`) ↔ reader `mmap.rs:175-178` (cited).
- compact JSON header, insertion order, contiguous `data_offsets` (`:129-169`).
- raw D2H byte copy, no F32 round-trip → BF16 stays BF16 (`:257-278`).
No divergence. The `_tensor_offsets` sentinel (`:111-126`) gives
`size = data_offsets[1]-data_offsets[0]` exactly as the reader expects.

---

## D. Validation-sampler LoRA-apply fidelity — FORWARD CONTRACT (file not yet written)

`training/validation_sampler.mojo` does not exist (only referenced at
`lora_save.mojo:13`). The "L2P trap" cannot be audited. When the builder writes
it, it MUST satisfy this contract or validation is meaningless:

1. **Apply to the SAME projections training updates.** Training puts its single
   trained adapter on the block-input projection via `_lora_fwd`
   (`train_step.mojo:164-183`), `delta = scale·(x@Aᵀ)@Bᵀ`. The sampler's
   LoRA-apply must hit the identical module set. The reference inference apply
   is the merge-at-load path in `pipeline/klein9b_lora_smoke.mojo:50-61`
   (`LoraSet.merge_into_indexed`), which routes split `to_q/to_k/to_v` into the
   fused `qkv.weight` RowRanges (`lora.mojo:_map_klein_split_qkv:296-340`).
   If the sampler instead applies to bare `to_q.weight` (the generic
   `_map_diffusion_model` path, `lora.mojo:282-292`), it will **silently
   no-op** against fused Klein — the exact trap flagged in
   `klein9b_lora_smoke.mojo:14-25` and `lora_save.mojo`'s split-qkv note.
2. **Use the same scale convention.** Sampler merge must pass
   `multiplier = alpha/rank` (here 1.0 for klein9b) so the merged delta equals
   the trained `scale·(B@A)`; see `lora.mojo:_module_scale:462-486`. If P1
   (write `.alpha`) lands, the sampler can rely on the file's alpha and pass
   `multiplier=1.0`.
3. **No L2P divergence:** the L2P trap (memory
   `project_l2p_no_subject_convergence_2026-05-30`) is "LoRA doesn't imprint at
   inference." The sampler must run a WITH-vs-WITHOUT pixel-diff smoke (nonzero
   diff ⇒ adapter actually applied) before any validation number is trusted.

No patch (nothing to patch); this is the acceptance contract for the sampler's
author.

---

## Summary of divergences (priority order)

| ID | Severity | File:line | Issue | Patch |
|---|---|---|---|---|
| D1 | medium (latent) | `lora_save.mojo:31-38,118-121` vs `eridiffusion-core/lora.rs:138` | save omits `<prefix>.alpha`; silent 2× over-apply if `alpha != rank` | P1a/P1b (HYPOTHESIS; verify rank-0 shape support) |
| D2 | low (brittleness) | `train_config.mojo`, `train_step.mojo:229-232` | AdamW betas/wd hardcoded, not on TrainConfig; `epochs`/`warmup`/`clip_grad_norm` dropped (scaffold-only) | defer to real-loop landing; no patch this round |

**Non-issues explicitly cleared (do not "fix"):** orientation round-trip
(A.2, correct), `param.N` checkpoint keys (A.4, intentional), `cfg.eps=1e-6` vs
AdamW `1e-8` (B.3, two distinct epsilons), `timestep_shift=1.8`
(B.2, project-validated not config-sourced), safetensors writer (C).

All patches are HYPOTHESES (Tenet 4) — the LEAD compiles + runs to confirm.
