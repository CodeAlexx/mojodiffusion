# SKEPTIC FINDINGS — BlockLoader (serenitymojo/offload/block_loader.mojo)

**Date:** 2026-05-25
**Reviewer stance:** adversarial. Assumed the builder's report ("15 tensors byte-match,
VRAM reclaimed exactly, reload OK") was a lie until proven with my own probes.
**Scope:** `serenitymojo/offload/block_loader.mojo` (+ `offload_smoke.mojo`), trusting
`io/sharded.mojo` and `tensor.mojo` (byte-parity-verified per task).
**Verdict up front:** the mechanism is **genuinely correct** — VRAM is bounded with zero
drift across 30 blocks, prefix matching is exact, Arc lifetime is real, reload is robust.
I found **one CORRECT-BUT-FRAGILE API divergence from Rust** (trailing-dot ownership) and
**one latent CORRECT-BUT-FRAGILE dtype divergence** (no BF16 coercion). No BLOCKERs.

Required-reading receipts (one specific fact each):
1. `block_loader.mojo:112` — `block[nm] = ArcPointer(t^)`: the device `Tensor` is moved
   into a fresh `ArcPointer`; the Dict value type is `ArcPointer[Tensor]` (`comptime Block`
   line 54). There is NO explicit `unload_block(self,...)` method on the struct — unload is
   the free-function `unload_block(var block)` at line 125 that consumes the Dict.
2. `inference-flame/src/offload.rs:58` — `let file_prefix = format!("{}{prefix}.", self.key_prefix);`
   Rust **manufactures the trailing dot itself** and then `key.starts_with(&file_prefix)`
   (line 60). `unload_block` = `self.cache.clear()` (line 86). Rust also coerces every
   non-BF16 tensor to BF16 on load (lines 68–72).
3. `io/sharded.mojo:437` — `names()` iterates `self.name_to_shard.items()` (order
   unspecified). `tensor_bytes`/`tensor_view` reach the owning shard via
   `self.shards[idx][]` (lines 472, 483); `shards` is a public field
   `List[ArcPointer[SafeTensors]]` (line 361) — which is exactly how `block_loader.mojo:96`
   reaches `prefetch_tensor`.

---

## VRAM trace across N blocks (P1 — the headline leak test)

Hammer: load→unload **all 30** transformer layers sequentially (`layers.{i}.` for i in 0..29),
measuring free VRAM (`ctx.get_memory_info()[0]`) at each load and each unload.
Probe: `serenitymojo/offload/skeptic_probe.mojo` (P1). Numbers (MB):

```
baseline free (no block loaded): 21146
idx  free_after_load  loaded_delta  free_after_unload  residual_vs_baseline  tensors
 0       20801            345            21146                 0                15
 1       20801            345            21146                 0                15
 ...  (identical every row) ...
29       20801            345            21146                 0                15
final free MB: 21146   baseline: 21146   net drift MB: 0   worst residual MB: 0
```

**BOUNDED, not creeping.** Every load consumes exactly 345 MB and every unload reclaims
exactly back to 21146 MB. Net drift after 30 cycles = **0 MB**. Worst-case residual after
any unload = **0 MB**. The `Tensor`/`DeviceBuffer` drop chain (synthesized `Tensor`
destructor → `DeviceBuffer.__del__`) and the `ArcPointer` refcount-to-0 free are both real.
No leak.

Stress (P4): 50 load/unload cycles (cycling layers 0..29), all counts == 15, **net drift 0 MB**,
no failure on the io re-open path.

---

## Findings

### F1 — Trailing-dot ownership diverges from Rust; caller-supplied dot is a footgun
- **Where:** `block_loader.mojo:98-113` (`load_block`), docstring lines 103-106.
- **What:** Rust `load_block(prefix)` builds `format!("{}{prefix}.", key_prefix)` — it
  **appends the trailing dot internally**, so a Rust caller passes `"layers.1"`. The Mojo
  `load_block(prefix, ctx)` does a raw `nm.startswith(prefix)` with **no dot appended**, so
  the Mojo caller must pass `"layers.1."` themselves. The two APIs have the *same name and
  the same role* but a **different prefix contract**.
- **Expected (if it truly "mirrors Rust one-for-one"):** a caller migrating Rust call sites
  would pass `"layers.1"` and expect the loader to add the dot. In Mojo that silently returns
  the WRONG set.
- **Evidence (Python ground-truth over the real 521-name index):**
  - Correct (Mojo, with dot) `load_block("layers.1.")` → **15** tensors, layer 1 only. ✅
    (confirmed live in probe P2: `layers.1. count: 15  captured only layer 1: True`).
  - Footgun (caller forgets dot) `load_block("layers.1")` → **165 tensors spanning layers
    [1,10,11,12,13,14,15,16,17,18,19]** — an 11× VRAM blowup (≈3.8 GB instead of 345 MB) and
    a wrong-weights block. `"layers.2"` → 165 tensors spanning [2,20..29]. Same trap.
  - `"layers.0"` (no dot) happens to be SAFE (15, layer 0 only) only because there is no
    `layers.0X` layer — pure luck of the index, not a property of the code.
- **Why it matters:** the module header (lines 7-13) and the comptime alias claim it mirrors
  Rust "one-for-one". It does *not* for the prefix contract. The smoke test only ever passes
  dotted prefixes, so it cannot catch a caller who mirrors the Rust call convention.
- **Severity:** **CORRECT-BUT-FRAGILE.** Behavior is correct *and documented* ("Callers pass
  the trailing dot", line 104) when used as documented. But it is an undefended, silent,
  high-blast-radius divergence from the API it claims to mirror. The robust fix is to make
  `load_block` append the dot like Rust (and have callers pass `"layers.1"`), OR rename to
  make the literal-prefix contract unmistakable (e.g. `load_block_literal`). At minimum the
  "mirrors Rust one-for-one" claim should be corrected to "mirrors Rust except the caller
  owns the trailing dot."

### F2 — No BF16 dtype coercion (Rust coerces; latent divergence, inert on Z-Image)
- **Where:** `block_loader.mojo:110-112` vs `offload.rs:68-72`.
- **What:** Rust `load_block` coerces every non-BF16 tensor to BF16 (`val.to_dtype(BF16)`).
  Mojo `load_block` uses `Tensor.from_view`, which **preserves the on-disk dtype** (tensor.mojo
  `from_view` validates the dtype is a supported compute dtype but does not convert).
- **Expected:** if a checkpoint stored a layer in F32 or F16, Rust would hand the model a
  BF16 tensor; Mojo would hand it F32/F16 — a dtype mismatch downstream.
- **Evidence:** I parsed all 521 transformer tensor headers across both shards —
  **dtype histogram = {BF16: 521}** (100% BF16). So for Z-Image Rust's coercion is a no-op
  (it takes the `val` identity branch) and Mojo's preservation is byte-identical. **Not active
  for this model.** Confirmed live: held tensor `layers.0.adaLN_modulation.0.bias` is BF16
  [15360] = 30720 bytes, byte-matches disk (probe2).
- **Severity:** **CORRECT-BUT-FRAGILE.** Inert today, but a mixed-dtype checkpoint would
  diverge from Rust silently. Worth a one-line note in the docstring or an assert that the
  view dtype is BF16.

### F3 — `unload_block` is a free function, not a method; double-call is compiler-prevented (not a bug)
- **Where:** `block_loader.mojo:125-131`.
- **What I tried to break:** double-free. You cannot call `unload_block(b^)` twice — the `^`
  transfer consumes `b`, so a literal double-unload is a **compile error**, not a runtime
  double-free. Re-loading the same prefix after an unload and unloading again works cleanly
  (probe2: `re-load layers.2. after unload -> count: 15 (no double-free crash)`).
- **Severity:** **CORRECT.** The move-semantics actually make the double-free footgun
  impossible at compile time — stronger than Rust's `cache.clear()` (which is idempotent but
  re-callable). Note for callers: the only "unload" is dropping the owned Dict; there is no
  idempotent method to call twice, by design.

---

## Clean checks (probed, genuinely correct)

| # | Hammer | Result | Evidence |
|---|--------|--------|----------|
| C1 | VRAM bounded across 30 blocks | **0 MB net drift, 0 worst residual**, every load 345 MB | probe P1 trace above |
| C2 | 50-cycle reload stress | all counts 15, **0 MB drift**, no failure | probe P4 |
| C3 | Per-prefix count vs Python ground truth | all 30 layers == **15** (Python: all 15) | probe P2 + Python |
| C4 | `layers.1.` ≠ `layers.1{0..9}.` (trailing-dot) | captures **only layer 1** (15 tensors) | probe P2 `captured only layer 1: True` + Python: indices captured = [1] |
| C5 | No-match prefix → empty Dict, no crash | `layers.999.`→0, `zzznotaprefix`→0 | probe P2 |
| C6 | Substring-not-from-start | `"attn"` → **0** (startswith only; Python: 0 names start with attn) | probe P2 |
| C7 | Empty prefix matches all | `block_count_for("")` = **521** = full index | probe P2 + Python total=521 |
| C8 | Arc keep-alive: hold 1 tensor, unload block | held tensor **byte-identical to disk before AND after** unload | probe2 `HELD-TENSOR-INTACT-AFTER-UNLOAD: True` |
| C9 | Double-load same prefix (two live blocks) | two **independent** 345 MB copies, both reclaimed to baseline | probe P3 |
| C10 | Empty-block unload | no crash | probe2 |
| C11 | Total tensor count | 521 (matches task's "521-name set") | Python |
| C12 | Layer count | **30** layers (0..29), each 15 tensors | Python (note: task said "~28"; actual is 30) |

C8 is the decisive Arc test: I copied one `ArcPointer` out of the block (refcount bump),
unloaded the whole block, then read the held tensor's bytes back from the device and compared
to disk — **byte-identical both before and after the unload**. A silent use-after-free that
preserved `nbytes` metadata would have shown corrupt bytes; it did not. The Arc refcount is real.

---

## Couldn't verify / out of scope

- **`prefetch_block` actually warming pages:** the call path runs without error (smoke +
  reachable via `self.sharded.shands[idx][].prefetch_tensor`), but `MADV_WILLNEED` is advisory
  and its *effect* (page-cache residency) is not observable without `/proc` mincore tooling.
  Verified only that it does not crash and reaches the owning shard for every block tensor.
- **`names()` ordering:** `Dict.items()` order is unspecified in both `sharded.mojo:437` and
  `safetensors.mojo:217`. `load_block` does not depend on order (it filters), so this is
  harmless here — but any future caller that assumes a stable tensor order across reloads
  would be wrong. Not a defect in `load_block`; flagged for downstream awareness.
- **Mixed-dtype checkpoint behavior (F2):** could not exercise — Z-Image transformer is 100%
  BF16. The divergence is real in code but unverifiable against a real F32/F16 shard here.
- **`BlockLoader` Movable-not-Copyable enforcement / origin escape:** trusted per task
  (`tensor.mojo`/`sharded.mojo` are byte-parity-verified); I did not re-derive the origin
  proofs. `from_view`'s "copies, does not alias mmap" claim is consistent with C8 (held tensor
  survived after the block — and would survive after the loader, since it owns its own VRAM).

---

## Bottom line

The builder's report is **not a lie**. I tried to break it on leak, prefix edges, Arc
lifetime, double-free, and reload — all held. The streaming mechanism is sound and
production-usable for Z-Image as-is.

The two things to fix are *fragility*, not *bugs*:
1. **F1 (trailing dot):** the API silently diverges from the Rust it claims to mirror; a
   caller who passes a dotless prefix gets 11× the tensors with no error. Either append the
   dot internally (true Rust parity) or rename to advertise the literal-prefix contract.
2. **F2 (BF16 coercion):** harmless on this all-BF16 model, but unlike Rust it will not
   normalize a mixed-dtype checkpoint. Add an assert or a docstring note.

Probes left in tree (NOT scope code, safe to delete):
`serenitymojo/offload/skeptic_probe.mojo`, `serenitymojo/offload/skeptic_probe2.mojo`.
