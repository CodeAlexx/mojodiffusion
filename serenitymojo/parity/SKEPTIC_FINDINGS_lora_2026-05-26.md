# SKEPTIC FINDINGS — LoRA merge-at-load port (lora.mojo) — 2026-05-26

Reviewer: fresh-eyes skeptic (not the builder). Assumed the port lies; hunted line-by-line
against `inference-flame/src/lora.rs` (canonical runtime path), `src/lora_merge.rs`
(the in-place fuse path the port claims to mirror), `src/models/lora_loader.rs`, and
`src/bin/klein_lora_infer.rs`. **CODE-ONLY — GPU wedged. `mojo build` only, did NOT run.**

## Compile honesty (re-run, EXIT read)
```
pixi run mojo build -I . -Xlinker -lm serenitymojo/lora_probe.mojo            -o /tmp/sklora    → EXIT=0
pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/klein9b_lora_smoke.mojo -o /tmp/sklora2 → EXIT=0
```
Both compile clean. No silent "0 errors because nothing compiled" — real codegen, real binaries.

## Op/API parity (verified, NOT bugs)
- **B@A orientation is CORRECT.** `lora.mojo:386-387` does `a_t = transpose(a,0,1)` then
  `linear(b, a_t)`. `ops/linear.mojo:100` is `y = x @ wᵀ`, so `linear(b, a_t) = b @ (Aᵀ)ᵀ = B@A`,
  shapes `[out,rank] @ ([in,rank])ᵀ = [out,rank]@[rank,in] = [out,in]`. Transpose applied to **A**,
  delta is **ADDED** (`add(base_w, delta)` in every slot branch, never subtracted/overwritten). ✅
- **Slot RowRange offsets match Rust.** `_map_zimage_trainer` (lora.mojo:206-224) emits
  `(0,3840)/(3840,3840)/(7680→2*3840,3840)` for to_q/k/v — bit-identical to
  `map_prefix_zimage_trainer` (lora_merge.rs:373-391, lora.rs:688-705). Klein-4B
  Rows(9216)/Cols(3072) constants match (lora.mojo:66-67 vs lora_merge.rs:46-47). ✅
- **`_apply_slot` head|mid|tail rejoin** (lora.mojo:535-555) matches lora_merge.rs:611-637
  including the head-only / tail-only / both-empty degenerate cases. ✅
- Op signatures (`linear`/`transpose`/`slice`/`concat`/`add`/`mul_scalar`/`cast_tensor`),
  `SafeTensors.{open,names,tensor_info,tensor_bytes,tensors}`, `Tensor.{from_view,to_host,
  dtype,shape}`, `from_parts` origin-inference idiom — all line up with the callsites. No
  foundation op reimplemented inside lora.mojo. ✅
- dtype cast handled: A cast to B's dtype before matmul (lora.mojo:382-385); delta cast to
  base dtype before add (lora.mojo:389-391). Matches lora_merge.rs:560-564. ✅
- `Dict[String,ArcPointer[Tensor]]` mutate-in-place uses `base[k] = ArcPointer(merged^)` and
  `[]`-deref to borrow — compiles, no use-after-move. ✅

---

## FINDINGS

### F1 — BLOCKER: train_klein split-QKV no-ops against the fused Klein base (48/144 modules dropped)
**file:** `serenitymojo/lora.mojo:237-252` (`_map_diffusion_model`), wired by
`klein9b_lora_smoke.mojo:36,52-60`.

**Evidence (real on-disk header, the file the smoke actually points at):**
`/home/alex/EriDiffusion/EriDiffusion-v2/output/klein_lr3e4_const_b1/klein_lora_step200.safetensors`
- 288 real LoRA weight tensors (+576 `__opt__/adamw` moments), all suffix `.lora_A.weight`/`.lora_B.weight`,
  **no `diffusion_model.` prefix** → `_detect_format` → `FMT_DIFFUSION_MODEL` (matches Rust).
- Modules (144 prefixes): per double-block `img_attn.{to_q,to_k,to_v,proj}`,
  `txt_attn.{to_q,to_k,to_v,proj}`, `{img,txt}_mlp.{0,2}`; per single-block `linear1`,`linear2`.
- Klein9B base (`models/dit/klein_dit.mojo:60,66,109,115`) stores **FUSED**
  `double_blocks.<i>.{img,txt}_attn.qkv.weight`. There is **no `...to_q.weight`/`to_k.weight`/`to_v.weight`**.

**What's wrong:** `_map_diffusion_model` only strips `diffusion_model.`/`.default` and appends `.weight`.
For `double_blocks.0.img_attn.to_q` it produces `double_blocks.0.img_attn.to_q.weight`, which is **absent
from the base** → dropped at `merge_into_indexed`'s `m.base_key not in name_to_idx` (lora.mojo:477).
**All 48 attention-QKV modules (the highest-signal LoRA targets) silently no-op.** Only proj/mlp/single
(96 modules) merge. A likeness/style LoRA loses its entire attention contribution.

**Why the Rust does NOT have this bug:** the canonical `map_prefix_diffusion_model` in **lora.rs:730-750**
(NOT lora_merge.rs) has explicit RowRange special-cases for `.attention.to_q/.to_k/.to_v/.to_out.0` →
fused `attention.qkv.weight`. The Mojo port mirrored the **older, thinner** `lora_merge.rs:143-149`
mapper (strip+append only) plus only the PEFT `.weight.weight` guard — it missed the split→fused RowRange
logic that lora.rs added. Note: lora.rs's special-case is keyed on `.attention.to_q` (Z-Image naming),
NOT `.img_attn.to_q` (Klein naming), so **even the Rust DiffusionModel path would no-op this exact Klein
file** — but Rust's intended Klein path is the **KleinTrainer** format via bare `.lora_A` (no `.weight`),
which routes `img_attn.qkv_proj`→fused `qkv.weight` Full-overlay. This train_klein file uses
`.lora_A.weight` + split `to_q/k/v`, so it matches neither clean path on either side.

**Honesty check:** the builder DID document this in the smoke header (klein9b_lora_smoke.mojo:14-25) and
in the `merge_into_indexed` docstring. So it is **documented, not silently wrong** — but it is documented
as "whether targets resolve depends on the LoRA's key layout", which undersells it: for the *one real
train_klein file the smoke names*, 1/3 of modules (all attention) no-op. That is a BLOCKER for producing a
LoRA that actually does anything to attention on Klein, not an "acceptable-with-a-flag".

**Minimal fix:** add a Klein split→fused RowRange branch to `_map_diffusion_model` (or a dedicated
`_map_klein_split` consulted before the generic append), analogous to the Z-Image block but with Klein
dims. Klein9B fused qkv is `[3*inner_dim, inner_dim] = [12288, 4096]` (inner_dim=4096, klein_dit.mojo:43),
so:
```
.img_attn.to_q → <rest>.img_attn.qkv.weight  RowRange(start=0,        len=4096)
.img_attn.to_k → <rest>.img_attn.qkv.weight  RowRange(start=4096,     len=4096)
.img_attn.to_v → <rest>.img_attn.qkv.weight  RowRange(start=2*4096,   len=4096)
.txt_attn.to_q/k/v → <rest>.txt_attn.qkv.weight  (same offsets)
```
Verify against the base qkv first dim at merge time (don't hardcode 4096 blindly — gate on
`base.shape()[0] == 3*len`). The slot machinery (`SLOT_ROWRANGE`) already exists and is tested in
lora_probe; only the mapper is missing the branch.

**Severity: BLOCKER.**

---

### F2 — HIGH (correctness divergence): no-`.alpha` scale uses file-level alpha/rank, not per-module rank
**file:** `serenitymojo/lora.mojo:356-369` (`_module_scale`) + `merge_into{,_indexed}` default_scale
(lora.mojo:430, 471).

**What's wrong:** when a module has no `<prefix>.alpha`, the Mojo falls back to
`default_scale = (alpha / rank) * multiplier` using the **caller-passed file-level `alpha`/`rank`**
(16/16 in the smoke). The canonical Rust `LoraStack::load` (lora.rs:273-282) does NOT use any file-level
alpha/rank — it derives **per-module** `module_rank = a.shape()[0]` and, when `.alpha` is absent, sets
`alpha_value = module_rank` so `scale = (module_rank/module_rank)*multiplier = multiplier`.

**Why it matters:** the binary `klein_lora_infer.rs:253-261` explicitly prints
*"--alpha/--rank are ignored when LoRA file ships per-module .alpha tensors"* and relies on the
per-module-rank derivation. The Mojo `merge_into` signature *requires* `alpha`+`rank` and uses them as the
no-`.alpha` default. For the named train_klein file (no `.alpha`, every module rank=16, smoke passes
16/16) the two agree at scale=1.0 — so **no visible bug for this file**. But:
- If a caller passes `rank=8` for a rank-16 file → Mojo computes `16/8=2.0` (2× too strong); Rust computes
  `16/16=1.0`. Divergence is real and silent.
- A mixed-rank LoRA (different rank per module) cannot be expressed by one file-level `rank` arg; Rust
  handles it per module, Mojo cannot.

This is the exact class of "wrong scale = silently too-strong/weak" the task flagged. The per-module
`.alpha` *override* path IS correct (lora.mojo:362-368 reads `a_info.shape[0]` for rank and the scalar
alpha, matching lora.rs:274-281); only the **absent-`.alpha` fallback** diverges.

**Minimal fix:** drop the file-level `alpha`/`rank` defaulting. In `_module_scale`, when `.alpha` is
absent, use `module_rank = a_info.shape[0]` and `alpha = module_rank` → `scale = multiplier` (mirror
lora.rs:280). Keep `multiplier` as the only caller knob; demote `alpha`/`rank` params to ignored/removed
(or assert they equal the per-module values). This also makes `merge_into` robust to rank-mismatched calls.

**Severity: HIGH** (latent mis-scale; benign only because the smoke happens to pass matching 16/16).

---

### F3 — MEDIUM: per-module `.alpha` read for ALL formats — matches lora.rs, DIVERGES from lora_merge.rs
**file:** `serenitymojo/lora.mojo:356-369`, `merge_into` docstring (lora.mojo:418-424).

Not a bug — a deliberate (and correct) choice — but flagging because the module header (lora.mojo:14)
claims to mirror `lora_merge.rs:32-37, 520-550`, and **lora_merge.rs reads `.alpha` only for kohya**
(lora_merge.rs:504 `if matches!(format, LoraFormat::KohyaSdxl)`). The Mojo reads it for every format,
which matches the **corrected** lora.rs:60-64/273-282 (where the kohya-only behavior is explicitly called
"a real bug … 32× too strong"). So the port followed the right reference for THIS, but its own header
cites the wrong one. **Fix:** update the lora.mojo header to cite lora.rs:60-64/273-282 as the authority
for the per-module-alpha-all-formats behavior, not lora_merge.rs. Severity: LOW/style (doc accuracy).

---

### F4 — LOW: optimizer-state tensors (`__opt__/adamw/*`) iterated but unfiltered
**file:** `serenitymojo/lora.mojo:319-332` (`LoraSet.load` iterates all `names()`).

The real train_klein file carries 576 `__opt__/adamw/m|v/...lora_A.weight` moment tensors alongside the
288 real weights. `LoraSet.load` strips `.lora_A.weight` off `__opt__/adamw/m/double_blocks.0.img_attn.to_q`
and resolves it to `__opt__/adamw/m/double_blocks.0.img_attn.to_q.weight`, appending a junk mapping. These
are dropped at merge (`base_key not in base/name_to_idx`), so **benign** — and Rust has the identical
non-filtering (load_file pulls every key; mapped junk dropped at base lookup). But it wastes ~576 mapping
entries and inflates `num_mappings()` reporting, which could mislead a "merged N/M" sanity check.
**Fix (optional):** skip `n.startswith("__opt__")` (or any non-model namespace) at the top of the load
loop. Severity: LOW.

---

### F5 — LOW (style/robustness): `_strip_suffix` rejects an exactly-equal string
**file:** `serenitymojo/lora.mojo:153-159`.

`_strip_suffix` returns `""` unless `byte_length() > suf.byte_length()` (strictly greater). For real LoRA
prefixes this is fine (prefix is never exactly the suffix), and `""` is the sentinel for no-match. But it
means a hypothetical key equal to the suffix returns no-match rather than empty-stem; the Rust
`strip_suffix` would return `Some("")`. No real key hits this. Severity: LOW/style — leave as is, but
the `> ` (vs `>=`) is load-bearing for the sentinel and worth a one-line comment.

---

## Cross-checks that PASSED (anti-regression record)
- Format detection order (kohya → DiffusionModel → Z-Image → KleinTrainer) matches detect_format
  (lora_merge.rs:91-111 / lora.rs). Real file → DiffusionModel, confirmed against header. ✅
- kohya `.alpha` scalar read via `to_host` upcast handles F16/BF16/F32 (`_read_scalar_alpha`,
  lora.mojo:267-278). Matches lora_merge.rs:509 `to_dtype(F32).to_vec()`. ✅
- kohya text-encoder skip (`lora_te1_/lora_te2_`) present (lora.mojo:325-328) — matches lora.rs:243-246.
  (Note: full kohya diffusers→LDM rewriter is intentionally NOT ported; documented at lora.mojo:256-258.
  SDXL UNet not in this stack, so acceptable.) ✅
- 4D conv-LoRA rejection (`_pair_present`, lora.mojo:401-402) matches lora.rs:289-294. ✅
- `rank <= 0` guard (lora.mojo:428,470) matches lora_merge.rs:419 / the InvalidInput path. ✅

---

## VERDICT

**BLOCKERS: 1** (F1 — attention-QKV split→fused no-op; 48/144 modules of the real train_klein file are
silently dropped, killing the LoRA's attention contribution on Klein).

**HIGH: 1** (F2 — no-`.alpha` scale fallback uses file-level alpha/rank instead of per-module rank;
silent mis-scale on any rank-mismatched or mixed-rank call; benign only for the 16/16 smoke).

Plus 1 MEDIUM doc-citation (F3), 2 LOW (F4 opt-state, F5 sentinel style).

The B@A transpose and the slot/RowRange math — the other two "likeliest real bugs" — are **correct**.
The split-vs-fused no-op IS the real bite, and it is bigger than the header documents it to be: for the
exact file the smoke names, it is not "depends on layout", it is "all attention no-ops". Fix F1 before any
parity/smoke run, and fix F2 to make scale match the canonical lora.rs semantics rather than the
superseded lora_merge.rs ones.
