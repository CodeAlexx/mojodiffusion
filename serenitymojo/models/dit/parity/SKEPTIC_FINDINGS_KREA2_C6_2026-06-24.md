# Skeptic findings — krea2 inference chunk 6 (TextFusionTransformer)

Reviewer: fresh-eyes adversarial pass (did NOT write the port).
Date: 2026-06-24.
Scope: `krea2_mha`, `krea2_text_fusion_block`, `Krea2TextFusionWeights`,
`build_krea2_text_mask`, `_krea2_text_mask_kernel`, `krea2_text_fusion` in
`serenitymojo/models/dit/krea2_dit.mojo` (lines 880-1118) + the parity probe
`krea2_txtfusion_parity_probe.mojo` and generator `gen_krea2_txtfusion.py`.
Reference read line-by-line: `ai-toolkit/.../krea2/src/mmdit.py`
(`TextFusionBlock` 245-264, `TextFusionTransformer` 267-309, `attention` 51-63,
`_mask` 66-68) + `src/pipeline.py` (`pad_text_features` 32-58, `predict_velocity`
93-130, sampler 185-260) + `src/text_encoder.py` (`encode_krea_prompt` 38-84).

---

## VERDICT ON THE MASK DECISION: **REJECT the no-mask gate. The gate is GREEN ON A LIE.**

The probe gates `krea2_text_fusion` with `mask=None` (refiner runs `sdpa_nomask`)
and its comment asserts the oracle "was run with mask=_mask(keep) over the
**all-ones keep**, which cuDNN renders identically to no-mask." **That is false for
the committed oracle.** The committed generator (`gen_krea2_txtfusion.py:58-61`)
sets `keep[0, LT-NPAD:] = 0.0` → **6 real padded tokens, not all-ones.** I
regenerated the oracle from the committed generator and the probe **FAILS**:

```
$ pixi run mojo run -I . .../krea2_txtfusion_parity_probe.mojo
krea2_text_fusion parity: ParityResult(cos=0.9987619626226566, max_abs=0.14453125, n=61440, FAIL)   EXIT=0
```

The probe only ever "passed" because a **stale all-ones oracle file** (out norm
169.51, keep all-ones) was sitting on disk from an earlier generator version that
did NOT match the committed generator (6-pad, out norm 169.67). mtimes confirm the
mismatch was latent: the on-disk oracle predated the current `keep` edit.

### Independent torch measurements (all reproduced myself, not taken on faith)

1. **All-ones additive mask IS softmax-invariant — TRUE, but irrelevant.**
   cuDNN(all-ones float mask) vs cuDNN(none) → **cos 1.0**; MATH-f32 → cos
   0.99999994. A uniform +1 on every column is a per-row constant shift that
   cancels in softmax. Confirmed mathematically and empirically. *But the oracle
   is not all-ones.*

2. **The +1/+0 additive mask is NOT softmax-invariant once ANY pad exists.**
   Single-attention, 6-pad: MATH-f32(6-pad) vs MATH-f32(none) → **cos 0.9670**.
   The mask gives kept columns +1 and padded columns +0 — a NON-uniform per-row
   shift — so it genuinely changes the distribution. The builder's "uniform shift
   cancels" reasoning silently assumed all-ones; it does not hold for the dumped
   oracle. (The full-transformer effect is smaller, ~0.9988, because only the 2
   refiner blocks are masked and the +1 bias is small vs the score magnitudes —
   but it is real and below the 0.999 bar.)

3. **At b==1 single-caption INFERENCE the keep really is all-ones — TRUE.**
   Traced `pad_text_features` (`pipeline.py:32-58`): it pads a *list* of per-sample
   features to the per-call batch max and sets mask=1 for `[:ln]`, 0 for padding.
   `encode_krea_prompt` (`text_encoder.py:64-73`) tokenizes a single prompt at its
   **natural length** (`truncation=True`, NO `padding="max_length"`). At b==1 the
   list has length 1 → `max_len == its own length` → mask all-ones, zero padding.
   cond and uncond are padded in SEPARATE `pad_text_features` calls
   (`pipeline.py:227` vs `231`), each at its own length, so neither carries padding
   at b==1. **So the no-mask refiner is the correct math for the REAL inference
   case.** The problem is the *oracle*, which exercises a case that never occurs at
   b==1 inference.

4. **The math-mode masked path is faithful to cuDNN — the divergence is NOT a
   port bug.** Single-attention, same 6-pad masked op: cuDNN(bf16) vs MATH-f32 →
   **cos 0.99999797**. serenity's `_sdpa_math` masked path (additive float mask,
   F32 softmax) reproduces cuDNN's masked output essentially exactly. So *if* you
   pass the mask, the port matches the reference; the 0.9988 FAIL above is purely
   the absence of the mask, not a numerical defect in the masked path.

### THE KEY QUESTION — recommendation (definitive)

For a **faithful inference port**, gating against `torch mask=None` is acceptable
**only if the oracle is also generated all-ones** (the genuine b==1 case). Today it
is NOT — the generator dumps a 6-pad oracle, so the gate is comparing two different
things and is meaningless. **Two acceptable fixes; pick one and make code+oracle
agree:**

- **(Preferred) Make the oracle match the real inference case:** set
  `keep = torch.ones(1, LT)` (NPAD=0) in `gen_krea2_txtfusion.py`, regenerate, and
  keep the no-mask Mojo path. Then the gate is honest: all-ones → softmax-invariant
  → no-mask == cuDNN-with-mask (cos 1.0), proven in measurement #1. This is what
  the probe comment *claims* is already happening — make it true.
- **(Alternative) Keep the 6-pad oracle and pass the mask** through `krea2_mha`'s
  masked branch. This matches the reference to ~0.99999 per-op (measurement #4) and
  exercises the masked code path that chunk 7 will need. **But this currently
  cannot run — see BLOCKER-1 (dtype).**

Either way, **the current committed state (6-pad oracle + no-mask probe) must not be
called "passing."** It is a FAIL.

---

## BLOCKER-1 — masked path raises `q/mask dtype mismatch`; chunk 7 will hit this. (`krea2_dit.mojo:996`, `build_krea2_text_mask` returns F32 at `:950`)

`build_krea2_text_mask` hardcodes `STDtype.F32` output. `krea2_mha` runs q/k/v in
the storage dtype (bf16 in the oracle/inference). `sdpa` enforces
`q.dtype()==mask.dtype()` (`ops/attention.mojo:1623`). I wired a temp masked probe
(build mask from the oracle `keep`, feed it to the refiner) and it dies immediately:

```
Unhandled exception caught during execution: sdpa: q/mask dtype mismatch   EXIT=1
```

So the entire masked branch of `krea2_mha` is **dead code that has never executed**
— the no-mask gate hid that it doesn't even run. Chunk 7's full forward builds
`mask = _mask(mask)` (reference `mmdit.py:441`) and feeds it to every
`SingleStreamBlock`; ported to Mojo with an F32 mask against a bf16 model, it will
raise the same way. **Fix:** give `build_krea2_text_mask` an `out_dtype` parameter
(0.0/1.0 are exact in bf16, so the bf16 mask is lossless and the math path adds it
in F32 — verified faithful in measurement #4), OR build the mask in the model
compute dtype at the call site. Severity **BLOCKER** for chunk 7; FRAGILE for
chunk-6-as-inference (no-mask never reaches it), but it means the masked branch has
zero test coverage.

---

## FRAGILE-2 — probe comment is actively misleading / self-contradictory. (`krea2_txtfusion_parity_probe.mojo:71-81`)

The comment states the oracle keep is all-ones; the generator it points to dumps a
6-pad keep. This is the proximate cause of the false-green. Whichever fix is chosen
above, rewrite this comment to state what the oracle actually contains. As written
it would lead a future reader (and chunk 7) to trust an unmasked refiner is
universally correct, which measurement #2 refutes.

---

## VERIFIED CORRECT (independently checked, no defect)

- **Projector axis (point 5) — CORRECT.** Ran the reference: `projector.weight`
  is `(1, 12)`, acts on the size-12 LAYER axis; the forward errors
  `mat1/mat2 ... (61440x10 and 12x1)` when fed n=10, proving n is the layer axis.
  Mojo `krea2_text_fusion` reshapes `[1,LT,12,d]→[LT,12,d]`, runs layerwise over
  S=12, then `transpose(x,1,2)→[LT,d,12]` + `linear(projector_w[1,12])` over the
  last (12) axis → `[LT,d,1]` → `[1,LT,d]`. Matches `rearrange (b l) n d -> b l d n`
  + `reshape(b*l,d,n)` + `Linear(n→1)` exactly.

- **Axis assignment (point 6) — CORRECT.** Layerwise blocks: Mojo `B=LT, S=NLAYERS,
  mask=None` ≡ reference `reshape(b*l, n, d)` SDPA over n, batched over b*l=LT, mask
  None (`mmdit.py:296-298`). Refiner blocks: Mojo `B=1, S=LT` ≡ reference attends
  over the LT tokens after the projector (`mmdit.py:306-307`). Refiner mask is
  `_mask(mask[:, :context.shape[1]])` (text-length slice) — matches the
  `build_krea2_text_mask([Lt])` contract.

- **`krea2_mha` structure (point 7) — CORRECT.** vs reference `Attention` with
  `freqs=None`, `gqa=False`: heads==kvheads=20, headdim=128, NO rope (no
  `ropeapply`), NO repeat_kv. QKNorm (`krea2_rmsnorm` over headdim, q/k only, v
  untouched), sigmoid-gate on wo's input (`merged * sigmoid(gate)`), wo with no
  bias. All present and ordered as `mmdit.py:215-226`.

- **`build_krea2_text_mask` value semantics — CORRECT (additive 0/1).**
  `out_m[i,j] = keep[i]*keep[j]`, fed as the additive score bias to math-mode sdpa
  (which adds the mask then softmaxes), matching `_mask` (outer product) + F.sdpa's
  float-mask-is-additive convention. (The *gate* is wrong, not this builder.)

- **`krea2_text_fusion_block` — CORRECT.** Plain residual ADD (no AdaLN gate);
  `x + attn(prenorm(x))`, `x + mlp(postnorm(x))` matches `mmdit.py:260-264`.

- **Mojo correctness (point 8) — clean.** All chunk-6 defs `raises`; comptime dims
  (`B/S/HEADS/HEADDIM`/`LT`/`NLAYERS`); `Optional[Tensor]` handled via
  `if mask:` / `mask.value()`; `Krea2TextFusionWeights` is ArcPointer-shared
  (Copyable/Movable over move-only Tensor); no `var ref` misuse; `^` transfers on
  the returned tensors. Probe compiles (only the pre-existing `fn`-deprecation
  warning from `ops/attention.mojo:260`).

---

## One-line verdict: **BLOCKERS: 1** (masked path dtype) **+ a FALSE-GREEN gate** — the committed 6-pad oracle + no-mask probe is a FAIL (cos 0.9988), passing only because of a stale all-ones oracle on disk. Make code and oracle agree (all-ones oracle + no-mask, OR 6-pad oracle + bf16 mask) before accepting chunk 6, and fix the F32-mask dtype before chunk 7.
