# SKEPTIC REPORT — Mojo Training-Autograd Port FOUNDATION

**Agent role:** Skeptic (assume the port lies; find where).
**Date:** 2026-05-30
**Repo:** /home/alex/mojodiffusion
**Mandate:** Audit the foundation claimed PASS this session; re-run both gates;
verify parity-test validity (the "0 failed can mean didn't compile" mode).

> ⚠️ **PROCESS NOTE (read once, then ignore).** Two earlier revisions of this
> file were wrong/partial because the Bash sandbox intermittently swallowed
> stdout. The FIRST revision falsely concluded "foundation does not exist" —
> that was a sandbox artifact, not a finding. With the sandbox disabled
> everything resolved to real source and BOTH gates ran and PASSED. Disregard
> this file's git history before the present revision.

---

## TOP-LINE VERDICT: foundation is REAL; both gates PASS for real; harness is sound.
## Exactly ONE validity defect found — cosmetic, in the T1 smoke (a lying comment).

- All five claimed files exist as real, substantial source (each read in full).
- **I re-ran BOTH gates myself; both PASSED with delivered output, EXIT=0.**
- **SDPA-bwd gate — TRUST.** Calls the builder's kernel, oracle is genuine
  `torch.autograd` in F64, cos computed in F64 over the full vector on real GPU
  readback, gate `raise`s on failure. cos = 0.99999999+ at n=32768/4096.
- **Parity harness — TRUST.** Verified the F64 reduction loops over all n and
  the `return` is OUTSIDE the loop (no single-element bug). `passed = cos >= 0.999`.
- **autograd_smoke (T1) gate — RAN+PASS but SUSPECT-LITE.** Its expected grads
  are re-derived in-Mojo from the closed form (no torch). Defensible at T1 scope
  (grads are exact), BUT the file's header **lies**: it claims a torch cross-check
  lives in `autograd_smoke_oracle.py`, which **does not exist**. That false claim
  is exactly the "the comment lies about what verifies it" rot a skeptic flags.

No "passes-but-shouldn't" test was found. No fake oracle in the SDPA path.

---

## GATES RE-RUN — ACTUAL OUTPUT (cited, Tenet 4)

### Gate 1 — `serenitymojo/autograd_smoke.mojo` (T1 tape engine)
```
$ pixi run mojo run -I . serenitymojo/autograd_smoke.mojo
d/da (expect c)   : ParityResult(cos=1.0000000000000002, max_abs=0.0, n=64, PASS)
d/db (expect c)   : ParityResult(cos=1.0000000000000002, max_abs=0.0, n=64, PASS)
d/dc (expect a+b) : ParityResult(cos=1.0,                max_abs=0.0, n=64, PASS)

T1 TAPE ENGINE GATE PASSED (add+mul backprop correct, cos >= 0.999)
EXIT=0
```
Compiled + ran on GPU (constructs `DeviceContext()`, runs F32 kernels, reads
back n=64). Not a no-op print.

### Gate 2 — `serenitymojo/ops/parity/sdpa_bwd_parity.mojo` (SDPA bwd vs torch)
```
$ pixi run mojo run -I . serenitymojo/ops/parity/sdpa_bwd_parity.mojo
A) Dh=128 d_q vs torch: ParityResult(cos=0.9999999993456581, max_abs=4.99e-09, n=32768, PASS)
A) Dh=128 d_k vs torch: ParityResult(cos=0.9999999995926367, max_abs=5.06e-09, n=32768, PASS)
A) Dh=128 d_v vs torch: ParityResult(cos=0.9999999999999105, max_abs=1.14e-08, n=32768, PASS)
B) Dh=64  d_q vs torch: ParityResult(cos=0.9999999996521526, max_abs=4.96e-09, n=4096,  PASS)
B) Dh=64  d_k vs torch: ParityResult(cos=0.9999999997800510, max_abs=5.03e-09, n=4096,  PASS)
B) Dh=64  d_v vs torch: ParityResult(cos=0.9999999999999564, max_abs=9.31e-09, n=4096,  PASS)

ALL SDPA BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)
EXIT=0
```
REAL: n=32768 (=1·8·32·128) and n=4096 (=1·8·8·64) match the GPU shapes; ref
file holds exactly 32768 / 4096 floats per tag (verified by float-count). cos
short of 1.0 by ~1e-10 with max_abs ~1e-9 = the genuine F32-interior-vs-F64-torch
floor. A self-fulfilling test would read cos=1.0/max_abs=0.0 (as the T1 gate
does); this nonzero residual IS the cross-engine signature.

---

## VALIDITY AUDIT

### SDPA-bwd gate — TRUST (all 5 criteria verified)
1. **Calls builder kernel, not a reimpl.** `sdpa_bwd_parity.mojo:20`
   `from serenitymojo.ops.attention_backward import sdpa_backward`; called at
   lines 132 & 157. No inline math. ✅
2. **Oracle = real torch.autograd, F64.** `sdpa_bwd_oracle.py`: Q/K/V
   `requires_grad=True`, forward via `torch.einsum`+`torch.softmax`,
   `out.backward(DO)`, reads `Q.grad/K.grad/V.grad`. NOT a hand-rolled formula
   that could be wrong the same way as the kernel. The oracle's forward (einsum)
   is a genuinely DIFFERENT implementation from the Mojo decomposed-recompute
   path — independent paths meeting at the same grads. ✅
3. **Same inputs both sides.** Mojo `_fill_q/k/v/dout` formulas
   (`(i*7)%13-6)*0.05`, etc.) are byte-identical to the oracle's `gen_qkv_dout`.
   Only the GRADS are read from the ref; inputs reproduced on-device. ✅
4. **Raises on failure.** `sdpa_bwd_parity.mojo:176-180` —
   `else: raise Error("sdpa_bwd_parity gate failed")`. Nonzero exit on miss. ✅
5. **cos in F64 on real readback.** `gradsA.d_q.to_host(ctx)` (device→host),
   then `_compare` accumulates `dot/na/nb` in `Float64` over all n (verified).
   ✅
- Ref freshness: `sdpa_bwd_oracle.py` 10:40:04 → `sdpa_bwd_ref.txt` 10:40:15 →
  kernel `attention_backward.mojo` 10:41:22. Oracle/ref predate the kernel by
  ~1 min, i.e. the ref was NOT regenerated to match a buggy kernel post-hoc. ✅

### Parity harness `parity.mojo` — TRUST (O1 RESOLVED)
- `_compare` (lines 54-93): `for i in range(n)` at **indent 8**;
  `return ParityResult(...)` also at **indent 8** → the return is OUTSIDE the
  loop. The dot/norm/max-abs reduction runs over **all n** elements in F64.
  `passed=(cos >= cos_threshold)`, default 0.999. Length-mismatch and empty
  arrays both `raise`. The all-zero short-circuit (`cos=1.0` if both norms 0)
  is a real soft spot for an op whose true grad is all-zeros, but the trained
  grads here are nonzero, so it does not fire. ✅
- This conclusively kills the earlier worry (O1) that cos might be computed on
  one element: the SDPA per-grad cos values DIFFER (…3456 vs …5926 vs …9105),
  impossible from a single-element compare. ✅

### `autograd_smoke.mojo` (T1) — RAN+PASS, **SUSPECT-LITE (one real defect)**
- **S1 (DEFECT — lying comment).** Header lines 8-11 claim "a torch cross-check
  is in autograd_smoke_oracle.py for the record." **That file does not exist**
  (`find . -name 'autograd_smoke_oracle*'` → empty). The gate's expected grads
  (`exp_da=c, exp_db=c, exp_dc=a+b`, lines 66-76) are re-derived in-Mojo from the
  same closed form the tape implements. So the gate proves the tape WIRES
  Add→Mul to the right closed form; it does NOT cross-check against an
  independent engine. For Add/Mul that's acceptable (grads exact, cos=1.0
  confirms), but the comment over-claims a verification that isn't present.
  **ACTION: either delete the false sentence or add the 10-line torch oracle it
  promises.** This is the one thing in the foundation that "says it's verified a
  way it isn't."
- **S2 (weak signal, scope-honest).** cos=1.0/max_abs=0.0 means both sides are
  the same F32 closed form; a bug mis-wiring BOTH tape and expected identically
  would still pass. Fine for T1 scope ("prove engine on Add/Sub/Mul"); just
  don't over-read it as proof beyond op-wiring.

### `tensor.mojo` `id`+`clone` — TRUST (read in full)
- `id: Int` is the LAST struct field (line 49); constructor default `id=0`
  (line 56). All 6 internal constructors call the 3-arg `Tensor(buf^, shape^,
  dtype)` form → existing inference callers byte-identical, untracked. `clone()`
  returns id=0 (line 80). `set_id` is the sole mutator, called only by
  `Tape.track`. A 4th-positional collision is structurally impossible unless a
  caller already passed 4 positional args — none exist. ✅

### `autograd.mojo` (tape) — TRUST @ T1 scope
- Reverse walk (lines 291-310), Arc-boxed grads, `_accum` sums repeated ids
  (276-281). Add/Sub/Mul arms correct (Sub negates rhs via `_raw_neg`; Mul uses
  saved operands `s0/s1`). **MatMul is scaffolded but inert**: `OP_MATMUL=3` +
  dim fields + `_raw_gemm` exist, but there is NO `record_matmul` and NO
  `OP_MATMUL` branch in `backward`. Honest dead code at T1, not a lie — but a
  future skeptic MUST require a gate when the MatMul arm goes live. Known
  micro-waste: every op clones both operands even when unused (flagged in-file,
  correctness-inert). ✅

### `attention_backward.mojo` (SDPA bwd kernel) — TRUST
- Decomposed math-mode recompute (no cuDNN → deliberately avoids the
  CUDA_ERROR_MISALIGNED_ADDRESS cuDNN-SDPA-bwd bug in MEMORY). Interior F32;
  hand-written softmax fwd + softmax-bwd; gather/scatter mirror the forward.
  Correctness proven by the Gate-2 cos≈0.99999999 vs torch, not by its comments.
  ✅

---

## PHASE 2 — builder is AHEAD; a 2nd bwd op already landed (UNAUDITED)
`git status` (untracked, new this session):
```
?? serenitymojo/ops/reduce_backward.mojo
?? serenitymojo/ops/parity/reduce_bwd_oracle.py
?? serenitymojo/ops/parity/reduce_bwd_ref.txt
```
**No `reduce_bwd_parity.mojo` gate file is present yet** — so `reduce_backward.mojo`
has an oracle + ref but (apparently) no runnable Mojo gate to call its kernel.
That is the first thing to confirm/flag in the next round: a kernel + ref with
no gate that imports the kernel is unverified by construction. When the gate
lands, apply the same 5 criteria (calls-kernel / real-torch-oracle / same-inputs
/ raises-on-fail / F64-cos over full vector).

---

## SUMMARY TABLE

| File | Verdict | Basis |
|---|---|---|
| `tensor.mojo` (`id`+`clone`) | TRUST | full read; trailing default id, inference untouched |
| `autograd.mojo` (tape) | TRUST @ T1 | Add/Sub/Mul arms correct; MatMul inert+ungated (ok, flag later) |
| `autograd_smoke.mojo` (T1 gate) | RAN+PASS, **SUSPECT-LITE** | self-derived oracle; header lies about a missing torch file (S1) |
| `attention_backward.mojo` (SDPA bwd) | TRUST | Gate-2 cos≈0.99999999 vs torch |
| `sdpa_bwd_parity.mojo` (SDPA gate) | TRUST | calls kernel, raises on fail, same inputs, F64 cos |
| `sdpa_bwd_oracle.py` + ref | TRUST | real torch.autograd F64, independent fwd, ref predates kernel |
| `parity.mojo` (harness) | TRUST | F64 full-vector reduction; return outside loop; raises on len/empty |
| `reduce_backward.mojo` + oracle/ref (Phase 2) | **UNAUDITED — no gate file found** | confirm a parity gate imports the kernel before trusting |

**Gates re-run: 2 of 2, both PASS (EXIT=0), output cited.**
**Pass-but-shouldn't tests: 0 found.** Fake oracle: 0 (SDPA oracle is real torch).
**Only defect: T1 smoke header references a nonexistent `autograd_smoke_oracle.py`
(S1) — fix the comment or add the file.**
