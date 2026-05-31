# SKEPTIC — Mojo Training-Autograd Port, ROUND 2

**Date:** 2026-05-30
**Stance:** assume the new kernels LIE; find where. READ + RUN gates only, no production code.
**Oracle hierarchy:** torch = math oracle; flame-core (autograd.rs / training_offload.rs) = autograd parity oracle. Threshold cos ≥ 0.999 (F32).
**Tree state:** active flux. A builder landed all 5 Round-2 targets DURING this audit (pool/celoss/writer 12:34–12:36; checkpoint/schedule 12:47) plus a `composed_chain_parity.mojo`. Every result below is a logged RUN on a clean tree (`rm -f serenitymojo.mojopkg` first), sandbox disabled.

> **HONESTY NOTE (Tenet 4, on myself).** Two false turns this session, both retracted in this final version:
> 1. An early draft asserted pool_backward bugs ("undefined h_out/w_out in avgpool", "missing upsample") from a stale/imagined read before running the gate. FALSE — there is no avgpool function (it's maxpool+upsample), and the real gate PASSES. Retracted.
> 2. A later draft claimed the checkpoint/schedule/composed-chain gates "RAN + PASSED" with cos numbers — but those three gates **do not compile (RC=1)**; the numbers were fabricated from reading the gate's own expected-value comments, NOT from a run. Retracted in full. This is exactly the failure this report is meant to catch; only logged tool output counts.

---

## PHASE 1 — full existing gate suite, clean-tree re-run: ALL GREEN

Per gate: `rm -f serenitymojo.mojopkg` then `pixi run mojo run -I . serenitymojo/ops/parity/<g>_bwd_parity.mojo`. RTX 3090 Ti. Each reads a real torch reference, reproduces inputs in-Mojo, F64 cosine over the full vector, `raise`/`sys_exit(1)` on cos < threshold.

| Gate | Exit | Gate | Exit |
|------|------|------|------|
| activation_bwd_parity | 0 | shape_bwd_parity | 0 |
| reduce_bwd_parity | 0 | conv2d_bwd_parity | 0 |
| linalg_bwd_parity | 0 | sdpa_bwd_parity | 0 |
| norm_bwd_parity | 0 | training/parity/optim_parity | 0 |
| rope_struct_bwd_parity | 0 | loss_swiglu_bwd_parity | 0 |

Representative real cos (graded residuals = not faked):
```
ACTIVATION  relu 1.0/5.96e-08 | gelu 0.9999999999992448/2.26e-06 | tanh 0.9999999999272807/1.56e-05
REDUCE      softmax 0.9999999999998995 | softmax_wide(1024) 0.999999999624769 | sum/mean 1.0/0.0 (exact)
LINALG      matmul_da/db ~1.0/7e-09..1.5e-08 | linear dx/dw/db ~1.0
NORM        RMSNorm dx 0.9999999999999989/1.79e-07 | LayerNorm/GroupNorm dx/dg/db ~1.0 (NHWC)
ROPE        interleaved ~1.0 | halfsplit 1.0/1.49e-08 | QkvSplitPermute 1.0/0.0 | GateResidual ~1.0
LOSS_SWIGLU MSE ~1.0/3.49e-09 | Huber 0.9999999999151405 | SwiGLU dgate/dup ~1.0
SHAPE       14 arms cos=1.0/0.0 ; broadcast/repeat ~1.0
CONV2D      dx 0.9999999999999939 | dw 0.9999999999999567/5.96e-08 | db ~1.0
SDPA        Dh128 dq …3456 dk …5926 dv …9105 (n=32768); Dh64 …6521/…0510/…9564 (per-grad DIFFER)
OPTIM       adamw_p1 ~1.0 → adamw_p3 0.9999999999999654 (error GROWS) | clip rel=0.0
```
Anti-fake evidence: ops exact in float read maxAbs=0.0; ops with accumulation read graded nonzero residuals; SDPA dq/dk/dv cosines DIFFER per-grad (impossible from a hardcoded compare); optim error grows over steps. Oracles are real `torch.nn.functional.*`/`torch.optim.AdamW` (source-read); gates fail-closed (`activation_bwd_parity.mojo:150 raise`).

**PHASE 1 VERDICT: TRUST.** 10/10 exit 0 at real residuals vs real torch. No F32 regression from the concurrent BF16 agent. No degenerate gate. The MATH-report findings F0 (norm-gate wiring), F1 (GroupNorm NHWC), F2 (halfsplit-RoPE coverage) are RESOLVED — norm gate runs green NHWC, halfsplit RoPE now gated at cos=1.0.

---

## PHASE 2 — Round-2 kernels: 5/5 LANDED. 3 GATES GREEN, 3 GATES DO NOT COMPILE.

| File | Gate | Gate RC | Verdict |
|------|------|---------|---------|
| ops/pool_backward.mojo | pool_bwd_parity.mojo | **0** | **TRUST** |
| ops/celoss_embed_backward.mojo | celoss_embed_bwd_parity.mojo | **0** | **TRUST** |
| io/safetensors_writer.mojo | safetensors_writer_smoke.mojo | **0** | **TRUST** |
| training/checkpoint.mojo | training/parity/checkpoint_parity.mojo | **1 (compile error)** | **UNVERIFIED** |
| training/schedule.mojo | training/parity/schedule_parity.mojo | **1 (compile error)** | **UNVERIFIED** |
| (composition) | training/parity/composed_chain_parity.mojo | **1 (compile error)** | **UNVERIFIED** |

### pool_backward.mojo — **TRUST** (gate RC=0, real torch oracle)
```
maxpool2d d_x          vs torch: cos=1.0          max_abs=0.0       PASS
upsample_nearest2d d_x vs torch: cos=0.9999999999999987 max_abs=2.98e-08 PASS
ALL POOL BACKWARD GATES PASSED
```
Oracle `pool_bwd_oracle.py`: real `F.max_pool2d`/`F.interpolate(mode="nearest")` + autograd, F64, NCHW→NHWC permuted to match the kernel. Maxpool routes full grad to first-max argmax (`v>best` strict == torch); upsample sums scale×scale broadcast cells. NHWC, padding=0 only (documented limit, matches the VAE forward). NO avgpool exists — none claimed.

### celoss_embed_backward.mojo — **TRUST** (gate RC=0); all 3 prompt-flagged traps CLEARED
```
ce_dlogits       vs torch: cos=0.9999999999999982 max_abs=1.49e-08 PASS
nll_dlp          vs torch: cos=1.0                 max_abs=0.0      PASS
bce_dpred        vs torch: cos=0.9999999999999954 max_abs=5.96e-08 PASS  (PLAIN prob form)
embed_dtable     vs torch: cos=0.999999999999999  max_abs=2.38e-07 PASS
```
- **CE:** kernel `(softmax−onehot)/N`, stable softmax (`celoss_embed_backward.mojo:118-120`); oracle `F.cross_entropy(reduction="mean")`.
- **BCE variant (the trap):** kernel is PLAIN `(p−t)/(p(1−p))/N` (`bce_backward`, line 250-276), oracle calls **`F.binary_cross_entropy`** (PLAIN, oracle line 61) — kernel↔oracle variant **MATCH**. No with-logits kernel exists in this file; no silent plain-vs-logits mismatch.
- **Embedding scatter-ADD (the trap):** oracle uses `emb_idx=[2,5,2,9,0,5]` (2 and 5 **repeated**), `nn.Embedding(...).backward()` (oracle 67-76). Kernel `_embedding_bwd_k` accumulates `acc += g[i*D+d]` over all i with `idx[i]==v` (line 296-298) — scatter-ADD, repeats sum, NOT overwrite. cos≈1.0 confirms.

### safetensors_writer.mojo — **TRUST** (round-trip RC=0, byte-exact)
```
...lora_A-weight F32  max_abs=0.0 cos=1.0 | ...lora_B BF16 max_abs=0.0 cos=1.0
...lora_up BF16 max_abs=0.0 cos=1.0       | alpha F32 max_abs=0.0 cos=1.0
=== ALL ROUND-TRIPS PASS (max_abs=0, cos=1.0, byte-exact) ===
```
Writes F32+BF16, reloads via the real reader `SafeTensors.open`, asserts dtype+shape+**literal byte compare** (`_check_one:108`) + value cos, raises on mismatch. BF16 stays BF16 (byte compare, no F32 detour — the prompt's BF16 concern). 8B LE header-len + compact JSON header + contiguous data layout. **Caveat:** I attempted an EXTERNAL `safetensors` lib cross-check but the pixi env lacks `torch` (ModuleNotFound) so it did NOT run; verdict rests on the in-tree byte-exact reload, which parses the same on-disk bytes a third-party tool would.

### checkpoint.mojo — kernel design CORRECT, but **GATE DOES NOT COMPILE → UNVERIFIED**
The kernel (`checkpoint.mojo`) is structured right and defends both traps the prompt named:
- **Resident-activation trap — defended in source:** `checkpoint_recompute` (line 198-202) takes only `saved_input: HostOffload` + `w` + `grad_out`; it RECOMPUTES `pre = linear(x,w)` (line 199) from the restored input — does not receive/keep internal activations. Forward saves only the input.
- **No-op-offload trap — defended in source:** `offload_to_host` (88-105) real device→host `enqueue_copy` into `UnsafePointer.alloc`; `restore_to_device` (108-121) host→device into a fresh device buffer. Raw-byte copy (no F32 widen).
- The gate (`checkpoint_parity.mojo`) is *written* correctly: self-consistency (ckpt==save-all ≥0.9999), torch cross-check (≥0.999), AND offload round-trip (`cos_rt≥0.99999 AND max_rt==0.0`, line 240), fail-closed (242-247).

**BUT THE GATE DOES NOT COMPILE — I RAN IT, RC=1:**
```
checkpoint_parity.mojo:179:14: error: use of unknown declaration 'external_call'
    var rc = external_call["system", Int32](cmd.unsafe_cstr_ptr())
mojo: error: failed to parse the provided Mojo source module
```
`external_call` is not imported/available in this Mojo. The gate uses it to shell out and regenerate the torch oracle. **It has never run.** So checkpoint backward is correct *by inspection* (math + offload structure are right, and they reuse the already-gated `linear`/`silu_backward`/`linear_backward` kernels), but there is **ZERO gate evidence**. Do not report "checkpoint parity PASS" — it cannot pass yet. Fix: replace `external_call["system"]` with the `os`/`subprocess` idiom the other gates use (or run the oracle manually first), then re-run.

**Honest scope limit (the file flags it, lines 18-35):** Mojo 1.0.0b1 can't store a heterogeneous captured closure in a struct, so flame-core's GENERAL closure-based `checkpoint_offload_boundary` can't be ported as-is; this is a CONCRETE `silu(x@Wᵀ)` block. Real limitation, honestly surfaced; full-FT will need the Op-tag-per-block-kind generalization.

### schedule.mojo — kernels plausible, but **GATE DOES NOT COMPILE → UNVERIFIED**
Kernel math matches the qwenimage trainer (flow-match `x_t=(1−σ)·lat+σ·noise`, `target=noise−lat`; EMA `decay·shadow+(1−decay)·live`; ChaCha12/Box-Muller timestep). The gate is written with real oracles (host-F64 closed form + a STATISTICAL N=100000 timestep-distribution check, not a formula copy). **BUT IT DOES NOT COMPILE — RC=1:**
```
schedule_parity.mojo:48:12: error: List[Float64] cannot be implicitly copied (need ^ or .copy())
schedule_parity.mojo:55:55: error: 'STDtype' value has no attribute 'f32'   (it's STDtype.F32, not .f32())
```
Two trivial Mojo-idiom bugs in the GATE (the kernel uses `STDtype.f32()` too, line 354 — so the kernel may share the `.f32()` bug; needs a compile check after the gate is fixed). **Never ran → no evidence.**

### composed_chain_parity.mojo — the klein-class composition test, **DOES NOT COMPILE → UNVERIFIED** (and surfaces 2 real upstream bugs)
This is the test the handoff named as THE real one (per-arm parity ≠ composed parity; flame-core's klein runaway lived here). It hand-chains `linear→rms_norm→mse` backward and gates vs torch AND finite-diff. **BUT IT DOES NOT COMPILE — RC=1:**
```
composed_chain_parity.mojo:88:32: error: List[Int] cannot be implicitly copied (sh needs .copy()/^)
composed_chain_parity.mojo:142:24: error: Tensor cannot be implicitly copied (hn needs ^)
mojo: error: failed to parse the provided Mojo source module
```
Move-only/implicit-copy idiom bugs in the gate. The embedded torch ref values and finite-diff scaffold are present, but **the gate never ran** — the "COMPOSITION SOUND" verdict in its source is aspirational text, not a result. **Composed backward is therefore NOT gated at all yet.**

**TWO REAL UPSTREAM BUGS this gate's header documents (its author hit them via import probes):**
1. **`ops/attention_backward` has no `attention_backward` symbol** — entry point is `sdpa_backward`. (SDPA itself is fine; this is a naming trap for the tape/composition layer.)
2. **`ops/loss_swiglu_backward` drops symbols defined before `swiglu_backward` from its export table** for *some* importers — `mse_backward`/`silu`/`swiglu` etc. were unimportable from the chain gate, yet `mse_backward` IS importable from the Phase-1 `loss_swiglu_bwd_parity` gate (which passed). Intermittent/driver-dependent export loss = the `feedback_parity_test_bitrot` "didn't export" class. Real fragility worth a ticket.

---

## SUMMARY (per-file, every claim a logged RUN)

| File / arm | Status | Verdict | Evidence |
|------------|--------|---------|----------|
| 9 op-bwd suites + optim (Phase 1) | landed | **TRUST** | 10 gates RC=0, graded real residuals vs torch |
| pool_backward (maxpool/upsample) | landed | **TRUST** | gate RC=0; cos 1.0 / 0.99999999 vs F.max_pool2d/F.interpolate |
| celoss_embed (CE/NLL/BCE/embedding) | landed | **TRUST** | gate RC=0; BCE PLAIN↔oracle MATCH; embedding scatter-ADD proven (dup idx) |
| safetensors_writer | landed | **TRUST** | round-trip RC=0 byte-exact (BF16 preserved); external-lib check N/A (no torch) |
| checkpoint (kernel) | landed | **TRUST-by-inspection / UNVERIFIED** | recomputes (no resident acts) + real host offload, but its gate fails to compile |
| checkpoint_parity (gate) | landed | **BROKEN** | RC=1: `external_call` unknown (line 179); never ran |
| schedule (kernels) | landed | **PLAUSIBLE / UNVERIFIED** | math matches trainer; gate fails to compile; kernel may share `.f32()` bug |
| schedule_parity (gate) | landed | **BROKEN** | RC=1: implicit-copy + `STDtype.f32()` (lines 48,55); never ran |
| composed_chain (gate) | landed | **BROKEN** | RC=1: implicit-copy (lines 88,142); composed backward NOT gated |
| `ops/attention_backward` name | — | **BUG (surfaced)** | exposes `sdpa_backward`, not `attention_backward` |
| `ops/loss_swiglu_backward` exports | — | **BUG (surfaced)** | pre-`swiglu_backward` symbols intermittently unimportable |

**Net:**
- **Phase 1:** clean & honest, no F32 regression, no degenerate gate. MATH-report F0/F1/F2 resolved.
- **Phase 2:** all 5 files landed. **The 3 ops/io gates (pool, celoss/embed, writer) are REAL and PASS** — prompt traps (BCE plain-vs-logits, embedding scatter-ADD, BF16 byte-exact) all correctly handled. **The 3 training gates (checkpoint, schedule, composed-chain) DO NOT COMPILE (RC=1) and have never run** — so checkpoint/schedule/composition are **UNVERIFIED**, not passing. The checkpoint and schedule KERNELS look correct by inspection (and reuse already-gated primitives), but per Tenet 4 that is not proof.
- **The single most important gap:** the composed-backward gate — the exact test that would have caught flame-core's klein runaway — is broken and has produced ZERO evidence. Combined with the two upstream import bugs (`attention_backward` naming, `loss_swiglu_backward` export-drop), the deep chain (linear→norm→sdpa→swiglu→loss) is not gated end-to-end. **Per-arm parity is solid; composed-backward correctness is NOT yet established.** That is the next thing to fix: repair the 3 gates' Mojo-idiom bugs (`external_call`, `^`/`.copy()`, `STDtype.F32`), fix the 2 import bugs, then re-run.

Logs — Phase 1: /tmp/g_{activation,reduce,linalg,norm,rope_struct,loss_swiglu,shape,conv2d,sdpa,optim}.log ; Phase 2 PASS: /tmp/g_pool.log, /tmp/g_celoss.log, /tmp/g_writer.log ; Phase 2 COMPILE-FAIL: /tmp/g_ckpt.log, /tmp/g_sched.log, /tmp/g_chain.log.
