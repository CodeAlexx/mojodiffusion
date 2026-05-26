# PORT_STATE: serenity-safetensors → Mojo (Serenitymojo io)

Phase: **✅ LOADER COMPLETE & BYTE-PARITY-CLEAN (chunk 1 + chunk 2)** — 2026-05-25.

## FINAL verified state (orchestrator-verified, not trusted)
- Chunk 1 (single-file reader: dtype/ffi/mmap/json_header/safetensors) — 1163/1163 byte-identical across all 6 Z-Image shards; origin-bound `tensor_bytes` (use-after-munmap = compile error).
- Chunk 2 (sharded.mojo ShardedSafeTensors + tensor_view.mojo TensorView) — transformer 521/521, text_encoder 398/398, single-file VAE 244/244 byte-identical THROUGH the loader (2 oracles). Nested-ownership lifetime safe via `List[ArcPointer[SafeTensors]]` + origin tracing; both escape paths (return + explicit-destroy) compile-rejected.
- Bugs caught & fixed by the pipeline: use-after-munmap (chunk1 F1), header partial-read (chunk1 F2), **ffi NUL-termination latent producer bug** (chunk2 F1 — `sys_open` now copies path to an owned NUL-terminated buffer), empty weight_map now raises (chunk2 F3).
- KNOWN LIMITATION (documented, not fixable now): reassigning the source binding (`sh = open(other)`) while a view is live is a use-after-free NOT caught by Mojo 1.0.0b1 origin tracking (chunk2 F2 — caller contract, docs corrected). Escape-return + explicit-destroy ARE caught.
- All probes/oracles kept at io/parity/. Build probes inside the package with `pixi run mojo build -I . <file>`.

PILOT VERDICT: Rust→Mojo pipeline (plan→build→skeptic→bugfix→orchestrator re-verify→byte-parity gate) PROVEN end-to-end. Reuse for flame-core forward kernels + models.

---
(historical) Phase: CHUNK 2 skeptic-complete → bugfix (2026-05-25). Chunk 1 COMPLETE & VERIFIED CLEAN (below).
Chunk-2 skeptic (SKEPTIC_FINDINGS_chunk2_2026-05-25.md): FULL parity GREEN — transformer 521/521, text_encoder 398/398, single-file VAE via loader 244/244 (2 independent oracles). Findings:
- **F1 BLOCKER (confirmed-reproduced)**: ffi.mojo `sys_open(path.unsafe_ptr())` assumes NUL-termination; `_join`-built paths aren't NUL-terminated → 2nd open of same dir fails (probe_same_twice: vae#2/tf#2 RAISE; probe_nulfix: NUL-term copy works). Chunk-1 ffi latent bug; chunk-2 first triggers. FIX in ffi.mojo (NUL-terminate paths to libc), re-verify BOTH chunks' parity.
- **F2 CORRECT-BUT-FRAGILE**: view-lifetime doc overstates — escape-return + explicit destroy ARE compile-rejected, but reassigning the bound handle while a view is live UAFs (Mojo origin limitation, inherited from chunk1). Soften docs, no code change.
- F3 (empty weight_map opens 0 tensors, should raise) + F4 (multi-file error msg) — cheap/style.
Chunk 2 built: tensor_view.mojo (TensorView[origin] + from_parts[o] which INFERS origin — naming origin_of(st) directly won't unify, real Mojo quirk) + sharded.mojo (ShardedSafeTensors). Nested-ownership origin SOLVED cleanly, NO unsafe cast: SafeTensors is Movable-not-Copyable → List[SafeTensors] uncompilable → used `List[ArcPointer[SafeTensors]]` (Arc Copyable, refcount keeps mmap alive); accessors return Span[UInt8, origin_of(self.shards)]. Dedicated `_parse_weight_map` (index schema ≠ header schema). Orchestrator-verified: smoke 521 tensors + cross-shard OK; escape probe FAILS with correct origin error (built -I .); parity 10/10 sampled byte-match. UNTESTED (skeptic must cover): all-521 + text_encoder(398, model.safetensors.index.json) full byte-parity through the loader; single-file fallback; _parse_weight_map on text_encoder index.

## Chunk 1 final state (orchestrator-verified, not trusted)
- F1 FIXED: `SafeTensors.tensor_bytes(self,name) -> Span[UInt8, origin_of(self)]` (origin-bound). Use-after-munmap now FAILS TO COMPILE — verified: `pixi run mojo build -I . serenitymojo/io/parity/probe_lifetime.mojo` → line-53 escaping-borrow error (NOTE: needs `-I .` or it fails with a misleading import error — bit-rot trap). Bare ptr demoted to private `_tensor_ptr_unsafe`.
- F2 FIXED: `_pread_exact` loop for header reads (mirrors Rust read_exact).
- F3-F7 DEFERRED (malformed-input-only divergence; documented in SKEPTIC_FINDINGS).
- PARITY GATE PASSED: compare.py PASS on all 6 Z-Image shards (vae244/tf1 423/tf2 98/te1 174/te2 219/te3 5 = 1163/1163 byte-identical vs Python safetensors oracle). Smoke runs via safe accessor. Oracle tooling kept at io/parity/.
- Modules: dtype, ffi, mmap, json_header, safetensors (+io/__init__) all compile + run.

## PROVEN Rust→Mojo pipeline (the "proven way for all our needs")
plan(BUILD_PLAN) → build agent → skeptic agent (adversarial, reproduces parity) → bugfix agent (minimal) → orchestrator re-verify (smoke + guard-compile + parity) → byte-parity gate. Demonstrated end-to-end on safetensors. Reuse for flame-core kernels + models.

---
(historical) Phase: skeptic-complete → bugfix (2026-05-25)

## Skeptic result (SKEPTIC_FINDINGS_2026-05-25.md; oracle kept at io/parity/)
PARITY PROVEN: 1163/1163 tensors byte-identical across all 6 Z-Image shards (incl. 9.97GB + 4GB shards → FFI 64-bit clean, JSON handles big headers), cross-checked vs official safetensors 0.7.0.
Triage:
- **F1 BLOCKER**: tensor_ptr() returns bare BytePtr (MutExternalOrigin) — no lifetime tie → use-after-munmap PROVEN (probe_lifetime.mojo → SIGSEGV 139, no compile diagnostic). FIX = origin-bound Span/view accessor tied to `self` (this seeds chunk-2 tensor_view; sets the "all weight access is lifetime-bound" API contract).
- **F2 CORRECT-BUT-FRAGILE (fix, cheap)**: header pread single-shot; loop until header_len bytes (EINTR/NFS/large-header).
- F3 (unknown dtype raises) / F4-F5 (neg/float offsets + surrogate pairs raise) / F6-F7 (brace-skip metadata, header_len==0): DEFER — divergence on malformed input only, never hits real safetensors. Document, don't fix now.
Couldn't verify: TOCTOU/SIGBUS-on-truncation, >100MB header on real file, thread-safety (no Mojo Send+Sync equiv), exotic-dtype byte round-trip (all Z-Image is BF16; F8/U16/U32/U64/BOOL unit-checked only).

## Chunk 1 result (independently verified by orchestrator)
All 5 modules + smoke compile & run (exit 0). Smoke opens Z-Image VAE = 244 tensors, dtype/shape/offset correct, mmap deref works. Files: dtype/ffi/mmap/json_header/safetensors + io/__init__.mojo + smoke_safetensors.mojo.
Builder self-reported 244/244 byte-identical vs a Python safetensors oracle — but DELETED the oracle artifacts → UNVERIFIED, skeptic must reproduce.
Builder skeptic-bait: (1) mmap ptr lifetime — `as_ptr()` has no origin tie to region; caller must keep SafeTensors alive or ASAP-destruction munmaps mid-use (FOOTGUN). (2) FFI off_t/size_t widths inferred empirically (>2GB untested). (3) JSON parser only tested on 244-tensor VAE; 521-tensor transformer + text_encoder NOT exercised; negative/float offsets + surrogate pairs unhandled.
Build-plan corrections: used lseek(SEEK_END) not fstat; STDtype needed ImplicitlyCopyable; needed io/__init__.mojo.
Reference: /home/alex/serenity-safetensors/src/mmap.rs + lib.rs:72-96 (dtype table)
Plan: serenitymojo/io/BUILD_PLAN.md

## Chunks
- [~] Chunk 1 (builder dispatched): dtype.mojo, ffi.mojo, mmap.mojo, json_header.mojo, safetensors.mojo + smoke (single-file VAE)
- [ ] Chunk 2: sharded.mojo, tensor_view.mojo, full parity test (Z-Image transformer+text_encoder shards vs Python safetensors oracle)

## Workflow position
port-build (chunk1) → [next] verify compile → port-skeptic → port-bugfix → chunk2 → parity gate

## Parity gate (definition of done)
Mojo reader's per-tensor (dtype, shape, nbytes, sha256(bytes)) == Python `safetensors.safe_open` for every tensor in Z-Image shards. 100% match.
