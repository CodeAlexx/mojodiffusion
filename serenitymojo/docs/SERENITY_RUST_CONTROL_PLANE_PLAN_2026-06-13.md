# Serenity Rust Control Plane — Spec-First Plan (2026-06-13)

**Status:** PLAN (no code written). Authorized direction: replace the pure-Mojo daemon
*control plane* with a **purpose-built Rust server**; keep the **inference data plane in
pure Mojo + MAX**. Bespoke to our exact needs — NOT a generic web framework — and
optimized to be performant.

**Inputs that produced this plan (all this-session, measured):**
- Read-only multi-agent graph-runtime audit (23 agents, adversarially verified) →
  `serenitymojo/docs/graph_runtime_audit_2026-06-13.raw.json`.
- Rust control-plane feasibility analysis (code-grounded).
- Self-verified linchpin: `serenity_daemon.mojo` (3709 L) and `workflow_graph.mojo`
  (3064 L) contain **0** GPU/MAX/tensor references → pure control plane.

---

## 0. Why Rust, why now (one paragraph)

The two files that OOM the 62 GB desktop at build time (`mojo build` ~60 GB peak →
`oomd` kills the GNOME session) are **pure control plane with zero GPU code**. The
audit found the executor *logic* sound but every **confirmed daemon-killing bug** in the
hand-wired runtime layer (epoll/WS/fork) — the exact layer Rust replaces. tokio/axum/serde
structurally eliminate those bug *classes* (busy-spin, single-loop starvation, manual
framing, unguarded step) and `cargo` incremental compile never builds a 3.7k-line monolith
into a 60 GB peak. Inference is **already** driven as a separate fork+exec'd worker over a
language-agnostic AF_UNIX newline-JSON IPC, so a Rust parent drives the **same Mojo workers
unchanged**. Net: clean swap, ~M effort, dominated by faithfully porting the (sound) executor.

## 1. Scope

**In scope (Rust):** HTTP + WebSocket server, JSON, jobs DB, the workflow graph IR +
executor, model→backend dispatch/routing, the IPC parent driver, admission/validation,
job queue + GPU residency.

**Out of scope — stays pure Mojo + MAX (untouched):** all inference. The per-model backend
workers (`zimage_backend`, `ideogram4_backend`, `qwenimage_backend`, `klein_backend`,
`sample_cli_backend`, `video_api`, …) and everything behind the `GenBackend` trait.

**Non-goals (explicit):** a generic graph engine; a generic REST framework; a custom-node
plugin system. We build only our `/v1/*` surface and only our allowlisted node set.

**Orthogonal (do not conflate):** the EriDiffusion/flame Rust stack stays ONLY as the
inference **parity oracle** (a test reference). This server is a different thing.

## 2. Architecture

```
            HTTP/WS (axum + tokio)
client ───────────────────────────►  ┌──────────────────────────────┐
                                      │   SERENITY RUST CONTROL PLANE │
                                      │  • /v1/* endpoints            │
                                      │  • workflow graph IR+executor │
                                      │  • admission preflight        │
                                      │  • job queue + GPU residency  │
                                      │  • job-level output cache     │
                                      └───────────────┬──────────────┘
                                                      │ AF_UNIX socketpair,
                                                      │ newline-framed JSON
                                                      │ (encode_start / encode_ev)
                                                      ▼
                                      ┌──────────────────────────────┐
                                      │  MOJO INFERENCE WORKER (kind) │  ← UNCHANGED
                                      │  fork+exec: `serenity_daemon  │
                                      │  worker <kind> <fd>`          │
                                      │  GenBackend.start/step/cancel │
                                      └──────────────────────────────┘
```

Single async server process; **one resident GPU model at a time** (residency switch =
SIGKILL child + waitpid so the OS reclaims VRAM — the only reclaim that works); **serial
job queue** (single GPU). The worker binary stays the Mojo build; the Rust server execs it.

## 3. Hard compatibility contracts (format fidelity = the real risk)

These MUST be reproduced byte-exact or gallery/parity break. Re-validated for free by the
existing Python gates (`check_workflow_graph_product_contract.py`,
`check_klein_lora_daemon_smoke.py`).

1. **`serenity.genparams.v1`** — PNG `tEXt` genparams written by the worker, read by
   `/v1/gallery`. The IPC already carries `params_json`; the Rust side must emit identical bytes.
2. **`serenity.workflow_graph.v1`** — the typed workflow graph schema accepted by the importers.
3. **`JobParams` ~60-field wire set** — `encode_start` / `decode_start`. In Rust this becomes
   **ONE `serde` struct = single source of truth** (today the field list is hand-maintained in
   ≥4 places that must stay byte-aligned — cross-cutting fix).
4. **`/v1/*` endpoint surface** — enumerate the exact 12 endpoints from `serenity_daemon.mojo`
   and reproduce request/response shapes verbatim.

## 4. Component specs (every requirement traces to an audit finding)

### 4.1 HTTP/WS layer — axum + tokio (replaces hand-wired epoll/WS)
- **R-NET-1** Per-connection async tasks; no single shared event loop that one client can
  starve. *(kills the `send_all_fd` EAGAIN busy-spin class — confirmed HIGH.)*
- **R-NET-2** WS progress fan-out is best-effort with backpressure: bounded per-client
  channel; on slow/dead client, drop frames / evict the client, never block generation.
  *(confirmed HIGH: a paused tab currently freezes all generation.)*
- **R-NET-3** 1 MiB body cap preserved; integer parsing overflow-safe (serde gives this;
  the Mojo `json/parser.mojo v=v*10+d` had no guard).
- **R-NET-4** Real graceful shutdown (tokio signal) — replaces best-effort signalfd.

### 4.2 Graph executor — faithful port of the SOUND Mojo executor + additive improvements
- **R-EXE-1 (preserve)** Keep the verified fail-loud spine verbatim in semantics: strict
  node allowlist, per-edge from/to validation, duplicate-node-id + duplicate-SetNode-name
  detection, cycle/no-progress trap, `MAX_DEPTH` cap, typed-input requirement checks.
- **R-EXE-2 (perf)** Replace the rescan-until-stable worklist with **Kahn topological order**.
  Precompute once: `id→index`, `(to_node,to_port)→link`, `(node,port)→value-index`.
  *(fixes confirmed O(N²·E), worst O(N³) → O(N+E); ComfyUI `ExecutionList` analogue.)*
- **R-EXE-3 (correctness guard)** A `nodes` body with no `edges` must **501**, never fall back
  to title-sniffed txt2img. *(confirmed HIGH legacy adapter.)*
- **R-EXE-4 (correctness guard)** >1 terminal sampler/output, or two writers to the same flat
  scalar field, must **501** — no first-writer-wins flatten. *(confirmed HIGH.)*
- **R-EXE-5 (correctness)** Type acceptance = tokenize-on-comma + **exact-equality** membership,
  not substring `.find()`. *(confirmed MEDIUM silent mis-route.)*
- **R-EXE-6 (correctness)** Typed scalar handles (GetImageSize INT, GetNode-carried scalars)
  must have a real scalar backing or the importer must resolve them — no advertised handle that
  no path can consume. *(confirmed MEDIUM; graphs ComfyUI runs trivially currently 501.)*
- **R-EXE-7 (UX, ComfyUI `execution_error` analogue)** Per-node error **attribution**: every
  raise carries the offending node id + reason so the editor can point at the broken node.
- **R-EXE-8 (parity)** Honor `mode==2` (mute) as inactive in the canvas importer (today only
  `mode==4` bypass is skipped → muted nodes run); lazy-evaluate `ComfySwitchNode` (require only
  the selected branch). *(ComfyUI `check_lazy_status` analogue.)*

### 4.3 Admission preflight — validate-before-run (ComfyUI `validate_prompt` analogue)
- **R-ADM-1** Before enqueue, classify the model via **ONE** authoritative `model→kind` router
  (today three divergent substring classifiers exist) and reject disabled/unknown models with a
  synchronous **501/422**. *(confirmed HIGH: SD3/Qwen/Flux/LTX/Wan return 200-then-fail.)*
- **R-ADM-2** Reject backend-incompatible feature metadata (mask / qwen-edit / lanpaint on a
  model that can't do it) at request time, before the job occupies the serial queue.

### 4.4 Job-level output cache (ComfyUI `IsChanged`/`HierarchicalCache`, flat-params variant)
- **R-CACHE-1** Hash the canonical flattened `JobParams`
  (model+prompt+negative+seed+steps+sampler+scheduler+cfg+size+init/mask/lora…); on exact hit,
  return the prior output without re-running the worker. *(biggest interactive win; today every
  `/v1/generate` re-runs full lowering + full forward even when nothing relevant changed.)*
- DEFER (v2): node-level output caching + partial/output-driven execution (ComfyUI's deepest
  wins) — they require the executor to own per-output tensor handles; our flatten-to-JobParams
  model doesn't support it yet. Documented non-goal for v1.

### 4.5 IPC parent driver (nix + serde_json) — drive the EXISTING Mojo workers
- **R-IPC-1** `socketpair → fork → execv("serenity_daemon", ["worker", kind, child_fd])`;
  parent fd `O_NONBLOCK`. Wire = newline-framed JSON: parent→child `{"cmd":"start",…}` /
  `{"cmd":"cancel"}`; child→parent `ready|progress|done|failed|cancelled`.
- **R-IPC-2 (safety)** In the fork child, before exec, set `setpgid(0,0)` + `PR_SET_PDEATHSIG,
  SIGKILL` so sample-CLI **grandchildren die with the worker**. *(confirmed HIGH GPU orphan;
  restores the kill-child→reclaim-VRAM guarantee for ALL kinds.)*
- **R-IPC-3 (safety)** Model switch / cancel = SIGKILL child + blocking waitpid (VRAM reclaim).
  Track child PID so **/v1/cancel actually interrupts in-flight work** (SIGTERM/SIGKILL), not
  at the next step boundary. Send `cancel` once per job, not every tick.
- **R-IPC-4 (safety)** A malformed child line must **never kill the server**: decode in a
  task-local try; on error, fail that job (mark failed, broadcast, persist), keep serving.
  *(this is the Rust structural answer to the confirmed CRITICAL unguarded `backend.step()`.)*

### 4.6 Jobs DB + lifecycle
- **R-DB-1** Persist JobRecords; **evict terminal records from the in-memory list** after
  persistence (PNG + tEXt already on disk) to bound host-RAM on a long-lived daemon.
- **R-DB-2** Async/batched DB writes off the request path.

### 4.7 One error model (cross-cutting)
- **R-ERR-1** A typed `Error` enum → HTTP status mapping. Replaces the fragile literal
  `'[501] '` string-prefix byte-slicing where a forgotten prefix silently becomes a 422.

## 5. Performance spec (the "PERFORMANT" requirement, concrete)
- O(N+E) lowering (R-EXE-2); job-level output cache (R-CACHE-1).
- Non-blocking WS fan-out (R-NET-1/2); single serde parse, no re-parse per pass.
- Zero-copy image handoff preserved: worker writes PNG to disk, server returns the path
  (no large-buffer marshaling across IPC).
- Serial GPU queue with next-job prep overlapped against current decode where safe.
- Targets: set a baseline against the current Mojo daemon first, then hold ≤ that latency
  on cold path and ≫ better on warm/cached path. (No fabricated numbers; measure first.)

## 6. Migration plan (low-risk, parity-gated)
- **Phase A — Spike (prove the seam):** Rust axum skeleton + IPC driver (R-IPC-1/2/3). Implement
  `/v1/generate` + `/v1/progress` for ONE kind (zimage), driving the existing Mojo zimage worker
  unchanged. Pass the Python daemon smoke gate. *Exit:* a Rust server produces a byte-identical
  zimage PNG (genparams.v1) vs the Mojo daemon.
- **Phase B — Port the executor + surface:** faithful executor port (R-EXE-1..8), all node types,
  all `/v1/*` endpoints, admission preflight (R-ADM), one error model (R-ERR), one JobParams serde
  struct. *Exit:* all existing Python parity/product-contract gates green against the Rust server.
- **Phase C — Performance + cutover:** job-level cache (R-CACHE-1), DB eviction (R-DB), real cancel
  (R-IPC-3). Run Rust server alongside Mojo daemon; cut SerenityUI over; retire the Mojo daemon
  build (kills the desktop-OOM dev hazard for good).
- Mojo workers still build occasionally → keep `build-daemon-safe` / `scripts/mem_safe.sh`.

## 7. Reuse / what's net-new
- **Reusable pattern:** `EriDiffusion/.../crates/inference/server.rs` (axum 0.7 + tokio + serde_json
  skeleton: router, ServerState, request structs) — copy the shape, not a drop-in (it has no graph
  executor and no AF_UNIX driver).
- **Net-new in Rust:** the workflow-graph executor (the bulk) and the AF_UNIX worker-spawn driver.

## 8. Open questions
- Crate layout: one binary, or `serenity-server` + `serenity-graph` + `serenity-ipc` crates
  (favored — keeps the executor independently testable and compile-isolated).
- Do we keep the pure-Mojo SQLite, or move jobs.db to a Rust sqlite crate? (Leaning Rust `rusqlite`
  for the server; the on-disk schema is the contract, not the library.)
- Whether to patch the current Mojo daemon's CONFIRMED CRITICAL (`backend.step()` guard) in the
  interim, given building it risks the desktop and it's being retired. **Recommendation: do not —
  fold every confirmed bug into this spec as a structural requirement; don't invest in the
  throwaway.** (User call.)

---

### Appendix A — Audit confirmed-bug → requirement traceability
| Confirmed bug (verified) | Sev | Rust requirement |
|---|---|---|
| Unguarded `backend.step()` kills daemon on bad child line | CRIT | R-IPC-4 (structural) |
| `send_all_fd` EAGAIN busy-spin freezes loop | HIGH | R-NET-1/2 |
| sample-CLI grandchild GPU orphan on SIGKILL | HIGH | R-IPC-2 |
| no-edges graph → title-sniffed txt2img | HIGH | R-EXE-3 |
| multi-sampler first-writer-wins flatten | HIGH | R-EXE-4 |
| disabled models 200-then-fail post-enqueue | HIGH | R-ADM-1 |
| O(N²·E) worklist re-parse | MED | R-EXE-2 |
| substring type acceptance mis-route | MED | R-EXE-5 |
| GetImageSize/scalar handle has no backing | MED | R-EXE-6 |
| dead duplicate executor in daemon (675-964) | MED | dropped on port (not carried) |

### Appendix B — ComfyUI ideas adopted vs deferred
Adopted v1: per-node error attribution (S), validate-before-run (S), topological ExecutionList
(M), job-level output cache (M), lazy switch branch (S), mute-mode honoring (S).
Deferred v2: node-level output caching + dirty-subtree re-exec (L), partial/output-driven
execution (L) — require per-output tensor handles our flatten model lacks.
