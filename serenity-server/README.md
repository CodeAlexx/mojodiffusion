# serenity-server — Rust control plane for serenitymojo

Purpose-built Rust replacement for the pure-Mojo daemon's **control plane**
(HTTP/WS/JSON/jobs/graph/dispatch). It drives the **unchanged Mojo inference
workers** over the existing AF_UNIX newline-JSON IPC. Inference stays pure Mojo+MAX.

Lives inside the Mojo repo (`github.com/CodeAlexx/mojodiffusion`), NOT any Python
"serenity" tree. Design: `serenitymojo/docs/SERENITY_RUST_CONTROL_PLANE_PLAN_2026-06-13.md`.

## Crates
- **`wire`** — FROZEN IPC contract (1:1 with `serenitymojo/serve/ipc_codec.mojo` +
  `backend.mojo`). `JobParams`, `WorkerEvent`, `CANCEL_LINE`. Do not edit field
  names/types/defaults without changing the Mojo side in lockstep. `cargo test -p serenity-wire`.
- **`ipc`** — spawn + drive a worker (re-host of `proc_ipc.mojo` +
  `process_isolated_backend.mojo` parent half).
- **`server`** — axum HTTP/WS, job queue, worker-driver thread.

## Phase A goal (current)
A Rust server drives the **already-built** `output/bin/serenity_worker_stub`
(CPU-only, 2.4 MB, built via `pixi run build-worker-stub-safe`) end-to-end:
`POST /v1/generate` → worker runs → `GET /v1/progress` (WS) streams events →
a PNG with a `serenity.genparams.v1` tEXt chunk lands in the out dir.

## Rules for contributors (human or agent)
- `cargo build` / `cargo test` are fine (incremental, never OOMs).
- **NEVER run `mojo build` or `pixi run build-*`** — building Mojo on this box has
  OOM-killed the GNOME session. The stub worker is already built; Mojo builds are
  done by the orchestrator only, capped (`build-worker-stub-safe`).
- Code against the FROZEN `wire` + `ipc` public APIs; if a signature must change,
  document it rather than editing across crate boundaries mid-parallel-work.

## Run (once implemented)
```
cargo build
./target/debug/serenity-server --worker ../output/bin/serenity_worker_stub --out-dir /tmp/serenity_out
# then: POST /v1/generate {model:"stub",prompt:"hi",steps:6,width:64,height:64} ; WS /v1/progress?job=<id>
```
