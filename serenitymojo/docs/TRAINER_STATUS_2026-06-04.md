# Trainer status — ground truth (2026-06-04)

This is the authoritative per-trainer status. It supersedes status claims in
PORT_GAP_AND_PLAN_2026-06-03.md / FINAL STATE, which proved stale on several
points. Status here comes from (a) the maintainer's direct confirmation and
(b) measured runs in-session — NOT from older planning docs.

## The distinction that matters

- **RUNS** = compiles + completes training steps without OOM. The mechanism
  executes end to end. Does NOT prove the model is learning.
- **TESTED WORKING** = a real run (real sigma schedule, enough steps) shows the
  loss **drop meaningfully** AND a generated sample **visibly shift** base→LoRA.
  This is the only thing that proves training actually learns.

A trainer can RUN without being TESTED WORKING. Do not conflate them.

## Status

| Trainer | Status | Evidence / note |
|---|---|---|
| anima   | WORKING (maintainer-confirmed) | worked on and working |
| ernie   | WORKING (maintainer-confirmed) | worked on and working |
| klein   | WORKING (maintainer-confirmed) | worked on and working |
| zimage (regular) | WORKING (maintainer-confirmed) | regular Z-Image works |
| chroma  | **RUNS — NOT yet tested-working** | 3-step bf16 smoke this session: completes, no OOM, peak host RSS 54 GB, loss 0.41559148→0.41556543 (Δ ~3e-5, i.e. tiny), LoRA-B grew 0→6255, grad_norm 0.0017–0.0025, no nonfinite. NO meaningful loss drop and NO base-vs-LoRA sample comparison yet → not proven to learn. |
| sd35    | IN PROGRESS | offload trainer being built (has checkpoint sd3.5_large + cache andrsd35ver1 → can run for real) |
| wan22   | CODED, NEVER TESTED | needs to be worked on / run |
| ltx2    | CODED, NEVER TESTED | needs to be worked on / run |
| qwenimage | CODED, NEVER TESTED | needs to be worked on / run |
| L2P (Z-Image pixel-space) | UNTESTED | reuses the working Z-Image DiT body; pixel-space L2P path never run. Checkpoint: checkpoints/L2P/model-1k-merge.safetensors |
| acestep | UNCONFIRMED | status not yet verified |

## bf16 carrier fix (done this session)

Independent of the per-trainer status above: the F32→bf16 carrier divergence
that OOM'd the chroma offload trainer is fixed and gated. flame-core is bf16
in/out; the port had leaked F32 via Tensor.to_host (upcast). Fix = to_host_bf16
/ from_host_bf16 carriers + native bf16 block compute (local F32 casts only at
rope_backward / cat_backward / gate_residual_backward). Applied to the shared
Flux block (chroma/sd35/l2p), wan22, ltx2, acestep blocks; re-gated vs bf16
oracle (cos 0.998–1.0). See ../../.claude memory project_bf16_carrier_fix.

## What "tested working" requires next

For chroma (and then each trainer): a longer real run — real sigma schedule,
enough steps to move the loss — that produces (1) a clear downward loss curve
and (2) a viewable base-vs-LoRA sample showing the LoRA changed the output.
Until that exists for a given trainer, it is RUNS at best, not WORKING.
