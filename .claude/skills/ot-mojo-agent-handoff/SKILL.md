---
name: ot-mojo-agent-handoff
description: Standard handoff and multi-agent workflow for OneTrainer-to-Mojo model work. Use when coordinating Codex, Claude Code, or subagents so build, verification, and documentation stay consistent.
---

# ot-mojo-agent-handoff

Use this when delegating or handing work between Codex and Claude Code.

## Shared Rules

- Do not revert other agents' changes.
- Own a disjoint file set per agent.
- Reference OneTrainer source paths and local artifact paths explicitly.
- Keep strict gates strict; expected failures are useful evidence.
- Lead agent reruns commands before accepting another agent's claim.
- Update docs/status with limits, not optimism.

## Standard Three-Agent Split

Use this when the user asks for agents or during a main-loop verification push:

1. Builder:
   - implements one bounded slice
   - owns named files
   - runs compile/static smoke
2. Verifier:
   - builds/runs parity or readiness gates
   - writes exact pass/fail/blocker output
   - does not weaken tests
3. Skeptic:
   - audits dtype boundaries, false claims, OneTrainer naming, speed/VRAM
   - looks for missing artifacts and stale docs
   - writes blockers, not broad refactors

## Handoff Packet

Every handoff must include:

- current objective and model target
- files changed
- commands run and exit codes
- generated artifacts and paths
- accepted evidence level
- exact blockers
- next command to run
- whether GPU is free or a process is still active

## File Naming And Mapping

For every model, keep a visible OT-to-Mojo map:

- OneTrainer file/class/function
- Mojo file/function
- parity script or manifest
- artifact path
- acceptance status

If a local file name cannot match OneTrainer exactly, include the OneTrainer
name in the manifest/status text.

## Stop Conditions

Stop and report instead of guessing when:

- a required reference artifact is missing
- CUDA/device setup fails
- strict dtype would require an F32 boundary
- OneTrainer source contradicts the local implementation
- a long run risks OOM without a bounded smoke
