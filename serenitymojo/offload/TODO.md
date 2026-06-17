# offload / turbo / VMM — lib TODO

Backlog for the GPU memory/offload lib. Source: `AUDIT_TURBO_VMM_2026-06-09.md`
(read-only audit; runtime claims are HYPOTHESIS until measured on-GPU). Ordered by
value/risk. Keep items unchecked until a tool result / on-GPU measurement proves
the production claim (Tenet 4).

## P0 — needed before any "turbo helps" claim
- [ ] **Wall-clock + peak-VRAM gate.** Time copy-stream vs default-stream and read
      `vmm_cuda.cu_mem_get_info()` peak across a real run. Commit as a smoke. Settles
      phase-3 **P3** (`async_enabled()` is a hardcoded `return True`; overlap is
      currently structural-only). Until this exists, "turbo overlaps" is unproven.
      Source for the gate now exists at
      `serenitymojo/pipeline/klein_turbo_timed_gate.mojo`, with static contract
      guard `scripts/check_klein_turbo_timed_gate_contract.py`. The external
      evidence runner is `scripts/run_klein_turbo_timed_gate.py`; it runs a
      prebuilt gate binary, samples `nvidia-smi`, parses parity/timing output,
      and writes JSON. Keep this item unchecked until the gate is built/run
      through the capped GPU path and the measured wall-clock + VRAM report is
      archived.

## P1 — adopted overlap contract; timed benefit still open (Findings 1 & 2)
- [x] **Adopt the fixed overlap contract in the planned-loader inference loops.**
      The legacy `prefetch(0)` + `prefetch_next(bi)` (no-ctx) + `await_block(bi)`
      shape static-traced to a per-block synchronous fallback + a clobbered
      prefetch (HYPOTHESIS: zero overlap; matches the 2026-05-31 ~46s/step note).
      Current inference/model loops use the same pattern the `*_stack_lora.mojo`
      training stacks already use:
      ```mojo
      loader.prefetch_with_ctx(0, ctx)
      for bi in range(count):
          var handle = loader.await_block(bi, ctx)
          loader.prefetch_next_with_ctx(bi, ctx)
          ... run block bi ...
          loader.mark_active_block_done(ctx)
      ```
      Applied on 2026-06-16 to Klein sync/turbo forward_full + forward_full_cfg,
      HiDream, QwenImage, SenseNova, and Lance model loops. Static guard:
      `scripts/check_planned_loader_overlap_contract.py` passes. Klein sync-vs-turbo
      parity passed byte-exact (`cosine=1.0`, `MAD=0.0`) on
      `klein_turbo_parity_smoke`. This proves the loop is correct, not faster.
- [ ] **Time the adopted overlap contract.** Verify per-step time drops (needs P0)
      and record peak VRAM before any "turbo helps" production claim.
- [ ] **Confirm the slot-reuse fence under real overlap.** `mark_active_block_done`
      is now called by the adopted model loops, so `compute_done{0,1}` can be armed
      (`turbo_loader.mojo:448-453, :602-617`). Still stress slot reuse under a
      timed real-overlap gate and check parity before closing this.

## P2 — cleanups surfaced by the audit
- [x] **Internalize the SFv2/Rust VMM model-handle layer locally.**
      `serenitymojo/offload/vmm_manager.mojo` now owns `VmmModelHandle` and
      `VmmModelManager` over the existing `VmmSlabAllocator`, and
      `vmm_manager_smoke.mojo` passed on GPU. This is a primitive only: it proves
      local region ownership/refcount/evict/destroy, not Turbo speed or model
      generation.
- [ ] **Make the dtype guard real.** `TurboPlannedLoader.await_block` has a no-op
      `pass` where the header claims a "loud assertion" (`turbo_planned_loader.mojo:
      260-263`). Raise if `force_bf16` is set on a non-BF16-on-disk checkpoint instead
      of silently raw-copying. Source guard now exists in
      `TurboPlannedLoader._assert_raw_copy_dtype_safe`, and
      `scripts/check_turbo_dtype_guard_contract.py` enforces that it runs before
      resident pinning, prefetch copy, and await fallback copy. Keep unchecked
      until the capped Mojo build/gate proves the edited source in toolchain.
- [ ] **`OffloadConfig.single_pass()`** for the non-CFG inference path so it stops
      driving CFG-paired residency bookkeeping (phase-3 P2; `klein_dit.mojo:891`).
      Source API now exists, model non-CFG loops use it, and
      `scripts/check_offload_single_pass_contract.py` enforces the contract.
      Keep unchecked until the capped Mojo build/gate proves the edited source
      in toolchain.

## P3 — only if P0/P1 show streaming is the bottleneck
- [ ] **Wire `VmmModelHandle` as a real `TurboVmmBackend`.** `vmm_cuda.mojo`,
      `vmm_slab.mojo`, and `vmm_manager.mojo` now provide the local VMM substrate.
      `turbo_loader` still uses two fixed full-size device slabs (`:279-282`);
      VMM only pays off paired with the residency/eviction budget (currently idle
      at 128 GB, `turbo_planned_loader.mojo:127-130`) to hold N>2 resident blocks
      with eviction. Big lift — do not start before measurement justifies it. The
      local contract is captured in
      `serenitymojo/offload/STAGEHAND_TURBO_SOURCE_MAP_2026-06-16.md`.

## Notes
- A *new* from-scratch memory lib is NOT warranted — the substrate (VMM FFI, slab,
  model handle, async double-buffer, residency) already exceeds a greenfield lib.
  The gaps are **hot-path wiring + measurement**, not missing primitives.
- Full evidence + line cites: `serenitymojo/offload/AUDIT_TURBO_VMM_2026-06-09.md`.
