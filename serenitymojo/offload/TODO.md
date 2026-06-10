# offload / turbo / VMM — lib TODO

Backlog for the GPU memory/offload lib. Source: `AUDIT_TURBO_VMM_2026-06-09.md`
(read-only audit; runtime claims are HYPOTHESIS until measured on-GPU). Ordered by
value/risk. **Nothing here is started.** Do NOT mark an item done without a tool
result / on-GPU measurement showing it (Tenet 4).

## P0 — needed before any "turbo helps" claim
- [ ] **Wall-clock + peak-VRAM gate.** Time copy-stream vs default-stream and read
      `vmm_cuda.cu_mem_get_info()` peak across a real run. Commit as a smoke. Settles
      phase-3 **P3** (`async_enabled()` is a hardcoded `return True`; overlap is
      currently structural-only). Until this exists, "turbo overlaps" is unproven.

## P1 — make turbo inference actually overlap (Findings 1 & 2)
- [ ] **Adopt the fixed overlap contract in the Klein inference loops.** They use the
      legacy `prefetch(0)` + `prefetch_next(bi)` (no-ctx) + `await_block(bi)` shape
      (`klein_dit.mojo:980-983, :1071-1074, :713-716, :806-809`), which static-traces
      to a per-block synchronous fallback + a clobbered prefetch (HYPOTHESIS: zero
      overlap; matches the 2026-05-31 ~46s/step note). Swap to the pattern the
      `*_stack_lora.mojo` training stacks already use:
      ```mojo
      loader.prefetch_with_ctx(0, ctx)
      for bi in range(count):
          var handle = loader.await_block(bi, ctx)
          loader.prefetch_next_with_ctx(bi, ctx)
          ... run block bi ...
          loader.mark_active_block_done(ctx)
      ```
      Apply to `forward_full` + `forward_full_cfg` in BOTH `Klein9BOffloadedTurbo` and
      `Klein9BDiT`. Verify per-step time drops (needs P0) and output parity vs the
      synchronous `PlannedBlockLoader` (oracle: `klein_turbo_parity_smoke`).
- [ ] **Confirm the slot-reuse fence under real overlap.** `mark_active_block_done`
      is called 0× on the inference path, so `compute_done{0,1}` is never armed
      (`turbo_loader.mojo:448-453, :602-617`). Latent race today (masked by the
      serial fallback). After P1 above, stress slot reuse and check parity.

## P2 — cleanups surfaced by the audit
- [ ] **Make the dtype guard real.** `TurboPlannedLoader.await_block` has a no-op
      `pass` where the header claims a "loud assertion" (`turbo_planned_loader.mojo:
      260-263`). Raise if `force_bf16` is set on a non-BF16-on-disk checkpoint instead
      of silently raw-copying.
- [ ] **`OffloadConfig.single_pass()`** for the non-CFG inference path so it stops
      driving CFG-paired residency bookkeeping (phase-3 P2; `klein_dit.mojo:891`).

## P3 — only if P0/P1 show streaming is the bottleneck
- [ ] **Wire `VmmSlabAllocator` as a real `TurboVmmBackend`.** `vmm_slab.mojo` /
      `vmm_cuda.mojo` are complete but referenced nowhere (parked). `turbo_loader`
      uses two fixed full-size device slabs (`:279-282`); VMM only pays off paired with
      the residency/eviction budget (currently idle at 128 GB, `turbo_planned_loader.
      mojo:127-130`) to hold N>2 resident blocks with eviction. Big lift — do not start
      before measurement justifies it. (Doc backend step 4:
      `docs/FULL_PORT_OFFLOAD_TURBO_2026-05-27.md:196-217`.)

## Notes
- A *new* from-scratch memory lib is NOT warranted — the substrate (VMM FFI, slab,
  async double-buffer, residency) already exceeds a greenfield lib. The gaps are
  **adoption + measurement**, not missing primitives.
- Full evidence + line cites: `serenitymojo/offload/AUDIT_TURBO_VMM_2026-06-09.md`.
