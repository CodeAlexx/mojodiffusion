# SKEPTIC FINDINGS — LTX-2 P5 full 48-block AV DiT stack (`dit_stack`)

**Date:** 2026-05-28  •  **Auditor:** SKEPTIC (independent)  •  **GPU:** RTX 3090 Ti 24 GB
**Subject:** Builder's claim for the full `forward_audio_video` 48-block stack — "AUDIO PASS (0.99983), VIDEO 0.99470 (below 0.999), honest verdict = does NOT fully pass."

## Verdict

**The builder's report is HONEST and ACCURATE. The PASS/FAIL split is real:**
- **AUDIO velocity cos = 0.99983 — PASS** (≥ 0.999), reproduced.
- **VIDEO velocity cos = 0.99470 — FAIL** (below 0.999), reproduced exactly.
- The gate **fails closed** (raises `VIDEO velocity parity FAIL`, exit 1).

Per the HARD RULE (cos ≥ 0.999 on the named oracle), **P5 dit_stack does NOT pass** — video is short by 0.005. The shortfall is a **deep-chain numerical floor (FP32 GEMM reduction-order drift), not a structural bug** — proven by per-block localization below. The 48-block assembly, FP8 streaming, boundary/inner residency, cross-modal paths, connector, per-block modulation indexing, and output stage are all **structurally correct**.

## What was audited (vs Rust spec `ltx2_model.rs:4453-5040`)

1. **All 48 blocks run** (not a subset). Confirmed: smoke loops `for i in range(48)`; each block loaded, run, dropped. Oracle (`ltx2_dit_forward_parity_ref.py`) also loops all 48. v_std checkpoints at blocks 12/24/36/48 track the oracle to <0.2% (Mojo 57.699 vs oracle 57.797 at block 48).
2. **Boundary-BF16 vs inner-FP8 handling — CORRECT.** Verified against the checkpoint header directly: blocks {0,1,2,3,47} store `attn1.to_q.weight` as **BF16** (no `weight_scale`); inner blocks 4-46 store **F8_E4M3** with per-tensor scales. `_is_boundary(i): i in {0,1,2,3,47}` matches. Boundary → `LTX2AVBlockWeights.load` (BF16). Inner → `LTX2BlockStream.load_block_bf16` (FP8 dequant via per-tensor scale) → `from_fp8_block`. Both `.to_f32()` for F32 block compute, matching the F32 oracle (oracle dequants FP8 inner identically: `t.f32()*scale → bf16 → f32`).
3. **Per-block modulation indexing — CORRECT.** `_ada_row_pertok` slices video table rows 0-5 (MSA shift/scale/gate, MLP shift/scale/gate) + 6-8 (CA shift/scale/gate); cross-modal `_compute_cross_mod` splits `scale_shift_table_a2v_ca_{video,audio}[5,*]` rows 0/1→a2v, 2/3→v2a, 4→gate. Matches Rust `compute_cross_attn_params` and the P3 block-0 skeptic finding.
4. **Block math** matches the Python `run_block` line-for-line: gated SDPA (`sigmoid(gate_logits)*2`, per-head, post-SDPA pre-`to_out`), attn2 NO-RoPE, cross-modal `to_out` widths (a2v→4096, v2a→2048), RoPE swap (a2v: q=ca_v/k=ca_a; v2a: q=ca_a/k=ca_v), gelu-tanh FFN.
5. **Output stage** (`layer_norm_no_affine → (scale_shift_table[2,dim]+embedded) modulate → proj_out`) matches Rust §9/§10.

## Fail-closed verification (mutation)

Mutated the inner loop to load **block 5's weights at position 25** (wrong per-block index):
| Run | VIDEO cos | AUDIO cos | Gate |
|---|---|---|---|
| clean baseline | 0.99470 | 0.99983 | FAIL (video) |
| block-25 ← block-5 weights | **0.81555** | **0.96531** | FAIL (both) |

Both streams collapse sharply → the gate is **not vacuous**; per-block weight indexing is load-bearing; the audio PASS is genuine (not spurious — audio drops to 0.965 under the bug).

## Per-block localization (the decisive test)

Dumped per-block VIDEO hidden state from both oracle and Mojo, computed cosine at every block:

| block | vhs_cos | | block | vhs_cos |
|---|---|---|---|---|
| 0 | 0.9999989 | | 32 | 0.9998474 |
| 10 | 0.9999961 | | 38 | 0.9996737 |
| 22 | 0.9999927 | | 41 | 0.9991690 |
| 25 | 0.9999312 | | 44 | 0.9969159 |
| 31 | 0.9998784 | | 46 | 0.9961313 |

**Smooth, monotonic drift accumulation — no sudden jump at any block.** Block 0 is essentially exact (0.9999989; consistent with the P3 single-block gate of 0.99995). The drift compounds gradually and steepens in the last ~6 blocks where v_std spikes 8.2→57.7 (magnitude growth amplifies any directional error). This is the signature of **FP32 reduction-order divergence (vendor cuBLAS vs torch)** — a structural bug would drop sharply at one block (as the block-25 mutation showed: 0.9947→0.8156).

**Why audio passes but video fails:** audio attention is Dh=64; video is Dh=128 over a 4096-dim residual. The deeper/wider video GEMMs accumulate more reduction-order error over 48 layers. `NVIDIA_TF32_OVERRIDE=0` (builder-reported) does not change the result — the matmul was already FP32, so this is reduction order, not TF32.

## Notes / residual risk

- The smoke ingests the oracle's pre-computed modulation/RoPE tensors verbatim (isolates the STACK assembly, which is the P5 deliverable; timestep-MLP and RoPE are gated at P2.5/P3). This is a legitimate scoping per the plan, but means P5 does **not** re-gate timestep-MLP / RoPE derivation end-to-end.
- The audio context (`audio_pre`) in the oracle is a seeded synthetic tensor, not a real projected audio context — fine for a numerical stack gate, but the real audio-context path is exercised only at P7.
- The plan (P7, risk #1) anticipates exactly this floor and uses a looser 0.99 for multi-forward chains; the video stack at 0.9947 would clear a 0.99 bar but not the stated 0.999.

## Bottom line

**Structurally GREEN, numerically RED on video.** No structural defect found; the gate correctly fails closed; fail-closed is verified non-vacuous on both streams. Closing the last 0.005 on video requires matching the video-attention FP32 GEMM reduction order to torch (kernel-level work), or accepting the documented deep-chain floor as the plan does for P7.

**Reproduce:**
- oracle: `python3 scripts/ltx2_dit_forward_parity_ref.py`
- gate: `NVIDIA_TF32_OVERRIDE=0 pixi run mojo run -I . serenitymojo/pipeline/ltx2_dit_forward_smoke.mojo`
