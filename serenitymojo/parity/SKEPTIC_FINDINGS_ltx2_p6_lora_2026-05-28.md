# SKEPTIC FINDINGS — LTX-2 P6 LoRA (additive at-dequant) — 2026-05-29

Verified by the skeptic against ground truth: lora_loader.rs (scale rule),
the live LoRA header, lora.mojo, models/dit/ltx2_dit.mojo, and the MVP pipeline.
All claims re-run / re-derived; frames viewed directly.

VERDICT: gates 1, 2, 3 reproduce and PASS as reported. NO fused/saved checkpoint
is written anywhere. The LoRA is genuinely ADDED at the dequanted block linear
per stream (re-applied each step), never fused. BUT there is one real,
underspecified gap (see CONCERN-1): the 28 GLOBAL LoRA modules are silently
NOT APPLIED at runtime, and the "fail-closed" guarantee covers only the 1632
block modules — not those 28 globals. The builder's "all 1660 applied" framing
is misleading: 1632 are applied, 28 are counted-but-dropped.

---

## 1. NO fused/saved checkpoint (the HARD RULE) — PASS

grep across lora.mojo, ltx2_dit.mojo, ltx2_block_stream.mojo, and the two
pipeline files for any serialize/save/write path: the only `save_*` is
`save_png` for the output frames. There is NO weight-serialization code in the
LoRA path at all. `add_delta_to` mutates an in-memory `ArcPointer[Tensor]`
(replaces it with `add(W, delta)`); nothing touches disk. CONFIRMED clean.

## 2. Applied at the DEQUANTED linear per stream (not fused once) — PASS

ltx2_t2v_av_mvp.mojo run() per-step loop (lines 614-624): each block is loaded
(`from_fp8_block(...).to_f32()` for inner blocks, `.load(...).to_f32()` for
boundary blocks) FRESH every step, then `lora.apply_to_av_block(i, w, ...)` is
called on that fresh dequant BEFORE the forward. The block weights are local to
the step iteration; nothing persists. So the delta is genuinely re-added to a
transient dequant each of the 8 steps × 48 blocks — exactly the at-dequant
discipline the rule demands. CONFIRMED.

## 3. Fail-closed key coverage — PASS for block keys, GAP for globals (CONCERN-1)

`apply_to_av_block` raises if a block-local LoRA key has no matching base linear
(`if not block.has_weight(local): raise`). I PROVED this with a mutation probe:
injected a bogus mapping `transformer_blocks.0.attn1.NONEXISTENT.weight` (real
A/B prefix so `_pair_present` passes) → apply RAISED:
  "LTX2 apply: block 0 has no base linear for LoRA key 'attn1.NONEXISTENT.weight'
   ... — fail-closed".
So the BLOCK path is genuinely fail-closed.

GATE-1 smoke re-run (verbatim): format LTX2Distilled; file A/B pairs 1660;
resolved mappings 1660; block-0 applied 34 == block-0 mapping count; total block
deltas 1632 + globals 28 == 1660. PASS.

>>> CONCERN-1 (the real gap): of the 1660 LoRA pairs, only 1632 (the 34/block ×
48 block modules) are EVER APPLIED. The remaining 28 are GLOBAL modules:
  patchify_proj, proj_out, audio_patchify_proj, audio_proj_out (4 projections)
  + 24 adaln_single linears across:
     adaln_single{.linear, .emb.timestep_embedder.linear_1/2},
     audio_adaln_single{...}, prompt_adaln_single{...},
     audio_prompt_adaln_single{...},
     av_ca_{video_scale_shift, audio_scale_shift, a2v_gate, v2a_gate}_adaln_single{...}.
(Enumerated directly from the file header — 28 non-block bases.)

These 28 are COUNTED by `ltx2_global_mapping_count()` to make the
"1632 + 28 == 1660" coverage tally pass, but NOTHING applies them:
 - The MVP pipeline loads patchify_proj/proj_out via `_load_global_f32` and the
   adaln tables via `_build_mod`/`_adaln_single` with NO LoRA hook.
 - There is no `apply_to_global` / equivalent. `apply_to_av_block` `continue`s
   (does NOT raise) on any mapping whose `_strip_block_prefix` returns ""
   (i.e. every global), so globals are silently dropped at RUNTIME, NOT
   fail-closed.
This contradicts the HARD RULE intent ("every LoRA key maps to a base linear;
raise on any unmapped/dropped key") for those 28 keys: they map to a base_key in
the LoraSet, but that base_key is never wired to a forward apply, and no raise
fires. The gate-1 tally is satisfied by COUNTING the globals, not by APPLYING
them — the count gate is not equivalent to a coverage gate here.

Severity: MODERATE. The 28 globals are the AdaLN modulation MLPs + the
patchify/unpatchify projections — these meaningfully shape the conditioning and
the latent<->token projection, so a fully-faithful LoRA application would
include them. The current run is "block-attention/FFN LoRA only." Gen is still
coherent and clearly LoRA-steered (the 1632 attn/ff deltas dominate the visible
effect), so this is not a correctness blocker for the MVP, but it IS an
incomplete application that the report did not disclose. RECOMMENDATION: either
(a) add a global-apply hook (apply the 28 deltas to the resident global linears
at load — these ARE persistent, so a one-time additive add is fine and still not
a saved fuse) and make the coverage gate APPLY-based not COUNT-based, or
(b) explicitly document that globals are intentionally excluded and assert the
global count is the ONLY unapplied set (so a future key family can't silently
join the dropped bucket).

## 4. Add-math vs host F64 — PASS

GATE-2 smoke re-run at block-0 attn1.to_q.weight (W=[4096,4096], rank=384,
scale=1.0): cos(lora_out, F64 ref W@x + scale·B@(A@x)) = 0.99999756;
max_abs(lora_out-ref)=0.0420, signal max_abs=9.732, normalized=0.43% (BF16-level);
delta magnitude (lora_out-base_out) max_abs=0.09375 ≠ 0 (delta genuinely
applied). The F64 reference is computed independently in the smoke (host loops
over A,B,W). The Mojo path computes delta=scale*(B@A) via linear(B, Aᵀ) and
adds it. CONFIRMED bit-close.

## 5. Coherent gen, deterministically different — PASS (frames viewed)

I VIEWED the frames directly (not just trusting md5):
 - base8/mvp_frame00: coherent dark space scene — glowing disc/portal top-right,
   purple light beam, starfield. Sharp, no NaN/gray/checkerboard.
 - lora8/mvp_frame00 + frame04: coherent industrial lattice-tower / bridge-crane
   structure against a purple cloudy sky with a bright sun. Sharp, recognizable.
All 9 frame pairs md5-DIFFER (deterministic, same seed/schedule). The LoRA
steered to a wholly different coherent subject — not noise, not a degenerate
perturbation. CONFIRMED.

## 6. Scale = alpha/rank * multiplier for rank-384 — CORRECT (for this file)

Ground truth lora_loader.rs:110-116 uses strength*(B@A) with NO alpha/rank
division. The Mojo `_module_scale` reads a per-module `<prefix>.alpha` if
present, else defaults alpha=module_rank → scale=multiplier. The live file has
ZERO `.alpha` tensors (header: 0 non-AB keys), and `__metadata__` reports
lora_alpha=384, lora_rank=384 (ratio 1.0). So scale=multiplier=1.0 is correct,
and gate-2 confirms scale=1.0. NOTE (minor): `_module_scale` ignores the
file-level `__metadata__` alpha/rank — harmless here (384/384=1) but would be
wrong for a hypothetical file with metadata alpha≠rank and no per-module .alpha
tensors. Not a defect for this LoRA.

---

## SUMMARY

| Gate | Claim | Skeptic result |
|------|-------|----------------|
| No fuse/save | none written | CONFIRMED — no serialize path exists |
| At-dequant per stream | re-applied each step | CONFIRMED — apply inside per-step block loop |
| Fail-closed coverage | every key maps, raises on miss | PASS for 1632 block keys (mutation-proven); 28 globals counted-but-not-applied and NOT fail-closed (CONCERN-1) |
| Add-math vs F64 | cos≥0.9999, delta≠0 | CONFIRMED cos=0.99999756, nmaxabs=0.43%, delta=0.094 |
| Coherent + different | both coherent, all frames differ | CONFIRMED (frames viewed) |
| Scale rule | mult (no alpha/rank div) | CORRECT for this file (alpha=rank=384→1.0) |

The work is sound and honors the HARD RULE (no fuse, additive at-dequant). The
one substantive issue is CONCERN-1: the report's "all 1660 applied / every key
maps" framing overstates coverage — 28 global modules are counted into the gate
but never applied at runtime and are not fail-closed. Recommend wiring or
explicitly excluding-and-asserting them before declaring full LoRA fidelity.
