# Ideogram4 trainer — flash SDPA wiring (mapped + WIRED + BLOCKED 2026-06-20)

## ⛔ MEASURED BLOCKER (2026-06-20): Dh=256 unsupported by the cuDNN flash BACKWARD
Wired end-to-end (Ideogram4LoRABlock.mojo, behind `IDEOGRAM4_SDPA_FLASH`, gated OFF).
The flash **forward** runs at Dh=256; the **backward** fails:
`sdpa_flash_backward_f32: shim rc=-1 (B=1 S=1280 H=18 Dh=256)`. Root cause (measured):
`serenitymojo/ops/cshim/cudnn_sdpa_bwd.cpp:5` — the frontend supports only
**head_dim ∈ {64, 96, 128}**; `g->build({HeurMode_t::A})` finds no plan for D=256.
klein/zimage use Dh=128 → flash works; **ideogram4 Dh=256 = 4608/18 does not.**
Flash is therefore NOT viable for ideogram4's backward (the slow part) without either a
cuDNN/frontend upgrade that adds a D=256 backward plan, or a custom Dh=256 flash-bwd kernel.
Wiring kept compiled-but-off (C13). The dtype boundary fix that WAS needed and works: ideogram4
is BF16 end-to-end, so cast the `_f32` flash output + grads back to BF16 (`cast_tensor`).

Remaining real lever for ideogram4's host-bound wall = the autograd_v2 graph-capture port
(SDPA-agnostic; the ~9% multi-phase campaign per MOJO_V2_ENGINE_PLAN.md).

---

# Original recipe (mapped 2026-06-20)

**Goal:** replace the custom Mojo SDPA (`sdpa_nomask`/`sdpa_backward`) in the
ideogram4 LoRA trainer with cuDNN **flash** SDPA. Highest-ROI speed lever
(measured): the custom SDPA is many kernels (host-bound contributor) + a slow
51 ms GPU backward; flash collapses both to one kernel (klein: 100.7 ms → 2.34 ms,
35–43×). Precedent: klein `single_block.mojo` `KLEIN_SDPA_FLASH` (lines 1103, 1464),
zimage `lora_block.mojo` `ZIMAGE_SDPA_FLASH`. Approved numerics change.

## Viability (confirmed)
- S (sequence) = NT(256) + NIMG(GH·GW = 32·32 = 1024) = **1280 = 10×128** → flash's
  `S % 128 == 0` requirement holds for the giger 512px cache. (A different
  res/bucket with non-128-aligned S would need `sdpa_flash_train_fwd_rect` or padding.)
- Heads=18, Dh=256 (cuDNN flash max head_dim) — OK.

## Flash API (serenitymojo/ops/attention_flash.mojo)
- `sdpa_flash_train_fwd_f32[B,S,H,Dh](q, k, v, scale, ctx) -> SdpaFlashF32Fwd`
  - `.att` = F32 output [B,S,H,Dh]; also returns bf16 q_bf/k_bf/v_bf, o_pad, F32 stats.
- `sdpa_flash_backward_f32[B,S,H,Dh](q_bf, k_bf, v_bf, o_bf, stats, d_att, scale, ctx) -> SdpaFlashGrads`
  - `.d_q/.d_k/.d_v` (F32), same semantics as `sdpa_backward` (grads wrt q_rope/k_rope/v).

## Code changes — serenity-trainer/src/serenity_trainer/model/Ideogram4LoRABlock.mojo
1. **Imports + flag** (near line 25/44):
   `from serenitymojo.ops.attention_flash import sdpa_flash_train_fwd_f32, sdpa_flash_backward_f32`
   (+ `SdpaFlashF32Fwd`, `SdpaFlashGrads` if needed). `comptime IDEOGRAM4_SDPA_FLASH = False` (C13 default-off).
2. **Struct `Ideogram4BlockActs`** (line 285): append 5 trailing fields
   `var flash_q/flash_k/flash_v/flash_o/flash_stats: Optional[TArc]` with `= None` defaults in
   `__init__` (keeps the 34-arg positional construction at line 468 working).
3. **Forward** (line 448) — `comptime if IDEOGRAM4_SDPA_FLASH:`
   `var ff = sdpa_flash_train_fwd_f32[1,S,Heads,Dh](q_rope, k_rope, v, scale, ctx)`;
   `attn4 = ff.att`; capture ff.q_bf/k_bf/v_bf/o_pad/stats into locals; after building
   `acts` (line 503) set `acts.flash_q = Optional(flash_q)` … (else branch keeps `sdpa_nomask`).
4. **Backward** (line 610) — `comptime if IDEOGRAM4_SDPA_FLASH:`
   `var sd = sdpa_flash_backward_f32[1,S,Heads,Dh](acts.flash_q.value(), acts.flash_k.value(),
   acts.flash_v.value(), acts.flash_o.value(), acts.flash_stats.value(), d_attn4, scale, ctx)`
   (fail-loud if a flash field is None = fwd/bwd flag mismatch, per klein). `sd.d_q/d_k/d_v` then
   feed the unchanged `rope_halfsplit_full_backward` + `rms_norm_backward_dx`.

## Build (serenity-trainer) — add the cuDNN shim link args
Append to the trainer build command:
`-Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa` (the .so is at
/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib/libserenity_cudnn_sdpa.so; rebuild via
mojodiffusion/serenitymojo/ops/cshim/build.sh if missing). Keep `--optimization-level 2 -j 4`.

## Runtime (the documented gotcha)
LD_LIBRARY_PATH must include BOTH:
- `/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib` (the shim), AND
- `~/.local/lib/python3.12/site-packages/nvidia/cudnn/lib` (real cuDNN v9).
The EriDiffusion `.venv_cache` cuDNN wheel FAILS ("No valid execution plans") — do not use it.

## Gate (flash is a numerics change — NOT byte-identical to math anchors)
- Op-level parity already gated (ops/tests/sdpa_flash_parity.mojo: cos ≥ 0.9999967).
- Trainer: verify 5-step loss stays in the value-class near the math anchors
  1.12493/1.141433/0.73936844/1.2301777/0.8488946 (close, not bit-equal — flash bwd dQ is
  nondeterministic run-to-run → 4dp value-class, NOT a bit gate).
- MEASURE s/step (vs 5.2 s math) — expect a drop from fewer SDPA launches + the 51 ms→~2 ms backward.

## Context (why this is the lever)
Ideogram4 trainer is HOST-op-construction-bound (GPU ~44% busy, ~5,500 launches/step) — it's the
only model NOT on autograd_v2's graph/capture path (klein/zimage are). The full graph-capture port is
a ~9% multi-phase campaign; flash SDPA is the bigger, tractable lever (cuts launch count + GPU time).
See memory project-ideogram4-trainer-hostbound + serenitymojo/docs/MOJO_V2_ENGINE_PLAN.md (Phase F).
