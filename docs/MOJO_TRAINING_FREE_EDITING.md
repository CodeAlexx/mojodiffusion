# Training-free editing in serenitymojo — field guide, gotchas, learnings

> The FlowEdit/DynaEdit surface built 2026-07-13/14: instruction editing (images +
> video), style transfer, insertion/composition, action/dynamics edits — zero
> training, pure sampler work, all on a 16GB RTX 5080. This doc is the distilled
> operating knowledge: exact recipes, what to watch out for, what NOT to do, and
> the measured learnings ledger. Module map: `docs/MOJO_MODULES.md` §"Training-free
> editing". Papers: FlowEdit arXiv 2412.08629 (ICCV'25), DynaEdit arXiv 2603.17989
> (implemented from the paper — no public code existed).

## 1. The recipes (copy-paste class)

### krea2 images, 512² (`pipeline/krea2_flowedit.mojo`) — ~18s/edit warm
- Encode 4 contexts (src/tgt × pos/neg) with `krea2_encode_cli`; stage source via
  `pipeline/krea2_stage_one.py <src> <out> 512 --crop smart`.
- Edit: `--steps 28 --nmax {15 color-change | 18 garment-swap | 21-24 semantic} --tgt-cfg 5.5 --auto-mask`.
- Style: **drop `--auto-mask`** (global edit), nmax 24-26 = strength.
- Final polish: feathered composite (dilate MaxFilter(17) + GaussianBlur(12), merge
  edit into original — bghira's tip; python post-step).
- int8 hybrid base + `<ckpt>.int8cache.safetensors` sidecar (startup ~10s) + cuDNN
  VAE (encode 0.51s / decode 0.85s) are all default-on.

### ideogram4 images, 1024² (`pipeline/ideogram4_flowedit.mojo`) — ~4min/edit
- Prompts are **structured JSON captions** (docs/IDEOGRAM4_PROMPTING.md); the edit
  pair = identical JSON with ONE element/field changed. Style edits = change
  `medium`/`aesthetics`/palette fields only — ideogram is the strongest style engine.
- `--steps 28 --nmax 24 --tgt-cfg 7.0 --auto-mask` (content edits).
- 16GB route: single-trunk CFG (uncond = cond trunk + zeroed text) + the layer-
  streamed TE (`ideogram_qwen3vl_streamed.mojo`) + tiled 3×3 decode. All default.

### LingBot video (`pipeline/lingbot_flowedit.mojo`) — 13f ≈ 1min, 121f ≈ 12min (27min DynaEdit)
- Geometry is `-D` comptime: default = 13f@576×320 (iteration size); full clips =
  `-D VFE_FRAMES=121 -D VFE_HEIGHT=480 -D VFE_WIDTH=832 -D VFE_L_SRC=.. -D VFE_L_TGT=..`.
- Appearance edit (FlowEdit): `--steps 30 --nmax 26 --src-cfg 1.5 --tgt-cfg 5.0 --auto-mask`.
- Action/dynamics edit (DynaEdit): add `--dynaedit --tau 1.0` (forces n_max=N) —
  cfg **2.5/4.5 for action changes on the subject**, **4.5/8.5 for object
  insertion/interaction** (see §3).
- Render: `uv run --with safetensors --with numpy --with pillow --with imageio
  --with opencv-python python parity/render_t2v.py <pixels.safetensors> <out>` →
  gif+mp4 (cv2 is the mp4 gate; without it mp4 is silently skipped).

### Klein native multi-ref (`pipeline/klein_edit_mojo.mojo`) — 49-62s/edit
- Klein-9B base IS a native edit model — 1-2 reference IMAGES in-context, zero
  training. Build `pixi run build-klein-edit`; caps via `klein_precache` (needs
  `"enforce_min_image_size": false` for 512²).
- Multi-ref T-offsets = **10.0 + r**. Single-ref identity transfer is decisive;
  2-ref composition works but dual-identity fidelity is the 512² base-model limit.

## 2. WATCH OUT FOR (the traps that actually bit)

- **Comptime text lengths.** Every prompt change can change L → the binaries
  fail-loud with "recompile with L_*=N" (lingbot) or "raise LT_SHARED" (krea2,
  cap 32 tokens ≈ 29-word prompts). BUDGET A REBUILD PER NEW PROMPT on lingbot
  121f (~34s build). krea2 style prompts must be SHORT.
- **Flash SDPA S must be 128-aligned** (krea2 `LFULL`); LTMAX must be a multiple
  of 128 there. `S must be 128-aligned` at startup = wrong `-D` math.
- **The auto-mask ERASES insertions.** It hard-copies source latents outside the
  subject mask — exactly where a new object must appear. Mask ON for
  appearance/action edits on an existing subject; mask OFF for insertion (and for
  styles, which are global).
- **Asymmetric CFG breaks the identity gate.** "tgt==src ⇒ output==source" only
  holds with MATCHED cfg (e.g. 1.5/1.5). Always run the recon gate matched; the
  asymmetric drift (MAD ~0.16) is expected, not a bug.
- **DynaEdit early steps carry no source signal** (n_max=N ⇒ first velocities are
  pure-noise-driven). Gate the mask saliency with `--mask-sigma-start 0.7`; a
  mask accumulated from step 1 is garbage.
- **T2V anchor drift.** DynaEdit's paper anchors on I2V first frames; on our T2V
  arm expect palette/scene re-render at n_max=N, worse at strong cfg. The i2v
  port is the structural fix (also LingBot's queued next lever).
- **fp8 "disk size" lies about resident size.** ideogram's TE is 8.2GB on disk →
  **15.1GB BF16 dequanted** (OOM). Check dequanted footprints before planning
  residency; the streamed-TE pattern (one ~0.4GB layer resident) is the escape.
- **`.serenity` model staging can be silently broken** — ideogram needed 3
  missing symlinks (uncond/TE/tokenizer). A path that "worked last week" may
  never have been exercised on this box. Fail-loud runs cheap; run one before a
  20-min job.
- **Tensor keys are not uniform**: staged clips use `pixel`, rendered outputs use
  `pixels`. Key-agnostic loading (`list(d.keys())[0]`) avoids the 20-min-run-then-
  KeyError sting.
- **Agents' background jobs die when the agent stops.** Long GPU runs launch from
  the ORCHESTRATOR session (or foreground inside the agent). Two builders lost
  runs this way before the pattern stuck.
- **One `mojo` compile at a time** — concurrent compiles corrupt the shared cache.
  And don't run 13-16GB GPU jobs while a builder's gate might start.

## 3. AVOID (measured dead ends)

- **Mild cfg for object insertion** (2.5/4.5): produced NO broom at all. Insertion
  needs 4.5/8.5 + tau 1.0. Conversely **4.5/8.5 on human subjects at 13f corrupted
  faces** — strong cfg is for chrome robots and big scene objects, not people.
- **Two-phase narratives on dense-1.3B** ("finds broom THEN sweeps"): the prop
  appears, the choreography doesn't. Single-activity-throughout prompts only;
  sequential events exceed the base model (paper's own limit class).
- **tau 0.01 for action edits**: near-argmax SGA weighting = stronger identity
  drift. tau 1.0 for actions; small tau only when motion must hug the source.
- **ComfyUI 'index' T-offset scheme (t=1,2) for Klein multi-ref**: collapses onto
  ref1. Use 10.0+r (measured).
- **Chasing the "0.85s/step" dense-T2V number**: it was a mis-recorded 192×320
  figure. 480×832 truth = 12.7s/CFG-step = **98% of flash roofline / 83% MFU**
  (nsys-proven). Headroom is ~1s/step of elementwise tail — polish, not a lever.
- **fp8_e4m3 resident for krea2 training on 16GB**: OOMs at step-1 autotune even
  for base t2i. `int_w8a8` is THE 16GB krea2 recipe (per reference-mojo-speed-parity).
- **Insertion-sort quantiles at video scale**: O(n²) at 193K voxels. Hoare
  quickselect = 5.4ms.
- **`git ... | head` for RC checks**: pipeline RC is head's. Capture grep's RC
  directly.

## 4. LEARNINGS LEDGER (what the numbers taught us)

- **The velocity difference is the whole game.** V_tgt − V_src localizes edits
  (auto-mask for free), measures them (saliency), and IS the edit (the ODE). One
  quantity, three uses.
- **Background preservation is solved in latent space**: hard-copy outside a
  velocity-diff mask reaches the VAE-roundtrip floor (images 1.43/255 vs floor
  1.59; video 0.00168 vs 0.00164; 121f 0.00286). The paper's own stated
  preservation limitation is fixable with ~30 lines — our addition beat it.
- **nmax is the universal strength knob**: 15 color / 18 garment / 21-26
  semantic-or-style; DynaEdit pins it to N because motion lives in the first steps.
- **Structural VRAM beats kernel VRAM**: single-trunk CFG (−9.3GB), layer-streamed
  TE (15.1→0.4GB), int8 hybrid resident+sidecar, tiled decode — every 16GB wall
  fell to a structure change, never to a smaller kernel.
- **The real hotspots hide in load paths, not math**: krea2's 62s/step was ~4.4s/
  step of per-forward DISK weight reloads + naive-conv VAE (93% of GPU time),
  not GEMMs. Profile before blaming kernels (reference-mojo-speed-parity discipline).
- **cuDNN conv via MAX SDK needs no new shims**: OIDHW checkpoint weights ARE
  FCQRS (identity repack); conv2d = depth-1 metadata lift. 11.6×/11.2× on the
  qwenimage VAE. The ldm (SDXL/SD3/Flux1/ideogram) VAE has the same seam mapped
  but ~30 comptime call sites — separate task.
- **Papers < 4 months old are implementable same-day** when they're sampler-only:
  FlowEdit (has code) ~1 day incl. mask; DynaEdit (NO code anywhere) ~3 hours
  from spec extraction to a working action edit. The extraction step (agent reads
  the HTML, maps equations onto our variable names) is what makes this fast.
- **Gate everything with a free invariant when one exists**: FlowEdit's
  "tgt==src ⇒ source" recon control caught wiring bugs in all three verticals
  before any subjective judgment was needed.
- **Composite-then-harmonize** (paste a user-provided cutout, then a gentle
  FlowEdit pass with "pasted cutout, flat lighting" → "standing naturally" prompts,
  nmax ~16) is the pragmatic multi-image path when native multi-ref falls short.

## 5. Open levers (next session shortlist)

1. **LingBot i2v port** — fixes DynaEdit's T2V anchor drift (identity/scene lock)
   AND was the queued LingBot lever anyway. Highest quality-per-effort.
2. ldm-VAE cuDNN (ideogram tiled decode + SDXL family) — seam mapped in
   `qwenimage_{encoder,decoder}.mojo`'s twin.
3. Klein multi-ref at 1024² (2-ref S=12800 — untested VRAM) for dual-identity
   fidelity; or seed-batch + pick via the SGA-style similarity score.
4. Cached text-embeds for lingbot FlowEdit (TE reload 7-18s of every run).
5. krea2 LT_SHARED bump (32→64, 128-aligned LFULL preserved: 64+1024=1088... NOT
   aligned — needs 128 ⇒ LFULL 1152, the LTMAX=128 trainer build already proves it).


## Server exposure (2026-07-16)

All three image edit verticals are now reachable through serenity-server
/v1/generate as graph nodes — Krea2FlowEdit (512²), Ideogram4FlowEdit (1024²,
JSON captions enforced pre-queue for BOTH prompts), and klein ReferenceLatent
(1–2 refs, 512²) — each e2e-verified (single-attribute edit proofs). Gotchas
that carried over exactly as this guide predicted: comptime geometry fail-louds
ride the worker error event (not crashes); LT_SHARED over-length prompts
fail-loud at context load; auto-mask SUPPRESSES global restyles by design
(day→night reconstructs the source) — localized single-field edits are the
contract. New gotcha for future flat keys: the server's GenerateRequest
middleman silently drops undeclared lowered keys — declare in wire+codec+
GenerateRequest+params_from_generate_request or the worker never sees them.

### Web production update (2026-07-21)

Krea2FlowEdit now dispatches both Raw and Turbo at compiled 512² and 1024²
shapes. Turbo uses a visible, editable 8-step profile with source/target CFG
zero; `_velocity_shape` returns the conditional velocity directly at CFG zero,
so each active FlowEdit step performs the required source and target forwards
without two unused unconditional forwards. The corrected 1024² Turbo path
measured 74.53s warm-cache versus 362.6s for the prior Raw-like schedule and
produced a visually coherent full-frame style edit. A cold file-cache/model
switch measured 138.35s and is not represented as warm performance.

The serving backend now keeps a keyed last-request cache for all four context
bins, the normalized source latent, and the matching int8 DiT. Exact 1024²
Regenerate measured 52.56s inside Mojo / 57s through the HTTP lifecycle with
text encode at 0.00003s, source-latent restore at 0.018s cumulative, and DiT
reuse at 0.001s. A changed target prompt drops the 20.7 GiB DiT before starting
the text encoder, then rebuilds it after conditioning; the measured warm-file
case remained safe and finished in 59.19s Mojo / 62s HTTP. A source, prompt,
Raw/Turbo checkpoint, or resident-block-profile change cannot reuse an
incompatible base.

The Canvas style workspace defaults Entire image on (`auto_mask=false`) because
the localization mask intentionally suppresses global restyles. SAM3 requests
use the Rust `/canvas/sam3/prepare` control-plane handshake before calling the
isolated mask service, preventing a resident 20+ GiB Krea2 worker from causing
an accessory-model OOM. Z-Image retains its Mojo-owned DiT/VAE state between
jobs: warm img2img and masked inpaint measured 1.60s and 4.22s end-to-end.

Canvas now exposes this work through one `Masked Edit - LanPaint` screen. Krea2
Raw/Turbo and the bounded Z-Image Base masked route are selectable only when
their live capability and local-model gates pass. The same selector inventories
the other upstream LanPaint model families as disabled entries. For model
families whose existing text/loader/VAE graph is reusable,
`WorkflowBuilder.buildLanPaintCandidate` preserves those model-specific nodes
and the LoRA chain and replaces only the latent tail with source encode, mask
extraction, `SetLatentNoiseMask`, `LanPaint_KSamplerAdvanced`, decode, and final
blend. These candidate graphs remain blocked by `/v1/preflight` until that
family has a real Mojo mask sampler; models absent on this machine are not
downloaded. The runtime handoff matrix is
`docs/SERENITY_LANPAINT_MODEL_MATRIX_2026-07-22.md`.

The 5080 branch also added pure-Mojo MageFlow instruction editing in
`pipeline/mageflow_edit_pipeline.mojo`. It uses Qwen3-VL image/text
conditioning, MageVAE source encode, clean-reference token concatenation beside
a pure-noise target, target-only four-step Euler updates, and aspect-preserving
decode. Its recorded reference/final latent cosines are 0.99979/0.99934 and the
render was visually matched to the oracle. This is a direct pipeline and parity
surface only; it is not yet a Canvas engine or Serenity worker, so the web UI
must continue to fail closed rather than aliasing it to FlowEdit or LanPaint.
