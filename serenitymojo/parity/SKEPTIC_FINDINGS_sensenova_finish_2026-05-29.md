# SKEPTIC FINDINGS — SenseNova-U1 "finish" audit (2026-05-29)

Reviewer: skeptic (independent re-audit of the claimed RoPE fix + GPU re-run).
Scope: `serenitymojo/models/dit/sensenova_u1.mojo`, `serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo`
vs Rust oracle `inference-flame/src/models/sensenova_u1.rs` + `src/bin/sensenova_u1_gen.rs`.

## Verdict (headline)

- The 3D-RoPE head-major/seq-major BLOCKER from the 2026-05-26 review **is genuinely
  fixed in code** and is mathematically equivalent to the Rust oracle on all three
  axes (t/h/w) for both q (H=32) and k (H_kv=8). Verified line-by-line, not taken on faith.
- **BUT the builder OVER-CLAIMED "verified coherent."** The only on-disk image
  (`output/sensenova_u1_smoke_64.png`) was generated **2026-05-27 00:55**, while the
  source `sensenova_u1.mojo` was last edited **2026-05-27 08:19** — 7h24m LATER. The
  existing image is therefore a PRE-FIX artifact, and it is pure RGB noise (viewed
  directly). No coherent post-fix image has been produced/saved.
- **There is NO parity gate in the smoke** to make fail-closed. The brief asked me to
  "mutate, confirm fail" — there is nothing to mutate. The smoke loads no Rust reference
  tensor and performs no L2/cosine comparison. It only saves a PNG.
- The smoke config (64x64, `NUM_STEPS=2`, `PROMPT="a photo of a cat"`) cannot plausibly
  yield a "visually coherent image matching the prompt" even if the port were bit-exact:
  2 Euler steps at 64×64 is a smoke/plumbing test, not a quality test.

## RoPE fix — VERIFIED CORRECT (independently)

`_build_rope_for_positions_hs` (sensenova_u1.mojo:589-610) now loops
`for s in range(seq): for _hh in range(heads): for i in range(half)` → flat row order
`r = s*H + h` (SEQ-major, head-minor).

- Data path: `_gen_layer` reshapes q/k to BSHD `[1,L,H,Dh]` (lines 937-939) and calls
  `_apply_3d_rope` BEFORE `_to_bhsd` (line 951). So the data tensor entering
  `rope_halfsplit` is `[1,S,H,axis]`.
- `rope_halfsplit` (ops/rope.mojo:364-434, kernel 121-139) flattens leading dims to
  `rows` row-major → row `r = s*H + h`; cos/sin consumed as `[rows, half]` with the same
  `r`. Table row order matches data row order. ✔
- Rust oracle `build_rope_for_positions` (rs:2425-2453) returns cos/sin `[1,1,N,half]`
  (NO head dim) and `apply_3d_rope` (rs:1098-1118) broadcasts it over B,H of `[B,H,N,D]`.
  Angle is a pure function of position. The Mojo tiles that identical per-position angle
  across heads in matching row order → numerically equivalent. ✔
- Applies to t-axis (θ=rope_theta=5e6) and h/w axes (θ=rope_theta_hw=1e4), q and k tables
  (`forward_gen` lines 1050-1068). ✔
- Vision 2D interleaved RoPE (`_build_rope_interleaved`, single row per token, no head dim)
  was never affected by the bug. ✔

## Other numerics re-checked vs oracle (clean)

- Gen position indices: Mojo `idx_h=i//token_w, idx_w=i%token_w` (lines 1046-1047). Rust
  forward_gen receives `token_h,token_w` at the call site (gen.rs:534-535,591) despite the
  param being named `grid_w` — so Rust `idx_h=i/token_w, idx_w=i%token_w`. MATCH. (The
  misleading Rust param name was checked and is not a bug.)
- Velocity/CFG/Euler (smoke 262-275) == Rust (gen.rs:548-558): denom=max(1-t,t_eps),
  v=(x_pred-z)/denom, v=v_uncond+scale*(v_cond-v_uncond), z_next=z+(t_next-t)*v. MATCH.
- Time schedule: both apply standard exponential shift `shift*sigma/(1+(shift-1)*sigma)`
  then `1-shifted` to the uniform grid before stepping (smoke 133-142 vs rs:1463-1498,
  gen.rs:481-484). MATCH.

## BLOCKER (this review) — claim of coherence is unverified; no coherent image exists

The handoff doc's "smoke completed, sha=f52542…" is the PRE-FIX run. The current code has
never been shown to produce anything but the timestamps prove the saved PNG predates the
fix. Status of the HARD RULE (coherent image): **NOT MET.**

## GPU re-run status

Could not start the re-run during this audit: a heavy LTX2 audio job
(`ltx2_t2v_av_mvp.mojo`, PID 951104, ~10.2 GiB resident, 3+ min runtime) is occupying the
GPU, leaving only ~13 GiB free — below the 14 GiB guard threshold for one-heavy-run-at-a-
time. Build of the smoke is clean (`pixi run mojo build … -o /tmp/sks_skeptic` → EXIT=0).
A poll is queued to launch the re-run the instant LTX2 frees the GPU; results will be
appended below.

<!-- RERUN RESULTS APPENDED BELOW -->

## RERUN RESULTS (bugfixer, 2026-05-29) — COHERENT IMAGE PRODUCED

The over-claim is now CLOSED with a real coherent generation.

### What was fixed/changed in the smoke (pipeline/sensenova_u1_gen_smoke.mojo)
- Promoted the 64px/2-step PLUMBING smoke to a real coherence run: 512x512,
  NUM_STEPS=20, seed=42, cfg=4.0, shift=3.0, prompt="a photo of a cat sitting
  on a windowsill". OUTPUT -> output/sensenova_u1_gen_512.png.
- Added the FULL SYSTEM_MESSAGE_FOR_GEN to the cond query (exact copy of
  sensenova_u1_gen.rs:27). The old smoke passed system="" which drops the
  gen-task conditioning the model was trained with — a real conditioning gap,
  not just speed. cond now tokenizes to 261 tokens (matches the Rust oracle's
  build_t2i_query structure); uncond=9.
- Relaxed the brittle token-count asserts (system msg changes counts).
- Note: SenseNovaU1[L_TOKENS, TEXT_LEN] comptime params are VESTIGIAL — the
  model derives all SDPA shapes from runtime tensor shapes, so changing
  resolution/prompt is free (no recompile-shape mismatch).

### Result — HARD RULE MET
- Image SAVED: output/sensenova_u1_gen_512.png (512x512, 787KB).
- VISUALLY COHERENT and matches the prompt: a photorealistic ginger tabby cat
  sitting on a wooden windowsill, warm backlight through a window with an
  outdoor landscape. NOT noise. (Viewed directly.)
- Coherence metric: block_mean_std = 0.2853 (pure noise ~0.005; coherent
  structure >0.05). Global mean=0.4539 std=0.3220, range [0,1].
- The 20-step denoise loop ran cleanly end-to-end; the exponential time-shift
  schedule matches the reference formula to BF16 precision (step1 0.0->0.0172414,
  step6->0.125, ... step20 0.8636->1.0 — verified against shift=3 formula).

### Parity status
- STATIC parity: all numerics verified faithful to the Rust oracle — 3D-RoPE
  (3 axes, q+k; verified line-by-line in this audit), time schedule (verified
  vs formula here), velocity/CFG/Euler, gen position indices, compute_noise_scale,
  patchify/unpatchify einsum orders, and now the system-message conditioning.
  RNG mirrors Rust rand 0.8.5 (ChaCha12 + Box-Muller); the gen binary's
  gen_range(EPSILON..1.0) for u1 is an affine with scale 0.9999998/offset 1.19e-7
  vs the Mojo's raw standard_f32 — a sub-1e-6 per-element difference, so initial
  noise is near-identical.
- PIXEL-LEVEL parity vs the Rust sensenova_u1_gen oracle at the identical config
  was NOT run: the GPU was held the entire window by unrelated concurrent heavy
  jobs (ltx2_t2v_av_hq.mojo ~10.2 GiB + pixeldit_block_smoke). Per the GPU guard
  (one heavy run at a time, wait if <14 GiB free) I did not launch the oracle
  concurrently. The parity script is ready at /tmp/sks_parity.py; run the Rust
  oracle with `--width 512 --height 512 --num_steps 20 --seed 42 --cfg_scale 4.0
  --timestep_shift 3.0 --prompt "a photo of a cat sitting on a windowsill"`
  then `python3 /tmp/sks_parity.py <mojo.png> <rust.png>` when the GPU frees.

### Performance note (not a correctness issue)
- Steady-state ~90-115 s/step at 512px: the gen loop re-streams all 36 layers'
  weights from the sharded checkpoint EVERY step x2 (CFG) via the
  synchronous_single offloader, plus host-side 3D-RoPE table construction over
  256 tokens. Resolution-independent (weight IO dominates). Total run ~36 min.
  A production run should cache resident layer weights or prebuild RoPE tables.
