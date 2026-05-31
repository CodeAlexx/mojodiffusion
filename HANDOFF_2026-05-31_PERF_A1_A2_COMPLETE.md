# PERF HANDOFF — serenitymojo Klein LoRA training residency (A1 + A2 DONE)
date: 2026-05-31 · supersedes the speed sections of HANDOFF_2026-05-30_PERF_ROADMAP.md

> Self-contained. Read top-to-bottom; you can resume from this doc. Every number
> below was MEASURED in-session with the tool named (Tenet 4). Where something is
> inferred, it's tagged HYPOTHESIS.

═══════════════════════════════════════════════════════════════════════════════
## §0 — TL;DR
═══════════════════════════════════════════════════════════════════════════════

The Klein-4B LoRA training step went **124.5 s → 58 s (−53%, 2.15×)**, loss
`2.734082` BIT-IDENTICAL at every stage, still pure F32 (no fusion, no BF16), all
parity gates green. Two increments landed:
- **A1** — block activation carriers host `List[Float32]` → device `Tensor`/`TArc`
  (single_block.mojo + double_block.mojo). 124.5→84.4 s.
- **A2** — frozen base weights host `List` → device `TArc`, uploaded ONCE not
  per-op (weights.mojo + the two block files + the two stack files). 84.4→58 s.

The NEXT measured lever is identified and ready to build (§5): **skip the dead
frozen base-weight gradient** (`d_x`-only `linear_backward` at frozen sites) —
the trainer computes ~42 GB/step of base `d_w`, reads it to host, and DISCARDS it
(only LoRA A/B are trained). That's the single biggest remaining waste.

═══════════════════════════════════════════════════════════════════════════════
## §1 — Measured timeline (all my own `train_klein_real.mojo` 1-step runs, 4B)
═══════════════════════════════════════════════════════════════════════════════

Config: 4B (D=3072, F=9216, H=24, Dh=128, 5 double + 20 single blocks),
512px (N_IMG=1024, N_TXT=512, S=1536), RANK=16, RUN_STEPS=1, DO_SAMPLE=False.

| state                         | s/step          | loss      | how measured |
|-------------------------------|-----------------|-----------|--------------|
| baseline (start of session)   | 124.5           | 2.734082  | dmon run |
| + A1 single_block (20 blocks) | 88.1  (−29%)    | 2.734082  | dmon run |
| + A1 double_block (5 blocks)  | 84.4  (−4% more)| 2.734082  | dmon run |
| + A2 frozen weights           | 57.8 / 58.8 / 58.75 (−31% more) | 2.734082 | 2 dmon + 1 nsys |

Cumulative **124.5 → 58 = −53%, 2.15×**. dmon SM-util rose 20%→29% mean; idle<20%
76%→~61% of samples. The step is STILL transfer/host-bound (most of 58s is host CPU
marshalling host Lists, not GPU compute — see §4).

NOT re-measured this session: the **9B** config (last known 236 s/step pre-opt).
These are primitive-level fixes so 9B should benefit proportionally — HYPOTHESIS,
run it to confirm.

═══════════════════════════════════════════════════════════════════════════════
## §2 — What changed, file by file (the klein/ tree is git-UNTRACKED — see §7 backups)
═══════════════════════════════════════════════════════════════════════════════

### A1 — single_block.mojo (now 680 lines)
- `SingleBlockSaved` (15 fields): `List[Float32]` → `Tensor`, struct made **Movable-only**
  (dropped Copyable — verified no caller `.copy()`s it). `SingleBlockForward` likewise.
- All 4 functions (base+LoRA fwd/bwd) thread device `Tensor` op-to-op; `from_host`
  ONCE on entry (x / d_out), `.to_host` ONCE per grad/output leaving.
- Host split/join loops (`_split2_cols`, `_qkv_split`, `_join2_cols`, `_qkv_join`)
  → device `slice(t, dim=1, start, len)` + `concat(1, ctx, ...)`.
- LoRA-delta branches still bridge to host (helpers in lora_block.mojo are host-typed;
  out of scope — same as everything below).

### A1 — double_block.mojo (now 1207 lines)
- `StreamSaved` (16 fields), `DoubleBlockSaved`, and forward helper structs
  `_StreamPre`/`_StreamPost`: `List[Float32]` → **`TArc`** (NOT Movable-only — forced by
  multi-read fan-out: q_rms/k_rms/v feed BOTH the joint concat AND the save; plus the
  destructor/partial-move constraint the file's own comments flag). `.copy()` on a TArc
  field is a refcount bump (no D2D, no sync).
- EXCEPTION: backward helpers `_StreamPostBack`/`_StreamPreBack` keep host `List` for
  `d_x`/`d_att` — the move-only op-result grad structs (`GateResidualGrads`,`LinearGrads`)
  can't hand a `Tensor` out except via `.to_host()`. (This is part of why double-block
  only gave −4%; see §5 — the backward still bounces.)
- Joint attention (concat txt|img q/k/v → rope → sdpa → slice back) was already device;
  kept device instead of `.to_host()`ing between ops.

### A2 — weight residency (5 files)
- `StreamWeights` (double): wqkv/wproj/wgu/wd/q_norm/k_norm → **TArc**.
- `SingleBlockWeights`: w1/w2/q_norm/k_norm → **TArc**.
- `KleinStackBase` (klein_stack.mojo): img_in/txt_in/final_lin/final_shift/final_scale → TArc.
- `weights.mojo` loader: returns TArc (Tensor.from_view → F32 → TArc) — each frozen
  matrix is H2D'd EXACTLY ONCE at load, never re-uploaded.
- Every `_t(w.<field>.copy(), [shape], ctx)` per-op upload site → `w.<field>[]` device
  deref. VERIFIED: zero `_t(w.<bigweight>` remain in either block file.
- LEFT host (correctly): ModVecs/SingleModVecs (per-step, recomputed from sigma, small),
  and LoRA A/B adapters (optimizer mutates them — separate harder task, NOT done).
- struct `__init__` still takes host `List[Float32]` + dims + ctx and uploads internally
  (keeps loader + gates byte-identical); parity-gate loaders got `+ ctx` and derived dims.

═══════════════════════════════════════════════════════════════════════════════
## §3 — Correctness gates (ALL re-run BY ME, real cos printed — not the "0 failed" trap)
═══════════════════════════════════════════════════════════════════════════════

cos threshold 0.99999999 (8 nines). Run recipe per gate's top comment; oracle .bin
files already exist in parity/.

| gate file (serenitymojo/models/klein/parity/)   | result | worst cos |
|--------------------------------------------------|--------|-----------|
| single_block_parity.mojo (9 grads)              | PASS   | 0.9999999999996552 |
| single_block_lora_parity.mojo (6)               | PASS   | 0.9999999999998571 |
| double_block_parity.mojo (28 grads)             | PASS   | 0.9999999999991572 |
| double_block_lora_parity.mojo (16)              | PASS   | 0.9999999999985525 |
| klein_stack_lora_parity.mojo (18, end-to-end)   | PASS   | 0.9999999999944037 |

To re-verify after any change: `cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
&& pixi run mojo run -I . serenitymojo/models/klein/parity/<gate>.mojo` — and READ the
printed cos numbers; a gate that fails to compile still exits 0.

DISCIPLINE (non-negotiable, handoff §6 of the prior doc):
1. ONE mojo compiles at a time (concurrent compiles corrupt the shared cache).
2. `rm -f serenitymojo.mojopkg` before EVERY compile.
3. F32-only. Any cos drop < 0.99999999 → REVERT that piece.
4. Re-measure (dmon + nsys) after each stage; ship on a measured drop, never "should be".
5. loss must stay 2.734082 at real dims (math-unchanged check).

═══════════════════════════════════════════════════════════════════════════════
## §4 — The profile (nsys CUDA trace) — WHERE the time goes now
═══════════════════════════════════════════════════════════════════════════════

Two traces on disk: `/tmp/klein_prof.nsys-rep` (A1-state, 83.5s) and
`/tmp/kp2.nsys-rep` (A2, 58.75s). Size-bucket attribution script: `/tmp/attrib.py`
(python3 + sqlite3 module; the `sqlite3` CLI is NOT installed — use python). Output
saved `/tmp/attrib_out.txt`.

### Per-step GPU memcpy volume (MB), A1-state → A2:
| kind | A1-state | A2 | what A2 did |
|------|----------|-----|-------------|
| H2D (kind=1) | 70,290 | **40,677** | −30 GB: frozen-weight uploads gone (324MB×60→×20) |
| D2H (kind=2) | 53,394 | 53,394 | UNCHANGED — A2 didn't touch the gradient readback |
| D2D (kind=8) | 31,950 | 31,950 | UNCHANGED — 883,661 copies, reshape-as-clone |

### GPU memcpy TIME, A2 trace: D2H 2.20s (43%) > H2D 1.66s (33%) > D2D 1.21s (24%).
But total GPU memcpy time ≈5s while WALL is 58s → **most of the step is host-CPU
marshalling** (packing/unpacking host `List[Float32]`), invisible to GPU-util.
cuStreamSynchronize was 23.4s of API time in the A1 trace.

### D2H attribution (the 53 GB, A2's untouched top lever), by copy size:
- 324 MB × 40 = 12,960 MB  ← single-block w1/w2 gradients ([3D+2F,D])
- 162/144/216/108/72/54 MB × … ≈ 29,000 MB  ← d_wqkv/d_wproj/d_wgu/d_wd (base weight grads)
- 18 MB × 405 = 7,290 MB  ← activation grads (d_x)
- **≈42 GB of the 53 GB D2H is FROZEN BASE-WEIGHT GRADIENTS** that the trainer
  computes, reads to host, and DISCARDS (only LoRA A/B are optimized).

### D2D (32 GB, 883,661 copies): 0.035MB×384000 + 0.070MB×92160 + 0.012MB×407040 +
big-tensor copies. This is reshape-as-clone (reshape does a full D2D copy for what is a
byte-identity view) + slice/concat allocs. LOW GPU time → lower priority than D2H.

═══════════════════════════════════════════════════════════════════════════════
## §5 — NEXT LEVER (measured, ready to build): skip the dead frozen-weight gradient
═══════════════════════════════════════════════════════════════════════════════

**The waste (measured §4):** ~42 GB/step D2H is base-weight `d_w` that's discarded.
The blocks call `linear_backward(...)` which computes BOTH `d_x` (needed) AND `d_w`
(the full weight gradient), then `.to_host()`s `d_w` and the trainer throws it away —
base weights are frozen; only LoRA A/B train (via the separate `klein_lora_bwd`).

**The fix:** add a `d_x`-only path to `linear_backward` (or a sibling
`linear_backward_dx`) that does NOT compute/allocate/return/readback `d_w`. Use it at
every FROZEN-weight backward site in single_block.mojo + double_block.mojo (the base
qkv/proj/gu/wd/w1/w2 linears). This eliminates ~42 GB D2H AND the GPU matmul that
produces each `d_w`. The LoRA `d_A`/`d_B` are unaffected (separate code path).

**Risk:** LOW for correctness (you're DROPPING an unused output, not changing math) —
but it changes `linear_backward`'s signature/return, so every caller must be updated and
the gates re-run. The LoRA-grad math must be untouched.
WILL-IT-HELP: strong — targets the measured top axis. Magnitude unproven until built +
re-measured (dmon). Expect a meaningful chunk of the ~50s host-marshalling to vanish
with the 42 GB readback.

**Gate:** all 5 §3 gates must stay green at cos≥0.99999999 (they check d_x and the LoRA
grads — exactly what must stay correct). Re-measure dmon after.

### Lever #2 (after #1, lower priority): reshape-as-view
`reshape` (tensor_algebra.mojo:472) does a full D2D `enqueue_copy` for a byte-identity
shape change. The 883k D2D copies (32 GB) are mostly this. Making reshape return a
view (share the buffer, change only shape metadata) would kill them. LOW current GPU
time so modest wall win, but trivial-ish and removes 883k ops. Gate: all parity gates.

### Lever #3 (deferred, gated): ring/pool allocator (roadmap §2) — still HYPOTHESIS,
profile-gated. Fusion (roadmap §3) LAST, F32-only. Do NOT promise Rust's 2.34s/step
from residency alone (that's Rust+BF16+fused).

═══════════════════════════════════════════════════════════════════════════════
## §6 — Method (reproduce before touching anything)
═══════════════════════════════════════════════════════════════════════════════

### Re-measure step time + GPU idle:
```
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
(nvidia-smi dmon -s ut -d 1 -c 210 > /tmp/dmon.log 2>&1 & D=$!; \
 pixi run mojo run -I . serenitymojo/training/train_klein_real.mojo > /tmp/run.log 2>&1; kill $D)
grep PROG /tmp/run.log    # secs/step + loss (loss MUST stay 2.734082)
awk 'NR>2 && $2~/^[0-9]+$/{s+=$2;n++;if($2<20)lo++} END{print "mean SM%",s/n," idle<20%",lo"/"n}' /tmp/dmon.log
```

### nsys profile + attribution:
```
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
nsys profile -t cuda --stats=false -o /tmp/kp3 --force-overwrite true \
  pixi run mojo run -I . serenitymojo/training/train_klein_real.mojo > /tmp/nsys.log 2>&1
nsys stats --report cuda_gpu_mem_size_sum --report cuda_gpu_mem_time_sum /tmp/kp3.nsys-rep
python3 /tmp/attrib.py   # edit the two paths inside to point at the new .sqlite
```
(`mojo run` works; a standalone `mojo build` of train_klein_real.mojo fails at link with
`undefined reference to sinf` — a libm link-flag env quirk, NOT a source error. The
A2 builder mis-reported this as a blocker; the trainer runs fine via `mojo run`.)

═══════════════════════════════════════════════════════════════════════════════
## §7 — Backups & key files
═══════════════════════════════════════════════════════════════════════════════

The klein/ tree is git-UNTRACKED, so /tmp backups are the ONLY revert points:
- /tmp/single_block_a1_applied.mojo   (A1 single, gates green)
- /tmp/double_block_pre_a1.mojo (orig host-List 1343L) + /tmp/double_block_a1_applied.mojo (A1)
- /tmp/weights_pre_a2.mojo, /tmp/klein_stack_pre_a2.mojo, /tmp/klein_stack_lora_pre_a2.mojo,
  /tmp/train_klein_real_pre_a2.mojo   (pre-A2)
- No stray .bak files in the klein/ tree (verified clean).
- Profiles: /tmp/klein_prof.nsys-rep (A1-state), /tmp/kp2.nsys-rep (A2). attrib: /tmp/attrib.py.

Current live source line counts (post-A2): single_block 680, double_block 1207,
weights 216, klein_stack 541, klein_stack_lora 567.

Trainer config: serenitymojo/training/train_klein_real.mojo (4B, RUN_STEPS=1,
DO_SAMPLE=False — the timing config). 9B is a comptime swap.

Memory: project_mojo_a1_carrier_win_2026-05-31 (the win + device-op facts + the next-lever
list); MEMORY.md index updated.

═══════════════════════════════════════════════════════════════════════════════
## §8 — Reusable facts (skeptic-verified this session)
═══════════════════════════════════════════════════════════════════════════════

- `slice(t, dim=1, start, len)` is a correct strided column gather: 2D [N,3D] → inner=1;
  4D [1,S,H,Dh] → inner=H*Dh. Reproduces the old host split loops exactly.
- `concat(1, ctx, txt, img)` is txt-FIRST and is the exact transpose of the two forward
  slices (N_TXT+N_IMG=S, no gap/overlap). Verified at all 6 double-block coupling sites.
- `reshape [N,D]↔[1,N,H,Dh]` is a byte no-op BUT currently does a full D2D copy (lever #2).
- `TArc = ArcPointer[Tensor]` (autograd.mojo:50): `.copy()` = refcount bump of the SAME
  device buffer (no D2D, no sync); deref with `[]`. Safe to share because all ops return
  FRESH buffers (Mojo has no in-place op API).
- Per-block recompute backward keeps only ONE block's saved activations alive at a time →
  VRAM bounded (not 5×/25×); TArc weight sharing means residency doesn't duplicate buffers.

═══════════════════════════════════════════════════════════════════════════════
## §9 — Process notes / honesty ledger (Tenet 4)
═══════════════════════════════════════════════════════════════════════════════

- The orchestrator (me) is gate-of-record: builder agents' pasted cos numbers were NOT
  trusted; every gate above was re-run by the orchestrator with real cos read. Keep this.
- The A2 builder falsely claimed train_klein_real "won't link (sinf)" — it runs fine via
  `mojo run`; that was a standalone `mojo build` artifact. Verify builder claims by running.
- Two self-corrections this session: (1) I wrote a fabricated "49.67s/-60%" into memory
  BEFORE a dmon run returned — corrected to the real 84.4s; (2) I briefly mislabeled the
  A2 re-profile "stale" after a failed extraction — the clean rerun (kp2) confirmed the
  H2D collapse. Both corrected in memory. Lesson: never record a perf number before the
  measurement tool result is in hand.
- Output-channel corruption recurred this session (batched tool calls cancelling on one
  error, dropped stdout). Mitigation: run compiles/measures ONE at a time, write results
  to /tmp files and Read them back.
