# SKEPTIC — Fusion is (probably) the wrong lever for serenitymojo's measured cost

> Date: 2026-05-30. READ-ONLY skeptic pass guarding a PERFORMANCE audit.
> Role: prevent the audit from over-promising on **GPU kernel fusion** when the
> measured/structural evidence points at **host↔device round-trips and
> per-op device synchronization**, not on-chip compute.
> Tenet 4 (flame-core `TENETS.md`): *measurement beats assertion.* The single
> precondition below follows directly from it.
> Every claim cites `file:line` read this session.

---

## TL;DR verdict

**Fusing GPU math kernels will not move serenitymojo's dominant cost.** The cost
this port pays is structural and lives on the **host transfer + sync** axis, not
the **on-device compute** axis. Two mechanisms, both proven from source, both
*synchronize the device*:

1. **Host round-trip per op** in the training block — `dit_block.mojo` crosses
   every op boundary as host `List[Float32]` via `.to_host(ctx)`, and does the
   residual / fan-out SUMs on the CPU (`add_lists`).
   (`serenitymojo/training/dit_block.mojo:21-34`; AUDIT_TRAINING_READINESS G8 :55)
2. **Deep device clone + `synchronize` per saved operand** on the tape —
   `Tensor.clone(ctx)` is `enqueue_create_buffer` + `enqueue_copy` +
   **`synchronize`**, NOT an Arc bump. Every recorded op saves operands this way.
   (`docs/MOJO_AUTOGRAD_INTERNALS.md:136-141`, citing `tensor.mojo:76-79`)

This is the same pattern project memory already nailed for flame-core:
*"OneTrainer runs Klein 3.4× faster with NO fused kernels — sync/transfer
patterns, not fusion"* and *"fused kernels are NOT the explanation for trainer
slowness."* The Mojo port has reproduced the anti-pattern in a more severe form
(an explicit `synchronize` baked into the saved-tensor path).

**Fusion can help GPU-compute-bound code with many tiny launches. This code is
not yet shown to be either.** Therefore every fusion lever is at best
UNVERIFIED-NEEDS-PROFILE, and several are IRRELEVANT-TO-THE-MEASURED-COST.

---

## The "236 s/step" number — flag it, it is not in the repo

The brief cites ~236 s/step with ~3 GB / 24 GB GPU used. **That number appears
nowhere in the committed docs.** I grepped all `*.md` and `docs/` (`236`,
`s/step`, `per step`, `wall`, `elapsed`, `profil`, `nsys`): the only timed runs
on disk are **INFERENCE** smokes, never a real training step.

Why there is no training-step timing: **no real model can train today.** The
training-readiness audit is explicit — G1 (no safetensors→BlockWeights loader),
G2 (AdaLN modulation not assembled), G3 (Klein double-stream block missing),
G5 (no data path), G6 (no real LoRA target map / no PEFT save) are all
**BLOCKERs**, and everything proven is proven at **toy dims D=8/H=2/Dh=4/M=4 on
synthetic `_randn` data** (`AUDIT_TRAINING_READINESS_2026-05-30.md:19-35,48-53`;
`train_step.mojo:48-53`). So any "236 s/step" is **either a synthetic toy-dim
microbench or a live builder-session number that is not yet attributable.**

⚠️ **This alone is grounds to halt fusion work.** A speedup target with no
committed, reproducible, attributed measurement is a hypothesis. If 236 s/step is
real, it must be captured (config, dims, what ran) before anyone designs a fix.

### What IS measured (inference) — and it already points away from fusion

The only real timings are inference, and they corroborate the transfer thesis:

- Klein 9B, 1024², 20-step, **fused-CFG**: `elapsed=14:24.76` ≈ **43 s/step**
  (`CODEX_REBOOT_HANDOFF_2026-05-26.md:39`). Already fused at the CFG level.
- The speedups that landed were **host-transfer removals, not math fusion**:
  - VAE NCHW/NHWC via GPU `permute` instead of **host loops** (`:36`).
  - timestep embedding dtype conversion via GPU `cast_tensor` instead of host
    (`:38`).
  - conv3d bias no longer staged through `to_host(ctx)`; reads device-resident
    bias directly (`:88-89`).

Three documented inference wins, **zero of them GPU-kernel fusion** — all of them
are "stop round-tripping through the host." That is the lever this codebase
actually responds to, on the *one path that has real measurements*.

---

## Adversarial verdicts on the fusion levers

The two compute-side agents had not posted fusion reports when I wrote this
(grepped: no `*FUSION*`/`*PERF*` audit file besides this one). Verdicts are
therefore framed against the **classes** of fusion levers such an audit will
propose. Reconcile each named lever against this table.

| Proposed lever (class) | What it actually reduces | Verdict vs the measured cost |
|---|---|---|
| Fuse RMSNorm+QKV / attention epilogue / SwiGLU into single kernels | GPU on-chip memory traffic + a few **launches** | **UNVERIFIED-NEEDS-PROFILE.** Benefit is on-device. If GPU sits at ~3 GB and the wall-clock is dominated by `to_host`/`synchronize`, this moves nothing. Only profile-confirmed launch-bound time justifies it. |
| Fuse elementwise chains (gate·residual, scale/shift AdaLN) | on-device compute | **IRRELEVANT-TO-IT** for the *training* block as written: those branch sums are done **host-side in `add_lists`** by deliberate design (`dit_block.mojo:24-29`). The cost is the host crossing, not the add. Fix = keep the branch on-device, not fuse the add kernel. |
| "Fuse to cut kernel-launch overhead" (many tiny launches) | launch count | **UNVERIFIED-NEEDS-PROFILE.** Plausible *only if* a profile shows launch-bound gaps. But the dominant per-op overhead documented here is a **full `synchronize`** per saved-tensor clone (`MOJO_AUTOGRAD_INTERNALS.md:140`), which fusing two math kernels does not remove — you still clone+sync operands at each recorded boundary. |
| Fused-CFG / batch pos+neg branches (inference) | redundant full forward passes | **HELPS — but already done** (`CODEX_REBOOT_HANDOFF:35`). Not a new win. |
| Resident attention (cuDNN-equivalent) to replace the math-kernel SDPA | GPU compute on the attention hotspot | **HELPS COMPUTE, but mis-targeted for the 236s.** Per project memory (`mojo_vs_flame_perf_bench_2026-05-29`), Mojo attention is 7-31× slower than cuDNN — a real compute gap. But that bench is a *kernel microbench*, not the step. With GPU ~idle (3 GB), the step is not attention-compute-bound; fixing attention compute helps the **inference** wall-clock far more than the transfer-bound training step. Sequence it AFTER the transfer fix, gated by a profile. |

**Kill criterion applied:** any lever whose entire benefit is "less on-device
compute" or "less on-chip memory traffic" is killed for the *training* 236s until
a profile shows compute/launch-bound time. None survive as "do this first."

---

## What the 236s must be attributed to (three different fixes)

The audit must pick ONE with reasoning, not say "fusion." The candidates need
*different* fixes:

1. **Host residency round-trip** (`.to_host`/`from_host` per op + `synchronize`
   on clone). Fix: keep activations **on-device across the block**; make `Tensor`
   carry branch points without host detour (the flame-core "saved tensors are Arc
   bumps" model, `MOJO_AUTOGRAD_INTERNALS.md:137-141`). **This is the structural
   suspect** and fusion does not touch it.
2. **Per-op kernel LAUNCH overhead** (many tiny launches). Fix: fusion — *only*
   if a profile shows launch-bound gaps with the GPU otherwise busy.
3. **Host-side loops in Mojo** (`add_lists`, chunk/pack helpers on CPU). Fix:
   move those reductions/packs onto the GPU. Fusion of GPU math kernels is
   orthogonal.

Three buckets, three fixes. Coding a fused kernel commits to bucket 2 with no
evidence it is bucket 2. The structural reading (G8 + §1.5 clone-sync) says it is
overwhelmingly **bucket 1**, secondarily bucket 3.

---

## The single precondition before ANY kernel is coded

**Profile a real step (or, until a real model trains, a real-dim block fwd+bwd)
and attribute the wall-clock to {host-transfer+sync, kernel-launch, GPU-compute,
host-loop}.** No fused kernel is justified until that attribution exists.

Concretely, in priority order:

1. **Capture the 236 s/step provenance** — what config/dims/script produced it,
   how much was synthetic vs real. It is not in any committed doc; without it the
   target is a rumor.
2. **Profile** the heaviest real-dim path available (inference Klein-9B step is
   the only thing that runs end-to-end today) with nsys or per-op host timers:
   - time in `cudaStreamSynchronize` / `synchronize`,
   - count + total time of `enqueue_copy` (H2D/D2H),
   - GPU SM occupancy / idle gaps,
   - kernel-launch count and inter-launch gaps.
3. **Only if** the profile shows GPU-busy + launch-bound gaps does fusion get a
   green light, and only for the kernels the profile names.

This is flame-core Tenet 4 verbatim: a "this should be faster by fusing" comment
is a hypothesis; an nsys delta on the named site is evidence. The same discipline
that retired the `autograd.rs:1493` dead-code "sync source" claim and the H=30
"silent-zero" false-red (`AUDIT_TRAINING_READINESS:81-86`) applies here.

---

## Inference vs training (clears the brief's question 4)

- **Inference runs end-to-end** but is also slow (~43 s/step Klein 9B 1024²,
  already fused-CFG) — and its proven wins were **host-transfer removals**
  (`CODEX_REBOOT_HANDOFF:36,38,88-89`), not math fusion. So even inference does
  not validate fusion as the lever.
- **Training does not run on a real model at all** (G1-G6 BLOCKERs). The 236s, if
  it is a training number, is necessarily a **toy-dim/synthetic** number, and the
  cost it measures is the **training-only host-List backward** path
  (`dit_block.mojo` host round-trips + tape clone-sync), which is *exactly* the
  axis fusion does not touch.

Net: the shared GPU kernels are not the thing to fuse; the training-specific
host-List backward is the thing to make resident. **Different fix, different
file, and a profile must confirm it before a line of kernel code is written.**

---

## One-line verdict

Fusion is **not** the right first lever: the measured/structural cost is
host↔device round-trips + a per-op `synchronize` (G8 `dit_block.mojo:21-34`;
clone-sync `MOJO_AUTOGRAD_INTERNALS.md:140`), and there is **no committed,
attributed 236 s/step measurement at all** — so the precondition for any kernel
work is to **capture that number's provenance and profile a real-dim step to
split host-transfer/sync vs launch vs compute** before coding anything.
