# Mojo Autograd Internals (serenitymojo)

Audit of the reverse-mode autograd TAPE engine in the Mojo training port
(`serenitymojo`). Modeled on flame-core's `FLAME_AUTOGRAD_INTERNALS.md`, but
every claim below cites a real `file:line` in the Mojo tree that was read
directly. Where the Mojo design DIVERGES from flame-core, that divergence is
called out explicitly — this is **not** a prose translation of the Rust doc.

All file references are relative to `/home/alex/mojodiffusion/serenitymojo/`.
Package: `serenitymojo`. Mojo 1.0.0b1, NVIDIA GPU.

> **TENET 4 (measurement beats assertion).** Per-op cos-vs-torch numbers in
> this doc are tagged **MEASURED-by-prior-lead** — sourced from the master
> handoff `HANDOFF_2026-05-30_MOJO_TRAINING_PORT_MASTER.md` §2, re-run by the
> lead on a clean serial build. They were NOT re-run while writing this doc
> (compilation is reserved for another agent). Structural claims (struct
> fields, dispatch arms, file:line) were read directly and are not estimates.

---

## 0. The single most important divergence from flame-core

flame-core's tape is a **global, mutex-protected singleton** (`AUTOGRAD_CONTEXT`,
a `lazy_static! Mutex`). The Mojo port deliberately does **not** do this.

From `autograd.mojo:5-8` (architecture decision, USER, 2026-05-30):

> EXPLICIT threaded Tape struct (no global thread-local; serenitymojo has no
> global mutable state, Mojo 1.0.0b1 globals are shaky). Tape passed `mut`.

So the entire "atomic flag prevents backward-deadlock on the global mutex"
machinery from flame-core §3.1 **has no analogue here and is not needed** —
there is no lock to re-enter. The tape is a plain value threaded through the
call chain. This is the central architectural difference; the rest of this
doc describes what actually exists.

---

## 1. Graph Construction

### 1.1 The Node Struct: `TapeEntry`

The computation graph is a linear list of `TapeEntry` nodes
(`autograd.mojo:244-275`). Unlike flame-core's `Op` enum with 47 variants,
the Mojo `TapeEntry` is a **single flat struct** with an integer `op_kind`
discriminator:

```mojo
struct TapeEntry(Copyable, Movable):     # autograd.mojo:244
    var out_id: Int          # id of the tensor this op produced
    var op_kind: Int         # which op (OP_ADD=0 .. OP_MSE=8)
    var lhs_id: Int          # autograd id of left input
    var rhs_id: Int          # autograd id of right input
    var saved0: TArc         # lhs clone (Mul/MatMul/Linear/RMSNorm/... bwd)
    var saved1: TArc         # rhs clone
    var dim_m: Int           # MatMul/Linear: C[M,N]=A[M,K]@B[K,N]
    var dim_n: Int
    var dim_k: Int
    var third_id: Int        # 3rd input slot (Linear bias)
    var saved2: Optional[TArc]   # 3rd saved tensor (Linear bias)
```

- `op_kind` is one of the 9 `comptime` constants `OP_ADD=0` through `OP_MSE=8`
  (`autograd.mojo:52-60`). There is **no enum** — dispatch is an integer
  `if/elif` ladder in `backward()`.
- `saved0`/`saved1` are **always populated** (every record method clones both
  operands), even for ops whose backward does not read them. The file flags
  this as a known micro-opt-for-later (`autograd.mojo:18-20`): Add/Sub don't
  need saved tensors but currently clone them anyway. Contrast flame-core,
  where Add saves nothing.
- `saved2`/`third_id` are the 3-input extension used only by `OP_LINEAR`
  (x, W, b). Defaults (`third_id=0`, `saved2=None`) keep every 2-input arm
  byte-identical (`autograd.mojo:254-257`).
- `dim_m/dim_n/dim_k` carry matmul/linear shape so the backward arm can call
  `mm_backward`/`linear_backward` without re-reading shapes off the tensors.

### 1.2 Where the Graph Lives: an explicit `Tape` value

The tape is a `struct Tape(Movable)` (`autograd.mojo:278-284`):

```mojo
struct Tape(Movable):
    var next_id: Int                 # monotonically increasing id counter
    var entries: List[TapeEntry]
```

It is constructed with `Tape()` (`next_id=1`, empty list, `autograd.mojo:282`)
and **passed by the caller** to every record method (`mut self`) and to
`backward(tape, loss, ctx)`. There is no singleton, no mutex, no thread-local.
A training step owns its `Tape`; freeing the `Tape` frees all `TapeEntry`s and
(via Arc refcount) the saved tensors.

### 1.3 TArc and the move-only-Tensor problem

`comptime TArc = ArcPointer[Tensor]` (`autograd.mojo:50`).

The reason: `Tensor` is **move-only** (`struct Tensor(Movable)`,
`tensor.mojo:32` — explicitly NOT `Copyable`). Mojo `List`/`Dict` require
`Copyable` elements, so a bare `Tensor` can't be stored in the tape's
`List[TapeEntry]` or the gradient `Dict`. Boxing in `ArcPointer[Tensor]`
makes it `Copyable` (a copy is a refcount bump), and the Arc refcount frees
the saved/grad tensors automatically — no manual alloc/free
(`autograd.mojo:10-13`). This is the same idiom the inference loaders already
use for `Dict[..., Tensor]`.

### 1.4 TensorId tracking (`Tensor.id`)

`Tensor` carries one training-only field: `var id: Int` (`tensor.mojo:49`).

- `id == 0` means **untracked** — the inference default. Every existing 3-arg
  `Tensor.__init__` caller stays untracked, byte-identical (`tensor.mojo:44-48`).
- `set_id(mut self, id: Int)` (`tensor.mojo:64`) stamps the id. Only the `Tape`
  calls it; inference never does.
- The `Tape` issues ids from `next_id` via `_fresh()` (`autograd.mojo:286-289`),
  starting at 1. `track(mut t)` (`autograd.mojo:291-293`) stamps a tensor only
  if its id is currently 0.

This DIVERGES from flame-core, where every tensor gets a global-atomic id at
creation (`TENSOR_ID_COUNTER`). Here ids are **per-Tape** and assigned lazily:
an op output gets `_fresh()` when its record method runs (`autograd.mojo:299`,
`:308`, etc.), and a leaf input is only stamped if `track()` is called on it.
Id `0` is the universal "no gradient flows here" sentinel (see §4.2).

### 1.5 Tracing `a + b` (exact code path)

There is **no free-function `add` that records**. Recording is a *method on the
Tape*: `record_add(mut self, a, b, ctx)` (`autograd.mojo:297-304`):

1. **Compute the result** — `_raw_add(a, b, ctx)` (`autograd.mojo:145-151`)
   launches the self-contained F32 elementwise kernel `_k_add`
   (`autograd.mojo:70-80`) and returns an untracked tensor (id 0).
2. **Stamp the output id** — `oid = self._fresh(); out.set_id(oid)`.
3. **Append the entry** — `TapeEntry(oid, OP_ADD, a.id, b.id, TArc(a.clone(ctx)),
   TArc(b.clone(ctx)))`.

Note `a.clone(ctx)` (`tensor.mojo:69-80`) is a **full device→device byte copy**
into a fresh buffer — NOT a cheap Arc bump of shared storage. This is another
divergence: flame-core's saved-tensor `.clone()` is an Arc increment over
shared `CudaSlice`; Mojo's `Tensor.clone` deep-copies the GPU bytes
(`tensor.mojo:76-79`: `enqueue_create_buffer` + `enqueue_copy` + `synchronize`).
So saving operands costs real GPU memory and a copy per op here.

### 1.6 The 9 recorded ops

Every tape-recording method is a `Tape` method (`autograd.mojo`):

| Method | op_kind | line | forward primitive | saved |
|---|---|---|---|---|
| `record_add` | OP_ADD | :297 | `_raw_add` | a, b (unused by bwd) |
| `record_sub` | OP_SUB | :306 | `_raw_sub` | a, b (unused by bwd) |
| `record_mul` | OP_MUL | :315 | `_raw_mul` | a, b |
| `record_matmul` | OP_MATMUL | :325 | `_raw_matmul` (vendor BLAS) | a, b + dims M,N,K |
| `record_linear` | OP_LINEAR | :338 | `ops.linear.linear` | x, W, b + dims; third_id=b |
| `record_rms_norm` | OP_RMSNORM | :359 | `ops.norm.rms_norm` | x, weight (eps = `_RMS_EPS`) |
| `record_silu` | OP_SILU | :374 | `ops.activations.silu` | x (rhs unused, saved1=clone of x) |
| `record_swiglu` | OP_SWIGLU | :386 | `ops.activations.swiglu` | gate, up |
| `mse_loss` | OP_MSE | :400 | sub→mul→reduce_sum (`autograd.mojo:412-417`) | pred, target |

The RMSNorm eps is a single shared `comptime _RMS_EPS = 1e-6`
(`autograd.mojo:66`) — the `TapeEntry` struct has no Float slot to carry a
per-op eps (the struct fields are "LEAD-OWNED and frozen for this task",
`autograd.mojo:62-65`), so `record_rms_norm` and the backward arm both read
the same constant.

---

## 2. Backward Pass

### 2.1 Entry point

`backward(tape: Tape, loss: Tensor, ctx: DeviceContext) -> Dict[Int, TArc]`
(`autograd.mojo:441-521`). It is a **free function**, not a Tensor method
(there is no `loss.backward()` — the caller passes the tape explicitly). It
returns an id→gradient `Dict[Int, TArc]`.

### 2.2 No validation gate

flame-core's `backward()` checks `loss.requires_grad` and that the loss is
scalar. The Mojo `backward()` has **no such guards** — it seeds the loss id
unconditionally (`autograd.mojo:445`) and walks the tape. The scalar-ness of
the loss is a convention (`mse_loss` emits a scalar via `reduce_sum`,
`autograd.mojo:417`), not an enforced precondition.

### 2.3 Reverse insertion order (no topo sort)

Like flame-core, there is **no explicit topological sort**. The tape is walked
in **reverse insertion order** (`autograd.mojo:447-448`):

```mojo
var i = len(tape.entries) - 1
while i >= 0:
    ...
    i -= 1
```

This is valid because record methods append in forward execution order, and
every op executes eagerly when recorded (no lazy graph).

### 2.4 Gradient initialization

```mojo
var grads = Dict[Int, TArc]()
grads[loss.id] = TArc(ones_like(loss, ctx))   # autograd.mojo:444-445
```

The loss gradient is a tensor of ones (`ones_like`, `autograd.mojo:210-216`,
F32, via the `_k_fill` kernel). There is **no `CompactIndex` / flat
`Vec<Option<Tensor>>`** like flame-core §2.4 — gradient storage is a plain
`Dict[Int, TArc]` keyed by autograd id (see §4). This is simpler but O(hash)
per lookup rather than flame-core's O(1) indexed access.

### 2.5 No frozen-weight filtering

flame-core builds a `needed_grad_ids: HashSet` to skip accumulating gradients
for frozen base weights (a ~5 GB saving for Klein 4B). The Mojo `backward()`
has **no frozen-weight filter** — it accumulates a gradient for every nonzero
input id encountered. The only "drop" mechanism is the id-0 sentinel: any
input left untracked (id 0) silently receives no gradient because `_accum`
early-returns on id 0 (`autograd.mojo:430-431`). This is how `mse_loss`
discards the target gradient — the caller leaves `target` untracked
(`autograd.mojo:421` passes `target.id`, which is 0 for a constant target;
`autograd.mojo:514-516`).

### 2.6 Per-node backward sequence

For each entry in reverse (`autograd.mojo:449-520`):

1. Read `op_kind`, `out_id`, `lhs_id`, `rhs_id` (`autograd.mojo:449-452`).
2. **Look up the output gradient**: `if grads.__contains__(eout): var gop =
   grads[eout]` — a refcount-bump copy of the out-grad (`autograd.mojo:453-454`).
   If no gradient reached this output, the entry is skipped entirely (no
   `take`/remove; the dict keeps the entry).
3. **Dispatch on `op_kind`** through the `if/elif` ladder, computing input
   gradients via the standalone backward kernels and accumulating them with
   `_accum`.

DIVERGENCE: flame-core uses `gradients.take(id)` to *move* the out-grad out of
the map (avoiding a clone and freeing it). The Mojo path does a refcount-bump
copy and never removes the out-grad from the dict, so intermediate gradients
stay resident until the whole `Dict` is dropped.

### 2.7 No checkpoint recomputation inside `backward()`

flame-core's `backward()` has an `Op::Checkpoint` arm that re-enables autograd,
re-runs the forward, drains a sub-tape, and sub-backwards through it. The Mojo
tape's `backward()` has **no Checkpoint op and no recomputation**. Gradient
checkpointing exists in the Mojo port but lives **outside** the tape, as
explicit hand-chained save/recompute helpers in `training/checkpoint.mojo` and
`training/checkpoint_block.mojo` (see §6).

---

## 3. Does Backward Create MORE Tape Entries?

### 3.1 Answer: NO — and for a different reason than flame-core

flame-core needs a lock-free atomic flag so that Tensor methods called *inside*
backward don't re-enter the global mutex and deadlock. **The Mojo port has no
global tape and no lock**, so this entire problem class does not exist.

The backward arms call the standalone backward kernels (`mm_backward`,
`linear_backward`, `rms_norm_backward`, `silu_backward`, `swiglu_backward`,
`mse_backward`) and the raw kernels (`_raw_add`, `_raw_mul`, `_raw_neg`)
directly. **None of these touch the `Tape`** — they are plain functions that
return untracked (id-0) `Tensor`s. So backward provably appends nothing to
`tape.entries`. The "recorded vs raw" split is structural: only `Tape.record_*`
methods append entries; everything in `ops/*_backward.mojo` is tape-blind.

### 3.2 The two-tier op design

This is the Mojo analogue of flame-core's "GpuOps (no autograd) vs Tensor
methods (autograd)" split:

- **Tape-recording tier**: `Tape.record_add/.../mse_loss` — the 9 methods that
  append `TapeEntry`s.
- **Raw / kernel tier**: `_raw_*` in `autograd.mojo` + every function in
  `ops/*_backward.mojo` and the forward `ops/*` — these never see the tape.

Backward lives entirely in the raw tier, so it cannot grow the graph.

---

## 4. Gradient Storage

### 4.1 Data structure: a plain Dict

`Dict[Int, TArc]` — autograd-id → Arc-boxed gradient tensor
(`autograd.mojo:444`). No dual-mode `vec_store`/`overflow`, no `CompactIndex`.
Simpler than flame-core's `GradientMap`, at the cost of hash lookups.

### 4.2 Accumulation: `_accum`

`_accum(mut grads, id, var g, ctx)` (`autograd.mojo:427-437`):

```mojo
if id == 0:
    return                                  # untracked sentinel → drop grad
if grads.__contains__(id):
    var oldarc = grads[id]
    var summed = _raw_add(oldarc[], g, ctx) # GPU add, fresh tensor
    grads[id] = TArc(summed^)
else:
    grads[id] = TArc(g^)                    # first contribution, store as-is
```

Two key behaviors:
- **id 0 is the universal drop**: any gradient routed to id 0 is discarded.
  This is how untracked constants (mse target) and the unused rhs of SiLU
  (`record_silu` sets `rhs_id=0`, `autograd.mojo:382`) get no gradient.
- **Fan-in sums via `_raw_add`**: when a tensor feeds multiple ops, each
  contribution is summed on the GPU. Multi-use accumulation is correct
  (proven by the composition gates, §5).

### 4.3 Dtype / precision policy: F32 interior, storage-dtype boundary

flame-core's policy is `InternalFP32_PublicBF16` — grads stored F32, exported
BF16. The Mojo tape engine is **F32-only throughout** for the 9 wired ops:

- The autograd kernels (`_k_add/_k_sub/_k_mul/_k_neg/_k_fill`,
  `autograd.mojo:70-126`) are all `DType.float32`. `_empty_f32`
  (`autograd.mojo:130-135`) allocates F32 outputs. The comment at
  `autograd.mojo:16` states "F32 only for the spike."
- The **standalone backward kernels** follow the broader flame-core convention
  documented at `ops/attention_backward.mojo` header: *"All interior math is
  F32 (matmuls accumulate F32; softmax/reduction F32). BF16/F16 only at the
  storage boundary (gather casts up, scatter casts down)."* The attention
  backward gather/scatter kernels (`_gather_bf16`/`_scatter_bf16`,
  `ops/attention_backward.mojo`) cast storage BF16↔F32 at the boundary and do
  all matmul/softmax in F32. norm/activation backward files carry `_f32`,
  `_bf16`, `_f16` kernel variants (`ops/activation_backward.mojo:84-196`),
  again computing in F32 and only differing at the load/store cast.

So the **F32-interior / storage-dtype-boundary contract holds**, but the *tape
engine itself* never produces BF16 — BF16/F16 only appear in the per-op
backward kernels' storage paths and in the mixed-precision training loop
(§6, `mixed_precision_parity` MEASURED-by-prior-lead cos 1.0 with F32 master /
BF16 compute).

---

## 5. The 9 Wired Ops + the Composition Contract

### 5.1 Per-op backward arms in `backward()` (`autograd.mojo:455-519`)

| op_kind | arm lines | backward math | calls |
|---|---|---|---|
| OP_ADD | :455-457 | d_a = d_b = grad_out | `_accum` (clone of gop) |
| OP_SUB | :458-460 | d_a = grad_out; d_b = −grad_out | `_raw_neg` |
| OP_MUL | :461-465 | d_a = grad·b; d_b = grad·a | `_raw_mul` (saved0/saved1) |
| OP_MATMUL | :466-478 | d_a = grad@Bᵀ; d_b = Aᵀ@grad | `mm_backward` |
| OP_LINEAR | :479-490 | d_x, d_W, d_b for y=x@Wᵀ+b | `linear_backward` |
| OP_RMSNORM | :491-499 | d_x, d_gamma | `rms_norm_backward(…, _RMS_EPS)` |
| OP_SILU | :500-503 | d_x = silu'(x)·grad | `silu_backward` |
| OP_SWIGLU | :504-511 | d_gate, d_up for silu(gate)·up | `swiglu_backward` |
| OP_MSE | :512-519 | d_pred (incl 2/N); **ignores incoming gop** | `mse_backward` |

Note the MSE arm is a special **loss leaf**: `mse_backward(pred, target)`
produces the full gradient including the 2/N factor and deliberately ignores
the incoming out-grad (`autograd.mojo:512-516`, and the `mse_loss` record
method comment `autograd.mojo:404-411`). The forward scalar is "cosmetic" — it
only exists to give the chain a head.

The `.clone(ctx)` calls in the MATMUL/LINEAR/RMSNORM/SWIGLU arms
(`autograd.mojo:477`, `:488-490`, `:498-499`, `:510-511`) are a Mojo-specific
necessity: you cannot partially move a field out of a still-live struct that
has a destructor ("field destroyed out of the middle of a value",
`autograd.mojo:474-476`), so the multi-output grad structs (`MatmulGrads`,
`LinearGrads`, etc.) are read via `.clone()` rather than moved.

### 5.2 MEASURED tape-engine parity (by prior lead, handoff §2)

Each wired op has a tape-level smoke gate (`autograd_*_smoke.mojo`):

| op | gate | result (MEASURED-by-prior-lead) |
|---|---|---|
| add/sub/mul | autograd_smoke.mojo | cos 1.0 |
| matmul | autograd_matmul_smoke.mojo | cos 1.0 |
| linear (x,W,b) | autograd_linear_smoke.mojo | cos 1.0 |
| rmsnorm | autograd_rmsnorm_smoke.mojo | cos 1.0 |
| silu | autograd_silu_smoke.mojo | cos 0.99999999 |
| swiglu | autograd_swiglu_smoke.mojo | cos 0.99999999 |
| mse | autograd_mse_smoke.mojo | cos 0.99999999 |

### 5.3 The composition contract: d_x(block N) = d_y(block N+1)

The load-bearing proof of correctness is **composition**, not per-op parity.
flame-core's tenet — "one block's d_x = next block's d_y" — is realized here by
`training/dit_block.mojo`'s `dit_block_forward`/`dit_block_backward`
(`dit_block.mojo:269` / `:314`), which hand-chain the backward kernels through
a full DiT block (rms→qkv→sdpa→out→residual→rms→swiglu-MLP→residual).

The backward explicitly preserves the two residual accumulations and two
fan-out sums (`dit_block.mojo:307-313`):
- residual #2 split: `d_r1(partial)=d_y`, `d_mlp=d_y`
- h2 fan-out: `d_h2 = d_h2_g + d_h2_u`
- residual #1 accum: `d_r1 += d_r1_norm`
- h1 fan-out: `d_h1 = d_h1_q + d_h1_k + d_h1_v`
- final: `d_x = d_x_norm + d_r1`

MEASURED-by-prior-lead (handoff §2):
- `block_composed_parity.mojo` — full DiT block backward vs torch + finite-diff:
  d_x cos 0.99999999, all 7 weight/gain grads 0.99999999. **VERDICT: block
  composition sound** (at H=2). Klein-class composition defect ABSENT.
- `stack_train_parity.mojo` — 3 stacked blocks train, loss 10.33→0.0011,
  inter-block d_x→d_y handoff bit-perfect (block0 first-step grads cos 1.0).
- `dit_block_unit_parity.mojo` — the reusable `dit_block_*` matches the inline
  block, cos 0.99999999.

**STATUS UPDATE (this session):** the toy `dit_block.mojo` proofs above use
**H=2**, but the SDPA backward they call is now proven correct at the **real
H=30** (the old "H=30 silent-zero d_q/d_k" was a degenerate-test-data artifact,
not a kernel bug — `MOJO_KERNELS.md` §11, `sdpa_bwd_nondegen_parity.mojo`,
MEASURED-this-session). And the composition contract is no longer proven only on
the toy `dit_block.mojo`: this session built and gated the **real Klein DiT
block + stack composition at H=32** — see §8.

---

## 6. Memory Lifecycle & Checkpointing

### 6.1 Saved-tensor lifetime

Saved tensors are `TArc` (Arc-boxed `Tensor`) inside `TapeEntry`. They are:
1. **Created** by `record_*` via `a.clone(ctx)` — a **deep device copy**
   (`tensor.mojo:69-80`), not an Arc bump of shared storage (divergence from
   flame-core).
2. **Held** by `tape.entries` until the `Tape` is dropped.
3. **Freed** by Arc refcount when the `TapeEntry` (and thus the `Tape`) is
   destroyed — no explicit `tape.clear()` call; Mojo's ownership handles it.

### 6.2 Gradient checkpointing lives OUTSIDE the tape

Unlike flame-core (where `Op::Checkpoint` is a tape op), the Mojo port
implements checkpointing as explicit save/recompute helpers:

- `training/checkpoint.mojo`: `offload_to_host` (`:79`) / `restore_to_device`
  (`:99`) move a saved tensor to host RAM and back; `block_forward` (`:135`),
  `block_backward_saveall` (`:147`), `checkpoint_recompute` (`:174`) are the
  toy save-only-input + recompute pattern. `struct HostOffload` (`:59`),
  `struct BlockGrads` (`:123`).
- `training/checkpoint_block.mojo`: the full-DiT-block version —
  `checkpoint_dit_block` (`:478`), `block_forward_acts` (`:245`),
  `block_backward_from_acts` (`:295`), with `struct DitBlockWeights` (`:93`),
  `struct BlockActs` (`:174`), `struct BlockGrads` (`:137`).

MEASURED-by-prior-lead (handoff §2): `checkpoint_parity.mojo` (toy) +
`checkpoint_block_parity.mojo` (full block) — save-only-input + recompute,
3-way (self/torch/byte-exact), full-block dx cos 0.99999999, host-offload
round-trip max_abs 0.

---

## 7. The Training Spine (what wraps the tape)

The tape engine is only the inner loop. The surrounding training machinery
(all in `serenitymojo/training/`):

- **optim.mojo**: `adamw_step` (`:142`, in-place F32 AdamW, decoupled WD, host
  bias-correction), `sgd_step` (`:198`, momentum + decoupled WD),
  `clip_grad_global_norm` (`:234`, 2-tensor global-L2 clip). All F32, all
  in-place. MEASURED-by-prior-lead: AdamW bowl ratio 9.8e-15, SGD+mom 1.4e-16.
- **schedule.mojo**: `flow_match_noise_target` (`:309` — the real v-target:
  `x_t=(1−σ)·latent+σ·noise`, `target=noise−latent`),
  `sample_timestep_logit_normal` (`:253`, ChaCha12/PCG32 RNG → logit-normal +
  FLUX shift), `ema_update` (`:374`), `grad_accumulate` (`:410`).
- **loop.mojo**: `struct TrainState` (`:79`), `save_checkpoint` (`:183`) /
  `load_checkpoint` (`:227`) — the F32-master/BF16 resumable harness.
  MEASURED-by-prior-lead: trains 1.04→0.0027, checkpoint round-trip byte-exact,
  resume continues descent (`loop_parity.mojo`).
- **dit_block.mojo**: the hand-chained block forward/backward (§5.3).
- **zimage_train_step.mojo**: a host-side single train-step scaffold (`main`,
  `:141`).

---

## 8. The Hand-Chained Klein Composition Layer  ✅ BUILT THIS SESSION

§5.3 proved the composition contract on the **toy** `dit_block.mojo` (a single
generic block, H=2). This session built the contract **end-to-end for a real
model** — the full Klein (FLUX.2) DiT, double + single stream, with and without
LoRA — by hand-chaining the already-verified backward kernels through real
block topology. **No new autograd kernels; pure composition** of the §1-§4 arms
plus the standalone `*_backward.mojo` kernels (incl. the new `modulate_backward`,
`MOJO_KERNELS.md` §6b). None of this touches the `Tape` — it is all in the
raw/kernel tier (§3.2), hand-chained in reverse.

All files under `serenitymojo/models/klein/`.

### 8.1 The inter-block handoff contract: `d_x(block N) = d_y(block N+1)`

The same contract as §5.3, now realized across **two different block types and
their seam**. Forward order is `input-proj → N double-stream blocks → concat →
M single-stream blocks → final layer`; backward walks it in exact reverse, and
at every seam the deeper block's input grad **is** the shallower block's output
grad (`klein_stack.mojo:46-62`, header). Specifically:

- **Final layer → single stack seam.** The final layer reads **only** the img
  rows (`out = linear(modulate(layer_norm(img_out)), Wf)`), so the single
  stack's d_x seed is `concat(zeros[N_TXT,D], d_img_out)` — txt rows get zero
  because the forward never read them (`klein_stack.mojo:339`+ backward body).
- **Single→single, double→double.** Each `single_block_backward` /
  `double_block_backward` returns the grad wrt its input, which is fed directly
  as the next-deeper block's output grad. Double blocks carry the **full
  `[S,D]`** stream (both txt and img) across the handoff; double blocks return
  `(d_img_x, d_txt_x)` consumed as the previous block's `(d_img_out, d_txt_out)`.
- **Double→single seam.** The single stack's final `d_x [S,D]` splits back into
  `d_txt [N_TXT,D]` and `d_img [N_IMG,D]` — the inverse of the forward
  `concat(1, txt, img)`.

### 8.2 The two real blocks (forward + backward + LoRA variants)

**`double_block.mojo`** — the double-stream (img+txt) block.
- `double_block_forward` (`:520`) / `double_block_backward` (`:787`).
- `double_block_lora_forward` (`:1004`) / `double_block_lora_backward` (`:1251`)
  — LoRA on img/txt × qkv/proj (4 adapters/block) via `klein_lora_*` (§ in
  `MOJO_KERNELS.md` §11b).
- The backward explicitly reconstructs the residual/fan-out structure per
  stream (`_stream_post_backward` `:628`, `_stream_pre_backward` `:733`).
- **MEASURED-this-session:** `double_block_parity.mojo` — **28/28 grads vs torch
  at H=32**, cos ≈ 1.0 (verified against `double_block_oracle.py`).
  `double_block_lora_parity.mojo` — **8 LoRA grads** (img/txt × qkv/proj d_A/d_B)
  cos ≥ 0.999999999.

**`single_block.mojo`** — the single-stream (parallel-attn+MLP) block.
- `single_block_forward` (`:371`) / `single_block_backward` (`:445`).
- `single_block_lora_forward` (`:617`) / `single_block_lora_backward` (`:688`)
  — LoRA on qkv (w1 rows) + out (w2 cols), 2 adapters/block.
- **MEASURED-this-session:** `single_block_parity.mojo` — **9/9 grads** cos ≈ 1.0
  (`single_block_oracle.py`). `single_block_lora_parity.mojo` — **4 LoRA grads**
  cos ≥ 0.999999999.

### 8.3 The full stack with per-block recompute checkpointing

**`klein_stack.mojo`** — the BASE full Klein DiT composition.
- `klein_stack_forward` (`:267`) saves only what backward needs;
  `klein_stack_backward` (`:339`) walks blocks in reverse.
- **Per-block RECOMPUTE checkpointing (the memory contract):** the backward does
  **not** retain every block's saved activations across the whole forward.
  Instead it **re-runs each block's forward** (one block, cheap) to regenerate
  that block's `saved`, then runs that block's verified backward
  (`klein_stack.mojo:15-21` header). Peak memory stays at ~one block's
  activation footprint + the resident inter-block stream tensors — which is what
  lets the real **8 double + 24 single** depth fit 24 GB without OOM. This is
  the §6.2 `checkpoint_block.mojo` idea applied per-block across the stack
  (the Mojo analogue of flame-core's `Op::Checkpoint`, but hand-written outside
  any tape).
- **MODULATION IS SHARED across all blocks** of a kind. `img_mod`/`txt_mod`/
  `single_mod` (`ModVecs`/`SingleModVecs`) are computed once by the caller and
  reused by every block, mirroring `klein_dit.mojo::forward_full`. The backward
  produces a modvec grad per block and **sums them across blocks** (shared
  parameter ⇒ fan-in accumulation), but does **not** backprop them into the
  modulation MLP — that link is the deferred finetune phase
  (`klein_stack.mojo:64-70` SCOPE).
- **MEASURED-this-session:** `klein_stack_parity.mojo` — full-Klein composition
  vs torch (`klein_stack_oracle.py`) cos ≈ 1.0, incl. the load-bearing
  input-token grads. `klein_stack_real_smoke.mojo` — **real depth 8+24 on real
  weight shapes**, finite grads, **no OOM**.

**`klein_stack_lora.mojo`** — the full stack with LoRA on every trained
attention projection.
- `KleinLoraSet` (`:112`) — one flat `List[LoraAdapter]` for **80 adapters** at
  real depth (8×4 + 24×2), indexed by a deterministic
  `(block_kind, block_idx, slot)` scheme (`:24-33` header,
  `_klein_lora_prefix` `:486`). `build_klein_lora_set` (`:145`) inits each (A
  small randn, B=0 — PEFT identity at step 0).
- `klein_stack_lora_forward` (`:245`) / `klein_stack_lora_backward` (`:312`) —
  same base composition as `klein_stack.mojo` with the per-block calls swapped
  for the LoRA variants; backward **scatters** each block's returned d_A/d_B
  back into the flat `KleinLoraGrads` (`:194`).
- `klein_lora_adamw_step` (`:462`) — walks both flat lists in lockstep and runs
  the trainer's `_lora_adamw` on every adapter (reused, not reimplemented).
- `save_klein_lora` (`:519`) — emits all 80 adapters via the PEFT/ai-toolkit
  writer (`save_lora_peft`); `load_klein_lora_resume` (`:540`) reloads them in
  the same flat order (AdamW moments zeroed, resumed from a `loop.mojo`
  TrainState checkpoint).
- **MEASURED-this-session:** `klein_stack_lora_parity.mojo` — 80-adapter stack,
  d_A/d_B vs torch (`klein_stack_lora_oracle.py`) cos ≥ 0.999; AdamW step applied;
  `save_klein_lora` round-trips **byte-exact**. `klein_stack_lora_real_smoke.mojo`
  — real depth, no OOM.

### 8.4 Why this is the capstone

This is the first time the Mojo port has a **complete, real-model training
backward** that is parity-gated against torch end-to-end (not just per-op, not
just a toy block). The §5.3 "block composition sound" verdict now extends to the
real Klein topology at H=32, on real weight shapes, at real depth, under the
recompute-checkpoint memory budget. The autograd *tape* engine (§1-§5) is the
inner loop; this composition layer is the hand-chained outer model that proves
the kernels compose into a trainable Klein DiT.

---

## Appendix: File Index (Mojo)

| File | Role |
|---|---|
| `autograd.mojo` | Tape engine: TapeEntry, Tape, 9 record methods, `backward()`, `_accum`, raw F32 kernels |
| `tensor.mojo` | Tensor struct (move-only), `id` field, `set_id`, deep-copy `clone` |
| `ops/linalg_backward.mojo` | mm/bmm/linear/addbias backward + grad structs |
| `ops/norm_backward.mojo` | rms/layer/group norm backward + grad structs |
| `ops/activation_backward.mojo` | relu/sigmoid/tanh/silu/gelu backward (F32/BF16/F16) |
| `ops/loss_swiglu_backward.mojo` | mse/huber backward + swiglu backward |
| `ops/reduce_backward.mojo` | sqrt/square/log/softmax/logsoftmax/sum/mean backward |
| `ops/rope_struct_backward.mojo` | rope/qkv-split/gate-residual backward |
| `ops/conv2d_backward.mojo` | conv2d dx/dw/db backward |
| `ops/pool_backward.mojo` | maxpool2d / upsample-nearest backward |
| `ops/celoss_embed_backward.mojo` | cross-entropy/nll/bce/embedding backward |
| `ops/shape_backward.mojo` | reshape/permute/transpose/cat/split/slice/broadcast/repeat/where/clamp/max/min/index-select backward |
| `ops/attention_backward.mojo` | decomposed SDPA backward (⚠ H=30 bug — see MOJO_KERNELS.md) |
| `training/optim.mojo` | AdamW/SGD/grad-clip |
| `training/schedule.mojo` | flow-match target, timestep RNG, EMA, grad-accum |
| `training/checkpoint{,_block}.mojo` | gradient checkpointing (outside the tape) |
| `training/dit_block.mojo` | hand-chained toy DiT block fwd/bwd (composition contract, H=2) |
| `training/loop.mojo` | TrainState, save/load checkpoint, F32-master/BF16 loop |
| `ops/elementwise_backward.mojo` | modulate (AdaLN) backward — last missing arm (THIS SESSION) |
| `models/klein/double_block.mojo` | real Klein double-stream block fwd/bwd + LoRA variants (28/28 grads, THIS SESSION) |
| `models/klein/single_block.mojo` | real Klein single-stream block fwd/bwd + LoRA variants (9/9 grads, THIS SESSION) |
| `models/klein/klein_stack.mojo` | full Klein DiT composition, per-block recompute checkpoint (THIS SESSION) |
| `models/klein/klein_stack_lora.mojo` | full Klein stack + 80 LoRA adapters, AdamW step, PEFT save (THIS SESSION) |
| `models/klein/lora_block.mojo` | per-projection LoRA fwd/bwd (composes linear_backward, THIS SESSION) |

---

## Divergences from flame-core (summary for the lead)

1. **No global tape / no mutex.** Explicit threaded `Tape` value. The atomic
   re-entrancy guard from flame-core is unnecessary and absent.
2. **`op_kind: Int` ladder, not a 47-variant `Op` enum.** Only 9 ops wired
   into `tape.backward()`; the other ~60 backward kernels exist standalone and
   are hand-chained (see MOJO_KERNELS.md).
3. **Saved tensors are deep device copies** (`Tensor.clone`), not Arc bumps of
   shared storage — real GPU memory + copy cost per recorded op.
4. **Every op saves both operands** even when backward doesn't need them
   (flagged micro-opt-for-later, `autograd.mojo:18-20`).
5. **Gradient store is a plain `Dict[Int, TArc]`**, not a `CompactIndex`-backed
   flat vec. No `take`/move-out; intermediates persist until the Dict drops.
6. **No frozen-weight gradient filter.** The only drop is the id-0 sentinel.
7. **No Checkpoint tape op.** Checkpointing is explicit save/recompute helpers
   outside the tape — now realized at model scale as **per-block recompute** in
   `klein_stack.mojo` (§8.3), fitting real 8+24 depth in 24 GB with no OOM.
8. **F32-only tape engine.** BF16/F16 appears only in the standalone per-op
   backward kernels' storage boundary and the mixed-precision loop, not in the
   9 wired-op tape kernels.
9. **Per-Tape lazy ids**, not a global atomic id counter.
