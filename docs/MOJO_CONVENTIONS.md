# serenitymojo conventions and gotchas (training port)

> The Mojo 1.0.0b1 idioms, dispatch tricks, and naming rules that each cost
> real time during the flame-core→Mojo training port. Read once, save a day.
> Modeled on flame-core/docs/FLAME_CONVENTIONS.md, folding in the hard-won
> idioms from HANDOFF_2026-05-30_MOJO_TRAINING_PORT_MASTER.md §4.
>
> Every idiom below is cited to either a `file:line` read in serenitymojo/ or
> to handoff §4. Items the handoff asserts but that could NOT be corroborated
> in source are flagged **[UNVERIFIED-IN-SOURCE]** at the end.

This is the "how not to waste a day in this codebase" doc.

---

## 0. The build / run dance (do this EVERY time)

```bash
cd /home/alex/mojodiffusion
rm -f serenitymojo.mojopkg          # MANDATORY before every run
pixi run mojo run -I . serenitymojo/path/to/file.mojo
```

- **`rm -f serenitymojo.mojopkg` before EVERY run** (handoff §4). A stray
  0-byte `.mojopkg` shadows the source tree → `"invalid magic bytes"`. This is
  the single most common time-waster.
- **`mojo run -I .` (transitive compile), NEVER `mojo package`** (handoff §4,
  §156). `mojo run -I . serenitymojo/path/to/file.mojo` compiles the named file
  AND transitively compiles every `serenitymojo.*` it imports on the fly — this
  is the working path for every gate/smoke. `mojo package` FAILS here because the
  in-package `*_smoke.mojo` / `*_parity.mojo` files each define their own
  `main()`, and packaging a tree with multiple `main()`-defining files conflicts.
  So: run files directly with `-I .`; do not try to pre-package the tree.
- **Build SERIAL only.** Concurrent compiles corrupt the shared compile cache
  → false `"invalid magic bytes"` AND false `"unimportable symbol"` reports.
  The "`mse_backward` unimportable" scare was THIS, three separate times — it
  imports fine on a clean serial run (handoff §4). **Do not launch a second
  `pixi run mojo` while one is in flight.** (For the doc agent this is moot —
  we only Read + Write markdown — but it is the rule for anyone compiling.)
- **toolchain location**: `pixi` is on PATH, or at `/home/alex/.pixi/bin/pixi`
  if bare `pixi` is missing (handoff §4).

---

## 1. Mojo 1.0.0b1 language idioms (the verbatim list)

These are syntax/semantics facts of the pinned toolchain. Getting any one
wrong is a compile error, not a runtime bug.

| Idiom | Rule | Evidence |
|---|---|---|
| `def` not `fn` at top level | `fn` is deprecated at module top level; use `def` | handoff §4; every `ops/*_backward.mojo` uses `def` |
| move-only returns | `return out^` (the `^` transfers ownership) | handoff §4; `tensor.mojo` ctors |
| `comptime` not `alias` | compile-time consts are `comptime X = ...` | `autograd.mojo:48-60` (`comptime TArc`, `comptime OP_ADD = 0`) |
| `STDtype.F32` is a VALUE | NOT `STDtype.f32()` — it's an enum value, no call | handoff §4; `io/dtype.mojo` |
| `from_host` arg order | `from_host(values, shape, dtype, ctx)` — **values first, ctx last** | handoff §4; `tensor.mojo` ctor block |
| no-bias linear | `linear(x, w, Optional[Tensor](), ctx)` — pass an empty Optional for the bias | handoff §4 |
| `ref` is a RESERVED WORD | never name a variable `ref` | handoff §4 |
| `List.copy()` may be missing | write a small helper if a deep copy is needed | handoff §4 |

---

## 2. The move-only Tensor — the dominant structural constraint

`struct Tensor(Movable)` (`tensor.mojo:32`) is **Movable, NOT Copyable** — it
uniquely owns its `DeviceBuffer`. This drives nearly every awkward pattern in
the training spine:

### 2a. Boxing for collections — `TArc = ArcPointer[Tensor]`
A move-only type can't be a `List`/`Dict` element (Mojo collections need
`Copyable`). The tape boxes every saved/grad tensor as
`comptime TArc = ArcPointer[Tensor]` (`autograd.mojo:48`) — a Copyable refcount
bump. This is the SAME idiom `offload/block_loader.mojo` /
`models/vae/wan22_decoder.mojo` / `sensenova_u1` already use for
`Dict[..., Tensor]` (`autograd.mojo:10-12` header). No manual alloc/free — the
Arc refcount frees saved/grad tensors.

### 2b. Multi-return via a struct, NOT a bare tuple
A backward that returns several grads returns a **`struct X(Movable)`**, not a
tuple (handoff §4). Examples in source: `RmsNormBackward` (returns d_x/d_g),
`SwigluGrads` (d_gate/d_up) — both imported in `autograd.mojo:38-42`. The
shape/celoss backward headers state the same rule ("multi-output arms return a
Movable struct — Tensor is move-only", `ops/shape_backward.mojo` header).

### 2c. CLONE struct grad fields — don't move 2+ fields out of a live struct
Moving two-or-more fields out of a still-live struct triggers `"field destroyed
out of the middle of a value"`. Instead **clone**: `g.d_x.clone(ctx)` (handoff
§4; `Tensor.clone(ctx)` exists per handoff §2). `training/dit_block.mojo`'s
header documents the same: grads cross its API as host `List[Float32]`, not GPU
Tensors, precisely to dodge the move-out problem.

### 2d. Consume-once grad carriers
`SdpaGrads` must be consumed once into a Copyable **host** carrier — see
`block_composed_parity`'s `SdpaHostGrads` (handoff §4). The pattern recurs:
when a backward yields multiple device tensors, read them to host immediately
into a Copyable struct rather than passing the move-only carrier around.

### 2e. No storable closures — grads-as-input, recompute open-coded
Mojo 1.0.0b1 **cannot store a heterogeneous captured closure in a struct
field** (no boxed `dyn Fn` / existential trait object). This is documented at
length in `training/checkpoint.mojo`'s header. Consequences:
- flame-core's `recompute_fn: impl Fn(&Tensor)->Result<Tensor>` stored in the
  TapeEntry has **no Mojo equivalent** — `training/checkpoint.mojo` and
  `training/checkpoint_block.mojo` **open-code** the recompute instead.
- `training/loop.mojo` is **grads-as-input, NOT callbacks** (`loop.mojo` header):
  the harness can't hold a `forward_backward_fn`; the caller runs its own
  fwd+bwd and hands grads back via `accumulate_grads`.
- `training/zimage_train_step.mojo` keeps host-list copies of intermediates and
  REBUILDS fresh tensors at each backward call (because each op consumes its
  move-only inputs) — `zimage_train_step.mojo` header.
- **Per-block recompute is the standing memory pattern for full stacks.**
  `models/klein/klein_stack.mojo:15-16, 181-184, 337, 389-423` does NOT retain
  per-block saved activations; the backward retains only each block's INPUT and
  RECOMPUTES that block's saved acts in reverse before running the verified
  per-block backward. This is the same gradient-checkpoint idea as
  `training/checkpoint_block.mojo`, applied per block to bound memory so the
  full-depth Klein-9B stack (8 double + 24 single at D=4096) fits — proven to run
  finite end-to-end in `models/klein/parity/klein_stack_real_smoke.mojo`.

---

### 2f. `Copyable` structs still need `.copy()` to move a field into a var
Even a `struct X(Copyable, Movable)` does NOT let you bind one of its fields to a
fresh `var` by plain assignment — `var x = s.field` tries to MOVE the field out
of a still-live `s` (the same "destroy out of the middle" hazard as 2c), because
the field's own type may be move-only or the binding consumes it. Spell it
`.copy()`:
```mojo
optim = oa.extra.copy()   # io/train_config_reader.mojo:321 (oa.extra is OptimExtra)
```
This recurs across the config + LoRA spine — `struct TrainConfig(Copyable,
Movable)` (`training/train_config.mojo:15`), `struct OptimExtra(Copyable,
Movable)` (`io/train_config_reader.mojo:46`), `struct LoraAdapter(Copyable,
Movable)` (`training/train_step.mojo:120`) — and the LoRA forward helpers reach
for `lo.a.copy()` / `lo.b.copy()` / `x_h.copy()` when staging a struct's host
`List[Float32]` field into `Tensor.from_host` (`training/train_step.mojo:169-176`).
Rule of thumb: any `var y = some_struct.list_or_tensor_field` should be
`some_struct.field.copy()` (or `^` if you genuinely want to consume the source).

---

## 3. Device-buffer / LayoutTensor conventions

### Buffers are `DType.uint8` → bitcast at the op boundary
Every `Tensor` stores raw bytes in a `DeviceBuffer[DType.uint8]`
(`tensor.mojo:32` doc). Kernels allocate
`enqueue_create_buffer[DType.uint8](n * bytesize)` and `.bitcast` to
`DType.float32` / `bfloat16` at the `LayoutTensor` boundary (handoff §4;
`ops/reduce_backward.mojo` header: "device buffers are DType.uint8, bitcast to
Float32 at the LayoutTensor boundary"). This keeps `Tensor` monomorphic — no
dtype type-parameter threaded through every module.

### F32 interior, storage dtype only at the edges
Every `ops/*_backward.mojo` runs **F32 interior** (matmuls accumulate F32,
reductions F32); BF16/F16 appears only at the gather (cast up) / scatter (cast
down) boundary. The training masters are F32 throughout
(`ops/loss_swiglu_backward.mojo` header: "the storage dtype IS F32 here — the
training engine keeps loss/activation backward in F32 master precision").
This mirrors flame-core's "no F32 fallback in *inference*" rule inverted: the
training path is deliberately F32-master.

### Vendor BLAS matmul import
```mojo
from linalg.matmul.vendor.blas import matmul   # transpose_a/transpose_b + c_row_major
```
(`autograd.mojo:32`; `ops/linalg_backward.mojo` header uses the same.) **Do NOT
name a top-level `def` with a token that collides with the imported `matmul`** —
the linalg backward names its wrapper `mm_backward` for exactly this reason
(handoff §4; `autograd.mojo:37` imports `mm_backward`).

### Kernel launch scaffolding (the sibling-copy template)
The proven shape for a new backward kernel (stated in `ops/reduce_backward.mojo`
+ `ops/shape_backward.mojo` + `ops/celoss_embed_backward.mojo` headers, which
all say "mirror `ops/attention_backward.mojo`, the proven SDPA-bwd template"):
- one **flat thread per element** for elementwise arms;
- one **block per row + shared-memory F32 tree-reduction** for the
  softmax/norm family (reuse the `_softmax_bwd_rows_f32` row approach from
  `attention_backward.mojo`);
- `ctx.synchronize()` then return a fresh `Tensor`;
- integer indices passed host-side as `List[Int]`, staged into a device int32
  buffer (the `index_select_backward` pattern).

---

## 4. Layout conventions (match the forward EXACTLY)

The backward MUST match its forward's layout byte-for-byte, or parity silently
drifts. From the backward headers:

| Op family | Layout | Source |
|---|---|---|
| SDPA fwd+bwd | **BSHD** `[B,S,H,Dh]` (storage dtype); returns d_q/d_k/d_v BSHD | `ops/attention_backward.mojo` header |
| conv2d fwd+bwd | x **NHWC** `[N,Hi,Wi,Cin]`, weight **RSCF** `[Kh,Kw,Cin,Cout]`, grad_y NHWC | `ops/conv2d_backward.mojo` header |
| pool/upsample bwd | **NHWC** throughout (VAE path) | `ops/pool_backward.mojo` header |
| group_norm bwd | **NHWC**, per `(n, group)` reduction | `ops/norm_backward.mojo` header |
| rms/layer_norm bwd | one block per row, feature dim D last | `ops/norm_backward.mojo` header |

**LayerNorm uses BIASED variance** (matches the forward + torch) —
`ops/norm_backward.mojo` header. **MaxPool tie-break = FIRST max** (lowest
row-major (kh,kw) flat index), recomputed from x so d_x is byte-position
comparable to torch — `ops/pool_backward.mojo` header.

### The asymmetric-pad downsample trap (VAE encoder)
diffusers `Downsample2D` is **NOT** a symmetric `pad=1` stride-2 conv. It is an
**asymmetric (0,1,0,1) pad (right + bottom only) followed by a stride-2, pad=0
("valid") 3×3 conv** — `models/vae/klein_encoder.mojo:14-15, 231-267, 321-322`
(`_pad_right_bottom_nhwc` concats a zero column on the W axis then a zero row on
the H axis of the NHWC tensor, then runs the stride-2 valid conv). Getting this
wrong (using symmetric pad=1, or padding left/top) shifts every downsampled
feature by half a pixel and the VAE latent silently drifts — it will not crash,
it will produce a wrong-std latent (see the encode gate, `pipeline/
klein_encode_smoke.mojo`, std ≈ 0.96 correct). The output spatial size is
`((H+1+0-3)//2 + 1) = H//2`, matching diffusers exactly
(`klein_encoder.mojo:272-274`).

### RoPE: two layouts, both the same 2×2 rotation
`ops/rope_struct_backward.mojo` ports both pairings (header):
- **INTERLEAVED** (FLUX/Klein): pair `(2i, 2i+1)`.
- **HALFSPLIT** (Z-Image): pair `(i, i+half)`.

Both backward as `R(theta)^T = R(-theta)`: `dx0 = g0·c + g1·s`,
`dx1 = -g0·s + g1·c`. cos/sin are non-learnable tables → only `d_x`, no grad on
the angle tables. (This is the Mojo analog of flame-core's `RopeLayout` tag
hazard — here the two pairings are separate code paths, so there is no
shape-sniffing trap, but you must call the right one to match the forward.)

---

## 5. Tape-wiring discipline

When wiring a new op into `autograd.mojo`'s `tape.backward()`:
1. Add `comptime OP_<X> = <next int>` (`autograd.mojo:52-60` shows 0–8 taken).
2. Extend `TapeEntry` if the op needs a 3rd input — there's a `third_id` +
   `saved2: Optional[TArc]` slot already (`autograd.mojo:244`), used by `linear`
   (x, W, b).
3. Add the dispatch arm in `backward()` (`autograd.mojo:441`).
4. **`grep -c "elif ek == OP_X"` after editing** — silent Edit failures
   inserted NO arm twice last session, surfacing only as a runtime
   `DictKeyError` (handoff §4). Verify the arm landed.
5. Add `autograd_<x>_smoke.mojo` driving the op through `tape.backward()` vs
   torch.

Only **9 ops** are tape-wired (OP_ADD..OP_MSE). The other ~68 backward arms are
hand-chained through `training/dit_block.mojo` — tape-wiring them is optional
for the T5 Z-Image run (master handoff §3 item 3).

---

## 6. The Python oracle — run it SEPARATELY

Parity references come from a Python/torch oracle, generated offline and read
in as host `List[Float32]` (`parity.mojo` header: "nothing here touches
Python").

- **Oracle venv**: `/home/alex/serenityflow-v2/.venv/bin/python` (torch 2.x +
  CUDA) — master handoff §6.
- **Run the oracle as a SEPARATE command, NOT chained with `&&` after the
  `mojo run`** (handoff §4). Chaining produces an `Errno 9` on the oracle's
  file write. Generate the reference first, then run the Mojo gate that reads
  it.

---

## 7. Reading NUL-corrupted files

`ops/attention_backward.mojo` (and "some files") display NUL-byte corruption
under `Read` / `cat` / `grep` — a **display artifact only** (0 real NUL bytes,
handoff §4 + master handoff §1). Read them with:

```bash
python3 -c "print(open('serenitymojo/ops/attention_backward.mojo','rb').read().decode('utf-8','replace'))"
```

Do the same for any file that renders as garbage — it's the same artifact, not
real corruption.

---

## 8. Naming conventions (observed in the tree)

| Pattern | Meaning | Example |
|---|---|---|
| `*_backward.mojo` | a backward-kernel module (training) | `ops/norm_backward.mojo` |
| `*_smoke.mojo` | a single-op / single-module smoke (defines `main()`) | `ops/embed_smoke.mojo`, `autograd_silu_smoke.mojo` |
| `*_parity.mojo` / `*_bwd_parity.mojo` | a cos-gate vs a torch host reference | `ops/parity/sdpa_bwd_parity.mojo` |
| `*_probe.mojo` | an inference forward probe / localization spike | `models/dit/flux1_dit_probe.mojo` |
| `*_contract.mojo` | a model shape-contract (config/dims) | `models/dit/zimage_l2p_contract.mojo` |
| `mm_backward` | matmul backward (renamed to avoid the imported `matmul` token) | `ops/linalg_backward.mojo` |
| `OP_<NAME>` | a tape op-kind `comptime` int | `autograd.mojo:52` |

Multi-return backward structs follow `<Op>Grads` / `<Op>Backward`
(`SwigluGrads`, `RmsNormBackward`).

---

## 9. Cross-reference to flame-core source

The port mirrors flame-core math verbatim, op by op. The backward headers cite
their flame-core origin — keep these for when a parity number drifts and you
need the reference math:
- SDPA bwd ← `flame-core/src/autograd.rs:1686` (`attention_backward_recompute`)
- AdamW ← `flame-core/src/adam.rs` (decoupled WD, the klein LoRA_A receipt)
- gelu-tanh derivative ← `flame-core/kernels/gelu_backward.cu`
- swiglu bwd ← `flame-core/kernels/swiglu_backward.cu`
- checkpoint ← `flame-core/src/autograd.rs:~3208` (`checkpoint_offload_boundary`)
- shape grads ← `flame-core/src/autograd.rs` @ `7be76ef` (line cites in
  `ops/shape_backward.mojo` header: maximum 6114, where 6157, repeat 5749,
  index_select 5869)
- schedule ← EDv2 `train_qwenimage.rs` (flow-match v-target, qwen shift)

Tenet 1 ("fix the primitive, ship every model") applies: the sdpa-bwd fix
belongs in `ops/attention_backward.mojo`, NOT in a trainer or a Z-Image-specific
workaround (master handoff §1 "FIX BELONGS IN attention_backward.mojo").

---

## [UNVERIFIED-IN-SOURCE] — flagged for the lead

These handoff §4 idioms are plausible and consistent with the tree, but I did
NOT open the specific line that proves them (the doc agent only Reads markdown +
module headers, and does not compile):
- The exact spelling `enqueue_create_buffer[DType.uint8](n*bytesize)` — asserted
  in handoff §4; the uint8-buffer + bitcast pattern IS corroborated in the
  backward-module headers, but the literal `enqueue_create_buffer` call site was
  not opened.
- `List.copy()` "may be missing" — handoff §4 caveat; not exercised in any
  header I read.
- The `Errno 9` on `&&`-chained oracle writes — handoff §4; a process/FS
  behavior, not visible in source.
- The "concurrent compiles corrupt the shared cache" and "false invalid magic
  bytes" failures — handoff §4 build-environment facts, not in-source.
- `SdpaHostGrads` in `block_composed_parity` — cited by handoff §4; I read the
  `block_composed_parity.mojo` filename in the tree but did not open its body to
  confirm the struct name.

All are build/runtime idioms a compiling agent should treat as authoritative
from the handoff; flagged only because Tenet 4 (measurement beats assertion)
requires honesty about what I read vs relayed.
