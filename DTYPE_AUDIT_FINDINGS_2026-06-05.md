# Runtime Dtype Contract — Audit Findings (2026-06-05)

> READ-ONLY audit of `/home/alex/mojodiffusion/serenitymojo` against the Runtime
> Dtype Contract in `CLAUDE_CODE_READONLY_DTYPE_AUDIT_PROMPT.md`. Two-agent
> sweep + a deeper second pass each. **No files were edited.** Every claim below
> is backed by an exact `file:line` from a tool result; precision-only and
> training-path observations are separated from contract violations.

## Contract (recap)
BF16 in/out at tensor boundaries; F32 allowed for **compute internals only**
(GEMM accumulators, reductions, norm/attention score math, scalar
schedules/sigmas, host stats, file-format conversion). Ops needing F32 must cast
internally and **return storage dtype**. Flame/Core style: **BF16 in/out, F32
inside compute.**

---

## Headline result

**No high-confidence runtime dtype-contract bugs found.** Every production
**inference** weight path resolves to BF16-preserving load + mixed GEMM (BF16
weights · F32 activations), or to a documented full-F32 model (PiD). The autograd
tape and all backward kernels preserve storage dtype. The remaining items are
(a) a few cross-boundary F32 *widening* loaders to confirm, (b) dead-for-inference
F32 resident structs to migrate-or-document, and (c) two stale comments.

### Important correction to the first pass
The first models/-pass flagged F32 weight uploads in Klein/QwenImage/Flux-stack/
sd35 as likely violations. The **deeper pass refuted this**: those F32 structs are
the **training / LoRA / parity / resident-fallback** paths, *not* what the generate
pipelines execute. Production Klein/Qwen inference streams BF16 via the offload
loader and runs mixed GEMM — see `models/klein/klein_stack_lora.mojo:304-310`
(`_block_tensor_base`, explicit rationale: keep BF16 on disk, let mixed GEMM
consume F32 activations with BF16 weights). Do not "fix" the training structs as
if they were inference bugs.

### Note on `autograd.mojo` history (measured)
Earlier this session a read of `serenitymojo/autograd.mojo` showed it **F32-only**
(header "F32 only for the spike", `_empty_f32` everywhere). A later read showed it
**rewritten** to dtype-following (header L15-18: "Tape storage follows the runtime
dtype: BF16/F16 activations and saved tensors stay BF16/F16, while GEMM/reduction
math may allocate transient F32 compute buffers"). The file changed between reads.
**Current measured state: dtype-following and contract-compliant** (full op- and
backward-arm trace below). The old F32 spike state no longer exists in the tree.

---

## High Confidence Bugs
**None** (ops/tensor/autograd/io scope AND models/training/vae/sampling/pipeline/
offload scope).

The 3 previously-known issues are confirmed **FIXED** in the current tree:
- conv1d F32 weight/bias staging — `ops/conv1d.mojo:50-53/92/136`, output sized to
  `x.dtype().byte_size()` (`:220`), bias-dtype-mismatch raise (`:227`).
- `shape_backward.where_backward` F32 cond-mask — dtype-matched dispatch
  `ops/shape_backward.mojo:705-713`.
- `reduce_var`/`reduce_std` hardcoded F32 — now pass `x.dtype()`
  `ops/reduce.mojo:374-388`; only explicit `*_f32` variants force F32.

---

## Needs Human Decision

**1. `Tensor.from_view_as_f32` (WIDENS BF16/F16→F32) has production callers — confirm legitimacy.**
`tensor.mojo:262-310`. Docstring claims "LTX-2 vocoder parity path," but grep shows
non-smoke callers:
- `models/pid/pid_net.mojo:72`
- `pipeline/anima_text_context.mojo:57`
- `models/anima/anima_text_context.mojo:219`
- `pipeline/giger3_prepare.mojo:119`
These are plausibly legit (PiD is documented full-F32; text-encoder/caption prep is
a host/oracle boundary), but each should carry a justifying comment. **Decision:**
confirm each is a real F32-needed boundary, or convert to BF16-preserving. Confidence: medium.

**2. Klein/QwenImage dual weight-struct architecture — F32 resident structs are live code reached only OFF the production offload path.**
F32 resident structs: `models/klein/double_block.mojo:244-249`,
`models/klein/single_block.mojo:219-229`, `models/klein/klein_stack.mojo:197-201`;
`models/qwenimage/qwenimage_block.mojo:181-194`,
`models/qwenimage/qwenimage_stack.mojo:96-101`.
Production inference deliberately bypasses them (Klein `_block_tensor_base` BF16).
**Decision:** if dead for inference, migrate to BF16 (mechanical — copy
`models/flux/block.mojo:141-142` `_t`→BF16 + `:243-252`; Klein already has its own
BF16 sibling) OR document them as parity/training-only so no future caller
resurrects an F32 inference path. No runtime bug today. Confidence: medium.

**3. `qwenimage_stack_lora.mojo:281-338` (`_block_f32_host`) upcasts every block matmul weight to F32 — training-only.**
Used only by `training/train_qwenimage_real.mojo` (no inference import). Training F32
is allowed by the contract, but unlike Klein's mixed-GEMM LoRA stack, Qwen training
carries ~2× weight VRAM. **Decision:** acceptable, or align with Klein's mixed GEMM
to cut training memory. Not an inference violation. Confidence: medium.

**4. `_colsum` bias-gradient reduces in storage dtype (BF16), not F32-accumulate — precision choice, contract-compliant.**
`ops/linalg_backward.mojo:229-264` returns `grad_y.dtype()`; BF16 branch
(`:247-252`) sums columns directly in BF16. Dtype-correct, but a many-row bias grad
may lose precision vs F32-accumulate-then-narrow (the MSE head uses `reduce_sum_f32`,
`autograd.mojo:356`, as precedent). **Decision:** leave as-is, or F32-accumulate the
bias grad. Not a violation.

---

## Stale Comments (claim F32 storage where code is BF16)

1. **`models/flux/weights.mojo:31-32`** — "the block structs upload to device F32 in
   __init__…" — **stale**; `models/flux/block.mojo:243-252` uploads **BF16**. Confidence: high. (Doc-only fix.)
   **RESOLVED (verified 2026-06-10):** comment now reads "checkpoint is BF16. Production
   loaders keep checkpoint tensors in their stored dtype with Tensor.from_view."
2. **`models/dit/kandinsky5_dit.mojo:202`** — "modulation weights stored F32…" —
   **inaccurate**; weight loaded via `from_view` (BF16) at `:527`, `linear()` is mixed
   GEMM. Math is F32, storage is BF16. Confidence: medium. (Doc-only fix.)
   **RESOLVED (verified 2026-06-10):** comment now reads "weights stay in their
   checkpoint storage dtype and are consumed by mixed GEMM."
3. `models/sd35/sd35_block.mojo:191` "per-stream weights (host F32 lists)" — **accurate**
   (this is the training block, genuinely host-F32); listed only to distinguish from the
   BF16 inference DiT `sd3_mmdit.mojo`. No change.

---

## Verified Compliant (false positives cleared, with evidence)

### autograd.mojo — full op + backward trace (dtype-following)
- Forward: `_raw_add/_sub/_mul` cast operand to `a.dtype()` and return it (`:88-106`);
  `_raw_matmul` casts the F32 BLAS result back via `cast_tensor(c32^, out_dt, ctx)` (`:180`).
- `_empty_f32` (`:80-85`) has ONE caller (`_raw_gemm` `:125`); its F32 output is consumed
  only by `_raw_matmul` and cast back — never reaches a public boundary.
- Backward arms (`:394-458`) each call a dtype-preserving fn: MATMUL→`mm_backward`
  (`linalg_backward.mojo:310-311`), LINEAR→`linear_backward` (`:428-429`), RMSNORM→
  `rms_norm_backward` (`norm_backward.mojo:226-228`), SILU→`silu_backward`
  (`activation_backward.mojo:250`), SWIGLU→`swiglu_backward`
  (`loss_swiglu_backward.mojo:369-370`), MSE→`mse_backward` (`:116`).
- F32 loss seed is isolated: `mse_loss` uses `reduce_sum_f32`→F32 scalar (`:356`);
  the OP_MSE arm ignores the incoming grad and emits BF16 `mse_backward` (`:458`); the
  F32 seed is keyed only to `loss.id`, never propagates into the BF16 graph.
- `_accum` (`:366-376`) sums into the first-stored grad's dtype; every backward fn
  hard-asserts `grad_out.dtype()==input.dtype()` (mm `:278`, linear `:395`, silu `:199`,
  rms `:158`, swiglu `:314`, mse `:74`) → any mismatch raises, no silent F32 leak.

### ops/ backward kernels — every public grad returns storage dtype
gqa (`gqa_backward.mojo:57/103/137/185`), conv2d (`conv2d_backward.mojo:372-375`),
reduce (`reduce_backward.mojo:258/342/408/444`), activation (`:250`), elementwise
(`elementwise_backward.mojo:194-196`), rope (`rope_struct_backward.mojo:260/362/486/...`),
shape (`shape_backward.mojo:299/482/612/778/975`), pool (`pool_backward.mojo:268/360`),
celoss (`celoss_embed_backward.mojo:181/260/328/418`), norm (rms `:226-228`, layer
`:526-528`, group `:903-905`), loss/swiglu (`:116/192/269/369-370`).
attention (`attention_backward.mojo`): only F32 refs are scratch accumulators
(`:73-74`); dq/dk/dv narrowed to `q.dtype()` via `_scatter_to_tensor` (`:376-408`);
F32-fast-path guards (`:439/911`) FALL THROUGH to general path for BF16/F16.
linalg (`linalg_backward.mojo`): all 20 F32 refs are `_new_f32`/scratch cast back.

### io/ — no dtype coercion; FP8/F16/BF16 handled
`io/dtype.mojo` full table incl. F8_E4M3/E5M2 (1 byte) and BF16/F16 (2 byte) with
correct `from_name` round-trip; `io/safetensors.mojo:162-166` + `io/sharded.mojo:508`
carry the on-disk dtype unchanged. `tensor.mojo from_view` (`:177`) preserves dtype;
`from_view_raw` (`:205`) preserves FP8 bytes for later GPU dequant (`ops/fp8.mojo`);
`from_view_as_bf16` (`:207-259`) only narrows F32/F16→BF16 (output always BF16).

### Production inference DiTs — all BF16-native (complete table)
flux, klein, qwenimage, sd3/sd35, ernie, zimage, l2p, anima, wan22, ltx2, acestep,
cosmos_predict25, hunyuan15, hidream_o1, nucleus, sensenova_u1, wan_vace, kandinsky5,
zimage_l2p, ernie_image, magihuman, lance, lens, ltx2_upsampler — each loads weights
via `from_view`/`from_view_as_bf16` (BF16) with F32 confined to norm scales (allowed)
or compute internals that round-trip. PiD is documented intentional full-F32
(`models/pid/pit_block.mojo:30`). Pipeline Euler/CFG F32 casts round-trip to BF16
before the next block (`pipeline/zimage_pipeline.mojo:232-233`,
`pipeline/asymflux2_klein9b_gen.mojo:575/583`).

---

## Suggested Verification (smallest existing gates — NOT run by this audit)
- Autograd dtype invariant: `serenitymojo/autograd_bf16_storage_smoke.mojo` (highest
  value) + per-arm `autograd_{matmul,linear,rmsnorm,silu,swiglu,mse}_smoke.mojo`.
- ops: `ops/reduce_smoke.mojo`, `ops/conv1d_smoke.mojo`, `ops/moe_smoke.mojo`,
  backward parity under `ops/parity/`.
- If migrating Klein/Qwen resident structs to BF16 (Decision #2):
  `models/klein/parity/double_block_parity.mojo`,
  `models/klein/parity/double_block_lora_parity.mojo`,
  `pipeline/klein9b_dit_smoke.mojo`, `pipeline/klein9b_pipeline_multistep_smoke.mojo`;
  Qwen `pipeline/qwenimage_contract_smoke.mojo`, `pipeline/qwenimage_pipeline_smoke.mojo`.
- Stale-comment regression: `pipeline/flux1_contract_smoke.mojo`.
- **No new BF16 kernels are required** for any flagged path — each already has a
  BF16/mixed-GEMM sibling in production use.

---
*Method: 2 read-only agents (ops/core + models/pipeline), each followed by a deeper
skeptical pass. No edits, no destructive commands. Findings quote `file:line` from
tool output. Where a prior claim was refuted by deeper inspection (the F32 weight
"violations"), the refutation is recorded above rather than the original claim.*
