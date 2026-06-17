# Skeptic findings — Boogu-Image DiT input embedders (Chunk C1)

Date: 2026-06-17
Reviewer stance: adversarial. Assume the port lies until each claim is checked
against the reference and the live config/checkpoint.

Files reviewed (review only — NOT edited):
- `serenitymojo/models/dit/boogu_dit.mojo` (`BooguEmbedders.load / .x_embed / .time_caption_embed`)
- `serenitymojo/models/dit/parity/boogu_c1_embed_probe.mojo`

Reference read line-by-line:
- `/home/alex/Boogu-Image/boogu/models/transformers/block_lumina2.py:177-219`
  (`Lumina2CombinedTimestepCaptionEmbedding`)
- `/home/alex/Boogu-Image/boogu/models/embeddings.py:24-77` (`TimestepEmbedding`)
- `/home/alex/Boogu-Image/boogu/models/transformers/transformer_boogu.py:786-855,
  974-985, 1010-1011, 1198-1251, 1253-1306`
- `/home/alex/.local/lib/python3.12/site-packages/diffusers/models/embeddings.py:26-77`
  (`get_timestep_embedding` — the body of `Timesteps`)
- `/home/alex/Boogu-Image/boogu/ops/triton/layer_norm.py:54-111, 1163-1203`
  (the actual `RMSNorm` Boogu instantiates)
- serenitymojo `ops/embeddings.mojo`, `ops/norm.mojo`, `ops/linear.mojo`,
  `ops/activations.mojo`, `ops/random.mojo`, `tensor.mojo`, `io/sharded.mojo`

Live config/checkpoint checked:
- `/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer/config.json`
- `.../transformer/diffusion_pytorch_model.safetensors.index.json` (weight_map)

Probe RE-RUN (the command in the prompt, verbatim):
```
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
  pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c1_embed_probe.mojo
```
Actual output:
```
x_embed out shape: 1 256 3360
x_embed std: 8.091518
time_embed shape: 1 1024
time_embed std: 239.70639
caption_embed shape: 1 16 3360
caption_embed std: 62.483337
boogu_c1_embed_probe OK
EXIT=0
```
Exit 0, all three forwards executed to completion (not partial output). Note the
probe uses SYNTHETIC randn weights, so the std values prove execution only, NOT
parity — parity is the orchestrator's torch-oracle job, correctly out of scope here.

---

## The #1 suspected silent bug — the timestep sinusoid pre-scale equivalence — is CORRECT

The builder's central claim (boogu_dit.mojo:34-40, 145) is that pre-scaling the
timestep by 1000 BEFORE the COS-first sinusoid is mathematically identical to
diffusers applying `scale=1000` to the whole product. Verified against the
diffusers source body of `Timesteps` (`get_timestep_embedding`):

diffusers (`embeddings.py:55-72`) with `downscale_freq_shift=0, scale=1000,
flip_sin_to_cos=True, max_period=10000`:
- `half_dim = 128`
- `exponent_i = -ln(10000) * i / (128 - 0)` → `freq_i = exp(exponent_i)`  (denom = half)
- `emb = t * freq`; `emb = scale * emb` → `1000 * t * freq_i`
- `cat([sin(emb), cos(emb)])` then `flip_sin_to_cos` swaps halves → final
  `[cos(1000·t·freq), sin(1000·t·freq)]`  → **COS-first**.

serenitymojo `timestep_embedding` (`ops/embeddings.mojo:71-75`):
`freq_i = exp(-ln(max_period)·i/half)`, `angle = t·freq_i`, `emb[i]=cos(angle)`,
`emb[half+i]=sin(angle)` → COS-first, denom = half.
With `t' = 1000·t` (mul_scalar, boogu_dit.mojo:146) → `angle = 1000·t·freq_i`.

Every term matches:
- (a) COS-first ordering: MATCHES (serenitymojo cos-first vs diffusers post-flip cos-first). The repo's SIN-first kernel (ERNIE) is correctly NOT used.
- (b) freq denominator `half=128` (downscale_freq_shift=0): MATCHES `dim/2`.
- (c) pre-scale equivalence `scale·t·freq == (scale·t)·freq`: MATCHES, and the flip does NOT break it because the flip only swaps the cos/sin halves — it does not reorder the per-frequency index `i`, so applying the scale to `t` vs to the product is identical column-by-column.
- (d) `max_period = 10000` (BOOGU_TIME_MAX_PERIOD, boogu_dit.mojo:69, passed as the last arg to `t_embedder`): MATCHES diffusers default 10000. Correctly NOT confused with timestep_scale.
- dim=256 is even → the odd-dim zero-pad path (`embeddings.py:75-76`) is never hit. Confirmed irrelevant.

Verdict on the headline risk: **clean.**

---

## Config-truth verification (the constructor defaults are a trap the builder avoided)

`transformer_boogu.py:798-805` constructor DEFAULTS are misleading:
`instruction_feat_dim=1024`, `reduce_type="mean"`, `timestep_scale=1.0`. A careless
port that trusted the constructor signature would use 1024 / scale=1.0 and be wrong.

The LIVE config.json overrides them, and the builder's config-truth matches the
live config exactly:
- `hidden_size: 3360`, `in_channels: 16`, `patch_size: 2` → x_embedder 64→3360. ✓
- `instruction_feat_dim: 4096`, `reduce_type: "mean"`, `num_instruction_feature_layers: 1`.
  Per `cal_preprocessed_instruction_feat_dim` (`:1210-1211`) "mean" → returns
  `instruction_feat_dim` = 4096 (mean over a stack of 1 = identity). So
  `preprocessed_instruction_feat_dim = 4096`, the RMSNorm/Linear in-dim. ✓
- `norm_eps: 1e-05`. ✓ (BOOGU_NORM_EPS = 1e-5, not 1e-6 — checked, correct)
- `timestep_scale: 1000.0`. ✓

Mean-reduce handling: with `num_instruction_feat_layers=1` and a Tensor input,
`preprocess_instruction_hidden_states` (`:1227-1228`) returns the tensor as-is; the
mean is identity. The builder correctly does NOT add a spurious reduction, and the
list-input mean/stack path is genuinely out of scope (single-tensor input). ✓

Verdict: **clean** — config assumptions are correct, not lucky.

---

## Findings

### F1 — caption_embedder RMSNorm `+1.0` (zero_centered_weight) trap — AVOIDED [STYLE / note]
File: boogu_dit.mojo:159-161 vs `boogu/ops/triton/layer_norm.py:80-83, 140-143, 1163-1203`.
Boogu's `RMSNorm` (the class actually imported by block_lumina2 when triton is
present) supports a `zero_centered_weight` mode that computes `weight = weight + 1.0`.
If that mode were on, serenitymojo `rms_norm` (plain `x·inv·gamma`, no +1) would be
WRONG. Checked: `block_lumina2.py:200` constructs `RMSNorm(instruction_feat_dim,
eps=norm_eps)` → `zero_centered_weight=False` (default), `reset_parameters` →
`ones_(weight)`, `bias=None`. So the effective math is plain RMSNorm and matches
serenitymojo exactly. No action needed; flagged only so a future reader does not
"fix" it the wrong way. NOT a blocker.

### F2 — checkpoint weight dtype/shape not yet verifiable from disk [FRAGILE]
File: boogu_dit.mojo:84-94 (claimed shapes), 11-13 (claimed BF16).
The 9 weight keys all exist in the index `weight_map` (verified):
`x_embedder.{weight,bias}`, `time_caption_embed.timestep_embedder.linear_1/2.{weight,bias}`,
`time_caption_embed.caption_embedder.0.weight` (RMSNorm gamma — only `.weight`, no
bias, confirming RMSNorm), `.caption_embedder.1.{weight,bias}` (Linear w/ bias).
BUT only shard 3-of-3 is on disk; the embedder weights live in shard 1 (still
downloading), so I could NOT read their actual on-disk dtype or shape. The BF16
claim is the standard diffusers convention and consistent with the model's
`_skip_layerwise_casting_patterns` / `.to(dtype)` flow, but it is currently an
ASSUMPTION, not a measurement.
Why only FRAGILE: `_load_w` uses `Tensor.from_view` which PRESERVES whatever dtype
the checkpoint stores (it does not assume BF16), and `linear`/`rms_norm` dispatch on
the runtime dtype. So if a tensor were actually F32, the load path still works; the
only real risk is a shape surprise (e.g. an unexpected fused/packed layout), which
the structural check against config makes unlikely.
Minimal action: when shard 1 lands, assert in the parity harness that each loaded
tensor's `dtype()`/`shape()` equals [3360,64]/[3360]/[1024,256]/[1024]/[1024,1024]/
[1024]/[4096]/[3360,4096]/[3360]. No code change to boogu_dit.mojo required.

### F3 — bias passed via `.clone(ctx)` is a redundant device copy [STYLE]
File: boogu_dit.mojo:124, 151, 153, 165.
Every Linear/MLP bias is passed as `Optional[Tensor](self.x_embedder_bias.clone(ctx))`
etc. `linear` only READS the bias (bias-add kernel), so the clone is an unnecessary
d2d copy + synchronize per forward. It is presumably done because `self.<field>` is a
borrowed member and `Optional[Tensor](...)` would otherwise try to move/copy a
non-Copyable Tensor out of `self`. Functionally correct and dtype-preserving
(`clone` keeps `_dtype`), just wasteful. Not a parity issue. If a borrow-friendly
`Optional` overload exists, prefer it; otherwise leave as-is for inference.

### F4 — probe synthetic RMSNorm gamma is randn, not ones [STYLE / probe-only]
File: boogu_c1_embed_probe.mojo:53-55.
The real `caption_embedder.0.weight` (RMSNorm gamma) initializes to ones; the probe
feeds a plain randn gamma. The probe's own comment acknowledges this. It only affects
the printed std (execution proof), never parity (orchestrator uses real weights).
NOT a bug.

---

## Mojo-correctness sweep (all pass)
- `Tensor` is `struct Tensor(Movable)` — Movable-not-Copyable. Weights owned directly,
  moved once with `^` into the constructor; no `List[Tensor]`, no use-after-move.
  Each forward returns fresh tensors with `^` (boogu_dit.mojo:169). ✓
- `comptime` used for all constants (boogu_dit.mojo:60-69), no `alias`. ✓
- Every fallible function is `def ... raises` (`_load_w`, `load`, `x_embed`,
  `time_caption_embed`). ✓
- No variable named `ref`. ✓
- File I/O only via `ShardedSafeTensors.open` + `tensor_view` + `Tensor.from_view`
  (io/sharded path) — no builtin `open`. `ShardedSafeTensors.open(dir)` and
  `.tensor_view(name)` signatures confirmed in io/sharded.mojo:384,497. ✓
- No reimplementation of an existing ops/ foundation op: reuses
  `ops/linear.linear`, `ops/norm.rms_norm`, `ops/embeddings.{timestep_embedding via
  t_embedder}`, `ops/tensor_algebra.mul_scalar`. Signatures match the call sites
  (`mul_scalar(a, s, ctx)`, `t_embedder(t, dim, mlp0_w, mlp0_b, mlp2_w, mlp2_b, ctx,
  max_period)`, `rms_norm(x, weight, eps, ctx)`, `linear(x, w, Optional[bias], ctx)`). ✓

## Numerical / dtype sweep (all pass)
- SiLU not GELU: `t_embedder` calls `ops/activations.silu` = `x*sigmoid(x)` (confirmed
  activations.mojo:3,37-38). Reference `TimestepEmbedding` uses `act_fn="silu"` →
  `torch.nn.SiLU` (confirmed via diffusers get_activation). MLP order Linear→SiLU→Linear
  matches (`embeddings.py:68-73`). No double sinusoid — `t_embedder` runs the sinusoid
  exactly once then the 2 linears. ✓
- BF16 store / F32 accumulate at op boundaries: `rms_norm` accumulates x² in F32 and
  casts the store down (norm.mojo bf16 kernel), `linear` GEMM accumulates in an F32 C
  buffer then bias-adds in F32 and casts to storage dtype. The time sinusoid is stored
  in the MLP weight dtype (BF16) inside `t_embedder` — exactly mirroring diffusers'
  `time_proj(...).to(dtype=bf16)` cast point. ✓
- Rank conventions: x_embed [B,L,64]→[B,L,3360] (linear flattens leading dims, preserves
  them on output). caption [B,L,4096]→[B,L,3360]. time [B]→[B,1024]. Consistent with the
  reference; no [L,4096]-vs-[1,L,4096] confusion. ✓
- Sinusoid precision: trig is computed in F32 inside the GPU kernel
  (`_timestep_embed_kernel`, all math F32, cast only at store). No F32-trig trap. ✓

## Compile honesty
Probe re-run by the reviewer (not trusting a prior log): exit 0, all three forwards
printed shapes + std and the final `boogu_c1_embed_probe OK`. Not partial output. ✓

---

## VERDICT: BLOCKERS: 0 (clean)

The C1 embedders faithfully reproduce the Boogu-Image input-side: x_embedder Linear,
the diffusers Timesteps→TimestepEmbedding time path (COS-first, pre-scale by 1000
proven equivalent, max_period 10000), and the RMSNorm(eps 1e-5, no +1, no bias)→Linear
caption path. Config-truth matches the live transformer/config.json. No parity
blockers found. Only open item is F2: confirm the on-disk weight dtype/shape once
shard 1-of-3 finishes downloading (the load path is dtype-agnostic, so this is a
verification step, not a code fix). F1/F3/F4 are notes/STYLE only.
