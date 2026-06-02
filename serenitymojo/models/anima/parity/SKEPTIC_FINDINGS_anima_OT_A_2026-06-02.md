# SKEPTIC FINDINGS — Anima OT LoRA-step, Chunk A (forward-faithfulness attack)

Date: 2026-06-02
Auditor stance: assume the green recipe gate LIES about forward-faithfulness until a
tool result confirms/refutes (Tenet 4). The recipe gate (predicted_flow cos 0.9999…,
loss rel-err 4e-7) is NOT re-checked here — it is real but BLIND to the one thing it
cannot see: whether the Mojo training FORWARD is the same forward OneTrainer trains.

Scope: the ONE thing the recipe gate cannot see — the train-time positional function
and any real-Cosmos extras the Mojo forward drops. Builds on P1b Attacks 1/2 (rope
non-degeneracy + half-split convention, both VERIFIED-CLEAN); does NOT repeat them.

---

## ATTACK 1 — Train-time RoPE: single-axis vs the real 3-axis (T,H,W) → **DIVERGENCE (HIGH / BLOCKING)**

### Claim under test
`train_anima_ot.mojo:_rope_tables` (lines 333-346) builds a SINGLE linear position
axis: `ang = Float32(s) / (10000 ** (2i/Dh))` for `s` = the flat token index
0..S_IMG-1, full theta=10000 across all 64 bins. The REAL Anima forward (the one OT
trains and the one the Mojo INFERENCE uses) is a 3-AXIS (T,H,W) split with per-axis
NTK theta.

### The real forward is 3-axis — proven from three independent sources
1. **diffusers `CosmosRotaryPosEmbed.forward`** (transformer_cosmos.py:480-518):
   `dim_h = dim_w = hidden//6*2`, `dim_t = rest`; per-axis `emb_h = outer(seq[:H],
   h_freqs)`, `emb_w = outer(seq[:W], w_freqs)`, `emb_t = outer(seq[:T], t_freqs)`,
   concatenated `[t,h,w]*2`. For an image (`fps=None`) each token's rotation depends
   on its (ih, iw) grid coordinate, NOT a flat index.
2. **Real Anima diffusers config** (`circlestone-labs/Anima-Base-v1.0-Diffusers`,
   `transformer/config.json`, fetched 2026-06-02):
   ```
   "_class_name": "CosmosTransformer3DModel",
   "num_attention_heads": 16, "attention_head_dim": 128,
   "rope_scale": [1.0, 4.0, 4.0],          # (t, h, w) NTK scales
   "extra_pos_embed_type": null, "img_context_dim_in": null,
   "use_crossattn_projection": false, "concat_padding_mask": true
   ```
   → `rope_scale (t=1.0, h=4.0, w=4.0)`. This is the table the OT trainer fits against.
3. **The Mojo INFERENCE forward** (anima_dit.mojo:299-385 `build_anima_3d_rope`,
   called from `forward_with_context:1130` via `build_anima_3d_rope(T, nH, nW, ...)`)
   uses exactly `h_ntk=4.0, w_ntk=4.0, t_ntk=1.0` (lines 329-331) — **MATCHES the real
   config** rope_scale (1,4,4). So the inference table is OT-faithful; only the trainer
   table is not.

### Measured divergence (the load-bearing number)
Non-degenerate 8×8 patch grid (nh=nw=8, T=1, S_IMG=64 — EXACTLY the trainer's
LATENT_HW=16 → S_IMG=64 case), real H=16, Dh=128, half-split RoPE on q/k then SDPA,
single-axis table vs real 3-axis (1,4,4) table, identical random q/k/v:

```
GRID nh=8 nw=8 T=1 S=64 Dh=128
cos(attn_out_singleaxis, attn_out_3axis) = 0.7076998844
rel L2 diff attn out = 0.764520
cos(cos_table_single, cos_table_3axis) = 0.631053
cos(sin_table_single, sin_table_3axis) = 0.243988
row 9 (ih=1,iw=1) single cos[:4]: [-0.9111, 0.0603, 0.8934, 0.9053]
row 9 (ih=1,iw=1) 3axis  cos[:4]: [1.0, 1.0, 1.0, 1.0]
```
Robustness — 5 random q/k seeds: `[0.7079, 0.7112, 0.6977, 0.7042, 0.7161] mean 0.7074`.
Teeth — single-vs-single table gives `cos=1.0` (experiment sound, not a bug in the harness).

**cos ≈ 0.71, far below 0.999; ~76% relative L2 error** on the attention output of the
very first sub-block. Token 9 makes the mechanism concrete: the single-axis table feeds
position index 9 into the rotation; the 3-axis table sees frame 0 → the 22 temporal bins
are all cos=1.0, and only the (ih=1, iw=1) spatial bins rotate. Completely different
rotation per token.

### Why the recipe gate is BLIND to this
`anima_ot_step_oracle.py:350-356` builds the SAME single-axis table the Mojo trainer
uses (`pos = arange(S_IMG)`, single theta ramp). Both sides of the gate share the
identical wrong positional function, so the gate cos stays ~1.0 — a **shared-error
tautology on the positional axis**. The recipe gate validates the recipe (scale_latents,
sigma, target, MSE) but cannot surface a rope-axis error because it never instantiates
the real 3-axis table. The P1b block gate is likewise blind: P1b Attack 1 explicitly
restricted its CLEAN verdict to fwd/bwd **math correctness** of the half-split arithmetic
("the specific frequency values are irrelevant … for testing fwd/bwd math correctness")
and flagged the 3-axis table as a LOW, optional, gate-coverage item — NOT a statement
that single-axis == 3-axis for the trained positional function. It does not. This finding
is the orthogonal axis P1b deferred.

### Severity / blocking
**HIGH, BLOCKING the L2P verdict.** A LoRA fit with single-axis rope is optimized against
a forward that differs from BOTH OneTrainer's training forward AND the Mojo inference
forward at cos≈0.71. The L2P "sample SHIFTS with the LoRA" check would then run the
inference 3-axis forward against adapters fit to single-axis positions — any sample shift
would be evidence of a *mismatched* adapter, not a correctly-trained one. The loss-drop
arm is also untrustworthy: loss can fall against the wrong forward (the FIXED_STEP_SMOKE
already proves adapters fit any fixed target) without the LoRA being the OT LoRA.

---

## ATTACK 2 — Recipe-oracle independence → **VERIFIED-CLEAN (HIGH)**

The transformer math is intentionally SHARED between the Mojo step and the oracle (the
stack_oracle lineage, already cos≥0.99999999) — that is acceptable and stated. The
question is whether the RECIPE pieces are derived from OT Python source, not back-copied
from the Mojo. Each traces to OT source (`/home/alex/OneTrainer-anima-ref`):

| Recipe piece | OT source line | matches Mojo+oracle |
|---|---|---|
| `scale_latents = (lat - mean) * (1.0/std)`, per-ch 16 | `AnimaModel.py:233-236` (`* latents_std` where `latents_std = 1.0/std`) | ✅ train_anima_ot.mojo:443; oracle:326 |
| `timestep / 1000` into embedder | `BaseAnimaSetup.py:137` | ✅ train:521; oracle:337 |
| `flow = noise - scaled` (target) | `BaseAnimaSetup.py:143` | ✅ train:515; oracle:334 |
| `all_timesteps = arange(1, N+1); sigma = all/N; sigma[ts]` → `(ts+1)/N` | `ModelSetupFlowMatchingMixin.py:23-29` | ✅ train:500; oracle:329 |
| `noisy = noise*sigma + scaled*(1-sigma)` | `ModelSetupFlowMatchingMixin.py:36-37` | ✅ train:514; oracle:333 |
| LOGIT_NORMAL `bias=noising_bias, scale=noising_weight+1, sigmoid(normal)` | `ModelSetupNoiseMixin.py:156-161` | ✅ train:197-199 |
| unmasked MSE `mean((pred-target)^2)` | `BaseAnimaSetup.calculate_loss → _flow_matching_losses` | ✅ train:541; oracle:375 |

All constants are independently grounded in OT source. The recipe gate is NOT tautological
on the recipe. (Sub-confirm: `_add_noise_discrete` indexes `__sigma[timestep]` with the
raw discrete timestep, and `__sigma = arange(1,N+1)/N`, so `sigma[ts] = (ts+1)/N` — the
`+1` in the Mojo is correct, not an off-by-one.)

---

## ATTACK 3 — Real-Cosmos extras the Mojo training forward may omit → **VERIFIED-CLEAN (HIGH)**

For each Cosmos default extra, checked (a) present in the real Anima checkpoint /config?
(b) consumed by the Mojo forward? A real weight ignored by Mojo = divergence; a Cosmos
default that the Anima config disables = correctly omitted.

Checkpoint key grep (`anima-base-v1.0.safetensors`, 685 keys) + real config:

| Cosmos extra | In Anima checkpoint? | In real config? | Mojo forward | Verdict |
|---|---|---|---|---|
| `learnable_pos_embed` (`CosmosLearnablePositionalEmbed`, adds to hidden) | **0 keys** (`pos_emb`/`learnable` → 0) | `extra_pos_embed_type: null` | not applied | CLEAN — correctly omitted |
| `img_context` / `q_img`/`k_img`/`v_img` (CosmosAttnProcessor2_5 dual path) | **0 keys** | `img_context_dim_in: null` | not applied | CLEAN — correctly omitted |
| `crossattn_proj` (text 1024→1024 + GELU) | 0 keys | `use_crossattn_projection: false` | not applied | CLEAN — correctly omitted |
| `concat_padding_mask` (cat mask channel → in_ch+1) | baked: `x_embedder.proj.1.weight` is **(2048, 68)** = (16+1)·2·2 | `concat_padding_mask: true` | `_patchify_in` appends a ZERO mask channel → 68 (train:230,246; oracle:280,289) | CLEAN — present & consumed |
| AdaLN-no-affine LayerNorm pre | (math) | `CosmosAdaLayerNormZero` LN eps 1e-6 | `_apply_adaln_modulate` / oracle `layer_norm_noaffine` | CLEAN (P1b Attack 5) |

Notes:
- OT's `BaseAnimaSetup.py:128-141` builds a **zeros** `padding_mask` `[1,1,H*8,W*8]` and
  passes it in; with `concat_padding_mask=true` the real model resizes+cats it as the
  extra channel. Since it is all zeros, the Mojo's zero mask channel is value-identical.
  (Resolves the P1b Attack-4 LOW "text-padding mask" note for the IMAGE-side mask: the
  image padding mask exists and is zero; the Mojo replicates it. The TEXT key-padding
  mask in cross-attn remains a separate Chunk-C question — see fix list.)
- `num_attention_heads: 16` (not the Cosmos default 32) confirmed by q_proj `(2048,2048)`
  → H=16, Dh=128; Mojo dims match.

No real weight is dropped by the Mojo forward. CLEAN.

---

## VERDICT

| Attack | Result | Severity |
|--------|--------|----------|
| 1 Train-time rope single-axis vs real 3-axis | **DIVERGENCE** | HIGH — BLOCKING |
| 2 Recipe-oracle independence | VERIFIED-CLEAN | HIGH |
| 3 Real-Cosmos extras omitted (pos-embed / img-context / crossattn-proj / padding-mask) | VERIFIED-CLEAN | HIGH |

The recipe gate does NOT lie about the recipe (Attack 2) and the Mojo forward drops no
real weight (Attack 3). But the gate is **blind by construction** to the positional
function, and on that axis the trainer forward is **not OT-faithful**: it fits the LoRA
against single-axis rope while OneTrainer (and the Mojo inference) use 3-axis (1,4,4) NTK
rope — measured cos 0.7077 (rel L2 0.76) at the trainer's own 8×8 grid, robust across
seeds. The green recipe gate is necessary but not sufficient; this divergence sits exactly
in its blind spot.

## Prioritized fix list

1. **[BLOCKING — before any trustworthy L2P verdict] Replace single-axis `_rope_tables`
   with the real 3-axis table.** `train_anima_ot.mojo:_rope_tables` (333-346) must build
   the (T,H,W) split with `rope_scale (t=1.0, h=4.0, w=4.0)` exactly as
   `anima_dit.mojo:build_anima_3d_rope` (299-385) already does for inference — call/port
   that same builder with `(T=1, nH=LATENT_HW/2, nW=LATENT_HW/2)`. The inference builder
   is already config-verified faithful, so reuse it rather than re-deriving. Until this
   lands, the L2P loss-drop AND sample-shift are both untrustworthy (loss drops against
   the wrong forward; sample shift reflects a mismatched adapter). The L2P verdict CANNOT
   be trusted with the single-axis table.
   - Also update `anima_ot_step_oracle.py:350-356` to the SAME 3-axis table, AND assert
     non-degeneracy, so the recipe gate stops sharing the single-axis error (otherwise the
     gate will still pass green on the wrong axis after the fix — keep them locked together
     but on the CORRECT table, ideally cross-checked against a real `CosmosRotaryPosEmbed`
     instance for one position).

2. **[Fold-in to Chunk B/C, NON-blocking for the rope fix] Text key-padding mask in
   cross-attn.** Carried from P1b Attack-4 LOW. The IMAGE padding mask is resolved (zero,
   replicated). The cross-attn over the 512-token context: OT pads captions to 512 and the
   real `CosmosAttnProcessor2_0` accepts `attention_mask`. The Mojo cross-attn is maskless
   and the trainer zero-pads context positions (train:464-475). Zero-padded K/V still
   contribute to softmax unless masked. Verify in Chunk C whether OT passes a text
   attention_mask that zeroes pad-token attention; if so, the Mojo 512-path must mask the
   pad positions. Not a rope-fix blocker, but a real-data (Chunk C/D) correctness item.

3. **[NON-blocking, hygiene] After the rope fix, re-run the recipe gate and confirm it is
   still green on the CORRECT 3-axis table** (it should be — the recipe is orthogonal), and
   add a one-shot parity of the 3-axis table builder vs a live `CosmosRotaryPosEmbed(
   hidden=128, rope_scale=(1,4,4))` for a 8×8 image grid to make the gate able to catch a
   future table regression (closes the blind spot permanently).

### Does single-axis MUST become 3-axis before L2P? — YES.
The L2P verdict (loss-drop + sample-shift) is the milestone acceptance. With single-axis
rope the adapters are fit to a forward that differs from inference at cos≈0.71, so a sample
shift would prove the LoRA is *mismatched*, not trained-correctly; the verdict would be a
false PASS. Fix #1 is a hard prerequisite for trusting B (VAE encoder), C (512 text path),
and D (end-to-end) — specifically D's L2P. B and C do not touch rope and can proceed in
parallel, but D's acceptance MUST wait on the 3-axis rope.
