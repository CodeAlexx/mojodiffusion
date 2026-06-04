# LTX-2.3 22B Audio-Video DiT — Mojo Port Handoff (2026-06-04)

Scope: **ltx2 only.** This is the handoff for porting the LTX-2.3 22B **audio-video joint
DiT block** to pure Mojo, for LoRA training in serenitymojo. Trainers for other models are
out of scope here.

Status in one line: the AV block **forward** is built and compiles, two real bugs were found
and fixed, and a **real** Mojo-vs-oracle numerical parity exists at **video cos 0.9994 / audio
cos 0.9962** — but it was measured against an oracle that was *adapted to fit the block*, and
there are named open divergences. Backward, LoRA, the offload stack, and the trainer are NOT
done. Read "Verification discipline" before trusting any green number here.

---

## 0. The critical discovery (why this work exists)

The pre-existing `models/ltx2/ltx2_block.mojo` models only **`attn1` (video self-attention) +
`ff`**. The real `ltx-2.3-22b-dev` checkpoint is a **joint audio-video DiT** whose every block
also has: video **cross-attention to text** (`attn2`), a full **audio branch**
(`audio_attn1/2`, `audio_ff`), **cross-modal** attention (`audio_to_video_attn`,
`video_to_audio_attn`), and **6 modulation tables**. The old block is a strict subset.

Worse: the old block "passed parity" against the **diffusers** `transformer_ltx.py`, which is
the *simplified video-only* model — a self-consistent oracle that did not reflect the real 22B
model. This is the same failure mode that bit kandinsky5. **Measured from the real checkpoint
header**, block-0 has 86 tensors including `attn2.*`, `audio_*`, `audio_to_video_attn.*`,
`video_to_audio_attn.*`, and `scale_shift_table [9,4096]` (not [6,...]).

Maintainer decision (2026-06-04): build the **full AV model** (audio+video+cross-modal). Do NOT
target the video-only LTX-Video 0.9.x family; delete that 0.9.x code path when convenient.

NOTE: there are two distinct LTX families in `.serenity`:
- **LTX-2.3** (`ltx-2.3-22b-dev` 46GB, `ltx2-diffusers` 19B) = audio-video joint. ← OUR TARGET
- **LTX-Video 0.9.x** (`ltx-video-0.9.7-dev`, `ltx-video/ltxv-13b-0.9.8-dev`) = video-only. ← NOT this.

---

## 1. File inventory

### The new AV block (the deliverable so far)
- `serenitymojo/models/ltx2/ltx2_av_block.mojo` (57 KB, ~1115 lines) — **AV block FORWARD**, compiles clean.
  - `struct AVBlockWeights` (line 808) — 86 weight fields, constructor order matters.
  - `struct AVBlockOut` (line 991).
  - `def ltx2_av_block_forward[B, Sv, N_TXT, Sa]` (line 1025) — see §3 for the full signature.
  - helpers: `_self_attn_path`, `_cross_attn_path`, `_cross_modal_attn`, `_ffn_path`,
    `_apply_head_gate`, `_apply_head_gate_cross`, `_ada_vec`, `_rms_no_affine`.

### Oracle + parity (the ground truth + how to gate)
- `scripts/ltx2_av_oracle.py` — builds the **real** musubi `BasicAVTransformerBlock`, loads
  block-0 weights (block-0 ONLY, never the 46GB), runs the actual `_forward`, saves outputs.
- `scripts/ltx2_av_oracle_intermediates.py` — same but captures activation after each of the 11 sublayers.
- `serenitymojo/models/ltx2/parity/ltx2_av_mojo_parity.mojo` (24 KB) — the **REAL** Mojo parity
  (runs the Mojo forward, loads block-0 weights via mmap, compares to oracle). USE THIS ONE.
- `serenitymojo/models/ltx2/parity/ltx2_av_block_parity.py` — **DO NOT TRUST** — this one runs
  musubi-vs-musubi (self-comparison). Kept only as a record of the trap. Its "cos>1.0,
  max_abs=0" are the tell.

### Oracle data files (output/ltx2_av/)
- `block0_ref.safetensors` (8.7MB, 28 tensors) — **original** ground-truth: inputs + the two
  GOLD outputs `video_out`/`audio_out` + pe/timestep tables. NO weights (weights come from the
  checkpoint). Made by `ltx2_av_oracle.py`. **This is the unmodified reference.**
- `block0_ref_meta.json` — shapes/dtypes/seed/config. See §2.
- `block0_intermediates.safetensors` (14MB, 22 tensors) — per-sublayer references (step01..step11).
- `block0_intermediates_meta.json` — per-step key descriptions.
- `block0_ref_mojo_compat.safetensors`, `block0_ref_halfd.safetensors`, `block0_halfd_pe.safetensors`
  — oracle variants the parity agent **regenerated to fit the Mojo block** (see §5 — these
  bake in at least one math change; do not treat as ground truth).

### Context (the OLD/partial work — superseded by the AV block)
- `models/ltx2/ltx2_block.mojo` — the OLD attn1+ff block (WRONG architecture). Delete once the AV block is wired.
- `offload/ltx2_block_stream.mojo`, `models/ltx2/{ltx2_stack_lora,weights}.mojo`,
  `training/train_ltx2_real.mojo`, `offload/ltx2_plan.mojo` — a trainer that was built against
  the OLD block. It compiles but trains the WRONG architecture. Must be rewired to the AV block (see §6).

---

## 2. Block config (measured from the real ltx-2.3-22b-dev checkpoint)

```
video_dim 4096, video_heads 32, video_head_dim 128, video_context_dim 4096
audio_dim 2048, audio_heads 32, audio_head_dim 64,  audio_context_dim 2048
apply_gated_attention = True   (every attention has to_gate_logits, gate = 2*sigmoid)
cross_attention_adaln = True   → scale_shift_table is [9, dim] (6 base + 3 cross-attn)
norm_eps = 1e-6
rope_type = INTERLEAVED        (NOT split — split is the diffusers/video-only model)
n_ada_params video/audio = 9
```

Oracle test inputs (seed=0): B=1, S_V=64, S_A=32, N_TXT=16. FFN: video 16384, audio 8192.

Checkpoint key prefix: `model.diffusion_model.transformer_blocks.{N}.`
Separate audio branch in the checkpoint is `...audio_embeddings_connector.transformer_1d_blocks.{N}.`
— that is a DIFFERENT module; the per-block audio lives under the SAME `transformer_blocks.{N}.`
prefix with `audio_*` sub-keys. The video LoRA targets `transformer_blocks.{N}` only.

### The 6 modulation tables (per block)
| Mojo field | ckpt key | shape |
|---|---|---|
| `v_table` | `scale_shift_table` | [9,4096] |
| `v_prompt_table` | `prompt_scale_shift_table` | [2,4096] |
| `v_a2v_table` | `scale_shift_table_a2v_ca_video` | [5,4096] |
| `a_table` | `audio_scale_shift_table` | [9,2048] |
| `a_prompt_table` | `audio_prompt_scale_shift_table` | [2,2048] |
| `a_a2v_table` | `scale_shift_table_a2v_ca_audio` | [5,2048] |

### AdaLN slice convention (CONFIRMED vs musubi)
Within a [9,dim] table: rows **[0:3] self-attn**, **[3:6] FFN**, **[6:9] cross-attn**.
Row order inside a slice is **(shift, scale, gate)**. modulate = `(1+scale)*rms_norm(x)+shift`,
computed in **F32** then cast back to bf16 (musubi does this to avoid bf16 overflow ~1e18).

### Weight key → AVBlockWeights field mapping (module → prefix)
`attn1→v_*`, `attn2→v2_*`, `audio_attn1→a_*`, `audio_attn2→a2_*`,
`audio_to_video_attn→a2v_*`, `video_to_audio_attn→v2a_*`, `ff→v_wff*`, `audio_ff→a_wff*`.
Each attention has `to_q/to_k/to_v/to_out.0 (+bias)`, `q_norm/k_norm`, `to_gate_logits (+bias) [32,dim]`.
Full field list + per-field ckpt key comments are in `ltx2_av_block.mojo:808`. `scripts/ltx2_av_oracle.py`
shows the canonical musubi key→module mapping if disambiguation is needed.

---

## 3. The forward API (final, post-fix)

```
def ltx2_av_block_forward[B, Sv, N_TXT, Sa](
    w: AVBlockWeights,
    vx: [B,Sv,4096] bf16, ax: [B,Sa,2048] bf16,
    context_v: [B,N_TXT,4096], context_a: [B,N_TXT,2048],   # text for cross-attn
    vtemb: [B, 9*4096], atemb: [B, 9*2048],                 # main AdaLN timestep embeds
    vprompt_temb: [B, 2*4096], aprompt_temb: [B, 2*2048],   # cross-attn prompt modulation
    vcross_ss_temb: [B, 4*4096], vcross_g_temb: [B, 1*4096], # a2v_ca_video scale-shift / gate
    across_ss_temb: [B, 4*2048], across_g_temb: [B, 1*2048], # a2v_ca_audio scale-shift / gate
    vrope_cos/sin: [Sv,4096], arope_cos/sin: [Sa,2048],      # self-attn RoPE
    vcross_pe_cos/sin: [Sv,2048], across_pe_cos/sin: [Sa,2048], # cross-modal RoPE (BUG1 fix)
    eps, ctx,
) -> AVBlockOut
```

### The 11 sublayers (CONFIRMED order vs musubi `_forward`)
1. video self-attn `attn1` — `v_table[0:3]`, RoPE interleaved, gate
2. video cross-attn `attn2` (text) — `v_table[6:9]` + `v_prompt_table`/`vprompt_temb` modulate K/V; NO rope
3. audio self-attn `audio_attn1` — `a_table[0:3]`, rope, gate
4. audio cross-attn `audio_attn2` (text) — `a_table[6:9]` + `a_prompt_table`/`aprompt_temb`
5. shared `rms_norm(vx)`, `rms_norm(ax)` before cross-modal (no table)
6. `get_av_ca_ada_values(a_a2v_table)` → 4 scale/shift + 1 gate_v2a
7. `get_av_ca_ada_values(v_a2v_table)` → 4 scale/shift + 1 gate_a2v
8. A2V `audio_to_video_attn`: Q=video, KV=audio, + gate_a2v
9. V2A `video_to_audio_attn`: Q=audio, KV=video, + gate_v2a
10. video FFN — `v_table[3:6]`, GELU(tanh), clamp ±60000
11. audio FFN — `a_table[3:6]`, GELU(tanh), clamp ±60000

Cross-modal AdaLN uses **two timestep streams**: scale-shift (4 params) + gate (1).
**gate_a2v comes from the VIDEO table; gate_v2a from the AUDIO table** (CONFIRMED). Cross-modal
RoPE: A2V Q=`vcross_pe`, K=`across_pe`; V2A Q=`across_pe`, K=`vcross_pe`.

---

## 4. What is DONE (measured this session)

- **Forward built + compiles** (`pixi run mojo build -I . serenitymojo/models/ltx2/ltx2_av_block.mojo -o /tmp/avb` → zero type/parse errors; library "no main" exit is fine).
- **2 bugs found by a code-diff skeptic and fixed:**
  - BUG1: cross-modal RoPE was missing — added `rope_interleaved` on A2V/V2A Q/K with the right per-path PE (`_cross_modal_attn` now takes `rope_q_cos/sin`, `rope_k_cos/sin`).
  - BUG2: cross-modal per-head gate was missing — added `a2v_gate_w/b`, `v2a_gate_w/b` weight fields + `_apply_head_gate_cross`.
- **A real Mojo-vs-oracle parity exists** (`ltx2_av_mojo_parity.mojo`): runs the Mojo forward on
  real block-0 weights (mmap, block-0 only). **Measured: video cos 0.9994, audio cos 0.9962**,
  cos in [−1,1], non-zero outputs → confirmed NOT a self-comparison.
- CONFIRMED-correct vs musubi (by code-diff): 11-step order, interleaved self-attn rope, AdaLN
  slices [0:3]/[3:6]/[6:9], two-timestep cross-modal gating (gate_a2v←video, gate_v2a←audio).

---

## 5. What is OPEN / NOT trustworthy (read before claiming "done")

### 5a. The parity was run against an ADAPTED oracle, not ground truth
The parity agent regenerated the oracle (`block0_ref_mojo_compat.safetensors`,
`block0_halfd_pe.safetensors`) with **3 input changes** to make it match the Mojo block:
1. timestep `[B,1,X]→[B,X]` squeeze — **benign** (T=1 broadcast dim).
2. RoPE full-D→half-D + `repeat_interleave(2)` — **probably benign** (standard interleaved
   equivalence) but CONFIRM the half-D angles equal musubi's interleaved formulation.
3. **prompt timestep `[B, N_TXT, 2D]` → mean over N_TXT → `[B, 2D]`** — **MATH CHANGE.** musubi
   modulates cross-attn K/V **per text token**; the Mojo block only supports **per-batch**. The
   agent averaged to force a match. This is **exact only if** the prompt timestep is constant
   across text tokens. **OPEN: confirm whether musubi varies prompt-ts per token.** If it does,
   the Mojo block is missing per-token prompt modulation and must add it; the parity currently
   hides this.

### 5b. The numbers aren't tight
audio cos **0.9962 < 0.999**, and **max_abs is large (video 1.72, audio 2.70)** even against the
adapted oracle. Direction matches but magnitudes diverge. **OPEN: use
`block0_intermediates.safetensors` (per-step) to find which sublayer the audio path diverges in.**

### 5c. Forward only — no backward, no LoRA, no trainer
- **Backward + LoRA** for the AV block: NOT built. (The bf16 carrier infra + the local-F32-cast
  pattern at rope_backward/cat_backward/gate_residual_backward are established repo-wide; follow
  the wan22/ltx2-old block backward style.)
- **Offload stack + trainer**: the existing `ltx2_stack_lora.mojo` / `train_ltx2_real.mojo` were
  built against the OLD attn1+ff block. They must be **rewired to `ltx2_av_block.mojo`** (new
  weight loader for the 86-tensor AV block, new stack streaming the AV blocks, LoRA targets per
  musubi `networks/lora_ltx2.py`). See §6.
- **Delete the 0.9.x video-only code** once the AV path is wired (maintainer request).

### 5d. No training run possible yet
There is **no prepared ltx2 latent+text+audio cache** on disk. Even with backward+trainer done,
a real "training works" verdict (loss drop + sample shift) is blocked on cache data.

### 5e. ltx2 vocoder (separate, audio output path)
`models/vocoder/ltx2_vocoder.mojo` is coded (BigVGAN-v2 + BWE) with a parity smoke, but its
oracle (`output/ltx2_vocoder/vocoder_ref.safetensors`) and ref script
(`scripts/ltx2_vocoder_ref.py`) are MISSING from disk → currently unverifiable. Regenerate from
`inference-flame/src/vae/ltx2_vocoder.rs` to gate it.

---

## 6. Recommended next steps (priority order)

1. **Settle 5a.3** — read musubi `transformer.py` `_apply_text_cross_attention` /
   `get_ada_values` to confirm whether prompt-ts is per-token or constant. If per-token, add
   per-token prompt modulation to `_cross_attn_path`. Then **re-gate against the UNMODIFIED
   `block0_ref.safetensors`** (regenerate Mojo-format inputs WITHOUT the mean-collapse).
2. **Chase 5b** — per-sublayer parity vs `block0_intermediates.safetensors` (step01..step11) to
   localize the audio residual; fix until audio cos ≥ 0.999.
3. **AV backward + LoRA** in `ltx2_av_block.mojo` (mirror the wan22 block bwd; native bf16,
   grads F32, local F32 casts only at rope_backward/cat_backward/gate_residual_backward).
4. **Rewire the trainer** to the AV block: new `weights.mojo` (86-tensor AV loader, video-only
   `transformer_blocks.{N}` prefix, strip `model.diffusion_model.`), new `*_stack_lora.mojo`
   (stream AV blocks), `train_ltx2_real.mojo` flow-match loop. LoRA targets from musubi
   `src/musubi_tuner/networks/lora_ltx2.py`. Put the block plan in `offload/ltx2_plan.mojo`
   (NOT shared `plan.mojo`).
5. **Delete the old attn1+ff `ltx2_block.mojo`** and 0.9.x references once the AV path is gated.
6. (when a cache exists) real training run for the loss-drop + sample-shift verdict.

---

## 7. Verification discipline (this bit a port twice — do not skip)

- **A green cos against a self-made or adapted oracle proves nothing.** The first AV "parity"
  ran musubi-vs-musubi (`cos>1.0, max_abs=0` — the tell). The "real" parity then adapted the
  oracle to fit the block (5a). Always gate the **Mojo forward** against the **unmodified**
  ground-truth oracle.
- **cos > 1.0 or max_abs == 0.0 ⇒ self-comparison.** Reject it.
- **Oracle must come from the real musubi `_forward` on real 22B weights**, NOT diffusers
  `transformer_ltx.py` (diffusers = the simplified video-only model; that's the kandinsky5 trap).
- **Use NON-DEGENERATE inputs** (random/sinusoidal, fixed seed), never modular fills.

---

## 8. Memory-safety (a host OOM killed agents this session)

- **NEVER load the full 46GB checkpoint.** Load block-0 ONLY (~86 tensors, ~774MB) via
  `safe_open` (Python) or Mojo `SafeTensors` mmap-by-key. A prior agent stalled loading the whole file.
- **Do NOT run full-model trainers concurrently with other agents.** The OOM was an SD3.5 training
  run (full model pinned in host) + concurrent agent builds exhausting the 62GB host → kernel
  OOM-killed the parent Claude process. Run heavy model jobs SOLO with an RSS monitor.
- The AV parity is block-0 only → light (GPU < 1GB, brief). Safe to run.

## 9. Commands

```bash
cd /home/alex/mojodiffusion
# compile the AV block
pixi run mojo build -I . serenitymojo/models/ltx2/ltx2_av_block.mojo -o /tmp/avb
# run the REAL Mojo parity (block-0 only, light)
pixi run mojo run -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_mojo_parity.mojo
# regenerate the ground-truth oracle (musubi venv has torch + musubi)
/home/alex/musubi-tuner/venv/bin/python scripts/ltx2_av_oracle.py
/home/alex/musubi-tuner/venv/bin/python scripts/ltx2_av_oracle_intermediates.py
```

Reference source (the ground truth):
`/home/alex/musubi-tuner/src/musubi_tuner/ltx_2/model/transformer/transformer.py`
— `BasicAVTransformerBlock.__init__` (~86), `_forward` (~466), `get_ada_values` (~195),
`_apply_text_cross_attention` (~267), `get_av_ca_ada_values`. Attention + rope:
`.../transformer/attention.py`, `.../transformer/rope.py` (`LTXRopeType.INTERLEAVED`).
