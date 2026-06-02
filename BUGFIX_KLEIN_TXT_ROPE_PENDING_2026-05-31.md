# BUGFIX (PENDING) — Klein text-token RoPE ids are all-zero, must be [0,0,0,k]

status: **DIAGNOSED, NOT APPLIED.** Hold until codex's ring-allocator / LoRA-device
work is merged and the tree is handed back. This doc is the apply-plan.
date: 2026-05-31 · author: Claude (read-only audit, no edits made)

───────────────────────────────────────────────────────────────────────────────
## Symptom
serenitymojo Klein-4B training loss is SUSTAINED ~2.7 over many steps. OT-parity
baseline (EriDiffusion-v2/baselines/klein9b.toml) is **0.75 steady / 1.12 step-1**.
The Mojo Klein *inference* is also recorded as outputting noise
(memory project_klein9b_mojo_noise_blocked) — same root cause (shared rope builder).

───────────────────────────────────────────────────────────────────────────────
## Root cause (source-confirmed; causal link is a STRONG HYPOTHESIS — not yet fix-and-measured)
Mojo gives every TEXT token all-zero 4-axis RoPE position ids `[0,0,0,0]`.
The correct Klein/Flux2 convention is **`[0,0,0,k]`** (axis 3 = sequential text-token
index k). Axis 3 carries rotary frequencies, so all-zero collapses every text token
to an identical positional phase → joint text↔image attention loses text ordering →
wrong velocity prediction → sustained high loss.

### Evidence
WRONG — Mojo, two sites (both set p3=0 for text tokens):
- `serenitymojo/models/dit/klein_dit.mojo` → `build_klein_rope_tables` (~L522)
- `serenitymojo/training/train_klein_real.mojo` → `_build_klein_rope_host` (~L170)

  Both loops, for `tok < N_TXT`: `p0=p1=p2=p3=0` (nothing assigned).
  For img tokens (`tok >= N_TXT`): `p1 = idx//img_w`, `p2 = idx%img_w` → [0,row,col,0]
  (CORRECT — matches the Rust reference; do NOT change the img branch).

CORRECT — EDv2 Rust trainer (the path that achieves the 0.75 baseline):
`EriDiffusion-v2/crates/eridiffusion-core/src/models/klein.rs:1597-1606`
```rust
// upstream Python Flux2Model.prepare_text_ids:
//   cartesian_prod(arange(1),arange(1),arange(1),arange(L)) → row k = [0,0,0,k]
//   ...each text token receives a distinct RoPE phase.
// Audit fix KLEIN_VERIFY §H2 / SKEPTIC §H2: previously all-zero,
// which collapsed text positions and lost ordering information.
for k in 0..n_txt { txt_ids_data[k * 4 + 3] = k as f32; }   // txt = [0,0,0,k]
```
The Rust port HAD this exact bug, found it by audit (KLEIN_VERIFY §H2), and fixed it.
The Mojo port replicated the pre-fix all-zero version. Upstream truth = Flux2Model.
prepare_text_ids = [0,0,0,k]. (See also memory project_klein_forward_divergence_2026-05-29.)

───────────────────────────────────────────────────────────────────────────────
## The fix (apply AFTER codex hands the tree back)
In BOTH rope builders, for text tokens (`tok < N_TXT`), set the 4th axis to the
text-token index instead of 0. Text tokens are first in the sequence, so the text
index IS `tok` (0 .. N_TXT-1):

- `klein_dit.mojo` `build_klein_rope_tables`: in the `for tok in range(S)` loop, add
  an `else`/text branch: `if tok < N_TXT: p3 = tok`  (keep p0=p1=p2=0).
- `train_klein_real.mojo` `_build_klein_rope_host`: identical change (this fn is a
  byte-for-byte host replica of the above — keep them in lockstep).

Axis check before applying: confirm Mojo's axis-3 block is the one getting the
`axes_dims[3]` rotary frequencies (16 freqs/axis × 4 = 64 = Dh/2 for Dh=128). If the
Mojo axis order differs from the Rust [0,1,2,3]=[t?,row,col,L] mapping, map `k` to
whichever axis is the "L" (text-length) axis. Per current read it is axis 3 (p3).

### MANDATORY companion changes (else the fix looks broken / unverifiable)
1. **Regenerate the Klein parity oracle with [0,0,0,k].** The current torch oracle was
   almost certainly generated with the same all-zero text ids, so the gates pass
   against a matching-but-WRONG reference. Until the oracle is regenerated, the gates
   cannot validate this fix (the exact trap the Rust port hit — relL2 collapsed 10×
   only after fixing ref.py). Find the Klein parity oracle .py and set text ids
   to [0,0,0,k].
2. **Add a real-dim forward parity check.** All current gates are toy dims
   (end-to-end gate D=32/N_IMG=4; per-block D=512). The real config is
   D=3072, N_IMG=1024, S=1536 — never gate-tested. A bug in this class is invisible
   to toy gates. Add a real-dim (or at least real-N_TXT) text-rope forward parity.

───────────────────────────────────────────────────────────────────────────────
## Verification (Tenet 4 — measure, don't assert)
After applying + oracle regen:
1. All 5 Klein parity gates green at cos ≥ 0.99999999 against the REGENERATED oracle.
2. Training loss over a multi-step run drops toward the ~0.75 baseline (from ~2.7).
   This is the real confirmation that the causal hypothesis was correct.
3. (Bonus) Klein inference stops producing noise.
Only then is this "fixed" — not before.

───────────────────────────────────────────────────────────────────────────────
## Already checked and CLEARED this session (not the cause)
- Noise: `_host_noise` is correct Box-Muller N(0,1) (advances PCG state between u1/u2).
- Latent packing: permute(0,2,3,1)+reshape, mirrors inference; no missing VAE scale.
- Loss formula: x_t=(1-σ)·latent+σ·noise, target=noise-latent, mean MSE — identical to Rust.
- theta=2000 matches KleinConfig and EDv2.
- TIMESTEP_SHIFT=1.8 (vs baseline default 1.0): real σ knob, shifts loss somewhat, but
  a documented sweet-spot — secondary, not the 2.7-vs-0.75 gap.

Memory: project_mojo_klein_txt_rope_bug_2026-05-31.
