# SKEPTIC FINDINGS — LTX-2 P3 joint-AV block-0 parity gate (2026-05-28)

Auditor: SKEPTIC (independent). Subject: builder's PASS claim for the dual-stream
LTX-2 block-0 forward (`ltx2_block_forward_av`) gated by
`serenitymojo/pipeline/ltx2_av_block_parity_smoke.mojo` against the Python oracle
`scripts/ltx2_av_block0_parity.py`, vs Rust ground truth
`LTX2TransformerBlock::forward_with_skip` (ltx2_model.rs:1148-1471).

## Verdict

**The PASS is REAL for the dominant paths, but the gate is NOT fully fail-closed.**
The implementation is line-for-line faithful to the Rust block (all 6 attention
paths, correct order/fusion, correct table indexing, correct gate placement,
correct cross-modal Q/KV dim mapping). The baseline reproduces independently:

- VIDEO cos = **0.99995196** (>= 0.999)  ✅
- AUDIO cos = **0.99998690** (>= 0.999)  ✅

Reproduced exactly across two clean runs. GPU peak well under guard (23.4 GB free).

HOWEVER: the **v2a cross-modal sub-path is effectively unverified** — zeroing the
entire v2a contribution to the audio stream still PASSES (AUDIO cos 0.99917 > 0.999).
See "Fail-closed" below. This is a gate-coverage gap, not a correctness bug in the
shipped code, but it means a future regression in `video_to_audio_attn` /
`audio_*_v2a` modulation would slip through silently.

## (1) Six attention paths — present, correct order, correct fusion  ✅

Verified against Rust residual application order (ltx2_model.rs):
1. vsa  attn1            video self  (1223-1230) — Mojo 1092-1096  ✅
2. asa  audio_attn1      audio self  (1254-1262) — Mojo 1111-1115  ✅
3. vca  attn2            video↔text  (1298-1299) — Mojo 1133-1137  ✅
4. aca  audio_attn2      audio↔text  (1330-1334) — Mojo 1155-1159  ✅
5. a2v  audio_to_video   Q=video/KV=audio (1386-1396) — Mojo 1173-1177  ✅
6. v2a  video_to_audio   Q=audio/KV=video (1419-1431) — Mojo 1182-1186  ✅
7. vff / 8. aff FFN (1438-1461) — Mojo 1189-1205  ✅

Shared-norm timing is correct: `norm_a2v`/`norm_v2a` are both computed from the
post-CA states BEFORE the a2v residual mutates `hs` (Rust 1345-1346 precede the
a2v add at 1396; Mojo 1162-1163 precede the a2v add at 1177). Both a2v and v2a
derive their Q/KV mods from these same pre-mutation norms. ✅

## (2) Audio scale_shift / a2v cross-modal table indexing — correct  ✅

`scale_shift_table_a2v_ca_video[5,4096]` and `..._audio[5,2048]`:
rows 0,1 = a2v scale/shift; rows 2,3 = v2a scale/shift; row 4 = gate.
- Rust `compute_cross_attn_params` (1623-1655): video rows split 0/1→a2v, 2/3→v2a,
  row 4→a2v_gate; audio rows 0/1→a2v, 2/3→v2a, row 4→v2a_gate.
- Mojo `_compute_cross_mod` (908-924): identical rows; `a2v_gate` from v_table row 4,
  `v2a_gate` from a_table row 4.  ✅
- Which stream each modulates: a2v_gate (video table row4) gates the a2v output added
  to the VIDEO stream; v2a_gate (audio table row4) gates the v2a output added to the
  AUDIO stream. Matches Rust 1396 / 1431.  ✅
- 9-coeff per-stream AdaLN: `_ada_row_pertok` idx 0..5 (msa shift/scale/gate, mlp
  shift/scale/gate) and idx 6,7,8 (CA shift/scale/gate) match Rust
  `compute_ada_params_6`/`compute_ada_params_ca`.  ✅
- KV (text) modulation `_kv_modulate` (combined = psst[2,dim] + prompt_ts; row0=shift,
  row1=scale) matches Rust 1281-1296 / 1316-1328.  ✅

## (3) Per-head gate before to_out — correct on all paths  ✅

Rust (640-712): gate_logits = linear(Q-input hidden_states), `2*sigmoid`, per-head,
applied to attn-out reshaped to [B,Sq,H,Dh] BEFORE the to_out projection.
Mojo `_av_attention` (1012-1026): `linear(hidden,...) → [1,Sq,H]`, `2*sigmoid`,
reshape [1,Sq,H,1], multiply, then to_out. Gate sourced from the Q-INPUT (post
modulation), conditional on `to_gate_logits.weight` presence — identical to Rust.  ✅
Oracle (253-290) mirrors this exactly. ✅

## (4) Cross-modal Q/KV dim mapping (4096↔2048) — correct  ✅

Both `audio_to_video_attn` and `video_to_audio_attn` are loaded with audio
heads/head_dim 32×64 (inner=2048) — Rust load_attention 2039/2043. The 4096↔2048
conversion lives in the loaded weight shapes:
- a2v: to_q [2048,4096] (Q=video 4096), to_k/to_v [2048,2048] (KV=audio), to_out
  [4096,2048] (→video). Gate weight [32,4096] sourced from the video Q-input.
- v2a: to_q [2048,2048] (Q=audio), to_k/to_v [2048,4096] (KV=video), to_out
  [2048,2048] (→audio).
Mojo uses `weights._linear_b` which reads the real weight shapes; `_av_attention[...,
32,64]` only sets the inner SDPA head geometry. The dim conversion is therefore
weight-driven and correct. Cross-modal RoPE assignment a2v(q=ca_v, k=ca_a) /
v2a(q=ca_a, k=ca_v) matches Rust 1386-1388 / 1419-1422 and oracle 445-455.  ✅

## (5) Fail-closed test — PARTIAL. Gate is NOT fully fail-closed.

Each mutation was applied to the clean tree, run on GPU, then reverted (file
restored byte-identical to backup; final clean baseline re-run PASSES at the
reported numbers).

| Mutation                                   | VIDEO cos  | AUDIO cos  | Gate result |
|--------------------------------------------|-----------:|-----------:|-------------|
| (none — clean baseline)                    | 0.99995196 | 0.99998690 | PASS        |
| swap a_v2a scale/shift rows 2,3 → 0,1      | 0.99995196 | 0.99995660 | **PASS (miss)** |
| zero v2a gate (kill v2a→audio entirely)    | 0.99995196 | 0.99916834 | **PASS (miss)** |
| zero a2v gate (kill a2v→video entirely)    | 0.98449487 | 0.99998690 | FAIL (raises)  ✅ |
| zero audio self-attn gate (audio_attn1)    | 0.99567443 | 0.55017970 | FAIL (raises)  ✅ |

Findings:
- The **a2v path (→video)** is well protected: removing it drops VIDEO cos to 0.9845
  and the smoke raises (exit 1).  ✅
- The **dominant audio path (self-attn)** is well protected: removing it craters
  AUDIO cos to 0.55, and the corruption even propagates through the a2v KV into the
  video stream (VIDEO 0.9957), so the smoke raises. This proves the streams are
  genuinely cross-coupled, not independent.  ✅
- **GAP: the v2a sub-path's contribution to the audio stream is too small to trip the
  0.999 bar.** Zeroing it entirely leaves AUDIO cos at 0.99917 — a PASS. A wrong
  cross-modal table row on the audio side (0.99996) is likewise invisible. At
  block-0 with S_A=8, the v2a residual is a tiny fraction of the audio output
  (dominated by self-attn + text-CA + FFN), so cosine 0.999 cannot see it.

Consequence: a future regression in `video_to_audio_attn`, `audio_v2a` modulation,
or `v2a_gate` (audio table row 4) would pass this gate silently. The v2a path is
*implemented correctly* (it matches Rust line-for-line and the oracle), but it is
*not protected* by the current gate.

## Punch list (gate hardening — code is correct, coverage is not)

1. **Add a per-path delta gate, not just an end-to-end cosine.** Dump the Rust/oracle
   intermediate `dump_after_v2a` (and `dump_after_a2v`, `dump_v2a_raw_out`) — the Rust
   already emits these under LTX2_DUMP_BLOCK0 (ltx2_model.rs:1424-1433) — and gate the
   v2a *raw output* and *post-v2a audio state* directly at cos>=0.999. That isolates
   the v2a path from the swamping residual.
2. **Or tighten the audio bar / use relative-error on the v2a delta.** A relative L2 on
   `(audio_out - audio_in_pre_v2a)` would expose the v2a contribution that cosine
   currently hides.
3. **Or pick a higher-energy v2a test input.** Block-0's tiny S_A=8 makes v2a
   negligible; a fixture with larger audio token count / non-trivial cross-modal
   coupling would make the end-to-end audio cosine sensitive to v2a.
4. Until one of the above lands, treat the **audio cross-modal (v2a) path as
   GATE-UNVERIFIED** in the P3 sign-off, even though the line-by-line audit and the
   oracle agree it is correct.

## Notes

- Oracle faithfulness: `scripts/ltx2_av_block0_parity.py::run_block` (377-470) and
  `attention` (253-290) are a faithful F32 port of the Rust block — verified
  line-by-line (residual order, table indexing, gate, RoPE q/k assignment, KV
  modulation). The oracle's direct [H,Sq,Skv] softmax vs the Mojo pad-to-SPAD + masked
  SDPA are mathematically equivalent given the pad-mask; the 0.99995 baseline confirms
  the padding/mask is correct.
- block-0 is genuinely BF16 (FP8 quant skips boundary blocks); the gate is BF16-faithful
  vs the F32 oracle, which is the standard parity bar. The FP8 dequant path (P5) is not
  exercised here.
- All mutations reverted; tree left clean (file byte-identical to pre-audit backup);
  final clean baseline re-run PASSES (VIDEO 0.99995196 / AUDIO 0.99998690). No git
  commit made.
