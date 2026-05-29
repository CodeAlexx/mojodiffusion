# Skeptic Findings — Microsoft Lens DiT Block-0 Full Forward

- Timestamp: 2026-05-28
- Auditor: Skeptic agent (read-only)
- Verdict: PASS with minor WARN — structural match against Rust oracle confirmed; one acknowledged scope-limit (synthetic inputs vs. real BF16 sidecar wiring).

## Files audited

- Rust oracle: `/home/alex/EriDiffusion/inference-flame/src/models/lens_dit.rs` (1600 lines)
- Mojo math: `/home/alex/mojodiffusion/serenitymojo/models/lens/lens_dit_math.mojo` (new gate at lines 1095-1733; helpers at 1132-1171)
- Mojo smoke: `/home/alex/mojodiffusion/serenitymojo/pipeline/lens_dit_block0_full_smoke.mojo` (41 lines)
- Contract constants: `/home/alex/mojodiffusion/serenitymojo/models/lens/lens_contract.mojo`

## Rust block-forward citations (anchor evidence)

### adaLN chunk ordering (`lens_dit.rs:651-666`)
```
let temb_act = temb.silu()?;
let img_mod_full = self.img_mod.forward(&temb_act)?;        // [B, 6*dim]
let txt_mod_full = self.txt_mod.forward(&temb_act)?;        // [B, 6*dim]
let img_halves = img_mod_full.chunk(2, 1)?;                 // 2 x [B, 3*dim]
let txt_halves = txt_mod_full.chunk(2, 1)?;
let img_m1 = img_halves[0].chunk(3, 1)?;                    // 3 x [B, dim]
let img_m2 = img_halves[1].chunk(3, 1)?;
let txt_m1 = txt_halves[0].chunk(3, 1)?;
let txt_m2 = txt_halves[1].chunk(3, 1)?;
let (img_shift1, img_scale1, img_gate1) = (&img_m1[0], &img_m1[1], &img_m1[2]);
let (img_shift2, img_scale2, img_gate2) = (&img_m2[0], &img_m2[1], &img_m2[2]);
let (txt_shift1, txt_scale1, txt_gate1) = (&txt_m1[0], &txt_m1[1], &txt_m1[2]);
let (txt_shift2, txt_scale2, txt_gate2) = (&txt_m2[0], &txt_m2[1], &txt_m2[2]);
```
Concretely the 6*dim row indexing is: `[0:dim)=shift1, [dim:2dim)=scale1, [2dim:3dim)=gate1, [3dim:4dim)=shift2, [4dim:5dim)=scale2, [5dim:6dim)=gate2`.

### SwiGLU (`lens_dit.rs:909-915`)
```
fn swiglu_mlp(x: &Tensor, w1: &Linear, w2: &Linear, w3: &Linear) -> Result<Tensor> {
    let gate = w1.forward(x)?;
    let up = w3.forward(x)?;
    let activated = flame_core::bf16_ops::swiglu_fused_bf16(&gate, &up)?;
    w2.forward(&activated)
}
```
i.e. `w2( silu(w1(x)) * w3(x) )` — silu on the w1 branch.

### Gate residual (`lens_dit.rs:697-700,715-716,730-731`)
```
let img_gate1_b = img_gate1.unsqueeze(1)?;                  // [B, 1, dim]
let hidden_states = hidden_states.add(&img_gate1_b.mul(&img_attn)?)?;
let encoder_hidden_states = encoder_hidden_states.add(&txt_gate1_b.mul(&txt_attn)?)?;
...
let hidden_states = hidden_states.add(&img_gate2_b.mul(&img_mlp_out)?)?;
let encoder_hidden_states = encoder_hidden_states.add(&txt_gate2_b.mul(&txt_mlp_out)?)?;
```
Raw gate (no tanh, no LayerNorm between gate and branch).

### Joint attention concat order (`lens_dit.rs:846-853`)
```
let (q, k, v) = if s_txt > 0 {
    let q = Tensor::cat(&[&img_q, txt_q.as_ref().unwrap()], 2)?;
    let k = Tensor::cat(&[&img_k, txt_k.as_ref().unwrap()], 2)?;
    let v = Tensor::cat(&[&img_v, txt_v.as_ref().unwrap()], 2)?;
```
Image-first then text along seq dim.

### Block norms eps (`lens_dit.rs:670-678, 703-722`)
All four block RMSNorms (`img_norm1/img_norm2/txt_norm1/txt_norm2`) use eps=1e-6.

### QK head norms eps (`lens_dit.rs:794-808`)
`norm_q/norm_k/norm_added_q/norm_added_k` all use eps=1e-5.

### 4-layer encoder feature stack (`lens_dit.rs:70, 88, 100-104, 1482-1495`)
```
selected_layer_index = [5, 11, 17, 23]               (line 88)
txt_in_dim = enc_hidden_dim * 4 = 11520              (line 103-104)
for (i, feat) in encoder_hidden_states.iter().enumerate() {
    let normed = flame_core::cuda_ops_bf16::rms_norm_bf16(feat, Some(&self.txt_norm[i]), 1e-5)?;
    normed_layers.push(normed);
}
let txt_cat = Tensor::cat(&normed_refs, 2)?;         // [B, S_txt, 11520]
let mut e = self.txt_in.forward(&txt_cat)?;          // [B, S_txt, 1536]
```
Per-layer RMSNorm with eps=1e-5, then concat along feature dim, then `txt_in` projects to inner_dim=1536.

---

## Audit results

### A. Rust source verification

- A1 — PASS. `lens_dit.rs:618-734` is dual-stream MMDiT: separate `hidden_states` (img) and `encoder_hidden_states` (txt) carried independently, joined only inside `joint_attention` (lens_dit.rs:737-893).
- A2 — PASS. `lens_dit.rs:651-666` (quoted above) produces exactly 6 modulation chunks per stream with order `[shift1, scale1, gate1, shift2, scale2, gate2]`. Builder's claim about row indexing of `[6*dim, dim]` matches `chunk(2,1)` followed by `chunk(3,1)` semantics.
- A3 — PASS. `lens_dit.rs:909-915` (quoted above) implements `w2(silu(w1(x)) * w3(x))`. No GELU; silu is on the w1 (gate) branch, not on w3.
- A4 — PASS. `lens_dit.rs:846-853` concats `[img, txt]` along dim=2 (seq dim of `[B, H, S, D]`). Output split via `attn_reshaped.narrow_owning(1, 0, s_img)` then `.narrow_owning(1, s_img, s_txt)` (lens_dit.rs:877-881).
- A5 — PASS. Block norms eps=1e-6 (`lens_dit.rs:672, 678, 706, 721`); QK head norms eps=1e-5 (`lens_dit.rs:795, 797, 802, 807`).
- A6 — PASS. `selected_layer_index = [5, 11, 17, 23]` (lens_dit.rs:88); `txt_in_dim = enc_hidden_dim * selected_layer_index.len()` (lens_dit.rs:103-104, gives 4*2880=11520). Per-layer `txt_norm.{0..3}` applied BEFORE concat (lens_dit.rs:1485-1494).

### B. Modulation chunk ordering correctness in Mojo

- B1 — PASS. `lens_dit_math.mojo:1365-1367, 1378-1380` read `shift1 = img_mod_out[d]`, `scale1 = img_mod_out[INNER_DIM + d]` → matches Rust `[0:dim)=shift1, [dim:2dim)=scale1`. Comment at lines 1292-1297 explicitly documents the mapping.
- B2 — PASS. Mojo offsets used in step 6 and step 7:
  - `gate1 = img_mod_out[2 * LENS_DIT_INNER_DIM + d]` (line 1584) → rows `[2dim:3dim)`. ✓
  - `shift2 = img_mod_out[3 * LENS_DIT_INNER_DIM + d]` (line 1611) → rows `[3dim:4dim)`. ✓
  - `scale2 = img_mod_out[4 * LENS_DIT_INNER_DIM + d]` (line 1612) → rows `[4dim:5dim)`. ✓
  - `gate2 = img_mod_out[5 * LENS_DIT_INNER_DIM + d]` (line 1636) → rows `[5dim:6dim)`. ✓
  - Same offsets used for `txt_mod_out`. ✓
- B3 — N/A (no failure).

### C. Per-stream attention math

- C1 — PASS. Image RoPE (lens_dit_math.mojo:1132-1147): frame band returns 0 (broadcast row 0, matching Rust `pos_freqs[0]` for axes_dim[0]=8), height band uses signed positions `y-(h//2)` and freq `1/theta^(2k/28)` (axes_dim[1]=28), width band identical for `x-(w//2)`. Rust uses `scale_rope=true` → `cat(neg[-(h-h/2):], pos[:h/2])` which spans signed integers `[-(h-h/2), ..., h/2 - 1]`; with h=8 → `[-4, -3, ..., 3]`. Mojo `y-(h//2)` for y in 0..7 → `[-4, -3, ..., 3]`. Equivalent.
  Text RoPE (lens_dit_math.mojo:1150-1171): row = `max(h//2, w//2) + token`, applied to all three bands with band-specific `theta^(...)` divisors `8 / 28 / 28`. Matches Rust `txt_freqs` (lens_dit.rs:446-492) which reads `pos_freqs[max_vid_index + i]` across the full 32-column half_dim.
- C2 — PASS. Mojo applies Q/K RMSNorm (per-head, head_dim) inside the same loop that applies RoPE (lines 1440-1453), but the norm multiplications happen BEFORE the rotation (`q0 = img_q[base+d0] * q_inv * norm_q_w[d0]`, then rotate). V is never RMSNormed and never RoPE'd. Matches Rust order: rms_norm_bf16 (lens_dit.rs:794-797) → rope_fused_bf16 (lens_dit.rs:819-820).
- C3 — PASS. Mojo lays out per-token as `[tok][head][d]` (lines 1389-1419), which is logically `[B=1, S, H, D]`. Joint SDPA loop traverses `for head: for qi: for kj` (lines 1501-1554), equivalent to the Rust `[B, H, S, D]` SDPA layout.

### D. SwiGLU correctness

- D1 — PASS. Mojo (line 1628): `activated.append(_silu(gate_proj[h_out]) * up_proj[h_out])` where `gate_proj = img_mlp_w1 @ x` (lines 1618-1624) and `up_proj = img_mlp_w3 @ x` (line 1623). So `silu(w1(x)) * w3(x)` — silu on w1 branch. Matches Rust `swiglu_fused_bf16(&gate, &up)` with `gate=w1.forward(x), up=w3.forward(x)`. Identical formula in txt branch (lines 1655-1665). The `_silu(x) = x / (1 + exp(-x))` definition at line 217 is canonical.
- D2 — PASS. `LENS_DIT_MLP_HIDDEN = 4096` (lens_contract.mojo:59). 1536 * 8 / 3 = 4096 exactly. Matches Rust comment "SwiGLU hidden dim: dim * 8 / 3" (lens_dit.rs:94).

### E. Gate-residual pattern

- E1 — PASS. Mojo gate1 residual (lines 1585-1588, 1594-1597): `hidden + gate1 * attn_proj` with gate1 broadcast over the seq dim (taken from `mod_out[2*dim:3*dim]` independently of `tok`). gate2 same (lines 1636-1637, 1673-1674). NO tanh and NO normalization between gate and branch. Matches Rust exactly (no `.tanh()` anywhere in the block forward).

### F. Joint attention concat

- F1 — PASS. Mojo SDPA loop (lines 1501-1554): for each q-index `qi` in `[0, s_total)`, gathers Q from img when `qi < N_IMG` else from txt offset `qi - N_IMG`; same indexing for K and V. This is equivalent to concatenating `[img, txt]` along the seq axis. Order matches Rust.
- F2 — PASS. Split (lines 1556-1577): img branch uses `attn_concat[tok=0..N_IMG]`, txt branch uses `attn_concat[qi = N_IMG + ti]`. Image-first split → image projection (`img_to_out_w/b`); text portion → text projection (`txt_to_out_w/b`, which loads from `to_add_out`). Matches Rust narrow_owning(1, 0, s_img) / narrow_owning(1, s_img, s_txt).

### G. txt_in 4-layer concat scheme

- G1 — PASS. `LENS_TEXT_FEATURE_LAYERS = 4` (lens_dit_math.mojo:53); `LENS_TEXT_CAT_DIM = 4 * 2880 = 11520` (lens_dit_math.mojo:54); txt_in.weight shape header check `[1536, 11520]` (line 1181). Matches Rust `selected_layer_index = [5, 11, 17, 23]` and `txt_in_dim = 4 * 2880`.
- G2 — PASS. Mojo (lines 1323-1342) computes per-layer RMSNorm with eps=1e-5 (line 1330) using `txt_norm0_w / txt_norm1_w / txt_norm2_w / txt_norm3_w` (the four separate `txt_norm.{i}.weight` tensors). Each layer's RMSNorm is applied to that layer's synthetic features, then the per-token output is laid out as `[tok * LENS_TEXT_CAT_DIM + (layer * LENS_GPT_OSS_HIDDEN + d)]` which is equivalent to `cat(normed_layers, dim=feature)`. Matches Rust ordering at lens_dit.rs:1484-1494.
- G3 — WARN (acknowledged scope-limit). The full block-0 smoke uses deterministic synthetic per-layer text features (`_det_f32(tok, d, 13+layer)`) rather than wiring real cached BF16 GPT-OSS layer outputs through the sidecar. This is explicitly called out in the builder's deviation list and is consistent with the gate's purpose (compile-only structural smoke). The real-sidecar parity oracle remains the responsibility of the prior text smoke (lens_dit_text_qk_rope_smoke.mojo) and a future BF16-capture parity stage.

### H. Compile + no regressions

- H1 — PASS. `pixi run mojo build -I . -Xlinker -lm serenitymojo/pipeline/lens_dit_block0_full_smoke.mojo -o /tmp/lens_dit_block0_full_smoke` → `exit=0`, binary 226,808 bytes (matches builder's claim).
- H2 — PASS. All four prior Lens smokes still build exit 0:
  - `lens_contract_smoke.mojo` exit=0
  - `lens_dit_qkv_smoke.mojo` exit=0
  - `lens_dit_qk_rope_smoke.mojo` exit=0
  - `lens_dit_text_qk_rope_smoke.mojo` exit=0

### I. Bounded vs production sizes

- I1 — PASS. Smoke uses `LENS_FULL_N_IMG=64, LENS_FULL_N_TXT=64, LENS_FULL_H=8, LENS_FULL_W=8` (lens_dit_math.mojo:1117-1120). Production hidden constants (`LENS_DIT_INNER_DIM=1536`, `LENS_DIT_HEADS=24`, `LENS_DIT_HEAD_DIM=64`, `LENS_DIT_MLP_HIDDEN=4096`, `LENS_TEXT_CAT_DIM=11520`, `LENS_QKV_WIDTH=4608`) are unchanged. Only the spatial/sequence dimensions were bounded.

---

## Summary tally

- PASS: 21 (A1-A6, B1, B2, C1-C3, D1, D2, E1, F1, F2, G1, G2, H1, H2, I1)
- WARN: 1 (G3 — synthetic vs. real sidecar; acknowledged in deviations)
- FAIL: 0

## Confirmations to builder

- MMDiT dual-stream structure (separate img/txt streams + joint attention image-first-then-text): CONFIRMED.
- SwiGLU `w2(silu(w1(x)) * w3(x))` formula: CONFIRMED.
- 4-layer encoder concat with per-layer `txt_norm.{0..3}` RMSNorm (eps=1e-5) → 11520 → txt_in → 1536: CONFIRMED.
- 6-chunk modulation order `[shift1, scale1, gate1, shift2, scale2, gate2]`: CONFIRMED.
- Block RMSNorm eps=1e-6, QK head RMSNorm eps=1e-5: CONFIRMED.
- Raw-gate residual (no tanh): CONFIRMED.

## Bugfix Worklist (ordered)

There are no correctness FAILs. Suggested follow-ups (deferred enhancements, not block-0 bugs):

1. (Deferred enhancement, not a bug) Wire a real-sidecar parity oracle: extend the smoke (or add a new pipeline) to ingest the cached BF16 GPT-OSS layer outputs for selected_layer_index `[5, 11, 17, 23]`, run them through `txt_norm.{0..3}` then `txt_in`, and compare against a Rust-captured `e` tensor at the block-0 boundary. This is the natural successor to lens_dit_text_qk_rope_smoke.
2. (Deferred) Add a parity capture for block-0 outputs `(encoder_hidden_states, hidden_states)` post-MLP-residual under real weights + a fixed real temb, to gate the full forward against the Rust oracle (cos>=0.999).
3. (Deferred, performance) The current Mojo gate is pure-CPU triple-nested loops (no MAX/GPU). For larger N this will be intractable; eventually port to MAX kernels for parity at production N=4096 image tokens.
4. (Optional) Add an explicit comment in `_lens_image_rope_angle_hw` noting the equivalence with Rust's `scale_rope=true` neg+pos concat construction, so future readers don't have to derive it.

## Verdict

PASS. The Mojo full-block-0 structural gate faithfully mirrors the Rust oracle for adaLN modulation ordering, RMSNorm eps split (block 1e-6 / QK 1e-5), Q/K RMSNorm-then-RoPE order with V untouched, joint image-first-then-text SDPA concat, image/text output projections, raw-gate residuals, and SwiGLU `w2(silu(w1)*w3)` with hidden=4096. The 4-layer encoder concat with per-layer `txt_norm.{0..3}` RMSNorm (eps=1e-5) is structurally correct. Sole WARN is the acknowledged use of synthetic deterministic text features instead of real cached BF16 sidecar — orthogonal to block-0 math correctness and explicitly out of scope.
