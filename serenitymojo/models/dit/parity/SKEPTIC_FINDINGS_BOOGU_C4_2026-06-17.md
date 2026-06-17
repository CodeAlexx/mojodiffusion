# Skeptic findings — Boogu Chunk C4 (DOUBLE-stream block) — 2026-06-17

Adversarial review of `BooguDoubleStreamBlock` + `_lumina_rms_norm_zero` +
`_joint_attn`/`_img_self_attn` in
`serenitymojo/models/dit/boogu_dit.mojo:535-955`, and the probe
`serenitymojo/models/dit/parity/boogu_c4_block_probe.mojo`.

Reference read line-by-line:
- `transformer_boogu.py:558-683` (`BooguImageDoubleStreamTransformerBlock.forward`,
  modulation=True 596-683)
- `attention_processor.py:505-877` (`BooguImageDoubleStreamSelfAttnProcessor`:
  `__call__` 706-877, `_concat` 593-646, `_split` 648-704)
- `attention_processor.py:1163-1275` (`BooguImageAttnProcessor`, the img self-attn)
- `block_lumina2.py:39-71` (`LuminaRMSNormZero`)
- `embeddings.py:80-135` (`apply_rotary_emb`, `use_real=False` branch)
- `rope.py:266-448` (joint/combined-img rope construction)
- reused C3 code (`build_boogu_rope_tables`, `_expand_rope_table_per_head`,
  `BooguBlock._attn`) and foundation ops (`linear`, `rms_norm`, `slice`, `concat`,
  `mul/add/add_scalar`, `repeat_kv_f32`, `rope_interleaved`, `sdpa_nomask`).

## Probe re-run (compile honesty)
```
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg \
  && pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c4_block_probe.mojo
```
Exit 0. Real output:
```
[c4-probe] block loaded (modulation=True)
img_out shape: 1 256 3360 (expect 1 256 3360 )
img_out std: 0.59399283
instruct_out shape: 1 16 3360 (expect 1 16 3360 )
instruct_out std: 0.5921206
```
Shapes correct, outputs non-degenerate. The checkpoint load (44 keys, incl.
`img_instruct_attn.processor.*`, `img_instruct_attn.to_out.0`,
`img_instruct_attn.norm_q/k`, `img_self_attn.*`) succeeded against the real
transformer dir, confirming all weight keys exist.

## Reference-fidelity checks (the high-risk spots)

- **Concat order (q/k/v) — INSTRUCT-FIRST.** `_joint_attn:766-768` does
  `concat(1, ctx, ins_q, img_q)` for q, k, AND v. `concat` (tensor_algebra.mojo:1407-1420)
  places operands in argument order along the dim, so the layout is instruct rows
  then img rows. Matches `_concat_instruction_image_features` (attn_processor.py:638
  instruct first, 640 img after). CORRECT.
- **Split after attention — INSTRUCT-FIRST + right halves + re-concat.**
  `_joint_attn:791-795`: `ins_h = slice(merged,1,0,L_INSTRUCT)` = rows [0:16],
  `img_h = slice(merged,1,L_INSTRUCT,L_IMG)` = rows [16:272] (slice 4th arg is
  LENGTH, tensor_algebra.mojo:1431/1458, so [16,16+256)=[16,272)). `instruct_out`
  applied to ins_h, `img_out` to img_h (793-794), re-concat instruct-first (795),
  then `to_out.0` (797). Matches attn_processor.py:854-874. CORRECT.
- **Separate projections, un-swapped.** `_joint_attn:759-764`: `ji_img_to_q/k/v`
  applied to `img_norm1_out`, `ji_ins_to_q/k/v` to `instruct_norm1_out`. The
  processor's own `instruct_out`/`img_out` (`ji_ins_out`/`ji_img_out`) are distinct
  from the joint `to_out.0` (`ji_to_out`). Load keys (719-727) confirm
  `processor.img_to_q` … `processor.instruct_out`/`img_out` vs the Attention-level
  `img_instruct_attn.to_out.0`. CORRECT.
- **Two ropes — joint 272 vs combined-img 256 = joint rows [16:272].** Joint attn
  fed `cos_j`/`sin_j` (full 272-row joint table, forward:902-903). Img self-attn fed
  `cos_img`/`sin_img` = `slice(cos_j,0,L_INSTRUCT,L_IMG)` = rows [16:272]
  (forward:863-864). Verified against rope.py:427-437: with `sum(ref_img_len)==0`,
  `combined_img_freqs_cis[:, 0:img_len] = freqs_cis[:, cap_len : cap_len+img_len]`,
  and `rotary_emb` returned == full `freqs_cis` (rope.py:443). So combined-img == joint
  rows [cap_len:cap_len+img_len] = [16:272], and joint == full table. Mojo slices
  [16:272], NOT [0:256]. CORRECT — this was the flagged silent killer; it is right.
- **Modulation chunk usage.** `_lumina_rms_norm_zero:623-633` returns
  `(out=rms_norm(x)*(1+chunk0), chunk1, chunk2, chunk3)` where `chunk` order is
  `(scale_msa, gate_msa, scale_mlp, gate_mlp)` (block_lumina2.py:69). `out` folds
  chunk0=scale_msa via `(1+scale_msa)`. forward:
  img_norm1 → (out, gate_msa=ch1, scale_mlp=ch2, gate_mlp=ch3) (870-873);
  img_norm2 → (out, shift_mlp=ch1) (878-879); img_norm3 → (out, gate_self=ch1)
  (884-885); instruct_norm1 → (out, gate_msa, scale_mlp, gate_mlp) (890-893);
  instruct_norm2 → (out, shift_mlp=ch1) (898-899). All read chunk1 for the
  shift/gate-self slots, matching the reference's
  `_, shift, _, _ = self.img_norm2(...)` style (transformer_boogu.py:601/602/610).
  CORRECT.
- **Residual math (transformer_boogu.py:651-683).**
  `img1 = img + tanh(img_gate_msa)*img_attn_norm(img_attn_out)` (916);
  `img2 = img1 + tanh(img_gate_self)*img_self_attn_norm(img_self_attn_out)` (921);
  `img_mlp_in = (1+img_scale_mlp)*img_norm2_out + img_shift_mlp` (923-927,
  modulates `img_norm2_out`, NOT `ffn_norm1(img)`);
  `img_out = img2 + tanh(img_gate_mlp)*img_ffn_norm2(img_ff(img_ffn_norm1(img_mlp_in)))`
  (928-932). Instruct path analogous, NO self-attn / NO norm3 (935-953).
  tanh on gates only, `(1+)` on scales only, ffn_norm1 inside / ffn_norm2 outside,
  `img_attn_out` taken from the SPLIT joint output (`slice(joint_attn_out,1,16,256)`
  forward:906) not the raw joint. CORRECT.
- **norm2/norm3 read the PRE-residual hidden.** All 5 LuminaRMSNormZero
  (forward:867-899) consume the ORIGINAL `img`/`instruct`, never an intermediate
  residual — matching transformer_boogu.py:598-612 (a Lumina2 quirk the port
  preserves). CORRECT.

## Mojo correctness
- Tensor is Movable-not-Copyable; tuple returns move owned locals (`return (out^, c1^,
  c2^, c3^)` :633; `(img_out^, ins_out^)` :955). Callers `.clone(ctx)` out of tuple
  subscripts (`m_n1[0].clone(ctx)` etc., never `^` out of a subscript). No
  use-after-move (each modulation result cloned before reuse). No `List[Tensor]`.
- No var named `ref`; all functions `def` (raises). `comptime` used (not `alias`);
  `comptime JOINT = L_INSTRUCT + L_IMG` is a comptime expr. sdpa comptime params are
  `[1, JOINT=272, 28, 120]` and `[1, L_IMG=256, 28, 120]`. Compiles (exit 0).
- I/O via `ShardedSafeTensors` + `Tensor.from_view` only (`_load_w`). No op
  reimplemented — `_joint_attn`/`_img_self_attn` reuse C3 helpers + foundation ops.
- concat/slice operate on dim 1 (seq) for q/k/v/split and dim 0 (row) for the rope
  table slice — both correct dims.

## Numerical / dtype
- qk-norm BEFORE rope (`rms_norm` :774-775 / :810-811 then `rope_interleaved`
  :781-782 / :816-817). Matches attn_processor.py:802-810 / 1206-1215.
- repeat_kv AFTER rope (:784-785 / :818-819). Matches reference (repeat_interleave at
  1260-1261, after rope).
- GQA head→kv mapping: `repeat_kv_f32` kernel uses `kvh = head // n_rep`
  (gqa_backward.mojo:44), i.e. repeat_interleave (dst head h reads src kv-head h//4),
  matching `key.repeat_interleave(query.size(-3)//key.size(-3), -3)`. CORRECT.
- BF16-store / F32-accum: every op (`linear`, `rms_norm`, `sdpa_nomask`, `silu`,
  `tanh_op`, `swiglu`, `rope_interleaved`, `_binary` mul/add/add_scalar) computes in
  F32 and stores x's dtype. Chain stays BF16 end-to-end, so `_binary`'s dtype-match
  guard (tensor_algebra.mojo:260) never fires. The probe's clean run confirms no
  dtype-mismatch raise on any of the ~20 mul/add calls.
- Modulation broadcast: chunks reshaped to `[1,1,3360]` (`_lumina_rms_norm_zero`
  :630-632) broadcast over the seq dim via NumPy right-aligned broadcasting in
  `_binary`, matching `gate.unsqueeze(1)` / `scale.unsqueeze(1)` in the reference.
  CORRECT.
- rope convention: `apply_rotary_emb(use_real=False)` (embeddings.py:126-134) is
  adjacent-pair interleaved (view_as_complex of reshape(...,D//2,2)); serenitymojo
  `rope_interleaved` (rope.mojo:7-9) is exactly `out[2i]=x[2i]cos-x[2i+1]sin,
  out[2i+1]=x[2i]sin+x[2i+1]cos` with table width 60 = head_dim/2. cos=real,
  sin=imag (same oracle gate as C3). CORRECT.

## Notes (not blockers)
- STYLE: `_joint_attn` returns the FULL joint output and the caller re-slices it into
  `instruct_attn_out`/`img_attn_out` (forward:905-906) — a second split after the
  processor already split+re-merged internally. This is faithful to the reference
  (the processor returns a merged joint tensor at attn_processor.py:871-877; the
  block then splits it again at transformer_boogu.py:626-641). Numerically identical;
  just two extra slice copies. Acceptable for inference.
- FRAGILE: the combined-img rope correctness depends on the no-ref T2I assumption
  (`ref_img_len == 0`). For any ref-image / I2I path the `slice(cos_j, 0, L_INSTRUCT,
  L_IMG)` derivation would NOT equal `combined_img_freqs_cis` (which would interleave
  ref-img rows). The code + probe are scoped to T2I no-ref (documented at
  boogu_dit.mojo:581-583), so this is correct for the targeted case, but a future
  ref-image extension must NOT reuse this slice.

BLOCKERS: 0 — clean
