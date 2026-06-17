# Skeptic findings — Boogu C6 full DiT forward wiring (2026-06-17)

Adversarial review of `struct BooguDiT` + `boogu_patchify` in
`serenitymojo/models/dit/boogu_dit.mojo` and the probe
`serenitymojo/models/dit/parity/boogu_c6_dit_probe.mojo`.

Reference read in full:
- `forward` `Boogu-Image/boogu/models/transformers/transformer_boogu.py:1253-1607`
- `flat_and_pad_to_seq:1096-1196`, `img_patch_embed_and_refine:987-1094`,
  `preprocess_instruction_hidden_states:1215-1251`
- `rope.py` forward `266-448` + `_get_freqs_cis:250-264`
- double-stream block forward `558-683`; double-stream attn processor
  `attention_processor.py:505-877`; single attn processor `1186-1275`
- `block_lumina2.py` LuminaRMSNormZero `39-71`, LuminaLayerNormContinuous `74-122`,
  LuminaFeedForward `125-174`, Lumina2CombinedTimestepCaptionEmbedding `177-219`
- C1–C5 structs in `boogu_dit.mojo`

Probe re-run (`rm -f serenitymojo.mojopkg && pixi run mojo run -I . .../boogu_c6_dit_probe.mojo`):
**exit 0**; velocity shape `1 16 32 32`, std 0.9464638. Compiles + executes.

---

## Block order, counts, and prefixes — VERIFIED CLEAN

Checkpoint index (`diffusion_pytorch_model.safetensors.index.json`) confirms exactly:
context_refiner 0-1 (2), noise_refiner 0-1 (2), double_stream_layers 0-7 (8),
single_stream_layers 0-31 (32). The port loops `range(2/2/8/32)` with `String(i)`
prefixes (`boogu_dit.mojo:1444-1477`) — no off-by-one, no missing/duplicated block.

Apply order in `forward` (`1521-1550`): context_refiner → x_embed+noise_refiner →
double_stream → fuse → single_stream → norm_out → extract → unpatchify. Matches the
reference forward exactly (`transformer_boogu.py:1342` context, `1350` patch+noise,
`1450-1485` double, `1494-1505` fuse, `1552-1566` single, `1574` norm_out,
`1578-1592` extract+unpatchify). context_refiner correctly runs before
noise_refiner (independent streams; order also faithful).

`ref_image_refiner` (2) and `ref_image_patch_embedder` exist in the checkpoint but
are correctly NOT loaded/used in the no-ref T2I path.

## Modulation flags — VERIFIED CLEAN

- context_refiner loaded `modulation=False` (`boogu_dit.mojo:1449`). Checkpoint keys
  for `context_refiner.0` have plain `norm1.weight` and NO `norm1.linear.*` — the
  else-branch in `BooguBlock.load:411-417` loads `norm1.weight`. A flag swap would
  fail-loud on the missing key.
- noise_refiner + single_stream loaded `modulation=True` (`1457`, `1474`). Their
  checkpoint keys have `norm1.linear.weight/bias` + `norm1.norm.weight`, loaded by
  the if-branch (`408-410`). Correct.

## RoPE per stage — VERIFIED CLEAN

`slice` signature is `slice(x, dim, start, length, ctx)` (`tensor_algebra.mojo:1430`).
Reference (`rope.py:411-437`, no-ref → `sum(ref_img_len)=0`):
cap = `freqs_cis[0:cap]`, img/combined-img = `freqs_cis[cap:cap+img_len]`,
rotary_emb = full joint.

Port matches:
- cap rope `slice(cos_j,0,0,CAP_LEN)` → rows[0:16] (`1516-1517`)
- img rope `slice(cos_j,0,CAP_LEN,IMG_LEN)` → rows[16:272], length 256 (`1518-1519`)
- single_stream uses full `cos_j/sin_j` [0:272] (`1548-1549`)
- double_stream builds joint internally `build_boogu_rope_tables(L_INSTRUCT,h,w)` and
  derives combined-img as `slice(...,0,L_INSTRUCT,L_IMG)` = rows[16:272] (`863-869`)
- noise_refiner gets img rope; context_refiner gets cap rope. Correct.

No off-by-one, no swap. `_get_freqs_cis` cat order axis0|axis1|axis2 (width 60) ==
`build_multiaxis_rope_tables([40,40,40])`; cos↔real / sin↔imag gated in C2.

## Fusion order & extract — VERIFIED CLEAN

Fuse `concat(1, ctx, caption, x)` → instruct-first [1,272,3360] (`1544`), matching
`transformer_boogu.py:1500-1505` (instruct rows first, img rows after).
Extract `slice(y, 1, CAP_LEN, IMG_LEN)` → cols[16:272] = the LAST 256 (image) rows
(`1556`), matching `hidden_states[i][seq_len-img_len:seq_len]` (`1583`). NOT [0:256].

Double-stream split also instruct-first: `joint[:, :16]`=instruct, `joint[:, 16:272]`=img
(`910-911`), matching processor split + block split (`628-641`).

## Modulation chunk semantics (double-stream) — VERIFIED CLEAN

LuminaRMSNormZero chunk order = (scale_msa, gate_msa, scale_mlp, gate_mlp)
(`block_lumina2.py:69`). Port `_lumina_rms_norm_zero` returns (out=norm*(1+scale_msa),
chunk1, chunk2, chunk3) (`611-638`). Reference reads:
- img_norm2 chunk1 as `img_shift_mlp`, img_norm3 chunk1 as `img_gate_self`
  (`transformer_boogu.py:601-602`). Port: `m_n2[1]`→shift, `m_n3[1]`→gate_self
  (`884`, `890`). Correct.
- mlp_in `(1+scale_mlp)*norm2_out + shift_mlp` then ffn_norm1 (`659-662`). Port `928-935`.
- residual order (msa gate, self gate, mlp gate; tanh on gates) matches `652-683`.

Single-stream chunk reads (`BooguBlock.forward:499-503`): chunk1=gate_msa,
chunk2=scale_mlp, chunk3=gate_mlp; residuals `352-360` of reference. MLP applies
`(1+scale_mlp)` AFTER ffn_norm1 (no shift) — matches `356`. Correct (differs from
double-stream which has a shift; both faithful to their refs).

## Patchify ⇆ unpatchify inverse — VERIFIED CLEAN

Reference rearrange `"c (h p1)(w p2) -> (h w)(p1 p2 c)"` → within-token flat index
`(p1*p+p2)*C + c` (c FASTEST), token `h*w_tok+w`.
- `_boogu_patch_kernel_*` (`1253-1319`): token→(h,w), within→(c=within%C, pp=within//C,
  p1=pp//p, p2=pp%p), reads `latent[c, h*p+p1, w*p+p2]`. ⇒ tokens[token,within]=
  latent[within%C, h*2+p1, w*2+p2] with within=(p1*2+p2)*16+c. Exactly the spec.
- `_boogu_unpatch_kernel_*` (`1066-1129`): exact inverse (out[c,oh,ow] reads
  tokens[h*w_tok+w, (p1*p+p2)*C+c]). h/w not transposed; channel order consistent.

## Mojo correctness — VERIFIED CLEAN

- Blocks in `List[ArcPointer[BooguBlock]]` / `List[ArcPointer[BooguDoubleStreamBlock]]`
  (`1413-1416`); accessed `self.<list>[i][].forward[...]` — deref then borrow-call,
  no move out of the ArcPointer, no use-after-move, no aliasing hazard.
- Tensor is Movable-not-Copyable: tuple results cloned (`tc[0].clone`, `jt[0].clone`,
  `joint_rope[0].clone`, `ds[0/1].clone`) since `^` out of a tuple subscript is illegal.
  Correct pattern.
- comptime `CAP_LEN/H_TOK/W_TOK` thread into `IMG_LEN`/`JOINT` and the `[16]/[256]/[272]`
  sdpa/forward instantiations (`forward[CAP_LEN]`, `forward[IMG_LEN]`, `forward[JOINT]`,
  `forward[CAP_LEN,IMG_LEN]`). Probe instantiates `[16,16,16]` and runs.
- No var named `ref` (only "no ref" in comments).
- One shared `ShardedSafeTensors.open(transformer_dir)` (`1441`) reused across every
  block load. (`BooguEmbedders.load` opens its OWN `st` at `122` — see FRAGILE-1.)
- No reimplemented op: linear/rms_norm/layer_norm_no_affine/silu/swiglu/tanh_op/
  mul/add/add_scalar/slice/concat/reshape/rope_interleaved/repeat_kv_f32/sdpa_nomask.

## dtype — VERIFIED CLEAN

Probe feeds latent BF16, instruction_feats BF16, timestep F32 (`probe:63-75`).
`time_caption_embed` pre-scales timestep ×1000 (F32) before the cos-first sinusoid;
sinusoid materialized in the MLP weight dtype (bf16) == reference `timestep_proj.to(dtype)`
(`block_lumina2.py:216`). Mathematically identical to `Timesteps(scale=1000,
downscale_freq_shift=0)`. caption path RMSNorm(eps 1e-5)→Linear. norm_out LayerNorm
eps 1e-6 (not 1e-5) — correct per `transformer_boogu.py:937-944`.

---

## Non-blocking observations

- **FRAGILE-1** (`boogu_dit.mojo:122` vs `1441`): `BooguEmbedders.load` opens its own
  `ShardedSafeTensors` while `BooguDiT.load` opens a separate shared one for the
  blocks. Not a correctness bug (both read the same files; embedders use disjoint
  keys), but it means the "one shared ShardedSafeTensors" intent is only partially
  honored — the embedders open + parse the index a second time. Minimal fix: add a
  `BooguEmbedders.load_from(st, ctx)` taking the already-open handle and call it from
  `BooguDiT.load`. STYLE/efficiency only.

- **STYLE-2** (`boogu_dit.mojo:416-417`): for `modulation=False`, the unused
  `n1_lin_w`/`n1_lin_b` fields are filled with clones of the plain `norm1.weight`
  ([3360] gamma) purely to satisfy the struct shape. They are never read in the
  `else` branch (`528-537`). Harmless (small wasted [3360] alloc per context block),
  but a reader could mistake them for AdaLN weights. A comment already flags this.

- **NOTE-3**: parity correctness (cos 0.9998966 vs torch oracle) is owned by the
  orchestrator's numeric gate, not this probe (probe only proves compile+execute).
  The cos<1.0 gap is consistent with bf16 accumulation across 2+2+8+32=44 blocks and
  the bf16 sinusoid/qk-norm/rope path; nothing in the wiring explains a *systematic*
  error beyond bf16 rounding. No wiring bug found that the gate would tolerate.

BLOCKERS: 0 — clean.
