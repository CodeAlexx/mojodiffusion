# Skeptic Findings: Qwen-Image any-prompt support

- Timestamp: 2026-05-28
- Scope: Verify builder's changes to `forward_cfg` / `forward_cfg_mixed_text`
  and the 512-multistep pipeline support arbitrary prompts via runtime padding
  mask without recompiling.
- Files audited:
  - `/home/alex/mojodiffusion/serenitymojo/models/dit/qwenimage_dit.mojo`
  - `/home/alex/mojodiffusion/serenitymojo/pipeline/qwenimage_pipeline_512_multistep.mojo`
  - `/home/alex/mojodiffusion/serenitymojo/pipeline/qwenimage_pipeline_smoke.mojo`
  - `/home/alex/mojodiffusion/serenitymojo/ops/attention.mojo` (sdpa mask
    contract)
  - `/home/alex/mojodiffusion/serenitymojo/ops/tensor_algebra.mojo` (slice
    semantics)
  - `/home/alex/EriDiffusion/inference-flame/src/bin/qwenimage_encode.rs`
    (Rust DROP_IDX cross-check)

## A. Signature consistency

### A1. `forward_cfg` parameter list — **PASS**
`qwenimage_dit.mojo:1083-1096` — signature is exactly:
`mut self, hidden_states, encoder_pos, encoder_neg, timestep, real_txt_len,
frame, h_latent, w_latent, ctx`. `real_txt_len: Int` sits immediately after
`timestep`, matching the documented order.

### A2. `forward_cfg_mixed_text` parameter list — **PASS**
`qwenimage_dit.mojo:1137-1151` — signature is exactly:
`mut self, hidden_states, encoder_pos, encoder_neg, timestep,
real_txt_len_pos, real_txt_len_neg, frame, h_latent, w_latent, ctx`. The two
length ints follow `timestep` in (pos, neg) order.

### A3. Multistep pipeline call site argument order — **PASS**
`qwenimage_pipeline_512_multistep.mojo:215-221` invokes
`forward_cfg_mixed_text[N_IMG, N_TXT_KEPT, S_POS, N_TXT_KEPT, S_NEG]` with
positional args `(xb, caps.pos, caps.neg, sigmas[i], caps.real_pos,
caps.real_neg, FRAME, FH, FW, ctx)`. The order exactly matches A2.

### A4. Smoke pipeline call site — **PASS**
`qwenimage_pipeline_smoke.mojo:184-186` invokes `forward_cfg[N_IMG, N_TXT, S]`
with positional args `(xb, caps.pos, caps.neg, t_sigma, N_TXT, FRAME, FH, FW,
ctx)`. `real_txt_len = N_TXT` is passed in the correct slot (after
`t_sigma`), and as builder claimed makes the padding mask degenerate to a
no-op (loop `range(N_TXT, N_TXT)` is empty).

## B. Mask correctness

### B1. Flat-buffer indexing matches `[1, heads, S, S]` row-major — **PASS**
`qwenimage_dit.mojo:801-829` — outer loop iterates `hd in range(heads)` with
`head_base = hd * S * S`; middle loop `q in range(S)` with
`row_base = head_base + q * S`; inner loop writes `data[row_base + k]`. Final
shape list appended is `[1, heads, S, S]` (lines 824-828). Indexing yields
offset `hd*S*S + q*S + k` for `(hd, q, k)`, which is exactly the row-major
layout for `[1, heads, S, S]` given the leading batch dim of 1 collapses to
a stride multiplier of 1. The `sdpa` contract (`ops/attention.mojo:618-626`)
verifies `mshape == [B, H, S, S]` with B=1.

### B2. Padded text keys masked for ALL queries — **PASS**
Line 820: `for q in range(S)` covers the entire joint sequence (both text
rows `[0, N_TXT)` and image rows `[N_TXT, S)`). Inner key loop on line 822
sets the pad column `[real_txt_len, N_TXT)` to -1e4 for every `q`. Image
queries cannot attend to padded text positions, satisfying the diffusers
convention.

### B3. Non-pad cells remain 0 — **PASS**
Pre-fill loop (lines 812-815) initializes the entire `heads * S * S` buffer
to 0.0. The only writes happen at key positions in `[real_txt_len, N_TXT)`.
Real-text key columns `[0, real_txt_len)` and all image-key columns
`[N_TXT, S)` are untouched (still 0.0) for every query.

### B4. Range validation — **PASS**
Lines 804-811: explicit `raise Error` when `real_txt_len < 0` or
`real_txt_len > N_TXT`, including the offending value in the error string.

### B5. `MASK_FILL` value — **PASS**
Line 816: `var neg_bias = Float32(-1.0e4)`. -1e4 in BF16 rounds to roughly
−9.98e3, which is finite and produces softmax(...) ≈ 0 without NaN risk. Not
`-inf`, not `-1e9` (which underflows to −inf in BF16). Matches the diffusers
additive-bias convention.

## C. Pipeline correctness (512_multistep)

### C1. Comptime constants — **PASS**
`qwenimage_pipeline_512_multistep.mojo:49-52`:
- `PAD_ID = 151643`
- `DROP_IDX = 34`
- `N_TXT_KEPT = 512`
- `N_ENC = N_TXT_KEPT + DROP_IDX` → 546.

### C2. Tokenize length checks — **PASS**
Lines 131-162: `_tokenize_for_encoder` raises only on `real_len <= DROP_IDX`
(too short to have any non-template text) or `real_len > N_ENC` (overflow).
No hardcoded equality check on the kept length. Pads with `PAD_ID` up to
`N_ENC`.

### C3. `_encode_trimmed` returns hidden + real_kept_len — **PASS**
Lines 168-177: encodes the full `N_ENC=546` ids, then
`slice(full, 1, DROP_IDX, N_TXT_KEPT, ctx)` returns a fixed
`[1, N_TXT_KEPT, 3584]` tensor. `real_kept_len = real_len - DROP_IDX` is
carried alongside in the `EncodedCaption` struct (lines 82-86), then
folded into the `QwenCaps` via `into_caps` (lines 87-91).

### C4. `QwenCaps` carries real lengths — **PASS**
Lines 74-79: `QwenCaps { pos, neg, real_pos: Int, real_neg: Int }`. Used in
the denoise call at line 219 (`caps.real_pos, caps.real_neg`).

### C5. Comptime sequence sizes — **PASS**
Lines 57-58: `S_POS = N_IMG + N_TXT_KEPT`, `S_NEG = N_IMG + N_TXT_KEPT`.
Both equal `N_IMG + 512`. With `N_IMG = (64/2)*(64/2) = 1024`, both come out
to 1536.

## D. No regressions

### D1. All callers updated — **PASS**
`grep -rn "forward_cfg\\b\\|forward_cfg_mixed_text\\b" serenitymojo/` returns
exactly four hits: two definitions in `qwenimage_dit.mojo` (lines 1083, 1137)
and two call sites (`qwenimage_pipeline_smoke.mojo:184` and
`qwenimage_pipeline_512_multistep.mojo:215`). Both call sites pass the new
`real_txt_len*` argument(s). No third caller forgotten.

### D2. No leftover 27/14/64 hardcoded constants — **PASS**
`grep -nE "N_TXT_POS *= *27|N_TXT_NEG *= *14|N_ENC *= *64"` against
`qwenimage_pipeline_512_multistep.mojo` matches only the comment on line 9
that documents the historical hardcoding. No active comptime constants
remain.

### D3. `_zeros_mask` still defined and used elsewhere — **PASS**
`qwenimage_dit.mojo:780` still defines `_zeros_mask`. Still used by
`forward` (line 903), `forward_step` (line 1069), and `forward_edit_cfg`
(line 1240). The builder only replaced it in `forward_cfg` and
`forward_cfg_mixed_text`.

### D4. Both pipelines build cleanly — **PASS**
- `pixi run mojo build -I . -Xlinker -lm
  serenitymojo/pipeline/qwenimage_pipeline_512_multistep.mojo
  -o /tmp/qwenimage_pipeline_512_multistep` → `multistep_exit=0`
- `pixi run mojo build -I . -Xlinker -lm
  serenitymojo/pipeline/qwenimage_pipeline_smoke.mojo
  -o /tmp/qwenimage_pipeline_smoke` → `smoke_exit=0`
- Build logs in `/tmp/q_multistep_build.log`, `/tmp/q_smoke_build.log`.

## E. Cross-check vs Rust

### E1. Rust `DROP_IDX = 34` matches — **PASS**
`/home/alex/EriDiffusion/inference-flame/src/bin/qwenimage_encode.rs:44`
`const DROP_IDX: usize = 34;`. Line 112: `hidden.narrow(1, DROP_IDX,
kept_len)`. Mojo's `slice(full, 1, DROP_IDX, N_TXT_KEPT, ctx)` mirrors this
operation; the only difference is Mojo holds the kept-length comptime-fixed
at 512 and uses an attention mask to zero out pad positions, while Rust
narrows to the variable `kept_len`.

### E2. Mojo mask behaviorally equivalent to Rust trim — **PASS**
Mojo: keep `[DROP_IDX, DROP_IDX + N_TXT_KEPT)` slice (real text at
`[0, real_kept_len)`, pad-token hidden states at
`[real_kept_len, N_TXT_KEPT)`), then mask the pad-column range
`[real_kept_len, N_TXT_KEPT)` to -1e4 for all query rows. Softmax over those
columns ≈ 0, so the effective attention budget matches Rust's variable-length
narrow. Mask blocks BOTH the text rows attending to pad keys AND the image
rows attending to pad keys (B2), which is the symmetric behavior needed.
Note: text-query rows in `[real_kept_len, N_TXT)` are still computed (their
own outputs are garbage) but they are discarded — `_finish` slices out only
the image-portion predictions (verified by inspection of
`forward_cfg_mixed_text`'s call to `self._finish[N_IMG](img_*, temb, ctx)`
which uses `img_*` not `txt_*`).

## Bugfix Worklist

(none — all 18 checks PASS)

The builder's change is clean. No FAILs and no WARNs. The runtime padding
mask correctly generalizes the qwen-image DiT to any prompt up to 512
non-template tokens without recompilation, and the multistep pipeline plumbs
real lengths through to the mask builder. Smoke pipeline correctly
degenerates to a zero-fill mask via `real_txt_len = N_TXT`. Both binaries
build clean.
