# SKEPTIC findings — PiD PixelDiT MMDiTBlockT2I (`pixeldit_block`)

Date: 2026-05-29
Auditor: skeptic subagent
Subject: `serenitymojo/models/pid/pixeldit_block.mojo` (`mmdit_block_forward` +
`_joint_attention`), gated by `serenitymojo/models/pid/pixeldit_block_smoke.mojo`
against `serenitymojo/models/pid/parity/gen_pixeldit_block_reference.py`.
Upstream: `/tmp/PiD_repo/pid/_src/networks/pixeldit_official.py` (lines 76-206,
517-682).

## Verdict: CONFIRMED CORRECT — but the gate is WEAK on this tiny config.

The Mojo block is numerically faithful to the PyTorch reference (both output
streams cos=1.0, max_abs=1.19e-07 = F32 epsilon). Every architectural claim in
the builder report checks out against repo source. However, the chosen
tiny-config gate has **weak discriminating power on the attention sub-path**
because the residual is the dominant term in the output cosine. Fail-closed was
demonstrated decisively (reference corruption → cos=-0.39 FAIL; proj_x ×50 →
cos=0.9985 FAIL), but several non-trivial weight/order mutations were masked by
residual dominance. See "Gate-strength caveat" below.

## Source audit (Mojo vs pixeldit_official.py) — all MATCH

1. **`apply_adaln` = `x*(1+scale)+shift`** (repo line 77). Mojo `modulate`
   (`ops/elementwise.mojo:96`) = `(1+scale)*x+shift`. ✅
2. **Chunk-6 order** `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
   gate_mlp` (repo lines 665-670). Mojo `_chunk6_vec` idx 0..5 in that exact
   order, separate img/txt AdaLN Linears C→6C with bias. ✅
3. **Separate `qkv_x`/`qkv_y`** (repo 533-534, no bias) — the PiD-specific
   feature. Mojo uses `w.qkv_x_w`/`w.qkv_y_w` separately, bias=None. ✅
4. **Per-head QK-RMSNorm over head_dim** with learnable weight (repo 537-540,
   `RMSNorm(head_dim)`, eps default 1e-6). Mojo `_qk_norm_per_head` reshapes
   `[1,N,C]→[N*H,hd]` (head-major contiguous, correct) and runs `rms_norm` with
   eps 1e-6 over the last dim. ✅ RMSNorm math (repo 117-122: F32, mean of
   squares, rsqrt, weight*) matches `ops/norm.mojo`.
5. **Image RoPE = interleaved complex** (repo `apply_rotary_emb` lines 196-206:
   `view_as_complex(x.reshape(...,-1,2)) * freqs_cis`). Mojo `rope_interleaved`
   kernel (`ops/rope.mojo:49-54`): `(x0*cos - x1*sin, x0*sin + x1*cos)` — this
   is exactly complex `(x0+i·x1)·(cos+i·sin)`. ✅ The cos/sin tables are
   `pos_img.real`/`pos_img.imag` (gen script lines 247-248), the correct
   per-pair real/imag of `freqs_cis`, in the interleaved x/y axis order produced
   by `precompute_freqs_cis_2d_ntk` (repo 191: `cat([x_cis, y_cis], -1)`). ✅
6. **RoPE head-broadcast**: repo broadcasts `freqs_cis[None,:,None,:]` (every
   head shares per-token freqs). Mojo `_broadcast_rope_to_heads` tiles
   `[N,half]→[N*H,half]` so row `n*H+h` = `cos[n,*]` — head-independent,
   correct. ✅
7. **text RoPE skipped** when `pos_txt=None` (repo 595-596). Mojo passes qy/ky
   through unrotated. ✅ Correct for the patch-stream backbone default; the
   `pos_txt` 1D-RoPE path is a PidNet-level wiring concern (out of scope).
8. **Joint sequence = `cat([text, image])`** along token axis (repo 608-610:
   `cat([qy, qx])`); split `out_y=[:Ny]`, `out_x=[Ny:]` (repo 614-615). Mojo
   `concat(1, ..., qy_bshd, qx_bshd)` then `slice(.,1,0,Ny)`/`slice(.,1,Ny,Nx)`.
   ✅ Order and split offsets match.
9. **Per-stream output proj WITH bias** (repo 543-544 `nn.Linear(dim,dim)`
   defaults bias=True). Mojo `proj_x`/`proj_y` pass `proj_*_b`. ✅
10. **SwiGLU FeedForward** `w2(silu(w1(x))*w3(x))`, all bias-free, inner dim
    `int(2*hidden/3)` (repo 125-135). Mojo `_swiglu_ffn` = `w2(swiglu(w1,w3))`,
    `swiglu`=`silu(g)*u` (`ops/activations.mojo:312`); fixture computes
    FF_HIDDEN with the same `int(2*int(H*4)/3)` reduction. ✅
11. **Residual gates**: `x = x + gate_msa*attn`, `x = x + gate_mlp*mlp` (repo
    676,680). Mojo `residual_gate(x, gate, y)=x+gate*y`
    (`ops/elementwise.mojo:239`). ✅

No deviations found. Dropout (attn_drop / proj_drop) is identity in eval and
correctly omitted.

## Gate re-run (clean) — REPRODUCED

`pixi run mojo run -I . serenitymojo/models/pid/pixeldit_block_smoke.mojo`
(RTX 3090 Ti, real GPU, ~20GB free):
- x_out (image stream): cos = 1.0, max_abs = 1.19e-07, n=288 — PASS
- y_out (text stream):  cos = 1.0, max_abs = 1.19e-07, n=192 — PASS

## Fail-closed verification (mutate → confirm fail)

| # | Mutation (reverted after) | x_out cos | Result |
|---|---------------------------|-----------|--------|
| 1 | `qkv_x_w[0] += 0.5` | 0.99999999 | PASS (masked) |
| 2 | `qkv_x_w *= 2` (all elems) | 0.9999996 | PASS (masked) |
| 3 | `qkv_x_w *= 10` | 0.9999705 | PASS (masked) |
| 4 | swap joint concat to `[image,text]` | 0.99999996 | PASS (masked) |
| 5 | `proj_x_w *= 50` | **0.9985** | **FAIL** ✅ (non-zero exit, raised) |
| 6 | corrupt `out_x_ref` (`-x+3`) | **-0.39** | **FAIL** ✅ (non-zero exit, raised) |

Tests 5 and 6 prove the gate fails closed on a genuine perturbation of the
image-stream computed path and on reference divergence: cos drops below 0.999,
the smoke prints `SOME GATES FAILED` and raises, exit non-zero. After reverting
all mutations the gate returns to cos=1.0 PASS, and `git diff --stat` on both
.mojo files is empty (restored).

## Gate-strength caveat (IMPORTANT for downstream wiring)

Tests 1-4 reveal the tiny gate is **insensitive to attention-path errors** that
would matter at scale:
- The output is `x_out = x + gate·attn + gate·mlp`. With tiny random inputs the
  residual `x` dominates, so even a 10× scaling of `qkv_x` (test 3) or doubling
  the QKV projection only nudges cosine in the 7th decimal. `max_abs` does track
  it (1.2e-7 → 3.3e-3 → 3.0e-2), so the path IS live — but the **cos≥0.999 bar
  is too loose for this config** to catch attention-internal regressions.
- Test 4 (swapped joint concat order) stayed PASS because full unmasked SDPA is
  permutation-equivariant over K/V: per-query outputs are independent of token
  order; only the split offset distinguishes streams, and with Nx=6≈Ny=4 +
  residual dominance the crossed split barely moved cosine. **A real
  text/image-stream swap could slip past this gate.**

Recommendation (not blocking — block is correct): when this block is integrated
into PidNet, the integration-level gate should either (a) compare the attention
sub-output directly (pre-residual), or (b) use a config with asymmetric Nx≫Ny
and larger weight scales so concat-order/stream-routing errors are visible in
cosine, not just max_abs. The per-block numeric correctness is not in doubt; the
risk is that a *future* edit to ordering/routing would not be caught by this
specific smoke.

## Files (all absolute)
- Module: `/home/alex/mojodiffusion/serenitymojo/models/pid/pixeldit_block.mojo`
- Smoke:  `/home/alex/mojodiffusion/serenitymojo/models/pid/pixeldit_block_smoke.mojo`
- Oracle: `/home/alex/mojodiffusion/serenitymojo/models/pid/parity/gen_pixeldit_block_reference.py`
- Fixture:`/home/alex/mojodiffusion/serenitymojo/models/pid/parity/pixeldit_block_ref_data.mojo`
- Repo src:`/tmp/PiD_repo/pid/_src/networks/pixeldit_official.py:76-206,517-682`
