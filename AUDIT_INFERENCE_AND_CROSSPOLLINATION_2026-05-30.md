# Audit — Mojo Inference Readiness + Training Cross-Pollination Inventory

**Date:** 2026-05-30 · **Package:** serenitymojo (`/home/alex/mojodiffusion`) ·
**Scope:** (A) inference-pipeline readiness, (B) which inference forward ops the
double-stream training backward can reuse / mirror.
**Method:** READ-ONLY. No Mojo compile (builder holds the compile lock). All
cos/perf numbers are MEASURED-by-prior-lead (carried from handoffs, not re-run).
Every claim cites a `file:line` read this session.

Authoritative source for Job A: `serenitymojo/docs/PORT_STATUS_2026-05-29.md`
(supersedes the 2026-05-28 reality tables). Job B is read directly from
`models/dit/klein_dit.mojo` + the `ops/` forward/backward source.

---

## JOB A — INFERENCE READINESS

### A.1 Status table (image models)

Per `PORT_STATUS_2026-05-29.md` the gate doctrine is hard: "working" = a
**visually coherent generated artifact viewed directly**, never compile/finite.
All rows below cite that doc unless noted.

| Model | Inference state | Entry / gate | Notes |
|---|---|---|---|
| **Z-Image (base)** | **WORKING end-to-end, native 1024²** | `pipeline/zimage_pipeline.mojo` (12 KB) | Denoise sign-convention bug RESOLVED (post-CFG negate) — `STATUS_ZIMAGE_DENOISE_DIVERGENCE.md:5`. Final-latent std 0.775 (was 3.4). 9 components each cos≥0.999. Open cleanup: hardcoded sigma table has a duplicate terminal 0.0 (`STATUS…:66,84`). |
| **Klein9B (FLUX.2)** | **✅ coherent** | `pipeline/klein9b_pipeline_multistep_smoke.mojo`, `…_1024_smoke.mojo` | honeycomb 1024; turbo-loader byte-exact (`PORT_STATUS:Image table`). Block-streamed offload (all-resident 1024 OOMs; one-block-at-a-time clears it — `STATUS…:159`). |
| **Qwen-Image 512 & 1024** | **✅ coherent** | `pipeline/qwenimage_pipeline_512_multistep.mojo`, `…_1024_multistep.mojo` | unblocked by adding sdpa `(546,28,128)` comptime case (commit 840eda8). |
| **ZImage L2P** | **✅ coherent** | `pipeline/zimage_l2p_pipeline_512_multistep.mojo` (26 KB) | 30-layer pixel-space (no VAE); matches Rust 512 oracle. H=30, head_dim=128. |
| **Anima (MiniTrainDIT)** | **✅ coherent** | `pipeline/anima_pipeline_1024_multistep.mojo` | 28-block; latent cos 0.9948 vs Rust. |
| **ERNIE-Image** | **✅ coherent** | `pipeline/ernie_pipeline_1024_multistep.mojo` | 36-layer streamed + Mistral3B sidecar + Klein VAE. `[sin|cos]` t-embed fix (sibling op `timestep_embedding_sin_first`). |
| **SD3.5 Medium / Large** | **✅ coherent** | `pipeline/sd3_medium_pipeline_1024_multistep.mojo`, `sd3_large_pipeline_1024_multistep.mojo` | 24-block / 38-block joint MMDiT + triple-encoder sidecar. |
| **Chroma** | **✅ coherent** | `pipeline/chroma_pipeline_1024_multistep.mojo` | 19+38 block + T5 sidecar + FLUX VAE (early "blob" was a VAE-dtype bug). |
| **Microsoft Lens** | **partial** — DiT math parity-verified (cos 0.99996); real-prompt image BLOCKED | `pipeline/lens_pipeline_1024_multistep.mojo` | GPT-OSS encoder OOMs at real-prompt; MXFP4 dequant kernel itself GPU-validated 7/7. |
| **SenseNova-U1** | **✅ coherent** (512 ginger-tabby) | `pipeline/sensenova_u1_gen_real.mojo` | RoPE blocker fixed + SYSTEM_MESSAGE conditioning. Pixel-parity vs Rust = follow-on. |
| **Qwen-Image-Edit** | **scaffold/deferred** | `pipeline/qwenimage_edit_*_smoke.mojo` | needs real reference-image encode path. |

### A.2 Video (LTX2)

`PORT_STATUS:LTX2` — first end-to-end **T2V + audio** clip generated in pure
Mojo+MAX (P7 MVP commit e74f187). Full chain: joint-AV 48-block DiT (6 attention
paths) + video VAE (cos 0.99998) + BigVGAN vocoder (cos 0.99996, audible) + audio
VAE (cos 0.99999) + LoRA (rank-384, added-never-fused). **Quality pass IN
PROGRESS** (res_2s 2nd-order sampler + HQ recipe; MVP at 256² was soft).
Open follow-ons: stage-x2/AdaIN upsampler unbuilt; video-velocity 0.9947
deep-chain bf16 reduction-order drift (no visual impact); audio not yet
full-DSP-parity-gated end-to-end.

### A.3 In-progress / parked / excluded

- **PiD** (NVIDIA pixel-diffusion decoder): ops foundation done (`models/pid/`);
  parity + decode phases pending the downloaded checkpoint.
- **Excluded by user:** Helios, Nucleus, Stable Cascade.
- Probe/contract/`*_smoke` files for every model are scaffolds — do NOT read a
  `*_contract_smoke` / `*_preblock_smoke` green as an image.

### A.4 Known-divergence docs to respect

- `STATUS_ZIMAGE_DENOISE_DIVERGENCE.md` — Z-Image post-CFG sign flip
  (RESOLVED, but the sigma-table duplicate-zero cleanup is still open).
- `BUG_sdpa_backward_H30_dq_dk_zero.md` — TRAINING-side bug (see Job B), not an
  inference blocker (the inference SDPA forward is clean, cos 0.9999971).

**Job A bottom line:** 10 image models + 1 video pipeline are visually coherent
end-to-end. Z-Image and Klein9B (the two named minima) both clear the bar. Lens
is the only image model still blocked (GPT-OSS encoder OOM). Everything else
flagged is scaffold-by-design (edit/PiD) or excluded.

---

## JOB B — CROSS-POLLINATION INVENTORY (the build list)

The Klein double-stream block (`klein_dit.mojo:_double_block`, :267) is the
target. Its forward, read line-by-line (`klein_dit.mojo:267-352`), is built from
exactly these ops. For each: the **inference forward** location, whether a
**training backward** already exists, and the note.

### B.1 What `_double_block` actually calls (proof from source)

`klein_dit.mojo` imports (`:17-24`):
`linear`, `rms_norm`/`layer_norm`, `silu`/`swiglu`, `modulate`/`residual_gate`,
`rope_interleaved`, `sdpa_nomask`, `reshape`/`slice`/`concat`. The block body
(`:296-352`) chains: `_modulate_pre` (= `layer_norm` then `modulate`,
`:210-211`) → `linear` (qkv) → `slice`+`reshape` (`_qkv_part`, :213-224) →
`rms_norm` (q/k norm) → `concat` → `rope_interleaved` (`_attn_rope_only`,
:235-236) → `sdpa_nomask[1,S,32,128]` (:237) → `slice`+`reshape` → `linear`
(proj) → `residual_gate` (:339-340) → `_modulate_pre` again → `swiglu` MLP
(`_swiglu_linear`, :256-265: `linear`→`slice`×2→`swiglu`→`linear`) →
`residual_gate` (:350-351) → `concat`.

### B.2 Op table

| Op | Inference forward (file:line) | Training backward exists? | Note |
|---|---|---|---|
| **modulate / AdaLN** `(1+scale)·x+shift` | `ops/elementwise.mojo:93` (`modulate`; kernel `_modulate_kernel_f32:32`, math `(1+sv)*xv+shv` at `:48`) | **NO** | **THE missing arm.** No `modulate_backward` anywhere in `ops/*_backward.mojo` (grep confirmed). Needs: d_x=grad·(1+scale); d_scale=Σ_rows(grad·x); d_shift=Σ_rows(grad). Per-channel scale/shift `[D]`. |
| `_modulate_pre` wrapper (layer_norm→modulate) | `klein_dit.mojo:203-211` | **layer_norm Y, modulate N** | The LN half has a backward (`layer_norm_backward`, `norm_backward.mojo:357`); only the `modulate` tail is missing. Chain them: modulate_bwd → layer_norm_bwd. |
| **residual_gate** `x + gate·y` | `ops/elementwise.mojo:236` (`residual_gate`; kernel `_resgate_kernel_f32:179`) | **YES** | `gate_residual_backward` (`rope_struct_backward.mojo:347`), forward documented `o = x + g*y`, g per-channel `[C]` — **EXACT match** to the forward. Returns `GateResidualGrads{d_x,d_g,d_y}` (:281). PROVEN (`rope_struct_bwd_parity`). Directly reusable. |
| **rope_interleaved** | `ops/rope.mojo:291` (`rope_interleaved`; halfsplit sibling at `:364`) | **YES** | `rope_backward(grad_out,cos,sin,interleaved,ctx)` (`rope_struct_backward.mojo:140`). `interleaved=True` selects `_rope_bwd_interleaved_kernel_f32` (:70); header `:153` states "interleaved=True = FLUX/Klein pairing (2i,2i+1)". **Correct variant for Klein** — NOT a halfsplit shape-sniff trap. (Z-Image L2P would pass `interleaved=False`.) PROVEN. |
| **sdpa_nomask** (math-mode SDPA) | `ops/attention.mojo:643` (`sdpa_nomask[B,S,H,Dh]`; masked `sdpa` at `:589`) | **YES but ⚠ partial** | `sdpa_backward[B,S,H,Dh]` (`attention_backward.mojo`, returns `SdpaGrads{d_q,d_k,d_v}`). **Klein is H=32** (`klein_dit.mojo:46` config + `:237` call `[1,S,32,128]`) → hits the H=32 path that **PASSES** the toy gate. The H=30 d_q/d_k-zero BUG is **Z-Image-only** (`zimage_dit.mojo` H=30). So Klein double-stream backward is NOT blocked by the SDPA bug; Z-Image is. |
| **concat** | `ops/tensor_algebra.mojo:675` (`concat(dim,ctx,*tensors)`) | **YES** | `cat_backward` (`shape_backward.mojo:314`) → `CatGrads2{d_0,d_1}`. Klein concats are 2-input (txt,img) so `CatGrads2` fits. PROVEN. |
| **slice** | `ops/tensor_algebra.mojo:743` (`slice`) | **YES** | `slice_backward` (`shape_backward.mojo:450`) scatters grad into the sliced region. PROVEN. |
| **reshape** | `ops/tensor_algebra.mojo:472` (`reshape`) | **YES** | `reshape_backward` (`shape_backward.mojo:114`). PROVEN. |
| **layer_norm** | `ops/norm.mojo:379` | **YES** | `layer_norm_backward` (`norm_backward.mojo:357`) → `{d_x,d_g,d_b}`. PROVEN (`norm_bwd_parity`). |
| **rms_norm** | `ops/norm.mojo:147` | **YES** | `rms_norm_backward` (`norm_backward.mojo:152`) → `{d_x,d_g}`. Tape-wired (`autograd.mojo:497`). PROVEN. |
| **swiglu** | `ops/activations.mojo:311` | **YES** | `swiglu_backward` (`loss_swiglu_backward.mojo:209`) → `SwigluGrads{d_gate,d_up}`. Tape-wired (`autograd.mojo:509`). PROVEN. Klein splits gate/up via `slice` (`klein_dit.mojo:262-263`) → `slice_backward` partners. |
| **silu** | `ops/activations.mojo:80` | **YES** | `silu_backward` (`activation_backward.mojo:270`). Tape-wired. PROVEN. (Klein MLP uses `swiglu` not raw `silu`, but silu is available.) |
| **linear** | `ops/linear.mojo:145` | **YES** | `linear_backward` (`linalg_backward.mojo:249`) → `{d_x,d_w,d_b}`. Tape-wired (`autograd.mojo:484`). Klein passes `None` bias → use d_b=0 / addbias-free path. PROVEN. |

### B.3 qkv split note

Klein does **NOT** use a fused `qkv_split_permute` — it `slice`s the packed qkv
(`_qkv_part`, `klein_dit.mojo:213-224`: `slice(qkv,2,part*inner,inner)` then
`reshape`). So its backward is `slice_backward` + `reshape_backward`, NOT
`qkv_split_permute_backward`. The fused `qkv_split_permute_backward`
(`rope_struct_backward.mojo:213`) exists and is PROVEN but is for a *different*
(fused-qkv) layout — do not force Klein onto it.

---

## DELIVERABLE — ops needing a NEW backward arm

**Exactly one** forward op in the Klein double-stream block has no backward:

1. **`modulate_backward`** — forward `ops/elementwise.mojo:93`
   (`out = (1+scale)·x + shift`, kernel math at `:48`).
   - Grads: `d_x = grad·(1+scale)`; `d_scale = Σ_rows(grad·x)` `[D]`;
     `d_shift = Σ_rows(grad)` `[D]`.
   - Write in `ops/elementwise_backward.mojo` (new file) or append to a sibling
     backward; gate with `ops/parity/modulate_bwd_parity.mojo` (cos≥0.999 vs
     torch host ref). Mirror the per-channel column-sum pattern already in
     `linalg_backward.mojo:_colsum_kernel:96` and the F32/uint8-bitcast
     scaffolding of `gate_residual_backward`.
   - It is a per-channel affine — trivial relative to the SDPA/norm arms.

Everything else the Klein double-stream block needs **already has a PROVEN
backward** and is **directly reusable** with no new code:
`residual_gate→gate_residual_backward`, `rope_interleaved→rope_backward(interleaved=True)`,
`concat→cat_backward`, `slice→slice_backward`, `reshape→reshape_backward`,
`layer_norm→layer_norm_backward`, `rms_norm→rms_norm_backward`,
`swiglu→swiglu_backward`, `silu→silu_backward`, `linear→linear_backward`.

## Two correctness caveats for the assembler

- **SDPA H is model-specific.** Klein9B is **H=32** → the existing
  `sdpa_backward` PASSES for Klein. The H=30 zero-d_q/d_k bug
  (`BUG_sdpa_backward_H30_dq_dk_zero.md`) blocks **Z-Image** training, not Klein.
  Do not let the Z-Image blocker stall the Klein double-stream build, but DO
  re-gate at H=30 before any Z-Image run.
- **`_modulate_pre` is a 2-op chain** (layer_norm then modulate). The composed
  backward is `modulate_backward` → `layer_norm_backward`. Only the first link is
  new.

---

**Report path:** `/home/alex/mojodiffusion/AUDIT_INFERENCE_AND_CROSSPOLLINATION_2026-05-30.md`
