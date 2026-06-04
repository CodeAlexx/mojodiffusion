# SKEPTIC FINDINGS — SDXL/LDM AutoencoderKL ENCODER port

Component: `sdxl_ldm_vae_encoder`
Files: `serenitymojo/models/vae/ldm_encoder.mojo`, `serenitymojo/models/vae/ldm_encoder_probe.mojo`
Reference: `inference-flame/src/vae/ldm_encoder.rs` (728 L, read line-by-line)
Oracle: diffusers `AutoencoderKL.encode(...).latent_dist.mode()` (UNSCALED)
Weights: `/home/alex/madebyollin_sdxl-vae-fp16-fix/sdxl_vae.safetensors` (250 tensors, F32 on disk)
Date: 2026-06-03

## §0 RECEIPTS
1. `serenitymojo/MAP.md` — module placement + foundation ops to reuse (conv2d/group_norm/silu/linear/sdpa live under `serenitymojo/ops/`; VAE kit under `models/vae/`).
2. `serenitymojo/docs/SERENITYMOJO_MODULES.md` — exact foundation signatures; verified against actual source: `conv2d[N,Hi,Wi,Cin,Kh,Kw,Cout,sh,sw,ph,pw](x,w,Optional[bias],ctx)` NHWC+RSCF F32-accum; `group_norm(x,w,b,num_groups,eps,ctx)` NHWC, F32-accum, requires x/w/b same dtype; `sdpa[N,S,H,Dh](q,k,v,mask,scale,ctx)`.
3. `ldm_encoder.rs` — GroupNorm eps=1e-6 num_groups=32 (`group_norm_nchw(...,32,...,1e-6)`); ResBlock order norm1→silu→conv1→norm2→silu→conv2 then residual=shortcut(x)|x add (:118-132); downsample = `pad2d_zeros(0,1,0,1)` right+bottom then Conv2d 3x3 stride2 pad0 (:399-423); mid = res0→attn→res1 (:294-298); attn scale = sdpa default `None` = 1/sqrt(C=512) single head (:228-229); quant_conv 8→8 1x1 BEFORE channel split (:692-695); scaling_factor 0.13025 applied ONLY in separate `encode_scaled` z=scale*(z-shift) (:714-722), NOT inside `encode`; `encode` returns mean only (narrow first latent_ch :705) — no logvar clamp / no sampling in the Rust path.
4. `/home/alex/.claude/skills/mojo-syntax/SKILL.md` — relevant traps: `comptime` not `alias`; `def` raises; move-only `Tensor` needs `^` transfer; no var named `ref`. All respected (no `alias`, no `fn`, `^` used on every loaded tensor into the constructor, no `ref` identifier).

## COMPILE HONESTY
Re-ran MYSELF: `pixi run mojo run -I . serenitymojo/models/vae/ldm_encoder_probe.mojo` → **EXIT=0**, genuine GPU forward.
```
ctx ok 0
encoder loaded (SDXL LDM, latent_ch=4, quant_conv present)
moments shape: 1 8 8 8 (expect 1 8 8 8)   moments mean=-10.97 std=12.59 bad=0
mean latent shape: 1 4 8 8                  mean mean=1.117 std=4.836 bad=0
sampled ...                                 sampled mean=1.117 std=4.836 bad=0
ldm_encoder probe OK
```
compiled:true is HONEST. Outputs finite. mean≈sampled is EXPECTED (logvar half is very negative on the synthetic ramp → std≈exp(0.5·−25)≈3.7e-6 → sample collapses to mean); this is correct DiagonalGaussian behavior, not a bug.

## WEIGHT-KEY VERIFICATION (against actual checkpoint header)
On-disk layout confirmed LDM/BFL, NOT diffusers — the builder's claim is TRUE:
- `encoder.conv_in.{w,b}` [128,3,3,3]; `encoder.conv_out.{w,b}` [8,512,3,3]
- `encoder.down.{0..3}.block.{0,1}.{norm1,conv1,norm2,conv2}`; `encoder.down.{0,1,2}.downsample.conv`
- `encoder.down.1.block.0.nin_shortcut` + `encoder.down.2.block.0.nin_shortcut` present (128→256, 256→512); down.0/down.3 have NO shortcut (channels equal). Matches Mojo `has_sc = Cin!=Cout`.
- `encoder.mid.block_{1,2}`, `encoder.mid.attn_1.{norm,q,k,v,proj_out}` (q/k/v/proj_out are Conv2d-1x1 [512,512,1,1]) — squeezed to [512,512] by `_load_attn_proj_ldm`.
- `encoder.norm_out`, top-level `quant_conv.weight` [8,8,1,1] + `quant_conv.bias` [8].
Every key the loader references EXISTS; no silent renames; channel transitions [128,256,512,512] match reference `ch_mult=[1,2,4,4]`. Block library (decoder2d ResnetBlock/AttnBlock + ldm_decoder `_load_resnet_ldm`/`_load_attn_ldm`) reused VERBATIM — consistent with decoder, no edits.

## FIDELITY CHECKS THAT PASS
- GroupNorm 32 / eps 1e-6 (`GN_GROUPS`/`GN_EPS` from decoder2d). ✓
- Resnet norm→silu→conv ×2 + shortcut(x)|x add. ✓ (decoder2d ResnetBlock.forward)
- Downsample asymmetric pad: `_pad_rb_nhwc` adds W+1 (right) then H+1 (bottom) NHWC zeros, then conv 3x3 stride2 pad0. Output H = (IH+1−3)//2+1 = IH/2. ✓ (probe 64→32→16→8)
- Mid res→attn→res. ✓  Attn scale 1/sqrt(512) single head, q/k/v/o = linear(x@wᵀ+b). ✓ (== reference squeeze_1x1 + linear_3d)
- conv_out 512→8 3x3 pad1; quant_conv 8→8 1x1 BEFORE split. ✓
- moments split on NHWC channel dim 3: mean=[:4], logvar=[4:]. ✓
- diag_gaussian_sample: clamp logvar[-30,20], std=exp(0.5·lv), z=mu+std·eps, F32-only. ✓ (matches diffusers DiagonalGaussianDistribution; note diffusers clamps in __init__, mathematically identical for mode/sample)
- Scope: pure Mojo+MAX, GPU compute, io via ShardedSafeTensors, foundation ops CALLED not reimplemented. No Rust/cargo/flame/autograd/Python-runtime leak. The host F32 transpose in `_load_conv_weight_rscf` is one-time load-time weight prep (same pattern as the shipped decoder), not a runtime CPU compute path. ✓
- Mojo correctness: `^` transfer on all loaded tensors; named `Tensor` fields (no `List[Tensor]`); `comptime` not `alias`; `def` raises; no `ref` identifier. ✓

## BLOCKERS
None. The `encode_mean` path is numerically faithful to `ldm_encoder.rs::encode` and to diffusers `AutoencoderKL.encode(...).mode()` (unscaled). Compile is honest.

## FRAGILE
- **[FRAGILE] ldm_encoder.mojo:124-125, 89-92, 349 — latent scaling boundary not exposed; `scale`/`shift` are DEAD fields.**
  `self.scale`/`self.shift` are stored on the struct (set to 0.13025/0.0 for SDXL) but NEVER read by `encode_moments`/`encode_mean`/`encode`. The reference exposes `encode_scaled` (z = scaling_factor·(z − shift_factor), rs:714-722); the Mojo port has no equivalent. The SDXL pipeline MUST multiply the mean latent by 0.13025 before feeding the UNet. As written, a caller gets an unscaled latent and the dead `scale` field invites the silent bug of forgetting the multiply (~7.7× magnitude error in the UNet input). This is fine for the UNSCALED parity gate (which is what the oracle should target) but is a downstream landmine.
  Minimal fix: add `def encode_scaled_mean(self, img, ctx) -> Tensor` that calls `encode_mean` then applies `mul_scalar((z - shift), scale)` using the stored fields; OR delete the dead fields and document that the caller scales. Do NOT bake scaling into `encode_mean` (would break parity vs `mode()`).

- **[FRAGILE] dtype-flow divergence vs reference — Mojo runs F32, Rust runs BF16.**
  `_load_weight`/`from_view` preserve the on-disk F32 dtype; `_load_conv_weight_rscf` re-uploads with `w.dtype()` = F32; the probe input is F32. So the whole encoder runs F32 end-to-end. The reference casts every weight + input to BF16 (rs:566-577). diffusers parity must therefore be run with the oracle in **F32** (`vae.to(torch.float32)`), not bf16, or the cos check will see legitimate dtype-rounding drift (NOT a code bug). `group_norm` additionally HARD-REQUIRES x/weight/bias same dtype (norm.mojo:901) — mixing would raise, so the F32-everywhere choice is internally consistent. Flag so the parity harness picks the matching oracle dtype.

- **[FRAGILE] ldm_encoder.mojo:179-188 — quant_conv dummy path keys on `has_quant` flag, not key presence.**
  Reference decides quant_conv by `w.contains_key("quant_conv.weight")` (rs:640). The Mojo port decides by the constructor `has_quant: Bool` arg (True for SDXL/SD1.5 factories). For the standalone SDXL file this is correct (key IS present), but a model that lacks quant_conv loaded with `has_quant=True` would raise on the missing key rather than degrade. Low risk for the SDXL/SD1.5 factories shipped; would bite a future no-quant model. Minimal fix: gate on `st.has(...)` like the reference, or keep the explicit flag but assert key presence matches.

## STYLE
- **[STYLE] ldm_encoder.mojo:262, 174 — `comptime ZC2 = 2 * Self.LATENT_CH` declared in `load` (:174) is unused** (the real one is recomputed in `encode_moments` :262). Dead local. Drop the one in `load`.
- **[STYLE]** Probe uses an 8×8 latent (64×64 image), so the mid-attention sequence length is 64 — it never exercises the >1024-token tiled-attention branch the reference has (rs:227-242). The Mojo AttnBlock has no tiling (single sdpa over S). At SDXL 1024² (latent 128 → S=16384) this is a single 16384×16384 attention; verify it does not OOM / that sdpa handles it. Not a parity issue at the gate size; raise for the full-res run.

---

{component:"sdxl_ldm_vae_encoder", compiles:true (re-ran ldm_encoder_probe.mojo myself, EXIT=0, finite moments/mean/sampled, weight keys verified against the 250-tensor checkpoint header), blockers:[], fragile:[{where:"ldm_encoder.mojo:124-125,349 scale/shift dead fields", what:"latent scaling boundary (×0.13025) never applied and not exposed; downstream landmine", fix:"add encode_scaled_mean using stored fields, or delete fields + document caller scales; do NOT bake into encode_mean"},{where:"dtype flow", what:"Mojo runs F32 end-to-end, reference runs BF16; group_norm requires uniform dtype", fix:"run diffusers parity oracle in F32 to match"},{where:"ldm_encoder.mojo:179-188", what:"quant_conv presence keyed on has_quant flag not checkpoint key", fix:"gate on st.has(quant_conv.weight) like reference"}], style:["ZC2 dead local in load()", "no tiled attention — verify 16384-token sdpa at 1024² full-res"], verdict:"clean (0 BLOCKERS) — encode_mean is parity-faithful to ldm_encoder.rs::encode and diffusers mode(); compile honest"}
