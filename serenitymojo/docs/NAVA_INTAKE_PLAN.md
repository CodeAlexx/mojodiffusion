# NAVA (baidu/NAVA) → pure-Mojo — INTAKE + PLAN

Audio-video generation model (Baidu). Pure Mojo + MAX, GPU-only, inference.
Same discipline as the Ideogram-4 port (oracle-first, chunked, agent triads,
per-chunk parity cos≥0.999, resident fp8 + cuBLAS, measure-don't-assert).
Previously SKIPPED ("no weights"); now porting for real. NOT built yet — this is
the intake + plan.

## Identity (from HF config.json + ports/nava reference)
- **MMDiT joint audio+video DiT** (`WanAVModel`, model_type ti2v): dim 3072, 24
  heads (head_dim 128), ffn 14336, **30 layers = 10 double-stream + 20
  single-stream** (FLUX-style), patch [1,2,2], vid_in/out 48, audio_in/out 128,
  text_len 512, qk_norm, cross_attn_norm, eps 1e-6, temporal_rope_scaling 0.24.
- **RoPE**: 3D for video (`rope_apply_3d`) + 1D for audio (`rope_apply_1d`), joint
  (`rope_apply_joint`); video+audio tokens concatenated and attended together
  with per-modality q/k/v + RMSNorm (`WanDoubleStreamSelfAttention`).
- **Weights fp8**: `NAVA_fp8.safetensors` (6.93 GB) — reuse the Ideogram fp8 path.
- **Video VAE** = Wan2.2-TI2V-5B (vid latent 48) — LOCAL + already ported.
- **Audio VAE** = nava_audio (audio latent 128) — NEW.
- **Text** = Qwen3-1.7B per nava.yaml (NOT local) and/or umt5-xxl (local, bundled
  with Wan2.2) via `FusionModel` — TEXT-ENCODER IDENTITY UNRESOLVED (read
  pipeline_nava.py). TTS: `spk_embed`/`spk_pos` (SeedTTS speaker conditioning).
- **Sampler**: UniPC (`scheduler_unipc`), flow-matching, **aligned/asymmetric CFG**
  (video_guidance 3.0, audio_guidance 2.0, align_3d_cfg) + i2v mode.

## Reference (authoritative)
- Python: `/home/alex/EriDiffusion/inference-flame/ports/nava/nava_src/`
  (`model_nava.py` wrapper, `pipeline_nava.py` AudioVideoPipeline,
  `models/nava/modules/{model_mm.py [1744L, the MMDiT], attention.py, t5.py,
  fusion.py}`, configs `configs/model/dit/NAVA_6B.json`).
- Rust port (arch ref): `inference-flame/src/models/{nava_av.rs, nava_blocks.rs
  [1104L], nava_loader.rs}`, `src/vae/nava_audio_wrap.rs`,
  `src/sampling/nava_sampling.rs`, `src/bin/nava_generate_av.rs`.
- Oracle runtime: `/home/alex/serenityflow-v2/.venv` (torch 2.10+cu128, tf 4.57).

## serenitymojo reuse (scan — HIGH)
| NAVA need | reuse |
|---|---|
| double/single-stream blocks, qk-norm, modulation | `models/dit/klein_dit.mojo` (FLUX double+single — closest sibling) |
| fp8 dequant + resident cache + cuBLAS | `ops/fp8.mojo`, Ideogram `ideogram4_resident` pattern, `ops/fp8_gemm` |
| 3D rope (video) | `ops/rope_tables.build_multiaxis_rope_tables` + `ops/rope` (interleaved) |
| video patch [1,2,2] | `ops/patchify3d.mojo` |
| long-seq attention | `ops/attention.sdpa_tiled`/`sdpa_nomask_tiled` (resolved 2026-06-03) |
| **video VAE (Wan2.2)** | `models/vae/wan22_decoder.mojo` (+`wan22_vae_encoder`) — EXACT |
| text encoder (Qwen3-1.7B) | `models/text_encoder/qwen3_encoder` (+ a 1.7B config) |
| umt5-xxl (if used) | `models/text_encoder/t5_encoder` (T5 family) |
| RMSNorm/LayerNorm, tensor algebra, timestep embed | `ops/{norm,tensor_algebra,embeddings}` |
| video frame PNG + mp4 mux | `image/png`, `components/artifacts` |
| schedule/CFG glue | `sampling/*` (flow-match base; UniPC new) |

## NEW (must build)
- Joint AV MMDiT block (double + single): video+audio fused attention, per-modality
  q/k/v + RMSNorm, 3D+1D joint rope, text cross-attn, adaLN modulation.
- Audio VAE decode (nava_audio).
- FusionModel + spk_embed/spk_pos (TTS speaker conditioning).
- NAVA UniPC sampler + aligned-3D asymmetric CFG (joint video+audio denoise).
- Joint sequence packing (video tokens ++ audio tokens) + 1D audio path.

## Chunked plan (oracle-first; builder→skeptic→bugfix triads; gate cos≥0.999)
0. **Oracle**: run `ports/nava/nava_src` on the fp8 weights (serenityflow venv) →
   dump fixtures: text features, vid+audio latents, per-block I/O, rope tables,
   velocity (vid+audio), decoded video frames + audio. (next_gate)
1. fp8 loader (REUSE) — dequant one NAVA Linear vs torch. **PASS cos=1.0** (nava_chunk1_fp8_probe.mojo; key backbone.double_blocks.0.cross_attn.k.weight [3072,3072]).
2. rope (DECODED from set_rope_params/rope_apply_3d/1d, build next):
   - video 3D: build_multiaxis_rope_tables(pos[f,h,w], axes_half=[22,21,21], theta=10000) -> rope_interleaved (full c=64, direct reuse). interleaved-complex (view_as_complex).
   - audio 1D: build_multiaxis_rope_tables(pos*0.24, axes_half=[22], theta=10000) -> partial rope on first 22 complex dims + passthrough (audio rotates 22 of 64). 0.24 = temporal_rope_scaling_factor folded into positions.
   - joint: 3D on video tokens (320) ++ 1D on audio tokens (34). REUSE ops/rope_tables + ops/rope.rope_interleaved (built for nava video).
   - **PASS**: video 3D rope cos=0.99999855 (nava_chunk2_rope_probe.mojo: axes [44,42,42] θ1e4), audio 1D rope cos=0.99999946 (nava_chunk2_audio_rope_probe.mojo: axes [44] θ1e4, pos*0.24, rotate first 44 dims + passthrough 84). Ref nava_fx_chunk2_rope.safetensors (CPU dump). Full reuse build_multiaxis_rope_tables + rope_interleaved.
3. video patchify [1,2,2] + audio patch — vs ref.
4. timestep embed + modulation — vs ref.
5. DOUBLE-stream joint block (WanDoubleStreamAttentionBlock.forward, model_mm.py:829) — DECODED, build next.
   - no_split_norm_ffn=TRUE -> norm1/norm2/norm3/ffn SHARED vid<->audio; only modulation/modulation_audio separate (6-way: shift/scale/gate × msa,mlp).
   - adaLN: x*(1+scale)+shift; norm1/norm2 LayerNorm affine=FALSE, norm3 affine=TRUE (cross_attn_norm).
   - self_attn JOINT (b=1 gather/scatter = no-op): qkv_fn(vid)+qkv_fn_audio(aud) w/ qk-RMSNorm -> cat to 354 -> rope_apply_joint (3D vid + 1D aud, chunk2 ✓) -> SDPA full -> split -> o(vid)+o_audio(aud).
   - cross_attn = WanT2VDoubleStreamCrossAttention (text k/v, separate vid/aud q) [READ NEXT].
   - ffn = Linear(3072->14336)->GELU(tanh)->Linear(14336->3072), gated by mlp modulation.
   - x = cat([x_vid(320), x_audio(34)]) = double0.out (1,354,3072).
   - NEEDS oracle fixtures: block0 input x, e_vid, e_audio, context(text k/v), freqs. NOT YET BUILT/GATED (triad effort).
6. one SINGLE-stream block — vs ref.
7. full 30-layer DiT → (velocity_vid, velocity_audio) — cos≥0.999.
8. text encoder (Qwen3-1.7B / umt5 + Fusion) — taps vs ref.
9. video VAE decode (Wan2.2 REUSE) + audio VAE decode (NEW) — pixel/audio parity.
10. UniPC + aligned CFG → end-to-end video+audio — final-latent cos + video PSNR + audio.
Perf: resident fp8 + cuBLAS + hoisted masks/transfers (Ideogram lessons).

## Blockers / open
- RESOLVED: audio VAE = **LTX VAE** (`init_ltx_vae` from `params/`) → reuse `models/vae/ltx2_audio_vae.mojo`; sampler = FlowUniPC → `sampling/unipc.mojo`. Weights: `NAVA_fp8.safetensors` (6.93 GB) + `params/` (audio VAE) downloading.
- RESOLVED: text encoder = **umt5-xxl** (T5 `t5_encoder.mojo`; weights LOCAL in Wan2.2 dir `models_t5_umt5-xxl-enc-bf16.pth` + `google/umt5-xxl`). NOT Qwen3-1.7B.
- Oracle: verify `nava_src` deps import + run in serenityflow venv (einops + the
  nava modules); confirm the reference produces video+audio end-to-end first.
- Audio path (audio VAE + TTS spk + 1D DiT) is the genuinely-new, least-reusable
  part — highest risk.
- Scope: ~3× Ideogram (two modalities, two VAEs, TTS). Multi-session.

## Weights (ALL local + byte-verified 2026-06-07)
- DiT: `NAVA_fp8.safetensors` 6.93GB, MATCH, 1432 tensors (1052 BF16 + 380 F8_E4M3, same fp8 scheme as Ideogram). Keys `backbone.{double,single}_blocks.*` with cross_attn `k`/`k_audio` (separate video+audio proj).
- Video VAE: `Wan2.2-TI2V-5B/Wan2.2_VAE.pth` (2.8GB). Text: `Wan2.2.../models_t5_umt5-xxl-enc-bf16.pth` + `google/umt5-xxl`. Audio VAE: `params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors` (LTX-2.3).
- Reference imports OK in serenityflow venv (WanAVModel). Entry: `inference_nava.py --config nava.yaml --ckpt NAVA_fp8.safetensors --data_file example_prompts.jsonl`.

## ORACLE BLOCKER (found 2026-06-07) + fix
Stock `load_fusion_checkpoint` does `torch.load(ckpt)["state_dict"]` (expects the 25GB pickle .ckpt, strict bf16 load) — CANNOT read the fp8 safetensors. Fix (no 25GB dl): custom oracle loader = `load_file(fp8)` → dequant `F8_E4M3 * weight_scale -> bf16` (Ideogram scheme; also makes the oracle run the SAME weights Mojo will) → drop `_scale` keys → `load_state_dict`. nava.yaml rewire: ckpt_dir=NAVA dir, audio_vae_ckpt_dir=params, run from ports/nava.

## ORACLE UP (2026-06-07) — stage 0a+0b PASS (measured)
Oracle script: `EriDiffusion/inference-flame/ports/nava/nava_oracle.py` (serenityflow venv).
- 0a: fp8-dequant load (380 fp8 -> 1052 keys) into WanAVModel(6.297B) -> load_state_dict **missing=0 unexpected=0**.
- 0b: umt5 encode (text 42x4096) -> one `predict_eps` -> **velocity vid (1280,48)** (5x16x16 latent, 30-layer MMDiT). Fixtures `serenitymojo/models/dit/parity/nava_fx_stage0.safetensors`: in_lat_vid, in_text, in_t, double0.out (1,320,3072), single0.out (1,320,3072), vel_vid.
- 24GB choreography (REQUIRED): build bf16 (`set_default_dtype`), umt5 CPU-offload, backbone->GPU AFTER encode (assign=True load leaves it CPU), conv dtype-safe wrap, **flash_attn->torch SDPA** monkeypatch (apples-to-apples w/ Mojo SDPA).
- Patch geometry: video [1,2,2] -> 320 tokens (5*8*8) @ dim 3072; audio 34 tokens. **JOINT block seq = 354 = 320 vid + 34 audio concatenated** (double0.out/single0.out (1,354,3072)). vel_vid (1280,48), vel_aud (34,128). Full AV fixtures in nava_fx_stage0.safetensors.

## next_gate
Write the oracle (fp8-dequant loader + minimal `sample()` 1-clip/few-step run from
ports/nava) → dump fixtures (text feats, latents, rope, per-block I/O, per-step
velocities vid+audio, final latents, decoded video+audio). Then chunk 1. No Mojo until oracle up.
