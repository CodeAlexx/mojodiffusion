# SKEPTIC-2 FINDINGS — weight-loading & checkpoint fidelity (2026-05-26)

Reviewer: second-pass skeptic. Theme: **does every weight key the Mojo requests EXIST in the
ACTUAL on-disk safetensors header?** Round-1 reviewed each model in isolation and MISSED the
SDXL-VAE diffusers-vs-LDM mismatch. This pass read the real headers (8-byte len + JSON) of every
checkpoint each `load()` path touches and diffed requested keys against on-disk keys.

**CODE-ONLY — GPU wedged. `mojo build` to compile-check only; nothing was run.** Compile-honesty
re-runs at the bottom.

Method: for each model I (1) read the `load()` + forward `_w(...)` callsites to enumerate the EXACT
requested key strings, (2) located the file via the smoke/pipeline path constants, (3) dumped the
real header, (4) diffed. For absent checkpoints (Nucleus) I inferred reference keys from the
canonical flame-core Rust (`inference-flame/src/...`) which the port claims to mirror.

---

## SEVERITY SUMMARY

| # | Model / file | Severity | One-line |
|---|---|---|---|
| S1 | Nucleus DiT `nucleus_dit.mojo` | **BLOCKER** | requests BARE keys; on-disk + flame-core use `model.` prefix on EVERY key → total load failure |
| S2 | Qwen-Image VAE `qwenimage_decoder.mojo` | **BLOCKER** (latent) | requests Wan-native keys but docstring says "diffusers"; pipeline points at diffusers `up_blocks` VAE → every key MISSes. Only works against the dedicated pre-remapped `qwen_image_vae.safetensors`, which no pipeline points at. |
| S3 | SDXL VAE `SDXLLdmDecoder` struct (ldm_decoder.mojo:141-295) | FRAGILE | dead diffusers-keyed decoder still present; would no-op against the real LDM `sdxl_vae.safetensors`. Pipeline uses the correct one (`load_sdxl_ldm_decoder`→`LdmVaeDecoder`), so it is a landmine, not a live bug. |
| S4 | Qwen-Image / Qwen2.5-VL / SDXL pipeline path constants | FRAGILE | `QWENIMAGE_DIR`/`VAE_DIR`/`TEXT_ENCODER_DIR` point at non-existent `/home/alex/.serenity/models/qwen-image/...`. Keys are right; the path is a placeholder. |
| S5 | `_load_conv_weight_rscf` docstrings (decoder2d.mojo:78, qwenimage_decoder dtype) | STYLE | docstrings say "re-uploaded as BF16" but code re-uploads `w.dtype()` (F32 for the F32 VAE files). Code correct, doc stale. |

**TOTAL BLOCKERS: 2** (S1 Nucleus, S2 Qwen-Image VAE — both the SDXL-class "requests keys the file
doesn't have" bug, exactly the class the task said to hunt for.)

Round-1 fixes that I re-checked and CONFIRM HELD: SDXL/FLUX `LdmVaeDecoder` (LDM keys, verified vs
both 250-tensor SDXL and 244-tensor FLUX headers); LoRA F1 split→fused Klein QKV
(`_map_klein_split_qkv`, lora.mojo:256-293) and F2 per-module-rank scale (`_module_scale`,
lora.mojo:416-429).

---

## FINDINGS

### S1 — BLOCKER: Nucleus DiT requests bare keys; checkpoint + canonical use `model.` prefix
**file:** `serenitymojo/models/dit/nucleus_dit.mojo:268-285` (`load`, stores on-disk names verbatim,
looks up bare) + every `_w("...")` callsite (e.g. lines 298, 332, 340, 345, 803, 807, 823, 833, and
all per-block `_w(prefix + "attn.to_q.weight")` where `prefix = "transformer_blocks.{i}."`).

**Checkpoint:** ABSENT. `nucleus_gen_smoke.mojo:71` `SNAPSHOT = ".../models--NucleusAI--Nucleus-Image/
snapshots/CHECKPOINT"` is a literal placeholder; no Nucleus weights anywhere on disk
(`find ... -iname '*nucleus*'` → nothing under hub/serenity). Audited by reference-key inference.

**On-disk key shape (from the canonical flame-core ref `inference-flame/src/models/nucleus_dit.rs`,
which the Mojo header line 15-18 claims to mirror):**
- `nucleus_dit.rs:996` — block prefix is **`model.transformer_blocks.{i}.`**
- `nucleus_dit.rs:1023-1045` — top-level reads **`model.img_in.weight`**, **`model.img_in.bias`**,
  **`model.time_text_embed.timestep_embedder.linear_1.weight`**, **`model.time_text_embed.norm.weight`**,
  **`model.txt_norm.weight`**, **`model.norm_out.linear.weight`**, **`model.proj_out.weight`**.
- i.e. the exported Nucleus checkpoint wraps the whole transformer under an outer **`model.`** (the
  diffusers *module* attrs are bare — `self.img_in` etc. at transformer_nucleusmoe_image.py:807 — but
  the saved state_dict is prefixed; `take(map, "model.img_in.weight")` proves it).

**Requested keys (Mojo):** BARE — `img_in.weight`, `time_text_embed.timestep_embedder.linear_1.weight`,
`time_text_embed.norm.weight`, `txt_norm.weight`, `norm_out.linear.weight`, `proj_out.weight`,
`transformer_blocks.{i}.attn.to_q.weight`, … NO `model.`, NO strip (`load()` never touches the name;
grep for `model.`/`strip`/`prefix` in nucleus_dit.mojo → nothing).

**Impact:** every `_w(...)` MISSes → `raise Error("NucleusDiT: missing weight: img_in.weight")` at the
first forward lookup. Total failure, no partial. This is the exact SDXL-class trap, and it is the
"Nucleus round-1 flagged a possible `model.` strip gap" item — **CONFIRMED REAL**.

**Minimal fix:** in `nucleus_dit.mojo:load`, strip a leading `model.` from each name as it's indexed
(or prepend `model.` in every `_w` lookup). Stripping at load is cleaner and matches how the other
verbatim-loaders behave: `if nm.startswith("model."): nm = nm[6:]` before `name_to_idx[nm]=idx`.
Gate it (`startswith`) so a bare-keyed file still works. **Severity: BLOCKER.**

---

### S2 — BLOCKER (latent): Qwen-Image VAE requests Wan-native keys; docstring/pipeline imply diffusers
**file:** `serenitymojo/models/vae/qwenimage_decoder.mojo:139-167` (`load`, "diffusers-format dir" in
the docstring, stores names verbatim, NO remap) + forward `_w("decoder.conv1...")` / `_conv3d_named`
on `decoder.middle.{0,1,2}`, `decoder.upsamples.{N}`, `decoder.head.{0,2}`, `residual.0.gamma`,
`conv2.*`.

**On-disk headers read:**
- `/home/alex/.serenity/models/vaes/qwen_image_vae.safetensors` — **194 tensors, Wan-native keys**
  (`decoder.conv1.weight [384,16,3,3,3]`, `decoder.middle.0.residual.0.gamma`,
  `decoder.upsamples.0.residual.6.weight`, `decoder.head.0.gamma`, `conv2.weight [16,16,1,1,1]`).
  123 Wan-native / 0 diffusers. → the decoder's requested keys MATCH **this** file.
- `models--ai-toolkit--wan2.1-vae/.../diffusion_pytorch_model.safetensors` — **194 tensors,
  DIFFUSERS keys** (`decoder.conv_in`, `decoder.mid_block.resnets.0`, `decoder.up_blocks...`).
  118 diffusers / 0 Wan-native. → every requested key MISSes.
- The HF `Qwen-Image-2512/vae` snapshot ships only `config.json` (weights not downloaded), and the
  diffusers Qwen-Image VAE is the SAME diffusers `up_blocks` layout (= ai-toolkit wan2.1-vae).

**The trap:** the Mojo `load()` docstring says "diffusers-format dir" and the model comment block
(qwenimage_decoder.mojo:53-60) presents a diffusers→Wan **mapping table** — but `load()` does **NOT
remap**. The canonical flame-core `inference-flame/src/vae/qwenimage_decoder.rs:25` loads the
DIFFUSERS file and calls `remap_qwenimage_to_wan21(raw)` to rewrite `up_blocks`→`upsamples` etc.
The Mojo port dropped the remap and silently requires a pre-remapped Wan-native file.

**Why it bites:** the two pipelines that build this decoder point at the WRONG (diffusers) layout:
- `qwenimage_pipeline_smoke.mojo:58,209` → `VAE_DIR = QWENIMAGE_DIR + "/vae"` (diffusers).
- `nucleus_gen_smoke.mojo:75,207` → `VAE_DIR = SNAPSHOT + "/vae"` (diffusers).
Neither points at the dedicated Wan-keyed `qwen_image_vae.safetensors`. Against the path they DO
name, every decoder lookup raises `"VAE: missing weight: decoder.conv1.weight"`. It is "latent"
only because the placeholder dirs don't exist yet, so it errors on path before key — but the moment
the diffusers VAE is present it is a guaranteed full miss.

**Minimal fix (pick one):**
(a) port `remap_qwenimage_to_wan21` into `load()` (mirror qwenimage_decoder.rs:45-…), then point the
   pipelines at the diffusers VAE; OR
(b) keep the no-remap loader but FIX the docstring ("expects Wan-native-keyed file, e.g.
   `qwen_image_vae.safetensors` — NOT the diffusers `up_blocks` layout") AND repoint both
   `VAE_DIR`s at `/home/alex/.serenity/models/vaes/qwen_image_vae.safetensors`.
Option (a) matches the canonical Rust. **Severity: BLOCKER** (the live pipelines cannot decode).

---

### S3 — FRAGILE: dead `SDXLLdmDecoder` struct still requests diffusers `up_blocks` keys
**file:** `serenitymojo/models/vae/ldm_decoder.mojo:141-295` (`SDXLLdmDecoder.load` lines 178-246).

This older struct requests DIFFUSERS keys: `decoder.conv_in`, `decoder.mid_block.resnets.0`,
`decoder.mid_block.attentions.0`, `decoder.up_blocks.{0..3}.resnets.{0..2}`,
`decoder.up_blocks.{N}.upsamplers.0`, `decoder.conv_norm_out`, `decoder.conv_out`. The on-disk
`sdxl_vae.safetensors` is **LDM** (250 tensors, F32, `decoder.mid.block_1`, `decoder.up.3.upsample`,
`nin_shortcut`, verified). Against the real file, every `up_blocks`/`mid_block`/`conv_norm_out`
lookup MISSes — this IS the round-1 bug, frozen in a now-unused struct.

**Why it is FRAGILE not BLOCKER:** the live pipeline imports `load_sdxl_ldm_decoder`
(sdxl_pipeline_smoke.mojo:36), which routes to `LdmVaeDecoder` (the round-1 FIX, LDM keys, verified
all-present vs the 250-tensor header). `SDXLLdmDecoder` is referenced ONLY by
`ldm_decoder_probe.mojo:18` against `"/nonexistent"`. So it compiles, never runs against a real file,
but is a copy-paste landmine: anyone wiring SDXL VAE to `SDXLLdmDecoder` reintroduces the exact
round-1 failure.

**Minimal fix:** delete `SDXLLdmDecoder` (lines 138-295) and repoint `ldm_decoder_probe.mojo` at
`LdmVaeDecoder[..,LATENT_CH]`. **Severity: FRAGILE.**

---

### S4 — FRAGILE: placeholder/non-existent path constants for Qwen-Image + Qwen2.5-VL + SDXL VAE dir
**files:** `qwenimage_pipeline_smoke.mojo:54-58` (`QWENIMAGE_DIR="/home/alex/.serenity/models/qwen-image"`
→ does not exist; real weights live in `~/.cache/.../models--Qwen--Qwen-Image-2512/snapshots/<sha>/`),
`nucleus_gen_smoke.mojo:71` (`CHECKPOINT` literal), `hidream_o1_smoke.mojo:189`
(`/home/alex/HiDream-O1-Image-Dev-weights` — this one EXISTS, 759-tensor index, keys verified).

These are config placeholders, not key-fidelity bugs — but they mean the Qwen-Image DiT/VAE/encoder
paths CANNOT run as-shipped, and they mask S2 (you hit "failed to open dir" before "missing key").
The DiT keys themselves are correct (verified vs the real `Qwen-Image-2512` index, 1933 tensors,
60 blocks, 0 missing) and the Qwen2.5-VL encoder keys are correct (verified vs the real
`Qwen2.5-VL-7B-Instruct` index: `model.layers.*`, Q/K/V biases present, skips `visual.*`/`lm_head`).
**Minimal fix:** point the constants at the real snapshot dirs (or document they must be overridden).
**Severity: FRAGILE.**

---

### S5 — STYLE: stale "re-uploaded as BF16" docstrings on conv loaders
**files:** `decoder2d.mojo:78` (`_load_conv_weight_rscf`), `qwenimage_decoder.mojo:190/204/230`.

The docstrings say conv weights are "re-uploaded as BF16", but the code re-uploads `w.dtype()`
(decoder2d.mojo:108, qwenimage_decoder.mojo:204/230). For the F32 Klein/SDXL/FLUX VAE files this
keeps F32 — which is CORRECT (those files are F32 on disk; the model's group_norm/conv math takes
the native dtype). No cast is lost. The doc just lies about the dtype. **Severity: STYLE** (fix the
comment). Note the deliberate BF16-VAE-weights-with-F32-norm-constants pattern in the Qwen/Wan VAE is
the known project-wide latent-precision floor, not a loading defect.

---

## PER-MODEL VERDICT (requested-key vs on-disk-header diff)

| Model | File(s) | Checkpoint read | Verdict |
|---|---|---|---|
| **Klein 9B DiT** | `klein_dit.mojo` | `flux-2-klein-base-9b.safetensors` (201 t, BF16) | **CLEAN** — all 20 probed keys present (img_in [4096,128], fused qkv [12288,4096], single linear1 [36864,4096]=3·4096+2·12288, norm.scale [128]). BFL-native naming matches. |
| **Klein VAE** | `klein_decoder.mojo` | `flux2-vae.safetensors` (251 t, F32) | **CLEAN** — diffusers keys (`decoder.up_blocks.N.resnets.M`, `bn.running_var/mean`, `post_quant_conv`, attn `to_q/k/v/to_out.0`) all present F32. Model correctly requires F32 latents (raises otherwise). |
| **Qwen-Image DiT** | `qwenimage_dit.mojo` | `Qwen-Image-2512/transformer` (1933 t, 60 blocks) | **CLEAN (keys)** — all 38 probed keys present (`transformer_blocks.N.attn.{to_q/k/v,add_{q,k,v}_proj,norm_{q,k},norm_added_{q,k},to_out.0,to_add_out}`, `img/txt_mod.1`, `img/txt_mlp.net.0.proj/net.2`, `time_text_embed.*`, `proj_out`). Path is S4. |
| **Qwen-Image VAE** | `qwenimage_decoder.mojo` | `qwen_image_vae.safetensors` (Wan) / diffusers VAE | **S2 BLOCKER** — keys match the Wan-native file only; pipeline points at diffusers layout; no remap. |
| **Qwen2.5-VL encoder** | `qwen25vl_encoder.mojo` | `Qwen2.5-VL-7B-Instruct` (729 t) | **CLEAN** — `model.layers.*` (NOT `model.language_model.*`), Q/K/V biases present, `model.norm`, `model.embed_tokens`, skips `visual.*`+`lm_head`. Path is S4. |
| **SDXL UNet** | `sdxl_unet.mojo` | `sdxl_unet_bf16.safetensors` (1680 t, BF16) | **CLEAN** — LDM-native bare keys, 0 missing of 32 probed (`input_blocks.N.M`, `time_embed.0/2`, `label_emb.0.0/0.2`, `in/out/emb_layers`, `transformer_blocks…attn1/2`, `proj_in/out` rank-2 Linear, `out.0/out.2`). RSCF conv-conversion correctly gated on rank-4. |
| **SDXL CLIP** | `clip_encoder.mojo` | `clip_l.safetensors` (196 t, F16) | **CLEAN** — `text_model.*` (token/position embed, encoder.layers.N self_attn q/k/v/out_proj, mlp.fc1/fc2, layer_norm1/2, final_layer_norm) all present; skips non-`text_model.*`. (text_projection is CLIP-G-only, loaded separately — correct.) Note: SDXL pipeline uses cached embeddings, so encoder not exercised there. |
| **SDXL VAE** | `ldm_decoder.mojo` | `OfficialStableDiffusion/sdxl_vae.safetensors` (250 t, F32) | **CLEAN via `load_sdxl_ldm_decoder`→`LdmVaeDecoder`** (LDM keys, post_quant [4,4,1,1], latent 4, all present). **But S3:** dead `SDXLLdmDecoder` struct still has the diffusers-key bug. |
| **FLUX.1 DiT** | `flux1_dit.mojo` | `flux1-dev.safetensors` (780 t, BF16, 19 dbl+38 sgl) | **CLEAN** — all 36 probed keys present incl. `guidance_in` (Dev), `vector_in`, per-block `img/txt_mod.lin`, `modulation.lin`, biases everywhere, `norm.query/key_norm.scale`. |
| **FLUX VAE** | `ldm_decoder.mojo` (`load_flux1_ldm_decoder`) | `ae.safetensors` (244 t, F32) | **CLEAN** — LDM keys, NO post_quant (has_pqc=False→inert dummy), latent 16, `nin_shortcut`, `mid.attn_1.{q,k,v,proj_out}` all present. |
| **FLUX T5** | `t5_encoder.mojo` | `t5xxl_fp16.safetensors` (220 t, F16, 24 blk) | **CLEAN** — gated-GELU `wi_0/wi_1/wo`, `relative_attention_bias` on block 0 only, robust `shared.weight`↔`encoder.embed_tokens.weight` alias (both present on disk). |
| **SenseNova U1** | `sensenova_u1.mojo` | `sensenova_u1/` (8 shards, 1116 t, 42 layers) | **CLEAN** — `language_model.model.layers.{i}.{base|_mot_gen}` 13+13/layer, q/k_norm + q/k_norm_hw [64], `model.norm`/`model.norm_mot_gen`, `fm_modules.{vision_model_mot_gen.embeddings.{patch,dense}_embedding, timestep_embedder, noise_scale_embedder, fm_head}` all present. |
| **HiDream-O1** | `hidream_o1.mojo` | `HiDream-O1-Image-Dev-weights/` (759 t) | **CLEAN** — `model.language_model.*` (embed_tokens, layers.N self_attn q/k/v/o + q/k_norm, mlp gate/up/down, in/post norms, `model.language_model.norm`), `model.x_embedder.proj1/proj2`, `model.t_embedder1.mlp`, `model.final_layer2.linear`; skips `model.visual.*`+`lm_head`. The round-1 "model. strip" concern does NOT apply here — model REQUESTS `model.language_model.*` and the file HAS it, so no strip needed and none done. |
| **Nucleus DiT + MoE** | `nucleus_dit.mojo`, `nucleus_moe.mojo` | ABSENT (CHECKPOINT placeholder) | **S1 BLOCKER** — bare keys vs `model.`-prefixed on-disk/canonical. (MoE gate/up ordering VERIFIED CORRECT, see below.) |
| **LoRA** | `lora.mojo` | `klein_lora_step200.safetensors` | round-1 F1+F2 fixes **HELD** (split→fused QKV RowRange + per-module-rank scale). |

---

## CROSS-CHECKS THAT PASSED (anti-regression record)

- **Nucleus MoE gate/up ordering is CORRECT despite looking inconsistent.** The routed-expert path
  (`nucleus_moe.mojo:229-246`) slices `gate=[0:inter], up=[inter:2*inter]` then `swiglu(g,u)=silu(g)*u`
  — matches diffusers `SwiGLUExperts`: `gate, up = (x@gate_up_proj).chunk(2); silu(gate)*up`
  (transformer_nucleusmoe_image.py:446-448). The SHARED-expert path (`_dense_ffn`, nucleus_dit.mojo:
  547-550) slices `up=[0:inner], gate=[inner:]` then `swiglu(gate,up)` — matches diffusers
  `SwiGLU.forward`: `hidden_states, gate = proj(x).chunk(2); hidden_states * silu(gate)` (i.e.
  up-first). The two on-disk tensors genuinely pack in opposite orders (`experts.gate_up_proj`=gate||up
  vs `shared_expert.net.0.proj`=up||gate), so the opposite Mojo slicing is right on BOTH. `swiglu(g,u)
  = silu(g)*u` confirmed (ops/activations.mojo:5). NOT a bug.
- **Klein/FLUX qkv split offsets** (q=0, k=inner, v=2·inner) match the fused `[3·inner, in]` layout
  on disk (Klein qkv [12288,4096]=3·4096; FLUX qkv [9216,3072]=3·3072).
- **Conv OIHW→RSCF / OIDHW→QRSCF remaps** preserve `w.dtype()` on re-upload (F32 stays F32 for the
  F32 VAE files); only the docstrings claim BF16 (S5).
- **Prefix-skip lists** are correct everywhere: T5 skips non-`encoder.*`; CLIP skips non-`text_model.*`;
  Qwen2.5-VL skips non-`model.*` (`visual.*`/`lm_head`); HiDream skips `model.visual.*`/`lm_head`;
  SenseNova streams `language_model.model.layers.*` via BlockLoader, rest resident; Qwen-Image VAE
  skips `encoder.*`/`quant_conv`.
- **Missing-key behavior is fail-loud** in every model: `_w`/`_shared` raise `"missing … weight: <name>"`
  (NOT a silent no-op). So S1/S2 fail with a clear error at first forward lookup — they will not produce
  garbage silently, unlike the LoRA-merge no-op class.
- **Round-1 LdmVaeDecoder fix** verified key-by-key vs BOTH real files (SDXL 250 t / FLUX 244 t):
  every `mid.block_{1,2}`, `mid.attn_1.{q,k,v,proj_out}`, `up.{0..3}.block.{0..2}[.nin_shortcut]`,
  `up.{1..3}.upsample.conv`, `norm_out`, `conv_out`, `post_quant_conv` (SDXL only) present.

---

## COMPILE HONESTY (build only — EXIT read; nothing run)

```
pixi run mojo build -I . -Xlinker -lm serenitymojo/models/dit/nucleus_dit_probe.mojo   -o /tmp/sk_nucleus → EXIT=0
pixi run mojo build -I . -Xlinker -lm serenitymojo/models/vae/ldm_decoder_probe.mojo   -o /tmp/sk_ldm     → EXIT=0 (1 dead-`if False` warning)
pixi run mojo build -I . -Xlinker -lm serenitymojo/models/dit/hidream_o1_probe.mojo    -o /tmp/sk_hidream → EXIT=0 (1 dead-`if False` warning)
```
Real codegen, real binaries — no "0 errors because nothing compiled". (The two BLOCKERS are key-string
mismatches, not type errors, so they compile fine; they fail only at runtime `_w()` lookup, which is
exactly why round-1 missed the class.)

---

## BOTTOM LINE

**2 BLOCKERS, both the SDXL-class "requests a key the file does not have":**
1. **Nucleus DiT** — bare keys vs `model.`-prefixed checkpoint. The round-1 "possible model. strip gap"
   was real. (Confirmed by reference; checkpoint absent.)
2. **Qwen-Image VAE** — Wan-native keys with a dropped diffusers→Wan remap, and pipelines pointed at the
   diffusers layout. Only works against an undocumented pre-remapped file no pipeline names.

Plus 1 FRAGILE landmine (dead `SDXLLdmDecoder` reintroduces round-1's exact bug if rewired), 1 FRAGILE
placeholder-path cluster (Qwen-Image/VL dirs don't exist), 1 STYLE doc lie.

Everything else — Klein/FLUX/SDXL-UNet/SenseNova/HiDream DiTs, Klein/FLUX/SDXL VAEs, CLIP/T5/Qwen2.5-VL
encoders — diffs CLEAN against the real headers. Fix S1 + S2 before any parity/smoke run.
