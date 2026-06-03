# VAE Port Gap Audit - 2026-06-03

Status: read-only audit from Rust/source side plus current Mojo coverage. This
is a planning note for future trainer/inference/sampler work; ERNIE's current
2500-step run is not blocked by these gaps.

## Summary

- ERNIE reuses the Klein/Flux2 VAE family. Mojo already has
  `models/vae/klein_encoder.mojo` and `models/vae/klein_decoder.mojo`.
- The highest-priority missing encoder is SDXL/LDM AutoencoderKL. Decoder exists
  through `models/vae/ldm_decoder.mojo`, but real SDXL prepare/training and
  img2img/inpaint need the encoder.
- LTX2 video and Wan2.2/Lance have decoder coverage but need VAE encoders for
  training/roundtrip workflows.
- Pixel-space models such as Z-Image L2P, SenseNova, and HiDream do not need a
  VAE port.

## Gap Matrix

| Model / Family | Rust/source VAE | Mojo encoder | Mojo decoder | Current use | Gap |
|---|---|---|---|---|---|
| Klein / Flux.2 / ERNIE | `/home/alex/EriDiffusion/inference-flame/src/vae/klein_vae.rs` | `serenitymojo/models/vae/klein_encoder.mojo` | `serenitymojo/models/vae/klein_decoder.mojo` | Klein/ERNIE prepare, validation sampler, decode | Harden full RGB oracle decode gate |
| Z-Image | Rust LDM encoder/decoder | `serenitymojo/models/vae/zimage_encoder.mojo` | `serenitymojo/models/vae/zimage_decoder.mojo` | Z-Image prepare/generate | Add real-weight encoder parity/roundtrip gate |
| Flux.1-dev | Rust/BFL LDM VAE | `serenitymojo/vae/flux_vae_encoder.mojo` | `serenitymojo/models/vae/ldm_decoder.mojo` | Flux prepare/trainer cache, decode | No VAE-specific port; text path remains separate |
| SDXL | Rust SDXL/LDM VAE | Missing | `serenitymojo/models/vae/ldm_decoder.mojo` | SDXL decode smoke/trainer cache | Port SDXL/LDM AutoencoderKL encoder |
| SD1.5 | Rust LDM VAE | Missing | `load_sd15_ldm_decoder` | SD1.5 VAE smoke | Port encoder if training/img2img/inpaint matters |
| SD3 | Embedded LDM VAE | Missing | `load_sd3_embedded_ldm_decoder` | SD3 VAE decode smokes | Port embedded encoder only if training/img2img needed |
| Qwen-Image / Edit / Anima | Rust QwenImage/Wan2.1-style VAE | `serenitymojo/models/vae/qwenimage_encoder.mojo` | `serenitymojo/models/vae/qwenimage_decoder.mojo` | Qwen/Anima prepare/decode | Wire stale Anima prepare paths to current encoder if used |
| LTX2 video | `/home/alex/EriDiffusion/inference-flame/src/vae/ltx2_encoder.rs` + `ltx2_vae.rs` | Missing | `serenitymojo/models/vae/ltx2_vae_decoder.mojo` | LTX2 video decode smokes | Port LTX2 video VAE encoder |
| LTX2 audio | Rust audio VAE/vocoder | Missing or partial | `serenitymojo/models/vae/ltx2_audio_vae.mojo` | Audio VAE/vocoder smokes | Port audio encoder if audio training/roundtrip needed |
| Lance / Wan2.2 | Rust Wan2.2 VAE | Missing | `serenitymojo/models/vae/wan22_decoder.mojo` | Lance image/video decode | Port Wan2.2 VAE encoder |
| Chroma | Flux-compatible LDM VAE | Missing / shared | `load_flux1_ldm_decoder` | Chroma decode | Flux-compatible encoder for staged training/inpaint |
| Lens | Flux2/Klein concepts | Reuse Klein when needed | `KleinVaeDecoder` | Lens decode through Klein VAE | Verify Lens latent convention before production |
| Z-Image L2P | None | N/A | N/A | Pixel-space | No VAE work |
| SenseNova / HiDream | None | N/A | N/A | Pixel-space | No VAE work |

## Priority

1. SDXL/LDM AutoencoderKL encoder.
2. LTX2 video VAE encoder.
3. Wan2.2/Lance VAE encoder.
4. SD3 embedded encoder only when SD3 training/img2img becomes active.
