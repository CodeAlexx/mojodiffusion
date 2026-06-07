#!/usr/bin/env python3
"""Static guard for OneTrainer VAE/postprocess sampler contracts.

Reference is intentionally limited to local OneTrainer and OneTrainer-anima-ref.
This guard checks the sampler/model source contracts mirrored by
serenitymojo/sampling/vae_postprocess_contract.mojo. It does not claim full image
parity; that still requires text conditioning, denoise, VAE decode, and image
write verification.
"""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")
ANIMA_REF = Path("/home/alex/OneTrainer-anima-ref")

MOJO_CONTRACT = REPO / "serenitymojo/sampling/vae_postprocess_contract.mojo"
MOJO_SMOKE = REPO / "serenitymojo/sampling/vae_postprocess_contract_smoke.mojo"
MOJO_ENCODER_CONTRACT = REPO / "serenitymojo/sampling/vae_encoder_contract.mojo"
MOJO_ENCODER_SMOKE = REPO / "serenitymojo/sampling/vae_encoder_contract_smoke.mojo"


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"[vae-sampler] missing file: {path}")
    return path.read_text(encoding="utf-8")


def require(path: Path, label: str, needles: list[str]) -> None:
    text = read(path)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        print(f"[vae-sampler] FAIL {label}: {path}")
        for needle in missing:
            print(f"  missing: {needle}")
        raise SystemExit(1)
    print(f"[vae-sampler] PASS {label}")


def require_no_forced_f32_storage(path: Path, label: str) -> None:
    text = read(path)
    forbidden = [
        "STDtype.F32",
        "DType.float32",
        ".to_host(",
        "Tensor.from_host",
    ]
    found = [needle for needle in forbidden if needle in text]
    if found:
        print(f"[vae-sampler] FAIL dtype boundary {label}: {path}")
        for needle in found:
            print(f"  forbidden tensor-storage pattern: {needle}")
        raise SystemExit(1)
    if "preserve BF16/F16 latent storage at boundaries" not in text:
        print(f"[vae-sampler] FAIL dtype comment {label}: {path}")
        print("  missing explicit latent storage dtype-boundary comment")
        raise SystemExit(1)
    print(f"[vae-sampler] PASS dtype boundary {label}")


def main() -> int:
    samplers = ONETRAINER / "modules/modelSampler"
    models = ONETRAINER / "modules/model"
    anima_samplers = ANIMA_REF / "modules/modelSampler"
    anima_models = ANIMA_REF / "modules/model"

    require(
        samplers / "BaseModelSampler.py",
        "OneTrainer base quantization",
        [
            "def quantize_resolution(resolution: int, quantization: int) -> int:",
            "return round(resolution / quantization) * quantization",
        ],
    )
    require(
        ONETRAINER / "modules/dataLoader/StableDiffusionXLBaseDataLoader.py",
        "OneTrainer SDXL VAE encode/precache",
        [
            "RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)",
            "EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae",
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
            "image_split_names = ['latent_image', 'original_resolution', 'crop_offset']",
        ],
    )
    require(
        ONETRAINER / "modules/dataLoader/StableDiffusion3BaseDataLoader.py",
        "OneTrainer SD3.5 VAE encode/precache",
        [
            "EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae",
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
            "encode_conditioning_image = EncodeVAE(in_name='conditioning_image'",
        ],
    )
    require(
        ONETRAINER / "modules/dataLoader/QwenBaseDataLoader.py",
        "OneTrainer Qwen VAE encode/precache",
        [
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
            "vae_frame_dim=True,  #...Qwen has a video-capable VAE. convert images to video dimensions",
            "image_split_names = ['latent_image', 'original_resolution', 'crop_offset']",
        ],
    )
    require(
        ONETRAINER / "modules/dataLoader/Flux2BaseDataLoader.py",
        "OneTrainer Flux2 dev/Klein VAE encode/precache",
        [
            "EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae",
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
            "image_split_names = ['latent_image', 'original_resolution', 'crop_offset']",
        ],
    )
    require(
        ONETRAINER / "modules/dataLoader/ErnieBaseDataLoader.py",
        "OneTrainer Ernie VAE encode/precache",
        [
            "EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae",
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
            "image_split_names = ['latent_image', 'original_resolution', 'crop_offset']",
        ],
    )
    require(
        anima_models / "AnimaModel.py",
        "OneTrainer-anima VAE scaling source",
        [
            "AutoencoderKLQwenImage",
            "def scale_latents(self, latents: Tensor) -> Tensor:",
            "return (latents - latents_mean) * latents_std",
        ],
    )
    require(
        ANIMA_REF / "modules/dataLoader/AnimaBaseDataLoader.py",
        "OneTrainer-anima VAE encode/precache",
        [
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
            "vae_frame_dim=True,  #...Anima has a video-capable VAE. convert images to video dimensions",
            "image_split_names = ['latent_image', 'original_resolution', 'crop_offset']",
        ],
    )
    require(
        ONETRAINER / "modules/dataLoader/FluxBaseDataLoader.py",
        "OneTrainer Flux VAE encode/precache",
        [
            "EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae",
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
        ],
    )
    require(
        ONETRAINER / "modules/dataLoader/ChromaBaseDataLoader.py",
        "OneTrainer Chroma VAE encode/precache",
        [
            "EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae",
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
        ],
    )
    require(
        ONETRAINER / "modules/dataLoader/ZImageBaseDataLoader.py",
        "OneTrainer Z-Image VAE encode/precache",
        [
            "EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae",
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
        ],
    )
    require(
        ONETRAINER / "modules/modelSetup/BaseFlux2Setup.py",
        "OneTrainer Flux2 dev/Klein prepared latent shape",
        [
            "latent_image = model.patchify_latents(batch['latent_image'].float())",
            "scaled_latent_image = model.scale_latents(latent_image)",
            "packed_latent_input = model.pack_latents(latent_input)",
        ],
    )
    require(
        ONETRAINER / "modules/modelSetup/BaseErnieSetup.py",
        "OneTrainer Ernie prepared latent shape",
        [
            "latent_image = model.patchify_latents(batch['latent_image'].float())",
            "scaled_latent_image = model.scale_latents(latent_image)",
            "hidden_states=scaled_noisy_latent_image.to(dtype=model.train_dtype.torch_dtype())",
        ],
    )
    require(
        ONETRAINER / "modules/modelSetup/BaseQwenSetup.py",
        "OneTrainer Qwen prepared latent shape",
        [
            "latent_image = batch['latent_image']",
            "scaled_latent_image = model.scale_latents(latent_image)",
            "packed_latent_input = model.pack_latents(latent_input)",
            "img_shapes = [[(",
        ],
    )
    require(
        ANIMA_REF / "modules/modelSetup/BaseAnimaSetup.py",
        "OneTrainer-anima prepared latent shape",
        [
            "latent_image = batch['latent_image']",
            "scaled_latent_image = model.scale_latents(latent_image)",
            "Anima latents are 5D (B,16,1,H/8,W/8)",
        ],
    )
    require(
        ONETRAINER / "modules/modelSetup/BaseStableDiffusionSetup.py",
        "OneTrainer SDXL prepared latent scale",
        ["scaled_latent_image = latent_image * vae_scaling_factor"],
    )
    require(
        ONETRAINER / "modules/modelSetup/BaseStableDiffusion3Setup.py",
        "OneTrainer SD3.5 prepared latent scale",
        ["scaled_latent_image = (latent_image - vae_shift_factor) * vae_scaling_factor"],
    )
    require(
        ONETRAINER / "modules/modelSetup/BaseFluxSetup.py",
        "OneTrainer Flux prepared latent scale",
        [
            "scaled_latent_image = (latent_image - vae_shift_factor) * vae_scaling_factor",
            "packed_latent_input = model.pack_latents(latent_input)",
        ],
    )
    require(
        ONETRAINER / "modules/modelSetup/BaseChromaSetup.py",
        "OneTrainer Chroma prepared latent scale",
        [
            "scaled_latent_image = (latent_image - vae_shift_factor) * vae_scaling_factor",
            "packed_latent_input = model.pack_latents(latent_input)",
        ],
    )
    require(
        ONETRAINER / "modules/modelSetup/BaseZImageSetup.py",
        "OneTrainer Z-Image prepared latent scale",
        ["scaled_latent_image = model.scale_latents(batch['latent_image'])"],
    )
    require(
        samplers / "StableDiffusionXLSampler.py",
        "OneTrainer SDXL VAE decode/postprocess",
        [
            "latent_image = latent_image.to(dtype=self.model.vae_train_dtype.torch_dtype())",
            "vae.decode(latent_image / vae.config.scaling_factor, return_dict=False)[0]",
            "image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)",
            "height=self.quantize_resolution(sample_config.height, 64)",
            "width=self.quantize_resolution(sample_config.width, 64)",
        ],
    )
    require(
        samplers / "StableDiffusion3Sampler.py",
        "OneTrainer SD3.5 VAE decode/postprocess",
        [
            "vae_scale_factor = self.pipeline.vae_scale_factor",
            "latents = (latent_image / vae.config.scaling_factor) + vae.config.shift_factor",
            "vae.decode(latents, return_dict=False)[0]",
            "image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)",
            "height=self.quantize_resolution(sample_config.height, 16)",
            "width=self.quantize_resolution(sample_config.width, 16)",
        ],
    )
    require(
        models / "QwenModel.py",
        "OneTrainer Qwen VAE latent layout",
        [
            "batch_size, channels, frames, height, width = latents.shape",
            "assert frames == 1",
            "latents_mean = torch.tensor(self.vae.config.latents_mean",
            "latents_std = 1.0 / torch.tensor(self.vae.config.latents_std",
            "return latents / latents_std + latents_mean",
        ],
    )
    require(
        samplers / "QwenSampler.py",
        "OneTrainer Qwen decode/postprocess",
        [
            "size=(1, num_latent_channels, 1, height // vae_scale_factor, width // vae_scale_factor)",
            "latent_image = self.model.pack_latents(latent_image)",
            "latent_image = self.model.unpack_latents(",
            "latents = self.model.unscale_latents(latent_image)",
            "vae.decode(latents, return_dict=False)[0].squeeze(-3)",
            "image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)",
            "height=self.quantize_resolution(sample_config.height, 64)",
        ],
    )
    require(
        models / "ErnieModel.py",
        "OneTrainer Ernie patchify/BN unscale",
        [
            "def patchify_latents(latents: torch.Tensor) -> torch.Tensor:",
            "latents = latents.permute(0, 1, 3, 5, 2, 4)",
            "mean = self.vae.bn.running_mean.view(1, -1, 1, 1)",
            "self.vae.config.batch_norm_eps",
            "return latents * std + mean",
        ],
    )
    require(
        samplers / "ErnieSampler.py",
        "OneTrainer Ernie decode/manual postprocess",
        [
            "latent_image = self.model.patchify_latents(latent_image)",
            "latents = self.model.unscale_latents(latent_image)",
            "latents = self.model.unpatchify_latents(latents)",
            "image = vae.decode(latents, return_dict=False)[0]",
            "image = (image.clamp(-1, 1) + 1) / 2",
            "PILImage.fromarray((img * 255).astype(np.uint8))",
            "height=self.quantize_resolution(sample_config.height, 64)",
        ],
    )
    require(
        anima_models / "AnimaModel.py",
        "OneTrainer-anima Qwen-image VAE scaling",
        [
            "AutoencoderKLQwenImage",
            "latents_mean = torch.tensor(self.vae.config.latents_mean",
            "latents_std = 1.0 / torch.tensor(self.vae.config.latents_std",
            "return latents / latents_std + latents_mean",
        ],
    )
    require(
        anima_samplers / "AnimaSampler.py",
        "OneTrainer-anima decode/postprocess",
        [
            "VaeImageProcessor(vae_scale_factor=8)",
            "size=(1, num_latent_channels, 1, height // vae_scale_factor, width // vae_scale_factor)",
            "latents = self.model.unscale_latents(latent_image)",
            "image = vae.decode(latents, return_dict=False)[0][:, :, 0]",
            "self.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)",
            "height=self.quantize_resolution(sample_config.height, 64)",
        ],
    )
    require(
        models / "FluxModel.py",
        "OneTrainer Flux pack/unpack",
        [
            "def prepare_latent_image_ids(",
            "def pack_latents(self, latents: Tensor) -> Tensor:",
            "latents = latents.permute(0, 2, 4, 1, 3, 5)",
            "def unpack_latents(self, latents, height: int, width: int):",
            "return math.exp(mu)",
        ],
    )
    require(
        samplers / "FluxSampler.py",
        "OneTrainer Flux decode/postprocess",
        [
            "vae_scale_factor = 8",
            "num_latent_channels = 16",
            "latent_image = self.model.pack_latents(latent_image)",
            "latent_image = self.model.unpack_latents(",
            "latents = (latent_image / vae.config.scaling_factor) + vae.config.shift_factor",
            "image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)",
            "height=self.quantize_resolution(sample_config.height, 64)",
        ],
    )
    require(
        models / "Flux2Model.py",
        "OneTrainer Flux2 dev/Klein patchify/BN",
        [
            "def patchify_latents(latents: torch.Tensor) -> torch.Tensor:",
            "latents = latents.permute(0, 1, 3, 5, 2, 4)",
            "def pack_latents(latents) -> Tensor:",
            "latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1)",
            "return latents * latents_bn_std + latents_bn_mean",
        ],
    )
    require(
        samplers / "Flux2Sampler.py",
        "OneTrainer Flux2 dev/Klein decode/postprocess",
        [
            "vae_scale_factor = 8",
            "num_latent_channels = 32",
            "latent_image = self.model.patchify_latents(latent_image)",
            "latent_image = self.model.unpack_latents(",
            "latents = self.model.unscale_latents(latent_image)",
            "latents = self.model.unpatchify_latents(latents)",
            "image = vae.decode(latents, return_dict=False)[0]",
            "image_processor.postprocess(image, output_type='pil')",
            "height=self.quantize_resolution(sample_config.height, 64)",
        ],
    )
    require(
        samplers / "ChromaSampler.py",
        "OneTrainer Chroma decode/postprocess",
        [
            "vae_scale_factor = 8",
            "num_latent_channels = 16",
            "latent_image = self.model.pack_latents(latent_image)",
            "latent_image = self.model.unpack_latents(",
            "latents = (latent_image / vae.config.scaling_factor) + vae.config.shift_factor",
            "image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)",
            "height=self.quantize_resolution(sample_config.height, 64)",
        ],
    )
    require(
        models / "ZImageModel.py",
        "OneTrainer Z-Image scale/unscale",
        [
            "return (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor",
            "return latents / self.vae.config.scaling_factor + self.vae.config.shift_factor",
        ],
    )
    require(
        samplers / "ZImageSampler.py",
        "OneTrainer Z-Image decode/postprocess",
        [
            "vae_scale_factor = 8",
            "num_latent_channels = transformer.in_channels",
            "latents = self.model.unscale_latents(latent_image)",
            "image = vae.decode(latents, return_dict=False)[0]",
            "image_processor.postprocess(image, output_type='pil')",
            "height=self.quantize_resolution(sample_config.height, 64)",
        ],
    )

    require(
        MOJO_CONTRACT,
        "Mojo VAE/postprocess contract",
        [
            "OT_VAE_LAYOUT_FLUX_2X2_SEQUENCE",
            "OT_VAE_LAYOUT_QWEN_2X2_SEQUENCE_5D",
            "OT_VAE_DECODE_QWEN_MEAN_STD",
            "OT_VAE_DECODE_BN_UNSCALE_UNPATCHIFY",
            "OT_IMAGE_POSTPROCESS_ERNIE_MANUAL_UINT8",
            "qwenimage_unscale_value",
            "bn_unscale_value",
            "ernie_manual_uint8_value",
        ],
    )
    require(MOJO_SMOKE, "Mojo VAE/postprocess smoke", ["OneTrainer VAE/postprocess contract smoke PASS"])
    require_no_forced_f32_storage(MOJO_CONTRACT, "Mojo VAE/postprocess contract")
    require(
        REPO / "serenitymojo/models/vae/ldm_encoder.mojo",
        "Mojo LDM encoder readiness source",
        [
            "def encode_mean(",
            "load_sdxl_ldm_encoder",
            "load_sd3_embedded_ldm_encoder",
            "Diffusers post-encode scaling (the pipeline boundary, NOT inside encode)",
        ],
    )
    require(
        REPO / "serenitymojo/models/vae/qwenimage_encoder.mojo",
        "Mojo Qwen/Anima encoder readiness source",
        [
            "QwenImageVaeEncoder",
            "def encode_mean(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:",
            "lift to [1,16,1,LH,LW] at the",
        ],
    )
    require(
        REPO / "serenitymojo/models/vae/klein_encoder.mojo",
        "Mojo Klein prepared encoder readiness source",
        [
            "KleinVaeEncoder",
            "_patchify_packed",
            "_bn_forward",
            "packed 128-channel latent",
        ],
    )
    require(
        REPO / "serenitymojo/models/vae/zimage_encoder.mojo",
        "Mojo Z-Image encoder readiness source",
        [
            "ZImageVaeEncoder",
            "def encode_mean(self, image_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:",
            "matching OneTrainer cache mode",
        ],
    )
    require(
        MOJO_ENCODER_CONTRACT,
        "Mojo VAE encoder/cache contract",
        [
            "SampleVAEDistribution(mode=\"mean\")",
            "OT_SAMPLER_FLUX2_DEV",
            "OT_VAE_ENCODER_PARTIAL_PREPARED_ONLY",
            "require_raw_cache_encode_ready",
            "cache_to_prepared_patch_size",
            "decode_postprocess_is_separate",
        ],
    )
    require(
        MOJO_ENCODER_SMOKE,
        "Mojo VAE encoder/cache smoke",
        [
            "OneTrainer VAE encoder/cache contract smoke PASS",
            "Flux2 dev raw cache encode must fail loud",
            "Flux2 raw cache encode must fail loud",
        ],
    )
    require_no_forced_f32_storage(MOJO_ENCODER_CONTRACT, "Mojo VAE encoder/cache contract")

    print("[vae-sampler] PASS all VAE encoder/postprocess source contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
