# model_manifest.mojo - small static metadata records for modular pipelines.
#
# These structs intentionally avoid hot-path polymorphism. A manifest chooses a
# model family and checkpoint set; each family still dispatches to specialized
# compile-time Mojo entry points for model math.


@fieldwise_init
struct ModelFamily(Copyable, Movable, ImplicitlyCopyable, Equatable):
    var tag: Int

    @staticmethod
    def text_to_image() -> ModelFamily:
        return ModelFamily(0)

    @staticmethod
    def image_to_image() -> ModelFamily:
        return ModelFamily(1)

    @staticmethod
    def text_to_video() -> ModelFamily:
        return ModelFamily(2)

    @staticmethod
    def video_to_video() -> ModelFamily:
        return ModelFamily(3)

    @staticmethod
    def audio_generation() -> ModelFamily:
        return ModelFamily(4)

    def name(self) -> String:
        if self.tag == 0:
            return "text_to_image"
        if self.tag == 1:
            return "image_to_image"
        if self.tag == 2:
            return "text_to_video"
        if self.tag == 3:
            return "video_to_video"
        if self.tag == 4:
            return "audio_generation"
        return "unknown"


@fieldwise_init
struct ModelManifest(Movable):
    var model_id: String
    var family: ModelFamily
    var variant: String
    var profile_name: String
    var checkpoint_root: String
    var tokenizer_path: String
    var text_encoder_root: String
    var denoiser_path: String
    var vae_path: String
    var default_width: Int
    var default_height: Int
    var default_frames: Int
    var latent_channels: Int
    var latent_downsample_t: Int
    var latent_downsample_s: Int
    var image_tokens: Int
    var text_tokens: Int
    var total_sequence: Int
    var patch_size: Int
    var production_entry: String

    def is_video(self) -> Bool:
        return (
            self.family == ModelFamily.text_to_video()
            or self.family == ModelFamily.video_to_video()
        )

    def latent_width(self) -> Int:
        return self.default_width // self.latent_downsample_s

    def latent_height(self) -> Int:
        return self.default_height // self.latent_downsample_s

    def latent_frames(self) -> Int:
        if not self.is_video():
            return 1
        return (self.default_frames - 1) // self.latent_downsample_t + 1

    def latent_cells(self) -> Int:
        return self.latent_frames() * self.latent_height() * self.latent_width()

    def uses_vae(self) -> Bool:
        return self.vae_path != ""


def zimage_default_manifest() -> ModelManifest:
    var zroot = String(
        "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
        "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
    )
    return ModelManifest(
        String("zimage"),
        ModelFamily.text_to_image(),
        String("base"),
        String("zimage_1024"),
        zroot,
        zroot + String("/tokenizer/tokenizer.json"),
        zroot + String("/text_encoder"),
        zroot + String("/transformer"),
        zroot + String("/vae"),
        1024,
        1024,
        1,
        16,
        1,
        8,
        4096,
        256,
        4352,
        2,
        String("serenitymojo/pipeline/zimage_pipeline.mojo"),
    )


def qwen_image_default_manifest() -> ModelManifest:
    var root = String("/home/alex/.serenity/models/checkpoints/qwen-image-2512")
    return ModelManifest(
        String("qwen_image"),
        ModelFamily.text_to_image(),
        String("qwen-image-2512"),
        String("qwen_image_1024"),
        root,
        root + String("/tokenizer/tokenizer.json"),
        root + String("/text_encoder"),
        root + String("/transformer"),
        root + String("/vae/diffusion_pytorch_model.safetensors"),
        1024,
        1024,
        1,
        16,
        1,
        8,
        4096,
        1024,
        5120,
        2,
        String("serenitymojo/pipeline/qwenimage_contract_smoke.mojo"),
    )


def qwen_image_edit_default_manifest() -> ModelManifest:
    var root = String(
        "/home/alex/.cache/huggingface/hub/"
        "models--Qwen--Qwen-Image-Edit-2511/"
        "snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
    )
    return ModelManifest(
        String("qwen_image_edit"),
        ModelFamily.image_to_image(),
        String("qwen-image-edit-2511"),
        String("qwen_image_edit_1024"),
        root,
        root + String("/processor/tokenizer.json"),
        root + String("/text_encoder"),
        root + String("/transformer"),
        root + String("/vae/diffusion_pytorch_model.safetensors"),
        1024,
        1024,
        1,
        16,
        1,
        8,
        8192,
        1024,
        9216,
        2,
        String("serenitymojo/pipeline/qwenimage_edit_contract_smoke.mojo"),
    )


def chroma_default_manifest() -> ModelManifest:
    var root = String(
        "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/"
        "snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0"
    )
    return ModelManifest(
        String("chroma"),
        ModelFamily.text_to_image(),
        String("chroma1-hd"),
        String("chroma_1024"),
        root,
        root + String("/tokenizer/spiece.model"),
        root + String("/text_encoder"),
        String("/home/alex/.serenity/models/checkpoints/chroma1_hd_bf16.safetensors"),
        root + String("/vae/diffusion_pytorch_model.safetensors"),
        1024,
        1024,
        1,
        16,
        1,
        8,
        4096,
        512,
        4608,
        2,
        String("serenitymojo/pipeline/chroma_contract_smoke.mojo"),
    )


def sd15_default_manifest() -> ModelManifest:
    var root = String(
        "/home/alex/.cache/huggingface/hub/"
        "models--stable-diffusion-v1-5--stable-diffusion-v1-5/"
        "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )
    return ModelManifest(
        String("sd15"),
        ModelFamily.text_to_image(),
        String("stable-diffusion-v1-5"),
        String("sd15_512"),
        root,
        root + String("/tokenizer/vocab.json"),
        root + String("/text_encoder"),
        root + String("/unet/diffusion_pytorch_model.safetensors"),
        root + String("/vae/diffusion_pytorch_model.safetensors"),
        512,
        512,
        1,
        4,
        1,
        8,
        4096,
        77,
        4173,
        1,
        String("serenitymojo/pipeline/sd15_contract_smoke.mojo"),
    )


def klein9b_default_manifest() -> ModelManifest:
    var qwen8_root = String(
        "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
        "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    )
    return ModelManifest(
        String("klein9b"),
        ModelFamily.text_to_image(),
        String("flux2-klein-base-9b"),
        String("klein9b_1024"),
        String("/home/alex/.serenity/models"),
        qwen8_root + String("/tokenizer.json"),
        qwen8_root,
        String("/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"),
        String("/home/alex/.serenity/models/vaes/flux2-vae.safetensors"),
        1024,
        1024,
        1,
        128,
        1,
        16,
        4096,
        512,
        4608,
        1,
        String("serenitymojo/pipeline/klein9b_pipeline_multistep_smoke.mojo"),
    )


def lance_t2v_default_manifest() -> ModelManifest:
    return ModelManifest(
        String("lance_t2v"),
        ModelFamily.text_to_video(),
        String("lance_3b_video"),
        String("lance_256_9f"),
        String("/home/alex/.serenity/models/lance/Lance_3B_Video"),
        String("/home/alex/.serenity/models/lance/Lance_3B_Video/tokenizer.json"),
        String("/home/alex/.serenity/models/lance/Lance_3B_Video"),
        String("/home/alex/.serenity/models/lance/Lance_3B_Video/model.safetensors"),
        String("/home/alex/.serenity/models/lance/Wan2.2_VAE.safetensors"),
        256,
        256,
        9,
        48,
        4,
        16,
        768,
        256,
        1028,
        1,
        String("serenitymojo/pipeline/lance_t2v_pipeline.mojo"),
    )


def flux1_dev_default_manifest() -> ModelManifest:
    return ModelManifest(
        String("flux1_dev"),
        ModelFamily.text_to_image(),
        String("flux1-dev"),
        String("flux1_dev_1024"),
        String("/home/alex/.serenity/models"),
        String("/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json"),
        String("/home/alex/.serenity/models/text_encoders"),
        String("/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"),
        String("/home/alex/.serenity/models/vaes/ae.safetensors"),
        1024,
        1024,
        1,
        16,
        1,
        8,
        4096,
        512,
        4608,
        2,
        String("serenitymojo/pipeline/flux1_pipeline_smoke.mojo"),
    )


def sdxl_default_manifest() -> ModelManifest:
    return ModelManifest(
        String("sdxl"),
        ModelFamily.text_to_image(),
        String("sdxl-base-1024-bf16"),
        String("sdxl_1024"),
        String("/home/alex/EriDiffusion/Models"),
        String("/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"),
        String("/home/alex/.serenity/models/text_encoders"),
        String("/home/alex/EriDiffusion/Models/checkpoints/sdxl_unet_bf16.safetensors"),
        String("/home/alex/.serenity/models/vaes/OfficialStableDiffusion/sdxl_vae.safetensors"),
        1024,
        1024,
        1,
        4,
        1,
        8,
        16384,
        77,
        16461,
        1,
        String("serenitymojo/pipeline/sdxl_pipeline_smoke.mojo"),
    )


def sd3_5_large_default_manifest() -> ModelManifest:
    var model_path = String(
        "/home/alex/.serenity/models/checkpoints/sd3.5_large.safetensors"
    )
    return ModelManifest(
        String("sd3_5_large"),
        ModelFamily.text_to_image(),
        String("stable-diffusion-v3.5-large"),
        String("sd3_5_large_1024"),
        String("/home/alex/.serenity/models"),
        String("/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"),
        String("/home/alex/.serenity/models/text_encoders"),
        model_path,
        model_path,
        1024,
        1024,
        1,
        16,
        1,
        8,
        4096,
        410,
        4506,
        2,
        String("serenitymojo/pipeline/sd3_pipeline_contract_smoke.mojo"),
    )


def sd3_5_medium_default_manifest() -> ModelManifest:
    var model_path = String(
        "/home/alex/.serenity/models/checkpoints/stablediffusion35_medium.safetensors"
    )
    return ModelManifest(
        String("sd3_5_medium"),
        ModelFamily.text_to_image(),
        String("stable-diffusion-v3.5-medium"),
        String("sd3_5_medium_1024"),
        String("/home/alex/.serenity/models"),
        String("/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"),
        String("/home/alex/.serenity/models/text_encoders"),
        model_path,
        model_path,
        1024,
        1024,
        1,
        16,
        1,
        8,
        4096,
        410,
        4506,
        2,
        String("serenitymojo/pipeline/sd3_medium_pipeline_contract_smoke.mojo"),
    )


def sensenova_u1_default_manifest() -> ModelManifest:
    var root = String("/home/alex/.serenity/models/sensenova_u1")
    return ModelManifest(
        String("sensenova_u1"),
        ModelFamily.text_to_image(),
        String("sensenova-u1-8b-mot"),
        String("sensenova_u1_2048"),
        root,
        root + String("/vocab.json"),
        root,
        root + String("/model.safetensors.index.json"),
        String(""),
        2048,
        2048,
        1,
        3,
        1,
        32,
        4096,
        512,
        4608,
        32,
        String("serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo"),
    )


def hidream_o1_dev_default_manifest() -> ModelManifest:
    var root = String("/home/alex/HiDream-O1-Image-Dev-weights")
    return ModelManifest(
        String("hidream_o1"),
        ModelFamily.text_to_image(),
        String("hidream-o1-image-dev-8b"),
        String("hidream_o1_dev_2048"),
        root,
        root + String("/tokenizer.json"),
        root,
        root + String("/model.safetensors.index.json"),
        String(""),
        2048,
        2048,
        1,
        3,
        1,
        32,
        4096,
        512,
        4608,
        32,
        String("serenitymojo/pipeline/hidream_o1_smoke.mojo"),
    )


def anima_default_manifest() -> ModelManifest:
    var root = String("/home/alex/.serenity/models/anima")
    return ModelManifest(
        String("anima"),
        ModelFamily.text_to_image(),
        String("anima-base-v1.0"),
        String("anima_1024"),
        root,
        String(""),
        root
        + String("/split_files/text_encoders/qwen_3_06b_base.safetensors"),
        root
        + String("/split_files/diffusion_models/anima-base-v1.0.safetensors"),
        root + String("/split_files/vae/qwen_image_vae.safetensors"),
        1024,
        1024,
        1,
        16,
        1,
        8,
        4096,
        256,
        4352,
        2,
        String("serenitymojo/pipeline/anima_contract_smoke.mojo"),
    )


def lens_default_manifest() -> ModelManifest:
    var root = String("/home/alex/.serenity/models/microsoft_lens")
    return ModelManifest(
        String("microsoft_lens"),
        ModelFamily.text_to_image(),
        String("lens"),
        String("lens_1024"),
        root,
        root + String("/tokenizer/tokenizer.json"),
        root + String("/text_encoder"),
        root + String("/transformer"),
        root + String("/vae/diffusion_pytorch_model.safetensors"),
        1024,
        1024,
        1,
        32,
        1,
        16,
        4096,
        415,
        4511,
        2,
        String("serenitymojo/pipeline/lens_contract_smoke.mojo"),
    )


def zimage_l2p_default_manifest() -> ModelManifest:
    var root = String("/home/alex/.serenity/models/checkpoints/L2P")
    return ModelManifest(
        String("zimage_l2p"),
        ModelFamily.text_to_image(),
        String("zimage-turbo-l2p"),
        String("zimage_l2p_1024"),
        root,
        String(""),
        String(""),
        root + String("/model-1k-merge.safetensors"),
        String(""),
        1024,
        1024,
        1,
        3,
        1,
        1,
        4096,
        0,
        4096,
        16,
        String("serenitymojo/pipeline/zimage_l2p_contract_smoke.mojo"),
    )


def ernie_image_default_manifest() -> ModelManifest:
    var root = String("/home/alex/models/ERNIE-Image")
    return ModelManifest(
        String("ernie_image"),
        ModelFamily.text_to_image(),
        String("ernie-image-8b"),
        String("ernie_image_1024"),
        root,
        root + String("/tokenizer/tokenizer.json"),
        root + String("/text_encoder"),
        root + String("/transformer"),
        root + String("/vae/diffusion_pytorch_model.safetensors"),
        1024,
        1024,
        1,
        128,
        1,
        16,
        4096,
        256,
        4352,
        1,
        String("serenitymojo/pipeline/ernie_contract_smoke.mojo"),
    )
