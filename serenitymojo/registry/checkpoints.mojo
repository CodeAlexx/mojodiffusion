# checkpoints.mojo - metadata-only checkpoint path checks.

from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.runtime.model_manifest import (
    ModelManifest,
    anima_default_manifest,
    chroma_default_manifest,
    ernie_image_default_manifest,
    flux1_dev_default_manifest,
    hidream_o1_dev_default_manifest,
    klein9b_default_manifest,
    lance_t2v_default_manifest,
    lens_default_manifest,
    qwen_image_edit_default_manifest,
    qwen_image_default_manifest,
    sd15_default_manifest,
    sd3_5_large_default_manifest,
    sd3_5_medium_default_manifest,
    sdxl_default_manifest,
    sensenova_u1_default_manifest,
    zimage_l2p_default_manifest,
    zimage_default_manifest,
)


@fieldwise_init
struct CheckpointStatus(Copyable, Movable, ImplicitlyCopyable):
    var checked: Int
    var missing: Int

    def ok(self) -> Bool:
        return self.missing == 0


def path_exists(path: String) -> Bool:
    if path == "":
        return True
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _count_path(path: String, mut checked: Int, mut missing: Int) -> CheckpointStatus:
    if path != "":
        checked += 1
        if not path_exists(path):
            missing += 1
    return CheckpointStatus(checked, missing)


def validate_manifest_paths(manifest: ModelManifest) -> CheckpointStatus:
    var checked = 0
    var missing = 0
    var s = _count_path(manifest.checkpoint_root, checked, missing)
    checked = s.checked
    missing = s.missing
    s = _count_path(manifest.tokenizer_path, checked, missing)
    checked = s.checked
    missing = s.missing
    s = _count_path(manifest.text_encoder_root, checked, missing)
    checked = s.checked
    missing = s.missing
    s = _count_path(manifest.denoiser_path, checked, missing)
    checked = s.checked
    missing = s.missing
    s = _count_path(manifest.vae_path, checked, missing)
    checked = s.checked
    missing = s.missing

    if manifest.model_id == "sensenova_u1":
        s = _count_path(manifest.checkpoint_root + String("/merges.txt"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/added_tokens.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(manifest.checkpoint_root + String("/config.json"), checked, missing)
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "hidream_o1":
        s = _count_path(manifest.checkpoint_root + String("/config.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(manifest.checkpoint_root + String("/merges.txt"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(manifest.checkpoint_root + String("/vocab.json"), checked, missing)
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "flux1_dev":
        s = _count_path(
            manifest.text_encoder_root + String("/clip_l.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/t5xxl_fp16.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "sdxl":
        s = _count_path(
            manifest.text_encoder_root + String("/clip_l.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/clip_g.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/clip_g.tokenizer.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "sd3_5_large":
        s = _count_path(
            manifest.text_encoder_root + String("/clip_l.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/clip_g.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/clip_g.tokenizer.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/t5xxl_fp16.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/t5xxl_fp16.tokenizer.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "sd3_5_medium":
        s = _count_path(
            manifest.text_encoder_root + String("/clip_l.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/clip_g.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/clip_g.tokenizer.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/t5xxl_fp16.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/t5xxl_fp16.tokenizer.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "microsoft_lens":
        s = _count_path(manifest.checkpoint_root + String("/model_index.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.denoiser_path + String("/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.denoiser_path
            + String("/diffusion_pytorch_model.safetensors.index.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/model.safetensors.index.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/tokenizer/chat_template.jinja"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/scheduler/scheduler_config.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/vae/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "ernie_image":
        s = _count_path(manifest.checkpoint_root + String("/model_index.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.denoiser_path + String("/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.denoiser_path
            + String("/diffusion_pytorch_model.safetensors.index.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.denoiser_path
            + String("/diffusion_pytorch_model-00001-of-00002.safetensors"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.denoiser_path
            + String("/diffusion_pytorch_model-00002-of-00002.safetensors"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/model.safetensors"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/tokenizer/tokenizer_config.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/scheduler/scheduler_config.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/vae/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "qwen_image":
        s = _count_path(manifest.checkpoint_root + String("/model_index.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.denoiser_path + String("/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.denoiser_path
            + String("/diffusion_pytorch_model.safetensors.index.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        for shard in range(1, 10):
            var suffix = String(shard)
            if shard < 10:
                suffix = String("0") + suffix
            s = _count_path(
                manifest.denoiser_path
                + String("/diffusion_pytorch_model-000")
                + suffix
                + String("-of-00009.safetensors"),
                checked,
                missing,
            )
            checked = s.checked
            missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/model.safetensors.index.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        for shard in range(1, 5):
            var suffix = String(shard)
            if shard < 10:
                suffix = String("0") + suffix
            s = _count_path(
                manifest.text_encoder_root
                + String("/model-000")
                + suffix
                + String("-of-00004.safetensors"),
                checked,
                missing,
            )
            checked = s.checked
            missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/tokenizer/tokenizer_config.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/tokenizer/chat_template.jinja"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/scheduler/scheduler_config.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/vae/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "qwen_image_edit":
        s = _count_path(manifest.checkpoint_root + String("/model_index.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.denoiser_path + String("/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.denoiser_path
            + String("/diffusion_pytorch_model.safetensors.index.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        for shard in range(1, 6):
            s = _count_path(
                manifest.denoiser_path
                + String("/diffusion_pytorch_model-0000")
                + String(shard)
                + String("-of-00005.safetensors"),
                checked,
                missing,
            )
            checked = s.checked
            missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.text_encoder_root + String("/model.safetensors.index.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        for shard in range(1, 5):
            s = _count_path(
                manifest.text_encoder_root
                + String("/model-0000")
                + String(shard)
                + String("-of-00004.safetensors"),
                checked,
                missing,
            )
            checked = s.checked
            missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/processor/chat_template.jinja"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/scheduler/scheduler_config.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            manifest.checkpoint_root + String("/vae/config.json"), checked, missing
        )
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "chroma":
        var chroma_root = manifest.checkpoint_root
        s = _count_path(chroma_root + String("/transformer/config.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(
            chroma_root + String("/transformer/diffusion_pytorch_model.safetensors.index.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            chroma_root + String("/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            chroma_root + String("/transformer/diffusion_pytorch_model-00002-of-00002.safetensors"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(chroma_root + String("/text_encoder/config.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(
            chroma_root + String("/text_encoder/model.safetensors.index.json"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            chroma_root + String("/text_encoder/model-00001-of-00002.safetensors"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(
            chroma_root + String("/text_encoder/model-00002-of-00002.safetensors"),
            checked,
            missing,
        )
        checked = s.checked
        missing = s.missing
        s = _count_path(chroma_root + String("/tokenizer/tokenizer_config.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(chroma_root + String("/scheduler/scheduler_config.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(chroma_root + String("/vae/config.json"), checked, missing)
        checked = s.checked
        missing = s.missing

    if manifest.model_id == "sd15":
        var sd15_root = manifest.checkpoint_root
        s = _count_path(sd15_root + String("/model_index.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(sd15_root + String("/tokenizer/merges.txt"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(sd15_root + String("/tokenizer/tokenizer_config.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(sd15_root + String("/scheduler/scheduler_config.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(sd15_root + String("/text_encoder/config.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(sd15_root + String("/text_encoder/model.safetensors"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(sd15_root + String("/unet/config.json"), checked, missing)
        checked = s.checked
        missing = s.missing
        s = _count_path(sd15_root + String("/vae/config.json"), checked, missing)
        checked = s.checked
        missing = s.missing

    return CheckpointStatus(checked, missing)


def default_manifest_count() -> Int:
    return 17


def default_manifest_at(index: Int) raises -> ModelManifest:
    if index == 0:
        return zimage_default_manifest()
    if index == 1:
        return klein9b_default_manifest()
    if index == 2:
        return qwen_image_default_manifest()
    if index == 3:
        return qwen_image_edit_default_manifest()
    if index == 4:
        return chroma_default_manifest()
    if index == 5:
        return sd15_default_manifest()
    if index == 6:
        return lance_t2v_default_manifest()
    if index == 7:
        return flux1_dev_default_manifest()
    if index == 8:
        return sdxl_default_manifest()
    if index == 9:
        return sensenova_u1_default_manifest()
    if index == 10:
        return hidream_o1_dev_default_manifest()
    if index == 11:
        return sd3_5_large_default_manifest()
    if index == 12:
        return sd3_5_medium_default_manifest()
    if index == 13:
        return anima_default_manifest()
    if index == 14:
        return lens_default_manifest()
    if index == 15:
        return zimage_l2p_default_manifest()
    if index == 16:
        return ernie_image_default_manifest()
    raise Error("default_manifest_at: index out of range")


def default_manifest_by_id(model_id: String) raises -> ModelManifest:
    for i in range(default_manifest_count()):
        var manifest = default_manifest_at(i)
        if manifest.model_id == model_id:
            return manifest^
    raise Error(String("default_manifest_by_id: unknown model id ") + model_id)


def validate_registered_manifest_paths() raises -> CheckpointStatus:
    var checked = 0
    var missing = 0
    for i in range(default_manifest_count()):
        var status = validate_manifest_paths(default_manifest_at(i))
        checked += status.checked
        missing += status.missing
    return CheckpointStatus(checked, missing)
