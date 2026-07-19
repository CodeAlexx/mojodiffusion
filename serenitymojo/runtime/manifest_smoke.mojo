# manifest_smoke.mojo - compile-only gate for modular runtime scaffolding.

from serenitymojo.registry.checkpoints import (
    default_manifest_at,
    default_manifest_by_id,
    default_manifest_count,
    validate_manifest_paths,
    validate_registered_manifest_paths,
)
from serenitymojo.runtime.execution_config import default_smoke_config
from serenitymojo.runtime.model_manifest import ModelManifest
from serenitymojo.runtime.production_guard import production_gpu_math_guard
from serenitymojo.runtime.request import default_t2i_request, default_t2v_request


def _print_manifest(manifest: ModelManifest):
    print(
        "[manifest]",
        manifest.model_id,
        manifest.profile_name,
        manifest.default_width,
        "x",
        manifest.default_height,
        "frames=",
        manifest.default_frames,
        "latent=",
        manifest.latent_height(),
        "x",
        manifest.latent_width(),
        "x",
        manifest.latent_frames(),
        "tokens=",
        manifest.image_tokens,
        "text=",
        manifest.text_tokens,
        "seq=",
        manifest.total_sequence,
    )


def _check_profile(manifest: ModelManifest) raises:
    if manifest.default_width <= 0 or manifest.default_height <= 0:
        raise Error(String("manifest has invalid image size: ") + manifest.model_id)
    if manifest.image_tokens <= 0:
        raise Error(String("manifest has no image token profile: ") + manifest.model_id)
    if manifest.total_sequence < manifest.image_tokens:
        raise Error(String("manifest sequence shorter than image tokens: ") + manifest.model_id)
    if manifest.patch_size <= 0:
        raise Error(String("manifest has invalid patch size: ") + manifest.model_id)


def main() raises:
    var cfg = default_smoke_config()
    var guard = production_gpu_math_guard()
    var req_i = default_t2i_request(String("klein9b"), String("fairy"), String("out.png"))
    var req_v = default_t2v_request(String("lance_t2v"), String("fairy"), String("out_frames"))

    var seen_sensenova = False
    var seen_hidream = False
    var seen_flux = False
    var seen_sdxl = False
    var seen_sd3 = False
    var seen_sd3_medium = False
    var seen_qwen = False
    var seen_qwen_edit = False
    var seen_chroma = False
    var seen_sd15 = False
    var seen_anima = False
    var seen_lens = False
    var seen_zimage_l2p = False
    var seen_lance = False
    var seen_ernie = False
    for i in range(default_manifest_count()):
        var manifest = default_manifest_at(i)
        _check_profile(manifest)
        _print_manifest(manifest)
        var status = validate_manifest_paths(manifest)
        print(
            "[manifest] paths",
            manifest.model_id,
            "checked/missing:",
            status.checked,
            status.missing,
        )
        if status.missing != 0:
            raise Error(String("manifest smoke missing registered path for ") + manifest.model_id)
        if manifest.model_id == "sensenova_u1":
            seen_sensenova = True
        if manifest.model_id == "hidream_o1":
            seen_hidream = True
        if manifest.model_id == "flux1_dev":
            seen_flux = True
        if manifest.model_id == "sdxl":
            seen_sdxl = True
        if manifest.model_id == "sd3_5_large":
            seen_sd3 = True
        if manifest.model_id == "sd3_5_medium":
            seen_sd3_medium = True
        if manifest.model_id == "qwen_image":
            seen_qwen = True
        if manifest.model_id == "qwen_image_edit":
            seen_qwen_edit = True
        if manifest.model_id == "chroma":
            seen_chroma = True
        if manifest.model_id == "sd15":
            seen_sd15 = True
        if manifest.model_id == "anima":
            seen_anima = True
        if manifest.model_id == "microsoft_lens":
            seen_lens = True
        if manifest.model_id == "zimage_l2p":
            seen_zimage_l2p = True
        if manifest.model_id == "lance_t2v":
            seen_lance = True
        if manifest.model_id == "ernie_image":
            seen_ernie = True

    if not seen_sensenova:
        raise Error("manifest smoke missing sensenova_u1")
    if not seen_hidream:
        raise Error("manifest smoke missing hidream_o1")
    if not seen_flux:
        raise Error("manifest smoke missing flux1_dev")
    if not seen_sdxl:
        raise Error("manifest smoke missing sdxl")
    if not seen_sd3:
        raise Error("manifest smoke missing sd3_5_large")
    if not seen_sd3_medium:
        raise Error("manifest smoke missing sd3_5_medium")
    if not seen_qwen:
        raise Error("manifest smoke missing qwen_image")
    if not seen_qwen_edit:
        raise Error("manifest smoke missing qwen_image_edit")
    if not seen_chroma:
        raise Error("manifest smoke missing chroma")
    if not seen_sd15:
        raise Error("manifest smoke missing sd15")
    if not seen_anima:
        raise Error("manifest smoke missing anima")
    if not seen_lens:
        raise Error("manifest smoke missing microsoft_lens")
    if not seen_zimage_l2p:
        raise Error("manifest smoke missing zimage_l2p")
    if not seen_lance:
        raise Error("manifest smoke missing lance_t2v")
    if not seen_ernie:
        raise Error("manifest smoke missing ernie_image")

    var sense = default_manifest_by_id(String("sensenova_u1"))
    var hidream = default_manifest_by_id(String("hidream_o1"))
    var sd3 = default_manifest_by_id(String("sd3_5_large"))
    var sd3_medium = default_manifest_by_id(String("sd3_5_medium"))
    var qwen = default_manifest_by_id(String("qwen_image"))
    var qwen_edit = default_manifest_by_id(String("qwen_image_edit"))
    var chroma = default_manifest_by_id(String("chroma"))
    var sd15 = default_manifest_by_id(String("sd15"))
    var anima = default_manifest_by_id(String("anima"))
    var lens = default_manifest_by_id(String("microsoft_lens"))
    var l2p = default_manifest_by_id(String("zimage_l2p"))
    var lance = default_manifest_by_id(String("lance_t2v"))
    var ernie = default_manifest_by_id(String("ernie_image"))
    if sense.image_tokens != 4096 or sense.patch_size != 32:
        raise Error("sensenova_u1 profile mismatch")
    if hidream.image_tokens != 4096 or hidream.patch_size != 32:
        raise Error("hidream_o1 profile mismatch")
    if sd3.latent_channels != 16 or sd3.patch_size != 2:
        raise Error("sd3_5_large profile mismatch")
    if sd3.text_tokens != 410 or sd3.total_sequence != 4506:
        raise Error("sd3_5_large text/sequence mismatch")
    if sd3_medium.latent_channels != 16 or sd3_medium.patch_size != 2:
        raise Error("sd3_5_medium profile mismatch")
    if sd3_medium.text_tokens != 410 or sd3_medium.total_sequence != 4506:
        raise Error("sd3_5_medium text/sequence mismatch")
    if qwen.latent_channels != 16 or qwen.text_tokens != 1024:
        raise Error("qwen_image profile mismatch")
    if qwen.image_tokens != 4096 or qwen.total_sequence != 5120:
        raise Error("qwen_image sequence mismatch")
    if qwen_edit.latent_channels != 16 or qwen_edit.text_tokens != 1024:
        raise Error("qwen_image_edit profile mismatch")
    if qwen_edit.image_tokens != 8192 or qwen_edit.total_sequence != 9216:
        raise Error("qwen_image_edit sequence mismatch")
    if chroma.latent_channels != 16 or chroma.text_tokens != 512:
        raise Error("chroma profile mismatch")
    if chroma.image_tokens != 4096 or chroma.total_sequence != 4608:
        raise Error("chroma sequence mismatch")
    if sd15.default_width != 512 or sd15.latent_channels != 4:
        raise Error("sd15 profile mismatch")
    if sd15.image_tokens != 4096 or sd15.text_tokens != 77:
        raise Error("sd15 token mismatch")
    if anima.latent_channels != 16 or anima.text_tokens != 256:
        raise Error("anima profile mismatch")
    if lens.latent_channels != 32 or lens.text_tokens != 415:
        raise Error("microsoft_lens profile mismatch")
    if l2p.uses_vae() or l2p.patch_size != 16 or l2p.latent_downsample_s != 1:
        raise Error("zimage_l2p profile mismatch")
    if lance.default_frames != 9 or lance.image_tokens != 768:
        raise Error("lance_t2v profile mismatch")
    if ernie.latent_channels != 128 or ernie.text_tokens != 256:
        raise Error("ernie_image profile mismatch")
    if ernie.image_tokens != 4096 or ernie.total_sequence != 4352:
        raise Error("ernie_image sequence mismatch")

    print("[manifest] smoke steps=", cfg.steps, "offload=", cfg.offload.name())
    var all_status = validate_registered_manifest_paths()
    print(
        "[manifest] registered paths checked/missing:",
        all_status.checked,
        all_status.missing,
    )
    if all_status.missing != 0:
        raise Error("manifest smoke has missing registered paths")
    print("[manifest] guard strict=", guard.strict())
    print("[manifest] requests video=", req_i.is_video(), req_v.is_video())
