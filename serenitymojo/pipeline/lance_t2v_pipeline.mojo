# lance_t2v_pipeline.mojo - production-entry contract for Lance T2V.
#
# This is intentionally a contract scaffold, not a second implementation of the
# heavy denoise/decode loop. It validates the static profile that a production
# Lance entrypoint must satisfy before dispatching to a specialized build target.

from serenitymojo.components.artifacts import ffmpeg_frame_pattern
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.runtime.execution_config import (
    ExecutionConfig,
    OffloadMode,
    default_quality_config,
)
from serenitymojo.runtime.model_manifest import (
    ModelFamily,
    ModelManifest,
    lance_t2v_default_manifest,
)
from serenitymojo.runtime.request import GenerationRequest, default_t2v_request


@fieldwise_init
struct LanceT2VRunProfile(Movable):
    var width: Int
    var height: Int
    var frames: Int
    var latent_t: Int
    var latent_h: Int
    var latent_w: Int
    var latent_tokens: Int
    var patch_dim: Int
    var steps: Int
    var guidance_scale: Float32
    var frame_prefix: String
    var frame_suffix: String
    var mp4_path: String
    var static_target: String

    def decoded_frames(self) -> Int:
        return (self.latent_t - 1) * 4 + 1

    def frame_pattern(self) -> String:
        return ffmpeg_frame_pattern(self.frame_prefix, self.frame_suffix)


def validate_lance_t2v_contract(
    manifest: ModelManifest,
    request: GenerationRequest,
    config: ExecutionConfig,
) raises -> LanceT2VRunProfile:
    if manifest.model_id != "lance_t2v":
        raise Error(String("Lance T2V contract got model: ") + manifest.model_id)
    if manifest.family != ModelFamily.text_to_video():
        raise Error("Lance T2V contract requires text_to_video family")
    if not request.is_video():
        raise Error("Lance T2V contract requires a video request")
    if request.width != manifest.default_width or request.height != manifest.default_height:
        raise Error("Lance T2V contract currently supports only the manifest static size")
    if request.frames != manifest.default_frames:
        raise Error("Lance T2V contract currently supports only the manifest frame count")
    if request.steps <= 0:
        raise Error("Lance T2V contract requires steps > 0")
    if request.guidance_scale < Float32(1.0):
        raise Error("Lance T2V contract expects CFG scale >= 1")
    if config.offload == OffloadMode.resident():
        raise Error("Lance T2V contract requires block_stream or turbo_slots offload")
    if request.width % manifest.latent_downsample_s != 0:
        raise Error("Lance T2V width must divide by spatial latent downsample")
    if request.height % manifest.latent_downsample_s != 0:
        raise Error("Lance T2V height must divide by spatial latent downsample")
    if (request.frames - 1) % manifest.latent_downsample_t != 0:
        raise Error("Lance T2V frames must map exactly to Wan temporal decode")

    var latent_t = (request.frames - 1) // manifest.latent_downsample_t + 1
    var latent_h = request.height // manifest.latent_downsample_s
    var latent_w = request.width // manifest.latent_downsample_s
    var latent_tokens = latent_t * latent_h * latent_w
    if latent_tokens != manifest.image_tokens:
        raise Error("Lance T2V manifest image token count mismatch")

    return LanceT2VRunProfile(
        request.width,
        request.height,
        request.frames,
        latent_t,
        latent_h,
        latent_w,
        latent_tokens,
        manifest.latent_channels,
        request.steps,
        request.guidance_scale,
        request.output_path + String("_frame"),
        String("_256.png"),
        request.output_path + String(".mp4"),
        String("serenitymojo/pipeline/lance_t2v_256_9f_dense_probe.mojo"),
    )


def validate_lance_t2v_artifacts(profile: LanceT2VRunProfile) raises -> Int:
    var found = 0
    for i in range(profile.frames):
        var path = profile.frame_prefix + String(i) + profile.frame_suffix
        if not path_exists(path):
            raise Error(String("Lance T2V missing decoded frame: ") + path)
        found += 1
    if not path_exists(profile.mp4_path):
        raise Error(String("Lance T2V missing mp4: ") + profile.mp4_path)
    found += 1
    return found


def main() raises:
    var manifest = lance_t2v_default_manifest()
    var request = default_t2v_request(
        String("lance_t2v"),
        String("fairy"),
        String("/home/alex/mojodiffusion/output/lance_t2v_256_9f_dense"),
    )
    var config = default_quality_config()
    var profile = validate_lance_t2v_contract(manifest, request, config)
    print(
        "[lance-pipeline] profile",
        profile.width,
        "x",
        profile.height,
        "frames=",
        profile.frames,
        "latent=",
        profile.latent_t,
        profile.latent_h,
        profile.latent_w,
        "tokens=",
        profile.latent_tokens,
    )
    print("[lance-pipeline] frame pattern ->", profile.frame_pattern())
    print("[lance-pipeline] mp4 ->", profile.mp4_path)
    print("[lance-pipeline] static target ->", profile.static_target)
    if path_exists(profile.mp4_path):
        var artifacts = validate_lance_t2v_artifacts(profile)
        print("[lance-pipeline] decoded artifact gate ->", artifacts, "files")
    else:
        print("[lance-pipeline] decoded artifact gate pending ->", profile.mp4_path)
