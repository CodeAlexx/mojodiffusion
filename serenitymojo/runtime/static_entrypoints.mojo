# static_entrypoints.mojo - compile-only static entrypoint contracts.
#
# SenseNova and HiDream need finite comptime shape dispatch. This module turns
# the lower-level specialization table into family-facing contracts without
# importing model math or checkpoint loaders.

from serenitymojo.runtime.model_manifest import ModelFamily
from serenitymojo.runtime.request import GenerationRequest
from serenitymojo.runtime.static_dispatch import (
    StaticSpecialization,
    find_static_specialization,
    static_specialization_at,
    static_specialization_count,
)


@fieldwise_init
struct StaticEntrypointContract(Copyable, Movable, ImplicitlyCopyable):
    var model_id: String
    var profile_name: String
    var static_type_name: String
    var entry_kind: String
    var wrapper_name: String
    var declared_entry_path: String
    var smoke_entry_path: String
    var planned_production_entry_path: String
    var width: Int
    var height: Int
    var frames: Int
    var image_tokens: Int
    var text_tokens: Int
    var total_sequence: Int
    var patch_size: Int
    var pixel_space: Bool
    var has_vae: Bool
    var supports_cfg: Bool
    var requires_prompt_padding: Bool
    var requires_common_cfg_sequence: Bool
    var runnable_smoke: Bool
    var production_ready: Bool

    def is_smoke(self) -> Bool:
        return self.entry_kind == "smoke"

    def shape_matches(self, width: Int, height: Int, frames: Int) -> Bool:
        return (
            self.width == width
            and self.height == height
            and self.frames == frames
        )

    def text_bucket_matches(self, text_tokens: Int) -> Bool:
        return self.text_tokens == text_tokens


def _grid_tokens(width: Int, height: Int, patch_size: Int) raises -> Int:
    if patch_size <= 0:
        raise Error("static entrypoint: patch_size must be positive")
    if width % patch_size != 0 or height % patch_size != 0:
        raise Error("static entrypoint: image size must divide by patch_size")
    return (width // patch_size) * (height // patch_size)


def _kind_for_profile(profile_name: String, smoke_name: String) -> String:
    if profile_name == smoke_name:
        return String("smoke")
    return String("production")


def _runnable_smoke(profile_name: String, smoke_name: String) -> Bool:
    return profile_name == smoke_name


def _sensenova_contract(spec: StaticSpecialization) -> StaticEntrypointContract:
    return StaticEntrypointContract(
        spec.model_id,
        spec.profile_name,
        spec.entry_name,
        _kind_for_profile(spec.profile_name, String("sensenova_u1_64_text18")),
        String("sensenova_u1_static_entry"),
        spec.pipeline_path,
        String("serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo"),
        String("serenitymojo/pipeline/sensenova_u1_pipeline.mojo"),
        spec.width,
        spec.height,
        spec.frames,
        spec.image_tokens,
        spec.text_tokens,
        spec.total_sequence,
        32,
        True,
        False,
        True,
        True,
        False,
        _runnable_smoke(spec.profile_name, String("sensenova_u1_64_text18")),
        False,
    )


def _hidream_contract(spec: StaticSpecialization) -> StaticEntrypointContract:
    return StaticEntrypointContract(
        spec.model_id,
        spec.profile_name,
        spec.entry_name,
        _kind_for_profile(spec.profile_name, String("hidream_o1_64_s20")),
        String("hidream_o1_static_entry"),
        spec.pipeline_path,
        String("serenitymojo/pipeline/hidream_o1_smoke.mojo"),
        String("serenitymojo/pipeline/hidream_o1_pipeline.mojo"),
        spec.width,
        spec.height,
        spec.frames,
        spec.image_tokens,
        spec.text_tokens,
        spec.total_sequence,
        32,
        True,
        False,
        True,
        True,
        True,
        _runnable_smoke(spec.profile_name, String("hidream_o1_64_s20")),
        False,
    )


def static_entrypoint_count() -> Int:
    return static_specialization_count()


def static_entrypoint_at(index: Int) raises -> StaticEntrypointContract:
    var spec = static_specialization_at(index)
    if spec.model_id == "sensenova_u1":
        return _sensenova_contract(spec)
    if spec.model_id == "hidream_o1":
        return _hidream_contract(spec)
    raise Error(
        String("static entrypoint: unsupported model family ") + spec.model_id
    )


def find_static_entrypoint(
    model_id: String, profile_name: String
) raises -> StaticEntrypointContract:
    var spec = find_static_specialization(model_id, profile_name)
    if model_id == "sensenova_u1":
        return _sensenova_contract(spec)
    if model_id == "hidream_o1":
        return _hidream_contract(spec)
    raise Error(String("static entrypoint: unsupported model family ") + model_id)


def validate_static_entrypoint(contract: StaticEntrypointContract) raises:
    if contract.width <= 0 or contract.height <= 0:
        raise Error("static entrypoint: invalid image size")
    if contract.frames != 1:
        raise Error("static entrypoint: SenseNova/HiDream contracts are T2I only")
    if contract.total_sequence != contract.image_tokens + contract.text_tokens:
        raise Error("static entrypoint: total_sequence does not match tokens")
    if (
        _grid_tokens(contract.width, contract.height, contract.patch_size)
        != contract.image_tokens
    ):
        raise Error("static entrypoint: image token grid mismatch")
    if not contract.pixel_space or contract.has_vae:
        raise Error("static entrypoint: SenseNova/HiDream must stay pixel-space")
    if contract.model_id == "sensenova_u1":
        if contract.requires_common_cfg_sequence:
            raise Error(
                "static entrypoint: SenseNova should not require shared CFG S"
            )
    elif contract.model_id == "hidream_o1":
        if not contract.requires_common_cfg_sequence:
            raise Error("static entrypoint: HiDream CFG needs common static S")
    else:
        raise Error(
            String("static entrypoint: unsupported model family ")
            + contract.model_id
        )


def validate_request_for_static_entrypoint(
    request: GenerationRequest, contract: StaticEntrypointContract
) raises:
    validate_static_entrypoint(contract)
    if request.model_id != contract.model_id:
        raise Error("static entrypoint: request model does not match contract")
    if request.family != ModelFamily.text_to_image():
        raise Error("static entrypoint: SenseNova/HiDream requests must be T2I")
    if not contract.shape_matches(request.width, request.height, request.frames):
        raise Error("static entrypoint: request shape does not match contract")
