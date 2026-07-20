# Automatic SCAIL-2 development/product-bound result sidecars.

from std.collections import List

from serenitymojo.components.artifacts import shell_quote
from serenitymojo.io.ffi import O_RDONLY, sys_close, sys_open, sys_rename, sys_system
from serenitymojo.models.scail2.scail2_fp8_stream import (
    SCAIL2_BLOCKS,
    scail2_block_cache_path,
    validate_scail2_14b_fp8_cache,
)
from serenitymojo.serve.product_manifest import json_escape, write_text_file


comptime SCAIL2_SOURCE_COMMIT = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"
comptime SCAIL2_MODEL_REVISION = "150cc0ca4e98e50e60b9295dacde39442fdccab2"
comptime SCAIL2_CHECKPOINT_SHA256 = "d6c73e94c57eb36e6351c800d1228e41ed7e45db1ccf410dd875bcfdd2945e7f"


def scail2_path_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def validate_scail2_checksum_manifest(manifest_path: String) -> Bool:
    """Validate the manifest plus every file bound by its SHA-256 sidecar."""
    var checksum_path = manifest_path + String(".sha256")
    if not scail2_path_exists(manifest_path) or not scail2_path_exists(checksum_path):
        return False
    # write_scail2_checksums always records the manifest first.  Checking that
    # digest independently prevents a well-formed but unrelated checksum list
    # from being accepted as this manifest's provenance.
    var command = (
        String("expected=$(sed -n '1{s/[[:space:]].*//;p;q;}' -- ")
        + shell_quote(checksum_path)
        + String(") && actual=$(sha256sum -- ")
        + shell_quote(manifest_path)
        + String(" | cut -d ' ' -f 1) && test \"$actual\" = \"$expected\"")
        + String(" && sha256sum --check --status --strict -- ")
        + shell_quote(checksum_path)
    )
    return sys_system(command) == 0


def _require_scail2_checksum_manifest(
    manifest_path: String, kind: String,
) raises:
    if not validate_scail2_checksum_manifest(manifest_path):
        raise Error(
            String("SCAIL-2 ") + kind
            + String(" manifest/checksum is missing or invalid: ") + manifest_path
        )


def preflight_scail2_animation_manifests(
    stage_path: String,
    reference_path: String,
    pose_path: String,
    text_path: String,
    clip_path: String,
    cache_dir: String,
) raises:
    """Fail closed on all animation provenance before CUDA is initialized."""
    _require_scail2_checksum_manifest(
        stage_path + String(".scail2_stage.json"), String("stage"),
    )
    _require_scail2_checksum_manifest(
        reference_path + String(".scail2_condition.json"), String("reference"),
    )
    _require_scail2_checksum_manifest(
        pose_path + String(".scail2_condition.json"), String("pose"),
    )
    _require_scail2_checksum_manifest(
        text_path + String(".scail2_prompt.json"), String("prompt"),
    )
    _require_scail2_checksum_manifest(
        clip_path + String(".scail2_condition.json"), String("CLIP"),
    )
    # This metadata-only validator checks the pinned source provenance, all 40
    # block artifacts and their checksum sidecars, and the shared cache file.
    if not validate_scail2_14b_fp8_cache(cache_dir):
        raise Error(
            String("SCAIL-2 FP8 cache provenance/checksums are missing or invalid: ")
            + cache_dir
        )


def preflight_scail2_decode_manifest(latent_path: String) raises:
    """Validate the latent run sidecar and its complete checksum closure."""
    _require_scail2_checksum_manifest(
        latent_path + String(".scail2_run.json"), String("latent run"),
    )


def preflight_scail2_additional_manifest(additional_path: String) raises:
    _require_scail2_checksum_manifest(
        additional_path + String(".scail2_condition.json"),
        String("additional reference"),
    )


def _write_atomic(path: String, content: String) raises:
    var tmp = path + String(".tmp")
    write_text_file(tmp, content)
    if sys_rename(tmp, path) != 0:
        raise Error(String("SCAIL-2 manifest rename failed: ") + path)


def write_scail2_checksums(manifest_path: String, paths: List[String]) raises:
    """Atomically bind a manifest and its regular-file inputs/outputs by SHA-256."""
    var checksum_path = manifest_path + String(".sha256")
    var tmp = checksum_path + String(".tmp")
    var command = String("sha256sum -- ") + shell_quote(manifest_path)
    for path in paths:
        command += String(" ") + shell_quote(path)
    command += String(" > ") + shell_quote(tmp)
    command += String(" && mv -- ") + shell_quote(tmp) + String(" ") + shell_quote(checksum_path)
    if sys_system(command) != 0:
        raise Error(String("SCAIL-2 checksum manifest failed: ") + checksum_path)


def write_scail2_stage_manifest(
    binary_path: String,
    image_path: String,
    image_mask_path: String,
    pose_path: String,
    driving_mask_path: String,
    output_path: String,
    height: Int,
    width: Int,
    frames: Int,
    additional_images: List[String],
    additional_masks: List[String],
) raises -> String:
    var path = output_path + String(".scail2_stage.json")
    var content = String("{\n")
    content += String('  "schema":"serenity.scail2.stage.v1",\n')
    content += String('  "model":"scail2",\n')
    content += String('  "source_commit":"') + String(SCAIL2_SOURCE_COMMIT) + String('",\n')
    content += String('  "image":"') + json_escape(image_path) + String('",\n')
    content += String('  "image_mask":"') + json_escape(image_mask_path) + String('",\n')
    content += String('  "pose":"') + json_escape(pose_path) + String('",\n')
    content += String('  "driving_mask":"') + json_escape(driving_mask_path) + String('",\n')
    content += String('  "width":') + String(width) + String(",\n")
    content += String('  "height":') + String(height) + String(",\n")
    content += String('  "frames":') + String(frames) + String(",\n")
    content += String('  "additional_reference_images":[\n')
    for i in range(len(additional_images)):
        content += String('    "') + json_escape(additional_images[i]) + String('"')
        if i + 1 != len(additional_images):
            content += String(",")
        content += String("\n")
    content += String("  ],\n")
    content += String('  "additional_reference_masks":[\n')
    for i in range(len(additional_masks)):
        content += String('    "') + json_escape(additional_masks[i]) + String('"')
        if i + 1 != len(additional_masks):
            content += String(",")
        content += String("\n")
    content += String("  ],\n")
    content += String('  "output":"') + json_escape(output_path) + String('"\n')
    content += String("}\n")
    _write_atomic(path, content)
    var files = List[String]()
    files.append(binary_path)
    files.append(image_path)
    files.append(image_mask_path)
    files.append(pose_path)
    files.append(driving_mask_path)
    for path in additional_images:
        files.append(path)
    for path in additional_masks:
        files.append(path)
    files.append(output_path)
    write_scail2_checksums(path, files)
    return path^


def write_scail2_condition_manifest(
    binary_path: String,
    kind: String,
    stage_path: String,
    model_path: String,
    output_path: String,
) raises -> String:
    var stage_manifest = stage_path + String(".scail2_stage.json")
    if not scail2_path_exists(stage_manifest):
        raise Error(String("SCAIL-2 stage manifest missing: ") + stage_manifest)
    var path = output_path + String(".scail2_condition.json")
    var content = String("{\n")
    content += String('  "schema":"serenity.scail2.condition.v1",\n')
    content += String('  "model":"scail2",\n')
    content += String('  "kind":"') + json_escape(kind) + String('",\n')
    content += String('  "source_commit":"') + String(SCAIL2_SOURCE_COMMIT) + String('",\n')
    content += String('  "stage_manifest":"') + json_escape(stage_manifest) + String('",\n')
    content += String('  "stage":"') + json_escape(stage_path) + String('",\n')
    content += String('  "checkpoint":"') + json_escape(model_path) + String('",\n')
    content += String('  "output":"') + json_escape(output_path) + String('"\n')
    content += String("}\n")
    _write_atomic(path, content)
    var files = List[String]()
    files.append(binary_path)
    files.append(stage_path)
    files.append(stage_manifest)
    files.append(stage_manifest + String(".sha256"))
    files.append(model_path)
    files.append(output_path)
    write_scail2_checksums(path, files)
    return path^


def write_scail2_prompt_manifest(
    binary_path: String,
    model_dir: String,
    tokenizer_path: String,
    positive: String,
    negative: String,
    positive_valid: Int,
    negative_valid: Int,
    output_path: String,
) raises -> String:
    var path = output_path + String(".scail2_prompt.json")
    var content = String("{\n")
    content += String('  "schema":"serenity.scail2.prompt.v1",\n')
    content += String('  "model":"scail2",\n')
    content += String('  "source_commit":"') + String(SCAIL2_SOURCE_COMMIT) + String('",\n')
    content += String('  "positive":"') + json_escape(positive) + String('",\n')
    content += String('  "negative":"') + json_escape(negative) + String('",\n')
    content += String('  "positive_valid_tokens":') + String(positive_valid) + String(",\n")
    content += String('  "negative_valid_tokens":') + String(negative_valid) + String(",\n")
    content += String('  "umt5_model_dir":"') + json_escape(model_dir) + String('",\n')
    content += String('  "tokenizer":"') + json_escape(tokenizer_path) + String('",\n')
    content += String('  "conditioning":"') + json_escape(output_path) + String('"\n')
    content += String("}\n")
    _write_atomic(path, content)
    var files = List[String]()
    files.append(binary_path)
    files.append(tokenizer_path)
    files.append(output_path)
    var index_path = model_dir + String("/model.safetensors.index.json")
    if scail2_path_exists(index_path):
        files.append(index_path)
    write_scail2_checksums(path, files)
    return path^


def write_scail2_animation_manifest(
    binary_path: String,
    stage_path: String,
    reference_path: String,
    pose_path: String,
    text_path: String,
    clip_path: String,
    cache_dir: String,
    output_path: String,
    steps: Int,
    seed: UInt64,
    mode: String,
    additional_reference_path: String,
) raises -> String:
    var prompt_manifest = text_path + String(".scail2_prompt.json")
    if not scail2_path_exists(prompt_manifest):
        raise Error(String("SCAIL-2 prompt manifest missing: ") + prompt_manifest)
    var stage_manifest = stage_path + String(".scail2_stage.json")
    if not scail2_path_exists(stage_manifest):
        raise Error(String("SCAIL-2 stage manifest missing: ") + stage_manifest)
    var reference_manifest = reference_path + String(".scail2_condition.json")
    var pose_manifest = pose_path + String(".scail2_condition.json")
    var clip_manifest = clip_path + String(".scail2_condition.json")
    if not scail2_path_exists(reference_manifest):
        raise Error(String("SCAIL-2 reference manifest missing: ") + reference_manifest)
    if not scail2_path_exists(pose_manifest):
        raise Error(String("SCAIL-2 pose manifest missing: ") + pose_manifest)
    if not scail2_path_exists(clip_manifest):
        raise Error(String("SCAIL-2 CLIP manifest missing: ") + clip_manifest)
    var provenance = cache_dir + String("/source_provenance.sha256")
    if not scail2_path_exists(provenance):
        raise Error(String("SCAIL-2 cache provenance missing: ") + provenance)
    var shared_checksum = cache_dir + String("/shared.safetensors.sha256")
    if not scail2_path_exists(shared_checksum):
        raise Error(String("SCAIL-2 shared cache checksum missing: ") + shared_checksum)
    var path = output_path + String(".scail2_run.json")
    var content = String("{\n")
    content += String('  "schema":"serenity.scail2.animation.v1",\n')
    content += String('  "model":"scail2",\n')
    content += String('  "source_commit":"') + String(SCAIL2_SOURCE_COMMIT) + String('",\n')
    content += String('  "model_revision":"') + String(SCAIL2_MODEL_REVISION) + String('",\n')
    content += String('  "checkpoint_sha256":"') + String(SCAIL2_CHECKPOINT_SHA256) + String('",\n')
    content += String('  "mode":"') + json_escape(mode) + String('",\n')
    content += String('  "seed":') + String(seed) + String(",\n")
    content += String('  "steps":') + String(steps) + String(",\n")
    content += String('  "scheduler":"unipc",\n')
    content += String('  "shift":3.0,\n')
    content += String('  "cfg":5.0,\n')
    content += String('  "width":896,\n')
    content += String('  "height":512,\n')
    content += String('  "frames":65,\n')
    content += String('  "prompt_manifest":"') + json_escape(prompt_manifest) + String('",\n')
    content += String('  "stage_manifest":"') + json_escape(stage_manifest) + String('",\n')
    content += String('  "reference_manifest":"') + json_escape(reference_manifest) + String('",\n')
    content += String('  "pose_manifest":"') + json_escape(pose_manifest) + String('",\n')
    content += String('  "clip_manifest":"') + json_escape(clip_manifest) + String('",\n')
    content += String('  "stage":"') + json_escape(stage_path) + String('",\n')
    content += String('  "reference_latent":"') + json_escape(reference_path) + String('",\n')
    content += String('  "pose_latent":"') + json_escape(pose_path) + String('",\n')
    content += String('  "text_context":"') + json_escape(text_path) + String('",\n')
    content += String('  "clip_context":"') + json_escape(clip_path) + String('",\n')
    content += String('  "additional_reference_latent":"') + json_escape(additional_reference_path) + String('",\n')
    content += String('  "cache_provenance":"') + json_escape(provenance) + String('",\n')
    content += String('  "fp8_shared_checksum_sidecar":"') + json_escape(shared_checksum) + String('",\n')
    content += String('  "fp8_block_checksum_sidecars":[\n')
    for bi in range(SCAIL2_BLOCKS):
        var block_checksum = scail2_block_cache_path(cache_dir, bi) + String(".sha256")
        if not scail2_path_exists(block_checksum):
            raise Error(
                String("SCAIL-2 FP8 block checksum missing: ") + block_checksum
            )
        content += String('    "') + json_escape(block_checksum) + String('"')
        if bi + 1 != SCAIL2_BLOCKS:
            content += String(",")
        content += String("\n")
    content += String("  ],\n")
    content += String('  "latent":"') + json_escape(output_path) + String('"\n')
    content += String("}\n")
    _write_atomic(path, content)
    var files = List[String]()
    files.append(binary_path)
    files.append(stage_path)
    files.append(stage_manifest)
    files.append(stage_manifest + String(".sha256"))
    files.append(reference_path)
    files.append(reference_manifest)
    files.append(reference_manifest + String(".sha256"))
    files.append(pose_path)
    files.append(pose_manifest)
    files.append(pose_manifest + String(".sha256"))
    files.append(text_path)
    files.append(prompt_manifest)
    files.append(prompt_manifest + String(".sha256"))
    files.append(clip_path)
    files.append(clip_manifest)
    files.append(clip_manifest + String(".sha256"))
    if additional_reference_path != String("-"):
        var additional_manifest = additional_reference_path + String(".scail2_condition.json")
        files.append(additional_reference_path)
        files.append(additional_manifest)
        files.append(additional_manifest + String(".sha256"))
    files.append(provenance)
    files.append(shared_checksum)
    for bi in range(SCAIL2_BLOCKS):
        files.append(scail2_block_cache_path(cache_dir, bi) + String(".sha256"))
    files.append(output_path)
    write_scail2_checksums(path, files)
    return path^


def write_scail2_decode_manifest(
    binary_path: String,
    latent_path: String,
    vae_path: String,
    audio_source: String,
    has_audio: Bool,
    mp4_path: String,
) raises -> String:
    var run_manifest = latent_path + String(".scail2_run.json")
    if not scail2_path_exists(run_manifest):
        raise Error(String("SCAIL-2 latent run manifest missing: ") + run_manifest)
    var path = mp4_path + String(".scail2_result.json")
    var content = String("{\n")
    content += String('  "schema":"serenity.scail2.video_result.v1",\n')
    content += String('  "model":"scail2",\n')
    content += String('  "latent_manifest":"') + json_escape(run_manifest) + String('",\n')
    content += String('  "latent":"') + json_escape(latent_path) + String('",\n')
    content += String('  "vae":"') + json_escape(vae_path) + String('",\n')
    content += String('  "audio_source":"') + json_escape(audio_source) + String('",\n')
    content += String('  "has_audio":') + (String("true") if has_audio else String("false")) + String(",\n")
    content += String('  "codec":"h264",\n')
    content += String('  "width":896,\n')
    content += String('  "height":512,\n')
    content += String('  "frames":65,\n')
    content += String('  "fps":16,\n')
    content += String('  "output":"') + json_escape(mp4_path) + String('"\n')
    content += String("}\n")
    _write_atomic(path, content)
    var files = List[String]()
    files.append(binary_path)
    files.append(latent_path)
    files.append(run_manifest)
    files.append(run_manifest + String(".sha256"))
    files.append(vae_path)
    if audio_source != String("-"):
        files.append(audio_source)
    files.append(mp4_path)
    write_scail2_checksums(path, files)
    return path^
