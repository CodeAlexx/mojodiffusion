# Metadata-only negative/positive gates for SCAIL-2 sidecar preflight.
# Usage: scail2_manifest_preflight_gate <temporary-dir>

from std.sys import argv

from serenitymojo.components.artifacts import shell_quote
from serenitymojo.io.ffi import sys_system
from serenitymojo.models.scail2.scail2_manifest import (
    preflight_scail2_animation_manifests,
    preflight_scail2_decode_manifest,
    validate_scail2_checksum_manifest,
    write_scail2_animation_manifest,
)
from serenitymojo.models.scail2.scail2_fp8_stream import (
    SCAIL2_BLOCKS,
    scail2_block_cache_path,
)
from serenitymojo.serve.product_manifest import write_text_file


def _expect_decode_rejected(latent_path: String, label: String) raises:
    var rejected = False
    try:
        preflight_scail2_decode_manifest(latent_path)
    except:
        rejected = True
    if not rejected:
        raise Error(String("SCAIL-2 preflight accepted ") + label)


def _gate_animation_checksum_bindings(root: String) raises:
    var cache_dir = root + String("/cache")
    if sys_system(String("mkdir -p -- ") + shell_quote(cache_dir)) != 0:
        raise Error("SCAIL-2 manifest fixture cache creation failed")
    var binary = root + String("/animation")
    var stage = root + String("/stage.safetensors")
    var reference = root + String("/reference.safetensors")
    var pose = root + String("/pose.safetensors")
    var text = root + String("/text.safetensors")
    var clip = root + String("/clip.safetensors")
    var output = root + String("/output.safetensors")
    var paths = [binary, stage, reference, pose, text, clip, output]
    for path in paths:
        write_text_file(path, String("fixture\n"))
    var sidecars = [
        stage + String(".scail2_stage.json"),
        reference + String(".scail2_condition.json"),
        pose + String(".scail2_condition.json"),
        text + String(".scail2_prompt.json"),
        clip + String(".scail2_condition.json"),
    ]
    for sidecar in sidecars:
        write_text_file(sidecar, String("{}\n"))
        write_text_file(sidecar + String(".sha256"), String("fixture\n"))
    write_text_file(
        cache_dir + String("/source_provenance.sha256"), String("fixture\n"),
    )
    write_text_file(
        cache_dir + String("/shared.safetensors.sha256"), String("fixture\n"),
    )
    for bi in range(SCAIL2_BLOCKS):
        write_text_file(
            scail2_block_cache_path(cache_dir, bi) + String(".sha256"),
            String("fixture\n"),
        )
    var manifest = write_scail2_animation_manifest(
        binary, stage, reference, pose, text, clip, cache_dir, output,
        1, UInt64(42), String("animation"), String("-"),
    )
    var exact_json = (
        String("test $(grep -Ec '")
        + String("block_[0-9][0-9]\\.safetensors\\.sha256")
        + String("' -- ") + shell_quote(manifest) + String(") -eq 40")
    )
    var exact_checksum = (
        String("test $(grep -Ec '")
        + String("block_[0-9][0-9]\\.safetensors\\.sha256$")
        + String("' -- ") + shell_quote(manifest + String(".sha256"))
        + String(") -eq 40")
    )
    if sys_system(exact_json) != 0 or sys_system(exact_checksum) != 0:
        raise Error("SCAIL-2 run provenance did not bind exactly 40 block checksums")


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: scail2_manifest_preflight_gate <temporary-dir>")
    var root = String(args[1])
    var latent_path = root + String("/latent.safetensors")
    var run_manifest = latent_path + String(".scail2_run.json")
    var checksum_path = run_manifest + String(".sha256")

    write_text_file(latent_path, String("latent-v1\n"))
    write_text_file(run_manifest, String("{\"schema\":\"test\"}\n"))
    var animation_rejected = False
    try:
        preflight_scail2_animation_manifests(
            root + String("/missing-stage"),
            root + String("/missing-reference"),
            root + String("/missing-pose"),
            root + String("/missing-text"),
            root + String("/missing-clip"),
            root + String("/missing-cache"),
        )
    except:
        animation_rejected = True
    if not animation_rejected:
        raise Error("SCAIL-2 animation preflight accepted missing sidecars")
    _expect_decode_rejected(latent_path, String("a missing checksum manifest"))

    write_text_file(checksum_path, String("not-a-sha256\n"))
    _expect_decode_rejected(latent_path, String("a malformed checksum manifest"))

    var publish = (
        String("sha256sum -- ") + shell_quote(run_manifest) + String(" ")
        + shell_quote(latent_path) + String(" > ") + shell_quote(checksum_path)
    )
    if sys_system(publish) != 0:
        raise Error("SCAIL-2 preflight fixture checksum publication failed")
    if not validate_scail2_checksum_manifest(run_manifest):
        raise Error("SCAIL-2 valid checksum manifest was rejected")
    preflight_scail2_decode_manifest(latent_path)

    write_text_file(latent_path, String("latent-corrupted\n"))
    _expect_decode_rejected(latent_path, String("a checksum-bound corrupted input"))
    _gate_animation_checksum_bindings(root)
    print("GATE PASS SCAIL-2 no-CUDA manifest preflight")
