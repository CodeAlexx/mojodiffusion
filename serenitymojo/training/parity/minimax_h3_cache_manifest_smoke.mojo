# Deterministic host-only MiniMax H3 cache-manifest contract smoke.
#
# Evidence level: deterministic smoke, not an artifact-consumer or tensor
# parity gate. No files, dataset, cache builder, or DeviceContext are touched.

from std.collections import List

from serenitymojo.training.minimax_h3.cache_manifest import (
    MINIMAX_H3_CACHE_MANIFEST_SCHEMA,
    MINIMAX_H3_LATENT_CACHE_FORMAT,
    MINIMAX_H3_TEXT_CACHE_FORMAT,
    MiniMaxH3CacheManifest,
    MiniMaxH3CacheSample,
    minimax_h3_cache_namespace,
    validate_minimax_h3_cache_manifest,
)
from serenitymojo.training.minimax_h3.contract import (
    MINIMAX_H3_INTAKE_DATASET_IDENTITY,
    MINIMAX_H3_ORACLE_COMMIT,
    MiniMaxH3TrainingContract,
    validate_minimax_h3_launch_preflight,
)


comptime SHA_A = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
comptime SHA_B = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
comptime SHA_C = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
comptime SHA_D = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax H3 cache manifest smoke failed: ") + message)


def _sample(sample_id: String) -> MiniMaxH3CacheSample:
    var suffix = String("a") if sample_id == String(SHA_A) else String("b")
    return MiniMaxH3CacheSample(
        sample_id,
        sample_id,
        suffix + String(".jpg"),
        suffix + String(".txt"),
        256,
        384,
        String("{\"/tmp/h3-fixture/dataset/") + suffix
            + String(".jpg\":\"stat:123:456\"}"),
        String(SHA_D),
        0,
        String("/tmp/h3-fixture/cache/") + suffix + String(".latent.safetensors"),
        String(SHA_C),
        String("/tmp/h3-fixture/cache/") + suffix + String(".text.safetensors"),
        String(SHA_D),
        22,
        [24, 7, 48, 32],
        [32, 2, 37],
        String("float32"),
        String("float32"),
        Float32(1.0),
        True,
        [37, 5120],
        String("bfloat16"),
        [37],
        String("int64"),
        True,
        False,
        False,
        False,
        0,
        0,
        0,
        0,
        -1,
        List[Int](),
    )


def _manifest(var samples: List[MiniMaxH3CacheSample]) -> MiniMaxH3CacheManifest:
    return MiniMaxH3CacheManifest(
        String(MINIMAX_H3_CACHE_MANIFEST_SCHEMA),
        String(MINIMAX_H3_ORACLE_COMMIT),
        String("/tmp/h3-fixture/cache/cache_manifest.json"),
        String(SHA_D),
        True,
        String("fixture_dataset"),
        String("/tmp/h3-fixture/dataset"),
        String(SHA_A),
        String("/tmp/h3-fixture/cache"),
        minimax_h3_cache_namespace(
            String("fixture_dataset"), String(SHA_A), String("t2va"), False,
        ),
        String("t2va"),
        False,
        0,
        String("bf16"),
        String(MINIMAX_H3_LATENT_CACHE_FORMAT),
        String(MINIMAX_H3_TEXT_CACHE_FORMAT),
        String(SHA_A),
        String(SHA_B),
        String(SHA_C),
        String(SHA_D),
        String(SHA_C),
        samples^,
    )


def _expect_rejects(manifest: MiniMaxH3CacheManifest, label: String) raises:
    var rejected = False
    try:
        validate_minimax_h3_cache_manifest(
            manifest,
            String("fixture_dataset"),
            String("/tmp/h3-fixture/dataset"),
            String(SHA_A),
            String(SHA_A),
            String(SHA_B),
            String(SHA_C),
            String(SHA_D),
            String("t2va"),
            False,
        )
    except error:
        rejected = True
        print("  expected rejection [", label, "]:", String(error))
    _check(rejected, label + String(" must reject"))


def main() raises:
    print("=== MiniMax H3 cache manifest deterministic smoke ===")
    _check(
        String(MINIMAX_H3_TEXT_CACHE_FORMAT) == String("minimax-h3-text-v2"),
        String("pinned text cache format v2"),
    )
    var samples = List[MiniMaxH3CacheSample]()
    samples.append(_sample(String(SHA_A)))
    samples.append(_sample(String(SHA_B)))
    var valid = _manifest(samples^)
    validate_minimax_h3_cache_manifest(
        valid,
        String("fixture_dataset"),
        String("/tmp/h3-fixture/dataset"),
        String(SHA_A),
        String(SHA_A),
        String(SHA_B),
        String(SHA_C),
        String(SHA_D),
        String("t2va"),
        False,
    )

    var wrong_identity = valid.copy()
    wrong_identity.dataset_identity = String("other_dataset")
    _expect_rejects(wrong_identity, String("dataset identity mismatch"))

    var wrong_path = valid.copy()
    wrong_path.canonical_dataset_path = String("/tmp/other-dataset")
    _expect_rejects(wrong_path, String("canonical path mismatch"))

    var wrong_fingerprint = valid.copy()
    wrong_fingerprint.dataset_fingerprint = String(SHA_B)
    _expect_rejects(wrong_fingerprint, String("dataset fingerprint mismatch"))

    var unverified = valid.copy()
    unverified.checksum_verified = False
    _expect_rejects(unverified, String("unverified manifest checksum"))

    var stale_text_format = valid.copy()
    stale_text_format.text_cache_format = String("minimax-h3-text-v1")
    _expect_rejects(stale_text_format, String("stale text cache format v1"))

    var duplicate_ids = valid.copy()
    duplicate_ids.samples[1].sample_id = String(SHA_A)
    duplicate_ids.samples[1].source_fingerprint = String(SHA_A)
    _expect_rejects(duplicate_ids, String("duplicate sample IDs"))

    var bad_text = valid.copy()
    bad_text.samples[0].text_hidden_shape[1] = 4096
    _expect_rejects(bad_text, String("wrong text width"))

    var bad_audio_presence = valid.copy()
    bad_audio_presence.samples[0].audio_present = Float32(0.5)
    _expect_rejects(bad_audio_presence, String("non-binary audio_present value"))

    var bad_condition = valid.copy()
    bad_condition.samples[0].has_first_condition = True
    _expect_rejects(bad_condition, String("t2va condition contamination"))

    var bad_frames = valid.copy()
    bad_frames.samples[0].source_frame_count = 23
    _expect_rejects(bad_frames, String("invalid 17*n+5 source frames"))

    var shared_artifact = valid.copy()
    shared_artifact.samples[1].latent_artifact_path = (
        shared_artifact.samples[0].latent_artifact_path
    )
    _expect_rejects(shared_artifact, String("cross-sample artifact sharing"))

    var product_contract = MiniMaxH3TrainingContract.intake_default()
    product_contract.dataset_identity = String("fixture_dataset")
    product_contract.dataset_path = String("/tmp/h3-fixture/dataset")
    product_contract.dataset_fingerprint = String(SHA_A)
    product_contract.video_vae_fingerprint = String(SHA_A)
    product_contract.audio_vae_fingerprint = String(SHA_B)
    product_contract.text_encoder_fingerprint = String(SHA_C)
    product_contract.processor_fingerprint = String(SHA_D)
    product_contract.task = String("t2va")
    validate_minimax_h3_launch_preflight(product_contract, valid)

    # The requested run is intentionally unresolved and must still fail before
    # a manifest is accepted. This does not inspect a similarly named dataset.
    var requested = MiniMaxH3TrainingContract.intake_default()
    requested.task = String("t2va")
    var requested_rejected = False
    try:
        validate_minimax_h3_launch_preflight(requested)
    except error:
        requested_rejected = True
        print(
            "  expected rejection [",
            MINIMAX_H3_INTAKE_DATASET_IDENTITY,
            " unresolved]:",
            String(error),
        )
    _check(requested_rejected, String("requested unresolved run must reject"))

    print("MiniMax H3 cache manifest deterministic SMOKE PASS")
