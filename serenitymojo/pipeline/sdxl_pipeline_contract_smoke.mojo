# SDXL cached-embedding pipeline contract smoke.
#
# This is intentionally metadata/scalar-only: no DeviceContext, no UNet/VAE
# imports, and no tensor allocations. It keeps the SDXL cached-embedding entry
# contract compile-checked. The one-step runtime smoke exists in
# `sdxl_pipeline_smoke.mojo`; the full 30-step quality/parity run is still open.

from serenitymojo.models.dit.sdxl_contract import (
    sdxl_cached_embedding_generator_command,
    sdxl_default_cached_embeddings_path,
    validate_sdxl_cached_embedding_header,
)
from serenitymojo.registry.checkpoints import (
    default_manifest_by_id,
    path_exists,
    validate_manifest_paths,
)
from serenitymojo.runtime.model_manifest import ModelFamily, ModelManifest
from serenitymojo.sampling.sdxl_euler import SDXLEulerScheduler


comptime WIDTH = 1024
comptime HEIGHT = 1024
comptime LATENT_DOWNSAMPLE = 8
comptime LATENT_H = HEIGHT // LATENT_DOWNSAMPLE
comptime LATENT_W = WIDTH // LATENT_DOWNSAMPLE
comptime LATENT_CHANNELS = 4
comptime TEXT_TOKENS = 77
comptime NUM_STEPS = 30
comptime CFG_SCALE = Float32(7.5)


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _check_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(String("SDXL contract bool mismatch: ") + name)


def _check_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            String("SDXL contract int mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _check_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(
            String("SDXL contract string mismatch: ")
            + name
            + String(" got=")
            + got
            + String(" expected=")
            + expected
        )


def _check_close(name: String, got: Float64, expected: Float64) raises:
    if _abs(got - expected) > 1.0e-5:
        raise Error(
            String("SDXL contract float mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _check_manifest(manifest: ModelManifest) raises:
    _check_string(String("model_id"), manifest.model_id, String("sdxl"))
    _check_bool(
        String("family"),
        manifest.family == ModelFamily.text_to_image(),
        True,
    )
    _check_string(
        String("variant"), manifest.variant, String("sdxl-base-1024-bf16")
    )
    _check_string(String("profile"), manifest.profile_name, String("sdxl_1024"))
    _check_string(
        String("entry"),
        manifest.production_entry,
        String("serenitymojo/pipeline/sdxl_pipeline_smoke.mojo"),
    )
    _check_int(String("width"), manifest.default_width, WIDTH)
    _check_int(String("height"), manifest.default_height, HEIGHT)
    _check_int(String("frames"), manifest.default_frames, 1)
    _check_int(String("latent_channels"), manifest.latent_channels, LATENT_CHANNELS)
    _check_int(String("latent_downsample_s"), manifest.latent_downsample_s, 8)
    _check_int(String("latent_height"), manifest.latent_height(), LATENT_H)
    _check_int(String("latent_width"), manifest.latent_width(), LATENT_W)
    _check_int(String("image_tokens"), manifest.image_tokens, LATENT_H * LATENT_W)
    _check_int(String("text_tokens"), manifest.text_tokens, TEXT_TOKENS)
    _check_int(
        String("total_sequence"),
        manifest.total_sequence,
        LATENT_H * LATENT_W + TEXT_TOKENS,
    )
    _check_bool(String("uses_vae"), manifest.uses_vae(), True)

    var status = validate_manifest_paths(manifest)
    _check_int(String("manifest path checks"), status.checked, 8)
    _check_int(String("manifest missing paths"), status.missing, 0)
    print("[sdxl-contract] manifest paths checked/missing:", status.checked, status.missing)


def _check_scheduler_contract() raises:
    var sched = SDXLEulerScheduler(NUM_STEPS)
    _check_int(String("steps"), sched.num_steps, NUM_STEPS)
    _check_close(String("cfg"), Float64(CFG_SCALE), 7.5)
    _check_close(String("timestep[0]"), Float64(sched.timestep(0)), 958.0)
    _check_close(String("sigma[0]"), Float64(sched.sigma(0)), 11.47684646)
    _check_close(String("sigma[last]"), Float64(sched.sigma(NUM_STEPS)), 0.0)
    _check_close(
        String("initial_noise_sigma"),
        Float64(sched.initial_noise_sigma()),
        11.52033006,
    )


def _check_cached_embedding_contract() raises -> Bool:
    var emb_path = sdxl_default_cached_embeddings_path()
    if not path_exists(emb_path):
        print("[sdxl-contract] cached embeddings missing; strict run blocked:")
        print("[sdxl-contract] ", emb_path)
        print("[sdxl-contract] generator handoff:")
        print("[sdxl-contract] ", sdxl_cached_embedding_generator_command(emb_path))
        return False

    validate_sdxl_cached_embedding_header(emb_path)
    print("[sdxl-contract] cached embedding artifact PASS")
    return True


def main() raises:
    var manifest = default_manifest_by_id(String("sdxl"))
    _check_manifest(manifest)
    _check_scheduler_contract()
    var embeddings_checked = _check_cached_embedding_contract()
    if embeddings_checked:
        print("SDXL cached-embedding pipeline contract PASS")
    else:
        print("SDXL static pipeline contract PASS; cached embedding schema SKIPPED")
