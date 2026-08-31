# MiniMax H3 first-slice training/preflight contract.
#
# This is intentionally not a runnable trainer.  It records the first policy
# boundary needed by the future Mojo-native product path. It represents the
# schedule/loss launch constraints below and binds the cache-manifest validator;
# artifact consumption and device/model execution remain separate gates.
#
# Oracle (development evidence only):
#   kohya-ss/musubi-tuner
#   commit b8717864713c9e4e7ef3d56eba1fc695a9b626a5
#   src/musubi_tuner/minimax_h3_train_network.py::_runtime_batch_plan

from std.math import isfinite

from serenitymojo.training.minimax_h3.schedule import (
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_VIDEO_SHIFT,
)
from serenitymojo.training.minimax_h3.cache_manifest import (
    MINIMAX_H3_CACHE_ORACLE_COMMIT,
    MiniMaxH3CacheManifest,
    validate_minimax_h3_cache_manifest,
)


comptime MINIMAX_H3_ORACLE_COMMIT = MINIMAX_H3_CACHE_ORACLE_COMMIT
comptime MINIMAX_H3_INTAKE_DATASET_IDENTITY = "eri_with_trigger"


@fieldwise_init
struct MiniMaxH3TrainingContract(Copyable, Movable):
    var dataset_identity: String
    var dataset_path: String
    var dataset_fingerprint: String
    var video_vae_fingerprint: String
    var audio_vae_fingerprint: String
    var text_encoder_fingerprint: String
    var processor_fingerprint: String
    var task: String
    var one_frame: Bool
    var batch_size: Int
    var gradient_accumulation_steps: Int
    var full_finetune: Bool
    var timestep_sampling: String
    var weighting_scheme: String
    var discrete_flow_shift: Float32
    var operating_bf16: Bool
    var lora_parameter_storage: String
    var lora_compute_bf16: Bool
    var lora_gradient_storage: String
    var optimizer_state_storage: String
    var base_storage: String  # bf16 or convrot_int8; base remains frozen
    var base_frozen: Bool
    var video_shift: Float32
    var audio_shift: Float32
    var audio_loss_weight: Float32
    var video_only: Bool

    @staticmethod
    def intake_default() -> MiniMaxH3TrainingContract:
        """Evidence receipt: exact dataset identity, path deliberately unresolved."""
        return MiniMaxH3TrainingContract(
            String(MINIMAX_H3_INTAKE_DATASET_IDENTITY),
            String(""),
            String(""),
            String(""),
            String(""),
            String(""),
            String(""),
            String(""),
            False,
            1,
            1,
            False,
            String("uniform"),
            String("none"),
            Float32(1.0),
            True,
            String("f32"),
            True,
            String("f32"),
            String("f32"),
            String("bf16"),
            True,
            MINIMAX_H3_VIDEO_SHIFT,
            MINIMAX_H3_AUDIO_SHIFT,
            Float32(1.0),
            False,
        )


def validate_minimax_h3_policy(contract: MiniMaxH3TrainingContract) raises:
    """Validate represented Musubi policy without pretending launch readiness."""
    if contract.dataset_identity.byte_length() == 0:
        raise Error("MiniMax H3 dataset identity must be nonempty")
    if (
        contract.task != String("t2va")
        and contract.task != String("fl2va")
        and contract.task != String("ref2va")
    ):
        raise Error("MiniMax H3 task must be t2va, fl2va, or ref2va")
    if contract.one_frame and contract.task == String("ref2va"):
        raise Error("MiniMax H3 one-frame training supports t2va or fl2va only")
    if contract.batch_size != 1:
        raise Error(
            "MiniMax H3 R1 requires batch_size=1; use gradient accumulation"
        )
    if contract.gradient_accumulation_steps <= 0:
        raise Error("MiniMax H3 gradient accumulation steps must be positive")
    if contract.full_finetune:
        raise Error(
            "MiniMax H3 R1 full finetune is not implemented; LoRA is required"
        )
    if contract.timestep_sampling != String("uniform"):
        raise Error("MiniMax H3 timestep sampling must be uniform")
    if contract.weighting_scheme != String("none"):
        raise Error("MiniMax H3 weighting scheme must be none")
    if contract.discrete_flow_shift != Float32(1.0):
        raise Error(
            "MiniMax H3 discrete_flow_shift must be 1.0; use the two H3 shifts"
        )
    if not contract.operating_bf16 or not contract.lora_compute_bf16:
        raise Error("MiniMax H3 transformer and LoRA projection compute must be BF16")
    if contract.lora_parameter_storage != String("f32"):
        raise Error("MiniMax H3 LoRA parameter storage must be f32")
    if contract.lora_gradient_storage != String("f32"):
        raise Error("MiniMax H3 accumulated LoRA gradient storage must be f32")
    if contract.optimizer_state_storage != String("f32"):
        raise Error("MiniMax H3 optimizer state storage must be f32")
    if (
        contract.base_storage != String("bf16")
        and contract.base_storage != String("convrot_int8")
    ):
        raise Error("MiniMax H3 frozen base storage must be bf16 or convrot_int8")
    if not contract.base_frozen:
        raise Error("MiniMax H3 R1 base weights must remain frozen")
    if (
        not isfinite(contract.video_shift)
        or contract.video_shift < Float32(0.01)
        or contract.video_shift > Float32(100.0)
        or not isfinite(contract.audio_shift)
        or contract.audio_shift < Float32(0.01)
        or contract.audio_shift > Float32(100.0)
    ):
        raise Error(
            "MiniMax H3 video/audio sigma shifts must be finite and in [0.01,100]"
        )
    if (
        not isfinite(contract.audio_loss_weight)
        or contract.audio_loss_weight < Float32(0.0)
    ):
        raise Error("MiniMax H3 audio loss weight must be finite and nonnegative")


def validate_minimax_h3_launch_preflight(
    contract: MiniMaxH3TrainingContract
) raises:
    """Fail before CUDA when no verified cache-manifest receipt is supplied."""
    validate_minimax_h3_policy(contract)
    if contract.base_storage != String("bf16"):
        raise Error(
            "MiniMax H3 launch requires the released mixed BF16/F32 base;"
            " convrot_int8 has no training backward"
        )
    if contract.dataset_path.byte_length() == 0:
        raise Error(
            String("MiniMax H3 dataset path for ") + contract.dataset_identity
            + String(" is unresolved; refusing launch")
        )
    raise Error(
        "MiniMax H3 verified cache manifest was not supplied; refusing launch"
    )


def validate_minimax_h3_launch_preflight(
    contract: MiniMaxH3TrainingContract,
    manifest: MiniMaxH3CacheManifest,
) raises:
    """Validate the complete cache receipt before any DeviceContext exists."""
    validate_minimax_h3_policy(contract)
    if contract.base_storage != String("bf16"):
        raise Error(
            "MiniMax H3 launch requires the released mixed BF16/F32 base;"
            " convrot_int8 has no training backward"
        )
    if contract.dataset_path.byte_length() == 0:
        raise Error("MiniMax H3 canonical dataset path is unresolved; refusing launch")
    if contract.dataset_fingerprint.byte_length() == 0:
        raise Error("MiniMax H3 dataset fingerprint is unresolved; refusing launch")
    if (
        contract.video_vae_fingerprint.byte_length() == 0
        or contract.audio_vae_fingerprint.byte_length() == 0
        or contract.text_encoder_fingerprint.byte_length() == 0
        or contract.processor_fingerprint.byte_length() == 0
    ):
        raise Error("MiniMax H3 cache model/processor fingerprints are unresolved; refusing launch")
    validate_minimax_h3_cache_manifest(
        manifest,
        contract.dataset_identity,
        contract.dataset_path,
        contract.dataset_fingerprint,
        contract.video_vae_fingerprint,
        contract.audio_vae_fingerprint,
        contract.text_encoder_fingerprint,
        contract.processor_fingerprint,
        contract.task,
        contract.one_frame,
    )
