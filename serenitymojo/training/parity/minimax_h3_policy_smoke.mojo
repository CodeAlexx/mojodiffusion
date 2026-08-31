# MiniMax H3 deterministic schedule/loss/preflight policy smoke.
#
# This is not a Musubi parity gate: expected values are intentionally visible
# and host-computed. A later parity gate must consume a versioned fixture emitted
# by the pinned Musubi environment.
#
# Lead-agent run (serialized with every other Mojo compile):
#   pixi run mojo run -I . serenitymojo/training/parity/minimax_h3_policy_smoke.mojo

from std.collections import List
from std.math import abs

from serenitymojo.training.minimax_h3.contract import (
    MINIMAX_H3_INTAKE_DATASET_IDENTITY,
    MINIMAX_H3_ORACLE_COMMIT,
    MiniMaxH3TrainingContract,
    validate_minimax_h3_launch_preflight,
    validate_minimax_h3_policy,
)
from serenitymojo.training.minimax_h3.loss import (
    minimax_h3_effective_audio_weight,
    minimax_h3_presence_gated_av_mse,
)
from serenitymojo.training.minimax_h3.schedule import (
    minimax_h3_native_targets,
    minimax_h3_noisy_values,
    minimax_h3_schedule_point,
    minimax_h3_shift_sigma,
)


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax H3 policy smoke failed: ") + message)


def _close(a: Float32, b: Float32, tolerance: Float32 = Float32(1.0e-6)) -> Bool:
    return abs(a - b) <= tolerance


def _check_values(
    got: List[Float32], expected: List[Float32], label: String
) raises:
    _check(len(got) == len(expected), label + String(" length"))
    for i in range(len(got)):
        _check(_close(got[i], expected[i]), label + String(" element ") + String(i))


def _expect_policy_rejects(contract: MiniMaxH3TrainingContract, label: String) raises:
    var rejected = False
    try:
        validate_minimax_h3_policy(contract)
    except e:
        rejected = True
        print("  expected rejection [", label, "]:", String(e))
    _check(rejected, label + String(" must reject"))


def _valid_fixture() -> MiniMaxH3TrainingContract:
    var contract = MiniMaxH3TrainingContract.intake_default()
    contract.task = String("t2va")
    return contract^


def main() raises:
    print("=== MiniMax H3 deterministic policy smoke ===")
    print("oracle receipt commit:", MINIMAX_H3_ORACLE_COMMIT)

    var point = minimax_h3_schedule_point(Float32(0.25))
    _check(_close(point.sigma_video, Float32(0.8)), "video shift 12")
    _check(_close(point.sigma_audio, Float32(0.5)), "audio shift 3")
    _check(_close(point.model_t_video, Float32(0.2)), "video model_t")
    _check(_close(point.model_t_audio, Float32(0.5)), "audio model_t")

    var video_latent: List[Float32] = [1.0, -2.0, 0.5]
    var video_noise: List[Float32] = [-1.0, 2.0, 1.5]
    _check_values(
        minimax_h3_noisy_values(video_latent, video_noise, point.sigma_video),
        [-0.6, 1.2, 1.3],
        "video noisy values",
    )
    _check_values(
        minimax_h3_native_targets(video_latent, video_noise),
        [2.0, -4.0, -1.0],
        "native latent-noise sign",
    )

    var present = minimax_h3_presence_gated_av_mse(
        [0.0, 2.0, -1.0, 3.0],
        [1.0, 0.0, -1.0, 1.0],
        [2.0, -1.0, 1.0, 3.0],
        [0.0, -1.0, 2.0, 1.0],
        Float32(0.5),
        Float32(1.0),
    )
    _check(_close(present.video, Float32(2.25)), "video mean MSE")
    _check(_close(present.audio, Float32(2.25)), "audio mean MSE")
    _check(_close(present.total, Float32(3.375)), "presence-gated AV total")

    var absent = minimax_h3_presence_gated_av_mse(
        [0.0], [1.5], List[Float32](), List[Float32](),
        Float32(0.5), Float32(0.0),
    )
    _check(_close(absent.total, Float32(2.25)), "absent audio skips audio tensors")

    var contract = _valid_fixture()
    _check(
        contract.dataset_identity == String(MINIMAX_H3_INTAKE_DATASET_IDENTITY),
        "requested-run dataset receipt",
    )
    validate_minimax_h3_policy(contract)

    var other_dataset = contract.copy()
    other_dataset.dataset_identity = String("another_valid_h3_dataset")
    validate_minimax_h3_policy(other_dataset)

    var launch_rejected = False
    try:
        validate_minimax_h3_launch_preflight(contract)
    except e:
        launch_rejected = True
        print("  expected launch rejection [unresolved cache manifest]:", String(e))
    _check(launch_rejected, "launch remains blocked")

    var nonempty_path = contract.copy()
    nonempty_path.dataset_path = String("/tmp/not-a-validated-h3-cache")
    launch_rejected = False
    try:
        validate_minimax_h3_launch_preflight(nonempty_path)
    except e:
        launch_rejected = True
        print("  expected launch rejection [unvalidated nonempty path]:", String(e))
    _check(launch_rejected, "nonempty path cannot bypass manifest validation")

    var batch_bad = contract.copy()
    batch_bad.batch_size = 2
    _expect_policy_rejects(batch_bad, "batch_size=2")
    var full_ft_bad = contract.copy()
    full_ft_bad.full_finetune = True
    _expect_policy_rejects(full_ft_bad, "full finetune")
    var ref_image_bad = contract.copy()
    ref_image_bad.task = String("ref2va")
    ref_image_bad.one_frame = True
    _expect_policy_rejects(ref_image_bad, "one-frame ref2va")
    var low_shift_bad = contract.copy()
    low_shift_bad.video_shift = Float32(0.001)
    _expect_policy_rejects(low_shift_bad, "video shift below bound")
    var high_shift_bad = contract.copy()
    high_shift_bad.audio_shift = Float32(1000.0)
    _expect_policy_rejects(high_shift_bad, "audio shift above bound")
    var accum_bad = contract.copy()
    accum_bad.gradient_accumulation_steps = 0
    _expect_policy_rejects(accum_bad, "zero gradient accumulation")
    var weighting_bad = contract.copy()
    weighting_bad.weighting_scheme = String("min_snr")
    _expect_policy_rejects(weighting_bad, "non-none weighting")

    var int8_base = contract.copy()
    int8_base.base_storage = String("convrot_int8")
    validate_minimax_h3_policy(int8_base)
    var bad_base_storage = contract.copy()
    bad_base_storage.base_storage = String("fp8")
    _expect_policy_rejects(bad_base_storage, "unsupported base storage")
    var trainable_base = contract.copy()
    trainable_base.base_frozen = False
    _expect_policy_rejects(trainable_base, "trainable base")
    var bf16_lora_parameters = contract.copy()
    bf16_lora_parameters.lora_parameter_storage = String("bf16")
    _expect_policy_rejects(bf16_lora_parameters, "BF16 LoRA parameter storage")
    var bf16_lora_gradients = contract.copy()
    bf16_lora_gradients.lora_gradient_storage = String("bf16")
    _expect_policy_rejects(bf16_lora_gradients, "BF16 LoRA gradient storage")
    var bf16_optimizer_state = contract.copy()
    bf16_optimizer_state.optimizer_state_storage = String("bf16")
    _expect_policy_rejects(bf16_optimizer_state, "BF16 optimizer state storage")

    var invalid_audio_present_rejected = False
    try:
        _ = minimax_h3_effective_audio_weight(1.0, 0.5)
    except e:
        invalid_audio_present_rejected = True
        print("  expected rejection [invalid audio_present]:", String(e))
    _check(invalid_audio_present_rejected, "invalid audio_present must reject")

    print("MiniMax H3 deterministic policy SMOKE PASS")
