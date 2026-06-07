# conductor_policy_smoke.mojo - no-CUDA gate for OneTrainer CPU_OFFLOADED policy.
#
# This smoke validates scalar conductor policy only. It does not move tensors,
# load checkpoints, or prove activation/layer offload runtime parity.

from serenitymojo.offload.conductor_policy import (
    onetrainer_conductor_policy_from_fields,
)


def _check_bool(name: String, got: Bool, expected: Bool) raises:
    print("[conductor-policy]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("conductor policy bool mismatch: ") + name)


def _check_int(name: String, got: Int, expected: Int) raises:
    print("[conductor-policy]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("conductor policy int mismatch: ") + name)


def _check_raises(name: String, raised: Bool) raises:
    print("[conductor-policy]", name, "raised=", raised)
    if not raised:
        raise Error(String("conductor policy expected raise: ") + name)


def main() raises:
    # OneTrainer Flux2 LoRA 8GB preset: CPU_OFFLOADED + layer_offload_fraction=0.7.
    var lora = onetrainer_conductor_policy_from_fields(
        True,
        True,
        True,
        Float64(0.7),
        True,
        100000,
    )
    _check_bool(String("lora activation offload"), lora.activation_offload, True)
    _check_bool(String("lora layer offload"), lora.layer_offload, True)
    _check_bool(String("lora async cuda"), lora.async_transfer, True)
    _check_int(String("lora target loaded bytes"), lora.target_loaded_bytes, 30000)
    _check_int(String("lora target offloaded bytes"), lora.target_offloaded_bytes(), 70000)
    _check_bool(String("lora needs conductor"), lora.needs_runtime_conductor(), True)

    # OneTrainer Flux2 finetune 16GB preset: CPU_OFFLOADED + layer_offload_fraction=0.6.
    var finetune = onetrainer_conductor_policy_from_fields(
        True,
        True,
        True,
        Float64(0.6),
        True,
        100000,
    )
    _check_int(String("finetune target loaded bytes"), finetune.target_loaded_bytes, 40000)
    _check_int(String("finetune target offloaded bytes"), finetune.target_offloaded_bytes(), 60000)

    var no_cuda = onetrainer_conductor_policy_from_fields(
        True,
        True,
        True,
        Float64(0.7),
        False,
        100000,
    )
    _check_bool(String("async requires cuda"), no_cuda.async_transfer, False)

    var activation_disabled = onetrainer_conductor_policy_from_fields(
        True,
        False,
        True,
        Float64(0.7),
        True,
        100000,
    )
    _check_bool(
        String("activation flag gates activation offload"),
        activation_disabled.activation_offload,
        False,
    )
    _check_bool(
        String("layer offload independent of activation flag"),
        activation_disabled.layer_offload,
        True,
    )

    var plain_on = onetrainer_conductor_policy_from_fields(
        False,
        True,
        True,
        Float64(0.7),
        True,
        100000,
    )
    _check_bool(String("plain checkpointing has no activation offload"), plain_on.activation_offload, False)
    _check_bool(String("plain checkpointing has no layer offload"), plain_on.layer_offload, False)
    _check_int(String("plain checkpointing keeps all bytes loaded"), plain_on.target_loaded_bytes, 100000)

    var raised = False
    try:
        _ = onetrainer_conductor_policy_from_fields(
            True,
            True,
            True,
            Float64(1.1),
            True,
            100000,
        )
    except:
        raised = True
    _check_raises(String("fraction > 1 rejected"), raised)

    raised = False
    try:
        _ = onetrainer_conductor_policy_from_fields(
            True,
            True,
            True,
            Float64(0.7),
            True,
            -1,
        )
    except:
        raised = True
    _check_raises(String("negative total bytes rejected"), raised)

    print("conductor_policy_smoke PASS")
