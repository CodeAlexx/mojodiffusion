# No-CUDA Klein activation tape accounting smoke.
#
# This validates the saved boundary activation byte target for the current
# Klein LoRA offload-turbo training tape. It does not run the model, move
# activations, or prove OneTrainer CPU_OFFLOADED parity.

from serenitymojo.models.klein.activation_tape_plan import (
    klein9b_lora_training_activation_tape_plan,
)


def _check(name: String, got: Int, expected: Int) raises:
    print("[klein-activation-tape]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("Klein activation tape plan mismatch: ") + name)


def _check_bool(name: String, got: Bool, expected: Bool) raises:
    print("[klein-activation-tape]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("Klein activation tape plan bool mismatch: ") + name)


def main() raises:
    var plan = klein9b_lora_training_activation_tape_plan()

    _check(String("seq_len"), plan.seq_len(), 1536)
    _check(String("dtype bytes"), plan.bytes_per_elem(), 2)
    _check(String("stream boundary elems"), plan.stream_boundary_elems(), 6291456)
    _check(String("input projection elems"), plan.input_projection_boundary_elems(), 6291456)
    _check(String("double input elems"), plan.double_input_boundary_elems(), 50331648)
    _check(String("single input elems"), plan.single_input_boundary_elems(), 150994944)
    _check(String("final boundary elems"), plan.final_boundary_elems(), 8388608)
    _check(String("current boundary elems"), plan.current_boundary_elems(), 216006656)
    _check(String("live backward boundary elems"), plan.live_backward_boundary_elems(), 209715200)
    _check(
        String("unused input projection elems"),
        plan.unused_input_projection_boundary_elems(),
        6291456,
    )
    _check(String("current boundary bytes bf16"), plan.current_boundary_bytes(), 432013312)
    _check(String("live backward boundary bytes bf16"), plan.live_backward_boundary_bytes(), 419430400)
    _check(
        String("unused input projection bytes bf16"),
        plan.unused_input_projection_boundary_bytes(),
        12582912,
    )
    _check(String("current boundary bytes f32"), plan.current_boundary_f32_bytes(), 864026624)
    _check(String("live backward boundary bytes f32"), plan.live_backward_boundary_f32_bytes(), 838860800)
    _check(
        String("current f32 extra bytes"),
        plan.current_boundary_f32_over_storage_bytes(),
        432013312,
    )
    _check(
        String("live f32 extra bytes"),
        plan.live_backward_f32_over_storage_bytes(),
        419430400,
    )
    _check_bool(String("internal tails recompute-only"), plan.internal_tail_is_recompute_only(), True)

    print("klein_activation_tape_plan_smoke PASS")
