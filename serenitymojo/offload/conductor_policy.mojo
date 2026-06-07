# OneTrainer CPU_OFFLOADED conductor policy.
#
# This module is scalar policy only. It mirrors OneTrainer's activation/layer
# offload enablement and loaded-byte target math without moving tensors,
# allocating device memory, or changing activation/weight storage dtype.


@fieldwise_init
struct OneTrainerConductorPolicy(Copyable, Movable, ImplicitlyCopyable):
    var checkpointing_offload: Bool
    var activation_offload: Bool
    var layer_offload: Bool
    var async_transfer: Bool
    var layer_offload_fraction: Float64
    var total_layer_bytes: Int
    var target_loaded_bytes: Int

    def target_offloaded_bytes(self) -> Int:
        return self.total_layer_bytes - self.target_loaded_bytes

    def needs_runtime_conductor(self) -> Bool:
        return self.activation_offload or self.layer_offload


def onetrainer_conductor_policy_from_fields(
    checkpointing_offload: Bool,
    enable_activation_offloading: Bool,
    enable_async_offloading: Bool,
    layer_offload_fraction: Float64,
    is_cuda: Bool,
    total_layer_bytes: Int,
) raises -> OneTrainerConductorPolicy:
    if total_layer_bytes < 0:
        raise Error("OneTrainerConductorPolicy: total_layer_bytes must be non-negative")
    if layer_offload_fraction < Float64(0.0) or layer_offload_fraction > Float64(1.0):
        raise Error("OneTrainerConductorPolicy: layer_offload_fraction must be within 0..1")

    var activation_offload = checkpointing_offload and enable_activation_offloading
    var layer_offload = checkpointing_offload and layer_offload_fraction > Float64(0.0)

    # Mirrors OneTrainer LayerOffloadConductor:
    # target_loaded_bytes = int(total_bytes * (1.0 - layer_offload_fraction)).
    # Float64 is scalar policy arithmetic only; no tensor boundary upcast.
    var target_loaded_bytes = total_layer_bytes
    if layer_offload:
        target_loaded_bytes = Int(
            Float64(total_layer_bytes)
            * (Float64(1.0) - layer_offload_fraction)
        )

    return OneTrainerConductorPolicy(
        checkpointing_offload,
        activation_offload,
        layer_offload,
        is_cuda and enable_async_offloading,
        layer_offload_fraction,
        total_layer_bytes,
        target_loaded_bytes,
    )
