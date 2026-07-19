# model_spec.mojo — the typed training seam (translates Serenity's
# BaseModelSetup.predict/calculate_loss). Serenity returns a stringly-typed
# dict {predicted, target, timestep, ...}; we use a TYPED struct so the shared
# train_step consumes only this.
#
# Serenity has inheritance (Base<Model>Setup); Mojo does not. The per-model
# surface becomes a `trait ModelSpec`, dispatched by comptime monomorphization at
# main() (PORT_MAP §4). This file defines the seam type + trait only; concrete
# conformances live in models/<m>/ (Phase 3+).
#
# All tensors are BF16 storage (port dtype policy).

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.autograd import Tape
from serenity_trainer.util.config.TrainConfig import TrainConfig


struct StepOutput(Movable):
    """The narrow seam the shared step consumes. Mirrors Serenity's
    model_output_data {'predicted','target','timestep'}. Movable (owns Tensors,
    which are Movable-not-Copyable)."""

    var predicted: Tensor   # model output (BF16, tape-tracked)
    var target: Tensor      # training target (BF16, constant — not tracked)
    var timestep: Float32   # the sampled timestep/sigma scalar for this step

    def __init__(out self, var predicted: Tensor, var target: Tensor, timestep: Float32):
        self.predicted = predicted^
        self.target = target^
        self.timestep = timestep


# A per-model spec exposes the genuinely model-specific surface Serenity puts in
# Base<Model>Setup. The shared driver/train_step calls these directly (no stored
# closures — Mojo can't store heterogeneous captured closures; grads-as-input).
#
# `predict` builds the noised input + samples a timestep, runs the (LoRA-wrapped)
# model on the tape, and returns the StepOutput. The shared loss then consumes it.
trait ModelSpec(Movable):
    # Forward: record the model on `tape`, return predicted/target/timestep.
    # `step` seeds per-step noise (matches reference trainer bf16_stochastic_rounding.set_seed).
    def predict(
        mut self, mut tape: Tape, config: TrainConfig, step: Int, ctx: DeviceContext
    ) raises -> StepOutput:
        ...
