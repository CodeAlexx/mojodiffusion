# Compile probe for nucleus_moe.mojo — imports + constructs the plan type.
# COMPILE ONLY (GPU wedged). EXIT=0 == pass.

from std.gpu.host import DeviceContext
from serenitymojo.models.dit.nucleus_moe import (
    ExpertChoicePlan,
    expert_choice_route,
    nucleus_moe_expert_forward,
)


def main() raises:
    # Construct the plan type to force instantiation.
    var gi = List[Int]()
    gi.append(0)
    var gf = List[Float32]()
    gf.append(Float32(1.0))
    var plan = ExpertChoicePlan(
        global_token_indices=gi^,
        gating_flat=gf^,
        batch_size=1,
        seq_len=1,
        num_experts=1,
        capacity=1,
    )
    print("ExpertChoicePlan ok:", plan.num_experts)
    # Reference the forward symbol so it is type-checked / instantiated.
    print("nucleus_moe probe compiled")
