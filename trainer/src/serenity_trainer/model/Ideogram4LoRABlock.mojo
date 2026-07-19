# RE-EXPORT SHIM (Stage 0, autograd_v2 ideogram4 port): the ideogram4 training
# block moved to serenitymojo/models/ideogram4/block.mojo so the v2 engine adapter
# (serenitymojo) can call its backward. Existing serenity-trainer importers unchanged.
from serenitymojo.models.ideogram4.block import (
    IDEOGRAM4_SDPA_FLASH,
    I4_SLOT_QKV, I4_SLOT_O, I4_SLOT_W1, I4_SLOT_W2, I4_SLOT_W3, I4_SLOT_ADALN,
    I4_SLOTS_PER_BLOCK, I4_EPS,
    Ideogram4BlockWeights, load_ideogram4_block_weights,
    load_ideogram4_block_weights_resident,
    Ideogram4LoraSet, build_ideogram4_lora_set, build_ideogram4_native_lora_set,
    Ideogram4BlockActs, Ideogram4BlockOut, ideogram4_block_lora_forward,
    Ideogram4BlockLoraGrads, Ideogram4BlockBwd, ideogram4_block_lora_backward,
    Ideogram4StackForward, ideogram4_stack_lora_forward,
    ideogram4_stack_lora_forward_resident,
    ideogram4_stack_lora_forward_resident_nosave, Ideogram4StackLoraGrads,
    ideogram4_stack_lora_backward, ideogram4_stack_lora_backward_resident,
    # TRUE batch-2 (row-stacked over a real batch=2 dim; grads SUM both samples =
    # batch gradient → fed to the SAME device optimizer as b1).
    ideogram4_block_lora_forward_b2, ideogram4_block_lora_backward_b2,
    ideogram4_stack_lora_forward_b2, ideogram4_stack_lora_forward_resident_b2,
    ideogram4_stack_lora_backward_b2, ideogram4_stack_lora_backward_resident_b2,
)
# P7 autograd_v2 graph variant of the stack backward (separate file so the
# import graph stays acyclic; see block_stack_graph.mojo header). Re-exported
# here so trainer files import it alongside the hand-chain.
from serenitymojo.models.ideogram4.block_stack_graph import (
    ideogram4_stack_lora_backward_graph,
    ideogram4_stack_lora_backward_graph_resident,
)
