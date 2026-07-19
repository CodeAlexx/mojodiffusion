# model/lens — Lens (double-stream MM-DiT) package. The LoRA set, the saved-tape
# struct contract, and the hand-chained backward live here (lens_backward.mojo);
# the resident/training forward (lens_forward_full_lora) is the sibling model unit
# (lens_stack_lora.mojo). Foundation (serenitymojo/{tensor,io,ops,autograd,util})
# is imported unchanged; the Lens block math is COPIED (namespaced) from
# serenitymojo/pipeline/lens_pipeline_1024_multistep.mojo per the borrow boundary.
