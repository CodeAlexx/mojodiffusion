# serenitymojo.models.mageflow — Mage-Flow per-model TRAINING surface.
# The double-stream block is BYTE-FOR-BYTE the Qwen-Image block
# (models/qwenimage/qwenimage_block.mojo); the ONE MageFlow delta — text tokens
# NOT roped — is input data supplied by build_mageflow_rope_tables
# (models/dit/mageflow_dit.mojo). This package holds the MageFlow-specific
# training glue + parity gates.
