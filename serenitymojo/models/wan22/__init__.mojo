# serenitymojo.models.wan22 — Wan2.2-TI2V-5B DiT per-model TRAINING surface:
# config (dims + recipe), the WanAttentionBlock kind (fwd saving acts +
# hand-chained bwd + LoRA variants). The genuinely-new compute vs the
# double-stream models: a SINGLE image stream with self-attn (qk-rms + 3-axis
# interleaved RoPE), CROSS-attn to text (distinct q/kv lengths -> rect SDPA),
# gelu-tanh FFN, and PER-TOKEN AdaLN (scale/shift/gate are [1,S,dim], not [D]).
