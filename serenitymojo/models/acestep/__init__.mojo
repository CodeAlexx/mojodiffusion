# serenitymojo.models.acestep — ACE-Step-1.5 DiT per-model TRAINING surface:
# config (dims + recipe, dims confirmed from the real safetensors header), the
# AceStep DiT layer kind (fwd saving acts + hand-chained bwd + LoRA variants).
# Genuinely-new compute vs wan22: GQA (16 q / 8 kv heads, n_rep=2 -> repeat_kv
# backward = grouped sum), PER-SAMPLE [H] AdaLN (modulate/residual_gate [D]-vector
# kernels), no-bias linears, halfsplit self-attn RoPE, SwiGLU MLP.
