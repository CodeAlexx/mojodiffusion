# serenitymojo.models.ltx2 - LTX-2 (22B video DiT) per-model training surface:
# config (dims + recipe), the core video transformer block (self-attn + FFN)
# fwd-save-acts + hand-chained backward + LoRA variants. Parity-gated vs
# torch.autograd under parity/. Consumed by the shared training pipeline.
