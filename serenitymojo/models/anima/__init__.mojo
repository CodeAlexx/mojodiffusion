# serenitymojo.models.anima — Anima (Cosmos-Predict2 MiniTrainDIT) per-model
# training surface: config (dims + recipe), per-block weight loader, block kind
# (self-attn + cross-attn + GELU MLP, AdaLN-LoRA modulation) fwd/bwd. Kept
# self-contained per the ANIMA INDEPENDENCE rule — no cross-import of a Cosmos
# Mojo body; patterns are copied from the proven Klein template.
