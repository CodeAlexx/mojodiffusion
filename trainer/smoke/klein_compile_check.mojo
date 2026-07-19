# Compile-check the port's Klein LoRA stack (2nd-pass dropout/guidance edits).
from serenity_trainer.model.klein.klein_stack_lora import build_klein_lora_set

def main() raises:
    var s = build_klein_lora_set(8, 24, 64, 128, 8, Float32(8.0))
    print("klein lora set constructed (8 double + 24 single)")
