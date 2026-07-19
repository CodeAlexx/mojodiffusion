# qwen_text_encoder_compile_check.mojo — compile gate for the Qwen3 text encoder
# port + chat-template tokenizer seam. References Qwen3Config/Qwen3Encoder/
# QwenChatTokenizer/text_encode so the compiler instantiates them.
# Compile only (.load()/.encode() need the real checkpoint + tokenizer.json):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -I trainer/src \
#       trainer/smoke/qwen_text_encoder_compile_check.mojo -o /tmp/qenc_check

from std.gpu.host import DeviceContext
from serenity_trainer.model.QwenTextEncoder import (
    Qwen3Config,
    Qwen3Encoder,
    QwenChatTokenizer,
    text_encode,
    ZIMAGE_HIDDEN,
    ZIMAGE_PENULTIMATE_LAYER,
    ZIMAGE_PAD_ID,
)


def main() raises:
    var _ctx = DeviceContext()

    var cfg = Qwen3Config.zimage()
    print("qwen3 hidden =", cfg.hidden_size, " layers =", cfg.num_layers,
          " heads =", cfg.num_heads, " kv =", cfg.num_kv_heads)
    print("penultimate layer (hidden_states[-2]) =", ZIMAGE_PENULTIMATE_LAYER)
    print("hidden =", ZIMAGE_HIDDEN, " pad_id =", ZIMAGE_PAD_ID)

    # Qwen3Config.zimage() (above) + the imports force Qwen3Encoder,
    # QwenChatTokenizer, and text_encode to parse and typecheck. These are
    # non-parametric defs/structs, so their bodies are checked at definition;
    # we don't run .load()/.encode()/text_encode here (those need the real
    # checkpoint + tokenizer.json).
    print("QWEN TEXT ENCODER COMPILE-CHECK OK")
