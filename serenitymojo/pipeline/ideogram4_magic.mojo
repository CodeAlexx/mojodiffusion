# pipeline/ideogram4_magic.mojo — pure-Mojo magic prompt: plain text -> JSON caption.
# Runs Qwen3-8B (lm_head present) autoregressively in Mojo with a magic-prompt
# system prompt, emitting an Ideogram-4 structured JSON caption. The JSON is then
# tokenized + fed to ideogram4_generate (separate step). No external LLM/API.
from std.gpu.host import DeviceContext
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.text_encoder.qwen3_magic import generate_greedy

comptime QWEN = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
comptime TOKJSON = QWEN + "/tokenizer.json"
comptime EOS = 151645      # <|im_end|>
comptime PAD = 151643      # <|endoftext|>


def _system_prompt() -> String:
    return String(
        "You convert a natural-language image idea into EXACTLY ONE minified JSON"
        " object (single line, no markdown, no commentary) that an image renderer"
        " consumes. Schema, strict key order:\n"
        '{"high_level_description":"<=50 words, one sentence, starts with the subject",'
        '"style_description":{"aesthetics":"...","lighting":"...","photo":"camera/lens (photographic) OR omit and use art_style","medium":"photograph|illustration|3d_render|digital_art|painting","color_palette":["#RRGGBB",...up to 16]},'
        '"compositional_deconstruction":{"background":"...","elements":[{"type":"obj","bbox":[y_min,x_min,y_max,x_max],"desc":"...","color_palette":["#RRGGBB"]}]}}\n'
        "Rules: style_description has EXACTLY ONE of photo (with medium:photograph)"
        " OR art_style (non-photo). bbox is normalized 0-1000, origin top-left,"
        " optional. text elements use {type:text,bbox,text,desc}. Uppercase"
        " #RRGGBB only. Output the JSON object and nothing else."
    )


def main() raises:
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(TOKJSON)
    var user = String("a red cube on a white table (aspect ratio 1:1) /no_think")
    var chat = (
        String("<|im_start|>system\n") + _system_prompt() + "<|im_end|>\n"
        + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
    )
    var ids = tok.encode(chat)
    print("prompt tokens:", len(ids))
    var qwen = Qwen3Encoder.load(QWEN, Qwen3Config.klein_9b(), ctx)
    print("Qwen3-8B loaded; generating magic-prompt JSON (greedy)...")
    var gen = generate_greedy(qwen, ids, 700, EOS, PAD, 1024, ctx)
    print("generated tokens:", len(gen))
    print("=== MAGIC PROMPT JSON ===")
    print(tok.decode(gen))
    print("=== END ===")
