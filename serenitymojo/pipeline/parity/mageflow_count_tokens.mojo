# mageflow_count_tokens.mojo — DEV probe: print the mage-flow templated token
# count (L_full) for each prompt so we can bake the comptime N_TXT = L_full-34
# for the MageFlow T2I pipeline (the DiT forward is comptime-specialized on the
# text sequence length, exactly like the Z-Image CAPLEN pattern).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . serenitymojo/pipeline/parity/mageflow_count_tokens.mojo
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

comptime TOK_JSON = (
    "/home/alex/.serenity/models/checkpoints/Mage-Flow-Turbo/text_encoder/tokenizer.json"
)
comptime DROP = 34

# mage-flow t2i template (utils.py PROMPT_TEMPLATE["mage-flow"]).
comptime PRE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size,"
    " texture, quantity, text, spatial relationships of the objects and"
    " background:<|im_end|>\n<|im_start|>user\n"
)
comptime POST = "<|im_end|>\n<|im_start|>assistant\n"

comptime GATE_PROMPT = (
    "A photorealistic portrait of an astronaut riding a horse on Mars."
)
comptime PROMPT1 = (
    "A close-up of a face adorned with intricate black and blue patterns. The"
    " left side of the face is predominantly yellow, with symbols and doodles,"
    " while the right side is dark, featuring mechanical elements. The eye on the"
    " left is a striking shade of yellow, contrasting sharply with the surrounding"
    " patterns. The face is partially covered by a hooded garment, cinematic_1940s"
    " cinematic_octane"
)
comptime PROMPT2 = (
    "realistic, vibrant, detailed. a cinematic photograph classily depicts a"
    " prestigious beautiful woman dressed in an all white dress. The woman has no"
    " face. Her head is a sphere, a colorful gradient foreign planet, and not a"
    " human head. Her outfit is adorned by luxurious gold trim and other attire."
    " Her skin is tan with smooth rough textures. She has a very detailed"
    " planetary sphere for her head that emits natural weather phenomenon and neon"
    " ethereal ozonous layers, swirls, and hues. The image is sharp and"
    " professional with studio quality. The contrast and style is very sharp, but"
    " liquid and vivid. Gorgeous background and details with subtle hints of"
    " illumination. glass, ether, spirit, breath, light, energy. Particles, high"
    " density quality, vibrant colors and hues. The overall style of the image is"
    " highly stylish, dynamic and free-flowing elegance., modest, sfw"
)


def _report(tok: Qwen3Tokenizer, tag: String, prompt: String) raises:
    var templated = String(PRE) + prompt + String(POST)
    var ids = tok.encode(templated)
    var L = len(ids)
    print(
        "[count]", tag, "L_full =", L, "| N_TXT (keep=L-34) =", L - DROP,
        "| first =", ids[0], "last =", ids[L - 1],
    )


def main() raises:
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    _report(tok, "GATE(astronaut)", String(GATE_PROMPT))
    _report(tok, "PROMPT1        ", String(PROMPT1))
    _report(tok, "PROMPT2        ", String(PROMPT2))
