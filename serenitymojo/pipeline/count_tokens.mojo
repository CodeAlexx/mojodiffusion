# count_tokens.mojo — DEV probe: print the templated token count for a prompt so
# we can bake the comptime CAPLEN for the Z-Image pipeline. The DiT's caption
# length is a compile-time constant (forward takes no mask; it pads to mult-of-32
# internally via the learned cap_pad_token), so cap_feats must have EXACTLY the
# real templated token count — which this probe reports.
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

# Z-Image canonical chat template (prepare_l2p.rs ZIMAGE_TEMPLATE_PRE/POST).
comptime TOK_JSON = "/home/alex/.serenity/models/zimage_base/tokenizer/tokenizer.json"
comptime PROMPT = (
    "a glamorous platinum blonde woman with vintage old Hollywood finger-wave"
    " hair screaming with mouth wide open, head tilted back, face bathed in a"
    " harsh dual-tone color gel treatment of deep cobalt blue shadow and acid"
    " green neon highlight, bold neon yellow-green spray paint XX marks slashed"
    " aggressively across her face with dripping paint trails running down over"
    " her nose and cheeks, her ringed hand pressed against her chin adorned with"
    " oversized diamond and jeweled rings, a thick chunky gold chain necklace"
    " draped at her chest, the entire image set against a saturated blood red"
    " background, heavy analog film grain and scratched surface texture over the"
    " entire composition, the photograph treated with a high-contrast duotone"
    " darkroom effect that renders skin in cold blue tones against the violent red"
    " field, the aesthetic of a subversive underground concert poster meets"
    " transgressive fashion editorial, punk energy, neo-noir, Andy Warhol meets"
    " Gaspar Noe, hyper-saturated analog photography, gritty film scan texture,"
    " cinematic portrait, ultra detailed, 8k"
)
comptime _OLD_PROMPT = (
    "The palette is dominated by dark tones. The texture is pronounced, with"
    " brushstrokes and traces of paint visible, imitating the process of painting"
    " using multi-layered tones. The composition is centered around the figures,"
    " and the background flows smoothly in color and tone. The style is abstract,"
    " almost sculptural. The style is abstract, almost sculptural. Free, gestural"
    " brushstrokes create a sense of movement and action. The muted color palette"
    " includes subtle variations in tone. The sense of depth is achieved through"
    " different densities of brushstrokes. The light source is diffused, giving the"
    " image a slightly cool and mysterious shade. The style of painting is"
    " reminiscent of abstract realism and expressionism.  Soft shadows that reflect"
    " every detail and add volume to the scene. The focus is entirely on the girl,"
    " utilizing a 2/3 rule composition. A hyperdetailed masterpiece rendered in"
    " contemporary digital painting with abstract realism and modern neo-figurative"
    " precision, showcasing a central full-body female figure posed confidently in a"
    " frontal eye-level view. She wears an opulent, rich crimson off-the-shoulder"
    " gown featuring a deep plunging neckline, one elegant long sleeve, a high"
    " dramatic slit revealing smooth, gradient-shaded skin, and a flowing, textured"
    " train that cascades dynamically. Her long, dark hair streams with life and"
    " movement, enhancing the sensual power conveyed by her poised stance. Abstract,"
    " jagged black and deep red fragments emerge like stylized, fragmented wings"
    " behind her, creating intense contrast against the warm earth-toned abstract"
    " void. The background swirls with textured, painterly motion in rich ochres,"
    " warm beiges, and deep browns, executed with expressive, rough brushstrokes"
    " that contrast sharply with the figure's hyperrealistic rendering. The digital"
    " technique employs meticulous layering, ultra-smooth gradient shading on the"
    " skin and luxurious fabric to achieve photorealistic depth and luminosity,"
    " while the abstract elements maintain deliberate roughness. Dominant colors are"
    " intense, saturated rich reds, warm golden ochres, deep earthy browns, and"
    " stark dark accents, creating bold, dramatic contrasts. The composition is"
    " perfectly balanced, with the figure centrally positioned as the undeniable"
    " focal point, radiating enigmatic confidence and sensual strength. This image"
    " embodies the dramatic, evocative spirit of Greg Rutkowski's digital fantasy"
    " art and modern digital surrealists, masterfully blending bold color contrasts"
    " with expressive abstraction. Every element is rendered with high details, high"
    " quality, and a definitive sense of digital artistry, resulting in a visually"
    " arresting, emotionally resonant, and unmistakably masterful piece."
)


def main() raises:
    var tok = Qwen3Tokenizer(TOK_JSON)
    var templated = String("<|im_start|>user\n") + PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
    var ids = tok.encode(templated)
    print("templated prompt:", templated)
    print("POS token count (CAPLEN):", len(ids))
    var first = String("[")
    for i in range(len(ids)):
        first += String(ids[i])
        if i < len(ids) - 1:
            first += ", "
    print("ids:", first + "]")

    # negative / uncond = empty prompt, same template
    var neg = String("<|im_start|>user\n") + "" + "<|im_end|>\n<|im_start|>assistant\n"
    var nids = tok.encode(neg)
    print("NEG token count (CAPLEN_neg):", len(nids))
