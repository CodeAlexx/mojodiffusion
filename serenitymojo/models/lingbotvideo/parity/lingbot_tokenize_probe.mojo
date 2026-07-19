# Parity gate: pure-Mojo LingBot-Video prompt tokenizer vs the HF processor.
#
# Loads the LingBot text_encoder/tokenizer.json into the pure-Mojo Qwen3Tokenizer,
# applies the LingBot PROMPT_TEMPLATE to the APPLE T2I prompt, and asserts the
# produced ids EXACTLY equal the processor's captured ids (oracle_text_ids.json),
# with crop_start == 140 and true_len == 597.
#
# Run:
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#       serenitymojo/models/lingbotvideo/parity/lingbot_tokenize_probe.mojo

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.lingbotvideo.lingbot_tokenize import tokenize_lingbot_prompt

comptime TOK_JSON: StaticString = "/mnt/disk1/models/lingbot-video-moe/text_encoder/tokenizer.json"

# The apple T2I prompt = json.dumps({...}) from parity/oracle_e_t2i.py (verbatim,
# compact separators ", "/": ", ensure_ascii -> all ASCII).
comptime PROMPT: StaticString = "{\"comprehensive_description\": \"A single glossy red apple rests at the center of a warm-toned wooden table, photographed as a crisp photorealistic studio still life. The apple's smooth waxy skin shows deep crimson and scarlet tones with subtle yellow-green mottling near its short brown stem, catching a bright specular highlight from a soft key light above. It sits on the horizontal planks of a rustic oak table whose visible grain runs left to right, lit by clean studio lighting that casts a soft contact shadow beneath the fruit against a smoothly graded neutral background.\", \"camera_info\": {\"color\": \"Warm\", \"frame_size\": \"Close Up\", \"shot_type_angle\": \"Eye level\", \"lens_size\": \"Medium\", \"composition\": \"Center\", \"lighting\": \"Soft light\", \"lighting_type\": \"Artificial light\"}, \"world_knowledge\": [], \"prominent_elements\": [{\"name\": \"red apple\", \"description\": \"A single ripe red apple, the dominant subject, with a rounded body and a short woody stem at the top.\", \"location\": \"center of the frame, resting on the table surface\", \"relative_size\": \"dominant\", \"shape_and_color\": \"rounded spherical form in deep red and crimson with faint yellow-green flecks\", \"texture\": \"smooth, waxy, glossy\", \"appearance_details\": \"bright specular highlight on the upper left, a small brown stem, faint natural mottling on the skin\", \"relationship\": \"sits on top of the wooden table, casting a soft shadow onto its surface\", \"orientation\": \"upright with the stem pointing up\"}, {\"name\": \"wooden table\", \"description\": \"A rustic oak tabletop that fills the lower portion of the frame and supports the apple.\", \"location\": \"lower half of the frame, extending horizontally\", \"relative_size\": \"large\", \"shape_and_color\": \"flat horizontal surface in warm honey-brown tones\", \"texture\": \"matte wood with visible grain\", \"appearance_details\": \"natural wood grain lines running left to right, subtle knots and color variation\", \"relationship\": \"supports the red apple and receives its soft contact shadow\", \"orientation\": \"horizontal\"}]}"


def _oracle_ids() -> List[Int]:
    var ids: List[Int] = [151644, 8948, 198, 22043, 264, 1196, 1946, 429, 1231, 2924, 264, 1467, 9934, 7484, 11, 264, 1467, 9934, 448, 458, 2168, 5785, 11, 476, 264, 1467, 9934, 448, 264, 2766, 5785, 476, 264, 2766, 5785, 7484, 11, 6923, 458, 330, 57468, 4874, 9934, 1, 429, 5707, 11682, 9124, 27787, 14452, 369, 2766, 9471, 13, 54115, 279, 2188, 315, 7716, 304, 279, 1196, 594, 1946, 25, 421, 432, 374, 4285, 11, 30418, 432, 553, 7842, 48349, 911, 7987, 11, 20816, 11, 12282, 11, 29853, 11, 17716, 11, 11379, 29195, 11, 6249, 7203, 11, 35915, 32724, 11, 323, 27979, 11871, 311, 1855, 42020, 11, 14175, 11, 323, 18965, 745, 55787, 16065, 311, 1855, 42020, 323, 14175, 16065, 13, 5209, 6923, 1172, 279, 23922, 4008, 369, 279, 9934, 3685, 323, 5648, 2670, 894, 5107, 30610, 476, 55081, 25, 151645, 198, 151644, 872, 198, 4913, 874, 52899, 11448, 788, 330, 32, 3175, 73056, 2518, 23268, 53231, 518, 279, 4126, 315, 264, 8205, 74635, 291, 22360, 1965, 11, 56203, 438, 264, 41854, 4503, 89768, 4532, 14029, 2058, 2272, 13, 576, 23268, 594, 10876, 289, 13773, 6787, 4933, 5538, 96019, 323, 22290, 1149, 41976, 448, 26447, 13753, 38268, 296, 1716, 2718, 3143, 1181, 2805, 13876, 19101, 11, 33068, 264, 9906, 85417, 11167, 504, 264, 8413, 1376, 3100, 3403, 13, 1084, 23011, 389, 279, 16202, 625, 4039, 315, 264, 57272, 37871, 1965, 6693, 9434, 23925, 8473, 2115, 311, 1290, 11, 13020, 553, 4240, 14029, 17716, 429, 56033, 264, 8413, 3645, 12455, 23969, 279, 13779, 2348, 264, 38411, 79173, 20628, 4004, 10465, 330, 24910, 3109, 788, 5212, 3423, 788, 330, 95275, 497, 330, 6763, 2368, 788, 330, 7925, 3138, 497, 330, 6340, 1819, 21727, 788, 330, 50058, 2188, 497, 330, 75, 724, 2368, 788, 330, 40994, 497, 330, 76807, 788, 330, 9392, 497, 330, 4145, 287, 788, 330, 30531, 3100, 497, 330, 4145, 287, 1819, 788, 330, 9286, 16488, 3100, 14345, 330, 14615, 4698, 51186, 788, 10071, 330, 24468, 13847, 22801, 788, 61753, 606, 788, 330, 1151, 23268, 497, 330, 4684, 788, 330, 32, 3175, 56696, 2518, 23268, 11, 279, 24456, 3832, 11, 448, 264, 17976, 2487, 323, 264, 2805, 23738, 1076, 19101, 518, 279, 1909, 10465, 330, 2527, 788, 330, 3057, 315, 279, 4034, 11, 40119, 389, 279, 1965, 7329, 497, 330, 20432, 2368, 788, 330, 5600, 85296, 497, 330, 12231, 8378, 6714, 788, 330, 43991, 64151, 1352, 304, 5538, 2518, 323, 96019, 448, 37578, 13753, 38268, 12962, 14553, 497, 330, 27496, 788, 330, 56866, 11, 289, 13773, 11, 73056, 497, 330, 96655, 13260, 788, 330, 72116, 85417, 11167, 389, 279, 8416, 2115, 11, 264, 2613, 13876, 19101, 11, 37578, 5810, 296, 1716, 2718, 389, 279, 6787, 497, 330, 36095, 788, 330, 82, 1199, 389, 1909, 315, 279, 22360, 1965, 11, 24172, 264, 8413, 12455, 8630, 1181, 7329, 497, 330, 24294, 788, 330, 454, 1291, 448, 279, 19101, 21633, 705, 14345, 5212, 606, 788, 330, 6660, 268, 1965, 497, 330, 4684, 788, 330, 32, 57272, 37871, 88471, 429, 40587, 279, 4722, 13348, 315, 279, 4034, 323, 11554, 279, 23268, 10465, 330, 2527, 788, 330, 14772, 4279, 315, 279, 4034, 11, 32359, 58888, 497, 330, 20432, 2368, 788, 330, 16767, 497, 330, 12231, 8378, 6714, 788, 330, 26229, 16202, 7329, 304, 8205, 25744, 1455, 4830, 41976, 497, 330, 27496, 788, 330, 8470, 665, 7579, 448, 9434, 23925, 497, 330, 96655, 13260, 788, 330, 52880, 7579, 23925, 5128, 4303, 2115, 311, 1290, 11, 26447, 60217, 323, 1894, 22990, 497, 330, 36095, 788, 330, 77709, 279, 2518, 23268, 323, 21189, 1181, 8413, 3645, 12455, 497, 330, 24294, 788, 330, 30629, 9207, 13989, 151645, 198, 151644, 77091, 198]
    return ids^


def main() raises:
    print("Loading LingBot text_encoder/tokenizer.json (pure-Mojo parse) ...")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    print("Loaded. Tokenizing PROMPT_TEMPLATE.format(apple-prompt).\n")

    var result = tokenize_lingbot_prompt(tok, String(PROMPT))
    var ids = result[0].copy()
    var crop_start = result[1]
    var true_len = len(ids)

    var oracle = _oracle_ids()

    print("crop_start =", crop_start, " (oracle 140)")
    print("true_len   =", true_len, " (oracle 597)")

    # Element-for-element compare, report FIRST divergence.
    var first_div = -1
    var m = len(ids)
    if len(oracle) < m:
        m = len(oracle)
    for i in range(m):
        if ids[i] != oracle[i]:
            first_div = i
            break

    var exact = (len(ids) == len(oracle)) and (first_div == -1)

    if exact:
        print("\n  IDS: EXACT MATCH (", len(ids), " tokens)")
    else:
        print("\n  IDS: MISMATCH")
        if len(ids) != len(oracle):
            print("        length got", len(ids), " oracle", len(oracle))
        if first_div >= 0:
            var lo = first_div - 3
            if lo < 0:
                lo = 0
            var hi = first_div + 4
            if hi > m:
                hi = m
            print("        first divergence at index", first_div)
            var gs = String("        got   [")
            var os = String("        oracle[")
            for i in range(lo, hi):
                if i != lo:
                    gs += String(", ")
                    os += String(", ")
                gs += String(ids[i])
                os += String(oracle[i])
            gs += String("]")
            os += String("]")
            print(gs)
            print(os)

    var crop_ok = crop_start == 140
    var len_ok = true_len == 597

    print("\n==================================================")
    if exact and crop_ok and len_ok:
        print("LINGBOT TOKENIZE GATE: PASS (exact ids, crop_start=140, true_len=597)")
    else:
        print("LINGBOT TOKENIZE GATE: FAIL")
        print("  exact_ids  =", exact)
        print("  crop_start =", crop_start, "(want 140)  ok=", crop_ok)
        print("  true_len   =", true_len, "(want 597)  ok=", len_ok)
    print("==================================================")
