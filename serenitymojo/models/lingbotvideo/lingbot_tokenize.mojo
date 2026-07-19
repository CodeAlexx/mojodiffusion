# Pure-Mojo tokenizer path for LingBot-Video prompts.
#
# Drops the Python `AutoProcessor` tokenization: applies the LingBot
# PROMPT_TEMPLATE to a prompt string and BPE-encodes it with the pure-Mojo
# Qwen3Tokenizer (loaded from text_encoder/tokenizer.json). Parity-critical —
# the ids EXACTLY match the HF processor's for the LingBot Qwen3-VL text path.
#
# Reference (creator source):
#   /mnt/disk1/lingbot-src/lingbot-video/lingbot_video/pipeline_lingbot_video.py:27-39
#     PROMPT_TEMPLATE = (
#       "<|im_start|>system\n...evaluations:<|im_end|>\n<|im_start|>user\n{}"
#       "<|im_end|>\n<|im_start|>assistant\n" )
#   pipeline_lingbot_video crop_start trick:
#     marked = PROMPT_TEMPLATE.format("<|USER_INPUT_MARKER|>")
#     pos = marked.find(marker); crop_start = len(tokenize(marked[:pos]))
#   -> marked[:pos] == the fixed system-prefix portion below (LINGBOT_TEMPLATE_PREFIX).
#
# The <|im_start|>/<|im_end|>/user/assistant markers tokenize as SINGLE special
# ids (151644 / 151645 / 872 / 77091) via Qwen3Tokenizer._split_on_specials.

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

# The template PREFIX: everything up to (and including "user\n") before the {}
# user-input slot. Constant -> crop_start is len(encode(this)).
comptime LINGBOT_TEMPLATE_PREFIX: StaticString = "<|im_start|>system\nGiven a user input that may include a text prompt alone, a text prompt with an image reference, or a text prompt with a video reference or a video reference alone, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user's input: if it is simple, enrich it by adding specifics about colors, shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal progression, and spatial relationships to create vivid, concrete, and temporally coherent scenes to create vivid and concrete scenes. Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:<|im_end|>\n<|im_start|>user\n"

# The template SUFFIX: everything after the {} slot.
comptime LINGBOT_TEMPLATE_SUFFIX: StaticString = "<|im_end|>\n<|im_start|>assistant\n"

# The image template inserted into the user slot for VLM-grounded i2v/ti2v
# (pipeline_lingbot_video.py IMG_PROMPT_TEMPLATE). The single <|image_pad|> is
# EXPANDED by the processor to grid.prod()//merge**2 copies before tokenizing.
comptime LINGBOT_VISION_START: StaticString = "<|vision_start|>"
comptime LINGBOT_IMAGE_PAD: StaticString = "<|image_pad|>"
comptime LINGBOT_VISION_END: StaticString = "<|vision_end|>"

# pipeline_lingbot_video.py DEFAULT_NEGATIVE_PROMPT (the VIDEO default the ti2v/i2v
# path uses; verbatim module-level constant, ASCII).
comptime LINGBOT_DEFAULT_NEGATIVE_PROMPT: StaticString = "{\"universal_negative\": {\"visual_quality\": [\"low quality\", \"worst quality\", \"blurry\", \"pixelated\", \"jpeg artifacts\", \"low resolution\", \"unstable color\", \"color flicker\", \"underexposed\", \"overexposed\", \"invisible subject\", \"subject hidden in darkness\"], \"artistic_style\": [\"painting\", \"illustration\", \"drawing\", \"cartoon\", \"3d render\", \"cgi\", \"sketch\", \"digital art\"], \"composition_and_content\": [\"text\", \"watermark\", \"signature\", \"logo\", \"subtitles\", \"pillarboxed\", \"side bars\", \"portrait image in landscape frame\"], \"temporal_and_motion_stability\": [\"flickering\", \"jittery\", \"motion blur\", \"temporal inconsistency\", \"warping\", \"morphing\", \"incoherent motion\", \"unnatural movement\", \"static object with sudden jump\", \"frame-to-frame inconsistency\"], \"material_and_structure\": [\"plastic-like glass\", \"unrealistic texture\", \"deformed bottle\", \"liquid freezing improperly\", \"distorted reflections\"]}}"


def apply_lingbot_template(prompt_str: String) -> String:
    # PROMPT_TEMPLATE.format(prompt_str)
    return String(LINGBOT_TEMPLATE_PREFIX) + prompt_str + String(LINGBOT_TEMPLATE_SUFFIX)


def tokenize_lingbot_prompt(
    tok: Qwen3Tokenizer, prompt_str: String
) raises -> Tuple[List[Int], Int]:
    # Returns (ids, crop_start).
    #   ids        = tokenize(PROMPT_TEMPLATE.format(prompt_str))  (full length)
    #   crop_start = tokenize(LINGBOT_TEMPLATE_PREFIX) length
    # Mirrors pipeline_lingbot_video.py: the DiT consumes hidden states cropped to
    # ids[crop_start:], so crop_start is the count of system-prefix tokens.
    var full = apply_lingbot_template(prompt_str)
    var ids = tok.encode(full)
    var prefix_ids = tok.encode(String(LINGBOT_TEMPLATE_PREFIX))
    var crop_start = len(prefix_ids)
    return (ids^, crop_start)


def tokenize_lingbot_image_prompt(
    tok: Qwen3Tokenizer, prompt_str: String, n_image_tokens: Int
) raises -> Tuple[List[Int], Int]:
    # VLM-grounded (i2v/ti2v) tokenization. Mirrors the Qwen3-VL processor: the
    # user slot is IMG_PROMPT_TEMPLATE + prompt, where the single <|image_pad|> is
    # expanded to `n_image_tokens` (= grid.prod()//merge**2) copies before encode.
    #   ids        = tokenize(PROMPT_TEMPLATE.format(vision_start
    #                  + image_pad*n_image_tokens + vision_end + prompt))
    #   crop_start = len(tokenize(PREFIX))  (the vision block sits AFTER the prefix)
    var vis = String(LINGBOT_VISION_START)
    var pad = String(LINGBOT_IMAGE_PAD)
    for _ in range(n_image_tokens):
        vis += pad
    vis += String(LINGBOT_VISION_END)
    var user = vis + prompt_str
    var full = apply_lingbot_template(user)
    var ids = tok.encode(full)
    var prefix_ids = tok.encode(String(LINGBOT_TEMPLATE_PREFIX))
    var crop_start = len(prefix_ids)
    return (ids^, crop_start)
