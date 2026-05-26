#!/usr/bin/env python3
"""SKEPTIC fresh adversarial oracle.

Generates ~70 NEW prompts (not the builder's set) a real image-gen user might
type, encodes each with HF tokenizers (same tokenizer.json), and writes a TSV
(escaped_text<TAB>ids) that the Mojo skeptic runner consumes. Also prints a
human-readable dump.

Usage: pixi run python serenitymojo/tokenizer/parity/skeptic_oracle.py
Writes: serenitymojo/tokenizer/parity/skeptic_cases.tsv
"""
import os
from tokenizers import Tokenizer

JSON = ("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
        "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json")

CASES = [
    # --- long descriptive image-gen prompts ---
    "a photorealistic portrait of a young woman with freckles, golden hour lighting, shot on Kodak Portra 400, shallow depth of field, bokeh background, highly detailed skin texture",
    "epic fantasy landscape, towering snow-capped mountains, a winding river, dramatic storm clouds, volumetric god rays, ultra wide angle, 8k, trending on artstation",
    "cyberpunk city street at night, neon signs reflecting in puddles, rain, a lone figure in a trench coat, cinematic, blade runner aesthetic, moody atmosphere",
    "cute corgi puppy wearing a tiny wizard hat, studio lighting, white background, product photography, adorable, fluffy",
    # --- prompt weighting syntax ---
    "(masterpiece:1.2), (best quality:1.4), 1girl, solo, long hair",
    "((detailed eyes)), (sharp focus:1.1), portrait, [blurry:0.5]",
    "a cat:1.3 and a dog:0.7 sitting together",
    "{red|blue|green} dress, wildcard prompt",
    # --- negative-prompt-ish ---
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
    # --- mixed CJK + Latin ---
    "a beautiful 桜 cherry blossom tree in 京都 Kyoto Japan",
    "cute 猫 cat with 大きな big eyes, kawaii style",
    "한국 traditional 한복 hanbok dress, elegant",
    "中文 prompt with English mixed 测试 test 123",
    "ロボット robot in a 未来 future city, anime style",
    # --- repeated punctuation ---
    "amazing!!! incredible!!! wow???",
    "wait... what?? really...!!",
    "so good~~~ love it<3<3",
    "----- divider ----- text",
    "###### heading ######",
    # --- numbers / resolutions ---
    "4k 8K resolution 1920x1080 and 3840x2160",
    "render at 512x512 then upscale to 2048x2048",
    "f/1.8 aperture, ISO 100, 1/200s shutter, 85mm lens",
    "$19.99 sale, 50% off, was $39.99",
    "year 2026, version 2.5.1, build #4096",
    "temperature -40 to 120 degrees, 99.9% accuracy",
    # --- hashtags / social ---
    "sunset vibes #photography #naturelovers #goldenhour #nofilter",
    "@username posted a #cool pic",
    "trending: #AI #art #midjourney #stablediffusion",
    # --- quotes / apostrophes ---
    "she said “hello there” with a smile",  # curly double quotes
    "it’s a beautiful day, isn’t it",         # curly apostrophe
    "it's a beautiful day, isn't it",                    # straight apostrophe
    "'tis the season to be jolly",
    "DON'T PANIC and CARRY ON",
    "the 'quoted' word and the \"double quoted\" word",
    "rock 'n' roll all night",
    "y'all'd've been there",
    # --- URLs / paths / code ---
    "https://example.com/image.png?width=512&height=512",
    "see https://github.com/user/repo/blob/main/file.py#L42",
    "C:\\Users\\name\\Pictures\\img.jpg",
    "/home/user/projects/model.safetensors",
    "ftp://files.example.org:21/path",
    # --- whitespace edge cases ---
    "   leading spaces here",
    "trailing spaces here   ",
    "  both  ends  spaced  ",
    "\ttab indented prompt",
    "trailing tab\t",
    "mixed\t \tspaces and tabs",
    "line one\nline two\nline three",
    "windows\r\nline\r\nendings",
    "\n\nleading newlines",
    "double  spaces   triple    quad",
    # --- empty / single ---
    "",
    " ",
    "a",
    "1",
    ".",
    "!",
    "é",  # single precomposed e-acute
    "\U0001F600",  # single emoji
    "中",  # single CJK
    # --- contraction-with-capital edge cases (item 4) ---
    "DON'T",
    "CAN'T WON'T SHAN'T",
    "I'M HE'S THEY'RE WE'VE YOU'LL SHE'D",
    "It'S a TeSt'Re mIxEd'Ve",
    # --- punct-then-letter boundaries / multi-space ---
    "word.word,word;word:word",
    "(open)[bracket]{brace}<angle>",
    "a.b.c.d.e",
    "...leading dots word",
    "word!!!and more",
    # --- special tokens inline ---
    "<|im_start|>system prompt here<|im_end|>",
    "use <think> reasoning </think> then answer",
    "tool: <tool_call>do thing</tool_call> done",
    "two<|im_start|><|im_end|>adjacent",
    "partial <|im_ token here",
    # --- emoji clusters / ZWJ ---
    "family emoji \U0001F468‍\U0001F469‍\U0001F467 here",  # ZWJ family
    "flag \U0001F1FA\U0001F1F8 emoji",  # regional indicator US flag
    "skin tone \U0001F44D\U0001F3FD thumbs up",
]


def esc(s: str) -> str:
    return (s.replace("\\", "\\\\")
             .replace("\t", "\\t")
             .replace("\n", "\\n")
             .replace("\r", "\\r"))


def main():
    tk = Tokenizer.from_file(JSON)
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "skeptic_cases.tsv")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in CASES:
            ids = tk.encode(s).ids
            f.write(esc(s) + "\t" + ",".join(str(i) for i in ids) + "\n")
    print(f"wrote {len(CASES)} cases to {out_path}")


if __name__ == "__main__":
    main()
