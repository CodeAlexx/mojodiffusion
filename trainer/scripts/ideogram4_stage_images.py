#!/usr/bin/env python3
# ideogram4_stage_images.py — stage A of the Ideogram-4 cache prepare.
# (Stage B = smoke/ideogram4_prepare_cache.mojo: VAE + Qwen3-VL encoders.)
#
# No JPEG decoder exists in the Mojo stack yet (the same gap that
# cache-blocks zimage/l2p prepare), so this one-shot stager does the
# image decode + caption templating in Python and hands Mojo a single
# safetensors of f32 image tensors + a rendered-prompts JSON.
#
# Per sample (dataset: <dir>/N.jpg + N.txt):
#   image: center-crop to square, resize SIZExSIZE, RGB f32 [-1,1], CHW
#          -> image.<i> [1,3,SIZE,SIZE] f32
#   caption: .txt wrapped in the minimal Ideogram-4 structured-JSON schema
#          (high_level_description first — key order is load-bearing), then
#          the Qwen3-VL chat template the trainer's tokenizer probe uses.
#          -> prompts.json {"<i>": rendered_string, ...}
#
# --uncond (T1.D caption dropout, flag-gated, default-off): ALSO write
#   uncond.txt = the empty-caption ("") render through the SAME schema +
#   chat-template pipeline as the real captions (SimpleTuner precedent:
#   helpers/models/common.py encode_dropout_caption encodes "" through the
#   normal _encode_prompts path). Stage B (ideogram4_prepare_cache.mojo)
#   reads it to encode the llm_uncond cache tensor; trainers substitute it
#   when the seeded dropout schedule fires. Per-sample outputs unchanged.
#
# Run:
#   /home/alex/EriDiffusion/.venv_cache/bin/python \
#     scripts/ideogram4_stage_images.py /home/alex/datasets/gigerver3 \
#     /home/alex/trainings/ideogram4_giger_stage 512 [--uncond]

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

# ai-toolkit (the production oracle) digests the caption before templating:
# plain-text captions pass through unchanged, structured JSON is minified
# (separators=(",",":")). The previous json.dumps({"high_level_description":...})
# wrap injected ~30% extra JSON-scaffold tokens (measured 30→23 ids) that leaked
# into the maskless DiT. Use ai-toolkit's exact digest.
sys.path.insert(0, "/home/alex/ai-toolkit")
from toolkit.ideogram_caption import digest_caption_string


def render_prompt(caption: str) -> str:
    """Caption -> ai-toolkit digest -> Qwen3-VL chat template.
    The ONE render pipeline for real captions and the --uncond empty render.

    Mirrors ai-toolkit extensions_built_in/diffusion_models/ideogram4/
    ideogram4.py::get_prompt_embeds (522-526):
      p = digest_caption_string(prompt)
      apply_chat_template([{role:user, content:[{type:text, text:p}]}],
                          add_generation_prompt=True, tokenize=False)
    The Qwen3-VL chat_template.jinja for a single user text message with
    add_generation_prompt=True (and no system message) expands EXACTLY to
    "<|im_start|>user\\n{p}<|im_end|>\\n<|im_start|>assistant\\n" — no default
    system prompt is injected. GATE-verified: this produces token-ids identical
    to apply_chat_template (23 ids vs the json-wrap's 30)."""
    p = digest_caption_string(caption)
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"


def main():
    args = [a for a in sys.argv[1:] if a != "--uncond"]
    emit_uncond = "--uncond" in sys.argv[1:]
    src = Path(args[0])
    out = Path(args[1])
    size = int(args[2]) if len(args) > 2 else 512
    out.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(src.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30)
    tensors, prompts, captions = {}, {}, {}
    kept = 0
    for p in jpgs:
        cap_path = p.with_suffix(".txt")
        if not cap_path.exists():
            print(f"skip {p.name}: no caption")
            continue
        img = Image.open(p).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((size, size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [H,W,3] in [-1,1]
        arr = arr.transpose(2, 0, 1)[None]                       # [1,3,H,W]
        tensors[f"image.{kept}"] = arr

        caption = cap_path.read_text().strip()
        # Minimal structured caption (schema: high_level_description first).
        prompts[str(kept)] = render_prompt(caption)
        captions[str(kept)] = caption
        kept += 1

    save_file(tensors, str(out / "images.safetensors"))
    (out / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=0))
    for k, v in prompts.items():
        (out / f"prompt.{k}.txt").write_text(v)
    for k, v in captions.items():
        (out / f"caption.{k}.txt").write_text(v)
    if emit_uncond:
        (out / "uncond.txt").write_text(render_prompt(""))
        print(f"staged uncond.txt (empty-caption render) -> {out}")
    print(f"staged {kept} samples -> {out} (size {size})")

if __name__ == "__main__":
    main()
