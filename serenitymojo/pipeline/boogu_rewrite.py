#!/usr/bin/env python
# boogu_rewrite.py — Boogu prompt ENHANCER (instruction reasoner / rewriter) using
# the LOCAL 8B mllm already on disk (NO 32B download). Reuses Boogu's own EN rewrite
# system prompt; expands a short/rough prompt into a fuller one. The rewriter's rules
# already avoid the white-border traps (no "negative space" words, no negations).
# Output is plain text -> feed it to the Mojo pipeline as INSTRUCTION.
#
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/pipeline/boogu_rewrite.py "<short prompt>"
import os, sys
os.environ.setdefault("device", "cuda:0")
sys.path.insert(0, "/home/alex/Boogu-Image")
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from utils.t2i_external_prompt_rewriter import build_messages   # reuse Boogu's EN system prompt

MLLM = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/mllm"   # the 8B already downloaded


def main():
    seed = sys.argv[1] if len(sys.argv) > 1 else "a lone red fox curled asleep in a snowy pine forest at dawn"
    print(f"[rewrite] seed: {seed}\n")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MLLM, dtype=torch.bfloat16, attn_implementation="sdpa"   # flash_attn absent -> sdpa
    ).to("cuda:0").eval()
    processor = AutoProcessor.from_pretrained(MLLM)
    inputs = processor.apply_chat_template(
        build_messages(seed, lang="en"),
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)
    text = processor.decode(gen[0][len(inputs.input_ids[0]):], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip().replace("\n", " ")
    print("[rewrite] enhanced prompt:\n")
    print(text)


if __name__ == "__main__":
    main()
