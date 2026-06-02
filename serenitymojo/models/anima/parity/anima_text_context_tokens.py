#!/usr/bin/env python3
# serenitymojo/models/anima/parity/anima_text_context_tokens.py
#
# DEV-TIME tokenizer sidecar exporter for the Anima text path (Chunk C).
# The two HF tokenizers OneTrainer uses (Qwen2Tokenizer for Qwen3 input ids,
# T5TokenizerFast for the adapter query ids) are NOT ported to Mojo. This helper
# tokenizes a prompt at max_length=512 (padding='max_length', truncation) EXACTLY
# as AnimaModel.encode_text (AnimaModel.py:190-208) and writes the three id
# arrays to a safetensors the Mojo pipeline (pipeline/anima_text_context.mojo)
# consumes:
#   qwen_input_ids       [512]  F32-encoded ints   (Qwen3 encoder input)
#   qwen_attention_mask  [512]  F32 0/1            (pad-zeroing mask)
#   t5_input_ids         [512]  F32-encoded ints   (adapter query ids)
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/anima_text_context_tokens.py \
#       "a prompt" /tmp/anima_tokens.safetensors

import sys
import torch
from transformers import AutoTokenizer, T5TokenizerFast
from safetensors.torch import save_file

PROMPT_MAX_LENGTH = 512


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a photo of a cat"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/anima_tokens.safetensors"

    # Qwen3-0.6B tokenizer (Qwen2Tokenizer class). The 0.6B checkpoint shares the
    # Qwen2 BPE vocab; AnimaModel uses self.tokenizer = Qwen2Tokenizer.
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    t5tok = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

    q = qtok([prompt], max_length=PROMPT_MAX_LENGTH, padding="max_length",
             truncation=True, return_tensors="pt")
    t = t5tok([prompt], max_length=PROMPT_MAX_LENGTH, padding="max_length",
              truncation=True, return_tensors="pt")

    qids = q.input_ids[0].to(torch.float32)
    qmask = q.attention_mask[0].to(torch.float32)
    t5ids = t.input_ids[0].to(torch.float32)

    print("qwen ids  :", qids.shape, "nonpad", int(qmask.sum().item()))
    print("t5 ids    :", t5ids.shape, "nonpad", int((t.attention_mask[0] != 0).sum().item()))

    save_file({
        "qwen_input_ids": qids.contiguous(),
        "qwen_attention_mask": qmask.contiguous(),
        "t5_input_ids": t5ids.contiguous(),
    }, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
