#!/usr/bin/env python
# oracle_text_ids.py — capture the EXACT Qwen3-VL input_ids + crop_start for the
# apple T2I prompt, WITHOUT loading the 8.9GB encoder (processor only).
#
# The captured prompt_embeds in oracle_e.safetensors (key `prompt_embeds`,
# [1,457,2560]) came from the torch Qwen3-VL fed these ids. This script
# reproduces oracle_e_t2i.py's tokenization VERBATIM (same PROMPT, same
# PROMPT_TEMPLATE, same processor call, same crop_start trick) so the Mojo
# text-encoder probe runs on IDENTICAL tokens. Verify: crop_start + 457 == len.
#
# Run:
#   /home/alex/SerenityTrainer/venv/bin/python \
#     /home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/oracle_text_ids.py
import os, sys, json

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
sys.path.insert(0, OUT)                                  # import oracle_e_t2i
sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")

import torch
from safetensors.torch import save_file
from transformers import AutoProcessor

# Reuse the oracle's PROMPT / template / crop helper verbatim (module import runs
# no model load — the 8.9GB encoder is only touched inside oracle_e_t2i.main()).
from oracle_e_t2i import (
    PROMPT, PROMPT_TEMPLATE, _apply_template, _compute_crop_start,
    TOKEN_LENGTH, PROC_DIR,
)


def main():
    print(f"[ids] loading processor from {PROC_DIR}", flush=True)
    processor = AutoProcessor.from_pretrained(PROC_DIR, trust_remote_code=True)

    crop_start = _compute_crop_start(processor)
    print(f"[ids] crop_start = {crop_start}", flush=True)

    text = _apply_template(PROMPT)
    inputs = processor(
        text=[text], images=None, videos=None, do_resize=False, truncation=True,
        max_length=TOKEN_LENGTH, padding="longest", return_tensors="pt",
    )
    ids = inputs["input_ids"][0].tolist()
    true_len = len(ids)
    post = true_len - crop_start
    print(f"[ids] true_len = {true_len}  post-crop = {post}", flush=True)

    # 151643 is the pad id the Mojo encoder uses to auto-detect real_len; it must
    # NOT appear in the real token span or padding detection would truncate early.
    assert 151643 not in ids, "pad id 151643 present in real tokens — pick another pad id"
    assert post == 457, f"post-crop length {post} != 457 (oracle prompt_embeds L)"

    save_file({
        "input_ids": torch.tensor(ids, dtype=torch.int32),
        "crop_start": torch.tensor([crop_start], dtype=torch.int32),
        "true_len": torch.tensor([true_len], dtype=torch.int32),
    }, os.path.join(OUT, "oracle_text_ids.safetensors"))
    json.dump(
        {"input_ids": ids, "crop_start": crop_start, "true_len": true_len},
        open(os.path.join(OUT, "oracle_text_ids.json"), "w"),
    )
    print(f"[ids] SAVED oracle_text_ids.safetensors + .json "
          f"(len={true_len}, crop_start={crop_start}, post=457)", flush=True)


if __name__ == "__main__":
    main()
