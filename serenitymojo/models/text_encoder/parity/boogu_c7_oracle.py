#!/usr/bin/env python
# boogu_c7_oracle.py — C7 (Qwen3-VL text encoder) parity oracle. Dev tool, NOT shipped.
#
# Loads the REAL mllm (Qwen3VLForConditionalGeneration, bf16) + Qwen3VLProcessor,
# builds the T2I chat template (system + user, text-only -> NO vision tower), runs
# mllm(output_hidden_states=True), takes all_hidden_states[-1] = instruction_feats
# [1,L,4096]. Dumps the EXACT input_ids (so the Mojo encoder gate uses byte-identical
# tokens, isolating tokenizer parity) + attention_mask + last_hidden_state.
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/text_encoder/parity/boogu_c7_oracle.py
import os
os.environ.setdefault("device", "cuda:0")

import numpy as np
import torch

MLLM_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/mllm"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boogu_dumps")
os.makedirs(OUT, exist_ok=True)

SYSTEM_PROMPT_4_T2I = ("You are a helpful assistant that generates high-quality images "
                       "based on user instructions. The instructions are as follows.")
INSTRUCTION = "A photorealistic portrait of an astronaut riding a horse on Mars."


def dump_f32(name, arr):
    v = np.asarray(arr).ravel().astype("<f4")
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(v.tobytes())
    return list(np.asarray(arr).shape)


def main():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    print(f"[c7-oracle] loading mllm (bf16) + processor from {MLLM_DIR}")
    processor = AutoProcessor.from_pretrained(MLLM_DIR, trust_remote_code=True)
    mllm = Qwen3VLForConditionalGeneration.from_pretrained(
        MLLM_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0").eval()

    prompt = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_4_T2I}]},
        {"role": "user", "content": [{"type": "text", "text": INSTRUCTION}]},
    ]
    vlm_inputs = processor.apply_chat_template(
        [prompt], padding="longest", padding_side="right",
        return_tensors="pt", tokenize=True, return_dict=True,
    )
    vlm_inputs = {k: (v.to("cuda:0") if isinstance(v, torch.Tensor) else v)
                  for k, v in vlm_inputs.items()}
    input_ids = vlm_inputs["input_ids"]
    attn = vlm_inputs["attention_mask"]
    print("[c7-oracle] input_ids shape", tuple(input_ids.shape),
          "| keys:", list(vlm_inputs.keys()))

    with torch.no_grad():
        out = mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)
    last_hidden = out.hidden_states[-1]            # [1, L, 4096]
    print("[c7-oracle] last_hidden", tuple(last_hidden.shape), last_hidden.dtype,
          "n_hidden_states", len(out.hidden_states))

    L = input_ids.shape[1]
    shapes = {}
    shapes["c7_input_ids.bin"] = dump_f32("c7_input_ids.bin", input_ids.cpu().numpy())
    shapes["c7_attn_mask.bin"] = dump_f32("c7_attn_mask.bin", attn.cpu().numpy())
    shapes["c7_last_hidden.bin"] = dump_f32("c7_last_hidden.bin", last_hidden.float().cpu().numpy())
    ids_list = input_ids[0].cpu().tolist()
    with open(os.path.join(OUT, "c7_meta.txt"), "w") as f:
        f.write(f"L={L} hidden=4096 n_hidden_states={len(out.hidden_states)}\n")
        f.write(f"system_prompt={SYSTEM_PROMPT_4_T2I!r}\n")
        f.write(f"instruction={INSTRUCTION!r}\n")
        f.write(f"input_ids={ids_list}\n")
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
        f.write(f"last_hidden.std={last_hidden.float().std():.5f}\n")
    print("[c7-oracle] dumped:", shapes)
    print(f"[c7-oracle] L={L} last_hidden.std={last_hidden.float().std():.5f}")
    print(f"[c7-oracle] input_ids[0]={ids_list}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
