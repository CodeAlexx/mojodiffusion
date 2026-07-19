#!/usr/bin/env python
# gen_krea2_te.py — Krea-2 Qwen3-VL-4B text-encoder parity oracle. Dev tool, NOT shipped.
#
# Replicates ai-toolkit krea2/src/text_encoder.py::encode_krea_prompt EXACTLY:
#   - load the REAL Qwen3-VL-4B-Instruct (bf16), drop the vision tower (text-only),
#   - tokenize PROMPT_TEMPLATE_ENCODE_PREFIX + prompt and (separately, via the
#     Qwen2TokenizerFast "processor") the assistant suffix; concat -> input_ids,
#   - run mllm(output_hidden_states=True), stack hidden_states[SELECT_LAYERS] on
#     dim=2 -> (1,L,12,2560), drop the leading 34 system-prefix rows
#     (PROMPT_TEMPLATE_ENCODE_START_IDX) -> (1, L-34, 12, 2560).
# Dumps the EXACT input_ids (so the Mojo encoder gate uses byte-identical tokens,
# isolating tokenizer parity) + the (1, L', 12, 2560) stack on cuda bf16->f32.
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/text_encoder/parity/gen_krea2_te.py
import os
os.environ.setdefault("device", "cuda:0")

import numpy as np
import torch

TE_PATH = "Qwen/Qwen3-VL-4B-Instruct"  # resolves to the local HF cache snapshot
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "krea2_dumps")
os.makedirs(OUT, exist_ok=True)

# --- VERBATIM from krea2/src/text_encoder.py ---------------------------------
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PROMPT_TEMPLATE_ENCODE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT_TEMPLATE_ENCODE_START_IDX = 34
MAX_LENGTH = 512
# -----------------------------------------------------------------------------

PROMPT = "A photorealistic portrait of an astronaut riding a horse on Mars."


def dump_f32(name, arr):
    v = np.asarray(arr).ravel().astype("<f4")
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(v.tobytes())
    return list(np.asarray(arr).shape)


@torch.no_grad()
def main():
    from transformers import (
        AutoTokenizer,
        Qwen2TokenizerFast,
        Qwen3VLForConditionalGeneration,
    )

    print(f"[krea2-te-oracle] loading Qwen3-VL-4B (bf16) + tokenizers from {TE_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TE_PATH, max_length=MAX_LENGTH)
    processor = Qwen2TokenizerFast.from_pretrained(TE_PATH, max_length=MAX_LENGTH)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        TE_PATH, torch_dtype=torch.bfloat16
    ).to("cuda:0").eval()
    # Text-only: drop the vision tower (krea2.py:213-216).
    if getattr(qwen.model, "visual", None) is not None:
        qwen.model.visual = None
    qwen.requires_grad_(False)

    device = qwen.device

    # --- encode_krea_prompt (text_encoder.py:56-84), byte-for-byte ---
    suffix_inputs = processor(
        text=[PROMPT_TEMPLATE_ENCODE_SUFFIX], return_tensors="pt"
    ).to(device)
    suffix_ids = suffix_inputs["input_ids"]
    suffix_mask = suffix_inputs["attention_mask"].bool()

    text = PROMPT_TEMPLATE_ENCODE_PREFIX + PROMPT
    inputs = tokenizer(
        [text],
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        max_length=MAX_LENGTH + PROMPT_TEMPLATE_ENCODE_START_IDX,
        return_tensors="pt",
    ).to(device)

    input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
    mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)

    states = qwen(
        input_ids=input_ids, attention_mask=mask, output_hidden_states=True
    )
    n_hs = len(states.hidden_states)
    # (1, L, num_layers, hidden)
    hiddens = torch.stack(
        [states.hidden_states[i] for i in SELECT_LAYERS], dim=2
    )
    # Drop the system-prefix tokens: prompt + suffix conditioning remain.
    hiddens = hiddens[:, PROMPT_TEMPLATE_ENCODE_START_IDX:]  # (1, L', 12, 2560)
    # -----------------------------------------------------------------

    L_full = input_ids.shape[1]
    L_keep = hiddens.shape[1]
    print(
        f"[krea2-te-oracle] L_full={L_full} (prefix={PROMPT_TEMPLATE_ENCODE_START_IDX} "
        f"-> keep={L_keep}), n_hidden_states={n_hs}, stack={tuple(hiddens.shape)} "
        f"{hiddens.dtype}"
    )

    shapes = {}
    # Dump the FULL input_ids (prefix+prompt+suffix) so the Mojo encoder runs the
    # identical sequence and applies the same DROP_IDX slice.
    shapes["krea2_input_ids.bin"] = dump_f32(
        "krea2_input_ids.bin", input_ids.cpu().numpy()
    )
    shapes["krea2_te_stack.bin"] = dump_f32(
        "krea2_te_stack.bin", hiddens.float().cpu().numpy()
    )
    ids_list = input_ids[0].cpu().tolist()
    suffix_list = suffix_ids[0].cpu().tolist()
    with open(os.path.join(OUT, "krea2_te_meta.txt"), "w") as f:
        f.write(
            f"L_full={L_full} drop_idx={PROMPT_TEMPLATE_ENCODE_START_IDX} "
            f"L_keep={L_keep} n_layers_stacked=12 hidden=2560 "
            f"n_hidden_states={n_hs}\n"
        )
        f.write(f"select_layers={SELECT_LAYERS}\n")
        f.write(f"prompt={PROMPT!r}\n")
        f.write(f"input_ids={ids_list}\n")
        f.write(f"suffix_ids={suffix_list}\n")
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
        f.write(f"stack.std={hiddens.float().std():.6f}\n")
        f.write(f"stack.mean={hiddens.float().mean():.6f}\n")
    print(f"[krea2-te-oracle] dumped: {shapes}")
    print(
        f"[krea2-te-oracle] stack.std={hiddens.float().std():.6f} "
        f"mean={hiddens.float().mean():.6f}"
    )
    print(f"[krea2-te-oracle] input_ids[0]={ids_list}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
