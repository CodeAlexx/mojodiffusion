#!/usr/bin/env python
# mageflow_textcond_oracle.py — Mage-Flow Qwen3-VL TEXT conditioning parity oracle.
#
# Dev tool, NOT shipped. Replicates mage_flow's TextEncoder.forward text path
# EXACTLY for a single prompt (the packed/varlen case reduces to this for one
# sequence — cu_seqlens over a single segment == standard causal attention):
#   - load the REAL Mage-Flow text_encoder (Qwen3VLForConditionalGeneration, bf16),
#   - tokenize PROMPT_TEMPLATE["mage-flow"].template.format(PROMPT),
#   - run the base model -> last_hidden_state (POST model.norm),  [1, L, 2560]
#   - drop the leading start_idx=34 system-prompt rows -> txt [L-34, 2560],
#   - vec = txt.mean(0)  [2560].
# ref: mage_flow/models/modules/text_encoder.py  (TextEncoder.forward)
#      mage_flow/models/utils.py                 (PROMPT_TEMPLATE)
#
# Dumps the EXACT full input_ids (so the Mojo probe runs byte-identical tokens,
# isolating tokenizer parity) + txt + vec (cuda bf16 -> f32) as raw <f4 bins.
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/OneTrainer/venv/bin/python \
#     .claude/worktrees/agent-a8b4b1b2e92480f73/serenitymojo/models/text_encoder/parity/mageflow_textcond_oracle.py
import os
import gc

import numpy as np
import torch

TE_DIR = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/text_encoder"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mageflow_dumps")
os.makedirs(OUT, exist_ok=True)

# --- VERBATIM from mage_flow/models/utils.py PROMPT_TEMPLATE["mage-flow"] -----
MAGEFLOW_T2I_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
MAGEFLOW_T2I_START_IDX = 34
TOKENIZER_MAX_LENGTH = 1024  # TextEncoder default; short prompt -> no truncation
# -----------------------------------------------------------------------------

PROMPT = "A photorealistic portrait of an astronaut riding a horse on Mars."


def dump_f32(name, arr):
    v = np.asarray(arr).ravel().astype("<f4")
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(v.tobytes())
    return list(np.asarray(arr).shape)


@torch.no_grad()
def main():
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    print(f"[mageflow-te-oracle] loading Mage-Flow text_encoder (bf16) from {TE_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(TE_DIR)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        TE_DIR, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda:0").eval()
    qwen.requires_grad_(False)
    device = qwen.device

    # tokenize exactly like pipeline.py _encode_texts_packed (one prompt).
    text = MAGEFLOW_T2I_TEMPLATE.format(PROMPT)
    enc = tokenizer(
        text,
        max_length=TOKENIZER_MAX_LENGTH + MAGEFLOW_T2I_START_IDX,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    input_ids = enc["input_ids"]  # [1, L]
    L = input_ids.shape[1]

    # Base model forward -> last_hidden_state (POST model.norm). This is exactly
    # `outputs.last_hidden_state` in TextEncoder.forward (embedding output mode).
    # position_ids=None -> HF get_rope_index yields text-only arange, identical
    # to TextEncoder's explicit arange per sequence.
    out = qwen.model(input_ids=input_ids, use_cache=False)
    hidden = out.last_hidden_state.squeeze(0)  # [L, 2560]

    drop = MAGEFLOW_T2I_START_IDX
    txt = hidden[drop:]           # [L-34, 2560]  == the DiT context
    vec = txt.mean(dim=0)         # [2560]        pooled

    print(
        f"[mageflow-te-oracle] L={L} drop_idx={drop} -> keep={txt.shape[0]} "
        f"hidden={hidden.shape[-1]} dtype={hidden.dtype}"
    )

    shapes = {}
    shapes["mageflow_input_ids.bin"] = dump_f32(
        "mageflow_input_ids.bin", input_ids.cpu().numpy()
    )
    shapes["mageflow_txt.bin"] = dump_f32("mageflow_txt.bin", txt.float().cpu().numpy())
    shapes["mageflow_vec.bin"] = dump_f32("mageflow_vec.bin", vec.float().cpu().numpy())

    ids_list = input_ids[0].cpu().tolist()
    with open(os.path.join(OUT, "mageflow_te_meta.txt"), "w") as f:
        f.write(
            f"L_full={L} drop_idx={drop} L_keep={txt.shape[0]} hidden=2560\n"
        )
        f.write(f"prompt={PROMPT!r}\n")
        f.write(f"template=mage-flow\n")
        f.write(f"input_ids={ids_list}\n")
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
        f.write(f"txt.std={txt.float().std():.6f} txt.mean={txt.float().mean():.6f}\n")
        f.write(f"vec.std={vec.float().std():.6f} vec.mean={vec.float().mean():.6f}\n")
    print(f"[mageflow-te-oracle] dumped: {shapes}")
    print(
        f"[mageflow-te-oracle] txt.std={txt.float().std():.6f} "
        f"vec.std={vec.float().std():.6f}"
    )
    print(f"[mageflow-te-oracle] input_ids[0]={ids_list}")

    # Free GPU immediately (shared 16 GB card).
    del qwen, out, hidden, txt, vec
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
