#!/usr/bin/env python3
# parity/qwenimage_te_streamed_oracle.py — torch reference for the streamed
# Qwen-Image TE gate (pair of qwenimage_te_streamed_parity.mojo).
#
# Loads the Qwen2.5-VL text tower from the qwen-image-2512 text_encoder dir
# (CPU float32 — the 7B tower does not fit this 16 GB card, which is the whole
# reason the streamed Mojo path exists), runs the SAME padded 546-token ids the
# Mojo smoke dumped, and compares last_hidden_state (== hidden_states[27] +
# final model.norm) against the Mojo dump over the REAL rows only (pad rows are
# garbage in both stacks and masked by the DiT via real_len).
#
# Run with a CUDA-torch venv interpreter (model still on CPU):
#   /home/alex/SerenityTrainer/venv/bin/python \
#     serenitymojo/models/text_encoder/parity/qwenimage_te_streamed_oracle.py
#
# PASS bar: cos >= 0.999 (repo parity standard).

import sys

import numpy as np
import torch

TEXT_ENCODER_DIR = "/home/alex/.serenity/models/checkpoints/qwen-image-2512/text_encoder"
IDS_PATH = "/tmp/qwenimage_te_parity.ids.txt"
HID_PATH = "/tmp/qwenimage_te_parity.mojo_f32.bin"
PAD_ID = 151643
N_ENC = 546
HIDDEN = 3584
DROP_IDX = 34  # qwenimage template-drop: the DiT consumes rows [34, real_len)
COS_BAR = 0.999
# Repo standard (mojo-port): gate vs a BF16 reference oracle — both stacks
# store BF16, so a float32 reference mixes dtype error into the parity number.
# Pass "f32" as argv[1] to run the float32 ablation.
DTYPE = torch.float32 if (len(sys.argv) > 1 and sys.argv[1] == "f32") else torch.bfloat16


def main():
    ids = [int(x) for x in open(IDS_PATH).read().split()]
    assert len(ids) == N_ENC, f"expected {N_ENC} ids, got {len(ids)}"
    real_len = N_ENC
    for i, t in enumerate(ids):
        if t == PAD_ID:
            real_len = i
            break
    print(f"[oracle] ids={len(ids)} real_len={real_len}")

    mojo = np.fromfile(HID_PATH, dtype=np.float32)
    assert mojo.size == N_ENC * HIDDEN, f"mojo dump numel {mojo.size}"
    mojo = mojo.reshape(N_ENC, HIDDEN)

    from transformers import Qwen2_5_VLForConditionalGeneration

    print(f"[oracle] loading Qwen2.5-VL text tower (CPU {DTYPE})…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        TEXT_ENCODER_DIR, torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    model.eval()

    input_ids = torch.tensor([ids], dtype=torch.long)
    attn = torch.zeros(1, N_ENC, dtype=torch.long)
    attn[0, :real_len] = 1

    with torch.no_grad():
        # Text-only path: drive the language tower directly (no vision inputs).
        # transformers layout differs across versions; try the known access
        # paths for the Qwen2.5-VL text model.
        lm = None
        for attr in ("language_model", "model"):
            cand = getattr(model, attr, None)
            if cand is not None and hasattr(cand, "embed_tokens"):
                lm = cand
                break
            if cand is not None:
                inner = getattr(cand, "language_model", None)
                if inner is not None and hasattr(inner, "embed_tokens"):
                    lm = inner
                    break
        assert lm is not None, "could not locate the Qwen2.5-VL text tower"
        out = lm(input_ids=input_ids, attention_mask=attn)
        ref = out.last_hidden_state[0].float().numpy()  # [546, 3584]

    def _cos(lo, hi):
        a = mojo[lo:hi].reshape(-1).astype(np.float64)
        b = ref[lo:hi].reshape(-1).astype(np.float64)
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        m = float(np.max(np.abs(a - b)))
        return c, m

    cos_all, mad_all = _cos(0, real_len)
    print(f"[oracle] cos={cos_all:.6f} max_abs_diff={mad_all:.4f} over rows [0,{real_len}) (full real seq)")
    # The gate region is what the DiT consumes: rows [DROP_IDX, real_len)
    # (the template rows are dropped by the qwenimage slice contract).
    cos_kept, mad_kept = _cos(DROP_IDX, real_len)
    print(f"[oracle] cos={cos_kept:.6f} max_abs_diff={mad_kept:.4f} over rows [{DROP_IDX},{real_len}) (DiT-consumed)")
    if cos_kept >= COS_BAR:
        print(f"PASS: DiT-consumed cos {cos_kept:.6f} >= {COS_BAR}")
        return 0
    print(f"FAIL: DiT-consumed cos {cos_kept:.6f} < {COS_BAR}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
