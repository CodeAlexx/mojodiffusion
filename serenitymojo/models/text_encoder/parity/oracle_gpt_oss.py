#!/usr/bin/env python
"""HF-transformers oracle for the GPT-OSS (microsoft/Lens) text encoder.

Loads GptOssForCausalLM from the .serenity text_encoder dir on CUDA in bf16,
forces the MXFP4 -> bf16 dequant on load (Mxfp4Config(dequantize=True), since an
RTX 3090 Ti / sm_86 has no native MXFP4 triton path), runs a fixed prompt's
token ids (SAME ids the Mojo driver uses, from the raw tokenizer.json), and
captures hidden_states at the requested capture layers.

HF hidden_states tuple is length num_layers+1: index 0 = embeddings (input to
layer 0), index i+1 = OUTPUT of layer i (post-residual, pre-final-norm). The
Mojo encoder captures the post-residual hidden after layer `li` completes, which
== hidden_states[li+1]. So Lens capture layers [5,11,17,23] map to HF indices
[6,12,18,24]. We also dump the final (hidden_states[-1] = output of layer 23,
which is hidden_states[24]) — for Lens that IS layer 23, so "final" == l23 here;
we additionally dump the final-normed last hidden for reference.

Writes oracle_captures.safetensors (bf16) with keys l5,l11,l17,l23 plus
oracle_final_normed for inspection, and prints token ids + shapes + stats.
"""
import sys
import json
import torch
from tokenizers import Tokenizer
from transformers import GptOssForCausalLM
from transformers import Mxfp4Config
from safetensors.torch import save_file

TE_DIR = "/home/alex/.serenity/models/microsoft_lens/text_encoder"
TOK = "/home/alex/.serenity/models/microsoft_lens/tokenizer/tokenizer.json"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/oracle_captures.safetensors"
PROMPT = "a photo of a cat"
CAPTURE_LAYERS = [5, 11, 17, 18, 19, 20, 21, 22, 23]  # 0-indexed layer outputs


def stats(t):
    f = t.float()
    return float(f.mean()), float(f.var(unbiased=False)), float(f.abs().max())


def main():
    tok = Tokenizer.from_file(TOK)
    enc = tok.encode(PROMPT)
    ids = enc.ids
    print(f"[oracle] prompt: {PROMPT!r}")
    print(f"[oracle] token ids: {ids}")
    print(f"[oracle] token strs: {enc.tokens}")
    print(f"[oracle] seq len: {len(ids)}")

    # Force MXFP4 -> bf16 dequant on load (no native MXFP4 on sm_86).
    # The fully-dequantized bf16 GPT-OSS-20B is ~60GB resident (MoE experts
    # expand ~4x off the 13.7GB MXFP4 on-disk) -> it does NOT fit on a 24GB GPU
    # as ONE resident model. The Mojo port avoids this by STREAMING one layer at
    # a time; HF transformers has no streamed forward. So the oracle runs the
    # bf16 forward on CPU (62GB RAM, plenty). Math is identical dtype (bf16
    # storage); CPU bf16 matmul accumulates in fp32 just like the GPU path, so
    # this is still a faithful bf16 oracle. 5 tokens x 24 layers is fast enough.
    qcfg = Mxfp4Config(dequantize=True)
    print("[oracle] loading GptOssForCausalLM on CPU (dequantize=True, bf16) ...")
    model = GptOssForCausalLM.from_pretrained(
        TE_DIR,
        dtype=torch.bfloat16,
        device_map="cpu",
        quantization_config=qcfg,
    )
    model.eval()
    print("[oracle] loaded on CPU. dtype check on a weight:",
          next(model.parameters()).dtype,
          "device:", next(model.parameters()).device)

    # HF GptOssModel appends hidden_states AFTER model.norm, so hidden_states[-1]
    # (the last-layer entry hs[24]) is POST final-norm. Lens / the Mojo encoder
    # (and the Rust ref, which skips model.norm) use the RAW pre-final-norm layer
    # output. Capture the last decoder layer's raw output via a forward hook.
    _raw_last = {}
    last_layer_idx = model.config.num_hidden_layers - 1  # 23

    def _last_layer_hook(_mod, _inp, out):
        _raw_last["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    model.model.layers[last_layer_idx].register_forward_hook(_last_layer_hook)

    input_ids = torch.tensor([ids], dtype=torch.long, device="cpu")

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    hs = out.hidden_states  # tuple len num_layers+1
    print(f"[oracle] num hidden_states: {len(hs)} (expect num_layers+1 = 25)")
    print(f"[oracle] hidden_states[0] shape (embeddings): {tuple(hs[0].shape)}")

    save = {}
    for li in CAPTURE_LAYERS:
        if li == last_layer_idx:
            # hs[li+1] would be POST model.norm; use the raw hooked layer output.
            h = _raw_last["h"]  # [1, S, hidden], pre-final-norm
            src = f"raw hook layers[{li}]"
        else:
            idx = li + 1  # output of layer li (post-residual, pre-final-norm)
            h = hs[idx]
            src = f"hidden_states[{idx}]"
        m, v, a = stats(h)
        print(f"[oracle] layer {li} -> {src} shape "
              f"{tuple(h.shape)} mean/var/absmax: {m:.6f} {v:.6f} {a:.6f}")
        save[f"l{li}"] = h.to(torch.bfloat16).contiguous().cpu()

    # final normed last hidden (after model.norm), for reference only.
    final_normed = out.last_hidden_state if hasattr(out, "last_hidden_state") else None
    if final_normed is None:
        # GptOssForCausalLM output has no last_hidden_state; run norm manually.
        last = hs[-1]
        try:
            final_normed = model.model.norm(last)
        except Exception as e:
            print("[oracle] could not apply final norm:", e)
            final_normed = last
    m, v, a = stats(final_normed)
    print(f"[oracle] final_normed shape {tuple(final_normed.shape)} "
          f"mean/var/absmax: {m:.6f} {v:.6f} {a:.6f}")
    save["oracle_final_normed"] = final_normed.to(torch.bfloat16).contiguous().cpu()

    save_file(save, OUT)
    print(f"[oracle] wrote {OUT}")
    # also dump ids for the mojo side to cross-check
    with open(OUT.replace(".safetensors", "_ids.json"), "w") as f:
        json.dump({"prompt": PROMPT, "ids": ids, "tokens": enc.tokens}, f)

    del model
    torch.cuda.empty_cache()
    print("[oracle] DONE")


if __name__ == "__main__":
    main()
