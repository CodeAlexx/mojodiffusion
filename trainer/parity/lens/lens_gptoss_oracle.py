#!/usr/bin/env python
"""GPT-OSS text-encoder oracle (Serenity-only) for the Lens Mojo encoder.

Runs LensGptOssEncoder (Serenity's lens.text_encoder dependency) on FIXED
input_ids and dumps the 4 selected-layer hidden states so the Mojo GPT-OSS
encoder port can be parity-gated on byte-identical token ids. S=160 (>128) so the
sliding-window-vs-full-causal mask alternation (even/odd layers) is exercised.

Run with the transformers-v5 env:
  /home/alex/ai-toolkit/venv/bin/python parity/lens/lens_gptoss_oracle.py
"""
import importlib.util
import json
import os
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TE_DIR = "/home/alex/.serenity/models/microsoft_lens/text_encoder"
LENS_TE_PY = "/home/alex/vendor-refs/Lens/lens/text_encoder.py"
SEL = [5, 11, 17, 23]
S = 160
SEED = 20260607


def main():
    spec = importlib.util.spec_from_file_location("lens_te", LENS_TE_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    enc = m.LensGptOssEncoder.from_pretrained(TE_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto").eval()
    enc.set_selected_layers(SEL)
    cfg = enc.config
    vocab = int(getattr(cfg, "vocab_size", 200000))

    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, min(vocab, 50000), (1, S), generator=g, dtype=torch.long)
    am = torch.ones(1, S, dtype=torch.long)

    with torch.no_grad():
        outs = enc.encode_layers(ids, am)   # list of 4 [1,S,2880] bf16

    dump = {"input_ids": ids.int().cpu().contiguous(), "attention_mask": am.int().cpu().contiguous()}
    stats = []
    for i, (li, feat) in enumerate(zip(SEL, outs)):
        dump[f"hidden_layer_{li:02d}"] = feat.float().cpu().contiguous()
        stats.append({"layer": li, "shape": list(feat.shape),
                      "mean": float(feat.float().mean()), "std": float(feat.float().std()),
                      "absmax": float(feat.float().abs().max())})
    save_file(dump, os.path.join(HERE, "gptoss_ref.safetensors"))
    meta = {
        "seed": SEED, "seq_len": S, "selected_layers": SEL, "hidden": 2880,
        "vocab_size": vocab,
        "sliding_window": int(getattr(cfg, "sliding_window", 128)),
        "num_layers": int(getattr(cfg, "num_hidden_layers", 24)),
        "layer_types": list(getattr(cfg, "layer_types", [])) if hasattr(cfg, "layer_types") else None,
        "per_layer": stats,
    }
    json.dump(meta, open(os.path.join(HERE, "gptoss_meta.json"), "w"), indent=2, default=str)
    print("GPTOSS ORACLE OK")
    for s in stats:
        print(f"  layer {s['layer']:2d}: shape {s['shape']} mean {s['mean']:.5f} std {s['std']:.5f} absmax {s['absmax']:.3f}")


if __name__ == "__main__":
    main()
