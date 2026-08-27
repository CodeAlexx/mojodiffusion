#!/usr/bin/env python3
# Stamp ai-toolkit-style metadata onto existing serenitymojo PEFT LoRAs
# (tensors byte-identical; header only). Usage: stamp.py <file> <name> <step>
import json
import sys

from safetensors.torch import load_file, save_file

f, name, step = sys.argv[1], sys.argv[2], sys.argv[3]
sd = load_file(f)
md = {
    "name": name,
    "ss_output_name": name,
    "format": "pt",
    "training_info": json.dumps({"step": int(step)}),
    "ss_base_model_version": "minimax_h3_fl2va",
    "ss_network_dim": "32",
    "ss_network_alpha": "32",
    "software": json.dumps({"name": "serenitymojo", "repo": "https://github.com/CodeAlexx/mojodiffusion"}),
}
save_file(sd, f, metadata=md)
print("stamped", f)
