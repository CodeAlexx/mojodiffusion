# GPU-bf16 oracle for the LTX-2.5 text encoder — Lightricks' OWN weights.
#
# Why this exists: the LTX-2.5 encoder is FINE-TUNED, not stock google. Measured
# byte diff vs `google/gemma-4-12B-it`: norm.weight differs in all 3840 elements,
# layers.23.mlp.down_proj in 57.4M of 59M, embed_tokens in ~3% of rows. (The
# frozen bits — layer_scalar, q_norm/k_norm — DO match, which is what makes it
# look stock at a glance.) So the earlier 0.99923 gate, taken against google
# weights, proves the architecture port only and carries NO numerical claim
# about LTX-2.5. This dump re-anchors the gate on the real checkpoint.
#
# The LTX file is a single-file/ComfyUI export: keys are `model.*` (not
# `model.language_model.*`) and it also carries text_embedding_projection.*,
# vision_model.*, audio_projector.*, multi_modal_projector.* and embedded U8
# assets that `Gemma4UnifiedTextModel` will not accept. We remap + filter in
# memory rather than writing a second 24 GB checkpoint to disk.
#
# Run:
#   PYTHONPATH=/home/alex/ltx25-parity-pkgs \
#     /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/text_encoder/parity/gemma4_ltx25_oracle_dump.py

import os
import torch
from safetensors.torch import save_file, safe_open

CFG = "/home/alex/.serenity/models/text_encoders/gemma-4-12b-it-standalone"  # arch only
LTX = "/home/alex/.serenity/models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma4_refs_ltx25")
MAX_TOKENS = 1024
PROMPT = (
    "A woman with long brown hair walks through a sunlit forest, leaves "
    "crunching underfoot as birdsong echoes between the tall pines."
)

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

from transformers import AutoTokenizer, Gemma4UnifiedForConditionalGeneration  # noqa: E402

tok = AutoTokenizer.from_pretrained(CFG)
# BOS must be prepended by hand: this checkpoint family sets add_bos_token=False
# AND ships an empty post_processor, so add_special_tokens=True is a silent
# no-op. Gating without BOS costs Gemma its attention sink and looks exactly
# like depth-accumulated drift (that mistake cost a full diagnosis pass).
BOS = tok.bos_token_id
assert BOS == 2, BOS
ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
assert ids[0] != BOS
ids = [BOS] + ids
real_len = len(ids)
pad = MAX_TOKENS - real_len
input_ids = torch.tensor([[0] * pad + ids], dtype=torch.long, device="cuda")
attention_mask = torch.tensor([[0] * pad + [1] * real_len], dtype=torch.long, device="cuda")
print("real_len", real_len, "| first ids", ids[:6])

model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    CFG, dtype=torch.bfloat16, device_map="cpu"
)
model.eval()
text_model = model.model.language_model

# --- swap in Lightricks' fine-tuned tower ---
want = dict(text_model.state_dict())
loaded, missing = 0, []
with safe_open(LTX, framework="pt", device="cpu") as f:
    have = set(f.keys())
    for name in want:
        src = "model." + name          # text_model params are `layers.N...`, `norm.weight`, `embed_tokens.weight`
        if src not in have:
            missing.append(name)
            continue
        t = f.get_tensor(src)
        if tuple(t.shape) != tuple(want[name].shape):
            raise SystemExit(f"shape mismatch {name}: ltx{tuple(t.shape)} vs {tuple(want[name].shape)}")
        want[name] = t.to(torch.bfloat16)
        loaded += 1
print(f"loaded {loaded} LTX tensors into the tower; missing {len(missing)}")
if missing:
    print("  first missing:", missing[:8])
    raise SystemExit("refusing to dump an oracle from a partially-loaded tower")
text_model.load_state_dict(want, strict=True)
print("tower now carries LTX-2.5 weights")

from accelerate import cpu_offload  # noqa: E402
cpu_offload(text_model, execution_device=torch.device("cuda:0"))

captures = {}


def cap(idx):
    layer = text_model.layers[idx]
    layer.register_forward_pre_hook(
        lambda m, a, kw: captures.__setitem__(f"layer{idx}_in", (a[0] if a else kw["hidden_states"]).detach().clone()),
        with_kwargs=True)
    layer.register_forward_hook(
        lambda m, a, kw, out: captures.__setitem__(f"layer{idx}_out", (out[0] if isinstance(out, tuple) else out).detach().clone()),
        with_kwargs=True)


cap(0)   # sliding
cap(5)   # global

with torch.no_grad():
    out = text_model(input_ids=input_ids, attention_mask=attention_mask,
                     output_hidden_states=True, use_cache=False)
states = out.hidden_states
assert len(states) == 49, len(states)

save_file({"input_ids_f32": input_ids.cpu().to(torch.float32),
           "real_len_f32": torch.tensor([float(real_len)], dtype=torch.float32)},
          os.path.join(OUT, "gemma4_oracle_meta_f32.safetensors"))
save_file({f"state_{i:02d}": s.detach().cpu() for i, s in enumerate(states)},
          os.path.join(OUT, "gemma4_oracle_states.safetensors"))
for li in (0, 5):
    save_file({k.replace(f"layer{li}_", ""): v.cpu() for k, v in captures.items() if k.startswith(f"layer{li}_")},
              os.path.join(OUT, f"gemma4_oracle_layer{li}.safetensors"))

for i in (0, 1, 6, 24, 48):
    s = states[i].float()
    print(f"state_{i:02d} std {s.std().item():.6e} absmax {s.abs().max().item():.6e}")
print("DONE ->", OUT)
