# DEV-ONLY parity oracle for the Ideogram-4 TEXT ENCODER (Qwen3-VL -> the 13-tap
# llm_features the DiT consumes), oracled against **ai-toolkit** (NOT ideogram4-ref).
#
# Mirrors EXACTLY ai-toolkit's production text-feature path:
#   ai-toolkit/extensions_built_in/diffusion_models/ideogram4/
#     ideogram4.py::_load_text_encoder() loads the PUBLIC bf16
#       "Qwen/Qwen3-VL-8B-Instruct" (NOT the ideogram fp8 copy -- the comment there
#       says bf16 is "higher precision than dequantizing the fp8 weights").
#     src/pipeline.py::get_qwen3_vl_features() runs the language_model layer-by-
#       layer, captures hidden_states AFTER each tapped decoder_layer for
#       QWEN3_VL_ACTIVATION_LAYERS = (0,3,6,9,12,15,18,21,24,27,30,33,35), then
#       stack(dim=0)->permute(1,2,3,0)->reshape => out[..,f*13+t] = tap_t[..,f],
#       masked by attention at non-text positions.  NO final model.norm applied.
#
# ISOLATION: this gate is the ENCODER MODEL given ids. We feed FIXED deterministic
# token ids (dumped here, fed byte-identically to the mojo encoder) -- tokenization
# / caption-JSON-wrapping is a SEPARATE concern, NOT re-litigated here.
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/text_encoder/parity/ideogram4_aitoolkit_encoder_oracle.py
import os
import struct
import warnings
import torch
from safetensors.torch import save_file
from transformers import Qwen3VLForConditionalGeneration
from transformers.masking_utils import create_causal_mask

# NOTE on the loader (verified 2026-06-25, transformers 4.57.6):
# ai-toolkit's _load_text_encoder uses AutoModel.from_pretrained(QWEN3_VL_PATH).
# In transformers 4.57.6 AutoModel maps qwen3_vl -> bare Qwen3VLModel, which expects
# keys "language_model.*". But the public checkpoint is saved as the architecture it
# declares -- Qwen3VLForConditionalGeneration -- whose keys are "model.language_model.*".
# So AutoModel.from_pretrained finds NO matching language_model key and RANDOM-inits the
# whole text tower ("Some weights ... were not initialized ... newly initialized"). That
# random model is a useless oracle. We instead load the DECLARED architecture
# Qwen3VLForConditionalGeneration (real weights, no random-init) and access its
# .model.language_model -- which is the SAME nn.Module ai-toolkit calls
# text_encoder.language_model on, just reached through the correctly-loaded wrapper.
# (ai-toolkit's own venv pins transformers 5.5.3 where AutoModel may strip the prefix;
# either way the production INTENT is the real bf16 Qwen3-VL text tower, which this
# loader delivers deterministically.) We PROVE real weights below by byte-comparing a
# loaded language_model tensor against the on-disk checkpoint shard (the "newly
# initialized" notice is emitted via `logging`, not `warnings`, so it can't be caught).

# Same constant ai-toolkit uses (src/transformer.py:40).
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
# ai-toolkit _load_text_encoder: te_path defaults to QWEN3_VL_PATH (the public bf16).
QWEN3_VL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

OUT = "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity"
DT = torch.bfloat16  # production text-encoder dtype (model dtype == bf16)
dev = torch.device("cuda")

# Fixed deterministic token ids. Length 32 -- the mojo encoder's SDPA dispatch
# (qwen3_encoder._sdpa_dispatch) only materializes power-of-two seq cases
# {8,16,32,64,...}; 32 is the smallest that comfortably exceeds the 13-tap structure
# while staying a supported kernel shape (no kernel change needed). Values are
# arbitrary-but-fixed real-ish Qwen3 vocab ids (vocab ~151k); they isolate the
# ENCODER from tokenization. All positions attended (no padding semantics here).
FIXED_IDS = [
    9707, 11, 419, 374, 264, 1273, 315, 279, 1467, 12,
    33800, 6730, 369, 25433, 12, 19, 8311, 13, 1416, 432,
    4278, 11, 6915, 13, 1207, 1131, 1467, 12522, 13, 8704,
    279, 1156,
]


@torch.no_grad()
def get_qwen3_vl_features(language_model, token_ids, attention_mask, pos_2d, want_taps):
    """Verbatim re-implementation of ai-toolkit src/pipeline.py::get_qwen3_vl_features
    (commit-pinned logic) that ALSO returns the per-tap raw hidden states so the gate
    can compare each tap layer individually, not just the concat. `language_model` is
    the SAME nn.Module ai-toolkit reaches as text_encoder.language_model."""

    inputs_embeds = language_model.embed_tokens(token_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    # transformers 4.57.6 create_causal_mask signature: kwarg is `input_embeds` (no
    # trailing s) and it requires `cache_position`. ai-toolkit runs on transformers
    # 5.5.3 (kwarg `inputs_embeds`, no cache_position) -- same lower-triangular result
    # for an all-ones attention_mask with sequential positions.
    seq_len_ = inputs_embeds.shape[1]
    cache_position = torch.arange(seq_len_, device=inputs_embeds.device)
    causal_mask = create_causal_mask(
        config=language_model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states

    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    batch_size, seq_len = token_ids.shape
    stacked = torch.stack(selected, dim=0)            # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))    # (B, L, H, num_taps)
    stacked = stacked.reshape(batch_size, seq_len, -1)  # (B, L, H*num_taps)

    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    stacked = stacked * text_mask

    taps = {i: captured[i] for i in want_taps}
    return stacked, taps


def _disk_tensor(name: str) -> torch.Tensor:
    """Read one tensor straight from the on-disk checkpoint shard (ground truth)."""
    import glob, json
    from safetensors import safe_open

    snap = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*"
        )
    )[0]
    idx = json.load(open(f"{snap}/model.safetensors.index.json"))
    shard = idx["weight_map"][name]
    with safe_open(f"{snap}/{shard}", framework="pt") as f:
        return f.get_tensor(name)


def main():
    print(f"[E] loading {QWEN3_VL_PATH} (bf16, ~16GB) via Qwen3VLForConditionalGeneration...")
    # NOTE: transformers emits the "newly initialized" notice via `logging`, NOT the
    # `warnings` module, so warnings.catch_warnings cannot detect random-init reliably.
    # Instead we PROVE real weights loaded by byte-comparing a loaded language_model
    # tensor against the on-disk checkpoint shard below.
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN3_VL_PATH, dtype=DT, low_cpu_mem_usage=True
    )
    full.eval()
    full.requires_grad_(False)
    full.to(dev)
    # The SAME nn.Module ai-toolkit calls text_encoder.language_model on.
    language_model = full.model.language_model

    # GROUND-TRUTH real-weight proof: a loaded layer-0 q_proj must byte-match the disk
    # checkpoint tensor "model.language_model.layers.0.self_attn.q_proj.weight". A random
    # -init tower (the AutoModel key-mismatch failure mode) would mismatch here -> abort.
    loaded_q = language_model.layers[0].self_attn.q_proj.weight.detach().to("cpu", torch.bfloat16)
    disk_q = _disk_tensor("model.language_model.layers.0.self_attn.q_proj.weight").to(torch.bfloat16)
    max_abs = float((loaded_q.float() - disk_q.float()).abs().max())
    assert max_abs == 0.0, (
        f"loaded q_proj != disk checkpoint (max|diff|={max_abs}) -- model was "
        "RANDOM-initialized (bad loader/key mismatch). Oracle would be garbage."
    )
    emb_std = float(language_model.embed_tokens.weight.float().std())
    print(f"[E] REAL weights verified (layer0 q_proj byte-matches disk). embed_std={emb_std:.5f}")

    L = len(FIXED_IDS)
    token_ids = torch.tensor([FIXED_IDS], dtype=torch.long, device=dev)  # (1, L)
    # ai-toolkit get_prompt_embeds: attention_mask = ones; pos_2d = cumsum(mask)-1.
    attention_mask = torch.ones_like(token_ids)
    pos_2d = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)

    # We compare a representative subset of taps individually (first/early/mid/late/
    # last) PLUS the full concatenated llm_features. Subset keeps the fixture small;
    # the final concat already folds in ALL 13 taps so a wrong tap anywhere shows up.
    want_taps = [0, 3, 18, 33, 35]

    llm_features, taps = get_qwen3_vl_features(
        language_model, token_ids, attention_mask, pos_2d, want_taps
    )

    print(f"[E] llm_features {tuple(llm_features.shape)} dtype={llm_features.dtype}")
    assert llm_features.shape[-1] == 4096 * len(QWEN3_VL_ACTIVATION_LAYERS), (
        f"llm_features dim {llm_features.shape[-1]} != {4096 * 13}"
    )

    fx = {
        # ids dumped as int32 so the mojo gate feeds byte-identical ids.
        "input_ids": token_ids.to(torch.int32).cpu().reshape(-1),     # [L]
        "llm_features": llm_features.float().cpu(),                    # [1,L,53248]
    }
    for i in want_taps:
        fx[f"tap_{i}"] = taps[i].float().cpu()                        # [1,L,4096]

    save_file(fx, f"{OUT}/ideogram4_aitoolkit_encoder.safetensors")

    # Also dump ids as a raw little-endian int32 .bin so the mojo gate can read them
    # byte-identically via FFI (avoids any safetensors int-dtype ambiguity in mojo).
    with open(f"{OUT}/ideogram4_aitoolkit_encoder_ids.bin", "wb") as f:
        f.write(struct.pack(f"<{L}i", *FIXED_IDS))

    # Per-tap stats for the report.
    print("[E] tap stats (raw post-layer hidden, pre-final-norm):")
    for i in want_taps:
        t = taps[i].float()
        print(f"    tap_{i:2d}: shape={tuple(t.shape)} std={t.std():.5f} mean={t.mean():+.5f}")
    lf = llm_features.float()
    print(f"[E] llm_features std={lf.std():.5f} mean={lf.mean():+.5f}")
    print(f"[E] saved fixture -> {OUT}/ideogram4_aitoolkit_encoder.safetensors")
    print(f"[E] L={L}  taps_compared={want_taps}  layer_set={QWEN3_VL_ACTIVATION_LAYERS}")


if __name__ == "__main__":
    main()
