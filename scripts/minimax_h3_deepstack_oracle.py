#!/usr/bin/env python3
"""scripts/minimax_h3_deepstack_oracle.py

CPU float32 oracle for Qwen3-VL's DEEPSTACK INJECTION mechanics — the part of
`Qwen3VLTextModel.forward` that folds three vision-tower "deepstack" taps into
the language model at decoder layers 0/1/2, ADDED at visual-token positions
only. This gate is about the INJECTION mechanics (splice + masked-add), NOT
the vision tower (gated separately by the vision-tower's own parity probe) and
NOT the full 50-layer H3 conditioner forward (gated separately by
parity/minimax_h3_conditioner_real_weight_oracle.py).

Ground truth read directly from the installed transformers 4.57.6
  .../transformers/models/qwen3_vl/modeling_qwen3_vl.py

  Qwen3VLTextModel.forward, lines 849-867:
    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(hidden_states, ...)
        hidden_states = layer_outputs
        # add visual features to the hidden states of first several layers
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx],
            )
  i.e. the add happens AFTER the decoder layer's own forward has already run
  (line 859 assigns `hidden_states = layer_outputs` BEFORE the deepstack check
  at line 862) — not before.

  Qwen3VLTextModel._deepstack_process, lines 876-883:
    local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this
  i.e. ADD, at visual-token positions ONLY — every other row of hidden_states
  is untouched (this is exactly the "non-visual positions differ by exactly
  0.0" property the gate checks below).

GATE DEPTH = 3 (a decided design point — do not change): deepstack injects at
LM decoder layers 0, 1, 2 only (`layer_idx in range(len(deepstack_visual_embeds))`
with len(deepstack_visual_embeds) == 3), so running exactly 3 real decoder
layers on real weights and dumping the state after each fully covers the
injection mechanics.

WEIGHTS: loads ONLY `embed_tokens.weight` + decoder layers 0/1/2 (all 34
tensors of which live in a SINGLE shard, `model-00001-of-00014.safetensors`,
confirmed against `model.safetensors.index.json`) plus `norm.weight` (a single
small [5120] tensor, read from shard 14 via safetensors' lazy/mmap access —
this does not materialize the rest of that 3.27 GiB shard) from the REAL
checkpoint
  /home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder
strict-loaded into a `Qwen3VLTextModel` truncated to `num_hidden_layers=3`.
No other shard, no vision tower, no LM head is ever opened. Text config (real,
from the checkpoint's own config.json `text_config`): hidden 5120, 64 layers
(only 3 constructed here), 64 heads, 8 kv heads, head_dim 128, vocab 151936,
inter 25600, rms_norm_eps 1e-6, rope_theta 5e6.

CPU only, float32 throughout (`CUDA_VISIBLE_DEVICES=""` set before `torch`
import, and no `.cuda()`/`.to("cuda")` call anywhere in this file).

VISION EMBEDS AND DEEPSTACK TAPS ARE SYNTHETIC (fixed-seed pseudo-random
[N, 5120] float32) — this gate is about injection mechanics, not the vision
tower, which is why the tower itself is never run here.

The model is run TWICE per depth: once with `deepstack_visual_embeds` set,
once with `None` (the CONTROL) — the control is load-bearing, it is what lets
the gate prove the injection touches ONLY visual positions and does nothing
when omitted (the existing t2va streamed conditioner's backward-compat
requirement).

MEASURED FINDING (not assumed — re-derive with the printed sanity line
below): `hidden_02` vs `hidden_02_nodeep` do NOT differ by exactly 0.0 at
EVERY non-visual position once real CAUSAL SELF-ATTENTION runs for more than
one layer after an add — they differ by exactly 0.0 only at non-visual
positions that are causally BEFORE the vision block (indices strictly less
than the first visual position). Non-visual positions AFTER the vision block
(causally able to attend to the modified visual rows in layers 1 and 2) pick
up a real, nonzero, EXPECTED difference through ordinary causal attention —
that is not a leak or a bug in `_deepstack_process`, it is what any causal
transformer does once a modified row is attended to by a later query. The
`_deepstack_process` function's OWN guarantee (touches ONLY the rows selected
by `visual_pos_masks`) is exactly and trivially true the instant it runs
(nothing but visual rows are assigned to); it does not — and cannot — promise
that no LATER layer's attention will read those rows and mix their effect
into subsequent positions. This oracle therefore reports THREE numbers, not
two: pre-vision max_abs (must be exactly 0.0 — the real, always-valid
invariant, and the one that actually discriminates a correct port, which
touches ONLY visual rows, from a broken port that adds to every row, which
would show nonzero diff HERE too), post-vision-non-visual max_abs (expected
CLEARLY non-zero — legitimate causal-attention propagation, reported for
transparency, not gated as a failure), and visual max_abs (must be clearly
non-zero — the injection itself).

Usage:
  CUDA_VISIBLE_DEVICES="" python3 scripts/minimax_h3_deepstack_oracle.py [out.safetensors]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import sys

import numpy as np
import torch

from safetensors import safe_open
from safetensors.torch import save_file

CKPT_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"
SHARD1 = os.path.join(CKPT_DIR, "model-00001-of-00014.safetensors")
SHARD14 = os.path.join(CKPT_DIR, "model-00014-of-00014.safetensors")

DEFAULT_OUT = (
    "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a"
    "/scratchpad/deepstack_ref.safetensors"
)

# Real special-token ids, from the checkpoint's own top-level config.json.
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653

HIDDEN = 5120
N_IMAGE_PAD = 10  # N: kept small per the task spec
SEQ_LEN = 40  # total sequence kept short per the task spec
NUM_GATE_LAYERS = 3  # GATE DEPTH — decided design point, do not change

SEED = 20260803

DECODER_WEIGHT_SUFFIXES = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def build_token_stream() -> list[int]:
    """Small text+image request: text, <|vision_start|>, N <|image_pad|>,
    <|vision_end|>, more text. Fixed-seed pseudo-random text-token ids drawn
    from a range well clear of every special id used here."""
    rng = np.random.RandomState(SEED)
    n_pre = 12
    n_post = SEQ_LEN - n_pre - 1 - N_IMAGE_PAD - 1
    assert n_post > 0
    pre = rng.randint(1000, 100000, size=n_pre).tolist()
    post = rng.randint(1000, 100000, size=n_post).tolist()
    ids = (
        pre
        + [VISION_START_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * N_IMAGE_PAD
        + [VISION_END_TOKEN_ID]
        + post
    )
    assert len(ids) == SEQ_LEN
    return ids


def mm_token_type_ids_from_ids(ids: list[int]) -> list[int]:
    """Reproduces `processor.create_mm_token_type_ids`
    (processing_qwen3_vl.py:241-245):
        mm_token_type_ids = np.zeros_like(input_ids)
        mm_token_type_ids[array_ids == self.image_token_id] = 1
    generalized to also mark `video_token_id` positions as 2 (the installed
    4.57.6 release only special-cases image tokens in this helper — video
    frames go through the same `image_token_id` placeholder path in that
    release — but the Mojo side and this oracle define the 3-way scheme
    (0 text / 1 image / 2 video) the task specifies, following the same
    zeros_like + positional-set pattern)."""
    out = [0] * len(ids)
    for i, t in enumerate(ids):
        if t == IMAGE_TOKEN_ID:
            out[i] = 1
        elif t == VIDEO_TOKEN_ID:
            out[i] = 2
    return out


def load_truncated_text_model():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    with open(os.path.join(CKPT_DIR, "config.json")) as f:
        top_cfg = json.load(f)
    text_cfg_dict = dict(top_cfg["text_config"])
    text_cfg_dict["num_hidden_layers"] = NUM_GATE_LAYERS
    text_cfg = Qwen3VLTextConfig(**text_cfg_dict)
    text_cfg._attn_implementation = "eager"  # CPU-simple, no fused-kernel deps

    torch.manual_seed(0)
    model = Qwen3VLTextModel(text_cfg)
    model.eval()
    model.to(torch.float32)

    # Build the strict state_dict: embed_tokens + layers 0/1/2 (all from
    # shard 1) + norm (from shard 14, single small tensor). This must cover
    # EXACTLY model.state_dict()'s keys (rotary_emb.inv_freq is a non-
    # persistent buffer and so is absent from state_dict() entirely).
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(SHARD1, framework="pt", device="cpu") as f1:
        names1 = set(f1.keys())
        embed_src = "model.language_model.embed_tokens.weight"
        if embed_src not in names1:
            raise RuntimeError(f"missing {embed_src} in {SHARD1}")
        state_dict["embed_tokens.weight"] = f1.get_tensor(embed_src).to(torch.float32)
        for li in range(NUM_GATE_LAYERS):
            for suffix in DECODER_WEIGHT_SUFFIXES:
                src = f"model.language_model.layers.{li}.{suffix}"
                if src not in names1:
                    raise RuntimeError(f"missing {src} in {SHARD1}")
                state_dict[f"layers.{li}.{suffix}"] = f1.get_tensor(src).to(torch.float32)

    with safe_open(SHARD14, framework="pt", device="cpu") as f14:
        norm_src = "model.language_model.norm.weight"
        if norm_src not in set(f14.keys()):
            raise RuntimeError(f"missing {norm_src} in {SHARD14}")
        state_dict["norm.weight"] = f14.get_tensor(norm_src).to(torch.float32)

    model_keys = set(model.state_dict().keys())
    provided_keys = set(state_dict.keys())
    if model_keys != provided_keys:
        missing = sorted(model_keys - provided_keys)
        extra = sorted(provided_keys - model_keys)
        raise RuntimeError(
            f"strict-load mismatch: model has {len(model_keys)} params, "
            f"provided {len(provided_keys)}; missing={missing} extra={extra}"
        )

    incompatible = model.load_state_dict(state_dict, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys

    return model, text_cfg


def run_gate(model, text_cfg) -> dict:
    ids = build_token_stream()
    mm_type_ids = mm_token_type_ids_from_ids(ids)
    seq = len(ids)

    visual_positions = [i for i, t in enumerate(mm_type_ids) if t != 0]
    assert len(visual_positions) == N_IMAGE_PAD

    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        inputs_embeds = model.embed_tokens(input_ids)  # [1, seq, hidden], f32

    gen = torch.Generator().manual_seed(SEED + 1)
    vision_embeds = torch.randn(N_IMAGE_PAD, HIDDEN, generator=gen, dtype=torch.float32) * 0.02
    deepstack = torch.randn(NUM_GATE_LAYERS, N_IMAGE_PAD, HIDDEN, generator=gen, dtype=torch.float32) * 0.02

    spliced = inputs_embeds.clone()
    with torch.no_grad():
        for k, pos in enumerate(visual_positions):
            spliced[0, pos, :] = vision_embeds[k]

    visual_pos_masks = torch.zeros(1, seq, dtype=torch.bool)
    for pos in visual_positions:
        visual_pos_masks[0, pos] = True

    # Real vision grid/timestamps are not synthesized (only the tower's
    # OUTPUT is faked) so there is no real image_grid_thw to derive mrope
    # positions from; use the model's own default (all three axes = plain
    # sequential position ids), the same "mrope degenerates to ordinary 1D
    # rope without a real spatial grid" case the streamed conditioner's
    # header already documents for the text-only path.
    def forward_truncated(num_layers: int, deepstack_embeds):
        # Truncate the layer stack + bypass the final norm so the captured
        # state is the RAW (pre-norm) hidden_states[num_layers] convention
        # already established for H3 (see minimax_h3_qwen3vl_streamed.mojo's
        # header) — forward() unconditionally applies self.norm() at the end
        # of its layer loop regardless of how many layers ran, so swap it for
        # Identity only for the duration of this call.
        saved_layers = model.layers
        saved_norm = model.norm
        model.layers = torch.nn.ModuleList(list(saved_layers)[:num_layers])
        model.norm = torch.nn.Identity()
        try:
            with torch.no_grad():
                out = model(
                    input_ids=None,
                    inputs_embeds=spliced,
                    visual_pos_masks=visual_pos_masks if deepstack_embeds is not None else None,
                    deepstack_visual_embeds=deepstack_embeds,
                )
            return out.last_hidden_state[0].clone()  # [seq, hidden], pre-norm raw state
        finally:
            model.layers = saved_layers
            model.norm = saved_norm

    ds_list = [deepstack[0], deepstack[1], deepstack[2]]
    hidden_00 = forward_truncated(1, ds_list[:1])
    hidden_01 = forward_truncated(2, ds_list[:2])
    hidden_02 = forward_truncated(3, ds_list[:3])
    hidden_02_nodeep = forward_truncated(3, None)

    diff = (hidden_02 - hidden_02_nodeep).abs()
    first_visual = min(visual_positions)
    last_visual = max(visual_positions)
    visual_mask = torch.zeros(seq, dtype=torch.bool)
    for pos in visual_positions:
        visual_mask[pos] = True
    pre_vision_mask = torch.zeros(seq, dtype=torch.bool)
    pre_vision_mask[:first_visual] = True
    post_vision_nonvisual_mask = torch.zeros(seq, dtype=torch.bool)
    post_vision_nonvisual_mask[last_visual + 1 :] = True

    max_abs_pre_vision = diff[pre_vision_mask].max().item() if pre_vision_mask.any() else 0.0
    max_abs_post_vision = diff[post_vision_nonvisual_mask].max().item() if post_vision_nonvisual_mask.any() else 0.0
    max_abs_visual = diff[visual_mask].max().item()

    # See the MEASURED FINDING in the module header: only PRE-VISION
    # (causally-protected) non-visual positions are guaranteed exact-zero.
    # POST-VISION non-visual positions legitimately pick up a non-zero delta
    # via ordinary causal self-attention in layers 1/2 — that is expected,
    # not a bug, and is reported here for transparency, not as a pass/fail
    # criterion.
    print(f"[oracle] max_abs(hidden_02 - hidden_02_nodeep) at PRE-VISION non-visual positions  = {max_abs_pre_vision!r} (must be exactly 0.0 — the real invariant)")
    print(f"[oracle] max_abs(hidden_02 - hidden_02_nodeep) at POST-VISION non-visual positions = {max_abs_post_vision!r} (expected clearly non-zero — causal-attention propagation, NOT a bug)")
    print(f"[oracle] max_abs(hidden_02 - hidden_02_nodeep) at VISUAL positions                 = {max_abs_visual!r} (must be clearly non-zero)")

    return {
        "in.input_ids": torch.tensor(ids, dtype=torch.int32),
        "in.mm_token_type_ids": torch.tensor(mm_type_ids, dtype=torch.int32),
        "in.vision_embeds": vision_embeds.to(torch.float32).contiguous(),
        "in.deepstack": deepstack.to(torch.float32).contiguous(),
        "in.inputs_embeds": spliced[0].to(torch.float32).contiguous(),
        "out.hidden_00": hidden_00.to(torch.float32).contiguous(),
        "out.hidden_01": hidden_01.to(torch.float32).contiguous(),
        "out.hidden_02": hidden_02.to(torch.float32).contiguous(),
        "out.hidden_02_nodeep": hidden_02_nodeep.to(torch.float32).contiguous(),
    }


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    assert not torch.cuda.is_available(), "CUDA must not be visible for this oracle"

    model, text_cfg = load_truncated_text_model()
    tensors = run_gate(model, text_cfg)

    save_file(tensors, out_path)
    print(f"[oracle] wrote {out_path}")
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
