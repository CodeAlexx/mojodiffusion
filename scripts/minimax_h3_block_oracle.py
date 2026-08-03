"""MiniMax-H3 transformer forward oracle, on a tiny random-weight model.

Reference: diffusers PR huggingface/diffusers#14355, cloned at
/home/alex/minimax_h3_ref/diffusers-src, head e1b518df.
  models/transformers/transformer_minimax_h3.py  MiniMaxH3Transformer3DModel

This gates the WHOLE DiT math without a checkpoint: RMSNorm, the wide
QKV projections, per-head q/k norms, partial rotary, full self-attention, the
SwiGLU feed-forward, the three-modality AdaLN modulation, the token refiner,
the final modulated norm, and both output heads.

The tiny config and the packed layout are the reference's OWN test fixture
(tests/models/transformers/test_models_transformer_minimax_h3.py), which exists
precisely to build fixtures. Two properties of it matter and are preserved:
`num_attention_heads * attention_head_dim` (32) differs from `hidden_size` (24),
as in the released checkpoint, and `2 * 3 * rope_freq_dim` (12) is smaller than
`attention_head_dim` (16) so the PARTIAL rotary path is exercised rather than
aliased away.

Batch is 1 here, not the fixture's 2: H3 runs batch 1 and the batch axis is pure
replication.

Weights are dumped in the DIFFUSERS layout — `to_q`/`to_k`/`to_v` split and
`ff.net.0.proj` as `[value; gate]`. Turning the original checkpoint's fused
per-head-interleaved QKV and `[gate; value]` fc1 into that layout is the
LOADER's job, already specified by `transformer_key_plan.txt`; this unit is
about the math.

Run:
    python3 scripts/minimax_h3_block_oracle.py
Writes: output/minimax_h3_block/block_ref.safetensors
"""

import json
import os
import sys

DIFFUSERS_SRC = "/home/alex/minimax_h3_ref/diffusers-src/src"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_block"

sys.path.insert(0, DIFFUSERS_SRC)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffusers import MiniMaxH3Transformer3DModel  # noqa: E402

NUM_TEXT_TOKENS = 4
NUM_AUDIO_TOKENS = 6
NUM_VIDEO_TOKENS = 8

INIT = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 24,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 32,
    "in_channels": 4,
    "audio_in_channels": 6,
    "patch_size": (1, 2, 2),
    "text_dim": 8,
    "freq_dim": 8,
    "time_embed_hidden_dim": 24,
    "time_embed_dim": 16,
    "rope_freq_dim": 2,
}


def packed_layout():
    """The reference fixture's layout: text rows, audio rows, video rows, two
    distinct timesteps so the (timestep, modality) AdaLN table is addressed on
    more than one row, and no padding tail so attention needs no mask."""
    sequence_length = NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + NUM_VIDEO_TOKENS
    text_indices = torch.arange(NUM_TEXT_TOKENS)
    audio_indices = torch.arange(NUM_TEXT_TOKENS, NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS)
    video_indices = torch.arange(NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, sequence_length)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0

    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[audio_indices] = 1

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32)
    position_ids[video_indices, 1] = torch.arange(NUM_VIDEO_TOKENS, dtype=torch.float32) % 4
    position_ids[video_indices, 2] = torch.arange(NUM_VIDEO_TOKENS, dtype=torch.float32) % 2

    return {
        "timestep": torch.tensor([0.7, 0.3]),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }


def main() -> None:
    torch.manual_seed(0)
    model = MiniMaxH3Transformer3DModel(**INIT).eval().to(torch.float32)

    # Default init leaves several tensors zero (register-style params and the
    # adaLN biases), which would hide sign and ordering errors. Re-randomize
    # every parameter so nothing in the graph is accidentally an identity.
    generator = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.15
            )

    patch = INIT["patch_size"]
    video_patch_dim = INIT["in_channels"] * patch[0] * patch[1] * patch[2]
    inputs = packed_layout()

    gen = torch.Generator().manual_seed(7)
    hidden_states = torch.randn(1, NUM_VIDEO_TOKENS, video_patch_dim, generator=gen)
    audio_hidden_states = torch.randn(
        1, NUM_AUDIO_TOKENS, INIT["audio_in_channels"], generator=gen
    )
    encoder_hidden_states = torch.randn(1, NUM_TEXT_TOKENS, INIT["text_dim"], generator=gen)

    with torch.no_grad():
        out = model(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **inputs,
        )

    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        tensors[f"w.{name}"] = parameter.detach().clone()

    tensors["in.hidden_states"] = hidden_states
    tensors["in.audio_hidden_states"] = audio_hidden_states
    tensors["in.encoder_hidden_states"] = encoder_hidden_states
    tensors["in.timestep"] = inputs["timestep"]
    tensors["in.timestep_indices"] = inputs["timestep_indices"].to(torch.int64)
    tensors["in.token_tags"] = inputs["token_tags"].to(torch.int64)
    tensors["in.position_ids"] = inputs["position_ids"]
    tensors["in.video_indices"] = inputs["video_indices"].to(torch.int64)
    tensors["in.audio_indices"] = inputs["audio_indices"].to(torch.int64)
    tensors["in.text_indices"] = inputs["text_indices"].to(torch.int64)

    tensors["out.sample"] = out.sample
    tensors["out.audio_sample"] = out.audio_sample

    # ---- ORIGINAL-checkpoint layout, for the LOADER gate ----------------
    # The converter turns the original layout into the diffusers one. Here we
    # run that transform BACKWARDS to synthesize what the real checkpoint
    # holds, so the port's loader can be gated end to end — original bytes in,
    # matching velocities out — before any checkpoint exists.
    heads = INIT["num_attention_heads"]
    head_dim = INIT["attention_head_dim"]
    state = dict(model.named_parameters())

    def interleave_qkv(q, k, v):
        """Inverse of `reorder_interleaved_qkv`: [q_all; k_all; v_all] back to
        per-head [head0: q k v, head1: q k v, ...]."""
        rest = q.shape[1:]
        q = q.reshape(heads, head_dim, *rest)
        k = k.reshape(heads, head_dim, *rest)
        v = v.reshape(heads, head_dim, *rest)
        return torch.cat([q, k, v], dim=1).reshape(heads * 3 * head_dim, *rest)

    for source_prefix, target_prefix, count in (
        ("transformer_blocks", "blocks", INIT["num_layers"]),
        ("token_refiner.refiner_blocks", "token_refiner.blocks", INIT["num_refiner_layers"]),
    ):
        for i in range(count):
            src = f"{source_prefix}.{i}"
            dst = f"{target_prefix}.{i}"
            tensors[f"orig.{dst}.attn.qkv_proj.weight"] = interleave_qkv(
                state[f"{src}.attn.to_q.weight"],
                state[f"{src}.attn.to_k.weight"],
                state[f"{src}.attn.to_v.weight"],
            ).detach().clone()
            # diffusers holds [value; gate]; the checkpoint holds [gate; value].
            value, gate = state[f"{src}.ff.net.0.proj.weight"].chunk(2, dim=0)
            tensors[f"orig.{dst}.mlp.fc1.weight"] = torch.cat(
                [gate, value], dim=0
            ).detach().clone()
            tensors[f"orig.{dst}.mlp.fc2.weight"] = state[f"{src}.ff.net.2.weight"].detach().clone()
            tensors[f"orig.{dst}.attn.out_proj.weight"] = state[f"{src}.attn.to_out.0.weight"].detach().clone()
            tensors[f"orig.{dst}.attn.q_norm.weight"] = state[f"{src}.attn.norm_q.weight"].detach().clone()
            tensors[f"orig.{dst}.attn.k_norm.weight"] = state[f"{src}.attn.norm_k.weight"].detach().clone()
            tensors[f"orig.{dst}.norm1.weight"] = state[f"{src}.norm1.weight"].detach().clone()
            tensors[f"orig.{dst}.norm2.weight"] = state[f"{src}.norm2.weight"].detach().clone()
            if source_prefix == "transformer_blocks":
                tensors[f"orig.{dst}.adaln_proj.linear.weight"] = state[f"{src}.adaln_proj.linear.weight"].detach().clone()
                tensors[f"orig.{dst}.adaln_proj.linear.bias"] = state[f"{src}.adaln_proj.linear.bias"].detach().clone()

    for original, diffusers_name in (
        ("video_patch_proj", "proj_in"),
        ("audio_patch_proj", "audio_proj_in"),
        ("condition_proj", "context_embedder"),
        ("final_layer.video_out", "proj_out"),
        ("final_layer.audio_out", "audio_proj_out"),
    ):
        tensors[f"orig.{original}.weight"] = state[f"{diffusers_name}.weight"].detach().clone()
        tensors[f"orig.{original}.bias"] = state[f"{diffusers_name}.bias"].detach().clone()
    tensors["orig.time_embedder.proj_in.weight"] = state["time_embedder.linear_1.weight"].detach().clone()
    tensors["orig.time_embedder.proj_in.bias"] = state["time_embedder.linear_1.bias"].detach().clone()
    tensors["orig.time_embedder.proj_out.weight"] = state["time_embedder.linear_2.weight"].detach().clone()
    tensors["orig.time_embedder.proj_out.bias"] = state["time_embedder.linear_2.bias"].detach().clone()
    tensors["orig.token_refiner.final_norm.weight"] = state["token_refiner.final_norm.weight"].detach().clone()
    tensors["orig.final_layer.norm.weight"] = state["norm_out.norm.weight"].detach().clone()
    tensors["orig.final_layer.adaln_proj.linear.weight"] = state["norm_out.linear.weight"].detach().clone()
    tensors["orig.final_layer.adaln_proj.linear.bias"] = state["norm_out.linear.bias"].detach().clone()

    meta = {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in INIT.items()},
        "num_text_tokens": NUM_TEXT_TOKENS,
        "num_audio_tokens": NUM_AUDIO_TOKENS,
        "num_video_tokens": NUM_VIDEO_TOKENS,
        "video_patch_dim": video_patch_dim,
        "sequence_length": NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + NUM_VIDEO_TOKENS,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "block_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "block_ref.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(tensors)} tensors -> {path}")
    print(f"parameters: {meta['num_parameters']}")
    print(f"sample      {list(out.sample.shape)}  mean {out.sample.mean():.6f}")
    print(f"audio_sample{list(out.audio_sample.shape)}  mean {out.audio_sample.mean():.6f}")
    print("\nweight names:")
    for name in sorted(n for n in tensors if n.startswith("w.")):
        print(f"  {name:<56} {list(tensors[name].shape)}")


if __name__ == "__main__":
    main()
