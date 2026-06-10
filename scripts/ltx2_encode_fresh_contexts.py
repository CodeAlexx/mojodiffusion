#!/usr/bin/env python3
"""Encode FRESH LTX-2 prompt contexts (pos + neg) and dump the PRE-connector
features in the canned-dump format that the Mojo refhq pipeline consumes
(it runs its own Embeddings1DConnector, gated cos 0.99999 vs reference).

Captures via forward_pre_hooks on EmbeddingsProcessor.{video,audio}_connector
during PromptEncoder.__call__ (Gemma-3-12b streamed, OffloadMode.CPU).

Output keys (matching ltx2_audio_context.safetensors):
  video_context     [1,1024,4096]   pre-connector video features (pos)
  audio_context     [1,1024,2048]   pre-connector audio features (pos)
  neg_video_context / neg_audio_context  (negative prompt)
  (+ post-connector copies under post_* for diagnostics)

Run:
  /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_encode_fresh_contexts.py \
      [--prompt "..."] [--neg "..."] [--out output/ltx2_fresh_contexts.safetensors]
"""
import argparse
import os
import sys
import types

sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-pipelines/src")
sys.modules.setdefault("OpenImageIO", types.ModuleType("OpenImageIO"))

import torch
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
GEMMA_ROOT = "/home/alex/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"

PROMPT = (
    "Medium shot of a woman climbing an industrial pegboard wall in a bright "
    "workshop, warm tungsten work lights overhead. She grips a rung, looks back "
    "over her shoulder at the camera, grinning, and says \"Almost there — watch "
    "this!\" in a clear, bright voice. The camera tracks upward with her as she "
    "pulls herself higher. Tools clink against the metal wall, and an upbeat "
    "workshop ambience of whirring machines hums beneath her laughter."
)
NEG_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, silence, mute"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--neg", default=NEG_PROMPT)
    ap.add_argument("--out", default="output/ltx2_fresh_contexts.safetensors")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    from ltx_pipelines.utils.blocks import PromptEncoder
    from ltx_pipelines.utils.types import OffloadMode

    # CPU end-to-end: the reference streaming text encoder OOMs the 24 GB card
    # (holds ~22 GB resident); gemma-3-12b bf16 fits host RAM and this is a
    # one-shot oracle. ~minutes per prompt is acceptable.
    torch.set_num_threads(os.cpu_count() or 16)
    dev = torch.device("cpu")
    pe = PromptEncoder(CKPT, GEMMA_ROOT, torch.bfloat16, dev,
                       offload_mode=OffloadMode.NONE)

    pre: list[dict] = []  # one dict per connector call, in call order

    def make_hook(tag):
        def hook(module, hook_args):
            # Embeddings1DConnector forward args: (features, mask, ...) —
            # capture the first tensor arg (pre-connector features).
            feats = hook_args[0]
            pre.append({"tag": tag, "feats": feats.detach().float().cpu()})
        return hook

    # The processor is built inside PromptEncoder.__call__; hook at module
    # creation by wrapping the builder method.
    orig_build = pe._build_embeddings_processor

    def build_and_hook():
        proc = orig_build()
        proc.video_connector.register_forward_pre_hook(make_hook("video"))
        if proc.audio_connector is not None:
            proc.audio_connector.register_forward_pre_hook(make_hook("audio"))
        return proc

    pe._build_embeddings_processor = build_and_hook

    print("encoding prompts (gemma streamed, OffloadMode.CPU)...")
    ctx_p, ctx_n = pe([args.prompt, args.neg])

    vids = [e["feats"] for e in pre if e["tag"] == "video"]
    auds = [e["feats"] for e in pre if e["tag"] == "audio"]
    print("captured pre-connector calls: video", len(vids), "audio", len(auds))
    for name, t in [("video", vids[0] if vids else None), ("audio", auds[0] if auds else None)]:
        if t is not None:
            print(f"  {name}: {tuple(t.shape)}")

    out = {}
    # Batched encode: one connector call with B=2 ([pos, neg]) OR two calls.
    if len(vids) == 1 and vids[0].shape[0] == 2:
        out["video_context"] = vids[0][0:1]
        out["neg_video_context"] = vids[0][1:2]
    elif len(vids) >= 2:
        out["video_context"] = vids[0][0:1]
        out["neg_video_context"] = vids[1][0:1]
    else:
        raise RuntimeError(f"unexpected video connector capture count {len(vids)}")
    if auds:
        if len(auds) == 1 and auds[0].shape[0] == 2:
            out["audio_context"] = auds[0][0:1]
            out["neg_audio_context"] = auds[0][1:2]
        elif len(auds) >= 2:
            out["audio_context"] = auds[0][0:1]
            out["neg_audio_context"] = auds[1][0:1]
        else:
            raise RuntimeError(f"unexpected audio capture count {len(auds)}")

    # diagnostics: post-connector encodings from PromptEncoder output
    out["post_video_context"] = ctx_p.video_encoding.detach().float().cpu()
    out["post_neg_video_context"] = ctx_n.video_encoding.detach().float().cpu()
    if ctx_p.audio_encoding is not None:
        out["post_audio_context"] = ctx_p.audio_encoding.detach().float().cpu()
        out["post_neg_audio_context"] = ctx_n.audio_encoding.detach().float().cpu()

    save_file({k: v.contiguous() for k, v in out.items()}, args.out)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)} std={v.std().item():.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
