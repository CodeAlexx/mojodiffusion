#!/usr/bin/env python3
"""Regenerate the LTX2 conditioning dumps the Mojo HQ pipeline consumes.

The original dumps were produced by the EriDiffusion (Rust flame) stack on the
old box and are gone. The Mojo pipeline (serenitymojo/pipeline/ltx2_t2v_av_hq.mojo)
consumes PRE-connector Gemma feature-extractor outputs and applies the
video/audio embeddings connectors itself (loaded from CKPT_FP8):

  AUDIO_CTX_DUMP :: {"video_context": [1,1024,4096] BF16,
                     "audio_context": [1,1024,2048] BF16}
  NEG_CTX_DUMP   :: {"text_hidden":   [1,1024,4096] BF16}   (null/negative ctx)

Faithful recipe (ltx_core, /home/alex/LTX-2 @ 7809842):
  LTXVGemmaTokenizer(gemma_root, max_length=1024, padding_side=LEFT)
  Gemma3ForConditionalGeneration.model(input_ids, mask, output_hidden_states=True)
  FeatureExtractorV2 (per-token RMS over 49 hidden states -> concat 3840*49
  -> rescale -> video_aggregate_embed[->4096] / audio_aggregate_embed[->2048])
  with weights from the LTX-2.3 checkpoint keys
  text_embedding_projection.{video,audio}_aggregate_embed.{weight,bias}.

CPU-only (Gemma bf16 ~23GB in RAM; the 5080 has 16GB VRAM).
Run: /home/alex/SerenityTrainer/venv/bin/python scripts/ltx2_make_context_dumps.py
"""
import argparse
import sys

import torch

sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-core/src")
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer  # noqa: E402
from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2  # noqa: E402

GEMMA_SNAPSHOT = (
    "/home/alex/.cache/huggingface/hub/"
    "models--Lightricks--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/"
    "d62fe4f1995ade703b49a0f3c0d0f161237ef437"
)
CKPT_BF16 = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
OUT_POS = (
    "/home/alex/EriDiffusion/inference-flame/output/audio_context_dump/"
    "ltx2_audio_context.safetensors"
)
OUT_NEG = "/home/alex/EriDiffusion/inference-flame/cached_ltx2_negative.safetensors"

DEFAULT_PROMPT = (
    "A cinematic close-up of a woman with long brown hair sitting in a sunlit "
    "kitchen, smiling and speaking warmly to the camera, natural window light, "
    "shallow depth of field, gentle ambient room tone, her voice clear and calm."
)
# ltx_pipelines/utils/constants.py DEFAULT_NEGATIVE_PROMPT (truncated fine at 1024 tok)
DEFAULT_NEGATIVE = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, unnatural skin tones, deformed facial features, asymmetrical face, "
    "missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts "
    "around text, inconsistent perspective, camera shake, incorrect depth of field, "
    "background too sharp, background clutter, distracting reflections, harsh shadows, "
    "inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, "
    "unrealistic materials, uncanny valley effect, exaggerated expressions, wrong gaze "
    "direction, mismatched lip sync, silent or muted audio, distorted voice, robotic "
    "voice, echo, background noise, off-sync audio, jittery movement, awkward pauses, "
    "incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat "
    "lighting"
)


def load_feature_extractor(ckpt: str) -> FeatureExtractorV2:
    from safetensors import safe_open

    with safe_open(ckpt, framework="pt", device="cpu") as h:
        keys = set(h.keys())
        prefix = "text_embedding_projection."
        if prefix + "video_aggregate_embed.weight" not in keys:
            # some exports carry the model.diffusion_model. prefix
            alt = "model.diffusion_model.text_embedding_projection."
            if alt + "video_aggregate_embed.weight" in keys:
                prefix = alt
            else:
                cand = sorted(k for k in keys if "aggregate_embed" in k)
                raise KeyError(
                    f"aggregate_embed keys not found under expected prefixes; candidates: {cand[:8]}"
                )
        vw = h.get_tensor(prefix + "video_aggregate_embed.weight")
        vb = h.get_tensor(prefix + "video_aggregate_embed.bias")
        aw = h.get_tensor(prefix + "audio_aggregate_embed.weight")
        ab = h.get_tensor(prefix + "audio_aggregate_embed.bias")
    print(f"[fe] prefix={prefix} video {tuple(vw.shape)} {vw.dtype}  audio {tuple(aw.shape)} {aw.dtype}")

    def mk_linear(w: torch.Tensor, b: torch.Tensor) -> torch.nn.Linear:
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=True, dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_(w.to(torch.bfloat16))
            lin.bias.copy_(b.to(torch.bfloat16))
        return lin

    fe = FeatureExtractorV2(
        video_aggregate_embed=mk_linear(vw, vb),
        embedding_dim=3840,  # gemma-3-12b hidden size
        audio_aggregate_embed=mk_linear(aw, ab),
    )
    fe.eval()
    return fe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--ckpt", default=CKPT_BF16)
    ap.add_argument("--gemma", default=GEMMA_SNAPSHOT)
    ap.add_argument("--out-pos", default=OUT_POS)
    ap.add_argument("--out-neg", default=OUT_NEG)
    args = ap.parse_args()

    import os

    from safetensors.torch import save_file
    from transformers import Gemma3ForConditionalGeneration

    fe = load_feature_extractor(args.ckpt)

    tok = LTXVGemmaTokenizer(args.gemma, max_length=1024)  # padding LEFT default
    print("[gemma] loading (bf16, CPU) ...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.gemma, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    from ltx_core.text_encoders.gemma.embeddings_processor import (
        _apply_right_pad_order,
        _compute_right_pad_order,
        convert_to_additive_mask,
    )

    def features(text: str):
        pairs = tok.tokenize_with_weights(text)["gemma"]
        ids = torch.tensor([[t for t, _ in pairs]])
        mask = torch.tensor([[w for _, w in pairs]])
        n_valid = int(mask.sum())
        print(f"[encode] {n_valid}/1024 valid tokens :: {text[:60]}...")
        with torch.no_grad():
            out = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hs = out.hidden_states
            print(f"[encode] {len(hs)} hidden states, each {tuple(hs[0].shape)} {hs[0].dtype}")
            # PAD CONVENTION (reference: ltx_core EmbeddingsProcessor +
            # Embeddings1DConnector): feature-extract with the REAL mask —
            # FeatureExtractorV2 ZEROES pad rows after the per-token RMS —
            # then reorder to RIGHT-padded [valid, pad] exactly like
            # EmbeddingsProcessor.create_embeddings does before the connector.
            # The consumer (ltx2_connector_forward valid_len=...) then replaces
            # rows >= valid_len with the checkpoint's learnable registers, the
            # reference's register-replacement semantics. (The earlier all-ones
            # -mask / no-replacement convention imprints a foreground-lattice
            # artifact — 2026-07-09 forensics, two prompts, both pad contents.)
            v, a = fe(hs, mask)
            additive = convert_to_additive_mask(mask, v.dtype)
            sort_idx, _ = _compute_right_pad_order(additive)
            v = _apply_right_pad_order(v, sort_idx)
            a = _apply_right_pad_order(a, sort_idx)
        print(
            f"[feats] video {tuple(v.shape)} std={v.float().std():.4f} "
            f"audio {tuple(a.shape)} std={a.float().std():.4f}"
        )
        return (
            v.to(torch.bfloat16).contiguous(),
            a.to(torch.bfloat16).contiguous(),
            n_valid,
        )

    pv, pa, p_len = features(args.prompt)
    nv, na, n_len = features(args.negative)

    os.makedirs(os.path.dirname(args.out_pos), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_neg), exist_ok=True)
    save_file({"video_context": pv, "audio_context": pa,
               "video_len": torch.tensor([float(p_len)], dtype=torch.float32),
               "audio_len": torch.tensor([float(p_len)], dtype=torch.float32)},
              args.out_pos,
              metadata={"prompt": args.prompt[:500], "producer": "ltx2_make_context_dumps.py",
                        "pad_convention": "right-padded, FE-masked, valid_len keys"})
    save_file({"text_hidden": nv,
               "audio_hidden": na,  # neg audio context (guided/CFG reference runs)
               "text_len": torch.tensor([float(n_len)], dtype=torch.float32)},
              args.out_neg,
              metadata={"prompt": args.negative[:500], "producer": "ltx2_make_context_dumps.py",
                        "pad_convention": "right-padded, FE-masked, valid_len keys"})
    print(f"[write] {args.out_pos}")
    print(f"[write] {args.out_neg}")
    print("DONE")


if __name__ == "__main__":
    main()
