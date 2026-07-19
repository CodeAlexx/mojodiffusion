#!/usr/bin/env python3
"""Produce the SINGLE-file PRE-connector contexts safetensors run_refhq consumes,
for an arbitrary prompt/negative — and optionally VERIFY the dump path against a
reference run's POST-connector contexts.safetensors (numeric-parity-testing:
dump + reference EmbeddingsProcessor connectors vs the native PromptEncoder).

Layout written (run_refhq contract, ltx2_t2v_av_hq.mojo):
  video_context      [1,1024,4096] BF16 (pre-connector, FE-masked, right-padded)
  audio_context      [1,1024,2048] BF16
  neg_video_context  [1,1024,4096] BF16
  neg_audio_context  [1,1024,2048] BF16
  video_len / neg_video_len  [1] F32 (valid token counts)

Reuses the faithful encode recipe of scripts/ltx2_make_context_dumps.py
(LTXVGemmaTokenizer LEFT-pad -> Gemma3.model hidden states -> FeatureExtractorV2
with the checkpoint aggregate_embed -> right-pad reorder), with the gemma
snapshot DEFAULTED to the complete Lightricks Gemma snapshot installed on this
box (the same snapshot used by scripts/ltx2_make_context_dumps.py).

Run (CPU-only, ~26GB RSS — do NOT run concurrently with a streamed 22B job):
  /home/alex/.local/share/LTXDesktop/python/bin/python3 \
      scripts/ltx2_refhq_contexts.py \
      --prompt "<same as oracle>" --neg-official \
      --out output/ltx2_mojo_10s/refhq_contexts.safetensors \
      --verify-against output/ltx2_oracle_10s_1920/contexts.safetensors
"""
import argparse
import os
import subprocess
import sys

import torch

CREATOR_ROOT = os.environ.get("LTX2_CREATOR_ROOT", "/home/alex/LTX-2")
CREATOR_REVISION = "780984275fd47128b02bef9b5c085404276866ee"
sys.path.insert(0, f"{CREATOR_ROOT}/packages/ltx-core/src")
sys.path.insert(0, f"{CREATOR_ROOT}/packages/ltx-pipelines/src")

# ltx_pipelines.utils.media_io imports OpenImageIO (not installed); stub it.
import types

sys.modules.setdefault("OpenImageIO", types.ModuleType("OpenImageIO"))

GEMMA_SNAPSHOT = (
    "/home/alex/.cache/huggingface/hub/"
    "models--Lightricks--gemma-3-12b-it-qat-q4_0-unquantized/"
    "snapshots/d62fe4f1995ade703b49a0f3c0d0f161237ef437"
)
CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"


def assert_creator_revision() -> None:
    head = subprocess.run(
        ["git", "-C", CREATOR_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", CREATOR_ROOT, "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != CREATOR_REVISION or status:
        raise SystemExit(
            f"Creator oracle must be clean at {CREATOR_REVISION}; "
            f"found head={head}, dirty={bool(status)} in {CREATOR_ROOT}"
        )


def main() -> None:
    assert_creator_revision()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--neg", default="")
    ap.add_argument("--neg-official", action="store_true",
                    help="use ltx_pipelines DEFAULT_NEGATIVE_PROMPT")
    ap.add_argument("--gemma", default=GEMMA_SNAPSHOT)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-against", default="",
                    help="a reference run's POST-connector contexts.safetensors; "
                         "applies the reference EmbeddingsProcessor to this dump "
                         "and reports per-tensor cos vs the native encode")
    args = ap.parse_args()

    if args.neg_official:
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
        args.neg = DEFAULT_NEGATIVE_PROMPT
    if not args.neg:
        raise SystemExit("--neg or --neg-official required")

    from safetensors.torch import save_file
    from transformers import Gemma3ForConditionalGeneration
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from ltx_core.text_encoders.gemma.embeddings_processor import (
        _apply_right_pad_order,
        _compute_right_pad_order,
        convert_to_additive_mask,
    )

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ltx2_make_context_dumps import load_feature_extractor

    fe = load_feature_extractor(args.ckpt)
    tok = LTXVGemmaTokenizer(args.gemma, max_length=1024)
    print("[gemma] loading (bf16, CPU) ...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.gemma, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    def features(text: str):
        pairs = tok.tokenize_with_weights(text)["gemma"]
        ids = torch.tensor([[t for t, _ in pairs]])
        mask = torch.tensor([[w for _, w in pairs]])
        n_valid = int(mask.sum())
        print(f"[encode] {n_valid}/1024 valid tokens :: {text[:60]}...")
        with torch.no_grad():
            out = model.model(input_ids=ids, attention_mask=mask,
                              output_hidden_states=True)
            v, a = fe(out.hidden_states, mask)
            additive = convert_to_additive_mask(mask, v.dtype)
            sort_idx, _ = _compute_right_pad_order(additive)
            v = _apply_right_pad_order(v, sort_idx)
            a = _apply_right_pad_order(a, sort_idx)
        return (v.to(torch.bfloat16).contiguous(),
                a.to(torch.bfloat16).contiguous(), n_valid)

    pv, pa, p_len = features(args.prompt)
    nv, na, n_len = features(args.neg)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(
        {"video_context": pv, "audio_context": pa,
         "neg_video_context": nv, "neg_audio_context": na,
         "video_len": torch.tensor([float(p_len)], dtype=torch.float32),
         "neg_video_len": torch.tensor([float(n_len)], dtype=torch.float32)},
        args.out,
        metadata={"prompt": args.prompt[:500], "neg": args.neg[:200],
                  "producer": "ltx2_refhq_contexts.py",
                  "creator_revision": CREATOR_REVISION,
                  "pad_convention": "right-padded, FE-masked, valid_len keys"},
    )
    print("[write]", args.out)

    if args.verify_against:
        del model  # free ~24GB before the connector build
        print("[verify] applying reference EmbeddingsProcessor (CPU) ...")
        from ltx_core.loader.single_gpu_model_builder import (
            SingleGPUModelBuilder as Builder,
        )
        from ltx_core.loader.registry import DummyRegistry
        from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
            EMBEDDINGS_PROCESSOR_KEY_OPS,
            EmbeddingsProcessorConfigurator,
        )
        from safetensors.torch import load_file

        ep = Builder(
            model_path=args.ckpt,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=DummyRegistry(),
        ).build(device=torch.device("cpu"), dtype=torch.bfloat16).eval()

        ref = load_file(args.verify_against)

        def one(vpre, apre, n_valid, kv, ka):
            mask = torch.zeros(1, vpre.shape[1], dtype=torch.int64)
            mask[:, :n_valid] = 1  # right-padded: valid rows first
            add = convert_to_additive_mask(mask, vpre.dtype)
            with torch.no_grad():
                v_post, a_post, _ = ep.create_embeddings(vpre, apre, add)
            for name, mine, theirs in ((kv, v_post, ref[kv]),
                                       (ka, a_post, ref[ka])):
                m = mine.float().flatten()
                t = theirs.float().flatten()
                cos = torch.dot(m, t) / (m.norm() * t.norm() + 1e-12)
                rel = (m - t).norm() / (t.norm() + 1e-12)
                print(f"[verify] {name}: cos={cos.item():.7f} relL2={rel.item():.5f}")

        one(pv, pa, p_len, "video_context", "audio_context")
        one(nv, na, n_len, "neg_video_context", "neg_audio_context")
    print("DONE")


if __name__ == "__main__":
    main()
