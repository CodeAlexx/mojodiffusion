#!/usr/bin/env python3
"""No-CUDA Klein Qwen3 chat-template/token contract guard."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")
OT_FLUX2_MODEL = ONETRAINER / "modules/model/Flux2Model.py"
OT_FLUX2_DATALOADER = ONETRAINER / "modules/dataLoader/Flux2BaseDataLoader.py"
LOCAL_PRECACHE = REPO / "serenitymojo/pipeline/klein9b_precache_sample_prompts.mojo"
TOKENIZER_DIR = Path(
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
DEFAULT_SAMPLES = REPO / "serenitymojo/configs/klein9b_alina_samples.json"
PAD_ID = 151643
SEQ = 512


def fail(message: str) -> None:
    raise SystemExit(f"[klein-conditioning] FAIL {message}")


def read(path: Path) -> str:
    if not path.exists():
        fail(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def require_markers(path: Path, label: str, markers: tuple[str, ...], quiet: bool) -> None:
    text = read(path)
    missing = [marker for marker in markers if marker not in text]
    if missing:
        fail(f"{label} missing markers: {missing}")
    if not quiet:
        print(f"[klein-conditioning] PASS {label}")


def local_klein_template(prompt: str) -> str:
    return (
        "<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def load_sample_prompts(path: Path) -> list[tuple[str, str]]:
    data = json.loads(read(path))
    if not isinstance(data, dict):
        fail(f"sample JSON root must be object: {path}")
    prompts = data.get("prompts")
    if not isinstance(prompts, list):
        fail(f"sample JSON has no prompts list: {path}")
    out = [("smoke", "hello"), ("empty_negative", "")]
    for index, item in enumerate(prompts):
        if not isinstance(item, dict):
            fail(f"prompt {index} is not object")
        label = str(item.get("id") or item.get("label") or f"prompt_{index}")
        prompt = item.get("prompt")
        negative = item.get("negative", "")
        if not isinstance(prompt, str):
            fail(f"prompt {label} missing string prompt")
        if not isinstance(negative, str):
            fail(f"prompt {label} negative must be string")
        out.append((label + ":pos", prompt))
        out.append((label + ":neg", negative))
    return out


def padded(ids: list[int]) -> list[int]:
    if len(ids) > SEQ:
        fail(f"tokenized prompt exceeds {SEQ}: {len(ids)}")
    return ids + [PAD_ID] * (SEQ - len(ids))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    require_markers(
        OT_FLUX2_MODEL,
        "OneTrainer Flux2Model Qwen3 chat contract",
        (
            "def qwen3_format_input",
            "add_generation_prompt=True",
            "enable_thinking=False",
            "QWEN3_HIDDEN_STATES_LAYERS = [9, 18, 27]",
        ),
        args.quiet,
    )
    require_markers(
        OT_FLUX2_DATALOADER,
        "OneTrainer Flux2BaseDataLoader Qwen3 tokenization contract",
        (
            "apply_chat_template = lambda caption: qwen3_format_input(caption)",
            "'add_generation_prompt': True",
            "'enable_thinking': False",
        ),
        args.quiet,
    )
    require_markers(
        LOCAL_PRECACHE,
        "Mojo Klein precache template contract",
        ("def _klein_template", "<|im_start|>user\\n", "<think>\\n\\n</think>\\n\\n", "PAD_ID = 151643"),
        args.quiet,
    )

    if not TOKENIZER_JSON.exists():
        fail(f"missing tokenizer.json: {TOKENIZER_JSON}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
        raw_tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    checked = 0
    for label, prompt in load_sample_prompts(args.samples):
        expected_text = local_klein_template(prompt)
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        ot_text = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if ot_text != expected_text:
            fail(f"{label}: local template differs from OneTrainer/Qwen3 chat template")

        hf_ids = hf_tok(
            ot_text,
            max_length=SEQ,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        raw_ids = padded(raw_tok.encode(expected_text).ids)
        if raw_ids != hf_ids:
            fail(f"{label}: tokenizer.json ids differ from AutoTokenizer ids")
        checked += 1
        if not args.quiet:
            print(
                f"[klein-conditioning] PASS {label}: raw_tokens={len(raw_tok.encode(expected_text).ids)} padded={len(raw_ids)}"
            )

    if not args.quiet:
        print(f"[klein-conditioning] PASS prompts_checked={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
