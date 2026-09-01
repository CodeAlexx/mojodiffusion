#!/usr/bin/env python3
# magicbrush_extract_triplets.py — MagicBrush parquet -> the edit-triplet layout
# that scripts/krea2_omini_stage_edit.py already reads.
#
# WHY THIS EXISTS. The krea2 OminiControl EDIT vertical was proven on
# /home/alex/datasets/qwen_edit_test: 96 triplets on disk as
#   <stem>.jpg            target (the edited result)
#   <stem>-condlabel.png  condition (the source image)
#   <stem>.txt            instruction
# MagicBrush ships as 51 parquet shards with the images as embedded bytes, so it
# cannot be staged directly. This script writes the SAME on-disk convention so
# the staging + cache-prep + training chain downstream is byte-for-byte the one
# already verified — no new code path on the data side.
#
# ORIENTATION — THE THING THAT MUST NOT BE GOT WRONG. In the qwen_edit_test set
# the condition is the DAMAGED photo and the target is the RESTORED one; the
# stager's header records that this was verified by eye, because reversing them
# trains an un-restore LoRA. MagicBrush's columns are unambiguous by name:
#   source_img = the image BEFORE the edit  -> condition -> <stem>-condlabel.png
#   target_img = the image AFTER  the edit  -> target    -> <stem>.jpg
# and the instruction ("change the table for a dog") describes source -> target.
# The --contact-sheet flag renders source|target pairs so this is checked by eye
# ONCE, on real data, rather than trusted from column names. Do that before you
# spend GPU hours.
#
# MULTI-TURN. MagicBrush is multi-turn: rows share img_id and differ by
# turn_index, and turn N's source is turn N-1's target. Each row is still a
# self-contained (source, instruction, target) triple, so every row is a usable
# training sample and all 8807 are emitted. Stems are "<img_id>_t<turn_index>",
# which keeps turns of one image adjacent under a sort and lets a later filter
# select single-turn only (--first-turn-only) without re-extracting.
#
# THE MASK IS NOT USED. MagicBrush ships mask_img. The OminiControl condition
# branch has no mask input — the condition is a whole VAE-encoded image — so the
# mask is deliberately dropped rather than silently half-wired. If masked editing
# is wanted later it is a model-side change, not a staging one.
#
# SPLITS. --split train (8807) is the training corpus. --split dev (528) is a
# GENUINE held-out set from the dataset authors — extract it to its own directory
# and it becomes the honest generalization eval the 96-pair dev set could not
# provide (there, every render was a pair the LoRA had seen ~32 times).
#
# Run (needs pyarrow + PIL; the ai-toolkit venv has no pyarrow, the
# simpletuner-bench one does — verified importable 2026-07-30):
#   /home/alex/simpletuner-bench/venv/bin/python \
#     scripts/magicbrush_extract_triplets.py \
#     /home/alex/datasets/MagicBrush/data /home/alex/datasets/magicbrush_edit \
#     --split train --contact-sheet

import argparse
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

# The stager re-reads these from disk and does its own crop+resize to the bucket,
# so quality here only has to be good enough not to lose detail before that. 95
# is visually lossless on photographic content at these sizes.
JPEG_QUALITY = 95

# krea2_omini_stage_edit.py skips any sample whose caption exceeds this (the
# Qwen3-VL encoder fails loud past 2048 SDPA tokens MID-RUN). MagicBrush
# instructions are one short sentence, so this should never fire — it is here to
# turn a silent downstream skip into a loud count at extraction time.
MAX_CAPTION_CHARS = 5000


def _decode(cell) -> Image.Image:
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


def _stem(img_id: str, turn_index: int) -> str:
    return f"{img_id}_t{turn_index}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="MagicBrush data/ directory of parquet shards")
    ap.add_argument("out", type=Path, help="output triplet directory")
    ap.add_argument("--split", default="train", choices=["train", "dev"])
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (smoke)")
    ap.add_argument("--first-turn-only", action="store_true",
                    help="keep only turn_index == 1 (drops the multi-turn chain)")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="also write orientation_check.png: 8 source|target pairs")
    args = ap.parse_args()

    shards = sorted(args.src.glob(f"{args.split}-*.parquet"))
    if not shards:
        raise SystemExit(f"no {args.split}-*.parquet shards in {args.src}")
    args.out.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_caption = 0
    skipped_turn = 0
    sheet_rows: list[tuple[Image.Image, Image.Image, str]] = []

    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                if args.first_turn_only and int(row["turn_index"]) != 1:
                    skipped_turn += 1
                    continue
                instruction = (row["instruction"] or "").strip()
                if not instruction or len(instruction) > MAX_CAPTION_CHARS:
                    skipped_caption += 1
                    continue

                stem = _stem(row["img_id"], int(row["turn_index"]))
                source = _decode(row["source_img"])   # BEFORE the edit -> condition
                target = _decode(row["target_img"])   # AFTER  the edit -> target

                target.save(args.out / f"{stem}.jpg", quality=JPEG_QUALITY)
                source.save(args.out / f"{stem}-condlabel.png")
                (args.out / f"{stem}.txt").write_text(instruction + "\n")

                if args.contact_sheet and len(sheet_rows) < 8:
                    sheet_rows.append((source.copy(), target.copy(), instruction))
                kept += 1
                if kept % 500 == 0:
                    print(f"  {kept} triplets ...", flush=True)
                if args.limit and kept >= args.limit:
                    break
            if args.limit and kept >= args.limit:
                break
        if args.limit and kept >= args.limit:
            break

    if sheet_rows:
        cell = 256
        sheet = Image.new("RGB", (2 * cell, cell * len(sheet_rows)), (16, 16, 16))
        for i, (s, t, _) in enumerate(sheet_rows):
            sheet.paste(s.resize((cell, cell)), (0, i * cell))
            sheet.paste(t.resize((cell, cell)), (cell, i * cell))
        sheet.save(args.out / "orientation_check.png")
        print("\nwrote orientation_check.png — LEFT column is the condition "
              "(source, before), RIGHT is the target (after). Look at it before "
              "training; a reversed dataset trains the inverse edit.")
        for _, _, ins in sheet_rows:
            print(f"  row: {ins}")

    print(f"\n{args.split}: {kept} triplets -> {args.out}")
    if skipped_turn:
        print(f"  skipped {skipped_turn} rows (--first-turn-only)")
    if skipped_caption:
        print(f"  skipped {skipped_caption} rows (empty or over-long instruction)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
