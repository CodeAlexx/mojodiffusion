#!/usr/bin/env python3
"""Extract HumanEdit parquet shards into the Krea2 edit-triplet layout.

HumanEdit orientation:
  INPUT_IMG + EDITING_INSTRUCTION -> OUTPUT_IMG

Krea2's existing edit stager consumes:
  <stem>-condlabel.png  source/condition image
  <stem>.jpg            edited target image
  <stem>.txt            edit instruction

MASK_IMG remains in the downloaded parquet shards. It is intentionally not
copied because the OminiControl edit condition is a whole image, not a mask.
"""

import argparse
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


JPEG_QUALITY = 95
MAX_CAPTION_CHARS = 5000


def _decode(cell) -> Image.Image:
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="directory containing HumanEdit parquet shards")
    parser.add_argument("out", type=Path, help="output Krea2 triplet directory")
    parser.add_argument("--limit", type=int, default=0, help="stop after N valid rows")
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="write orientation_check.png with eight source|target pairs",
    )
    args = parser.parse_args()

    shards = sorted(args.src.glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"no train-*.parquet shards in {args.src}")
    args.out.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_caption = 0
    sheet_rows = []

    for shard in shards:
        parquet = pq.ParquetFile(shard)
        columns = ["IMAGE_ID", "EDITING_INSTRUCTION", "INPUT_IMG", "OUTPUT_IMG"]
        for row_group in range(parquet.num_row_groups):
            rows = parquet.read_row_group(row_group, columns=columns).to_pylist()
            for row in rows:
                instruction = (row["EDITING_INSTRUCTION"] or "").strip()
                if not instruction or len(instruction) > MAX_CAPTION_CHARS:
                    skipped_caption += 1
                    continue

                source = _decode(row["INPUT_IMG"])
                target = _decode(row["OUTPUT_IMG"])
                stem = f"humanedit_{kept:06d}"

                source.save(args.out / f"{stem}-condlabel.png")
                target.save(args.out / f"{stem}.jpg", quality=JPEG_QUALITY)
                (args.out / f"{stem}.txt").write_text(instruction + "\n")

                if args.contact_sheet and len(sheet_rows) < 8:
                    sheet_rows.append((source.copy(), target.copy(), instruction))
                kept += 1
                if kept % 250 == 0:
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
        for index, (source, target, _) in enumerate(sheet_rows):
            sheet.paste(source.resize((cell, cell)), (0, index * cell))
            sheet.paste(target.resize((cell, cell)), (cell, index * cell))
        sheet.save(args.out / "orientation_check.png")
        print(
            "\nwrote orientation_check.png: LEFT is INPUT_IMG (condition), "
            "RIGHT is OUTPUT_IMG (target)"
        )
        for _, _, instruction in sheet_rows:
            print(f"  row: {instruction}")

    print(f"\nHumanEdit: {kept} triplets -> {args.out}")
    if skipped_caption:
        print(f"  skipped {skipped_caption} rows (empty or over-long instruction)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
