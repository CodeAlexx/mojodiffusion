"""Fetch the shards of Qwen3-VL-32B-Instruct that MiniMax-H3 actually reads.

H3 conditions on the unnormalized hidden state after decoder layer 50 of 64
(`MINIMAX_H3_TEXT_ENCODER_LAYER` in the diffusers PR's packing.py), so layers
54..63 and the LM head are never evaluated. Shards 12 and 13 hold exactly those
layers and are skipped; shard 14 is kept because it carries the entire vision
tower (351 tensors) alongside the unused lm_head and final norm.

Whole-shard granularity on purpose. A byte-range fetch would save a further
~5 GiB, but it produces truncated safetensors files whose headers describe
tensors that are not on disk, which then need a repack step — a silent-corruption
risk for 5 GiB. Files are downloaded intact and verified against the Hub's own
size listing.

CAVEAT recorded at fetch time: the MiniMax-H3 repository is not published yet,
and neither the diffusers converter nor packing.py names a source repo for the
conditioner — only the class `Qwen3VLForConditionalGeneration`. Instruct is the
assumption here. If H3's own `text_encoder/` turns out to be Thinking or a
fine-tune, this download is the wrong checkpoint and the parity numbers taken
against it do not transfer.

Run:
    python3 scripts/fetch_qwen3vl_32b_for_h3.py
"""

import json
import os
import sys

REPO = "Qwen/Qwen3-VL-32B-Instruct"
LOCAL_DIR = "/home/alex/minimax_h3_ref/creator-MiniMax-H3/FL2VA/processor"

# Shards holding layers 0..53 (we need 0..49), embeddings, and the vision tower.
KEEP_SHARDS = [f"model-{i:05d}-of-00014.safetensors" for i in list(range(1, 12)) + [14]]
SKIP_SHARDS = [f"model-{i:05d}-of-00014.safetensors" for i in (12, 13)]

ALLOW = KEEP_SHARDS + [
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
]


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(REPO, files_metadata=True)
    sizes = {f.rfilename: (f.size or 0) for f in info.siblings}
    want = sum(sizes.get(name, 0) for name in ALLOW)
    skipped = sum(sizes.get(name, 0) for name in SKIP_SHARDS)
    print(f"repo     : {REPO}")
    print(f"revision : {info.sha}")
    print(f"fetching : {len(ALLOW)} files, {want / 1024**3:.2f} GiB ({want / 1e9:.2f} GB)")
    print(f"skipping : {len(SKIP_SHARDS)} shards, {skipped / 1024**3:.2f} GiB (layers 54-63, never read)")
    print(f"target   : {LOCAL_DIR}")
    sys.stdout.flush()

    free = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize
    print(f"free now : {free / 1024**3:.2f} GiB")
    if free < want * 1.05:
        raise SystemExit("refusing to start: less than 5% headroom over the download size")

    path = snapshot_download(
        repo_id=REPO,
        local_dir=LOCAL_DIR,
        allow_patterns=ALLOW,
        max_workers=4,
    )

    print("\nverifying against the Hub size listing")
    bad = []
    for name in ALLOW:
        expected = sizes.get(name, 0)
        local = os.path.join(path, name)
        if not os.path.exists(local):
            bad.append(f"{name}: MISSING")
            continue
        actual = os.path.getsize(local)
        if expected and actual != expected:
            bad.append(f"{name}: {actual} != {expected}")
    total = sum(
        os.path.getsize(os.path.join(path, n)) for n in ALLOW if os.path.exists(os.path.join(path, n))
    )
    with open(os.path.join(LOCAL_DIR, "FETCH_PROVENANCE.json"), "w") as f:
        json.dump(
            {
                "repo": REPO,
                "revision": info.sha,
                "kept_shards": KEEP_SHARDS,
                "skipped_shards": SKIP_SHARDS,
                "skipped_reason": "layers 54-63; H3 reads hidden_states[50] of 64",
                "bytes_on_disk": total,
                "hub_sizes": {n: sizes.get(n, 0) for n in ALLOW},
                "instruct_vs_thinking": "UNRESOLVED at fetch time; H3 repo unpublished",
            },
            f,
            indent=2,
        )
    if bad:
        print("SIZE MISMATCHES:")
        for line in bad:
            print("  ", line)
        raise SystemExit(1)
    print(f"OK: {total / 1024**3:.2f} GiB on disk, every file matches the Hub size listing")
    free_after = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize
    print(f"free after: {free_after / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
