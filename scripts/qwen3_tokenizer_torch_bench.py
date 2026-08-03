"""What the Python/Rust tokenizer costs, for comparison with the Mojo one.

Same tokenizer.json, same prompts, same iteration counts as
serenitymojo/tokenizer/parity/qwen3_tokenizer_bench.mojo, so the numbers are
directly comparable.

Three costs are separated because they are paid differently:
  * `import transformers` — paid once per process, and it is not small
  * `from_pretrained`     — the tokenizer construction itself (Rust backend)
  * `__call__`            — the per-request encode

The Rust `tokenizers` backend is fast and well optimized; this is a real
baseline, not a strawman.

Run:
    python3 scripts/qwen3_tokenizer_torch_bench.py
"""

import time

TOKENIZER_DIR = "/home/alex/minimax_h3_ref/creator-MiniMax-H3/FL2VA/processor"

PROMPTS = [
    ("short", "a cat"),
    (
        "typical",
        "A red fox trotting through a snowy pine forest, snow crunching underfoot, "
        "golden hour light through the branches, shallow depth of field, 35mm",
    ),
    (
        "long",
        "A sweeping aerial shot over a rain-slicked neon city at night, camera "
        "descending past glass towers into a crowded street market where vendors "
        "sell steaming food under paper lanterns, reflections rippling in puddles, "
        "a lone figure in a red coat walking against the crowd, cinematic color "
        "grade, anamorphic lens flares, volumetric fog, 24fps, shallow depth of "
        "field, the camera slowly pushing in as she turns to look directly at us, "
        "her expression unreadable, the sound of rain and distant traffic",
    ),
]

ITERATIONS = 50


def main() -> None:
    t0 = time.perf_counter()
    from transformers import AutoTokenizer  # noqa: F401

    t1 = time.perf_counter()
    import_seconds = t1 - t0

    print("Qwen3 tokenizer benchmark — transformers / Rust `tokenizers`")
    print(f"  tokenizer dir: {TOKENIZER_DIR}")
    print()
    print("[import]")
    print(f"  import transformers  {import_seconds:.6f} s")

    print()
    print("[construct]")
    c0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    c1 = time.perf_counter()
    construct_seconds = c1 - c0
    print(f"  from_pretrained      {construct_seconds:.6f} s")
    print(f"  fast backend         {tokenizer.is_fast}")
    print(f"  vocab size           {tokenizer.vocab_size}")

    print()
    print("[encode]")
    encode_times = {}
    for name, text in PROMPTS:
        warm = tokenizer(text, add_special_tokens=False)["input_ids"]
        e0 = time.perf_counter()
        for _ in range(ITERATIONS):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        e1 = time.perf_counter()
        per_call = (e1 - e0) / ITERATIONS
        encode_times[name] = per_call
        print(
            f"   {name:<8} chars {len(text):<4} tokens {len(warm):<4} "
            f"per call {per_call * 1000:.6f} ms"
        )

    print()
    print("[totals for one cold request]")
    typical = encode_times["typical"]
    print(f"  import + construct + one typical encode: "
          f"{import_seconds + construct_seconds + typical:.6f} s")
    print(f"  construct + one typical encode (warm interpreter): "
          f"{construct_seconds + typical:.6f} s")
    print()
    print("  Mojo, for comparison (measured, -O2):")
    print("    construct from JSON   0.186558 s")
    print("    construct from cache  0.054635 s")
    print("    typical encode        0.073538 ms")


if __name__ == "__main__":
    main()
