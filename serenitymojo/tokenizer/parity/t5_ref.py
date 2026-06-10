#!/usr/bin/env python3
# Ground-truth oracle for the pure-Mojo T5 Unigram tokenizer.
#
# Prints `text -> input_ids` for a fixed set of strings using the HF
# T5TokenizerFast (the exact tokenizer Chroma/SD3.5/FLUX/Anima feed the T5
# text encoder). CPU-only; no GPU. Diff this against t5_tokenizer_smoke output.
#
# Run:  python3 serenitymojo/tokenizer/parity/t5_ref.py
#
# input_ids already include the trailing EOS (id 1) appended by the
# post_processor, and NO padding -- matching T5Tokenizer.encode in Mojo.

from transformers import AutoTokenizer

TESTS = [
    "a photo of a cat",
    "don't",
    "3 cats",
    "HELLO World",
    "café déjà vu — naïve",            # unicode (note: NFKC-stable accents)
    "",                                  # empty
    ("a highly detailed photograph of a majestic lion standing on a rocky "
     "cliff at sunset, golden hour lighting, 8k"),   # long prompt
    "  leading and trailing  ",          # leading/trailing/repeat spaces
]


def main():
    tok = AutoTokenizer.from_pretrained("t5-base")
    # Emit ONE id-list per line, in order, nothing else -- so a raw `diff`
    # against the Mojo smoke output is exact and meaningful.
    for t in TESTS:
        print(tok(t).input_ids)


if __name__ == "__main__":
    main()
