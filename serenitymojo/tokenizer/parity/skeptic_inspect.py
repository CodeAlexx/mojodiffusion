#!/usr/bin/env python3
"""SKEPTIC: inspect tokenizer.json config + re-derive the hardcoded smoke ids
FRESH from HF tokenizers, so we can prove the gate is not circular.

Usage: pixi run python serenitymojo/tokenizer/parity/skeptic_inspect.py
"""
import json

JSON = ("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
        "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json")

# ---- config inspection (raw JSON) ----
with open(JSON, "r", encoding="utf-8") as f:
    cfg = json.load(f)

print("=== CONFIG ===")
print("model.type:", cfg["model"]["type"])
print("model.byte_fallback:", cfg["model"].get("byte_fallback"))
print("model.ignore_merges:", cfg["model"].get("ignore_merges"))
print("vocab size:", len(cfg["model"]["vocab"]))
print("merges count:", len(cfg["model"]["merges"]))
norm = cfg.get("normalizer")
print("normalizer:", json.dumps(norm) if norm is not None and not isinstance(norm, dict) else (norm["type"] if isinstance(norm, dict) else norm))
print("normalizer raw:", json.dumps(norm)[:200])
pt = cfg.get("pre_tokenizer")
print("pre_tokenizer.type:", pt["type"] if isinstance(pt, dict) else pt)
if isinstance(pt, dict) and pt.get("type") == "Sequence":
    for i, sub in enumerate(pt["pretokenizers"]):
        print(f"  pre[{i}] {sub.get('type')}: {json.dumps(sub)[:300]}")
print("post_processor:", json.dumps(cfg.get("post_processor"))[:200])
added = cfg.get("added_tokens", [])
print("added_tokens count:", len(added))
for a in added:
    print(f"  id={a['id']} content={a['content']!r} special={a.get('special')} normalized={a.get('normalized')} lstrip={a.get('lstrip')} rstrip={a.get('rstrip')} single_word={a.get('single_word')}")

# ---- merges format check ----
m0 = cfg["model"]["merges"][0]
print("merges[0] type:", type(m0).__name__, "value:", m0)

# ---- vocab keys with escapes (item 6) ----
print("\n=== VOCAB KEYS WITH POTENTIAL JSON-ESCAPE ISSUES ===")
raw = open(JSON, "rb").read()
# look for \u, \", \\ in the vocab region by scanning the parsed keys for chars
suspicious = 0
for k in cfg["model"]["vocab"].keys():
    # any char that the file would have escaped as \", \\, \n, \t, \r, or \uXXXX
    for ch in k:
        o = ord(ch)
        if ch in ('"', "\\") or o in (0x0a, 0x09, 0x0d) or o < 0x20:
            suspicious += 1
            if suspicious <= 40:
                print(f"  key={k!r} ids={cfg['model']['vocab'][k]} contains char U+{o:04X}")
            break
print("total vocab keys containing escape-relevant chars:", suspicious)

# ---- re-derive smoke ids FRESH ----
from tokenizers import Tokenizer
tk = Tokenizer.from_file(JSON)

SMOKE = {
    "plain ASCII":           ("hello world", [14990, 1879]),
    "mixed case + space":    ("Hello World", [9707, 4337]),
    "leading spaces":        ("  leading spaces", [220, 6388, 12621]),
    "trailing spaces":       ("trailing spaces  ", [376, 14277, 12621, 256]),
    "ascii+digits+punct":    ("The quick brown fox jumps over 13 lazy dogs.", [785, 3974, 13876, 38835, 34208, 916, 220, 16, 18, 15678, 12590, 13]),
    "letter/digit boundary": ("a1b2c3", [64, 16, 65, 17, 66, 18]),
    "contractions+punct":    ("Don't stop, won't you?", [8002, 944, 2936, 11, 2765, 944, 498, 30]),
    "mixed case words":      ("MixedCASE Words", [86433, 40371, 27630]),
    "digits + punct":        ("123 456.78", [16, 17, 18, 220, 19, 20, 21, 13, 22, 23]),
    "embedded newline":      ("newline\ntest", [89202, 198, 1944]),
    "special token":         ("<|im_start|>user", [151644, 872]),
    "cafe precomposed":      ("café", [924, 58858]),
    "emoji":                 ("I love \U0001F680 rockets", [40, 2948, 11162, 248, 222, 51998]),
}

print("\n=== FRESH RE-DERIVE OF HARDCODED SMOKE IDS ===")
nfail = 0
for name, (text, hard) in SMOKE.items():
    fresh = tk.encode(text).ids
    ok = fresh == hard
    if not ok:
        nfail += 1
    print(f"  [{'OK ' if ok else 'BAD'}] {name}: hardcoded={hard}")
    if not ok:
        print(f"        FRESH HF ={fresh}")
print(f"\nHardcoded-vs-fresh smoke mismatches: {nfail}/{len(SMOKE)}")

# ---- also re-derive the documented 'known divergence' oracle comments ----
print("\n=== documented known-divergence oracle comments ===")
dec = "café"
print("decomposed 'cafe\\u0301' fresh HF =", tk.encode(dec).ids, "(smoke comment claims oracle=[924,58858])")
arabic = "٤٥٦"
print("arabic-indic digits fresh HF =", tk.encode(arabic).ids, "(smoke comment claims oracle=[149,97,149,98,149,99])")
