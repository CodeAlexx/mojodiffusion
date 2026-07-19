# Parity gate: pure-Mojo o200k (GPT-OSS) tokenizer vs Serenity reference for
# the Lens prompt.
#
# Renders the FIXED Lens chat template (the exact `rendered` string produced by
# lens/pipeline.py for the woman+owl caption; the assistant turn is left open at
# "<|message|>"), tokenizes it with the pure-Mojo o200k tokenizer, and compares
# the real (non-pad) ids to parity/lens/tok_ref.json ("input_ids", real_len 298).
#
# GATE: >= 297/298 real ids nmatch (allow at most 1 off for chat-template edges).
#
# Build+run (from /home/alex/mojodiffusion):
#   rm -f serenitymojo.mojopkg && pixi run mojo build -I . \
#     -I trainer/src -Xlinker -lm \
#     trainer/src/serenity_trainer/smoke/lens_tok_parity_smoke.mojo \
#     -o /tmp/lens_tok && /tmp/lens_tok

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer, _read_utf8_file

comptime TOK_JSON: StaticString = "/home/alex/.serenity/models/microsoft_lens/tokenizer/tokenizer.json"
comptime REF_JSON: StaticString = "trainer/parity/lens/tok_ref.json"
comptime REAL_LEN: Int = 298

# The exact `rendered` string from tok_ref.json (Lens chat template applied to
# the woman+owl caption). This is the INPUT to tokenize, not the output ids.
comptime RENDERED: StaticString = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-06-07\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background.\n\n<|end|><|start|>user<|message|>Sharp lines, a close-up, artistic portrait featuring a woman and an owl. The composition is tightly framed, focusing on the woman's face and the owl's head, both occupying equal space. The woman has pale skin, with striking, large eyes that are a light shade, possibly blue or gray. Her expression is intense and slightly melancholic, with subtle scratches or marks on her face adding a sense of mystery. Her dark hair blends into the background, enhancing the focus on her facial features. The owl, positioned to the left, has detailed feathers in shades of brown and gray, with large, expressive orange eyes that mirror the intensity of the woman's gaze. The overall color palette is muted, with cool tones dominating the scene, creating a harmonious and enigmatic atmosphere. The lighting is soft, highlighting the textures of the skin and feathers, while the background remains blurred, drawing attention to the subjects.<|end|><|start|>assistant<|channel|>analysis<|message|>Need to generate one image according to the description.<|end|><|start|>assistant<|channel|>final<|message|>"


def _load_ref_ids(path: String, want: Int) raises -> List[Int]:
    # Read tok_ref.json and extract the first `want` ints from "input_ids".
    var text = _read_utf8_file(path)
    var bytes = text.as_bytes()
    var n = len(bytes)
    # locate the "input_ids" key
    var needle = String('"input_ids"')
    var nb = needle.as_bytes()
    var nlen = len(nb)
    var start = -1
    var i = 0
    while i + nlen <= n:
        var matched = True
        for j in range(nlen):
            if bytes[i + j] != nb[j]:
                matched = False
                break
        if matched:
            start = i + nlen
            break
        i += 1
    if start < 0:
        raise Error(String("tok_ref.json: input_ids not found"))
    # advance to '['
    var p = start
    while p < n and bytes[p] != 0x5B:  # '['
        p += 1
    p += 1
    var out = List[Int]()
    while p < n and len(out) < want:
        var b = Int(bytes[p])
        if b == 0x5D:  # ']'
            break
        if (b >= 48 and b <= 57) or b == 0x2D:  # digit or '-'
            var neg = False
            if b == 0x2D:
                neg = True
                p += 1
            var v = 0
            while p < n and Int(bytes[p]) >= 48 and Int(bytes[p]) <= 57:
                v = v * 10 + (Int(bytes[p]) - 48)
                p += 1
            out.append(-v if neg else v)
        else:
            p += 1
    return out^


def main() raises:
    print("== Lens o200k tokenizer parity ==")
    var tok = Qwen3Tokenizer(String(TOK_JSON), True)  # o200k=True

    var got = tok.encode(String(RENDERED))
    var refids = _load_ref_ids(String(REF_JSON), REAL_LEN)

    print("mojo ids:", len(got), " refids real ids:", len(refids))

    var compare_n = REAL_LEN
    if len(got) < compare_n:
        compare_n = len(got)
    if len(refids) < compare_n:
        compare_n = len(refids)

    var nmatch = 0
    var first_mismatch = -1
    for k in range(compare_n):
        if got[k] == refids[k]:
            nmatch += 1
        elif first_mismatch < 0:
            first_mismatch = k

    # account for length differences as mismatches against the 298 target
    print("#nmatch:", nmatch, "/", REAL_LEN)
    if len(got) != REAL_LEN:
        print("WARN: mojo produced", len(got), "ids (expected", REAL_LEN, ")")

    if first_mismatch >= 0:
        print(
            "first mismatch @", first_mismatch,
            " mojo=", got[first_mismatch],
            " refids=", refids[first_mismatch],
        )
        # context window around the mismatch
        var lo = first_mismatch - 3
        if lo < 0:
            lo = 0
        var hi = first_mismatch + 4
        if hi > compare_n:
            hi = compare_n
        var sm = String("  mojo[")
        var sr = String("  refids [")
        for k in range(lo, hi):
            if k != lo:
                sm += String(", ")
                sr += String(", ")
            sm += String(got[k])
            sr += String(refids[k])
        sm += String("]")
        sr += String("]")
        print(sm)
        print(sr)
    else:
        print("no mismatch in first", compare_n, "ids")

    if nmatch >= REAL_LEN - 1 and len(got) == REAL_LEN:
        print("GATE: PASS (", nmatch, "/", REAL_LEN, ")")
    else:
        print("GATE: FAIL (", nmatch, "/", REAL_LEN, ")")
