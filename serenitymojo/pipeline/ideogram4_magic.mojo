# pipeline/ideogram4_magic.mojo — pure-Mojo magic prompt: plain text -> JSON caption.
# Runs Qwen3-8B (lm_head present) autoregressively in Mojo with a magic-prompt
# system prompt, emitting an Ideogram-4 structured JSON caption. The JSON is then
# tokenized + fed to ideogram4_generate (separate step). No external LLM/API.
from std.gpu.host import DeviceContext
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.text_encoder.qwen3_magic import generate_greedy
from json.parser import loads
from json.canonical import minify
from serenitymojo.io.ffi import BytePtr

comptime QWEN = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
comptime TOKJSON = QWEN + "/tokenizer.json"
comptime EOS = 151645      # <|im_end|>
comptime PAD = 151643      # <|endoftext|>


def _system_prompt() -> String:
    return String(
        "You convert a natural-language image idea into EXACTLY ONE minified JSON"
        " object (single line, no markdown, no commentary) that an image renderer"
        " consumes. Schema, strict key order:\n"
        '{"high_level_description":"<=50 words, one sentence, starts with the subject",'
        '"style_description":{"aesthetics":"...","lighting":"...","photo":"camera/lens (photographic) OR omit and use art_style","medium":"photograph|illustration|3d_render|digital_art|painting","color_palette":["#RRGGBB",...up to 16]},'
        '"compositional_deconstruction":{"background":"...","elements":[{"type":"obj","bbox":[y_min,x_min,y_max,x_max],"desc":"...","color_palette":["#RRGGBB"]}]}}\n'
        "Rules: style_description has EXACTLY ONE of photo (with medium:photograph)"
        " OR art_style (non-photo). bbox is normalized 0-1000, origin top-left,"
        " optional. text elements use {type:text,bbox,text,desc}. Uppercase"
        " #RRGGBB only. Output the JSON object and nothing else."
    )


def _byte_substr(s: String, start: Int, end: Int) -> String:
    """Byte substring s[start:end). Mojo String has no slice operator, so view the
    backing bytes and copy out the range. (Mirrors serve/proc_ipc.byte_substr.)"""
    var n = end - start
    if n <= 0:
        return String("")
    var sp = s.as_bytes()
    var base = BytePtr(unsafe_from_address=Int(sp.unsafe_ptr()) + start)
    return String(StringSlice(ptr=base, length=n))


def _strip_trailing_commas(s: String) raises -> String:
    """Remove any ',' immediately followed (ignoring whitespace) by '}' or ']'.
    LLMs emit these trailing commas; json.parser.loads (strict RFC8259) rejects
    them. A comma is dropped ONLY when, scanning forward past spaces/tabs/newlines/
    carriage-returns, the next non-space byte is '}' or ']'. Commas inside double-
    quoted string literals are never touched: we track whether we are inside a
    string and honor backslash escapes."""
    var bs = s.as_bytes()
    var n = s.byte_length()
    var out = List[UInt8]()
    var in_str = False
    var escaped = False
    for i in range(n):
        var c = bs[i]
        if in_str:
            out.append(c)
            if escaped:
                escaped = False
            elif c == 0x5C:        # backslash -> next char is escaped
                escaped = True
            elif c == 0x22:        # closing quote
                in_str = False
            continue
        # not inside a string literal
        if c == 0x22:              # opening quote
            in_str = True
            out.append(c)
            continue
        if c == 0x2C:              # ',' — look ahead past whitespace
            var j = i + 1
            while j < n:
                var d = bs[j]
                if d == 0x20 or d == 0x09 or d == 0x0A or d == 0x0D:
                    j += 1
                    continue
                break
            if j < n and (bs[j] == 0x7D or bs[j] == 0x5D):  # '}' or ']'
                # drop this comma (do not append); whitespace is emitted as normal
                # by subsequent iterations
                continue
            out.append(c)
            continue
        out.append(c)
    return String(from_utf8=out)


def _normalize_caption(raw: String) raises -> String:
    """Turn a raw LLM completion into CANONICAL minified Ideogram-4 JSON.

    1. Trim. If a markdown code fence (```) is present, take the content between the
       first fence pair and strip a leading `json` language tag.
    2. Slice from the FIRST '{' to the LAST '}' (the JSON object), discarding any
       LLM preamble/suffix.
    3. Strip trailing commas (LLMs emit them; strict loads rejects them).
    4. loads() -> minify() (INSERTION-ORDER keys, preserving trained key order).
       If loads raises, re-raise with a short prefix of the raw input (FAIL LOUD —
       never return un-parseable JSON)."""
    var s = String(raw.strip())

    # 1. markdown code fence extraction
    var fence = String("```")
    var f0 = s.find(fence)
    if f0 >= 0:
        var inner_start = f0 + 3
        var f1 = s.find(fence, inner_start)
        if f1 > inner_start:
            s = String(_byte_substr(s, inner_start, f1).strip())
        else:
            # opening fence only; take everything after it
            s = String(_byte_substr(s, inner_start, s.byte_length()).strip())
        # strip a leading `json` language tag
        if s.find("json") == 0:
            s = String(_byte_substr(s, 4, s.byte_length()).strip())

    # 2. slice FIRST '{' .. LAST '}'
    var lb = s.find("{")
    var rb = s.rfind("}")
    if lb < 0 or rb < 0 or rb < lb:
        raise Error(
            String("magic_expand: no JSON object found in LLM output; prefix=")
            + _byte_substr(String(raw.strip()), 0, 200)
        )
    s = _byte_substr(s, lb, rb + 1)

    # 3. strip trailing commas
    s = _strip_trailing_commas(s)

    # 4. parse (strict) then canonical minify (insertion order)
    try:
        var v = loads(s)
        return minify(v)
    except e:
        raise Error(
            String("magic_expand: LLM emitted un-parseable JSON (")
            + String(e)
            + String("); cleaned prefix=")
            + _byte_substr(s, 0, 200)
        )


def magic_expand(plain: String, aspect: String, ctx: DeviceContext) raises -> String:
    """Expand a plain prompt into a CANONICAL minified Ideogram-4 JSON caption.

    Builds the chat (magic-prompt system + user line), encodes with the Qwen3
    tokenizer, loads Qwen3-8B, runs greedy generation, decodes, and normalizes the
    result to canonical minified JSON. Loud-fails if the model output is not a
    parseable JSON object."""
    var tok = Qwen3Tokenizer(TOKJSON)
    var user = plain + String(" (aspect ratio ") + aspect + String(") /no_think")
    var chat = (
        String("<|im_start|>system\n") + _system_prompt() + "<|im_end|>\n"
        + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
    )
    var ids = tok.encode(chat)
    var qwen = Qwen3Encoder.load(QWEN, Qwen3Config.klein_9b(), ctx)
    var gen = generate_greedy(qwen, ids, 1700, EOS, PAD, 2048, ctx)
    var decoded = tok.decode(gen)
    return _normalize_caption(decoded)


def main() raises:
    var ctx = DeviceContext()
    var plain = String("a red cube on a white table")
    var aspect = String("1:1")
    print("expanding magic prompt (greedy)...")
    var caption = magic_expand(plain, aspect, ctx)
    print("=== MAGIC PROMPT JSON ===")
    print(caption)
    print("=== END ===")
