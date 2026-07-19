# SKEPTIC edge-case spot check: directly encode tricky inputs and print ids,
# so we don't rely on the TSV roundtrip hiding anything.
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

comptime TOK_JSON: StaticString = "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/tokenizer/tokenizer.json"

def _show(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        if i != 0:
            s += String(",")
        s += String(ids[i])
    s += String("]")
    return s^

def p(tok: Qwen3Tokenizer, label: String, text: String) raises:
    print(label, "->", _show(tok.encode(text)))

def main() raises:
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    p(tok, "empty            ", String(""))
    p(tok, "single space     ", String(" "))
    p(tok, "single bang      ", String("!"))
    p(tok, "curly apostrophe ", String("it") + String(chr(0x2019)) + String("s"))
    p(tok, "curly dquote     ", String(chr(0x201C)) + String("hi") + String(chr(0x201D)))
    p(tok, "tis              ", String(chr(0x27)) + String("tis the season"))
    p(tok, "DON'T            ", String("DON") + String(chr(0x27)) + String("T"))
    p(tok, "yalldve          ", String("y") + String(chr(0x27)) + String("all") + String(chr(0x27)) + String("d") + String(chr(0x27)) + String("ve"))
    p(tok, "two specials adj ", String("two<|im_start|><|im_end|>adjacent"))
    p(tok, "partial special  ", String("partial <|im_ token here"))
    p(tok, "think tokens     ", String("use <think> reasoning </think> then answer"))
    p(tok, "backslash path   ", String("C:") + String(chr(0x5C)) + String("Users") + String(chr(0x5C)) + String("name"))
    p(tok, "crlf             ", String("windows") + String(chr(0x0D)) + String(chr(0x0A)) + String("line"))
    p(tok, "ZWJ family       ", String(chr(0x1F468)) + String(chr(0x200D)) + String(chr(0x1F469)) + String(chr(0x200D)) + String(chr(0x1F467)))
    p(tok, "regional flag US ", String(chr(0x1F1FA)) + String(chr(0x1F1F8)))
    p(tok, "skin tone        ", String(chr(0x1F44D)) + String(chr(0x1F3FD)))
    # NFC severity probes
    p(tok, "cafe precomposed ", String("caf") + String(chr(0xE9)))
    p(tok, "cafe decomposed  ", String("cafe") + String(chr(0x0301)))
    p(tok, "u-umlaut precomp ", String(chr(0xFC)))
    p(tok, "u-umlaut decomp  ", String("u") + String(chr(0x0308)))
    p(tok, "korean precomp   ", String(chr(0xD55C)) + String(chr(0xAD6D)))  # 한국
    p(tok, "korean jamo      ", String(chr(0x1112)) + String(chr(0x1161)) + String(chr(0x11AB)) + String(chr(0x1100)) + String(chr(0x116E)) + String(chr(0x11A8)))
