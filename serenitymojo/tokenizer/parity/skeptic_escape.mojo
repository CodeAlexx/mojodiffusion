# SKEPTIC: inputs whose correct tokenization needs escape-heavy vocab keys
# (containing " or backslash) to have been parsed correctly by the hand-rolled
# JSON reader. If \" or \\ handling is wrong, these diverge from HF.
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
    var q = String(chr(0x22))    # "
    var bs = String(chr(0x5C))   # backslash
    p(tok, "bare dquote      ", q)
    p(tok, "bare backslash   ", bs)
    p(tok, "quote comma      ", q + String(","))
    p(tok, "eq quote         ", String("=") + q)
    p(tok, "paren quote      ", String("(") + q)
    p(tok, "quote newline    ", q + String(chr(0x0A)))
    p(tok, "space backslash  ", String(" ") + bs)
    var bs2 = String(chr(0x5C)) + String(chr(0x5C))
    p(tok, "double backslash ", bs2)
    p(tok, "json-ish snippet ", String("{") + q + String("key") + q + String(": ") + q + String("val") + q + String("}"))
    p(tok, "html attr        ", String("<a href=") + q + String("url") + q + String(">"))
    p(tok, "win path quoted  ", q + String("C:") + bs + String("dir") + bs + String("f") + q)
    p(tok, "escaped quote str", String("text ") + bs + q + String("inside") + bs + q)
