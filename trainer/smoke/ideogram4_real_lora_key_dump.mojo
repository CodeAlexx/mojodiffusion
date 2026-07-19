# ideogram4_real_lora_key_dump.mojo — print a small summary of a real ai-toolkit
# Ideogram-4 LoRA safetensors file. This is an audit helper, not a gate.
from std.gpu.host import DeviceContext

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor


comptime REAL_LORA = "/home/alex/Downloads/dever_arcane_style_ideogram4%20%28arcvfx%29.safetensors"


def _contains(text: String, token: String) -> Bool:
    var text_len = text.byte_length()
    var token_len = token.byte_length()
    if token_len <= 0:
        return True
    if text_len < token_len:
        return False
    var last = text_len - token_len
    for i in range(last + 1):
        if String(text[byte=i:i + token_len]) == token:
            return True
    return False


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(REAL_LORA))
    var n = 0
    var a = 0
    var b = 0
    var down = 0
    var up = 0
    var alpha = 0
    print("real ideogram4 lora:", String(REAL_LORA))
    print("num_tensors:", st.num_tensors())
    for ref nm in st.names():
        n += 1
        if _contains(nm, String(".lora_A.weight")):
            a += 1
        if _contains(nm, String(".lora_B.weight")):
            b += 1
        if _contains(nm, String(".lora_down.weight")):
            down += 1
        if _contains(nm, String(".lora_up.weight")):
            up += 1
        if _contains(nm, String(".alpha")):
            alpha += 1
        if n <= 30:
            var t = Tensor.from_view(st.tensor_view(nm), ctx)
            var sh = t.shape()
            var line = nm + String(" dtype=") + t.dtype().name() + String(" shape=[")
            for i in range(len(sh)):
                if i > 0:
                    line += String(",")
                line += String(sh[i])
            line += String("]")
            print(line)
    print("counts: lora_A=", a, " lora_B=", b, " lora_down=", down, " lora_up=", up, " alpha=", alpha)
