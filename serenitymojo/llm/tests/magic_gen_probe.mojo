# magic_gen_probe.mojo — DIAGNOSE the captioner over-generation. Runs the exact
# magic-prompt chat + generation, then reports: prompt len, generated len (==1700
# ⇒ never hit EOS), and decoded head/tail (what does it emit after the JSON?).
from std.gpu.host import DeviceContext
from serenitymojo.pipeline.ideogram4_magic import _system_prompt
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.text_encoder.qwen3_magic import generate_greedy_cached

comptime QWEN = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
comptime TOKJSON = QWEN + "/tokenizer.json"
comptime EOS = 151645


def _sub(ids: List[Int], a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(a, b):
        out.append(ids[i])
    return out^


def main() raises:
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(TOKJSON)
    var user = (
        String("TARGET IMAGE ASPECT RATIO: 1:1 (width:height).\nUser idea: ")
        + String("a red cube on a white table /no_think")
    )
    var chat = (
        String("<|im_start|>system\n") + _system_prompt() + "<|im_end|>\n"
        + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
    )
    var ids = tok.encode(chat)
    print("prompt tokens:", len(ids))
    var qwen = Qwen3Encoder.load(QWEN, Qwen3Config.klein_9b(), ctx)
    var gen = generate_greedy_cached(qwen, ids, 1700, EOS, ctx)
    print("generated tokens:", len(gen), " (== 1700 ⇒ hit max_new, never emitted EOS)")
    print("first 40 gen ids:")
    for i in range(min(40, len(gen))): print(gen[i], end=" ")
    print("")
    print("last 40 gen ids:")
    for i in range(max(0, len(gen) - 40), len(gen)): print(gen[i], end=" ")
    print("")
    print("=== decoded HEAD (first 60 tokens):")
    print(tok.decode(_sub(gen, 0, min(60, len(gen)))))
    if len(gen) > 120:
        print("=== decoded TAIL (last 60 tokens):")
        print(tok.decode(_sub(gen, len(gen) - 60, len(gen))))
