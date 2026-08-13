# magic_time_breakdown.mojo — locate the captioner's ~4:45 wall time: split it into
# tokenize / model-load / prompt-prime / generate. decode_step calls inlined with
# perf_counter checkpoints (mirrors generate_greedy_cached).
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.pipeline.ideogram4_magic import _system_prompt
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.llm.decoder import KVCache, decode_step

comptime QWEN = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
comptime TOKJSON = QWEN + "/tokenizer.json"
comptime EOS = 151645


def _s(a: UInt, b: UInt) -> Float64:
    return Float64(Int(b) - Int(a)) / 1.0e9


def _argmax(t: Tensor, ctx: DeviceContext) raises -> Int:
    var h = t.to_host(ctx)
    var best = 0; var bv = h[0]
    for i in range(1, len(h)):
        if h[i] > bv: bv = h[i]; best = i
    return best


def main() raises:
    var ctx = DeviceContext()
    var t0 = perf_counter_ns()
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
    var t_tok = perf_counter_ns()
    var qwen = Qwen3Encoder.load(QWEN, Qwen3Config.klein_9b(), ctx)
    var t_load = perf_counter_ns()

    # prime (want_logits=False except last)
    var cache = KVCache(qwen.config.num_layers)
    var n = len(ids)
    for pos in range(n - 1):
        _ = decode_step(qwen, cache, ids[pos], pos, ctx, want_logits=False)
    var last = decode_step(qwen, cache, ids[n - 1], n - 1, ctx, want_logits=True)
    ctx.synchronize()
    var t_prime = perf_counter_ns()

    # generate
    var gen = List[Int]()
    var pos = n
    for _ in range(1700):
        var best = _argmax(last, ctx)
        if best == EOS: break
        gen.append(best)
        last = decode_step(qwen, cache, best, pos, ctx, want_logits=True)
        pos += 1
    ctx.synchronize()
    var t_gen = perf_counter_ns()

    print("prompt tokens:", n, " generated:", len(gen))
    print("tokenize  s:", _s(t0, t_tok))
    print("model load s:", _s(t_tok, t_load))
    print("prime     s:", _s(t_load, t_prime), " (", _s(t_load, t_prime) / Float64(n) * 1000.0, "ms/tok )")
    print("generate  s:", _s(t_prime, t_gen), " (", _s(t_prime, t_gen) / Float64(len(gen)) * 1000.0 if len(gen) > 0 else 0.0, "ms/tok )")
