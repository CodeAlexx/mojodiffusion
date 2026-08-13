# prefix_cache_gate.mojo — GATE the prefix-cache path: generation from a saved+
# loaded prefix K/V must be TOKEN-FOR-TOKEN identical to a fresh full prime over
# prefix+tail (greedy ⇒ exact equality). Real Qwen3-8B + the real captioner chat.
# Also times both paths (the measured win) and the save/load round-trip.
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.pipeline.ideogram4_magic import _system_prompt
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.text_encoder.qwen3_magic import (
    generate_greedy_cached, generate_greedy_cached_resume,
)
from serenitymojo.llm.decoder import KVCache, decode_step
from serenitymojo.llm.prefix_cache import save_prefix_cache, load_prefix_cache

comptime QWEN = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
comptime TOKJSON = QWEN + "/tokenizer.json"
comptime EOS = 151645
comptime CACHE_PATH = "/tmp/prefix_cache_gate.safetensors"


def _s(a: UInt, b: UInt) -> Float64:
    return Float64(Int(b) - Int(a)) / 1.0e9


def main() raises:
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(TOKJSON)
    var prefix_str = (
        String("<|im_start|>system\n") + _system_prompt() + "<|im_end|>\n"
        + "<|im_start|>user\n"
    )
    var user = (
        String("TARGET IMAGE ASPECT RATIO: 1:1 (width:height).\nUser idea: ")
        + String("a red cube on a white table /no_think")
    )
    var chat = prefix_str + user + "<|im_end|>\n<|im_start|>assistant\n"
    var ids = tok.encode(chat)
    var prefix_ids = tok.encode(prefix_str)
    print("full ids:", len(ids), " prefix ids:", len(prefix_ids))

    # BPE seam: full ids must start with prefix ids.
    var seam = len(ids) > len(prefix_ids)
    if seam:
        for i in range(len(prefix_ids)):
            if ids[i] != prefix_ids[i]:
                seam = False
                break
    print("seam (full starts with prefix):", "OK" if seam else "MISMATCH")
    if not seam:
        print("GATE FAIL: BPE seam mismatch — prefix caching invalid for this prompt")
        return

    var qwen = Qwen3Encoder.load(QWEN, Qwen3Config.klein_9b(), ctx)

    # ── path A: fresh FULL prime over prefix+tail (reference) ──
    var tA0 = perf_counter_ns()
    var gen_a = generate_greedy_cached(qwen, ids, 400, EOS, ctx)
    var tA1 = perf_counter_ns()

    # ── path B: prime prefix → SAVE → LOAD → resume tail ──
    var tB0 = perf_counter_ns()
    var cache = KVCache(qwen.config.num_layers)
    for pos in range(len(prefix_ids)):
        _ = decode_step(qwen, cache, prefix_ids[pos], pos, ctx, want_logits=False)
    var tB1 = perf_counter_ns()
    save_prefix_cache(cache, prefix_ids, CACHE_PATH, ctx)
    var tB2 = perf_counter_ns()
    var stored_ids = List[Int]()
    var lcache = load_prefix_cache(CACHE_PATH, qwen.config.num_layers, ctx, stored_ids)
    var tB3 = perf_counter_ns()
    if len(stored_ids) != len(prefix_ids):
        print("GATE FAIL: stored prefix_ids length mismatch")
        return
    var tail = List[Int]()
    for i in range(len(prefix_ids), len(ids)):
        tail.append(ids[i])
    var gen_b = generate_greedy_cached_resume(
        qwen, lcache, len(prefix_ids), tail, 400, EOS, ctx
    )
    var tB4 = perf_counter_ns()

    # ── token-for-token equality ──
    var same = len(gen_a) == len(gen_b)
    if same:
        for i in range(len(gen_a)):
            if gen_a[i] != gen_b[i]:
                same = False
                break
    print("gen A (full prime):", len(gen_a), " tokens in", _s(tA0, tA1), "s")
    print("gen B breakdown: prime", _s(tB0, tB1), "s  save", _s(tB1, tB2),
          "s  load", _s(tB2, tB3), "s  resume+gen", _s(tB3, tB4), "s")
    print("CACHED-PATH per-caption cost (load + resume+gen):",
          _s(tB2, tB3) + _s(tB3, tB4), "s")
    if same:
        print("GATE PASS: prefix-cached generation token-for-token identical (",
              len(gen_a), "tokens )")
    else:
        print("GATE FAIL: sequences diverge")
