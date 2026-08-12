# CPU-only exact-ID gate for the tokenizer.json embedded in the real LTX-2.5
# Gemma checkpoint. No DeviceContext or CUDA symbols are used.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime LTX25_GEMMA_TE = "/home/alex/.serenity/models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
comptime STANDALONE_TOKENIZER = "/home/alex/.serenity/models/text_encoders/gemma-4-12b-it-standalone/tokenizer.json"
comptime EXPECTED_TOKENIZER_JSON_BYTES = 32169626
comptime HANDOFF_PROMPT = "A woman with long brown hair walks through a sunlit forest, leaves crunching underfoot as birdsong echoes."


def _embedded_pair() raises -> Tuple[List[Int], List[Int]]:
    # `st` owns the mmap for the full origin-bound parsing operation.
    var st = SafeTensors.open(String(LTX25_GEMMA_TE))
    if not st.has_tensor("tokenizer_json"):
        raise Error("probe: checkpoint is missing tokenizer_json")
    var info = st.tensor_info("tokenizer_json")
    if info.dtype != STDtype.U8:
        raise Error(
            String("probe: tokenizer_json dtype is ") + info.dtype.name()
            + String(", expected U8")
        )
    if len(info.shape) != 1 or info.shape[0] <= 0:
        raise Error("probe: tokenizer_json must be nonempty rank-1")
    if info.shape[0] != info.size or info.size != EXPECTED_TOKENIZER_JSON_BYTES:
        raise Error(
            String("probe: tokenizer_json shape/bytes mismatch: shape=")
            + String(info.shape[0]) + String(" bytes=") + String(info.size)
        )
    var bytes = st.tensor_bytes("tokenizer_json")
    if len(bytes) != info.size:
        raise Error("probe: mapped tokenizer_json byte count mismatch")
    var tokenizer = Qwen3Tokenizer.from_json_bytes(bytes)
    var positive = tokenizer.encode_gemma(String(HANDOFF_PROMPT))
    var negative = tokenizer.encode_gemma(String(""))
    return (positive^, negative^)


def _standalone_pair() raises -> Tuple[List[Int], List[Int]]:
    var tokenizer = Qwen3Tokenizer(String(STANDALONE_TOKENIZER))
    var positive = tokenizer.encode_gemma(String(HANDOFF_PROMPT))
    var negative = tokenizer.encode_gemma(String(""))
    return (positive^, negative^)


def _require_exact(
    label: String, embedded: List[Int], standalone: List[Int]
) raises:
    print(
        label,
        "embedded_len=", len(embedded),
        "standalone_len=", len(standalone),
    )
    if len(embedded) == 0 or len(standalone) == 0:
        raise Error(label + String(": empty token IDs"))
    if embedded[0] != 2 or standalone[0] != 2:
        raise Error(
            label + String(": BOS mismatch embedded=") + String(embedded[0])
            + String(" standalone=") + String(standalone[0])
        )
    if len(embedded) != len(standalone):
        raise Error(label + String(": token length mismatch"))
    for i in range(len(embedded)):
        if embedded[i] != standalone[i]:
            raise Error(
                label + String(": token mismatch at ") + String(i)
                + String(" embedded=") + String(embedded[i])
                + String(" standalone=") + String(standalone[i])
            )
    print(label, "BOS=2 exact_ids=PASS")


def main() raises:
    var embedded = _embedded_pair()
    var standalone = _standalone_pair()
    _require_exact("handoff_prompt", embedded[0], standalone[0])
    _require_exact("empty_negative", embedded[1], standalone[1])
    print("LTX-2.5 embedded tokenizer parity PASS")
