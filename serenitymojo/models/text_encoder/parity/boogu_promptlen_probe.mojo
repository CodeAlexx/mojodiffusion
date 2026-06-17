# boogu_promptlen_probe.mojo — verify the Mojo tokenizer's cond token count for the
# pipeline's INSTRUCTION matches the comptime CAP_LEN_COND (must == processor's 317).
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.boogu_tokenizer import boogu_tokenize
from serenitymojo.pipeline.boogu_pipeline import INSTRUCTION

comptime TOK_JSON = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/mllm/tokenizer.json"


def main() raises:
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var ids = boogu_tokenize(tok, String(INSTRUCTION))
    print("mojo cond token count =", len(ids), " (expect 317 to match the processor)")
