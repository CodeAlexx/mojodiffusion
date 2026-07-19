# wan22_encode_prompt.mojo — Wan2.2 umt5-xxl prompt-encode (separate process).
#
# Loads the umt5-xxl text encoder, encodes the positive + negative prompts, and
# writes both embeddings + their valid lengths to a single safetensors, then
# main() returns and the process exits. Process death frees ALL ~11 GB of the
# umt5 encoder GPU memory — the hardest possible separation from the Wan2.2
# TI2V-5B DiT, which runs in a SEPARATE process (pipeline/wan22_t2v.mojo) that
# never imports Umt5Encoder. This mirrors klein9b_encode_smoke.mojo exactly.
#
# Encoder: serenitymojo/models/text_encoder/umt5_encoder.mojo. The BF16 weights
# at UMT5_PATH (242 keys, blocks.{i}.* schema) are the SAME umt5-xxl encoder
# Wan2.2 ships (models_t5_umt5-xxl-enc-bf16.pth); the port is gated cos>=0.999
# against the NAVA oracle (models/text_encoder/parity/nava_chunk8_umt5_probe.mojo).
#
# Tokenizer: the pure-Mojo T5 Unigram tokenizer (tokenizer/t5_tokenizer.mojo)
# over Wan's google/umt5-xxl tokenizer.json (Unigram, <pad>=0, </s>=1, <unk>=3;
# the tokenizer reads unk_id from the JSON and appends EOS=1 == </s>).
#
# PADDING / MASK CONVENTION:
#   diffusers WanPipeline pads to max_length=512 AND passes an attention_mask so
#   pad tokens are invisible, then slices the valid rows and zero-pads the OUTPUT
#   back to 512. `Umt5Encoder.encode_masked` now applies the same key-padding
#   mask inside every encoder layer. We still write the full [1,SEQ,4096] and
#   record pos_len/neg_len so the DiT can zero the trailing rows exactly.
#
# OUTPUT CONTRACT (wan22_conds.safetensors):
#   pos_embed  [1, SEQ, 4096]  BF16   umt5 encode of the positive prompt
#   neg_embed  [1, SEQ, 4096]  BF16   umt5 encode of the negative prompt
#   pos_len    [1]             F32    valid (non-pad) token count of pos
#   neg_len    [1]             F32    valid (non-pad) token count of neg
#
# argv: <prompt.txt|"prompt string"> <neg.txt|"neg string"> <out.safetensors>
#   arg 1/2: if it ends in ".txt" and the file exists it is read as UTF-8, else
#            the argument is used literally as the prompt text.

from std.collections import List
from std.sys import argv
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer
from serenitymojo.tokenizer.tokenizer import _read_utf8_file
from serenitymojo.models.text_encoder.umt5_encoder import Umt5Encoder, Umt5Config


# umt5-xxl BF16 weights (Wan2.2's encoder, blocks.{i}.* schema; gated cos>=0.999).
comptime UMT5_PATH = (
    "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-Mojo/umt5"
)
# Wan's google/umt5-xxl Unigram tokenizer (256300 pieces; <pad>=0,</s>=1,<unk>=3).
comptime TOK_JSON = (
    "/home/alex/.serenity/models/checkpoints/"
    "Wan2.2-TI2V-5B-Mojo/tokenizer.json"
)
comptime SEQ = 512     # Wan2.2 max_sequence_length (diffusers __call__ default).
comptime PAD_ID = 0    # umt5 <pad> id.
comptime EOS_ID = 1    # umt5 </s> id (T5Tokenizer appends this).


def _is_txt_path(s: String) -> Bool:
    """True if `s` ends with the literal ".txt" (2E 74 78 74)."""
    var b = s.as_bytes()
    var n = len(b)
    if n < 4:
        return False
    return (
        Int(b[n - 4]) == 0x2E
        and Int(b[n - 3]) == 0x74
        and Int(b[n - 2]) == 0x78
        and Int(b[n - 1]) == 0x74
    )


def _read_arg_text(a: String) raises -> String:
    """Return the prompt text for argument `a`: read the file if it names a .txt,
    otherwise use the argument literally."""
    if _is_txt_path(a):
        return _read_utf8_file(a)
    return a


def _pad_to_seq(ids_full: List[Int]) -> List[Int]:
    """Right-pad `ids_full` (already EOS-terminated) to SEQ with PAD_ID, or
    truncate to SEQ tokens keeping the final EOS (diffusers truncation=True)."""
    var ids = List[Int](capacity=SEQ)
    var valid = len(ids_full)
    if valid > SEQ:
        for i in range(SEQ - 1):
            ids.append(ids_full[i])
        ids.append(EOS_ID)
    else:
        for i in range(valid):
            ids.append(ids_full[i])
        for _ in range(SEQ - valid):
            ids.append(PAD_ID)
    return ids^


def _valid_count(ids_full: List[Int]) -> Int:
    """Valid (non-pad) token count == min(len(ids_full), SEQ)."""
    var v = len(ids_full)
    if v > SEQ:
        return SEQ
    return v


def _len_tensor(valid: Int, ctx: DeviceContext) raises -> Tensor:
    """[1] F32 tensor holding the valid token count."""
    var vals = List[Float32]()
    vals.append(Float32(valid))
    var sh = List[Int]()
    sh.append(1)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


def main() raises:
    var a = argv()
    if len(a) < 4:
        raise Error(
            'usage: wan22_encode_prompt <prompt.txt|"text"> '
            '<neg.txt|"text"> <out.safetensors>'
        )
    var pos_text = _read_arg_text(String(a[1]))
    var neg_text = _read_arg_text(String(a[2]))
    var out_path = String(a[3])

    var ctx = DeviceContext()
    print("=== Wan2.2 umt5-xxl prompt encode (separate process) ===")

    # ── tokenize (Unigram + trailing EOS) and pad/truncate to SEQ, no mask ─────
    var tok = T5Tokenizer.load(TOK_JSON)
    var pos_full = tok.encode(pos_text)
    var neg_full = tok.encode(neg_text)
    var pos_ids = _pad_to_seq(pos_full)
    var neg_ids = _pad_to_seq(neg_full)
    var pos_valid = _valid_count(pos_full)
    var neg_valid = _valid_count(neg_full)
    print("  pos tokens:", len(pos_full), "-> pad", SEQ, " valid=", pos_valid)
    print("  neg tokens:", len(neg_full), "-> pad", SEQ, " valid=", neg_valid)

    # ── load umt5-xxl encoder ONCE and encode both prompts ─────────────────────
    print("[load] umt5-xxl encoder (242 keys, ~11GB BF16)", UMT5_PATH)
    var enc = Umt5Encoder[SEQ].load(UMT5_PATH, Umt5Config.umt5_xxl(), ctx)
    print("[encode] positive prompt through 24 umt5 layers ...")
    var pos_embed = enc.encode_masked(
        pos_ids, pos_valid, ctx
    )   # [1, SEQ, 4096] BF16
    print("[encode] negative prompt through 24 umt5 layers ...")
    var neg_embed = enc.encode_masked(
        neg_ids, neg_valid, ctx
    )   # [1, SEQ, 4096] BF16

    var ps = pos_embed.shape()
    var ns = neg_embed.shape()
    print(
        "  pos_embed:", ps[0], ps[1], ps[2],
        "dtype.tag", pos_embed.dtype().tag, "nbytes", pos_embed.nbytes(),
    )
    print(
        "  neg_embed:", ns[0], ns[1], ns[2],
        "dtype.tag", neg_embed.dtype().tag, "nbytes", neg_embed.nbytes(),
    )
    if ps[1] != SEQ or ps[2] != 4096:
        raise Error("pos_embed shape wrong (expect [1,512,4096])")
    if ns[1] != SEQ or ns[2] != 4096:
        raise Error("neg_embed shape wrong (expect [1,512,4096])")

    var pos_len = _len_tensor(pos_valid, ctx)
    var neg_len = _len_tensor(neg_valid, ctx)

    # ── write wan22_conds.safetensors ─────────────────────────────────────────
    var names = List[String]()
    names.append(String("pos_embed"))
    names.append(String("neg_embed"))
    names.append(String("pos_len"))
    names.append(String("neg_len"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(pos_embed^))
    tensors.append(ArcPointer(neg_embed^))
    tensors.append(ArcPointer(pos_len^))
    tensors.append(ArcPointer(neg_len^))
    save_safetensors(names, tensors, out_path, ctx)
    print("[done] wrote", out_path)
    print("[done] encoder GPU memory freed on process exit")
