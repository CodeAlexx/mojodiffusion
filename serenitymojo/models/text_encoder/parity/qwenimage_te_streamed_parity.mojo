# parity/qwenimage_te_streamed_parity.mojo — dump side of the streamed-TE gate.
#
# Gates models/text_encoder/qwenimage_qwen25vl_streamed.mojo (the 16GB-fit
# layer-streamed Qwen2.5-VL encode) against the HF torch oracle
# (qwenimage_te_streamed_oracle.py): cos >= 0.999 over the REAL (non-pad) rows
# of the final-normed last hidden state, per the repo parity standard.
#
# WHY a torch oracle and not the resident Mojo encoder: the resident
# Qwen25VLEncoder.load needs ~17.4 GB (measured child preflight) — it cannot
# load on this 16 GB card at all, so the reference is the SerenityTrainer venv's
# CUDA/CPU torch Qwen2.5-VL text tower last_hidden_state on the SAME padded
# ids (the oracle handles masking; rows >= real_len are excluded — both stacks
# leave pad rows as garbage and the DiT masks them via real_len).
#
# Flow (run under a systemd-run memory scope like all GPU builds):
#   1. pixi run build-qwenimage-te-streamed-parity   (binary in output/bin)
#   2. output/bin/qwenimage_te_streamed_parity        (writes /tmp dumps)
#   3. /home/alex/SerenityTrainer/venv/bin/python \
#        serenitymojo/models/text_encoder/parity/qwenimage_te_streamed_oracle.py
#      → prints cos + PASS/FAIL.
#
# Tokenization here is a byte-copy of qwenimage_sample_cli's contract
# (_qwen_template + pad-to-N_ENC with PAD 151643); it is NOT imported to keep
# this smoke's build light (importing the CLI pulls the full DiT/VAE stack).

from max.gpu.host import DeviceContext
from std.memory import alloc

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_pwrite, sys_close, O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwenimage_qwen25vl_streamed import (
    encode_qwen25vl_final_streamed,
)

# ── qwenimage encode contract (byte-copy of qwenimage_sample_cli comptime) ────
comptime QWENIMAGE_DIR = "/home/alex/.serenity/models/checkpoints/qwen-image-2512"
comptime TEXT_ENCODER_DIR = QWENIMAGE_DIR + "/text_encoder"
comptime TOK_JSON = QWENIMAGE_DIR + "/tokenizer/tokenizer.json"
comptime PAD_ID = 151643
comptime DROP_IDX = 34
comptime N_TXT_KEPT = 512
comptime N_ENC = N_TXT_KEPT + DROP_IDX   # 546
comptime EXTRACT_LAYER = 27
comptime HIDDEN = 3584

comptime PROMPT = "A photorealistic red fox sitting in a snowy forest clearing at golden hour, highly detailed fur"

comptime IDS_PATH = "/tmp/qwenimage_te_parity.ids.txt"
comptime HID_PATH = "/tmp/qwenimage_te_parity.mojo_f32.bin"


# Byte-copy of qwenimage_sample_cli._qwen_template.
def _qwen_template(prompt: String) -> String:
    return (
        String("<|im_start|>system\nDescribe the image by detailing the color,"
        " shape, size, texture, quantity, text, spatial relationships of the"
        " objects and background:<|im_end|>\n<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def _write_bytes(path: String, p: BytePtr, n: Int) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("parity: open failed: ") + path)
    var w = sys_pwrite(fd, p, n, 0)
    _ = sys_close(fd)
    if w != n:
        raise Error(String("parity: short write: ") + path)


def _write_text(path: String, s: String) raises:
    var n = s.byte_length()
    var buf = alloc[UInt8](n)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    _write_bytes(path, BytePtr(unsafe_from_address=Int(buf)), n)
    buf.free()


def _write_f32(path: String, vals: List[Float32]) raises:
    var n = len(vals)
    var buf = alloc[Float32](n)
    for i in range(n):
        buf[i] = vals[i]
    _write_bytes(path, BytePtr(unsafe_from_address=Int(buf)), n * 4)
    buf.free()


def main() raises:
    var ctx = DeviceContext()
    print("=== qwenimage streamed-TE parity dump ===")
    print("  prompt:", PROMPT)

    # Tokenize + pad exactly as qwenimage_sample_cli._tokenize_for_encoder.
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var ids_full = tok.encode(_qwen_template(String(PROMPT)))
    var real_len = len(ids_full)
    if real_len <= DROP_IDX or real_len > N_ENC:
        raise Error("parity: prompt token count out of contract range")
    var ids = List[Int](capacity=N_ENC)
    for i in range(real_len):
        ids.append(ids_full[i])
    for _ in range(N_ENC - real_len):
        ids.append(PAD_ID)
    print("  tokens:", real_len, "of", N_ENC, "(kept", real_len - DROP_IDX, ")")

    # Streamed encode: layers 0..27 + final model.norm → [1, 546, 3584] BF16.
    var ids_list = List[List[Int]]()
    ids_list.append(ids.copy())
    var outs = encode_qwen25vl_final_streamed(
        String(TEXT_ENCODER_DIR), ids_list, EXTRACT_LAYER, ctx
    )
    var full = cast_tensor(outs[0][], STDtype.F32, ctx)
    var host = full.to_host(ctx)
    if len(host) != N_ENC * HIDDEN:
        raise Error("parity: unexpected hidden numel")

    # Dumps for the torch oracle.
    var ids_txt = String("")
    for i in range(N_ENC):
        ids_txt += String(ids[i]) + String("\n")
    _write_text(String(IDS_PATH), ids_txt)
    _write_f32(String(HID_PATH), host)
    print("  wrote:", IDS_PATH)
    print("  wrote:", HID_PATH, "(", N_ENC, "x", HIDDEN, "f32 LE )")
    print("  real_len:", real_len)
    print("DONE — run qwenimage_te_streamed_oracle.py to compare")
