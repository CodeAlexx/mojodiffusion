# parity/ideogram4_te_streamed_parity.mojo — gate the 16GB layer-streamed
# Ideogram-4 TE (encode_ideogram_taps_streamed) against the RESIDENT encoder's
# output preserved in the giger trainer cache (produced by ideogram4_prepare's
# load_ideogram_qwen3vl + encode_ideogram_taps on the 24GB card).
#
# Oracle mapping (verified 2026-07-14): cache llm.2 has text_len.2 == 237, and
# /home/alex/datasets/gigerver3/3.txt is the ONLY gigerver3 caption whose
# chat-templated tokenization is 237 tokens -> llm.2 == resident 13-tap encode
# of caption 3.txt (rows [0,237); pad rows [237,256) are zeroed in the cache).
#
# Gate: cos >= 0.999 over the 237 real rows [1,237,53248].
#
# BUILD+RUN:
#   cd /home/alex/mojodiffusion && pixi run mojo build --optimization-level 2 \
#     -I . -I /home/alex/MOJO-libs -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/text_encoder/parity/ideogram4_te_streamed_parity.mojo \
#     -o /tmp/i4_te_streamed_parity
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib /tmp/i4_te_streamed_parity

from std.gpu.host import DeviceContext
from std.memory import alloc

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, BytePtr, O_RDONLY
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.ideogram_qwen3vl_streamed import (
    encode_ideogram_taps_streamed,
)

comptime TE = "/home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors"
comptime TOK_JSON = "/home/alex/.serenity/models/ideogram-4-fp8/tokenizer/tokenizer.json"
comptime CACHE = "/home/alex/trainings/ideogram4_giger_cache/cache.safetensors"
comptime CAPTION = "/home/alex/datasets/gigerver3/3.txt"
comptime PAD_ID = 151643
comptime NT = 256
comptime REAL_LEN = 237


def _read_text_file(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("te_streamed_parity: file not found: ") + path)
    var bytes = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var nread = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if nread < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("te_streamed_parity: read error: ") + path)
        if nread == 0:
            break
        for i in range(nread):
            bytes.append(buf[i])
        offset += nread
        if nread < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    while len(bytes) > 0 and (bytes[len(bytes) - 1] == 10 or bytes[len(bytes) - 1] == 13):
        _ = bytes.pop()
    return String(unsafe_from_utf8=bytes)


def main() raises:
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var rendered = (
        String("<|im_start|>user\n") + _read_text_file(String(CAPTION))
        + String("<|im_end|>\n<|im_start|>assistant\n")
    )
    var ids = tok.encode(rendered)
    print("[te-parity] caption tokens =", len(ids), " (oracle text_len.2 =", REAL_LEN, ")")
    if len(ids) != REAL_LEN:
        raise Error("te_streamed_parity: token count mismatch — oracle mapping broken")
    for _ in range(NT - len(ids)):
        ids.append(PAD_ID)

    var ids_list = List[List[Int]]()
    ids_list.append(ids^)
    var feats = encode_ideogram_taps_streamed(String(TE), ids_list^, ctx)
    var mine = slice(feats[0][], 1, 0, REAL_LEN, ctx)         # [1,237,53248] BF16

    var cache = ShardedSafeTensors.open(String(CACHE))
    var oracle_full = Tensor.from_view(cache.tensor_view("llm.2"), ctx)  # [1,256,53248] BF16
    var oracle = slice(oracle_full, 1, 0, REAL_LEN, ctx)
    var oracle_host = cast_tensor(oracle, STDtype.F32, ctx).to_host(ctx)

    var res = ParityHarness(0.999).compare(mine, oracle_host, ctx)
    print("[te-parity] streamed-vs-resident 13-tap:", res)
