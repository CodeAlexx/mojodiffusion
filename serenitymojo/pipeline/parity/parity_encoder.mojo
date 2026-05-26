# parity_encoder.mojo — STAGE 2 parity: my Qwen3 encoder cap_feats vs diffusers.
# Compares my encode(PROMPT)/encode("") (template + layer-34 + slice) against the
# oracle's cond.bin / uncond.bin (diffusers ZImagePipeline.encode_prompt outputs).
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#        serenitymojo/pipeline/parity/parity_encoder.mojo
from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.ops.tensor_algebra import reshape, slice
from serenitymojo.parity import ParityHarness
comptime ZROOT = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
)
comptime TEXT_ENCODER = ZROOT + "/text_encoder"
comptime TOK_JSON = ZROOT + "/tokenizer/tokenizer.json"
comptime PROMPT = (
    "Masterpiece, best quality, high resolution, detailed, very detailed,"
    " intricate detailed, (hourglass:1.2), gyroid, gyroid lattice, gyroid fill,"
    " Pearlum, filigree brass, pearls, beautiful photo of a sculpture made of"
    " fluorite mineral, flu0rite, translucent gems, geode, made out of transistors,"
    " LED, wires, eh1, ethereon, filigree brass, detailed background, complex"
    " background, dynamic composition, cinematic scene, perfect composition, matte"
    " finish, 85mm lens, f/1.8, layered textures, dreamy, nostalgic, perfect"
    " composition, intricate detail, depth of field, (bokeh:0.5), professional 4k"
    " highly detailed, Canon 5d mark 4, moody lighting,"
)
comptime PD = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity"
comptime HIDDEN = 2560
comptime ENC_SEQ = 512
comptime PAD_ID = 151643
comptime EXTRACT_LAYER = 34


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var count = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(count):
        out.append(fp[i])
    buf.free()
    return out^


def _encode(tok: Qwen3Tokenizer, enc: Qwen3Encoder, prompt: String, caplen: Int, ctx: DeviceContext) raises -> Tensor:
    var templated = String("<|im_start|>user\n") + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    var ids_full = tok.encode(templated)
    if len(ids_full) != caplen:
        raise Error(String("token count ") + String(len(ids_full)) + " != " + String(caplen))
    var ids = List[Int](capacity=ENC_SEQ)
    for i in range(caplen):
        ids.append(ids_full[i])
    for _ in range(ENC_SEQ - caplen):
        ids.append(PAD_ID)
    var cf = enc.encode(ids, EXTRACT_LAYER, ctx)
    var cf_real = slice(cf, 1, 0, caplen, ctx)
    return reshape(cf_real, [caplen, HIDDEN], ctx)


def main() raises:
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(TOK_JSON)
    var enc = Qwen3Encoder.load(TEXT_ENCODER, Qwen3Config.zimage(), ctx)
    var harness = ParityHarness(0.99)

    print("=== STAGE 2: encoder cap_feats parity vs diffusers ===")
    var cond_ref = _read_f32_bin(String(PD) + "/cond.bin")
    var cond = _encode(tok, enc, PROMPT, len(cond_ref) // HIDDEN, ctx)
    print("  cond (", len(cond_ref) // HIDDEN, "x", HIDDEN, "):", harness.compare(cond, cond_ref, ctx))

    var unc_ref = _read_f32_bin(String(PD) + "/uncond.bin")
    var unc = _encode(tok, enc, String(""), len(unc_ref) // HIDDEN, ctx)
    print("  uncond (", len(unc_ref) // HIDDEN, "x", HIDDEN, "):", harness.compare(unc, unc_ref, ctx))
