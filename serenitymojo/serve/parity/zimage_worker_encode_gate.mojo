# serenitymojo/serve/parity/zimage_worker_encode_gate.mojo
#
# GATES: the zimage worker's runtime Qwen3-4B text encode against HF Qwen3-4B
#   ground truth. Worker path — pipeline/zimage_generate.mojo::_encode_text_fixed.
#   This gate replicates that path EXACTLY (template
#   "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n" — NO think block,
#   pad-to-512 @ 151643, Qwen3Encoder.load(max_layer=34).encode(ids,34)
#   PRE-final-norm, slice first 256) and DUMPS the cap [256,2560] + the token ids
#   for the python reference (serve/parity/ref/ref_zimage.py) on BYTE-IDENTICAL ids.
#
#   EXTRACT_LAYER=34 (Mojo states[34]) == HF hidden_states[35] (penultimate).
#
# VERDICT (MJ-1052): PASS — worker layer-34 penultimate extraction matches HF.
#   PASS BAR (checked by ref_zimage.py, NOT in-Mojo): min real-row cos >= 0.99 over
#   real rows [0:18]. MEASURED: cap vs hs[35] = 0.999752. SANITY: cap vs hs[34]
#   (one layer earlier) = 0.845 -> layer index CORRECT; cap vs hs[36] (post-final-
#   norm) = 0.240 -> pre-norm CORRECT.
#
# PAD-ROW NOTE (benign): overall 0.909 is dragged only by pad rows [18:256] via the
#   worker's key-padding mask at real_len (same convention as klein; HF applies no
#   pad mask). Real rows unaffected.
#
# RE-RUN (GPU + Qwen3-4B + python ref): see docs/MOJO_MODULES.md serve/parity section.
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.ops.tensor_algebra import reshape, slice as t_slice
from serenitymojo.io.safetensors_writer import save_safetensors

comptime ZROOT = "/home/alex/.serenity/models/zimage_base"
comptime TEXT_ENCODER = ZROOT + "/text_encoder"
comptime TOK_JSON = ZROOT + "/tokenizer/tokenizer.json"
comptime HIDDEN = 2560
comptime ENC_SEQ = 512
comptime CAPLEN_MAX = 256
comptime PAD_ID = 151643
comptime EXTRACT_LAYER = 34
comptime PROMPT = "a photorealistic red fox sitting in autumn leaves"
comptime OUT = "/home/alex/mojodiffusion/output/checks/phase4a/zimage_worker_cond.safetensors"


def _ids_tensor(ids: List[Int], n: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for i in range(len(ids)):
        h.append(Float32(ids[i]))
    return Tensor.from_host(h, [n], STDtype.F32, ctx)


def main() raises:
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var enc = Qwen3Encoder.load(String(TEXT_ENCODER), Qwen3Config.zimage(), ctx, max_layer=EXTRACT_LAYER)

    var templated = (
        String("<|im_start|>user\n") + String(PROMPT) + "<|im_end|>\n<|im_start|>assistant\n"
    )
    var ids_full = tok.encode(templated)
    var real_caplen = len(ids_full)
    if real_caplen > CAPLEN_MAX:
        real_caplen = CAPLEN_MAX
    var ids = List[Int](capacity=ENC_SEQ)
    for i in range(len(ids_full)):
        ids.append(ids_full[i])
    for _ in range(ENC_SEQ - len(ids_full)):
        ids.append(PAD_ID)
    print("[zimage-gate] real_caplen =", real_caplen, "enc_seq", ENC_SEQ)

    var cf = enc.encode(ids, EXTRACT_LAYER, ctx)          # [1,512,2560] PRE-final-norm
    var cf_fixed = t_slice(cf, 1, 0, CAPLEN_MAX, ctx)     # [1,256,2560]
    var rank2 = reshape(cf_fixed, [CAPLEN_MAX, HIDDEN], ctx)  # [256,2560]

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String("cap"))
    tensors.append(ArcPointer(rank2^))
    names.append(String("ids"))
    tensors.append(ArcPointer(_ids_tensor(ids, ENC_SEQ, ctx)))
    save_safetensors(names, tensors, String(OUT), ctx)
    print("[zimage-gate] wrote", OUT)
