# serenitymojo/serve/parity/sd3_worker_encode_gate.mojo
#
# GATES: the SD3 worker's runtime triple-encoder conditioning against an HF
#   reference. Worker path — serve/sd3_backend.mojo::_assemble_one (the EXACT
#   function the SD3 worker uses). This gate calls it on one fixed prompt and
#   DUMPS the assembled context [1,410,4096] + pooled [1,2048] plus the token ids
#   fed to each sub-encoder, so the python reference can compare each sub-encoder
#   on BYTE-IDENTICAL ids: CLIP-L penultimate | CLIP-G penultimate | T5-XXL.
#
#   context[1,410,4096] = cat( pad(CLIP-L penult[-2],4096), pad(CLIP-G penult[-2],
#                              4096), T5-XXL last_hidden[256,4096] )
#   pooled[1,2048]      = cat( CLIP-L pooler(768), CLIP-G text_projection(1280) )
#
# VERDICT (MJ-1052): PASS — all three sub-encoders faithful. This gate also
#   independently CONFIRMS SD3 taps the CLIP penultimate hidden_states[-2] (the
#   correct SDXL/SD3 convention) — exactly the tensor the SDXL worker failed to
#   use before MJ-1061 (see sdxl_worker_encode_gate.mojo).
#
# PASS BAR (checked by the python refs, NOT in-Mojo — this file only dumps):
#   CLIP real rows [0:11], T5 real rows [0:14]; min real-row cos >= 0.99.
#   MEASURED: CLIP-L penult 0.999997, CLIP-G penult 0.999792, CLIP-L pooled
#   0.999998, CLIP-G pooled(proj) 0.999975, T5-XXL 0.999488 (vs fp32 ground truth).
#   T5 DTYPE NOTE: compare against fp32 T5, NOT bf16 — HF's OWN bf16-vs-fp32 T5-XXL
#   noise floor on this input is 0.984 (24 bf16 layers), BELOW the bar; the worker
#   (0.999488) is actually closer to fp32 than HF-bf16 is. See ref_sd3_t5_noise.py.
#
# RE-RUN (GPU + models + python refs): see docs/MOJO_MODULES.md serve/parity section.
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tokenizer.clip_tokenizer import ClipTokenizer
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer
from serenitymojo.models.text_encoder.clip_encoder import ClipEncoder, ClipConfig
from serenitymojo.models.text_encoder.t5_encoder import T5Encoder, T5Config
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.serve.sd3_backend import _assemble_one, _fit_clip_ids, _fit_t5_ids

comptime CLIP_L_PATH = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
comptime CLIP_G_PATH = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors"
comptime T5_PATH = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors"
comptime CLIP_L_TOK = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"
comptime CLIP_G_TOK = "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json"
comptime T5_TOK = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json"
comptime T5_LEN = 256
comptime PROMPT = "a photorealistic red fox sitting in autumn leaves"
comptime OUT = "/home/alex/mojodiffusion/output/checks/phase4a/sd3_worker_cond.safetensors"


def _ids_tensor(ids: List[Int], n: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for i in range(len(ids)):
        h.append(Float32(ids[i]))
    return Tensor.from_host(h, [n], STDtype.F32, ctx)


def main() raises:
    var ctx = DeviceContext()
    var clip_l = ClipEncoder.load(String(CLIP_L_PATH), ClipConfig.clip_l(), ctx)
    var clip_g = ClipEncoder.load(String(CLIP_G_PATH), ClipConfig.clip_g(), ctx)
    var g_st = ShardedSafeTensors.open(String(CLIP_G_PATH))
    var text_proj_g = Tensor.from_view(g_st.tensor_view(String("text_projection.weight")), ctx)
    var t5 = T5Encoder[T5_LEN].load(String(T5_PATH), T5Config.t5_xxl(), ctx)
    var clip_l_tok = ClipTokenizer(String(CLIP_L_TOK))
    var clip_g_tok = ClipTokenizer(String(CLIP_G_TOK))
    var t5_tok = T5Tokenizer(String(T5_TOK))

    var out = _assemble_one(
        String(PROMPT), clip_l, clip_g, text_proj_g, t5,
        clip_l_tok, clip_g_tok, t5_tok, ctx,
    )
    var context = out[0].clone(ctx)   # [1,410,4096]
    var pooled = out[1].clone(ctx)    # [1,2048]

    var l_ids = _fit_clip_ids(clip_l_tok.encode(String(PROMPT)))
    var g_ids = _fit_clip_ids(clip_g_tok.encode(String(PROMPT)))
    var t5_ids = _fit_t5_ids(t5_tok.encode(String(PROMPT)))

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String("context"))
    tensors.append(ArcPointer(context^))
    names.append(String("pooled"))
    tensors.append(ArcPointer(pooled^))
    names.append(String("l_ids"))
    tensors.append(ArcPointer(_ids_tensor(l_ids, 77, ctx)))
    names.append(String("g_ids"))
    tensors.append(ArcPointer(_ids_tensor(g_ids, 77, ctx)))
    names.append(String("t5_ids"))
    tensors.append(ArcPointer(_ids_tensor(t5_ids, T5_LEN, ctx)))
    save_safetensors(names, tensors, String(OUT), ctx)
    print("[sd3-gate] wrote", OUT)
