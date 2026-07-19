# serenitymojo/serve/parity/sdxl_worker_encode_gate.mojo
#
# GATES: the SDXL worker's runtime CLIP text encode against an HF reference.
#   Worker path — serve/sdxl_backend.mojo::_encode_one -> ClipEncoder.encode_sdxl
#   (clip_l + clip_g). This gate runs the EXACT encoder call the worker uses, for
#   one fixed prompt, and DUMPS the produced hidden states + the token ids fed to
#   the encoder so the python reference (serve/parity/ref/ref_sdxl_clip.py) can be
#   compared on BYTE-IDENTICAL ids.
#
# WHAT CHANGED (MJ-1061): the gate calls encode_sdxl(..., penultimate_context=True)
#   — the diffusers/SDXL convention that returns hidden_states[-2] (penultimate,
#   NO final LN) as the cross-attn context, with a causal-only mask. This is the
#   fix for the MJ-1052 bug where the worker returned the POST-final-LN
#   last_hidden_state instead (measured 0.727/0.680 min real-row cos off the HF
#   penultimate reference — a FAIL). SD3 already taps the penultimate correctly
#   (see sd3_worker_encode_gate.mojo), which independently corroborated the bug.
#
# PASS BAR (checked by ref_sdxl_clip.py, NOT in-Mojo — this file only produces the
#   dump): worker context vs HF hidden_states[-2] min real-row cos >= 0.99.
#   MEASURED post-fix: min real-row cos 0.999797 (CLIP-L/CLIP-G); pooled cos 1.000.
#
# RE-RUN (GPU + models + python ref): see docs/MOJO_MODULES.md serve/parity section.
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.clip_tokenizer import ClipTokenizer
from serenitymojo.models.text_encoder.clip_encoder import ClipEncoder, ClipConfig
from serenitymojo.io.safetensors_writer import save_safetensors

comptime CLIP_L_PATH = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
comptime CLIP_G_PATH = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors"
comptime CLIP_L_TOK = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"
comptime CLIP_G_TOK = "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json"
comptime CLIP_LEN = 77
comptime EOS_ID = 49407
comptime PROMPT = "a photorealistic red fox sitting in autumn leaves"
comptime OUT = "/home/alex/mojodiffusion/output/checks/phase4a/sdxl_worker_clip.safetensors"


def _fit(var ids: List[Int]) -> List[Int]:
    # pad/truncate to 77 with EOS — matches encode_sdxl's internal fit and the
    # backend's _fit_clip_ids, so the dumped ids == what encode_sdxl consumes.
    if len(ids) > CLIP_LEN:
        var t = List[Int]()
        for i in range(CLIP_LEN):
            t.append(ids[i])
        t[CLIP_LEN - 1] = EOS_ID
        return t^
    while len(ids) < CLIP_LEN:
        ids.append(EOS_ID)
    return ids^


def _ids_tensor(ids: List[Int], ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for i in range(len(ids)):
        h.append(Float32(ids[i]))
    return Tensor.from_host(h, [CLIP_LEN], STDtype.F32, ctx)


def main() raises:
    var ctx = DeviceContext()
    var tok_l = ClipTokenizer(String(CLIP_L_TOK))
    var tok_g = ClipTokenizer(String(CLIP_G_TOK))
    var l_ids = _fit(tok_l.encode(String(PROMPT)))
    var g_ids = _fit(tok_g.encode(String(PROMPT)))
    print("[sdxl-gate] l real tokens then pad; first non-eos count logged by encoder")

    # penultimate_context=True is the MJ-1061 fix: return hidden_states[-2].
    var clip_l = ClipEncoder.load(String(CLIP_L_PATH), ClipConfig.clip_l(), ctx)
    var l_out = clip_l.encode_sdxl[CLIP_LEN](l_ids.copy(), ctx, penultimate_context=True)
    var l_hidden = l_out[0].clone(ctx)
    var l_pooled = l_out[1].clone(ctx)

    var clip_g = ClipEncoder.load(String(CLIP_G_PATH), ClipConfig.clip_g(), ctx)
    var g_out = clip_g.encode_sdxl[CLIP_LEN](g_ids.copy(), ctx, penultimate_context=True)
    var g_hidden = g_out[0].clone(ctx)
    var g_pooled = g_out[1].clone(ctx)

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String("l_ids"))
    tensors.append(ArcPointer(_ids_tensor(l_ids, ctx)))
    names.append(String("g_ids"))
    tensors.append(ArcPointer(_ids_tensor(g_ids, ctx)))
    names.append(String("l_hidden"))
    tensors.append(ArcPointer(l_hidden^))
    names.append(String("l_pooled"))
    tensors.append(ArcPointer(l_pooled^))
    names.append(String("g_hidden"))
    tensors.append(ArcPointer(g_hidden^))
    names.append(String("g_pooled"))
    tensors.append(ArcPointer(g_pooled^))
    save_safetensors(names, tensors, String(OUT), ctx)
    print("[sdxl-gate] wrote", OUT)
