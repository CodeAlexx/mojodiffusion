# serenitymojo/serve/parity/klein_worker_encode_gate.mojo
#
# GATES: the klein worker's runtime Qwen3 text encode against HF Qwen3-8B ground
#   truth. Worker path — serve/klein_runtime_backend.mojo::_encode_text_pair
#   (variant 9b = Qwen3-8B). This gate replicates that path EXACTLY (the
#   _klein_template chat wrap + 512-token pad @ PAD_ID=151643 +
#   Qwen3Encoder.encode_klein stacking layers [8,17,26]) and DUMPS the joint
#   [1,512,12288] conditioning + the token ids for the python reference
#   (serve/parity/ref/ref_klein.py) to compare on BYTE-IDENTICAL ids.
#
#   joint[1,512,12288] = cat(states[8], states[17], states[26]) PRE-final-norm.
#   HF mapping (embedding-indexed): states[8]=hs[9], states[17]=hs[18], states[26]=hs[27].
#
# VERDICT (MJ-1052): PASS — real-token conditioning matches HF at all three tapped
#   layers. PASS BAR (checked by ref_klein.py, NOT in-Mojo): min real-row cos >= 0.99
#   over real rows [0:22]. MEASURED: states[8] 0.999896, states[17] 0.997114,
#   states[26] 0.999423.
#
# PAD-ROW NOTE (benign): the worker builds a causal + KEY-PADDING mask at real_len
#   (qwen3_encoder.mojo:740; Rust qwen3_encoder.rs:476 convention). HF runs with NO
#   attention_mask (pure causal), so HF pad-query rows attend to pad keys while the
#   worker masks them -> pad rows [22:512] diverge and drag the OVERALL cos below
#   0.999. Real rows are unaffected. The worker's key-padding is the RUST klein
#   convention the Klein DiT was trained with, so this is correct, not a defect.
#
# RE-RUN (GPU + Qwen3-8B + python ref): see docs/MOJO_MODULES.md serve/parity section.
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.io.safetensors_writer import save_safetensors

comptime QWEN8_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
comptime PAD_ID = 151643
comptime SEQ = 512
comptime PROMPT = "a photorealistic red fox sitting in autumn leaves"
comptime OUT = "/home/alex/mojodiffusion/output/checks/phase4a/klein_worker_cond.safetensors"


def _klein_template(prompt: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _tokenize_512(tok: Qwen3Tokenizer, prompt: String) raises -> Tuple[List[Int], Int]:
    var ids_full = tok.encode(_klein_template(prompt))
    if len(ids_full) > SEQ:
        raise Error("klein-gate: prompt too long for 512")
    var ids = List[Int](capacity=SEQ)
    for i in range(len(ids_full)):
        ids.append(ids_full[i])
    for _ in range(SEQ - len(ids_full)):
        ids.append(PAD_ID)
    return (ids^, len(ids_full))


def _ids_tensor(ids: List[Int], ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for i in range(len(ids)):
        h.append(Float32(ids[i]))
    return Tensor.from_host(h, [SEQ], STDtype.F32, ctx)


def main() raises:
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(String(QWEN8_DIR) + "/tokenizer.json")
    var enc = Qwen3Encoder.load(String(QWEN8_DIR), Qwen3Config.klein_9b(), ctx)
    var tk = _tokenize_512(tok, String(PROMPT))
    var ids = tk[0].copy()
    print("[klein-gate] real tokens =", tk[1], "padded_to", SEQ)
    var joint = enc.encode_klein(ids, ctx)   # [1,512,12288]
    var joint_c = joint.clone(ctx)

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String("joint"))
    tensors.append(ArcPointer(joint_c^))
    names.append(String("ids"))
    tensors.append(ArcPointer(_ids_tensor(ids, ctx)))
    save_safetensors(names, tensors, String(OUT), ctx)
    print("[klein-gate] wrote", OUT)
