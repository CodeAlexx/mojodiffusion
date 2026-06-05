# serenitymojo/pipeline/anima_text_context.mojo
#
# Anima TEXT -> CONTEXT @512 end-to-end (Chunk C, TRAINING_PLAN_anima_OT.md §C).
#
# Runs the FULL OneTrainer text path (AnimaModel.encode_text) in pure Mojo:
#   1. Qwen3-0.6B encoder -> last_hidden_state [1,512,1024]  (reuse qwen3_encoder)
#   2. zero pad positions  (AnimaModel.py:218: hidden * tokens_mask.unsqueeze(-1))
#   3. net.llm_adapter (6 blocks, F32) -> FROZEN context [1,512,1024]
#   4. cache to safetensors as key `context_cond` (mirrors the captured sidecar
#      schema) so train_anima_ot.mojo consumes a real per-caption context.
#
# TOKENIZER NOTE (deviation, stated per task): the OneTrainer recipe tokenizes
# with HF Qwen2Tokenizer (Qwen3 input ids) + T5TokenizerFast (adapter query ids)
# at max_length=512. Those two tokenizers are NOT ported to Mojo. This binary
# therefore reads the THREE token-id arrays (qwen_ids[512], qwen_mask[512],
# t5_ids[512]) from a sidecar safetensors produced by
#   parity/anima_text_context_tokens.py  (HF tokenizers, dev-time only).
# The Qwen3 ENCODER + adapter compute are 100% Mojo. Run the encoder gate
# (parity/qwen3_parity) for the encoder arm and anima_text_context_parity for
# the adapter arm; together they cover the whole path.
#
# Run (after generating a token sidecar with the python helper):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/pipeline/anima_text_context.mojo -o /tmp/anima_text_context
#   /tmp/anima_text_context <tokens_sidecar.safetensors> <out_context.safetensors>

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import ArcPointer
from std.sys import argv
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Config, Qwen3Encoder
from serenitymojo.models.anima.anima_text_context import (
    AnimaAdapterWeights, anima_llm_adapter_forward, zero_pad_positions_f32,
)

comptime QWEN3_DIR = (
    "/home/alex/.serenity/models/anima/split_files/text_encoders/"
    "qwen_3_06b_base.safetensors"
)
comptime CKPT = (
    "/home/alex/.serenity/models/anima/split_files/diffusion_models/"
    "anima-base-v1.0.safetensors"
)
comptime S_TXT = 512
comptime DIM = 1024


def _read_ids(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Int]:
    """Read a dev-time token sidecar array into host List[Int].

    `from_view_as_f32` is intentional here: these are tokenizer ids, not model
    activations or weights, and the result leaves tensor storage immediately."""
    var t = Tensor.from_view_as_f32(st.tensor_view(name), ctx)
    var host = t.to_host(ctx)
    var out = List[Int]()
    for i in range(len(host)):
        out.append(Int(host[i]))
    return out^


def anima_text_context_from_tokens(
    qwen_ids: List[Int],
    qwen_mask: List[Int],
    t5_ids: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    """Full OT text path -> FROZEN context [1,512,1024] (F32). Forward only."""
    # 1) Qwen3-0.6B encoder -> last_hidden_state (apply final model.norm).
    var enc = Qwen3Encoder.load(QWEN3_DIR, Qwen3Config.qwen3_06b(), ctx)
    var n_layers = enc.config.num_layers
    var pre_norm = enc.encode(qwen_ids, n_layers - 1, ctx)  # [1,512,1024] BF16
    var last_hidden = enc.final_norm(pre_norm, ctx)          # last_hidden_state
    var hidden_f32 = cast_tensor(last_hidden, STDtype.F32, ctx)

    # 2) zero pad positions (AnimaModel.py:218).
    var hidden_zeroed = zero_pad_positions_f32(hidden_f32, qwen_mask, ctx)

    # 3) net.llm_adapter -> context.
    var wts = AnimaAdapterWeights.load_checkpoint(CKPT, ctx)
    return anima_llm_adapter_forward(t5_ids, hidden_zeroed, qwen_mask, wts, ctx)


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error(
            "usage: anima_text_context <tokens_sidecar.safetensors>"
            " <out_context.safetensors>"
        )
    var tokens_path = String(args[1])
    var out_path = String(args[2])
    var ctx = DeviceContext()

    print("[anima-text-context] reading token sidecar:", tokens_path)
    var st = ShardedSafeTensors.open(tokens_path)
    var qwen_ids = _read_ids(st, String("qwen_input_ids"), ctx)
    var qwen_mask = _read_ids(st, String("qwen_attention_mask"), ctx)
    var t5_ids = _read_ids(st, String("t5_input_ids"), ctx)
    if len(qwen_ids) != S_TXT or len(qwen_mask) != S_TXT or len(t5_ids) != S_TXT:
        raise Error("token sidecar arrays must each be length 512")

    print("[anima-text-context] running Qwen3-0.6B + zero-pad + adapter...")
    var context = anima_text_context_from_tokens(qwen_ids, qwen_mask, t5_ids, ctx)

    var cs = context.shape()
    print("[anima-text-context] context shape = [",
          cs[0], ",", cs[1], ",", cs[2], "]")
    if len(cs) != 3 or cs[0] != 1 or cs[1] != S_TXT or cs[2] != DIM:
        raise Error("context shape != [1,512,1024]")

    # finiteness + magnitude report
    var host = context.to_host(ctx)
    var sabs = Float64(0.0)
    var allfin = True
    for i in range(len(host)):
        var v = Float64(host[i])
        var av = v if v >= 0.0 else -v
        sabs += av
        if not (v == v):  # NaN
            allfin = False
    print("[anima-text-context] mean_abs =", sabs / Float64(len(host)),
          " finite =", allfin)
    if not allfin:
        raise Error("context contains non-finite values")

    # cache as context_cond (mirror sidecar schema)
    var names = List[String]()
    names.append(String("context_cond"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(context^))
    save_safetensors(names, tensors, out_path, ctx)
    print("[anima-text-context] wrote context_cond [1,512,1024] ->", out_path)
