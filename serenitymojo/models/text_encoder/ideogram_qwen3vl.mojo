# models/text_encoder/ideogram_qwen3vl.mojo — Ideogram-4 Qwen3-VL text path.
# Reuses Qwen3Encoder (the text decoder layer is byte-identical to Qwen3VL text:
# input_layernorm/q,k,v,o_proj + q_norm,k_norm/post_attention_layernorm/
# mlp.gate,up,down — verified vs the .serenity checkpoint keys). Differences:
#   - weights are weight-only FP8 (*_proj.weight) -> dequant to BF16 at load,
#   - keys are prefixed language_model.* -> remap to model.*,
#   - config theta = 5e6 (Ideogram), and
#   - conditioning = 13-tap interleaved concat (ref pipeline_ideogram4 _encode_text
#     414-480): stack(taps)->permute(1,2,3,0)->reshape => out[..,f*13+t]=tap_t[..,f].
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.fp8 import load_fp8_dequant
from serenitymojo.ops.tensor_algebra import concat, reshape
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config


def _add(
    st: ShardedSafeTensors,
    mut weights: List[ArcPointer[Tensor]],
    mut n2i: Dict[String, Int],
    dst: String, src: String, ctx: DeviceContext,
) raises:
    var t: Tensor
    if src.endswith("_proj.weight"):
        t = load_fp8_dequant(st, src, ctx)  # F8_E4M3 + per-row scale -> BF16
    else:
        t = Tensor.from_view(st.tensor_view(src), ctx)  # BF16 (norm / embed)
    n2i[dst] = len(weights)
    weights.append(ArcPointer(t^))


def load_ideogram_qwen3vl(dir_or_file: String, ctx: DeviceContext) raises -> Qwen3Encoder:
    var st = ShardedSafeTensors.open(dir_or_file)
    var weights = List[ArcPointer[Tensor]]()
    var n2i = Dict[String, Int]()
    var cfg = Qwen3Config(4096, 36, 32, 8, 128, Float32(1.0e-6), Float64(5000000.0))
    _add(st, weights, n2i, "model.embed_tokens.weight", "language_model.embed_tokens.weight", ctx)
    for i in range(36):
        var ps = String("language_model.layers.") + String(i) + "."
        var pd = String("model.layers.") + String(i) + "."
        _add(st, weights, n2i, pd + "input_layernorm.weight", ps + "input_layernorm.weight", ctx)
        _add(st, weights, n2i, pd + "self_attn.q_proj.weight", ps + "self_attn.q_proj.weight", ctx)
        _add(st, weights, n2i, pd + "self_attn.k_proj.weight", ps + "self_attn.k_proj.weight", ctx)
        _add(st, weights, n2i, pd + "self_attn.v_proj.weight", ps + "self_attn.v_proj.weight", ctx)
        _add(st, weights, n2i, pd + "self_attn.o_proj.weight", ps + "self_attn.o_proj.weight", ctx)
        _add(st, weights, n2i, pd + "self_attn.q_norm.weight", ps + "self_attn.q_norm.weight", ctx)
        _add(st, weights, n2i, pd + "self_attn.k_norm.weight", ps + "self_attn.k_norm.weight", ctx)
        _add(st, weights, n2i, pd + "post_attention_layernorm.weight", ps + "post_attention_layernorm.weight", ctx)
        _add(st, weights, n2i, pd + "mlp.gate_proj.weight", ps + "mlp.gate_proj.weight", ctx)
        _add(st, weights, n2i, pd + "mlp.up_proj.weight", ps + "mlp.up_proj.weight", ctx)
        _add(st, weights, n2i, pd + "mlp.down_proj.weight", ps + "mlp.down_proj.weight", ctx)
    _add(st, weights, n2i, "model.norm.weight", "language_model.norm.weight", ctx)
    return Qwen3Encoder(weights^, n2i^, cfg)


def encode_ideogram_taps(enc: Qwen3Encoder, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
    """13-tap interleaved concat -> [1, L, 4096*13=53248] F32-castable BF16."""
    var states = enc.encode_layer_states(ids, ctx)  # 36 x [1,L,4096]
    var L = states[0][].shape()[1]
    var H = states[0][].shape()[2]
    var s4 = [1, L, H, 1]
    var r0 = reshape(states[0][], s4.copy(), ctx)
    var r1 = reshape(states[3][], s4.copy(), ctx)
    var r2 = reshape(states[6][], s4.copy(), ctx)
    var r3 = reshape(states[9][], s4.copy(), ctx)
    var r4 = reshape(states[12][], s4.copy(), ctx)
    var r5 = reshape(states[15][], s4.copy(), ctx)
    var r6 = reshape(states[18][], s4.copy(), ctx)
    var r7 = reshape(states[21][], s4.copy(), ctx)
    var r8 = reshape(states[24][], s4.copy(), ctx)
    var r9 = reshape(states[27][], s4.copy(), ctx)
    var r10 = reshape(states[30][], s4.copy(), ctx)
    var r11 = reshape(states[33][], s4.copy(), ctx)
    var r12 = reshape(states[35][], s4.copy(), ctx)
    var cat = concat(3, ctx, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12)
    return reshape(cat, [1, L, H * 13], ctx)
