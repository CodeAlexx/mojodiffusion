# models/text_encoder/boogu_qwen3vl.mojo — Boogu-Image Qwen3-VL TEXT path (LM only).
#
# Text-only T2I: the VISION tower is NOT used. Reuses Qwen3Encoder (the Qwen3-VL
# text decoder layer is byte-identical to the Qwen3 text encoder: input_layernorm
# / q,k,v,o_proj + q_norm,k_norm / post_attention_layernorm / mlp.gate,up,down).
# Deltas vs the Ideogram loader (ideogram_qwen3vl.mojo, the template):
#   - keys are prefixed `model.language_model.*` (one extra `model.` vs Ideogram's
#     bare `language_model.*`)  -> remap to the `model.*` keys Qwen3Encoder wants,
#   - dtype is BF16 (NOT fp8): load every tensor via Tensor.from_view (raw H2D),
#   - config theta = 5e6 (Boogu mllm/config.json text_config.rope_theta), and
#   - output = the oracle's hidden_states[-1] (37 hidden states = embeddings + 36
#     layers). We expose BOTH the RAW last-layer hidden (pre-final-norm) and the
#     final_norm'd hidden; the orchestrator gates which matches.
#
# Config truth (mllm/config.json text_config, Qwen3VLForConditionalGeneration):
#   hidden 4096, layers 36, heads 32, kv_heads 8, head_dim 128, eps 1e-6,
#   vocab 151936, rope_theta 5e6, intermediate 12288, bf16.
#
# FLAG (do not mask): text_config.rope_scaling = {mrope_interleaved: true,
#   mrope_section: [24,20,20]}. The Qwen3-VL text RoPE is mROPE-interleaved, NOT
#   the plain half-split RoPE Qwen3Encoder implements. For text-only inputs with
#   a single (1D) position sequence the three mrope sections collapse to identical
#   per-position angles, so plain RoPE is the standard text-only reduction — but
#   this is the most likely source of any residual parity gap. Surfaced here, not
#   silently absorbed.
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config


# Boogu pad/bos id (Qwen3 pad_token == bos 151643). Not present in T2I prompt
# tokens (which use 151644/151645), so padding with it lets encode_layer_states'
# pad auto-detect recover real_len.
comptime _BOOGU_PAD_ID = 151643
# SDPA comptime seq wall: the encoder is causal and _sdpa_dispatch supports
# {8,16,32,64,128,256,...}. The T2I COND prompt is L=45; the CFG UNCOND prompt
# (SYSTEM_PROMPT_DROP + empty user) is L=66, which exceeds 64. Pad both to 128
# (a supported SDPA case) and mask with real_len = the true L (encode_layer_states
# recovers real_len from the 151643 padding). Padding to 128 vs 64 is numerically
# identical for the real tokens under the causal mask.
comptime _BOOGU_PAD_SEQ = 128


def _add(
    st: ShardedSafeTensors,
    mut weights: List[ArcPointer[Tensor]],
    mut n2i: Dict[String, Int],
    dst: String, src: String, ctx: DeviceContext,
) raises:
    """Load one BF16 tensor `src` (raw H2D, dtype preserved) and register it
    under the remapped name `dst` that Qwen3Encoder._layer/_embed/_w request."""
    var t = Tensor.from_view(st.tensor_view(src), ctx)
    n2i[dst] = len(weights)
    weights.append(ArcPointer(t^))


def load_boogu_qwen3vl(mllm_dir: String, ctx: DeviceContext) raises -> Qwen3Encoder:
    """Load the Boogu mllm Qwen3-VL language-model (text) stack into a
    Qwen3Encoder. Remaps `model.language_model.*` -> `model.*` and loads BF16.
    The VISION tower (`model.visual.*`) and `lm_head.weight` are NOT loaded
    (text-only T2I, encoder forward only)."""
    var st = ShardedSafeTensors.open(mllm_dir)
    var weights = List[ArcPointer[Tensor]]()
    var n2i = Dict[String, Int]()
    var cfg = Qwen3Config(4096, 36, 32, 8, 128, Float32(1.0e-6), Float64(5000000.0))
    _add(st, weights, n2i,
         "model.embed_tokens.weight", "model.language_model.embed_tokens.weight", ctx)
    for i in range(36):
        var ps = String("model.language_model.layers.") + String(i) + "."
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
    _add(st, weights, n2i, "model.norm.weight", "model.language_model.norm.weight", ctx)
    return Qwen3Encoder(weights^, n2i^, cfg)


def _pad_ids(ids: List[Int]) raises -> List[Int]:
    """Pad the token list UP to the smallest comptime-supported SDPA seq
    (64/128/256/512/1024/2048) with _BOOGU_PAD_ID, so encode_layer_states' pad
    auto-detect (first 151643) recovers real_len = the original L. The encoder is
    causal so pad columns are masked out — numerically identical to the unpadded L.
    Robust for any prompt up to 2048 tokens (cond/uncond vary by prompt)."""
    var L = len(ids)
    var pad: Int
    if L <= 64:
        pad = 64
    elif L <= 128:
        pad = 128
    elif L <= 256:
        pad = 256
    elif L <= 512:
        pad = 512
    elif L <= 1024:
        pad = 1024
    elif L <= 2048:
        pad = 2048
    else:
        raise Error(
            String("boogu encode: L=") + String(L)
            + " exceeds 2048 (max supported encoder SDPA seq)"
        )
    var out = List[Int]()
    for i in range(L):
        out.append(ids[i])
    for _i in range(pad - L):
        out.append(_BOOGU_PAD_ID)
    return out^


def boogu_encode(enc: Qwen3Encoder, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
    """RAW last-layer hidden (after all 36 decoder layers, NO final norm),
    sliced to the original L rows -> [1, L, 4096]. Handles L=45 via pad-to-64 +
    causal real_len (encode_layer_states detects 151643-padding)."""
    var L = len(ids)
    var padded = _pad_ids(ids)
    var full = enc.encode(padded, enc.config.num_layers - 1, ctx)  # [1, 64, 4096]
    return slice(full, 1, 0, L, ctx)  # [1, L, 4096]


def boogu_encode_normed(enc: Qwen3Encoder, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
    """final_norm(RAW last-layer hidden) -> [1, L, 4096]. This mirrors HF's
    output_hidden_states[-1], where the final entry has model.norm applied. Pad
    to 64, norm the padded [1,64,4096], then slice to L."""
    var L = len(ids)
    var padded = _pad_ids(ids)
    var full = enc.encode(padded, enc.config.num_layers - 1, ctx)  # [1, 64, 4096]
    var normed = enc.final_norm(full, ctx)  # [1, 64, 4096]
    return slice(normed, 1, 0, L, ctx)  # [1, L, 4096]
