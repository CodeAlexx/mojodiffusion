# models/text_encoder/ideogram_qwen3vl_streamed.mojo — 16GB-fit Ideogram-4 TE.
#
# The ideogram TE (Qwen3-VL-8B text tower: hidden 4096, 36 layers, mlp 12288)
# dequants from weight-only FP8 to ~15.1GB BF16 — load_ideogram_qwen3vl barely
# fits a 16GB RTX 5080 and the seq-1024 forward OOMs (measured 2026-07-14, task
# #25 gate 2). This variant streams the encoder LAYER-BY-LAYER from disk: only
# ONE layer's dequanted weights (~0.4GB) is GPU-resident at a time, so peak TE
# VRAM is ~1.9GB (embed table 1.2GB + layer + activations).
#
# ZERO new math: each layer is executed by the PARITY-GATED Qwen3Encoder._layer
# (a per-layer mini Qwen3Encoder holds just that layer's 11 tensors; _layer only
# ever looks up "model.layers.<i>.*" so the subset is complete), embedding by
# Qwen3Encoder._embed, RoPE/mask by the same module-level builders
# encode_layer_states uses, and the FP8 dequant by ideogram_qwen3vl._add. The
# 13-tap interleave mirrors encode_ideogram_taps (layers 0,3,...,33 + 35,
# PRE-final-norm — model.norm is never applied, exactly like the resident path).
#
# Multi-prompt: all prompts advance through each layer before its weights drop,
# so the 8.7GB safetensors is read ONCE regardless of prompt count (FlowEdit
# needs src+tgt). All ids lists must share one padded seq len (the fixed
# 1024-token window); per-prompt real_len is recovered from PAD 151643 for the
# causal mask, identical to encode_layer_states' pad auto-detect.
#
# Parity: gated vs the resident encoder output cached in the giger trainer cache
# (parity/ideogram4_te_streamed_parity — cos vs llm.<i>).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import concat, reshape
from serenitymojo.models.text_encoder.qwen3_encoder import (
    Qwen3Encoder,
    Qwen3Config,
    _build_rope_tables,
    _build_causal_mask,
    _clone,
)
from serenitymojo.models.text_encoder.ideogram_qwen3vl import _add

comptime TArc = ArcPointer[Tensor]

comptime _PAD_ID = 151643
comptime _HIDDEN = 4096
comptime _LAYERS = 36
comptime _HEADS = 32
comptime _KV_HEADS = 8
comptime _HEAD_DIM = 128
comptime _THETA = Float64(5000000.0)


def _cfg() -> Qwen3Config:
    return Qwen3Config(
        _HIDDEN, _LAYERS, _HEADS, _KV_HEADS, _HEAD_DIM,
        Float32(1.0e-6), _THETA,
    )


# Embed all prompts with a transient mini-encoder holding ONLY embed_tokens
# (~1.2GB BF16); it frees on return.
def _embed_all(
    st: ShardedSafeTensors, ids_list: List[List[Int]], ctx: DeviceContext
) raises -> List[TArc]:
    var weights = List[TArc]()
    var n2i = Dict[String, Int]()
    _add(st, weights, n2i,
         String("model.embed_tokens.weight"),
         String("language_model.embed_tokens.weight"), ctx)
    var emb = Qwen3Encoder(weights^, n2i^, _cfg())
    var out = List[TArc]()
    for i in range(len(ids_list)):
        out.append(TArc(emb._embed(ids_list[i], ctx)))
    return out^


# One layer's 11 tensors -> mini Qwen3Encoder (FP8 *_proj dequanted to BF16 by
# ideogram_qwen3vl._add — the exact resident-loader code path).
def _load_layer(
    st: ShardedSafeTensors, li: Int, ctx: DeviceContext
) raises -> Qwen3Encoder:
    var weights = List[TArc]()
    var n2i = Dict[String, Int]()
    var ps = String("language_model.layers.") + String(li) + "."
    var pd = String("model.layers.") + String(li) + "."
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
    return Qwen3Encoder(weights^, n2i^, _cfg())


def _is_tap(li: Int) -> Bool:
    # encode_ideogram_taps: states[0,3,6,...,33] (12 taps) + states[35].
    if li == 35:
        return True
    return li <= 33 and li % 3 == 0


# ids_list: N prompts, ALL padded to one shared seq (PAD 151643). Returns N
# tensors [1, seq, 53248] BF16 (13-tap interleaved concat, pad rows NOT zeroed
# — zero them at the call site per the ideogram4_prepare convention).
def encode_ideogram_taps_streamed(
    dir_or_file: String, ids_list: List[List[Int]], ctx: DeviceContext
) raises -> List[TArc]:
    var n = len(ids_list)
    if n < 1:
        raise Error("encode_ideogram_taps_streamed: no prompts")
    var seq = len(ids_list[0])
    for i in range(n):
        if len(ids_list[i]) != seq:
            raise Error(
                "encode_ideogram_taps_streamed: all prompts must share one padded seq"
            )
    var st = ShardedSafeTensors.open(dir_or_file)

    # ── embed (transient 1.2GB table) ──────────────────────────────────────────
    var hiddens = _embed_all(st, ids_list, ctx)     # N x [1,seq,4096] BF16
    ctx.synchronize()

    # ── shared RoPE tables + per-prompt causal masks (encode_layer_states) ─────
    var dtype = hiddens[0][].dtype()
    var q_tables = _build_rope_tables(seq, _HEADS, _HEAD_DIM, _THETA)
    var k_tables = _build_rope_tables(seq, _KV_HEADS, _HEAD_DIM, _THETA)
    comptime half = _HEAD_DIM // 2
    var cq_sh = List[Int]()
    cq_sh.append(seq * _HEADS * half)
    var ck_sh = List[Int]()
    ck_sh.append(seq * _KV_HEADS * half)
    var cos_q = Tensor.from_host(q_tables[0], cq_sh.copy(), dtype, ctx)
    var sin_q = Tensor.from_host(q_tables[1], cq_sh.copy(), dtype, ctx)
    var cos_k = Tensor.from_host(k_tables[0], ck_sh.copy(), dtype, ctx)
    var sin_k = Tensor.from_host(k_tables[1], ck_sh.copy(), dtype, ctx)

    var masks = List[TArc]()
    for p in range(n):
        var real_len = seq
        for i in range(seq):
            if ids_list[p][i] == _PAD_ID:
                real_len = i
                break
        var mask_data = _build_causal_mask(seq, _HEADS, real_len)
        var mask_sh = List[Int]()
        mask_sh.append(1)
        mask_sh.append(_HEADS)
        mask_sh.append(seq)
        mask_sh.append(seq)
        masks.append(TArc(Tensor.from_host(mask_data, mask_sh^, dtype, ctx)))

    # ── layer stream: ONE layer resident at a time, all prompts advance ───────
    var taps = List[List[TArc]]()   # per prompt: 13 x [1,seq,4096]
    for _ in range(n):
        taps.append(List[TArc]())
    for li in range(_LAYERS):
        var lw = _load_layer(st, li, ctx)
        for p in range(n):
            var h_new = lw._layer(
                li, hiddens[p][], cos_q, sin_q, cos_k, sin_k, masks[p][], ctx
            )
            hiddens[p] = TArc(h_new^)
            if _is_tap(li):
                taps[p].append(TArc(_clone(hiddens[p][], ctx)))
        ctx.synchronize()   # layer weights drop here; mempool slot reused

    # ── 13-tap interleaved concat (mirror encode_ideogram_taps) ───────────────
    var out = List[TArc]()
    for p in range(n):
        if len(taps[p]) != 13:
            raise Error("encode_ideogram_taps_streamed: tap count != 13")
        var s4 = [1, seq, _HIDDEN, 1]
        var cat = reshape(taps[p][0][], s4.copy(), ctx)
        for t in range(1, 13):
            var r = reshape(taps[p][t][], s4.copy(), ctx)
            cat = concat(3, ctx, cat, r)
        out.append(TArc(reshape(cat, [1, seq, _HIDDEN * 13], ctx)))
    return out^
