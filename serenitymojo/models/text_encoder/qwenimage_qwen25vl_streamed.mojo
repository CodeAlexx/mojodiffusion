# models/text_encoder/qwenimage_qwen25vl_streamed.mojo — 16GB-fit Qwen-Image TE.
#
# The Qwen-Image text encoder (Qwen2.5-VL-7B text tower: hidden 3584, 28
# layers, GQA 28/4 heads, mlp 18944) is ~15 GB BF16 loaded whole — the resident
# Qwen25VLEncoder.load CANNOT fit a 16 GB RTX 5080 even in a fresh child
# process (measured: encoder-child preflight "free VRAM 14770 MiB < required
# 17400 MiB"). This variant streams the encoder LAYER-BY-LAYER from disk: only
# ONE layer's weights (~0.5 GB BF16) is GPU-resident at a time, so peak TE VRAM
# is ~2.7 GB (embed table ~1.1 GB transient + 1 layer + seq-546 activations).
#
# ZERO new math: each layer is executed by the PARITY-GATED
# Qwen25VLEncoder._layer (a per-layer mini Qwen25VLEncoder holds just that
# layer's 13 tensors — q/k/v proj + biases, o_proj, both layernorms,
# gate/up/down; _layer only ever looks up "model.layers.<i>.*" so the subset is
# complete), embedding by Qwen25VLEncoder._embed, RoPE/mask by the same
# module-level builders encode_layer_states uses (same pad auto-detect on
# PAD 151643), and the final model.norm by Qwen25VLEncoder.final_norm. Running
# layers 0..extract_layer then final_norm reproduces exactly what
# qwenimage_sample_cli's resident path computes:
#   enc.encode(ids, EXTRACT_LAYER) → enc.final_norm(...)  (the caller slices
#   [DROP_IDX, DROP_IDX+N_TXT_KEPT) afterwards — NOT done here).
#
# Multi-prompt: all prompts advance through each layer before its weights drop,
# so the ~15 GB safetensors is read ONCE regardless of prompt count (the
# qwenimage CFG pair needs pos+neg). All ids lists must share one padded seq
# len (the fixed N_ENC=546 window); per-prompt real_len is recovered from
# PAD 151643 for the causal mask, identical to encode_layer_states.
#
# Structure mirrors ideogram_qwen3vl_streamed.mojo (the shipped 16GB-fit
# layer-streamed TE this file ports to Qwen2.5-VL); parity gated by
# parity/qwenimage_te_streamed_parity.mojo (cos >= 0.999 vs the torch oracle
# last_hidden_state — the resident Mojo encoder cannot load on this card).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.

from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.qwen25vl_encoder import (
    Qwen25VLEncoder,
    Qwen25VLConfig,
    _build_rope_tables,
    _build_causal_mask,
)

comptime TArc = ArcPointer[Tensor]

comptime _PAD_ID = 151643  # right-pad id (qwen25vl_encoder.rs:455 convention)


# Load one tensor from the sharded text_encoder dir into a mini weight table
# (plain BF16 Tensor.from_view H2D — the exact resident-loader code path; the
# Qwen-Image text encoder ships BF16, no dequant involved).
def _add_w(
    st: ShardedSafeTensors,
    mut weights: List[TArc],
    mut n2i: Dict[String, Int],
    name: String,
    ctx: DeviceContext,
) raises:
    var t = Tensor.from_view(st.tensor_view(name), ctx)
    n2i[name] = len(weights)
    weights.append(TArc(t^))


# Embed all prompts with a transient mini-encoder holding ONLY embed_tokens
# (~1.1 GB BF16); it frees on return.
def _embed_all(
    st: ShardedSafeTensors, ids_list: List[List[Int]], ctx: DeviceContext
) raises -> List[TArc]:
    var weights = List[TArc]()
    var n2i = Dict[String, Int]()
    _add_w(st, weights, n2i, String("model.embed_tokens.weight"), ctx)
    var emb = Qwen25VLEncoder(weights^, n2i^, Qwen25VLConfig.qwen_image())
    var out = List[TArc]()
    for i in range(len(ids_list)):
        out.append(TArc(emb._embed(ids_list[i], ctx)))
    return out^


# One layer's 13 tensors -> mini Qwen25VLEncoder. _layer's lookups are exactly
# these names (q/k/v have biases in Qwen2.5-VL; o_proj/gate/up/down are
# bias-free; no q_norm/k_norm — see qwen25vl_encoder.mojo header).
def _load_layer(
    st: ShardedSafeTensors, li: Int, ctx: DeviceContext
) raises -> Qwen25VLEncoder:
    var weights = List[TArc]()
    var n2i = Dict[String, Int]()
    var p = String("model.layers.") + String(li) + "."
    _add_w(st, weights, n2i, p + "input_layernorm.weight", ctx)
    _add_w(st, weights, n2i, p + "self_attn.q_proj.weight", ctx)
    _add_w(st, weights, n2i, p + "self_attn.q_proj.bias", ctx)
    _add_w(st, weights, n2i, p + "self_attn.k_proj.weight", ctx)
    _add_w(st, weights, n2i, p + "self_attn.k_proj.bias", ctx)
    _add_w(st, weights, n2i, p + "self_attn.v_proj.weight", ctx)
    _add_w(st, weights, n2i, p + "self_attn.v_proj.bias", ctx)
    _add_w(st, weights, n2i, p + "self_attn.o_proj.weight", ctx)
    _add_w(st, weights, n2i, p + "post_attention_layernorm.weight", ctx)
    _add_w(st, weights, n2i, p + "mlp.gate_proj.weight", ctx)
    _add_w(st, weights, n2i, p + "mlp.up_proj.weight", ctx)
    _add_w(st, weights, n2i, p + "mlp.down_proj.weight", ctx)
    return Qwen25VLEncoder(weights^, n2i^, Qwen25VLConfig.qwen_image())


# ids_list: N prompts, ALL padded to one shared seq (PAD 151643). Runs layers
# 0..extract_layer (inclusive) with ONE layer resident at a time, then applies
# model.norm (final_norm). Returns N tensors [1, seq, 3584] BF16 — the same
# values as resident `enc.final_norm(enc.encode(ids, extract_layer, ctx), ctx)`.
# The qwenimage template-drop slice [DROP_IDX, DROP_IDX+N_TXT_KEPT) is the
# caller's job (mirrors _encode_trimmed in qwenimage_sample_cli).
def encode_qwen25vl_final_streamed(
    dir: String, ids_list: List[List[Int]], extract_layer: Int, ctx: DeviceContext
) raises -> List[TArc]:
    var n = len(ids_list)
    if n < 1:
        raise Error("encode_qwen25vl_final_streamed: no prompts")
    var seq = len(ids_list[0])
    for i in range(n):
        if len(ids_list[i]) != seq:
            raise Error(
                "encode_qwen25vl_final_streamed: all prompts must share one padded seq"
            )
    var cfg = Qwen25VLConfig.qwen_image()
    if extract_layer < 0 or extract_layer >= cfg.num_layers:
        raise Error("encode_qwen25vl_final_streamed: extract_layer out of range")
    var st = ShardedSafeTensors.open(dir)

    # ── embed (transient ~1.1 GB table) ────────────────────────────────────────
    var hiddens = _embed_all(st, ids_list, ctx)     # N x [1,seq,3584] BF16
    ctx.synchronize()

    # ── shared RoPE tables + per-prompt causal masks (encode_layer_states) ─────
    var dtype = hiddens[0][].dtype()
    var q_tables = _build_rope_tables(seq, cfg.num_heads, cfg.head_dim, cfg.rope_theta)
    var k_tables = _build_rope_tables(seq, cfg.num_kv_heads, cfg.head_dim, cfg.rope_theta)
    var half = cfg.head_dim // 2
    var cq_sh = List[Int]()
    cq_sh.append(seq * cfg.num_heads * half)
    var ck_sh = List[Int]()
    ck_sh.append(seq * cfg.num_kv_heads * half)
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
        var mask_data = _build_causal_mask(seq, cfg.num_heads, real_len)
        var mask_sh = List[Int]()
        mask_sh.append(1)
        mask_sh.append(cfg.num_heads)
        mask_sh.append(seq)
        mask_sh.append(seq)
        masks.append(TArc(Tensor.from_host(mask_data, mask_sh^, dtype, ctx)))

    # ── layer stream: ONE layer resident at a time, all prompts advance ───────
    for li in range(extract_layer + 1):
        var lw = _load_layer(st, li, ctx)
        for p in range(n):
            var h_new = lw._layer(
                li, hiddens[p][], cos_q, sin_q, cos_k, sin_k, masks[p][], ctx
            )
            hiddens[p] = TArc(h_new^)
        ctx.synchronize()   # layer weights drop here; mempool slot reused

    # ── final model.norm (transient tiny mini-encoder) ─────────────────────────
    var nw = List[TArc]()
    var nn2i = Dict[String, Int]()
    _add_w(st, nw, nn2i, String("model.norm.weight"), ctx)
    var fin = Qwen25VLEncoder(nw^, nn2i^, cfg)
    var out = List[TArc]()
    for p in range(n):
        out.append(TArc(fin.final_norm(hiddens[p][], ctx)))
    return out^
