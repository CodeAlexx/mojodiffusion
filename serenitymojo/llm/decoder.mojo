# serenitymojo/llm/decoder.mojo — KV-cached incremental decode for Qwen3.
#
# Mirrors Qwen3Encoder._layer EXACTLY for a single new token, but:
#   - computes q/k/v for 1 row,
#   - appends the rope'd K,V to a persistent per-layer cache,
#   - attends the 1 query row over the whole cache via the verified single-query
#     GQA kernel (llm.sqa) instead of the square sdpa.
# Reuses the encoder's verified ops; does NOT modify qwen3_encoder.mojo.
#
# Correctness gate (no HF needed): cached-decode logits == the verified no-cache
# Qwen3Encoder.lm_logits_last, token-for-token.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.text_encoder.qwen3_encoder import (
    Qwen3Encoder, _reshape, _add, _build_rope_tables,
)
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.tensor_algebra import concat
from serenitymojo.llm.sqa import sqa_gpu


struct KVCache(Movable):
    var k: List[ArcPointer[Tensor]]   # per layer: [1, L, H_kv, dh]
    var v: List[ArcPointer[Tensor]]
    var has: List[Bool]
    var length: Int

    def __init__(out self, num_layers: Int):
        self.k = List[ArcPointer[Tensor]]()
        self.v = List[ArcPointer[Tensor]]()
        self.has = List[Bool]()
        for _ in range(num_layers):
            # placeholder; replaced on first append
            self.has.append(False)
        self.length = 0


def _sqa_tensor(
    q: Tensor, ck: Tensor, cv: Tensor, H: Int, H_kv: Int, dh: Int, ctx: DeviceContext
) raises -> Tensor:
    """q [1,1,H,dh], ck/cv [1,L,H_kv,dh] -> attn [1,1,H*dh] (bf16-stored).
    Pulls to host F32, reorders cache [L,H_kv,dh]->[H_kv,L,dh], runs the verified
    GQA single-query kernel, pushes back."""
    var qh = q.to_host(ctx)
    var ckh = ck.to_host(ctx)
    var cvh = cv.to_host(ctx)
    var L = len(ckh) // (H_kv * dh)

    var ql = List[Float32]()
    for i in range(H * dh):
        ql.append(Float32(qh[i]))
    # reorder [L,H_kv,dh] -> [H_kv,L,dh]
    var kl = List[Float32]()
    var vl = List[Float32]()
    kl.resize(H_kv * L * dh, Float32(0.0))
    vl.resize(H_kv * L * dh, Float32(0.0))
    for l in range(L):
        for hk in range(H_kv):
            for d in range(dh):
                var src = (l * H_kv + hk) * dh + d
                var dstv = (hk * L + l) * dh + d
                kl[dstv] = Float32(ckh[src])
                vl[dstv] = Float32(cvh[src])
    var o = sqa_gpu(ctx, ql, kl, vl, H, H_kv, L, dh)   # [H*dh]
    var sh = List[Int]()
    sh.append(1); sh.append(1); sh.append(H * dh)
    return Tensor.from_host(o, sh^, q.dtype(), ctx)


def decode_step(
    enc: Qwen3Encoder, mut cache: KVCache, token_id: Int, pos: Int, ctx: DeviceContext
) raises -> Tensor:
    """One cached decode step at absolute position `pos`. Returns [1,1,vocab]."""
    var cfg = enc.config
    var H = cfg.num_heads
    var H_kv = cfg.num_kv_heads
    var dh = cfg.head_dim
    var eps = cfg.rms_norm_eps
    var half = dh // 2
    var dtype = enc._w(String("model.embed_tokens.weight")).dtype()

    # single-token embedding [1,1,hidden]
    var toks = List[Int]()
    toks.append(token_id)
    var hidden = enc._embed(toks, ctx)

    # RoPE tables for absolute position `pos` (slice the pos-th block).
    var qt = _build_rope_tables(pos + 1, H, dh, cfg.rope_theta)
    var kt = _build_rope_tables(pos + 1, H_kv, dh, cfg.rope_theta)
    var qoff = pos * H * half
    var koff = pos * H_kv * half
    var cosq = List[Float32](); var sinq = List[Float32]()
    for i in range(H * half):
        cosq.append(qt[0][qoff + i]); sinq.append(qt[1][qoff + i])
    var cosk = List[Float32](); var sink = List[Float32]()
    for i in range(H_kv * half):
        cosk.append(kt[0][koff + i]); sink.append(kt[1][koff + i])
    var cq_sh = List[Int](); cq_sh.append(H * half)
    var ck_sh = List[Int](); ck_sh.append(H_kv * half)
    var cos_q = Tensor.from_host(cosq, cq_sh.copy(), dtype, ctx)
    var sin_q = Tensor.from_host(sinq, [H * half], dtype, ctx)
    var cos_k = Tensor.from_host(cosk, ck_sh.copy(), dtype, ctx)
    var sin_k = Tensor.from_host(sink, [H_kv * half], dtype, ctx)

    for layer in range(cfg.num_layers):
        var p = String("model.layers.") + String(layer)
        var normed = rms_norm(hidden, enc._w(p + ".input_layernorm.weight"), eps, ctx)
        var q = linear(normed, enc._w(p + ".self_attn.q_proj.weight"), None, ctx)
        var k = linear(normed, enc._w(p + ".self_attn.k_proj.weight"), None, ctx)
        var v = linear(normed, enc._w(p + ".self_attn.v_proj.weight"), None, ctx)
        q = _reshape(q, [1, 1, H, dh], ctx)
        k = _reshape(k, [1, 1, H_kv, dh], ctx)
        v = _reshape(v, [1, 1, H_kv, dh], ctx)
        q = rms_norm(q, enc._w(p + ".self_attn.q_norm.weight"), eps, ctx)
        k = rms_norm(k, enc._w(p + ".self_attn.k_norm.weight"), eps, ctx)
        q = rope_halfsplit(q, cos_q, sin_q, ctx)
        k = rope_halfsplit(k, cos_k, sin_k, ctx)

        # append k,v to the per-layer cache (concat along seq dim 1)
        if not cache.has[layer]:
            cache.k.append(ArcPointer(k^))
            cache.v.append(ArcPointer(v^))
            cache.has[layer] = True
        else:
            var nk = concat(1, ctx, cache.k[layer][], k)
            var nv = concat(1, ctx, cache.v[layer][], v)
            cache.k[layer] = ArcPointer(nk^)
            cache.v[layer] = ArcPointer(nv^)

        var attn = _sqa_tensor(q, cache.k[layer][], cache.v[layer][], H, H_kv, dh, ctx)
        var attn_out = linear(attn, enc._w(p + ".self_attn.o_proj.weight"), None, ctx)
        var hidden2 = _add(hidden, attn_out, ctx)

        var normed2 = rms_norm(hidden2, enc._w(p + ".post_attention_layernorm.weight"), eps, ctx)
        var gate = linear(normed2, enc._w(p + ".mlp.gate_proj.weight"), None, ctx)
        var up = linear(normed2, enc._w(p + ".mlp.up_proj.weight"), None, ctx)
        var act = swiglu(gate, up, ctx)
        var mlp_out = linear(act, enc._w(p + ".mlp.down_proj.weight"), None, ctx)
        hidden = _add(hidden2, mlp_out, ctx)
        if layer == cfg.num_layers - 1:
            cache.length = cache.length  # no-op; length tracked by caller

    var normed_f = rms_norm(hidden, enc._w(String("model.norm.weight")), eps, ctx)
    return linear(normed_f, enc._w(String("lm_head.weight")), None, ctx)
