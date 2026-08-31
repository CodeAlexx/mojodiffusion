# MiniMax-H3 LoRA block -- PARITY-ONLY F32 host reference.
#
# Oracle: kohya-ss/musubi-tuner dev at
# b8717864713c9e4e7ef3d56eba1fc695a9b626a5:
#   src/musubi_tuner/minimax_h3/model.py::DiTBlock.forward
#   src/musubi_tuner/networks/lora_minimax_h3.py
#
# This file is deliberately under parity/: it is not product trainer compute,
# must never be imported by a training entrypoint, and makes no device or
# released-geometry claim. It establishes the differentiable reference contract
# for the four Musubi LoRA
# targets: fused attn.qkv_proj, attn.out_proj, mlp.fc1 and mlp.fc2.  The base
# weights, norms and AdaLN vectors are frozen; gradients are returned for x and
# every LoRA A/B tensor.
#
# Weight-order contract at the Musubi module/LoRA boundary:
#   qkv_proj rows = [all-q; all-k; all-v] (Musubi module in-memory order)
#   mlp.fc1 rows  = [gate; value]
# The published checkpoint stores QKV rows per head; Musubi converts that file
# layout before the module runs. Product training must apply the same conversion
# to both frozen base rows and LoRA-up rows at its checkpoint boundary.
# The Serenity inference block consumes transformed [all-q; all-k; all-v] QKV
# and [value; gate] fc1.  Training code must not silently cross that boundary.

from std.collections import List
from std.math import exp, sqrt


@fieldwise_init
struct MiniMaxH3TrainingBlockConfig(Copyable, Movable):
    var heads: Int
    var head_dim: Int
    var hidden: Int
    var ffn: Int
    var rotary_dim: Int
    var eps: Float32
    var qk_eps: Float32

    def inner(self) -> Int:
        return self.heads * self.head_dim

    def validate(self) raises:
        if self.heads <= 0 or self.head_dim <= 0 or self.hidden <= 0:
            raise Error("MiniMax-H3 training block dimensions must be positive")
        if self.ffn <= 0 or self.rotary_dim <= 0:
            raise Error("MiniMax-H3 ffn/rotary dimensions must be positive")
        if self.rotary_dim > self.head_dim or self.rotary_dim % 2 != 0:
            raise Error("MiniMax-H3 rotary_dim must be even and <= head_dim")
        if self.inner() == self.hidden:
            raise Error("MiniMax-H3 fixture must exercise inner != hidden")


@fieldwise_init
struct MiniMaxH3TrainingBlockWeights(Copyable, Movable):
    var norm1: List[Float32]       # [D]
    var qkv: List[Float32]         # [3*I,D], module [all-q;all-k;all-v]
    var q_norm: List[Float32]      # [Dh]
    var k_norm: List[Float32]      # [Dh]
    var out_proj: List[Float32]    # [D,I]
    var norm2: List[Float32]       # [D]
    var fc1: List[Float32]         # [2F,D], raw [gate;value]
    var fc2: List[Float32]         # [D,F]


@fieldwise_init
struct MiniMaxH3BlockModulation(Copyable, Movable):
    # Already gathered per packed-sequence row. H3's caller obtains these with
    # row = timestep_index * 3 + modality_tag.
    var shift_msa: List[Float32]   # [S,D]
    var scale_msa: List[Float32]   # [S,D]
    var gate_msa: List[Float32]    # [S,D]
    var shift_mlp: List[Float32]   # [S,D]
    var scale_mlp: List[Float32]   # [S,D]
    var gate_mlp: List[Float32]    # [S,D]


@fieldwise_init
struct MiniMaxH3LoraAdapter(Copyable, Movable):
    var a: List[Float32]           # [R,in]
    var b: List[Float32]           # [out,R]
    var rank: Int
    var in_features: Int
    var out_features: Int
    var scale: Float32


@fieldwise_init
struct MiniMaxH3TrainingBlockLora(Copyable, Movable):
    var qkv: MiniMaxH3LoraAdapter
    var out_proj: MiniMaxH3LoraAdapter
    var fc1: MiniMaxH3LoraAdapter
    var fc2: MiniMaxH3LoraAdapter


@fieldwise_init
struct MiniMaxH3LoraGrad(Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]


@fieldwise_init
struct MiniMaxH3TrainingBlockBackward(Movable):
    var d_x: List[Float32]
    var qkv: MiniMaxH3LoraGrad
    var out_proj: MiniMaxH3LoraGrad
    var fc1: MiniMaxH3LoraGrad
    var fc2: MiniMaxH3LoraGrad


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        out.append(0.0)
    return out^


def _add(a: List[Float32], b: List[Float32]) raises -> List[Float32]:
    if len(a) != len(b):
        raise Error("MiniMax-H3 training block add shape mismatch")
    var out = List[Float32](capacity=len(a))
    for i in range(len(a)):
        out.append(a[i] + b[i])
    return out^


def _linear(x: List[Float32], w: List[Float32], rows: Int, inf: Int, outf: Int) raises -> List[Float32]:
    if len(x) != rows * inf or len(w) != outf * inf:
        raise Error("MiniMax-H3 training block linear shape mismatch")
    var y = _zeros(rows * outf)
    for r in range(rows):
        for o in range(outf):
            var acc = Float32(0.0)
            for i in range(inf):
                acc += x[r * inf + i] * w[o * inf + i]
            y[r * outf + o] = acc
    return y^


@fieldwise_init
struct _LinearBackward(Movable):
    var d_x: List[Float32]
    var d_w: List[Float32]


def _linear_backward(dy: List[Float32], x: List[Float32], w: List[Float32], rows: Int, inf: Int, outf: Int) raises -> _LinearBackward:
    if len(dy) != rows * outf:
        raise Error("MiniMax-H3 training block linear backward shape mismatch")
    var dx = _zeros(rows * inf)
    var dw = _zeros(outf * inf)
    for r in range(rows):
        for o in range(outf):
            var g = dy[r * outf + o]
            for i in range(inf):
                dx[r * inf + i] += g * w[o * inf + i]
                dw[o * inf + i] += g * x[r * inf + i]
    return _LinearBackward(dx^, dw^)


def _lora_forward(x: List[Float32], base_w: List[Float32], lo: MiniMaxH3LoraAdapter, rows: Int) raises -> List[Float32]:
    var base = _linear(x, base_w, rows, lo.in_features, lo.out_features)
    var t = _linear(x, lo.a, rows, lo.in_features, lo.rank)
    var delta = _linear(t, lo.b, rows, lo.rank, lo.out_features)
    for i in range(len(base)):
        base[i] += lo.scale * delta[i]
    return base^


@fieldwise_init
struct _LoraBackward(Movable):
    var d_x: List[Float32]
    var d_a: List[Float32]
    var d_b: List[Float32]


def _lora_backward(dy: List[Float32], x: List[Float32], base_w: List[Float32], lo: MiniMaxH3LoraAdapter, rows: Int) raises -> _LoraBackward:
    var base = _linear_backward(dy, x, base_w, rows, lo.in_features, lo.out_features)
    var t = _linear(x, lo.a, rows, lo.in_features, lo.rank)
    var scaled_dy = dy.copy()
    for i in range(len(scaled_dy)):
        scaled_dy[i] *= lo.scale
    var bb = _linear_backward(scaled_dy, t, lo.b, rows, lo.rank, lo.out_features)
    var ab = _linear_backward(bb.d_x, x, lo.a, rows, lo.in_features, lo.rank)
    # Copy fields out of aggregate results: moving one List out of the middle
    # would partially destroy the struct before its remaining fields are read.
    var dx = _add(base.d_x, ab.d_x)
    var d_a = ab.d_w.copy()
    var d_b = bb.d_w.copy()
    return _LoraBackward(dx^, d_a^, d_b^)


def _rms_norm(x: List[Float32], gamma: List[Float32], rows: Int, width: Int, eps: Float32) raises -> List[Float32]:
    var y = _zeros(rows * width)
    for r in range(rows):
        var ss = Float32(0.0)
        for i in range(width):
            var v = x[r * width + i]
            ss += v * v
        var inv = Float32(1.0) / sqrt(ss / Float32(width) + eps)
        for i in range(width):
            y[r * width + i] = x[r * width + i] * inv * gamma[i]
    return y^


def _rms_norm_backward_dx(dy: List[Float32], x: List[Float32], gamma: List[Float32], rows: Int, width: Int, eps: Float32) -> List[Float32]:
    var dx = _zeros(rows * width)
    for r in range(rows):
        var ss = Float32(0.0)
        var dot = Float32(0.0)
        for i in range(width):
            var xv = x[r * width + i]
            ss += xv * xv
            dot += dy[r * width + i] * gamma[i] * xv
        var inv = Float32(1.0) / sqrt(ss / Float32(width) + eps)
        var corr = inv * inv * inv * dot / Float32(width)
        for i in range(width):
            dx[r * width + i] = dy[r * width + i] * gamma[i] * inv - x[r * width + i] * corr
    return dx^


def _modulate(x: List[Float32], scale: List[Float32], shift: List[Float32]) raises -> List[Float32]:
    if len(x) != len(scale) or len(x) != len(shift):
        raise Error("MiniMax-H3 modulation shape mismatch")
    var y = _zeros(len(x))
    for i in range(len(x)):
        y[i] = x[i] * (1.0 + scale[i]) + shift[i]
    return y^


def _modulate_backward_dx(dy: List[Float32], scale: List[Float32]) -> List[Float32]:
    var dx = _zeros(len(dy))
    for i in range(len(dy)):
        dx[i] = dy[i] * (1.0 + scale[i])
    return dx^


def _residual_gate(x: List[Float32], gate: List[Float32], branch: List[Float32]) -> List[Float32]:
    var y = _zeros(len(x))
    for i in range(len(x)):
        y[i] = x[i] + gate[i] * branch[i]
    return y^


def _qkv_split_raw(qkv: List[Float32], S: Int, H: Int, Dh: Int) -> List[List[Float32]]:
    # Musubi Attention.forward does `.split(inner_dim, dim=-1)`: the fused
    # projection is contiguous [all-q;all-k;all-v].
    var q = _zeros(S * H * Dh)
    var k = _zeros(S * H * Dh)
    var v = _zeros(S * H * Dh)
    for s in range(S):
        for h in range(H):
            for c in range(Dh):
                var dst = (s * H + h) * Dh + c
                var base = s * 3 * H * Dh + h * Dh + c
                q[dst] = qkv[base]
                k[dst] = qkv[base + H * Dh]
                v[dst] = qkv[base + 2 * H * Dh]
    var out = List[List[Float32]]()
    out.append(q^); out.append(k^); out.append(v^)
    return out^


def _qkv_join_raw(q: List[Float32], k: List[Float32], v: List[Float32], S: Int, H: Int, Dh: Int) -> List[Float32]:
    var out = _zeros(S * H * 3 * Dh)
    for s in range(S):
        for h in range(H):
            for c in range(Dh):
                var src = (s * H + h) * Dh + c
                var base = s * 3 * H * Dh + h * Dh + c
                out[base] = q[src]
                out[base + H * Dh] = k[src]
                out[base + 2 * H * Dh] = v[src]
    return out^


def _partial_rope(x: List[Float32], cos: List[Float32], sin: List[Float32], S: Int, H: Int, Dh: Int, R: Int) -> List[Float32]:
    var y = x.copy()
    var half = R // 2
    for s in range(S):
        for h in range(H):
            var base = (s * H + h) * Dh
            for i in range(half):
                var x0 = x[base + i]
                var x1 = x[base + i + half]
                y[base + i] = x0 * cos[s * R + i] - x1 * sin[s * R + i]
                y[base + i + half] = x1 * cos[s * R + i + half] + x0 * sin[s * R + i + half]
    return y^


def _partial_rope_backward(dy: List[Float32], cos: List[Float32], sin: List[Float32], S: Int, H: Int, Dh: Int, R: Int) -> List[Float32]:
    var dx = dy.copy()
    var half = R // 2
    for s in range(S):
        for h in range(H):
            var base = (s * H + h) * Dh
            for i in range(half):
                var g0 = dy[base + i]
                var g1 = dy[base + i + half]
                dx[base + i] = g0 * cos[s * R + i] + g1 * sin[s * R + i + half]
                dx[base + i + half] = -g0 * sin[s * R + i] + g1 * cos[s * R + i + half]
    return dx^


@fieldwise_init
struct _AttentionForward(Movable):
    var y: List[Float32]
    var probs: List[Float32]       # [H,S,S]


def _attention(q: List[Float32], k: List[Float32], v: List[Float32], S: Int, H: Int, Dh: Int) -> _AttentionForward:
    var probs = _zeros(H * S * S)
    var y = _zeros(S * H * Dh)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    for h in range(H):
        for i in range(S):
            var max_score = Float32(-3.4028235e38)
            for j in range(S):
                var score = Float32(0.0)
                for c in range(Dh):
                    score += q[(i * H + h) * Dh + c] * k[(j * H + h) * Dh + c]
                score *= scale
                probs[(h * S + i) * S + j] = score
                if score > max_score:
                    max_score = score
            var denom = Float32(0.0)
            for j in range(S):
                var p = exp(probs[(h * S + i) * S + j] - max_score)
                probs[(h * S + i) * S + j] = p
                denom += p
            for j in range(S):
                var p = probs[(h * S + i) * S + j] / denom
                probs[(h * S + i) * S + j] = p
                for c in range(Dh):
                    y[(i * H + h) * Dh + c] += p * v[(j * H + h) * Dh + c]
    return _AttentionForward(y^, probs^)


def _attention_backward(dy: List[Float32], q: List[Float32], k: List[Float32], v: List[Float32], probs: List[Float32], S: Int, H: Int, Dh: Int) -> List[List[Float32]]:
    var dq = _zeros(S * H * Dh)
    var dk = _zeros(S * H * Dh)
    var dv = _zeros(S * H * Dh)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    for h in range(H):
        for i in range(S):
            var weighted_dp = Float32(0.0)
            var dp = _zeros(S)
            for j in range(S):
                for c in range(Dh):
                    dp[j] += dy[(i * H + h) * Dh + c] * v[(j * H + h) * Dh + c]
                weighted_dp += probs[(h * S + i) * S + j] * dp[j]
            for j in range(S):
                var p = probs[(h * S + i) * S + j]
                var ds = p * (dp[j] - weighted_dp)
                for c in range(Dh):
                    var qi = (i * H + h) * Dh + c
                    var kj = (j * H + h) * Dh + c
                    dq[qi] += ds * k[kj] * scale
                    dk[kj] += ds * q[qi] * scale
                    dv[kj] += p * dy[qi]
    var out = List[List[Float32]]()
    out.append(dq^); out.append(dk^); out.append(dv^)
    return out^


def _swiglu_raw(fc1: List[Float32], S: Int, F: Int) -> List[Float32]:
    var y = _zeros(S * F)
    for s in range(S):
        for i in range(F):
            var g = fc1[s * 2 * F + i]
            var u = fc1[s * 2 * F + F + i]
            var sig = Float32(1.0) / (1.0 + exp(-g))
            y[s * F + i] = g * sig * u
    return y^


def _swiglu_raw_backward(dy: List[Float32], fc1: List[Float32], S: Int, F: Int) -> List[Float32]:
    var dx = _zeros(S * 2 * F)
    for s in range(S):
        for i in range(F):
            var g = fc1[s * 2 * F + i]
            var u = fc1[s * 2 * F + F + i]
            var sig = Float32(1.0) / (1.0 + exp(-g))
            var go = dy[s * F + i]
            dx[s * 2 * F + i] = go * u * sig * (1.0 + g * (1.0 - sig))
            dx[s * 2 * F + F + i] = go * g * sig
    return dx^


def minimax_h3_training_block_reference_forward(x: List[Float32], w: MiniMaxH3TrainingBlockWeights, mod: MiniMaxH3BlockModulation, lo: MiniMaxH3TrainingBlockLora, cos: List[Float32], sin: List[Float32], S: Int, cfg: MiniMaxH3TrainingBlockConfig) raises -> List[Float32]:
    cfg.validate()
    var D = cfg.hidden
    var n1 = _rms_norm(x, w.norm1, S, D, cfg.eps)
    var a1 = _modulate(n1, mod.scale_msa, mod.shift_msa)
    var qkv = _lora_forward(a1, w.qkv, lo.qkv, S)
    var parts = _qkv_split_raw(qkv, S, cfg.heads, cfg.head_dim)
    var qn = _rms_norm(parts[0], w.q_norm, S * cfg.heads, cfg.head_dim, cfg.qk_eps)
    var kn = _rms_norm(parts[1], w.k_norm, S * cfg.heads, cfg.head_dim, cfg.qk_eps)
    var qr = _partial_rope(qn, cos, sin, S, cfg.heads, cfg.head_dim, cfg.rotary_dim)
    var kr = _partial_rope(kn, cos, sin, S, cfg.heads, cfg.head_dim, cfg.rotary_dim)
    var attn = _attention(qr, kr, parts[2], S, cfg.heads, cfg.head_dim)
    var ao = _lora_forward(attn.y, w.out_proj, lo.out_proj, S)
    var x1 = _residual_gate(x, mod.gate_msa, ao)
    var n2 = _rms_norm(x1, w.norm2, S, D, cfg.eps)
    var a2 = _modulate(n2, mod.scale_mlp, mod.shift_mlp)
    var fc1 = _lora_forward(a2, w.fc1, lo.fc1, S)
    var act = _swiglu_raw(fc1, S, cfg.ffn)
    var fc2 = _lora_forward(act, w.fc2, lo.fc2, S)
    return _residual_gate(x1, mod.gate_mlp, fc2)


def minimax_h3_training_block_reference_backward(dy: List[Float32], x: List[Float32], w: MiniMaxH3TrainingBlockWeights, mod: MiniMaxH3BlockModulation, lo: MiniMaxH3TrainingBlockLora, cos: List[Float32], sin: List[Float32], S: Int, cfg: MiniMaxH3TrainingBlockConfig) raises -> MiniMaxH3TrainingBlockBackward:
    # Per-block recompute: no full-stack activation tape.
    cfg.validate()
    var D = cfg.hidden
    var n1 = _rms_norm(x, w.norm1, S, D, cfg.eps)
    var a1 = _modulate(n1, mod.scale_msa, mod.shift_msa)
    var qkv = _lora_forward(a1, w.qkv, lo.qkv, S)
    var parts = _qkv_split_raw(qkv, S, cfg.heads, cfg.head_dim)
    var qn = _rms_norm(parts[0], w.q_norm, S * cfg.heads, cfg.head_dim, cfg.qk_eps)
    var kn = _rms_norm(parts[1], w.k_norm, S * cfg.heads, cfg.head_dim, cfg.qk_eps)
    var qr = _partial_rope(qn, cos, sin, S, cfg.heads, cfg.head_dim, cfg.rotary_dim)
    var kr = _partial_rope(kn, cos, sin, S, cfg.heads, cfg.head_dim, cfg.rotary_dim)
    var attn = _attention(qr, kr, parts[2], S, cfg.heads, cfg.head_dim)
    var ao = _lora_forward(attn.y, w.out_proj, lo.out_proj, S)
    var x1 = _residual_gate(x, mod.gate_msa, ao)
    var n2 = _rms_norm(x1, w.norm2, S, D, cfg.eps)
    var a2 = _modulate(n2, mod.scale_mlp, mod.shift_mlp)
    var fc1 = _lora_forward(a2, w.fc1, lo.fc1, S)
    var act = _swiglu_raw(fc1, S, cfg.ffn)

    var d_fc2 = _zeros(len(dy))
    for i in range(len(dy)):
        d_fc2[i] = dy[i] * mod.gate_mlp[i]
    var b_fc2 = _lora_backward(d_fc2, act, w.fc2, lo.fc2, S)
    var d_fc1 = _swiglu_raw_backward(b_fc2.d_x, fc1, S, cfg.ffn)
    var b_fc1 = _lora_backward(d_fc1, a2, w.fc1, lo.fc1, S)
    var d_n2 = _modulate_backward_dx(b_fc1.d_x, mod.scale_mlp)
    var d_x1_norm = _rms_norm_backward_dx(d_n2, x1, w.norm2, S, D, cfg.eps)
    var d_x1 = _add(dy, d_x1_norm)

    var d_ao = _zeros(len(d_x1))
    for i in range(len(d_x1)):
        d_ao[i] = d_x1[i] * mod.gate_msa[i]
    var b_out = _lora_backward(d_ao, attn.y, w.out_proj, lo.out_proj, S)
    var b_attn = _attention_backward(b_out.d_x, qr, kr, parts[2], attn.probs, S, cfg.heads, cfg.head_dim)
    var d_qn = _partial_rope_backward(b_attn[0], cos, sin, S, cfg.heads, cfg.head_dim, cfg.rotary_dim)
    var d_kn = _partial_rope_backward(b_attn[1], cos, sin, S, cfg.heads, cfg.head_dim, cfg.rotary_dim)
    var d_q = _rms_norm_backward_dx(d_qn, parts[0], w.q_norm, S * cfg.heads, cfg.head_dim, cfg.qk_eps)
    var d_k = _rms_norm_backward_dx(d_kn, parts[1], w.k_norm, S * cfg.heads, cfg.head_dim, cfg.qk_eps)
    var d_qkv = _qkv_join_raw(d_q, d_k, b_attn[2], S, cfg.heads, cfg.head_dim)
    var b_qkv = _lora_backward(d_qkv, a1, w.qkv, lo.qkv, S)
    var d_n1 = _modulate_backward_dx(b_qkv.d_x, mod.scale_msa)
    var d_x_norm = _rms_norm_backward_dx(d_n1, x, w.norm1, S, D, cfg.eps)
    var d_x = _add(d_x1, d_x_norm)

    return MiniMaxH3TrainingBlockBackward(
        d_x^,
        MiniMaxH3LoraGrad(b_qkv.d_a.copy(), b_qkv.d_b.copy()),
        MiniMaxH3LoraGrad(b_out.d_a.copy(), b_out.d_b.copy()),
        MiniMaxH3LoraGrad(b_fc1.d_a.copy(), b_fc1.d_b.copy()),
        MiniMaxH3LoraGrad(b_fc2.d_a.copy(), b_fc2.d_b.copy()),
    )
