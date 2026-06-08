# models/text_encoder/umt5_encoder.mojo — umt5-xxl encoder (NAVA text encoder).
#
# Pure-Mojo, inference-only port of the umt5-xxl text encoder used by NAVA
# (Baidu audio-video MMDiT). Architecture is T5-v1.1 XXL with per-layer relative
# position bias (shared_pos=False) instead of T5's shared layer-0 bias.
#
# Config (umt5-xxl):
#   vocab   256384, d_model 4096, num_layers 24, d_ff 10240,
#   num_heads 64, d_kv 64, num_buckets 32, max_dist 128, eps 1e-6.
#
# Weight key format (umt5_xxl_enc.safetensors, all BF16):
#   token_embedding.weight                      [256384, 4096]
#   blocks.{i}.norm1.weight                     [4096]
#   blocks.{i}.attn.{q,k,v,o}.weight            [4096, 4096]  (no bias)
#   blocks.{i}.norm2.weight                     [4096]
#   blocks.{i}.ffn.gate.0.weight                [10240, 4096] (gelu branch)
#   blocks.{i}.ffn.fc1.weight                   [10240, 4096] (linear branch)
#   blocks.{i}.ffn.fc2.weight                   [4096, 10240] (down proj)
#   blocks.{i}.pos_embedding.embedding.weight   [32, 64]      (per-layer bias)
#   norm.weight                                 [4096]        (final norm)
#
# Key difference vs T5Encoder (t5_encoder.mojo):
#   Each of the 24 layers has its OWN position bias table
#   (blocks.{i}.pos_embedding.embedding.weight). T5 uses one shared table.
#   Bucket computation is identical (bidirectional, num_buckets=32, max_dist=128).
#   Bucket ids are built ONCE on host (shape S*S), then per-layer we gather
#   different rows and form [1, H, S, S] additive bias.
#
# Layer math (pre-norm residual, T5 style, scale=1.0):
#   n1  = rms_norm(hidden, norm1)
#   q,k,v = linear(n1, attn.{q,k,v}.weight)  -> [1,S,4096] -> [1,S,64,64]
#   attn  = sdpa[1,S,64,64](q, k, v, bias_i, scale=1.0)
#   ao    = linear(reshape(attn,[1,S,4096]), attn.o.weight)
#   h1    = hidden + ao
#   n2    = rms_norm(h1, norm2)
#   gate  = gelu(linear(n2, ffn.gate.0.weight))   # gelu branch
#   up    = linear(n2, ffn.fc1.weight)             # linear branch
#   ffn   = linear(gate * up, ffn.fc2.weight)
#   hidden = h1 + ffn
# Final: rms_norm(hidden, norm.weight).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in ops.

from std.math import log as flog
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.tensor_algebra import add, mul, reshape, permute, gather_rows


# ── Config ────────────────────────────────────────────────────────────────────
@fieldwise_init
struct Umt5Config(Copyable, Movable, ImplicitlyCopyable):
    """umt5-xxl encoder hyperparameters (NAVA text encoder)."""

    var vocab_size: Int
    var d_model: Int
    var num_layers: Int
    var d_ff: Int
    var num_heads: Int
    var d_kv: Int
    var num_buckets: Int
    var max_dist: Int
    var layer_norm_eps: Float32

    @staticmethod
    def umt5_xxl() -> Umt5Config:
        """umt5-xxl: 24 layers, 4096-dim, 64 heads, d_kv 64, d_ff 10240."""
        return Umt5Config(256384, 4096, 24, 10240, 64, 64, 32, 128, Float32(1e-6))


# ── relative position bucket (identical to T5, bidirectional) ─────────────────
# Same logic as t5_encoder._t5_relative_position_bucket. Copied here so this
# module is self-contained (no cross-module host-fn dependency).
def _t5_relative_position_bucket(
    relative_position: Int,
    bidirectional: Bool,
    num_buckets: Int,
    max_distance: Int,
) -> Int:
    var relative_buckets = 0
    var nb = num_buckets
    var rp: Int

    if bidirectional:
        nb = nb // 2
        if relative_position > 0:
            relative_buckets += nb
        if relative_position < 0:
            rp = -relative_position
        else:
            rp = relative_position
    else:
        if relative_position < 0:
            rp = -relative_position
        else:
            rp = 0

    var max_exact = nb // 2
    var is_small = rp < max_exact

    var bucket: Int
    if is_small:
        bucket = rp
    else:
        var val = Float64(max_exact) + (
            (flog(Float64(rp) / Float64(max_exact)))
            / (flog(Float64(max_distance) / Float64(max_exact)))
        ) * Float64(nb - max_exact)
        var b = Int(val)
        if b > nb - 1:
            b = nb - 1
        bucket = b

    return relative_buckets + bucket


# ── Umt5Encoder ───────────────────────────────────────────────────────────────
struct Umt5Encoder[S: Int = 42]:
    """umt5-xxl encoder. Owns all weights resident on GPU (ArcPointer because
    Tensor is Movable-not-Copyable). S (sequence length) is a comptime param
    so the comptime-shaped sdpa can be called.

    Key difference vs T5Encoder: each layer has its OWN position bias table
    (blocks.{i}.pos_embedding.embedding.weight [32,64]). Bucket ids are
    computed once then each layer gathers its own bias rows."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: Umt5Config

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: Umt5Config,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(
        path: String, config: Umt5Config, ctx: DeviceContext
    ) raises -> Umt5Encoder[Self.S]:
        """Load all weights from the umt5_xxl_enc.safetensors file."""
        var sharded = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return Umt5Encoder[Self.S](weights^, name_to_idx^, config)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing umt5 weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    # ── Build bucket ids (host; shape S*S) — done ONCE, reused per layer ──────
    def _bucket_ids(self) raises -> List[Int]:
        """Build the S*S list of relative-position bucket indices (host).
        rel = j - i (memory - context), bidirectional, num_buckets=32, max_dist=128.
        Returned as a flat List[Int] in row-major order."""
        var cfg = self.config
        comptime SS = Self.S
        var bucket_ids = List[Int]()
        for i in range(SS):
            for j in range(SS):
                var rel = j - i
                var b = _t5_relative_position_bucket(
                    rel, True, cfg.num_buckets, cfg.max_dist
                )
                bucket_ids.append(b)
        return bucket_ids^

    # ── Per-layer relative position bias -> [1, H, S, S] ─────────────────────
    def _layer_bias(
        self,
        layer_idx: Int,
        bucket_ids: List[Int],
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Build [1, num_heads, S, S] additive bias for layer `layer_idx`.
        Uses blocks.{i}.pos_embedding.embedding.weight [32, 64]."""
        var cfg = self.config
        comptime SS = Self.S
        var key = String("blocks.") + String(layer_idx) + String(".pos_embedding.embedding.weight")
        ref bias_weight = self._w(key)

        # gather_rows: [S*S, num_heads]
        var gathered = gather_rows(bias_weight, bucket_ids, ctx)
        # reshape [S, S, num_heads]
        var rsh = List[Int]()
        rsh.append(SS)
        rsh.append(SS)
        rsh.append(cfg.num_heads)
        var reshaped = reshape(gathered, rsh^, ctx)
        # permute [num_heads, S, S]  (axis k from perm[k])
        var perm = List[Int]()
        perm.append(2)
        perm.append(0)
        perm.append(1)
        var permuted = permute(reshaped, perm, ctx)  # [H, S, S]
        # unsqueeze front -> [1, H, S, S]
        var msh = List[Int]()
        msh.append(1)
        msh.append(cfg.num_heads)
        msh.append(SS)
        msh.append(SS)
        return reshape(permuted, msh^, ctx)

    # ── One encoder layer ─────────────────────────────────────────────────────
    def _layer(
        self,
        layer_idx: Int,
        hidden: Tensor,
        bucket_ids: List[Int],
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var h = cfg.num_heads
        var d = cfg.d_kv
        var eps = cfg.layer_norm_eps
        var p = String("blocks.") + String(layer_idx)
        comptime SS = Self.S

        # Build per-layer bias [1, H, S, S]
        var position_bias = self._layer_bias(layer_idx, bucket_ids, ctx)

        # --- Self-attention (pre-norm) ---
        var normed = rms_norm(hidden, self._w(p + ".norm1.weight"), eps, ctx)

        var q = linear(normed, self._w(p + ".attn.q.weight"), None, ctx)
        var k = linear(normed, self._w(p + ".attn.k.weight"), None, ctx)
        var v = linear(normed, self._w(p + ".attn.v.weight"), None, ctx)

        # reshape [1,S,H*d] -> [1,S,H,d]
        var bshd = List[Int]()
        bshd.append(1)
        bshd.append(SS)
        bshd.append(h)
        bshd.append(d)
        q = reshape(q, bshd.copy(), ctx)
        k = reshape(k, bshd.copy(), ctx)
        v = reshape(v, bshd^, ctx)

        # umt5: NO 1/sqrt(d) scaling (scale=1.0). position_bias is additive.
        var attn = sdpa[1, SS, 64, 64](q, k, v, position_bias, Float32(1.0), ctx)

        # [1,S,H,d] -> [1,S,H*d]
        var flat = List[Int]()
        flat.append(1)
        flat.append(SS)
        flat.append(h * d)
        attn = reshape(attn, flat^, ctx)

        var attn_out = linear(attn, self._w(p + ".attn.o.weight"), None, ctx)
        var h1 = add(hidden, attn_out, ctx)

        # --- Gated-GELU FFN (pre-norm) ---
        var normed2 = rms_norm(h1, self._w(p + ".norm2.weight"), eps, ctx)
        # gate.0 is the GELU branch, fc1 is the linear branch
        var gate_act = linear(normed2, self._w(p + ".ffn.gate.0.weight"), None, ctx)
        var up = linear(normed2, self._w(p + ".ffn.fc1.weight"), None, ctx)
        var gated = mul(gelu(gate_act, ctx), up, ctx)
        var ffn_out = linear(gated, self._w(p + ".ffn.fc2.weight"), None, ctx)
        return add(h1, ffn_out, ctx)

    # ── Full encode: token ids -> [1, S, 4096] hidden states ──────────────────
    def encode(self, token_ids: List[Int], ctx: DeviceContext) raises -> Tensor:
        """Encode `token_ids` (length == S) to [1, S, 4096] BF16 hidden states.
        The caller must ensure len(token_ids) == S (no padding applied here;
        NAVA passes exactly the 42 valid token ids)."""
        var cfg = self.config
        comptime SS = Self.S
        if len(token_ids) != SS:
            raise Error(
                String("Umt5Encoder.encode: expected ")
                + String(SS)
                + " token ids, got "
                + String(len(token_ids))
            )

        # 1. Token embeddings (no additive position embeddings; positions encoded
        #    via per-layer relative bias tables).
        ref embed_w = self._w(String("token_embedding.weight"))
        var emb = gather_rows(embed_w, token_ids, ctx)  # [S, 4096]
        var hsh = List[Int]()
        hsh.append(1)
        hsh.append(SS)
        hsh.append(cfg.d_model)
        var hidden = reshape(emb, hsh^, ctx)  # [1, S, 4096]

        # 2. Build bucket ids ONCE (host; shape S*S).
        var bucket_ids = self._bucket_ids()

        # 3. All 24 layers (each with its own position bias table).
        for i in range(cfg.num_layers):
            hidden = self._layer(i, hidden, bucket_ids, ctx)

        # 4. Final layer norm.
        return rms_norm(
            hidden, self._w(String("norm.weight")), cfg.layer_norm_eps, ctx
        )
