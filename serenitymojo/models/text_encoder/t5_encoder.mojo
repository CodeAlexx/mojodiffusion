# models/text_encoder/t5_encoder.mojo — T5-XXL v1.1 encoder (FLUX.1 text encoder).
#
# Pure-Mojo, inference-only port of
#   /home/alex/EriDiffusion/inference-flame/src/models/t5_encoder.rs (read FULL, 477 L)
# which itself mirrors HF transformers/models/t5/modeling_t5.py.
#
# Architecture (T5 v1.1 XXL encoder-only):
#   - 24 layers, d_model=4096, 64 heads, d_kv=64 (head*kv = 4096)
#   - Gated-GELU FFN: gelu(wi_0(x)) * wi_1(x) -> wo   (d_ff=10240)
#   - T5 LayerNorm == RMSNorm (no bias, no mean subtraction, standard formulation)
#   - Relative position bias (computed once from layer-0 weight, shared all layers)
#   - NO position embeddings (positions encoded via the relative bias)
#   - NO biases on attention or FFN projections
#   - Attention does NOT scale Q*K^T (the 1/sqrt(d) is absorbed into q_proj init,
#     Mesh-TF style). So scale = 1.0. position_bias is an ADDITIVE float, fed as
#     the foundation sdpa mask [1, H, S, S].
#
# Weight key format (t5xxl safetensors):
#   encoder.embed_tokens.weight   ([32128,4096])  (or `shared.weight`)
#   encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight   [4096,4096]
#   encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight [32,64]
#   encoder.block.{i}.layer.0.layer_norm.weight   [4096]
#   encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight  [10240,4096] (gate)
#   encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight  [10240,4096] (up)
#   encoder.block.{i}.layer.1.DenseReluDense.wo.weight    [4096,10240] (down)
#   encoder.block.{i}.layer.1.layer_norm.weight   [4096]
#   encoder.final_layer_norm.weight   [4096]
#
# Foundation ops reused: rms_norm (T5 LayerNorm), linear (no bias), sdpa (additive
# bias mask, scale=1.0), gelu (tanh-approx; T5 v1.1 uses gelu_new == tanh GELU),
# tensor_algebra.{add, mul, reshape, gather_rows}. T5-LOCAL glue (not foundation):
#   relative-position-bucket bias table (host-built bucket ids -> gather rows ->
#   permute to [1,H,S,S] additive mask). No foundation op for relative bucketing.
#
# Sequence length S is a comptime param on the encoder struct so the comptime-
# shaped sdpa can be called. FLUX.1 pads/truncates T5 to S=512 (pad id 0); no
# attention mask beyond the relative bias (BFL passes attention_mask=None).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation.

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
from serenitymojo.ops.cast import cast_tensor


# ── Config ────────────────────────────────────────────────────────────────────
@fieldwise_init
struct T5Config(Copyable, Movable, ImplicitlyCopyable):
    """T5-XXL v1.1 encoder hyperparameters (FLUX.1 text encoder)."""

    var vocab_size: Int
    var d_model: Int
    var num_layers: Int
    var d_ff: Int
    var num_heads: Int
    var d_kv: Int
    var relative_attention_num_buckets: Int
    var relative_attention_max_distance: Int
    var layer_norm_eps: Float32
    var max_seq_len: Int

    @staticmethod
    def t5_xxl() -> T5Config:
        """T5-XXL v1.1: 24 layers, 4096-dim, 64 heads, d_kv 64, d_ff 10240."""
        return T5Config(32128, 4096, 24, 10240, 64, 64, 32, 128, Float32(1e-6), 512)


# ── relative position bucket (HF _relative_position_bucket, bidirectional) ───
# Verbatim from t5_encoder.rs:404-442. Host-side scalar computation.
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
        # abs(relative_position)
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


# ── T5 Encoder ────────────────────────────────────────────────────────────────
struct T5Encoder[S: Int = 512]:
    """T5-XXL encoder. Owns all `encoder.*` weights resident on GPU (ArcPointer
    because Tensor is Movable-not-Copyable). S (sequence length) is a comptime
    param so the comptime-shaped sdpa can be called (FLUX.1 uses S=512)."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: T5Config

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: T5Config,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(
        dir_or_file: String, config: T5Config, ctx: DeviceContext
    ) raises -> T5Encoder[Self.S]:
        """Load all `encoder.*` tensors from a T5 safetensors file/dir. If the
        checkpoint exposes the embed table as `shared.weight` (not
        `encoder.embed_tokens.weight`), alias it. Non-`encoder.*` keys are skipped
        (the decoder, which FLUX.1 does not use)."""
        var sharded = ShardedSafeTensors.open(dir_or_file)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        var has_embed = False
        var shared_idx = -1
        for ref nm in sharded.names():
            var keep = nm.startswith("encoder.")
            if keep:
                var tv = sharded.tensor_view(nm)
                # Cast to BF16: the t5xxl_fp16 file stores weights in fp16, but
                # T5-XXL's residual stream grows to tens of thousands by mid-stack
                # and OVERFLOWS fp16 (max 65504) → inf → NaN (measured: residual
                # hit ±inf at layer 10). bf16 (range ~3e38) absorbs it; matches the
                # Rust reference's stated "BF16 storage, F32 accumulation".
                var t = cast_tensor(Tensor.from_view(tv, ctx), STDtype.BF16, ctx)
                var idx = len(weights)
                weights.append(ArcPointer(t^))
                name_to_idx[nm] = idx
                if nm == "encoder.embed_tokens.weight":
                    has_embed = True
            elif nm == "shared.weight":
                var tv = sharded.tensor_view(nm)
                var t = cast_tensor(Tensor.from_view(tv, ctx), STDtype.BF16, ctx)
                shared_idx = len(weights)
                weights.append(ArcPointer(t^))
                name_to_idx[String("shared.weight")] = shared_idx
        # Alias shared.weight -> encoder.embed_tokens.weight if needed.
        if not has_embed:
            if shared_idx < 0:
                raise Error(
                    "T5Encoder.load: neither encoder.embed_tokens.weight nor"
                    " shared.weight present"
                )
            name_to_idx[String("encoder.embed_tokens.weight")] = shared_idx
        return T5Encoder[Self.S](weights^, name_to_idx^, config)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing T5 weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    # ── relative position bias -> additive mask [1, H, S, S] ────────────────
    # bias_weight: [num_buckets=32, num_heads=64]. Build per-(i,j) bucket id,
    # gather the [seq*seq, H] rows, reshape [S,S,H] then permute to [1,H,S,S].
    # (t5_encoder.rs:177-231 compute_relative_bias.)
    def _relative_bias(self, ctx: DeviceContext) raises -> Tensor:
        var cfg = self.config
        ref bias_weight = self._w(
            String("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
        )
        var num_heads = bias_weight.shape()[1]
        comptime SS = Self.S

        var bucket_ids = List[Int]()
        for i in range(SS):
            for j in range(SS):
                var relative_position = j - i  # memory - context
                var bucket = _t5_relative_position_bucket(
                    relative_position,
                    True,  # bidirectional (encoder)
                    cfg.relative_attention_num_buckets,
                    cfg.relative_attention_max_distance,
                )
                bucket_ids.append(bucket)

        # gather rows: [S*S, num_heads]
        var gathered = gather_rows(bias_weight, bucket_ids, ctx)
        # reshape [S, S, num_heads]
        var rsh = List[Int]()
        rsh.append(SS)
        rsh.append(SS)
        rsh.append(num_heads)
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
        msh.append(num_heads)
        msh.append(SS)
        msh.append(SS)
        return reshape(permuted, msh^, ctx)

    # ── one encoder layer ────────────────────────────────────────────────────
    def _layer(
        self,
        layer_idx: Int,
        hidden: Tensor,
        position_bias: Tensor,  # [1, H, S, S]
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var h = cfg.num_heads
        var d = cfg.d_kv
        var eps = cfg.layer_norm_eps
        var p = String("encoder.block.") + String(layer_idx)
        comptime SS = Self.S

        # --- Self-attention ---
        var normed = rms_norm(
            hidden, self._w(p + ".layer.0.layer_norm.weight"), eps, ctx
        )

        var q = linear(normed, self._w(p + ".layer.0.SelfAttention.q.weight"), None, ctx)
        var k = linear(normed, self._w(p + ".layer.0.SelfAttention.k.weight"), None, ctx)
        var v = linear(normed, self._w(p + ".layer.0.SelfAttention.v.weight"), None, ctx)

        # reshape [1,S,H*d] -> BSHD [1,S,H,d]  (foundation sdpa wants BSHD)
        var bshd = List[Int]()
        bshd.append(1)
        bshd.append(SS)
        bshd.append(h)
        bshd.append(d)
        q = reshape(q, bshd.copy(), ctx)
        k = reshape(k, bshd.copy(), ctx)
        v = reshape(v, bshd^, ctx)

        # T5: NO 1/sqrt(d) scaling (scale=1.0). position_bias is additive.
        # Dh=64 -> flash path. num_heads=64 comptime.
        var attn = sdpa[1, SS, 64, 64](q, k, v, position_bias, Float32(1.0), ctx)

        # [1,S,H,d] -> [1,S,H*d]
        var flat = List[Int]()
        flat.append(1)
        flat.append(SS)
        flat.append(h * d)
        attn = reshape(attn, flat^, ctx)

        var attn_out = linear(
            attn, self._w(p + ".layer.0.SelfAttention.o.weight"), None, ctx
        )
        var h1 = add(hidden, attn_out, ctx)

        # --- Gated-GELU FFN ---
        var normed2 = rms_norm(h1, self._w(p + ".layer.1.layer_norm.weight"), eps, ctx)
        var gate = linear(
            normed2, self._w(p + ".layer.1.DenseReluDense.wi_0.weight"), None, ctx
        )
        var up = linear(
            normed2, self._w(p + ".layer.1.DenseReluDense.wi_1.weight"), None, ctx
        )
        var gated = mul(gelu(gate, ctx), up, ctx)
        var ffn_out = linear(
            gated, self._w(p + ".layer.1.DenseReluDense.wo.weight"), None, ctx
        )
        return add(h1, ffn_out, ctx)

    # ── full encode: token ids -> [1, S, 4096] hidden states ─────────────────
    # token_ids are right-padded/truncated to S (pad id 0) by the CALLER (the
    # pipeline applies BFL's padding="max_length", max_length=512). This forward
    # consumes exactly S ids.
    def encode(self, token_ids: List[Int], ctx: DeviceContext) raises -> Tensor:
        var cfg = self.config
        comptime SS = Self.S
        if len(token_ids) != SS:
            raise Error(
                String("T5Encoder.encode: expected ")
                + String(SS)
                + " token ids (pad to max_seq_len first), got "
                + String(len(token_ids))
            )

        # 1. token embeddings (no position embeddings; T5 uses relative bias).
        ref embed_w = self._w(String("encoder.embed_tokens.weight"))
        var emb = gather_rows(embed_w, token_ids, ctx)  # [S, 4096]
        var hsh = List[Int]()
        hsh.append(1)
        hsh.append(SS)
        hsh.append(cfg.d_model)
        var hidden = reshape(emb, hsh^, ctx)  # [1, S, 4096]

        # 2. relative position bias (shared across all layers).
        var position_bias = self._relative_bias(ctx)

        # 3. all layers.
        for i in range(cfg.num_layers):
            hidden = self._layer(i, hidden, position_bias, ctx)

        # 4. final layer norm.
        return rms_norm(
            hidden, self._w(String("encoder.final_layer_norm.weight")), cfg.layer_norm_eps, ctx
        )
