# models/dit/ltx2_dit.mojo — LTX-2 transformer block (video self-attn + FFN),
# pure-Mojo inference port. Block-0, video-only path.
#
# Port of inference-flame/src/models/ltx2_model.rs:
#   LTX2TransformerBlock::forward_video_only_with_skip (line 959)  — self-attn
#       (attn1) + cross-attn (attn2) + FFN.
#   LTX2Attention::forward_with_skip                    (line 765)  — qkv,
#       QK-RMSNorm, RoPE, SDPA, per-head gate, to_out.
#   compute_ada_params_6                                (line 1505) — modulation.
#
# *** SCOPE (per task): block-0 self-attention (attn1) + FFN ONLY. ***
# The Rust forward_video_only ALSO runs a text cross-attention (attn2) between
# self-attn and FFN. This Mojo port implements the self-attn + gated residual
# and the FFN + gated residual, and OMITS the cross-attn stage (no
# encoder_hidden_states in the bounded smoke). See report deviations.
#
# ── scale_shift_table chunk order (ltx2_model.rs:1533-1538) ──────────────────
#   ada = table[0:6] + temb            (broadcast add over tokens)
#   (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
#   i.e. SHIFT first, then SCALE, then GATE — per branch (msa = self-attn,
#   mlp = feed-forward). Modulation per branch:  norm(x) * (1 + scale) + shift.
#   Gated residual:  x = x + gate * branch_out.
#
# The real checkpoint scale_shift_table is [9, dim] (ComfyUI 9-param: rows 6-8
# are cross-attn modulation). We use only the first 6 rows here.
#
# ── attn1 math (ltx2_model.rs:786-882) ───────────────────────────────────────
#   q = to_q(mod_h)+bias ; k = to_k ; v = to_v          [1, N, inner_dim]
#   q = rms_norm(q, q_norm) ; k = rms_norm(k, k_norm)    (over FULL inner_dim)
#   reshape q,k,v -> [1, N, H, Dh] (BSHD)
#   q = rope(q) ; k = rope(k)                            (split RoPE, per head)
#   attn = sdpa(q, k, v)                                  full attention
#   if to_gate_logits present:  gates = 2*sigmoid(gate_logits) ; attn *= gates
#   out = to_out.0(attn_flat)+bias
#
# Weight keys are the diffusers names from the docstring (norm_q/norm_k). The
# ComfyUI checkpoint at .serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors
# uses q_norm/k_norm and model.diffusion_model.transformer_blocks.{i}. prefix;
# LTX2BlockWeights.load resolves BOTH naming conventions.
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.
# *** CODE-ONLY: compile-verified; NOT executed. ***

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import gelu, sigmoid
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.tensor_algebra import (
    reshape,
    add,
    mul,
    mul_scalar,
    add_scalar,
    slice,
    concat,
)
from serenitymojo.models.dit.ltx2_rope import apply_ltx2_rope
from serenitymojo.offload.ltx2_block_stream import FP8Block


# ── Config ────────────────────────────────────────────────────────────────────
@fieldwise_init
struct LTX2Config(Copyable, Movable, ImplicitlyCopyable):
    """LTX-2 video-stream hyperparameters (ltx2_model.rs:97-133 default)."""

    var num_layers: Int          # 48
    var num_heads: Int           # 32
    var head_dim: Int            # 128
    var inner_dim: Int           # 4096 (heads*head_dim)
    var ffn_hidden: Int          # 16384 (inner_dim * 4)
    var rope_theta: Float64      # 10000.0
    var eps: Float32             # 1e-6

    @staticmethod
    def ltx2() -> LTX2Config:
        return LTX2Config(
            48, 32, 128, 4096, 16384, Float64(10000.0), Float32(1e-6)
        )


# ── Block weights ─────────────────────────────────────────────────────────────
struct LTX2BlockWeights(Movable):
    """One block's video self-attn (attn1) + FFN weights + scale_shift_table.

    Owns weights as List[ArcPointer[Tensor]] looked up by canonical key
    (Tensor is Movable-not-Copyable). norm1 has no affine in LTX-2 (no
    norm1.weight in the checkpoint) so it is absent; the self-attn modulation
    applies plain RMSNorm (no weight) which we implement via a ones vector.
    to_gate_logits is optional (present in the ComfyUI 22B checkpoint)."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var has_gate: Bool
    var has_gate2: Bool
    var config: LTX2Config

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        has_gate: Bool,
        has_gate2: Bool,
        config: LTX2Config,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.has_gate = has_gate
        self.has_gate2 = has_gate2
        self.config = config

    @staticmethod
    def load(
        checkpoint_path: String,
        block_idx: Int,
        config: LTX2Config,
        ctx: DeviceContext,
    ) raises -> LTX2BlockWeights:
        """Load attn1 + ff + scale_shift_table for one block. Tries the
        diffusers prefix `transformer_blocks.{i}.` first, then the ComfyUI
        prefix `model.diffusion_model.transformer_blocks.{i}.`. QK-norm key is
        `norm_q`/`norm_k` (diffusers) or `q_norm`/`k_norm` (ComfyUI). All
        tensors uploaded as BF16 (scale_shift_table is F32 on disk)."""
        var st = ShardedSafeTensors.open(checkpoint_path)

        # Determine prefix.
        var pfx_diff = String("transformer_blocks.") + String(block_idx) + "."
        var pfx_comfy = (
            String("model.diffusion_model.transformer_blocks.")
            + String(block_idx)
            + "."
        )
        var prefix = pfx_diff
        if not _st_has(st, pfx_diff + "attn1.to_q.weight"):
            if _st_has(st, pfx_comfy + "attn1.to_q.weight"):
                prefix = pfx_comfy
            else:
                raise Error(
                    String("LTX2: block ")
                    + String(block_idx)
                    + " attn1.to_q.weight not found under either prefix"
                )

        # QK-norm naming (attn1).
        var qn = String("attn1.norm_q.weight")
        var kn = String("attn1.norm_k.weight")
        if not _st_has(st, prefix + qn):
            qn = String("attn1.q_norm.weight")
            kn = String("attn1.k_norm.weight")

        # QK-norm naming (attn2 cross-attn).
        var qn2 = String("attn2.norm_q.weight")
        var kn2 = String("attn2.norm_k.weight")
        if not _st_has(st, prefix + qn2):
            qn2 = String("attn2.q_norm.weight")
            kn2 = String("attn2.k_norm.weight")

        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        # canonical keys we look up later -> source key under the prefix
        var keys = List[Tuple[String, String]]()
        keys.append(("attn1.to_q.weight", "attn1.to_q.weight"))
        keys.append(("attn1.to_q.bias", "attn1.to_q.bias"))
        keys.append(("attn1.to_k.weight", "attn1.to_k.weight"))
        keys.append(("attn1.to_k.bias", "attn1.to_k.bias"))
        keys.append(("attn1.to_v.weight", "attn1.to_v.weight"))
        keys.append(("attn1.to_v.bias", "attn1.to_v.bias"))
        keys.append(("attn1.norm_q.weight", qn))
        keys.append(("attn1.norm_k.weight", kn))
        keys.append(("attn1.to_out.0.weight", "attn1.to_out.0.weight"))
        keys.append(("attn1.to_out.0.bias", "attn1.to_out.0.bias"))
        keys.append(("ff.net.0.proj.weight", "ff.net.0.proj.weight"))
        keys.append(("ff.net.0.proj.bias", "ff.net.0.proj.bias"))
        keys.append(("ff.net.2.weight", "ff.net.2.weight"))
        keys.append(("ff.net.2.bias", "ff.net.2.bias"))
        keys.append(("scale_shift_table", "scale_shift_table"))

        # attn2 cross-attention (text). Full sibling of attn1, ALSO gated.
        # KV context modulated by prompt_scale_shift_table [2, dim].
        keys.append(("attn2.to_q.weight", "attn2.to_q.weight"))
        keys.append(("attn2.to_q.bias", "attn2.to_q.bias"))
        keys.append(("attn2.to_k.weight", "attn2.to_k.weight"))
        keys.append(("attn2.to_k.bias", "attn2.to_k.bias"))
        keys.append(("attn2.to_v.weight", "attn2.to_v.weight"))
        keys.append(("attn2.to_v.bias", "attn2.to_v.bias"))
        keys.append(("attn2.norm_q.weight", qn2))
        keys.append(("attn2.norm_k.weight", kn2))
        keys.append(("attn2.to_out.0.weight", "attn2.to_out.0.weight"))
        keys.append(("attn2.to_out.0.bias", "attn2.to_out.0.bias"))
        keys.append(("prompt_scale_shift_table", "prompt_scale_shift_table"))

        for ref kv in keys:
            var canon = kv[0]
            var src = prefix + kv[1]
            var tv = st.tensor_view(src)
            var t = Tensor.from_view_as_bf16(tv, ctx)
            name_to_idx[canon] = len(weights)
            weights.append(ArcPointer(t^))

        # Optional per-head gate logits.
        var has_gate = False
        if _st_has(st, prefix + "attn1.to_gate_logits.weight"):
            has_gate = True
            var gw = st.tensor_view(prefix + "attn1.to_gate_logits.weight")
            name_to_idx["attn1.to_gate_logits.weight"] = len(weights)
            weights.append(ArcPointer(Tensor.from_view_as_bf16(gw, ctx)))
            var gb = st.tensor_view(prefix + "attn1.to_gate_logits.bias")
            name_to_idx["attn1.to_gate_logits.bias"] = len(weights)
            weights.append(ArcPointer(Tensor.from_view_as_bf16(gb, ctx)))

        # Optional attn2 per-head gate logits (present in ComfyUI 22B).
        var has_gate2 = False
        if _st_has(st, prefix + "attn2.to_gate_logits.weight"):
            has_gate2 = True
            var gw2 = st.tensor_view(prefix + "attn2.to_gate_logits.weight")
            name_to_idx["attn2.to_gate_logits.weight"] = len(weights)
            weights.append(ArcPointer(Tensor.from_view_as_bf16(gw2, ctx)))
            var gb2 = st.tensor_view(prefix + "attn2.to_gate_logits.bias")
            name_to_idx["attn2.to_gate_logits.bias"] = len(weights)
            weights.append(ArcPointer(Tensor.from_view_as_bf16(gb2, ctx)))

        return LTX2BlockWeights(
            weights^, name_to_idx^, has_gate, has_gate2, config
        )

    @staticmethod
    def from_fp8_block(
        var block: FP8Block,
        config: LTX2Config,
        ctx: DeviceContext,
    ) raises -> LTX2BlockWeights:
        """Build block weights from an FP8-streamed, already-dequantized block.

        `block` is an `LTX2BlockStream.load_block_bf16` result: prefix-stripped
        canonical names (e.g. `attn1.to_q.weight`, `attn1.q_norm.weight`,
        `ff.net.0.proj.weight`) mapping to BF16 device Tensors — FP8 weights
        already dequantized on-use with their per-tensor weight_scale, BF16/F32
        tensors upcast. This rehomes them into the `LTX2BlockWeights` layout the
        block forward expects, mapping the ComfyUI QK-norm names
        (`q_norm`/`k_norm`) to the forward's canonical `norm_q`/`norm_k`.

        No re-read from disk and no second dequant: the same device Tensors are
        moved in (ArcPointer move), so this is zero-copy on top of the stream's
        single dequant. Drives the P2 spike: stream → dequant → real forward."""
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        # (canonical-key-the-forward-wants, source-key-in-the-streamed-block)
        var keys = List[Tuple[String, String]]()
        keys.append(("attn1.to_q.weight", "attn1.to_q.weight"))
        keys.append(("attn1.to_q.bias", "attn1.to_q.bias"))
        keys.append(("attn1.to_k.weight", "attn1.to_k.weight"))
        keys.append(("attn1.to_k.bias", "attn1.to_k.bias"))
        keys.append(("attn1.to_v.weight", "attn1.to_v.weight"))
        keys.append(("attn1.to_v.bias", "attn1.to_v.bias"))
        keys.append(("attn1.to_out.0.weight", "attn1.to_out.0.weight"))
        keys.append(("attn1.to_out.0.bias", "attn1.to_out.0.bias"))
        keys.append(("ff.net.0.proj.weight", "ff.net.0.proj.weight"))
        keys.append(("ff.net.0.proj.bias", "ff.net.0.proj.bias"))
        keys.append(("ff.net.2.weight", "ff.net.2.weight"))
        keys.append(("ff.net.2.bias", "ff.net.2.bias"))
        keys.append(("scale_shift_table", "scale_shift_table"))
        keys.append(("attn2.to_q.weight", "attn2.to_q.weight"))
        keys.append(("attn2.to_q.bias", "attn2.to_q.bias"))
        keys.append(("attn2.to_k.weight", "attn2.to_k.weight"))
        keys.append(("attn2.to_k.bias", "attn2.to_k.bias"))
        keys.append(("attn2.to_v.weight", "attn2.to_v.weight"))
        keys.append(("attn2.to_v.bias", "attn2.to_v.bias"))
        keys.append(("attn2.to_out.0.weight", "attn2.to_out.0.weight"))
        keys.append(("attn2.to_out.0.bias", "attn2.to_out.0.bias"))
        keys.append(("prompt_scale_shift_table", "prompt_scale_shift_table"))

        # QK-norm: forward wants norm_q/norm_k; streamed block has q_norm/k_norm
        # (ComfyUI) or norm_q/norm_k (diffusers). Resolve per block.
        var qn = String("attn1.norm_q.weight")
        var kn = String("attn1.norm_k.weight")
        if "attn1.q_norm.weight" in block:
            qn = String("attn1.q_norm.weight")
            kn = String("attn1.k_norm.weight")
        keys.append(("attn1.norm_q.weight", qn))
        keys.append(("attn1.norm_k.weight", kn))
        var qn2 = String("attn2.norm_q.weight")
        var kn2 = String("attn2.norm_k.weight")
        if "attn2.q_norm.weight" in block:
            qn2 = String("attn2.q_norm.weight")
            kn2 = String("attn2.k_norm.weight")
        keys.append(("attn2.norm_q.weight", qn2))
        keys.append(("attn2.norm_k.weight", kn2))

        for ref kv in keys:
            var canon = kv[0]
            var src = kv[1]
            if src not in block:
                raise Error(
                    String("from_fp8_block: streamed block missing ") + src
                )
            name_to_idx[canon] = len(weights)
            weights.append(block[src])

        var has_gate = False
        if "attn1.to_gate_logits.weight" in block:
            has_gate = True
            name_to_idx["attn1.to_gate_logits.weight"] = len(weights)
            weights.append(block["attn1.to_gate_logits.weight"])
            name_to_idx["attn1.to_gate_logits.bias"] = len(weights)
            weights.append(block["attn1.to_gate_logits.bias"])

        var has_gate2 = False
        if "attn2.to_gate_logits.weight" in block:
            has_gate2 = True
            name_to_idx["attn2.to_gate_logits.weight"] = len(weights)
            weights.append(block["attn2.to_gate_logits.weight"])
            name_to_idx["attn2.to_gate_logits.bias"] = len(weights)
            weights.append(block["attn2.to_gate_logits.bias"])

        return LTX2BlockWeights(
            weights^, name_to_idx^, has_gate, has_gate2, config
        )

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("LTX2: missing weight ") + name)
        return self.weights[self.name_to_idx[name]][]

    def _clone(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    def _linear_b(
        self, x: Tensor, w_key: String, b_key: String, ctx: DeviceContext
    ) raises -> Tensor:
        ref w = self._w(w_key)
        ref b = self._w(b_key)
        return linear(x, w, Optional[Tensor](self._clone(b, ctx)), ctx)


# ── module-level helpers ──────────────────────────────────────────────────────
def _st_has(st: ShardedSafeTensors, name: String) -> Bool:
    for ref nm in st.names():
        if nm == name:
            return True
    return False


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^


# RMSNorm with NO affine (ones weight) over the last dim.
def _rms_norm_no_affine(
    x: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    var d = x.shape()[len(x.shape()) - 1]
    var ones = List[Float32]()
    for _ in range(d):
        ones.append(Float32(1.0))
    var w = Tensor.from_host(ones, _shape1(d), x.dtype(), ctx)
    return rms_norm(x, w, eps, ctx)


# AdaLN modulate: normed * (1 + scale) + shift.  scale/shift are [1, dim]
# vectors broadcasting against [1, N, dim].
def _modulate(
    normed: Tensor, scale: Tensor, shift: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var one_plus = add_scalar(scale, Float32(1.0), ctx)
    return add(mul(normed, one_plus, ctx), shift, ctx)


# Extract row `idx` of the [6, dim] scale_shift_table as [1, 1, dim], add the
# matching temb chunk (temb is [1, 6*dim]) -> the modulation vector [1, dim].
#   ada_row = table[idx, :] + temb[:, idx*dim : (idx+1)*dim]
def _ada_param(
    table: Tensor,  # [9, dim] (we use rows 0..5), BF16
    temb: Tensor,   # [1, 6*dim], BF16
    idx: Int,
    dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var trow = slice(table, 0, idx, 1, ctx)          # [1, dim]
    var tchunk = slice(temb, 1, idx * dim, dim, ctx)  # [1, dim]
    return add(trow, tchunk, ctx)


# Reshape [1, N, H*Dh] <-> [1, N, H, Dh].
def _to_bshd(x: Tensor, n: Int, h: Int, dh: Int, ctx: DeviceContext) raises -> Tensor:
    return reshape(x, _shape4(1, n, h, dh), ctx)


def _from_bshd(x: Tensor, n: Int, h: Int, dh: Int, ctx: DeviceContext) raises -> Tensor:
    return reshape(x, _shape3(1, n, h * dh), ctx)


# Per-head attention gate (ltx2_model.rs:858-872): given the modulated attention
# INPUT (`mod_in`, the same tensor fed to to_q/k/v), compute
#   gate_logits = linear(mod_in, gate_w, gate_b)   -> [1, S, H]
#   gates       = 2 * sigmoid(gate_logits)         -> [1, S, H]
# then broadcast-multiply the flat attention output [1, S, H*Dh] by gates: we
# reshape attn to [1, S, H, 1] broadcast partner and use the tensor_algebra
# `mul` broadcast (gates -> [1, S, H, 1], attn4 -> [1, S, H, Dh]). Returns the
# gated, flattened [1, S, H*Dh]. Applied BEFORE to_out.
def _apply_head_gate[S: Int](
    attn_flat: Tensor,   # [1, S, H*Dh]
    mod_in: Tensor,      # [1, S, H*Dh] modulated attention input (gate logits src)
    gate_w: Tensor,      # [H, H*Dh]
    var gate_b: Tensor,  # [H] (owned clone; transferred into the bias Optional)
    num_heads: Int,
    head_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var gate_logits = linear(mod_in, gate_w, Optional[Tensor](gate_b^), ctx)  # [1,S,H]
    var gates = mul_scalar(sigmoid(gate_logits, ctx), Float32(2.0), ctx)     # [1,S,H]
    var gates4 = reshape(gates, _shape4(1, S, num_heads, 1), ctx)            # [1,S,H,1]
    var attn4 = reshape(attn_flat, _shape4(1, S, num_heads, head_dim), ctx)  # [1,S,H,Dh]
    var gated = mul(attn4, gates4, ctx)                                      # broadcast
    return reshape(gated, _shape3(1, S, num_heads * head_dim), ctx)


# Additive cross-attention mask [1, H, S, S]: 0.0 for the first `n_kv` KV columns
# (the real text tokens), large-negative for the padded columns [n_kv, S). This
# lets us reuse the square `sdpa` for a non-square Q(S) x KV(n_kv) cross-attn by
# zero-padding the K/V seq up to S and masking out the pad.
def _cross_pad_mask[S: Int](
    heads: Int, n_kv: Int, dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var neg = Float32(-1.0e30)
    var data = List[Float32]()
    for _ in range(heads):
        for _ in range(S):          # query row
            for j in range(S):      # key col
                if j < n_kv:
                    data.append(Float32(0.0))
                else:
                    data.append(neg)
    return Tensor.from_host(data, _shape4(1, heads, S, S), dtype, ctx)


# ── block forward (video self-attn + cross-attn + FFN, video-only) ────────────
# hidden:    [1, S, inner_dim] BF16
# temb:      [1, 9*inner_dim]  BF16   (timestep embedding, pre-broadcast;
#            rows 0-5 self-attn+FFN, rows 6-8 cross-attn query modulation)
# context:   [1, N_TXT, inner_dim] BF16   (text encoder hidden states for attn2)
# rope_cos/sin: [S*num_heads, head_dim/2] BF16   (from build_ltx2_rope)
def ltx2_block_forward_video_only[B: Int, S: Int, N_TXT: Int](
    weights: LTX2BlockWeights,
    hidden: Tensor,
    temb: Tensor,
    context: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    num_heads: Int,
    head_dim: Int,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime assert B == 1, "LTX2 block smoke supports B==1"
    comptime assert N_TXT <= S, "cross-attn pad-mask requires N_TXT <= S"
    var dim = num_heads * head_dim
    var scale = Float32(1.0) / sqrt(Float32(head_dim))

    # ── modulation params: scale_shift_table[0:6] + temb (ltx2_model.rs:1533) ──
    ref table = weights._w("scale_shift_table")
    var shift_msa = _ada_param(table, temb, 0, dim, ctx)
    var scale_msa = _ada_param(table, temb, 1, dim, ctx)
    var gate_msa = _ada_param(table, temb, 2, dim, ctx)
    var shift_mlp = _ada_param(table, temb, 3, dim, ctx)
    var scale_mlp = _ada_param(table, temb, 4, dim, ctx)
    var gate_mlp = _ada_param(table, temb, 5, dim, ctx)
    # cross-attn query modulation: rows 6-8 (shift_ca_q, scale_ca_q, gate_ca)
    # (ltx2_model.rs:1024, compute_ada_params_ca def 1563-1574).
    var shift_ca = _ada_param(table, temb, 6, dim, ctx)
    var scale_ca = _ada_param(table, temb, 7, dim, ctx)
    var gate_ca = _ada_param(table, temb, 8, dim, ctx)

    # ── 1. self-attn (attn1) with AdaLN ──
    var norm_h = _rms_norm_no_affine(hidden, eps, ctx)
    var mod_h = _modulate(norm_h, scale_msa, shift_msa, ctx)  # [1, S, dim]

    var q = weights._linear_b(mod_h, "attn1.to_q.weight", "attn1.to_q.bias", ctx)
    var k = weights._linear_b(mod_h, "attn1.to_k.weight", "attn1.to_k.bias", ctx)
    var v = weights._linear_b(mod_h, "attn1.to_v.weight", "attn1.to_v.bias", ctx)

    # QK RMSNorm over the FULL inner_dim (ltx2_model.rs:797-798).
    ref nq = weights._w("attn1.norm_q.weight")
    ref nk = weights._w("attn1.norm_k.weight")
    q = rms_norm(q, nq, eps, ctx)
    k = rms_norm(k, nk, eps, ctx)

    # reshape to BSHD [1, S, H, Dh]
    var q4 = _to_bshd(q, S, num_heads, head_dim, ctx)
    var k4 = _to_bshd(k, S, num_heads, head_dim, ctx)
    var v4 = _to_bshd(v, S, num_heads, head_dim, ctx)

    # split RoPE on q,k (rows are (token, head); table matches)
    q4 = apply_ltx2_rope(q4, rope_cos, rope_sin, ctx)
    k4 = apply_ltx2_rope(k4, rope_cos, rope_sin, ctx)

    var attn = sdpa[B, S, 32, 128](q4, k4, v4, _zeros_mask[S](32, hidden.dtype(), ctx), scale, ctx)
    var attn_flat = _from_bshd(attn, S, num_heads, head_dim, ctx)  # [1, S, dim]

    # Per-head gate (ltx2_model.rs:858-872): gates = 2*sigmoid(to_gate_logits(
    # mod_h)); attn *= gates per head; THEN to_out. The gate logit input is the
    # MODULATED attention input `mod_h` (the same tensor fed to to_q/k/v, the
    # `hidden_states` arg seen inside forward_with_skip).
    if weights.has_gate:
        attn_flat = _apply_head_gate[S](
            attn_flat,
            mod_h,
            weights._w("attn1.to_gate_logits.weight"),
            weights._clone(weights._w("attn1.to_gate_logits.bias"), ctx),
            num_heads,
            head_dim,
            ctx,
        )

    var attn_out = weights._linear_b(
        attn_flat, "attn1.to_out.0.weight", "attn1.to_out.0.bias", ctx
    )
    # gated residual: hidden + gate_msa * attn_out (ltx2_model.rs:1018)
    var hs = add(hidden, mul(gate_msa, attn_out, ctx), ctx)

    # ── 2. cross-attn (attn2, text) — ltx2_model.rs:1021-1069 (9-param) ──
    # Query: modulate hs by (scale_ca, shift_ca) (rows 7,6). No norm2 affine.
    var norm_q2 = _rms_norm_no_affine(hs, eps, ctx)
    var mod_q2 = _modulate(norm_q2, scale_ca, shift_ca, ctx)        # [1, S, dim]
    # Context (KV): the bounded smoke has no prompt_timestep, so per Rust
    # (1051-1052) the modulated context == raw encoder_hidden_states (`context`).
    var q2 = weights._linear_b(mod_q2, "attn2.to_q.weight", "attn2.to_q.bias", ctx)
    var k2 = weights._linear_b(context, "attn2.to_k.weight", "attn2.to_k.bias", ctx)
    var v2 = weights._linear_b(context, "attn2.to_v.weight", "attn2.to_v.bias", ctx)

    # attn2 has its OWN QK-RMSNorm over full inner_dim (no per-head). NO RoPE on
    # cross-attn (text K/V have no spatial position) — ltx2_model.rs:1056 passes
    # None for query/key rope.
    q2 = rms_norm(q2, weights._w("attn2.norm_q.weight"), eps, ctx)
    k2 = rms_norm(k2, weights._w("attn2.norm_k.weight"), eps, ctx)

    # Non-square Q(S) x KV(N_TXT): pad K/V seq up to S with zeros and mask the
    # pad columns (large-negative additive bias) so the square sdpa is exact.
    var q2_4 = _to_bshd(q2, S, num_heads, head_dim, ctx)            # [1, S, H, Dh]
    var k2_4 = _to_bshd(k2, N_TXT, num_heads, head_dim, ctx)        # [1, N_TXT, H, Dh]
    var v2_4 = _to_bshd(v2, N_TXT, num_heads, head_dim, ctx)
    var kv_pad = _zeros4[S](N_TXT, num_heads, head_dim, hidden.dtype(), ctx)
    var k2_pad = concat(1, ctx, k2_4, kv_pad)                       # [1, S, H, Dh]
    var v2_pad = concat(1, ctx, v2_4, _clone_t(kv_pad, ctx))
    var cmask = _cross_pad_mask[S](num_heads, N_TXT, hidden.dtype(), ctx)
    var attn2 = sdpa[B, S, 32, 128](q2_4, k2_pad, v2_pad, cmask, scale, ctx)
    var attn2_flat = _from_bshd(attn2, S, num_heads, head_dim, ctx)  # [1, S, dim]

    # attn2's OWN per-head gate; gate input = the modulated query mod_q2.
    if weights.has_gate2:
        attn2_flat = _apply_head_gate[S](
            attn2_flat,
            mod_q2,
            weights._w("attn2.to_gate_logits.weight"),
            weights._clone(weights._w("attn2.to_gate_logits.bias"), ctx),
            num_heads,
            head_dim,
            ctx,
        )

    var ca_out = weights._linear_b(
        attn2_flat, "attn2.to_out.0.weight", "attn2.to_out.0.bias", ctx
    )
    # gated residual: hs + gate_ca * ca_out (ltx2_model.rs:1060)
    hs = add(hs, mul(gate_ca, ca_out, ctx), ctx)

    # ── 3. FFN with AdaLN (ltx2_model.rs:1074-1086) ──
    var norm_ff = _rms_norm_no_affine(hs, eps, ctx)
    var mod_ff = _modulate(norm_ff, scale_mlp, shift_mlp, ctx)
    var ff = weights._linear_b(
        mod_ff, "ff.net.0.proj.weight", "ff.net.0.proj.bias", ctx
    )
    ff = gelu(ff, ctx)  # GELU-approximate (tanh) — ltx2_model.rs:279-281
    ff = weights._linear_b(ff, "ff.net.2.weight", "ff.net.2.bias", ctx)
    hs = add(hs, mul(gate_mlp, ff, ctx), ctx)

    return hs^


# All-zeros additive attention mask [1, H, S, S] for full attention (the Mojo
# sdpa requires a mask; Rust passes None). Built once per block here.
def _zeros_mask[S: Int](
    heads: Int, dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var data = List[Float32]()
    for _ in range(heads * S * S):
        data.append(Float32(0.0))
    return Tensor.from_host(data, _shape4(1, heads, S, S), dtype, ctx)


# Zero pad block [1, S-n_kv, H, Dh] to extend cross-attn K/V seq to S.
def _zeros4[S: Int](
    n_kv: Int, h: Int, dh: Int, dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var pad = S - n_kv
    var data = List[Float32]()
    for _ in range(pad * h * dh):
        data.append(Float32(0.0))
    return Tensor.from_host(data, _shape4(1, pad, h, dh), dtype, ctx)


# D2D buffer clone (Tensor is uniquely-owning; concat consumes by reference but
# we need two independent pad tensors).
def _clone_t(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())
