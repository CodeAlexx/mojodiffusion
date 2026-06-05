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
# *** SCOPE: LTX-2 video-only and AV block inference paths. ***
# The video-only path includes attn1, text cross-attn attn2, and FFN. The AV
# path includes video/audio self-attn, text cross-attn, and cross-modal A2V/V2A
# attention.
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
from serenitymojo.ops.attention import (
    sdpa_nomask,
    sdpa_nomask_tiled,
    sdpa_cross_nomask,
)
from serenitymojo.ops.tensor_algebra import (
    reshape,
    add,
    mul,
    mul_scalar,
    add_scalar,
    slice,
)
from serenitymojo.models.dit.ltx2_rope import apply_ltx2_rope
from serenitymojo.models.dit.ltx2_nag import NAGContext
from serenitymojo.offload.ltx2_block_stream import FP8Block
from serenitymojo.ops.cast import cast_tensor


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

    var attn: Tensor
    comptime score_mib = (B * S * S * 32 * 4) // (1024 * 1024)
    comptime if score_mib >= 3584:
        attn = sdpa_nomask_tiled[B, S, 32, 128](q4, k4, v4, scale, ctx)
    else:
        attn = sdpa_nomask[B, S, 32, 128](q4, k4, v4, scale, ctx)
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

    # Non-square Q(S) x KV(N_TXT): rectangular online-softmax attention.
    # This is exact and avoids the old square pad/mask [S,S] allocation.
    var q2_4 = _to_bshd(q2, S, num_heads, head_dim, ctx)            # [1, S, H, Dh]
    var k2_4 = _to_bshd(k2, N_TXT, num_heads, head_dim, ctx)        # [1, N_TXT, H, Dh]
    var v2_4 = _to_bshd(v2, N_TXT, num_heads, head_dim, ctx)
    var attn2 = sdpa_cross_nomask[B, S, N_TXT, 32, 128](
        q2_4, k2_4, v2_4, scale, ctx,
    )
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


# D2D buffer clone for contexts reused across regular and NAG branches.
def _clone_t(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ════════════════════════════════════════════════════════════════════════════
# JOINT-AV DUAL-STREAM BLOCK (Plan P3 — the keystone block)
#
# Full port of LTX2TransformerBlock::forward_with_skip
# (inference-flame/src/models/ltx2_model.rs:1148-1465). SIX attention paths:
#   attn1            video self-attn   (Q/KV=video 4096, 32 heads x 128)
#   audio_attn1      audio self-attn   (Q/KV=audio 2048, 32 heads x 64)
#   attn2            video<->text      (Q=video, KV=video_context)
#   audio_attn2      audio<->text      (Q=audio, KV=audio_context)
#   audio_to_video   a2v cross-modal   (Q=video, KV=audio, to_out->4096)
#   video_to_audio   v2a cross-modal   (Q=audio, KV=video, to_out->2048)
#
# Modulation: per-stream 9-coeff AdaLN (rows 0-5 self+ff, 6-8 video/audio CA
# query) + per-stream prompt_scale_shift_table KV modulation + per-block
# scale_shift_table_a2v_ca_{video,audio} [5,*] cross-modal AdaLN combined with
# the AV-cross global modulation tensors. Gated SDPA (2*sigmoid(to_gate_logits))
# on every path; SwiGLU/GELU FFN per stream.
#
# Block 0 of the distilled-fp8 checkpoint is a BOUNDARY block stored BF16 (no
# FP8 dequant); this struct loads it BF16 and runs the BF16 attention/rope/
# linear path (cos>=0.999 vs the F32 oracle, the standard parity bar).
# ════════════════════════════════════════════════════════════════════════════


struct LTX2AVBlockWeights(Movable):
    """All weights for one full dual-stream block (video + audio + cross-modal).

    Owns weights by canonical key. Canonical keys keep the diffusers/ComfyUI
    naming the checkpoint uses (q_norm/k_norm). Cross-modal scale_shift tables
    are stored under the checkpoint names
    `scale_shift_table_a2v_ca_{video,audio}` ([5,*])."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var lora_names: List[String]
    var lora_a: List[ArcPointer[Tensor]]
    var lora_b: List[ArcPointer[Tensor]]
    var lora_scales: List[Float32]
    var config: LTX2Config

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: LTX2Config,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.lora_names = List[String]()
        self.lora_a = List[ArcPointer[Tensor]]()
        self.lora_b = List[ArcPointer[Tensor]]()
        self.lora_scales = List[Float32]()
        self.config = config

    @staticmethod
    def load(
        checkpoint_path: String,
        block_idx: Int,
        config: LTX2Config,
        ctx: DeviceContext,
    ) raises -> LTX2AVBlockWeights:
        """Load every AV-block tensor for `block_idx` from the checkpoint.
        Resolves the ComfyUI `model.diffusion_model.` prefix automatically and
        the q_norm/k_norm vs norm_q/norm_k naming. All compute tensors uploaded
        BF16; scale_shift tables are F32 on disk and also cast BF16."""
        var st = ShardedSafeTensors.open(checkpoint_path)

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
                    String("LTX2AV: block ")
                    + String(block_idx)
                    + " attn1.to_q.weight not found"
                )

        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        # All six attention modules: to_{q,k,v,out.0} {w,b}, qk-norm, gate.
        var attn_mods = List[String]()
        attn_mods.append("attn1")
        attn_mods.append("audio_attn1")
        attn_mods.append("attn2")
        attn_mods.append("audio_attn2")
        attn_mods.append("audio_to_video_attn")
        attn_mods.append("video_to_audio_attn")

        var keys = List[Tuple[String, String]]()  # (canonical, src under prefix)
        for ref m in attn_mods:
            keys.append((m + ".to_q.weight", m + ".to_q.weight"))
            keys.append((m + ".to_q.bias", m + ".to_q.bias"))
            keys.append((m + ".to_k.weight", m + ".to_k.weight"))
            keys.append((m + ".to_k.bias", m + ".to_k.bias"))
            keys.append((m + ".to_v.weight", m + ".to_v.weight"))
            keys.append((m + ".to_v.bias", m + ".to_v.bias"))
            keys.append((m + ".to_out.0.weight", m + ".to_out.0.weight"))
            keys.append((m + ".to_out.0.bias", m + ".to_out.0.bias"))
            # Resolve qk-norm naming (norm_q/norm_k vs q_norm/k_norm) per module.
            var qn = m + ".norm_q.weight"
            var kn = m + ".norm_k.weight"
            if not _st_has(st, prefix + qn):
                qn = m + ".q_norm.weight"
                kn = m + ".k_norm.weight"
            keys.append((m + ".norm_q.weight", qn))
            keys.append((m + ".norm_k.weight", kn))

        # FFN (video + audio).
        keys.append(("ff.net.0.proj.weight", "ff.net.0.proj.weight"))
        keys.append(("ff.net.0.proj.bias", "ff.net.0.proj.bias"))
        keys.append(("ff.net.2.weight", "ff.net.2.weight"))
        keys.append(("ff.net.2.bias", "ff.net.2.bias"))
        keys.append(("audio_ff.net.0.proj.weight", "audio_ff.net.0.proj.weight"))
        keys.append(("audio_ff.net.0.proj.bias", "audio_ff.net.0.proj.bias"))
        keys.append(("audio_ff.net.2.weight", "audio_ff.net.2.weight"))
        keys.append(("audio_ff.net.2.bias", "audio_ff.net.2.bias"))

        # AdaLN tables.
        keys.append(("scale_shift_table", "scale_shift_table"))
        keys.append(("audio_scale_shift_table", "audio_scale_shift_table"))
        keys.append((
            "scale_shift_table_a2v_ca_video",
            "scale_shift_table_a2v_ca_video",
        ))
        keys.append((
            "scale_shift_table_a2v_ca_audio",
            "scale_shift_table_a2v_ca_audio",
        ))

        for ref kv in keys:
            var canon = kv[0]
            var src = prefix + kv[1]
            var tv = st.tensor_view(src)
            name_to_idx[canon] = len(weights)
            weights.append(ArcPointer(Tensor.from_view_as_bf16(tv, ctx)))

        # Optional per-path gate logits + optional per-stream prompt tables +
        # optional cross-modal-norm affines (these may be absent).
        var opt_keys = List[String]()
        for ref m in attn_mods:
            opt_keys.append(m + ".to_gate_logits.weight")
            opt_keys.append(m + ".to_gate_logits.bias")
        opt_keys.append("prompt_scale_shift_table")
        opt_keys.append("audio_prompt_scale_shift_table")
        opt_keys.append("norm1.weight")
        opt_keys.append("audio_norm1.weight")
        opt_keys.append("norm2.weight")
        opt_keys.append("audio_norm2.weight")
        opt_keys.append("norm3.weight")
        opt_keys.append("audio_norm3.weight")
        opt_keys.append("audio_to_video_norm.weight")
        opt_keys.append("video_to_audio_norm.weight")

        for ref ok in opt_keys:
            if _st_has(st, prefix + ok):
                var tv = st.tensor_view(prefix + ok)
                name_to_idx[ok] = len(weights)
                weights.append(ArcPointer(Tensor.from_view_as_bf16(tv, ctx)))

        return LTX2AVBlockWeights(weights^, name_to_idx^, config)

    @staticmethod
    def from_fp8_block(
        var block: FP8Block,
        config: LTX2Config,
        ctx: DeviceContext,
    ) raises -> LTX2AVBlockWeights:
        """Build a full dual-stream AV block from an FP8-streamed, already-
        dequantized block (LTX2BlockStream.load_block_bf16). Used for the inner
        blocks 4-46 of the distilled-fp8 checkpoint: the streamer reads ALL keys
        under the block prefix (FP8 weights dequantized on-use with their
        per-tensor weight_scale, BF16/F32 copied), so the streamed dict carries
        the full AV key set (video + audio + cross-modal + all tables). This
        rehomes those device Tensors (zero-copy ArcPointer move) into the
        LTX2AVBlockWeights layout the AV forward expects, resolving the
        q_norm/k_norm vs norm_q/norm_k naming per module.

        Mirrors the on-disk `load` key set exactly — same six attn modules, both
        FFNs, all four AdaLN tables, and the optional gate/prompt/norm keys."""
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        var attn_mods = List[String]()
        attn_mods.append("attn1")
        attn_mods.append("audio_attn1")
        attn_mods.append("attn2")
        attn_mods.append("audio_attn2")
        attn_mods.append("audio_to_video_attn")
        attn_mods.append("video_to_audio_attn")

        var keys = List[Tuple[String, String]]()  # (canonical, src-in-block)
        for ref m in attn_mods:
            keys.append((m + ".to_q.weight", m + ".to_q.weight"))
            keys.append((m + ".to_q.bias", m + ".to_q.bias"))
            keys.append((m + ".to_k.weight", m + ".to_k.weight"))
            keys.append((m + ".to_k.bias", m + ".to_k.bias"))
            keys.append((m + ".to_v.weight", m + ".to_v.weight"))
            keys.append((m + ".to_v.bias", m + ".to_v.bias"))
            keys.append((m + ".to_out.0.weight", m + ".to_out.0.weight"))
            keys.append((m + ".to_out.0.bias", m + ".to_out.0.bias"))
            var qn = m + ".norm_q.weight"
            var kn = m + ".norm_k.weight"
            if (m + ".q_norm.weight") in block:
                qn = m + ".q_norm.weight"
                kn = m + ".k_norm.weight"
            keys.append((m + ".norm_q.weight", qn))
            keys.append((m + ".norm_k.weight", kn))

        keys.append(("ff.net.0.proj.weight", "ff.net.0.proj.weight"))
        keys.append(("ff.net.0.proj.bias", "ff.net.0.proj.bias"))
        keys.append(("ff.net.2.weight", "ff.net.2.weight"))
        keys.append(("ff.net.2.bias", "ff.net.2.bias"))
        keys.append(("audio_ff.net.0.proj.weight", "audio_ff.net.0.proj.weight"))
        keys.append(("audio_ff.net.0.proj.bias", "audio_ff.net.0.proj.bias"))
        keys.append(("audio_ff.net.2.weight", "audio_ff.net.2.weight"))
        keys.append(("audio_ff.net.2.bias", "audio_ff.net.2.bias"))

        keys.append(("scale_shift_table", "scale_shift_table"))
        keys.append(("audio_scale_shift_table", "audio_scale_shift_table"))
        keys.append((
            "scale_shift_table_a2v_ca_video",
            "scale_shift_table_a2v_ca_video",
        ))
        keys.append((
            "scale_shift_table_a2v_ca_audio",
            "scale_shift_table_a2v_ca_audio",
        ))

        for ref kv in keys:
            var canon = kv[0]
            var src = kv[1]
            if src not in block:
                raise Error(
                    String("from_fp8_block(AV): streamed block missing ") + src
                )
            name_to_idx[canon] = len(weights)
            weights.append(block[src])

        # Optional keys (gate logits per module, prompt tables, cross-modal norms).
        var opt_keys = List[String]()
        for ref m in attn_mods:
            opt_keys.append(m + ".to_gate_logits.weight")
            opt_keys.append(m + ".to_gate_logits.bias")
        opt_keys.append("prompt_scale_shift_table")
        opt_keys.append("audio_prompt_scale_shift_table")
        opt_keys.append("norm1.weight")
        opt_keys.append("audio_norm1.weight")
        opt_keys.append("norm2.weight")
        opt_keys.append("audio_norm2.weight")
        opt_keys.append("norm3.weight")
        opt_keys.append("audio_norm3.weight")
        opt_keys.append("audio_to_video_norm.weight")
        opt_keys.append("video_to_audio_norm.weight")

        for ref ok in opt_keys:
            if ok in block:
                name_to_idx[ok] = len(weights)
                weights.append(block[ok])

        return LTX2AVBlockWeights(weights^, name_to_idx^, config)

    def to_f32(self, ctx: DeviceContext) raises -> LTX2AVBlockWeights:
        """Return a copy of this block with every weight cast to F32.

        The 48-block residual stream (especially the 4096-dim VIDEO stream)
        grows to |x| ~ 3e3 across the stack, where BF16 storage (~3 sig. figs,
        step ~16 at that scale) injects framework-dependent rounding that two
        independent BF16 implementations cannot agree on to cos >= 0.999 over 48
        accumulations. The Python velocity oracle runs the block math in F32
        (op-identical to the Rust BF16 path, strictly more accurate); running
        the Mojo blocks in F32 makes the full-stack gate apples-to-apples,
        exactly as the connector (ltx2_connector.mojo) already does for its own
        8-block F32 stream. FP8-sourced inner weights stay at FP8 precision
        (they were dequantized to BF16 first, then upcast here) — the F32 cast
        only changes activation/accumulation precision, not the weights' value.
        """
        var weights = List[ArcPointer[Tensor]]()
        for ref w in self.weights:
            weights.append(ArcPointer(cast_tensor(w[], STDtype.F32, ctx)))
        return LTX2AVBlockWeights(weights^, self.name_to_idx.copy(), self.config)

    def _has(self, name: String) -> Bool:
        return name in self.name_to_idx

    def has_weight(self, name: String) -> Bool:
        """Public: does this block hold a weight under canonical `name`?"""
        return name in self.name_to_idx

    def weight_host(self, name: String, ctx: DeviceContext) raises -> List[Float32]:
        """Public: host (F32) copy of the resident weight `name`. For gates."""
        return self._w(name).to_host(ctx)

    def weight_shape(self, name: String) raises -> List[Int]:
        return self._w(name).shape()

    def linear_apply(
        self, name: String, x: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """Public: out = x @ W[name]ᵀ (no bias) using the resident weight. For
        the add-math gate to evaluate base(x) and (base+delta)(x)."""
        return linear(x, self._w(name), None, ctx)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("LTX2AV: missing weight ") + name)
        return self.weights[self.name_to_idx[name]][]

    def add_delta_to(
        mut self, name: String, delta: Tensor, ctx: DeviceContext
    ) raises:
        """ADD a LoRA delta IN PLACE to the resident dequanted weight `name`:
        `W[name] = W[name] + delta` (shapes must match; same dtype assumed —
        the caller casts the delta to the weight dtype). This is the LTX-2
        at-dequant LoRA application hook: blocks are FP8-streamed and dequanted
        transiently per step, so there is no persistent resident W — the delta
        is re-added to the freshly-dequanted block weight every time the block
        streams in (NEVER a one-time fuse, NEVER written to disk). Replaces the
        ArcPointer with the summed tensor (the same discipline lora.mojo uses
        for resident merges)."""
        if name not in self.name_to_idx:
            raise Error(String("LTX2AV.add_delta_to: missing weight ") + name)
        var idx = self.name_to_idx[name]
        var bdims = self.weights[idx][].shape()
        var ddims = delta.shape()
        if len(bdims) != len(ddims):
            raise Error(
                String("LTX2AV.add_delta_to: rank mismatch for ") + name
            )
        for i in range(len(bdims)):
            if bdims[i] != ddims[i]:
                raise Error(
                    String("LTX2AV.add_delta_to: shape mismatch for ") + name
                )
        var summed = add(self.weights[idx][], delta, ctx)
        self.weights[idx] = ArcPointer[Tensor](summed^)

    def add_lora_factor(
        mut self,
        name: String,
        var a: Tensor,
        var b: Tensor,
        scale: Float32,
    ) raises:
        """Attach one factorized LoRA adapter for a block-local linear weight.

        Runtime applies this as:
            linear(x, W, bias) + scale * linear(linear(x, A), B)
        where A is [rank, in] and B is [out, rank]. This avoids materializing the
        full [out, in] delta tensor for LTX-2's very large AV blocks.
        """
        if name not in self.name_to_idx:
            raise Error(String("LTX2AV.add_lora_factor: missing weight ") + name)
        self.lora_names.append(name)
        self.lora_a.append(ArcPointer[Tensor](a^))
        self.lora_b.append(ArcPointer[Tensor](b^))
        self.lora_scales.append(scale)

    def add_lora_factor_arc(
        mut self,
        name: String,
        var a: ArcPointer[Tensor],
        var b: ArcPointer[Tensor],
        scale: Float32,
    ) raises:
        """Attach an already device-resident factorized LoRA adapter.

        This is the hot inference path: A/B tensors are preloaded once and the
        transient streamed block just holds ArcPointer refs while it runs.
        """
        if name not in self.name_to_idx:
            raise Error(String("LTX2AV.add_lora_factor_arc: missing weight ") + name)
        self.lora_names.append(name)
        self.lora_a.append(a^)
        self.lora_b.append(b^)
        self.lora_scales.append(scale)

    def _linear_lora_delta(
        self,
        x: Tensor,
        w_key: String,
        var base_out: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var out = base_out^
        for i in range(len(self.lora_names)):
            if self.lora_names[i] == w_key:
                var down = linear(x, self.lora_a[i][], None, ctx)
                var up = linear(down, self.lora_b[i][], None, ctx)
                var scaled = mul_scalar(up, self.lora_scales[i], ctx)
                out = add(out, scaled, ctx)
        return out^

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
        var out = linear(x, w, Optional[Tensor](self._clone(b, ctx)), ctx)
        return self._linear_lora_delta(x, w_key, out^, ctx)


# ── RMSNorm with an OPTIONAL affine weight (cross-modal norms may have one) ──
def _rms_norm_opt(
    x: Tensor,
    weights: LTX2AVBlockWeights,
    w_key: String,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    if weights._has(w_key):
        return rms_norm(x, weights._w(w_key), eps, ctx)
    return _rms_norm_no_affine(x, eps, ctx)


# ── broadcast modulate: x*(1+scale)+shift where scale/shift are [1,1,dim] ──
# (cross-modal mod params broadcast across tokens). tensor_algebra mul/add are
# NumPy-broadcast aware, so [1,S,dim]*[1,1,dim] works directly.
def _modulate_bc(
    x: Tensor, scale: Tensor, shift: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var one_plus = add_scalar(scale, Float32(1.0), ctx)
    return add(mul(x, one_plus, ctx), shift, ctx)


# ── per-token AdaLN row: table[idx]+temb chunk, broadcasting over tokens. ──
#   table: [9, dim]  temb: [1, N, 9*dim] -> returns [1, N, dim]
def _ada_row_pertok(
    table: Tensor, temb: Tensor, idx: Int, dim: Int, n: Int, ctx: DeviceContext
) raises -> Tensor:
    var trow = slice(table, 0, idx, 1, ctx)              # [1, dim]
    var trow3 = reshape(trow, _shape3(1, 1, dim), ctx)   # [1,1,dim] broadcast
    var tchunk = slice(temb, 2, idx * dim, dim, ctx)     # [1, N, dim]
    return add(trow3, tchunk, ctx)


# ── AV-cross-attn modulation params (port of compute_cross_attn_params). ──
# Each *_ss is [1,1,4*dim]; *_gate is [1,1,dim]; tables are [5,dim].
# Returns broadcast [1,1,dim] tensors.
struct _CrossMod(Movable):
    var a2v_gate: Tensor
    var v2a_gate: Tensor
    var v_a2v_scale: Tensor
    var v_a2v_shift: Tensor
    var v_v2a_scale: Tensor
    var v_v2a_shift: Tensor
    var a_a2v_scale: Tensor
    var a_a2v_shift: Tensor
    var a_v2a_scale: Tensor
    var a_v2a_shift: Tensor

    def __init__(
        out self,
        var a2v_gate: Tensor, var v2a_gate: Tensor,
        var v_a2v_scale: Tensor, var v_a2v_shift: Tensor,
        var v_v2a_scale: Tensor, var v_v2a_shift: Tensor,
        var a_a2v_scale: Tensor, var a_a2v_shift: Tensor,
        var a_v2a_scale: Tensor, var a_v2a_shift: Tensor,
    ):
        self.a2v_gate = a2v_gate^
        self.v2a_gate = v2a_gate^
        self.v_a2v_scale = v_a2v_scale^
        self.v_a2v_shift = v_a2v_shift^
        self.v_v2a_scale = v_v2a_scale^
        self.v_v2a_shift = v_v2a_shift^
        self.a_a2v_scale = a_a2v_scale^
        self.a_a2v_shift = a_a2v_shift^
        self.a_v2a_scale = a_v2a_scale^
        self.a_v2a_shift = a_v2a_shift^


# Build a broadcast [1,1,dim] modulation row from table row `r` + the matching
# chunk of a [1,1,4*dim] global scale_shift tensor.
def _cm_row(
    table: Tensor, r: Int, ss: Tensor, chunk: Int, dim: Int, ctx: DeviceContext
) raises -> Tensor:
    var trow = reshape(slice(table, 0, r, 1, ctx), _shape3(1, 1, dim), ctx)
    var c = slice(ss, 2, chunk * dim, dim, ctx)          # [1,1,dim]
    return add(trow, c, ctx)


def _compute_cross_mod(
    v_table: Tensor, a_table: Tensor,
    v_ca_ss: Tensor, a_ca_ss: Tensor,
    v_ca_gate: Tensor, a_ca_gate: Tensor,
    vdim: Int, adim: Int, ctx: DeviceContext,
) raises -> _CrossMod:
    # rows 0,1 = a2v scale/shift ; rows 2,3 = v2a scale/shift ; row 4 = gate.
    var v_a2v_scale = _cm_row(v_table, 0, v_ca_ss, 0, vdim, ctx)
    var v_a2v_shift = _cm_row(v_table, 1, v_ca_ss, 1, vdim, ctx)
    var v_v2a_scale = _cm_row(v_table, 2, v_ca_ss, 2, vdim, ctx)
    var v_v2a_shift = _cm_row(v_table, 3, v_ca_ss, 3, vdim, ctx)
    var a2v_gate = add(
        reshape(slice(v_table, 0, 4, 1, ctx), _shape3(1, 1, vdim), ctx),
        v_ca_gate, ctx,
    )
    var a_a2v_scale = _cm_row(a_table, 0, a_ca_ss, 0, adim, ctx)
    var a_a2v_shift = _cm_row(a_table, 1, a_ca_ss, 1, adim, ctx)
    var a_v2a_scale = _cm_row(a_table, 2, a_ca_ss, 2, adim, ctx)
    var a_v2a_shift = _cm_row(a_table, 3, a_ca_ss, 3, adim, ctx)
    var v2a_gate = add(
        reshape(slice(a_table, 0, 4, 1, ctx), _shape3(1, 1, adim), ctx),
        a_ca_gate, ctx,
    )
    return _CrossMod(
        a2v_gate^, v2a_gate^,
        v_a2v_scale^, v_a2v_shift^, v_v2a_scale^, v_v2a_shift^,
        a_a2v_scale^, a_a2v_shift^, a_v2a_scale^, a_v2a_shift^,
    )


# ── KV (text-context) modulation: context*(1+scale_kv)+shift_kv where
# combined = psst[2,dim] + prompt_ts[1,seq,2*dim]. (Port of the 9-param CA KV
# modulation, ltx2_model.rs:1281-1296.) ──
def _kv_modulate(
    context: Tensor,   # [1, seq, dim]
    psst: Tensor,      # [2, dim]
    prompt_ts: Tensor, # [1, seq, 2*dim]
    seq: Int, dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    # combined[:, :, 0, :] = psst[0] + prompt_ts[..,:dim]  (shift)
    # combined[:, :, 1, :] = psst[1] + prompt_ts[..,dim:]  (scale)
    var shift_row = reshape(slice(psst, 0, 0, 1, ctx), _shape3(1, 1, dim), ctx)
    var scale_row = reshape(slice(psst, 0, 1, 1, ctx), _shape3(1, 1, dim), ctx)
    var pt_shift = slice(prompt_ts, 2, 0, dim, ctx)       # [1, seq, dim]
    var pt_scale = slice(prompt_ts, 2, dim, dim, ctx)     # [1, seq, dim]
    var shift_kv = add(shift_row, pt_shift, ctx)
    var scale_kv = add(scale_row, pt_scale, ctx)
    return _modulate_bc(context, scale_kv, shift_kv, ctx)


# ── generic attention path (port of LTX2Attention::forward) ──
# Q-input `hidden` [1, S_q, q_dim]; KV-input `kv` [1, S_kv, kv_dim]. Loaded with
# `num_heads`/`head_dim` (inner = num_heads*head_dim). to_out maps inner->out
# (4096 for a2v, 2048 elsewhere). Per-head gate from `hidden`. Optional q/k rope
# tables (pre-permuted to [S,H,hrd] row order so they match the BSHD flatten).
# Returns [1, S_q, out_dim].
def _av_attention[SQ: Int, SKV: Int, SPAD: Int, H: Int, DH: Int](
    weights: LTX2AVBlockWeights,
    mod_name: String,
    hidden: Tensor,            # [1, S_q, q_dim]
    kv: Tensor,                # [1, S_kv, kv_dim]
    has_q_rope: Bool,
    q_rope_cos: Tensor,        # [SQ*H, DH/2]  (s,h) row order (dummy if absent)
    q_rope_sin: Tensor,
    has_k_rope: Bool,
    k_rope_cos: Tensor,        # [SKV*H, DH/2]
    k_rope_sin: Tensor,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var inner = H * DH
    var scale = Float32(1.0) / sqrt(Float32(DH))

    var q = weights._linear_b(hidden, mod_name + ".to_q.weight", mod_name + ".to_q.bias", ctx)
    var k = weights._linear_b(kv, mod_name + ".to_k.weight", mod_name + ".to_k.bias", ctx)
    var v = weights._linear_b(kv, mod_name + ".to_v.weight", mod_name + ".to_v.bias", ctx)

    q = rms_norm(q, weights._w(mod_name + ".norm_q.weight"), eps, ctx)
    k = rms_norm(k, weights._w(mod_name + ".norm_k.weight"), eps, ctx)

    var q4 = reshape(q, _shape4(1, SQ, H, DH), ctx)
    var k4 = reshape(k, _shape4(1, SKV, H, DH), ctx)
    var v4 = reshape(v, _shape4(1, SKV, H, DH), ctx)

    if has_q_rope:
        q4 = apply_ltx2_rope(q4, q_rope_cos, q_rope_sin, ctx)
    # K rope: explicit k rope, else fall back to q rope (matches Rust
    # key_rope.or(query_rope)). Only applied if SOME rope is present.
    if has_k_rope:
        k4 = apply_ltx2_rope(k4, k_rope_cos, k_rope_sin, ctx)
    elif has_q_rope:
        k4 = apply_ltx2_rope(k4, q_rope_cos, q_rope_sin, ctx)

    # Square self-attn uses the exact no-mask path. Rectangular cross-attn uses
    # online-softmax directly over Q(SQ) x KV(SKV) without square padding.
    var attn_flat: Tensor
    comptime if SQ == SKV and SQ == SPAD:
        # Square self-attention: the additive mask is all-zeros (full attention).
        # Use vendor-matmul SDPA while the score slab fits the explicit budget.
        # The scalar online-softmax path is for truly huge grids such as
        # 2x AudioSync stage-2, not for 97-frame stage-1.
        var attn: Tensor
        comptime score_mib = (SPAD * SPAD * H * 4) // (1024 * 1024)
        comptime if score_mib >= 3584:
            attn = sdpa_nomask_tiled[1, SPAD, H, DH](q4, k4, v4, scale, ctx)
        else:
            attn = sdpa_nomask[1, SPAD, H, DH](q4, k4, v4, scale, ctx)
        attn_flat = reshape(attn, _shape3(1, SQ, inner), ctx)
    else:
        # Rectangular cross-attention. The old path padded Q/K/V to SPAD and
        # built [SPAD,SPAD] masks; AudioSync-size video would OOM there.
        var attn = sdpa_cross_nomask[1, SQ, SKV, H, DH](q4, k4, v4, scale, ctx)
        attn_flat = reshape(attn, _shape3(1, SQ, inner), ctx)

    # Per-head gate: gate_logits = linear(hidden) -> [1,SQ,H]; *2sigmoid.
    if weights._has(mod_name + ".to_gate_logits.weight"):
        var gl = weights._linear_b(
            hidden,
            mod_name + ".to_gate_logits.weight",
            mod_name + ".to_gate_logits.bias",
            ctx,
        )                                                # [1, SQ, H]
        var gates = mul_scalar(sigmoid(gl, ctx), Float32(2.0), ctx)
        var gates4 = reshape(gates, _shape4(1, SQ, H, 1), ctx)
        var a4 = reshape(attn_flat, _shape4(1, SQ, H, DH), ctx)
        attn_flat = reshape(mul(a4, gates4, ctx), _shape3(1, SQ, inner), ctx)

    return weights._linear_b(
        attn_flat, mod_name + ".to_out.0.weight", mod_name + ".to_out.0.bias", ctx
    )

# ── full dual-stream AV block forward ──
# Consumes pre-computed modulation/rope tensors (dumped by the oracle):
#   hidden    [1, S_V, 4096]   ahs  [1, S_A, 2048]
#   enc/aenc  [1, N_TXT, 4096/2048]
#   v_temb    [1, S_V, 9*4096]  a_temb [1, S_A, 9*2048]
#   v_ca_ss [1,1,4*4096] a_ca_ss [1,1,4*2048] v_ca_gate [1,1,4096] a_ca_gate [1,1,2048]
#   *_prompt_ts [1, N_TXT, 2*dim]
#   rope tables pre-permuted to [S*H, head_dim/2] (s,h) row order.
# Returns (video_out [1,S_V,4096], audio_out [1,S_A,2048], v2a_delta [1,S_A,2048]).
# v2a_delta = v2a_out * v2a_gate — the exact addend applied to the audio stream by the
# video-to-audio cross-modal path.  It is a debug output only; the math is unchanged.
def ltx2_block_forward_av[
    S_V: Int, S_A: Int, N_TXT: Int, S_VPAD: Int, S_APAD: Int
](
    weights: LTX2AVBlockWeights,
    hidden: Tensor, ahs: Tensor,
    enc: Tensor, aenc: Tensor,
    v_temb: Tensor, a_temb: Tensor,
    v_ca_ss: Tensor, a_ca_ss: Tensor,
    v_ca_gate: Tensor, a_ca_gate: Tensor,
    v_prompt_ts: Tensor, a_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor,
    ca_a_cos: Tensor, ca_a_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor, Tensor]:
    var VD = 4096
    var AD = 2048

    # Dummy rope table for the no-rope cross-attn paths (attn2/audio_attn2).
    var dummy_data = List[Float32]()
    dummy_data.append(Float32(1.0))
    var dummy_sh = List[Int]()
    dummy_sh.append(1)
    dummy_sh.append(1)
    var dummy = Tensor.from_host(dummy_data, dummy_sh^, hidden.dtype(), ctx)

    # ---- 1. Video self-attn (AdaLN, rope, gated) ----
    ref vtab = weights._w("scale_shift_table")
    var v_shift_msa = _ada_row_pertok(vtab, v_temb, 0, VD, S_V, ctx)
    var v_scale_msa = _ada_row_pertok(vtab, v_temb, 1, VD, S_V, ctx)
    var v_gate_msa = _ada_row_pertok(vtab, v_temb, 2, VD, S_V, ctx)
    var v_shift_mlp = _ada_row_pertok(vtab, v_temb, 3, VD, S_V, ctx)
    var v_scale_mlp = _ada_row_pertok(vtab, v_temb, 4, VD, S_V, ctx)
    var v_gate_mlp = _ada_row_pertok(vtab, v_temb, 5, VD, S_V, ctx)

    var mod_h = _modulate_bc(
        _rms_norm_opt(hidden, weights, "norm1.weight", eps, ctx),
        v_scale_msa, v_shift_msa, ctx,
    )
    var v_attn = _av_attention[S_V, S_V, S_V, 32, 128](
        weights, "attn1", mod_h, mod_h,
        True, v_cos, v_sin, False, dummy, dummy, eps, ctx,
    )
    var hs = add(hidden, mul(v_gate_msa, v_attn, ctx), ctx)

    # ---- Audio self-attn ----
    ref atab = weights._w("audio_scale_shift_table")
    var a_shift_msa = _ada_row_pertok(atab, a_temb, 0, AD, S_A, ctx)
    var a_scale_msa = _ada_row_pertok(atab, a_temb, 1, AD, S_A, ctx)
    var a_gate_msa = _ada_row_pertok(atab, a_temb, 2, AD, S_A, ctx)
    var a_shift_mlp = _ada_row_pertok(atab, a_temb, 3, AD, S_A, ctx)
    var a_scale_mlp = _ada_row_pertok(atab, a_temb, 4, AD, S_A, ctx)
    var a_gate_mlp = _ada_row_pertok(atab, a_temb, 5, AD, S_A, ctx)

    var mod_a = _modulate_bc(
        _rms_norm_opt(ahs, weights, "audio_norm1.weight", eps, ctx),
        a_scale_msa, a_shift_msa, ctx,
    )
    var a_attn = _av_attention[S_A, S_A, S_A, 32, 64](
        weights, "audio_attn1", mod_a, mod_a,
        True, a_cos, a_sin, False, dummy, dummy, eps, ctx,
    )
    var ahss = add(ahs, mul(a_gate_msa, a_attn, ctx), ctx)

    # ---- 2. Video cross-attn (text) ----
    var v_shift_ca = _ada_row_pertok(vtab, v_temb, 6, VD, S_V, ctx)
    var v_scale_ca = _ada_row_pertok(vtab, v_temb, 7, VD, S_V, ctx)
    var v_gate_ca = _ada_row_pertok(vtab, v_temb, 8, VD, S_V, ctx)
    var mod_h2 = _modulate_bc(
        _rms_norm_opt(hs, weights, "norm2.weight", eps, ctx),
        v_scale_ca, v_shift_ca, ctx,
    )
    var mv_ctx: Tensor
    if weights._has("prompt_scale_shift_table"):
        mv_ctx = _kv_modulate(
            enc, weights._w("prompt_scale_shift_table"), v_prompt_ts,
            N_TXT, VD, ctx,
        )
    else:
        mv_ctx = _clone_t(enc, ctx)
    var v_ca_out = _av_attention[S_V, N_TXT, S_VPAD, 32, 128](
        weights, "attn2", mod_h2, mv_ctx,
        False, dummy, dummy, False, dummy, dummy, eps, ctx,
    )
    hs = add(hs, mul(v_gate_ca, v_ca_out, ctx), ctx)

    # ---- Audio cross-attn (text) ----
    var a_shift_ca = _ada_row_pertok(atab, a_temb, 6, AD, S_A, ctx)
    var a_scale_ca = _ada_row_pertok(atab, a_temb, 7, AD, S_A, ctx)
    var a_gate_ca = _ada_row_pertok(atab, a_temb, 8, AD, S_A, ctx)
    var mod_a2 = _modulate_bc(
        _rms_norm_opt(ahss, weights, "audio_norm2.weight", eps, ctx),
        a_scale_ca, a_shift_ca, ctx,
    )
    var ma_ctx: Tensor
    if weights._has("audio_prompt_scale_shift_table"):
        ma_ctx = _kv_modulate(
            aenc, weights._w("audio_prompt_scale_shift_table"), a_prompt_ts,
            N_TXT, AD, ctx,
        )
    else:
        ma_ctx = _clone_t(aenc, ctx)
    var a_ca_out = _av_attention[S_A, N_TXT, S_APAD, 32, 64](
        weights, "audio_attn2", mod_a2, ma_ctx,
        False, dummy, dummy, False, dummy, dummy, eps, ctx,
    )
    ahss = add(ahss, mul(a_gate_ca, a_ca_out, ctx), ctx)

    # ---- 3. A2V / V2A cross-modal ----
    var norm_a2v = _rms_norm_opt(hs, weights, "audio_to_video_norm.weight", eps, ctx)
    var norm_v2a = _rms_norm_opt(ahss, weights, "video_to_audio_norm.weight", eps, ctx)
    var cm = _compute_cross_mod(
        weights._w("scale_shift_table_a2v_ca_video"),
        weights._w("scale_shift_table_a2v_ca_audio"),
        v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, VD, AD, ctx,
    )

    # A2V: Q=video (mod v_a2v), KV=audio (mod a_a2v), to_out->4096.
    var mod_video_a2v = _modulate_bc(norm_a2v, cm.v_a2v_scale, cm.v_a2v_shift, ctx)
    var mod_audio_a2v = _modulate_bc(norm_v2a, cm.a_a2v_scale, cm.a_a2v_shift, ctx)
    var a2v_out = _av_attention[S_V, S_A, S_VPAD, 32, 64](
        weights, "audio_to_video_attn", mod_video_a2v, mod_audio_a2v,
        True, ca_v_cos, ca_v_sin, True, ca_a_cos, ca_a_sin, eps, ctx,
    )
    hs = add(hs, mul(cm.a2v_gate, a2v_out, ctx), ctx)

    # V2A: Q=audio (mod a_v2a), KV=video (mod v_v2a), to_out->2048.
    var mod_video_v2a = _modulate_bc(norm_a2v, cm.v_v2a_scale, cm.v_v2a_shift, ctx)
    var mod_audio_v2a = _modulate_bc(norm_v2a, cm.a_v2a_scale, cm.a_v2a_shift, ctx)
    # V2A: SQ=S_A (audio query), SKV=S_V (video KV). The square-SDPA pad target
    # must be >= max(S_A, S_V). At the stage-2 grid S_V (3072) > S_APAD (1024),
    # so we pad to S_VPAD (= max(S_V,N_TXT,S_A)), NOT S_APAD — using S_APAD gave a
    # NEGATIVE pad (S_APAD - S_V) and the from_host(-4194304) crash.
    var v2a_out = _av_attention[S_A, S_V, S_VPAD, 32, 64](
        weights, "video_to_audio_attn", mod_audio_v2a, mod_video_v2a,
        True, ca_a_cos, ca_a_sin, True, ca_v_cos, ca_v_sin, eps, ctx,
    )
    # Capture the v2a contribution (raw addend) BEFORE mutating ahss.
    # This is a debug-only output; the math is identical to the 2-return version.
    var v2a_delta = mul(cm.v2a_gate, v2a_out, ctx)
    ahss = add(ahss, v2a_delta, ctx)

    # ---- 4. FFN (video + audio) ----
    var mod_ff = _modulate_bc(
        _rms_norm_opt(hs, weights, "norm3.weight", eps, ctx),
        v_scale_mlp, v_shift_mlp, ctx,
    )
    var ff = weights._linear_b(mod_ff, "ff.net.0.proj.weight", "ff.net.0.proj.bias", ctx)
    ff = gelu(ff, ctx)
    ff = weights._linear_b(ff, "ff.net.2.weight", "ff.net.2.bias", ctx)
    hs = add(hs, mul(v_gate_mlp, ff, ctx), ctx)

    var mod_aff = _modulate_bc(
        _rms_norm_opt(ahss, weights, "audio_norm3.weight", eps, ctx),
        a_scale_mlp, a_shift_mlp, ctx,
    )
    var aff = weights._linear_b(mod_aff, "audio_ff.net.0.proj.weight", "audio_ff.net.0.proj.bias", ctx)
    aff = gelu(aff, ctx)
    aff = weights._linear_b(aff, "audio_ff.net.2.weight", "audio_ff.net.2.bias", ctx)
    ahss = add(ahss, mul(a_gate_mlp, aff, ctx), ctx)

    return (hs^, ahss^, v2a_delta^)


# ════════════════════════════════════════════════════════════════════════════
# NAG-AWARE AV BLOCK FORWARD (Normalized Attention Guidance hook)
#
# Identical to `ltx2_block_forward_av` EXCEPT the two text cross-attention
# stages (attn2 video, audio_attn2 audio) are run TWICE when `nag.enabled`:
# once with the positive (real-prompt) KV-context and once with the NULL-prompt
# KV-context carried in `nag`. The two cross-attn outputs are fused by
# `NAGContext.combine_*` (= nag.py:_nag_combine) BEFORE the gated residual add.
# This is the exact placement of NAGPatch (nag.py:67-90): the wrap is around the
# attn2 forward, so the combined output replaces the single attn2 output and
# THEN the per-block gate (v_gate_ca / a_gate_ca) and residual apply unchanged.
#
# When `nag.enabled == False` this reduces to the same math as
# `ltx2_block_forward_av` (the positive-only path); callers can route through
# this single function for both the NAG (stage1/stage2) and the no-NAG paths.
#
# The null KV-context is built with the SAME modulation the positive context
# gets (_kv_modulate when the prompt table exists, else a raw clone) so that the
# only difference between out_pos and out_neg is the encoder hidden states —
# matching nag.py, which calls the unmodified `original(x, ctx, None, ...)` with
# the null `ctx` and no attention mask.
# ════════════════════════════════════════════════════════════════════════════
def ltx2_block_forward_av_nag[
    S_V: Int, S_A: Int, N_TXT: Int, S_VPAD: Int, S_APAD: Int
](
    weights: LTX2AVBlockWeights,
    hidden: Tensor, ahs: Tensor,
    enc: Tensor, aenc: Tensor,
    v_temb: Tensor, a_temb: Tensor,
    v_ca_ss: Tensor, a_ca_ss: Tensor,
    v_ca_gate: Tensor, a_ca_gate: Tensor,
    v_prompt_ts: Tensor, a_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor,
    ca_a_cos: Tensor, ca_a_sin: Tensor,
    nag: NAGContext,
    eps: Float32, ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor, Tensor]:
    var VD = 4096
    var AD = 2048

    var dummy_data = List[Float32]()
    dummy_data.append(Float32(1.0))
    var dummy_sh = List[Int]()
    dummy_sh.append(1)
    dummy_sh.append(1)
    var dummy = Tensor.from_host(dummy_data, dummy_sh^, hidden.dtype(), ctx)

    # ---- 1. Video self-attn ----
    ref vtab = weights._w("scale_shift_table")
    var v_shift_msa = _ada_row_pertok(vtab, v_temb, 0, VD, S_V, ctx)
    var v_scale_msa = _ada_row_pertok(vtab, v_temb, 1, VD, S_V, ctx)
    var v_gate_msa = _ada_row_pertok(vtab, v_temb, 2, VD, S_V, ctx)
    var v_shift_mlp = _ada_row_pertok(vtab, v_temb, 3, VD, S_V, ctx)
    var v_scale_mlp = _ada_row_pertok(vtab, v_temb, 4, VD, S_V, ctx)
    var v_gate_mlp = _ada_row_pertok(vtab, v_temb, 5, VD, S_V, ctx)

    var mod_h = _modulate_bc(
        _rms_norm_opt(hidden, weights, "norm1.weight", eps, ctx),
        v_scale_msa, v_shift_msa, ctx,
    )
    var v_attn = _av_attention[S_V, S_V, S_V, 32, 128](
        weights, "attn1", mod_h, mod_h,
        True, v_cos, v_sin, False, dummy, dummy, eps, ctx,
    )
    var hs = add(hidden, mul(v_gate_msa, v_attn, ctx), ctx)

    # ---- Audio self-attn ----
    ref atab = weights._w("audio_scale_shift_table")
    var a_shift_msa = _ada_row_pertok(atab, a_temb, 0, AD, S_A, ctx)
    var a_scale_msa = _ada_row_pertok(atab, a_temb, 1, AD, S_A, ctx)
    var a_gate_msa = _ada_row_pertok(atab, a_temb, 2, AD, S_A, ctx)
    var a_shift_mlp = _ada_row_pertok(atab, a_temb, 3, AD, S_A, ctx)
    var a_scale_mlp = _ada_row_pertok(atab, a_temb, 4, AD, S_A, ctx)
    var a_gate_mlp = _ada_row_pertok(atab, a_temb, 5, AD, S_A, ctx)

    var mod_a = _modulate_bc(
        _rms_norm_opt(ahs, weights, "audio_norm1.weight", eps, ctx),
        a_scale_msa, a_shift_msa, ctx,
    )
    var a_attn = _av_attention[S_A, S_A, S_A, 32, 64](
        weights, "audio_attn1", mod_a, mod_a,
        True, a_cos, a_sin, False, dummy, dummy, eps, ctx,
    )
    var ahss = add(ahs, mul(a_gate_msa, a_attn, ctx), ctx)

    # ---- 2. Video cross-attn (text) — NAG-wrapped ----
    var v_shift_ca = _ada_row_pertok(vtab, v_temb, 6, VD, S_V, ctx)
    var v_scale_ca = _ada_row_pertok(vtab, v_temb, 7, VD, S_V, ctx)
    var v_gate_ca = _ada_row_pertok(vtab, v_temb, 8, VD, S_V, ctx)
    var mod_h2 = _modulate_bc(
        _rms_norm_opt(hs, weights, "norm2.weight", eps, ctx),
        v_scale_ca, v_shift_ca, ctx,
    )
    var mv_ctx: Tensor
    if weights._has("prompt_scale_shift_table"):
        mv_ctx = _kv_modulate(
            enc, weights._w("prompt_scale_shift_table"), v_prompt_ts,
            N_TXT, VD, ctx,
        )
    else:
        mv_ctx = _clone_t(enc, ctx)
    var v_ca_out = _av_attention[S_V, N_TXT, S_VPAD, 32, 128](
        weights, "attn2", mod_h2, mv_ctx,
        False, dummy, dummy, False, dummy, dummy, eps, ctx,
    )
    if nag.enabled:
        # NULL-context cross-attn (same modulation, null encoder states).
        var mv_ctx_null: Tensor
        if weights._has("prompt_scale_shift_table"):
            mv_ctx_null = _kv_modulate(
                nag.nag_v_ctx, weights._w("prompt_scale_shift_table"),
                v_prompt_ts, N_TXT, VD, ctx,
            )
        else:
            mv_ctx_null = _clone_t(nag.nag_v_ctx, ctx)
        var v_ca_neg = _av_attention[S_V, N_TXT, S_VPAD, 32, 128](
            weights, "attn2", mod_h2, mv_ctx_null,
            False, dummy, dummy, False, dummy, dummy, eps, ctx,
        )
        v_ca_out = nag.combine_video(v_ca_out, v_ca_neg, ctx)
    hs = add(hs, mul(v_gate_ca, v_ca_out, ctx), ctx)

    # ---- Audio cross-attn (text) — NAG-wrapped when audio NAG present ----
    var a_shift_ca = _ada_row_pertok(atab, a_temb, 6, AD, S_A, ctx)
    var a_scale_ca = _ada_row_pertok(atab, a_temb, 7, AD, S_A, ctx)
    var a_gate_ca = _ada_row_pertok(atab, a_temb, 8, AD, S_A, ctx)
    var mod_a2 = _modulate_bc(
        _rms_norm_opt(ahss, weights, "audio_norm2.weight", eps, ctx),
        a_scale_ca, a_shift_ca, ctx,
    )
    var ma_ctx: Tensor
    if weights._has("audio_prompt_scale_shift_table"):
        ma_ctx = _kv_modulate(
            aenc, weights._w("audio_prompt_scale_shift_table"), a_prompt_ts,
            N_TXT, AD, ctx,
        )
    else:
        ma_ctx = _clone_t(aenc, ctx)
    var a_ca_out = _av_attention[S_A, N_TXT, S_APAD, 32, 64](
        weights, "audio_attn2", mod_a2, ma_ctx,
        False, dummy, dummy, False, dummy, dummy, eps, ctx,
    )
    if nag.enabled and nag.has_audio:
        var ma_ctx_null: Tensor
        if weights._has("audio_prompt_scale_shift_table"):
            ma_ctx_null = _kv_modulate(
                nag.nag_a_ctx, weights._w("audio_prompt_scale_shift_table"),
                a_prompt_ts, N_TXT, AD, ctx,
            )
        else:
            ma_ctx_null = _clone_t(nag.nag_a_ctx, ctx)
        var a_ca_neg = _av_attention[S_A, N_TXT, S_APAD, 32, 64](
            weights, "audio_attn2", mod_a2, ma_ctx_null,
            False, dummy, dummy, False, dummy, dummy, eps, ctx,
        )
        a_ca_out = nag.combine_audio(a_ca_out, a_ca_neg, ctx)
    ahss = add(ahss, mul(a_gate_ca, a_ca_out, ctx), ctx)

    # ---- 3. A2V / V2A cross-modal ----
    var norm_a2v = _rms_norm_opt(hs, weights, "audio_to_video_norm.weight", eps, ctx)
    var norm_v2a = _rms_norm_opt(ahss, weights, "video_to_audio_norm.weight", eps, ctx)
    var cm = _compute_cross_mod(
        weights._w("scale_shift_table_a2v_ca_video"),
        weights._w("scale_shift_table_a2v_ca_audio"),
        v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, VD, AD, ctx,
    )

    var mod_video_a2v = _modulate_bc(norm_a2v, cm.v_a2v_scale, cm.v_a2v_shift, ctx)
    var mod_audio_a2v = _modulate_bc(norm_v2a, cm.a_a2v_scale, cm.a_a2v_shift, ctx)
    var a2v_out = _av_attention[S_V, S_A, S_VPAD, 32, 64](
        weights, "audio_to_video_attn", mod_video_a2v, mod_audio_a2v,
        True, ca_v_cos, ca_v_sin, True, ca_a_cos, ca_a_sin, eps, ctx,
    )
    hs = add(hs, mul(cm.a2v_gate, a2v_out, ctx), ctx)

    var mod_video_v2a = _modulate_bc(norm_a2v, cm.v_v2a_scale, cm.v_v2a_shift, ctx)
    var mod_audio_v2a = _modulate_bc(norm_v2a, cm.a_v2a_scale, cm.a_v2a_shift, ctx)
    # V2A: SQ=S_A (audio query), SKV=S_V (video KV). The square-SDPA pad target
    # must be >= max(S_A, S_V). At the stage-2 grid S_V (3072) > S_APAD (1024),
    # so we pad to S_VPAD (= max(S_V,N_TXT,S_A)), NOT S_APAD — using S_APAD gave a
    # NEGATIVE pad (S_APAD - S_V) and the from_host(-4194304) crash.
    var v2a_out = _av_attention[S_A, S_V, S_VPAD, 32, 64](
        weights, "video_to_audio_attn", mod_audio_v2a, mod_video_v2a,
        True, ca_a_cos, ca_a_sin, True, ca_v_cos, ca_v_sin, eps, ctx,
    )
    var v2a_delta = mul(cm.v2a_gate, v2a_out, ctx)
    ahss = add(ahss, v2a_delta, ctx)

    # ---- 4. FFN (video + audio) ----
    var mod_ff = _modulate_bc(
        _rms_norm_opt(hs, weights, "norm3.weight", eps, ctx),
        v_scale_mlp, v_shift_mlp, ctx,
    )
    var ff = weights._linear_b(mod_ff, "ff.net.0.proj.weight", "ff.net.0.proj.bias", ctx)
    ff = gelu(ff, ctx)
    ff = weights._linear_b(ff, "ff.net.2.weight", "ff.net.2.bias", ctx)
    hs = add(hs, mul(v_gate_mlp, ff, ctx), ctx)

    var mod_aff = _modulate_bc(
        _rms_norm_opt(ahss, weights, "audio_norm3.weight", eps, ctx),
        a_scale_mlp, a_shift_mlp, ctx,
    )
    var aff = weights._linear_b(mod_aff, "audio_ff.net.0.proj.weight", "audio_ff.net.0.proj.bias", ctx)
    aff = gelu(aff, ctx)
    aff = weights._linear_b(aff, "audio_ff.net.2.weight", "audio_ff.net.2.bias", ctx)
    ahss = add(ahss, mul(a_gate_mlp, aff, ctx), ctx)

    return (hs^, ahss^, v2a_delta^)
