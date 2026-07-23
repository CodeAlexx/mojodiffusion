# serenitymojo/models/mageflow/weights.mojo — Mage-Flow-Base weight loading.
#
# RESIDENT non-block weights (img_in, txt_norm+txt_in, timestep-embedder MLP,
# norm_out AdaLN linear, proj_out) loaded from the single-file diffusers
# transformer safetensors, mirroring the key set the frozen inference loader
# uses (models/dit/mageflow_dit.mojo::MageFlowDiT.load comment block +
# _img_in/_txt_in/_temb) and the QwenOffloadBase layout
# (models/qwenimage/qwenimage_stack_lora.mojo:864-887).
#
# Per-block weights are NOT loaded here: they stream through TurboPlannedLoader
# (build_mageflow_block_plan below) and are materialized per iteration by the
# qwenimage offload reader `_double_block_weights_from_block`
# (qwenimage_stack_lora.mojo:793-798) — the key names are IDENTICAL
# (transformer_blocks.N.attn.to_q/..., img_mlp.net.0.proj, img_mod.1, ...);
# verified against the Mage-Flow-Turbo safetensors header (same arch as Base).
#
# Also hosts the mage bf16-rounded timestep sinusoid (free-function replica of
# the FROZEN MageFlowDiT._mage_time_sinusoid, mageflow_dit.mojo:249-272) and
# the silu(temb) pipeline the per-block/final mod computation consumes
# (mirrors compute_silu_temb, qwenimage_stack_lora.mojo:892-911).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.offload.plan import BlockPlan, BlockKind
from serenitymojo.models.qwenimage.qwenimage_stack import QwenStackBase

from serenitymojo.models.mageflow.config import MageFlowTrainSpec


comptime TArc = ArcPointer[Tensor]


# ── single-file safetensors -> host lists (wan22 trainer pattern) ─────────────
def _st_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _st_host_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    return cast_tensor(_st_tensor(st, name, ctx), STDtype.F32, ctx).to_host(ctx)


def _st_host_bf16(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[BFloat16]:
    var t = _st_tensor(st, name, ctx)
    if t.dtype() == STDtype.BF16:
        return t.to_host_bf16(ctx)
    return cast_tensor(t, STDtype.BF16, ctx).to_host_bf16(ctx)


def _require_shape(st: SafeTensors, name: String, d0: Int, d1: Int) raises:
    # d1 < 0 => rank-1 [d0]
    var info = st.tensor_info(name)
    if d1 < 0:
        if len(info.shape) != 1 or info.shape[0] != d0:
            raise Error(String("mageflow weights: ") + name + " shape != [" + String(d0) + "]")
        return
    if len(info.shape) != 2 or info.shape[0] != d0 or info.shape[1] != d1:
        raise Error(
            String("mageflow weights: ") + name + " shape != ["
            + String(d0) + "," + String(d1) + "]"
        )


# ── resident non-block weights ────────────────────────────────────────────────
# stack: img_in / txt_in / proj_out (QwenStackBase — the offload stack's frozen
# linears). txt_norm: RMSNorm(2560) applied to the RAW Qwen3-VL context BEFORE
# txt_in (mageflow_dit.mojo::_txt_in; the qwen stack takes pre-normed txt, so
# the mageflow stack applies this itself). norm_out/te_*: host BF16 lists in
# the QwenOffloadBase convention (qwenimage_stack_lora.mojo:864-887).
struct MageFlowBase(Movable):
    var stack: QwenStackBase
    var txt_norm_w: TArc             # [2560]  txt_norm.weight (RMSNorm scale)
    var norm_out_w: List[BFloat16]   # [2D, D] norm_out.linear.weight
    var norm_out_b: List[BFloat16]   # [2D]
    var te_lin1_w: List[BFloat16]    # [D, 256] timestep_embedder.linear_1
    var te_lin1_b: List[BFloat16]    # [D]
    var te_lin2_w: List[BFloat16]    # [D, D]   timestep_embedder.linear_2
    var te_lin2_b: List[BFloat16]    # [D]

    def __init__(
        out self,
        var stack: QwenStackBase, var txt_norm_w: TArc,
        var norm_out_w: List[BFloat16], var norm_out_b: List[BFloat16],
        var te_lin1_w: List[BFloat16], var te_lin1_b: List[BFloat16],
        var te_lin2_w: List[BFloat16], var te_lin2_b: List[BFloat16],
    ):
        self.stack = stack^
        self.txt_norm_w = txt_norm_w^
        self.norm_out_w = norm_out_w^
        self.norm_out_b = norm_out_b^
        self.te_lin1_w = te_lin1_w^
        self.te_lin1_b = te_lin1_b^
        self.te_lin2_w = te_lin2_w^
        self.te_lin2_b = te_lin2_b^


def load_mageflow_base(
    st: SafeTensors, spec: MageFlowTrainSpec, ctx: DeviceContext
) raises -> MageFlowBase:
    var D = spec.inner_dim
    var in_ch = spec.in_channels
    var txt_ch = spec.context_in_dim
    var out_ch = spec.out_channels
    var td = spec.timestep_dim

    # CONFIRM dims against the REAL on-disk header (Base transformer).
    _require_shape(st, String("img_in.weight"), D, in_ch)
    _require_shape(st, String("txt_in.weight"), D, txt_ch)
    _require_shape(st, String("txt_norm.weight"), txt_ch, -1)
    _require_shape(st, String("time_text_embed.timestep_embedder.linear_1.weight"), D, td)
    _require_shape(st, String("time_text_embed.timestep_embedder.linear_2.weight"), D, D)
    _require_shape(st, String("norm_out.linear.weight"), 2 * D, D)
    _require_shape(st, String("proj_out.weight"), out_ch, D)
    # depth check: last block must exist, one past must not.
    var last_key = String("transformer_blocks.") + String(spec.depth - 1) \
        + String(".attn.to_q.weight")
    _require_shape(st, last_key, D, D)

    var stack = QwenStackBase(
        _st_host_f32(st, String("img_in.weight"), ctx),
        _st_host_f32(st, String("img_in.bias"), ctx),
        _st_host_f32(st, String("txt_in.weight"), ctx),
        _st_host_f32(st, String("txt_in.bias"), ctx),
        _st_host_f32(st, String("proj_out.weight"), ctx),
        _st_host_f32(st, String("proj_out.bias"), ctx),
        D, in_ch, txt_ch, out_ch, ctx,
    )
    var txt_norm = TArc(cast_tensor(
        _st_tensor(st, String("txt_norm.weight"), ctx), STDtype.BF16, ctx, False
    ))
    return MageFlowBase(
        stack^, txt_norm^,
        _st_host_bf16(st, String("norm_out.linear.weight"), ctx),
        _st_host_bf16(st, String("norm_out.linear.bias"), ctx),
        _st_host_bf16(st, String("time_text_embed.timestep_embedder.linear_1.weight"), ctx),
        _st_host_bf16(st, String("time_text_embed.timestep_embedder.linear_1.bias"), ctx),
        _st_host_bf16(st, String("time_text_embed.timestep_embedder.linear_2.weight"), ctx),
        _st_host_bf16(st, String("time_text_embed.timestep_embedder.linear_2.bias"), ctx),
    )


# ── block-swap plan: 12 double blocks, qwenimage key layout ───────────────────
# Hints measured from the Mage-Flow-Turbo transformer header: 32 tensors,
# 679,662,592 bytes per block (same per-block payload as qwen-image at D=3072).
def build_mageflow_block_plan(depth: Int) -> BlockPlan:
    var plan = BlockPlan(String("mageflow"))
    for i in range(depth):
        plan.append(
            String("transformer_blocks.") + String(i),
            BlockKind.double_stream(),
            32,
            679662592,
        )
    return plan^


# ── mage bf16-rounded timestep sinusoid (FROZEN helper replica) ───────────────
# Free-function replica of MageFlowDiT._mage_time_sinusoid
# (models/dit/mageflow_dit.mojo:249-272): get_timestep_embedding with
# num_channels=256, flip_sin_to_cos=True (COS first), downscale_freq_shift=0,
# scale=1000, max_period=10000; BOTH the timestep and the freq table are
# bf16-rounded BEFORE the multiply (mage_layers.py:45 + mage_flow.py:112).
# Returns the host [256] sinusoid; the caller uploads it as BF16 (matching the
# .to(bf16) before linear_1, mage_layers.py:98).
def mage_time_sinusoid_host(timestep: Float32) raises -> List[Float32]:
    comptime half = 128
    var t_bf = timestep.cast[DType.bfloat16]().cast[DType.float32]()
    var neg_ln = -flog(Float32(10000.0))
    var cos_part = List[Float32]()
    var sin_part = List[Float32]()
    for i in range(half):
        var exponent = neg_ln * (Float32(i) / Float32(half))
        var freq = fexp(exponent)
        var freq_bf = freq.cast[DType.bfloat16]().cast[DType.float32]()
        var angle = Float32(1000.0) * (t_bf * freq_bf)
        cos_part.append(fcos(angle))
        sin_part.append(fsin(angle))
    var vals = List[Float32]()
    for i in range(half):
        vals.append(cos_part[i])
    for i in range(half):
        vals.append(sin_part[i])
    return vals^


# ── silu(temb) for the per-block + final mod computation ──────────────────────
# temb = linear_2(silu(linear_1(sinusoid)));  returned value = silu(temb).
# Mirrors compute_silu_temb (qwenimage_stack_lora.mojo:892-911): the offload
# stack's `_modvecs_from_block` and `_compute_final_modvecs` both take the
# PRE-ACTIVATED silu(temb) and apply only the frozen mod linear (each block's
# img_mod/txt_mod is Sequential(SiLU, Linear) — index .1 is the linear).
# timestep is the RAW sigma in [0,1] (the sinusoid folds scale=1000), exactly
# what inference feeds (mage pipeline _velocity passes scheduler sigma).
def compute_mageflow_silu_temb(
    base: MageFlowBase, timestep: Float32, spec: MageFlowTrainSpec,
    ctx: DeviceContext,
) raises -> List[Float32]:
    var D = spec.inner_dim
    var td = spec.timestep_dim
    var sinus_h = mage_time_sinusoid_host(timestep)
    var t_emb = Tensor.from_host(sinus_h^, [1, td], STDtype.BF16, ctx)
    var b1 = Tensor.from_host_bf16(base.te_lin1_b.copy(), [D], ctx)
    var h1 = linear(
        t_emb,
        Tensor.from_host_bf16(base.te_lin1_w.copy(), [D, td], ctx),
        Optional[Tensor](b1^), ctx,
    )
    var h1_silu = silu(h1, ctx)
    var b2 = Tensor.from_host_bf16(base.te_lin2_b.copy(), [D], ctx)
    var temb_out = linear(
        h1_silu,
        Tensor.from_host_bf16(base.te_lin2_w.copy(), [D, D], ctx),
        Optional[Tensor](b2^), ctx,
    )
    return silu(temb_out, ctx).to_host(ctx)
