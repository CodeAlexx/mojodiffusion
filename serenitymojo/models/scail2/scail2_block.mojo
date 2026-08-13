# One SCAIL-2 transformer block over the existing Wan GPU primitives.
#
# Oracle: zai-org/SCAIL-2 commit 5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5,
# wan/modules/model_scail2.py::WanAttentionBlock and WanI2VCrossAttention.
# The image and text attentions are deliberately separate: their outputs are
# added before the single cross_attn.o projection, exactly as the oracle does.

from std.collections import Dict
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.wan22_dit import (
    _chunk6,
    _from_bshd,
    _lin,
    _to_bshd,
    _wan22_sdpa_infer,
    _wan22_sdpa_infer_rect,
    wan22_mod_pre,
)
from serenitymojo.models.scail2.scail2_config import Scail2Config
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.fp8_cublas_linear import fp8_cublas_linear_outer
from serenitymojo.ops.norm import layer_norm, rms_norm
from serenitymojo.ops.rope import (
    rope_interleaved,
    rope_interleaved_head_broadcast,
)
from serenitymojo.ops.tensor_algebra import add, mul
from serenitymojo.tensor import Tensor


comptime TArc = ArcPointer[Tensor]


def _scail2_lin(
    x: Tensor,
    w: Dict[String, TArc],
    weight_name: String,
    bias_name: String,
    ctx: DeviceContext,
) raises -> Tensor:
    if w[weight_name][].dtype() == STDtype.F8_E4M3:
        var scale_name = weight_name + String(".__fp8_scale")
        if scale_name not in w:
            raise Error(String("SCAIL-2 FP8 scale missing: ") + scale_name)
        return fp8_cublas_linear_outer(
            x, w[weight_name][], w[scale_name][], w[bias_name][], ctx
        )
    return _lin(x, w, weight_name, bias_name, ctx)


def scail2_gated_residual_f32(
    x: Tensor, y: Tensor, gate_f32: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Pinned SCAIL residual stream: accumulate and retain F32."""
    return add(
        cast_tensor(x, STDtype.F32, ctx),
        mul(cast_tensor(y, STDtype.F32, ctx), gate_f32, ctx),
        ctx,
    )


def scail2_residual_f32(
    x: Tensor, y: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Ungated SCAIL cross-attention residual, retained in F32."""
    return add(
        cast_tensor(x, STDtype.F32, ctx),
        cast_tensor(y, STDtype.F32, ctx),
        ctx,
    )


def scail2_block_forward[
    S: Int, TXT: Int, IMG: Int, H: Int, DH: Int
](
    x: Tensor,
    e0: Tensor,
    text_context: Tensor,
    image_context: Tensor,
    cos: Tensor,
    sin: Tensor,
    w: Dict[String, TArc],
    cfg: Scail2Config,
    ctx: DeviceContext,
    text_valid: Int = TXT,
) raises -> Tensor:
    """Exact inference forward for one streamed SCAIL-2 block."""
    cfg.validate()
    if IMG != cfg.image_len:
        raise Error("SCAIL-2 image context must contain exactly 257 tokens")
    if text_valid != TXT:
        # The pinned forward pads to 512 then passes context_lens=None. Masking
        # padding would be a different attention distribution.
        raise Error("SCAIL-2 oracle attends all 512 padded text tokens")
    var dim = cfg.dim
    var scale = 1.0 / Float32(DH) ** Float32(0.5)

    # e0 is the batch-1 broadcastable [1,1,6,D] form. The official expanded
    # [1,6,D] row is identical for every token, so no S-sized F32 copy is made.
    var modulation_rows = 1
    if e0.numel() != 6 * dim:
        if e0.numel() == S * 6 * dim:
            modulation_rows = S
        else:
            raise Error("SCAIL-2 block timestep modulation shape mismatch")
    var block_mod = cast_tensor(w["modulation"][], STDtype.F32, ctx)
    var e_all = add(block_mod, e0, ctx)
    var shift_sa = _chunk6(e_all, 0, modulation_rows, dim, ctx)
    var scale_sa = _chunk6(e_all, 1, modulation_rows, dim, ctx)
    var gate_sa = _chunk6(e_all, 2, modulation_rows, dim, ctx)
    var shift_ffn = _chunk6(e_all, 3, modulation_rows, dim, ctx)
    var scale_ffn = _chunk6(e_all, 4, modulation_rows, dim, ctx)
    var gate_ffn = _chunk6(e_all, 5, modulation_rows, dim, ctx)

    # Self attention with the SCAIL-specific prebuilt RoPE table.
    var sa_in = cast_tensor(
        wan22_mod_pre(x, scale_sa, shift_sa, dim, cfg.eps, ctx),
        STDtype.BF16,
        ctx,
    )
    var q = _scail2_lin(sa_in, w, "self_attn.q.weight", "self_attn.q.bias", ctx)
    var k = _scail2_lin(sa_in, w, "self_attn.k.weight", "self_attn.k.bias", ctx)
    var v = _scail2_lin(sa_in, w, "self_attn.v.weight", "self_attn.v.bias", ctx)
    q = rms_norm(q, w["self_attn.norm_q.weight"][], cfg.eps, ctx)
    k = rms_norm(k, w["self_attn.norm_k.weight"][], cfg.eps, ctx)
    var q4 = _to_bshd(q, S, H, DH, ctx)
    var k4 = _to_bshd(k, S, H, DH, ctx)
    var v4 = _to_bshd(v, S, H, DH, ctx)
    if cos.numel() == S * (DH // 2) and sin.numel() == S * (DH // 2):
        q4 = rope_interleaved_head_broadcast(q4, cos, sin, H, ctx)
        k4 = rope_interleaved_head_broadcast(k4, cos, sin, H, ctx)
    elif (
        cos.numel() == S * H * (DH // 2)
        and sin.numel() == S * H * (DH // 2)
    ):
        # Retain expanded-table support for independent real-block parity
        # fixtures and callers that already own such a table.
        q4 = rope_interleaved(q4, cos, sin, ctx)
        k4 = rope_interleaved(k4, cos, sin, ctx)
    else:
        raise Error("SCAIL-2 block RoPE table shape mismatch")
    var self_attn: Tensor
    comptime if S > 512:
        self_attn = _wan22_sdpa_infer[S, H, DH](q4, k4, v4, scale, ctx)
    else:
        self_attn = sdpa_nomask[1, S, H, DH](q4, k4, v4, scale, ctx)
    var self_out = _scail2_lin(
        _from_bshd(self_attn, S, dim, ctx),
        w,
        "self_attn.o.weight",
        "self_attn.o.bias",
        ctx,
    )
    var x_sa = scail2_gated_residual_f32(x, self_out, gate_sa, ctx)

    # I2V cross attention. Query is shared; text and image K/V projections and
    # flash attentions are distinct. Add their flattened outputs before o.
    var n3 = layer_norm(
        cast_tensor(x_sa, w["norm3.weight"][].dtype(), ctx),
        w["norm3.weight"][], w["norm3.bias"][], cfg.eps, ctx
    )
    var caq = _scail2_lin(n3, w, "cross_attn.q.weight", "cross_attn.q.bias", ctx)
    caq = rms_norm(caq, w["cross_attn.norm_q.weight"][], cfg.eps, ctx)
    var text_k = _scail2_lin(
        text_context, w, "cross_attn.k.weight", "cross_attn.k.bias", ctx
    )
    var text_v = _scail2_lin(
        text_context, w, "cross_attn.v.weight", "cross_attn.v.bias", ctx
    )
    text_k = rms_norm(
        text_k, w["cross_attn.norm_k.weight"][], cfg.eps, ctx
    )
    var image_k = _scail2_lin(
        image_context, w, "cross_attn.k_img.weight", "cross_attn.k_img.bias", ctx
    )
    var image_v = _scail2_lin(
        image_context, w, "cross_attn.v_img.weight", "cross_attn.v_img.bias", ctx
    )
    image_k = rms_norm(
        image_k, w["cross_attn.norm_k_img.weight"][], cfg.eps, ctx
    )

    var q_cross = _to_bshd(caq, S, H, DH, ctx)
    var k_text = _to_bshd(text_k, TXT, H, DH, ctx)
    var v_text = _to_bshd(text_v, TXT, H, DH, ctx)
    var text_attn = _wan22_sdpa_infer_rect[S, TXT, H, DH](
        q_cross, k_text, v_text, scale, ctx
    )
    var image_attn = _wan22_sdpa_infer_rect[S, IMG, H, DH](
        q_cross,
        _to_bshd(image_k, IMG, H, DH, ctx),
        _to_bshd(image_v, IMG, H, DH, ctx),
        scale,
        ctx,
    )
    var joined_attn = add(
        _from_bshd(text_attn, S, dim, ctx),
        _from_bshd(image_attn, S, dim, ctx),
        ctx,
    )
    var cross_out = _scail2_lin(
        joined_attn, w, "cross_attn.o.weight", "cross_attn.o.bias", ctx
    )
    var x_ca = scail2_residual_f32(x_sa, cross_out, ctx)

    var ffn_in = cast_tensor(
        wan22_mod_pre(x_ca, scale_ffn, shift_ffn, dim, cfg.eps, ctx),
        STDtype.BF16,
        ctx,
    )
    var hidden = _scail2_lin(ffn_in, w, "ffn.0.weight", "ffn.0.bias", ctx)
    hidden = gelu(hidden, ctx)
    var ffn_out = _scail2_lin(hidden, w, "ffn.2.weight", "ffn.2.bias", ctx)
    return scail2_gated_residual_f32(x_ca, ffn_out, gate_ffn, ctx)
