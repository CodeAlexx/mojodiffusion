# Ideogram4LoRATrainStep.mojo — ONE real LoRA training step for the Ideogram-4
# BLOCK adapters (the 34×6 per-layer LoRA targets).
#
# Wires the verified predict packing + the verified block-stack LoRA fwd/bwd +
# AdamW into a single flow-matching training step:
#
#   embed(FROZEN) → stack_lora_forward → final(FROZEN) → MSE flow loss
#     → final-backward → stack_lora_backward → AdamW
#
# FROZEN in THIS chunk: the embed layers (input_proj, t_embedding, adaln_proj,
# llm_cond_*, embed_image_indicator) and the final layers (final_layer.linear,
# final_layer.adaln_modulation). ONLY the 34×6 block adapters train. The 7
# global-target adapters are a LATER chunk — NOT attempted here.
#
# ── SOURCE MAP (serenitymojo/models/dit/ideogram4_dit.mojo) ───────────────────
# ideogram4_forward_prefinal_hidden[S]  (lines 220-300):
#   embed (lines 230-273): input_proj + t_embedding→adaln_proj→silu→adaln_input
#     + llm_cond + image-indicator embed → x_in (the [1,S,Hidden] fed to block 0).
#   block loop (276-293): REPLACED by ideogram4_stack_lora_forward.
#   pre-final (295-300): final_layer.adaln_modulation(silu(adaln_input)) → fscale;
#     hn = layer_norm_no_affine(h,1e-6) * fscale.
# ideogram4_forward[S] (303-320): hn → final_layer.linear → F32 cast → out.
#
# We SPLIT it: ideogram4_lora_embed = lines 230-273 (produces x_in + adaln_input);
# ideogram4_lora_final_forward = lines 295-300 + 317-320 (pre-final + final linear).
# The block loop between them is the trainable stack.
#
# Velocity convention is byte-identical to Ideogram4Predict.ideogram4_predict_velocity
# (slice image rows → reshape [1,GH,GW,128] → permute [0,3,1,2] → negate), so with
# B=0 (zero-init LoRA b) the produced velocity matches the predict B=0 gate / oracle.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors

from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm_no_affine
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.ops.reduce import reduce_mean_f32
from serenitymojo.ops.tensor_algebra import (
    add, sub, mul, mul_scalar, add_scalar,
    reshape, slice, concat, permute, zeros_device, gather_rows,
)
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward_dx

from serenitymojo.models.dit.ideogram4_dit import (
    load_w_fp8, load_w_bf16, ideogram4_t_embedding,
)
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights

from serenity_trainer.model.Ideogram4LoRABlock import (
    Ideogram4LoraSet,
    Ideogram4StackForward,
    Ideogram4StackLoraGrads,
    ideogram4_stack_lora_forward,
    ideogram4_stack_lora_backward,
    ideogram4_stack_lora_backward_graph,
    ideogram4_stack_lora_forward_resident,
    ideogram4_stack_lora_forward_resident_nosave,
    ideogram4_stack_lora_backward_resident,
    ideogram4_stack_lora_backward_graph_resident,
    ideogram4_stack_lora_forward_resident_b2,
    ideogram4_stack_lora_backward_resident_b2,
)
from serenity_trainer.trainer.Ideogram4StackTrain import (
    IDEOGRAM4_V2_GRAPH_PATH,
)
from serenity_trainer.model.Ideogram4Predict import (
    ideogram4_build_packed_inputs,
    ideogram4_flow_target,
)
from serenity_trainer.trainer.Ideogram4StackTrain import (
    Ideogram4LoraAdamState,
    Ideogram4StackTrainResult,
    apply_ideogram4_lora_grads,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig

# T1 levers (TIER1_PARITY_CAMPAIGN_2026-06-11.md): the shared runtime-config
# loss dispatch. LeversConfig is serenitymojo's TrainConfig (the lever-key
# carrier), DISTINCT from serenity-trainer's own TrainConfig above — the
# serenity-trainer struct keeps owning the AdamW/LoRA recipe scalars.
from serenitymojo.training.levers import levers_loss_active, levers_loss_grad
from serenitymojo.training.train_config import TrainConfig as LeversConfig
from serenity_trainer.modelSampler.Ideogram4Sampler import (
    IDEOGRAM4_NUM_LAYERS,
    IDEOGRAM4_NUM_HEADS,
    IDEOGRAM4_HEAD_DIM,
    IDEOGRAM4_HIDDEN,
    IDEOGRAM4_INTERMEDIATE_SIZE,
    IDEOGRAM4_ADALN_DIM,
    IDEOGRAM4_PACKED_CHANNELS,
    IDEOGRAM4_TEXT_FEATURE_DIM,
    IDEOGRAM4_MROPE_SECTION_0,
    IDEOGRAM4_MROPE_SECTION_1,
    IDEOGRAM4_MROPE_SECTION_2,
    IDEOGRAM4_MROPE_THETA,
    IDEOGRAM4_IMAGE_OFFSET,
    IDEOGRAM4_LLM_TOKEN_INDICATOR,
    IDEOGRAM4_OUTPUT_IMAGE_INDICATOR,
)


comptime I4_FINAL_EPS = Float32(1.0e-6)


# ── result of a full training step ────────────────────────────────────────────
@fieldwise_init
struct Ideogram4LoRATrainResult(Copyable, Movable):
    var loss: Float32
    var adapter_b_l1: Float32
    var did_update: Bool


# ── embed-only output (lines 230-273): x_in fed to block 0 + adaln_input ───────
struct Ideogram4EmbedOut(Movable):
    var x_in: Tensor          # [1, SEQ, Hidden] bf16
    var adaln_input: Tensor   # [1, 1, Adaln]    bf16

    def __init__(out self, var x_in: Tensor, var adaln_input: Tensor):
        self.x_in = x_in^
        self.adaln_input = adaln_input^


# ── forward result: velocity for the loss + everything the backward needs ─────
struct Ideogram4LoRATrainForward(Movable):
    var velocity: Tensor          # [1, 128, GH, GW] F32 (matches predict velocity)
    var stack_fwd: Ideogram4StackForward
    var h: Tensor                 # [1, SEQ, Hidden] bf16 — stack output (for ln bwd)
    var fscale: Tensor            # [1, 1, Hidden] bf16 — 1 + final adaln_modulation
    var flw: Tensor               # [128, Hidden] bf16 — final_layer.linear weight (frozen)
    var adaln_input: Tensor       # [1, 1, Adaln] bf16 — silu'd adaln (stack bwd needs it)
    var cosf: Tensor              # [1, SEQ, Dh] bf16
    var sinf: Tensor              # [1, SEQ, Dh] bf16

    def __init__(
        out self,
        var velocity: Tensor,
        var stack_fwd: Ideogram4StackForward,
        var h: Tensor,
        var fscale: Tensor,
        var flw: Tensor,
        var adaln_input: Tensor,
        var cosf: Tensor,
        var sinf: Tensor,
    ):
        self.velocity = velocity^
        self.stack_fwd = stack_fwd^
        self.h = h^
        self.fscale = fscale^
        self.flw = flw^
        self.adaln_input = adaln_input^
        self.cosf = cosf^
        self.sinf = sinf^


# ──────────────────────────────────────────────────────────────────────────────
# EMBED (FROZEN) — 1:1 mirror of ideogram4_dit.mojo lines 230-273.
#   masks from indicator (host): llm==3, image==2, image-ids 0/1.
#   x = input_proj(x_bf)*img_mask ; adaln_input = silu(adaln_proj(t_embed(t)));
#   llm = llm_cond_proj(rms_norm(llm_bf,1e-6))*llm_mask ;
#   x_in = x + llm + embed_image_indicator[img_ids].
# Returns x_in [1,SEQ,Hidden] (fed to block 0) and adaln_input [1,1,Adaln].
# ──────────────────────────────────────────────────────────────────────────────
def ideogram4_lora_embed(
    st: ShardedSafeTensors,
    x_bf: Tensor,            # [1, SEQ, 128]    bf16 (packed, text rows zeroed)
    llm_bf: Tensor,          # [1, SEQ, 53248]  bf16 (packed, image rows zeroed)
    model_t: Tensor,         # [1] f32 (= 1 - t_flow)
    indicator: Tensor,       # [1, SEQ] f32 (text→3, image→2)
    hidden: Int,
    ctx: DeviceContext,
) raises -> Ideogram4EmbedOut:
    var L = x_bf.shape()[1]

    # masks from indicator (dit lines 232-244)
    var ind_h = indicator.to_host(ctx)
    var llm_mask_v = List[Float32]()
    var img_mask_v = List[Float32]()
    var img_ids = List[Int]()
    for i in range(L):
        var vi = ind_h[i]
        llm_mask_v.append(Float32(1.0) if (vi > 2.5 and vi < 3.5) else Float32(0.0))
        var is_img = (vi > 1.5 and vi < 2.5)
        img_mask_v.append(Float32(1.0) if is_img else Float32(0.0))
        img_ids.append(1 if is_img else 0)
    var llm_mask = Tensor.from_host(llm_mask_v^, [1, L, 1], STDtype.BF16, ctx)
    var img_mask = Tensor.from_host(img_mask_v^, [1, L, 1], STDtype.BF16, ctx)

    # input_proj (dit lines 246-250)
    var llm = mul(llm_bf, llm_mask, ctx)
    var x = mul(x_bf, img_mask, ctx)
    var ipw = load_w_fp8(st, "input_proj.weight", ctx)
    var ipb = load_w_bf16(st, "input_proj.bias", ctx)
    x = mul(linear(x, ipw, Optional[Tensor](ipb.clone(ctx)), ctx), img_mask, ctx)

    # t → adaln_input (dit lines 252-260)
    var miw = load_w_fp8(st, "t_embedding.mlp_in.weight", ctx)
    var mib = load_w_bf16(st, "t_embedding.mlp_in.bias", ctx)
    var mow = load_w_fp8(st, "t_embedding.mlp_out.weight", ctx)
    var mob = load_w_bf16(st, "t_embedding.mlp_out.bias", ctx)
    var t_cond = reshape(
        ideogram4_t_embedding(model_t, hidden, miw, mib, mow, mob, ctx),
        [1, 1, hidden], ctx,
    )
    var apw = load_w_fp8(st, "adaln_proj.weight", ctx)
    var apb = load_w_bf16(st, "adaln_proj.bias", ctx)
    var adaln_input = silu(
        linear(t_cond, apw, Optional[Tensor](apb.clone(ctx)), ctx), ctx
    )  # [1,1,Adaln]

    # llm conditioning (dit lines 262-267)
    var lcn = load_w_bf16(st, "llm_cond_norm.weight", ctx)
    llm = rms_norm(llm, lcn, Float32(1.0e-6), ctx)
    var lcpw = load_w_fp8(st, "llm_cond_proj.weight", ctx)
    var lcpb = load_w_bf16(st, "llm_cond_proj.bias", ctx)
    llm = mul(linear(llm, lcpw, Optional[Tensor](lcpb.clone(ctx)), ctx), llm_mask, ctx)

    # x_in = x + llm + image-indicator embed (dit lines 269-273)
    var h = add(x, llm, ctx)
    var eii = load_w_bf16(st, "embed_image_indicator.weight", ctx)  # [2,hidden]
    var iemb = reshape(gather_rows(eii, img_ids, ctx), [1, L, hidden], ctx)
    h = add(h, iemb, ctx)

    return Ideogram4EmbedOut(h^, adaln_input^)


def _resident_w_fp8(
    rw: Ideogram4Weights, name: String, ctx: DeviceContext
) raises -> Tensor:
    return fp8_e4m3_dequant_perrow_to_bf16(
        rw.w(name), rw.w(name + String("_scale")), ctx
    )


def ideogram4_lora_embed_resident(
    rw: Ideogram4Weights,
    x_bf: Tensor,
    llm_bf: Tensor,
    model_t: Tensor,
    indicator: Tensor,
    hidden: Int,
    ctx: DeviceContext,
) raises -> Ideogram4EmbedOut:
    var L = x_bf.shape()[1]

    var ind_h = indicator.to_host(ctx)
    var llm_mask_v = List[Float32]()
    var img_mask_v = List[Float32]()
    var img_ids = List[Int]()
    for i in range(L):
        var vi = ind_h[i]
        llm_mask_v.append(Float32(1.0) if (vi > 2.5 and vi < 3.5) else Float32(0.0))
        var is_img = (vi > 1.5 and vi < 2.5)
        img_mask_v.append(Float32(1.0) if is_img else Float32(0.0))
        img_ids.append(1 if is_img else 0)
    var llm_mask = Tensor.from_host(llm_mask_v^, [1, L, 1], STDtype.BF16, ctx)
    var img_mask = Tensor.from_host(img_mask_v^, [1, L, 1], STDtype.BF16, ctx)

    var llm = mul(llm_bf, llm_mask, ctx)
    var x = mul(x_bf, img_mask, ctx)
    var ipw = _resident_w_fp8(rw, String("input_proj.weight"), ctx)
    x = mul(
        linear(x, ipw, Optional[Tensor](rw.w(String("input_proj.bias")).clone(ctx)), ctx),
        img_mask,
        ctx,
    )

    var miw = _resident_w_fp8(rw, String("t_embedding.mlp_in.weight"), ctx)
    var mow = _resident_w_fp8(rw, String("t_embedding.mlp_out.weight"), ctx)
    var t_cond = reshape(
        ideogram4_t_embedding(
            model_t,
            hidden,
            miw,
            rw.w(String("t_embedding.mlp_in.bias")).clone(ctx),
            mow,
            rw.w(String("t_embedding.mlp_out.bias")).clone(ctx),
            ctx,
        ),
        [1, 1, hidden], ctx,
    )
    var apw = _resident_w_fp8(rw, String("adaln_proj.weight"), ctx)
    var adaln_input = silu(
        linear(t_cond, apw, Optional[Tensor](rw.w(String("adaln_proj.bias")).clone(ctx)), ctx),
        ctx,
    )

    llm = rms_norm(llm, rw.w(String("llm_cond_norm.weight")), Float32(1.0e-6), ctx)
    var lcpw = _resident_w_fp8(rw, String("llm_cond_proj.weight"), ctx)
    llm = mul(
        linear(llm, lcpw, Optional[Tensor](rw.w(String("llm_cond_proj.bias")).clone(ctx)), ctx),
        llm_mask,
        ctx,
    )

    var h = add(x, llm, ctx)
    var iemb = reshape(
        gather_rows(rw.w(String("embed_image_indicator.weight")), img_ids, ctx),
        [1, L, hidden],
        ctx,
    )
    h = add(h, iemb, ctx)

    return Ideogram4EmbedOut(h^, adaln_input^)


def ideogram4_lora_embed_resident_split[
    NT: Int, GH: Int, GW: Int
](
    rw: Ideogram4Weights,
    noisy_latents: Tensor,
    llm_features: Tensor,
    model_t: Tensor,
    hidden: Int,
    ctx: DeviceContext,
    text_len: Int = NT,
) raises -> Ideogram4EmbedOut:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG

    var real_len = text_len
    if real_len > NT:
        real_len = NT
    if real_len < 0:
        real_len = 0

    var text_mask_v = List[Float32]()
    var img_ids = List[Int]()
    for i in range(NT):
        if i < real_len:
            text_mask_v.append(Float32(1.0))
        else:
            text_mask_v.append(Float32(0.0))
        img_ids.append(0)
    for _ in range(NIMG):
        img_ids.append(1)
    var text_mask = Tensor.from_host(text_mask_v^, [1, NT, 1], STDtype.BF16, ctx)

    var text_h: Tensor
    if real_len == 0:
        text_h = zeros_device([1, NT, hidden], STDtype.BF16, ctx)
    else:
        var llm_bf = cast_tensor(llm_features, STDtype.BF16, ctx)
        var llm = mul(llm_bf, text_mask, ctx)
        llm = rms_norm(llm, rw.w(String("llm_cond_norm.weight")), Float32(1.0e-6), ctx)
        var lcpw = _resident_w_fp8(rw, String("llm_cond_proj.weight"), ctx)
        text_h = mul(
            linear(
                llm,
                lcpw,
                Optional[Tensor](rw.w(String("llm_cond_proj.bias")).clone(ctx)),
                ctx,
            ),
            text_mask,
            ctx,
        )

    var img_perm = permute(noisy_latents, [0, 2, 3, 1], ctx)
    var image_tokens = reshape(
        img_perm, [1, NIMG, IDEOGRAM4_PACKED_CHANNELS], ctx
    )
    var x_bf = cast_tensor(image_tokens, STDtype.BF16, ctx)
    var ipw = _resident_w_fp8(rw, String("input_proj.weight"), ctx)
    var image_h = linear(
        x_bf, ipw, Optional[Tensor](rw.w(String("input_proj.bias")).clone(ctx)), ctx
    )

    var miw = _resident_w_fp8(rw, String("t_embedding.mlp_in.weight"), ctx)
    var mow = _resident_w_fp8(rw, String("t_embedding.mlp_out.weight"), ctx)
    var t_cond = reshape(
        ideogram4_t_embedding(
            model_t,
            hidden,
            miw,
            rw.w(String("t_embedding.mlp_in.bias")).clone(ctx),
            mow,
            rw.w(String("t_embedding.mlp_out.bias")).clone(ctx),
            ctx,
        ),
        [1, 1, hidden], ctx,
    )
    var apw = _resident_w_fp8(rw, String("adaln_proj.weight"), ctx)
    var adaln_input = silu(
        linear(t_cond, apw, Optional[Tensor](rw.w(String("adaln_proj.bias")).clone(ctx)), ctx),
        ctx,
    )

    var h = concat(1, ctx, text_h, image_h)
    var iemb = reshape(
        gather_rows(rw.w(String("embed_image_indicator.weight")), img_ids, ctx),
        [1, SEQ, hidden],
        ctx,
    )
    h = add(h, iemb, ctx)

    return Ideogram4EmbedOut(h^, adaln_input^)


# ──────────────────────────────────────────────────────────────────────────────
# FINAL forward (FROZEN) — dit lines 295-300 (pre-final) + 317-320 (final linear).
#   fscale = 1 + final_layer.adaln_modulation(silu(adaln_input))    [1,1,Hidden]
#   hn     = layer_norm_no_affine(h,1e-6) * fscale                  [1,SEQ,Hidden]
#   out    = final_layer.linear(hn) → F32                           [1,SEQ,128]
# Returns (out, fscale, flw) — fscale + flw are reused by the backward.
# ──────────────────────────────────────────────────────────────────────────────
struct _FinalFwd(Movable):
    var out: Tensor       # [1,SEQ,128] F32
    var fscale: Tensor    # [1,1,Hidden] bf16
    var flw: Tensor       # [128,Hidden] bf16

    def __init__(out self, var out: Tensor, var fscale: Tensor, var flw: Tensor):
        self.out = out^
        self.fscale = fscale^
        self.flw = flw^


def ideogram4_lora_final_forward(
    st: ShardedSafeTensors,
    h: Tensor,               # [1, SEQ, Hidden] bf16 (stack output)
    adaln_input: Tensor,     # [1, 1, Adaln] bf16
    ctx: DeviceContext,
) raises -> _FinalFwd:
    # pre-final modulation (dit 295-300)
    var fmw = load_w_fp8(st, "final_layer.adaln_modulation.weight", ctx)
    var fmb = load_w_bf16(st, "final_layer.adaln_modulation.bias", ctx)
    var fscale = add_scalar(
        linear(silu(adaln_input, ctx), fmw, Optional[Tensor](fmb.clone(ctx)), ctx),
        Float32(1.0), ctx,
    )                                                   # [1,1,Hidden]
    var hn = mul(layer_norm_no_affine(h, I4_FINAL_EPS, ctx), fscale, ctx)

    # final linear (dit 317-320)
    var flw = load_w_fp8(st, "final_layer.linear.weight", ctx)   # [128,Hidden]
    var flb = load_w_bf16(st, "final_layer.linear.bias", ctx)
    var out_bf = linear(hn, flw, Optional[Tensor](flb.clone(ctx)), ctx)  # [1,SEQ,128] bf16
    var out = cast_tensor(out_bf, STDtype.F32, ctx)
    return _FinalFwd(out^, fscale^, flw.clone(ctx))


def ideogram4_lora_final_forward_resident(
    rw: Ideogram4Weights,
    h: Tensor,
    adaln_input: Tensor,
    ctx: DeviceContext,
) raises -> _FinalFwd:
    var fmw = _resident_w_fp8(rw, String("final_layer.adaln_modulation.weight"), ctx)
    var fscale = add_scalar(
        linear(
            silu(adaln_input, ctx),
            fmw,
            Optional[Tensor](rw.w(String("final_layer.adaln_modulation.bias")).clone(ctx)),
            ctx,
        ),
        Float32(1.0), ctx,
    )
    var hn = mul(layer_norm_no_affine(h, I4_FINAL_EPS, ctx), fscale, ctx)

    var flw = _resident_w_fp8(rw, String("final_layer.linear.weight"), ctx)
    var out_bf = linear(
        hn,
        flw,
        Optional[Tensor](rw.w(String("final_layer.linear.bias")).clone(ctx)),
        ctx,
    )
    var out = cast_tensor(out_bf, STDtype.F32, ctx)
    return _FinalFwd(out^, fscale^, flw.clone(ctx))


# ──────────────────────────────────────────────────────────────────────────────
# FORWARD: noisy_latents → velocity (+ cached tensors for the backward).
#   packed inputs → mrope → embed → stack_lora_forward → final → velocity.
# ──────────────────────────────────────────────────────────────────────────────
def ideogram4_lora_train_forward[
    NT: Int, GH: Int, GW: Int
](
    st: ShardedSafeTensors,
    noisy_latents: Tensor,   # [1, 128, GH, GW] F32
    t_flow: Float32,         # flow time in [0,1] (1 = noise)
    llm_features: Tensor,    # [1, NT, 53248]
    loras: Ideogram4LoraSet,
    ctx: DeviceContext,
    text_len: Int = NT,      # natural caption length; pad rows get indicator 0
) raises -> Ideogram4LoRATrainForward:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG

    # packed transformer inputs (Ideogram4Predict.ideogram4_build_packed_inputs)
    var packed = ideogram4_build_packed_inputs[NT, GH, GW](
        noisy_latents, llm_features, ctx, text_len
    )

    # model_t = 1 - t_flow
    var tv = List[Float32]()
    tv.append(Float32(1.0) - t_flow)
    var model_t = Tensor.from_host(tv^, [1], STDtype.F32, ctx)

    # rope cos/sin from position_ids (interleaved MRoPE), bf16.
    var sec = List[Int]()
    sec.append(IDEOGRAM4_MROPE_SECTION_0)
    sec.append(IDEOGRAM4_MROPE_SECTION_1)
    sec.append(IDEOGRAM4_MROPE_SECTION_2)
    var cs = build_ideogram4_mrope(
        packed.position_ids, IDEOGRAM4_HEAD_DIM, sec, IDEOGRAM4_MROPE_THETA,
        ctx, STDtype.BF16,
    )
    var cosf = cs[0].clone(ctx)
    var sinf = cs[1].clone(ctx)

    # feed BF16 (transformer compute dtype)
    var x_bf = cast_tensor(packed.x, STDtype.BF16, ctx)
    var llm_bf = cast_tensor(packed.llm_full, STDtype.BF16, ctx)

    # embed (FROZEN) → x_in + adaln_input
    var emb = ideogram4_lora_embed(
        st, x_bf, llm_bf, model_t, packed.indicator, IDEOGRAM4_HIDDEN, ctx
    )

    # trainable block stack (LoRA) — replaces the dit 34-block loop
    var stack_fwd = ideogram4_stack_lora_forward[
        SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
        IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM,
    ](emb.x_in, emb.adaln_input, cosf, sinf, st, loras, ctx)

    var h = stack_fwd.out.clone(ctx)   # keep stack output for the ln backward

    # final (FROZEN) → out [1,SEQ,128] F32
    var fin = ideogram4_lora_final_forward(st, h, emb.adaln_input, ctx)

    # velocity = -( out[:,NT:].reshape(1,GH,GW,128).permute(0,3,1,2) )
    var image_velocity = slice(fin.out, 1, NT, NIMG, ctx)               # [1,NIMG,128]
    var iv4 = reshape(image_velocity, [1, GH, GW, IDEOGRAM4_PACKED_CHANNELS], ctx)
    var iv = permute(iv4, [0, 3, 1, 2], ctx)                           # [1,128,GH,GW]
    var velocity = mul_scalar(iv, Float32(-1.0), ctx)

    return Ideogram4LoRATrainForward(
        velocity^, stack_fwd^, h^, fin.fscale.clone(ctx), fin.flw.clone(ctx),
        emb.adaln_input.clone(ctx), cosf^, sinf^,
    )


def ideogram4_lora_train_forward_resident[
    NT: Int, GH: Int, GW: Int
](
    rw: Ideogram4Weights,
    noisy_latents: Tensor,
    t_flow: Float32,
    llm_features: Tensor,
    loras: Ideogram4LoraSet,
    ctx: DeviceContext,
    text_len: Int = NT,      # natural caption length; pad rows get indicator 0
) raises -> Ideogram4LoRATrainForward:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG

    var packed = ideogram4_build_packed_inputs[NT, GH, GW](
        noisy_latents, llm_features, ctx, text_len
    )

    var tv = List[Float32]()
    tv.append(Float32(1.0) - t_flow)
    var model_t = Tensor.from_host(tv^, [1], STDtype.F32, ctx)

    var sec = List[Int]()
    sec.append(IDEOGRAM4_MROPE_SECTION_0)
    sec.append(IDEOGRAM4_MROPE_SECTION_1)
    sec.append(IDEOGRAM4_MROPE_SECTION_2)
    var cs = build_ideogram4_mrope(
        packed.position_ids, IDEOGRAM4_HEAD_DIM, sec, IDEOGRAM4_MROPE_THETA,
        ctx, STDtype.BF16,
    )
    var cosf = cs[0].clone(ctx)
    var sinf = cs[1].clone(ctx)

    var x_bf = cast_tensor(packed.x, STDtype.BF16, ctx)
    var llm_bf = cast_tensor(packed.llm_full, STDtype.BF16, ctx)

    var emb = ideogram4_lora_embed_resident(
        rw, x_bf, llm_bf, model_t, packed.indicator, IDEOGRAM4_HIDDEN, ctx
    )

    var stack_fwd = ideogram4_stack_lora_forward_resident[
        SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
        IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM,
    ](emb.x_in, emb.adaln_input, cosf, sinf, rw, loras, ctx)

    var h = stack_fwd.out.clone(ctx)
    var fin = ideogram4_lora_final_forward_resident(rw, h, emb.adaln_input, ctx)

    var image_velocity = slice(fin.out, 1, NT, NIMG, ctx)
    var iv4 = reshape(image_velocity, [1, GH, GW, IDEOGRAM4_PACKED_CHANNELS], ctx)
    var iv = permute(iv4, [0, 3, 1, 2], ctx)
    var velocity = mul_scalar(iv, Float32(-1.0), ctx)

    return Ideogram4LoRATrainForward(
        velocity^, stack_fwd^, h^, fin.fscale.clone(ctx), fin.flw.clone(ctx),
        emb.adaln_input.clone(ctx), cosf^, sinf^,
    )


def ideogram4_lora_sample_position_ids[
    NT: Int, GH: Int, GW: Int
](
    ctx: DeviceContext,
    text_len: Int = NT,
) raises -> Tensor:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG

    var real_len = text_len
    if real_len > NT:
        real_len = NT
    if real_len < 0:
        real_len = 0

    var pos_host = List[Float32]()
    for i in range(NT):
        var p = i
        if i >= real_len:
            p = real_len - 1
        if p < 0:
            p = 0
        pos_host.append(Float32(p))
        pos_host.append(Float32(p))
        pos_host.append(Float32(p))
    for tok in range(NIMG):
        var h = tok // GW
        var w = tok % GW
        pos_host.append(Float32(IDEOGRAM4_IMAGE_OFFSET))
        pos_host.append(Float32(h + IDEOGRAM4_IMAGE_OFFSET))
        pos_host.append(Float32(w + IDEOGRAM4_IMAGE_OFFSET))
    return Tensor.from_host(pos_host^, [1, SEQ, 3], STDtype.F32, ctx)


def ideogram4_lora_sample_velocity_resident[
    NT: Int, GH: Int, GW: Int
](
    rw: Ideogram4Weights,
    noisy_latents: Tensor,
    t_flow: Float32,
    llm_features: Tensor,
    loras: Ideogram4LoraSet,
    ctx: DeviceContext,
    text_len: Int = NT,
) raises -> Tensor:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG

    var tv = List[Float32]()
    tv.append(Float32(1.0) - t_flow)
    var model_t = Tensor.from_host(tv^, [1], STDtype.F32, ctx)

    var sec = List[Int]()
    sec.append(IDEOGRAM4_MROPE_SECTION_0)
    sec.append(IDEOGRAM4_MROPE_SECTION_1)
    sec.append(IDEOGRAM4_MROPE_SECTION_2)
    var position_ids = ideogram4_lora_sample_position_ids[NT, GH, GW](
        ctx, text_len
    )
    var cs = build_ideogram4_mrope(
        position_ids, IDEOGRAM4_HEAD_DIM, sec, IDEOGRAM4_MROPE_THETA,
        ctx, STDtype.BF16,
    )
    var cosf = cs[0].clone(ctx)
    var sinf = cs[1].clone(ctx)

    var emb = ideogram4_lora_embed_resident_split[NT, GH, GW](
        rw, noisy_latents, llm_features, model_t, IDEOGRAM4_HIDDEN, ctx,
        text_len,
    )

    var h = ideogram4_stack_lora_forward_resident_nosave[
        SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
        IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM,
    ](emb.x_in, emb.adaln_input, cosf, sinf, rw, loras, ctx)

    var fin = ideogram4_lora_final_forward_resident(rw, h, emb.adaln_input, ctx)

    var image_velocity = slice(fin.out, 1, NT, NIMG, ctx)
    var iv4 = reshape(image_velocity, [1, GH, GW, IDEOGRAM4_PACKED_CHANNELS], ctx)
    var iv = permute(iv4, [0, 3, 1, 2], ctx)
    var velocity = mul_scalar(iv, Float32(-1.0), ctx)
    return velocity^


# ──────────────────────────────────────────────────────────────────────────────
# FINAL backward (FROZEN) — exact mirror of the forward velocity+final transform.
#   d_velocity [1,128,GH,GW] F32 →
#     un-negate ( ×-1 )                              → d_iv      [1,128,GH,GW]
#     un-permute [0,2,3,1] (inverse of [0,3,1,2])    → d_iv4     [1,GH,GW,128]
#     un-reshape                                     → d_img_tok [1,NIMG,128]
#     scatter (text rows 0)                          → d_out     [1,SEQ,128]
#     final_layer.linear^T (FROZEN: d_x only)        → d_hn      [1,SEQ,Hidden]
#     × fscale (frozen const)                        → d_lnout   [1,SEQ,Hidden]
#     layer_norm_no_affine bwd (weight = ones)       → d_h       [1,SEQ,Hidden]
# Returns d_h (= d_stack_out) bf16, fed to ideogram4_stack_lora_backward.
# ──────────────────────────────────────────────────────────────────────────────
def ideogram4_lora_final_backward[
    NT: Int, GH: Int, GW: Int
](
    d_velocity: Tensor,      # [1, 128, GH, GW] F32
    h: Tensor,               # [1, SEQ, Hidden] bf16 (stack output)
    fscale: Tensor,          # [1, 1, Hidden] bf16
    flw: Tensor,             # [128, Hidden] bf16 (frozen final linear weight)
    ctx: DeviceContext,
) raises -> Tensor:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG

    var d_iv = mul_scalar(d_velocity, Float32(-1.0), ctx)              # un-negate
    var d_iv4 = permute(d_iv, [0, 2, 3, 1], ctx)                       # [1,GH,GW,128]
    var d_img_tok = reshape(d_iv4, [1, NIMG, IDEOGRAM4_PACKED_CHANNELS], ctx)

    # scatter into d_out [1,SEQ,128]: text rows (0:NT) = 0, image rows = d_img_tok
    var text_zeros = zeros_device(
        [1, NT, IDEOGRAM4_PACKED_CHANNELS], STDtype.F32, ctx
    )
    var d_out = concat(1, ctx, text_zeros, d_img_tok)                  # [1,SEQ,128] F32

    # through final_layer.linear (FROZEN): only d_x (no weight grad)
    var d_hn = linear_backward_dx(
        d_out, flw, SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_PACKED_CHANNELS, ctx
    )                                                                 # [SEQ,Hidden] F32
    var d_hn3 = reshape(d_hn, [1, SEQ, IDEOGRAM4_HIDDEN], ctx)
    var d_hn_bf = cast_tensor(d_hn3, STDtype.BF16, ctx)

    # through fscale multiply (fscale frozen → no grad into it)
    var d_lnout = mul(d_hn_bf, fscale, ctx)                            # broadcast [1,1,H]

    # through layer_norm_no_affine (weight = ones → no-affine dx)
    var ones_w = add_scalar(
        zeros_device([IDEOGRAM4_HIDDEN], STDtype.BF16, ctx), Float32(1.0), ctx
    )
    var d_h = layer_norm_backward_dx(d_lnout, h, ones_w, I4_FINAL_EPS, ctx)
    return d_h^


# ──────────────────────────────────────────────────────────────────────────────
# T1.A LOSS LEVER SEAM — the ONE ideogram4 flow-loss site.
#   DEFAULT (levers_loss_active(lcfg) == False, i.e. loss_fn==MSE and
#   min_snr_gamma_flow==0): the literal pre-existing GPU MSE block
#   (sub → mul → reduce_mean_f32 → 2/N scale), byte-identical op chain (C13).
#   LEVER ACTIVE: velocity/target are brought to HOST and levers_loss_grad
#   dispatches huber/smooth_l1/min-SNR (torch-oracle-gated math in
#   ops/loss_fns.mojo), sigma := this step's t_flow (flow time, 1 = noise);
#   d_velocity is re-uploaded as the F32 backward seed. Correctness over
#   speed: one D2H pair + one H2D per lever-active step.
# NOTE: masked loss (T1.E) is NOT wired — the ideogram4 stager emits no masks;
# the trainer driver fails loud when lcfg.masked_training is set.
# ──────────────────────────────────────────────────────────────────────────────
struct _I4LossDvel(Movable):
    var loss: Float32
    var d_velocity: Tensor   # [1, 128, GH, GW] F32

    def __init__(out self, loss: Float32, var d_velocity: Tensor):
        self.loss = loss
        self.d_velocity = d_velocity^


def _i4_flow_loss_and_dvel(
    velocity: Tensor,    # [1, 128, GH, GW] F32
    target: Tensor,      # [1, 128, GH, GW] F32
    t_flow: Float32,     # this step's flow-match sigma
    n_elems: Int,        # velocity element count (128 * GH * GW)
    lcfg: LeversConfig,
    ctx: DeviceContext,
) raises -> _I4LossDvel:
    if levers_loss_active(lcfg):
        var pred_h = velocity.to_host(ctx)
        var tgt_h = target.to_host(ctx)
        var lg = levers_loss_grad(pred_h, tgt_h, t_flow, lcfg)
        var d_vel = Tensor.from_host(
            lg.d_pred, velocity.shape(), STDtype.F32, ctx
        )
        return _I4LossDvel(lg.loss, d_vel^)
    # literal legacy block (the trainers' inline MSE — default path):
    # loss = mean((velocity - target)^2) ; d_velocity = (2/N)*(velocity - target)
    var diff = sub(velocity, target, ctx)
    var sq = mul(diff, diff, ctx)
    var dims = List[Int]()
    dims.append(0); dims.append(1); dims.append(2); dims.append(3)
    var loss_t = reduce_mean_f32(sq, dims, False, ctx)
    var loss_h = loss_t.to_host(ctx)
    var d_velocity = mul_scalar(diff, Float32(2.0) / Float32(n_elems), ctx)
    return _I4LossDvel(loss_h[0], d_velocity^)


# compute-only entry: loss (via loss_out) + the per-adapter grads, NO
# optimizer applied. The driver chooses the optimizer (literal fused AdamW
# default vs the T1.C levers host path) — see
# Ideogram4LoRATrainer.train_ideogram4_lora_from_cache.
def ideogram4_lora_train_compute_resident[
    NT: Int, GH: Int, GW: Int
](
    rw: Ideogram4Weights,
    noisy_latents: Tensor,
    clean: Tensor,
    noise: Tensor,
    t_flow: Float32,
    llm_features: Tensor,
    loras: Ideogram4LoraSet,
    lcfg: LeversConfig,
    mut loss_out: Float32,
    ctx: DeviceContext,
    text_len: Int = NT,      # natural caption length; pad rows get indicator 0
) raises -> Ideogram4StackLoraGrads:
    """forward → loss (T1.A lever seam) → final backward → stack backward.
    Identical math to ideogram4_lora_train_step_resident minus the AdamW
    apply; with lcfg at LeversConfig.default() the loss path is the literal
    legacy MSE block (C13)."""
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG
    comptime N = IDEOGRAM4_PACKED_CHANNELS * NIMG

    var fwd = ideogram4_lora_train_forward_resident[NT, GH, GW](
        rw, noisy_latents, t_flow, llm_features, loras, ctx, text_len
    )

    var target = ideogram4_flow_target(noise, clean, ctx)
    var ld = _i4_flow_loss_and_dvel(fwd.velocity, target, t_flow, N, lcfg, ctx)
    loss_out = ld.loss

    var d_h = ideogram4_lora_final_backward[NT, GH, GW](
        ld.d_velocity, fwd.h, fwd.fscale, fwd.flw, ctx
    )

    comptime if IDEOGRAM4_V2_GRAPH_PATH:
        # P7: per-block graph-engine backward (resident weights stream); same
        # conductor loop + arg list as the hand-chain. Bit gate:
        # ideogram4_block_parity.
        var grads = ideogram4_stack_lora_backward_graph_resident[
            SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
            IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM,
        ](d_h, fwd.adaln_input, fwd.cosf, fwd.sinf, rw, loras, fwd.stack_fwd^, ctx)
        return grads^
    else:
        var grads = ideogram4_stack_lora_backward_resident[
            SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
            IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM,
        ](d_h, fwd.adaln_input, fwd.cosf, fwd.sinf, rw, loras, fwd.stack_fwd^, ctx)
        return grads^


# LyCORIS/carrier variant: forces the HAND-CHAIN stack backward regardless of
# IDEOGRAM4_V2_GRAPH_PATH. The v2 graph node (ops_record.record_ideogram4_block)
# records a SINGLE per-block rank (loras[0].rank) and applies it to all 6 slots —
# fine for the plain LoRA set (uniform rank) but WRONG for the LyCORIS carrier
# set, whose slots carry DIFFERENT ranks (active r_eff vs inactive rank-1). The
# hand-chain backward reads ad.rank per slot (block.mojo _lora_linear_bwd), so it
# is the correct path for a mixed-rank carrier set. Forward is the same plain
# per-slot forward the graph path also uses.
def ideogram4_lora_train_compute_resident_handchain[
    NT: Int, GH: Int, GW: Int
](
    rw: Ideogram4Weights,
    noisy_latents: Tensor,
    clean: Tensor,
    noise: Tensor,
    t_flow: Float32,
    llm_features: Tensor,
    loras: Ideogram4LoraSet,
    lcfg: LeversConfig,
    mut loss_out: Float32,
    ctx: DeviceContext,
    text_len: Int = NT,
) raises -> Ideogram4StackLoraGrads:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG
    comptime N = IDEOGRAM4_PACKED_CHANNELS * NIMG

    var fwd = ideogram4_lora_train_forward_resident[NT, GH, GW](
        rw, noisy_latents, t_flow, llm_features, loras, ctx, text_len
    )
    var target = ideogram4_flow_target(noise, clean, ctx)
    var ld = _i4_flow_loss_and_dvel(fwd.velocity, target, t_flow, N, lcfg, ctx)
    loss_out = ld.loss
    var d_h = ideogram4_lora_final_backward[NT, GH, GW](
        ld.d_velocity, fwd.h, fwd.fscale, fwd.flw, ctx
    )
    var grads = ideogram4_stack_lora_backward_resident[
        SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
        IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM,
    ](d_h, fwd.adaln_input, fwd.cosf, fwd.sinf, rw, loras, fwd.stack_fwd^, ctx)
    return grads^


# ══════════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 (row-stacked) compute — DEVICE-GRAD (mandatory for ideogram4: both
# arms feed the device optimizer). krea2's b2 is the reference. The FROZEN embed +
# final layers run PER-SAMPLE (each sample's per-sample text_len pad is baked into
# its own x_in exactly like krea2's per-sample conditioning); ONLY the 34-block
# trainable stack is batched ([2,SEQ,Hidden]), so the block GEMMs run over 2*SEQ
# rows and every LoRA dA/dB SUMS both samples = the batch gradient. The returned
# Ideogram4StackLoraGrads is the SAME device-tensor grad set the b1 path returns,
# so apply_ideogram4_lora_grads streams it into the fused multitensor AdamW (or the
# 408-launch fallback) UNCHANGED — the batch grads land in the device optimizer via
# the exact same path as b1. Only the default MSE loss is supported (levers fenced).
#
# Joint 2N-mean loss: loss_B2 = mean(loss0, loss1); each sample's d_velocity is
# scaled by 0.5 so the summed grads == the MEAN of the two b1 grads == grad-accum=2.
# ══════════════════════════════════════════════════════════════════════════════
struct _I4EmbedRope(Movable):
    var x_in: Tensor          # [1, SEQ, Hidden] bf16
    var adaln_input: Tensor   # [1, 1, Adaln]    bf16
    var cosf: Tensor          # [1, SEQ, Dh]     bf16
    var sinf: Tensor          # [1, SEQ, Dh]     bf16

    def __init__(
        out self, var x_in: Tensor, var adaln_input: Tensor,
        var cosf: Tensor, var sinf: Tensor,
    ):
        self.x_in = x_in^
        self.adaln_input = adaln_input^
        self.cosf = cosf^
        self.sinf = sinf^


def _ideogram4_embed_rope_resident[
    NT: Int, GH: Int, GW: Int
](
    rw: Ideogram4Weights,
    noisy_latents: Tensor,
    t_flow: Float32,
    llm_features: Tensor,
    ctx: DeviceContext,
    text_len: Int = NT,
) raises -> _I4EmbedRope:
    """Per-sample FROZEN embed + MRoPE tables (the front half of
    ideogram4_lora_train_forward_resident, up to but not including the block
    stack). Each sample owns its text_len (pad zeroing) and its rope table."""
    var packed = ideogram4_build_packed_inputs[NT, GH, GW](
        noisy_latents, llm_features, ctx, text_len
    )
    var tv = List[Float32]()
    tv.append(Float32(1.0) - t_flow)
    var model_t = Tensor.from_host(tv^, [1], STDtype.F32, ctx)

    var sec = List[Int]()
    sec.append(IDEOGRAM4_MROPE_SECTION_0)
    sec.append(IDEOGRAM4_MROPE_SECTION_1)
    sec.append(IDEOGRAM4_MROPE_SECTION_2)
    var cs = build_ideogram4_mrope(
        packed.position_ids, IDEOGRAM4_HEAD_DIM, sec, IDEOGRAM4_MROPE_THETA,
        ctx, STDtype.BF16,
    )
    var cosf = cs[0].clone(ctx)
    var sinf = cs[1].clone(ctx)

    var x_bf = cast_tensor(packed.x, STDtype.BF16, ctx)
    var llm_bf = cast_tensor(packed.llm_full, STDtype.BF16, ctx)
    var emb = ideogram4_lora_embed_resident(
        rw, x_bf, llm_bf, model_t, packed.indicator, IDEOGRAM4_HIDDEN, ctx
    )
    # clone (not partial-move) so `emb` stays whole for its synthesized destructor.
    return _I4EmbedRope(emb.x_in.clone(ctx), emb.adaln_input.clone(ctx), cosf^, sinf^)


def _ideogram4_velocity_from_final[
    NT: Int, GH: Int, GW: Int
](final_out: Tensor, ctx: DeviceContext) raises -> Tensor:
    """velocity = -( out[:, NT:].reshape(1,GH,GW,128).permute(0,3,1,2) ) — the same
    transform ideogram4_lora_train_forward_resident applies, per sample."""
    comptime NIMG = GH * GW
    var image_velocity = slice(final_out, 1, NT, NIMG, ctx)
    var iv4 = reshape(image_velocity, [1, GH, GW, IDEOGRAM4_PACKED_CHANNELS], ctx)
    var iv = permute(iv4, [0, 3, 1, 2], ctx)
    return mul_scalar(iv, Float32(-1.0), ctx)


def ideogram4_lora_train_compute_resident_b2[
    NT: Int, GH: Int, GW: Int
](
    rw: Ideogram4Weights,
    noisy0: Tensor, clean0: Tensor, noise0: Tensor, t0: Float32,
    llm0: Tensor, text_len0: Int,
    noisy1: Tensor, clean1: Tensor, noise1: Tensor, t1: Float32,
    llm1: Tensor, text_len1: Int,
    loras: Ideogram4LoraSet,
    lcfg: LeversConfig,
    mut loss_out: Float32,
    mut loss0_out: Float32,
    mut loss1_out: Float32,
    ctx: DeviceContext,
) raises -> Ideogram4StackLoraGrads:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG
    comptime N = IDEOGRAM4_PACKED_CHANNELS * NIMG

    if levers_loss_active(lcfg):
        raise Error(
            "ideogram4 batch_size=2 supports only the default MSE loss this wave"
            " (loss levers — Huber/smooth-L1/min-SNR — not wired for b2)"
        )

    # ── per-sample FROZEN embed + rope (each its own text_len/pad + rope table) ──
    var er0 = _ideogram4_embed_rope_resident[NT, GH, GW](
        rw, noisy0, t0, llm0, ctx, text_len0
    )
    var er1 = _ideogram4_embed_rope_resident[NT, GH, GW](
        rw, noisy1, t1, llm1, ctx, text_len1
    )

    # ── row-stack the batched trainable stack inputs ────────────────────────────
    var x_in_b2 = concat(0, ctx, er0.x_in, er1.x_in)              # [2,SEQ,Hidden]
    var adaln_b2 = concat(0, ctx, er0.adaln_input, er1.adaln_input)  # [2,1,Adaln]
    var cos_b2 = concat(1, ctx, er0.cosf, er1.cosf)              # [1,2SEQ,Dh]
    var sin_b2 = concat(1, ctx, er0.sinf, er1.sinf)

    var stack_fwd = ideogram4_stack_lora_forward_resident_b2[
        SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
        IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM,
    ](x_in_b2, adaln_b2, cos_b2, sin_b2, rw, loras, ctx)
    var h_b2 = stack_fwd.out.clone(ctx)                          # [2,SEQ,Hidden]

    # ── per-sample FROZEN final → velocity → loss → final backward ──────────────
    var h0 = slice(h_b2, 0, 0, 1, ctx)                          # [1,SEQ,Hidden]
    var h1 = slice(h_b2, 0, 1, 1, ctx)
    var fin0 = ideogram4_lora_final_forward_resident(rw, h0, er0.adaln_input, ctx)
    var fin1 = ideogram4_lora_final_forward_resident(rw, h1, er1.adaln_input, ctx)
    var vel0 = _ideogram4_velocity_from_final[NT, GH, GW](fin0.out, ctx)
    var vel1 = _ideogram4_velocity_from_final[NT, GH, GW](fin1.out, ctx)
    var tgt0 = ideogram4_flow_target(noise0, clean0, ctx)
    var tgt1 = ideogram4_flow_target(noise1, clean1, ctx)
    var ld0 = _i4_flow_loss_and_dvel(vel0, tgt0, t0, N, lcfg, ctx)
    var ld1 = _i4_flow_loss_and_dvel(vel1, tgt1, t1, N, lcfg, ctx)
    loss0_out = ld0.loss
    loss1_out = ld1.loss
    loss_out = Float32(0.5) * (ld0.loss + ld1.loss)             # joint 2N-mean

    # each per-sample d_velocity ×0.5: summed grads == MEAN(g0,g1) == grad-accum=2.
    var d_vel0 = mul_scalar(ld0.d_velocity, Float32(0.5), ctx)
    var d_vel1 = mul_scalar(ld1.d_velocity, Float32(0.5), ctx)
    var d_h0 = ideogram4_lora_final_backward[NT, GH, GW](
        d_vel0, h0, fin0.fscale, fin0.flw, ctx
    )
    var d_h1 = ideogram4_lora_final_backward[NT, GH, GW](
        d_vel1, h1, fin1.fscale, fin1.flw, ctx
    )
    var d_h_b2 = concat(0, ctx, d_h0, d_h1)                     # [2,SEQ,Hidden]

    var grads = ideogram4_stack_lora_backward_resident_b2[
        SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
        IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM,
    ](d_h_b2, adaln_b2, cos_b2, sin_b2, rw, loras, stack_fwd^, ctx)
    return grads^


# ──────────────────────────────────────────────────────────────────────────────
# FULL TRAINING STEP.
#   forward → velocity ; target = noise - clean ; loss = mean((velocity-target)^2)
#   d_velocity = (2/N)*(velocity-target) ; final-backward → d_stack_out
#   stack_lora_backward → grads ; apply_ideogram4_lora_grads → AdamW.
# Returns loss + B-L1 + did_update.
# ──────────────────────────────────────────────────────────────────────────────
def ideogram4_lora_train_step[
    NT: Int, GH: Int, GW: Int
](
    st: ShardedSafeTensors,
    noisy_latents: Tensor,   # [1, 128, GH, GW] F32
    clean: Tensor,           # [1, 128, GH, GW] F32
    noise: Tensor,           # [1, 128, GH, GW] F32
    t_flow: Float32,         # flow time in [0,1]
    llm_features: Tensor,    # [1, NT, 53248]
    loras: Ideogram4LoraSet,
    mut opt_state: Ideogram4LoraAdamState,
    optimizer_step: Int,
    cfg: TrainConfig,
    ctx: DeviceContext,
    text_len: Int = NT,      # natural caption length; pad rows get indicator 0
) raises -> Ideogram4LoRATrainResult:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG
    comptime N = IDEOGRAM4_PACKED_CHANNELS * NIMG   # velocity element count

    # forward → velocity (+ cached tensors)
    var fwd = ideogram4_lora_train_forward[NT, GH, GW](
        st, noisy_latents, t_flow, llm_features, loras, ctx, text_len
    )

    # flow target = noise - clean (Ideogram4Predict.ideogram4_flow_target)
    var target = ideogram4_flow_target(noise, clean, ctx)             # [1,128,GH,GW] F32

    # loss = mean((velocity - target)^2) ; d_velocity = (2/N)*(velocity - target)
    # (the shared loss seam at LeversConfig.default() == the literal MSE block;
    # this non-resident path is gate-only — the production resident path takes
    # the live lcfg via ideogram4_lora_train_compute_resident)
    var ld = _i4_flow_loss_and_dvel(
        fwd.velocity, target, t_flow, N, LeversConfig.default(), ctx
    )
    var loss = ld.loss

    # final-backward (FROZEN) → d_stack_out
    var d_h = ideogram4_lora_final_backward[NT, GH, GW](
        ld.d_velocity, fwd.h, fwd.fscale, fwd.flw, ctx
    )

    # stack LoRA backward → per-adapter grads
    # AdamW update (REUSED — never hand-rolled)
    comptime if IDEOGRAM4_V2_GRAPH_PATH:
        # P7: per-block graph-engine backward; same conductor loop + arg list as
        # the hand-chain. Bit gate: ideogram4_block_parity.
        var grads = ideogram4_stack_lora_backward_graph[
            SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
            IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM,
        ](d_h, fwd.adaln_input, fwd.cosf, fwd.sinf, st, loras, fwd.stack_fwd^, ctx)
        var res = apply_ideogram4_lora_grads(
            loras, opt_state, grads^, optimizer_step, cfg, ctx
        )
        return Ideogram4LoRATrainResult(loss, res.adapter_b_l1, True)
    else:
        var grads = ideogram4_stack_lora_backward[
            SEQ, IDEOGRAM4_HIDDEN, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
            IDEOGRAM4_INTERMEDIATE_SIZE, IDEOGRAM4_ADALN_DIM,
        ](d_h, fwd.adaln_input, fwd.cosf, fwd.sinf, st, loras, fwd.stack_fwd^, ctx)
        var res = apply_ideogram4_lora_grads(
            loras, opt_state, grads^, optimizer_step, cfg, ctx
        )
        return Ideogram4LoRATrainResult(loss, res.adapter_b_l1, True)


def ideogram4_lora_train_step_resident[
    NT: Int, GH: Int, GW: Int
](
    rw: Ideogram4Weights,
    noisy_latents: Tensor,
    clean: Tensor,
    noise: Tensor,
    t_flow: Float32,
    llm_features: Tensor,
    loras: Ideogram4LoraSet,
    mut opt_state: Ideogram4LoraAdamState,
    optimizer_step: Int,
    cfg: TrainConfig,
    ctx: DeviceContext,
    text_len: Int = NT,      # natural caption length; pad rows get indicator 0
) raises -> Ideogram4LoRATrainResult:
    # compute (default levers config == literal legacy loss path) + AdamW apply.
    var loss = Float32(0.0)
    var grads = ideogram4_lora_train_compute_resident[NT, GH, GW](
        rw, noisy_latents, clean, noise, t_flow, llm_features, loras,
        LeversConfig.default(), loss, ctx, text_len,
    )
    var res = apply_ideogram4_lora_grads(
        loras, opt_state, grads^, optimizer_step, cfg, ctx
    )
    return Ideogram4LoRATrainResult(loss, res.adapter_b_l1, True)
