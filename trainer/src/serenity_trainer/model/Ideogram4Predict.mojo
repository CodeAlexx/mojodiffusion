# Ideogram4Predict.mojo — Ideogram-4 TRAINING predict path (1:1 port).
#
# Pure Mojo + MAX, INFERENCE-style forward (no autograd here — the LoRA backward
# is a separate later chunk). This mirrors the ai-toolkit training pipeline's
# velocity prediction so the trainer can compute a flow-matching loss against the
# transformer's output.
#
# ── PORT SPEC (1:1) ───────────────────────────────────────────────────────────
# ai-toolkit extensions_built_in/diffusion_models/ideogram4/src/pipeline.py
#   ::predict_velocity (lines 152-250) — packs [text ++ image] tokens, builds the
#   indicator / segment_ids / position_ids tensors, flips flow-time t -> model-time
#   (1 - t), runs the transformer, slices the image tokens back to a (B,128,gh,gw)
#   velocity, and NEGATES it (toolkit velocity convention noise->clean).
#   Plus the flow helpers:
#     add_noise:        noisy  = (1 - t) * clean + t * noise   (pipeline.py)
#     get_loss_target:  target = noise - clean                 (pipeline.py)
#
# BORROW BOUNDARY — the numeric core is already gated in serenitymojo (the working
# 1:1 inference port of pipeline_ideogram4). We CALL, never reimplement:
#   serenitymojo.models.dit.ideogram4_dit.ideogram4_forward[S]
#   serenitymojo.models.dit.ideogram4_mrope.build_ideogram4_mrope
#
# segment_ids NOTE: ai-toolkit passes segment_ids (text region = 1 where
# text_mask, else SEQUENCE_PADDING_INDICATOR=-1; image region = 1). For a SINGLE
# sample (b=1) with all-real text (text_mask all ones), segment_ids is all-1, and
# ideogram4_forward does not take segment_ids — position info enters only via the
# rope cos/sin built from position_ids. So segment_ids is a no-op here and is
# intentionally not constructed. (Document for the future b>1 / padded-text case.)
#
# DTYPE: noisy_latents / clean / noise persist F32 (Euler-latent rule). The packed
# x and llm_full are cast to BF16 only to feed ideogram4_forward; the transformer
# returns F32; the returned velocity is F32.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    add, sub, mul_scalar, reshape, permute, slice, concat, zeros_device,
)
from serenitymojo.models.dit.ideogram4_dit import ideogram4_forward
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope

from serenity_trainer.modelSampler.Ideogram4Sampler import (
    IDEOGRAM4_NUM_LAYERS,
    IDEOGRAM4_NUM_HEADS,
    IDEOGRAM4_HEAD_DIM,
    IDEOGRAM4_HIDDEN,
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


# ── packed transformer inputs (pipeline.py predict_velocity, the cat/build block)─
# Holds the four tensors ideogram4_forward consumes that depend on the packed
# [text ++ image] layout. SEQ = NT + GH*GW.
#   x            [1, SEQ, 128]    F32  — text region zeroed, image region = latents
#   llm_full     [1, SEQ, 53248]  (llm dtype) — image region zeroed
#   position_ids [1, SEQ, 3]      F32  — text rows [i,i,i]; image rows [off,h+off,w+off]
#   indicator    [1, SEQ]         F32  — text -> 3, image -> 2
struct Ideogram4PackedInputs(Movable):
    var x: Tensor
    var llm_full: Tensor
    var position_ids: Tensor
    var indicator: Tensor

    def __init__(
        out self,
        var x: Tensor,
        var llm_full: Tensor,
        var position_ids: Tensor,
        var indicator: Tensor,
    ):
        self.x = x^
        self.llm_full = llm_full^
        self.position_ids = position_ids^
        self.indicator = indicator^


# ──────────────────────────────────────────────────────────────────────────────
# (1) build_packed_inputs — pipeline.py predict_velocity lines 161-188.
#   image_tokens = latents.permute(0,2,3,1).reshape(b, num_image_tokens, c)
#   x            = cat([zeros(b, nt, c), image_tokens], dim=1)
#   llm_full     = cat([llm_features, zeros(b, nimg, 53248)], dim=1)
#   indicator    : text -> text_mask*LLM_TOKEN_INDICATOR(=3), image -> OUTPUT_IMAGE_INDICATOR(=2)
#   position_ids : text rows = (cumsum(text_mask)-1).clamp(0) expanded to 3
#                  (all-ones mask -> 0..NT-1), image rows = [t=0,h,w]+IMAGE_OFFSET,
#                  h outer (tok//GW), w inner (tok%GW).
#
# text_len: the NATURAL (pre-pad) token count of THIS sample's caption. When a
# caption is shorter than the fixed NT bucket, positions [text_len, NT) are
# right-padding. ai-toolkit pipeline.py:249 sets the indicator there to 0
# (indicator[:, :nt] = text_mask_long * LLM_TOKEN_INDICATOR; pad -> 0), and the
# encoder already zeroed those FEATURE rows (pipeline.py:156-157 stacked*text_mask
# = our prepare-time zeroing). Default text_len = NT (all real, no padding) keeps
# every existing all-real-text caller / the parity fixture byte-identical.
# (b=1 single training sample.)
# ──────────────────────────────────────────────────────────────────────────────
def ideogram4_build_packed_inputs[
    NT: Int, GH: Int, GW: Int
](
    noisy_latents: Tensor,   # [1, 128, GH, GW] F32
    llm_features: Tensor,    # [1, NT, 53248]  (bf16 / f32)
    ctx: DeviceContext,
    text_len: Int = NT,      # natural token count; [text_len, NT) are pad rows
) raises -> Ideogram4PackedInputs:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG

    # Clamp defensively: a caption can fill or exceed the bucket (then no pad).
    var real_len = text_len
    if real_len > NT:
        real_len = NT
    if real_len < 0:
        real_len = 0

    # image_tokens = latents.permute(0,2,3,1).reshape(1, NIMG, 128)  (h outer, w inner)
    var img_perm = permute(noisy_latents, [0, 2, 3, 1], ctx)         # [1,GH,GW,128]
    var image_tokens = reshape(img_perm, [1, NIMG, IDEOGRAM4_PACKED_CHANNELS], ctx)

    # x = cat([zeros(1,NT,128), image_tokens], dim=1)  (text region zeroed)
    var text_zeros = zeros_device(
        [1, NT, IDEOGRAM4_PACKED_CHANNELS], STDtype.F32, ctx
    )
    var x = concat(1, ctx, text_zeros, image_tokens)                 # [1,SEQ,128] F32

    # llm_full = cat([llm_features, zeros(1,NIMG,53248)], dim=1)  (image region zeroed)
    var llm_zeros = zeros_device(
        [1, NIMG, IDEOGRAM4_TEXT_FEATURE_DIM], llm_features.dtype(), ctx
    )
    var llm_full = concat(1, ctx, llm_features, llm_zeros)           # [1,SEQ,53248]

    # position_ids: host-built F32 [1,SEQ,3].
    # ai-toolkit pipeline.py:262 text_pos = (text_mask.cumsum(-1)-1).clamp(min=0):
    #   real position i (i < real_len) -> i ; pad position (i >= real_len) holds
    #   at the last real index (real_len-1, clamped to >=0). All-real (real_len==NT)
    #   reproduces the old 0..NT-1 ramp.
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
        var h = tok // GW                                            # h outer
        var w = tok % GW                                             # w inner
        # image row [t=0, h, w] + IMAGE_POSITION_OFFSET (added to all 3)
        pos_host.append(Float32(IDEOGRAM4_IMAGE_OFFSET))
        pos_host.append(Float32(h + IDEOGRAM4_IMAGE_OFFSET))
        pos_host.append(Float32(w + IDEOGRAM4_IMAGE_OFFSET))
    var position_ids = Tensor.from_host(pos_host^, [1, SEQ, 3], STDtype.F32, ctx)

    # indicator: host-built F32 [1,SEQ]; text -> 3 (mask*LLM_TOKEN_INDICATOR),
    # text-pad -> 0 (ai-toolkit pipeline.py:249: text_mask_long * 3), image -> 2.
    var ind_host = List[Float32]()
    for i in range(NT):
        if i < real_len:
            ind_host.append(Float32(IDEOGRAM4_LLM_TOKEN_INDICATOR))
        else:
            ind_host.append(Float32(0.0))
    for _ in range(NIMG):
        ind_host.append(Float32(IDEOGRAM4_OUTPUT_IMAGE_INDICATOR))
    var indicator = Tensor.from_host(ind_host^, [1, SEQ], STDtype.F32, ctx)

    return Ideogram4PackedInputs(x^, llm_full^, position_ids^, indicator^)


# ──────────────────────────────────────────────────────────────────────────────
# (2) predict_velocity — pipeline.py predict_velocity lines 152-250 (full).
#   model_t = 1 - t                                          (flow-time -> model-time)
#   out     = transformer(llm_full, x, model_t, position_ids, indicator)
#   image_velocity = out[:, NT:].reshape(b,gh,gw,c).permute(0,3,1,2)
#   return -image_velocity
# t_flow is the flow time in [0,1] (1 = noise); the rope cos/sin carry the position
# info (segment_ids is a b=1 no-op, see header).
# ──────────────────────────────────────────────────────────────────────────────
def ideogram4_predict_velocity[
    NT: Int, GH: Int, GW: Int
](
    st: ShardedSafeTensors,
    noisy_latents: Tensor,   # [1, 128, GH, GW] F32
    t_flow: Float32,         # flow time in [0,1] (1 = noise)
    llm_features: Tensor,    # [1, NT, 53248]
    ctx: DeviceContext,
) raises -> Tensor:          # [1, 128, GH, GW] F32 velocity
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG

    var packed = ideogram4_build_packed_inputs[NT, GH, GW](
        noisy_latents, llm_features, ctx
    )

    # model_t = 1 - t  (pipeline.py: model_t = 1.0 - t)
    var tv = List[Float32]()
    tv.append(Float32(1.0) - t_flow)
    var model_t = Tensor.from_host(tv^, [1], STDtype.F32, ctx)

    # rope cos/sin from position_ids (build_ideogram4_mrope, interleaved MRoPE).
    var sec = List[Int]()
    sec.append(IDEOGRAM4_MROPE_SECTION_0)
    sec.append(IDEOGRAM4_MROPE_SECTION_1)
    sec.append(IDEOGRAM4_MROPE_SECTION_2)
    var cs = build_ideogram4_mrope(
        packed.position_ids, IDEOGRAM4_HEAD_DIM, sec, IDEOGRAM4_MROPE_THETA,
        ctx, STDtype.BF16,
    )

    # feed BF16 (the transformer's compute dtype).
    var x_bf = cast_tensor(packed.x, STDtype.BF16, ctx)
    var llm_bf = cast_tensor(packed.llm_full, STDtype.BF16, ctx)

    var out = ideogram4_forward[SEQ](
        st, x_bf, llm_bf, model_t, packed.indicator, cs[0], cs[1],
        IDEOGRAM4_NUM_LAYERS, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM,
        IDEOGRAM4_HIDDEN, ctx,
    )                                                               # [1,SEQ,128] F32

    # image_velocity = out[:, NT:].reshape(b,gh,gw,c).permute(0,3,1,2)
    var image_velocity = slice(out, 1, NT, NIMG, ctx)               # [1,NIMG,128]
    var iv4 = reshape(
        image_velocity, [1, GH, GW, IDEOGRAM4_PACKED_CHANNELS], ctx
    )                                                               # [1,GH,GW,128]
    var iv = permute(iv4, [0, 3, 1, 2], ctx)                        # [1,128,GH,GW]

    # return -image_velocity  (negate -> toolkit velocity, noise->clean)
    return mul_scalar(iv, Float32(-1.0), ctx)


# ──────────────────────────────────────────────────────────────────────────────
# (3) add_noise — pipeline.py: noisy = (1 - t) * clean + t * noise.
# ──────────────────────────────────────────────────────────────────────────────
def ideogram4_add_noise[
    GH: Int, GW: Int
](
    clean: Tensor,           # [1,128,GH,GW] F32
    noise: Tensor,           # [1,128,GH,GW] F32
    t_flow: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    return add(
        mul_scalar(clean, Float32(1.0) - t_flow, ctx),
        mul_scalar(noise, t_flow, ctx),
        ctx,
    )


# ──────────────────────────────────────────────────────────────────────────────
# (4) get_loss_target — pipeline.py: target = noise - clean.
# ──────────────────────────────────────────────────────────────────────────────
def ideogram4_flow_target(
    noise: Tensor,           # [1,128,GH,GW] F32
    clean: Tensor,           # [1,128,GH,GW] F32
    ctx: DeviceContext,
) raises -> Tensor:
    return sub(noise, clean, ctx)
