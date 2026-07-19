# Ideogram4SampleLoop.mojo — Ideogram-4 INFERENCE denoise driver (the loop the
# Flux2Sampler PORT SPEC calls "PENDING", assembled here for Ideogram-4).
#
# Built on the Flux2Sampler.mojo template (SampleOutput + DenoiseState + per-step
# helpers + F32-persistent-latent dtype discipline), but ASSEMBLED into a complete
# driver because the numeric core is already proven: this borrows serenitymojo's
# gated ideogram4_forward / build_ideogram4_mrope / VAE decoder (the working 1:1
# port of pipeline_ideogram4, final_z/final_latent/decoded all gated vs torch).
#
# ── PORT SPEC (1:1) ───────────────────────────────────────────────────────────
# ai-toolkit ideogram4/src/pipeline.py denoise + decode (mirrored by the torch
# oracle ideogram4_oracle.py::stage_E and serenitymojo/pipeline/ideogram4_pipeline.mojo):
#   schedule: mean = known_mean + 0.5*log(HW / 512^2)   (scheduler.py get_schedule)
#             si   = make_step_intervals(steps)
#             sigma(i) = logitnormal(si[i], mean)        (scheduler.py)
#   loop i = steps-1 .. 0  (under no_grad — the INFERENCE forward, no tape):
#     t = sigma(i+1) ;  s = sigma(i)
#     pos_v = cond_transformer([text_zpad ++ z], llm_full)[:, nt:]   # text+image tokens
#     neg_v = uncond_transformer(z, neg_llm)                         # image tokens only
#     v = cfg*pos_v + (1-cfg)*neg_v                                  # asymmetric CFG
#     z = z + v*(s - t)                                              # Euler
#   decode: z = z*scale + shift ; unpatchify [1,gh,gw,2,2,32]->[1,32,2gh,2gw] ;
#           vae.decode -> image [1,3,H,W].
#
# DTYPE: the persistent Euler latent z stays F32 across all steps; it is cast to
# BF16 only to feed each transformer (matches Flux2Sampler's F32-latent rule and
# pipeline_ideogram4 latents.to(dtype)). The two DiTs are loaded fp8 -> dequant
# BF16 (the resident pattern). NB: NO leading minus on v (the transformer output
# IS the flow); the Euler update is z += v*(s-t) with s<t (sigmas descend).
#
# Two transformers: Ideogram-4 ships a SEPARATE unconditional transformer
# (asymmetric CFG); the cond forward runs over text+image tokens, the uncond over
# image tokens only (no text). This is NOT the Flux2 pos/neg-batch CFG.

from std.gpu.host import DeviceContext
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    add, mul, mul_scalar, reshape, permute, slice, concat,
)
from serenitymojo.image.png import save_png
from serenitymojo.models.dit.ideogram4_dit import ideogram4_forward
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder

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
    ideogram4_logitnormal,
    ideogram4_schedule_mean,
)
from serenitymojo.sampling.ideogram4_schedule import make_step_intervals


# ── sampler output: decoded image [1,3,H,W] ───────────────────────────────────
struct Ideogram4SampleOutput(Movable):
    var image: Tensor

    def __init__(out self, var image: Tensor):
        self.image = image^


# ── per-step helpers (Flux2Sampler-style seams) ───────────────────────────────
# Asymmetric CFG combine (pipeline_ideogram4): v = cfg*pos + (1-cfg)*neg.
def ideogram4_cfg_combine(
    pos_v: Tensor, neg_v: Tensor, guidance: Float32, ctx: DeviceContext
) raises -> Tensor:
    return add(
        mul_scalar(pos_v, guidance, ctx),
        mul_scalar(neg_v, Float32(1.0) - guidance, ctx),
        ctx,
    )


# One Euler step on the F32 latent: z += v*(s - t)  (sigmas descend so s<t).
def ideogram4_euler_step(
    z: Tensor, v: Tensor, s_val: Float32, t_val: Float32, ctx: DeviceContext
) raises -> Tensor:
    return add(z, mul_scalar(v, s_val - t_val, ctx), ctx)


# ── the denoise loop (cond + uncond DiTs, asymmetric CFG, logit-normal Euler) ──
# z0:        [1, NIMG, 128] F32 (initial noise tokens)
# llm_full:  [1, TOTAL, 53248] bf16 (text features over [text ++ image], image zero-padded)
# neg_llm:   [1, NIMG, 53248] bf16 (zeros — uncond has no text)
# pos/npos:  [1, TOTAL, 3] / [1, NIMG, 3] F32 position_ids
# ind/nind:  [1, TOTAL] / [1, NIMG] F32 indicators (0/2/3)
# text_zpad: [1, NT, 128] F32 zeros (image-latent slot for the text tokens)
# Returns the final F32 latent z [1, NIMG, 128].
def ideogram4_denoise[
    NT: Int, NIMG: Int, TOTAL: Int
](
    cond_st: ShardedSafeTensors,
    uncond_st: ShardedSafeTensors,
    var z: Tensor,
    llm_full: Tensor,
    neg_llm: Tensor,
    pos_ids: Tensor,
    neg_pos_ids: Tensor,
    ind: Tensor,
    nind: Tensor,
    text_zpad: Tensor,
    steps: Int,
    guidance: Float32,
    height: Int,
    width: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var sec = List[Int]()
    sec.append(IDEOGRAM4_MROPE_SECTION_0)
    sec.append(IDEOGRAM4_MROPE_SECTION_1)
    sec.append(IDEOGRAM4_MROPE_SECTION_2)
    var cs = build_ideogram4_mrope(
        pos_ids, IDEOGRAM4_HEAD_DIM, sec, IDEOGRAM4_MROPE_THETA, ctx, STDtype.BF16
    )
    var sec2 = List[Int]()
    sec2.append(IDEOGRAM4_MROPE_SECTION_0)
    sec2.append(IDEOGRAM4_MROPE_SECTION_1)
    sec2.append(IDEOGRAM4_MROPE_SECTION_2)
    var ncs = build_ideogram4_mrope(
        neg_pos_ids, IDEOGRAM4_HEAD_DIM, sec2, IDEOGRAM4_MROPE_THETA, ctx, STDtype.BF16
    )

    var mean = ideogram4_schedule_mean(height, width, 0.5)
    var si = make_step_intervals(steps)

    for step in range(steps - 1, -1, -1):
        var t_val = ideogram4_logitnormal(Float64(si[step + 1]), mean)
        var s_val = ideogram4_logitnormal(Float64(si[step]), mean)
        var tv = List[Float32]()
        tv.append(t_val)
        var t = Tensor.from_host(tv^, [1], STDtype.F32, ctx)

        # cond: [text_zpad ++ z] over TOTAL tokens, take the image slice.
        var pos_z = cast_tensor(concat(1, ctx, text_zpad, z), STDtype.BF16, ctx)
        var cout = ideogram4_forward[TOTAL](
            cond_st, pos_z, llm_full, t, ind, cs[0], cs[1],
            IDEOGRAM4_NUM_LAYERS, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM, IDEOGRAM4_HIDDEN, ctx,
        )
        var pos_v = slice(cout, 1, NT, NIMG, ctx)   # [1,NIMG,128] F32

        # uncond: z over NIMG image tokens only, zero text features.
        var z_bf = cast_tensor(z, STDtype.BF16, ctx)
        var tv2 = List[Float32]()
        tv2.append(t_val)
        var t2 = Tensor.from_host(tv2^, [1], STDtype.F32, ctx)
        var neg_v = ideogram4_forward[NIMG](
            uncond_st, z_bf, neg_llm, t2, nind, ncs[0], ncs[1],
            IDEOGRAM4_NUM_LAYERS, IDEOGRAM4_NUM_HEADS, IDEOGRAM4_HEAD_DIM, IDEOGRAM4_HIDDEN, ctx,
        )

        var v = ideogram4_cfg_combine(pos_v, neg_v, guidance, ctx)
        z = ideogram4_euler_step(z, v, s_val, t_val, ctx)
    return z^


# ── decode: final latent tokens -> image (denorm -> unpatchify -> VAE decode) ──
# z_final [1,NIMG,128] F32 ; latent_scale/latent_shift [128] ; GH=gw, GW=gw
# (packed grid; latent spatial = 2*GH x 2*GW). Returns image [1,3,2*GH*8/... ].
def ideogram4_decode[
    GH: Int, GW: Int
](
    z_final: Tensor,
    latent_scale: Tensor,
    latent_shift: Tensor,
    vae_path: String,
    ctx: DeviceContext,
) raises -> Ideogram4SampleOutput:
    var scale = reshape(latent_scale, [1, 1, IDEOGRAM4_PACKED_CHANNELS], ctx)
    var shift = reshape(latent_shift, [1, 1, IDEOGRAM4_PACKED_CHANNELS], ctx)
    var zd = add(mul(z_final, scale, ctx), shift, ctx)               # [1,NIMG,128] F32
    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)                    # [1,32,GH,2,GW,2]
    var latent = reshape(zp, [1, 32, 2 * GH, 2 * GW], ctx)          # [1,32,2GH,2GW]
    var dec = load_ideogram4_vae_decoder[2 * GH, 2 * GW](vae_path, ctx)
    var img = dec.decode(cast_tensor(latent, STDtype.BF16, ctx), ctx)
    return Ideogram4SampleOutput(img^)
