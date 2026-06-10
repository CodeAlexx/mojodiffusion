# LTX-2.3 T2V + AUDIO HQ capstone — HIGH-QUALITY recipe upgrade of the P7 MVP.
#
# This is a copy of pipeline/ltx2_t2v_av_mvp.mojo upgraded to the PROVEN HQ
# recipe from the desktop app (archive_pre_lightricks_20260411/{pipeline,config}.py
# + HANDOFF_HQ_SAMPLER_DEBUG.md). The HANDOFF proved on real hardware:
#   "Distilled FP8 + res2s + LoRA conditioning -> GOOD output"
# i.e. the DISTILLED model (the fp8 one we already stream) + the second-order
# Runge-Kutta res_2s sampler + the runtime LoRA path is the proven-sharp path.
# The dev model -> garbage (HANDOFF bugs 2/3), so we stay on distilled.
#
# UPGRADES vs the MVP (each a quality lever the soft MVP was missing):
#   1. RESOLUTION 768x512 (was 256x256): NH=16, NW=24 -> S_V=NF*16*24.
#   2. FULL 1024-token text/audio context (was a 128-token tail slice): N_TXT=1024,
#      using the entire feature_extract_and_project dump.
#   3. res_2s SECOND-ORDER sampler (was Euler): 15-step LTX2Scheduler with the
#      token-count-dependent sigma shift (ltx-core LTX2Scheduler.execute), TWO
#      model evals per step (current sigma + geometric-mean midpoint sub_sigma).
#      The model is a velocity predictor; res_2s wants a DENOISER, so per eval we
#      convert  denoised = x - v*sigma  (rectified-flow x0 estimate).
#   4. AudioSync HQ LoRA stack applied through the P6 LoraSet RUNTIME ADD
#      (HARD RULE: never fused): distilled support + camera static + IC detailer.
#
# CFG NOTE: in the reference, DISTILLED mode runs `simple_denoising_func` — a
# single un-guided forward (guidance_scale=1, no negative context). True CFG-star
# is the DEV path (`multi_modal_guider_denoising_func`, needs a negative-prompt
# Gemma encode we do not have a dump for). We therefore run the proven distilled
# simple-denoise per eval. The cfg_star combine in ltx2_guidance.mojo is wired and
# available, but with only the positive context the honest, proven recipe is the
# un-guided distilled forward + res_2s + HQ LoRA stack — the sampler/res upgrades are
# the dominant sharpness levers the HANDOFF measured.
#
# Run (GPU; FP8 streaming keeps DiT bounded):
#   pixi run mojo run -I . serenitymojo/pipeline/ltx2_t2v_av_hq.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt, cos as fcos, sin as fsin, pow as fpow, log as flog, exp as fexp, pi
from std.memory import alloc, ArcPointer
from std.sys import argv
from std.time import perf_counter

from serenitymojo.io.ffi import (
    sys_open, sys_pwrite, sys_close, sys_system,
    O_WRONLY, O_CREAT, O_TRUNC, BytePtr,
)
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.activations import silu
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    reshape, add, sub, mul, div, mul_scalar, add_scalar, slice, permute,
)
from serenitymojo.ops.reduce import reduce_mean_f32, reduce_std_f32
from serenitymojo.models.dit.ltx2_dit import (
    LTX2Config, LTX2AVBlockWeights, ltx2_block_forward_av,
    ltx2_block_forward_av_nag,
)
from serenitymojo.models.dit.ltx2_nag import NAGContext
from serenitymojo.models.dit.ltx2_connector import (
    LTX2ConnectorConfig, LTX2ConnectorWeights, ltx2_connector_forward,
)
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream
from serenitymojo.sampling.ltx2_sampling import (
    LTX2Scheduler, res2s_coefficients, res2s_substep, res2s_combine, Res2sCoeffs,
    res2s_sde_step, res2s_bong_refine, res2s_bong_active,
    ltx2_stage2_distilled_sigmas,
)
from serenitymojo.sampling.ltx2_res2s_ref import (
    NoiseSource, res2s_ref_loop,
)
from serenitymojo.sampling.ltx2_multimodal_guider import guider_calculate
from serenitymojo.models.vae.ltx2_vae_decoder import (
    LTX2VaeDecoderWeights, decode as decode_video,
)
from serenitymojo.models.vae.ltx2_audio_vae import (
    LTX2AudioVaeDecoderWeights, decode as decode_audio,
)
from serenitymojo.models.vocoder.ltx2_vocoder import LTX2VocoderWithBWE
from serenitymojo.models.upsampler.ltx2_upsampler import (
    LatentUpsampler, upsample_video,
)
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.lora import LoraSet, FMT_LTX2_DISTILLED, LTX2BlockLoraDeltaSet


# ── paths ─────────────────────────────────────────────────────────────────────
comptime CKPT_FP8 = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
comptime CKPT_BF16 = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
comptime AUDIO_CTX_DUMP = "/home/alex/EriDiffusion/inference-flame/output/audio_context_dump/ltx2_audio_context.safetensors"
comptime NEG_CTX_DUMP = "/home/alex/EriDiffusion/inference-flame/cached_ltx2_negative.safetensors"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_hq"
comptime OUT_DIR2 = "/home/alex/mojodiffusion/output/ltx2_hq2"
comptime OUT_DIR_LONG = "/home/alex/mojodiffusion/output/ltx2_hq_long"
comptime OUT_DIR_AUDIOSYNC = "/home/alex/mojodiffusion/output/ltx2_audiosync_97f"
comptime SPATIAL_UPSCALER = "/home/alex/.serenity/models/ltx2_upscalers/ltx-2-spatial-upscaler-x2-1.0.safetensors"
comptime LORA_DISTILLED_PATH = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors"
comptime LORA_CAMERA_STATIC_PATH = "/home/alex/.serenity/models/loras/ltx-2-19b-lora-camera-control-static.safetensors"
comptime LORA_DETAILER_PATH = "/home/alex/.serenity/models/loras/ltx-2-19b-ic-lora-detailer.safetensors"
# Legacy single-stage default. The HQ two-stage path below overrides the
# distilled support LoRA per stage to match Lightricks' hq_2_stage_arg_parser.
comptime LORA_DISTILLED_MULT = Float32(1.0)
comptime LORA_DISTILLED_MULT_HQ_STAGE1 = Float32(0.25)
comptime LORA_DISTILLED_MULT_HQ_STAGE2 = Float32(0.5)
comptime LORA_CAMERA_STATIC_MULT = Float32(0.3)
# The detailer IC LoRA is disabled by default until IC/reference conditioning is
# wired. Applying it raw without the IC context warps faces.
comptime LORA_DETAILER_MULT = Float32(0.0)

# ── HQ short/staged shape (768x512 -> optional 2x stage) ─────────────────────
# latent_h = 512//32 = 16 ; latent_w = 768//32 = 24
# NUM_FRAMES=16 -> latent_f = (16-1)//8 + 1 = 2 ; S_V = 2*16*24 = 768
# AudioLatentShape.from_video_pixel_shape uses round(frames / fps * 25).
# At the AudioSync/official 24 fps profile: round(16/24*25) = 17.
comptime NUM_FRAMES = 16
comptime NF = 2          # latent frames
comptime NH = 16         # latent height (512/32)
comptime NW = 24         # latent width  (768/32)
comptime S_V = NF * NH * NW   # 768
comptime S_A = 17             # audio tokens
comptime N_TXT = 1024         # FULL text-context length (whole dump, no slice)
comptime S_VPAD = 1024        # max(S_V=768, N_TXT=1024, S_A=17)
comptime S_APAD = 1024
comptime NUM_LAYERS = 48

# ── Production single-stage temporal target (768x512) ────────────────────────
# Video VAE decode emits 1 + (latent_f - 1) * 8 frames, so NF_LONG=4 is a real
# 25-frame artifact at the 24 fps AudioSync/official profile instead of the
# legacy 9-frame smoke clip.
comptime NUM_FRAMES_LONG = 25
comptime NF_LONG = 4
comptime S_V_LONG = NF_LONG * NH * NW   # 1536
comptime S_A_LONG = 26
comptime S_VPAD_LONG = 1536             # max(S_V_LONG=1536, N_TXT=1024, S_A=26)
comptime S_APAD_LONG = 1024             # max(N_TXT=1024, S_A_LONG=26)

# ── AudioSync workflow temporal target (LTX-Desktop: 97 frames @ 24 fps) ──
# NUM_FRAMES=97 -> latent_f = (97-1)//8 + 1 = 13 ; S_V = 13*16*24 = 4992.
# AudioLatentShape.from_video_pixel_shape gives round(97/24*25) = 101 audio
# latent tokens for video-duration-matched generation. The separate 162-token
# reference-audio conditioning shape is tracked by ltx2_audiosync_profile_smoke.
comptime NUM_FRAMES_AUDIOSYNC = 97
comptime NF_AUDIOSYNC = 13
comptime S_V_AUDIOSYNC = NF_AUDIOSYNC * NH * NW   # 4992
comptime S_A_AUDIOSYNC = 101
comptime S_VPAD_AUDIOSYNC = 4992
comptime S_APAD_AUDIOSYNC = 1024

# ── HQ two-stage first production target ─────────────────────────────────────
# User-accepted target: 121 decoded frames @ 24 fps, final 768x512, muxed audio.
# Stage 1 runs at half spatial resolution (384x256), then the LTX spatial
# upsampler doubles the latent grid for stage-2 refine.
comptime NUM_FRAMES_STAGED = 121
comptime NF_STAGED = 16
comptime NH_STAGED = 8
comptime NW_STAGED = 12
comptime S_V_STAGED = NF_STAGED * NH_STAGED * NW_STAGED       # 1536
comptime S_A_STAGED = 126                                     # round(121/24*25)
comptime S_VPAD_STAGED = 1536                                 # max(S_V,N_TXT,S_A)
comptime S_APAD_STAGED = 1024
comptime OUT_W_STAGE1 = 384
comptime OUT_H_STAGE1 = 256

# Stage-2 shape (spatial 2x upsample of stage-1 latent).
comptime NH2_STAGED = NH_STAGED * 2                            # 16
comptime NW2_STAGED = NW_STAGED * 2                            # 24
comptime S_V2_STAGED = NF_STAGED * NH2_STAGED * NW2_STAGED     # 6144
comptime S_VPAD2_STAGED = 6144
comptime OUT_W2_STAGED = 768
comptime OUT_H2_STAGED = 512

comptime VD = 4096      # video inner_dim
comptime AD = 2048      # audio inner_dim
comptime V_HEADS = 32
comptime V_HDIM = 128
comptime A_HEADS = 32
comptime A_HDIM = 64
comptime CA_DIM = 2048  # audio_cross_attention_dim
comptime EPS = Float32(1e-6)

# config constants (LTX2Config::default)
comptime ROPE_THETA = Float64(10000.0)
comptime POS_EMBED_MAX_POS = Float64(20.0)
comptime BASE_HW = Float64(2048.0)
comptime CAUSAL_OFFSET = Float64(1.0)
comptime VAE_SF0 = Float64(8.0)
comptime VAE_SF1 = Float64(32.0)
comptime VAE_SF2 = Float64(32.0)
comptime AUDIO_SCALE_FACTOR = Float64(4.0)
comptime FRAME_RATE = Float64(24.0)
comptime TS_MULT = Float32(1000.0)
comptime SEED = UInt64(42)

comptime AUDIO_C = 8     # audio latent channels
comptime AUDIO_MEL = 16  # audio latent mel bins (C*F=128)

# AudioSync production profile uses 20 HQ denoise steps.
comptime HQ_STEPS = 20
comptime RES2S_TERMINAL_SIGMA = Float32(0.0011)
comptime RES2S_BONG_ITERS = 100
# FP8 residency keeps raw block bytes on GPU and avoids repeated host/device
# streaming in each denoise eval. Full 4..46 raw residency is ~15.5 GiB before
# the current block's BF16 materialization plus VAE decode still in scope.
comptime RESIDENT_FIRST = 4
comptime RESIDENT_LAST_SINGLE = 40
comptime RESIDENT_LAST_AUDIOSYNC = 12
comptime RESIDENT_LAST_STAGED = 12
# LTX2Scheduler.execute defaults (ltx-core schedulers.py:25-29)
comptime SCHED_MAX_SHIFT = Float64(2.05)
comptime SCHED_BASE_SHIFT = Float64(0.95)
comptime SCHED_TERMINAL = Float64(0.1)
comptime SCHED_BASE_ANCHOR = Float64(1024.0)
comptime SCHED_MAX_ANCHOR = Float64(4096.0)


struct _HQLoraStack(Movable):
    var distilled: LoraSet
    var camera_static: LoraSet
    var detailer: LoraSet

    def __init__(
        out self,
        var distilled: LoraSet,
        var camera_static: LoraSet,
        var detailer: LoraSet,
    ):
        self.distilled = distilled^
        self.camera_static = camera_static^
        self.detailer = detailer^

    @staticmethod
    def load() raises -> _HQLoraStack:
        return _HQLoraStack(
            LoraSet.load(String(LORA_DISTILLED_PATH)),
            LoraSet.load(String(LORA_CAMERA_STATIC_PATH)),
            LoraSet.load(String(LORA_DETAILER_PATH)),
        )

    def validate(self) raises:
        if self.distilled.format != FMT_LTX2_DISTILLED:
            raise Error("HQ: distilled LoRA not FMT_LTX2_DISTILLED")
        if self.distilled.num_lora_pairs_in_file() != self.distilled.num_mappings():
            raise Error("HQ: distilled LoRA has unmapped pairs")
        if self.camera_static.num_lora_pairs_in_file() != self.camera_static.num_mappings():
            raise Error("HQ: camera-static LoRA has unmapped pairs")
        if self.detailer.num_lora_pairs_in_file() != self.detailer.num_mappings():
            raise Error("HQ: detailer LoRA has unmapped pairs")

    def print_summary_scaled(
        self,
        distilled_mult: Float32,
        camera_static_mult: Float32,
        detailer_mult: Float32,
    ) raises:
        print("  [lora] distilled", self.distilled.num_mappings(),
              "mappings @", distilled_mult)
        print("  [lora] camera-static", self.camera_static.num_mappings(),
              "mappings @", camera_static_mult)
        print("  [lora] detailer", self.detailer.num_mappings(),
              "mappings @", detailer_mult, "(disabled until IC)")

    def print_summary(self) raises:
        self.print_summary_scaled(
            LORA_DISTILLED_MULT, LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT
        )

    def apply_to_globals_scaled(
        self,
        mut gw: Dict[String, ArcPointer[Tensor]],
        distilled_mult: Float32,
        camera_static_mult: Float32,
        detailer_mult: Float32,
        ctx: DeviceContext,
    ) raises -> Int:
        var total = 0
        if distilled_mult != Float32(0.0):
            total += self.distilled.apply_to_globals(gw, distilled_mult, ctx)
        if camera_static_mult != Float32(0.0):
            total += self.camera_static.apply_to_globals(gw, camera_static_mult, ctx)
        if detailer_mult != Float32(0.0):
            total += self.detailer.apply_to_globals(gw, detailer_mult, ctx)
        return total

    def apply_to_globals(
        self,
        mut gw: Dict[String, ArcPointer[Tensor]],
        ctx: DeviceContext,
    ) raises -> Int:
        return self.apply_to_globals_scaled(
            gw, LORA_DISTILLED_MULT, LORA_CAMERA_STATIC_MULT,
            LORA_DETAILER_MULT, ctx
        )

    def preload_block_factors(mut self, ctx: DeviceContext) raises -> Int:
        var total = 0
        total += self.distilled.preload_ltx2_block_factors(ctx)
        total += self.camera_static.preload_ltx2_block_factors(ctx)
        total += self.detailer.preload_ltx2_block_factors(ctx)
        return total

    def apply_to_av_block(
        self,
        block_idx: Int,
        mut block: LTX2AVBlockWeights,
        ctx: DeviceContext,
    ) raises -> Int:
        var total = 0
        if LORA_DISTILLED_MULT != Float32(0.0):
            total += self.distilled.apply_to_av_block(
                block_idx, block, LORA_DISTILLED_MULT, ctx
            )
        if LORA_CAMERA_STATIC_MULT != Float32(0.0):
            total += self.camera_static.apply_to_av_block(
                block_idx, block, LORA_CAMERA_STATIC_MULT, ctx
            )
        if LORA_DETAILER_MULT != Float32(0.0):
            total += self.detailer.apply_to_av_block(
                block_idx, block, LORA_DETAILER_MULT, ctx
            )
        return total

    def apply_to_av_block_merged(
        self,
        block_idx: Int,
        mut block: LTX2AVBlockWeights,
        ctx: DeviceContext,
    ) raises -> Int:
        var ds = LTX2BlockLoraDeltaSet()
        if LORA_DISTILLED_MULT != Float32(0.0):
            _ = self.distilled.accumulate_ltx2_block_deltas(
                block_idx, ds, LORA_DISTILLED_MULT, ctx
            )
        if LORA_CAMERA_STATIC_MULT != Float32(0.0):
            _ = self.camera_static.accumulate_ltx2_block_deltas(
                block_idx, ds, LORA_CAMERA_STATIC_MULT, ctx
            )
        if LORA_DETAILER_MULT != Float32(0.0):
            _ = self.detailer.accumulate_ltx2_block_deltas(
                block_idx, ds, LORA_DETAILER_MULT, ctx
            )
        return ds.apply_to_av_block(block, ctx)

    def attach_to_av_block_factorized_scaled(
        self,
        block_idx: Int,
        mut block: LTX2AVBlockWeights,
        distilled_mult: Float32,
        camera_static_mult: Float32,
        detailer_mult: Float32,
        ctx: DeviceContext,
    ) raises -> Int:
        var total = 0
        if distilled_mult != Float32(0.0):
            total += self.distilled.attach_ltx2_cached_block_factors(
                block_idx, block, distilled_mult, ctx
            )
        if camera_static_mult != Float32(0.0):
            total += self.camera_static.attach_ltx2_cached_block_factors(
                block_idx, block, camera_static_mult, ctx
            )
        if detailer_mult != Float32(0.0):
            total += self.detailer.attach_ltx2_cached_block_factors(
                block_idx, block, detailer_mult, ctx
            )
        return total

    def attach_to_av_block_factorized(
        self,
        block_idx: Int,
        mut block: LTX2AVBlockWeights,
        ctx: DeviceContext,
    ) raises -> Int:
        return self.attach_to_av_block_factorized_scaled(
            block_idx, block, LORA_DISTILLED_MULT, LORA_CAMERA_STATIC_MULT,
            LORA_DETAILER_MULT, ctx
        )


# ── shape helpers ──────────────────────────────────────────────────────────────
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^

def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^

def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d)
    return s^

def _sh5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d)
    s.append(e); return s^

def _sh1d(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^


def _is_boundary(i: Int) -> Bool:
    return i == 0 or i == 1 or i == 2 or i == 3 or i == 47


def _st_has(st: ShardedSafeTensors, name: String) -> Bool:
    for ref nm in st.names():
        if nm == name:
            return True
    return False


def _load_global_bf16(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var key = String("model.diffusion_model.") + name
    if not _st_has(st, key):
        key = name
    var tv = st.tensor_view(key)
    return Tensor.from_view_as_bf16(tv, ctx)


def _load_global_weights_dict(
    st: ShardedSafeTensors, ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    var gw = Dict[String, ArcPointer[Tensor]]()
    var patch_keys = List[String]()
    patch_keys.append(String("patchify_proj.weight"))
    patch_keys.append(String("audio_patchify_proj.weight"))
    patch_keys.append(String("proj_out.weight"))
    patch_keys.append(String("audio_proj_out.weight"))
    var adaln_families = List[String]()
    adaln_families.append(String("adaln_single"))
    adaln_families.append(String("audio_adaln_single"))
    adaln_families.append(String("prompt_adaln_single"))
    adaln_families.append(String("audio_prompt_adaln_single"))
    adaln_families.append(String("av_ca_video_scale_shift_adaln_single"))
    adaln_families.append(String("av_ca_audio_scale_shift_adaln_single"))
    adaln_families.append(String("av_ca_a2v_gate_adaln_single"))
    adaln_families.append(String("av_ca_v2a_gate_adaln_single"))
    for ref fam in patch_keys:
        gw[fam] = ArcPointer[Tensor](_load_global_bf16(st, fam, ctx))
    for ref fam in adaln_families:
        var k1 = fam + ".emb.timestep_embedder.linear_1.weight"
        var k2 = fam + ".emb.timestep_embedder.linear_2.weight"
        var k3 = fam + ".linear.weight"
        gw[k1] = ArcPointer[Tensor](_load_global_bf16(st, k1, ctx))
        gw[k2] = ArcPointer[Tensor](_load_global_bf16(st, k2, ctx))
        gw[k3] = ArcPointer[Tensor](_load_global_bf16(st, k3, ctx))
    return gw^


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _linear_b(x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return linear(x, w, Optional[Tensor](_clone(b, ctx)), ctx)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises -> Bool:
    var h = t.to_host(ctx)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    var finite = True
    for i in range(len(h)):
        var v = Float64(h[i])
        if not (v == v) or v > 1.0e38 or v < -1.0e38:
            finite = False
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(len(h))
    var var_ = s2 / Float64(len(h)) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print("   [stat]", name, "mean=", Float32(mean), "std=", Float32(sqrt(var_)),
          "absmax=", Float32(amax), "finite=", finite)
    return finite


# ── HQ LTX2Scheduler sigma schedule (token-count shift) ──
# Port of ltx-core LTX2Scheduler.execute (schedulers.py:21-57) for a velocity
# model. tokens = prod(latent.shape[2:]) = NF*NH*NW = S_V (video). Returns
# HQ_STEPS+1 sigmas descending from 1.0 to 0.0 (trailing 0.0 terminator).
def _ltx2_scheduler_sigmas(steps: Int, tokens: Int) -> List[Float32]:
    var x1 = SCHED_BASE_ANCHOR
    var x2 = SCHED_MAX_ANCHOR
    var mm = (SCHED_MAX_SHIFT - SCHED_BASE_SHIFT) / (x2 - x1)
    var b = SCHED_BASE_SHIFT - mm * x1
    var sigma_shift = Float64(tokens) * mm + b
    var es = fexp(sigma_shift)

    # linspace(1.0, 0.0, steps+1)
    var lin = List[Float64]()
    for i in range(steps + 1):
        lin.append(1.0 - Float64(i) / Float64(steps))

    # power=1: shifted = es / (es + (1/sigma - 1))   (sigma!=0); 0 stays 0.
    var shifted = List[Float64]()
    for i in range(steps + 1):
        var s = lin[i]
        if s != 0.0:
            shifted.append(es / (es + (1.0 / s - 1.0)))
        else:
            shifted.append(0.0)

    # stretch so terminal (last non-zero) matches SCHED_TERMINAL.
    # one_minus_z = 1 - nonzero ; scale = one_minus_z[-1] / (1 - terminal)
    # stretched = 1 - one_minus_z/scale
    var last_nz = -1
    for i in range(steps + 1):
        if shifted[i] != 0.0:
            last_nz = i
    var scale_factor = (1.0 - shifted[last_nz]) / (1.0 - SCHED_TERMINAL)
    var out = List[Float32]()
    for i in range(steps + 1):
        var s = shifted[i]
        if s != 0.0:
            var omz = 1.0 - s
            out.append(Float32(1.0 - omz / scale_factor))
        else:
            out.append(Float32(0.0))
    return out^


# ── sinusoidal timestep embedding (diffusers get_timestep_embedding) ──
def _timestep_embedding(
    ts: List[Float32], dim: Int, dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var n = len(ts)
    var half = dim // 2
    var out = List[Float32]()
    out.resize(n * dim, Float32(0.0))
    for r in range(n):
        var t = Float64(ts[r])
        for i in range(half):
            var freq = fpow(Float64(2.718281828459045), -Float64(i) * flog(10000.0) / Float64(half))
            var arg = t * freq
            out[r * dim + i] = Float32(fcos(arg))
            out[r * dim + half + i] = Float32(fsin(arg))
    return Tensor.from_host(out, _sh2(n, dim), dtype, ctx)


def _adaln_single(
    st: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    base: String,
    ts_vals: List[Float32],
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var emb = _timestep_embedding(ts_vals, 256, STDtype.BF16, ctx)
    var w1 = _clone(gw[base + ".emb.timestep_embedder.linear_1.weight"][], ctx)
    var b1 = _load_global_bf16(st, base + ".emb.timestep_embedder.linear_1.bias", ctx)
    var h = _linear_b(emb, w1, b1, ctx)
    h = silu(h, ctx)
    var w2 = _clone(gw[base + ".emb.timestep_embedder.linear_2.weight"][], ctx)
    var b2 = _load_global_bf16(st, base + ".emb.timestep_embedder.linear_2.bias", ctx)
    var embedded = _linear_b(h, w2, b2, ctx)
    var h2 = silu(embedded, ctx)
    var lw = _clone(gw[base + ".linear.weight"][], ctx)
    var lb = _load_global_bf16(st, base + ".linear.bias", ctx)
    var mod = _linear_b(h2, lw, lb, ctx)
    return (mod^, embedded^)


def _compute_rope(
    coords: List[Float64],
    num_pos_dims: Int,
    P: Int,
    dim: Int,
    max_pos: List[Float64],
    theta: Float64,
    num_heads: Int,
    dtype: STDtype,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var num_rope_elems = num_pos_dims * 2
    var freq_count = dim // num_rope_elems
    var half_dim = dim // 2
    var rope_freqs = freq_count * num_pos_dims
    var pad = half_dim - rope_freqs
    if pad < 0:
        raise Error("rope: rope_freqs > half_dim")

    var denom = Float64(freq_count - 1)
    if denom < 1.0:
        denom = 1.0
    var freq = List[Float64]()
    for i in range(freq_count):
        freq.append(fpow(theta, Float64(i) / denom) * pi / 2.0)

    var cos_tok = List[Float32]()
    var sin_tok = List[Float32]()
    cos_tok.resize(P * half_dim, Float32(0.0))
    sin_tok.resize(P * half_dim, Float32(0.0))
    var half_head = (dim // 2) // num_heads

    for p in range(P):
        var scaled = List[Float64]()
        for d in range(num_pos_dims):
            var s0 = coords[(d * P + p) * 2 + 0]
            var e0 = coords[(d * P + p) * 2 + 1]
            var mid = (s0 + e0) * 0.5
            var g = mid / max_pos[d]
            scaled.append(2.0 * g - 1.0)
        var base = p * half_dim
        for q in range(pad):
            cos_tok[base + q] = Float32(1.0)
            sin_tok[base + q] = Float32(0.0)
        var off = base + pad
        for i in range(freq_count):
            for d in range(num_pos_dims):
                var a = scaled[d] * freq[i]
                cos_tok[off] = Float32(fcos(a))
                sin_tok[off] = Float32(fsin(a))
                off += 1

    var cos_rows = List[Float32]()
    var sin_rows = List[Float32]()
    for p in range(P):
        for h in range(num_heads):
            var src = p * half_dim + h * half_head
            for j in range(half_head):
                cos_rows.append(cos_tok[src + j])
                sin_rows.append(sin_tok[src + j])
    var sh = _sh2(P * num_heads, half_head)
    var cos_t = Tensor.from_host(cos_rows, sh.copy(), dtype, ctx)
    var sin_t = Tensor.from_host(sin_rows, sh^, dtype, ctx)
    return (cos_t^, sin_t^)


def _build_video_coords_dims(nf: Int, nh: Int, nw: Int) -> List[Float64]:
    """Video 3D RoPE coord boxes for an arbitrary latent grid (nf,nh,nw).

    The H/W coordinate scales (VAE_SF1/2 = 32) are PIXEL-space step sizes per
    latent cell and DO NOT change with resolution — stage-2 has more, finer
    cells but the same per-cell pixel extent, exactly matching the reference
    VideoLatentShape positions builder. So we just iterate the new grid.
    """
    var s_v = nf * nh * nw
    var out = List[Float64]()
    out.resize(3 * s_v * 2, Float64(0.0))
    var vae_t = VAE_SF0
    for f in range(nf):
        for h in range(nh):
            for w in range(nw):
                var tok = f * nh * nw + h * nw + w
                var fs = Float64(f) * VAE_SF0
                var fe = Float64(f + 1) * VAE_SF0
                var fsc = fs + CAUSAL_OFFSET - vae_t
                var fec = fe + CAUSAL_OFFSET - vae_t
                if fsc < 0.0: fsc = 0.0
                if fec < 0.0: fec = 0.0
                fsc = fsc / FRAME_RATE
                fec = fec / FRAME_RATE
                out[(0 * s_v + tok) * 2 + 0] = fsc
                out[(0 * s_v + tok) * 2 + 1] = fec
                out[(1 * s_v + tok) * 2 + 0] = Float64(h) * VAE_SF1
                out[(1 * s_v + tok) * 2 + 1] = Float64(h + 1) * VAE_SF1
                out[(2 * s_v + tok) * 2 + 0] = Float64(w) * VAE_SF2
                out[(2 * s_v + tok) * 2 + 1] = Float64(w + 1) * VAE_SF2
    return out^


def _build_video_coords() -> List[Float64]:
    return _build_video_coords_dims(NF, NH, NW)


def _build_audio_coords_dims(s_a: Int) -> List[Float64]:
    var out = List[Float64]()
    out.resize(s_a * 2, Float64(0.0))
    var mel_to_sec = 16000.0 / 160.0
    var scale = AUDIO_SCALE_FACTOR
    for t in range(s_a):
        var ms = Float64(t) * scale
        var me = Float64(t + 1) * scale
        var msc = ms + CAUSAL_OFFSET - scale
        var mec = me + CAUSAL_OFFSET - scale
        if msc < 0.0: msc = 0.0
        if mec < 0.0: mec = 0.0
        out[t * 2 + 0] = msc / mel_to_sec
        out[t * 2 + 1] = mec / mel_to_sec
    return out^


def _build_audio_coords() -> List[Float64]:
    return _build_audio_coords_dims(S_A)


def _video_temporal_coords_dims(vc: List[Float64], s_v: Int) -> List[Float64]:
    var out = List[Float64]()
    out.resize(s_v * 2, Float64(0.0))
    for p in range(s_v):
        out[p * 2 + 0] = vc[(0 * s_v + p) * 2 + 0]
        out[p * 2 + 1] = vc[(0 * s_v + p) * 2 + 1]
    return out^


def _video_temporal_coords(vc: List[Float64]) -> List[Float64]:
    return _video_temporal_coords_dims(vc, S_V)


def _mp1() -> List[Float64]:
    var m = List[Float64](); m.append(POS_EMBED_MAX_POS); return m^

def _mp3() -> List[Float64]:
    var m = List[Float64](); m.append(POS_EMBED_MAX_POS); m.append(BASE_HW)
    m.append(BASE_HW); return m^


struct _Globals(Movable):
    var v_pin_w: Tensor
    var v_pin_b: Tensor
    var a_pin_w: Tensor
    var a_pin_b: Tensor
    var v_pout_w: Tensor
    var v_pout_b: Tensor
    var a_pout_w: Tensor
    var a_pout_b: Tensor
    var v_sst: Tensor
    var a_sst: Tensor

    def __init__(out self, var v_pin_w: Tensor, var v_pin_b: Tensor,
                 var a_pin_w: Tensor, var a_pin_b: Tensor,
                 var v_pout_w: Tensor, var v_pout_b: Tensor,
                 var a_pout_w: Tensor, var a_pout_b: Tensor,
                 var v_sst: Tensor, var a_sst: Tensor):
        self.v_pin_w = v_pin_w^; self.v_pin_b = v_pin_b^
        self.a_pin_w = a_pin_w^; self.a_pin_b = a_pin_b^
        self.v_pout_w = v_pout_w^; self.v_pout_b = v_pout_b^
        self.a_pout_w = a_pout_w^; self.a_pout_b = a_pout_b^
        self.v_sst = v_sst^; self.a_sst = a_sst^


def _output_stage(
    hs: Tensor, sst: Tensor, embedded: Tensor,
    proj_w: Tensor, proj_b: Tensor, dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    var ones = List[Float32](); var zeros = List[Float32]()
    for _ in range(dim):
        ones.append(Float32(1.0)); zeros.append(Float32(0.0))
    var w_ln = Tensor.from_host(ones, _sh1d(dim), hs.dtype(), ctx)
    var b_ln = Tensor.from_host(zeros, _sh1d(dim), hs.dtype(), ctx)
    var normed = layer_norm(hs, w_ln, b_ln, EPS, ctx)
    var shift_row = reshape(slice(sst, 0, 0, 1, ctx), _sh3(1, 1, dim), ctx)
    var scale_row = reshape(slice(sst, 0, 1, 1, ctx), _sh3(1, 1, dim), ctx)
    var v_shift = add(shift_row, embedded, ctx)
    var v_scale = add(scale_row, embedded, ctx)
    var one_plus = add_scalar(v_scale, Float32(1.0), ctx)
    var out = add(mul(normed, one_plus, ctx), v_shift, ctx)
    return _linear_b(out, proj_w, proj_b, ctx)


struct _Mod(Movable):
    var v_temb: Tensor
    var a_temb: Tensor
    var v_embedded: Tensor
    var a_embedded: Tensor
    var v_ca_ss: Tensor
    var a_ca_ss: Tensor
    var v_ca_gate: Tensor
    var a_ca_gate: Tensor
    var v_prompt_ts: Tensor
    var a_prompt_ts: Tensor

    def __init__(out self, var v_temb: Tensor, var a_temb: Tensor,
                 var v_embedded: Tensor, var a_embedded: Tensor,
                 var v_ca_ss: Tensor, var a_ca_ss: Tensor,
                 var v_ca_gate: Tensor, var a_ca_gate: Tensor,
                 var v_prompt_ts: Tensor, var a_prompt_ts: Tensor):
        self.v_temb = v_temb^; self.a_temb = a_temb^
        self.v_embedded = v_embedded^; self.a_embedded = a_embedded^
        self.v_ca_ss = v_ca_ss^; self.a_ca_ss = a_ca_ss^
        self.v_ca_gate = v_ca_gate^; self.a_ca_gate = a_ca_gate^
        self.v_prompt_ts = v_prompt_ts^; self.a_prompt_ts = a_prompt_ts^


def _build_mod_dims(
    st: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    sigma: Float32,
    s_v: Int,
    s_a: Int,
    ctx: DeviceContext,
) raises -> _Mod:
    var ts_v = List[Float32]()
    for _ in range(s_v): ts_v.append(sigma * TS_MULT)
    var vt = _adaln_single(st, gw, String("adaln_single"), ts_v, ctx)
    var v_temb = reshape(vt[0], _sh3(1, s_v, 9 * VD), ctx)
    var v_embedded = reshape(vt[1], _sh3(1, s_v, VD), ctx)

    var ts_a = List[Float32]()
    for _ in range(s_a): ts_a.append(sigma * TS_MULT)
    var at = _adaln_single(st, gw, String("audio_adaln_single"), ts_a, ctx)
    var a_temb = reshape(at[0], _sh3(1, s_a, 9 * AD), ctx)
    var a_embedded = reshape(at[1], _sh3(1, s_a, AD), ctx)

    var g1 = List[Float32](); g1.append(sigma * TS_MULT)
    var vcs = _adaln_single(st, gw, String("av_ca_video_scale_shift_adaln_single"), g1, ctx)
    var v_ca_ss = reshape(vcs[0], _sh3(1, 1, 4 * VD), ctx)
    var acs = _adaln_single(st, gw, String("av_ca_audio_scale_shift_adaln_single"), g1, ctx)
    var a_ca_ss = reshape(acs[0], _sh3(1, 1, 4 * AD), ctx)
    var vcg = _adaln_single(st, gw, String("av_ca_a2v_gate_adaln_single"), g1, ctx)
    var v_ca_gate = reshape(vcg[0], _sh3(1, 1, VD), ctx)
    var acg = _adaln_single(st, gw, String("av_ca_v2a_gate_adaln_single"), g1, ctx)
    var a_ca_gate = reshape(acg[0], _sh3(1, 1, AD), ctx)

    var ts_p = List[Float32]()
    for _ in range(N_TXT): ts_p.append(sigma * TS_MULT)
    var vpt = _adaln_single(st, gw, String("prompt_adaln_single"), ts_p, ctx)
    var v_prompt_ts = reshape(vpt[0], _sh3(1, N_TXT, 2 * VD), ctx)
    var apt = _adaln_single(st, gw, String("audio_prompt_adaln_single"), ts_p, ctx)
    var a_prompt_ts = reshape(apt[0], _sh3(1, N_TXT, 2 * AD), ctx)

    return _Mod(v_temb^, a_temb^, v_embedded^, a_embedded^, v_ca_ss^, a_ca_ss^,
                v_ca_gate^, a_ca_gate^, v_prompt_ts^, a_prompt_ts^)


def _build_mod(
    st: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    sigma: Float32,
    ctx: DeviceContext,
) raises -> _Mod:
    return _build_mod_dims(st, gw, sigma, S_V, S_A, ctx)


def _load_video_nag_context(
    v_conn: LTX2ConnectorWeights,
    ctx: DeviceContext,
) raises -> Tensor:
    """Load the cached null/negative video context and project it like positive context."""
    var neg_st = ShardedSafeTensors.open(String(NEG_CTX_DUMP))
    var neg_pre = Tensor.from_view_as_bf16(neg_st.tensor_view("text_hidden"), ctx)
    return ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](v_conn, neg_pre, ctx)


def _nag_audio_placeholder(ctx: DeviceContext) raises -> Tensor:
    var z = List[Float32]()
    z.append(Float32(0.0))
    var sh = List[Int]()
    sh.append(1)
    return Tensor.from_host(z, sh^, STDtype.BF16, ctx)


# ── ONE model forward at a given sigma. Returns velocity (video, audio) in the
#    LATENT layout: video [1,128,NF,NH,NW], audio [1,8,S_A,16]. This is the same
#    forward the MVP did per step, factored out so res_2s can call it twice/step
#    (current sigma + midpoint sub_sigma). All inputs (contexts/rope/globals) are
#    sigma-independent and passed in; only the modulation rebuilds per sigma.    ──
def _model_forward_p[
    S_V_CT: Int, S_A_CT: Int, S_VPAD_CT: Int, S_APAD_CT: Int
](
    ck: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    lora_stack: _HQLoraStack,
    apply_lora: Bool,
    distilled_lora_mult: Float32,
    camera_static_lora_mult: Float32,
    detailer_lora_mult: Float32,
    cfg: LTX2Config,
    g: _Globals,
    stream: LTX2BlockStream,
    video_x: Tensor,
    audio_x: Tensor,
    enc: Tensor, aenc: Tensor,
    v_cos: Tensor, v_sin: Tensor, a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor, ca_a_cos: Tensor, ca_a_sin: Tensor,
    nag: NAGContext,
    sigma: Float32,
    nf: Int, nh: Int, nw: Int,
    profile: Bool,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """ONE DiT forward at `sigma`. Comptime params are the (token-count, pad)
    needed by the block kernel; runtime nf/nh/nw drive the (un)patchify reshape
    and the per-sigma modulation. Stage 1 uses [S_V,S_VPAD]; stage 2 (post 2x
    spatial upsample) uses [S_V2,S_VPAD2] with nh/nw doubled."""
    # patchify
    var v_flat = permute(reshape(video_x, _sh3(1, 128, S_V_CT), ctx),
                         _video_perm(), ctx)
    var a_flat = reshape(permute(audio_x, _audio_perm(), ctx),
                         _sh3(1, S_A_CT, 128), ctx)
    var hs = _linear_b(v_flat, g.v_pin_w, g.v_pin_b, ctx)
    var ahs = _linear_b(a_flat, g.a_pin_w, g.a_pin_b, ctx)

    var mod = _build_mod_dims(ck, gw, sigma, S_V_CT, S_A_CT, ctx)

    var t_load = Float64(0.0)
    var t_lora = Float64(0.0)
    var t_block = Float64(0.0)
    for i in range(NUM_LAYERS):
        var t0 = Float64(0.0)
        if profile:
            ctx.synchronize()
            t0 = perf_counter()
        var w: LTX2AVBlockWeights
        if _is_boundary(i):
            w = LTX2AVBlockWeights.load(String(CKPT_FP8), i, cfg, ctx)
        else:
            var blk = stream.load_block_bf16(i, ctx)
            w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx)
        if profile:
            ctx.synchronize()
            t_load += perf_counter() - t0
        if apply_lora:
            if profile:
                t0 = perf_counter()
            _ = lora_stack.attach_to_av_block_factorized_scaled(
                i, w, distilled_lora_mult, camera_static_lora_mult,
                detailer_lora_mult, ctx
            )
            if profile:
                ctx.synchronize()
                t_lora += perf_counter() - t0
        if nag.enabled:
            if profile:
                t0 = perf_counter()
            var outs_nag = ltx2_block_forward_av_nag[
                S_V_CT, S_A_CT, N_TXT, S_VPAD_CT, S_APAD_CT
            ](
                w, hs, ahs, enc, aenc,
                mod.v_temb, mod.a_temb, mod.v_ca_ss, mod.a_ca_ss,
                mod.v_ca_gate, mod.a_ca_gate,
                mod.v_prompt_ts, mod.a_prompt_ts,
                v_cos, v_sin, a_cos, a_sin,
                ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
                nag, EPS, ctx,
            )
            hs = _clone(outs_nag[0], ctx)
            ahs = _clone(outs_nag[1], ctx)
            if profile:
                ctx.synchronize()
                t_block += perf_counter() - t0
        else:
            if profile:
                t0 = perf_counter()
            var outs = ltx2_block_forward_av[
                S_V_CT, S_A_CT, N_TXT, S_VPAD_CT, S_APAD_CT
            ](
                w, hs, ahs, enc, aenc,
                mod.v_temb, mod.a_temb, mod.v_ca_ss, mod.a_ca_ss,
                mod.v_ca_gate, mod.a_ca_gate,
                mod.v_prompt_ts, mod.a_prompt_ts,
                v_cos, v_sin, a_cos, a_sin,
                ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, EPS, ctx,
            )
            hs = _clone(outs[0], ctx)
            ahs = _clone(outs[1], ctx)
            if profile:
                ctx.synchronize()
                t_block += perf_counter() - t0

    var v_vel_flat = _output_stage(hs, g.v_sst, mod.v_embedded, g.v_pout_w,
                                   g.v_pout_b, VD, ctx)
    var a_vel_flat = _output_stage(ahs, g.a_sst, mod.a_embedded, g.a_pout_w,
                                   g.a_pout_b, AD, ctx)
    var v_vel = reshape(permute(v_vel_flat, _unvideo_perm(), ctx),
                        _sh5(1, 128, nf, nh, nw), ctx)
    var a_vel = permute(
        reshape(a_vel_flat, _sh4(1, S_A_CT, AUDIO_C, AUDIO_MEL), ctx),
        _unaudio_perm(), ctx,
    )
    if profile:
        ctx.synchronize()
        print(
            "  [profile] forward S_V=", S_V_CT, " sigma=", sigma,
            " load=", Float32(t_load), "s lora=", Float32(t_lora),
            "s block=", Float32(t_block), "s",
        )
    return (v_vel^, a_vel^)


def _model_forward(
    ck: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    lora_stack: _HQLoraStack,
    apply_lora: Bool,
    cfg: LTX2Config,
    g: _Globals,
    stream: LTX2BlockStream,
    video_x: Tensor,
    audio_x: Tensor,
    enc: Tensor, aenc: Tensor,
    v_cos: Tensor, v_sin: Tensor, a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor, ca_a_cos: Tensor, ca_a_sin: Tensor,
    sigma: Float32,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var nag = NAGContext.disabled(ctx)
    return _model_forward_p[S_V, S_A, S_VPAD, S_APAD](
        ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT,
        LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream, video_x,
        audio_x, enc, aenc,
        v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
        nag,
        sigma, NF, NH, NW, False, ctx,
    )


# denoised x0 estimate from a velocity prediction (rectified flow): x - v*sigma.
def _denoise_from_vel(x: Tensor, vel: Tensor, sigma: Float32, ctx: DeviceContext) raises -> Tensor:
    return sub(x, mul_scalar(vel, sigma, ctx), ctx)


# NCDHW [1,128,F,H,W] -> NDHWC [1,F,H,W,128]  (perm 0,2,3,4,1).
def _ncdhw_to_ndhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(4); p.append(1)
    return permute(x, p^, ctx)


# NDHWC [1,F,H,W,128] -> NCDHW [1,128,F,H,W]  (perm 0,4,1,2,3).
def _ndhwc_to_ncdhw(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var p = List[Int]()
    p.append(0); p.append(4); p.append(1); p.append(2); p.append(3)
    return permute(x, p^, ctx)


# Load a VAE per-channel-stat [128] from the bf16 checkpoint as BF16 storage.
def _load_vae_stat(name: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(String(CKPT_BF16))
    var tv = st.tensor_view(String("vae.per_channel_statistics.") + name)
    return Tensor.from_view_as_bf16(tv, ctx)


# Stage-2 forward-noise the upscaled latent at noise_scale (GaussianNoiser, mask=1):
#   x = noise * noise_scale + upscaled * (1 - noise_scale)
# (ltx-core components/noisers.py GaussianNoiser.__call__, denoise_mask all-ones.)
def _noise_init(
    upscaled: Tensor, noise_scale: Float32, seed: UInt64, ctx: DeviceContext
) raises -> Tensor:
    # rng-contract: mojo-native-not-pytorch-parity. LTX-2 Python uses
    # torch.Generator + torch.randn(... dtype=latent.dtype); parity gates must
    # import oracle noise unless this RNG is proven same-seed equivalent.
    var noise = randn(upscaled.shape().copy(), seed, upscaled.dtype(), ctx)
    var a = mul_scalar(noise, noise_scale, ctx)
    var b = mul_scalar(upscaled, Float32(1.0) - noise_scale, ctx)
    return add(a, b, ctx)


def _res2s_hq_noise(x: Tensor, seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    """Reference HQ SDE noise prep: global-normalize, then channelwise normalize.

    Mirrors `_get_new_noise` in LTX-2 `utils/samplers.py` for shape and
    normalization, not for same-seed PyTorch RNG parity. Mojo currently
    normalizes in an explicit F32 GPU workspace, then casts back to the latent
    dtype at the sampler boundary.
    """
    var all_dims = List[Int]()
    # rng-contract: mojo-native-not-pytorch-parity. Use oracle noise tensors for
    # PyTorch parity gates.
    var noise_storage = randn(x.shape().copy(), seed, x.dtype(), ctx)
    var sh = noise_storage.shape()
    for i in range(len(sh)):
        all_dims.append(i)
    var global_mean = reduce_mean_f32(noise_storage, all_dims.copy(), True, ctx)
    # dtype-contract: allow-f32-boundary. SDE noise normalization reductions are
    # compute workspace only; return value is cast back to x.dtype().
    var noise = cast_tensor(noise_storage, global_mean.dtype(), ctx)
    var global_std = reduce_std_f32(noise_storage, all_dims^, True, ctx)
    var centered = sub(noise, global_mean, ctx)
    var globally_normed = div(centered, global_std, ctx)

    var channel_dims = List[Int]()
    channel_dims.append(-2)
    channel_dims.append(-1)
    var channel_mean = reduce_mean_f32(globally_normed, channel_dims.copy(), True, ctx)
    var channel_std = reduce_std_f32(globally_normed, channel_dims^, True, ctx)
    var normalized = div(sub(globally_normed, channel_mean, ctx), channel_std, ctx)
    return cast_tensor(normalized, x.dtype(), ctx)


def run_single_p[
    NUM_FRAMES_CT: Int,
    NF_CT: Int,
    NH_CT: Int,
    NW_CT: Int,
    S_V_CT: Int,
    S_A_CT: Int,
    S_VPAD_CT: Int,
    S_APAD_CT: Int,
](
    apply_lora: Bool, out_dir: String, max_steps: Int, use_resident: Bool,
    include_audio: Bool, use_nag: Bool,
) raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2.3 T2V HQ (res_2s) 768x512 distilled ===")
    print("  target frames:", NUM_FRAMES_CT, " decoded frames:", 1 + (NF_CT - 1) * 8)
    print("  res:768x512  NF/NH/NW:", NF_CT, NH_CT, NW_CT, " S_V:", S_V_CT, " S_A:", S_A_CT,
          " N_TXT:", N_TXT, " blocks:", NUM_LAYERS)
    print("  sampler: res_2s (2nd-order RK)  steps:", HQ_STEPS,
          "  LoRA:", "HQ stack" if apply_lora else "OFF", " out_dir:", out_dir)
    print("  weights:", "FP8-resident warm range" if use_resident else "FP8 stream")
    print("  audio:", "ON (A/V default)" if include_audio else "OFF (explicit noaudio)")
    print("  NAG:", "video-only ON" if use_nag else "OFF")
    var frame_prefix = String("hq_frame")
    if max_steps != 0:
        frame_prefix = String("dev_frame")
        print("  [quality] DEV SMOKE ONLY: incomplete denoise, not an HQ artifact")

    var ck = ShardedSafeTensors.open(String(CKPT_FP8))

    var lora_stack = _HQLoraStack.load()
    if apply_lora:
        lora_stack.validate()
        lora_stack.print_summary()

    print("  [load] globals")
    var gw = _load_global_weights_dict(ck, ctx)
    if apply_lora:
        var n_global = lora_stack.apply_to_globals(gw, ctx)
        print("  [lora] global deltas applied (one-time additive):", n_global)
    var g = _Globals(
        _clone(gw[String("patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "patchify_proj.bias", ctx),
        _clone(gw[String("audio_patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "audio_patchify_proj.bias", ctx),
        _clone(gw[String("proj_out.weight")][], ctx),
        _load_global_bf16(ck, "proj_out.bias", ctx),
        _clone(gw[String("audio_proj_out.weight")][], ctx),
        _load_global_bf16(ck, "audio_proj_out.bias", ctx),
        _load_global_bf16(ck, "scale_shift_table", ctx),
        _load_global_bf16(ck, "audio_scale_shift_table", ctx),
    )

    print("  [connector] loading + running video/audio (BF16 storage)")
    var v_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    )

    # FULL 1024-token contexts (no tail slice): use the entire dump.
    var dump = ShardedSafeTensors.open(String(AUDIO_CTX_DUMP))
    var video_pre = Tensor.from_view_as_bf16(dump.tensor_view("video_context"), ctx)  # [1,1024,4096]
    var audio_pre = Tensor.from_view_as_bf16(dump.tensor_view("audio_context"), ctx)  # [1,1024,2048]
    _ = _stats(String("video_pre(FULL 1024)"), video_pre, ctx)
    _ = _stats(String("audio_pre(FULL 1024 REAL)"), audio_pre, ctx)

    var enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](v_conn, video_pre, ctx)
    var aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](a_conn, audio_pre, ctx)
    _ = _stats(String("enc"), enc, ctx)
    _ = _stats(String("aenc"), aenc, ctx)
    var nag = NAGContext.disabled(ctx)
    if use_nag:
        print("  [NAG] loading video null context (audio NAG disabled: no audio null dump)")
        var nag_v = _load_video_nag_context(v_conn, ctx)
        _ = _stats(String("nag_video_context"), nag_v, ctx)
        var nag_a = _nag_audio_placeholder(ctx)
        nag = NAGContext.defaults(nag_v^, nag_a^, False)

    print("  [rope] building 3D video + 1D audio + temporal cross tables")
    var vc = _build_video_coords_dims(NF_CT, NH_CT, NW_CT)
    var ac = _build_audio_coords_dims(S_A_CT)
    var vtc = _video_temporal_coords_dims(vc, S_V_CT)
    var vrope = _compute_rope(
        vc, 3, S_V_CT, VD, _mp3(), ROPE_THETA, V_HEADS, STDtype.BF16, ctx
    )
    var arope = _compute_rope(
        ac, 1, S_A_CT, AD, _mp1(), ROPE_THETA, A_HEADS, STDtype.BF16, ctx
    )
    var cavrope = _compute_rope(
        vtc, 1, S_V_CT, CA_DIM, _mp1(), ROPE_THETA, V_HEADS, STDtype.BF16, ctx
    )
    var caarope = _compute_rope(
        ac, 1, S_A_CT, CA_DIM, _mp1(), ROPE_THETA, A_HEADS, STDtype.BF16, ctx
    )
    var v_cos = _clone(vrope[0], ctx); var v_sin = _clone(vrope[1], ctx)
    var a_cos = _clone(arope[0], ctx); var a_sin = _clone(arope[1], ctx)
    var ca_v_cos = _clone(cavrope[0], ctx); var ca_v_sin = _clone(cavrope[1], ctx)
    var ca_a_cos = _clone(caarope[0], ctx); var ca_a_sin = _clone(caarope[1], ctx)

    print("  [noise] init video + audio latents")
    # rng-contract: mojo-native-not-pytorch-parity. Runtime noise is BF16
    # storage; same-seed PyTorch parity requires oracle noise tensors.
    var video_x = randn(_sh5(1, 128, NF_CT, NH_CT, NW_CT), SEED, STDtype.BF16, ctx)
    var audio_x = randn(_sh4(1, AUDIO_C, S_A_CT, AUDIO_MEL), SEED + 1, STDtype.BF16, ctx)
    if max_steps < 0:
        raise Error("max_steps must be >= 0")

    # Keep the block stream scoped to denoise only, so the resident FP8 cache is
    # freed before VAE/vocoder decode allocate their own working sets.
    var run_denoise_scope = max_steps >= 0
    if run_denoise_scope:
        var stream = LTX2BlockStream.open(String(CKPT_FP8))
        if stream.block_count() != NUM_LAYERS:
            raise Error("stream block_count != 48")
        if use_resident:
            var resident_last = RESIDENT_LAST_SINGLE
            comptime if S_V_CT >= S_V_AUDIOSYNC:
                resident_last = RESIDENT_LAST_AUDIOSYNC
            print(
                "  [resident] preloading FP8 blocks", RESIDENT_FIRST, "..",
                resident_last,
            )
            stream.enable_fp8_resident_range(
                RESIDENT_FIRST, resident_last, ctx,
            )
            print("  [resident] resident storage bytes:", stream.resident_bytes())

        # ── HQ res_2s sampler: 15-step token-shifted LTX2Scheduler ──
        var sig_list = _ltx2_scheduler_sigmas(HQ_STEPS, S_V_CT)
        var sched = LTX2Scheduler(sig_list.copy())
        var sigmas = sched.sigmas()
        print("  [sampler] LTX2Scheduler token-shifted sigmas (", len(sigmas), "):", end="")
        for i in range(len(sigmas)):
            print(" ", sigmas[i], end="")
        print("")

        var n_steps = sched.num_steps
        if max_steps > 0 and max_steps < n_steps:
            n_steps = max_steps

        for step in range(n_steps):
            var sigma = sigmas[step]
            var sigma_next = sigmas[step + 1]
            var sigma_next_eff = sigma_next
            if sigma_next == 0.0:
                sigma_next_eff = RES2S_TERMINAL_SIGMA
            print("  --- step", step + 1, "/", sched.num_steps, " sigma=", sigma,
                  " -> ", sigma_next_eff, "---")

            var c = res2s_coefficients(sigma, sigma_next_eff)  # h, a21, b1, b2, sub_sigma

            # STAGE 1 — model @ current sigma. denoised_1 = x - v*sigma.
            var s1 = _model_forward_p[S_V_CT, S_A_CT, S_VPAD_CT, S_APAD_CT](
                ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT,
                LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream,
                video_x, audio_x, enc, aenc,
                v_cos, v_sin, a_cos, a_sin,
                ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
                nag,
                sigma, NF_CT, NH_CT, NW_CT, False, ctx,
            )
            var v_den1 = _denoise_from_vel(video_x, s1[0], sigma, ctx)
            var a_den1 = _denoise_from_vel(audio_x, s1[1], sigma, ctx)

            # midpoint sample: x_mid = x + h*a21*(denoised_1 - x)
            var v_mid = res2s_substep(video_x, v_den1, c.h, c.a21, ctx)
            var a_mid = res2s_substep(audio_x, a_den1, c.h, c.a21, ctx)
            v_mid = res2s_sde_step(
                video_x, v_mid, Float64(sigma), Float64(c.sub_sigma),
                _res2s_hq_noise(v_mid, SEED + UInt64(10000 + step), ctx), ctx,
            )
            a_mid = res2s_sde_step(
                audio_x, a_mid, Float64(sigma), Float64(c.sub_sigma),
                _res2s_hq_noise(a_mid, SEED + UInt64(11000 + step), ctx), ctx,
            )
            var bong_iters = RES2S_BONG_ITERS if res2s_bong_active(c.h, sigma) else 0
            var v_anchor = res2s_bong_refine(video_x, v_mid, v_den1, c.h, c.a21, bong_iters, ctx)
            var a_anchor = res2s_bong_refine(audio_x, a_mid, a_den1, c.h, c.a21, bong_iters, ctx)

            # STAGE 2 — model @ geometric-mean midpoint sigma.
            var s2 = _model_forward_p[S_V_CT, S_A_CT, S_VPAD_CT, S_APAD_CT](
                ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT,
                LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream,
                v_mid, a_mid, enc, aenc,
                v_cos, v_sin, a_cos, a_sin,
                ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
                nag,
                c.sub_sigma, NF_CT, NH_CT, NW_CT, False, ctx,
            )
            # denoised_2 = x_mid - v*sub_sigma, but residual is vs the ANCHOR x:
            #   eps_2 = denoised_2 - x  (res2s_combine subtracts x internally)
            var v_den2 = _denoise_from_vel(v_mid, s2[0], c.sub_sigma, ctx)
            var a_den2 = _denoise_from_vel(a_mid, s2[1], c.sub_sigma, ctx)

            # COMBINE: x_next = x + h*(b1*(den1-x) + b2*(den2-x))
            var v_next = res2s_combine(v_anchor, v_den1, v_den2, c.h, c.b1, c.b2, ctx)
            var a_next = res2s_combine(a_anchor, a_den1, a_den2, c.h, c.b1, c.b2, ctx)
            video_x = res2s_sde_step(
                v_anchor, v_next, Float64(sigma), Float64(sigma_next_eff),
                _res2s_hq_noise(v_next, SEED + UInt64(20000 + step), ctx), ctx,
            )
            audio_x = res2s_sde_step(
                a_anchor, a_next, Float64(sigma), Float64(sigma_next_eff),
                _res2s_hq_noise(a_next, SEED + UInt64(21000 + step), ctx), ctx,
            )
            _ = _stats(String("video_x"), video_x, ctx)
            _ = _stats(String("audio_x"), audio_x, ctx)

        if n_steps == sched.num_steps:
            print("  [sampler] final denoise @ sigma=", RES2S_TERMINAL_SIGMA)
            var fz = _model_forward_p[S_V_CT, S_A_CT, S_VPAD_CT, S_APAD_CT](
                ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT,
                LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream,
                video_x, audio_x, enc, aenc,
                v_cos, v_sin, a_cos, a_sin,
                ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
                nag,
                RES2S_TERMINAL_SIGMA, NF_CT, NH_CT, NW_CT, False, ctx,
            )
            video_x = _denoise_from_vel(video_x, fz[0], RES2S_TERMINAL_SIGMA, ctx)
            audio_x = _denoise_from_vel(audio_x, fz[1], RES2S_TERMINAL_SIGMA, ctx)
            _ = _stats(String("video_x(final denoise)"), video_x, ctx)
            _ = _stats(String("audio_x(final denoise)"), audio_x, ctx)

    print("  [denoise] done -> decoding")

    # ── DECODE VIDEO ──
    print("  [decode] video VAE (latent -> frames)")
    var vae = LTX2VaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var video_x_bf16 = cast_tensor(video_x, STDtype.BF16, ctx)
    var frames = decode_video[1, 128, NF_CT, NH_CT, NW_CT](vae, video_x_bf16, ctx)
    var fsh = frames.shape()
    var n_frames_out = fsh[2]
    print("  frames NCDHW: [", fsh[0], ",", fsh[1], ",", fsh[2], ",", fsh[3],
          ",", fsh[4], "]")
    _ = _stats(String("frames"), frames, ctx)

    var png_paths = List[String]()
    for fr in range(n_frames_out):
        var fslice = slice(frames, 2, fr, 1, ctx)
        var fs = fslice.shape()
        var chw = reshape(fslice, _sh4(fs[0], fs[1], fs[3], fs[4]), ctx)
        var p = out_dir + "/" + frame_prefix + _pad2(fr) + ".png"
        save_png(chw, p, ctx, ValueRange.SIGNED)
        png_paths.append(p)
    print("  saved", n_frames_out, "frame PNGs ->", out_dir)

    if not include_audio:
        var mp4_video = out_dir + "/ltx2_t2v_hq.mp4"
        if max_steps != 0:
            mp4_video = out_dir + "/ltx2_t2v_dev_smoke.mp4"
        print("  [decode] audio disabled by explicit noaudio; muxing video-only mp4")
        _mux_video_mp4(out_dir, mp4_video, frame_prefix)
        print("=== HQ VIDEO DONE ===")
        print("  mp4:", mp4_video)
        print("  frames:", n_frames_out)
        return

    var wav_out = out_dir + "/hq_audio.wav"
    var mp4_out = out_dir + "/ltx2_t2v_av_hq.mp4"
    if max_steps != 0:
        wav_out = out_dir + "/dev_audio.wav"
        mp4_out = out_dir + "/ltx2_t2v_av_dev_smoke.mp4"
        print("  [quality] DEV SMOKE ONLY: audio mux is tested, output is not HQ quality")
    print("  [decode] audio VAE (latent -> mel) -> vocoder (mel -> 48kHz wav)")
    var avae = LTX2AudioVaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var audio_x_bf16 = cast_tensor(audio_x, STDtype.BF16, ctx)
    var mel_raw = decode_audio(avae, audio_x_bf16, ctx)
    var mel = cast_tensor(mel_raw, STDtype.BF16, ctx)
    var msh = mel.shape()
    print("  mel NCHW: [", msh[0], ",", msh[1], ",", msh[2], ",", msh[3], "] BF16 storage")
    _ = _stats(String("mel"), mel, ctx)

    var voc = LTX2VocoderWithBWE.from_file(String(CKPT_BF16), ctx)
    var wav = voc.forward(mel, ctx)
    var wsh = wav.shape()
    var L = wsh[2]
    print("  wav [", wsh[0], ",", wsh[1], ",", wsh[2], "] @", voc.output_sample_rate(), "Hz")

    var wh = wav.to_host(ctx)
    var rms = 0.0
    var amax = 0.0
    var bad = False
    for i in range(len(wh)):
        var v = Float64(wh[i])
        if not (v == v): bad = True
        rms += v * v
        var av = v if v >= 0.0 else -v
        if av > amax: amax = av
    rms = sqrt(rms / Float64(len(wh)))
    print("  AUDIO rms=", Float32(rms), " absmax=", Float32(amax), " finite=",
          not bad, " nonsilent=", rms > 1.0e-4)

    var inter = List[Float32]()
    inter.resize(L * 2, Float32(0.0))
    for ch in range(2):
        for s in range(L):
            inter[s * 2 + ch] = wh[ch * L + s]
    _write_wav(wav_out, inter, voc.output_sample_rate())
    print("  wrote wav:", wav_out)

    print("  [mux] ffmpeg frames + wav -> mp4")
    _mux_mp4(out_dir, n_frames_out, wav_out, mp4_out, frame_prefix)
    print("=== HQ DONE ===")
    print("  mp4:", mp4_out)
    print("  wav:", wav_out)
    print("  frames:", out_dir, "/", frame_prefix, "00.png ..")


def run(
    apply_lora: Bool, out_dir: String, max_steps: Int, use_resident: Bool,
    include_audio: Bool, use_nag: Bool,
) raises:
    run_single_p[NUM_FRAMES, NF, NH, NW, S_V, S_A, S_VPAD, S_APAD](
        apply_lora, out_dir, max_steps, use_resident, include_audio, use_nag
    )


def run_long(
    apply_lora: Bool, out_dir: String, max_steps: Int, use_resident: Bool,
    include_audio: Bool, use_nag: Bool,
) raises:
    run_single_p[
        NUM_FRAMES_LONG, NF_LONG, NH, NW, S_V_LONG, S_A_LONG, S_VPAD_LONG,
        S_APAD_LONG,
    ](apply_lora, out_dir, max_steps, use_resident, include_audio, use_nag)


def run_audiosync(
    apply_lora: Bool, out_dir: String, max_steps: Int, use_resident: Bool,
    include_audio: Bool, use_nag: Bool,
) raises:
    run_single_p[
        NUM_FRAMES_AUDIOSYNC, NF_AUDIOSYNC, NH, NW, S_V_AUDIOSYNC,
        S_A_AUDIOSYNC, S_VPAD_AUDIOSYNC, S_APAD_AUDIOSYNC,
    ](apply_lora, out_dir, max_steps, use_resident, include_audio, use_nag)


# ════════════════════════════════════════════════════════════════════════════
# FULL STAGED HQ generate() — faithful to pipeline.py:1139-1337.
#   Stage 1 : LTX2Scheduler 15-step token-shifted sigmas + res_2s denoise.
#   UPSCALE : un_normalize -> LatentUpsampler(spatial x2) -> normalize.
#   Stage 2 : forward-noise upscaled @ s2_sigmas[0]=0.909375, then 3-step res_2s
#             refine (STAGE_2_DISTILLED_SIGMA_VALUES) at the 2x latent grid.
#   decode video (2x res) + audio VAE -> vocoder, mux to mp4.
#
# NAG: pipeline.py applies NAGPatch (null-Gemma-encoding baseline) around BOTH
# stages. serenitymojo HAS the NAG combine + ltx2_block_forward_av_nag wired and
# gated, but the NULL-context input is the Gemma encoding of the empty string ""
# (nag.py / pipeline.py:466) — and NO such null-encoding dump exists on disk
# (only the positive video/audio context dump). NAG is therefore reported as
# blocked-on-input here; the dominant sharpness levers (the previously-MISSING
# spatial upscale + stage-2 refine) ARE wired and run. Audio is single-stage
# (refined only by the video-coupled stage-2; the reference carries s1 audio
# into stage 2 unchanged when no audio refine).
# ════════════════════════════════════════════════════════════════════════════
def run_staged(
    apply_lora: Bool, out_dir: String, max_steps: Int, use_resident: Bool,
    include_audio: Bool, use_nag: Bool, profile_dit: Bool,
) raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2.3 T2V HQ *FULL STAGED* (res_2s + 2x upscale + refine) ===")
    print("  target frames:", NUM_FRAMES_STAGED, " decoded frames:",
          1 + (NF_STAGED - 1) * 8)
    print("  Stage1 res:", OUT_W_STAGE1, "x", OUT_H_STAGE1, " NF/NH/NW:",
          NF_STAGED, NH_STAGED, NW_STAGED, " S_V:", S_V_STAGED,
          " S_A:", S_A_STAGED)
    print("  Stage2 res:", OUT_W2_STAGED, "x", OUT_H2_STAGED, " NF/NH2/NW2:",
          NF_STAGED, NH2_STAGED, NW2_STAGED, " S_V2:", S_V2_STAGED)
    print("  sampler: res_2s  Stage1 steps:", HQ_STEPS,
          "  Stage2 steps: 3 (STAGE_2_DISTILLED_SIGMA_VALUES)")
    print("  LoRA:", "HQ stack" if apply_lora else "OFF", " out_dir:", out_dir)
    print("  weights:", "FP8-resident warm range" if use_resident else "FP8 stream")
    print("  audio:", "ON (A/V default)" if include_audio else "OFF (explicit noaudio)")
    print("  NAG:", "video-only ON" if use_nag else "OFF")
    print("  profile:", "DiT phase timers ON" if profile_dit else "OFF")
    var frame_prefix = String("hq_frame")
    if max_steps != 0:
        frame_prefix = String("dev_frame")
        print("  [quality] DEV SMOKE ONLY: incomplete staged denoise, not an HQ artifact")

    var ck = ShardedSafeTensors.open(String(CKPT_FP8))

    var lora_stack = _HQLoraStack.load()
    if apply_lora:
        lora_stack.validate()
        print("  [lora] Stage1 scales")
        lora_stack.print_summary_scaled(
            LORA_DISTILLED_MULT_HQ_STAGE1, LORA_CAMERA_STATIC_MULT,
            LORA_DETAILER_MULT,
        )
        print("  [lora] Stage2 scales")
        lora_stack.print_summary_scaled(
            LORA_DISTILLED_MULT_HQ_STAGE2, LORA_CAMERA_STATIC_MULT,
            LORA_DETAILER_MULT,
        )

    print("  [load] stage-1 globals")
    var gw = _load_global_weights_dict(ck, ctx)
    if apply_lora:
        var n_global = lora_stack.apply_to_globals_scaled(
            gw, LORA_DISTILLED_MULT_HQ_STAGE1, LORA_CAMERA_STATIC_MULT,
            LORA_DETAILER_MULT, ctx,
        )
        print("  [lora] stage-1 global deltas applied:", n_global)
    var g = _Globals(
        _clone(gw[String("patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "patchify_proj.bias", ctx),
        _clone(gw[String("audio_patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "audio_patchify_proj.bias", ctx),
        _clone(gw[String("proj_out.weight")][], ctx),
        _load_global_bf16(ck, "proj_out.bias", ctx),
        _clone(gw[String("audio_proj_out.weight")][], ctx),
        _load_global_bf16(ck, "audio_proj_out.bias", ctx),
        _load_global_bf16(ck, "scale_shift_table", ctx),
        _load_global_bf16(ck, "audio_scale_shift_table", ctx),
    )

    print("  [connector] loading + running video/audio (BF16 storage)")
    var v_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    )
    var dump = ShardedSafeTensors.open(String(AUDIO_CTX_DUMP))
    var video_pre = Tensor.from_view_as_bf16(dump.tensor_view("video_context"), ctx)
    var audio_pre = Tensor.from_view_as_bf16(dump.tensor_view("audio_context"), ctx)
    var enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](v_conn, video_pre, ctx)
    var aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](a_conn, audio_pre, ctx)
    var nag = NAGContext.disabled(ctx)
    if use_nag:
        print("  [NAG] loading video null context (audio NAG disabled: no audio null dump)")
        var nag_v = _load_video_nag_context(v_conn, ctx)
        _ = _stats(String("nag_video_context"), nag_v, ctx)
        var nag_a = _nag_audio_placeholder(ctx)
        nag = NAGContext.defaults(nag_v^, nag_a^, False)

    var stream = LTX2BlockStream.open(String(CKPT_FP8))
    if stream.block_count() != NUM_LAYERS:
        raise Error("stream block_count != 48")
    if use_resident:
        print(
            "  [resident] preloading FP8 blocks", RESIDENT_FIRST, "..",
            RESIDENT_LAST_STAGED,
        )
        stream.enable_fp8_resident_range(RESIDENT_FIRST, RESIDENT_LAST_STAGED, ctx)
        print("  [resident] resident storage bytes:", stream.resident_bytes())

    # ── Audio RoPE (sigma-independent, same both stages) ──
    var ac = _build_audio_coords_dims(S_A_STAGED)
    var arope = _compute_rope(
        ac, 1, S_A_STAGED, AD, _mp1(), ROPE_THETA, A_HEADS, STDtype.BF16, ctx
    )
    var caarope = _compute_rope(
        ac, 1, S_A_STAGED, CA_DIM, _mp1(), ROPE_THETA, A_HEADS, STDtype.BF16, ctx
    )
    var a_cos = _clone(arope[0], ctx); var a_sin = _clone(arope[1], ctx)
    var ca_a_cos = _clone(caarope[0], ctx); var ca_a_sin = _clone(caarope[1], ctx)

    # ── STAGE-1 RoPE (NF_STAGED,NH_STAGED,NW_STAGED) ──
    print("  [rope] stage-1 video tables (S_V=", S_V_STAGED, ")")
    var vc1 = _build_video_coords_dims(NF_STAGED, NH_STAGED, NW_STAGED)
    var vtc1 = _video_temporal_coords_dims(vc1, S_V_STAGED)
    var vrope1 = _compute_rope(
        vc1, 3, S_V_STAGED, VD, _mp3(), ROPE_THETA, V_HEADS, STDtype.BF16, ctx
    )
    var cavrope1 = _compute_rope(
        vtc1, 1, S_V_STAGED, CA_DIM, _mp1(), ROPE_THETA, V_HEADS, STDtype.BF16, ctx
    )
    var v_cos1 = _clone(vrope1[0], ctx); var v_sin1 = _clone(vrope1[1], ctx)
    var ca_v_cos1 = _clone(cavrope1[0], ctx); var ca_v_sin1 = _clone(cavrope1[1], ctx)

    print("  [noise] init video + audio latents")
    # rng-contract: mojo-native-not-pytorch-parity. Runtime noise is BF16
    # storage; same-seed PyTorch parity requires oracle noise tensors.
    var video_x = randn(
        _sh5(1, 128, NF_STAGED, NH_STAGED, NW_STAGED), SEED, STDtype.BF16, ctx
    )
    var audio_x = randn(
        _sh4(1, AUDIO_C, S_A_STAGED, AUDIO_MEL), SEED + 1, STDtype.BF16, ctx
    )

    var sig1 = _ltx2_scheduler_sigmas(HQ_STEPS, S_V_STAGED)
    var sched1 = LTX2Scheduler(sig1.copy())
    var sigmas1 = sched1.sigmas()
    var n1 = sched1.num_steps
    if max_steps > 0 and max_steps < n1:
        n1 = max_steps
    print("  [Stage1] res_2s denoise,", n1, "steps")

    for step in range(n1):
        var sigma = sigmas1[step]
        var sigma_next = sigmas1[step + 1]
        var sigma_next_eff = sigma_next
        if sigma_next == 0.0:
            sigma_next_eff = RES2S_TERMINAL_SIGMA
        print("  S1 step", step + 1, "/", sched1.num_steps, " sigma=", sigma, "->", sigma_next_eff)
        var c = res2s_coefficients(sigma, sigma_next_eff)
        var s1a = _model_forward_p[
            S_V_STAGED, S_A_STAGED, S_VPAD_STAGED, S_APAD_STAGED
        ](
            ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT_HQ_STAGE1,
            LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream,
            video_x, audio_x, enc, aenc,
            v_cos1, v_sin1, a_cos, a_sin, ca_v_cos1, ca_v_sin1, ca_a_cos, ca_a_sin,
            nag,
            sigma, NF_STAGED, NH_STAGED, NW_STAGED, profile_dit, ctx,
        )
        var v_den1 = _denoise_from_vel(video_x, s1a[0], sigma, ctx)
        var a_den1 = _denoise_from_vel(audio_x, s1a[1], sigma, ctx)
        var v_mid = res2s_substep(video_x, v_den1, c.h, c.a21, ctx)
        var a_mid = res2s_substep(audio_x, a_den1, c.h, c.a21, ctx)
        v_mid = res2s_sde_step(
            video_x, v_mid, Float64(sigma), Float64(c.sub_sigma),
            _res2s_hq_noise(v_mid, SEED + UInt64(30000 + step), ctx), ctx,
        )
        a_mid = res2s_sde_step(
            audio_x, a_mid, Float64(sigma), Float64(c.sub_sigma),
            _res2s_hq_noise(a_mid, SEED + UInt64(31000 + step), ctx), ctx,
        )
        var bong_iters = RES2S_BONG_ITERS if res2s_bong_active(c.h, sigma) else 0
        var v_anchor = res2s_bong_refine(video_x, v_mid, v_den1, c.h, c.a21, bong_iters, ctx)
        var a_anchor = res2s_bong_refine(audio_x, a_mid, a_den1, c.h, c.a21, bong_iters, ctx)
        var s2a = _model_forward_p[
            S_V_STAGED, S_A_STAGED, S_VPAD_STAGED, S_APAD_STAGED
        ](
            ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT_HQ_STAGE1,
            LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream,
            v_mid, a_mid, enc, aenc,
            v_cos1, v_sin1, a_cos, a_sin, ca_v_cos1, ca_v_sin1, ca_a_cos, ca_a_sin,
            nag,
            c.sub_sigma, NF_STAGED, NH_STAGED, NW_STAGED, profile_dit, ctx,
        )
        var v_den2 = _denoise_from_vel(v_mid, s2a[0], c.sub_sigma, ctx)
        var a_den2 = _denoise_from_vel(a_mid, s2a[1], c.sub_sigma, ctx)
        var v_next = res2s_combine(v_anchor, v_den1, v_den2, c.h, c.b1, c.b2, ctx)
        var a_next = res2s_combine(a_anchor, a_den1, a_den2, c.h, c.b1, c.b2, ctx)
        video_x = res2s_sde_step(
            v_anchor, v_next, Float64(sigma), Float64(sigma_next_eff),
            _res2s_hq_noise(v_next, SEED + UInt64(40000 + step), ctx), ctx,
        )
        audio_x = res2s_sde_step(
            a_anchor, a_next, Float64(sigma), Float64(sigma_next_eff),
            _res2s_hq_noise(a_next, SEED + UInt64(41000 + step), ctx), ctx,
        )
    if n1 == sched1.num_steps:
        print("  [Stage1] final denoise @ sigma=", RES2S_TERMINAL_SIGMA)
        var fz1 = _model_forward_p[
            S_V_STAGED, S_A_STAGED, S_VPAD_STAGED, S_APAD_STAGED
        ](
            ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT_HQ_STAGE1,
            LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream,
            video_x, audio_x, enc, aenc,
            v_cos1, v_sin1, a_cos, a_sin, ca_v_cos1, ca_v_sin1, ca_a_cos, ca_a_sin,
            nag,
            RES2S_TERMINAL_SIGMA, NF_STAGED, NH_STAGED, NW_STAGED, profile_dit, ctx,
        )
        video_x = _denoise_from_vel(video_x, fz1[0], RES2S_TERMINAL_SIGMA, ctx)
        audio_x = _denoise_from_vel(audio_x, fz1[1], RES2S_TERMINAL_SIGMA, ctx)
    _ = _stats(String("video_x(stage1)"), video_x, ctx)
    print("  [Stage1] done")

    # ══════════════ SPATIAL UPSAMPLE (x2) ══════════════
    print("  [upscale] loading spatial-x2 LatentUpsampler + VAE per-channel stats")
    var up_st = ShardedSafeTensors.open(String(SPATIAL_UPSCALER))
    var upsampler = LatentUpsampler(up_st, False, ctx)  # is_temporal=False
    var std_t = _load_vae_stat(String("std-of-means"), ctx)
    var mean_t = _load_vae_stat(String("mean-of-means"), ctx)
    # NCDHW [1,128,NF,NH,NW] -> NDHWC, upsample (un_normalize/normalize wrapped),
    # back to NCDHW [1,128,NF,2NH,2NW].
    var v_ndhwc = _ncdhw_to_ndhwc(video_x, ctx)
    var up_ndhwc = upsample_video(v_ndhwc, std_t, mean_t, upsampler, ctx)
    var upscaled = _ndhwc_to_ncdhw(up_ndhwc, ctx)
    var ush = upscaled.shape()
    print("  [upscale] upscaled latent NCDHW: [", ush[0], ",", ush[1], ",",
          ush[2], ",", ush[3], ",", ush[4], "]  (expect [1,128,",
          NF_STAGED, ",", NH2_STAGED, ",", NW2_STAGED, "])")
    _ = _stats(String("upscaled"), upscaled, ctx)

    # ══════════════ STAGE-2 RoPE (doubled spatial grid) ══════════════
    print("  [rope] stage-2 video tables (S_V2=", S_V2_STAGED, ")")
    var vc2 = _build_video_coords_dims(NF_STAGED, NH2_STAGED, NW2_STAGED)
    var vtc2 = _video_temporal_coords_dims(vc2, S_V2_STAGED)
    var vrope2 = _compute_rope(
        vc2, 3, S_V2_STAGED, VD, _mp3(), ROPE_THETA, V_HEADS, STDtype.BF16, ctx
    )
    var cavrope2 = _compute_rope(
        vtc2, 1, S_V2_STAGED, CA_DIM, _mp1(), ROPE_THETA, V_HEADS, STDtype.BF16, ctx
    )
    var v_cos2 = _clone(vrope2[0], ctx); var v_sin2 = _clone(vrope2[1], ctx)
    var ca_v_cos2 = _clone(cavrope2[0], ctx); var ca_v_sin2 = _clone(cavrope2[1], ctx)

    print("  [load] stage-2 globals")
    gw = _load_global_weights_dict(ck, ctx)
    if apply_lora:
        var n_global2 = lora_stack.apply_to_globals_scaled(
            gw, LORA_DISTILLED_MULT_HQ_STAGE2, LORA_CAMERA_STATIC_MULT,
            LORA_DETAILER_MULT, ctx,
        )
        print("  [lora] stage-2 global deltas applied:", n_global2)
    g = _Globals(
        _clone(gw[String("patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "patchify_proj.bias", ctx),
        _clone(gw[String("audio_patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "audio_patchify_proj.bias", ctx),
        _clone(gw[String("proj_out.weight")][], ctx),
        _load_global_bf16(ck, "proj_out.bias", ctx),
        _clone(gw[String("audio_proj_out.weight")][], ctx),
        _load_global_bf16(ck, "audio_proj_out.bias", ctx),
        _load_global_bf16(ck, "scale_shift_table", ctx),
        _load_global_bf16(ck, "audio_scale_shift_table", ctx),
    )

    # ══════════════ STAGE-2 refine (3-step res_2s) ══════════════
    var s2sig = ltx2_stage2_distilled_sigmas()  # [0.909375, 0.725, 0.421875, 0.0]
    var s2_noise_scale = s2sig[0]
    print("  [Stage2] forward-noise upscaled @", s2_noise_scale, " then 3-step res_2s refine (joint A/V)")
    # Forward-noise BOTH video (upscaled) and audio (stage-1 result) at noise_scale.
    # GaussianNoiser, denoise_mask=1: x = noise*scale + init*(1-scale). (Reference
    # ti2vid_two_stages_hq.py passes both initial_video_latent + initial_audio_latent.)
    var vx2 = _noise_init(upscaled, s2_noise_scale, SEED + 100, ctx)
    var ax2 = _noise_init(audio_x, s2_noise_scale, SEED + 101, ctx)
    var sched2 = LTX2Scheduler(s2sig.copy())
    var sigmas2 = sched2.sigmas()
    var n2 = sched2.num_steps
    if max_steps > 0 and max_steps < n2:
        n2 = max_steps
        print("  [Stage2] fast gate clamp:", n2, "/", sched2.num_steps, "steps")

    for step in range(n2):
        var sigma = sigmas2[step]
        var sigma_next = sigmas2[step + 1]
        var sigma_next_eff = sigma_next
        if sigma_next == 0.0:
            sigma_next_eff = RES2S_TERMINAL_SIGMA
        print("  S2 step", step + 1, "/", n2, " sigma=", sigma, "->", sigma_next_eff)
        var c = res2s_coefficients(sigma, sigma_next_eff)
        var s1a = _model_forward_p[
            S_V2_STAGED, S_A_STAGED, S_VPAD2_STAGED, S_APAD_STAGED
        ](
            ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT_HQ_STAGE2,
            LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream,
            vx2, ax2, enc, aenc,
            v_cos2, v_sin2, a_cos, a_sin, ca_v_cos2, ca_v_sin2, ca_a_cos, ca_a_sin,
            nag,
            sigma, NF_STAGED, NH2_STAGED, NW2_STAGED, profile_dit, ctx,
        )
        var v_den1 = _denoise_from_vel(vx2, s1a[0], sigma, ctx)
        var a_den1 = _denoise_from_vel(ax2, s1a[1], sigma, ctx)
        var v_mid = res2s_substep(vx2, v_den1, c.h, c.a21, ctx)
        var a_mid = res2s_substep(ax2, a_den1, c.h, c.a21, ctx)
        v_mid = res2s_sde_step(
            vx2, v_mid, Float64(sigma), Float64(c.sub_sigma),
            _res2s_hq_noise(v_mid, SEED + UInt64(50000 + step), ctx), ctx,
        )
        a_mid = res2s_sde_step(
            ax2, a_mid, Float64(sigma), Float64(c.sub_sigma),
            _res2s_hq_noise(a_mid, SEED + UInt64(51000 + step), ctx), ctx,
        )
        var bong_iters = RES2S_BONG_ITERS if res2s_bong_active(c.h, sigma) else 0
        var v_anchor = res2s_bong_refine(vx2, v_mid, v_den1, c.h, c.a21, bong_iters, ctx)
        var a_anchor = res2s_bong_refine(ax2, a_mid, a_den1, c.h, c.a21, bong_iters, ctx)
        var s2a = _model_forward_p[
            S_V2_STAGED, S_A_STAGED, S_VPAD2_STAGED, S_APAD_STAGED
        ](
            ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT_HQ_STAGE2,
            LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream,
            v_mid, a_mid, enc, aenc,
            v_cos2, v_sin2, a_cos, a_sin, ca_v_cos2, ca_v_sin2, ca_a_cos, ca_a_sin,
            nag,
            c.sub_sigma, NF_STAGED, NH2_STAGED, NW2_STAGED, profile_dit, ctx,
        )
        var v_den2 = _denoise_from_vel(v_mid, s2a[0], c.sub_sigma, ctx)
        var a_den2 = _denoise_from_vel(a_mid, s2a[1], c.sub_sigma, ctx)
        var v_next = res2s_combine(v_anchor, v_den1, v_den2, c.h, c.b1, c.b2, ctx)
        var a_next = res2s_combine(a_anchor, a_den1, a_den2, c.h, c.b1, c.b2, ctx)
        vx2 = res2s_sde_step(
            v_anchor, v_next, Float64(sigma), Float64(sigma_next_eff),
            _res2s_hq_noise(v_next, SEED + UInt64(60000 + step), ctx), ctx,
        )
        ax2 = res2s_sde_step(
            a_anchor, a_next, Float64(sigma), Float64(sigma_next_eff),
            _res2s_hq_noise(a_next, SEED + UInt64(61000 + step), ctx), ctx,
        )
    if n2 == sched2.num_steps:
        print("  [Stage2] final denoise @ sigma=", RES2S_TERMINAL_SIGMA)
        var fz2 = _model_forward_p[
            S_V2_STAGED, S_A_STAGED, S_VPAD2_STAGED, S_APAD_STAGED
        ](
            ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT_HQ_STAGE2,
            LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream,
            vx2, ax2, enc, aenc,
            v_cos2, v_sin2, a_cos, a_sin, ca_v_cos2, ca_v_sin2, ca_a_cos, ca_a_sin,
            nag,
            RES2S_TERMINAL_SIGMA, NF_STAGED, NH2_STAGED, NW2_STAGED, profile_dit, ctx,
        )
        vx2 = _denoise_from_vel(vx2, fz2[0], RES2S_TERMINAL_SIGMA, ctx)
        ax2 = _denoise_from_vel(ax2, fz2[1], RES2S_TERMINAL_SIGMA, ctx)
    _ = _stats(String("vx2(stage2)"), vx2, ctx)
    print("  [Stage2] done -> decoding at 2x resolution")

    # ══════════════ DECODE VIDEO (2x res) ══════════════
    var vae = LTX2VaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var vx2_bf16 = cast_tensor(vx2, STDtype.BF16, ctx)
    var frames = decode_video[
        1, 128, NF_STAGED, NH2_STAGED, NW2_STAGED
    ](vae, vx2_bf16, ctx)
    var fsh = frames.shape()
    var n_frames_out = fsh[2]
    print("  frames NCDHW: [", fsh[0], ",", fsh[1], ",", fsh[2], ",", fsh[3],
          ",", fsh[4], "]")
    _ = _stats(String("frames"), frames, ctx)

    for fr in range(n_frames_out):
        var fslice = slice(frames, 2, fr, 1, ctx)
        var fs = fslice.shape()
        var chw = reshape(fslice, _sh4(fs[0], fs[1], fs[3], fs[4]), ctx)
        save_png(chw, out_dir + "/" + frame_prefix + _pad2(fr) + ".png", ctx, ValueRange.SIGNED)
    print("  saved", n_frames_out, "frame PNGs ->", out_dir)

    if not include_audio:
        var mp4_video = out_dir + "/ltx2_t2v_hq2.mp4"
        if max_steps != 0:
            mp4_video = out_dir + "/ltx2_t2v_stage2_dev_smoke.mp4"
        print("  [decode] audio disabled by explicit noaudio; muxing video-only mp4")
        _mux_video_mp4(out_dir, mp4_video, frame_prefix)
        print("=== HQ STAGED VIDEO DONE ===")
        print("  mp4:", mp4_video)
        print("  frames:", n_frames_out)
        return

    # ══════════════ AUDIO + MUX ══════════════
    var wav_out = out_dir + "/hq_audio.wav"
    var mp4_out = out_dir + "/ltx2_t2v_av_hq2.mp4"
    if max_steps != 0:
        wav_out = out_dir + "/dev_audio.wav"
        mp4_out = out_dir + "/ltx2_t2v_av_stage2_dev_smoke.mp4"
        print("  [quality] DEV SMOKE ONLY: audio mux is tested, output is not HQ quality")
    print("  [decode] audio VAE -> vocoder")
    var avae = LTX2AudioVaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var audio_x_bf16 = cast_tensor(ax2, STDtype.BF16, ctx)
    var mel = cast_tensor(decode_audio(avae, audio_x_bf16, ctx), STDtype.BF16, ctx)
    var voc = LTX2VocoderWithBWE.from_file(String(CKPT_BF16), ctx)
    var wav = voc.forward(mel, ctx)
    var wsh = wav.shape()
    var L = wsh[2]
    var wh = wav.to_host(ctx)
    var inter = List[Float32]()
    inter.resize(L * 2, Float32(0.0))
    for ch in range(2):
        for s in range(L):
            inter[s * 2 + ch] = wh[ch * L + s]
    _write_wav(wav_out, inter, voc.output_sample_rate())
    print("  wrote wav:", wav_out)
    _mux_mp4(out_dir, n_frames_out, wav_out, mp4_out, frame_prefix)
    print("=== HQ STAGED DONE ===")
    print("  mp4:", mp4_out)
    print("  frames:", out_dir, "/", frame_prefix, "00.png ..  (",
          OUT_W2_STAGED, "x", OUT_H2_STAGED, ")")


# ════════════════════════════════════════════════════════════════════════════
# `refhq` MODE — the REAL LTX-2 HQ recipe, gated per-step against the reference
# (scripts/ltx2_hq_ref_run.py + scripts/ltx2_hq_step_compare.py).
#
# Faithful to ltx_pipelines/ti2vid_two_stages_hq.py via the instrumented
# reference run:
#   Stage 1 : GUIDED res_2s (res2s_audio_video_denoising_loop, samplers.py:199-433)
#             at HALF spatial resolution. Per eval EXACTLY 3 sequential
#             forwards (denoisers.py `_guided_denoise`, HQ params => passes
#             cond / uncond / mod): cond (pos ctx), uncond (NEGATIVE ctx), mod
#             (pos ctx with SKIP_A2V_CROSS_ATTN + SKIP_V2A_CROSS_ATTN ==
#             skip_cross_modal=True). The transformer is an X0Model — each
#             pass's velocity is converted to x0 (to_denoised: x - v*sigma,
#             F32 calc) BEFORE the MultiModalGuider combine.
#             Guider params (constants.py LTX_2_3_HQ_PARAMS):
#               video cfg=3.0 stg=0 rescale=0.45 mod=3.0
#               audio cfg=7.0 stg=0 rescale=1.0  mod=3.0
#   UPSCALE : un_normalize -> spatial-x2 LatentUpsampler -> normalize.
#   Stage 2 : SIMPLE res_2s (no guidance), STAGE_2 sigmas
#             [0.909375, 0.725, 0.421875, 0.0], at the full grid; both video
#             (upsampled) and audio (stage-1 result) forward-noised at
#             noise_scale = 0.909375.
#   LoRA    : distilled-LoRA ONLY (ref passes a single LoraPathStrengthAndSDOps),
#             strength 0.25 stage 1 / 0.5 stage 2, FACTORIZED runtime add
#             (never fused). Globals get the additive delta at the same
#             per-stage strength.
#
# PARITY-GATE SHAPE (compile-time; matches the oracle run `--width 384
# --height 256 --num-frames 17 --steps 15 --seed 42` with fps 25):
#   stage 1: 192x128  -> NF=3 NH=4 NW=6  S_V=72 ; audio S_A=round(17/25*25)=17
#   stage 2: 384x256  -> NH=8 NW=12      S_V=288
#
# MEMORY DEVIATION (documented): the plan called for the MVP all-48-resident
# fp8 preload, but this GPU has 24 GB and resident blocks (20.5 GB) + the
# rank-384 distilled LoRA factor cache (7.6 GB) cannot coexist. refhq therefore
# uses the PROVEN streamed pattern from run_staged: raw-fp8 resident RANGE
# (stream.enable_fp8_resident_range) + per-eval dequant to BF16 + per-eval
# factorized LoRA attach from the PRELOADED factor cache
# (LoraSet.preload_ltx2_block_factors -> attach_ltx2_cached_block_factors ->
# add_lora_factor_arc). The factorized path is verified to engage by the
# printed per-eval attach count.
#
# Dump contract (parity inputs for scripts/ltx2_hq_step_compare.py):
#   stage1_steps.safetensors : s1_s{NN}_video / s1_s{NN}_audio  (patchified,
#       state AFTER the step-level SDE injection, BF16-rounded, stored F32)
#   stage2_steps.safetensors : s2_s{NN}_*
#   upsampler.safetensors    : {in, out}   (latent NCDHW layout, F32)
#   final_latents.safetensors: {video, audio} (unpatchified latent layout, F32)
#
# Noise/init fixture (written by scripts/ltx2_hq_ref_run.py): keys
#   init_video/init_audio (patchified init latents), s2init_video/s2init_audio
#   (raw stage-2 GaussianNoiser draws), s{1,2}_sub{NN}_{video,audio} and
#   s{1,2}_stp{NN}_{video,audio} (SDE noises, post-_get_new_noise
#   normalization, in consumption order). Pass "-" for production Mojo randn.
# ════════════════════════════════════════════════════════════════════════════
# Production target 1024x576 @ 121 frames (stage-1 = half res per the reference
# two-stage recipe). MEASURED 2026-06-10: at 768x512 the OFFICIAL pipeline
# produces the same soft-video/static-audio class as Mojo (audio rms 0.033 vs
# 0.039) — the gap to the quality bar is the recipe OPERATING POINT (the HQ
# recipe is designed for 1920x1088 final), not Mojo fidelity. Stepping up.
# The 384x256/17f parity shapes this mode was gated at are recoverable by
# reverting this block (parity dumps preserved under output/ltx2_refhq_parity/).
comptime REFHQ_NUM_FRAMES = 121
comptime REFHQ_FPS = Float64(25.0)
comptime REFHQ_NF = 16           # (121-1)//8 + 1
comptime REFHQ_NH1 = 9           # 288/32 (stage-1 height 576/2)
comptime REFHQ_NW1 = 16          # 512/32 (stage-1 width 1024/2)
comptime REFHQ_S_V1 = REFHQ_NF * REFHQ_NH1 * REFHQ_NW1   # 2304
comptime REFHQ_S_A = 121         # round(121/25*25)
comptime REFHQ_NH2 = 18          # 576/32
comptime REFHQ_NW2 = 32          # 1024/32
comptime REFHQ_S_V2 = REFHQ_NF * REFHQ_NH2 * REFHQ_NW2   # 9216
comptime REFHQ_SPAD = 9216       # max(S_V*, N_TXT=1024, S_A)
comptime REFHQ_STEPS = 15
# HQ guider params (ltx_pipelines/utils/constants.py LTX_2_3_HQ_PARAMS)
comptime REFHQ_V_CFG = Float32(3.0)
comptime REFHQ_V_RESCALE = Float32(0.45)
comptime REFHQ_A_CFG = Float32(7.0)
comptime REFHQ_A_RESCALE = Float32(1.0)
comptime REFHQ_MOD_SCALE = Float32(3.0)
comptime REFHQ_LORA_S1 = Float32(0.25)
comptime REFHQ_LORA_S2 = Float32(0.5)


# fps-aware video RoPE coord boxes (refhq runs the reference fps=25 profile;
# the legacy paths keep the comptime FRAME_RATE=24 builder).
def _build_video_coords_fps(
    nf: Int, nh: Int, nw: Int, fps: Float64
) -> List[Float64]:
    var s_v = nf * nh * nw
    var out = List[Float64]()
    out.resize(3 * s_v * 2, Float64(0.0))
    var vae_t = VAE_SF0
    for f in range(nf):
        for h in range(nh):
            for w in range(nw):
                var tok = f * nh * nw + h * nw + w
                var fs = Float64(f) * VAE_SF0
                var fe = Float64(f + 1) * VAE_SF0
                var fsc = fs + CAUSAL_OFFSET - vae_t
                var fec = fe + CAUSAL_OFFSET - vae_t
                if fsc < 0.0: fsc = 0.0
                if fec < 0.0: fec = 0.0
                fsc = fsc / fps
                fec = fec / fps
                out[(0 * s_v + tok) * 2 + 0] = fsc
                out[(0 * s_v + tok) * 2 + 1] = fec
                out[(1 * s_v + tok) * 2 + 0] = Float64(h) * VAE_SF1
                out[(1 * s_v + tok) * 2 + 1] = Float64(h + 1) * VAE_SF1
                out[(2 * s_v + tok) * 2 + 0] = Float64(w) * VAE_SF2
                out[(2 * s_v + tok) * 2 + 1] = Float64(w + 1) * VAE_SF2
    return out^


struct _RefhqRope(Movable):
    var v_cos: Tensor
    var v_sin: Tensor
    var ca_v_cos: Tensor
    var ca_v_sin: Tensor

    def __init__(out self, var v_cos: Tensor, var v_sin: Tensor,
                 var ca_v_cos: Tensor, var ca_v_sin: Tensor):
        self.v_cos = v_cos^; self.v_sin = v_sin^
        self.ca_v_cos = ca_v_cos^; self.ca_v_sin = ca_v_sin^


def _refhq_video_rope(
    nf: Int, nh: Int, nw: Int, s_v: Int, ctx: DeviceContext
) raises -> _RefhqRope:
    var vc = _build_video_coords_fps(nf, nh, nw, REFHQ_FPS)
    var vtc = _video_temporal_coords_dims(vc, s_v)
    var vrope = _compute_rope(
        vc, 3, s_v, VD, _mp3(), ROPE_THETA, V_HEADS, STDtype.BF16, ctx
    )
    var cav = _compute_rope(
        vtc, 1, s_v, CA_DIM, _mp1(), ROPE_THETA, V_HEADS, STDtype.BF16, ctx
    )
    return _RefhqRope(
        _clone(vrope[0], ctx), _clone(vrope[1], ctx),
        _clone(cav[0], ctx), _clone(cav[1], ctx),
    )


# ONE flat DiT forward for refhq: consumes/returns PATCHIFIED tensors
# (video [1,S_V,128], audio [1,S_A,128] — the layout the res_2s ref loop
# carries), takes a PREBUILT _Mod (the 3 guidance passes share one sigma, so
# the modulation is built once per eval), threads `skip_cross_modal` into the
# block forward (the reference "mod" pass), and applies the distilled LoRA via
# the FACTORIZED cached path at `lora_mult`.
def _refhq_forward_flat[
    S_V_CT: Int, S_A_CT: Int, S_VPAD_CT: Int, S_APAD_CT: Int
](
    lora: LoraSet,
    lora_mult: Float32,
    cfg: LTX2Config,
    g: _Globals,
    stream: LTX2BlockStream,
    v_flat: Tensor, a_flat: Tensor,
    enc: Tensor, aenc: Tensor,
    mod: _Mod,
    vr: _RefhqRope,
    a_cos: Tensor, a_sin: Tensor, ca_a_cos: Tensor, ca_a_sin: Tensor,
    skip_cross_modal: Bool,
    verbose_lora: Bool,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var hs = _linear_b(v_flat, g.v_pin_w, g.v_pin_b, ctx)
    var ahs = _linear_b(a_flat, g.a_pin_w, g.a_pin_b, ctx)
    var n_lora_total = 0
    for i in range(NUM_LAYERS):
        var w: LTX2AVBlockWeights
        if _is_boundary(i):
            w = LTX2AVBlockWeights.load(String(CKPT_FP8), i, cfg, ctx)
        else:
            var blk = stream.load_block_bf16(i, ctx)
            w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx)
        if lora_mult != Float32(0.0):
            n_lora_total += lora.attach_ltx2_cached_block_factors(
                i, w, lora_mult, ctx
            )
        var outs = ltx2_block_forward_av[
            S_V_CT, S_A_CT, N_TXT, S_VPAD_CT, S_APAD_CT
        ](
            w, hs, ahs, enc, aenc,
            mod.v_temb, mod.a_temb, mod.v_ca_ss, mod.a_ca_ss,
            mod.v_ca_gate, mod.a_ca_gate,
            mod.v_prompt_ts, mod.a_prompt_ts,
            vr.v_cos, vr.v_sin, a_cos, a_sin,
            vr.ca_v_cos, vr.ca_v_sin, ca_a_cos, ca_a_sin, EPS, ctx,
            skip_cross_modal,
        )
        hs = _clone(outs[0], ctx)
        ahs = _clone(outs[1], ctx)
    if verbose_lora:
        print(
            "  [refhq] factorized LoRA attach count (48 blocks @ mult",
            lora_mult, "):", n_lora_total,
        )
    var v_vel = _output_stage(hs, g.v_sst, mod.v_embedded, g.v_pout_w,
                              g.v_pout_b, VD, ctx)
    var a_vel = _output_stage(ahs, g.a_sst, mod.a_embedded, g.a_pout_w,
                              g.a_pout_b, AD, ctx)
    return (v_vel^, a_vel^)


# X0Model conversion (ltx-core utils.py to_denoised): x0 = x - v*sigma,
# computed in F32 and cast back to the sample dtype (BF16) — applied PER PASS
# before the guider combine, exactly like X0Model wraps the velocity model.
def _refhq_x0(x: Tensor, vel: Tensor, sigma: Float32, ctx: DeviceContext) raises -> Tensor:
    var xf = cast_tensor(x, STDtype.F32, ctx)
    var vf = cast_tensor(vel, STDtype.F32, ctx)
    return cast_tensor(sub(xf, mul_scalar(vf, sigma, ctx), ctx), STDtype.BF16, ctx)


# patchify/unpatchify between latent layout and the loop's token layout.
def _refhq_patchify_video(x: Tensor, s_v: Int, ctx: DeviceContext) raises -> Tensor:
    return permute(reshape(x, _sh3(1, 128, s_v), ctx), _video_perm(), ctx)


def _refhq_unpatchify_video(
    x: Tensor, nf: Int, nh: Int, nw: Int, ctx: DeviceContext
) raises -> Tensor:
    return reshape(permute(x, _unvideo_perm(), ctx), _sh5(1, 128, nf, nh, nw), ctx)


def _refhq_patchify_audio(x: Tensor, s_a: Int, ctx: DeviceContext) raises -> Tensor:
    return reshape(permute(x, _audio_perm(), ctx), _sh3(1, s_a, 128), ctx)


def _refhq_unpatchify_audio(x: Tensor, s_a: Int, ctx: DeviceContext) raises -> Tensor:
    return permute(reshape(x, _sh4(1, s_a, AUDIO_C, AUDIO_MEL), ctx), _unaudio_perm(), ctx)


# Stage-2 forward-noise init (GaussianNoiser, all-ones mask):
#   x = noise*scale + init*(1-scale)   computed F32, stored BF16.
def _refhq_noise_blend(
    init_bf16: Tensor, noise_f32: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    var i32 = cast_tensor(init_bf16, STDtype.F32, ctx)
    var a = mul_scalar(noise_f32, scale, ctx)
    var b = mul_scalar(i32, Float32(1.0) - scale, ctx)
    return cast_tensor(add(a, b, ctx), STDtype.BF16, ctx)


def _refhq_save(
    var names: List[String],
    var tensors: List[ArcPointer[Tensor]],
    path: String,
    ctx: DeviceContext,
) raises:
    save_safetensors(names^, tensors^, path, ctx)
    print("  [dump]", path)


def run_refhq(
    contexts_path: String,
    noises_path: String,
    out_dir: String,
    steps: Int,
) raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2.3 refhq — REAL HQ recipe, reference-gated ===")
    print("  contexts:", contexts_path)
    print("  noises:", noises_path, "(- = production Mojo randn)")
    print("  out:", out_dir, " steps:", steps)
    print("  stage1:", REFHQ_NF, "x", REFHQ_NH1, "x", REFHQ_NW1, " S_V=", REFHQ_S_V1,
          " S_A=", REFHQ_S_A, "  stage2: S_V2=", REFHQ_S_V2)

    var ck = ShardedSafeTensors.open(String(CKPT_FP8))

    # ── distilled LoRA only (the reference passes exactly one LoRA) ──
    var lora = LoraSet.load(String(LORA_DISTILLED_PATH))
    if lora.format != FMT_LTX2_DISTILLED:
        raise Error("refhq: distilled LoRA not FMT_LTX2_DISTILLED")
    var n_fac = lora.preload_ltx2_block_factors(ctx)
    print("  [lora] distilled mappings:", lora.num_mappings(),
          " preloaded block factors:", n_fac)

    # ── globals + stage-1 LoRA strength 0.25 ──
    print("  [load] globals (stage-1 LoRA", REFHQ_LORA_S1, ")")
    var gw = _load_global_weights_dict(ck, ctx)
    var n_g1 = lora.apply_to_globals(gw, REFHQ_LORA_S1, ctx)
    print("  [lora] global deltas (stage-1):", n_g1)
    var g = _Globals(
        _clone(gw[String("patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "patchify_proj.bias", ctx),
        _clone(gw[String("audio_patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "audio_patchify_proj.bias", ctx),
        _clone(gw[String("proj_out.weight")][], ctx),
        _load_global_bf16(ck, "proj_out.bias", ctx),
        _clone(gw[String("audio_proj_out.weight")][], ctx),
        _load_global_bf16(ck, "audio_proj_out.bias", ctx),
        _load_global_bf16(ck, "scale_shift_table", ctx),
        _load_global_bf16(ck, "audio_scale_shift_table", ctx),
    )

    # ── contexts: positive + NEGATIVE, both through the connectors ──
    print("  [connector] projecting pos + neg contexts")
    var v_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    )
    var cf = ShardedSafeTensors.open(contexts_path)
    var v_pre = Tensor.from_view_as_bf16(cf.tensor_view("video_context"), ctx)
    var a_pre = Tensor.from_view_as_bf16(cf.tensor_view("audio_context"), ctx)
    var has_neg_v = _st_has(cf, String("neg_video_context"))
    var has_neg_a = _st_has(cf, String("neg_audio_context"))
    var vn_pre: Tensor
    if has_neg_v:
        vn_pre = Tensor.from_view_as_bf16(cf.tensor_view("neg_video_context"), ctx)
    else:
        # zeros negative (the MATH-gate fallback the ref run also uses)
        vn_pre = cast_tensor(mul_scalar(v_pre, Float32(0.0), ctx), STDtype.BF16, ctx)
    var an_pre: Tensor
    if has_neg_a:
        an_pre = Tensor.from_view_as_bf16(cf.tensor_view("neg_audio_context"), ctx)
    else:
        an_pre = cast_tensor(mul_scalar(a_pre, Float32(0.0), ctx), STDtype.BF16, ctx)
    var enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](v_conn, v_pre, ctx)
    var aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](a_conn, a_pre, ctx)
    var neg_enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](v_conn, vn_pre, ctx)
    var neg_aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](a_conn, an_pre, ctx)
    _ = _stats(String("enc"), enc, ctx)
    _ = _stats(String("neg_enc"), neg_enc, ctx)

    # ── RoPE (fps=25 profile) ──
    print("  [rope] stage-1 + stage-2 video tables, audio tables (fps=25)")
    var vr1 = _refhq_video_rope(REFHQ_NF, REFHQ_NH1, REFHQ_NW1, REFHQ_S_V1, ctx)
    var vr2 = _refhq_video_rope(REFHQ_NF, REFHQ_NH2, REFHQ_NW2, REFHQ_S_V2, ctx)
    var ac = _build_audio_coords_dims(REFHQ_S_A)
    var arope = _compute_rope(
        ac, 1, REFHQ_S_A, AD, _mp1(), ROPE_THETA, A_HEADS, STDtype.BF16, ctx
    )
    var caarope = _compute_rope(
        ac, 1, REFHQ_S_A, CA_DIM, _mp1(), ROPE_THETA, A_HEADS, STDtype.BF16, ctx
    )
    var a_cos = _clone(arope[0], ctx); var a_sin = _clone(arope[1], ctx)
    var ca_a_cos = _clone(caarope[0], ctx); var ca_a_sin = _clone(caarope[1], ctx)

    # ── stream (raw-fp8 resident range, like run_staged) ──
    var stream = LTX2BlockStream.open(String(CKPT_FP8))
    if stream.block_count() != NUM_LAYERS:
        raise Error("stream block_count != 48")
    stream.enable_fp8_resident_range(RESIDENT_FIRST, RESIDENT_LAST_STAGED, ctx)
    print("  [resident] raw-fp8 range", RESIDENT_FIRST, "..", RESIDENT_LAST_STAGED,
          " bytes:", stream.resident_bytes())

    # ── noise source + init latents (PATCHIFIED layout) ──
    var ns: NoiseSource
    var video_x: Tensor
    var audio_x: Tensor
    if noises_path != "-":
        ns = NoiseSource.fixture(noises_path)
        if ns.has_key(String("init_video")) and ns.has_key(String("init_audio")):
            print("  [init] latents INJECTED from fixture (init_video/init_audio)")
            video_x = cast_tensor(
                ns.load_key_f32(String("init_video"), ctx), STDtype.BF16, ctx
            )
            audio_x = cast_tensor(
                ns.load_key_f32(String("init_audio"), ctx), STDtype.BF16, ctx
            )
        else:
            raise Error(
                "refhq: noises fixture given but init_video/init_audio missing "
                "— the parity contract requires injected init latents"
            )
    else:
        ns = NoiseSource.production(SEED)
        # rng-contract: mojo-native-not-pytorch-parity.
        video_x = randn(_sh3(1, REFHQ_S_V1, 128), SEED, STDtype.BF16, ctx)
        audio_x = randn(_sh3(1, REFHQ_S_A, 128), SEED + 1, STDtype.BF16, ctx)

    # ── stage-1 sigmas: LTX2Scheduler token shift, tokens = S_V1 ──
    var sig1 = _ltx2_scheduler_sigmas(steps, REFHQ_S_V1)
    print("  [sampler] stage-1 sigmas (", len(sig1), "):", end="")
    for i in range(len(sig1)):
        print(" ", sig1[i], end="")
    print("")

    # ── STAGE 1: guided res_2s ──
    print("  [Stage1] GUIDED res_2s,", steps, "steps x 2 evals x 3 passes")
    var lora_mult_s1 = REFHQ_LORA_S1
    var first_eval = True

    @parameter
    def _den_s1(vx: Tensor, ax: Tensor, sigma: Float32) raises -> Tuple[Tensor, Tensor]:
        var mod = _build_mod_dims(ck, gw, sigma, REFHQ_S_V1, REFHQ_S_A, ctx)
        # pass 1: cond (positive contexts)
        var c = _refhq_forward_flat[REFHQ_S_V1, REFHQ_S_A, REFHQ_SPAD, REFHQ_SPAD](
            lora, lora_mult_s1, cfg, g, stream, vx, ax, enc, aenc, mod, vr1,
            a_cos, a_sin, ca_a_cos, ca_a_sin, False, first_eval, ctx,
        )
        # pass 2: uncond (NEGATIVE contexts)
        var u = _refhq_forward_flat[REFHQ_S_V1, REFHQ_S_A, REFHQ_SPAD, REFHQ_SPAD](
            lora, lora_mult_s1, cfg, g, stream, vx, ax, neg_enc, neg_aenc, mod, vr1,
            a_cos, a_sin, ca_a_cos, ca_a_sin, False, False, ctx,
        )
        # pass 3: mod (positive contexts, cross-modal attention SKIPPED)
        var m = _refhq_forward_flat[REFHQ_S_V1, REFHQ_S_A, REFHQ_SPAD, REFHQ_SPAD](
            lora, lora_mult_s1, cfg, g, stream, vx, ax, enc, aenc, mod, vr1,
            a_cos, a_sin, ca_a_cos, ca_a_sin, True, False, ctx,
        )
        first_eval = False
        # X0Model conversion per pass, THEN guider combine on x0
        var c_v = _refhq_x0(vx, c[0], sigma, ctx)
        var u_v = _refhq_x0(vx, u[0], sigma, ctx)
        var m_v = _refhq_x0(vx, m[0], sigma, ctx)
        var c_a = _refhq_x0(ax, c[1], sigma, ctx)
        var u_a = _refhq_x0(ax, u[1], sigma, ctx)
        var m_a = _refhq_x0(ax, m[1], sigma, ctx)
        var den_v = guider_calculate(
            c_v, u_v, m_v, REFHQ_V_CFG, 0.0, REFHQ_V_RESCALE, REFHQ_MOD_SCALE, ctx
        )
        var den_a = guider_calculate(
            c_a, u_a, m_a, REFHQ_A_CFG, 0.0, REFHQ_A_RESCALE, REFHQ_MOD_SCALE, ctx
        )
        return (den_v^, den_a^)

    var s1_names = List[String]()
    var s1_tensors = List[ArcPointer[Tensor]]()
    var t_s1 = perf_counter()
    var s1_out = res2s_ref_loop[_den_s1](
        sig1, video_x^, audio_x^, ns, String("s1"), s1_names, s1_tensors, ctx,
    )
    video_x = s1_out[0].clone(ctx)
    audio_x = s1_out[1].clone(ctx)
    print("  [Stage1] done in", Float32(perf_counter() - t_s1), "s")
    _refhq_save(s1_names^, s1_tensors^, out_dir + "/stage1_steps.safetensors", ctx)
    _ = _stats(String("video_x(stage1)"), video_x, ctx)
    _ = _stats(String("audio_x(stage1)"), audio_x, ctx)

    # ── UPSAMPLE (un_normalize -> x2 -> normalize) ──
    print("  [upscale] spatial x2 LatentUpsampler")
    var v_lat1 = _refhq_unpatchify_video(video_x, REFHQ_NF, REFHQ_NH1, REFHQ_NW1, ctx)
    var up_st = ShardedSafeTensors.open(String(SPATIAL_UPSCALER))
    var upsampler = LatentUpsampler(up_st, False, ctx)
    var std_t = _load_vae_stat(String("std-of-means"), ctx)
    var mean_t = _load_vae_stat(String("mean-of-means"), ctx)
    var up_ndhwc = upsample_video(_ncdhw_to_ndhwc(v_lat1, ctx), std_t, mean_t, upsampler, ctx)
    var upscaled = _ndhwc_to_ncdhw(up_ndhwc, ctx)
    var up_names = List[String]()
    var up_tensors = List[ArcPointer[Tensor]]()
    up_names.append(String("in"))
    up_tensors.append(ArcPointer[Tensor](cast_tensor(v_lat1, STDtype.F32, ctx)))
    up_names.append(String("out"))
    up_tensors.append(ArcPointer[Tensor](cast_tensor(upscaled, STDtype.F32, ctx)))
    _refhq_save(up_names^, up_tensors^, out_dir + "/upsampler.safetensors", ctx)

    # ── stage-2 globals: distilled LoRA strength 0.5 (fresh reload) ──
    print("  [load] stage-2 globals (LoRA", REFHQ_LORA_S2, ")")
    gw = _load_global_weights_dict(ck, ctx)
    var n_g2 = lora.apply_to_globals(gw, REFHQ_LORA_S2, ctx)
    print("  [lora] global deltas (stage-2):", n_g2)
    g = _Globals(
        _clone(gw[String("patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "patchify_proj.bias", ctx),
        _clone(gw[String("audio_patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "audio_patchify_proj.bias", ctx),
        _clone(gw[String("proj_out.weight")][], ctx),
        _load_global_bf16(ck, "proj_out.bias", ctx),
        _clone(gw[String("audio_proj_out.weight")][], ctx),
        _load_global_bf16(ck, "audio_proj_out.bias", ctx),
        _load_global_bf16(ck, "scale_shift_table", ctx),
        _load_global_bf16(ck, "audio_scale_shift_table", ctx),
    )

    # ── STAGE 2 init: forward-noise upscaled video + stage-1 audio ──
    var s2sig = ltx2_stage2_distilled_sigmas()   # [0.909375, 0.725, 0.421875, 0.0]
    var s2_scale = s2sig[0]
    var up_flat = _refhq_patchify_video(upscaled, REFHQ_S_V2, ctx)
    var up_flat_b = cast_tensor(up_flat, STDtype.BF16, ctx)
    var s2n_v: Tensor
    var s2n_a: Tensor
    if noises_path != "-":
        s2n_v = ns.load_key_f32(String("s2init_video"), ctx)
        s2n_a = ns.load_key_f32(String("s2init_audio"), ctx)
    else:
        # rng-contract: mojo-native-not-pytorch-parity (plain randn, like
        # GaussianNoiser — NOT _get_new_noise-normalized).
        s2n_v = randn(_sh3(1, REFHQ_S_V2, 128), SEED + 100, STDtype.F32, ctx)
        s2n_a = randn(_sh3(1, REFHQ_S_A, 128), SEED + 101, STDtype.F32, ctx)
    var vx2 = _refhq_noise_blend(up_flat_b, s2n_v, s2_scale, ctx)
    var ax2 = _refhq_noise_blend(audio_x, s2n_a, s2_scale, ctx)
    print("  [Stage2] SIMPLE res_2s, sigmas 0.909375/0.725/0.421875/0  noise_scale=", s2_scale)

    var lora_mult_s2 = REFHQ_LORA_S2
    var first_eval2 = True

    @parameter
    def _den_s2(vx: Tensor, ax: Tensor, sigma: Float32) raises -> Tuple[Tensor, Tensor]:
        var mod = _build_mod_dims(ck, gw, sigma, REFHQ_S_V2, REFHQ_S_A, ctx)
        var c = _refhq_forward_flat[REFHQ_S_V2, REFHQ_S_A, REFHQ_SPAD, REFHQ_SPAD](
            lora, lora_mult_s2, cfg, g, stream, vx, ax, enc, aenc, mod, vr2,
            a_cos, a_sin, ca_a_cos, ca_a_sin, False, first_eval2, ctx,
        )
        first_eval2 = False
        return (
            _refhq_x0(vx, c[0], sigma, ctx),
            _refhq_x0(ax, c[1], sigma, ctx),
        )

    var s2_names = List[String]()
    var s2_tensors = List[ArcPointer[Tensor]]()
    var t_s2 = perf_counter()
    var s2_out = res2s_ref_loop[_den_s2](
        s2sig, vx2^, ax2^, ns, String("s2"), s2_names, s2_tensors, ctx,
    )
    vx2 = s2_out[0].clone(ctx)
    ax2 = s2_out[1].clone(ctx)
    print("  [Stage2] done in", Float32(perf_counter() - t_s2), "s")
    _refhq_save(s2_names^, s2_tensors^, out_dir + "/stage2_steps.safetensors", ctx)

    # ── final latents (unpatchified latent layout, F32) ──
    var v_final = _refhq_unpatchify_video(vx2, REFHQ_NF, REFHQ_NH2, REFHQ_NW2, ctx)
    var a_final = _refhq_unpatchify_audio(ax2, REFHQ_S_A, ctx)
    var f_names = List[String]()
    var f_tensors = List[ArcPointer[Tensor]]()
    f_names.append(String("video"))
    f_tensors.append(ArcPointer[Tensor](cast_tensor(v_final, STDtype.F32, ctx)))
    f_names.append(String("audio"))
    f_tensors.append(ArcPointer[Tensor](cast_tensor(a_final, STDtype.F32, ctx)))
    _refhq_save(f_names^, f_tensors^, out_dir + "/final_latents.safetensors", ctx)
    _ = _stats(String("video(final)"), v_final, ctx)
    _ = _stats(String("audio(final)"), a_final, ctx)

    # ── free the DiT working set BEFORE the VAE loads (24 GB discipline, same
    # as the staged path: the resident fp8 range (~20.5 GB) and the rank-384
    # LoRA factor cache (~7.6 GB) must be gone before decoder weights + frame
    # buffers allocate). Reassignment destroys the old values -> VRAM freed.
    print("  [decode] freeing resident fp8 blocks + LoRA factor cache")
    stream = LTX2BlockStream.open(String(CKPT_FP8))
    lora = LoraSet.load(String(LORA_DISTILLED_PATH))

    # ── DECODE VIDEO (latent [1,128,NF,NH2,NW2] -> 121 frames @ 768x512) ──
    print("  [decode] video VAE (latent -> frames)")
    var vae = LTX2VaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var v_final_bf16 = cast_tensor(v_final, STDtype.BF16, ctx)
    var frames = decode_video[1, 128, REFHQ_NF, REFHQ_NH2, REFHQ_NW2](
        vae, v_final_bf16, ctx
    )
    var fsh = frames.shape()
    var n_frames_out = fsh[2]
    print("  frames NCDHW: [", fsh[0], ",", fsh[1], ",", fsh[2], ",", fsh[3],
          ",", fsh[4], "]")
    _ = _stats(String("frames"), frames, ctx)

    var frame_prefix = String("refhq_frame")
    for fr in range(n_frames_out):
        var fslice = slice(frames, 2, fr, 1, ctx)
        var fs = fslice.shape()
        var chw = reshape(fslice, _sh4(fs[0], fs[1], fs[3], fs[4]), ctx)
        save_png(
            chw, out_dir + "/" + frame_prefix + _pad3(fr) + ".png", ctx,
            ValueRange.SIGNED,
        )
    print("  saved", n_frames_out, "frame PNGs ->", out_dir)

    # ── DECODE AUDIO (latent [1,8,S_A,16] -> mel -> 48kHz stereo wav) ──
    var wav_out = out_dir + "/refhq_audio.wav"
    print("  [decode] audio VAE (latent -> mel) -> vocoder (mel -> 48kHz wav)")
    var avae = LTX2AudioVaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var a_final_bf16 = cast_tensor(a_final, STDtype.BF16, ctx)
    var mel = cast_tensor(decode_audio(avae, a_final_bf16, ctx), STDtype.BF16, ctx)
    var msh = mel.shape()
    print("  mel NCHW: [", msh[0], ",", msh[1], ",", msh[2], ",", msh[3], "] BF16 storage")
    _ = _stats(String("mel"), mel, ctx)

    var voc = LTX2VocoderWithBWE.from_file(String(CKPT_BF16), ctx)
    var wav = voc.forward(mel, ctx)
    var wsh = wav.shape()
    var L = wsh[2]
    print("  wav [", wsh[0], ",", wsh[1], ",", wsh[2], "] @", voc.output_sample_rate(), "Hz")

    var wh = wav.to_host(ctx)
    var rms = 0.0
    var amax = 0.0
    var bad = False
    for i in range(len(wh)):
        var v = Float64(wh[i])
        if not (v == v): bad = True
        rms += v * v
        var av = v if v >= 0.0 else -v
        if av > amax: amax = av
    rms = sqrt(rms / Float64(len(wh)))
    print("  AUDIO rms=", Float32(rms), " absmax=", Float32(amax), " finite=",
          not bad, " nonsilent=", rms > 1.0e-4)

    var inter = List[Float32]()
    inter.resize(L * 2, Float32(0.0))
    for ch in range(2):
        for s in range(L):
            inter[s * 2 + ch] = wh[ch * L + s]
    _write_wav(wav_out, inter, voc.output_sample_rate())
    print("  wrote wav:", wav_out)

    # ── MUX (25 fps refhq profile, 3-digit frame numbers) ──
    var mp4_out = out_dir + "/ltx2_refhq.mp4"
    print("  [mux] ffmpeg frames + wav -> mp4")
    var cmd = String("ffmpeg -y -framerate 25 -i ")
    cmd += out_dir + "/" + frame_prefix + "%03d.png -i " + wav_out
    cmd += " -c:v libx264 -pix_fmt yuv420p -af apad -c:a aac -shortest -movflags +faststart "
    cmd += mp4_out + " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        print("  [mux] WARNING: ffmpeg returned", rc, "(frames+wav still saved)")

    print("=== refhq DONE ===")
    print("  mp4:", mp4_out)
    print("  wav:", wav_out)
    print("  frames:", out_dir + "/" + frame_prefix + "000.png ..",
          frame_prefix + _pad3(n_frames_out - 1) + ".png")


def _mkdir(path: String) raises:
    var cmd = String("mkdir -p ") + path + " >/dev/null 2>&1"
    _ = sys_system(cmd)


# ── argv-driven entry ──
#   hq                                                   -> 25-frame 768x512 LoRA resident A/V target
#   hq long [base|lora] [resident|stream] [audio|noaudio] [nag|nonag] [out] [steps]
#   hq audiosync [base|lora] [resident|stream] [audio|noaudio] [nag|nonag] [out] [steps]
#   hq single [base|lora] [stream|resident] [audio|noaudio] [nag|nonag] [out] [steps]
#   hq staged [base|lora] [stream|resident] [audio|noaudio] [nag|nonag] [out] [steps]
# Audio is on by default. `noaudio` is only an explicit debug escape hatch.
def main() raises:
    var a = argv()
    # refhq parity mode:
    #   hq refhq <contexts.safetensors> <noises.safetensors|-> <out_dir> [steps]
    if len(a) >= 2 and String(a[1]) == "refhq":
        if len(a) < 5:
            raise Error(
                "usage: refhq <contexts.safetensors> <noises.safetensors|-> "
                "<out_dir> [steps]"
            )
        var refhq_steps = REFHQ_STEPS
        if len(a) >= 6:
            refhq_steps = atol(String(a[5]))
        var refhq_out = String(a[4])
        _mkdir(refhq_out)
        run_refhq(String(a[2]), String(a[3]), refhq_out, refhq_steps)
        return
    var mode = String("long")
    var argbase = 1
    if len(a) >= 2 and String(a[1]) == "single":
        mode = String("single")
        argbase = 2
    elif len(a) >= 2 and String(a[1]) == "long":
        mode = String("long")
        argbase = 2
    elif len(a) >= 2 and String(a[1]) == "audiosync":
        mode = String("audiosync")
        argbase = 2
    elif len(a) >= 2 and String(a[1]) == "staged":
        mode = String("staged")
        argbase = 2

    var apply_lora = True
    var use_resident = True
    var include_audio = True
    var use_nag = False
    var profile_dit = False
    var max_steps = 0
    var out_dir = String(OUT_DIR_LONG)
    if mode == "single":
        out_dir = String(OUT_DIR)
    elif mode == "audiosync":
        out_dir = String(OUT_DIR_AUDIOSYNC)
    elif mode == "staged":
        out_dir = String(OUT_DIR2)
    var positional = 0
    var i = argbase
    while i < len(a):
        var tok = String(a[i])
        if tok == "base":
            apply_lora = False
        elif tok == "lora":
            apply_lora = True
        elif tok == "resident":
            use_resident = True
        elif tok == "stream":
            use_resident = False
        elif tok == "audio":
            include_audio = True
        elif tok == "noaudio" or tok == "video":
            include_audio = False
        elif tok == "nag":
            use_nag = True
        elif tok == "nonag":
            use_nag = False
        elif tok == "profile":
            profile_dit = True
        elif positional == 0:
            out_dir = tok
            positional += 1
        elif positional == 1:
            max_steps = atol(tok)
            positional += 1
        else:
            raise Error(String("unknown extra hq argument: ") + tok)
        i += 1
    _mkdir(out_dir)
    if mode == "staged":
        run_staged(
            apply_lora, out_dir, max_steps, use_resident, include_audio,
            use_nag, profile_dit,
        )
    elif mode == "single":
        run(apply_lora, out_dir, max_steps, use_resident, include_audio, use_nag)
    elif mode == "audiosync":
        run_audiosync(apply_lora, out_dir, max_steps, use_resident, include_audio, use_nag)
    else:
        run_long(apply_lora, out_dir, max_steps, use_resident, include_audio, use_nag)


# ── patchify/unpatchify permutations ──
def _video_perm() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); return p^

def _unvideo_perm() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); return p^

def _audio_perm() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); p.append(3)
    return p^

def _unaudio_perm() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); p.append(3)
    return p^


def _pad2(n: Int) -> String:
    if n < 10:
        return String("0") + String(n)
    return String(n)


def _pad3(n: Int) -> String:
    if n < 10:
        return String("00") + String(n)
    if n < 100:
        return String("0") + String(n)
    return String(n)


# ── WAV writer (16-bit PCM LE stereo) ──
def _write_wav(path: String, samples: List[Float32], sr: Int) raises:
    var n = len(samples)
    var channels = 2
    var bits = 16
    var byte_rate = sr * channels * (bits // 8)
    var block_align = channels * (bits // 8)
    var data_bytes = n * (bits // 8)
    var total = 44 + data_bytes
    var buf = alloc[UInt8](total)

    var riff = String("RIFF")
    for i in range(4): buf[i] = UInt8(ord(riff[byte=i]))
    var v0 = 36 + data_bytes
    buf[4] = UInt8(v0 & 0xFF); buf[5] = UInt8((v0 >> 8) & 0xFF)
    buf[6] = UInt8((v0 >> 16) & 0xFF); buf[7] = UInt8((v0 >> 24) & 0xFF)
    var wave = String("WAVE")
    for i in range(4): buf[8 + i] = UInt8(ord(wave[byte=i]))
    var fmt = String("fmt ")
    for i in range(4): buf[12 + i] = UInt8(ord(fmt[byte=i]))
    buf[16] = 16; buf[17] = 0; buf[18] = 0; buf[19] = 0
    buf[20] = 1; buf[21] = 0
    buf[22] = UInt8(channels & 0xFF); buf[23] = UInt8((channels >> 8) & 0xFF)
    buf[24] = UInt8(sr & 0xFF); buf[25] = UInt8((sr >> 8) & 0xFF)
    buf[26] = UInt8((sr >> 16) & 0xFF); buf[27] = UInt8((sr >> 24) & 0xFF)
    buf[28] = UInt8(byte_rate & 0xFF); buf[29] = UInt8((byte_rate >> 8) & 0xFF)
    buf[30] = UInt8((byte_rate >> 16) & 0xFF); buf[31] = UInt8((byte_rate >> 24) & 0xFF)
    buf[32] = UInt8(block_align & 0xFF); buf[33] = UInt8((block_align >> 8) & 0xFF)
    buf[34] = UInt8(bits & 0xFF); buf[35] = UInt8((bits >> 8) & 0xFF)
    var data = String("data")
    for i in range(4): buf[36 + i] = UInt8(ord(data[byte=i]))
    buf[40] = UInt8(data_bytes & 0xFF); buf[41] = UInt8((data_bytes >> 8) & 0xFF)
    buf[42] = UInt8((data_bytes >> 16) & 0xFF); buf[43] = UInt8((data_bytes >> 24) & 0xFF)

    for i in range(n):
        var v = samples[i]
        if v < Float32(-1.0): v = Float32(-1.0)
        elif v > Float32(1.0): v = Float32(1.0)
        var s16 = Int(v * Float32(32767.0))
        if s16 < -32768: s16 = -32768
        elif s16 > 32767: s16 = 32767
        var u = s16 if s16 >= 0 else (s16 + 65536)
        var off = 44 + i * 2
        buf[off] = UInt8(u & 0xFF)
        buf[off + 1] = UInt8((u >> 8) & 0xFF)

    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        buf.free()
        raise Error(String("write_wav: cannot open ") + path)
    var bp = BytePtr(unsafe_from_address=Int(buf))
    var done = 0
    while done < total:
        var got = sys_pwrite(fd, bp + done, total - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    buf.free()
    if done != total:
        raise Error("write_wav: short write")


# ── ffmpeg mux via system() ──
def _mux_video_mp4(out_dir: String, mp4: String, frame_prefix: String) raises:
    var cmd = String("ffmpeg -y -framerate 24 -i ")
    cmd += out_dir + "/" + frame_prefix + "%02d.png"
    cmd += " -c:v libx264 -pix_fmt yuv420p -movflags +faststart " + mp4
    cmd += " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        print("  [mux] WARNING: ffmpeg returned", rc, "(frames still saved)")


def _mux_mp4(out_dir: String, n_frames: Int, wav: String, mp4: String, frame_prefix: String) raises:
    var cmd = String("ffmpeg -y -framerate 24 -i ")
    cmd += out_dir + "/" + frame_prefix + "%02d.png -i " + wav
    cmd += " -c:v libx264 -pix_fmt yuv420p -af apad -c:a aac -shortest -movflags +faststart " + mp4
    cmd += " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        print("  [mux] WARNING: ffmpeg returned", rc, "(frames+wav still saved)")
