# LTX-2.3 T2V + AUDIO HQ capstone — HIGH-QUALITY recipe upgrade of the P7 MVP.
#
# This began as a copy of pipeline/ltx2_t2v_av_mvp.mojo upgraded to the earlier
# proven distilled recipe from the desktop app. That legacy staged/smoke path is
# retained. The production `refhq` path is separately pinned and parity tested
# against Creator LTX revision 780984275fd47128b02bef9b5c085404276866ee and uses
# the official LTX-2.3 dev-FP8 base, 1.1 distilled LoRA, and 1.1 x2 upscaler.
# The earlier HANDOFF proved on real hardware:
#   "Distilled FP8 + res2s + LoRA conditioning -> GOOD output"
# i.e. the DISTILLED model (the fp8 one we already stream) + the second-order
# Runge-Kutta res_2s sampler + the runtime LoRA path is the proven-sharp path.
# That statement applies only to the older smoke route; RefHQ does not use it.
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

from max.gpu.host import DeviceContext
from std.math import sqrt, cos as fcos, sin as fsin, pow as fpow, log as flog, exp as fexp, pi
from std.memory import alloc, ArcPointer
from std.sys import argv
from std.sys.defines import get_defined_int
from std.time import perf_counter

from serenitymojo.io.ffi import (
    sys_open, sys_pread, sys_pwrite, sys_close, sys_system, sys_rename,
    O_RDONLY, O_WRONLY, O_CREAT, O_TRUNC, BytePtr,
)
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.env import (
    env_or, serenity_checkpoint, serenity_model_root, serenity_output,
)
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.activations import silu
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    reshape, add, sub, mul, div, mul_scalar, add_scalar, slice, permute,
    concat, full_device, zeros_device,
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
from serenitymojo.offload.vmm_cuda import cu_mem_get_info, cu_mempool_trim_current
from serenitymojo.sampling.ltx2_sampling import (
    LTX2Scheduler, res2s_coefficients, res2s_substep, res2s_combine, Res2sCoeffs,
    res2s_sde_step, res2s_bong_refine, res2s_bong_active,
    ltx2_creator_noiser_from_noise, ltx2_distilled_sigmas,
    ltx2_stage2_distilled_sigmas,
    ltx2_sulphur_stage1_sigmas, ltx2_sulphur_stage2_sigmas,
    ltx2_euler_ancestral_cfgpp_step, ltx2_lcm_step,
)
from serenitymojo.sampling.ltx2_res2s_ref import (
    NoiseSource, RES2S_REF_TERMINAL_SIGMA, res2s_post_process_latent,
    res2s_ref_loop,
)
from serenitymojo.sampling.ltx2_multimodal_guider import guider_calculate
from serenitymojo.models.vae.ltx2_vae_decoder import (
    LTX2VaeDecoderWeights, decode as decode_video,
)
from serenitymojo.models.vae.ltx2_vae_encoder import (
    LTX2VaeEncoderWeights, encode as encode_conditioning_video,
)
from serenitymojo.models.vae.ltx2_tiled_decode import (
    LTX2_DESKTOP_FIRST_OUT_CHUNK,
    LTX2_DESKTOP_MIDDLE_OUT_CHUNK,
    ltx2_desktop_decode_temporal_group,
    ltx2_desktop_decode_temporal_group_portrait,
    ltx2_desktop_decode_temporal_group_full,
    ltx2_desktop_decode_temporal_group_full_portrait,
    ltx2_desktop_temporal_carry,
)
from serenitymojo.models.vae.ltx2_audio_vae import (
    LTX2AudioVaeDecoderWeights, decode as decode_audio,
    encode_spectrogram as encode_audio_spectrogram,
)
from serenitymojo.models.vae.ltx2_audio_processor import (
    load_source_waveform_f32,
    waveform_to_creator_log_mel,
)
from serenitymojo.models.vocoder.ltx2_vocoder import LTX2VocoderWithBWE
from serenitymojo.models.upsampler.ltx2_upsampler import (
    LatentUpsampler, upsample_video,
)
from serenitymojo.image.png import save_png, save_rgb24_video, ValueRange
from serenitymojo.serve.image_io import (
    decode_image_any, image_bilinear_center_crop_to_signed_nchw,
    image_rgb_at,
    raw_rgb24_video_to_signed_ncdhw_bf16_device,
)
from serenitymojo.components.artifacts import shell_quote
from serenitymojo.lora import (
    LoraSet, FMT_LTX2_DISTILLED, LTX2BlockLoraDeltaSet, LoraStreamMults,
)
from serenitymojo.serve.product_manifest import (
    json_bool, json_escape, peak_vram_mib, write_text_file,
)
from serenitymojo.serve.proc_ipc import (
    prefault_self_executable, lock_mapped_runtime_files, lock_future_mappings,
    unlock_all_mappings,
)


# ── libc getenv → String ("" if unset). For the L2P trained-LoRA overlay. ─────
from std.ffi import external_call

comptime _HQEnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_str(name: String) -> String:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var ret = external_call["getenv", _HQEnvPtr](_HQEnvPtr(unsafe_from_address=Int(buf)))
    buf.free()
    var out = String("")
    if Int(ret) == 0:
        return out
    var i = 0
    while ret[i] != UInt8(0):
        out += chr(Int(ret[i]))
        i += 1
    return out


# ── portable runtime paths ────────────────────────────────────────────────────
# Every path can be overridden independently. Defaults live below the shared
# Serenity model/output roots instead of assuming a user name or checkout path.
def _model_file(relative: String) -> String:
    return serenity_model_root() + String("/") + relative


def _ckpt_fp8() -> String:
    return env_or("LTX2_CKPT_FP8", serenity_checkpoint(String("ltx-2.3-22b-distilled-fp8.safetensors")))


def _ckpt_bf16() -> String:
    return env_or("LTX2_CKPT_BF16", serenity_checkpoint(String("ltx-2.3-22b-distilled.safetensors")))


def _refhq_ckpt_fp8() -> String:
    return env_or("LTX2_REFHQ_CKPT_FP8", serenity_checkpoint(String("ltx-2.3-22b-dev-fp8.safetensors")))


def _refhq_ckpt_bf16() -> String:
    return env_or(
        "LTX2_REFHQ_CKPT_BF16",
        serenity_checkpoint(
            String("ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors")
        ),
    )


def _request_hq_ckpt(quant: String) -> String:
    if quant == String("bf16"):
        return _refhq_ckpt_bf16()
    return _refhq_ckpt_fp8()


def _request_hq_audio_vae_ckpt(request_checkpoint: String) -> String:
    return env_or("LTX2_AUDIO_VAE_CKPT", request_checkpoint)


def _request_hq_video_vae_ckpt(request_checkpoint: String) -> String:
    return env_or("LTX2_VIDEO_VAE_CKPT", request_checkpoint)


def _refhq_uses_support_lora() raises -> Bool:
    var mode = String(_env_str("LTX2_REFHQ_SUPPORT_LORA").lower())
    if mode.byte_length() == 0 or mode == String("official"):
        return True
    if mode == String("baked"):
        return False
    raise Error(
        String("LTX2_REFHQ_SUPPORT_LORA must be 'official' or 'baked'; got '")
        + mode + String("'")
    )


def _audio_ctx_dump() -> String:
    return env_or("LTX2_AUDIO_CONTEXT", serenity_output(String("fixtures/ltx2_audio_context.safetensors")))


def _neg_ctx_dump() -> String:
    return env_or("LTX2_NEGATIVE_CONTEXT", serenity_output(String("fixtures/cached_ltx2_negative.safetensors")))


def _spatial_upscaler() -> String:
    return env_or("LTX2_SPATIAL_UPSCALER", _model_file(String("ltx2_upscalers/ltx-2-spatial-upscaler-x2-1.0.safetensors")))


def _lora_distilled() -> String:
    return env_or("LTX2_DISTILLED_LORA", serenity_checkpoint(String("ltx-2.3-22b-distilled-lora-384.safetensors")))


def _refhq_spatial_upscaler() -> String:
    return env_or("LTX2_REFHQ_SPATIAL_UPSCALER", serenity_checkpoint(String("ltx-2.3-spatial-upscaler-x2-1.1.safetensors")))


def _refhq_lora_distilled() -> String:
    return env_or("LTX2_REFHQ_DISTILLED_LORA", serenity_checkpoint(String("ltx-2.3-22b-distilled-lora-384-1.1.safetensors")))


def _request_hq_distillation_weight() raises -> Float32:
    var text = _env_str("LTX2_REQUEST_DISTILLATION_MULT")
    if text.byte_length() == 0:
        return Float32(1.0)
    var value = Float32(Float64(text))
    if value < Float32(-10.0) or value > Float32(10.0):
        raise Error("LTX2 request distillation adapter weight must be in [-10, 10]")
    return value


def _request_hq_distillation_stage_weight(
    env_name: String, fallback: Float32
) raises -> Float32:
    var text = _env_str(env_name)
    if text.byte_length() == 0:
        return fallback
    var value = Float32(Float64(text))
    if value < Float32(-10.0) or value > Float32(10.0):
        raise Error("LTX2 request distillation stage weight must be in [-10, 10]")
    return value


def _lora_camera_static() -> String:
    return env_or("LTX2_CAMERA_STATIC_LORA", _model_file(String("loras/ltx-2-19b-lora-camera-control-static.safetensors")))


def _lora_detailer() -> String:
    return env_or("LTX2_DETAILER_LORA", _model_file(String("loras/ltx-2-19b-ic-lora-detailer.safetensors")))
# Legacy single-stage default. The HQ two-stage path below overrides the
# distilled support LoRA per stage to match Lightricks' hq_2_stage_arg_parser.
comptime LORA_DISTILLED_MULT = Float32(1.0)
comptime LORA_DISTILLED_MULT_HQ_STAGE1 = Float32(0.25)
comptime LORA_DISTILLED_MULT_HQ_STAGE2 = Float32(0.5)
comptime LORA_CAMERA_STATIC_MULT = Float32(0.3)
# The detailer IC LoRA is disabled by default until IC/reference conditioning is
# wired. Applying it raw without the IC context warps faces.
comptime LORA_DETAILER_MULT = Float32(0.0)
# Legacy direct-run overlay strength. Product requests use the indexed
# LTX2_TRAINED_LORA_{i} / LTX2_TRAINED_LORA_MULT_{i} environment contract.
comptime LORA_TRAINED_MULT = Float32(1.0)

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
    # Runtime TRAINED LoRA overlays (Mojo trainer / musubi comfy format,
    # diffusion_model.transformer_blocks.* keys). The indexed environment
    # contract has no fixed stack length. LTX2_TRAINED_LORA remains a legacy
    # one-overlay compatibility input for direct runner invocations.
    var trained: List[ArcPointer[LoraSet]]
    var trained_mults: List[Float32]

    def __init__(
        out self,
        var distilled: LoraSet,
        var camera_static: LoraSet,
        var detailer: LoraSet,
        var trained: List[ArcPointer[LoraSet]],
        var trained_mults: List[Float32],
    ):
        self.distilled = distilled^
        self.camera_static = camera_static^
        self.detailer = detailer^
        self.trained = trained^
        self.trained_mults = trained_mults^

    @staticmethod
    def load() raises -> _HQLoraStack:
        var trained = List[ArcPointer[LoraSet]]()
        var trained_mults = List[Float32]()
        var count_text = _env_str("LTX2_TRAINED_LORA_COUNT")
        var count = 0
        if count_text.byte_length() > 0:
            count = atol(count_text)
            if count < 0:
                raise Error("LTX2_TRAINED_LORA_COUNT must be >= 0")
        for i in range(count):
            var path = _env_str(String("LTX2_TRAINED_LORA_") + String(i))
            if path.byte_length() == 0:
                raise Error(
                    String("missing LTX2_TRAINED_LORA_") + String(i)
                )
            var mult_text = _env_str(
                String("LTX2_TRAINED_LORA_MULT_") + String(i)
            )
            var mult = Float32(1.0)
            if mult_text.byte_length() > 0:
                mult = Float32(Float64(mult_text))
            if mult < Float32(-10.0) or mult > Float32(10.0):
                raise Error(
                    String("LTX2_TRAINED_LORA_MULT_") + String(i)
                    + String(" must be in [-10, 10]")
                )
            trained.append(ArcPointer(LoraSet.load(path)))
            trained_mults.append(mult)
        if count == 0:
            var legacy_path = _env_str("LTX2_TRAINED_LORA")
            if legacy_path.byte_length() > 0:
                trained.append(ArcPointer(LoraSet.load(legacy_path)))
                trained_mults.append(LORA_TRAINED_MULT)
        return _HQLoraStack(
            LoraSet.load(_lora_distilled()),
            LoraSet.load(_lora_camera_static()),
            LoraSet.load(_lora_detailer()),
            trained^,
            trained_mults^,
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
        if len(self.trained) != len(self.trained_mults):
            raise Error("HQ: trained LoRA path/weight count mismatch")
        for i in range(len(self.trained)):
            var pairs = self.trained[i][].num_lora_pairs_in_file()
            if pairs == 0 or pairs != self.trained[i][].num_mappings():
                raise Error(
                    String("HQ: trained LoRA ") + String(i)
                    + String(" has zero or unmapped LTX2 pairs")
                )

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
        for i in range(len(self.trained)):
            print(
                "  [lora] TRAINED overlay", i,
                self.trained[i][].num_mappings(), "mappings @",
                self.trained_mults[i],
            )

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
        for i in range(len(self.trained)):
            total += self.trained[i][].preload_ltx2_block_factors(ctx)
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
        for i in range(len(self.trained)):
            total += self.trained[i][].attach_ltx2_cached_block_factors(
                block_idx, block, self.trained_mults[i], ctx
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


# The request-driven Creator-HQ route intentionally has a different LoRA
# contract from the legacy AudioSync stack above: official 2.3 support LoRA +
# every authored trained overlay. Camera/detailer are not implicit request
# substitutions and therefore are not loaded here.
struct _RequestHQLoraStack(Movable):
    # A distillation adapter belongs to the selected checkpoint, not to
    # Serenity.  Third-party dev/full finetunes must be able to run pristine,
    # and a directly distilled full checkpoint must never receive a second,
    # unrelated distillation delta.
    var distilled: Optional[LoraSet]
    var trained: List[ArcPointer[LoraSet]]
    var trained_mults: List[Float32]
    var trained_streams: List[LoraStreamMults]

    def __init__(
        out self,
        var distilled: Optional[LoraSet],
        var trained: List[ArcPointer[LoraSet]],
        var trained_mults: List[Float32],
        var trained_streams: List[LoraStreamMults],
    ):
        self.distilled = distilled^
        self.trained = trained^
        self.trained_mults = trained_mults^
        self.trained_streams = trained_streams^

    @staticmethod
    def _parse_streams(text: String, i: Int) raises -> LoraStreamMults:
        """`LTX2_TRAINED_LORA_STREAMS_{i}` encoding: five comma-joined floats
        `video,video_to_audio,audio,audio_to_video,other`, each in [0, 1]
        (the KJ LTX2LoraLoaderAdvanced slider range)."""
        var parts = text.split(String(","))
        if len(parts) != 5:
            raise Error(
                String("LTX2_TRAINED_LORA_STREAMS_") + String(i)
                + String(" must be 5 comma-joined floats")
            )
        var vals = List[Float32]()
        for p in parts:
            var v = Float32(Float64(String(p)))
            if v < Float32(0.0) or v > Float32(1.0):
                raise Error(
                    String("LTX2_TRAINED_LORA_STREAMS_") + String(i)
                    + String(" entries must be in [0, 1]")
                )
            vals.append(v)
        return LoraStreamMults(vals[0], vals[1], vals[2], vals[3], vals[4])

    @staticmethod
    def load() raises -> _RequestHQLoraStack:
        var distilled = Optional[LoraSet](None)
        var distilled_path = _env_str("LTX2_REQUEST_DISTILLATION_LORA")
        if distilled_path.byte_length() > 0:
            distilled = Optional[LoraSet](LoraSet.load(distilled_path))
        var trained = List[ArcPointer[LoraSet]]()
        var trained_mults = List[Float32]()
        var trained_streams = List[LoraStreamMults]()
        var count_text = _env_str("LTX2_TRAINED_LORA_COUNT")
        var count = 0
        if count_text.byte_length() > 0:
            count = atol(count_text)
            if count < 0:
                raise Error("LTX2_TRAINED_LORA_COUNT must be >= 0")
        for i in range(count):
            var path = _env_str(String("LTX2_TRAINED_LORA_") + String(i))
            if path.byte_length() == 0:
                raise Error(String("missing LTX2_TRAINED_LORA_") + String(i))
            var mult_text = _env_str(
                String("LTX2_TRAINED_LORA_MULT_") + String(i)
            )
            var mult = Float32(1.0)
            if mult_text.byte_length() > 0:
                mult = Float32(Float64(mult_text))
            if mult < Float32(-10.0) or mult > Float32(10.0):
                raise Error(
                    String("LTX2_TRAINED_LORA_MULT_") + String(i)
                    + String(" must be in [-10, 10]")
                )
            var streams_text = _env_str(
                String("LTX2_TRAINED_LORA_STREAMS_") + String(i)
            )
            var streams = LoraStreamMults.identity()
            if streams_text.byte_length() > 0:
                streams = _RequestHQLoraStack._parse_streams(streams_text, i)
            trained.append(ArcPointer(LoraSet.load(path)))
            trained_mults.append(mult)
            trained_streams.append(streams)
        # `distilled` was already declared at the top of this function for the
        # explicit `LTX2_REQUEST_DISTILLATION_LORA` path; re-declaring it here
        # is what broke the parse ("invalid redefinition of 'distilled'"), and
        # a plain assignment would be worse than the error: with
        # LTX2_REFHQ_SUPPORT_LORA unset (i.e. "official", the DEFAULT) it would
        # silently overwrite an adapter the request explicitly asked for.
        #
        # The struct contract above is explicit that "a directly distilled full
        # checkpoint must never receive a second, unrelated distillation delta",
        # so at most ONE adapter may ever be selected. Order: an explicit
        # request adapter wins; otherwise fall back to the official support
        # LoRA; 'baked' selects none.
        if (not distilled) and _refhq_uses_support_lora():
            distilled = Optional[LoraSet](
                LoraSet.load(_refhq_lora_distilled())
            )
        return _RequestHQLoraStack(
            distilled^, trained^, trained_mults^,
            trained_streams^,
        )

    def validate(self) raises:
        if self.distilled:
            ref distilled = self.distilled.value()
            if distilled.format != FMT_LTX2_DISTILLED:
                raise Error(
                    "LTX2 request HQ: selected distillation adapter is not LTX2 distilled format"
                )
            if distilled.num_lora_pairs_in_file() != distilled.num_mappings():
                raise Error(
                    "LTX2 request HQ: selected distillation adapter has unmapped pairs"
                )
        if len(self.trained) != len(self.trained_mults):
            raise Error("LTX2 request HQ: trained LoRA path/weight count mismatch")
        for i in range(len(self.trained)):
            var pairs = self.trained[i][].num_lora_pairs_in_file()
            if pairs == 0 or pairs != self.trained[i][].num_mappings():
                raise Error(
                    String("LTX2 request HQ: trained LoRA ") + String(i)
                    + String(" has zero or unmapped pairs")
                )

    def print_summary(self):
        if self.distilled:
            print(
                "  [lora] selected distillation adapter mappings:",
                self.distilled.value().num_mappings(),
                "@ creator stage multipliers",
            )
        else:
            print("  [lora] distillation adapter: none (pristine checkpoint)")
        for i in range(len(self.trained)):
            print("  [lora] authored overlay", i,
                  self.trained[i][].num_mappings(), "mappings @",
                  self.trained_mults[i])
            if not self.trained_streams[i].is_identity():
                print("  [lora]   streams v", self.trained_streams[i].video,
                      "v2a", self.trained_streams[i].video_to_audio,
                      "a", self.trained_streams[i].audio,
                      "a2v", self.trained_streams[i].audio_to_video,
                      "other", self.trained_streams[i].other)

    def apply_to_globals(
        self,
        mut gw: Dict[String, ArcPointer[Tensor]],
        distilled_mult: Float32,
        ctx: DeviceContext,
    ) raises -> Int:
        var total = 0
        if self.distilled and distilled_mult != Float32(0.0):
            total += self.distilled.value().apply_to_globals(
                gw, distilled_mult, ctx
            )
        for i in range(len(self.trained)):
            total += self.trained[i][].apply_to_globals_streamed(
                gw, self.trained_mults[i], self.trained_streams[i], ctx
            )
        return total

    def apply_to_av_block(
        self,
        block_idx: Int,
        mut block: LTX2AVBlockWeights,
        distilled_mult: Float32,
        ctx: DeviceContext,
    ) raises -> Int:
        var deltas = LTX2BlockLoraDeltaSet()
        if self.distilled and distilled_mult != Float32(0.0):
            _ = self.distilled.value().accumulate_ltx2_block_deltas(
                block_idx, deltas, distilled_mult, ctx
            )
        for i in range(len(self.trained)):
            _ = self.trained[i][].accumulate_ltx2_block_deltas_streamed(
                block_idx, deltas, self.trained_mults[i],
                self.trained_streams[i], ctx
            )
        return deltas.apply_to_av_block(block, ctx)

    def preload_block_factors(mut self, ctx: DeviceContext) raises -> Int:
        var total = 0
        if self.distilled:
            total += self.distilled.value().preload_ltx2_block_factors(ctx)
        for i in range(len(self.trained)):
            total += self.trained[i][].preload_ltx2_block_factors(ctx)
        return total

    def attach_to_av_block(
        self,
        block_idx: Int,
        mut block: LTX2AVBlockWeights,
        distilled_mult: Float32,
        ctx: DeviceContext,
    ) raises -> Int:
        var total = 0
        if self.distilled and distilled_mult != Float32(0.0):
            total += self.distilled.value().attach_ltx2_cached_block_factors(
                block_idx, block, distilled_mult, ctx
            )
        for i in range(len(self.trained)):
            total += self.trained[i][].attach_ltx2_cached_block_factors_streamed(
                block_idx, block, self.trained_mults[i],
                self.trained_streams[i], ctx
            )
        return total


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
        var b1 = fam + ".emb.timestep_embedder.linear_1.bias"
        var k2 = fam + ".emb.timestep_embedder.linear_2.weight"
        var b2 = fam + ".emb.timestep_embedder.linear_2.bias"
        var k3 = fam + ".linear.weight"
        var b3 = fam + ".linear.bias"
        gw[k1] = ArcPointer[Tensor](_load_global_bf16(st, k1, ctx))
        gw[b1] = ArcPointer[Tensor](_load_global_bf16(st, b1, ctx))
        gw[k2] = ArcPointer[Tensor](_load_global_bf16(st, k2, ctx))
        gw[b2] = ArcPointer[Tensor](_load_global_bf16(st, b2, ctx))
        gw[k3] = ArcPointer[Tensor](_load_global_bf16(st, k3, ctx))
        gw[b3] = ArcPointer[Tensor](_load_global_bf16(st, b3, ctx))
    return gw^


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


@fieldwise_init
struct _TensorPair(Movable):
    var video: Tensor
    var audio: Tensor

    def move_into(deinit self, mut video_out: Tensor, mut audio_out: Tensor):
        video_out = self.video^
        audio_out = self.audio^


def _clone_pair(video: Tensor, audio: Tensor, ctx: DeviceContext) raises -> _TensorPair:
    var v_dev = ctx.enqueue_create_buffer[DType.uint8](video.nbytes())
    var a_dev = ctx.enqueue_create_buffer[DType.uint8](audio.nbytes())
    ctx.enqueue_copy(dst_buf=v_dev, src_buf=video.buf)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=audio.buf)
    ctx.synchronize()
    return _TensorPair(
        Tensor(v_dev^, video.shape(), video.dtype()),
        Tensor(a_dev^, audio.shape(), audio.dtype()),
    )


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


def _refhq_creator_stage1_sigmas() -> List[Float32]:
    """Pinned Creator float32 schedule for the fixed 15-step ref-HQ arm.

    Creator builds this schedule through torch.float32 operations.  The
    generic Mojo port above intentionally supports arbitrary shapes/steps but
    computes its intermediates in Float64, which was measured to move several
    entries by multiple ULPs at REFHQ_S_V1=8160.  Trajectory parity requires
    the exact pinned-oracle values rather than a numerically close schedule.
    """
    var out = List[Float32]()
    out.append(Float32(1.0))
    out.append(Float32(0.9934906959533691))
    out.append(Float32(0.9860149025917053))
    out.append(Float32(0.9773396253585815))
    out.append(Float32(0.9671507477760315))
    out.append(Float32(0.9550145864486694))
    out.append(Float32(0.9403136372566223))
    out.append(Float32(0.9221396446228027))
    out.append(Float32(0.8990960121154785))
    out.append(Float32(0.8689230680465698))
    out.append(Float32(0.8277072310447693))
    out.append(Float32(0.7680275440216064))
    out.append(Float32(0.6738965511322021))
    out.append(Float32(0.5033778548240662))
    out.append(Float32(0.10000002384185791))
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
    var b1 = _clone(gw[base + ".emb.timestep_embedder.linear_1.bias"][], ctx)
    var h = _linear_b(emb, w1, b1, ctx)
    h = silu(h, ctx)
    var w2 = _clone(gw[base + ".emb.timestep_embedder.linear_2.weight"][], ctx)
    var b2 = _clone(gw[base + ".emb.timestep_embedder.linear_2.bias"][], ctx)
    var embedded = _linear_b(h, w2, b2, ctx)
    var h2 = silu(embedded, ctx)
    var lw = _clone(gw[base + ".linear.weight"][], ctx)
    var lb = _clone(gw[base + ".linear.bias"][], ctx)
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


def _request_hq_video_timestep_values(
    sigma: Float32,
    sequence_tokens: Int,
    conditioned_video_tokens: Int,
    conditioned_video_denoise: Float32,
    uniform_timestep: Bool = False,
    conditioned_video_denoise_values: Optional[List[Float32]] = None,
) raises -> List[Float32]:
    """Materialize the official `sigma * denoise_mask` video timesteps.

    Pure T2V may retain one broadcast row. I2V uses a conditioned prefix and
    V2V conditions the complete sequence, so those paths always materialize
    every token. Values include LTX2's required timestep multiplier.
    """
    if sequence_tokens <= 0:
        raise Error("LTX2 request: video sequence token count must be positive")
    if conditioned_video_tokens < 0 or conditioned_video_tokens > sequence_tokens:
        raise Error("LTX2 request: conditioned video token count out of range")
    if (
        conditioned_video_denoise < Float32(0.0)
        or conditioned_video_denoise > Float32(1.0)
    ):
        raise Error("LTX2 request: conditioned video denoise must be in [0, 1]")
    var explicit_values = List[Float32]()
    if conditioned_video_denoise_values:
        explicit_values = conditioned_video_denoise_values.value().copy()
        if len(explicit_values) != sequence_tokens:
            raise Error(
                "LTX2 request: explicit video denoise mask length mismatch"
            )
        for value in explicit_values:
            if value < Float32(0.0) or value > Float32(1.0):
                raise Error(
                    "LTX2 request: explicit video denoise values must be in [0, 1]"
                )
    var rows = (
        1 if uniform_timestep and conditioned_video_tokens == 0
        and len(explicit_values) == 0
        else sequence_tokens
    )
    var values = List[Float32]()
    for i in range(rows):
        var token_denoise = Float32(1.0)
        if len(explicit_values) > 0:
            token_denoise = explicit_values[i]
        elif i < conditioned_video_tokens:
            token_denoise = conditioned_video_denoise
        values.append(sigma * token_denoise * TS_MULT)
    return values^


def _build_mod_dims(
    st: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    sigma: Float32,
    s_v: Int,
    s_a: Int,
    ctx: DeviceContext,
    uniform_timestep: Bool = False,
    conditioned_video_tokens: Int = 0,
    conditioned_video_denoise: Float32 = 1.0,
    conditioned_video_denoise_values: Optional[List[Float32]] = None,
    audio_denoise: Float32 = 1.0,
    conditioned_audio_denoise_values: Optional[List[Float32]] = None,
) raises -> _Mod:
    # Pure T2V uses one sigma for every video token. The AV block's modulation
    # ops are broadcast-aware, so retaining one video row avoids a multi-GiB
    # full-grid tensor. Keep audio/prompt rows materialized: bounded Creator
    # parity measured a small audio first-denoiser drift when those were also
    # broadcast, and their memory cost is bounded.
    # A conditioned frame makes the model timestep spatially non-uniform:
    # official `timesteps_from_mask` is `sigma * denoise_mask`, so preserved
    # frame-zero tokens receive zero at strength 1.0. Stage 2 normally uses one
    # broadcast row; materialize its rows only when I2V requires the prefix.
    var ts_v = _request_hq_video_timestep_values(
        sigma, s_v, conditioned_video_tokens, conditioned_video_denoise,
        uniform_timestep, conditioned_video_denoise_values,
    )
    var v_rows = len(ts_v)
    var a_rows = s_a
    var prompt_rows = N_TXT
    var vt = _adaln_single(st, gw, String("adaln_single"), ts_v, ctx)
    var v_temb = reshape(vt[0], _sh3(1, v_rows, 9 * VD), ctx)
    var v_embedded = reshape(vt[1], _sh3(1, v_rows, VD), ctx)

    var ts_a = List[Float32]()
    var explicit_audio_values = List[Float32]()
    if conditioned_audio_denoise_values:
        explicit_audio_values = conditioned_audio_denoise_values.value().copy()
        if len(explicit_audio_values) != a_rows:
            raise Error(
                "LTX2 request: explicit audio denoise mask length mismatch"
            )
        for value in explicit_audio_values:
            if value < Float32(0.0) or value > Float32(1.0):
                raise Error(
                    "LTX2 request: explicit audio denoise values must be in [0, 1]"
                )
    for i in range(a_rows):
        var token_denoise = audio_denoise
        if len(explicit_audio_values) > 0:
            token_denoise = explicit_audio_values[i]
        ts_a.append(sigma * token_denoise * TS_MULT)
    var at = _adaln_single(st, gw, String("audio_adaln_single"), ts_a, ctx)
    var a_temb = reshape(at[0], _sh3(1, a_rows, 9 * AD), ctx)
    var a_embedded = reshape(at[1], _sh3(1, a_rows, AD), ctx)

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
    for _ in range(prompt_rows): ts_p.append(sigma * TS_MULT)
    var vpt = _adaln_single(st, gw, String("prompt_adaln_single"), ts_p, ctx)
    var v_prompt_ts = reshape(vpt[0], _sh3(1, prompt_rows, 2 * VD), ctx)
    var apt = _adaln_single(st, gw, String("audio_prompt_adaln_single"), ts_p, ctx)
    var a_prompt_ts = reshape(apt[0], _sh3(1, prompt_rows, 2 * AD), ctx)

    return _Mod(v_temb^, a_temb^, v_embedded^, a_embedded^, v_ca_ss^, a_ca_ss^,
                v_ca_gate^, a_ca_gate^, v_prompt_ts^, a_prompt_ts^)


def _build_mod(
    st: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    sigma: Float32,
    ctx: DeviceContext,
) raises -> _Mod:
    return _build_mod_dims(st, gw, sigma, S_V, S_A, ctx)


# ── Valid (non-pad) row count from a [1] F32 dump key ────────────────────────
# The context dumps are RIGHT-padded [valid, pad]; rows >= valid are replaced
# with the connector's learnable registers (reference semantics — unmasked raw
# pads imprint the foreground-lattice artifact, 2026-07-09 forensics).
# Missing key -> -1 = legacy no-replacement behavior (old dumps), loudly.
def _load_ctx_len(
    st: ShardedSafeTensors, key: String, ctx: DeviceContext
) raises -> Int:
    if not _st_has(st, key):
        print("  [ctx] WARNING: dump has no '" + key + "' — connector runs",
              "WITHOUT register replacement (lattice-artifact class);",
              "regenerate with scripts/ltx2_make_context_dumps.py")
        return -1
    var h = Tensor.from_view(st.tensor_view(key), ctx).to_host(ctx)
    if len(h) < 1:
        return -1
    var v = Int(h[0])
    if v < 0:
        v = 0
    if v > N_TXT:
        v = N_TXT
    return v


def _load_video_nag_context(
    v_conn: LTX2ConnectorWeights,
    ctx: DeviceContext,
    path: String = String(""),
) raises -> Tensor:
    """Load the cached null/negative video context and project it like positive context."""
    var resolved = path.copy()
    if resolved.byte_length() == 0:
        resolved = _neg_ctx_dump()
    var neg_st = ShardedSafeTensors.open(resolved)
    var tensor_key = String("text_hidden")
    var len_key = String("text_len")
    if _st_has(neg_st, String("neg_video_context")):
        tensor_key = String("neg_video_context")
        len_key = String("neg_video_len")
    var neg_pre = Tensor.from_view_as_bf16(
        neg_st.tensor_view(tensor_key), ctx
    )
    var neg_valid = _load_ctx_len(neg_st, len_key, ctx)
    print("  [ctx] NAG/neg valid rows:", neg_valid, "/", N_TXT)
    return ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](
        v_conn, neg_pre, ctx, valid_len=neg_valid
    )


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
        if _is_boundary(i) and not stream.has_int4():
            w = LTX2AVBlockWeights.load(_ckpt_fp8(), i, cfg, ctx)
        else:
            # int4: boundary blocks route through the stream too (slab has them).
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
            var cloned = _clone_pair(outs_nag[0], outs_nag[1], ctx)
            cloned^.move_into(hs, ahs)
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
            var cloned = _clone_pair(outs[0], outs[1], ctx)
            cloned^.move_into(hs, ahs)
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
    # LTX-2.3 bundles the VAE INSIDE the DiT checkpoint under a `vae.` prefix.
    # LTX-2.5 ships the VAE as a STANDALONE file whose keys carry no prefix
    # (measured: ltx-2.5-video-vae-conv-bf16.safetensors = 170 tensors,
    # 84 decoder / 84 encoder / 2 `per_channel_statistics.*`). Try the bundled
    # layout first, then fall back to the standalone file.
    var st = ShardedSafeTensors.open(_ckpt_bf16())
    var bundled = String("vae.per_channel_statistics.") + name
    if st.has_tensor(bundled):
        return Tensor.from_view_as_bf16(st.tensor_view(bundled), ctx)
    var vst = ShardedSafeTensors.open(
        _request_hq_video_vae_ckpt(_ckpt_bf16())
    )
    return Tensor.from_view_as_bf16(
        vst.tensor_view(String("per_channel_statistics.") + name), ctx
    )


def _refhq_spatial_upscale_scoped(
    video_ncdhw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Run the official x2 latent upsampler with weights scoped to this call."""
    var up_st = ShardedSafeTensors.open(_refhq_spatial_upscaler())
    var upsampler = LatentUpsampler(up_st, False, ctx)
    var std_t = _load_vae_stat(String("std-of-means"), ctx)
    var mean_t = _load_vae_stat(String("mean-of-means"), ctx)
    var up_ndhwc = upsample_video(
        _ncdhw_to_ndhwc(video_ncdhw, ctx), std_t, mean_t, upsampler, ctx
    )
    return _ndhwc_to_ncdhw(up_ndhwc, ctx)


def _refhq_spatial_upscale_resident(
    video_ncdhw: Tensor,
    upsampler: LatentUpsampler,
    std_t: Tensor,
    mean_t: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Run the official x2 upscaler from request-owned resident weights."""
    var up_ndhwc = upsample_video(
        _ncdhw_to_ndhwc(video_ncdhw, ctx),
        std_t,
        mean_t,
        upsampler,
        ctx,
    )
    return _ndhwc_to_ncdhw(up_ndhwc, ctx)


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


def _write_ltx2_atomic(path: String, content: String) raises:
    var tmp = path + String(".tmp")
    write_text_file(tmp, content)
    if sys_rename(tmp, path) != 0:
        raise Error(String("LTX2 request: cannot publish ") + path)


def _write_ltx2_status(
    out_dir: String,
    state: String,
    phase: String,
    step: Int,
    total: Int,
    message: String,
) raises:
    var body = String("{\n")
    body += String('  "schema":"serenity.ltx2.status.v1",\n')
    body += String('  "readiness_label":"experimental",\n')
    body += String('  "state":"') + json_escape(state) + String('",\n')
    body += String('  "phase":"') + json_escape(phase) + String('",\n')
    body += String('  "step":') + String(step) + String(",\n")
    body += String('  "total":') + String(total) + String(",\n")
    body += String('  "message":"') + json_escape(message) + String('"\n')
    body += String("}\n")
    _write_ltx2_atomic(out_dir + String("/status.json"), body)


def _ltx2_proc_read_bytes() raises -> Int:
    """Read the kernel's physical-I/O counter for this exact Mojo process."""
    var fd = sys_open(String("/proc/self/io"), O_RDONLY, 0)
    if fd < 0:
        raise Error("LTX2 hardware safety gate: cannot open /proc/self/io")
    var cap = 4096
    var buf = alloc[UInt8](cap)
    var n = sys_pread(fd, buf, cap, 0)
    _ = sys_close(fd)
    if n <= 0:
        buf.free()
        raise Error("LTX2 hardware safety gate: cannot read /proc/self/io")
    var text = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=buf, length=n)))
    buf.free()
    for line in text.split("\n"):
        var row = String(line)
        if row.startswith("read_bytes:"):
            var fields = row.split(":")
            if len(fields) != 2:
                break
            return atol(String(fields[1]).strip())
    raise Error(
        "LTX2 hardware safety gate: /proc/self/io has no read_bytes field"
    )


def _write_ltx2_denoise_io(
    out_dir: String, state: String, start_bytes: Int, current_bytes: Int
) raises:
    var body = String("{\n")
    body += String('  "schema":"serenity.denoise_io.v1",\n')
    body += String('  "state":"') + json_escape(state) + String('",\n')
    body += String('  "read_bytes_start":') + String(start_bytes) + String(",\n")
    body += String('  "read_bytes_current":') + String(current_bytes) + String(",\n")
    body += String('  "read_bytes_delta":') \
        + String(current_bytes - start_bytes) + String("\n}\n")
    _write_ltx2_atomic(out_dir + String("/denoise_io.json"), body)


def _assert_ltx2_no_denoise_disk_reads(
    out_dir: String, start_bytes: Int, allow_checkpoint_streaming: Bool = False
) raises:
    var current = _ltx2_proc_read_bytes()
    if allow_checkpoint_streaming:
        _write_ltx2_denoise_io(
            out_dir, String("checkpoint_stream"), start_bytes, current
        )
        return
    if current != start_bytes:
        _write_ltx2_denoise_io(
            out_dir, String("failed"), start_bytes, current
        )
        raise Error(
            String("LTX2 hardware safety gate: process read ")
            + String(current - start_bytes)
            + String(" physical bytes after the denoise boundary")
        )


def _record_ltx2_min_free(current: Int) raises -> Int:
    var mem = cu_mem_get_info()
    if current == 0 or mem.free_bytes < current:
        return mem.free_bytes
    return current


def _write_ltx2_result(
    out_dir: String,
    artifact_path: String,
    width: Int,
    height: Int,
    frame_count: Int,
    fps: Float64,
    steps: Int,
    seed: UInt64,
    include_audio: Bool,
    sampler: String,
    scheduler: String,
    quant: String,
    context_path: String,
    negative_context_path: String,
    request_lora_count: Int,
    load_seconds: Float64,
    conditioning_seconds: Float64,
    source_encode_seconds: Float64,
    prepare_seconds: Float64,
    denoise_seconds: Float64,
    video_decode_seconds: Float64,
    audio_decode_seconds: Float64,
    mux_seconds: Float64,
    total_seconds: Float64,
    total_vram_bytes: Int,
    min_free_bytes: Int,
) raises:
    var denoise_per_step = Float64(0.0)
    if steps > 0:
        denoise_per_step = denoise_seconds / Float64(steps)
    var peak_mib = Float64(0.0)
    if total_vram_bytes > 0 and min_free_bytes > 0:
        peak_mib = peak_vram_mib(total_vram_bytes, min_free_bytes)
    var duration_seconds = Float64(0.0)
    if fps > 0.0:
        duration_seconds = Float64(frame_count) / fps

    var body = String("{\n")
    body += String('  "schema":"serenity.ltx2.result.v1",\n')
    body += String('  "readiness_label":"experimental",\n')
    body += String('  "state":"done",\n')
    body += String('  "accepted_sampler_parity":false,\n')
    body += String('  "accepted_speed_parity":false,\n')
    body += String('  "request_path":"') + json_escape(out_dir + String("/request.json")) + String('",\n')
    body += String('  "artifact_path":"') + json_escape(artifact_path) + String('",\n')
    body += String('  "width":') + String(width) + String(",\n")
    body += String('  "height":') + String(height) + String(",\n")
    body += String('  "frame_count":') + String(frame_count) + String(",\n")
    body += String('  "fps":') + String(fps) + String(",\n")
    body += String('  "duration_seconds":') + String(duration_seconds) + String(",\n")
    body += String('  "steps":') + String(steps) + String(",\n")
    body += String('  "seed":') + String(seed) + String(",\n")
    body += String('  "include_audio":') + json_bool(include_audio) + String(",\n")
    body += String('  "executed_sampler":"') + json_escape(sampler) + String('",\n')
    body += String('  "executed_scheduler":"') + json_escape(scheduler) + String('",\n')
    body += String('  "quant":"') + json_escape(quant) + String('",\n')
    body += String('  "conditioning_positive":"') + json_escape(context_path) + String('",\n')
    body += String('  "conditioning_negative":"') + json_escape(negative_context_path) + String('",\n')
    body += String('  "request_lora_count":') + String(request_lora_count) + String(",\n")
    body += String('  "request_loras":[')
    for i in range(request_lora_count):
        if i > 0:
            body += String(",")
        var name = _env_str(String("LTX2_TRAINED_LORA_NAME_") + String(i))
        var path = _env_str(String("LTX2_TRAINED_LORA_") + String(i))
        var weight = _env_str(String("LTX2_TRAINED_LORA_MULT_") + String(i))
        if weight.byte_length() == 0:
            weight = String("1.0")
        body += String('{"name":"') + json_escape(name)
        body += String('","path":"') + json_escape(path)
        body += String('","weight":') + weight + String("}")
    body += String("],\n")
    var dtype_contract = String(
        "int4_resident_w4a16_bf16_activations_f32_reductions"
    )
    if quant == String("bf16"):
        dtype_contract = String(
            "bf16_transformer_weights_bf16_activations_f32_reductions"
        )
    elif quant == String("fp8"):
        dtype_contract = String(
            "fp8_transformer_bf16_activations_f32_reductions"
        )
    body += String('  "dtype_contract":"') + dtype_contract + String('",\n')
    body += String('  "timings":{\n')
    body += String('    "load_seconds":') + String(load_seconds) + String(",\n")
    body += String('    "conditioning_seconds":') + String(conditioning_seconds) + String(",\n")
    body += String('    "source_encode_seconds":') + String(source_encode_seconds) + String(",\n")
    body += String('    "prepare_seconds":') + String(prepare_seconds) + String(",\n")
    body += String('    "denoise_seconds":') + String(denoise_seconds) + String(",\n")
    body += String('    "denoise_seconds_per_step":') + String(denoise_per_step) + String(",\n")
    body += String('    "video_decode_seconds":') + String(video_decode_seconds) + String(",\n")
    body += String('    "audio_decode_seconds":') + String(audio_decode_seconds) + String(",\n")
    body += String('    "mux_seconds":') + String(mux_seconds) + String(",\n")
    body += String('    "total_wall_seconds":') + String(total_seconds) + String("\n")
    body += String("  },\n")
    body += String('  "peak_vram_mib":') + String(peak_mib) + String(",\n")
    body += String('  "artifact_paths":["') + json_escape(artifact_path) + String('","') + json_escape(out_dir + String("/result.json")) + String('"]\n')
    body += String("}\n")
    _write_ltx2_atomic(out_dir + String("/result.json"), body)


def _write_ltx2_decode_handoff(
    out_dir: String,
    steps: Int,
    seed: UInt64,
    include_audio: Bool,
    guidance_mode: String,
    quant: String,
    context_path: String,
    negative_context_path: String,
    request_lora_count: Int,
    load_seconds: Float64,
    conditioning_seconds: Float64,
    source_encode_seconds: Float64,
    prepare_seconds: Float64,
    denoise_seconds: Float64,
    elapsed_seconds: Float64,
    total_vram_bytes: Int,
    min_free_bytes: Int,
) raises:
    """Persist the exact denoise metrics needed by the fresh decode process."""
    var body = String("{\n")
    body += String('  "schema":"serenity.ltx2.decode_handoff.v1",\n')
    body += String('  "steps":') + String(steps) + String(",\n")
    body += String('  "seed":') + String(seed) + String(",\n")
    body += String('  "include_audio":') + json_bool(include_audio) + String(",\n")
    body += String('  "guidance_mode":"') + json_escape(guidance_mode) + String('",\n')
    body += String('  "quant":"') + json_escape(quant) + String('",\n')
    body += String('  "context_path":"') + json_escape(context_path) + String('",\n')
    body += String('  "negative_context_path":"') + json_escape(negative_context_path) + String('",\n')
    body += String('  "request_lora_count":') + String(request_lora_count) + String(",\n")
    body += String('  "load_seconds":') + String(load_seconds) + String(",\n")
    body += String('  "conditioning_seconds":') + String(conditioning_seconds) + String(",\n")
    body += String('  "source_encode_seconds":') + String(source_encode_seconds) + String(",\n")
    body += String('  "prepare_seconds":') + String(prepare_seconds) + String(",\n")
    body += String('  "denoise_seconds":') + String(denoise_seconds) + String(",\n")
    body += String('  "elapsed_seconds":') + String(elapsed_seconds) + String(",\n")
    body += String('  "total_vram_bytes":') + String(total_vram_bytes) + String(",\n")
    body += String('  "min_free_bytes":') + String(min_free_bytes) + String("\n")
    body += String("}\n")
    _write_ltx2_atomic(out_dir + String("/decode_handoff.json"), body)


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
    steps: Int = HQ_STEPS,
    seed: UInt64 = SEED,
    fps: Float64 = 24.0,
    context_path: String = String(""),
    negative_context_path: String = String(""),
    distilled_euler: Bool = False,
) raises:
    var total_t0 = perf_counter()
    var load_seconds = Float64(0.0)
    var conditioning_seconds = Float64(0.0)
    var source_encode_seconds = Float64(0.0)
    var prepare_seconds = Float64(0.0)
    var denoise_seconds = Float64(0.0)
    var video_decode_seconds = Float64(0.0)
    var audio_decode_seconds = Float64(0.0)
    var mux_seconds = Float64(0.0)
    _write_ltx2_status(
        out_dir, String("running"), String("loading_model"), 0, steps,
        String("Loading LTX2 model and LoRA weights"),
    )
    var ctx = DeviceContext()
    var mem0 = cu_mem_get_info()
    var total_vram_bytes = mem0.total_bytes
    var min_free_bytes = mem0.free_bytes
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2.3 T2V HQ (res_2s) 768x512 distilled ===")
    print("  target frames:", NUM_FRAMES_CT, " decoded frames:", 1 + (NF_CT - 1) * 8)
    print("  res:768x512  NF/NH/NW:", NF_CT, NH_CT, NW_CT, " S_V:", S_V_CT, " S_A:", S_A_CT,
          " N_TXT:", N_TXT, " blocks:", NUM_LAYERS)
    print("  sampler:", "distilled Euler" if distilled_euler else "res_2s",
          " steps:", steps,
          "  LoRA:", "HQ stack" if apply_lora else "OFF", " out_dir:", out_dir)
    print("  weights:", "FP8-resident warm range" if use_resident else "FP8 stream")
    print("  audio:", "ON (A/V default)" if include_audio else "OFF (explicit noaudio)")
    print("  NAG:", "video-only ON" if use_nag else "OFF")
    var frame_prefix = String("hq_frame")
    if max_steps != 0:
        frame_prefix = String("dev_frame")
        print("  [quality] DEV SMOKE ONLY: incomplete denoise, not an HQ artifact")

    var ck = ShardedSafeTensors.open(_ckpt_fp8())

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
    load_seconds = perf_counter() - total_t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    _write_ltx2_status(
        out_dir, String("running"), String("conditioning"), 0, steps,
        String("Loading and projecting prompt conditioning"),
    )
    var conditioning_t0 = perf_counter()
    print("  [connector] loading + running video/audio (BF16 storage)")
    var v_conn = LTX2ConnectorWeights.load(
        _ckpt_fp8(), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_conn = LTX2ConnectorWeights.load(
        _ckpt_fp8(), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    )

    # FULL 1024-token contexts (no tail slice): use the entire dump.
    var _ctx_dump_path = context_path.copy()
    if _ctx_dump_path.byte_length() == 0:
        _ctx_dump_path = _env_str("LTX2_CTX_DUMP")
    if _ctx_dump_path.byte_length() == 0:
        _ctx_dump_path = _audio_ctx_dump()
    else:
        print("  [ctx] OVERRIDE dump:", _ctx_dump_path)
    var dump = ShardedSafeTensors.open(_ctx_dump_path)
    var video_pre = Tensor.from_view_as_bf16(dump.tensor_view("video_context"), ctx)  # [1,1024,4096]
    var audio_pre = Tensor.from_view_as_bf16(dump.tensor_view("audio_context"), ctx)  # [1,1024,2048]
    _ = _stats(String("video_pre(FULL 1024)"), video_pre, ctx)
    _ = _stats(String("audio_pre(FULL 1024 REAL)"), audio_pre, ctx)

    var ctx_valid = _load_ctx_len(dump, String("video_len"), ctx)
    print("  [ctx] valid rows:", ctx_valid, "/", N_TXT,
          "(pads -> learnable registers)")
    var enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](
        v_conn, video_pre, ctx, valid_len=ctx_valid
    )
    var aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](
        a_conn, audio_pre, ctx, valid_len=ctx_valid
    )
    _ = _stats(String("enc"), enc, ctx)
    _ = _stats(String("aenc"), aenc, ctx)
    var nag = NAGContext.disabled(ctx)
    if use_nag:
        print("  [NAG] loading video null context (audio NAG disabled: no audio null dump)")
        var nag_v = _load_video_nag_context(
            v_conn, ctx, negative_context_path
        )
        _ = _stats(String("nag_video_context"), nag_v, ctx)
        var nag_a = _nag_audio_placeholder(ctx)
        nag = NAGContext.defaults(nag_v^, nag_a^, False)
    conditioning_seconds = perf_counter() - conditioning_t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    var prepare_t0 = perf_counter()
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
    var video_x = randn(_sh5(1, 128, NF_CT, NH_CT, NW_CT), seed, STDtype.BF16, ctx)
    var audio_x = randn(_sh4(1, AUDIO_C, S_A_CT, AUDIO_MEL), seed + 1, STDtype.BF16, ctx)
    if steps <= 0:
        raise Error("steps must be > 0")
    if fps <= 0.0:
        raise Error("fps must be > 0")
    if max_steps < 0:
        raise Error("max_steps must be >= 0")
    prepare_seconds = perf_counter() - prepare_t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    # Keep the block stream scoped to denoise only, so the resident FP8 cache is
    # freed before VAE/vocoder decode allocate their own working sets.
    var run_denoise_scope = max_steps >= 0
    if run_denoise_scope:
        _write_ltx2_status(
            out_dir, String("running"), String("denoising"), 0, steps,
            String("Preparing denoiser"),
        )
        var denoise_t0 = perf_counter()
        var stream = LTX2BlockStream.open(_ckpt_fp8())
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

        var sched = LTX2Scheduler(
            _ltx2_scheduler_sigmas(steps, S_V_CT)
        )
        if distilled_euler:
            sched = LTX2Scheduler.distilled()
            if steps != sched.num_steps:
                raise Error(
                    String("distilled Euler requires exactly ")
                    + String(sched.num_steps) + String(" steps; request supplied ")
                    + String(steps)
                )
        var sigmas = sched.sigmas()
        print("  [sampler] LTX2Scheduler token-shifted sigmas (", len(sigmas), "):", end="")
        for i in range(len(sigmas)):
            print(" ", sigmas[i], end="")
        print("")

        var n_steps = sched.num_steps
        if max_steps > 0 and max_steps < n_steps:
            n_steps = max_steps

        for step in range(n_steps):
            _write_ltx2_status(
                out_dir, String("running"), String("denoising"), step + 1,
                n_steps,
                String("Denoising step ") + String(step + 1) + String(" of ")
                + String(n_steps),
            )
            var sigma = sigmas[step]
            var sigma_next = sigmas[step + 1]
            if distilled_euler:
                print(
                    "  --- step", step + 1, "/", sched.num_steps,
                    " sigma=", sigma, " -> ", sigma_next, "---",
                )
                var vel = _model_forward_p[
                    S_V_CT, S_A_CT, S_VPAD_CT, S_APAD_CT
                ](
                    ck, gw, lora_stack, apply_lora, LORA_DISTILLED_MULT,
                    LORA_CAMERA_STATIC_MULT, LORA_DETAILER_MULT, cfg, g, stream,
                    video_x, audio_x, enc, aenc,
                    v_cos, v_sin, a_cos, a_sin,
                    ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
                    nag,
                    sigma, NF_CT, NH_CT, NW_CT, False, ctx,
                )
                video_x = sched.step(video_x, vel[0], step, ctx)
                audio_x = sched.step(audio_x, vel[1], step, ctx)
                _ = _stats(String("video_x"), video_x, ctx)
                _ = _stats(String("audio_x"), audio_x, ctx)
                continue
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
                _res2s_hq_noise(v_mid, seed + UInt64(10000 + step), ctx), ctx,
            )
            a_mid = res2s_sde_step(
                audio_x, a_mid, Float64(sigma), Float64(c.sub_sigma),
                _res2s_hq_noise(a_mid, seed + UInt64(11000 + step), ctx), ctx,
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
                _res2s_hq_noise(v_next, seed + UInt64(20000 + step), ctx), ctx,
            )
            audio_x = res2s_sde_step(
                a_anchor, a_next, Float64(sigma), Float64(sigma_next_eff),
                _res2s_hq_noise(a_next, seed + UInt64(21000 + step), ctx), ctx,
            )
            _ = _stats(String("video_x"), video_x, ctx)
            _ = _stats(String("audio_x"), audio_x, ctx)

        if not distilled_euler and n_steps == sched.num_steps:
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

        denoise_seconds = perf_counter() - denoise_t0
        min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    print("  [denoise] done -> decoding")

    # ── DECODE VIDEO ──
    _write_ltx2_status(
        out_dir, String("running"), String("decoding_video"), steps, steps,
        String("Decoding video frames"),
    )
    var video_decode_t0 = perf_counter()
    print("  [decode] video VAE (latent -> frames)")
    var vae = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
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
    video_decode_seconds = perf_counter() - video_decode_t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    if not include_audio:
        var mp4_video = out_dir + "/ltx2_t2v_hq.mp4"
        if max_steps != 0:
            mp4_video = out_dir + "/ltx2_t2v_dev_smoke.mp4"
        print("  [decode] audio disabled by explicit noaudio; muxing video-only mp4")
        _write_ltx2_status(
            out_dir, String("running"), String("muxing"), steps, steps,
            String("Muxing video"),
        )
        var mux_t0 = perf_counter()
        _mux_video_mp4(out_dir, mp4_video, frame_prefix, fps)
        mux_seconds = perf_counter() - mux_t0
        min_free_bytes = _record_ltx2_min_free(min_free_bytes)
        var sampler_name = String("euler") if distilled_euler else String("res2s")
        var scheduler_name = String("ltx2_distilled") if distilled_euler else String("ltx2")
        _write_ltx2_result(
            out_dir, mp4_video, fsh[4], fsh[3], n_frames_out, fps, steps,
            seed, False, sampler_name, scheduler_name, String("fp8"), context_path,
            negative_context_path, len(lora_stack.trained), load_seconds,
            conditioning_seconds, source_encode_seconds, prepare_seconds,
            denoise_seconds,
            video_decode_seconds, audio_decode_seconds, mux_seconds,
            perf_counter() - total_t0, total_vram_bytes, min_free_bytes,
        )
        _write_ltx2_status(
            out_dir, String("done"), String("done"), steps, steps,
            String("Video ready"),
        )
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
    _write_ltx2_status(
        out_dir, String("running"), String("decoding_audio"), steps, steps,
        String("Decoding audio"),
    )
    var audio_decode_t0 = perf_counter()
    print("  [decode] audio VAE (latent -> mel) -> vocoder (mel -> 48kHz wav)")
    var avae = LTX2AudioVaeDecoderWeights.load(_ckpt_bf16(), ctx)
    var audio_x_bf16 = cast_tensor(audio_x, STDtype.BF16, ctx)
    var mel_raw = decode_audio(avae, audio_x_bf16, ctx)
    var mel = cast_tensor(mel_raw, STDtype.BF16, ctx)
    var msh = mel.shape()
    print("  mel NCHW: [", msh[0], ",", msh[1], ",", msh[2], ",", msh[3], "] BF16 storage")
    _ = _stats(String("mel"), mel, ctx)

    var voc = LTX2VocoderWithBWE.from_file(_ckpt_bf16(), ctx)
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
    audio_decode_seconds = perf_counter() - audio_decode_t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    print("  [mux] ffmpeg frames + wav -> mp4")
    _write_ltx2_status(
        out_dir, String("running"), String("muxing"), steps, steps,
        String("Muxing video and audio"),
    )
    var mux_t0 = perf_counter()
    _mux_mp4(out_dir, n_frames_out, wav_out, mp4_out, frame_prefix, fps)
    mux_seconds = perf_counter() - mux_t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)
    var sampler_name = String("euler") if distilled_euler else String("res2s")
    var scheduler_name = String("ltx2_distilled") if distilled_euler else String("ltx2")
    _write_ltx2_result(
        out_dir, mp4_out, fsh[4], fsh[3], n_frames_out, fps, steps, seed,
        True, sampler_name, scheduler_name, String("fp8"), context_path,
        negative_context_path, len(lora_stack.trained), load_seconds,
        conditioning_seconds, source_encode_seconds, prepare_seconds,
        denoise_seconds,
        video_decode_seconds, audio_decode_seconds, mux_seconds,
        perf_counter() - total_t0, total_vram_bytes, min_free_bytes,
    )
    _write_ltx2_status(
        out_dir, String("done"), String("done"), steps, steps,
        String("Video and audio ready"),
    )
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


def run_request_profile(
    checkpoint: String,
    quant: String,
    width: Int,
    height: Int,
    frames: Int,
    steps: Int,
    seed: UInt64,
    fps: Float64,
    sampler: String,
    scheduler: String,
    guidance_mode: String,
    context_path: String,
    negative_context_path: String,
    contexts_are_projected: Bool,
    noise_fixture_path: String,
    image_path: String,
    image_strength: Float64,
    last_image_path: String,
    last_image_strength: Float64,
    video_path: String,
    video_strength: Float64,
    video_mask_path: String,
    video_edit_mode: String,
    video_edit_start: Float64,
    video_edit_end: Float64,
    video_source_frames: Int,
    source_audio_path: String,
    source_audio_sample_rate: Int,
    source_audio_channels: Int,
    source_audio_samples: Int,
    regenerate_source_audio: Bool,
    include_audio: Bool,
    defer_decode: Bool,
    out_dir: String,
) raises:
    """Admit a runtime request onto one of the compiled LTX2 video kernels.

    Request values are never substituted. Unsupported geometry/sampler pairs
    fail before model loading so the UI can report the exact rejected values.
    """
    if checkpoint.byte_length() == 0:
        raise Error("LTX2 request: checkpoint cannot be empty")
    if (
        quant != String("bf16")
        and quant != String("fp8")
        and quant != String("int4")
    ):
        raise Error(
            String("LTX2 request: unsupported quant '") + quant
            + String("'; verified request modes: bf16, fp8, int4")
        )
    var sampler_key = String(sampler.lower())
    var scheduler_key = String(scheduler.lower())
    var res2s = (
        sampler_key == String("res2s") or sampler_key == String("res_2s")
    ) and scheduler_key == String("ltx2")
    var distilled_euler = sampler_key == String("euler") and (
        scheduler_key == String("ltx2_distilled")
    )
    var workflow_profile = _env_str("LTX2_REQUEST_WORKFLOW_PROFILE")
    var sulphur_creator = (
        workflow_profile == String("sulphur-2-base-distilled-v1")
    )
    var sulphur_sampler = (
        sampler_key == String("euler_ancestral_cfg_pp")
        and scheduler_key == String("sulphur_creator_8_3")
    )
    if sulphur_creator and (
        guidance_mode != String("distilled") or not sulphur_sampler
    ):
        raise Error(
            String("LTX2 Sulphur creator workflow requires distilled/")
            + String("euler_ancestral_cfg_pp/sulphur_creator_8_3; got ")
            + guidance_mode + String("/") + sampler + String("/") + scheduler
        )
    if (
        guidance_mode == String("distilled")
        and not distilled_euler
        and not (sulphur_creator and sulphur_sampler)
    ):
        raise Error(
            String("LTX2 request: distilled guidance requires euler/")
            + String("ltx2_distilled or its registered creator workflow; got ")
            + sampler + String("/")
            + scheduler
        )
    if guidance_mode == String("dev") and not res2s:
        raise Error(
            String("LTX2 request: dev guidance requires res2s/ltx2; got '")
            + sampler + String("/") + scheduler
            + String("'")
        )
    if context_path.byte_length() == 0:
        raise Error("LTX2 request: caps_positive conditioning path is required")
    if negative_context_path.byte_length() == 0:
        raise Error(
            "LTX2 request: verified Creator-HQ profile requires negative conditioning"
        )
    if image_strength < 0.0 or image_strength > 1.0:
        raise Error("LTX2 request: image_strength must be in [0, 1]")
    if image_path.byte_length() == 0 and image_strength != 1.0:
        raise Error(
            "LTX2 request: image_strength requires a non-empty image_path"
        )
    if last_image_strength < 0.0 or last_image_strength > 1.0:
        raise Error("LTX2 request: last_image_strength must be in [0, 1]")
    if last_image_path.byte_length() == 0 and last_image_strength != 1.0:
        raise Error(
            "LTX2 request: last_image_strength requires last_image_path"
        )
    if video_strength < 0.0 or video_strength > 1.0:
        raise Error("LTX2 request: video_strength must be in [0, 1]")
    if video_path.byte_length() == 0 and video_strength != 1.0:
        raise Error(
            "LTX2 request: video_strength requires a non-empty video_path"
        )
    if video_mask_path.byte_length() > 0 and video_path.byte_length() == 0:
        raise Error(
            "LTX2 request: video_mask_path requires a non-empty video_path"
        )
    if (
        video_edit_mode != String("standard")
        and video_edit_mode != String("retake")
        and video_edit_mode != String("extend_start")
        and video_edit_mode != String("extend_end")
    ):
        raise Error(
            String("LTX2 request: unsupported video_edit_mode '")
            + video_edit_mode + String("'")
        )
    if video_edit_mode != String("standard"):
        if video_path.byte_length() == 0:
            raise Error(
                "LTX2 request: temporal editing requires video_path"
            )
        if video_mask_path.byte_length() > 0:
            raise Error(
                "LTX2 request: temporal editing cannot use video_mask_path"
            )
        if video_source_frames <= 1:
            raise Error(
                "LTX2 request: temporal editing requires video_source_frames"
            )
        if video_edit_start < 0.0 or video_edit_end <= video_edit_start:
            raise Error("LTX2 request: invalid temporal edit window")
        if source_audio_path.byte_length() > 0 and (
            source_audio_sample_rate <= 0
            or source_audio_channels != 2
            or source_audio_samples <= 0
        ):
            raise Error("LTX2 request: invalid creator source-audio metadata")
    if image_path.byte_length() > 0 and video_path.byte_length() > 0:
        raise Error(
            "LTX2 request: image_path and video_path are mutually exclusive"
        )
    if last_image_path.byte_length() > 0 and video_path.byte_length() > 0:
        raise Error(
            "LTX2 request: last_image_path and video_path are mutually exclusive"
        )
    if last_image_path.byte_length() > 0 and (
        video_edit_mode != String("standard")
    ):
        raise Error(
            "LTX2 keyframe interpolation cannot be combined with Retake/Extend"
        )
    if last_image_path.byte_length() > 0 and noise_fixture_path.byte_length() > 0:
        raise Error(
            "LTX2 keyframe interpolation does not accept target-only noise fixtures"
        )
    if width != REQUEST_HQ_WIDTH or height != REQUEST_HQ_HEIGHT:
        raise Error(
            String("LTX2 request: unsupported compiled size ")
            + String(width) + String("x") + String(height)
            + String("; verified size: ") + String(REQUEST_HQ_WIDTH)
            + String("x") + String(REQUEST_HQ_HEIGHT)
        )
    if frames != REQUEST_HQ_NUM_FRAMES:
        raise Error(
            String("LTX2 request: unsupported compiled frame count ")
            + String(frames) + String("; verified count: ")
            + String(REQUEST_HQ_NUM_FRAMES)
        )
    if video_edit_mode == String("retake") and (
        video_source_frames != REQUEST_HQ_NUM_FRAMES
    ):
        raise Error(
            String("LTX2 Retake source frame count must match target ")
            + String(REQUEST_HQ_NUM_FRAMES) + String("; got ")
            + String(video_source_frames)
        )
    if (
        video_edit_mode == String("extend_start")
        or video_edit_mode == String("extend_end")
    ):
        var extend_frames = REQUEST_HQ_NUM_FRAMES - video_source_frames
        if extend_frames <= 0 or extend_frames % 8 != 0:
            raise Error(
                String("LTX2 Extend requires a positive 8-frame-aligned delta; got ")
                + String(extend_frames)
            )
    if guidance_mode == String("distilled") and steps != 8:
        raise Error(
            String("LTX2 request: distilled Euler requires exactly 8 steps; got ")
            + String(steps)
        )
    if guidance_mode == String("dev") and (
        steps < 1 or steps > REQUEST_HQ_STEPS
    ):
        raise Error(
            String("LTX2 request: unsupported step count ") + String(steps)
            + String("; compiled range: 1..") + String(REQUEST_HQ_STEPS)
        )
    if guidance_mode != String("distilled") and (
        guidance_mode != String("dev")
    ):
        raise Error(
            String("LTX2 request: unsupported guidance_mode '")
            + guidance_mode + String("'; use distilled or dev")
        )
    if fps != REQUEST_HQ_FPS:
        raise Error(
            String("LTX2 request: unsupported FPS ") + String(fps)
            + String("; verified FPS: ") + String(REQUEST_HQ_FPS)
        )
    run_request_hq(
        context_path, negative_context_path, contexts_are_projected,
        noise_fixture_path, image_path, Float32(image_strength),
        last_image_path, Float32(last_image_strength),
        video_path, Float32(video_strength), video_mask_path,
        video_edit_mode, Float32(video_edit_start), Float32(video_edit_end),
        video_source_frames,
        source_audio_path, source_audio_sample_rate,
        source_audio_channels, source_audio_samples,
        regenerate_source_audio,
        out_dir, seed,
        include_audio, quant, steps, guidance_mode, defer_decode,
    )


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
def _json_bool(v: Bool) -> String:
    return String("true") if v else String("false")


struct _RunnerTimings(Movable):
    var load_globals_seconds: Float64
    var connector_seconds: Float64
    var rope_stage1_seconds: Float64
    var stage1_denoise_seconds: Float64
    var upscale_seconds: Float64
    var stage2_rope_seconds: Float64
    var stage2_load_globals_seconds: Float64
    var stage2_denoise_seconds: Float64
    var video_decode_seconds: Float64
    var frame_png_write_seconds: Float64
    var video_mux_seconds: Float64
    var audio_vae_seconds: Float64
    var vocoder_seconds: Float64
    var wav_write_seconds: Float64
    var audio_mux_seconds: Float64
    var total_runner_seconds: Float64

    def __init__(out self):
        self.load_globals_seconds = 0.0
        self.connector_seconds = 0.0
        self.rope_stage1_seconds = 0.0
        self.stage1_denoise_seconds = 0.0
        self.upscale_seconds = 0.0
        self.stage2_rope_seconds = 0.0
        self.stage2_load_globals_seconds = 0.0
        self.stage2_denoise_seconds = 0.0
        self.video_decode_seconds = 0.0
        self.frame_png_write_seconds = 0.0
        self.video_mux_seconds = 0.0
        self.audio_vae_seconds = 0.0
        self.vocoder_seconds = 0.0
        self.wav_write_seconds = 0.0
        self.audio_mux_seconds = 0.0
        self.total_runner_seconds = 0.0

    def postprocess_seconds(self) -> Float64:
        return (
            self.video_decode_seconds
            + self.frame_png_write_seconds
            + self.video_mux_seconds
            + self.audio_vae_seconds
            + self.vocoder_seconds
            + self.wav_write_seconds
            + self.audio_mux_seconds
        )

    def write_staged(
        self,
        path: String,
        include_audio: Bool,
        max_steps: Int,
        frame_count: Int,
        sample_rate: Int,
    ) raises:
        """Full two-stage staged run: mode "staged" at the stage-2 grid."""
        self.write_staged_mode(
            path, String("staged"), OUT_W2_STAGED, OUT_H2_STAGED,
            include_audio, max_steps, frame_count, sample_rate,
        )

    def write_staged_mode(
        self,
        path: String,
        mode: String,
        width: Int,
        height: Int,
        include_audio: Bool,
        max_steps: Int,
        frame_count: Int,
        sample_rate: Int,
    ) raises:
        """Same fields as write_staged with the mode marker + output grid
        parameterized — the s1out stage-1 product path writes
        mode "staged-s1out" at the STAGE-1 grid (OUT_W_STAGE1 x OUT_H_STAGE1)."""
        var content = String("{\n")
        content += String('  "schema":"serenity.ltx2_runner_timings.v1",\n')
        content += String('  "mode":"') + mode + String('",\n')
        content += String('  "include_audio":') + _json_bool(include_audio) + String(",\n")
        content += String('  "max_steps":') + String(max_steps) + String(",\n")
        content += String('  "frame_count":') + String(frame_count) + String(",\n")
        content += String('  "width":') + String(width) + String(",\n")
        content += String('  "height":') + String(height) + String(",\n")
        content += String('  "audio_sample_rate":') + String(sample_rate) + String(",\n")
        content += String('  "timings":{\n')
        content += String('    "load_globals_seconds":') + String(self.load_globals_seconds) + String(",\n")
        content += String('    "connector_seconds":') + String(self.connector_seconds) + String(",\n")
        content += String('    "rope_stage1_seconds":') + String(self.rope_stage1_seconds) + String(",\n")
        content += String('    "stage1_denoise_seconds":') + String(self.stage1_denoise_seconds) + String(",\n")
        content += String('    "upscale_seconds":') + String(self.upscale_seconds) + String(",\n")
        content += String('    "stage2_rope_seconds":') + String(self.stage2_rope_seconds) + String(",\n")
        content += String('    "stage2_load_globals_seconds":') + String(self.stage2_load_globals_seconds) + String(",\n")
        content += String('    "stage2_denoise_seconds":') + String(self.stage2_denoise_seconds) + String(",\n")
        content += String('    "video_decode_seconds":') + String(self.video_decode_seconds) + String(",\n")
        content += String('    "frame_png_write_seconds":') + String(self.frame_png_write_seconds) + String(",\n")
        content += String('    "video_mux_seconds":') + String(self.video_mux_seconds) + String(",\n")
        content += String('    "audio_vae_seconds":') + String(self.audio_vae_seconds) + String(",\n")
        content += String('    "vocoder_seconds":') + String(self.vocoder_seconds) + String(",\n")
        content += String('    "wav_write_seconds":') + String(self.wav_write_seconds) + String(",\n")
        content += String('    "audio_mux_seconds":') + String(self.audio_mux_seconds) + String(",\n")
        content += String('    "postprocess_seconds":') + String(self.postprocess_seconds()) + String(",\n")
        content += String('    "total_runner_seconds":') + String(self.total_runner_seconds) + String("\n")
        content += String("  }\n")
        content += String("}\n")
        _write_text(path, content)


def run_staged(
    apply_lora: Bool, out_dir: String, max_steps: Int, use_resident: Bool,
    include_audio: Bool, use_nag: Bool, profile_dit: Bool,
    stage1_product: Bool,
) raises:
    var ctx = DeviceContext()
    var total_t0 = perf_counter()
    var timings = _RunnerTimings()
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
    if stage1_product:
        print("  product: STAGE-1 (s1out) — upsampler/stage-2 SKIPPED "
              "(mesh-artifact fix; stage-1 decodes clean)")
    var frame_prefix = String("hq_frame")
    if max_steps != 0:
        frame_prefix = String("dev_frame")
        print("  [quality] DEV SMOKE ONLY: incomplete staged denoise, not an HQ artifact")

    var ck = ShardedSafeTensors.open(_ckpt_fp8())

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
    ctx.synchronize()
    var t0 = perf_counter()
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
    ctx.synchronize()
    timings.load_globals_seconds = perf_counter() - t0

    print("  [connector] loading + running video/audio (BF16 storage)")
    ctx.synchronize()
    t0 = perf_counter()
    var v_conn = LTX2ConnectorWeights.load(
        _ckpt_fp8(), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_conn = LTX2ConnectorWeights.load(
        _ckpt_fp8(), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    )
    var _ctx_dump_path = _env_str("LTX2_CTX_DUMP")   # prompt-conds override (L2P verdicts)
    if len(_ctx_dump_path) == 0:
        _ctx_dump_path = _audio_ctx_dump()
    else:
        print("  [ctx] OVERRIDE dump:", _ctx_dump_path)
    var dump = ShardedSafeTensors.open(_ctx_dump_path)
    var video_pre = Tensor.from_view_as_bf16(dump.tensor_view("video_context"), ctx)
    var audio_pre = Tensor.from_view_as_bf16(dump.tensor_view("audio_context"), ctx)
    var ctx_valid = _load_ctx_len(dump, String("video_len"), ctx)
    print("  [ctx] valid rows:", ctx_valid, "/", N_TXT,
          "(pads -> learnable registers)")
    var enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](
        v_conn, video_pre, ctx, valid_len=ctx_valid
    )
    var aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](
        a_conn, audio_pre, ctx, valid_len=ctx_valid
    )
    var nag = NAGContext.disabled(ctx)
    if use_nag:
        print("  [NAG] loading video null context (audio NAG disabled: no audio null dump)")
        var nag_v = _load_video_nag_context(v_conn, ctx)
        _ = _stats(String("nag_video_context"), nag_v, ctx)
        var nag_a = _nag_audio_placeholder(ctx)
        nag = NAGContext.defaults(nag_v^, nag_a^, False)
    ctx.synchronize()
    timings.connector_seconds = perf_counter() - t0

    var stream = LTX2BlockStream.open(_ckpt_fp8())
    if stream.block_count() != NUM_LAYERS:
        raise Error("stream block_count != 48")
    # INT4 override: env LTX2_INT4_SLAB → reconstruct blocks bf16 from the
    # svdint4 slab (dequant4+low-rank) instead of fp8. Disables the fp8-resident
    # preload (int4 blocks stream+reconstruct per eval; slab warms in page cache).
    var _int4_slab = _env_str("LTX2_INT4_SLAB")
    if len(_int4_slab) > 0:
        stream.attach_int4(_int4_slab)
        print("  [int4] blocks reconstructed from slab:", _int4_slab)
        if use_resident:
            print("  [int4-resident] preloading all 48 int4 blocks (~15.4GB, reconstruct on-use)")
            stream.enable_int4_resident_range(0, NUM_LAYERS - 1, ctx)
            print("  [int4-resident] resident storage bytes:", stream.resident_bytes())
    elif use_resident:
        print(
            "  [resident] preloading FP8 blocks", RESIDENT_FIRST, "..",
            RESIDENT_LAST_STAGED,
        )
        stream.enable_fp8_resident_range(RESIDENT_FIRST, RESIDENT_LAST_STAGED, ctx)
        print("  [resident] resident storage bytes:", stream.resident_bytes())

    # ── Audio RoPE (sigma-independent, same both stages) ──
    ctx.synchronize()
    t0 = perf_counter()
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
    ctx.synchronize()
    timings.rope_stage1_seconds = perf_counter() - t0

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
    ctx.synchronize()
    t0 = perf_counter()

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
    ctx.synchronize()
    timings.stage1_denoise_seconds = perf_counter() - t0

    # ── STAGE-1 PRODUCT mode (mirrors the refhq s1out branch; 2026-07-16
    # oracle isolation, MAP.md 313-360: the upsampler+stage-2 leg corrupts —
    # the on-disk spatial upscaler is a 19b-era 2.0 export feeding a 2.3 model,
    # collapsing latent variance and imprinting the dot/knit mesh, while the
    # SAME run's stage-1 latent decodes clean in every arm. Stage-1 IS the
    # product: decode video at the STAGE-1 grid (NF/NH/NW_STAGED →
    # OUT_W_STAGE1 x OUT_H_STAGE1), decode audio from the STAGE-1 audio
    # latent, and mux with the exact staged tail. Stage-1 math above is
    # untouched — this is an early exit.)
    if stage1_product:
        print("  [s1out] STAGE-1 PRODUCT mode — decoding stage-1, skipping "
              "upsampler/stage-2")
        # ══════════ DECODE VIDEO (stage-1 res) ══════════
        ctx.synchronize()
        t0 = perf_counter()
        var vae1 = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
        var vx1_bf16 = cast_tensor(video_x, STDtype.BF16, ctx)
        var frames1 = decode_video[
            1, 128, NF_STAGED, NH_STAGED, NW_STAGED
        ](vae1, vx1_bf16, ctx)
        var fsh1 = frames1.shape()
        var n_frames1 = fsh1[2]
        print("  frames NCDHW: [", fsh1[0], ",", fsh1[1], ",", fsh1[2], ",",
              fsh1[3], ",", fsh1[4], "]")
        _ = _stats(String("frames(s1out)"), frames1, ctx)
        ctx.synchronize()
        timings.video_decode_seconds = perf_counter() - t0

        ctx.synchronize()
        t0 = perf_counter()
        for fr in range(n_frames1):
            var fslice = slice(frames1, 2, fr, 1, ctx)
            var fs = fslice.shape()
            var chw = reshape(fslice, _sh4(fs[0], fs[1], fs[3], fs[4]), ctx)
            save_png(chw, out_dir + "/" + frame_prefix + _pad2(fr) + ".png", ctx, ValueRange.SIGNED)
        print("  saved", n_frames1, "frame PNGs ->", out_dir)
        ctx.synchronize()
        timings.frame_png_write_seconds = perf_counter() - t0

        if not include_audio:
            var mp4_video1 = out_dir + "/ltx2_t2v_hq2.mp4"
            if max_steps != 0:
                mp4_video1 = out_dir + "/ltx2_t2v_stage2_dev_smoke.mp4"
            print("  [decode] audio disabled by explicit noaudio; muxing video-only mp4")
            ctx.synchronize()
            t0 = perf_counter()
            _mux_video_mp4(out_dir, mp4_video1, frame_prefix)
            timings.video_mux_seconds = perf_counter() - t0
            timings.total_runner_seconds = perf_counter() - total_t0
            timings.write_staged_mode(
                out_dir + "/ltx2_runner_timings.json", String("staged-s1out"),
                OUT_W_STAGE1, OUT_H_STAGE1, include_audio, max_steps,
                n_frames1, 0,
            )
            print("=== HQ STAGED S1OUT VIDEO DONE ===")
            print("  mp4:", mp4_video1)
            print("  frames:", n_frames1)
            return

        # ══════════ AUDIO (stage-1 latent) + MUX ══════════
        var wav_out1 = out_dir + "/hq_audio.wav"
        var mp4_out1 = out_dir + "/ltx2_t2v_av_hq2.mp4"
        if max_steps != 0:
            wav_out1 = out_dir + "/dev_audio.wav"
            mp4_out1 = out_dir + "/ltx2_t2v_av_stage2_dev_smoke.mp4"
            print("  [quality] DEV SMOKE ONLY: audio mux is tested, output is not HQ quality")
        print("  [decode] audio VAE -> vocoder")
        ctx.synchronize()
        t0 = perf_counter()
        var avae1 = LTX2AudioVaeDecoderWeights.load(_ckpt_bf16(), ctx)
        var audio_x1_bf16 = cast_tensor(audio_x, STDtype.BF16, ctx)
        var mel1 = cast_tensor(decode_audio(avae1, audio_x1_bf16, ctx), STDtype.BF16, ctx)
        ctx.synchronize()
        timings.audio_vae_seconds = perf_counter() - t0
        ctx.synchronize()
        t0 = perf_counter()
        var voc1 = LTX2VocoderWithBWE.from_file(_ckpt_bf16(), ctx)
        var wav1 = voc1.forward(mel1, ctx)
        ctx.synchronize()
        timings.vocoder_seconds = perf_counter() - t0
        t0 = perf_counter()
        var wsh1 = wav1.shape()
        var L1 = wsh1[2]
        var wh1 = wav1.to_host(ctx)
        var inter1 = List[Float32]()
        inter1.resize(L1 * 2, Float32(0.0))
        for ch in range(2):
            for s in range(L1):
                inter1[s * 2 + ch] = wh1[ch * L1 + s]
        _write_wav(wav_out1, inter1, voc1.output_sample_rate())
        timings.wav_write_seconds = perf_counter() - t0
        print("  wrote wav:", wav_out1)
        t0 = perf_counter()
        _mux_mp4(out_dir, n_frames1, wav_out1, mp4_out1, frame_prefix)
        timings.audio_mux_seconds = perf_counter() - t0
        timings.total_runner_seconds = perf_counter() - total_t0
        timings.write_staged_mode(
            out_dir + "/ltx2_runner_timings.json", String("staged-s1out"),
            OUT_W_STAGE1, OUT_H_STAGE1, include_audio, max_steps,
            n_frames1, voc1.output_sample_rate(),
        )
        print("=== HQ STAGED S1OUT DONE ===")
        print("  mp4:", mp4_out1)
        print("  frames:", out_dir, "/", frame_prefix, "00.png ..  (",
              OUT_W_STAGE1, "x", OUT_H_STAGE1, ")")
        return

    # ══════════════ SPATIAL UPSAMPLE (x2) ══════════════
    print("  [upscale] loading spatial-x2 LatentUpsampler + VAE per-channel stats")
    ctx.synchronize()
    t0 = perf_counter()
    var up_st = ShardedSafeTensors.open(_spatial_upscaler())
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
    ctx.synchronize()
    timings.upscale_seconds = perf_counter() - t0

    # ══════════════ STAGE-2 RoPE (doubled spatial grid) ══════════════
    print("  [rope] stage-2 video tables (S_V2=", S_V2_STAGED, ")")
    ctx.synchronize()
    t0 = perf_counter()
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
    ctx.synchronize()
    timings.stage2_rope_seconds = perf_counter() - t0

    print("  [load] stage-2 globals")
    ctx.synchronize()
    t0 = perf_counter()
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
    ctx.synchronize()
    timings.stage2_load_globals_seconds = perf_counter() - t0

    # ══════════════ STAGE-2 refine (3-step res_2s) ══════════════
    ctx.synchronize()
    t0 = perf_counter()
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
    ctx.synchronize()
    timings.stage2_denoise_seconds = perf_counter() - t0

    # ══════════════ DECODE VIDEO (2x res) ══════════════
    ctx.synchronize()
    t0 = perf_counter()
    var vae = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
    var vx2_bf16 = cast_tensor(vx2, STDtype.BF16, ctx)
    var frames = decode_video[
        1, 128, NF_STAGED, NH2_STAGED, NW2_STAGED
    ](vae, vx2_bf16, ctx)
    var fsh = frames.shape()
    var n_frames_out = fsh[2]
    print("  frames NCDHW: [", fsh[0], ",", fsh[1], ",", fsh[2], ",", fsh[3],
          ",", fsh[4], "]")
    _ = _stats(String("frames"), frames, ctx)
    ctx.synchronize()
    timings.video_decode_seconds = perf_counter() - t0

    ctx.synchronize()
    t0 = perf_counter()
    for fr in range(n_frames_out):
        var fslice = slice(frames, 2, fr, 1, ctx)
        var fs = fslice.shape()
        var chw = reshape(fslice, _sh4(fs[0], fs[1], fs[3], fs[4]), ctx)
        save_png(chw, out_dir + "/" + frame_prefix + _pad2(fr) + ".png", ctx, ValueRange.SIGNED)
    print("  saved", n_frames_out, "frame PNGs ->", out_dir)
    ctx.synchronize()
    timings.frame_png_write_seconds = perf_counter() - t0

    if not include_audio:
        var mp4_video = out_dir + "/ltx2_t2v_hq2.mp4"
        if max_steps != 0:
            mp4_video = out_dir + "/ltx2_t2v_stage2_dev_smoke.mp4"
        print("  [decode] audio disabled by explicit noaudio; muxing video-only mp4")
        ctx.synchronize()
        t0 = perf_counter()
        _mux_video_mp4(out_dir, mp4_video, frame_prefix)
        timings.video_mux_seconds = perf_counter() - t0
        timings.total_runner_seconds = perf_counter() - total_t0
        timings.write_staged(
            out_dir + "/ltx2_runner_timings.json", include_audio, max_steps,
            n_frames_out, 0,
        )
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
    ctx.synchronize()
    t0 = perf_counter()
    var avae = LTX2AudioVaeDecoderWeights.load(_ckpt_bf16(), ctx)
    var audio_x_bf16 = cast_tensor(ax2, STDtype.BF16, ctx)
    var mel = cast_tensor(decode_audio(avae, audio_x_bf16, ctx), STDtype.BF16, ctx)
    ctx.synchronize()
    timings.audio_vae_seconds = perf_counter() - t0
    ctx.synchronize()
    t0 = perf_counter()
    var voc = LTX2VocoderWithBWE.from_file(_ckpt_bf16(), ctx)
    var wav = voc.forward(mel, ctx)
    ctx.synchronize()
    timings.vocoder_seconds = perf_counter() - t0
    t0 = perf_counter()
    var wsh = wav.shape()
    var L = wsh[2]
    var wh = wav.to_host(ctx)
    var inter = List[Float32]()
    inter.resize(L * 2, Float32(0.0))
    for ch in range(2):
        for s in range(L):
            inter[s * 2 + ch] = wh[ch * L + s]
    _write_wav(wav_out, inter, voc.output_sample_rate())
    timings.wav_write_seconds = perf_counter() - t0
    print("  wrote wav:", wav_out)
    t0 = perf_counter()
    _mux_mp4(out_dir, n_frames_out, wav_out, mp4_out, frame_prefix)
    timings.audio_mux_seconds = perf_counter() - t0
    timings.total_runner_seconds = perf_counter() - total_t0
    timings.write_staged(
        out_dir + "/ltx2_runner_timings.json", include_audio, max_steps,
        n_frames_out, voc.output_sample_rate(),
    )
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
#             strength 0.25 stage 1 / 0.5 stage 2, dense BF16 delta fused into
#             each transient streamed block like Creator's WeightsProvider.
#             Nothing is fused to disk. Globals get the additive delta at the
#             same per-stage strength.
#
# PARITY-GATE SHAPE (compile-time; matches the oracle run `--width 384
# --height 256 --num-frames 17 --steps 15 --seed 42` with fps 25):
#   stage 1: 192x128  -> NF=3 NH=4 NW=6  S_V=72 ; audio S_A=round(17/25*25)=17
#   stage 2: 384x256  -> NH=8 NW=12      S_V=288
#
# MEMORY CONTRACT (measured on the 16 GiB product GPU): the rank-384 distilled
# LoRA factor set is 7.6 GB and cannot be pinned beside the resident raw-FP8
# range. Keep the safetensors mmap (and therefore the Linux page cache), upload
# only the current module's A/B tensors, materialize its BF16 delta, add it to
# the current streamed block, then release it. This mirrors Creator's dense
# block fusion while bounding device LoRA memory to one module/block at a time.
#
# Dump contract (parity inputs for scripts/ltx2_hq_step_compare.py):
#   stage1_steps.safetensors : s1_s{NN}_video / s1_s{NN}_audio  (patchified,
#       state AFTER the step-level SDE injection, BF16-rounded, stored F32)
#   stage2_steps.safetensors : s2_s{NN}_*
#   upsampler.safetensors    : {in, out}   (latent NCDHW layout, F32)
#   final_latents.safetensors: {video, audio} (unpatchified latent layout, BF16)
#       The denoiser state is already BF16.  Preserve that storage dtype across
#       the fresh-process handoff instead of widening values that the decoder
#       immediately casts back to BF16.
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
# 121f @ 768x512 final (stage-1 384x256) — the production lattice-repro
# geometry, run through the reference-faithful refhq path. The 17f/768x512
# parity shapes matching the on-box golden reference
# (output/ltx2_hq_ref_golden) are:
#   NUM_FRAMES=17 NF=3 NH1=8 NW1=12 (S_V1=288) S_A=17 NH2=16 NW2=24
#   (S_V2=1152) SPAD=1152     [ltx2_refhq17_* runners were built with these]
# The original 1024x576 121f values: NH1=9 NW1=16 (S_V1=2304) NH2=18 NW2=32
# (S_V2=9216) SPAD=9216 — untested on the 16GB 5080.
# Pinned Creator LTX_2_3_HQ_PARAMS / Desktop 16 GB operating point:
# 121 frames @ 24 fps, stage-1 960x544, stage-2 1920x1088. A prior 241-frame
# experimental arm doubled stage-2 tokens to 63,240 and is not the Creator
# default memory contract; keep long-video admission separate from refhq.
comptime REFHQ_NUM_FRAMES = 121
comptime REFHQ_FPS = Float64(24.0)
comptime REFHQ_MUX_DURATION = "5.041667"  # 121 frames / 24 fps
comptime REFHQ_NF = 16           # (121-1)//8 + 1
comptime REFHQ_NH1 = 17          # ceil(544/32)
comptime REFHQ_NW1 = 30          # 960/32
comptime REFHQ_S_V1 = REFHQ_NF * REFHQ_NH1 * REFHQ_NW1   # 8160
comptime REFHQ_S_A = 126         # round(121/24*25) — AudioLatentShape.from_duration
comptime REFHQ_NH2 = 34          # 1088/32
comptime REFHQ_NW2 = 60          # 1920/32
comptime REFHQ_S_V2 = REFHQ_NF * REFHQ_NH2 * REFHQ_NW2   # 32640
comptime REFHQ_SPAD = 32640      # max(S_V*, N_TXT=1024, S_A)
comptime REFHQ_STEPS = 15
# HQ guider params (ltx_pipelines/utils/constants.py LTX_2_3_HQ_PARAMS)
comptime REFHQ_V_CFG = Float32(3.0)
comptime REFHQ_V_RESCALE = Float32(0.45)
comptime REFHQ_A_CFG = Float32(7.0)
comptime REFHQ_A_RESCALE = Float32(1.0)
comptime REFHQ_MOD_SCALE = Float32(3.0)
comptime REFHQ_LORA_S1 = Float32(0.25)
comptime REFHQ_LORA_S2 = Float32(0.5)

# Exact compiled product profile recovered from the visually verified Eri2
# step-3000 Creator run. Runtime requests must match these values; the adapter
# rejects everything else instead of silently substituting this profile.
comptime REQUEST_HQ_WIDTH = get_defined_int["LTX2_REQUEST_WIDTH", 512]()
comptime REQUEST_HQ_HEIGHT = get_defined_int["LTX2_REQUEST_HEIGHT", 768]()
comptime REQUEST_HQ_NUM_FRAMES = get_defined_int[
    "LTX2_REQUEST_FRAMES", 121
]()
comptime REQUEST_HQ_FPS_INT = get_defined_int["LTX2_REQUEST_FPS", 25]()
comptime REQUEST_HQ_FPS = Float64(REQUEST_HQ_FPS_INT)
comptime REQUEST_HQ_STEPS = 20
# Desktop rounds ordinary two-stage requests to the nearest 64 pixels before
# entering the pipeline (960x544 becomes 960x512). One-stage Retake/Extend
# preserves source-native 32-aligned geometry such as 960x544. Do not invent a
# larger latent grid and crop decoded pixels afterward.
comptime REQUEST_HQ_NF = (REQUEST_HQ_NUM_FRAMES - 1) // 8 + 1
comptime REQUEST_HQ_NH1 = REQUEST_HQ_HEIGHT // 64
comptime REQUEST_HQ_NW1 = REQUEST_HQ_WIDTH // 64
comptime REQUEST_HQ_S_V1 = REQUEST_HQ_NF * REQUEST_HQ_NH1 * REQUEST_HQ_NW1
# AudioLatentShape.from_video_pixel_shape uses 25 latent frames per second.
comptime REQUEST_HQ_S_A = (
    REQUEST_HQ_NUM_FRAMES * 25 + REQUEST_HQ_FPS_INT // 2
) // REQUEST_HQ_FPS_INT
comptime REQUEST_HQ_NH2 = REQUEST_HQ_HEIGHT // 32
comptime REQUEST_HQ_NW2 = REQUEST_HQ_WIDTH // 32
comptime REQUEST_HQ_S_V2 = REQUEST_HQ_NF * REQUEST_HQ_NH2 * REQUEST_HQ_NW2
comptime REQUEST_HQ_KEYFRAME_TOKENS1 = REQUEST_HQ_NH1 * REQUEST_HQ_NW1
comptime REQUEST_HQ_KEYFRAME_TOKENS2 = REQUEST_HQ_NH2 * REQUEST_HQ_NW2
comptime REQUEST_HQ_S_V1_KEYFRAME = (
    REQUEST_HQ_S_V1 + REQUEST_HQ_KEYFRAME_TOKENS1
)
comptime REQUEST_HQ_S_V2_KEYFRAME = (
    REQUEST_HQ_S_V2 + REQUEST_HQ_KEYFRAME_TOKENS2
)
# Flash-attention specialization pads sequence lengths to a 512-token boundary.
comptime REQUEST_HQ_VPAD1_BASE = (
    REQUEST_HQ_S_V1 if REQUEST_HQ_S_V1 > 1024 else 1024
)
comptime REQUEST_HQ_VPAD1 = (
    (REQUEST_HQ_VPAD1_BASE + 511) // 512
) * 512
comptime REQUEST_HQ_APAD_BASE = (
    REQUEST_HQ_S_A if REQUEST_HQ_S_A > 1024 else 1024
)
comptime REQUEST_HQ_APAD = (
    (REQUEST_HQ_APAD_BASE + 511) // 512
) * 512
comptime REQUEST_HQ_VPAD2_BASE = (
    REQUEST_HQ_S_V2 if REQUEST_HQ_S_V2 > 1024 else 1024
)
comptime REQUEST_HQ_VPAD2 = (
    (REQUEST_HQ_VPAD2_BASE + 511) // 512
) * 512
comptime REQUEST_HQ_VPAD1_KEYFRAME_BASE = (
    REQUEST_HQ_S_V1_KEYFRAME
    if REQUEST_HQ_S_V1_KEYFRAME > 1024 else 1024
)
comptime REQUEST_HQ_VPAD1_KEYFRAME = (
    (REQUEST_HQ_VPAD1_KEYFRAME_BASE + 511) // 512
) * 512
comptime REQUEST_HQ_VPAD2_KEYFRAME_BASE = (
    REQUEST_HQ_S_V2_KEYFRAME
    if REQUEST_HQ_S_V2_KEYFRAME > 1024 else 1024
)
comptime REQUEST_HQ_VPAD2_KEYFRAME = (
    (REQUEST_HQ_VPAD2_KEYFRAME_BASE + 511) // 512
) * 512


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


def _build_video_coords_with_keyframe_fps(
    nf: Int,
    nh: Int,
    nw: Int,
    fps: Float64,
    frame_idx: Int,
) raises -> List[Float64]:
    """Append one creator keyframe guide grid to the generated-token RoPE.

    This is the exact `VideoConditionByKeyframeIndex` coordinate contract:
    generated tokens retain their ordinary causal-VAE bounds, while the
    appended single-image guide uses non-causal spatial bounds and the narrow
    temporal interval `[frame_idx, frame_idx + 1) / fps`.
    """
    if frame_idx <= 0:
        raise Error("LTX2 keyframe guide index must be greater than zero")
    var target = _build_video_coords_fps(nf, nh, nw, fps)
    var target_tokens = nf * nh * nw
    var guide_tokens = nh * nw
    var total_tokens = target_tokens + guide_tokens
    var out = List[Float64]()
    out.resize(3 * total_tokens * 2, Float64(0.0))
    for d in range(3):
        for tok in range(target_tokens):
            out[(d * total_tokens + tok) * 2 + 0] = (
                target[(d * target_tokens + tok) * 2 + 0]
            )
            out[(d * total_tokens + tok) * 2 + 1] = (
                target[(d * target_tokens + tok) * 2 + 1]
            )
    for h in range(nh):
        for w in range(nw):
            var tok = target_tokens + h * nw + w
            out[(0 * total_tokens + tok) * 2 + 0] = (
                Float64(frame_idx) / fps
            )
            out[(0 * total_tokens + tok) * 2 + 1] = (
                Float64(frame_idx + 1) / fps
            )
            out[(1 * total_tokens + tok) * 2 + 0] = Float64(h) * VAE_SF1
            out[(1 * total_tokens + tok) * 2 + 1] = Float64(h + 1) * VAE_SF1
            out[(2 * total_tokens + tok) * 2 + 0] = Float64(w) * VAE_SF2
            out[(2 * total_tokens + tok) * 2 + 1] = Float64(w + 1) * VAE_SF2
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


def _request_hq_video_rope(
    nf: Int, nh: Int, nw: Int, s_v: Int, ctx: DeviceContext
) raises -> _RefhqRope:
    var vc = _build_video_coords_fps(nf, nh, nw, REQUEST_HQ_FPS)
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


def _request_hq_video_rope_with_keyframe(
    nf: Int,
    nh: Int,
    nw: Int,
    s_v: Int,
    frame_idx: Int,
    ctx: DeviceContext,
) raises -> _RefhqRope:
    var vc = _build_video_coords_with_keyframe_fps(
        nf, nh, nw, REQUEST_HQ_FPS, frame_idx
    )
    var vtc = _video_temporal_coords_dims(vc, s_v)
    var vrope = _compute_rope(
        vc, 3, s_v, VD, _mp3(), ROPE_THETA, V_HEADS, STDtype.BF16, ctx
    )
    var cav = _compute_rope(
        vtc, 1, s_v, CA_DIM, _mp1(), ROPE_THETA, V_HEADS,
        STDtype.BF16, ctx,
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
# the Creator-parity dense per-block path at `lora_mult`.
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
        # The dev-FP8 and distilled-FP8 checkpoints have different BF16 block
        # boundaries. The stream inspects each tensor's actual storage dtype,
        # so route every refhq block through it instead of applying the legacy
        # distilled `_is_boundary` map to the dev quality checkpoint.
        var blk = stream.load_block_bf16(i, ctx)
        var w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx)
        if lora_mult != Float32(0.0):
            n_lora_total += lora.apply_to_av_block(i, w, lora_mult, ctx)
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
        var cloned = _clone_pair(outs[0], outs[1], ctx)
        cloned^.move_into(hs, ahs)
    if verbose_lora:
        print(
            "  [refhq] dense LoRA apply count (48 blocks @ mult",
            lora_mult, "):", n_lora_total,
        )
    var v_vel = _output_stage(hs, g.v_sst, mod.v_embedded, g.v_pout_w,
                              g.v_pout_b, VD, ctx)
    var a_vel = _output_stage(ahs, g.a_sst, mod.a_embedded, g.a_pout_w,
                              g.a_pout_b, AD, ctx)
    return (v_vel^, a_vel^)


# Creator-HQ forward with the request-owned trained-LoRA stack. The existing
# refhq parity function above stays frozen for its historical one-LoRA gates.
def _request_hq_forward_flat[
    S_V_CT: Int, S_A_CT: Int, S_VPAD_CT: Int, S_APAD_CT: Int
](
    loras: _RequestHQLoraStack,
    distilled_mult: Float32,
    cfg: LTX2Config,
    g: _Globals,
    mut stream: LTX2BlockStream,
    v_flat: Tensor, a_flat: Tensor,
    enc: Tensor, aenc: Tensor,
    mod: _Mod,
    vr: _RefhqRope,
    a_cos: Tensor, a_sin: Tensor, ca_a_cos: Tensor, ca_a_sin: Tensor,
    skip_cross_modal: Bool,
    verbose_lora: Bool,
    factorized_lora: Bool,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var hs = _linear_b(v_flat, g.v_pin_w, g.v_pin_b, ctx)
    var ahs = _linear_b(a_flat, g.a_pin_w, g.a_pin_b, ctx)
    var n_lora_total = 0
    for i in range(NUM_LAYERS):
        var blk = stream.load_block_bf16_product(i, ctx)
        var w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx)
        if factorized_lora:
            n_lora_total += loras.attach_to_av_block(
                i, w, distilled_mult, ctx
            )
        else:
            n_lora_total += loras.apply_to_av_block(
                i, w, distilled_mult, ctx
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
        var cloned = _clone_pair(outs[0], outs[1], ctx)
        cloned^.move_into(hs, ahs)
    if verbose_lora:
        print("  [request-hq] factorized LoRA attachments:", n_lora_total,
              " support multiplier:", distilled_mult)
    var v_vel = _output_stage(
        hs, g.v_sst, mod.v_embedded, g.v_pout_w, g.v_pout_b, VD, ctx
    )
    var a_vel = _output_stage(
        ahs, g.a_sst, mod.a_embedded, g.a_pout_w, g.a_pout_b, AD, ctx
    )
    return (v_vel^, a_vel^)


# Direct-checkpoint forward used by LTX Desktop-style Retake/Extend. The
# creator's Retake pipeline loads a distilled transformer checkpoint directly;
# it does not merge the two-stage support LoRA into a dev model.
def _request_edit_forward_flat[
    S_V_CT: Int, S_A_CT: Int, S_VPAD_CT: Int, S_APAD_CT: Int
](
    loras: _RequestHQLoraStack,
    distillation_mult: Float32,
    cfg: LTX2Config,
    g: _Globals,
    mut stream: LTX2BlockStream,
    v_flat: Tensor, a_flat: Tensor,
    enc: Tensor, aenc: Tensor,
    mod: _Mod,
    vr: _RefhqRope,
    a_cos: Tensor, a_sin: Tensor, ca_a_cos: Tensor, ca_a_sin: Tensor,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var hs = _linear_b(v_flat, g.v_pin_w, g.v_pin_b, ctx)
    var ahs = _linear_b(a_flat, g.a_pin_w, g.a_pin_b, ctx)
    for i in range(NUM_LAYERS):
        var blk = stream.load_block_bf16_product(i, ctx)
        var w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx)
        if distillation_mult != Float32(0.0):
            _ = loras.apply_to_av_block(
                i, w, distillation_mult, ctx
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
            False,
        )
        var cloned = _clone_pair(outs[0], outs[1], ctx)
        cloned^.move_into(hs, ahs)
    var v_vel = _output_stage(
        hs, g.v_sst, mod.v_embedded, g.v_pout_w, g.v_pout_b, VD, ctx
    )
    var a_vel = _output_stage(
        ahs, g.a_sst, mod.a_embedded, g.a_pout_w, g.a_pout_b, AD, ctx
    )
    return (v_vel^, a_vel^)


# ── STEP-1 DUMP instrumentation (divergence hunt, 2026-07-10) ────────────────
# Same math as _refhq_forward_flat, but host-dumps (as F32) the post-patchify
# tokens and the video/audio hidden states after blocks 0,1,2,8,24,47 into the
# provided lists, prefixed `p{pass}_`. Compared against the torch-hook dump
# from scripts/ltx2_ref_step1_dump.py by scripts/ltx2_step1_compare.py.
def _dump_f32(
    mut names: List[String],
    mut tensors: List[ArcPointer[Tensor]],
    name: String,
    t: Tensor,
    ctx: DeviceContext,
) raises:
    names.append(name)
    tensors.append(ArcPointer[Tensor](cast_tensor(t, STDtype.F32, ctx)))


def _is_dump_block(i: Int) -> Bool:
    return i == 0 or i == 1 or i == 2 or i == 8 or i == 24 or i == 47


def _refhq_forward_flat_dump[
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
    pass_tag: String,
    mut dnames: List[String],
    mut dtensors: List[ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var hs = _linear_b(v_flat, g.v_pin_w, g.v_pin_b, ctx)
    var ahs = _linear_b(a_flat, g.a_pin_w, g.a_pin_b, ctx)
    _dump_f32(dnames, dtensors, pass_tag + "_v_x", hs, ctx)
    _dump_f32(dnames, dtensors, pass_tag + "_a_x", ahs, ctx)
    for i in range(NUM_LAYERS):
        var blk = stream.load_block_bf16(i, ctx)
        var w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx)
        if lora_mult != Float32(0.0):
            _ = lora.apply_to_av_block(i, w, lora_mult, ctx)
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
        var cloned = _clone_pair(outs[0], outs[1], ctx)
        cloned^.move_into(hs, ahs)
        if _is_dump_block(i):
            _dump_f32(dnames, dtensors, pass_tag + "_blk" + _pad2(i) + "_v", hs, ctx)
            _dump_f32(dnames, dtensors, pass_tag + "_blk" + _pad2(i) + "_a", ahs, ctx)
    var v_vel = _output_stage(hs, g.v_sst, mod.v_embedded, g.v_pout_w,
                              g.v_pout_b, VD, ctx)
    var a_vel = _output_stage(ahs, g.a_sst, mod.a_embedded, g.a_pout_w,
                              g.a_pout_b, AD, ctx)
    _dump_f32(dnames, dtensors, pass_tag + "_vel_v", v_vel, ctx)
    _dump_f32(dnames, dtensors, pass_tag + "_vel_a", a_vel, ctx)
    return (v_vel^, a_vel^)


# X0Model conversion (ltx-core utils.py to_denoised): x0 = x - v*sigma,
# computed in F32 and cast back to the sample dtype (BF16) — applied PER PASS
# before the guider combine, exactly like X0Model wraps the velocity model.
def _refhq_x0(x: Tensor, vel: Tensor, sigma: Float32, ctx: DeviceContext) raises -> Tensor:
    var xf = cast_tensor(x, STDtype.F32, ctx)
    var vf = cast_tensor(vel, STDtype.F32, ctx)
    return cast_tensor(sub(xf, mul_scalar(vf, sigma, ctx), ctx), STDtype.BF16, ctx)


def _request_edit_euler_step_x0(
    sample: Tensor,
    denoised: Tensor,
    sigma: Float32,
    sigma_next: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Creator EulerDiffusionStep from an already post-processed x0 estimate.

    LTX Desktop freezes source tokens by applying `post_process_latent` to x0
    before this update. Preserve its eager F32 arithmetic and BF16 storage:
      velocity = (sample - denoised) / sigma
      next = sample + velocity * (sigma_next - sigma)
    """
    if sigma <= Float32(0.0):
        raise Error("LTX2 edit Euler step requires sigma > 0")
    var sample_f32 = cast_tensor(sample, STDtype.F32, ctx)
    var denoised_f32 = cast_tensor(denoised, STDtype.F32, ctx)
    var velocity = mul_scalar(
        sub(sample_f32, denoised_f32, ctx), Float32(1.0) / sigma, ctx
    )
    var next_f32 = add(
        sample_f32,
        mul_scalar(velocity, sigma_next - sigma, ctx),
        ctx,
    )
    return cast_tensor(next_f32, STDtype.BF16, ctx)


# First-frame I2V conditionings are encoded independently at both stage
# resolutions, matching combined_image_conditionings in the official
# two-stage pipeline. Keeping the VAE load inside this helper makes its weights
# leave scope before the streamed 22B transformer is opened.
def _request_hq_encode_i2v_first_frames(
    image_path: String, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    var image = decode_image_any(image_path)
    var stage1_width = REQUEST_HQ_NW1 * 32
    var stage1_height = REQUEST_HQ_NH1 * 32
    var stage2_width = REQUEST_HQ_NW2 * 32
    var stage2_height = REQUEST_HQ_NH2 * 32
    var vae = LTX2VaeEncoderWeights.load(_ckpt_bf16(), ctx)

    var s1_values = image_bilinear_center_crop_to_signed_nchw(
        image, stage1_width, stage1_height
    )
    var s1_pixels = Tensor.from_host(
        s1_values, _sh5(1, 3, 1, stage1_height, stage1_width),
        STDtype.BF16, ctx,
    )
    var s1_latent = encode_conditioning_video(vae, s1_pixels, ctx)
    var s1_tokens = _refhq_patchify_video(
        s1_latent, REQUEST_HQ_NH1 * REQUEST_HQ_NW1, ctx
    )

    var s2_values = image_bilinear_center_crop_to_signed_nchw(
        image, stage2_width, stage2_height
    )
    var s2_pixels = Tensor.from_host(
        s2_values, _sh5(1, 3, 1, stage2_height, stage2_width),
        STDtype.BF16, ctx,
    )
    var s2_latent = encode_conditioning_video(vae, s2_pixels, ctx)
    var s2_tokens = _refhq_patchify_video(
        s2_latent, REQUEST_HQ_NH2 * REQUEST_HQ_NW2, ctx
    )
    ctx.synchronize()
    return (
        cast_tensor(s1_tokens, STDtype.BF16, ctx),
        cast_tensor(s2_tokens, STDtype.BF16, ctx),
    )


def _request_hq_transcode_v2v_source(
    video_path: String,
    raw_path: String,
    width: Int,
    height: Int,
    edit_mode: String,
    source_frames: Int,
) raises:
    var vf = (
        String("fps=") + String(REQUEST_HQ_FPS_INT)
        + String(",scale=") + String(width) + String(":") + String(height)
        + String(":force_original_aspect_ratio=increase,crop=")
        + String(width) + String(":") + String(height)
        + String(",setsar=1")
    )
    var pad_frames = REQUEST_HQ_NUM_FRAMES - source_frames
    if edit_mode == String("extend_start"):
        vf += (
            String(",trim=end_frame=") + String(source_frames)
            + String(",tpad=start_mode=clone:start=") + String(pad_frames)
        )
    elif edit_mode == String("extend_end"):
        vf += (
            String(",trim=end_frame=") + String(source_frames)
            + String(",tpad=stop_mode=clone:stop=") + String(pad_frames)
        )
    var cmd = (
        String("ffmpeg -y -hide_banner -loglevel error -i ")
        + shell_quote(video_path)
        + String(" -an -sn -dn -vf ") + shell_quote(vf)
        + String(" -frames:v ") + String(REQUEST_HQ_NUM_FRAMES)
        + String(" -f rawvideo -pix_fmt rgb24 ") + shell_quote(raw_path)
    )
    var rc = sys_system(cmd)
    if rc != 0:
        raise Error(
            String("LTX2 V2V: ffmpeg source decode failed with raw status ")
            + String(rc)
        )


def _request_hq_transcode_creator_edit_source(
    video_path: String,
    raw_path: String,
    width: Int,
    height: Int,
    source_frames: Int,
) raises:
    """Decode only the real Retake/Extend source; never synthesize edge pixels."""
    if source_frames <= 1:
        raise Error("LTX2 creator temporal edit source has no frames")
    var vf = (
        String("scale=") + String(width) + String(":") + String(height)
        + String(":force_original_aspect_ratio=increase,crop=")
        + String(width) + String(":") + String(height)
        + String(",setsar=1")
    )
    var cmd = (
        String("ffmpeg -y -hide_banner -loglevel error -i ")
        + shell_quote(video_path)
        + String(" -an -sn -dn -vf ") + shell_quote(vf)
        + String(" -frames:v ") + String(source_frames)
        + String(" -f rawvideo -pix_fmt rgb24 ") + shell_quote(raw_path)
    )
    var rc = sys_system(cmd)
    if rc != 0:
        raise Error(
            String("LTX2 creator temporal source decode failed with raw status ")
            + String(rc)
        )


def _request_creator_tile_count(
    dimension: Int, tile_size: Int, overlap: Int
) -> Int:
    if dimension <= tile_size:
        return 1
    return (
        dimension + tile_size - 2 * overlap - 1
    ) // (tile_size - overlap)


def _request_hq_encode_creator_tiled_video(
    pixels: Tensor,
    checkpoint_path: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """LTX Desktop Retake's exact 256/64 spatial + 24/16 temporal VAE tiles.

    Creator encoding uses rectangular latent masks spatially (discarding the
    boundary latent affected by per-tile convolution padding) and overlapping
    temporal tiles whose retained causal frames are BF16-accumulated and
    normalized by their contribution count.
    """
    comptime TILE_SPATIAL = 256
    comptime OVERLAP_SPATIAL = 64
    comptime TILE_TEMPORAL = 24
    comptime OVERLAP_TEMPORAL = 16
    comptime SCALE_SPATIAL = 32
    comptime SCALE_TEMPORAL = 8

    var shape = pixels.shape()
    if len(shape) != 5 or shape[0] != 1 or shape[1] != 3:
        raise Error("LTX2 creator tiled encode requires [1,3,T,H,W]")
    var frames = shape[2]
    var height = shape[3]
    var width = shape[4]
    if (frames - 1) % SCALE_TEMPORAL != 0:
        raise Error("LTX2 creator tiled encode requires 8k+1 frames")
    if height % SCALE_SPATIAL != 0 or width % SCALE_SPATIAL != 0:
        raise Error("LTX2 creator tiled encode requires 32-aligned geometry")

    var latent_frames = (frames - 1) // SCALE_TEMPORAL + 1
    var latent_height = height // SCALE_SPATIAL
    var latent_width = width // SCALE_SPATIAL
    var temporal_tiles = _request_creator_tile_count(
        frames, TILE_TEMPORAL, OVERLAP_TEMPORAL
    )
    var height_tiles = _request_creator_tile_count(
        height, TILE_SPATIAL, OVERLAP_SPATIAL
    )
    var width_tiles = _request_creator_tile_count(
        width, TILE_SPATIAL, OVERLAP_SPATIAL
    )
    print(
        "  [Temporal edit VAE] creator tiles:", temporal_tiles, "x",
        height_tiles, "x", width_tiles,
        "(24/16 frames, 256/64 pixels)",
    )

    var accumulators = List[ArcPointer[Tensor]]()
    var counts = List[Int]()
    for _ in range(latent_frames):
        accumulators.append(
            ArcPointer[Tensor](
                zeros_device(
                    _sh5(1, 128, 1, latent_height, latent_width),
                    STDtype.BF16,
                    ctx,
                )
            )
        )
        counts.append(0)

    var vae = LTX2VaeEncoderWeights.load(checkpoint_path, ctx)
    var spatial_stride = TILE_SPATIAL - OVERLAP_SPATIAL
    var temporal_stride = TILE_TEMPORAL - OVERLAP_TEMPORAL
    for temporal_index in range(temporal_tiles):
        var temporal_start = temporal_index * temporal_stride
        var temporal_end = (
            frames
            if temporal_index == temporal_tiles - 1
            else temporal_start + TILE_TEMPORAL + 1
        )
        var temporal_pixels = slice(
            pixels, 2, temporal_start, temporal_end - temporal_start, ctx
        )
        var complete_tile: Optional[Tensor] = None
        for height_index in range(height_tiles):
            var height_start = height_index * spatial_stride
            var height_end = (
                height
                if height_index == height_tiles - 1
                else height_start + TILE_SPATIAL
            )
            var height_pixels = slice(
                temporal_pixels, 3, height_start, height_end - height_start, ctx
            )
            var complete_row: Optional[Tensor] = None
            for width_index in range(width_tiles):
                var width_start = width_index * spatial_stride
                var width_end = (
                    width
                    if width_index == width_tiles - 1
                    else width_start + TILE_SPATIAL
                )
                var video_tile = slice(
                    height_pixels, 4, width_start, width_end - width_start, ctx
                )
                var latent_tile = encode_conditioning_video(
                    vae, video_tile, ctx
                )
                var latent_shape = latent_tile.shape()
                var crop_top = (
                    0
                    if height_index == 0
                    else OVERLAP_SPATIAL // SCALE_SPATIAL - 1
                )
                var crop_bottom = (
                    0 if height_index == height_tiles - 1 else 1
                )
                var crop_left = (
                    0
                    if width_index == 0
                    else OVERLAP_SPATIAL // SCALE_SPATIAL - 1
                )
                var crop_right = (
                    0 if width_index == width_tiles - 1 else 1
                )
                var cropped = slice(
                    latent_tile,
                    3,
                    crop_top,
                    latent_shape[3] - crop_bottom - crop_top,
                    ctx,
                )
                cropped = slice(
                    cropped,
                    4,
                    crop_left,
                    latent_shape[4] - crop_right - crop_left,
                    ctx,
                )
                if width_index == 0:
                    complete_row = Optional[Tensor](cropped^)
                else:
                    var joined_row = concat(
                        4, ctx, complete_row.value(), cropped
                    )
                    complete_row = Optional[Tensor](joined_row^)
            var completed_row = complete_row.value().clone(ctx)
            if completed_row.shape()[4] != latent_width:
                raise Error(
                    "LTX2 creator tiled encode reconstructed wrong latent width"
                )
            if height_index == 0:
                complete_tile = Optional[Tensor](completed_row^)
            else:
                var joined_tile = concat(
                    3, ctx, complete_tile.value(), completed_row
                )
                complete_tile = Optional[Tensor](joined_tile^)
        var completed_tile = complete_tile.value().clone(ctx)
        var complete_shape = completed_tile.shape()
        if complete_shape[3] != latent_height \
            or complete_shape[4] != latent_width:
            raise Error(
                "LTX2 creator tiled encode reconstructed wrong spatial shape"
            )

        var output_start = temporal_start // SCALE_TEMPORAL
        var left_ramp = (
            0
            if temporal_index == 0
            else 1 + (OVERLAP_TEMPORAL - 1) // SCALE_TEMPORAL
        )
        for local_frame in range(complete_shape[2]):
            if local_frame < left_ramp:
                continue
            var output_frame = output_start + local_frame
            if output_frame < 0 or output_frame >= latent_frames:
                raise Error(
                    "LTX2 creator tiled encode temporal mapping out of range"
            )
            var contribution = slice(
                completed_tile, 2, local_frame, 1, ctx
            )
            var accumulated = add(
                accumulators[output_frame][], contribution, ctx
            )
            accumulators[output_frame] = ArcPointer[Tensor](accumulated^)
            counts[output_frame] += 1

    var final_latent: Optional[Tensor] = None
    for frame in range(latent_frames):
        if counts[frame] <= 0:
            raise Error(
                "LTX2 creator tiled encode left an uncovered latent frame"
            )
        var normalized = accumulators[frame][].clone(ctx)
        if counts[frame] > 1:
            normalized = mul_scalar(
                normalized, Float32(1.0) / Float32(counts[frame]), ctx
            )
        if frame == 0:
            final_latent = Optional[Tensor](normalized^)
        else:
            var joined_latent = concat(
                2, ctx, final_latent.value(), normalized
            )
            final_latent = Optional[Tensor](joined_latent^)
    return final_latent.value().clone(ctx)


def _request_hq_encode_v2v_stage(
    video_path: String,
    out_dir: String,
    width: Int,
    height: Int,
    nh: Int,
    nw: Int,
    tag: String,
    edit_mode: String,
    source_frames: Int,
    progress_total: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var raw_path = out_dir + String("/.ltx2_v2v_") + tag + String(".rgb24")
    _request_hq_transcode_v2v_source(
        video_path, raw_path, width, height, edit_mode, source_frames
    )
    _write_ltx2_status(
        out_dir,
        String("running"),
        String("encoding_source"),
        0,
        progress_total,
        String("VAE encoding V2V source at ")
        + String(width) + String("x") + String(height),
    )
    var pixels = raw_rgb24_video_to_signed_ncdhw_bf16_device(
        raw_path,
        REQUEST_HQ_NUM_FRAMES,
        width,
        height,
        ctx,
    )
    var remove_rc = sys_system(
        String("rm -f -- ") + shell_quote(raw_path)
    )
    if remove_rc != 0:
        print("  [V2V] warning: could not remove temporary RGB stream:", raw_path)
    var vae = LTX2VaeEncoderWeights.load(_ckpt_bf16(), ctx)
    var latent = encode_conditioning_video(vae, pixels, ctx)
    var tokens = _refhq_patchify_video(
        latent, REQUEST_HQ_NF * nh * nw, ctx
    )
    ctx.synchronize()
    return cast_tensor(tokens, STDtype.BF16, ctx)


def _request_hq_encode_v2v_video(
    video_path: String,
    out_dir: String,
    edit_mode: String,
    source_frames: Int,
    progress_total: Int,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """Encode once at stage 1, then use the official latent x2 upscaler.

    Running the complete 121-frame VAE encoder independently at both spatial
    grids made source preparation slower than the denoiser itself. Stage 2 of
    this pipeline already derives its clean latent with the official LTX2 x2
    latent upscaler, so use that same model for the source conditioning.
    """
    var stage1 = _request_hq_encode_v2v_stage(
        video_path,
        out_dir,
        REQUEST_HQ_NW1 * 32,
        REQUEST_HQ_NH1 * 32,
        REQUEST_HQ_NH1,
        REQUEST_HQ_NW1,
        String("stage1"),
        edit_mode,
        source_frames,
        progress_total,
        ctx,
    )
    _write_ltx2_status(
        out_dir,
        String("running"),
        String("upscaling_source"),
        0,
        progress_total,
        String("Upscaling V2V source latents for stage 2"),
    )
    var stage1_latent = _refhq_unpatchify_video(
        stage1,
        REQUEST_HQ_NF,
        REQUEST_HQ_NH1,
        REQUEST_HQ_NW1,
        ctx,
    )
    var stage2_latent = _refhq_spatial_upscale_scoped(stage1_latent, ctx)
    var stage2 = cast_tensor(
        _refhq_patchify_video(stage2_latent, REQUEST_HQ_S_V2, ctx),
        STDtype.BF16,
        ctx,
    )
    return (stage1^, stage2^)


def _request_hq_encode_retake_video(
    video_path: String,
    out_dir: String,
    edit_mode: String,
    source_frames: Int,
    progress_total: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Encode Retake/Extend source directly at final resolution with creator tiles.

    LTX Desktop's Retake pipeline does not run the two-stage spatial latent
    upscaler. Its ImageConditioner encodes the source clip at the requested
    output width and height using 256/64 spatial and 24/16 temporal tiles.
    Extend then zero-pads that encoded latent at the requested end, exactly as
    `_pad_latent_frames` does in the creator pipeline.
    """
    var width = REQUEST_HQ_NW2 * 32
    var height = REQUEST_HQ_NH2 * 32
    var edit_label = String("Extend") if (
        edit_mode == String("extend_start")
        or edit_mode == String("extend_end")
    ) else String("Retake")
    var raw_path = out_dir + String("/.ltx2_temporal_edit_creator.rgb24")
    _request_hq_transcode_creator_edit_source(
        video_path, raw_path, width, height, source_frames
    )
    _write_ltx2_status(
        out_dir,
        String("running"),
        String("encoding_source"),
        0,
        progress_total,
        String("Creator-tiled VAE encoding ") + edit_label
        + String(" source at ")
        + String(width) + String("x") + String(height),
    )
    var pixels = raw_rgb24_video_to_signed_ncdhw_bf16_device(
        raw_path,
        source_frames,
        width,
        height,
        ctx,
    )
    var remove_rc = sys_system(
        String("rm -f -- ") + shell_quote(raw_path)
    )
    if remove_rc != 0:
        print(
            "  [Temporal edit VAE] warning: could not remove temporary RGB stream:",
            raw_path,
        )
    var request_checkpoint = _request_hq_ckpt(String("bf16"))
    var vae_checkpoint = _request_hq_video_vae_ckpt(request_checkpoint)
    print("  [Temporal edit VAE] complete checkpoint:", vae_checkpoint)
    var latent = _request_hq_encode_creator_tiled_video(
        pixels, vae_checkpoint, ctx
    )
    var source_nf = (source_frames - 1) // 8 + 1
    if latent.shape()[2] != source_nf:
        raise Error("LTX2 creator temporal source latent frame mismatch")
    var pad_frames = REQUEST_HQ_NF - source_nf
    if pad_frames < 0:
        raise Error("LTX2 creator temporal source exceeds target latent")
    if pad_frames > 0:
        var pad = zeros_device(
            _sh5(
                1, 128, pad_frames,
                REQUEST_HQ_NH2, REQUEST_HQ_NW2,
            ),
            latent.dtype(),
            ctx,
        )
        if edit_mode == String("extend_start"):
            latent = concat(2, ctx, pad, latent)
        elif edit_mode == String("extend_end"):
            latent = concat(2, ctx, latent, pad)
        else:
            raise Error(
                "LTX2 temporal edit source unexpectedly needs temporal padding"
            )
    var tokens = _refhq_patchify_video(
        latent, REQUEST_HQ_S_V2, ctx
    )
    ctx.synchronize()
    return cast_tensor(tokens, STDtype.BF16, ctx)


def _request_hq_i2v_mask(
    sequence_tokens: Int,
    frame_tokens: Int,
    conditioned_denoise: Float32,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    if frame_tokens <= 0 or frame_tokens >= sequence_tokens:
        raise Error("LTX2 I2V: invalid frame-zero token span")
    var prefix = full_device(
        _sh3(1, frame_tokens, 1),
        conditioned_denoise * scale, STDtype.F32, ctx,
    )
    var suffix = full_device(
        _sh3(1, sequence_tokens - frame_tokens, 1),
        scale, STDtype.F32, ctx,
    )
    return concat(1, ctx, prefix, suffix)


def _request_hq_v2v_mask(
    sequence_tokens: Int,
    conditioned_denoise: Float32,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    if sequence_tokens <= 0:
        raise Error("LTX2 V2V: invalid sequence token count")
    return full_device(
        _sh3(1, sequence_tokens, 1),
        conditioned_denoise * scale,
        STDtype.F32,
        ctx,
    )


def _request_hq_v2v_spatial_mask_values(
    mask_path: String,
    nf: Int,
    nh: Int,
    nw: Int,
    conditioned_denoise: Float32,
) raises -> List[Float32]:
    """Downsample a painted image mask onto LTX2's patchified video grid.

    The Canvas convention is white=edit and black=preserve. LTX2's official
    denoise mask has the same direction: zero preserves the clean source while
    larger values admit more noise. Max pooling makes a thin painted stroke
    survive the 32-pixel latent cell reduction, and the static image is tiled
    through all latent frames.
    """
    if nf <= 0 or nh <= 0 or nw <= 0:
        raise Error("LTX2 V2V mask: latent grid dimensions must be positive")
    if conditioned_denoise < Float32(0.0) or conditioned_denoise > Float32(1.0):
        raise Error("LTX2 V2V mask: conditioned denoise must be in [0, 1]")
    var image = decode_image_any(mask_path)
    var px: List[Int] = [0, 0, 0]
    var spatial = List[Float32]()
    var edited_cells = 0
    for h in range(nh):
        var y0 = (h * image.height) // nh
        var y1 = ((h + 1) * image.height + nh - 1) // nh
        if y1 <= y0:
            y1 = y0 + 1
        if y1 > image.height:
            y1 = image.height
        for w in range(nw):
            var x0 = (w * image.width) // nw
            var x1 = ((w + 1) * image.width + nw - 1) // nw
            if x1 <= x0:
                x1 = x0 + 1
            if x1 > image.width:
                x1 = image.width
            var peak = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    image_rgb_at(image, x, y, px)
                    var value = max(px[0], max(px[1], px[2]))
                    if value > peak:
                        peak = value
            var edit = Float32(peak) / Float32(255.0)
            if edit > Float32(0.0):
                edited_cells += 1
            spatial.append(edit * conditioned_denoise)
    if edited_cells == 0:
        raise Error("LTX2 V2V mask: mask contains no editable pixels")
    var out = List[Float32]()
    for _ in range(nf):
        for value in spatial:
            out.append(value)
    return out^


def _request_hq_v2v_temporal_mask_values(
    nf: Int,
    nh: Int,
    nw: Int,
    fps: Float32,
    start_time: Float32,
    end_time: Float32,
    conditioned_denoise: Float32,
) raises -> List[Float32]:
    """Expand a Retake/Extend time window across the patchified video grid.

    LTX2's video VAE has temporal factor 8. Each patchified latent frame owns
    every spatial token at that time coordinate. Zero freezes the clean source;
    one fully regenerates the selected time coordinate.
    """
    if nf <= 0 or nh <= 0 or nw <= 0 or fps <= Float32(0.0):
        raise Error("LTX2 temporal mask: invalid latent grid or FPS")
    if start_time < Float32(0.0) or end_time <= start_time:
        raise Error("LTX2 temporal mask: invalid edit window")
    if (
        conditioned_denoise < Float32(0.0)
        or conditioned_denoise > Float32(1.0)
    ):
        raise Error("LTX2 temporal mask: denoise must be in [0, 1]")
    var out = List[Float32]()
    var edited_frames = 0
    for f in range(nf):
        # Match LTX Core's TemporalRegionMask exactly. Its patch grid contains
        # latent bounds [f, f+1); get_pixel_coords applies temporal scale 8 and
        # the causal VAE fix `+1-8`, clamped at zero. A token is selected when
        # its pixel-time interval overlaps [start_time, end_time).
        var pixel_start = f * 8 + 1 - 8
        var pixel_end = (f + 1) * 8 + 1 - 8
        if pixel_start < 0:
            pixel_start = 0
        if pixel_end < 0:
            pixel_end = 0
        var token_start = Float32(pixel_start) / fps
        var token_end = Float32(pixel_end) / fps
        var edit = Float32(0.0)
        if token_end > start_time and token_start < end_time:
            edit = conditioned_denoise
            edited_frames += 1
        for _ in range(nh * nw):
            out.append(edit)
    if edited_frames == 0:
        raise Error("LTX2 temporal mask: edit window selects no latent frames")
    return out^


def _request_hq_audio_temporal_mask_values(
    frames: Int,
    start_time: Float32,
    end_time: Float32,
    conditioned_denoise: Float32,
) raises -> List[Float32]:
    """Match AudioPatchifier's causal time bounds for TemporalRegionMask."""
    if frames <= 0:
        raise Error("LTX2 audio temporal mask: invalid latent frame count")
    if start_time < Float32(0.0) or end_time <= start_time:
        raise Error("LTX2 audio temporal mask: invalid edit window")
    if (
        conditioned_denoise < Float32(0.0)
        or conditioned_denoise > Float32(1.0)
    ):
        raise Error("LTX2 audio temporal mask: denoise must be in [0, 1]")
    var out = List[Float32]()
    var edited_frames = 0
    for frame in range(frames):
        # AudioPatchifier maps latent index i through a 4x causal mel stride,
        # hop_length=160, sample_rate=16000:
        #   start=max(4*i-3, 0)*0.01
        #   end=max(4*(i+1)-3, 0)*0.01
        var sample_start = frame * 4 - 3
        var sample_end = (frame + 1) * 4 - 3
        if sample_start < 0:
            sample_start = 0
        if sample_end < 0:
            sample_end = 0
        var token_start = Float32(sample_start) * Float32(0.01)
        var token_end = Float32(sample_end) * Float32(0.01)
        var edit = Float32(0.0)
        if token_end > start_time and token_start < end_time:
            edit = conditioned_denoise
            edited_frames += 1
        out.append(edit)
    if edited_frames == 0:
        raise Error("LTX2 audio temporal mask selects no latent frames")
    return out^


def _request_hq_v2v_spatial_mask(
    values: List[Float32],
    sequence_tokens: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    if len(values) != sequence_tokens:
        raise Error("LTX2 V2V mask: patchified mask length mismatch")
    var scaled = List[Float32]()
    for value in values:
        scaled.append(value * scale)
    return Tensor.from_host(
        scaled, _sh3(1, sequence_tokens, 1), STDtype.F32, ctx
    )


def _request_hq_optional_mask_values(
    values: List[Float32],
) -> Optional[List[Float32]]:
    if len(values) == 0:
        return None
    return Optional[List[Float32]](values.copy())


def _request_hq_i2v_stage1_clean(
    first_frame_tokens: Tensor,
    sequence_tokens: Int,
    frame_tokens: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var suffix = zeros_device(
        _sh3(1, sequence_tokens - frame_tokens, 128), STDtype.BF16, ctx
    )
    return concat(1, ctx, first_frame_tokens, suffix)


def _request_hq_i2v_stage2_clean(
    first_frame_tokens: Tensor,
    upscaled_tokens: Tensor,
    sequence_tokens: Int,
    frame_tokens: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var suffix = slice(
        upscaled_tokens, 1, frame_tokens, sequence_tokens - frame_tokens, ctx
    )
    return concat(1, ctx, first_frame_tokens, suffix)


def _request_hq_keyframe_mask_values(
    target_tokens: Int,
    first_frame_tokens: Int,
    first_enabled: Bool,
    first_denoise: Float32,
    guide_tokens: Int,
    guide_denoise: Float32,
) raises -> List[Float32]:
    """Creator denoise mask for generated tokens plus one appended keyframe."""
    if target_tokens <= 0 or guide_tokens <= 0:
        raise Error("LTX2 keyframe conditioning requires positive token counts")
    if first_enabled and (
        first_frame_tokens <= 0 or first_frame_tokens >= target_tokens
    ):
        raise Error("LTX2 first-frame conditioning span is invalid")
    if (
        first_denoise < Float32(0.0) or first_denoise > Float32(1.0)
        or guide_denoise < Float32(0.0)
        or guide_denoise > Float32(1.0)
    ):
        raise Error("LTX2 keyframe denoise values must be in [0, 1]")
    var values = List[Float32]()
    for i in range(target_tokens):
        values.append(
            first_denoise
            if first_enabled and i < first_frame_tokens
            else Float32(1.0)
        )
    for _ in range(guide_tokens):
        values.append(guide_denoise)
    return values^


def _request_hq_keyframe_clean(
    target_clean: Tensor,
    guide_tokens: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Append the clean creator guide tokens after the generated sequence."""
    return concat(1, ctx, target_clean, guide_tokens)


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
    init_bf16: Tensor, noise: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    var i32 = cast_tensor(init_bf16, STDtype.F32, ctx)
    var n32 = cast_tensor(noise, STDtype.F32, ctx)
    var a = mul_scalar(n32, scale, ctx)
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


def _refhq_decode_audio_wav(
    a_final: Tensor, out_dir: String, ctx: DeviceContext
) raises -> String:
    """Existing refhq audio VAE -> vocoder -> WAV tail, shared by video decoders."""
    var wav_out = out_dir + "/refhq_audio.wav"
    print("  [decode] audio VAE (latent -> mel) -> vocoder (mel -> 48kHz wav)")
    # LTX 2.3 bundled decoder/vocoder tensors in the main checkpoint. LTX 2.5
    # ships them in a dedicated audio artifact. Route both consumers through
    # the request-aware checkpoint selector instead of accidentally reopening
    # the transformer checkpoint during postprocess.
    var audio_checkpoint = _request_hq_audio_vae_ckpt(_ckpt_bf16())
    var avae = LTX2AudioVaeDecoderWeights.load(audio_checkpoint, ctx)
    var a_final_bf16 = cast_tensor(a_final, STDtype.BF16, ctx)
    var mel = cast_tensor(decode_audio(avae, a_final_bf16, ctx), STDtype.BF16, ctx)
    var msh = mel.shape()
    print("  mel NCHW: [", msh[0], ",", msh[1], ",", msh[2], ",", msh[3], "] BF16 storage")
    _ = _stats(String("mel"), mel, ctx)

    var voc = LTX2VocoderWithBWE.from_file(audio_checkpoint, ctx)
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
    return wav_out


def _refhq_mux_av_3digit(
    out_dir: String, wav_out: String, mp4_out: String
) raises:
    """Existing refhq ffmpeg contract for three-digit frame filenames."""
    print("  [mux] ffmpeg frames + wav -> mp4 (fps=", Int(REFHQ_FPS), ")")
    var cmd = String("ffmpeg -y -framerate ") + String(Int(REFHQ_FPS)) + String(" -i ")
    cmd += out_dir + "/refhq_frame%03d.png -i " + wav_out
    # Infinite apad combined with -shortest leaves a measured ~0.678 s silent
    # AAC tail. Bound the mux to the exact 121/24 video duration.
    cmd += " -c:v libx264 -pix_fmt yuv420p -af apad -c:a aac -t "
    cmd += String(REFHQ_MUX_DURATION) + String(" -movflags +faststart ")
    cmd += mp4_out + " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        print("  [mux] WARNING: ffmpeg returned", rc, "(frames+wav still saved)")


def _request_mux_video_3digit(
    out_dir: String, mp4_out: String, fps: Float64, frame_count: Int
) raises:
    """Mux bounded tiled request frames without synthesizing an audio stream."""
    print("  [mux] ffmpeg tiled frames -> mp4 (fps=", fps, ")")
    var cmd = String("ffmpeg -y -framerate ") + String(fps)
    cmd += String(" -start_number 0 -i ")
    cmd += out_dir + "/refhq_frame%03d.png"
    cmd += String(" -frames:v ") + String(frame_count)
    cmd += " -c:v libx264 -pix_fmt yuv420p -movflags +faststart "
    cmd += mp4_out + " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        raise Error(
            String("LTX2 request tiled mux failed with ffmpeg exit ")
            + String(rc)
        )


def _request_mux_av_3digit(
    out_dir: String,
    wav_out: String,
    mp4_out: String,
    fps: Float64,
    frame_count: Int,
) raises:
    """Mux exact tiled request frames with optional decoded audio."""
    print("  [mux] ffmpeg tiled frames + audio -> mp4 (fps=", fps, ")")
    var cmd = String("ffmpeg -y -framerate ") + String(fps)
    cmd += String(" -start_number 0 -i ")
    cmd += out_dir + "/refhq_frame%03d.png -i " + wav_out
    cmd += String(" -frames:v ") + String(frame_count)
    cmd += " -c:v libx264 -pix_fmt yuv420p -af apad -c:a aac -shortest"
    cmd += " -movflags +faststart " + mp4_out + " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        raise Error(
            String("LTX2 request tiled A/V mux failed with ffmpeg exit ")
            + String(rc)
        )


# Shared decode+mux tail for refhq (full two-stage AND the s1out stage-1
# product mode). Caller must FREE the DiT working set (stream/LoRA factor
# cache) BEFORE calling — 24 GB discipline.
def _refhq_decode_mux[
    NF_CT: Int, NH_CT: Int, NW_CT: Int
](
    v_final: Tensor, a_final: Tensor, out_dir: String, ctx: DeviceContext
) raises:
    print("  [decode] video VAE (latent -> frames)")
    var vae = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
    var v_final_bf16 = cast_tensor(v_final, STDtype.BF16, ctx)
    var frames = decode_video[1, 128, NF_CT, NH_CT, NW_CT](
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

    var wav_out = _refhq_decode_audio_wav(a_final, out_dir, ctx)
    var mp4_out = out_dir + "/ltx2_refhq.mp4"
    _refhq_mux_av_3digit(out_dir, wav_out, mp4_out)

    print("=== refhq DONE ===")
    print("  mp4:", mp4_out)
    print("  wav:", wav_out)
    print("  frames:", out_dir + "/" + frame_prefix + "000.png ..",
          frame_prefix + _pad3(n_frames_out - 1) + ".png")


def _refhq_save_tiled_frames(
    frames: Tensor,
    local_start: Int,
    count: Int,
    global_start: Int,
    out_dir: String,
    ctx: DeviceContext,
) raises:
    var sh = frames.shape()
    if (
        len(sh) != 5 or sh[0] != 1 or sh[1] != 3
        or not (
            (sh[3] == 544 and sh[4] == 960)
            or (sh[3] == 960 and sh[4] == 544)
            or (sh[3] == 1088 and sh[4] == 1920)
            or (sh[3] == 1920 and sh[4] == 1088)
        )
        or local_start < 0 or count < 0 or local_start + count > sh[2]
    ):
        raise Error("refhq tiled decode: invalid frame chunk")
    for fr in range(count):
        var fslice = slice(frames, 2, local_start + fr, 1, ctx)
        var fs = fslice.shape()
        var chw = reshape(fslice, _sh4(fs[0], fs[1], fs[3], fs[4]), ctx)
        save_png(
            chw,
            out_dir + "/refhq_frame" + _pad3(global_start + fr) + ".png",
            ctx,
            ValueRange.SIGNED,
        )


def _refhq_decode_video_desktop_tiled(
    v_final: Tensor, out_dir: String, ctx: DeviceContext
) raises -> Int:
    """Decode 960x544x121 with the local LTX Desktop 15 GB tile contract."""
    print("  [decode] video VAE (Desktop 15GB tiled: 512/64 px, 64/24 frames)")
    var vae = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
    var latent = cast_tensor(v_final, STDtype.BF16, ctx)

    # Desktop split_temporal_causal groups are [0:8], [4:13], [9:16].
    # They emit [32,40,49] finalized frames without retaining the full movie.
    var previous = ltx2_desktop_decode_temporal_group[8](
        vae, latent, 0, ctx
    )
    var current = ltx2_desktop_decode_temporal_group[9](
        vae, latent, 4, ctx
    )
    _refhq_save_tiled_frames(
        previous, 0, LTX2_DESKTOP_FIRST_OUT_CHUNK, 0, out_dir, ctx
    )
    previous = ltx2_desktop_temporal_carry(previous, current, ctx)

    current = ltx2_desktop_decode_temporal_group[7](vae, latent, 9, ctx)
    _refhq_save_tiled_frames(
        previous, 0, LTX2_DESKTOP_MIDDLE_OUT_CHUNK, 32, out_dir, ctx
    )
    previous = ltx2_desktop_temporal_carry(previous, current, ctx)

    _refhq_save_tiled_frames(previous, 0, 49, 72, out_dir, ctx)
    print("  saved 121 tiled frame PNGs ->", out_dir)
    return 121


def _refhq_decode_request_tail_landscape[
    NF_CT: Int, TAIL_CT: Int
](
    vae: LTX2VaeDecoderWeights,
    latent: Tensor,
    f0: Int,
    var previous: Tensor,
    global_start: Int,
    out_dir: String,
    ctx: DeviceContext,
) raises:
    var current = ltx2_desktop_decode_temporal_group[TAIL_CT](
        vae, latent, f0, ctx
    )
    _refhq_save_tiled_frames(
        previous, 0, LTX2_DESKTOP_MIDDLE_OUT_CHUNK,
        global_start, out_dir, ctx,
    )
    previous = ltx2_desktop_temporal_carry(previous, current, ctx)
    comptime tail_frames = (TAIL_CT - 1) * 8 + 1
    _refhq_save_tiled_frames(
        previous, 0, tail_frames,
        global_start + LTX2_DESKTOP_MIDDLE_OUT_CHUNK, out_dir, ctx,
    )


def _refhq_decode_request_tail_portrait[
    NF_CT: Int, TAIL_CT: Int
](
    vae: LTX2VaeDecoderWeights,
    latent: Tensor,
    f0: Int,
    var previous: Tensor,
    global_start: Int,
    out_dir: String,
    ctx: DeviceContext,
) raises:
    var current = ltx2_desktop_decode_temporal_group_portrait[TAIL_CT](
        vae, latent, f0, ctx
    )
    _refhq_save_tiled_frames(
        previous, 0, LTX2_DESKTOP_MIDDLE_OUT_CHUNK,
        global_start, out_dir, ctx,
    )
    previous = ltx2_desktop_temporal_carry(previous, current, ctx)
    comptime tail_frames = (TAIL_CT - 1) * 8 + 1
    _refhq_save_tiled_frames(
        previous, 0, tail_frames,
        global_start + LTX2_DESKTOP_MIDDLE_OUT_CHUNK, out_dir, ctx,
    )


def _refhq_decode_video_desktop_tiled_request[
    NF_CT: Int, NUM_FRAMES_CT: Int
](
    v_final: Tensor, out_dir: String, ctx: DeviceContext
) raises -> Int:
    """Stream any registered 960x544 request without retaining all frames."""
    comptime if (
        not (
            (NF_CT == 16 and NUM_FRAMES_CT == 121)
            or (NF_CT == 19 and NUM_FRAMES_CT == 145)
            or (NF_CT == 25 and NUM_FRAMES_CT == 193)
            or (NF_CT == 31 and NUM_FRAMES_CT == 241)
            or (NF_CT == 61 and NUM_FRAMES_CT == 481)
        )
    ):
        raise Error("unsupported LTX2 Desktop temporal profile")
    print(
        "  [decode] video VAE ", NUM_FRAMES_CT,
        " frames (Desktop tiled, streaming finalized chunks)",
    )
    var vae = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
    var latent = cast_tensor(v_final, STDtype.BF16, ctx)
    var previous = ltx2_desktop_decode_temporal_group[8](
        vae, latent, 0, ctx
    )
    var current = ltx2_desktop_decode_temporal_group[9](
        vae, latent, 4, ctx
    )
    _refhq_save_tiled_frames(
        previous, 0, LTX2_DESKTOP_FIRST_OUT_CHUNK, 0, out_dir, ctx
    )
    previous = ltx2_desktop_temporal_carry(previous, current, ctx)

    var global_start = LTX2_DESKTOP_FIRST_OUT_CHUNK
    var f0 = 9
    while f0 + 9 < NF_CT:
        current = ltx2_desktop_decode_temporal_group[9](
            vae, latent, f0, ctx
        )
        _refhq_save_tiled_frames(
            previous, 0, LTX2_DESKTOP_MIDDLE_OUT_CHUNK,
            global_start, out_dir, ctx,
        )
        previous = ltx2_desktop_temporal_carry(previous, current, ctx)
        global_start += LTX2_DESKTOP_MIDDLE_OUT_CHUNK
        f0 += 5

    comptime if NF_CT == 19:
        _refhq_decode_request_tail_landscape[NF_CT, 5](
            vae, latent, f0, previous^, global_start, out_dir, ctx
        )
    elif NF_CT == 25:
        _refhq_decode_request_tail_landscape[NF_CT, 6](
            vae, latent, f0, previous^, global_start, out_dir, ctx
        )
    else:
        _refhq_decode_request_tail_landscape[NF_CT, 7](
            vae, latent, f0, previous^, global_start, out_dir, ctx
        )
    print("  saved ", NUM_FRAMES_CT, " tiled frame PNGs ->", out_dir)
    return NUM_FRAMES_CT


def _refhq_decode_video_desktop_tiled_request_portrait[
    NF_CT: Int, NUM_FRAMES_CT: Int
](
    v_final: Tensor, out_dir: String, ctx: DeviceContext
) raises -> Int:
    """Stream any registered 544x960 request without retaining all frames."""
    comptime if (
        not (
            (NF_CT == 16 and NUM_FRAMES_CT == 121)
            or (NF_CT == 19 and NUM_FRAMES_CT == 145)
            or (NF_CT == 25 and NUM_FRAMES_CT == 193)
            or (NF_CT == 31 and NUM_FRAMES_CT == 241)
            or (NF_CT == 61 and NUM_FRAMES_CT == 481)
        )
    ):
        raise Error("unsupported LTX2 Desktop portrait temporal profile")
    print(
        "  [decode] video VAE portrait ", NUM_FRAMES_CT,
        " frames (Desktop tiled, streaming finalized chunks)",
    )
    var vae = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
    var latent = cast_tensor(v_final, STDtype.BF16, ctx)
    var previous = ltx2_desktop_decode_temporal_group_portrait[8](
        vae, latent, 0, ctx
    )
    var current = ltx2_desktop_decode_temporal_group_portrait[9](
        vae, latent, 4, ctx
    )
    _refhq_save_tiled_frames(
        previous, 0, LTX2_DESKTOP_FIRST_OUT_CHUNK, 0, out_dir, ctx
    )
    previous = ltx2_desktop_temporal_carry(previous, current, ctx)

    var global_start = LTX2_DESKTOP_FIRST_OUT_CHUNK
    var f0 = 9
    while f0 + 9 < NF_CT:
        current = ltx2_desktop_decode_temporal_group_portrait[9](
            vae, latent, f0, ctx
        )
        _refhq_save_tiled_frames(
            previous, 0, LTX2_DESKTOP_MIDDLE_OUT_CHUNK,
            global_start, out_dir, ctx,
        )
        previous = ltx2_desktop_temporal_carry(previous, current, ctx)
        global_start += LTX2_DESKTOP_MIDDLE_OUT_CHUNK
        f0 += 5

    comptime if NF_CT == 19:
        _refhq_decode_request_tail_portrait[NF_CT, 5](
            vae, latent, f0, previous^, global_start, out_dir, ctx
        )
    elif NF_CT == 25:
        _refhq_decode_request_tail_portrait[NF_CT, 6](
            vae, latent, f0, previous^, global_start, out_dir, ctx
        )
    else:
        _refhq_decode_request_tail_portrait[NF_CT, 7](
            vae, latent, f0, previous^, global_start, out_dir, ctx
        )
    print(
        "  saved ", NUM_FRAMES_CT,
        " portrait tiled frame PNGs ->", out_dir,
    )
    return NUM_FRAMES_CT


def _refhq_decode_video_desktop_tiled_full(
    v_final: Tensor, out_dir: String, ctx: DeviceContext
) raises -> Int:
    """Decode full two-stage 1920x1088x121 with Desktop 15GB tiling."""
    print("  [decode] video VAE FULL 1920x1088 "
          "(Desktop 15GB tiled: 512/64 px, 64/24 frames)")
    var vae = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
    var latent = cast_tensor(v_final, STDtype.BF16, ctx)

    var previous = ltx2_desktop_decode_temporal_group_full[8](
        vae, latent, 0, ctx
    )
    var current = ltx2_desktop_decode_temporal_group_full[9](
        vae, latent, 4, ctx
    )
    _refhq_save_tiled_frames(
        previous, 0, LTX2_DESKTOP_FIRST_OUT_CHUNK, 0, out_dir, ctx
    )
    previous = ltx2_desktop_temporal_carry(previous, current, ctx)

    current = ltx2_desktop_decode_temporal_group_full[7](vae, latent, 9, ctx)
    _refhq_save_tiled_frames(
        previous, 0, LTX2_DESKTOP_MIDDLE_OUT_CHUNK, 32, out_dir, ctx
    )
    previous = ltx2_desktop_temporal_carry(previous, current, ctx)

    _refhq_save_tiled_frames(previous, 0, 49, 72, out_dir, ctx)
    print("  saved 121 full-resolution tiled frame PNGs ->", out_dir)
    return 121


def _refhq_decode_video_desktop_tiled_full_portrait(
    v_final: Tensor, out_dir: String, ctx: DeviceContext
) raises -> Int:
    """Decode 1088x1920x121 with the transposed Desktop 15GB tile grid."""
    print("  [decode] video VAE FULL 1088x1920 "
          "(Desktop 15GB tiled: 512/64 px, 64/24 frames)")
    var vae = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
    var latent = cast_tensor(v_final, STDtype.BF16, ctx)

    var previous = ltx2_desktop_decode_temporal_group_full_portrait[8](
        vae, latent, 0, ctx
    )
    var current = ltx2_desktop_decode_temporal_group_full_portrait[9](
        vae, latent, 4, ctx
    )
    _refhq_save_tiled_frames(
        previous, 0, LTX2_DESKTOP_FIRST_OUT_CHUNK, 0, out_dir, ctx
    )
    previous = ltx2_desktop_temporal_carry(previous, current, ctx)

    current = ltx2_desktop_decode_temporal_group_full_portrait[7](
        vae, latent, 9, ctx
    )
    _refhq_save_tiled_frames(
        previous, 0, LTX2_DESKTOP_MIDDLE_OUT_CHUNK, 32, out_dir, ctx
    )
    previous = ltx2_desktop_temporal_carry(previous, current, ctx)

    _refhq_save_tiled_frames(previous, 0, 49, 72, out_dir, ctx)
    print("  saved 121 portrait full-resolution tiled frame PNGs ->", out_dir)
    return 121


def _refhq_decode_tiled_mux(
    v_final: Tensor, a_final: Tensor, out_dir: String, ctx: DeviceContext
) raises:
    # Video lives in its own function scope so decoder weights and tile
    # intermediates are dead before the audio VAE/vocoder stage begins.
    var vsh = v_final.shape()
    var n_frames_out: Int
    if vsh[3] == REFHQ_NH1 and vsh[4] == REFHQ_NW1:
        n_frames_out = _refhq_decode_video_desktop_tiled(v_final, out_dir, ctx)
    elif vsh[3] == REFHQ_NH2 and vsh[4] == REFHQ_NW2:
        n_frames_out = _refhq_decode_video_desktop_tiled_full(
            v_final, out_dir, ctx
        )
    elif vsh[3] == REFHQ_NW2 and vsh[4] == REFHQ_NH2:
        n_frames_out = _refhq_decode_video_desktop_tiled_full_portrait(
            v_final, out_dir, ctx
        )
    else:
        raise Error("refhq tiled decode: unsupported video latent grid")

    var wav_out = _refhq_decode_audio_wav(a_final, out_dir, ctx)
    var mp4_out = out_dir + "/ltx2_refhq.mp4"
    _refhq_mux_av_3digit(out_dir, wav_out, mp4_out)
    print("=== refhq tiled DONE ===")
    print("  mp4:", mp4_out)
    print("  wav:", wav_out)
    print("  frames:", n_frames_out)


def run_refhq_decode(latents_path: String, out_dir: String) raises:
    """Decode preserved stage-1 or full two-stage latents in a fresh process.

    The 16 GB product contract cannot retain the streamed DiT allocation pool
    while loading the video VAE.  `run_refhq` writes these latents before its
    decode tail, so this entry point resumes that exact existing tail without
    rerunning denoising or substituting another runtime.
    """
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(latents_path)
    var video = Tensor.from_view(st.tensor_view("video"), ctx)
    var audio = Tensor.from_view(st.tensor_view("audio"), ctx)
    # Decode geometry is carried by the cache. Validate its structure without
    # pinning a fresh process to the dimensions used to build this binary.
    var vsh = video.shape()
    var ash = audio.shape()
    if len(vsh) != 5 or vsh[0] != 1 or vsh[1] != 128:
        raise Error(
            String("refhq-decode: video latent must be [1,128,NF,NH,NW]; got rank ")
            + String(len(vsh))
        )
    if vsh[2] < 1 or vsh[3] < 1 or vsh[4] < 1:
        raise Error("refhq-decode: video latent has a zero/negative dimension")
    if len(ash) != 4 or ash[0] != 1 or ash[1] != AUDIO_C or ash[3] != AUDIO_MEL:
        raise Error(
            String("refhq-decode: audio latent must be [1,")
            + String(AUDIO_C) + String(",S_A,") + String(AUDIO_MEL) + String("]")
        )
    print(
        "  [dims] video", vsh[2], "x", vsh[3], "x", vsh[4],
        " audio S_A=", ash[2], " (read from cache)",
    )
    _mkdir(out_dir)
    print("=== LTX-2.3 refhq fresh-process decode ===")
    print("  latents:", latents_path)
    print("  out:", out_dir)
    _refhq_decode_tiled_mux(video, audio, out_dir, ctx)


def run_refhq_upscale(
    stage1_cache_path: String,
    out_dir: String,
    dims_nf: Int = 0,
    dims_nh: Int = 0,
    dims_nw: Int = 0,
) raises:
    """Create the official x2 latent cache in a fresh, bounded process.

    The stage-1 transformer process must exit before this command is run.  This
    keeps its allocator reservation out of the upsampler/stage-2 memory budget
    while preserving the existing official LatentUpsampler implementation.
    """
    var ctx = DeviceContext()
    print("=== LTX-2.3 refhq fresh-process official x2 upscale ===")
    print("  stage1 cache:", stage1_cache_path)
    print("  out:", out_dir)
    var cache = NoiseSource.fixture(stage1_cache_path)
    if not cache.has_key(String("video")) or not cache.has_key(String("audio")):
        raise Error("refhq-upscale: stage1 cache missing video/audio")
    var video = cast_tensor(
        cache.load_key_f32(String("video"), ctx), STDtype.BF16, ctx
    )
    var audio = cast_tensor(
        cache.load_key_f32(String("audio"), ctx), STDtype.BF16, ctx
    )
    var vsh = video.shape()
    var ash = audio.shape()
    if len(ash) < 2 or ash[0] != 1:
        raise Error(
            String("refhq-upscale: audio cache must be [1,...]; got rank ")
            + String(len(ash))
        )
    var v_lat1: Tensor
    if len(vsh) == 5:
        if vsh[0] != 1 or vsh[1] != 128:
            raise Error("refhq-upscale: 5-D video cache must be [1,128,NF,NH,NW]")
        print(
            "  [dims] 5-D cache; NF/NH/NW read from shape:",
            vsh[2], "x", vsh[3], "x", vsh[4],
        )
        v_lat1 = video^
    elif len(vsh) == 3:
        if vsh[0] != 1 or vsh[2] != 128:
            raise Error("refhq-upscale: 3-D video cache must be [1,S_V,128]")
        var nf = dims_nf
        var nh = dims_nh
        var nw = dims_nw
        if nf <= 0 or nh <= 0 or nw <= 0:
            nf = REFHQ_NF
            nh = REFHQ_NH1
            nw = REFHQ_NW1
        if nf * nh * nw != vsh[1]:
            raise Error(
                String("refhq-upscale: dimensions ") + String(nf) + String("x")
                + String(nh) + String("x") + String(nw) + String(" do not match cached S_V ")
                + String(vsh[1])
            )
        print(
            "  [dims] 3-D cache; unpatchify at", nf, "x", nh, "x", nw,
            " S_V=", vsh[1],
        )
        v_lat1 = _refhq_unpatchify_video(video, nf, nh, nw, ctx)
    else:
        raise Error(
            String("refhq-upscale: video cache rank must be 3 or 5; got ")
            + String(len(vsh))
        )
    print("  [upscale] spatial x2 LatentUpsampler (scoped official weights)")
    var upscaled = _refhq_spatial_upscale_scoped(v_lat1, ctx)
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String("in"))
    tensors.append(ArcPointer[Tensor](cast_tensor(v_lat1, STDtype.F32, ctx)))
    names.append(String("out"))
    tensors.append(ArcPointer[Tensor](cast_tensor(upscaled, STDtype.F32, ctx)))
    _refhq_save(names^, tensors^, out_dir + "/upsampler.safetensors", ctx)
    _ = _stats(String("upscaled"), upscaled, ctx)


def run_refhq(
    contexts_path: String,
    noises_path: String,
    out_dir: String,
    steps: Int,
    apply_lora: Bool = True,
    stage1_product: Bool = False,
    defer_decode: Bool = False,
    stage1_max_steps: Int = 0,
    seed: UInt64 = SEED,
    stage1_cache_path: String = "",
    upscaler_cache_path: String = "",
    stage2_only: Bool = False,
    stage1_only: Bool = False,
) raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2.3 refhq — REAL HQ recipe, reference-gated ===")
    print("  contexts:", contexts_path)
    print("  noises:", noises_path, "(- = production Mojo randn)")
    print("  out:", out_dir, " steps:", steps, " seed:", seed)
    print("  stage1:", REFHQ_NF, "x", REFHQ_NH1, "x", REFHQ_NW1, " S_V=", REFHQ_S_V1,
          " S_A=", REFHQ_S_A, "  stage2: S_V2=", REFHQ_S_V2)

    # Creator TI2VidTwoStagesHQ uses the full/dev base. Keep the older distilled
    # checkpoint confined to the legacy modes above; refhq is the quality path.
    var ck = ShardedSafeTensors.open(_refhq_ckpt_fp8())

    # ── distilled LoRA only (the reference passes exactly one LoRA) ──
    # 2026-07-10 divergence-hunt finding: the golden reference harness passed
    # sd_ops=None so its LoRA lookups ALL silently missed — the golden clip is
    # a NO-LORA trajectory. `nolora` reproduces that contract for parity runs.
    var lora = LoraSet.load(_refhq_lora_distilled())
    if lora.format != FMT_LTX2_DISTILLED:
        raise Error("refhq: distilled LoRA not FMT_LTX2_DISTILLED")
    if apply_lora:
        print("  [lora] distilled mappings:", lora.num_mappings(),
              " block deltas: mmap-cached / per-module GPU fusion")
    else:
        print("  [lora] DISABLED (golden-reference no-LoRA contract)")

    if stage2_only and (
        stage1_cache_path.byte_length() == 0 or
        upscaler_cache_path.byte_length() == 0
    ):
        raise Error("refhq: stage2-only requires stage1-cache and upscaler-cache")

    # A fresh stage-2 process loads its 0.5 globals directly, avoiding a
    # retained stage-1 allocator reservation. The normal path starts at 0.25.
    var initial_lora_mult = REFHQ_LORA_S2 if stage2_only else REFHQ_LORA_S1
    print("  [load] globals (LoRA", initial_lora_mult, ")")
    var gw = _load_global_weights_dict(ck, ctx)
    if apply_lora:
        var n_g1 = lora.apply_to_globals(gw, initial_lora_mult, ctx)
        print("  [lora] global deltas:", n_g1)
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
        _refhq_ckpt_fp8(), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_conn = LTX2ConnectorWeights.load(
        _refhq_ckpt_fp8(), String("audio_embeddings_connector"),
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
    var pos_valid = _load_ctx_len(cf, String("video_len"), ctx)
    var neg_valid = _load_ctx_len(cf, String("neg_video_len"), ctx)
    print("  [ctx] valid rows: pos=", pos_valid, " neg=", neg_valid, "/", N_TXT)
    var enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](
        v_conn, v_pre, ctx, valid_len=pos_valid
    )
    var aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](
        a_conn, a_pre, ctx, valid_len=pos_valid
    )
    var neg_enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](
        v_conn, vn_pre, ctx, valid_len=neg_valid
    )
    var neg_aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](
        a_conn, an_pre, ctx, valid_len=neg_valid
    )
    _ = _stats(String("enc"), enc, ctx)
    _ = _stats(String("neg_enc"), neg_enc, ctx)

    # ── RoPE (REFHQ_FPS profile) ──
    print("  [rope] stage-1 + stage-2 video tables, audio tables (fps=",
          REFHQ_FPS, ")")
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
    var stream = LTX2BlockStream.open(_refhq_ckpt_fp8())
    if stream.block_count() != NUM_LAYERS:
        raise Error("stream block_count != 48")
    if stage2_only:
        print("  [stream] fresh stage-2 mmap/OS cache; no GPU-resident block range")
    else:
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
        ns = NoiseSource.production(seed)
        # rng-contract: mojo-native-not-pytorch-parity.
        video_x = randn(_sh3(1, REFHQ_S_V1, 128), seed, STDtype.BF16, ctx)
        audio_x = randn(_sh3(1, REFHQ_S_A, 128), seed + 1, STDtype.BF16, ctx)

    # ── stage-1 sigmas: LTX2Scheduler token shift, tokens = S_V1 ──
    if steps != REFHQ_STEPS:
        raise Error("refhq: pinned Creator schedule requires exactly 15 steps")
    var sig1 = _refhq_creator_stage1_sigmas()
    print("  [sampler] stage-1 sigmas (", len(sig1), "):", end="")
    for i in range(len(sig1)):
        print(" ", sig1[i], end="")
    print("")

    # ── STAGE 1: guided res_2s ──
    print("  [Stage1] GUIDED res_2s,", steps, "steps x 2 evals x 3 passes")
    var lora_mult_s1 = REFHQ_LORA_S1 if apply_lora else Float32(0.0)
    var first_eval = True

    @parameter
    def _den_s1(vx: Tensor, ax: Tensor, sigma: Float32) raises -> Tuple[Tensor, Tensor]:
        var mod = _build_mod_dims(
            ck, gw, sigma, REFHQ_S_V1, REFHQ_S_A, ctx
        )
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
    if stage1_cache_path.byte_length() > 0:
        # stage1_steps stores each BF16-rounded post-step state as F32. The
        # terminal sigma=0.0011 denoise is intentionally not one of those
        # per-step entries, so resume from s1_s14 and perform that exact final
        # model evaluation before entering the upsampler.
        print("  [Stage1] resume post-step state:", stage1_cache_path)
        var s1_cache = NoiseSource.fixture(stage1_cache_path)
        var has_final_s1 = s1_cache.has_key(String("video")) and s1_cache.has_key(String("audio"))
        if has_final_s1:
            video_x = cast_tensor(
                s1_cache.load_key_f32(String("video"), ctx), STDtype.BF16, ctx
            )
            audio_x = cast_tensor(
                s1_cache.load_key_f32(String("audio"), ctx), STDtype.BF16, ctx
            )
        else:
            if not s1_cache.has_key(String("s1_s14_video")) or not s1_cache.has_key(String("s1_s14_audio")):
                raise Error("refhq: stage1 cache missing final or s1_s14 video/audio")
            video_x = cast_tensor(
                s1_cache.load_key_f32(String("s1_s14_video"), ctx), STDtype.BF16, ctx
            )
            audio_x = cast_tensor(
                s1_cache.load_key_f32(String("s1_s14_audio"), ctx), STDtype.BF16, ctx
            )
        var vsh = video_x.shape()
        var ash = audio_x.shape()
        if len(vsh) != 3 or vsh[0] != 1 or vsh[1] != REFHQ_S_V1 or vsh[2] != 128:
            raise Error("refhq: stage1 cached video shape mismatch")
        if len(ash) != 3 or ash[0] != 1 or ash[1] != REFHQ_S_A or ash[2] != 128:
            raise Error("refhq: stage1 cached audio shape mismatch")
        if has_final_s1:
            print("  [Stage1] resumed terminal cached state")
        else:
            var final_s1 = _den_s1(
                video_x, audio_x, RES2S_REF_TERMINAL_SIGMA
            )
            video_x = cast_tensor(final_s1[0], STDtype.BF16, ctx)
            audio_x = cast_tensor(final_s1[1], STDtype.BF16, ctx)
            print("  [Stage1] resumed final denoise @ sigma=", RES2S_REF_TERMINAL_SIGMA)
    else:
        var s1_out = res2s_ref_loop[_den_s1](
            sig1, video_x^, audio_x^, ns, String("s1"), s1_names, s1_tensors, ctx,
            max_steps=stage1_max_steps,
            dump_evals=stage1_max_steps > 0,
        )
        video_x = s1_out[0].clone(ctx)
        audio_x = s1_out[1].clone(ctx)
        _refhq_save(s1_names^, s1_tensors^, out_dir + "/stage1_steps.safetensors", ctx)
    print("  [Stage1] done in", Float32(perf_counter() - t_s1), "s")
    _ = _stats(String("video_x(stage1)"), video_x, ctx)
    _ = _stats(String("audio_x(stage1)"), audio_x, ctx)
    var s1_final_names = List[String]()
    var s1_final_tensors = List[ArcPointer[Tensor]]()
    s1_final_names.append(String("video"))
    s1_final_tensors.append(ArcPointer[Tensor](cast_tensor(video_x, STDtype.BF16, ctx)))
    s1_final_names.append(String("audio"))
    s1_final_tensors.append(ArcPointer[Tensor](cast_tensor(audio_x, STDtype.BF16, ctx)))
    _refhq_save(
        s1_final_names^, s1_final_tensors^,
        out_dir + "/stage1_final.safetensors", ctx,
    )

    if stage1_only:
        print("  [handoff] stage-1 cache saved; exiting before upscaler weights")
        print("  [handoff] resume with: refhq-upscale ",
              out_dir + "/stage1_final.safetensors ", out_dir)
        return

    if stage1_max_steps > 0:
        print("  [Stage1] bounded parity capture complete; skipping decode/stage-2")
        var bv = _refhq_unpatchify_video(
            video_x, REFHQ_NF, REFHQ_NH1, REFHQ_NW1, ctx
        )
        var ba = _refhq_unpatchify_audio(audio_x, REFHQ_S_A, ctx)
        var bn = List[String]()
        var bt = List[ArcPointer[Tensor]]()
        bn.append(String("video"))
        bt.append(ArcPointer[Tensor](cast_tensor(bv, STDtype.BF16, ctx)))
        bn.append(String("audio"))
        bt.append(ArcPointer[Tensor](cast_tensor(ba, STDtype.BF16, ctx)))
        _refhq_save(bn^, bt^, out_dir + "/final_latents.safetensors", ctx)
        return

    # ── STAGE-1 PRODUCT mode (2026-07-16 oracle isolation: at the design
    # operating point the upsampler+stage-2 leg collapses latent variance
    # (std 1.03 -> 0.55) and imprints a dot/knit occluder mesh in the OFFICIAL
    # torch stack across all LoRA arms, while the SAME run's stage-1 latent
    # decodes clean — see output/ltx2_oracle4_*.) `s1out` argv delivers the
    # measured-clean stage-1 as the product and skips upsampler/stage-2.
    if stage1_product:
        print("  [s1out] STAGE-1 PRODUCT mode — decoding stage-1, skipping "
              "upsampler/stage-2")
        var v1 = _refhq_unpatchify_video(video_x, REFHQ_NF, REFHQ_NH1, REFHQ_NW1, ctx)
        var a1 = _refhq_unpatchify_audio(audio_x, REFHQ_S_A, ctx)
        var f1_names = List[String]()
        var f1_tensors = List[ArcPointer[Tensor]]()
        f1_names.append(String("video"))
        f1_tensors.append(ArcPointer[Tensor](cast_tensor(v1, STDtype.BF16, ctx)))
        f1_names.append(String("audio"))
        f1_tensors.append(ArcPointer[Tensor](cast_tensor(a1, STDtype.BF16, ctx)))
        _refhq_save(f1_names^, f1_tensors^, out_dir + "/final_latents.safetensors", ctx)
        if defer_decode:
            print("  [handoff] final latents saved; exiting before VAE decode")
            print("  [handoff] resume with: refhq-decode ",
                  out_dir + "/final_latents.safetensors ", out_dir)
            return
        print("  [decode] freeing resident fp8 blocks + LoRA factor cache")
        stream = LTX2BlockStream.open(_refhq_ckpt_fp8())
        lora = LoraSet.load(_refhq_lora_distilled())
        _refhq_decode_mux[REFHQ_NF, REFHQ_NH1, REFHQ_NW1](v1, a1, out_dir, ctx)
        return

    # ── UPSAMPLE (un_normalize -> x2 -> normalize) ──
    var v_lat1 = _refhq_unpatchify_video(video_x, REFHQ_NF, REFHQ_NH1, REFHQ_NW1, ctx)
    var upscaled: Tensor
    if upscaler_cache_path.byte_length() > 0:
        print("  [upscale] resume cached official x2 output:", upscaler_cache_path)
        var up_cache = NoiseSource.fixture(upscaler_cache_path)
        if not up_cache.has_key(String("out")):
            raise Error("refhq: upscaler cache missing out")
        upscaled = up_cache.load_key_f32(String("out"), ctx)
        var ush = upscaled.shape()
        if len(ush) != 5 or ush[0] != 1 or ush[1] != 128 or ush[2] != REFHQ_NF or ush[3] != REFHQ_NH2 or ush[4] != REFHQ_NW2:
            raise Error("refhq: cached upscaler output shape mismatch")
    else:
        print("  [upscale] spatial x2 LatentUpsampler (scoped weights)")
        upscaled = _refhq_spatial_upscale_scoped(v_lat1, ctx)
        var up_names = List[String]()
        var up_tensors = List[ArcPointer[Tensor]]()
        up_names.append(String("in"))
        up_tensors.append(ArcPointer[Tensor](cast_tensor(v_lat1, STDtype.F32, ctx)))
        up_names.append(String("out"))
        up_tensors.append(ArcPointer[Tensor](cast_tensor(upscaled, STDtype.F32, ctx)))
        _refhq_save(up_names^, up_tensors^, out_dir + "/upsampler.safetensors", ctx)

    # Full-grid activations are 4x stage-1. Drop the stage-1 raw-FP8 GPU
    # resident range; stage-2 uses the same existing mmap/OS-cached stream.
    print("  [release] stage-1 resident FP8 range before full-grid stage-2")
    stream = LTX2BlockStream.open(_refhq_ckpt_fp8())

    # ── stage-2 globals: distilled LoRA strength 0.5 (fresh reload) ──
    if stage2_only:
        print("  [load] stage-2 globals already loaded in fresh process")
    else:
        print("  [load] stage-2 globals (LoRA", REFHQ_LORA_S2, ")")
        gw = _load_global_weights_dict(ck, ctx)
        if apply_lora:
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
        # GaussianNoiser — NOT _get_new_noise-normalized). Keep storage BF16 at
        # the product boundary; _refhq_noise_blend casts internally for F32 math.
        s2n_v = randn(_sh3(1, REFHQ_S_V2, 128), seed + 100, STDtype.BF16, ctx)
        s2n_a = randn(_sh3(1, REFHQ_S_A, 128), seed + 101, STDtype.BF16, ctx)
    var vx2 = _refhq_noise_blend(up_flat_b, s2n_v, s2_scale, ctx)
    var ax2 = _refhq_noise_blend(audio_x, s2n_a, s2_scale, ctx)
    print("  [Stage2] SIMPLE res_2s, sigmas 0.909375/0.725/0.421875/0  noise_scale=", s2_scale)

    var lora_mult_s2 = REFHQ_LORA_S2 if apply_lora else Float32(0.0)
    var first_eval2 = True

    @parameter
    def _den_s2(vx: Tensor, ax: Tensor, sigma: Float32) raises -> Tuple[Tensor, Tensor]:
        var mod = _build_mod_dims(
            ck, gw, sigma, REFHQ_S_V2, REFHQ_S_A, ctx,
            uniform_timestep=True,
        )
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

    # ── final latents (unpatchified latent layout, BF16) ──
    var v_final = _refhq_unpatchify_video(vx2, REFHQ_NF, REFHQ_NH2, REFHQ_NW2, ctx)
    var a_final = _refhq_unpatchify_audio(ax2, REFHQ_S_A, ctx)
    var f_names = List[String]()
    var f_tensors = List[ArcPointer[Tensor]]()
    f_names.append(String("video"))
    f_tensors.append(ArcPointer[Tensor](cast_tensor(v_final, STDtype.BF16, ctx)))
    f_names.append(String("audio"))
    f_tensors.append(ArcPointer[Tensor](cast_tensor(a_final, STDtype.BF16, ctx)))
    _refhq_save(f_names^, f_tensors^, out_dir + "/final_latents.safetensors", ctx)
    _ = _stats(String("video(final)"), v_final, ctx)
    _ = _stats(String("audio(final)"), a_final, ctx)

    if defer_decode:
        print("  [handoff] final latents saved; exiting before VAE decode")
        print("  [handoff] resume with: refhq-decode ",
              out_dir + "/final_latents.safetensors ", out_dir)
        return

    # ── free the DiT working set BEFORE the VAE loads (24 GB discipline, same
    # as the staged path: the resident fp8 range (~20.5 GB) and the rank-384
    # LoRA factor cache (~7.6 GB) must be gone before decoder weights + frame
    # buffers allocate). Reassignment destroys the old values -> VRAM freed.
    print("  [decode] freeing resident fp8 blocks + LoRA factor cache")
    stream = LTX2BlockStream.open(_refhq_ckpt_fp8())
    lora = LoraSet.load(_refhq_lora_distilled())

    _refhq_decode_mux[REFHQ_NF, REFHQ_NH2, REFHQ_NW2](v_final, a_final, out_dir, ctx)


# ── LTX Desktop Retake: one direct distilled full-resolution stage ──────────
def _request_hq_encode_source_audio(
    request_checkpoint: String,
    source_audio_path: String,
    source_audio_sample_rate: Int,
    source_audio_channels: Int,
    source_audio_samples: Int,
    video_edit_mode: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """Creator AudioProcessor + AudioVAE encoder -> [1,S_A,128] tokens."""
    if source_audio_path.byte_length() == 0:
        raise Error("LTX2 creator source-audio path is empty")
    print(
        "  [Temporal edit audio] creator PyAV samples:", source_audio_samples,
        "@", source_audio_sample_rate, "Hz channels:", source_audio_channels,
    )
    var waveform = load_source_waveform_f32(
        source_audio_path,
        source_audio_sample_rate,
        source_audio_channels,
        source_audio_samples,
        ctx,
    )
    var mel = waveform_to_creator_log_mel(waveform, ctx)
    var audio_vae_path = _request_hq_audio_vae_ckpt(request_checkpoint)
    print("  [Temporal edit audio] AudioVAE encoder:", audio_vae_path)
    var encoder = LTX2AudioVaeDecoderWeights.load_encoder(
        audio_vae_path, ctx
    )
    var latent = encode_audio_spectrogram(encoder, mel, ctx)
    var latent_frames = latent.shape()[2]
    if latent_frames > REQUEST_HQ_S_A:
        latent = slice(latent, 2, 0, REQUEST_HQ_S_A, ctx)
    elif latent_frames < REQUEST_HQ_S_A:
        var pad = zeros_device(
            _sh4(
                1, AUDIO_C, REQUEST_HQ_S_A - latent_frames, AUDIO_MEL
            ),
            latent.dtype(),
            ctx,
        )
        if video_edit_mode == String("extend_start"):
            latent = concat(2, ctx, pad, latent)
        else:
            latent = concat(2, ctx, latent, pad)
    return _refhq_patchify_audio(latent, REQUEST_HQ_S_A, ctx)


def _run_request_retake_single_stage(
    contexts_path: String,
    negative_contexts_path: String,
    contexts_are_projected: Bool,
    source_clean_tokens: Tensor,
    source_audio_path: String,
    source_audio_sample_rate: Int,
    source_audio_channels: Int,
    source_audio_samples: Int,
    video_edit_mode: String,
    video_edit_start: Float32,
    video_edit_end: Float32,
    regenerate_source_audio: Bool,
    out_dir: String,
    seed: UInt64,
    include_audio: Bool,
    quant: String,
    steps: Int,
    defer_decode: Bool,
    source_encode_seconds: Float64,
    total_t0: Float64,
    total_vram_bytes: Int,
    min_free_at_entry: Int,
    ctx: DeviceContext,
) raises:
    """Port LTX Desktop's `LTXRetakePipeline` topology without substitutions.

    One direct distilled transformer runs at the final latent resolution. The
    selected TemporalRegionMask initializes noise only inside the edit window,
    supplies `sigma * mask` model timesteps, restores clean-source x0 outside
    the window before each Euler update, and never loads the two-stage support
    LoRA or spatial upscaler.
    """
    if steps != 8:
        raise Error("LTX2 Retake creator path requires distilled 8-step Euler")
    if not defer_decode:
        raise Error(
            "LTX2 temporal edit requires fresh-process decode to release the 22B transformer"
        )
    var edit_label = String("Extend") if (
        video_edit_mode == String("extend_start")
        or video_edit_mode == String("extend_end")
    ) else String("Retake")
    var edit_heading = String("EXTEND") if (
        video_edit_mode == String("extend_start")
        or video_edit_mode == String("extend_end")
    ) else String("RETAKE")
    var edit_phase = String("extend") if (
        video_edit_mode == String("extend_start")
        or video_edit_mode == String("extend_end")
    ) else String("retake")
    var progress_total = steps + 3
    var min_free_bytes = min_free_at_entry
    var load_seconds = Float64(0.0)
    var conditioning_seconds = Float64(0.0)
    var source_conditioning_seconds = source_encode_seconds
    var prepare_seconds = Float64(0.0)
    var denoise_seconds = Float64(0.0)
    var cfg = LTX2Config.ltx2()
    var t0 = perf_counter()
    var checkpoint_path = _request_hq_ckpt(quant)
    var has_source_audio = source_audio_path.byte_length() > 0
    var clean_audio = zeros_device(
        _sh3(1, REQUEST_HQ_S_A, 128), STDtype.BF16, ctx
    )
    if has_source_audio:
        var audio_encode_t0 = perf_counter()
        clean_audio = _request_hq_encode_source_audio(
            checkpoint_path,
            source_audio_path,
            source_audio_sample_rate,
            source_audio_channels,
            source_audio_samples,
            video_edit_mode,
            ctx,
        )
        source_conditioning_seconds += perf_counter() - audio_encode_t0
    print(
        "=== LTX-2.3", edit_heading,
        "— LTX DESKTOP ONE-STAGE ORACLE ===",
    )
    print("  checkpoint:", checkpoint_path)
    print(
        "  full-resolution latent:", REQUEST_HQ_NF, "x",
        REQUEST_HQ_NH2, "x", REQUEST_HQ_NW2,
        "tokens:", REQUEST_HQ_S_V2,
    )
    print("  temporal window:", video_edit_start, "..", video_edit_end)
    print("  upscaler stage: none")
    var ck = ShardedSafeTensors.open(checkpoint_path)
    var loras = _RequestHQLoraStack.load()
    loras.validate()
    if len(loras.trained) != 0:
        raise Error(
            "LTX2 temporal edit creator path accepts only the selected checkpoint's "
            "distillation adapter, not authored overlay LoRAs"
        )
    var distillation_mult = _request_hq_distillation_weight() \
        if loras.distilled else Float32(0.0)
    loras.print_summary()
    var gw = _load_global_weights_dict(ck, ctx)
    var n_global = loras.apply_to_globals(
        gw, distillation_mult, ctx
    )
    print("  [lora]", edit_label, "global deltas:", n_global)
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
    ctx.synchronize()
    load_seconds = perf_counter() - t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    _write_ltx2_status(
        out_dir, String("running"), String("conditioning"), 0,
        progress_total, String("Loading ") + edit_label
        + String(" conditioning"),
    )
    t0 = perf_counter()
    var pos_st = ShardedSafeTensors.open(contexts_path)
    var enc: Tensor
    var aenc: Tensor
    if contexts_are_projected:
        enc = Tensor.from_view_as_bf16(
            pos_st.tensor_view("video_context"), ctx
        )
        aenc = Tensor.from_view_as_bf16(
            pos_st.tensor_view("audio_context"), ctx
        )
    else:
        var v_conn = LTX2ConnectorWeights.load(
            checkpoint_path, String("video_embeddings_connector"),
            LTX2ConnectorConfig.video(), ctx,
        )
        var a_conn = LTX2ConnectorWeights.load(
            checkpoint_path, String("audio_embeddings_connector"),
            LTX2ConnectorConfig.audio(), ctx,
        )
        var v_pre = Tensor.from_view_as_bf16(
            pos_st.tensor_view("video_context"), ctx
        )
        var a_pre = Tensor.from_view_as_bf16(
            pos_st.tensor_view("audio_context"), ctx
        )
        var valid = _load_ctx_len(pos_st, String("video_len"), ctx)
        enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](
            v_conn, v_pre, ctx, valid_len=valid
        )
        aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](
            a_conn, a_pre, ctx, valid_len=valid
        )
    ctx.synchronize()
    conditioning_seconds = perf_counter() - t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    _write_ltx2_status(
        out_dir, String("running"), String("preparing"), 0,
        progress_total, String("Preparing one-stage ") + edit_label
        + String(" temporal mask"),
    )
    t0 = perf_counter()
    var vr = _request_hq_video_rope(
        REQUEST_HQ_NF, REQUEST_HQ_NH2, REQUEST_HQ_NW2,
        REQUEST_HQ_S_V2, ctx,
    )
    var ac = _build_audio_coords_dims(REQUEST_HQ_S_A)
    var arope = _compute_rope(
        ac, 1, REQUEST_HQ_S_A, AD, _mp1(), ROPE_THETA, A_HEADS,
        STDtype.BF16, ctx,
    )
    var caarope = _compute_rope(
        ac, 1, REQUEST_HQ_S_A, CA_DIM, _mp1(), ROPE_THETA, A_HEADS,
        STDtype.BF16, ctx,
    )
    var a_cos = _clone(arope[0], ctx)
    var a_sin = _clone(arope[1], ctx)
    var ca_a_cos = _clone(caarope[0], ctx)
    var ca_a_sin = _clone(caarope[1], ctx)
    var stream = LTX2BlockStream.open(checkpoint_path)
    if stream.block_count() != NUM_LAYERS:
        raise Error(
            "LTX2 temporal edit: selected checkpoint does not have 48 AV blocks"
        )

    var mask_values = _request_hq_v2v_temporal_mask_values(
        REQUEST_HQ_NF, REQUEST_HQ_NH2, REQUEST_HQ_NW2,
        Float32(REQUEST_HQ_FPS), video_edit_start, video_edit_end,
        Float32(1.0),
    )
    var mask = _request_hq_v2v_spatial_mask(
        mask_values, REQUEST_HQ_S_V2, Float32(1.0), ctx
    )
    var clean = source_clean_tokens.clone(ctx)
    var video_noise = randn(
        _sh3(1, REQUEST_HQ_S_V2, 128), seed, STDtype.BF16, ctx
    )
    var video_x = ltx2_creator_noiser_from_noise(
        clean, video_noise, mask, ctx
    )
    var audio_noise = randn(
        _sh3(1, REQUEST_HQ_S_A, 128), seed + 1, STDtype.BF16, ctx
    )
    var audio_mask = zeros_device(
        _sh3(1, REQUEST_HQ_S_A, 1), STDtype.F32, ctx
    )
    var audio_mask_values = List[Float32]()
    var audio_x = audio_noise.clone(ctx)
    if has_source_audio and regenerate_source_audio:
        audio_mask_values = _request_hq_audio_temporal_mask_values(
            REQUEST_HQ_S_A,
            video_edit_start,
            video_edit_end,
            Float32(1.0),
        )
        audio_mask = _request_hq_v2v_spatial_mask(
            audio_mask_values, REQUEST_HQ_S_A, Float32(1.0), ctx
        )
        audio_x = ltx2_creator_noiser_from_noise(
            clean_audio, audio_noise, audio_mask, ctx
        )
    var scheduler = LTX2Scheduler.distilled()
    ctx.synchronize()
    prepare_seconds = perf_counter() - t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    t0 = perf_counter()
    for step in range(scheduler.num_steps):
        var sigma = scheduler.sigma(step)
        _write_ltx2_status(
            out_dir, String("running"),
            String("denoising_") + edit_phase,
            step + 1, progress_total,
            edit_label + String(" step ") + String(step + 1) + String(" of ")
            + String(scheduler.num_steps),
        )
        var mod = _build_mod_dims(
            ck, gw, sigma, REQUEST_HQ_S_V2, REQUEST_HQ_S_A, ctx,
            conditioned_video_tokens=REQUEST_HQ_S_V2,
            conditioned_video_denoise=Float32(1.0),
            conditioned_video_denoise_values=Optional[List[Float32]](
                mask_values.copy()
            ),
            audio_denoise=(
                Float32(0.0)
                if has_source_audio and not regenerate_source_audio
                else Float32(1.0)
            ),
            conditioned_audio_denoise_values=_request_hq_optional_mask_values(
                audio_mask_values
            ),
        )
        var prediction = _request_edit_forward_flat[
            REQUEST_HQ_S_V2, REQUEST_HQ_S_A,
            REQUEST_HQ_VPAD2, REQUEST_HQ_APAD,
        ](
            loras, distillation_mult,
            cfg, g, stream, video_x, audio_x, enc, aenc, mod, vr,
            a_cos, a_sin, ca_a_cos, ca_a_sin, ctx,
        )
        var video_x0 = _refhq_x0(video_x, prediction[0], sigma, ctx)
        video_x0 = res2s_post_process_latent(
            video_x0, mask, clean, ctx
        )
        video_x = _request_edit_euler_step_x0(
            video_x, video_x0, sigma, scheduler.sigma(step + 1), ctx
        )
        if has_source_audio:
            var audio_x0 = _refhq_x0(
                audio_x, prediction[1], sigma, ctx
            )
            audio_x0 = res2s_post_process_latent(
                audio_x0, audio_mask, clean_audio, ctx
            )
            audio_x = _request_edit_euler_step_x0(
                audio_x, audio_x0, sigma,
                scheduler.sigma(step + 1), ctx,
            )
        else:
            audio_x = scheduler.step(audio_x, prediction[1], step, ctx)
    ctx.synchronize()
    denoise_seconds = perf_counter() - t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    var v_final = _refhq_unpatchify_video(
        video_x, REQUEST_HQ_NF, REQUEST_HQ_NH2, REQUEST_HQ_NW2, ctx
    )
    var a_final = _refhq_unpatchify_audio(audio_x, REQUEST_HQ_S_A, ctx)
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String("video"))
    tensors.append(
        ArcPointer[Tensor](cast_tensor(v_final, STDtype.BF16, ctx))
    )
    names.append(String("audio"))
    tensors.append(
        ArcPointer[Tensor](cast_tensor(a_final, STDtype.BF16, ctx))
    )
    _refhq_save(
        names^, tensors^, out_dir + "/final_latents.safetensors", ctx
    )

    # Destroy the streamed transformer handle before the fresh decoder starts.
    stream = LTX2BlockStream.open(checkpoint_path)
    _write_ltx2_decode_handoff(
        out_dir, steps, seed, include_audio, String("distilled"), quant,
        contexts_path, negative_contexts_path, 0,
        load_seconds, conditioning_seconds, source_conditioning_seconds,
        prepare_seconds, denoise_seconds, perf_counter() - total_t0,
        total_vram_bytes, min_free_bytes,
    )
    _write_ltx2_status(
        out_dir, String("running"), String("decode_handoff"),
        progress_total, progress_total,
        String("Releasing one-stage ") + edit_label
        + String(" denoiser before decode"),
    )
    print("=== LTX-2.3", edit_heading, "DENOISE COMPLETE ===")


# ── Exact request-driven Creator-HQ product path ─────────────────────────────
def run_request_hq(
    contexts_path: String,
    negative_contexts_path: String,
    contexts_are_projected: Bool,
    noise_fixture_path: String,
    image_path: String,
    image_strength: Float32,
    last_image_path: String,
    last_image_strength: Float32,
    video_path: String,
    video_strength: Float32,
    video_mask_path: String,
    video_edit_mode: String,
    video_edit_start: Float32,
    video_edit_end: Float32,
    video_source_frames: Int,
    source_audio_path: String,
    source_audio_sample_rate: Int,
    source_audio_channels: Int,
    source_audio_samples: Int,
    regenerate_source_audio: Bool,
    out_dir: String,
    seed: UInt64,
    include_audio: Bool,
    quant: String,
    steps: Int,
    guidance_mode: String,
    defer_decode: Bool,
) raises:
    var total_t0 = perf_counter()
    var load_seconds = Float64(0.0)
    var conditioning_seconds = Float64(0.0)
    var source_encode_seconds = Float64(0.0)
    var prepare_seconds = Float64(0.0)
    var denoise_seconds = Float64(0.0)
    var video_decode_seconds = Float64(0.0)
    var audio_decode_seconds = Float64(0.0)
    var mux_seconds = Float64(0.0)
    var progress_total = steps + 3

    _write_ltx2_status(
        out_dir, String("running"), String("loading_model"), 0,
        progress_total, String("Loading LTX2 dev model"),
    )
    var ctx = DeviceContext()
    var mem0 = cu_mem_get_info()
    var total_vram_bytes = mem0.total_bytes
    var min_free_bytes = mem0.free_bytes
    var cfg = LTX2Config.ltx2()
    var i2v_enabled = image_path.byte_length() > 0
    var last_keyframe_enabled = last_image_path.byte_length() > 0
    var v2v_enabled = video_path.byte_length() > 0
    var v2v_mask_enabled = video_mask_path.byte_length() > 0
    var temporal_edit_enabled = video_edit_mode != String("standard")
    if (i2v_enabled or last_keyframe_enabled) and v2v_enabled:
        raise Error(
            "LTX2 request HQ: image keyframes and video sources are mutually exclusive"
        )
    if v2v_mask_enabled and not v2v_enabled:
        raise Error("LTX2 request HQ: a V2V mask requires a video source")
    if temporal_edit_enabled and not v2v_enabled:
        raise Error("LTX2 request HQ: temporal editing requires a video source")
    if temporal_edit_enabled and v2v_mask_enabled:
        raise Error(
            "LTX2 request HQ: temporal and spatial V2V masks are mutually exclusive"
        )
    if temporal_edit_enabled and (
        video_source_frames <= 1
        or video_edit_start < Float32(0.0)
        or video_edit_end <= video_edit_start
    ):
        raise Error("LTX2 request HQ: invalid temporal edit contract")
    if last_keyframe_enabled and temporal_edit_enabled:
        raise Error(
            "LTX2 request HQ: keyframe interpolation cannot use temporal editing"
        )
    var source_enabled = i2v_enabled or last_keyframe_enabled or v2v_enabled
    var source_strength = image_strength if i2v_enabled else video_strength
    var source_stage1: Optional[Tensor] = None
    var source_stage2: Optional[Tensor] = None
    var last_stage1: Optional[Tensor] = None
    var last_stage2: Optional[Tensor] = None
    if i2v_enabled:
        var source_t0 = perf_counter()
        _write_ltx2_status(
            out_dir, String("running"), String("encoding_source"), 0,
            progress_total, String("Encoding I2V source image at both stages"),
        )
        print("  [I2V] source:", image_path, "strength:", image_strength)
        var encoded_frames = _request_hq_encode_i2v_first_frames(
            image_path, ctx
        )
        source_stage1 = Optional[Tensor](encoded_frames[0].clone(ctx))
        source_stage2 = Optional[Tensor](encoded_frames[1].clone(ctx))
        ctx.synchronize()
        source_encode_seconds = perf_counter() - source_t0
        min_free_bytes = _record_ltx2_min_free(min_free_bytes)
    if last_keyframe_enabled:
        var keyframe_t0 = perf_counter()
        _write_ltx2_status(
            out_dir, String("running"), String("encoding_source"), 0,
            progress_total,
            String("Encoding creator last-frame guide at both stages"),
        )
        print(
            "  [keyframe] last source:", last_image_path,
            "frame:", REQUEST_HQ_NUM_FRAMES - 1,
            "strength:", last_image_strength,
        )
        var encoded_last = _request_hq_encode_i2v_first_frames(
            last_image_path, ctx
        )
        last_stage1 = Optional[Tensor](encoded_last[0].clone(ctx))
        last_stage2 = Optional[Tensor](encoded_last[1].clone(ctx))
        ctx.synchronize()
        source_encode_seconds += perf_counter() - keyframe_t0
        min_free_bytes = _record_ltx2_min_free(min_free_bytes)
    if v2v_enabled:
        var source_t0 = perf_counter()
        _write_ltx2_status(
            out_dir, String("running"), String("encoding_source"), 0,
            progress_total, String("Decoding and encoding V2V source clip"),
        )
        print("  [V2V] source:", video_path, "strength:", video_strength)
        if temporal_edit_enabled:
            print(
                "  [V2V] temporal edit:", video_edit_mode,
                video_edit_start, "..", video_edit_end,
                "source frames:", video_source_frames,
            )
        if v2v_mask_enabled:
            print("  [V2V] spatial mask:", video_mask_path)
        if temporal_edit_enabled:
            var retake_source = _request_hq_encode_retake_video(
                video_path, out_dir, video_edit_mode,
                video_source_frames, progress_total, ctx
            )
            source_stage2 = Optional[Tensor](retake_source^)
        else:
            var encoded_video = _request_hq_encode_v2v_video(
                video_path, out_dir, video_edit_mode, video_source_frames,
                progress_total, ctx
            )
            source_stage1 = Optional[Tensor](encoded_video[0].clone(ctx))
            source_stage2 = Optional[Tensor](encoded_video[1].clone(ctx))
        ctx.synchronize()
        source_encode_seconds = perf_counter() - source_t0
        min_free_bytes = _record_ltx2_min_free(min_free_bytes)
    if temporal_edit_enabled:
        var retake_lora_count = _env_str("LTX2_TRAINED_LORA_COUNT")
        if retake_lora_count.byte_length() > 0 and atol(retake_lora_count) != 0:
            raise Error(
                "LTX2 Retake creator path does not silently apply two-stage request LoRAs"
            )
        _run_request_retake_single_stage(
            contexts_path, negative_contexts_path, contexts_are_projected,
            source_stage2.value(),
            source_audio_path, source_audio_sample_rate,
            source_audio_channels, source_audio_samples,
            video_edit_mode, video_edit_start, video_edit_end,
            regenerate_source_audio,
            out_dir, seed, include_audio, quant, steps, defer_decode,
            source_encode_seconds, total_t0, total_vram_bytes,
            min_free_bytes, ctx,
        )
        return
    print("=== LTX-2.3 REQUEST HQ — VERIFIED CREATOR PROFILE ===")
    print("  final:", REQUEST_HQ_WIDTH, "x", REQUEST_HQ_HEIGHT,
          REQUEST_HQ_NUM_FRAMES, "frames @", REQUEST_HQ_FPS, "fps")
    print("  stage1:", REQUEST_HQ_NF, "x", REQUEST_HQ_NH1, "x",
          REQUEST_HQ_NW1, "S_V=", REQUEST_HQ_S_V1,
          " stage2 S_V=", REQUEST_HQ_S_V2)
    print("  sampler:",
          "Creator fast-distilled Euler" if guidance_mode == String("distilled")
          else "dev CFG-star res_2s", steps,
          "steps + official 3-step stage2; guidance:", guidance_mode,
          "seed:", seed)
    var request_mode = String("T2V")
    if i2v_enabled and last_keyframe_enabled:
        request_mode = String("first/last keyframe interpolation")
    elif last_keyframe_enabled:
        request_mode = String("last-frame keyframe guidance")
    elif i2v_enabled:
        request_mode = String("I2V first-frame conditioning")
    elif v2v_enabled:
        request_mode = (
            String("V2V temporal ") + video_edit_mode
            if temporal_edit_enabled
            else String("V2V full-video conditioning")
        )
    print("  mode:", request_mode)

    var t0 = perf_counter()
    var request_ckpt_path = _request_hq_ckpt(quant)
    print("  transformer precision:", quant, " checkpoint:", request_ckpt_path)
    var ck = ShardedSafeTensors.open(request_ckpt_path)
    var loras = _RequestHQLoraStack.load()
    loras.validate()
    loras.print_summary()
    var workflow_profile = _env_str("LTX2_REQUEST_WORKFLOW_PROFILE")
    var sulphur_creator = (
        workflow_profile == String("sulphur-2-base-distilled-v1")
    )
    var distillation_weight = _request_hq_distillation_weight()
    var request_distill_s1 = _request_hq_distillation_stage_weight(
        String("LTX2_REQUEST_DISTILLATION_S1"),
        REFHQ_LORA_S1 * distillation_weight,
    )
    var request_distill_s2 = _request_hq_distillation_stage_weight(
        String("LTX2_REQUEST_DISTILLATION_S2"),
        REFHQ_LORA_S2 * distillation_weight,
    )
    if workflow_profile.byte_length() > 0:
        print("  creator workflow profile:", workflow_profile)
    print(
        "  [lora] effective distillation stage weights:",
        request_distill_s1, request_distill_s2,
    )
    var preload_lora_factors = quant == String("fp8")
    if preload_lora_factors:
        print("  [lora] preloading factor tensors once for the complete request")
        var n_preloaded = loras.preload_block_factors(ctx)
        print("  [lora] preloaded block factors:", n_preloaded)
    else:
        print(
            "  [lora] factorized per-block streaming to reserve VRAM for ",
            quant,
        )

    print("  [load] stage-1 globals")
    var gw = _load_global_weights_dict(ck, ctx)
    var n_global = loras.apply_to_globals(gw, request_distill_s1, ctx)
    print("  [lora] stage-1 global deltas:", n_global)
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
    # Stage 2 uses a different distillation multiplier. Build its complete
    # global set before the denoise boundary so the stage transition never
    # returns to the checkpoint mapping.
    print("  [load] stage-2 globals before denoise")
    var gw_stage2 = _load_global_weights_dict(ck, ctx)
    var n_global2 = loras.apply_to_globals(
        gw_stage2, request_distill_s2, ctx
    )
    print("  [lora] stage-2 global deltas:", n_global2)
    var g_stage2 = _Globals(
        _clone(gw_stage2[String("patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "patchify_proj.bias", ctx),
        _clone(gw_stage2[String("audio_patchify_proj.weight")][], ctx),
        _load_global_bf16(ck, "audio_patchify_proj.bias", ctx),
        _clone(gw_stage2[String("proj_out.weight")][], ctx),
        _load_global_bf16(ck, "proj_out.bias", ctx),
        _clone(gw_stage2[String("audio_proj_out.weight")][], ctx),
        _load_global_bf16(ck, "audio_proj_out.bias", ctx),
        _load_global_bf16(ck, "scale_shift_table", ctx),
        _load_global_bf16(ck, "audio_scale_shift_table", ctx),
    )
    ctx.synchronize()
    load_seconds = perf_counter() - t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    var conditioning_message = String("Loading projected conditioning") \
        if contexts_are_projected else \
        String("Projecting positive and negative conditioning")
    _write_ltx2_status(
        out_dir, String("running"), String("conditioning"), 0,
        progress_total, conditioning_message,
    )
    t0 = perf_counter()
    var pos_st = ShardedSafeTensors.open(contexts_path)
    var neg_st = ShardedSafeTensors.open(negative_contexts_path)
    var neg_v_key = String("neg_video_context")
    var neg_a_key = String("neg_audio_context")
    var neg_len_key = String("neg_video_len")
    if not _st_has(neg_st, neg_v_key):
        neg_v_key = String("video_context")
        neg_a_key = String("audio_context")
        neg_len_key = String("video_len")
    if not _st_has(neg_st, neg_a_key):
        raise Error("LTX2 request HQ: negative audio context is missing")
    var enc: Tensor
    var aenc: Tensor
    var neg_enc: Tensor
    var neg_aenc: Tensor
    if contexts_are_projected:
        print("  [ctx] consuming explicit post-connector cache")
        enc = Tensor.from_view_as_bf16(
            pos_st.tensor_view("video_context"), ctx
        )
        aenc = Tensor.from_view_as_bf16(
            pos_st.tensor_view("audio_context"), ctx
        )
        neg_enc = Tensor.from_view_as_bf16(
            neg_st.tensor_view(neg_v_key), ctx
        )
        neg_aenc = Tensor.from_view_as_bf16(
            neg_st.tensor_view(neg_a_key), ctx
        )
    else:
        var v_conn = LTX2ConnectorWeights.load(
            request_ckpt_path, String("video_embeddings_connector"),
            LTX2ConnectorConfig.video(), ctx,
        )
        var a_conn = LTX2ConnectorWeights.load(
            request_ckpt_path, String("audio_embeddings_connector"),
            LTX2ConnectorConfig.audio(), ctx,
        )
        var v_pre = Tensor.from_view_as_bf16(
            pos_st.tensor_view("video_context"), ctx
        )
        var a_pre = Tensor.from_view_as_bf16(
            pos_st.tensor_view("audio_context"), ctx
        )
        var vn_pre = Tensor.from_view_as_bf16(
            neg_st.tensor_view(neg_v_key), ctx
        )
        var an_pre = Tensor.from_view_as_bf16(
            neg_st.tensor_view(neg_a_key), ctx
        )
        var pos_valid = _load_ctx_len(pos_st, String("video_len"), ctx)
        var neg_valid = _load_ctx_len(neg_st, neg_len_key, ctx)
        print("  [ctx] valid rows pos/neg:", pos_valid, "/", neg_valid,
              "of", N_TXT)
        enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](
            v_conn, v_pre, ctx, valid_len=pos_valid
        )
        aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](
            a_conn, a_pre, ctx, valid_len=pos_valid
        )
        neg_enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](
            v_conn, vn_pre, ctx, valid_len=neg_valid
        )
        neg_aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](
            a_conn, an_pre, ctx, valid_len=neg_valid
        )
    ctx.synchronize()
    conditioning_seconds = perf_counter() - t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    _write_ltx2_status(
        out_dir, String("running"), String("preparing"), 0,
        progress_total, String("Preparing two-stage latents and scheduler"),
    )
    t0 = perf_counter()
    var vr1 = _request_hq_video_rope(
        REQUEST_HQ_NF, REQUEST_HQ_NH1, REQUEST_HQ_NW1,
        REQUEST_HQ_S_V1, ctx,
    )
    var vr2 = _request_hq_video_rope(
        REQUEST_HQ_NF, REQUEST_HQ_NH2, REQUEST_HQ_NW2,
        REQUEST_HQ_S_V2, ctx,
    )
    if last_keyframe_enabled:
        vr1 = _request_hq_video_rope_with_keyframe(
            REQUEST_HQ_NF, REQUEST_HQ_NH1, REQUEST_HQ_NW1,
            REQUEST_HQ_S_V1_KEYFRAME,
            REQUEST_HQ_NUM_FRAMES - 1, ctx,
        )
        vr2 = _request_hq_video_rope_with_keyframe(
            REQUEST_HQ_NF, REQUEST_HQ_NH2, REQUEST_HQ_NW2,
            REQUEST_HQ_S_V2_KEYFRAME,
            REQUEST_HQ_NUM_FRAMES - 1, ctx,
        )
    var ac = _build_audio_coords_dims(REQUEST_HQ_S_A)
    var arope = _compute_rope(
        ac, 1, REQUEST_HQ_S_A, AD, _mp1(), ROPE_THETA, A_HEADS,
        STDtype.BF16, ctx,
    )
    var caarope = _compute_rope(
        ac, 1, REQUEST_HQ_S_A, CA_DIM, _mp1(), ROPE_THETA, A_HEADS,
        STDtype.BF16, ctx,
    )
    var a_cos = _clone(arope[0], ctx)
    var a_sin = _clone(arope[1], ctx)
    var ca_a_cos = _clone(caarope[0], ctx)
    var ca_a_sin = _clone(caarope[1], ctx)
    var stream = LTX2BlockStream.open(request_ckpt_path)
    if stream.block_count() != NUM_LAYERS:
        raise Error("LTX2 request HQ: stream block_count != 48")
    if quant == String("int4"):
        var int4_slab = _env_str("LTX2_INT4_SLAB")
        if int4_slab.byte_length() == 0:
            raise Error(
                "LTX2 request HQ: quant int4 requires LTX2_INT4_SLAB"
            )
        stream.attach_int4(int4_slab)
        print("  [int4-resident] preloading all 48 dev-model blocks:", int4_slab)
        stream.enable_int4_resident_range(0, NUM_LAYERS - 1, ctx)
        print("  [int4-resident] resident storage bytes:", stream.resident_bytes())
    var product_block_source = _env_str("LTX2_PRODUCT_BLOCK_SOURCE").lower()
    if product_block_source.byte_length() == 0:
        product_block_source = String("resident")
    if (
        product_block_source != String("resident")
        and product_block_source != String("checkpoint_stream")
    ):
        raise Error(
            String("LTX2_PRODUCT_BLOCK_SOURCE must be resident or ")
            + String("checkpoint_stream; got '") + product_block_source
            + String("'")
        )
    var memory_blocks = 0
    var checkpoint_streaming = (
        product_block_source == String("checkpoint_stream")
    )
    if checkpoint_streaming:
        _write_ltx2_status(
            out_dir, String("running"), String("streaming_denoiser"), 0,
            progress_total,
            String("Using bounded checkpoint streaming for denoiser blocks"),
        )
        stream.enable_product_checkpoint_streaming()
        ck.release_to_os()
        print(
            "  [checkpoint-stream] denoiser blocks: bounded one-block GPU ",
            "window; clean checkpoint pages released after every upload",
        )
    else:
        _write_ltx2_status(
            out_dir, String("running"), String("preloading_denoiser"), 0,
            progress_total,
            String("Copying every denoiser block into memory before step 0"),
        )
        memory_blocks = stream.enable_product_memory_resident(ctx)
        ck.release_to_os()
        print(
            "  [memory-resident] denoiser blocks:", memory_blocks, "/",
            NUM_LAYERS, "host bytes:", stream.product_host_bytes(),
        )
    # The official spatial upscaler sits between the two denoise stages. Keep
    # its weights and VAE statistics resident for the whole request; reopening
    # its safetensors after stage 1 would perform physical I/O inside the
    # protected generation interval.
    _write_ltx2_status(
        out_dir, String("running"), String("preloading_upscaler"), 0,
        progress_total,
        String("Loading the stage-2 upscaler into memory before step 0"),
    )
    var request_up_st = ShardedSafeTensors.open(_refhq_spatial_upscaler())
    var request_upsampler = LatentUpsampler(request_up_st, False, ctx)
    var request_up_std = _load_vae_stat(String("std-of-means"), ctx)
    var request_up_mean = _load_vae_stat(String("mean-of-means"), ctx)
    ctx.synchronize()
    request_up_st.release_to_os()
    var ns: NoiseSource
    var video_x: Tensor
    var audio_x: Tensor
    var source_clean1: Optional[Tensor] = None
    var source_mask1: Optional[Tensor] = None
    var source_mask_values1 = List[Float32]()
    var i2v_frame_tokens1 = REQUEST_HQ_NH1 * REQUEST_HQ_NW1
    var source_conditioned_denoise = Float32(1.0) - source_strength
    var last_conditioned_denoise = Float32(1.0) - last_image_strength
    var source_mod_tokens1 = 0
    if i2v_enabled and not last_keyframe_enabled:
        source_mod_tokens1 = i2v_frame_tokens1
    elif v2v_enabled:
        source_mod_tokens1 = REQUEST_HQ_S_V1
    if noise_fixture_path.byte_length() > 0:
        print("  [noise] paired oracle fixture:", noise_fixture_path)
        ns = NoiseSource.fixture(noise_fixture_path)
        if not ns.has_key(String("init_video")) or not ns.has_key(String("init_audio")):
            raise Error("LTX2 request HQ: noise fixture lacks init_video/init_audio")
        video_x = cast_tensor(
            ns.load_key_f32(String("init_video"), ctx), STDtype.BF16, ctx
        )
        audio_x = cast_tensor(
            ns.load_key_f32(String("init_audio"), ctx), STDtype.BF16, ctx
        )
    else:
        ns = NoiseSource.production(seed)
        # rng-contract: mojo-native-not-pytorch-parity
        if last_keyframe_enabled:
            video_x = randn(
                _sh3(1, REQUEST_HQ_S_V1_KEYFRAME, 128),
                seed, STDtype.BF16, ctx,
            )
        else:
            video_x = randn(
                _sh3(1, REQUEST_HQ_S_V1, 128), seed, STDtype.BF16, ctx
            )
        audio_x = randn(
            _sh3(1, REQUEST_HQ_S_A, 128), seed + 1, STDtype.BF16, ctx
        )
    if source_enabled:
        var clean1: Tensor
        var mask1: Tensor
        if last_keyframe_enabled:
            var target_clean1 = zeros_device(
                _sh3(1, REQUEST_HQ_S_V1, 128), STDtype.BF16, ctx
            )
            if i2v_enabled:
                target_clean1 = _request_hq_i2v_stage1_clean(
                    source_stage1.value(), REQUEST_HQ_S_V1,
                    i2v_frame_tokens1, ctx,
                )
            clean1 = _request_hq_keyframe_clean(
                target_clean1, last_stage1.value(), ctx
            )
            source_mask_values1 = _request_hq_keyframe_mask_values(
                REQUEST_HQ_S_V1, i2v_frame_tokens1, i2v_enabled,
                Float32(1.0) - image_strength,
                REQUEST_HQ_KEYFRAME_TOKENS1, last_conditioned_denoise,
            )
            mask1 = _request_hq_v2v_spatial_mask(
                source_mask_values1, REQUEST_HQ_S_V1_KEYFRAME,
                Float32(1.0), ctx,
            )
        elif i2v_enabled:
            clean1 = _request_hq_i2v_stage1_clean(
                source_stage1.value(), REQUEST_HQ_S_V1,
                i2v_frame_tokens1, ctx,
            )
            mask1 = _request_hq_i2v_mask(
                REQUEST_HQ_S_V1, i2v_frame_tokens1,
                source_conditioned_denoise, Float32(1.0), ctx,
            )
        elif temporal_edit_enabled:
            clean1 = source_stage1.value().clone(ctx)
            source_mask_values1 = _request_hq_v2v_temporal_mask_values(
                REQUEST_HQ_NF, REQUEST_HQ_NH1, REQUEST_HQ_NW1,
                Float32(REQUEST_HQ_FPS), video_edit_start, video_edit_end,
                source_conditioned_denoise,
            )
            mask1 = _request_hq_v2v_spatial_mask(
                source_mask_values1, REQUEST_HQ_S_V1, Float32(1.0), ctx
            )
        elif v2v_mask_enabled:
            clean1 = source_stage1.value().clone(ctx)
            source_mask_values1 = _request_hq_v2v_spatial_mask_values(
                video_mask_path, REQUEST_HQ_NF, REQUEST_HQ_NH1,
                REQUEST_HQ_NW1, source_conditioned_denoise,
            )
            mask1 = _request_hq_v2v_spatial_mask(
                source_mask_values1, REQUEST_HQ_S_V1, Float32(1.0), ctx
            )
        else:
            clean1 = source_stage1.value().clone(ctx)
            mask1 = _request_hq_v2v_mask(
                REQUEST_HQ_S_V1, source_conditioned_denoise,
                Float32(1.0), ctx,
            )
        video_x = ltx2_creator_noiser_from_noise(
            clean1, cast_tensor(video_x, STDtype.BF16, ctx), mask1, ctx
        )
        source_clean1 = Optional[Tensor](clean1^)
        source_mask1 = Optional[Tensor](mask1^)
    var sig1 = _ltx2_scheduler_sigmas(steps, REQUEST_HQ_S_V1)
    if sulphur_creator:
        sig1 = ltx2_sulphur_stage1_sigmas(REQUEST_HQ_S_V1)
    # Execute throwaway exact-shape forwards before arming the physical-I/O
    # counter. CUDA/cuDNN lazily page in kernel modules on their first exact
    # invocation. The warmups use only request-owned resident weights and do
    # not mutate the scheduled latent trajectory.
    _write_ltx2_status(
        out_dir, String("running"), String("warming_denoiser"), 0,
        progress_total,
        String("Warming stage-1, upscaler, and stage-2 GPU paths before step 0"),
    )
    var warm_sigma = sig1[0]
    var warm_s_v1 = (
        REQUEST_HQ_S_V1_KEYFRAME
        if last_keyframe_enabled else REQUEST_HQ_S_V1
    )
    var warm_mod = _build_mod_dims(
        ck, gw, warm_sigma, warm_s_v1, REQUEST_HQ_S_A, ctx,
        conditioned_video_tokens=source_mod_tokens1,
        conditioned_video_denoise=source_conditioned_denoise,
        conditioned_video_denoise_values=_request_hq_optional_mask_values(
            source_mask_values1
        ),
    )
    var warm_s1_out: Tuple[Tensor, Tensor]
    if last_keyframe_enabled:
        warm_s1_out = _request_hq_forward_flat[
            REQUEST_HQ_S_V1_KEYFRAME, REQUEST_HQ_S_A,
            REQUEST_HQ_VPAD1_KEYFRAME, REQUEST_HQ_APAD,
        ](
            loras, request_distill_s1, cfg, g, stream,
            video_x, audio_x, enc, aenc, warm_mod, vr1,
            a_cos, a_sin, ca_a_cos, ca_a_sin,
            False, False, True, ctx,
        )
    else:
        warm_s1_out = _request_hq_forward_flat[
            REQUEST_HQ_S_V1, REQUEST_HQ_S_A,
            REQUEST_HQ_VPAD1, REQUEST_HQ_APAD,
        ](
            loras, request_distill_s1, cfg, g, stream,
            video_x, audio_x, enc, aenc, warm_mod, vr1,
            a_cos, a_sin, ca_a_cos, ca_a_sin,
            False, False, True, ctx,
        )
    # Warm the resident upscaler and the larger stage-2 denoiser shape too.
    # Both are first used after stage 1, so a stage-1-only warmup is not enough
    # to prove the complete two-stage request is physically I/O-free.
    var warm_generated_video1 = video_x.clone(ctx)
    if last_keyframe_enabled:
        warm_generated_video1 = slice(
            video_x, 1, 0, REQUEST_HQ_S_V1, ctx
        )
    var warm_v_lat1 = _refhq_unpatchify_video(
        warm_generated_video1,
        REQUEST_HQ_NF,
        REQUEST_HQ_NH1,
        REQUEST_HQ_NW1,
        ctx,
    )
    var warm_upscaled = _refhq_spatial_upscale_resident(
        warm_v_lat1,
        request_upsampler,
        request_up_std,
        request_up_mean,
        ctx,
    )
    var warm_vx2 = cast_tensor(
        _refhq_patchify_video(warm_upscaled, REQUEST_HQ_S_V2, ctx),
        STDtype.BF16,
        ctx,
    )
    var warm_s2sig = ltx2_stage2_distilled_sigmas()
    if sulphur_creator:
        warm_s2sig = ltx2_sulphur_stage2_sigmas()
    var warm_mod2 = _build_mod_dims(
        ck,
        gw_stage2,
        warm_s2sig[0],
        REQUEST_HQ_S_V2,
        REQUEST_HQ_S_A,
        ctx,
        uniform_timestep=True,
    )
    var warm_s2_out = _request_hq_forward_flat[
        REQUEST_HQ_S_V2, REQUEST_HQ_S_A,
        REQUEST_HQ_VPAD2, REQUEST_HQ_APAD,
    ](
        loras, request_distill_s2, cfg, g_stage2, stream,
        warm_vx2, audio_x, enc, aenc, warm_mod2, vr2,
        a_cos, a_sin, ca_a_cos, ca_a_sin,
        False, False, True, ctx,
    )
    # The denoiser warmups above execute every model/LoRA path, but distilled
    # generation also launches Euler update kernels after each forward. Warm
    # those exact stage shapes before the physical-I/O fence; otherwise their
    # first CUDA module fault can occur at step 0 even though all weights are
    # already resident (measured as one 4096-byte read with camera LoRA).
    if guidance_mode == String("distilled"):
        var warm_s1_sched_sigmas = ltx2_distilled_sigmas()
        var warm_s2_sched_sigmas = ltx2_stage2_distilled_sigmas()
        if sulphur_creator:
            warm_s1_sched_sigmas = ltx2_sulphur_stage1_sigmas(
                REQUEST_HQ_S_V1
            )
            warm_s2_sched_sigmas = ltx2_sulphur_stage2_sigmas()
        var warm_s1_sched = LTX2Scheduler(warm_s1_sched_sigmas^)
        var warm_s2_sched = LTX2Scheduler(warm_s2_sched_sigmas^)
        if sulphur_creator:
            _ = ltx2_euler_ancestral_cfgpp_step(
                video_x, warm_s1_out[0], warm_s1_out[0], video_x,
                warm_s1_sched.sigma(0), warm_s1_sched.sigma(1), ctx,
            )
            _ = ltx2_euler_ancestral_cfgpp_step(
                audio_x, warm_s1_out[1], warm_s1_out[1], audio_x,
                warm_s1_sched.sigma(0), warm_s1_sched.sigma(1), ctx,
            )
            _ = ltx2_euler_ancestral_cfgpp_step(
                warm_vx2, warm_s2_out[0], warm_s2_out[0], warm_vx2,
                warm_s2_sched.sigma(0), warm_s2_sched.sigma(1), ctx,
            )
            _ = ltx2_euler_ancestral_cfgpp_step(
                audio_x, warm_s2_out[1], warm_s2_out[1], audio_x,
                warm_s2_sched.sigma(0), warm_s2_sched.sigma(1), ctx,
            )
        else:
            _ = warm_s1_sched.step(video_x, warm_s1_out[0], 0, ctx)
            _ = warm_s1_sched.step(audio_x, warm_s1_out[1], 0, ctx)
            _ = warm_s2_sched.step(warm_vx2, warm_s2_out[0], 0, ctx)
            _ = warm_s2_sched.step(audio_x, warm_s2_out[1], 0, ctx)
    ctx.synchronize()
    prepare_seconds = perf_counter() - t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)
    # Checkpoint streaming is the low-host-RAM fallback, but a one-block GPU
    # window leaves most of a 24 GiB card idle. The exact stage-1/stage-2
    # warmups above establish the request's real activation high-water mark.
    # Fill the remaining device capacity with a persistent BF16 block prefix,
    # retaining 4 GiB of driver-visible headroom for allocator variation. The
    # cached prefix is reused by every denoise eval; only the uncached suffix
    # performs checkpoint I/O. Reassigning `stream` before decode releases the
    # cache, so the VAE still receives a clean denoiser lifetime boundary.
    if checkpoint_streaming:
        var cache_policy = _env_str(
            "LTX2_CHECKPOINT_GPU_CACHE_BLOCKS"
        ).lower()
        if cache_policy.byte_length() == 0:
            cache_policy = String("auto")
        var cache_blocks = 0
        var cache_block_bytes = stream.resident_storage_bytes_for_block(0)
        if cache_policy == String("auto"):
            # Exact-shape warmup leaves dead activation buffers in CUDA's
            # caching pool. They are reusable, but the driver free counter does
            # not include them and would under-size the persistent cache. Trim
            # only freed pool pages after synchronization; live globals,
            # latents, RoPE, and warm outputs remain allocated and visible.
            ctx.synchronize()
            cu_mempool_trim_current(0)
            ctx.synchronize()
            var cache_mem = cu_mem_get_info()
            var cache_reserve_bytes = 4 * 1024 * 1024 * 1024
            var cache_budget = cache_mem.free_bytes - cache_reserve_bytes
            if cache_budget > 0 and cache_block_bytes > 0:
                cache_blocks = cache_budget // cache_block_bytes
                if cache_blocks > NUM_LAYERS:
                    cache_blocks = NUM_LAYERS
        elif cache_policy == String("off") or cache_policy == String("0"):
            cache_blocks = 0
        else:
            cache_blocks = atol(cache_policy)
            if cache_blocks < 0 or cache_blocks > NUM_LAYERS:
                raise Error(
                    String("LTX2_CHECKPOINT_GPU_CACHE_BLOCKS must be auto, off, ")
                    + String("or an integer from 0 to ") + String(NUM_LAYERS)
                    + String("; got '") + cache_policy + String("'")
                )
        if cache_blocks > 0:
            _write_ltx2_status(
                out_dir, String("running"), String("caching_denoiser_gpu"), 0,
                progress_total,
                String("Keeping ") + String(cache_blocks)
                + String(" denoiser blocks resident in GPU memory"),
            )
            stream.enable_fp8_resident_range(0, cache_blocks - 1, ctx)
            ctx.synchronize()
            stream.release_checkpoint_pages()
            var cache_after = cu_mem_get_info()
            print(
                "  [checkpoint-gpu-cache] blocks: 0..", cache_blocks - 1,
                " resident bytes:", stream.resident_bytes(),
                " free bytes after cache:", cache_after.free_bytes,
            )
            min_free_bytes = _record_ltx2_min_free(min_free_bytes)
        else:
            print("  [checkpoint-gpu-cache] disabled by policy or capacity")
    # The complete checkpoint-to-RAM pass is large enough to evict cold pages
    # from this runner image. Re-read the small executable once before arming
    # the zero-I/O boundary so first-use sampler/stat code cannot fault from
    # disk during denoise. This is the same measured product boundary used by
    # the resident Krea worker; it does not touch checkpoint data.
    print(
        "  [hardware-safety] prefaulted runner image bytes:",
        prefault_self_executable(),
    )
    print(
        "  [hardware-safety] locked mapped runtime bytes:",
        lock_mapped_runtime_files(),
    )
    lock_future_mappings()
    print("  [hardware-safety] future runtime mappings locked")
    var denoise_read_bytes_start = _ltx2_proc_read_bytes()
    _write_ltx2_denoise_io(
        out_dir, String("armed"), denoise_read_bytes_start,
        denoise_read_bytes_start,
    )
    print(
        "  [hardware-safety] denoise read_bytes baseline:",
        denoise_read_bytes_start,
    )

    if guidance_mode == String("distilled"):
        if sulphur_creator:
            print("  [Stage1] Sulphur creator distilled workflow")
        else:
            print("  [Stage1] Creator fast-distilled Euler", steps,
                  "steps x 1 evaluation")
    else:
        print("  [Stage1] dev CFG-star res_2s", steps,
              "steps x 2 evaluations x 3 passes")
    var s1_eval = 0
    var first_s1_eval = True
    @parameter
    def _den_s1(
        vx: Tensor, ax: Tensor, sigma: Float32
    ) raises -> Tuple[Tensor, Tensor]:
        if s1_eval % 2 == 0:
            var visible_step = s1_eval // 2 + 1
            if visible_step > steps:
                visible_step = steps
            _write_ltx2_status(
                out_dir, String("running"), String("denoising_stage1"),
                visible_step, progress_total,
                String("Stage 1 step ") + String(visible_step) + String(" of ")
                + String(steps),
            )
        s1_eval += 1
        var active_s_v1 = (
            REQUEST_HQ_S_V1_KEYFRAME
            if last_keyframe_enabled else REQUEST_HQ_S_V1
        )
        var mod = _build_mod_dims(
            ck, gw, sigma, active_s_v1, REQUEST_HQ_S_A, ctx,
            conditioned_video_tokens=source_mod_tokens1,
            conditioned_video_denoise=source_conditioned_denoise,
            conditioned_video_denoise_values=_request_hq_optional_mask_values(
                source_mask_values1
            ),
        )
        var c: Tuple[Tensor, Tensor]
        var u: Tuple[Tensor, Tensor]
        var m: Tuple[Tensor, Tensor]
        if last_keyframe_enabled:
            c = _request_hq_forward_flat[
                REQUEST_HQ_S_V1_KEYFRAME, REQUEST_HQ_S_A,
                REQUEST_HQ_VPAD1_KEYFRAME, REQUEST_HQ_APAD,
            ](
                loras, request_distill_s1, cfg, g, stream, vx, ax,
                enc, aenc, mod, vr1, a_cos, a_sin, ca_a_cos, ca_a_sin,
                False, first_s1_eval, True, ctx,
            )
            u = _request_hq_forward_flat[
                REQUEST_HQ_S_V1_KEYFRAME, REQUEST_HQ_S_A,
                REQUEST_HQ_VPAD1_KEYFRAME, REQUEST_HQ_APAD,
            ](
                loras, request_distill_s1, cfg, g, stream, vx, ax,
                neg_enc, neg_aenc, mod, vr1, a_cos, a_sin, ca_a_cos,
                ca_a_sin, False, False, True, ctx,
            )
            m = _request_hq_forward_flat[
                REQUEST_HQ_S_V1_KEYFRAME, REQUEST_HQ_S_A,
                REQUEST_HQ_VPAD1_KEYFRAME, REQUEST_HQ_APAD,
            ](
                loras, request_distill_s1, cfg, g, stream, vx, ax,
                enc, aenc, mod, vr1, a_cos, a_sin, ca_a_cos, ca_a_sin,
                True, False, True, ctx,
            )
        else:
            c = _request_hq_forward_flat[
                REQUEST_HQ_S_V1, REQUEST_HQ_S_A,
                REQUEST_HQ_VPAD1, REQUEST_HQ_APAD,
            ](
                loras, request_distill_s1, cfg, g, stream, vx, ax,
                enc, aenc, mod, vr1, a_cos, a_sin, ca_a_cos, ca_a_sin,
                False, first_s1_eval, True, ctx,
            )
            u = _request_hq_forward_flat[
                REQUEST_HQ_S_V1, REQUEST_HQ_S_A,
                REQUEST_HQ_VPAD1, REQUEST_HQ_APAD,
            ](
                loras, request_distill_s1, cfg, g, stream, vx, ax,
                neg_enc, neg_aenc, mod, vr1, a_cos, a_sin, ca_a_cos,
                ca_a_sin, False, False, True, ctx,
            )
            m = _request_hq_forward_flat[
                REQUEST_HQ_S_V1, REQUEST_HQ_S_A,
                REQUEST_HQ_VPAD1, REQUEST_HQ_APAD,
            ](
                loras, request_distill_s1, cfg, g, stream, vx, ax,
                enc, aenc, mod, vr1, a_cos, a_sin, ca_a_cos, ca_a_sin,
                True, False, True, ctx,
            )
        first_s1_eval = False
        var c_v = _refhq_x0(vx, c[0], sigma, ctx)
        var u_v = _refhq_x0(vx, u[0], sigma, ctx)
        var m_v = _refhq_x0(vx, m[0], sigma, ctx)
        var c_a = _refhq_x0(ax, c[1], sigma, ctx)
        var u_a = _refhq_x0(ax, u[1], sigma, ctx)
        var m_a = _refhq_x0(ax, m[1], sigma, ctx)
        var guided_v = guider_calculate(
            c_v, u_v, m_v, REFHQ_V_CFG, 0.0, REFHQ_V_RESCALE,
            REFHQ_MOD_SCALE, ctx,
        )
        var guided_a = guider_calculate(
            c_a, u_a, m_a, REFHQ_A_CFG, 0.0, REFHQ_A_RESCALE,
            REFHQ_MOD_SCALE, ctx,
        )
        _assert_ltx2_no_denoise_disk_reads(
            out_dir, denoise_read_bytes_start, checkpoint_streaming
        )
        return (guided_v^, guided_a^)

    t0 = perf_counter()
    if guidance_mode == String("distilled"):
        var s1_distilled_sigmas = ltx2_distilled_sigmas()
        if sulphur_creator:
            s1_distilled_sigmas = ltx2_sulphur_stage1_sigmas(
                REQUEST_HQ_S_V1
            )
            print(
                "  [Stage1] Sulphur creator euler_ancestral_cfg_pp",
                steps, "steps x cond+uncond",
            )
        var s1_sched = LTX2Scheduler(s1_distilled_sigmas^)
        for step in range(s1_sched.num_steps):
            var sigma = s1_sched.sigma(step)
            _write_ltx2_status(
                out_dir, String("running"), String("denoising_stage1"),
                step + 1, progress_total,
                String("Stage 1 step ") + String(step + 1) + String(" of ")
                + String(s1_sched.num_steps),
            )
            var active_s_v1 = (
                REQUEST_HQ_S_V1_KEYFRAME
                if last_keyframe_enabled else REQUEST_HQ_S_V1
            )
            var mod = _build_mod_dims(
                ck, gw, sigma, active_s_v1, REQUEST_HQ_S_A, ctx,
                conditioned_video_tokens=source_mod_tokens1,
                conditioned_video_denoise=source_conditioned_denoise,
                conditioned_video_denoise_values=_request_hq_optional_mask_values(
                    source_mask_values1
                ),
            )
            var c: Tuple[Tensor, Tensor]
            if last_keyframe_enabled:
                c = _request_hq_forward_flat[
                    REQUEST_HQ_S_V1_KEYFRAME, REQUEST_HQ_S_A,
                    REQUEST_HQ_VPAD1_KEYFRAME, REQUEST_HQ_APAD,
                ](
                    loras, request_distill_s1, cfg, g, stream,
                    video_x, audio_x, enc, aenc, mod, vr1,
                    a_cos, a_sin, ca_a_cos, ca_a_sin,
                    False, first_s1_eval, True, ctx,
                )
            else:
                c = _request_hq_forward_flat[
                    REQUEST_HQ_S_V1, REQUEST_HQ_S_A,
                    REQUEST_HQ_VPAD1, REQUEST_HQ_APAD,
                ](
                    loras, request_distill_s1, cfg, g, stream,
                    video_x, audio_x, enc, aenc, mod, vr1,
                    a_cos, a_sin, ca_a_cos, ca_a_sin,
                    False, first_s1_eval, True, ctx,
                )
            first_s1_eval = False
            if sulphur_creator:
                var u: Tuple[Tensor, Tensor]
                if last_keyframe_enabled:
                    u = _request_hq_forward_flat[
                        REQUEST_HQ_S_V1_KEYFRAME, REQUEST_HQ_S_A,
                        REQUEST_HQ_VPAD1_KEYFRAME, REQUEST_HQ_APAD,
                    ](
                        loras, request_distill_s1, cfg, g, stream,
                        video_x, audio_x, neg_enc, neg_aenc, mod, vr1,
                        a_cos, a_sin, ca_a_cos, ca_a_sin,
                        False, False, True, ctx,
                    )
                else:
                    u = _request_hq_forward_flat[
                        REQUEST_HQ_S_V1, REQUEST_HQ_S_A,
                        REQUEST_HQ_VPAD1, REQUEST_HQ_APAD,
                    ](
                        loras, request_distill_s1, cfg, g, stream,
                        video_x, audio_x, neg_enc, neg_aenc, mod, vr1,
                        a_cos, a_sin, ca_a_cos, ca_a_sin,
                        False, False, True, ctx,
                    )
                var cond_v = _refhq_x0(video_x, c[0], sigma, ctx)
                var cond_a = _refhq_x0(audio_x, c[1], sigma, ctx)
                var uncond_v = _refhq_x0(video_x, u[0], sigma, ctx)
                var uncond_a = _refhq_x0(audio_x, u[1], sigma, ctx)
                var ancestral_v: Tensor
                if last_keyframe_enabled:
                    ancestral_v = randn(
                        _sh3(1, REQUEST_HQ_S_V1_KEYFRAME, 128),
                        seed + UInt64(1000 + step * 2),
                        STDtype.BF16, ctx,
                    )
                else:
                    ancestral_v = randn(
                        _sh3(1, REQUEST_HQ_S_V1, 128),
                        seed + UInt64(1000 + step * 2),
                        STDtype.BF16, ctx,
                    )
                var ancestral_a = randn(
                    _sh3(1, REQUEST_HQ_S_A, 128),
                    seed + UInt64(1001 + step * 2),
                    STDtype.BF16, ctx,
                )
                video_x = ltx2_euler_ancestral_cfgpp_step(
                    video_x, cond_v, uncond_v, ancestral_v,
                    sigma, s1_sched.sigma(step + 1), ctx,
                )
                audio_x = ltx2_euler_ancestral_cfgpp_step(
                    audio_x, cond_a, uncond_a, ancestral_a,
                    sigma, s1_sched.sigma(step + 1), ctx,
                )
            else:
                video_x = s1_sched.step(video_x, c[0], step, ctx)
                audio_x = s1_sched.step(audio_x, c[1], step, ctx)
            if source_enabled:
                video_x = res2s_post_process_latent(
                    video_x, source_mask1.value(), source_clean1.value(), ctx
                )
            _assert_ltx2_no_denoise_disk_reads(
                out_dir, denoise_read_bytes_start, checkpoint_streaming
            )
    else:
        var s1_names = List[String]()
        var s1_tensors = List[ArcPointer[Tensor]]()
        var s1_out = res2s_ref_loop[_den_s1](
            sig1, video_x^, audio_x^, ns, String("s1"), s1_names,
            s1_tensors, ctx,
            video_denoise_mask=source_mask1,
            video_clean_latent=source_clean1,
        )
        video_x = s1_out[0].clone(ctx)
        audio_x = s1_out[1].clone(ctx)
    _assert_ltx2_no_denoise_disk_reads(
        out_dir, denoise_read_bytes_start, checkpoint_streaming
    )
    var stage1_read_bytes_end = _ltx2_proc_read_bytes()
    _write_ltx2_denoise_io(
        out_dir,
        (
            String("stage1_checkpoint_stream")
            if checkpoint_streaming else String("stage1_passed")
        ),
        denoise_read_bytes_start, stage1_read_bytes_end,
    )
    print("  [Stage1] complete")

    _write_ltx2_status(
        out_dir, String("running"), String("upscaling"),
        steps, progress_total,
        String("Upscaling stage-1 latents"),
    )
    var generated_video1 = video_x.clone(ctx)
    if last_keyframe_enabled:
        generated_video1 = slice(
            video_x, 1, 0, REQUEST_HQ_S_V1, ctx
        )
    var v_lat1 = _refhq_unpatchify_video(
        generated_video1,
        REQUEST_HQ_NF, REQUEST_HQ_NH1, REQUEST_HQ_NW1, ctx,
    )
    var upscaled = _refhq_spatial_upscale_resident(
        v_lat1,
        request_upsampler,
        request_up_std,
        request_up_mean,
        ctx,
    )
    _ = _stats(String("upscaled"), upscaled, ctx)
    ctx.synchronize()

    print("  [memory-resident] activating preloaded stage-2 globals")
    gw = gw_stage2^
    g = g_stage2^

    var s2sig = ltx2_stage2_distilled_sigmas()
    if sulphur_creator:
        s2sig = ltx2_sulphur_stage2_sigmas()
    var s2_scale = s2sig[0]
    var up_flat = cast_tensor(
        _refhq_patchify_video(upscaled, REQUEST_HQ_S_V2, ctx),
        STDtype.BF16, ctx,
    )
    var source_clean2: Optional[Tensor] = None
    var source_mask2: Optional[Tensor] = None
    var source_mask_values2 = List[Float32]()
    var i2v_frame_tokens2 = REQUEST_HQ_NH2 * REQUEST_HQ_NW2
    var source_mod_tokens2 = 0
    if i2v_enabled and not last_keyframe_enabled:
        source_mod_tokens2 = i2v_frame_tokens2
    elif v2v_enabled:
        source_mod_tokens2 = REQUEST_HQ_S_V2
    var s2n_v: Tensor
    var s2n_a: Tensor
    if noise_fixture_path.byte_length() > 0:
        if not ns.has_key(String("s2init_video")) or not ns.has_key(String("s2init_audio")):
            raise Error("LTX2 request HQ: noise fixture lacks s2init video/audio")
        s2n_v = ns.load_key_f32(String("s2init_video"), ctx)
        s2n_a = ns.load_key_f32(String("s2init_audio"), ctx)
    else:
        # rng-contract: mojo-native-not-pytorch-parity
        var stage2_seed = seed + 100
        if sulphur_creator:
            stage2_seed = UInt64(42)
        if last_keyframe_enabled:
            s2n_v = randn(
                _sh3(1, REQUEST_HQ_S_V2_KEYFRAME, 128),
                stage2_seed, STDtype.BF16, ctx,
            )
        else:
            s2n_v = randn(
                _sh3(1, REQUEST_HQ_S_V2, 128),
                stage2_seed, STDtype.BF16, ctx,
            )
        s2n_a = randn(
            _sh3(1, REQUEST_HQ_S_A, 128), stage2_seed + 1, STDtype.BF16, ctx
        )
    var vx2: Tensor
    if source_enabled:
        var clean2: Tensor
        var mask2: Tensor
        if last_keyframe_enabled:
            var target_clean2 = up_flat.clone(ctx)
            if i2v_enabled:
                target_clean2 = _request_hq_i2v_stage2_clean(
                    source_stage2.value(), up_flat, REQUEST_HQ_S_V2,
                    i2v_frame_tokens2, ctx,
                )
            clean2 = _request_hq_keyframe_clean(
                target_clean2, last_stage2.value(), ctx
            )
            source_mask_values2 = _request_hq_keyframe_mask_values(
                REQUEST_HQ_S_V2, i2v_frame_tokens2, i2v_enabled,
                Float32(1.0) - image_strength,
                REQUEST_HQ_KEYFRAME_TOKENS2, last_conditioned_denoise,
            )
            mask2 = _request_hq_v2v_spatial_mask(
                source_mask_values2, REQUEST_HQ_S_V2_KEYFRAME,
                Float32(1.0), ctx,
            )
        elif i2v_enabled:
            clean2 = _request_hq_i2v_stage2_clean(
                source_stage2.value(), up_flat, REQUEST_HQ_S_V2,
                i2v_frame_tokens2, ctx,
            )
            mask2 = _request_hq_i2v_mask(
                REQUEST_HQ_S_V2, i2v_frame_tokens2,
                source_conditioned_denoise, Float32(1.0), ctx,
            )
        elif temporal_edit_enabled:
            clean2 = source_stage2.value().clone(ctx)
            source_mask_values2 = _request_hq_v2v_temporal_mask_values(
                REQUEST_HQ_NF, REQUEST_HQ_NH2, REQUEST_HQ_NW2,
                Float32(REQUEST_HQ_FPS), video_edit_start, video_edit_end,
                source_conditioned_denoise,
            )
            mask2 = _request_hq_v2v_spatial_mask(
                source_mask_values2, REQUEST_HQ_S_V2, Float32(1.0), ctx
            )
        elif v2v_mask_enabled:
            clean2 = source_stage2.value().clone(ctx)
            source_mask_values2 = _request_hq_v2v_spatial_mask_values(
                video_mask_path, REQUEST_HQ_NF, REQUEST_HQ_NH2,
                REQUEST_HQ_NW2, source_conditioned_denoise,
            )
            mask2 = _request_hq_v2v_spatial_mask(
                source_mask_values2, REQUEST_HQ_S_V2, Float32(1.0), ctx
            )
        else:
            clean2 = source_stage2.value().clone(ctx)
            mask2 = _request_hq_v2v_mask(
                REQUEST_HQ_S_V2, source_conditioned_denoise,
                Float32(1.0), ctx,
            )
        var scaled_mask2 = mul_scalar(mask2, s2_scale, ctx)
        vx2 = ltx2_creator_noiser_from_noise(
            clean2, cast_tensor(s2n_v, STDtype.BF16, ctx), scaled_mask2, ctx
        )
        source_clean2 = Optional[Tensor](clean2^)
        source_mask2 = Optional[Tensor](mask2^)
    else:
        vx2 = _refhq_noise_blend(up_flat, s2n_v, s2_scale, ctx)
    var ax2 = _refhq_noise_blend(audio_x, s2n_a, s2_scale, ctx)
    ctx.synchronize()
    print(
        "  [hardware-safety] stage-2 prefaulted runner image bytes:",
        prefault_self_executable(),
    )
    print(
        "  [hardware-safety] stage-2 locked mapped runtime bytes:",
        lock_mapped_runtime_files(),
    )
    lock_future_mappings()
    print("  [hardware-safety] stage-2 future runtime mappings locked")
    denoise_read_bytes_start = _ltx2_proc_read_bytes()
    _write_ltx2_denoise_io(
        out_dir, String("stage2_armed"), denoise_read_bytes_start,
        denoise_read_bytes_start,
    )
    print(
        "  [hardware-safety] stage-2 denoise read_bytes baseline:",
        denoise_read_bytes_start,
    )
    var s2_eval = 0
    var first_s2_eval = True
    @parameter
    def _den_s2(
        vx: Tensor, ax: Tensor, sigma: Float32
    ) raises -> Tuple[Tensor, Tensor]:
        if s2_eval % 2 == 0:
            var stage2_step = s2_eval // 2 + 1
            if stage2_step > 3:
                stage2_step = 3
            _write_ltx2_status(
                out_dir, String("running"), String("denoising_stage2"),
                steps + stage2_step, progress_total,
                String("Stage 2 step ") + String(stage2_step)
                + String(" of 3"),
            )
        s2_eval += 1
        var active_s_v2 = (
            REQUEST_HQ_S_V2_KEYFRAME
            if last_keyframe_enabled else REQUEST_HQ_S_V2
        )
        var mod = _build_mod_dims(
            ck, gw, sigma, active_s_v2, REQUEST_HQ_S_A, ctx,
            uniform_timestep=True,
            conditioned_video_tokens=source_mod_tokens2,
            conditioned_video_denoise=source_conditioned_denoise,
            conditioned_video_denoise_values=_request_hq_optional_mask_values(
                source_mask_values2
            ),
        )
        var c: Tuple[Tensor, Tensor]
        if last_keyframe_enabled:
            c = _request_hq_forward_flat[
                REQUEST_HQ_S_V2_KEYFRAME, REQUEST_HQ_S_A,
                REQUEST_HQ_VPAD2_KEYFRAME, REQUEST_HQ_APAD,
            ](
                loras, request_distill_s2, cfg, g, stream,
                vx, ax, enc, aenc, mod, vr2,
                a_cos, a_sin, ca_a_cos, ca_a_sin, False,
                first_s2_eval, True, ctx,
            )
        else:
            c = _request_hq_forward_flat[
                REQUEST_HQ_S_V2, REQUEST_HQ_S_A,
                REQUEST_HQ_VPAD2, REQUEST_HQ_APAD,
            ](
                loras, request_distill_s2, cfg, g, stream,
                vx, ax, enc, aenc, mod, vr2,
                a_cos, a_sin, ca_a_cos, ca_a_sin, False,
                first_s2_eval, True, ctx,
            )
        first_s2_eval = False
        var denoised_v = _refhq_x0(vx, c[0], sigma, ctx)
        var denoised_a = _refhq_x0(ax, c[1], sigma, ctx)
        _assert_ltx2_no_denoise_disk_reads(
            out_dir, denoise_read_bytes_start, checkpoint_streaming
        )
        return (denoised_v^, denoised_a^)

    if guidance_mode == String("distilled"):
        if sulphur_creator:
            print("  [Stage2] Sulphur creator LCM 3 steps x 1 evaluation")
        else:
            print("  [Stage2] Creator fast-distilled Euler 3 steps x 1 evaluation")
        var s2_distilled_sigmas = ltx2_stage2_distilled_sigmas()
        if sulphur_creator:
            s2_distilled_sigmas = ltx2_sulphur_stage2_sigmas()
        var s2_sched = LTX2Scheduler(s2_distilled_sigmas^)
        for step in range(s2_sched.num_steps):
            var sigma = s2_sched.sigma(step)
            _write_ltx2_status(
                out_dir, String("running"), String("denoising_stage2"),
                steps + step + 1, progress_total,
                String("Stage 2 step ") + String(step + 1)
                + String(" of ") + String(s2_sched.num_steps),
            )
            var active_s_v2 = (
                REQUEST_HQ_S_V2_KEYFRAME
                if last_keyframe_enabled else REQUEST_HQ_S_V2
            )
            var mod = _build_mod_dims(
                ck, gw, sigma, active_s_v2, REQUEST_HQ_S_A, ctx,
                uniform_timestep=True,
                conditioned_video_tokens=source_mod_tokens2,
                conditioned_video_denoise=source_conditioned_denoise,
                conditioned_video_denoise_values=_request_hq_optional_mask_values(
                    source_mask_values2
                ),
            )
            var c: Tuple[Tensor, Tensor]
            if last_keyframe_enabled:
                c = _request_hq_forward_flat[
                    REQUEST_HQ_S_V2_KEYFRAME, REQUEST_HQ_S_A,
                    REQUEST_HQ_VPAD2_KEYFRAME, REQUEST_HQ_APAD,
                ](
                    loras, request_distill_s2, cfg, g, stream,
                    vx2, ax2, enc, aenc, mod, vr2,
                    a_cos, a_sin, ca_a_cos, ca_a_sin, False,
                    first_s2_eval, True, ctx,
                )
            else:
                c = _request_hq_forward_flat[
                    REQUEST_HQ_S_V2, REQUEST_HQ_S_A,
                    REQUEST_HQ_VPAD2, REQUEST_HQ_APAD,
                ](
                    loras, request_distill_s2, cfg, g, stream,
                    vx2, ax2, enc, aenc, mod, vr2,
                    a_cos, a_sin, ca_a_cos, ca_a_sin, False,
                    first_s2_eval, True, ctx,
                )
            first_s2_eval = False
            if sulphur_creator:
                var denoised_v = _refhq_x0(vx2, c[0], sigma, ctx)
                var denoised_a = _refhq_x0(ax2, c[1], sigma, ctx)
                var lcm_v: Tensor
                if last_keyframe_enabled:
                    lcm_v = randn(
                        _sh3(1, REQUEST_HQ_S_V2_KEYFRAME, 128),
                        UInt64(4200 + step * 2), STDtype.BF16, ctx,
                    )
                else:
                    lcm_v = randn(
                        _sh3(1, REQUEST_HQ_S_V2, 128),
                        UInt64(4200 + step * 2), STDtype.BF16, ctx,
                    )
                var lcm_a = randn(
                    _sh3(1, REQUEST_HQ_S_A, 128),
                    UInt64(4201 + step * 2), STDtype.BF16, ctx,
                )
                vx2 = ltx2_lcm_step(
                    denoised_v, lcm_v, s2_sched.sigma(step + 1), ctx
                )
                ax2 = ltx2_lcm_step(
                    denoised_a, lcm_a, s2_sched.sigma(step + 1), ctx
                )
            else:
                vx2 = s2_sched.step(vx2, c[0], step, ctx)
                ax2 = s2_sched.step(ax2, c[1], step, ctx)
            if source_enabled:
                vx2 = res2s_post_process_latent(
                    vx2, source_mask2.value(), source_clean2.value(), ctx
                )
            _assert_ltx2_no_denoise_disk_reads(
                out_dir, denoise_read_bytes_start, checkpoint_streaming
            )
    else:
        var s2_names = List[String]()
        var s2_tensors = List[ArcPointer[Tensor]]()
        var s2_out = res2s_ref_loop[_den_s2](
            s2sig, vx2^, ax2^, ns, String("s2"), s2_names, s2_tensors, ctx,
            video_denoise_mask=source_mask2,
            video_clean_latent=source_clean2,
        )
        vx2 = s2_out[0].clone(ctx)
        ax2 = s2_out[1].clone(ctx)
    ctx.synchronize()
    _assert_ltx2_no_denoise_disk_reads(
        out_dir, denoise_read_bytes_start, checkpoint_streaming
    )
    var stage2_read_bytes_end = _ltx2_proc_read_bytes()
    _write_ltx2_denoise_io(
        out_dir,
        (
            String("checkpoint_stream_passed")
            if checkpoint_streaming else String("passed")
        ),
        denoise_read_bytes_start, stage2_read_bytes_end,
    )
    unlock_all_mappings()
    print("  [hardware-safety] runtime mapping locks released after denoise")
    denoise_seconds = perf_counter() - t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    var generated_video2 = vx2.clone(ctx)
    if last_keyframe_enabled:
        generated_video2 = slice(
            vx2, 1, 0, REQUEST_HQ_S_V2, ctx
        )
    var v_final = _refhq_unpatchify_video(
        generated_video2,
        REQUEST_HQ_NF, REQUEST_HQ_NH2, REQUEST_HQ_NW2, ctx,
    )
    var a_final = _refhq_unpatchify_audio(ax2, REQUEST_HQ_S_A, ctx)
    var f_names = List[String]()
    var f_tensors = List[ArcPointer[Tensor]]()
    f_names.append(String("video"))
    f_tensors.append(ArcPointer[Tensor](cast_tensor(v_final, STDtype.BF16, ctx)))
    f_names.append(String("audio"))
    f_tensors.append(ArcPointer[Tensor](cast_tensor(a_final, STDtype.BF16, ctx)))
    _refhq_save(
        f_names^, f_tensors^, out_dir + "/final_latents.safetensors", ctx
    )

    # Drop the streamed transformer handle before loading the VAE.
    stream = LTX2BlockStream.open(request_ckpt_path)
    # Persist recovery metadata for every run, not only an explicitly deferred
    # decode. If postprocess fails after final latents are written, the decode
    # command can resume in a fresh process without repeating denoising.
    _write_ltx2_decode_handoff(
        out_dir, steps, seed, include_audio, guidance_mode, quant,
        contexts_path, negative_contexts_path, len(loras.trained),
        load_seconds, conditioning_seconds, source_encode_seconds,
        prepare_seconds,
        denoise_seconds, perf_counter() - total_t0,
        total_vram_bytes, min_free_bytes,
    )
    if defer_decode:
        _write_ltx2_status(
            out_dir, String("running"), String("decode_handoff"),
            progress_total, progress_total,
            String("Releasing LTX2 denoiser before fresh-process decode"),
        )
        print("  [handoff] final latents ready for fresh-process decode")
        return
    _write_ltx2_status(
        out_dir, String("running"), String("decoding_video"),
        progress_total, progress_total,
        String("Decoding ") + String(REQUEST_HQ_NUM_FRAMES)
        + String(" video frames"),
    )
    t0 = perf_counter()
    var vae = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
    var frames = decode_video[
        1, 128, REQUEST_HQ_NF, REQUEST_HQ_NH2, REQUEST_HQ_NW2
    ](vae, cast_tensor(v_final, STDtype.BF16, ctx), ctx)
    var fsh = frames.shape()
    var n_frames_out = fsh[2]
    if n_frames_out != REQUEST_HQ_NUM_FRAMES:
        raise Error("LTX2 request HQ: VAE returned unexpected frame count")
    var raw_video = out_dir + "/hq_frames.rgb"
    save_rgb24_video(
        frames, raw_video, ctx, ValueRange.SIGNED,
        REQUEST_HQ_HEIGHT, REQUEST_HQ_WIDTH,
    )
    ctx.synchronize()
    video_decode_seconds = perf_counter() - t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    var wav_out = String("")
    if include_audio:
        _write_ltx2_status(
            out_dir, String("running"), String("decoding_audio"),
            progress_total, progress_total, String("Decoding audio"),
        )
        t0 = perf_counter()
        wav_out = _refhq_decode_audio_wav(a_final, out_dir, ctx)
        audio_decode_seconds = perf_counter() - t0

    _write_ltx2_status(
        out_dir, String("running"), String("muxing"), progress_total,
        progress_total, String("Muxing video"),
    )
    var mp4_out = out_dir + "/ltx2_t2v_hq.mp4"
    t0 = perf_counter()
    if include_audio:
        _mux_raw_mp4(
            raw_video, REQUEST_HQ_WIDTH, REQUEST_HQ_HEIGHT,
            wav_out, mp4_out, REQUEST_HQ_FPS,
        )
    else:
        _mux_raw_video_mp4(
            raw_video, REQUEST_HQ_WIDTH, REQUEST_HQ_HEIGHT,
            mp4_out, REQUEST_HQ_FPS
        )
    mux_seconds = perf_counter() - t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)
    var executed_sampler = (
        String("euler_ancestral_cfg_pp") if sulphur_creator
        else String("euler") if guidance_mode == String("distilled")
        else String("res2s")
    )
    var executed_scheduler = (
        String("sulphur_creator_8_3") if sulphur_creator
        else String("ltx2_distilled") if guidance_mode == String("distilled")
        else String("ltx2")
    )
    _write_ltx2_result(
        out_dir, mp4_out, REQUEST_HQ_WIDTH, REQUEST_HQ_HEIGHT,
        n_frames_out, REQUEST_HQ_FPS,
        steps, seed, include_audio, executed_sampler,
        executed_scheduler, quant, contexts_path, negative_contexts_path,
        len(loras.trained), load_seconds, conditioning_seconds,
        source_encode_seconds, prepare_seconds, denoise_seconds,
        video_decode_seconds,
        audio_decode_seconds, mux_seconds, perf_counter() - total_t0,
        total_vram_bytes, min_free_bytes,
    )
    _write_ltx2_status(
        out_dir, String("done"), String("done"), progress_total,
        progress_total, String("Video ready"),
    )
    print("=== LTX-2.3 REQUEST HQ DONE ===")
    print("  mp4:", mp4_out)
    print("  frames:", n_frames_out)


def decode_request_profile(
    latents_path: String,
    out_dir: String,
    steps: Int,
    seed: UInt64,
    include_audio: Bool,
    guidance_mode: String,
    quant: String,
    context_path: String,
    negative_context_path: String,
    request_lora_count: Int,
    load_seconds: Float64,
    conditioning_seconds: Float64,
    source_encode_seconds: Float64,
    prepare_seconds: Float64,
    denoise_seconds: Float64,
    elapsed_before_decode: Float64,
    denoise_total_vram_bytes: Int,
    denoise_min_free_bytes: Int,
) raises:
    """Decode a request's exact latent profile in a fresh Mojo process.

    High-resolution denoising can leave the CUDA allocator too fragmented for
    the VAE even after transformer handles are dropped. This entry point keeps
    the authored geometry and compiled decoder unchanged while giving decode a
    clean device context.
    """
    var decode_t0 = perf_counter()
    _write_ltx2_status(
        out_dir, String("running"), String("decoding_video"),
        steps + 3, steps + 3,
        String("Loading fresh LTX2 VAE for ")
        + String(REQUEST_HQ_NUM_FRAMES) + String(" frames"),
    )
    var ctx = DeviceContext()
    var mem0 = cu_mem_get_info()
    var total_vram_bytes = mem0.total_bytes
    if denoise_total_vram_bytes > total_vram_bytes:
        total_vram_bytes = denoise_total_vram_bytes
    var min_free_bytes = mem0.free_bytes
    if denoise_min_free_bytes > 0 and denoise_min_free_bytes < min_free_bytes:
        min_free_bytes = denoise_min_free_bytes

    var st = ShardedSafeTensors.open(latents_path)
    var v_final = Tensor.from_view(st.tensor_view("video"), ctx)
    var a_final = Tensor.from_view(st.tensor_view("audio"), ctx)
    var vsh = v_final.shape()
    if (
        len(vsh) != 5 or vsh[0] != 1 or vsh[1] != 128
        or vsh[2] != REQUEST_HQ_NF
        or vsh[3] != REQUEST_HQ_NH2
        or vsh[4] != REQUEST_HQ_NW2
    ):
        raise Error(
            String("LTX2 request decode: latent geometry does not match ")
            + String(REQUEST_HQ_NF) + String("x")
            + String(REQUEST_HQ_NH2) + String("x")
            + String(REQUEST_HQ_NW2)
        )
    var ash = a_final.shape()
    if (
        len(ash) != 4 or ash[0] != 1 or ash[1] != AUDIO_C
        or ash[2] != REQUEST_HQ_S_A or ash[3] != AUDIO_MEL
    ):
        raise Error("LTX2 request decode: audio latent geometry mismatch")

    var video_t0 = perf_counter()
    var raw_video = out_dir + "/hq_frames.rgb"
    var n_frames_out = 0
    comptime if (
        REQUEST_HQ_NUM_FRAMES == 121
        and REQUEST_HQ_WIDTH == 1920
        and REQUEST_HQ_HEIGHT == 1088
    ):
        n_frames_out = _refhq_decode_video_desktop_tiled_full(
            v_final, out_dir, ctx
        )
    elif (
        REQUEST_HQ_NUM_FRAMES == 121
        and REQUEST_HQ_WIDTH == 1088
        and REQUEST_HQ_HEIGHT == 1920
    ):
        n_frames_out = _refhq_decode_video_desktop_tiled_full_portrait(
            v_final, out_dir, ctx
        )
    elif (
        REQUEST_HQ_WIDTH == 960
        and REQUEST_HQ_HEIGHT == 544
    ):
        n_frames_out = _refhq_decode_video_desktop_tiled_request[
            REQUEST_HQ_NF, REQUEST_HQ_NUM_FRAMES
        ](v_final, out_dir, ctx)
    elif (
        REQUEST_HQ_WIDTH == 544
        and REQUEST_HQ_HEIGHT == 960
    ):
        n_frames_out = _refhq_decode_video_desktop_tiled_request_portrait[
            REQUEST_HQ_NF, REQUEST_HQ_NUM_FRAMES
        ](v_final, out_dir, ctx)
    else:
        var vae = LTX2VaeDecoderWeights.load(
        _request_hq_video_vae_ckpt(_ckpt_bf16()), ctx
    )
        var decoded = decode_video[
            1, 128, REQUEST_HQ_NF, REQUEST_HQ_NH2, REQUEST_HQ_NW2
        ](vae, cast_tensor(v_final, STDtype.BF16, ctx), ctx)
        var dsh = decoded.shape()
        n_frames_out = dsh[2]
        save_rgb24_video(
            decoded, raw_video, ctx, ValueRange.SIGNED,
            REQUEST_HQ_HEIGHT, REQUEST_HQ_WIDTH,
        )
    if n_frames_out != REQUEST_HQ_NUM_FRAMES:
        raise Error("LTX2 request decode: VAE returned unexpected frame count")
    ctx.synchronize()
    var video_decode_seconds = perf_counter() - video_t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    var wav_out = String("")
    var audio_decode_seconds = Float64(0.0)
    if include_audio:
        _write_ltx2_status(
            out_dir, String("running"), String("decoding_audio"),
            steps + 3, steps + 3, String("Decoding audio in fresh process"),
        )
        var audio_t0 = perf_counter()
        wav_out = _refhq_decode_audio_wav(a_final, out_dir, ctx)
        audio_decode_seconds = perf_counter() - audio_t0
        min_free_bytes = _record_ltx2_min_free(min_free_bytes)

    _write_ltx2_status(
        out_dir, String("running"), String("muxing"),
        steps + 3, steps + 3, String("Muxing video"),
    )
    var mp4_out = out_dir + "/ltx2_t2v_hq.mp4"
    var mux_t0 = perf_counter()
    comptime if (
        (
            REQUEST_HQ_NUM_FRAMES == 121
            and (
                (
                    REQUEST_HQ_WIDTH == 1920
                    and REQUEST_HQ_HEIGHT == 1088
                )
                or (
                    REQUEST_HQ_WIDTH == 1088
                    and REQUEST_HQ_HEIGHT == 1920
                )
            )
        )
        or (
            REQUEST_HQ_WIDTH == 960
            and REQUEST_HQ_HEIGHT == 544
        )
        or (
            REQUEST_HQ_WIDTH == 544
            and REQUEST_HQ_HEIGHT == 960
        )
    ):
        if include_audio:
            _request_mux_av_3digit(
                out_dir, wav_out, mp4_out, REQUEST_HQ_FPS,
                REQUEST_HQ_NUM_FRAMES,
            )
        else:
            _request_mux_video_3digit(
                out_dir, mp4_out, REQUEST_HQ_FPS, REQUEST_HQ_NUM_FRAMES
            )
    else:
        if include_audio:
            _mux_raw_mp4(
                raw_video, REQUEST_HQ_WIDTH, REQUEST_HQ_HEIGHT,
                wav_out, mp4_out, REQUEST_HQ_FPS,
            )
        else:
            _mux_raw_video_mp4(
                raw_video, REQUEST_HQ_WIDTH, REQUEST_HQ_HEIGHT,
                mp4_out, REQUEST_HQ_FPS,
            )
    var mux_seconds = perf_counter() - mux_t0
    min_free_bytes = _record_ltx2_min_free(min_free_bytes)
    var sulphur_creator = (
        _env_str("LTX2_REQUEST_WORKFLOW_PROFILE")
        == String("sulphur-2-base-distilled-v1")
    )
    var executed_sampler = (
        String("euler_ancestral_cfg_pp") if sulphur_creator
        else String("euler") if guidance_mode == String("distilled")
        else String("res2s")
    )
    var executed_scheduler = (
        String("sulphur_creator_8_3") if sulphur_creator
        else String("ltx2_distilled") if guidance_mode == String("distilled")
        else String("ltx2")
    )
    _write_ltx2_result(
        out_dir, mp4_out, REQUEST_HQ_WIDTH, REQUEST_HQ_HEIGHT,
        n_frames_out, REQUEST_HQ_FPS,
        steps, seed, include_audio, executed_sampler,
        executed_scheduler, quant, context_path, negative_context_path,
        request_lora_count, load_seconds, conditioning_seconds,
        source_encode_seconds, prepare_seconds, denoise_seconds,
        video_decode_seconds,
        audio_decode_seconds, mux_seconds,
        elapsed_before_decode + (perf_counter() - decode_t0),
        total_vram_bytes, min_free_bytes,
    )
    _write_ltx2_status(
        out_dir, String("done"), String("done"), steps + 3,
        steps + 3, String("Video ready"),
    )
    print("=== LTX-2.3 REQUEST FRESH DECODE DONE ===")
    print("  mp4:", mp4_out)
    print("  frames:", n_frames_out)


# ── STEP-1 EVAL-1 dump mode (divergence hunt): identical setup to run_refhq,
# but runs ONE guided eval at sigma=sig1[0] with full instrumentation and
# saves everything to <out_dir>/step1_mojo_dump.safetensors, then exits.
def run_refhq_step1(
    contexts_path: String,
    noises_path: String,
    out_dir: String,
    apply_lora: Bool,
    detailed: Bool = True,
) raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2.3 refhq STEP-1 DUMP ===")
    print("  contexts:", contexts_path)
    print("  noises:", noises_path)
    print("  out:", out_dir)
    print("  lora:", apply_lora)
    print("  detailed:", detailed)

    var ck = ShardedSafeTensors.open(_refhq_ckpt_fp8())
    var lora = LoraSet.load(_refhq_lora_distilled())
    if lora.format != FMT_LTX2_DISTILLED:
        raise Error("refhq1: distilled LoRA not FMT_LTX2_DISTILLED")
    var lora_mult = REFHQ_LORA_S1 if apply_lora else Float32(0.0)
    if apply_lora:
        print("  [lora] block deltas: mmap-cached / per-module GPU fusion")

    var gw = _load_global_weights_dict(ck, ctx)
    if apply_lora:
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

    var v_conn = LTX2ConnectorWeights.load(
        _refhq_ckpt_fp8(), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_conn = LTX2ConnectorWeights.load(
        _refhq_ckpt_fp8(), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    )
    var cf = ShardedSafeTensors.open(contexts_path)
    var v_pre = Tensor.from_view_as_bf16(cf.tensor_view("video_context"), ctx)
    var a_pre = Tensor.from_view_as_bf16(cf.tensor_view("audio_context"), ctx)
    var vn_pre = Tensor.from_view_as_bf16(cf.tensor_view("neg_video_context"), ctx)
    var an_pre = Tensor.from_view_as_bf16(cf.tensor_view("neg_audio_context"), ctx)
    var pos_valid = _load_ctx_len(cf, String("video_len"), ctx)
    var neg_valid = _load_ctx_len(cf, String("neg_video_len"), ctx)
    print("  [ctx] valid rows: pos=", pos_valid, " neg=", neg_valid, "/", N_TXT)
    var enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](
        v_conn, v_pre, ctx, valid_len=pos_valid
    )
    var aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](
        a_conn, a_pre, ctx, valid_len=pos_valid
    )
    var neg_enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](
        v_conn, vn_pre, ctx, valid_len=neg_valid
    )
    var neg_aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](
        a_conn, an_pre, ctx, valid_len=neg_valid
    )

    var vr1 = _refhq_video_rope(REFHQ_NF, REFHQ_NH1, REFHQ_NW1, REFHQ_S_V1, ctx)
    var ac = _build_audio_coords_dims(REFHQ_S_A)
    var arope = _compute_rope(
        ac, 1, REFHQ_S_A, AD, _mp1(), ROPE_THETA, A_HEADS, STDtype.BF16, ctx
    )
    var caarope = _compute_rope(
        ac, 1, REFHQ_S_A, CA_DIM, _mp1(), ROPE_THETA, A_HEADS, STDtype.BF16, ctx
    )
    var a_cos = _clone(arope[0], ctx); var a_sin = _clone(arope[1], ctx)
    var ca_a_cos = _clone(caarope[0], ctx); var ca_a_sin = _clone(caarope[1], ctx)

    var stream = LTX2BlockStream.open(_refhq_ckpt_fp8())
    stream.enable_fp8_resident_range(RESIDENT_FIRST, RESIDENT_LAST_STAGED, ctx)

    var ns = NoiseSource.fixture(noises_path)
    var video_x = cast_tensor(
        ns.load_key_f32(String("init_video"), ctx), STDtype.BF16, ctx
    )
    var audio_x = cast_tensor(
        ns.load_key_f32(String("init_audio"), ctx), STDtype.BF16, ctx
    )

    var sig1 = _refhq_creator_stage1_sigmas()
    var sigma = sig1[0]
    print("  [step1] sigma =", sigma)

    var dn = List[String]()
    var dt = List[ArcPointer[Tensor]]()
    # The detailed product-geometry dump retains more than 14 GiB of debug
    # tensors and cannot complete all three guidance passes on a 16 GiB card.
    # Lite mode uses the exact non-retaining inference forward and preserves
    # only velocities/x0/guider outputs needed for Creator first-forward parity.
    if detailed:
        _dump_f32(dn, dt, String("enc"), enc, ctx)
        _dump_f32(dn, dt, String("aenc"), aenc, ctx)
        _dump_f32(dn, dt, String("neg_enc"), neg_enc, ctx)
        _dump_f32(dn, dt, String("neg_aenc"), neg_aenc, ctx)
        _dump_f32(dn, dt, String("v_pe_cos"), vr1.v_cos, ctx)
        _dump_f32(dn, dt, String("v_pe_sin"), vr1.v_sin, ctx)
        _dump_f32(dn, dt, String("v_cpe_cos"), vr1.ca_v_cos, ctx)
        _dump_f32(dn, dt, String("v_cpe_sin"), vr1.ca_v_sin, ctx)
        _dump_f32(dn, dt, String("a_pe_cos"), a_cos, ctx)
        _dump_f32(dn, dt, String("a_pe_sin"), a_sin, ctx)
        _dump_f32(dn, dt, String("a_cpe_cos"), ca_a_cos, ctx)
        _dump_f32(dn, dt, String("a_cpe_sin"), ca_a_sin, ctx)
    _dump_f32(dn, dt, String("init_v"), video_x, ctx)
    _dump_f32(dn, dt, String("init_a"), audio_x, ctx)

    var mod = _build_mod_dims(
        ck, gw, sigma, REFHQ_S_V1, REFHQ_S_A, ctx
    )
    if detailed:
        _dump_f32(dn, dt, String("v_timesteps"), mod.v_temb, ctx)
        _dump_f32(dn, dt, String("a_timesteps"), mod.a_temb, ctx)
        _dump_f32(dn, dt, String("v_embedded"), mod.v_embedded, ctx)
        _dump_f32(dn, dt, String("a_embedded"), mod.a_embedded, ctx)
        _dump_f32(dn, dt, String("v_ca_ss"), mod.v_ca_ss, ctx)
        _dump_f32(dn, dt, String("a_ca_ss"), mod.a_ca_ss, ctx)
        _dump_f32(dn, dt, String("v_ca_gate"), mod.v_ca_gate, ctx)
        _dump_f32(dn, dt, String("a_ca_gate"), mod.a_ca_gate, ctx)
        _dump_f32(dn, dt, String("v_prompt_ts"), mod.v_prompt_ts, ctx)
        _dump_f32(dn, dt, String("a_prompt_ts"), mod.a_prompt_ts, ctx)

    # pass 0: cond
    var c: Tuple[Tensor, Tensor]
    if detailed:
        c = _refhq_forward_flat_dump[
            REFHQ_S_V1, REFHQ_S_A, REFHQ_SPAD, REFHQ_SPAD
        ](
            lora, lora_mult, cfg, g, stream, video_x, audio_x, enc, aenc, mod,
            vr1, a_cos, a_sin, ca_a_cos, ca_a_sin, False, String("p0"), dn, dt, ctx,
        )
    else:
        c = _refhq_forward_flat[
            REFHQ_S_V1, REFHQ_S_A, REFHQ_SPAD, REFHQ_SPAD
        ](
            lora, lora_mult, cfg, g, stream, video_x, audio_x, enc, aenc, mod,
            vr1, a_cos, a_sin, ca_a_cos, ca_a_sin, False, False, ctx,
        )
        _dump_f32(dn, dt, String("p0_vel_v"), c[0], ctx)
        _dump_f32(dn, dt, String("p0_vel_a"), c[1], ctx)
    print("  [step1] cond pass done")
    # pass 1: uncond (negative contexts)
    var u: Tuple[Tensor, Tensor]
    if detailed:
        u = _refhq_forward_flat_dump[
            REFHQ_S_V1, REFHQ_S_A, REFHQ_SPAD, REFHQ_SPAD
        ](
            lora, lora_mult, cfg, g, stream, video_x, audio_x, neg_enc,
            neg_aenc, mod, vr1, a_cos, a_sin, ca_a_cos, ca_a_sin, False,
            String("p1"), dn, dt, ctx,
        )
    else:
        u = _refhq_forward_flat[
            REFHQ_S_V1, REFHQ_S_A, REFHQ_SPAD, REFHQ_SPAD
        ](
            lora, lora_mult, cfg, g, stream, video_x, audio_x, neg_enc,
            neg_aenc, mod, vr1, a_cos, a_sin, ca_a_cos, ca_a_sin, False,
            False, ctx,
        )
        _dump_f32(dn, dt, String("p1_vel_v"), u[0], ctx)
        _dump_f32(dn, dt, String("p1_vel_a"), u[1], ctx)
    print("  [step1] uncond pass done")
    # pass 2: mod (positive contexts, cross-modal skipped)
    var m: Tuple[Tensor, Tensor]
    if detailed:
        m = _refhq_forward_flat_dump[
            REFHQ_S_V1, REFHQ_S_A, REFHQ_SPAD, REFHQ_SPAD
        ](
            lora, lora_mult, cfg, g, stream, video_x, audio_x, enc, aenc, mod,
            vr1, a_cos, a_sin, ca_a_cos, ca_a_sin, True, String("p2"), dn, dt, ctx,
        )
    else:
        m = _refhq_forward_flat[
            REFHQ_S_V1, REFHQ_S_A, REFHQ_SPAD, REFHQ_SPAD
        ](
            lora, lora_mult, cfg, g, stream, video_x, audio_x, enc, aenc, mod,
            vr1, a_cos, a_sin, ca_a_cos, ca_a_sin, True, False, ctx,
        )
        _dump_f32(dn, dt, String("p2_vel_v"), m[0], ctx)
        _dump_f32(dn, dt, String("p2_vel_a"), m[1], ctx)
    print("  [step1] mod pass done")

    var c_v = _refhq_x0(video_x, c[0], sigma, ctx)
    var u_v = _refhq_x0(video_x, u[0], sigma, ctx)
    var m_v = _refhq_x0(video_x, m[0], sigma, ctx)
    var c_a = _refhq_x0(audio_x, c[1], sigma, ctx)
    var u_a = _refhq_x0(audio_x, u[1], sigma, ctx)
    var m_a = _refhq_x0(audio_x, m[1], sigma, ctx)
    _dump_f32(dn, dt, String("x0_cond_v"), c_v, ctx)
    _dump_f32(dn, dt, String("x0_uncond_v"), u_v, ctx)
    _dump_f32(dn, dt, String("x0_mod_v"), m_v, ctx)
    _dump_f32(dn, dt, String("x0_cond_a"), c_a, ctx)
    _dump_f32(dn, dt, String("x0_uncond_a"), u_a, ctx)
    _dump_f32(dn, dt, String("x0_mod_a"), m_a, ctx)
    var den_v = guider_calculate(
        c_v, u_v, m_v, REFHQ_V_CFG, 0.0, REFHQ_V_RESCALE, REFHQ_MOD_SCALE, ctx
    )
    var den_a = guider_calculate(
        c_a, u_a, m_a, REFHQ_A_CFG, 0.0, REFHQ_A_RESCALE, REFHQ_MOD_SCALE, ctx
    )
    _dump_f32(dn, dt, String("guided_v"), den_v, ctx)
    _dump_f32(dn, dt, String("guided_a"), den_a, ctx)

    _refhq_save(dn^, dt^, out_dir + "/step1_mojo_dump.safetensors", ctx)
    print("=== STEP-1 DUMP DONE ===")


def _mkdir(path: String) raises:
    var cmd = String("mkdir -p ") + path + " >/dev/null 2>&1"
    _ = sys_system(cmd)


# ── argv-driven entry ──
#   hq                                                   -> 25-frame 768x512 LoRA resident A/V target
#   hq long [base|lora] [resident|stream] [audio|noaudio] [nag|nonag] [out] [steps]
#   hq audiosync [base|lora] [resident|stream] [audio|noaudio] [nag|nonag] [out] [steps]
#   hq single [base|lora] [stream|resident] [audio|noaudio] [nag|nonag] [out] [steps]
#   hq staged [base|lora] [stream|resident] [audio|noaudio] [nag|nonag] [s1out|full] [out] [steps]
# Audio is on by default. `noaudio` is only an explicit debug escape hatch.
# `s1out` (staged only): STAGE-1 PRODUCT mode — decode+mux the measured-clean
# stage-1 output and skip the corrupting upsampler+stage-2 leg (dot/knit mesh;
# see the staged s1out branch in run_staged). `full` = explicit default.
def main() raises:
    var a = argv()
    # Fresh-process official x2 latent-cache producer.  Run after the stage-1
    # process exits and before the fresh stage-2 process starts:
    #   hq refhq-upscale <stage1_final.safetensors> <out_dir>
    if len(a) >= 2 and String(a[1]) == "refhq-upscale":
        if len(a) != 4 and len(a) != 5:
            raise Error(
                "usage: refhq-upscale <stage1_final.safetensors> <out_dir>"
                " [NFxNHxNW]"
            )
        var u_nf = 0
        var u_nh = 0
        var u_nw = 0
        if len(a) == 5:
            var spec = String(a[4])
            var parts = spec.split("x")
            if len(parts) != 3:
                raise Error(
                    String("refhq-upscale: dimensions must be NFxNHxNW, got ")
                    + spec
                )
            u_nf = Int(atol(String(parts[0])))
            u_nh = Int(atol(String(parts[1])))
            u_nw = Int(atol(String(parts[2])))
        _mkdir(String(a[3]))
        run_refhq_upscale(String(a[2]), String(a[3]), u_nf, u_nh, u_nw)
        return
    # Fresh-process decode for a preserved refhq stage-1 product:
    #   hq refhq-decode <final_latents.safetensors> <out_dir>
    if len(a) >= 2 and String(a[1]) == "refhq-decode":
        if len(a) != 4:
            raise Error(
                "usage: refhq-decode <final_latents.safetensors> <out_dir>"
            )
        run_refhq_decode(String(a[2]), String(a[3]))
        return
    # refhq parity mode:
    #   hq refhq <contexts.safetensors> <noises.safetensors|-> <out_dir> [steps]
    # refhq1 step-1 dump modes (divergence hunt): detailed retains selected
    # block states; lite is the 16 GiB-safe velocity/guider parity capture.
    #   hq refhq1[lite] <contexts> <noises> <out_dir> [lora|nolora]
    if len(a) >= 2 and (
        String(a[1]) == "refhq1" or String(a[1]) == "refhq1lite"
    ):
        if len(a) < 5:
            raise Error(
                "usage: refhq1[lite] <contexts.safetensors> <noises.safetensors> "
                "<out_dir> [lora|nolora]"
            )
        var s1_lora = True
        if len(a) >= 6 and String(a[5]) == "nolora":
            s1_lora = False
        _mkdir(String(a[4]))
        run_refhq_step1(
            String(a[2]), String(a[3]), String(a[4]), s1_lora,
            detailed=String(a[1]) == "refhq1",
        )
        return
    if len(a) >= 2 and String(a[1]) == "refhq":
        if len(a) < 5:
            raise Error(
                "usage: refhq <contexts.safetensors> <noises.safetensors|-> "
                "<out_dir> [steps]"
            )
        var refhq_steps = REFHQ_STEPS
        var refhq_lora = True
        var refhq_s1out = False
        var refhq_defer_decode = False
        var refhq_stage1_max_steps = 0
        var refhq_seed = SEED
        var expect_refhq_seed = False
        var refhq_stage1_cache = String("")
        var expect_refhq_stage1_cache = False
        var refhq_upscaler_cache = String("")
        var expect_refhq_upscaler_cache = False
        var refhq_stage2_only = False
        var refhq_stage1_only = False
        for ai in range(5, len(a)):
            var tok = String(a[ai])
            if expect_refhq_upscaler_cache:
                refhq_upscaler_cache = tok
                expect_refhq_upscaler_cache = False
            elif expect_refhq_stage1_cache:
                refhq_stage1_cache = tok
                expect_refhq_stage1_cache = False
            elif expect_refhq_seed:
                refhq_seed = UInt64(atol(tok))
                expect_refhq_seed = False
            elif tok == "stage1-cache":
                expect_refhq_stage1_cache = True
            elif tok == "upscaler-cache":
                expect_refhq_upscaler_cache = True
            elif tok == "stage2-only":
                refhq_stage2_only = True
            elif tok == "stage1-only":
                refhq_stage1_only = True
            elif tok == "seed":
                expect_refhq_seed = True
            elif tok == "nolora":
                refhq_lora = False
            elif tok == "lora":
                refhq_lora = True
            elif tok == "s1out":
                refhq_s1out = True
            elif tok == "full":
                refhq_s1out = False
            elif tok == "denoise-only":
                refhq_defer_decode = True
            elif tok == "stage1-step0":
                refhq_stage1_max_steps = 1
            else:
                refhq_steps = atol(tok)
        if expect_refhq_seed:
            raise Error("refhq: seed requires an integer value")
        if expect_refhq_stage1_cache:
            raise Error("refhq: stage1-cache requires a safetensors path")
        if expect_refhq_upscaler_cache:
            raise Error("refhq: upscaler-cache requires a safetensors path")
        var refhq_out = String(a[4])
        _mkdir(refhq_out)
        run_refhq(
            String(a[2]), String(a[3]), refhq_out, refhq_steps,
            refhq_lora, refhq_s1out, refhq_defer_decode,
            refhq_stage1_max_steps,
            refhq_seed,
            refhq_stage1_cache,
            refhq_upscaler_cache,
            refhq_stage2_only,
            refhq_stage1_only,
        )
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
    var stage1_product = False
    var max_steps = 0
    var out_dir = serenity_output(String("ltx2_hq_long"))
    if mode == "single":
        out_dir = serenity_output(String("ltx2_hq"))
    elif mode == "audiosync":
        out_dir = serenity_output(String("ltx2_audiosync_97f"))
    elif mode == "staged":
        out_dir = serenity_output(String("ltx2_hq2"))
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
        elif tok == "s1out":
            if mode != "staged":
                raise Error(
                    "s1out is a STAGED-mode token (stage-1 product early exit);"
                    " use `hq staged ... s1out`"
                )
            stage1_product = True
        elif tok == "full":
            if mode != "staged":
                raise Error(
                    "full is a STAGED-mode token (explicit two-stage default);"
                    " use `hq staged ... full`"
                )
            stage1_product = False
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
            use_nag, profile_dit, stage1_product,
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


def _write_text(path: String, text: String) raises:
    var n = text.byte_length()
    var buf = alloc[UInt8](n)
    var src = text.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        buf.free()
        raise Error(String("write_text: cannot open ") + path)
    var bp = BytePtr(unsafe_from_address=Int(buf))
    var done = 0
    while done < n:
        var got = sys_pwrite(fd, bp + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    buf.free()
    if done != n:
        raise Error("write_text: short write")


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
def _mux_video_mp4(
    out_dir: String, mp4: String, frame_prefix: String,
    fps: Float64 = 24.0,
) raises:
    var cmd = String("ffmpeg -y -framerate ") + String(fps) + String(" -i ")
    cmd += out_dir + "/" + frame_prefix + "%02d.png"
    cmd += " -c:v libx264 -pix_fmt yuv420p -movflags +faststart " + mp4
    cmd += " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        print("  [mux] WARNING: ffmpeg returned", rc, "(frames still saved)")


def _mux_raw_video_mp4(
    raw_video: String, width: Int, height: Int, mp4: String,
    fps: Float64 = 24.0,
) raises:
    var cmd = String("ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size ")
    cmd += String(width) + String("x") + String(height)
    cmd += String(" -framerate ") + String(fps) + String(" -i ") + raw_video
    cmd += " -c:v libx264 -pix_fmt yuv420p -movflags +faststart " + mp4
    cmd += " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        print("  [mux] WARNING: ffmpeg returned", rc, "(raw video still saved)")


def _mux_mp4(
    out_dir: String, n_frames: Int, wav: String, mp4: String,
    frame_prefix: String, fps: Float64 = 24.0,
) raises:
    var cmd = String("ffmpeg -y -framerate ") + String(fps) + String(" -i ")
    cmd += out_dir + "/" + frame_prefix + "%02d.png -i " + wav
    cmd += " -c:v libx264 -pix_fmt yuv420p -af apad -c:a aac -shortest -movflags +faststart " + mp4
    cmd += " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        print("  [mux] WARNING: ffmpeg returned", rc, "(frames+wav still saved)")


def _mux_raw_mp4(
    raw_video: String, width: Int, height: Int, wav: String, mp4: String,
    fps: Float64 = 24.0,
) raises:
    var cmd = String("ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size ")
    cmd += String(width) + String("x") + String(height)
    cmd += String(" -framerate ") + String(fps) + String(" -i ") + raw_video
    cmd += String(" -i ") + wav
    cmd += " -c:v libx264 -pix_fmt yuv420p -af apad -c:a aac -shortest"
    cmd += " -movflags +faststart " + mp4
    cmd += " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        print(
            "  [mux] WARNING: ffmpeg returned", rc,
            "(raw video+wav still saved)",
        )
