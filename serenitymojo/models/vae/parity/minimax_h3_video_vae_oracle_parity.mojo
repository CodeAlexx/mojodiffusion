# serenitymojo/models/vae/parity/minimax_h3_video_vae_oracle_parity.mojo
#
# NUMERIC PARITY gate (phase [3], done after all — see
# minimax_h3_video_vae_real_weight_gate.mojo's header for why phase [3] was
# initially flagged as possibly impractical and how that assessment was
# overturned): the vendor's OWN `AutoencoderKLLegacy` (klvae.py), loaded with
# the REAL checkpoint via `load_state_dict(strict=False)` — missing=0,
# unexpected=0, confirming no hidden key-conversion — run on a FIXED random
# input, dumped via `scripts` (see below), and compared against THIS
# rebuild's real-weight output on the BYTE-IDENTICAL input (same tensor,
# NCDHW->NDHWC permuted, no regeneration).
#
# ORACLE GENERATION (not run from Mojo — Python, GPU, real weights):
#   /tmp/.../scratchpad/vvae_oracle_gen.py (this session's scratchpad; the
#   commands below reproduce it against any checkpoint copy):
#     1. AutoencoderKLLegacy(**source/config.json) instantiated, real weights
#        loaded via safetensors.torch.load_file + load_state_dict(strict=False)
#        (missing=0, unexpected=0).
#     2. encoder: x = torch.randn(1,3,4,32,32) NCDHW, seed 1234 ->
#        model.encode(x) -> moments (1,48,1,2,2) NCDHW. Both permuted to
#        NDHWC and dumped to vvae_encoder_oracle.safetensors
#        (x_ndhwc, moments_ndhwc).
#     3. decoder: z = torch.randn(1,24,1,1,1) NCDHW -> model.decode(z) ->
#        pixels (1,3,4,16,16) NCDHW. Permuted to NDHWC, dumped to
#        vvae_decoder_oracle.safetensors (z_ndhwc, pixels_ndhwc).
#   Oracle magnitudes (informational, same order as this file's own
#   real-weight-gate run on a DIFFERENT random input, cross-checked before
#   this byte-identical gate existed): encoder mean_abs 4.062 max_abs 8.095;
#   decoder mean_abs 0.693 max_abs 1.984.
#
# BAR: cos >= 0.999 (numeric-parity-testing discipline: forward pass /
# transform, direction+magnitude both matter), max_abs and the L2 magnitude
# ratio reported alongside since cos alone is magnitude-blind.
#
# Run (package-relative imports need -I .; needs the cuDNN shim, which also
# provides the F32 conv3d path's linked symbols):
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/vae/parity/minimax_h3_video_vae_oracle_parity.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.parity import ParityHarness
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    minimax_h3_video_released_encoder_config,
    minimax_h3_video_encode_device,
)
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    minimax_h3_video_released_decoder_config,
    minimax_h3_video_decoder_device_load,
    minimax_h3_video_decode_device,
)

comptime ENC_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/video_vae/source"
comptime DEC_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/video_vae/source"
comptime ENC_ORACLE = "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/scratchpad/vvae_encoder_oracle.safetensors"
comptime DEC_ORACLE = "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/scratchpad/vvae_decoder_oracle.safetensors"

comptime DEC_S = 6
comptime DEC_HEADS = 32
comptime DEC_DH = 64


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_shape(ref st: SafeTensors, name: String) raises -> List[Int]:
    return st.tensor_info(name).shape.copy()


def _l2(v: List[Float32]) -> Float64:
    var acc = Float64(0.0)
    for i in range(len(v)):
        acc += Float64(v[i]) * Float64(v[i])
    return sqrt(acc)


def main() raises:
    var ctx = DeviceContext()
    var n_fail = 0

    # ── ENCODER ──
    print("[encoder] loading oracle dump", ENC_ORACLE, "...")
    var enc_st = SafeTensors.open(String(ENC_ORACLE))
    var x_shape = _load_shape(enc_st, "x_ndhwc")
    var x_vals = _load_f32(enc_st, "x_ndhwc")
    var moments_ref = _load_f32(enc_st, "moments_ndhwc")
    print("  x_ndhwc shape", x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4])

    print("[encoder] loading real weights + running on the ORACLE's own input ...")
    var enc_cfg = minimax_h3_video_released_encoder_config()
    var encoder = MiniMaxH3VideoEncoderDevice.load(ENC_DIR, enc_cfg, ctx)
    var x = Tensor.from_host(x_vals, x_shape^, STDtype.F32, ctx)
    var moments = minimax_h3_video_encode_device(encoder, x, ctx)

    var enc_harness = ParityHarness(0.999)
    var enc_result = enc_harness.compare(moments, moments_ref, ctx)
    var moments_host = moments.to_host(ctx)
    var norm_mine_e = _l2(moments_host)
    var norm_ref_e = _l2(moments_ref)
    var ratio_e = Float64(-1.0)
    if norm_ref_e != 0.0:
        ratio_e = norm_mine_e / norm_ref_e
    print("  ", enc_result)
    print("   |mine| =", norm_mine_e, " |ref| =", norm_ref_e, " ratio =", ratio_e)
    if enc_result.passed:
        print("  ENCODER GATE PASS cos=", enc_result.cos)
    else:
        print("  ENCODER GATE FAIL cos=", enc_result.cos)
        n_fail += 1

    # ── DECODER ──
    print("")
    print("[decoder] loading oracle dump", DEC_ORACLE, "...")
    var dec_st = SafeTensors.open(String(DEC_ORACLE))
    var z_shape = _load_shape(dec_st, "z_ndhwc")
    var z_vals = _load_f32(dec_st, "z_ndhwc")
    var pixels_ref = _load_f32(dec_st, "pixels_ndhwc")
    print("  z_ndhwc shape", z_shape[0], z_shape[1], z_shape[2], z_shape[3], z_shape[4])

    print("[decoder] loading real weights (qkv split-at-load) + running on the ORACLE's own latent ...")
    var dec_cfg = minimax_h3_video_released_decoder_config()
    var decoder = minimax_h3_video_decoder_device_load(DEC_DIR, dec_cfg, ctx)
    var z = Tensor.from_host(z_vals, z_shape^, STDtype.F32, ctx)
    var pixels = minimax_h3_video_decode_device[DEC_S, DEC_HEADS, DEC_DH](decoder, z, ctx)

    var dec_harness = ParityHarness(0.999)
    var dec_result = dec_harness.compare(pixels, pixels_ref, ctx)
    var pixels_host = pixels.to_host(ctx)
    var norm_mine_d = _l2(pixels_host)
    var norm_ref_d = _l2(pixels_ref)
    var ratio_d = Float64(-1.0)
    if norm_ref_d != 0.0:
        ratio_d = norm_mine_d / norm_ref_d
    print("  ", dec_result)
    print("   |mine| =", norm_mine_d, " |ref| =", norm_ref_d, " ratio =", ratio_d)
    if dec_result.passed:
        print("  DECODER GATE PASS cos=", dec_result.cos)
    else:
        print("  DECODER GATE FAIL cos=", dec_result.cos)
        n_fail += 1

    print("")
    if n_fail != 0:
        raise Error(String("minimax_h3_video_vae_oracle_parity: ") + String(n_fail) + " gate(s) FAILED")
    print("ALL GATES PASS")
