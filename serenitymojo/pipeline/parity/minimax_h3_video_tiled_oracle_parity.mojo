# serenitymojo/pipeline/parity/minimax_h3_video_tiled_oracle_parity.mojo
#
# TILED decode vs the vendor's OWN tiled_decode, real weights, production H/W.
#
# WHY THIS GATE EXISTS. The untiled decoder is oracle-gated near bit-exact
# (models/vae/parity/minimax_h3_video_vae_oracle_parity.mojo), but production
# runs the TILED path (untiled OOMs at >= ~10k ViT tokens), and the tiling
# layer's own probes are seam tests — they can prove the blend runs, but they
# compare the tiled path against ITSELF, so a defect common to every tile
# passes. Tiling is also an inherent approximation (per-tile attention vs
# global), so "tiled vs untiled interior diff" cannot separate approximation
# from bug. The only decisive oracle is the vendor's own tiled_decode
# (klvae.py:365) on a byte-identical latent, which this gate consumes.
#
# The first e2e video (2026-08-03) showed a residual fine grid texture after
# the channel-order fix. This gate decides whether that texture is OUR tiling
# layer diverging from the vendor, or is present in the vendor's own tiled
# output too (i.e. inherent to tiling/steps/resolution, not a port bug).
#
# Oracle: /tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/
#   scratchpad/vvae_tiled_oracle_gen.py (OneTrainer venv, GPU F32) —
#   z [1,24,2,30,52] NCDHW randn(seed 4321) -> tiled_decode -> [1,3,8,480,832],
#   both dumped NDHWC. Splits it reported: H starts [0,112,224] overlap 144 px,
#   W starts [0,192,384,576] overlap 64 px — every tile 256 px = 16 latent,
#   satisfying the Mojo layer's uniform-tile comptime requirement.
#
# Run (cuDNN shim on the link line for the F32 conv3d symbols):
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/pipeline/parity/minimax_h3_video_tiled_oracle_parity.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.parity import ParityHarness
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    minimax_h3_video_released_decoder_config,
    minimax_h3_video_decoder_device_load,
)
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    minimax_h3_video_tiled_decode,
    minimax_h3_video_released_tiling_config,
)

comptime ORACLE = "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/scratchpad/vvae_tiled_oracle.safetensors"
comptime DEC_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/video_vae/source"


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
    var info = st.tensor_info(name)
    return info.shape.copy()


def main() raises:
    var ctx = DeviceContext()
    print("[tiled] loading oracle dump ...")
    var st = SafeTensors.open(String(ORACLE))
    var z_shape = _load_shape(st, "z_ndhwc")
    var z_vals = _load_f32(st, "z_ndhwc")
    var ref_vals = _load_f32(st, "dec_ndhwc")
    var ref_shape = _load_shape(st, "dec_ndhwc")
    print(
        "  z", z_shape[0], z_shape[1], z_shape[2], z_shape[3], z_shape[4],
        " ref", ref_shape[0], ref_shape[1], ref_shape[2], ref_shape[3], ref_shape[4],
    )

    print("[tiled] loading real decoder weights (9.6 GiB F32) ...")
    var cfg = minimax_h3_video_released_decoder_config()
    var decoder = minimax_h3_video_decoder_device_load(String(DEC_DIR), cfg, ctx)

    var z = Tensor.from_host(z_vals, z_shape^, STDtype.F32, ctx)
    var tiling = minimax_h3_video_released_tiling_config()
    print("[tiled] running minimax_h3_video_tiled_decode (T=2, 30x52 latent, 3x4 tiles) ...")
    var pixels = minimax_h3_video_tiled_decode[16, 16, 32, 64, 5, 2](
        decoder, z, tiling, ctx
    )
    print("  pixels shape:", pixels.shape())

    var harness = ParityHarness(0.999)
    var result = harness.compare(pixels, ref_vals, ctx)
    print("  ", result)

    # magnitude ratio — cos is magnitude-blind and this repo has been burned
    var mine_host = pixels.to_host(ctx)
    var s_mine = Float64(0)
    var s_ref = Float64(0)
    for i in range(len(mine_host)):
        s_mine += Float64(mine_host[i]) * Float64(mine_host[i])
        s_ref += Float64(ref_vals[i]) * Float64(ref_vals[i])
    print("  |mine|", sqrt(s_mine), " |ref|", sqrt(s_ref), " ratio", sqrt(s_mine) / sqrt(s_ref))

    if not result.passed:
        raise Error("minimax_h3_video_tiled_oracle_parity FAILED")
    print("minimax_h3_video_tiled_oracle_parity PASS")
