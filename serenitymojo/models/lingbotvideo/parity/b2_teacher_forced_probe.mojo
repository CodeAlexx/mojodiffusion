# models/lingbotvideo/parity/b2_teacher_forced_probe.mojo — DECISIVE B2 diagnostic.
#
# Isolates PER-BLOCK math from the input-cascade. For each block i in 0..47:
#   input  = oracle's INPUT to block i  (joint_preblock for i=0, else block_{i-1})
#   run    = lingbot_video_block[S] i with the REAL streamed weights + oracle
#            temb6 / freqs_cos / freqs_sin (the block's own AdaLN + RoPE)
#   compare output vs oracle block_{i}   -> per-block cos.
#
# If EVERY block is cos ~>= 0.9995 on its CORRECT (teacher-forced) input, the
# per-block math is PERFECT and the free-running ~0.99 velocity is pure router-
# flip cascade over 48 layers (inherent). If a block systematically underperforms
# (cos << 0.999 on a correct input, well below A3's single-block 0.99998) that is
# a REAL per-block bug to localize.
#
# Run (JIT; streams each block ~1.2GB from NVMe):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/b2_teacher_forced_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.env import env_or
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig,
    lingbot_video_block,
    LingBotBlockW,
    _load_block_weights,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime MODEL_DIR_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer"
# Override with LINGBOT_CKPT (e.g. .../transformer_fp8) for the fp8 dequant path.
comptime S = 72
comptime DEPTH = 48
comptime H = 2048


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view(st.tensor_view(name), ctx)


def _load_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    # oracle stored block outputs as float(bf16 values); cast back to bf16 gives
    # the exact bf16 activations the reference block consumed.
    return cast_tensor(_load_f32(st, name, ctx), STDtype.BF16, ctx)


def _cos(mine: List[Float32], reference: List[Float32]) -> Float64:
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(mine)):
        var a = Float64(mine[i])
        var b = Float64(reference[i])
        dot += a * b
        na += a * a
        nb += b * b
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()

    print("[TF] loading oracle_b2.safetensors  S =", S, " depth =", DEPTH)
    var oracle = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_b2.safetensors")

    # oracle temb6 [S,12288] f32 ; freqs_cos/sin [S,64] f32 (feed the block's own
    # AdaLN modulation + RoPE, so we test ONLY the per-block math on correct input).
    # The block now takes the single modulation row [1,12288] (device AdaLN); all
    # oracle temb6 rows are identical, so take row 0.
    var temb6 = _load_f32(oracle, "temb6", ctx)          # [S,12288]
    var temb6_host = temb6.to_host(ctx)
    var trow_host = List[Float32]()
    trow_host.resize(12288, Float32(0.0))
    for j in range(12288):
        trow_host[j] = temb6_host[j]
    var temb_row = Tensor.from_host(trow_host, [1, 12288], STDtype.F32, ctx)  # [1,12288]
    var cos = _load_f32(oracle, "freqs_cos", ctx)        # [S,64]
    var sin = _load_f32(oracle, "freqs_sin", ctx)        # [S,64]

    var model_dir = env_or(String("LINGBOT_CKPT"), String(MODEL_DIR_DEFAULT))
    print("[TF] opening transformer shards (streaming):", model_dir)
    var model = ShardedSafeTensors.open(model_dir)

    var worst_cos: Float64 = 1.0
    var worst_i = 0
    var all_pass = True
    print("[TF] ===== teacher-forced per-block cos (input = oracle block_{i-1}) =====")
    for i in range(DEPTH):
        var in_name = String("joint_preblock") if i == 0 else String("block_") + String(i - 1)
        var x_in = _load_bf16(oracle, in_name, ctx)   # [1,S,2048] bf16

        var w = _load_block_weights(model, i, ctx)
        var out = lingbot_video_block[S](
            x_in, temb_row, w.scale_shift_table[],
            w.w_q[], w.w_k[], w.w_v[], w.w_o[], w.b_o[],
            w.norm_q_w[], w.norm_k_w[], cos, sin,
            w.norm1_w[], w.norm2_w[], w.norm_post_attn_w[], w.norm_post_ffn_w[],
            w.router_weight[], w.e_score_correction_bias[],
            w.w1[], w.w3[], w.w2[], w.shared_gate[], w.shared_up[], w.shared_down[],
            cfg, ctx,
        )  # [1,S,2048] bf16

        var mine = out.to_host(ctx)
        var oref = _load_f32(oracle, String("block_") + String(i), ctx).to_host(ctx)
        var c = _cos(mine, oref)
        var tag = "PASS" if c >= 0.9995 else "under"
        if c < worst_cos:
            worst_cos = c
            worst_i = i
        if c < 0.9995:
            all_pass = False
        print("[TF] block", i, " (in=", in_name, ")  cos =", c, "  ", tag)

        _ = w^
        ctx.synchronize()

    print("[TF] ===================================================================")
    print("[TF] worst block =", worst_i, "  worst cos =", worst_cos)
    if all_pass:
        print("[TF] VERDICT: per-block math PERFECT (every block cos >= 0.9995 on")
        print("[TF]          teacher-forced input) -> free-running gap is router cascade.")
    else:
        print("[TF] VERDICT: at least one block underperforms on CORRECT input ->")
        print("[TF]          REAL per-block bug at block", worst_i, "— localize + fix.")
