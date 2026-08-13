# autograd_v2/tests/ltx2_weight_dtype_probe.mojo — L5 VERIFY-gap (a): MEASURE
# the resident weight dtype on the LTX-2 VIDEO training path (the exact
# get_block used by ltx2_video_stack's backward). Decides whether the block
# slab twins need linear_fp8_slab (F8_E4M3 resident) or plain linear_slab
# (BF16 dequant-on-use). One-print, block 0, production default arm
# (run_f32=False, same as ltx2_block_parity.mojo).
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/ltx2_weight_dtype_probe.mojo -o /tmp/ltx2_weight_dtype_probe
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/ltx2_weight_dtype_probe

from max.gpu.host import DeviceContext
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_video_stack import LTX2VideoBlockSource

comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"


def _probe(w_key: String, w: LTX2AVBlockWeights, mut any_fp8: Bool) raises:
    var dt = w._w(w_key).dtype()
    var is_fp8 = dt == STDtype.F8_E4M3
    if is_fp8:
        any_fp8 = True
    print("  ", w_key, "dtype=", dt.name(), "  fp8=", is_fp8)


def main() raises:
    print("=== LTX-2 VIDEO training-path weight dtype probe (block 0) ===")
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    # bf16 stack (run_f32=False) = the production default arm, exactly as
    # ltx2_video_stack backward and ltx2_block_parity build it.
    var src = LTX2VideoBlockSource.open(CKPT, cfg, False)
    var w = src.get_block(0, ctx)
    var any_fp8 = False
    # The matmul weights the block twins route through _linear_b / _av_attention:
    _probe("attn1.to_q.weight", w, any_fp8)
    _probe("attn2.to_q.weight", w, any_fp8)
    _probe("attn2.to_k.weight", w, any_fp8)
    _probe("attn2.to_v.weight", w, any_fp8)
    _probe("attn2.to_out.0.weight", w, any_fp8)
    _probe("ff.net.0.proj.weight", w, any_fp8)
    _probe("ff.net.2.weight", w, any_fp8)
    # Presence of the pre-norm affine weights decides _rms_norm_opt's branch:
    # present ⇒ rms_norm (affine); absent ⇒ _rms_norm_no_affine (ones weight).
    print("  norm1.weight present=", w._has("norm1.weight"))
    print("  norm2.weight present=", w._has("norm2.weight"))
    print("  norm3.weight present=", w._has("norm3.weight"))
    print("  attn1.norm_q.weight present=", w._has("attn1.norm_q.weight"))
    print("  prompt_scale_shift_table present=", w._has("prompt_scale_shift_table"))
    print("  scales_count=", len(w.scales), " (nonzero ⇒ resident-fp8 mode)")
    if any_fp8 or len(w.scales) != 0:
        print("VERDICT: F8_E4M3 resident weights present ⇒ block twins NEED linear_fp8_slab")
    else:
        print("VERDICT: BF16 dequant-on-use, NO resident-fp8 ⇒ linear_slab (bf16) suffices; linear_fp8_slab NOT needed")
