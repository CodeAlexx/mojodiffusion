# models/nldiffusion/parity/backbone_stack_probe.mojo — CHUNK B gate.
#
# 1. Builds the Mojo-native YaRN rope tables and GATES cos/sin (+ inv_freq)
#    against `rope_oracle.safetensors` (expect near-exact, cos >= 0.99999).
# 2. Feeds oracle `layer0.in` [1,1141,4096] (post-embedding stack input) through
#    all 34 `nldiff_decoder_layer`s (weights STREAMED from the 6-shard model) +
#    the final `encoder.norm`, and GATES the result against `norm.out`
#    (cos >= 0.999). Prints cos, max_abs, and the magnitude ratio |mine|/|ref|.
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/nldiffusion/parity/backbone_stack_probe.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.nldiffusion.backbone import NLDiffConfig
from serenitymojo.models.nldiffusion.backbone_stack import (
    nldiff_build_yarn_rope,
    nldiff_yarn_inv_freq_tensor,
    nldiff_yarn_perhead,
    nldiff_backbone,
)

comptime MODEL_DIR = "/mnt/disk1/models/NL-Diffusion-Image"
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"
comptime SEQ = 1141


def _load_oracle_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _l2(xs: List[Float32]) -> Float64:
    var acc: Float64 = 0.0
    for i in range(len(xs)):
        var v = Float64(xs[i])
        acc += v * v
    return sqrt(acc)


def main() raises:
    var ctx = DeviceContext()
    var cfg = NLDiffConfig.default()

    var rope = ShardedSafeTensors.open(
        String(PARITY_DIR) + "/rope_oracle.safetensors"
    )
    var oracle = ShardedSafeTensors.open(
        String(PARITY_DIR) + "/oracle1.safetensors"
    )

    # ── (1) YaRN rope-table gate ─────────────────────────────────────────────
    print("[chunk B] building Mojo-native YaRN rope tables (seq=", SEQ, ")")
    var cs = nldiff_build_yarn_rope[SEQ](ctx)  # (cos_full, sin_full) [1,S,128]

    var rope_cos_ref = _load_oracle_f32(rope, "rope.cos", ctx).to_host(ctx)
    var rope_sin_ref = _load_oracle_f32(rope, "rope.sin", ctx).to_host(ctx)
    var inv_ref = _load_oracle_f32(rope, "rope.inv_freq", ctx).to_host(ctx)

    var rope_harness = ParityHarness(0.99999)
    var cos_res = rope_harness.compare(cs[0], rope_cos_ref, ctx)
    var sin_res = rope_harness.compare(cs[1], rope_sin_ref, ctx)
    var inv_t = nldiff_yarn_inv_freq_tensor(ctx)
    var inv_res = rope_harness.compare(inv_t, inv_ref, ctx)

    print("[chunk B] rope.cos     :", cos_res)
    print("[chunk B] rope.sin     :", sin_res)
    print("[chunk B] rope.inv_freq:", inv_res)
    print(
        "[chunk B] rope cos =",
        cos_res.cos,
        " sin cos =",
        sin_res.cos,
        " inv_freq cos =",
        inv_res.cos,
    )
    if cos_res.passed and sin_res.passed and inv_res.passed:
        print("[chunk B] ROPE GATE PASS (cos >= 0.99999)")
    else:
        print("[chunk B] ROPE GATE FAIL")

    # ── (2) full 34-layer backbone + final norm ──────────────────────────────
    print("[chunk B] loading oracle layer0.in (post-embedding stack input)")
    var x_f32 = _load_oracle_f32(oracle, "layer0.in", ctx)
    var x0 = cast_tensor(x_f32, STDtype.BF16, ctx)  # mirror bf16 reference compute

    print("[chunk B] building half-split per-head YaRN tables")
    var qh = nldiff_yarn_perhead[SEQ](cfg.num_heads, ctx)  # 32 heads
    var kh = nldiff_yarn_perhead[SEQ](cfg.num_kv_heads, ctx)  # 8 kv-heads

    print("[chunk B] streaming 34 layers + final norm from", MODEL_DIR)
    var model = ShardedSafeTensors.open(MODEL_DIR)
    var out = nldiff_backbone[SEQ](
        x0, model, cfg, qh[0], qh[1], kh[0], kh[1], ctx
    )

    # ── parity vs norm.out ───────────────────────────────────────────────────
    var ref_host = _load_oracle_f32(oracle, "norm.out", ctx).to_host(ctx)
    var out_host = out.to_host(ctx)

    var harness = ParityHarness(0.999)
    var res = harness.compare_host(out_host, ref_host)
    var mag_mine = _l2(out_host)
    var mag_ref = _l2(ref_host)
    var ratio = mag_mine / mag_ref if mag_ref != 0.0 else 0.0

    print("[chunk B] norm.out parity:", res)
    print(
        "[chunk B] cos =",
        res.cos,
        " max_abs =",
        res.max_abs,
        " |mine|/|ref| =",
        ratio,
        " (|mine|=",
        mag_mine,
        " |ref|=",
        mag_ref,
        ")",
    )
    if res.passed:
        print("[chunk B] BACKBONE GATE PASS (cos >= 0.999)")
    else:
        print("[chunk B] BACKBONE GATE FAIL (cos < 0.999)")
