# NAVA chunk 5: joint AV double-stream block (block 0) parity probe.
# Loads fixtures from nava_fx_chunk5_block0.safetensors (all F32 → cast to BF16),
# loads and dequantises weights from NAVA_fp8.safetensors (prefix backbone.double_blocks.0.),
# runs nava_double_block, then prints ParityHarness comparison vs double0_out.
# Gate: cos >= 0.999.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.nava_block import (
    load_nava_double_block, nava_double_block, build_nava_rope_tables,
)

comptime NAVA = "/home/alex/.serenity/models/checkpoints/NAVA/NAVA_fp8.safetensors"
comptime FX   = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_chunk5_block0.safetensors"


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA chunk5 double-block-0 parity ===")

    # ── Load fixtures ─────────────────────────────────────────────────────────
    var fx = ShardedSafeTensors.open(FX)

    # Inputs are F32; cast to BF16 for the block forward.
    var x_f32       = Tensor.from_view(fx.tensor_view("x"),       ctx)  # [1,354,3072]
    var e_vid_f32   = Tensor.from_view(fx.tensor_view("e_vid"),   ctx)  # [1,320,6,3072]
    var e_audio_f32 = Tensor.from_view(fx.tensor_view("e_audio"), ctx)  # [1,34,6,3072]
    var ctx_f32     = Tensor.from_view(fx.tensor_view("context"), ctx)  # [1,512,3072]

    var x_bf16       = cast_tensor(x_f32,       STDtype.BF16, ctx)
    var e_vid_bf16   = cast_tensor(e_vid_f32,   STDtype.BF16, ctx)
    var e_audio_bf16 = cast_tensor(e_audio_f32, STDtype.BF16, ctx)
    var ctx_bf16     = cast_tensor(ctx_f32,     STDtype.BF16, ctx)

    # Reference output stays F32 on host for comparison.
    var ref_host     = Tensor.from_view(fx.tensor_view("double0_out"), ctx).to_host(ctx)

    # ── Load weights ──────────────────────────────────────────────────────────
    var st = ShardedSafeTensors.open(NAVA)
    var wb = load_nava_double_block(st, "backbone.double_blocks.0.", ctx)

    # ── Build rope tables (one-time, same math as the old inline build) ───────
    var rope = build_nava_rope_tables(ctx)

    # ── Forward ───────────────────────────────────────────────────────────────
    var out_bf16 = nava_double_block(x_bf16, e_vid_bf16, e_audio_bf16, ctx_bf16, wb, rope, ctx)

    # Cast output to F32 for parity comparison (ref is F32).
    var out_f32 = cast_tensor(out_bf16, STDtype.F32, ctx)

    # ── Parity ────────────────────────────────────────────────────────────────
    var harness = ParityHarness(0.999)
    var result = harness.compare(out_f32, ref_host, ctx)
    print("NAVA double-block-0 vs torch:", result)
    if result.passed:
        print("GATE PASS cos=", result.cos)
    else:
        print("GATE FAIL cos=", result.cos)
