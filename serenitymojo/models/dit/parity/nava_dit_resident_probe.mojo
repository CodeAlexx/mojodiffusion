# NAVA resident probe — Deliverable 3.
#
# Regression gate: NavaDiT.load + dit.forward must reproduce the chunk-7 result.
# Gate: cos >= 0.999 for BOTH vel_vid and vel_aud.
# Loop-readiness: call dit.forward TWICE; confirm 2nd forward gives identical cos.
#
# Fixture keys (nava_fx_stage0.safetensors, all F32):
#   in_lat_vid [1280,48], in_lat_aud [34,128], in_text [42,4096], in_t [1]
#   vel_vid    [1280,48], vel_aud [34,128]
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.nava_dit import NavaDiT

comptime NAVA = "/home/alex/.serenity/models/checkpoints/NAVA/NAVA_fp8.safetensors"
comptime FX   = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_stage0.safetensors"


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA resident DiT probe ===")

    # ── Load fixtures ─────────────────────────────────────────────────────────
    var fx = ShardedSafeTensors.open(FX)

    # Inputs (F32 → cast to BF16 except in_t which stays F32)
    var in_lat_vid_f32 = Tensor.from_view(fx.tensor_view("in_lat_vid"), ctx)  # [1280,48] F32
    var in_lat_aud_f32 = Tensor.from_view(fx.tensor_view("in_lat_aud"), ctx)  # [34,128]  F32
    var in_text_f32    = Tensor.from_view(fx.tensor_view("in_text"),    ctx)  # [42,4096] F32
    var in_t           = Tensor.from_view(fx.tensor_view("in_t"),       ctx)  # [1]       F32

    var in_lat_vid = cast_tensor(in_lat_vid_f32, STDtype.BF16, ctx)
    var in_lat_aud = cast_tensor(in_lat_aud_f32, STDtype.BF16, ctx)
    var in_text    = cast_tensor(in_text_f32,    STDtype.BF16, ctx)

    # Reference targets (F32 on host)
    var vel_vid_ref = Tensor.from_view(fx.tensor_view("vel_vid"), ctx).to_host(ctx)
    var vel_aud_ref = Tensor.from_view(fx.tensor_view("vel_aud"), ctx).to_host(ctx)

    # ── Load resident model (all weights + RoPE once) ─────────────────────────
    var st = ShardedSafeTensors.open(NAVA)
    print("Loading NavaDiT (all 30 blocks dequant → BF16 resident) ...")
    var dit = NavaDiT.load(st, ctx)
    print("NavaDiT loaded.")

    # ── First forward pass ────────────────────────────────────────────────────
    print("Forward pass 1 ...")
    var out1 = dit.forward(in_lat_vid, in_lat_aud, in_text, in_t, ctx)

    var vel_vid1_f32 = cast_tensor(out1.vel_vid, STDtype.F32, ctx)
    var vel_aud1_f32 = cast_tensor(out1.vel_aud, STDtype.F32, ctx)

    var harness = ParityHarness(0.999)

    print("--- vel_vid (pass 1 vs torch) ---")
    var res_vid1 = harness.compare(vel_vid1_f32, vel_vid_ref, ctx)
    print("vel_vid pass1 vs torch:", res_vid1)

    print("--- vel_aud (pass 1 vs torch) ---")
    var res_aud1 = harness.compare(vel_aud1_f32, vel_aud_ref, ctx)
    print("vel_aud pass1 vs torch:", res_aud1)

    # ── Second forward pass (loop-readiness: weights + rope must be intact) ───
    print("Forward pass 2 (loop-readiness check) ...")
    var out2 = dit.forward(in_lat_vid, in_lat_aud, in_text, in_t, ctx)

    var vel_vid2_f32 = cast_tensor(out2.vel_vid, STDtype.F32, ctx)
    var vel_aud2_f32 = cast_tensor(out2.vel_aud, STDtype.F32, ctx)

    print("--- vel_vid (pass 2 vs torch) ---")
    var res_vid2 = harness.compare(vel_vid2_f32, vel_vid_ref, ctx)
    print("vel_vid pass2 vs torch:", res_vid2)

    print("--- vel_aud (pass 2 vs torch) ---")
    var res_aud2 = harness.compare(vel_aud2_f32, vel_aud_ref, ctx)
    print("vel_aud pass2 vs torch:", res_aud2)

    # ── Summary ───────────────────────────────────────────────────────────────
    var gate1 = res_vid1.passed and res_aud1.passed
    var gate2 = res_vid2.passed and res_aud2.passed
    if gate1 and gate2:
        print("=== GATE PASS: both passes, both outputs cos >= 0.999 ===")
        print("  PASS1  vel_vid cos=", res_vid1.cos, "  vel_aud cos=", res_aud1.cos)
        print("  PASS2  vel_vid cos=", res_vid2.cos, "  vel_aud cos=", res_aud2.cos)
    else:
        print("=== GATE FAIL ===")
        print("  P1 vid:", res_vid1.cos, "passed=", res_vid1.passed)
        print("  P1 aud:", res_aud1.cos, "passed=", res_aud1.passed)
        print("  P2 vid:", res_vid2.cos, "passed=", res_vid2.passed)
        print("  P2 aud:", res_aud2.cos, "passed=", res_aud2.passed)
