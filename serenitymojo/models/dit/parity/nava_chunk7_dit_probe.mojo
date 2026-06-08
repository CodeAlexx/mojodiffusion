# NAVA chunk 7: full 30-layer WanAV DiT forward parity probe.
#
# Loads fp8 checkpoint + stage-0 fixtures, runs nava_dit_forward,
# then gates vel_vid and vel_aud against references.
#
# Fixture keys (nava_fx_stage0.safetensors, all F32):
#   in_lat_vid [1280,48], in_lat_aud [34,128], in_text [42,4096], in_t [1]
#   vel_vid    [1280,48], vel_aud [34,128]
#
# Gate: cos >= 0.999 for BOTH vel_vid and vel_aud.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.nava_dit import nava_dit_forward

comptime NAVA = "/home/alex/.serenity/models/checkpoints/NAVA/NAVA_fp8.safetensors"
comptime FX   = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_stage0.safetensors"


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA chunk7 full-DiT forward parity ===")

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
    var vel_vid_ref = Tensor.from_view(fx.tensor_view("vel_vid"), ctx).to_host(ctx)  # [1280,48]
    var vel_aud_ref = Tensor.from_view(fx.tensor_view("vel_aud"), ctx).to_host(ctx)  # [34,128]

    # ── Open checkpoint ───────────────────────────────────────────────────────
    var st = ShardedSafeTensors.open(NAVA)

    # ── Full forward ──────────────────────────────────────────────────────────
    print("Running 30-layer NAVA forward (10 double + 20 single blocks) ...")
    var out = nava_dit_forward(in_lat_vid, in_lat_aud, in_text, in_t, st, ctx)

    # Cast outputs to F32 for parity comparison (refs are F32)
    var vel_vid_f32 = cast_tensor(out.vel_vid, STDtype.F32, ctx)
    var vel_aud_f32 = cast_tensor(out.vel_aud, STDtype.F32, ctx)

    # ── Parity gates ──────────────────────────────────────────────────────────
    var harness = ParityHarness(0.999)

    print("--- vel_vid ---")
    var res_vid = harness.compare(vel_vid_f32, vel_vid_ref, ctx)
    print("vel_vid vs torch:", res_vid)

    print("--- vel_aud ---")
    var res_aud = harness.compare(vel_aud_f32, vel_aud_ref, ctx)
    print("vel_aud vs torch:", res_aud)

    # Summary
    if res_vid.passed and res_aud.passed:
        print("=== GATE PASS: both vel_vid and vel_aud cos >= 0.999 ===")
        print("  vel_vid  cos=", res_vid.cos,  "max_abs=", res_vid.max_abs)
        print("  vel_aud  cos=", res_aud.cos,  "max_abs=", res_aud.max_abs)
    else:
        print("=== GATE FAIL ===")
        print("  vel_vid  cos=", res_vid.cos,  "passed=", res_vid.passed)
        print("  vel_aud  cos=", res_aud.cos,  "passed=", res_aud.passed)
