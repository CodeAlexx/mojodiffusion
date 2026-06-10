# NAVA hi-res probe — gates the 832×480 (VID=1950, SEQ=1984) DiT forward.
#
# Fixture: nava_fx_hires.safetensors (all F32)
#   in_lat_vid [7800,48], in_lat_aud [34,128], in_text [122,4096], in_t [1]
#   blk0_x [1,1984,3072], blk0_e_vid [1,1950,6,3072],
#   blk0_e_audio [1,34,6,3072], blk0_context [1,512,3072], blk0_out [1,1984,3072]
#   vel_vid [7800,48], vel_aud [34,128]
#
# Gates (both must pass >= 0.999):
#   1. full forward: vel_vid vs fixture vel_vid
#   2. full forward: vel_aud vs fixture vel_aud
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.models.dit.nava_dit import NavaDiTHires

comptime NAVA = "/home/alex/.serenity/models/checkpoints/NAVA/NAVA_fp8.safetensors"
comptime FX   = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_hires.safetensors"


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA hi-res DiT probe (832x480, VID=1950, SEQ=1984) ===")

    # ── Load fixtures ─────────────────────────────────────────────────────────
    var fx = ShardedSafeTensors.open(FX)

    # Inputs (F32 → cast to BF16 except in_t which stays F32)
    var in_lat_vid_f32 = Tensor.from_view(fx.tensor_view("in_lat_vid"), ctx)  # [7800,48] F32
    var in_lat_aud_f32 = Tensor.from_view(fx.tensor_view("in_lat_aud"), ctx)  # [34,128]  F32
    var in_text_f32    = Tensor.from_view(fx.tensor_view("in_text"),    ctx)  # [122,4096] F32
    var in_t           = Tensor.from_view(fx.tensor_view("in_t"),       ctx)  # [1]       F32

    var in_lat_vid = torch_f32_to_bf16_rne(in_lat_vid_f32, ctx)
    var in_lat_aud = torch_f32_to_bf16_rne(in_lat_aud_f32, ctx)
    var in_text    = torch_f32_to_bf16_rne(in_text_f32,    ctx)

    # Reference targets (F32 on host)
    var vel_vid_ref = Tensor.from_view(fx.tensor_view("vel_vid"), ctx).to_host(ctx)
    var vel_aud_ref = Tensor.from_view(fx.tensor_view("vel_aud"), ctx).to_host(ctx)

    print("Inputs: in_lat_vid", in_lat_vid.shape(),
          " in_lat_aud", in_lat_aud.shape(),
          " in_text", in_text.shape())

    # ── Load resident hi-res model ────────────────────────────────────────────
    var st = ShardedSafeTensors.open(NAVA)
    print("Loading NavaDiTHires (all 30 blocks dequant → BF16 resident) ...")
    var dit = NavaDiTHires.load(st, ctx)
    print("NavaDiTHires loaded.")

    # ── Forward pass (masking_modality=False, joint attention) ───────────────
    print("Forward pass (joint, masking_modality=False) ...")
    var out = dit.forward(in_lat_vid, in_lat_aud, in_text, in_t, ctx, masking_modality=False)

    var vel_vid_f32 = cast_tensor(out.vel_vid, STDtype.F32, ctx)
    var vel_aud_f32 = cast_tensor(out.vel_aud, STDtype.F32, ctx)

    var harness = ParityHarness(0.999)

    print("--- vel_vid (hires forward vs torch) ---")
    var res_vid = harness.compare(vel_vid_f32, vel_vid_ref, ctx)
    print("vel_vid cos:", res_vid.cos, " passed:", res_vid.passed)

    print("--- vel_aud (hires forward vs torch) ---")
    var res_aud = harness.compare(vel_aud_f32, vel_aud_ref, ctx)
    print("vel_aud cos:", res_aud.cos, " passed:", res_aud.passed)

    # ── Summary ───────────────────────────────────────────────────────────────
    if res_vid.passed and res_aud.passed:
        print("=== GATE PASS: 832x480 DiT both outputs cos >= 0.999 ===")
        print("  vel_vid cos=", res_vid.cos)
        print("  vel_aud cos=", res_aud.cos)
    else:
        print("=== GATE FAIL ===")
        print("  vel_vid cos=", res_vid.cos, " passed=", res_vid.passed)
        print("  vel_aud cos=", res_aud.cos, " passed=", res_aud.passed)
