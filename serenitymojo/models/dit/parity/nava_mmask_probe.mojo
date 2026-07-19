# NAVA masking_modality gate probe — Part 1 deliverable.
#
# Tests:
#   1. masking_modality=True  → vel_vid / vel_aud cos >= 0.999 vs nava_fx_mmask.safetensors
#   2. masking_modality=False → vel_vid / vel_aud cos >= 0.999 vs nava_fx_stage0.safetensors
#      (joint regression: must match existing chunk-7 result)
#
# Fixture keys:
#   nava_fx_stage0.safetensors : in_lat_vid[1280,48] in_lat_aud[34,128]
#                                in_text[42,4096] in_t[1] (all F32)
#                                vel_vid[1280,48] vel_aud[34,128] (joint reference)
#   nava_fx_mmask.safetensors  : vel_mmask_vid[1280,48] vel_mmask_aud[34,128] (non-joint ref)
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/dit/parity/nava_mmask_probe.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.nava_dit import NavaDiT

comptime NAVA  = "/home/alex/.serenity/models/checkpoints/NAVA/NAVA_fp8.safetensors"
comptime FX0   = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_stage0.safetensors"
comptime FXMM  = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_mmask.safetensors"


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA masking_modality gate probe ===")

    # ── Load input fixtures (shared by both tests) ────────────────────────────
    var fx0 = ShardedSafeTensors.open(FX0)

    var in_lat_vid_f32 = Tensor.from_view(fx0.tensor_view("in_lat_vid"), ctx)  # [1280,48] F32
    var in_lat_aud_f32 = Tensor.from_view(fx0.tensor_view("in_lat_aud"), ctx)  # [34,128]  F32
    var in_text_f32    = Tensor.from_view(fx0.tensor_view("in_text"),    ctx)  # [42,4096] F32
    var in_t           = Tensor.from_view(fx0.tensor_view("in_t"),       ctx)  # [1]       F32

    # Joint reference targets (for regression test)
    var vel_vid_joint_ref = Tensor.from_view(fx0.tensor_view("vel_vid"), ctx).to_host(ctx)
    var vel_aud_joint_ref = Tensor.from_view(fx0.tensor_view("vel_aud"), ctx).to_host(ctx)

    # Non-joint (masking_modality=True) reference targets
    var fxmm = ShardedSafeTensors.open(FXMM)
    var vel_mmask_vid_ref = Tensor.from_view(fxmm.tensor_view("vel_mmask_vid"), ctx).to_host(ctx)
    var vel_mmask_aud_ref = Tensor.from_view(fxmm.tensor_view("vel_mmask_aud"), ctx).to_host(ctx)

    # BF16 inputs (clone for each forward since Tensor is move-only)
    var in_lat_vid_bf16 = cast_tensor(in_lat_vid_f32, STDtype.BF16, ctx)
    var in_lat_aud_bf16 = cast_tensor(in_lat_aud_f32, STDtype.BF16, ctx)
    var in_text_bf16    = cast_tensor(in_text_f32,    STDtype.BF16, ctx)

    # ── Load resident model ───────────────────────────────────────────────────
    var st = ShardedSafeTensors.open(NAVA)
    print("Loading NavaDiT (all 30 blocks dequant → BF16 resident) ...")
    var dit = NavaDiT.load(st, ctx)
    print("NavaDiT loaded.")

    var harness = ParityHarness(0.999)

    # ── Test 1: masking_modality=True (non-joint) ─────────────────────────────
    print("\n--- Test 1: masking_modality=True (non-joint) ---")
    # Need fresh BF16 copies for this forward (move semantics)
    var lv1 = cast_tensor(in_lat_vid_f32, STDtype.BF16, ctx)
    var la1 = cast_tensor(in_lat_aud_f32, STDtype.BF16, ctx)
    var tx1 = cast_tensor(in_text_f32,    STDtype.BF16, ctx)
    var t1_f32 = Tensor.from_view(fx0.tensor_view("in_t"), ctx)

    var mmask_out = dit.forward(lv1, la1, tx1, t1_f32, ctx, masking_modality=True)

    var mmask_vid_f32 = cast_tensor(mmask_out.vel_vid, STDtype.F32, ctx)
    var mmask_aud_f32 = cast_tensor(mmask_out.vel_aud, STDtype.F32, ctx)

    print("vel_mmask_vid vs torch:")
    var res_mmask_vid = harness.compare(mmask_vid_f32, vel_mmask_vid_ref, ctx)
    print(res_mmask_vid)

    print("vel_mmask_aud vs torch:")
    var res_mmask_aud = harness.compare(mmask_aud_f32, vel_mmask_aud_ref, ctx)
    print(res_mmask_aud)

    # ── Test 2: masking_modality=False (joint regression) ────────────────────
    print("\n--- Test 2: masking_modality=False (joint regression) ---")
    var lv2 = cast_tensor(in_lat_vid_f32, STDtype.BF16, ctx)
    var la2 = cast_tensor(in_lat_aud_f32, STDtype.BF16, ctx)
    var tx2 = cast_tensor(in_text_f32,    STDtype.BF16, ctx)
    var t2_f32 = Tensor.from_view(fx0.tensor_view("in_t"), ctx)

    var joint_out = dit.forward(lv2, la2, tx2, t2_f32, ctx, masking_modality=False)

    var joint_vid_f32 = cast_tensor(joint_out.vel_vid, STDtype.F32, ctx)
    var joint_aud_f32 = cast_tensor(joint_out.vel_aud, STDtype.F32, ctx)

    print("vel_vid (joint) vs torch:")
    var res_joint_vid = harness.compare(joint_vid_f32, vel_vid_joint_ref, ctx)
    print(res_joint_vid)

    print("vel_aud (joint) vs torch:")
    var res_joint_aud = harness.compare(joint_aud_f32, vel_aud_joint_ref, ctx)
    print(res_joint_aud)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== GATE SUMMARY ===")
    var mmask_pass = res_mmask_vid.passed and res_mmask_aud.passed
    var joint_pass = res_joint_vid.passed and res_joint_aud.passed

    if mmask_pass:
        print("PASS  masking_modality=True:  vel_vid cos=", res_mmask_vid.cos,
              " vel_aud cos=", res_mmask_aud.cos)
    else:
        print("FAIL  masking_modality=True:  vel_vid cos=", res_mmask_vid.cos,
              "(passed=", res_mmask_vid.passed, ")",
              " vel_aud cos=", res_mmask_aud.cos,
              "(passed=", res_mmask_aud.passed, ")")

    if joint_pass:
        print("PASS  masking_modality=False: vel_vid cos=", res_joint_vid.cos,
              " vel_aud cos=", res_joint_aud.cos)
    else:
        print("FAIL  masking_modality=False: vel_vid cos=", res_joint_vid.cos,
              "(passed=", res_joint_vid.passed, ")",
              " vel_aud cos=", res_joint_aud.cos,
              "(passed=", res_joint_aud.passed, ")")

    if mmask_pass and joint_pass:
        print("=== PART 1 GATE: ALL PASS ===")
    else:
        print("=== PART 1 GATE: FAIL — DO NOT PROCEED TO PART 2 ===")
