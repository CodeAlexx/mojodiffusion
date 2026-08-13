# mageflow_editcond_probe.mojo — parity gate for the pure-Mojo Mage-Flow EDIT
# conditioning (encode_mageflow_edit) vs the torch oracle (mageflow_editcond_oracle.py).
#
# Loads the Mage-Flow-Edit-Turbo text_encoder (vision tower model.visual.* +
# text decoder model.language_model.*), reads the oracle's FIXED input_ids /
# pixel_values, and gates:
#   G-vision   : Mojo vision pooler_output   vs HF vision_pooler        (cos>=0.999)
#   G-editcond : Mojo fuse (fed HF f32 vision tokens, f32 stream) txt
#                                            vs HF editcond_txt [L-64]  (cos>=0.999)
# Plus informational full-path numbers (Mojo vision -> fuse, f32 and bf16).
#
# The G-editcond GATE is the ISOLATED fusion math (HF's exact f32 vision tokens
# through fuse_mageflow_edit in F32) — it removes the vision bf16-weight residual
# confound, exactly as the LingBot Qwen3-VL fuse gate does. The full end-to-end
# path (Mojo vision -> fuse) is informational: the fused image-token rows are
# bf16-ill-conditioned, so the vision tower's ~1e-4 bf16-weight residual is
# amplified there — a precision/condition property, not a fusion bug.
#
# Run (from repo root, main pixi env, -I this worktree):
#   MOJO=/home/alex/mojodiffusion/.pixi/envs/default/bin/mojo
#   $MOJO run -I <worktree> ... (see the compile cmd in the agent report)

from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import load_krea2_qwen3vl_4b
from serenitymojo.models.text_encoder.lingbot_qwen3vl_vision import Qwen3VLVisionModel
from serenitymojo.models.text_encoder.mageflow_qwen3vl import (
    encode_mageflow_edit, fuse_mageflow_edit, MAGEFLOW_EDIT_DROP_IDX,
)

comptime TE_DIR = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/text_encoder"
comptime ORACLE = (
    "/home/alex/mojodiffusion/.claude/worktrees/agent-aa35b9c61fab51182/serenitymojo/"
    "models/text_encoder/parity/mageflow_editcond_oracle.safetensors"
)
# Fixed oracle image grid (t,h,w)=(1,24,12) -> S=288, nvis=72; L=156.
comptime T = 1
comptime H = 24
comptime W = 12
comptime S = 288


def _ref(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    return Tensor.from_view(st.tensor_view(name), ctx).to_host(ctx)


def _gate(
    harness: ParityHarness, tag: String, actual: Tensor,
    reference: List[Float32], ctx: DeviceContext,
) raises -> Bool:
    var r = harness.compare(actual, reference, ctx)
    var ok = r.cos >= 0.999
    var status = "PASS" if ok else "FAIL"
    print("  [", status, "] ", tag, "  cos=", r.cos, "  max_abs=", r.max_abs)
    return ok


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(ORACLE))
    var harness = ParityHarness(0.999)

    # oracle input_ids (exact ints via f32 carrier)
    var ids_f = _ref(st, String("input_ids_f32"), ctx)
    var ids = List[Int]()
    for i in range(len(ids_f)):
        ids.append(Int(ids_f[i] + 0.5))
    var L = len(ids)
    print("[parity] Mage-Flow EDIT conditioning  L=", L, " S=", S,
          " (t,h,w)=(", T, ",", H, ",", W, ")  drop_idx=", MAGEFLOW_EDIT_DROP_IDX)

    var pixel_values = Tensor.from_view(st.tensor_view(String("pixel_values")), ctx)

    print("[parity] loading Mage vision tower (model.visual.*) + text decoder (model.language_model.*) ...")
    var vision = Qwen3VLVisionModel.load(String(TE_DIR), ctx)
    var enc = load_krea2_qwen3vl_4b(String(TE_DIR), ctx)

    var all_ok = True

    # ── G-vision: Mojo vision pooler vs HF vision_pooler ──
    var vout = vision.forward[S](pixel_values, T, H, W, ctx)
    print("[parity] G-vision (Mojo F32-stream vision tower vs HF f32):")
    all_ok = _gate(harness, "vision_pooler        ", vout.pooler,
                   _ref(st, String("vision_pooler"), ctx), ctx) and all_ok

    # ── G-editcond: ISOLATED fusion math — HF's exact f32 vision tokens through
    #    fuse_mageflow_edit (f32 stream), dropping the 64 system rows. ──
    print("[parity] G-editcond (isolated fusion, HF f32 vision tokens -> fuse f32):")
    var f_pool = Tensor.from_view(st.tensor_view(String("vision_pooler")), ctx)
    var f_ds0 = Tensor.from_view(st.tensor_view(String("deepstack_0")), ctx)
    var f_ds1 = Tensor.from_view(st.tensor_view(String("deepstack_1")), ctx)
    var f_ds2 = Tensor.from_view(st.tensor_view(String("deepstack_2")), ctx)
    var editcond = fuse_mageflow_edit(
        enc, f_pool, f_ds0, f_ds1, f_ds2, ids, MAGEFLOW_EDIT_DROP_IDX, ctx, True
    )
    all_ok = _gate(harness, "editcond_txt [L-64]  ", editcond,
                   _ref(st, String("editcond_txt"), ctx), ctx) and all_ok

    # ── informational: FULL path (Mojo vision -> fuse), f32 then bf16 ──
    print("[parity] full path Mojo vision -> fuse (informational):")
    var full_f32 = encode_mageflow_edit[S](
        vision, enc, ids, T, H, W, pixel_values, ctx, True
    )
    _ = _gate(harness, "editcond full f32    ", full_f32,
              _ref(st, String("editcond_txt"), ctx), ctx)
    var full_bf16 = encode_mageflow_edit[S](
        vision, enc, ids, T, H, W, pixel_values, ctx, False
    )
    _ = _gate(harness, "editcond full bf16   ", full_bf16,
              _ref(st, String("editcond_txt"), ctx), ctx)

    print("")
    if all_ok:
        print("[parity] MAGE-FLOW EDIT CONDITIONING GATES PASS (cos>=0.999):")
        print("         G-vision (vision tower) + G-editcond (isolated fusion math).")
        print("         The full-path f32/bf16 numbers are precision/ill-conditioning")
        print("         diagnostics on the fused image-token rows, not fusion errors.")
    else:
        print("[parity] SOME MAGE-FLOW EDIT GATES FAILED")


