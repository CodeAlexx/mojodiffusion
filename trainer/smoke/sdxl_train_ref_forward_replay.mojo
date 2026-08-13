# Replay the real Serenity SDXL train-step dump through the Mojo conv-UNet forward.
#
# Mirrors smoke/{ernie,anima}_train_ref_forward_replay.mojo in STRUCTURE, adapted
# to SDXL's conv-UNet eps-predictor and the real-dims resident forward the
# production driver uses (serenitymojo/training/train_sdxl_real.mojo:524
# sdxl_real_forward). It loads the REAL SDXL UNet (sdxl_unet_bf16.safetensors),
# the frozen oracle conditioning (pooled, add_time_ids, encoder_hidden_states,
# timestep) from parity/sdxl_train_ref_step000.safetensors, builds the SDXL ADM
# vector EXACTLY as the driver does (train_sdxl_real.mojo:467-480), and runs the
# B=0 LoRA forward.
#
# SDXL predict pipeline (mirrors train_sdxl_real.mojo, eps-pred NOT flow):
#   - ADM y = concat( pooled_clip_g[1280] , sin_embed_256(each of 6 add_time_ids)
#     -> 1536 ) = [1,2816]                            (train_sdxl.rs:861-867 / :467-480)
#   - context = encoder_hidden_states [1,77,2048]  (CLIP-L 768 + CLIP-G 1280 concat)
#   - t = unet_timestep (integer) as Float32 [1]
#   - x = scaled_noisy_latent_image -> NHWC -> sdxl_real_forward[L] -> eps NHWC.
#
# ── HONEST GATE STATUS (NOT a cos-vs-output.predicted parity gate) ──
# The frozen oracle was produced by the real PyTorch SDXL at the NATIVE aspect
# bucket: latent / output.predicted are [1,4,168,96] (RECTANGULAR). The
# serenitymojo real-dims forward sdxl_real_forward[L] is SQUARE-ONLY
# (H0=L, H1=L//2, H2=L//4) — it cannot consume a 168x96 latent, and a square crop
# run through a conv-UNet (receptive field + global GroupNorm) is NOT element-
# comparable to a crop of the full-res output. Additionally the serenitymojo SDXL
# LoRA covers only the 11 SpatialTransformer blocks, whereas the oracle trains a
# FULL-UNet LoRA (794 modules incl. convs/embeds). Therefore the native forward
# cos and the grad-norm cannot be gated against this dump with the current
# production spine. The oracle-consuming NUMERIC gate that DOES hold is the loss
# replay (smoke/sdxl_train_ref_loss_replay.mojo, PASS @ 1.4e-6).
#
# THIS smoke is the real-weight WIRING gate: it proves the full conv-UNet forward
# (embed + conv_in + 17 ResBlocks + 11 ST LoRA blocks + 2 down/2 up + final
# GN/SiLU/conv_out) runs end-to-end on the REAL checkpoint with the REAL oracle
# conditioning and produces a finite eps of the right shape. It runs on a square
# crop (L) of the real noisy latent. The MAIN LOOP owns the final bar.

from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor
from serenitymojo.models.vae.decoder2d import nchw_to_nhwc

from serenity_trainer.model.sdxl.weights import (
    build_sdxl_real_weights, sdxl_st_C, sdxl_st_Cff, sdxl_st_depth,
)
from serenity_trainer.model.sdxl.sdxl_unet_stack_lora import (
    SdxlLoraSet, build_sdxl_lora_set, sdxl_real_forward, N_ST, SDXL_SLOTS,
)


# ── arch (sdxl.json) ──────────────────────────────────────────────────────────
comptime CCTX = 2048
comptime NKV = 77
comptime ADM = 2816
comptime RANK = 16
comptime ALPHA = Float32(16.0)

# ── square crop spatial for the wiring forward (oracle is rectangular 168x96) ──
comptime L = 16

# ── dump latent dims [1,4,168,96] ─────────────────────────────────────────────
comptime LAT_C = 4
comptime LAT_H = 168
comptime LAT_W = 96

comptime PARITY = "trainer/parity/sdxl_train_ref_step000.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors"


def _dump_f32(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> List[Float32]:
    var t = Tensor.from_view(st.tensor_view(key), ctx)
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    if t.dtype() == STDtype.F32:
        return t.to_host(ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


# Read an I32 dump tensor directly from raw bytes (cast_tensor has no I32 path).
def _dump_i32(st: ShardedSafeTensors, key: String) raises -> List[Int]:
    var tv = st.tensor_view(key)
    var n = 1
    for i in range(len(tv.shape)):
        n *= tv.shape[i]
    var out = List[Int]()
    for i in range(n):
        var v = (
            Int(tv.data[i * 4 + 0])
            | (Int(tv.data[i * 4 + 1]) << 8)
            | (Int(tv.data[i * 4 + 2]) << 16)
            | (Int(tv.data[i * 4 + 3]) << 24)
        )
        out.append(v)
    return out^


# sin_embed_256 — EXACT copy of train_sdxl_real.mojo:230-241 (sdxl_sampler.rs).
def _sin_embed_256(value: Float32) -> List[Float32]:
    comptime DIM = 256
    comptime half = DIM // 2
    var data = List[Float32]()
    for _ in range(DIM):
        data.append(0.0)
    for j in range(half):
        var freq = Float32(fexp(-flog(10000.0) * Float64(j) / Float64(half)))
        var angle = value * freq
        data[j] = Float32(fcos(Float64(angle)))
        data[half + j] = Float32(fsin(Float64(angle)))
    return data^


def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def main() raises:
    var ctx = DeviceContext()

    print("=== SDXL train-ref forward replay (real-weight WIRING gate) ===")
    print("[parity]", PARITY)
    print("[ckpt]  ", CKPT)
    print("[note]  RECTANGULAR real-dims forward at", LAT_H, "x", LAT_W,
          "vs oracle output.predicted; cos printed at end.")

    var st = ShardedSafeTensors.open(String(PARITY))

    # ── oracle conditioning ──
    var pooled = _dump_f32(st, String("trace.added_cond_text_embeds"), ctx)   # [1,1280]
    var time_ids = _dump_f32(st, String("trace.added_cond_time_ids"), ctx)    # [1,6]
    var tstep = _dump_i32(st, String("trace.unet_timestep"))                  # [1] i32
    print("[cond] pooled n =", len(pooled), " time_ids =",
          time_ids[0], time_ids[1], time_ids[2], time_ids[3], time_ids[4], time_ids[5],
          " timestep =", tstep[0])

    # ADM y = concat(pooled[1280], sin_embed_256 of 6 time_ids -> 1536) = 2816.
    var y_h = List[Float32]()
    for i in range(len(pooled)):
        y_h.append(pooled[i])
    for k in range(6):
        var se = _sin_embed_256(time_ids[k])
        for j in range(len(se)):
            y_h.append(se[j])
    if len(y_h) != ADM:
        raise Error(String("ADM y length ") + String(len(y_h)) + " != 2816")
    var ys = List[Int](); ys.append(1); ys.append(ADM)
    # NOTE: build ADM y in BF16 (the bf16-training model dtype). The production
    # driver (train_sdxl_real.mojo:480) builds y as F32; that hits an elementwise
    # dtype mismatch in embed_forward's add(time_emb BF16, label_emb F32) because
    # timestep_embedding emits BF16 (weight dtype) while the F32-y label linear
    # stays F32 — a latent bug in the "not production-tested" driver. Feeding y as
    # BF16 matches the time path and the real bf16 forward.
    var y_f32 = Tensor.from_host(y_h^, ys^, STDtype.F32, ctx)
    var y = cast_tensor(y_f32, STDtype.BF16, ctx)

    # context = encoder_hidden_states [1,77,2048] (load in stored dtype).
    var context = Tensor.from_view(st.tensor_view(String("trace.encoder_hidden_states")), ctx)
    print("[ctx] encoder_hidden_states dims =", context.shape()[0], context.shape()[1], context.shape()[2])

    # t [1] = unet_timestep (integer) as Float32.
    var t_h = List[Float32](); t_h.append(Float32(tstep[0]))
    var t_s = List[Int](); t_s.append(1)
    var t = Tensor.from_host(t_h^, t_s^, STDtype.F32, ctx)

    # ── noisy latent: FULL rectangular trace.scaled_noisy_latent_image NCHW ──
    var noisy = _dump_f32(st, String("trace.scaled_noisy_latent_image"), ctx)   # [1,4,168,96] flat NCHW
    # BF16 latent (matches the driver's bf16 cache latent + bf16 UNet); the real-
    # dims forward widens to its F32 activation stream internally.
    var noisy_nchw_f32 = Tensor.from_host(noisy^, _sh4(1, LAT_C, LAT_H, LAT_W), STDtype.F32, ctx)
    var noisy_nchw = cast_tensor(noisy_nchw_f32, STDtype.BF16, ctx)
    var noisy_nhwc = nchw_to_nhwc(noisy_nchw, ctx)   # [1,LAT_H,LAT_W,4]
    print("[input] rectangular noisy NHWC [1,", LAT_H, ",", LAT_W, ",4]")

    # ── LoRA carrier (one set per ST, B=0 init -> base-UNet forward) ──
    var lora = List[SdxlLoraSet]()
    var n_adapters = 0
    for i in range(N_ST):
        var ls = build_sdxl_lora_set(sdxl_st_depth(i), sdxl_st_C(i), CCTX, sdxl_st_Cff(i), RANK, ALPHA)
        n_adapters += ls.num_blocks * SDXL_SLOTS
        lora.append(ls^)
    print("[lora] sets:", N_ST, " adapters:", n_adapters, " (B=0 identity)")

    # ── real base weights (frozen) ──
    print("[load] opening UNet checkpoint + assembling real weights ...")
    var stw = SafeTensors.open(String(CKPT))
    var w = build_sdxl_real_weights(stw, ctx)
    print("[load] weights ready")

    # ── forward (REAL rectangular dims) ──
    var fwd = sdxl_real_forward[LAT_H, LAT_W](noisy_nhwc, t, y.clone(ctx), context.clone(ctx), w, lora, ctx)
    var pred = fwd.out.to_host(ctx)   # NHWC flat [LAT_H*LAT_W*4]

    var n = len(pred)
    var nonfinite = 0
    var sumsq = Float64(0.0)
    var mn = Float32(1.0e30)
    var mx = Float32(-1.0e30)
    for i in range(n):
        var v = pred[i]
        if v != v or (v - v) != Float32(0.0):
            nonfinite += 1
            continue
        sumsq += Float64(v) * Float64(v)
        if v < mn: mn = v
        if v > mx: mx = v
    var rms = sqrt(sumsq / Float64(n))
    print("[out] eps NHWC n =", n, " rms =", rms, " min =", mn, " max =", mx,
          " nonfinite =", nonfinite)
    print("[out] eps[0:4] =", pred[0], pred[1], pred[2], pred[3])

    if n != LAT_H * LAT_W * LAT_C:
        raise Error("SDXL forward replay: output shape mismatch")
    if nonfinite != 0:
        raise Error("SDXL forward replay: nonfinite output")

    # ── PARITY: cos vs the oracle eps-pred output.predicted [1,4,LAT_H,LAT_W] NCHW ──
    # Mojo eps is NHWC [LAT_H,LAT_W,4]; map both into a common order for cosine.
    var ref_pred = _dump_f32(st, String("output.predicted"), ctx)   # NCHW flat
    if len(ref_pred) != LAT_C * LAT_H * LAT_W:
        raise Error("SDXL forward replay: output.predicted size mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for hh in range(LAT_H):
        for ww in range(LAT_W):
            for c in range(LAT_C):
                var a = Float64(pred[(hh * LAT_W + ww) * LAT_C + c])     # NHWC
                var b = Float64(ref_pred[(c * LAT_H + hh) * LAT_W + ww]) # NCHW
                dot += a * b
                na += a * a
                nb += b * b
    var cos = dot / ((na ** 0.5) * (nb ** 0.5) + 1.0e-30)
    print("[parity] cos(mojo eps, output.predicted) =", cos)
    print("[parity] |mojo|_2 =", na ** 0.5, "  |oracle|_2 =", nb ** 0.5)
    print("SDXL TRAIN REF FORWARD REPLAY (RECTANGULAR) DONE — cos printed above; main loop owns the bar")
