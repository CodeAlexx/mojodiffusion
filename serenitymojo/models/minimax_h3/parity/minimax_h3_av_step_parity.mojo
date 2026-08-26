# minimax_h3_av_step_parity — gate the AV training-step RECIPE math for BOTH
# modalities against the torch oracle dump (minimax_h3_av_step_oracle.py):
# per-modality sigma shift, bf16 noising, velocity targets, masked
# token-balanced joint loss, and d_pred — plus the audio row-pack order
# cross-checked against the GATED host unpack (rearrange.mojo).
#
# PASS bars: x_t / target / d_pred BIT-equal per modality; element counts
# exact; joint loss within 1e-9 relative; pack order exact vs
# minimax_h3_unpack_audio.
#
# Regenerate inputs:
#   scripts/minimax_h3_synth_cache.py ~/datasets/h3_synth_cache/av256 av256 2
#   parity/minimax_h3_av_step_oracle.py <item latent> output/checks/h3_av_step_oracle.safetensors
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.models.minimax_h3.h3_train_cache import h3_read_latent_cache
from serenitymojo.models.minimax_h3.h3_train_sigma import (
    h3_shift_sigma, h3_noisy_input, h3_velocity_target,
    h3_modality_loss, h3_joint_token_loss, h3_loss_grad,
)
from serenitymojo.models.minimax_h3.h3_train_av import (
    h3_audio_latents_to_rows, h3_audio_rows_to_latents, h3_audio_mask_tensor,
)
from serenitymojo.models.minimax_h3.rearrange import minimax_h3_unpack_audio

comptime ITEM = "/home/alex/datasets/h3_synth_cache/av256/synth_av256_000_0448x0256_mmh3.safetensors"
comptime ORACLE = "/home/alex/mojodiffusion/output/checks/h3_av_step_oracle.safetensors"
comptime U = Float32(0.4375)
comptime EXPECT_LOSS = Float64(3.0016357313587654)
comptime EXPECT_NV = 397824
comptime EXPECT_NA = 13056


def _load(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _bits(a: Tensor, b: Tensor, label: String, ctx: DeviceContext) raises:
    var ah = a.to_host_bf16(ctx)
    var bh = b.to_host_bf16(ctx)
    if len(ah) != len(bh):
        raise Error(label + ": length mismatch")
    var diff = 0
    for i in range(len(ah)):
        if ah[i] != bh[i]:
            diff += 1
    print(label, "diffs", diff)
    if diff != 0:
        raise Error(label + " not bit-equal")


def main() raises:
    var ctx = DeviceContext()
    var lat = h3_read_latent_cache(String(ITEM), ctx)
    if not lat.has_audio or lat.audio_t != 207:
        raise Error("expected the av256 fixture item")
    var orc = SafeTensors.open(String(ORACLE))

    # ── audio pack order vs the GATED host unpack ───────────────────────────
    var rows = h3_audio_latents_to_rows(lat.audio[], ctx)
    var rows_host = rows.to_host(ctx)          # [2T, 32] f32 host
    var unpacked = minimax_h3_unpack_audio(rows_host, lat.audio_t, 32)
    var audio_host = lat.audio[].to_host(ctx)  # [2, 32, T] f32 host
    if len(unpacked) != len(audio_host):
        raise Error("pack order: length mismatch")
    for i in range(len(unpacked)):
        if unpacked[i] != audio_host[i]:
            raise Error("pack order mismatch vs gated unpack at " + String(i))
    print("audio pack order matches gated unpack")
    var back = h3_audio_rows_to_latents(rows, lat.audio_t, ctx)
    _bits(back, lat.audio[], String("audio pack roundtrip"), ctx)

    # ── per-modality sigma + noising + targets ──────────────────────────────
    var sigma_v = h3_shift_sigma(U, Float32(12.0))
    var sigma_a = h3_shift_sigma(U, Float32(3.0))
    print("sigma_v", sigma_v, "sigma_a", sigma_a)

    var noise_v = _load(orc, String("noise_v"), ctx)
    var noise_a = _load(orc, String("noise_a"), ctx)
    var xt_v = h3_noisy_input(lat.video[], noise_v, sigma_v, ctx)
    var xt_a = h3_noisy_input(lat.audio[], noise_a, sigma_a, ctx)
    _bits(xt_v, _load(orc, String("xt_v"), ctx), String("x_t video"), ctx)
    _bits(xt_a, _load(orc, String("xt_a"), ctx), String("x_t audio"), ctx)

    var tgt_v = h3_velocity_target(lat.video[], noise_v, ctx)
    var tgt_a = h3_velocity_target(lat.audio[], noise_a, ctx)
    _bits(tgt_v, _load(orc, String("tgt_v"), ctx), String("target video"), ctx)
    _bits(tgt_a, _load(orc, String("tgt_a"), ctx), String("target audio"), ctx)

    # ── masked token-balanced joint loss ────────────────────────────────────
    var pred_v = _load(orc, String("pred_v"), ctx)
    var pred_a = _load(orc, String("pred_a"), ctx)
    var no_mask = List[Bool]()
    var ml_v = h3_modality_loss(pred_v, tgt_v, no_mask, ctx)
    var ml_a = h3_modality_loss(pred_a, tgt_a, lat.audio_loss_mask, ctx)
    if ml_v.elements != EXPECT_NV or ml_a.elements != EXPECT_NA:
        raise Error("element counts diverge from oracle")
    var loss = h3_joint_token_loss(ml_v, ml_a, Float64(1.0), Float64(1.0))
    var rel = abs(loss - EXPECT_LOSS) / EXPECT_LOSS
    print("loss", loss, "rel_err", rel)
    if rel > 1e-9:
        raise Error("joint loss diverges from oracle")

    # ── d_pred both modalities ──────────────────────────────────────────────
    var denom = Float64(ml_v.elements + ml_a.elements)
    var none_mask = Optional[Tensor](None)
    var d_v = h3_loss_grad(pred_v, tgt_v, none_mask^, 1.0, denom, ctx)
    var mask_t = h3_audio_mask_tensor(lat.audio_loss_mask, 32, ctx)
    var some_mask = Optional[Tensor](mask_t^)
    var d_a = h3_loss_grad(pred_a, tgt_a, some_mask^, 1.0, denom, ctx)
    _bits(d_v, _load(orc, String("d_pred_v"), ctx), String("d_pred video"), ctx)
    _bits(d_a, _load(orc, String("d_pred_a"), ctx), String("d_pred audio"), ctx)

    print("minimax_h3_av_step_parity PASS")
