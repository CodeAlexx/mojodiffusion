# lens_sampler_smoke.mojo — Lens sampler schedule + VAE decode-tail parity gates.
#
# Reference policy: Serenity ONLY. The oracle (parity/lens/lens_sampler_oracle.py)
# dumps the FlowMatchEulerDiscreteScheduler schedule and the EXACT LensSampler.py
# :139-147 decode tail (unpack -> unscale -> unpatchify -> vae.decode, plain
# AutoencoderKLFlux2) against a fixed packed scaled latent. No Rust.
#
# GATE A (schedule parity): Mojo build_lens_shifted_sigmas / timesteps for
#   H=W=128, steps=20 (image_seq_len=64) vs sampler_schedule_ref.json. The Mojo
#   list has exactly N=20 shifted sigmas (the terminal 0.0 lives only as the
#   final Euler sigma_next), so we compare against ref[0:20]. timesteps = sigma*1000.
#   PASS within 1e-4 abs.
#
# GATE B (decode-tail parity): load sampler_tail_in.safetensors (packed scaled
#   latent [1,64,128]) + the REAL Lens VAE, run the SAME tail the sampler wires
#   (unpack(8,8) -> LensVAE.decode, which fuses the SINGLE unscale + unpatchify +
#   conv decode internally), compare to sampler_tail_out.safetensors [1,3,128,128].
#   PASS PSNR >= 25 dB (BF16 VAE vs F32 oracle). A double-unscale tanks PSNR.

from max.gpu.host import DeviceContext
from std.math import log10

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value
from serenitymojo.io.train_config_reader import _parse_number

from serenity_trainer.util.config.TrainConfigReader import _read_file_bytes
from serenity_trainer.sampling.lens_flowmatch import build_lens_shifted_sigmas
from serenity_trainer.modelSetup.BaseLensSetup import unpack_latents
from serenity_trainer.model.LensVAE import LensVAE
# Force the sampler driver itself to compile (step 3 gate).
from serenity_trainer.modelSampler.LensSampler import (
    sample_lens, LensSampleOutput, cfg_norm_rescale_pair,
)


comptime SCHEDULE_REF = "trainer/parity/lens/sampler_schedule_ref.json"
comptime TAIL_IN      = "trainer/parity/lens/sampler_tail_in.safetensors"
comptime TAIL_OUT     = "trainer/parity/lens/sampler_tail_out.safetensors"
comptime VAE_DIR      = "/home/alex/.serenity/models/microsoft_lens/vae"

comptime STEPS   = 20
comptime SEQ_LEN = 64          # (128//8//2) * (128//8//2) = 8*8
comptime H_LAT   = 8           # packed latent spatial (post-patchify) for H=128
comptime W_LAT   = 8


# ── JSON float-array parser (reuses the proven signed/scientific number parser
#    from serenitymojo.io.train_config_reader; json_header._parse_int is int-only).
def _parse_float_array(mut cur: _Cursor) raises -> List[Float64]:
    var out = List[Float64]()
    cur.expect(0x5B)              # '['
    cur.skip_ws()
    if cur.peek() == 0x5D:        # ']' empty
        cur.advance()
        return out^
    while True:
        out.append(_parse_number(cur))
        cur.skip_ws()
        var c = cur.peek()
        if c == 0x2C:            # ','
            cur.advance()
            continue
        if c == 0x5D:            # ']'
            cur.advance()
            break
        raise Error(String("JSON float-array: expected ',' or ']' at byte ")
                    + String(cur.pos))
    return out^


# Extract named top-level float arrays from the schedule ref JSON.
def _load_named_arrays(path: String, name_a: String, name_b: String) raises -> Tuple[List[Float64], List[Float64]]:
    var bytes = _read_file_bytes(path)
    var cur = _Cursor(bytes^)
    var arr_a = List[Float64]()
    var arr_b = List[Float64]()
    cur.expect(0x7B)             # '{'
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return (arr_a^, arr_b^)
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)        # ':'
        if key == name_a:
            arr_a = _parse_float_array(cur)
        elif key == name_b:
            arr_b = _parse_float_array(cur)
        else:
            _skip_value(cur)
        cur.skip_ws()
        var c = cur.peek()
        if c == 0x2C:
            cur.advance()
            continue
        if c == 0x7D:
            cur.advance()
            break
        raise Error(String("schedule ref: expected ',' or '}' at byte ")
                    + String(cur.pos))
    return (arr_a^, arr_b^)


def _load_x(path: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    return Tensor.from_view(st.tensor_view(String("x")), ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== Lens sampler schedule + decode-tail parity smoke ===")
    print("  reference: Serenity lens_sampler_oracle.py (no Rust)")

    # ───────────────────────── GATE A: schedule parity ─────────────────────────
    var refs = _load_named_arrays(String(SCHEDULE_REF),
                                  String("shifted_sigmas"), String("timesteps"))
    var ref_sigmas = refs[0].copy()
    var ref_timesteps = refs[1].copy()
    print("")
    print("── GATE A: schedule parity (steps=20, image_seq_len=64) ─────────")
    print("  ref shifted_sigmas len =", len(ref_sigmas), " (incl terminal 0)")
    print("  ref timesteps len      =", len(ref_timesteps))

    var mojo_sigmas = build_lens_shifted_sigmas(STEPS, SEQ_LEN)
    if len(mojo_sigmas) != STEPS:
        raise Error(String("mojo sigmas len ") + String(len(mojo_sigmas))
                    + String(" != ") + String(STEPS))

    var max_sig_diff = Float64(0.0)
    var max_ts_diff = Float64(0.0)
    for i in range(STEPS):
        var ms = Float64(mojo_sigmas[i])
        var ds = ms - ref_sigmas[i]
        if ds < 0: ds = -ds
        if ds > max_sig_diff: max_sig_diff = ds
        var mt = ms * 1000.0                 # timestep = sigma * 1000 (FlowMatch)
        var dt = mt - ref_timesteps[i]
        if dt < 0: dt = -dt
        if dt > max_ts_diff: max_ts_diff = dt

    print("  max |Δ shifted_sigma|  =", max_sig_diff)
    print("  max |Δ timestep|       =", max_ts_diff)
    var gate_a = (max_sig_diff <= 1.0e-4) and (max_ts_diff <= 1.0e-4)
    print("  GATE A:", "OK" if gate_a else "FAIL")

    # ───────────────────────── GATE B: decode-tail parity ──────────────────────
    print("")
    print("── GATE B: decode-tail parity (unpack -> vae.decode) ────────────")
    var packed = _load_x(String(TAIL_IN), ctx)         # [1,64,128] F32
    var psh = packed.shape()
    print("  packed in shape        = [", psh[0], ",", psh[1], ",", psh[2], "]")

    var vae = LensVAE[H_LAT, W_LAT].load(String(VAE_DIR), ctx)
    # SAME tail the sampler wires: unpack(8,8) -> LensVAE.decode (the decoder fuses
    # the single unscale + unpatchify + conv internally). Exactly ONE unscale.
    var unpacked = unpack_latents(packed, H_LAT, W_LAT, ctx)   # [1,128,8,8] packed scaled
    var image = vae.decode(unpacked, ctx)                      # [1,3,128,128]
    var ish = image.shape()
    print("  decoded image shape    = [", ish[0], ",", ish[1], ",", ish[2], ",", ish[3], "]")

    var got = image.to_host(ctx)
    var ref_img_t = _load_x(String(TAIL_OUT), ctx)
    var ref_img = ref_img_t.to_host(ctx)
    if len(got) != len(ref_img):
        raise Error(String("decode length mismatch ") + String(len(got))
                    + String(" vs ") + String(len(ref_img)))

    var mse = Float64(0.0)
    var mad = Float64(0.0)
    var rmin = ref_img[0]
    var rmax = ref_img[0]
    for i in range(len(ref_img)):
        var d = Float64(got[i]) - Float64(ref_img[i])
        mse += d * d
        var ad = d
        if ad < 0: ad = -ad
        mad += ad
        if ref_img[i] < rmin: rmin = ref_img[i]
        if ref_img[i] > rmax: rmax = ref_img[i]
    var n = Float64(len(ref_img))
    mse /= n
    mad /= n
    var data_range = Float64(rmax) - Float64(rmin)
    var psnr = Float64(0.0)
    if mse <= 0.0:
        psnr = 999.0
    else:
        psnr = 10.0 * log10((data_range * data_range) / mse)

    print("  ref image range        = [", rmin, ",", rmax, "]  (data_range =", data_range, ")")
    print("  mean-abs-diff          =", mad)
    print("  MSE                    =", mse)
    print("  PSNR (dB)              =", psnr)
    var gate_b = psnr >= 25.0
    print("  GATE B:", "OK (PSNR>=25dB)" if gate_b else "FAIL (PSNR<25dB)")

    print("")
    print("──────────────────────────────────────────────────────────")
    print("  GATE A (schedule) =", "OK" if gate_a else "FAIL")
    print("  GATE B (decode)   =", "OK" if gate_b else "FAIL")
    var gate = gate_a and gate_b
    print("  OVERALL:", "OK" if gate else "FAIL")
    if not gate:
        raise Error("lens_sampler_smoke GATE FAIL")
