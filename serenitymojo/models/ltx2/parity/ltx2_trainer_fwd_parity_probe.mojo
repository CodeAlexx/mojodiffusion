# ltx2_trainer_fwd_parity_probe.mojo — MOJO side of the trainer forward+loss
# parity gate vs musubi's own runtime (scripts/ltx2_parity_ref_gen.py).
#
# Byte-matched inputs: the SAME cache latents/TE embeds (bf16 files) and the
# SAME fixture noise (BF16) + sigma; runs the production driver's exact
# forward path (bf16 head -> F32 48-block stack, fp8-resident base, LoRA
# absent == B=0) and the driver's loss (F32 MSE vs noise-latent target).
# Dumps pred_k (F32 tokens [1,S_V,C]) + loss_k to a safetensors for
# scripts/ltx2_parity_compare.py. Bars live in the compare script
# (pred cosine >= 0.999; loss reported with the bf16-target-class note).
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/ltx2/parity/ltx2_trainer_fwd_parity_probe.mojo \
#     -o /tmp/ltx2_fwd_parity_probe
#   (launch via scripts/ltx2_parity_run_probe.py — argv carries the pairs)
#
# argv: <fixture.safetensors> <ckpt> <out.safetensors> then per pair 4 args:
#       <arm(video|image)> <lat_path> <te_path> <sigma>

from std.sys import argv
from std.collections import List
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape, permute, sub, mul_scalar, add

from serenitymojo.models.dit.ltx2_dit import LTX2Config
from serenitymojo.models.ltx2.ltx2_video_stack import (
    LTX2VideoBlockSource, LTX2VideoTail, LTX2VideoStackHead,
    ltx2_video_stack_lora_forward,
)
from serenitymojo.training.ltx2.config import _parse_float32

comptime C = 128
comptime VD = 4096
comptime N_TXT = 1024
comptime NUM_LAYERS = 48
comptime FRAME_RATE = Float64(25.0)
comptime EPS = Float32(1e-6)


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _sh5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d); s.append(e)
    return s^


def _perm021() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); return p^


def _load_bf16(path: String, name: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    return Tensor.from_view_as_bf16(st.tensor_view(name), ctx)


def _zero_lora(
    n: Int, ctx: DeviceContext,
) raises -> Tuple[List[ArcPointer[Tensor]], List[ArcPointer[Tensor]]]:
    # rank-1 all-zero adapters: B=0 => LoRA contributes exactly nothing.
    var la = List[ArcPointer[Tensor]]()
    var lb = List[ArcPointer[Tensor]]()
    for _ in range(n):
        var az = List[Float32]()
        for _ in range(VD):
            az.append(Float32(0.0))
        var bz = List[Float32]()
        for _ in range(VD):
            bz.append(Float32(0.0))
        la.append(ArcPointer[Tensor](Tensor.from_host(az^, [1, VD], STDtype.F32, ctx)))
        lb.append(ArcPointer[Tensor](Tensor.from_host(bz^, [VD, 1], STDtype.F32, ctx)))
    return (la^, lb^)


struct Pair(Copyable, Movable):
    var arm: String
    var lat: String
    var te: String
    var sigma: Float32
    var idx: Int

    def __init__(out self, var arm: String, var lat: String, var te: String,
                 sigma: Float32, idx: Int):
        self.arm = arm^
        self.lat = lat^
        self.te = te^
        self.sigma = sigma
        self.idx = idx


def _run_arm[S_Vp: Int, NFp: Int, NHp: Int, NWp: Int](
    pairs: List[Pair], want_arm: String, fixture: String, ckpt: String,
    mut out_names: List[String], mut out_tensors: List[ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises:
    var todo = List[Int]()
    for i in range(len(pairs)):
        if pairs[i].arm == want_arm:
            todo.append(i)
    if len(todo) == 0:
        return
    var lat_key: String
    if want_arm == "video":
        lat_key = String("latents_4x9x16_bfloat16")
    else:
        lat_key = String("latents_1x16x16_bfloat16")

    print("[probe]", want_arm, "arm:", len(todo), "pairs; loading stack from", ckpt)
    var model_cfg = LTX2Config.ltx2()
    var head = LTX2VideoStackHead.load(ckpt, ctx)
    var tail = LTX2VideoTail.load(ckpt, True, ctx)
    var src = LTX2VideoBlockSource.open(ckpt, model_cfg, True)
    src.stream.enable_fp8_resident_range(0, 41, ctx)
    var dev = _zero_lora(NUM_LAYERS * 8, ctx)

    for ref ti in todo:
        ref pr = pairs[ti]
        var latent5 = reshape(
            _load_bf16(pr.lat, lat_key, ctx), _sh5(1, C, NFp, NHp, NWp), ctx)
        var enc = reshape(
            _load_bf16(pr.te, String("text_bfloat16"), ctx), _sh3(1, N_TXT, VD), ctx)
        var fixture_st = ShardedSafeTensors.open(fixture)
        var noise_b = reshape(
            Tensor.from_view_as_bf16(
                fixture_st.tensor_view(String("noise_") + String(pr.idx)), ctx),
            _sh5(1, C, NFp, NHp, NWp), ctx)

        var sigma = pr.sigma
        var latf = cast_tensor(latent5, STDtype.F32, ctx)
        var noise = cast_tensor(noise_b, STDtype.F32, ctx)
        var noisy = add(
            mul_scalar(latf, Float32(1.0) - sigma, ctx),
            mul_scalar(noise, sigma, ctx), ctx)
        var target5 = sub(noise, latf, ctx)

        var noisy_bf16 = cast_tensor(noisy, STDtype.BF16, ctx)
        var ho = head.forward[S_Vp, N_TXT](
            noisy_bf16, enc, sigma, NFp, NHp, NWp, FRAME_RATE, ctx)
        var hidden = cast_tensor(ho.hidden, STDtype.F32, ctx)
        var v_temb = cast_tensor(ho.v_temb, STDtype.F32, ctx)
        var v_embedded = cast_tensor(ho.v_embedded, STDtype.F32, ctx)
        var v_prompt_ts = cast_tensor(ho.v_prompt_ts, STDtype.F32, ctx)
        var v_cos = cast_tensor(ho.v_cos, STDtype.F32, ctx)
        var v_sin = cast_tensor(ho.v_sin, STDtype.F32, ctx)
        var encf = cast_tensor(enc, STDtype.F32, ctx)

        var fwd = ltx2_video_stack_lora_forward[S_Vp, N_TXT](
            hidden, encf, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
            tail, src, dev[0], dev[1], Float32(1.0), NUM_LAYERS, EPS, ctx,
        )

        # driver loss: host F32/F64-accum MSE over [1,S_V,C] tokens
        var predh = fwd.pred.to_host(ctx)
        var targh = permute(
            reshape(target5, _sh3(1, C, S_Vp), ctx), _perm021(), ctx
        ).to_host(ctx)
        if len(predh) != len(targh):
            raise Error("pred/target numel mismatch")
        var loss64 = 0.0
        for i in range(len(predh)):
            var d = Float64(predh[i]) - Float64(targh[i])
            loss64 += d * d
        var loss = Float32(loss64 / Float64(len(predh)))
        print("[probe] pair", pr.idx, "(", pr.arm, "s=", sigma, "): mojo loss =", loss)

        out_names.append(String("pred_") + String(pr.idx))
        out_tensors.append(ArcPointer[Tensor](Tensor.from_host(
            predh.copy(), _sh3(1, S_Vp, C), STDtype.F32, ctx)))
        var lt = List[Float32]()
        lt.append(loss)
        out_names.append(String("loss_") + String(pr.idx))
        out_tensors.append(ArcPointer[Tensor](
            Tensor.from_host(lt^, [1], STDtype.F32, ctx)))


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(len(raw)):
        args.append(String(raw[i]))
    if len(args) < 8 or (len(args) - 4) % 4 != 0:
        raise Error("usage: <fixture> <ckpt> <out> [<arm> <lat> <te> <sigma>]...")
    var fixture = String(args[1])
    var ckpt = String(args[2])
    var out_path = String(args[3])

    var pairs = List[Pair]()
    var i = 4
    var idx = 0
    while i + 3 < len(args):
        pairs.append(Pair(
            String(args[i]), String(args[i + 1]), String(args[i + 2]),
            _parse_float32(String(args[i + 3])), idx,
        ))
        i += 4
        idx += 1
    print("[probe]", len(pairs), "pairs")

    var ctx = DeviceContext()
    var out_names = List[String]()
    var out_tensors = List[ArcPointer[Tensor]]()

    _run_arm[576, 4, 9, 16](pairs, String("video"), fixture, ckpt,
                            out_names, out_tensors, ctx)
    _run_arm[256, 1, 16, 16](pairs, String("image"), fixture, ckpt,
                             out_names, out_tensors, ctx)

    save_safetensors(out_names, out_tensors, out_path, ctx)
    print("[probe] DONE:", len(out_names), "tensors ->", out_path)
