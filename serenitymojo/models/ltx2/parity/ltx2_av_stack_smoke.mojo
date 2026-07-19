# serenitymojo/models/ltx2/parity/ltx2_av_stack_smoke.mojo
#
# LTX-2 joint-AV STACK real-depth FINITE smoke (P6.1 gate b). Runs the FULL
# NUM_LAYERS (48) AV stack — head + 48 gated AV blocks (base only, LoRA off) +
# tail — at the gate geometry (S_V=128/S_A=16/N_TXT=128) on the oracle's inputs,
# and asserts finite outputs (no OOM at real depth; per-block recompute discipline
# means acts are NOT retained here, save_acts_k=0). Torch can't do a 48-block
# full-real-dim parity cheaply — this + the 2-block cos gate = composition proven.
#
# Run: rm -f serenitymojo.mojopkg; pixi run mojo build -O2 -I . -Xlinker -lm \
#   -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_stack_smoke.mojo \
#   -o /tmp/ltx2_av_stack_smoke && /tmp/ltx2_av_stack_smoke

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.env import (
    env_int, env_or, serenity_checkpoint, serenity_output,
)
from serenitymojo.models.dit.ltx2_dit import LTX2Config
from serenitymojo.models.ltx2.ltx2_av_stack import (
    LTX2AVBlockSource, ltx2_av_build_head, ltx2_av_stack_forward, _patchify,
)

comptime CKPT_NAME = "ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
comptime REF_PATH = "ltx2_av_stack/av_stack2_ref.safetensors"
comptime S_V = 128
comptime S_A = 16
comptime N_TXT = 128
comptime NUM_LAYERS = 48
comptime SIGMA = Float32(0.7)
comptime EPS = Float32(1e-6)


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx)


def _load_rope(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var t = Tensor.from_view_as_f32(st.tensor_view(name), ctx)
    var sh = t.shape()
    var H = sh[0]; var S = sh[1]; var hrd = sh[2]
    var host = t.to_host(ctx)
    var out = List[Float32]()
    out.resize(S * H * hrd, Float32(0.0))
    for h in range(H):
        for s in range(S):
            for j in range(hrd):
                out[(s * H + h) * hrd + j] = host[(h * S + s) * hrd + j]
    return Tensor.from_host(out, _sh2(S * H, hrd), STDtype.F32, ctx)


def _finite(name: String, t: Tensor, ctx: DeviceContext) raises -> Int:
    var h = t.to_host(ctx)
    var bad = 0
    var mn = Float64(1e30); var mx = Float64(-1e30); var acc = 0.0
    for i in range(len(h)):
        var x = Float64(h[i])
        if x != x or (x - x != 0.0):
            bad += 1
        else:
            if x < mn: mn = x
            if x > mx: mx = x
            acc += x * x
    var std = (acc / Float64(len(h))) ** 0.5
    print("  ", "OK  " if bad == 0 else "BAD ", name, "n=", len(h), "nonfinite=", bad,
          "min=", mn, "max=", mx, "rms=", std)
    return bad


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var ckpt = env_or("LTX2_AV_CKPT", serenity_checkpoint(String(CKPT_NAME)))
    var ref_path = env_or("LTX2_AV_REF", serenity_output(String(REF_PATH)))
    print("=== LTX-2 joint-AV STACK real-depth (", NUM_LAYERS, "-block) FINITE smoke ===")
    print("  S_V/S_A/N_TXT:", S_V, S_A, N_TXT, " sigma:", SIGMA, " LoRA: OFF")

    var dump = ShardedSafeTensors.open(ref_path)
    var v_lat = _load(dump, "v_lat", ctx)
    var a_lat = _load(dump, "a_lat", ctx)
    var enc = _load(dump, "enc", ctx)
    var aenc = _load(dump, "aenc", ctx)
    var v_cos = _load_rope(dump, "v_cos", ctx)
    var v_sin = _load_rope(dump, "v_sin", ctx)
    var a_cos = _load_rope(dump, "a_cos", ctx)
    var a_sin = _load_rope(dump, "a_sin", ctx)
    var ca_v_cos = _load_rope(dump, "ca_v_cos", ctx)
    var ca_v_sin = _load_rope(dump, "ca_v_sin", ctx)
    var ca_a_cos = _load_rope(dump, "ca_a_cos", ctx)
    var ca_a_sin = _load_rope(dump, "ca_a_sin", ctx)

    var st = ShardedSafeTensors.open(ckpt)

    # residency A/B — see ltx2_av_stack_bwd_parity.mojo for the rationale.
    var blk_ckpt = env_or("LTX2_AV_BLOCK_CKPT", ckpt)
    var src = LTX2AVBlockSource.open(blk_ckpt, cfg)
    var n_res = env_int("LTX2_RESIDENT_BLOCKS", 0)
    if n_res > NUM_LAYERS:
        n_res = NUM_LAYERS
    print("  [blocks] source:", blk_ckpt)
    if n_res > 0:
        src.enable_resident(0, n_res - 1, ctx)
        print("  [blocks] RESIDENT 0 ..", n_res - 1, " bytes:", src.resident_bytes())
    else:
        print("  [blocks] streamed (residency OFF)")
    var head = ltx2_av_build_head[S_V, S_A, N_TXT](st, SIGMA, ctx)
    var hidden = _patchify(v_lat, st, String("patchify_proj"), ctx)
    var ahs = _patchify(a_lat, st, String("audio_patchify_proj"), ctx)

    var empty = List[ArcPointer[Tensor]]()   # LoRA off -> base only
    print("  [forward] head +", NUM_LAYERS, "blocks + tail (streaming per-block weights)")
    var fwd = ltx2_av_stack_forward[S_V, S_A, N_TXT](
        st, src, hidden, ahs, enc, aenc, head,
        v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
        empty, empty, Float32(0.0), NUM_LAYERS, EPS, ctx, 0,
    )

    var bad = 0
    bad += _finite(String("video_vel"), fwd.video_vel, ctx)
    bad += _finite(String("audio_vel"), fwd.audio_vel, ctx)
    print("")
    if bad == 0:
        print("LTX2 AV stack", NUM_LAYERS, "-block FINITE smoke PASS: finite outputs, no OOM at real depth")
    else:
        raise Error(String("LTX2 AV stack finite smoke FAIL: ") + String(bad) + " nonfinite elements")
