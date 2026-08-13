# serenitymojo/models/ltx2/parity/ltx2_av_stack_parity.mojo
#
# LTX-2 joint-AV 2-block STACK forward parity gate (P6.1 gate a).
#
# Compares the Mojo AV training stack
#   models/ltx2/ltx2_av_stack.mojo  ltx2_av_build_head + ltx2_av_stack_forward
# against the torch oracle
#   scripts/ltx2_av_stack_oracle.py -> output/ltx2_av_stack/av_stack2_ref.safetensors
#
# Reads the oracle's BYTE-IDENTICAL inputs (v_lat/a_lat/enc/aenc + 4 rope pairs +
# per-block 24-pair LoRA), computes the head from sigma=0.7, runs head+2 blocks+
# tail, and gates cos >= 0.999 on: video_vel/audio_vel (full stack), the pre-tail
# hidden (stage isolation), and ALL 16 saved acts per block (the acts contract
# the gated backward consumes). REAL block-0/1 weights, F32 (synthetic-dims gate).
#
# Run:
#   ${SERENITY_ORACLE_PYTHON:-python3} scripts/ltx2_av_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/models/ltx2/parity/ltx2_av_stack_parity.mojo -o /tmp/ltx2_av_stack_parity
#   /tmp/ltx2_av_stack_parity

from max.gpu.host import DeviceContext
from std.collections import List
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.env import (
    env_int, env_or, serenity_checkpoint, serenity_output,
)
from serenitymojo.models.dit.ltx2_dit import LTX2Config
from serenitymojo.models.ltx2.ltx2_av_stack import (
    LTX2AVBlockSource, ltx2_av_build_head, ltx2_av_stack_forward, _patchify, _av_lora_slots,
)

comptime CKPT_NAME = "ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
comptime REF_PATH = "ltx2_av_stack/av_stack2_ref.safetensors"
comptime S_V = 128
comptime S_A = 16
comptime N_TXT = 128
comptime N_BLOCKS = 2
comptime SIGMA = Float32(0.7)
comptime LORA_SCALE = Float32(0.5)
comptime EPS = Float32(1e-6)
comptime COS_GATE = 0.999


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx)


# oracle rope [H,S,hrd] -> block row order [(s,h),hrd] (block-gate convention).
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


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error(String("cos: length ") + String(len(a)) + " vs " + String(len(b)))
    var dot = 0.0; var na = 0.0; var nb = 0.0
    for i in range(len(a)):
        var x = Float64(a[i]); var y = Float64(b[i])
        if x != x or y != y:
            raise Error("cos: NaN")
        dot += x * y; na += x * x; nb += y * y
    return dot / (sqrt(na) * sqrt(nb) + 1e-30)


def _check(tag: String, mojo: Tensor, ref_name: String, dump: ShardedSafeTensors, ctx: DeviceContext) raises -> Int:
    var m = mojo.to_host(ctx)
    var r = _load(dump, ref_name, ctx).to_host(ctx)
    var c = _cos(m, r)
    var ok = c >= COS_GATE
    print("  ", "PASS" if ok else "FAIL", tag, "cos=", c)
    return 0 if ok else 1


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var ckpt = env_or("LTX2_AV_CKPT", serenity_checkpoint(String(CKPT_NAME)))
    var ref_path = env_or("LTX2_AV_REF", serenity_output(String(REF_PATH)))
    print("=== LTX-2 joint-AV 2-block STACK forward parity gate ===")
    print("  S_V/S_A/N_TXT:", S_V, S_A, N_TXT, " blocks:", N_BLOCKS, " sigma:", SIGMA)
    print("  oracle:", ref_path)

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

    # per-block LoRA, flat [N_BLOCKS*24] in slot order.
    var slots = _av_lora_slots()
    var lora_a = List[ArcPointer[Tensor]]()
    var lora_b = List[ArcPointer[Tensor]]()
    for i in range(N_BLOCKS):
        for s in range(len(slots)):
            lora_a.append(ArcPointer[Tensor](_load(dump, String("b") + String(i) + ".lora." + slots[s] + ".A", ctx)))
            lora_b.append(ArcPointer[Tensor](_load(dump, String("b") + String(i) + ".lora." + slots[s] + ".B", ctx)))
    print("  [lora] loaded", len(lora_a), "adapters over", N_BLOCKS, "blocks")

    var st = ShardedSafeTensors.open(ckpt)

    # residency A/B — see ltx2_av_stack_bwd_parity.mojo for the rationale:
    # LTX2_AV_BLOCK_CKPT swaps ONLY the block-weight source, head/rope/LoRA hold.
    var blk_ckpt = env_or("LTX2_AV_BLOCK_CKPT", ckpt)
    var src = LTX2AVBlockSource.open(blk_ckpt, cfg)
    var n_res = env_int("LTX2_RESIDENT_BLOCKS", 0)
    if n_res > N_BLOCKS:
        n_res = N_BLOCKS
    print("  [blocks] source:", blk_ckpt)
    if n_res > 0:
        src.enable_resident(0, n_res - 1, ctx)
        print("  [blocks] RESIDENT 0 ..", n_res - 1, " bytes:", src.resident_bytes())
    else:
        print("  [blocks] streamed (residency OFF)")
    print("  [head] build modulation from sigma")
    var head = ltx2_av_build_head[S_V, S_A, N_TXT](st, SIGMA, ctx)
    var hidden = _patchify(v_lat, st, String("patchify_proj"), ctx)
    var ahs = _patchify(a_lat, st, String("audio_patchify_proj"), ctx)

    print("  [forward] head + ", N_BLOCKS, " blocks + tail")
    var fwd = ltx2_av_stack_forward[S_V, S_A, N_TXT](
        st, src, hidden, ahs, enc, aenc, head,
        v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
        lora_a, lora_b, LORA_SCALE, N_BLOCKS, EPS, ctx, N_BLOCKS,
    )

    var fails = 0
    # full-stack output.
    fails += _check(String("video_vel"), fwd.video_vel, String("video_vel"), dump, ctx)
    fails += _check(String("audio_vel"), fwd.audio_vel, String("audio_vel"), dump, ctx)
    # stage isolation: pre-tail hidden = last block output. Reads the explicit
    # v_last/a_last fields (which is also what gates them against the oracle) —
    # saved[N_BLOCKS-1] only exists when save_acts_k == N_BLOCKS.
    fails += _check(String("video_hidden(pre-tail)"), fwd.v_last, String("video_hidden"), dump, ctx)
    fails += _check(String("audio_hidden(pre-tail)"), fwd.a_last, String("audio_hidden"), dump, ctx)

    # acts contract — all 16 LTX2AVBlockActs fields per block.
    for i in range(N_BLOCKS):
        ref ac = fwd.saved[i][].acts
        var p = String("b") + String(i) + ".act."
        fails += _check(p + "hidden", ac.hidden, p + "hidden", dump, ctx)
        fails += _check(p + "ahs", ac.ahs, p + "ahs", dump, ctx)
        fails += _check(p + "at1", ac.at1.out, p + "at1", dump, ctx)
        fails += _check(p + "aat1", ac.aat1.out, p + "aat1", dump, ctx)
        fails += _check(p + "hs1", ac.hs1, p + "hs1", dump, ctx)
        fails += _check(p + "ahss1", ac.ahss1, p + "ahss1", dump, ctx)
        fails += _check(p + "at2", ac.at2.out, p + "at2", dump, ctx)
        fails += _check(p + "aat2", ac.aat2.out, p + "aat2", dump, ctx)
        fails += _check(p + "hs2", ac.hs2, p + "hs2", dump, ctx)
        fails += _check(p + "ahss2", ac.ahss2, p + "ahss2", dump, ctx)
        fails += _check(p + "a2v", ac.a2v.out, p + "a2v", dump, ctx)
        fails += _check(p + "v2a", ac.v2a.out, p + "v2a", dump, ctx)
        fails += _check(p + "hs3", ac.hs3, p + "hs3", dump, ctx)
        fails += _check(p + "ahss3", ac.ahss3, p + "ahss3", dump, ctx)
        fails += _check(p + "h1_v", ac.h1_v, p + "h1_v", dump, ctx)
        fails += _check(p + "h1_a", ac.h1_a, p + "h1_a", dump, ctx)

    print("")
    if fails == 0:
        print("LTX2 AV stack forward parity PASS: head+2blocks+tail cos>=", COS_GATE,
              "; all 16 acts/block match the contract")
    else:
        raise Error(String("LTX2 AV stack forward parity FAIL: ") + String(fails) + " checks below gate")
