# serenitymojo/models/ltx2/parity/ltx2_av_stack_bwd_parity.mojo
#
# LTX-2 joint-AV 2-block STACK BACKWARD parity gate (P6.2 part 5).
#
# Compares the Mojo AV stack backward (models/ltx2/ltx2_av_stack.mojo
# ltx2_av_stack_backward) against the torch.autograd oracle
# scripts/ltx2_av_stack_oracle.py --backward ->
#   output/ltx2_av_stack/av_stack2_bwd_ref.safetensors
#
# Reads the oracle's BYTE-IDENTICAL inputs (v_lat/a_lat/enc/aenc + rope + per-block
# LoRA + the tail cotangents d_video/d_audio), runs the Mojo FORWARD (saving acts),
# then the Mojo BACKWARD, and gates cos >= 0.999 PER grad: both stream input-grads
# (d_hidden/d_ahs) + all 24 LoRA d_A/d_B x N_BLOCKS. The per-block backward is
# already gated (ltx2_block_backward_av); this proves the COMPOSITION — tail-grad
# seeding + per-block d_x->d_y across BOTH streams + the head/adaln seam. F32.
#
# Prep: ${SERENITY_ORACLE_PYTHON:-python3} scripts/ltx2_av_stack_oracle.py --backward
# Run:  rm -f serenitymojo.mojopkg; pixi run mojo build -O2 -I . -Xlinker -lm \
#   -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_stack_bwd_parity.mojo \
#   -o /tmp/ltx2_av_stack_bwd_parity && /tmp/ltx2_av_stack_bwd_parity

from std.gpu.host import DeviceContext
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
    LTX2AVBlockSource, ltx2_av_build_head, ltx2_av_stack_forward, ltx2_av_stack_backward, _patchify, _av_lora_slots,
)

comptime CKPT_NAME = "ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
comptime REF_PATH = "ltx2_av_stack/av_stack2_bwd_ref.safetensors"
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


def _load_rope(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var t = Tensor.from_view_as_f32(st.tensor_view(name), ctx)
    ref sh = t.shape()
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


def _check(tag: String, mojo: List[Float32], ref_name: String, dump: ShardedSafeTensors, ctx: DeviceContext) raises -> Int:
    var r = _load(dump, ref_name, ctx).to_host(ctx)
    var c = _cos(mojo, r)
    var ok = c >= COS_GATE
    print("  ", "PASS" if ok else "FAIL", tag, "cos=", c)
    return 0 if ok else 1


def _mods() -> List[String]:
    var m = List[String]()
    m.append(String("attn1")); m.append(String("attn2"))
    m.append(String("audio_attn1")); m.append(String("audio_attn2"))
    m.append(String("audio_to_video_attn")); m.append(String("video_to_audio_attn"))
    return m^


def _projs() -> List[String]:
    var p = List[String]()
    p.append(String("to_q")); p.append(String("to_k")); p.append(String("to_v")); p.append(String("to_out.0"))
    return p^


def _ffns() -> List[String]:
    # P6.2 (a): the 4 FFN LoRA slots (indices 24-27), matching _av_lora_slots order.
    var f = List[String]()
    f.append(String("audio_ff.net.0.proj")); f.append(String("audio_ff.net.2"))
    f.append(String("ff.net.0.proj")); f.append(String("ff.net.2"))
    return f^


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var ckpt = env_or("LTX2_AV_CKPT", serenity_checkpoint(String(CKPT_NAME)))
    var ref_path = env_or("LTX2_AV_BWD_REF", serenity_output(String(REF_PATH)))
    print("=== LTX-2 joint-AV 2-block STACK BACKWARD parity gate ===")
    print("  S_V/S_A/N_TXT:", S_V, S_A, N_TXT, " blocks:", N_BLOCKS, " sigma:", SIGMA)
    print("  oracle:", ref_path)

    var dump = ShardedSafeTensors.open(ref_path)
    var v_lat = _load(dump, "v_lat", ctx)
    var a_lat = _load(dump, "a_lat", ctx)
    var enc = _load(dump, "enc", ctx)
    var aenc = _load(dump, "aenc", ctx)
    var d_video = _load(dump, "d_video", ctx)
    var d_audio = _load(dump, "d_audio", ctx)

    var v_cos = _load_rope(dump, "v_cos", ctx)
    var v_sin = _load_rope(dump, "v_sin", ctx)
    var a_cos = _load_rope(dump, "a_cos", ctx)
    var a_sin = _load_rope(dump, "a_sin", ctx)
    var ca_v_cos = _load_rope(dump, "ca_v_cos", ctx)
    var ca_v_sin = _load_rope(dump, "ca_v_sin", ctx)
    var ca_a_cos = _load_rope(dump, "ca_a_cos", ctx)
    var ca_a_sin = _load_rope(dump, "ca_a_sin", ctx)

    var slots = _av_lora_slots()
    var lora_a = List[ArcPointer[Tensor]]()
    var lora_b = List[ArcPointer[Tensor]]()
    for i in range(N_BLOCKS):
        for s in range(len(slots)):
            lora_a.append(ArcPointer[Tensor](_load(dump, String("b") + String(i) + ".lora." + slots[s] + ".A", ctx)))
            lora_b.append(ArcPointer[Tensor](_load(dump, String("b") + String(i) + ".lora." + slots[s] + ".B", ctx)))

    var st = ShardedSafeTensors.open(ckpt)

    # ── residency A/B (bit-exactness gate for the fp8-resident block store) ──
    # LTX2_AV_BLOCK_CKPT overrides ONLY the BLOCK-weight source; head/patchify
    # (st), rope, LoRA and the oracle refs stay on CKPT. That isolates the single
    # variable residency actually touches, so an unchanged digit set is proof
    # about the block store and nothing else. LTX2_RESIDENT_BLOCKS>0 parks
    # blocks 0..N-1 in VRAM (needs the fp8 file; blocks 0..1 here are BF16
    # boundary blocks, still a real exercise of the resident dispatch).
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
    var head = ltx2_av_build_head[S_V, S_A, N_TXT](st, SIGMA, ctx)
    var hidden = _patchify(v_lat, st, String("patchify_proj"), ctx)
    var ahs = _patchify(a_lat, st, String("audio_patchify_proj"), ctx)

    # FORWARD then BACKWARD. LTX2_SAVE_ACTS selects the conductor arm: 0 (default)
    # = recompute every block from its saved inputs; K>0 = retain blocks [0,K)'s
    # acts and skip their recompute. BOTH arms must produce identical grads — that
    # equivalence is the contract, so this gate is run at K=0 and K=N_BLOCKS.
    var SAVE_ACTS_K = env_int("LTX2_SAVE_ACTS", 0)
    if SAVE_ACTS_K > N_BLOCKS:
        SAVE_ACTS_K = N_BLOCKS
    print("  [forward] head +", N_BLOCKS, "blocks + tail (save_acts_k=", SAVE_ACTS_K, ")")
    var fwd = ltx2_av_stack_forward[S_V, S_A, N_TXT](
        st, src, hidden, ahs, enc, aenc, head,
        v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
        lora_a, lora_b, LORA_SCALE, N_BLOCKS, EPS, ctx, SAVE_ACTS_K,
    )
    var v_last = fwd.v_last.clone(ctx)
    var a_last = fwd.a_last.clone(ctx)
    print("  [backward] tail + reverse block loop (SAVE_ACTS_K=", SAVE_ACTS_K, ")")
    var grads = ltx2_av_stack_backward[S_V, S_A, N_TXT](
        d_video, d_audio, fwd.saved_v_in, fwd.saved_a_in, v_last, a_last,
        enc, aenc, head,
        v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
        st, src, lora_a, lora_b, LORA_SCALE, N_BLOCKS, EPS, ctx,
        SAVE_ACTS_K, fwd.saved,
    )

    var fails = 0
    fails += _check(String("d_hidden (block-0 video input grad)"), grads.d_hidden, String("g.d_hidden"), dump, ctx)
    fails += _check(String("d_ahs (block-0 audio input grad)"), grads.d_ahs, String("g.d_ahs"), dump, ctx)
    if grads.nonfinite != 0:
        print("   FAIL nonfinite grads:", grads.nonfinite); fails += 1

    var mods = _mods()
    var projs = _projs()
    for i in range(N_BLOCKS):
        for mi in range(len(mods)):
            for pi in range(len(projs)):
                var s = mi * len(projs) + pi
                var idx = i * len(slots) + s
                var pfx = String("b") + String(i) + "." + mods[mi] + "." + projs[pi]
                fails += _check(pfx + ".dA", grads.d_a[idx],
                                String("g.b") + String(i) + ".dA." + mods[mi] + "." + projs[pi], dump, ctx)
                fails += _check(pfx + ".dB", grads.d_b[idx],
                                String("g.b") + String(i) + ".dB." + mods[mi] + "." + projs[pi], dump, ctx)

    # FFN LoRA pairs (P6.2 (a)): slots 24-27 (audio_ff.net.0.proj/net.2 + ff.net.0.proj/net.2).
    var ffns = _ffns()
    for i in range(N_BLOCKS):
        for fi in range(len(ffns)):
            var s = len(mods) * len(projs) + fi
            var idx = i * len(slots) + s
            var pfx = String("b") + String(i) + "." + ffns[fi]
            fails += _check(pfx + ".dA", grads.d_a[idx],
                            String("g.b") + String(i) + ".dA." + ffns[fi], dump, ctx)
            fails += _check(pfx + ".dB", grads.d_b[idx],
                            String("g.b") + String(i) + ".dB." + ffns[fi], dump, ctx)

    print("")
    if fails == 0:
        print("LTX2 AV stack BACKWARD parity PASS: composition grads cos >=", COS_GATE,
              "(both stream input-grads + all 24 LoRA pairs x", N_BLOCKS, "blocks)")
    else:
        raise Error(String("LTX2 AV stack backward parity FAIL: ") + String(fails) + " grads below gate")
