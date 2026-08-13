# serenitymojo/models/wan22/parity/wan22_stack_devnative_parity.mojo
#
# SAME-PROCESS BIT GATE for the Phase-1b WAN_DEVNATIVE loader-free block-STACK:
#   (1) host recompute == host save-all      (recompute-discipline gate)
#   (2) device-native  == host recompute     (device tape gate)
# on a reduced-depth (3-block) synthetic Wan2.2 T2V stack. Wan is math-mode /
# deterministic, so both comparisons must be BIT-EQUAL (n_mismatch=0) on d_x_tokens
# and all 3*10 adapters' d_A/d_B. A teeth check proves the comparator bites.
#
# The 3 blocks reuse the tiny torch-dumped block weights (lin_*.bin from the block
# LoRA parity gate) — identical weights per block (a determinism gate, not a torch-
# value gate), but each block gets its OWN nonzero-A / nonzero-B LoRA (so d_A is
# non-degenerate and the two backward paths exercise distinct per-block grads).
#
# Run:
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/wan22/parity/wan22_stack_devnative_parity.mojo \
#     -o /tmp/wan22_stack_devnative_parity
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/wan22_stack_devnative_parity

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanModVecs,
)
from serenitymojo.models.wan22.wan22_stack_lora import (
    Wan22LoraSet, Wan22LoraGradSet, Wan22LoraDeviceGradSet,
    Wan22BlockStackForward,
    wan22_blockstack_lora_forward, wan22_blockstack_lora_backward,
    wan22_blockstack_lora_backward_devnative,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/wan22/parity/"

comptime H = 24
comptime Dh = 8
comptime DIM = H * Dh        # 192
comptime S = 5
comptime TXT = 4
comptime FFN = 40
comptime EPS = Float32(1e-06)
comptime RANK = 4
comptime LSCALE = Float32(1.0)
comptime NBLK = 3


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the LoRA oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read_bin_f32(REF_DIR + name + ".bin")


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _load_weights(ctx: DeviceContext) raises -> WanBlockWeights:
    return WanBlockWeights(
        _in("lin_sa_wq"), _in("lin_sa_wk"), _in("lin_sa_wv"), _in("lin_sa_wo"),
        _in("lin_sa_bq"), _in("lin_sa_bk"), _in("lin_sa_bv"), _in("lin_sa_bo"),
        _in("lin_sa_qn"), _in("lin_sa_kn"),
        _in("lin_ca_wq"), _in("lin_ca_wk"), _in("lin_ca_wv"), _in("lin_ca_wo"),
        _in("lin_ca_bq"), _in("lin_ca_bk"), _in("lin_ca_bv"), _in("lin_ca_bo"),
        _in("lin_ca_qn"), _in("lin_ca_kn"),
        _in("lin_n3_w"), _in("lin_n3_b"),
        _in("lin_ffn0_w"), _in("lin_ffn0_b"), _in("lin_ffn2_w"), _in("lin_ffn2_b"),
        DIM, FFN, Dh, ctx,
    )


def _load_mod() raises -> WanModVecs:
    return WanModVecs(
        _in("lin_shift_sa"), _in("lin_scale_sa"), _in("lin_gate_sa"),
        _in("lin_shift_ffn"), _in("lin_scale_ffn"), _in("lin_gate_ffn"),
    )


def _make_adapter(
    var a: List[Float32], var b: List[Float32], in_f: Int, out_f: Int,
) -> LoraAdapter:
    return LoraAdapter(
        a^, b^, RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


# Build a NBLK-block Wan22LoraSet with distinct nonzero A/B per (block, slot).
def _build_lora_set() -> Wan22LoraSet:
    var ad = List[LoraAdapter]()
    for bi in range(NBLK):
        var base = UInt64(1000) + UInt64(bi) * UInt64(100)
        # 8 attention adapters (DIM x DIM)
        for slot in range(8):
            var sa = base + UInt64(slot) * UInt64(2)
            ad.append(_make_adapter(
                _randn(RANK * DIM, sa, 0.07),
                _randn(DIM * RANK, sa + UInt64(1), 0.05),
                DIM, DIM,
            ))
        # ffn.0 (DIM -> FFN)
        ad.append(_make_adapter(
            _randn(RANK * DIM, base + UInt64(50), 0.06),
            _randn(FFN * RANK, base + UInt64(51), 0.04),
            DIM, FFN,
        ))
        # ffn.2 (FFN -> DIM)
        ad.append(_make_adapter(
            _randn(RANK * FFN, base + UInt64(60), 0.06),
            _randn(DIM * RANK, base + UInt64(61), 0.04),
            FFN, DIM,
        ))
    return Wan22LoraSet(ad^, NBLK, RANK)


def _n_mismatch(a: List[Float32], b: List[Float32]) -> Int:
    if len(a) != len(b):
        return -1
    var m = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            m += 1
    return m


def _check(name: String, got: List[Float32], want: List[Float32], mut allok: Bool):
    var nm = _n_mismatch(got, want)
    var verdict = "PASS" if nm == 0 else "FAIL"
    if nm != 0:
        allok = False
    print("  ", name, ": n=", len(want), " n_mismatch=", nm, "  ", verdict)


def main() raises:
    var ctx = DeviceContext()
    print("==== wan22_stack_devnative_parity (recompute==save-all, device==host) ====")
    print("NBLK=", NBLK, " DIM=", DIM, " S=", S, " TXT=", TXT, " FFN=", FFN, " RANK=", RANK)

    var sequence = _in("lin_x")
    var context = _in("lin_context")
    var cos = Tensor.from_host(_in("lin_cos"), [S, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("lin_sin"), [S, Dh // 2], STDtype.F32, ctx)
    var d_out = _in("lin_d_out")

    var weights = List[WanBlockWeights]()
    var modvecs = List[WanModVecs]()
    for _ in range(NBLK):
        weights.append(_load_weights(ctx))
        modvecs.append(_load_mod())
    var lora = _build_lora_set()

    var fwd = wan22_blockstack_lora_forward[H, Dh, S, TXT](
        sequence.copy(), modvecs, context.copy(), weights, lora, cos, sin,
        DIM, FFN, EPS, ctx,
    )

    # host save-all (recompute=False)
    var g_save = wan22_blockstack_lora_backward[H, Dh, S, TXT](
        d_out.copy(), modvecs, context.copy(), weights, lora, cos, sin, fwd,
        DIM, FFN, EPS, ctx, recompute=False,
    )
    # host recompute (recompute=True)
    var g_rc = wan22_blockstack_lora_backward[H, Dh, S, TXT](
        d_out.copy(), modvecs, context.copy(), weights, lora, cos, sin, fwd,
        DIM, FFN, EPS, ctx, recompute=True,
    )
    # device-native recompute
    var gd = wan22_blockstack_lora_backward_devnative[H, Dh, S, TXT](
        d_out.copy(), modvecs, context.copy(), weights, lora, cos, sin, fwd,
        DIM, FFN, EPS, ctx,
    )

    var n_adapters = NBLK * 10
    var allok1 = True
    print("")
    print("---- GATE 1: host recompute == host save-all (BIT) ----")
    _check("d_x_tokens", g_rc.d_x_tokens, g_save.d_x_tokens, allok1)
    for i in range(n_adapters):
        _check(String("dA[") + String(i) + "]", g_rc.d_a[i], g_save.d_a[i], allok1)
        _check(String("dB[") + String(i) + "]", g_rc.d_b[i], g_save.d_b[i], allok1)

    var allok2 = True
    print("")
    print("---- GATE 2: device-native == host recompute (to_host, BIT) ----")
    _check("d_x_tokens", gd.d_x_tokens, g_rc.d_x_tokens, allok2)
    for i in range(n_adapters):
        _check(String("dA[") + String(i) + "]", gd.d_a[i][].to_host(ctx), g_rc.d_a[i], allok2)
        _check(String("dB[") + String(i) + "]", gd.d_b[i][].to_host(ctx), g_rc.d_b[i], allok2)

    # teeth: d_x must be nonzero, and dA[0] must be nonzero (non-degenerate).
    var teeth_dx = _n_mismatch(g_rc.d_x_tokens, _zeros(len(g_rc.d_x_tokens)))
    var teeth_da = _n_mismatch(g_rc.d_a[0], _zeros(len(g_rc.d_a[0])))
    print("")
    print("teeth: d_x_tokens vs zeros n_mismatch=", teeth_dx, " (must be > 0)")
    print("teeth: dA[0] vs zeros    n_mismatch=", teeth_da, " (must be > 0)")

    var allok = allok1 and allok2 and (teeth_dx > 0) and (teeth_da > 0)
    print("")
    if allok:
        print("VERDICT: PASS — recompute==save-all AND device-native==host (BIT-EQUAL)")
    else:
        print("VERDICT: FAIL — a stack backward path diverged")
