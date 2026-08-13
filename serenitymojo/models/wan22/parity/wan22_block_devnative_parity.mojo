# serenitymojo/models/wan22/parity/wan22_block_devnative_parity.mojo
#
# SAME-PROCESS BIT GATE for the Phase-1 WAN_DEVNATIVE device-native block
# backward (wan22_block_lora_backward_devnative) vs the host hand-chain oracle
# (wan22_block_lora_backward). Wan is math-mode / deterministic, so the device
# grads (to_host) MUST be BIT-EQUAL (n_mismatch=0) to the host grads on d_x,
# d_context, and all 10 adapters' d_A/d_B. A teeth check proves the comparator
# detects a real difference (d_x vs zeros -> n_mismatch > 0).
#
# Reuses the tiny torch-dumped block inputs (lin_*.bin) from the existing LoRA
# parity gate for the 8 attention adapters (nonzero B). ffn.0/ffn.2 have no tiny-
# config torch bins, so they are synthesized in-process with NONZERO randn A/B —
# valid because this is a same-process oracle-vs-devnative determinism gate (both
# paths consume the identical adapters), and it exercises all 10 slots.
#
# Run:
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/wan22/parity/wan22_block_devnative_parity.mojo \
#     -o /tmp/wan22_block_devnative_parity
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/wan22_block_devnative_parity

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanModVecs, WanBlockLora,
    wan22_block_lora_forward, wan22_block_lora_backward,
    wan22_block_lora_backward_devnative, WanBlockLoraDeviceGrads,
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
comptime LSCALE = Float32(1.0)   # alpha/rank = 4/4


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


# Deterministic small randn-ish generator (LCG) for the synthetic ffn adapters.
def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0)  # [0,1)
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


def _attn_adapter(name: String) raises -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](
        _make_adapter(_in("lin_" + name + "_A"), _in("lin_" + name + "_B"), DIM, DIM)
    )


# Synthetic nonzero A/B (so d_A is non-degenerate) for ffn.0 / ffn.2.
def _ffn_adapter(seed: UInt64, in_f: Int, out_f: Int) -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](
        _make_adapter(
            _randn(RANK * in_f, seed, 0.08),
            _randn(out_f * RANK, seed + UInt64(777), 0.05),
            in_f, out_f,
        )
    )


def _load_lora() raises -> WanBlockLora:
    return WanBlockLora(
        _attn_adapter("sa_q"), _attn_adapter("sa_k"), _attn_adapter("sa_v"), _attn_adapter("sa_o"),
        _attn_adapter("ca_q"), _attn_adapter("ca_k"), _attn_adapter("ca_v"), _attn_adapter("ca_o"),
        _ffn_adapter(UInt64(11), DIM, FFN), _ffn_adapter(UInt64(29), FFN, DIM),
    )


def _n_mismatch(a: List[Float32], b: List[Float32]) -> Int:
    if len(a) != len(b):
        return -1
    var m = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            m += 1
    return m


def _check(
    name: String, dev: List[Float32], host: List[Float32], mut allok: Bool,
):
    var nm = _n_mismatch(dev, host)
    var verdict = "PASS" if nm == 0 else "FAIL"
    if nm != 0:
        allok = False
    print("  ", name, ": n=", len(host), " n_mismatch=", nm, "  ", verdict)


def _slot(g: WanBlockLoraDeviceGrads, i: Int, ctx: DeviceContext) raises -> List[Float32]:
    if g.d_a[i]:
        return g.d_a[i].value()[].to_host(ctx)
    return List[Float32]()


def _slot_b(g: WanBlockLoraDeviceGrads, i: Int, ctx: DeviceContext) raises -> List[Float32]:
    if g.d_b[i]:
        return g.d_b[i].value()[].to_host(ctx)
    return List[Float32]()


def main() raises:
    var ctx = DeviceContext()
    print("==== wan22_block_devnative_parity (device-native == host hand-chain, BIT) ====")
    print("H=", H, " Dh=", Dh, " DIM=", DIM, " S=", S, " TXT=", TXT, " FFN=", FFN, " RANK=", RANK)

    var x = _in("lin_x")
    var context = _in("lin_context")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var lora = _load_lora()
    var cos = Tensor.from_host(_in("lin_cos"), [S, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("lin_sin"), [S, Dh // 2], STDtype.F32, ctx)

    var fwd = wan22_block_lora_forward[H, Dh, S, TXT](
        x.copy(), context.copy(), mv, w, lora, cos, sin, DIM, FFN, EPS, ctx,
    )

    var d_out = _in("lin_d_out")

    # Host hand-chain oracle.
    var gh = wan22_block_lora_backward[H, Dh, S, TXT](
        d_out.copy(), mv, w, lora, fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )

    # Device-native path (same saved, same d_out uploaded as BF16 exactly like the
    # oracle's _ta16).
    var d_out_dev = Tensor.from_host(d_out.copy(), [S, DIM], STDtype.BF16, ctx)
    var gd = wan22_block_lora_backward_devnative[H, Dh, S, TXT](
        d_out_dev, mv, w, lora, fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )

    var allok = True
    print("")
    print("---- input grads (device-native to_host vs host, BIT-EQUAL) ----")
    _check("d_x (img)   ", gd.d_x.to_host(ctx), gh.base.d_x, allok)
    _check("d_context   ", gd.d_context.to_host(ctx), gh.base.d_context, allok)

    print("")
    print("---- 10 adapters d_A / d_B (BIT-EQUAL) ----")
    _check("sa_q dA", _slot(gd, 0, ctx), gh.sa_q_da, allok)
    _check("sa_q dB", _slot_b(gd, 0, ctx), gh.sa_q_db, allok)
    _check("sa_k dA", _slot(gd, 1, ctx), gh.sa_k_da, allok)
    _check("sa_k dB", _slot_b(gd, 1, ctx), gh.sa_k_db, allok)
    _check("sa_v dA", _slot(gd, 2, ctx), gh.sa_v_da, allok)
    _check("sa_v dB", _slot_b(gd, 2, ctx), gh.sa_v_db, allok)
    _check("sa_o dA", _slot(gd, 3, ctx), gh.sa_o_da, allok)
    _check("sa_o dB", _slot_b(gd, 3, ctx), gh.sa_o_db, allok)
    _check("ca_q dA", _slot(gd, 4, ctx), gh.ca_q_da, allok)
    _check("ca_q dB", _slot_b(gd, 4, ctx), gh.ca_q_db, allok)
    _check("ca_k dA", _slot(gd, 5, ctx), gh.ca_k_da, allok)
    _check("ca_k dB", _slot_b(gd, 5, ctx), gh.ca_k_db, allok)
    _check("ca_v dA", _slot(gd, 6, ctx), gh.ca_v_da, allok)
    _check("ca_v dB", _slot_b(gd, 6, ctx), gh.ca_v_db, allok)
    _check("ca_o dA", _slot(gd, 7, ctx), gh.ca_o_da, allok)
    _check("ca_o dB", _slot_b(gd, 7, ctx), gh.ca_o_db, allok)
    _check("ffn0 dA", _slot(gd, 8, ctx), gh.ffn0_da, allok)
    _check("ffn0 dB", _slot_b(gd, 8, ctx), gh.ffn0_db, allok)
    _check("ffn2 dA", _slot(gd, 9, ctx), gh.ffn2_da, allok)
    _check("ffn2 dB", _slot_b(gd, 9, ctx), gh.ffn2_db, allok)

    # Teeth: the comparator must detect a real difference, and d_x must be nonzero.
    var teeth = _n_mismatch(gh.base.d_x, _zeros(len(gh.base.d_x)))
    print("")
    print("teeth: d_x vs zeros n_mismatch=", teeth, " (must be > 0)")
    if teeth <= 0:
        allok = False

    print("")
    if allok:
        print("VERDICT: PASS — device-native block backward BIT-EQUAL to host hand-chain")
    else:
        print("VERDICT: FAIL — device-native diverged from host hand-chain")
