# sd35_lora_all_slots_smoke.mojo -- shape/finite gate for every SD3.5 LoRA slot.
#
# This is not a torch numerical oracle. It verifies that the block API wires all
# eight per-block LoRA targets through forward/backward and returns populated
# d_A/d_B buffers for ctx/x qkv, proj, fc1, and fc2.

from std.collections import List, Optional
from std.gpu.host import DeviceContext

from serenitymojo.models.sd35.sd35_block import (
    JointBlockWeights,
    ModVecs,
    StreamLoraGrads,
    StreamWeights,
    sd35_joint_block_backward,
    sd35_joint_block_forward,
)
from serenitymojo.training.train_step import LoraAdapter


comptime H = 2
comptime Dh = 4
comptime D = H * Dh
comptime MLP = 12
comptime N_CTX = 2
comptime N_IMG = 3
comptime S = N_CTX + N_IMG
comptime RANK = 2


def _rand(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _ones(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(1.0))
    return out^


def _abs_sum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= Float32(0.0) else -x
    return s


def _nonfinite_count(v: List[Float32]) -> Int:
    var n = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            n += 1
    return n


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(String("sd35_lora_all_slots_smoke FAILED: ") + msg)


def _make_lora(in_f: Int, out_f: Int, seed: UInt64) -> LoraAdapter:
    var ma = _zeros(RANK * in_f)
    var va = _zeros(RANK * in_f)
    var mb = _zeros(out_f * RANK)
    var vb = _zeros(out_f * RANK)
    return LoraAdapter(
        _rand(RANK * in_f, seed, Float32(0.04)),
        _rand(out_f * RANK, seed + 1, Float32(0.04)),
        RANK,
        in_f,
        out_f,
        Float32(1.0) / Float32(RANK),
        ma^,
        va^,
        mb^,
        vb^,
    )


def _stream_weights(seed: UInt64) -> StreamWeights:
    return StreamWeights(
        _rand(3 * D * D, seed + 1, Float32(0.03)),
        _rand(3 * D, seed + 2, Float32(0.01)),
        _rand(D * D, seed + 3, Float32(0.03)),
        _rand(D, seed + 4, Float32(0.01)),
        _rand(MLP * D, seed + 5, Float32(0.03)),
        _rand(MLP, seed + 6, Float32(0.01)),
        _rand(D * MLP, seed + 7, Float32(0.03)),
        _rand(D, seed + 8, Float32(0.01)),
        _ones(Dh),
        _ones(Dh),
    )


def _mods(seed: UInt64) -> ModVecs:
    return ModVecs(
        _rand(D, seed + 1, Float32(0.03)),
        _rand(D, seed + 2, Float32(0.03)),
        _rand(D, seed + 3, Float32(0.30)),
        _rand(D, seed + 4, Float32(0.03)),
        _rand(D, seed + 5, Float32(0.03)),
        _rand(D, seed + 6, Float32(0.30)),
    )


def _check_pair(label: String, d_a: List[Float32], d_b: List[Float32], ad: LoraAdapter) raises:
    _require(len(d_a) == len(ad.a), label + String(" d_A len"))
    _require(len(d_b) == len(ad.b), label + String(" d_B len"))
    _require(_nonfinite_count(d_a) == 0, label + String(" d_A finite"))
    _require(_nonfinite_count(d_b) == 0, label + String(" d_B finite"))
    _require(_abs_sum(d_a) > Float32(0.0), label + String(" d_A nonzero"))
    _require(_abs_sum(d_b) > Float32(0.0), label + String(" d_B nonzero"))


def _check_stream(label: String, g: StreamLoraGrads, qkv: LoraAdapter, proj: LoraAdapter, fc1: LoraAdapter, fc2: LoraAdapter) raises:
    _check_pair(label + String(" qkv"), g.qkv_d_a, g.qkv_d_b, qkv)
    _check_pair(label + String(" proj"), g.proj_d_a, g.proj_d_b, proj)
    _check_pair(label + String(" fc1"), g.fc1_d_a, g.fc1_d_b, fc1)
    _check_pair(label + String(" fc2"), g.fc2_d_a, g.fc2_d_b, fc2)


def main() raises:
    var ctx = DeviceContext()
    var w = JointBlockWeights(_stream_weights(100), _stream_weights(200))
    var cm = _mods(300)
    var xm = _mods(400)
    var context = _rand(N_CTX * D, 500, Float32(0.10))
    var x = _rand(N_IMG * D, 600, Float32(0.10))
    var d_context = _rand(N_CTX * D, 700, Float32(0.03))
    var d_x = _rand(N_IMG * D, 800, Float32(0.03))

    var ctx_qkv = _make_lora(D, 3 * D, 1000)
    var ctx_proj = _make_lora(D, D, 1100)
    var ctx_fc1 = _make_lora(D, MLP, 1200)
    var ctx_fc2 = _make_lora(MLP, D, 1300)
    var x_qkv = _make_lora(D, 3 * D, 1400)
    var x_proj = _make_lora(D, D, 1500)
    var x_fc1 = _make_lora(D, MLP, 1600)
    var x_fc2 = _make_lora(MLP, D, 1700)

    var fwd = sd35_joint_block_forward[1, S, H, Dh](
        context,
        x,
        w,
        cm.copy(),
        xm.copy(),
        N_CTX,
        N_IMG,
        D,
        MLP,
        Float32(1.0e-6),
        Float32(1.0e-6),
        Float32(0.5),
        ctx,
        Optional[List[Float32]](None),
        Optional[LoraAdapter](ctx_qkv.copy()),
        Optional[LoraAdapter](ctx_proj.copy()),
        Optional[LoraAdapter](ctx_fc1.copy()),
        Optional[LoraAdapter](ctx_fc2.copy()),
        Optional[LoraAdapter](x_qkv.copy()),
        Optional[LoraAdapter](x_proj.copy()),
        Optional[LoraAdapter](x_fc1.copy()),
        Optional[LoraAdapter](x_fc2.copy()),
    )
    var g = sd35_joint_block_backward[1, S, H, Dh](
        d_context,
        d_x,
        w,
        cm,
        xm,
        fwd,
        N_CTX,
        N_IMG,
        D,
        MLP,
        Float32(1.0e-6),
        Float32(1.0e-6),
        Float32(0.5),
        ctx,
        Optional[LoraAdapter](ctx_qkv.copy()),
        Optional[LoraAdapter](ctx_proj.copy()),
        Optional[LoraAdapter](ctx_fc1.copy()),
        Optional[LoraAdapter](ctx_fc2.copy()),
        Optional[LoraAdapter](x_qkv.copy()),
        Optional[LoraAdapter](x_proj.copy()),
        Optional[LoraAdapter](x_fc1.copy()),
        Optional[LoraAdapter](x_fc2.copy()),
    )

    _check_stream(String("ctx"), g.ctx_lora, ctx_qkv, ctx_proj, ctx_fc1, ctx_fc2)
    _check_stream(String("x"), g.x_lora, x_qkv, x_proj, x_fc1, x_fc2)
    print("sd35_lora_all_slots_smoke PASS")
