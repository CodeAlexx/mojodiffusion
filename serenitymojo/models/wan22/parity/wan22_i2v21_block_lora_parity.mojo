# serenitymojo/models/wan22/parity/wan22_i2v21_block_lora_parity.mojo
#
# TORCHREF-TUNER PARITY GATE for the Wan2.1 I2V-14B WanAttentionBlock (dual
# cross-attention, WanI2VCrossAttention) LoRA training unit
# (models/wan22/wan22_block.mojo wan22_i2v_block_lora_*). Consumes the dump from
# wan22_i2v21_block_lora_torchref_oracle.py (reference driven through Torchref's REAL
# WanAttentionBlock.forward at real 14B dims). Compares x_out, d_x, d_context
# (IMG + TXT), and the 12 adapters' d_A/d_B against the Torchref reference.
#
# Core outputs (x_out, d_x, d_context) gate at cos>=0.999 (strict). LoRA d_A/d_B
# gate at cos>=0.995 (the SAME bf16 threshold the certified base gate
# wan22_block_lora_parity_torchref.mojo uses — native bf16 rank-r factor grads
# carry more rounding than full activations). Raw cos is printed for all.
#
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/torchref-image/venv/bin/python \
#       serenitymojo/models/wan22/parity/wan22_i2v21_block_lora_torchref_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/wan22/parity/wan22_i2v21_block_lora_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanModVecs, WanBlockLora,
    WanI2VBlockWeights, WanI2VBlockLora,
    wan22_i2v_block_lora_forward, wan22_i2v_block_lora_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/wan22/parity/"

# Real Wan2.1 I2V-14B block dims, small sequence, IMG=257 (torchref fixed).
comptime H = 40
comptime Dh = 128
comptime DIM = H * Dh          # 5120
comptime S = 16
comptime TXT = 8
comptime IMG = 257
comptime FFN = 13824
comptime EPS = Float32(1e-06)
comptime RANK = 8
comptime LSCALE = Float32(1.0)   # alpha/rank = 8/8


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
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


def _load_base_weights(ctx: DeviceContext) raises -> WanBlockWeights:
    return WanBlockWeights(
        _in("i2vin_sa_wq"), _in("i2vin_sa_wk"), _in("i2vin_sa_wv"), _in("i2vin_sa_wo"),
        _in("i2vin_sa_bq"), _in("i2vin_sa_bk"), _in("i2vin_sa_bv"), _in("i2vin_sa_bo"),
        _in("i2vin_sa_qn"), _in("i2vin_sa_kn"),
        _in("i2vin_ca_wq"), _in("i2vin_ca_wk"), _in("i2vin_ca_wv"), _in("i2vin_ca_wo"),
        _in("i2vin_ca_bq"), _in("i2vin_ca_bk"), _in("i2vin_ca_bv"), _in("i2vin_ca_bo"),
        _in("i2vin_ca_qn"), _in("i2vin_ca_kn"),
        _in("i2vin_n3_w"), _in("i2vin_n3_b"),
        _in("i2vin_ffn0_w"), _in("i2vin_ffn0_b"), _in("i2vin_ffn2_w"), _in("i2vin_ffn2_b"),
        DIM, FFN, Dh, ctx,
    )


def _load_weights(ctx: DeviceContext) raises -> WanI2VBlockWeights:
    var base = _load_base_weights(ctx)
    return WanI2VBlockWeights(
        base^,
        _in("i2vin_ca_wk_img"), _in("i2vin_ca_bk_img"),
        _in("i2vin_ca_wv_img"), _in("i2vin_ca_bv_img"),
        _in("i2vin_ca_kn_img"),
        DIM, ctx,
    )


def _load_mod() raises -> WanModVecs:
    return WanModVecs(
        _in("i2vin_shift_sa"), _in("i2vin_scale_sa"), _in("i2vin_gate_sa"),
        _in("i2vin_shift_ffn"), _in("i2vin_scale_ffn"), _in("i2vin_gate_ffn"),
    )


def _make_adapter(a: List[Float32], b: List[Float32], in_f: Int, out_f: Int) -> LoraAdapter:
    return LoraAdapter(
        a.copy(), b.copy(), RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _adapter(name: String) raises -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](
        _make_adapter(_in("i2vin_" + name + "_A"), _in("i2vin_" + name + "_B"), DIM, DIM)
    )


def _adapter_shaped(name: String, in_f: Int, out_f: Int) raises -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](
        _make_adapter(_in("i2vin_" + name + "_A"), _in("i2vin_" + name + "_B"), in_f, out_f)
    )


def _load_lora() raises -> WanI2VBlockLora:
    var base = WanBlockLora(
        _adapter("sa_q"), _adapter("sa_k"), _adapter("sa_v"), _adapter("sa_o"),
        _adapter("ca_q"), _adapter("ca_k"), _adapter("ca_v"), _adapter("ca_o"),
        _adapter_shaped("ffn0", DIM, FFN),
        _adapter_shaped("ffn2", FFN, DIM),
    )
    return WanI2VBlockLora(
        base^,
        _adapter("img_k"),   # k_img: dim->dim
        _adapter("img_v"),   # v_img: dim->dim
    )


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def _check_lora(
    name: String, actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var h = ParityHarness(0.995)
    var r = h.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL", "(bf16 gate 0.995)")
    if not r.passed:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== wan22_i2v21_block_lora_parity (Wan2.1 I2V-14B DUAL cross-attn + 12 LoRA vs Torchref) ====")
    print("H=", H, " Dh=", Dh, " DIM=", DIM, " S=", S, " TXT=", TXT, " IMG=", IMG, " FFN=", FFN, " RANK=", RANK)

    var x = _in("i2vin_x")
    var context_txt = _in("i2vin_context_txt")
    var context_img = _in("i2vin_context_img")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var lora = _load_lora()
    var cos = Tensor.from_host(_in("i2vin_cos"), [S, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("i2vin_sin"), [S, Dh // 2], STDtype.F32, ctx)

    var fwd = wan22_i2v_block_lora_forward[H, Dh, S, TXT, IMG](
        x.copy(), context_txt.copy(), context_img.copy(), mv, w, lora,
        cos, sin, DIM, FFN, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs Torchref ----")
    _check(harness, "x_out", fwd.x_out, _in("i2vref_x_out"), allok)

    var d_out = _in("i2vin_d_out")
    var g = wan22_i2v_block_lora_backward[H, Dh, S, TXT, IMG](
        d_out, mv, w, lora, fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )

    print("")
    print("---- input grads vs Torchref (incl LoRA branch) ----")
    _check(harness, "d_x (img latent) ", g.base.base.d_x, _in("i2vref_d_x"), allok)
    _check(harness, "d_context_txt    ", g.base.base.d_context, _in("i2vref_d_context_txt"), allok)
    _check(harness, "d_context_img    ", g.d_context_img, _in("i2vref_d_context_img"), allok)

    print("")
    print("---- LoRA d_A / d_B vs Torchref (12 adapters; bf16 gate 0.995) ----")
    _check_lora("sa_q dA", g.base.sa_q_da, _in("i2vref_sa_q_dA"), allok)
    _check_lora("sa_q dB", g.base.sa_q_db, _in("i2vref_sa_q_dB"), allok)
    _check_lora("sa_k dA", g.base.sa_k_da, _in("i2vref_sa_k_dA"), allok)
    _check_lora("sa_k dB", g.base.sa_k_db, _in("i2vref_sa_k_dB"), allok)
    _check_lora("sa_v dA", g.base.sa_v_da, _in("i2vref_sa_v_dA"), allok)
    _check_lora("sa_v dB", g.base.sa_v_db, _in("i2vref_sa_v_dB"), allok)
    _check_lora("sa_o dA", g.base.sa_o_da, _in("i2vref_sa_o_dA"), allok)
    _check_lora("sa_o dB", g.base.sa_o_db, _in("i2vref_sa_o_dB"), allok)
    _check_lora("ca_q dA", g.base.ca_q_da, _in("i2vref_ca_q_dA"), allok)
    _check_lora("ca_q dB", g.base.ca_q_db, _in("i2vref_ca_q_dB"), allok)
    _check_lora("ca_k dA", g.base.ca_k_da, _in("i2vref_ca_k_dA"), allok)
    _check_lora("ca_k dB", g.base.ca_k_db, _in("i2vref_ca_k_dB"), allok)
    _check_lora("ca_v dA", g.base.ca_v_da, _in("i2vref_ca_v_dA"), allok)
    _check_lora("ca_v dB", g.base.ca_v_db, _in("i2vref_ca_v_dB"), allok)
    _check_lora("ca_o dA", g.base.ca_o_da, _in("i2vref_ca_o_dA"), allok)
    _check_lora("ca_o dB", g.base.ca_o_db, _in("i2vref_ca_o_dB"), allok)
    _check_lora("ffn0 dA", g.base.ffn0_da, _in("i2vref_ffn0_dA"), allok)
    _check_lora("ffn0 dB", g.base.ffn0_db, _in("i2vref_ffn0_dB"), allok)
    _check_lora("ffn2 dA", g.base.ffn2_da, _in("i2vref_ffn2_dA"), allok)
    _check_lora("ffn2 dB", g.base.ffn2_db, _in("i2vref_ffn2_dB"), allok)
    print("  -- NEW image-branch cross-attn adapters (the genuinely new compute) --")
    _check_lora("img_k dA", g.img_k_da, _in("i2vref_img_k_dA"), allok)
    _check_lora("img_k dB", g.img_k_db, _in("i2vref_img_k_dB"), allok)
    _check_lora("img_v dA", g.img_v_da, _in("i2vref_img_v_dA"), allok)
    _check_lora("img_v dB", g.img_v_db, _in("i2vref_img_v_dB"), allok)

    print("")
    if allok:
        print("VERDICT: PASS -- Wan2.1 I2V-14B dual-cross-attn block LoRA fwd+bwd matches Torchref")
    else:
        print("VERDICT: FAIL -- at least one output diverged (see FAIL lines above)")
