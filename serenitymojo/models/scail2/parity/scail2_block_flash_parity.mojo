# serenitymojo/models/scail2/parity/scail2_block_flash_parity.mojo
#
# G1 FLASH PARITY GATE (CHUNK 2c) for the SCAIL-2 i2v block LoRA training unit
# with the FLASH self-attention path enabled (scail2_block_lora_{forward,backward}
# instantiated with FLASH=True). Reuses the SAME chunk-1 torch-autograd dump from
# scail2_block_lora_oracle.py (dense math attention reference); the flash path is
# a different softmax summation order but the SAME attention, so it must still
# match at cos>=0.999. Compares x_out, d_x, d_context (IMG + TXT), and all 12
# adapters' d_A/d_B.
#
# ALL comparisons gate at cos>=0.999 (forward, d_x, and EVERY LoRA d_A/d_B).
# NOTE: flash bwd dQ is nondeterministic run-to-run (~1e-4) -> d_x / sa_q / sa_k
# grads wobble; this is a VALUE-class gate. Run the binary TWICE; both runs must
# stay >=0.999 (report the worst cos across the two runs).
#
# Flash needs cuDNN-shim + CUDA linkage -> build a BINARY (JIT can't cuInit):
#   /home/alex/torchref-image/venv/bin/python \
#       serenitymojo/models/scail2/parity/scail2_block_lora_oracle.py   # oracle FIRST
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 -I . \
#     -Xlinker -lm -Xlinker -ldl \
#     -Xlinker -L"$CONDA_PREFIX/targets/x86_64-linux/lib/stubs" -Xlinker -lcuda \
#     -Xlinker -L"$CONDA_PREFIX/lib" -Xlinker -rpath-link -Xlinker "$CONDA_PREFIX/lib" \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/scail2/parity/scail2_block_flash_parity.mojo \
#     -o output/bin/scail2_block_flash_parity
#   LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib:.pixi/envs/default/lib \
#     output/bin/scail2_block_flash_parity

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
)
from serenitymojo.models.scail2.scail2_block_train import (
    scail2_block_lora_forward, scail2_block_lora_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/scail2/parity/"

# Reduced SCAIL-2 i2v block dims (DH real = 128); IMG=257 fixed by torchref split.
comptime H = 4
comptime DH = 128
comptime DIM = H * DH           # 512
comptime S = 9
comptime TXT = 16
comptime IMG = 257
comptime FFN = 1024
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
        _in("s2in_sa_wq"), _in("s2in_sa_wk"), _in("s2in_sa_wv"), _in("s2in_sa_wo"),
        _in("s2in_sa_bq"), _in("s2in_sa_bk"), _in("s2in_sa_bv"), _in("s2in_sa_bo"),
        _in("s2in_sa_qn"), _in("s2in_sa_kn"),
        _in("s2in_ca_wq"), _in("s2in_ca_wk"), _in("s2in_ca_wv"), _in("s2in_ca_wo"),
        _in("s2in_ca_bq"), _in("s2in_ca_bk"), _in("s2in_ca_bv"), _in("s2in_ca_bo"),
        _in("s2in_ca_qn"), _in("s2in_ca_kn"),
        _in("s2in_n3_w"), _in("s2in_n3_b"),
        _in("s2in_ffn0_w"), _in("s2in_ffn0_b"), _in("s2in_ffn2_w"), _in("s2in_ffn2_b"),
        DIM, FFN, DH, ctx,
    )


def _load_weights(ctx: DeviceContext) raises -> WanI2VBlockWeights:
    var base = _load_base_weights(ctx)
    return WanI2VBlockWeights(
        base^,
        _in("s2in_ca_wk_img"), _in("s2in_ca_bk_img"),
        _in("s2in_ca_wv_img"), _in("s2in_ca_bv_img"),
        _in("s2in_ca_kn_img"),
        DIM, ctx,
    )


def _load_mod() raises -> WanModVecs:
    return WanModVecs(
        _in("s2in_shift_sa"), _in("s2in_scale_sa"), _in("s2in_gate_sa"),
        _in("s2in_shift_ffn"), _in("s2in_scale_ffn"), _in("s2in_gate_ffn"),
    )


def _make_adapter(a: List[Float32], b: List[Float32], in_f: Int, out_f: Int) -> LoraAdapter:
    return LoraAdapter(
        a.copy(), b.copy(), RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _adapter(name: String) raises -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](
        _make_adapter(_in("s2in_" + name + "_A"), _in("s2in_" + name + "_B"), DIM, DIM)
    )


def _adapter_shaped(name: String, in_f: Int, out_f: Int) raises -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](
        _make_adapter(_in("s2in_" + name + "_A"), _in("s2in_" + name + "_B"), in_f, out_f)
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
        _adapter("img_k"),
        _adapter("img_v"),
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


def main() raises:
    var ctx = DeviceContext()
    print("==== scail2_block_flash_parity (SCAIL-2 i2v FLASH self-attn + 12 LoRA) ====")
    print("H=", H, " DH=", DH, " DIM=", DIM, " S=", S, " TXT=", TXT, " IMG=", IMG, " FFN=", FFN, " RANK=", RANK)

    var x = _in("s2in_x")
    var context_txt = _in("s2in_context_txt")
    var context_img = _in("s2in_context_img")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var lora = _load_lora()
    var cos = Tensor.from_host(_in("s2in_cos"), [S, DH // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("s2in_sin"), [S, DH // 2], STDtype.F32, ctx)

    var fwd = scail2_block_lora_forward[S, TXT, IMG, H, DH, True](
        x.copy(), context_txt.copy(), context_img.copy(), mv, w, lora,
        cos, sin, DIM, FFN, EPS, ctx,
    )

    var harness = ParityHarness()      # cos>=0.999
    var allok = True

    print("")
    print("---- forward output vs torch (F32 residual) ----")
    _check(harness, "x_out", fwd.x_out, _in("s2ref_x_out"), allok)

    var d_out = _in("s2in_d_out")
    var g = scail2_block_lora_backward[S, TXT, IMG, H, DH, True](
        d_out, mv, w, lora, fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )

    print("")
    print("---- input grads vs torch ----")
    _check(harness, "d_x (img latent) ", g.base.base.d_x, _in("s2ref_d_x"), allok)
    _check(harness, "d_context_txt    ", g.base.base.d_context, _in("s2ref_d_context_txt"), allok)
    _check(harness, "d_context_img    ", g.d_context_img, _in("s2ref_d_context_img"), allok)

    print("")
    print("---- LoRA d_A / d_B vs torch (12 adapters; gate 0.999) ----")
    _check(harness, "sa_q dA", g.base.sa_q_da, _in("s2ref_sa_q_dA"), allok)
    _check(harness, "sa_q dB", g.base.sa_q_db, _in("s2ref_sa_q_dB"), allok)
    _check(harness, "sa_k dA", g.base.sa_k_da, _in("s2ref_sa_k_dA"), allok)
    _check(harness, "sa_k dB", g.base.sa_k_db, _in("s2ref_sa_k_dB"), allok)
    _check(harness, "sa_v dA", g.base.sa_v_da, _in("s2ref_sa_v_dA"), allok)
    _check(harness, "sa_v dB", g.base.sa_v_db, _in("s2ref_sa_v_dB"), allok)
    _check(harness, "sa_o dA", g.base.sa_o_da, _in("s2ref_sa_o_dA"), allok)
    _check(harness, "sa_o dB", g.base.sa_o_db, _in("s2ref_sa_o_dB"), allok)
    _check(harness, "ca_q dA", g.base.ca_q_da, _in("s2ref_ca_q_dA"), allok)
    _check(harness, "ca_q dB", g.base.ca_q_db, _in("s2ref_ca_q_dB"), allok)
    _check(harness, "ca_k dA", g.base.ca_k_da, _in("s2ref_ca_k_dA"), allok)
    _check(harness, "ca_k dB", g.base.ca_k_db, _in("s2ref_ca_k_dB"), allok)
    _check(harness, "ca_v dA", g.base.ca_v_da, _in("s2ref_ca_v_dA"), allok)
    _check(harness, "ca_v dB", g.base.ca_v_db, _in("s2ref_ca_v_dB"), allok)
    _check(harness, "ca_o dA", g.base.ca_o_da, _in("s2ref_ca_o_dA"), allok)
    _check(harness, "ca_o dB", g.base.ca_o_db, _in("s2ref_ca_o_dB"), allok)
    _check(harness, "ffn0 dA", g.base.ffn0_da, _in("s2ref_ffn0_dA"), allok)
    _check(harness, "ffn0 dB", g.base.ffn0_db, _in("s2ref_ffn0_dB"), allok)
    _check(harness, "ffn2 dA", g.base.ffn2_da, _in("s2ref_ffn2_dA"), allok)
    _check(harness, "ffn2 dB", g.base.ffn2_db, _in("s2ref_ffn2_dB"), allok)
    print("  -- image-branch cross-attn adapters --")
    _check(harness, "img_k dA", g.img_k_da, _in("s2ref_img_k_dA"), allok)
    _check(harness, "img_k dB", g.img_k_db, _in("s2ref_img_k_dB"), allok)
    _check(harness, "img_v dA", g.img_v_da, _in("s2ref_img_v_dA"), allok)
    _check(harness, "img_v dB", g.img_v_db, _in("s2ref_img_v_dB"), allok)

    print("")
    if allok:
        print("VERDICT: PASS -- SCAIL-2 i2v F32-residual block LoRA fwd+bwd matches torch autograd")
    else:
        print("VERDICT: FAIL -- at least one output diverged (see FAIL lines above)")
