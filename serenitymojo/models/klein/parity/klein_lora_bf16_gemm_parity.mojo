# serenitymojo/models/klein/parity/klein_lora_bf16_gemm_parity.mojo
#
# PARITY GATE for the bf16 tensor-core LoRA backward GEMMs
# (KLEIN_LORA_BF16_GEMM, models/klein/lora_block.mojo, 2026-07-11 — APPROVED
# numerics change, reference trainer-autocast-matching class):
#
#   =0 (reference): klein_lora_bwd_device_resident_tensors_unfused —
#       dA/dB run F32×F32 cutlass simt sgemm (CUDA cores, split-K);
#       t/d_t/d_x already ran bf16-input tensor-core via the mixed arms.
#   =1 (change):    klein_lora_bwd_device_resident_tensors_bf16gemm —
#       SAME 5-GEMM chain; hoisted RNE bf16 casts (one per reused operand);
#       dA/dB become bf16-input tensor-core with F32 accumulate + F32 output.
#
# WHAT ROUNDS (=1 vs =0):
#   d_x : NOTHING new — same operands (bf16(d_t), bf16 A), same GEMM call.
#         Expected BIT-IDENTICAL (max_abs 0.0); cuBLAS algo-selection wobble
#         from buffer-pointer alignment is the only accepted deviation class.
#   d_B : d_dy rounds to bf16 (same rounded value the d_t GEMM consumes) and
#         t rounds to bf16 (NEW). cos >= 0.999 vs the F32-product reference.
#   d_A : d_t rounds to bf16 (same rounded value the d_x GEMM consumes) and
#         x rounds to bf16 (same rounded value the t GEMM consumes).
#         cos >= 0.999 vs the F32-product reference.
#
# Build+run:
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#     serenitymojo/models/klein/parity/klein_lora_bf16_gemm_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import ArcPointer
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice,
    klein_lora_bwd_device_resident_tensors_bf16gemm,
    klein_lora_bwd_device_resident_tensors_unfused,
)


def _lcg(mut state: UInt64) -> Float32:
    state = state * 6364136223846793005 + 1442695040888963407
    var u = (state >> 40) & 0xFFFFFF
    return (Float32(Int(u)) / Float32(1 << 23)) - 1.0


def _rand(n: Int, mut state: UInt64, scale: Float32) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(_lcg(state) * scale)
    return v^


def _check(
    name: String, got: Tensor, expected: Tensor, bar: Float64,
    require_bits: Bool, ctx: DeviceContext,
) raises:
    var ref_h = expected.to_host(ctx)
    var h = ParityHarness(bar)
    var r = h.compare(got, ref_h, ctx)
    print(name, " cos=", r.cos, " max_abs=", r.max_abs, " pass=", r.passed)
    if not r.passed:
        raise Error("lora bf16-gemm parity FAILED: " + name)
    if require_bits and r.max_abs != 0.0:
        print("  NOTE:", name, "not bit-identical (accepted only if algo-selection class)")


def _run_case(
    name: String, M: Int, in_f: Int, out_f: Int, s: Float32,
    ctx: DeviceContext,
) raises:
    var rank = 16
    var seed: UInt64 = 0xBF16 ^ UInt64(M * 131 + in_f * 7 + out_f)
    var xh = _rand(M * in_f, seed, 2.0)
    var dh = _rand(M * out_f, seed, 0.02)
    var ah = _rand(rank * in_f, seed, 0.4)
    var bh = _rand(out_f * rank, seed, 0.4)
    var x = Tensor.from_host(xh, [M, in_f], STDtype.F32, ctx)
    var d_contrib = Tensor.from_host(dh, [M, out_f], STDtype.F32, ctx)
    var a = Tensor.from_host(ah, [rank, in_f], STDtype.BF16, ctx)
    var b = Tensor.from_host(bh, [out_f, rank], STDtype.BF16, ctx)
    var lo = LoraAdapterDevice(
        ArcPointer[Tensor](a^), ArcPointer[Tensor](b^),
        rank, in_f, out_f, s,
    )

    var ref_g = klein_lora_bwd_device_resident_tensors_unfused(
        d_contrib, x, lo, M, ctx
    )
    var got_g = klein_lora_bwd_device_resident_tensors_bf16gemm(
        d_contrib, x, lo, M, ctx
    )
    # d_x carries no new rounding — expect bit-identity (8-nines floor).
    _check(name + " d_x", got_g.d_x[], ref_g.d_x[], 0.99999999, True, ctx)
    # dA/dB are THE approved change: bf16 tensor-core inputs. cos >= 0.999.
    _check(name + " d_a", got_g.d_a[], ref_g.d_a[], 0.999, False, ctx)
    _check(name + " d_b", got_g.d_b[], ref_g.d_b[], 0.999, False, ctx)


def main() raises:
    var ctx = DeviceContext()
    # Klein-9B rank-16 slot shapes (512px run: N_IMG=1024, N_TXT=512, D=4096,
    # F=12288): double q/k/v/out [4096->4096], single qkv+mlp fused out
    # (3D+2F=36864, scaled here), ff_in [4096->2F], ff_out [12288->4096].
    _run_case("dbl qkv img  M1024 4096->4096 ", 1024, 4096, 4096, Float32(1.0), ctx)
    _run_case("dbl qkv txt  M512  4096->4096 ", 512, 4096, 4096, Float32(1.0), ctx)
    _run_case("dbl ff_in    M1024 4096->24576", 1024, 4096, 24576, Float32(1.0), ctx)
    _run_case("dbl ff_out   M1024 12288->4096", 1024, 12288, 4096, Float32(1.0), ctx)
    _run_case("sgl qkv_mlp  M1536 4096->36864", 1536, 4096, 36864, Float32(1.0), ctx)
    _run_case("sgl out      M1536 16384->4096", 1536, 16384, 4096, Float32(1.0), ctx)
    _run_case("frac scale   M768  4096->4096 ", 768, 4096, 4096, Float32(0.8125), ctx)
    print("ALL lora bf16-gemm parity cases PASSED")
