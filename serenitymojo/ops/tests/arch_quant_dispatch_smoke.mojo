# ops/tests/arch_quant_dispatch_smoke.mojo — verify device-arch quant dispatch.
#
# Portable gate: the SAME binary runs on both dev machines and validates the
# cross-arch seam from ops/arch.mojo. On each box it (1) queries the real
# compute capability, (2) checks the capability predicates are internally
# consistent, (3) confirms select_quant_backend picks an IMPLEMENTED backend,
# and (4) proves require_quant_backend FAILS LOUD for the backend this silicon
# cannot run (int4 on Blackwell / fp8 on Ampere).
#
# Build (native, no --target-accelerator — nldiff recipe):
#   pixi run mojo run -I . -I /home/alex/MOJO-libs -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/ops/tests/arch_quant_dispatch_smoke.mojo
#
# Expected on the 5080: sm_120, int4=0 fp8=1 fp4=1, selected=fp8-e4m3,
#   require(int4) RAISES. On the 3090 Ti: sm_86, int4=1 fp8=0 fp4=0,
#   selected=int4-w4a4, require(fp8) RAISES.

from std.gpu.host import DeviceContext
from std.testing import assert_true, assert_equal, assert_raises

from serenitymojo.ops.arch import (
    GpuArch, query_gpu_arch, select_quant_backend, require_quant_backend,
    backend_name, QB_INT4_W4A4, QB_FP8_E4M3, QB_FP4, QB_INT8_W8A8,
)


def main() raises:
    # DeviceContext() initializes the driver and asserts a GPU is present.
    var ctx = DeviceContext()
    _ = ctx

    var arch = query_gpu_arch()
    var sm = arch.sm()
    var int4 = arch.has_int4_imma()
    var fp8 = arch.has_fp8_tensorcores()
    var fp4 = arch.has_fp4_tensorcores()
    var sel = select_quant_backend(arch)

    print("device arch:", arch, " (sm", sm, ")")
    print("  int4_imma:", int4, " fp8_tc:", fp8, " fp4_tc:", fp4)
    print("  selected quant backend:", backend_name(sel))

    # (1) sm reconstruction is sane for a real NVIDIA part.
    assert_true(sm >= 70, "sm must be >= 70 for a supported NVIDIA GPU")

    # (2) predicates are internally consistent with their sm ranges.
    assert_equal(int4, sm >= 75 and sm <= 89, "int4_imma range")
    assert_equal(fp8, sm >= 89, "fp8 range")
    assert_equal(fp4, sm >= 100, "fp4 range")
    assert_true(fp4 == False or fp8 == True, "fp4 silicon always has fp8")

    # (3) the selected backend is one with a real ops/ implementation, and it
    #     matches the hardware: int4 box -> W4A4; fp8-capable box -> fp8;
    #     else universal int8.
    if int4:
        assert_equal(sel, QB_INT4_W4A4, "int4-capable GPU selects W4A4")
    elif fp8:
        assert_equal(sel, QB_FP8_E4M3, "fp8-capable (non-int4) GPU selects fp8")
    else:
        assert_equal(sel, QB_INT8_W8A8, "no int4/fp8 tensor cores -> int8")

    # (4) the guard passes for the selected backend and FAILS LOUD for the one
    #     this silicon lacks — the whole point of the seam.
    require_quant_backend(arch, sel)  # must not raise

    if int4:
        # Ampere/Ada box: fp8 is unavailable -> guard must raise.
        with assert_raises():
            require_quant_backend(arch, QB_FP8_E4M3)
        print("  OK: require(fp8) correctly raised on an int4-only GPU")
    else:
        # Blackwell (or Hopper) box: int4 IMMA is gone -> guard must raise.
        with assert_raises():
            require_quant_backend(arch, QB_INT4_W4A4)
        print("  OK: require(int4-w4a4) correctly raised on a no-int4 GPU")

    # fp4 guard fails on anything below Blackwell.
    if not fp4:
        with assert_raises():
            require_quant_backend(arch, QB_FP4)

    print("arch_quant_dispatch_smoke: PASS")
