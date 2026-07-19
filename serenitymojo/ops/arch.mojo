# ops/arch.mojo — device compute-capability + quant-backend selection.
#
# One source, two GPUs. The stack builds NATIVELY on each machine (no
# --target-accelerator), and the quantized-GEMM backends differ by silicon:
#
#   RTX 3090 Ti (sm_86, Ampere): int4 tensor cores (s4 IMMA) → SVDQuant W4A4.
#     NO fp8 tensor cores (need sm_89+).
#   RTX 5080  (sm_120, Blackwell): fp8 + fp4 tensor cores. NO int4 IMMA —
#     s4 tensor-core GEMM was REMOVED on sm_90+; it cannot run here at all.
#
# So the choice is a HARDWARE fact, not a config knob. This module reports the
# device's compute capability at runtime (CUDA driver query, same bindings as
# offload/vmm_cuda.mojo) plus pure capability predicates, and picks the fastest
# quant backend the silicon actually supports. The per-layer hot path never
# calls this — a loader queries ONCE, tags the layer, and quantized_linear
# dispatches on the tag. require_quant_backend() turns "wrong backend for this
# GPU" into a clear fail-loud instead of a cryptic CUTLASS/driver error.
#
# Mojo 1.0.0b1, NVIDIA GPU (Linux x86-64).

from std.ffi import external_call
from std.memory import alloc

from serenitymojo.io.ffi import BytePtr


# ── CUDA driver attribute enums (cuda.h) ──────────────────────────────────────
comptime CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Int32 = 75
comptime CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Int32 = 76


# ── Quant-backend tags (stored on a layer by its loader; dispatched on cheaply) ─
comptime QB_BF16: Int = 0        # no quant GEMM — dense bf16 fallback (any GPU)
comptime QB_INT8_W8A8: Int = 1   # int8 W8A8 (ops/int8_linear) — any sm_75+ GPU
comptime QB_FP8_E4M3: Int = 2    # weight-only fp8 E4M3 (ops/fp8_gemm) — sm_89+
comptime QB_INT4_W4A4: Int = 3   # SVDQuant W4A4 (ops/svdquant_w4a4) — sm_75..89
comptime QB_FP4: Int = 4         # nvfp4 / mxfp4 — sm_100+ (Blackwell); capability
                                 # flag only until a general ops/ fp4 linear lands


def backend_name(tag: Int) -> String:
    if tag == QB_BF16: return String("bf16")
    if tag == QB_INT8_W8A8: return String("int8-w8a8")
    if tag == QB_FP8_E4M3: return String("fp8-e4m3")
    if tag == QB_INT4_W4A4: return String("int4-w4a4")
    if tag == QB_FP4: return String("fp4")
    return String("quant-backend?(") + String(tag) + ")"


def _ptr[pointee: AnyType](p: UnsafePointer[pointee, MutAnyOrigin]) -> BytePtr:
    return BytePtr(unsafe_from_address=Int(p))


# cuInit is idempotent; call it so query works even before a DeviceContext exists.
def _cu_init() raises:
    var rc = Int(external_call["cuInit", Int32](Int32(0)))
    if rc != 0:
        raise Error(String("cuInit failed CUresult=") + String(rc))


def _cu_device_get(ordinal: Int32) raises -> Int32:
    var out = alloc[Int32](1)
    out[0] = 0
    var rc = Int(external_call["cuDeviceGet", Int32](_ptr(out), ordinal))
    var dev = out[0]
    out.free()
    if rc != 0:
        raise Error(String("cuDeviceGet failed CUresult=") + String(rc))
    return dev


def _cu_device_get_attribute(device: Int32, attribute: Int32) raises -> Int32:
    var out = alloc[Int32](1)
    out[0] = 0
    var rc = Int(external_call["cuDeviceGetAttribute", Int32](
        _ptr(out), attribute, device
    ))
    var value = out[0]
    out.free()
    if rc != 0:
        raise Error(String("cuDeviceGetAttribute failed CUresult=") + String(rc))
    return value


# ── GpuArch — device compute capability + capability predicates ────────────────
@fieldwise_init
struct GpuArch(Copyable, Movable, Writable):
    var major: Int
    var minor: Int

    # sm number, e.g. (8,6) -> 86, (12,0) -> 120. Minor is single-digit on every
    # shipping NVIDIA part, so major*10+minor is unambiguous.
    def sm(self) -> Int:
        return self.major * 10 + self.minor

    # int4 s4 IMMA tensor-core GEMM: introduced Turing (sm_75), present through
    # Ampere (sm_80/86/87) and Ada (sm_89); REMOVED on Hopper/Blackwell (sm_90+).
    def has_int4_imma(self) -> Bool:
        var s = self.sm()
        return s >= 75 and s <= 89

    # fp8 E4M3/E5M2 tensor cores: Ada (sm_89), Hopper (sm_90), Blackwell (sm_120+).
    def has_fp8_tensorcores(self) -> Bool:
        return self.sm() >= 89

    # fp4 (nvfp4/mxfp4) tensor cores: Blackwell (sm_100 datacenter, sm_120 consumer).
    def has_fp4_tensorcores(self) -> Bool:
        return self.sm() >= 100

    def write_to(self, mut writer: Some[Writer]):
        writer.write("sm_", self.major, self.minor)


# ── query the physical device (runtime) ───────────────────────────────────────
def query_gpu_arch(ordinal: Int32 = 0) raises -> GpuArch:
    _cu_init()
    var dev = _cu_device_get(ordinal)
    var major = _cu_device_get_attribute(
        dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    var minor = _cu_device_get_attribute(
        dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    return GpuArch(Int(major), Int(minor))


# ── selection policy: fastest IMPLEMENTED quant backend for this device ────────
# Every tag returned here has a real ops/ backend today:
#   INT4_W4A4 -> ops/svdquant_w4a4 (3090)   FP8_E4M3 -> ops/fp8_gemm (5080)
#   INT8_W8A8 -> ops/int8_linear (universal fallback)
# fp4 is NOT auto-selected yet — capability is detected (has_fp4_tensorcores)
# but there is no general ops/ fp4 linear to route to. Add it here when there is.
def select_quant_backend(arch: GpuArch) -> Int:
    if arch.has_int4_imma():
        return QB_INT4_W4A4        # 3090 sm_86: SVDQuant W4A4 (4.4–7.3× cuBLAS bf16)
    if arch.has_fp8_tensorcores():
        return QB_FP8_E4M3         # 5080 sm_120: weight-only fp8 (resident, 1 B/param)
    return QB_INT8_W8A8            # sm_75–88 without fp8: universal int8 W8A8


# ── fail-loud guard: does THIS device support the requested backend? ───────────
# Call once at load/quantize time (not per layer). int8 and bf16 run everywhere,
# so only the tensor-core-gated backends are guarded.
def require_quant_backend(arch: GpuArch, want: Int) raises:
    if want == QB_INT4_W4A4 and not arch.has_int4_imma():
        raise Error(
            String("quant backend int4-w4a4 requires sm 75–89 (Turing..Ada); ")
            + "this device is " + String(arch) + " — s4 IMMA was removed on "
            + "sm_90+. Use fp8-e4m3 or int8-w8a8 on this GPU.")
    if want == QB_FP8_E4M3 and not arch.has_fp8_tensorcores():
        raise Error(
            String("quant backend fp8-e4m3 requires sm>=89 (Ada+); this device ")
            + "is " + String(arch) + " — no fp8 tensor cores. Use int4-w4a4 "
            + "(SVDQuant W4A4) or int8-w8a8 on this GPU.")
    if want == QB_FP4 and not arch.has_fp4_tensorcores():
        raise Error(
            String("quant backend fp4 requires sm>=100 (Blackwell); this device ")
            + "is " + String(arch) + ".")
