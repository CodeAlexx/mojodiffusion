# safetensors_writer_smoke.mojo — round-trip gate for io/safetensors_writer.mojo.
#
# Tenet 4 (measurement beats assertion): the writer is only "done" if a file it
# writes reloads BYTE-FOR-BYTE through the existing reader (io/safetensors.mojo).
# This smoke:
#   1. builds a few Tensors with known values (F32 and BF16 storage),
#   2. saves them via save_safetensors to /tmp/st_roundtrip.safetensors,
#   3. reloads via SafeTensors.open (the real reader),
#   4. asserts, per tensor:
#        - dtype name matches,
#        - shape matches,
#        - VALUE round-trip: max_abs == 0 and cos == 1.0 between the original
#          stored values (Tensor.to_host) and the reloaded values,
#        - BYTE round-trip: the reloaded raw bytes equal the original device
#          bytes exactly (the strongest check — proves BF16 stayed BF16, no
#          F32 detour).
#   5. prints the safetensors header bytes for eyeball + leaves the file on disk
#      so an external `python -c "from safetensors import safe_open"` can open it.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/io/parity/safetensors_writer_smoke.mojo
#
# Loud-fail: any mismatch raises and aborts with a descriptive message.

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error(
            String("length mismatch: ") + String(len(a)) + " vs " + String(len(b))
        )
    var m: Float32 = 0.0
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def _cos_sim(a: List[Float32], b: List[Float32]) raises -> Float64:
    # Accumulate in F64 (same as ParityHarness) so the dot/norm reduction itself
    # does not lose precision. For bit-identical inputs the analytic cos is
    # exactly 1.0; F64 reduction lands within ~1e-15 of it (vs ~1e-7 for an F32
    # reduction, which is purely the reduction's rounding, NOT a data diff).
    if len(a) != len(b):
        raise Error("length mismatch in cos_sim")
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    var denom = sqrt(na) * sqrt(nb)
    if denom == 0.0:
        # Both all-zero -> define cos as 1.0 (identical).
        if na == 0.0 and nb == 0.0:
            return 1.0
        return 0.0
    return dot / denom


def _check_one(
    name: String,
    src: Tensor,
    loaded: SafeTensors,
    ctx: DeviceContext,
) raises:
    """Compare a saved Tensor against what the reader returns for `name`."""
    # dtype + shape from the reader's index.
    var info = loaded.tensor_info(name)
    if info.dtype != src.dtype():
        raise Error(
            String("dtype mismatch for '")
            + name
            + "': reader="
            + info.dtype.name()
            + " src="
            + src.dtype().name()
        )
    var ss = src.shape()
    if len(info.shape) != len(ss):
        raise Error(String("shape rank mismatch for '") + name + "'")
    for i in range(len(ss)):
        if info.shape[i] != ss[i]:
            raise Error(String("shape dim mismatch for '") + name + "'")

    # BYTE round-trip: reader's raw bytes vs the original device bytes.
    # Reload the source's device bytes to host for comparison.
    var src_bytes = _tensor_raw_bytes(src, ctx)
    var rd = loaded.tensor_bytes(name)
    if len(rd) != len(src_bytes):
        raise Error(
            String("byte length mismatch for '")
            + name
            + "': reader="
            + String(len(rd))
            + " src="
            + String(len(src_bytes))
        )
    for i in range(len(src_bytes)):
        if rd[i] != src_bytes[i]:
            raise Error(
                String("BYTE mismatch for '")
                + name
                + "' at byte "
                + String(i)
                + ": reader="
                + String(Int(rd[i]))
                + " src="
                + String(Int(src_bytes[i]))
            )

    # VALUE round-trip: build a Tensor from the reloaded bytes and upcast to F32.
    var reloaded = _tensor_from_loaded(loaded, name, ctx)
    var a = src.to_host(ctx)
    var b = reloaded.to_host(ctx)
    var mad = _max_abs_diff(a, b)
    var cos = _cos_sim(a, b)
    print(
        "  ",
        name,
        " dtype=",
        info.dtype.name(),
        " n=",
        len(a),
        " max_abs=",
        mad,
        " cos=",
        cos,
    )
    # The STRONG gate is byte-exactness: the byte loop above already proved the
    # reloaded bytes equal the original device bytes, and max_abs==0 confirms the
    # values are bit-identical. cos is reported for completeness; in F64 it lands
    # within rounding of 1.0 for identical data (an F32-reduction cos of
    # 0.9999999 is the reduction's own rounding, not a data difference, which is
    # exactly why max_abs is the authoritative check here).
    if mad != 0.0:
        raise Error(String("max_abs != 0 for '") + name + "'")
    if cos < 1.0 - 1e-9:
        raise Error(
            String("cos below tolerance for '")
            + name
            + "': cos="
            + String(cos)
        )


def _tensor_raw_bytes(t: Tensor, ctx: DeviceContext) raises -> List[UInt8]:
    """Copy a Tensor's device bytes to a host List[UInt8] (byte-exact)."""
    var nbytes = t.nbytes()
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=host, src_buf=t.buf)
    ctx.synchronize()
    var hp = host.unsafe_ptr()
    var out = List[UInt8]()
    for i in range(nbytes):
        out.append(hp[i])
    return out^


def _tensor_from_loaded(
    loaded: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """Build a Tensor from the reader's bytes for `name`, preserving dtype.
    Uploads the reader's mmap'd bytes verbatim to a device buffer."""
    var info = loaded.tensor_info(name)
    var span = loaded.tensor_bytes(name)
    var nbytes = len(span)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var hp = host.unsafe_ptr()
    for i in range(nbytes):
        hp[i] = span[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, info.shape.copy(), info.dtype)


def main() raises:
    print("=== safetensors_writer round-trip smoke ===")
    var ctx = DeviceContext()
    var path = String("/tmp/st_roundtrip.safetensors")

    # ── Build a few tensors with known values. ──────────────────────────────
    # F32 weight [2,3].
    var w_vals = List[Float32]()
    w_vals.append(1.0)
    w_vals.append(-2.5)
    w_vals.append(3.25)
    w_vals.append(0.0)
    w_vals.append(100.0)
    w_vals.append(-0.125)
    var w_shape = List[Int]()
    w_shape.append(2)
    w_shape.append(3)
    var w = Tensor.from_host(w_vals, w_shape^, STDtype.F32, ctx)

    # BF16 LoRA-down [4] (values chosen to be exactly representable in BF16:
    # 1.0, 2.0, -4.0, 0.5 — powers of two have no mantissa rounding).
    var b_vals = List[Float32]()
    b_vals.append(1.0)
    b_vals.append(2.0)
    b_vals.append(-4.0)
    b_vals.append(0.5)
    var b_shape = List[Int]()
    b_shape.append(4)
    var lora_b = Tensor.from_host(b_vals, b_shape^, STDtype.BF16, ctx)

    # BF16 LoRA-up [3,2] with a value that DOES round in BF16 (3.140625 is the
    # BF16 nearest to pi) — the smoke compares against the STORED value
    # (src.to_host), so this still round-trips to max_abs=0/cos=1.0 and proves
    # we never re-round.
    var u_vals = List[Float32]()
    u_vals.append(3.14159265)
    u_vals.append(-1.0)
    u_vals.append(0.0)
    u_vals.append(256.0)
    u_vals.append(-0.001953125)  # 2^-9, exact in BF16
    u_vals.append(42.0)
    var u_shape = List[Int]()
    u_shape.append(3)
    u_shape.append(2)
    var lora_u = Tensor.from_host(u_vals, u_shape^, STDtype.BF16, ctx)

    # F32 scalar-ish [1].
    var s_vals = List[Float32]()
    s_vals.append(-7.0)
    var s_shape = List[Int]()
    s_shape.append(1)
    var scal = Tensor.from_host(s_vals, s_shape^, STDtype.F32, ctx)

    # Names exercise '.', '/', '-', digits (real LoRA key shapes).
    var names = List[String]()
    names.append(String("transformer.blocks.0/attn.to_q.lora_A-weight"))
    names.append(String("transformer.blocks.0/attn.to_q.lora_B.weight"))
    names.append(String("transformer.blocks.12/mlp.lora_up.weight"))
    names.append(String("alpha"))

    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(w^))
    tensors.append(ArcPointer(lora_b^))
    tensors.append(ArcPointer(lora_u^))
    tensors.append(ArcPointer(scal^))

    # ── Save. ───────────────────────────────────────────────────────────────
    save_safetensors(names, tensors, path, ctx)
    print("wrote ", path)

    # ── Reload via the real reader and compare. ─────────────────────────────
    var loaded = SafeTensors.open(path)
    print("reloaded ", loaded.count(), " tensors")
    if loaded.count() != len(names):
        raise Error(
            String("count mismatch: wrote ")
            + String(len(names))
            + " read "
            + String(loaded.count())
        )

    print("per-tensor round-trip:")
    for i in range(len(names)):
        _check_one(names[i], tensors[i][], loaded, ctx)

    print("=== ALL ROUND-TRIPS PASS (max_abs=0, cos=1.0, byte-exact) ===")
    print("file left at ", path, " for external safetensors verification")
