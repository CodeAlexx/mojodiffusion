# parity.mojo — ParityHarness: compare a GPU `Tensor` against a host reference.
#
# Phase A foundation (PHASE_AB_PLAN.md): op-level cos / max-abs-diff vs a
# reference dump (numpy oracle produced offline under `parity/`, or a flame-core
# /torch per-layer dump). The gate for every op is cosine similarity >= 0.999.
#
# Cosine similarity:    cos = dot(a, b) / (||a|| * ||b||)
# Max-abs-diff:         max_i |a_i - b_i|
#
# Both are computed in F64 on the host after a `to_host()` readback, so the
# comparison itself never loses precision relative to the BF16/F16 device data.
# Python (numpy) is a DEV-ONLY oracle: references are generated under `parity/`
# and passed in as a host `List[Float32]`. Nothing here touches Python.

from std.math import sqrt
from serenitymojo.tensor import Tensor
from std.gpu.host import DeviceContext


@fieldwise_init
struct ParityResult(Copyable, Movable, Writable):
    """Outcome of a parity comparison."""

    var cos: Float64
    var max_abs: Float64
    var passed: Bool
    var n: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ParityResult(cos=",
            self.cos,
            ", max_abs=",
            self.max_abs,
            ", n=",
            self.n,
            ", ",
            "PASS" if self.passed else "FAIL",
            ")",
        )


comptime DEFAULT_COS_THRESHOLD = 0.999


struct ParityHarness:
    """Compares GPU `Tensor` outputs against host references (cos + max-abs)."""

    var cos_threshold: Float64

    def __init__(out self, cos_threshold: Float64 = DEFAULT_COS_THRESHOLD):
        self.cos_threshold = cos_threshold

    @staticmethod
    def _compare(
        actual: List[Float32], reference: List[Float32], cos_threshold: Float64
    ) raises -> ParityResult:
        """Core comparison on two host arrays. Computes cos + max-abs in F64."""
        var n = len(actual)
        if n != len(reference):
            raise Error(
                String("parity: length mismatch actual=")
                + String(n)
                + " reference="
                + String(len(reference))
            )
        if n == 0:
            raise Error("parity: empty arrays")
        var dot: Float64 = 0.0
        var na: Float64 = 0.0
        var nb: Float64 = 0.0
        var max_abs: Float64 = 0.0
        for i in range(n):
            var a = Float64(actual[i])
            var b = Float64(reference[i])
            dot += a * b
            na += a * a
            nb += b * b
            var d = a - b
            if d < 0.0:
                d = -d
            if d > max_abs:
                max_abs = d
        var denom = sqrt(na) * sqrt(nb)
        # If both vectors are all-zero, treat as a perfect match (cos = 1).
        var cos: Float64
        if denom == 0.0:
            cos = 1.0 if (na == 0.0 and nb == 0.0) else 0.0
        else:
            cos = dot / denom
        return ParityResult(
            cos=cos, max_abs=max_abs, passed=(cos >= cos_threshold), n=n
        )

    def compare_host(
        self, actual: List[Float32], reference: List[Float32]
    ) raises -> ParityResult:
        """Compare two host arrays directly."""
        return Self._compare(actual, reference, self.cos_threshold)

    def compare(
        self, t: Tensor, reference: List[Float32], ctx: DeviceContext
    ) raises -> ParityResult:
        """Read a GPU `Tensor` back to host (as F32) and compare against the
        host reference."""
        var actual = t.to_host(ctx)
        return Self._compare(actual, reference, self.cos_threshold)
