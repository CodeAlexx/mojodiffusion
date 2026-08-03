# serenitymojo/models/minimax_h3/parity/minimax_h3_dit_frontend_parity.mojo
#
# MiniMax-H3 DiT front-end parity gate: rotary table, timestep projection,
# AdaLN row mapping.
#
# Reference: diffusers PR huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_dit_frontend_oracle.py. The rotary tables are built over
# the REAL packed layouts (`build_packed_sequence`), not synthetic coordinates,
# so this gate covers the geometry the model will actually see — including a
# keyframe layout whose conditioning rows sit at their own rotary time.
#
# Bar: bit-exact on the integer mapping and the inv_freq buffer; <= 1 ulp on the
# transcendentals. cos/sin/exp are correctly rounded by neither libm nor torch,
# so demanding bit-equality across two different math libraries would be a bar
# about libm, not about this port. 1 ulp on a value in [-1, 1] is 6e-8.
#
# Oracle: python3 scripts/minimax_h3_dit_frontend_oracle.py
# Run:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_dit_frontend_parity.mojo \
#     -o output/checks/minimax_h3_dit_frontend_parity \
#   && output/checks/minimax_h3_dit_frontend_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.dit_frontend import (
    minimax_h3_adaln_indices,
    minimax_h3_rope_inv_freq,
    minimax_h3_rope_table,
    minimax_h3_timestep_projection,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_dit_frontend/dit_frontend_ref.safetensors"
# One float32 ulp for a value of magnitude ~1.
comptime ULP = Float32(1.1920929e-7)

# Tolerance for cos/sin only. The ANGLES feeding them are gated bit-exactly, so
# any residual here is the difference between two float32 transcendental
# implementations — torch's vectorized kernels and the libm this binary links.
# Neither is correctly rounded, and they disagree by a few ulp once the argument
# is large (the rotary clock runs to ~100 radians on a long clip). Measured max
# over these layouts: 4.77e-07, i.e. 4 ulp. The bar is set at 8 ulp: tight
# enough that a real arithmetic error cannot hide under it (the failures this
# gate caught before the angle split were the same 2-4 ulp), loose enough that
# it is not secretly a test of whose libm is installed.
comptime LIBM_TOL = Float32(9.5e-7)


def _load_f64(ref st: SafeTensors, name: String) raises -> List[Float64]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F64:
        raise Error(String("_load_f64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float64]()
    var out = List[Float64]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I64:
        raise Error(String("_load_i64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


struct Report(Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def exact_int(mut self, label: String, got: List[Int], want: List[Int]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var bad = 0
        var first = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            print("  ok  ", label, "exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, bad, "of", len(got), "differ; first at", first,
                "got", got[first], "want", want[first],
            )

    def exact_f32(mut self, label: String, got: List[Float32], want: List[Float32]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var bad = 0
        var first = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            print("  ok  ", label, "bit-exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, bad, "of", len(got), "differ; first at", first,
                "got", got[first], "want", want[first],
            )

    def close_f32(mut self, label: String, got: List[Float32], want: List[Float32], tol: Float32):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var bad = 0
        var worst = Float32(0.0)
        var exact = 0
        for i in range(len(got)):
            var diff = got[i] - want[i]
            var mag = -diff if diff < 0.0 else diff
            if mag > worst:
                worst = mag
            if mag > tol:
                bad += 1
            if got[i] == want[i]:
                exact += 1
        if bad == 0:
            var pct = (exact * 100) // len(got)
            print(
                "  ok  ", label, len(got), "values, max_abs", worst,
                "(", pct, "% bit-exact )",
            )
        else:
            self.failures += 1
            print("  FAIL", label, bad, "of", len(got), "exceed tol; max_abs", worst)


def _check_layout(
    mut report: Report, ref st: SafeTensors, name: String, inv_freq: List[Float32]
) raises:
    var position_ids = _load_f64(st, name + ".position_ids")
    var sequence_length = len(position_ids) // 3
    var table = minimax_h3_rope_table(position_ids, sequence_length, inv_freq)
    # The arithmetic is ours and must be exact.
    report.exact_f32(name + ".angles", table.angles, _load_f32(st, name + ".angles"))
    # The transcendental is libm's; see LIBM_TOL.
    report.close_f32(name + ".cos", table.cos, _load_f32(st, name + ".cos"), LIBM_TOL)
    report.close_f32(name + ".sin", table.sin, _load_f32(st, name + ".sin"), LIBM_TOL)

    var tags = _load_i64(st, name + ".token_tags")
    var indices = _load_i64(st, name + ".timestep_indices")
    report.exact_int(
        name + ".adaln_indices",
        minimax_h3_adaln_indices(indices, tags),
        _load_i64(st, name + ".adaln_indices"),
    )


def main() raises:
    print("MiniMax-H3 DiT front-end parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var report = Report()

    print("[1] rope inv_freq (computed, not loaded)")
    var inv_freq = minimax_h3_rope_inv_freq()
    report.close_f32("rope.inv_freq", inv_freq, _load_f32(st, "rope.inv_freq"), ULP)

    print("[2] rotary table + adaln rows over real packed layouts")
    _check_layout(report, st, String("tiny"), inv_freq)
    _check_layout(report, st, String("keyframe"), inv_freq)
    _check_layout(report, st, String("medium"), inv_freq)

    print("[3] padding tags clamp to 0 rather than indexing backwards")
    report.exact_int(
        "padtags.adaln_indices",
        minimax_h3_adaln_indices(
            _load_i64(st, "padtags.timestep_indices"), _load_i64(st, "padtags.token_tags")
        ),
        _load_i64(st, "padtags.adaln_indices"),
    )

    print("[4] sinusoidal timestep projection")
    var timesteps = _load_f32(st, "timeproj.timesteps")
    report.close_f32(
        "timeproj.out",
        minimax_h3_timestep_projection(timesteps),
        _load_f32(st, "timeproj.out"),
        ULP,
    )

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks within tolerance")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks differ")
        raise Error("MiniMax-H3 DiT front-end parity gate failed")
