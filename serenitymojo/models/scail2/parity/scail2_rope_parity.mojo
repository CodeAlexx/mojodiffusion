# Parity gate for SCAIL-2 sequence order, shifts, and pose complex-frequency
# average pooling. Generate fixtures with scail2_rope_oracle.py first.

from max.gpu.host import DeviceContext
from std.memory import alloc
from std.sys import argv
from serenitymojo.io.ffi import (
    sys_open, sys_close, sys_pread, file_size, O_RDONLY,
)
from serenitymojo.parity import ParityHarness
from serenitymojo.models.scail2.scail2_rope import (
    Scail2SequencePlan,
    build_scail2_position_descriptors,
    build_scail2_rope_tables,
)
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.wan22_dit import _expand_rope_per_head
from serenitymojo.ops.rope import (
    rope_interleaved,
    rope_interleaved_head_broadcast,
)
from serenitymojo.tensor import Tensor


def _join(dir: String, name: String) -> String:
    return dir + "/" + name


def _read_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open parity fixture: ") + path)
    var nbytes = file_size(fd)
    if nbytes <= 0 or nbytes % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("invalid parity fixture: ") + path)
    var buf = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, buf + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    if done != nbytes:
        buf.free()
        raise Error(String("short parity fixture read: ") + path)
    var values = List[Float32]()
    var fp = buf.bitcast[Float32]()
    for i in range(nbytes // 4):
        values.append(fp[i])
    buf.free()
    return values^


def _gate_case(
    name: String,
    plan: Scail2SequencePlan,
    fixture_dir: String,
    ctx: DeviceContext,
) raises:
    var positions_ref = _read_f32(_join(fixture_dir, name + "_positions.f32"))
    var cos_ref = _read_f32(_join(fixture_dir, name + "_cos.f32"))
    var sin_ref = _read_f32(_join(fixture_dir, name + "_sin.f32"))
    if len(positions_ref) != plan.sequence_length() * 4:
        raise Error(String("position fixture shape mismatch: ") + name)
    if len(cos_ref) != plan.sequence_length() * 64:
        raise Error(String("cos fixture shape mismatch: ") + name)
    if len(sin_ref) != plan.sequence_length() * 64:
        raise Error(String("sin fixture shape mismatch: ") + name)

    var positions = build_scail2_position_descriptors(plan, ctx)
    var tables = build_scail2_rope_tables(plan, ctx, STDtype.F32)
    var tables_bf16 = build_scail2_rope_tables(plan, ctx, STDtype.BF16)
    var harness = ParityHarness(0.999)
    var pos_result = harness.compare(positions, positions_ref, ctx)
    var cos_result = harness.compare(tables[0], cos_ref, ctx)
    var sin_result = harness.compare(tables[1], sin_ref, ctx)
    var cos_bf16_result = harness.compare(tables_bf16[0], cos_ref, ctx)
    var sin_bf16_result = harness.compare(tables_bf16[1], sin_ref, ctx)
    print(name, " positions: ", pos_result)
    print(name, " cos: ", cos_result)
    print(name, " sin: ", sin_result)
    print(name, " cos bf16: ", cos_bf16_result)
    print(name, " sin bf16: ", sin_bf16_result)
    if (
        not pos_result.passed or not cos_result.passed or not sin_result.passed
        or not cos_bf16_result.passed or not sin_bf16_result.passed
    ):
        raise Error(String("SCAIL-2 sequence/RoPE gate failed: ") + name)


def _gate_head_broadcast(ctx: DeviceContext) raises:
    comptime H = 4
    comptime DH = 128
    var plan = Scail2SequencePlan(
        video_t=3, grid_h=4, grid_w=6,
        additional_ref_count=2, replace_flag=False,
    )
    var rows = plan.sequence_length()
    var values = List[Float32]()
    for i in range(rows * H * DH):
        values.append(Float32((i % 257) - 128) / 97.0)
    var shape = [1, rows, H, DH]
    var harness = ParityHarness(1.0)

    var tables_f32 = build_scail2_rope_tables(plan, ctx, STDtype.F32)
    var expanded_cos_f32 = _expand_rope_per_head(
        tables_f32[0], rows, H, DH // 2, ctx
    )
    var expanded_sin_f32 = _expand_rope_per_head(
        tables_f32[1], rows, H, DH // 2, ctx
    )
    var x_f32 = Tensor.from_host(values, shape.copy(), STDtype.F32, ctx)
    var expected_f32 = rope_interleaved(
        x_f32.clone(ctx), expanded_cos_f32, expanded_sin_f32, ctx
    ).to_host(ctx)
    var actual_f32 = rope_interleaved_head_broadcast(
        x_f32, tables_f32[0], tables_f32[1], H, ctx
    )
    var f32_result = harness.compare(actual_f32, expected_f32, ctx)

    var tables_bf16 = build_scail2_rope_tables(plan, ctx, STDtype.BF16)
    var expanded_cos_bf16 = _expand_rope_per_head(
        tables_bf16[0], rows, H, DH // 2, ctx
    )
    var expanded_sin_bf16 = _expand_rope_per_head(
        tables_bf16[1], rows, H, DH // 2, ctx
    )
    var x_bf16 = Tensor.from_host(values, shape.copy(), STDtype.BF16, ctx)
    var expected_bf16 = rope_interleaved(
        x_bf16.clone(ctx), expanded_cos_bf16, expanded_sin_bf16, ctx
    ).to_host(ctx)
    var actual_bf16 = rope_interleaved_head_broadcast(
        x_bf16, tables_bf16[0], tables_bf16[1], H, ctx
    )
    var bf16_result = harness.compare(actual_bf16, expected_bf16, ctx)
    print("head-broadcast f32: ", f32_result)
    print("head-broadcast bf16: ", bf16_result)
    if not f32_result.passed or not bf16_result.passed:
        raise Error("SCAIL-2 compact head-broadcast RoPE gate failed")


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: scail2_rope_parity.mojo FIXTURE_DIR")
    var fixture_dir = String(args[1])
    var ctx = DeviceContext()
    _gate_case(
        "animation_add2",
        Scail2SequencePlan(
            video_t=3, grid_h=4, grid_w=6,
            additional_ref_count=2, replace_flag=False,
        ),
        fixture_dir,
        ctx,
    )
    _gate_case(
        "replacement",
        Scail2SequencePlan(
            video_t=3, grid_h=4, grid_w=6,
            additional_ref_count=0, replace_flag=True,
        ),
        fixture_dir,
        ctx,
    )
    _gate_head_broadcast(ctx)
    print("SCAIL-2 sequence/RoPE parity PASS")
