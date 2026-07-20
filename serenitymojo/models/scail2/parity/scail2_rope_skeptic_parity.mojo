# Independent adversarial gate for SCAIL-2 sequence layout and RoPE tables.
# Fixtures come only from the hash-pinned official oracle in
# scail2_rope_skeptic_oracle.py and are separately hash-gated by
# FIXTURES.sha256.

from std.gpu.host import DeviceContext
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


def _join(dir: String, name: String) -> String:
    return dir + "/" + name


def _read_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open skeptic fixture: ") + path)
    var nbytes = file_size(fd)
    if nbytes <= 0 or nbytes % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("invalid skeptic fixture: ") + path)
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
        raise Error(String("short skeptic fixture read: ") + path)
    var values = List[Float32]()
    var fp = buf.bitcast[Float32]()
    for i in range(nbytes // 4):
        values.append(fp[i])
    buf.free()
    return values^


def _require_pass(label: String, passed: Bool) raises:
    if not passed:
        raise Error(String("SCAIL-2 skeptic gate failed: ") + label)


def _gate_case(
    name: String,
    plan: Scail2SequencePlan,
    expected_rows: Int,
    fixture_dir: String,
    ctx: DeviceContext,
) raises:
    if plan.sequence_length() != expected_rows:
        raise Error(String("sequence-length accounting mismatch: ") + name)
    var positions_ref = _read_f32(_join(fixture_dir, name + "_positions.f32"))
    var cos_ref = _read_f32(_join(fixture_dir, name + "_cos.f32"))
    var sin_ref = _read_f32(_join(fixture_dir, name + "_sin.f32"))
    if len(positions_ref) != expected_rows * 4:
        raise Error(String("position fixture shape mismatch: ") + name)
    if len(cos_ref) != expected_rows * 64 or len(sin_ref) != expected_rows * 64:
        raise Error(String("RoPE fixture shape mismatch: ") + name)

    var positions = build_scail2_position_descriptors(plan, ctx)
    var f32_tables = build_scail2_rope_tables(plan, ctx, STDtype.F32)
    var bf16_tables = build_scail2_rope_tables(plan, ctx, STDtype.BF16)
    var f16_tables = build_scail2_rope_tables(plan, ctx, STDtype.F16)
    var strict = ParityHarness(0.999)
    var pos_result = strict.compare(positions, positions_ref, ctx)
    var cos_result = strict.compare(f32_tables[0], cos_ref, ctx)
    var sin_result = strict.compare(f32_tables[1], sin_ref, ctx)
    var cos_bf16_result = strict.compare(bf16_tables[0], cos_ref, ctx)
    var sin_bf16_result = strict.compare(bf16_tables[1], sin_ref, ctx)
    var cos_f16_result = strict.compare(f16_tables[0], cos_ref, ctx)
    var sin_f16_result = strict.compare(f16_tables[1], sin_ref, ctx)
    print(name, " positions: ", pos_result)
    print(name, " cos f32: ", cos_result)
    print(name, " sin f32: ", sin_result)
    print(name, " cos bf16: ", cos_bf16_result)
    print(name, " sin bf16: ", sin_bf16_result)
    print(name, " cos f16: ", cos_f16_result)
    print(name, " sin f16: ", sin_f16_result)
    _require_pass(name + " positions", pos_result.passed)
    _require_pass(name + " cos f32", cos_result.passed)
    _require_pass(name + " sin f32", sin_result.passed)
    _require_pass(name + " cos bf16", cos_bf16_result.passed)
    _require_pass(name + " sin bf16", sin_bf16_result.passed)
    _require_pass(name + " cos f16", cos_f16_result.passed)
    _require_pass(name + " sin f16", sin_f16_result.passed)


def _expect_invalid(label: String, plan: Scail2SequencePlan) raises:
    var raised = False
    try:
        plan.validate()
    except e:
        raised = True
    if not raised:
        raise Error(String("SCAIL-2 accepted invalid plan: ") + label)


def _expect_valid(label: String, plan: Scail2SequencePlan) raises:
    try:
        plan.validate()
    except e:
        raise Error(
            String("SCAIL-2 rejected valid boundary plan: ") + label
            + String(": ") + String(e)
        )


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: scail2_rope_skeptic_parity.mojo FIXTURE_DIR")
    var fixture_dir = String(args[1])
    var ctx = DeviceContext()

    # 0 + 4 + 8 + 2 = 14 rows. This exercises the exact pose W=120/121
    # boundary and verifies that no optional-prefix branch is required.
    _gate_case(
        "animation_zero_ref_min_pose",
        Scail2SequencePlan(
            video_t=2, grid_h=2, grid_w=2,
            additional_ref_count=0, replace_flag=False,
        ),
        14,
        fixture_dir,
        ctx,
    )

    # 3*48 + 48 + 2*48 + 2*3*4 = 312 rows. This exercises replacement
    # H=120..125 and three distinct optional-reference temporal positions.
    _gate_case(
        "replacement_add3",
        Scail2SequencePlan(
            video_t=2, grid_h=6, grid_w=8,
            additional_ref_count=3, replace_flag=True,
        ),
        312,
        fixture_dir,
        ctx,
    )

    _expect_invalid(
        "zero temporal length",
        Scail2SequencePlan(
            video_t=0, grid_h=2, grid_w=2,
            additional_ref_count=0, replace_flag=False,
        ),
    )
    _expect_invalid(
        "odd pose grid",
        Scail2SequencePlan(
            video_t=1, grid_h=3, grid_w=2,
            additional_ref_count=0, replace_flag=False,
        ),
    )
    _expect_invalid(
        "negative optional-reference count",
        Scail2SequencePlan(
            video_t=1, grid_h=2, grid_w=2,
            additional_ref_count=-1, replace_flag=False,
        ),
    )
    _expect_invalid(
        "animation temporal shift overflow",
        Scail2SequencePlan(
            video_t=8192, grid_h=2, grid_w=2,
            additional_ref_count=0, replace_flag=False,
        ),
    )
    _expect_invalid(
        "replacement temporal/additional-ref overflow",
        Scail2SequencePlan(
            video_t=8190, grid_h=2, grid_w=2,
            additional_ref_count=3, replace_flag=True,
        ),
    )
    _expect_invalid(
        "pose width shift overflow",
        Scail2SequencePlan(
            video_t=1, grid_h=2, grid_w=8074,
            additional_ref_count=0, replace_flag=False,
        ),
    )
    _expect_invalid(
        "replacement height shift overflow",
        Scail2SequencePlan(
            video_t=1, grid_h=8074, grid_w=2,
            additional_ref_count=0, replace_flag=True,
        ),
    )
    _expect_invalid(
        "primary reference temporal shift overflow",
        Scail2SequencePlan(
            video_t=1, grid_h=2, grid_w=2,
            additional_ref_count=8192, replace_flag=True,
        ),
    )
    _expect_valid(
        "animation shifted-axis maxima",
        Scail2SequencePlan(
            video_t=1, grid_h=8192, grid_w=8072,
            additional_ref_count=8190, replace_flag=False,
        ),
    )
    _expect_valid(
        "replacement shifted-axis maxima",
        Scail2SequencePlan(
            video_t=1, grid_h=8072, grid_w=8072,
            additional_ref_count=8191, replace_flag=True,
        ),
    )
    print("SCAIL-2 adversarial sequence/RoPE skeptic PASS")
