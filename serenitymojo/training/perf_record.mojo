# training/perf_record.mojo — shared trainer speed scorecard schema.
#
# This is intentionally a contract module, not a profiler. Product trainers fill
# this record from their measured loop/profiler data; parity and speed gates can
# then compare OneTrainer/ai-toolkit, Mojo, and Rust/Flame lanes without each
# model inventing its own log format.

comptime PERF_LANE_ONETRAINER = 0
comptime PERF_LANE_AI_TOOLKIT = 1
comptime PERF_LANE_MOJO_CURRENT = 2
comptime PERF_LANE_RUST_FLAME_OP = 3

comptime PERF_FAST_PATH_DEVICE = 0
comptime PERF_FAST_PATH_HOST_GRAD_COMPAT = 1


@fieldwise_init
struct TrainingPhaseTimings(Copyable, Movable, Writable):
    var forward_seconds: Float64
    var backward_seconds: Float64
    var loss_seconds: Float64
    var grad_norm_seconds: Float64
    var clip_seconds: Float64
    var optimizer_seconds: Float64
    var save_seconds: Float64
    var sample_seconds: Float64

    def total_known_seconds(self) -> Float64:
        return (
            self.forward_seconds
            + self.backward_seconds
            + self.loss_seconds
            + self.grad_norm_seconds
            + self.clip_seconds
            + self.optimizer_seconds
            + self.save_seconds
            + self.sample_seconds
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "TrainingPhaseTimings(fwd=",
            self.forward_seconds,
            ", bwd=",
            self.backward_seconds,
            ", loss=",
            self.loss_seconds,
            ", norm=",
            self.grad_norm_seconds,
            ", clip=",
            self.clip_seconds,
            ", opt=",
            self.optimizer_seconds,
            ", save=",
            self.save_seconds,
            ", sample=",
            self.sample_seconds,
            ")",
        )


def empty_training_phase_timings() -> TrainingPhaseTimings:
    return TrainingPhaseTimings(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@fieldwise_init
struct TrainingPerfRecord(Movable, Writable):
    var model: String
    var lane: Int
    var preset_config_hash: String
    var dtype: String
    var rank: Int
    var batch: Int
    var resolution: String
    var optimizer: String
    var enabled_flags: String
    var warmup_steps: Int
    var measured_steps: Int
    var total_seconds_per_step: Float64
    var phases: TrainingPhaseTimings
    var peak_vram_bytes: Int
    var host_device_transfer_count: Int
    var full_tensor_readback_count: Int
    var sync_count: Int
    var fast_path_kind: Int
    var attention_backend: String
    var profiler_artifact_path: String

    def is_device_fast_path(self) -> Bool:
        return (
            self.fast_path_kind == PERF_FAST_PATH_DEVICE
            and self.full_tensor_readback_count == 0
        )

    def lane_name(self) -> String:
        return training_perf_lane_name(self.lane)

    def validate(self) raises:
        if self.model == String(""):
            raise Error("TrainingPerfRecord: model is required")
        if self.preset_config_hash == String(""):
            raise Error("TrainingPerfRecord: preset/config hash is required")
        if self.dtype == String(""):
            raise Error("TrainingPerfRecord: dtype is required")
        if self.optimizer == String(""):
            raise Error("TrainingPerfRecord: optimizer is required")
        if self.rank < 0:
            raise Error("TrainingPerfRecord: rank must be nonnegative")
        if self.batch <= 0:
            raise Error("TrainingPerfRecord: batch must be positive")
        if self.resolution == String(""):
            raise Error("TrainingPerfRecord: resolution is required")
        if self.warmup_steps < 0:
            raise Error("TrainingPerfRecord: warmup_steps must be nonnegative")
        if self.measured_steps <= 0:
            raise Error("TrainingPerfRecord: measured_steps must be positive")
        if self.total_seconds_per_step <= 0.0:
            raise Error("TrainingPerfRecord: seconds/step must be positive")
        if self.peak_vram_bytes < 0:
            raise Error("TrainingPerfRecord: peak_vram_bytes must be measured or zero")
        if self.host_device_transfer_count < 0:
            raise Error("TrainingPerfRecord: transfer count must be nonnegative")
        if self.full_tensor_readback_count < 0:
            raise Error("TrainingPerfRecord: full tensor readback count must be nonnegative")
        if self.sync_count < 0:
            raise Error("TrainingPerfRecord: sync count must be nonnegative")
        if self.fast_path_kind != PERF_FAST_PATH_DEVICE and self.fast_path_kind != PERF_FAST_PATH_HOST_GRAD_COMPAT:
            raise Error("TrainingPerfRecord: invalid fast_path_kind")
        if self.fast_path_kind == PERF_FAST_PATH_DEVICE and self.full_tensor_readback_count != 0:
            raise Error("TrainingPerfRecord: device fast path cannot include full tensor readbacks")
    def summary(self) -> String:
        return (
            String("perf model=")
            + self.model
            + String(" lane=")
            + self.lane_name()
            + String(" dtype=")
            + self.dtype
            + String(" rank=")
            + String(self.rank)
            + String(" batch=")
            + String(self.batch)
            + String(" res=")
            + self.resolution
            + String(" opt=")
            + self.optimizer
            + String(" sec_per_step=")
            + String(self.total_seconds_per_step)
            + String(" peak_vram_bytes=")
            + String(self.peak_vram_bytes)
            + String(" transfers=")
            + String(self.host_device_transfer_count)
            + String(" full_readbacks=")
            + String(self.full_tensor_readback_count)
            + String(" syncs=")
            + String(self.sync_count)
            + String(" attn=")
            + self.attention_backend
        )

    def to_jsonl(self) raises -> String:
        self.validate()
        var out = String("{")
        out += _json_pair("model", self.model) + String(",")
        out += _json_pair("lane", self.lane_name()) + String(",")
        out += _json_pair("preset_config_hash", self.preset_config_hash) + String(",")
        out += _json_pair("dtype", self.dtype) + String(",")
        out += _json_int_pair("rank", self.rank) + String(",")
        out += _json_int_pair("batch", self.batch) + String(",")
        out += _json_pair("resolution", self.resolution) + String(",")
        out += _json_pair("optimizer", self.optimizer) + String(",")
        out += _json_pair("enabled_flags", self.enabled_flags) + String(",")
        out += _json_int_pair("warmup_steps", self.warmup_steps) + String(",")
        out += _json_int_pair("measured_steps", self.measured_steps) + String(",")
        out += _json_float_pair("total_seconds_per_step", self.total_seconds_per_step) + String(",")
        out += String('"phases":{')
        out += _json_float_pair("forward_seconds", self.phases.forward_seconds) + String(",")
        out += _json_float_pair("backward_seconds", self.phases.backward_seconds) + String(",")
        out += _json_float_pair("loss_seconds", self.phases.loss_seconds) + String(",")
        out += _json_float_pair("grad_norm_seconds", self.phases.grad_norm_seconds) + String(",")
        out += _json_float_pair("clip_seconds", self.phases.clip_seconds) + String(",")
        out += _json_float_pair("optimizer_seconds", self.phases.optimizer_seconds) + String(",")
        out += _json_float_pair("save_seconds", self.phases.save_seconds) + String(",")
        out += _json_float_pair("sample_seconds", self.phases.sample_seconds)
        out += String("},")
        out += _json_int_pair("peak_vram_bytes", self.peak_vram_bytes) + String(",")
        out += _json_int_pair("host_device_transfer_count", self.host_device_transfer_count) + String(",")
        out += _json_int_pair("full_tensor_readback_count", self.full_tensor_readback_count) + String(",")
        out += _json_int_pair("sync_count", self.sync_count) + String(",")
        out += _json_pair("fast_path_kind", training_perf_fast_path_name(self.fast_path_kind)) + String(",")
        out += _json_pair("attention_backend", self.attention_backend) + String(",")
        out += _json_pair("profiler_artifact_path", self.profiler_artifact_path)
        out += String("}")
        return out^

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.summary())


def emit_training_perf_record(record: TrainingPerfRecord) raises:
    """Emit the machine-readable trainer speed record on stdout.

    Product trainers should call this after the measured loop. Keep the stable
    prefix so scripts can collect perf JSONL from normal operator logs.
    """
    print("[training-perf-json]", record.to_jsonl())


def training_perf_lane_name(lane: Int) -> String:
    if lane == PERF_LANE_ONETRAINER:
        return String("onetrainer")
    if lane == PERF_LANE_AI_TOOLKIT:
        return String("ai-toolkit")
    if lane == PERF_LANE_MOJO_CURRENT:
        return String("mojo-current")
    if lane == PERF_LANE_RUST_FLAME_OP:
        return String("rust-flame-op-reference")
    return String("unknown")


def training_perf_fast_path_name(kind: Int) -> String:
    if kind == PERF_FAST_PATH_DEVICE:
        return String("device")
    if kind == PERF_FAST_PATH_HOST_GRAD_COMPAT:
        return String("host-grad-compat-slow")
    return String("unknown")


def _json_string(value: String) raises -> String:
    var bytes = value.as_bytes()
    for i in range(value.byte_length()):
        var ch = bytes[i]
        if ch < 0x20 or ch == 0x22 or ch == 0x5C:
            raise Error("TrainingPerfRecord JSON string needs escaping; use simple artifact labels")
    return String('"') + value + String('"')


def _json_pair(key: String, value: String) raises -> String:
    return _json_string(key) + String(":") + _json_string(value)


def _json_int_pair(key: String, value: Int) raises -> String:
    return _json_string(key) + String(":") + String(value)


def _json_float_pair(key: String, value: Float64) raises -> String:
    return _json_string(key) + String(":") + String(value)
