# training/benchmark_matrix.mojo — canonical cross-model trainer speed matrix.

from serenitymojo.training.perf_record import (
    PERF_LANE_AI_TOOLKIT,
    PERF_LANE_ONETRAINER,
    training_perf_lane_name,
)


@fieldwise_init
struct TrainingBenchmarkCase(Copyable, Movable, Writable):
    var model: String
    var correctness_lane: Int
    var mojo_runner: String
    var reference_path_hint: String
    var architecture_family: String
    var default_dtype: String
    var target_batch: Int
    var target_resolution: String
    var required_flags: String
    var rust_flame_scope: String

    def validate(self) raises:
        if self.model == String(""):
            raise Error("TrainingBenchmarkCase: model is required")
        if self.mojo_runner == String(""):
            raise Error("TrainingBenchmarkCase: mojo runner is required")
        if self.reference_path_hint == String(""):
            raise Error("TrainingBenchmarkCase: reference path hint is required")
        if self.architecture_family == String(""):
            raise Error("TrainingBenchmarkCase: architecture family is required")
        if self.default_dtype == String(""):
            raise Error("TrainingBenchmarkCase: dtype is required")
        if self.target_batch <= 0:
            raise Error("TrainingBenchmarkCase: target batch must be positive")
        if self.target_resolution == String(""):
            raise Error("TrainingBenchmarkCase: resolution is required")
        if self.rust_flame_scope == String(""):
            raise Error("TrainingBenchmarkCase: rust/flame scope is required")

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "TrainingBenchmarkCase(model=",
            self.model,
            ", correctness_lane=",
            training_perf_lane_name(self.correctness_lane),
            ", runner=",
            self.mojo_runner,
            ", family=",
            self.architecture_family,
            ", dtype=",
            self.default_dtype,
            ", batch=",
            self.target_batch,
            ", resolution=",
            self.target_resolution,
            ")",
        )


def training_benchmark_matrix_size() -> Int:
    return 4


def training_benchmark_case(index: Int) raises -> TrainingBenchmarkCase:
    if index == 0:
        return TrainingBenchmarkCase(
            String("krea2"),
            PERF_LANE_AI_TOOLKIT,
            String("serenitymojo/models/krea2/train_krea2.mojo"),
            String("/home/alex/ai-toolkit or local ai-toolkit-origin Krea2 artifacts"),
            String("single-stream DiT LoRA"),
            String("BF16"),
            1,
            String("512 or 1024"),
            String("length-bucket-pad,flow-mse,cudnn-padmask-eligible"),
            String("op/block speed only; not a convergence oracle"),
        )
    if index == 1:
        return TrainingBenchmarkCase(
            String("zimage"),
            PERF_LANE_ONETRAINER,
            String("serenitymojo/models/zimage/train.mojo"),
            String("/home/alex/OneTrainer one-step and 100-step ZImage artifacts"),
            String("large transformer LoRA"),
            String("BF16"),
            2,
            String("1024"),
            String("OneTrainer-primary,batch-2,device-grads"),
            String("SDPA/block microkernels only"),
        )
    if index == 2:
        return TrainingBenchmarkCase(
            String("klein"),
            PERF_LANE_ONETRAINER,
            String("serenitymojo/training/train_klein_cadence.mojo"),
            String("/home/alex/OneTrainer Klein artifacts"),
            String("offloaded DiT LoRA"),
            String("BF16"),
            1,
            String("1024"),
            String("offload,cadence,sampler-separate-process"),
            String("offload and SDPA block speed only"),
        )
    if index == 3:
        return TrainingBenchmarkCase(
            String("sdxl"),
            PERF_LANE_ONETRAINER,
            String("serenitymojo/models/sdxl/sdxl_real_train.mojo"),
            String("/home/alex/OneTrainer SDXL or UNet-family artifacts"),
            String("UNet/cross-attention LoRA"),
            String("BF16"),
            1,
            String("1024"),
            String("non-transformer-check,rectangular-attention"),
            String("op speed only; catches transformer-only assumptions"),
        )
    raise Error("training_benchmark_case: index out of range")
