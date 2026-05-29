# execution_config.mojo - runtime knobs shared by modular pipeline wrappers.


@fieldwise_init
struct PrecisionMode(Copyable, Movable, ImplicitlyCopyable, Equatable):
    var tag: Int

    @staticmethod
    def bf16() -> PrecisionMode:
        return PrecisionMode(0)

    @staticmethod
    def f16() -> PrecisionMode:
        return PrecisionMode(1)

    @staticmethod
    def f32() -> PrecisionMode:
        return PrecisionMode(2)

    def name(self) -> String:
        if self.tag == 0:
            return "bf16"
        if self.tag == 1:
            return "f16"
        if self.tag == 2:
            return "f32"
        return "unknown"


@fieldwise_init
struct OffloadMode(Copyable, Movable, ImplicitlyCopyable, Equatable):
    var tag: Int

    @staticmethod
    def resident() -> OffloadMode:
        return OffloadMode(0)

    @staticmethod
    def block_stream() -> OffloadMode:
        return OffloadMode(1)

    @staticmethod
    def turbo_slots() -> OffloadMode:
        return OffloadMode(2)

    def name(self) -> String:
        if self.tag == 0:
            return "resident"
        if self.tag == 1:
            return "block_stream"
        if self.tag == 2:
            return "turbo_slots"
        return "unknown"


@fieldwise_init
struct ExecutionConfig(Copyable, Movable, ImplicitlyCopyable):
    var steps: Int
    var seed: UInt64
    var guidance_scale: Float32
    var precision: PrecisionMode
    var offload: OffloadMode
    var artifact_root: String
    var allow_gpu_heavy_validation: Bool


def default_smoke_config() -> ExecutionConfig:
    return ExecutionConfig(
        1,
        UInt64(42),
        Float32(1.0),
        PrecisionMode.bf16(),
        OffloadMode.block_stream(),
        String("/home/alex/mojodiffusion/output"),
        False,
    )


def default_quality_config() -> ExecutionConfig:
    return ExecutionConfig(
        20,
        UInt64(42),
        Float32(4.0),
        PrecisionMode.bf16(),
        OffloadMode.block_stream(),
        String("/home/alex/mojodiffusion/output"),
        True,
    )

