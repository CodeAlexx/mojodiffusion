# plan.mojo - metadata-only offload block planning.
#
# This is the first shared layer above BlockLoader. It does not load tensors or
# allocate GPU memory; it describes block order, branch scheduling, dtype policy,
# and lookahead. Existing model loops can adopt this without changing math.


@fieldwise_init
struct BlockKind(Copyable, Movable, ImplicitlyCopyable, Equatable):
    var tag: Int

    @staticmethod
    def transformer() -> BlockKind:
        return BlockKind(0)

    @staticmethod
    def double_stream() -> BlockKind:
        return BlockKind(1)

    @staticmethod
    def single_stream() -> BlockKind:
        return BlockKind(2)

    @staticmethod
    def unet_down() -> BlockKind:
        return BlockKind(3)

    @staticmethod
    def unet_mid() -> BlockKind:
        return BlockKind(4)

    @staticmethod
    def unet_up() -> BlockKind:
        return BlockKind(5)

    def name(self) -> String:
        if self.tag == 0:
            return "transformer"
        if self.tag == 1:
            return "double_stream"
        if self.tag == 2:
            return "single_stream"
        if self.tag == 3:
            return "unet_down"
        if self.tag == 4:
            return "unet_mid"
        if self.tag == 5:
            return "unet_up"
        return "unknown"


@fieldwise_init
struct DTypePolicy(Copyable, Movable, ImplicitlyCopyable, Equatable):
    var tag: Int

    @staticmethod
    def preserve() -> DTypePolicy:
        return DTypePolicy(0)

    @staticmethod
    def force_bf16() -> DTypePolicy:
        return DTypePolicy(1)

    def name(self) -> String:
        if self.tag == 0:
            return "preserve"
        if self.tag == 1:
            return "force_bf16"
        return "unknown"


@fieldwise_init
struct BranchSchedule(Copyable, Movable, ImplicitlyCopyable, Equatable):
    var tag: Int

    @staticmethod
    def single() -> BranchSchedule:
        return BranchSchedule(0)

    @staticmethod
    def cfg_paired() -> BranchSchedule:
        return BranchSchedule(1)

    def branch_count(self) -> Int:
        if self.tag == 1:
            return 2
        return 1

    def name(self) -> String:
        if self.tag == 0:
            return "single"
        if self.tag == 1:
            return "cfg_paired"
        return "unknown"


@fieldwise_init
struct OffloadConfig(Copyable, Movable, ImplicitlyCopyable):
    var slot_count: Int
    var lookahead: Int
    var dtype_policy: DTypePolicy
    var branch_schedule: BranchSchedule

    @staticmethod
    def synchronous_single() -> OffloadConfig:
        return OffloadConfig(1, 1, DTypePolicy.preserve(), BranchSchedule.single())

    @staticmethod
    def synchronous_cfg_paired() -> OffloadConfig:
        return OffloadConfig(1, 1, DTypePolicy.preserve(), BranchSchedule.cfg_paired())

    @staticmethod
    def bf16_cfg_paired() -> OffloadConfig:
        return OffloadConfig(1, 1, DTypePolicy.force_bf16(), BranchSchedule.cfg_paired())

    @staticmethod
    def bf16_single() -> OffloadConfig:
        return OffloadConfig(1, 1, DTypePolicy.force_bf16(), BranchSchedule.single())


@fieldwise_init
struct BlockRecord(Copyable, Movable):
    var prefix: String
    var kind: BlockKind
    var tensor_count_hint: Int
    var byte_count_hint: Int


struct BlockPlan(Movable):
    var name: String
    var records: List[BlockRecord]

    def __init__(out self, name: String):
        self.name = name
        self.records = List[BlockRecord]()

    def append(
        mut self,
        prefix: String,
        kind: BlockKind,
        tensor_count_hint: Int = 0,
        byte_count_hint: Int = 0,
    ):
        self.records.append(BlockRecord(prefix, kind, tensor_count_hint, byte_count_hint))

    def count(self) -> Int:
        return len(self.records)

    def prefix(self, index: Int) raises -> String:
        if index < 0 or index >= len(self.records):
            raise Error("BlockPlan.prefix: index out of range")
        return self.records[index].prefix

    def normalized_prefix(self, index: Int) raises -> String:
        var p = self.prefix(index)
        return p if p.endswith(".") else p + "."

    def kind(self, index: Int) raises -> BlockKind:
        if index < 0 or index >= len(self.records):
            raise Error("BlockPlan.kind: index out of range")
        return self.records[index].kind

    def total_tensor_count_hint(self) -> Int:
        var total = 0
        for i in range(len(self.records)):
            total += self.records[i].tensor_count_hint
        return total

    def total_byte_count_hint(self) -> Int:
        var total = 0
        for i in range(len(self.records)):
            total += self.records[i].byte_count_hint
        return total

    def branch_visits(self, config: OffloadConfig) -> Int:
        return len(self.records) * config.branch_schedule.branch_count()

    def prefetch_index(self, index: Int, config: OffloadConfig) -> Int:
        var next = index + config.lookahead
        if next >= 0 and next < len(self.records):
            return next
        return -1


def build_klein_block_plan(num_double: Int, num_single: Int) -> BlockPlan:
    var plan = BlockPlan(String("klein"))
    for i in range(num_double):
        plan.append(
            String("double_blocks.") + String(i),
            BlockKind.double_stream(),
        )
    for i in range(num_single):
        plan.append(
            String("single_blocks.") + String(i),
            BlockKind.single_stream(),
        )
    return plan^


def build_klein9b_block_plan() -> BlockPlan:
    return build_klein_block_plan(8, 24)


def build_qwenimage_block_plan() -> BlockPlan:
    var plan = BlockPlan(String("qwen_image"))
    for i in range(60):
        plan.append(
            String("transformer_blocks.") + String(i),
            BlockKind.double_stream(),
            32,
            679662592,
        )
    return plan^


def build_lance_t2v_block_plan() -> BlockPlan:
    var plan = BlockPlan(String("lance_t2v"))
    for i in range(36):
        plan.append(
            String("language_model.model.layers.") + String(i),
            BlockKind.transformer(),
        )
    return plan^


def build_hidream_o1_block_plan() -> BlockPlan:
    var plan = BlockPlan(String("hidream_o1"))
    for i in range(36):
        plan.append(
            String("model.language_model.layers.") + String(i),
            BlockKind.transformer(),
        )
    return plan^


def build_sensenova_u1_block_plan() -> BlockPlan:
    var plan = BlockPlan(String("sensenova_u1"))
    for i in range(42):
        plan.append(
            String("language_model.model.layers.") + String(i),
            BlockKind.transformer(),
        )
    return plan^


def build_flux_block_plan(num_double: Int, num_single: Int) -> BlockPlan:
    # flux1-dev block order (BFL keys, verified against
    # flux1-dev.safetensors header): 19 double_blocks.<i> then 38
    # single_blocks.<i>. Mirrors build_klein_block_plan (same double/single
    # DiT topology); the prefixes match the on-disk safetensors layout so the
    # offload loader can stream one block at a time. flux1-dev default is
    # (num_double=19, num_single=38) — see configs/flux.json / config.mojo.
    var plan = BlockPlan(String("flux"))
    for i in range(num_double):
        plan.append(
            String("double_blocks.") + String(i),
            BlockKind.double_stream(),
        )
    for i in range(num_single):
        plan.append(
            String("single_blocks.") + String(i),
            BlockKind.single_stream(),
        )
    return plan^


def build_flux1_dev_block_plan() -> BlockPlan:
    return build_flux_block_plan(19, 38)
