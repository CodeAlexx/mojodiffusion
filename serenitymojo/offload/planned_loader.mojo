# planned_loader.mojo - BlockPlan-aware wrapper over BlockLoader.
#
# This is the first runner-facing API above the raw prefix loader. It keeps the
# current synchronous mmap/H2D backend, but moves block order, lookahead,
# branch accounting, and dtype policy into one shared object.

from std.gpu.host import DeviceContext

from serenitymojo.offload.block_loader import Block, BlockLoader
from serenitymojo.offload.plan import BlockPlan, DTypePolicy, OffloadConfig


struct PlannedOffloadStats(Copyable, Movable, ImplicitlyCopyable):
    var prefetch_calls: Int
    var load_calls: Int
    var branch_visits: Int
    var blocks_seen: Int

    def __init__(out self):
        self.prefetch_calls = 0
        self.load_calls = 0
        self.branch_visits = 0
        self.blocks_seen = 0

    def __init__(
        out self,
        prefetch_calls: Int,
        load_calls: Int,
        branch_visits: Int,
        blocks_seen: Int,
    ):
        self.prefetch_calls = prefetch_calls
        self.load_calls = load_calls
        self.branch_visits = branch_visits
        self.blocks_seen = blocks_seen


struct PlannedBlockHandle(Movable):
    var index: Int
    var prefix: String
    var block: Block

    def __init__(out self, index: Int, prefix: String, var block: Block):
        self.index = index
        self.prefix = prefix
        self.block = block^

    def tensor_count(self) -> Int:
        return len(self.block)


struct PlannedBlockLoader(Movable):
    var loader: BlockLoader
    var plan: BlockPlan
    var config: OffloadConfig
    var stats: PlannedOffloadStats

    @staticmethod
    def open(dir: String, var plan: BlockPlan, config: OffloadConfig) raises -> PlannedBlockLoader:
        var loader = BlockLoader.open(dir)
        return PlannedBlockLoader(loader^, plan^, config)

    def __init__(
        out self,
        var loader: BlockLoader,
        var plan: BlockPlan,
        config: OffloadConfig,
    ):
        self.loader = loader^
        self.plan = plan^
        self.config = config
        self.stats = PlannedOffloadStats()

    def count(self) -> Int:
        return self.plan.count()

    def block_count(self) -> Int:
        return self.count()

    def branch_visits(self) -> Int:
        return self.plan.branch_visits(self.config)

    def pinned_bytes(self) -> Int:
        return 0

    def prefetch_index(self, index: Int) -> Int:
        return self.plan.prefetch_index(index, self.config)

    def prefetch(mut self, index: Int) raises:
        if index < 0 or index >= self.plan.count():
            return
        self.loader.prefetch_block(self.plan.normalized_prefix(index))
        self.stats.prefetch_calls += 1

    def prefetch_next(mut self, index: Int) raises:
        self.prefetch(self.prefetch_index(index))

    def await_block(mut self, index: Int, ctx: DeviceContext) raises -> PlannedBlockHandle:
        var prefix = self.plan.prefix(index)
        var load_prefix = self.plan.normalized_prefix(index)
        if self.config.dtype_policy == DTypePolicy.force_bf16():
            var block = self.loader.load_block_as_bf16(load_prefix, ctx)
            self.stats.load_calls += 1
            self.stats.blocks_seen += 1
            self.stats.branch_visits += self.config.branch_schedule.branch_count()
            return PlannedBlockHandle(index, prefix, block^)

        var block = self.loader.load_block(load_prefix, ctx)
        self.stats.load_calls += 1
        self.stats.blocks_seen += 1
        self.stats.branch_visits += self.config.branch_schedule.branch_count()
        return PlannedBlockHandle(index, prefix, block^)

    def snapshot_stats(self) -> PlannedOffloadStats:
        return self.stats
