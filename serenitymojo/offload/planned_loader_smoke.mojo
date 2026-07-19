# planned_loader_smoke.mojo - compile gate for the BlockPlan-aware loader API.
#
# This smoke intentionally avoids opening checkpoints or loading tensors. Importing
# PlannedBlockLoader still typechecks the real load path.

from serenitymojo.offload.plan import (
    OffloadConfig,
    build_klein9b_block_plan,
    build_lance_t2v_block_plan,
)
from serenitymojo.offload.planned_loader import PlannedBlockLoader, PlannedOffloadStats


def _check(name: String, got: Int, expected: Int) raises:
    print("[planned-loader]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("planned loader mismatch: ") + name)


def main() raises:
    var klein = build_klein9b_block_plan()
    var lance = build_lance_t2v_block_plan()
    var single = OffloadConfig.single_pass()
    var cfg = OffloadConfig.synchronous_cfg_paired()
    var bf16 = OffloadConfig.bf16_cfg_paired()
    var bf16_single = OffloadConfig.bf16_single()
    var stats = PlannedOffloadStats()

    _check(String("zero prefetch calls"), stats.prefetch_calls, 0)
    _check(String("zero load calls"), stats.load_calls, 0)
    _check(String("klein blocks"), klein.count(), 32)
    _check(String("klein cfg branch visits"), klein.branch_visits(cfg), 64)
    _check(String("klein single branch visits"), klein.branch_visits(single), 32)
    _check(String("klein first lookahead"), klein.prefetch_index(0, cfg), 1)
    _check(String("klein last lookahead"), klein.prefetch_index(klein.count() - 1, cfg), -1)
    _check(String("lance bf16 branch visits"), lance.branch_visits(bf16), 72)
    _check(String("lance bf16 single visits"), lance.branch_visits(bf16_single), 36)

    print("[planned-loader] klein first:", klein.normalized_prefix(0))
    print("[planned-loader] klein last:", klein.normalized_prefix(klein.count() - 1))
    print("[planned-loader] lance first:", lance.normalized_prefix(0))
    print("[planned-loader] dtype policy:", bf16.dtype_policy.name())
    print("[planned-loader] block_count/pinned_bytes names compile")
