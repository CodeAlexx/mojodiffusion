# plan_smoke.mojo - compile-only gate for shared offload planning.

from serenitymojo.offload.plan import (
    OffloadConfig,
    build_hidream_o1_block_plan,
    build_klein9b_block_plan,
    build_lance_t2v_block_plan,
    build_qwenimage_block_plan,
    build_sensenova_u1_block_plan,
)


def _check(name: String, got: Int, expected: Int) raises:
    print("[offload-plan]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("offload plan mismatch: ") + name)


def main() raises:
    var klein = build_klein9b_block_plan()
    var qwen = build_qwenimage_block_plan()
    var lance = build_lance_t2v_block_plan()
    var hidream = build_hidream_o1_block_plan()
    var sensenova = build_sensenova_u1_block_plan()
    var single = OffloadConfig.synchronous_single()
    var cfg = OffloadConfig.synchronous_cfg_paired()

    _check(String("klein block count"), klein.count(), 32)
    _check(String("klein cfg visits"), klein.branch_visits(cfg), 64)
    _check(String("klein single visits"), klein.branch_visits(single), 32)
    _check(String("qwen block count"), qwen.count(), 60)
    _check(String("qwen cfg visits"), qwen.branch_visits(cfg), 120)
    _check(String("qwen tensor hint"), qwen.total_tensor_count_hint(), 1920)
    _check(String("lance block count"), lance.count(), 36)
    _check(String("hidream block count"), hidream.count(), 36)
    _check(String("sensenova block count"), sensenova.count(), 42)

    print("[offload-plan] klein first:", klein.normalized_prefix(0), klein.kind(0).name())
    print("[offload-plan] klein last:", klein.normalized_prefix(klein.count() - 1), klein.kind(klein.count() - 1).name())
    print("[offload-plan] qwen first:", qwen.normalized_prefix(0), qwen.kind(0).name())
    print("[offload-plan] qwen last:", qwen.normalized_prefix(qwen.count() - 1), qwen.kind(qwen.count() - 1).name())
    print("[offload-plan] lance first:", lance.normalized_prefix(0), lance.kind(0).name())
    print("[offload-plan] hidream first:", hidream.normalized_prefix(0), hidream.kind(0).name())
    print("[offload-plan] sensenova first:", sensenova.normalized_prefix(0), sensenova.kind(0).name())
    print("[offload-plan] lance prefetch from 0:", lance.prefetch_index(0, single))
    print("[offload-plan] lance prefetch from last:", lance.prefetch_index(lance.count() - 1, single))
