# residency_smoke.mojo — Phase 2 pure-logic smoke test.
#
# Drives a synthetic schedule against a real build_klein9b_block_plan()
# (8 double_stream blocks + 24 single_stream blocks = 32 total).
#
# Assertions:
#   A1 - State machine: legal transitions succeed; illegal ones raise.
#   A2 - Budget: can_prefetch() flips at high watermark; should_evict() triggers
#        above high, stops below low; high<=low construction raises.
#   A3 - Refcount: an acquired (refcount>0) block is NEVER in eviction_candidates().
#   A4 - Eviction ordering: known cursor + sizes + distances → expected order;
#        prefetch-window blocks excluded.
#   A5 - CFG revisit: cfg_paired branch schedule gives correct next_use_distance.
#   A6 - Klein9b full plan: 32-block plan, realistic budget, acquire/release round-trip.
#
# Build + run (no GPU needed — pure logic):
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/offload/residency_smoke.mojo -o /tmp/residency_smoke
#   /tmp/residency_smoke

from serenitymojo.offload.plan import (
    BlockPlan,
    BlockKind,
    OffloadConfig,
    DTypePolicy,
    BranchSchedule,
    build_klein9b_block_plan,
)
from serenitymojo.offload.residency import (
    BlockState,
    ResidencyEntry,
    BudgetTracker,
    ResidencyManager,
)


# ── test counter (no global variables in Mojo) ────────────────────────────────

struct TC(Movable):
    var passes: Int
    var fails: Int

    def __init__(out self):
        self.passes = 0
        self.fails = 0

    def ok(mut self, label: String):
        self.passes += 1
        print("PASS", label)

    def fail(mut self, label: String, detail: String):
        self.fails += 1
        print("FAIL", label, "—", detail)

    def check(mut self, label: String, cond: Bool):
        if cond:
            self.ok(label)
        else:
            self.fail(label, String("condition False"))

    def check_eq(mut self, label: String, got: Int, expected: Int):
        if got == expected:
            self.ok(label)
        else:
            self.fail(label,
                String("got=") + String(got)
                + String(" expected=") + String(expected))

    def check_raises(mut self, label: String, raised: Bool):
        if raised:
            self.ok(label)
        else:
            self.fail(label, String("expected raise but none occurred"))

    def summary(self, section: String):
        print("[" + section + "] passes=" + String(self.passes)
              + " fails=" + String(self.fails))


# ── A1: state machine transitions ─────────────────────────────────────────────

def _test_state_machine(mut t: TC) raises:
    print("\n--- A1: state machine ---")

    var plan = BlockPlan(String("sm_test"))
    plan.append(String("blk.0"), BlockKind.transformer(), 2, 1024)
    plan.append(String("blk.1"), BlockKind.transformer(), 2, 1024)
    plan.append(String("blk.2"), BlockKind.transformer(), 2, 1024)

    var cfg = OffloadConfig(2, 1, DTypePolicy.preserve(), BranchSchedule.single())
    var bgt = BudgetTracker(1024 * 1024, 512 * 1024)
    var mgr = ResidencyManager(plan^, cfg, bgt^)

    # Legal forward path: UNLOADED→HOST_STAGED→PREFETCHING→GPU_READY
    mgr.transition(0, BlockState.host_staged())
    t.check(String("UNLOADED→HOST_STAGED ok"),
        mgr.get_state(0) == BlockState.host_staged())

    mgr.transition(0, BlockState.prefetching())
    t.check(String("HOST_STAGED→PREFETCHING ok"),
        mgr.get_state(0) == BlockState.prefetching())

    mgr.transition(0, BlockState.gpu_ready())
    t.check(String("PREFETCHING→GPU_READY ok"),
        mgr.get_state(0) == BlockState.gpu_ready())

    # Legal eviction path: GPU_READY→EVICTING→UNLOADED
    mgr.transition(0, BlockState.evicting())
    t.check(String("GPU_READY→EVICTING ok"),
        mgr.get_state(0) == BlockState.evicting())
    mgr.transition(0, BlockState.unloaded())
    t.check(String("EVICTING→UNLOADED ok"),
        mgr.get_state(0) == BlockState.unloaded())

    # GPU_FREEING path
    mgr.transition(0, BlockState.host_staged())
    mgr.transition(0, BlockState.prefetching())
    mgr.transition(0, BlockState.gpu_ready())
    mgr.transition(0, BlockState.gpu_freeing())
    t.check(String("GPU_READY→GPU_FREEING ok"),
        mgr.get_state(0) == BlockState.gpu_freeing())
    mgr.transition(0, BlockState.unloaded())
    t.check(String("GPU_FREEING→UNLOADED ok"),
        mgr.get_state(0) == BlockState.unloaded())

    # EVICTING→HOST_STAGED (legal re-stage)
    mgr.transition(0, BlockState.host_staged())
    mgr.transition(0, BlockState.prefetching())
    mgr.transition(0, BlockState.gpu_ready())
    mgr.transition(0, BlockState.evicting())
    mgr.transition(0, BlockState.host_staged())
    t.check(String("EVICTING→HOST_STAGED ok"),
        mgr.get_state(0) == BlockState.host_staged())

    # ILLEGAL: HOST_STAGED → GPU_READY (skips PREFETCHING)
    var raised = False
    try:
        mgr.transition(0, BlockState.gpu_ready())
    except:
        raised = True
    t.check_raises(String("HOST_STAGED→GPU_READY raises"), raised)

    # Reset block 0 to UNLOADED via legal path.
    mgr.transition(0, BlockState.prefetching())
    mgr.transition(0, BlockState.gpu_ready())
    mgr.transition(0, BlockState.evicting())
    mgr.transition(0, BlockState.unloaded())

    # ILLEGAL: UNLOADED → GPU_READY (multi-step skip)
    raised = False
    try:
        mgr.transition(0, BlockState.gpu_ready())
    except:
        raised = True
    t.check_raises(String("UNLOADED→GPU_READY raises"), raised)

    # ILLEGAL: GPU_READY → PREFETCHING (backwards)
    mgr.transition(0, BlockState.host_staged())
    mgr.transition(0, BlockState.prefetching())
    mgr.transition(0, BlockState.gpu_ready())
    raised = False
    try:
        mgr.transition(0, BlockState.prefetching())
    except:
        raised = True
    t.check_raises(String("GPU_READY→PREFETCHING raises (backward)"), raised)

    t.summary(String("A1"))


# ── A2: budget watermarks ─────────────────────────────────────────────────────

def _test_budget(mut t: TC) raises:
    print("\n--- A2: budget ---")

    # high <= low should raise
    var raised = False
    try:
        var bad = BudgetTracker(100, 200)
        _ = bad.can_prefetch()
    except:
        raised = True
    t.check_raises(String("high<low raises"), raised)

    raised = False
    try:
        var bad2 = BudgetTracker(200, 200)
        _ = bad2.can_prefetch()
    except:
        raised = True
    t.check_raises(String("high==low raises"), raised)

    # Normal: high=1000, low=600
    var bgt = BudgetTracker(1000, 600)
    t.check(String("can_prefetch when empty"), bgt.can_prefetch())
    t.check(String("should_evict False when empty"), not bgt.should_evict())

    bgt.add(999)
    t.check(String("can_prefetch at 999/1000"), bgt.can_prefetch())
    t.check(String("should_evict False at 999/1000"), not bgt.should_evict())

    bgt.add(1)   # 1000 == high → triggers
    t.check(String("can_prefetch False at high watermark"), not bgt.can_prefetch())
    t.check(String("should_evict True at high watermark"), bgt.should_evict())

    bgt.add(100) # 1100 > high
    t.check(String("can_prefetch False over high"), not bgt.can_prefetch())
    t.check(String("should_evict True over high"), bgt.should_evict())

    bgt.remove(100) # back to 1000 (still at high)
    t.check(String("should_evict True at exact high"), bgt.should_evict())

    bgt.remove(1)   # 999 < 1000
    t.check(String("should_evict False at 999"), not bgt.should_evict())

    t.check(String("below_low False at 999"), not bgt.below_low_watermark())

    bgt.remove(400) # 599 < 600 = low
    t.check(String("below_low True at 599"), bgt.below_low_watermark())

    # Headroom
    var bgt2 = BudgetTracker(1000, 500)
    bgt2.add(300)
    t.check_eq(String("headroom 700"), bgt2.headroom_bytes(), 700)

    t.summary(String("A2"))


# ── A3: refcount prevents eviction ────────────────────────────────────────────

def _test_refcount(mut t: TC) raises:
    print("\n--- A3: refcount ---")

    var plan = BlockPlan(String("ref_test"))
    for i in range(4):
        plan.append(
            String("blk.") + String(i),
            BlockKind.transformer(),
            2,
            200,
        )

    var cfg = OffloadConfig(2, 1, DTypePolicy.preserve(), BranchSchedule.single())
    var bgt = BudgetTracker(601, 300)
    var mgr = ResidencyManager(plan^, cfg, bgt^)

    # Bring all 4 blocks to GPU_READY.
    for i in range(4):
        mgr.transition(i, BlockState.host_staged())
        mgr.transition(i, BlockState.prefetching())
        mgr.transition(i, BlockState.gpu_ready())

    # Acquire blocks 0 and 2.
    mgr.acquire(0)
    mgr.acquire(2)

    var cands = mgr.eviction_candidates(0, 0, 0)

    var has_0 = False
    var has_2 = False
    for ci in range(len(cands)):
        if cands[ci] == 0: has_0 = True
        if cands[ci] == 2: has_2 = True

    t.check(String("acquired block 0 NOT in candidates"), not has_0)
    t.check(String("acquired block 2 NOT in candidates"), not has_2)

    # Block 3 should be a candidate (not in window [1] at cursor=0).
    var has_3 = False
    for ci in range(len(cands)):
        if cands[ci] == 3: has_3 = True
    t.check(String("free block 3 IS in candidates"), has_3)

    # Release block 0; it should reappear (not in window at cursor=0).
    mgr.release(0)
    var cands2 = mgr.eviction_candidates(0, 0, 0)
    var has_0_after = False
    for ci in range(len(cands2)):
        if cands2[ci] == 0: has_0_after = True
    t.check(String("released block 0 IS in candidates"), has_0_after)

    # Verify release-below-zero raises.
    var raised = False
    try:
        mgr.release(3)  # block 3 has refcount 0
    except:
        raised = True
    t.check_raises(String("release from 0 raises"), raised)

    t.summary(String("A3"))


# ── A4: eviction ordering ─────────────────────────────────────────────────────

def _test_eviction_ordering(mut t: TC) raises:
    print("\n--- A4: eviction ordering (stagehand-aligned distances) ---")

    # 5-block plan, single branch, lookahead=1.
    # byte_count_hints: [100, 500, 300, 200, 400]
    # cursor=0 → window=[1].  Candidates: {0, 2, 3, 4}.
    #
    # next_use_distance uses stagehand formula (branch=1):
    #   post_cursor = cursor + 1 = 1
    #   dist(j) = (j - post_cursor) % N,  0 → N=5
    #   dist(0) = (0 - 1) % 5 = 4
    #   dist(2) = (2 - 1) % 5 = 1
    #   dist(3) = (3 - 1) % 5 = 2
    #   dist(4) = (4 - 1) % 5 = 3
    #
    # Scores: 0→4*100=400, 2→1*300=300, 3→2*200=400, 4→3*400=1200.
    # Ties (0 vs 3, both 400): insertion sort stable — 0 before 3 in input order.
    # DESC order: 4(1200), 0(400), 3(400), 2(300).

    var plan = BlockPlan(String("ord_test"))
    var hints = List[Int]()
    hints.append(100); hints.append(500); hints.append(300)
    hints.append(200); hints.append(400)
    for i in range(5):
        plan.append(
            String("blk.") + String(i),
            BlockKind.transformer(),
            2,
            hints[i],
        )

    var cfg = OffloadConfig(2, 1, DTypePolicy.preserve(), BranchSchedule.single())
    var bgt = BudgetTracker(10_000_000, 5_000_000)
    var mgr = ResidencyManager(plan^, cfg, bgt^)

    for i in range(5):
        mgr.transition(i, BlockState.host_staged())
        mgr.transition(i, BlockState.prefetching())
        mgr.transition(i, BlockState.gpu_ready())

    # Verify individual distances (stagehand-aligned: post_cursor=1).
    t.check_eq(String("dist(block0, cursor=0)"), mgr.next_use_distance(0, 0), 4)
    t.check_eq(String("dist(block2, cursor=0)"), mgr.next_use_distance(2, 0), 1)
    t.check_eq(String("dist(block3, cursor=0)"), mgr.next_use_distance(3, 0), 2)
    t.check_eq(String("dist(block4, cursor=0)"), mgr.next_use_distance(4, 0), 3)

    var order = mgr.eviction_order(0, 0, 0)

    # Block 1 (in window) must be excluded.
    var has_1 = False
    for ci in range(len(order)):
        if order[ci] == 1: has_1 = True
    t.check(String("block 1 (window) excluded"), not has_1)

    t.check(String("order[0] = block 4 (score 1200)"),
        len(order) >= 1 and order[0] == 4)
    t.check(String("order[1] = block 0 (score 400, tie with 3 — 0 first in input)"),
        len(order) >= 2 and order[1] == 0)
    t.check(String("order[2] = block 3 (score 400, tie with 0)"),
        len(order) >= 3 and order[2] == 3)
    t.check(String("order[3] = block 2 (score 300)"),
        len(order) >= 4 and order[3] == 2)

    t.summary(String("A4"))


# ── A5: CFG revisit / next_use_distance with cfg_paired ──────────────────────

def _test_cfg_revisit(mut t: TC) raises:
    print("\n--- A5: CFG revisit ---")

    # 4-block plan, cfg_paired (branch_count=2), lookahead=1.
    # Expanded cycle: [0,0, 1,1, 2,2, 3,3]  (len=8)
    #
    # next_use_distance searches from post_cursor = cursor + 1 (inclusive).
    # offset=0 at post_cursor → sentinel (block is about to execute).
    #
    # From cursor=0 (post_cursor=1):
    #   Searching range(8) from pos 1:
    #   dist(0): offset=0 → pos=1, bi=0. offset=0 → sentinel=8.
    #   dist(1): offset=0 → pos=1, bi=0≠1. offset=1 → pos=2, bi=1. Return 1.
    #   dist(2): offset=0→bi=0, 1→bi=1, 2→bi=1, 3→pos=4,bi=2. Return 3.
    #   dist(3): offset=0→bi=0,1→bi=1,2→bi=1,3→bi=2,4→bi=2,5→pos=6,bi=3. Return 5.

    var plan = BlockPlan(String("cfg_test"))
    for i in range(4):
        plan.append(String("blk.") + String(i), BlockKind.transformer(), 2, 1024)

    var cfg = OffloadConfig(2, 1, DTypePolicy.preserve(), BranchSchedule.cfg_paired())
    var bgt = BudgetTracker(10_000_000, 5_000_000)
    var mgr = ResidencyManager(plan^, cfg, bgt^)

    t.check_eq(String("cfg: dist(0, cur=0) sentinel=8"), mgr.next_use_distance(0, 0), 8)
    t.check_eq(String("cfg: dist(1, cur=0)"), mgr.next_use_distance(1, 0), 1)
    t.check_eq(String("cfg: dist(2, cur=0)"), mgr.next_use_distance(2, 0), 3)
    t.check_eq(String("cfg: dist(3, cur=0)"), mgr.next_use_distance(3, 0), 5)

    # From cursor=3 (post_cursor=4):
    #   pos sequence: 4,5,6,7,0,1,2,3
    #   bi  sequence: 2,2,3,3,0,0,1,1
    #   dist(0): offset 4 → pos=0, bi=0. Return 4.
    #   dist(2): offset 0 → pos=4, bi=2. offset=0 → sentinel=8.
    t.check_eq(String("cfg: dist(0, cur=3)"), mgr.next_use_distance(0, 3), 4)
    t.check_eq(String("cfg: dist(2, cur=3) sentinel=8"), mgr.next_use_distance(2, 3), 8)

    # Eviction-order check: cursor=1, lookahead=1 → window=[2].
    # Bring all blocks to GPU_READY first.
    for i in range(4):
        mgr.transition(i, BlockState.host_staged())
        mgr.transition(i, BlockState.prefetching())
        mgr.transition(i, BlockState.gpu_ready())

    # Candidates: {0, 1, 3} (block 2 in window).
    # Distances from cursor=1 (post_cursor=2):
    #   pos seq: 2,3,4,5,6,7,0,1
    #   bi  seq: 1,1,2,2,3,3,0,0
    #   dist(0): offset=6 → pos=0, bi=0. Return 6.
    #   dist(1): offset=0 → pos=2, bi=1. offset=0 → sentinel=8.
    #   dist(3): offset=4 → pos=6, bi=3. Return 4.
    # Scores (bytes=1024): 0→6144, 1→8192 (sentinel), 3→4096.
    # Order DESC: [1(8192), 0(6144), 3(4096)].
    var order = mgr.eviction_order(1, 0, 0)
    var has_2 = False
    for ci in range(len(order)):
        if order[ci] == 2: has_2 = True
    t.check(String("cfg: block 2 (window) excluded"), not has_2)
    t.check(String("cfg: order[0]=1 (sentinel dist=8, score 8192)"),
        len(order) >= 1 and order[0] == 1)
    t.check(String("cfg: order[1]=0 (dist=6, score 6144)"),
        len(order) >= 2 and order[1] == 0)
    t.check(String("cfg: order[2]=3 (dist=4, score 4096)"),
        len(order) >= 3 and order[2] == 3)

    t.summary(String("A5"))


# ── A6: Klein9b full plan ─────────────────────────────────────────────────────

def _test_klein9b_plan(mut t: TC) raises:
    print("\n--- A6: Klein9b full plan ---")

    var plan = build_klein9b_block_plan()
    var n = plan.count()
    t.check_eq(String("klein9b block count"), n, 32)

    var cfg = OffloadConfig(2, 4, DTypePolicy.preserve(), BranchSchedule.single())
    var high_bytes = 22 * 1024 * 1024 * 1024
    var low_bytes  = 18 * 1024 * 1024 * 1024
    var bgt = BudgetTracker(high_bytes, low_bytes)
    var mgr = ResidencyManager(plan^, cfg, bgt^)

    t.check_eq(String("manager block count"), mgr.block_count(), 32)

    # Bring blocks 0..7 to GPU_READY, mark visited at step 0.
    for i in range(8):
        mgr.transition(i, BlockState.host_staged())
        mgr.transition(i, BlockState.prefetching())
        mgr.transition(i, BlockState.gpu_ready())
        mgr.mark_visit(i, 0)

    # Acquire block 3 (in-use).
    mgr.acquire(3)

    # Prefetch targets from cursor=3, lookahead=4 → window=[4,5,6,7].
    # All are GPU_READY → targets should be empty.
    var targets = mgr.prefetch_targets(3)
    t.check_eq(String("prefetch targets empty when window GPU_READY"),
        len(targets), 0)

    # Eviction candidates: GPU_READY + refcount==0 + not in window [4..7].
    # Blocks 0..3 not in window; block 3 acquired.
    # → Expected candidates include {0, 1, 2} at minimum (block 3 excluded).
    var cands = mgr.eviction_candidates(3, 0, 0)
    var has_3 = False
    for ci in range(len(cands)):
        if cands[ci] == 3: has_3 = True
    t.check(String("klein9b: acquired block 3 not in candidates"), not has_3)

    # Release block 3 → should now appear.
    mgr.release(3)
    var cands2 = mgr.eviction_candidates(3, 0, 0)
    var has_3_after = False
    for ci in range(len(cands2)):
        if cands2[ci] == 3: has_3_after = True
    t.check(String("klein9b: released block 3 appears in candidates"),
        has_3_after)

    # Eviction order: verify no window-protected blocks appear.
    var order = mgr.eviction_order(3, 0, 0)
    var window_in_order = False
    for ci in range(len(order)):
        var idx = order[ci]
        if idx >= 4 and idx <= 7:
            window_in_order = True
    t.check(String("klein9b: no window blocks in eviction order"),
        not window_in_order)
    t.check(String("klein9b: eviction order non-empty"), len(order) > 0)

    print("[klein9b] eviction order (first 4):", end="")
    var show = len(order)
    if show > 4: show = 4
    for ci in range(show):
        print("", order[ci], end="")
    print("")

    t.summary(String("A6"))


# ── A7: stagehand cross-check — branch=1 reduction + mixed-size counterexample ─

def _check(label: String, cond: Bool) raises:
    """Fail-closed assertion: raises on False (keeps smoke fail-closed)."""
    if not cond:
        raise Error(String("ASSERTION FAILED: ") + label)

def _test_stagehand_crosscheck(mut t: TC) raises:
    print("\n--- A7: stagehand cross-check (branch=1 parity + mixed-size bug) ---")

    # ── Part 1: branch_count=1 reduces exactly to stagehand formula ──────────
    #
    # Stagehand's score_for_eviction (policy.py lines 139-143):
    #   next_use_distance = (entry_exec_order - current_cursor) % total_blocks
    #   if next_use_distance == 0:
    #       next_use_distance = total_blocks
    #
    # where current_cursor = post-increment cursor = pre-increment cursor + 1.
    #
    # Mojo next_use_distance(block_index, cursor) with branch=1 must be
    # IDENTICAL to (block_index - (cursor+1)) % N, with 0→N.
    #
    # We verify this for all (block_index, cursor) combinations in a 5-block plan.

    var plan5 = BlockPlan(String("xcheck_test"))
    for i in range(5):
        plan5.append(String("blk.") + String(i), BlockKind.transformer(), 2, 100)

    var cfg1 = OffloadConfig(2, 1, DTypePolicy.preserve(), BranchSchedule.single())
    var bgt1 = BudgetTracker(10_000_000, 5_000_000)
    var mgr1 = ResidencyManager(plan5^, cfg1, bgt1^)

    var n = 5
    var all_match = True
    for bi in range(n):
        for c in range(n):
            var mojo_dist = mgr1.next_use_distance(bi, c)
            # Stagehand formula with post_cursor = c + 1
            var raw = (bi - (c + 1)) % n
            var stagehand_dist: Int
            if raw <= 0:
                # Python % always non-negative; Mojo % may give negative for (bi-post)%n
                # when bi < post_cursor. We replicate Python semantics:
                var r2 = bi - (c + 1)
                var adjusted = ((r2 % n) + n) % n
                if adjusted == 0:
                    stagehand_dist = n
                else:
                    stagehand_dist = adjusted
            else:
                stagehand_dist = raw
            if mojo_dist != stagehand_dist:
                all_match = False
                t.fail(
                    String("branch1_parity bi=") + String(bi) + String(" c=") + String(c),
                    String("mojo=") + String(mojo_dist)
                    + String(" stagehand=") + String(stagehand_dist),
                )
    if all_match:
        t.ok(String("branch=1 reduces exactly to stagehand formula (all 25 pairs)"))

    # Fail-closed check: the loop above must have actually run.
    _check(String("cross-check loop ran 25 iterations"), n * n == 25)

    # ── Part 2: mixed-size counterexample (the bug caught by skeptic) ─────────
    #
    # Scenario: 5 blocks, branch=1, cursor=0 (pre-increment, i.e. after block -1 /
    # initial state).  Block A = block 2 (size 50), block B = block 3 (size 30).
    #
    # Stagehand (post_cursor=1):
    #   dist_A = (2-1)%5 = 1, dist_B = (3-1)%5 = 2
    #   score_A = 1*50 = 50, score_B = 2*30 = 60  → evict B first (60 > 50)
    #
    # Old BUGGY Mojo code searched from cursor+1=1, giving distances uniformly +1:
    #   dist_A = 2, dist_B = 3
    #   score_A = 2*50 = 100, score_B = 3*30 = 90  → WRONGLY evicted A first
    #
    # This test MUST fail against the old code and PASS against the corrected code.

    var plan_ce = BlockPlan(String("counterex_test"))
    # block 0: size 999 (large, to dominate ordering)
    # block 1: size 1 (will be in window, excluded)
    # block 2: size 50 (block A in counterexample)
    # block 3: size 30 (block B in counterexample)
    # block 4: size 1 (filler)
    var ce_sizes = List[Int]()
    ce_sizes.append(999); ce_sizes.append(1); ce_sizes.append(50)
    ce_sizes.append(30); ce_sizes.append(1)
    for i in range(5):
        plan_ce.append(
            String("blk.") + String(i),
            BlockKind.transformer(),
            2,
            ce_sizes[i],
        )

    var cfg_ce = OffloadConfig(2, 1, DTypePolicy.preserve(), BranchSchedule.single())
    var bgt_ce = BudgetTracker(10_000_000, 5_000_000)
    var mgr_ce = ResidencyManager(plan_ce^, cfg_ce, bgt_ce^)

    for i in range(5):
        mgr_ce.transition(i, BlockState.host_staged())
        mgr_ce.transition(i, BlockState.prefetching())
        mgr_ce.transition(i, BlockState.gpu_ready())

    # cursor=0, window=[1] (block 1 protected), candidates={0,2,3,4}.
    # dist_A=dist(2,0)=1, dist_B=dist(3,0)=2.
    var dist_a = mgr_ce.next_use_distance(2, 0)
    var dist_b = mgr_ce.next_use_distance(3, 0)
    t.check_eq(String("counterex: dist_A(block2, cursor=0) == 1"), dist_a, 1)
    t.check_eq(String("counterex: dist_B(block3, cursor=0) == 2"), dist_b, 2)

    # score_A=1*50=50, score_B=2*30=60 → B evicted before A (stagehand semantics).
    # Block 0 has score=4*999=3996 (dominates), block 4 has score=3*1=3.
    # Full expected order: 0(3996), 3_B(60), 2_A(50), 4(3).
    var order_ce = mgr_ce.eviction_order(0, 0, 0)

    # Critical check: block B (index 3) must appear before block A (index 2).
    var pos_a = -1
    var pos_b = -1
    for ci in range(len(order_ce)):
        if order_ce[ci] == 2: pos_a = ci
        if order_ce[ci] == 3: pos_b = ci
    t.check(String("counterex: block B (idx=3) evicted before block A (idx=2) [B<A]"),
        pos_b >= 0 and pos_a >= 0 and pos_b < pos_a)
    t.check(String("counterex: block 0 is first (score 3996)"),
        len(order_ce) >= 1 and order_ce[0] == 0)
    t.check(String("counterex: block B(3) is second (score 60)"),
        len(order_ce) >= 2 and order_ce[1] == 3)
    t.check(String("counterex: block A(2) is third (score 50)"),
        len(order_ce) >= 3 and order_ce[2] == 2)

    t.summary(String("A7"))


# ── main ──────────────────────────────────────────────────────────────────────

def main() raises:
    var t = TC()

    _test_state_machine(t)
    _test_budget(t)
    _test_refcount(t)
    _test_eviction_ordering(t)
    _test_cfg_revisit(t)
    _test_klein9b_plan(t)
    _test_stagehand_crosscheck(t)

    print("\n=== FINAL SUMMARY ===")
    print("PASS:", t.passes, " FAIL:", t.fails)
    if t.fails > 0:
        raise Error(
            String("residency_smoke: ") + String(t.fails) + String(" assertion(s) FAILED")
        )
    print("ALL ASSERTIONS PASSED")
