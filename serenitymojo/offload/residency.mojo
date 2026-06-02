# residency.mojo — Phase 2: block residency / budget / scheduling logic.
#
# Faithful port of stagehand's:
#   residency.py  → BlockState enum + ResidencyEntry + state-machine transitions
#   budget.py     → BudgetTracker (watermark accounting; NO torch.cuda calls —
#                   we track bytes ourselves, caller drives the numbers)
#   scheduler.py  → eviction_candidates / eviction_order / prefetch_targets /
#                   next_use_distance (lines ~868-897 and ~1142-1196)
#
# INFERENCE SIMPLIFICATION (intentional and documented):
#   Stagehand's _evict_block() has a save_back flag: when True it issues a D2H
#   copy before freeing the device slot (needed during training if the optimizer
#   has written updated weights back to GPU).  In inference mode save_back is
#   always False — frozen file-backed weights are not modified on GPU, so
#   eviction just releases the device slot with no write-back.  This file
#   implements ONLY the inference path.  The D2H write-back path is out of scope.
#
# Dict value types MUST be Copyable+ImplicitlyDestructible in Mojo 26.3.
# ResidencyEntry is a plain value struct (all fields are primitives or String),
# so it satisfies this constraint directly.
#
# Sorting: Mojo List has no sort() in 26.3; we use a simple insertion-sort on
# the small (<200-block) candidate lists that arise in practice.

from serenitymojo.offload.plan import BlockPlan, OffloadConfig, BranchSchedule


# ── BlockState enum ───────────────────────────────────────────────────────────
# Mirror of residency.py's BlockState.
# Integer tags match the declaration order; names used for diagnostics only.
#
#   UNLOADED      (0) → HOST_STAGED
#   HOST_STAGED   (1) → PREFETCHING
#   PREFETCHING   (2) → GPU_READY
#   GPU_READY     (3) → EVICTING | GPU_FREEING
#   EVICTING      (4) → HOST_STAGED | UNLOADED
#   GPU_FREEING   (5) → UNLOADED

@fieldwise_init
struct BlockState(Copyable, Movable, ImplicitlyCopyable, Equatable):
    var tag: Int

    @staticmethod
    def unloaded() -> BlockState:
        return BlockState(0)

    @staticmethod
    def host_staged() -> BlockState:
        return BlockState(1)

    @staticmethod
    def prefetching() -> BlockState:
        return BlockState(2)

    @staticmethod
    def gpu_ready() -> BlockState:
        return BlockState(3)

    @staticmethod
    def evicting() -> BlockState:
        return BlockState(4)

    @staticmethod
    def gpu_freeing() -> BlockState:
        return BlockState(5)

    def name(self) -> String:
        if self.tag == 0: return "UNLOADED"
        if self.tag == 1: return "HOST_STAGED"
        if self.tag == 2: return "PREFETCHING"
        if self.tag == 3: return "GPU_READY"
        if self.tag == 4: return "EVICTING"
        if self.tag == 5: return "GPU_FREEING"
        return "UNKNOWN"

    def is_gpu_resident(self) -> Bool:
        # GPU_READY or PREFETCHING = device slot in use.
        return self.tag == 2 or self.tag == 3


# ── legal transition table ────────────────────────────────────────────────────
# Encoded as a flat lookup: _ALLOWED[from.tag] is a List of allowed to.tags.
# Built once as a comptime constant via a factory function so no heap alloc.

def _transition_allowed(from_tag: Int, to_tag: Int) -> Bool:
    """Return True iff (from_tag → to_tag) is a legal state transition."""
    # UNLOADED → HOST_STAGED
    if from_tag == 0: return to_tag == 1
    # HOST_STAGED → PREFETCHING
    if from_tag == 1: return to_tag == 2
    # PREFETCHING → GPU_READY
    if from_tag == 2: return to_tag == 3
    # GPU_READY → EVICTING | GPU_FREEING
    if from_tag == 3: return to_tag == 4 or to_tag == 5
    # EVICTING → HOST_STAGED | UNLOADED
    if from_tag == 4: return to_tag == 1 or to_tag == 0
    # GPU_FREEING → UNLOADED
    if from_tag == 5: return to_tag == 0
    return False


# ── ResidencyEntry ────────────────────────────────────────────────────────────
# Per-block mutable metadata.  Plain value struct; all fields are primitives.
# Mirrors stagehand's ResidencyEntry dataclass (minus torch tensors — we don't
# hold device/host tensors here; the TurboBlockLoader owns those).

struct ResidencyEntry(Copyable, Movable, ImplicitlyCopyable):
    var state: BlockState
    var refcount: Int
    var last_used_step: Int   # step when block was last consumed by compute
    var byte_count: Int       # byte_count_hint from BlockRecord (0 = unknown)
    var slot_index: Int       # physical slot index or -1 if not resident

    def __init__(out self):
        self.state = BlockState.unloaded()
        self.refcount = 0
        self.last_used_step = -1
        self.byte_count = 0
        self.slot_index = -1

    def __init__(out self, byte_count: Int):
        self.state = BlockState.unloaded()
        self.refcount = 0
        self.last_used_step = -1
        self.byte_count = byte_count
        self.slot_index = -1

    def acquire(mut self):
        """Increment refcount: block is in active use by compute."""
        self.refcount += 1

    def release(mut self) raises:
        """Decrement refcount. Raises if already 0 (mirrors stagehand ValueError)."""
        if self.refcount <= 0:
            raise Error("ResidencyEntry.release: refcount is already 0")
        self.refcount -= 1

    def can_evict(self) -> Bool:
        """True iff GPU_READY with refcount == 0 (stagehand ResidencyMap.can_evict)."""
        return self.state == BlockState.gpu_ready() and self.refcount == 0


# ── BudgetTracker ─────────────────────────────────────────────────────────────
# Pure byte-counting watermark tracker — no torch.cuda calls.
# Caller is responsible for providing actual bytes-in-use.
#
# Mirrors budget.py BudgetManager, replacing torch.cuda.memory_allocated() with
# a tracked counter (self._current_bytes) that the caller updates via add/remove.
# The predicates (can_prefetch, should_evict) have identical semantics.

struct BudgetTracker(Movable):
    var _high_bytes: Int    # trigger eviction above this
    var _low_bytes: Int     # stop evicting once below this
    var _current_bytes: Int # bytes currently tracked as device-resident

    def __init__(out self, high_bytes: Int, low_bytes: Int) raises:
        if high_bytes <= low_bytes:
            raise Error(
                String("BudgetTracker: high_bytes (")
                + String(high_bytes)
                + String(") must be greater than low_bytes (")
                + String(low_bytes)
                + String(")")
            )
        self._high_bytes = high_bytes
        self._low_bytes = low_bytes
        self._current_bytes = 0

    def high_bytes(self) -> Int:
        return self._high_bytes

    def low_bytes(self) -> Int:
        return self._low_bytes

    def current_bytes(self) -> Int:
        return self._current_bytes

    def headroom_bytes(self) -> Int:
        return self._high_bytes - self._current_bytes

    def above_high_watermark(self) -> Bool:
        return self._current_bytes >= self._high_bytes

    def below_low_watermark(self) -> Bool:
        return self._current_bytes < self._low_bytes

    def can_prefetch(self) -> Bool:
        """True if usage is below the high watermark (mirrors budget.py)."""
        return not self.above_high_watermark()

    def should_evict(self) -> Bool:
        """True if usage exceeds the high watermark (mirrors budget.py)."""
        return self.above_high_watermark()

    def add(mut self, bytes: Int):
        """Record that `bytes` additional bytes are now device-resident."""
        self._current_bytes += bytes

    def remove(mut self, bytes: Int):
        """Record that `bytes` bytes have been freed from the device."""
        self._current_bytes -= bytes
        if self._current_bytes < 0:
            self._current_bytes = 0


# ── Scored eviction candidate (internal pair) ─────────────────────────────────
# A plain Copyable struct so we can store it in a List.

struct _ScoredCandidate(Copyable, Movable, ImplicitlyCopyable):
    var score: Int   # next_use_distance * byte_count (larger → evict first)
    var index: Int   # block index

    def __init__(out self, score: Int, index: Int):
        self.score = score
        self.index = index


# ── ResidencyManager ──────────────────────────────────────────────────────────
# Owns one ResidencyEntry per block in the plan, a BudgetTracker, and the config.
#
# Implements:
#   - State-machine enforcement via transition()
#   - Refcount discipline: acquire / release / mark_visit
#   - Eviction candidacy: eviction_candidates() / eviction_order()
#   - Prefetch targeting: prefetch_targets(cursor)
#   - Distance arithmetic: next_use_distance(i, cursor)
#
# Dict[Int, ResidencyEntry] maps block_index → entry.
# Int keys, ResidencyEntry values — both satisfy Copyable+ImplicitlyDestructible.

struct ResidencyManager(Movable):
    var _entries: Dict[Int, ResidencyEntry]
    var _budget: BudgetTracker
    var _plan: BlockPlan
    var _config: OffloadConfig
    var _block_count: Int

    def __init__(
        out self,
        var plan: BlockPlan,
        config: OffloadConfig,
        var budget: BudgetTracker,
    ):
        self._plan = plan^
        self._config = config
        self._budget = budget^
        self._block_count = self._plan.count()
        self._entries = Dict[Int, ResidencyEntry]()
        for i in range(self._block_count):
            var bc = self._plan.records[i].byte_count_hint
            self._entries[i] = ResidencyEntry(bc)

    # ── block count ───────────────────────────────────────────────────

    def block_count(self) -> Int:
        return self._block_count

    # ── state machine ─────────────────────────────────────────────────

    def get_state(self, block_index: Int) raises -> BlockState:
        return self._entries[block_index].state

    def transition(mut self, block_index: Int, new_state: BlockState) raises:
        """Advance block_index to new_state.  Raises on illegal transition.

        Also keeps BudgetTracker in sync:
          * GPU_READY arrival  → add byte_count
          * GPU_READY departure → remove byte_count

        BUDGET ACCOUNTING TIMING NOTE (P3 — known limitation):
          Bytes are added to the budget at PREFETCHING→GPU_READY, not at
          HOST_STAGED→PREFETCHING.  This is correct for the current sequential
          two-slot loader where only one H2D transfer is in flight at a time:
          the block is not yet consuming GPU memory during PREFETCHING.
          LIMITATION: if the loader is ever extended to issue multiple
          concurrent H2D transfers (multiple blocks simultaneously in
          PREFETCHING state), the bytes in flight would be invisible to
          can_prefetch(), potentially causing budget overruns.  A production
          concurrent loader should call _budget.add() at HOST_STAGED→PREFETCHING
          (and _budget.remove() on PREFETCHING cancellation) to account for
          in-flight allocations.
        """
        var entry = self._entries[block_index]
        var from_tag = entry.state.tag
        var to_tag = new_state.tag
        if not _transition_allowed(from_tag, to_tag):
            raise Error(
                String("ResidencyManager.transition: illegal ")
                + entry.state.name()
                + String(" -> ")
                + new_state.name()
                + String(" for block ")
                + String(block_index)
            )
        # Budget accounting: track when a block enters or leaves GPU_READY.
        # GPU_READY arrival: PREFETCHING(2) → GPU_READY(3)
        if from_tag == 2 and to_tag == 3:
            self._budget.add(entry.byte_count)
        # GPU_READY departure: GPU_READY(3) → EVICTING(4) or GPU_FREEING(5)
        if from_tag == 3 and (to_tag == 4 or to_tag == 5):
            self._budget.remove(entry.byte_count)
        entry.state = new_state
        self._entries[block_index] = entry

    # ── refcount & scheduling metadata ───────────────────────────────

    def acquire(mut self, block_index: Int) raises:
        """Increment refcount for block_index (block is consumed by compute)."""
        var entry = self._entries[block_index]
        entry.acquire()
        self._entries[block_index] = entry

    def release(mut self, block_index: Int) raises:
        """Decrement refcount for block_index."""
        var entry = self._entries[block_index]
        entry.release()
        self._entries[block_index] = entry

    def mark_visit(mut self, block_index: Int, step: Int) raises:
        """Record that block_index was consumed at `step`."""
        var entry = self._entries[block_index]
        entry.last_used_step = step
        self._entries[block_index] = entry

    # ── budget pass-through ───────────────────────────────────────────

    def can_prefetch(self) -> Bool:
        return self._budget.can_prefetch()

    def should_evict(self) -> Bool:
        return self._budget.should_evict()

    def budget_current_bytes(self) -> Int:
        return self._budget.current_bytes()

    # ── next-use distance ─────────────────────────────────────────────

    def next_use_distance(self, block_index: Int, cursor: Int) -> Int:
        """Steps until block_index is next visited.

        CURSOR CONVENTION (pre-increment, consistent with eviction_candidates):
          `cursor` is the PRE-INCREMENT value — the last completed position
          index in the (branch-expanded) visit sequence.  This matches how
          `eviction_candidates` uses cursor to define the prefetch window
          [cursor+1 .. cursor+lookahead].

          Callers should pass cursor = (last completed block index) for
          branch_count=1, or (last completed expanded-list position) for
          branch_count=2.

        STAGEHAND ALIGNMENT (branch_count=1):
          With branch_count=1 this function returns distances IDENTICAL to
          stagehand's `policy.score_for_eviction` formula:
            (entry_exec_order - current_cursor) % N,  with 0 → N
          where stagehand's `current_cursor` is the POST-INCREMENT cursor
          (= pre-increment cursor + 1).

          PROOF: For branch=1, cycle=N. We search from post_cursor = cursor+1
          forward.  The first match for block j is at expanded-offset:
            effective_offset = (j - post_cursor) % N  (= (j - cursor - 1) % N)
          returning that value, with 0 mapped to N.
          This equals stagehand's `(exec_order - post_cursor) % N` with 0→N
          because branch=1 → expanded-offset == block-offset for a single
          occurrence per cycle.  QED.

        BRANCH-EXPANDED EXTENSION (deliberate, not a bug — P2):
          serenitymojo supports CFG-paired diffusion where each block is
          visited branch_count (=2) times per diffusion step (conditional
          + unconditional passes).  Stagehand's `score_for_eviction` has no
          branch_count — its cursor is a plain block index 0..N-1.
          We extend by treating the visit sequence as the expanded list
          [0,0,1,1,...,N-1,N-1] (each block repeated branch_count times),
          so cursor is a position 0..N*branch_count-1 in that list.

          For branch_count=1 the expanded list is [0,1,...,N-1] and the
          formula reduces identically to stagehand (see PROOF above).

          For branch_count=2, distances are in branch-visit units.  Each
          block occupies 2 consecutive slots; the iterative search finds the
          nearest occurrence.  The sentinel (0 → cycle) applies when the
          nearest occurrence is at post_cursor itself, meaning the block is
          about to execute at the very next expanded position — analogous to
          stagehand setting distance=0 → N for the next-to-execute block.

        ALGORITHM:
          post_cursor = cursor + 1   (post-increment, stagehand convention)
          For offset = 0 .. cycle-1:
            pos = (post_cursor + offset) % cycle
            bi  = pos // branch
            if bi == block_index:
              return cycle if offset == 0 else offset
          return cycle  (sentinel: not found = full cycle away)
        """
        var n = self._block_count
        if n == 0:
            return 0
        var branch = self._config.branch_schedule.branch_count()
        # Total positions per cycle = n * branch_count.
        var cycle = n * branch
        # Post-increment cursor: the next position to be executed.
        var post_cursor = cursor + 1
        # Search forward from post_cursor for the nearest occurrence of block_index.
        # offset=0 → block is at post_cursor (about to execute) → sentinel.
        # offset>0 → block is that many steps ahead.
        for offset in range(cycle):
            var pos = (post_cursor + offset) % cycle
            var bi = pos // branch  # block index for this expanded position
            if bi == block_index:
                if offset == 0:
                    # Block is at post_cursor — it is the very next to execute.
                    # Sentinel: treat as cycle (farthest) so it is evicted last,
                    # mirroring stagehand's 0 → total_blocks fixup.
                    return cycle
                return offset
        # Should not be reached for valid block_index; return cycle as sentinel.
        return cycle

    # ── eviction candidates (raw, unscored) ───────────────────────────

    def eviction_candidates(
        self,
        cursor: Int,
        cooldown_steps: Int = 0,
        current_step: Int = 0,
    ) raises -> List[Int]:
        """Block indices eligible for eviction.

        A block is a candidate iff ALL of:
          1. State == GPU_READY
          2. refcount == 0
          3. last_used_step <= current_step - cooldown_steps
          4. NOT within the prefetch window [cursor+1 .. cursor+lookahead]

        Mirrors stagehand ResidencyMap.eviction_candidates() + the
        prefetch-window guard in scheduler._run_eviction().

        Returns indices in arbitrary order; call eviction_order() for scoring.
        """
        var lookahead = self._config.lookahead
        # Build prefetch-window set (block indices protected from eviction).
        var in_window = Dict[Int, Bool]()
        for offset in range(1, lookahead + 1):
            var idx = cursor + offset
            if idx >= 0 and idx < self._block_count:
                in_window[idx] = True

        var threshold = current_step - cooldown_steps
        var result = List[Int]()
        for i in range(self._block_count):
            var entry = self._entries[i]
            if entry.state != BlockState.gpu_ready():
                continue
            if entry.refcount != 0:
                continue
            if entry.last_used_step > threshold:
                continue
            if i in in_window:
                continue
            result.append(i)
        return result^

    # ── scored eviction order ─────────────────────────────────────────

    def eviction_order(
        self,
        cursor: Int,
        cooldown_steps: Int = 0,
        current_step: Int = 0,
    ) raises -> List[Int]:
        """Eviction candidates sorted by score DESC (highest score evicted first).

        Score = next_use_distance(i, cursor) * byte_count
        Highest score = furthest-next-needed AND/OR largest — evict that first
        (mirrors stagehand scheduler._run_eviction scoring).

        Uses insertion sort (sufficient for <200-element candidate lists).
        """
        var candidates = self.eviction_candidates(cursor, cooldown_steps, current_step)
        var scored = List[_ScoredCandidate]()
        for ci in range(len(candidates)):
            var i = candidates[ci]
            var dist = self.next_use_distance(i, cursor)
            var entry = self._entries[i]
            var score = dist * entry.byte_count
            scored.append(_ScoredCandidate(score, i))

        # Insertion sort DESC by score.
        for k in range(1, len(scored)):
            var key = scored[k]
            var j = k - 1
            while j >= 0 and scored[j].score < key.score:
                scored[j + 1] = scored[j]
                j -= 1
            scored[j + 1] = key

        var result = List[Int]()
        for ci in range(len(scored)):
            result.append(scored[ci].index)
        return result^

    # ── prefetch targets ──────────────────────────────────────────────

    def prefetch_targets(self, cursor: Int) raises -> List[Int]:
        """Block indices to prefetch: (cursor, cursor+lookahead], not yet resident.

        Mirrors stagehand scheduler._prefetch_ahead():
          For each index in cursor+1 .. cursor+lookahead (inclusive):
            - Skip if already GPU_READY or PREFETCHING.
            - Stop (don't add remaining) if budget.can_prefetch() is False.

        Returns the list in prefetch order (closest first).
        """
        var lookahead = self._config.lookahead
        var result = List[Int]()
        for offset in range(1, lookahead + 1):
            var idx = cursor + offset
            if idx < 0 or idx >= self._block_count:
                continue
            if not self._budget.can_prefetch():
                break
            var entry = self._entries[idx]
            var s = entry.state
            if s == BlockState.gpu_ready() or s == BlockState.prefetching():
                continue
            result.append(idx)
        return result^
