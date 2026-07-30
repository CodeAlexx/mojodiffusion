# training/krea2_omini_layout.mojo — OminiControl EDIT row layout for krea2 (C-scaffold).
#
# The OminiControl port (intake: ominicontrol_krea2_intake.md §3) extends krea2's
# length-bucket sequence [TXT_real(lt) | IMG | TXT_pad] with a CONDITION token
# segment fed through the same `first` projection, giving the joint-attention row
# layout this module owns:
#
#     [ TXT_real(0:lt) | IMG(lt : lt+S_IMG) | COND(.. +S_COND) | TXT_pad(tail) ]
#     LFULL_EDIT = LTMAX + S_IMG + S_COND        real_len = lt + S_IMG + S_COND
#
# WHY this order: the cuDNN flash-padmask masks ONLY a tail — the valid tokens
# must be a CONTIGUOUS PREFIX (train_krea2.mojo:795-798 and the reorder comment
# at train_krea2.mojo:840-844). COND rows are valid (they attend + are attended,
# OminiControl v1 mode is fully bidirectional, intake §1.4), so they must sit
# BEFORE the text pad. Everything here is pure layout/selector math (host Int/F32
# arithmetic) — no tensors, no GPU; the block/trainer code consumes these offsets.
#
# Per-segment modulation (the one structural change, intake §3.3): OminiControl
# gives condition tokens temb(t=0) while text+image ride temb(t)
# (trainer.py:232 `timesteps=[t, t] + [zeros]*len(conditions)`). The selector
# functions below express that row policy; pad rows are DON'T-CARE (their
# attention output is masked garbage the pad d_out zeroes downstream).
#
# Run the layout checks (CPU-only structure math, no parity claim):
#   pixi run mojo run -I . serenitymojo/training/krea2_omini_layout.mojo
#
# ══════════════════════════════════════════════════════════════════════════════
# SEAM PLAN — where the block/trainer code will consume this module (next chunks;
# NOTHING in krea2_block.mojo / train_krea2.mojo is touched by this chunk).
# All line refs verified against the in-tree sources 2026-07-30.
#
# A) TRAINER SEQUENCE ASSEMBLY — serenitymojo/models/krea2/train_krea2.mojo
#    1. `_build_conditioning` combined concat (train_krea2.mojo:845-853): today
#       `concat(real_text, img_e, pad_text)`. Edit build inserts the projected
#       cond tokens: `concat(real_text, img_e, cond_e, pad_text)` — offsets are
#       exactly Krea2OminiLayout{txt_real,img,cond,pad}_off below. cond_e =
#       krea2_first(cond_img_tokens) + zero-init img_in_ref delta
#       (krea2_img_in_ref_param.mojo, intake §3.4 x_embedder row).
#    2. blk_vec chain (train_krea2.mojo:808-822): run the SAME
#       krea2_temb→krea2_tmlp→krea2_tproj chain a second time on t=0 to get
#       `blk_vec2_cond [1,6F]` for the cond segment (per-segment modulation).
#    3. pos reorder `_reorder_pos_for_combined` (train_krea2.mojo:880-894, and
#       the inline copy at 857-869): generalizes to `combined_src_row()` below —
#       source pos layout [TXT_zeros(LTMAX) | IMG grid | COND grid(+delta,scale)]
#       reordered to the combined row order. COND grid positions are built with
#       `krea2_omini_cond_pos()` (EDIT: delta [0,0], scale 1.0 → cond pos ==
#       img pos, spatially overlapping — intake §1.2/§3.2).
#    4. text grad gather `_combined_text_grad` (train_krea2.mojo:969-982): the
#       pad slice start moves from `real_text_len + IMGLEN` to
#       `real_text_len + IMGLEN + condlen` == pad_off(); `text_grad_src_row()`
#       below is that mapping. Loss keeps reading ONLY the IMG rows
#       (img_off..img_off+s_img), matching Omini discarding cond outputs
#       (intake §1.6).
#    5. comptime shape: LFULL_EDIT = LTMAX + IMGLEN + CONDLEN needs a new
#       `-D KREA2_CONDLEN` beside KREA2_LTMAX (train_krea2.mojo:407-408);
#       CONDLEN=0 must degenerate bit-equal to today (checked in main()).
#
# B) BLOCK FORWARD — serenitymojo/models/krea2/krea2_block.mojo
#    `krea2_single_stream_block_lora` (krea2_block.mojo:547):
#    1. Per-segment modulation: `_mod6` (def krea2_block.mojo:519, call site
#       :573) currently yields ONE 6-chunk set applied to ALL rows via
#       `modulate` (:584, :695) and `residual_gate` (:691, :728). Edit path
#       takes BOTH vecs (blk_vec2, blk_vec2_cond), computes two `_mod6` sets,
#       and applies them row-segmented per `mod_selector()`: rows
#       [0, cond_off) → mods(t); rows [cond_off, pad_off) → mods(0); pad rows
#       don't matter (keep mods(t) — cheapest, value never read). Concretely:
#       slice x at cond_off/pad_off, modulate each slice, concat — or a
#       segmented modulate/residual_gate op taking (row_off,row_len) pairs.
#    2. Cond-row LoRA routing — LANDED IN C3. krea2_block gained a third
#       optional param `cond_len`; when it is supplied alongside a valid
#       cond_off the block computes each LoRA delta on the cond row slice and
#       scatter-adds it back (helpers `_lora_delta_rows` / `_add_delta_rows`,
#       both with a c_off < 0 sentinel that reduces to the pre-C3 kernels).
#       Gated by parity/krea2_omini_c3_lora_gate.mojo. Original plan text kept
#       below for the line references; the M note is now realised as `c_len`.
#       (intake §1.3/§3.4 — adapter acts ONLY on cond
#       rows; text+img rows run frozen base): base matmuls stay full-sequence
#       (`_base_fwd` def :187; sites :587-590 q/k/v/gate and :697-698
#       mlp_gate/mlp_up; `_linear_lora` sites :688 wo idx4 and :726 mlp_down
#       idx7). The LoRA delta (`_lora_fwd` ungrouped sites :623-635 and
#       :718-724; grouped :592-622/:700-717; and inside `_linear_lora` :155)
#       changes input from full `xm` to `slice(xm, 1, cond_off, s_cond)` and
#       the add becomes a scatter-add into rows [cond_off, cond_off+s_cond) of
#       the base output. M (LoRA row count, :569) becomes s_cond, not L.
#
# C) BLOCK BACKWARD — krea2_block.mojo — LANDED IN C4.
#    `krea2_single_stream_block_lora_backward` and the device path
#    `krea2_single_stream_block_lora_backward_dev` both took the same three
#    optional switches the C2/C3 forward has (vec_cond / cond_off / cond_len),
#    with all three absent leaving the pre-C4 kernels and arguments untouched.
#    Gated by parity/krea2_omini_c4_bwd_gate.mojo (PASS on block00 + block27).
#    1. dX through the frozen base stays FULL-SEQUENCE via `_base_dx` inside
#       `_linear_bwd_dx` / `_linear_bwd_dx_dev` — as planned, unchanged.
#    2. dA/dB come from the COND-ROW SLICES of (d_y, x) only: those two helpers
#       gained c_off/c_len and slice both inputs before the LoRA backward, then
#       scatter-add the LoRA d_x contribution back with `_add_delta_rows` (now
#       rank-aware, since the backward's linear_backward_dx returns [L, C]).
#       The routed path calls them ungrouped (4x / 2x) instead of the batched
#       `_linear_bwd_dx_group4/2` — grouping is a launch-count optimization
#       whose shared down-projection GEMM would have to be re-derived for the
#       c_len-row window; that is left for a speed chunk.
#    3. d_mod accumulates per segment: `_modulate_backward_seg2` /
#       `_gate_residual_backward_seg2` run the SAME kernels once per span and
#       return the chunk grads separately; `_pack_d_vec` packs each set into
#       Krea2BlockGrads.d_vec_t (rows [0, cond_off)) and .d_vec_cond (rows
#       [cond_off, REAL_LEN)) — the two temb chains.
#    4. ⚠ THE PAD TAIL IS EXCLUDED FROM (3), and that was decided WITH EVIDENCE,
#       not by the "don't-care" note at the top of this file. Under per-segment
#       modulation the pad tail rides the t=0 span, so the naive [cond_off, L)
#       reduction would fold pad-row gradient into the CONDITION temb chain.
#       Measured on the CUDA fixture: pad-row gradient leaves d_x[0:real_len],
#       d_vec_t and all 16 dA/dB BIT-EQUAL (pad KEY columns get exp(-inf)=0
#       exactly, so their dK/dV are exactly zero) while moving d_vec_cond to
#       cos = 0.679 at block 0 / 0.994 at block 27. The reduction therefore
#       stops at real_len; pad rows still get their d_x (same chunks, param
#       grads off), so the returned d_x is complete.
#    5. NOT COVERED: the BATCH-2 backwards. The b2 FORWARD has no EDIT switches
#       at all and its [2, features] per-sample modulation would need a 4-way
#       (sample x segment) split that ops/elementwise `modulate` does not
#       express, so b2 + EDIT is not a reachable configuration. Building it
#       starts with the b2 forward.
#
# D) CACHE READER — serenitymojo/models/krea2/krea2_cache_reader.mojo
#    `krea2_build_pos` (:84-100) grows a cond grid section built from
#    `krea2_omini_cond_pos()`; `krea2_build_pad_mask` / `_krea2_pad_mask_kernel`
#    (:117-149) is already parameterized on (lt, ltmax, imglen) — the edit mask
#    passes imglen+condlen so the masked key columns stay [lt, ltmax) relative
#    to a [TXT|IMG|COND] source order; the flash path only needs real_len().
# ══════════════════════════════════════════════════════════════════════════════

# Row-modulation selector values (per-segment AdaLN policy, intake §1.5/§3.3).
comptime KREA2_OMINI_MOD_T = 0     # rows modulated with mods(t):   TXT_real + IMG
comptime KREA2_OMINI_MOD_COND = 1  # rows modulated with mods(t=0): COND
comptime KREA2_OMINI_MOD_PAD = 2   # TXT_pad tail — value never read downstream


@fieldwise_init
struct Krea2OminiLayout(Copyable, Movable):
    """Row layout [TXT_real(lt) | IMG(s_img) | COND(s_cond) | TXT_pad(ltmax-lt)]
    for the krea2 OminiControl EDIT sequence. Pure offset/length math; the
    comptime shape (LTMAX/IMGLEN/CONDLEN -D flags) instantiates one of these per
    sample with the sample's runtime lt."""

    var ltmax: Int   # text length bucket (KREA2_LTMAX, train_krea2.mojo:407)
    var s_img: Int   # image token count (IMGLEN, train_krea2.mojo:386)
    var s_cond: Int  # condition token count (new KREA2_CONDLEN; 0 = no-cond build)
    var lt: Int      # this sample's real caption length (<= ltmax)

    def validate(self) raises:
        if self.ltmax <= 0 or self.s_img <= 0:
            raise Error("Krea2OminiLayout: ltmax and s_img must be positive")
        if self.s_cond < 0:
            raise Error("Krea2OminiLayout: s_cond must be >= 0")
        if self.lt < 0 or self.lt > self.ltmax:
            raise Error("Krea2OminiLayout: lt out of [0, ltmax]")

    # ── segment offsets/lengths in the COMBINED (reordered) sequence ─────────
    def lfull(self) -> Int:
        return self.ltmax + self.s_img + self.s_cond

    def txt_real_off(self) -> Int:
        return 0

    def txt_real_len(self) -> Int:
        return self.lt

    def img_off(self) -> Int:
        return self.lt

    def img_len(self) -> Int:
        return self.s_img

    def cond_off(self) -> Int:
        return self.lt + self.s_img

    def cond_len(self) -> Int:
        return self.s_cond

    def pad_off(self) -> Int:
        return self.lt + self.s_img + self.s_cond

    def pad_len(self) -> Int:
        return self.ltmax - self.lt

    def real_len(self) -> Int:
        """Flash-padmask valid CONTIGUOUS-PREFIX length (== pad_off): txt_real +
        img + cond are all attention-valid; [real_len, lfull) is masked tail."""
        return self.lt + self.s_img + self.s_cond

    # ── flash-padmask contiguous-prefix constraint (train_krea2.mojo:795-798,
    #    840-844): valid segments tile [0, real_len) with no gaps, pad is
    #    exactly the tail. Arithmetic identities — raise if the layout ever
    #    stops satisfying them (e.g. someone reorders segments). ──────────────
    def check_flash_prefix(self) raises:
        if self.txt_real_off() != 0:
            raise Error("flash prefix: TXT_real must start at row 0")
        if self.img_off() != self.txt_real_off() + self.txt_real_len():
            raise Error("flash prefix: IMG must abut TXT_real")
        if self.cond_off() != self.img_off() + self.img_len():
            raise Error("flash prefix: COND must abut IMG")
        if self.pad_off() != self.cond_off() + self.cond_len():
            raise Error("flash prefix: TXT_pad must abut COND")
        if self.pad_off() != self.real_len():
            raise Error("flash prefix: pad must start exactly at real_len")
        if self.pad_off() + self.pad_len() != self.lfull():
            raise Error("flash prefix: pad tail must end at LFULL")

    # ── per-segment modulation selector (intake §1.5/§3.3) ───────────────────
    def mod_selector(self, row: Int) -> Int:
        """Which modulation set row uses: MOD_T (mods(t)) for TXT_real+IMG,
        MOD_COND (mods(t=0)) for COND, MOD_PAD (don't-care) for the pad tail."""
        if row < self.cond_off():
            return KREA2_OMINI_MOD_T
        if row < self.pad_off():
            return KREA2_OMINI_MOD_COND
        return KREA2_OMINI_MOD_PAD

    # ── reorder maps ─────────────────────────────────────────────────────────
    # SOURCE (cache/builder) order: [TXT padded(ltmax) | IMG(s_img) | COND(s_cond)]
    # (krea2_build_pos builds txt-then-img today, cache_reader.mojo:84-100; the
    # edit builder appends the cond grid after img). COMBINED order is this
    # struct's layout. These maps generalize _reorder_pos_for_combined
    # (train_krea2.mojo:880-894) and _combined_text_grad (train_krea2.mojo:969-982).
    def combined_src_row(self, row: Int) -> Int:
        """Source-order row index feeding combined row `row` (gather map for the
        pos table and any per-row side data)."""
        if row < self.img_off():                       # TXT_real
            return row
        if row < self.cond_off():                      # IMG
            return self.ltmax + (row - self.img_off())
        if row < self.pad_off():                       # COND
            return self.ltmax + self.s_img + (row - self.cond_off())
        # TXT_pad: combined pad slot p maps to source txt row lt + p.
        return self.lt + (row - self.pad_off())

    def text_grad_src_row(self, txt_row: Int) -> Int:
        """Combined-sequence row holding source TEXT row txt_row (0 <= txt_row
        < ltmax) — the gather _combined_text_grad performs: real text rows sit
        at the head, pad text rows after IMG+COND."""
        if txt_row < self.lt:
            return txt_row
        return txt_row + self.s_img + self.s_cond


# ── block per-segment modulation split (seam B.1) ────────────────────────────
# THE contract between this module and krea2_block.krea2_single_stream_block_lora:
# ONE row boundary. Rows [0, split) take the mods(t) chunks, rows [split, L) take
# the mods(t=0) chunks. Returning -1 when there is no condition segment is what
# makes the CONDLEN=0 build bit-equal to today's trainer: the block's guard
# (`split > 0 and split < L`) rejects -1 and runs the pre-existing uniform-
# modulation code path unchanged, so the caller can pass this value
# UNCONDITIONALLY and never branch itself.
#
# WHY the split is a single boundary and the TXT_pad tail lands in the t=0 span:
# that is exactly what the ai-toolkit krea2 reference does (mmdit.py
# SingleStreamBlock.forward tuple-vec branch `(vec, refvec, split)`), and pad
# rows' values are never read downstream, so matching the reference costs
# nothing and removes a needless divergence. (An earlier draft of the seam plan
# in this file's header suggested keeping mods(t) on the pad tail — the
# reference-identical choice below supersedes it.)
def krea2_omini_mod_split(lay: Krea2OminiLayout) -> Int:
    """Row boundary for the block's per-segment modulation, or -1 when the
    layout has NO condition segment (s_cond == 0 -> unchanged krea2 path)."""
    if lay.s_cond <= 0:
        return -1
    return lay.cond_off()


# ── position tables over the new layout (seam D; krea2_cache_reader.mojo:84-100)
# SOURCE order [TXT zeros(ltmax) | IMG grid(gh*gw) | COND grid(gh*gw)], flattened
# token-major as [(ltmax + s_img + s_cond) * 3] F32 — index t*3 + a holds token
# t's position along axis a, which is precisely the flat layout
# build_krea2_rope consumes (models/dit/krea2_dit.mojo:535, and the trainer's
# `reshape(pos_re, [LFULL*3])` at train_krea2.mojo:874). The first two sections
# are byte-for-byte what krea2_build_pos already emits (cache_reader.mojo:88-99);
# the COND section is appended via krea2_omini_cond_pos() (EDIT: dh=dw=0,
# scale=1.0 -> cond grid == img grid).
def krea2_omini_pos_src(
    lay: Krea2OminiLayout, gh: Int, gw: Int, dh: Int, dw: Int, scale: Float32
) raises -> List[Float32]:
    """SOURCE-order flat position table [(ltmax+s_img+s_cond)*3] F32."""
    lay.validate()
    if gh <= 0 or gw <= 0:
        raise Error("krea2_omini_pos_src: gh/gw must be positive")
    if gh * gw != lay.s_img:
        raise Error("krea2_omini_pos_src: gh*gw != s_img")
    if lay.s_cond != 0 and gh * gw != lay.s_cond:
        raise Error(
            "krea2_omini_pos_src: EDIT condition grid must equal the img grid"
        )
    var host = List[Float32]()
    for _ in range(lay.ltmax * 3):
        host.append(Float32(0.0))            # txt positions: all zeros
    for hi in range(gh):                     # IMG grid (0, hi, wi), row-major
        for wi in range(gw):
            host.append(Float32(0.0))
            host.append(Float32(hi))
            host.append(Float32(wi))
    if lay.s_cond > 0:                       # COND grid (0, hi+dh, wi+dw)*scale
        for hi in range(gh):
            for wi in range(gw):
                var p = krea2_omini_cond_pos(hi, wi, dh, dw, scale)
                host.append(p.g)
                host.append(p.h)
                host.append(p.w)
    return host^


def krea2_omini_pos_combined(
    lay: Krea2OminiLayout, gh: Int, gw: Int, dh: Int, dw: Int, scale: Float32
) raises -> List[Float32]:
    """COMBINED-order flat position table [lfull*3] F32 — the SOURCE table
    gathered through combined_src_row(). Generalizes the trainer's
    _reorder_pos_for_combined (train_krea2.mojo:880-894): with s_cond == 0 the
    gather IS that function (checked in main())."""
    var src = krea2_omini_pos_src(lay, gh, gw, dh, dw, scale)
    var out = List[Float32]()
    for row in range(lay.lfull()):
        var s = lay.combined_src_row(row)
        out.append(src[s * 3 + 0])
        out.append(src[s * 3 + 1])
        out.append(src[s * 3 + 2])
    return out^


# ── condition-token RoPE position (intake §1.2/§3.2) ─────────────────────────
# OminiControl Condition.encode order (flux_omini.py:128-141): delta FIRST, then
# scale with bias (scale-1)/2, h/w axes only; global axis stays 0. Units are
# latent-grid/2 cells = pixels/16 — same units as krea2's (0, hi, wi) grid
# (krea2_build_pos, cache_reader.mojo:94-98), so the floats feed
# build_krea2_rope unchanged. EDIT condition type: dh=dw=0, scale=1.0 → cond
# positions EQUAL the img grid (spatial overlap).
@fieldwise_init
struct Krea2OminiCondPos(Copyable, Movable):
    var g: Float32  # axis 0 (global) — always 0
    var h: Float32  # axis 1
    var w: Float32  # axis 2


def krea2_omini_cond_pos(
    hi: Int, wi: Int, dh: Int, dw: Int, scale: Float32
) -> Krea2OminiCondPos:
    var h = Float32(hi + dh)
    var w = Float32(wi + dw)
    if scale != 1.0:
        var bias = (scale - 1.0) / 2.0
        h = h * scale + bias
        w = w * scale + bias
    return Krea2OminiCondPos(Float32(0.0), h, w)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT CHECKS (structure math only — CPU asserts, NO numeric-parity claim).
# 512px shapes: S_IMG = S_COND = (64/2)*(64/2) = 1024; LTMAX = 384 (the
# KREA2_LTMAX default, train_krea2.mojo:407). Exits nonzero on any mismatch.
# ══════════════════════════════════════════════════════════════════════════════
def _check(cond: Bool, name: String) raises:
    if not cond:
        raise Error("FAIL " + name)
    print("PASS", name)


def _check_layout_invariants(lay: Krea2OminiLayout, tag: String) raises:
    lay.validate()
    lay.check_flash_prefix()
    # Segment lengths tile LFULL exactly.
    var total = (
        lay.txt_real_len() + lay.img_len() + lay.cond_len() + lay.pad_len()
    )
    _check(total == lay.lfull(), tag + ": segment lengths tile LFULL")
    # combined_src_row is a BIJECTION [0,LFULL) -> [0,LFULL) (every source row
    # gathered exactly once — the reorder loses/duplicates nothing).
    var seen = List[Int]()
    for _ in range(lay.lfull()):
        seen.append(0)
    for row in range(lay.lfull()):
        var s = lay.combined_src_row(row)
        if s < 0 or s >= lay.lfull():
            raise Error("FAIL " + tag + ": src row out of range")
        seen[s] += 1
    var dup = 0
    for i in range(lay.lfull()):
        if seen[i] != 1:
            dup += 1
    _check(dup == 0, tag + ": combined_src_row bijection")
    # Text-grad gather agrees with the reorder: source text row j lands at
    # combined row text_grad_src_row(j), and gathering back yields j.
    var txt_ok = True
    for j in range(lay.ltmax):
        var row = lay.text_grad_src_row(j)
        if lay.combined_src_row(row) != j:
            txt_ok = False
    _check(txt_ok, tag + ": text_grad_src_row inverts combined_src_row on text")
    # Modulation selector: segment row counts + contiguity (MOD_T then MOD_COND
    # then MOD_PAD, no interleaving).
    var n_t = 0
    var n_c = 0
    var n_p = 0
    var last = -1
    var monotone = True
    for row in range(lay.lfull()):
        var s = lay.mod_selector(row)
        if s < last:
            monotone = False
        last = s
        if s == KREA2_OMINI_MOD_T:
            n_t += 1
        elif s == KREA2_OMINI_MOD_COND:
            n_c += 1
        else:
            n_p += 1
    _check(
        n_t == lay.lt + lay.s_img and n_c == lay.s_cond
        and n_p == lay.ltmax - lay.lt and monotone,
        tag + ": mod selector segments (t | t=0 | pad, contiguous)",
    )


def main() raises:
    comptime LTMAX = 384      # KREA2_LTMAX default (train_krea2.mojo:407)
    comptime S_IMG = 1024     # 512px IMGLEN (train_krea2.mojo:384-386)
    comptime S_COND = 1024    # EDIT condition @512px, same grid as target
    comptime LFULL_EDIT = LTMAX + S_IMG + S_COND

    # ── 512px EDIT layout, a realistic lt ────────────────────────────────────
    var lay = Krea2OminiLayout(LTMAX, S_IMG, S_COND, 282)
    _check(lay.lfull() == LFULL_EDIT and LFULL_EDIT == 2432, "LFULL_EDIT = 2432")
    _check(lay.txt_real_off() == 0 and lay.txt_real_len() == 282, "TXT_real [0,282)")
    _check(lay.img_off() == 282 and lay.img_len() == 1024, "IMG [282,1306)")
    _check(lay.cond_off() == 1306 and lay.cond_len() == 1024, "COND [1306,2330)")
    _check(lay.pad_off() == 2330 and lay.pad_len() == 102, "TXT_pad [2330,2432)")
    _check(lay.real_len() == 2330, "real_len = lt + S_IMG + S_COND = 2330")
    _check_layout_invariants(lay, "lt=282")

    # ── lt sweep incl. boundaries (lt=0 no real text; lt=LTMAX no pad) ───────
    var lts = [0, 1, 383, 384]
    for i in range(len(lts)):
        var l2 = Krea2OminiLayout(LTMAX, S_IMG, S_COND, lts[i])
        _check_layout_invariants(l2, "lt=" + String(lts[i]))

    # ── CONDLEN=0 degeneration == today's [TXT_real | IMG | TXT_pad] trainer
    #    layout (regression shape for the bit-equal no-cond build, intake C2) ─
    var l0 = Krea2OminiLayout(LTMAX, S_IMG, 0, 282)
    _check_layout_invariants(l0, "condlen=0")
    _check(
        l0.lfull() == LTMAX + S_IMG and l0.real_len() == 282 + S_IMG
        and l0.pad_off() == 282 + S_IMG,
        "condlen=0: LFULL/real_len match current trainer (LTMAX+IMGLEN, lt+IMGLEN)",
    )
    # _reorder_pos_for_combined equivalence (train_krea2.mojo:886-891): img rows
    # gather from source [LTMAX, LTMAX+S_IMG), pad rows from source [lt, LTMAX).
    _check(
        l0.combined_src_row(282) == LTMAX
        and l0.combined_src_row(282 + S_IMG - 1) == LTMAX + S_IMG - 1
        and l0.combined_src_row(282 + S_IMG) == 282
        and l0.combined_src_row(l0.lfull() - 1) == LTMAX - 1,
        "condlen=0: gather map == _reorder_pos_for_combined",
    )
    # _combined_text_grad equivalence (train_krea2.mojo:978-982): d_real from
    # rows [0,lt), d_pad from rows [lt+IMGLEN, ...).
    _check(
        l0.text_grad_src_row(0) == 0 and l0.text_grad_src_row(281) == 281
        and l0.text_grad_src_row(282) == 282 + S_IMG
        and l0.text_grad_src_row(LTMAX - 1) == LTMAX - 1 + S_IMG,
        "condlen=0: text-grad map == _combined_text_grad",
    )

    # ── EDIT positions: delta [0,0], scale 1.0 → cond pos == img grid pos ────
    var pos_ok = True
    for hi in range(4):
        for wi in range(4):
            var p = krea2_omini_cond_pos(hi, wi, 0, 0, Float32(1.0))
            if p.g != 0.0 or p.h != Float32(hi) or p.w != Float32(wi):
                pos_ok = False
    _check(pos_ok, "EDIT cond pos (delta [0,0], scale 1.0) == img grid pos")
    # Subject-style delta (negative w offset) + OC2 compact-token scale spot
    # checks against the flux_omini.py:128-141 math.
    var ps = krea2_omini_cond_pos(0, 5, 0, -32, Float32(1.0))
    _check(ps.h == 0.0 and ps.w == -27.0, "subject delta [0,-32]: w = wi - 32")
    var pc = krea2_omini_cond_pos(3, 1, 0, 0, Float32(2.0))
    _check(pc.h == 6.5 and pc.w == 2.5, "position_scale 2.0: v*2 + 0.5")

    # ── modulation split contract (what krea2_block consumes) ────────────────
    _check(krea2_omini_mod_split(lay) == 1306, "mod split = cond_off when s_cond>0")
    _check(
        krea2_omini_mod_split(l0) == -1,
        "condlen=0: mod split = -1 (block runs the UNCHANGED uniform path)",
    )

    # ── position tables (host math; the C2 gate compares these to the CUDA
    #    oracle's kin_pos_src / kin_pos) ──────────────────────────────────────
    comptime GRID = 32                       # 64x64 latent, patch 2 -> 32x32
    var psrc = krea2_omini_pos_src(lay, GRID, GRID, 0, 0, Float32(1.0))
    _check(len(psrc) == LFULL_EDIT * 3, "pos_src length = LFULL_EDIT*3")
    var txt_zero = True
    for i in range(LTMAX * 3):
        if psrc[i] != 0.0:
            txt_zero = False
    _check(txt_zero, "pos_src: text section all-zero (krea2_build_pos:88-89)")
    var grid_ok = True
    for hi in range(GRID):
        for wi in range(GRID):
            var ti = LTMAX + hi * GRID + wi
            var tc = LTMAX + S_IMG + hi * GRID + wi
            if psrc[ti * 3] != 0.0 or psrc[ti * 3 + 1] != Float32(hi):
                grid_ok = False
            if psrc[ti * 3 + 2] != Float32(wi):
                grid_ok = False
            # EDIT: cond grid EQUALS img grid (delta [0,0], scale 1.0)
            if psrc[tc * 3] != psrc[ti * 3]:
                grid_ok = False
            if psrc[tc * 3 + 1] != psrc[ti * 3 + 1]:
                grid_ok = False
            if psrc[tc * 3 + 2] != psrc[ti * 3 + 2]:
                grid_ok = False
    _check(grid_ok, "pos_src: img grid (0,hi,wi) and EDIT cond grid == img grid")

    var pcomb = krea2_omini_pos_combined(lay, GRID, GRID, 0, 0, Float32(1.0))
    _check(len(pcomb) == LFULL_EDIT * 3, "pos_combined length = LFULL_EDIT*3")
    var gather_ok = True
    for row in range(lay.lfull()):
        var s = lay.combined_src_row(row)
        for a in range(3):
            if pcomb[row * 3 + a] != psrc[s * 3 + a]:
                gather_ok = False
    _check(gather_ok, "pos_combined == pos_src gathered by combined_src_row")

    # condlen=0: the combined pos table must equal the trainer's
    # _reorder_pos_for_combined output (train_krea2.mojo:886-891) applied to
    # krea2_build_pos's [txt zeros(LTMAX) | img grid] table.
    var p0 = krea2_omini_pos_combined(l0, GRID, GRID, 0, 0, Float32(1.0))
    var p0src = krea2_omini_pos_src(l0, GRID, GRID, 0, 0, Float32(1.0))
    _check(len(p0) == (LTMAX + S_IMG) * 3, "condlen=0: pos length = LTMAX+S_IMG")
    var re_ok = True
    for r in range(282):                                     # TXT_real
        for a in range(3):
            if p0[r * 3 + a] != p0src[r * 3 + a]:
                re_ok = False
    for r in range(S_IMG):                                   # IMG
        for a in range(3):
            if p0[(282 + r) * 3 + a] != p0src[(LTMAX + r) * 3 + a]:
                re_ok = False
    for r in range(LTMAX - 282):                             # TXT_pad
        for a in range(3):
            if p0[(282 + S_IMG + r) * 3 + a] != p0src[(282 + r) * 3 + a]:
                re_ok = False
    _check(re_ok, "condlen=0: pos_combined == _reorder_pos_for_combined")

    print("krea2_omini_layout: ALL LAYOUT CHECKS PASSED (structure only)")
