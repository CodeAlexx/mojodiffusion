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
#    2. Cond-row LoRA routing (intake §1.3/§3.4 — adapter acts ONLY on cond
#       rows; text+img rows run frozen base): base matmuls stay full-sequence
#       (`_base_fwd` def :187; sites :587-590 q/k/v/gate and :697-698
#       mlp_gate/mlp_up; `_linear_lora` sites :688 wo idx4 and :726 mlp_down
#       idx7). The LoRA delta (`_lora_fwd` ungrouped sites :623-635 and
#       :718-724; grouped :592-622/:700-717; and inside `_linear_lora` :155)
#       changes input from full `xm` to `slice(xm, 1, cond_off, s_cond)` and
#       the add becomes a scatter-add into rows [cond_off, cond_off+s_cond) of
#       the base output. M (LoRA row count, :569) becomes s_cond, not L.
#
# C) BLOCK BACKWARD — krea2_block.mojo
#    `krea2_single_stream_block_lora_backward` (def :1635, `_mod6` recompute at
#    :1661) and the device path `krea2_single_stream_block_lora_backward_dev`
#    (def :2353, `_mod6` at :2372):
#    1. dX through frozen base stays full-sequence via `_base_dx` (def :222)
#       inside `_linear_bwd_dx` (:956, site :965), `_linear_bwd_dx_dev` (:992,
#       site :1001), and the grouped variants `_linear_bwd_dx_group2/4[_dev]`
#       (:1077/:1148/:1260/:1325; `_base_dx` sites :1110-1111, :1202-1205,
#       :1293-1294, :1379-1382).
#    2. dA/dB are computed from the COND-ROW SLICES of (d_y, xm) only —
#       same slice bounds as B.2; Krea2LoraGrad/Krea2BlockGrads structs reused.
#    3. d_mod accumulates per segment: grads of mods(t) chunks gather over rows
#       [0, cond_off), grads of mods(0) chunks over [cond_off, pad_off) —
#       mirrored at every modulate/residual_gate backward in both paths.
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

    print("krea2_omini_layout: ALL LAYOUT CHECKS PASSED (structure only)")
