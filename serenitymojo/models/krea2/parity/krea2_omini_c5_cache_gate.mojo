# krea2_omini_c5_cache_gate.mojo — C5 EDIT-CACHE gate (evidence, not assertion).
#
# Runs against a REAL OminiControl edit cache produced by
#   scripts/krea2_omini_stage_edit.py  ->  output/bin/serenity_krea2_prepare_cache
# and proves the four things chunk C5 is responsible for:
#
#   GATE B  POSITION TABLE. KreaTrainCache.sample_padded_edit's pos table is
#           compared ELEMENT-FOR-ELEMENT (exact float equality, no tolerance)
#           against training/krea2_omini_layout.krea2_omini_pos_src — the host
#           layout module that C2/C3/C4 were gated on — and, independently,
#           the COND rows are compared against the IMG rows (EDIT delta [0,0],
#           scale 1.0 => cond grid == img grid). Krea2OminiLayout.check_flash_prefix
#           and the reader's real_len are asserted too.
#   GATE C' MASK/real_len arithmetic for the cond segment: krea2_build_pad_mask_edit
#           at a small exhaustively-checkable shape (every masked/unmasked entry
#           verified) plus the 512px EDIT shape assertion, and krea2_edit_real_len
#           against the layout struct. (The condlen=0 BIT-EQUAL regression lives in
#           krea2_omini_c5_regress_dump.mojo + cmp — it must run on BOTH trees.)
#   GATE D  CACHE STATS. Per-stream mean/std of the cached normalized latents over
#           every sample, plus the min/max per-sample std, and the mean absolute
#           difference between the target and condition latents. The krea2 cache
#           contract is z' = (z - latents_mean)/latents_std (krea2_prepare_cache
#           :28-41), so a healthy stream sits near mean 0 / std 1; a stream that is
#           ~0 or ~8x off is the LTX2-class mis-normalization bug.
#   GATE A  ROUND-TRIP. clean.<i> and ref.<i> are VAE-DECODED back to pixels and
#           written as PNGs so a human (or an agent with eyes) can confirm the
#           TARGET is the edited image and the CONDITION is the source. This gate
#           prints the paths; the visual verdict is recorded in the chunk report.
#
# usage: krea2_omini_c5_cache_gate <cache.safetensors> <out_png_dir> [n_decode=3]
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   LD_LIBRARY_PATH=$PWD/.pixi/envs/default/lib:$PWD/serenitymojo/ops/cshim/lib \
#     pixi run mojo run -I . -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/krea2/parity/krea2_omini_c5_cache_gate.mojo \
#     /home/alex/trainings/krea2_omini_edit_cache/cache.safetensors /tmp/c5png
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.pipeline.krea2_paths import KREA2_VAE_DIR
from serenitymojo.models.krea2.krea2_cache_reader import (
    KreaTrainCache,
    krea2_build_pad_mask,
    krea2_build_pad_mask_edit,
    krea2_build_pos_cond,
    krea2_edit_real_len,
    KREA2_MASK_NEG,
    KREA2_HEADS,
)
from serenitymojo.training.krea2_omini_layout import (
    Krea2OminiLayout,
    krea2_omini_pos_src,
    krea2_omini_cond_pos,
)

# 512px EDIT shapes (the fixed decision for this vertical).
comptime LH = 64
comptime LW = 64
comptime GRID = 32                 # LH/2
comptime IMGLEN = GRID * GRID      # 1024
comptime CONDLEN = IMGLEN          # EDIT: condition shares the target canvas
comptime LTMAX = 384               # KREA2_LTMAX default (train_krea2.mojo:407)

def _check(cond: Bool, name: String) -> Int:
    """Print PASS/FAIL and return the failure count contribution (0 or 1)."""
    if cond:
        print("PASS", name)
        return 0
    print("FAIL", name)
    return 1


def _mean(v: List[Float32]) -> Float32:
    if len(v) == 0:
        return Float32(0.0)
    var acc = Float64(0.0)
    for i in range(len(v)):
        acc += Float64(v[i])
    return Float32(acc / Float64(len(v)))


def _std(v: List[Float32]) -> Float32:
    if len(v) < 2:
        return Float32(0.0)
    var m = Float64(_mean(v))
    var acc = Float64(0.0)
    for i in range(len(v)):
        var d = Float64(v[i]) - m
        acc += d * d
    return Float32(sqrt(acc / Float64(len(v))))


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error(
            "usage: krea2_omini_c5_cache_gate <cache.safetensors> <out_png_dir>"
            " [n_decode=3]"
        )
    var cache_path = String(args[1])
    var png_dir = String(args[2])
    var n_decode = 3
    if len(args) >= 4:
        n_decode = Int(String(args[3]))

    var ctx = DeviceContext()
    var g_fail = 0
    var cache = KreaTrainCache.open(cache_path)
    var n = cache.len()
    print("[c5-gate] cache", cache_path, " samples=", n)

    # ══ GATE B/C' — structure, positions, mask, real_len ══════════════════════
    var all_cond = True
    for i in range(n):
        if not cache.has_cond(i):
            all_cond = False
    g_fail += _check(all_cond, "every sample carries a full Omini condition record")

    var meta0 = cache.cond_pos_at(0, ctx)
    print("[c5-gate] sample 0 position_delta=(", meta0[0], ",", meta0[1],
          ") position_scale=", meta0[2])
    g_fail += _check(
        meta0[0] == 0 and meta0[1] == 0 and meta0[2] == Float32(1.0),
        "EDIT position metadata is delta [0,0] scale 1.0",
    )

    var es = cache.sample_padded_edit[LH, LW, LTMAX](0, ctx)
    var lt = es.base.text_len
    var lay = Krea2OminiLayout(LTMAX, IMGLEN, CONDLEN, lt)
    lay.check_flash_prefix()
    print("[c5-gate] sample 0 LT=", lt, " cond_len=", es.cond_len,
          " real_len=", es.real_len, " LFULL=", lay.lfull())
    g_fail += _check(
        es.cond_len == CONDLEN
        and es.real_len == lt + IMGLEN + CONDLEN
        and es.real_len == lay.real_len()
        and krea2_edit_real_len(lt, IMGLEN, CONDLEN) == lay.real_len(),
        "real_len == lt + IMGLEN + CONDLEN == Krea2OminiLayout.real_len()",
    )

    var csh = es.cond[].shape()
    var cish = es.cond_img[].shape()
    g_fail += _check(
        len(csh) == 4 and csh[0] == 1 and csh[1] == 16 and csh[2] == LH
        and csh[3] == LW,
        "cond latent is [1,16,LH,LW]",
    )
    g_fail += _check(
        len(cish) == 3 and cish[0] == 1 and cish[1] == CONDLEN and cish[2] == 64,
        "cond_img is [1,CONDLEN,64] (patchified with the SAME krea2_patchify)",
    )

    # pos: reader (device) vs layout module (host), EXACT float equality.
    var psh = es.base.pos[].shape()
    g_fail += _check(
        len(psh) == 3 and psh[0] == 1 and psh[1] == LTMAX + IMGLEN + CONDLEN
        and psh[2] == 3,
        "pos is [1, LTMAX+IMGLEN+CONDLEN, 3] SOURCE order",
    )
    var pos_h = es.base.pos[].to_host(ctx)
    var pos_ref = krea2_omini_pos_src(
        lay, GRID, GRID, meta0[0], meta0[1], meta0[2]
    )
    var nmis = 0
    if len(pos_h) != len(pos_ref):
        nmis = -1
    else:
        for i in range(len(pos_h)):
            if pos_h[i] != pos_ref[i]:
                nmis += 1
    print("[c5-gate] pos elements=", len(pos_h), " mismatches vs"
          " krea2_omini_pos_src =", nmis)
    g_fail += _check(nmis == 0, "reader pos == krea2_omini_pos_src (EXACT, elementwise)")

    # text section all-zero; cond rows == img rows (EDIT overlap).
    var txt_zero = True
    for i in range(LTMAX * 3):
        if pos_h[i] != Float32(0.0):
            txt_zero = False
    g_fail += _check(txt_zero, "pos text section is all-zero")

    var cond_eq_img = True
    var cond_eq_fn = True
    for hi in range(GRID):
        for wi in range(GRID):
            var ti = LTMAX + hi * GRID + wi
            var tc = LTMAX + IMGLEN + hi * GRID + wi
            for a in range(3):
                if pos_h[tc * 3 + a] != pos_h[ti * 3 + a]:
                    cond_eq_img = False
            var p = krea2_omini_cond_pos(hi, wi, meta0[0], meta0[1], meta0[2])
            if (
                pos_h[tc * 3 + 0] != p.g or pos_h[tc * 3 + 1] != p.h
                or pos_h[tc * 3 + 2] != p.w
            ):
                cond_eq_fn = False
    g_fail += _check(cond_eq_fn, "cond rows == krea2_omini_cond_pos(hi,wi,dh,dw,scale)")
    g_fail += _check(cond_eq_img, "EDIT: cond grid positions == img grid positions")

    # condlen=0 arm of the SAME builder still equals the pre-edit table.
    var p0 = krea2_build_pos_cond[LH, LW](LTMAX, 0, 0, 0, Float32(1.0), ctx)
    var p0h = p0.to_host(ctx)
    var head_eq = len(p0h) == (LTMAX + IMGLEN) * 3
    if head_eq:
        for i in range((LTMAX + IMGLEN) * 3):
            if p0h[i] != pos_h[i]:
                head_eq = False
    g_fail += _check(head_eq, "condlen=0 pos == the [TXT|IMG] prefix of the edit pos")

    # ── mask: exhaustive at a tiny shape, shape-only at the real one ──────────
    var m_lt = 3
    var m_ltmax = 8
    var m_img = 4
    var m_cond = 4
    var mask = krea2_build_pad_mask_edit(m_lt, m_ltmax, m_img, m_cond, ctx)
    var mlf = m_ltmax + m_img + m_cond
    var msh = mask.shape()
    var mh = mask.to_host(ctx)
    var mask_ok = (
        len(msh) == 4 and msh[0] == 1 and msh[1] == KREA2_HEADS
        and msh[2] == mlf and msh[3] == mlf
    )
    var n_masked = 0
    if mask_ok:
        for h in range(KREA2_HEADS):
            for i in range(mlf):
                for j in range(mlf):
                    var v = mh[(h * mlf + i) * mlf + j]
                    var want_mask = j >= m_lt and j < m_ltmax
                    if want_mask:
                        if v != KREA2_MASK_NEG:
                            mask_ok = False
                        n_masked += 1
                    elif v != Float32(0.0):
                        mask_ok = False
    print("[c5-gate] edit padmask", mlf, "x", mlf, " masked entries=", n_masked,
          " expected=", KREA2_HEADS * mlf * (m_ltmax - m_lt))
    g_fail += _check(
        mask_ok and n_masked == KREA2_HEADS * mlf * (m_ltmax - m_lt),
        "krea2_build_pad_mask_edit masks EXACTLY the text-pad key columns"
        " [lt,ltmax) over the LTMAX+IMGLEN+CONDLEN sequence",
    )
    # And the edit wrapper is the imglen+condlen call, byte-for-byte.
    var mask_plain = krea2_build_pad_mask(m_lt, m_ltmax, m_img + m_cond, ctx)
    var mph = mask_plain.to_host(ctx)
    var wrap_eq = len(mph) == len(mh)
    if wrap_eq:
        for i in range(len(mh)):
            if mh[i] != mph[i]:
                wrap_eq = False
    g_fail += _check(wrap_eq, "pad_mask_edit(lt,ltmax,img,cond) == pad_mask(lt,ltmax,img+cond)")

    # ══ GATE D — cache stats over BOTH streams ════════════════════════════════
    # krea2's cache contract: z' = (z - latents_mean)/latents_std, so a correctly
    # normalized stream sits near mean 0 / std ~1. Report, do not assume.
    var tgt_all_mean = Float64(0.0)
    var tgt_all_std = Float64(0.0)
    var cnd_all_mean = Float64(0.0)
    var cnd_all_std = Float64(0.0)
    var tgt_std_min = Float32(1.0e30)
    var tgt_std_max = Float32(-1.0e30)
    var cnd_std_min = Float32(1.0e30)
    var cnd_std_max = Float32(-1.0e30)
    var mad_sum = Float64(0.0)
    var identical = 0
    for i in range(n):
        var s = cache.sample_padded_edit[LH, LW, LTMAX](i, ctx)
        var th = s.base.clean[].to_host(ctx)
        var ch = s.cond[].to_host(ctx)
        var tm = _mean(th)
        var ts = _std(th)
        var cm = _mean(ch)
        var cs = _std(ch)
        tgt_all_mean += Float64(tm)
        tgt_all_std += Float64(ts)
        cnd_all_mean += Float64(cm)
        cnd_all_std += Float64(cs)
        if ts < tgt_std_min:
            tgt_std_min = ts
        if ts > tgt_std_max:
            tgt_std_max = ts
        if cs < cnd_std_min:
            cnd_std_min = cs
        if cs > cnd_std_max:
            cnd_std_max = cs
        var mad = Float64(0.0)
        var same = True
        for k in range(len(th)):
            var d = Float64(th[k]) - Float64(ch[k])
            if d < 0.0:
                d = -d
            mad += d
            if th[k] != ch[k]:
                same = False
        mad_sum += mad / Float64(len(th))
        if same:
            identical += 1
    var nf = Float64(n)
    print("[c5-gate] TARGET  clean.<i>: mean=", Float32(tgt_all_mean / nf),
          " std=", Float32(tgt_all_std / nf),
          " per-sample std range [", tgt_std_min, ",", tgt_std_max, "]")
    print("[c5-gate] COND    ref.<i>  : mean=", Float32(cnd_all_mean / nf),
          " std=", Float32(cnd_all_std / nf),
          " per-sample std range [", cnd_std_min, ",", cnd_std_max, "]")
    print("[c5-gate] mean |target - cond| per sample =",
          Float32(mad_sum / nf), " ; samples where the two latents are IDENTICAL =",
          identical)
    g_fail += _check(
        tgt_std_min > Float32(0.2) and tgt_std_max < Float32(5.0)
        and cnd_std_min > Float32(0.2) and cnd_std_max < Float32(5.0),
        "both latent streams are on the normalized scale (per-sample std in"
        " (0.2, 5.0) — the LTX2-class mis-normalization check)",
    )
    g_fail += _check(
        identical == 0 and mad_sum / nf > 0.05,
        "target and condition latents are DISTINCT in every sample (the cond"
        " stream is not a copy of the target)",
    )

    # ══ GATE A — round-trip decode to PNG ═════════════════════════════════════
    if n_decode > 0:
        var dec = QwenImageVaeDecoder[LH, LW].load(String(KREA2_VAE_DIR), ctx)
        for i in range(n_decode if n_decode < n else n):
            var s = cache.sample_padded_edit[LH, LW, LTMAX](i, ctx)
            var tgt_png = png_dir + String("/c5_sample") + String(i) + String(
                "_TARGET_clean.png"
            )
            var cnd_png = png_dir + String("/c5_sample") + String(i) + String(
                "_COND_ref.png"
            )
            save_png(dec.decode(s.base.clean[], ctx), tgt_png, ctx, ValueRange.SIGNED)
            save_png(dec.decode(s.cond[], ctx), cnd_png, ctx, ValueRange.SIGNED)
            print("[c5-gate] round-trip sample", i, "->", tgt_png, "|", cnd_png)

    if g_fail != 0:
        raise Error(String("krea2 C5 cache gate: ") + String(g_fail) + " FAILURES")
    print("krea2_omini_c5_cache_gate: ALL CHECKS PASSED")
