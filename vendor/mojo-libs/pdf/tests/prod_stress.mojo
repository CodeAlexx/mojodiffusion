# pdf/tests/prod_stress.mojo
# Adversarial stress test for the production PDF paths -> /tmp/pdf_prostress.pdf
#
# Pushes: 32 pages (each FlateDecode, embedded-font unicode line + wrapped
# paragraph + link annot + color), a flat outline of 32 sibling bookmarks,
# long/empty/special-char strings in BOTH standard-14 and embedded paths, a
# codepoint NOT in the font cmap (gid 0), and two image sizes (1x1 and 64x64).

from pdf.document import PdfDoc
from pdf.objects import append_str, append_int
from pdf.text import begin_text, end_text, set_font, text_pos, show_text, standard_font_object
from pdf.ttf import load_ttf
from pdf.font_embed import embed_font, show_text_unicode, collect_codepoints, add_codepoints
from pdf.layout import wrap_text, align_x
from pdf.image import image_xobject_stream, draw_image
from pdf.content import Content
from pdf.extras import (
    set_fill_gray, set_fill_cmyk, link_uri_annotation,
    extgstate_object, apply_gstate, outline_item_object, outlines_root_object,
)


def _build_resources(std_font_num: Int, type0_font_num: Int, image_num: Int, extg_num: Int) raises -> List[UInt8]:
    var r = List[UInt8]()
    append_str(r, "<< /Font << /F1 ")
    append_int(r, std_font_num)
    append_str(r, " 0 R /F2 ")
    append_int(r, type0_font_num)
    append_str(r, " 0 R >> /XObject << /Im1 ")
    append_int(r, image_num)
    append_str(r, " 0 R >> /ExtGState << /GS1 ")
    append_int(r, extg_num)
    append_str(r, " 0 R >> >>")
    return r^


def main() raises:
    var doc = PdfDoc()
    doc.set_info(String("Stress (test)"), String("agent \\ six"), String("a (b) c"), String("MojoPDF"))

    var std_font_num = doc.add_object(standard_font_object("Helvetica"))

    var dejavu = load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    # Collect the FULL alphabet + accents. Embed ONLY these.
    var alpha = String("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 Ünïcödé©®—()\\")
    var cps = collect_codepoints(alpha)
    var emb = embed_font(doc, dejavu, "DejaVuSans", cps)
    var type0_font_num = emb.obj_num

    # Fail-loud guard: a codepoint NOT in the embedded set must RAISE rather than
    # silently emit unextractable text. U+4E2D (中) is absent from the set above.
    var guard_fired = False
    var probe = List[UInt8]()
    try:
        show_text_unicode(probe, dejavu, emb, String("中"))
    except:
        guard_fired = True
    if not guard_fired:
        raise Error("fail-loud guard did NOT fire on an un-embedded codepoint")
    print("fail-loud guard fired on un-embedded codepoint: OK")

    # Two images: tiny 1x1 and 64x64.
    var small = List[UInt8]()
    small.append(UInt8(200)); small.append(UInt8(100)); small.append(UInt8(50))
    var img_small = doc.add_object(image_xobject_stream(1, 1, small))
    var big = List[UInt8]()
    for y in range(64):
        for x in range(64):
            big.append(UInt8((x * 255) // 63)); big.append(UInt8((y * 255) // 63)); big.append(UInt8(64))
    var img_big = doc.add_object(image_xobject_stream(64, 64, big))

    var extg_num = doc.add_object(extgstate_object(0.5, 0.7))

    var NPAGES = 32
    var page_nums = List[Int]()
    var para = String(
        "This is a long wrapped paragraph repeated to force multiple lines so that"
        " the greedy word wrap in pdf.layout actually emits several lines of body"
        " text inside a fixed measure of points across the page width here."
    )

    for pidx in range(NPAGES):
        var c = Content()
        # standard-14 with special chars + empty string
        begin_text(c.b); set_font(c.b, "F1", 14.0); text_pos(c.b, 72.0, 740.0)
        show_text(c.b, String("Page ") + String(pidx) + String(" special: (a) \\ b )("))
        end_text(c.b)
        begin_text(c.b); set_font(c.b, "F1", 10.0); text_pos(c.b, 72.0, 728.0)
        show_text(c.b, String(""))  # empty string
        end_text(c.b)
        # embedded-font unicode with special chars (all in the embedded set)
        begin_text(c.b); set_font(c.b, "F2", 12.0); text_pos(c.b, 72.0, 712.0)
        show_text_unicode(c.b, dejavu, emb, String("Ünïcödé (©®) — abc ") )
        end_text(c.b)
        # wrapped paragraph, ACTUALLY emitting every line
        var lines = wrap_text(para, "Helvetica", 10.0, 460.0)
        begin_text(c.b); set_font(c.b, "F1", 10.0)
        var ty = 670.0
        var lx0 = align_x(lines[0], "Helvetica", 10.0, 72.0, 460.0, 0)
        text_pos(c.b, lx0, ty)
        show_text(c.b, lines[0])
        for i in range(1, len(lines)):
            text_pos(c.b, 0.0, -12.0)
            show_text(c.b, lines[i])
        end_text(c.b)
        # color + transparency
        c.save_state(); set_fill_gray(c.b, 0.5); c.rect(72.0, 560.0, 50.0, 50.0); c.fill(); c.restore_state()
        c.save_state(); set_fill_cmyk(c.b, 1.0, 0.0, 0.0, 0.0); c.rect(140.0, 560.0, 50.0, 50.0); c.fill(); c.restore_state()
        c.save_state(); apply_gstate(c.b, "GS1"); c.set_fill_rgb(1.0, 0.0, 0.0); c.rect(72.0, 500.0, 60.0, 40.0); c.fill(); c.restore_state()
        # image (alternate sizes)
        if pidx % 2 == 0:
            draw_image(c.b, "Im1", 300.0, 560.0, 64.0, 64.0)
        # link annot
        var link_num = doc.add_object(link_uri_annotation(72.0, 480.0, 200.0, 495.0, String("https://example.com/p") + String(pidx)))
        var img_for_page = img_big if pidx % 2 == 0 else img_small
        var res = _build_resources(std_font_num, type0_font_num, img_for_page, extg_num)
        var annots = List[Int]()
        annots.append(link_num)
        var pn = doc.add_page_ext(612.0, 792.0, c.bytes(), res, annots, True)
        page_nums.append(pn)

    # Flat outline: one bookmark per page.
    var base = len(doc.objects) + 1
    var root_num = base + NPAGES  # items occupy base..base+NPAGES-1, root last
    var item_nums = List[Int]()
    for i in range(NPAGES):
        item_nums.append(base + i)
    for i in range(NPAGES):
        var prev = item_nums[i - 1] if i > 0 else 0
        var nxt = item_nums[i + 1] if i < NPAGES - 1 else 0
        var it = outline_item_object(String("Bookmark ") + String(i), page_nums[i], 760.0, root_num, prev, nxt)
        var added = doc.add_object(it)
        if added != item_nums[i]:
            raise Error("outline item number mismatch")
    var root = outlines_root_object(item_nums[0], item_nums[NPAGES - 1], NPAGES)
    var added_root = doc.add_object(root)
    if added_root != root_num:
        raise Error("outline root number mismatch")
    doc.set_outlines(root_num)

    doc.save("/tmp/pdf_prostress.pdf")
    print("wrote /tmp/pdf_prostress.pdf pages:", len(doc.page_numbers), "objects:", len(doc.objects))
