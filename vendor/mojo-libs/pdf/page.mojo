# pdf/page.mojo
# Ergonomic high-level page facade over PdfDoc.
#
# A Page collects a content-stream buffer plus the resource bookkeeping
# (font name -> object number, xobject name -> object number) so callers don't
# hand-build "<< /Font << /F1 N 0 R >> ... >>" dicts. Draw helpers from
# pdf.text / pdf.content / pdf.image append directly into `page.buf`.

from pdf.objects import append_str, append_int
from pdf.document import PdfDoc


struct Page(Movable):
    # Public content-stream buffer; draw helpers append into this.
    var buf: List[UInt8]
    var width: Float64
    var height: Float64
    # Registered font objects, in registration order. names are "F1","F2",...
    var font_objs: List[Int]
    # Registered xobjects, in registration order. names are "Im1","Im2",...
    var xobject_objs: List[Int]

    def __init__(out self, width: Float64, height: Float64):
        self.buf = List[UInt8]()
        self.width = width
        self.height = height
        self.font_objs = List[Int]()
        self.xobject_objs = List[Int]()

    def add_font(mut self, obj_num: Int) -> String:
        # Register a font object; return its resource name ("F1","F2",...).
        self.font_objs.append(obj_num)
        return String("F") + String(len(self.font_objs))

    def add_xobject(mut self, obj_num: Int) -> String:
        # Register an XObject; return its resource name ("Im1","Im2",...).
        self.xobject_objs.append(obj_num)
        return String("Im") + String(len(self.xobject_objs))

    def resources_bytes(self) raises -> List[UInt8]:
        # Build "<< /Font << /F1 N 0 R ... >> /XObject << /Im1 M 0 R ... >> >>".
        # Empty sub-dicts are omitted; an all-empty resource set yields "<< >>".
        var r = List[UInt8]()
        append_str(r, "<<")
        if len(self.font_objs) > 0:
            append_str(r, " /Font <<")
            for i in range(len(self.font_objs)):
                append_str(r, " /F")
                append_int(r, i + 1)
                append_str(r, " ")
                append_int(r, self.font_objs[i])
                append_str(r, " 0 R")
            append_str(r, " >>")
        if len(self.xobject_objs) > 0:
            append_str(r, " /XObject <<")
            for i in range(len(self.xobject_objs)):
                append_str(r, " /Im")
                append_int(r, i + 1)
                append_str(r, " ")
                append_int(r, self.xobject_objs[i])
                append_str(r, " 0 R")
            append_str(r, " >>")
        append_str(r, " >>")
        return r.copy()

    def finish(self, mut doc: PdfDoc, flate: Bool) raises -> Int:
        # Emit this page into the document. Returns the page object number.
        var res = self.resources_bytes()
        var content = self.buf.copy()
        if flate:
            return doc.add_page_flate(self.width, self.height, content, res)
        return doc.add_page(self.width, self.height, content, res)


def new_page(width: Float64, height: Float64) -> Page:
    # Factory: a blank page of the given size (in PDF points).
    return Page(width, height)
