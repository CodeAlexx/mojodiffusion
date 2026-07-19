# Unit tests for http.headers — content negotiation, conditional requests,
# range parsing and HTTP-date round-trips. Pure Mojo, no network.
# Run from the repo root:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . http/tests/headers_test.mojo
#
# The HTTP-date expectations were cross-checked against Python's
# email.utils.formatdate/parsedate_to_datetime (values pasted in the report).

from http.headers import (
    MediaRange, parse_accept, select_media_type, select_encoding,
    etag_strong, etag_weak, if_none_match_satisfied, if_modified_since_satisfied,
    ByteRange, parse_range, content_range_header,
    format_http_date, parse_http_date,
)


struct TT(Copyable, Movable):
    var p: Int
    var f: Int
    def __init__(out self):
        self.p = 0
        self.f = 0
    def ck(mut self, cond: Bool, name: String):
        if cond:
            self.p += 1
        else:
            self.f += 1
            print("  FAIL:", name)


def main() raises:
    var t = TT()

    # ----------------------------------------------------------------
    # 1) parse_accept ordering by q
    # ----------------------------------------------------------------
    var acc = String("text/html,application/json;q=0.9,*/*;q=0.1")
    var mrs = parse_accept(acc)
    print("=== parse_accept('" + acc + "') ===")
    for i in range(len(mrs)):
        print("  ", mrs[i].type + "/" + mrs[i].subtype, "q=", mrs[i].q)
    t.ck(len(mrs) == 3, "parse_accept len == 3")
    t.ck(mrs[0].type == "text" and mrs[0].subtype == "html" and mrs[0].q == 1.0,
         "parse_accept[0] = text/html q=1.0 (highest)")
    t.ck(mrs[1].type == "application" and mrs[1].subtype == "json" and mrs[1].q == 0.9,
         "parse_accept[1] = application/json q=0.9")
    t.ck(mrs[2].type == "*" and mrs[2].subtype == "*" and mrs[2].q == 0.1,
         "parse_accept[2] = */* q=0.1 (lowest)")

    # ----------------------------------------------------------------
    # 2) select_media_type
    # ----------------------------------------------------------------
    var offered: List[String] = ["application/json", "text/plain"]
    var chosen = select_media_type(acc, offered)
    print("=== select_media_type ===")
    print("  offered [application/json, text/plain] from above accept ->", chosen)
    t.ck(chosen == "application/json", "select picks application/json")

    # text/html is offered and is the q=1.0 exact match
    var off_html: List[String] = ["text/html", "text/plain"]
    var chosen2 = select_media_type(acc, off_html)
    print("  offered [text/html, text/plain] ->", chosen2)
    t.ck(chosen2 == "text/html", "select picks text/html (q=1.0 exact)")

    # */* fallback: an accept that only has */* takes whatever is offered first
    var off_png: List[String] = ["image/png", "text/plain"]
    var chosen3 = select_media_type("*/*", off_png)
    print("  accept '*/*' offered [image/png, text/plain] ->", chosen3)
    t.ck(chosen3 == "image/png", "*/* fallback picks first offered")

    # type wildcard: image/* matches image/png but not text/plain
    var off_png2: List[String] = ["text/plain", "image/png"]
    var chosen4 = select_media_type("image/*", off_png2)
    print("  accept 'image/*' offered [text/plain, image/png] ->", chosen4)
    t.ck(chosen4 == "image/png", "image/* matches image/png only")

    # unacceptable -> ""
    var off_txt: List[String] = ["text/plain", "text/html"]
    var chosen5 = select_media_type("application/json", off_txt)
    print("  accept 'application/json' offered [text/plain, text/html] -> '" + chosen5 + "'")
    t.ck(chosen5 == "", "unacceptable -> empty string")

    # q=0 means not acceptable
    var off_txt2: List[String] = ["text/plain", "text/html"]
    var chosen6 = select_media_type("text/plain;q=0,*/*;q=0.5", off_txt2)
    print("  accept 'text/plain;q=0,*/*;q=0.5' offered [text/plain, text/html] ->", chosen6)
    t.ck(chosen6 == "text/html", "q=0 on exact -> falls to wildcard match")

    # empty accept -> first offered
    var off_txt3: List[String] = ["text/plain", "text/html"]
    var chosen7 = select_media_type("", off_txt3)
    t.ck(chosen7 == "text/plain", "empty accept -> first offered")

    # ----------------------------------------------------------------
    # 2b) select_encoding
    # ----------------------------------------------------------------
    var off_enc: List[String] = ["br", "gzip"]
    var enc = select_encoding("gzip, br;q=0.9, identity;q=0.1", off_enc)
    print("=== select_encoding ===")
    print("  accept-encoding 'gzip, br;q=0.9, identity;q=0.1' offered [br, gzip] ->", enc)
    t.ck(enc == "gzip", "select_encoding picks gzip (q=1.0)")
    var off_enc2: List[String] = ["br", "gzip"]
    var enc2 = select_encoding("*", off_enc2)
    t.ck(enc2 == "br", "select_encoding '*' -> first offered")
    var off_enc3: List[String] = ["br", "deflate"]
    var enc3 = select_encoding("gzip", off_enc3)
    t.ck(enc3 == "", "select_encoding none acceptable -> ''")

    # ----------------------------------------------------------------
    # 3) ETag
    # ----------------------------------------------------------------
    var body_a: List[UInt8] = []
    for c in String("Hello, World").as_bytes():
        body_a.append(c)
    var body_b: List[UInt8] = []
    for c in String("Hello, world").as_bytes():  # differs by one byte
        body_b.append(c)
    var et_a = etag_strong(body_a)
    var et_a2 = etag_strong(body_a.copy())
    var et_b = etag_strong(body_b)
    var et_aw = etag_weak(body_a)
    print("=== ETag ===")
    print("  etag_strong('Hello, World') =", et_a)
    print("  etag_strong('Hello, world') =", et_b)
    print("  etag_weak  ('Hello, World') =", et_aw)
    t.ck(et_a == et_a2, "same body -> same etag")
    t.ck(et_a != et_b, "different body -> different etag")
    t.ck(if_none_match_satisfied("*", et_a), "if_none_match '*' -> True")
    t.ck(if_none_match_satisfied(et_a, et_a), "if_none_match exact -> True")
    var inm_list = String('"deadbeefdeadbeef", ') + et_a
    print("  if_none_match comma-list '" + inm_list + "' vs", et_a)
    t.ck(if_none_match_satisfied(inm_list, et_a), "if_none_match comma-list match -> True")
    t.ck(not if_none_match_satisfied('"nope"', et_a), "if_none_match no match -> False")
    # weak compare: W/ prefix ignored on both sides
    t.ck(if_none_match_satisfied(et_aw, et_a), "if_none_match weak vs strong (same opaque) -> True")
    t.ck(not if_none_match_satisfied("", et_a), "empty if_none_match -> False")

    # if_modified_since
    print("=== if_modified_since ===")
    # resource last modified at epoch 784111777; IMS = same date -> not modified -> True
    t.ck(if_modified_since_satisfied("Sun, 06 Nov 1994 08:49:37 GMT", 784111777),
         "ims == last_modified -> not modified (True)")
    # resource modified AFTER ims -> modified -> False
    t.ck(not if_modified_since_satisfied("Sun, 06 Nov 1994 08:49:37 GMT", 784111778),
         "last_modified after ims -> modified (False)")
    t.ck(not if_modified_since_satisfied("garbage", 100), "malformed ims -> False")

    # ----------------------------------------------------------------
    # 4) Range
    # ----------------------------------------------------------------
    print("=== Range (total=1234) ===")
    var r1 = parse_range("bytes=0-499", 1234)
    print("  bytes=0-499 ->", "[" + String(r1[0].start) + "," + String(r1[0].end) + "]")
    t.ck(len(r1) == 1 and r1[0].start == 0 and r1[0].end == 499, "bytes=0-499 -> [0,499]")

    var r2 = parse_range("bytes=500-", 1234)
    print("  bytes=500- ->", "[" + String(r2[0].start) + "," + String(r2[0].end) + "]")
    t.ck(len(r2) == 1 and r2[0].start == 500 and r2[0].end == 1233, "bytes=500- -> [500,1233]")

    var r3 = parse_range("bytes=-500", 1234)
    print("  bytes=-500 ->", "[" + String(r3[0].start) + "," + String(r3[0].end) + "]")
    t.ck(len(r3) == 1 and r3[0].start == 734 and r3[0].end == 1233, "bytes=-500 -> [734,1233]")

    var r4 = parse_range("bytes=2000-3000", 1234)
    print("  bytes=2000-3000 (unsatisfiable) -> len", len(r4))
    t.ck(len(r4) == 0, "bytes=2000-3000 unsatisfiable -> empty")

    # multi-range
    var r5 = parse_range("bytes=0-9,20-29", 1234)
    t.ck(len(r5) == 2 and r5[0].start == 0 and r5[0].end == 9
         and r5[1].start == 20 and r5[1].end == 29, "multi-range parsed")

    # open end clamps to total-1
    var r6 = parse_range("bytes=1000-99999", 1234)
    t.ck(len(r6) == 1 and r6[0].start == 1000 and r6[0].end == 1233, "end clamps to total-1")

    # bad unit -> empty
    t.ck(len(parse_range("items=0-1", 1234)) == 0, "non-bytes unit -> empty")

    var cr = content_range_header(0, 499, 1234)
    print("  content_range_header(0,499,1234) ->", cr)
    t.ck(cr == "bytes 0-499/1234", "content_range_header correct")

    # ----------------------------------------------------------------
    # 5) HTTP-date round-trips + RFC example
    # ----------------------------------------------------------------
    print("=== HTTP-date ===")
    var rfc = parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT")
    print("  parse 'Sun, 06 Nov 1994 08:49:37 GMT' ->", rfc, "(expect 784111777)")
    t.ck(rfc == 784111777, "RFC example parses to 784111777")

    # round-trip several epochs
    var epochs = [0, 784111777, 1000000000, 1700000000, 1800000000]
    var ok_round = True
    for i in range(len(epochs)):
        var e = epochs[i]
        var formatted = format_http_date(e)
        var back = parse_http_date(formatted)
        print("  epoch", e, "->", formatted, "-> parse", back)
        if back != e:
            ok_round = False
    t.ck(ok_round, "format/parse round-trip for all epochs")

    # exact format strings must equal Python's email.utils.formatdate (pasted in report)
    t.ck(format_http_date(0) == "Thu, 01 Jan 1970 00:00:00 GMT", "format(0) matches Python")
    t.ck(format_http_date(784111777) == "Sun, 06 Nov 1994 08:49:37 GMT", "format(784111777) matches Python")
    t.ck(format_http_date(1000000000) == "Sun, 09 Sep 2001 01:46:40 GMT", "format(1e9) matches Python")
    t.ck(format_http_date(1700000000) == "Tue, 14 Nov 2023 22:13:20 GMT", "format(1.7e9) matches Python")
    t.ck(format_http_date(1800000000) == "Fri, 15 Jan 2027 08:00:00 GMT", "format(1.8e9) matches Python")

    print("---")
    print("passed:", t.p, "failed:", t.f)
    if t.f == 0:
        print("ALL HEADERS TESTS PASSED")
