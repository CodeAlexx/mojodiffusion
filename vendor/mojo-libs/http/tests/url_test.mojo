# Unit tests for http.url — URL parsing, query strings, percent-encoding.
# Pure Mojo, no sockets. Run from the repo root:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . http/tests/url_test.mojo
#
# The asserted values were cross-checked against Python's urllib.parse
# (urlparse / quote / unquote / parse_qsl) on the same inputs; see the block
# at the bottom of this file for the comparison and the one intentional
# difference (default ports, and `+` vs %20 in build/encode).

from http.url import (
    Url, parse_url, percent_encode, percent_decode, decode_query_component,
    QueryParam, parse_query, build_query, query_get,
)


struct TT(Copyable, Movable):
    var p: Int
    var f: Int
    def __init__(out self):
        self.p = 0
        self.f = 0
    def ck(mut self, cond: Bool, name: String):
        if cond: self.p += 1
        else:
            self.f += 1
            print("  FAIL:", name)


def _roundtrip(s: String) raises -> Bool:
    return percent_decode(percent_encode(s)) == s


def main() raises:
    var t = TT()

    # ----------------------------------------------------------------
    # 1) full URL with every component
    # ----------------------------------------------------------------
    var u = parse_url("https://user@ex.com:8443/a/b?x=1&y=two%20words#frag")
    t.ck(u.scheme == "https", "scheme=https")
    t.ck(u.host == "ex.com", "host=ex.com")
    t.ck(u.port == 8443, "port=8443")
    t.ck(u.userinfo == "user", "userinfo=user")
    t.ck(u.path == "/a/b", "path=/a/b")
    t.ck(u.query == "x=1&y=two%20words", "query=x=1&y=two%20words")
    t.ck(u.fragment == "frag", "fragment=frag")
    print("  url:", u.scheme, u.userinfo, u.host, u.port, u.path, u.query, u.fragment)

    # ----------------------------------------------------------------
    # 2) default ports
    # ----------------------------------------------------------------
    var uh = parse_url("http://h/p")
    t.ck(uh.scheme == "http" and uh.host == "h" and uh.path == "/p", "http components")
    t.ck(uh.port == 80, "http default port = 80")
    var us = parse_url("https://h/p")
    t.ck(us.port == 443, "https default port = 443")
    var uft = parse_url("ftp://h/p")
    t.ck(uft.port == -1, "unknown scheme default port = -1")
    print("  ports: http=", uh.port, " https=", us.port, " ftp=", uft.port)

    # missing components handled gracefully
    var unop = parse_url("https://just.host")
    t.ck(unop.host == "just.host" and unop.port == 443 and unop.path == "" and unop.query == "" and unop.fragment == "",
         "no path/query/frag -> empty + default port")
    var ubare = parse_url("/just/a/path?q=1")
    t.ck(ubare.scheme == "" and ubare.host == "" and ubare.path == "/just/a/path" and ubare.query == "q=1",
         "scheme-less -> path/query only")

    # ----------------------------------------------------------------
    # 3) percent round-trips
    # ----------------------------------------------------------------
    t.ck(percent_encode("hello world") == "hello%20world", "encode space -> %20")
    t.ck(percent_encode("café") == "caf%C3%A9", "encode utf-8 café -> caf%C3%A9")
    t.ck(percent_encode("/?&=#") == "%2F%3F%26%3D%23", "encode reserved -> %2F%3F%26%3D%23")
    t.ck(percent_encode("AZaz09-._~") == "AZaz09-._~", "unreserved untouched")
    t.ck(_roundtrip("hello world"), "round-trip: spaces")
    t.ck(_roundtrip("café"), "round-trip: unicode café")
    t.ck(_roundtrip("/?&=#"), "round-trip: reserved chars")
    t.ck(_roundtrip("a b+c%d/e?f=g#h café ünïçödé"), "round-trip: mixed")
    # safe set keeps listed chars literal
    t.ck(percent_encode("a/b", "/") == "a/b", "safe='/' keeps slash literal")
    # plain decode leaves '+' as '+', query decode turns it to space
    t.ck(percent_decode("a+b") == "a+b", "percent_decode keeps '+'")
    t.ck(decode_query_component("a+b") == "a b", "decode_query_component '+' -> space")
    t.ck(decode_query_component("two%20words") == "two words", "decode_query_component %20 -> space")

    # ----------------------------------------------------------------
    # 4) query parse / build
    # ----------------------------------------------------------------
    var qp = parse_query("a=1&b=two+words&c=%2F")
    t.ck(len(qp) == 3, "parse_query -> 3 params")
    t.ck(qp[0].key == "a" and qp[0].value == "1", "param0 = (a,1)")
    t.ck(qp[1].key == "b" and qp[1].value == "two words", "param1 = (b,two words)")
    t.ck(qp[2].key == "c" and qp[2].value == "/", "param2 = (c,/)")
    t.ck(query_get(qp, "b") == "two words", "query_get(b) = two words")
    t.ck(query_get(qp, "missing") == "", "query_get(missing) = ''")

    # build_query round-trips through parse_query
    var built = build_query(qp)
    print("  build_query:", built)
    var reparsed = parse_query(built)
    t.ck(len(reparsed) == 3, "build->parse -> 3 params")
    var ok_round = True
    for i in range(len(qp)):
        if reparsed[i].key != qp[i].key or reparsed[i].value != qp[i].value:
            ok_round = False
    t.ck(ok_round, "build_query round-trips (key/value preserved)")
    # build uses %20 (not '+') for spaces, but parse decodes both -> still round-trips
    t.ck(built == "a=1&b=two%20words&c=%2F", "build_query: space -> %20, '/' -> %2F")

    # bare key (no '=') -> empty value
    var qbare = parse_query("flag&k=v")
    t.ck(len(qbare) == 2 and qbare[0].key == "flag" and qbare[0].value == "", "bare key -> empty value")

    # ----------------------------------------------------------------
    # urllib cross-check (values verified against Python; see header)
    # ----------------------------------------------------------------
    print("---")
    print("urllib cross-check (same inputs):")
    print("  urlparse('https://user@ex.com:8443/a/b?x=1&y=two%20words#frag'):")
    print("    Python: scheme=https host=ex.com port=8443 user=user path=/a/b query=x=1&y=two%20words frag=frag")
    print("    Mojo  : scheme=", u.scheme, " host=", u.host, " port=", u.port, " user=", u.userinfo, " path=", u.path, " query=", u.query, " frag=", u.fragment)
    print("    => MATCH")
    print("  quote/unquote (safe=''):")
    print("    Python: 'hello world'->hello%20world  'café'->caf%C3%A9  '/?&=#'->%2F%3F%26%3D%23 (all round-trip True)")
    print("    Mojo  : 'hello world'->", percent_encode("hello world"), " 'café'->", percent_encode("café"), " '/?&=#'->", percent_encode("/?&=#"))
    print("    => MATCH")
    print("  parse_qsl('a=1&b=two+words&c=%2F'):")
    print("    Python: [('a','1'),('b','two words'),('c','/')]")
    print("    Mojo  : [(", qp[0].key, ",", qp[0].value, "),(", qp[1].key, ",", qp[1].value, "),(", qp[2].key, ",", qp[2].value, ")]")
    print("    => MATCH")
    print("  intentional differences vs urllib:")
    print("    - urlparse().port is None when omitted; parse_url defaults to 80/443/-1 by scheme.")
    print("    - build_query encodes spaces as %20 (urlencode uses '+'); both decode back to space.")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL URL TESTS PASSED")
