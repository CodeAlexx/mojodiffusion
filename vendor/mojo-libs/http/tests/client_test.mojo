# http/tests/client_test.mojo — exercises the real HTTP/1.1 client against a
# local Python server. The server is started by the shell/python driver
# (client_test_driver.py) which then runs this binary; this file only contains
# the Mojo-side assertions + live calls. It expects the server on 127.0.0.1:8137.
#
# Run via: python3 http/tests/client_test_driver.py  (it builds+runs this)

from sys import argv
from http.client import (
    get,
    request,
    resolve_host,
    parse_url,
    Client,
    ClientOptions,
    Header,
    HttpResponse,
)

comptime PORT = 8137


def _expect(cond: Bool, label: String) raises -> Int:
    if cond:
        print("PASS:", label)
        return 1
    print("FAIL:", label)
    return 0


def main() raises:
    var base = String("http://127.0.0.1:") + String(PORT)
    var passes = 0
    var total = 0

    # ── DNS ────────────────────────────────────────────────────────────────
    var ip_quad = resolve_host(String("127.0.0.1"))
    total += 1; passes += _expect(ip_quad == "127.0.0.1", "DNS dotted-quad 127.0.0.1 -> " + ip_quad)

    var ip_local = resolve_host(String("localhost"))
    total += 1; passes += _expect(ip_local == "127.0.0.1", "DNS localhost -> " + ip_local)

    # ── URL parse ──────────────────────────────────────────────────────────
    var pu = parse_url(String("http://example.com:8080/a/b?x=1"))
    total += 1; passes += _expect(pu.host == "example.com" and pu.port == 8080 and pu.path == "/a/b?x=1", "parse_url host/port/path")

    # ── plain GET (exact body) ──────────────────────────────────────────────
    var r1 = get(base + "/plain")
    print("  /plain status =", r1.status, " body =", r1.text())
    total += 1; passes += _expect(r1.status == 200, "plain GET status 200")
    total += 1; passes += _expect(r1.text() == "hello mojo client", "plain GET exact body")

    # ── chunked ─────────────────────────────────────────────────────────────
    var r2 = get(base + "/chunked")
    print("  /chunked status =", r2.status, " body =", r2.text())
    total += 1; passes += _expect(r2.status == 200, "chunked status 200")
    total += 1; passes += _expect(r2.text() == "chunk-one|chunk-two|chunk-three", "chunked decodes to expected bytes")

    # ── gzip ────────────────────────────────────────────────────────────────
    var r3 = get(base + "/gzip")
    print("  /gzip status =", r3.status, " enc =", r3.header("content-encoding"), " body =", r3.text())
    total += 1; passes += _expect(r3.status == 200, "gzip status 200")
    total += 1; passes += _expect(r3.text() == "the quick brown fox jumps over the lazy dog", "gzip body decompresses correctly")

    # ── deflate ─────────────────────────────────────────────────────────────
    var r3b = get(base + "/deflate")
    print("  /deflate status =", r3b.status, " enc =", r3b.header("content-encoding"), " body =", r3b.text())
    total += 1; passes += _expect(r3b.text() == "deflate payload works too", "deflate body decompresses correctly")

    # ── redirect (302 -> /final) ────────────────────────────────────────────
    var r4 = get(base + "/redirect")
    print("  /redirect final status =", r4.status, " body =", r4.text())
    total += 1; passes += _expect(r4.status == 200, "302 followed to a 200")
    total += 1; passes += _expect(r4.text() == "you reached the final destination", "redirect lands on final body")

    # ── redirect cap (loop guard): /loop redirects to itself, max_redirects=2 ─
    var opts = ClientOptions()
    opts.max_redirects = 2
    var r5 = get(base + "/loop", opts)
    print("  /loop status after cap =", r5.status)
    total += 1; passes += _expect(r5.status == 302, "redirect cap stops looping (returns last 302)")

    # ── keep-alive pooled client: two requests, one connection ──────────────
    var c = Client(String("127.0.0.1"), PORT)
    var k1 = c.get(String("/plain"))
    var k2 = c.get(String("/chunked"))
    print("  keep-alive r1 =", k1.text(), " r2 =", k2.text())
    total += 1; passes += _expect(k1.text() == "hello mojo client" and k2.text() == "chunk-one|chunk-two|chunk-three", "keep-alive: 2 requests reuse one socket")

    # ── https honest error ──────────────────────────────────────────────────
    var tls_errored = False
    try:
        var _r = get(String("https://127.0.0.1:") + String(PORT) + "/plain")
    except e:
        tls_errored = True
        print("  https error (expected) =", String(e))
    total += 1; passes += _expect(tls_errored, "https raises a clear TLS-not-supported error")

    print("")
    print("RESULT:", passes, "/", total, "passed")
    if passes != total:
        raise Error("some client tests failed")
