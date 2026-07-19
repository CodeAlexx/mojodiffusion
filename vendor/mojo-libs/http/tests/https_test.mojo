# http/tests/https_test.mojo — client-side TLS (https) against a local TLS server.
# Driven by https_test_driver.py (generates a self-signed cert, starts a TLS
# server on 127.0.0.1:8443, builds this with -lssl -lcrypto, runs it).
#
# The server cert is self-signed, so:
#   * verify_tls = False  -> handshake + encrypted I/O succeed (data-path test)
#   * verify_tls = True   -> MUST fail (untrusted self-signed) -> proves verify works

from http.client import get, ClientOptions


def main() raises:
    var passed = 0
    var failed = 0
    var base = String("https://127.0.0.1:8443")

    # 1) plain https GET (verify off — self-signed)
    var opt = ClientOptions()
    opt.verify_tls = False
    var r1 = get(base + "/plain", opt)
    print("[/plain] status =", r1.status, " body =", r1.text())
    if r1.status == 200:
        passed += 1
    else:
        failed += 1
        print("  FAIL: /plain status")
    if r1.text() == "secure hello over TLS":
        passed += 1
    else:
        failed += 1
        print("  FAIL: /plain body")

    # 2) chunked over https
    var r2 = get(base + "/chunked", opt)
    print("[/chunked] status =", r2.status, " body =", r2.text())
    if r2.status == 200 and r2.text() == "tls-chunk-1|tls-chunk-2|tls-chunk-3":
        passed += 1
    else:
        failed += 1
        print("  FAIL: /chunked decode over TLS")

    # 3) gzip over https (exercises the encrypted-read + inflate path together)
    var r3 = get(base + "/gzip", opt)
    print("[/gzip] status =", r3.status, " enc =", r3.header("content-encoding"), " body =", r3.text())
    if r3.status == 200 and r3.text() == "gzip body delivered over TLS":
        passed += 1
    else:
        failed += 1
        print("  FAIL: /gzip over TLS")

    # 4) verification ON against the self-signed cert MUST raise.
    var opt_v = ClientOptions()
    opt_v.verify_tls = True
    var raised = False
    var msg = String("")
    try:
        var rv = get(base + "/plain", opt_v)
        print("  (unexpected) verify=True returned status", rv.status)
    except e:
        raised = True
        msg = String(e)
    if raised:
        passed += 1
        print("[verify=True on self-signed] correctly raised:", msg)
    else:
        failed += 1
        print("  FAIL: verify=True did NOT reject the self-signed cert")

    print("passed:", passed, " failed:", failed)
    if failed == 0:
        print("ALL HTTPS TESTS PASSED")
