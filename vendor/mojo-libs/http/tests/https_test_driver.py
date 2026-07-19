#!/usr/bin/env python3
# Driver for http/tests/https_test.mojo:
#   1. generate a self-signed cert for 127.0.0.1
#   2. start a TLS server on 127.0.0.1:8443 (/plain, /chunked, /gzip)
#   3. build the Mojo https test with -lssl -lcrypto and run it
#   4. tear everything down
import os, ssl, subprocess, sys, threading, time, gzip, tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PORT = 8443
TMP = tempfile.mkdtemp(prefix="mojo_https_")
CERT = os.path.join(TMP, "cert.pem")
KEY = os.path.join(TMP, "key.pem")


def make_cert():
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", KEY, "-out", CERT, "-days", "1",
         "-subj", "/CN=127.0.0.1",
         "-addext", "subjectAltName=IP:127.0.0.1"],
        check=True, capture_output=True)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/plain":
            body = b"secure hello over TLS"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/chunked":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in [b"tls-chunk-1", b"|tls-chunk-2", b"|tls-chunk-3"]:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
            self.wfile.write(b"0\r\n\r\n")
        elif self.path == "/gzip":
            body = gzip.compress(b"gzip body delivered over TLS")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


def serve(httpd):
    try:
        httpd.serve_forever()
    except Exception:
        pass


def main():
    make_cert()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT, KEY)
    httpd = HTTPServer(("127.0.0.1", PORT), H)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    t = threading.Thread(target=serve, args=(httpd,), daemon=True)
    t.start()
    time.sleep(0.5)

    binp = os.path.join(TMP, "https_test")
    build = subprocess.run(
        ["pixi", "run", "--manifest-path", "/home/alex/rill/pixi.toml",
         "mojo", "build", "-I", ".", "http/tests/https_test.mojo",
         "-o", binp, "-Xlinker", "-lssl", "-Xlinker", "-lcrypto"],
        cwd=REPO, capture_output=True, text=True)
    if not os.path.exists(binp):
        print("BUILD FAILED:")
        print(build.stdout[-2000:])
        print(build.stderr[-2000:])
        httpd.shutdown()
        sys.exit(1)

    run = subprocess.run([binp], cwd=REPO, capture_output=True, text=True, timeout=60)
    print(run.stdout)
    if run.stderr.strip():
        print("--- stderr ---")
        print(run.stderr[-1500:])
    httpd.shutdown()
    sys.exit(0 if "ALL HTTPS TESTS PASSED" in run.stdout else 1)


if __name__ == "__main__":
    main()
