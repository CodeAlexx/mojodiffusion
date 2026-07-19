#!/usr/bin/env python3
# client_test_driver.py — start a small HTTP server exercising every framing the
# Mojo client must handle (plain Content-Length, chunked, gzip, deflate, 302
# redirect, self-redirect loop), then run the Mojo client_test against it.
#
# Usage (from /home/alex/MOJO-libs):
#   python3 http/tests/client_test_driver.py
import gzip
import socket
import subprocess
import sys
import threading
import time
import zlib

HOST = "127.0.0.1"
PORT = 8137


def build_response(start_line, headers, body_bytes):
    out = start_line.encode() + b"\r\n"
    for k, v in headers:
        out += f"{k}: {v}".encode() + b"\r\n"
    out += b"\r\n"
    out += body_bytes
    return out


def serve_conn(conn):
    # Serve repeated requests on one connection (keep-alive), one at a time.
    conn.settimeout(5)
    try:
        while True:
            data = b""
            try:
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    data += chunk
            except socket.timeout:
                return
            req_line = data.split(b"\r\n", 1)[0].decode(errors="replace")
            parts = req_line.split(" ")
            path = parts[1] if len(parts) >= 2 else "/"
            respond(conn, path)
    except Exception as e:
        sys.stderr.write(f"[server] conn error: {e}\n")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def respond(conn, path):
    if path == "/plain":
        body = b"hello mojo client"
        conn.sendall(build_response("HTTP/1.1 200 OK",
            [("Content-Type", "text/plain"), ("Content-Length", str(len(body))),
             ("Connection", "keep-alive")], body))
    elif path == "/chunked":
        pieces = [b"chunk-one|", b"chunk-two|", b"chunk-three"]
        out = build_response("HTTP/1.1 200 OK",
            [("Content-Type", "text/plain"), ("Transfer-Encoding", "chunked"),
             ("Connection", "keep-alive")], b"")
        for p in pieces:
            out += f"{len(p):x}".encode() + b"\r\n" + p + b"\r\n"
        out += b"0\r\n\r\n"
        conn.sendall(out)
    elif path == "/gzip":
        raw = b"the quick brown fox jumps over the lazy dog"
        body = gzip.compress(raw)
        conn.sendall(build_response("HTTP/1.1 200 OK",
            [("Content-Type", "text/plain"), ("Content-Encoding", "gzip"),
             ("Content-Length", str(len(body))), ("Connection", "keep-alive")], body))
    elif path == "/deflate":
        raw = b"deflate payload works too"
        body = zlib.compress(raw)
        conn.sendall(build_response("HTTP/1.1 200 OK",
            [("Content-Type", "text/plain"), ("Content-Encoding", "deflate"),
             ("Content-Length", str(len(body))), ("Connection", "keep-alive")], body))
    elif path == "/redirect":
        conn.sendall(build_response("HTTP/1.1 302 Found",
            [("Location", "/final"), ("Content-Length", "0"),
             ("Connection", "keep-alive")], b""))
    elif path == "/final":
        body = b"you reached the final destination"
        conn.sendall(build_response("HTTP/1.1 200 OK",
            [("Content-Type", "text/plain"), ("Content-Length", str(len(body))),
             ("Connection", "keep-alive")], body))
    elif path == "/loop":
        conn.sendall(build_response("HTTP/1.1 302 Found",
            [("Location", "/loop"), ("Content-Length", "0"),
             ("Connection", "keep-alive")], b""))
    else:
        body = b"not found"
        conn.sendall(build_response("HTTP/1.1 404 Not Found",
            [("Content-Length", str(len(body))), ("Connection", "keep-alive")], body))


def server_loop(sock, stop):
    while not stop.is_set():
        try:
            conn, _ = sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        threading.Thread(target=serve_conn, args=(conn,), daemon=True).start()


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(64)
    srv.settimeout(0.5)
    stop = threading.Event()
    t = threading.Thread(target=server_loop, args=(srv, stop), daemon=True)
    t.start()
    print(f"[driver] test server listening on {HOST}:{PORT}")
    time.sleep(0.3)

    # http.client now imports net.tls (OpenSSL FFI), so the program must be
    # BUILT with -lssl -lcrypto (the JIT `mojo run` can't resolve SSL_* symbols).
    binp = "/tmp/client_test_bin"
    build = [
        "pixi", "run", "--manifest-path", "/home/alex/rill/pixi.toml",
        "mojo", "build", "-I", ".", "http/tests/client_test.mojo",
        "-o", binp, "-Xlinker", "-lssl", "-Xlinker", "-lcrypto",
    ]
    print("[driver] building:", " ".join(build))
    brc = subprocess.call(build, cwd="/home/alex/MOJO-libs")
    if brc != 0:
        stop.set()
        try: srv.close()
        except Exception: pass
        print(f"[driver] BUILD FAILED rc={brc}")
        raise SystemExit(1)
    print("[driver] running:", binp)
    rc = subprocess.call([binp], cwd="/home/alex/MOJO-libs")
    stop.set()
    try:
        srv.close()
    except Exception:
        pass
    print(f"[driver] mojo client_test exit code = {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
