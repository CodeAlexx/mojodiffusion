# net.tls — server-side TLS via OpenSSL FFI. Requires linking -lssl -lcrypto
# (build with: -Xlinker -lssl -Xlinker -lcrypto). Verified against OpenSSL 3.x.
#
# OpenSSL handles (SSL_CTX*, SSL*) are opaque; we pass them around as integer
# addresses (Dicts can hold Int but not arbitrary pointers). conn_read/write map
# OpenSSL's WANT_READ/WANT_WRITE onto a small status protocol the event loop can
# act on.

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]
comptime SSL_FILETYPE_PEM: Int32 = 1
comptime SSL_ERROR_WANT_READ: Int32 = 2
comptime SSL_ERROR_WANT_WRITE: Int32 = 3
comptime SSL_ERROR_ZERO_RETURN: Int32 = 6


def _nullp() -> BytePtr:
    return BytePtr(unsafe_from_address=Int(0))


def _cbuf(s: String) -> BytePtr:
    """NUL-terminated copy of `s` for libc/OpenSSL string args."""
    var n = s.byte_length()
    var b = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        b[i] = src[i]
    b[n] = 0
    return BytePtr(unsafe_from_address=Int(b))


def tls_err() -> String:
    """Top OpenSSL error string (for diagnostics)."""
    var e = external_call["ERR_get_error", UInt64]()
    var p = external_call["ERR_error_string", BytePtr](e, _nullp())
    var n = 0
    while p[n] != 0:
        n += 1
    return String(StringSlice(ptr=p, length=n))


def make_ctx(cert_path: String, key_path: String) raises -> Int:
    """Initialise OpenSSL, build a TLS server context, load the certificate and
    private key. Returns the SSL_CTX* as an integer address."""
    _ = external_call["OPENSSL_init_ssl", Int32](UInt64(0), _nullp())
    var method = external_call["TLS_server_method", BytePtr]()
    var ctx = external_call["SSL_CTX_new", BytePtr](method)
    var cp = _cbuf(cert_path)
    var kp = _cbuf(key_path)
    var c1 = external_call["SSL_CTX_use_certificate_file", Int32](
        ctx, cp, SSL_FILETYPE_PEM
    )
    var c2 = external_call["SSL_CTX_use_PrivateKey_file", Int32](
        ctx, kp, SSL_FILETYPE_PEM
    )
    cp.free()
    kp.free()
    if c1 != 1 or c2 != 1:
        raise Error("TLS cert/key load failed: " + tls_err())
    return Int(ctx)


def conn_new(ctx_addr: Int, fd: Int32) -> Int:
    """Wrap an accepted fd in a new SSL*. Returns the SSL* address."""
    var ctx = BytePtr(unsafe_from_address=ctx_addr)
    var ssl = external_call["SSL_new", BytePtr](ctx)
    _ = external_call["SSL_set_fd", Int32](ssl, fd)
    return Int(ssl)


def accept_blocking(ssl_addr: Int) -> Bool:
    """Run the TLS handshake to completion (blocking). True on success."""
    var ssl = BytePtr(unsafe_from_address=ssl_addr)
    return external_call["SSL_accept", Int32](ssl) == 1


def conn_read(ssl_addr: Int, buf: BytePtr, cap: Int) -> Int:
    """Decrypted read. Returns: >0 bytes, 0 = peer closed (close_notify),
    -1 = would block (retry on next readable), -2 = fatal error."""
    var ssl = BytePtr(unsafe_from_address=ssl_addr)
    var r = external_call["SSL_read", Int32](ssl, buf, Int32(cap))
    if r > 0:
        return Int(r)
    var e = external_call["SSL_get_error", Int32](ssl, r)
    if e == SSL_ERROR_WANT_READ or e == SSL_ERROR_WANT_WRITE:
        return -1
    if e == SSL_ERROR_ZERO_RETURN:
        return 0
    return -2


def conn_write_all(ssl_addr: Int, data: String):
    """Encrypt and send the whole string. Spins on WANT_WRITE (responses here are
    small); stops on a fatal error."""
    var ssl = BytePtr(unsafe_from_address=ssl_addr)
    var n = data.byte_length()
    if n == 0:
        return
    var buf = alloc[UInt8](n)
    var src = data.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var p = BytePtr(unsafe_from_address=Int(buf))
    var total = 0
    while total < n:
        var w = external_call["SSL_write", Int32](ssl, p + total, Int32(n - total))
        if w > 0:
            total += Int(w)
        else:
            var e = external_call["SSL_get_error", Int32](ssl, w)
            if e == SSL_ERROR_WANT_READ or e == SSL_ERROR_WANT_WRITE:
                continue
            break
    buf.free()


def conn_free(ssl_addr: Int):
    """Shut down and free the SSL* (does not close the fd)."""
    var ssl = BytePtr(unsafe_from_address=ssl_addr)
    _ = external_call["SSL_shutdown", Int32](ssl)
    _ = external_call["SSL_free", Int32](ssl)


def set_alpn(ctx_addr: Int):
    """Enable ALPN "h2" selection on the server context (cshim/alpn_shim.c).
    Lets a client negotiate HTTP/2 over TLS during the handshake."""
    var ctx = BytePtr(unsafe_from_address=ctx_addr)
    external_call["tls_set_alpn", NoneType](ctx)


def negotiated_h2(ssl_addr: Int) -> Bool:
    """After the handshake: did the client negotiate "h2" via ALPN?"""
    var ssl = BytePtr(unsafe_from_address=ssl_addr)
    return external_call["tls_is_h2", Int32](ssl) != 0


# ─────────────────────────────────────────────────────────────────────────────
# Client side: connect to a TLS server (https). Reuses conn_new / conn_read /
# conn_write_all / conn_free above (those are method-agnostic). Requires the same
# -lssl -lcrypto link.
# ─────────────────────────────────────────────────────────────────────────────

comptime SSL_VERIFY_NONE: Int32 = 0
comptime SSL_VERIFY_PEER: Int32 = 1
comptime SSL_CTRL_SET_TLSEXT_HOSTNAME: Int32 = 55
comptime TLSEXT_NAMETYPE_host_name: Int = 0
comptime X509_V_OK: Int = 0


def make_client_ctx(verify: Bool) raises -> Int:
    """Build a TLS *client* context. When `verify` is True, load the system CA
    store and require peer-certificate verification; otherwise accept any cert
    (use only for self-signed / testing). Returns the SSL_CTX* address."""
    _ = external_call["OPENSSL_init_ssl", Int32](UInt64(0), _nullp())
    var method = external_call["TLS_client_method", BytePtr]()
    var ctx = external_call["SSL_CTX_new", BytePtr](method)
    if Int(ctx) == 0:
        raise Error("SSL_CTX_new(client) failed: " + tls_err())
    if verify:
        var ok = external_call["SSL_CTX_set_default_verify_paths", Int32](ctx)
        if ok != 1:
            raise Error("TLS: failed to load system CA store: " + tls_err())
        _ = external_call["SSL_CTX_set_verify", NoneType](
            ctx, SSL_VERIFY_PEER, _nullp()
        )
    else:
        _ = external_call["SSL_CTX_set_verify", NoneType](
            ctx, SSL_VERIFY_NONE, _nullp()
        )
    return Int(ctx)


def client_connect(ssl_addr: Int, host: String, verify: Bool) raises:
    """Run the client TLS handshake on an SSL* (already bound to a connected fd
    via conn_new). Sets the SNI server-name to `host`; when `verify` is True also
    pins hostname verification and checks the certificate chain result."""
    var ssl = BytePtr(unsafe_from_address=ssl_addr)
    # SNI: SSL_set_tlsext_host_name(ssl, host) == SSL_ctrl(ssl, 55, 0, host).
    # OpenSSL strdup's the name during this call, so the buffer can be freed after.
    var hb = _cbuf(host)
    _ = external_call["SSL_ctrl", Int64](
        ssl, SSL_CTRL_SET_TLSEXT_HOSTNAME, TLSEXT_NAMETYPE_host_name, hb
    )
    if verify:
        # Pin the expected hostname so the chain check also validates the name.
        var hb2 = _cbuf(host)
        _ = external_call["SSL_set1_host", Int32](ssl, hb2)
        hb2.free()
    var r = external_call["SSL_connect", Int32](ssl)
    hb.free()
    if r != 1:
        raise Error("SSL_connect (TLS handshake) failed: " + tls_err())
    if verify:
        var vr = Int(external_call["SSL_get_verify_result", Int64](ssl))
        if vr != X509_V_OK:
            raise Error(
                "TLS certificate verification failed (X509 code "
                + String(vr)
                + ")"
            )


def conn_send_bytes(ssl_addr: Int, p: BytePtr, n: Int) -> Int:
    """Encrypt and send up to n bytes from p; returns bytes written (n on success,
    < n on a fatal error). Spins on WANT_READ/WANT_WRITE."""
    var ssl = BytePtr(unsafe_from_address=ssl_addr)
    var total = 0
    while total < n:
        var w = external_call["SSL_write", Int32](ssl, p + total, Int32(n - total))
        if w > 0:
            total += Int(w)
        else:
            var e = external_call["SSL_get_error", Int32](ssl, w)
            if e == SSL_ERROR_WANT_READ or e == SSL_ERROR_WANT_WRITE:
                continue
            return total
    return total


def free_ctx(ctx_addr: Int):
    """Free an SSL_CTX* created by make_client_ctx / make_ctx."""
    var ctx = BytePtr(unsafe_from_address=ctx_addr)
    _ = external_call["SSL_CTX_free", NoneType](ctx)
