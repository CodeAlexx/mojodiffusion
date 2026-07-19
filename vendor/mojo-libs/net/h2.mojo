# net.h2 — Mojo bindings for the HTTP/2 C shim (cshim/h2_shim.c, nghttp2).
#
# nghttp2 is callback-driven and Mojo can't hand it C-ABI callbacks, so the shim
# owns the session and we only shuttle bytes + whole requests/responses across.
# Conn/session handles are opaque C pointers carried as Int addresses.
#
# Build needs the shim object + nghttp2 linked, e.g.:
#   -Xlinker cshim/h2_shim.o -Xlinker <nghttp2>/lib/libnghttp2.so
#   -Xlinker -rpath -Xlinker <nghttp2>/lib

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


def h2_new() -> Int:
    """Create a server session (queues initial SETTINGS). Returns conn handle."""
    return external_call["h2_new", Int]()


def h2_recv(conn: Int, data: BytePtr, length: Int) -> Int:
    """Feed bytes read off the socket into the session. Returns consumed count,
    or <0 on a protocol error."""
    return external_call["h2_recv", Int](conn, data, length)


def h2_send(conn: Int, dst: BytePtr, cap: Int) -> Int:
    """Pull pending output bytes (frames to write to the socket). 0 = none."""
    return external_call["h2_send", Int](conn, dst, cap)


def h2_respond(
    conn: Int, sid: Int32, status: Int32, body: BytePtr, blen: Int, ctype: BytePtr
) -> Int32:
    return external_call["h2_respond", Int32](conn, sid, status, body, blen, ctype)


def h2_alive(conn: Int) -> Bool:
    return external_call["h2_alive", Int32](conn) != 0


def h2_free(conn: Int):
    _ = external_call["h2_free", Int32](conn)  # C returns void; ignore


def h2_request_body(conn: Int, sid: Int32, dst: BytePtr, cap: Int) -> Int:
    """Copy the popped request's body into dst[0:cap]; returns the actual body
    length (a value > cap means dst holds a truncated prefix)."""
    return external_call["h2_request_body", Int](conn, sid, dst, cap)


@fieldwise_init
struct H2Req(Copyable, Movable):
    var found: Bool
    var sid: Int
    var method: String
    var path: String
    var body: String


def h2_next_request(conn: Int) -> H2Req:
    """Pop one completed request from the session, if any (method, path, body)."""
    var sidb = alloc[Int32](1)
    var mb = alloc[UInt8](16)
    var pb = alloc[UInt8](1024)
    var blenb = alloc[Int](1)
    blenb[0] = 0
    var r = external_call["h2_next_request", Int32](
        conn,
        BytePtr(unsafe_from_address=Int(sidb)),
        BytePtr(unsafe_from_address=Int(mb)),
        16,
        BytePtr(unsafe_from_address=Int(pb)),
        1024,
        BytePtr(unsafe_from_address=Int(blenb)),
    )
    var res = H2Req(False, 0, String(""), String(""), String(""))
    if r == 1:
        var ml = 0
        while ml < 16 and mb[ml] != 0:
            ml += 1
        var pl = 0
        while pl < 1024 and pb[pl] != 0:
            pl += 1
        var sid = Int(sidb[0])
        var body = String("")
        var blen = blenb[0]
        if blen > 0:
            var bodyb = alloc[UInt8](blen)
            var bp = BytePtr(unsafe_from_address=Int(bodyb))
            var actual = h2_request_body(conn, Int32(sid), bp, blen)
            # blen came from out_blen so cap == actual; copy is exact
            var got = actual if actual < blen else blen
            body = String(StringSlice(ptr=bp, length=got))
            bodyb.free()
        res = H2Req(
            True,
            sid,
            String(StringSlice(ptr=BytePtr(unsafe_from_address=Int(mb)), length=ml)),
            String(StringSlice(ptr=BytePtr(unsafe_from_address=Int(pb)), length=pl)),
            body,
        )
    sidb.free()
    mb.free()
    pb.free()
    blenb.free()
    return res^
