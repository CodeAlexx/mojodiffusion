# net.tcp — TCPListener / TCPStream, a small std::net-style blocking TCP API.
#
# TCPListener binds + listens in its constructor and hands back a TCPStream per
# accepted connection. TCPStream reads/writes that connection and closes it when
# it goes out of scope (the inner Socket's RAII). All errors carry strerror text.

from net.syscalls import (
    BytePtr,
    sys_socket,
    sys_bind,
    sys_listen,
    sys_accept,
    sys_connect,
    errno_str,
    AF_INET,
    SOCK_STREAM,
)
from net.socket import Socket
from std.memory import alloc


def _fill_sockaddr_in(buf: BytePtr, port: Int):
    """Hand-pack a 16-byte `struct sockaddr_in` (Linux x86-64), bound to
    INADDR_ANY on `port`. Layout: sin_family(2) sin_port(2,BE) sin_addr(4) pad(8).
    Equivalent to server.c's _hs_bind_localhost with ipaddr == NULL."""
    for i in range(16):
        buf[i] = 0
    buf[0] = 2  # sin_family = AF_INET (little-endian low byte)
    buf[2] = UInt8((port >> 8) & 0xFF)  # sin_port high byte (network order)
    buf[3] = UInt8(port & 0xFF)  # sin_port low byte
    # sin_addr = INADDR_ANY = 0 → bytes 4..7 stay zero


struct TCPStream(Movable):
    """A connected client socket. Owns the connection fd via `sock`."""

    var sock: Socket

    def __init__(out self, var sock: Socket):
        self.sock = sock^

    def read(mut self, max_bytes: Int = 65536) raises -> String:
        """Read up to `max_bytes` from one recv. Returns "" if the peer closed
        with no data. Raises on a recv error."""
        var buf = alloc[UInt8](max_bytes)
        var p = BytePtr(unsafe_from_address=Int(buf))
        var n = self.sock.recv_into(p, max_bytes)
        if n < 0:
            buf.free()
            raise Error("recv failed: " + errno_str())
        var out = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=buf, length=n)))
        buf.free()
        return out

    def write_all(mut self, data: String) raises:
        """Write the whole string, looping until every byte is sent."""
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
            var sent = self.sock.send_bytes(p + total, n - total)
            if sent <= 0:
                buf.free()
                raise Error("send failed: " + errno_str())
            total += sent
        buf.free()


struct TCPListener(Movable):
    """A listening server socket. Binds + listens on construction."""

    var sock: Socket
    var port: Int

    def __init__(out self, port: Int) raises:
        var fd = sys_socket(AF_INET, SOCK_STREAM, Int32(0))
        if fd < 0:
            raise Error("socket() failed: " + errno_str())
        var s = Socket(fd)
        s.set_reuse()

        var addr = alloc[UInt8](16)
        var ap = BytePtr(unsafe_from_address=Int(addr))
        _fill_sockaddr_in(ap, port)
        var brc = sys_bind(fd, ap, Int32(16))
        addr.free()
        if brc < 0:
            # `s` destructs here, closing fd.
            raise Error("bind() failed on port " + String(port) + ": " + errno_str())
        if sys_listen(fd, Int32(128)) < 0:
            raise Error("listen() failed: " + errno_str())

        self.sock = s^
        self.port = port

    def accept(mut self) raises -> TCPStream:
        """Block until a client connects; return its stream."""
        var null = BytePtr(unsafe_from_address=Int(0))
        var cfd = sys_accept(self.sock.fd, null, null)
        if cfd < 0:
            raise Error("accept() failed: " + errno_str())
        return TCPStream(Socket(cfd))


def _parse_ipv4(ip: String) raises -> List[UInt8]:
    """Dotted-quad "a.b.c.d" -> 4 network-order bytes. Raises on malformed."""
    var parts = ip.split(".")
    if len(parts) != 4:
        raise Error("not a dotted-quad IPv4: " + ip)
    var out = List[UInt8]()
    for i in range(4):
        var seg = String(parts[i])
        var v = 0
        var sb = seg.as_bytes()
        if seg.byte_length() == 0:
            raise Error("bad IPv4 octet in: " + ip)
        for j in range(seg.byte_length()):
            var c = Int(sb[j])
            if c < 48 or c > 57:
                raise Error("bad IPv4 octet in: " + ip)
            v = v * 10 + (c - 48)
        out.append(UInt8(v & 0xFF))
    return out^


def tcp_connect(ip: String, port: Int) raises -> Socket:
    """Connect to `ip`:`port` (dotted-quad IPv4, no DNS). Returns a connected,
    blocking Socket. Used by the HTTP client."""
    var fd = sys_socket(AF_INET, SOCK_STREAM, Int32(0))
    if fd < 0:
        raise Error("socket() failed: " + errno_str())
    var s = Socket(fd)
    var ipb = _parse_ipv4(ip)
    var addr = alloc[UInt8](16)
    var ap = BytePtr(unsafe_from_address=Int(addr))
    for i in range(16):
        addr[i] = 0
    addr[0] = 2  # AF_INET
    addr[2] = UInt8((port >> 8) & 0xFF)  # sin_port (network order)
    addr[3] = UInt8(port & 0xFF)
    addr[4] = ipb[0]  # sin_addr (already network order: a.b.c.d)
    addr[5] = ipb[1]
    addr[6] = ipb[2]
    addr[7] = ipb[3]
    var rc = sys_connect(fd, ap, Int32(16))
    addr.free()
    if rc < 0:
        raise Error("connect() failed: " + errno_str())
    return s^
