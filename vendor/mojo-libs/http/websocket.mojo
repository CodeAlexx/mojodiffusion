# http.websocket — RFC 6455 WebSocket support (handshake + framing).
#
# No crypto library is linked, so SHA-1 is implemented here in pure Mojo (verified
# against the SHA1("abc") test vector and the RFC 6455 handshake example). The
# handshake upgrades an HTTP connection; after that the same fd carries WebSocket
# frames instead of HTTP requests.
#
# Frame layout (RFC 6455 §5.2):
#   byte0: FIN(0x80) | opcode(0x0F)        opcodes: 0x1 text, 0x2 binary,
#   byte1: MASK(0x80) | len7(0x7F)                  0x8 close, 0x9 ping, 0xA pong
#   [len16 if len7==126 | len64 if len7==127]
#   [4-byte masking key if MASK]           client→server frames are ALWAYS masked
#   payload (XOR-unmasked with key[i%4])   server→client frames are NEVER masked

from std.memory import alloc, UnsafePointer

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]
comptime B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
comptime WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Opcodes
comptime OP_CONT = 0x0   # continuation frame (a fragment of an earlier message)
comptime OP_TEXT = 0x1
comptime OP_BINARY = 0x2
comptime OP_CLOSE = 0x8
comptime OP_PING = 0x9
comptime OP_PONG = 0xA


# ── byte buffer → String ─────────────────────────────────────────────────────
def _b2s(b: List[UInt8]) -> String:
    var n = len(b)
    if n == 0:
        return String("")
    var buf = alloc[UInt8](n)
    for i in range(n):
        buf[i] = b[i]
    var s = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=BytePtr(unsafe_from_address=Int(buf)), length=n)))
    buf.free()
    return s


# ── SHA-1 (pure Mojo) ────────────────────────────────────────────────────────
def _rotl32(x: UInt32, c: Int) -> UInt32:
    return (x << UInt32(c)) | (x >> UInt32(32 - c))


def sha1(data: String) -> List[UInt8]:
    """SHA-1 digest (20 bytes). Verified against SHA1("abc")."""
    var msg = List[UInt8]()
    var src = data.as_bytes()
    var ml = data.byte_length()
    for i in range(ml):
        msg.append(src[i])
    msg.append(0x80)
    while (len(msg) % 64) != 56:
        msg.append(0)
    var bitlen = UInt64(ml) * 8
    for i in range(8):
        msg.append(UInt8(Int((bitlen >> UInt64(56 - 8 * i)) & 0xFF)))

    var h0: UInt32 = 0x67452301
    var h1: UInt32 = 0xEFCDAB89
    var h2: UInt32 = 0x98BADCFE
    var h3: UInt32 = 0x10325476
    var h4: UInt32 = 0xC3D2E1F0
    var nchunks = len(msg) // 64
    for ci in range(nchunks):
        var w = List[UInt32]()
        for i in range(16):
            var b = ci * 64 + i * 4
            var word = (
                (UInt32(Int(msg[b])) << 24)
                | (UInt32(Int(msg[b + 1])) << 16)
                | (UInt32(Int(msg[b + 2])) << 8)
                | UInt32(Int(msg[b + 3]))
            )
            w.append(word)
        for i in range(16, 80):
            w.append(_rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1))
        var a = h0
        var bb = h1
        var c = h2
        var d = h3
        var e = h4
        for i in range(80):
            var f: UInt32
            var k: UInt32
            if i < 20:
                f = (bb & c) | ((~bb) & d)
                k = 0x5A827999
            elif i < 40:
                f = bb ^ c ^ d
                k = 0x6ED9EBA1
            elif i < 60:
                f = (bb & c) | (bb & d) | (c & d)
                k = 0x8F1BBCDC
            else:
                f = bb ^ c ^ d
                k = 0xCA62C1D6
            var tmp = _rotl32(a, 5) + f + e + k + w[i]
            e = d
            d = c
            c = _rotl32(bb, 30)
            bb = a
            a = tmp
        h0 += a
        h1 += bb
        h2 += c
        h3 += d
        h4 += e

    var out = List[UInt8]()
    var hs = [h0, h1, h2, h3, h4]
    for hv in hs:
        for i in range(4):
            out.append(UInt8(Int((hv >> UInt32(24 - 8 * i)) & 0xFF)))
    return out^


# ── base64 ───────────────────────────────────────────────────────────────────
def base64_encode(data: List[UInt8]) -> String:
    var b64s = String(B64)  # keep alive while we index its bytes
    var alpha = b64s.as_bytes()
    var out = List[UInt8]()
    var n = len(data)
    var i = 0
    while i + 3 <= n:
        var b0 = Int(data[i])
        var b1 = Int(data[i + 1])
        var b2 = Int(data[i + 2])
        out.append(alpha[(b0 >> 2) & 0x3F])
        out.append(alpha[((b0 & 3) << 4) | (b1 >> 4)])
        out.append(alpha[((b1 & 15) << 2) | (b2 >> 6)])
        out.append(alpha[b2 & 0x3F])
        i += 3
    var rem = n - i
    if rem == 1:
        var b0 = Int(data[i])
        out.append(alpha[(b0 >> 2) & 0x3F])
        out.append(alpha[(b0 & 3) << 4])
        out.append(UInt8(61))  # '='
        out.append(UInt8(61))
    elif rem == 2:
        var b0 = Int(data[i])
        var b1 = Int(data[i + 1])
        out.append(alpha[(b0 >> 2) & 0x3F])
        out.append(alpha[((b0 & 3) << 4) | (b1 >> 4)])
        out.append(alpha[(b1 & 15) << 2])
        out.append(UInt8(61))
    return _b2s(out)


# ── handshake ────────────────────────────────────────────────────────────────
def ws_accept_key(client_key: String) -> String:
    """Sec-WebSocket-Accept = base64(SHA1(client_key + GUID))."""
    return base64_encode(sha1(client_key + WS_GUID))


def handshake_response(client_key: String) -> String:
    """The 101 Switching Protocols response that completes the upgrade."""
    var out = String("HTTP/1.1 101 Switching Protocols\r\n")
    out += "Upgrade: websocket\r\n"
    out += "Connection: Upgrade\r\n"
    out += "Sec-WebSocket-Accept: " + ws_accept_key(client_key) + "\r\n\r\n"
    return out


# ── framing ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct WsFrame(Copyable, Movable):
    var opcode: Int
    var payload: String
    var fin: Bool
    var consumed: Int  # bytes used from the buffer; -1 = need more bytes


def decode_frame(buf: String) -> WsFrame:
    """Decode one frame from the front of `buf`. consumed == -1 means the buffer
    doesn't yet hold a full frame (wait for more bytes)."""
    var sp = buf.as_bytes()
    var n = buf.byte_length()
    if n < 2:
        return WsFrame(0, String(""), False, -1)
    var b0 = Int(sp[0])
    var b1 = Int(sp[1])
    var fin = (b0 & 0x80) != 0
    var opcode = b0 & 0x0F
    var masked = (b1 & 0x80) != 0
    var len7 = b1 & 0x7F
    var idx = 2
    var paylen = len7
    if len7 == 126:
        if n < 4:
            return WsFrame(0, String(""), False, -1)
        paylen = (Int(sp[2]) << 8) | Int(sp[3])
        idx = 4
    elif len7 == 127:
        if n < 10:
            return WsFrame(0, String(""), False, -1)
        paylen = 0
        for i in range(8):
            paylen = (paylen << 8) | Int(sp[2 + i])
        idx = 10
    var m0 = 0
    var m1 = 0
    var m2 = 0
    var m3 = 0
    if masked:
        if n < idx + 4:
            return WsFrame(0, String(""), False, -1)
        m0 = Int(sp[idx])
        m1 = Int(sp[idx + 1])
        m2 = Int(sp[idx + 2])
        m3 = Int(sp[idx + 3])
        idx += 4
    if n < idx + paylen:
        return WsFrame(0, String(""), False, -1)
    var pb = List[UInt8]()
    for i in range(paylen):
        var c = Int(sp[idx + i])
        if masked:
            var mk = m0
            if i % 4 == 1:
                mk = m1
            elif i % 4 == 2:
                mk = m2
            elif i % 4 == 3:
                mk = m3
            c = c ^ mk
        pb.append(UInt8(c))
    return WsFrame(opcode, _b2s(pb), fin, idx + paylen)


# ── fragmentation reassembly (RFC 6455 §5.4) ─────────────────────────────────
# A message may arrive as multiple frames: an initial TEXT/BINARY frame with
# FIN=0, then CONT (0x0) frames, the last with FIN=1. Control frames (close/
# ping/pong) are never fragmented and may be interjected mid-message, so they
# pass straight through without touching the reassembly buffer.
@fieldwise_init
struct WsMessage(Copyable, Movable):
    var opcode: Int      # OP_TEXT/OP_BINARY for a data message; control opcode otherwise
    var payload: String
    var ready: Bool      # False = fragment buffered, nothing to deliver yet


struct WsReassembler(Copyable, Movable):
    """Feeds decoded frames in; emits a WsMessage when one is complete. Holds the
    in-progress data message across fragments. One per connection."""

    var frag_opcode: Int  # opcode of the message being assembled; 0 = none in progress
    var frag_buf: String

    def __init__(out self):
        self.frag_opcode = 0
        self.frag_buf = String("")

    def in_progress(self) -> Bool:
        return self.frag_opcode != 0

    def push(mut self, fr: WsFrame) -> WsMessage:
        """Advance reassembly with one decoded frame. Returns a ready message, or
        ready=False if more fragments are still needed."""
        var op = fr.opcode
        # control frames are self-contained — never fragmented, never buffered
        if op == OP_CLOSE or op == OP_PING or op == OP_PONG:
            return WsMessage(op, fr.payload, True)
        if op == OP_CONT:
            if self.frag_opcode == 0:
                return WsMessage(0, String(""), False)  # stray continuation — ignore
            self.frag_buf = self.frag_buf + fr.payload
            if fr.fin:
                var done = WsMessage(self.frag_opcode, self.frag_buf, True)
                self.frag_opcode = 0
                self.frag_buf = String("")
                return done^
            return WsMessage(0, String(""), False)
        # a data frame (TEXT/BINARY)
        if fr.fin:
            return WsMessage(op, fr.payload, True)  # unfragmented, single frame
        # first fragment of a new message
        self.frag_opcode = op
        self.frag_buf = fr.payload
        return WsMessage(0, String(""), False)


def _encode_with_b0(b0: Int, payload: String) -> String:
    """Encode a server→client frame with a caller-supplied first byte (FIN+opcode),
    no mask (server frames are never masked). Shared by the fragment encoders."""
    var pb = payload.as_bytes()
    var n = payload.byte_length()
    var out = List[UInt8]()
    out.append(UInt8(b0))
    if n < 126:
        out.append(UInt8(n))
    elif n < 65536:
        out.append(UInt8(126))
        out.append(UInt8((n >> 8) & 0xFF))
        out.append(UInt8(n & 0xFF))
    else:
        out.append(UInt8(127))
        for i in range(8):
            out.append(UInt8((n >> (56 - 8 * i)) & 0xFF))
    for i in range(n):
        out.append(pb[i])
    return _b2s(out)


def encode_first(opcode: Int, payload: String) -> String:
    """First frame of a fragmented server→client message: TEXT/BINARY, FIN=0."""
    return _encode_with_b0(opcode & 0x0F, payload)  # FIN bit clear


def encode_cont(payload: String, fin: Bool) -> String:
    """A continuation frame (opcode 0x0). FIN=1 marks the last fragment."""
    var b0 = 0x80 | OP_CONT if fin else OP_CONT
    return _encode_with_b0(b0, payload)


def encode_frame(opcode: Int, payload: String) -> String:
    """Encode a server→client frame (FIN set, never masked)."""
    var pb = payload.as_bytes()
    var n = payload.byte_length()
    var out = List[UInt8]()
    out.append(UInt8(0x80 | opcode))  # FIN + opcode
    if n < 126:
        out.append(UInt8(n))
    elif n < 65536:
        out.append(UInt8(126))
        out.append(UInt8((n >> 8) & 0xFF))
        out.append(UInt8(n & 0xFF))
    else:
        out.append(UInt8(127))
        for i in range(8):
            out.append(UInt8((n >> (56 - 8 * i)) & 0xFF))
    for i in range(n):
        out.append(pb[i])
    return _b2s(out)


def encode_text(payload: String) -> String:
    return encode_frame(OP_TEXT, payload)


def encode_close() -> String:
    return encode_frame(OP_CLOSE, String(""))


def encode_pong(payload: String) -> String:
    return encode_frame(OP_PONG, payload)
