# http.response — an HTTP/1.1 response and its serializer.
#
# Build a Response, set status / headers / body, then serialize() to the bytes
# sent on the wire. Content-Length is computed from the body and Connection
# reflects keep_alive. status_text mirrors httpserver.h's hs_status_text table.

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


def http_date() -> String:
    """Current time as an RFC 7231 HTTP-date (e.g. "Tue, 09 Jun 2026 18:29:17 GMT")
    via libc time/gmtime/strftime."""
    var now = external_call["time", Int](BytePtr(unsafe_from_address=Int(0)))
    var tt = alloc[Int](1)
    tt[0] = now
    var tm = external_call["gmtime", BytePtr](BytePtr(unsafe_from_address=Int(tt)))
    var fmt = String("%a, %d %b %Y %H:%M:%S GMT")
    var fb = alloc[UInt8](fmt.byte_length() + 1)
    var fs = fmt.as_bytes()
    for i in range(fmt.byte_length()):
        fb[i] = fs[i]
    fb[fmt.byte_length()] = 0
    var buf = alloc[UInt8](64)
    var n = external_call["strftime", Int](
        BytePtr(unsafe_from_address=Int(buf)), 64,
        BytePtr(unsafe_from_address=Int(fb)), tm,
    )
    var s = String(StringSlice(ptr=BytePtr(unsafe_from_address=Int(buf)), length=n))
    tt.free()
    fb.free()
    buf.free()
    return s


def status_text(code: Int) -> String:
    """Reason phrase for a status code (the codes httpserver.h ships)."""
    if code == 200:
        return String("OK")
    if code == 201:
        return String("Created")
    if code == 204:
        return String("No Content")
    if code == 206:
        return String("Partial Content")
    if code == 301:
        return String("Moved Permanently")
    if code == 302:
        return String("Found")
    if code == 304:
        return String("Not Modified")
    if code == 307:
        return String("Temporary Redirect")
    if code == 400:
        return String("Bad Request")
    if code == 401:
        return String("Unauthorized")
    if code == 403:
        return String("Forbidden")
    if code == 404:
        return String("Not Found")
    if code == 405:
        return String("Method Not Allowed")
    if code == 413:
        return String("Payload Too Large")
    if code == 500:
        return String("Internal Server Error")
    return String("")


struct Response(Copyable, Movable):
    """An HTTP response: status code, headers, and a body.

    Two body shapes:
      * buffered  (body_mode 0)  — the bytes live in `body`; Content-Length is
        the body length. This is the default and covers every handler that
        builds a String.
      * streamed  (body_mode 1/2) — the body is NOT materialized here; the
        server streams it from `stream_path` chunk-by-chunk so a large file
        never sits in memory whole. Mode 1 frames with a known Content-Length
        (`content_length_override`); mode 2 frames with Transfer-Encoding:
        chunked (unknown length). The server's write loop does the pumping."""

    var status: Int
    var headers: Dict[String, String]
    var body: String
    var keep_alive: Bool  # emits Connection: keep-alive vs close
    var body_mode: Int    # 0 = buffered, 1 = file w/ Content-Length, 2 = file chunked
    var stream_path: String           # file to stream when body_mode != 0
    var content_length_override: Int  # >=0 overrides the Content-Length (mode 1)

    def __init__(out self, status: Int = 200):
        self.status = status
        self.headers = Dict[String, String]()
        self.body = String("")
        self.keep_alive = False
        self.body_mode = 0
        self.stream_path = String("")
        self.content_length_override = -1

    def set_header(mut self, key: String, value: String):
        self.headers[key] = value

    def set_body(mut self, body: String):
        self.body = body

    def set_keep_alive(mut self, value: Bool):
        self.keep_alive = value

    def stream_file(mut self, path: String):
        """Stream `path` with a known Content-Length (server stats it)."""
        self.body_mode = 1
        self.stream_path = path

    def stream_file_chunked(mut self, path: String):
        """Stream `path` with Transfer-Encoding: chunked (length not pre-declared)."""
        self.body_mode = 2
        self.stream_path = path

    def is_streaming(self) -> Bool:
        return self.body_mode != 0

    def _head(self) -> String:
        """Status line + headers + framing + Connection, ending with the blank
        line. Framing is Transfer-Encoding: chunked for mode 2, else
        Content-Length (override for a streamed file, body length otherwise —
        the latter is correct for HEAD)."""
        var out = String("HTTP/1.1 ")
        out += String(self.status)
        out += " "
        out += status_text(self.status)
        out += "\r\n"
        if self.body_mode == 2:
            out += "Transfer-Encoding: chunked\r\n"
        else:
            var cl = self.body.byte_length()
            if self.content_length_override >= 0:
                cl = self.content_length_override
            out += "Content-Length: "
            out += String(cl)
            out += "\r\n"
        for entry in self.headers.items():
            out += entry.key
            out += ": "
            out += entry.value
            out += "\r\n"
        if self.keep_alive:
            out += "Connection: keep-alive\r\n\r\n"
        else:
            out += "Connection: close\r\n\r\n"
        return out

    def serialize(self) -> String:
        """Full HTTP/1.1 response (headers + body)."""
        return self._head() + self.body

    def serialize_head(self) -> String:
        """Headers only (for HEAD requests): same headers + Content-Length, no body."""
        return self._head()
