# app — the application: route table, handlers, and middleware.
#
# Shared by both servers (plaintext main.mojo and TLS tls_server.mojo) so routing
# and handlers live in exactly one place. Handlers can't be stored as values in
# Mojo, so routes carry an integer id and dispatch() maps id -> handler.

from http.request import Request
from http.response import Response, http_date
from http.router import Router
from http.compress import gzip_bytes, brotli_bytes
from http.staticfiles import read_file, content_type_for
from json.parser import loads
from json.serialize import dumps
from json.value import JSONValue
from json.codec import req_int, req_str, req_bool, opt_str, field_list

comptime MAX_BODY = 1024 * 1024  # 1 MiB request-body cap (middleware)
comptime GZIP_MIN = 256  # don't bother compressing bodies smaller than this


# ── a typed request/response model (the FastAPI "model" equivalent) ──────────
struct User(Copyable, Movable):
    var id: Int
    var name: String
    var active: Bool
    var role: String
    def __init__(out self):
        self.id = 0
        self.name = String("")
        self.active = False
        self.role = String("user")


def parse_user(text: String) raises -> User:
    var obj = loads(text)
    if not obj.is_object():
        raise Error("expected a JSON object")
    var u = User()
    u.id = req_int(obj, "id")
    u.name = req_str(obj, "name")
    u.active = req_bool(obj, "active")
    u.role = opt_str(obj, "role", String("user"))
    return u^


def user_to_json(u: User) raises -> String:
    var o = JSONValue.new_object()
    o.set("id", JSONValue.from_int(u.id))
    o.set("name", JSONValue.from_string(u.name))
    o.set("active", JSONValue.from_bool(u.active))
    o.set("role", JSONValue.from_string(u.role))
    return dumps(o)


# ── Handlers ─────────────────────────────────────────────────────────────────
def home() -> Response:
    var r = Response(200)
    r.set_header("Content-Type", "text/plain")
    r.set_body("Hello from the Mojo HTTP Server!")
    return r^


def health() -> Response:
    var r = Response(200)
    r.set_header("Content-Type", "application/json")
    r.set_body('{"status":"healthy","server":"Mojo HTTP Server"}')
    return r^


def status_resp() -> Response:
    var r = Response(200)
    r.set_header("Content-Type", "application/json")
    r.set_body('{"server":"Mojo HTTP Server","version":"3.2.0","status":"operational"}')
    return r^


def user_page(req: Request) raises -> Response:
    var r = Response(200)
    r.set_header("Content-Type", "application/json")
    r.set_body('{"user_id":"' + req.param("id") + '"}')
    return r^


def echo(req: Request) -> Response:
    var r = Response(200)
    r.set_header("Content-Type", "text/plain")
    r.set_body("echo: " + req.query("msg"))
    return r^


def echo_body(req: Request) -> Response:
    """POST /echo — return the request body verbatim. Exercises the request-body
    path (HTTP/1.1 and HTTP/2 alike); used to verify byte-identical body capture."""
    var r = Response(200)
    r.set_header("Content-Type", "application/octet-stream")
    r.set_body(req.body)
    return r^


def big() -> Response:
    """A large, compressible body — demonstrates gzip (Content-Encoding: gzip)."""
    var line = String('{"item":"the quick brown fox jumps over the lazy dog"},\n')
    var body = String('{"items":[\n')
    for _ in range(80):
        body += line
    body += "]}"
    var r = Response(200)
    r.set_header("Content-Type", "application/json")
    r.set_body(body)
    return r^


def not_found() -> Response:
    var r = Response(404)
    r.set_header("Content-Type", "text/html")
    r.set_body("<html><body><h1>404 Not Found</h1></body></html>")
    return r^


def static_file(req: Request) raises -> Response:
    """Serve www/<file>. Single path segment only; rejects traversal."""
    var name = req.param("file")
    if name.find("..") >= 0 or name.find("/") >= 0 or name == "":
        var f = Response(403)
        f.set_header("Content-Type", "text/plain")
        f.set_body("Forbidden")
        return f^
    var content = read_file("www/" + name)
    if content:
        var r = Response(200)
        r.set_header("Content-Type", content_type_for(name))
        r.set_body(content.value().copy())
        return r^
    return not_found()


def _safe_www(name: String) -> Bool:
    return not (name.find("..") >= 0 or name.find("/") >= 0 or name == "")


def stream_file(req: Request) raises -> Response:
    """Serve www/<file> by STREAMING it from disk (body_mode 1): the server
    pumps it chunk-by-chunk with a known Content-Length, so the whole file
    never sits in memory. Same bytes as /static, but O(chunk) memory."""
    var name = req.param("file")
    if not _safe_www(name):
        var f = Response(403)
        f.set_header("Content-Type", "text/plain")
        f.set_body("Forbidden")
        return f^
    var r = Response(200)
    r.set_header("Content-Type", content_type_for(name))
    r.stream_file("www/" + name)  # open/size/404 decided by the server
    return r^


def chunked_file(req: Request) raises -> Response:
    """Serve www/<file> with Transfer-Encoding: chunked (body_mode 2) — the
    unknown-length streaming case (the length is never pre-declared)."""
    var name = req.param("file")
    if not _safe_www(name):
        var f = Response(403)
        f.set_header("Content-Type", "text/plain")
        f.set_body("Forbidden")
        return f^
    var r = Response(200)
    r.set_header("Content-Type", content_type_for(name))
    r.stream_file_chunked("www/" + name)
    return r^


def create_user(req: Request) raises -> Response:
    """POST /users — validate the JSON body into a User, echo it back. The
    FastAPI flow: parse + validate (codec) -> typed model -> serialize. A
    validation failure becomes a 422 with the reason, like FastAPI."""
    try:
        var u = parse_user(req.body)
        var r = Response(201)
        r.set_header("Content-Type", "application/json")
        r.set_body(user_to_json(u))
        return r^
    except e:
        # e's message is a FastAPI-shaped error entry; wrap as {"detail":[ ... ]}
        var r = Response(422)
        r.set_header("Content-Type", "application/json")
        r.set_body('{"detail":[' + String(e) + "]}")
        return r^


def openapi_json() raises -> Response:
    """GET /openapi.json — an OpenAPI 3.0 document. The User schema's `required`
    list comes from reflection (field_list[User]); property types are declared
    here (per-field-type reflection in a loop is a Mojo 1.0.0b1 limitation)."""
    var doc = String('{"openapi":"3.0.0","info":{"title":"Mojo API","version":"1.0.0"},')
    doc += '"paths":{'
    doc += '"/users":{"post":{"summary":"Create a user",'
    doc += '"requestBody":{"required":true,"content":{"application/json":{"schema":{"$ref":"#/components/schemas/User"}}}},'
    doc += '"responses":{"201":{"description":"Created"},"422":{"description":"Validation Error"}}}},'
    doc += '"/health":{"get":{"summary":"Health check","responses":{"200":{"description":"OK"}}}}'
    doc += '},'
    doc += '"components":{"schemas":{"User":{"type":"object",'
    doc += '"required":' + field_list[User]() + ','  # ← from reflection
    doc += '"properties":{"id":{"type":"integer"},"name":{"type":"string"},'
    doc += '"active":{"type":"boolean"},"role":{"type":"string"}}}}}}'
    var r = Response(200)
    r.set_header("Content-Type", "application/json")
    r.set_body(doc)
    return r^


def docs_html() -> Response:
    """GET /docs — Swagger UI (from CDN) pointed at /openapi.json."""
    var h = String("<!doctype html><html><head><title>Mojo API docs</title>")
    h += '<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css"></head>'
    h += '<body><div id="swagger-ui"></div>'
    h += '<script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>'
    h += '<script>window.onload=function(){SwaggerUIBundle({url:"/openapi.json",dom_id:"#swagger-ui"})}</script>'
    h += "</body></html>"
    var r = Response(200)
    r.set_header("Content-Type", "text/html")
    r.set_body(h)
    return r^


def dispatch(handler_id: Int, req: Request) raises -> Response:
    if handler_id == 0:
        return home()
    if handler_id == 1:
        return health()
    if handler_id == 2:
        return status_resp()
    if handler_id == 3:
        return user_page(req)
    if handler_id == 4:
        return echo(req)
    if handler_id == 5:
        return big()
    if handler_id == 6:
        return static_file(req)
    if handler_id == 7:
        return create_user(req)
    if handler_id == 8:
        return openapi_json()
    if handler_id == 9:
        return docs_html()
    if handler_id == 10:
        return stream_file(req)
    if handler_id == 11:
        return chunked_file(req)
    if handler_id == 12:
        return echo_body(req)
    return not_found()


def build_router() -> Router:
    var r = Router()
    r.add("GET", "/", 0)
    r.add("GET", "/health", 1)
    r.add("GET", "/api/status", 2)
    r.add("GET", "/user/{id}", 3)
    r.add("GET", "/echo", 4)
    r.add("GET", "/big", 5)
    r.add("GET", "/static/{file}", 6)
    r.add("POST", "/users", 7)        # typed+validated body
    r.add("GET", "/openapi.json", 8)  # reflection-fed schema
    r.add("GET", "/docs", 9)          # Swagger UI
    r.add("GET", "/stream/{file}", 10)   # streamed file (Content-Length)
    r.add("GET", "/chunked/{file}", 11)  # streamed file (Transfer-Encoding: chunked)
    r.add("POST", "/echo", 12)           # echo the request body verbatim
    return r^


# ── Middleware hooks (applied to every request) ──────────────────────────────
def before_request(req: Request) -> Optional[Response]:
    """Return a Response to short-circuit (skip the handler), or None to continue."""
    if req.body.byte_length() > MAX_BODY:
        var r = Response(413)
        r.set_body("Payload Too Large")
        return r^
    return None


def after_response(req: Request, mut resp: Response):
    """Cross-cutting headers on every response."""
    resp.set_header("Server", "MojoHttpServer/3.2")
    resp.set_header("Date", http_date())


def keep_alive_wanted(req: Request) raises -> Bool:
    var connl = String(req.header("connection").lower())
    if req.version == "HTTP/1.1":
        return connl != "close"
    return connl == "keep-alive"


def is_ws_upgrade(req: Request) raises -> Bool:
    var up = String(req.header("upgrade").lower())
    var conn = String(req.header("connection").lower())
    return up == "websocket" and conn.find("upgrade") >= 0


def maybe_compress(req: Request, mut resp: Response) raises:
    """Compress the response body when the client accepts it and the body is large
    enough. Prefers brotli (br) over gzip. Sets Content-Encoding; serialize()
    recomputes Content-Length from the compressed body."""
    if resp.body.byte_length() < GZIP_MIN:
        return
    if "Content-Encoding" in resp.headers:
        return  # already encoded
    var ae = String(req.header("accept-encoding").lower())
    if ae.find("br") >= 0:
        var br = brotli_bytes(resp.body)
        if br.byte_length() > 0 and br.byte_length() < resp.body.byte_length():
            resp.set_body(br)
            resp.set_header("Content-Encoding", "br")
            return
    if ae.find("gzip") >= 0:
        var gz = gzip_bytes(resp.body)
        if gz.byte_length() > 0 and gz.byte_length() < resp.body.byte_length():
            resp.set_body(gz)
            resp.set_header("Content-Encoding", "gzip")
