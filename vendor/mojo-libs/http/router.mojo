# http.router — a small data-driven router with path parameters.
#
# Mojo can't store function values (no dynamic traits yet), so a route can't hold
# a handler *callable* the way aiohttp does. Instead each Route carries an integer
# `handler_id`; the application maps id -> handler in one dispatch function. The
# router's job is matching: given (method, path) it returns the matching id and
# any captured path params, or -1 for no match.
#
#   var r = Router()
#   r.add("GET", "/user/{id}", 3)
#   var m = r.find("GET", "/user/42")   # m.handler_id == 3, m.params["id"] == "42"

from http.request import byte_substr


@fieldwise_init
struct Route(Copyable, Movable):
    var method: String  # "GET", "POST", … or "*" to match any method
    var pattern: String  # "/user/{id}" — {name} segments capture params
    var handler_id: Int


@fieldwise_init
struct RouteMatch(Copyable, Movable):
    var handler_id: Int  # -1 = no route matched
    var params: Dict[String, String]


def _match(pattern: String, path: String, mut params: Dict[String, String]) -> Bool:
    """Segment-wise match. Equal segment counts required; a `{name}` pattern
    segment captures the corresponding path segment into `params`."""
    var pp = pattern.split("/")
    var sp = path.split("/")
    if len(pp) != len(sp):
        return False
    for i in range(len(pp)):
        var pseg = String(pp[i])
        var sseg = String(sp[i])
        if pseg.startswith("{") and pseg.endswith("}"):
            var name = byte_substr(pseg, 1, pseg.byte_length() - 1)
            params[name] = sseg
        elif pseg != sseg:
            return False
    return True


struct Router(Movable):
    """An ordered list of routes; first match wins."""

    var routes: List[Route]

    def __init__(out self):
        self.routes = List[Route]()

    def add(mut self, method: String, pattern: String, handler_id: Int):
        self.routes.append(Route(method, pattern, handler_id))

    def allow_for(self, path: String) -> String:
        """Comma-joined methods whose pattern matches `path` (any method), or ""
        if no route matches the path. Used to build the Allow header for 405/OPTIONS."""
        var methods = String("")
        for i in range(len(self.routes)):
            var rt = self.routes[i].copy()
            var params = Dict[String, String]()
            if _match(rt.pattern, path, params):
                if methods.find(rt.method) < 0:
                    if methods != "":
                        methods += ", "
                    methods += rt.method
        return methods

    def find(self, method: String, path: String) -> RouteMatch:
        """First route whose method (or "*") and pattern match `path`. A HEAD
        request falls back to the matching GET route (HTTP semantics: HEAD is GET
        without a body — the server should send headers only)."""
        # effective method: HEAD reuses GET routes
        var meth = String("GET") if method == "HEAD" else method
        for i in range(len(self.routes)):
            var rt = self.routes[i].copy()
            if rt.method != "*" and rt.method != meth:
                continue
            var params = Dict[String, String]()
            if _match(rt.pattern, path, params):
                return RouteMatch(rt.handler_id, params^)
        return RouteMatch(-1, Dict[String, String]())
