# Mojo HTTP client CLI — fetch a URL path from a host and print the result.
#
#   mojo build client_main.mojo -o /tmp/mojoget
#   /tmp/mojoget /health          # GET 127.0.0.1:8080/health
#   /tmp/mojoget /health 8080 127.0.0.1

from sys import argv
from http.client import get


def _atoi(s: String) -> Int:
    var v = 0
    var sb = s.as_bytes()
    for i in range(s.byte_length()):
        var c = Int(sb[i])
        if c < 48 or c > 57:
            break
        v = v * 10 + (c - 48)
    return v


def main() raises:
    var a = argv()
    var path = String(a[1]) if len(a) >= 2 else String("/")
    var port = _atoi(String(a[2])) if len(a) >= 3 else 8080
    var ip = String(a[3]) if len(a) >= 4 else String("127.0.0.1")

    var r = get(ip, port, path, String(""))
    print("GET", ip + ":" + String(port) + path, "->", r.status)
    print("---")
    print(r.body)
