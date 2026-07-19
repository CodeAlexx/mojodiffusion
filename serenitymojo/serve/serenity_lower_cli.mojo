# serenitymojo.serve.serenity_lower_cli — LOWER-ONLY workflow oracle binary.
#
# A standalone CLI that runs the SAME workflow-graph lowering the daemon applies
# to an incoming /v1/generate request, then prints the lowered request object to
# stdout. It is the PARITY ORACLE for the Rust executor: given a request body it
# emits the canonical flat genparams the daemon would produce, so the Rust port
# can be diffed byte-for-byte against this Mojo reference.
#
# NO GPU. NO MAX. NO generation. NO server. NO sockets. It imports ONLY:
#   - json.parser.loads / json.serialize.dumps / json.value.JSONValue
#   - serenitymojo.serve.workflow_graph.apply_workflow_params
#   - serenitymojo.io.ffi  (sys_open / file_size / sys_pread / sys_close, O_RDONLY)
# workflow_graph itself imports only `json` (no tensors, no backends), so this
# binary has the same tiny footprint as serenity_worker_stub.
#
# >>> BUILD NOTE (for the orchestrator) <<<
# The orchestrator builds this CAPPED to:  output/bin/serenity_lower
# It pulls 0 GPU (workflow_graph imports only json) so it builds tiny like the
# stub. DO NOT BUILD IT from inside an agent session — building Mojo OOM-kills
# the desktop. The orchestrator runs the one capped Mojo build.
#
# Invoked:  serenity_lower <request.json>
#   argv[1] = path to a /v1/generate-shaped request body JSON. The file is read,
#             parsed with loads(), passed through apply_workflow_params(mut obj)
#             (which reads obj["workflow"] and writes the flat backend params in
#             place — the exact daemon lowering), then dumps(obj) is printed.
#
# Exit codes:
#   0  lowering succeeded; lowered JSON on stdout
#   1  usage / file-read / parse / lowering error (message on stderr-ish stdout)

from std.sys import argv
from std.memory import alloc

from json.parser import loads
from json.serialize import dumps
from json.value import JSONValue

from serenitymojo.serve.workflow_graph import apply_workflow_params
from serenitymojo.io.ffi import sys_open, sys_close, file_size, sys_pread, O_RDONLY


def _read_text_file(path: String) raises -> String:
    """Read an entire file into a String via the libc read path.

    Identical to model_scan._read_text_file / the daemon's file reader:
    sys_open(O_RDONLY) -> file_size -> pread the whole length -> sys_close.
    Reused here (rather than the stdlib `open`) for the same reason the rest of
    the codebase does: a single `external_call["open"]` declaration avoids the
    2-arg/3-arg lowering collision documented in io/ffi.mojo.
    """
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("serenity_lower: cannot open ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        return String("")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var k = sys_pread(fd, buf + done, n - done, done)
        if k <= 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("serenity_lower: read failed on ") + path)
        done += k
    _ = sys_close(fd)
    var s = String(StringSlice(ptr=buf, length=n))
    buf.free()
    return s^


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: serenity_lower <request.json>")
        return
    var path = String(args[1])
    var text = _read_text_file(path)
    var obj = loads(text)
    # Run the EXACT daemon lowering: reads obj["workflow"] (or a bare Comfy
    # graph) and writes the flat backend genparams onto obj in place.
    apply_workflow_params(obj)
    # Emit the lowered request object verbatim. This stdout IS the parity oracle.
    print(dumps(obj))
