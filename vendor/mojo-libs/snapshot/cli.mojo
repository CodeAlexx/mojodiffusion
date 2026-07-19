# snapshot.cli — runnable entrypoint.
#
#   mojo run -I . snapshot/cli.mojo <dir> [-o PREFIX] [--brief]
#   mojo run -I . snapshot/cli.mojo --diff OLD.tsv NEW.tsv
#
# With no -o, the markdown map is printed to stdout (lands straight in a tool
# result). With -o PREFIX it writes PREFIX.md (human map) + PREFIX.tsv
# (machine manifest for `--diff`).

from std.sys import argv

from snapshot.snapshot import scan, default_config, render_markdown, render_tsv, write_file
from snapshot.diff import diff


def _usage():
    print("usage:")
    print("  snapshot <dir> [-o PREFIX] [--brief]")
    print("  snapshot --diff OLD.tsv NEW.tsv")


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(len(raw)):
        args.append(String(raw[i]))

    if len(args) < 2:
        _usage()
        return

    if args[1] == "--diff":
        if len(args) < 4:
            _usage()
            return
        print(diff(args[2], args[3]))
        return

    if args[1] == "-h" or args[1] == "--help":
        _usage()
        return

    var root = args[1]
    var cfg = default_config()
    var out_prefix = String("")

    var i = 2
    while i < len(args):
        var a = args[i]
        if a == "-o" or a == "--out":
            if i + 1 < len(args):
                i += 1
                out_prefix = args[i]
        elif a == "--brief":
            cfg.brief = True
        i += 1

    var snap = scan(root, cfg)
    if out_prefix.byte_length() == 0:
        print(render_markdown(snap, cfg))
    else:
        write_file(String(out_prefix + ".md"), render_markdown(snap, cfg))
        write_file(String(out_prefix + ".tsv"), render_tsv(snap))
        print("wrote " + out_prefix + ".md + " + out_prefix + ".tsv  (" + String(len(snap.files)) + " files)")
