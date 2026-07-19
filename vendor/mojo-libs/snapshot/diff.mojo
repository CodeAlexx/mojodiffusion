# snapshot.diff — compare two TSV manifests produced by `render_tsv`.
#
# Reports files added / removed / changed between an OLD and a NEW snapshot.
# "changed" = content hash differs (text files) or size differs (binary/other).

from snapshot.snapshot import FileEntry, human_size


def _atoi(s: String) -> Int:
    var bs = s.as_bytes()
    var i = 0
    var neg = False
    if len(bs) > 0 and Int(bs[0]) == ord("-"):
        neg = True
        i = 1
    var v = 0
    while i < len(bs):
        var c = Int(bs[i])
        if c < ord("0") or c > ord("9"):
            break
        v = v * 10 + (c - ord("0"))
        i += 1
    return -v if neg else v


def parse_manifest(path: String) raises -> Dict[String, FileEntry]:
    var m = Dict[String, FileEntry]()
    var f = open(path, "r")
    var data = f.read()
    f.close()
    var lines = data.split("\n")
    for i in range(len(lines)):
        var line = lines[i]
        if line.byte_length() == 0:
            continue
        if Int(line.as_bytes()[0]) == ord("#"):
            continue
        var c = line.split("\t")
        if len(c) < 5:
            continue
        var path_ = String(c[0])
        var is_text = c[4] == "T"
        m[path_] = FileEntry(
            path_, _atoi(String(c[1])), _atoi(String(c[2])), String(c[3]), is_text
        )
    return m^


def _sort_strs(mut xs: List[String]):
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            var tmp = xs[j]
            xs[j] = xs[j - 1]
            xs[j - 1] = tmp
            j -= 1


def diff(old_path: String, new_path: String) raises -> String:
    var oldm = parse_manifest(old_path)
    var newm = parse_manifest(new_path)

    var added = List[String]()
    var removed = List[String]()
    var changed = List[String]()

    for entry in newm.items():
        if entry.key not in oldm:
            added.append(entry.key)
        else:
            var o = oldm[entry.key]
            var n = entry.value
            var differ = (n.is_text and o.hash != n.hash) or (
                (not n.is_text) and o.size != n.size
            )
            if differ:
                changed.append(entry.key)
    for entry in oldm.items():
        if entry.key not in newm:
            removed.append(entry.key)

    _sort_strs(added)
    _sort_strs(removed)
    _sort_strs(changed)

    var out = String("")
    out += "# Snapshot diff\n"
    out += "old: " + old_path + "\nnew: " + new_path + "\n"
    out += (
        "+"
        + String(len(added))
        + " added · -"
        + String(len(removed))
        + " removed · ~"
        + String(len(changed))
        + " changed\n\n"
    )

    if len(added) > 0:
        out += "## added\n"
        for i in range(len(added)):
            var n = newm[added[i]]
            out += "+ " + added[i] + "  " + human_size(n.size)
            if n.is_text:
                out += "  " + String(n.lines) + "L"
            out += "\n"
        out += "\n"

    if len(removed) > 0:
        out += "## removed\n"
        for i in range(len(removed)):
            out += "- " + removed[i] + "\n"
        out += "\n"

    if len(changed) > 0:
        out += "## changed\n"
        for i in range(len(changed)):
            var k = changed[i]
            var o = oldm[k]
            var n = newm[k]
            out += "~ " + k
            if n.is_text:
                var dl = n.lines - o.lines
                var sign = String("+") if dl >= 0 else String("")
                out += "  " + String(o.lines) + "→" + String(
                    n.lines
                ) + "L (" + sign + String(dl) + ")"
            else:
                out += "  " + human_size(o.size) + "→" + human_size(n.size)
            out += "\n"
        out += "\n"

    return out
