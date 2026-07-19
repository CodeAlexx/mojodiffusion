from serenitymojo.io.json_header import parse_header

def run(label: String, json: String):
    print("=== ", label, " ===")
    var bytes = List[UInt8]()
    for b in json.as_bytes():
        bytes.append(b)
    try:
        var entries = parse_header(bytes^)
        print("  OK entries=", len(entries))
        for ref e in entries:
            print("    name=", e.name, " off=[", e.off_start, ",", e.off_end, "]")
    except err:
        print("  RAISED:", String(err))

def main():
    # 1. __metadata__ value contains braces/brackets INSIDE strings.
    run("metadata-brace-in-string",
        String('{"__metadata__":{"note":"contains } and ] and { chars"},'
               '"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'))
    # 2. tensor name with escaped chars.
    run("escaped-name",
        String('{"a\\tb\\n":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'))
    # 3. surrogate pair (non-BMP emoji U+1F600 = 😀). BMP-only decoder.
    run("surrogate-pair",
        String('{"x\\ud83d\\ude00":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'))
    # 4. __metadata__ value is a STRING (not object) - some writers do this? 
    run("metadata-string-value",
        String('{"__metadata__":"just a string",'
               '"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'))
    # 5. duplicate dtype key (last wins in serde; Mojo overwrites too).
    run("duplicate-dtype",
        String('{"t":{"dtype":"F32","dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'))
    # 6. unknown extra field in tensor object (tolerated?)
    run("extra-field",
        String('{"t":{"dtype":"F32","extra":{"nested":[1,2]},"shape":[1],"data_offsets":[0,4]}}'))
