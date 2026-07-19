# probe_json.mojo — adversarial edge cases for the hand-rolled header parser.
from serenitymojo.io.json_header import parse_header, HeaderEntry


def run(label: String, json: String):
    print("=== ", label, " ===")
    var bytes = List[UInt8]()
    for b in json.as_bytes():
        bytes.append(b)
    try:
        var entries = parse_header(bytes^)
        print("  OK entries=", len(entries))
        for ref e in entries:
            var sh = String("[")
            for i in range(len(e.shape)):
                if i > 0:
                    sh += ","
                sh += String(e.shape[i])
            sh += "]"
            print(
                "    name=", e.name, " dtype=", e.dtype, " shape=", sh,
                " off=[", e.off_start, ",", e.off_end, "]",
            )
    except err:
        print("  RAISED:", String(err))


def main():
    # 1. Missing dtype -> Rust defaults "F32".
    run(
        "missing-dtype",
        String('{"t":{"shape":[2,2],"data_offsets":[0,8]}}'),
    )
    # 2. Empty shape (scalar tensor).
    run(
        "scalar-empty-shape",
        String('{"t":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}'),
    )
    # 3. Nested __metadata__ object with nested object + array + escaped quote.
    run(
        "nested-metadata",
        String(
            '{"__metadata__":{"a":{"b":[1,2,{"c":"x\\"y"}]},"f":"pt"},'
            '"t":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}'
        ),
    )
    # 4. Large offset > 2^32 (10 GB) — must NOT truncate.
    run(
        "huge-offset",
        String(
            '{"t":{"dtype":"BF16","shape":[1],"data_offsets":[9973681280,9973681282]}}'
        ),
    )
    # 5. Key order: data_offsets before dtype before shape.
    run(
        "key-order",
        String(
            '{"t":{"data_offsets":[0,4],"dtype":"F32","shape":[1]}}'
        ),
    )
    # 6. Negative data_offset — Rust serde as_u64 returns None -> filtered ->
    #    unwrap_or default. Mojo _parse_int only reads digits.
    run(
        "negative-offset",
        String('{"t":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]}}'),
    )
    # 7. Float data_offset — Rust as_u64 None. Mojo _parse_int stops at '.'.
    run(
        "float-offset",
        String('{"t":{"dtype":"F32","shape":[1],"data_offsets":[0.0,4.0]}}'),
    )
    # 8. __metadata__ AFTER a tensor (order independence).
    run(
        "metadata-last",
        String(
            '{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
            '"__metadata__":{"k":"v"}}'
        ),
    )
    # 9. Empty object.
    run("empty-object", String("{}"))
    # 10. Whitespace everywhere.
    run(
        "whitespace",
        String(
            '{  "t" : { "dtype" : "F32" , "shape" : [ 1 , 2 ] ,'
            ' "data_offsets" : [ 0 , 8 ] } }'
        ),
    )
