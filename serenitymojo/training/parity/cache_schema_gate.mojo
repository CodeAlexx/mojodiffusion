# serenitymojo/training/parity/cache_schema_gate.mojo
#
# Cross-model cache-SCHEMA gate (audit item 8, 2026-07-07) — the anti-
# silent-zero-conditioning guard for the training caches.
#
# The training-cache readers were consolidated onto the shared klein_dataset
# primitives (load_cache_tensor / list_sorted_safetensors / cache_has_key /
# cache_tensor_dims). This gate is the paired check: for each <model>=<cache_dir>
# passed on argv, it opens the FIRST cached sample and FAILS LOUD with a named
# error if a required key is missing, a dtype is wrong, or a shape is off. It
# guards the bug class the recall ledger tracks:
#   ERI-0225  a missing text/context key silently ZERO-conditions the model
#             (garbage training, no crash) — here a missing REQUIRED key raises
#             cache_schema_gate[MISSING_KEY].
#   ERI-0220  a latent stored HWC instead of CHW (channels on the wrong axis) —
#             here the channel-axis constraint (ernie latent axis1=128, anima
#             axis1=16, l2p pixel axis0=3) raises cache_schema_gate[BAD_SHAPE].
#
# Generic by design: all model knowledge is in the small `expected_schema`
# table below (keys, required-ness, rank, per-axis dim constraints, dtype class).
# Add a model = add a branch there; the check loop is model-agnostic.
#
# Usage:
#   cache_schema_gate <model>=<cache_dir> [<model>=<cache_dir> ...]
# With NO args it prints the known-model table and exits 0 (lets CI build+run the
# gate without a live cache present). Any schema violation exits non-zero.
#
# Build (from /home/alex/mojodiffusion, -O2 to dodge the -O3 build OOM):
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm \
#       -Xlinker -lcuda \
#       -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib \
#       -Xlinker -L/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#       -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#       -Xlinker -rpath -Xlinker /home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#       -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/.pixi/envs/default/lib \
#       serenitymojo/training/parity/cache_schema_gate.mojo -o /tmp/cache_schema_gate

from std.sys import argv
from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.klein_dataset import (
    list_sorted_safetensors, cache_has_key, cache_tensor_dims,
)


# dtype class an expected key must satisfy.
comptime DT_ANY = 0     # no dtype constraint (e.g. an int token-count sidecar)
comptime DT_FLOAT = 1   # must be BF16 | F16 | F32 (real conditioning/latent data)


# ── one expected tensor in a model's cache sample ─────────────────────────────
# rank == -1 means "any rank". Up to two per-axis constraints (axisN, valN);
# axisN == -1 means that slot is unused. This is enough to pin the load-bearing
# geometry (latent channel count, text/context hidden dim) without over-fitting
# to a specific H/W/seq that legitimately varies per bucket.
@fieldwise_init
struct KeySpec(Copyable, Movable):
    var name: String
    var required: Bool
    var rank: Int
    var axis0: Int
    var val0: Int
    var axis1: Int
    var val1: Int
    var dtype_class: Int


def expected_schema(model: String) raises -> List[KeySpec]:
    """The per-model expected-schema table. Add a model here, not in the loop."""
    var out = List[KeySpec]()
    if model == "klein" or model == "zimage":
        # KleinCache contract: latent [B,128,h,w], text_embedding [B,512,D],
        # text_mask [B,512]. (klein_dataset.mojo LATENT/TEXT/MASK_KEY.)
        out.append(KeySpec(String("latent"), True, 4, -1, -1, -1, -1, DT_FLOAT))
        out.append(KeySpec(String("text_embedding"), True, 3, -1, -1, -1, -1, DT_FLOAT))
        out.append(KeySpec(String("text_mask"), True, 2, -1, -1, -1, -1, DT_FLOAT))
        return out^
    if model == "ernie":
        # train_ernie_real: latent [1,128,32,32], text_embedding [1,512,3072],
        # text_real_len [1] OPTIONAL (all-N_TXT fallback when absent).
        out.append(KeySpec(String("latent"), True, 4, 1, 128, -1, -1, DT_FLOAT))
        out.append(KeySpec(String("text_embedding"), True, 3, 2, 3072, -1, -1, DT_FLOAT))
        out.append(KeySpec(String("text_real_len"), False, 1, -1, -1, -1, -1, DT_ANY))
        return out^
    if model == "anima":
        # train_anima_real: per-sample cache is latent-only [1,16,H',W']; the
        # cross-attn context is a SEPARATE sidecar (CONTEXT_PATH), not per-sample.
        out.append(KeySpec(String("latent"), True, 4, 1, 16, -1, -1, DT_FLOAT))
        return out^
    if model == "l2p":
        # L2PCache contract: pixel [3,H,W] rank-3 (no batch axis), cap_feats
        # [1,seq,2560] (seq varies). NO text_mask. (klein_dataset PIXEL/CAP_FEATS.)
        out.append(KeySpec(String("pixel"), True, 3, 0, 3, -1, -1, DT_FLOAT))
        out.append(KeySpec(String("cap_feats"), True, 3, 2, 2560, -1, -1, DT_FLOAT))
        return out^
    raise Error(
        String("cache_schema_gate: unknown model '") + model
        + String("'; known: klein, zimage, ernie, anima, l2p")
    )


def _is_float(dt: STDtype) -> Bool:
    return dt == STDtype.F32 or dt == STDtype.F16 or dt == STDtype.BF16


def _check_axis(
    model: String, path: String, key: String,
    dims: List[Int], axis: Int, expected: Int,
) raises:
    if axis >= len(dims):
        raise Error(
            String("cache_schema_gate[BAD_SHAPE]: model=") + model
            + String(" key '") + key + String("' has rank ") + String(len(dims))
            + String(" but schema constrains axis ") + String(axis)
            + String(" in ") + path
        )
    if dims[axis] != expected:
        raise Error(
            String("cache_schema_gate[BAD_SHAPE]: model=") + model
            + String(" key '") + key + String("' axis ") + String(axis)
            + String(" expected ") + String(expected)
            + String(", got ") + String(dims[axis]) + String(" in ") + path
        )


def check_cache_dir(model: String, dir: String) raises:
    """Open the first sorted sample in `dir` and validate it against `model`'s
    expected schema. Raises a named cache_schema_gate[...] error on violation."""
    var schema = expected_schema(model)
    var files = list_sorted_safetensors(dir)
    if len(files) == 0:
        raise Error(
            String("cache_schema_gate[EMPTY_CACHE]: model=") + model
            + String(" dir=") + dir + String(" has no .safetensors files")
        )
    var path = files[0]
    var st = SafeTensors.open(path)
    print("  [", model, "] first sample:", path, " (", len(files), "files)")
    for i in range(len(schema)):
        var ks = schema[i].copy()
        if not cache_has_key(st, ks.name):
            if ks.required:
                raise Error(
                    String("cache_schema_gate[MISSING_KEY]: model=") + model
                    + String(" dir=") + dir
                    + String(" required key '") + ks.name
                    + String("' absent from first sample ") + path
                    + String(" — silent-zero-conditioning bug class (ERI-0225);")
                    + String(" re-run this model's prepare/precache")
                )
            print("    key '", ks.name, "': ABSENT (optional) — ok")
            continue
        var dims = cache_tensor_dims(st, ks.name)
        if ks.rank >= 0 and len(dims) != ks.rank:
            raise Error(
                String("cache_schema_gate[BAD_RANK]: model=") + model
                + String(" key '") + ks.name + String("' expected rank ")
                + String(ks.rank) + String(", got rank ") + String(len(dims))
                + String(" in ") + path
            )
        if ks.axis0 >= 0:
            _check_axis(model, path, ks.name, dims, ks.axis0, ks.val0)
        if ks.axis1 >= 0:
            _check_axis(model, path, ks.name, dims, ks.axis1, ks.val1)
        var info = st.tensor_info(ks.name)
        if ks.dtype_class == DT_FLOAT and not _is_float(info.dtype):
            raise Error(
                String("cache_schema_gate[BAD_DTYPE]: model=") + model
                + String(" key '") + ks.name
                + String("' expected float (BF16/F16/F32), got ")
                + info.dtype.name() + String(" in ") + path
            )
        print("    key '", ks.name, "': rank", len(dims),
              " dtype", info.dtype.name(), " ok")
    print("  [", model, "] PASS")


def _find_byte(s: String, target: UInt8) -> Int:
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        if bs[i] == target:
            return i
    return -1


def _substr(s: String, start: Int, end: Int) -> String:
    var bs = s.as_bytes()
    var out = String("")
    for i in range(start, end):
        out += chr(Int(bs[i]))
    return out^


def _print_usage():
    print("cache_schema_gate — cross-model cache-schema anti-silent-zero-conditioning gate")
    print("usage: cache_schema_gate <model>=<cache_dir> [<model>=<cache_dir> ...]")
    print("known models + required keys (first sample is opened + checked):")
    print("  klein / zimage : latent[rank4], text_embedding[rank3], text_mask[rank2]")
    print("  ernie          : latent[rank4,ch=128], text_embedding[rank3,D=3072], text_real_len[rank1] optional")
    print("  anima          : latent[rank4,ch=16]   (cross-attn context is a separate sidecar)")
    print("  l2p            : pixel[rank3,ch=3], cap_feats[rank3,D=2560]")


def main() raises:
    var args = argv()
    if len(args) < 2:
        _print_usage()
        print("no cache dirs given; nothing to check. exit 0.")
        return
    print("==== cache_schema_gate ====")
    var n_ok = 0
    for i in range(1, len(args)):
        var tok = String(args[i])
        var eq = _find_byte(tok, 0x3D)   # '='
        if eq < 0:
            _print_usage()
            raise Error(
                String("cache_schema_gate: argument '") + tok
                + String("' must be model=/path/to/cache")
            )
        var model = _substr(tok, 0, eq)
        var dir = _substr(tok, eq + 1, tok.byte_length())
        check_cache_dir(model, dir)
        n_ok += 1
    print("cache_schema_gate: ALL", n_ok, "cache dir(s) PASS")
