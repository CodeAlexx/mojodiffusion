# probe_lifetime.mojo — REGRESSION ARTIFACT for F1 (use-after-munmap).
#
# This file MUST NOT COMPILE. It is the proof that F1 is fixed.
#
# Before the fix, SafeTensors.tensor_ptr() returned a bare BytePtr with
# MutExternalOrigin (lifetime UNTRACKED). A function could open a SafeTensors,
# grab a pointer, and RETURN that pointer while the SafeTensors (and its
# MmapRegion) was destroyed at function exit (Mojo ASAP destruction) — the
# region munmap'd, and the caller's deref hit unmapped memory -> SIGSEGV 139.
# That version COMPILED WITH NO DIAGNOSTIC.
#
# After the fix, the public accessor is tensor_bytes(), which returns
# `Span[UInt8, origin_of(self)]`. The Span carries self's origin, so the
# compiler keeps the SafeTensors alive for as long as the Span is used and
# FORBIDS returning the Span past the owner's lifetime. The misuse below — open
# locally, grab a Span, return it while `st` drops — is now a COMPILE ERROR
# (origin escape), so the use-after-munmap can no longer be written.
#
# Expected: `mojo build` of this file FAILS with an origin diagnostic on the
# escape:
#   error: cannot implicitly convert 'Span[UInt8, origin_of(st)]' value to
#          'Span[UInt8, MutAnyOrigin]'
# i.e. the Span's origin (`origin_of(st)`, the local handle) cannot be widened
# to an origin that outlives `st`. If it ever compiles again, F1 has regressed.
#
# Exact protection (verified 2026-05-25, Mojo 1.0.0b1 / MAX 26.3):
#   * ESCAPE (this file): returning a tensor_bytes() Span past the owner's scope
#     is a HARD COMPILE ERROR. The dangling-pointer-return that the old bare
#     tensor_ptr() allowed can no longer be written.
#   * IN-SCOPE: within a single scope, the origin keeps the SafeTensors alive
#     for as long as the Span is used — even an explicit `_ = st^` consume is
#     deferred past the last Span use, so an in-scope deref reads valid bytes
#     (no SIGSEGV) instead of munmap'd memory. (Tested: open st, grab span,
#     `_ = st^`, deref span -> prints the correct first byte, exit 0.)
# So the use-after-munmap is closed on both axes: the escape won't compile and
# the in-scope use is kept safe by origin-extended liveness.

from serenitymojo.io.safetensors import SafeTensors

comptime VAE_PATH = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae/"
    "diffusion_pytorch_model.safetensors"
)


def leak_span(out result: Span[UInt8, MutAnyOrigin]) raises:
    # Open, grab an origin-bound view, try to return it. `st` is destroyed at
    # function exit -> the Span would dangle. With origin tracking the compiler
    # MUST reject this: the Span borrows `st`, which does not outlive the call.
    var st = SafeTensors.open(String(VAE_PATH))
    var s = st.tensor_bytes(String("decoder.conv_in.bias"))
    result = s  # <-- MUST NOT COMPILE: escaping borrow of local `st`.
    # st.__del__ -> region.__del__ -> munmap here


def main() raises:
    var s = leak_span()
    print("if this compiled, F1 regressed:", Int(s[0]))
