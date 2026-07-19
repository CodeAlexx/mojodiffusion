# serenitymojo.serve — the SerenityUI generation daemon (skeleton stage).
#
# Pure-Mojo localhost HTTP+WebSocket daemon built on /home/alex/MOJO-libs
# (net/http/json/image/sqlite). The model backend is a pluggable seam
# (`backend.GenBackend`); stage 1 ships a stub (`stub_backend.StubBackend`)
# that simulates denoise steps and emits a real PNG. The real Z-Image
# binding implements the same trait later.
