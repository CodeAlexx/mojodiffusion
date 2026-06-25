# serenitymojo.captioner — pure-Mojo host-side caption logic (no GPU, no LLM).
#
# Modules:
#   ideogram_caption — faithful 1:1 port of ai-toolkit's
#                       toolkit/ideogram_caption.py (module A): normalize /
#                       migrate / minify already-structured Ideogram-4 captions
#                       into the compact model-string, pass prose through
#                       byte-for-byte. Pure string/JSON logic.
#   ideogram_bc_glue — faithful 1:1 port of the DETERMINISTIC glue around module A
#                       for the two Ideogram-4 captioner pipelines:
#                       B = Ideogram4Captioner (image→JSON), C = upsample
#                       (idea→JSON). build_prompt (template substitution),
#                       extract_json (hand-rolled lazy-DOTALL fence scanner + first
#                       `{`…last `}` + strict parse), per-element bbox fix (B SWAPS
#                       x/y, C does NOT), module-A normalize tail, and TWO new
#                       json.dumps serializer modes (default ", "/": " and pretty
#                       indent=2) matching CPython byte-for-byte. The LLM call is a
#                       pluggable CaptionGenerator backend (smoke-only, not gated).
#                       The two system prompts live in prompts/ as byte-exact data
#                       files (hash-asserted against the spec anchors).
